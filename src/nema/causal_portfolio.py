"""Causal proposal controls followed by the frozen AEMA hard portfolio.

This module intentionally sits outside :mod:`nema.solvers`: the active native
experiment attests that file while it is running.  The routines here expose a
small common interface for later preregistered controls whose only difference
is the four score matrices supplied to the same discrete solver.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import torch

from nema.association import AssociationGraph
from nema.graph import GraphPair
from nema.metrics import johnson_similarity
from nema.models.nema import NEMAModel
from nema.rounding import perturb_and_refine
from nema.solvers import SolveResult, source_to_target_mapping


def fixed_unit_score(
    association: AssociationGraph,
    *,
    device: str,
    sinkhorn_max_iterations: int,
) -> tuple[torch.Tensor, dict]:
    """Return the deterministic fixed/fixed/:math:`M=1` mirror trajectory."""

    model = NEMAModel().to(device).eval()
    with torch.no_grad():
        output = model(
            association,
            initializer_mode="fixed",
            schedule_mode="fixed",
            metric_mode="unit",
            line_search=True,
            stationary_safeguard=True,
            sinkhorn_tolerance=1e-5,
            sinkhorn_max_iterations=sinkhorn_max_iterations,
            acceptance_tolerance=0.0,
        )
    return output.assignment.detach().cpu(), {
        "accepted_steps": output.accepted_steps,
        "backtracking_trials": output.backtracking_trials,
        "proposal_sources": output.proposal_sources,
        "max_row_residual": output.max_row_residual,
        "max_column_excess": output.max_column_excess,
        "soft_objective": float(output.objectives[-1]),
    }


def random_positive_scores(
    shape: tuple[int, int],
    *,
    seed: int,
    streams: int = 4,
) -> list[torch.Tensor]:
    """Generate preregistered positive random score matrices on the CPU."""

    generator = torch.Generator(device="cpu").manual_seed(seed)
    return [
        torch.rand(shape, generator=generator, dtype=torch.float32).clamp_min_(1e-6)
        for _ in range(streams)
    ]


def solve_score_portfolio(
    pair: GraphPair,
    score_streams: Sequence[torch.Tensor],
    *,
    stream_labels: Sequence[str],
    base_search_seed: int,
    started: float | None = None,
    discrete_restarts: int = 64,
    refinement_passes: int = 30,
    anneal_steps: int = 2500,
    lns_steps: int = 250,
    method: str,
    proposal_metadata: Sequence[dict] | None = None,
) -> SolveResult:
    """Apply exactly the frozen per-stream hard-search portfolio.

    The score source is the only experimental variable.  Stream 0--2 use the
    same search-seed offsets as AEMA's three primary trajectories; stream 3
    uses the same seed as AEMA's deterministic fourth/reference trajectory.
    """

    if len(score_streams) != 4 or len(stream_labels) != 4:
        raise ValueError("the decisive causal suite requires exactly four score streams")
    if proposal_metadata is not None and len(proposal_metadata) != 4:
        raise ValueError("proposal metadata must align with the four streams")

    started = time.perf_counter() if started is None else started
    left, right, swapped = pair.oriented()
    association = AssociationGraph.build(left, right)
    expected_shape = association.shape
    hard_seeds = [
        base_search_seed,
        base_search_seed + 1009,
        base_search_seed + 2 * 1009,
        base_search_seed,
    ]
    best_mapping: torch.Tensor | None = None
    best_stats = (-1, -1)
    selected_stream = -1
    stream_records: list[dict] = []

    for index, (raw_scores, label, hard_seed) in enumerate(
        zip(score_streams, stream_labels, hard_seeds, strict=True)
    ):
        scores = raw_scores.detach().cpu().float()
        if tuple(scores.shape) != expected_shape:
            raise ValueError(
                f"stream {index} ({label}) has shape {tuple(scores.shape)}, "
                f"expected {expected_shape}"
            )
        if not bool(torch.isfinite(scores).all()):
            raise ValueError(f"stream {index} ({label}) contains non-finite scores")
        scores = scores.clamp_min(1e-12)
        stream_started = time.perf_counter()
        mapping = perturb_and_refine(
            association,
            scores,
            restarts=discrete_restarts,
            max_passes=refinement_passes,
            anneal_steps=anneal_steps,
            lns_steps=lns_steps,
            seed=hard_seed,
        )
        stats = association.hard_statistics(mapping)
        stream_records.append(
            {
                "index": index,
                "label": label,
                "hard_search_seed": hard_seed,
                "common_edges": stats[0],
                "common_nodes": stats[1],
                "hard_search_runtime_seconds": time.perf_counter() - stream_started,
                **((proposal_metadata or ({},) * 4)[index]),
            }
        )
        if stats > best_stats:
            best_mapping = mapping
            best_stats = stats
            selected_stream = index

    assert best_mapping is not None
    edges, nodes = best_stats
    left_size = pair.left.num_nodes + pair.left.num_edges
    right_size = pair.right.num_nodes + pair.right.num_edges
    similarity = johnson_similarity(nodes, edges, left_size, right_size)
    accuracy = edges / pair.true_edges if pair.true_edges else None
    squared_error = (
        (similarity - pair.true_similarity) ** 2
        if pair.true_similarity is not None
        else None
    )
    serialized = source_to_target_mapping(pair, best_mapping, swapped)
    return SolveResult(
        method=method,
        key=pair.key,
        common_edges=edges,
        common_nodes=nodes,
        similarity=similarity,
        runtime_seconds=time.perf_counter() - started,
        mapping=serialized.tolist(),
        true_edges=pair.true_edges,
        true_nodes=pair.true_nodes,
        true_similarity=pair.true_similarity,
        accuracy=accuracy,
        similarity_squared_error=squared_error,
        soft_objective=None,
        metadata={
            "algorithm_version": "aema-matched-hard-portfolio-v1",
            "swapped": swapped,
            "mapping_encoding": "source_to_target; -1 denotes an unmatched source node",
            "mapping_normalized_from_oriented": swapped,
            "stream_count": 4,
            "stream_labels": list(stream_labels),
            "selected_stream_index": selected_stream,
            "selected_stream_label": stream_labels[selected_stream],
            "stream_records": stream_records,
            "discrete_restarts": discrete_restarts,
            "constructive_restarts": discrete_restarts,
            "refinement_passes": refinement_passes,
            "anneal_steps": anneal_steps,
            "lns_steps": lns_steps,
            "input_provenance": pair.metadata,
        },
    )
