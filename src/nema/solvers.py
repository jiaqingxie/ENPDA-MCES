"""Complete NGA and NEMA solvers with a common evaluation interface."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import torch

from nema.association import AssociationGraph
from nema.certificate import solve_lifted_milp
from nema.graph import GraphPair
from nema.metrics import johnson_similarity
from nema.models.nema import NEMAModel
from nema.models.nga import PerInstanceNGA
from nema.rounding import hungarian_mapping, perturb_and_refine, refine_mapping
from nema.sinkhorn import sample_gumbel_like


@dataclass
class SolveResult:
    method: str
    key: str
    common_edges: int
    common_nodes: int
    similarity: float
    runtime_seconds: float
    mapping: list[int]
    true_edges: int | None = None
    true_nodes: int | None = None
    true_similarity: float | None = None
    accuracy: float | None = None
    similarity_squared_error: float | None = None
    soft_objective: float | None = None
    metadata: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def source_to_target_mapping(
    pair: GraphPair,
    oriented_mapping: torch.Tensor,
    swapped: bool,
) -> torch.Tensor:
    """Express an oriented injective mapping in the original pair direction."""

    oriented_mapping = oriented_mapping.detach().cpu().long()
    if not swapped:
        if oriented_mapping.numel() != pair.left.num_nodes:
            raise ValueError("mapping length does not match the original source graph")
        return oriented_mapping
    if oriented_mapping.numel() != pair.right.num_nodes:
        raise ValueError("swapped mapping length does not match the original target graph")
    mapping = torch.full((pair.left.num_nodes,), -1, dtype=torch.long)
    for target_index, source_index in enumerate(oriented_mapping.tolist()):
        if source_index < 0:
            continue
        if source_index >= pair.left.num_nodes or mapping[source_index] >= 0:
            raise ValueError("swapped mapping is out of range or non-injective")
        mapping[source_index] = target_index
    return mapping


def _finalize(
    method: str,
    pair: GraphPair,
    association: AssociationGraph,
    mapping: torch.Tensor,
    started: float,
    soft_objective: float | None = None,
    metadata: dict | None = None,
) -> SolveResult:
    edges, nodes = association.hard_statistics(mapping)
    left_size = pair.left.num_nodes + pair.left.num_edges
    right_size = pair.right.num_nodes + pair.right.num_edges
    similarity = johnson_similarity(nodes, edges, left_size, right_size)
    accuracy = edges / pair.true_edges if pair.true_edges else None
    squared_error = (
        (similarity - pair.true_similarity) ** 2 if pair.true_similarity is not None else None
    )
    metadata = dict(metadata or {})
    swapped = bool(metadata.get("swapped", False))
    serialized_mapping = source_to_target_mapping(pair, mapping, swapped)
    metadata["mapping_encoding"] = "source_to_target; -1 denotes an unmatched source node"
    metadata["mapping_normalized_from_oriented"] = swapped
    if pair.metadata is not None:
        metadata["input_provenance"] = pair.metadata
    return SolveResult(
        method=method,
        key=pair.key,
        common_edges=edges,
        common_nodes=nodes,
        similarity=similarity,
        runtime_seconds=time.perf_counter() - started,
        mapping=serialized_mapping.tolist(),
        true_edges=pair.true_edges,
        true_nodes=pair.true_nodes,
        true_similarity=pair.true_similarity,
        accuracy=accuracy,
        similarity_squared_error=squared_error,
        soft_objective=soft_objective,
        metadata=metadata,
    )


class NEMASolver:
    def __init__(
        self,
        model: NEMAModel | None = None,
        continuous_restarts: int = 4,
        discrete_restarts: int = 8,
        refinement_passes: int = 20,
        anneal_steps: int = 1000,
        lns_steps: int = 100,
        certificate_seconds: float = 0.0,
        noise_scale: float = 0.5,
        seed: int = 0,
        device: str = "cpu",
        trajectory: str = "learned",
        initializer_mode: str = "learned",
        schedule_mode: str = "learned",
        unit_fallback: bool = True,
        line_search: bool = True,
        stationary_safeguard: bool = True,
        sinkhorn_tolerance: float | None = 1e-5,
        sinkhorn_max_iterations: int | None = 250,
        acceptance_tolerance: float = 0.0,
    ) -> None:
        if trajectory not in {"learned", "unit"}:
            raise ValueError(f"unknown NEMA trajectory {trajectory!r}")
        if initializer_mode not in {"learned", "fixed"}:
            raise ValueError(f"unknown initializer mode {initializer_mode!r}")
        if schedule_mode not in {"learned", "fixed"}:
            raise ValueError(f"unknown schedule mode {schedule_mode!r}")
        self.model = model or NEMAModel()
        self.model.to(device)
        self.model.eval()
        self.continuous_restarts = continuous_restarts
        self.discrete_restarts = discrete_restarts
        self.refinement_passes = refinement_passes
        self.anneal_steps = anneal_steps
        self.lns_steps = lns_steps
        self.certificate_seconds = certificate_seconds
        self.noise_scale = noise_scale
        self.seed = seed
        self.device = device
        self.trajectory = trajectory
        self.initializer_mode = initializer_mode
        self.schedule_mode = schedule_mode
        self.unit_fallback = unit_fallback
        self.line_search = line_search
        self.stationary_safeguard = stationary_safeguard
        self.sinkhorn_tolerance = sinkhorn_tolerance
        self.sinkhorn_max_iterations = sinkhorn_max_iterations
        self.acceptance_tolerance = acceptance_tolerance

    def solve(self, pair: GraphPair) -> SolveResult:
        started = time.perf_counter()
        left, right, swapped = pair.oriented()
        association = AssociationGraph.build(left, right)
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        best_mapping: torch.Tensor | None = None
        best_stats = (-1, -1)
        best_soft = None
        best_output = None
        best_source = f"{self.trajectory}_metric"

        with torch.no_grad():
            for restart in range(self.continuous_restarts):
                if restart == 0:
                    noise = None
                else:
                    template = torch.empty(association.shape, device=self.device)
                    noise = self.noise_scale * sample_gumbel_like(template, generator)
                output = self.model(
                    association,
                    initial_noise=noise,
                    initializer_mode=self.initializer_mode,
                    schedule_mode=self.schedule_mode,
                    line_search=self.line_search,
                    metric_mode=self.trajectory,
                    stationary_safeguard=self.stationary_safeguard,
                    sinkhorn_tolerance=self.sinkhorn_tolerance,
                    sinkhorn_max_iterations=self.sinkhorn_max_iterations,
                    acceptance_tolerance=self.acceptance_tolerance,
                )
                score = output.assignment.detach().cpu()
                mapping = perturb_and_refine(
                    association,
                    score,
                    restarts=self.discrete_restarts,
                    max_passes=self.refinement_passes,
                    anneal_steps=self.anneal_steps,
                    lns_steps=self.lns_steps,
                    seed=self.seed + 1009 * restart,
                )
                stats = association.hard_statistics(mapping)
                if stats > best_stats:
                    best_mapping = mapping
                    best_stats = stats
                    best_soft = float(output.objectives[-1])
                    best_output = output

            # A high relaxed QAP objective need not round to a high-quality
            # integer assignment.  Include ordinary mirror ascent (M=1) as a
            # label-free control trajectory and select between trajectories by
            # the directly observable hard MCES objective.  This is an
            # anytime-safe fallback, not test-label checkpoint selection.
            if self.unit_fallback:
                fallback_model = NEMAModel(
                    steps=self.model.steps,
                    hidden_dim=self.model.metric.local[0].out_features,
                    sinkhorn_iterations=self.model.sinkhorn_iterations,
                    max_backtracks=self.model.max_backtracks,
                    armijo=self.model.armijo,
                    safeguard_alignment=self.model.safeguard_alignment,
                ).to(self.device)
                fallback_model.eval()
                fallback_output = fallback_model(
                    association,
                    initializer_mode="fixed",
                    schedule_mode="fixed",
                    line_search=self.line_search,
                    metric_mode="unit",
                    stationary_safeguard=self.stationary_safeguard,
                    sinkhorn_tolerance=self.sinkhorn_tolerance,
                    sinkhorn_max_iterations=self.sinkhorn_max_iterations,
                    acceptance_tolerance=self.acceptance_tolerance,
                )
                fallback_mapping = perturb_and_refine(
                    association,
                    fallback_output.assignment.detach().cpu(),
                    restarts=self.discrete_restarts,
                    max_passes=self.refinement_passes,
                    anneal_steps=self.anneal_steps,
                    lns_steps=self.lns_steps,
                    seed=self.seed,
                )
                fallback_stats = association.hard_statistics(fallback_mapping)
                if fallback_stats > best_stats:
                    best_mapping = fallback_mapping
                    best_stats = fallback_stats
                    best_soft = float(fallback_output.objectives[-1])
                    best_output = fallback_output
                    best_source = "unit_metric_fallback"

        assert best_mapping is not None and best_output is not None
        certificate_metadata = None
        if self.certificate_seconds > 0:
            certificate = solve_lifted_milp(
                association, time_limit=self.certificate_seconds, relative_gap=0.0
            )
            if certificate.mapping is not None:
                certified_stats = association.hard_statistics(certificate.mapping)
                if certified_stats > best_stats:
                    best_mapping = certificate.mapping
                    best_stats = certified_stats
            combined_lower = best_stats[0]
            if (
                certificate.upper_bound is not None
                and certificate.upper_bound + 1e-6 < combined_lower
            ):
                raise RuntimeError("certificate upper bound fell below a legal NEMA lower bound")
            combined_gap = (
                max(0.0, certificate.upper_bound - combined_lower)
                / max(abs(certificate.upper_bound), 1.0)
                if certificate.upper_bound is not None
                else None
            )
            certificate_metadata = {
                "lower_bound": combined_lower,
                "upper_bound": certificate.upper_bound,
                "raw_upper_bound": certificate.raw_upper_bound,
                "upper_inflation": certificate.upper_inflation,
                "relative_gap": combined_gap,
                "certified_optimal": combined_gap is not None and combined_gap <= 1e-7,
                "status": certificate.status,
                "message": certificate.message,
                "mode": certificate.mode,
                "runtime_seconds": certificate.runtime_seconds,
                "variables": certificate.variables,
                "constraints": certificate.constraints,
            }
        return _finalize(
            "NEMA+Cert" if self.certificate_seconds > 0 else "NEMA",
            pair,
            association,
            best_mapping,
            started,
            soft_objective=best_soft,
            metadata={
                "algorithm_version": "nema-safeguarded-lifted-kl-v1",
                "swapped": swapped,
                "accepted_steps": best_output.accepted_steps,
                "backtracking_trials": best_output.backtracking_trials,
                "proposal_sources": best_output.proposal_sources,
                "unit_residuals": best_output.unit_residuals,
                "max_row_residual": best_output.max_row_residual,
                "max_column_excess": best_output.max_column_excess,
                "metric_min": best_output.metric_min,
                "metric_max": best_output.metric_max,
                "continuous_restarts": self.continuous_restarts,
                "primary_trajectory": f"{self.trajectory}_metric",
                "initializer_mode": self.initializer_mode,
                "schedule_mode": self.schedule_mode,
                "unit_metric_fallback": self.unit_fallback,
                "line_search": self.line_search,
                "stationary_safeguard": self.stationary_safeguard,
                "safeguard_alignment": self.model.safeguard_alignment,
                "sinkhorn_tolerance": self.sinkhorn_tolerance,
                "sinkhorn_max_iterations": self.sinkhorn_max_iterations,
                "acceptance_tolerance": self.acceptance_tolerance,
                "trajectory_portfolio": [
                    f"{self.trajectory}_metric",
                    *(["unit_metric_fallback"] if self.unit_fallback else []),
                ],
                "selected_trajectory": best_source,
                "discrete_restarts": self.discrete_restarts,
                "constructive_restarts": self.discrete_restarts,
                "refinement_passes": self.refinement_passes,
                "anneal_steps": self.anneal_steps,
                "lns_steps": self.lns_steps,
                "certificate_seconds": self.certificate_seconds,
                "certificate": certificate_metadata,
            },
        )


class NGASolver:
    """Train a fresh NGA optimizer for one graph pair, as in the paper."""

    def __init__(
        self,
        epochs: int = 200,
        learning_rate: float = 1e-3,
        hidden_dim: int = 32,
        encoder_layers: int = 8,
        steps: int = 4,
        samples: int = 10,
        sinkhorn_iterations: int = 20,
        evaluation_interval: int = 5,
        refine: bool = False,
        refinement_passes: int = 20,
        time_budget: float = 60.0,
        variant: str = "acg",
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.encoder_layers = encoder_layers
        self.steps = steps
        self.samples = samples
        self.sinkhorn_iterations = sinkhorn_iterations
        self.evaluation_interval = evaluation_interval
        self.refine = refine
        self.refinement_passes = refinement_passes
        self.time_budget = time_budget
        self.variant = variant
        self.seed = seed
        self.device = device

    def solve(self, pair: GraphPair) -> SolveResult:
        started = time.perf_counter()
        torch.manual_seed(self.seed)
        left, right, swapped = pair.oriented()
        association = AssociationGraph.build(left, right)
        model = PerInstanceNGA(
            association,
            hidden_dim=self.hidden_dim,
            encoder_layers=self.encoder_layers,
            steps=self.steps,
            samples=self.samples,
            sinkhorn_iterations=self.sinkhorn_iterations,
            variant=self.variant,
        ).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        best_mapping: torch.Tensor | None = None
        best_stats = (-1, -1)
        best_soft = None
        completed_epochs = 0

        for epoch in range(1, self.epochs + 1):
            optimizer.zero_grad(set_to_none=True)
            assignment, objectives = model(add_gumbel=True)
            # The released NGA code optimizes the raw QAP objective. Normalizing
            # here materially slows its per-instance learning under the paper LR.
            loss = -objectives.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            completed_epochs = epoch

            should_evaluate = epoch >= self.epochs // 2 and (
                epoch % self.evaluation_interval == 0 or epoch == self.epochs
            )
            if should_evaluate:
                with torch.no_grad():
                    for sample, objective in zip(assignment, objectives, strict=True):
                        mapping = hungarian_mapping(sample)
                        if self.refine:
                            mapping = refine_mapping(
                                association, mapping, max_passes=self.refinement_passes
                            )
                        stats = association.hard_statistics(mapping)
                        if stats > best_stats:
                            best_mapping = mapping
                            best_stats = stats
                            best_soft = float(objective)
            if time.perf_counter() - started >= self.time_budget:
                break

        if best_mapping is None:
            with torch.no_grad():
                assignment, objectives = model(add_gumbel=True)
                for sample, objective in zip(assignment, objectives, strict=True):
                    mapping = hungarian_mapping(sample)
                    stats = association.hard_statistics(mapping)
                    if stats > best_stats:
                        best_mapping, best_stats, best_soft = mapping, stats, float(objective)
        assert best_mapping is not None
        return _finalize(
            "NGA",
            pair,
            association,
            best_mapping,
            started,
            soft_objective=best_soft,
            metadata={
                "swapped": swapped,
                "epochs": completed_epochs,
                "temperatures": model.temperatures.detach().cpu().tolist(),
                "samples": self.samples,
                "refined": self.refine,
                "variant": self.variant,
            },
        )
