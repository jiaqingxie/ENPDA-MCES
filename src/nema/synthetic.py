"""Deterministic synthetic graph-pair families with an analytic MCES optimum."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from nema.graph import GraphPair, LabeledGraph


@dataclass(frozen=True)
class ExactSyntheticPair:
    pair: GraphPair
    planted_mapping: torch.Tensor
    family: str
    source_size: int
    deletion_rate: float
    instance_seed: int


def _random_edges(size: int, density: float, generator: np.random.Generator) -> set[tuple[int, int]]:
    edges = {(index, index + 1) for index in range(size - 1)}
    for left in range(size):
        for right in range(left + 1, size):
            if generator.random() < density:
                edges.add((left, right))
    return edges


def _motif_edges(size: int, motif_size: int) -> set[tuple[int, int]]:
    if size % motif_size:
        raise ValueError("repeated-motif sizes must be divisible by motif_size")
    edges: set[tuple[int, int]] = set()
    modules = size // motif_size
    for module in range(modules):
        offset = module * motif_size
        for index in range(motif_size):
            left = offset + index
            right = offset + ((index + 1) % motif_size)
            edges.add((min(left, right), max(left, right)))
        edges.add((offset, offset + 2))
    if modules > 1:
        for module in range(modules):
            left = module * motif_size
            right = ((module + 1) % modules) * motif_size
            edges.add((min(left, right), max(left, right)))
    return edges


def _graph(size: int, edges: set[tuple[int, int]]) -> LabeledGraph:
    ordered = sorted(edges)
    edge_index = (
        torch.tensor(ordered, dtype=torch.long).t().contiguous()
        if ordered
        else torch.empty((2, 0), dtype=torch.long)
    )
    return LabeledGraph(
        node_labels=torch.zeros(size, dtype=torch.long),
        edge_index=edge_index,
        edge_labels=torch.zeros(len(ordered), dtype=torch.long),
    )


def make_exact_pair(
    source_size: int,
    family: str,
    deletion_rate: float,
    distractor_fraction: float,
    random_edge_density: float,
    motif_size: int,
    seed: int,
) -> ExactSyntheticPair:
    """Create a pair whose exact optimal preserved-edge count is known.

    The target retains a nonempty subset of source edges under a random
    injection and adds only isolated distractor nodes.  The injection attains
    every target edge, while the target edge count is a universal upper bound.
    """

    if not 0 <= deletion_rate < 1:
        raise ValueError("deletion_rate must lie in [0, 1)")
    generator = np.random.default_rng(seed)
    if family == "asymmetric_random":
        source_edges = _random_edges(source_size, random_edge_density, generator)
    elif family == "repeated_motif":
        source_edges = _motif_edges(source_size, motif_size)
    else:
        raise ValueError(f"unknown synthetic family {family!r}")
    ordered_source = sorted(source_edges)
    keep_count = max(1, int(round((1.0 - deletion_rate) * len(ordered_source))))
    retained_indices = set(
        int(index)
        for index in generator.choice(len(ordered_source), size=keep_count, replace=False)
    )
    retained = [edge for index, edge in enumerate(ordered_source) if index in retained_indices]

    distractors = max(1, int(round(source_size * distractor_fraction)))
    target_size = source_size + distractors
    target_positions = generator.permutation(target_size)
    planted = torch.tensor(target_positions[:source_size], dtype=torch.long)
    target_edges = {
        (
            min(int(planted[left]), int(planted[right])),
            max(int(planted[left]), int(planted[right])),
        )
        for left, right in retained
    }
    source = _graph(source_size, source_edges)
    target = _graph(target_size, target_edges)
    incident_nodes = len({node for edge in target_edges for node in edge})
    pair = GraphPair(
        left=source,
        right=target,
        true_edges=target.num_edges,
        true_nodes=incident_nodes,
        key=f"{family}-n{source_size}-d{deletion_rate:.2f}-s{seed}",
        metadata={
            "synthetic_exact_optimum": True,
            "family": family,
            "source_size": source_size,
            "deletion_rate": deletion_rate,
            "instance_seed": seed,
        },
    )
    return ExactSyntheticPair(
        pair=pair,
        planted_mapping=planted,
        family=family,
        source_size=source_size,
        deletion_rate=deletion_rate,
        instance_seed=seed,
    )


def generate_split(
    sizes: list[int],
    families: list[str],
    deletion_rates: list[float],
    pairs_per_cell: int,
    split_seed: int,
    distractor_fraction: float,
    random_edge_density: float,
    motif_size: int,
) -> list[ExactSyntheticPair]:
    generated = []
    sequence = np.random.SeedSequence(split_seed)
    count = len(sizes) * len(families) * len(deletion_rates) * pairs_per_cell
    child_seeds = sequence.spawn(count)
    position = 0
    for size in sizes:
        for family in families:
            for deletion_rate in deletion_rates:
                for _ in range(pairs_per_cell):
                    seed = int(child_seeds[position].generate_state(1)[0])
                    position += 1
                    generated.append(
                        make_exact_pair(
                            source_size=size,
                            family=family,
                            deletion_rate=deletion_rate,
                            distractor_fraction=distractor_fraction,
                            random_edge_density=random_edge_density,
                            motif_size=motif_size,
                            seed=seed,
                        )
                    )
    return generated
