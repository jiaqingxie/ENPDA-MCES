"""Strict post-Core anytime traces for matched ENPDA proposal comparisons."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from nema.association import AssociationGraph
from nema.rounding import (
    HardObjective,
    anneal_mapping,
    constructive_mapping,
    hungarian_mapping,
    large_neighborhood_refine,
    refine_mapping,
)


@dataclass(frozen=True)
class AnytimeSnapshot:
    budget_seconds: float
    completed_at_seconds: float
    common_edges: int
    common_nodes: int
    mapping: torch.Tensor
    source: str


def search_trace(
    association: AssociationGraph,
    scores: torch.Tensor,
    initial_mapping: torch.Tensor,
    budgets: tuple[float, ...],
    *,
    restarts: int,
    max_passes: int,
    anneal_steps: int,
    lns_steps: int,
    seed: int,
    on_candidate: Callable[[torch.Tensor, str], None] | None = None,
) -> tuple[dict[float, AnytimeSnapshot], dict]:
    """Return the best fully completed incumbent at every post-Core deadline.

    The initial partial-Hungarian mapping is available at time zero.  Later
    candidates are timestamped only after their complete search operation;
    an operation crossing a deadline is never back-filled into that deadline.
    """

    ordered_budgets = tuple(sorted({float(value) for value in budgets}))
    if not ordered_budgets or ordered_budgets[0] < 0:
        raise ValueError("budgets must be a non-empty set of nonnegative seconds")
    if restarts < 1:
        raise ValueError("restarts must be positive")

    started = time.perf_counter()
    objective = HardObjective(association)
    events: list[tuple[float, tuple[int, int], torch.Tensor, str]] = []

    def add(mapping: torch.Tensor, source: str) -> tuple[int, int]:
        candidate = mapping.detach().cpu().long().clone()
        stats = objective(candidate)
        completed_at = time.perf_counter() - started
        events.append((completed_at, stats, candidate, source))
        if on_candidate is not None:
            on_candidate(candidate, source)
        return stats

    # The Core output is the zero-search operating point by definition.
    initial = initial_mapping.detach().cpu().long().clone()
    initial_stats = objective(initial)
    events.append((0.0, initial_stats, initial, "core_partial_hungarian"))

    maximum_budget = ordered_budgets[-1]
    rng = np.random.default_rng(seed)
    def callback(stage):
        return None if on_candidate is None else lambda mapping: on_candidate(mapping, stage)

    full = hungarian_mapping(scores)
    best = refine_mapping(association, full, max_passes=max_passes,
                          on_candidate=callback('full_hungarian_refine:incumbent'))
    best_stats = add(best, "full_hungarian_refine")

    for restart in range(1, restarts):
        if time.perf_counter() - started >= maximum_budget:
            break
        mode = restart % 3
        if mode == 0:
            noise_scale = 0.25 + 1.25 * (restart / max(restarts - 1, 1))
            noise = torch.from_numpy(rng.gumbel(size=scores.shape)).to(scores) * noise_scale
            proposal = hungarian_mapping(scores.clamp_min(1e-12).log() + noise)
            source = f"gumbel_{restart}"
        elif mode == 1:
            proposal = constructive_mapping(association, scores, rng,
                on_candidate=callback(f'constructive_interleaved_{restart}:incumbent'))
            source = f"constructive_interleaved_{restart}"
        else:
            proposal = best.clone()
            swaps = 2 + restart % max(2, min(8, association.left.num_nodes // 3))
            for _ in range(swaps):
                i, j = rng.choice(association.left.num_nodes, size=2, replace=False)
                old_i = proposal[i].clone()
                proposal[i] = proposal[j]
                proposal[j] = old_i
            source = f"swap_{restart}"
        proposal = anneal_mapping(association, proposal, anneal_steps, rng,
            on_candidate=callback(source + ':anneal_incumbent'))
        proposal = refine_mapping(association, proposal, max_passes=max_passes,
            on_candidate=callback(source + ':refine_incumbent'))
        stats = add(proposal, source)
        if stats > best_stats:
            best, best_stats = proposal, stats

    constructive_rng = np.random.default_rng(seed)
    for restart in range(restarts):
        if time.perf_counter() - started >= maximum_budget:
            break
        proposal = constructive_mapping(association, scores, constructive_rng,
            on_candidate=callback(f'constructive_{restart}:incumbent'))
        proposal = refine_mapping(association, proposal, max_passes=max_passes,
            on_candidate=callback(f'constructive_{restart}:refine_incumbent'))
        stats = add(proposal, f"constructive_{restart}")
        if stats > best_stats:
            best, best_stats = proposal, stats

    if time.perf_counter() - started < maximum_budget and lns_steps > 0:
        proposal = large_neighborhood_refine(association, best, scores, lns_steps, rng,
            on_candidate=callback('lns:incumbent'))
        stats = add(proposal, "lns")
        if stats > best_stats:
            best, best_stats = proposal, stats
    if time.perf_counter() - started < maximum_budget:
        proposal = refine_mapping(association, best, max_passes=max_passes,
            on_candidate=callback('final_refine:incumbent'))
        add(proposal, "final_refine")

    snapshots: dict[float, AnytimeSnapshot] = {}
    for budget in ordered_budgets:
        eligible = [event for event in events if event[0] <= budget]
        if not eligible:
            raise RuntimeError(f"no incumbent available at budget {budget}")
        completed_at, stats, mapping, source = max(eligible, key=lambda event: event[1])
        snapshots[budget] = AnytimeSnapshot(
            budget_seconds=budget,
            completed_at_seconds=completed_at,
            common_edges=stats[0],
            common_nodes=stats[1],
            mapping=mapping,
            source=source,
        )
    return snapshots, {
        "event_count": len(events),
        "search_elapsed_seconds": time.perf_counter() - started,
        "last_event_seconds": events[-1][0],
        "maximum_budget_seconds": maximum_budget,
    }
