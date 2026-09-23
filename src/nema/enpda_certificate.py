"""Cheap valid MCES upper bounds repaired from ENPDA target prices.

The learned prices are *not* by themselves a QAP dual certificate.  We first
dominate each possible matched node's incident-edge contribution by a linear
assignment weight, then repair the target prices into a feasible dual of that
linear assignment relaxation.  Weak duality makes the resulting value a
rigorous MCES upper bound for every network output, including an untrained or
out-of-distribution one.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from nema.association import AssociationGraph
from nema.graph import LabeledGraph


@dataclass(frozen=True)
class PriceCertificate:
    upper_bound: float
    raw_upper_bound: float
    best_scale: float
    row_potentials: torch.Tensor
    target_potentials: torch.Tensor
    incident_weights: torch.Tensor
    label_histogram_upper_bound: int
    edge_count_upper_bound: int
    exact_linear_assignment_upper_bound: float | None
    runtime_seconds: float


def _incident_signatures(graph: LabeledGraph) -> list[Counter[tuple[int, int]]]:
    signatures = [Counter() for _ in range(graph.num_nodes)]
    for edge in range(graph.num_edges):
        u = int(graph.edge_index[0, edge])
        v = int(graph.edge_index[1, edge])
        label = int(graph.edge_labels[edge])
        signatures[u][(label, int(graph.node_labels[v]))] += 1
        signatures[v][(label, int(graph.node_labels[u]))] += 1
    return signatures


def incident_assignment_weights(association: AssociationGraph) -> torch.Tensor:
    """Return a linear upper-bound weight for each compatible node pair.

    A preserved edge incident to ``i -> j`` must use the same edge label and
    map the opposite endpoint to a node with the same label.  Consequently its
    contribution to the preserved degree is bounded by the multiset
    intersection of incident ``(edge label, neighbour label)`` signatures.
    Dividing by two converts the sum of preserved degrees into edge count.
    """

    left = _incident_signatures(association.left)
    right = _incident_signatures(association.right)
    weights = torch.zeros(association.shape, dtype=torch.float64)
    mask = association.candidate_mask.detach().cpu()
    for i, j in torch.nonzero(mask, as_tuple=False).tolist():
        overlap = sum(min(count, right[j].get(signature, 0)) for signature, count in left[i].items())
        weights[i, j] = 0.5 * float(overlap)
    return weights


def _edge_signature_histogram(graph: LabeledGraph) -> Counter[tuple[int, int, int]]:
    histogram: Counter[tuple[int, int, int]] = Counter()
    for edge in range(graph.num_edges):
        u = int(graph.edge_index[0, edge])
        v = int(graph.edge_index[1, edge])
        lu, lv = int(graph.node_labels[u]), int(graph.node_labels[v])
        histogram[(min(lu, lv), int(graph.edge_labels[edge]), max(lu, lv))] += 1
    return histogram


def label_histogram_upper_bound(association: AssociationGraph) -> int:
    left = _edge_signature_histogram(association.left)
    right = _edge_signature_histogram(association.right)
    return int(sum(min(count, right.get(signature, 0)) for signature, count in left.items()))


def exact_linear_assignment_bound(
    weights: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> float:
    """Maximize the incident linear bound over partial injective assignments."""

    values = weights.detach().cpu().numpy().astype(np.float64, copy=True)
    mask = candidate_mask.detach().cpu().numpy().astype(bool, copy=False)
    rows, columns = values.shape
    augmented = np.zeros((rows, columns + rows), dtype=np.float64)
    augmented[:, :columns] = np.where(mask, values, 0.0)
    row_ids, col_ids = linear_sum_assignment(-augmented)
    return float(augmented[row_ids, col_ids].sum())


def repair_target_prices(
    weights: torch.Tensor,
    candidate_mask: torch.Tensor,
    prices: torch.Tensor,
    *,
    scales: int = 129,
) -> tuple[float, float, torch.Tensor, torch.Tensor]:
    """Repair nonnegative neural prices into a feasible assignment dual.

    For any scale ``a >= 0``, set ``c_j=a p_j`` and
    ``r_i=max(0,max_j(w_ij-c_j))`` on compatible entries.  Then
    ``r_i+c_j >= w_ij`` and the potentials are dual-feasible for the partial
    assignment LP.  We choose the best member of a deterministic finite scale
    grid; choosing among valid dual points cannot compromise validity.
    """

    if scales < 2:
        raise ValueError("at least two scale candidates are required")
    w = weights.detach().cpu().to(torch.float64)
    mask = candidate_mask.detach().cpu().bool()
    p = prices.detach().cpu().to(torch.float64).clamp_min(0.0)
    if p.shape != (w.shape[1],):
        raise ValueError(f"expected {w.shape[1]} target prices, got {tuple(p.shape)}")
    positive = mask & (p[None, :] > 0) & (w > 0)
    ratios = (w / p.clamp_min(torch.finfo(torch.float64).tiny)[None, :])[positive]
    if ratios.numel():
        probabilities = torch.linspace(0.0, 1.0, scales - 2, dtype=torch.float64)
        candidates = torch.cat((torch.tensor([0.0, 1.0]), torch.quantile(ratios, probabilities))).unique()
    else:
        candidates = torch.tensor([0.0, 1.0], dtype=torch.float64)
    best_value = math.inf
    best_scale = 0.0
    best_rows = torch.zeros(w.shape[0], dtype=torch.float64)
    best_columns = torch.zeros(w.shape[1], dtype=torch.float64)
    negative_infinity = torch.tensor(-torch.inf, dtype=torch.float64)
    for scale in candidates.tolist():
        columns = float(scale) * p
        slack = torch.where(mask, w - columns[None, :], negative_infinity)
        rows = slack.max(dim=1).values.clamp_min(0.0)
        value = float(rows.sum() + columns.sum())
        if value < best_value:
            best_value = value
            best_scale = float(scale)
            best_rows = rows
            best_columns = columns
    # One final explicit feasibility audit guards all future refactors.
    if bool(((best_rows[:, None] + best_columns[None, :] + 1e-12 < w) & mask).any()):
        raise RuntimeError("price repair failed dual feasibility")
    return best_value, best_scale, best_rows, best_columns


def certify_from_prices(
    association: AssociationGraph,
    prices: torch.Tensor,
    *,
    compute_exact_linear_bound: bool = True,
) -> PriceCertificate:
    started = time.perf_counter()
    weights = incident_assignment_weights(association)
    price_value, scale, rows, columns = repair_target_prices(
        weights, association.candidate_mask, prices
    )
    histogram = label_histogram_upper_bound(association)
    edge_cap = min(association.left.num_edges, association.right.num_edges)
    exact_linear = (
        exact_linear_assignment_bound(weights, association.candidate_mask)
        if compute_exact_linear_bound
        else None
    )
    components = [float(price_value), float(histogram), float(edge_cap)]
    if exact_linear is not None:
        components.append(float(exact_linear))
    raw = min(components)
    safe = float(np.nextafter(raw + 1e-10 * max(1.0, abs(raw)), np.inf))
    return PriceCertificate(
        upper_bound=safe,
        raw_upper_bound=raw,
        best_scale=scale,
        row_potentials=rows,
        target_potentials=columns,
        incident_weights=weights,
        label_histogram_upper_bound=histogram,
        edge_count_upper_bound=edge_cap,
        exact_linear_assignment_upper_bound=exact_linear,
        runtime_seconds=time.perf_counter() - started,
    )
