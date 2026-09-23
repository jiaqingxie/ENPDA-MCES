from __future__ import annotations

import itertools

import torch

from nema.association import AssociationGraph
from nema.enpda_certificate import certify_from_prices, incident_assignment_weights
from nema.graph import LabeledGraph


def graph(labels: list[int], edges: list[tuple[int, int, int]]) -> LabeledGraph:
    return LabeledGraph(
        torch.tensor(labels),
        torch.tensor([[u for u, _, _ in edges], [v for _, v, _ in edges]], dtype=torch.long),
        torch.tensor([label for _, _, label in edges]),
    )


def brute_force(association: AssociationGraph) -> int:
    rows, columns = association.shape
    best = 0
    # -1 is unmatched; reject repeated real columns.
    for values in itertools.product(range(-1, columns), repeat=rows):
        real = [value for value in values if value >= 0]
        if len(real) != len(set(real)):
            continue
        mapping = torch.tensor(values)
        invalid = any(value >= 0 and not association.candidate_mask[row, value] for row, value in enumerate(values))
        if invalid:
            continue
        best = max(best, association.hard_statistics(mapping)[0])
    return best


def test_price_certificate_bounds_exact_mces() -> None:
    left = graph([0, 1, 1, 2], [(0, 1, 0), (0, 2, 0), (2, 3, 1)])
    right = graph([0, 1, 1, 2, 2], [(0, 1, 0), (0, 2, 0), (1, 3, 1), (2, 4, 1)])
    association = AssociationGraph.build(left, right)
    exact = brute_force(association)
    for prices in (torch.zeros(right.num_nodes), torch.tensor([0.2, 1.7, 0.1, 2.0, 0.4])):
        result = certify_from_prices(association, prices)
        assert result.upper_bound + 1e-8 >= exact
        assert result.exact_linear_assignment_upper_bound is not None
        assert result.exact_linear_assignment_upper_bound + 1e-8 >= exact


def test_incident_bound_counts_edges_once_after_halving_degrees() -> None:
    left = graph([0, 1], [(0, 1, 3)])
    right = graph([0, 1], [(0, 1, 3)])
    association = AssociationGraph.build(left, right)
    weights = incident_assignment_weights(association)
    assert weights[0, 0] == 0.5
    assert weights[1, 1] == 0.5
    result = certify_from_prices(association, torch.ones(2))
    assert result.raw_upper_bound == 1.0
