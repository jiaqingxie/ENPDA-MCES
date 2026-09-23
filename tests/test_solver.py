import torch

from nema.graph import GraphPair, LabeledGraph
from nema.models.nema import NEMAModel
from nema.solvers import NGASolver, NEMASolver


def _graph(labels, edges):
    return LabeledGraph(
        torch.tensor(labels),
        torch.tensor([[u, v] for u, v, _ in edges]).t(),
        torch.tensor([label for _, _, label in edges]),
    )


def test_nema_and_nga_smoke():
    left = _graph([1, 1, 2], [(0, 1, 1), (1, 2, 2)])
    right = _graph([2, 1, 1, 3], [(1, 2, 1), (2, 0, 2), (0, 3, 7)])
    pair = GraphPair(left, right, true_edges=2, true_nodes=3, true_similarity=0.25, key="toy")

    nema = NEMASolver(
        NEMAModel(steps=2, hidden_dim=8),
        continuous_restarts=1,
        discrete_restarts=1,
        refinement_passes=3,
    ).solve(pair)
    assert nema.common_edges == 2

    nga = NGASolver(
        epochs=2,
        hidden_dim=8,
        encoder_layers=2,
        steps=2,
        samples=2,
        evaluation_interval=1,
        time_budget=10,
    ).solve(pair)
    assert 0 <= nga.common_edges <= 2


def test_nema_ablation_metadata_disables_fallback():
    left = _graph([1, 1, 2], [(0, 1, 1), (1, 2, 2)])
    right = _graph([2, 1, 1], [(1, 2, 1), (2, 0, 2)])
    pair = GraphPair(left, right, true_edges=2, true_nodes=3, key="ablation")

    result = NEMASolver(
        NEMAModel(steps=1, hidden_dim=8),
        continuous_restarts=1,
        discrete_restarts=1,
        refinement_passes=2,
        trajectory="unit",
        initializer_mode="fixed",
        schedule_mode="fixed",
        unit_fallback=False,
        line_search=False,
    ).solve(pair)

    assert result.metadata["primary_trajectory"] == "unit_metric"
    assert result.metadata["initializer_mode"] == "fixed"
    assert result.metadata["schedule_mode"] == "fixed"
    assert result.metadata["trajectory_portfolio"] == ["unit_metric"]
    assert result.metadata["unit_metric_fallback"] is False
    assert result.metadata["line_search"] is False
