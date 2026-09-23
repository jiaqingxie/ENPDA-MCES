import torch

from nema.association import AssociationGraph
from nema.graph import LabeledGraph
from nema.models.capacity import CapacityNEMAModel
from nema.features import structural_features


def graph(labels, edges):
    edge_index = torch.tensor([[u, v] for u, v, _ in edges], dtype=torch.long).t()
    edge_labels = torch.tensor([label for _, _, label in edges], dtype=torch.long)
    return LabeledGraph(torch.tensor(labels), edge_index, edge_labels)


def test_acg_message_metric_starts_at_exact_unit_and_is_equivariant():
    left = graph([1, 1, 2], [(0, 1, 4), (1, 2, 5), (0, 2, 5)])
    right = graph([2, 1, 1, 7], [(1, 2, 4), (2, 0, 5), (1, 0, 5), (0, 3, 9)])
    association = AssociationGraph.build(left, right)
    model = CapacityNEMAModel("acg_gnn", 16, steps=1)
    model.metric.bind(association)
    metric = model.metric(torch.randn(*association.shape, 11))
    model.metric.bind(None)
    assert torch.equal(metric, torch.ones_like(metric))

    with torch.no_grad():
        reference = model(association, iterations=1, line_search=False, stationary_safeguard=False)
    p_left = torch.tensor([2, 0, 1]); p_right = torch.tensor([1, 3, 0, 2])
    permuted = AssociationGraph.build(left.permute(p_left), right.permute(p_right))
    with torch.no_grad():
        candidate = model(permuted, iterations=1, line_search=False, stationary_safeguard=False)
    expected = torch.empty_like(reference.assignment)
    for i in range(left.num_nodes):
        for j in range(right.num_nodes):
            expected[p_left[i], p_right[j]] = reference.assignment[i, j]
    assert torch.allclose(candidate.assignment, expected, atol=2e-5)


def test_acg_message_metric_receives_gradients_after_unit_anchor():
    left = graph([1, 1, 2], [(0, 1, 4), (1, 2, 5)])
    right = graph([1, 2, 1], [(0, 1, 5), (2, 0, 4)])
    association = AssociationGraph.build(left, right)
    model = CapacityNEMAModel("acg_gnn", 16, steps=1)
    output = model(association, iterations=1, line_search=False, stationary_safeguard=False)
    (-output.objectives[-1]).backward()
    assert any(parameter.grad is not None for parameter in model.metric.parameters())
