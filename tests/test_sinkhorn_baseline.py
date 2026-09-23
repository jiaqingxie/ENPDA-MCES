import torch

from nema.association import AssociationGraph
from nema.graph import LabeledGraph
from nema.models.sinkhorn_baseline import TrainOnceSinkhorn


def graph(labels, edges):
    return LabeledGraph(
        torch.tensor(labels),
        torch.tensor(edges, dtype=torch.long).t().contiguous(),
        torch.zeros(len(edges), dtype=torch.long),
    )


def test_train_once_sinkhorn_is_feasible_and_hard_mapping_is_injective():
    left = graph([0, 1, 0], [(0, 1), (1, 2)])
    right = graph([0, 1, 0, 1], [(0, 1), (1, 2), (2, 3)])
    association = AssociationGraph.build(left, right)
    model = TrainOnceSinkhorn(hidden_dim=8, message_layers=1)
    output = model(association)
    assert output.row_residual < 1e-5
    # The control intentionally matches NGA's finite 20-iteration projection;
    # hard feasibility is provided by the single Hungarian projection below.
    assert output.max_column_excess < 5e-2
    mapping = output.hard_mapping()
    assigned = mapping[mapping >= 0]
    assert len(assigned) == len(torch.unique(assigned))


def test_gumbel_sinkhorn_is_reproducible_under_fixed_generator():
    left = graph([0, 1, 0], [(0, 1), (1, 2)])
    right = graph([0, 1, 0, 1], [(0, 1), (1, 2), (2, 3)])
    association = AssociationGraph.build(left, right)
    model = TrainOnceSinkhorn(hidden_dim=8, message_layers=1)
    one, _ = model.gumbel_mappings(
        association,
        samples=4,
        generator=torch.Generator().manual_seed(7),
    )
    two, _ = model.gumbel_mappings(
        association,
        samples=4,
        generator=torch.Generator().manual_seed(7),
    )
    assert all(torch.equal(a, b) for a, b in zip(one, two, strict=True))
