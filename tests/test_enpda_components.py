import torch

from nema.association import AssociationGraph
from nema.enpda_components import component_forward
from nema.graph import LabeledGraph
from nema.models.enpda import ENPDAModel


def _graph(labels, edges):
    return LabeledGraph(
        node_labels=torch.tensor(labels),
        edge_index=torch.tensor([[u, v] for u, v, _ in edges], dtype=torch.long).t(),
        edge_labels=torch.tensor([label for _, _, label in edges], dtype=torch.long),
    )


def _association():
    left = _graph([1, 1, 2], [(0, 1, 4), (1, 2, 5), (0, 2, 5)])
    right = _graph([2, 1, 1, 7], [(1, 2, 4), (2, 0, 5), (1, 0, 5)])
    return AssociationGraph.build(left, right)


def test_no_price_arms_keep_dual_prices_identically_zero_and_restore_model():
    association = _association()
    model = ENPDAModel(steps=3, hidden_dim=16, message_layers=1)
    original = model.base_dual_step
    for arm in ("no_price", "full_no_price"):
        output = component_forward(model, association, arm)
        assert torch.equal(output.prices, torch.zeros_like(output.prices))
        assert model.base_dual_step == original


def test_component_arms_share_shape_rounds_and_injective_rounding():
    association = _association()
    model = ENPDAModel(steps=2, hidden_dim=16, message_layers=1)
    for arm in (
        "no_price",
        "full_no_price",
        "analytic",
        "initializer_only",
        "dynamics_only",
        "full",
    ):
        output = component_forward(model, association, arm)
        mapping = output.hard_mapping()
        active = mapping[mapping >= 0]
        assert output.assignment.shape == association.shape
        assert output.objectives.shape == (3,)
        assert len(set(active.tolist())) == active.numel()
