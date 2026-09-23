import torch

from nema.association import AssociationGraph
from nema.enpda_solver import ENPDASolver
from nema.graph import GraphPair, LabeledGraph
from nema.models.enpda import ENPDAModel


def _graph(labels, edges):
    return LabeledGraph(
        node_labels=torch.tensor(labels),
        edge_index=torch.tensor([[u, v] for u, v, _ in edges], dtype=torch.long).t(),
        edge_labels=torch.tensor([label for _, _, label in edges], dtype=torch.long),
    )


def _toy_association():
    left = _graph([1, 1, 2], [(0, 1, 4), (1, 2, 5), (0, 2, 5)])
    right = _graph(
        [2, 1, 1, 7],
        [(1, 2, 4), (2, 0, 5), (1, 0, 5), (0, 3, 9)],
    )
    return left, right, AssociationGraph.build(left, right)


def test_enpda_analytic_initialization_and_partial_feasibility():
    _, _, association = _toy_association()
    model = ENPDAModel(steps=3, hidden_dim=16, message_layers=1)

    output = model(association, mode="analytic")
    mapping = output.hard_mapping()

    assert output.objectives.shape == (4,)
    assert output.prices.shape == (association.shape[1],)
    assert output.row_residual < 1e-6
    assert len(set(mapping[mapping >= 0].tolist())) == int((mapping >= 0).sum())
    assert all(value < association.shape[1] for value in mapping[mapping >= 0].tolist())


def test_enpda_starts_at_exact_analytic_auction():
    _, _, association = _toy_association()
    model = ENPDAModel(steps=2, hidden_dim=16, message_layers=1)

    learned = model(association, mode="learned")
    analytic = model(association, mode="analytic")

    assert torch.equal(learned.assignment, analytic.assignment)
    assert torch.equal(learned.prices, analytic.prices)
    assert torch.equal(learned.objectives, analytic.objectives)


def test_enpda_commutes_with_independent_node_permutations():
    left, right, association = _toy_association()
    model = ENPDAModel(steps=2, hidden_dim=16, message_layers=2)
    with torch.no_grad():
        # Move away from the zero-residual initialization so the test exercises
        # the learned initializer, correction, and learned step heads.
        for parameter in model.parameters():
            parameter.add_(0.01 * torch.randn_like(parameter))

    reference = model(association, mode="learned")
    p_left = torch.tensor([2, 0, 1])
    p_right = torch.tensor([1, 3, 0, 2])
    permuted = AssociationGraph.build(left.permute(p_left), right.permute(p_right))
    candidate = model(permuted, mode="learned")

    expected = torch.empty_like(reference.assignment)
    expected[p_left[:, None], p_right[None, :]] = reference.assignment
    expected_prices = torch.empty_like(reference.prices)
    expected_prices[p_right] = reference.prices
    assert torch.allclose(candidate.assignment, expected, atol=2e-5)
    assert torch.allclose(candidate.prices, expected_prices, atol=2e-5)


def test_enpda_neural_dynamics_receives_hard_teacher_gradients():
    _, _, association = _toy_association()
    model = ENPDAModel(steps=2, hidden_dim=16, message_layers=1)
    output = model(association, mode="learned")
    mapping = output.hard_mapping()
    active = mapping >= 0
    loss = -output.assignment[active, mapping[active]].clamp_min(1e-12).log().mean()
    loss.backward()

    assert model.correction_head.weight.grad is not None
    assert model.initial_residual.weight.grad is not None


def test_enpda_initial_noise_is_explicit_and_reproducible():
    _, _, association = _toy_association()
    model = ENPDAModel(steps=1, hidden_dim=16, message_layers=1)
    noise = torch.randn(association.shape)
    plain = model(association, mode="analytic")
    perturbed = model(association, mode="analytic", initial_noise=noise)
    repeated = model(association, mode="analytic")

    assert not torch.allclose(plain.assignment, perturbed.assignment)
    assert torch.allclose(plain.assignment, repeated.assignment)


def test_enpda_core_solver_returns_an_injective_partial_mapping():
    left, right, _ = _toy_association()
    pair = GraphPair(left=left, right=right, key="toy")
    model = ENPDAModel(steps=1, hidden_dim=16, message_layers=1)
    result = ENPDASolver(
        model,
        trajectory="learned",
        hard_search=False,
        device="cpu",
    ).solve(pair)
    active = [value for value in result.mapping if value >= 0]

    assert len(active) == len(set(active))
    assert result.metadata["hard_search"] is False
