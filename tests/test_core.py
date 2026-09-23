import numpy as np
import pytest
import torch

from nema.association import AssociationGraph
from nema.graph import LabeledGraph
from nema.metrics import retrieval_metrics
from nema.retrieval_baselines import size_upper_bound_similarity, wl_cosine_similarity
from nema.models.nema import NEMAModel
from nema.rounding import HardObjective, perturb_and_refine, refine_mapping
from nema.sinkhorn import (
    SinkhornDiagnostics,
    log_sinkhorn,
    partial_assignment_residual,
    partial_dummy_logits,
)


def graph(labels, edges):
    edge_index = torch.tensor([[u, v] for u, v, _ in edges], dtype=torch.long).t()
    edge_labels = torch.tensor([label for _, _, label in edges], dtype=torch.long)
    return LabeledGraph(torch.tensor(labels), edge_index, edge_labels)


def test_acg_objective_matches_preserved_edge_count():
    left = graph([1, 1, 2], [(0, 1, 4), (1, 2, 5), (0, 2, 5)])
    right = graph([2, 1, 1, 7], [(1, 2, 4), (2, 0, 5), (1, 0, 5), (0, 3, 9)])
    association = AssociationGraph.build(left, right)
    mapping = torch.tensor([1, 2, 0])
    edges, nodes = association.hard_statistics(mapping)
    assert (edges, nodes) == (3, 3)
    hard = torch.zeros(3, 4)
    hard[torch.arange(3), mapping] = 1
    assert association.objective(hard).item() == 2 * edges


def test_partial_sinkhorn_constraints_and_gradients():
    logits = torch.randn(2, 3, requires_grad=True)
    result = log_sinkhorn(logits, iterations=50)
    assert torch.allclose(result.sum(dim=1), torch.ones(2), atol=1e-4)
    assert bool((result.sum(dim=0) <= 1.0 + 1e-4).all())
    result.square().sum().backward()
    assert torch.isfinite(logits.grad).all()


def test_adaptive_partial_sinkhorn_reports_numerical_feasibility():
    generator = torch.Generator().manual_seed(17)
    logits = 8.0 * torch.randn(5, 9, generator=generator, dtype=torch.float64)
    result, diagnostics = log_sinkhorn(
        logits,
        iterations=2,
        tolerance=1e-9,
        max_iterations=500,
        return_diagnostics=True,
    )
    row_residual, column_excess = partial_assignment_residual(result)
    assert diagnostics.iterations >= 2
    assert diagnostics.iterations <= 500
    assert diagnostics.max_row_residual == pytest.approx(float(row_residual))
    assert diagnostics.max_column_excess == pytest.approx(float(column_excess))
    assert float(row_residual) <= 1e-9
    assert float(column_excess) <= 1e-9


def test_partial_dummy_reference_completes_column_slack():
    assignment = log_sinkhorn(
        torch.randn(3, 7, dtype=torch.float64),
        iterations=5,
        tolerance=1e-10,
        max_iterations=500,
    )
    dummy_logits = partial_dummy_logits(assignment)
    assert dummy_logits is not None
    augmented = torch.cat((assignment, dummy_logits.exp()), dim=0)
    assert torch.allclose(augmented.sum(dim=0), torch.ones(7, dtype=torch.float64), atol=1e-9)
    assert torch.allclose(augmented.sum(dim=1), torch.ones(7, dtype=torch.float64), atol=1e-9)


def test_qap_analytic_objective_expansion():
    left = graph([1, 1, 2], [(0, 1, 4), (1, 2, 5), (0, 2, 5)])
    right = graph([2, 1, 1, 7], [(1, 2, 4), (2, 0, 5), (1, 0, 5), (0, 3, 9)])
    association = AssociationGraph.build(left, right)
    current = torch.rand(association.shape)
    proposal = torch.rand(association.shape)
    displacement = proposal - current
    gradient = 2.0 * association.matvec(current)
    expanded = (
        association.objective(current)
        + (gradient * displacement).sum()
        + association.objective(displacement)
    )
    assert torch.allclose(expanded, association.objective(proposal), atol=1e-5)


def test_nema_is_monotone_and_equivariant():
    torch.manual_seed(4)
    left = graph([1, 1, 2], [(0, 1, 4), (1, 2, 5), (0, 2, 5)])
    right = graph([2, 1, 1, 7], [(1, 2, 4), (2, 0, 5), (1, 0, 5), (0, 3, 9)])
    model = NEMAModel(steps=4, hidden_dim=8)
    model.eval()
    with torch.no_grad():
        output = model(
            AssociationGraph.build(left, right),
            sinkhorn_tolerance=1e-7,
            sinkhorn_max_iterations=200,
            return_trace=True,
        )
    assert bool((output.objectives[1:] >= output.objectives[:-1] - 1e-6).all())
    assert output.metric_min > 0
    assert len(output.proposal_sources) == model.steps
    assert len(output.backtracking_trials) == model.steps
    assert all(1 <= value <= model.max_backtracks for value in output.backtracking_trials)
    assert len(output.unit_residuals) == model.steps
    assert output.assignment_trace is not None
    assert len(output.assignment_trace) == model.steps + 1
    assert output.max_row_residual <= 1e-6
    assert output.max_column_excess <= 1e-6

    p_left = torch.tensor([2, 0, 1])
    p_right = torch.tensor([1, 3, 0, 2])
    permuted = AssociationGraph.build(left.permute(p_left), right.permute(p_right))
    with torch.no_grad():
        permuted_output = model(
            permuted,
            sinkhorn_tolerance=1e-7,
            sinkhorn_max_iterations=200,
        )
    expected = torch.empty_like(output.assignment)
    for old_i in range(left.num_nodes):
        for old_j in range(right.num_nodes):
            expected[p_left[old_i], p_right[old_j]] = output.assignment[old_i, old_j]
    assert torch.allclose(permuted_output.assignment, expected, atol=2e-5)


def test_unit_proposal_is_tested_when_learned_projection_is_infeasible(monkeypatch):
    left = graph([1, 1], [(0, 1, 4)])
    right = graph([1, 1], [(0, 1, 4)])
    calls = []

    def fake_sinkhorn(logits, *args, **kwargs):
        calls.append(logits.detach().clone())
        if len(calls) == 1:  # Feasible initializer.
            assignment = torch.full_like(logits, 0.5)
            residual = 0.0
        elif len(calls) == 2:  # Learned proposal fails its finite audit.
            assignment = torch.full_like(logits, 0.5)
            residual = 1.0
        else:  # Unit proposal at the same eta remains eligible.
            assignment = torch.eye(2, dtype=logits.dtype, device=logits.device)
            residual = 0.0
        diagnostics = SinkhornDiagnostics(
            iterations=1,
            max_row_residual=residual,
            max_column_excess=0.0,
        )
        return assignment, diagnostics

    monkeypatch.setattr("nema.models.nema.log_sinkhorn", fake_sinkhorn)
    model = NEMAModel(steps=1, hidden_dim=8, max_backtracks=1).eval()
    with torch.no_grad():
        output = model(
            AssociationGraph.build(left, right),
            line_search=False,
            stationary_safeguard=True,
            sinkhorn_tolerance=1e-5,
            sinkhorn_max_iterations=20,
        )
    assert len(calls) == 3
    assert output.proposal_sources == ("unit_safeguard",)
    assert torch.equal(output.assignment, torch.eye(2))


def test_unit_metric_ablation_is_exact_and_validated():
    left = graph([1, 1, 2], [(0, 1, 4), (1, 2, 5)])
    right = graph([1, 2, 1], [(0, 1, 5), (2, 0, 4)])
    model = NEMAModel(steps=2, hidden_dim=8)
    with torch.no_grad():
        output = model(AssociationGraph.build(left, right), metric_mode="unit")
    assert output.metric_min == 1.0
    assert output.metric_max == 1.0
    with pytest.raises(ValueError, match="metric_mode"):
        model(AssociationGraph.build(left, right), metric_mode="invalid")


def test_fixed_initializer_and_schedule_are_independent_of_learned_parameters():
    left = graph([1, 1, 2], [(0, 1, 4), (1, 2, 5)])
    right = graph([1, 2, 1], [(0, 1, 5), (2, 0, 4)])
    association = AssociationGraph.build(left, right)
    model = NEMAModel(steps=2, hidden_dim=8)
    reference = NEMAModel(steps=2, hidden_dim=8)
    with torch.no_grad():
        model.initial_weights.add_(17.0)
        model.initial_bias.add_(11.0)
        model.step_logits.fill_(-8.0)
        fixed = model(
            association,
            initializer_mode="fixed",
            schedule_mode="fixed",
            metric_mode="unit",
            line_search=False,
            stationary_safeguard=False,
        )
        expected = reference(
            association,
            initializer_mode="fixed",
            schedule_mode="fixed",
            metric_mode="unit",
            line_search=False,
            stationary_safeguard=False,
        )
        learned_initializer = model(
            association,
            initializer_mode="learned",
            schedule_mode="fixed",
            metric_mode="unit",
            line_search=False,
            stationary_safeguard=False,
            iterations=0,
        )
    assert torch.allclose(fixed.assignment, expected.assignment)
    assert torch.allclose(fixed.objectives, expected.objectives)
    assert not torch.allclose(learned_initializer.assignment, expected.assignment)
    with pytest.raises(ValueError, match="initializer_mode"):
        model(association, initializer_mode="invalid")
    with pytest.raises(ValueError, match="schedule_mode"):
        model(association, schedule_mode="invalid")


def test_nema_can_continue_a_fixed_checkpoint_for_stationarity_audit():
    left = graph([1, 1, 2], [(0, 1, 4), (1, 2, 5)])
    right = graph([1, 2, 1], [(0, 1, 5), (2, 0, 4)])
    model = NEMAModel(steps=2, hidden_dim=8)
    with torch.no_grad():
        output = model(
            AssociationGraph.build(left, right),
            iterations=7,
            return_trace=True,
        )
    assert output.objectives.shape == (8,)
    assert len(output.accepted_steps) == 7
    assert len(output.backtracking_trials) == 7
    assert output.assignment_trace is not None and len(output.assignment_trace) == 8


def test_discrete_refinement_improves_bad_mapping():
    left = graph([1, 1, 2], [(0, 1, 4), (1, 2, 5), (0, 2, 5)])
    right = graph([2, 1, 1], [(1, 2, 4), (2, 0, 5), (1, 0, 5)])
    association = AssociationGraph.build(left, right)
    bad = torch.tensor([0, 1, 2])
    improved = refine_mapping(association, bad)
    assert association.hard_statistics(improved) > association.hard_statistics(bad)
    assert association.hard_statistics(improved) == (3, 3)


def test_vectorized_hard_objective_matches_acg_statistics():
    left = graph([1, 1, 2], [(0, 1, 4), (1, 2, 5), (0, 2, 5)])
    right = graph([2, 1, 1, 7], [(1, 2, 4), (2, 0, 5), (1, 0, 5), (0, 3, 9)])
    association = AssociationGraph.build(left, right)
    objective = HardObjective(association)
    for mapping in (
        torch.tensor([0, 1, 2]),
        torch.tensor([1, 2, 0]),
        torch.tensor([3, 0, 2]),
        torch.tensor([-1, 2, 0]),
    ):
        assert objective(mapping) == association.hard_statistics(mapping)


def test_constructive_restart_portfolio_is_anytime():
    left = graph([1, 1, 2], [(0, 1, 4), (1, 2, 5), (0, 2, 5)])
    right = graph([2, 1, 1, 7], [(1, 2, 4), (2, 0, 5), (1, 0, 5), (0, 3, 9)])
    association = AssociationGraph.build(left, right)
    scores = torch.rand(association.shape, generator=torch.Generator().manual_seed(5))
    short = perturb_and_refine(
        association,
        scores,
        restarts=1,
        anneal_steps=5,
        lns_steps=0,
        seed=3,
    )
    long = perturb_and_refine(
        association,
        scores,
        restarts=8,
        anneal_steps=5,
        lns_steps=0,
        seed=3,
    )
    assert association.hard_statistics(long) >= association.hard_statistics(short)


def test_retrieval_metrics_perfect_ranking():
    truth = torch.linspace(0, 1, 100).numpy()
    metrics = retrieval_metrics(truth, truth)
    assert metrics["MRR"] == 1.0
    assert metrics["Top10-overlap"] == 1.0
    assert metrics["Top10-AP"] == 1.0
    assert metrics["NDCG@10"] == 1.0
    assert metrics["queries"] == 1
    assert 0 <= metrics["P@10"] <= 1
    assert 0 <= metrics["MAP"] <= 1


def test_retrieval_metrics_match_pinned_official_evaluator():
    rng = np.random.default_rng(7)
    predicted = rng.random(200)
    truth = rng.random(200)
    metrics = retrieval_metrics(predicted, truth)
    assert metrics["MRR"] == pytest.approx(0.07083333333333333)
    assert metrics["P@10"] == pytest.approx(0.7)
    assert metrics["MAP"] == pytest.approx(0.5635083913803101)


def test_fast_retrieval_baselines_are_one_on_identical_graphs():
    graph = LabeledGraph(
        torch.tensor([6, 6, 8]),
        torch.tensor([[0, 1], [1, 2]]),
        torch.tensor([1, 2]),
    )
    assert size_upper_bound_similarity(graph, graph) == 1.0
    assert wl_cosine_similarity(graph, graph) == pytest.approx(1.0)
