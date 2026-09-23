import pytest
import torch

from nema.association import AssociationGraph
from nema.certificate import solve_lifted_lp, solve_lifted_milp
from nema.graph import LabeledGraph


def _graph(labels, edges):
    return LabeledGraph(
        torch.tensor(labels),
        torch.tensor([[u, v] for u, v, _ in edges], dtype=torch.long).t(),
        torch.tensor([label for _, _, label in edges], dtype=torch.long),
    )


def test_sparse_lifted_lp_and_milp_bound_known_optimum():
    left = _graph([1, 1, 2], [(0, 1, 4), (1, 2, 5), (0, 2, 5)])
    right = _graph(
        [2, 1, 1, 7],
        [(1, 2, 4), (2, 0, 5), (1, 0, 5), (0, 3, 9)],
    )
    association = AssociationGraph.build(left, right)

    lp = solve_lifted_lp(association)
    milp = solve_lifted_milp(association, time_limit=5.0)

    assert lp.status == 0
    assert lp.mode == "lp"
    assert lp.upper_bound is not None and lp.upper_bound >= 3.0
    assert milp.mapping is not None
    assert milp.lower_bound == 3
    assert milp.upper_bound == pytest.approx(3.0, abs=1e-6)
    assert milp.relative_gap == pytest.approx(0.0, abs=1e-6)
    assert milp.variables > 0
    assert milp.constraints > 0


def test_empty_acg_is_immediately_certified():
    left = _graph([1, 1], [(0, 1, 3)])
    right = _graph([1, 1], [(0, 1, 9)])
    association = AssociationGraph.build(left, right)
    for result in (solve_lifted_lp(association), solve_lifted_milp(association)):
        assert result.lower_bound == 0
        assert result.upper_bound == 0.0
        assert result.relative_gap == 0.0
