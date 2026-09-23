import torch

from nema.association import AssociationGraph
from nema.graph import LabeledGraph
from scripts.run_hard_self_distill_pilot import (
    active_teacher_rows,
    teacher_nll,
    validation_pass,
)


def path_graph(labels):
    count = len(labels)
    return LabeledGraph(
        node_labels=torch.tensor(labels),
        edge_index=torch.tensor([[i for i in range(count - 1)], [i + 1 for i in range(count - 1)]]),
        edge_labels=torch.ones(count - 1, dtype=torch.long),
    )


def test_active_teacher_rows_excludes_unmatched_padding_rows():
    association = AssociationGraph.build(path_graph([1, 1, 2]), path_graph([1, 1, 2, 3]))
    mapping = torch.tensor([0, 1, 3])
    active = active_teacher_rows(association, mapping)
    assert active.tolist() == [0, 1]


def test_teacher_nll_prefers_assignment_concentrated_on_teacher():
    mapping = torch.tensor([0, 1])
    active = torch.tensor([0, 1])
    good = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    bad = torch.tensor([[0.1, 0.9], [0.9, 0.1]])
    assert teacher_nll(good, mapping, active) < teacher_nll(bad, mapping, active)


def test_validation_gate_requires_effect_and_two_positive_seeds_everywhere():
    config = {"validation": {"pass_min_dataset_mean": 0.005, "pass_positive_seeds_per_dataset": 2}}
    summary = {"datasets": {
        "AIDS": {"mean": 0.006, "positive_seeds": 3},
        "MOLHIV": {"mean": 0.007, "positive_seeds": 2},
        "MCF-7": {"mean": 0.008, "positive_seeds": 2},
    }}
    assert validation_pass(summary, config)
    summary["datasets"]["MCF-7"]["mean"] = 0.004
    assert not validation_pass(summary, config)
