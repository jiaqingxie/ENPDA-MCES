import torch

from nema.association import AssociationGraph
from nema.enpda_anytime import search_trace
from nema.graph import LabeledGraph


def _graph(labels, edges):
    return LabeledGraph(
        node_labels=torch.tensor(labels),
        edge_index=torch.tensor([[u, v] for u, v, _ in edges], dtype=torch.long).t(),
        edge_labels=torch.tensor([label for _, _, label in edges], dtype=torch.long),
    )


def test_zero_budget_is_the_supplied_partial_hungarian_and_snapshots_are_monotone():
    left = _graph([1, 1, 2], [(0, 1, 4), (1, 2, 5), (0, 2, 5)])
    right = _graph([2, 1, 1, 7], [(1, 2, 4), (2, 0, 5), (1, 0, 5)])
    association = AssociationGraph.build(left, right)
    scores = torch.rand(association.shape)
    initial = torch.tensor([1, 2, -1])
    snapshots, audit = search_trace(
        association,
        scores,
        initial,
        (0.0, 0.01),
        restarts=1,
        max_passes=1,
        anneal_steps=0,
        lns_steps=0,
        seed=7,
    )

    assert torch.equal(snapshots[0.0].mapping, initial)
    assert snapshots[0.0].source == "core_partial_hungarian"
    assert (snapshots[0.01].common_edges, snapshots[0.01].common_nodes) >= (
        snapshots[0.0].common_edges,
        snapshots[0.0].common_nodes,
    )
    assert audit["event_count"] >= 1
