"""Permutation-equivariant, label-aware candidate features."""

from __future__ import annotations

from collections import Counter

import torch

from nema.association import AssociationGraph
from nema.graph import LabeledGraph


def _neighborhood_counters(
    graph: LabeledGraph,
) -> tuple[list[Counter], list[Counter], list[Counter], list[Counter]]:
    neighbor_labels = [Counter() for _ in range(graph.num_nodes)]
    bond_labels = [Counter() for _ in range(graph.num_nodes)]
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(graph.num_nodes)]
    for k in range(graph.num_edges):
        u, v = int(graph.edge_index[0, k]), int(graph.edge_index[1, k])
        bond = int(graph.edge_labels[k])
        adjacency[u].append((v, bond))
        adjacency[v].append((u, bond))
        neighbor_labels[u][int(graph.node_labels[v])] += 1
        neighbor_labels[v][int(graph.node_labels[u])] += 1
        bond_labels[u][bond] += 1
        bond_labels[v][bond] += 1
    path2 = [Counter() for _ in range(graph.num_nodes)]
    path3 = [Counter() for _ in range(graph.num_nodes)]
    for root in range(graph.num_nodes):
        for first, edge1 in adjacency[root]:
            label1 = int(graph.node_labels[first])
            for second, edge2 in adjacency[first]:
                if second == root:
                    continue
                label2 = int(graph.node_labels[second])
                path2[root][(edge1, label1, edge2, label2)] += 1
                for third, edge3 in adjacency[second]:
                    if third in (root, first):
                        continue
                    label3 = int(graph.node_labels[third])
                    path3[root][(edge1, label1, edge2, label2, edge3, label3)] += 1
    return neighbor_labels, bond_labels, path2, path3


def _counter_similarity(left: Counter, right: Counter) -> float:
    keys = left.keys() | right.keys()
    if not keys:
        return 1.0
    overlap = sum(min(left[k], right[k]) for k in keys)
    union = sum(max(left[k], right[k]) for k in keys)
    return overlap / max(union, 1)


def structural_features(association: AssociationGraph) -> torch.Tensor:
    """Return [label compatibility, degree, neighbor-label, bond-label] features."""

    left, right = association.left, association.right
    left_neigh, left_bond, left_path2, left_path3 = _neighborhood_counters(left)
    right_neigh, right_bond, right_path2, right_path3 = _neighborhood_counters(right)
    result = torch.empty((*association.shape, 6), dtype=torch.float32)
    for i in range(left.num_nodes):
        for j in range(right.num_nodes):
            d1, d2 = float(left.degrees[i]), float(right.degrees[j])
            result[i, j, 0] = float(left.node_labels[i] == right.node_labels[j])
            result[i, j, 1] = 1.0 / (1.0 + abs(d1 - d2))
            result[i, j, 2] = _counter_similarity(left_neigh[i], right_neigh[j])
            result[i, j, 3] = _counter_similarity(left_bond[i], right_bond[j])
            result[i, j, 4] = _counter_similarity(left_path2[i], right_path2[j])
            result[i, j, 5] = _counter_similarity(left_path3[i], right_path3[j])
    return result
