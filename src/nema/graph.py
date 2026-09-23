"""Small, dependency-light graph containers used by both solvers."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LabeledGraph:
    """An undirected simple graph with integer node and edge labels.

    ``edge_index`` contains every undirected edge exactly once. Its shape is
    ``[2, num_edges]``.
    """

    node_labels: torch.Tensor
    edge_index: torch.Tensor
    edge_labels: torch.Tensor

    def __post_init__(self) -> None:
        if self.node_labels.ndim != 1:
            raise ValueError("node_labels must be one-dimensional")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if self.edge_labels.ndim != 1 or self.edge_labels.numel() != self.edge_index.shape[1]:
            raise ValueError("one edge label is required per edge")

    @property
    def num_nodes(self) -> int:
        return int(self.node_labels.numel())

    @property
    def num_edges(self) -> int:
        return int(self.edge_labels.numel())

    @property
    def degrees(self) -> torch.Tensor:
        degree = torch.zeros(self.num_nodes, dtype=torch.float32)
        if self.num_edges:
            ones = torch.ones(self.num_edges, dtype=torch.float32)
            degree.index_add_(0, self.edge_index[0], ones)
            degree.index_add_(0, self.edge_index[1], ones)
        return degree

    @classmethod
    def from_pyg(cls, data: object) -> "LabeledGraph":
        """Convert a PyG ``Data`` object and remove duplicated edge directions."""

        x = torch.as_tensor(data.x).detach().cpu()
        if x.ndim > 1:
            x = x[:, 0]
        raw_index = torch.as_tensor(data.edge_index).detach().cpu().long()
        raw_label = torch.as_tensor(data.edge_attr).detach().cpu()
        if raw_label.ndim > 1:
            raw_label = raw_label[:, 0]
        raw_label = raw_label.long()

        edges: dict[tuple[int, int], int] = {}
        for k in range(raw_index.shape[1]):
            u, v = (int(raw_index[0, k]), int(raw_index[1, k]))
            if u == v:
                continue
            key = (u, v) if u < v else (v, u)
            label = int(raw_label[k])
            if key in edges and edges[key] != label:
                raise ValueError(f"conflicting labels for undirected edge {key}")
            edges[key] = label

        ordered = sorted(edges)
        if ordered:
            edge_index = torch.tensor(ordered, dtype=torch.long).t().contiguous()
            edge_labels = torch.tensor([edges[e] for e in ordered], dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_labels = torch.empty((0,), dtype=torch.long)
        return cls(x.long().contiguous(), edge_index, edge_labels)

    def permute(self, permutation: torch.Tensor) -> "LabeledGraph":
        """Return a relabeled graph; ``permutation[old]`` is the new index."""

        permutation = permutation.long().cpu()
        if sorted(permutation.tolist()) != list(range(self.num_nodes)):
            raise ValueError("permutation must contain every node exactly once")
        new_labels = torch.empty_like(self.node_labels)
        new_labels[permutation] = self.node_labels
        new_edges = permutation[self.edge_index]
        if self.num_edges:
            lo = torch.minimum(new_edges[0], new_edges[1])
            hi = torch.maximum(new_edges[0], new_edges[1])
            order = torch.argsort(lo * self.num_nodes + hi)
            new_edges = torch.stack((lo[order], hi[order]))
            new_edge_labels = self.edge_labels[order]
        else:
            new_edge_labels = self.edge_labels.clone()
        return LabeledGraph(new_labels, new_edges, new_edge_labels)


@dataclass(frozen=True)
class GraphPair:
    left: LabeledGraph
    right: LabeledGraph
    true_edges: int | None = None
    true_nodes: int | None = None
    true_similarity: float | None = None
    key: str = ""
    metadata: dict | None = None

    def oriented(self) -> tuple[LabeledGraph, LabeledGraph, bool]:
        """Put the smaller graph on the row side of the partial assignment."""

        if self.left.num_nodes <= self.right.num_nodes:
            return self.left, self.right, False
        return self.right, self.left, True
