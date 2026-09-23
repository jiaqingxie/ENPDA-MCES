"""Sparse Association Common Graph (ACG) construction and objectives."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from nema.graph import LabeledGraph


@dataclass(frozen=True)
class AssociationGraph:
    """Sparse ACG over the Cartesian node-pair grid.

    ``edge_u`` and ``edge_v`` store each undirected ACG edge once. For a hard
    injective assignment, the number of selected ACG edges is exactly the MCES
    objective achieved by that assignment.
    """

    left: LabeledGraph
    right: LabeledGraph
    candidate_mask: torch.Tensor
    edge_u: torch.Tensor
    edge_v: torch.Tensor

    @property
    def shape(self) -> tuple[int, int]:
        return self.left.num_nodes, self.right.num_nodes

    @property
    def num_candidates(self) -> int:
        return self.left.num_nodes * self.right.num_nodes

    @property
    def num_edges(self) -> int:
        return int(self.edge_u.numel())

    @property
    def device(self) -> torch.device:
        return self.candidate_mask.device

    def to(self, device: torch.device | str) -> "AssociationGraph":
        return replace(
            self,
            candidate_mask=self.candidate_mask.to(device),
            edge_u=self.edge_u.to(device),
            edge_v=self.edge_v.to(device),
        )

    @classmethod
    def build(cls, left: LabeledGraph, right: LabeledGraph) -> "AssociationGraph":
        n1, n2 = left.num_nodes, right.num_nodes
        mask = left.node_labels[:, None].eq(right.node_labels[None, :])

        right_by_label: dict[int, list[tuple[int, int]]] = {}
        for k in range(right.num_edges):
            label = int(right.edge_labels[k])
            right_by_label.setdefault(label, []).append(
                (int(right.edge_index[0, k]), int(right.edge_index[1, k]))
            )

        acg_edges: set[tuple[int, int]] = set()
        for k in range(left.num_edges):
            u, v = int(left.edge_index[0, k]), int(left.edge_index[1, k])
            label = int(left.edge_labels[k])
            for x, y in right_by_label.get(label, ()):  # two possible orientations
                if bool(mask[u, x] and mask[v, y]):
                    a, b = u * n2 + x, v * n2 + y
                    acg_edges.add((a, b) if a < b else (b, a))
                if bool(mask[u, y] and mask[v, x]):
                    a, b = u * n2 + y, v * n2 + x
                    acg_edges.add((a, b) if a < b else (b, a))

        ordered = sorted(acg_edges)
        if ordered:
            edge_u = torch.tensor([e[0] for e in ordered], dtype=torch.long)
            edge_v = torch.tensor([e[1] for e in ordered], dtype=torch.long)
        else:
            edge_u = torch.empty(0, dtype=torch.long)
            edge_v = torch.empty(0, dtype=torch.long)
        return cls(left, right, mask, edge_u, edge_v)

    def matvec(self, assignment: torch.Tensor) -> torch.Tensor:
        """Compute ``A_acg @ vec(assignment)`` without forming a dense ACG."""

        original_shape = assignment.shape
        if assignment.shape[-2:] != self.shape:
            raise ValueError(f"expected trailing shape {self.shape}, got {assignment.shape[-2:]}")
        flat = assignment.reshape(-1, self.num_candidates)
        out = torch.zeros_like(flat)
        if self.num_edges:
            out.index_add_(1, self.edge_u, flat[:, self.edge_v])
            out.index_add_(1, self.edge_v, flat[:, self.edge_u])
        return out.reshape(original_shape)

    def objective(self, assignment: torch.Tensor) -> torch.Tensor:
        """Return ``vec(S)^T A_acg vec(S)`` (twice the hard edge count)."""

        flat = assignment.reshape(-1, self.num_candidates)
        if not self.num_edges:
            values = torch.zeros(flat.shape[0], device=flat.device, dtype=flat.dtype)
        else:
            values = 2.0 * (flat[:, self.edge_u] * flat[:, self.edge_v]).sum(dim=-1)
        return values.reshape(assignment.shape[:-2])

    def hard_statistics(self, mapping: torch.Tensor) -> tuple[int, int]:
        """Count preserved edges and their incident nodes for a row-to-column map."""

        mapping = mapping.detach().cpu().long()
        if mapping.numel() != self.left.num_nodes:
            raise ValueError("mapping has the wrong number of rows")
        selected = torch.zeros(self.num_candidates, dtype=torch.bool)
        valid_rows = (mapping >= 0) & (mapping < self.right.num_nodes)
        rows = torch.arange(self.left.num_nodes)[valid_rows]
        selected[rows * self.right.num_nodes + mapping[valid_rows]] = True
        edge_u, edge_v = self.edge_u.cpu(), self.edge_v.cpu()
        active = selected[edge_u] & selected[edge_v]
        edge_count = int(active.sum())
        if not edge_count:
            return 0, 0
        incident = torch.zeros(self.num_candidates, dtype=torch.bool)
        incident[edge_u[active]] = True
        incident[edge_v[active]] = True
        return edge_count, int(incident.sum())

