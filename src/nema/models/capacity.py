"""Higher-capacity equivariant preconditioners for controlled capacity sweeps."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from nema.association import AssociationGraph
from nema.models.nema import NEMAModel


class BoundedPooledMetric(nn.Module):
    """The pooled equivariant architecture, anchored exactly at M=1."""

    def __init__(self, input_dim: int = 11, hidden_dim: int = 32):
        super().__init__()
        self.local = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.output = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        nn.init.zeros_(self.output[-1].weight); nn.init.zeros_(self.output[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        h = self.local(features)
        row = h.mean(dim=-2, keepdim=True).expand_as(h)
        col = h.mean(dim=-3, keepdim=True).expand_as(h)
        glob = h.mean(dim=(-3, -2), keepdim=True).expand_as(h)
        raw = self.output(torch.cat((h, row, col, glob), dim=-1)).squeeze(-1)
        return torch.exp(math.log(2.0) * torch.tanh(raw))


class BoundedACGMessageMetric(nn.Module):
    """Two-layer ACG message passing followed by row/column/global pooling.

    The sparse ACG and every pooling operation commute with independent node
    permutations, so this is an equivariant positive diagonal preconditioner.
    """

    def __init__(self, input_dim: int = 11, hidden_dim: int = 64, layers: int = 2):
        super().__init__()
        self.local = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU())
        self.messages = nn.ModuleList(
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(layers))
        self.output = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        nn.init.zeros_(self.output[-1].weight); nn.init.zeros_(self.output[-1].bias)
        self._association: AssociationGraph | None = None

    def bind(self, association: AssociationGraph | None) -> None:
        self._association = association

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        association = self._association
        if association is None:
            raise RuntimeError("ACG message metric must be bound to an AssociationGraph")
        h_grid = self.local(features)
        shape = h_grid.shape
        h = h_grid.reshape(association.num_candidates, -1)
        edge_u, edge_v = association.edge_u, association.edge_v
        for update, norm in zip(self.messages, self.norms, strict=True):
            aggregate = torch.zeros_like(h)
            degree = torch.zeros((h.shape[0], 1), dtype=h.dtype, device=h.device)
            if edge_u.numel():
                aggregate.index_add_(0, edge_u, h[edge_v])
                aggregate.index_add_(0, edge_v, h[edge_u])
                ones = torch.ones((edge_u.numel(), 1), dtype=h.dtype, device=h.device)
                degree.index_add_(0, edge_u, ones); degree.index_add_(0, edge_v, ones)
            aggregate = aggregate / degree.clamp_min(1.0)
            h = norm(h + update(torch.cat((h, aggregate), dim=-1)))
        h = h.reshape(shape)
        row = h.mean(dim=-2, keepdim=True).expand_as(h)
        col = h.mean(dim=-3, keepdim=True).expand_as(h)
        glob = h.mean(dim=(-3, -2), keepdim=True).expand_as(h)
        raw = self.output(torch.cat((h, row, col, glob), dim=-1)).squeeze(-1)
        return torch.exp(math.log(2.0) * torch.tanh(raw))


class CapacityNEMAModel(NEMAModel):
    """NEMA with an isolated capacity-controlled preconditioner replacement."""

    def __init__(self, architecture: str, hidden_dim: int, **kwargs):
        super().__init__(hidden_dim=32, **kwargs)
        if architecture == "pooled":
            self.metric = BoundedPooledMetric(hidden_dim=hidden_dim)
        elif architecture == "acg_gnn":
            self.metric = BoundedACGMessageMetric(hidden_dim=hidden_dim, layers=2)
        else:
            raise ValueError(f"unknown capacity architecture {architecture!r}")
        self.capacity_architecture = architecture
        self.capacity_hidden_dim = hidden_dim

    def forward(self, association: AssociationGraph, *args, **kwargs):
        if isinstance(self.metric, BoundedACGMessageMetric):
            self.metric.bind(association.to(self.initial_weights.device))
        try:
            return super().forward(association, *args, **kwargs)
        finally:
            if isinstance(self.metric, BoundedACGMessageMetric):
                self.metric.bind(None)


def load_capacity_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[CapacityNEMAModel, dict]:
    """Load a capacity-sweep checkpoint without consulting native test data.

    Capacity checkpoints serialize the complete model state together with the
    frozen architecture declaration.  Reconstructing from those two objects
    keeps deployment evaluations independent of the training script.
    """

    payload = torch.load(Path(path), map_location=device, weights_only=False)
    variant = payload.get("variant")
    if not isinstance(variant, dict):
        raise ValueError(f"capacity checkpoint has no variant declaration: {path}")
    state = payload.get("model")
    if not isinstance(state, dict) or "step_logits" not in state:
        raise ValueError(f"capacity checkpoint has no complete model state: {path}")
    model = CapacityNEMAModel(
        architecture=str(variant["architecture"]),
        hidden_dim=int(variant["hidden_dim"]),
        steps=int(state["step_logits"].numel()),
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, payload
