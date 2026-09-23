"""Paper-faithful per-instance Neural Graduated Assignment baseline."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from nema.association import AssociationGraph
from nema.graph import LabeledGraph
from nema.sinkhorn import log_sinkhorn, sample_gumbel_like


class MolecularMessagePassing(nn.Module):
    """A compact shared GCN used only to initialize one NGA graph pair."""

    def __init__(self, hidden_dim: int = 32, layers: int = 8):
        super().__init__()
        self.node_embedding = nn.Embedding(128, hidden_dim)
        self.edge_embedding = nn.Embedding(32, hidden_dim)
        self.self_layers = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(layers))
        self.message_layers = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(layers))

    def forward(self, graph: LabeledGraph, device: torch.device) -> torch.Tensor:
        node_labels = graph.node_labels.to(device).clamp(0, 127)
        edge_index = graph.edge_index.to(device)
        edge_labels = graph.edge_labels.to(device).clamp(0, 31)
        h = self.node_embedding(node_labels)
        for self_layer, message_layer, norm in zip(
            self.self_layers, self.message_layers, self.norms, strict=True
        ):
            messages = torch.zeros_like(h)
            if graph.num_edges:
                u, v = edge_index
                edge = self.edge_embedding(edge_labels)
                messages.index_add_(0, u, h[v] + edge)
                messages.index_add_(0, v, h[u] + edge)
            degree = graph.degrees.to(device).clamp_min(1.0).unsqueeze(-1)
            h = norm(F.relu(self_layer(h) + message_layer(messages / degree)))
        return h


class PerInstanceNGA(nn.Module):
    """NGA from Algorithms 1--2: each graph pair receives fresh parameters."""

    def __init__(
        self,
        association: AssociationGraph,
        hidden_dim: int = 32,
        encoder_layers: int = 8,
        steps: int = 4,
        samples: int = 10,
        sinkhorn_iterations: int = 20,
        variant: str = "acg",
    ) -> None:
        super().__init__()
        self.association_cpu = association
        self.encoder = MolecularMessagePassing(hidden_dim, encoder_layers)
        self.w1 = nn.Parameter(torch.empty(steps, hidden_dim))
        self.w2 = nn.Parameter(torch.empty(steps, hidden_dim))
        nn.init.normal_(self.w1, std=0.15)
        nn.init.normal_(self.w2, std=0.15)
        self.steps = steps
        if variant not in {"paper", "acg"}:
            raise ValueError("NGA variant must be 'paper' or 'acg'")
        self.variant = variant
        self.candidate_updates = nn.ModuleList(
            CandidateMessageBlock(hidden_dim=hidden_dim, layers=3) for _ in range(steps)
        )
        self.samples = samples
        self.sinkhorn_iterations = sinkhorn_iterations

    @property
    def temperatures(self) -> torch.Tensor:
        return (self.w1 * self.w2).sum(dim=-1)

    def forward(self, add_gumbel: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.w1.device
        association = self.association_cpu.to(device)
        left_h = self.encoder(association.left, device)
        right_h = self.encoder(association.right, device)
        logits = left_h @ right_h.t() / math.sqrt(left_h.shape[-1])
        logits = logits.unsqueeze(0).expand(self.samples, -1, -1)
        if add_gumbel:
            logits = logits + sample_gumbel_like(logits)
        assignment = log_sinkhorn(
            logits,
            association.candidate_mask,
            iterations=self.sinkhorn_iterations,
        )
        if self.variant == "paper":
            for beta in self.temperatures:
                compatibility = association.matvec(assignment)
                assignment = log_sinkhorn(
                    beta * compatibility,
                    association.candidate_mask,
                    iterations=self.sinkhorn_iterations,
                )
        else:
            for update in self.candidate_updates:
                update_logits = update(assignment, association)
                assignment = log_sinkhorn(
                    update_logits,
                    association.candidate_mask,
                    iterations=self.sinkhorn_iterations,
                )
        objective = association.objective(assignment)
        return assignment, objective


class CandidateMessageBlock(nn.Module):
    """Per-layer ACG message network used by the released NGA MCS2 path."""

    def __init__(self, hidden_dim: int = 32, layers: int = 3):
        super().__init__()
        dimensions = [1] + [hidden_dim] * layers
        self.self_layers = nn.ModuleList(
            nn.Linear(dimensions[i], dimensions[i + 1], bias=False) for i in range(layers)
        )
        self.neighbor_layers = nn.ModuleList(
            nn.Linear(dimensions[i], dimensions[i + 1], bias=False) for i in range(layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(layers - 1))
        self.output = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, assignment: torch.Tensor, association: AssociationGraph) -> torch.Tensor:
        batch = assignment.shape[0]
        h = assignment.reshape(batch, association.num_candidates, 1)
        states = []
        for layer, (self_layer, neighbor_layer) in enumerate(
            zip(self.self_layers, self.neighbor_layers, strict=True)
        ):
            aggregate = torch.zeros_like(h)
            if association.num_edges:
                aggregate.index_add_(1, association.edge_u, h[:, association.edge_v])
                aggregate.index_add_(1, association.edge_v, h[:, association.edge_u])
            h = self_layer(h) + neighbor_layer(aggregate)
            if layer < len(self.norms):
                h = self.norms[layer](torch.sigmoid(h))
            states.append(h)
        combined = torch.stack(states, dim=0).amax(dim=0)
        return self.output(combined).reshape_as(assignment)
