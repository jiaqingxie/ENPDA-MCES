"""Train-once equivariant Sinkhorn controls for MCES.

The controls deliberately share ENPDA's fixed structural features and sparse
ACG encoder, but contain no target prices, primal--dual recurrence, or hard
search.  A single learned affinity grid is projected once; the stochastic arm
only adds standard Gumbel perturbations before the same projection.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from nema.association import AssociationGraph
from nema.features import structural_features
from nema.models.enpda import SparseACGEncoder
from nema.models.nema import DEFAULT_INITIAL_BIAS, DEFAULT_INITIAL_WEIGHTS
from nema.rounding import hungarian_mapping
from nema.sinkhorn import log_sinkhorn, sample_gumbel_like


@dataclass
class SinkhornBaselineOutput:
    assignment: torch.Tensor
    dummy_assignment: torch.Tensor
    logits: torch.Tensor
    dummy_logits: torch.Tensor
    objective: torch.Tensor
    row_residual: float
    max_column_excess: float

    def hard_mapping(self) -> torch.Tensor:
        rows, columns = self.assignment.shape
        augmented = torch.cat((self.assignment, self.dummy_assignment), dim=1)
        mapping = hungarian_mapping(augmented)
        mapping[mapping >= columns] = -1
        return mapping


class TrainOnceSinkhorn(nn.Module):
    """One shared equivariant affinity model followed by one partial Sinkhorn."""

    def __init__(
        self,
        hidden_dim: int = 64,
        message_layers: int = 2,
        incompatibility_penalty: float = 8.0,
        initial_dummy_logit: float = -1.0,
        sinkhorn_iterations: int = 20,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if sinkhorn_iterations <= 0 or temperature <= 0:
            raise ValueError("Sinkhorn iterations and temperature must be positive")
        self.encoder = SparseACGEncoder(6, hidden_dim, message_layers)
        self.affinity_residual = nn.Linear(4 * hidden_dim, 1)
        self.dummy_residual = nn.Linear(4 * hidden_dim, 1)
        self.incompatibility_penalty = incompatibility_penalty
        self.initial_dummy_logit = initial_dummy_logit
        self.sinkhorn_iterations = sinkhorn_iterations
        self.temperature = temperature
        nn.init.zeros_(self.affinity_residual.weight)
        nn.init.zeros_(self.affinity_residual.bias)
        nn.init.zeros_(self.dummy_residual.weight)
        nn.init.zeros_(self.dummy_residual.bias)

    def affinity(
        self,
        association: AssociationGraph,
        fixed_features: torch.Tensor | None = None,
        *,
        learned: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, AssociationGraph]:
        device = next(self.parameters()).device
        association = association.to(device)
        fixed = (
            structural_features(association).to(device)
            if fixed_features is None
            else fixed_features.to(device)
        )
        weights = fixed.new_tensor(DEFAULT_INITIAL_WEIGHTS)
        logits = (fixed * weights).sum(dim=-1) + DEFAULT_INITIAL_BIAS
        logits = logits - (~association.candidate_mask).to(logits.dtype) * self.incompatibility_penalty
        dummy = logits.new_full((association.shape[0],), self.initial_dummy_logit)
        if learned:
            _, context = self.encoder(fixed, association)
            logits = logits + self.affinity_residual(context).squeeze(-1)
            dummy = dummy + self.dummy_residual(context).mean(dim=1).squeeze(-1)
        return logits, dummy, fixed, association

    def project(
        self,
        logits: torch.Tensor,
        dummy_logits: torch.Tensor,
        association: AssociationGraph,
        *,
        samples: int = 1,
        generator: torch.Generator | None = None,
        gumbel: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if samples <= 0:
            raise ValueError("samples must be positive")
        rows, columns = association.shape
        augmented = torch.cat(
            (logits, dummy_logits[:, None].expand(rows, rows)), dim=1
        ) / self.temperature
        batch = augmented.unsqueeze(0).expand(samples, -1, -1).clone()
        if gumbel:
            batch = batch + sample_gumbel_like(batch, generator)
        real_mask = association.candidate_mask
        mask = torch.cat(
            (real_mask, torch.ones((rows, rows), dtype=torch.bool, device=real_mask.device)),
            dim=1,
        )
        assignment = log_sinkhorn(
            batch,
            mask=mask,
            iterations=self.sinkhorn_iterations,
            invalid_logit=-30.0,
        )
        return assignment[:, :, :columns], assignment[:, :, columns:]

    def forward(
        self,
        association: AssociationGraph,
        fixed_features: torch.Tensor | None = None,
        *,
        learned: bool = True,
    ) -> SinkhornBaselineOutput:
        logits, dummy_logits, _, association = self.affinity(
            association, fixed_features, learned=learned
        )
        real, dummy = self.project(logits, dummy_logits, association)
        assignment, dummy_assignment = real[0], dummy[0]
        row_residual = (assignment.sum(1) + dummy_assignment.sum(1) - 1.0).abs().max()
        column_excess = torch.relu(assignment.sum(0) - 1.0).max()
        return SinkhornBaselineOutput(
            assignment=assignment,
            dummy_assignment=dummy_assignment,
            logits=logits,
            dummy_logits=dummy_logits,
            objective=association.objective(assignment),
            row_residual=float(row_residual.detach()),
            max_column_excess=float(column_excess.detach()),
        )

    @torch.inference_mode()
    def gumbel_mappings(
        self,
        association: AssociationGraph,
        fixed_features: torch.Tensor | None = None,
        *,
        samples: int = 10,
        generator: torch.Generator | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        logits, dummy_logits, _, association = self.affinity(
            association, fixed_features, learned=True
        )
        real, dummy = self.project(
            logits,
            dummy_logits,
            association,
            samples=samples,
            generator=generator,
            gumbel=True,
        )
        mappings = []
        columns = association.shape[1]
        for index in range(samples):
            mapping = hungarian_mapping(torch.cat((real[index], dummy[index]), dim=1))
            mapping[mapping >= columns] = -1
            mappings.append(mapping)
        return mappings, real
