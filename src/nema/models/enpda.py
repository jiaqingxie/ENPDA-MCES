"""Equivariant neural primal--dual assignment for MCES.

This module is deliberately independent of :mod:`nema.models.nema`.  ENPDA
learns a reusable fixed-depth auction dynamics rather than a diagonal metric
inside the existing mirror solver.  The row simplex is parameterized exactly
by a softmax with an unmatched state; target capacities are represented by
non-negative dual prices and the final discrete map is made injective by one
Hungarian projection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from nema.association import AssociationGraph
from nema.features import structural_features
from nema.models.nema import DEFAULT_INITIAL_BIAS, DEFAULT_INITIAL_WEIGHTS
from nema.rounding import hungarian_mapping


class SparseACGEncoder(nn.Module):
    """Permutation-equivariant message passing on the sparse ACG."""

    def __init__(self, input_dim: int, hidden_dim: int, layers: int) -> None:
        super().__init__()
        self.input = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU())
        self.messages = nn.ModuleList(
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(layers))

    def forward(
        self,
        features: torch.Tensor,
        association: AssociationGraph,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if features.shape[:2] != association.shape:
            raise ValueError(
                f"expected feature grid {association.shape}, got {tuple(features.shape[:2])}"
            )
        rows, columns = association.shape
        h = self.input(features).reshape(rows * columns, -1)
        edge_u, edge_v = association.edge_u, association.edge_v
        for update, norm in zip(self.messages, self.norms, strict=True):
            aggregate = torch.zeros_like(h)
            degree = torch.zeros((h.shape[0], 1), dtype=h.dtype, device=h.device)
            if edge_u.numel():
                aggregate.index_add_(0, edge_u, h[edge_v])
                aggregate.index_add_(0, edge_v, h[edge_u])
                ones = torch.ones((edge_u.numel(), 1), dtype=h.dtype, device=h.device)
                degree.index_add_(0, edge_u, ones)
                degree.index_add_(0, edge_v, ones)
            aggregate = aggregate / degree.clamp_min(1.0)
            h = norm(h + update(torch.cat((h, aggregate), dim=-1)))

        local = h.reshape(rows, columns, -1)
        row = local.mean(dim=1, keepdim=True).expand_as(local)
        column = local.mean(dim=0, keepdim=True).expand_as(local)
        global_state = local.mean(dim=(0, 1), keepdim=True).expand_as(local)
        context = torch.cat((local, row, column, global_state), dim=-1)
        return local, context


@dataclass
class ENPDAOutput:
    """A fixed-depth primal--dual trajectory and its final soft state."""

    assignment: torch.Tensor
    dummy_assignment: torch.Tensor
    logits: torch.Tensor
    dummy_logits: torch.Tensor
    prices: torch.Tensor
    objectives: torch.Tensor
    column_excesses: torch.Tensor
    row_residual: float
    max_column_excess: float
    assignment_trace: tuple[torch.Tensor, ...] | None = None
    dummy_trace: tuple[torch.Tensor, ...] | None = None

    def hard_mapping(self) -> torch.Tensor:
        """Return an injective partial map; ``-1`` denotes unmatched rows."""

        rows, columns = self.assignment.shape
        # Duplicating each row's unmatched score across ``rows`` dummy columns
        # lets every source remain unmatched while preserving one-to-one real
        # target assignments in the single Hungarian projection.
        augmented = torch.cat(
            (self.logits, self.dummy_logits[:, None].expand(rows, rows)), dim=1
        )
        mapping = hungarian_mapping(augmented)
        mapping[mapping >= columns] = -1
        return mapping


class ENPDAModel(nn.Module):
    """Shared equivariant neural auction dynamics for partial assignment.

    ``analytic`` mode removes all neural residuals while retaining the same
    primal--dual state parameterization.  ``initializer_only`` and
    ``dynamics_only`` provide causal controls without changing the number of
    auction rounds or the final Hungarian projection.
    """

    MODES = {"learned", "analytic", "initializer_only", "dynamics_only"}

    def __init__(
        self,
        steps: int = 4,
        hidden_dim: int = 64,
        message_layers: int = 2,
        base_primal_step: float = 0.5,
        base_dual_step: float = 0.5,
        incompatibility_penalty: float = 8.0,
        initial_dummy_logit: float = -1.0,
    ) -> None:
        super().__init__()
        if steps < 0:
            raise ValueError("steps must be non-negative")
        if base_primal_step <= 0 or base_dual_step <= 0:
            raise ValueError("primal and dual steps must be positive")
        self.steps = steps
        self.base_primal_step = base_primal_step
        self.base_dual_step = base_dual_step
        self.incompatibility_penalty = incompatibility_penalty
        self.initial_dummy_logit = initial_dummy_logit

        self.initial_encoder = SparseACGEncoder(6, hidden_dim, message_layers)
        self.initial_residual = nn.Linear(4 * hidden_dim, 1)
        self.initial_dummy_residual = nn.Linear(4 * hidden_dim, 1)

        # fixed(6), primal mass, normalized structural marginal, target price,
        # column demand, and row entropy = 11 dynamic scalar features.
        self.dynamic_encoder = SparseACGEncoder(11, hidden_dim, message_layers)
        self.correction_head = nn.Linear(4 * hidden_dim, 1)
        self.primal_step_head = nn.Linear(4 * hidden_dim, 1)
        self.dual_step_head = nn.Linear(4 * hidden_dim, 1)
        self.dummy_update_head = nn.Linear(4 * hidden_dim, 1)

        # ENPDA starts exactly at the analytic auction.  Training must earn
        # every neural deviation from that reference dynamics.
        for layer in (
            self.initial_residual,
            self.initial_dummy_residual,
            self.correction_head,
            self.primal_step_head,
            self.dual_step_head,
            self.dummy_update_head,
        ):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    @staticmethod
    def _row_softmax(
        logits: torch.Tensor,
        dummy_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probabilities = torch.softmax(torch.cat((logits, dummy_logits[:, None]), dim=1), dim=1)
        return probabilities[:, :-1], probabilities[:, -1]

    @staticmethod
    def _bounded_multiplier(raw: torch.Tensor) -> torch.Tensor:
        # A factor in [1/2, 2], exactly one at neural initialization.
        return torch.exp(math.log(2.0) * torch.tanh(raw))

    @staticmethod
    def _dynamic_features(
        fixed: torch.Tensor,
        assignment: torch.Tensor,
        dummy_assignment: torch.Tensor,
        gradient: torch.Tensor,
        prices: torch.Tensor,
    ) -> torch.Tensor:
        rows, columns = assignment.shape
        demand = assignment.sum(dim=0) - 1.0
        entropy_terms = torch.cat((assignment, dummy_assignment[:, None]), dim=1)
        row_entropy = -(
            entropy_terms * entropy_terms.clamp_min(1e-12).log()
        ).sum(dim=1)
        row_entropy = row_entropy / max(math.log(columns + 1), 1.0)
        return torch.cat(
            (
                fixed,
                assignment.unsqueeze(-1),
                gradient.unsqueeze(-1),
                prices[None, :, None].expand(rows, -1, -1),
                demand[None, :, None].expand(rows, -1, -1),
                row_entropy[:, None, None].expand(-1, columns, -1),
            ),
            dim=-1,
        )

    def forward(
        self,
        association: AssociationGraph,
        fixed_features: torch.Tensor | None = None,
        mode: str = "learned",
        iterations: int | None = None,
        initial_noise: torch.Tensor | None = None,
        return_trace: bool = False,
    ) -> ENPDAOutput:
        if mode not in self.MODES:
            raise ValueError(f"unknown ENPDA mode {mode!r}")
        device = next(self.parameters()).device
        association = association.to(device)
        fixed = (
            structural_features(association).to(device)
            if fixed_features is None
            else fixed_features.to(device)
        )
        if fixed.shape != (*association.shape, 6):
            raise ValueError(
                f"expected six fixed features on {association.shape}, got {tuple(fixed.shape)}"
            )

        learned_initializer = mode in {"learned", "initializer_only"}
        learned_dynamics = mode in {"learned", "dynamics_only"}
        fixed_weights = fixed.new_tensor(DEFAULT_INITIAL_WEIGHTS)
        logits = (fixed * fixed_weights).sum(dim=-1) + DEFAULT_INITIAL_BIAS
        logits = logits - (~association.candidate_mask).to(logits.dtype) * self.incompatibility_penalty
        if initial_noise is not None:
            if initial_noise.shape != logits.shape:
                raise ValueError(
                    f"expected initial noise on {tuple(logits.shape)}, "
                    f"got {tuple(initial_noise.shape)}"
                )
            logits = logits + initial_noise.to(device=device, dtype=logits.dtype)
        dummy_logits = logits.new_full((association.shape[0],), self.initial_dummy_logit)
        if learned_initializer:
            _, initial_context = self.initial_encoder(fixed, association)
            logits = logits + self.initial_residual(initial_context).squeeze(-1)
            dummy_logits = dummy_logits + self.initial_dummy_residual(initial_context).mean(dim=1).squeeze(-1)

        prices = logits.new_zeros(association.shape[1])
        assignment, dummy_assignment = self._row_softmax(logits, dummy_logits)
        objectives = [association.objective(assignment)]
        excesses = [torch.relu(assignment.sum(dim=0) - 1.0).max()]
        assignment_trace = [assignment] if return_trace else None
        dummy_trace = [dummy_assignment] if return_trace else None

        total_steps = self.steps if iterations is None else iterations
        if total_steps < 0:
            raise ValueError("iterations must be non-negative")
        for _ in range(total_steps):
            gradient = 2.0 * association.matvec(assignment)
            valid = gradient[association.candidate_mask]
            scale = (
                valid.square().mean().sqrt().clamp_min(1e-6)
                if valid.numel()
                else gradient.new_tensor(1.0)
            )
            gradient = gradient / scale
            demand = assignment.sum(dim=0) - 1.0

            if learned_dynamics:
                dynamic = self._dynamic_features(
                    fixed, assignment, dummy_assignment, gradient, prices
                )
                _, context = self.dynamic_encoder(dynamic, association)
                correction = torch.tanh(self.correction_head(context).squeeze(-1))
                primal_step = self.base_primal_step * self._bounded_multiplier(
                    self.primal_step_head(context).squeeze(-1)
                )
                # Mean over source rows preserves target permutation equivariance.
                dual_raw = self.dual_step_head(context).squeeze(-1).mean(dim=0)
                dual_step = self.base_dual_step * self._bounded_multiplier(dual_raw)
                dummy_update = torch.tanh(
                    self.dummy_update_head(context).squeeze(-1).mean(dim=1)
                )
            else:
                correction = torch.zeros_like(gradient)
                primal_step = gradient.new_full(gradient.shape, self.base_primal_step)
                dual_step = demand.new_full(demand.shape, self.base_dual_step)
                dummy_update = dummy_logits.new_zeros(dummy_logits.shape)

            prices = torch.relu(prices + dual_step * demand).clamp_max(20.0)
            logits = logits + primal_step * (gradient + correction - prices[None, :])
            # Keep incompatible real bids unattractive throughout the auction.
            logits = logits - (~association.candidate_mask).to(logits.dtype) * 0.1
            dummy_logits = dummy_logits + self.base_primal_step * dummy_update
            assignment, dummy_assignment = self._row_softmax(logits, dummy_logits)
            objectives.append(association.objective(assignment))
            excesses.append(torch.relu(assignment.sum(dim=0) - 1.0).max())
            if assignment_trace is not None:
                assignment_trace.append(assignment)
                assert dummy_trace is not None
                dummy_trace.append(dummy_assignment)

        row_residual = (assignment.sum(dim=1) + dummy_assignment - 1.0).abs().max()
        column_excess = torch.relu(assignment.sum(dim=0) - 1.0).max()
        return ENPDAOutput(
            assignment=assignment,
            dummy_assignment=dummy_assignment,
            logits=logits,
            dummy_logits=dummy_logits,
            prices=prices,
            objectives=torch.stack(objectives),
            column_excesses=torch.stack(excesses),
            row_residual=float(row_residual.detach()),
            max_column_excess=float(column_excess.detach()),
            assignment_trace=tuple(assignment_trace) if assignment_trace is not None else None,
            dummy_trace=tuple(dummy_trace) if dummy_trace is not None else None,
        )
