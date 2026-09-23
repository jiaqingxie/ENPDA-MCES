"""Neural Equivariant Mirror Assignment."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from nema.association import AssociationGraph
from nema.features import structural_features
from nema.sinkhorn import log_sinkhorn, partial_assignment_residual, partial_dummy_logits


DEFAULT_INITIAL_WEIGHTS = (5.0, 1.5, 2.0, 1.0, 1.5, 1.0)
DEFAULT_INITIAL_BIAS = -1.0
DEFAULT_STEP_LOGIT = 1.85


class EquivariantMetric(nn.Module):
    """Positive row/column permutation-equivariant diagonal preconditioner.

    Pointwise features are augmented by row, column and global pooled states.
    All operations commute with independent permutations of the two input graphs.
    """

    def __init__(self, input_dim: int = 9, hidden_dim: int = 32, minimum: float = 0.05):
        super().__init__()
        self.minimum = minimum
        self.local = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.output = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Start from the ordinary mirror-ascent metric M=1.
        nn.init.zeros_(self.output[-1].weight)
        target = torch.tensor(1.0 - minimum)
        nn.init.constant_(self.output[-1].bias, float(torch.log(torch.expm1(target))))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        h = self.local(features)
        row = h.mean(dim=-2, keepdim=True).expand_as(h)
        col = h.mean(dim=-3, keepdim=True).expand_as(h)
        glob = h.mean(dim=(-3, -2), keepdim=True).expand_as(h)
        raw = self.output(torch.cat((h, row, col, glob), dim=-1)).squeeze(-1)
        return self.minimum + F.softplus(raw)


@dataclass
class NEMAOutput:
    assignment: torch.Tensor
    objectives: torch.Tensor
    accepted_steps: tuple[float, ...]
    backtracking_trials: tuple[int, ...]
    metric_min: float
    metric_max: float
    proposal_sources: tuple[str, ...]
    unit_residuals: tuple[float, ...]
    max_row_residual: float
    max_column_excess: float
    assignment_trace: tuple[torch.Tensor, ...] | None = None


class NEMAModel(nn.Module):
    """Shared neural preconditioner embedded in feasible entropic mirror ascent."""

    def __init__(
        self,
        steps: int = 8,
        hidden_dim: int = 32,
        sinkhorn_iterations: int = 20,
        max_backtracks: int = 32,
        armijo: float = 1e-4,
        safeguard_alignment: float = 0.1,
    ) -> None:
        super().__init__()
        self.steps = steps
        self.sinkhorn_iterations = sinkhorn_iterations
        self.max_backtracks = max_backtracks
        self.armijo = armijo
        if not 0.0 < safeguard_alignment <= 1.0:
            raise ValueError("safeguard_alignment must lie in (0, 1]")
        self.safeguard_alignment = safeguard_alignment
        self.initial_weights = nn.Parameter(torch.tensor(DEFAULT_INITIAL_WEIGHTS))
        self.initial_bias = nn.Parameter(torch.tensor(DEFAULT_INITIAL_BIAS))
        self.metric = EquivariantMetric(input_dim=11, hidden_dim=hidden_dim)
        # Softplus(1.85) is about 2; line search determines the accepted distance.
        self.step_logits = nn.Parameter(torch.full((steps,), DEFAULT_STEP_LOGIT))

    def initial_logits(
        self,
        fixed_features: torch.Tensor,
        initializer_mode: str = "learned",
    ) -> torch.Tensor:
        if initializer_mode == "learned":
            weights = self.initial_weights
            bias = self.initial_bias
        elif initializer_mode == "fixed":
            weights = fixed_features.new_tensor(DEFAULT_INITIAL_WEIGHTS)
            bias = fixed_features.new_tensor(DEFAULT_INITIAL_BIAS)
        else:
            raise ValueError(f"unknown initializer_mode {initializer_mode!r}")
        return (fixed_features * weights).sum(dim=-1) + bias

    @staticmethod
    def _dynamic_features(
        fixed: torch.Tensor,
        assignment: torch.Tensor,
        gradient: torch.Tensor,
    ) -> torch.Tensor:
        eps = torch.finfo(assignment.dtype).eps
        row_entropy = -(assignment * assignment.clamp_min(eps).log()).sum(dim=-1, keepdim=True)
        row_entropy = row_entropy / max(math.log(assignment.shape[-1]), 1.0)
        row_entropy = row_entropy.expand_as(assignment)
        col_entropy = -(assignment * assignment.clamp_min(eps).log()).sum(dim=-2, keepdim=True)
        col_entropy = col_entropy / max(math.log(assignment.shape[-2]), 1.0)
        col_entropy = col_entropy.expand_as(assignment)
        return torch.cat(
            (
                fixed,
                assignment.unsqueeze(-1),
                gradient.unsqueeze(-1),
                (assignment * gradient).unsqueeze(-1),
                row_entropy.unsqueeze(-1),
                col_entropy.unsqueeze(-1),
            ),
            dim=-1,
        )

    def forward(
        self,
        association: AssociationGraph,
        fixed_features: torch.Tensor | None = None,
        initial_noise: torch.Tensor | None = None,
        initializer_mode: str = "learned",
        schedule_mode: str = "learned",
        line_search: bool = True,
        metric_mode: str = "learned",
        stationary_safeguard: bool = True,
        sinkhorn_tolerance: float | None = None,
        sinkhorn_max_iterations: int | None = None,
        initial_sinkhorn_max_iterations: int | None = None,
        acceptance_tolerance: float = 0.0,
        return_trace: bool = False,
        iterations: int | None = None,
    ) -> NEMAOutput:
        if metric_mode not in {"learned", "unit"}:
            raise ValueError(f"unknown metric_mode {metric_mode!r}")
        if initializer_mode not in {"learned", "fixed"}:
            raise ValueError(f"unknown initializer_mode {initializer_mode!r}")
        if schedule_mode not in {"learned", "fixed"}:
            raise ValueError(f"unknown schedule_mode {schedule_mode!r}")
        device = self.initial_weights.device
        association = association.to(device)
        if fixed_features is None:
            fixed_features = structural_features(association).to(device)
        else:
            fixed_features = fixed_features.to(device)

        def project(
            project_logits: torch.Tensor,
            mask: torch.Tensor | None,
            dummy_logits: torch.Tensor | None = None,
            projection_max_iterations: int | None = None,
        ) -> tuple[torch.Tensor, bool]:
            projected = log_sinkhorn(
                project_logits,
                mask,
                iterations=self.sinkhorn_iterations,
                tolerance=sinkhorn_tolerance,
                max_iterations=(
                    sinkhorn_max_iterations
                    if projection_max_iterations is None
                    else projection_max_iterations
                ),
                return_diagnostics=sinkhorn_tolerance is not None,
                dummy_logits=dummy_logits,
            )
            if sinkhorn_tolerance is None:
                assert isinstance(projected, torch.Tensor)
                return projected, True
            proposal, diagnostics = projected
            feasible = max(
                diagnostics.max_row_residual,
                diagnostics.max_column_excess,
            ) <= sinkhorn_tolerance
            return proposal, feasible

        logits = self.initial_logits(fixed_features, initializer_mode=initializer_mode)
        if initial_noise is not None:
            logits = logits + initial_noise.to(device)
        assignment, initial_feasible = project(
            logits,
            # Compatibility is a structural feature, not a hard support cut:
            # full positive support gives a well-conditioned lifted KL map.
            mask=None,
            projection_max_iterations=initial_sinkhorn_max_iterations,
        )
        if not initial_feasible:
            raise RuntimeError(
                "partial Sinkhorn initializer did not reach the requested feasibility tolerance"
            )
        objectives = [association.objective(assignment)]
        assignment_trace = [assignment] if return_trace else None
        accepted_steps: list[float] = []
        backtracking_trials: list[int] = []
        proposal_sources: list[str] = []
        unit_residuals: list[float] = []
        metric_low, metric_high = float("inf"), 0.0

        total_steps = self.steps if iterations is None else iterations
        if total_steps < 0:
            raise ValueError("iterations must be nonnegative")
        for step in range(total_steps):
            gradient = 2.0 * association.matvec(assignment)
            valid_gradient = gradient[association.candidate_mask]
            scale = (
                valid_gradient.square().mean().sqrt().clamp_min(1e-6)
                if valid_gradient.numel()
                else gradient.new_tensor(1.0)
            )
            normalized_gradient = gradient / scale
            features = self._dynamic_features(fixed_features, assignment, normalized_gradient)
            if metric_mode == "learned":
                metric = self.metric(features).clamp(max=10.0)
            else:
                # Keep the initialization and learned step schedule fixed while
                # isolating the contribution of the equivariant metric M_theta.
                metric = torch.ones_like(assignment)
            metric_low = min(metric_low, float(metric.detach().min()))
            metric_high = max(metric_high, float(metric.detach().max()))

            # The learned schedule can be continued at inference time.  Cycling
            # preserves its positive bounded step family while allowing an
            # empirical long-horizon stationarity audit without changing any
            # trained weights.
            step_logit = (
                self.step_logits[step % self.steps]
                if schedule_mode == "learned"
                else assignment.new_tensor(DEFAULT_STEP_LOGIT)
            )
            maximum_step = F.softplus(step_logit)
            old_objective = objectives[-1]
            accepted = assignment
            accepted_objective = old_objective
            accepted_eta = 0.0
            accepted_source = "no_step"
            accepted_unit_residual = 0.0
            trials = self.max_backtracks if line_search else 1
            trials_used = 0
            for backtrack in range(trials):
                trials_used += 1
                eta = maximum_step * (0.5**backtrack)
                mirror_logits = assignment.clamp_min(1e-12).log()
                mirror_logits = mirror_logits + eta * metric * normalized_gradient
                proposal, proposal_feasible = project(
                    mirror_logits,
                    # The initializer suppresses label-incompatible padding,
                    # but subsequent KL steps must retain the full positive
                    # reference measure to be one consistent mirror map.
                    mask=None,
                    dummy_logits=partial_dummy_logits(assignment),
                )
                displacement = proposal - assignment
                direction = (gradient * displacement).sum()
                # For J(S)=vec(S)^T A vec(S), this expansion is exact:
                # J(S+D)=J(S)+<2AS,D>+J(D).  It gives the line search an
                # analytic monotonicity check without differentiating through
                # a boolean accept/reject decision.
                proposal_objective = (
                    old_objective + direction + association.objective(displacement)
                )
                threshold = old_objective + self.armijo * direction.clamp_min(0.0)
                # A finite Sinkhorn approximation is never eligible for
                # acceptance.  Feasibility is part of each proposal's own
                # acceptance test so failure of the learned projection does
                # not suppress the independent unit proposal at the same eta.
                proposal_passes = proposal_feasible and (
                    (not line_search)
                    or bool((proposal_objective >= threshold - acceptance_tolerance).detach())
                )

                if metric_mode == "learned" and stationary_safeguard:
                    unit_proposal, unit_feasible = project(
                        assignment.clamp_min(1e-12).log() + eta * normalized_gradient,
                        mask=None,
                        dummy_logits=partial_dummy_logits(assignment),
                    )
                    unit_displacement = unit_proposal - assignment
                    unit_direction = (gradient * unit_displacement).sum()
                    unit_objective = (
                        old_objective
                        + unit_direction
                        + association.objective(unit_displacement)
                    )
                    unit_threshold = (
                        old_objective + self.armijo * unit_direction.clamp_min(0.0)
                    )
                    unit_passes = unit_feasible and (
                        (not line_search)
                        or bool(
                            (unit_objective >= unit_threshold - acceptance_tolerance).detach()
                        )
                    )
                    learned_aligned = bool(
                        (
                            direction
                            >= self.safeguard_alignment * unit_direction.clamp_min(0.0)
                        ).detach()
                    )
                    trial_unit_residual = float(
                        (unit_displacement.norm() / eta.clamp_min(1e-12)).detach()
                    )
                    if proposal_passes and learned_aligned:
                        accepted = proposal
                        accepted_objective = proposal_objective
                        accepted_eta = float(eta.detach())
                        accepted_source = "learned"
                        accepted_unit_residual = trial_unit_residual
                        break
                    if unit_passes:
                        accepted = unit_proposal
                        accepted_objective = unit_objective
                        accepted_eta = float(eta.detach())
                        accepted_source = "unit_safeguard"
                        accepted_unit_residual = trial_unit_residual
                        break
                    accepted_unit_residual = trial_unit_residual
                    continue

                if proposal_passes:
                    accepted = proposal
                    accepted_objective = proposal_objective
                    accepted_eta = float(eta.detach())
                    accepted_source = "unit" if metric_mode == "unit" else "learned"
                    accepted_unit_residual = float(
                        (displacement.norm() / eta.clamp_min(1e-12)).detach()
                    )
                    break
            assignment = accepted
            objectives.append(accepted_objective)
            if assignment_trace is not None:
                assignment_trace.append(assignment)
            accepted_steps.append(accepted_eta)
            backtracking_trials.append(trials_used)
            proposal_sources.append(accepted_source)
            unit_residuals.append(accepted_unit_residual)

        if metric_low == float("inf"):
            metric_low = metric_high = 1.0
        row_residual, column_excess = partial_assignment_residual(assignment)
        return NEMAOutput(
            assignment=assignment,
            objectives=torch.stack(objectives),
            accepted_steps=tuple(accepted_steps),
            backtracking_trials=tuple(backtracking_trials),
            metric_min=metric_low,
            metric_max=metric_high,
            proposal_sources=tuple(proposal_sources),
            unit_residuals=tuple(unit_residuals),
            max_row_residual=float(row_residual.detach()),
            max_column_excess=float(column_excess.detach()),
            assignment_trace=tuple(assignment_trace) if assignment_trace is not None else None,
        )
