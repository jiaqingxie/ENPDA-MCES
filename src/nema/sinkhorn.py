"""Log-domain partial Sinkhorn projection."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SinkhornDiagnostics:
    iterations: int
    max_row_residual: float
    max_column_excess: float


def partial_assignment_residual(assignment: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return maximum row-equality residual and column-capacity excess.

    The function accepts a single matrix or a batch with assignment dimensions
    in the final two axes.  Column sums below one are feasible and therefore do
    not contribute to the second residual.
    """

    if assignment.ndim < 2:
        raise ValueError("assignment must have at least two dimensions")
    row_residual = (assignment.sum(dim=-1) - 1.0).abs().amax()
    column_excess = (assignment.sum(dim=-2) - 1.0).clamp_min(0.0).amax()
    return row_residual, column_excess


def partial_dummy_logits(assignment: torch.Tensor) -> torch.Tensor | None:
    """Return the symmetric dummy-row reference induced by column slack.

    Appending these rows embeds a feasible rectangular assignment in the
    square Birkhoff polytope.  Reusing this reference between mirror steps is
    what makes the update a genuine KL/Bregman projection rather than a fresh
    dummy-row reinitialization.
    """

    if assignment.ndim < 2:
        raise ValueError("assignment must have at least two dimensions")
    rows, cols = assignment.shape[-2:]
    if rows > cols:
        raise ValueError("partial assignment expects rows <= columns")
    if rows == cols:
        return None
    dummy_rows = cols - rows
    slack = (1.0 - assignment.sum(dim=-2)).clamp_min(torch.finfo(assignment.dtype).tiny)
    dummy = (slack / dummy_rows).unsqueeze(-2).expand(
        *assignment.shape[:-2], dummy_rows, cols
    )
    return dummy.log()


def log_sinkhorn(
    logits: torch.Tensor,
    mask: torch.Tensor | None = None,
    iterations: int = 20,
    invalid_logit: float = -30.0,
    tolerance: float | None = None,
    max_iterations: int | None = None,
    return_diagnostics: bool = False,
    dummy_logits: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, SinkhornDiagnostics]:
    """Project rectangular logits onto the injective-assignment relaxation.

    The row dimension must not exceed the column dimension. Dummy rows turn the
    problem square; after projection they are removed, leaving real row sums near
    one and real column sums at most one.
    """

    matrix_input = logits.ndim == 2
    if matrix_input:
        logits = logits.unsqueeze(0)
    if logits.ndim != 3:
        raise ValueError("logits must have shape [rows, cols] or [batch, rows, cols]")
    batch, rows, cols = logits.shape
    if rows > cols:
        raise ValueError("partial Sinkhorn expects rows <= columns")

    if mask is not None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.shape[0] == 1 and batch > 1:
            mask = mask.expand(batch, -1, -1)
        if mask.shape != logits.shape:
            raise ValueError("mask and logits shapes do not match")
        logits = logits.masked_fill(~mask, invalid_logit)

    if rows < cols:
        if dummy_logits is None:
            dummy = torch.zeros(
                (batch, cols - rows, cols), dtype=logits.dtype, device=logits.device
            )
        else:
            if dummy_logits.ndim == 2:
                dummy_logits = dummy_logits.unsqueeze(0)
            if dummy_logits.shape[0] == 1 and batch > 1:
                dummy_logits = dummy_logits.expand(batch, -1, -1)
            expected = (batch, cols - rows, cols)
            if dummy_logits.shape != expected:
                raise ValueError(
                    f"dummy_logits must have shape {expected}, got {dummy_logits.shape}"
                )
            dummy = dummy_logits.to(dtype=logits.dtype, device=logits.device)
        log_s = torch.cat((logits, dummy), dim=1)
    else:
        if dummy_logits is not None:
            raise ValueError("dummy_logits are invalid for a square assignment")
        log_s = logits

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if tolerance is not None and tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    total_iterations = iterations if tolerance is None else (max_iterations or max(iterations, 100))
    if total_iterations < iterations:
        raise ValueError("max_iterations must be at least iterations")

    finalized = False
    used_iterations = 0
    for used_iterations in range(1, total_iterations + 1):
        log_s = log_s - torch.logsumexp(log_s, dim=-1, keepdim=True)
        log_s = log_s - torch.logsumexp(log_s, dim=-2, keepdim=True)
        if tolerance is not None and used_iterations >= iterations:
            candidate = log_s - torch.logsumexp(log_s, dim=-1, keepdim=True)
            retained = candidate[:, :rows].exp()
            row_residual, column_excess = partial_assignment_residual(retained)
            if bool((torch.maximum(row_residual, column_excess) <= tolerance).detach()):
                log_s = candidate
                finalized = True
                break
    # Finish with a row normalization; adaptive mode additionally audits the
    # real-column capacity residual after this normalization.
    if not finalized:
        log_s = log_s - torch.logsumexp(log_s, dim=-1, keepdim=True)
    result = log_s[:, :rows].exp()
    row_residual, column_excess = partial_assignment_residual(result)
    output = result[0] if matrix_input else result
    if not return_diagnostics:
        return output
    diagnostics = SinkhornDiagnostics(
        iterations=used_iterations,
        max_row_residual=float(row_residual.detach()),
        max_column_excess=float(column_excess.detach()),
    )
    return output, diagnostics


def sample_gumbel_like(tensor: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    uniform = torch.rand(tensor.shape, dtype=tensor.dtype, device=tensor.device, generator=generator)
    uniform = uniform.clamp_(1e-8, 1.0 - 1e-8)
    return -torch.log(-torch.log(uniform))
