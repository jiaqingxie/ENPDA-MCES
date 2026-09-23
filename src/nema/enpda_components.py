"""Controlled component views of the frozen ENPDA Core.

The formal ENPDA implementation remains unchanged.  These views only select
an existing forward mode or set the analytic dual step to zero for the
``lambda == 0`` control.  Every arm therefore retains the same feature grid,
four auction rounds, row-simplex parameterization, and final Hungarian map.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from nema.association import AssociationGraph
from nema.models.enpda import ENPDAModel, ENPDAOutput


ARM_TO_MODE = {
    "no_price": "analytic",
    "full_no_price": "learned",
    "analytic": "analytic",
    "initializer_only": "initializer_only",
    "dynamics_only": "dynamics_only",
    "full": "learned",
}


@contextmanager
def _dual_step_override(model: ENPDAModel, value: float | None) -> Iterator[None]:
    original = model.base_dual_step
    if value is not None:
        model.base_dual_step = value
    try:
        yield
    finally:
        model.base_dual_step = original


def component_forward(
    model: ENPDAModel,
    association: AssociationGraph,
    arm: str,
) -> ENPDAOutput:
    """Run one preregistered component arm without changing its budget."""

    if arm not in ARM_TO_MODE:
        raise ValueError(f"unknown ENPDA component arm {arm!r}")
    # With zero initial prices, a zero dual step makes lambda_t identically
    # zero while leaving the real/unmatched row simplex and all logits intact.
    dual_step = 0.0 if arm in {"no_price", "full_no_price"} else None
    with _dual_step_override(model, dual_step):
        return model(association, mode=ARM_TO_MODE[arm])
