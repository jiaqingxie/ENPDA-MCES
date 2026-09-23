"""Sparse lifted ACG bounds and optional exact/anytime primal recovery."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix, csr_matrix

from nema.association import AssociationGraph


@dataclass(frozen=True)
class CertificateResult:
    mapping: torch.Tensor | None
    lower_bound: int | None
    upper_bound: float | None
    relative_gap: float | None
    status: int
    message: str
    mode: str
    runtime_seconds: float
    variables: int
    constraints: int
    raw_upper_bound: float | None
    upper_inflation: float | None


@dataclass(frozen=True)
class _LiftedFormulation:
    valid_grid: torch.Tensor
    edge_pairs: tuple[tuple[int, int], ...]
    objective: np.ndarray
    matrix: csr_matrix
    upper: np.ndarray
    x_variables: int

    @property
    def variables(self) -> int:
        return int(self.objective.size)

    @property
    def constraints(self) -> int:
        return int(self.upper.size)


def _lifted_formulation(association: AssociationGraph) -> _LiftedFormulation:
    """Build the sparse McCormick lifting in maximization sign convention.

    The solver minimizes ``objective``, hence every lifted edge variable has
    coefficient -1.  All constraints are represented as ``matrix @ z <= upper``.
    """

    rows, cols = association.shape
    valid_grid = torch.nonzero(association.candidate_mask.cpu(), as_tuple=False)
    grid_ids = valid_grid[:, 0] * cols + valid_grid[:, 1]
    x_lookup = {int(grid_id): index for index, grid_id in enumerate(grid_ids.tolist())}
    edge_pairs = tuple(
        (x_lookup[int(u)], x_lookup[int(v)])
        for u, v in zip(
            association.edge_u.tolist(), association.edge_v.tolist(), strict=True
        )
        if int(u) in x_lookup and int(v) in x_lookup
    )
    nx, ny = len(x_lookup), len(edge_pairs)
    variables = nx + ny

    matrix_row: list[int] = []
    matrix_col: list[int] = []
    matrix_value: list[float] = []
    upper: list[float] = []
    constraint = 0

    # One-to-one partial matching constraints.
    for row in range(rows):
        for index in torch.nonzero(valid_grid[:, 0] == row, as_tuple=True)[0].tolist():
            matrix_row.append(constraint)
            matrix_col.append(index)
            matrix_value.append(1.0)
        upper.append(1.0)
        constraint += 1
    for col in range(cols):
        for index in torch.nonzero(valid_grid[:, 1] == col, as_tuple=True)[0].tolist():
            matrix_row.append(constraint)
            matrix_col.append(index)
            matrix_value.append(1.0)
        upper.append(1.0)
        constraint += 1

    for edge_index, (left_x, right_x) in enumerate(edge_pairs):
        y_index = nx + edge_index
        # y_e <= x_u and y_e <= x_v.
        for endpoint in (left_x, right_x):
            matrix_row.extend((constraint, constraint))
            matrix_col.extend((y_index, endpoint))
            matrix_value.extend((1.0, -1.0))
            upper.append(0.0)
            constraint += 1
        # Full McCormick envelope: y_e >= x_u + x_v - 1.
        matrix_row.extend((constraint, constraint, constraint))
        matrix_col.extend((left_x, right_x, y_index))
        matrix_value.extend((1.0, 1.0, -1.0))
        upper.append(1.0)
        constraint += 1

    matrix = coo_matrix(
        (matrix_value, (matrix_row, matrix_col)), shape=(constraint, variables)
    ).tocsr()
    objective = np.zeros(variables, dtype=np.float64)
    objective[nx:] = -1.0
    return _LiftedFormulation(
        valid_grid=valid_grid,
        edge_pairs=edge_pairs,
        objective=objective,
        matrix=matrix,
        upper=np.asarray(upper, dtype=np.float64),
        x_variables=nx,
    )


def _safe_upper_bound(value: float | None) -> tuple[float | None, float | None]:
    """Round a floating solver bound outward, toward +infinity."""

    if value is None or not math.isfinite(value):
        return None, None
    inflated = value + 1e-8 * max(1.0, abs(value))
    safe = float(np.nextafter(inflated, np.inf))
    return safe, safe - value


def _relative_gap(lower: int | None, upper: float | None) -> float | None:
    if lower is None or upper is None:
        return None
    return max(0.0, float(upper - lower)) / max(abs(float(upper)), 1.0)


def solve_lifted_lp(
    association: AssociationGraph,
    time_limit: float | None = None,
) -> CertificateResult:
    """Solve the continuous sparse lifted LP relaxation.

    An optimal LP value is a valid upper bound on the integer MCES edge count.
    A time-limited, non-optimal LP solve deliberately returns no upper bound:
    SciPy's public ``linprog`` result does not expose a safe dual objective then.
    """

    started = time.perf_counter()
    formulation = _lifted_formulation(association)
    if not formulation.edge_pairs:
        return CertificateResult(
            mapping=None,
            lower_bound=0,
            upper_bound=0.0,
            relative_gap=0.0,
            status=0,
            message="empty ACG",
            mode="lp",
            runtime_seconds=time.perf_counter() - started,
            variables=formulation.variables,
            constraints=formulation.constraints,
            raw_upper_bound=0.0,
            upper_inflation=0.0,
        )
    options: dict[str, float | bool] = {"presolve": True}
    if time_limit is not None:
        options["time_limit"] = float(time_limit)
    result = linprog(
        c=formulation.objective,
        A_ub=formulation.matrix,
        b_ub=formulation.upper,
        bounds=(0.0, 1.0),
        method="highs",
        options=options,
    )
    raw_upper = -float(result.fun) if result.success else None
    upper_bound, upper_inflation = _safe_upper_bound(raw_upper)
    return CertificateResult(
        mapping=None,
        lower_bound=None,
        upper_bound=upper_bound,
        relative_gap=None,
        status=int(result.status),
        message=str(result.message),
        mode="lp",
        runtime_seconds=time.perf_counter() - started,
        variables=formulation.variables,
        constraints=formulation.constraints,
        raw_upper_bound=raw_upper,
        upper_inflation=upper_inflation,
    )


def solve_lifted_milp(
    association: AssociationGraph,
    time_limit: float = 10.0,
    relative_gap: float = 0.0,
) -> CertificateResult:
    """Solve the binary-x sparse lifting with HiGHS.

    The incumbent is a legal MCES lower bound.  HiGHS' branch-and-bound dual
    bound is rounded outward and reported as an anytime upper bound.
    """

    started = time.perf_counter()
    formulation = _lifted_formulation(association)
    rows, cols = association.shape
    nx = formulation.x_variables
    if not formulation.edge_pairs:
        return CertificateResult(
            mapping=None,
            lower_bound=0,
            upper_bound=0.0,
            relative_gap=0.0,
            status=0,
            message="empty ACG",
            mode="milp",
            runtime_seconds=time.perf_counter() - started,
            variables=formulation.variables,
            constraints=formulation.constraints,
            raw_upper_bound=0.0,
            upper_inflation=0.0,
        )

    integrality = np.zeros(formulation.variables, dtype=np.uint8)
    integrality[:nx] = 1
    result = milp(
        c=formulation.objective,
        integrality=integrality,
        bounds=Bounds(
            np.zeros(formulation.variables), np.ones(formulation.variables)
        ),
        constraints=LinearConstraint(
            formulation.matrix,
            np.full(formulation.constraints, -np.inf),
            formulation.upper,
        ),
        options={
            "time_limit": float(time_limit),
            "mip_rel_gap": float(relative_gap),
            "presolve": True,
        },
    )

    mapping = None
    lower_bound = None
    if result.x is not None:
        selected = np.flatnonzero(result.x[:nx] > 0.5)
        mapping = torch.full((rows,), -1, dtype=torch.long)
        used: set[int] = set()
        for x_index in selected:
            row, col = formulation.valid_grid[x_index].tolist()
            mapping[row] = col
            used.add(col)
        remaining = [col for col in range(cols) if col not in used]
        for row in torch.nonzero(mapping < 0, as_tuple=True)[0].tolist():
            mapping[row] = remaining.pop()
        lower_bound = association.hard_statistics(mapping)[0]

    dual = getattr(result, "mip_dual_bound", None)
    raw_upper = -float(dual) if dual is not None else None
    upper_bound, upper_inflation = _safe_upper_bound(raw_upper)
    return CertificateResult(
        mapping=mapping,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        relative_gap=_relative_gap(lower_bound, upper_bound),
        status=int(result.status),
        message=str(result.message),
        mode="milp",
        runtime_seconds=time.perf_counter() - started,
        variables=formulation.variables,
        constraints=formulation.constraints,
        raw_upper_bound=raw_upper,
        upper_inflation=upper_inflation,
    )
