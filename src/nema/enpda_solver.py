"""End-to-end Core and matched-budget Solver wrappers for ENPDA."""

from __future__ import annotations

import time

import torch

from nema.association import AssociationGraph
from nema.graph import GraphPair
from nema.models.enpda import ENPDAModel, ENPDAOutput
from nema.rounding import perturb_and_refine
from nema.sinkhorn import sample_gumbel_like
from nema.solvers import SolveResult, _finalize


class ENPDASolver:
    """Run an amortized neural primal--dual trajectory and optional hard search.

    ``hard_search=False`` is ENPDA-Core: one deterministic fixed-depth
    trajectory and one partial Hungarian projection. ``hard_search=True`` is
    ENPDA-Solver: three learned proposal streams and one analytic reference
    stream, each receiving the frozen discrete budget.
    """

    def __init__(
        self,
        model: ENPDAModel,
        *,
        trajectory: str = "learned",
        hard_search: bool = False,
        continuous_restarts: int = 3,
        discrete_restarts: int = 64,
        refinement_passes: int = 30,
        anneal_steps: int = 2500,
        lns_steps: int = 250,
        include_analytic_reference: bool = True,
        noise_scale: float = 0.5,
        seed: int = 0,
        device: str = "cuda",
    ) -> None:
        if trajectory not in {"learned", "analytic"}:
            raise ValueError(f"unknown ENPDA trajectory {trajectory!r}")
        if continuous_restarts < 1:
            raise ValueError("continuous_restarts must be positive")
        self.model = model.to(device).eval()
        self.trajectory = trajectory
        self.hard_search = hard_search
        self.continuous_restarts = continuous_restarts
        self.discrete_restarts = discrete_restarts
        self.refinement_passes = refinement_passes
        self.anneal_steps = anneal_steps
        self.lns_steps = lns_steps
        self.include_analytic_reference = include_analytic_reference
        self.noise_scale = noise_scale
        self.seed = seed
        self.device = device

    def _hard_candidate(
        self,
        association: AssociationGraph,
        output: ENPDAOutput,
        *,
        seed: int,
    ) -> torch.Tensor:
        if not self.hard_search:
            return output.hard_mapping().detach().cpu()
        return perturb_and_refine(
            association,
            output.assignment.detach().cpu(),
            restarts=self.discrete_restarts,
            max_passes=self.refinement_passes,
            anneal_steps=self.anneal_steps,
            lns_steps=self.lns_steps,
            seed=seed,
        )

    def solve(self, pair: GraphPair) -> SolveResult:
        started = time.perf_counter()
        left, right, swapped = pair.oriented()
        association = AssociationGraph.build(left, right)
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        best_mapping: torch.Tensor | None = None
        best_stats = (-1, -1)
        best_output: ENPDAOutput | None = None
        best_source = ""
        evaluated_sources: list[str] = []

        with torch.inference_mode():
            restarts = self.continuous_restarts if self.hard_search else 1
            for restart in range(restarts):
                noise = None
                if restart:
                    template = torch.empty(association.shape, device=self.device)
                    noise = self.noise_scale * sample_gumbel_like(template, generator)
                output = self.model(
                    association,
                    mode=self.trajectory,
                    initial_noise=noise,
                )
                source = f"{self.trajectory}_stream_{restart}"
                mapping = self._hard_candidate(
                    association,
                    output,
                    seed=self.seed + 1009 * restart,
                )
                stats = association.hard_statistics(mapping)
                evaluated_sources.append(source)
                if stats > best_stats:
                    best_mapping, best_stats = mapping, stats
                    best_output, best_source = output, source

            if (
                self.hard_search
                and self.trajectory == "learned"
                and self.include_analytic_reference
            ):
                output = self.model(association, mode="analytic")
                mapping = self._hard_candidate(association, output, seed=self.seed)
                stats = association.hard_statistics(mapping)
                source = "analytic_reference"
                evaluated_sources.append(source)
                if stats > best_stats:
                    best_mapping, best_stats = mapping, stats
                    best_output, best_source = output, source

        assert best_mapping is not None and best_output is not None
        return _finalize(
            "ENPDA-Solver" if self.hard_search else "ENPDA-Core",
            pair,
            association,
            best_mapping,
            started,
            soft_objective=float(best_output.objectives[-1]),
            metadata={
                "algorithm_version": "enpda-primal-dual-v1",
                "swapped": swapped,
                "trajectory": self.trajectory,
                "hard_search": self.hard_search,
                "fixed_depth_steps": self.model.steps,
                "continuous_restarts": restarts,
                "analytic_reference_streams": int(
                    self.hard_search
                    and self.trajectory == "learned"
                    and self.include_analytic_reference
                ),
                "evaluated_sources": evaluated_sources,
                "selected_source": best_source,
                "discrete_restarts": self.discrete_restarts if self.hard_search else 0,
                "constructive_restarts": self.discrete_restarts if self.hard_search else 0,
                "refinement_passes": self.refinement_passes if self.hard_search else 0,
                "anneal_steps": self.anneal_steps if self.hard_search else 0,
                "lns_steps": self.lns_steps if self.hard_search else 0,
                "noise_scale": self.noise_scale,
                "row_residual": best_output.row_residual,
                "max_column_excess": best_output.max_column_excess,
                "final_price_mean": float(best_output.prices.mean()),
                "final_price_max": float(best_output.prices.max()),
            },
        )
