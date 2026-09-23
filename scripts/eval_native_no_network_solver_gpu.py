#!/usr/bin/env python3
"""Formal native strictly non-neural solver control.

The arm deliberately loads no checkpoint and selects fixed initialization,
fixed step sizes, and M=1.  It nevertheless receives the same three primary
streams, deterministic reference stream, and per-stream hard-search budget as
AEMA-full.  The redundant reference stream is retained because removing it
would silently reduce the comparator's candidate-evaluation budget.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from nema.association import AssociationGraph
from nema.benchmark import run_benchmark
from nema.data import load_pair, pair_paths
from nema.models.nema import NEMAModel


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = REPO_ROOT / "configs/native_no_network_solver.json"
CODE_PATHS = (
    REPO_ROOT / "src/nema/models/nema.py",
    REPO_ROOT / "src/nema/solvers.py",
    REPO_ROOT / "src/nema/benchmark.py",
    Path(__file__).resolve(),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hardware_guard() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("native no-network evaluation refuses CPU execution")
    device = torch.cuda.get_device_name(0)
    if not any(name in device for name in ("H100", "H200")):
        raise RuntimeError(f"formal evaluation requires H100/H200, found {device}")
    if torch.version.cuda != "12.8":
        raise RuntimeError(f"formal evaluation requires CUDA 12.8, found {torch.version.cuda}")
    return {
        "device": device,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_capability": list(torch.cuda.get_device_capability(0)),
    }


def parameter_invariance_check(dataset: str, sinkhorn_max_iterations: int) -> dict:
    """Prove at runtime that this arm is independent of neural parameters."""

    paths = pair_paths("data/official", dataset, split="test", limit=100)
    pair = load_pair(paths[0])
    left, right, _ = pair.oriented()
    association = AssociationGraph.build(left, right).to("cuda")

    torch.manual_seed(17)
    first = NEMAModel().to("cuda").eval()
    torch.manual_seed(991)
    second = NEMAModel().to("cuda").eval()
    with torch.no_grad():
        for parameter in second.parameters():
            parameter.copy_(torch.randn_like(parameter).mul_(100.0).add_(37.0))
        common = dict(
            initializer_mode="fixed",
            schedule_mode="fixed",
            metric_mode="unit",
            line_search=True,
            stationary_safeguard=True,
            sinkhorn_tolerance=1e-5,
            sinkhorn_max_iterations=sinkhorn_max_iterations,
            acceptance_tolerance=0.0,
        )
        output_a = first(association, **common)
        output_b = second(association, **common)
    assignment_error = float((output_a.assignment - output_b.assignment).abs().max().cpu())
    objective_error = max(
        abs(float(left_value) - float(right_value))
        for left_value, right_value in zip(output_a.objectives, output_b.objectives)
    )
    # Sparse CUDA reductions are not guaranteed bitwise reproducible across
    # two launches.  Judge independence at the same numerical scale as the
    # formal partial-Sinkhorn feasibility contract, rather than demanding
    # impossible bitwise equality from floating-point GPU kernels.
    numerical_tolerance = 2e-5
    if assignment_error > numerical_tolerance or objective_error > numerical_tolerance:
        raise RuntimeError(
            "fixed/fixed/unit control unexpectedly depends on trainable parameters: "
            f"assignment={assignment_error}, objective={objective_error}"
        )
    return {
        "source_path": str(paths[0]),
        "parameter_perturbation": "independent initialization followed by N(37,100^2) overwrite",
        "max_assignment_error": assignment_error,
        "max_objective_error": objective_error,
        "numerical_tolerance": numerical_tolerance,
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["AIDS", "MOLHIV", "MCF-7"], required=True)
    parser.add_argument("--search-seed", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    hardware = hardware_guard()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    sinkhorn_max_iterations = int(
        protocol["matched_solver_budget"]["sinkhorn_max_iterations"][args.dataset]
    )
    invariance = parameter_invariance_check(args.dataset, sinkhorn_max_iterations)
    code_sha256 = {str(path.relative_to(REPO_ROOT)): _sha256(path) for path in CODE_PATHS}
    protocol_sha256 = _sha256(PROTOCOL)

    args.attestation.parent.mkdir(parents=True, exist_ok=True)
    start_payload = {
        "status": "started",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": protocol_sha256,
        "dataset": args.dataset,
        "search_seed": args.search_seed,
        "checkpoint_loaded": False,
        "neural_parameter_invariance": invariance,
        "hardware": hardware,
        "code_sha256": code_sha256,
        "output": str(args.output),
    }
    args.attestation.write_text(
        json.dumps(start_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary = run_benchmark(
        data_root="data/official",
        dataset=args.dataset,
        method="nema",
        output=args.output,
        split="test",
        limit=args.limit,
        workers=args.workers,
        device="cuda",
        checkpoint=None,
        seed=args.search_seed,
        continuous_restarts=3,
        discrete_restarts=64,
        refinement_passes=30,
        anneal_steps=2500,
        lns_steps=250,
        certificate_seconds=0.0,
        nema_trajectory="unit",
        initializer_mode="fixed",
        schedule_mode="fixed",
        # Preserve AEMA-full's fourth deterministic reference stream exactly.
        unit_fallback=True,
        line_search=True,
        stationary_safeguard=True,
        sinkhorn_tolerance=1e-5,
        sinkhorn_max_iterations=sinkhorn_max_iterations,
        acceptance_tolerance=0.0,
    )

    if {str(path.relative_to(REPO_ROOT)): _sha256(path) for path in CODE_PATHS} != code_sha256:
        raise RuntimeError("formal solver code changed while this shard was running")
    completed_payload = {
        **start_payload,
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "output_sha256": _sha256(args.output),
        "invalid_manifest_sha256": _sha256(
            args.output.with_name(args.output.name + ".invalid.json")
        ),
    }
    temporary = args.attestation.with_name(args.attestation.name + ".tmp")
    temporary.write_text(
        json.dumps(completed_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.attestation)
    print(json.dumps(completed_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
