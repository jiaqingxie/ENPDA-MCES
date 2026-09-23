#!/usr/bin/env python3
"""Evaluate learned, analytic, and zero repaired-price certificates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.torch_version import TorchVersion

from nema.association import AssociationGraph
from nema.data import load_pair, pair_paths
from nema.enpda_certificate import incident_assignment_weights, repair_target_prices
from nema.models.enpda import ENPDAModel


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/enpda_price_source_ablation.json"
MOLHIV_RECOVERED = {"23", "46", "48", "54", "61", "64", "76"}
CODE = (
    ROOT / "src/nema/models/enpda.py",
    ROOT / "src/nema/enpda_certificate.py",
    ROOT / "src/nema/association.py",
    Path(__file__).resolve(),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def relative_gap(lower: int, upper: float) -> float:
    return max(0.0, upper - lower) / max(abs(upper), 1.0)


def safe_upper(value: float) -> float:
    return float(np.nextafter(value + 1e-10 * max(1.0, abs(value)), np.inf))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("AIDS", "MOLHIV", "MCF-7"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--solver-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--marker", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise RuntimeError("formal price-source ablation refuses CPU execution")
    accelerator = torch.cuda.get_device_name(0)
    if not any(name in accelerator for name in config["hardware"]["accelerator_any"]):
        raise RuntimeError(f"requires H100/H200, found {accelerator}")
    if not str(torch.version.cuda).startswith(config["hardware"]["cuda_runtime_prefix"]):
        raise RuntimeError(f"requires CUDA 12.8, found {torch.version.cuda}")

    torch.serialization.add_safe_globals([TorchVersion])
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if int(payload.get("training_seed", -1)) != args.seed:
        raise RuntimeError("checkpoint training seed mismatch")
    model = ENPDAModel(**payload["model_config"])
    model.load_state_dict(payload["model"], strict=True)
    model = model.cuda().eval()

    solver_rows = {row["source_path"]: row for row in read_jsonl(args.solver_records)}
    paths = pair_paths("data/official", args.dataset, split="test", limit=100)
    if args.dataset == "MOLHIV":
        paths = [path for path in paths if path.stem.rsplit("_", 1)[-1] not in MOLHIV_RECOVERED]
    expected = {str(path) for path in paths}
    if not expected <= set(solver_rows):
        raise RuntimeError("frozen Solver records do not cover the native certificate scope")

    config_hash = sha256(CONFIG)
    code_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in CODE}
    checkpoint_hash = sha256(args.checkpoint)
    solver_hash = sha256(args.solver_records)
    started = {
        "status": "started",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": config["protocol_version"],
        "config_sha256": config_hash,
        "dataset": args.dataset,
        "seed": args.seed,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "solver_records": str(args.solver_records),
        "solver_records_sha256": solver_hash,
        "hardware": {"device": accelerator, "torch": torch.__version__, "cuda_runtime": torch.version.cuda},
        "code_sha256": code_hashes,
    }
    args.marker.parent.mkdir(parents=True, exist_ok=True)
    args.marker.write_text(json.dumps(started, indent=2) + "\n", encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as stream:
        for index, path in enumerate(paths, 1):
            pair = load_pair(path)
            left, right, _ = pair.oriented()
            association = AssociationGraph.build(left, right)
            torch.cuda.synchronize()
            forward_started = time.perf_counter()
            with torch.inference_mode():
                learned = model(association, mode="learned")
                analytic = model(association, mode="analytic")
            torch.cuda.synchronize()
            forward_seconds = time.perf_counter() - forward_started
            weights = incident_assignment_weights(association)
            prices = {
                "learned": learned.prices,
                "analytic": analytic.prices,
                "zero": torch.zeros_like(analytic.prices),
            }
            lower = int(solver_rows[str(path)]["common_edges"])
            reference = int(solver_rows[str(path)]["true_edges"])
            sources = {}
            for source, vector in prices.items():
                certificate_started = time.perf_counter()
                raw, scale, row_potentials, target_potentials = repair_target_prices(
                    weights, association.candidate_mask, vector
                )
                upper = safe_upper(raw)
                elapsed = time.perf_counter() - certificate_started
                if upper + 1e-7 < max(lower, reference):
                    raise RuntimeError(f"{source} upper bound violates weak duality on {path}")
                sources[source] = {
                    "upper_bound": upper,
                    "relative_gap": relative_gap(lower, upper),
                    "certified_optimal": bool(upper < lower + 1.0 - 1e-7),
                    "best_scale": float(scale),
                    "row_potential_sum": float(row_potentials.sum()),
                    "target_potential_sum": float(target_potentials.sum()),
                    "certificate_seconds": elapsed,
                }
            row = {
                "dataset": args.dataset,
                "seed": args.seed,
                "key": pair.key,
                "source_path": str(path),
                "lower_bound": lower,
                "released_reference": reference,
                "forward_seconds": forward_seconds,
                "sources": sources,
            }
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            if index % 10 == 0 or index == len(paths):
                gaps = "/".join(f"{100*sources[name]['relative_gap']:.1f}" for name in ("learned", "analytic", "zero"))
                print(f"[{index}/{len(paths)}] {args.dataset} s{args.seed} gaps L/A/0={gaps}%", flush=True)

    if {str(path.relative_to(ROOT)): sha256(path) for path in CODE} != code_hashes:
        raise RuntimeError("price-source code changed during evaluation")
    completed = {
        **started,
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "records": len(paths),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
    }
    temporary = args.marker.with_name(args.marker.name + ".tmp")
    temporary.write_text(json.dumps(completed, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.marker)
    print(json.dumps(completed, indent=2), flush=True)


if __name__ == "__main__":
    main()
