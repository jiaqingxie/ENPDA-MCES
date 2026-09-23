#!/usr/bin/env python3
"""Evaluate ENPDA's repaired-price certificate on the native test pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.torch_version import TorchVersion

from nema.association import AssociationGraph
from nema.data import load_pair, pair_paths
from nema.enpda_certificate import certify_from_prices, label_histogram_upper_bound, repair_target_prices, incident_assignment_weights
from nema.models.enpda import ENPDAModel


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/enpda_sota.json"
MOLHIV_RECOVERED = {"23", "46", "48", "54", "61", "64", "76"}
CODE = (
    ROOT / "src/nema/models/enpda.py",
    ROOT / "src/nema/enpda_certificate.py",
    ROOT / "src/nema/association.py",
    Path(__file__).resolve(),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_checkpoint(path: Path) -> dict:
    torch.serialization.add_safe_globals([TorchVersion])
    return torch.load(path, map_location="cpu", weights_only=True)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def gap(lower: int, upper: float) -> float:
    return max(0.0, upper - lower) / max(abs(upper), 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("AIDS", "MOLHIV", "MCF-7"), required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/enpda_formal/seed0.best.pt"))
    parser.add_argument("--solver-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("formal price-certificate evaluation refuses CPU execution")
    accelerator = torch.cuda.get_device_name(0)
    if "H100" not in accelerator and "H200" not in accelerator:
        raise RuntimeError(f"requires H100/H200, found {accelerator}")
    if not str(torch.version.cuda).startswith("12.8"):
        raise RuntimeError(f"requires CUDA 12.8, found {torch.version.cuda}")
    payload = load_checkpoint(args.checkpoint)
    model = ENPDAModel(**payload["model_config"])
    model.load_state_dict(payload["model"], strict=True)
    model = model.cuda().eval()
    solver_rows = {row["source_path"]: row for row in read_jsonl(args.solver_records)}
    paths = pair_paths("data/official", args.dataset, split="test", limit=100)
    if args.dataset == "MOLHIV":
        paths = [path for path in paths if path.stem.rsplit("_", 1)[-1] not in MOLHIV_RECOVERED]
    certificate_paths = {str(path) for path in paths}
    if not certificate_paths <= set(solver_rows):
        missing = sorted(certificate_paths - set(solver_rows))
        raise RuntimeError(
            f"solver lower-bound records miss {len(missing)} certificate paths: {missing[:3]}"
        )
    code_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in CODE}
    started = {
        "status": "started",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "checkpoint_sha256": sha256(args.checkpoint),
        "solver_records_sha256": sha256(args.solver_records),
        "hardware": {"device": accelerator, "torch": torch.__version__, "cuda_runtime": torch.version.cuda},
        "code_sha256": code_hashes,
    }
    args.attestation.parent.mkdir(parents=True, exist_ok=True)
    args.attestation.write_text(json.dumps(started, indent=2) + "\n", encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    with args.output.open("w", encoding="utf-8") as stream:
        for index, path in enumerate(paths, 1):
            pair = load_pair(path)
            left, right, _ = pair.oriented()
            association = AssociationGraph.build(left, right)
            torch.cuda.synchronize()
            neural_started = time.perf_counter()
            with torch.inference_mode():
                output = model(association, mode="learned")
            torch.cuda.synchronize()
            neural_seconds = time.perf_counter() - neural_started

            # Preserve the repaired-price bound separately before taking the
            # minimum with deterministic graph bounds.
            weights = incident_assignment_weights(association)
            price_raw, scale, row_potential, target_potential = repair_target_prices(
                weights, association.candidate_mask, output.prices
            )
            certificate = certify_from_prices(association, output.prices)
            lower = int(solver_rows[str(path)]["common_edges"])
            reference = int(solver_rows[str(path)]["true_edges"])
            if certificate.upper_bound + 1e-7 < lower:
                raise RuntimeError(f"upper bound below legal incumbent on {path}")
            if certificate.upper_bound + 1e-7 < reference:
                raise RuntimeError(f"upper bound below released legal reference on {path}")
            row = {
                "dataset": args.dataset,
                "key": pair.key,
                "source_path": str(path),
                "lower_bound": lower,
                "released_reference": reference,
                "price_dual_upper_bound": float(np.nextafter(price_raw + 1e-10 * max(1.0, price_raw), np.inf)),
                "combined_instant_upper_bound": certificate.upper_bound,
                "combined_relative_gap": gap(lower, certificate.upper_bound),
                "certified_optimal": bool(certificate.upper_bound < lower + 1.0 - 1e-7),
                "best_price_scale": scale,
                "row_potential_sum": float(row_potential.sum()),
                "target_potential_sum": float(target_potential.sum()),
                "label_histogram_upper_bound": certificate.label_histogram_upper_bound,
                "edge_count_upper_bound": certificate.edge_count_upper_bound,
                "exact_incident_assignment_upper_bound": certificate.exact_linear_assignment_upper_bound,
                "neural_forward_seconds": neural_seconds,
                "certificate_seconds": certificate.runtime_seconds,
            }
            records.append(row)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            if index % 10 == 0 or index == len(paths):
                print(f"[{index}/{len(paths)}] {args.dataset} LB={lower} UB={certificate.upper_bound:.4f} cert={row['certified_optimal']}", flush=True)
    if {str(path.relative_to(ROOT)): sha256(path) for path in CODE} != code_hashes:
        raise RuntimeError("certificate code changed during formal evaluation")
    summary = {
        "pairs": len(records),
        "valid_upper_bounds": len(records),
        "certified_optimal": sum(row["certified_optimal"] for row in records),
        "certified_optimal_percent": 100.0 * float(np.mean([row["certified_optimal"] for row in records])),
        "median_combined_gap_percent": 100.0 * float(np.median([row["combined_relative_gap"] for row in records])),
        "median_certificate_milliseconds": 1000.0 * float(np.median([row["certificate_seconds"] for row in records])),
        "p95_certificate_milliseconds": 1000.0 * float(np.percentile([row["certificate_seconds"] for row in records], 95)),
        "median_neural_forward_milliseconds": 1000.0 * float(np.median([row["neural_forward_seconds"] for row in records])),
        "median_price_only_gap_percent": 100.0 * float(np.median([gap(row["lower_bound"], row["price_dual_upper_bound"]) for row in records])),
    }
    completed = {
        **started,
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "output_sha256": sha256(args.output),
    }
    args.attestation.write_text(json.dumps(completed, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(completed, indent=2), flush=True)


if __name__ == "__main__":
    main()
