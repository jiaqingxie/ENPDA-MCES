#!/usr/bin/env python3
"""Resumable long-budget sparse lifted certificates on the frozen 75 pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import torch

from nema.association import AssociationGraph
from nema.certificate import solve_lifted_milp
from nema.data import load_pair


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/long_certificate.json"
CODE_PATHS = (
    REPO / "src/nema/association.py",
    REPO / "src/nema/certificate.py",
    Path(__file__).resolve(),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: dict) -> str:
    value = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(value).hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _hardware_guard(config: dict) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("formal long-certificate run refuses CPU-only jobs")
    device = torch.cuda.get_device_name(0)
    allowed = tuple(config["formal_hardware"]["accelerator_contains"])
    if not any(name in device for name in allowed):
        raise RuntimeError(f"formal run requires {allowed}, found {device}")
    expected_cuda = str(config["formal_hardware"]["cuda_runtime"])
    if torch.version.cuda != expected_cuda:
        raise RuntimeError(
            f"formal run requires CUDA {expected_cuda}, found {torch.version.cuda}"
        )
    return {
        "device": device,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_capability": list(torch.cuda.get_device_capability(0)),
    }


def _frozen_records(config: dict, dataset: str) -> list[dict]:
    source = REPO / config["source_certificate_records"]
    records = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [record for record in records if record["dataset"] == dataset]
    expected = int(config["pairs_per_dataset"])
    if len(selected) != expected:
        raise ValueError(f"expected {expected} frozen {dataset} pairs, found {len(selected)}")
    paths = [record["source_path"] for record in selected]
    if len(paths) != len(set(paths)):
        raise ValueError(f"duplicate frozen source paths for {dataset}")
    for record in selected:
        path = REPO / record["source_path"]
        if _sha256(path) != record["source_sha256"]:
            raise ValueError(f"frozen source hash changed: {path}")
    return sorted(selected, key=lambda item: int(item["official_position"]))


def _worker(payload: dict) -> dict:
    started = time.perf_counter()
    source = REPO / payload["source_path"]
    if _sha256(source) != payload["source_sha256"]:
        raise ValueError(f"source changed inside worker: {source}")
    pair = load_pair(source)
    left, right, _ = pair.oriented()
    association = AssociationGraph.build(left, right)
    result = solve_lifted_milp(
        association,
        time_limit=float(payload["budget_seconds"]),
        relative_gap=float(payload["relative_mip_gap"]),
    )
    frozen_lower = int(payload["frozen_lower_bound"])
    solver_lower = result.lower_bound if result.lower_bound is not None else 0
    combined_lower = max(frozen_lower, int(solver_lower))
    if result.upper_bound is not None and result.upper_bound + 1e-6 < combined_lower:
        raise RuntimeError(
            f"dual upper bound {result.upper_bound} is below legal lower {combined_lower}"
        )
    gap = (
        max(0.0, float(result.upper_bound) - combined_lower)
        / max(abs(float(result.upper_bound)), 1.0)
        if result.upper_bound is not None
        else None
    )
    return {
        "protocol_sha256": payload["protocol_sha256"],
        "dataset": payload["dataset"],
        "official_position": int(payload["official_position"]),
        "pair_key": pair.key,
        "source_path": payload["source_path"],
        "source_sha256": payload["source_sha256"],
        "budget_seconds": float(payload["budget_seconds"]),
        "frozen_lower_bound": frozen_lower,
        "solver_lower_bound": result.lower_bound,
        "combined_lower_bound": combined_lower,
        "upper_bound": result.upper_bound,
        "raw_upper_bound": result.raw_upper_bound,
        "upper_inflation": result.upper_inflation,
        "relative_gap": gap,
        "certified_optimal": gap is not None and gap <= 1e-7,
        "status": result.status,
        "message": result.message,
        "solver_runtime_seconds": result.runtime_seconds,
        "worker_runtime_seconds": time.perf_counter() - started,
        "variables": result.variables,
        "constraints": result.constraints,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("AIDS", "MOLHIV", "MCF-7"), required=True)
    parser.add_argument("--budget", type=int, choices=(10, 300, 1800), required=True)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    args = parser.parse_args()

    os.chdir(REPO)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if args.budget not in config["budgets_seconds"]:
        raise ValueError("budget is outside the frozen protocol")
    protocol_sha256 = _canonical_sha256(config)
    config_file_sha256 = _sha256(CONFIG)
    code_sha256 = {str(path.relative_to(REPO)): _sha256(path) for path in CODE_PATHS}
    hardware = _hardware_guard(config)
    frozen = _frozen_records(config, args.dataset)

    completed: dict[str, dict] = {}
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("protocol_sha256") != protocol_sha256:
                raise ValueError("existing output belongs to a different protocol")
            if record.get("dataset") != args.dataset or record.get("budget_seconds") != float(args.budget):
                raise ValueError("existing output belongs to a different shard")
            source = record["source_path"]
            if source in completed:
                raise ValueError(f"duplicate source path in output: {source}")
            completed[source] = record
    expected_paths = {record["source_path"] for record in frozen}
    if set(completed) - expected_paths:
        raise ValueError("existing output contains records outside the frozen sample")

    workers = args.workers or int(config["workers_per_shard"])
    payloads = [
        {
            "protocol_sha256": protocol_sha256,
            "dataset": args.dataset,
            "official_position": record["official_position"],
            "source_path": record["source_path"],
            "source_sha256": record["source_sha256"],
            "budget_seconds": args.budget,
            "relative_mip_gap": config["relative_mip_gap"],
            "frozen_lower_bound": int(record["nema_lower_bound"]),
        }
        for record in frozen
        if record["source_path"] not in completed
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as stream:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_worker, payload): payload for payload in payloads}
            for index, future in enumerate(as_completed(futures), 1):
                record = future.result()
                completed[record["source_path"]] = record
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                print(
                    f"[{args.dataset} {args.budget}s {index}/{len(payloads)}] "
                    f"lower={record['combined_lower_bound']} upper={record['upper_bound']} "
                    f"exact={record['certified_optimal']}",
                    flush=True,
                )

    if len(completed) != len(frozen):
        raise RuntimeError(f"incomplete shard: {len(completed)}/{len(frozen)}")
    if {str(path.relative_to(REPO)): _sha256(path) for path in CODE_PATHS} != code_sha256:
        raise RuntimeError("certificate implementation changed while the shard was running")
    exact = sum(bool(record["certified_optimal"]) for record in completed.values())
    payload = {
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": config["protocol_version"],
        "protocol_sha256": protocol_sha256,
        "config_file_sha256": config_file_sha256,
        "code_sha256": code_sha256,
        "dataset": args.dataset,
        "budget_seconds": args.budget,
        "workers": workers,
        "records": len(completed),
        "certified_optimal": exact,
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
        "hardware": hardware,
    }
    _atomic_json(args.attestation, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
