#!/usr/bin/env python3
"""Formal native ENPDA-Core/ENPDA-Solver evaluation shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.torch_version import TorchVersion

from nema.data import load_pairs, pair_paths
from nema.enpda_solver import ENPDASolver
from nema.models.enpda import ENPDAModel


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/enpda_sota.json"
CODE_PATHS = (
    ROOT / "src/nema/models/enpda.py",
    ROOT / "src/nema/enpda_solver.py",
    ROOT / "src/nema/association.py",
    ROOT / "src/nema/rounding.py",
    Path(__file__).resolve(),
)
_MODEL_CACHE: dict[tuple[str, str], ENPDAModel] = {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hardware_guard(config: dict) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("formal ENPDA native evaluation refuses CPU execution")
    accelerator = torch.cuda.get_device_name(0)
    if not any(name in accelerator for name in config["hardware"]["accelerator_any"]):
        raise RuntimeError(f"formal evaluation requires H100/H200, found {accelerator}")
    expected = str(config["hardware"]["cuda_runtime_prefix"])
    if not str(torch.version.cuda).startswith(expected):
        raise RuntimeError(f"formal evaluation requires CUDA {expected}, found {torch.version.cuda}")
    return {
        "device": accelerator,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_capability": list(torch.cuda.get_device_capability(0)),
    }


def _load_checkpoint(path: str | Path) -> dict:
    """Load the locally produced checkpoint under an explicit safe allowlist."""

    torch.serialization.add_safe_globals([TorchVersion])
    return torch.load(path, map_location="cpu", weights_only=True)


def _load_model(checkpoint_path: str, device: str) -> ENPDAModel:
    key = (checkpoint_path, device)
    if key not in _MODEL_CACHE:
        payload = _load_checkpoint(checkpoint_path)
        model = ENPDAModel(**payload["model_config"])
        model.load_state_dict(payload["model"], strict=True)
        _MODEL_CACHE[key] = model.to(device).eval()
    return _MODEL_CACHE[key]


def _solve(payload: dict) -> dict:
    pairs = load_pairs(payload["path"])
    if len(pairs) != 1:
        raise RuntimeError(f"native path must contain one pair: {payload['path']}")
    pair = pairs[0]
    model = _load_model(payload["checkpoint"], payload["device"])
    arm = payload["arm"]
    solver = ENPDASolver(
        model,
        trajectory="analytic" if arm == "analytic_core" else "learned",
        hard_search=arm == "solver",
        continuous_restarts=payload["continuous_restarts"],
        discrete_restarts=payload["discrete_restarts"],
        refinement_passes=payload["refinement_passes"],
        anneal_steps=payload["anneal_steps"],
        lns_steps=payload["lns_steps"],
        include_analytic_reference=True,
        noise_scale=payload["noise_scale"],
        seed=payload["search_seed"],
        device=payload["device"],
    )
    record = solver.solve(pair).to_dict()
    metadata = dict(record.get("metadata") or {})
    metadata.update(
        {
            "training_seed": payload["training_seed"],
            "search_seed": payload["search_seed"],
            "formal_arm": arm,
            "checkpoint": {
                "path": payload["checkpoint"],
                "sha256": payload["checkpoint_sha256"],
            },
        }
    )
    record["metadata"] = metadata
    record["source_path"] = payload["path"]
    return record


def _summary(records: list[dict]) -> dict:
    accuracy = [float(row["accuracy"]) for row in records if row["accuracy"] is not None]
    mse = [
        float(row["similarity_squared_error"])
        for row in records
        if row["similarity_squared_error"] is not None
    ]
    return {
        "pairs": len(records),
        "accuracy_percent": 100.0 * float(np.mean(accuracy)),
        "similarity_mse_x1e3": 1000.0 * float(np.mean(mse)),
        "mean_seconds_per_pair": float(np.mean([row["runtime_seconds"] for row in records])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["AIDS", "MOLHIV", "MCF-7"], required=True)
    parser.add_argument("--training-seed", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument("--arm", choices=["core", "analytic_core", "solver"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    hardware = _hardware_guard(config)
    checkpoint_payload = _load_checkpoint(args.checkpoint)
    if checkpoint_payload.get("training_seed") != args.training_seed:
        raise RuntimeError("checkpoint training seed does not match evaluation shard")
    if checkpoint_payload.get("config_sha256") != _sha256(CONFIG):
        raise RuntimeError("checkpoint was not trained under the frozen SOTA config")
    checkpoint_sha256 = _sha256(args.checkpoint)
    code_sha256 = {str(path.relative_to(ROOT)): _sha256(path) for path in CODE_PATHS}
    budget = config["native_test"]["solver"]
    start_payload = {
        "status": "started",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": config["protocol_version"],
        "config": str(CONFIG.relative_to(ROOT)),
        "config_sha256": _sha256(CONFIG),
        "dataset": args.dataset,
        "training_seed": args.training_seed,
        "arm": args.arm,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "hardware": hardware,
        "code_sha256": code_sha256,
        "output": str(args.output),
    }
    args.attestation.parent.mkdir(parents=True, exist_ok=True)
    args.attestation.write_text(
        json.dumps(start_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    paths = pair_paths("data/official", args.dataset, split="test", limit=100)
    completed: dict[str, dict] = {}
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                source = record["source_path"]
                if source in completed:
                    raise RuntimeError(f"duplicate resumable source {source}")
                completed[source] = record
    expected = {str(path) for path in paths}
    if set(completed) - expected:
        raise RuntimeError("output contains sources outside the frozen native shard")

    common = {
        "arm": args.arm,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "training_seed": args.training_seed,
        "search_seed": 0,
        "device": "cuda",
        "continuous_restarts": int(budget["learned_primary_streams"]),
        "discrete_restarts": int(budget["discrete_restarts_per_stream"]),
        "refinement_passes": int(budget["refinement_passes"]),
        "anneal_steps": int(budget["anneal_steps"]),
        "lns_steps": int(budget["lns_steps"]),
        "noise_scale": float(budget["noise_scale"]),
    }
    payloads = [{**common, "path": str(path)} for path in paths if str(path) not in completed]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    workers = 1 if args.arm != "solver" else args.workers
    with args.output.open("a", encoding="utf-8") as stream:
        if workers == 1:
            for index, payload in enumerate(payloads, 1):
                record = _solve(payload)
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                print(
                    f"[{index}/{len(payloads)}] {args.dataset} {args.arm} "
                    f"edges={record['common_edges']}/{record['true_edges']} "
                    f"time={record['runtime_seconds']:.3f}s",
                    flush=True,
                )
        else:
            context = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                futures = {pool.submit(_solve, payload): payload for payload in payloads}
                for index, future in enumerate(as_completed(futures), 1):
                    record = future.result()
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    print(
                        f"[{index}/{len(payloads)}] {args.dataset} {args.arm} "
                        f"edges={record['common_edges']}/{record['true_edges']} "
                        f"time={record['runtime_seconds']:.3f}s",
                        flush=True,
                    )

    records = [
        json.loads(line)
        for line in args.output.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != len(paths) or {row["source_path"] for row in records} != expected:
        raise RuntimeError("formal shard did not cover every released native path exactly once")
    if {str(path.relative_to(ROOT)): _sha256(path) for path in CODE_PATHS} != code_sha256:
        raise RuntimeError("formal ENPDA evaluation code changed during the shard")
    completed_payload = {
        **start_payload,
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary_all_released": _summary(records),
        "released_pairs": len(paths),
        "recovered_pairs": sum(
            bool((row.get("metadata") or {}).get("input_provenance")) for row in records
        ),
        "output_sha256": _sha256(args.output),
    }
    temporary = args.attestation.with_name(args.attestation.name + ".tmp")
    temporary.write_text(
        json.dumps(completed_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.attestation)
    print(json.dumps(completed_payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
