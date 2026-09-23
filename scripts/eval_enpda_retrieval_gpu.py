#!/usr/bin/env python3
"""Formal frozen-checkpoint ENPDA-Fast evaluation on a 20x500 pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.torch_version import TorchVersion

from nema.data import load_pair, pair_paths
from nema.enpda_solver import ENPDASolver
from nema.models.enpda import ENPDAModel


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/enpda_sota.json"
CODE = (
    ROOT / "src/nema/models/enpda.py",
    ROOT / "src/nema/enpda_solver.py",
    ROOT / "src/nema/association.py",
    ROOT / "src/nema/rounding.py",
    Path(__file__).resolve(),
)
_CACHE: dict[tuple[str, str], ENPDAModel] = {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint(path: str | Path) -> dict:
    torch.serialization.add_safe_globals([TorchVersion])
    return torch.load(path, map_location="cpu", weights_only=True)


def model(path: str, device: str) -> ENPDAModel:
    key = (path, device)
    if key not in _CACHE:
        payload = checkpoint(path)
        instance = ENPDAModel(**payload["model_config"])
        instance.load_state_dict(payload["model"], strict=True)
        _CACHE[key] = instance.to(device).eval()
    return _CACHE[key]


def hardware_guard() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("formal ENPDA retrieval refuses CPU execution")
    device = torch.cuda.get_device_name(0)
    if "H100" not in device and "H200" not in device:
        raise RuntimeError(f"formal retrieval requires H100/H200, found {device}")
    if not str(torch.version.cuda).startswith("12.8"):
        raise RuntimeError(f"formal retrieval requires CUDA 12.8, found {torch.version.cuda}")
    return {"device": device, "torch": torch.__version__, "cuda_runtime": torch.version.cuda}


def solve(payload: dict) -> dict:
    torch.set_num_threads(1)
    pair = load_pair(payload["path"])
    solver = ENPDASolver(
        model(payload["checkpoint"], "cuda"),
        trajectory="learned",
        hard_search=True,
        continuous_restarts=1,
        discrete_restarts=4,
        refinement_passes=10,
        anneal_steps=0,
        lns_steps=0,
        include_analytic_reference=False,
        noise_scale=0.5,
        seed=payload["protocol_seed"],
        device="cuda",
    )
    record = solver.solve(pair).to_dict()
    record["method"] = "ENPDA-Fast"
    record["source_path"] = payload["path"]
    metadata = dict(record.get("metadata") or {})
    metadata.update(
        {
            "protocol": "20-query x 500-candidate RASCAL-labeled hard retrieval",
            "training_seed": payload["training_seed"],
            "protocol_seed": payload["protocol_seed"],
            "checkpoint_sha256": payload["checkpoint_sha256"],
            "fast_budget": {
                "learned_streams": 1,
                "analytic_reference_streams": 0,
                "discrete_restarts": 4,
                "refinement_passes": 10,
                "anneal_steps": 0,
                "lns_steps": 0,
            },
        }
    )
    record["metadata"] = metadata
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("MOLHIV", "MCF-7"), required=True)
    parser.add_argument("--protocol-seed", type=int, choices=(20260823, 20260824, 20260825), required=True)
    parser.add_argument("--training-seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    ckpt = checkpoint(args.checkpoint)
    if int(ckpt["training_seed"]) != args.training_seed:
        raise RuntimeError("training seed/checkpoint mismatch")
    if ckpt["config_sha256"] != sha256(CONFIG):
        raise RuntimeError("checkpoint/config hash mismatch")
    hardware = hardware_guard()
    checkpoint_sha = sha256(args.checkpoint)
    code_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in CODE}
    data_root = ROOT / f"data/rascal-hard/500-way/seed-{args.protocol_seed}"
    paths = pair_paths(data_root, args.dataset, retrieval=True)
    if len(paths) != 10_000:
        raise RuntimeError(f"expected 10,000 retrieval pairs, found {len(paths)}")
    started = {
        "status": "started",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "protocol_seed": args.protocol_seed,
        "training_seed": args.training_seed,
        "checkpoint_sha256": checkpoint_sha,
        "hardware": hardware,
        "code_sha256": code_hashes,
        "pair_count": len(paths),
    }
    args.attestation.parent.mkdir(parents=True, exist_ok=True)
    args.attestation.write_text(json.dumps(started, indent=2) + "\n", encoding="utf-8")

    done: dict[str, dict] = {}
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line:
                row = json.loads(line)
                if row["source_path"] in done:
                    raise RuntimeError("duplicate resumable retrieval source")
                done[row["source_path"]] = row
    expected = {str(path) for path in paths}
    if set(done) - expected:
        raise RuntimeError("retrieval output contains an out-of-pool source")
    common = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "training_seed": args.training_seed,
        "protocol_seed": args.protocol_seed,
    }
    payloads = [{**common, "path": str(path)} for path in paths if str(path) not in done]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    with args.output.open("a", encoding="utf-8") as stream:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            futures = {pool.submit(solve, item): item for item in payloads}
            for index, future in enumerate(as_completed(futures), 1):
                row = future.result()
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                if index % 50 == 0 or index == len(payloads):
                    print(
                        f"[{index}/{len(payloads)}] {args.dataset}/seed-{args.protocol_seed} "
                        f"edges={row['common_edges']}/{row['true_edges']} time={row['runtime_seconds']:.3f}s",
                        flush=True,
                    )
    rows = [json.loads(line) for line in args.output.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != len(paths) or {row["source_path"] for row in rows} != expected:
        raise RuntimeError("formal retrieval shard is incomplete")
    if {str(path.relative_to(ROOT)): sha256(path) for path in CODE} != code_hashes:
        raise RuntimeError("retrieval code changed while the shard was running")
    completed = {
        **started,
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_sha256": sha256(args.output),
        "mean_seconds_per_pair": sum(float(row["runtime_seconds"]) for row in rows) / len(rows),
    }
    temporary = args.attestation.with_suffix(args.attestation.suffix + ".tmp")
    temporary.write_text(json.dumps(completed, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.attestation)
    print(json.dumps(completed, indent=2), flush=True)


if __name__ == "__main__":
    main()
