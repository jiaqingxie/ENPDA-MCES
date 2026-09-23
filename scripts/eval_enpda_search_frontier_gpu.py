#!/usr/bin/env python3
"""Evaluate one matched learned/analytic post-Core search frontier shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.torch_version import TorchVersion

from nema.association import AssociationGraph
from nema.data import load_pairs, pair_paths
from nema.enpda_anytime import search_trace
from nema.models.enpda import ENPDAModel


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/enpda_search_frontier.json"
CODE_PATHS = (
    ROOT / "src/nema/models/enpda.py",
    ROOT / "src/nema/enpda_anytime.py",
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


def _load_model(path: str, device: str) -> ENPDAModel:
    key = (path, device)
    if key not in _MODEL_CACHE:
        torch.serialization.add_safe_globals([TorchVersion])
        payload = torch.load(path, map_location="cpu", weights_only=True)
        model = ENPDAModel(**payload["model_config"])
        model.load_state_dict(payload["model"], strict=True)
        _MODEL_CACHE[key] = model.to(device).eval()
    return _MODEL_CACHE[key]


def _restore_orientation(mapping: torch.Tensor, pair, swapped: bool) -> torch.Tensor:
    if not swapped:
        return mapping.detach().cpu().long()
    inverse = torch.full((pair.left.num_nodes,), -1, dtype=torch.long)
    for oriented_left, oriented_right in enumerate(mapping.tolist()):
        if 0 <= oriented_right < inverse.numel():
            inverse[oriented_right] = oriented_left
    return inverse


def _solve(payload: dict) -> dict:
    pairs = load_pairs(payload["path"])
    if len(pairs) != 1:
        raise RuntimeError(f"expected one pair in {payload['path']}")
    pair = pairs[0]
    model = _load_model(payload["checkpoint"], payload["device"])
    total_started = time.perf_counter()
    left, right, swapped = pair.oriented()
    association = AssociationGraph.build(left, right)
    with torch.inference_mode():
        output = model(association, mode=payload["arm"])
    core_elapsed = time.perf_counter() - total_started
    snapshots, search_audit = search_trace(
        association,
        output.assignment.detach().cpu(),
        output.hard_mapping().detach().cpu(),
        tuple(payload["budgets"]),
        restarts=payload["restarts"],
        max_passes=payload["max_passes"],
        anneal_steps=payload["anneal_steps"],
        lns_steps=payload["lns_steps"],
        seed=payload["search_seed"],
    )
    snapshot_rows = {}
    for budget, snapshot in snapshots.items():
        mapping = _restore_orientation(snapshot.mapping, pair, swapped)
        snapshot_rows[str(int(budget) if float(budget).is_integer() else budget)] = {
            "budget_seconds": budget,
            "candidate_completed_at_seconds": snapshot.completed_at_seconds,
            "end_to_end_available_at_seconds": core_elapsed + snapshot.completed_at_seconds,
            "common_edges": snapshot.common_edges,
            "common_nodes": snapshot.common_nodes,
            "accuracy": None if pair.true_edges is None else snapshot.common_edges / pair.true_edges,
            "source": snapshot.source,
            "mapping": mapping.tolist(),
        }
    return {
        "key": pair.key,
        "source_path": payload["path"],
        "dataset": payload["dataset"],
        "arm": payload["arm"],
        "training_seed": payload["training_seed"],
        "search_seed": payload["search_seed"],
        "true_edges": pair.true_edges,
        "true_nodes": pair.true_nodes,
        "input_provenance": (pair.metadata or {}).get("input_provenance"),
        "core_elapsed_seconds": core_elapsed,
        "snapshots": snapshot_rows,
        "search_audit": search_audit,
        "checkpoint_sha256": payload["checkpoint_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["AIDS", "MOLHIV", "MCF-7"], required=True)
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument("--arm", choices=["learned", "analytic"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise RuntimeError("formal ENPDA search frontier refuses CPU execution")
    device_name = torch.cuda.get_device_name(0)
    if not any(name in device_name for name in config["hardware"]["accelerator_any"]):
        raise RuntimeError(f"formal search frontier requires H100/H200, found {device_name}")
    if not str(torch.version.cuda).startswith(config["hardware"]["cuda_runtime_prefix"]):
        raise RuntimeError(f"formal search frontier requires CUDA 12.8, found {torch.version.cuda}")
    torch.serialization.add_safe_globals([TorchVersion])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if int(checkpoint.get("training_seed", -1)) != args.seed:
        raise RuntimeError("checkpoint training seed mismatch")

    config_hash = _sha256(CONFIG)
    checkpoint_hash = _sha256(args.checkpoint)
    code_hashes = {str(path.relative_to(ROOT)): _sha256(path) for path in CODE_PATHS}
    paths = pair_paths("data/official", args.dataset, split="test", limit=100)
    completed = {}
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                completed[row["source_path"]] = row
    expected = {str(path) for path in paths}
    if set(completed) - expected:
        raise RuntimeError("frontier output contains an unexpected source")

    search = config["search"]
    common = {
        "dataset": args.dataset,
        "arm": args.arm,
        "training_seed": args.seed,
        "search_seed": args.seed,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "device": "cuda",
        "budgets": config["search_budget_seconds"],
        "restarts": int(search["maximum_restarts"]),
        "max_passes": int(search["refinement_passes"]),
        "anneal_steps": int(search["anneal_steps"]),
        "lns_steps": int(search["lns_steps"]),
    }
    payloads = [{**common, "path": str(path)} for path in paths if str(path) not in completed]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    with args.output.open("a", encoding="utf-8") as stream:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            futures = {pool.submit(_solve, payload): payload for payload in payloads}
            for index, future in enumerate(as_completed(futures), 1):
                row = future.result()
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                end = row["snapshots"][str(int(max(config["search_budget_seconds"])))]
                print(
                    f"[{index}/{len(payloads)}] {args.dataset} s{args.seed} {args.arm} "
                f"edges@{max(config['search_budget_seconds']):g}={end['common_edges']}/{row['true_edges']} "
                    f"events={row['search_audit']['event_count']}",
                    flush=True,
                )

    rows = [json.loads(line) for line in args.output.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != len(paths) or {row["source_path"] for row in rows} != expected:
        raise RuntimeError("frontier shard is incomplete")
    if {str(path.relative_to(ROOT)): _sha256(path) for path in CODE_PATHS} != code_hashes:
        raise RuntimeError("frontier code changed during evaluation")
    marker = {
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": config["protocol_version"],
        "config_sha256": config_hash,
        "dataset": args.dataset,
        "seed": args.seed,
        "arm": args.arm,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "hardware": {"device": device_name, "cuda_runtime": torch.version.cuda, "torch": torch.__version__},
        "workers": args.workers,
        "code_sha256": code_hashes,
        "records": len(rows),
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
    }
    args.marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.marker.with_name(args.marker.name + ".tmp")
    temporary.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.marker)
    print(json.dumps(marker, indent=2), flush=True)


if __name__ == "__main__":
    main()
