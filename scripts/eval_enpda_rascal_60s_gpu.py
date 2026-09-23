#!/usr/bin/env python3
"""Formal globally capped 60-second ENPDA-Solver evaluation."""

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

import numpy as np
import torch
from torch.torch_version import TorchVersion

from nema.association import AssociationGraph
from nema.data import load_pair, pair_paths
from nema.enpda_anytime import search_trace
from nema.models.enpda import ENPDAModel
from nema.sinkhorn import sample_gumbel_like


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/enpda_rascal_60s.json"
MOLHIV_RECOVERED = {"23", "46", "48", "54", "61", "64", "76"}
CODE = (
    ROOT / "src/nema/models/enpda.py",
    ROOT / "src/nema/enpda_anytime.py",
    ROOT / "src/nema/rounding.py",
    ROOT / "src/nema/association.py",
    Path(__file__).resolve(),
)
_MODEL_CACHE: dict[tuple[str, str], ENPDAModel] = {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(path: str, device: str) -> ENPDAModel:
    key = (path, device)
    if key not in _MODEL_CACHE:
        torch.serialization.add_safe_globals([TorchVersion])
        payload = torch.load(path, map_location="cpu", weights_only=True)
        model = ENPDAModel(**payload["model_config"])
        model.load_state_dict(payload["model"], strict=True)
        _MODEL_CACHE[key] = model.to(device).eval()
    return _MODEL_CACHE[key]


def restore_orientation(mapping: torch.Tensor, pair, swapped: bool) -> torch.Tensor:
    if not swapped:
        return mapping.detach().cpu().long()
    inverse = torch.full((pair.left.num_nodes,), -1, dtype=torch.long)
    for oriented_left, oriented_right in enumerate(mapping.tolist()):
        if 0 <= oriented_right < inverse.numel():
            inverse[oriented_right] = oriented_left
    return inverse


def solve(payload: dict) -> dict:
    pair = load_pair(payload["source_path"])
    model = load_model(payload["checkpoint"], payload["device"])
    cap = float(payload["cap_seconds"])
    started = time.perf_counter()
    left, right, swapped = pair.oriented()
    association = AssociationGraph.build(left, right)
    best_mapping: torch.Tensor | None = None
    best_stats = (-1, -1)
    best_source = ""
    best_available = 0.0
    stream_records: list[dict] = []

    def consider(mapping: torch.Tensor, source: str, available: float) -> None:
        nonlocal best_mapping, best_stats, best_source, best_available
        if available > cap + 1e-9:
            return
        candidate = mapping.detach().cpu().long()
        stats = association.hard_statistics(candidate)
        if stats > best_stats:
            best_mapping = candidate.clone()
            best_stats = stats
            best_source = source
            best_available = available

    generator = torch.Generator(device=payload["device"]).manual_seed(payload["pair_seed"])
    streams: list[dict] = []
    with torch.inference_mode():
        for index in range(int(payload["learned_streams"])):
            if time.perf_counter() - started >= cap:
                break
            noise = None
            if index:
                template = torch.empty(association.shape, device=payload["device"])
                noise = float(payload["noise_scale"]) * sample_gumbel_like(template, generator)
            output = model(association, mode="learned", initial_noise=noise)
            torch.cuda.synchronize()
            available = time.perf_counter() - started
            mapping = output.hard_mapping().detach().cpu()
            source = f"learned_stream_{index}"
            consider(mapping, source + ":core", available)
            streams.append({"source": source, "score": output.assignment.detach().cpu(), "mapping": mapping})
        if time.perf_counter() - started < cap:
            output = model(association, mode="analytic")
            torch.cuda.synchronize()
            available = time.perf_counter() - started
            mapping = output.hard_mapping().detach().cpu()
            consider(mapping, "analytic_stream:core", available)
            streams.append({"source": "analytic_stream", "score": output.assignment.detach().cpu(), "mapping": mapping})

    if best_mapping is None:
        # The association construction itself should never consume the cap on
        # the native scope, but keep a legal deterministic emergency return.
        best_mapping = torch.arange(association.left.num_nodes, dtype=torch.long)
        best_mapping[best_mapping >= association.right.num_nodes] = -1
        best_stats = association.hard_statistics(best_mapping)
        best_source = "canonical_emergency"
        best_available = min(time.perf_counter() - started, cap)

    if streams:
        rotation = int(payload["search_seed"]) % len(streams)
        streams = streams[rotation:] + streams[:rotation]
    for index, stream in enumerate(streams):
        now = time.perf_counter() - started
        remaining = cap - now
        streams_left = len(streams) - index
        if remaining <= 0 or streams_left <= 0:
            break
        allocation = remaining / streams_left
        local_started = time.perf_counter()
        snapshots, audit = search_trace(
            association,
            stream["score"],
            stream["mapping"],
            (allocation,),
            restarts=int(payload["restarts"]),
            max_passes=int(payload["max_passes"]),
            anneal_steps=int(payload["anneal_steps"]),
            lns_steps=int(payload["lns_steps"]),
            seed=int(payload["pair_seed"]) + 1009 * index,
        )
        snapshot = snapshots[allocation]
        absolute_available = (local_started - started) + snapshot.completed_at_seconds
        consider(snapshot.mapping, stream["source"] + ":search", absolute_available)
        stream_records.append(
            {
                "source": stream["source"],
                "allocation_seconds": allocation,
                "candidate_available_at_seconds": absolute_available,
                "candidate_edges": snapshot.common_edges,
                "candidate_nodes": snapshot.common_nodes,
                "search_audit": audit,
            }
        )

    elapsed = time.perf_counter() - started
    mapping = restore_orientation(best_mapping, pair, swapped)
    return {
        "protocol_sha256": payload["protocol_sha256"],
        "method": "ENPDA-Solver-60s",
        "dataset": payload["dataset"],
        "seed": payload["seed"],
        "search_seed": payload["search_seed"],
        "key": pair.key,
        "source_path": payload["source_path"],
        "source_sha256": payload["source_sha256"],
        "true_edges": pair.true_edges,
        "true_nodes": pair.true_nodes,
        "common_edges": best_stats[0],
        "common_nodes": best_stats[1],
        "accuracy": None if pair.true_edges is None else best_stats[0] / pair.true_edges,
        "mapping": mapping.tolist(),
        "selected_source": best_source,
        "incumbent_available_at_seconds": best_available,
        "wall_clock_cap_seconds": cap,
        "total_worker_seconds": elapsed,
        "deadline_overrun_seconds": max(0.0, elapsed - cap),
        "streams": stream_records,
        "checkpoint_sha256": payload["checkpoint_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("AIDS", "MOLHIV", "MCF-7"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--marker", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise RuntimeError("formal ENPDA-60 evaluation refuses CPU execution")
    device = torch.cuda.get_device_name(0)
    if not any(name in device for name in config["hardware"]["accelerator_any"]):
        raise RuntimeError(f"requires H100/H200, found {device}")
    if not str(torch.version.cuda).startswith(config["hardware"]["cuda_runtime_prefix"]):
        raise RuntimeError(f"requires CUDA 12.8, found {torch.version.cuda}")
    torch.serialization.add_safe_globals([TorchVersion])
    checkpoint_payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if int(checkpoint_payload.get("training_seed", -1)) != args.seed:
        raise RuntimeError("checkpoint seed mismatch")

    config_hash = sha256(CONFIG)
    checkpoint_hash = sha256(args.checkpoint)
    code_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in CODE}
    paths = pair_paths("data/official", args.dataset, split="test", limit=100)
    if args.dataset == "MOLHIV":
        paths = [path for path in paths if path.stem.rsplit("_", 1)[-1] not in MOLHIV_RECOVERED]
    expected = {str(path) for path in paths}
    completed = {}
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line:
                row = json.loads(line)
                completed[row["source_path"]] = row
    if set(completed) - expected:
        raise RuntimeError("existing output contains paths outside the frozen native scope")

    settings = config["enpda"]
    payloads = []
    for index, path in enumerate(paths):
        if str(path) in completed:
            continue
        payloads.append(
            {
                "protocol_sha256": config_hash,
                "dataset": args.dataset,
                "seed": args.seed,
                "search_seed": args.seed,
                "pair_seed": args.seed + 10007 * index,
                "source_path": str(path),
                "source_sha256": sha256(path),
                "checkpoint": str(args.checkpoint),
                "checkpoint_sha256": checkpoint_hash,
                "device": "cuda",
                "cap_seconds": float(config["wall_clock_cap_seconds"]),
                "learned_streams": int(settings["learned_streams"]),
                "noise_scale": float(settings["noise_scale"]),
                "restarts": int(settings["maximum_restarts_per_stream"]),
                "max_passes": int(settings["refinement_passes"]),
                "anneal_steps": int(settings["anneal_steps"]),
                "lns_steps": int(settings["lns_steps"]),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    with args.output.open("a", encoding="utf-8") as stream:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            futures = {pool.submit(solve, payload): payload for payload in payloads}
            for index, future in enumerate(as_completed(futures), 1):
                row = future.result()
                completed[row["source_path"]] = row
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                print(
                    f"[{index}/{len(payloads)}] {args.dataset} s{args.seed} "
                    f"edges={row['common_edges']}/{row['true_edges']} "
                    f"elapsed={row['total_worker_seconds']:.2f}s",
                    flush=True,
                )

    if len(completed) != len(paths) or set(completed) != expected:
        raise RuntimeError("ENPDA-60 shard is incomplete")
    if {str(path.relative_to(ROOT)): sha256(path) for path in CODE} != code_hashes:
        raise RuntimeError("ENPDA-60 code changed during evaluation")
    marker = {
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": config["protocol_version"],
        "config_sha256": config_hash,
        "dataset": args.dataset,
        "seed": args.seed,
        "records": len(completed),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "hardware": {"device": device, "torch": torch.__version__, "cuda_runtime": torch.version.cuda},
        "workers": args.workers,
        "code_sha256": code_hashes,
        "output": str(args.output),
        "output_sha256": sha256(args.output),
    }
    args.marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.marker.with_name(args.marker.name + ".tmp")
    temporary.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.marker)
    print(json.dumps(marker, indent=2), flush=True)


if __name__ == "__main__":
    main()
