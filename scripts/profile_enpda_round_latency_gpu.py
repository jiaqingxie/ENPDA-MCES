#!/usr/bin/env python3
"""Synchronized, same-process ENPDA-Core latency profile for T=4 versus T=8."""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.torch_version import TorchVersion

from nema.association import AssociationGraph
from nema.data import load_pairs, pair_paths
from nema.models.enpda import ENPDAModel


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/enpda_sota.json"
OUTDIR = ROOT / "results/enpda_round_latency"
RAW = OUTDIR / "raw.jsonl"
SUMMARY = OUTDIR / "summary.json"
COMPLETE = OUTDIR / "COMPLETE.json"
DATASETS = ("AIDS", "MOLHIV", "MCF-7")
SEEDS = (0, 1, 2)
ROUNDS = (4, 8)
REPEATS = 2
MOLHIV_EXCLUDED = {"23", "46", "48", "54", "61", "64", "76"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sync() -> None:
    torch.cuda.synchronize()


def timed_call(model: ENPDAModel, pair, rounds: int) -> dict:
    sync()
    total_started = time.perf_counter()

    build_started = time.perf_counter()
    left, right, _ = pair.oriented()
    association = AssociationGraph.build(left, right)
    build_seconds = time.perf_counter() - build_started

    sync()
    trajectory_started = time.perf_counter()
    with torch.inference_mode():
        output = model(association, mode="learned", iterations=rounds)
    sync()
    trajectory_seconds = time.perf_counter() - trajectory_started

    projection_started = time.perf_counter()
    mapping = output.hard_mapping().detach().cpu()
    common_edges, common_nodes = association.hard_statistics(mapping)
    sync()
    projection_seconds = time.perf_counter() - projection_started
    total_seconds = time.perf_counter() - total_started
    return {
        "common_edges": int(common_edges),
        "common_nodes": int(common_nodes),
        "build_seconds": float(build_seconds),
        "trajectory_seconds": float(trajectory_seconds),
        "projection_seconds": float(projection_seconds),
        "total_seconds": float(total_seconds),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("latency profile refuses CPU execution")
    device_name = torch.cuda.get_device_name(0)
    if not any(token in device_name for token in ("H100", "H200")):
        raise RuntimeError(f"expected H100/H200, found {device_name}")
    if not str(torch.version.cuda).startswith("12.8"):
        raise RuntimeError(f"expected CUDA 12.8, found {torch.version.cuda}")

    config_hash = sha256(CONFIG)
    paths_by_dataset = {}
    for dataset in DATASETS:
        paths = pair_paths("data/official", dataset, split="test", limit=100)
        if dataset == "MOLHIV":
            paths = [p for p in paths if p.stem.rsplit("_", 1)[-1] not in MOLHIV_EXCLUDED]
        paths_by_dataset[dataset] = paths
    if {k: len(v) for k, v in paths_by_dataset.items()} != {
        "AIDS": 100, "MOLHIV": 91, "MCF-7": 100
    }:
        raise RuntimeError("native scope mismatch")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    if RAW.exists() or COMPLETE.exists():
        raise RuntimeError("refusing to mix a new timing run with existing output")

    rows = []
    with RAW.open("w", encoding="utf-8") as stream:
        for seed in SEEDS:
            checkpoint = ROOT / f"checkpoints/enpda_formal/seed{seed}.best.pt"
            torch.serialization.add_safe_globals([TorchVersion])
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if int(payload["training_seed"]) != seed or payload.get("config_sha256") != config_hash:
                raise RuntimeError("checkpoint provenance mismatch")
            model = ENPDAModel(**payload["model_config"])
            model.load_state_dict(payload["model"], strict=True)
            model = model.cuda().eval()

            warm_pair = load_pairs(paths_by_dataset["AIDS"][0])[0]
            warm_left, warm_right, _ = warm_pair.oriented()
            warm_association = AssociationGraph.build(warm_left, warm_right)
            with torch.inference_mode():
                for _ in range(5):
                    for rounds in ROUNDS:
                        warm_output = model(warm_association, mode="learned", iterations=rounds)
                        warm_output.hard_mapping()
            sync()

            for dataset in DATASETS:
                for pair_index, path in enumerate(paths_by_dataset[dataset]):
                    pair = load_pairs(path)[0]
                    for repeat in range(REPEATS):
                        order = ROUNDS if (pair_index + repeat) % 2 == 0 else tuple(reversed(ROUNDS))
                        for rounds in order:
                            measured = timed_call(model, pair, rounds)
                            row = {
                                "dataset": dataset,
                                "pair_key": str(pair.key),
                                "source_path": str(path),
                                "training_seed": seed,
                                "repeat": repeat,
                                "rounds": rounds,
                                **measured,
                            }
                            rows.append(row)
                            stream.write(json.dumps(row) + "\n")
                            stream.flush()
                    print(
                        f"{dataset} seed={seed} pair={pair_index + 1}/{len(paths_by_dataset[dataset])}",
                        flush=True,
                    )

    expected = sum(map(len, paths_by_dataset.values())) * len(SEEDS) * len(ROUNDS) * REPEATS
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} timing rows, found {len(rows)}")

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["rounds"])].append(row)
    summary = {
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": {"device": device_name, "cuda_runtime": torch.version.cuda},
        "timing_boundary": (
            "pair loading excluded; total includes orientation, ACG construction, synchronized "
            "model trajectory, one Hungarian projection, and hard-score recomputation"
        ),
        "warmup": "five alternating T=4/T=8 passes per checkpoint before measurement",
        "repeats": REPEATS,
        "records": len(rows),
        "raw_sha256": sha256(RAW),
        "cells": {},
    }
    for dataset in DATASETS:
        for rounds in ROUNDS:
            cell = grouped[(dataset, rounds)]
            metrics = {}
            for field in ("build_seconds", "trajectory_seconds", "projection_seconds", "total_seconds"):
                values = np.asarray([float(row[field]) for row in cell])
                metrics[field] = {
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "p95": float(np.quantile(values, 0.95)),
                }
            summary["cells"][f"{dataset}|{rounds}"] = {"records": len(cell), **metrics}
        if not (
            summary["cells"][f"{dataset}|8"]["trajectory_seconds"]["median"]
            > summary["cells"][f"{dataset}|4"]["trajectory_seconds"]["median"]
        ):
            raise RuntimeError(f"trajectory timing did not increase on {dataset}")

    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    COMPLETE.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
