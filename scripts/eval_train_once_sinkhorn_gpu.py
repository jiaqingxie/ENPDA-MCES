#!/usr/bin/env python3
"""Formal native evaluation of train-once Sinkhorn and Gumbel-Sinkhorn controls."""

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
from nema.models.sinkhorn_baseline import TrainOnceSinkhorn


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/train_once_sinkhorn.json"
CODE_PATHS = (
    ROOT / "src/nema/models/sinkhorn_baseline.py",
    ROOT / "src/nema/models/enpda.py",
    ROOT / "src/nema/sinkhorn.py",
    ROOT / "src/nema/association.py",
    ROOT / "src/nema/rounding.py",
    Path(__file__).resolve(),
)
MOLHIV_EXCLUDED_KEYS = {"23", "46", "48", "54", "61", "64", "76"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def hardware_guard(config: dict) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("formal train-once Sinkhorn evaluation refuses CPU")
    accelerator = torch.cuda.get_device_name(0)
    if not any(name in accelerator for name in config["hardware"]["accelerator_any"]):
        raise RuntimeError(f"requires H100/H200, found {accelerator}")
    expected = str(config["hardware"]["cuda_runtime_prefix"])
    if not str(torch.version.cuda).startswith(expected):
        raise RuntimeError(f"requires CUDA {expected}, found {torch.version.cuda}")
    return {
        "device": accelerator,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "capability": list(torch.cuda.get_device_capability(0)),
    }


def load_checkpoint(path: Path, seed: int, config: dict) -> tuple[TrainOnceSinkhorn, dict]:
    torch.serialization.add_safe_globals([TorchVersion])
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("training_seed") != seed:
        raise RuntimeError("training seed/checkpoint mismatch")
    if payload.get("config_sha256") != sha256(CONFIG):
        raise RuntimeError("checkpoint config hash mismatch")
    if payload.get("native_test_paths_loaded") != []:
        raise RuntimeError("training checkpoint accessed native tests")
    model = TrainOnceSinkhorn(**payload["model_config"])
    model.load_state_dict(payload["model"], strict=True)
    return model.cuda().eval(), payload


def metrics(pair, edges: int) -> tuple[float, float]:
    accuracy = edges / pair.true_edges
    denominator = pair.left.num_edges + pair.right.num_edges - edges
    similarity = edges / denominator if denominator > 0 else 0.0
    mse = (similarity - pair.true_similarity) ** 2
    return float(accuracy), float(mse)


def evaluate_pair(
    model: TrainOnceSinkhorn,
    source: Path,
    dataset: str,
    training_seed: int,
    pair_index: int,
    samples: int,
) -> dict:
    pair = load_pair(source)
    left, right, swapped = pair.oriented()
    build_started = time.perf_counter()
    association = AssociationGraph.build(left, right)
    build_seconds = time.perf_counter() - build_started

    torch.cuda.synchronize()
    deterministic_started = time.perf_counter()
    with torch.inference_mode():
        deterministic = model(association, learned=True)
    torch.cuda.synchronize()
    deterministic_network = time.perf_counter() - deterministic_started
    round_started = time.perf_counter()
    deterministic_mapping = deterministic.hard_mapping().cpu()
    deterministic_stats = association.hard_statistics(deterministic_mapping)
    deterministic_round = time.perf_counter() - round_started

    generator_seed = 2026083000 + 100000 * training_seed + pair_index
    generator = torch.Generator(device="cuda").manual_seed(generator_seed)
    torch.cuda.synchronize()
    gumbel_started = time.perf_counter()
    with torch.inference_mode():
        mappings, _ = model.gumbel_mappings(
            association,
            samples=samples,
            generator=generator,
        )
    torch.cuda.synchronize()
    gumbel_network = time.perf_counter() - gumbel_started
    round_started = time.perf_counter()
    scored = [(association.hard_statistics(mapping), mapping.cpu()) for mapping in mappings]
    best_index = max(range(len(scored)), key=lambda index: scored[index][0])
    gumbel_stats, gumbel_mapping = scored[best_index]
    gumbel_round = time.perf_counter() - round_started

    deterministic_accuracy, deterministic_mse = metrics(pair, deterministic_stats[0])
    gumbel_accuracy, gumbel_mse = metrics(pair, gumbel_stats[0])
    return {
        "protocol_version": json.loads(CONFIG.read_text())["protocol_version"],
        "dataset": dataset,
        "training_seed": training_seed,
        "source_path": str(source),
        "source_sha256": sha256(source),
        "key": pair.key,
        "swapped": swapped,
        "left_nodes": left.num_nodes,
        "right_nodes": right.num_nodes,
        "true_edges": pair.true_edges,
        "acg_candidates": association.num_candidates,
        "acg_edges": association.num_edges,
        "acg_build_seconds": build_seconds,
        "deterministic": {
            "method": "Train-once Sinkhorn",
            "mapping": deterministic_mapping.tolist(),
            "common_edges": deterministic_stats[0],
            "common_nodes": deterministic_stats[1],
            "accuracy": deterministic_accuracy,
            "similarity_mse": deterministic_mse,
            "network_seconds": deterministic_network,
            "hungarian_seconds": deterministic_round,
            "total_seconds": build_seconds + deterministic_network + deterministic_round,
            "row_residual": deterministic.row_residual,
            "max_column_excess": deterministic.max_column_excess,
        },
        "gumbel": {
            "method": f"Train-once Gumbel-Sinkhorn ({samples})",
            "samples": samples,
            "generator_seed": generator_seed,
            "selected_sample": best_index,
            "mapping": gumbel_mapping.tolist(),
            "common_edges": gumbel_stats[0],
            "common_nodes": gumbel_stats[1],
            "accuracy": gumbel_accuracy,
            "similarity_mse": gumbel_mse,
            "network_seconds": gumbel_network,
            "hungarian_seconds": gumbel_round,
            "total_seconds": build_seconds + gumbel_network + gumbel_round,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["AIDS", "MOLHIV", "MCF-7"], required=True)
    parser.add_argument("--training-seed", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--completion", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    hardware = hardware_guard(config)
    model, checkpoint = load_checkpoint(args.checkpoint, args.training_seed, config)
    checkpoint_sha = sha256(args.checkpoint)
    code_sha = {str(path.relative_to(ROOT)): sha256(path) for path in CODE_PATHS}
    paths = pair_paths("data/official", args.dataset, split="test", limit=100)
    if args.dataset == "MOLHIV":
        # Frozen native protocol: these seven released tensors are empty and
        # their SMILES reconstructions change topology relative to the stored
        # reference labels.  This exclusion predates all baseline results.
        paths = [path for path in paths if path.stem.removeprefix("graphs_") not in MOLHIV_EXCLUDED_KEYS]
    completed = {}
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                completed[row["source_path"]] = row
    expected = {str(path) for path in paths}
    if set(completed) - expected:
        raise RuntimeError("resumable output contains paths outside the frozen test shard")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as stream:
        for index, path in enumerate(paths):
            if str(path) in completed:
                continue
            row = evaluate_pair(
                model,
                path,
                args.dataset,
                args.training_seed,
                index,
                int(config["evaluation"]["gumbel_samples"]),
            )
            row["checkpoint_sha256"] = checkpoint_sha
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            print(
                f"[{index + 1}/{len(paths)}] {args.dataset} "
                f"S={row['deterministic']['common_edges']} G={row['gumbel']['common_edges']}",
                flush=True,
            )
    rows = [json.loads(line) for line in args.output.read_text().splitlines() if line.strip()]
    if len(rows) != len(paths) or {row["source_path"] for row in rows} != expected:
        raise RuntimeError("native shard coverage mismatch")
    if {str(path.relative_to(ROOT)): sha256(path) for path in CODE_PATHS} != code_sha:
        raise RuntimeError("evaluation code changed during the shard")
    marker = {
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": config["protocol_version"],
        "config_sha256": sha256(CONFIG),
        "dataset": args.dataset,
        "training_seed": args.training_seed,
        "records": len(rows),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "checkpoint_sha256": checkpoint_sha,
        "selected_training_epoch": checkpoint["selected_validation"]["epoch"],
        "hardware": hardware,
        "code_sha256": code_sha,
    }
    atomic_json(args.completion, marker)
    print(json.dumps(marker, indent=2), flush=True)


if __name__ == "__main__":
    main()
