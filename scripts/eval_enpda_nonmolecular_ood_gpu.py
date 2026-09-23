#!/usr/bin/env python3
"""Frozen molecular-to-social ENPDA-Core evaluation with analytic optima."""

from __future__ import annotations

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
from nema.graph import GraphPair, LabeledGraph
from nema.models.enpda import ENPDAModel


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/enpda_nonmolecular_ood.json"
PARENT = ROOT / "configs/enpda_sota.json"
MANIFEST = ROOT / "results/enpda_nonmolecular_ood/manifest.json"
RAW = ROOT / "results/enpda_nonmolecular_ood/raw.jsonl"
SUMMARY = ROOT / "results/enpda_nonmolecular_ood/summary.json"
COMPLETE = ROOT / "results/enpda_nonmolecular_ood/COMPLETE.json"
OUTPUT_TEX = ROOT / "paper/generated/enpda_nonmolecular_ood.tex"
CODE_PATHS = (
    ROOT / "src/nema/models/enpda.py",
    ROOT / "src/nema/association.py",
    ROOT / "src/nema/graph.py",
    ROOT / "src/nema/rounding.py",
    Path(__file__).resolve(),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path: Path) -> dict:
    torch.serialization.add_safe_globals([TorchVersion])
    return torch.load(path, map_location="cpu", weights_only=True)


def hardware_guard(config: dict) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("formal non-molecular OOD evaluation refuses CPU execution")
    name = torch.cuda.get_device_name(0)
    if not any(token in name for token in config["hardware"]["accelerator_any"]):
        raise RuntimeError(f"expected H100/H200, found {name}")
    expected = str(config["hardware"]["cuda_runtime_prefix"])
    if not str(torch.version.cuda).startswith(expected):
        raise RuntimeError(f"expected CUDA {expected}, found {torch.version.cuda}")
    return {
        "device": name,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_capability": list(torch.cuda.get_device_capability(0)),
    }


def graph(nodes: int, edges: list[list[int]]) -> LabeledGraph:
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )
    return LabeledGraph(
        node_labels=torch.zeros(nodes, dtype=torch.long),
        edge_index=edge_index,
        edge_labels=torch.zeros(len(edges), dtype=torch.long),
    )


def build_pair(specification: dict) -> tuple[GraphPair, torch.Tensor]:
    nodes = int(specification["num_nodes"])
    left = graph(nodes, specification["source_edges"])
    right = graph(nodes, specification["target_edges"])
    optimum = int(specification["exact_optimum_edges"])
    pair = GraphPair(
        left=left,
        right=right,
        true_edges=optimum,
        key=str(specification["pair_id"]),
        metadata={"dataset_index": int(specification["dataset_index"]), "domain": "social"},
    )
    planted = torch.tensor(specification["target_permutation_old_to_new"], dtype=torch.long)
    association = AssociationGraph.build(left, right)
    planted_edges, _ = association.hard_statistics(planted)
    if planted_edges != optimum or optimum != right.num_edges:
        raise RuntimeError(
            f"invalid exact construction {pair.key}: planted={planted_edges}, "
            f"target={right.num_edges}, claimed={optimum}"
        )
    return pair, planted


def bootstrap_effect(matrix: np.ndarray, seed: int, replicates: int) -> tuple[float, list[float]]:
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for position in range(replicates):
        seed_indices = rng.integers(0, matrix.shape[0], matrix.shape[0])
        pair_indices = rng.integers(0, matrix.shape[1], matrix.shape[1])
        draws[position] = matrix[np.ix_(seed_indices, pair_indices)].mean()
    return float(matrix.mean()), [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    if manifest["protocol_version"] != config["protocol_version"]:
        raise RuntimeError("manifest protocol mismatch")
    if manifest["config_sha256"] != sha256(CONFIG):
        raise RuntimeError("configuration changed after pair construction")
    if len(manifest["pairs"]) != int(config["dataset"]["num_pairs"]):
        raise RuntimeError("wrong number of frozen OOD pairs")
    if len({pair["dataset_index"] for pair in manifest["pairs"]}) != len(manifest["pairs"]):
        raise RuntimeError("source graphs are not graph-disjoint")

    hardware = hardware_guard(config)
    code_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in CODE_PATHS}
    pairs = [(specification, *build_pair(specification)) for specification in manifest["pairs"]]
    RAW.parent.mkdir(parents=True, exist_ok=True)
    completed: dict[str, dict] = {}
    if RAW.exists():
        for line in RAW.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row["record_id"])
            if key in completed:
                raise RuntimeError(f"duplicate row {key}")
            completed[key] = row

    with RAW.open("a", encoding="utf-8") as stream:
        for training_seed in config["training_seeds"]:
            checkpoint_path = ROOT / config["evaluation"]["checkpoint_pattern"].format(seed=training_seed)
            payload = load_checkpoint(checkpoint_path)
            if int(payload["training_seed"]) != int(training_seed):
                raise RuntimeError("checkpoint training-seed mismatch")
            if payload.get("config_sha256") != sha256(PARENT):
                raise RuntimeError("checkpoint parent-protocol mismatch")
            model = ENPDAModel(**payload["model_config"])
            model.load_state_dict(payload["model"], strict=True)
            model = model.cuda().eval()
            for pair_specification, pair, _ in pairs:
                association = AssociationGraph.build(pair.left, pair.right)
                for arm in config["arms"]:
                    key = f"{training_seed}|{pair.key}|{arm}|T{config['rounds']}"
                    if key in completed:
                        continue
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    with torch.inference_mode():
                        output = model(
                            association,
                            mode=str(arm),
                            iterations=int(config["rounds"]),
                        )
                        mapping = output.hard_mapping().detach().cpu()
                    common_edges, common_nodes = association.hard_statistics(mapping)
                    torch.cuda.synchronize()
                    runtime = time.perf_counter() - started
                    optimum = int(pair.true_edges)
                    row = {
                        "record_id": key,
                        "training_seed": int(training_seed),
                        "pair_id": pair.key,
                        "dataset_index": int(pair_specification["dataset_index"]),
                        "arm": str(arm),
                        "rounds": int(config["rounds"]),
                        "num_nodes": pair.left.num_nodes,
                        "source_edges": pair.left.num_edges,
                        "target_edges": pair.right.num_edges,
                        "common_edges": int(common_edges),
                        "common_nodes": int(common_nodes),
                        "exact_optimum_edges": optimum,
                        "accuracy": float(common_edges / optimum),
                        "missing_edges": int(optimum - common_edges),
                        "exact_recovery": bool(common_edges == optimum),
                        "runtime_seconds": float(runtime),
                        "soft_objective": float(output.objectives[-1]),
                        "row_residual": float(output.row_residual),
                        "max_column_excess": float(output.max_column_excess),
                        "checkpoint_sha256": sha256(checkpoint_path),
                    }
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    completed[key] = row
                    print(
                        f"[{len(completed)}/600] seed={training_seed} {pair.key} {arm} "
                        f"edges={common_edges}/{optimum} exact={common_edges == optimum} "
                        f"time={runtime:.4f}s",
                        flush=True,
                    )

    rows = list(completed.values())
    expected = len(pairs) * len(config["training_seeds"]) * len(config["arms"])
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} rows, found {len(rows)}")
    if {str(path.relative_to(ROOT)): sha256(path) for path in CODE_PATHS} != code_hashes:
        raise RuntimeError("evaluation code changed during execution")

    cells: dict[str, dict] = {}
    for arm in config["arms"]:
        arm_rows = [row for row in rows if row["arm"] == arm]
        seed_accuracy = [
            100.0 * float(np.mean([row["accuracy"] for row in arm_rows if row["training_seed"] == seed]))
            for seed in config["training_seeds"]
        ]
        cells[arm] = {
            "records": len(arm_rows),
            "accuracy_percent": 100.0 * float(np.mean([row["accuracy"] for row in arm_rows])),
            "seed_accuracy_percent": seed_accuracy,
            "seed_std_percent": float(np.std(seed_accuracy, ddof=1)),
            "exact_recovery_percent": 100.0 * float(np.mean([row["exact_recovery"] for row in arm_rows])),
            "mean_missing_edges": float(np.mean([row["missing_edges"] for row in arm_rows])),
            "mean_runtime_seconds": float(np.mean([row["runtime_seconds"] for row in arm_rows])),
        }

    lookup = {
        (int(row["training_seed"]), str(row["pair_id"]), str(row["arm"])): row
        for row in rows
    }
    accuracy_difference = np.asarray(
        [
            [
                lookup[(seed, pair_specification["pair_id"], "learned")]["accuracy"]
                - lookup[(seed, pair_specification["pair_id"], "analytic")]["accuracy"]
                for pair_specification in manifest["pairs"]
            ]
            for seed in config["training_seeds"]
        ],
        dtype=np.float64,
    )
    exact_difference = np.asarray(
        [
            [
                float(lookup[(seed, pair_specification["pair_id"], "learned")]["exact_recovery"])
                - float(lookup[(seed, pair_specification["pair_id"], "analytic")]["exact_recovery"])
                for pair_specification in manifest["pairs"]
            ]
            for seed in config["training_seeds"]
        ],
        dtype=np.float64,
    )
    replicates = int(config["statistics"]["bootstrap_replicates"])
    accuracy_mean, accuracy_ci = bootstrap_effect(accuracy_difference, 20260830, replicates)
    exact_mean, exact_ci = bootstrap_effect(exact_difference, 20260831, replicates)
    edge_difference = np.asarray(
        [
            lookup[(seed, pair_specification["pair_id"], "learned")]["common_edges"]
            - lookup[(seed, pair_specification["pair_id"], "analytic")]["common_edges"]
            for seed in config["training_seeds"]
            for pair_specification in manifest["pairs"]
        ]
    )
    effects = {
        "accuracy_gain_points": 100.0 * accuracy_mean,
        "accuracy_gain_ci95_points": [100.0 * value for value in accuracy_ci],
        "exact_recovery_gain_points": 100.0 * exact_mean,
        "exact_recovery_gain_ci95_points": [100.0 * value for value in exact_ci],
        "edge_wins_ties_losses": [
            int((edge_difference > 0).sum()),
            int((edge_difference == 0).sum()),
            int((edge_difference < 0).sum()),
        ],
    }
    node_counts = np.asarray([pair_specification["num_nodes"] for pair_specification in manifest["pairs"]])
    optimum_counts = np.asarray([pair_specification["exact_optimum_edges"] for pair_specification in manifest["pairs"]])
    summary = {
        "status": "complete",
        "protocol_version": config["protocol_version"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": hardware,
        "config_sha256": sha256(CONFIG),
        "parent_config_sha256": sha256(PARENT),
        "manifest_sha256": sha256(MANIFEST),
        "raw_sha256": sha256(RAW),
        "code_sha256": code_hashes,
        "scope": {
            "pairs": len(pairs),
            "training_seeds": len(config["training_seeds"]),
            "records": len(rows),
            "nodes_min_median_max": [int(node_counts.min()), float(np.median(node_counts)), int(node_counts.max())],
            "optimum_edges_min_median_max": [int(optimum_counts.min()), float(np.median(optimum_counts)), int(optimum_counts.max())],
        },
        "cells": cells,
        "paired_effects": effects,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    COMPLETE.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lower, upper = effects["accuracy_gain_ci95_points"]
    wins, ties, losses = effects["edge_wins_ties_losses"]
    table = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Frozen molecular-to-social transfer on 100 graph-disjoint IMDB-BINARY topologies.  Targets delete 30\% of source edges and independently permute nodes, so the planted map attains the analytic MCES optimum.  Both Core arms use four rounds, universal labels, one Hungarian projection, and no refinement.}",
        r"\label{tab:enpda-nonmolecular-ood}",
        r"\setlength{\tabcolsep}{5pt}\renewcommand{\arraystretch}{.94}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Method & Accuracy (\%) & Exact (\%) & Missing edges\\",
        r"\midrule",
        f"Analytic Core & {cells['analytic']['accuracy_percent']:.2f} & {cells['analytic']['exact_recovery_percent']:.1f} & {cells['analytic']['mean_missing_edges']:.2f}\\",
        f"ENPDA-Core & {cells['learned']['accuracy_percent']:.2f}$\\pm${cells['learned']['seed_std_percent']:.2f} & {cells['learned']['exact_recovery_percent']:.1f} & {cells['learned']['mean_missing_edges']:.2f}\\",
        r"\bottomrule",
        r"\end{tabular}",
        f"\\[-1pt]{{\\footnotesize Paired accuracy gain: {effects['accuracy_gain_points']:+.2f} pp "
        f"[95\\% CI {lower:+.2f},{upper:+.2f}]; edge W/T/L={wins}/{ties}/{losses}.}}",
        r"\end{table}",
    ]
    OUTPUT_TEX.write_text("\n".join(table) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
