#!/usr/bin/env python3
"""Frozen-checkpoint native ENPDA round-prefix ablation at T=1/2/4/8.

The experiment changes only the number of executed primal--dual rounds.  Each
prefix receives one Hungarian projection and no hard refinement.  Both the
learned and analytic dynamics use the same model implementation and graph
pairs.  The script refuses CPU execution and records resumable per-pair rows.
"""

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
from nema.data import load_pairs, pair_paths
from nema.models.enpda import ENPDAModel


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/enpda_round_ablation.json"
PARENT = ROOT / "configs/enpda_sota.json"
OUTPUT = ROOT / "results/enpda_round_ablation/raw.jsonl"
SUMMARY = ROOT / "results/enpda_round_ablation/summary.json"
ATTESTATION = ROOT / "results/enpda_round_ablation/COMPLETE.json"
OUTPUT_TEX = ROOT / "paper/generated/enpda_round_ablation.tex"
CODE_PATHS = (
    ROOT / "src/nema/models/enpda.py",
    ROOT / "src/nema/association.py",
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
        raise RuntimeError("round ablation refuses CPU execution")
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


def record_id(seed: int, dataset: str, source: str, arm: str, rounds: int) -> str:
    return f"{seed}|{dataset}|{source}|{arm}|{rounds}"


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    hardware = hardware_guard(config)
    code_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in CODE_PATHS}
    excluded = set(config["molhiv_excluded_keys"])
    paths_by_dataset: dict[str, list[Path]] = {}
    for dataset in config["datasets"]:
        paths = pair_paths("data/official", dataset, split="test", limit=100)
        if dataset == "MOLHIV":
            paths = [
                path for path in paths
                if path.stem.rsplit("_", 1)[-1] not in excluded
            ]
        paths_by_dataset[dataset] = paths
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    completed: dict[str, dict] = {}
    if OUTPUT.exists():
        for line in OUTPUT.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row["record_id"])
            if key in completed:
                raise RuntimeError(f"duplicate result row {key}")
            completed[key] = row

    with OUTPUT.open("a", encoding="utf-8") as stream:
        for seed in config["training_seeds"]:
            checkpoint_path = ROOT / f"checkpoints/enpda_formal/seed{seed}.best.pt"
            payload = load_checkpoint(checkpoint_path)
            if int(payload["training_seed"]) != int(seed):
                raise RuntimeError("checkpoint seed mismatch")
            if payload.get("config_sha256") != sha256(PARENT):
                raise RuntimeError("checkpoint parent-protocol mismatch")
            model = ENPDAModel(**payload["model_config"])
            model.load_state_dict(payload["model"], strict=True)
            model = model.cuda().eval()

            for dataset in config["datasets"]:
                for path in paths_by_dataset[dataset]:
                    pair = load_pairs(path)[0]
                    source = str(path)
                    for arm in config["arms"]:
                        for rounds in config["round_budgets"]:
                            key = record_id(seed, dataset, source, arm, int(rounds))
                            if key in completed:
                                continue
                            torch.cuda.synchronize()
                            started = time.perf_counter()
                            left, right, _ = pair.oriented()
                            association = AssociationGraph.build(left, right)
                            with torch.inference_mode():
                                output = model(
                                    association,
                                    mode=str(arm),
                                    iterations=int(rounds),
                                )
                                mapping = output.hard_mapping().detach().cpu()
                            common_edges, common_nodes = association.hard_statistics(mapping)
                            torch.cuda.synchronize()
                            elapsed = time.perf_counter() - started
                            row = {
                                "record_id": key,
                                "training_seed": int(seed),
                                "dataset": dataset,
                                "source_path": source,
                                "pair_key": str(pair.key),
                                "arm": arm,
                                "rounds": int(rounds),
                                "common_edges": int(common_edges),
                                "common_nodes": int(common_nodes),
                                "true_edges": pair.true_edges,
                                "accuracy": (
                                    float(common_edges / pair.true_edges)
                                    if pair.true_edges not in (None, 0)
                                    else None
                                ),
                                "runtime_seconds": float(elapsed),
                                "soft_objective": float(output.objectives[-1]),
                                "row_residual": output.row_residual,
                                "max_column_excess": output.max_column_excess,
                                "checkpoint_sha256": sha256(checkpoint_path),
                            }
                            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                            stream.flush()
                            os.fsync(stream.fileno())
                            completed[key] = row
                            print(
                                f"[{len(completed)}] {dataset} seed={seed} {arm} "
                                f"T={rounds} edges={common_edges}/{pair.true_edges} "
                                f"time={elapsed:.4f}s",
                                flush=True,
                            )

    rows = list(completed.values())
    expected = sum(len(paths) for paths in paths_by_dataset.values())
    expected *= len(config["training_seeds"]) * len(config["arms"]) * len(config["round_budgets"])
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} rows, found {len(rows)}")
    if {str(path.relative_to(ROOT)): sha256(path) for path in CODE_PATHS} != code_hashes:
        raise RuntimeError("evaluation code changed during execution")

    scoped = rows
    cells: dict[str, dict] = {}
    for dataset in config["datasets"]:
        for arm in config["arms"]:
            for rounds in config["round_budgets"]:
                cell_rows = [
                    row for row in scoped
                    if row["dataset"] == dataset
                    and row["arm"] == arm
                    and int(row["rounds"]) == int(rounds)
                ]
                accuracies = [float(row["accuracy"]) for row in cell_rows]
                seed_means = [
                    float(np.mean([
                        float(row["accuracy"]) for row in cell_rows
                        if int(row["training_seed"]) == int(seed)
                    ]))
                    for seed in config["training_seeds"]
                ]
                cells[f"{dataset}|{arm}|{rounds}"] = {
                    "records": len(cell_rows),
                    "pairs_per_seed": len(cell_rows) // len(config["training_seeds"]),
                    "accuracy_percent": 100.0 * float(np.mean(accuracies)),
                    "seed_means_percent": [100.0 * value for value in seed_means],
                    "seed_std_percent": 100.0 * float(np.std(seed_means, ddof=1)),
                    "mean_runtime_seconds": float(np.mean([
                        float(row["runtime_seconds"]) for row in cell_rows
                    ])),
                }
    effects: dict[str, dict] = {}
    for dataset_index, dataset in enumerate(config["datasets"]):
        sources = [str(path) for path in paths_by_dataset[dataset]]
        for rounds in config["round_budgets"]:
            lookup = {
                (int(row["training_seed"]), str(row["source_path"]), str(row["arm"])): row
                for row in scoped
                if row["dataset"] == dataset and int(row["rounds"]) == int(rounds)
            }
            difference = np.asarray([
                [
                    float(lookup[(int(seed), source, "learned")]["accuracy"])
                    - float(lookup[(int(seed), source, "analytic")]["accuracy"])
                    for source in sources
                ]
                for seed in config["training_seeds"]
            ])
            edge_differences = np.asarray([
                int(lookup[(int(seed), source, "learned")]["common_edges"])
                - int(lookup[(int(seed), source, "analytic")]["common_edges"])
                for seed in config["training_seeds"]
                for source in sources
            ])
            rng = np.random.default_rng(20260829 + 97 * dataset_index + int(rounds))
            bootstrap = np.empty(20_000, dtype=np.float64)
            for draw in range(bootstrap.size):
                seed_indices = rng.integers(0, difference.shape[0], difference.shape[0])
                pair_indices = rng.integers(0, difference.shape[1], difference.shape[1])
                bootstrap[draw] = difference[np.ix_(seed_indices, pair_indices)].mean()
            effects[f"{dataset}|{rounds}"] = {
                "learned_minus_analytic_points": 100.0 * float(difference.mean()),
                "ci95_points": [
                    100.0 * float(np.quantile(bootstrap, 0.025)),
                    100.0 * float(np.quantile(bootstrap, 0.975)),
                ],
                "edge_wins_ties_losses": [
                    int((edge_differences > 0).sum()),
                    int((edge_differences == 0).sum()),
                    int((edge_differences < 0).sum()),
                ],
            }
    summary = {
        "protocol_version": config["protocol_version"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": hardware,
        "config_sha256": sha256(CONFIG),
        "parent_config_sha256": sha256(PARENT),
        "code_sha256": code_hashes,
        "raw_sha256": sha256(OUTPUT),
        "native_scope": {"AIDS": 100, "MOLHIV": 91, "MCF-7": 100},
        "cells": cells,
        "paired_effects": effects,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    table = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Independent H200 ENPDA-Core horizon audit.  Both arms share graph pairs, analytic base logits, round budget, and one-shot Hungarian projection; ENPDA additionally applies its learned initializer and dynamics residuals and positive step multipliers.}",
        r"\label{tab:enpda-round-ablation}",
        r"\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{.92}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Dataset & $T$ & Analytic (\%) & ENPDA-Core (\%) & Gain [95\% CI]\\",
        r"\midrule",
    ]
    for dataset in config["datasets"]:
        for position, rounds in enumerate(config["round_budgets"]):
            analytic = cells[f"{dataset}|analytic|{rounds}"]
            learned = cells[f"{dataset}|learned|{rounds}"]
            effect = effects[f"{dataset}|{rounds}"]
            dataset_cell = dataset if position == 0 else ""
            lower, upper = effect["ci95_points"]
            table.append(
                f"{dataset_cell} & {rounds} & {analytic['accuracy_percent']:.2f} & "
                f"{learned['accuracy_percent']:.2f}$\\pm${learned['seed_std_percent']:.2f} & "
                f"{effect['learned_minus_analytic_points']:+.2f} "
                f"[{lower:.2f},{upper:.2f}]\\\\"
            )
        if dataset != config["datasets"][-1]:
            table.append(r"\midrule")
    table.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    OUTPUT_TEX.write_text("\n".join(table) + "\n", encoding="utf-8")
    ATTESTATION.write_text(
        json.dumps({**summary, "status": "complete"}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
