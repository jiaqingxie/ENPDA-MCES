#!/usr/bin/env python3
"""Validation-only hard-search self-distillation for NEMA.

The teacher is generated solely from graph structure and the exact hard MCES
objective on released training pairs.  Native test paths are never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from nema.association import AssociationGraph
from nema.benchmark import load_nema_checkpoint
from nema.data import load_pairs
from nema.features import structural_features
from nema.models.nema import NEMAModel
from nema.rounding import hungarian_mapping, refine_mapping
from scripts.run_unit_anchored_st_pilot import (
    atomic_json,
    atomic_torch,
    canonical_digest,
    checkpoint_score,
    hardware_guard,
    metric_parameter_anchor,
    reset_metric_to_unit,
    sha256,
)


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path("configs/hard_self_distill_pilot.json")
CODE_PATHS = (
    Path("src/nema/models/nema.py"),
    Path("src/nema/sinkhorn.py"),
    Path("src/nema/association.py"),
    Path("src/nema/features.py"),
    Path("src/nema/data.py"),
    Path("src/nema/rounding.py"),
    Path("scripts/run_unit_anchored_st_pilot.py"),
    Path("scripts/run_hard_self_distill_pilot.py"),
)


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not config.get("exploratory_validation_only"):
        raise RuntimeError("self-distillation pilot must remain validation-only")
    if config["training"]["uses_native_test_pairs"]:
        raise RuntimeError("self-distillation pilot must not inspect native tests")
    if config["teacher"]["uses_released_mapping"] or config["teacher"]["uses_released_optimum"]:
        raise RuntimeError("teacher must be generated only from the hard graph objective")
    if config["training"]["uses_ground_truth_correspondence"] or config["training"]["uses_optimum_edge_count"]:
        raise RuntimeError("student training must remain ground-truth-free")
    return config


def active_teacher_rows(association: AssociationGraph, mapping: torch.Tensor) -> torch.Tensor:
    """Rows incident to compatible edges preserved by ``mapping``."""

    mapping = mapping.detach().cpu().long()
    rows, cols = association.shape
    selected = torch.zeros(rows * cols, dtype=torch.bool)
    source_rows = torch.arange(rows)
    valid = (mapping >= 0) & (mapping < cols)
    selected[source_rows[valid] * cols + mapping[valid]] = True
    edge_u = association.edge_u.detach().cpu()
    edge_v = association.edge_v.detach().cpu()
    preserved = selected[edge_u] & selected[edge_v]
    active_candidates = torch.cat((edge_u[preserved], edge_v[preserved]))
    if not active_candidates.numel():
        return torch.empty(0, dtype=torch.long)
    return torch.unique(active_candidates // cols, sorted=True)


def teacher_nll(
    assignment: torch.Tensor,
    mapping: torch.Tensor,
    active_rows: torch.Tensor,
) -> torch.Tensor:
    if not active_rows.numel():
        return assignment.new_zeros(())
    rows = active_rows.to(assignment.device)
    columns = mapping[active_rows].to(assignment.device)
    return -assignment[rows, columns].clamp_min(1e-12).log().mean()


def validation_pass(summary: dict, config: dict) -> bool:
    minimum = float(config["validation"]["pass_min_dataset_mean"])
    required_seeds = int(config["validation"]["pass_positive_seeds_per_dataset"])
    return all(
        values["mean"] > minimum and values["positive_seeds"] >= required_seeds
        for values in summary["datasets"].values()
    )


def prepare(config_path: Path) -> None:
    config = load_config(config_path)
    parent_path = Path(config["parent_split_manifest"])
    if sha256(parent_path) != config["parent_split_manifest_sha256"]:
        raise RuntimeError("parent train/validation split manifest changed")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent.get("native_test_paths_loaded") != []:
        raise RuntimeError("parent split manifest records native-test access")
    target = Path(config["outputs"]["manifest"])
    root = Path(config["outputs"]["root"])
    root.mkdir(parents=True, exist_ok=True)
    if target.exists() or any(path.is_file() for path in root.rglob("*")):
        raise RuntimeError("self-distillation output root is not empty")
    checkpoints = []
    for seed, value in zip(config["training_seeds"], config["base_checkpoints"], strict=True):
        checkpoints.append({"training_seed": seed, "path": value, "sha256": sha256(value)})
    manifest = {
        "protocol_version": config["protocol_version"],
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "config_canonical_sha256": canonical_digest(config),
        "code_sha256": {str(path): sha256(REPO / path) for path in CODE_PATHS},
        "parent_split_manifest": str(parent_path),
        "parent_split_manifest_sha256": sha256(parent_path),
        "splits": parent["splits"],
        "base_checkpoints": checkpoints,
        "native_test_paths_loaded": [],
    }
    atomic_json(target, manifest)
    print(f"froze self-distillation validation manifest: {target}", flush=True)


def load_frozen(config_path: Path) -> tuple[dict, dict, str]:
    config = load_config(config_path)
    target = Path(config["outputs"]["manifest"])
    manifest = json.loads(target.read_text(encoding="utf-8"))
    if manifest["config_sha256"] != sha256(config_path):
        raise RuntimeError("self-distillation config changed after freeze")
    if manifest.get("native_test_paths_loaded") != []:
        raise RuntimeError("self-distillation manifest records native-test access")
    for relative, digest in manifest["code_sha256"].items():
        if sha256(REPO / relative) != digest:
            raise RuntimeError(f"self-distillation code changed: {relative}")
    for checkpoint in manifest["base_checkpoints"]:
        if sha256(checkpoint["path"]) != checkpoint["sha256"]:
            raise RuntimeError(f"base checkpoint changed: {checkpoint['path']}")
    for split in manifest["splits"].values():
        for item in split["training_files"] + split["validation_files"]:
            if sha256(item["path"]) != item["sha256"]:
                raise RuntimeError(f"training source changed: {item['path']}")
    return config, manifest, sha256(target)


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def existing_jsonl(path: Path, manifest_sha: str) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if any(row["manifest_sha256"] != manifest_sha for row in rows):
        raise RuntimeError(f"{path} mixes protocols")
    indexed = {row["key"]: row for row in rows}
    if len(indexed) != len(rows):
        raise RuntimeError(f"{path} contains duplicate teacher keys")
    return indexed


def load_prepared(
    file_items: list[dict],
    device: str,
) -> list[tuple[str, AssociationGraph, torch.Tensor, int]]:
    prepared = []
    for file_item in file_items:
        for index, pair in enumerate(load_pairs(file_item["path"])):
            left, right, _ = pair.oriented()
            association = AssociationGraph.build(left, right).to(device)
            features = structural_features(association).to(device)
            normalizer = max(min(left.num_edges, right.num_edges), 1)
            prepared.append((f"{file_item['path']}#{index}", association, features, normalizer))
    return prepared


@torch.inference_mode()
def generate_teachers(
    model: NEMAModel,
    prepared: list[tuple[str, AssociationGraph, torch.Tensor, int]],
    config: dict,
    manifest_sha: str,
    seed: int,
) -> tuple[dict[str, dict], dict]:
    output_path = Path(config["outputs"]["teachers"]) / f"seed{seed}.jsonl"
    done = existing_jsonl(output_path, manifest_sha)
    settings = config["teacher"]
    for position, (key, association, features, normalizer) in enumerate(prepared, 1):
        if key in done:
            continue
        unit = model(
            association,
            fixed_features=features,
            initializer_mode=settings["initializer_mode"],
            schedule_mode=settings["schedule_mode"],
            metric_mode="unit",
            line_search=False,
            stationary_safeguard=False,
            sinkhorn_tolerance=None,
            iterations=int(settings["unit_mirror_steps"]),
        )
        unit_mapping = hungarian_mapping(unit.assignment)
        teacher_mapping = refine_mapping(
            association,
            unit_mapping,
            max_passes=int(settings["refinement_passes"]),
        )
        unit_edges, _ = association.hard_statistics(unit_mapping)
        teacher_edges, _ = association.hard_statistics(teacher_mapping)
        if teacher_edges < unit_edges:
            raise RuntimeError("hard refinement degraded its unit incumbent")
        active = active_teacher_rows(association, teacher_mapping)
        record = {
            "protocol_version": config["protocol_version"],
            "manifest_sha256": manifest_sha,
            "training_seed": seed,
            "key": key,
            "unit_edges": unit_edges,
            "teacher_edges": teacher_edges,
            "normalized_teacher_gain": (teacher_edges - unit_edges) / normalizer,
            "teacher_mapping": teacher_mapping.tolist(),
            "active_rows": active.tolist(),
        }
        append_jsonl(output_path, record)
        done[key] = record
        if position % 100 == 0 or position == len(prepared):
            print(f"seed={seed} teachers={position}/{len(prepared)}", flush=True)
    if set(done) != {item[0] for item in prepared}:
        raise RuntimeError("teacher key set incomplete")
    values = list(done.values())
    summary = {
        "record_count": len(values),
        "mean_normalized_teacher_gain": float(np.mean([v["normalized_teacher_gain"] for v in values])),
        "improved_pairs": sum(v["teacher_edges"] > v["unit_edges"] for v in values),
        "zero_edge_teachers": sum(not v["active_rows"] for v in values),
        "output": str(output_path),
        "output_sha256": sha256(output_path),
    }
    atomic_json(
        output_path.with_suffix(".completion.json"),
        {"status": "complete", "manifest_sha256": manifest_sha, "training_seed": seed, **summary},
    )
    return done, summary


def hard_edges(association: AssociationGraph, assignment: torch.Tensor) -> int:
    return association.hard_statistics(hungarian_mapping(assignment))[0]


@torch.inference_mode()
def evaluate_validation(
    model: NEMAModel,
    prepared_by_dataset: dict[str, list[tuple[str, AssociationGraph, torch.Tensor, int]]],
    config: dict,
    epoch: int,
) -> dict:
    if epoch == 0:
        return {
            "epoch": 0,
            "aggregate_normalized_advantage": 0.0,
            "datasets": {
                dataset: {"comparisons": len(items) * len(config["validation"]["mirror_step_budgets"]), "mean_normalized_advantage": 0.0}
                for dataset, items in prepared_by_dataset.items()
            },
        }
    model.eval()
    repetitions = int(config["validation"]["repetitions"])
    datasets = {}
    total_advantage = 0.0
    total_count = 0
    for dataset, prepared in prepared_by_dataset.items():
        normalized_advantage = 0.0
        comparisons = 0
        for _, association, features, normalizer in prepared:
            for steps in config["validation"]["mirror_step_budgets"]:
                values = {"learned": [], "unit": []}
                for mode in values:
                    for _ in range(repetitions):
                        output = model(
                            association,
                            fixed_features=features,
                            initializer_mode="learned",
                            schedule_mode="learned",
                            metric_mode=mode,
                            line_search=False,
                            stationary_safeguard=False,
                            sinkhorn_tolerance=None,
                            iterations=int(steps),
                        )
                        values[mode].append(hard_edges(association, output.assignment))
                learned = float(np.median(values["learned"]))
                unit = float(np.median(values["unit"]))
                normalized_advantage += (learned - unit) / normalizer
                comparisons += 1
        mean = normalized_advantage / max(comparisons, 1)
        datasets[dataset] = {"comparisons": comparisons, "mean_normalized_advantage": mean}
        total_advantage += normalized_advantage
        total_count += comparisons
    model.train()
    return {
        "epoch": epoch,
        "aggregate_normalized_advantage": total_advantage / max(total_count, 1),
        "datasets": datasets,
    }


def train(config_path: Path, seed: int) -> None:
    config, manifest, manifest_sha = load_frozen(config_path)
    hardware = hardware_guard(config)
    base = next(item for item in manifest["base_checkpoints"] if item["training_seed"] == seed)
    model = load_nema_checkpoint(base["path"], device="cuda").to("cuda")
    reset_metric_to_unit(model, seed=2026082600 + seed)
    reference = {name: value.detach().clone() for name, value in model.metric.named_parameters()}
    optimizer = torch.optim.Adam(model.metric.parameters(), lr=float(config["training"]["learning_rate"]))

    train_prepared = []
    validation_prepared = {}
    for dataset in config["datasets"]:
        split = manifest["splits"][dataset]
        train_prepared.extend(load_prepared(split["training_files"], "cuda"))
        validation_prepared[dataset] = load_prepared(split["validation_files"], "cuda")
    teachers, teacher_summary = generate_teachers(model, train_prepared, config, manifest_sha, seed)

    training = config["training"]
    budgets = [int(value) for value in training["mirror_step_budgets"]]
    validation_epochs = set(int(value) for value in config["validation"]["checkpoint_epochs"])
    history = [evaluate_validation(model, validation_prepared, config, epoch=0)]
    history[0]["training"] = None
    best = deepcopy(history[0])
    best_state = deepcopy(model.state_dict())
    rng = random.Random(2026082600 + seed)
    order = list(range(len(train_prepared)))
    accumulation = int(training["gradient_accumulation"])
    for epoch in range(1, int(training["epochs"]) + 1):
        rng.shuffle(order)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals = {"distill": 0.0, "margin": 0.0, "soft": 0.0, "anchor": 0.0, "loss": 0.0}
        used = 0
        for position, index in enumerate(order, 1):
            key, association, features, normalizer = train_prepared[index]
            teacher = teachers[key]
            active = torch.tensor(teacher["active_rows"], dtype=torch.long)
            if not active.numel():
                continue
            mapping = torch.tensor(teacher["teacher_mapping"], dtype=torch.long)
            steps = budgets[(position + epoch + seed) % len(budgets)]
            with torch.no_grad():
                unit = model(
                    association,
                    fixed_features=features,
                    initializer_mode="learned",
                    schedule_mode="learned",
                    metric_mode="unit",
                    line_search=False,
                    stationary_safeguard=False,
                    sinkhorn_tolerance=None,
                    iterations=steps,
                )
                unit_nll = teacher_nll(unit.assignment, mapping, active)
            learned = model(
                association,
                fixed_features=features,
                initializer_mode="learned",
                schedule_mode="learned",
                metric_mode="learned",
                line_search=False,
                stationary_safeguard=False,
                sinkhorn_tolerance=None,
                iterations=steps,
            )
            distill = teacher_nll(learned.assignment, mapping, active)
            margin = torch.relu(distill - unit_nll + float(training["unit_nll_margin"]))
            soft = learned.objectives[-1] / (2 * normalizer)
            anchor = metric_parameter_anchor(model, reference)
            loss = (
                float(training["distill_weight"]) * distill
                + float(training["unit_margin_weight"]) * margin
                - float(training["soft_objective_weight"]) * soft
                + float(training["unit_anchor_weight"]) * anchor
            )
            (loss / accumulation).backward()
            used += 1
            for name, value in (("distill", distill), ("margin", margin), ("soft", soft), ("anchor", anchor), ("loss", loss)):
                totals[name] += float(value.detach())
            if used % accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.metric.parameters(), float(training["gradient_clip_norm"]))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        if used % accumulation:
            torch.nn.utils.clip_grad_norm_(model.metric.parameters(), float(training["gradient_clip_norm"]))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        training_record = {name: value / max(used, 1) for name, value in totals.items()}
        training_record["used_pairs"] = used
        if epoch in validation_epochs:
            validation = evaluate_validation(model, validation_prepared, config, epoch)
            validation["training"] = training_record
            history.append(validation)
            if checkpoint_score(validation) > checkpoint_score(best):
                best = deepcopy(validation)
                best_state = deepcopy(model.state_dict())
            print(
                f"seed={seed} epoch={epoch} distill={training_record['distill']:.5f} "
                f"val_adv={validation['aggregate_normalized_advantage']:+.6f} best={best['epoch']}",
                flush=True,
            )
        else:
            print(f"seed={seed} epoch={epoch} distill={training_record['distill']:.5f}", flush=True)

    best_path = Path(config["outputs"]["checkpoints"]) / f"best_seed{seed}.pt"
    atomic_torch(
        best_path,
        {
            "protocol_version": config["protocol_version"],
            "manifest_sha256": manifest_sha,
            "training_seed": seed,
            "base_checkpoint_sha256": base["sha256"],
            "model": best_state,
            "history": history,
            "teacher_summary": teacher_summary,
            "selected_validation": best,
            "metric_parameterization": "bounded log residual around exact unit",
            "uses_native_test_pairs": False,
            "hardware": hardware,
        },
    )
    atomic_json(
        best_path.with_suffix(".completion.json"),
        {
            "status": "complete",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "manifest_sha256": manifest_sha,
            "training_seed": seed,
            "selected_epoch": best["epoch"],
            "selected_validation": best,
            "teacher_summary": teacher_summary,
            "checkpoint": str(best_path),
            "checkpoint_sha256": sha256(best_path),
            "uses_native_test_pairs": False,
            "hardware": hardware,
        },
    )


def finalize(config_path: Path) -> None:
    config, _, manifest_sha = load_frozen(config_path)
    markers = []
    for seed in config["training_seeds"]:
        checkpoint = Path(config["outputs"]["checkpoints"]) / f"best_seed{seed}.pt"
        marker = json.loads(checkpoint.with_suffix(".completion.json").read_text())
        if marker["manifest_sha256"] != manifest_sha or marker["checkpoint_sha256"] != sha256(checkpoint):
            raise RuntimeError(f"self-distillation completion mismatch for seed {seed}")
        if marker.get("uses_native_test_pairs") is not False:
            raise RuntimeError("self-distillation marker lacks validation-only attestation")
        markers.append(marker)
    datasets = {}
    for dataset in config["datasets"]:
        values = [m["selected_validation"]["datasets"][dataset]["mean_normalized_advantage"] for m in markers]
        datasets[dataset] = {
            "seed_values": values,
            "mean": float(np.mean(values)),
            "positive_seeds": sum(value > 0 for value in values),
        }
    summary = {
        "status": "complete",
        "protocol_version": config["protocol_version"],
        "manifest_sha256": manifest_sha,
        "validation_only": True,
        "native_test_paths_loaded": [],
        "selected_epochs": {str(m["training_seed"]): m["selected_epoch"] for m in markers},
        "datasets": datasets,
        "pass_rule": config["validation"]["pass_rule"],
    }
    summary["passed"] = validation_pass(summary, config)
    summary["next_action"] = (
        "freeze separate native isolation before any test access"
        if summary["passed"]
        else "stop neural-gain experiments and reframe the paper"
    )
    atomic_json(Path(config["outputs"]["summary"]), summary)
    print(json.dumps({"passed": summary["passed"], "datasets": datasets}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "train", "finalize"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.config)
    elif args.action == "train":
        if args.seed is None:
            parser.error("train requires --seed")
        train(args.config, args.seed)
    else:
        finalize(args.config)


if __name__ == "__main__":
    main()
