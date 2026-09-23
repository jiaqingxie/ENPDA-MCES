#!/usr/bin/env python3
"""Train a shared equivariant Sinkhorn affinity control without native-test access."""

from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from nema.models.sinkhorn_baseline import TrainOnceSinkhorn
from run_enpda_pilot import (
    ROOT,
    active_nll,
    atomic_json,
    atomic_torch,
    hardware_record,
    load_dataset_items,
    load_teachers,
    sha256,
)


DEFAULT_CONFIG = Path("configs/train_once_sinkhorn.json")
CODE_PATHS = (
    Path("src/nema/models/sinkhorn_baseline.py"),
    Path("src/nema/models/enpda.py"),
    Path("src/nema/sinkhorn.py"),
    Path("src/nema/association.py"),
    Path("src/nema/features.py"),
    Path("src/nema/rounding.py"),
    Path("scripts/train_once_sinkhorn.py"),
)


def split_count(items: list[dict]) -> int:
    values = [item.get("pairs") for item in items]
    if any(value is None for value in values):
        raise RuntimeError("split manifest is missing pair counts")
    return sum(int(value) for value in values)


@torch.inference_mode()
def evaluate(model: TrainOnceSinkhorn, validation: dict[str, list[dict]], epoch: int) -> dict:
    model.eval()
    datasets = {}
    aggregate = []
    for dataset, items in validation.items():
        values = []
        edges = []
        for item in items:
            output = model(item["association"], fixed_features=item["features"], learned=True)
            common_edges = item["association"].hard_statistics(output.hard_mapping())[0]
            normalized = common_edges / item["normalizer"]
            values.append(normalized)
            edges.append(common_edges)
            aggregate.append(normalized)
        datasets[dataset] = {
            "pairs": len(items),
            "mean_normalized_hard_edges": float(np.mean(values)),
            "mean_hard_edges": float(np.mean(edges)),
        }
    return {
        "epoch": epoch,
        "aggregate_mean_normalized_hard_edges": float(np.mean(aggregate)),
        "datasets": datasets,
    }


def train(config_path: Path, seed: int, device: str, smoke: bool) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not config.get("frozen_before_native_results"):
        raise RuntimeError("train-once Sinkhorn protocol is not frozen")
    if seed not in config["training_seeds"]:
        raise ValueError(f"seed {seed} is outside the frozen seed set")
    parent_path = Path(config["parent_split_manifest"])
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent.get("native_test_paths_loaded") != []:
        raise RuntimeError("parent split manifest accessed native tests")
    teacher_path = Path(config["teacher_pattern"].format(seed=seed))
    teachers = load_teachers(teacher_path)

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    hardware = hardware_record(device, config)
    settings = config["training"]
    train_items = []
    validation = {}
    split_counts = {}
    for dataset in config["datasets"]:
        split = parent["splits"][dataset]
        train_n = split_count(split["training_files"])
        val_n = split_count(split["validation_files"])
        if smoke:
            train_n = min(train_n, int(config["smoke"]["train_pairs_per_dataset"]))
            val_n = min(val_n, int(config["smoke"]["validation_pairs_per_dataset"]))
        dataset_train = load_dataset_items(
            split["training_files"],
            device,
            train_n,
            selection_seed=2026083000 + seed,
            teachers=teachers,
        )
        dataset_validation = load_dataset_items(
            split["validation_files"],
            device,
            val_n,
            selection_seed=2026083090 + seed,
        )
        train_items.extend(dataset_train)
        validation[dataset] = dataset_validation
        split_counts[dataset] = {"training": train_n, "validation": val_n}
        print(f"prepared {dataset}: train={train_n} validation={val_n}", flush=True)
    if not smoke and set(item["key"] for item in train_items) != set(teachers):
        raise RuntimeError("teacher keys do not exactly match the formal training split")

    model = TrainOnceSinkhorn(**config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    checkpoint_epochs = set(int(value) for value in settings["checkpoint_epochs"])
    epochs = int(config["smoke"]["epochs"] if smoke else settings["epochs"])
    if smoke:
        checkpoint_epochs = {0, epochs}
    history = [evaluate(model, validation, 0)]
    best = deepcopy(history[0])
    best_state = deepcopy(model.state_dict())
    order = list(range(len(train_items)))
    rng = random.Random(2026083000 + seed)
    accumulation = int(settings["gradient_accumulation"])

    for epoch in range(1, epochs + 1):
        rng.shuffle(order)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals = {"nll": 0.0, "margin": 0.0, "soft": 0.0, "loss": 0.0}
        used = 0
        for index in order:
            item = train_items[index]
            with torch.no_grad():
                reference = model(
                    item["association"], fixed_features=item["features"], learned=False
                )
                reference_nll = active_nll(
                    reference.assignment, item["teacher_mapping"], item["active_rows"]
                )
            output = model(item["association"], fixed_features=item["features"], learned=True)
            nll = active_nll(output.assignment, item["teacher_mapping"], item["active_rows"])
            margin = torch.relu(
                nll - reference_nll + float(settings["analytic_nll_margin"])
            )
            soft = output.objective / (2 * item["normalizer"])
            loss = (
                float(settings["teacher_nll_weight"]) * nll
                + float(settings["analytic_margin_weight"]) * margin
                - float(settings["soft_objective_weight"]) * soft
            )
            (loss / accumulation).backward()
            used += 1
            for name, value in (("nll", nll), ("margin", margin), ("soft", soft), ("loss", loss)):
                totals[name] += float(value.detach())
            if used % accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings["gradient_clip_norm"]))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        if used % accumulation:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings["gradient_clip_norm"]))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        training = {key: value / max(used, 1) for key, value in totals.items()}
        if epoch in checkpoint_epochs:
            current = evaluate(model, validation, epoch)
            current["training"] = training
            history.append(current)
            if (
                current["aggregate_mean_normalized_hard_edges"]
                > best["aggregate_mean_normalized_hard_edges"]
            ):
                best = deepcopy(current)
                best_state = deepcopy(model.state_dict())
            print(
                f"epoch={epoch} loss={training['loss']:.6f} "
                f"val={current['aggregate_mean_normalized_hard_edges']:.6f} "
                f"best_epoch={best['epoch']}",
                flush=True,
            )
        else:
            print(f"epoch={epoch} loss={training['loss']:.6f}", flush=True)

    suffix = ".smoke" if smoke else ""
    checkpoint = Path(config["outputs"]["checkpoint_pattern"].format(seed=seed) + suffix)
    result_path = Path(config["outputs"]["training_result_pattern"].format(seed=seed) + suffix)
    provenance = {
        "protocol_version": config["protocol_version"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "parent_manifest_sha256": sha256(parent_path),
        "teacher_sha256": sha256(teacher_path),
        "code_sha256": {str(path): sha256(ROOT / path) for path in CODE_PATHS},
        "training_seed": seed,
        "split_counts": split_counts,
        "native_test_paths_loaded": [],
        "hardware": hardware,
        "smoke": smoke,
    }
    atomic_torch(
        checkpoint,
        {
            **provenance,
            "model_config": config["model"],
            "model": best_state,
            "selected_validation": best,
            "history": history,
        },
    )
    atomic_json(
        result_path,
        {
            **provenance,
            "selected_epoch": best["epoch"],
            "selected_validation": best,
            "history": history,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
        },
    )
    print(json.dumps({"checkpoint": str(checkpoint), "result": str(result_path), "best": best}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    train(args.config, args.seed, args.device, args.smoke)


if __name__ == "__main__":
    main()
