#!/usr/bin/env python3
"""Validation-only training pilot for equivariant neural primal-dual assignment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from nema.association import AssociationGraph
from nema.data import load_pairs
from nema.features import structural_features
from nema.models.enpda import ENPDAModel


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path("configs/enpda_pilot.json")
CODE_PATHS = (
    Path("src/nema/models/enpda.py"),
    Path("src/nema/association.py"),
    Path("src/nema/features.py"),
    Path("src/nema/rounding.py"),
    Path("scripts/run_enpda_pilot.py"),
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_torch(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    temporary.replace(path)


def active_nll(
    assignment: torch.Tensor,
    mapping: torch.Tensor,
    active_rows: torch.Tensor,
) -> torch.Tensor:
    if not active_rows.numel():
        return assignment.new_zeros(())
    rows = active_rows.to(assignment.device)
    columns = mapping[active_rows].to(assignment.device)
    return -assignment[rows, columns].clamp_min(1e-12).log().mean()


def hardware_record(device: str, config: dict) -> dict:
    record = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": device,
    }
    if device.startswith("cuda"):
        record["accelerator"] = torch.cuda.get_device_name(0)
        allowed = config["hardware"]["accelerator_any"]
        if not any(name in record["accelerator"] for name in allowed):
            raise RuntimeError(f"pilot requires one of {allowed}, found {record['accelerator']}")
        expected = str(config["hardware"]["cuda_runtime_prefix"])
        if not str(torch.version.cuda).startswith(expected):
            raise RuntimeError(f"pilot requires CUDA {expected}, found {torch.version.cuda}")
    return record


def load_teachers(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    teachers = {row["key"]: row for row in rows}
    if len(teachers) != len(rows):
        raise RuntimeError(f"duplicate teacher keys in {path}")
    return teachers


def load_dataset_items(
    file_items: list[dict],
    device: str,
    limit: int,
    selection_seed: int,
    teachers: dict[str, dict] | None = None,
) -> list[dict]:
    raw_items = []
    for file_item in file_items:
        for index, pair in enumerate(load_pairs(file_item["path"])):
            key = f"{file_item['path']}#{index}"
            rank = hashlib.sha256(f"{selection_seed}:{key}".encode()).hexdigest()
            raw_items.append((rank, key, pair))
    raw_items.sort(key=lambda value: value[0])

    items: list[dict] = []
    for _, key, pair in raw_items[:limit]:
        left, right, _ = pair.oriented()
        association = AssociationGraph.build(left, right).to(device)
        item = {
            "key": key,
            "association": association,
            "features": structural_features(association).to(device),
            "normalizer": max(min(left.num_edges, right.num_edges), 1),
            "true_edges": pair.true_edges,
        }
        if teachers is not None:
            teacher = teachers.get(key)
            if teacher is None:
                raise RuntimeError(f"missing teacher for {key}")
            item["teacher_mapping"] = torch.tensor(
                teacher["teacher_mapping"], dtype=torch.long
            )
            item["active_rows"] = torch.tensor(teacher["active_rows"], dtype=torch.long)
        items.append(item)
    return items


def hard_edges(item: dict, output) -> int:
    return item["association"].hard_statistics(output.hard_mapping())[0]


@torch.inference_mode()
def evaluate(
    model: ENPDAModel,
    validation: dict[str, list[dict]],
    budgets: list[int],
    epoch: int,
    device: str,
) -> dict:
    model.eval()
    datasets: dict[str, dict] = {}
    aggregate_advantages: list[float] = []
    for dataset, items in validation.items():
        budget_rows = {}
        dataset_advantages: list[float] = []
        for steps in budgets:
            learned_edges, analytic_edges, learned_times, analytic_times = [], [], [], []
            excesses = []
            for item in items:
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                start = time.perf_counter()
                learned = model(
                    item["association"],
                    fixed_features=item["features"],
                    mode="learned",
                    iterations=steps,
                )
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                learned_times.append(time.perf_counter() - start)

                start = time.perf_counter()
                analytic = model(
                    item["association"],
                    fixed_features=item["features"],
                    mode="analytic",
                    iterations=steps,
                )
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                analytic_times.append(time.perf_counter() - start)
                learned_edges.append(hard_edges(item, learned))
                analytic_edges.append(hard_edges(item, analytic))
                excesses.append(learned.max_column_excess)

            advantages = [
                (learned - analytic) / item["normalizer"]
                for learned, analytic, item in zip(
                    learned_edges, analytic_edges, items, strict=True
                )
            ]
            dataset_advantages.extend(advantages)
            aggregate_advantages.extend(advantages)
            budget_rows[str(steps)] = {
                "pairs": len(items),
                "learned_mean_edges": float(np.mean(learned_edges)),
                "analytic_mean_edges": float(np.mean(analytic_edges)),
                "mean_normalized_advantage": float(np.mean(advantages)),
                "positive_pairs": int(sum(value > 0 for value in advantages)),
                "negative_pairs": int(sum(value < 0 for value in advantages)),
                "learned_mean_seconds": float(np.mean(learned_times)),
                "analytic_mean_seconds": float(np.mean(analytic_times)),
                "learned_mean_column_excess": float(np.mean(excesses)),
            }
        datasets[dataset] = {
            "mean_normalized_advantage": float(np.mean(dataset_advantages)),
            "positive_budgets": int(
                sum(row["mean_normalized_advantage"] > 0 for row in budget_rows.values())
            ),
            "budgets": budget_rows,
        }
    return {
        "epoch": epoch,
        "aggregate_normalized_advantage": float(np.mean(aggregate_advantages)),
        "datasets": datasets,
    }


def passes_gate(validation: dict, config: dict) -> bool:
    required_budgets = 2
    return validation["aggregate_normalized_advantage"] > 0 and all(
        value["mean_normalized_advantage"] > 0
        and value["positive_budgets"] >= required_budgets
        for value in validation["datasets"].values()
    )


def run(config_path: Path, seed: int, device: str, smoke: bool) -> None:
    config = json.loads(config_path.read_text())
    if not config.get("exploratory_validation_only"):
        raise RuntimeError("ENPDA pilot must remain validation-only")
    parent_path = Path(config["parent_split_manifest"])
    parent = json.loads(parent_path.read_text())
    if parent.get("native_test_paths_loaded") != []:
        raise RuntimeError("parent split manifest accessed native tests")
    teacher_path = Path(config["teacher_pattern"].format(seed=seed))
    teachers = load_teachers(teacher_path)
    settings = dict(config["training"])
    validation_settings = dict(config["validation"])
    if smoke:
        settings.update(
            {
                "epochs": config["smoke"]["epochs"],
                "train_pairs_per_dataset": config["smoke"]["train_pairs_per_dataset"],
            }
        )
        validation_settings.update(
            {
                "pairs_per_dataset": config["smoke"]["pairs_per_dataset"],
                "step_budgets": config["smoke"]["step_budgets"],
                "checkpoint_epochs": [0, int(config["smoke"]["epochs"])],
            }
        )

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    hardware = hardware_record(device, config)
    train_items: list[dict] = []
    validation: dict[str, list[dict]] = {}
    print("preparing ENPDA train/validation ACGs", flush=True)
    for dataset in config["datasets"]:
        split = parent["splits"][dataset]
        dataset_train = load_dataset_items(
            split["training_files"],
            device,
            int(settings["train_pairs_per_dataset"]),
            selection_seed=2026082900 + seed,
            teachers=teachers,
        )
        dataset_validation = load_dataset_items(
            split["validation_files"],
            device,
            int(validation_settings["pairs_per_dataset"]),
            selection_seed=2026082990 + seed,
        )
        train_items.extend(dataset_train)
        validation[dataset] = dataset_validation
        print(
            f"prepared {dataset}: train={len(dataset_train)} validation={len(dataset_validation)}",
            flush=True,
        )

    model = ENPDAModel(**config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    budgets = [int(value) for value in validation_settings["step_budgets"]]
    checkpoint_epochs = set(int(value) for value in validation_settings["checkpoint_epochs"])
    history = [evaluate(model, validation, budgets, 0, device)]
    best = deepcopy(history[0])
    best_state = deepcopy(model.state_dict())
    print(
        f"epoch=0 val_adv={best['aggregate_normalized_advantage']:+.6f}", flush=True
    )

    order = list(range(len(train_items)))
    rng = random.Random(2026082900 + seed)
    accumulation = int(settings["gradient_accumulation"])
    for epoch in range(1, int(settings["epochs"]) + 1):
        rng.shuffle(order)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals = {"nll": 0.0, "intermediate": 0.0, "margin": 0.0, "soft": 0.0, "capacity": 0.0, "loss": 0.0}
        used = 0
        for index in order:
            item = train_items[index]
            association = item["association"]
            mapping = item["teacher_mapping"]
            active = item["active_rows"]
            with torch.no_grad():
                analytic = model(
                    association,
                    fixed_features=item["features"],
                    mode="analytic",
                    return_trace=True,
                )
                analytic_nll = active_nll(analytic.assignment, mapping, active)
            learned = model(
                association,
                fixed_features=item["features"],
                mode="learned",
                return_trace=True,
            )
            nll = active_nll(learned.assignment, mapping, active)
            assert learned.assignment_trace is not None
            intermediate = torch.stack(
                [active_nll(value, mapping, active) for value in learned.assignment_trace[1:]]
            ).mean()
            margin = torch.relu(nll - analytic_nll + float(settings["analytic_nll_margin"]))
            soft = learned.objectives[-1] / (2 * item["normalizer"])
            capacity = learned.column_excesses.square().mean()
            price_penalty = learned.prices.square().mean()
            loss = (
                float(settings["teacher_nll_weight"]) * nll
                + float(settings["intermediate_nll_weight"]) * intermediate
                + float(settings["analytic_margin_weight"]) * margin
                - float(settings["soft_objective_weight"]) * soft
                + float(settings["capacity_penalty_weight"]) * capacity
                + float(settings["price_penalty_weight"]) * price_penalty
            )
            (loss / accumulation).backward()
            used += 1
            for name, value in (
                ("nll", nll),
                ("intermediate", intermediate),
                ("margin", margin),
                ("soft", soft),
                ("capacity", capacity),
                ("loss", loss),
            ):
                totals[name] += float(value.detach())
            if used % accumulation == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(settings["gradient_clip_norm"])
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        if used % accumulation:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(settings["gradient_clip_norm"])
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        training = {name: value / max(used, 1) for name, value in totals.items()}
        if epoch in checkpoint_epochs:
            current = evaluate(model, validation, budgets, epoch, device)
            current["training"] = training
            history.append(current)
            if current["aggregate_normalized_advantage"] > best["aggregate_normalized_advantage"]:
                best = deepcopy(current)
                best_state = deepcopy(model.state_dict())
            print(
                f"epoch={epoch} loss={training['loss']:.5f} "
                f"val_adv={current['aggregate_normalized_advantage']:+.6f} "
                f"best_epoch={best['epoch']}",
                flush=True,
            )
        else:
            print(f"epoch={epoch} loss={training['loss']:.5f}", flush=True)

    output_root = Path(config["outputs"]["smoke_root"] if smoke else config["outputs"]["root"])
    checkpoint = (
        output_root / f"seed{seed}.best.pt"
        if smoke
        else Path(config["outputs"]["checkpoint_pattern"].format(seed=seed))
    )
    result_path = (
        output_root / f"seed{seed}.json"
        if smoke
        else Path(config["outputs"]["result_pattern"].format(seed=seed))
    )
    provenance = {
        "protocol_version": config["protocol_version"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "parent_manifest": str(parent_path),
        "parent_manifest_sha256": sha256(parent_path),
        "teacher": str(teacher_path),
        "teacher_sha256": sha256(teacher_path),
        "code_sha256": {str(path): sha256(ROOT / path) for path in CODE_PATHS},
        "seed": seed,
        "smoke": smoke,
        "native_test_paths_loaded": [],
        "hardware": hardware,
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
    result = {
        **provenance,
        "selected_epoch": best["epoch"],
        "selected_validation": best,
        "history": history,
        "passed_validation_gate": passes_gate(best, config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
    }
    atomic_json(result_path, result)
    print(json.dumps({
        "selected_epoch": best["epoch"],
        "advantage": best["aggregate_normalized_advantage"],
        "passed": result["passed_validation_gate"],
        "result": str(result_path),
    }, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.config, args.seed, args.device, args.smoke)


if __name__ == "__main__":
    main()
