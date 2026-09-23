#!/usr/bin/env python3
"""Validation-only pilot for unit-anchored hard-aligned NEMA training.

This script never loads native test pairs.  It resets only the preconditioner
to its exact M=1 initialization, retains the frozen initializer and schedule
from a NEMA-Soft checkpoint, and selects an epoch on a file-disjoint subset of
the released training pairs using hard one-shot MCES score relative to unit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn

from nema.association import AssociationGraph
from nema.benchmark import load_nema_checkpoint
from nema.data import load_pairs, pair_paths
from nema.features import structural_features
from nema.models.nema import EquivariantMetric, NEMAModel
from nema.rounding import hungarian_mapping


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path("configs/unit_anchored_st_pilot.json")
CODE_PATHS = (
    Path("src/nema/models/nema.py"),
    Path("src/nema/sinkhorn.py"),
    Path("src/nema/association.py"),
    Path("src/nema/features.py"),
    Path("src/nema/data.py"),
    Path("src/nema/rounding.py"),
    Path("scripts/run_unit_anchored_st_pilot.py"),
)


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not config.get("exploratory_validation_only"):
        raise RuntimeError("unit-anchored pilot must remain validation-only")
    training = config["training"]
    if training["uses_native_test_pairs"]:
        raise RuntimeError("pilot must not inspect native test pairs")
    if training["uses_ground_truth_correspondence"] or training["uses_optimum_edge_count"]:
        raise RuntimeError("pilot must remain ground-truth-free")
    if training["mirror_step_budgets"] != config["validation"]["mirror_step_budgets"]:
        raise RuntimeError("training and validation step budgets must match")
    return config


def split_files(config: dict, dataset: str) -> tuple[list[Path], list[Path]]:
    files = list(pair_paths(config["data_root"], dataset, split="train"))
    ranked = sorted(
        files,
        key=lambda path: hashlib.sha256(
            f"{config['split_seed']}|{dataset}|{path}".encode()
        ).hexdigest(),
    )
    validation_count = int(config["validation_file_counts"][dataset])
    if not 0 < validation_count < len(ranked):
        raise RuntimeError(f"invalid validation file count for {dataset}")
    validation = set(ranked[:validation_count])
    return [path for path in files if path not in validation], [path for path in files if path in validation]


class UnitAnchoredMetric(EquivariantMetric):
    """Bounded equivariant residual around the exact unit preconditioner."""

    def __init__(self, input_dim: int, hidden_dim: int, radius: float = math.log(2.0)):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, minimum=0.0)
        self.radius = radius
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        h = self.local(features)
        row = h.mean(dim=-2, keepdim=True).expand_as(h)
        col = h.mean(dim=-3, keepdim=True).expand_as(h)
        glob = h.mean(dim=(-3, -2), keepdim=True).expand_as(h)
        raw = self.output(torch.cat((h, row, col, glob), dim=-1)).squeeze(-1)
        return torch.exp(self.radius * torch.tanh(raw))


def reset_metric_to_unit(model: NEMAModel, seed: int) -> None:
    """Replace a previously trained metric by a fresh exact-unit metric."""

    hidden_dim = int(model.metric.local[0].out_features)
    device = model.initial_weights.device
    torch.manual_seed(seed)
    model.metric = UnitAnchoredMetric(
        input_dim=11,
        hidden_dim=hidden_dim,
    ).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.metric.parameters():
        parameter.requires_grad_(True)


def straight_through_matrix(assignment: torch.Tensor) -> torch.Tensor:
    mapping = hungarian_mapping(assignment.detach().cpu())
    hard = torch.zeros_like(assignment)
    valid = mapping >= 0
    device_valid = valid.to(assignment.device)
    rows = torch.arange(mapping.numel(), device=assignment.device)[device_valid]
    columns = mapping[valid].to(assignment.device)
    hard[rows, columns] = 1.0
    return hard.detach() - assignment.detach() + assignment


def metric_parameter_anchor(model: NEMAModel, reference: dict[str, torch.Tensor]) -> torch.Tensor:
    terms = []
    for name, parameter in model.metric.named_parameters():
        terms.append((parameter - reference[name]).square().mean())
    return torch.stack(terms).mean()


def checkpoint_score(validation: dict) -> tuple[float, int]:
    """Larger is better; ties prefer the earlier epoch."""

    return float(validation["aggregate_normalized_advantage"]), -int(validation["epoch"])


def hardware_guard(config: dict) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("unit-anchored pilot refused: CUDA unavailable")
    name = torch.cuda.get_device_name(0)
    required = config["formal_hardware"]
    if required["accelerator_contains"] not in name or torch.version.cuda != required["cuda_runtime"]:
        raise RuntimeError(
            f"requires H100/CUDA {required['cuda_runtime']}, found {name}/{torch.version.cuda}"
        )
    return {
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": name,
        "capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }


def prepare(config_path: Path) -> None:
    config = load_config(config_path)
    manifest_path = Path(config["outputs"]["manifest"])
    root = Path(config["outputs"]["root"])
    root.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        raise RuntimeError(f"pilot manifest already exists: {manifest_path}")
    existing = [str(path) for path in root.rglob("*") if path.is_file()]
    if existing:
        raise RuntimeError(f"pilot output root is not empty: {existing}")

    splits = {}
    for dataset in config["datasets"]:
        training, validation = split_files(config, dataset)
        all_files = training + validation
        pair_count = sum(len(load_pairs(path)) for path in all_files)
        if pair_count != int(config["expected_training_pairs"][dataset]):
            raise RuntimeError(f"{dataset}: expected training-pair count drift")
        splits[dataset] = {
            "training_files": [
                {"path": str(path), "sha256": sha256(path), "pairs": len(load_pairs(path))}
                for path in training
            ],
            "validation_files": [
                {"path": str(path), "sha256": sha256(path), "pairs": len(load_pairs(path))}
                for path in validation
            ],
        }
    checkpoints = []
    for seed, path_value in zip(
        config["training_seeds"], config["base_checkpoints"], strict=True
    ):
        checkpoints.append(
            {"training_seed": int(seed), "path": path_value, "sha256": sha256(path_value)}
        )
    manifest = {
        "protocol_version": config["protocol_version"],
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "config_canonical_sha256": canonical_digest(config),
        "code_sha256": {str(path): sha256(REPO / path) for path in CODE_PATHS},
        "base_checkpoints": checkpoints,
        "splits": splits,
        "native_test_paths_loaded": [],
    }
    atomic_json(manifest_path, manifest)
    print(f"froze validation-only pilot manifest: {manifest_path}", flush=True)


def load_frozen(config_path: Path) -> tuple[dict, dict, str]:
    config = load_config(config_path)
    manifest_path = Path(config["outputs"]["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["config_sha256"] != sha256(config_path):
        raise RuntimeError("pilot config changed after manifest freeze")
    if manifest.get("native_test_paths_loaded") != []:
        raise RuntimeError("pilot manifest records native-test access")
    for relative, digest in manifest["code_sha256"].items():
        if sha256(REPO / relative) != digest:
            raise RuntimeError(f"pilot code changed after freeze: {relative}")
    for checkpoint in manifest["base_checkpoints"]:
        if sha256(checkpoint["path"]) != checkpoint["sha256"]:
            raise RuntimeError(f"base checkpoint changed: {checkpoint['path']}")
    for split in manifest["splits"].values():
        for item in split["training_files"] + split["validation_files"]:
            if sha256(item["path"]) != item["sha256"]:
                raise RuntimeError(f"training source changed: {item['path']}")
    return config, manifest, sha256(manifest_path)


def prepare_pairs(items: list[dict], device: str) -> list[tuple[str, AssociationGraph, torch.Tensor, int]]:
    prepared = []
    for file_item in items:
        for index, pair in enumerate(load_pairs(file_item["path"])):
            left, right, _ = pair.oriented()
            association = AssociationGraph.build(left, right).to(device)
            features = structural_features(association).to(device)
            normalizer = max(min(left.num_edges, right.num_edges), 1)
            prepared.append((f"{file_item['path']}#{index}", association, features, normalizer))
    return prepared


def hard_edges(association: AssociationGraph, assignment: torch.Tensor) -> int:
    mapping = hungarian_mapping(assignment.detach().cpu())
    return association.hard_statistics(mapping)[0]


@torch.inference_mode()
def validate(
    model: NEMAModel,
    prepared_by_dataset: dict[str, list[tuple[str, AssociationGraph, torch.Tensor, int]]],
    budgets: list[int],
    epoch: int,
) -> dict:
    model.eval()
    datasets = {}
    aggregate_advantage = 0.0
    aggregate_count = 0
    for dataset, prepared in prepared_by_dataset.items():
        learned_edges = unit_edges = 0
        normalized_advantage = 0.0
        comparisons = 0
        for _, association, features, normalizer in prepared:
            for steps in budgets:
                learned = model(
                    association,
                    fixed_features=features,
                    initializer_mode="learned",
                    schedule_mode="learned",
                    line_search=False,
                    metric_mode="learned",
                    stationary_safeguard=False,
                    sinkhorn_tolerance=None,
                    iterations=steps,
                )
                unit = model(
                    association,
                    fixed_features=features,
                    initializer_mode="learned",
                    schedule_mode="learned",
                    line_search=False,
                    metric_mode="unit",
                    stationary_safeguard=False,
                    sinkhorn_tolerance=None,
                    iterations=steps,
                )
                learned_value = hard_edges(association, learned.assignment)
                unit_value = hard_edges(association, unit.assignment)
                learned_edges += learned_value
                unit_edges += unit_value
                normalized_advantage += (learned_value - unit_value) / normalizer
                comparisons += 1
        dataset_advantage = normalized_advantage / max(comparisons, 1)
        datasets[dataset] = {
            "comparisons": comparisons,
            "learned_edges": learned_edges,
            "unit_edges": unit_edges,
            "mean_normalized_advantage": dataset_advantage,
        }
        aggregate_advantage += normalized_advantage
        aggregate_count += comparisons
    model.train()
    return {
        "epoch": epoch,
        "aggregate_normalized_advantage": aggregate_advantage / max(aggregate_count, 1),
        "datasets": datasets,
    }


def train(config_path: Path, seed: int) -> None:
    config, manifest, manifest_sha = load_frozen(config_path)
    hardware = hardware_guard(config)
    base = next(
        item for item in manifest["base_checkpoints"] if item["training_seed"] == seed
    )
    model = load_nema_checkpoint(base["path"], device="cuda").to("cuda")
    reset_metric_to_unit(model, seed=2026082500 + seed)
    reference = {
        name: parameter.detach().clone() for name, parameter in model.metric.named_parameters()
    }
    optimizer = torch.optim.Adam(
        model.metric.parameters(), lr=float(config["training"]["learning_rate"])
    )
    train_prepared = []
    validation_prepared = {}
    for dataset in config["datasets"]:
        split = manifest["splits"][dataset]
        train_prepared.extend(prepare_pairs(split["training_files"], "cuda"))
        validation_prepared[dataset] = prepare_pairs(split["validation_files"], "cuda")
    expected_total = sum(config["expected_training_pairs"].values())
    if len(train_prepared) + sum(map(len, validation_prepared.values())) != expected_total:
        raise RuntimeError("prepared pilot pair count drift")

    training = config["training"]
    budgets = [int(value) for value in training["mirror_step_budgets"]]
    history = [validate(model, validation_prepared, budgets, epoch=0)]
    history[0]["training"] = None
    best = deepcopy(history[0])
    best_state = deepcopy(model.state_dict())
    rng = random.Random(2026082500 + seed)
    order = list(range(len(train_prepared)))
    accumulation = int(training["gradient_accumulation"])
    epochs = int(training["epochs"])
    for epoch in range(1, epochs + 1):
        rng.shuffle(order)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_hard = total_soft = total_anchor = total_loss = 0.0
        for position, index in enumerate(order, 1):
            _, association, features, normalizer = train_prepared[index]
            steps = budgets[(position + epoch + seed) % len(budgets)]
            output = model(
                association,
                fixed_features=features,
                initializer_mode="learned",
                schedule_mode="learned",
                line_search=False,
                metric_mode="learned",
                stationary_safeguard=False,
                sinkhorn_tolerance=None,
                iterations=steps,
            )
            hard = association.objective(straight_through_matrix(output.assignment)) / (2 * normalizer)
            soft = output.objectives[-1] / (2 * normalizer)
            anchor = metric_parameter_anchor(model, reference)
            raw_loss = (
                -float(training["hard_weight"]) * hard
                -float(training["soft_weight"]) * soft
                +float(training["unit_anchor_weight"]) * anchor
            )
            (raw_loss / accumulation).backward()
            total_hard += float(hard.detach())
            total_soft += float(soft.detach())
            total_anchor += float(anchor.detach())
            total_loss += float(raw_loss.detach())
            if position % accumulation == 0 or position == len(order):
                torch.nn.utils.clip_grad_norm_(
                    model.metric.parameters(), float(training["gradient_clip_norm"])
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        validation = validate(model, validation_prepared, budgets, epoch=epoch)
        validation["training"] = {
            "hard_objective": total_hard / len(order),
            "soft_objective": total_soft / len(order),
            "unit_anchor": total_anchor / len(order),
            "loss": total_loss / len(order),
        }
        history.append(validation)
        if checkpoint_score(validation) > checkpoint_score(best):
            best = deepcopy(validation)
            best_state = deepcopy(model.state_dict())
        latest_path = Path(config["outputs"]["checkpoints"]) / f"latest_seed{seed}.pt"
        atomic_torch(
            latest_path,
            {
                "protocol_version": config["protocol_version"],
                "manifest_sha256": manifest_sha,
                "training_seed": seed,
                "base_checkpoint_sha256": base["sha256"],
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
                "hardware": hardware,
            },
        )
        print(
            f"seed={seed} epoch={epoch}/{epochs} "
            f"train_hard={validation['training']['hard_objective']:.6f} "
            f"val_adv={validation['aggregate_normalized_advantage']:+.6f} "
            f"best_epoch={best['epoch']}",
            flush=True,
        )

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
            "selected_validation": best,
            "selection_metric": training["checkpoint_selection"],
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
            "checkpoint": str(best_path),
            "checkpoint_sha256": sha256(best_path),
            "uses_native_test_pairs": False,
            "hardware": hardware,
        },
    )


def finalize(config_path: Path) -> None:
    config, _, manifest_sha = load_frozen(config_path)
    records = []
    for seed in config["training_seeds"]:
        path = Path(config["outputs"]["checkpoints"]) / f"best_seed{seed}.pt"
        marker = json.loads(path.with_suffix(".completion.json").read_text(encoding="utf-8"))
        if marker["manifest_sha256"] != manifest_sha or marker["checkpoint_sha256"] != sha256(path):
            raise RuntimeError(f"pilot completion mismatch for seed {seed}")
        if marker.get("uses_native_test_pairs") is not False:
            raise RuntimeError("pilot completion does not attest validation-only execution")
        records.append(marker)
    datasets = {}
    for dataset in config["datasets"]:
        values = [
            record["selected_validation"]["datasets"][dataset]["mean_normalized_advantage"]
            for record in records
        ]
        datasets[dataset] = {"seed_values": values, "mean": sum(values) / len(values)}
    passed = all(item["mean"] > 0.0 for item in datasets.values())
    atomic_json(
        Path(config["outputs"]["summary"]),
        {
            "status": "complete",
            "protocol_version": config["protocol_version"],
            "manifest_sha256": manifest_sha,
            "validation_only": True,
            "native_test_paths_loaded": [],
            "selected_epochs": {str(r["training_seed"]): r["selected_epoch"] for r in records},
            "datasets": datasets,
            "pass_rule": config["validation"]["pass_rule"],
            "passed": passed,
            "next_action": (
                "freeze one native-test protocol before any test access"
                if passed
                else "do not access native test; move to hard-search self-distillation"
            ),
        },
    )
    print(json.dumps({"passed": passed, "datasets": datasets}, indent=2), flush=True)


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
