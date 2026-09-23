"""Fresh, timed reference/teacher/student training on the corrected split."""
from __future__ import annotations

import argparse
import gc
import json
import random
import time
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from nema.association import AssociationGraph
from nema.enpda_components import _dual_step_override
from nema.features import structural_features
from nema.models.enpda import ENPDAModel
from nema.models.nema import NEMAModel
from nema.training import train_nema
from run_enpda_pilot import active_nll, atomic_json, atomic_torch, evaluate, hardware_record, sha256
from run_hard_self_distill_pilot import generate_teachers

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/enpda_graph_disjoint_v1.json"


class ControlledModel(ENPDAModel):
    """Use the same priced analytic reference in both training objectives."""
    price_enabled = True
    analytic_price_enabled = True

    def forward(self, *args, mode="learned", **kwargs):
        enabled = self.analytic_price_enabled if mode == "analytic" else self.price_enabled
        with _dual_step_override(self, None if enabled else 0.0):
            return super().forward(*args, mode=mode, **kwargs)


def prepare_items(rows, device, teachers=None):
    items = []
    for r in rows:
        pair = r["pair"]
        left, right, _ = pair.oriented()
        association = AssociationGraph.build(left, right).to(device)
        item = {"key": r["source_key"], "association": association,
                "features": structural_features(association).to(device),
                "normalizer": max(min(left.num_edges, right.num_edges), 1)}
        if teachers is not None:
            teacher = teachers[item["key"]]
            item["teacher_mapping"] = torch.tensor(teacher["teacher_mapping"], dtype=torch.long)
            item["active_rows"] = torch.tensor(teacher["active_rows"], dtype=torch.long)
        items.append(item)
    return items


def train_student(config, seed, arm, training, validation_rows, teachers, device, checkpoint, provenance):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = ControlledModel(**config["model"]).to(device)
    model.price_enabled = arm == "full"
    settings = config["training"]
    train_items = prepare_items(training, device, teachers)
    assert {x["key"] for x in train_items} == set(teachers)
    validation = {d: prepare_items([r for r in validation_rows if r["dataset"] == d], device)
                  for d in config["datasets"]}
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"])
    budgets = config["validation"]["step_budgets"]
    history = [evaluate(model, validation, budgets, 0, device)]
    best = deepcopy(history[0])
    best_state = deepcopy(model.state_dict())
    order = list(range(len(train_items)))
    rng = random.Random(2026082900 + seed)
    accumulation = settings["gradient_accumulation"]
    print(f"{arm} epoch=0 validation={best['aggregate_normalized_advantage']:+.6f}", flush=True)
    for epoch in range(1, settings["epochs"] + 1):
        rng.shuffle(order)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for position, index in enumerate(order, 1):
            item = train_items[index]
            a, features = item["association"], item["features"]
            mapping, active = item["teacher_mapping"], item["active_rows"]
            with torch.no_grad():
                analytic = model(a, fixed_features=features, mode="analytic")
                analytic_nll = active_nll(analytic.assignment, mapping, active)
            learned = model(a, fixed_features=features, return_trace=True)
            nll = active_nll(learned.assignment, mapping, active)
            intermediate = torch.stack([active_nll(v, mapping, active)
                                        for v in learned.assignment_trace[1:]]).mean()
            margin = torch.relu(nll - analytic_nll + settings["analytic_nll_margin"])
            soft = learned.objectives[-1] / (2 * item["normalizer"])
            capacity = learned.column_excesses.square().mean()
            price_penalty = learned.prices.square().mean()
            if arm == "trained_no_price":
                assert torch.count_nonzero(learned.prices).item() == 0
            loss = (settings["teacher_nll_weight"] * nll
                    + settings["intermediate_nll_weight"] * intermediate
                    + settings["analytic_margin_weight"] * margin
                    - settings["soft_objective_weight"] * soft
                    + settings["capacity_penalty_weight"] * capacity
                    + settings["price_penalty_weight"] * price_penalty)
            assert torch.isfinite(loss).item()
            (loss / accumulation).backward()
            losses.append(float(loss.detach()))
            if position % accumulation == 0 or position == len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip_norm"])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        if epoch in config["validation"]["checkpoint_epochs"]:
            current = evaluate(model, validation, budgets, epoch, device)
            current["mean_training_loss"] = float(np.mean(losses))
            history.append(current)
            if current["aggregate_normalized_advantage"] > best["aggregate_normalized_advantage"]:
                best = deepcopy(current)
                best_state = deepcopy(model.state_dict())
            print(f"{arm} epoch={epoch} loss={np.mean(losses):.5f} validation={current['aggregate_normalized_advantage']:+.6f} selected={best['epoch']}", flush=True)
        else:
            print(f"{arm} epoch={epoch} loss={np.mean(losses):.5f}", flush=True)
    payload = {**provenance, "model_config": config["model"], "model": best_state,
               "experiment_arm": arm, "training_seed": seed,
               "selected_validation": best, "history": history,
               "training_pairs": len(training), "validation_pairs": len(validation_rows),
               "test_inputs_loaded_during_training": False}
    atomic_torch(checkpoint, payload)
    atomic_json(checkpoint.with_suffix(".json"), {k:v for k,v in payload.items() if k != "model"})
    print(f"saved {arm}: epoch {best['epoch']}", flush=True)
    del model, train_items, validation, optimizer, best_state
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


def load_model(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = ControlledModel(**payload["model_config"]).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.price_enabled = payload["experiment_arm"] == "full"
    return model.eval()


@torch.inference_mode()
def evaluate_native(config, seed, test, checkpoints, output, device):
    models = {arm: load_model(p, device) for arm,p in checkpoints.items()}
    with output.open("w") as stream:
        for position, record in enumerate(test, 1):
            pair = record["pair"]
            for arm in config["core_evaluation_arms"]:
                model = models["trained_no_price"] if arm == "trained_no_price" else models["full"]
                model.price_enabled = arm not in ("trained_no_price", "full_prices_removed")
                model.analytic_price_enabled = arm != "analytic_no_price"
                mode = "analytic" if arm.startswith("analytic") else "learned"
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                started = time.perf_counter()
                left, right, swapped = pair.oriented()
                association = AssociationGraph.build(left, right)
                result = model(association, mode=mode)
                mapping = result.hard_mapping().cpu()
                edges, nodes = association.hard_statistics(mapping)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                seconds = time.perf_counter() - started
                assigned = mapping[mapping >= 0].tolist()
                assert len(assigned) == len(set(assigned))
                if arm in ("analytic_no_price", "full_prices_removed", "trained_no_price"):
                    assert torch.count_nonzero(result.prices).item() == 0
                if swapped:
                    inverse = torch.full((pair.left.num_nodes,), -1, dtype=torch.long)
                    for i,j in enumerate(mapping.tolist()):
                        if j >= 0:
                            inverse[j] = i
                    mapping = inverse
                row = {"dataset": record["dataset"], "source_path": record["path"],
                       "pair_index": record["index"], "identities": record["identities"],
                       "arm": arm, "seed": seed, "mapping": mapping.tolist(),
                       "common_edges": edges, "common_nodes": nodes,
                       "true_edges": pair.true_edges, "accuracy": edges / pair.true_edges,
                       "runtime_seconds": seconds, "max_column_excess": result.max_column_excess,
                       "prices": result.prices.cpu().tolist()}
                stream.write(json.dumps(row) + "\n")
            stream.flush()
            if position % 25 == 0 or position == len(test):
                print(f"native {position}/{len(test)}", flush=True)


def run(args):
    pipeline_start = time.perf_counter()
    config = json.loads(CONFIG.read_text())
    manifest_path = ROOT / config["split_manifest"]
    manifest = json.loads(manifest_path.read_text())
    assert all(v == 0 for v in manifest["graph_intersections"].values())
    split_audit = json.loads(manifest_path.with_name("audit.json").read_text())
    assert split_audit["status"] == "passed"
    assert split_audit["manifest_sha256"] == sha256(manifest_path)
    if not args.smoke and not args.device.startswith("cuda"):
        raise RuntimeError("Formal experiment requires H100/H200; use --smoke for CPU diagnostics")
    torch.set_num_threads(4 if args.device.startswith("cuda") else 1)
    hardware = hardware_record(args.device, config)
    out = ROOT / config["outputs"]["root"] / ("smoke" if args.smoke else "formal") / f"seed{args.seed}"
    cp = ROOT / config["outputs"]["checkpoints"] / ("smoke" if args.smoke else "formal") / f"seed{args.seed}"
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Refusing a second writer or unaccounted resume in {out}")
    out.mkdir(parents=True, exist_ok=True)
    cp.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        config["reference_training"]["epochs"] = 1
        config["training"]["epochs"] = 1
        config["validation"]["checkpoint_epochs"] = [0,1]
        config["teacher"]["refinement_passes"] = 1
    code_paths = [Path(__file__), ROOT / "src/nema/training.py", ROOT / "src/nema/models/nema.py",
                  ROOT / "src/nema/models/enpda.py", ROOT / "src/nema/association.py",
                  ROOT / "src/nema/features.py", ROOT / "src/nema/rounding.py",
                  ROOT / "src/nema/graph.py", ROOT / "src/nema/enpda_components.py",
                  ROOT / "scripts/run_enpda_pilot.py", ROOT / "scripts/run_hard_self_distill_pilot.py"]
    provenance = {"protocol_version": config["protocol_version"], "config_sha256": sha256(CONFIG),
                  "manifest_sha256": sha256(manifest_path), "hardware": hardware,
                  "split_audit_sha256": sha256(manifest_path.with_name("audit.json")),
                  "smoke": args.smoke, "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in code_paths}}
    atomic_json(out / "provenance.json", provenance)
    ledger = []

    @contextmanager
    def timed(stage):
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        entry = {"stage": stage, "started_at_utc": datetime.now(timezone.utc).isoformat(), "status": "running"}
        ledger.append(entry)
        atomic_json(out / "timings.json", ledger)
        start = time.perf_counter()
        try:
            yield
            entry["status"] = "complete"
        except BaseException:
            entry["status"] = "failed"
            raise
        finally:
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            entry["elapsed_seconds"] = time.perf_counter() - start
            entry["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            atomic_json(out / "timings.json", ledger)
            print(f"stage={stage} {entry['status']} seconds={entry['elapsed_seconds']:.3f}", flush=True)

    def load(split):
        f = manifest["materialized_files"][split]
        path = ROOT / f["path"]
        assert sha256(path) == f["sha256"]
        rows = torch.load(path, map_location="cpu", weights_only=False)
        if split != "test":
            assert all(r["pair"].true_edges is None and r["pair"].true_nodes is None
                       and r["pair"].true_similarity is None for r in rows)
        if args.smoke:
            rows = [next(r for r in rows if r["dataset"] == d) for d in config["datasets"]]
        return rows

    with timed("reference_pretraining"):
        training = load("train")
        settings = config["reference_training"]
        torch.manual_seed(args.seed)
        reference = NEMAModel(steps=settings["steps"], hidden_dim=settings["hidden_dim"]).to(args.device)
        best = {"score": -float("inf"), "state": None, "epoch": None}

        def select(history, optimizer):
            value = history[-1]["soft_objective"]
            if value > best["score"]:
                best.update(score=value, epoch=history[-1]["epoch"],
                            state={k:v.detach().cpu().clone() for k,v in reference.state_dict().items()})

        history = train_nema(reference, [r["pair"] for r in training], epochs=settings["epochs"],
                             learning_rate=settings["learning_rate"], accumulation=settings["gradient_accumulation"],
                             seed=args.seed, device=args.device, on_epoch=select)
        reference.load_state_dict(best["state"])
        atomic_torch(cp / "reference.best.pt", {**provenance, "model": best["state"],
                     "selected_epoch": best["epoch"], "history": history,
                     "training_keys": [r["source_key"] for r in training]})
    with timed("teacher_generation"):
        items = prepare_items(training, args.device)
        prepared = [(x["key"], x["association"], x["features"], x["normalizer"]) for x in items]
        teacher_config = {**config, "outputs": {"teachers": str(out / "teachers")}}
        teachers, teacher_summary = generate_teachers(reference.eval(), prepared, teacher_config,
                                                      provenance["manifest_sha256"], args.seed)
        del reference, items, prepared, best
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    checkpoints = {}
    for arm in config["arms"]:
        checkpoints[arm] = cp / f"{arm}.best.pt"
        with timed("student_" + arm):
            validation = load("validation")
            train_student(config, args.seed, arm, training, validation, teachers, args.device,
                          checkpoints[arm], {**provenance, "teacher_sha256": teacher_summary["output_sha256"]})
    # The test tensors are first loaded after both validation-selected checkpoints exist.
    with timed("native_evaluation"):
        test = load("test")
        evaluate_native(config, args.seed, test, checkpoints, out / "native.jsonl", args.device)
    atomic_json(out / "COMPLETE.json", {**provenance, "status": "complete",
                "native_sha256": sha256(out / "native.jsonl"),
                "timings_sha256": sha256(out / "timings.json"),
                "checkpoint_sha256": {a: sha256(p) for a,p in checkpoints.items()},
                "native_pairs": len(test), "arms": config["core_evaluation_arms"],
                "pipeline_wall_seconds": time.perf_counter() - pipeline_start,
                "unassigned_setup_seconds": max(0.0, time.perf_counter() - pipeline_start
                                                 - sum(x["elapsed_seconds"] for x in ledger))})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0,1,2), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
