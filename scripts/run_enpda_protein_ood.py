#!/usr/bin/env python3
"""Frozen PROTEINS/ENZYMES extension of the IMDB planted MCES protocol."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch_geometric.datasets import TUDataset

from eval_enpda_nonmolecular_ood_gpu import (
    bootstrap_effect, graph, hardware_guard, load_checkpoint, sha256,
)
from prepare_enpda_nonmolecular_ood import unique_undirected_edges
from nema.association import AssociationGraph
from nema.models.enpda import ENPDAModel

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/enpda_protein_ood.json"
OUT = ROOT / "results/enpda_protein_ood"
PARENT = ROOT / "configs/enpda_sota.json"
CODE = [
    Path(__file__).resolve(),
    ROOT / "scripts/eval_enpda_nonmolecular_ood_gpu.py",
    ROOT / "scripts/prepare_enpda_nonmolecular_ood.py",
    *[ROOT / f"src/nema/{name}.py" for name in
      ("models/enpda", "association", "graph", "rounding")],
]


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def hashes(paths: list[Path]) -> dict:
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def score_mapping(spec: dict, mapping: list[int]) -> int:
    """Independently score the injective map directly on the raw simple graphs."""
    n = spec["num_nodes"]
    if len(mapping) != n or any(type(x) is not int or x < -1 or x >= n for x in mapping):
        raise ValueError("invalid mapping shape or index")
    used = [x for x in mapping if x >= 0]
    if len(used) != len(set(used)):
        raise ValueError("noninjective mapping")
    target = {tuple(edge) for edge in spec["target_edges"]}
    return sum(
        mapping[u] >= 0 and mapping[v] >= 0
        and tuple(sorted((mapping[u], mapping[v]))) in target
        for u, v in spec["source_edges"]
    )


def prepare(config: dict) -> None:
    if (OUT / "FROZEN.json").exists():
        frozen_check()
        print("Retaining already frozen inputs", flush=True)
        return
    for name in config["datasets"]:
        destination = OUT / name / "manifest.json"
        if destination.exists():
            manifest = json.loads(destination.read_text())
            assert manifest["config_sha256"] == sha256(CONFIG), "frozen config changed"
            print(f"retaining frozen manifest: {destination}", flush=True)
            continue
        raw_dir = ROOT / f"data/nonmolecular/{name}/raw"
        if not (raw_dir / f"{name}_A.txt").exists():
            # Avoid PyG's incompatible LocalFileSystem.mv wrapper in this CPU image.
            archive = raw_dir.parent / f"{name}.zip"
            raw_dir.mkdir(parents=True, exist_ok=True)
            url = f"https://www.chrsmrrs.com/graphkerneldatasets/{name}.zip"
            with urllib.request.urlopen(url, timeout=60) as response:
                archive.write_bytes(response.read())
            with zipfile.ZipFile(archive) as files:
                for entry in files.infolist():
                    basename = Path(entry.filename).name
                    if basename.startswith(name + "_") and basename.endswith(".txt"):
                        (raw_dir / basename).write_bytes(files.read(entry))
        dataset = TUDataset(root=str(ROOT / "data/nonmolecular"), name=name)
        candidates = []
        edge_cache = {}
        for index, data in enumerate(dataset):
            edges = unique_undirected_edges(data)
            if config["minimum_nodes"] <= data.num_nodes <= config["maximum_nodes"] and edges:
                candidates.append(index)
                edge_cache[index] = edges
        if len(candidates) < config["num_pairs_per_dataset"]:
            raise RuntimeError(f"{name}: only {len(candidates)} eligible graphs")
        selected = np.asarray(candidates, dtype=np.int64)
        np.random.default_rng(config["selection_seed"]).shuffle(selected)
        pairs = []
        for rank, index in enumerate(selected[:config["num_pairs_per_dataset"]].tolist()):
            n = int(dataset[index].num_nodes)
            edges = edge_cache[index]
            rng = np.random.default_rng(config["selection_seed"] + 104729 * (rank + 1))
            count = max(1, int(round((1 - config["edge_deletion_fraction"]) * len(edges))))
            retained = sorted(int(k) for k in rng.choice(len(edges), count, replace=False))
            permutation = rng.permutation(n).tolist()
            target = sorted(sorted((permutation[edges[k][0]], permutation[edges[k][1]]))
                            for k in retained)
            spec = {
                "pair_id": f"{name}-{index:04d}", "dataset_index": index, "num_nodes": n,
                "source_edges": [list(edge) for edge in edges], "target_edges": target,
                "retained_source_edge_indices": retained,
                "target_permutation_old_to_new": permutation, "exact_optimum_edges": count,
            }
            assert score_mapping(spec, permutation) == count == len(target)
            pairs.append(spec)
        manifest = {
            "created_at_utc": now(), "protocol_version": config["protocol_version"],
            "config_sha256": sha256(CONFIG), "dataset": name,
            "raw_dataset_sha256": hashes(sorted((ROOT / f"data/nonmolecular/{name}/raw").glob("*.txt"))),
            "total_graphs": len(dataset), "eligible_graphs": len(candidates),
            "selected_graphs": len(pairs), "pairs": pairs,
            "selection_depends_only_on_size_nonempty_edges_and_frozen_seed": True,
        }
        write_json(destination, manifest)
        print(json.dumps({k: v for k, v in manifest.items() if k not in ("pairs", "raw_dataset_sha256")}), flush=True)
    write_json(OUT / "FROZEN.json", {
        "frozen_at_utc": now(), "config_sha256": sha256(CONFIG), "code_sha256": hashes(CODE),
        "manifest_sha256": hashes([OUT / name / "manifest.json" for name in config["datasets"]]),
        "checkpoint_sha256": hashes([ROOT / config["checkpoint_pattern"].format(seed=s)
                                      for s in config["training_seeds"]]),
        "parent_config_sha256": sha256(PARENT),
    })


def frozen_check() -> dict:
    frozen = json.loads((OUT / "FROZEN.json").read_text())
    assert frozen["config_sha256"] == sha256(CONFIG)
    assert frozen["parent_config_sha256"] == sha256(PARENT)
    for field in ("code_sha256", "manifest_sha256", "checkpoint_sha256"):
        for name, digest in frozen[field].items():
            assert sha256(ROOT / name) == digest, f"changed frozen input: {name}"
    return frozen


def evaluate(config: dict) -> None:
    frozen = frozen_check()
    hardware = hardware_guard(config)
    torch.set_num_threads(1)
    hardware["torch_num_threads"] = torch.get_num_threads()
    hardware["timing"] = config["timing"]
    write_json(OUT / "hardware.json", hardware)
    started = time.perf_counter()
    for name in config["datasets"]:
        manifest = json.loads((OUT / name / "manifest.json").read_text())
        path = OUT / name / "raw.jsonl"
        done = {}
        if path.exists():
            for line in path.read_text().splitlines():
                row = json.loads(line)
                assert row["record_id"] not in done
                assert row["config_sha256"] == frozen["config_sha256"]
                done[row["record_id"]] = row
        expected = len(manifest["pairs"]) * len(config["training_seeds"]) * 2
        with path.open("a") as stream:
            for seed in config["training_seeds"]:
                ckpt = ROOT / config["checkpoint_pattern"].format(seed=seed)
                payload = load_checkpoint(ckpt)
                assert payload["training_seed"] == seed
                assert payload["config_sha256"] == sha256(PARENT)
                model = ENPDAModel(**payload["model_config"])
                model.load_state_dict(payload["model"], strict=True)
                model.cuda().eval()
                for rank, spec in enumerate(manifest["pairs"]):
                    n = spec["num_nodes"]
                    left = graph(n, spec["source_edges"])
                    right = graph(n, spec["target_edges"])
                    build_start = time.perf_counter()
                    association = AssociationGraph.build(left, right)
                    build_seconds = time.perf_counter() - build_start
                    arms = ["analytic", "learned"] if (rank + seed) % 2 == 0 else ["learned", "analytic"]
                    if rank == 0:
                        with torch.inference_mode():
                            for arm in arms:
                                model(association, mode=arm, iterations=config["rounds"]).hard_mapping()
                        torch.cuda.synchronize()
                    for arm in arms:
                        key = f"{seed}|{spec['pair_id']}|{arm}|T{config['rounds']}"
                        if key in done:
                            continue
                        torch.cuda.synchronize()
                        tick = time.perf_counter()
                        with torch.inference_mode():
                            output = model(association, mode=arm, iterations=config["rounds"])
                            mapping = output.hard_mapping().detach().cpu()
                        common_edges, common_nodes = association.hard_statistics(mapping)
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter() - tick
                        encoded = mapping.tolist()
                        assert score_mapping(spec, encoded) == common_edges
                        optimum = spec["exact_optimum_edges"]
                        assert 0 <= common_edges <= optimum
                        row = {
                            "record_id": key, "dataset": name, "pair_id": spec["pair_id"],
                            "training_seed": seed, "arm": arm, "rounds": config["rounds"],
                            "num_nodes": n, "source_edges": left.num_edges, "target_edges": right.num_edges,
                            "common_edges": common_edges, "common_nodes": common_nodes,
                            "exact_optimum_edges": optimum, "mapping": encoded,
                            "accuracy": common_edges / optimum, "exact_recovery": common_edges == optimum,
                            "missing_edges": optimum - common_edges,
                            "core_seconds": elapsed, "association_build_seconds": build_seconds,
                            "build_plus_core_seconds": build_seconds + elapsed,
                            "checkpoint_sha256": frozen["checkpoint_sha256"][str(ckpt.relative_to(ROOT))],
                            "config_sha256": frozen["config_sha256"],
                            "manifest_sha256": sha256(OUT / name / "manifest.json"),
                        }
                        stream.write(json.dumps(row) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                        done[key] = row
                    if (rank + 1) % 20 == 0:
                        print(f"{name}: {len(done)}/{expected} records; seed={seed}; {time.perf_counter()-started:.1f}s elapsed", flush=True)
                del model
        assert len(done) == expected
    frozen_check()
    write_json(OUT / "EVALUATION_COMPLETE.json", {"completed_at_utc": now(), "hardware": hardware,
               "elapsed_seconds": time.perf_counter() - started})


def finalize(config: dict) -> None:
    frozen = frozen_check()
    hardware = json.loads((OUT / "hardware.json").read_text())
    datasets = {}
    for dataset_index, name in enumerate(config["datasets"]):
        manifest_path = OUT / name / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for path, digest in manifest["raw_dataset_sha256"].items():
            assert sha256(ROOT / path) == digest
        pairs = manifest["pairs"]
        assert len({p["dataset_index"] for p in pairs}) == len(pairs)
        for p in pairs:
            assert score_mapping(p, p["target_permutation_old_to_new"]) == p["exact_optimum_edges"] == len(p["target_edges"])
        records = [json.loads(line) for line in (OUT / name / "raw.jsonl").read_text().splitlines()]
        by_id = {p["pair_id"]: p for p in pairs}
        lookup = {}
        for row in records:
            key = (row["training_seed"], row["pair_id"], row["arm"])
            assert key not in lookup
            spec = by_id[row["pair_id"]]
            assert row["common_edges"] == score_mapping(spec, row["mapping"])
            assert row["accuracy"] == row["common_edges"] / spec["exact_optimum_edges"]
            assert row["exact_recovery"] == (row["common_edges"] == spec["exact_optimum_edges"])
            assert row["missing_edges"] == spec["exact_optimum_edges"] - row["common_edges"]
            assert row["config_sha256"] == frozen["config_sha256"]
            assert row["manifest_sha256"] == sha256(manifest_path)
            ckpt = config["checkpoint_pattern"].format(seed=row["training_seed"])
            assert row["checkpoint_sha256"] == frozen["checkpoint_sha256"][ckpt]
            assert row["rounds"] == config["rounds"]
            assert 0 < row["core_seconds"] <= row["build_plus_core_seconds"]
            lookup[key] = row
        expected = {(s, p["pair_id"], arm) for s in config["training_seeds"] for p in pairs
                    for arm in ("analytic", "learned")}
        assert set(lookup) == expected
        cells = {}
        for arm in ("analytic", "learned"):
            values = [r for r in records if r["arm"] == arm]
            seeds = [100 * np.mean([r["accuracy"] for r in values if r["training_seed"] == s])
                     for s in config["training_seeds"]]
            cells[arm] = {
                "accuracy_percent": 100 * float(np.mean([r["accuracy"] for r in values])),
                "seed_accuracy_percent": seeds, "seed_std_percent": float(np.std(seeds, ddof=1)),
                "exact_recovery_percent": 100 * float(np.mean([r["exact_recovery"] for r in values])),
                "mean_missing_edges": float(np.mean([r["missing_edges"] for r in values])),
                **{f"mean_{field}": float(np.mean([r[field] for r in values])) for field in
                   ("core_seconds", "association_build_seconds", "build_plus_core_seconds")},
                "p95_core_seconds": float(np.quantile([r["core_seconds"] for r in values], .95)),
            }
        effects = {}
        for field in ("accuracy", "exact_recovery"):
            delta = np.asarray([[float(lookup[s, p["pair_id"], "learned"][field])
                                 - float(lookup[s, p["pair_id"], "analytic"][field]) for p in pairs]
                                for s in config["training_seeds"]])
            mean, ci = bootstrap_effect(delta, config["bootstrap_seed"] + dataset_index, config["statistics"]["bootstrap_replicates"])
            effects[field] = {"gain_points": 100 * mean, "ci95_points": [100 * x for x in ci]}
        edge_delta = [lookup[s, p["pair_id"], "learned"]["common_edges"]
                      - lookup[s, p["pair_id"], "analytic"]["common_edges"]
                      for s in config["training_seeds"] for p in pairs]
        effects["edge_wins_ties_losses"] = [sum(d > 0 for d in edge_delta), sum(d == 0 for d in edge_delta), sum(d < 0 for d in edge_delta)]
        sizes = [p["num_nodes"] for p in pairs]
        datasets[name] = {
            "pairs": len(pairs), "records": len(records), "eligible_graphs": manifest["eligible_graphs"],
            "nodes_min_median_max": [min(sizes), float(np.median(sizes)), max(sizes)],
            "cells": cells, "paired_effects": effects,
            "raw_sha256": sha256(OUT / name / "raw.jsonl"),
        }
    summary = {"status": "complete_and_independently_rescored", "completed_at_utc": now(),
               "protocol_version": config["protocol_version"], "frozen_inputs": frozen,
               "hardware": hardware, "datasets": datasets,
               "total_pairs": sum(d["pairs"] for d in datasets.values()),
               "total_records": sum(d["records"] for d in datasets.values())}
    write_json(OUT / "summary.json", summary)
    write_json(OUT / "COMPLETE.json", {"status": summary["status"], "summary_sha256": sha256(OUT / "summary.json"),
                                     "total_pairs": summary["total_pairs"], "total_records": summary["total_records"]})
    lines = ["# Frozen ENPDA protein topology transfer", "",
             "200 distinct source graphs per dataset; 24–64 nodes; universal labels; 30% edge deletion and node permutation. Three frozen molecular checkpoints, four rounds, one Hungarian projection, no refinement or retraining. Accuracy is retained common edges / certified planted optimum; exact means attaining the edge optimum, not recovering a unique permutation.", "",
             "| Dataset | Analytic accuracy | ENPDA accuracy | Paired gain, 95% CI (pp) | Exact analytic / learned | Core ms analytic / learned |", "|---|---:|---:|---|---|---|"]
    for name, data in datasets.items():
        a, b = data["cells"]["analytic"], data["cells"]["learned"]
        effect = data["paired_effects"]["accuracy"]
        lo, hi = effect["ci95_points"]
        lines.append(f"| {name} | {a['accuracy_percent']:.2f}% | {b['accuracy_percent']:.2f}% | {effect['gain_points']:+.2f} [{lo:+.2f}, {hi:+.2f}] | {a['exact_recovery_percent']:.1f}% / {b['exact_recovery_percent']:.1f}% | {1000*a['mean_core_seconds']:.1f} / {1000*b['mean_core_seconds']:.1f} |")
    lines.extend(["", "CIs: 20,000 paired graph bootstraps with independent resampling of three training seeds. Core timing follows the existing IMDB interval (inference, projection, transfer and hard scoring); ACG construction is reported separately in summary.json. Uniform labels deliberately measure topology transfer; original protein attributes are not used. Source graph IDs do not repeat within a dataset. These are controlled planted tasks, separate from the 291 native pairs.", ""])
    (OUT / "report.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "evaluate", "finalize"))
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    {"prepare": prepare, "evaluate": evaluate, "finalize": finalize}[args.stage](config)


if __name__ == "__main__":
    main()
