"""Freeze a globally graph-disjoint split while retaining all 291 native tests."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import replace
from pathlib import Path

import torch

from nema.data import load_pairs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/enpda_graph_disjoint_v1"
SEED = "enpda-graph-disjoint-20260917-v1"
DATASETS = ("AIDS", "MOLHIV", "MCF-7")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    audit = ROOT / "results/enpda_graph_overlap"
    pairs = [json.loads(l) for l in (audit / "pairs.jsonl").read_text().splitlines()]
    occurrences = [json.loads(l) for l in (audit / "graph_occurrences.jsonl").read_text().splitlines()]
    identities = {o["identity"]: o["ordered_tensor_sha256"] for o in occurrences}
    test = [p for p in pairs if p["split"] == "test"]
    test_ids = {i for p in test for i in p["identities"]}
    eligible = [p for p in pairs if p["split"] != "test" and not set(p["identities"]) & test_ids]

    def pair_key(p):
        return tuple(sorted(identities[i] for i in p["identities"]))

    def rank(p):
        return hashlib.sha256((SEED + ":" + ":".join(pair_key(p))).encode()).hexdigest()

    unique = {}
    for p in sorted(eligible, key=lambda p: (DATASETS.index(p["dataset"]), p["path"], p["index"])):
        unique.setdefault(pair_key(p), p)
    eligible = sorted(unique.values(), key=rank)
    # Reserve validation pairs using identities alone. Greedy disjoint anchors
    # guarantee a nonempty validation slice even in the small remaining AIDS pool.
    val_ids = set()
    anchors = []
    quotas = {}
    for dataset in DATASETS:
        candidates = [p for p in eligible if p["dataset"] == dataset]
        quota = max(5, math.ceil(0.1 * len(candidates)))
        chosen = []
        for p in candidates:
            if not set(p["identities"]) & val_ids:
                chosen.append(p)
                val_ids.update(p["identities"])
                if len(chosen) == quota:
                    break
        assert len(chosen) == quota, (dataset, quota, len(chosen))
        quotas[dataset] = quota
        anchors.extend(chosen)
    validation = [p for p in eligible if set(p["identities"]) <= val_ids]
    train = [p for p in eligible if not set(p["identities"]) & val_ids]
    train_ids = {i for p in train for i in p["identities"]}
    assert not train_ids & val_ids and not train_ids & test_ids and not val_ids & test_ids
    selected = {"train": train, "validation": validation, "test": test}
    counts = {s: dict(Counter(p["dataset"] for p in rows)) for s, rows in selected.items()}
    assert counts["test"] == {"AIDS": 100, "MOLHIV": 91, "MCF-7": 100}
    assert all(counts[s].get(d, 0) > 0 for s in ("train", "validation") for d in DATASETS)
    manifest = {
        "protocol": SEED,
        "source_audit_sha256": sha(audit / "summary.json"),
        "source_pair_records_sha256": sha(audit / "pairs.jsonl"),
        "rule": "Preserve all sanitized native tests; exclude every test graph globally; deduplicate remaining unordered graph pairs; per dataset reserve hash-ordered mutually graph-disjoint validation anchors equal to max(5,ceil(10% eligible unique pairs)); assign pairs wholly in that graph pool to validation, wholly outside to training, and discard cross-boundary pairs.",
        "selection_uses_model_outcomes": False,
        "selection_uses_optimum_labels": False,
        "training_and_validation_optimum_labels_removed": True,
        "validation_anchor_quotas": quotas,
        "counts": counts,
        "unique_graphs": {"train": len(train_ids), "validation": len(val_ids), "test": len(test_ids)},
        "graph_intersections": {"train_validation": 0, "train_test": 0, "validation_test": 0},
        "splits": selected,
        "materialized_files": {},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    cache = {}
    for split, records in selected.items():
        materialized = []
        for r in records:
            if r["path"] not in cache:
                cache[r["path"]] = load_pairs(ROOT / r["path"])
            pair = cache[r["path"]][r["index"]]
            if split != "test":
                pair = replace(pair, true_edges=None, true_nodes=None, true_similarity=None)
            materialized.append({**r, "original_split": r["split"], "split": split,
                                 "source_key": f"{r['path']}#{r['index']}", "pair": pair})
        path = OUT / f"{split}.pt"
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite a frozen split: {path}")
        torch.save(materialized, path)
        manifest["materialized_files"][split] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("counts", "unique_graphs", "graph_intersections", "validation_anchor_quotas")}, indent=2))


if __name__ == "__main__":
    main()
