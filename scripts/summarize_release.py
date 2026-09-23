"""Summarize new native runs; never substitute archived or embedded scores."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def connected_pair_blocks(pairs):
    """Group pairs that share any exact labeled-graph identity, transitively."""
    parent = list(range(len(pairs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owner = {}
    for i, pair in enumerate(pairs):
        for identity in pair["identities"]:
            if identity in owner:
                parent[find(i)] = find(owner[identity])
            else:
                owner[identity] = i
    groups = {}
    for i in range(len(pairs)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def paired_interval(delta, blocks, draws=20000, seed=20260917):
    """Paired training-seed and connected-component bootstrap of a pair mean."""
    rng = np.random.default_rng(seed)
    values = np.empty(draws)
    for b in range(draws):
        seeds = rng.integers(0, len(delta), len(delta))
        sampled_blocks = rng.integers(0, len(blocks), len(blocks))
        indices = [i for g in sampled_blocks for i in blocks[g]]
        values[b] = delta[seeds][:, indices].mean()
    return np.quantile(values, (0.025, 0.975)).tolist()


def main():
    manifest_path = ROOT / "data/enpda_graph_disjoint_v1/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    outputs = {}
    hashes = {}
    for dataset in ("AIDS", "MOLHIV", "MCF-7"):
        pairs = [r for r in manifest["splits"]["test"] if r["dataset"] == dataset]
        paths = [r["path"] for r in pairs]
        blocks = connected_pair_blocks(pairs)
        accuracies = {}
        summary = {"pairs": len(paths), "connected_graph_components": len(blocks), "arms": {}}
        for arm in ("core", "analytic_core", "solver"):
            seeds = []
            times = []
            for seed in range(3):
                path = ROOT / f"results/enpda_formal/native/seed{seed}/enpda_{arm}_{dataset}.jsonl"
                records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                indexed = {str(Path(r["source_path"]).relative_to(ROOT)) if Path(r["source_path"]).is_absolute()
                           else r["source_path"]: r for r in records}
                assert len(records) == len(indexed) == len(paths) and set(indexed) == set(paths)
                seeds.append([100 * float(indexed[p]["accuracy"]) for p in paths])
                times.extend(float(indexed[p]["runtime_seconds"]) for p in paths)
                hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            accuracy = np.asarray(seeds)
            accuracies[arm] = accuracy
            summary["arms"][arm] = {"accuracy_percent": float(accuracy.mean()),
                                     "seed_means_percent": accuracy.mean(axis=1).tolist(),
                                     "mean_seconds_per_pair": float(np.mean(times))}
        delta = accuracies["core"] - accuracies["analytic_core"]
        summary["core_minus_analytic"] = {"gain_percentage_points": float(delta.mean()),
                                           "paired_component_seed_bootstrap_95ci": paired_interval(delta, blocks)}
        outputs[dataset] = summary
    dest = ROOT / "results/reproduction_summary.json"
    dest.write_text(json.dumps({"datasets": outputs, "input_sha256": hashes,
                               "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                               "reference": "Released reference labels; accuracy is not clipped to 100 percent."}, indent=2) + "\n")
    print(dest)


if __name__ == "__main__":
    main()
