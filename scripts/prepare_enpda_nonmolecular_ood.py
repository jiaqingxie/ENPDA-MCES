#!/usr/bin/env python3
"""Freeze exact planted IMDB-BINARY pairs before ENPDA evaluation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from torch_geometric.datasets import TUDataset


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/enpda_nonmolecular_ood.json"
OUTPUT = ROOT / "results/enpda_nonmolecular_ood/manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_undirected_edges(data: object) -> list[tuple[int, int]]:
    edge_index = data.edge_index.detach().cpu()
    edges: set[tuple[int, int]] = set()
    for position in range(edge_index.shape[1]):
        u = int(edge_index[0, position])
        v = int(edge_index[1, position])
        if u == v:
            continue
        edges.add((u, v) if u < v else (v, u))
    return sorted(edges)


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    specification = config["dataset"]
    dataset = TUDataset(root=str(ROOT / "data/nonmolecular"), name=specification["name"])
    candidates: list[int] = []
    edge_cache: dict[int, list[tuple[int, int]]] = {}
    for index, data in enumerate(dataset):
        edges = unique_undirected_edges(data)
        nodes = int(data.num_nodes)
        if (
            int(specification["minimum_nodes"]) <= nodes <= int(specification["maximum_nodes"])
            and len(edges) >= int(specification["minimum_edges"])
        ):
            candidates.append(index)
            edge_cache[index] = edges
    if len(candidates) < int(specification["num_pairs"]):
        raise RuntimeError(f"only {len(candidates)} eligible source graphs")

    selection_rng = np.random.default_rng(int(specification["selection_seed"]))
    selected = np.asarray(candidates, dtype=np.int64)
    selection_rng.shuffle(selected)
    selected = selected[: int(specification["num_pairs"])]
    deletion = float(config["pair_construction"]["edge_deletion_fraction"])
    pairs: list[dict] = []
    for rank, dataset_index in enumerate(selected.tolist()):
        data = dataset[dataset_index]
        nodes = int(data.num_nodes)
        source_edges = edge_cache[dataset_index]
        rng = np.random.default_rng(int(specification["selection_seed"]) + 104729 * (rank + 1))
        keep_count = max(1, int(round((1.0 - deletion) * len(source_edges))))
        keep_indices = sorted(
            int(value) for value in rng.choice(len(source_edges), size=keep_count, replace=False)
        )
        retained = [source_edges[position] for position in keep_indices]
        permutation = rng.permutation(nodes).astype(np.int64)
        target_edges = sorted(
            (
                min(int(permutation[u]), int(permutation[v])),
                max(int(permutation[u]), int(permutation[v])),
            )
            for u, v in retained
        )
        if len(set(target_edges)) != len(target_edges):
            raise RuntimeError("node permutation introduced duplicate edges")
        pairs.append(
            {
                "pair_id": f"IMDB-BINARY-{dataset_index:04d}",
                "dataset_index": dataset_index,
                "class_label": int(data.y.reshape(-1)[0]),
                "num_nodes": nodes,
                "source_edges": [list(edge) for edge in source_edges],
                "retained_source_edge_indices": keep_indices,
                "target_edges": [list(edge) for edge in target_edges],
                "target_permutation_old_to_new": permutation.tolist(),
                "exact_optimum_edges": keep_count,
            }
        )

    raw_root = ROOT / specification["raw_root"]
    raw_hashes = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in sorted(raw_root.glob("IMDB-BINARY_*.txt"))
    }
    manifest = {
        "protocol_version": config["protocol_version"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256(CONFIG),
        "raw_dataset_sha256": raw_hashes,
        "eligible_graphs": len(candidates),
        "selected_graphs": len(pairs),
        "selection_depends_only_on_graph_size_and_frozen_seed": True,
        "pairs": pairs,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    node_counts = np.asarray([pair["num_nodes"] for pair in pairs])
    edge_counts = np.asarray([pair["exact_optimum_edges"] for pair in pairs])
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "sha256": sha256(OUTPUT),
                "pairs": len(pairs),
                "nodes": {
                    "min": int(node_counts.min()),
                    "median": float(np.median(node_counts)),
                    "max": int(node_counts.max()),
                },
                "optimum_edges": {
                    "min": int(edge_counts.min()),
                    "median": float(np.median(edge_counts)),
                    "max": int(edge_counts.max()),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
