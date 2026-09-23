"""Auditable adapter for the public MCSNet corpus graphs.

The MCSNet code release contains 800 unlabeled MIVIA-format corpus graphs for
each dataset even when its separate, shortened full-dataset download is not
available.  This module reads those original graph files and constructs a
strictly graph-disjoint zero-shot MCES transfer benchmark.  It does not import
or execute the authors' Python implementation.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
import struct
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import torch
from torch_geometric.data import Data

from nema.graph import LabeledGraph
from nema.metrics import johnson_similarity
from nema.oracle import rascal_mces_truth


MCSNET_REPOSITORY = "https://github.com/Indradyumna/MCSNET"
MCSNET_PAPER = "https://proceedings.neurips.cc/paper_files/paper/2022/hash/cf7a83a5342befd11d3d65beba1be5b0-Abstract-Conference.html"
MCSNET_CODE_ARCHIVE_SHA256 = (
    "211a8901a608551530ee61f77f90f5e9ae76ac30f386f2755124b4d9d90e2843"
)

DATASETS = {
    "MSRC-21": "msrc_21",
    "PTC-MM": "ptc_mm",
    "COX2": "cox2",
}
_DATASET_OFFSETS = {"MSRC-21": 1103, "PTC-MM": 2207, "COX2": 3301}


@dataclass(frozen=True)
class ExternalPairSpec:
    key: int
    left_index: int
    right_index: int
    left_path: str
    right_path: str
    left_sha256: str
    right_sha256: str

    def to_dict(self) -> dict[str, int | str]:
        return {
            "key": self.key,
            "left_index": self.left_index,
            "right_index": self.right_index,
            "left_path": self.left_path,
            "right_path": self.right_path,
            "left_sha256": self.left_sha256,
            "right_sha256": self.right_sha256,
        }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_unlabeled_mivia(path: str | Path) -> LabeledGraph:
    """Read one symmetric MIVIA graph and canonicalize its universal labels.

    MCSNet writes these files with every vertex and edge label equal to zero.
    Internally we use atomic number 6 and single-bond label 1 so that the same
    topology can also be passed to the independent RDKit RASCAL MCES oracle.
    Since both are universal constants, this does not change compatibility.
    """

    path = Path(path)
    raw = path.read_bytes()
    if len(raw) % 2:
        raise ValueError(f"odd-sized MIVIA file: {path}")
    values = iter(struct.unpack(f"<{len(raw) // 2}H", raw))
    try:
        num_nodes = next(values)
        source_node_labels = [next(values) for _ in range(num_nodes)]
        directed: dict[tuple[int, int], int] = {}
        for source in range(num_nodes):
            degree = next(values)
            for _ in range(degree):
                target, label = next(values), next(values)
                if target >= num_nodes:
                    raise ValueError(f"MIVIA target {target} outside graph in {path}")
                if source == target:
                    raise ValueError(f"self-loop in simple MIVIA graph {path}")
                directed[(source, target)] = label
    except StopIteration as error:
        raise ValueError(f"truncated MIVIA file: {path}") from error
    try:
        next(values)
    except StopIteration:
        pass
    else:
        raise ValueError(f"trailing MIVIA values in {path}")
    if any(source_node_labels) or any(directed.values()):
        raise ValueError(f"expected MCSNet's unlabeled MIVIA encoding in {path}")
    for source, target in directed:
        if (target, source) not in directed:
            raise ValueError(f"asymmetric adjacency {source}-{target} in {path}")

    edges = sorted((source, target) for source, target in directed if source < target)
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )
    return LabeledGraph(
        node_labels=torch.full((num_nodes,), 6, dtype=torch.long),
        edge_index=edge_index,
        edge_labels=torch.ones(len(edges), dtype=torch.long),
    )


def graph_to_pyg(graph: LabeledGraph, truth: tuple[int, int, float]) -> Data:
    if graph.num_edges:
        edge_index = torch.cat((graph.edge_index, graph.edge_index.flip(0)), dim=1)
        edge_attr = torch.cat((graph.edge_labels, graph.edge_labels), dim=0)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty(0, dtype=torch.long)
    return Data(
        x=graph.node_labels.clone(),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor(truth, dtype=torch.float64),
    )


def select_pair_specs(
    corpus_root: str | Path,
    dataset: str,
    pair_count: int = 100,
    seed: int = 20260823,
) -> list[ExternalPairSpec]:
    """Select graph-disjoint pairs from the final 200 released corpus graphs."""

    if dataset not in DATASETS:
        raise ValueError(f"unsupported MCSNet dataset {dataset!r}")
    if not 1 <= pair_count <= 100:
        raise ValueError("pair_count must be between 1 and 100 for graph-disjoint selection")
    corpus_root = Path(corpus_root)
    prefix = DATASETS[dataset]
    paths = [corpus_root / f"{prefix}_corpus_graphs_{index}.mivia" for index in range(800)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"MCSNet corpus is incomplete ({len(missing)} missing); first: {missing[0]}"
        )
    indices = list(range(600, 800))
    random.Random(seed + _DATASET_OFFSETS[dataset]).shuffle(indices)
    selected = indices[: 2 * pair_count]
    specs = []
    for key in range(1, pair_count + 1):
        left_index, right_index = selected[2 * (key - 1) : 2 * key]
        left_path, right_path = paths[left_index], paths[right_index]
        specs.append(
            ExternalPairSpec(
                key=key,
                left_index=left_index,
                right_index=right_index,
                left_path=str(left_path),
                right_path=str(right_path),
                left_sha256=sha256_file(left_path),
                right_sha256=sha256_file(right_path),
            )
        )
    return specs


def _oracle_one(payload: tuple[ExternalPairSpec, int]) -> dict[str, object]:
    spec, timeout_seconds = payload
    left = read_unlabeled_mivia(spec.left_path)
    right = read_unlabeled_mivia(spec.right_path)
    started = time.perf_counter()
    truth = rascal_mces_truth(left, right, timeout_seconds=timeout_seconds)
    runtime = time.perf_counter() - started
    if bool(truth["timed_out"]):
        raise TimeoutError(
            f"RASCAL timed out for {spec.left_path} versus {spec.right_path}"
        )
    true_edges = int(truth["common_edges"])
    true_nodes = int(truth["common_nodes"])
    # Recompute from integer counts to use the same exact similarity definition
    # as NGA and NEMA, while retaining RASCAL's value as an audit field.
    true_similarity = johnson_similarity(
        true_nodes,
        true_edges,
        left.num_nodes + left.num_edges,
        right.num_nodes + right.num_edges,
    )
    return {
        **spec.to_dict(),
        "true_edges": true_edges,
        "true_nodes": true_nodes,
        "true_similarity": true_similarity,
        "rascal_similarity": float(truth["similarity"]),
        "rascal_timed_out": False,
        "oracle_runtime_seconds": runtime,
        "left_nodes": left.num_nodes,
        "left_edges": left.num_edges,
        "right_nodes": right.num_nodes,
        "right_edges": right.num_edges,
    }


def _write_pair(path: Path, record: dict[str, object]) -> None:
    left = read_unlabeled_mivia(str(record["left_path"]))
    right = read_unlabeled_mivia(str(record["right_path"]))
    truth = (
        int(record["true_edges"]),
        int(record["true_nodes"]),
        float(record["true_similarity"]),
    )
    left_data, right_data = graph_to_pyg(left, truth), graph_to_pyg(right, truth)
    metadata = {
        "external_dataset": record["dataset"],
        "source_repository": MCSNET_REPOSITORY,
        "left_source_index": record["left_index"],
        "right_source_index": record["right_index"],
        "unlabeled_canonicalization": "all vertices=C(6), all edges=single(1)",
    }
    left_data.nema_metadata = metadata
    right_data.nema_metadata = metadata
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(([left_data], [right_data]), stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def build_external_dataset(
    corpus_root: str | Path,
    output_root: str | Path,
    dataset: str,
    pair_count: int = 100,
    seed: int = 20260823,
    workers: int = 8,
    timeout_seconds: int = 300,
) -> dict[str, object]:
    """Build or resume one graph-disjoint MCSNet transfer dataset."""

    specs = select_pair_specs(corpus_root, dataset, pair_count, seed)
    output_root = Path(output_root)
    dataset_root = output_root / "MCES" / f"{dataset}-test"
    raw_root = dataset_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    manifest_path = dataset_root / "manifest.json"
    manifest = {
        "protocol": "MCSNet corpus graph-disjoint zero-shot MCES transfer",
        "dataset": dataset,
        "pair_count": pair_count,
        "seed": seed,
        "source_repository": MCSNET_REPOSITORY,
        "source_paper": MCSNET_PAPER,
        "source_code_archive_sha256": MCSNET_CODE_ARCHIVE_SHA256,
        "source_graph_range": [600, 799],
        "selection": "seeded shuffle; each source graph appears in at most one pair",
        "label_policy": "released universal labels canonicalized to C/single bond",
        "oracle": "RDKit RASCAL MCES, completeAromaticRings=False",
        "oracle_timeout_seconds": timeout_seconds,
        "pairs": [spec.to_dict() for spec in specs],
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        stable_keys = (
            "protocol",
            "dataset",
            "pair_count",
            "seed",
            "source_code_archive_sha256",
            "source_graph_range",
            "selection",
            "label_policy",
            "pairs",
        )
        if any(previous.get(key) != manifest.get(key) for key in stable_keys):
            raise ValueError(f"refusing to change existing selection in {manifest_path}")
    else:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    oracle_path = dataset_root / "oracle.jsonl"
    completed: dict[int, dict[str, object]] = {}
    if oracle_path.exists():
        for line in oracle_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                key = int(record["key"])
                if key in completed:
                    raise ValueError(f"duplicate oracle key {key} in {oracle_path}")
                completed[key] = record
    spec_by_key = {spec.key: spec for spec in specs}
    unexpected = set(completed) - set(spec_by_key)
    if unexpected:
        raise ValueError(f"oracle contains keys outside manifest: {sorted(unexpected)[:3]}")

    pending = [spec for spec in specs if spec.key not in completed]
    with oracle_path.open("a", encoding="utf-8") as stream:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_oracle_one, (spec, timeout_seconds)): spec.key for spec in pending
            }
            for progress, future in enumerate(as_completed(futures), 1):
                record = future.result()
                record["dataset"] = dataset
                completed[int(record["key"])] = record
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                if progress == 1 or progress % 10 == 0 or progress == len(pending):
                    print(
                        f"[{progress}/{len(pending)}] {dataset} RASCAL "
                        f"edges={record['true_edges']} time={record['oracle_runtime_seconds']:.3f}s",
                        flush=True,
                    )

    for key, record in completed.items():
        destination = raw_root / f"graphs_{key}.pkl"
        if not destination.exists():
            _write_pair(destination, record)
    ordered = [completed[key] for key in sorted(completed)]
    if len(ordered) != pair_count:
        raise RuntimeError(f"incomplete oracle: {len(ordered)}/{pair_count}")
    source_indices = [
        int(record[field])
        for record in ordered
        for field in ("left_index", "right_index")
    ]
    if len(source_indices) != len(set(source_indices)):
        raise RuntimeError("source-graph disjointness audit failed")
    summary = {
        "dataset": dataset,
        "pairs": len(ordered),
        "unique_source_graphs": len(set(source_indices)),
        "source_graph_overlap": len(source_indices) - len(set(source_indices)),
        "rascal_timed_out_pairs": sum(bool(record["rascal_timed_out"]) for record in ordered),
        "mean_true_edges": sum(int(record["true_edges"]) for record in ordered) / len(ordered),
        "mean_true_similarity": sum(float(record["true_similarity"]) for record in ordered)
        / len(ordered),
        "mean_oracle_runtime_seconds": sum(
            float(record["oracle_runtime_seconds"]) for record in ordered
        )
        / len(ordered),
        "max_oracle_runtime_seconds": max(
            float(record["oracle_runtime_seconds"]) for record in ordered
        ),
    }
    (dataset_root / "oracle_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary
