"""Construction helpers for a method-independent RASCAL retrieval benchmark."""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path

import torch

from nema.data import recover_empty_molecular_data
from nema.graph import LabeledGraph
from nema.oracle import rascal_mces_truth


@dataclass
class BankGraph:
    fingerprint: str
    data: object
    source_path: str
    source_side: str
    recovered_from_smiles: bool

    def manifest_record(self) -> dict[str, object]:
        graph = LabeledGraph.from_pyg(self.data)
        return {
            "fingerprint": self.fingerprint,
            "source_path": self.source_path,
            "source_side": self.source_side,
            "recovered_from_smiles": self.recovered_from_smiles,
            "num_nodes": graph.num_nodes,
            "num_edges": graph.num_edges,
            "smiles": getattr(self.data, "smiles", None),
        }


def graph_fingerprint(data: object) -> str:
    """Hash a graph after normalizing duplicate directions and edge order."""

    graph = LabeledGraph.from_pyg(data)
    digest = hashlib.sha256()
    digest.update(graph.node_labels.contiguous().numpy().tobytes())
    digest.update(graph.edge_index.contiguous().numpy().tobytes())
    digest.update(graph.edge_labels.contiguous().numpy().tobytes())
    return digest.hexdigest()


def iter_pickle_graphs(path: Path):
    with path.open("rb") as stream:
        left, right = pickle.load(stream)
    for side, graphs in (("left", left), ("right", right)):
        for graph in graphs:
            recovered, changed = recover_empty_molecular_data(graph)
            if int(recovered.num_nodes) > 0:
                yield side, recovered, changed


def collect_graph_bank(data_root: Path, dataset: str) -> tuple[list[BankGraph], set[str]]:
    """Collect unique graphs not used by the NEMA MCES training split."""

    training_fingerprints: set[str] = set()
    train_root = data_root / "MCES" / f"{dataset}-train" / "raw"
    for path in sorted(train_root.glob("graphs_*.pkl")):
        for _, graph, _ in iter_pickle_graphs(path):
            training_fingerprints.add(graph_fingerprint(graph))

    roots = [
        data_root / "MCES" / f"{dataset}-val" / "raw",
        data_root / "MCES" / f"{dataset}-test" / "raw",
        data_root / "retrieval" / dataset / "raw" / "train",
        data_root / "retrieval" / dataset / "raw" / "val",
        data_root / "retrieval" / dataset / "raw" / "test",
    ]
    unique: dict[str, BankGraph] = {}
    for root in roots:
        for path in sorted(root.glob("graphs_*.pkl")):
            for side, graph, recovered in iter_pickle_graphs(path):
                fingerprint = graph_fingerprint(graph)
                if fingerprint in training_fingerprints or fingerprint in unique:
                    continue
                clean = graph.clone()
                if hasattr(clean, "y"):
                    del clean.y
                unique[fingerprint] = BankGraph(
                    fingerprint=fingerprint,
                    data=clean,
                    source_path=str(path),
                    source_side=side,
                    recovered_from_smiles=recovered,
                )
    return [unique[key] for key in sorted(unique)], training_fingerprints


def select_disjoint_graphs(
    bank: list[BankGraph],
    queries: int,
    candidates: int,
    seed: int,
) -> tuple[list[BankGraph], list[BankGraph]]:
    if len(bank) < queries + candidates:
        raise ValueError(
            f"need {queries + candidates} graph-disjoint entries but only {len(bank)} are "
            "outside the NEMA MCES training set"
        )
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(bank), generator=generator).tolist()
    selected_queries = [bank[index] for index in order[:queries]]
    selected_candidates = [bank[index] for index in order[queries : queries + candidates]]
    return selected_queries, selected_candidates


def select_controlled_graphs(
    bank: list[BankGraph],
    queries: int,
    negative_pool: int,
    seed: int,
    small_graph_pool: int = 300,
) -> tuple[list[BankGraph], list[BankGraph]]:
    """Select train-disjoint parents while bounding exact-oracle complexity.

    MCES has an exponential long tail.  The reconstruction therefore declares
    a size-controlled regime up front and samples only inside the smallest
    ``small_graph_pool`` held-out graphs.  The query and negative parent sets
    remain fingerprint-disjoint.
    """

    required = queries + negative_pool
    if len(bank) < required:
        raise ValueError(
            f"need {required} train-disjoint graph parents but only {len(bank)} are available"
        )
    pool_size = min(len(bank), max(required, small_graph_pool))
    ordered = sorted(
        bank,
        key=lambda entry: (
            LabeledGraph.from_pyg(entry.data).num_edges,
            LabeledGraph.from_pyg(entry.data).num_nodes,
            entry.fingerprint,
        ),
    )[:pool_size]
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(ordered), generator=generator).tolist()
    selected = [ordered[index] for index in order[:required]]
    return selected[:queries], selected[queries:]


def delete_undirected_edges(data: object, count: int, seed: int) -> object:
    """Return a deterministic structural corruption of a PyG graph."""

    graph = LabeledGraph.from_pyg(data)
    if count <= 0 or count >= graph.num_edges:
        raise ValueError(f"edge deletion count must be in [1, {graph.num_edges - 1}]")
    generator = torch.Generator().manual_seed(seed)
    selected = torch.randperm(graph.num_edges, generator=generator)[:count].tolist()
    deleted = {
        tuple(int(value) for value in graph.edge_index[:, index].tolist())
        for index in selected
    }

    candidate = data.clone()
    raw_index = torch.as_tensor(candidate.edge_index).detach().cpu().long()
    keep = []
    for index in range(raw_index.shape[1]):
        source, target = int(raw_index[0, index]), int(raw_index[1, index])
        edge = (source, target) if source < target else (target, source)
        keep.append(edge not in deleted)
    mask = torch.tensor(keep, dtype=torch.bool, device=candidate.edge_index.device)
    candidate.edge_index = candidate.edge_index[:, mask]
    candidate.edge_attr = candidate.edge_attr[mask]
    if hasattr(candidate, "y"):
        del candidate.y
    candidate.controlled_deleted_edges = count
    candidate.controlled_corruption_seed = seed
    return candidate


def controlled_candidate_specs(
    query_edge_counts: list[int],
    negative_pool: int,
    candidates: int,
    positives: int,
    seed: int,
) -> list[list[dict[str, int | str]]]:
    """Build shuffled per-query pools with known structural positives."""

    negative_count = candidates - positives
    if positives <= 0 or negative_count <= 0:
        raise ValueError("controlled retrieval needs both positives and negatives")
    if negative_pool < negative_count:
        raise ValueError("negative parent pool is too small")
    all_specs: list[list[dict[str, int | str]]] = []
    for query_index, edge_count in enumerate(query_edge_counts):
        generator = torch.Generator().manual_seed(seed + (query_index + 1) * 10007)
        deletion_counts = [
            max(1, edge_count * 2 * rank // (positives * 5))
            for rank in range(1, positives + 1)
        ]
        if len(set(deletion_counts)) != positives or deletion_counts[-1] >= edge_count:
            raise ValueError(
                f"cannot build {positives} distinct corruption levels for {edge_count} edges"
            )
        specs: list[dict[str, int | str]] = [
            {
                "kind": "controlled_edge_deletion",
                "deletion_count": deletion_count,
                "seed": seed + (query_index + 1) * 1000003 + deletion_count,
            }
            for deletion_count in deletion_counts
        ]
        negative_indices = torch.randperm(negative_pool, generator=generator)[
            :negative_count
        ].tolist()
        specs.extend(
            {"kind": "held_out_negative", "negative_index": index}
            for index in negative_indices
        )
        shuffle = torch.randperm(candidates, generator=generator).tolist()
        all_specs.append([specs[index] for index in shuffle])
    return all_specs


def build_rascal_pair(
    query: BankGraph,
    candidate: BankGraph,
    query_index: int,
    candidate_index: int,
    similarity_threshold: float = 0.0,
    timeout_seconds: int = 20000,
) -> tuple[list[list[object]], dict[str, object]]:
    left = query.data.clone()
    right = candidate.data.clone()
    oracle = rascal_mces_truth(
        LabeledGraph.from_pyg(left),
        LabeledGraph.from_pyg(right),
        similarity_threshold=similarity_threshold,
        timeout_seconds=timeout_seconds,
    )
    truth = torch.tensor(
        [[oracle["common_edges"], oracle["common_nodes"], oracle["similarity"]]],
        dtype=torch.float32,
    )
    left.y = truth.clone()
    right.y = truth.clone()
    left.reconstructed_query_index = query_index
    right.reconstructed_candidate_index = candidate_index
    payload = [[left], [right]]
    record = {
        "query_index": query_index,
        "candidate_index": candidate_index,
        "key": query_index * 100 + candidate_index + 1,
        "query_fingerprint": query.fingerprint,
        "candidate_fingerprint": candidate.fingerprint,
        "true_edges": oracle["common_edges"],
        "true_nodes": oracle["common_nodes"],
        "true_similarity": oracle["similarity"],
        "rascal_similarity_threshold": similarity_threshold,
        "rascal_timeout_seconds": timeout_seconds,
        "rascal_timed_out": oracle["timed_out"],
    }
    return payload, record
