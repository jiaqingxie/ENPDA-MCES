"""Strict graph-disjoint leave-one-dataset-out data preparation."""

from __future__ import annotations

import hashlib
from pathlib import Path

from nema.data import load_pairs, pair_paths
from nema.graph import GraphPair, LabeledGraph


DATASETS = ("AIDS", "MOLHIV", "MCF-7")


def graph_tensor_fingerprint(graph: LabeledGraph) -> str:
    """Hash the normalized labeled tensor representation including shapes."""

    digest = hashlib.sha256()
    for tensor in (graph.node_labels, graph.edge_index, graph.edge_labels):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(contiguous.numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _sourceless_pair(pair: GraphPair) -> GraphPair:
    """Remove every released label before a pair enters unsupervised training."""

    return GraphPair(
        left=pair.left,
        right=pair.right,
        key=pair.key,
        metadata=pair.metadata,
    )


def prepare_strict_ood_pairs(
    data_root: str | Path,
    heldout: str,
) -> tuple[list[GraphPair], dict]:
    """Filter cross-dataset exact graph leakage from an OOD training fold."""

    if heldout not in DATASETS:
        raise ValueError(f"unsupported held-out dataset {heldout!r}")
    data_root = Path(data_root)
    train_datasets = [dataset for dataset in DATASETS if dataset != heldout]

    banned: set[str] = set()
    heldout_pairs = 0
    heldout_native_pairs = 0
    heldout_recovered_pairs = 0
    for path in pair_paths(data_root, heldout, split="test"):
        for pair in load_pairs(path):
            heldout_pairs += 1
            if pair.metadata is None:
                heldout_native_pairs += 1
            else:
                heldout_recovered_pairs += 1
            banned.add(graph_tensor_fingerprint(pair.left))
            banned.add(graph_tensor_fingerprint(pair.right))

    kept: list[GraphPair] = []
    removed: list[dict] = []
    per_dataset = {}
    kept_fingerprints: set[str] = set()
    for dataset in train_datasets:
        released_count = 0
        kept_count = 0
        removed_count = 0
        for path in pair_paths(data_root, dataset, split="train"):
            for pair in load_pairs(path):
                released_count += 1
                left = graph_tensor_fingerprint(pair.left)
                right = graph_tensor_fingerprint(pair.right)
                left_banned, right_banned = left in banned, right in banned
                if left_banned or right_banned:
                    removed_count += 1
                    removed.append(
                        {
                            "dataset": dataset,
                            "source_path": str(path),
                            "pair_key": pair.key,
                            "left_banned": left_banned,
                            "right_banned": right_banned,
                            "left_fingerprint": left,
                            "right_fingerprint": right,
                        }
                    )
                    continue
                kept.append(_sourceless_pair(pair))
                kept_count += 1
                kept_fingerprints.update((left, right))
        per_dataset[dataset] = {
            "released_training_pairs": released_count,
            "removed_pairs": removed_count,
            "remaining_pairs": kept_count,
        }

    residual = sorted(banned & kept_fingerprints)
    if residual:
        raise AssertionError(f"strict OOD filtering left {len(residual)} banned fingerprints")
    manifest = {
        "protocol": "strict leave-one-dataset-out with exact normalized-tensor graph disjointness",
        "heldout_dataset": heldout,
        "training_datasets": train_datasets,
        "heldout_split": "test",
        "heldout_labels_used": False,
        "training_labels_used": False,
        "heldout_test_pairs": heldout_pairs,
        "heldout_native_pairs": heldout_native_pairs,
        "heldout_recovered_pairs": heldout_recovered_pairs,
        "banned_fingerprint_count": len(banned),
        "banned_fingerprints": sorted(banned),
        "released_training_pairs": sum(
            item["released_training_pairs"] for item in per_dataset.values()
        ),
        "removed_training_pairs": len(removed),
        "remaining_training_pairs": len(kept),
        "heldout_test_filtered_train_overlap": len(residual),
        "per_dataset": per_dataset,
        "removed": removed,
    }
    return kept, manifest
