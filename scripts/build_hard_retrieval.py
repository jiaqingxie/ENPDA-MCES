#!/usr/bin/env python3
"""Build a train-disjoint, method-independent hard RASCAL retrieval benchmark."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from nema.data import OFFICIAL_DATA_SHA256
from nema.graph import LabeledGraph
from nema.hard_retrieval import FORMAL_SEEDS, HardPoolEntry, build_hard_protocol
from nema.oracle import rascal_mces_truth
from nema.reconstructed import BankGraph, collect_graph_bank


_QUERIES: list[BankGraph] = []
_POOLS: list[list[HardPoolEntry]] = []
_RAW: Path | None = None
_CANDIDATE_COUNT = 0
_RELEVANCE_THRESHOLD = 0.5
_RASCAL_TIMEOUT_SECONDS = 60
_PROTOCOL_SEED = 0


def _build_index(index: int) -> dict[str, object]:
    if _RAW is None:
        raise RuntimeError("worker benchmark is not initialized")
    query_index, candidate_index = divmod(index, _CANDIDATE_COUNT)
    query = _QUERIES[query_index]
    entry = _POOLS[query_index][candidate_index]
    left = query.data.clone()
    right = entry.candidate.graph.data.clone()
    started = time.perf_counter()
    truth = rascal_mces_truth(
        LabeledGraph.from_pyg(left),
        LabeledGraph.from_pyg(right),
        similarity_threshold=_RELEVANCE_THRESHOLD,
        timeout_seconds=_RASCAL_TIMEOUT_SECONDS,
    )
    label = torch.tensor(
        [[truth["common_edges"], truth["common_nodes"], truth["similarity"]]],
        dtype=torch.float32,
    )
    left.y = label.clone()
    right.y = label.clone()
    left.hard_query_index = query_index
    right.hard_candidate_index = candidate_index
    destination = _RAW / f"graphs_{index + 1}.pkl"
    temporary = destination.with_suffix(".pkl.tmp")
    with temporary.open("wb") as stream:
        pickle.dump([[left], [right]], stream)
    os.replace(temporary, destination)
    return {
        "key": index + 1,
        "query_index": query_index,
        "candidate_index": candidate_index,
        "protocol_seed": _PROTOCOL_SEED,
        "query_fingerprint": query.fingerprint,
        "candidate_fingerprint": entry.candidate.graph.fingerprint,
        "candidate_id": entry.candidate.candidate_id,
        "candidate_kind": entry.candidate.candidate_kind,
        "candidate_parent_fingerprint": entry.candidate.parent_fingerprint,
        "size_similarity": entry.size_similarity,
        "wl_similarity": entry.wl_similarity,
        "hard_score": entry.hard_score,
        "true_edges": truth["common_edges"],
        "true_nodes": truth["common_nodes"],
        "true_similarity": truth["similarity"],
        "rascal_similarity_threshold": _RELEVANCE_THRESHOLD,
        "rascal_timeout_seconds": _RASCAL_TIMEOUT_SECONDS,
        "rascal_timed_out": truth["timed_out"],
        "source_path": str(destination),
        "oracle_runtime_seconds": time.perf_counter() - started,
    }


def _summarize(records: list[dict], candidate_count: int) -> dict[str, object]:
    records.sort(key=lambda record: int(record["key"]))
    relevant_counts: list[int] = []
    controlled_relevant_counts: list[int] = []
    hard_relevant_counts: list[int] = []
    for start in range(0, len(records), candidate_count):
        group = records[start : start + candidate_count]
        relevant = [record for record in group if float(record["true_similarity"]) > 0.5]
        relevant_counts.append(len(relevant))
        controlled_relevant_counts.append(
            sum(str(record["candidate_kind"]).startswith("controlled_") for record in relevant)
        )
        hard_relevant_counts.append(
            sum(not str(record["candidate_kind"]).startswith("controlled_") for record in relevant)
        )
    return {
        "pair_count": len(records),
        "query_count": len(relevant_counts),
        "candidate_count": candidate_count,
        "relevant_counts_gt_0.5": relevant_counts,
        "controlled_relevant_counts_gt_0.5": controlled_relevant_counts,
        "hard_relevant_counts_gt_0.5": hard_relevant_counts,
        "zero_relevant_queries": sum(count == 0 for count in relevant_counts),
        "mean_relevant_per_query": sum(relevant_counts) / len(relevant_counts),
        "mean_controlled_relevant_per_query": sum(controlled_relevant_counts)
        / len(controlled_relevant_counts),
        "mean_hard_relevant_per_query": sum(hard_relevant_counts) / len(hard_relevant_counts),
        "label_limited_max_mean_p10": sum(min(count, 10) / 10 for count in relevant_counts)
        / len(relevant_counts),
        "mean_oracle_runtime_seconds": sum(
            float(record["oracle_runtime_seconds"]) for record in records
        )
        / len(records),
        "max_oracle_runtime_seconds": max(
            float(record["oracle_runtime_seconds"]) for record in records
        ),
        "rascal_timed_out_pairs": sum(
            bool(record.get("rascal_timed_out", False)) for record in records
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/official"))
    parser.add_argument("--output-root", type=Path, default=Path("data/rascal-hard"))
    parser.add_argument("--dataset", choices=["MOLHIV", "MCF-7"], required=True)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--candidates", type=int, default=500)
    parser.add_argument("--controlled-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--parent-pool-size", type=int, default=650)
    parser.add_argument("--derived-per-parent", type=int, default=2)
    parser.add_argument("--relevance-threshold", type=float, default=0.5)
    parser.add_argument("--rascal-timeout", type=int, default=60)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    if args.candidates <= args.controlled_count:
        parser.error("--candidates must exceed --controlled-count")

    bank, training_fingerprints = collect_graph_bank(args.data_root, args.dataset)
    queries, pools, library, selection_audit = build_hard_protocol(
        bank,
        training_fingerprints,
        args.queries,
        args.candidates,
        args.controlled_count,
        args.seed,
        args.parent_pool_size,
        args.derived_per_parent,
    )
    dataset_root = (
        args.output_root
        / f"{args.candidates}-way"
        / f"seed-{args.seed}"
        / "retrieval"
        / args.dataset
    )
    raw = dataset_root / "raw" / "test"
    raw.mkdir(parents=True, exist_ok=True)
    manifest_path = dataset_root / "manifest.json"
    manifest = {
        "protocol": "RASCAL hard-negative mixed-edit retrieval v1",
        "dataset": args.dataset,
        "seed": args.seed,
        "recommended_formal_seeds": list(FORMAL_SEEDS),
        "queries": args.queries,
        "candidates": args.candidates,
        "pair_count": args.queries * args.candidates,
        "official_source_archive_sha256": OFFICIAL_DATA_SHA256,
        "selection_pool": "MCES val/test plus retrieval train/val/test",
        "exclusion": "all exact graph fingerprints in NEMA MCES train",
        "parent_disjointness": "query and candidate parent fingerprints are disjoint",
        "molecule_validity": "all parents and derived graphs pass full RDKit sanitization",
        "candidate_selection": (
            "hard candidates ranked only by 0.65*WL-cosine + 0.35*size-upper-bound; "
            "NEMA/NGA outputs are never used"
        ),
        "controlled_construction": (
            f"{args.controlled_count} deterministic variants/query (<10, so they cannot "
            "alone saturate P@10) across edge deletion, "
            "atom-label substitution, bond-label substitution, degree-preserving rewiring, "
            "and mixed edits; minimum two primitive edits and no identity variant"
        ),
        "relevance": "RASCAL similarity > threshold only; construction kinds are not labels",
        "rascal_similarity_threshold": args.relevance_threshold,
        "rascal_timeout_seconds_for_new_pairs": args.rascal_timeout,
        "selection_audit": selection_audit,
        "excluded_training_fingerprint_count": len(training_fingerprints),
        "query_graphs": [entry.manifest_record() for entry in queries],
        "candidate_library": [entry.manifest_record() for entry in library],
        "controlled_candidates": [
            [
                entry.candidate.manifest_record()
                for entry in pool
                if entry.candidate.candidate_kind.startswith("controlled_")
            ]
            for pool in pools
        ],
        "query_pools": [
            [entry.manifest_record() for entry in pool] for pool in pools
        ],
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        comparable = (
            "protocol",
            "dataset",
            "seed",
            "queries",
            "candidates",
            "candidate_selection",
            "controlled_construction",
            "query_graphs",
            "candidate_library",
            "controlled_candidates",
            "query_pools",
        )
        if any(previous.get(key) != manifest.get(key) for key in comparable):
            raise ValueError(f"refusing to change existing protocol: {manifest_path}")
    else:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    if args.manifest_only:
        print(json.dumps(selection_audit, indent=2), flush=True)
        return

    oracle_path = dataset_root / "oracle.jsonl"
    completed: dict[int, dict] = {}
    if oracle_path.exists():
        for line in oracle_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                completed[int(record["key"]) - 1] = record
    total = args.queries * args.candidates
    if args.limit is not None:
        total = min(total, args.limit)
    pending = [index for index in range(total) if index not in completed]

    global _QUERIES, _POOLS, _RAW, _CANDIDATE_COUNT
    global _RELEVANCE_THRESHOLD, _RASCAL_TIMEOUT_SECONDS, _PROTOCOL_SEED
    _QUERIES = queries
    _POOLS = pools
    _RAW = raw
    _CANDIDATE_COUNT = args.candidates
    _RELEVANCE_THRESHOLD = args.relevance_threshold
    _RASCAL_TIMEOUT_SECONDS = args.rascal_timeout
    _PROTOCOL_SEED = args.seed
    with oracle_path.open("a", encoding="utf-8") as stream:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_build_index, index): index for index in pending}
            for progress, future in enumerate(as_completed(futures), 1):
                record = future.result()
                completed[int(record["key"]) - 1] = record
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                if progress == 1 or progress % 100 == 0 or progress == len(pending):
                    print(
                        f"[{progress}/{len(pending)}] {args.dataset} seed={args.seed} "
                        f"key={record['key']} oracle={record['oracle_runtime_seconds']:.3f}s",
                        flush=True,
                    )
    if len(completed) == args.queries * args.candidates:
        summary = _summarize(list(completed.values()), args.candidates)
        (dataset_root / "oracle_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
