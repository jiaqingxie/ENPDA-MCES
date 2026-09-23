#!/usr/bin/env python3
"""Audit timeout-aware MRR/P@10/MAP intervals without mutating benchmark data."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch

from nema.retrieval_bounds import (
    label_metric_bounds,
    mrr_bounds,
    read_jsonl_unique_records,
    released_point_metrics,
    validate_relative_code_provenance,
    validate_oracle_grid,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_result(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--result must be METHOD=PATH")
    method, raw_path = value.split("=", 1)
    if not method or not raw_path:
        raise argparse.ArgumentTypeError("--result must be METHOD=PATH")
    return method, Path(raw_path)


def interval(lower: float, point: float, upper: float) -> dict[str, float | bool]:
    tolerance = 2e-6
    return {
        "lower": lower,
        "point": point,
        "upper": upper,
        "contains_point": lower - tolerance <= point <= upper + tolerance,
        "width": upper - lower,
    }


def method_bounds(
    result_path: Path,
    oracle_records: list[dict],
    classification: dict[int, dict],
    threshold: float,
    *,
    hard_only: bool,
) -> dict:
    result_records = read_jsonl_unique_records(result_path)
    predictions = {int(record["key"]): record for record in result_records}
    oracle_keys = {int(record["key"]) for record in oracle_records}
    if len(result_records) != len(oracle_records) or set(predictions) != oracle_keys:
        raise ValueError(f"result coverage differs from oracle: {result_path}")
    groups: dict[int, list[dict]] = {}
    for record in oracle_records:
        if hard_only and str(record["candidate_kind"]).startswith("controlled_"):
            continue
        groups.setdefault(int(record["query_index"]), []).append(record)

    per_query: list[dict] = []
    ap_proofs: set[str] = set()
    for query_index, records in sorted(groups.items()):
        records.sort(key=lambda record: int(record["key"]))
        predicted = [float(predictions[int(record["key"])]["similarity"]) for record in records]
        point_truth = [float(record["true_similarity"]) for record in records]
        lower: list[float] = []
        upper: list[float] = []
        labels: list[int | None] = []
        ambiguous_keys: list[int] = []
        for record in records:
            key = int(record["key"])
            if bool(record.get("rascal_timed_out", False)):
                classified = classification[key]
                lo = float(classified["rascal_lower_bound"])
                hi = float(classified["relaxation_upper_bound"])
            else:
                lo = hi = float(record["true_similarity"])
            lower.append(lo)
            upper.append(hi)
            if lo > threshold:
                labels.append(1)
            elif hi <= threshold:
                labels.append(0)
            else:
                labels.append(None)
                ambiguous_keys.append(key)

        point = released_point_metrics(predicted, point_truth, relevance_threshold=threshold)
        label_bounds, ap_proof = label_metric_bounds(predicted, labels)
        ap_proofs.add(ap_proof)
        query_mrr = mrr_bounds(predicted, lower, upper)
        per_query.append(
            {
                "query_index": query_index,
                "candidate_count": len(records),
                "ambiguous_keys": ambiguous_keys,
                "MRR": interval(query_mrr[0], point["MRR"], query_mrr[1]),
                "P@10": interval(
                    label_bounds["P@10"][0], point["P@10"], label_bounds["P@10"][1]
                ),
                "MAP": interval(
                    label_bounds["MAP"][0], point["MAP"], label_bounds["MAP"][1]
                ),
                "ap_bound_proof": ap_proof,
            }
        )

    aggregate: dict[str, dict] = {}
    for metric in ("MRR", "P@10", "MAP"):
        count = len(per_query)
        aggregate[metric] = interval(
            sum(float(query[metric]["lower"]) for query in per_query) / count,
            sum(float(query[metric]["point"]) for query in per_query) / count,
            sum(float(query[metric]["upper"]) for query in per_query) / count,
        )
    if not all(bool(value["contains_point"]) for value in aggregate.values()):
        raise AssertionError(f"computed interval does not cover point estimate: {result_path}")
    return {
        "result_path": str(result_path),
        "result_sha256": sha256(result_path),
        "candidate_count_per_query": len(next(iter(groups.values()))),
        "ambiguous_label_pairs": sum(len(query["ambiguous_keys"]) for query in per_query),
        "metrics": aggregate,
        "ap_bound_proofs": sorted(ap_proofs),
        "per_query": per_query,
    }


def pairwise_order(methods: dict[str, dict], slice_name: str) -> list[dict]:
    output = []
    names = sorted(methods)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            for metric in ("MRR", "P@10", "MAP"):
                a = methods[left][slice_name]["metrics"][metric]
                b = methods[right][slice_name]["metrics"][metric]
                if a["lower"] > b["upper"]:
                    relation = f"{left}>{right}"
                    certified = True
                elif b["lower"] > a["upper"]:
                    relation = f"{right}>{left}"
                    certified = True
                else:
                    relation = "intervals_overlap"
                    certified = False
                point_relation = (
                    f"{left}>{right}"
                    if a["point"] > b["point"]
                    else (f"{right}>{left}" if b["point"] > a["point"] else "point_tie")
                )
                output.append(
                    {
                        "metric": metric,
                        "left": left,
                        "right": right,
                        "point_relation": point_relation,
                        "interval_relation": relation,
                        "point_order_certified": certified and relation == point_relation,
                    }
                )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--classification", type=Path, required=True)
    parser.add_argument("--result", action="append", type=parse_result, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--runtime-provenance",
        type=Path,
        help="H100/CUDA12.8 provenance from the same process image; required before paper approval",
    )
    args = parser.parse_args()

    manifest_path = args.oracle.parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"oracle sibling manifest is required: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    oracle_records = read_jsonl_unique_records(args.oracle)
    queries, candidates = validate_oracle_grid(
        oracle_records,
        expected_queries=int(manifest["queries"]),
        expected_candidates=int(manifest["candidates"]),
    )
    classification_payload = json.loads(args.classification.read_text(encoding="utf-8"))
    if classification_payload.get("oracle_sha256") != sha256(args.oracle):
        raise ValueError("classification/oracle SHA mismatch")
    classified: dict[int, dict] = {}
    for record in classification_payload.get("records", []):
        key = int(record["key"])
        if key in classified:
            raise ValueError(f"duplicate classification key: {key}")
        classified[key] = record
    timed_out = {
        int(record["key"])
        for record in oracle_records
        if bool(record.get("rascal_timed_out", False))
    }
    if not timed_out <= set(classified):
        raise ValueError("classification does not cover every current oracle timeout")
    threshold = float(classification_payload["relevance_threshold"])
    if threshold != float(manifest["rascal_similarity_threshold"]):
        raise ValueError("classification/manifest relevance thresholds differ")

    runtime = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "provenance_path": None,
        "provenance_sha256": None,
        "h100_cuda128_verified": False,
    }
    if args.runtime_provenance is not None:
        provenance = json.loads(args.runtime_provenance.read_text(encoding="utf-8"))
        if "H100" not in str(provenance.get("device", "")):
            raise ValueError("metric-bounds runtime provenance is not H100")
        if provenance.get("cuda_runtime") != "12.8":
            raise ValueError("metric-bounds runtime provenance is not CUDA 12.8")
        if provenance.get("torch") != torch.__version__:
            raise ValueError("metric-bounds runtime/provenance torch versions differ")
        validate_relative_code_provenance(
            provenance.get("code_sha256", {}),
            (
                Path("src/nema/retrieval_bounds.py"),
                Path("scripts/audit_hard_retrieval_metric_bounds.py"),
            ),
        )
        runtime.update(
            {
                "provenance_path": str(args.runtime_provenance),
                "provenance_sha256": sha256(args.runtime_provenance),
                "h100_cuda128_verified": True,
                "provenance": provenance,
            }
        )

    methods: dict[str, dict] = {}
    for method, result_path in args.result:
        if method in methods:
            raise ValueError(f"duplicate method: {method}")
        methods[method] = {
            "full": method_bounds(
                result_path, oracle_records, classified, threshold, hard_only=False
            ),
            "hard_only": method_bounds(
                result_path, oracle_records, classified, threshold, hard_only=True
            ),
        }

    required = {
        key
        for key in timed_out
        if bool(classified[key].get("official_mrr_p10_map_resolution_required", True))
    }
    payload = {
        "schema_version": 1,
        "status": (
            "standalone_timeout_metric_bounds_audit_h100_runtime_review_pending"
            if runtime["h100_cuda128_verified"]
            else "local_development_metric_bounds_only_runtime_unverified"
        ),
        "oracle": str(args.oracle),
        "oracle_sha256": sha256(args.oracle),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "classification": str(args.classification),
        "classification_sha256": sha256(args.classification),
        "runtime": runtime,
        "semantics": {
            "relevance": f"strict true_similarity > {threshold}",
            "MRR": "reciprocal predicted rank of first original-order maximally true-similar candidate; NumPy argsort(pred)[::-1] tie behavior",
            "P@10": "relevant count in first min(10,n) NumPy-ranked candidates divided by 10, matching released evaluator",
            "MAP": "released PyTorch topk/sort AP; exact prefix DP when top-k sets are sort prefixes, otherwise exact equal-score tie-block label-relaxation DP within explicit complexity limits; rigorous [0,1] fallback beyond those limits",
        },
        "pair_count": len(oracle_records),
        "queries": queries,
        "candidates_per_query": candidates,
        "timed_out_pairs": len(timed_out),
        "timed_out_rate": len(timed_out) / len(oracle_records),
        "official_resolution_required_pairs": len(required),
        "official_resolution_required_rate": len(required) / len(oracle_records),
        "methods": methods,
        "pairwise_order": {
            slice_name: pairwise_order(methods, slice_name)
            for slice_name in ("full", "hard_only")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "timed_out_pairs": len(timed_out), "required_pairs": len(required)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
