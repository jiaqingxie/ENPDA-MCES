#!/usr/bin/env python3
"""Resumable RASCAL MCES quality/runtime control on the released test pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from nema.data import load_pair, pair_paths
from nema.metrics import johnson_similarity
from nema.oracle import rascal_mces_truth


def _solve(payload: tuple[str, int]) -> dict:
    path_text, timeout_seconds = payload
    path = Path(path_text)
    pair = load_pair(path)
    started = time.perf_counter()
    result = rascal_mces_truth(pair.left, pair.right, timeout_seconds=timeout_seconds)
    runtime = time.perf_counter() - started
    edges, nodes = int(result["common_edges"]), int(result["common_nodes"])
    similarity = johnson_similarity(
        nodes,
        edges,
        pair.left.num_nodes + pair.left.num_edges,
        pair.right.num_nodes + pair.right.num_edges,
    )
    return {
        "method": "RASCAL",
        "key": pair.key,
        "source_path": path_text,
        "common_edges": edges,
        "common_nodes": nodes,
        "similarity": similarity,
        "runtime_seconds": runtime,
        "timed_out": bool(result["timed_out"]),
        "true_edges": pair.true_edges,
        "true_nodes": pair.true_nodes,
        "true_similarity": pair.true_similarity,
        "accuracy": edges / pair.true_edges if pair.true_edges else None,
        "similarity_squared_error": (
            (similarity - pair.true_similarity) ** 2
            if pair.true_similarity is not None
            else None
        ),
        "metadata": {"input_provenance": pair.metadata} if pair.metadata else {},
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/official"))
    parser.add_argument("--output-root", type=Path, default=Path("results/rascal-mces"))
    parser.add_argument("--datasets", nargs="+", default=["AIDS", "MOLHIV", "MCF-7"])
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--hardware", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict] = {}
    for dataset in args.datasets:
        output = args.output_root / f"rascal_{dataset}.jsonl"
        paths = pair_paths(args.data_root, dataset, split="test", limit=100)
        completed: dict[str, dict] = {}
        if output.exists():
            for line in output.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                source = str(record["source_path"])
                if source in completed:
                    raise ValueError(f"duplicate source path in {output}: {source}")
                completed[source] = record
        expected = {str(path) for path in paths}
        if set(completed) - expected:
            raise ValueError(f"{output} contains records outside the frozen test split")
        pending = [str(path) for path in paths if str(path) not in completed]
        with output.open("a", encoding="utf-8") as stream:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(_solve, (path, args.timeout_seconds)): path for path in pending
                }
                for index, future in enumerate(as_completed(futures), 1):
                    record = future.result()
                    completed[str(record["source_path"])] = record
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    if index == 1 or index % 10 == 0 or index == len(pending):
                        print(
                            f"[{dataset} {index}/{len(pending)}] "
                            f"edges={record['common_edges']}/{record['true_edges']} "
                            f"time={record['runtime_seconds']:.3f}s "
                            f"timeout={record['timed_out']}",
                            flush=True,
                        )
        if len(completed) != len(paths):
            raise RuntimeError(f"incomplete {dataset}: {len(completed)}/{len(paths)}")
        native = [
            completed[str(path)]
            for path in paths
            if not (completed[str(path)].get("metadata") or {}).get("input_provenance")
        ]
        accuracies = np.asarray([record["accuracy"] for record in native], dtype=float)
        runtimes = np.asarray([record["runtime_seconds"] for record in native], dtype=float)
        summaries[dataset] = {
            "released_pairs": len(paths),
            "native_pairs": len(native),
            "accuracy_percent": 100.0 * float(accuracies.mean()),
            "mean_runtime_seconds": float(runtimes.mean()),
            "median_runtime_seconds": float(np.median(runtimes)),
            "max_runtime_seconds": float(runtimes.max()),
            "timed_out_native_pairs": sum(bool(record["timed_out"]) for record in native),
            "result": str(output),
            "result_sha256": _sha256(output),
        }

    hardware = json.loads(args.hardware.read_text(encoding="utf-8"))
    if "H100" not in hardware.get("device", "") or hardware.get("cuda_runtime") != "12.8":
        raise ValueError("refusing non-H100/CUDA-12.8 RASCAL timing provenance")
    summary = {
        "protocol_version": "rascal-mces-runtime-v1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "timeout_seconds": args.timeout_seconds,
        "workers": args.workers,
        "note": "RASCAL is CPU-bound but timed inside the same formal H100/CUDA-12.8 job class.",
        "hardware": hardware,
        "datasets": summaries,
    }
    destination = args.output_root / "complete.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(destination)


if __name__ == "__main__":
    main()
