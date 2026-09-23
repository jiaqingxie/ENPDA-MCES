"""Resumable benchmark runners for the three paper tables."""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from nema.data import load_pair, pair_paths
from nema.metrics import retrieval_metrics
from nema.models.nema import NEMAModel
from nema.official_nga import solve_official_nga_path
from nema.solvers import NGASolver, NEMASolver


def invalid_manifest_path(output: str | Path) -> Path:
    """Return the sidecar used to document unusable released pairs."""

    output = Path(output)
    return output.with_suffix(output.suffix + ".invalid.json")


def _preflight_paths(paths: list[Path]) -> tuple[list[Path], list[dict], list[dict]]:
    """Reject corrupt pairs before starting a long-lived CUDA worker pool."""

    valid: list[Path] = []
    invalid: list[dict] = []
    recovered: list[dict] = []
    for path in paths:
        try:
            pair = load_pair(path)
            if pair.left.num_nodes <= 0 or pair.right.num_nodes <= 0:
                invalid.append(
                    {
                        "source_path": str(path),
                        "reason": "released pair contains an empty graph",
                        "left_nodes": pair.left.num_nodes,
                        "right_nodes": pair.right.num_nodes,
                        "true_edges": pair.true_edges,
                        "true_nodes": pair.true_nodes,
                        "true_similarity": pair.true_similarity,
                    }
                )
            else:
                valid.append(path)
                if pair.metadata is not None:
                    recovered.append(
                        {
                            "source_path": str(path),
                            **pair.metadata,
                        }
                    )
        except Exception as error:
            invalid.append(
                {
                    "source_path": str(path),
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
    return valid, invalid, recovered


def load_nema_checkpoint(path: str | Path | None, device: str = "cpu") -> NEMAModel:
    if path is None:
        return NEMAModel()
    payload = torch.load(path, map_location=device, weights_only=True)
    config = payload.get("config", {})
    model = NEMAModel(**config)
    model.load_state_dict(payload["model"])
    return model


def _solve_path(payload: dict) -> dict:
    torch.set_num_threads(int(payload.get("torch_threads", 1)))
    started = time.perf_counter()
    pair = load_pair(payload["path"])
    method = payload["method"]
    if method == "nema":
        model = load_nema_checkpoint(payload.get("checkpoint"), payload["device"])
        solver = NEMASolver(
            model=model,
            continuous_restarts=payload["continuous_restarts"],
            discrete_restarts=payload["discrete_restarts"],
            refinement_passes=payload["refinement_passes"],
            anneal_steps=payload["anneal_steps"],
            lns_steps=payload["lns_steps"],
            certificate_seconds=payload["certificate_seconds"],
            seed=payload["seed"],
            device=payload["device"],
            trajectory=payload.get("nema_trajectory", "learned"),
            initializer_mode=payload.get("initializer_mode", "learned"),
            schedule_mode=payload.get("schedule_mode", "learned"),
            unit_fallback=payload.get("unit_fallback", True),
            line_search=payload.get("line_search", True),
            stationary_safeguard=payload.get("stationary_safeguard", True),
            sinkhorn_tolerance=payload.get("sinkhorn_tolerance", 1e-5),
            sinkhorn_max_iterations=payload.get("sinkhorn_max_iterations", 250),
            acceptance_tolerance=payload.get("acceptance_tolerance", 0.0),
        )
        result = solver.solve(pair)
    elif method == "nga":
        best = None
        aggregate_mse = None
        run_summaries = []
        for run in range(payload["nga_runs"]):
            candidate = solve_official_nga_path(
                payload["path"],
                epochs=payload["epochs"],
                learning_rate=payload["learning_rate"],
                samples=payload["samples"],
                time_budget=payload["time_budget"],
                seed=payload["seed"] + run,
                device=payload["device"],
            )
            run_summaries.append(
                {
                    "run": run,
                    "seed": payload["seed"] + run,
                    "common_edges": candidate.common_edges,
                    "common_nodes": candidate.common_nodes,
                    "similarity": candidate.similarity,
                    "similarity_squared_error": candidate.similarity_squared_error,
                    "runtime_seconds": candidate.runtime_seconds,
                    "epochs": (candidate.metadata or {}).get("epochs"),
                }
            )
            if best is None:
                best = candidate
                aggregate_mse = candidate.similarity_squared_error
            elif candidate.common_edges > best.common_edges:
                # This deliberately mirrors evaluate.py: a later launch is
                # selected only when MCES accuracy strictly improves, while its
                # Table-2 error is accumulated with min(previous, current).
                if aggregate_mse is None:
                    aggregate_mse = candidate.similarity_squared_error
                elif candidate.similarity_squared_error is not None:
                    aggregate_mse = min(aggregate_mse, candidate.similarity_squared_error)
                best = candidate
        result = best
    elif method == "nga-paper":
        best = None
        for run in range(payload["nga_runs"]):
            solver = NGASolver(
                epochs=payload["epochs"],
                learning_rate=payload["learning_rate"],
                samples=payload["samples"],
                time_budget=payload["time_budget"],
                refine=payload["nga_refine"],
                variant=payload["nga_variant"],
                seed=payload["seed"] + run,
                device=payload["device"],
            )
            candidate = solver.solve(pair)
            if best is None or (candidate.common_edges, candidate.common_nodes) > (
                best.common_edges,
                best.common_nodes,
            ):
                best = candidate
        result = best
    else:
        raise ValueError(f"unknown method {method}")
    assert result is not None
    record = result.to_dict()
    if method == "nga":
        record["similarity_squared_error"] = aggregate_mse
        metadata = dict(record.get("metadata") or {})
        metadata.update(
            {
                "best_of_runs": payload["nga_runs"],
                "run_summaries": run_summaries,
                "aggregate_wall_time_seconds": time.perf_counter() - started,
                "selection_protocol": "official evaluate.py strict accuracy improvement",
            }
        )
        record["metadata"] = metadata
    elif method == "nema":
        metadata = dict(record.get("metadata") or {})
        metadata["checkpoint"] = payload.get("checkpoint_provenance")
        record["metadata"] = metadata
    record["source_path"] = str(payload["path"])
    return record


def run_benchmark(
    data_root: str | Path,
    dataset: str,
    method: str,
    output: str | Path,
    retrieval: bool = False,
    split: str = "test",
    limit: int | None = None,
    workers: int = 1,
    device: str = "cpu",
    checkpoint: str | Path | None = None,
    seed: int = 0,
    continuous_restarts: int = 4,
    discrete_restarts: int = 8,
    refinement_passes: int = 20,
    anneal_steps: int = 1000,
    lns_steps: int = 100,
    certificate_seconds: float = 0.0,
    epochs: int = 200,
    learning_rate: float = 1e-3,
    samples: int = 10,
    time_budget: float = 60.0,
    nga_runs: int = 1,
    nga_refine: bool = False,
    nga_variant: str = "acg",
    nema_trajectory: str = "learned",
    initializer_mode: str = "learned",
    schedule_mode: str = "learned",
    unit_fallback: bool = True,
    line_search: bool = True,
    stationary_safeguard: bool = True,
    sinkhorn_tolerance: float | None = 1e-5,
    sinkhorn_max_iterations: int | None = 250,
    acceptance_tolerance: float = 0.0,
) -> dict[str, object]:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = pair_paths(data_root, dataset, split, retrieval, limit)
    valid_paths, invalid, recovered = _preflight_paths(paths)
    manifest = invalid_manifest_path(output)
    manifest_payload = {
        "dataset": dataset,
        "split": split,
        "task": "retrieval" if retrieval else "mces",
        "released_pair_count": len(paths),
        "valid_pair_count": len(valid_paths),
        "invalid_pair_count": len(invalid),
        "recovered_pair_count": len(recovered),
        "invalid": invalid,
        "recovered": recovered,
    }
    manifest.write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if invalid:
        print(
            f"warning: excluded {len(invalid)}/{len(paths)} unusable released "
            f"{dataset} {'retrieval' if retrieval else 'MCES'} pairs; details: {manifest}",
            flush=True,
        )
    if recovered:
        print(
            f"notice: recovered {len(recovered)}/{len(paths)} released {dataset} pairs "
            f"from stored SMILES; details: {manifest}",
            flush=True,
        )
    completed: set[str] = set()
    if output.exists():
        for line in output.read_text().splitlines():
            if line.strip():
                source_path = json.loads(line)["source_path"]
                if source_path in completed:
                    raise ValueError(f"duplicate source_path in resumable output: {source_path}")
                completed.add(source_path)
    valid_sources = {str(path) for path in valid_paths}
    unexpected = completed - valid_sources
    if unexpected:
        examples = ", ".join(sorted(unexpected)[:3])
        raise ValueError(
            f"output contains {len(unexpected)} pairs outside this benchmark selection; "
            f"use a fresh output path (examples: {examples})"
        )
    indexed_pending = [
        (index, path)
        for index, path in enumerate(valid_paths)
        if str(path) not in completed
    ]

    checkpoint_provenance = None
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint)
        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        history = checkpoint_payload.get("history", [])
        checkpoint_provenance = {
            "path": str(checkpoint_path),
            "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            "completed_epochs": checkpoint_payload.get("completed_epochs"),
            "selected_soft_objective": history[-1]["soft_objective"] if history else None,
        }

    common = {
        "method": method,
        "device": device,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_provenance": checkpoint_provenance,
        "seed": seed,
        "continuous_restarts": continuous_restarts,
        "discrete_restarts": discrete_restarts,
        "refinement_passes": refinement_passes,
        "anneal_steps": anneal_steps,
        "lns_steps": lns_steps,
        "certificate_seconds": certificate_seconds,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "samples": samples,
        "time_budget": time_budget,
        "nga_runs": nga_runs,
        "nga_refine": nga_refine,
        "nga_variant": nga_variant,
        "nema_trajectory": nema_trajectory,
        "initializer_mode": initializer_mode,
        "schedule_mode": schedule_mode,
        "unit_fallback": unit_fallback,
        "line_search": line_search,
        "stationary_safeguard": stationary_safeguard,
        "sinkhorn_tolerance": sinkhorn_tolerance,
        "sinkhorn_max_iterations": sinkhorn_max_iterations,
        "acceptance_tolerance": acceptance_tolerance,
        "torch_threads": max(1, 64 // max(workers, 1)) if workers == 1 else 1,
    }
    payloads = [
        {**common, "path": str(path), "seed": seed + index * 10007}
        for index, path in indexed_pending
    ]

    with output.open("a", encoding="utf-8") as stream:
        if workers <= 1:
            iterator = enumerate(payloads, 1)
            for index, payload in iterator:
                record = _solve_path(payload)
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                print(
                    f"[{index}/{len(payloads)}] {dataset} {method.upper()} "
                    f"edges={record['common_edges']}/{record['true_edges']} "
                    f"time={record['runtime_seconds']:.2f}s",
                    flush=True,
                )
        else:
            # CUDA cannot be initialized safely in a forked child.  The CLI's
            # device validation already touches the CUDA runtime, so GPU worker
            # pools must use fresh interpreters.
            mp_context = mp.get_context("spawn") if device.startswith("cuda") else None
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp_context) as pool:
                futures = {pool.submit(_solve_path, payload): payload for payload in payloads}
                for index, future in enumerate(as_completed(futures), 1):
                    record = future.result()
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    print(
                        f"[{index}/{len(payloads)}] {dataset} {method.upper()} "
                        f"edges={record['common_edges']}/{record['true_edges']} "
                        f"time={record['runtime_seconds']:.2f}s",
                        flush=True,
                    )
    summary = summarize_results(output, retrieval=retrieval)
    summary.update(
        {
            key: value
            for key, value in manifest_payload.items()
            if key not in {"invalid", "recovered"}
        }
    )
    summary["invalid_manifest"] = str(manifest)
    return summary


def _record_sort_key(record: dict) -> tuple[int, int]:
    key = str(record["key"])
    head, _, tail = key.partition(":")
    return int(head), int(tail or 0)


def summarize_results(
    path: str | Path,
    retrieval: bool = False,
    allow_partial_retrieval: bool = False,
    exclude_recovered: bool = False,
) -> dict[str, object]:
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    all_record_count = len(records)
    if exclude_recovered:
        records = [
            record
            for record in records
            if not (record.get("metadata") or {}).get("input_provenance")
        ]
    records.sort(key=_record_sort_key)
    if not records:
        return {"count": 0}
    accuracies = [record.get("accuracy") for record in records]
    valid_accuracies = [value for value in accuracies if value is not None]
    summary = {
        "count": len(records),
        "accuracy_percent": (
            100.0 * float(np.mean(valid_accuracies)) if valid_accuracies else None
        ),
        "mse_x1e3": 1000.0
        * float(np.mean([r["similarity_squared_error"] for r in records])),
        "mean_runtime_seconds": float(np.mean([r["runtime_seconds"] for r in records])),
        "optimal_solution_percent": (
            100.0
            * float(np.mean([r["common_edges"] == r["true_edges"] for r in records]))
            if len(valid_accuracies) == len(records)
            else None
        ),
    }
    if exclude_recovered:
        summary["excluded_recovered_count"] = all_record_count - len(records)
    aggregate_runtime = [
        (record.get("metadata") or {}).get("aggregate_wall_time_seconds")
        for record in records
    ]
    if all(runtime is not None for runtime in aggregate_runtime):
        summary["mean_aggregate_runtime_seconds"] = float(np.mean(aggregate_runtime))
    if retrieval:
        if len(records) % 100:
            if allow_partial_retrieval and len(records) < 100:
                summary.update(
                    retrieval_metrics(
                        [r["similarity"] for r in records],
                        [r["true_similarity"] for r in records],
                        group_size=len(records),
                    )
                )
                summary["partial_retrieval"] = True
                summary["candidate_group_size"] = len(records)
                summary["retrieval_warning"] = (
                    f"partial ranking over {len(records)} candidates; not comparable "
                    "to the paper's 100-candidate metric"
                )
            else:
                summary["retrieval_warning"] = "pair count is not divisible by 100"
        else:
            summary.update(
                retrieval_metrics(
                    [r["similarity"] for r in records],
                    [r["true_similarity"] for r in records],
                )
            )
    return summary
