#!/usr/bin/env python3
"""Train and evaluate one protocol-matched neural retrieval baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch

from nema.data import load_pairs, recover_empty_molecular_data
from nema.graph import GraphPair, LabeledGraph
from nema.metrics import retrieval_metrics
from nema.retrieval_baselines import EncodedGraph, build_retrieval_baseline


def numeric_key(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[-1])


def graph_digest(graph: LabeledGraph) -> str:
    digest = hashlib.sha256()
    digest.update(graph.node_labels.numpy().tobytes())
    digest.update(graph.edge_index.numpy().tobytes())
    digest.update(graph.edge_labels.numpy().tobytes())
    return digest.hexdigest()


def load_training_split(
    data_root: Path,
    dataset: str,
    validation_fraction: float,
) -> tuple[list[GraphPair], list[GraphPair], dict[str, object]]:
    root = data_root / "MCES" / f"{dataset}-train" / "raw"
    paths = sorted(root.glob("graphs_*.pkl"), key=numeric_key)
    if not paths:
        raise FileNotFoundError(root)
    validation_files = max(1, int(round(len(paths) * validation_fraction)))
    split = len(paths) - validation_files
    train = [pair for path in paths[:split] for pair in load_pairs(path)]
    validation = [pair for path in paths[split:] for pair in load_pairs(path)]
    validation_fingerprints = {
        graph_digest(graph)
        for pair in validation
        for graph in (pair.left, pair.right)
    }
    before = len(train)
    train = [
        pair
        for pair in train
        if graph_digest(pair.left) not in validation_fingerprints
        and graph_digest(pair.right) not in validation_fingerprints
    ]
    return train, validation, {
        "source_files": len(paths),
        "train_files": split,
        "validation_files": validation_files,
        "train_pairs_before_graph_disjoint_filter": before,
        "train_pairs": len(train),
        "validation_pairs": len(validation),
        "train_validation_graph_overlap": 0,
    }


def encode_pairs(
    pairs: list[GraphPair], device: torch.device
) -> tuple[list[tuple[str, str, float]], dict[str, EncodedGraph]]:
    cache: dict[str, EncodedGraph] = {}
    encoded: list[tuple[str, str, float]] = []
    for pair in pairs:
        if pair.true_similarity is None:
            raise ValueError(f"missing similarity target for training pair {pair.key}")
        left_key, right_key = graph_digest(pair.left), graph_digest(pair.right)
        cache.setdefault(left_key, EncodedGraph.from_graph(pair.left, device))
        cache.setdefault(right_key, EncodedGraph.from_graph(pair.right, device))
        encoded.append((left_key, right_key, float(pair.true_similarity)))
    return encoded, cache


@torch.no_grad()
def validation_mse(
    model: torch.nn.Module,
    records: list[tuple[str, str, float]],
    cache: dict[str, EncodedGraph],
) -> float:
    model.eval()
    errors = []
    for left, right, target in records:
        prediction = model(cache[left], cache[right])
        errors.append(float(torch.square(prediction - target)))
    return float(np.mean(errors))


def train_model(
    model: torch.nn.Module,
    train_records: list[tuple[str, str, float]],
    train_cache: dict[str, EncodedGraph],
    validation_records: list[tuple[str, str, float]],
    validation_cache: dict[str, EncodedGraph],
    *,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    accumulation: int,
    seed: int,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float]], int]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    best_mse = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    history: list[dict[str, float]] = []
    stale = 0
    for epoch in range(epochs):
        started = time.perf_counter()
        model.train()
        order = torch.randperm(len(train_records), generator=generator).tolist()
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        for position, index in enumerate(order, 1):
            left, right, target = train_records[index]
            prediction = model(train_cache[left], train_cache[right])
            loss = torch.square(prediction - target)
            (loss / accumulation).backward()
            train_loss += float(loss.detach())
            if position % accumulation == 0 or position == len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        score = validation_mse(model, validation_records, validation_cache)
        row = {
            "epoch": epoch + 1,
            "train_mse": train_loss / len(order),
            "validation_mse": score,
            "seconds": time.perf_counter() - started,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if score < best_mse - 1e-8:
            best_mse = score
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    return best_state, history, best_epoch


def load_oracle(path: Path) -> list[dict[str, object]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    records.sort(key=lambda row: int(row["key"]))
    return records


def load_test_cache(
    records: list[dict[str, object]], device: torch.device
) -> dict[str, EncodedGraph]:
    cache: dict[str, EncodedGraph] = {}
    for progress, record in enumerate(records, 1):
        left_key = str(record["query_fingerprint"])
        right_key = str(record["candidate_fingerprint"])
        if left_key in cache and right_key in cache:
            continue
        path = Path(str(record["source_path"]))
        with path.open("rb") as stream:
            left_list, right_list = pickle.load(stream)
        left_data, _ = recover_empty_molecular_data(left_list[0])
        right_data, _ = recover_empty_molecular_data(right_list[0])
        if left_key not in cache:
            cache[left_key] = EncodedGraph.from_graph(LabeledGraph.from_pyg(left_data), device)
        if right_key not in cache:
            cache[right_key] = EncodedGraph.from_graph(LabeledGraph.from_pyg(right_data), device)
        if progress % 1000 == 0:
            print(f"[cache] {progress}/{len(records)} records; {len(cache)} unique graphs", flush=True)
    return cache


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    records: list[dict[str, object]],
    cache: dict[str, EncodedGraph],
    output: Path,
    method: str,
) -> tuple[dict[str, float], float]:
    model.eval()
    predictions: list[float] = []
    targets: list[float] = []
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with output.open("w", encoding="utf-8") as stream:
        for index, record in enumerate(records, 1):
            pair_started = time.perf_counter()
            score = float(
                model(
                    cache[str(record["query_fingerprint"])],
                    cache[str(record["candidate_fingerprint"])],
                )
            )
            target = float(record["true_similarity"])
            predictions.append(score)
            targets.append(target)
            stream.write(
                json.dumps(
                    {
                        "method": method,
                        "key": str(record["key"]),
                        "similarity": score,
                        "true_similarity": target,
                        "runtime_seconds": time.perf_counter() - pair_started,
                        "source_path": record["source_path"],
                    }
                )
                + "\n"
            )
            if index % 1000 == 0:
                print(f"[eval] {index}/{len(records)}", flush=True)
    elapsed = time.perf_counter() - started
    return retrieval_metrics(predictions, targets, group_size=500), elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/unified_retrieval_baselines.json"))
    parser.add_argument("--dataset", choices=("MOLHIV", "MCF-7"), required=True)
    parser.add_argument("--method", choices=("simgnn", "gmn", "neuromatch"), required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--protocol-seed", type=int, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/official"))
    parser.add_argument("--hard-root", type=Path, default=Path("data/rascal-hard/500-way"))
    parser.add_argument("--output-root", type=Path, default=Path("results/unified_retrieval"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints/unified_retrieval"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise RuntimeError("this formal experiment requires CUDA")
    device = torch.device("cuda")
    random.seed(args.model_seed)
    np.random.seed(args.model_seed)
    torch.manual_seed(args.model_seed)
    torch.cuda.manual_seed_all(args.model_seed)
    print(
        {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "dataset": args.dataset,
            "method": args.method,
            "model_seed": args.model_seed,
            "protocol_seed": args.protocol_seed,
        },
        flush=True,
    )

    train_pairs, validation_pairs, split_audit = load_training_split(
        args.data_root,
        args.dataset,
        float(config["validation_fraction_by_source_file"]),
    )
    train_records, train_cache = encode_pairs(train_pairs, device)
    validation_records, validation_cache = encode_pairs(validation_pairs, device)
    model = build_retrieval_baseline(
        args.method,
        hidden_dim=int(config["hidden_dim"]),
        layers=int(config["message_passing_layers"]),
    ).to(device)
    training_started = time.perf_counter()
    best_state, history, best_epoch = train_model(
        model,
        train_records,
        train_cache,
        validation_records,
        validation_cache,
        epochs=int(config["epochs"]),
        patience=int(config["patience"]),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        accumulation=int(config["gradient_accumulation_pairs"]),
        seed=args.model_seed,
    )
    training_seconds = time.perf_counter() - training_started
    model.load_state_dict(best_state)
    checkpoint_dir = args.checkpoint_root / args.dataset / args.method
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"seed{args.model_seed}.best.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "config": config,
            "dataset": args.dataset,
            "method": args.method,
            "model_seed": args.model_seed,
            "best_epoch": best_epoch,
            "history": history,
            "split_audit": split_audit,
        },
        checkpoint,
    )

    dataset_root = args.hard_root / f"seed-{args.protocol_seed}" / "retrieval" / args.dataset
    oracle = load_oracle(dataset_root / "oracle.jsonl")
    expected = int(config["queries"]) * int(config["candidates_per_query"])
    if len(oracle) != expected:
        raise ValueError(f"expected {expected} test records, found {len(oracle)}")
    excluded_training_graphs = set(train_cache) | set(validation_cache)
    test_graphs = {
        str(record[key])
        for record in oracle
        for key in ("query_fingerprint", "candidate_fingerprint")
    }
    overlap = excluded_training_graphs & test_graphs
    if overlap:
        raise RuntimeError(
            f"retrieval leakage audit failed: {len(overlap)} train/test graph fingerprints overlap"
        )
    test_cache = load_test_cache(oracle, device)
    result_dir = args.output_root / f"seed-{args.protocol_seed}" / args.dataset / args.method
    predictions_path = result_dir / f"model_seed{args.model_seed}.jsonl"
    metrics, evaluation_seconds = evaluate(model, oracle, test_cache, predictions_path, args.method)
    summary = {
        "status": "complete",
        "protocol": config["protocol"],
        "dataset": args.dataset,
        "method": args.method,
        "model_seed": args.model_seed,
        "protocol_seed": args.protocol_seed,
        "checkpoint": str(checkpoint),
        "best_epoch": best_epoch,
        "training_seconds": training_seconds,
        "evaluation_seconds": evaluation_seconds,
        "mean_inference_seconds_per_pair": evaluation_seconds / len(oracle),
        "metrics": metrics,
        "split_audit": split_audit,
        "training_test_graph_overlap": 0,
        "training_and_validation_unique_graphs": len(excluded_training_graphs),
        "test_pair_count": len(oracle),
        "unique_test_graphs": len(test_cache),
        "predictions": str(predictions_path),
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / f"model_seed{args.model_seed}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
