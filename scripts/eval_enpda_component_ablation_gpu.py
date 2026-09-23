#!/usr/bin/env python3
"""Evaluate one frozen ENPDA component arm on one native dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.torch_version import TorchVersion

from nema.association import AssociationGraph
from nema.data import load_pairs, pair_paths
from nema.enpda_components import ARM_TO_MODE, component_forward
from nema.metrics import johnson_similarity


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/enpda_component_ablation.json"
CODE_PATHS = (
    ROOT / "src/nema/models/enpda.py",
    ROOT / "src/nema/enpda_components.py",
    Path(__file__).resolve(),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_model(path: Path, device: str):
    torch.serialization.add_safe_globals([TorchVersion])
    payload = torch.load(path, map_location="cpu", weights_only=True)
    from nema.models.enpda import ENPDAModel

    model = ENPDAModel(**payload["model_config"])
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval(), payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["AIDS", "MOLHIV", "MCF-7"], required=True)
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument("--arm", choices=sorted(ARM_TO_MODE), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise RuntimeError("formal component ablation refuses CPU execution")
    device_name = torch.cuda.get_device_name(0)
    if not any(name in device_name for name in config["hardware"]["accelerator_any"]):
        raise RuntimeError(f"formal component ablation requires H100/H200, found {device_name}")
    if not str(torch.version.cuda).startswith(config["hardware"]["cuda_runtime_prefix"]):
        raise RuntimeError(f"formal component ablation requires CUDA 12.8, found {torch.version.cuda}")

    model, checkpoint = _load_model(args.checkpoint, "cuda")
    if int(checkpoint.get("training_seed", -1)) != args.seed:
        raise RuntimeError("checkpoint training seed mismatch")
    config_hash = _sha256(config_path)
    checkpoint_hash = _sha256(args.checkpoint)
    code_hashes = {str(path.relative_to(ROOT)): _sha256(path) for path in CODE_PATHS}
    paths = pair_paths("data/official", args.dataset, split="test", limit=100)
    completed = {}
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                completed[row["source_path"]] = row

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as stream, torch.inference_mode():
        for index, path in enumerate(paths, 1):
            source = str(path)
            if source in completed:
                continue
            pairs = load_pairs(path)
            if len(pairs) != 1:
                raise RuntimeError(f"expected one pair in {path}")
            pair = pairs[0]
            started = time.perf_counter()
            left, right, swapped = pair.oriented()
            association = AssociationGraph.build(left, right)
            output = component_forward(model, association, args.arm)
            mapping = output.hard_mapping().cpu()
            edges, nodes = association.hard_statistics(mapping)
            elapsed = time.perf_counter() - started
            if swapped:
                inverse = torch.full((pair.left.num_nodes,), -1, dtype=torch.long)
                for oriented_left, oriented_right in enumerate(mapping.tolist()):
                    if 0 <= oriented_right < inverse.numel():
                        inverse[oriented_right] = oriented_left
                mapping = inverse
            similarity = johnson_similarity(
                nodes,
                edges,
                pair.left.num_nodes + pair.left.num_edges,
                pair.right.num_nodes + pair.right.num_edges,
            )
            row = {
                "key": pair.key,
                "source_path": source,
                "arm": args.arm,
                "mode": ARM_TO_MODE[args.arm],
                "seed": args.seed,
                "common_edges": edges,
                "common_nodes": nodes,
                "true_edges": pair.true_edges,
                "accuracy": None if pair.true_edges is None else edges / pair.true_edges,
                "similarity": similarity,
                "runtime_seconds": elapsed,
                "mapping": mapping.tolist(),
                "input_provenance": (pair.metadata or {}).get("input_provenance"),
                "final_price_mean": float(output.prices.mean()),
                "final_price_max": float(output.prices.max()),
                "max_column_excess": output.max_column_excess,
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            print(
                f"[{index}/{len(paths)}] {args.dataset} s{args.seed} {args.arm} "
                f"edges={edges}/{pair.true_edges} time={elapsed:.3f}s",
                flush=True,
            )

    rows = [json.loads(line) for line in args.output.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != len(paths) or {row["source_path"] for row in rows} != {str(path) for path in paths}:
        raise RuntimeError("component shard is incomplete")
    if {str(path.relative_to(ROOT)): _sha256(path) for path in CODE_PATHS} != code_hashes:
        raise RuntimeError("component code changed during evaluation")
    marker = {
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": config["protocol_version"],
        "config_sha256": config_hash,
        "dataset": args.dataset,
        "seed": args.seed,
        "arm": args.arm,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "hardware": {"device": device_name, "cuda_runtime": torch.version.cuda, "torch": torch.__version__},
        "code_sha256": code_hashes,
        "records": len(rows),
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
    }
    args.marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.marker.with_name(args.marker.name + ".tmp")
    temporary.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.marker)
    print(json.dumps(marker, indent=2), flush=True)


if __name__ == "__main__":
    main()
