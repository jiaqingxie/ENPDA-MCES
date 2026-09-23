#!/usr/bin/env python3
"""Natural D&D protein-graph scaling benchmark for ENPDA and direct Sinkhorn controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.torch_version import TorchVersion
from torch_geometric.datasets import TUDataset

from nema.association import AssociationGraph
from nema.graph import LabeledGraph
from nema.models.enpda import ENPDAModel
from nema.models.sinkhorn_baseline import TrainOnceSinkhorn


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/enpda_dd_scaling.json"
CODE_PATHS = (
    ROOT / "src/nema/models/enpda.py",
    ROOT / "src/nema/models/sinkhorn_baseline.py",
    ROOT / "src/nema/association.py",
    ROOT / "src/nema/features.py",
    ROOT / "src/nema/rounding.py",
    Path(__file__).resolve(),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def load_config() -> dict:
    config = json.loads(CONFIG.read_text())
    if not config.get("frozen_before_results") or not config["models"]["parameters_frozen"]:
        raise RuntimeError("D&D scaling protocol is not frozen")
    return config


def dataset(config: dict) -> TUDataset:
    for name, expected in config["dataset"]["raw_sha256"].items():
        path = ROOT / config["dataset"]["root"] / "DD/raw" / name
        if sha256(path) != expected:
            raise RuntimeError(f"D&D source hash mismatch: {path}")
    return TUDataset(str(ROOT / config["dataset"]["root"]), name="DD", use_node_attr=True)


def graph(data) -> LabeledGraph:
    if data.x is None or data.x.ndim != 2:
        raise RuntimeError("D&D graph is missing original one-hot node labels")
    labels = data.x.argmax(dim=1).detach().cpu().long()
    raw = data.edge_index.detach().cpu().long()
    edges = set()
    for index in range(raw.shape[1]):
        u, v = int(raw[0, index]), int(raw[1, index])
        if u != v:
            edges.add((u, v) if u < v else (v, u))
    ordered = sorted(edges)
    edge_index = (
        torch.tensor(ordered, dtype=torch.long).t().contiguous()
        if ordered
        else torch.empty((2, 0), dtype=torch.long)
    )
    return LabeledGraph(labels, edge_index, torch.zeros(len(ordered), dtype=torch.long))


def graph_digest(value: LabeledGraph) -> str:
    return digest(
        {
            "nodes": value.node_labels.tolist(),
            "edges": value.edge_index.t().tolist(),
            "edge_labels": value.edge_labels.tolist(),
        }
    )


def selected_pairs(config: dict, collection: TUDataset) -> list[dict]:
    used = set()
    result = []
    per_bin = int(config["dataset"]["pairs_per_bin"])
    selection_seed = int(config["dataset"]["selection_seed"])
    for label, bounds in config["dataset"]["size_bins"].items():
        lower, upper = map(int, bounds)
        candidates = [
            index
            for index, item in enumerate(collection)
            if lower <= int(item.num_nodes) < upper and index not in used
        ]
        candidates.sort(key=lambda index: hashlib.sha256(f"{selection_seed}:{label}:{index}".encode()).hexdigest())
        chosen = candidates[: 2 * per_bin]
        if len(chosen) != 2 * per_bin:
            raise RuntimeError(f"not enough D&D graphs for bin {label}")
        used.update(chosen)
        for pair_index in range(per_bin):
            left_index, right_index = chosen[2 * pair_index : 2 * pair_index + 2]
            left, right = graph(collection[left_index]), graph(collection[right_index])
            if left.num_nodes > right.num_nodes:
                left_index, right_index, left, right = right_index, left_index, right, left
            result.append(
                {
                    "key": f"DD-{label}-{pair_index}",
                    "size_bin": label,
                    "left_index": left_index,
                    "right_index": right_index,
                    "left_nodes": left.num_nodes,
                    "right_nodes": right.num_nodes,
                    "left_edges": left.num_edges,
                    "right_edges": right.num_edges,
                    "left_sha256": graph_digest(left),
                    "right_sha256": graph_digest(right),
                }
            )
    return result


def prepare() -> None:
    config = load_config()
    target = ROOT / config["outputs"]["manifest"]
    raw = ROOT / config["outputs"]["raw"]
    if raw.exists() and not target.exists():
        raise RuntimeError("scaling result exists before its manifest")
    collection = dataset(config)
    manifest = {
        "protocol_version": config["protocol_version"],
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256(CONFIG),
        "code_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in CODE_PATHS},
        "pairs": selected_pairs(config, collection),
        "result_files_present_when_frozen": [str(raw.relative_to(ROOT))] if raw.exists() else [],
    }
    if target.exists():
        previous = json.loads(target.read_text())
        for key in ("protocol_version", "config_sha256", "code_sha256", "pairs"):
            if previous.get(key) != manifest.get(key):
                raise RuntimeError(f"existing scaling manifest differs at {key}")
        print(f"manifest already valid: {target}")
        return
    atomic_write(target, json.dumps(manifest, indent=2) + "\n")
    print(f"froze {len(manifest['pairs'])} natural D&D pairs in {target}")


def hardware_guard(config: dict) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("formal D&D scaling refuses CPU")
    name = torch.cuda.get_device_name(0)
    if not any(value in name for value in config["hardware"]["accelerator_any"]):
        raise RuntimeError(f"requires H100/H200, found {name}")
    if not str(torch.version.cuda).startswith(str(config["hardware"]["cuda_runtime_prefix"])):
        raise RuntimeError(f"requires CUDA 12.8, found {torch.version.cuda}")
    return {"device": name, "torch": torch.__version__, "cuda_runtime": torch.version.cuda}


def load_models(config: dict) -> tuple[ENPDAModel, TrainOnceSinkhorn, dict]:
    torch.serialization.add_safe_globals([TorchVersion])
    paths = {name: ROOT / path for name, path in (
        ("enpda", config["models"]["enpda_checkpoint"]),
        ("sinkhorn", config["models"]["sinkhorn_checkpoint"]),
    )}
    enpda_payload = torch.load(paths["enpda"], map_location="cpu", weights_only=True)
    sinkhorn_payload = torch.load(paths["sinkhorn"], map_location="cpu", weights_only=True)
    enpda = ENPDAModel(**enpda_payload["model_config"])
    enpda.load_state_dict(enpda_payload["model"], strict=True)
    sinkhorn = TrainOnceSinkhorn(**sinkhorn_payload["model_config"])
    sinkhorn.load_state_dict(sinkhorn_payload["model"], strict=True)
    return enpda.cuda().eval(), sinkhorn.cuda().eval(), {name: sha256(path) for name, path in paths.items()}


def timed_arm(name: str, call, association: AssociationGraph, upper: int) -> dict:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    started = time.perf_counter()
    try:
        result = call()
        torch.cuda.synchronize()
        forward = time.perf_counter() - started
        round_started = time.perf_counter()
        if name == "gumbel":
            mappings = result
            scored = [(association.hard_statistics(mapping), mapping) for mapping in mappings]
            best = max(range(len(scored)), key=lambda index: scored[index][0])
            stats, mapping = scored[best]
        else:
            output = result
            mapping = output.hard_mapping().cpu()
            stats = association.hard_statistics(mapping)
            best = 0
        rounding = time.perf_counter() - round_started
        return {
            "status": "ok",
            "common_edges": stats[0],
            "common_nodes": stats[1],
            "trivial_upper_edges": upper,
            "lower_over_trivial_upper": stats[0] / max(upper, 1),
            "mapping": mapping.tolist(),
            "selected_sample": best if name == "gumbel" else None,
            "forward_seconds": forward,
            "hungarian_seconds": rounding,
            "peak_cuda_incremental_bytes": max(torch.cuda.max_memory_allocated() - baseline, 0),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
    except torch.cuda.OutOfMemoryError as error:
        torch.cuda.empty_cache()
        return {"status": "oom", "error": str(error)}
    except RuntimeError as error:
        return {"status": "failure", "error": str(error)}


def evaluate() -> None:
    config = load_config()
    manifest_path = ROOT / config["outputs"]["manifest"]
    manifest = json.loads(manifest_path.read_text())
    if manifest["config_sha256"] != sha256(CONFIG):
        raise RuntimeError("scaling config changed after freeze")
    if manifest["code_sha256"] != {str(path.relative_to(ROOT)): sha256(path) for path in CODE_PATHS}:
        raise RuntimeError("scaling code changed after freeze")
    hardware = hardware_guard(config)
    collection = dataset(config)
    models = load_models(config)
    enpda, sinkhorn, checkpoint_sha = models
    output_path = ROOT / config["outputs"]["raw"]
    completed = {}
    if output_path.exists():
        for line in output_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                completed[row["key"]] = row
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Warm both models outside the measured region on the first natural pair.
    first = manifest["pairs"][0]
    warm_association = AssociationGraph.build(graph(collection[first["left_index"]]), graph(collection[first["right_index"]]))
    with torch.inference_mode():
        enpda(warm_association, mode="learned")
        sinkhorn(warm_association)
    torch.cuda.synchronize()

    with output_path.open("a", encoding="utf-8") as stream:
        for position, spec in enumerate(manifest["pairs"], 1):
            if spec["key"] in completed:
                continue
            left, right = graph(collection[spec["left_index"]]), graph(collection[spec["right_index"]])
            if graph_digest(left) != spec["left_sha256"] or graph_digest(right) != spec["right_sha256"]:
                raise RuntimeError(f"natural graph drift: {spec['key']}")
            build_started = time.perf_counter()
            association = AssociationGraph.build(left, right)
            build_seconds = time.perf_counter() - build_started
            upper = min(left.num_edges, right.num_edges)
            with torch.inference_mode():
                enpda_result = timed_arm("enpda", lambda: enpda(association, mode="learned"), association, upper)
                sinkhorn_result = timed_arm("sinkhorn", lambda: sinkhorn(association), association, upper)
                generator = torch.Generator(device="cuda").manual_seed(2026083000 + position)
                gumbel_result = timed_arm(
                    "gumbel",
                    lambda: sinkhorn.gumbel_mappings(
                        association,
                        samples=int(config["models"]["gumbel_samples"]),
                        generator=generator,
                    )[0],
                    association,
                    upper,
                )
            record = {
                "protocol_version": config["protocol_version"],
                "manifest_sha256": sha256(manifest_path),
                "checkpoint_sha256": checkpoint_sha,
                **spec,
                "full_grid_candidates": association.num_candidates,
                "compatible_candidates": int(association.candidate_mask.sum()),
                "acg_edges": association.num_edges,
                "acg_build_seconds": build_seconds,
                "enpda": enpda_result,
                "sinkhorn": sinkhorn_result,
                "gumbel": gumbel_result,
            }
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            print(
                f"[{position}/{len(manifest['pairs'])}] {spec['key']} "
                f"nodes={left.num_nodes}x{right.num_nodes} build={build_seconds:.2f}s "
                f"status={enpda_result['status']}/{sinkhorn_result['status']}/{gumbel_result['status']}",
                flush=True,
            )
    rows = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
    if len(rows) != len(manifest["pairs"]) or {row["key"] for row in rows} != {row["key"] for row in manifest["pairs"]}:
        raise RuntimeError("D&D scaling coverage mismatch")
    marker = {
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": config["protocol_version"],
        "config_sha256": sha256(CONFIG),
        "manifest_sha256": sha256(manifest_path),
        "output_sha256": sha256(output_path),
        "records": len(rows),
        "hardware": hardware,
    }
    atomic_write(ROOT / config["outputs"]["completion"], json.dumps(marker, indent=2) + "\n")
    print(json.dumps(marker, indent=2))


def finalize() -> None:
    config = load_config()
    raw_path = ROOT / config["outputs"]["raw"]
    completion_path = ROOT / config["outputs"]["completion"]
    marker = json.loads(completion_path.read_text())
    if marker["output_sha256"] != sha256(raw_path) or marker["config_sha256"] != sha256(CONFIG):
        raise RuntimeError("D&D scaling completion mismatch")
    rows = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
    summary = {}
    for label in config["dataset"]["size_bins"]:
        subset = [row for row in rows if row["size_bin"] == label]
        item = {
            "pairs": len(subset),
            "median_nodes": float(np.median([(row["left_nodes"] + row["right_nodes"]) / 2 for row in subset])),
            "median_full_grid_candidates": float(np.median([row["full_grid_candidates"] for row in subset])),
            "median_compatible_candidates": float(np.median([row["compatible_candidates"] for row in subset])),
            "median_acg_edges": float(np.median([row["acg_edges"] for row in subset])),
            "median_acg_build_seconds": float(np.median([row["acg_build_seconds"] for row in subset])),
        }
        for arm in ("enpda", "sinkhorn", "gumbel"):
            valid = [row[arm] for row in subset if row[arm]["status"] == "ok"]
            item[arm] = {
                "successes": len(valid),
                "median_forward_seconds": float(np.median([row["forward_seconds"] for row in valid])) if valid else None,
                "median_hungarian_seconds": float(np.median([row["hungarian_seconds"] for row in valid])) if valid else None,
                "median_peak_incremental_gib": float(np.median([row["peak_cuda_incremental_bytes"] for row in valid]) / 2**30) if valid else None,
                "median_lower_over_trivial_upper": float(np.median([row["lower_over_trivial_upper"] for row in valid])) if valid else None,
            }
        summary[label] = item
    payload = {
        "status": "complete_and_audited",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": config["protocol_version"],
        "summary": summary,
    }
    summary_path = ROOT / config["outputs"]["summary"]
    atomic_write(summary_path, json.dumps(payload, indent=2) + "\n")

    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Natural non-molecular scaling on graph-disjoint D\&D protein pairs.  Graphs are paired without planted correspondences or edit-derived targets.  Times and memory are medians over five pairs; $L/U_0$ is the returned legal lower bound divided by the trivial edge upper bound, not exact MCES accuracy.}",
        r"\label{tab:enpda-dd-scaling}",
        r"\setlength{\tabcolsep}{3.2pt}\renewcommand{\arraystretch}{.92}",
        r"\begin{tabular}{rrrrrrrr}", r"\toprule",
        r"Nodes & Grid & Compatible & ACG edges & Build s & ENPDA s & Peak GiB & $L/U_0$\\", r"\midrule",
    ]
    for label in config["dataset"]["size_bins"]:
        item = summary[label]
        arm = item["enpda"]
        lines.append(
            f"{item['median_nodes']:.0f} & {item['median_full_grid_candidates']/1000:.1f}k & "
            f"{item['median_compatible_candidates']/1000:.1f}k & {item['median_acg_edges']/1000:.1f}k & "
            f"{item['median_acg_build_seconds']:.2f} & {arm['median_forward_seconds'] + arm['median_hungarian_seconds']:.3f} & "
            f"{arm['median_peak_incremental_gib']:.2f} & {arm['median_lower_over_trivial_upper']:.3f}\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    atomic_write(ROOT / config["outputs"]["table"], "\n".join(lines))

    labels = list(config["dataset"]["size_bins"])
    nodes = [summary[label]["median_nodes"] for label in labels]
    build = [summary[label]["median_acg_build_seconds"] for label in labels]
    enpda_time = [summary[label]["enpda"]["median_forward_seconds"] + summary[label]["enpda"]["median_hungarian_seconds"] for label in labels]
    memory = [summary[label]["enpda"]["median_peak_incremental_gib"] for label in labels]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65))
    axes[0].plot(nodes, build, "o-", label="Sparse ACG build")
    axes[0].plot(nodes, enpda_time, "s-", label="ENPDA + Hungarian")
    axes[0].set_yscale("log"); axes[0].set_xlabel("Median nodes per graph"); axes[0].set_ylabel("Median seconds (log)"); axes[0].grid(alpha=.25); axes[0].legend(frameon=False)
    axes[1].plot(nodes, memory, "o-", color="#0b7285")
    axes[1].set_xlabel("Median nodes per graph"); axes[1].set_ylabel("Peak incremental GPU GiB"); axes[1].grid(alpha=.25)
    fig.tight_layout()
    figure_path = ROOT / config["outputs"]["figure"]
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "evaluate", "finalize"])
    args = parser.parse_args()
    {"prepare": prepare, "evaluate": evaluate, "finalize": finalize}[args.command]()


if __name__ == "__main__":
    main()
