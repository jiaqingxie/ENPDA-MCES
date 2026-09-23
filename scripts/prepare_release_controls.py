"""Generate compatibility paths for the current graph-disjoint protocol."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2) + "\n"
    if path.exists() and path.read_text() != text:
        raise RuntimeError(f"Refusing to change an existing generated configuration: {path}")
    path.write_text(text)


def main():
    manifest_path = ROOT / "data/enpda_graph_disjoint_v1/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    parent = {"native_test_paths_loaded": [],
              "graph_disjoint_manifest": "data/enpda_graph_disjoint_v1/manifest.json",
              "graph_disjoint_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "splits": {}}
    for dataset in ("AIDS", "MOLHIV", "MCF-7"):
        parent["splits"][dataset] = {
            key: [{"materialized_split": split, "dataset": dataset,
                   "pairs": manifest["counts"][split][dataset]}]
            for key, split in (("training_files", "train"), ("validation_files", "validation"))}
    write(ROOT / "data/sinkhorn_disjoint_parent.json", parent)
    for seed in range(3):
        target = ROOT / f"checkpoints/enpda_graph_disjoint_v1/formal/seed{seed}/full.best.pt"
        link = ROOT / f"checkpoints/enpda_formal/seed{seed}.best.pt"
        link.parent.mkdir(parents=True, exist_ok=True)
        expected = os.path.relpath(target, link.parent)
        if link.is_symlink():
            assert os.readlink(link) == expected
        elif link.exists():
            raise RuntimeError(f"Refusing to replace checkpoint: {link}")
        else:
            link.symlink_to(expected)


if __name__ == "__main__":
    main()
