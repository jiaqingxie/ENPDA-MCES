"""Regenerate the input inventory and graph-disjoint split without saved results."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from nema.data import download_official_data, load_pairs, pair_paths

ROOT = Path(__file__).resolve().parents[1]
VALIDATION_IDS = {
    "AIDS": {1, 2, 4, 11, 13, 16, 34, 38, 51, 53, 54, 58, 62, 66, 74, 80, 84, 86},
    "MOLHIV": {1, 6},
    "MCF-7": {14, 21, 22, 30, 31, 34, 42, 43, 46, 49, 55, 56, 62, 69, 71, 74, 77, 81, 87, 92},
}


def main():
    os.chdir(ROOT)
    download_official_data(ROOT / "data/official")
    inventory = {"purpose": "Input enumeration only; final learning split is graph-disjoint.", "splits": {}}
    for dataset, validation in VALIDATION_IDS.items():
        files = {"training_files": [], "validation_files": []}
        paths = pair_paths(ROOT / "data/official", dataset, split="train")
        expected = {"AIDS": 90, "MOLHIV": 9, "MCF-7": 99}[dataset]
        assert len(paths) == expected, (dataset, len(paths), expected)
        for path in paths:
            number = int(path.stem.rsplit("_", 1)[1])
            key = "validation_files" if number in validation else "training_files"
            files[key].append({"path": str(path.relative_to(ROOT)),
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                               "pairs": len(load_pairs(path))})
        inventory["splits"][dataset] = files
    dest = ROOT / "data/native_input_inventory.json"
    value = json.dumps(inventory, indent=2) + "\n"
    if dest.exists() and dest.read_text() != value:
        raise RuntimeError("Existing input inventory differs; use a fresh checkout/data directory.")
    dest.write_text(value)
    for script in ("audit_enpda_graph_overlap.py", "prepare_enpda_graph_disjoint.py",
                   "audit_enpda_materialized_split.py", "prepare_release_controls.py"):
        subprocess.run([sys.executable, str(ROOT / "scripts" / script)], check=True)


if __name__ == "__main__":
    main()
