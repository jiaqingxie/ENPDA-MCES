"""Select the fixed 75-pair certificate sample directly from raw native inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from nema.data import pair_paths

ROOT = Path(__file__).resolve().parents[1]


def main():
    records = []
    for dataset in ("AIDS", "MOLHIV", "MCF-7"):
        paths = pair_paths(ROOT / "data/official", dataset)
        for position in np.rint(np.linspace(0, len(paths) - 1, 25)).astype(int):
            path = paths[int(position)]
            records.append({"dataset": dataset, "official_position": int(position),
                            "source_path": str(path.relative_to(ROOT)),
                            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "nema_lower_bound": 0})
    # The empty mapping is a valid independent incumbent. Fresh ENPDA incumbents
    # are joined only after solving; no historical solver result is required.
    dest = ROOT / "data/certificate_sample.jsonl"
    value = "".join(json.dumps(r) + "\n" for r in records)
    if dest.exists() and dest.read_text() != value:
        raise RuntimeError("Existing certificate sample differs")
    dest.write_text(value)


if __name__ == "__main__":
    main()
