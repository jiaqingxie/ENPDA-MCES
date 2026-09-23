"""Combine freshly computed valid certificate bounds across time budgets."""
from __future__ import annotations

import json
from pathlib import Path

from nema.data import load_pair

ROOT = Path(__file__).resolve().parents[1]


def main():
    envelope = {}
    for dataset in ("AIDS", "MOLHIV", "MCF-7"):
        for budget in (10, 300, 1800):
            path = ROOT / f"results/long-certificate/{budget}s/certificate_{dataset}.jsonl"
            records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            assert len(records) == 25 and len({r["source_path"] for r in records}) == 25
            for row in records:
                source = row["source_path"]
                pair = load_pair(ROOT / source)
                record = envelope.setdefault(source, {"dataset": dataset, "envelope": []})
                previous = record["envelope"][-1] if record["envelope"] else None
                lower = max(int(row["combined_lower_bound"]), previous["lower_bound"] if previous else 0)
                upper = min(pair.left.num_edges, pair.right.num_edges)
                if row["upper_bound"] is not None:
                    upper = min(upper, float(row["upper_bound"]))
                if previous:
                    upper = min(upper, previous["upper_bound"])
                assert lower <= upper + 1e-7
                record["envelope"].append({"budget_seconds": budget, "lower_bound": lower,
                                           "upper_bound": upper, "certified_optimal": upper < lower + 1 - 1e-7})
    dest = ROOT / "results/long-certificate/per_pair_envelope.json"
    dest.write_text(json.dumps(envelope, indent=2) + "\n")


if __name__ == "__main__":
    main()
