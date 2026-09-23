#!/usr/bin/env python3
"""Join independent long-horizon dual bounds to the frozen ENPDA incumbent."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("AIDS", "MOLHIV", "MCF-7")
OUT = ROOT / "results/enpda_followups/long_certificate.json"
TEX = ROOT / "paper/generated/enpda_long_certificate.tex"


def main() -> None:
    envelope = json.loads((ROOT / "results/long-certificate/per_pair_envelope.json").read_text())
    incumbents = {}
    for dataset in DATASETS:
        path = ROOT / f"results/enpda_formal/native/seed0/enpda_solver_{dataset}.jsonl"
        for line in path.read_text().splitlines():
            row = json.loads(line)
            incumbents[row["source_path"]] = row
    summary = {}
    for budget in (10, 300, 1800):
        rows = []
        by_dataset = {}
        for source, record in envelope.items():
            item = next(value for value in record["envelope"] if int(value["budget_seconds"]) == budget)
            lower = int(incumbents[source]["common_edges"])
            upper = float(item["upper_bound"])
            if upper + 1e-7 < lower:
                raise RuntimeError(f"independent upper bound below ENPDA incumbent: {source}")
            gap = max(0.0, upper - lower) / max(abs(upper), 1.0)
            closed = upper < lower + 1.0 - 1e-7
            row = {"dataset": record["dataset"], "source_path": source, "lower": lower, "upper": upper, "gap": gap, "certified": closed}
            rows.append(row)
            by_dataset.setdefault(record["dataset"], []).append(row)
        summary[str(budget)] = {
            "pairs": len(rows),
            "certified": sum(row["certified"] for row in rows),
            "certified_percent": 100.0 * float(np.mean([row["certified"] for row in rows])),
            "median_gap_percent": 100.0 * float(np.median([row["gap"] for row in rows])),
            "by_dataset": {
                dataset: {
                    "pairs": len(values),
                    "certified": sum(row["certified"] for row in values),
                    "median_gap_percent": 100.0 * float(np.median([row["gap"] for row in values])),
                }
                for dataset, values in by_dataset.items()
            },
        }
    payload = {
        "status": "complete_postprocessing_only",
        "incumbent": "frozen ENPDA-Solver seed-0 hard mapping",
        "upper_bound": "independent sparse lifted MILP dual envelope",
        "summary": summary,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        r"\begin{table}[H]", r"\centering",
        r"\caption{Optional sparse lifted certificate on a frozen 75-pair audit.  ``Certified'' means the independent dual bound proves the frozen ENPDA-Solver incumbent globally optimal.}",
        r"\label{tab:long-certificate-enpda}",
        r"\setlength{\tabcolsep}{7pt}\renewcommand{\arraystretch}{.92}",
        r"\begin{tabular}{rrrr}", r"\toprule",
        r"Budget & Valid upper bounds & ENPDA incumbent certified & Median gap\\", r"\midrule",
    ]
    for budget in (10, 300, 1800):
        row = summary[str(budget)]
        lines.append(f"{budget}s & {row['pairs']}/{row['pairs']} & {row['certified']}/{row['pairs']} ({row['certified_percent']:.1f}\\%) & {row['median_gap_percent']:.2f}\\%\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    TEX.parent.mkdir(parents=True, exist_ok=True)
    TEX.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
