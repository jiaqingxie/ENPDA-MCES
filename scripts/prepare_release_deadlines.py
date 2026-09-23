"""Build deadline-comparison inputs and observer from source, without old results."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import subprocess
from pathlib import Path

from nema.data import pair_paths

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def graph(data):
    edges = {}
    for (u, v), label in zip(data.edge_index.t().tolist(), data.edge_attr.tolist(), strict=True):
        key = tuple(sorted((int(u), int(v))))
        if key in edges:
            assert edges[key] == int(label)
        edges[key] = int(label)
    return {"nodes": data.x.reshape(-1).tolist(),
            "edges": [[u, v, label] for (u, v), label in edges.items()],
            "smiles": getattr(data, "smiles", None)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule", choices=("unrestricted", "complete_aromatic_cycles"), required=True)
    args = parser.parse_args()
    import rdkit
    if rdkit.__version__ != "2024.03.5":
        raise RuntimeError("The native incumbent observer requires rdkit==2024.3.5; use the deadline environment.")
    lib = next((Path(rdkit.__file__).parent.parent / "rdkit.libs").glob("*RascalMCES*"))
    symbols = subprocess.check_output(["nm", "-D", str(lib)], text=True)
    source = (ROOT / "scripts/rascal_incumbent_trace.cpp").read_text()
    import re
    symbol = re.search(r'const char \*symbol = "([^"]+)"', source).group(1)
    if symbol not in symbols:
        raise RuntimeError("RDKit binary does not expose the pinned observer symbol")
    dest = ROOT / "artifacts/deadlines" / args.rule
    if dest.exists():
        raise RuntimeError(f"Use a fresh deadline artifact directory: {dest}")
    dest.mkdir(parents=True)
    shutil.copytree(ROOT / "src", dest / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("run_aromatic_ring_trial.py", "reproduce_nga_rascal.py",
                 "reproduce_nga_rascal_incumbent60.py", "rascal_incumbent_trace.cpp"):
        shutil.copyfile(ROOT / "scripts" / name, dest / name)
    (dest / "vendor").symlink_to("../../../vendor", target_is_directory=True)
    subprocess.run(["g++", "-O2", "-std=c++17", "-shared", "-fPIC",
                    str(dest / "rascal_incumbent_trace.cpp"), "-ldl", "-o",
                    str(dest / "rascal_incumbent_trace.so")], check=True)
    pairs = []
    for dataset in ("AIDS", "MOLHIV", "MCF-7"):
        for path in pair_paths(ROOT / "data/official", dataset):
            left, right = pickle.loads(path.read_bytes())
            assert len(left) == len(right) == 1
            a, b = left[0], right[0]
            assert min(int(a.num_nodes), int(b.num_nodes)) > 0
            truth = a.y.reshape(-1).tolist()
            pairs.append({"dataset": dataset, "key": path.stem.rsplit("_", 1)[1],
                          "source_path": str(path.relative_to(ROOT)), "source_sha256": sha(path),
                          "left": graph(a), "right": graph(b),
                          "true_edges": int(round(truth[0])), "true_nodes": int(round(truth[1]))})
    assert len(pairs) == 291
    (dest / "pairs.json").write_text(json.dumps(pairs, indent=2) + "\n")
    methods = ["enpda", "rascal_unfiltered"]
    if args.rule == "complete_aromatic_cycles":
        methods += ["rascal", "nga", "nga_anytime"]
    protocol = {"output_rule": args.rule, "budget_seconds": 60, "methods": methods,
                "nga": {"seeds": [0, 1, 2]}, "pairs_sha256": sha(dest / "pairs.json"),
                "selection": "All 291 sanitized native tests; no outcome-dependent filtering.",
                "checkpoint_sha256": {str(s): sha(ROOT / f"checkpoints/enpda_graph_disjoint_v1/formal/seed{s}/full.best.pt") for s in range(3)},
                "code_sha256": {str(p.relative_to(dest)): sha(p) for p in dest.rglob("*")
                                if p.is_file() and p.suffix in (".py", ".cpp", ".so")}}
    (dest / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(dest)


if __name__ == "__main__":
    main()
