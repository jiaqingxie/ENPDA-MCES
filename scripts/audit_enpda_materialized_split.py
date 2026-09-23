"""Independently verify the serialized inputs consumed by corrected training."""
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/enpda_graph_disjoint_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((DATA / "manifest.json").read_text())
    graphs, pairs, counts = {}, {}, {}
    for split, f in manifest["materialized_files"].items():
        path = ROOT / f["path"]
        assert sha(path) == f["sha256"]
        rows = torch.load(path, map_location="cpu", weights_only=False)
        ids, pair_keys = set(), set()
        for r in rows:
            hashes = []
            for g in (r["pair"].left, r["pair"].right):
                d = {"nodes": g.node_labels.tolist(),
                     "edges": sorted((int(g.edge_index[0,k]), int(g.edge_index[1,k]), int(g.edge_labels[k]))
                                     for k in range(g.num_edges))}
                h = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()
                ids.add(h)
                hashes.append(h)
            pair_keys.add(tuple(sorted(hashes)))
            if split != "test":
                assert all(getattr(r["pair"], field) is None for field in
                           ("true_edges", "true_nodes", "true_similarity"))
        graphs[split], pairs[split], counts[split] = ids, pair_keys, len(rows)
        if split != "test":
            assert len(pair_keys) == len(rows), "Duplicate unordered training/validation pair"
    for a,b in (("train","validation"),("train","test"),("validation","test")):
        assert not graphs[a] & graphs[b]
        assert not pairs[a] & pairs[b]
    result = {"status": "passed", "manifest_sha256": sha(DATA / "manifest.json"),
              "method": "Rehash actual serialized graph tensors, independently of assigned identity IDs; original exact-isomorphism audit established that every repeated identity is an ordered-tensor repeat.",
              "counts": counts, "unique_graphs": {s:len(v) for s,v in graphs.items()},
              "graph_intersections": {"train_validation":0,"train_test":0,"validation_test":0},
              "pair_intersections": {"train_validation":0,"train_test":0,"validation_test":0},
              "training_validation_optimum_labels_absent": True}
    (DATA / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
