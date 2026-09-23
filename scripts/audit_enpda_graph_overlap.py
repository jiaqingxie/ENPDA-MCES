"""Audit actual ENPDA split graphs with exact label-preserving isomorphism.

WL hashes only select candidates; VF2 establishes identity. Stored SMILES are
audited separately and are not used to reconstruct or change model inputs.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import torch

from nema.data import load_pairs, pair_paths

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/enpda_graph_overlap"
MANIFEST = ROOT / "data/native_input_inventory.json"
DATASETS = ("AIDS", "MOLHIV", "MCF-7")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    warnings.filterwarnings("ignore", category=FutureWarning)
    manifest = json.loads(MANIFEST.read_text())
    buckets = defaultdict(list)
    representatives = []
    occurrences = []
    pairs = []
    inputs = []
    exact_tensor_cache = {}
    vf2_checks = 0
    for dataset in DATASETS:
        paths = {"train": manifest["splits"][dataset]["training_files"],
                 "validation": manifest["splits"][dataset]["validation_files"]}
        paths["test"] = [{"path": str(p.relative_to(ROOT))}
                         for p in pair_paths(ROOT / "data/official", dataset)]
        assert len(paths["test"]) == (91 if dataset == "MOLHIV" else 100)
        for split, files in paths.items():
            for f in files:
                path = ROOT / f["path"]
                sha = digest(path)
                assert "sha256" not in f or f["sha256"] == sha, str(path)
                with path.open("rb") as stream:
                    raw = pickle.load(stream)
                loaded = load_pairs(path)
                if "pairs" in f:
                    assert len(loaded) == f["pairs"]
                inputs.append({"dataset": dataset, "split": split,
                               "path": f["path"], "sha256": sha, "pairs": len(loaded)})
                for index, pair in enumerate(loaded):
                    pair_record = {"dataset": dataset, "split": split, "path": f["path"],
                                   "index": index, "identities": []}
                    for side, graph in enumerate((pair.left, pair.right)):
                        graph_data = {"nodes": graph.node_labels.tolist(),
                                      "edges": sorted((int(graph.edge_index[0,k]),
                                                       int(graph.edge_index[1,k]),
                                                       int(graph.edge_labels[k]))
                                                      for k in range(graph.num_edges))}
                        ordered_hash = hashlib.sha256(json.dumps(graph_data, sort_keys=True).encode()).hexdigest()
                        if ordered_hash in exact_tensor_cache:
                            identity = exact_tensor_cache[ordered_hash]
                        else:
                            g = nx.Graph()
                            g.add_nodes_from((i, {"label": int(l)}) for i,l in enumerate(graph.node_labels))
                            g.add_edges_from((u,v,{"label": l}) for u,v,l in graph_data["edges"])
                            wl = nx.weisfeiler_lehman_graph_hash(g, node_attr="label", edge_attr="label", iterations=4)
                            key = (g.number_of_nodes(), g.number_of_edges(), wl)
                            identity = None
                            for candidate in buckets[key]:
                                vf2_checks += 1
                                if nx.is_isomorphic(g, representatives[candidate],
                                                    node_match=nx.algorithms.isomorphism.categorical_node_match("label", None),
                                                    edge_match=nx.algorithms.isomorphism.categorical_edge_match("label", None)):
                                    identity = candidate
                                    break
                            if identity is None:
                                identity = len(representatives)
                                representatives.append(g)
                                buckets[key].append(identity)
                            exact_tensor_cache[ordered_hash] = identity
                        pair_record["identities"].append(identity)
                        occurrences.append({"dataset": dataset, "split": split, "path": f["path"],
                                            "pair_index": index, "side": side, "identity": identity,
                                            "ordered_tensor_sha256": ordered_hash,
                                            "smiles": getattr(raw[side][index], "smiles", None),
                                            "nodes": graph.num_nodes, "edges": graph.num_edges,
                                            "recovered": bool(pair.metadata and str(("left","right")[side]) in pair.metadata.get("recovered_sides",[]))})
            print(dataset, split, "files",len(files), "unique identities so far",len(representatives),flush=True)

    def summarize(selected_occurrences, selected_pairs):
        sets = {s: {o["identity"] for o in selected_occurrences if o["split"]==s}
                for s in ("train","validation","test")}
        smiles_sets = {s: {o["smiles"] for o in selected_occurrences if o["split"]==s and o["smiles"]}
                       for s in sets}
        comparisons = {}
        for a,b in (("train","validation"),("train","test"),("validation","test")):
            common = sets[a] & sets[b]
            comparisons[f"{a}__{b}"] = {
                "shared_labeled_graph_identities":len(common),
                "shared_stored_smiles":len(smiles_sets[a]&smiles_sets[b]),
                "affected_pairs_in_second_split":sum(bool(set(p["identities"]) & common)
                                                       for p in selected_pairs if p["split"]==b)}
        exposed = sets["train"] | sets["validation"]
        test_pairs = [p for p in selected_pairs if p["split"]=="test"]
        neither = [p for p in test_pairs if not set(p["identities"]) & exposed]
        return {"pair_counts":dict(Counter(p["split"] for p in selected_pairs)),
                "graph_occurrences":dict(Counter(o["split"] for o in selected_occurrences)),
                "unique_labeled_graphs":{s:len(v) for s,v in sets.items()},
                "overlaps":comparisons,
                "test_pairs_with_neither_graph_in_train_or_validation":len(neither),
                "test_pairs_with_both_graphs_in_train_or_validation":sum(set(p["identities"])<=exposed for p in test_pairs),
                "retained_test_paths":[p["path"] for p in neither]}

    # Pair records are constructed from occurrence order, independent of any method outcome.
    grouped = {}
    for o in occurrences:
        key=(o["dataset"],o["split"],o["path"],o["pair_index"])
        if key not in grouped:
            grouped[key]={"dataset":o["dataset"],"split":o["split"],"path":o["path"],
                          "index":o["pair_index"],"identities":[]}
        grouped[key]["identities"].append(o["identity"])
    pairs=list(grouped.values())
    assert Counter(p["split"] for p in pairs) == {"train":1989,"validation":522,"test":291}
    results={d:summarize([o for o in occurrences if o["dataset"]==d],
                         [p for p in pairs if p["dataset"]==d]) for d in DATASETS}
    global_result=summarize(occurrences,pairs)
    global_exposed={o["identity"] for o in occurrences if o["split"] in ("train","validation")}
    for d in DATASETS:
        test=[p for p in pairs if p["dataset"]==d and p["split"]=="test"]
        results[d]["test_pairs_disjoint_from_all_datasets_train_validation"]=[p["path"] for p in test if not set(p["identities"]) & global_exposed]
    report={"status":"complete","manifest_sha256":digest(MANIFEST),
            "identity":"Exact node- and edge-label-preserving isomorphism of model-input simple graphs; ordered tensor hashes cache repeats and WL hashes only prefilter VF2 checks.",
            "scope":"Frozen 1989 student-training pairs, 522 validation pairs, and sanitized 291 native test pairs. Different graphs with identical labeled input topology are conservatively treated as the same identity.",
            "networkx_version":nx.__version__,"vf2_checks":vf2_checks,
            "unique_graphs_global":len(representatives),"datasets":results,"global":global_result,
            "input_files":inputs,
            "limitations":"Stored SMILES equality is reported separately; this is not a stereochemistry-aware molecular identity audit. Native test selection retains the predeclared sanitation. Audit and any retained subset depend only on identities, not prediction quality. Teacher pretraining scope must be checked separately."}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    (OUT/'graph_occurrences.jsonl').write_text(''.join(json.dumps(o)+'\n' for o in occurrences))
    (OUT/'pairs.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in pairs))
    print(json.dumps({"pair_counts": global_result["pair_counts"], "unique_graphs": len(representatives)}, indent=2))


if __name__=="__main__":
    main()
