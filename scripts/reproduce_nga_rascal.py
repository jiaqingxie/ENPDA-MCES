#!/usr/bin/env python3
"""Run fixed, provenance-labelled RASCAL reproduction arms on exported NGA tests."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import rdkit
from rdkit import Chem
from rdkit.Chem import rdRascalMCES


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def molecule(graph, representation):
    if representation == "smiles":
        mol = Chem.MolFromSmiles(graph["smiles"])
        if mol is None:
            raise ValueError("Stored SMILES did not parse")
        assert [a.GetAtomicNum() for a in mol.GetAtoms()] == graph["nodes"], "SMILES node-order mismatch"
        edges = {(min(b.GetBeginAtomIdx(),b.GetEndAtomIdx()), max(b.GetBeginAtomIdx(),b.GetEndAtomIdx())):
                 int(b.GetBondType()) for b in mol.GetBonds()}
        expected = {(min(u,v),max(u,v)): label for u,v,label in graph["edges"]}
        assert edges == expected, "SMILES edge tensor mismatch"
        return mol, []
    edit = Chem.RWMol()
    for label in graph["nodes"]:
        edit.AddAtom(Chem.Atom(int(label)))
    for u, v, label in graph["edges"]:
        edit.AddBond(u, v, Chem.BondType.values[label])
    mol = edit.GetMol()
    if representation == "sanitized":
        # Mirrors upstream data.py:to_rdmol with kekulize=False.
        Chem.SanitizeMol(mol)
        Chem.AssignStereochemistry(mol)
    else:
        mol.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(mol)
    changed = [(u, v, label, int(mol.GetBondBetweenAtoms(u, v).GetBondType()))
               for u, v, label in graph["edges"]
               if label != int(mol.GetBondBetweenAtoms(u, v).GetBondType())]
    return mol, changed


def options(arm, timeout):
    opts = rdRascalMCES.RascalOptions()
    opts.similarityThreshold = 0.0
    opts.completeAromaticRings = arm["complete_rings"]
    opts.maxBondMatchPairs = 100000
    opts.timeout = timeout
    return opts


def option_dict(opts):
    out = {}
    for key in dir(opts):
        if not key.startswith("_"):
            val = getattr(opts, key)
            if isinstance(val, (bool, int, float, str)):
                out[key] = val
    return out


def solve(payload):
    pair, arm, timeout = payload
    base = {key: pair[key] for key in ("dataset", "key", "source_path", "source_sha256", "true_edges")}
    base.update({"arm": arm["name"], "rdkit": rdkit.__version__})
    started = time.perf_counter()
    if os.environ.get("RASCAL_REQUEST_START_FD"):
        ready_fd = int(os.environ["RASCAL_REQUEST_START_FD"])
        os.write(ready_fd, (str(started)+"\n").encode())
        os.close(ready_fd)
    try:
        left, lc = molecule(pair["left"], arm["representation"])
        right, rc = molecule(pair["right"], arm["representation"])
        search_started = time.perf_counter()
        matches = rdRascalMCES.FindMCES(left, right, options(arm, timeout))
        search_seconds = time.perf_counter() - search_started
        atoms = list(matches[0].atomMatches()) if matches else []
        bonds = list(matches[0].bondMatches()) if matches else []
        mapping = dict(atoms)
        injective = len(mapping) == len(atoms) == len(set(mapping.values()))
        atom_labels = all(pair["left"]["nodes"][u] == pair["right"]["nodes"][v] for u, v in atoms)
        le = {(min(u,v), max(u,v)): label for u,v,label in pair["left"]["edges"]}
        re = {(min(u,v), max(u,v)): label for u,v,label in pair["right"]["edges"]}
        compatible = []
        for (u,v), label in le.items():
            if u in mapping and v in mapping:
                dest = tuple(sorted((mapping[u],mapping[v])))
                if re.get(dest) == label:
                    compatible.append([u,v,mapping[u],mapping[v]])
        invalid_bonds = []
        witness = []
        for a,b in bonds:
            x,y = left.GetBondWithIdx(a), right.GetBondWithIdx(b)
            u,v = x.GetBeginAtomIdx(),x.GetEndAtomIdx()
            s,t = y.GetBeginAtomIdx(),y.GetEndAtomIdx()
            ok = (u in mapping and v in mapping and {mapping[u], mapping[v]} == {s,t}
                  and le.get(tuple(sorted((u,v)))) == re.get(tuple(sorted((s,t)))))
            witness.append([u,v,s,t])
            if not ok:
                invalid_bonds.append([u,v,s,t])
        valid = injective and atom_labels and not invalid_bonds
        return {**base, "status": "ok", "runtime_seconds": time.perf_counter()-started,
                "search_seconds": search_seconds, "timed_out": bool(matches[0].timedOut) if matches else False,
                "no_result": not bool(matches), "common_edges": len(bonds), "common_nodes": len(atoms),
                "accuracy": len(bonds)/pair["true_edges"], "atom_matches": atoms, "bond_matches": bonds,
                "bond_witness": witness, "mapping_common_edges": len(compatible), "mapping_injective": injective,
                "mapping_node_labels_valid": atom_labels, "invalid_bond_witness": invalid_bonds,
                "witness_valid": valid, "representation_changed_bonds": {"left": lc, "right": rc}}
    except Exception as exc:
        return {**base, "status": "error", "runtime_seconds": time.perf_counter()-started,
                "error": str(exc), "traceback": traceback.format_exc()}


def isolated_solve(payload):
    pair, arm, timeout = payload
    started = time.perf_counter()
    try:
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--single-pair"],
                                input=json.dumps(payload), text=True, capture_output=True, timeout=180)
        if result.returncode == 0:
            marker = "RASCAL_RECORD_JSON="
            lines = result.stdout.splitlines()
            encoded = [line[len(marker):] for line in lines if line.startswith(marker)]
            if len(encoded) == 1:
                row = json.loads(encoded[0])
                row["worker_stderr"] = result.stderr
                row["worker_native_stdout"] = "\n".join(line for line in lines if not line.startswith(marker))
                row["process_wall_seconds"] = time.perf_counter()-started
                return row
            error = "Native worker produced no unique structured result marker"
        else:
            error = f"Native worker exited with code {result.returncode}"
        stderr = result.stderr + "\nSTDOUT:\n" + result.stdout
    except subprocess.TimeoutExpired as exc:
        error = "Outer process safety deadline (180s) exceeded; not a normal 60s RASCAL incumbent"
        stderr = str(exc.stderr or "")
    return {**{k:pair[k] for k in ("dataset", "key", "source_path", "source_sha256", "true_edges")},
            "arm":arm["name"],"rdkit":rdkit.__version__,"status":"error","error":error,
            "worker_stderr":stderr,"runtime_seconds":time.perf_counter()-started}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifact-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    manifest_path = args.artifact_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    pairs_path = args.artifact_root / "pairs.json"
    assert sha(pairs_path) == manifest["data_sha256"]
    pairs = json.loads(pairs_path.read_text())
    arms = [a for a in manifest["arms"] if a["name"] in args.arms]
    assert len(arms) == len(args.arms)
    assert all(a["version"] == rdkit.__version__ for a in arms), rdkit.__version__
    args.output_root.mkdir(parents=True, exist_ok=True)
    provenance = {"rdkit": rdkit.__version__, "python": sys.version, "hostname": platform.node(),
                  "manifest_sha256": sha(manifest_path), "input_sha256": sha(pairs_path),
                  "script_sha256": sha(__file__), "workers": args.workers,
                  "cpu": subprocess.run(["lscpu"], capture_output=True, text=True).stdout,
                  "versions": subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout,
                  "started_utc": datetime.now(timezone.utc).isoformat(),
                  "options": {a["name"]: option_dict(options(a, manifest["timeout_seconds"])) for a in arms}}
    (args.output_root / f"environment_{rdkit.__version__}.json").write_text(json.dumps(provenance, indent=2))
    if args.prepare_only:
        checks = {}
        for arm in arms:
            errors, changes = [], []
            for pair in pairs:
                try:
                    for side in ("left", "right"):
                        _, changed = molecule(pair[side], arm["representation"])
                        if changed: changes.append([pair["source_path"], side, changed])
                except Exception as exc:
                    errors.append([pair["source_path"], str(exc)])
            checks[arm["name"]] = {"errors": errors, "changed_bond_labels": changes}
        print(json.dumps(checks, indent=2))
        return
    for arm in arms:
        dest = args.output_root / (arm["name"] + ".jsonl")
        if dest.exists() and not args.resume:
            raise FileExistsError(f"Refusing to overwrite {dest}")
        records = [json.loads(line) for line in dest.read_text().splitlines()] if dest.exists() else []
        completed = {r["source_path"] for r in records}
        assert len(completed) == len(records)
        lookup = {r["source_path"]:r for r in pairs}
        for row in records:
            assert row["arm"] == arm["name"] and row["rdkit"] == arm["version"]
            assert row["source_sha256"] == lookup[row["source_path"]]["source_sha256"]
        with dest.open("a" if args.resume else "x") as stream, ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(isolated_solve, (pair, arm, manifest["timeout_seconds"]))
                       for pair in pairs if pair["source_path"] not in completed]
            for future in as_completed(futures):
                rec = future.result()
                records.append(rec)
                stream.write(json.dumps(rec, ensure_ascii=False)+"\n")
                stream.flush()
                if len(records) % 25 == 0 or len(records) == len(pairs):
                    print(json.dumps({"arm": arm["name"], "completed": len(records), "total": len(pairs)}), flush=True)
        summaries = {}
        for dataset in ("AIDS", "MOLHIV", "MCF-7", "pooled"):
            selected = [r for r in records if dataset == "pooled" or r["dataset"] == dataset]
            ok = [r for r in selected if r["status"] == "ok"]
            summaries[dataset] = {"pairs": len(selected), "errors": len(selected)-len(ok),
                                  "invalid_witnesses": sum(not r["witness_valid"] for r in ok),
                                  "accuracy_percent": (100*sum(r["accuracy"] for r in ok)/len(selected)
                                                       if len(ok) == len(selected) and all(r["witness_valid"] for r in ok) else None),
                                  "raw_api_accuracy_percent_successful_only": 100*sum(r["accuracy"] for r in ok)/len(ok) if ok else None,
                                  "timeouts": sum(r["timed_out"] for r in ok),
                                  "no_results": sum(r["no_result"] for r in ok),
                                  "mean_seconds": sum(r["runtime_seconds"] for r in selected)/len(selected),
                                  "complete_validated_metric": len(ok) == len(selected) and all(r["witness_valid"] for r in ok)}
        summary = {"arm": arm, "datasets": summaries, "records_sha256": sha(dest),
                   "complete_utc": datetime.now(timezone.utc).isoformat(), "provenance": provenance}
        (args.output_root / (arm["name"] + ".summary.json")).write_text(json.dumps(summary, indent=2))
        print(json.dumps({"arm_complete": arm["name"], "datasets": summaries}), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--single-pair"]:
        import resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        print("\nRASCAL_RECORD_JSON=" + json.dumps(solve(json.load(sys.stdin))), flush=True)
    else:
        main()
