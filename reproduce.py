#!/usr/bin/env python3
"""Portable, inspectable entry point for ENPDA reproduction.

Use --dry-run to print every command before launching the GPU experiments.
Stages use fresh outputs; --resume skips only previous successful commands.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASETS = ("AIDS", "MOLHIV", "MCF-7")


def command(script, *args):
    return [sys.executable, "scripts/" + script, *map(str, args)]


def commands(stage, seeds):
    from scripts.release_plan import evaluation_tasks

    if stage == "bootstrap":
        return [command("bootstrap.py")]
    if stage == "prepare":
        return [command("prepare_release_inputs.py")]
    if stage == "smoke":
        return [command("run_enpda_graph_disjoint.py", "--seed", s, "--device", "cpu", "--smoke") for s in seeds]
    if stage == "train":
        return [command("run_enpda_graph_disjoint.py", "--seed", s, "--device", "cuda") for s in seeds]
    tasks = evaluation_tasks()
    prefixes = {"native": ("native-",), "controls": ("sink-", "comp-", "search-", "price-"),
                "rounds": ("rounds", "round-time"), "ood": ("imdb-", "protein-", "dd-", "short-")}
    if stage in prefixes:
        selected = [t for t in tasks if t["key"].startswith(prefixes[stage])
                    and set(t["seeds"]).issubset(seeds)]
        result = []
        if stage == "ood":
            result = [command("prepare_enpda_nonmolecular_ood.py"),
                      command("run_enpda_protein_ood.py", "prepare"),
                      command("run_enpda_dd_scaling.py", "prepare"),
                      command("run_enpda_submission_followups.py", "--suite", "prepare"),
                      command("certify_enpda_natural_followup.py"),
                      command("certify_natural_clique_reduction.py"),
                      command("merge_enpda_natural_certificates.py")]
        return result + [[sys.executable, *c] for t in selected for c in t["commands"]]
    if stage == "baselines":
        result = []
        for dataset in DATASETS:
            for seed in seeds:
                folder = f"results/baselines/nga/seed{seed}"
                result.append([sys.executable, "-m", "nema.cli", "benchmark", "--dataset", dataset,
                               "--method", "nga-paper", "--device", "cuda", "--seed", str(seed),
                               "--nga-runs", "3", "--workers", "1", "--epochs", "200",
                               "--time-budget", "60", "--output", f"{folder}/{dataset}.jsonl"])
                folder = f"results/native-no-network/seed{seed}"
                result.append(command("eval_native_no_network_solver_gpu.py", "--dataset", dataset,
                                      "--search-seed", seed, "--output", f"{folder}/no_network_{dataset}.jsonl",
                                      "--attestation", f"{folder}/complete_{dataset}.json"))
        return result
    if stage == "retrieval":
        result = []
        for seed in seeds:
            for dataset in ("MOLHIV", "MCF-7"):
                common = ("--dataset", dataset, "--model-seed", seed)
                result.append(command("run_enpda_retrieval_disjoint_v2.py", *common, "--oracle-only"))
                for method in ("simgnn", "gmn", "neuromatch"):
                    result.append(command("run_unified_retrieval_baseline_gpu.py", *common,
                                          "--method", method, "--protocol-seed", 20260823 + seed,
                                          "--hard-root", "data/enpda_retrieval_disjoint_v2/500-way"))
                result.append(command("run_enpda_retrieval_disjoint_v2.py", *common))
        return result
    if stage == "certificates":
        result = [command("prepare_release_certificates.py")]
        for budget in (10, 300, 1800):
            for dataset in DATASETS:
                folder = f"results/long-certificate/{budget}s"
                result.append(command("run_long_certificate.py", "--dataset", dataset, "--budget", budget,
                                      "--output", f"{folder}/certificate_{dataset}.jsonl",
                                      "--attestation", f"{folder}/complete_{dataset}.json"))
        return result + [command("summarize_release_certificates.py"), command("analyze_enpda_long_certificate.py")]
    if stage in ("deadlines", "aromatic"):
        rule = "unrestricted" if stage == "deadlines" else "complete_aromatic_cycles"
        result = [command("prepare_release_deadlines.py", "--rule", rule)]
        for seed in seeds:
            methods = ["enpda"]
            if stage == "aromatic":
                methods += ["nga", "nga_anytime"]
            if seed == 0:
                methods += ["rascal_unfiltered"]
                if stage == "aromatic":
                    methods += ["rascal"]
            result.append([sys.executable, f"artifacts/deadlines/{rule}/run_aromatic_ring_trial.py",
                           "--root", str(ROOT), "--output", f"results/{stage}/seed{seed}",
                           "--seed", str(seed), "--workers", "1", "--budget", "60", "--methods", *methods])
        return result
    if stage == "summarize":
        return [command("summarize_release.py")]
    raise ValueError(stage)


def source_digest():
    digest = hashlib.sha256()
    paths = [ROOT / "reproduce.py", ROOT / "pyproject.toml"]
    for folder in ("src", "scripts", "configs"):
        paths.extend(p for p in (ROOT / folder).rglob("*") if p.suffix in (".py", ".json", ".cpp"))
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("bootstrap", "prepare", "smoke", "train", "native", "controls",
                                         "rounds", "baselines", "ood", "retrieval", "certificates",
                                         "deadlines", "aromatic", "summarize", "all"))
    parser.add_argument("--seeds", nargs="+", type=int, choices=(0, 1, 2), default=[0, 1, 2])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.stage in ("rounds", "ood", "deadlines", "aromatic", "all") and set(args.seeds) != {0, 1, 2}:
        parser.error("This stage uses all three frozen checkpoints; use --seeds 0 1 2")
    os.chdir(ROOT)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH", "")])
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = "1"
    stages = (["bootstrap", "prepare", "train", "native", "controls", "rounds", "baselines",
               "ood", "retrieval", "certificates", "summarize"] if args.stage == "all" else [args.stage])
    fingerprint = source_digest()
    for stage in stages:
        for argv in commands(stage, args.seeds):
            print(shlex.join(argv), flush=True)
            if args.dry_run:
                continue
            key = hashlib.sha256(json.dumps(argv).encode()).hexdigest()
            receipt = ROOT / "artifacts/reproduction_receipts" / (key + ".json")
            if args.resume and receipt.exists():
                previous = json.loads(receipt.read_text())
                if previous["source_sha256"] != fingerprint:
                    raise RuntimeError("Source/configuration changed since the completed command; use a fresh run.")
                print("Completed command recorded; skipping. Keep its generated outputs intact.", flush=True)
                continue
            subprocess.run(argv, cwd=ROOT, env=env, check=True)
            receipt.parent.mkdir(parents=True, exist_ok=True)
            receipt.write_text(json.dumps({"command": argv, "source_sha256": fingerprint, "exit_code": 0}, indent=2) + "\n")


if __name__ == "__main__":
    main()
