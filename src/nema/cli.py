"""Command-line entry points for training and reproducing the paper tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nema.benchmark import run_benchmark, summarize_results
from nema.data import download_official_data, iter_pairs, pair_paths
from nema.models.nema import NEMAModel
from nema.training import train_nema


def _device(value: str) -> str:
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise argparse.ArgumentTypeError("CUDA was requested but is unavailable")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nema")
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser("download-data", help="download pinned official NGA pairs")
    download.add_argument("--root", default="data/official")
    download.add_argument("--force", action="store_true")

    train = sub.add_parser("train-nema", help="unsupervised amortized NEMA training")
    train.add_argument("--data-root", default="data/official")
    train.add_argument("--datasets", nargs="+", default=["AIDS", "MOLHIV", "MCF-7"])
    train.add_argument("--file-limit", type=int)
    train.add_argument("--pair-limit", type=int)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--accumulation", type=int, default=8)
    train.add_argument("--steps", type=int, default=8)
    train.add_argument("--hidden", type=int, default=32)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--device", type=_device, default="cpu")
    train.add_argument("--torch-threads", type=int, default=1)
    train.add_argument("--output", default="checkpoints/nema.pt")

    benchmark = sub.add_parser("benchmark", help="run MCES/similarity or retrieval evaluation")
    benchmark.add_argument("--data-root", default="data/official")
    benchmark.add_argument("--dataset", choices=["AIDS", "MOLHIV", "MCF-7"], required=True)
    benchmark.add_argument("--method", choices=["nema", "nga", "nga-paper"], required=True)
    benchmark.add_argument("--task", choices=["mces", "retrieval"], default="mces")
    benchmark.add_argument("--split", default="test")
    benchmark.add_argument("--limit", type=int)
    benchmark.add_argument("--workers", type=int, default=1)
    benchmark.add_argument("--device", type=_device, default="cpu")
    benchmark.add_argument("--checkpoint")
    benchmark.add_argument("--output")
    benchmark.add_argument("--seed", type=int, default=0)
    benchmark.add_argument("--continuous-restarts", type=int, default=4)
    benchmark.add_argument("--discrete-restarts", type=int, default=8)
    benchmark.add_argument("--refinement-passes", type=int, default=20)
    benchmark.add_argument("--anneal-steps", type=int, default=1000)
    benchmark.add_argument("--lns-steps", type=int, default=100)
    benchmark.add_argument("--certificate-seconds", type=float, default=0.0)
    benchmark.add_argument("--nema-trajectory", choices=["learned", "unit"], default="learned")
    benchmark.add_argument("--initializer-mode", choices=["learned", "fixed"], default="learned")
    benchmark.add_argument("--schedule-mode", choices=["learned", "fixed"], default="learned")
    benchmark.add_argument("--no-unit-fallback", action="store_true")
    benchmark.add_argument("--no-line-search", action="store_true")
    benchmark.add_argument("--no-stationary-safeguard", action="store_true")
    benchmark.add_argument("--sinkhorn-tolerance", type=float, default=1e-5)
    benchmark.add_argument("--sinkhorn-max-iterations", type=int, default=250)
    benchmark.add_argument("--acceptance-tolerance", type=float, default=0.0)
    benchmark.add_argument("--epochs", type=int, default=200)
    benchmark.add_argument("--lr", type=float, default=1e-3)
    benchmark.add_argument("--samples", type=int, default=10)
    benchmark.add_argument("--time-budget", type=float, default=60.0)
    benchmark.add_argument("--nga-runs", type=int, default=1)
    benchmark.add_argument("--nga-refine", action="store_true")
    benchmark.add_argument("--nga-variant", choices=["acg", "paper"], default="acg")

    summarize = sub.add_parser("summarize", help="summarize a JSONL benchmark")
    summarize.add_argument("path")
    summarize.add_argument("--retrieval", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "download-data":
        print(download_official_data(args.root, force=args.force))
        return
    if args.command == "train-nema":
        torch.set_num_threads(args.torch_threads)
        all_pairs = []
        for dataset in args.datasets:
            paths = pair_paths(args.data_root, dataset, split="train", limit=args.file_limit)
            all_pairs.extend(iter_pairs(paths))
        if args.pair_limit is not None:
            all_pairs = all_pairs[: args.pair_limit]
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        # Seed before module construction: the metric MLP initialization is a
        # material part of a training-seed replicate.
        torch.manual_seed(args.seed)
        model_config = {"steps": args.steps, "hidden_dim": args.hidden}
        model = NEMAModel(steps=args.steps, hidden_dim=args.hidden)
        initial_history = []
        optimizer_state = None
        if output.exists():
            payload = torch.load(output, map_location="cpu", weights_only=True)
            if payload.get("config") != model_config:
                raise ValueError("existing NEMA checkpoint has incompatible model configuration")
            if payload.get("datasets") != args.datasets:
                raise ValueError("existing NEMA checkpoint has incompatible datasets")
            if payload.get("training_seed", args.seed) != args.seed:
                raise ValueError("existing NEMA checkpoint has incompatible training seed")
            model.load_state_dict(payload["model"])
            initial_history = payload.get("history", [])
            optimizer_state = payload.get("optimizer")
            print(f"resuming NEMA from epoch {len(initial_history)}", flush=True)

        def save_checkpoint(history, optimizer) -> None:
            best_soft_objective = max(
                (record["soft_objective"] for record in history),
                default=float("-inf"),
            )
            checkpoint = {
                "algorithm_version": "nema-safeguarded-lifted-kl-v1",
                "model": model.state_dict(),
                "config": model_config,
                "history": history,
                "optimizer": optimizer.state_dict(),
                "datasets": args.datasets,
                "pairs": len(all_pairs),
                "completed_epochs": len(history),
                "best_soft_objective": best_soft_objective,
                "training_seed": args.seed,
                "selection_metric": "maximum label-free training soft_objective",
            }
            temporary = output.with_name(output.name + ".tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(output)
            if history[-1]["soft_objective"] >= best_soft_objective:
                best_output = output.with_name(f"{output.stem}.best{output.suffix}")
                best_temporary = best_output.with_name(best_output.name + ".tmp")
                torch.save(checkpoint, best_temporary)
                best_temporary.replace(best_output)

        history = train_nema(
            model,
            all_pairs,
            epochs=args.epochs,
            learning_rate=args.lr,
            accumulation=args.accumulation,
            seed=args.seed,
            device=args.device,
            initial_history=initial_history,
            optimizer_state=optimizer_state,
            on_epoch=save_checkpoint,
        )
        print(json.dumps({"checkpoint": str(output), "pairs": len(all_pairs), "history": history}))
        return
    if args.command == "benchmark":
        output = args.output or f"results/{args.method}_{args.task}_{args.dataset}.jsonl"
        summary = run_benchmark(
            data_root=args.data_root,
            dataset=args.dataset,
            method=args.method,
            output=output,
            retrieval=args.task == "retrieval",
            split=args.split,
            limit=args.limit,
            workers=args.workers,
            device=args.device,
            checkpoint=args.checkpoint,
            seed=args.seed,
            continuous_restarts=args.continuous_restarts,
            discrete_restarts=args.discrete_restarts,
            refinement_passes=args.refinement_passes,
            anneal_steps=args.anneal_steps,
            lns_steps=args.lns_steps,
            certificate_seconds=args.certificate_seconds,
            epochs=args.epochs,
            learning_rate=args.lr,
            samples=args.samples,
            time_budget=args.time_budget,
            nga_runs=args.nga_runs,
            nga_refine=args.nga_refine,
            nga_variant=args.nga_variant,
            nema_trajectory=args.nema_trajectory,
            initializer_mode=args.initializer_mode,
            schedule_mode=args.schedule_mode,
            unit_fallback=not args.no_unit_fallback,
            line_search=not args.no_line_search,
            stationary_safeguard=not args.no_stationary_safeguard,
            sinkhorn_tolerance=args.sinkhorn_tolerance,
            sinkhorn_max_iterations=args.sinkhorn_max_iterations,
            acceptance_tolerance=args.acceptance_tolerance,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return
    if args.command == "summarize":
        print(json.dumps(summarize_results(args.path, args.retrieval), indent=2))


if __name__ == "__main__":
    main()
