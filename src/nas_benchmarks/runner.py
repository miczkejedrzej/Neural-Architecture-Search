"""Benchmark runner for NAS-Bench-201 optimizer comparisons."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

from naslib import utils
from naslib.optimizers.discrete.bananas.optimizer import Bananas
from naslib.optimizers.discrete.re.optimizer import RegularizedEvolution

from nas_benchmarks.cli import parse_args, parse_methods
from nas_benchmarks.config import build_config
from nas_benchmarks.data import load_nb201_api
from nas_benchmarks.nb201 import make_search_space, query_nb201_arch
from nas_benchmarks.optimizers.darts_proxy import TabularDARTSProxyOptimizer
from nas_benchmarks.optimizers.random_search import RandomSearchOptimizer
from nas_benchmarks.optimizers.rl import RLControllerOptimizer
from nas_benchmarks.output import build_row, write_outputs

LOGGER = logging.getLogger("naslib-benchmark")


def benchmark_tabular_optimizer(
    name: str,
    args: argparse.Namespace,
    dataset_api: dict[str, Any],
    run_id: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    set_seed_if_needed(args)
    if name == "random":
        optimizer = RandomSearchOptimizer(args, dataset_api)
    elif name == "rl":
        optimizer = RLControllerOptimizer(args, dataset_api)
    else:
        optimizer = TabularDARTSProxyOptimizer(args, dataset_api)

    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    query_index = 0
    while should_continue(start, query_index, args):
        step = (
            optimizer.step(query_index)
            if name == "darts_proxy"
            else optimizer.step()
        )
        query_index += 1
        rows.append(
            build_row(
                run_id=run_id,
                optimizer=name,
                epoch=query_index,
                sampled_arch=step["sampled_arch"],
                best_arch=step["best_arch"],
                best_metrics=step["best_metrics"],
                reward=step["reward"],
                loss=step["loss"],
                entropy=step["entropy"],
                wall_time_sec=time.perf_counter() - start,
            )
        )

    final = rows[-1].copy()
    final["queries"] = len(rows)
    final["time_limit_sec"] = args.time_limit_sec
    final.update(optimizer.final_metadata())
    return rows, final


def benchmark_optimizer(
    name: str,
    args: argparse.Namespace,
    dataset_api: dict[str, Any],
    run_id: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if name in {"random", "rl", "darts_proxy"}:
        return benchmark_tabular_optimizer(name, args, dataset_api, run_id)

    config = build_config(args, name)
    set_seed_if_needed(args)

    if name == "re":
        optimizer = RegularizedEvolution(config)
    elif name == "bananas":
        optimizer = Bananas(config)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")
    optimizer.adapt_search_space(
        make_search_space(args.instantiate_model), dataset_api=dataset_api
    )
    optimizer.before_training()

    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    query_index = 0
    while should_continue(start, query_index, args):
        optimizer.new_epoch(query_index)
        train_acc, val_acc, test_acc, train_time = optimizer.train_statistics(
            report_incumbent=True
        )
        query_index += 1
        best_arch = optimizer.get_final_architecture()
        if name == "re":
            sampled_arch = tuple(int(x) for x in optimizer.population[-1].arch.get_hash())
        else:
            sampled_arch = tuple(int(x) for x in optimizer.train_data[-1].arch.get_hash())
        sampled_metrics = query_nb201_arch(sampled_arch, args.dataset, dataset_api)
        best_arch_hash = tuple(int(x) for x in best_arch.get_hash())
        best_metrics = {
            "train_acc": float(train_acc),
            "val_acc": float(val_acc),
            "test_acc": float(test_acc),
            "train_time": float(train_time),
        }
        rows.append(
            build_row(
                run_id=run_id,
                optimizer=name,
                epoch=query_index,
                sampled_arch=sampled_arch,
                best_arch=best_arch_hash,
                best_metrics=best_metrics,
                reward=sampled_metrics["val_acc"] / 100.0,
                loss=None,
                entropy=None,
                wall_time_sec=time.perf_counter() - start,
            )
        )

    optimizer.after_training()
    final = rows[-1].copy()
    final["queries"] = len(rows)
    final["time_limit_sec"] = args.time_limit_sec
    return rows, final


def run(args: argparse.Namespace) -> dict[str, Any]:
    methods = parse_methods(args.methods)

    validate_time_budget(args)

    dataset_api = load_nb201_api(args.dataset, args.nb201_data)
    all_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "dataset": args.dataset,
        "seed": None if args.no_seed else args.seed,
        "seeded": not args.no_seed,
        "n_runs": args.n_runs,
        "min_time_sec": args.min_time_sec,
        "time_limit_sec": args.time_limit_sec,
        "max_queries": args.max_queries,
        "predictor": args.predictor,
        "methods": methods,
        "optimizers": {},
        "runs": [],
    }

    optimizer_runs: dict[str, list[dict[str, Any]]] = {name: [] for name in methods}

    for run_id in range(args.n_runs):
        run_args = make_run_args(args, run_id)
        run_summary = {
            "run_id": run_id,
            "seed": None if run_args.no_seed else run_args.seed,
            "optimizers": {},
        }
        for optimizer_name in methods:
            LOGGER.info("Run %s/%s: %s", run_id + 1, args.n_runs, optimizer_name)
            rows, final = benchmark_optimizer(optimizer_name, run_args, dataset_api, run_id)
            all_rows.extend(rows)
            optimizer_runs[optimizer_name].append(final)
            run_summary["optimizers"][optimizer_name] = final
            LOGGER.info(
                "%s final: val_acc=%.4f test_acc=%.4f best_arch=%s",
                optimizer_name,
                final["val_acc"],
                final["test_acc"],
                final["best_arch"],
            )
        summary["runs"].append(run_summary)

    summary["optimizers"] = average_final_metrics(optimizer_runs)

    write_outputs(Path(args.out_dir), all_rows, summary, make_plots=not args.no_plots)
    return summary


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    summary = run(parse_args(argv))
    print(json.dumps(summary, indent=2))


def should_continue(start: float, completed_queries: int, args: argparse.Namespace) -> bool:
    elapsed = time.perf_counter() - start
    if elapsed < args.min_time_sec:
        return True
    if completed_queries >= args.max_queries:
        return False
    if completed_queries == 0:
        return True
    return elapsed < args.time_limit_sec


def validate_time_budget(args: argparse.Namespace) -> None:
    if args.min_time_sec < 0:
        raise ValueError("--min-time-sec must be non-negative")
    if args.time_limit_sec <= 0:
        raise ValueError("--time-limit-sec must be positive")
    if args.time_limit_sec < args.min_time_sec:
        raise ValueError("--time-limit-sec must be >= --min-time-sec")
    if args.max_queries <= 0:
        raise ValueError("--max-queries must be positive")
    if args.n_runs <= 0:
        raise ValueError("--n-runs must be positive")


def set_seed_if_needed(args: argparse.Namespace) -> None:
    if not args.no_seed:
        utils.set_seed(args.seed)


def make_run_args(args: argparse.Namespace, run_id: int) -> argparse.Namespace:
    if args.no_seed:
        return args
    run_seed = args.seed + run_id
    return argparse.Namespace(**{**vars(args), "seed": run_seed})


def average_final_metrics(
    optimizer_runs: dict[str, list[dict[str, Any]]]
) -> dict[str, dict[str, Any]]:
    averaged: dict[str, dict[str, Any]] = {}
    numeric_fields = {
        "train_acc",
        "val_acc",
        "test_acc",
        "train_time",
        "reward",
        "loss",
        "entropy",
        "wall_time_sec",
    }
    for optimizer, finals in optimizer_runs.items():
        if not finals:
            continue
        base = dict(finals[0])
        for field in numeric_fields:
            values = [final[field] for final in finals if final.get(field) is not None]
            if values:
                base[field] = float(sum(values)) / float(len(values))
            else:
                base[field] = None
        base["queries"] = finals[0].get("queries")
        base["time_limit_sec"] = finals[0].get("time_limit_sec")
        base["runs"] = finals
        averaged[optimizer] = base
    return averaged
