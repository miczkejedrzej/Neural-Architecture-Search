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
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if name in {"random", "rl", "darts_proxy"}:
        return benchmark_tabular_optimizer(name, args, dataset_api)

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
        "time_limit_sec": args.time_limit_sec,
        "max_queries": args.max_queries,
        "predictor": args.predictor,
        "methods": methods,
        "optimizers": {},
    }

    for optimizer_name in methods:
        LOGGER.info("Running %s", optimizer_name)
        rows, final = benchmark_optimizer(optimizer_name, args, dataset_api)
        all_rows.extend(rows)
        summary["optimizers"][optimizer_name] = final
        LOGGER.info(
            "%s final: val_acc=%.4f test_acc=%.4f best_arch=%s",
            optimizer_name,
            final["val_acc"],
            final["test_acc"],
            final["best_arch"],
        )

    write_outputs(Path(args.out_dir), all_rows, summary, make_plots=not args.no_plots)
    return summary


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    summary = run(parse_args(argv))
    print(json.dumps(summary, indent=2))


def should_continue(start: float, completed_queries: int, args: argparse.Namespace) -> bool:
    if completed_queries >= args.max_queries:
        return False
    if completed_queries == 0:
        return True
    return time.perf_counter() - start < args.time_limit_sec


def validate_time_budget(args: argparse.Namespace) -> None:
    if args.time_limit_sec <= 0:
        raise ValueError("--time-limit-sec must be positive")
    if args.max_queries <= 0:
        raise ValueError("--max-queries must be positive")


def set_seed_if_needed(args: argparse.Namespace) -> None:
    if not args.no_seed:
        utils.set_seed(args.seed)
