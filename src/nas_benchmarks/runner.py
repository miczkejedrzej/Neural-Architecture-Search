"""Benchmark runner for four-way NAS-Bench-201 comparisons."""

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
from nas_benchmarks.optimizers.rl import RLControllerOptimizer
from nas_benchmarks.output import build_row, write_outputs

LOGGER = logging.getLogger("naslib-benchmark")


def benchmark_tabular_optimizer(
    name: str,
    args: argparse.Namespace,
    dataset_api: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    utils.set_seed(args.seed)
    optimizer = (
        RLControllerOptimizer(args, dataset_api)
        if name == "rl"
        else TabularDARTSProxyOptimizer(args, dataset_api)
    )

    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    for epoch in range(args.epochs):
        step = optimizer.step(epoch) if name == "darts_proxy" else optimizer.step()
        rows.append(
            build_row(
                optimizer=name,
                epoch=epoch + 1,
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
    final["epochs"] = args.epochs
    final.update(optimizer.final_metadata())
    return rows, final


def benchmark_optimizer(
    name: str,
    args: argparse.Namespace,
    dataset_api: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if name in {"rl", "darts_proxy"}:
        return benchmark_tabular_optimizer(name, args, dataset_api)

    config = build_config(args, name)
    utils.set_seed(args.seed)

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
    for epoch in range(args.epochs):
        optimizer.new_epoch(epoch)
        train_acc, val_acc, test_acc, train_time = optimizer.train_statistics(
            report_incumbent=True
        )
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
                epoch=epoch + 1,
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
    final["epochs"] = args.epochs
    return rows, final


def run(args: argparse.Namespace) -> dict[str, Any]:
    methods = parse_methods(args.methods)

    if args.epochs < max(args.population_size, args.num_init):
        LOGGER.warning(
            "epochs=%s is below population_size=%s or num_init=%s; comparison will mostly show initialization.",
            args.epochs,
            args.population_size,
            args.num_init,
        )

    dataset_api = load_nb201_api(args.dataset, args.nb201_data)
    all_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "dataset": args.dataset,
        "seed": args.seed,
        "epochs": args.epochs,
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

    write_outputs(Path(args.out_dir), all_rows, summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    summary = run(parse_args(argv))
    print(json.dumps(summary, indent=2))

