"""Random search baseline for tabular NAS-Bench-201 experiments."""

from __future__ import annotations

import argparse
from typing import Any

from nas_benchmarks.nb201 import query_nb201_arch, random_valid_arch


class RandomSearchOptimizer:
    def __init__(self, args: argparse.Namespace, dataset_api: dict[str, Any]):
        self.args = args
        self.dataset_api = dataset_api
        self.best_arch: tuple[int, ...] | None = None
        self.best_metrics: dict[str, float] | None = None

    def step(self) -> dict[str, Any]:
        sampled_arch = random_valid_arch()
        sampled_metrics = query_nb201_arch(
            sampled_arch, self.args.dataset, self.dataset_api
        )
        self._update_best(sampled_arch, sampled_metrics)
        return {
            "sampled_arch": sampled_arch,
            "best_arch": self.best_arch,
            "best_metrics": self.best_metrics,
            "reward": sampled_metrics["val_acc"] / 100.0,
            "loss": None,
            "entropy": None,
        }

    def _update_best(self, arch: tuple[int, ...], metrics: dict[str, float]) -> None:
        if self.best_metrics is None or metrics["val_acc"] > self.best_metrics["val_acc"]:
            self.best_arch = arch
            self.best_metrics = metrics

    def final_metadata(self) -> dict[str, Any]:
        return {}

