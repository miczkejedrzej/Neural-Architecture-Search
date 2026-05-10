"""Benchmark result formatting and persistence."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from nas_benchmarks.constants import OUTPUT_COLUMNS
from nas_benchmarks.plots import write_plots

LOGGER = logging.getLogger("naslib-benchmark")


def build_row(
    optimizer: str,
    epoch: int,
    sampled_arch: tuple[int, ...],
    best_arch: tuple[int, ...],
    best_metrics: dict[str, float],
    reward: float,
    loss: float | None,
    entropy: float | None,
    wall_time_sec: float,
) -> dict[str, Any]:
    return {
        "optimizer": optimizer,
        "epoch": epoch,
        "sampled_arch": str(sampled_arch),
        "best_arch": str(best_arch),
        "train_acc": best_metrics["train_acc"],
        "val_acc": best_metrics["val_acc"],
        "test_acc": best_metrics["test_acc"],
        "train_time": best_metrics["train_time"],
        "reward": reward,
        "loss": loss,
        "entropy": entropy,
        "wall_time_sec": wall_time_sec,
    }


def write_outputs(
    out_dir: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    make_plots: bool = True,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "trajectory.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    if make_plots:
        plot_paths = write_plots(out_dir / "plots", rows, summary)
        summary["plots"] = [str(path.relative_to(out_dir)) for path in plot_paths]

    json_path = out_dir / "summary.json"
    with json_path.open("w") as handle:
        json.dump(summary, handle, indent=2)

    LOGGER.info("Wrote %s", csv_path)
    LOGGER.info("Wrote %s", json_path)
    if make_plots:
        LOGGER.info("Wrote plots under %s", out_dir / "plots")
