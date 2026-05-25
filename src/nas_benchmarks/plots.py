"""Plot generation for NAS-Bench-201 benchmark results."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STYLE = {
    "random": {"color": "#525252", "marker": "x", "label": "Random Search"},
    "re": {"color": "#2563eb", "marker": "o", "label": "Regularized Evolution"},
    "bananas": {"color": "#dc2626", "marker": "s", "label": "BANANAS"},
    "rl": {"color": "#16a34a", "marker": "^", "label": "RL Controller"},
    "darts_proxy": {"color": "#9333ea", "marker": "D", "label": "DARTS Proxy"},
}

TIME_AXIS_KEY = "wall_time_sec"
ADJUSTED_TIME_AXIS_KEY = "adjusted_wall_time_sec"


def write_plots(
    out_dir: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    grouped = group_rows(average_rows(rows))

    paths = [
        plot_metric_trajectory(
            out_dir / "best_val_accuracy.png",
            grouped,
            metric="val_acc",
            ylabel="Best validation accuracy (%)",
            title="NAS-Bench-201 validation trajectory",
        ),
        plot_metric_trajectory(
            out_dir / "best_test_accuracy.png",
            grouped,
            metric="test_acc",
            ylabel="Best test accuracy (%)",
            title="NAS-Bench-201 test trajectory",
        ),
        plot_metric_trajectory(
            out_dir / "sampled_reward.png",
            grouped,
            metric="reward",
            ylabel="Sampled reward",
            title="Sampled architecture reward",
        ),
        plot_best_val_accuracy_by_time_with_range(
            out_dir / "best_val_accuracy_by_time.png",
            rows,
            ylabel="Best validation accuracy (%)",
            title="NAS-Bench-201 validation trajectory over time",
        ),
        plot_metric_trajectory_by_time(
            out_dir / "best_test_accuracy_by_time.png",
            grouped,
            metric="test_acc",
            ylabel="Best test accuracy (%)",
            title="NAS-Bench-201 test trajectory over time",
        ),
        plot_metric_trajectory_by_time(
            out_dir / "sampled_reward_by_time.png",
            grouped,
            metric="reward",
            ylabel="Sampled reward",
            title="Sampled architecture reward over time",
        ),
        plot_metric_trajectory_by_time(
            out_dir / "runtime_vs_val_accuracy.png",
            grouped,
            metric="val_acc",
            ylabel="Best validation accuracy (%)",
            title="Validation accuracy vs wall time",
        ),
        plot_final_bars(
            out_dir / "final_test_accuracy.png",
            summary,
            metric="test_acc",
            ylabel="Final test accuracy (%)",
            title="Final test accuracy by optimizer",
        ),
    ]
    return paths


def write_adjusted_time_plots(
    out_dir: Path,
    rows: list[dict[str, Any]],
    extra_time_sec: float,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    adjusted_rows = add_adjusted_wall_time(rows, extra_time_sec)
    grouped = group_rows(average_rows(adjusted_rows))
    cutoff_time_sec = compute_bananas_adjusted_time_cutoff(adjusted_rows)
    final_summary = build_final_test_accuracy_summary(adjusted_rows, cutoff_time_sec)

    paths = [
        plot_metric_trajectory_by_time(
            out_dir / "best_test_accuracy_by_time.png",
            grouped,
            metric="test_acc",
            ylabel="Best test accuracy (%)",
            title="NAS-Bench-201 test trajectory over time",
            time_key=ADJUSTED_TIME_AXIS_KEY,
            max_time_sec=cutoff_time_sec,
            dense_initial_hours_until=50,
            dense_initial_hours_step=10,
        ),
        plot_metric_trajectory_by_time(
            out_dir / "runtime_vs_val_accuracy.png",
            grouped,
            metric="val_acc",
            ylabel="Best validation accuracy (%)",
            title="Validation accuracy vs wall time",
            time_key=ADJUSTED_TIME_AXIS_KEY,
            max_time_sec=cutoff_time_sec,
            dense_initial_hours_until=50,
            dense_initial_hours_step=10,
        ),
        plot_final_bars(
            out_dir / "final_test_accuracy.png",
            final_summary,
            metric="test_acc",
            ylabel="Final test accuracy (%)",
            title="Final test accuracy by optimizer",
        ),
    ]
    return paths


def group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["optimizer"])].append(row)
    for optimizer_rows in grouped.values():
        optimizer_rows.sort(key=lambda row: int(row["epoch"]))
    return dict(grouped)


def group_rows_by_run(
    rows: list[dict[str, Any]],
) -> dict[str, dict[int, list[dict[str, Any]]]]:
    grouped: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        optimizer = str(row["optimizer"])
        run_id = int(row.get("run_id", 0))
        grouped[optimizer][run_id].append(row)
    for optimizer_runs in grouped.values():
        for run_rows in optimizer_runs.values():
            run_rows.sort(key=lambda row: float(row[TIME_AXIS_KEY]))
    return {opt: dict(runs) for opt, runs in grouped.items()}


def add_adjusted_wall_time(
    rows: list[dict[str, Any]],
    extra_time_sec: float,
) -> list[dict[str, Any]]:
    grouped = group_rows_by_optimizer_and_run(rows)
    adjusted_rows: list[dict[str, Any]] = []
    for optimizer_runs in grouped.values():
        for run_rows in optimizer_runs.values():
            repaired_rows = fill_missing_wall_times(run_rows)
            for evaluation_index, row in enumerate(repaired_rows, start=1):
                adjusted_row = dict(row)
                adjusted_row[ADJUSTED_TIME_AXIS_KEY] = (
                    float(row[TIME_AXIS_KEY]) + evaluation_index * extra_time_sec
                )
                adjusted_rows.append(adjusted_row)
    return adjusted_rows


def group_rows_by_optimizer_and_run(
    rows: list[dict[str, Any]],
) -> dict[str, dict[int, list[dict[str, Any]]]]:
    grouped: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        optimizer = str(row["optimizer"])
        run_id = int(row.get("run_id", 0))
        grouped[optimizer][run_id].append(row)
    for optimizer_runs in grouped.values():
        for run_rows in optimizer_runs.values():
            run_rows.sort(key=lambda row: int(row["epoch"]))
    return {optimizer: dict(runs) for optimizer, runs in grouped.items()}


def fill_missing_wall_times(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    repaired_rows = [dict(row) for row in rows]
    times = [parse_float(row.get(TIME_AXIS_KEY)) for row in repaired_rows]
    epochs = [int(row["epoch"]) for row in repaired_rows]

    for index, value in enumerate(times):
        if value is not None:
            repaired_rows[index][TIME_AXIS_KEY] = value
            continue

        prev_index = index - 1
        while prev_index >= 0 and times[prev_index] is None:
            prev_index -= 1

        next_index = index + 1
        while next_index < len(times) and times[next_index] is None:
            next_index += 1

        if prev_index >= 0 and next_index < len(times):
            prev_time = float(times[prev_index])
            next_time = float(times[next_index])
            prev_epoch = epochs[prev_index]
            next_epoch = epochs[next_index]
            epoch = epochs[index]
            span = next_epoch - prev_epoch
            if span <= 0:
                filled = prev_time
            else:
                filled = prev_time + (next_time - prev_time) * ((epoch - prev_epoch) / span)
        elif prev_index >= 0:
            filled = float(times[prev_index])
        elif next_index < len(times):
            filled = float(times[next_index])
        else:
            raise ValueError("Cannot infer missing wall_time_sec from empty trajectory run.")

        times[index] = filled
        repaired_rows[index][TIME_AXIS_KEY] = filled

    return repaired_rows


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        if match is None:
            raise
        return float(match.group(0))


def compute_shared_adjusted_time_cutoff(
    grouped: dict[str, list[dict[str, Any]]],
) -> float:
    final_times = [
        float(rows[-1][ADJUSTED_TIME_AXIS_KEY])
        for rows in grouped.values()
        if rows
    ]
    if not final_times:
        raise ValueError("Cannot compute cutoff for empty trajectory rows.")
    return min(final_times)


def compute_bananas_adjusted_time_cutoff(
    rows: list[dict[str, Any]],
) -> float:
    grouped = group_rows_by_optimizer_and_run(rows)
    bananas_runs = grouped.get("bananas")
    if not bananas_runs:
        raise ValueError("Cannot compute adjusted cutoff without BANANAS trajectory rows.")

    shortest_run = min(
        bananas_runs.values(),
        key=lambda run_rows: (len(run_rows), int(run_rows[-1]["epoch"])),
    )
    if not shortest_run:
        raise ValueError("Cannot compute adjusted cutoff from empty BANANAS run.")
    return float(shortest_run[-1][ADJUSTED_TIME_AXIS_KEY])


def build_final_test_accuracy_summary(
    rows: list[dict[str, Any]],
    cutoff_time_sec: float,
) -> dict[str, Any]:
    optimizers: dict[str, dict[str, float | str]] = {}
    grouped = group_rows_by_optimizer_and_run(rows)
    for optimizer, runs in grouped.items():
        run_values: list[float] = []
        for run_rows in runs.values():
            eligible_rows = [
                row for row in run_rows if float(row[ADJUSTED_TIME_AXIS_KEY]) <= cutoff_time_sec
            ]
            if not eligible_rows:
                continue
            run_values.append(float(eligible_rows[-1]["test_acc"]))
        if not run_values:
            continue
        optimizers[optimizer] = {
            "optimizer": optimizer,
            "test_acc": float(sum(run_values)) / float(len(run_values)),
        }
    return {"optimizers": optimizers}


def filter_rows_by_max_time(
    rows: list[dict[str, Any]],
    time_key: str,
    max_time_sec: float | None,
) -> list[dict[str, Any]]:
    if max_time_sec is None:
        return rows
    return [row for row in rows if float(row[time_key]) <= max_time_sec]


def average_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    run_ids = {row.get("run_id") for row in rows if "run_id" in row}
    if len(run_ids) <= 1:
        return rows

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
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["optimizer"]), int(row["epoch"]))].append(row)

    averaged: list[dict[str, Any]] = []
    for (optimizer, epoch), group in grouped.items():
        base = dict(group[0])
        base["optimizer"] = optimizer
        base["epoch"] = epoch
        for field in numeric_fields:
            values = [
                parsed
                for row in group
                if (parsed := parse_float(row.get(field))) is not None
            ]
            if values:
                base[field] = float(sum(values)) / float(len(values))
            else:
                base[field] = None
        averaged.append(base)
    averaged.sort(key=lambda row: (str(row["optimizer"]), int(row["epoch"])))
    return averaged


def plot_best_val_accuracy_by_time_with_range(
    path: Path,
    rows: list[dict[str, Any]],
    ylabel: str,
    title: str,
) -> Path:
    run_ids = {row.get("run_id") for row in rows if "run_id" in row}
    if len(run_ids) <= 1:
        grouped = group_rows(average_rows(rows))
        return plot_metric_trajectory_by_time(
            path,
            grouped,
            metric="val_acc",
            ylabel=ylabel,
            title=title,
        )

    grouped = group_rows_by_run(rows)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    for optimizer, runs in grouped.items():
        run_curves = build_best_so_far_curves(runs)
        if not run_curves:
            continue
        times, avg_vals, min_vals, max_vals = aggregate_run_curves(run_curves)
        style = STYLE.get(optimizer, {})
        color = style.get("color")
        label = style.get("label", optimizer)
        ax.fill_between(times, min_vals, max_vals, color=color, alpha=0.25)
        ax.plot(
            times,
            avg_vals,
            linewidth=2,
            marker=style.get("marker", "o"),
            markersize=4,
            color=color,
            alpha=0.7,
            label=label,
        )
        ax.plot(times, min_vals, linewidth=1.2, color=color, alpha=0.4)
        ax.plot(times, max_vals, linewidth=1.2, color=color, alpha=0.4)
    finish_axes(ax, xlabel="Wall time (seconds)", ylabel=ylabel, title=title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def build_best_so_far_curves(
    runs: dict[int, list[dict[str, Any]]]
) -> list[tuple[list[float], list[float]]]:
    curves: list[tuple[list[float], list[float]]] = []
    for run_rows in runs.values():
        times: list[float] = []
        bests: list[float] = []
        best_val: float | None = None
        for row in run_rows:
            val = row.get("val_acc")
            if val is None:
                continue
            best_val = max(best_val, float(val)) if best_val is not None else float(val)
            times.append(float(row[TIME_AXIS_KEY]))
            bests.append(best_val)
        if times:
            curves.append((times, bests))
    return curves


def aggregate_run_curves(
    curves: list[tuple[list[float], list[float]]]
) -> tuple[list[float], list[float], list[float], list[float]]:
    time_points = sorted({t for times, _ in curves for t in times})
    run_series: list[list[float | None]] = []
    for times, values in curves:
        idx = 0
        current: float | None = None
        series: list[float | None] = []
        for t in time_points:
            while idx < len(times) and times[idx] <= t:
                current = values[idx]
                idx += 1
            series.append(current)
        run_series.append(series)

    filtered_times: list[float] = []
    avg_vals: list[float] = []
    min_vals: list[float] = []
    max_vals: list[float] = []
    for i, t in enumerate(time_points):
        values = [series[i] for series in run_series if series[i] is not None]
        if not values:
            continue
        filtered_times.append(t)
        avg_vals.append(float(sum(values)) / float(len(values)))
        min_vals.append(float(min(values)))
        max_vals.append(float(max(values)))
    return filtered_times, avg_vals, min_vals, max_vals


def plot_metric_trajectory(
    path: Path,
    grouped: dict[str, list[dict[str, Any]]],
    metric: str,
    ylabel: str,
    title: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    for optimizer, rows in grouped.items():
        queries = [int(row["epoch"]) for row in rows]
        values = [float(row[metric]) for row in rows]
        style = STYLE.get(optimizer, {})
        ax.plot(
            queries,
            values,
            linewidth=2,
            marker=style.get("marker", "o"),
            markersize=4,
            color=style.get("color"),
            label=style.get("label", optimizer),
        )
    finish_axes(ax, xlabel="Architecture queries", ylabel=ylabel, title=title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_metric_trajectory_by_time(
    path: Path,
    grouped: dict[str, list[dict[str, Any]]],
    metric: str,
    ylabel: str,
    title: str,
    time_key: str = TIME_AXIS_KEY,
    max_time_sec: float | None = None,
    dense_initial_hours_until: float | None = None,
    dense_initial_hours_step: float | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    max_time_hours: float | None = None
    for optimizer, rows in grouped.items():
        filtered_rows = filter_rows_by_max_time(rows, time_key, max_time_sec)
        if not filtered_rows:
            continue
        wall_times = np.array([float(row[time_key]) for row in filtered_rows])

        if max_time_sec is not None:
            max_time_hours = float(np.round(max_time_sec / 3600, 0))
        # Cast seconds to hours on the plot.
        wall_times = np.round(wall_times / 3600, 0)
        values = [float(row[metric]) for row in filtered_rows]
        style = STYLE.get(optimizer, {})
        ax.plot(
            wall_times,
            values,
            linewidth=2,
            marker=style.get("marker", "o"),
            markersize=4,
            color=style.get("color"),
            label=style.get("label", optimizer),
        )
    finish_axes(ax, xlabel="Wall time (hours)", ylabel=ylabel, title=title)
    if max_time_hours is not None:
        ax.set_xlim(right=max_time_hours)
    if (
        dense_initial_hours_until is not None
        and dense_initial_hours_step is not None
        and dense_initial_hours_step > 0
    ):
        current_ticks = ax.get_xticks()
        dense_ticks = np.arange(0, dense_initial_hours_until + dense_initial_hours_step, dense_initial_hours_step)
        merged_ticks = sorted(
            {
                float(tick)
                for tick in current_ticks
                if tick >= dense_initial_hours_until
            }
            | {float(tick) for tick in dense_ticks}
        )
        ax.set_xticks(merged_ticks)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_final_bars(
    path: Path,
    summary: dict[str, Any],
    metric: str,
    ylabel: str,
    title: str,
) -> Path:
    optimizers = list(summary["optimizers"])
    values = [float(summary["optimizers"][optimizer][metric]) for optimizer in optimizers]
    labels = [STYLE.get(optimizer, {}).get("label", optimizer) for optimizer in optimizers]
    colors = [STYLE.get(optimizer, {}).get("color", "#525252") for optimizer in optimizers]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    bars = ax.bar(labels, values, color=colors, width=0.65)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    finish_axes(ax, xlabel="", ylabel=ylabel, title=title, legend=False)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def finish_axes(
    ax: plt.Axes,
    xlabel: str,
    ylabel: str,
    title: str,
    legend: bool = True,
) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    if legend:
        ax.legend(frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
