"""Plot generation for NAS-Bench-201 benchmark results."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STYLE = {
    "random": {"color": "#525252", "marker": "x", "label": "Random Search"},
    "re": {"color": "#2563eb", "marker": "o", "label": "Regularized Evolution"},
    "bananas": {"color": "#dc2626", "marker": "s", "label": "BANANAS"},
    "rl": {"color": "#16a34a", "marker": "^", "label": "RL Controller"},
    "darts_proxy": {"color": "#9333ea", "marker": "D", "label": "DARTS Proxy"},
}


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
            run_rows.sort(key=lambda row: float(row["wall_time_sec"]))
    return {opt: dict(runs) for opt, runs in grouped.items()}


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
            values = [row[field] for row in group if row.get(field) is not None]
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
            times.append(float(row["wall_time_sec"]))
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
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    for optimizer, rows in grouped.items():
        wall_times = [float(row["wall_time_sec"]) for row in rows]
        values = [float(row[metric]) for row in rows]
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
    finish_axes(ax, xlabel="Wall time (seconds)", ylabel=ylabel, title=title)
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
