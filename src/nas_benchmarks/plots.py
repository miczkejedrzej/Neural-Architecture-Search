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
    grouped = group_rows(rows)

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
        plot_runtime_vs_metric(
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


def plot_metric_trajectory(
    path: Path,
    grouped: dict[str, list[dict[str, Any]]],
    metric: str,
    ylabel: str,
    title: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    for optimizer, rows in grouped.items():
        epochs = [int(row["epoch"]) for row in rows]
        values = [float(row[metric]) for row in rows]
        style = STYLE.get(optimizer, {})
        ax.plot(
            epochs,
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


def plot_runtime_vs_metric(
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
