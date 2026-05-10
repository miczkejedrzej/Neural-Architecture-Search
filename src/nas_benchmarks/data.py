"""NAS-Bench-201 data loading and normalization."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def load_nb201_api(dataset: str, data_path: str | None) -> dict[str, Any]:
    path = Path(data_path) if data_path is not None else default_nb201_data_path()

    if not path.exists():
        raise FileNotFoundError(
            f"NAS-Bench-201 data missing: {path}. "
            "Pass --nb201-data or place nb201_all.pickle under NASLib/naslib/data/."
        )

    with path.open("rb") as handle:
        data = pickle.load(handle)

    if not isinstance(data, dict):
        raise TypeError(f"Expected NAS-Bench-201 pickle dict, got {type(data)!r}")

    first_arch = next(iter(data.values()))
    if dataset not in first_arch and not (
        dataset == "cifar10" and "cifar10-valid" in first_arch
    ):
        raise KeyError(f"Dataset {dataset!r} not found in {path}")

    normalize_nb201_data(data)
    return {"nb201_data": data}


def default_nb201_data_path() -> Path:
    for parent in [Path.cwd(), *Path(__file__).resolve().parents]:
        candidate = parent / "NASLib" / "naslib" / "data" / "nb201_all.pickle"
        if candidate.exists():
            return candidate
    return (
        Path(__file__).resolve().parents[2]
        / "NASLib"
        / "naslib"
        / "data"
        / "nb201_all.pickle"
    )


def normalize_nb201_data(data: dict[str, Any]) -> None:
    curve_keys = (
        "train_acc1es",
        "train_losses",
        "train_times",
        "eval_acc1es",
        "eval_times",
        "eval_losses",
    )
    for arch_data in data.values():
        for split_data in arch_data.values():
            for key in curve_keys:
                if key in split_data and not isinstance(split_data[key], list):
                    split_data[key] = [split_data[key]]
            if "cost_info" not in split_data:
                split_data["cost_info"] = {
                    "train_time": value_at_end(split_data.get("train_times", 0.0)),
                    "params": float(split_data.get("params", 0.0)),
                    "flops": float(split_data.get("flop", 0.0)),
                    "latency": float(split_data.get("latency", 0.0)),
                }


def value_at_end(value: Any) -> float:
    if isinstance(value, list):
        return float(value[-1])
    return float(value)

