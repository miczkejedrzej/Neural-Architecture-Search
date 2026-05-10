"""NAS-Bench-201 architecture helpers."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.nasbench201.graph import NasBench201SearchSpace

from nas_benchmarks.constants import NUM_NB201_EDGES, NUM_NB201_OPS


def make_search_space(instantiate_model: bool) -> NasBench201SearchSpace:
    search_space = NasBench201SearchSpace()
    search_space.instantiate_model = instantiate_model
    return search_space


def is_valid_nb201_arch(arch: tuple[int, ...]) -> bool:
    return not (
        (arch[0] == arch[1] == arch[2] == 1)
        or (arch[2] == arch[4] == arch[5] == 1)
    )


def random_valid_arch() -> tuple[int, ...]:
    while True:
        arch = tuple(
            int(x) for x in torch.randint(NUM_NB201_OPS, (NUM_NB201_EDGES,)).tolist()
        )
        if is_valid_nb201_arch(arch):
            return arch


def arch_to_tensor(arch: tuple[int, ...]) -> torch.Tensor:
    tokens = torch.tensor(arch, dtype=torch.long)
    return F.one_hot(tokens, num_classes=NUM_NB201_OPS).float().flatten()


def query_nb201_arch(
    arch: tuple[int, ...],
    dataset: str,
    dataset_api: dict[str, Any],
) -> dict[str, float]:
    search_space = make_search_space(instantiate_model=False)
    search_space.set_spec(arch)
    return {
        "train_acc": float(
            search_space.query(Metric.TRAIN_ACCURACY, dataset, dataset_api=dataset_api)
        ),
        "val_acc": float(
            search_space.query(Metric.VAL_ACCURACY, dataset, dataset_api=dataset_api)
        ),
        "test_acc": float(
            search_space.query(Metric.TEST_ACCURACY, dataset, dataset_api=dataset_api)
        ),
        "train_time": float(
            search_space.query(Metric.TRAIN_TIME, dataset, dataset_api=dataset_api)
        ),
    }

