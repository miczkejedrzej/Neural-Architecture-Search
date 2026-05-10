"""Shared constants for NAS-Bench-201 benchmark runs."""

NUM_NB201_EDGES = 6
NUM_NB201_OPS = 5

SUPPORTED_METHODS = {"re", "bananas", "rl", "darts_proxy"}

OUTPUT_COLUMNS = [
    "optimizer",
    "epoch",
    "sampled_arch",
    "best_arch",
    "train_acc",
    "val_acc",
    "test_acc",
    "train_time",
    "reward",
    "loss",
    "entropy",
    "wall_time_sec",
]

