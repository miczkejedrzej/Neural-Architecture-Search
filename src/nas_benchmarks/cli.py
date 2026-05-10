"""Command-line interface for NAS-Bench-201 optimizer comparisons."""

from __future__ import annotations

import argparse

from nas_benchmarks.constants import SUPPORTED_METHODS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark NAS optimizers on NAS-Bench-201."
    )
    parser.add_argument("--methods", default="random,re,bananas,rl,darts_proxy")
    parser.add_argument(
        "--dataset",
        default="cifar10",
        choices=["cifar10", "cifar100", "ImageNet16-120"],
    )
    parser.add_argument(
        "--time-limit-sec",
        type=float,
        default=60.0,
        help="Wall-clock search budget per optimizer.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=1_000_000,
        help="Safety cap for architecture queries per optimizer.",
    )
    parser.add_argument("--epochs", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-seed",
        action="store_true",
        help="Do not seed random generators.",
    )
    parser.add_argument("--out-dir", default="runs/naslib_four_way")
    parser.add_argument(
        "--nb201-data",
        default=None,
        help="Optional path to NAS-Bench-201 pickle. Defaults to NASLib/naslib/data/nb201_all.pickle.",
    )
    parser.add_argument("--population-size", type=int, default=30)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--num-init", type=int, default=10)
    parser.add_argument("--bananas-k", type=int, default=10)
    parser.add_argument("--num-ensemble", type=int, default=3)
    parser.add_argument(
        "--predictor",
        default="bananas",
        help="NASLib BANANAS surrogate, e.g. bananas, rf, xgb, lgb, ngb, gp.",
    )
    parser.add_argument("--num-candidates", type=int, default=50)
    parser.add_argument("--num-arches-to-mutate", type=int, default=1)
    parser.add_argument("--max-mutations", type=int, default=1)
    parser.add_argument("--rl-hidden-size", type=int, default=32)
    parser.add_argument("--rl-lr", type=float, default=3.5e-4)
    parser.add_argument("--rl-baseline-momentum", type=float, default=0.9)
    parser.add_argument("--rl-entropy-weight", type=float, default=0.01)
    parser.add_argument("--darts-lr", type=float, default=0.01)
    parser.add_argument("--darts-temperature", type=float, default=1.0)
    parser.add_argument("--darts-warmup", type=int, default=10)
    parser.add_argument("--darts-surrogate-epochs", type=int, default=50)
    parser.add_argument(
        "--instantiate-model",
        action="store_true",
        help="Instantiate PyTorch modules for sampled architectures. Slower; not needed for tabular benchmarks.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip PNG plot generation.",
    )
    args = parser.parse_args(argv)
    if args.epochs is not None:
        args.max_queries = args.epochs
    return args


def parse_methods(methods: str) -> list[str]:
    selected = [method.strip() for method in methods.split(",") if method.strip()]
    unknown = sorted(set(selected) - SUPPORTED_METHODS)
    if unknown:
        supported = ", ".join(sorted(SUPPORTED_METHODS))
        raise ValueError(f"Unknown methods: {', '.join(unknown)}. Supported: {supported}")
    return selected
