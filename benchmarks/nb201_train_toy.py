#!/usr/bin/env python3
"""Compatibility entry point for NAS-Bench-201 CIFAR-100 training."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nas_benchmarks.toy_train import main


if __name__ == "__main__":
    main()
