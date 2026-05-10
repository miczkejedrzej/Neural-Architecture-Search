#!/usr/bin/env python3
"""Compatibility entry point for the NAS-Bench-201 optimizer comparison."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nas_benchmarks.runner import main


if __name__ == "__main__":
    main()
