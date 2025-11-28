"""Compatibility shim for the weighted least squares solver."""

from __future__ import annotations

import sys
from pathlib import Path

if "batchbench" not in sys.modules:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

from batchbench._lagrange_price import solve_wls_weighted_profit

__all__ = ["solve_wls_weighted_profit"]
