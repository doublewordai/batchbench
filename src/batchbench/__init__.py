"""batchbench brings offline and online benchmarking utilities under one roof."""

from __future__ import annotations

from importlib import resources

try:  # pragma: no cover
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version  # type: ignore


def package_version() -> str:
    """Return the installed package version or a placeholder when run from source."""
    try:
        return version("batchbench")
    except PackageNotFoundError:
        return "0.0.0"


__all__ = ["package_version", "resources"]
__version__ = package_version()
