"""Wrapper that launches the bundled Rust-powered online benchmark."""

from __future__ import annotations

import os
import subprocess
import sys

from importlib import resources


BINARY_NAME = "batchbench"


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]

    resource = resources.files(__package__).joinpath("bin", BINARY_NAME)
    if not resource.is_file():  # pragma: no cover
        raise SystemExit(
            "Bundled online benchmarking binary not found. "
            "Please reinstall batchbench or report an issue."
        )

    env = os.environ.copy()

    with resources.as_file(resource) as binary_path:
        command = [str(binary_path), *args]
        try:
            completed = subprocess.run(command, check=False, env=env)
        except FileNotFoundError as exc:  # pragma: no cover
            raise SystemExit(f"Failed to execute bundled binary: {exc}") from exc

    return int(completed.returncode)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
