import sys

from ._core import run_cli


def main(argv: list[str] | None = None) -> int:
    run_cli(list(sys.argv[1:] if argv is None else argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
