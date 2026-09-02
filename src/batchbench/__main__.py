import sys


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "export-plans":
        from .export_plans import main as export_plans_main

        return export_plans_main(args[1:])

    from ._core import run_cli

    run_cli(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
