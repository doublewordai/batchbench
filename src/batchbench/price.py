"""CLI entrypoint for solving the batchbench pricing system from a CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ._lagrange_price import solve_wls_weighted_profit_nonneg as solve_price


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv",
        type=Path,
        help="CSV with columns: input_tokens, output_tokens, total_cost, weight",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.0,
        help="Profit margin buffer 'eps' applied to the weighted constraint (default: 0.0)",
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=0.0,
        help="Optional ridge regularization added to H to help ill-conditioned systems",
    )
    return parser.parse_args(argv)


def read_csv_rows(path: Path) -> list[list[float]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    rows: list[list[float]] = []
    data_started = False

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for idx, row in enumerate(reader, start=1):
            if not row or all(not cell.strip() for cell in row):
                continue
            if row[0].lstrip().startswith("#"):
                continue

            if len(row) < 4:
                raise ValueError(
                    f"Row {idx} in {path} has {len(row)} columns; at least 4 are required."
                )
            try:
                numeric = [float(cell) for cell in row]
            except ValueError:
                if not data_started:
                    # Skip header-like rows before the numeric payload.
                    continue
                raise ValueError(
                    f"Row {idx} in {path} contains non-numeric values: {row[:4]}"
                ) from None

            rows.append(numeric)
            data_started = True

    if not rows:
        raise ValueError(f"No numeric rows found in {path}")

    return rows


def build_matrices(rows: Iterable[Sequence[float]]):
    array = np.asarray(list(rows), dtype=float)
    if array.ndim != 2 or array.shape[1] < 4:
        raise ValueError(
            "Rows must form a 2D array with at least four columns: input, output, cost, weight"
        )

    A = array[:, :2]
    b = array[:, 2]
    w = array[:, 3]
    return A, b, w

def solve_price_average(input_prices, output_prices, weights):
    average_input = np.average(input_prices, weights=weights)
    average_output = np.average(output_prices, weights=weights)
    return np.array([average_input, average_output])

def compute_prices_from_csv(csv_path: Path, *, eps: float = 0.0, ridge: float = 0.0):
    rows = read_csv_rows(csv_path)
    A, b, w = build_matrices(rows)
    prices = solve_price(A, b, w=w, eps=eps)
    avg_prices = solve_price_average(np.array(rows)[:,-2], np.array(rows)[:,-1], w)
    print("\n")
    print("LS Pricing Results:")
    print(f"Input Price: {prices[0]:.3f}")
    print(f"Output Price: {prices[1]:.3f}")
    print("\n")

    print("Average Results:")
    print(f"Input Price: {avg_prices[0]:.3f}")
    print(f"Output Price: {avg_prices[1]:.3f}")
    print("\n")
    
    total_cost = np.dot(b, w)
    total_ls_revenue = np.dot(A, prices) @ w / 1_000_000
    total_avg_revenue = np.dot(A, avg_prices) @ w / 1_000_000

    ls_margin_total = (total_ls_revenue - total_cost) / total_cost * 100 if total_cost != 0 else 0.0
    avg_margin_total = (total_avg_revenue - total_cost) / total_cost * 100 if total_cost != 0 else 0.0

    print(f"Total Cost: {total_cost:<.3f}")
    print(f"Total LS Revenue: {total_ls_revenue:.3f}  (Margin: {ls_margin_total:.3f}%)")
    print(f"Total Avg Revenue: {total_avg_revenue:.3f}  (Margin: {avg_margin_total:.3f}%)")
    print("\n")

    # nicely formatted table of per-row income and cost for LS and average pricing
    incomes_ls = A @ prices / 1_000_000
    incomes_avg = A @ avg_prices / 1_000_000

    # Nicely
    row_fmt = "{:>4}  {:>5}  {:>5}  {:>3}  {:>3}  {:>3}  {:>3}  {:>3}"
    sep = "-" * 100

    # Table header and formatting
    headers = [
        "idx",
        "in_tokens",
        "out_tokens",
        "cost",
        "LS_rev",
        "LS_margin(%)",
        "Avg_rev",
        "Avg_margin(%)",
    ]

    # Column widths chosen for better alignment
    # idx: 4, in/out: 12, money columns: 14
    row_fmt = (
        "{idx:>4}  "
        "{in_tok:>12}  "
        "{out_tok:>12}  "
        "{cost:>14}  "
        "{ls_rev:>14}  "
        "{ls_marg:>14}  "
        "{avg_rev:>14}  "
        "{avg_marg:>14}"
    )
    sep = "-" * 112

    print()
    print(
        row_fmt.format(
            idx=headers[0],
            in_tok=headers[1],
            out_tok=headers[2],
            cost=headers[3],
            ls_rev=headers[4],
            ls_marg=headers[5],
            avg_rev=headers[6],
            avg_marg=headers[7],
        )
    )
    print(sep)

    for i, (a_row, cost, inc_ls, inc_avg) in enumerate(
        zip(A, b, incomes_ls, incomes_avg)
    ):
        ls_margin = (inc_ls - cost) / cost * 100 if cost != 0 else 0.0
        avg_margin = (inc_avg - cost) / cost * 100 if cost != 0 else 0.0
        print(
            row_fmt.format(
            idx=i,
            in_tok=f"{a_row[0]:.3f}",
            out_tok=f"{a_row[1]:.3f}",
            cost=f"{cost:.3f}",
            ls_rev=f"{inc_ls:.3f}",
            ls_marg=f"{ls_margin:.3f}",
            avg_rev=f"{inc_avg:.3f}",
            avg_marg=f"{avg_margin:.3f}",
            )
        )
    print("\n")
    return prices


def format_output(prices: np.ndarray) -> str:
    labels = ["input_tokens", "output_tokens"]
    formatted = []
    for idx, value in enumerate(prices):
        label = labels[idx] if idx < len(labels) else f"component_{idx}"
        formatted.append(f"{label}: {value:.8f}")
    return "\n".join(formatted)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        prices = compute_prices_from_csv(args.csv, eps=args.eps, ridge=args.ridge)
    except Exception as exc:  # pragma: no cover - surface friendly errors in CLI
        raise SystemExit(str(exc)) from exc

    print(format_output(prices))  # noqa: T201 - CLI intentionally prints to stdout
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
