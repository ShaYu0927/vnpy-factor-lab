"""Evaluate one batch factor with Alphalens Reloaded.

Example:
    python examples/alphalens_factor_analysis.py ^
        --factors output/factor_values.csv ^
        --factor momentum_20 ^
        --prices output/daily_prices.csv ^
        --show

The price CSV must contain ``datetime``, ``vt_symbol`` and ``close`` columns.
"""

from __future__ import annotations

import argparse

import pandas as pd

from vnpy.alpha.modeling import AlphalensEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an Alphalens factor analysis")
    parser.add_argument("--factors", required=True, help="Batch factor CSV or SQLite")
    parser.add_argument("--factor", required=True, help="Factor name to evaluate")
    parser.add_argument("--prices", required=True, help="Long-form daily price CSV")
    parser.add_argument("--periods", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument("--max-loss", type=float, default=0.35)
    parser.add_argument("--show", action="store_true", help="Render the full tear sheet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluator = AlphalensEvaluator(
        periods=tuple(args.periods),
        quantiles=args.quantiles,
        max_loss=args.max_loss,
    )

    if args.factors.lower().endswith((".db", ".sqlite", ".sqlite3")):
        factor = evaluator.factor_from_batch_sqlite(args.factors, args.factor)
    else:
        factor = evaluator.factor_from_batch_csv(args.factors, args.factor)

    prices = pd.read_csv(args.prices)
    report = evaluator.evaluate(factor, prices)

    print("\nMean IC")
    print(report.mean_information_coefficient.to_string())
    print("\nICIR")
    print(report.information_ratio.to_string())
    print("\nMean return by quantile")
    print(report.quantile_returns.to_string())

    if args.show:
        evaluator.create_full_tear_sheet(report)


if __name__ == "__main__":
    main()
