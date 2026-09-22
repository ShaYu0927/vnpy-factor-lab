"""Run from the project directory: python -m examples.alpha_formula_pipeline.

Replace the demonstration frame with your loaded Polars market-data frame.
To calculate immediately after importing real data, add this top-level entry
to config/runtime.json (the existing parquet_import settings still apply):

"alphas": [
    {"name": "momentum_5", "formula": "close / Ref(close, 5) - 1"},
    {"name": "momentum_20", "formula": "close / Ref(close, 20) - 1"},
    {"name": "volatility", "formula": "Std(close / Ref(close, 1) - 1, 20)"}
]

The import branch loads the entire snapshot into memory and writes the factor
table to <snapshot root>/factors.parquet. Without alphas, it only imports data.
Replay also accepts alphas; Rank requires explicit parquet.symbols membership.

Supported aliases: Ref, Delta, Mean, Sum, Std, Min, Max, Corr, Rank, Abs,
Sign, Log, Pow. Canonical operator names (ts_mean, cs_rank, etc.) also work.
Arithmetic: + - * / **, unary +/- and parentheses. Ref uses past bars only.
Windows count bars per symbol, not calendar days. Full valid rolling windows
are required; warm-up rows are null. Std uses ddof=0. Rank is the average
ordinal rank within each datetime. Non-finite final results become null.

The engine snapshots formulas when constructed. Rebuild it after editing an
Alpha. Alpha.expression exposes the parsed tree for inspection; formula is
the source of truth. Old AlphaDefinition inputs retain their legacy semantics.
"""

from datetime import datetime, timedelta

import polars as pl

from vnpy.alpha import Alpha, AlphaEngine
from vnpy.alpha.expression import ExpressionParser, PolarsCompiler, PolarsExecutor


def main() -> None:
    frame = pl.DataFrame([
        {"datetime": datetime(2026, 1, 1) + timedelta(days=i),
         "vt_symbol": symbol, "close": base + i + (i % 3) * 0.2}
        for symbol, base in [("DEMO_A", 10), ("DEMO_B", 30)]
        for i in range(30)
    ])
    alphas = [
        Alpha("momentum_5", "close / Ref(close, 5) - 1"),
        Alpha("momentum_20", "close / Ref(close, 20) - 1"),
        Alpha("volatility", "Std(close / Ref(close, 1) - 1, 20)"),
    ]

    # Normal use: the engine performs parsing, analysis, compilation and execution.
    engine = AlphaEngine(alphas)
    print(engine.calculate(frame).tail(6).to_dicts())
    print("Parsed tree:", alphas[0].expression)
    print("Required history:", engine.min_bars)

    # Debug each stage independently when needed.
    tree = ExpressionParser().parse(alphas[0].formula)
    compiled = PolarsCompiler().compile(tree, available_fields=set(frame.columns))
    print("Analysis:", compiled.analysis)
    print("Polars stages:", compiled.stages)
    result = PolarsExecutor(frame).run(compiled, name=alphas[0].name)
    print(result.tail(2).to_dicts())


if __name__ == "__main__":
    main()
