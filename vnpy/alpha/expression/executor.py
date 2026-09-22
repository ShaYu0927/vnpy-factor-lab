"""Execute compiled formulas on keyed market data."""

from __future__ import annotations

import polars as pl

from .polars_compiler import CompiledExpression


class PolarsExecutor:
    def __init__(self, frame: pl.DataFrame) -> None:
        keys = {"datetime", "vt_symbol"}
        if missing := keys - set(frame.columns):
            raise ValueError(f"alpha input is missing columns: {', '.join(sorted(missing))}")
        if any(frame[key].null_count() for key in keys):
            raise ValueError("alpha input keys must not be null")
        if frame.select(pl.struct("datetime", "vt_symbol").is_duplicated().any()).item():
            raise ValueError("alpha input contains duplicate datetime/symbol rows")
        self.frame = frame.sort(["vt_symbol", "datetime"])

    def run(self, compiled: CompiledExpression, name: str = "data") -> pl.DataFrame:
        if not name.isidentifier() or name in {"datetime", "vt_symbol"}:
            raise ValueError("alpha name must be an identifier distinct from the index columns")
        if missing := compiled.analysis.fields - set(self.frame.columns):
            raise ValueError(f"alpha input is missing columns: {', '.join(sorted(missing))}")
        # Discard unrelated columns, including any that share temporary names.
        columns = ["datetime", "vt_symbol"] + sorted(
            compiled.analysis.fields - {"datetime", "vt_symbol"}
        )
        query = self.frame.select(columns).lazy()
        for stage in compiled.stages:
            query = query.with_columns(stage)
        return (
            query.select("datetime", "vt_symbol", compiled.expression.cast(pl.Float64).alias(name))
            .with_columns(pl.when(pl.col(name).is_finite()).then(pl.col(name)).otherwise(None).alias(name))
            .sort(["datetime", "vt_symbol"])
            .collect()
        )
