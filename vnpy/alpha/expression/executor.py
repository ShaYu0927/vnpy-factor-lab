"""Execute compiled formulas on keyed market data."""

from __future__ import annotations

import polars as pl

from .polars_compiler import CompiledExpression


class PolarsExecutor:
    """使用 Polars 执行已编译的因子表达式。

    输入数据以 (datetime, vt_symbol) 唯一标识一条股票记录。
    """
    def __init__(self, frame: pl.DataFrame) -> None:
        """校验输入数据，并按股票和时间排序。

        时序算子需要同一只股票的数据按时间排列，因此在执行因子前
        统一完成排序。
        """
        keys = {"datetime", "vt_symbol"}
        if missing := keys - set(frame.columns):
            raise ValueError(f"alpha input is missing columns: {', '.join(sorted(missing))}")
        if any(frame[key].null_count() for key in keys):
            raise ValueError("alpha input keys must not be null")
        if frame.select(pl.struct("datetime", "vt_symbol").is_duplicated().any()).item():
            raise ValueError("alpha input contains duplicate datetime/symbol rows")
        self.frame = frame.sort(["vt_symbol", "datetime"])

    def run(self, compiled: CompiledExpression, name: str = "data") -> pl.DataFrame:
        """计算因子并返回 datetime、vt_symbol 和因子值三列。

        Args:
            compiled: 已编译的因子表达式，包含依赖字段、计算阶段和结果表达式。
            name: 输出的因子列名。

        Returns:
            按时间和股票代码排序的因子数据；无穷大及 NaN 结果转为 null。
        """
        if not name.isidentifier() or name in {"datetime", "vt_symbol"}:
            raise ValueError("alpha name must be an identifier distinct from the index columns")
        if missing := compiled.analysis.fields - set(self.frame.columns):
            raise ValueError(f"alpha input is missing columns: {', '.join(sorted(missing))}")
         # 只保留索引和依赖字段，避免原始数据列与编译器的临时列重名。.
        columns = ["datetime", "vt_symbol"] + sorted(compiled.analysis.fields - {"datetime", "vt_symbol"})
        query = self.frame.select(columns).lazy()
        for stage in compiled.stages:
            query = query.with_columns(stage)
        return (
            query.select("datetime", "vt_symbol", compiled.expression.cast(pl.Float64).alias(name))
            .with_columns(pl.when(pl.col(name).is_finite()).then(pl.col(name)).otherwise(None).alias(name))
            .sort(["datetime", "vt_symbol"])
            .collect()
        )
