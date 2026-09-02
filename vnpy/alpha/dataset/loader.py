from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import polars as pl

from .utility import calculate_by_expression


FeatureConfig = tuple[Sequence[str], Sequence[str]]


_QLIB_OPERATOR_ALIASES: Mapping[str, str] = {
    "Ref": "ts_delay",
    "Mean": "ts_mean",
    "Std": "ts_std",
}


def translate_qlib_expression(expression: str) -> str:
    """Translate the supported Qlib expression syntax to local operators."""
    translated = re.sub(r"\$([A-Za-z_]\w*)", r"\1", expression)
    for qlib_name, local_name in _QLIB_OPERATOR_ALIASES.items():
        translated = re.sub(
            rf"\b{re.escape(qlib_name)}\b",
            local_name,
            translated,
        )
    return translated


class QlibDataLoader:
    """Calculate Qlib-style ``fields``/``names`` configs with Polars."""

    def __init__(self, config: Mapping[str, FeatureConfig] | None = None) -> None:
        self.config: dict[str, FeatureConfig] = dict(config or {})

    def load(self, df: pl.DataFrame) -> pl.DataFrame:
        """Return one wide feature matrix keyed by datetime and symbol."""
        if "datetime" not in df.columns:
            raise ValueError("input data must contain 'datetime'")
        symbol_column = self._symbol_column(df)
        source = (
            df.rename({symbol_column: "vt_symbol"})
            if symbol_column != "vt_symbol"
            else df
        )
        source = source.sort(["vt_symbol", "datetime"])
        self._validate_source(source)

        result = source.select(["datetime", "vt_symbol"])
        for task in ("feature", "label"):
            task_config = self.config.get(task)
            if task_config is None:
                continue

            fields, names = task_config
            self._validate_config(task, fields, names)
            for field, name in zip(fields, names, strict=True):
                expression = translate_qlib_expression(field)
                calculated = calculate_by_expression(source, expression).rename(
                    {"data": name}
                )
                result = result.join(
                    calculated,
                    on=["datetime", "vt_symbol"],
                    how="left",
                )

        result = result.sort(["datetime", "vt_symbol"])
        if symbol_column == "instrument":
            result = result.rename({"vt_symbol": "instrument"})
        return result

    @staticmethod
    def _symbol_column(df: pl.DataFrame) -> str:
        if "vt_symbol" in df.columns:
            return "vt_symbol"
        if "instrument" in df.columns:
            return "instrument"
        raise ValueError("input data must contain 'vt_symbol' or 'instrument'")

    @staticmethod
    def _validate_source(df: pl.DataFrame) -> None:
        duplicates = (
            df.group_by(["datetime", "vt_symbol"])
            .len()
            .filter(pl.col("len") > 1)
        )
        if duplicates.height:
            raise ValueError("input data contains duplicate datetime/symbol rows")

    @staticmethod
    def _validate_config(task: str, fields: Sequence[str], names: Sequence[str],) -> None:
        if len(fields) != len(names):
            raise ValueError(
                f"{task} fields and names must have the same length"
            )
        if len(names) != len(set(names)):
            raise ValueError(f"{task} names must be unique")
