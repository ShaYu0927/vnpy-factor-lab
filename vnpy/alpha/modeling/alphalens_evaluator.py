from __future__ import annotations

import sqlite3
from contextlib import redirect_stdout
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl


@dataclass(slots=True)
class AlphalensReport:
    """Structured Alphalens output suitable for batch jobs and notebooks."""
    clean_data: pd.DataFrame
    information_coefficient: pd.DataFrame
    mean_information_coefficient: pd.Series | pd.DataFrame
    information_ratio: pd.Series
    quantile_returns: pd.DataFrame
    quantile_standard_error: pd.DataFrame
    turnover: dict[tuple[int, int], pd.Series]


class AlphalensEvaluator:
    """Adapt Polars factor and price frames to Alphalens Reloaded."""

    def __init__(
        self,
        periods: tuple[int, ...] = (1, 5, 10),
        quantiles: int = 5,
        max_loss: float = 0.35,
        group_adjust: bool = False,
    ) -> None:
        if not periods or any(period <= 0 for period in periods):
            raise ValueError("periods must contain positive integers")
        if quantiles < 2:
            raise ValueError("quantiles must be at least 2")
        if not 0 <= max_loss <= 1:
            raise ValueError("max_loss must be between 0 and 1")
        self.periods = tuple(dict.fromkeys(periods))
        self.quantiles = quantiles
        self.max_loss = max_loss
        self.group_adjust = group_adjust

    def evaluate(
        self,
        factor: pl.DataFrame | pd.DataFrame | pd.Series,
        prices: pl.DataFrame | pd.DataFrame,
        *,
        factor_column: str = "factor",
        datetime_column: str = "datetime",
        symbol_column: str = "vt_symbol",
        price_column: str = "close",
        groups: Mapping[str, str] | None = None,
    ) -> AlphalensReport:
        """Calculate reusable metrics without rendering charts."""
        if self.group_adjust and groups is None:
            raise ValueError("groups are required when group_adjust is enabled")

        from alphalens import performance
        from alphalens.utils import get_clean_factor_and_forward_returns

        factor_series = self.prepare_factor(
            factor,
            factor_column=factor_column,
            datetime_column=datetime_column,
            symbol_column=symbol_column,
        )
        price_frame = self.prepare_prices(
            prices,
            datetime_column=datetime_column,
            symbol_column=symbol_column,
            price_column=price_column,
        )
        # Alphalens prints its data-loss report directly to stdout. Keep the
        # evaluator usable in quiet batch jobs; failures still propagate.
        with redirect_stdout(StringIO()):
            clean_data = get_clean_factor_and_forward_returns(
                factor=factor_series,
                prices=price_frame,
                periods=self.periods,
                quantiles=self.quantiles,
                max_loss=self.max_loss,
                groupby=None if groups is None else dict(groups),
            )
        information_coefficient = performance.factor_information_coefficient(
            clean_data,
            group_adjust=self.group_adjust,
        )
        mean_ic = performance.mean_information_coefficient(
            clean_data,
            group_adjust=self.group_adjust,
        )
        ic_std = information_coefficient.std().replace(0.0, np.nan)
        information_ratio = information_coefficient.mean() / ic_std
        quantile_returns, quantile_standard_error = performance.mean_return_by_quantile(
            clean_data,
            demeaned=True,
            group_adjust=self.group_adjust,
        )
        turnover = self._calculate_turnover(clean_data)
        return AlphalensReport(
            clean_data=clean_data,
            information_coefficient=information_coefficient,
            mean_information_coefficient=mean_ic,
            information_ratio=information_ratio,
            quantile_returns=quantile_returns,
            quantile_standard_error=quantile_standard_error,
            turnover=turnover,
        )

    def create_full_tear_sheet(
        self,
        report: AlphalensReport,
        *,
        long_short: bool = True,
        group_neutral: bool = False,
        by_group: bool = False,
    ) -> Any:
        """Render Alphalens charts explicitly when an interactive report is wanted."""
        from alphalens.tears import create_full_tear_sheet

        return create_full_tear_sheet(
            report.clean_data,
            long_short=long_short,
            group_neutral=group_neutral,
            by_group=by_group,
        )

    @staticmethod
    def prepare_factor(
        factor: pl.DataFrame | pd.DataFrame | pd.Series,
        *,
        factor_column: str = "factor",
        datetime_column: str = "datetime",
        symbol_column: str = "vt_symbol",
    ) -> pd.Series:
        """Return a finite, unique Series indexed by date and asset."""
        if isinstance(factor, pd.Series):
            series = factor.copy()
            if not isinstance(series.index, pd.MultiIndex) or series.index.nlevels != 2:
                raise ValueError("factor Series must use a two-level date/asset MultiIndex")
            series.index = series.index.set_names(["date", "asset"])
        else:
            frame = factor.to_pandas() if isinstance(factor, pl.DataFrame) else factor.copy()
            required = {datetime_column, symbol_column, factor_column}
            missing = required.difference(frame.columns)
            if missing:
                raise ValueError(f"factor data is missing columns: {sorted(missing)}")
            frame = frame.loc[:, [datetime_column, symbol_column, factor_column]].copy()
            frame[datetime_column] = pd.to_datetime(frame[datetime_column])
            if frame.duplicated([datetime_column, symbol_column]).any():
                raise ValueError("factor data contains duplicate date/symbol rows")
            series = frame.set_index([datetime_column, symbol_column])[factor_column]
            series.index = series.index.set_names(["date", "asset"])

        numeric = pd.to_numeric(series, errors="coerce")
        numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna().sort_index()
        if numeric.empty:
            raise ValueError("factor data contains no finite values")
        return numeric.astype(float)

    @classmethod
    def factor_from_batch_csv(cls, path: str | Path, factor_name: str,) -> pd.Series:
        """Load one ready factor from ``CsvFactorResultWriter`` output."""
        frame = pd.read_csv(
            Path(path).expanduser().resolve(),
            usecols=["trade_date", "symbol", "factor_name", "value", "status"],
        )
        return cls._prepare_batch_factor(frame, factor_name)

    @classmethod
    def factor_from_batch_sqlite(cls, path: str | Path, factor_name: str,) -> pd.Series:
        """Load one ready factor from ``SqliteFactorResultWriter`` output."""
        database = Path(path).expanduser().resolve()
        connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
        try:
            frame = pd.read_sql_query(
                """
                SELECT trade_date, symbol, factor_name, value, status
                FROM factor_values
                WHERE factor_name = ? AND status = 'ready' AND value IS NOT NULL
                ORDER BY trade_date, symbol
                """,
                connection,
                params=(factor_name,),
            )
        finally:
            connection.close()
        return cls._prepare_batch_factor(frame, factor_name)

    @classmethod
    def _prepare_batch_factor(cls, frame: pd.DataFrame, factor_name: str,) -> pd.Series:
        selected = frame.loc[
            (frame["factor_name"] == factor_name) & (frame["status"] == "ready"),
            ["trade_date", "symbol", "value"],
        ].rename(columns={
            "trade_date": "datetime",
            "symbol": "vt_symbol",
            "value": "factor",
        })
        if selected.empty:
            raise ValueError(f"no ready values found for factor {factor_name!r}")
        return cls.prepare_factor(selected)

    @staticmethod
    def prepare_prices(prices: pl.DataFrame | pd.DataFrame, *, datetime_column: str = "datetime", symbol_column: str = "vt_symbol", price_column: str = "close",) -> pd.DataFrame:
        """Return the date-by-asset price matrix required by Alphalens."""
        frame = prices.to_pandas() if isinstance(prices, pl.DataFrame) else prices.copy()
        long_columns = {datetime_column, symbol_column, price_column}
        if long_columns.issubset(frame.columns):
            if frame.duplicated([datetime_column, symbol_column]).any():
                raise ValueError("price data contains duplicate date/symbol rows")
            frame[datetime_column] = pd.to_datetime(frame[datetime_column])
            frame = frame.pivot(
                index=datetime_column,
                columns=symbol_column,
                values=price_column,
            )
        else:
            if not isinstance(frame.index, pd.DatetimeIndex):
                try:
                    frame.index = pd.to_datetime(frame.index)
                except Exception as exc:
                    raise ValueError(
                        "wide price data must use a DatetimeIndex"
                    ) from exc

        frame.index.name = "date"
        frame.columns.name = "asset"
        frame = frame.apply(pd.to_numeric, errors="coerce")
        frame = frame.replace([np.inf, -np.inf], np.nan).sort_index()
        if frame.empty or not frame.notna().any().any():
            raise ValueError("price data contains no finite values")
        return frame

    @classmethod
    def prices_from_bars(
        cls,
        bars: Iterable[Any],
        *,
        price_field: str = "close",
    ) -> pd.DataFrame:
        """Build an Alphalens price matrix from MarketBar/VnPy-like objects."""
        rows: list[dict[str, Any]] = []
        for bar in bars:
            if isinstance(bar, Mapping):
                symbol = bar.get("symbol") or bar.get("vt_symbol")
                timestamp = bar.get("bob") or bar.get("datetime")
                price = bar.get(price_field)
            else:
                symbol = getattr(bar, "symbol", None) or getattr(bar, "vt_symbol", None)
                timestamp = getattr(bar, "bob", None) or getattr(bar, "datetime", None)
                price = getattr(bar, price_field, None)
            if symbol is None or timestamp is None or price is None:
                continue
            rows.append({
                "datetime": timestamp,
                "vt_symbol": str(symbol),
                "close": price,
            })
        if not rows:
            raise ValueError("bars contain no usable symbol/date/price rows")
        return cls.prepare_prices(pd.DataFrame(rows))

    def _calculate_turnover(
        self,
        clean_data: pd.DataFrame,
    ) -> dict[tuple[int, int], pd.Series]:
        from alphalens.performance import quantile_turnover

        quantile_factor = clean_data["factor_quantile"]
        result: dict[tuple[int, int], pd.Series] = {}
        for period in self.periods:
            for quantile in range(1, self.quantiles + 1):
                result[(period, quantile)] = quantile_turnover(
                    quantile_factor,
                    quantile,
                    period=period,
                )
        return result
