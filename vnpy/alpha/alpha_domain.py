"""Shared market and factor tables for Alpha calculation and research."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import polars as pl

from .alpha_formula import Alpha
from .alpha_time_split import SessionSplit, TimeSplitConfig, build_session_split


_KEY_COLUMNS = ("datetime", "vt_symbol")


def _validate_keys(frame: pl.DataFrame, kind: str) -> None:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError(f"{kind} must be a Polars DataFrame")
    missing = set(_KEY_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{kind} is missing columns: {', '.join(sorted(missing))}")
    if frame.schema["datetime"].base_type() not in {pl.Date, pl.Datetime}:
        raise ValueError(f"{kind}.datetime must be a Polars Date or Datetime column")
    if frame.schema["vt_symbol"] != pl.String:
        raise ValueError(f"{kind}.vt_symbol must be a string column")
    if any(frame[key].null_count() for key in _KEY_COLUMNS):
        raise ValueError(f"{kind} keys must not be null")
    if frame.select(pl.struct(*_KEY_COLUMNS).is_duplicated().any()).item():
        raise ValueError(f"{kind} contains duplicate datetime/symbol rows")


@dataclass(frozen=True, slots=True, eq=False)
class MarketFrame:
    """Observed market rows, unique by timestamp and symbol.

    Missing bars stay missing. Price/volume values may be null or invalid; the
    consuming evaluation decides whether a particular row is usable.
    """

    frame: pl.DataFrame

    def __post_init__(self) -> None:
        _validate_keys(self.frame, "market")
        if "close" not in self.frame.columns:
            raise ValueError("market is missing columns: close")
        if not self.frame.schema["close"].is_numeric():
            raise ValueError("market.close must be numeric")
        object.__setattr__(self, "frame", self.frame.sort(["vt_symbol", "datetime"]))

    @property
    def sessions(self) -> tuple[date | datetime, ...]:
        """The observed, sorted trading calendar of the complete stock panel."""
        return tuple(self.frame["datetime"].unique().sort().to_list())

    def split(self, config: TimeSplitConfig) -> SessionSplit:
        return build_session_split(self.sessions, config)


@dataclass(frozen=True, slots=True, eq=False)
class FactorFrame:
    """One nullable numeric factor value per timestamp/symbol and factor name."""

    frame: pl.DataFrame

    def __post_init__(self) -> None:
        _validate_keys(self.frame, "factors")
        frame = self.frame
        for name in self.names:
            if not name.isidentifier():
                raise ValueError(f"factor column {name!r} must be named and numeric")
            if frame.schema[name] == pl.Null:
                frame = frame.with_columns(pl.col(name).cast(pl.Float64))
            elif not frame.schema[name].is_numeric():
                raise ValueError(f"factor column {name!r} must be named and numeric")
        object.__setattr__(self, "frame", frame.sort(["datetime", "vt_symbol"]))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(name for name in self.frame.columns if name not in _KEY_COLUMNS)

    def require_market_keys(self, market: MarketFrame) -> None:
        """Reject factor observations without a matching market observation."""
        if self.frame.schema["datetime"] != market.frame.schema["datetime"]:
            raise ValueError(
                "factor and market datetime columns must use the same type"
            )
        unknown = self.frame.select(_KEY_COLUMNS).join(
            market.frame.select(_KEY_COLUMNS), on=_KEY_COLUMNS, how="anti"
        )
        if not unknown.is_empty():
            raise ValueError("factor keys must belong to market data")


__all__ = ["Alpha", "MarketFrame", "FactorFrame", "SessionSplit", "TimeSplitConfig"]
