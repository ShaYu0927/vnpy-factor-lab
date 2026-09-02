from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from math import isfinite
from collections import defaultdict, deque
from typing import Any, Deque, Iterable, Mapping, Sequence

import polars as pl

from .dataset.utility import calculate_by_expression
from .definition import AlphaDefinition


@dataclass(frozen=True, slots=True)
class AlphaSample:
    """Alpha values known at one symbol/time; labels never belong here."""
    symbol: str
    datetime: datetime
    close: float
    features: Mapping[str, float | str | bool]

    def __getattr__(self, name: str) -> float | str | bool:
        """Allow transitional strategies to read a feature by name."""
        try:
            return self.features[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class AlphaSampleCache:
    """Cache observable alpha features without manufacturing future labels."""

    def __init__(self, maxlen: int = 30_000) -> None:
        self._samples: dict[str, Deque[AlphaSample]] = defaultdict(lambda: deque(maxlen=maxlen))

    def add(self, sample: AlphaSample) -> None:
        samples = self._samples[sample.symbol]
        if samples and samples[-1].datetime == sample.datetime:
            samples[-1] = sample
        else:
            samples.append(sample)

    def get_latest(self, symbol: str) -> AlphaSample | None:
        samples = self._samples.get(symbol)
        return samples[-1] if samples else None

    def get_recent(self, symbol: str, count: int) -> list[AlphaSample]:
        return list(self._samples.get(symbol, ())) [-max(count, 0):] if count > 0 else []

    def symbols(self) -> list[str]:
        return list(self._samples)

    def clear_all(self) -> None:
        self._samples.clear()


class AlphaEngine:
    """Evaluate the same alpha definitions for offline and live data."""

    def __init__(self, definitions: Sequence[AlphaDefinition]) -> None:
        names = [item.name for item in definitions]
        if len(names) != len(set(names)):
            raise ValueError("alpha names must be unique")
        self.definitions = tuple(definitions)

    @property
    def min_bars(self) -> int:
        return max((item.lookback for item in self.definitions), default=1)

    @property
    def requires_cross_section(self) -> bool:
        return any(item.uses_cross_section for item in self.definitions)

    def calculate(self, frame: pl.DataFrame) -> pl.DataFrame:
        source = self._normalize_frame(frame)
        result = source.select(["datetime", "vt_symbol"])
        for definition in self.definitions:
            values = calculate_by_expression(source, definition.expression).rename({"data": definition.name})
            result = result.join(values, on=["datetime", "vt_symbol"], how="left")
        return result.sort(["datetime", "vt_symbol"])

    def calculate_latest(self, frame: pl.DataFrame, at: datetime | None = None) -> list[AlphaSample]:
        source = self._normalize_frame(frame)
        calculated = self.calculate(source)
        if at is None:
            latest = calculated.group_by("vt_symbol").agg(pl.all().sort_by("datetime").last())
        else:
            latest = calculated.filter(pl.col("datetime") == pl.lit(at))
        closes = source.select(["datetime", "vt_symbol", "close"])
        latest = latest.join(closes, on=["datetime", "vt_symbol"], how="left")
        samples: list[AlphaSample] = []
        for row in latest.iter_rows(named=True):
            features = {
                definition.name: float(row[definition.name])
                for definition in self.definitions
                if row.get(definition.name) is not None and isfinite(float(row[definition.name]))
            }
            if len(features) != len(self.definitions):
                continue
            samples.append(AlphaSample(
                symbol=str(row["vt_symbol"]), datetime=row["datetime"],
                close=float(row["close"]), features=features,
            ))
        return sorted(samples, key=lambda item: item.symbol)

    def from_bars(self, bars_by_symbol: Mapping[str, Iterable[Any]]) -> pl.DataFrame:
        rows = []
        for symbol, bars in bars_by_symbol.items():
            for bar in bars:
                dt = _bar_value(bar, "datetime", "bob", "eob")
                row = {"datetime": dt, "vt_symbol": symbol}
                for field in ("open", "high", "low", "close", "volume", "amount", "turn"):
                    value = _bar_value(bar, field, required=False)
                    if value is not None:
                        row[field] = value
                rows.append(row)
        if not rows:
            return pl.DataFrame({"datetime": [], "vt_symbol": []})
        return pl.DataFrame(rows).sort(["vt_symbol", "datetime"])

    def to_observations(self, samples: Sequence[AlphaSample]):
        from .modeling.schema import FactorObservation

        return [
            FactorObservation(
                trade_date=_to_date(sample.datetime), symbol=sample.symbol,
                close=sample.close, features=dict(sample.features),
            )
            for sample in samples
        ]

    @staticmethod
    def _normalize_frame(frame: pl.DataFrame) -> pl.DataFrame:
        required = {"datetime", "vt_symbol", "close"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"alpha input is missing columns: {', '.join(sorted(missing))}")
        duplicates = frame.group_by(["datetime", "vt_symbol"]).len().filter(pl.col("len") > 1)
        if duplicates.height:
            raise ValueError("alpha input contains duplicate datetime/symbol rows")
        return frame.sort(["vt_symbol", "datetime"])


def _bar_value(bar: Any, *names: str, required: bool = True) -> Any:
    for name in names:
        value = bar.get(name) if isinstance(bar, Mapping) else getattr(bar, name, None)
        if value is not None:
            return value
    if required:
        raise ValueError(f"bar is missing required field: {'/'.join(names)}")
    return None


def _to_date(value: datetime) -> date:
    return value.date()
