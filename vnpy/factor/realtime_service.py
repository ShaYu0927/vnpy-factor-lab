"""Realtime bridge from market bars into the unified Alpha engine."""

from __future__ import annotations

from datetime import datetime
from typing import Mapping, Sequence

from vnpy.alpha.definition import AlphaDefinition
from vnpy.alpha.alpha import Alpha
from vnpy.alpha.engine import AlphaEngine, AlphaSample, AlphaSampleCache
from vnpy.datafeed.data_bar_cache import BarCache
from vnpy.factor.core.factor_engine import FactorBatchResult, FactorValue


class RealtimeAlphaService:
    """Calculate registered Alpha expressions from cached current/past bars."""

    def __init__(
        self,
        bar_cache: BarCache,
        sample_cache: AlphaSampleCache,
        definitions: Sequence[AlphaDefinition | Alpha] = (),
        frequency: str = "60s",
        universe: Sequence[str] | None = None,
        alpha_engine: AlphaEngine | None = None,
        optional_names: Sequence[str] = (),
        **_legacy_options,
    ) -> None:
        self.bar_cache = bar_cache
        self.sample_cache = sample_cache
        self.frequency = frequency
        self.definitions = tuple(definitions)
        self.alpha_engine = alpha_engine or AlphaEngine(self.definitions)
        self.optional_names = tuple(optional_names)
        if set(self.optional_names) - {item.name for item in self.alpha_engine.definitions}:
            raise ValueError("optional alpha names must be registered definitions")
        self.required_min_bars = max(
            (item.lookback for item in self.alpha_engine.definitions if item.name not in self.optional_names),
            default=1,
        )
        self.universe = tuple(dict.fromkeys(universe or ()))
        self._universe_set = frozenset(self.universe)
        self.latest_batch_result = FactorBatchResult()
        self.latest_samples: list[AlphaSample] = []

    def on_bar(self, bar) -> AlphaSample | None:
        samples = self.on_bars([bar] if bar is not None else [])
        return next((sample for sample in samples if sample.symbol == bar.symbol), None)

    def on_bars(self, bars: Sequence) -> list[AlphaSample]:
        """Calculate one timestamp in a single vectorized batch, preserving per-symbol history."""
        self.latest_samples = []
        self.latest_batch_result = FactorBatchResult()
        bars = [bar for bar in bars if bar is not None and (
            not self._universe_set or bar.symbol in self._universe_set
        )]
        if not bars or not self.definitions:
            return []
        at = _bar_datetime(bars[0])
        if any(_bar_datetime(bar) != at for bar in bars):
            raise ValueError("a bar batch must contain a single timestamp")
        symbols = tuple(bar.symbol for bar in bars)
        if len(set(symbols)) != len(symbols):
            raise ValueError("a bar batch must contain unique symbols")
        for bar in bars:
            if not getattr(bar, "frequency", None):
                bar.frequency = self.frequency
            previous = self.bar_cache.get_last_bar(bar.symbol, self.frequency)
            if previous is not None and _bar_datetime(previous) > at:
                raise ValueError(f"out-of-order bar for {bar.symbol}")
        for bar in bars:
            self.bar_cache.update(bar)
        if self.alpha_engine.requires_cross_section:
            symbols = self.universe or tuple(self.bar_cache.symbols())
        bars_by_symbol = {
            symbol: self.bar_cache.get_bars(
                symbol=symbol,
                frequency=self.frequency,
                count=self.alpha_engine.min_bars,
            )
            for symbol in symbols
        }
        bars_by_symbol = {symbol: bars for symbol, bars in bars_by_symbol.items() if len(bars) >= self.required_min_bars}
        if not bars_by_symbol:
            return []

        if self.alpha_engine.requires_cross_section:
            if not self.universe or len(bars_by_symbol) != len(self.universe):
                return []
            if any(_bar_datetime(bars[-1]) != at for bars in bars_by_symbol.values()):
                return []

        frame = self.alpha_engine.from_bars(bars_by_symbol)
        samples = self.alpha_engine.calculate_latest(frame, at=at, optional_names=self.optional_names)
        self.latest_batch_result = _to_factor_result(samples)
        self.latest_samples = samples
        for sample in samples:
            self.sample_cache.add(sample)
        return samples

    def calculate_latest_cross_section(
        self,
        symbols: Sequence[str],
        at: datetime | None = None,
        count: int | None = None,
        **_legacy_options,
    ) -> list[AlphaSample]:
        required = count or self.alpha_engine.min_bars
        bars = {
            symbol: self.bar_cache.get_bars(symbol=symbol, frequency=self.frequency, count=required)
            for symbol in symbols
        }
        frame = self.alpha_engine.from_bars({key: value for key, value in bars.items() if value})
        return self.alpha_engine.calculate_latest(frame, at=at, optional_names=self.optional_names)

    def calculate(self, frame) -> object:
        """Expose the same expression engine to offline callers."""
        return self.alpha_engine.calculate(frame)


def _bar_datetime(bar) -> datetime:
    value = getattr(bar, "datetime", None) or getattr(bar, "bob", None) or getattr(bar, "eob", None)
    if not isinstance(value, datetime):
        raise ValueError("bar must contain a datetime/bob/eob datetime")
    return value


def _to_factor_result(samples: Sequence[AlphaSample]) -> FactorBatchResult:
    values = [
        FactorValue(
            symbol=sample.symbol,
            factor_name=name,
            value=value,
            trade_date=sample.datetime.isoformat(),
            primary_field="value",
        )
        for sample in samples
        for name, value in sample.features.items()
    ]
    return FactorBatchResult(values=values)


# Keep the former import path working while routing it through Alpha.
RealtimeFactorService = RealtimeAlphaService

__all__ = ["RealtimeAlphaService", "RealtimeFactorService"]
