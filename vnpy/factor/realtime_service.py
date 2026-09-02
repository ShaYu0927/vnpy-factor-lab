"""Realtime bridge from market bars into the unified Alpha engine."""

from __future__ import annotations

from datetime import datetime
from typing import Mapping, Sequence

from vnpy.alpha.definition import AlphaDefinition
from vnpy.alpha.engine import AlphaEngine, AlphaSample, AlphaSampleCache
from vnpy.datafeed.bar_cache import BarCache
from vnpy.factor.core.factor_engine import FactorBatchResult, FactorValue


class RealtimeAlphaService:
    """Calculate registered Alpha expressions from cached current/past bars."""

    def __init__(
        self,
        bar_cache: BarCache,
        sample_cache: AlphaSampleCache,
        definitions: Sequence[AlphaDefinition] = (),
        frequency: str = "60s",
        universe: Sequence[str] | None = None,
        alpha_engine: AlphaEngine | None = None,
        **_legacy_options,
    ) -> None:
        self.bar_cache = bar_cache
        self.sample_cache = sample_cache
        self.frequency = frequency
        self.definitions = tuple(definitions)
        self.alpha_engine = alpha_engine or AlphaEngine(self.definitions)
        self.universe = tuple(dict.fromkeys(universe or ()))
        self.latest_batch_result = FactorBatchResult()

    def on_bar(self, bar) -> AlphaSample | None:
        if bar is None or not self.definitions:
            return None
        if not getattr(bar, "frequency", None):
            bar.frequency = self.frequency
        self.bar_cache.update(bar)
        at = _bar_datetime(bar)
        symbols = self.universe or tuple(self.bar_cache.symbols())
        bars_by_symbol = {
            symbol: self.bar_cache.get_bars(
                symbol=symbol,
                frequency=self.frequency,
                count=self.alpha_engine.min_bars,
            )
            for symbol in symbols
        }
        bars_by_symbol = {symbol: bars for symbol, bars in bars_by_symbol.items() if len(bars) >= self.alpha_engine.min_bars}
        if not bars_by_symbol:
            return None

        if self.alpha_engine.requires_cross_section:
            if not self.universe or len(bars_by_symbol) != len(self.universe):
                return None
            if any(_bar_datetime(bars[-1]) != at for bars in bars_by_symbol.values()):
                return None

        frame = self.alpha_engine.from_bars(bars_by_symbol)
        samples = self.alpha_engine.calculate_latest(frame, at=at)
        self.latest_batch_result = _to_factor_result(samples)
        for sample in samples:
            self.sample_cache.add(sample)
        return next((sample for sample in samples if sample.symbol == bar.symbol), None)

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
        return self.alpha_engine.calculate_latest(frame, at=at)

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
