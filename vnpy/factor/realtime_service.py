"""Realtime bridge from market bars into the unified Alpha engine."""

from __future__ import annotations

from datetime import datetime
from math import isfinite
from typing import Mapping, Sequence

from vnpy.alpha.definition import AlphaDefinition
from vnpy.alpha.engine import AlphaEngine, AlphaSample, AlphaSampleCache
from vnpy.datafeed.bar_cache import BarCache
from vnpy.factor.core.factor_engine import FactorBatchResult, FactorValue, FactorStatus
from vnpy.alpha.logger import logger


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
        alpha101_factors: Sequence[int] | None = None,
        alpha101_history: int = 320,
        **_legacy_options,
    ) -> None:
        self.bar_cache = bar_cache
        self.sample_cache = sample_cache
        self.frequency = frequency
        self.definitions = tuple(definitions)
        self.alpha_engine = alpha_engine or AlphaEngine(self.definitions)
        self.universe = tuple(dict.fromkeys(universe or ()))
        self.latest_batch_result = FactorBatchResult()
        self.latest_samples: list[AlphaSample] = []
        self.alpha101_factors = None if alpha101_factors is None else tuple(alpha101_factors)
        self.alpha101_history = alpha101_history
        self._last_alpha101_at: datetime | None = None
        if self.alpha101_factors is not None:
            if not self.alpha101_factors or any(type(n) is not int or not 1 <= n <= 101 for n in self.alpha101_factors):
                raise ValueError("alpha101_factors must contain integers in 1..101")
            if len(set(self.alpha101_factors)) != len(self.alpha101_factors):
                raise ValueError("alpha101_factors must be unique")
            if self.definitions:
                raise ValueError("configure Alpha101 and expression alphas in separate factor modules")
            if not self.universe:
                raise ValueError("Alpha101 events require an explicit universe")
            if frequency != "1d":
                raise ValueError("Alpha101 events require daily bars (frequency='1d')")
            if not 1 <= alpha101_history <= bar_cache.maxlen:
                raise ValueError("alpha101_history must be positive and fit in bar cache")
            logger.info("Alpha101 events enabled: universe=%s factors=%d history=%d",
                        self.universe, len(self.alpha101_factors), alpha101_history)

    def on_bar(self, bar) -> AlphaSample | None:
        self.latest_samples = []
        self.latest_batch_result = FactorBatchResult()
        if self.alpha101_factors is not None:
            return self._on_alpha101_bar(bar)
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
        self.latest_samples = samples
        for sample in samples:
            self.sample_cache.add(sample)
        return next((sample for sample in samples if sample.symbol == bar.symbol), None)

    def _on_alpha101_bar(self, bar) -> AlphaSample | None:
        if bar is None or bar.symbol not in self.universe or bar.frequency != self.frequency:
            return None
        at = _bar_datetime(bar)
        previous = self.bar_cache.get_last_bar(bar.symbol, self.frequency)
        if (self._last_alpha101_at is not None and at <= self._last_alpha101_at
                or previous is not None and at < _bar_datetime(previous)):
            logger.debug("Alpha101 ignored duplicate/late bar: symbol=%s at=%s", bar.symbol, at)
            return None
        self.bar_cache.update(bar)
        bars = {symbol: self.bar_cache.get_bars(symbol, count=self.alpha101_history, frequency=self.frequency)
                for symbol in self.universe}
        waiting = [symbol for symbol, history in bars.items() if not history or _bar_datetime(history[-1]) != at]
        if waiting:
            logger.debug("Alpha101 waiting: at=%s symbols=%s", at, waiting)
            return None
        frame = self.alpha_engine.from_bars(bars)
        calculated = self.alpha_engine.calculate_alpha101(frame, self.alpha101_factors)
        import polars as pl
        rows = calculated.filter(pl.col("datetime") == at).to_dicts()
        values = []
        for row in rows:
            symbol = row["vt_symbol"]
            features = {}
            for number in self.alpha101_factors:
                name = f"alpha{number:03d}"
                raw = row[name]
                valid = raw is not None and isfinite(float(raw))
                if valid:
                    features[name] = float(raw)
                values.append(FactorValue(
                    symbol=symbol, factor_name=name, value=float(raw) if valid else None,
                    trade_date=at.isoformat(),
                    status=FactorStatus.READY if valid else FactorStatus.INVALID,
                    reason="" if valid else "insufficient history, missing input or undefined formula result",
                ))
            sample = AlphaSample(symbol=symbol, datetime=at, close=float(bars[symbol][-1].close), features=features)
            self.latest_samples.append(sample)
            self.sample_cache.add(sample)
        self.latest_batch_result = FactorBatchResult(values=values)
        self._last_alpha101_at = at
        logger.info("Alpha101 event batch: at=%s symbols=%d ready=%d/%d", at,
                    len(self.latest_samples), sum(value.is_ready for value in values), len(values))
        return next((sample for sample in self.latest_samples if sample.symbol == bar.symbol), None)

    def calculate_latest_cross_section(
        self,
        symbols: Sequence[str],
        at: datetime | None = None,
        count: int | None = None,
        **_legacy_options,
    ) -> list[AlphaSample]:
        required = count or self.alpha_engine.min_bars
        if self.alpha101_factors is not None:
            required = count or self.alpha101_history
        bars = {
            symbol: self.bar_cache.get_bars(symbol=symbol, frequency=self.frequency, count=required)
            for symbol in symbols
        }
        frame = self.alpha_engine.from_bars({key: value for key, value in bars.items() if value})
        if self.alpha101_factors is not None:
            return self.alpha_engine.calculate_alpha101_latest(frame, self.alpha101_factors, at=at)
        return self.alpha_engine.calculate_latest(frame, at=at)

    def calculate(self, frame) -> object:
        """Expose the same expression engine to offline callers."""
        if self.alpha101_factors is not None:
            return self.alpha_engine.calculate_alpha101(frame, self.alpha101_factors)
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
