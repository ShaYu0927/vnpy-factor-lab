from dataclasses import dataclass
from datetime import datetime, timedelta

import polars as pl
import pytest

from vnpy.alpha import AlphaDefinition, AlphaEngine
from vnpy.alpha.engine import AlphaSampleCache
from vnpy.datafeed.bar_cache import BarCache
from vnpy.factor.realtime_service import RealtimeAlphaService


@dataclass
class SampleBar:
    symbol: str
    frequency: str
    bob: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float = 0.0
    turn: float = 0.0


def make_frame() -> pl.DataFrame:
    start = datetime(2025, 1, 1)
    return pl.DataFrame([
        {
            "datetime": start + timedelta(days=day),
            "vt_symbol": symbol,
            "close": base + day,
            "volume": 100 + day,
        }
        for symbol, base in (("A", 10.0), ("B", 20.0))
        for day in range(4)
    ])


def test_same_definitions_produce_frame_samples_and_observations() -> None:
    definition = AlphaDefinition("mean_2", "ts_mean(close, 2)", lookback=2)
    engine = AlphaEngine([definition])

    calculated = engine.calculate(make_frame())
    samples = engine.calculate_latest(make_frame())
    observations = engine.to_observations(samples)

    assert "mean_2" in calculated.columns
    assert [sample.symbol for sample in samples] == ["A", "B"]
    assert samples[0].features["mean_2"] == pytest.approx(12.5)
    assert observations[0].features == {"mean_2": pytest.approx(12.5)}


def test_alpha_definition_rejects_future_or_label_access() -> None:
    with pytest.raises(ValueError, match="future labels"):
        AlphaDefinition("bad", "future_return + close", lookback=1)
    with pytest.raises(ValueError, match="cannot look forward"):
        AlphaDefinition("bad", "ts_delay(close, -1)", lookback=1)


def test_cross_section_requirement_is_explicit() -> None:
    definition = AlphaDefinition("rank_close", "cs_rank(close)", lookback=1)
    assert definition.uses_cross_section
    assert AlphaEngine([definition]).requires_cross_section


def test_realtime_service_uses_the_same_alpha_definition() -> None:
    definition = AlphaDefinition("mean_2", "ts_mean(close, 2)", lookback=2)
    service = RealtimeAlphaService(
        BarCache(), AlphaSampleCache(), [definition], frequency="1d"
    )
    start = datetime(2025, 1, 1)

    first = SampleBar("A", "1d", start, 10, 10, 10, 10, 100)
    second = SampleBar("A", "1d", start + timedelta(days=1), 12, 12, 12, 12, 100)

    assert service.on_bar(first) is None
    sample = service.on_bar(second)
    assert sample is not None
    assert sample.features == {"mean_2": pytest.approx(11.0)}
