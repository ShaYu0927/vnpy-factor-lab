from datetime import datetime, timedelta

import polars as pl
from polars.testing import assert_frame_equal
import pytest

from vnpy import main
from vnpy.alpha import Alpha, AlphaEngine
from vnpy.alpha.engine import AlphaSampleCache
from vnpy.alpha.expression import PolarsCompiler, PolarsExecutor
from vnpy.config.runtime_config import ParquetConfig
from vnpy.datafeed.data_bar_cache import BarCache
from vnpy.datafeed.data_model import MarketBar
from vnpy.event.engine import ModuleEngine
from vnpy.event.event import EventType
from vnpy.factor.realtime_service import RealtimeAlphaService
from vnpy.strategy.strategy_engine import StrategyEngine


def test_combined_plan_matches_independent_expressions():
    frame = pl.DataFrame([
        {"datetime": datetime(2026, 1, 1) + timedelta(days=i), "vt_symbol": symbol,
         "close": base + i % 3, "volume": float(i * 2)}
        for symbol, base in (("B", 20), ("A", 10)) for i in reversed(range(12))
    ])
    definitions = [Alpha(name, formula) for name, formula in (
        ("bias", "close / Mean(close, 3) - 1"),
        ("mean", "Mean(close, 3)"),
        ("nested", "Std(close / Ref(close, 1) - 1, 3)"),
        ("rank", "Rank(Mean(close, 3))"),
        ("rank_history", "Mean(Rank(close), 3)"),
        ("zero_div", "close / 0"),
        ("__alpha_stage_0", "volume + 1"),
    )]
    actual = AlphaEngine(definitions).calculate(frame)
    executor = PolarsExecutor(frame)
    for definition in definitions:
        expected = executor.run(PolarsCompiler().compile(definition.parse()), definition.name)
        assert_frame_equal(actual.select("datetime", "vt_symbol", definition.name), expected)


@pytest.mark.parametrize("cross_section", [False, True])
def test_batch_matches_per_bar_with_missing_bars_and_optional_factors(cross_section):
    definitions = [Alpha("fixed", "close / Ref(close, 1)"), Alpha("optional", "Mean(close, 4)")]
    if cross_section:
        definitions.append(Alpha("rank", "Rank(close)"))
    def service():
        return RealtimeAlphaService(BarCache(), AlphaSampleCache(), definitions,
                                    frequency="1d", universe=("A", "B"), optional_names=("optional",))
    single, batch = service(), service()
    for i in range(8):
        bars = [MarketBar(symbol, datetime(2026, 1, 1) + timedelta(days=i),
                          base + i, base + i, base + i, base + i, frequency="1d")
                for symbol, base in (("A", 10), ("B", 20)) if not (i == 4 and symbol == "B")]
        expected = []
        for bar in bars:
            single.on_bar(bar)
            expected.extend(single.latest_samples)
        actual = batch.on_bars(bars)
        assert [(s.symbol, s.datetime) for s in actual] == [(s.symbol, s.datetime) for s in expected]
        for left, right in zip(actual, expected):
            assert left.features == pytest.approx(right.features)


def test_invalid_batch_does_not_partially_update_cache():
    service = RealtimeAlphaService(BarCache(), AlphaSampleCache(), [Alpha("fixed", "close")], frequency="1d")
    a = MarketBar("A", datetime(2026, 1, 1), 10, 10, 10, 10, frequency="1d")
    b = MarketBar("B", datetime(2026, 1, 2), 10, 10, 10, 10, frequency="1d")
    with pytest.raises(ValueError, match="single timestamp"):
        service.on_bars([a, b])
    with pytest.raises(ValueError, match="unique symbols"):
        service.on_bars([a, a])
    assert not service.bar_cache.symbols()


def replay_config(tmp_path, count=100):
    folder = tmp_path / "SHSE_2026"
    folder.mkdir()
    symbols = [f"SHSE.{600000+i}" for i in range(count)]
    pl.DataFrame([
        {"trade_date": f"2026-01-{day:02d}", "symbol": symbol,
         "open": 10.0 + i + day, "high": 12.0 + i + day, "low": 9.0 + i + day,
         "close": 11.0 + i + day, "volume": 100.0, "amount": 1100.0}
        for day in range(1, 9) for i, symbol in enumerate(symbols)
    ]).write_parquet(folder / "dists_day_bar.parquet")
    return ParquetConfig(str(tmp_path), "2026-01-01", "2026-01-08",
                         symbols=",".join(symbols), markets="SHSE", batch_size=33)


def test_100_stock_replay_delivers_ordered_isolated_results_and_final_partial_batch(tmp_path, monkeypatch):
    config = replay_config(tmp_path)
    monkeypatch.setattr(main, "module_engine", ModuleEngine())
    received = []
    original = StrategyEngine.on_event
    def observe(self, event):
        original(self, event)
        if event.event_type == EventType.FACTOR:
            sample = event.get("sample")
            assert {v.symbol for v in event.get("factor_result").values} == {sample.symbol}
            assert event.get("factor_result").values[0].value == sample.features["mean"]
            received.append(sample)
    monkeypatch.setattr(StrategyEngine, "on_event", observe)
    summary = main.run_parquet_replay(config, alphas=[{"name": "mean", "formula": "Mean(close, 3)"}])
    assert summary["bars"] == 800 and summary["symbols"] == 100
    assert summary["batches"] == 32
    assert summary["factor_samples"] == summary["strategy_samples"] == len(received) == 600
    assert len({(s.symbol, s.datetime) for s in received}) == 600
    assert [s.datetime for s in received] == sorted(s.datetime for s in received)
    service = main.module_engine.get_context("factor").get_object("factor_service")
    assert all(service.bar_cache.size(symbol, "1d") == 3 for symbol in service.bar_cache.symbols())
    for sample in received:
        i = int(sample.symbol.split(".")[1]) - 600000
        assert sample.features["mean"] == pytest.approx(10 + i + sample.datetime.day)


def test_worker_failure_is_reported_to_replay_caller(tmp_path, monkeypatch):
    config = replay_config(tmp_path, count=2)
    monkeypatch.setattr(main, "module_engine", ModuleEngine())
    def fail(*args):
        raise ValueError("calculation failed")
    monkeypatch.setattr(RealtimeAlphaService, "on_bars", fail)
    with pytest.raises(RuntimeError, match="module factor failed"):
        main.run_parquet_replay(config, alphas=[{"name": "fixed", "formula": "close"}])
