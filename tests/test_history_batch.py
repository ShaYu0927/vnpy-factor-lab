from datetime import datetime
import json
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal
import pytest

from vnpy import main
from vnpy.alpha import Alpha, AlphaEngine
from vnpy.config.runtime_config import load_runtime_config
from vnpy.datafeed.data_parquet_feed import ParquetDataFeed
from vnpy.factor.history_batch import parse_history_batch_config
from vnpy.datafeed.data_market_module import unregister_market_module
from vnpy.event.engine import ModuleEngine


@pytest.fixture(autouse=True)
def isolated_modules(monkeypatch):
    engine = ModuleEngine()
    monkeypatch.setattr(main, "module_engine", engine)
    yield
    unregister_market_module(engine)


def source(tmp_path):
    for year, dates in ((2025, ("2025-12-30", "2025-12-31")),
                        (2026, ("2026-01-01", "2026-01-02", "2026-01-03"))):
        for market in ("SHSE", "SZSE"):
            folder = tmp_path / f"{market}_{year}"
            folder.mkdir()
            rows = [
                dict(trade_date=day, symbol=f"{market}.{symbol}", open=10.0 + i + symbol,
                     high=14.0 + i + symbol, low=9.0, close=11.0 + i + symbol,
                     volume=0 if i == 1 and symbol == 1 else 100, amount=1100.0,
                     industry="bank", ext_data="unused metadata")
                for i, day in enumerate(dates) for symbol in range(3)
                if not (market == "SZSE" and symbol == 2 and i == 0)
            ]
            rows.append(dict(rows[0], symbol=f"{market}.invalid", high=1.0))
            rows.append(dict(rows[0], symbol=f"{market}.null", close=None))
            pl.DataFrame(rows[::-1]).write_parquet(folder / "dists_day_bar.parquet")
    return ParquetDataFeed(tmp_path)


@pytest.mark.parametrize("filtered", [False, True])
@pytest.mark.parametrize("symbols", [None, "szse.0, shse.1"])
def test_columnar_read_matches_event_reader_across_years(tmp_path, filtered, symbols):
    feed = source(tmp_path)
    options = dict(start="2025-12-31", end="2026-01-02", symbols=symbols,
                   skip_zero_volume=filtered, skip_invalid_ohlc=filtered)
    grouped = {}
    for bar in feed.iter_history(**options):
        grouped.setdefault(bar.symbol, []).append(bar)
    expected = AlphaEngine([]).from_bars(grouped)
    actual = feed.load_frame(**options)
    assert "ext_data" not in actual.columns
    assert_frame_equal(actual.select(expected.columns), expected, check_dtypes=False)


def test_parallel_matches_complete_plan_with_missing_bars_and_nested_cross_sections(tmp_path):
    market = source(tmp_path).load_frame(start="2025-12-30", end="2026-01-03")
    engine = AlphaEngine([
        Alpha("return_1", "close / Ref(close, 1) - 1"),
        Alpha("mean", "Mean(close, 3)"),
        Alpha("rank", "Rank(close)"),
        Alpha("nested", "Mean(Rank(close), 3)"),
        Alpha("rank_mean", "Rank(Mean(close, 3))"),
        Alpha("invalid", "close / 0"),
    ])
    expected = engine.calculate(market)
    for workers in (1, 2, 4, 30):
        assert_frame_equal(engine.calculate_parallel(market, workers=workers), expected)
    # The first 2026 observation must retain the previous year's history.
    value = expected.filter((pl.col("vt_symbol") == "SHSE.0") &
                            (pl.col("datetime").dt.date() == datetime(2026, 1, 1).date()))["return_1"].item()
    assert value == pytest.approx(11 / 12 - 1)
    with pytest.raises(ValueError, match="duplicate"):
        engine.calculate_parallel(pl.concat([market, market.head(1)]), workers=2)
    with pytest.raises(ValueError, match="missing columns"):
        AlphaEngine([Alpha("a", "close"), Alpha("b", "vwap")]).calculate_parallel(market, workers=2)


def test_empty_and_invalid_source_requests(tmp_path):
    feed = source(tmp_path)
    empty = feed.load_frame(start="2026-02-01", end="2026-02-02")
    assert empty.is_empty() and "close" in empty.columns
    assert AlphaEngine([Alpha("a", "close")]).calculate_parallel(empty).is_empty()
    with pytest.raises(FileNotFoundError, match="2024"):
        feed.load_frame(start="2024-01-01", end="2026-01-01")
    assert not feed.load_frame(start="2024-01-01", end="2026-01-01", allow_missing_years=True).is_empty()
    with pytest.raises(ValueError, match="start"):
        feed.load_frame(start="2026-02-01", end="2026-01-01")
    with pytest.raises(ValueError, match="frequency"):
        feed.load_frame(start="2026-01-01", end="2026-01-02", frequency="1m")
    with pytest.raises(ValueError, match="not enabled"):
        feed.load_frame(start="2026-01-01", end="2026-01-02", symbols="SZSE.0", markets="SHSE")


@pytest.mark.parametrize("raw", [{"enabled": "yes"}, {"workers": True}, {"workers": 0},
                                  {"workers": 1.5}, {"output_root": ""}, {"oops": 1}, []])
def test_invalid_batch_configuration(raw, tmp_path):
    with pytest.raises(ValueError):
        parse_history_batch_config(raw, tmp_path)


def test_main_batch_reads_and_calculates_once_reuses_result_for_evaluation(tmp_path, monkeypatch):
    feed = source(tmp_path)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    raw = {
        "mode": "parquet", "parquet": {"root": str(tmp_path), "start": "2025-12-30", "end": "2026-01-03"},
        "alphas": [{"name": "rank", "formula": "Rank(close)"}],
        "expression_iteration": {"enabled": True, "batches": 1, "batch_size": 1,
                                 "output_root": "../generated", "search_space": {"fields": ["volume"], "operators": []}},
        "history_batch": {"enabled": True, "workers": 2, "output_root": "../batch"},
        "factor_evaluation": {"enabled": True},
    }
    path = config_dir / "runtime.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.setattr(main, "run_parquet_replay", lambda *a, **k: pytest.fail("unexpected replay"))
    monkeypatch.setattr(ParquetDataFeed, "iter_history", lambda *a, **k: pytest.fail("per-row read"))
    from vnpy.alpha.modeling import runtime_evaluation
    observed = []
    def evaluate(market, factors, definitions, options, **kwargs):
        observed.append((market, factors, definitions))
        return {"reused": True}
    monkeypatch.setattr(runtime_evaluation, "evaluate_frames", evaluate)
    monkeypatch.chdir(tmp_path.parent)
    result = main.run_from_config(load_runtime_config(path))
    output = Path(result["output"])
    assert output.parent == tmp_path / "batch"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete" and manifest["factors"] == 2
    assert result["evaluation"] == {"reused": True}
    assert len(observed) == 1
    assert_frame_equal(pl.read_parquet(output / "factors.parquet"), observed[0][1])
    assert_frame_equal(pl.read_parquet(output / "market.parquet"), observed[0][0])
    expected = AlphaEngine([Alpha(**definition) for definition in observed[0][2]]).calculate(
        feed.load_frame(start="2025-12-30", end="2026-01-03", skip_zero_volume=True, skip_invalid_ohlc=True))
    assert_frame_equal(observed[0][1], expected)
    raw["parquet_import"] = {"enabled": True}
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot both"):
        load_runtime_config(path)
