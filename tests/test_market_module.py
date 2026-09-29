from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, current_thread

import polars as pl
from polars.testing import assert_frame_equal
import pytest

from vnpy import main
from vnpy.alpha import Alpha, AlphaEngine
from vnpy.config.runtime_config import ParquetConfig, RunMode, RuntimeConfig
from vnpy.datafeed.data_market_module import (
    clear_market_data, get_market_store, load_market_snapshot, unregister_market_module,
)
from vnpy.datafeed.data_parquet_feed import ParquetDataFeed
from vnpy.event.engine import ModuleEngine


@pytest.fixture
def application(tmp_path, monkeypatch):
    folder = tmp_path / "SHSE_2026"
    folder.mkdir()
    pl.DataFrame([
        dict(symbol=symbol, trade_date=f"2026-01-0{day}", open=10.0 + day,
             high=12.0 + day, low=9.0, close=11.0 + day, volume=100.0, amount=1000.0)
        for symbol in ("SHSE.A", "SHSE.B") for day in (1, 2, 3)
    ]).write_parquet(folder / "dists_day_bar.parquet")
    engine = ModuleEngine()
    monkeypatch.setattr(main, "module_engine", engine)
    config = RuntimeConfig(
        RunMode.PARQUET,
        ParquetConfig(str(tmp_path), "2026-01-01", "2026-01-03", markets="SHSE"),
        raw={"alphas": [{"name": "mean", "formula": "Mean(close, 2)"}],
             "history_batch": {"enabled": True, "output_root": str(tmp_path / "results")}},
        config_dir=tmp_path,
    )
    yield engine, config
    unregister_market_module(engine)


def test_market_survives_rounds_and_changed_formulas_do_not_reread(application, monkeypatch):
    engine, config = application
    calls = []
    original = ParquetDataFeed.load_frame
    def observe(self, **kwargs):
        calls.append(current_thread().name)
        return original(self, **kwargs)
    monkeypatch.setattr(ParquetDataFeed, "load_frame", observe)
    first = main.run_from_config(config)
    store = get_market_store(engine)
    assert store.snapshot().frame.height == 6
    assert engine.is_module_started("market")
    revised = replace(config, raw={**config.raw, "alphas": [{"name": "rank", "formula": "Rank(close)"}]})
    second = main.run_from_config(revised)
    assert calls == ["Module-market"]
    assert first["market_snapshot_id"] == second["market_snapshot_id"]
    assert not first["market_reused"] and second["market_reused"]
    assert engine.get_context("market").get_state("load_count") == 1
    assert engine.get_context("market").get_state("reuse_count") == 1
    expected = AlphaEngine([Alpha("rank", "Rank(close)")]).calculate(store.snapshot().frame)
    assert_frame_equal(pl.read_parquet(second["factors_path"]), expected)
    assert first["output"] != second["output"]


def test_snapshot_mutations_are_isolated_and_refresh_keeps_borrowed_data_valid(application):
    engine, config = application
    first = load_market_snapshot(engine, config.parquet)
    store = get_market_store(engine)
    caller = store.snapshot()
    caller.frame.replace_column(caller.frame.get_column_index("close"), pl.Series("close", [999.0] * 6))
    assert store.snapshot().frame["close"].max() == 14
    path = config.config_dir / "SHSE_2026/dists_day_bar.parquet"
    pl.read_parquet(path).with_columns((pl.col("close") + 0.5).alias("close")).write_parquet(path)
    reused = load_market_snapshot(engine, config.parquet)
    assert reused.reused and reused.snapshot.snapshot_id == first.snapshot.snapshot_id
    fresh = load_market_snapshot(engine, config.parquet, reload=True)
    assert not fresh.reused and fresh.snapshot.snapshot_id != first.snapshot.snapshot_id
    assert fresh.snapshot.frame["close"].max() == 14.5
    assert first.snapshot.frame["close"].max() == 14


def test_load_failure_preserves_previous_snapshot_and_can_retry(application, monkeypatch):
    engine, config = application
    first = load_market_snapshot(engine, config.parquet)
    original = ParquetDataFeed.load_frame
    def fail(*args, **kwargs):
        raise OSError("read failed")
    monkeypatch.setattr(ParquetDataFeed, "load_frame", fail)
    with pytest.raises(OSError, match="read failed"):
        load_market_snapshot(engine, config.parquet, reload=True)
    assert get_market_store(engine).snapshot().snapshot_id == first.snapshot.snapshot_id
    assert engine.get_context("market").get_state("status") == "failed"
    monkeypatch.setattr(ParquetDataFeed, "load_frame", original)
    assert load_market_snapshot(engine, config.parquet, reload=True).snapshot.frame.height == 6
    assert engine.get_context("market").get_state("status") == "ready"


def test_selection_changes_reload_but_processing_options_do_not(application):
    engine, config = application
    first = load_market_snapshot(engine, config.parquet)
    assert load_market_snapshot(engine, replace(config.parquet, batch_size=1)).reused
    subset = load_market_snapshot(engine, replace(config.parquet, symbols="SHSE.A", end="2026-01-02"))
    assert not subset.reused and subset.snapshot.frame.height == 2
    assert subset.snapshot.snapshot_id != first.snapshot.snapshot_id
    assert first.snapshot.frame.height == 6


def test_clear_and_unregister_release_module_owned_data(application):
    engine, config = application
    first = load_market_snapshot(engine, config.parquet)
    store = get_market_store(engine)
    clear_market_data(engine)
    with pytest.raises(RuntimeError, match="not been loaded"):
        store.snapshot()
    assert engine.module_exists("market")
    second = load_market_snapshot(engine, config.parquet)
    assert second.snapshot.snapshot_id != first.snapshot.snapshot_id
    unregister_market_module(engine)
    assert not engine.module_exists("market")
    with pytest.raises(RuntimeError, match="not been loaded"):
        store.snapshot()
    # Outstanding readers own their references until their own work finishes.
    assert first.snapshot.frame.height == second.snapshot.frame.height == 6


def test_factor_failure_does_not_discard_market(application):
    engine, config = application
    invalid = replace(config, raw={**config.raw, "alphas": [{"name": "missing", "formula": "vwap"}]})
    with pytest.raises(ValueError, match="missing columns"):
        main.run_from_config(invalid)
    snapshot_id = get_market_store(engine).snapshot().snapshot_id
    result = main.run_from_config(config)
    assert result["market_reused"] and result["market_snapshot_id"] == snapshot_id


def test_rejected_queue_is_reported_without_waiting(application, monkeypatch):
    engine, config = application
    monkeypatch.setattr(engine, "post_event", lambda *args, **kwargs: False)
    with pytest.raises(RuntimeError, match="rejected"):
        load_market_snapshot(engine, config.parquet)


def test_application_exit_unregisters_market(application, monkeypatch):
    engine, config = application
    monkeypatch.setattr(main, "load_runtime_config", lambda _: config)
    monkeypatch.setattr(main, "init_logger", lambda: None)
    monkeypatch.setattr(main, "shutdown_global_logger", lambda: None)
    main.main("unused")
    assert not engine.module_exists("market")


def test_bad_keys_are_rejected_before_publishing_a_snapshot(application):
    engine, config = application
    path = config.config_dir / "SHSE_2026/dists_day_bar.parquet"
    frame = pl.read_parquet(path)
    pl.concat([frame, frame.head(1)]).write_parquet(path)
    with pytest.raises(ValueError, match="duplicate"):
        load_market_snapshot(engine, config.parquet)
    with pytest.raises(RuntimeError, match="not been loaded"):
        get_market_store(engine).snapshot()


def test_concurrent_consumers_share_one_module_and_one_load(application, monkeypatch):
    engine, config = application
    barrier = Barrier(4)
    calls = []
    original = ParquetDataFeed.load_frame
    def observe(self, **kwargs):
        calls.append(current_thread().name)
        return original(self, **kwargs)
    monkeypatch.setattr(ParquetDataFeed, "load_frame", observe)
    def request(_):
        barrier.wait(timeout=5)
        return load_market_snapshot(engine, config.parquet)
    with ThreadPoolExecutor(max_workers=4) as consumers:
        results = list(consumers.map(request, range(4)))
    assert calls == ["Module-market"]
    assert len({result.snapshot.snapshot_id for result in results}) == 1
    assert sum(result.reused for result in results) == 3
