from datetime import date

import pandas as pd
import polars as pl
import pytest

from examples.alpha101_debug import demo_frame
from vnpy.alpha.alphas.alpha101 import Alpha101
from vnpy.datafeed.daily_store import DailyMarketStore
from vnpy.event.context import ModuleContext
from vnpy.event.event import EngineEvent, EventType
from vnpy.factor.batch_module import DailyAlphaBatchModule


class Sink:
    def __init__(self):
        self.events = []
        self.reject = False

    def post_event(self, target, event):
        if self.reject:
            return False
        self.events.append(event)
        return True


def setup(tmp_path):
    store = DailyMarketStore(tmp_path / "daily")
    frame = demo_frame().filter(pl.col("datetime") < pl.datetime(2025, 1, 16))
    store.upsert(frame)
    sink = Sink()
    ctx = ModuleContext("daily_alpha", sink)
    ctx.config.update(daily_store=str(store.root), result_store=str(tmp_path / "results"),
                      alpha101_factors=[3, 33, 56, 101], batch_targets=["consumer"])
    module = DailyAlphaBatchModule(ctx)
    request = EngineEvent(EventType.MARKET_DATA_READY,
                         {"trade_date": "2025-01-15", "expected_symbols": ["A", "B", "C"]})
    return store, frame, module, sink, request


def test_partition_upsert_and_bounded_loading(tmp_path):
    store, frame, *_ = setup(tmp_path)
    changed = frame.tail(1).with_columns(pl.lit(123.0).alias("close"))
    store.upsert(changed)
    recent = store.load_window("2025-01-15", 2)
    assert recent.height == 6
    assert recent.filter(pl.col("vt_symbol") == "C")["close"][-1] == 123
    assert store.load_window("2025-01-14", 2)["datetime"].max() == date(2025, 1, 14)


@pytest.mark.parametrize("suffix", ["csv", "parquet"])
def test_chunked_import(tmp_path, suffix):
    frame = demo_frame().head(7)
    source = tmp_path / f"source.{suffix}"
    getattr(frame, f"write_{suffix}")(source)
    store = DailyMarketStore(tmp_path / "daily")
    assert store.import_file(source, batch_size=2) == 7
    assert store.load_window("2025-01-07").height == 7


def test_batch_result_equivalence_and_small_event(tmp_path):
    store, frame, module, sink, request = setup(tmp_path)
    module.handle(request)
    event = sink.events[-1]
    assert event.event_type == EventType.FACTOR_BATCH_READY
    result = pl.read_parquet(event.get("result_path"))
    expected = Alpha101(frame.rename({"datetime": "date", "vt_symbol": "symbol"}).to_pandas()).compute_all([3, 33, 56, 101])
    expected = expected.xs(pd.Timestamp("2025-01-15"), level="date").sort_index()
    for name in expected.columns:
        assert result.sort("vt_symbol")[name].fill_null(float("nan")).to_list() == pytest.approx(expected[name].to_list(), nan_ok=True)
    assert "sample" not in event.data and "factor_result" not in event.data
    assert result.shape == (3, 7)
    module.handle(request)
    assert len(sink.events) == 1


def test_missing_data_failure_and_explicit_exclusion(tmp_path):
    _, _, module, sink, request = setup(tmp_path)
    request.data["expected_symbols"].append("SUSPENDED")
    module.handle(request)
    assert sink.events[-1].event_type == EventType.FACTOR_BATCH_FAILED
    assert "SUSPENDED" in sink.events[-1].get("error")
    request.data["excluded_symbols"] = {"SUSPENDED": "exchange suspension"}
    module.handle(request)
    assert sink.events[-1].event_type == EventType.FACTOR_BATCH_READY


def test_rejected_delivery_reuses_calculation(tmp_path, monkeypatch):
    _, _, module, sink, request = setup(tmp_path)
    sink.reject = True
    module.handle(request)
    monkeypatch.setattr(module, "calculate", lambda _: pytest.fail("must reuse saved result"))
    sink.reject = False
    module.handle(request)
    assert len(sink.events) == 1


def test_latest_only_all_factors_matches_full_history():
    frame = demo_frame().rename({"datetime": "date", "vt_symbol": "symbol"}).to_pandas()
    full = Alpha101(frame).compute_all()
    latest = Alpha101(frame).compute_all(at=frame["date"].max())
    pd.testing.assert_frame_equal(latest, full.loc[full.index.get_level_values("date") == frame["date"].max()])


def test_invalid_today_does_not_publish_success(tmp_path):
    store, frame, module, sink, request = setup(tmp_path)
    store.upsert(frame.tail(1).with_columns(pl.lit(float("nan")).alias("close")))
    module.handle(request)
    assert sink.events[-1].event_type == EventType.FACTOR_BATCH_FAILED
    assert "close" in sink.events[-1].get("error")


def test_future_rows_never_change_batch(tmp_path):
    store, frame, module, sink, request = setup(tmp_path)
    module.handle(request)
    before = pl.read_parquet(sink.events[-1].get("result_path"))
    store.upsert(demo_frame().filter(pl.col("datetime") > pl.datetime(2025, 1, 15)))
    module.handle(EngineEvent(EventType.MARKET_DATA_READY, dict(request.data)))
    after = pl.read_parquet(sink.events[-1].get("result_path"))
    assert before.equals(after)
