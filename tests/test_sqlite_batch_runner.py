from datetime import datetime
from types import SimpleNamespace

import polars as pl
import pytest

from vnpy.datafeed.model import MarketBar
from vnpy.factor.sqlite_batch_runner import run_sqlite_alpha101


class Feed:
    def iter_history(self, **kwargs):
        for day in (1, 2):
            for symbol in ("A", "B"):
                yield MarketBar(symbol, datetime(2025, 1, day), 10, 12, 9, 11, 1000, frequency="1d")


def test_sqlite_batch_runner_uses_latest_real_source_date(tmp_path):
    config = SimpleNamespace(symbols="A,B", root="unused", start="2025-01-01", end="2025-01-04",
                             frequency="1d", markets="SHSE", skip_zero_volume=True,
                             skip_invalid_ohlc=True, allow_missing_years=False, progress_every=1000)
    result = run_sqlite_alpha101(config, {"root": str(tmp_path), "factors": [101]}, feed=Feed())
    assert result["trade_date"] == "2025-01-02"
    frame = pl.read_parquet(result["result_path"])
    assert frame.height == 2
    assert frame["alpha101"].to_list() == pytest.approx([1 / 3.001] * 2)
    assert result["factor_count"] == 1
