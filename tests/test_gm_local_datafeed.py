from datetime import datetime

import pandas as pd
import pytest

from vnpy.datafeed.gm_local_datafeed import GmLocalDataFeed
from vnpy.datafeed.model import BarSource, MarketBar, normalize_bar


def make_feed(history_func, history_n_func=lambda **kwargs: []):
    return GmLocalDataFeed(
        history_func=history_func,
        history_n_func=history_n_func,
        set_token_func=lambda token: None,
    )


def test_load_history_converts_and_sorts_bars() -> None:
    captured = {}

    def history_func(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame(
            [
                {
                    "symbol": "SZSE.000001",
                    "bob": "2026-07-10 09:32:00",
                    "open": 11,
                    "high": 12,
                    "low": 10,
                    "close": 11.5,
                    "volume": 200,
                    "amount": 2300,
                },
                {
                    "symbol": "SZSE.000001",
                    "bob": "2026-07-10 09:31:00",
                    "open": 10,
                    "high": 11,
                    "low": 9,
                    "close": 10.5,
                    "volume": 100,
                    "amount": 1050,
                },
            ]
        )

    bars = make_feed(history_func).load_history(
        symbols=["SZSE.000001"],
        frequency="60s",
        start="2026-07-10 09:30:00",
        end="2026-07-10 15:00:00",
    )

    assert captured["symbol"] == "SZSE.000001"
    assert captured["df"] is True
    assert [bar.bob for bar in bars] == [
        datetime(2026, 7, 10, 9, 31),
        datetime(2026, 7, 10, 9, 32),
    ]
    assert bars[0].frequency == "60s"
    assert bars[0].source == BarSource.GM_LOCAL.value
    assert bars[0].close == 10.5


def test_load_recent_calls_history_n() -> None:
    captured = {}

    def history_n_func(**kwargs):
        captured.update(kwargs)
        return []

    bars = make_feed(lambda **kwargs: [], history_n_func).load_recent(
        symbols="SHSE.600519, SZSE.000001",
        frequency="1d",
        count=20,
        end="2026-07-12 15:00:00",
    )

    assert bars == []
    assert captured["symbol"] == "SHSE.600519,SZSE.000001"
    assert captured["count"] == 20


def test_load_recent_rejects_invalid_count() -> None:
    with pytest.raises(ValueError, match="count"):
        make_feed(lambda **kwargs: []).load_recent("SHSE.600519", "1d", 0)


def test_normalize_vnpy_style_bar() -> None:
    class VnpyBar:
        symbol = "600519"
        datetime = datetime(2026, 7, 10, 9, 31)
        open_price = 10
        high_price = 12
        low_price = 9
        close_price = 11
        volume = 100
        turnover = 1100

    bar = normalize_bar(
        VnpyBar(),
        frequency="60s",
        source=BarSource.VNPY,
    )

    assert isinstance(bar, MarketBar)
    assert bar.close == 11
    assert bar.close_price == 11
    assert bar.datetime == datetime(2026, 7, 10, 9, 31)
    assert bar.amount == 1100
