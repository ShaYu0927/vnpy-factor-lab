import csv
from datetime import date, datetime, timedelta
from pathlib import Path
import sqlite3

import pytest

from vnpy.config.runtime_config import GmSqliteBatchConfig
from vnpy.datafeed.model import MarketBar
from vnpy.factor.batch_runner import (
    iter_bars_by_date,
    read_last_trade_date,
    run_gm_sqlite_batch,
)


def make_bar(symbol: str, day: int) -> MarketBar:
    return MarketBar(
        symbol=symbol,
        bob=datetime(2025, 1, day),
        open=10,
        high=12,
        low=9,
        close=10 + day / 10,
        volume=100 + day,
        frequency="1d",
    )


def create_day_bar_database(path: Path, days: int = 23) -> None:
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE dists_day_bar (
            symbol TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            sec_id INTEGER NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL,
            volume INTEGER NOT NULL,
            amount REAL NOT NULL,
            pre_close REAL NOT NULL,
            Position INTEGER NOT NULL,
            updated_at TEXT NOT NULL,
            ext_data TEXT,
            PRIMARY KEY (symbol, trade_date)
        )
        """
    )

    start = date(2025, 1, 1)
    rows = []
    for offset in range(days):
        trade_date = start + timedelta(days=offset)
        for index, symbol in enumerate(("SHSE.600000", "SHSE.600001")):
            close = 10 + offset / 10 + index
            rows.append(
                (
                    symbol,
                    trade_date.isoformat(),
                    index + 1,
                    close - 0.1,
                    close + 0.2,
                    close - 0.2,
                    close,
                    1000 + offset,
                    10_000,
                    close - 0.1,
                    0,
                    f"{trade_date.isoformat()}T16:00:00+08:00",
                    None,
                )
            )

    connection.executemany(
        "INSERT INTO dists_day_bar VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    connection.commit()
    connection.close()


def test_iter_bars_by_date_preserves_groups_across_batches() -> None:
    batches = [
        [make_bar("SHSE.600000", 1)],
        [make_bar("SHSE.600001", 1), make_bar("SHSE.600000", 2)],
    ]

    groups = list(iter_bars_by_date(batches))

    assert [item[0].isoformat() for item in groups] == [
        "2025-01-01",
        "2025-01-02",
    ]
    assert [len(item[1]) for item in groups] == [2, 1]


def test_run_gm_sqlite_batch_writes_factor_snapshots(tmp_path) -> None:
    day_bar = tmp_path / "basic_data" / "day_bar"
    day_bar.mkdir(parents=True)
    create_day_bar_database(day_bar / "SHSE_2025.dat")
    output = tmp_path / "result" / "factors.csv"

    summary = run_gm_sqlite_batch(
        GmSqliteBatchConfig(
            root=str(tmp_path),
            start="2025-01-01",
            end="2025-01-23",
            output=str(output),
            markets="SHSE",
            batch_size=3,
            window_size=21,
            factors=["momentum_20", "volatility_20", "volume_20"],
            factor_mode="sync",
            max_workers=1,
            progress_every_dates=100,
        )
    )

    with output.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert summary.bars_read == 46
    assert summary.dates_processed == 23
    assert summary.symbols_seen == 2
    assert summary.values_written == 18
    assert len(rows) == 18
    assert {row["factor_name"] for row in rows} == {
        "momentum_20",
        "volatility_20",
        "volume_20",
    }
    assert {row["trade_date"] for row in rows} == {
        "2025-01-21",
        "2025-01-22",
        "2025-01-23",
    }


def test_run_gm_sqlite_batch_rejects_small_window(tmp_path) -> None:
    with pytest.raises(ValueError, match="at least 21"):
        run_gm_sqlite_batch(
            GmSqliteBatchConfig(
                root=str(tmp_path),
                start="2025-01-01",
                end="2025-12-31",
                output=str(tmp_path / "factors.csv"),
                window_size=20,
            )
        )


def test_run_gm_sqlite_batch_resumes_without_duplicate_rows(tmp_path) -> None:
    day_bar = tmp_path / "basic_data" / "day_bar"
    day_bar.mkdir(parents=True)
    create_day_bar_database(day_bar / "SHSE_2025.dat", days=23)
    output = tmp_path / "factors.csv"
    common = dict(
        root=str(tmp_path),
        start="2025-01-01",
        output=str(output),
        markets="SHSE",
        batch_size=3,
        window_size=21,
        factors=["momentum_20", "volatility_20", "volume_20"],
        factor_mode="sync",
        max_workers=1,
        progress_every_dates=100,
    )

    run_gm_sqlite_batch(
        GmSqliteBatchConfig(end="2025-01-22", overwrite=True, **common)
    )
    assert read_last_trade_date(output) == date(2025, 1, 22)

    summary = run_gm_sqlite_batch(
        GmSqliteBatchConfig(end="2025-01-23", resume=True, **common)
    )

    with output.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 18
    assert summary.values_written == 6
    assert [row["trade_date"] for row in rows].count("2025-01-22") == 6
    assert [row["trade_date"] for row in rows].count("2025-01-23") == 6


def test_run_gm_sqlite_batch_trains_model_and_writes_latest_signals(tmp_path) -> None:
    day_bar = tmp_path / "basic_data" / "day_bar"
    day_bar.mkdir(parents=True)
    create_day_bar_database(day_bar / "SHSE_2025.dat", days=45)
    signal_output = tmp_path / "latest_signals.csv"
    model_output = tmp_path / "linear_model.pkl"

    summary = run_gm_sqlite_batch(
        GmSqliteBatchConfig(
            root=str(tmp_path),
            start="2025-01-01",
            end="2025-02-14",
            output="console",
            markets="SHSE",
            window_size=21,
            factors=["momentum_20", "volatility_20", "volume_20"],
            factor_mode="sync",
            max_workers=1,
            progress_every_dates=100,
            train_model=True,
            label_horizon=5,
            model_output=str(model_output),
            signal_output=str(signal_output),
        )
    )

    with signal_output.open(encoding="utf-8", newline="") as file:
        signals = list(csv.DictReader(file))

    assert summary.model_training_samples > 0
    assert summary.model_test_samples > 0
    assert model_output.is_file()
    assert len(signals) == 2
    assert [row["rank"] for row in signals] == ["1", "2"]
