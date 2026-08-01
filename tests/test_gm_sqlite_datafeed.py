import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from vnpy.config.runtime_config import RunMode, load_runtime_config
from vnpy.datafeed.gm_sqlite_datafeed import GmSqliteDataFeed
from vnpy.datafeed.model import BarSource


def create_day_bar(path, rows) -> None:
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
    connection.executemany(
        "INSERT INTO dists_day_bar VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    connection.commit()
    connection.close()


def make_row(symbol, trade_date, close, volume=100, high=12, low=9):
    return (
        symbol,
        trade_date,
        1,
        10,
        high,
        low,
        close,
        volume,
        1_000,
        10,
        0,
        f"{trade_date}T16:00:00+08:00",
        None,
    )


class TestGmSqliteDataFeed(unittest.TestCase):
    def test_streams_multiple_markets_in_global_date_order(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            day_bar = root / "basic_data" / "day_bar"
            day_bar.mkdir(parents=True)
            create_day_bar(
                day_bar / "SHSE_2025.dat",
                [
                    make_row("SHSE.600000", "2025-01-03", 11),
                    make_row("SHSE.600000", "2025-01-02", 10.5),
                ],
            )
            create_day_bar(
                day_bar / "SZSE_2025.dat",
                [make_row("SZSE.000001", "2025-01-02", 12)],
            )

            bars = list(
                GmSqliteDataFeed(root).iter_history(
                    start="2025-01-01",
                    end="2025-12-31",
                )
            )

            self.assertEqual(
                [(bar.bob.date().isoformat(), bar.symbol) for bar in bars],
                [
                    ("2025-01-02", "SHSE.600000"),
                    ("2025-01-02", "SZSE.000001"),
                    ("2025-01-03", "SHSE.600000"),
                ],
            )
            self.assertEqual(bars[0].source, BarSource.GM_SQLITE.value)
            self.assertEqual(bars[0].extra["pre_close"], 10)

    def test_filters_zero_volume_and_invalid_ohlc(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            day_bar = root / "day_bar"
            day_bar.mkdir()
            create_day_bar(
                day_bar / "SHSE_2025.dat",
                [
                    make_row("SHSE.600000", "2025-01-02", 10.5),
                    make_row("SHSE.600001", "2025-01-02", 10.5, volume=0),
                    make_row("SHSE.600002", "2025-01-02", 13),
                ],
            )

            bars = list(
                GmSqliteDataFeed(root).iter_history(
                    start="2025-01-01",
                    end="2025-12-31",
                    markets="SHSE",
                    skip_zero_volume=True,
                    skip_invalid_ohlc=True,
                )
            )

            self.assertEqual([bar.symbol for bar in bars], ["SHSE.600000"])

    def test_missing_years_are_rejected_by_default(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            day_bar = root / "day_bar"
            day_bar.mkdir()
            create_day_bar(
                day_bar / "SHSE_2025.dat",
                [make_row("SHSE.600000", "2025-01-02", 10.5)],
            )

            feed = GmSqliteDataFeed(root)
            with self.assertRaisesRegex(FileNotFoundError, "SHSE_2024"):
                list(
                    feed.iter_history(
                        start="2024-01-01",
                        end="2025-12-31",
                        markets="SHSE",
                    )
                )

    def test_iter_batches_keeps_memory_unit_bounded(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            day_bar = root / "day_bar"
            day_bar.mkdir()
            create_day_bar(
                day_bar / "SHSE_2025.dat",
                [
                    make_row("SHSE.600000", "2025-01-02", 10),
                    make_row("SHSE.600000", "2025-01-03", 11),
                    make_row("SHSE.600000", "2025-01-06", 12),
                ],
            )

            batches = list(
                GmSqliteDataFeed(root).iter_batches(
                    batch_size=2,
                    start="2025-01-01",
                    end="2025-12-31",
                    markets="SHSE",
                )
            )

            self.assertEqual([len(batch) for batch in batches], [2, 1])

    def test_runtime_config_loads_gm_sqlite_mode(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.json"
            path.write_text(
                json.dumps(
                    {
                        "mode": "gm_sqlite",
                        "gm_sqlite": {
                            "root": "F:/Quantitative",
                            "start": "2025-01-01",
                            "end": "2025-12-31",
                        },
                    }
                ),
                encoding="utf-8",
            )

            config = load_runtime_config(path)
            self.assertEqual(config.mode, RunMode.GM_SQLITE)
            self.assertIsNotNone(config.gm_sqlite)
            self.assertEqual(config.gm_sqlite.root, "F:/Quantitative")


if __name__ == "__main__":
    unittest.main()
