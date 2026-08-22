import sqlite3
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from vnpy.alpha.modeling import AlphalensEvaluator


def make_alphalens_frames() -> tuple[pl.DataFrame, pl.DataFrame]:
    factor_rows: list[dict[str, object]] = []
    price_rows: list[dict[str, object]] = []
    start = datetime(2025, 1, 1)
    symbols = [f"STOCK{index:02d}.SSE" for index in range(10)]
    for day in range(35):
        current = start + timedelta(days=day)
        for symbol_index, symbol in enumerate(symbols):
            price_rows.append({
                "datetime": current,
                "vt_symbol": symbol,
                "close": 100.0 + day * (0.1 + symbol_index * 0.02),
            })
            if day < 25:
                factor_rows.append({
                    "datetime": current,
                    "vt_symbol": symbol,
                    "factor": float(symbol_index) + day * 0.001,
                })
    return pl.DataFrame(factor_rows), pl.DataFrame(price_rows)


def test_alphalens_evaluator_returns_structured_metrics() -> None:
    factors, prices = make_alphalens_frames()
    evaluator = AlphalensEvaluator(periods=(1, 5), quantiles=5, max_loss=0.5)

    report = evaluator.evaluate(factors, prices)

    assert {"1D", "5D", "factor", "factor_quantile"}.issubset(report.clean_data.columns)
    assert list(report.information_coefficient.columns) == ["1D", "5D"]
    assert np.isfinite(report.mean_information_coefficient.to_numpy()).all()
    assert set(report.quantile_returns.index) == {1, 2, 3, 4, 5}
    assert set(report.turnover) == {
        (period, quantile)
        for period in (1, 5)
        for quantile in range(1, 6)
    }


def test_alphalens_adapter_rejects_duplicate_factor_rows() -> None:
    factors, _ = make_alphalens_frames()
    duplicated = pl.concat([factors, factors.head(1)])

    with pytest.raises(ValueError, match="duplicate"):
        AlphalensEvaluator.prepare_factor(duplicated)


def test_alphalens_adapter_accepts_wide_prices() -> None:
    _, prices = make_alphalens_frames()
    wide = AlphalensEvaluator.prepare_prices(prices)

    assert wide.index.name == "date"
    assert wide.columns.name == "asset"
    assert wide.shape == (35, 10)


def test_alphalens_adapter_reads_batch_csv(tmp_path) -> None:
    path = tmp_path / "factors.csv"
    path.write_text(
        "trade_date,symbol,factor_name,value,primary_field,status,reason,version,fields_json\n"
        "2025-01-01,STOCK01.SSE,momentum_20,0.5,momentum,ready,,1,{}\n"
        "2025-01-01,STOCK02.SSE,momentum_20,,momentum,invalid,bad,1,{}\n",
        encoding="utf-8",
    )

    factor = AlphalensEvaluator.factor_from_batch_csv(path, "momentum_20")

    assert factor.loc[(datetime(2025, 1, 1), "STOCK01.SSE")] == 0.5
    assert len(factor) == 1


def test_alphalens_adapter_reads_batch_sqlite(tmp_path) -> None:
    path = tmp_path / "factors.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE factor_values ("
            "trade_date TEXT, symbol TEXT, factor_name TEXT, value REAL, status TEXT)"
        )
        connection.execute(
            "INSERT INTO factor_values VALUES (?, ?, ?, ?, ?)",
            ("2025-01-01", "STOCK01.SSE", "momentum_20", 0.5, "ready"),
        )

    factor = AlphalensEvaluator.factor_from_batch_sqlite(path, "momentum_20")

    assert factor.loc[(datetime(2025, 1, 1), "STOCK01.SSE")] == 0.5


def test_alphalens_adapter_builds_prices_from_bars() -> None:
    bars = [
        {"datetime": datetime(2025, 1, 1), "symbol": "A", "close": 10.0},
        {"datetime": datetime(2025, 1, 1), "symbol": "B", "close": 20.0},
    ]

    prices = AlphalensEvaluator.prices_from_bars(bars)

    assert prices.loc[datetime(2025, 1, 1), "A"] == 10.0
    assert prices.loc[datetime(2025, 1, 1), "B"] == 20.0


def test_alphalens_group_adjust_requires_groups() -> None:
    factors, prices = make_alphalens_frames()
    evaluator = AlphalensEvaluator(group_adjust=True)

    with pytest.raises(ValueError, match="groups are required"):
        evaluator.evaluate(factors, prices)
