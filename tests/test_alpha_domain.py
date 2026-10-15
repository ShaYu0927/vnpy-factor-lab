"""The shared market, formula, factor and calendar contracts."""

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta

import polars as pl
import pytest

from vnpy.alpha.alpha_domain import (
    Alpha,
    FactorFrame,
    MarketFrame,
    SessionSplit,
    TimeSplitConfig,
)
from vnpy.alpha.alpha_engine import AlphaEngine


def market_rows() -> pl.DataFrame:
    first = datetime(2026, 1, 1)
    return pl.DataFrame(
        {
            "datetime": [
                first + timedelta(days=1),
                first,
                first,
                first + timedelta(days=1),
            ],
            "vt_symbol": ["B", "B", "A", "A"],
            "close": [21.0, 20.0, 10.0, 11.0],
        }
    )


def test_four_contracts_connect_market_calculation_and_split() -> None:
    market = MarketFrame(market_rows())
    alpha = Alpha("return_1", "close / Ref(close, 1) - 1")
    factors = AlphaEngine([alpha]).calculate_factors(market)
    split = market.split(TimeSplitConfig(0.5))

    assert isinstance(factors, FactorFrame)
    assert isinstance(split, SessionSplit)
    assert market.frame.select("vt_symbol", "datetime").to_dicts() == [
        {"vt_symbol": "A", "datetime": datetime(2026, 1, 1)},
        {"vt_symbol": "A", "datetime": datetime(2026, 1, 2)},
        {"vt_symbol": "B", "datetime": datetime(2026, 1, 1)},
        {"vt_symbol": "B", "datetime": datetime(2026, 1, 2)},
    ]
    assert factors.names == ("return_1",)
    values = factors.frame["return_1"].to_list()
    assert values[:2] == [None, None]
    assert values[2:] == pytest.approx([0.1, 0.05])
    assert split.holdout_start == datetime(2026, 1, 2)
    with pytest.raises(FrozenInstanceError):
        alpha.formula = "open"


def test_market_contract_rejects_bad_keys_and_price_type() -> None:
    source = market_rows()
    with pytest.raises(ValueError, match="duplicate"):
        MarketFrame(pl.concat((source, source.head(1))))
    with pytest.raises(ValueError, match="keys must not be null"):
        MarketFrame(
            source.with_columns(
                pl.when(pl.col("vt_symbol") == "A")
                .then(None)
                .otherwise(pl.col("vt_symbol"))
                .alias("vt_symbol")
            )
        )
    with pytest.raises(ValueError, match="close must be numeric"):
        MarketFrame(source.with_columns(pl.col("close").cast(pl.String)))


def test_daily_import_keeps_date_keys() -> None:
    market = MarketFrame(market_rows().with_columns(pl.col("datetime").cast(pl.Date)))
    factors = AlphaEngine([Alpha("price", "close")]).calculate_factors(market)
    assert market.frame.schema["datetime"] == pl.Date
    assert factors.frame.schema["datetime"] == pl.Date
    assert market.split(TimeSplitConfig(0.5)).holdout_start.isoformat() == "2026-01-02"


def test_factor_contract_requires_numeric_values_and_observed_market_keys() -> None:
    market = MarketFrame(market_rows())
    factors = FactorFrame(
        pl.DataFrame(
            {
                "datetime": [datetime(2026, 1, 1)],
                "vt_symbol": ["A"],
                "momentum": [None],
            },
            schema_overrides={"momentum": pl.Float64},
        )
    )
    factors.require_market_keys(market)
    with pytest.raises(ValueError, match="numeric"):
        FactorFrame(factors.frame.with_columns(pl.col("momentum").cast(pl.String)))
    with pytest.raises(ValueError, match="belong to market"):
        FactorFrame(
            factors.frame.with_columns(pl.lit("UNKNOWN").alias("vt_symbol"))
        ).require_market_keys(market)
    with pytest.raises(ValueError, match="same type"):
        FactorFrame(
            factors.frame.with_columns(pl.col("datetime").cast(pl.Date))
        ).require_market_keys(market)
    all_missing = FactorFrame(
        factors.frame.with_columns(pl.lit(None).alias("momentum"))
    )
    assert all_missing.frame.schema["momentum"] == pl.Float64
