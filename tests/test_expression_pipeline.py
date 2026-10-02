"""Check the path from a user formula to keyed factor values."""

from datetime import datetime

import polars as pl
import pytest

from vnpy.alpha.alpha import Alpha
from vnpy.alpha.engine import AlphaEngine
from vnpy.alpha.expression import ExpressionAnalyzer, ExpressionParser


def market_frame() -> pl.DataFrame:
    return pl.DataFrame({
        "datetime": [datetime(2026, 1, day) for day in (1, 2, 3) for _ in range(2)],
        "vt_symbol": ["A", "B"] * 3,
        "close": [10.0, 20.0, 12.0, 22.0, 15.0, 24.0],
    })


def test_nested_formula_from_tree_to_market_values() -> None:
    formula = "ts_mean(close / ts_delay(close, 1) - 1, 2)"
    tree = ExpressionParser().parse(formula)
    analysis = ExpressionAnalyzer().analyze(tree, set(market_frame().columns))

    assert tree.name == "ts_mean"
    assert tree.depth == 5
    assert analysis.fields == frozenset({"close"})
    assert analysis.lookback == 3
    assert not analysis.uses_cross_section

    engine = AlphaEngine([Alpha(name="mean_return", formula=formula)])
    assert engine.min_bars == analysis.lookback
    rows = engine.calculate(market_frame()).to_dicts()

    assert [row["mean_return"] for row in rows[:4]] == [None] * 4
    assert rows[4]["vt_symbol"] == "A"
    assert rows[4]["mean_return"] == pytest.approx((0.2 + 0.25) / 2)
    assert rows[5]["vt_symbol"] == "B"
    assert rows[5]["mean_return"] == pytest.approx((0.1 + (24 / 22 - 1)) / 2)


def test_cross_section_formula_uses_each_dates_complete_market() -> None:
    formula = "cs_rank(ts_mean(close, 2))"
    tree = ExpressionParser().parse(formula)
    analysis = ExpressionAnalyzer().analyze(tree)

    assert analysis.lookback == 2
    assert analysis.uses_cross_section

    engine = AlphaEngine([Alpha(name="price_rank", formula=formula)])
    rows = engine.calculate(market_frame()).to_dicts()

    assert [row["price_rank"] for row in rows[:2]] == [None, None]
    assert [row["price_rank"] for row in rows[2:]] == [1.0, 2.0, 1.0, 2.0]


@pytest.mark.parametrize("formula", [
    "ts_mean(close, 0)",
    "ts_mean(close, -1)",
    "close.__class__",
    "unknown_operator(close)",
])
def test_invalid_formula_is_rejected_before_execution(formula: str) -> None:
    with pytest.raises((TypeError, ValueError)):
        AlphaEngine([Alpha(name="invalid", formula=formula)])
