from datetime import datetime, timedelta
from statistics import pstdev
from unittest.mock import patch

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from vnpy.alpha import Alpha, AlphaDefinition, AlphaEngine
from vnpy.alpha.expression import (
    ExpressionAnalyzer, ExpressionParser, PolarsCompiler, PolarsExecutor,
    compile_expression, op, window,
)


@pytest.fixture
def frame():
    # Deliberately interleave and reverse symbols/dates to catch ordering leaks.
    prices = {"A": [10., 12., 15., 18., 21.], "B": [100., 80., 120., 60., 90.]}
    return pl.DataFrame([
        {"datetime": datetime(2026, 1, 1) + timedelta(days=i),
         "vt_symbol": symbol, "close": prices[symbol][i], "volume": float(i + 1)}
        for i in reversed(range(5)) for symbol in ("B", "A")
    ])


def evaluate(frame, formula):
    return AlphaEngine([Alpha("factor", formula)]).calculate(frame)


def values(result, symbol="A", name="factor"):
    return result.filter(pl.col("vt_symbol") == symbol)[name].to_list()


def test_parser_reuses_tree_and_precedence():
    tree = ExpressionParser().parse("close / Ref(close, 5) - 1")
    assert tree == op("sub", op("div", "close", op("ts_delay", "close", window(5))), 1)
    assert compile_expression(tree) == "((close / ts_delay(close, 5)) - 1)"
    assert compile_expression(ExpressionParser().parse("close + 2 * volume")) == "(close + (2 * volume))"
    assert compile_expression(ExpressionParser().parse("-close ** 2")) == "(0 - pow1(close, 2))"


@pytest.mark.parametrize("formula", [
    "", "close +", "__import__('os')", "close.__class__", "close[0]",
    "(lambda: close)()", "Mean(close, window=2)", "Unknown(close)",
    "Ref(close, -1)", "Ref(close, 0)", "Mean(close, 1.5)",
    "Mean(close, True)", "Mean(close, volume)", "Mean(close)",
    "Mean(close, 1 + 1)", "Mean(1, 2)", "True", "'close'",
    "[close]", "close > 1", "label + close", "future_return", "datetime + close",
])
def test_invalid_formulas_fail_without_execution(formula):
    with pytest.raises((ValueError, TypeError)):
        AlphaEngine([Alpha("bad", formula)])


def test_analysis_infers_nested_history_and_fields():
    tree = ExpressionParser().parse("Std(close / Ref(close, 1) - 1, 20)")
    analysis = ExpressionAnalyzer().analyze(tree, {"close"})
    assert analysis.lookback == 21
    assert analysis.fields == frozenset({"close"})
    assert not analysis.uses_cross_section
    with pytest.raises(ValueError, match="unknown field"):
        PolarsCompiler().compile(tree, {"volume"})


def test_momentum_per_symbol_and_nested_volatility(frame):
    engine = AlphaEngine([
        Alpha("momentum", "close / Ref(close, 2) - 1"),
        Alpha("volatility", "Std(close / Ref(close, 1) - 1, 2)"),
    ])
    result = engine.calculate(frame)
    assert engine.min_bars == 3
    assert result.columns == ["datetime", "vt_symbol", "momentum", "volatility"]
    for symbol in ("A", "B"):
        prices = frame.filter(pl.col("vt_symbol") == symbol).sort("datetime")["close"].to_list()
        momentum = [prices[i] / prices[i - 2] - 1 for i in range(2, 5)]
        returns = [prices[i] / prices[i - 1] - 1 for i in range(1, 5)]
        volatility = [pstdev(returns[i - 1:i + 1]) for i in range(1, 4)]
        assert values(result, symbol, "momentum") == pytest.approx([None, None] + momentum)
        assert values(result, symbol, "volatility") == pytest.approx([None, None] + volatility)
    latest = engine.calculate_latest(frame)
    assert [sample.symbol for sample in latest] == ["A", "B"]
    assert latest[0].features["momentum"] == pytest.approx(21 / 15 - 1)


@pytest.mark.parametrize(("formula", "expected"), [
    ("Mean(close, 2)", [None, 11, 13.5, 16.5, 19.5]),
    ("Sum(close, 2)", [None, 22, 27, 33, 39]),
    ("Min(close, 2)", [None, 10, 12, 15, 18]),
    ("Max(close, 2)", [None, 12, 15, 18, 21]),
    ("Std(close, 2)", [None, 1, 1.5, 1.5, 1.5]),
    ("Delta(close, 2)", [None, None, 5, 6, 6]),
    ("Corr(close, volume, 2)", [None, 1, 1, 1, 1]),
    ("Abs(-close) + Sign(close) + Pow(close, 2) / close", [21, 25, 31, 37, 43]),
    ("Log(close / close)", [0, 0, 0, 0, 0]),
    ("2 + close * 2 - 1", [21, 25, 31, 37, 43]),
    ("Mean(Mean(close, 2), 2)", [None, None, 12.25, 15, 18]),
])
def test_native_operators_against_hand_calculation(frame, formula, expected):
    assert values(evaluate(frame, formula)) == pytest.approx(expected)


def test_nested_cross_section_and_time_series(frame):
    # A is smaller than B on every date, including when the input is shuffled.
    assert values(evaluate(frame, "Mean(Rank(close), 2)")) == [None, 1, 1, 1, 1]
    assert values(evaluate(frame, "Rank(Mean(close, 2))"), "B") == [None, 2, 2, 2, 2]
    assert values(evaluate(frame, "Ref(Rank(close), 1)"), "B") == [None, 2, 2, 2, 2]
    assert AlphaEngine([Alpha("rank", "Rank(close)")]).requires_cross_section
    tied = frame.with_columns(pl.lit(1.).alias("close"))
    assert values(evaluate(tied, "Rank(close)")) == [1.5] * 5


def test_explicit_stages_and_formula_engine_agree(frame):
    tree = ExpressionParser().parse("Mean(close / Ref(close, 1) - 1, 2)")
    compiled = PolarsCompiler().compile(tree, set(frame.columns))
    assert len(compiled.stages) == 2
    assert_frame_equal(
        PolarsExecutor(frame).run(compiled, "factor"),
        evaluate(frame, "Mean(close / Ref(close, 1) - 1, 2)"),
    )
    assert_frame_equal(
        PolarsExecutor(frame).run(PolarsCompiler().compile(op("ts_mean", "close", window(2))), "factor"),
        evaluate(frame, "Mean(close, 2)"),
    )


def test_new_formulas_do_not_call_legacy_eval(frame):
    with patch("vnpy.alpha.engine.calculate_by_expression", side_effect=AssertionError("legacy execution")):
        assert evaluate(frame, "Mean(close, 2)").height == frame.height


def test_legacy_definition_can_coexist(frame):
    result = AlphaEngine([
        AlphaDefinition("old", "close / ts_delay(close, 1) - 1", lookback=2),
        Alpha("new", "close / Ref(close, 1) - 1"),
    ]).calculate(frame)
    assert result["old"].to_list() == pytest.approx(result["new"].to_list())


def test_reparse_and_engine_snapshot(frame):
    alpha = Alpha("factor", "close")
    original = AlphaEngine([alpha])
    assert alpha.expression == ExpressionParser().parse("close")
    alpha.formula = "close * 2"
    updated = AlphaEngine([alpha])
    assert values(original.calculate(frame)) == [10, 12, 15, 18, 21]
    assert values(updated.calculate(frame)) == [20, 24, 30, 36, 42]


def test_missing_fields_duplicates_null_keys_and_names(frame):
    with pytest.raises(ValueError, match="missing columns: high"):
        evaluate(frame, "high / close")
    with pytest.raises(ValueError, match="duplicate"):
        evaluate(pl.concat([frame, frame.head(1)]), "close")
    with pytest.raises(ValueError, match="must not be null"):
        evaluate(frame.with_columns(pl.lit(None).alias("vt_symbol")), "close")
    with pytest.raises(ValueError, match="unique"):
        AlphaEngine([Alpha("same", "close"), Alpha("same", "volume")])
    with pytest.raises(ValueError, match="index column"):
        Alpha("datetime", "close")


def test_nulls_nonfinite_and_internal_column_collisions(frame):
    assert values(evaluate(frame, "close / 0")) == [None] * 5
    frame = frame.with_columns(pl.col("close").alias("__alpha_stage_0"))
    assert values(evaluate(frame, "Mean(__alpha_stage_0, 2)")) == [None, 11, 13.5, 16.5, 19.5]
    broken = frame.with_columns(
        pl.when(pl.col("close") == 15).then(float("nan")).otherwise(pl.col("close")).alias("close")
    )
    assert values(evaluate(broken, "Mean(close, 2)")) == [None, 11, None, None, 19.5]


def test_empty_frame_keeps_result_schema(frame):
    result = evaluate(frame.head(0), "Mean(close / Ref(close, 1) - 1, 2)")
    assert result.is_empty()
    assert result.columns == ["datetime", "vt_symbol", "factor"]


def test_future_rows_do_not_change_past_results(frame):
    cutoff = datetime(2026, 1, 4)
    formula = "Rank(Std(close / Ref(close, 1) - 1, 2))"
    past = frame.filter(pl.col("datetime") <= cutoff)
    assert_frame_equal(
        evaluate(frame, formula).filter(pl.col("datetime") <= cutoff),
        evaluate(past, formula),
    )


def test_import_entry_calculates_across_daily_partitions(frame, tmp_path, monkeypatch):
    from vnpy import main
    from vnpy.config.runtime_config import ParquetConfig, RunMode, RuntimeConfig
    from vnpy.datafeed.daily_store import DailyMarketStore
    from vnpy.factor import parquet_batch_runner

    store = DailyMarketStore(tmp_path / "snapshot")
    market = frame.with_columns(*(pl.col("close").alias(name) for name in ("open", "high", "low")))
    store.upsert(market)
    snapshot = {"root": str(store.root), "latest_date": "2026-01-05", "rows": frame.height}
    monkeypatch.setattr(parquet_batch_runner, "import_parquet_history", lambda *args: snapshot)
    config = RuntimeConfig(
        RunMode.PARQUET,
        ParquetConfig(root="unused", start="2026-01-01", end="2026-01-05"),
        raw={"parquet_import": {"enabled": True},
             "alphas": [{"name": "factor", "formula": "close / Ref(close, 2) - 1"}]},
    )
    main.run_from_config(config)
    result = pl.read_parquet(store.root / "factors.parquet")
    expected = evaluate(frame.with_columns(pl.col("datetime").cast(pl.Date)), "close / Ref(close, 2) - 1")
    assert_frame_equal(result, expected)


def test_replay_receives_formula_configuration(monkeypatch):
    from vnpy import main
    from vnpy.config.runtime_config import ParquetConfig, RunMode, RuntimeConfig

    alphas = [{"name": "factor", "formula": "Mean(close, 2)"}]
    config = RuntimeConfig(RunMode.PARQUET, ParquetConfig("unused", "2026-01-01", "2026-01-05"),
                           raw={"alphas": alphas})
    received = []
    monkeypatch.setattr(main, "run_parquet_replay", lambda setting, alphas: received.extend(alphas))
    main.run_from_config(config)
    assert received == alphas


def test_realtime_module_accepts_formula_config():
    from vnpy.factor.realtime_module import RealtimeFactorModule

    class Harness:
        def __init__(self):
            self.objects = {}

        def get_object(self, name):
            return self.objects.get(name)

        def set_object(self, name, value):
            self.objects[name] = value

        def get_config(self, name, default=None):
            if name == "alphas":
                return [{"name": "momentum", "formula": "close / Ref(close, 5) - 1"}]
            return default

    service = RealtimeFactorModule.factor_service.fget(Harness())
    assert service.alpha_engine.min_bars == 6
    assert service.alpha_engine.definitions[0].name == "momentum"
