"""Shared boundaries and sample accounting must agree across both entry points."""

from datetime import datetime, timedelta
import json
from pathlib import Path

import polars as pl
import pytest

from vnpy.alpha.alpha_formula import Alpha
from vnpy.alpha.alpha_engine import AlphaEngine
from vnpy.alpha.expression import FieldNode, OperatorNode, WindowNode
from vnpy.alpha.mining.ga import EvolutionConfig, FitnessEvaluator
from vnpy.alpha.mining.mining_evolution import run as run_evolution
from vnpy.alpha.modeling.modeling_evaluation_config import (
    FactorEvaluationConfig,
    parse_evaluation_config,
)
from vnpy.alpha.modeling.modeling_runtime_evaluation import (
    evaluate_frames,
    run_factor_evaluation,
)
from vnpy.alpha.alpha_time_split import (
    TimeSplitConfig,
    build_session_split,
    parse_time_split_config,
)
from vnpy.config.config_runtime_config import load_runtime_config


def market_frame() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "datetime": datetime(2026, 1, 1) + timedelta(days=day),
                "vt_symbol": f"S{asset}",
                "close": 100 * (1 + (asset + 1) * 0.001) ** day,
                "volume": 1000 + asset,
            }
            for day in range(56)
            for asset in range(12)
        ]
    )


def test_split_is_based_on_dates_despite_shuffling_and_missing_stocks():
    market = market_frame()
    expected = build_session_split(market["datetime"].to_list(), TimeSplitConfig())
    uneven = market.filter(
        ~((pl.col("vt_symbol") == "S0") & (pl.col("datetime").dt.day() % 2 == 0))
    )
    actual = build_session_split(
        uneven.sample(fraction=1, shuffle=True, seed=5)["datetime"].to_list(),
        TimeSplitConfig(),
    )
    assert actual == expected
    assert actual.train_count == 39
    assert len(actual.dates_for("holdout")) == 17
    assert actual.as_dict()["train"]["end"] < actual.as_dict()["holdout"]["start"]


@pytest.mark.parametrize("fraction", [True, 0, 1, -0.1, float("nan"), "0.7"])
def test_invalid_fraction_is_rejected(fraction):
    with pytest.raises(ValueError):
        TimeSplitConfig(fraction)


def test_too_short_calendar_does_not_silently_clamp_boundary():
    with pytest.raises(ValueError, match="at least one"):
        build_session_split([datetime(2026, 1, 1)], TimeSplitConfig())
    with pytest.raises(ValueError, match="at least one"):
        build_session_split(
            [datetime(2026, 1, 1), datetime(2026, 1, 2)], TimeSplitConfig(0.1)
        )


def test_legacy_config_is_supported_and_conflicting_settings_fail():
    assert (
        parse_time_split_config(
            {"factor_evaluation": {"train_fraction": 0.8}}
        ).train_fraction
        == 0.8
    )
    with pytest.raises(ValueError, match="conflicting"):
        parse_time_split_config(
            {
                "time_split": {"train_fraction": 0.8},
                "factor_evaluation": {"train_fraction": 0.7},
            }
        )


def test_runtime_report_and_genetic_scorer_use_shared_eighty_percent(tmp_path):
    source = Path(__file__).resolve().parents[1] / "config/config_runtime_basic_alphas.json"
    raw = json.loads(source.read_text(encoding="utf-8"))
    raw["time_split"]["train_fraction"] = 0.8
    config_path = tmp_path / "config_runtime.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    runtime = load_runtime_config(config_path)
    report = parse_evaluation_config(
        runtime.raw["factor_evaluation"], tmp_path, time_split=runtime.time_split
    )
    options = EvolutionConfig(
        min_assets=3, min_days=2, train_fraction=runtime.time_split.train_fraction
    )
    market = market_frame()
    report_split = build_session_split(
        market["datetime"].to_list(), TimeSplitConfig(report.train_fraction)
    )
    scorer = FitnessEvaluator(market, options, split=report_split)
    assert scorer.split.train_count == 44
    assert len(scorer.split.dates_for("holdout")) == 12
    assert scorer.split_date == report_split.holdout_start


def test_evolution_entrypoint_reads_shared_fraction(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "config/config_runtime_basic_alphas.json"
    raw = json.loads(source.read_text(encoding="utf-8"))
    raw["time_split"]["train_fraction"] = 0.8
    raw["expression_iteration"]["output_root"] = str(tmp_path / "mining")
    config_path = tmp_path / "config_runtime.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    market = market_frame().with_columns(
        pl.col("close").alias("open"),
        pl.col("close").alias("high"),
        pl.col("close").alias("low"),
    )
    monkeypatch.setattr(
        "vnpy.alpha.mining.mining_evolution.ParquetDataFeed.__init__",
        lambda self, root: None,
    )
    monkeypatch.setattr(
        "vnpy.alpha.mining.mining_evolution.ParquetDataFeed.load_frame",
        lambda self, **kwargs: market,
    )
    result = run_evolution(config_path, population=20, generations=1)
    assert result.split.train_count == 44
    assert len(result.split.dates_for("holdout")) == 12
    assert len(result.candidates) == 20
    assert not (tmp_path / "mining").exists()
    admissions = result.seed_admission
    assert len(admissions) == len(raw["alphas"])
    assert sum(row["status"] == "admitted" for row in admissions) >= 1
    assert any(
        row["name"] == "volatility_5" and "max_depth" in row["reason"]
        for row in admissions
    )
    assert any(
        item.origin == "seed"
        and item.seed_name in {alpha["name"] for alpha in raw["alphas"]}
        for item in result.candidates
    )
    assert len(result.holdout) == 20


def test_imported_report_reuses_runtime_boundary_even_with_old_options(
    tmp_path, monkeypatch
):
    source = Path(__file__).resolve().parents[1] / "config/config_runtime_basic_alphas.json"
    raw = json.loads(source.read_text(encoding="utf-8"))
    raw["time_split"]["train_fraction"] = 0.8
    config_path = tmp_path / "config_runtime.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    runtime = load_runtime_config(config_path)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    market_frame().write_parquet(snapshot / "2026-01-01.parquet")

    def capture(market, factors, definitions, options, **kwargs):
        return options, kwargs["split"]

    monkeypatch.setattr(
        "vnpy.alpha.modeling.modeling_runtime_evaluation.evaluate_frames", capture
    )
    options, split = run_factor_evaluation(
        runtime,
        FactorEvaluationConfig(),
        AlphaEngine([Alpha("price", "close")]),
        [{"name": "price", "formula": "close"}],
        snapshot={"root": str(snapshot)},
    )
    assert options.train_fraction == 0.8
    assert split.train_count == 44


def test_report_and_fitness_sample_losses_match_and_holdout_has_six_days(tmp_path):
    market = market_frame()
    alphas = [Alpha("mean5", "Mean(close, 5)"), Alpha("constant", "close - close")]
    factors = AlphaEngine(alphas).calculate(market)
    options = FactorEvaluationConfig(
        periods=(1, 10), quantiles=3, min_assets=3, output_root=str(tmp_path)
    )
    result = evaluate_frames(
        market,
        factors,
        [{"name": a.name, "formula": a.formula} for a in alphas],
        options,
    )
    output = Path(result["output"])
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    rows = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert manifest["time_split"]["train"]["sessions"] == 39
    assert manifest["time_split"]["holdout"]["sessions"] == 17
    assert (
        json.loads((output / "split.json").read_text(encoding="utf-8"))
        == manifest["time_split"]
    )
    assert (
        next(
            r
            for r in manifest["label_capacity"]
            if r["segment"] == "holdout" and r["period"] == 10
        )["max_label_days"]
        == 6
    )
    tree = OperatorNode("ts_mean", (FieldNode("close"), WindowNode(5)))
    for period in (1, 10):
        evolution = EvolutionConfig(period=period, min_assets=3, min_days=2)
        scorer = FitnessEvaluator(market, evolution)
        for holdout, segment in ((False, "train"), (True, "holdout")):
            row = next(
                r
                for r in rows
                if r["factor"] == "mean5"
                and r["period"] == period
                and r["segment"] == segment
            )
            fitness = scorer.score(tree, holdout=holdout)
            assert fitness.ic_days == row["ic_days"]
            for key, value in fitness.sample_counts.items():
                assert row[key] == value
            assert (
                row["ic_rows"]
                + sum(row[key + "_rows"] for key in manifest["sample_loss_order"])
                == row["input_rows"]
            )
            assert (
                row["ic_days"]
                + sum(row[key + "_days"] for key in manifest["sample_loss_order"])
                == row["input_days"]
            )
    holdout10 = next(
        r
        for r in rows
        if r["factor"] == "mean5" and r["period"] == 10 and r["segment"] == "holdout"
    )
    assert holdout10["ic_days"] == 6
    assert holdout10["missing_future_days"] == 11
    assert all(
        r["status"] == "skipped" and r["constant_factor_days"] > 0
        for r in rows
        if r["factor"] == "constant"
    )


def test_missing_prices_are_counted_and_holdout_changes_cannot_affect_training():
    market = market_frame()
    options = EvolutionConfig(min_assets=3, min_days=2)
    changed = market.with_columns(
        pl.when(
            (pl.col("datetime") == datetime(2026, 1, 4)) & (pl.col("vt_symbol") == "S0")
        )
        .then(None)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    baseline = FitnessEvaluator(changed, options)
    first = baseline.score(FieldNode("volume"))
    assert first.sample_counts["invalid_price_rows"] == 2
    assert first.sample_counts["invalid_price_days"] == 0
    changed_holdout = changed.with_columns(
        pl.when(pl.col("datetime") >= baseline.split_date)
        .then(pl.lit(None))
        .otherwise(pl.col("close"))
        .alias("close")
    )
    assert (
        FitnessEvaluator(changed_holdout, options).score(FieldNode("volume")) == first
    )
