import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
from scipy.stats import spearmanr

from vnpy import main
from vnpy.alpha import Alpha, AlphaEngine
from vnpy.alpha.modeling.alphalens_evaluator import AlphalensEvaluator
from vnpy.alpha.modeling.evaluation_config import FactorEvaluationConfig, parse_evaluation_config
from vnpy.alpha.modeling.runtime_evaluation import build_forward_labels, evaluate_frames
from vnpy.config.runtime_config import ParquetConfig, RunMode, RuntimeConfig, load_runtime_config
from vnpy.event.engine import ModuleEngine


def test_forward_labels_use_next_session_entry_and_do_not_fill_missing_prices():
    dates = pd.bdate_range("2026-01-01", periods=8, tz="Asia/Shanghai")
    prices = pd.DataFrame({"A": [10, 20, 60, 120, 240, 480, 960, 1920],
                           "B": [10, np.nan, 30, 0, 50, 60, 70, 80]}, index=dates)
    labels = build_forward_labels(prices, (1, 3), 1)
    assert labels.loc[(dates[0], "A"), "1D"] == 2  # 60 / 20 - 1, not 20 / 10 - 1
    assert labels.loc[(dates[0], "A"), "3D"] == 11  # 240 / 20 - 1
    assert labels.loc[(dates[0], "A"), "entry_date"] == dates[1]
    assert labels.loc[(dates[0], "A"), "exit_3D"] == dates[4]
    assert pd.isna(labels.loc[(dates[0], "B"), "1D"])
    assert pd.isna(labels.loc[(dates[1], "B"), "1D"])
    assert pd.isna(labels.loc[(dates[2], "B"), "1D"])
    assert pd.isna(labels.loc[(dates[-2], "A"), "1D"])
    assert labels.loc[(dates[-4], "A"), "1D"] == 1
    assert pd.isna(labels.loc[(dates[-4], "A"), "3D"])


def test_alphalens_rank_ic_uses_all_pairs_even_when_quantile_ties_fail():
    index = pd.MultiIndex.from_product([pd.bdate_range("2026-01-01", periods=4), list("ABCDEF")], names=["date", "asset"])
    values = [0, 0, 0, 0, 0, 1]
    factor = pd.Series(values * 4, index=index)
    returns = pd.DataFrame({"1D": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] * 4}, index=index)
    report = AlphalensEvaluator(periods=(1,), quantiles=5).evaluate_forward_returns(factor, returns)
    assert report.information_coefficient["1D"].dropna().tolist() == pytest.approx([spearmanr(values, [1, 2, 3, 4, 5, 6]).statistic] * 4)
    assert report.clean_data.empty
    assert len(report.ic_data) == 24
    assert report.quantile_returns.empty


def frames():
    dates = pd.bdate_range("2026-01-01", periods=20).to_pydatetime()
    rows = [{"datetime": date, "vt_symbol": f"SHSE.{600000+i}",
             "open": 100 * (1 + (i + 1) * .01) ** t,
             "high": 101 * (1 + (i + 1) * .01) ** t,
             "low": 99 * (1 + (i + 1) * .01) ** t,
             "close": 100 * (1 + (i + 1) * .01) ** t,
             "volume": float(i+1), "amount": float((i+1)*100)}
            for t, date in enumerate(dates) for i in range(6)]
    market = pl.DataFrame(rows)
    definitions = [Alpha("positive", "volume"), Alpha("negative", "0-volume"),
                   Alpha("constant", "close/close"), Alpha("invalid", "close/0")]
    return market, definitions


def run_frames(market, definitions, options):
    result = evaluate_frames(market, AlphaEngine(definitions).calculate(market),
                             [{"name": a.name, "formula": a.formula} for a in definitions], options)
    path = Path(result["output"])
    return path, json.loads((path / "summary.json").read_text(encoding="utf-8"))


def test_reports_positive_negative_constant_invalid_and_purges_train_boundary(tmp_path):
    market, definitions = frames()
    options = FactorEvaluationConfig(periods=(1, 3), quantiles=2, min_assets=3, output_root=str(tmp_path))
    output, rows = run_frames(market, definitions, options)
    assert len(rows) == 24
    for row in rows:
        if row["factor"] in ("positive", "negative"):
            assert row["status"] == "ok"
            assert row["rank_ic"] == pytest.approx(1 if row["factor"] == "positive" else -1)
        else:
            assert row["status"] == "skipped" and row["rank_ic"] is None
    one = next(r for r in rows if r["factor"] == "positive" and r["segment"] == "train" and r["period"] == 1)
    three = next(r for r in rows if r["factor"] == "positive" and r["segment"] == "train" and r["period"] == 3)
    assert one["boundary_dropped"] == 12 and one["ic_days"] == 12
    assert three["boundary_dropped"] == 24 and three["ic_days"] == 10
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["evaluated_rows"] == manifest["skipped_rows"] == 12
    cutoff = pd.Timestamp(manifest["holdout_start"])
    changed = market.with_columns(pl.when(pl.col("datetime") >= cutoff.to_pydatetime()).then(pl.col("close") * pl.col("volume") * 10).otherwise(pl.col("close")).alias("close"))
    other, revised = run_frames(changed, definitions, options)
    assert output != other
    assert [r for r in rows if r["segment"] == "train"] == [r for r in revised if r["segment"] == "train"]
    assert (output / "report.html").is_file()
    assert (output / "labels.parquet").is_file()


@pytest.mark.parametrize("raw", [
    {"enabled": "yes"}, {"enabled": True, "entry_offset": 0},
    {"enabled": True, "periods": [True]}, {"enabled": True, "periods": []},
    {"enabled": True, "train_fraction": 1}, {"enabled": True, "min_assets": 2},
    {"enabled": True, "unknown": 1},
])
def test_bad_evaluation_settings_fail_early(raw, tmp_path):
    with pytest.raises(ValueError):
        parse_evaluation_config(raw, tmp_path)


@pytest.mark.parametrize("import_enabled", [False, True])
def test_main_evaluates_all_history_and_resolves_output_from_config(tmp_path, monkeypatch, import_enabled):
    market, _ = frames()
    folder = tmp_path / "source" / "SHSE_2026"
    folder.mkdir(parents=True)
    market.rename({"vt_symbol": "symbol"}).with_columns(pl.col("datetime").dt.strftime("%Y-%m-%d").alias("trade_date")).drop("datetime").write_parquet(folder / "dists_day_bar.parquet")
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    raw = {"mode": "parquet", "parquet": {"root": str(folder.parent), "start": "2026-01-01", "end": "2026-02-01", "markets": "SHSE"},
           "alphas": [{"name": "positive", "formula": "volume"}],
           "factor_evaluation": {"enabled": True, "periods": [1], "quantiles": 2, "min_assets": 3, "output_root": "../reports"},
           "parquet_import": {"enabled": import_enabled, "root": str(tmp_path / "imports")}}
    path = config_dir / "runtime.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.chdir(tmp_path.parent)
    monkeypatch.setattr(main, "module_engine", ModuleEngine())
    original = main.register_factor_module
    def small_cache(*args, **kwargs):
        original(*args, **kwargs)
        main.module_engine.get_context("factor").set_config("sample_maxlen", 1)
    monkeypatch.setattr(main, "register_factor_module", small_cache)
    result = main.run_from_config(load_runtime_config(path))
    output = Path(result["evaluation"]["output"])
    assert output.parent == tmp_path / "reports"
    rows = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    full = next(row for row in rows if row["segment"] == "all")
    assert full["input_rows"] == 120 and full["ic_days"] == 18
    assert full["rank_ic"] == pytest.approx(1)


def test_disabled_evaluation_preserves_existing_replay(tmp_path, monkeypatch):
    config = RuntimeConfig(RunMode.PARQUET, ParquetConfig("unused", "2026-01-01", "2026-01-02"),
                           raw={"alphas": [{"name": "fixed", "formula": "close"}]}, config_dir=tmp_path)
    monkeypatch.setattr(main, "run_parquet_replay", lambda *args, **kwargs: {"bars": 2})
    assert main.run_from_config(config) == {"bars": 2}
    assert not (tmp_path / "reports").exists()
