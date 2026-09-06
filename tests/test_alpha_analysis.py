from datetime import date, timedelta

import numpy as np
import polars as pl

from vnpy.alpha.modeling import AlphaAnalyzer, AlphaDatasetBuilder, AlphaFactorSelector


def make_frames(days: int = 6) -> tuple[pl.DataFrame, pl.DataFrame]:
    features = []
    prices = []
    start = date(2025, 1, 1)
    for day in range(days):
        for index, symbol in enumerate(("A", "B", "C"), start=1):
            features.append({
                "datetime": start + timedelta(days=day),
                "vt_symbol": symbol,
                "alpha001": float(index),
                "alpha002": float(-index),
            })
            prices.append({
                "datetime": start + timedelta(days=day),
                "vt_symbol": symbol,
                "close": 100.0 * (1.0 + index * 0.01) ** day,
            })
    return pl.DataFrame(features), pl.DataFrame(prices)


def test_alpha_dataset_uses_next_bar_entry_label() -> None:
    features, prices = make_frames()

    dataset = AlphaDatasetBuilder(horizon=1, entry_offset=1).build(features, prices)

    first_a = dataset.filter(
        (pl.col("datetime") == date(2025, 1, 1)) & (pl.col("vt_symbol") == "A")
    )
    assert np.isclose(first_a["label"][0], 0.01)
    assert dataset["datetime"].max() == date(2025, 1, 4)


def test_alpha_analyzer_reports_ic_and_rank_ic() -> None:
    features, prices = make_frames()
    dataset = AlphaDatasetBuilder().build(features, prices)

    report = AlphaAnalyzer(min_assets=3).evaluate(dataset)
    metrics = {item.factor_name: item for item in report.metrics}

    assert np.isclose(metrics["alpha001"].mean_ic, 1.0)
    assert np.isclose(metrics["alpha001"].mean_rank_ic, 1.0)
    assert np.isclose(metrics["alpha002"].mean_rank_ic, -1.0)
    assert metrics["alpha001"].observations == 4
    assert set(report.daily.columns) == {
        "datetime", "factor_name", "ic", "rank_ic", "asset_count"
    }


def test_alpha_analyzer_ignores_dates_without_enough_assets() -> None:
    features, prices = make_frames()
    dataset = AlphaDatasetBuilder().build(features, prices).filter(pl.col("vt_symbol") != "C")

    report = AlphaAnalyzer(min_assets=3).evaluate(dataset)

    assert report.daily.is_empty()
    assert all(item.observations == 0 for item in report.metrics)


def test_factor_selector_reuses_the_labeled_training_frame() -> None:
    features, prices = make_frames(days=8)
    dataset = AlphaDatasetBuilder().build(features, prices).with_columns(
        pl.when(
            (pl.col("datetime") == date(2025, 1, 6))
            & (pl.col("vt_symbol") == "B")
        ).then(0.03).when(
            (pl.col("datetime") == date(2025, 1, 6))
            & (pl.col("vt_symbol") == "C")
        ).then(0.02).otherwise(pl.col("label")).alias("label")
    )
    selector = AlphaFactorSelector(
        min_abs_rank_ic=0.5,
        min_abs_rank_ic_ir=0.0,
        min_assets=3,
        min_observations=5,
    )

    selection = selector.select(dataset, ["alpha001", "alpha002"])

    assert selection.selected_features == ("alpha001", "alpha002")
    assert all(item.observations == 6 for item in selection.metrics)
