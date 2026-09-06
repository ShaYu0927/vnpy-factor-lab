from datetime import date, timedelta

import numpy as np
import polars as pl

from vnpy.alpha.modeling import (
    ChronologicalSplitter,
    FactorObservation,
    FactorSelectionMetric,
    FactorSelectionResult,
    AlphaModelWorkflow,
    ModelArtifact,
    StandardFeaturePipeline,
)


class SelectMomentumOnly:
    def select(self, training, feature_names):
        return FactorSelectionResult(
            selected_features=("momentum",),
            metrics=(FactorSelectionMetric("momentum", 0.1, 0.5, 1, True),),
        )


def make_observations(days: int = 30) -> list[FactorObservation]:
    observations: list[FactorObservation] = []
    start = date(2025, 1, 1)
    for offset in range(days):
        for symbol_index, symbol in enumerate(("AAA.SSE", "BBB.SSE")):
            trend = 1.0 + symbol_index * 0.2
            observations.append(FactorObservation(
                trade_date=start + timedelta(days=offset),
                symbol=symbol,
                close=100.0 + trend * offset,
                features={
                    "momentum": trend * offset / 100.0,
                    "volatility": 0.01 + offset / 1000.0,
                },
            ))
    return observations


def test_chronological_splitter_keeps_dates_in_one_segment() -> None:
    frame = pl.DataFrame({
        "datetime": [date(2025, 1, day) for day in range(1, 11) for _ in range(2)],
        "vt_symbol": ["AAA", "BBB"] * 10,
        "feature": range(20),
        "label": range(20),
    })

    split = ChronologicalSplitter().split(frame)
    train_dates = set(split.train["datetime"].to_list())
    valid_dates = set(split.valid["datetime"].to_list())
    test_dates = set(split.test["datetime"].to_list())

    assert train_dates.isdisjoint(valid_dates | test_dates)
    assert valid_dates.isdisjoint(test_dates)
    assert max(train_dates) < min(valid_dates) < min(test_dates)


def test_chronological_splitter_purges_forward_label_overlap() -> None:
    frame = pl.DataFrame({
        "datetime": [date(2025, 1, day) for day in range(1, 16)],
        "vt_symbol": ["AAA"] * 15,
        "feature": range(15),
        "label": range(15),
    })

    split = ChronologicalSplitter(purge_horizon=2).split(frame)

    assert split.train["datetime"].max() == date(2025, 1, 6)
    assert split.valid["datetime"].to_list() == [
        date(2025, 1, 9),
        date(2025, 1, 10),
    ]
    assert split.test["datetime"].min() == date(2025, 1, 13)


def test_standard_pipeline_reuses_training_statistics() -> None:
    pipeline = StandardFeaturePipeline(["feature"])
    train = pl.DataFrame({"feature": [1.0, 2.0, 3.0]})
    future = pl.DataFrame({"feature": [100.0]})

    pipeline.fit(train)
    transformed = pipeline.transform(future)

    assert pipeline.means["feature"] == 2.0
    assert transformed["feature"][0] > 100


def test_workflow_saves_model_with_inference_contract(tmp_path) -> None:
    workflow = AlphaModelWorkflow(
        ["momentum", "volatility"],
        horizon=3,
        standardize=True,
    )
    result = workflow.run(make_observations())
    output = tmp_path / "model.pkl"

    result.artifact.save(output)
    restored = ModelArtifact.load(output)

    assert restored.feature_names == ("momentum", "volatility")
    assert restored.label_horizon == 3
    assert set(restored.preprocessor.means) == {"momentum", "volatility"}
    assert len(result.predictions) == 2
    assert np.isfinite(result.metrics.mse)


def test_workflow_trains_and_predicts_with_selected_features_only() -> None:
    workflow = AlphaModelWorkflow(
        ["momentum", "volatility"],
        horizon=3,
        factor_selector=SelectMomentumOnly(),  # type: ignore[arg-type]
    )

    result = workflow.run(make_observations())

    assert result.artifact.feature_names == ("momentum",)
    assert result.artifact.model.feature_names == ["momentum"]
    assert len(result.predictions) == 2
