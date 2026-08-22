from datetime import datetime, timedelta

import numpy as np
import polars as pl

from vnpy.alpha.dataset import Segment
from vnpy.alpha.model.models import LinearRegressionModel
from vnpy.alpha.model import Reweighter


class FirstSampleOnly(Reweighter):
    def reweight(self, frame: pl.DataFrame) -> np.ndarray:
        weights = np.zeros(frame.height)
        weights[0] = 1.0
        return weights


class MockDataset:
    def __init__(self) -> None:
        rows: list[dict] = []
        start = datetime(2025, 1, 1)
        for index in range(30):
            momentum = index / 100
            volatility = 0.01 + index / 1000
            rows.append({
                "datetime": start + timedelta(days=index),
                "vt_symbol": "000001.SSE",
                "momentum_20": momentum,
                "volatility_20": volatility,
                "label": 0.001 + 0.30 * momentum - 0.20 * volatility,
            })

        self.train = pl.DataFrame(rows[:20])
        self.valid = pl.DataFrame(rows[20:25])
        self.test = pl.DataFrame(rows[25:])

    def fetch_learn(self, segment: Segment) -> pl.DataFrame:
        return self.train if segment == Segment.TRAIN else self.valid

    def fetch_infer(self, segment: Segment) -> pl.DataFrame:
        assert segment == Segment.TEST
        return self.test


def test_linear_regression_model_fit_and_predict() -> None:
    dataset = MockDataset()
    model = LinearRegressionModel()

    model.fit(dataset)  # type: ignore[arg-type]
    prediction = model.predict(dataset, Segment.TEST)  # type: ignore[arg-type]

    assert model.model is not None
    assert model.feature_names == ["momentum_20", "volatility_20"]
    assert np.allclose(prediction, dataset.test["label"].to_numpy())


def test_linear_regression_model_requires_fit() -> None:
    model = LinearRegressionModel()

    try:
        model.predict(MockDataset(), Segment.TEST)  # type: ignore[arg-type]
    except ValueError as exc:
        assert str(exc) == "model is not fitted yet!"
    else:
        raise AssertionError("predict() should reject an unfitted model")


def test_model_is_callable_and_serializable(tmp_path) -> None:
    dataset = MockDataset()
    model = LinearRegressionModel()
    model.fit(dataset)
    output = tmp_path / "linear-model.pkl"

    model.save(output)
    restored = LinearRegressionModel.load(output)

    assert np.allclose(restored(dataset, Segment.TEST), model.predict(dataset, Segment.TEST))


def test_linear_regression_accepts_sample_reweighter() -> None:
    dataset = MockDataset()
    model = LinearRegressionModel()

    model.fit(dataset, FirstSampleOnly())

    assert model.model is not None
