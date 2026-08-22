"""Run a complete close-price-to-prediction linear regression example."""

from datetime import datetime, timedelta
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import mean_squared_error, r2_score

# Allow the example to run directly from a source checkout.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vnpy.alpha.dataset import Segment
from vnpy.alpha.model.models import LinearRegressionModel


class DemoDataset:
    """Small adapter exposing the dataset methods required by AlphaModel."""

    def __init__(self, frame: pd.DataFrame) -> None:
        train_end = int(len(frame) * 0.6)
        valid_end = int(len(frame) * 0.8)

        self.frames = {
            Segment.TRAIN: pl.from_pandas(frame.iloc[:train_end]),
            Segment.VALID: pl.from_pandas(frame.iloc[train_end:valid_end]),
            Segment.TEST: pl.from_pandas(frame.iloc[valid_end:]),
        }

    def fetch_learn(self, segment: Segment) -> pl.DataFrame:
        return self.frames[segment]

    def fetch_infer(self, segment: Segment) -> pl.DataFrame:
        return self.frames[segment]


def make_price_data(sample_count: int = 300, seed: int = 42) -> pd.DataFrame:
    """Create reproducible historical closing prices for the runnable demo."""
    random = np.random.default_rng(seed)
    daily_returns = random.normal(0.0005, 0.015, sample_count)
    close = 100 * np.cumprod(1 + daily_returns)

    return pd.DataFrame({
        "datetime": [datetime(2025, 1, 1) + timedelta(days=i) for i in range(sample_count)],
        "vt_symbol": "DEMO.SSE",
        "close": close,
    })


def build_factor_data(price_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate two factors and the actual forward five-day return label."""
    data = price_data.copy()
    daily_return = data["close"].pct_change()

    data["momentum_20"] = data["close"] / data["close"].shift(20) - 1
    data["volatility_20"] = daily_return.rolling(20).std()
    data["label"] = data["close"].shift(-5) / data["close"] - 1

    columns = [
        "datetime",
        "vt_symbol",
        "momentum_20",
        "volatility_20",
        "label",
    ]
    return data[columns].dropna().reset_index(drop=True)


def run_demo() -> dict[str, object]:
    """Build labels, split chronologically, fit, and predict the test period."""
    price_data = make_price_data()
    factor_data = build_factor_data(price_data)
    dataset = DemoDataset(factor_data)

    model = LinearRegressionModel()
    model.fit(dataset)  # type: ignore[arg-type]

    predictions = model.predict(dataset, Segment.TEST)  # type: ignore[arg-type]
    actual = dataset.frames[Segment.TEST]["label"].to_numpy()
    mse = float(mean_squared_error(actual, predictions))
    r2 = float(r2_score(actual, predictions))

    assert model.model is not None
    print("========== 收盘价到未来5日收益预测 ==========")
    print(f"原始行情数量: {len(price_data)}")
    print(f"可用因子样本: {len(factor_data)}")
    print(f"测试样本数量: {len(actual)}")
    print(f"截距: {model.model.intercept_:.8f}")
    for name, coefficient in zip(model.feature_names, model.model.coef_, strict=False):
        print(f"{name} 权重: {coefficient:.8f}")
    print(f"测试集 MSE: {mse:.8f}")
    print(f"测试集 R2: {r2:.6f}")
    print(f"第一条真实未来5日收益: {actual[0]:.4%}")
    print(f"第一条模型预测收益: {predictions[0]:.4%}")

    return {
        "price_data": price_data,
        "factor_data": factor_data,
        "model": model,
        "predictions": predictions,
        "actual": actual,
        "mse": mse,
        "r2": r2,
    }


if __name__ == "__main__":
    run_demo()
