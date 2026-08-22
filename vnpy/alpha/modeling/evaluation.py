import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from .schema import RegressionMetrics


class RegressionEvaluator:
    """Evaluate predictions without coupling metrics to model training."""

    def evaluate(self, actual: np.ndarray, predicted: np.ndarray) -> RegressionMetrics:
        return RegressionMetrics(
            mse=float(mean_squared_error(actual, predicted)),
            r2=float(r2_score(actual, predicted)),
        )

