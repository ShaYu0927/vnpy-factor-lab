import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression  # type: ignore

from vnpy.alpha import AlphaDataset, AlphaModel, Segment, logger
from vnpy.alpha.model.reweighter import Reweighter, validate_weights


class LinearRegressionModel(AlphaModel):
    """Ordinary least-squares linear regression factor model."""

    def __init__(self, fit_intercept: bool = True) -> None:
        self.fit_intercept: bool = fit_intercept
        self.model: LinearRegression | None = None
        self.feature_names: list[str] = []

    def fit(
        self,
        dataset: AlphaDataset,
        reweighter: Reweighter | None = None,
    ) -> None:
        """Fit the model with the chronological train and valid segments."""
        df_train: pl.DataFrame = dataset.fetch_learn(Segment.TRAIN)
        df_valid: pl.DataFrame = dataset.fetch_learn(Segment.VALID)

        df_learn: pl.DataFrame = (
            pl.concat([df_train, df_valid])
            .unique(subset=["datetime", "vt_symbol"])
            .sort(["datetime", "vt_symbol"])
        )

        if "label" not in df_learn.columns:
            raise ValueError("learning data must contain a 'label' column")

        self.feature_names = [
            column
            for column in df_learn.columns
            if column not in {"datetime", "vt_symbol", "label"}
        ]
        if not self.feature_names:
            raise ValueError("learning data must contain at least one feature")

        x: np.ndarray = df_learn.select(self.feature_names).to_numpy()
        y: np.ndarray = df_learn["label"].to_numpy()

        self.model = LinearRegression(fit_intercept=self.fit_intercept)
        sample_weight = None
        if reweighter is not None:
            sample_weight = validate_weights(reweighter(df_learn), df_learn.height)
        self.model.fit(x, y, sample_weight=sample_weight)

    def predict(self, dataset: AlphaDataset, segment: Segment) -> np.ndarray:
        """Predict labels for a dataset segment in time-symbol order."""
        if self.model is None:
            raise ValueError("model is not fitted yet!")

        df: pl.DataFrame = dataset.fetch_infer(segment)
        df = df.sort(["datetime", "vt_symbol"])
        x: np.ndarray = df.select(self.feature_names).to_numpy()
        return self.model.predict(x)

    def detail(self) -> None:
        """Log the intercept and coefficient of every factor."""
        if self.model is None:
            raise ValueError("model is not fitted yet!")

        logger.info(f"LinearRegression intercept: {float(self.model.intercept_):.6f}")
        for name, coefficient in zip(
            self.feature_names,
            self.model.coef_,
            strict=False,
        ):
            logger.info(f"{name}: {float(coefficient):.6f}")
