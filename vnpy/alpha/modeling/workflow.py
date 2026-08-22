from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import polars as pl

from vnpy.alpha.dataset import Segment
from vnpy.alpha.model.models import LinearRegressionModel
from vnpy.alpha.model.template import AlphaModel
from vnpy.alpha.model.reweighter import Reweighter

from .artifact import ModelArtifact
from .dataset import ChronologicalSplitter, DatasetSplit, ForwardReturnDatasetBuilder, FrameDataset
from .evaluation import RegressionEvaluator
from .factor_selection import AlphalensFactorSelector, FactorSelectionResult
from .preprocessing import StandardFeaturePipeline
from .schema import FactorObservation, ModelPrediction, RegressionMetrics


@dataclass(slots=True)
class TrainingResult:
    artifact: ModelArtifact   # 训练完成后的模型产物，包含模型对象、参数配置、特征信息等，可用于后续预测
    split: DatasetSplit       # 数据集划分结果，包含训练集、验证集、测试集，记录模型训练使用的数据范围
    metrics: RegressionMetrics   # 模型评估指标，例如 MSE、RMSE、R² 等，用于衡量预测效果
    predictions: list[ModelPrediction]  # 模型在测试集或验证集上的预测结果，用于回测分析和误差评估
    factor_selection: FactorSelectionResult | None = None   # 因子筛选结果，记录通过 Alphalens 过滤后的有效因子及对应评价指标


class LinearModelWorkflow:
    """Orchestrate preparation, training, evaluation, and latest ranking."""

    def __init__(
        self,
        feature_names: Sequence[str],
        horizon: int = 5,
        model: AlphaModel | None = None,
        standardize: bool = False,
        factor_selector: AlphalensFactorSelector | None = None,
        reweighter: Reweighter | None = None,
    ) -> None:
        self.feature_names = tuple(feature_names)
        self.builder = ForwardReturnDatasetBuilder(self.feature_names, horizon)
        self.splitter = ChronologicalSplitter(purge_horizon=horizon)
        self.preprocessor = StandardFeaturePipeline(self.feature_names, enabled=standardize)
        self.model = model or LinearRegressionModel()
        self.evaluator = RegressionEvaluator()
        self.horizon = horizon
        self.factor_selector = factor_selector
        self.reweighter = reweighter

    def run(self, observations: Sequence[FactorObservation]) -> TrainingResult:
        labeled = self.builder.build(observations)
        raw_split = self.splitter.split(labeled)
        factor_selection = None
        if self.factor_selector is not None:
            train_end = raw_split.train["datetime"].max()
            factor_selection = self.factor_selector.select(
                observations,
                self.feature_names,
                train_end,
            )
            self.feature_names = factor_selection.selected_features
            self.preprocessor = StandardFeaturePipeline(
                self.feature_names,
                enabled=self.preprocessor.enabled,
            )
            selected_columns = [
                "datetime",
                "vt_symbol",
                *self.feature_names,
                "label",
            ]
            raw_split = DatasetSplit(
                raw_split.train.select(selected_columns),
                raw_split.valid.select(selected_columns),
                raw_split.test.select(selected_columns),
            )
        self.preprocessor.fit(pl.concat([raw_split.train, raw_split.valid]))
        split = DatasetSplit(
            self.preprocessor.transform(raw_split.train),
            self.preprocessor.transform(raw_split.valid),
            self.preprocessor.transform(raw_split.test),
        )
        dataset = FrameDataset(split.train, split.valid, split.test)
        if self.reweighter is None:
            # Preserve compatibility with existing third-party AlphaModel
            # implementations that still expose fit(dataset).
            self.model.fit(dataset)  # type: ignore[arg-type]
        else:
            self.model.fit(dataset, self.reweighter)  # type: ignore[arg-type]
        test_predictions = self.model.predict(dataset, Segment.TEST)  # type: ignore[arg-type]
        metrics = self.evaluator.evaluate(split.test["label"].to_numpy(), test_predictions)
        artifact = ModelArtifact(
            model=self.model,
            preprocessor=self.preprocessor,
            feature_names=self.feature_names,
            label_horizon=self.horizon,
        )
        predictions = self._predict_latest(observations, artifact)
        return TrainingResult(artifact, split, metrics, predictions, factor_selection)

    def _predict_latest(
        self,
        observations: Sequence[FactorObservation],
        artifact: ModelArtifact,
    ) -> list[ModelPrediction]:
        latest_date = max(item.trade_date for item in observations)
        latest = sorted(
            (item for item in observations if item.trade_date == latest_date),
            key=lambda item: item.symbol,
        )
        frame = pl.DataFrame([
            {"datetime": item.trade_date, "vt_symbol": item.symbol, **item.features}
            for item in latest
        ])
        frame = artifact.preprocessor.transform(frame)
        fitted_model = getattr(artifact.model, "model", None)
        if fitted_model is None:
            raise ValueError("model is not fitted yet!")
        values = fitted_model.predict(frame.select(artifact.feature_names).to_numpy())
        ranked = sorted(
            zip((item.symbol for item in latest), values, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
        return [
            ModelPrediction(latest_date, symbol, float(value), rank)
            for rank, (symbol, value) in enumerate(ranked, start=1)
        ]
