from __future__ import annotations

import json
from datetime import date, timedelta

import polars as pl
import pytest

from vnpy.alpha.modeling import FactorObservation
from vnpy.quant import (
    ComponentRegistry,
    ComponentSpec,
    LinearTrainingPipeline,
    ModelBundle,
    PipelineConfig,
    RunStatus,
    SignalFrame,
)


def make_observations(days: int = 30) -> list[FactorObservation]:
    rows: list[FactorObservation] = []
    for offset in range(days):
        for symbol_index, symbol in enumerate(("AAA.SSE", "BBB.SSE")):
            slope = 1.0 + symbol_index * 0.2
            rows.append(FactorObservation(
                trade_date=date(2025, 1, 1) + timedelta(days=offset),
                symbol=symbol,
                close=100 + slope * offset,
                features={"momentum": slope * offset / 100, "volatility": 0.01 + offset / 1000},
            ))
    return rows


def make_config(tmp_path) -> PipelineConfig:
    return PipelineConfig.from_mapping({
        "experiment": {
            "name": "compatibility",
            "recorder": {"type": "local", "kwargs": {"uri": str(tmp_path)}},
        },
        "training": {
            "feature_names": ["momentum", "volatility"],
            "horizon": 3,
            "standardize": True,
        },
        "model": {"type": "linear", "kwargs": {}},
    })


def test_registry_only_creates_allowlisted_components() -> None:
    registry = ComponentRegistry()
    registry.register("model", "sample", lambda value=1: {"value": value})

    assert registry.create("model", ComponentSpec("sample", {"value": 2})) == {"value": 2}
    with pytest.raises(KeyError, match="unknown model component"):
        registry.create("model", ComponentSpec("module.Class"))


def test_pipeline_config_rejects_duplicate_features(tmp_path) -> None:
    value = make_config(tmp_path).as_dict()
    value["training"]["feature_names"] = ["same", "same"]

    with pytest.raises(ValueError, match="duplicates"):
        PipelineConfig.from_mapping(value)


def test_signal_frame_rejects_duplicate_identity() -> None:
    frame = pl.DataFrame({
        "datetime": [date(2025, 1, 1)] * 2,
        "vt_symbol": ["AAA.SSE"] * 2,
        "score": [0.1, 0.2],
        "rank": [1, 2],
        "model_id": ["model"] * 2,
        "horizon": [5] * 2,
        "generated_at": [date(2025, 1, 1)] * 2,
    })
    with pytest.raises(ValueError, match="duplicate"):
        SignalFrame(frame)


def test_compatible_training_pipeline_records_reproducible_artifacts(tmp_path) -> None:
    result = LinearTrainingPipeline(make_config(tmp_path)).run(make_observations())

    metadata = json.loads((result.run_path / "run.json").read_text(encoding="utf-8"))
    restored = ModelBundle.load(result.run_path / "artifacts" / "model_bundle.pkl")

    assert metadata["status"] == RunStatus.FINISHED.value
    assert metadata["metrics"]["mse"] == pytest.approx(result.metrics.mse)
    assert restored.model_id == result.bundle.model_id
    assert restored.data_fingerprint == result.bundle.data_fingerprint
    assert result.signals.frame.height == 2
    assert (result.run_path / "artifacts" / "signals.parquet").exists()
    assert (result.run_path / "artifacts" / "config.json").exists()


def test_failed_pipeline_marks_run_failed(tmp_path) -> None:
    config = make_config(tmp_path)
    broken = make_observations(days=2)

    with pytest.raises(ValueError):
        LinearTrainingPipeline(config).run(broken)

    run_path = next((tmp_path / "compatibility").iterdir())
    metadata = json.loads((run_path / "run.json").read_text(encoding="utf-8"))
    assert metadata["status"] == RunStatus.FAILED.value
    assert metadata["error"].startswith("ValueError:")
