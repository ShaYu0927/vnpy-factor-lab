from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import polars as pl

from vnpy.alpha.modeling import ModelArtifact, ModelPrediction


SIGNAL_COLUMNS = ("datetime", "vt_symbol", "score", "rank", "model_id", "horizon", "generated_at")


@dataclass(frozen=True, slots=True)
class SignalFrame:
    """Validated, indexed model scores passed to portfolio strategies."""

    frame: pl.DataFrame

    def __post_init__(self) -> None:
        missing = set(SIGNAL_COLUMNS) - set(self.frame.columns)
        if missing:
            raise ValueError(f"signal frame is missing columns: {sorted(missing)}")
        duplicate_count = (
            self.frame.group_by(["datetime", "vt_symbol", "model_id"])
            .len()
            .filter(pl.col("len") > 1)
            .height
        )
        if duplicate_count:
            raise ValueError("signal frame contains duplicate datetime/symbol/model rows")
        if self.frame["score"].is_null().any():
            raise ValueError("signal scores must not be null")

    @classmethod
    def from_predictions(
        cls,
        predictions: Iterable[ModelPrediction],
        model_id: str,
        horizon: int,
        generated_at: datetime | None = None,
    ) -> SignalFrame:
        if horizon <= 0:
            raise ValueError("horizon must be greater than zero")
        timestamp = generated_at or datetime.now(timezone.utc)
        rows = [
            {
                "datetime": item.trade_date,
                "vt_symbol": item.symbol,
                "score": item.predicted_return,
                "rank": item.rank,
                "model_id": model_id,
                "horizon": horizon,
                "generated_at": timestamp,
            }
            for item in predictions
        ]
        schema = {
            "datetime": pl.Date,
            "vt_symbol": pl.String,
            "score": pl.Float64,
            "rank": pl.Int64,
            "model_id": pl.String,
            "horizon": pl.Int64,
            "generated_at": pl.Datetime(time_zone="UTC"),
        }
        return cls(pl.DataFrame(rows, schema=schema).select(list(SIGNAL_COLUMNS)))

    def write_parquet(self, path: str | Path) -> None:
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        self.frame.write_parquet(output)


@dataclass(slots=True)
class ModelBundle:
    """Model artifact plus the metadata required to reproduce inference."""

    artifact: ModelArtifact
    model_id: str
    trained_at: datetime
    data_fingerprint: str
    metadata: dict[str, Any] = field(default_factory=dict)
    format_version: int = 1

    @classmethod
    def create(
        cls,
        artifact: ModelArtifact,
        data_fingerprint: str,
        metadata: dict[str, Any] | None = None,
    ) -> ModelBundle:
        trained_at = datetime.now(timezone.utc)
        identity = json.dumps({
            "features": artifact.feature_names,
            "horizon": artifact.label_horizon,
            "data": data_fingerprint,
            "trained_at": trained_at.isoformat(),
        }, sort_keys=True).encode()
        model_id = hashlib.sha256(identity).hexdigest()[:16]
        return cls(artifact, model_id, trained_at, data_fingerprint, metadata or {})

    def save(self, path: str | Path) -> None:
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: str | Path) -> ModelBundle:
        with Path(path).expanduser().resolve().open("rb") as file:
            value = pickle.load(file)
        if not isinstance(value, cls):
            raise TypeError("model file does not contain a ModelBundle")
        if value.format_version != 1:
            raise ValueError(f"unsupported ModelBundle version: {value.format_version}")
        return value


def fingerprint_observations(rows: Iterable[object]) -> str:
    """Produce a stable fingerprint without depending on object hash randomization."""
    digest = hashlib.sha256()
    for row in rows:
        trade_date = getattr(row, "trade_date")
        symbol = getattr(row, "symbol")
        close = getattr(row, "close")
        features = getattr(row, "features")
        if isinstance(trade_date, (date, datetime)):
            trade_date = trade_date.isoformat()
        payload = json.dumps(
            [trade_date, symbol, close, sorted(features.items())],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        digest.update(payload.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
