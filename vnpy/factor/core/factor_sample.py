"""Compatibility layer for generic Alpha samples."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from vnpy.alpha.engine import AlphaSample, AlphaSampleCache
from .factor_engine import FactorBatchResult


FactorSample = AlphaSample
FastFactorSampleCache = AlphaSampleCache


class FactorSampleBuilder:
    """Build a generic feature map from normalized engine output."""

    @staticmethod
    def build(bar: Any, batch_result: FactorBatchResult) -> AlphaSample | None:
        values = batch_result.for_symbol(bar.symbol)
        features = {
            name: float(value.value)
            for name, value in values.items()
            if value.is_ready and value.value is not None
        }
        if not features:
            return None
        timestamp = (
            getattr(bar, "datetime", None)
            or getattr(bar, "bob", None)
            or getattr(bar, "eob", None)
        )
        if not isinstance(timestamp, datetime):
            raise ValueError("bar must contain a datetime/bob/eob datetime")
        return AlphaSample(
            symbol=bar.symbol,
            datetime=timestamp,
            close=float(bar.close),
            features=features,
        )


__all__ = ["FactorSample", "FactorSampleBuilder", "FastFactorSampleCache"]
