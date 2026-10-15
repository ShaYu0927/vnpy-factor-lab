"""Shared numeric operations for math formula classes."""

from __future__ import annotations

import math
from typing import Iterable


class MathBase:
    """Base class for stateless math formula collections."""

    EPSILON = 1e-12

    @staticmethod
    def is_valid(value: float | int | None) -> bool:
        """Return whether a value can be converted to a finite float."""
        if value is None:
            return False

        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    @classmethod
    def clean(cls, values: Iterable[float | int | None]) -> list[float]:
        """Discard None, NaN, and infinite values."""
        return [float(value) for value in values if cls.is_valid(value)]

    @classmethod
    def tail(cls, values: list[float | int | None], count: int) -> list[float]:
        """Return valid values from the last count input positions."""
        if count <= 0:
            return []

        return cls.clean(values[-count:])

    @classmethod
    def safe_div(cls, numerator: float | int | None, denominator: float | int | None, default: float = 0.0,) -> float:
        """Divide finite values, returning default for a near-zero denominator."""
        if not cls.is_valid(numerator) or not cls.is_valid(denominator):
            return default

        denominator = float(denominator)
        if abs(denominator) < cls.EPSILON:
            return default

        return float(numerator) / denominator

    @staticmethod
    def clip(value: float, min_value: float, max_value: float) -> float:
        """Constrain a value to the inclusive interval."""
        return max(min_value, min(value, max_value))

    @classmethod
    def clean_pairs(cls, x_values: list[float | int | None], y_values: list[float | int | None],) -> list[tuple[float, float]]:
        """Discard pairs with an invalid member."""
        pairs: list[tuple[float, float]] = []

        for x, y in zip(x_values, y_values):
            if cls.is_valid(x) and cls.is_valid(y):
                pairs.append((float(x), float(y)))

        return pairs
