from __future__ import annotations

import math
from typing import Iterable


class MathCommon:
    """
    数学公式通用基础工具
    """

    EPSILON = 1e-12

    @staticmethod
    def is_valid(value: float | int | None) -> bool:
        """
        判断数值是否合法
        """
        if value is None:
            return False

        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def clean(values: Iterable[float | int | None]) -> list[float]:
        """
        过滤 None / NaN / Inf
        """
        return [float(v) for v in values if MathCommon.is_valid(v)]

    @staticmethod
    def tail(values: list[float | int | None], count: int) -> list[float]:
        """
        获取尾部有效数据
        """
        if count <= 0:
            return []

        return MathCommon.clean(values[-count:])

    @staticmethod
    def safe_div(numerator: float | int | None, denominator: float | int | None, default: float = 0.0,) -> float:
        """
        安全除法
        """
        if not MathCommon.is_valid(numerator):
            return default

        if not MathCommon.is_valid(denominator):
            return default

        denominator = float(denominator)

        if abs(denominator) < MathCommon.EPSILON:
            return default

        return float(numerator) / denominator

    @staticmethod
    def clip(value: float, min_value: float, max_value: float) -> float:
        """
        数值截断
        """
        return max(min_value, min(value, max_value))

    @staticmethod
    def clean_pairs(x_values: list[float | int | None], y_values: list[float | int | None],) -> list[tuple[float, float]]:
        """
        清洗成对数据

        x 或 y 任意一个非法，则丢弃这一组。
        """
        pairs: list[tuple[float, float]] = []

        for x, y in zip(x_values, y_values):
            if MathCommon.is_valid(x) and MathCommon.is_valid(y):
                pairs.append((float(x), float(y)))

        return pairs