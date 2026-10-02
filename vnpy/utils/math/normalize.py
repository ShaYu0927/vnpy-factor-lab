from __future__ import annotations

import math
from typing import Iterable

from .base import MathBase
from .stat import StatMath


class NormalizeMath(MathBase):
    """
    标准化、分位数、缩尾处理
    """

    @staticmethod
    def z_score(value: float | int | None, values: Iterable[float | int | None], default: float = 0.0,) -> float:
        """
        Z-Score

        z = (value - mean) / std
        """
        if not NormalizeMath.is_valid(value):
            return default

        clean_values = NormalizeMath.clean(values)

        if len(clean_values) < 2:
            return default

        avg = StatMath.mean(clean_values)
        std = StatMath.std(clean_values)

        return NormalizeMath.safe_div(float(value) - avg, std, default=default)

    @staticmethod
    def latest_z_score(values: list[float | int | None], window: int, default: float = 0.0,) -> float:
        """
        最新值在最近 window 个数据里的 Z-Score
        """
        tail_values = NormalizeMath.tail(values, window)

        if len(tail_values) < window:
            return default

        return NormalizeMath.z_score(tail_values[-1], tail_values, default=default)

    @staticmethod
    def percentile(values: Iterable[float | int | None], q: float, default: float = 0.0,) -> float:
        """
        分位数

        q 范围：[0, 1]
        """
        clean_values = sorted(NormalizeMath.clean(values))

        if not clean_values:
            return default

        q = NormalizeMath.clip(q, 0.0, 1.0)

        index = q * (len(clean_values) - 1)
        lower = int(math.floor(index))
        upper = int(math.ceil(index))

        if lower == upper:
            return clean_values[lower]

        weight = index - lower

        return clean_values[lower] * (1.0 - weight) + clean_values[upper] * weight

    @staticmethod
    def rank_pct(value: float | int | None, values: Iterable[float | int | None], default: float = 0.0,) -> float:
        """
        百分位排名

        返回范围：[0, 1]
        """
        if not NormalizeMath.is_valid(value):
            return default

        clean_values = NormalizeMath.clean(values)

        if not clean_values:
            return default

        value = float(value)

        less_equal_count = sum(1 for v in clean_values if v <= value)

        return NormalizeMath.safe_div(less_equal_count, len(clean_values), default=default,)

    @staticmethod
    def winsorize(values: Iterable[float | int | None], lower_q: float = 0.05, upper_q: float = 0.95,) -> list[float]:
        """
        缩尾处理

        用于降低极端值对因子的影响。
        """
        clean_values = NormalizeMath.clean(values)

        if not clean_values:
            return []

        lower = NormalizeMath.percentile(clean_values, lower_q)
        upper = NormalizeMath.percentile(clean_values, upper_q)

        return [NormalizeMath.clip(v, lower, upper) for v in clean_values]

    @staticmethod
    def min_max_scale(value: float, min_value: float, max_value: float, default: float = 0.0,) -> float:
        """
        Min-Max 归一化
        """
        return NormalizeMath.safe_div(value - min_value, max_value - min_value, default=default,)