from __future__ import annotations

import math
from typing import Iterable

from .common import MathCommon


class StatMath:
    """
    统计数学公式
    """
    @staticmethod
    def mean(values: Iterable[float | int | None], default: float = 0.0) -> float:
        clean_values = MathCommon.clean(values)

        if not clean_values:
            return default

        return sum(clean_values) / len(clean_values)

    @staticmethod
    def variance(values: Iterable[float | int | None], ddof: int = 0, default: float = 0.0,) -> float:
        clean_values = MathCommon.clean(values)
        n = len(clean_values)

        if n <= ddof:
            return default

        avg = StatMath.mean(clean_values)

        return sum((v - avg) ** 2 for v in clean_values) / (n - ddof)

    @staticmethod
    def std(values: Iterable[float | int | None], ddof: int = 0, default: float = 0.0,) -> float:
        var = StatMath.variance(values, ddof=ddof, default=default)

        if var < 0:
            return default

        return math.sqrt(var)

    @staticmethod
    def covariance(x_values: list[float | int | None], y_values: list[float | int | None], ddof: int = 0, default: float = 0.0,) -> float:
        pairs = MathCommon.clean_pairs(x_values, y_values)
        n = len(pairs)

        if n <= ddof:
            return default

        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]

        x_mean = StatMath.mean(xs)
        y_mean = StatMath.mean(ys)

        return sum((x - x_mean) * (y - y_mean) for x, y in pairs) / (n - ddof)

    @staticmethod
    def correlation(x_values: list[float | int | None], y_values: list[float | int | None], default: float = 0.0,) -> float:
        """
        皮尔逊相关系数
        """
        pairs = MathCommon.clean_pairs(x_values, y_values)

        if len(pairs) < 2:
            return default

        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]

        cov = StatMath.covariance(xs, ys)
        x_std = StatMath.std(xs)
        y_std = StatMath.std(ys)

        return MathCommon.safe_div(cov, x_std * y_std, default=default)

    @staticmethod
    def linear_slope(values: list[float | int | None], default: float = 0.0,) -> float:
        """
        线性回归斜率

        x 默认使用 0, 1, 2, ..., n-1
        """
        ys = MathCommon.clean(values)
        n = len(ys)

        if n < 2:
            return default

        xs = list(range(n))

        x_mean = StatMath.mean(xs)
        y_mean = StatMath.mean(ys)

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        denominator = sum((x - x_mean) ** 2 for x in xs)

        return MathCommon.safe_div(numerator, denominator, default=default)