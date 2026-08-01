from __future__ import annotations

import math

from .common import MathCommon
from .stat import StatMath


class ReturnMath:
    """
    收益率、动量、波动率相关公式
    """

    @staticmethod
    def pct_change(current: float | int | None, previous: float | int | None, default: float = 0.0,) -> float:
        """
        普通收益率

        ret = current / previous - 1
        """
        if not MathCommon.is_valid(current) or not MathCommon.is_valid(previous):
            return default

        previous = float(previous)

        if abs(previous) < MathCommon.EPSILON:
            return default

        return float(current) / previous - 1.0

    @staticmethod
    def log_return(current: float | int | None, previous: float | int | None, default: float = 0.0,) -> float:
        """
        对数收益率

        ret = ln(current / previous)
        """
        if not MathCommon.is_valid(current) or not MathCommon.is_valid(previous):
            return default

        current = float(current)
        previous = float(previous)

        if current <= 0 or previous <= 0:
            return default

        return math.log(current / previous)

    @staticmethod
    def pct_returns(values: list[float | int | None]) -> list[float]:
        """
        普通收益率序列
        """
        clean_values = MathCommon.clean(values)

        if len(clean_values) < 2:
            return []

        returns: list[float] = []

        for i in range(1, len(clean_values)):
            returns.append(
                ReturnMath.pct_change(clean_values[i], clean_values[i - 1])
            )

        return returns

    @staticmethod
    def log_returns(values: list[float | int | None]) -> list[float]:
        """
        对数收益率序列
        """
        clean_values = MathCommon.clean(values)

        if len(clean_values) < 2:
            return []

        returns: list[float] = []

        for i in range(1, len(clean_values)):
            returns.append(
                ReturnMath.log_return(clean_values[i], clean_values[i - 1])
            )

        return returns

    @staticmethod
    def momentum(values: list[float | int | None], window: int, default: float = 0.0,) -> float:
        """
        动量

        momentum = latest / previous - 1
        """
        clean_values = MathCommon.clean(values)

        if len(clean_values) <= window:
            return default

        latest = clean_values[-1]
        previous = clean_values[-window - 1]

        return ReturnMath.pct_change(latest, previous, default=default)

    @staticmethod
    def volatility(values: list[float | int | None], window: int, use_log_return: bool = False, default: float = 0.0,) -> float:
        """
        波动率

        默认使用普通收益率标准差。
        """
        tail_values = MathCommon.tail(values, window + 1)

        if len(tail_values) < window + 1:
            return default

        if use_log_return:
            returns = ReturnMath.log_returns(tail_values)
        else:
            returns = ReturnMath.pct_returns(tail_values)

        return StatMath.std(returns, default=default)