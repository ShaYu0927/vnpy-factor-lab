from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional
import polars as pl
from vnpy.trader.object import BarData


@dataclass
class FactorResult:
    """
    单个因子的计算结果
    name:
        因子名称，例如 ma_trend、volume_price、macd
    raw_value:
        原始因子值:例如涨幅、成交量放大倍数、MACD值
    score:
        标准化后的因子分数：建议统一为 0.0 ~ 1.0
    """

    name: str
    raw_value: float
    score: float
    reason: str = ""


class BaseFactor(ABC):
    """
    因子基类
    """

    name: str = ""

    @abstractmethod
    def calculate(self, bars: List[BarData]) -> FactorResult:
        pass


@dataclass
class FactorScoreDetail:
    """
    因子加权后的明细
    """
    name: str
    raw_value: float
    score: float
    weight: float
    weighted_score: float
    reason: str = ""


@dataclass
class StockScoreResult:
    """
    单只股票的综合评分结果
    """
    vt_symbol: str                          # stock
    signal: float                           # Final composite score used for ranking and stock selection.
    details: Dict[str, FactorScoreDetail]   # Factor-level scoring details, keyed by factor name.


class FactorScorer:
    """
    多因子打分器

    设计原则：
    1. 单个因子负责输出 0.0 ~ 1.0 的 score
    2. FactorScorer 负责按权重加权
    3. 最终生成 signal
    4. signal 越高，股票越值得买入
    """
    def __init__(self, factors: List[BaseFactor], weights: Dict[str, float], min_score: float = 0.0, normalize_weight: bool = False,) -> None:
        """
        Parameters

        weights:
            因子权重，例如：
            {
                "ma_trend": 25,
                "volume_price": 25,
                "macd": 25,
                "amount": 25,
            }

        min_score:
            最低入选分数

        normalize_weight:
            是否把权重归一化到 100 分
        """

        self.factors = factors
        self.weights = weights
        self.min_score = min_score
        self.normalize_weight = normalize_weight

        self._check_config()

    def _check_config(self) -> None:
        """
        检查因子和权重配置
        """
        factor_names = {factor.name for factor in self.factors}

        for name in factor_names:
            if name not in self.weights:
                raise ValueError(f"factor weight missing: {name}")

        for name in self.weights:
            if name not in factor_names:
                raise ValueError(f"weight config has unknown factor: {name}")

    def score_one(self, vt_symbol: str, bars: List[BarData]) -> Optional[StockScoreResult]:
        """
        计算单只股票的综合得分
        """
        total_score = 0.0
        details: Dict[str, FactorScoreDetail] = {}

        weight_sum = sum(self.weights.values())

        if weight_sum <= 0:
            raise ValueError("factor weight sum must be positive")

        for factor in self.factors:
            result = factor.calculate(bars)

            # 防御性处理，避免因子返回异常分数
            score = self._clip_score(result.score)

            weight = self.weights[result.name]

            if self.normalize_weight:
                weight = weight / weight_sum * 100

            weighted_score = score * weight
            total_score += weighted_score

            details[result.name] = FactorScoreDetail(
                name=result.name,
                raw_value=result.raw_value,
                score=score,
                weight=weight,
                weighted_score=weighted_score,
                reason=result.reason,
            )

        if total_score < self.min_score:
            return None

        return StockScoreResult(
            vt_symbol=vt_symbol,
            signal=total_score,
            details=details,
        )

    def score_many(self, history_bars: Dict[str, List[BarData]]) -> pl.DataFrame:
        """
        计算多只股票的综合得分，并返回 Polars DataFrame。

        返回字段至少包含：
        - vt_symbol
        - signal

        这样可以直接接到策略里的 get_signal()
        """

        rows: list[dict] = []

        for vt_symbol, bars in history_bars.items():
            result = self.score_one(vt_symbol, bars)

            if result is None:
                continue

            row = {
                "vt_symbol": result.vt_symbol,
                "signal": result.signal,
            }

            for factor_name, detail in result.details.items():
                row[f"{factor_name}_score"] = detail.score
                row[f"{factor_name}_weight"] = detail.weight
                row[f"{factor_name}_weighted_score"] = detail.weighted_score
                row[f"{factor_name}_raw"] = detail.raw_value
                row[f"{factor_name}_reason"] = detail.reason

            rows.append(row)

        if not rows:
            return pl.DataFrame(
                {
                    "vt_symbol": [],
                    "signal": [],
                }
            )

        return pl.DataFrame(rows).sort("signal", descending=True)

    @staticmethod
    def _clip_score(score: float) -> float:
        """
        将因子分数限制在 0.0 ~ 1.0
        """

        if score < 0:
            return 0.0

        if score > 1:
            return 1.0

        return score