from dataclasses import dataclass
import math
from typing import Optional


@dataclass
class VolatilityResult:
    """
    单只股票的波动率因子结果
    """
    symbol: str
    window: int
    ret_1: float = 0.0                  # 最新K线收益率
    mean_return: float = 0.0            # window 根K线收益率均值
    volatility: float = 0.0             # 最近 window 根K线收益率标准差，核心波动率因子
    price_range: float = 0.0            # 波动区间：最近 window 根K线最高价和最低价对应的价格振幅    
    volatility_level: str = "UNKNOWN"   # 波动等级：LOW / MEDIUM / HIGH
    reason: str = ""                    # 解释说明
    
class VolatilityFactor:
    """
    波动率因子
    核心思想:
        价格波动越剧烈，收益率序列的标准差越大
        价格越平稳，收益率序列的标准差越小
    """
    def __init__(self, window: int = 20, low_threshold: float = 0.005, high_threshold: float = 0.02):
        self.window = window
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        
    def calculate(self, symbol: str, closes: list[float]) -> VolatilityResult:
        result = VolatilityResult(symbol=symbol, window=self.window)
        if len(closes) < self.window + 1:
            result.reason = f"Insufficient data: at least {self.window + 1} close prices are required"
            return result
        
        recent_closes = closes[-(self.window + 1):]
        returns = self._calc_returns(recent_closes)
        if len(returns) < self.window:
            result.reason =  f"Insufficient valid returns: at least {self.window} returns are required"
            return result
        
        result.ret_1 = returns[-1]
        # 平均收盘价
        result.mean_return = sum(returns) / len(returns)
        
        result.volatility = self._calc_std(returns)
        min_close = min(recent_closes)
        max_close = max(recent_closes)
        
        if min_close > 0:
            result.price_range = max_close / min_close - 1
        else:
            result.price_range = 0.0

        if result.volatility >= self.high_threshold:
            result.volatility_level = "HIGH"
            result.reason = (f"The return standard deviation over the latest {self.window} bars,"f"is {result.volatility:.6f}, which is above the high-volatility,threshold {self.high_threshold:.6f}. Short-term volatility is high.")
        elif result.volatility <= self.low_threshold:
            result.volatility_level = "LOW"
            result.reason = (f"The return standard deviation over the latest {self.window} bars ,"f"is {result.volatility:.6f}, which is below the low-volatility threshold {self.low_threshold:.6f}. Short-term volatility is low.")
        else:
            result.volatility_level = "MEDIUM"
            result.reason = (f"最近{self.window}根K线收益率标准差为{result.volatility:.6f},"f"处于正常波动区间")
            
        return result
    
    def _calc_returns(self, closes: list[float]) -> list[float]:
        returns = []
        for i in range(1, len(closes)):
            prev_close = closes[i - 1]
            curr_close = closes[i]

            if prev_close <= 0:
                continue

            ret = curr_close / prev_close - 1
            returns.append(ret)

        return returns

        
    def _calc_std(self, values: list[float]) -> float:
        if not values:
            return 0.0

        mean_value = sum(values) / len(values)
        variance = sum((x - mean_value) ** 2 for x in values) / len(values)

        return math.sqrt(variance)