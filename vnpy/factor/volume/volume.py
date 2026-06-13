from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class VolumeFactorResult:
    """成交量因子计算结果"""

    symbol: str                 # 股票代码
    latest_volume: float        # 最新一根K线成交量
    volume_ma5: float           # 最近5根K线平均成交量
    volume_ma10: float          # 最近10根K线平均成交量
    volume_ratio: float         # 当前成交量 / 最近10根K线平均成交量
    volume_change: float        # 当前成交量相对上一根K线的变化率
    volume_level: str           # 成交量级别：放量 / 缩量 / 正常
    price_volume_signal: str    # 量价关系信号
    reason: str                 # 结果解释
    
class VolumeFactor:
    """
    成交量因子
    """
    
    def __init__(self, ma_short: int = 5, ma_long: int = 10, high_volume_threshold: float = 1.5, low_volume_threshold: float = 0.7,window: int = 5):
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.high_volume_threshold = high_volume_threshold
        self.low_volume_threshold = low_volume_threshold
        self.window = window
        
    def calculate(self, symbol: str, closes: list[float], volumes: list[float]) -> Optional[VolumeFactorResult]:
        """
        bars: 最近N根K线 要求至少包含 close 和 volume 字段
        """
        
        if any(v is None for v in closes) or any(v is None for v in volumes):
            return None
        
        closes = [float(v) for v in closes]
        volumes = [float(v) for v in volumes]
        
        latest_close = closes[-1]
        prev_close = closes[-2]
        
        latest_volume = volumes[-1]
        prev_volume = volumes[-2]
        
        volume_ma5 = self._mean(volumes[-self.ma_short:])
        volume_ma10 = self._mean(volumes[-self.ma_long:])
        
        if volume_ma10 <= 0:
            volume_ratio = 0.0
        else:
            volume_ratio = latest_volume / volume_ma10
            
        if prev_volume <= 0:
            volume_change = 0.0
        else:
            volume_change = latest_volume / prev_volume - 1
            
        price_change = latest_close / prev_close - 1 if prev_close > 0 else 0.0
        
        volume_level = self._judge_volume_level(volume_ratio)
        price_volume_signal, reason = self._judge_price_volume_signal(price_change=price_change,volume_ratio=volume_ratio,volume_change=volume_change)
        
        return VolumeFactorResult(
            symbol=symbol,
            latest_volume=latest_volume,
            volume_ma5=volume_ma5,
            volume_ma10=volume_ma10,
            volume_ratio=volume_ratio,
            volume_change=volume_change,
            volume_level=volume_level,
            price_volume_signal=price_volume_signal,
            reason=reason,
        )
        
    def _judge_volume_level(self, volume_ratio: float) -> str:
        """
        判断当前成交量状态
        """
        if volume_ratio >= self.high_volume_threshold:
            return "HIGH_VOLUME"
        if volume_ratio <= self.low_volume_threshold:
            return "LOW_VOLUME"
        return "NORMAL_VOLUME"
    
    def _judge_price_volume_signal(
        self,
        price_change: float,
        volume_ratio: float,
        volume_change: float,
    ) -> tuple[str, str]:
        """
        判断量价关系
        """
        is_price_up = price_change > 0
        is_price_down = price_change < 0
        is_high_volume = volume_ratio >= self.high_volume_threshold
        is_low_volume = volume_ratio <= self.low_volume_threshold

        if is_price_up and is_high_volume:
            return ("PRICE_UP_VOLUME_UP", "价格上涨且成交量放大，说明上涨有成交量配合，短线偏强")

        if is_price_down and is_high_volume:
            return ("PRICE_DOWN_VOLUME_UP", "价格下跌且成交量放大，说明下跌伴随明显成交，短线风险偏高")

        if is_price_up and is_low_volume:
            return ("PRICE_UP_VOLUME_LOW", "价格上涨但成交量偏低，说明上涨力度可能不足")

        if is_price_down and is_low_volume:
            return ("PRICE_DOWN_VOLUME_LOW", "价格下跌但成交量偏低，可能只是弱势缩量下跌")

        if volume_change > 1.0:
            return ("VOLUME_SPIKE", "当前成交量相比上一根K线明显放大,需要关注是否出现异动")

        return ("NORMAL", "当前量价关系暂未出现明显异常")

    def _mean(self, values: List[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _get_value(self, bar: Any, field: str):
        """
        兼容对象和字典两种K线结构
        """
        if isinstance(bar, dict):
            return bar.get(field)

        return getattr(bar, field, None)
