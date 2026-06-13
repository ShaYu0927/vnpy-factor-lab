from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Deque, Dict, List, Optional
from typing import Optional
import numpy as np


@dataclass
class FactorSample:
    """
    A factor sample generated from one K-line bar.
    """

    # Basic information
    symbol: str                 # 股票代码，例如：SZSE.000858
    datetime: str               # K线时间，例如：2026-05-01 10:01:00
    close: float                # 当前K线收盘价

    # Momentum factor
    momentum: float             # 动量因子，通常表示当前价格相对N周期前价格的涨跌幅
    strength: float             # 动量强度，用于描述上涨或下跌的强弱程度
    trend: str                  # 趋势方向，例如：UP / DOWN / FLAT

    # Volume factor
    volatility: float
    latest_volume: float        # 最新一根K线成交量
    volume_ma5: float           # 最近5根K线平均成交量
    volume_ma10: float          # 最近10根K线平均成交量
    volume_ratio: float         # 量比，当前成交量 / 最近N根K线平均成交量
    volume_change: float        # 成交量变化率，当前成交量相对上一根K线的变化幅度
    volume_level: str           # 成交量级别，例如：HIGH_VOLUME / LOW_VOLUME / NORMAL_VOLUME
    price_volume_signal: str    # 量价关系信号，例如：价涨量增、价跌量增、价涨量缩等

    # Explanation
    momentum_reason: str        # 动量因子结果解释
    volume_reason: str          # 成交量因子结果解释

    # Label for machine learning
    future_ret_5: Optional[float] = None  # 未来5根K线收益率，作为机器学习标签，离线训练时再填充

    def to_dict(self) -> dict:
        return asdict(self)




class FactorSampleBuilder:
    """
    因子样本构建器。

    作用：
    把当前K线 bar + 因子计算结果，组装成一条 FactorSample。
    """

    @staticmethod
    def build(bar, momentum_result, volume_result, volatility_result=None,) -> Optional[FactorSample]:

        if momentum_result is None or volume_result is None:
            return None

        bar_time = (
            getattr(bar, "datetime", None)
            or getattr(bar, "bob", None)
            or getattr(bar, "eob", None)
            or getattr(bar, "time", None)
        )

        if bar_time is None:
            print("bar 没有时间字段:", getattr(bar, "__dict__", bar))
            return None

        sample = FactorSample(
            symbol=bar.symbol,
            datetime=str(bar_time),
            close=bar.close,

            # Momentum factor
            momentum=getattr(momentum_result, "ret_1", 0.0),
            strength=getattr(
                momentum_result,
                "strength",
                getattr(momentum_result, "momentum_strength", 0.0),
            ),
            trend=getattr(momentum_result, "trend", "UNKNOWN"),

            # Volume factor
            volatility=getattr(volatility_result, "volatility", 0.0),
            latest_volume=getattr(volume_result, "latest_volume", 0.0),
            volume_ma5=getattr(volume_result, "volume_ma5", 0.0),
            volume_ma10=getattr(volume_result, "volume_ma10", 0.0),
            volume_ratio=getattr(volume_result, "volume_ratio", 0.0),
            volume_change=getattr(volume_result, "volume_change", 0.0),
            volume_level=getattr(volume_result, "volume_level", "UNKNOWN"),
            price_volume_signal=getattr(volume_result, "price_volume_signal", "UNKNOWN"),

            # Explanation
            momentum_reason=getattr(momentum_result, "reason", ""),
            volume_reason=getattr(volume_result, "reason", ""),

            # Label 后面再补
            future_ret_5=None,
        )

        return sample

class FastFactorSampleCache:
    """
    实时因子样本缓存
    适合 on_bar 实时追加使用

    设计：
    - 每个 symbol 对应一个 deque
    - 每次只追加当前样本
    - 每次只给 horizon 根之前的样本补 future_ret_5
    - 不做全量扫描
    """

    def __init__(self, maxlen: Optional[int] = 30000, horizon: int = 5):
        """
        :param maxlen: 每只股票最多缓存多少条样本
        :param horizon: 未来多少根K线收益率: 例如 5 表示 future_ret_5
        """
        self.maxlen = maxlen
        self.horizon = horizon

        self._samples: Dict[str, Deque[FactorSample]] = defaultdict(lambda: deque(maxlen=maxlen))

    def add(self, sample: FactorSample) -> None:
        """
        添加一条样本，并增量补标签
        """
        samples = self._samples[sample.symbol]

        # 如果最新一条时间相同，认为是同一根K线重复推送，直接覆盖
        if samples and samples[-1].datetime == sample.datetime:
            samples[-1] = sample
            return

        # 在 append 之前，用当前 close 给 horizon 根之前的样本打标签
        if len(samples) >= self.horizon:
            target_sample = samples[-self.horizon]

            if target_sample.future_ret_5 is None and target_sample.close > 0:
                target_sample.future_ret_5 = sample.close / target_sample.close - 1.0

        # 添加当前样本
        samples.append(sample)

    def get_latest(self, symbol: str) -> Optional[FactorSample]:
        """
        获取某只股票最新样本
        """
        samples = self._samples.get(symbol)
        if not samples:
            return None
        return samples[-1]

    def get_recent(self, symbol: str, count: int) -> List[FactorSample]:
        """
        获取某只股票最近 count 条样本
        """
        if count <= 0:
            return []

        samples = self._samples.get(symbol)

        if not samples:
            return []

        return list(samples)[-count:]

    def get_all(self, symbol: str) -> List[FactorSample]:
        """
        获取某只股票当前缓存中的全部样本
        """
        return list(self._samples.get(symbol, []))

    def size(self, symbol: str) -> int:
        """
        获取某只股票样本数量
        """
        return len(self._samples.get(symbol, []))

    def total_size(self) -> int:
        """
        获取所有股票样本总数
        """
        return sum(len(samples) for samples in self._samples.values())

    def symbols(self) -> List[str]:
        """
        获取当前缓存中的所有股票代码
        """
        return list(self._samples.keys())

    def clear_symbol(self, symbol: str) -> None:
        """
        清空某只股票样本
        """
        if symbol in self._samples:
            del self._samples[symbol]

    def clear_all(self) -> None:
        """
        清空所有样本
        """
        self._samples.clear()

class LabelBuilder:
    @staticmethod
    def build_future_return_label(samples: List[FactorSample], horizon: int = 5,) -> List[FactorSample]:
        if len(samples) <= horizon:
            return []

        labeled_samples: List[FactorSample] = []

        for i in range(len(samples) - horizon):
            current = samples[i]
            future = samples[i + horizon]

            if current.close <= 0:
                continue

            future_ret = future.close / current.close - 1.0

            current.future_ret_5 = future_ret
            labeled_samples.append(current)

        return labeled_samples
    
class LinearFactorModel:
    """
    线性因子模型

    作用：
    使用当前因子预测未来收益率 future_ret_5
    """
    def __init__(self):
        from sklearn.linear_model import LinearRegression

        self.model = LinearRegression()

        self.feature_names = [
            "momentum",
            "strength",
            "volatility",
            "volume_ratio",
            "volume_change",
        ]

        self.trained = False
        
    def _build_xy(self, samples: List[FactorSample]):
        feature_rows = []
        labels = []

        for sample in samples:
            if sample.future_ret_5 is None:
                continue

            feature_rows.append(self._sample_to_features(sample))
            labels.append(sample.future_ret_5)

        if not feature_rows:
            raise ValueError("No trainable samples found. Build future_ret_5 labels first.")

        return np.array(feature_rows, dtype=float), np.array(labels, dtype=float)

    def train(self, samples: List[FactorSample]):
        x, y = self._build_xy(samples)

        self.model.fit(x, y)
        self.trained = True
        
        pred = self.model.predict(x)

        from sklearn.metrics import mean_absolute_error, mean_squared_error

        mse = mean_squared_error(y, pred)
        mae = mean_absolute_error(y, pred)

        print("========== 线性因子模型训练完成 ==========")
        print(f"训练样本数: {len(y)}")
        print(f"MSE: {mse:.8f}")
        print(f"MAE: {mae:.8f}")

        print("========== 因子权重 ==========")
        for name, coef in zip(self.feature_names, self.model.coef_):
            print(f"{name}: {coef:.8f}")

        print(f"bias: {self.model.intercept_:.8f}")
        
    def predict_one(self, sample: FactorSample) -> float:
        """
        预测单条样本的未来收益率。
        """
        if not self.trained:
            raise RuntimeError("模型还没有训练，请先调用 train()")

        x = np.array([[*self._sample_to_features(sample)]], dtype=float)

        pred = self.model.predict(x)

        return float(pred[0])
    
    def predict_signal(self, sample: FactorSample) -> str:
        """
        根据预测收益率生成交易信号。
        """

        pred = self.predict_one(sample)

        if pred > 0.005:
            return "BUY"
        elif pred < -0.005:
            return "SELL"
        else:
            return "HOLD"

    def _sample_to_features(self, sample: FactorSample) -> list[float]:
        return [
            sample.momentum,
            sample.strength,
            sample.volatility,
            sample.volume_ratio,
            sample.volume_change,
        ]
