import csv
from typing import Dict, List, Optional
from typing import Any

from .factorEngine import Factor, FactorContext
from vnpy.factor.momentum import MomentumFactor
from vnpy.factor.volatility import VolatilityFactor
from vnpy.factor.volume import VolumeFactor
from dataclasses import dataclass
from datetime import datetime

from collections import defaultdict
from pathlib import Path
from typing import Dict, List


@dataclass(slots=True)
class BarData:
    """
    K线数据结构。
    """

    symbol: str
    datetime: datetime
    frequency: str

    open: float
    high: float
    low: float
    close: float
    preclose: float

    volume: float
    amount: float

    adjustflag: int
    turn: float
    tradestatus: int
    pct_chg: float

    pe_ttm: float
    pb_mrq: float
    ps_ttm: float
    pcf_ncf_ttm: float

    is_st: int
    
def _to_float(value: str, default: float = 0.0) -> float:
    """
    Convert csv string value to float safely.
    """
    if value is None or value == "":
        return default
    return float(value)


def _to_int(value: str, default: int = 0) -> int:
    """
    Convert csv string value to int safely.
    """
    if value is None or value == "":
        return default
    return int(float(value))

class FactorDataBuilder:
    """
    因子数据构建器

    职责：
    1. 从 BarCache 批量读取历史K线
    2. 构造 FactorEngine 需要的 symbol_data_map
    3. 避免每个因子重复读取 BarCache。
    """

    def __init__(self, bar_cache):
        self.bar_cache = bar_cache

    def build_symbol_data_map(self, symbols: list[str], frequency: str = "60s", count: int = 21,) -> Dict[str, List]:
        symbol_data_map = {}

        for symbol in symbols:
            history_bars = self.bar_cache.get_bars(symbol=symbol, frequency=frequency, count=count,)

            if len(history_bars) < count:
                continue

            symbol_data_map[symbol] = history_bars

        return symbol_data_map
    

    
    def load_bars_from_csv(self, file_path: str | Path, frequency: str = "1d", only_trading: bool = True, exclude_st: bool = False,) -> List[BarData]:
        """
        从 Baostock CSV 文件读取 K 线数据
            date,code,open,high,low,close,preclose,volume,amount,
            adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,
            pcfNcfTTM,isST
        """

        file_path = Path(file_path)
        bars: List[BarData] = []

        with file_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:
                trade_status = _to_int(row.get("tradestatus", "0"))
                is_st = _to_int(row.get("isST", "0"))

                if only_trading and trade_status != 1:
                    continue

                if exclude_st and is_st == 1:
                    continue

                bar = BarData(
                    symbol=row["code"],
                    datetime=datetime.strptime(row["date"], "%Y-%m-%d"),
                    frequency=frequency,

                    open=_to_float(row.get("open")),
                    high=_to_float(row.get("high")),
                    low=_to_float(row.get("low")),
                    close=_to_float(row.get("close")),
                    preclose=_to_float(row.get("preclose")),

                    volume=_to_float(row.get("volume")),
                    amount=_to_float(row.get("amount")),

                    adjustflag=_to_int(row.get("adjustflag")),
                    turn=_to_float(row.get("turn")),
                    tradestatus=trade_status,
                    pct_chg=_to_float(row.get("pctChg")),

                    pe_ttm=_to_float(row.get("peTTM")),
                    pb_mrq=_to_float(row.get("pbMRQ")),
                    ps_ttm=_to_float(row.get("psTTM")),
                    pcf_ncf_ttm=_to_float(row.get("pcfNcfTTM")),

                    is_st=is_st,
                )

                bars.append(bar)

        bars.sort(key=lambda x: x.datetime)
        return bars
    
    def build_symbol_data_map_from_csv(self, file_path: str | Path, frequency: str = "1d", count: Optional[int] = None, only_trading: bool = True, exclude_st: bool = False,) -> Dict[str, List[BarData]]:
        """
         从 CSV 文件构造 FactorEngine 输入数据。

        返回格式：
            {
                "sh.600004": [bar1, bar2, bar3, ...],
                "sh.600000": [bar1, bar2, bar3, ...],
            }
        """
 
        bars = self.load_bars_from_csv(file_path=file_path, frequency=frequency, only_trading=only_trading, exclude_st=exclude_st,)

        symbol_data_map: Dict[str, List[BarData]] = defaultdict(list)

        for bar in bars:
            symbol_data_map[bar.symbol].append(bar)

        result: Dict[str, List[BarData]] = {}

        for symbol, symbol_bars in symbol_data_map.items():
            symbol_bars.sort(key=lambda x: x.datetime)

            if count is not None:
                if len(symbol_bars) < count:
                    continue

                symbol_bars = symbol_bars[-count:]

            result[symbol] = symbol_bars

        return result
    
class BasicMomentumEngineFactor(Factor):
    """
    适配 FactorEngine 的动量因子
    """

    def __init__(
        self,
        window: int = 20,
        up_threshold: float = 0.002,
        down_threshold: float = -0.002,
    ):
        self.window = window
        self.name = f"momentum_{window}"
        self.min_bars = window + 1

        self.factor = MomentumFactor(
            window=window,
            up_threshold=up_threshold,
            down_threshold=down_threshold,
        )

    def validate(self, symbol: str, data: Any, context: FactorContext) -> bool:
        return data is not None and len(data) >= self.min_bars

    def calculate(self, symbol: str, data: Any, context: FactorContext):
        closes = [bar.close for bar in data]

        if len(closes) < self.min_bars:
            return None

        return self.factor.calculate(symbol, closes)
    
    
class BasicVolatilityEngineFactor(Factor):
    """
    适配 FactorEngine 的波动率因子
    """

    def __init__(self, window: int = 20):
        self.window = window
        self.name = f"volatility_{window}"
        self.min_bars = window + 1

        self.factor = VolatilityFactor(window=window)

    def validate(self, symbol: str, data: Any, context: FactorContext) -> bool:
        return data is not None and len(data) >= self.min_bars

    def calculate(self, symbol: str, data: Any, context: FactorContext):
        closes = [bar.close for bar in data]

        if len(closes) < self.min_bars:
            return None

        return self.factor.calculate(symbol, closes)
    
class BasicVolumeEngineFactor(Factor):
    """
    适配 FactorEngine 的成交量因子
    """

    def __init__(self, window: int = 20):
        self.window = window
        self.name = f"volume_{window}"
        self.min_bars = window + 1

        self.factor = VolumeFactor(window=window)

    def validate(self, symbol: str, data: Any, context: FactorContext) -> bool:
        return data is not None and len(data) >= self.min_bars

    def calculate(self, symbol: str, data: Any, context: FactorContext):
        closes = [bar.close for bar in data]
        volumes = [bar.volume for bar in data]

        if len(closes) < self.min_bars or len(volumes) < self.min_bars:
            return None

        return self.factor.calculate(symbol, closes, volumes)
