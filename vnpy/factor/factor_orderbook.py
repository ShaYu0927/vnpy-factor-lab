from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

@dataclass
class OrderBookSnapshot:
    symbol: str                     # Stock symbol
    datetime: datetime              # Snapshot timestamp

    bid_prices: List[float]         # Bid prices, level 1 to level 5
    ask_prices: List[float]         # Ask prices, level 1 to level 5
    bid_volumes: List[float]        # Bid volumes, level 1 to level 5
    ask_volumes: List[float]        # Ask volumes, level 1 to level 5

    volume: float = 0.0             # Traded volume in current interval
    amount: float = 0.0             # Traded amount in current interval

@dataclass
class OrderBookFactorResult:
    """
    Order book factor calculation result.
    """

    symbol: str                     # 股票代码
    datetime: datetime              # 因子时间

    voi1: Optional[float] = None    # 一档订单失衡因子
    voi2: Optional[float] = None    # 五档加权订单失衡因子
    oir: Optional[float] = None     # 订单失衡率因子
    mpb: Optional[float] = None     # 市价偏离度因子
    
    
class OrderBookFactorCalculator:
    """
    Order book factor calculator.

    It calculates:
        VOI1: level-1 volume order imbalance
        VOI2: weighted five-level volume order imbalance
        OIR : order imbalance ratio
        MPB : mid-price basis
    """
    def __init__(self) -> None:
        self._prev_snapshot: dict[str, OrderBookSnapshot] = {}
        self._prev_trade_price: dict[str, float] = {}
        
    def calculate(self, snapshot: OrderBookSnapshot) -> OrderBookFactorResult:
        """
        Calculate order book factors from one snapshot.
        """
        result = OrderBookFactorResult(symbol=snapshot.symbol, datetime=snapshot.datetime,)
        prev = self._prev_snapshot.get(snapshot.symbol)

        if prev is not None:
            result.voi1 = self._calc_voi1(snapshot, prev)
            result.voi2 = self._calc_voi2(snapshot, prev)

        result.oir = self._calc_oir(snapshot)
        result.mpb = self._calc_mpb(snapshot)

        self._prev_snapshot[snapshot.symbol] = snapshot

        return result
    
    def _calc_voi1(self, cur: OrderBookSnapshot, prev: OrderBookSnapshot,) -> float:
        pass
    
    def _calc_voi2(self, cur: OrderBookSnapshot, prev: OrderBookSnapshot,) -> float:
        pass
    
    def _calc_oir(self, snapshot: OrderBookSnapshot,) -> Optional[float]:
        pass
    
    def _calc_mpb(self, snapshot: OrderBookSnapshot,) -> Optional[float]:
        pass
