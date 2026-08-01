from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class AccountState:
    """
    Account risk state.
    """
    account_id: str
    total_value: float              # 账户总资产
    available_cash: float           # 可用资金
    total_position_value: float     # 总持仓市值
    daily_pnl: float = 0.0          # 当日盈亏
    max_drawdown: float = 0.0       # 当前最大回撤
    trading_enabled: bool = True    # 是否允许交易


@dataclass
class PositionState:
    """
    Position risk state for one symbol.
    """
    vt_symbol: str
    volume: int
    cost_price: float
    last_price: float
    market_value: float
    highest_price: float = 0.0
    holding_bars: int = 0


@dataclass
class OrderRequest:
    """
    Order request before sending to gateway.
    """
    vt_symbol: str
    direction: str       # "buy" / "sell"
    price: float
    volume: int
    reason: str = ""


@dataclass
class BarData:
    """
    Simplified bar data for risk checking.
    """
    vt_symbol: str
    datetime: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    
    
@dataclass
class RiskContext:
    """
    Risk checking context.

    This object provides all runtime state needed by risk rules.
    It should be treated as a read-only snapshot during risk checking.
    """

    account: AccountState
    positions: Dict[str, PositionState] = field(default_factory=dict)
    active_orders: List[OrderRequest] = field(default_factory=list)
    bars: Dict[str, BarData] = field(default_factory=dict)

    current_time: Optional[datetime] = None
    trading_day: str = ""

    def get_position(self, vt_symbol: str) -> Optional[PositionState]:
        """
        Get current position by symbol.
        """
        return self.positions.get(vt_symbol)

    def get_bar(self, vt_symbol: str) -> Optional[BarData]:
        """
        Get latest bar by symbol.
        """
        return self.bars.get(vt_symbol)

    def get_active_orders(self, vt_symbol: Optional[str] = None) -> List[OrderRequest]:
        """
        Get active orders. If vt_symbol is provided, only return active orders of that symbol.
        """
        if vt_symbol is None:
            return self.active_orders

        return [
            order for order in self.active_orders
            if order.vt_symbol == vt_symbol
        ]

@dataclass
class RiskConfig:
    """
    Risk control configuration.
    """

    max_order_ratio: float = 0.10
    max_symbol_position_ratio: float = 0.20
    max_total_position_ratio: float = 0.80

    max_daily_loss_ratio: float = 0.03
    max_drawdown_ratio: float = 0.10

    stop_loss_ratio: float = 0.05
    take_profit_ratio: float = 0.10

    max_active_orders: int = 20
    max_active_orders_per_symbol: int = 1