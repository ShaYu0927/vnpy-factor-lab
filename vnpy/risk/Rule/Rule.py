from enum import Enum
from dataclasses import dataclass
from typing import Optional

class RiskAction(Enum):
    PASS = "pass"              # 通过
    REJECT = "reject"          # 拒绝订单
    REDUCE = "reduce"          # 降低数量
    FORCE_SELL = "force_sell"  # 强制卖出
    WARNING = "warning"        # 只告警，不拦截

@dataclass
class RiskDecision:
    action: RiskAction
    reason: str = ""
    rule_name: str = ""
    adjusted_volume: Optional[int] = None

    @property
    def passed(self) -> bool:
        return self.action == RiskAction.PASS


class RiskRule:
    """
    Base class of all risk rules.
    """

    name = "base_rule"
    priority = 100
    enabled = True

    def check_order(self, order, context) -> RiskDecision:
        return RiskDecision(action=RiskAction.PASS)

    def check_position(self, position, context) -> RiskDecision:
        return RiskDecision(action=RiskAction.PASS)

    def check_account(self, context) -> RiskDecision:
        return RiskDecision(action=RiskAction.PASS)

    def on_order(self, order) -> None:
        """
        Optional event hook.
        """
        pass

    def on_trade(self, trade) -> None:
        """
        Optional event hook.
        """
        pass

    def on_bar(self, bar) -> None:
        """
        Optional event hook.
        """
        pass