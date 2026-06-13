from .Rule import RiskRule, RiskDecision, RiskAction


class ActiveOrderRule(RiskRule):
    """
    活动委托数量检查规则
    """

    name = "active_order_rule"
    priority = 20

    def __init__(self, active_order_limit: int = 50):
        self.active_order_limit = active_order_limit
        self.active_orders = {}
        self.active_order_count = 0

    def check_order(self, order, context):
        if self.active_order_count >= self.active_order_limit:
            return RiskDecision(
                action=RiskAction.REJECT,
                rule_name=self.name,
                reason=( f"active order count reached limit: " f"current={self.active_order_count}, " f"limit={self.active_order_limit}")
            )

        return RiskDecision(action=RiskAction.PASS)

    def on_order(self, order):
        order_id = getattr(order, "order_id", None) or getattr(order, "orderid", None)

        if not order_id:
            return

        status = getattr(order.status, "value", order.status)

        if status in {"submitting", "not_traded", "part_traded"}:
            self.active_orders[order_id] = order
        else:
            self.active_orders.pop(order_id, None)

        self.active_order_count = len(self.active_orders)
