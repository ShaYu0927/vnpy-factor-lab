from typing import List
from vnpy.common.logger import get_logger
from .Rule.Rule import RiskRule, RiskDecision, RiskAction


class RiskEngine:
    """
    Risk engine.

    It manages all risk rules and provides unified risk checking APIs.
    """

    def __init__(self):
        self.order_rules: List[RiskRule] = []
        self.position_rules: List[RiskRule] = []
        self.account_rules: List[RiskRule] = []
        self.logger = get_logger("risk.engine")

    def add_order_rule(self, rule: RiskRule) -> None:
        self.order_rules.append(rule)
        self.order_rules.sort(key=lambda r: r.priority)

    def add_position_rule(self, rule: RiskRule) -> None:
        self.position_rules.append(rule)
        self.position_rules.sort(key=lambda r: r.priority)

    def add_account_rule(self, rule: RiskRule) -> None:
        self.account_rules.append(rule)
        self.account_rules.sort(key=lambda r: r.priority)

    def check_order(self, order, context) -> RiskDecision:
        """
        Check order before sending it to gateway.
        """

        # 1. Account risk check first.
        account_decision = self.check_account(context)

        if account_decision.action not in (RiskAction.PASS, RiskAction.WARNING):
            return account_decision

        # 2. Order risk rules.
        for rule in self.order_rules:
            if not rule.enabled:
                continue

            decision = rule.check_order(order, context)

            if decision.action == RiskAction.PASS:
                continue

            if not decision.rule_name:
                decision.rule_name = rule.name

            # WARNING only logs, does not block order.
            if decision.action == RiskAction.WARNING:
                self._log_warning(order, decision)
                continue

            # REJECT / REDUCE directly returns.
            self._log_reject(order, decision)
            return decision

        return RiskDecision(action=RiskAction.PASS)

    def check_position(self, position, context) -> RiskDecision:
        """
        Check whether current position should be held or force closed.
        """
        for rule in self.position_rules:
            if not rule.enabled:
                continue

            decision = rule.check_position(position, context)

            if decision.action == RiskAction.PASS:
                continue

            if not decision.rule_name:
                decision.rule_name = rule.name

            if decision.action == RiskAction.WARNING:
                self._log_warning(position, decision)
                continue

            self._log_reject(position, decision)
            return decision

        return RiskDecision(action=RiskAction.PASS)

    def check_account(self, context) -> RiskDecision:
        """
        Check account-level risk.
        """
        for rule in self.account_rules:
            if not rule.enabled:
                continue

            decision = rule.check_account(context)

            if decision.action == RiskAction.PASS:
                continue

            if not decision.rule_name:
                decision.rule_name = rule.name

            if decision.action == RiskAction.WARNING:
                self._log_warning(getattr(context, "account", context), decision)
                continue

            self._log_reject(getattr(context, "account", context), decision)
            return decision

        return RiskDecision(action=RiskAction.PASS)

    def on_order(self, order) -> None:
        """
        Notify risk rules when an order is created or updated.
        """

        for rule in self._all_rules():
            rule.on_order(order)

    def on_trade(self, trade) -> None:
        """
        Notify risk rules when a trade is filled.
        """

        for rule in self._all_rules():
            rule.on_trade(trade)

    def on_bar(self, bar) -> None:
        """
        Notify risk rules when a new bar arrives.
        """

        for rule in self._all_rules():
            rule.on_bar(bar)

    def _all_rules(self) -> List[RiskRule]:
        return self.order_rules + self.position_rules + self.account_rules

    def _log_warning(self, target, decision: RiskDecision) -> None:
        self.logger.warning(
            "Risk warning: rule=%s action=%s reason=%s target=%s",
            decision.rule_name,
            decision.action.value,
            decision.reason,
            target,
        )

    def _log_reject(self, target, decision: RiskDecision) -> None:
        self.logger.warning(
            "Risk blocked: rule=%s action=%s reason=%s target=%s",
            decision.rule_name,
            decision.action.value,
            decision.reason,
            target,
        )
