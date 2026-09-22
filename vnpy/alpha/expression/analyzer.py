from __future__ import annotations

from dataclasses import dataclass

from .node import ConstantNode, FieldNode, Node, OperatorNode, WindowNode
from .operator import (
    ArgumentKind,
    LookbackRule,
    OperatorCategory,
    OperatorRegistry,
    create_default_registry,
)


@dataclass(frozen=True, slots=True)
class ExpressionAnalysis:
    """Static properties derived from an expression tree."""

    fields: frozenset[str]
    lookback: int
    uses_cross_section: bool


class ExpressionAnalyzer:
    """Validate an expression tree and derive its data requirements."""

    def __init__(self, registry: OperatorRegistry | None = None) -> None:
        self.registry = registry or create_default_registry()

    def analyze(
        self,
        node: Node,
        available_fields: set[str] | None = None,
    ) -> ExpressionAnalysis:
        analysis = self._analyze_node(node, available_fields)
        if not analysis.fields:
            raise ValueError("factor expression must reference at least one field")
        return analysis

    def _analyze_node(
        self,
        node: Node,
        available_fields: set[str] | None,
    ) -> ExpressionAnalysis:
        if isinstance(node, FieldNode):
            if node.name.lower() in {"label", "target", "future", "future_return", "future_ret"}:
                raise ValueError(f"alpha expressions cannot reference future labels: {node.name}")
            if node.name in {"datetime", "vt_symbol"}:
                raise ValueError(f"index column cannot be used as a numeric field: {node.name}")
            if available_fields is not None and node.name not in available_fields:
                raise ValueError(f"unknown field: {node.name}")
            return ExpressionAnalysis(frozenset({node.name}), 1, False)

        if isinstance(node, (ConstantNode, WindowNode)):
            return ExpressionAnalysis(frozenset(), 0, False)

        if not isinstance(node, OperatorNode):
            raise TypeError(f"unsupported expression node: {type(node).__name__}")

        spec = self.registry.get(node.name)
        spec.validate_arity(len(node.args))
        kinds = spec.argument_kinds or (
            ArgumentKind.EXPRESSION,
        ) * len(node.args)
        self._validate_arguments(node, kinds)

        child_analyses = []
        for index, (child, kind) in enumerate(zip(node.args, kinds), start=1):
            if kind not in {ArgumentKind.EXPRESSION, ArgumentKind.VALUE}:
                continue
            child_analysis = self._analyze_node(child, available_fields)
            if kind is ArgumentKind.EXPRESSION and not child_analysis.fields:
                raise TypeError(
                    f"operator {node.name!r} argument {index} must depend on a field"
                )
            child_analyses.append(child_analysis)
        fields = frozenset(
            field
            for analysis in child_analyses
            for field in analysis.fields
        )
        base_lookback = max(
            (analysis.lookback for analysis in child_analyses),
            default=0,
        )
        uses_cross_section = (
            spec.category is OperatorCategory.CROSS_SECTION
            or any(analysis.uses_cross_section for analysis in child_analyses)
        )

        if spec.lookback_rule is LookbackRule.SAME:
            lookback = base_lookback
        else:
            window = self._window_value(node.args[-1])
            extension = window if spec.lookback_rule is LookbackRule.DELAY else window - 1
            lookback = base_lookback + extension

        return ExpressionAnalysis(fields, lookback, uses_cross_section)

    @staticmethod
    def _validate_arguments(
        node: OperatorNode,
        kinds: tuple[ArgumentKind, ...],
    ) -> None:
        if not kinds:
            return

        for index, (argument, kind) in enumerate(zip(node.args, kinds), start=1):
            if kind is ArgumentKind.WINDOW:
                try:
                    ExpressionAnalyzer._window_value(argument)
                except (TypeError, ValueError) as exc:
                    raise type(exc)(
                        f"operator {node.name!r} argument {index}: {exc}"
                    ) from exc
            elif isinstance(argument, WindowNode):
                raise TypeError(
                    f"operator {node.name!r} argument {index} must be a value expression"
                )
            elif kind is ArgumentKind.EXPRESSION and isinstance(argument, ConstantNode):
                raise TypeError(
                    f"operator {node.name!r} argument {index} must depend on a field"
                )

    @staticmethod
    def _window_value(node: Node) -> int:
        if isinstance(node, WindowNode):
            return node.value
        if isinstance(node, ConstantNode):
            value = node.value
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
        raise TypeError("window must be a positive integer constant")
