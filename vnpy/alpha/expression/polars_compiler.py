"""Compile expression trees into native, staged Polars expressions."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from .analyzer import ExpressionAnalysis, ExpressionAnalyzer
from .node import ConstantNode, FieldNode, Node, OperatorNode, WindowNode


@dataclass(frozen=True)
class CompiledExpression:
    """Window stages are evaluated in order before the final expression."""

    expression: pl.Expr
    stages: tuple[pl.Expr, ...]
    analysis: ExpressionAnalysis


class PolarsCompiler:
    """Use full rolling windows; Std is population standard deviation (ddof=0)."""

    def compile(self, tree: Node, available_fields: set[str] | None = None,) -> CompiledExpression:
        analysis = ExpressionAnalyzer().analyze(tree, available_fields)
        stages: list[pl.Expr] = []
        used_names = set(analysis.fields) | {"datetime", "vt_symbol"}

        def materialize(expression: pl.Expr) -> pl.Expr:
            index = len(stages)
            name = f"__alpha_stage_{index}"
            while name in used_names:
                index += 1
                name = f"__alpha_stage_{index}"
            used_names.add(name)
            stages.append(expression.alias(name))
            return pl.col(name)

        def visit(node: Node) -> pl.Expr:
            if isinstance(node, FieldNode):
                return pl.col(node.name).cast(pl.Float64).fill_nan(None)
            if isinstance(node, (ConstantNode, WindowNode)):
                return pl.lit(node.value)
            if not isinstance(node, OperatorNode):
                raise TypeError(f"unsupported expression node: {type(node).__name__}")

            args = [visit(child) for child in node.args]
            first = args[0]
            name = node.name
            if name == "add":
                return first + args[1]
            if name == "sub":
                return first - args[1]
            if name == "mul":
                return first * args[1]
            if name == "div":
                return first / args[1]
            if name == "pow":
                return first.pow(args[1])
            if name == "abs":
                return first.abs()
            if name == "sign":
                return first.sign()
            if name == "log":
                return first.log()
            if name == "cs_rank":
                return materialize(first.rank(method="average").over("datetime"))

            window = ExpressionAnalyzer._window_value(node.args[-1])
            if name == "ts_delay":
                expression = first.shift(window)
            elif name == "ts_delta":
                expression = first - first.shift(window)
            elif name == "ts_mean":
                expression = first.rolling_mean(window, min_samples=window)
            elif name == "ts_sum":
                expression = first.rolling_sum(window, min_samples=window)
            elif name == "ts_std":
                expression = first.rolling_std(window, min_samples=window, ddof=0)
            elif name == "ts_min":
                expression = first.rolling_min(window, min_samples=window)
            elif name == "ts_max":
                expression = first.rolling_max(window, min_samples=window)
            elif name == "ts_corr":
                expression = pl.rolling_corr(first, args[1], window_size=window, min_samples=window)
            else:
                raise ValueError(f"operator has no Polars implementation: {name}")
            # Materialize each window so nested time-series/cross-section windows
            # never become illegal nested .over() expressions in Polars.
            return materialize(expression.over("vt_symbol"))

        expression = visit(tree)
        return CompiledExpression(expression, tuple(stages), analysis)
