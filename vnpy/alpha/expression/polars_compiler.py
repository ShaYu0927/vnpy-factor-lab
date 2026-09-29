"""将表达树编译为分阶段执行的 Polars 表达式。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping

import polars as pl

from .analyzer import ExpressionAnalysis, ExpressionAnalyzer
from .node import ConstantNode, FieldNode, Node, OperatorNode, WindowNode


@dataclass(frozen=True)
class CompiledExpression:
    """单个因子的编译结果；先计算 stages, 再计算 expression。"""

    expression: pl.Expr
    stages: tuple[pl.Expr, ...]
    analysis: ExpressionAnalysis


@dataclass(frozen=True)
class CompiledBatch:
    """多个因子的编译结果，可共享相同的中间计算阶段。"""

    expressions: Mapping[str, pl.Expr]
    stages: tuple[pl.Expr, ...]
    fields: frozenset[str]


class PolarsCompiler:
    """使用完整滚动窗口:Std 使用总体标准差(ddof=0)。"""

    def compile(self, tree: Node, available_fields: set[str] | None = None,) -> CompiledExpression:
        """编译单棵表达树，返回最终表达式、中间阶段和分析结果。"""
        analysis = ExpressionAnalyzer().analyze(tree, available_fields)

        # 复用批量编译逻辑，使用固定名称 data 取得编译结果。
        batch = self.compile_many({"data": tree}, available_fields)
        return CompiledExpression(batch.expressions["data"], batch.stages, analysis)

    def compile_many(self, trees: Mapping[str, Node], available_fields: set[str] | None = None,) -> CompiledBatch:
        """编译多棵表达树，并复用相同的窗口子表达式。"""
        # 分析所有表达树，汇总它们依赖的原始字段。
        fields = frozenset().union(
            *(
                ExpressionAnalyzer().analyze(tree, available_fields).fields
                for tree in trees.values()
            )
        )

        # stages 按依赖顺序保存中间计算结果。
        stages: list[pl.Expr] = []

        # 生成中间列名时，避开已使用的字段名、因子名和主键列名。
        used_names = set(fields) | set(trees) | {"datetime", "vt_symbol"}

        def materialize(expression: pl.Expr) -> pl.Expr:
            """将窗口表达式保存为中间阶段，并返回对中间列的引用。"""
            index = len(stages)
            name = f"__alpha_stage_{index}"

            while name in used_names:
                index += 1
                name = f"__alpha_stage_{index}"

            used_names.add(name)
            stages.append(expression.alias(name))
            return pl.col(name)

        @lru_cache(maxsize=None)
        def visit(node: Node) -> pl.Expr:
            """递归遍历表达树，将当前节点转换为 Polars 表达式。"""
            if isinstance(node, FieldNode):
                # 字段统一转为 Float64，并将 NaN 转为 null。
                return pl.col(node.name).cast(pl.Float64).fill_nan(None)

            if isinstance(node, (ConstantNode, WindowNode)):
                # 常量值和窗口大小都编译为字面量。
                return pl.lit(node.value)

            if not isinstance(node, OperatorNode):
                raise TypeError(f"unsupported expression node: {type(node).__name__}")

            # 先编译子节点，再根据当前算子组合表达式。
            args = [visit(child) for child in node.args]
            first = args[0]
            name = node.name

            # 普通算术算子直接组合子表达式。
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
                # 在同一个交易日内，对不同股票的值进行截面排名。
                return materialize(
                    first.rank(method="average").over("datetime")
                )

            # 时间序列算子的最后一个参数是窗口大小。
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
                expression = pl.rolling_corr(
                    first,
                    args[1],
                    window_size=window,
                    min_samples=window,
                )
            else:
                raise ValueError(f"operator has no Polars implementation: {name}")

            # 每只股票分别沿时间计算。结果先存入中间列，
            # 供外层算子引用，避免在一个表达式中嵌套窗口计算。
            return materialize(expression.over("vt_symbol"))

        # 同一次编译中的相同节点由 lru_cache 复用。
        expressions = {
            name: visit(tree)
            for name, tree in trees.items()
        }

        return CompiledBatch(expressions, tuple(stages), fields)