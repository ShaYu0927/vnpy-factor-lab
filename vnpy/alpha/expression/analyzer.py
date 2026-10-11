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
    """表达树的静态分析结果。"""

    # 表达式依赖的行情字段，例如 {"close", "volume"}。
    fields: frozenset[str]

    # 计算一个有效结果所需的数据条数，包含当前这一条。
    # 例如 close 为 1，Mean(close, 5) 为 5，Ref(close, 1) 为 2。
    lookback: int

    # 是否包含截面运算，例如在同一时刻对不同股票进行排名。
    uses_cross_section: bool


class ExpressionAnalyzer:
    """检查表达树是否合法，并推导计算所需的数据条件。"""

    def __init__(self, registry: OperatorRegistry | None = None) -> None:
        # 算子注册表提供参数类型、参数数量、类别和回看规则。
        # 未传入时，使用默认注册表。
        self.registry = registry or create_default_registry()

    def analyze(self, node: Node, available_fields: set[str] | None = None,) -> ExpressionAnalysis:
        """分析整棵树；可选传入实际行情中可用的字段集合。"""
        analysis = self._analyze_node(node, available_fields)

        # 因子至少需要依赖一个行情字段，不能只是纯常量表达式。
        if not analysis.fields:
            raise ValueError(
                "factor expression must reference at least one field"
            )

        return analysis

    def _analyze_node(self, node: Node, available_fields: set[str] | None,) -> ExpressionAnalysis:
        """递归分析节点：先分析子节点，再汇总父节点的属性。"""

        if isinstance(node, FieldNode):
            # 禁止直接引用这些未来标签字段，降低未来信息泄漏风险。
            # 这是字段名检查，其他名称的未来数据仍需数据层保证不被输入。
            if node.name.lower() in {
                "label", "target", "future", "future_return", "future_ret"
            }:
                raise ValueError(f"alpha expressions cannot reference future labels: {node.name}")

            # 日期和股票代码用于定位数据，不能作为数值行情字段计算。
            if node.name in {"datetime", "vt_symbol"}:
                raise ValueError(f"index column cannot be used as a numeric field: {node.name}")

            # 如果提供了可用字段集合，检查字段是否存在。
            if (
                available_fields is not None
                and node.name not in available_fields
            ):
                raise ValueError(f"unknown field: {node.name}")

            # 普通字段只依赖自身，读取当前一条数据即可。
            return ExpressionAnalysis(
                frozenset({node.name}),
                1,
                False,
            )

        if isinstance(node, (ConstantNode, WindowNode)):
            # 常量和窗口参数自身不依赖行情，也不需要历史数据。
            return ExpressionAnalysis(frozenset(), 0, False)

        # 除字段、常量、窗口和算子之外，不支持其他节点类型。
        if not isinstance(node, OperatorNode):
            raise TypeError(
                f"unsupported expression node: {type(node).__name__}"
            )

        # 查找算子定义，并检查参数数量是否符合要求。
        spec = self.registry.get(node.name)
        spec.validate_arity(len(node.args))

        # 优先使用算子声明的参数类型。
        # 没有声明时，默认每个参数都必须是依赖行情字段的表达式。
        kinds = spec.argument_kinds or (
            ArgumentKind.EXPRESSION,
        ) * len(node.args)

        # 先检查窗口、常量等参数是否放在正确的位置。
        self._validate_arguments(node, kinds)

        child_analyses = []

        for index, (child, kind) in enumerate(
            zip(node.args, kinds),
            start=1,
        ):
            # 只有表达式和值参数需要递归分析。
            # 窗口参数已经校验，其数值稍后用于推导回看长度。
            if kind not in {
                ArgumentKind.EXPRESSION,
                ArgumentKind.VALUE,
            }:
                continue

            child_analysis = self._analyze_node(
                child,
                available_fields,
            )

            # EXPRESSION 参数必须直接或间接依赖行情字段。
            # 递归检查可以发现 Add(1, 2) 这样的纯常量子树。
            # VALUE 参数则允许不依赖字段的普通数值。
            if (
                kind is ArgumentKind.EXPRESSION
                and not child_analysis.fields
            ):
                raise TypeError(
                    f"operator {node.name!r} argument {index} "
                    "must depend on a field"
                )

            child_analyses.append(child_analysis)

        # 合并所有子表达式依赖的字段，并自动去重。
        fields = frozenset(
            field
            for analysis in child_analyses
            for field in analysis.fields
        )

        # 父节点至少要满足历史需求最大的那个子表达式。
        base_lookback = max(
            (analysis.lookback for analysis in child_analyses),
            default=0,
        )

        # 当前算子或任意子表达式包含截面运算，
        # 则整个表达式都需要截面数据。
        uses_cross_section = (
            spec.category is OperatorCategory.CROSS_SECTION
            or any(
                analysis.uses_cross_section
                for analysis in child_analyses
            )
        )

        if spec.lookback_rule is LookbackRule.SAME:
            # 普通算子不额外增加历史需求。
            # 例如 close / volume 仍然只需要当前一条数据。
            lookback = base_lookback
        else:
            # 此处约定：需要扩展历史的算子，最后一个参数是窗口值。
            window = self._window_value(node.args[-1])

            # 延迟 n 条，需要额外读取 n 条历史数据。
            # 滚动窗口 n 条包含当前条，因此额外增加 n-1 条。
            extension = (
                window
                if spec.lookback_rule is LookbackRule.DELAY
                else window - 1
            )

            # 嵌套表达式的历史需求会逐层累加。
            lookback = base_lookback + extension

        return ExpressionAnalysis(
            fields,
            lookback,
            uses_cross_section,
        )

    @staticmethod
    def _validate_arguments(node: OperatorNode, kinds: tuple[ArgumentKind, ...],) -> None:
        """检查各参数是否符合算子声明的类型要求。"""
        if not kinds:
            return

        for index, (argument, kind) in enumerate(
            zip(node.args, kinds),
            start=1,
        ):
            if kind is ArgumentKind.WINDOW:
                # 窗口参数必须能解析为正整数常量。
                try:
                    ExpressionAnalyzer._window_value(argument)
                except (TypeError, ValueError) as exc:
                    # 保留异常类型，并补充算子名和参数位置。
                    raise type(exc)(
                        f"operator {node.name!r} argument {index}: {exc}"
                    ) from exc

            elif isinstance(argument, WindowNode):
                # 专用窗口节点只能出现在 WINDOW 参数位置。
                raise TypeError(
                    f"operator {node.name!r} argument {index} "
                    "must be a value expression"
                )

            elif (
                kind is ArgumentKind.EXPRESSION
                and isinstance(argument, ConstantNode)
            ):
                # EXPRESSION 参数不能直接使用纯常量。
                # 更深层的纯常量子树由递归分析检查。
                raise TypeError(
                    f"operator {node.name!r} argument {index} "
                    "must depend on a field"
                )

    @staticmethod
    def _window_value(node: Node) -> int:
        """提取窗口长度；普通常量必须是正整数，且不能是布尔值。"""
        if isinstance(node, WindowNode):
            # 专用窗口节点直接返回数值；
            # 这里依赖 WindowNode 自身保证 value 合法。
            return node.value

        if isinstance(node, ConstantNode):
            value = node.value

            # Python 中 bool 属于 int 的子类，因此需要单独排除。
            if (
                isinstance(value, int)
                and not isinstance(value, bool)
                and value > 0
            ):
                return value

        raise TypeError("window must be a positive integer constant")