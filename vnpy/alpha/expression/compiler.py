from __future__ import annotations

from .node import ConstantNode, FieldNode, Node, OperatorNode, WindowNode


class ExpressionCompiler:
    """
    将表达树转换为当前 Alpha 引擎使用的 DSL 公式字符串

    通过递归遍历节点，将字段、常量、窗口和算子拼接为公式
    这里只生成字符串，不执行公式，也不计算因子值

    输入的表达树应提前完成字段、算子和参数合法性校验
    """

    # 将表达树中的算子名称映射为中缀运算符。
    # 例如：add(close, open) 转换为 (close + open)。
    _INFIX_OPERATORS = {
        "add": "+",
        "sub": "-",
        "mul": "*",
        "div": "/",
    }

    def compile(self, node: Node) -> str:
        """
        递归编译一个表达树节点。

        Args:
            node: 待编译的字段、常量、窗口或算子节点

        Returns:
            该节点对应的 DSL 公式字符串

        Raises:
            TypeError: 节点类型不受支持
            ValueError: 加减乘除或幂运算的参数数量不是两个
        """
        # 字段节点直接返回字段名称，例如 "close"、"volume"。
        if isinstance(node, FieldNode):
            return node.name

        # 常量节点使用 repr 保留 Python 数值的字符串表示。
        # 例如：1.0 转换为 "1.0"。
        if isinstance(node, ConstantNode):
            return repr(node.value)

        # 窗口节点转换为窗口长度字符串，例如 20 转换为 "20"。
        if isinstance(node, WindowNode):
            return str(node.value)

        # 排除基础节点后，剩余节点必须是算子节点。
        if not isinstance(node, OperatorNode):
            raise TypeError(
                f"unsupported expression node: {type(node).__name__}"
            )

        # 先递归编译所有参数，再拼接当前算子的公式。
        # 参数本身也可以是包含其他算子的子树。
        arguments = tuple(
            self.compile(argument) for argument in node.args
        )

        # 加减乘除采用中缀写法，并用括号保留表达树的运算顺序。
        if node.name in self._INFIX_OPERATORS:
            if len(arguments) != 2:
                raise ValueError(f"operator {node.name!r} requires two arguments")

            symbol = self._INFIX_OPERATORS[node.name]
            return f"({arguments[0]} {symbol} {arguments[1]})"

        # 幂运算根据指数节点的类型，选择引擎对应的函数：
        # 指数为 ConstantNode 时使用 pow1，否则使用 pow2。
        if node.name == "pow":
            if len(arguments) != 2:
                raise ValueError("operator 'pow' requires two arguments")

            function = (
                "pow1"
                if isinstance(node.args[1], ConstantNode)
                else "pow2"
            )
            return f"{function}({arguments[0]}, {arguments[1]})"

        # 其他算子采用普通函数调用格式。
        # 例如：节点名称为 Mean，参数为 close、20，
        # 则生成 "Mean(close, 20)"。
        return f"{node.name}({', '.join(arguments)})"


def compile_expression(node: Node) -> str:
    """
    将已经校验的表达树编译为 Alpha 引擎可执行的公式字符串。

    Args:
        node: 表达树的根节点。

    Returns:
        完整的 DSL 公式字符串；本函数不执行该公式。
    """
    return ExpressionCompiler().compile(node)