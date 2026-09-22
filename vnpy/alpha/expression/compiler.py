from __future__ import annotations

from .node import ConstantNode, FieldNode, Node, OperatorNode, WindowNode


class ExpressionCompiler:
    """Compile an expression tree to the DSL used by the current alpha engine."""

    _INFIX_OPERATORS = {
        "add": "+",
        "sub": "-",
        "mul": "*",
        "div": "/",
    }

    def compile(self, node: Node) -> str:
        if isinstance(node, FieldNode):
            return node.name

        if isinstance(node, ConstantNode):
            return repr(node.value)

        if isinstance(node, WindowNode):
            return str(node.value)

        if not isinstance(node, OperatorNode):
            raise TypeError(f"unsupported expression node: {type(node).__name__}")

        arguments = tuple(self.compile(argument) for argument in node.args)

        if node.name in self._INFIX_OPERATORS:
            if len(arguments) != 2:
                raise ValueError(f"operator {node.name!r} requires two arguments")
            symbol = self._INFIX_OPERATORS[node.name]
            return f"({arguments[0]} {symbol} {arguments[1]})"

        if node.name == "pow":
            if len(arguments) != 2:
                raise ValueError("operator 'pow' requires two arguments")
            function = "pow1" if isinstance(node.args[1], ConstantNode) else "pow2"
            return f"{function}({arguments[0]}, {arguments[1]})"

        return f"{node.name}({', '.join(arguments)})"


def compile_expression(node: Node) -> str:
    """Compile a validated expression tree to an executable formula string."""
    return ExpressionCompiler().compile(node)
