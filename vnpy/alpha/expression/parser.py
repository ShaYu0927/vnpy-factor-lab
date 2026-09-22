"""Parse formula text into the shared expression tree without executing Python."""

from __future__ import annotations

import ast

from .node import ConstantNode, FieldNode, Node, OperatorNode, WindowNode
from .operator import ArgumentKind, OperatorRegistry, create_default_registry


class ExpressionParser:
    """Accept arithmetic, numeric literals, fields and registered function calls."""

    ALIASES = {
        "Ref": "ts_delay",
        "Delta": "ts_delta",
        "Mean": "ts_mean",
        "Sum": "ts_sum",
        "Std": "ts_std",
        "Min": "ts_min",
        "Max": "ts_max",
        "Corr": "ts_corr",
        "Rank": "cs_rank",
        "Abs": "abs",
        "Sign": "sign",
        "Log": "log",
        "Pow": "pow",
    }
    _BINARY = {
        ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul",
        ast.Div: "div", ast.Pow: "pow",
    }

    def __init__(self, registry: OperatorRegistry | None = None) -> None:
        self.registry = registry or create_default_registry()

    def parse(self, formula: str) -> Node:
        if not isinstance(formula, str) or not formula.strip():
            raise ValueError("alpha formula must be a non-empty string")
        try:
            parsed = ast.parse(formula.strip(), mode="eval")
        except SyntaxError as exc:
            raise ValueError(
                f"invalid alpha formula at line {exc.lineno}, column {exc.offset}: {exc.msg}"
            ) from exc
        return self._convert(parsed.body)

    def _convert(self, node: ast.expr) -> Node:
        if isinstance(node, ast.Name):
            return FieldNode(node.id)
        if isinstance(node, ast.Constant):
            return ConstantNode(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in self._BINARY:
            return OperatorNode(
                self._BINARY[type(node.op)],
                (self._convert(node.left), self._convert(node.right)),
            )
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = self._convert(node.operand)
            if isinstance(node.op, ast.UAdd):
                return operand
            if isinstance(operand, ConstantNode):
                return ConstantNode(-operand.value)
            return OperatorNode("sub", (ConstantNode(0), operand))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.keywords:
                raise ValueError("alpha functions require a plain name and positional arguments")
            name = self.ALIASES.get(node.func.id, node.func.id)
            try:
                spec = self.registry.get(name)
            except KeyError as exc:
                raise ValueError(f"unknown alpha operator: {node.func.id}") from exc
            spec.validate_arity(len(node.args))
            arguments = [self._convert(arg) for arg in node.args]
            for index, kind in enumerate(spec.argument_kinds):
                if kind is ArgumentKind.WINDOW:
                    argument = arguments[index]
                    if not isinstance(argument, ConstantNode):
                        raise ValueError(f"{node.func.id} window must be a positive integer literal")
                    arguments[index] = WindowNode(argument.value)
            return OperatorNode(name, tuple(arguments))
        raise ValueError(
            f"unsupported alpha syntax: {type(node).__name__} "
            f"at line {node.lineno}, column {node.col_offset + 1}"
        )
