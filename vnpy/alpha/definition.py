from __future__ import annotations

import ast
from dataclasses import dataclass


ALPHA_FUNCTIONS = frozenset({
    "ts_delay", "ts_min", "ts_max", "ts_argmax", "ts_argmin", "ts_rank",
    "ts_sum", "ts_mean", "ts_std", "ts_slope", "ts_quantile", "ts_rsquare",
    "ts_resi", "ts_corr", "ts_less", "ts_greater", "ts_log", "ts_abs",
    "ts_delta", "ts_cov", "ts_decay_linear", "ts_product", "cs_rank",
    "cs_mean", "cs_std", "cs_sum", "cs_scale", "less", "greater", "log",
    "abs", "sign", "pow1", "pow2", "quesval", "quesval2",
})
_CROSS_SECTION_FUNCTIONS = frozenset({"cs_rank", "cs_mean", "cs_std", "cs_sum", "cs_scale"})
_FORBIDDEN_NAMES = frozenset({"label", "target", "future", "future_return", "future_ret"})
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Call,
    ast.Name, ast.Load, ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div,
    ast.Pow, ast.Mod, ast.USub, ast.UAdd, ast.And, ast.Or, ast.Gt, ast.GtE,
    ast.Lt, ast.LtE, ast.Eq, ast.NotEq,
)


@dataclass(frozen=True, slots=True)
class AlphaDefinition:
    """A versioned, past-only alpha expression shared by research and runtime."""

    name: str
    expression: str
    lookback: int
    version: str = "1"

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError("alpha name must be a valid identifier")
        if self.lookback < 1:
            raise ValueError("alpha lookback must be at least 1")
        tree = _parse_expression(self.expression)
        for node in ast.walk(tree):
            if not isinstance(node, _ALLOWED_NODES):
                raise ValueError(f"unsupported alpha syntax: {type(node).__name__}")
            if isinstance(node, ast.Name) and node.id.lower() in _FORBIDDEN_NAMES:
                raise ValueError(f"alpha expressions cannot reference future labels: {node.id}")
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name) or node.func.id not in ALPHA_FUNCTIONS:
                    raise ValueError("alpha expressions may only call registered operators")
                if node.func.id == "ts_delay":
                    if len(node.args) < 2:
                        raise ValueError("ts_delay requires a positive integer literal")
                    delay = _integer_literal(node.args[1])
                    if not isinstance(delay, int) or isinstance(delay, bool) or delay <= 0:
                        raise ValueError("ts_delay cannot look forward or use zero delay")

    @property
    def uses_cross_section(self) -> bool:
        tree = _parse_expression(self.expression)
        return any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _CROSS_SECTION_FUNCTIONS
            for node in ast.walk(tree)
        )


def _parse_expression(expression: str) -> ast.Expression:
    try:
        return ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid alpha expression: {exc.msg}") from exc


def _integer_literal(node: ast.expr) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
        value = node.operand.value
        if isinstance(value, int) and not isinstance(value, bool):
            return -value
    return None
