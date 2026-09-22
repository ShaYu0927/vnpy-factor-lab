from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .expression import Node

# Alpha公式允许调用的算子白名单。
#
# 主要分为三类：
# 1. ts_*：时间序列算子，在同一只股票的历史数据上计算；
# 2. cs_*：截面算子，在同一时间点的多只股票之间计算；
# 3. 普通算子：对当前数据逐元素计算。
#
# 使用不可变集合可以防止运行过程中意外修改算子白名单。
ALPHA_FUNCTIONS = frozenset({
    "ts_delay", "ts_min", "ts_max", "ts_argmax", "ts_argmin", "ts_rank",
    "ts_sum", "ts_mean", "ts_std", "ts_slope", "ts_quantile", "ts_rsquare",
    "ts_resi", "ts_corr", "ts_less", "ts_greater", "ts_log", "ts_abs",
    "ts_delta", "ts_cov", "ts_decay_linear", "ts_product", "cs_rank",
    "cs_mean", "cs_std", "cs_sum", "cs_scale", "less", "greater", "log",
    "abs", "sign", "pow1", "pow2", "quesval", "quesval2",
})

# 需要同时使用同一时间点多只股票数据的截面算子。
#
# 例如：
#     cs_rank(close)
#
# 必须先获得当日整个股票池的close，才能计算每只股票的排名，
# 因此不能像普通时序因子一样逐只股票独立计算。
_CROSS_SECTION_FUNCTIONS = frozenset({"cs_rank", "cs_mean", "cs_std", "cs_sum", "cs_scale"})

# 禁止在Alpha特征公式中引用的字段名称。
# 这些字段通常代表未来收益率或模型训练标签。如果它们进入特征，
# 就会产生未来数据泄漏，使回测结果失真。
_FORBIDDEN_NAMES = frozenset({"label", "target", "future", "future_return", "future_ret"})
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Call,
    ast.Name, ast.Load, ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div,
    ast.Pow, ast.Mod, ast.USub, ast.UAdd, ast.And, ast.Or, ast.Gt, ast.GtE,
    ast.Lt, ast.LtE, ast.Eq, ast.NotEq,
)

# Alpha公式中允许出现的Python AST节点类型。
#
# ast.parse()会把公式字符串解析成Python抽象语法树。例如：
#
#     close / ts_delay(close, 5) - 1
#
# 大致会被解析成：
#
#     Expression
#       └── BinOp(Sub)
#           ├── BinOp(Div)
#           │   ├── Name(close)
#           │   └── Call(ts_delay)
#           └── Constant(1)
#
# 通过AST节点白名单，可以禁止属性访问、下标访问、列表推导、
# Lambda表达式等Alpha公式不需要或可能存在安全风险的语法。
@dataclass(frozen=True, slots=True)
class AlphaDefinition:
    """
    表示一条经过验证、只能使用当前或历史数据的Alpha因子定义。
    AlphaDefinition是表达树与实际执行引擎之间的中间结构,主要保存:

    1. 因子名称；
    2. 可执行的因子表达式；
    3. 计算因子需要的历史回看长度；
    4. 因子版本。

    frozen=True表示对象创建后不能修改,防止运行期间因子定义变化。
    slots=True用于减少对象占用的内存,并限制只能使用已声明字段。

    Attributes:
        name:
            Alpha因子的唯一名称,通常也作为结果DataFrame中的列名。
        expression:
            可执行的Alpha表达式字符串,例如:
            "close / ts_delay(close, 5) - 1"
        lookback:
            计算该因子至少需要准备的历史K线数量。
        version:
            Alpha因子的版本号,用于区分同名因子的不同实现。
    """

    name: str
    expression: str
    lookback: int
    version: str = "1"

    @classmethod
    def from_tree(cls, name: str, tree: Node, available_fields: set[str] | None = None, version: str = "1",) -> AlphaDefinition:
        """
        根据领域表达树创建可执行的AlphaDefinition。

        该方法会先分析表达树，再把表达树编译为底层可执行表达式。
        推荐优先使用该方法创建AlphaDefinition,因为lookback会根据
        表达树自动推导，不需要调用方手动填写。
        """
        from .expression import ExpressionAnalyzer, compile_expression

        analysis = ExpressionAnalyzer().analyze(tree, available_fields)
        return cls(
            name=name,
            expression=compile_expression(tree),
            lookback=analysis.lookback,
            version=version,
        )

    def __post_init__(self) -> None:
        """
        在AlphaDefinition创建完成后验证名称、lookback和表达式安全性。
        验证内容包括：
        1. Alpha名称必须是合法标识符;
        2. lookback至少为1;
        3. 表达式必须是合法的Python表达式;
        4. AST中只能出现允许的语法节点;
        5. 不允许引用未来收益率或训练标签；
        6. 只能调用已注册的Alpha算子;
        7. ts_delay只能使用正整数常量读取历史数据。

        Raises:
            ValueError:
                任意一项验证失败。
        """
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
        """
        判断Alpha公式是否使用了截面算子。
        截面算子需要同时获得同一时间点多只股票的数据。例如：
            cs_rank(close)
        需要比较同一天所有股票的close,因此不能逐只股票独立计算。
        Returns:
            表达式中存在cs_rank、cs_mean、cs_std、cs_sum或
            cs_scale时返回True,否则返回False。
        """
        tree = _parse_expression(self.expression)
        
        # 遍历所有节点，只要发现一个截面函数调用，就返回True。
        return any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _CROSS_SECTION_FUNCTIONS
            for node in ast.walk(tree)
        )


def _parse_expression(expression: str) -> ast.Expression:
    """
    将Alpha公式字符串解析成Python表达式AST。
    使用mode="eval"表示只允许输入一条表达式,而不是多条Python语句
    合法示例：
        close / ts_delay(close, 5) - 1
    非法示例：
        value = close
        value + 1
    Args:
        expression:
            需要解析的Alpha公式字符串。
    Returns:
        解析完成的AST表达式根节点
    Raises:
        ValueError:
            表达式存在Python语法错误。
    """
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
