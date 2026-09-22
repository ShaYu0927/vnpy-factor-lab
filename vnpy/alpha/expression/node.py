from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import isfinite
from typing import Iterator


class Node(ABC):
    "表达树中所有基类"
    
    @property
    def children(self) -> tuple["Node", ...]:
        """返回当前节点的子节点"""
        return ()
    
    @abstractmethod
    def to_formula(self) -> str:
        """转换成可读的因子公式"""
        raise NotImplementedError
    
    @property
    def depth(self) -> int:
        """整棵子树的深度"""
        if not self.children:
            return 1
        return 1 + max(child.depth for child in self.children)
    
    def walk(self) -> Iterator["Node"]:
        """从当前节点开始遍历整棵树"""
        yield self

        for child in self.children:
            yield from child.walk()
            
    def __str__(self) -> str:
        return self.to_formula()


@dataclass(frozen=True, slots=True)
class FieldNode(Node):
    """行情字段节点，例如 close、open、volume"""
    
    name: str
    
    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("field name must be a string")

        if not self.name:
            raise ValueError("field name must not be empty")

        if not self.name.isidentifier():
            raise ValueError(f"field name must be a valid identifier: {self.name!r}")
        
    def to_formula(self) -> str:
        return self.name
    
@dataclass(frozen=True, slots=True)
class ConstantNode(Node):
    """常数节点，例如 5、10、20"""
    value: int | float

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise TypeError("constant must be an integer or float")
        if not isfinite(self.value):
            raise ValueError("constant must be finite")
    
    def to_formula(self) -> str:
        return str(self.value)


@dataclass(frozen=True, slots=True)
class WindowNode(Node):
    """Positive bar count used by time-series operators."""

    value: int

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, int):
            raise TypeError("window must be an integer")
        if self.value < 1:
            raise ValueError("window must be a positive integer")

    def to_formula(self) -> str:
        return str(self.value)
    
    
@dataclass(frozen=True, slots=True)
class OperatorNode(Node):
    """运算符节点，例如 Rank、Delta、Add"""
    
    name: str
    args: tuple[Node, ...]
    
    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("operator name must be a string")

        if not self.name:
            raise ValueError("operator name must not be empty")

        if not self.name.isidentifier():
            raise ValueError(f"operator name must be a valid identifier: {self.name!r}")

        if not self.args:
            raise ValueError("operator must accept at least one argument")

        if not isinstance(self.args, tuple):
            raise TypeError("operator arguments must be a tuple")

        if not all(isinstance(arg, Node) for arg in self.args):
            raise TypeError("operator arguments must be expression nodes")


    @property
    def children(self) -> tuple[Node, ...]:
        return self.args
    
    def to_formula(self) -> str:
        arguments = ", ".join(child.to_formula() for child in self.args)
        return f"{self.name}({arguments})"
    
NodeLike = Node | str | int | float

def as_node(value: NodeLike) -> Node:
    """把字段名或数字自动转换成节点"""
    
    if isinstance(value, Node):
        return value

    if isinstance(value, str):
        return FieldNode(value)

    if isinstance(value, bool):
        raise TypeError("boolean values are not supported")

    if isinstance(value, (int, float)):
        return ConstantNode(value)

    raise TypeError(f"unsupported expression node value: {type(value).__name__}")


def op(name: str, *args: NodeLike) -> OperatorNode:
    """快速创建运算符节点"""
    
    return OperatorNode(name=name, args=tuple(as_node(arg) for arg in args),)


def window(value: int) -> WindowNode:
    """Create an explicit time-series window node."""
    return WindowNode(value)



    
    
