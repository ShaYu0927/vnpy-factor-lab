"""表达树遍历和替换工具；替换时返回新树，不修改原树。"""

from __future__ import annotations

from vnpy.alpha.expression import FieldNode, Node, OperatorNode


from collections.abc import Iterator

from vnpy.alpha.expression import FieldNode, Node, OperatorNode


def _replace(node: Node, path: tuple[int, ...], replacement: Node,) -> Node:
    """替换指定路径上的子树，保留原树"""
    if not path:
        # 相同内容不必创建新树。
        return node if node == replacement else replacement

    if not isinstance(node, OperatorNode):
        raise ValueError("replacement path continues beyond a leaf node")

    index = path[0]

    # 排除负数、布尔值和其他非整数下标。
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError(f"path index must be an integer: {index!r}")

    if not 0 <= index < len(node.args):
        raise ValueError(
            f"path index {index} is out of range "
            f"for operator {node.name!r} with {len(node.args)} arguments"
        )

    old_child = node.args[index]
    new_child = _replace(old_child, path[1:], replacement)

    # 子树没有变化时，保留当前节点及其祖先。
    if new_child is old_child:
        return node

    children = list(node.args)
    children[index] = new_child
    return OperatorNode(node.name, tuple(children))


def _nodes(node: Node, path: tuple[int, ...] = (),) -> Iterator[tuple[tuple[int, ...], Node]]:
    """按先序遍历返回节点路径及节点"""
    yield path, node

    if isinstance(node, OperatorNode):
        for index, child in enumerate(node.args):
            yield from _nodes(child, path + (index,))


def _field_dependent(node: Node) -> bool:
    """检查当前树是否包含行情字段节点，不进行数学化简。"""
    return any(
        isinstance(child, FieldNode)
        for child in node.walk()
    )


