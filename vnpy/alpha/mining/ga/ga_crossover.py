"""Subtree crossover between two alpha expressions."""
from __future__ import annotations
from random import Random
from vnpy.alpha.expression import Node, WindowNode
from .ga_tree import _field_dependent, _nodes, _replace


class Crossover:
    """交叉操作：用右侧父代的子树替换左侧父代的局部节点，生成候选因子"""

    def __init__(self, random: Random) -> None:
        # 使用外部传入的随机数生成器，便于统一控制随机种子、复现结果。
        self.random = random

    def cross(self, left: Node, right: Node) -> Node | None:
        """
        尝试交叉两个表达树；缺少可用位置或捐赠子树时返回 None

        本方法只负责生成新组合。
        参数签名、树深度、节点数量和回看长度由外层引擎检查。
        """
        # 遍历左侧父代，收集可被替换的节点及其路径：
        # 1. path 非空：排除根节点，保留左侧表达式的外层结构。
        # 2. 排除 WindowNode：避免把窗口参数替换为行情表达式。
        destinations = [
            (path, node)
            for path, node in _nodes(left)
            if path and not isinstance(node, WindowNode)
        ]

        # 遍历右侧父代，收集可用于替换的子树：
        # 1. 排除根节点，避免直接搬入整个右侧表达式。
        # 2. 要求子树依赖行情字段，避免只搬入纯常量表达式。
        donors = [
            node
            for path, node in _nodes(right)
            if path and _field_dependent(node)
        ]

        # 任意一侧没有可用候选，就无法执行交叉。
        if not destinations or not donors:
            return None

        # 随机选择左侧的替换位置；这里只需要路径，不需要原节点。
        path, _ = self.random.choice(destinations)

        # 随机选择右侧子树，替换到左侧指定位置，返回新表达树。
        return _replace(left, path, self.random.choice(donors))
