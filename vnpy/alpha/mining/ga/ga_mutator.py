"""One-site mutation of alpha expression trees."""

from __future__ import annotations

from random import Random

from vnpy.alpha.expression import ConstantNode, FieldNode, Node, OperatorNode, WindowNode
from vnpy.alpha.mining.mining_config import SearchSpace
from vnpy.alpha.mining.mining_generator import ExpressionGenerator

from .ga_tree import _field_dependent, _nodes, _replace


class Mutator:
    """随机修改表达树中的一个位置，结果由外层引擎验证"""

    def __init__(self, space: SearchSpace, random: Random) -> None:
        self.space = space
        self.random = random

        # 为子树生成器生成一个随机种子。
        seed = random.randrange(2**32)
        self.generator = ExpressionGenerator(space, seed=seed)

        # 保存搜索空间中可用的算子定义。
        self.specs = self.generator.specs

    def mutate(self, tree: Node) -> Node:
        # 收集整棵树的节点及其路径。
        sites = list(_nodes(tree))

        # 随机选择一个变异位置。
        site = self.random.choice(sites)
        path = site[0]
        node = site[1]

        # 默认不修改；如果没有其他可选值，就保留原节点。
        replacement = node

        if isinstance(node, FieldNode):
            # 字段变异，例如 close -> open。
            choices = []

            for name in self.space.fields:
                if name != node.name:
                    choices.append(name)

            if len(choices) > 0:
                name = self.random.choice(choices)
                replacement = FieldNode(name)

        elif isinstance(node, WindowNode):
            # 窗口变异，例如 5 -> 10。
            choices = []

            for value in self.space.windows:
                if value != node.value:
                    choices.append(value)

            if len(choices) > 0:
                value = self.random.choice(choices)
                replacement = WindowNode(value)

        elif isinstance(node, ConstantNode):
            # 常量变异，例如 0.5 -> 1.0。
            choices = []

            for value in self.space.constants:
                if value != node.value:
                    choices.append(value)

            if len(choices) > 0:
                value = self.random.choice(choices)
                replacement = ConstantNode(value)

        elif isinstance(node, OperatorNode):
            # 找到当前算子的定义。
            current = None

            for spec in self.specs:
                if spec.name == node.name:
                    current = spec
                    break

            # 比原代码明确：找不到算子时给出错误原因。
            if current is None:
                raise ValueError(f"找不到算子定义：{node.name}")

            # 寻找参数类型声明相同的其他算子。
            choices = []

            for spec in self.specs:
                if spec.name != current.name:
                    if spec.argument_kinds == current.argument_kinds:
                        choices.append(spec)

            # 有可选算子时，以 50% 的概率只换算子名称，
            # 保留原来的所有参数。
            if len(choices) > 0 and self.random.random() < 0.5:
                selected = self.random.choice(choices)

                replacement = OperatorNode(
                    selected.name,
                    node.args,
                )

            else:
                # 另一种方式：重新生成选中位置的整棵子树。

                # 该位置越深，新子树允许的深度越小。
                remaining_depth = self.space.max_depth - len(path)

                # 统计被替换子树的节点数。
                subtree_size = 0

                for child in node.walk():
                    subtree_size += 1

                # 原树移除旧子树后，还剩多少节点。
                outside_size = len(sites) - subtree_size

                # 新子树最多能使用的节点数量。
                budget = self.space.max_nodes - outside_size

                if budget < 1:
                    budget = 1

                # 原子树包含字段时，新子树也要求包含字段。
                require_field = _field_dependent(node)

                replacement = self.generator.grow_subtree(
                    depth=remaining_depth,
                    budget=budget,
                    require_field=require_field,
                )

        else:
            raise TypeError(f"不支持的节点类型：{type(node).__name__}")

        # 按路径替换，返回新的表达树。
        return _replace(tree, path, replacement)