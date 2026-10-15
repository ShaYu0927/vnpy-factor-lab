"""随机生成大小受限的表达树；不同批次共享随机数状态和去重记录"""

from __future__ import annotations

from collections import Counter
from random import Random
from typing import Iterator

from vnpy.alpha.expression import (
    ArgumentKind, ConstantNode, ExpressionAnalyzer, FieldNode, Node, OperatorNode,
    PolarsCompiler, WindowNode, create_default_registry,
)

from .mining_config import ExpressionIterationConfig, SearchSpace, positive_int
from .mining_schema import Candidate, GenerationBatch, tree_fingerprint


class ExpressionGenerator:
    """
    随机生成候选因子表达树

    负责：
    1. 按照算子的参数类型生成表达树。
    2. 限制树的深度、节点数和历史回看长度。
    3. 去重，并检查表达树是否能够分析和编译。

    此处不使用行情数据计算因子值，也不评价 IC 或盈利能力。
    """

    def __init__(self, space: SearchSpace, seed: int = 42) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        self.space = space
        self.random = Random(seed)
        registry = create_default_registry()
        self.specs = tuple(registry.get(name) for name in space.operators)
        self.analyzer = ExpressionAnalyzer()
        self.compiler = PolarsCompiler()
        self.seen: set[str] = set()
        self.batch_index = 0

    def _grow(self, depth: int, budget: int, require_field: bool) -> Node:
        """
        递归生成表达树或子树。

        depth: 这一棵子树允许的最大深度。
        budget: 这一棵子树允许使用的最大节点数。
        require_field: 是否要求生成结果包含行情字段。
        """
        
         # 算子自己占一个节点，每个参数至少占一个节点。
        choices = [spec for spec in self.specs if 1 + spec.min_args <= budget]
        if depth <= 1 or not choices or self.random.random() < 0.30:
            if not require_field and self.space.constants and self.random.random() < 0.35:
                return ConstantNode(self.random.choice(self.space.constants))
            return FieldNode(self.random.choice(self.space.fields))
        
        
        # 以下情况停止生长，返回一个叶子节点：
        # 1. 深度只剩一层；
        # 2. 没有预算装得下的算子；
        # 3. 随机提前停止，概率为 30%。
        spec = self.random.choice(choices)
        kinds = spec.argument_kinds or (ArgumentKind.EXPRESSION,) * spec.min_args
        value_indices = [i for i, kind in enumerate(kinds) if kind is not ArgumentKind.WINDOW]
        field_index = self.random.choice(value_indices) if require_field else -1
        remaining = budget - 1
        children = []
         # 当前算子自身占一个节点，剩余预算分配给子节点。
        for index, kind in enumerate(kinds):
            if kind is ArgumentKind.WINDOW:
                child = WindowNode(self.random.choice(self.space.windows))
            else:
                reserve = len(kinds) - index - 1
                child_budget = self.random.randint(1, remaining - reserve)
                child = self._grow(
                    depth - 1, child_budget,
                    require_field=kind is ArgumentKind.EXPRESSION or index == field_index,
                )
            children.append(child)
            remaining -= sum(1 for _ in child.walk())
         # 将算子和生成的参数组合成一个节点。
        return OperatorNode(spec.name, tuple(children))

    def grow_tree(self) -> Node:
        """按配置生成完整候选树，尚未进行分析和编译检查。"""
        return self._grow(self.space.max_depth, self.space.max_nodes, require_field=True)

    def grow_subtree(self, *, depth: int, budget: int, require_field: bool) -> Node:
        """
        按照调用方给出的限制生成子树。

        后续实现变异时，可以用新子树替换原表达树的一部分
        """
        return self._grow(depth, budget, require_field)

    def generate_batch(self, count: int, max_attempts: int) -> GenerationBatch:
        """
        生成一批候选。

        count: 希望获得的候选数量。
        max_attempts: 允许的最大尝试次数。

        达到尝试上限仍未凑满时，返回已有候选，
        并通过 complete=False 表示该批次未完成。
        """
        positive_int("count", count)
        positive_int("max_attempts", max_attempts)
        self.batch_index += 1
        candidates: list[Candidate] = []
        rejected: Counter[str] = Counter()
        attempts = duplicates = 0
        # 凑满目标数量，或用完尝试次数，就结束
        while len(candidates) < count and attempts < max_attempts:
            attempts += 1
            tree = self.grow_tree()  # 生成一棵表达树
            key = tree_fingerprint(tree)  # 为表达树生成结构指纹，用于去重
            if key in self.seen:
                duplicates += 1
                continue
            # Remember rejected structures as well, so they are not recompiled in later batches.
            self.seen.add(key)
            analysis = self.analyzer.analyze(tree, available_fields=set(self.space.fields))  # 分析合法性、使用的字段和历史回看长度
            if analysis.lookback > self.space.max_lookback:
                rejected["lookback_limit"] += 1
                continue
            self.compiler.compile(tree, available_fields=set(self.space.fields))
            
            # 保存通过检查的候选及其元信息
            candidates.append(Candidate(tree, self.batch_index, analysis.lookback, tuple(sorted(analysis.fields))))
        # 返回候选和生成统计。
        return GenerationBatch(
            self.batch_index,               # 当前批次编号
            tuple(candidates),              # 通过检查的候选
            attempts,                       # 总尝试次数
            duplicates,                     # 重复结构次数
            dict(sorted(rejected.items())), # 拒绝原因统计
            len(candidates) == count,       # 是否凑满目标数量
        )



def iterate_expressions(config: ExpressionIterationConfig) -> Iterator[GenerationBatch]:
    """连续生成多批候选，每次 yield 一批"""
    generator = ExpressionGenerator(config.search_space, config.seed)
    for _ in range(config.batches):
        batch = generator.generate_batch(config.batch_size, config.max_attempts_per_batch)
        yield batch
        if not batch.complete:
            break
