"""Bounded random tree generation; batches share their RNG and deduplication state."""

from __future__ import annotations

from collections import Counter
from random import Random
from typing import Iterator

from vnpy.alpha.expression import (
    ArgumentKind, ConstantNode, ExpressionAnalyzer, FieldNode, Node, OperatorNode,
    PolarsCompiler, WindowNode, create_default_registry,
)

from .config import ExpressionIterationConfig, SearchSpace, positive_int
from .schema import Candidate, GenerationBatch, tree_fingerprint


class ExpressionGenerator:
    """Generate new candidates without mutating the existing expression engine.

    Generation is guided by argument signatures and a node/depth budget. Accepted
    trees are analyzed and compiled, but are not scored or checked for profitability.
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
        choices = [spec for spec in self.specs if 1 + spec.min_args <= budget]
        if depth <= 1 or not choices or self.random.random() < 0.30:
            if not require_field and self.space.constants and self.random.random() < 0.35:
                return ConstantNode(self.random.choice(self.space.constants))
            return FieldNode(self.random.choice(self.space.fields))

        spec = self.random.choice(choices)
        kinds = spec.argument_kinds or (ArgumentKind.EXPRESSION,) * spec.min_args
        value_indices = [i for i, kind in enumerate(kinds) if kind is not ArgumentKind.WINDOW]
        field_index = self.random.choice(value_indices) if require_field else -1
        remaining = budget - 1
        children = []
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
        return OperatorNode(spec.name, tuple(children))

    def generate_batch(self, count: int, max_attempts: int) -> GenerationBatch:
        """Return a partial exhausted batch when no more candidates can be found in budget."""
        positive_int("count", count)
        positive_int("max_attempts", max_attempts)
        self.batch_index += 1
        candidates: list[Candidate] = []
        rejected: Counter[str] = Counter()
        attempts = duplicates = 0
        while len(candidates) < count and attempts < max_attempts:
            attempts += 1
            tree = self._grow(self.space.max_depth, self.space.max_nodes, require_field=True)
            key = tree_fingerprint(tree)
            if key in self.seen:
                duplicates += 1
                continue
            # Remember rejected structures as well, so they are not recompiled in later batches.
            self.seen.add(key)
            analysis = self.analyzer.analyze(tree, available_fields=set(self.space.fields))
            if analysis.lookback > self.space.max_lookback:
                rejected["lookback_limit"] += 1
                continue
            self.compiler.compile(tree, available_fields=set(self.space.fields))
            candidates.append(Candidate(tree, self.batch_index, analysis.lookback, tuple(sorted(analysis.fields))))
        return GenerationBatch(
            self.batch_index, tuple(candidates), attempts, duplicates,
            dict(sorted(rejected.items())), len(candidates) == count,
        )


def iterate_expressions(config: ExpressionIterationConfig) -> Iterator[GenerationBatch]:
    """Fresh runs reproduce the sequence; consecutive batches never restart the RNG."""
    generator = ExpressionGenerator(config.search_space, config.seed)
    for _ in range(config.batches):
        batch = generator.generate_batch(config.batch_size, config.max_attempts_per_batch)
        yield batch
        if not batch.complete:
            break
