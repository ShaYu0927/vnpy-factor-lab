"""Candidate identity is derived from the immutable expression tree."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

from vnpy.alpha.alpha import Alpha
from vnpy.alpha.expression import ConstantNode, FieldNode, Node, OperatorNode, WindowNode


def serialize_tree(tree: Node) -> list:
    if isinstance(tree, OperatorNode):
        return ["operator", tree.name, [serialize_tree(child) for child in tree.args]]
    if isinstance(tree, FieldNode):
        return ["field", tree.name]
    if isinstance(tree, WindowNode):
        return ["window", tree.value]
    if isinstance(tree, ConstantNode):
        return ["constant", tree.value]
    raise TypeError(f"unsupported node: {type(tree).__name__}")


def tree_fingerprint(tree: Node) -> str:
    """Structural identity, not mathematical equivalence."""
    payload = json.dumps(serialize_tree(tree), separators=(",", ":"), allow_nan=False)
    return sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Candidate:
    tree: Node
    batch: int
    lookback: int
    fields: tuple[str, ...]

    @property
    def fingerprint(self) -> str:
        return tree_fingerprint(self.tree)

    @property
    def formula(self) -> str:
        return self.tree.to_formula()

    def to_alpha(self) -> Alpha:
        return Alpha(name=f"candidate_{self.fingerprint}", formula=self.formula)

    def as_dict(self) -> dict:
        return {
            "fingerprint": self.fingerprint, "batch": self.batch,
            "formula": self.formula, "tree": serialize_tree(self.tree),
            "depth": self.tree.depth, "node_count": sum(1 for _ in self.tree.walk()),
            "lookback": self.lookback, "fields": list(self.fields),
            "status": "unevaluated", "origin": "random",
        }


@dataclass(frozen=True)
class GenerationBatch:
    index: int
    candidates: tuple[Candidate, ...]
    attempts: int
    duplicates: int
    rejected: dict[str, int]
    complete: bool

    def summary(self) -> dict:
        return {
            "batch": self.index, "generated": len(self.candidates),
            "attempts": self.attempts, "duplicates": self.duplicates,
            "rejected": self.rejected, "status": "complete" if self.complete else "exhausted",
        }
