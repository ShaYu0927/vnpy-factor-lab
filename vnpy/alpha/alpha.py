from __future__ import annotations

from dataclasses import dataclass

from .expression import ExpressionParser, Node


@dataclass
class Alpha:
    """User-facing formula; expression holds the most recently parsed tree."""

    name: str
    formula: str
    expression: Node | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError("alpha name must be a valid identifier")
        if self.name in {"datetime", "vt_symbol"}:
            raise ValueError("alpha name must not shadow an index column")
        if not isinstance(self.formula, str) or not self.formula.strip():
            raise ValueError("alpha formula must be a non-empty string")

    def parse(self) -> Node:
        """Reparse the current formula so edited formulas cannot use stale trees."""
        self.expression = ExpressionParser().parse(self.formula)
        return self.expression
