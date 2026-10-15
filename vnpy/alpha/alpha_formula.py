from __future__ import annotations

from dataclasses import dataclass

from .expression import ExpressionParser, Node


@dataclass(frozen=True, slots=True)
class Alpha:
    """Immutable name and formula for one observable Alpha feature."""

    name: str
    formula: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError("alpha name must be a valid identifier")
        if self.name in {"datetime", "vt_symbol"}:
            raise ValueError("alpha name must not shadow an index column")
        if not isinstance(self.formula, str) or not self.formula.strip():
            raise ValueError("alpha formula must be a non-empty string")

    def parse(self) -> Node:
        """Parse the formula without changing the definition."""
        return ExpressionParser().parse(self.formula)
