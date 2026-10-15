from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class OperatorCategory(Enum):
    """Expression operator execution category."""

    ELEMENT = "element"
    TIME_SERIES = "time_series"
    CROSS_SECTION = "cross_section"


class ArgumentKind(Enum):
    """Semantic kind expected at an operator argument position."""

    EXPRESSION = "expression"
    VALUE = "value"
    WINDOW = "window"


class LookbackRule(Enum):
    """How an operator extends the history required by its operands."""

    SAME = "same"
    DELAY = "delay"
    ROLLING = "rolling"


@dataclass(frozen=True, slots=True)
class OperatorSpec:
    """Static metadata describing an expression operator."""

    name: str
    min_args: int
    max_args: int
    category: OperatorCategory
    argument_kinds: tuple[ArgumentKind, ...] = ()
    lookback_rule: LookbackRule = LookbackRule.SAME

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("operator name must not be empty")
        if not self.name.isidentifier():
            raise ValueError(f"operator name must be a valid identifier: {self.name!r}")
        if self.min_args < 1:
            raise ValueError("operator must accept at least one argument")
        if self.max_args < self.min_args:
            raise ValueError("max_args must be greater than or equal to min_args")
        if self.argument_kinds and not (
            self.min_args <= len(self.argument_kinds) <= self.max_args
        ):
            raise ValueError("argument_kinds length must match the operator arity")
        if self.lookback_rule is not LookbackRule.SAME:
            if not self.argument_kinds or self.argument_kinds[-1] is not ArgumentKind.WINDOW:
                raise ValueError("lookback operator must have a final window argument")

    def validate_arity(self, argument_count: int) -> None:
        """Validate the number of arguments supplied to this operator."""
        if not self.min_args <= argument_count <= self.max_args:
            raise ValueError(
                f"operator {self.name!r} expects "
                f"{self.min_args} to {self.max_args} arguments, "
                f"but got {argument_count}"
            )


class OperatorRegistry:
    """Registry containing all operators allowed in factor expressions."""

    def __init__(self) -> None:
        self._operators: dict[str, OperatorSpec] = {}

    def register(self, spec: OperatorSpec) -> None:
        if spec.name in self._operators:
            raise ValueError(f"operator already registered: {spec.name}")
        self._operators[spec.name] = spec

    def get(self, name: str) -> OperatorSpec:
        try:
            return self._operators[name]
        except KeyError:
            raise KeyError(f"unknown operator: {name}") from None

    def contains(self, name: str) -> bool:
        return name in self._operators

    def __contains__(self, name: str) -> bool:
        return self.contains(name)

    def __len__(self) -> int:
        return len(self._operators)


def create_default_registry() -> OperatorRegistry:
    """Create the built-in operator registry used by alpha expressions."""
    registry = OperatorRegistry()
    expression = (ArgumentKind.EXPRESSION,)
    binary = (ArgumentKind.VALUE, ArgumentKind.VALUE)
    power = (ArgumentKind.EXPRESSION, ArgumentKind.VALUE)
    rolling = (ArgumentKind.EXPRESSION, ArgumentKind.WINDOW)
    pair_rolling = (
        ArgumentKind.EXPRESSION,
        ArgumentKind.EXPRESSION,
        ArgumentKind.WINDOW,
    )

    specs = (
        OperatorSpec("abs", 1, 1, OperatorCategory.ELEMENT, expression),
        OperatorSpec("sign", 1, 1, OperatorCategory.ELEMENT, expression),
        OperatorSpec("log", 1, 1, OperatorCategory.ELEMENT, expression),
        OperatorSpec("add", 2, 2, OperatorCategory.ELEMENT, binary),
        OperatorSpec("sub", 2, 2, OperatorCategory.ELEMENT, binary),
        OperatorSpec("mul", 2, 2, OperatorCategory.ELEMENT, binary),
        OperatorSpec("div", 2, 2, OperatorCategory.ELEMENT, binary),
        OperatorSpec("pow", 2, 2, OperatorCategory.ELEMENT, power),
        OperatorSpec(
            "ts_delay", 2, 2, OperatorCategory.TIME_SERIES,
            rolling, LookbackRule.DELAY,
        ),
        OperatorSpec(
            "ts_delta", 2, 2, OperatorCategory.TIME_SERIES,
            rolling, LookbackRule.DELAY,
        ),
        OperatorSpec(
            "ts_mean", 2, 2, OperatorCategory.TIME_SERIES,
            rolling, LookbackRule.ROLLING,
        ),
        OperatorSpec(
            "ts_sum", 2, 2, OperatorCategory.TIME_SERIES,
            rolling, LookbackRule.ROLLING,
        ),
        OperatorSpec(
            "ts_std", 2, 2, OperatorCategory.TIME_SERIES,
            rolling, LookbackRule.ROLLING,
        ),
        OperatorSpec(
            "ts_min", 2, 2, OperatorCategory.TIME_SERIES,
            rolling, LookbackRule.ROLLING,
        ),
        OperatorSpec(
            "ts_max", 2, 2, OperatorCategory.TIME_SERIES,
            rolling, LookbackRule.ROLLING,
        ),
        OperatorSpec(
            "ts_corr", 3, 3, OperatorCategory.TIME_SERIES,
            pair_rolling, LookbackRule.ROLLING,
        ),
        OperatorSpec("cs_rank", 1, 1, OperatorCategory.CROSS_SECTION, expression),
    )

    for spec in specs:
        registry.register(spec)

    return registry
