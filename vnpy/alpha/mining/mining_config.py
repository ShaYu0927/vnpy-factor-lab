"""Configuration for reproducible, bounded expression generation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

from vnpy.alpha.expression import ConstantNode, ExpressionParser, OperatorCategory, WindowNode, create_default_registry


def positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class SearchSpace:
    """First-stage search uses current/past market fields and time-series operators."""

    fields: tuple[str, ...] = ("open", "high", "low", "close", "volume")
    operators: tuple[str, ...] = (
        "add", "sub", "mul", "div", "ts_delay", "ts_delta", "ts_mean", "ts_std",
    )
    constants: tuple[int | float, ...] = (-1, 0, 1, 2)
    windows: tuple[int, ...] = (5, 10, 20)
    max_depth: int = 4
    max_nodes: int = 20
    max_lookback: int = 60

    def __post_init__(self) -> None:
        for name in ("fields", "operators", "constants", "windows"):
            values = getattr(self, name)
            if not isinstance(values, (list, tuple)):
                raise ValueError(f"{name} must be a list or tuple")
            object.__setattr__(self, name, tuple(values))
        for name in ("max_depth", "max_nodes", "max_lookback"):
            positive_int(name, getattr(self, name))
        if self.max_depth > 12:
            raise ValueError("max_depth must not exceed 12 in the initial generator")
        if self.max_nodes > 255:
            raise ValueError("max_nodes must not exceed 255 in the initial generator")
        allowed_fields = {"open", "high", "low", "close", "volume", "amount", "turn", "vwap", "market_cap"}
        if not self.fields or any(not isinstance(x, str) or x not in allowed_fields for x in self.fields):
            raise ValueError("fields must contain supported market fields only; labels are not allowed")
        if len(set(self.fields)) != len(self.fields):
            raise ValueError("fields must be unique")
        if not self.windows:
            raise ValueError("windows must not be empty")
        for value in self.windows:
            WindowNode(value)
        for value in self.constants:
            ConstantNode(value)
        if len(set(self.windows)) != len(self.windows) or len(set(self.constants)) != len(self.constants):
            raise ValueError("windows and constants must be unique")
        registry = create_default_registry()
        names = []
        for name in self.operators:
            if not isinstance(name, str):
                raise ValueError("operator names must be strings")
            canonical = ExpressionParser.ALIASES.get(name, name)
            if canonical not in registry:
                raise ValueError(f"unknown operator: {name}")
            if registry.get(canonical).category is OperatorCategory.CROSS_SECTION:
                raise ValueError("cross-sectional search is not supported in this stage")
            names.append(canonical)
        if len(set(names)) != len(names):
            raise ValueError("operators must be unique after alias normalization")
        object.__setattr__(self, "operators", tuple(names))


@dataclass(frozen=True)
class ExpressionIterationConfig:
    search_space: SearchSpace = field(default_factory=SearchSpace)
    seed: int = 42
    batches: int = 5
    batch_size: int = 20
    max_attempts_per_batch: int = 2000
    output_root: str = "data/alpha_mining"

    def __post_init__(self) -> None:
        if not isinstance(self.search_space, SearchSpace):
            raise ValueError("search_space must be a SearchSpace")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        for name in ("batches", "batch_size", "max_attempts_per_batch"):
            positive_int(name, getattr(self, name))
        if not isinstance(self.output_root, str) or not self.output_root.strip():
            raise ValueError("output_root must be a non-empty path")


def load_iteration_config(path: str | Path) -> ExpressionIterationConfig:
    """Resolve output_root relative to the configuration file, independent of the IDE cwd."""
    path = Path(path).expanduser().resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    return parse_iteration_config(raw, path.parent)


def parse_iteration_config(raw: dict, base_dir: str | Path) -> ExpressionIterationConfig:
    """Validate expression-generation settings and resolve output paths."""
    if not isinstance(raw, dict):
        raise ValueError("iteration config must be an object")
    raw = dict(raw)
    space = raw.pop("search_space", {})
    if not isinstance(space, dict):
        raise ValueError("search_space must be an object")
    try:
        config = ExpressionIterationConfig(search_space=SearchSpace(**space), **raw)
    except TypeError as exc:
        raise ValueError(f"invalid expression iteration config: {exc}") from exc
    output = Path(config.output_root).expanduser()
    if not output.is_absolute():
        output = Path(base_dir) / output
    return replace(config, output_root=str(output.resolve()))


def runtime_iteration_config(raw: object, base_dir: str | Path) -> ExpressionIterationConfig | None:
    """Runtime opt-in: absent/disabled settings preserve the fixed-alpha workflow."""
    if not isinstance(raw, dict):
        raise ValueError("expression_iteration must be an object")
    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("expression_iteration.enabled must be a boolean")
    if not enabled:
        return None
    return parse_iteration_config({key: value for key, value in raw.items() if key != "enabled"}, base_dir)
