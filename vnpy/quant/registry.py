from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .config import ComponentSpec


class ComponentRegistry:
    """Allowlisted component factory registry used by config assembly."""

    def __init__(self) -> None:
        self._factories: dict[tuple[str, str], Callable[..., object]] = {}

    def register(
        self,
        kind: str,
        name: str,
        factory: Callable[..., object],
        *,
        replace: bool = False,
    ) -> None:
        key = (_normalize(kind), _normalize(name))
        if key in self._factories and not replace:
            raise ValueError(f"component already registered: {kind}/{name}")
        self._factories[key] = factory

    def create(self, kind: str, spec: ComponentSpec, **overrides: Any) -> object:
        key = (_normalize(kind), _normalize(spec.type))
        try:
            factory = self._factories[key]
        except KeyError as exc:
            available = ", ".join(self.names(kind)) or "<none>"
            raise KeyError(
                f"unknown {kind} component {spec.type!r}; registered: {available}"
            ) from exc
        kwargs = dict(spec.kwargs)
        kwargs.update(overrides)
        return factory(**kwargs)

    def names(self, kind: str) -> tuple[str, ...]:
        normalized_kind = _normalize(kind)
        return tuple(sorted(name for item_kind, name in self._factories if item_kind == normalized_kind))


def _normalize(value: str) -> str:
    normalized = value.strip().lower()
    if not normalized:
        raise ValueError("component kind and name must not be empty")
    return normalized
