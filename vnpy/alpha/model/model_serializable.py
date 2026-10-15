from __future__ import annotations

import pickle
from pathlib import Path
from typing import TypeVar


SerializableT = TypeVar("SerializableT", bound="Serializable")


class Serializable:
    """Mixin for persisting standalone model objects with pickle."""

    def save(self, path: str | Path) -> None:
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls: type[SerializableT], path: str | Path) -> SerializableT:
        with Path(path).expanduser().resolve().open("rb") as file:
            value = pickle.load(file)
        if not isinstance(value, cls):
            raise TypeError(f"serialized object is not a {cls.__name__}")
        return value
