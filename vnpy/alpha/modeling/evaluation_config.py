"""Runtime settings for descriptive factor evaluation after replay/import."""

from dataclasses import dataclass, replace
from pathlib import Path
import math


@dataclass(frozen=True)
class FactorEvaluationConfig:
    periods: tuple[int, ...] = (1, 5, 10)
    entry_offset: int = 1
    quantiles: int = 5
    min_assets: int = 10
    train_fraction: float = 0.7
    output_root: str = "../data/alpha_evaluation"

    def __post_init__(self):
        if not isinstance(self.periods, (list, tuple)) or not self.periods:
            raise ValueError("factor_evaluation.periods must be a non-empty list")
        for value in (*self.periods, self.entry_offset, self.quantiles, self.min_assets):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("evaluation periods, entry_offset, quantiles and min_assets must be positive integers")
        if self.quantiles < 2 or self.min_assets < self.quantiles:
            raise ValueError("factor_evaluation requires min_assets >= quantiles >= 2")
        object.__setattr__(self, "periods", tuple(sorted(set(self.periods))))
        if isinstance(self.train_fraction, bool) or not isinstance(self.train_fraction, (int, float)) or not math.isfinite(self.train_fraction) or not 0 < self.train_fraction < 1:
            raise ValueError("factor_evaluation.train_fraction must be between zero and one")
        if not isinstance(self.output_root, str) or not self.output_root.strip():
            raise ValueError("factor_evaluation.output_root must be a non-empty path")


def parse_evaluation_config(raw: object, base_dir: Path) -> FactorEvaluationConfig | None:
    if not isinstance(raw, dict):
        raise ValueError("factor_evaluation must be an object")
    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("factor_evaluation.enabled must be a boolean")
    if not enabled:
        return None
    try:
        config = FactorEvaluationConfig(**{key: value for key, value in raw.items() if key != "enabled"})
    except TypeError as exc:
        raise ValueError(f"invalid factor_evaluation configuration: {exc}") from exc
    root = Path(config.output_root).expanduser()
    if not root.is_absolute():
        root = base_dir / root
    return replace(config, output_root=str(root.resolve()))
