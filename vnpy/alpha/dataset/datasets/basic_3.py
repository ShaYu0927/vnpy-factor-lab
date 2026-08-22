from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..loader import FeatureConfig, QlibDataLoader


class Basic3DL(QlibDataLoader):
    """Qlib-style loader for the three factors used by batch training."""

    _FIELDS: Mapping[str, str] = {
        "momentum_20": "$close/Ref($close, 20)-1",
        "volatility_20": "Std($close/Ref($close, 1)-1, 20)",
        "volume_20": "$volume/(Mean($volume, 20)+1e-12)",
    }

    def __init__(
        self,
        config: Mapping[str, FeatureConfig] | None = None,
        feature_names: Sequence[str] | None = None,
    ) -> None:
        loader_config: dict[str, FeatureConfig] = {
            "feature": self.get_feature_config(feature_names),
        }
        if config is not None:
            loader_config.update(config)
        super().__init__(config=loader_config)

    @classmethod
    def get_feature_config(
        cls,
        feature_names: Sequence[str] | None = None,
    ) -> tuple[list[str], list[str]]:
        names = list(feature_names or cls._FIELDS)
        unknown = sorted(set(names) - cls._FIELDS.keys())
        if unknown:
            raise ValueError(f"unknown Basic3 factor names: {', '.join(unknown)}")
        fields = [cls._FIELDS[name] for name in names]
        return fields, names

    @classmethod
    def supported_features(cls) -> tuple[str, ...]:
        """Return the feature names available through this loader."""
        return tuple(cls._FIELDS)
