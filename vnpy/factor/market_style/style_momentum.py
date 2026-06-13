from dataclasses import dataclass
from typing import Any, Optional
from vnpy.factor.core.factorEngine import Factor, FactorContext


@dataclass
class StyleMomentumResult:
    """
    Result of one style momentum calculation.
    """
    symbol: str
    window: int
    ret_n: float = 0.0
    style_momentum: float = 0.0
    style_label: str = "UNKNOWN"
    reason: str = ""


class StyleMomentumFactor:
    """
    Style momentum factor.

    This is the algorithm class. Keep concrete calculation logic here.

    Current template logic:
        1. Read close prices from bar data.
        2. Calculate latest N-bar return.
        3. Classify it as STRONG / WEAK / NEUTRAL.

    Later you can replace _calculate_style_momentum() with real style logic,
    such as large/small cap style, value/growth style, industry style,
    or cross-section style portfolio momentum.
    """
    def __init__(self, window: int = 20, strong_threshold: float = 0.02, weak_threshold: float = -0.02,) -> None:
        self.window = window
        self.strong_threshold = strong_threshold
        self.weak_threshold = weak_threshold

    def calculate(self, symbol: str, data: Any, context: Optional[FactorContext] = None,) -> StyleMomentumResult:
        result = StyleMomentumResult(symbol=symbol, window=self.window)

        if context is None:
            result.reason = "Missing FactorContext"
            return result

        style_label = self._get_symbol_style(symbol, context)
        if not style_label:
            result.reason = f"No style label found for symbol={symbol}"
            return result

        style_momentum = self._get_style_momentum(style_label, context)
        if style_momentum is None:
            result.reason = f"No style momentum found for style={style_label}"
            result.style_label = style_label
            return result

        result.style_label = style_label
        result.style_momentum = style_momentum
        result.strength_label = self._classify(style_momentum)
        result.reason = self._build_reason(result)

        return result
    
    def _get_symbol_style(self, symbol: str, context: FactorContext,) -> Optional[str]:
        """
        从 context 中读取 symbol 对应的风格标签
        """
        symbol_style_map = self._get_context_value(context, "symbol_style_map", {})
        return symbol_style_map.get(symbol)
    
    def _get_style_momentum(self, style_label: str, context: FactorContext) -> Optional[float]:
        """
        从 context 中读取某个风格的动量值
        """
        style_momentum_map = self._get_context_value(context, "style_momentum_map", {})
        value = style_momentum_map.get(style_label)

        if value is None:
            return None

        return float(value)
    
    def _get_context_value(self, context: FactorContext, field: str, default: Any = None,) -> Any:
        """
        兼容 object context 和 dict context。
        """
        if isinstance(context, dict):
            return context.get(field, default)

        return getattr(context, field, default)

    def _extract_closes(self, data: Any) -> list[float]:
        """
        Extract close prices from object bars or dict bars.
        """
        if data is None:
            return []

        closes: list[float] = []
        for bar in data:
            value = self._get_value(bar, "close")
            if value is None:
                continue
            closes.append(float(value))

        return closes

    def _calculate_style_momentum(self, closes: list[float]) -> Optional[float]:
        """
        Calculate raw style momentum value.

        Override or replace this method when the factor becomes a real
        cross-section style factor.
        """
        close_now = closes[-1]
        close_n = closes[-(self.window + 1)]

        if close_n <= 0:
            return None

        return close_now / close_n - 1.0

    def _classify(self, momentum: float) -> str:
        if momentum >= self.strong_threshold:
            return "STRONG"

        if momentum <= self.weak_threshold:
            return "WEAK"

        return "NEUTRAL"

    def _build_reason(self, style_label: str, momentum: float) -> str:
        return (
            f"style_momentum={momentum:.6f}, "
            f"label={style_label}, "
            f"window={self.window}"
        )

    def _get_value(self, bar: Any, field: str) -> Any:
        if isinstance(bar, dict):
            return bar.get(field)

        return getattr(bar, field, None)


class StyleMomentumEngineFactor(Factor):
    """
    FactorEngine adapter for StyleMomentumFactor.

    FactorEngine only knows the Factor interface:
        name
        min_bars
        validate()
        calculate()

    The adapter should stay thin. Put formulas in StyleMomentumFactor.
    """

    def __init__(self, window: int = 20, strong_threshold: float = 0.02, weak_threshold: float = -0.02,) -> None:
        self.window = window
        self.name = f"style_momentum_{window}"
        self.min_bars = window + 1
        self.factor = StyleMomentumFactor(
            window=window,
            strong_threshold=strong_threshold,
            weak_threshold=weak_threshold,
        )

    def validate(self, symbol: str, data: Any, context: FactorContext) -> bool:
        if context is None:
            return False

        symbol_style_map = getattr(context, "symbol_style_map", None)
        style_momentum_map = getattr(context, "style_momentum_map", None)

        if symbol_style_map is None or style_momentum_map is None:
            return False

        style_label = symbol_style_map.get(symbol)
        if not style_label:
            return False

        return style_label in style_momentum_map

    def calculate(self, symbol: str, data: Any, context: FactorContext,) -> StyleMomentumResult:
        return self.factor.calculate(symbol=symbol, data=data, context=context)
