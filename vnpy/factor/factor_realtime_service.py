"""将实时行情 Bar 接入统一的 Alpha 引擎，使用缓存历史计算最新因子。"""

from __future__ import annotations

from datetime import datetime
from typing import Mapping, Sequence

from vnpy.alpha.definition import AlphaDefinition
from vnpy.alpha.alpha import Alpha
from vnpy.alpha.engine import AlphaEngine, AlphaSample, AlphaSampleCache
from vnpy.datafeed.data_bar_cache import BarCache
from vnpy.factor.core.factor_engine import FactorBatchResult, FactorValue


class RealtimeAlphaService:
    """管理实时行情缓存、因子批量计算和计算结果缓存。"""

    def __init__(
        self,
        bar_cache: BarCache,
        sample_cache: AlphaSampleCache,
        definitions: Sequence[AlphaDefinition | Alpha] = (),
        frequency: str = "60s",
        universe: Sequence[str] | None = None,
        alpha_engine: AlphaEngine | None = None,
        optional_names: Sequence[str] = (),
        **_legacy_options,
    ) -> None:
        """
        初始化实时因子计算服务。

        参数:
            bar_cache: 保存各股票历史 Bar 的行情缓存。
            sample_cache: 保存计算结果的 Alpha 样本缓存。
            definitions: 因子定义，用于创建默认 Alpha 引擎。
            frequency: 服务读取缓存时使用的行情周期。
            universe: 固定股票池；截面计算要求股票池完整且时间同步。
            alpha_engine: 可选的外部 Alpha 引擎。
            optional_names: 可选因子名称，不参与必需历史长度的计算。
            _legacy_options: 接收旧接口参数，当前实现不使用。
        """
        self.bar_cache = bar_cache
        self.sample_cache = sample_cache
        self.frequency = frequency
        self.definitions = tuple(definitions)

        # 优先使用外部引擎，否则根据传入的因子定义创建引擎。
        self.alpha_engine = alpha_engine or AlphaEngine(self.definitions)
        self.optional_names = tuple(optional_names)

        # 可选因子必须已经注册到实际使用的引擎中。
        if set(self.optional_names) - {
            item.name for item in self.alpha_engine.definitions
        }:
            raise ValueError("optional alpha names must be registered definitions")

        # 取所有非可选因子的最大回看长度，作为最低历史数量要求。
        # 没有非可选因子时，默认至少需要一根 Bar。
        self.required_min_bars = max(
            (
                item.lookback
                for item in self.alpha_engine.definitions
                if item.name not in self.optional_names
            ),
            default=1,
        )

        # 股票池去重并保留原始顺序，集合用于快速判断股票是否属于池内。
        self.universe = tuple(dict.fromkeys(universe or ()))
        self._universe_set = frozenset(self.universe)

        # 保存最近一次 on_bars 调用生成的两种结果表示。
        self.latest_batch_result = FactorBatchResult()
        self.latest_samples: list[AlphaSample] = []

    def on_bar(self, bar) -> AlphaSample | None:
        """
        接收单根 Bar，通过批量入口更新行情并计算因子。

        返回:
            当前股票的 Alpha 样本；没有生成对应结果时返回 None。
        """
        samples = self.on_bars([bar] if bar is not None else [])

        # 从批量结果中查找当前股票的样本。
        return next(
            (sample for sample in samples if sample.symbol == bar.symbol),
            None,
        )

    def on_bars(self, bars: Sequence) -> list[AlphaSample]:
        """
        接收同一时间戳的一批 Bar 使用各股票缓存历史计算最新因子。

        流程:
            校验输入 → 更新行情缓存 → 收集历史 → 计算 → 保存样本。

        返回:
            本次生成的 Alpha 样本；无可计算数据时返回空列表。

        异常:
            ValueError: 批次时间戳不一致、股票重复或行情发生时间倒退。
        """
        # 每次调用先清空上一次结果，避免提前返回时残留旧结果。
        self.latest_samples = []
        self.latest_batch_result = FactorBatchResult()

        # 排除空 Bar；配置股票池时，只接收池内股票。
        bars = [
            bar
            for bar in bars
            if bar is not None
            and (
                not self._universe_set
                or bar.symbol in self._universe_set
            )
        ]

        # 没有有效行情或服务没有配置因子时，不更新缓存、不计算。
        if not bars or not self.definitions:
            return []

        # 一个批次只能对应同一时间点。
        at = _bar_datetime(bars[0])
        if any(_bar_datetime(bar) != at for bar in bars):
            raise ValueError("a bar batch must contain a single timestamp")

        # 同一批次内，每只股票只能出现一次。
        symbols = tuple(bar.symbol for bar in bars)
        if len(set(symbols)) != len(symbols):
            raise ValueError("a bar batch must contain unique symbols")

        # 先校验全部 Bar，再统一更新缓存。
        for bar in bars:
            # 输入未提供周期时，补上服务配置的默认周期。
            if not getattr(bar, "frequency", None):
                bar.frequency = self.frequency

            # 拒绝早于缓存最新时间的行情；同时间戳行情不在此处拒绝。
            previous = self.bar_cache.get_last_bar(
                bar.symbol,
                self.frequency,
            )
            if previous is not None and _bar_datetime(previous) > at:
                raise ValueError(f"out-of-order bar for {bar.symbol}")

        for bar in bars:
            self.bar_cache.update(bar)

        # 时序因子只收集本批次股票。
        # 截面因子则收集完整股票池，供排名等跨股票运算使用。
        if self.alpha_engine.requires_cross_section:
            symbols = self.universe or tuple(self.bar_cache.symbols())

        # 为每只股票读取引擎要求的最近若干根历史 Bar。
        bars_by_symbol = {
            symbol: self.bar_cache.get_bars(
                symbol=symbol,
                frequency=self.frequency,
                count=self.alpha_engine.min_bars,
            )
            for symbol in symbols
        }

        # 剔除历史不足以计算必需因子的股票。
        bars_by_symbol = {
            symbol: bars
            for symbol, bars in bars_by_symbol.items()
            if len(bars) >= self.required_min_bars
        }
        if not bars_by_symbol:
            return []

        if self.alpha_engine.requires_cross_section:
            # 截面计算必须显式配置股票池，且所有股票历史数量达标。
            if not self.universe or len(bars_by_symbol) != len(self.universe):
                return []

            # 所有股票最新一根 Bar 必须到达当前时间点。
            # 尚未同步时先保留已更新的缓存，等待后续行情到达。
            if any(
                _bar_datetime(bars[-1]) != at
                for bars in bars_by_symbol.values()
            ):
                return []

        # 将各股票历史转换为引擎使用的表格数据。
        frame = self.alpha_engine.from_bars(bars_by_symbol)

        # 计算指定时间点的最新因子样本，并传入可选因子名单。
        samples = self.alpha_engine.calculate_latest(
            frame,
            at=at,
            optional_names=self.optional_names,
        )

        # 同时保留 Alpha 样本和旧 Factor 接口需要的结果格式。
        self.latest_batch_result = _to_factor_result(samples)
        self.latest_samples = samples

        # 将本次样本加入结果缓存，供后续策略或其他模块读取。
        for sample in samples:
            self.sample_cache.add(sample)

        return samples

    def calculate_latest_cross_section(
        self,
        symbols: Sequence[str],
        at: datetime | None = None,
        count: int | None = None,
        **_legacy_options,
    ) -> list[AlphaSample]:
        """
        从缓存读取指定股票的历史，直接计算最新因子样本。

        参数:
            symbols: 参与计算的股票列表。
            at: 目标时间，具体处理方式由 Alpha 引擎决定。
            count: 每只股票读取的历史数量，默认使用引擎最低要求。
            _legacy_options: 接收旧接口参数，当前实现不使用。

        说明:
            此入口不执行 on_bars 中的股票池完整性和时间同步检查，
            也不更新 latest_samples、latest_batch_result 或样本缓存。
        """
        # count 为 None 或 0 时，使用引擎默认历史数量。
        required = count or self.alpha_engine.min_bars

        # 读取指定股票、指定周期的缓存历史。
        bars = {
            symbol: self.bar_cache.get_bars(
                symbol=symbol,
                frequency=self.frequency,
                count=required,
            )
            for symbol in symbols
        }

        # 排除没有历史数据的股票，再转换为引擎输入表。
        frame = self.alpha_engine.from_bars(
            {key: value for key, value in bars.items() if value}
        )

        return self.alpha_engine.calculate_latest(
            frame,
            at=at,
            optional_names=self.optional_names,
        )

    def calculate(self, frame) -> object:
        """
        使用同一个 Alpha 引擎计算外部传入的数据表。

        供离线调用方使用，不读取或更新实时缓存。
        """
        return self.alpha_engine.calculate(frame)


def _bar_datetime(bar) -> datetime:
    """
    提取 Bar 的时间戳。

    按 datetime、bob、eob 的顺序取第一个真值字段，
    并要求最终取出的值是 datetime 对象。
    """
    value = (
        getattr(bar, "datetime", None)
        or getattr(bar, "bob", None)
        or getattr(bar, "eob", None)
    )
    if not isinstance(value, datetime):
        raise ValueError("bar must contain a datetime/bob/eob datetime")
    return value


def _to_factor_result(samples: Sequence[AlphaSample],) -> FactorBatchResult:
    """
    将 Alpha 样本转换为旧 Factor 接口的批量结果。

    每个样本中的每个因子生成一个 FactorValue
    保存股票、因子名称、因子值及样本时间。
    """
    values = [
        FactorValue(
            symbol=sample.symbol,
            factor_name=name,
            value=value,
            trade_date=sample.datetime.isoformat(),
            primary_field="value",
        )
        for sample in samples
        for name, value in sample.features.items()
    ]
    return FactorBatchResult(values=values)


# 保留旧类名，使原有导入和调用继续使用新的 Alpha 服务。
RealtimeFactorService = RealtimeAlphaService

__all__ = ["RealtimeAlphaService", "RealtimeFactorService"]