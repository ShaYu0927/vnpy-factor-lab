from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from math import isfinite
from collections import defaultdict, deque
from typing import Any, Deque, Iterable, Mapping, Sequence

import polars as pl

from .dataset.utility import calculate_by_expression
from .definition import AlphaDefinition
from .alpha import Alpha
from .expression import CompiledExpression, PolarsCompiler, PolarsExecutor


@dataclass(frozen=True, slots=True)
class AlphaSample:
    """
    表示某只股票在某个时间点已经计算完成的一组 Alpha 因子值。

    AlphaSample 只保存当前时刻可以观测到的特征，不允许保存未来收益率等标签，
    避免在回测或实盘中产生未来数据泄漏。

    Attributes:
        symbol: 股票代码。
        datetime: 因子对应的行情时间。
        close: 当前收盘价。
        features: 因子名称到因子值的映射。
    """
    symbol: str
    datetime: datetime
    close: float
    features: Mapping[str, float | str | bool]

    def __getattr__(self, name: str) -> float | str | bool:
        """
        允许通过属性形式读取 features 中的因子
        Args:
            name: 因子名称
        Returns:
            对应的因子值
        Raises:
            AttributeError: features 中不存在该因子
        """
        try:
            return self.features[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class AlphaSampleCache:
    """
    按股票缓存最近计算完成的 AlphaSample

    每只股票对应一个固定长度的双端队列，超过 maxlen 后会自动删除最旧数据
    缓存只保存可观测的 Alpha 特征，不生成或保存未来标签
    """

    def __init__(self, maxlen: int = 30_000) -> None:
        self._samples: dict[str, Deque[AlphaSample]] = defaultdict(lambda: deque(maxlen=maxlen))

    def add(self, sample: AlphaSample) -> None:
        samples = self._samples[sample.symbol]
        if samples and samples[-1].datetime == sample.datetime:
            samples[-1] = sample
        else:
            samples.append(sample)

    def get_latest(self, symbol: str) -> AlphaSample | None:
        samples = self._samples.get(symbol)
        return samples[-1] if samples else None

    def get_recent(self, symbol: str, count: int) -> list[AlphaSample]:
        return list(self._samples.get(symbol, ())) [-max(count, 0):] if count > 0 else []

    def symbols(self) -> list[str]:
        return list(self._samples)

    def clear_all(self) -> None:
        self._samples.clear()


class AlphaEngine:
    """
    Alpha 因子计算引擎。
    该引擎统一处理离线历史数据和实盘数据，支持两种因子定义：
    1. Alpha:
       使用公式字符串定义，经过解析、表达树构建和 Polars 编译后执行。
    2. AlphaDefinition:
       已经包含底层表达式的旧式或兼容性定义，通过
       calculate_by_expression() 执行
    基本执行链路：
        Alpha公式
            -> 表达树
            -> PolarsCompiler
            -> CompiledExpression
            -> PolarsExecutor
            -> 因子结果
    """

    def __init__(self, definitions: Sequence[AlphaDefinition | Alpha]) -> None:
        names = [item.name for item in definitions]
        if len(names) != len(set(names)):
            raise ValueError("alpha names must be unique")
        if set(names) & {"datetime", "vt_symbol"}:
            raise ValueError("alpha names must not shadow index columns")
        resolved: list[AlphaDefinition] = []
        self._compiled: dict[str, CompiledExpression] = {}
        for item in definitions:
            if isinstance(item, Alpha):
                try:
                    tree = item.parse()
                    self._compiled[item.name] = PolarsCompiler().compile(tree)
                    resolved.append(AlphaDefinition.from_tree(item.name, tree))
                except (TypeError, ValueError, KeyError) as exc:
                    raise ValueError(f"alpha {item.name!r}, formula {item.formula!r}: {exc}") from exc
            else:
                resolved.append(item)
        # Freeze definitions at construction, including the inferred lookbacks.
        self.definitions = tuple(resolved)

    @property
    def min_bars(self) -> int:
        """
        返回计算全部 Alpha 至少需要准备的历史K线数量。

        取所有因子 lookback 的最大值。例如因子分别需要5、20、60根K线
        则整个引擎至少需要准备60根K线。

        Returns:
            最大历史回看周期:没有因子时返回1。
        """
        return max((item.lookback for item in self.definitions), default=1)

    """
    判断当前因子集合是否需要横截面数据。

    Rank、行业中性化等横截面算子需要同一时间点的多只股票
    同时参与计算，不能只传入单只股票数据。

    Returns:
        只要存在一个横截面因子就返回 True。
    """
    @property
    def requires_cross_section(self) -> bool:
        return any(item.uses_cross_section for item in self.definitions)

    """
    批量计算历史行情中的全部 Alpha 因子

    输入数据经过主键检查和排序后，分别执行已经编译的新式
    Alpha 表达式和兼容式 AlphaDefinition 表达式，最后按照 datetime、vt_symbol 合并所有因子结果。

    Args:
        frame: 包含 datetime、vt_symbol、close 以及因子所需
            行情字段的 Polars DataFrame

    Returns:
        包含 datetime、vt_symbol 和全部因子值的 DataFrame
    """
    def calculate(self, frame: pl.DataFrame) -> pl.DataFrame:
        source = self._normalize_frame(frame)
        result = source.select(["datetime", "vt_symbol"])
        executor = PolarsExecutor(source) if self._compiled else None
        for definition in self.definitions:
            if definition.name in self._compiled:
                assert executor is not None
                values = executor.run(self._compiled[definition.name], name=definition.name)
            else:
                values = calculate_by_expression(source, definition.expression).rename({"data": definition.name})
            result = result.join(values, on=["datetime", "vt_symbol"], how="left")
        return result.sort(["datetime", "vt_symbol"])

    """
    计算每只股票在指定时间或最新时间的因子快照。

    Args:
        frame: 历史行情数据。
        at: 指定计算时刻。为空时返回每只股票的最新结果。

    Returns:
        因子完整且数值有效的 AlphaSample 列表。
    """
    def calculate_latest(self, frame: pl.DataFrame, at: datetime | None = None) -> list[AlphaSample]:
        source = self._normalize_frame(frame)
        calculated = self.calculate(source)
        return self._latest_samples(source, calculated, [item.name for item in self.definitions], at)

    """
    将因子计算结果转换为最新的 AlphaSample。

    每只股票只保留指定时刻或最新时刻的数据，并补充对应的
    收盘价。任何因子为空、NaN 或无穷大的样本都会被跳过。

    Args:
        source: 标准化后的原始行情。
        calculated: 已经计算完成的因子数据。
        names: 要写入样本的因子名称。
        at: 指定时间；为空时取每只股票的最新时间。

    Returns:
        按股票代码排序的有效 AlphaSample 列表。
    """
    def _latest_samples(self, source: pl.DataFrame, calculated: pl.DataFrame, names: Sequence[str], at: datetime | None,) -> list[AlphaSample]:
        if at is None:
            latest = calculated.group_by("vt_symbol").agg(pl.all().sort_by("datetime").last())
        else:
            latest = calculated.filter(pl.col("datetime") == pl.lit(at))
        closes = source.select(["datetime", "vt_symbol", "close"])
        latest = latest.join(closes, on=["datetime", "vt_symbol"], how="left")
        samples: list[AlphaSample] = []
        for row in latest.iter_rows(named=True):
            features = {
                name: float(row[name])
                for name in names
                if row.get(name) is not None and isfinite(float(row[name]))
            }
            if len(features) != len(names):
                continue
            samples.append(AlphaSample(
                symbol=str(row["vt_symbol"]), datetime=row["datetime"],
                close=float(row["close"]), features=features,
            ))
        return sorted(samples, key=lambda item: item.symbol)

    """
    将按股票组织的 Bar 对象转换为 Polars DataFrame。

    支持字典和普通对象形式的 Bar, 并提取基础行情字段以及
    vwap、市值、行业等可选字段。

    Args:
        bars_by_symbol: 股票代码到历史 Bar 序列的映射

    Returns:
        按股票代码和时间排序的行情 DataFrame
    """
    def from_bars(self, bars_by_symbol: Mapping[str, Iterable[Any]]) -> pl.DataFrame:
        rows = []
        for symbol, bars in bars_by_symbol.items():
            for bar in bars:
                dt = _bar_value(bar, "datetime", "bob", "eob")
                row = {"datetime": dt, "vt_symbol": symbol}
                for field in ("open", "high", "low", "close", "volume", "amount", "turn"):
                    value = _bar_value(bar, field, required=False)
                    if value is not None:
                        row[field] = value
                extra = _bar_value(bar, "extra", required=False) or {}
                for field in ("vwap", "market_cap", "industry", "sector", "subindustry"):
                    value = _bar_value(bar, field, required=False)
                    if value is None:
                        value = extra.get(field)
                    if value is not None:
                        row[field] = value
                rows.append(row)
        if not rows:
            return pl.DataFrame({"datetime": [], "vt_symbol": []})
        return pl.DataFrame(rows).sort(["vt_symbol", "datetime"])

    """
    将 AlphaSample 转换为模型层使用的 FactorObservation。

    该方法用于连接因子计算层与后续的模型训练、股票评分
    和交易信号生成模块。

    Args:
        samples: 已经计算完成的因子快照。

    Returns:
        FactorObservation 列表。
    """
    def to_observations(self, samples: Sequence[AlphaSample]):
        from .modeling.schema import FactorObservation

        return [
            FactorObservation(
                trade_date=_to_date(sample.datetime), symbol=sample.symbol,
                close=sample.close, features=dict(sample.features),
            )
            for sample in samples
        ]

    """
    检查并标准化因子计算所需的行情数据。

    检查必要字段、空主键和重复主键，然后按照股票代码及
    时间排序，保证 Ref、Mean、Delta 等时序算子顺序正确。

    Args:
        frame: 原始行情 DataFrame。

    Returns:
        排序完成的行情 DataFrame。

    Raises:
        ValueError: 缺少必要字段、主键为空或存在重复记录。
    """
    @staticmethod
    def _normalize_frame(frame: pl.DataFrame) -> pl.DataFrame:
        required = {"datetime", "vt_symbol", "close"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"alpha input is missing columns: {', '.join(sorted(missing))}")
        if frame["datetime"].null_count() or frame["vt_symbol"].null_count():
            raise ValueError("alpha input keys must not be null")
        duplicates = frame.group_by(["datetime", "vt_symbol"]).len().filter(pl.col("len") > 1)
        if duplicates.height:
            raise ValueError("alpha input contains duplicate datetime/symbol rows")
        return frame.sort(["vt_symbol", "datetime"])

    """
    从字典或普通对象中读取第一个有效字段值。

    Args:
        bar: 字典形式或属性形式的 Bar 对象。
        *names: 按优先级排列的候选字段名。
        required: 找不到字段时是否抛出异常。

    Returns:
        第一个非空字段值；非必需字段不存在时返回 None。

    Raises:
        ValueError: 必需字段不存在。
    """
def _bar_value(bar: Any, *names: str, required: bool = True) -> Any:
    for name in names:
        value = bar.get(name) if isinstance(bar, Mapping) else getattr(bar, name, None)
        if value is not None:
            return value
    if required:
        raise ValueError(f"bar is missing required field: {'/'.join(names)}")
    return None


def _to_date(value: datetime) -> date:
    return value.date()
