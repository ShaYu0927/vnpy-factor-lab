from __future__ import annotations

from collections import defaultdict
from importlib import import_module
from typing import Any, Callable, DefaultDict, Iterable, Optional, Sequence

from vnpy.event.event import EngineEvent, EventType
from vnpy.risk.RiskContext import OrderRequest
from vnpy.risk.Rule.Rule import RiskAction
from vnpy.strategy.StratContext import StrategyContext, StrategySignal, TargetPosition
from vnpy.strategy.StratTemplate import StrategyOutput, StrategyTemplate


# -----------------------------------------------------------------------------
# Event post callback type.
#
# 参数：
#   1. target module name，例如 "order" / "trade"。
#   2. EngineEvent 事件对象。
# 返回：
#   True  表示事件投递成功。
#   False 表示事件投递失败。
#
# StrategyEngine 不直接调用订单模块或交易模块，而是通过 post_event 把事件投递出去。
# 这样 StrategyEngine 和其他模块之间保持解耦。
# -----------------------------------------------------------------------------
PostEvent = Callable[[str, EngineEvent], bool]


class StrategyEngine:
    """
    Multi-strategy manager for the module event system.

    StrategyEngine 是策略层的调度中心，不是具体策略本身。

    核心职责：
    1. 管理策略生命周期：add / init / start / stop / remove。
    2. 维护策略订阅关系：
       - symbol_strategy_map：某个股票应该分发给哪些策略。
       - factor_strategy_map：某个因子应该分发给哪些策略。
    3. 接收外部事件：BAR / FACTOR / TRADE_SIGNAL / ORDER。
    4. 根据 symbol + factor_names 把事件分发给具体 StrategyTemplate。
    5. 接收策略输出：StrategySignal / TargetPosition。
    6. 在需要时统一创建 OrderRequest，并执行风险检查。

    推荐调用链：
        FactorEngine 计算因子
            -> 发 EVENT_FACTOR
            -> StrategyEngine.on_event()
            -> StrategyEngine.on_factor()
            -> 具体策略 strategy.on_factor()
            -> 策略输出 StrategySignal / TargetPosition
            -> StrategyEngine 转发给 trade_module
    """

    def __init__(
        self,
        post_event: PostEvent,
        source: str = "strategy",
        order_module: str = "order",
        trade_module: str | None = None,
        risk_engine: Any = None,
        risk_context: Any = None,
        strategy_context: StrategyContext | None = None,
    ) -> None:
        # 事件投递函数，由外部事件引擎传入。
        # StrategyEngine 本身不关心事件队列怎么实现，只负责调用这个回调。
        self.post_event = post_event

        # 当前模块名，后续发事件时作为 event.source。
        self.source = source

        # 订单模块名，send_order() 会把 ORDER 事件投递到这个模块。
        self.order_module = order_module

        # 交易模块名，策略输出信号后，会把 TRADE_SIGNAL 事件投递到这个模块。
        # 如果为 None，StrategyEngine 只记录 latest_outputs，不继续向外转发。
        self.trade_module = trade_module

        # 风控引擎和风控上下文。
        # 如果没有传入，则 send_order() 默认不做风控检查。
        self.risk_engine = risk_engine
        self.risk_context = risk_context

        # 策略共享上下文，里面通常放账户、持仓、行情缓存、参数、状态等信息。
        self.context = strategy_context or StrategyContext()

        # 所有策略实例：strategy_name -> StrategyTemplate。
        self.strategies: dict[str, StrategyTemplate] = {}

        # 当前处于 trading 状态的策略名称集合。
        # 只有 active_strategies 中的策略才会收到 BAR / FACTOR / SIGNAL 事件。
        self.active_strategies: set[str] = set()

        # 股票订阅表：symbol -> set(strategy_name)。
        # 例如："600519.SH" -> {"tail_close", "momentum"}
        # 特殊 key "*" 表示订阅全部 symbol。
        self.symbol_strategy_map: DefaultDict[str, set[str]] = defaultdict(set)

        # 因子订阅表：factor_name -> set(strategy_name)。
        # 例如："momentum_20" -> {"tail_close", "trend_follow"}
        # 特殊 key "*" 表示订阅全部 factor。
        self.factor_strategy_map: DefaultDict[str, set[str]] = defaultdict(set)

        # 最近一次事件处理产生的策略输出。
        # 注意：当前实现会在每次 on_event() 开始时清空。
        self.latest_outputs: list[StrategyOutput] = []

    def add_strategy(
        self,
        strategy_name: str,
        strategy_class: type[StrategyTemplate] | str,
        symbols: Sequence[str] | None = None,
        factor_names: Sequence[str] | None = None,
        setting: dict[str, Any] | None = None,
        auto_init: bool = True,
    ) -> StrategyTemplate:
        """
        Add one strategy instance into StrategyEngine.

        作用：
        1. 根据 strategy_class 创建具体策略对象。
        2. 保存到 self.strategies。
        3. 建立 symbol 订阅关系。
        4. 建立 factor 订阅关系。
        5. 如果 auto_init=True，自动调用 init_strategy()。

        参数说明：
        - strategy_name：策略实例名，必须全局唯一。
        - strategy_class：可以传类对象，也可以传 "module.ClassName" 字符串。
        - symbols：策略关注的股票列表；为空表示关注全部 symbol。
        - factor_names：策略依赖的因子列表；为空表示关注全部 factor。
        - setting：策略参数字典，会传给具体策略。
        - auto_init：添加策略后是否立即初始化。
        """
        if not strategy_name:
            raise ValueError("strategy_name must not be empty")

        if strategy_name in self.strategies:
            raise ValueError(f"strategy already exists: {strategy_name}")

        # 支持两种方式：
        # 1. 直接传 Strategy 类。
        # 2. 传 "package.module.StrategyClass" 字符串，运行时动态 import。
        cls = self._resolve_strategy_class(strategy_class)

        # 创建策略实例。
        # 这里要求具体策略类的 __init__ 支持这些参数。
        strategy = cls(
            strategy_engine=self,
            strategy_name=strategy_name,
            symbols=symbols or [],
            factor_names=factor_names or [],
            setting=setting or {},
        )

        # 保存策略实例。
        self.strategies[strategy_name] = strategy

        # 建立 symbol -> strategy 映射。
        # 后续 BAR / SIGNAL / ORDER 事件按 symbol 分发时会用到。
        self._bind_symbols(strategy_name, strategy.symbols)

        # 建立 factor -> strategy 映射。
        # 后续 FACTOR 事件按 factor_names 分发时会用到。
        self._bind_factors(strategy_name, strategy.factor_names)

        if auto_init:
            self.init_strategy(strategy_name)

        return strategy

    def remove_strategy(self, strategy_name: str) -> bool:
        """
        Remove one strategy instance.

        删除策略时要做完整清理：
        1. 如果策略正在运行，先 stop。
        2. 删除 symbol 订阅关系。
        3. 删除 factor 订阅关系。
        4. 从 strategies 中移除实例。
        """
        strategy = self.strategies.get(strategy_name)
        if strategy is None:
            return False

        self.stop_strategy(strategy_name)
        self._unbind_symbols(strategy_name)
        self._unbind_factors(strategy_name)
        del self.strategies[strategy_name]
        return True

    def init_strategy(self, strategy_name: str) -> bool:
        """
        Initialize strategy.

        只做初始化，不代表策略开始交易。

        一般用于：
        - 加载历史数据。
        - 初始化内部状态。
        - 初始化模型对象。
        - 读取策略参数。

        调用具体策略的：
            strategy.on_init(context)
        """
        strategy = self.strategies.get(strategy_name)
        if strategy is None:
            return False

        if strategy.inited:
            return True

        strategy.on_init(self.context)
        strategy.inited = True
        return True

    def start_strategy(self, strategy_name: str) -> bool:
        """
        Start strategy trading.

        策略只有 start 之后才会进入 active_strategies，
        也只有 active strategy 才会收到 BAR / FACTOR / SIGNAL 分发。
        """
        strategy = self.strategies.get(strategy_name)
        if strategy is None:
            return False

        # 如果还没初始化，启动前先初始化。
        if not strategy.inited:
            self.init_strategy(strategy_name)

        if strategy.trading:
            return True

        strategy.trading = True
        self.active_strategies.add(strategy_name)
        strategy.on_start(self.context)
        return True

    def stop_strategy(self, strategy_name: str) -> bool:
        """
        Stop strategy trading.

        stop 后策略仍然保留在 self.strategies 中，
        但是不再接收 active 事件分发。
        """
        strategy = self.strategies.get(strategy_name)
        if strategy is None:
            return False

        if not strategy.trading:
            return True

        strategy.trading = False
        self.active_strategies.discard(strategy_name)
        strategy.on_stop(self.context)
        return True

    def start_all(self) -> None:
        """
        Start all registered strategies.
        """
        # list(self.strategies) 是为了避免遍历过程中 dict 被修改导致异常。
        for strategy_name in list(self.strategies):
            self.start_strategy(strategy_name)

    def stop_all(self) -> None:
        """
        Stop all active strategies.
        """
        # 只停止 active_strategies，避免重复调用未启动策略。
        for strategy_name in list(self.active_strategies):
            self.stop_strategy(strategy_name)

    def on_event(self, event: EngineEvent) -> None:
        """
        Main event entrance of StrategyEngine.

        外部 EventEngine 收到事件后，可以统一调用 StrategyEngine.on_event(event)。
        StrategyEngine 根据 event_type 再分发到对应处理函数。

        当前支持：
        - BAR：行情 K 线事件。
        - FACTOR：因子结果事件。
        - TRADE_SIGNAL：交易信号事件。
        - ORDER：订单回报 / 订单事件。
        """
        # 每次处理新事件前，清空上次的策略输出。
        self.latest_outputs = []

        if event.event_type == EventType.BAR:
            self.on_bar(event.get("bar"), event.symbol)
            return

        if event.event_type == EventType.FACTOR:
            self.on_factor(
                sample=event.get("sample"),
                factor_result=event.get("factor_result"),
                symbol=event.symbol,
            )
            return

        if event.event_type == EventType.TRADE_SIGNAL:
            self.on_signal(event.get("signal"), event.symbol)
            return

        if event.event_type == EventType.ORDER:
            self.on_order(event.get("order"), event.symbol)

    def on_bar(self, bar: Any, symbol: str | None = None) -> None:
        """
        Handle BAR event.

        分发规则：
        1. 从 event.symbol 或 bar.symbol 获取 symbol。
        2. 找到订阅该 symbol 的 active 策略。
        3. 调用 strategy.on_bar(context, bar)。
        4. 收集并发布策略输出。
        """
        # symbol 优先使用外部传入值；没有时从 bar 对象中读取。
        event_symbol = symbol or getattr(bar, "symbol", None)

        for strategy in self._iter_active_by_symbol(event_symbol):
            self._publish_outputs(strategy.on_bar(self.context, bar))

    def on_factor(self, sample: Any, factor_result: Any = None, symbol: str | None = None,) -> None:
        """
        Handle FACTOR event.

        这是因子层和策略层衔接的核心函数

        推荐约定：
        - FactorEngine 每次发送一个完整的 FactorSnapshot / FactorResult
        - factor_result.values 中包含本次可用的所有因子值
        - StrategyEngine 根据 factor_names 判断哪些策略可以被触发

        分发规则：
        1. 获取事件所属 symbol。
        2. 从 factor_result 中提取本次事件包含的 factor_names。
        3. 找到同时满足以下条件的策略：
           - 已启动 active。
           - 订阅了该 symbol 或订阅了 "*"
           - 订阅了本次 factor 或订阅了 "*"
           - 策略 required factors 都存在于本次 factor_names 中
        4. 调用 strategy.on_factor(context, sample, factor_result)

        注意：
        如果 FactorEngine 是“单因子一个事件”，则 required.issubset(factor_names)
        可能导致多因子策略无法触发。多因子策略更适合一次发送完整因子快照。
        """
        event_symbol = symbol or getattr(sample, "symbol", None)
        factor_names = self._get_factor_names(factor_result)

        for strategy in self._iter_active_by_factor(event_symbol, factor_names):
            self._publish_outputs(strategy.on_factor(self.context, sample, factor_result))

    def on_signal(self, signal: Any, symbol: str | None = None) -> None:
        """
        Handle TRADE_SIGNAL event.

        这个入口用于策略之间、模块之间传递信号。
        例如：
        - 一个上游策略产生信号。
        - 另一个策略订阅同一 symbol 后继续加工信号。

        当前分发只按 symbol 不按 factor
        """
        event_symbol = symbol or getattr(signal, "symbol", None)

        for strategy in self._iter_active_by_symbol(event_symbol):
            self._publish_outputs(strategy.on_signal(self.context, signal))

    def on_order(self, order: Any, symbol: str | None = None) -> None:
        """
        Handle ORDER event.

        订单事件通常来自订单模块或交易接口回报。
        这里按 symbol 分发给相关策略，让策略更新内部状态。

        注意：
        当前用 _iter_by_symbol()，不是 _iter_active_by_symbol()。
        也就是说，即使策略 stop 了，只要还注册在引擎中，也可能收到订单回报。
        这样做有一个好处：停止策略后仍然可以同步订单状态。
        """
        # 不同订单对象可能字段名不同，所以依次尝试：
        # 1. 外部传入 symbol。
        # 2. order.vt_symbol。
        # 3. order.symbol。
        event_symbol = symbol or getattr(order, "vt_symbol", None) or getattr(order, "symbol", None)

        for strategy in self._iter_by_symbol(event_symbol):
            self._publish_outputs(strategy.on_order(self.context, order))

    def _publish_outputs(self, outputs: list[StrategyOutput] | None) -> None:
        """
        Publish strategy outputs.

        策略 on_bar/on_factor/on_signal/on_order 可以返回
        - StrategySignal:交易观点 / 信号
        - TargetPosition:目标仓位

        StrategyEngine 会把这些输出包装成 TRADE_SIGNAL 事件，转发给 trade_module

        注意：
        当前实现中 latest_outputs 会被赋值为 outputs。
        如果同一个事件触发多个策略输出，后面的输出可能覆盖前面的输出。
        如果你想保留所有策略输出，可以改成：
            self.latest_outputs.extend(outputs)
        """
        if not outputs:
            return

        # 当前写法：保存本次 publish 的输出。
        # 如果多个策略依次输出，这里会覆盖前一个策略的 outputs。
        self.latest_outputs = outputs

        # 没有配置 trade_module 时，只保存 latest_outputs，不向外投递。
        if not self.trade_module:
            return

        for output in outputs:
            # StrategySignal 表示策略产生交易信号。
            # 其他 StrategyOutput 默认当作 TargetPosition。
            # 更严格的写法可以显式判断 TargetPosition，避免未知类型被误发。
            key = "signal" if isinstance(output, StrategySignal) else "target_position"

            event = EngineEvent(
                event_type=EventType.TRADE_SIGNAL,
                source=self.source,
                target=self.trade_module,
                symbol=output.symbol,
                data={key: output},
            )
            self.post_event(self.trade_module, event)

    def send_order(
        self,
        strategy_name: str,
        symbol: str,
        direction: str,
        price: float,
        volume: int,
        reason: str = "",
    ) -> bool:
        """
        Create and post one order request.

        作用：
        1. 检查策略是否处于 active 状态。
        2. 构造 OrderRequest。
        3. 调用 RiskEngine 做风控检查。
        4. 如果通过风控，投递 ORDER 事件给 order_module。

        返回：
        - True:订单事件投递成功。
        - False:策略未启动 / 风控拒绝 / 事件投递失败。
        """
        # 未启动的策略不允许下单。
        if strategy_name not in self.active_strategies:
            return False

        order = OrderRequest(
            vt_symbol=symbol,
            direction=direction,
            price=price,
            volume=volume,
            reason=reason,
        )


        event = EngineEvent(
            event_type=EventType.ORDER,
            source=self.source,
            target=self.order_module,
            symbol=symbol,
            data={
                "order": order,
                "strategy_name": strategy_name,
            },
        )
        return self.post_event(self.order_module, event)

    def _iter_active_by_symbol(self, symbol: str | None) -> Iterable[StrategyTemplate]:
        """
        Iterate active strategies subscribed to symbol.

        只返回同时满足：
        1. 订阅了该 symbol,或订阅了 "*"。
        2. strategy_name 在 active_strategies 中。
        """
        for strategy in self._iter_by_symbol(symbol):
            if strategy.strategy_name in self.active_strategies:
                yield strategy

    def _iter_active_by_factor(self, symbol: str | None, factor_names: set[str],) -> Iterable[StrategyTemplate]:
        """
        Iterate active strategies matched by symbol and factor names.

        这是 FACTOR 事件分发的核心过滤函数

        过滤过程：
        1. symbol_names:先找订阅该 symbol 的策略。
        2. candidate_names:再找订阅本次 factor 的策略。
        3. active_strategies:只保留已启动策略。
        4. required.issubset(factor_names)：策略需要的因子必须全部在本次事件里。

        举例：
        策略 A 订阅：
            symbols = ["600519.SH"]
            factor_names = ["momentum_20", "volatility_20"]

        本次事件：
            symbol = "600519.SH"
            factor_names = {"momentum_20", "volatility_20", "volume_ratio"}

        那么策略 A 会被触发。
        """
        # 获取订阅该 symbol 的策略集合。
        symbol_names = self._get_strategy_names_by_symbol(symbol)

        # 先加入订阅所有因子的策略。
        candidate_names = set(self.factor_strategy_map.get("*", set()))

        # 再加入订阅本次事件中任意一个因子的策略。
        for factor_name in factor_names:
            candidate_names.update(self.factor_strategy_map.get(factor_name, set()))

        # 同时满足：
        # - 订阅 symbol。
        # - 订阅 factor。
        # - 策略已启动。
        for strategy_name in candidate_names & symbol_names & self.active_strategies:
            strategy = self.strategies.get(strategy_name)
            if strategy is None:
                continue

            # 如果策略声明了 factor_names，要求本次事件必须包含所有 required factors。
            # 这个设计默认 FactorEngine 发的是“完整因子快照”。
            required = set(strategy.factor_names)
            if required and not required.issubset(factor_names):
                continue

            yield strategy

    def _iter_by_symbol(self, symbol: str | None) -> Iterable[StrategyTemplate]:
        """
        Iterate all strategies subscribed to symbol.

        不检查策略是否 active。
        适合 ORDER 回报这类需要同步状态的事件。
        """
        for strategy_name in self._get_strategy_names_by_symbol(symbol):
            strategy = self.strategies.get(strategy_name)
            if strategy is not None:
                yield strategy

    def _get_strategy_names_by_symbol(self, symbol: str | None) -> set[str]:
        """
        Get strategy names subscribed to symbol.

        规则：
        - symbol 为空：返回全部策略。
        - symbol 非空：返回订阅该 symbol 的策略 + 订阅 "*" 的策略
        """
        if not symbol:
            return set(self.strategies)

        names = set(self.symbol_strategy_map.get(symbol, set()))
        names.update(self.symbol_strategy_map.get("*", set()))
        return names

    def _bind_symbols(self, strategy_name: str, symbols: Sequence[str]) -> None:
        """
        Bind strategy to symbols.

        symbols 为空时，表示该策略订阅全部股票，用 "*" 保存
        """
        if not symbols:
            self.symbol_strategy_map["*"].add(strategy_name)
            return

        for symbol in symbols:
            self.symbol_strategy_map[symbol].add(strategy_name)

    def _unbind_symbols(self, strategy_name: str) -> None:
        """
        Remove strategy from all symbol bindings.

        删除策略时调用，避免 symbol_strategy_map 中残留无效策略名
        """
        empty_symbols = []

        for symbol, strategy_names in self.symbol_strategy_map.items():
            strategy_names.discard(strategy_name)
            if not strategy_names:
                empty_symbols.append(symbol)

        # 遍历 dict 时不能直接删除 key，所以先收集空 key，再统一删除。
        for symbol in empty_symbols:
            del self.symbol_strategy_map[symbol]

    def _bind_factors(self, strategy_name: str, factor_names: Sequence[str]) -> None:
        """
        Bind strategy to factor names.

        factor_names 为空时，表示该策略订阅全部因子，用 "*" 保存。
        """
        if not factor_names:
            self.factor_strategy_map["*"].add(strategy_name)
            return

        for factor_name in factor_names:
            self.factor_strategy_map[factor_name].add(strategy_name)

    def _unbind_factors(self, strategy_name: str) -> None:
        """
        Remove strategy from all factor bindings.

        删除策略时调用，避免 factor_strategy_map 中残留无效策略名.
        """
        empty_factors = []

        for factor_name, strategy_names in self.factor_strategy_map.items():
            strategy_names.discard(strategy_name)
            if not strategy_names:
                empty_factors.append(factor_name)

        # 遍历 dict 时不能直接删除 key，所以先收集空 key，再统一删除。
        for factor_name in empty_factors:
            del self.factor_strategy_map[factor_name]

    @staticmethod
    def _get_factor_names(factor_result: Any) -> set[str]:
        """
        Extract factor names from factor_result.

        当前假设 factor_result.values 是一个列表，列表元素具有 factor_name 字段。

        例如：
            factor_result.values = [
                FactorValue(factor_name="momentum_20", value=0.03),
                FactorValue(factor_name="volatility_20", value=0.01),
            ]

        返回：
            {"momentum_20", "volatility_20"}

        如果后续 factor_result.values 改成 dict,可以改成:
            if isinstance(values, dict):
                return set(values.keys())
        """
        values = getattr(factor_result, "values", None)
        if not values:
            return set()

        return {
            value.factor_name
            for value in values
            if getattr(value, "factor_name", None)
        }

    def _resolve_strategy_class(self, strategy_class: type[StrategyTemplate] | str,) -> type[StrategyTemplate]:
        """
        Resolve strategy class.

        支持两种输入：
        1. 直接传类对象：
            TailCloseStrategy

        2. 传字符串路径：
            "vnpy.strategy.tail_close.TailCloseStrategy"

        字符串方式适合配置化加载策略。
        """
        if isinstance(strategy_class, str):
            module_name, class_name = strategy_class.rsplit(".", 1)
            module = import_module(module_name)
            strategy_class = getattr(module, class_name)

        if not issubclass(strategy_class, StrategyTemplate):
            raise TypeError("strategy_class must inherit StrategyTemplate")

        return strategy_class
