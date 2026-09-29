"""由全局 ModuleEngine 管理的行情模块，以及加载/刷新/清空请求接口。"""

from concurrent.futures import Future
from threading import RLock

from vnpy.common.logger import get_logger
from vnpy.config.runtime_config import ParquetConfig
from vnpy.event.base_module import BaseModule, make_module_entry
from vnpy.event.engine import ModuleEngine
from vnpy.event.event import EngineEvent, EventType
from .data_market_store import MarketDataRequest, MarketDataStore, MarketLoadResult


MARKET_MODULE = "market"
_REGISTRATION_LOCK = RLock()


class MarketDataModule(BaseModule):
    def handle(self, event: EngineEvent) -> None:
        if event.event_type not in (EventType.MARKET_LOAD_REQUEST, EventType.MARKET_CLEAR_REQUEST):
            return
        reply: Future = event.get("reply")
        try:
            store: MarketDataStore = self.get_object("market_store")
            if event.event_type == EventType.MARKET_CLEAR_REQUEST:
                store.clear()
                self.set_state("snapshot_id", None)
                self.set_state("rows", 0)
                self.set_state("status", "empty")
                self.set_state("error", None)
                reply.set_result(None)
                return
            self.set_state("status", "loading")
            result = store.load(event.get("request"), reload=event.get("reload", False))
            self.set_state("snapshot_id", result.snapshot.snapshot_id)
            self.set_state("rows", result.snapshot.frame.height)
            self.set_state("status", "ready")
            self.set_state("error", None)
            self.set_state("load_count", self.get_state("load_count", 0) + int(not result.reused))
            self.set_state("reuse_count", self.get_state("reuse_count", 0) + int(result.reused))
            reply.set_result(result)
        except Exception as exc:
            # A failed request must reach its caller without killing later retries.
            self.set_state("status", "failed")
            self.set_state("error", f"{type(exc).__name__}: {exc}")
            get_logger("market").error("[market/request] %s", exc)
            reply.set_exception(exc)


market_module_entry = make_module_entry(MarketDataModule)


def register_market_module(engine: ModuleEngine) -> MarketDataStore:
    """注册一次；同一个引擎中的后续研究轮次复用模块及行情。"""
    with _REGISTRATION_LOCK:
        if not engine.module_exists(MARKET_MODULE):
            if not engine.register_module(MARKET_MODULE, market_module_entry, queue_size=32):
                raise RuntimeError("could not register market module")
            engine.get_context(MARKET_MODULE).set_object("market_store", MarketDataStore())
        store = get_market_store(engine)
        if not engine.start_module(MARKET_MODULE):
            raise RuntimeError("could not start market module")
        return store


def get_market_store(engine: ModuleEngine) -> MarketDataStore:
    """其他模块通过引擎定位行情服务，无需从某一轮运行函数取行情。"""
    context = engine.get_context(MARKET_MODULE)
    store = context.get_object("market_store") if context is not None else None
    if not isinstance(store, MarketDataStore):
        raise RuntimeError("market module is not registered with a MarketDataStore")
    return store


def _request(engine: ModuleEngine, event_type: EventType, **data):
    with _REGISTRATION_LOCK:
        register_market_module(engine)
        reply = Future()
        if not engine.post_event(MARKET_MODULE, EngineEvent(
            event_type=event_type, source="history_batch", target=MARKET_MODULE,
            data={**data, "reply": reply},
        )):
            raise RuntimeError("market module rejected request")
    return reply.result()


def load_market_snapshot(
    engine: ModuleEngine, config: ParquetConfig, *, reload: bool = False,
) -> MarketLoadResult:
    """向行情模块投递加载请求；同一范围复用，reload=True 强制读盘刷新。"""
    return _request(engine, EventType.MARKET_LOAD_REQUEST,
                    request=MarketDataRequest.from_config(config), reload=reload)


def clear_market_data(engine: ModuleEngine) -> None:
    """显式清空行情，保留模块供下一次加载。"""
    _request(engine, EventType.MARKET_CLEAR_REQUEST)


def unregister_market_module(engine: ModuleEngine) -> None:
    """应用退出时排空请求、清空行情并注销模块；不在计算轮次结束时调用。"""
    with _REGISTRATION_LOCK:
        node = engine.get_module(MARKET_MODULE)
        if node is None:
            return
        node._queue.join()
        get_market_store(engine).clear()
        engine.unregister_module(MARKET_MODULE)
