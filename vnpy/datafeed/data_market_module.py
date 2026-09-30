"""由全局 ModuleEngine 管理的行情模块，以及加载/刷新/清空请求接口。"""

import csv
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from contextlib import nullcontext
from math import isfinite
from pathlib import Path
import sys
from threading import RLock

import polars as pl

from vnpy.common.logger import get_logger
from vnpy.config.runtime_config import ParquetConfig
from vnpy.event.base_module import BaseModule, make_module_entry
from vnpy.event.engine import ModuleEngine
from vnpy.event.event import EngineEvent, EventType
from .data_market_store import MarketDataRequest, MarketDataStore, MarketLoadResult


MARKET_MODULE = "market"
DEFAULT_REQUEST_TIMEOUT = 300.0
MARKET_PREVIEW_ROWS = 5
_REGISTRATION_LOCK = RLock()


class MarketDataModule(BaseModule):
    def handle(self, event: EngineEvent) -> None:
        if event.event_type not in (EventType.MARKET_LOAD_REQUEST, EventType.MARKET_CLEAR_REQUEST):
            return
        reply: Future = event.get("reply")
        # 超时且尚未开始的请求会被取消，不能再执行其加载/清空操作。
        if not reply.set_running_or_notify_cancel():
            return
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
            if not result.reused:
                frame = result.snapshot.frame
                with pl.Config(tbl_cols=-1, tbl_width_chars=180, fmt_str_lengths=28):
                    preview = str(frame.head(MARKET_PREVIEW_ROWS))
                get_logger("market").info(
                    "[market/table] snapshot=%s rows=%d columns=%d; first %d rows (sorted by symbol/time):\n%s",
                    result.snapshot.snapshot_id, frame.height, frame.width,
                    min(frame.height, MARKET_PREVIEW_ROWS), preview,
                )
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


def _request(engine: ModuleEngine, event_type: EventType, *, timeout: float = DEFAULT_REQUEST_TIMEOUT, **data):
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not isfinite(timeout) or timeout <= 0:
        raise ValueError("market request timeout must be a positive finite number")
    node = engine.get_module(MARKET_MODULE)
    if node is not None and node.is_worker_thread():
        raise RuntimeError("market worker cannot synchronously wait for its own queue")
    with _REGISTRATION_LOCK:
        register_market_module(engine)
        reply = Future()
        if not engine.post_event(MARKET_MODULE, EngineEvent(
            event_type=event_type, source="history_batch", target=MARKET_MODULE,
            data={**data, "reply": reply},
        )):
            raise RuntimeError("market module rejected request")
    try:
        return reply.result(timeout=timeout)
    except FutureTimeoutError:
        # 业务函数自身也可能抛 TimeoutError；保留已完成请求的原始异常。
        if reply.done():
            return reply.result()
        cancelled = reply.cancel()
        detail = "cancelled before execution" if cancelled else "already running; it may still complete"
        raise TimeoutError(f"market request timed out after {timeout}s ({detail})") from None


def load_market_snapshot(engine: ModuleEngine, config: ParquetConfig, *, reload: bool = False, timeout: float = DEFAULT_REQUEST_TIMEOUT,) -> MarketLoadResult:
    """向行情模块投递加载请求 timeout 限制等待回复时间，不中断已开始的 I/O。"""
    return _request(engine, EventType.MARKET_LOAD_REQUEST,request=MarketDataRequest.from_config(config), reload=reload, timeout=timeout)


def clear_market_data(engine: ModuleEngine, *, timeout: float = DEFAULT_REQUEST_TIMEOUT) -> None:
    """显式清空行情，保留模块供下一次加载"""
    _request(engine, EventType.MARKET_CLEAR_REQUEST, timeout=timeout)


def print_market_table(engine: ModuleEngine, *, output_path: str | Path | None = None) -> int:
    """逐行打印当前全局行情快照的全部列和全部行；可写入 TSV 文件。

    保留本次快照引用，刷新或清空不会让打印过程切换到另一版行情。
    不依赖 Polars 的表格显示设置，因此不会省略中间行或列。
    """
    frame = get_market_store(engine).snapshot().frame
    target = Path(output_path).expanduser() if output_path is not None else None
    if target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
    destination = target.open("w", encoding="utf-8", newline="") if target is not None else nullcontext(sys.stdout)
    with destination as stream:
        writer = csv.writer(stream, dialect="excel-tab", lineterminator="\n")
        writer.writerow(frame.columns)
        writer.writerows(frame.iter_rows())
    return frame.height


def unregister_market_module(engine: ModuleEngine) -> None:
    """使排队请求失败，停止后清空并注销；停止超时则保留上下文供重试。"""
    with _REGISTRATION_LOCK:
        node = engine.get_module(MARKET_MODULE)
        if node is None:
            return
        store = get_market_store(engine)
        if not engine.unregister_module(MARKET_MODULE):
            raise RuntimeError("market module is still stopping; retry unregister after the active request finishes")
        store.clear()
