import atexit
from concurrent.futures import Future, InvalidStateError
from queue import Queue, Empty, Full
from threading import Thread, Event, RLock, current_thread
from typing import Callable, Optional

from .context import ModuleContext
from .event import EngineEvent, EventType


ModuleEntry = Callable[[ModuleContext, EngineEvent], None]

_INTERPRETER_SHUTTING_DOWN = False


def _mark_interpreter_shutting_down() -> None:
    global _INTERPRETER_SHUTTING_DOWN
    _INTERPRETER_SHUTTING_DOWN = True


atexit.register(_mark_interpreter_shutting_down)


class ModuleStoppedError(RuntimeError):
    """A queued request could not run because its module stopped."""


class ModuleNode:
    """
    模块节点。

    一个 ModuleNode 表示系统里的一个独立模块，例如：
    - market 行情模块
    - factor 因子模块
    - alpha Alpha信号模块
    - notify 通知模块
    - trade 交易模块

    每个模块节点都有：
    1. 模块名称
    2. 模块上下文
    3. 模块事件队列
    4. 模块线程
    5. 模块入口函数 entry
    """

    def __init__(self, name: str, context: ModuleContext, entry: ModuleEntry, queue_size: int = 10000,):
        """
        :param name: 模块名称
        :param context: 模块上下文
        :param entry: 模块入口函数
        :param queue_size: 模块事件队列大小
        """
        self.name = name
        self.context = context
        self.entry = entry

        self._queue: Queue[EngineEvent] = Queue(maxsize=queue_size)

        self._stop_event = Event()
        self._thread: Optional[Thread] = None
        self._lifecycle_lock = RLock()
        self._stopping = False
        self._closed = False

        self._started = False
        self.last_error: Exception | None = None
        self.processed_events = 0
        self.peak_queue_size = 0

    def start(self) -> bool:
        """
        启动模块线程
        """
        with self._lifecycle_lock:
            if self._closed or self._stopping:
                return False
            if self._thread and self._thread.is_alive():
                return not self._stop_event.is_set()
            self._stop_event.clear()
            self._thread = Thread(target=self._run, name=f"Module-{self.name}", daemon=True)
            self._started = True
            self._thread.start()
            return True

    def stop(self, *, timeout: float = 3.0, close: bool = False) -> bool:
        """拒绝新事件，使排队请求失败；等待正在执行的事件，超时返回 False。

        close=True 用于注销，永久禁止本节点重新启动。
        """
        with self._lifecycle_lock:
            self._closed = self._closed or close
            self._stopping = True
            self._stop_event.set()
            thread = self._thread
            pending = self._take_pending()
            if thread and thread.is_alive():
                # 内部唤醒不经过 post_event：停止后该接口已拒绝新事件。
                self._queue.put_nowait(EngineEvent(EventType.STOP, {}, source="engine", target=self.name))
        self._reject_pending(pending)
        if thread is current_thread():
            return False
        if thread:
            thread.join(timeout=timeout)
        with self._lifecycle_lock:
            stopped = thread is None or not thread.is_alive()
            leftovers = []
            if stopped and self._thread is thread:
                # 线程可能已经进入退出清理，另一个 stop 才投递唤醒事件。
                leftovers = self._take_pending()
                self._started = False
                self._stopping = False
        self._reject_pending(leftovers)
        return stopped

    def post_event(self, event: EngineEvent) -> bool:
        """
        向当前模块自己的队列投递事件
        """
        # 与 stop 的关闭入口、清队列动作互斥，避免检查通过后才入队。
        with self._lifecycle_lock:
            if _INTERPRETER_SHUTTING_DOWN or self._closed or self._stop_event.is_set():
                return False
            try:
                self._queue.put_nowait(event)
            except Full:
                print(f"[ModuleNode:{self.name}] queue full, "f"drop event_id={event.event_id}, event_type={event.event_type}")
                return False
            self.peak_queue_size = max(self.peak_queue_size, self._queue.qsize())
            return True

    def queue_size(self) -> int:
        """
        获取当前模块队列里的事件数量
        """
        return self._queue.qsize()

    def is_started(self) -> bool:
        """
        判断模块是否已启动
        """
        return self._started

    def is_worker_thread(self) -> bool:
        return current_thread() is self._thread

    def _take_pending(self) -> list[EngineEvent]:
        pending = []
        while True:
            try:
                pending.append(self._queue.get_nowait())
            except Empty:
                return pending

    @staticmethod
    def _fail_reply(event: EngineEvent, error: Exception) -> None:
        reply = event.get("reply")
        if isinstance(reply, Future):
            try:
                reply.set_exception(error)
            except InvalidStateError:
                # 调用方可能已取消请求，或业务处理器已经返回结果。
                pass

    def _reject_pending(self, events: list[EngineEvent]) -> None:
        for event in events:
            try:
                self._fail_reply(event, ModuleStoppedError(f"module {self.name!r} stopped before request execution"))
            finally:
                self._queue.task_done()

    def _run(self) -> None:
        """
        模块线程主循环
        """
        try:
            while not self._stop_event.is_set():
                try:
                    event = self._queue.get(timeout=1)
                except Empty:
                    continue
                try:
                    with self._lifecycle_lock:
                        stopping = self._stop_event.is_set() or _INTERPRETER_SHUTTING_DOWN or event.event_type == EventType.STOP
                        if stopping:
                            self._stop_event.set()
                    if stopping:
                        self._fail_reply(event, ModuleStoppedError(f"module {self.name!r} stopped before request execution"))
                        break
                    self.entry(self.context, event)
                    self.processed_events += 1
                except Exception as e:
                    self.last_error = e
                    self._fail_reply(event, e)
                    print(f"[ModuleNode:{self.name}] entry handle failed, event_id={event.event_id}, error={e}")
                finally:
                    self._queue.task_done()
        finally:
            with self._lifecycle_lock:
                self._stop_event.set()
                pending = self._take_pending()
                self._started = False
            self._reject_pending(pending)
