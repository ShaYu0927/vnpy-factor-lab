from queue import Queue, Empty
from threading import Thread, Event
from typing import Callable, Optional

from .context import ModuleContext
from .event import EngineEvent, EventType


ModuleEntry = Callable[[ModuleContext, EngineEvent], None]


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

        self._started = False

    def start(self) -> bool:
        """
        启动模块线程
        """
        if self._thread and self._thread.is_alive():
            print(f"[ModuleNode:{self.name}] already started")
            return True

        self._stop_event.clear()

        self._thread = Thread(target=self._run, name=f"Module-{self.name}",daemon=True,)

        self._thread.start()
        self._started = True

        print(f"[ModuleNode:{self.name}] started")
        return True

    def stop(self) -> bool:
        """
        停止模块线程
        """
        if not self._started:
            return True

        self._stop_event.set()

        self.post_event(
            EngineEvent(
                event_type=EventType.STOP,
                source="engine",
                target=self.name,
                data={},
            )
        )

        if self._thread:
            self._thread.join(timeout=3)

        self._started = False

        print(f"[ModuleNode:{self.name}] stopped")
        return True

    def post_event(self, event: EngineEvent) -> bool:
        """
        向当前模块自己的队列投递事件
        """
        if self._queue.full():
            print(f"[ModuleNode:{self.name}] queue full, "f"drop event_id={event.event_id}, event_type={event.event_type}")
            return False

        self._queue.put(event)
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

    def _run(self) -> None:
        """
        模块线程主循环
        """
        while not self._stop_event.is_set():
            try:
                event = self._queue.get(timeout=1)
            except Empty:
                continue

            if event.event_type == EventType.STOP:
                self._queue.task_done()
                break

            try:
                self.entry(self.context, event)
            except Exception as e:
                print(
                    f"[ModuleNode:{self.name}] entry handle failed, "
                    f"event_id={event.event_id}, error={e}"
                )

            self._queue.task_done()