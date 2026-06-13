from collections import defaultdict
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Thread
from time import sleep
from typing import Any, Callable

from .base_module import BaseModule, make_module_entry
from .engine import ModuleEngine
from .event import EngineEvent, EventType


EVENT_TIMER = "eTimer"


@dataclass
class Event:
    type: str
    data: Any = None


class EventEngine:
    """
    Lightweight compatibility event engine used by vnpy.trader.
    """

    def __init__(self, interval: float = 1) -> None:
        self._interval = interval
        self._queue: Queue[Event] = Queue()
        self._active = False
        self._thread = Thread(target=self._run, daemon=True)
        self._timer_thread = Thread(target=self._run_timer, daemon=True)
        self._handlers: dict[str, list[Callable[[Event], None]]] = defaultdict(list)
        self._general_handlers: list[Callable[[Event], None]] = []

    def start(self) -> None:
        if self._active:
            return

        self._active = True
        if not self._thread.is_alive():
            self._thread = Thread(target=self._run, daemon=True)
            self._thread.start()
        if not self._timer_thread.is_alive():
            self._timer_thread = Thread(target=self._run_timer, daemon=True)
            self._timer_thread.start()

    def stop(self) -> None:
        self._active = False
        self._thread.join(timeout=3)
        self._timer_thread.join(timeout=3)

    def put(self, event: Event) -> None:
        self._queue.put(event)

    def register(self, type: str, handler: Callable[[Event], None]) -> None:
        handlers = self._handlers[type]
        if handler not in handlers:
            handlers.append(handler)

    def unregister(self, type: str, handler: Callable[[Event], None]) -> None:
        handlers = self._handlers[type]
        if handler in handlers:
            handlers.remove(handler)

        if not handlers:
            self._handlers.pop(type, None)

    def register_general(self, handler: Callable[[Event], None]) -> None:
        if handler not in self._general_handlers:
            self._general_handlers.append(handler)

    def unregister_general(self, handler: Callable[[Event], None]) -> None:
        if handler in self._general_handlers:
            self._general_handlers.remove(handler)

    def _run(self) -> None:
        while self._active:
            try:
                event = self._queue.get(timeout=1)
                self._process(event)
            except Empty:
                continue

    def _process(self, event: Event) -> None:
        for handler in self._handlers.get(event.type, []):
            handler(event)

        for handler in self._general_handlers:
            handler(event)

    def _run_timer(self) -> None:
        while self._active:
            sleep(self._interval)
            self.put(Event(EVENT_TIMER))


__all__ = [
    "EVENT_TIMER",
    "BaseModule",
    "EngineEvent",
    "Event",
    "EventEngine",
    "EventType",
    "ModuleEngine",
    "make_module_entry",
]
