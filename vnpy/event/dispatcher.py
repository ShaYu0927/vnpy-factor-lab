from typing import Optional

from .event import EngineEvent
from .registry import ModuleRegistry


class EventDispatcher:
    """
    事件投递器
    """

    def __init__(self, registry: ModuleRegistry):
        """
        :param registry: 模块注册表
        """
        self._registry = registry

    def post_event(self, target: str, event: EngineEvent) -> bool:
        """
        向指定模块投递事件

        :param target: 目标模块名，例如 factor / alpha / notify
        :param event: 要投递的事件
        :return: True 投递成功 False 投递失败
        """
        if not target:
            print("[EventDispatcher] post_event failed: target is empty")
            return False

        node = self._registry.get(target)
        if node is None:
            print(f"[EventDispatcher] post_event failed: target module not found: {target}")
            return False

        event.target = target
        return node.post_event(event)

    def post_event_by_target(self, event: EngineEvent) -> bool:
        """
        根据 event.target 投递事件
        """
        if not event.target:
            print("[EventDispatcher] post_event_by_target failed: event.target is empty")
            return False

        return self.post_event(event.target, event)

    def broadcast_event(
        self,
        event: EngineEvent,
        exclude_source: bool = True,
    ) -> int:
        """
        广播事件给所有模块

        :param event: 要广播的事件
        :param exclude_source: 是否跳过事件来源模块
        :return: 成功投递的模块数量
        """
        success_count = 0

        module_names = self._registry.list_modules()

        for name in module_names:
            if exclude_source and name == event.source:
                continue

            copied_event = EngineEvent(
                event_type=event.event_type,
                source=event.source,
                target=name,
                symbol=event.symbol,
                data=dict(event.data),
            )

            if self.post_event(name, copied_event):
                success_count += 1

        return success_count

    def module_exists(self, name: str) -> bool:
        """
        判断模块是否存在。
        """
        return self._registry.exists(name)