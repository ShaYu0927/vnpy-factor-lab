"""
Module engine.

这个文件只做门面封装，不直接处理队列、线程、事件分发细节。

内部组合：
- ModuleRegistry
- EventDispatcher
- ModuleLifecycle
"""

from typing import Optional, List, Any, Dict

from .context import ModuleContext
from .event import EngineEvent
from .module_node import ModuleNode, ModuleEntry
from .registry import ModuleRegistry
from .dispatcher import EventDispatcher
from .lifecycle import ModuleLifecycle


class ModuleEngine:
    """
    模块引擎。

    对外提供统一接口：

    1. 注册模块
    2. 释放模块
    3. 启动模块
    4. 停止模块
    5. 向指定模块投递事件
    6. 广播事件
    7. 获取模块上下文
    """

    def __init__(self):
        """
        初始化模块引擎。
        """
        self._registry = ModuleRegistry()
        self._dispatcher = EventDispatcher(self._registry)
        self._lifecycle = ModuleLifecycle(self._registry)

    def register_module(
        self,
        name: str,
        entry: ModuleEntry,
        queue_size: int = 1000000,
        config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        注册模块。

        :param name: 模块名称，例如 factor / alpha / notify
        :param entry: 模块入口函数，格式为 entry(ctx, event)
        :param queue_size: 模块事件队列大小
        :param config: 模块初始配置
        :return: True 注册成功 False 注册失败
        """
        if not name:
            print("[ModuleEngine] register failed: module name is empty")
            return False

        if entry is None:
            print(f"[ModuleEngine] register failed: entry is None, module={name}")
            return False

        if self._registry.exists(name):
            print(f"[ModuleEngine] register failed: module already exists: {name}")
            return False

        context = ModuleContext(
            module_name=name,
            engine=self,
        )

        if config:
            for key, value in config.items():
                context.set_config(key, value)

        node = ModuleNode(
            name=name,
            context=context,
            entry=entry,
            queue_size=queue_size,
        )

        return self._registry.register(node)

    def unregister_module(self, name: str) -> bool:
        """
        释放模块
        """
        node = self._registry.get(name)

        if node is None:
            print(f"[ModuleEngine] unregister failed: module not found: {name}")
            return False

        node.stop()
        node.context.clear()

        removed_node = self._registry.unregister(name)

        return removed_node is not None

    def start_module(self, name: str) -> bool:
        """
        启动指定模块
        """
        return self._lifecycle.start_module(name)

    def stop_module(self, name: str) -> bool:
        """
        停止指定模块
        """
        return self._lifecycle.stop_module(name)

    def start_all(self) -> None:
        """
        启动所有模块
        """
        self._lifecycle.start_all()

    def stop_all(self) -> None:
        """
        停止所有模块
        """
        self._lifecycle.stop_all()

    def post_event(self, target: str, event: EngineEvent) -> bool:
        """
        向指定模块投递事件
        """
        return self._dispatcher.post_event(target, event)

    def post_event_by_target(self, event: EngineEvent) -> bool:
        """
        根据 event.target 投递事件

        适合事件对象已经提前设置 target 的场景
        """
        return self._dispatcher.post_event_by_target(event)

    def broadcast_event(
        self,
        event: EngineEvent,
        exclude_source: bool = True,
    ) -> int:
        """
        广播事件给所有模块

        :param event: 事件对象
        :param exclude_source: 是否排除事件来源模块
        :return: 成功投递的模块数量
        """
        return self._dispatcher.broadcast_event(
            event=event,
            exclude_source=exclude_source,
        )

    def get_context(self, name: str) -> Optional[ModuleContext]:
        """
        获取指定模块的上下文
        """
        node = self._registry.get(name)

        if node is None:
            return None

        return node.context

    def get_module(self, name: str) -> Optional[ModuleNode]:
        """
        获取指定模块节点
        """
        return self._registry.get(name)

    def module_exists(self, name: str) -> bool:
        """
        判断模块是否存在。
        """
        return self._registry.exists(name)

    def list_modules(self) -> List[str]:
        """
        返回所有已注册模块名。
        """
        return self._registry.list_modules()

    def module_count(self) -> int:
        """
        返回模块数量。
        """
        return self._registry.count()

    def is_module_started(self, name: str) -> bool:
        """
        判断模块是否已经启动。
        """
        return self._lifecycle.is_module_started(name)

    def queue_size(self, name: str) -> int:
        """
        查看指定模块当前队列长度。
        """
        node = self._registry.get(name)

        if node is None:
            return -1

        return node.queue_size()