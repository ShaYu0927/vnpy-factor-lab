from threading import RLock
from typing import Dict, List, Optional, Callable

from .module_node import ModuleNode


class ModuleRegistry:
    """
    模块注册表。

    只负责管理 ModuleNode：
    1. 注册模块
    2. 删除模块
    3. 查找模块
    4. 判断模块是否存在
    5. 遍历模块

    不负责：
    - 启动模块
    - 停止模块
    - 投递事件
    """

    def __init__(self):
        self._modules: Dict[str, ModuleNode] = {}
        self._lock = RLock()

    def register(self, node: ModuleNode) -> bool:
        """
        注册模块节点。

        :param node: 模块节点
        :return: True 注册成功 False 注册失败
        """
        if node is None:
            print("[ModuleRegistry] register failed: node is None")
            return False

        with self._lock:
            if node.name in self._modules:
                print(f"[ModuleRegistry] register failed: module already exists: {node.name}")
                return False

            self._modules[node.name] = node
            print(f"[ModuleRegistry] register module: {node.name}")
            return True

    def unregister(self, name: str) -> Optional[ModuleNode]:
        """
        从注册表移除模块。

        注意：
        这里只从注册表删除，不负责 stop。
        stop 应该交给 lifecycle 或 engine 处理。

        :param name: 模块名称
        :return: 被删除的 ModuleNode 如果不存在则返回 None
        """
        with self._lock:
            node = self._modules.pop(name, None)

            if node is None:
                print(f"[ModuleRegistry] unregister failed: module not found: {name}")
                return None

            print(f"[ModuleRegistry] unregister module: {name}")
            return node

    def get(self, name: str) -> Optional[ModuleNode]:
        """
        根据模块名查找模块节点
        """
        with self._lock:
            return self._modules.get(name)

    def exists(self, name: str) -> bool:
        """
        判断模块是否已经注册
        """
        with self._lock:
            return name in self._modules

    def list_modules(self) -> List[str]:
        """
        返回所有模块名称
        """
        with self._lock:
            return list(self._modules.keys())

    def count(self) -> int:
        """
        返回当前注册模块数量
        """
        with self._lock:
            return len(self._modules)

    def for_each(self, callback: Callable[[ModuleNode], None]) -> None:
        """
        遍历所有模块
        """
        with self._lock:
            nodes = list(self._modules.values())

        for node in nodes:
            callback(node)

    def clear(self) -> None:
        """
        清空注册表。
        """
        with self._lock:
            self._modules.clear()
            print("[ModuleRegistry] clear all modules")