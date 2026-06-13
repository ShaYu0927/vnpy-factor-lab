from .registry import ModuleRegistry


class ModuleLifecycle:
    """
    模块生命周期管理器
    """

    def __init__(self, registry: ModuleRegistry):
        """
        :param registry: 模块注册表
        """
        self._registry = registry

    def start_module(self, name: str) -> bool:
        """
        启动指定模块

        :param name: 模块名称
        :return: True 启动成功 False 启动失败
        """
        node = self._registry.get(name)

        if node is None:
            print(f"[ModuleLifecycle] start failed: module not found: {name}")
            return False

        return node.start()

    def stop_module(self, name: str) -> bool:
        """
        停止指定模块

        :param name: 模块名称
        :return: True 停止成功 False 停止失败
        """
        node = self._registry.get(name)

        if node is None:
            print(f"[ModuleLifecycle] stop failed: module not found: {name}")
            return False

        return node.stop()

    def start_all(self) -> None:
        """
        启动所有模块
        """

        def _start(node):
            node.start()

        self._registry.for_each(_start)

        print("[ModuleLifecycle] start all modules")

    def stop_all(self) -> None:
        """
        停止所有模块
        """

        def _stop(node):
            node.stop()

        self._registry.for_each(_stop)

        print("[ModuleLifecycle] stop all modules")

    def is_module_started(self, name: str) -> bool:
        """
        判断指定模块是否已经启动
        """
        node = self._registry.get(name)

        if node is None:
            return False

        return node.is_started()