from typing import Any, Dict, Optional


class ModuleContext:
    """
    模块上下文

    每个模块都会持有一个独立的 ModuleContext
    用来保存模块自己的运行状态、配置和私有对象
    """

    def __init__(self, module_name: str, engine: Any = None):
        """
        :param module_name: 模块名称，例如 factor / alpha / notify
        :param engine: 模块引擎引用，用于模块内部继续投递事件
        """
        self.module_name = module_name
        self.engine = engine

        # 模块配置，例如窗口大小、阈值、webhook地址等
        self.config: Dict[str, Any] = {}

        # 模块运行状态，例如是否初始化、是否启动、最新信号等
        self.state: Dict[str, Any] = {}

        # 模块私有对象，例如 BarCache、FactorCalculator、模型对象等
        self.objects: Dict[str, Any] = {}

    def set_config(self, key: str, value: Any) -> None:
        """
        设置模块配置
        """
        self.config[key] = value

    def get_config(self, key: str, default: Optional[Any] = None) -> Any:
        """
        获取模块配置
        """
        return self.config.get(key, default)

    def set_state(self, key: str, value: Any) -> None:
        """
        设置模块状态
        """
        self.state[key] = value

    def get_state(self, key: str, default: Optional[Any] = None) -> Any:
        """
        获取模块状态
        """
        return self.state.get(key, default)

    def set_object(self, key: str, value: Any) -> None:
        """
        保存模块私有对象
        """
        self.objects[key] = value

    def get_object(self, key: str, default: Optional[Any] = None) -> Any:
        """
        获取模块私有对象
        """
        return self.objects.get(key, default)

    def remove_object(self, key: str) -> None:
        """
        删除模块私有对象
        """
        if key in self.objects:
            del self.objects[key]

    def clear(self) -> None:
        """
        清空模块上下文
        """
        self.config.clear()
        self.state.clear()
        self.objects.clear()