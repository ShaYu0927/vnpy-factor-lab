from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
import uuid


class EventType(str, Enum):
    """
    模块引擎内部事件类型。
    BAR:
        K线事件: 一般由行情模块产生。

    FACTOR:
        因子事件: 一般由因子模块产生。

    ALPHA_SIGNAL:
        Alpha信号事件: 一般由Alpha模块产生。

    TRADE_SIGNAL:
        交易信号事件，一般由策略/风控模块产生。

    ORDER:
        下单事件，一般发给交易模块。

    NOTIFY:
        通知事件，一般发给通知模块。

    TIMER:
        定时器事件。

    STOP:
        模块停止事件。

    ERROR:
        异常事件。
    """

    BAR = "bar"
    FACTOR = "factor"
    ALPHA_SIGNAL = "alpha_signal"
    TRADE_SIGNAL = "trade_signal"
    ORDER = "order"
    NOTIFY = "notify"
    TIMER = "timer"
    STOP = "stop"
    ERROR = "error"


@dataclass
class EngineEvent:
    """
    模块引擎事件。

    一个事件就是模块之间通信的最小单位
    """

    event_type: EventType
    data: Dict[str, Any]

    source: str = "system"
    target: Optional[str] = None
    symbol: Optional[str] = None

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def is_type(self, event_type: EventType) -> bool:
        """
        判断事件类型
        """
        return self.event_type == event_type

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """
        从事件数据里取字段
        """
        return self.data.get(key, default)

    def set_target(self, target: str) -> None:
        """
        设置事件目标模块
        """
        self.target = target

    def to_dict(self) -> Dict[str, Any]:
        """
        转成字典，方便打印、日志、序列化
        """
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "source": self.source,
            "target": self.target,
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "data": self.data,
        }