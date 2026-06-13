from abc import ABC, abstractmethod
from typing import Any, Callable, TypeVar

from .context import ModuleContext
from .event import EngineEvent, EventType


ModuleT = TypeVar("ModuleT", bound="BaseModule")
ModuleEntry = Callable[[ModuleContext, EngineEvent], None]


class BaseModule(ABC):
    """
    Base class for business modules managed by ModuleEngine.

    ModuleEngine still works with entry functions. This class gives each module
    a stable object shape for config, state, object storage and event posting.
    """

    def __init__(self, ctx: ModuleContext) -> None:
        self.ctx = ctx

    @abstractmethod
    def handle(self, event: EngineEvent) -> None:
        """
        Handle one event from this module's queue.
        """

    @property
    def name(self) -> str:
        return self.ctx.module_name

    def get_config(self, key: str, default: Any = None) -> Any:
        return self.ctx.get_config(key, default)

    def set_state(self, key: str, value: Any) -> None:
        self.ctx.set_state(key, value)

    def get_state(self, key: str, default: Any = None) -> Any:
        return self.ctx.get_state(key, default)

    def set_object(self, key: str, value: Any) -> None:
        self.ctx.set_object(key, value)

    def get_object(self, key: str, default: Any = None) -> Any:
        return self.ctx.get_object(key, default)

    def post(
        self,
        target: str,
        event_type: EventType,
        data: dict[str, Any],
        symbol: str | None = None,
    ) -> bool:
        return self.ctx.engine.post_event(
            target=target,
            event=EngineEvent(
                event_type=event_type,
                source=self.name,
                symbol=symbol,
                data=data,
            ),
        )


def get_module_instance(ctx: ModuleContext, module_class: type[ModuleT]) -> ModuleT:
    key = f"_module_instance:{module_class.__name__}"
    instance = ctx.get_object(key)
    if instance is None:
        instance = module_class(ctx)
        ctx.set_object(key, instance)
    return instance


def make_module_entry(module_class: type[ModuleT]) -> ModuleEntry:
    def entry(ctx: ModuleContext, event: EngineEvent) -> None:
        module = get_module_instance(ctx, module_class)
        module.handle(event)

    return entry
