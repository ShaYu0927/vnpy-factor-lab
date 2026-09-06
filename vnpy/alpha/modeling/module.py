from __future__ import annotations

from vnpy.common.logger import get_logger
from vnpy.event.base_module import BaseModule, make_module_entry
from vnpy.event.event import EngineEvent, EventType

from .service import (
    DefaultModelTrainingService,
    ModelTrainingService,
    training_request_from_event,
)

logger = get_logger("alpha.model.module")


class ModelModule(BaseModule):
    """Event adapter around the synchronous alpha-modeling workflow."""

    def handle(self, event: EngineEvent) -> None:
        if event.event_type != EventType.MODEL_TRAIN_REQUEST:
            return

        response_target = event.get("response_target") or event.source
        try:
            request = training_request_from_event(event.data)
            result = self.training_service.train(request)
        except Exception as exc:
            self.set_state("latest_error", exc)
            logger.exception(
                "model training failed request_id=%s error=%s",
                event.event_id,
                exc,
            )
            if response_target:
                self.post(
                    target=response_target,
                    event_type=EventType.MODEL_FAILED,
                    symbol=event.symbol,
                    data={"error": exc, "request_id": event.event_id},
                )
            return

        self.set_state("latest_result", result)
        self.set_state("latest_error", None)
        if response_target:
            self.post(
                target=response_target,
                event_type=EventType.MODEL_TRAINED,
                symbol=event.symbol,
                data={"result": result, "request_id": event.event_id},
            )

        signal_target = event.get("signal_target")
        if signal_target and result.predictions:
            self.post(
                target=signal_target,
                event_type=EventType.ALPHA_SIGNAL,
                symbol=event.symbol,
                data={
                    "predictions": result.predictions,
                    "selected_features": result.selected_features,
                    "request_id": event.event_id,
                },
            )

    @property
    def training_service(self) -> ModelTrainingService:
        """Return the injected implementation through the abstract interface."""
        service = self.get_object("training_service")
        if service is None:
            service = DefaultModelTrainingService()
            self.set_object("training_service", service)
        if not isinstance(service, ModelTrainingService):
            raise TypeError("training_service must implement ModelTrainingService")
        return service


model_module_entry = make_module_entry(ModelModule)
