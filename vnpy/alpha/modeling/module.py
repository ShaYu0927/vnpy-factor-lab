from __future__ import annotations

from typing import Any

from vnpy.common.logger import get_logger
from vnpy.event.base_module import BaseModule, make_module_entry
from vnpy.event.event import EngineEvent, EventType

from .schema import FactorObservation

logger = get_logger("alpha.model.module")


class ModelModule(BaseModule):
    """Event adapter around the synchronous alpha-modeling workflow."""

    def handle(self, event: EngineEvent) -> None:
        if event.event_type != EventType.MODEL_TRAIN_REQUEST:
            return

        response_target = event.get("response_target") or event.source
        observations = event.get("observations")
        feature_names = event.get("feature_names")
        try:
            result = self._train(event)
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

    @staticmethod
    def _train(event: EngineEvent) -> Any:
        # Import lazily so merely registering the module does not load optional
        # modeling dependencies.
        from vnpy.factor.model_pipeline import train_and_predict_latest

        observations = event.get("observations")
        feature_names = event.get("feature_names")
        if not isinstance(observations, (list, tuple)):
            raise TypeError("model training observations must be a sequence")
        if not all(isinstance(item, FactorObservation) for item in observations):
            raise TypeError("model training observations contain invalid items")
        if not isinstance(feature_names, (list, tuple)) or not feature_names:
            raise ValueError("model training feature_names must not be empty")

        return train_and_predict_latest(
            observations=observations,
            feature_names=feature_names,
            horizon=int(event.get("horizon", 5)),
            model_output=event.get("model_output"),
            signal_output=event.get("signal_output"),
            evaluate_factors=bool(event.get("evaluate_factors", False)),
            factor_quantiles=int(event.get("factor_quantiles", 2)),
            min_abs_ic=float(event.get("min_abs_ic", 0.02)),
            min_abs_ic_ir=float(event.get("min_abs_ic_ir", 0.20)),
        )


model_module_entry = make_module_entry(ModelModule)
