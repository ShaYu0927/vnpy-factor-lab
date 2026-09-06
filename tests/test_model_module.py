from datetime import date
from types import SimpleNamespace

from vnpy.alpha.modeling.module import model_module_entry
from vnpy.alpha.modeling.schema import FactorObservation
from vnpy.alpha.modeling.service import ModelTrainingRequest, ModelTrainingService
from vnpy.event.engine import ModuleEngine
from vnpy.event.event import EngineEvent, EventType


def test_model_module_emits_trained_and_alpha_signal(monkeypatch) -> None:
    result = SimpleNamespace(
        training_samples=100,
        test_samples=20,
        predictions=[("SHSE.600000", 0.01)],
        selected_features=("momentum_20",),
    )
    monkeypatch.setattr(
        "vnpy.alpha.modeling.module.DefaultModelTrainingService.train",
        lambda *_args, **_kwargs: result,
    )
    responses = []
    signals = []
    engine = ModuleEngine()
    engine.register_module("research", lambda _ctx, event: responses.append(event))
    engine.register_module("strategy", lambda _ctx, event: signals.append(event))
    engine.register_module("model", model_module_entry)
    engine.start_all()
    try:
        request = EngineEvent(
            event_type=EventType.MODEL_TRAIN_REQUEST,
            source="research",
            data={
                "observations": [
                    FactorObservation(
                        date(2026, 1, 2),
                        "SHSE.600000",
                        10.0,
                        {"momentum_20": 0.1},
                    )
                ],
                "feature_names": ["momentum_20"],
                "signal_target": "strategy",
            },
        )
        assert engine.post_event("model", request)
        engine.get_module("model")._queue.join()
        engine.get_module("research")._queue.join()
        engine.get_module("strategy")._queue.join()

        assert responses[0].event_type == EventType.MODEL_TRAINED
        assert responses[0].get("request_id") == request.event_id
        assert responses[0].get("result") is result
        assert signals[0].event_type == EventType.ALPHA_SIGNAL
        assert signals[0].get("predictions") == result.predictions
    finally:
        engine.stop_all()


def test_model_module_returns_failure_event_for_invalid_payload() -> None:
    responses = []
    engine = ModuleEngine()
    engine.register_module("research", lambda _ctx, event: responses.append(event))
    engine.register_module("model", model_module_entry)
    engine.start_all()
    try:
        assert engine.post_event(
            "model",
            EngineEvent(
                event_type=EventType.MODEL_TRAIN_REQUEST,
                source="research",
                data={"observations": "invalid", "feature_names": []},
            ),
        )
        engine.get_module("model")._queue.join()
        engine.get_module("research")._queue.join()

        assert responses[0].event_type == EventType.MODEL_FAILED
        assert isinstance(responses[0].get("error"), TypeError)
    finally:
        engine.stop_all()


class StubTrainingService(ModelTrainingService):
    def __init__(self, result) -> None:
        self.result = result
        self.requests: list[ModelTrainingRequest] = []

    def train(self, request: ModelTrainingRequest):
        self.requests.append(request)
        return self.result


def test_model_module_dispatches_through_polymorphic_training_service() -> None:
    result = SimpleNamespace(
        predictions=[],
        selected_features=("momentum_20",),
    )
    service = StubTrainingService(result)
    responses = []
    engine = ModuleEngine()
    engine.register_module("research", lambda _ctx, event: responses.append(event))
    engine.register_module("model", model_module_entry)
    model_context = engine.get_context("model")
    assert model_context is not None
    model_context.set_object("training_service", service)
    engine.start_all()
    try:
        request = EngineEvent(
            event_type=EventType.MODEL_TRAIN_REQUEST,
            source="research",
            data={
                "observations": [
                    FactorObservation(
                        date(2026, 1, 2),
                        "SHSE.600000",
                        10.0,
                        {"momentum_20": 0.1},
                    )
                ],
                "feature_names": ["momentum_20"],
                "horizon": 3,
            },
        )
        assert engine.post_event("model", request)
        engine.get_module("model")._queue.join()
        engine.get_module("research")._queue.join()

        assert responses[0].event_type == EventType.MODEL_TRAINED
        assert responses[0].get("result") is result
        assert service.requests[0].horizon == 3
        assert service.requests[0].feature_names == ("momentum_20",)
    finally:
        engine.stop_all()
