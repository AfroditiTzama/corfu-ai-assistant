from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import pytest
from fastapi.testclient import TestClient
from numpy.typing import NDArray

from corfu_assistant.config import Settings
from corfu_assistant.data import load_dataset
from corfu_assistant.intent_extractor import (
    IntentConfigurationError,
    IntentExtractionError,
    IntentProviderError,
    IntentRateLimitError,
    IntentTimeoutError,
    IntentTransientProviderError,
    InvalidStructuredOutputError,
)
from corfu_assistant.main import create_app
from corfu_assistant.models import (
    ConversationState,
    Itinerary,
    ItineraryStop,
    UserIntent,
    WeatherForecast,
)
from corfu_assistant.orchestrator import AnswerGenerationError, OrchestrationRequest
from corfu_assistant.repository import POIRepository
from corfu_assistant.services import ServiceOverrides
from corfu_assistant.state_manager import ConversationStateManager
from corfu_assistant.weather import (
    FixtureWeatherProvider,
    WeatherProvider,
    WeatherProviderError,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTENT_ACTIONS = (
    "requested_date",
    "start_time",
    "end_time",
    "timezone",
    "semantic_interests",
    "filter_category",
    "filter_tags",
    "filter_setting",
    "filter_family_suitability",
    "filter_excluded_categories",
    "filter_excluded_tags",
    "filter_excluded_poi_ids",
    "travel_mode",
    "family_required",
    "child_ages",
    "itinerary_modification",
    "question",
    "clarification_question",
    "weather_request",
)


class FixedEmbeddings:
    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))


class FakeExtractor:
    def __init__(self, outputs: Sequence[UserIntent | Exception]) -> None:
        self.outputs = list(outputs)
        self.calls: list[tuple[str, ConversationState | None]] = []

    def extract(
        self,
        message: str,
        conversation_state: ConversationState | None = None,
    ) -> UserIntent:
        self.calls.append(
            (
                message,
                conversation_state.model_copy(deep=True)
                if conversation_state is not None
                else None,
            )
        )
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output


class FakeWeatherProvider:
    def __init__(
        self,
        forecast: WeatherForecast | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.forecast = forecast
        self.error = error

    def next_24_hours(self) -> WeatherForecast:
        start = datetime.now(ZoneInfo("Europe/Athens"))
        end = (start.astimezone(timezone.utc) + timedelta(hours=24)).astimezone(
            ZoneInfo("Europe/Athens")
        )
        return self.get_forecast(start, end)

    def get_forecast(self, start_at: datetime, end_at: datetime) -> WeatherForecast:
        if self.error is not None:
            raise self.error
        if self.forecast is None:
            raise AssertionError("Forecast was not configured")
        return self.forecast.model_copy(deep=True)

    def get_current(self) -> WeatherForecast:
        if self.error is not None:
            raise self.error
        if self.forecast is None:
            raise AssertionError("Forecast was not configured")
        return self.forecast.model_copy(deep=True)


def make_intent(
    kind: str = "poi_recommendation",
    *,
    values: dict[str, object] | None = None,
    actions: dict[str, str] | None = None,
) -> UserIntent:
    payload = dict(values or {})
    field_actions = dict.fromkeys(INTENT_ACTIONS, "preserve")
    field_actions.update(actions or {})
    payload.update(intent=kind, field_actions=field_actions)
    return UserIntent.model_validate(payload)


def planning_intent() -> UserIntent:
    return make_intent(
        "itinerary_creation",
        values={
            "requested_date": "2026-10-06",
            "time_window": {"start_time": "11:00", "end_time": "15:00"},
            "travel_mode": "walking",
            "hard_filters": {"category": "history"},
            "semantic_preferences": {"interests": ["historical walking"]},
        },
        actions={
            "requested_date": "set",
            "start_time": "set",
            "end_time": "set",
            "travel_mode": "set",
            "filter_category": "set",
            "semantic_interests": "set",
        },
    )


def make_repository() -> POIRepository:
    return POIRepository(
        load_dataset(PROJECT_ROOT / "corfu_dataset.json"),
        embedding_model=FixedEmbeddings(),
    )


def make_app(
    outputs: Sequence[UserIntent | Exception] = (),
    *,
    settings: Settings | None = None,
    state_manager: ConversationStateManager | None = None,
    repository: POIRepository | None = None,
    weather_provider: WeatherProvider | None = None,
):
    extractor = FakeExtractor(outputs)
    overrides = ServiceOverrides(
        poi_repository=repository or make_repository(),
        state_manager=state_manager,
        intent_extractor=extractor,
        weather_provider=weather_provider,
    )
    app = create_app(
        settings
        or Settings(
            dataset_path=PROJECT_ROOT / "corfu_dataset.json",
            google_api_key=None,
        ),
        service_overrides=overrides,
    )
    return app, extractor


def test_health_and_readiness_still_work_with_app_scoped_services(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from corfu_assistant import main

    original_loader = main.load_dataset
    load_count = 0

    def counting_loader(path: Path):
        nonlocal load_count
        load_count += 1
        return original_loader(path)

    monkeypatch.setattr(main, "load_dataset", counting_loader)
    app, _ = make_app()

    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").json() == {"status": "ready", "poi_count": 50}
        assert client.get("/ready").status_code == 200
        assert app.state.services is not None
        assert app.state.orchestrator is app.state.services.orchestrator
        assert app.state.state_manager is app.state.services.state_manager
        assert load_count == 1


def test_chat_creates_conversation_and_serializes_grounded_tourism_evidence() -> None:
    app, _ = make_app(
        [
            make_intent(
                "tourism_qa",
                values={"question": "Tell me about Old Fortress"},
                actions={"question": "set"},
            )
        ]
    )

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"message": "Tell me about Old Fortress"},
        )

    assert response.status_code == 200
    result = response.json()
    assert result["conversation_id"]
    assert result["intent"]["intent"] == "tourism_qa"
    assert result["status"] == "completed"
    assert result["relevant_poi_ids"] == ["CF-001"]
    assert result["evidence"][0]["poi_id"] == "CF-001"
    assert result["conversation_state"]["conversation_id"] == result["conversation_id"]


def test_consecutive_chat_requests_share_conversation_state() -> None:
    app, extractor = make_app(
        [
            make_intent(
                values={"semantic_preferences": {"interests": ["local history"]}},
                actions={"semantic_interests": "set"},
            ),
            make_intent(
                "tourism_qa",
                values={"question": "What is Old Fortress?"},
                actions={"question": "set"},
            ),
        ]
    )

    with TestClient(app) as client:
        first = client.post("/chat", json={"message": "I like local history"}).json()
        second = client.post(
            "/chat",
            json={
                "message": "Tell me about Old Fortress",
                "conversation_id": first["conversation_id"],
            },
        ).json()

    assert second["conversation_id"] == first["conversation_id"]
    assert second["conversation_state"]["preferences"]["interests"] == ["local history"]
    assert extractor.calls[1][1].preferences.interests == ["local history"]


def test_family_friendly_recommendations_filter_to_confirmed_fixture_values() -> None:
    app, _ = make_app(
        [
            make_intent(
                values={
                    "family_requirements": {"required": True, "child_ages": [10]}
                },
                actions={"family_required": "set", "child_ages": "set"},
            )
        ]
    )

    with TestClient(app) as client:
        result = client.post(
            "/chat",
            json={
                "message": "Recommend places for my 10-year-old daughter",
                "top_k": 50,
            },
        ).json()

    assert result["status"] == "completed"
    assert result["relevant_poi_ids"]
    assert result["response"].startswith("Based on the available place descriptions")
    assert "CF-" not in result["response"]
    repo = app.state.poi_repository
    assert result["recommended_poi_ids"]
    assert set(result["recommended_poi_ids"]).issubset(result["relevant_poi_ids"])
    assert len(result["recommended_poi_ids"]) == len(set(result["recommended_poi_ids"]))
    assert all(
        repo.get_by_id(poi_id).family_suitability_fixture is True
        for poi_id in result["relevant_poi_ids"]
    )


def test_chat_bounds_top_k_and_returns_typed_planning_response() -> None:
    app, _ = make_app([planning_intent()])

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"message": "Plan a historical walking itinerary", "top_k": 500},
        )
        result = client.post(
            "/chat",
            json={"message": "Plan a historical walking itinerary"},
        )

    assert response.status_code == 422
    assert result.status_code == 200
    body = result.json()
    assert body["status"] in {"feasible", "uncertain", "infeasible"}
    planning = body["planning_result"]
    assert planning is not None
    assert planning["validation"]["issues"] is not None
    if planning["itinerary"] is not None:
        assert planning["itinerary"]["stops"]
        assert "transfers" in planning
        assert all("poi_id" in stop and "starts_at" in stop for stop in planning["itinerary"]["stops"])


def test_chat_stop_replacement_preserves_unaffected_poi_ids_and_order() -> None:
    replacement = make_intent(
        "itinerary_modification",
        values={
            "itinerary_modification": {
                "action": "replace_stop",
                "stop_index": 1,
                "preserve_other_stops": True,
            },
            "family_requirements": {"required": True, "child_ages": [10]},
        },
        actions={
            "itinerary_modification": "set",
            "family_required": "set",
            "child_ages": "set",
        },
    )
    app, _ = make_app([planning_intent(), replacement])

    with TestClient(app) as client:
        first_response = client.post(
            "/chat",
            json={"message": "Plan a historical walking itinerary"},
        )
        first = first_response.json()
        original_ids = [
            stop["poi_id"]
            for stop in first["planning_result"]["itinerary"]["stops"]
        ]
        assert len(original_ids) >= 2
        revised_response = client.post(
            "/chat",
            json={
                "message": "Replace the second stop with a place suitable for my 10-year-old daughter",
                "conversation_id": first["conversation_id"],
            },
        )

    assert first_response.status_code == revised_response.status_code == 200
    revised = revised_response.json()
    assert revised["modification_applied"] is True
    revised_ids = [
        stop["poi_id"]
        for stop in revised["planning_result"]["itinerary"]["stops"]
    ]
    assert len(revised_ids) == len(original_ids)
    assert revised_ids[0] == original_ids[0]
    assert revised_ids[2:] == original_ids[2:]
    assert revised_ids[1] != original_ids[1]
    assert (
        app.state.poi_repository.get_by_id(revised_ids[1]).family_suitability_fixture
        is True
    )


def test_repeated_request_id_does_not_repeat_extraction_or_modification() -> None:
    app, extractor = make_app(
        [
            make_intent(
                "tourism_qa",
                values={"question": "Old Fortress"},
                actions={"question": "set"},
            )
        ]
    )
    request_id = str(uuid4())

    with TestClient(app) as client:
        first = client.post(
            "/chat",
            json={
                "message": "Tell me about Old Fortress",
                "request_id": request_id,
            },
        )
        replay = client.post(
            "/chat",
            json={
                "message": "Tell me about Old Fortress",
                "conversation_id": first.json()["conversation_id"],
                "request_id": request_id,
            },
        )

    assert first.status_code == replay.status_code == 200
    assert replay.json() == first.json()
    assert len(extractor.calls) == 1


def test_repeated_replacement_request_id_cannot_apply_stop_change_twice() -> None:
    replacement = make_intent(
        "itinerary_modification",
        values={
            "itinerary_modification": {
                "action": "replace_stop",
                "stop_index": 1,
                "preserve_other_stops": True,
            },
            "family_requirements": {"required": True, "child_ages": [10]},
        },
        actions={
            "itinerary_modification": "set",
            "family_required": "set",
            "child_ages": "set",
        },
    )
    app, extractor = make_app([planning_intent(), replacement])
    request_id = str(uuid4())

    with TestClient(app) as client:
        first = client.post(
            "/chat",
            json={"message": "Plan a historical walking itinerary"},
        ).json()
        replacement_payload = {
            "message": "Replace the second stop for my 10-year-old",
            "conversation_id": first["conversation_id"],
            "request_id": request_id,
        }
        applied = client.post("/chat", json=replacement_payload)
        replay = client.post("/chat", json=replacement_payload)

    assert applied.status_code == replay.status_code == 200
    assert applied.json()["modification_applied"] is True
    assert replay.json() == applied.json()
    assert len(extractor.calls) == 2


def test_live_weather_query_returns_typed_mocked_forecast() -> None:
    zone = ZoneInfo("Europe/Athens")
    forecast = WeatherForecast(
        place="Corfu Town",
        timezone="Europe/Athens",
        source="live",
        retrieved_at=datetime(2026, 10, 6, 10, tzinfo=zone),
        hourly=[
            {
                "at": datetime(2026, 10, 6, 10, tzinfo=zone),
                "temperature_c": 24,
                "precipitation_mm": 0,
            }
        ],
    )
    app, _ = make_app(
        [
            make_intent(
                "live_weather_query",
                values={
                    "weather_request": {
                        "relative_horizon": "next_24_hours",
                        "duration_hours": 24,
                    }
                },
                actions={"weather_request": "set"},
            )
        ],
        weather_provider=FakeWeatherProvider(forecast),
    )

    with TestClient(app) as client:
        result = client.post(
            "/chat",
            json={"message": "Weather in Corfu for the next 24 hours"},
        ).json()

    assert result["status"] == "completed"
    assert result["weather_forecast"]["source"] == "live"
    assert result["weather_forecast"]["timezone"] == "Europe/Athens"
    assert result["weather_forecast"]["hourly"][0]["precipitation_mm"] == 0


def test_weather_provider_failure_preserves_typed_unavailable_result() -> None:
    app, _ = make_app(
        [
            make_intent(
                "live_weather_query",
                values={
                    "weather_request": {
                        "relative_horizon": "next_24_hours",
                        "duration_hours": 24,
                    }
                },
                actions={"weather_request": "set"},
            )
        ],
        weather_provider=FakeWeatherProvider(
            error=WeatherProviderError("upstream unavailable")
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"message": "Weather in Corfu for the next 24 hours"},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "weather_unavailable"
    assert response.json()["weather_forecast"] is None
    assert response.json()["weather_error"] == "WeatherProviderError"


def test_weather_replanning_uses_injected_synthetic_fixture() -> None:
    repository = make_repository()
    manager = ConversationStateManager()
    intent = make_intent(
        "itinerary_creation",
        values={
            "requested_date": "2026-10-07",
            "time_window": {"start_time": "13:00", "end_time": "17:00"},
            "travel_mode": "walking",
            "hard_filters": {"category": "history"},
        },
        actions={
            "requested_date": "set",
            "start_time": "set",
            "end_time": "set",
            "travel_mode": "set",
            "filter_category": "set",
        },
    )
    state = manager.create_conversation()
    manager.apply_intent(
        state.conversation_id,
        intent,
        message="Create a historical walking itinerary on October 7, 13:00 to 17:00",
    )
    zone = ZoneInfo("Europe/Athens")
    starts = datetime(2026, 10, 7, 13, tzinfo=zone)
    current = datetime(2026, 10, 7, 14, 30, tzinfo=zone)
    stops = []
    for poi_id in ("CF-003", "CF-005"):
        poi = repository.get_by_id(poi_id)
        ends = current + timedelta(minutes=poi.suggested_visit_minutes)
        stops.append(ItineraryStop(poi_id=poi_id, starts_at=current, ends_at=ends))
        current = ends + timedelta(minutes=20)
    manager.update_itinerary(
        state.conversation_id,
        Itinerary(
            conversation_id=state.conversation_id,
            timezone="Europe/Athens",
            starts_at=starts,
            ends_at=datetime(2026, 10, 7, 17, tzinfo=zone),
            stops=stops,
        ),
    )
    provider = FixtureWeatherProvider(
        PROJECT_ROOT / "corfu_synthetic_rain.json",
        clock=lambda: datetime(2026, 10, 7, 12, tzinfo=zone),
    )
    app, _ = make_app(
        [make_intent("weather_based_replanning")],
        state_manager=manager,
        repository=repository,
        weather_provider=provider,
    )

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={
                "conversation_id": str(state.conversation_id),
                "message": "Adjust my outdoor historical itinerary for the rain",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["weather_forecast"]["source"] == "synthetic_fixture"
    assert body["rain_assessment"]["first_affected_interval"]["start"].startswith(
        "2026-10-07T15:00"
    )
    assert body["weather_replanning"]["original_poi_ids"] == ["CF-003", "CF-005"]
    assert body["weather_replanning"]["status"] in {
        "feasible",
        "uncertain",
    }, body["weather_replanning"]


@pytest.mark.parametrize(
    ("error", "expected_status", "failure_stage"),
    [
        (IntentRateLimitError("provider-secret-payload"), 429, "intent_extraction"),
        (IntentTimeoutError("provider-secret-payload"), 504, "intent_extraction"),
        (IntentTransientProviderError("provider-secret-payload"), 503, "intent_extraction"),
        (InvalidStructuredOutputError("provider-secret-payload"), 502, "intent_extraction"),
        (IntentProviderError("provider-secret-payload"), 502, "intent_extraction"),
        (IntentExtractionError("provider-secret-payload"), 502, "intent_extraction"),
        (IntentConfigurationError("GOOGLE_API_KEY=must-not-leak"), 503, "intent_extraction"),
    ],
)
def test_gemini_errors_map_to_safe_http_responses(
    error: Exception,
    expected_status: int,
    failure_stage: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    app, _ = make_app([error])

    with caplog.at_level(logging.ERROR, logger="corfu_assistant.main"):
        with TestClient(app) as client:
            response = client.post("/chat", json={"message": "Tell me about Corfu"})

    assert response.status_code == expected_status
    assert "must-not-leak" not in response.text
    assert "provider-secret-payload" not in response.text
    records = [record for record in caplog.records if hasattr(record, "failure_stage")]
    assert len(records) == 1
    record = records[0]
    assert record.failure_stage == failure_stage
    assert record.exception_class == type(error).__name__
    assert "failure_stage=intent_extraction" in caplog.text
    assert f"exception_class={type(error).__name__}" in caplog.text
    assert "provider_status_code=unknown" in caplog.text
    assert "exception_chain=" in caplog.text
    assert "traceback_frames=[" in caplog.text
    assert "test_api.py:" in caplog.text


def test_structured_intent_diagnostic_logs_safe_validation_path_and_attempts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    error = InvalidStructuredOutputError(
        "sensitive provider response must not appear",
        validation_issues=("value_error:requested_date",),
        attempts_used=2,
        retry_budget=1,
    )
    app, _ = make_app([error])

    with caplog.at_level(logging.ERROR, logger="corfu_assistant.main"):
        with TestClient(app) as client:
            response = client.post("/chat", json={"message": "Plan a visit"})

    assert response.status_code == 502
    assert "validation_issues=value_error:requested_date" in caplog.text
    assert "structured_attempts=2" in caplog.text
    assert "structured_retries=1/1" in caplog.text
    assert "sensitive provider response" not in caplog.text
    assert "sensitive provider response" not in response.text
    assert "provider-secret-payload" not in caplog.text
    assert "must-not-leak" not in caplog.text


def test_answer_generation_error_logs_safe_traceback_and_preserves_http_contract(
    caplog: pytest.LogCaptureFixture,
) -> None:
    app, _ = make_app()

    with TestClient(app) as client:
        def fail_answer_generation(_request: OrchestrationRequest):
            raise AnswerGenerationError("provider-secret-payload")

        app.state.services.orchestrator.process = fail_answer_generation
        with caplog.at_level(logging.ERROR, logger="corfu_assistant.main"):
            response = client.post("/chat", json={"message": "Tell me about Corfu"})

    assert response.status_code == 502
    assert response.json() == {"detail": "Grounded answer generation failed."}
    assert "provider-secret-payload" not in response.text
    records = [record for record in caplog.records if hasattr(record, "failure_stage")]
    assert len(records) == 1
    assert records[0].failure_stage == "answer_generation"
    assert records[0].exception_class == "AnswerGenerationError"
    assert "failure_stage=answer_generation" in caplog.text
    assert "exception_class=AnswerGenerationError" in caplog.text
    assert "provider_status_code=unknown" in caplog.text
    assert "traceback_frames=[" in caplog.text
    assert "test_api.py:" in caplog.text
    assert "provider-secret-payload" not in caplog.text


def test_rendered_provider_diagnostic_includes_safe_status_and_chain_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class ProviderFailure(RuntimeError):
        status_code = 503

    cause = ProviderFailure("response body and GOOGLE_API_KEY=must-not-log")
    error = IntentTransientProviderError("private provider message")
    error.__cause__ = cause
    app, _ = make_app([error])

    with caplog.at_level(logging.ERROR, logger="corfu_assistant.main"):
        with TestClient(app) as client:
            response = client.post("/chat", json={"message": "private request prompt"})

    assert response.status_code == 503
    assert "failure_stage=intent_extraction" in caplog.text
    assert "exception_class=IntentTransientProviderError" in caplog.text
    assert "provider_status_code=503" in caplog.text
    assert "exception_chain=IntentTransientProviderError>ProviderFailure" in caplog.text
    assert "traceback_frames=[" in caplog.text
    assert "test_api.py:" in caplog.text
    assert "response body" not in caplog.text
    assert "must-not-log" not in caplog.text
    assert "private provider message" not in caplog.text
    assert "private request prompt" not in caplog.text


def test_rendered_invalid_argument_diagnostic_uses_allowlisted_provider_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class ProviderFailure(RuntimeError):
        code = 400
        status = "INVALID_ARGUMENT"
        message = "Request contains an invalid argument."

    error = IntentProviderError("private provider response")
    error.__cause__ = ProviderFailure("GOOGLE_API_KEY=must-not-log")
    app, _ = make_app([error])

    with caplog.at_level(logging.ERROR, logger="corfu_assistant.main"):
        with TestClient(app) as client:
            response = client.post("/chat", json={"message": "private prompt"})

    assert response.status_code == 502
    assert "provider_status_code=400" in caplog.text
    assert "provider_status_label=INVALID_ARGUMENT" in caplog.text
    assert "provider_rejection=Request contains an invalid argument." in caplog.text
    assert "must-not-log" not in caplog.text
    assert "private provider response" not in caplog.text
    assert "private prompt" not in caplog.text


def test_invalid_extracted_metadata_constraint_is_a_clear_client_error() -> None:
    app, _ = make_app(
        [
            make_intent(
                values={"hard_filters": {"category": "not-a-dataset-category"}},
                actions={"filter_category": "set"},
            )
        ]
    )

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"message": "Recommend places in this category"},
        )

    assert response.status_code == 422
    assert response.json() == {
        "detail": "A requested POI category or tag is not supported."
    }


def test_missing_gemini_credentials_do_not_break_liveness() -> None:
    app = create_app(
        Settings(
            dataset_path=PROJECT_ROOT / "corfu_dataset.json",
            google_api_key=None,
        )
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 200
        response = client.post("/chat", json={"message": "Tell me about Corfu"})

    assert response.status_code == 503
    assert "GOOGLE_API_KEY" not in response.text


def test_invalid_and_empty_messages_are_rejected_by_request_validation() -> None:
    app, extractor = make_app([])

    with TestClient(app) as client:
        empty = client.post("/chat", json={"message": "   "})
        missing = client.post("/chat", json={})
        wrong_type = client.post("/chat", json={"message": 42})

    assert empty.status_code == 422
    assert missing.status_code == 422
    assert wrong_type.status_code == 422
    assert extractor.calls == []


def test_unknown_conversation_returns_404() -> None:
    app, extractor = make_app([])

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={
                "message": "Tell me about Corfu",
                "conversation_id": str(uuid4()),
            },
        )

    assert response.status_code == 404
    assert response.json() == {"detail": "Conversation not found"}
    assert extractor.calls == []


def test_failed_extraction_preserves_previous_itinerary() -> None:
    repository = make_repository()
    manager = ConversationStateManager()
    state = manager.create_conversation()
    zone = ZoneInfo("Europe/Athens")
    start = datetime(2026, 10, 6, 11, tzinfo=zone)
    stop_end = start + timedelta(minutes=20)
    itinerary = Itinerary(
        conversation_id=state.conversation_id,
        timezone="Europe/Athens",
        starts_at=start,
        ends_at=datetime(2026, 10, 6, 15, tzinfo=zone),
        stops=[
            ItineraryStop(
                poi_id="CF-003",
                starts_at=start,
                ends_at=stop_end,
            )
        ],
    )
    manager.update_itinerary(state.conversation_id, itinerary)
    app, _ = make_app(
        [InvalidStructuredOutputError("bad output")],
        state_manager=manager,
        repository=repository,
    )

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={
                "message": "Replace the second stop",
                "conversation_id": str(state.conversation_id),
            },
        )
        persisted = client.get(f"/conversations/{state.conversation_id}").json()

    assert response.status_code == 502
    assert persisted["itinerary"]["stops"][0]["poi_id"] == "CF-003"


def test_unexpected_error_is_sanitized() -> None:
    app, _ = make_app([RuntimeError("secret internals")])

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/chat", json={"message": "Tell me about Corfu"})

    assert response.status_code == 500
    assert response.json() == {"detail": "An unexpected server error occurred."}
    assert "secret internals" not in response.text


def test_cors_preflight_allows_only_configured_vite_origins() -> None:
    app, _ = make_app()

    with TestClient(app) as client:
        allowed = client.options(
            "/chat",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        denied = client.options(
            "/chat",
            headers={
                "Origin": "http://localhost:9000",
                "Access-Control-Request-Method": "POST",
            },
        )

    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "access-control-allow-origin" not in denied.headers
    assert "access-control-allow-credentials" not in allowed.headers


def test_conversation_get_and_delete_routes() -> None:
    manager = ConversationStateManager()
    state = manager.create_conversation()
    app, _ = make_app([], state_manager=manager)

    with TestClient(app) as client:
        loaded = client.get(f"/conversations/{state.conversation_id}")
        deleted = client.delete(f"/conversations/{state.conversation_id}")
        missing = client.get(f"/conversations/{state.conversation_id}")
        deleted_again = client.delete(f"/conversations/{state.conversation_id}")

    assert loaded.status_code == 200
    assert loaded.json()["conversation_id"] == str(state.conversation_id)
    assert deleted.status_code == 204
    assert missing.status_code == 404
    assert deleted_again.status_code == 404


def test_invalid_cors_configuration_rejects_wildcard() -> None:
    with pytest.raises(ValueError, match="CORS origins"):
        Settings(cors_origins=["*"])


def test_additional_explicit_cors_origins_can_be_configured() -> None:
    app, _ = make_app(
        [],
        settings=Settings(
            dataset_path=PROJECT_ROOT / "corfu_dataset.json",
            google_api_key=None,
            cors_origins=[
                "http://localhost:5173",
                "http://127.0.0.1:5173",
                "http://localhost:5174",
            ],
        ),
    )

    with TestClient(app) as client:
        response = client.options(
            "/chat",
            headers={
                "Origin": "http://localhost:5174",
                "Access-Control-Request-Method": "POST",
            },
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5174"


def test_orchestration_request_contract_enforces_top_k_bound() -> None:
    assert OrchestrationRequest(message="valid").top_k == 5
    with pytest.raises(ValueError):
        OrchestrationRequest(message="too many", top_k=51)
