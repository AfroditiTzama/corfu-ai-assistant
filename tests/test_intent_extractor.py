from datetime import date
import json
import logging
from typing import Any

import pytest
from pydantic import SecretStr, ValidationError
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser

from corfu_assistant.config import Settings
from corfu_assistant.intent_extractor import (
    GeminiIntentExtractor,
    IntentConfigurationError,
    IntentProviderError,
    IntentRateLimitError,
    IntentTransientProviderError,
    IntentTimeoutError,
    InvalidStructuredOutputError,
)
from corfu_assistant.models import ConversationState, UserIntent
from corfu_assistant.state_manager import ConversationStateManager


_ACTION_FIELDS = (
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


def intent_actions(**updates: str) -> dict[str, str]:
    actions = dict.fromkeys(_ACTION_FIELDS, "preserve")
    actions.update(updates)
    return actions


def payload_with_actions(payload: dict[str, Any]) -> dict[str, Any]:
    actions = intent_actions()
    if payload.get("requested_date") is not None:
        actions["requested_date"] = "set"
    window = payload.get("time_window")
    if window:
        for field in ("start_time", "end_time"):
            if window.get(field) is not None:
                actions[field] = "set"
    semantic = payload.get("semantic_preferences")
    if semantic and semantic.get("interests") is not None:
        actions["semantic_interests"] = "set"
    filters = payload.get("hard_filters")
    if filters:
        for source, target in (
            ("category", "filter_category"),
            ("tags", "filter_tags"),
            ("setting", "filter_setting"),
            ("family_suitability_fixture", "filter_family_suitability"),
            ("excluded_categories", "filter_excluded_categories"),
            ("excluded_tags", "filter_excluded_tags"),
            ("excluded_poi_ids", "filter_excluded_poi_ids"),
        ):
            if source in filters and filters[source] is not None:
                actions[target] = "set"
    if payload.get("travel_mode") is not None:
        actions["travel_mode"] = "set"
    family = payload.get("family_requirements")
    if family:
        for source in ("required", "child_ages"):
            if source in family and family[source] is not None:
                actions["family_required" if source == "required" else "child_ages"] = "set"
    for field in (
        "itinerary_modification",
        "question",
        "clarification_question",
        "weather_request",
    ):
        if payload.get(field) is not None:
            actions[field] = "set"
    return {**payload, "field_actions": actions}


class FakeStructuredRunnable:
    def __init__(self, response: object = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.inputs: list[object] = []

    def invoke(self, input: object) -> object:
        self.inputs.append(input)
        if self.error is not None:
            raise self.error
        return self.response


class SequencedStructuredRunnable:
    def __init__(self, outcomes: list[object | Exception]) -> None:
        self.outcomes = list(outcomes)
        self.inputs: list[object] = []

    def invoke(self, input: object) -> object:
        self.inputs.append(input)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeStatusError(RuntimeError):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"provider returned HTTP {status_code}")
        self.status_code = status_code


def test_extracts_typed_intent_and_keeps_semantic_preferences_separate() -> None:
    runnable = FakeStructuredRunnable(
        {
            "parsed": payload_with_actions({
                "intent": "itinerary_creation",
                "requested_date": "2026-10-06",
                "time_window": {"start_time": "11:00", "end_time": "15:00"},
                "semantic_preferences": {"interests": ["Venetian history"]},
                "hard_filters": {"setting": "outdoor"},
                "travel_mode": "walking",
                "family_requirements": {"required": True, "child_ages": [10]},
            }),
            "parsing_error": None,
            "raw": object(),
        }
    )
    extractor = GeminiIntentExtractor(
        api_key=None,
        structured_runnable=runnable,
    )

    intent = extractor.extract("Plan a walk focused on Venetian history.")

    assert intent.intent == "itinerary_creation"
    assert intent.requested_date == date(2026, 10, 6)
    assert intent.time_window.start_time == "11:00"
    assert intent.time_window.end_time == "15:00"
    assert intent.semantic_preferences.interests == ["Venetian history"]
    assert intent.hard_filters.setting == "outdoor"
    assert intent.travel_mode == "walking"
    assert intent.family_requirements.child_ages == [10]
    messages = runnable.inputs[0]
    assert "untrusted data" in messages[1].content


def test_multi_interest_focus_instruction_keeps_preferences_out_of_tag_intersection() -> None:
    intent_payload = payload_with_actions(
        {
            "intent": "itinerary_creation",
            "requested_date": "2026-10-07",
            "time_window": {"start_time": "14:00", "end_time": "18:00"},
            "semantic_preferences": {
                "interests": ["gardens", "waterfront views", "scenic outdoor attractions"]
            },
            "hard_filters": {"setting": "outdoor"},
            "travel_mode": "walking",
        }
    )
    runnable = FakeStructuredRunnable(
        {"parsed": intent_payload, "parsing_error": None, "raw": object()}
    )
    extractor = GeminiIntentExtractor(
        api_key=None,
        structured_runnable=runnable,
        metadata_categories=("history", "nature"),
        metadata_tags=("garden", "outdoor", "waterfront"),
    )

    intent = extractor.extract(
        "Plan a four-hour walking itinerary in Corfu on Wednesday, October 7, 2026, "
        "from 14:00 to 18:00, focusing on gardens, waterfront views and scenic outdoor attractions."
    )

    assert intent.semantic_preferences.interests == [
        "gardens",
        "waterfront views",
        "scenic outdoor attractions",
    ]
    assert intent.hard_filters.setting == "outdoor"
    assert intent.hard_filters.tags is None
    assert intent.field_actions.filter_tags == "preserve"
    system_message = runnable.inputs[0][0]
    assert "Do not turn each item in a multi-interest list into a required tag" in system_message.content


def test_unspecified_and_explicit_empty_updates_remain_distinguishable() -> None:
    intent = UserIntent(
        intent="poi_recommendation",
        semantic_preferences={"interests": []},
        field_actions=intent_actions(semantic_interests="set"),
    )

    assert intent.field_actions.requested_date == "preserve"
    assert intent.requested_date is None
    assert intent.field_actions.semantic_interests == "set"
    assert intent.semantic_preferences.interests == []
    assert intent.effective_timezone == "Europe/Athens"


def test_recommendation_count_is_typed_and_bounded() -> None:
    intent = UserIntent.model_validate(
        payload_with_actions(
            {"intent": "poi_recommendation", "recommendation_count": 4}
        )
    )

    assert intent.recommendation_count == 4
    with pytest.raises(ValidationError):
        UserIntent.model_validate(
            payload_with_actions(
                {"intent": "poi_recommendation", "recommendation_count": 0}
            )
        )
    with pytest.raises(ValidationError):
        UserIntent.model_validate(
            payload_with_actions(
                {"intent": "tourism_qa", "recommendation_count": 2}
            )
        )


def test_extractor_preserves_explicit_recommendation_count() -> None:
    runnable = FakeStructuredRunnable(
        {
            "parsed": payload_with_actions(
                {"intent": "poi_recommendation", "recommendation_count": 3}
            ),
            "parsing_error": None,
            "raw": object(),
        }
    )
    extractor = GeminiIntentExtractor(api_key=None, structured_runnable=runnable)

    intent = extractor.extract("Recommend three places for an art lover")

    assert intent.recommendation_count == 3


def test_extractor_supports_named_poi_and_metadata_exclusions() -> None:
    payload = payload_with_actions(
        {
            "intent": "poi_recommendation",
            "hard_filters": {
                "excluded_tags": ["museum"],
                "excluded_poi_ids": ["CF-022"],
            },
        }
    )
    runnable = FakeStructuredRunnable({"parsed": payload})
    extractor = GeminiIntentExtractor(
        api_key=None,
        metadata_categories=["history"],
        metadata_tags=["museum"],
        metadata_poi_names={"CF-022": "Mon Repos Museum of Palaiopolis"},
        structured_runnable=runnable,
    )

    intent = extractor.extract("Recommend outdoor places, not that museum")

    assert intent.hard_filters is not None
    assert intent.hard_filters.excluded_tags == ["museum"]
    assert intent.hard_filters.excluded_poi_ids == ["CF-022"]
    assert intent.field_actions.filter_excluded_tags == "set"
    assert intent.field_actions.filter_excluded_poi_ids == "set"
    system_message = runnable.inputs[0][0]
    assert "CF-022: Mon Repos Museum of Palaiopolis" in system_message.content


@pytest.mark.parametrize(
    "payload",
    [
        payload_with_actions({"intent": "tourism_qa", "question": "What is inside the old fortress?"}),
        payload_with_actions({"intent": "poi_recommendation", "semantic_preferences": {"interests": ["art"]}}),
        payload_with_actions({"intent": "itinerary_creation", "travel_mode": "walking"}),
        payload_with_actions({
            "intent": "itinerary_modification",
            "itinerary_modification": {"action": "replace_stop", "stop_index": 1},
        }),
        payload_with_actions({"intent": "live_weather_query"}),
        payload_with_actions({"intent": "weather_based_replanning"}),
        payload_with_actions({"intent": "clarification", "clarification_question": "Which date?"}),
    ],
)
def test_user_intent_supports_each_requested_intent_kind(payload: dict[str, Any]) -> None:
    assert UserIntent.model_validate(payload).intent == payload["intent"]


@pytest.mark.parametrize(
    "payload",
    [
        payload_with_actions({"intent": "clarification"}),
        payload_with_actions({"intent": "itinerary_modification"}),
        payload_with_actions({"intent": "poi_recommendation", "travel_mode": "helicopter"}),
        payload_with_actions({"intent": "poi_recommendation", "hard_filters": {"setting": "covered"}}),
        payload_with_actions({"intent": "poi_recommendation", "availability": "open"}),
        payload_with_actions({"intent": "itinerary_creation", "time_window": {"start_time": "15:00", "end_time": "11:00"}}),
    ],
)
def test_user_intent_rejects_invalid_or_untrusted_fields(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        UserIntent.model_validate(payload)


def test_extract_requires_structured_output() -> None:
    extractor = GeminiIntentExtractor(
        api_key=None,
        structured_runnable=FakeStructuredRunnable(
            {"parsed": None, "parsing_error": ValueError("invalid"), "raw": object()}
        ),
    )

    with pytest.raises(InvalidStructuredOutputError, match="could not be parsed"):
        extractor.extract("Find a place to visit")


def test_extract_rejects_schema_invalid_provider_result() -> None:
    extractor = GeminiIntentExtractor(
        api_key=None,
        structured_runnable=FakeStructuredRunnable(
            payload_with_actions({"intent": "made_up"})
        ),
    )

    with pytest.raises(InvalidStructuredOutputError, match="does not match"):
        extractor.extract("Find a place to visit")


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (TimeoutError("secret-looking detail"), IntentTimeoutError),
        (RuntimeError("429 too many requests"), IntentRateLimitError),
        (RuntimeError("service unavailable"), IntentProviderError),
    ],
)
def test_provider_errors_are_classified_without_returning_raw_error_text(
    error: Exception,
    expected: type[Exception],
) -> None:
    extractor = GeminiIntentExtractor(
        api_key=None,
        structured_runnable=FakeStructuredRunnable(error=error),
    )

    with pytest.raises(expected) as raised:
        extractor.extract("Plan a visit")
    assert str(error) not in str(raised.value)
    assert raised.value.__cause__ is error


@pytest.mark.parametrize("status_code", [500, 502, 503, 504])
def test_transient_provider_errors_retry_with_bounded_jitter(
    status_code: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("corfu_assistant.intent_extractor.time.sleep", sleeps.append)
    monkeypatch.setattr(
        "corfu_assistant.intent_extractor.random.uniform",
        lambda lower, upper: upper,
    )
    runnable = SequencedStructuredRunnable(
        [
            FakeStatusError(status_code),
            payload_with_actions({"intent": "clarification", "clarification_question": "Which date?"}),
        ]
    )
    extractor = GeminiIntentExtractor(
        api_key=None,
        retry_budget=1,
        structured_runnable=runnable,
    )

    intent = extractor.extract("Plan a visit")

    assert intent.intent == "clarification"
    assert len(runnable.inputs) == 2
    assert sleeps == [0.25]


def test_transient_error_exhaustion_preserves_final_provider_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("corfu_assistant.intent_extractor.time.sleep", sleeps.append)
    monkeypatch.setattr(
        "corfu_assistant.intent_extractor.random.uniform",
        lambda lower, upper: upper,
    )
    errors = [FakeStatusError(503) for _ in range(3)]
    runnable = SequencedStructuredRunnable(errors)
    extractor = GeminiIntentExtractor(
        api_key=None,
        retry_budget=2,
        structured_runnable=runnable,
    )

    with pytest.raises(IntentTransientProviderError) as raised:
        extractor.extract("Plan a visit")

    assert len(runnable.inputs) == 3
    assert sleeps == [0.25, 0.5]
    assert raised.value.__cause__ is errors[-1]


@pytest.mark.parametrize("status_code", [400, 401, 403, 429])
def test_client_errors_and_rate_limits_are_not_retried(
    status_code: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("corfu_assistant.intent_extractor.time.sleep", sleeps.append)
    error = FakeStatusError(status_code)
    runnable = SequencedStructuredRunnable([error])
    extractor = GeminiIntentExtractor(api_key=None, structured_runnable=runnable)
    expected_error = IntentRateLimitError if status_code == 429 else IntentProviderError

    with pytest.raises(expected_error) as raised:
        extractor.extract("Plan a visit")

    assert len(runnable.inputs) == 1
    assert sleeps == []
    assert raised.value.__cause__ is error


def test_invalid_structured_output_is_not_retried() -> None:
    runnable = SequencedStructuredRunnable(
        [payload_with_actions({"intent": "not-a-real-intent"})]
    )
    extractor = GeminiIntentExtractor(
        api_key=None,
        structured_output_retry_budget=0,
        structured_runnable=runnable,
    )

    with pytest.raises(InvalidStructuredOutputError):
        extractor.extract("Plan a visit")

    assert len(runnable.inputs) == 1


@pytest.mark.parametrize("missing_value", [False, True])
def test_malformed_requested_date_recovers_with_one_schema_retry(
    missing_value: bool,
) -> None:
    malformed = payload_with_actions({"intent": "itinerary_creation"})
    malformed["field_actions"]["requested_date"] = "set"
    if not missing_value:
        malformed["requested_date"] = None
    valid = payload_with_actions(
        {"intent": "clarification", "clarification_question": "Which date?"}
    )
    runnable = SequencedStructuredRunnable([malformed, valid])
    extractor = GeminiIntentExtractor(api_key=None, structured_runnable=runnable)

    intent = extractor.extract("Plan a visit")

    assert intent.intent == "clarification"
    assert intent.requested_date is None
    assert len(runnable.inputs) == 2
    retry_messages = runnable.inputs[1]
    assert "did not satisfy the UserIntent schema" in retry_messages[1].content
    assert "If a new date was requested" in retry_messages[1].content


def test_malformed_structured_output_exhausts_separate_bounded_budget() -> None:
    malformed = payload_with_actions({"intent": "itinerary_creation"})
    malformed["field_actions"]["requested_date"] = "set"
    malformed["requested_date"] = None
    runnable = SequencedStructuredRunnable([malformed, malformed])
    extractor = GeminiIntentExtractor(api_key=None, structured_runnable=runnable)

    with pytest.raises(InvalidStructuredOutputError, match="does not match"):
        extractor.extract("Plan a visit")

    assert len(runnable.inputs) == 2


@pytest.mark.parametrize("field", ["requested_date", "semantic_interests"])
def test_weather_replanning_recovers_omitted_inherited_field(field: str) -> None:
    manager = ConversationStateManager()
    state = manager.create_conversation()
    initial = UserIntent.model_validate(payload_with_actions({
        "intent": "itinerary_creation",
        "requested_date": "2026-10-07",
        "time_window": {"start_time": "13:00", "end_time": "17:00"},
        "semantic_preferences": {"interests": ["outdoor history"]},
        "travel_mode": "walking",
    }))
    state = manager.apply_intent(state.conversation_id, initial, message="Plan Wednesday walk").state
    inherited_date = state.active_request.requested_date
    inherited_interests = list(state.preferences.interests)

    malformed = payload_with_actions({"intent": "weather_based_replanning"})
    malformed["field_actions"][field] = "set"
    if field == "requested_date":
        malformed["requested_date"] = None
    else:
        malformed["semantic_preferences"] = {"interests": None}
    valid = payload_with_actions({"intent": "weather_based_replanning"})
    runnable = SequencedStructuredRunnable([malformed, valid])
    extractor = GeminiIntentExtractor(api_key=None, structured_runnable=runnable)

    extracted = extractor.extract(
        "Check the forecast and adapt my existing walk for rain.", state
    )
    updated = manager.apply_intent(
        state.conversation_id,
        extracted,
        message="Check the forecast and adapt my existing walk for rain.",
    ).state

    assert len(runnable.inputs) == 2
    assert extracted.intent == "weather_based_replanning"
    assert extracted.field_actions.requested_date == "preserve"
    assert extracted.field_actions.semantic_interests == "preserve"
    assert updated.active_request.requested_date == inherited_date
    assert updated.preferences.interests == inherited_interests


@pytest.mark.parametrize("invalid_action", ["preserve", "clear"])
def test_initial_itinerary_recovers_interest_list_with_mismatched_action(
    invalid_action: str,
) -> None:
    malformed = payload_with_actions({
        "intent": "itinerary_creation",
        "semantic_preferences": {"interests": ["outdoor history"]},
    })
    malformed["field_actions"]["semantic_interests"] = invalid_action
    valid = payload_with_actions({
        "intent": "itinerary_creation",
        "semantic_preferences": {"interests": ["outdoor history"]},
    })
    runnable = SequencedStructuredRunnable([malformed, valid])
    extractor = GeminiIntentExtractor(api_key=None, structured_runnable=runnable)

    extracted = extractor.extract("Plan an outdoor historical itinerary")

    assert extracted.field_actions.semantic_interests == "set"
    assert extracted.semantic_preferences.interests == ["outdoor history"]
    assert len(runnable.inputs) == 2
    retry_guidance = runnable.inputs[1][1].content
    assert "semantic_preferences.interests list requires" in retry_guidance
    assert "Do not drop an explicitly stated interest" in retry_guidance


def test_weather_followup_retries_mismatched_new_interests_without_discarding_them() -> None:
    manager = ConversationStateManager()
    state = manager.create_conversation()
    initial = UserIntent.model_validate(payload_with_actions({
        "intent": "itinerary_creation",
        "semantic_preferences": {"interests": ["history"]},
    }))
    state = manager.apply_intent(
        state.conversation_id, initial, message="Plan a history walk"
    ).state
    malformed = payload_with_actions({
        "intent": "weather_based_replanning",
        "semantic_preferences": {"interests": ["architecture"]},
    })
    malformed["field_actions"]["semantic_interests"] = "preserve"
    valid = payload_with_actions({
        "intent": "weather_based_replanning",
        "semantic_preferences": {"interests": ["architecture"]},
    })
    runnable = SequencedStructuredRunnable([malformed, valid])
    extractor = GeminiIntentExtractor(api_key=None, structured_runnable=runnable)

    extracted = extractor.extract(
        "Keep the weather replan but change my interest to architecture", state
    )
    updated = manager.apply_intent(
        state.conversation_id,
        extracted,
        message="Keep the weather replan but change my interest to architecture",
    ).state

    assert len(runnable.inputs) == 2
    assert extracted.field_actions.semantic_interests == "set"
    assert updated.preferences.interests == ["architecture"]


def test_weather_replanning_exhausted_malformed_fields_remains_strict_and_state_safe() -> None:
    manager = ConversationStateManager()
    state = manager.create_conversation()
    initial = UserIntent.model_validate(payload_with_actions({
        "intent": "itinerary_creation",
        "requested_date": "2026-10-07",
        "time_window": {"start_time": "13:00", "end_time": "17:00"},
        "semantic_preferences": {"interests": ["outdoor history"]},
        "travel_mode": "walking",
    }))
    state = manager.apply_intent(state.conversation_id, initial, message="Plan Wednesday walk").state
    before = state.model_copy(deep=True)
    malformed = payload_with_actions({"intent": "weather_based_replanning"})
    malformed["field_actions"]["requested_date"] = "set"
    malformed["requested_date"] = None
    malformed["field_actions"]["semantic_interests"] = "set"
    malformed["semantic_preferences"] = {"interests": None}
    runnable = SequencedStructuredRunnable([malformed, malformed])
    extractor = GeminiIntentExtractor(api_key=None, structured_runnable=runnable)

    with pytest.raises(InvalidStructuredOutputError) as raised:
        extractor.extract("Check rain and adapt the walk")

    assert raised.value.attempts_used == 2
    assert raised.value.retry_budget == 1
    assert len(runnable.inputs) == 2
    assert manager.get(state.conversation_id) == before


def test_langchain_wrapped_pydantic_error_enters_schema_recovery() -> None:
    malformed = payload_with_actions({"intent": "itinerary_creation"})
    malformed["field_actions"]["requested_date"] = "set"
    malformed["requested_date"] = None
    parser = PydanticOutputParser(pydantic_object=UserIntent)
    with pytest.raises(OutputParserException) as parser_failure:
        parser.parse(json.dumps(malformed))

    valid = payload_with_actions(
        {"intent": "clarification", "clarification_question": "Which date?"}
    )
    runnable = SequencedStructuredRunnable(
        [
            {"parsed": None, "parsing_error": parser_failure.value},
            {"parsed": valid, "parsing_error": None},
        ]
    )
    extractor = GeminiIntentExtractor(api_key=None, structured_runnable=runnable)

    intent = extractor.extract("Plan a visit")

    assert intent.intent == "clarification"
    assert len(runnable.inputs) == 2


def test_langchain_parser_error_exhaustion_has_safe_validation_diagnostics(
    caplog: pytest.LogCaptureFixture,
) -> None:
    malformed = payload_with_actions({"intent": "itinerary_creation"})
    malformed["field_actions"]["requested_date"] = "set"
    malformed["requested_date"] = None
    parser = PydanticOutputParser(pydantic_object=UserIntent)
    with pytest.raises(OutputParserException) as parser_failure:
        parser.parse(json.dumps(malformed))
    runnable = SequencedStructuredRunnable(
        [
            {"parsed": None, "parsing_error": parser_failure.value},
            {"parsed": None, "parsing_error": parser_failure.value},
        ]
    )
    extractor = GeminiIntentExtractor(api_key=None, structured_runnable=runnable)

    with caplog.at_level(logging.INFO, logger="corfu_assistant.intent_extractor"):
        with pytest.raises(InvalidStructuredOutputError) as raised:
            extractor.extract("Plan a visit")

    assert len(runnable.inputs) == 2
    assert raised.value.validation_issues == ("value_error:requested_date",)
    assert raised.value.attempts_used == 2
    assert raised.value.retry_budget == 1
    assert "outcome=schema_retry" in caplog.text
    assert "validation_issues=value_error:requested_date" in caplog.text
    assert "attempts_used=2" in caplog.text
    assert "requested_date action 'set' requires a value" not in caplog.text


def test_requested_date_diagnostics_render_only_safe_classifications(
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = ConversationStateManager()
    state = manager.create_conversation()
    initial = UserIntent.model_validate(payload_with_actions({
        "intent": "itinerary_creation",
        "requested_date": "2026-10-07",
        "time_window": {"start_time": "13:00", "end_time": "17:00"},
    }))
    state = manager.apply_intent(
        state.conversation_id, initial, message="private initial request"
    ).state
    malformed = payload_with_actions({"intent": "weather_based_replanning"})
    malformed["field_actions"]["requested_date"] = "set"
    malformed["requested_date"] = "private-invalid-date-value"
    extractor = GeminiIntentExtractor(
        api_key=None,
        structured_output_retry_budget=0,
        structured_runnable=SequencedStructuredRunnable([malformed]),
    )

    with caplog.at_level(logging.WARNING, logger="corfu_assistant.intent_extractor"):
        with pytest.raises(InvalidStructuredOutputError):
            extractor.extract("private weather follow-up", state)

    rendered = caplog.text
    assert "intent_type=weather_based_replanning" in rendered
    assert "requested_date_action=set" in rendered
    assert "value_classification=malformed" in rendered
    assert "value_type=str" in rendered
    assert "active_request_exists=True" in rendered
    assert "active_request_requested_date_exists=True" in rendered
    assert "validation_error_category=date_from_datetime_parsing" in rendered
    assert "normalization_stage=after" in rendered
    assert "private-invalid-date-value" not in rendered
    assert "2026-10-07" not in rendered
    assert "private initial request" not in rendered
    assert "private weather follow-up" not in rendered


def test_pre_normalization_parser_failure_logs_no_raw_payload_details(
    caplog: pytest.LogCaptureFixture,
) -> None:
    malformed = payload_with_actions({"intent": "itinerary_creation"})
    malformed["field_actions"]["requested_date"] = "set"
    malformed["requested_date"] = None
    parser = PydanticOutputParser(pydantic_object=UserIntent)
    with pytest.raises(OutputParserException) as parser_failure:
        parser.parse(json.dumps(malformed))
    extractor = GeminiIntentExtractor(
        api_key=None,
        structured_output_retry_budget=0,
        structured_runnable=SequencedStructuredRunnable([{
            "parsed": None,
            "parsing_error": parser_failure.value,
            "raw": "private-provider-body",
        }]),
    )

    with caplog.at_level(logging.WARNING, logger="corfu_assistant.intent_extractor"):
        with pytest.raises(InvalidStructuredOutputError):
            extractor.extract("private user prompt")

    rendered = caplog.text
    assert "intent_type=unknown" in rendered
    assert "requested_date_action=unknown" in rendered
    assert "value_classification=missing" in rendered
    assert "value_type=missing" in rendered
    assert "active_request_exists=False" in rendered
    assert "active_request_requested_date_exists=False" in rendered
    assert "validation_error_category=value_error" in rendered
    assert "normalization_stage=before" in rendered
    assert "private-provider-body" not in rendered
    assert "private user prompt" not in rendered
    assert "requested_date action 'set' requires a value" not in rendered


@pytest.mark.parametrize(
    ("action", "preferences", "classification", "value_type", "error_category"),
    [
        ("set", {"interests": None}, "null", "none", "value_error"),
        ("set", None, "null", "none", "value_error"),
        ("preserve", {"interests": ["private-history-theme"]}, "present", "list", "value_error"),
        ("set", {"interests": "private-history-theme"}, "malformed", "str", "list_type"),
    ],
)
def test_semantic_interests_diagnostics_render_only_safe_classifications(
    action: str,
    preferences: object,
    classification: str,
    value_type: str,
    error_category: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    malformed = payload_with_actions({
        "intent": "itinerary_creation",
        "semantic_preferences": preferences,
    })
    malformed["field_actions"]["semantic_interests"] = action
    extractor = GeminiIntentExtractor(
        api_key=None,
        structured_output_retry_budget=0,
        structured_runnable=SequencedStructuredRunnable([malformed]),
    )

    with caplog.at_level(logging.WARNING, logger="corfu_assistant.intent_extractor"):
        with pytest.raises(InvalidStructuredOutputError):
            extractor.extract("private initial itinerary request")

    rendered = caplog.text
    assert "intent_type=itinerary_creation" in rendered
    assert f"semantic_interests_action={action}" in rendered
    assert f"value_classification={classification}" in rendered
    assert f"value_type={value_type}" in rendered
    assert "existing_conversation_interests=False" in rendered
    assert f"validation_error_category={error_category}" in rendered
    assert "normalization_stage=after" in rendered
    assert "private-history-theme" not in rendered
    assert "private initial itinerary request" not in rendered


def test_semantic_interests_parser_failure_logs_unknown_shape_without_payload(
    caplog: pytest.LogCaptureFixture,
) -> None:
    malformed = payload_with_actions({"intent": "itinerary_creation"})
    malformed["field_actions"]["semantic_interests"] = "set"
    malformed["semantic_preferences"] = {"interests": None}
    parser = PydanticOutputParser(pydantic_object=UserIntent)
    with pytest.raises(OutputParserException) as parser_failure:
        parser.parse(json.dumps(malformed))
    extractor = GeminiIntentExtractor(
        api_key=None,
        structured_output_retry_budget=0,
        structured_runnable=SequencedStructuredRunnable([{
            "parsed": None,
            "parsing_error": parser_failure.value,
            "raw": "private-semantic-provider-body",
        }]),
    )

    with caplog.at_level(logging.WARNING, logger="corfu_assistant.intent_extractor"):
        with pytest.raises(InvalidStructuredOutputError):
            extractor.extract("private semantic prompt")

    rendered = caplog.text
    assert "semantic_interests_action=unknown" in rendered
    assert "value_classification=missing" in rendered
    assert "value_type=missing" in rendered
    assert "normalization_stage=before" in rendered
    assert "private-semantic-provider-body" not in rendered
    assert "private semantic prompt" not in rendered


def test_parser_exception_raised_during_invoke_does_not_enter_schema_retry() -> None:
    malformed = payload_with_actions({"intent": "itinerary_creation"})
    malformed["field_actions"]["requested_date"] = "set"
    malformed["requested_date"] = None
    parser = PydanticOutputParser(pydantic_object=UserIntent)
    with pytest.raises(OutputParserException) as parser_failure:
        parser.parse(json.dumps(malformed))
    runnable = SequencedStructuredRunnable([parser_failure.value])
    extractor = GeminiIntentExtractor(api_key=None, structured_runnable=runnable)

    with pytest.raises(IntentProviderError):
        extractor.extract("Plan a visit")

    assert len(runnable.inputs) == 1


def test_structured_output_retry_budget_is_capped() -> None:
    with pytest.raises(IntentConfigurationError, match="structured_output_retry_budget"):
        GeminiIntentExtractor(api_key=None, structured_output_retry_budget=2)


@pytest.mark.parametrize("retry_budget", [-1, 4, True])
def test_retry_budget_must_be_small_nonnegative_integer(retry_budget: int) -> None:
    with pytest.raises(IntentConfigurationError, match="retry_budget"):
        GeminiIntentExtractor(api_key=None, retry_budget=retry_budget)


def test_empty_message_is_rejected_before_provider_call() -> None:
    runnable = FakeStructuredRunnable(
        payload_with_actions(
            {"intent": "clarification", "clarification_question": "Which date?"}
        )
    )
    extractor = GeminiIntentExtractor(api_key=None, structured_runnable=runnable)

    with pytest.raises(ValueError, match="message must not be blank"):
        extractor.extract(" ")
    assert runnable.inputs == []


def test_gemini_requires_key_without_an_injected_runnable() -> None:
    with pytest.raises(IntentConfigurationError, match="GOOGLE_API_KEY"):
        GeminiIntentExtractor(api_key=None)


def test_settings_load_google_api_key_as_secret(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "unit-test-key")

    settings = Settings()

    assert settings.google_api_key is not None
    assert settings.google_api_key.get_secret_value() == "unit-test-key"
    assert "unit-test-key" not in repr(settings)


def test_prior_conversation_state_is_passed_as_context_only() -> None:
    runnable = FakeStructuredRunnable(
        payload_with_actions(
            {
                "intent": "itinerary_modification",
                "itinerary_modification": {"action": "replace_stop", "stop_index": 1},
            }
        )
    )
    extractor = GeminiIntentExtractor(api_key=None, structured_runnable=runnable)

    extractor.extract("Replace the second stop.", ConversationState())

    human_message = runnable.inputs[0][1]
    assert "Prior conversation state (context only)" in human_message.content
    assert '"preferences"' in human_message.content


def test_dataset_metadata_vocabulary_is_explicitly_given_to_model() -> None:
    runnable = FakeStructuredRunnable(
        payload_with_actions({"intent": "poi_recommendation"})
    )
    extractor = GeminiIntentExtractor(
        api_key=None,
        structured_runnable=runnable,
        metadata_categories=("history", "nature"),
        metadata_tags=("museum", "venetian"),
    )

    extractor.extract("Recommend a history site")

    system_message = runnable.inputs[0][0]
    assert "Allowed dataset categories: history, nature." in system_message.content
    assert "Allowed dataset tags: museum, venetian." in system_message.content


@pytest.mark.parametrize(
    ("message", "categories", "tags", "payload", "expected"),
    [
        (
            "Recommend indoor museums for my daughter",
            ("food", "history", "local_market", "nature", "nightlife"),
            ("museum", "venetian", "culture"),
            {
                "intent": "poi_recommendation",
                "hard_filters": {"setting": "indoor", "tags": ["museum"]},
                "family_requirements": {"required": True},
            },
            {
                "setting": "indoor",
                "tags": ["museum"],
                "family_required": True,
            },
        ),
        (
            "Recommend indoor museums suitable for my 10-year-old daughter",
            ("food", "history", "local_market", "nature", "nightlife"),
            ("museum", "venetian", "culture"),
            {
                "intent": "poi_recommendation",
                "hard_filters": {"setting": "indoor", "tags": ["museum"]},
                "family_requirements": {"required": True, "child_ages": [10]},
            },
            {
                "setting": "indoor",
                "tags": ["museum"],
                "family_required": True,
                "child_ages": [10],
            },
        ),
        (
            "Can you recommend indoor museums in Corfu suitable for my 10-year-old daughter? "
            "I am particularly interested in local history.",
            ("food", "history", "local_market", "nature", "nightlife"),
            ("museum", "venetian", "culture"),
            {
                "intent": "poi_recommendation",
                "semantic_preferences": {"interests": ["local history"]},
                "hard_filters": {
                    "setting": "indoor",
                    "tags": ["museum"],
                    "category": None,
                },
                "family_requirements": {"required": True, "child_ages": [10]},
            },
            {
                "setting": "indoor",
                "tags": ["museum"],
                "category": None,
                "filter_setting": "set",
                "filter_tags": "set",
                "filter_category": "preserve",
                "semantic_interests": ["local history"],
                "family_required": True,
                "child_ages": [10],
            },
        ),
        (
            "I enjoy museums and Venetian architecture",
            ("history", "nature"),
            ("museum", "venetian"),
            {
                "intent": "poi_recommendation",
                "semantic_preferences": {"interests": ["Venetian architecture"]},
                "hard_filters": {"tags": ["museum"]},
            },
            {
                "tags": ["museum"],
                "semantic_interests": ["Venetian architecture"],
            },
        ),
        (
            "I enjoy history and I prefer historical places",
            ("history", "nature"),
            ("museum",),
            {
                "intent": "poi_recommendation",
                "semantic_preferences": {"interests": ["history", "historical places"]},
                "hard_filters": {"category": None},
            },
            {
                "category": None,
                "filter_category": "preserve",
                "semantic_interests": ["history", "historical places"],
            },
        ),
        (
            "Only outdoor historical places",
            ("food", "history", "nature"),
            ("museum", "venetian"),
            {
                "intent": "poi_recommendation",
                "hard_filters": {"setting": "outdoor", "category": "history"},
            },
            {"setting": "outdoor", "category": "history"            },
        ),
        (
            "Only show places in the history category",
            ("food", "history", "nature"),
            ("museum",),
            {
                "intent": "poi_recommendation",
                "hard_filters": {"category": "history"},
            },
            {
                "category": "history",
                "filter_category": "set",
            },
        ),
        (
            "Recommend indoor cultural attractions",
            ("food", "history", "nature"),
            ("museum", "culture"),
            {
                "intent": "poi_recommendation",
                "semantic_preferences": {"interests": ["cultural attractions"]},
                "hard_filters": {"setting": "indoor"},
            },
            {
                "setting": "indoor",
                "category": None,
                "tags": None,
                "semantic_interests": ["cultural attractions"],
            },
        ),
        (
            "Recommend indoor museums",
            (),
            (),
            {
                "intent": "poi_recommendation",
                "semantic_preferences": {"interests": ["museums"]},
                "hard_filters": {"setting": "indoor"},
            },
            {
                "setting": "indoor",
                "category": None,
                "tags": None,
                "filter_category": "preserve",
                "filter_tags": "preserve",
            },
        ),
    ],
)
def test_entity_types_and_soft_interests_map_to_consistent_filters(
    message: str,
    categories: tuple[str, ...],
    tags: tuple[str, ...],
    payload: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    runnable = FakeStructuredRunnable(payload_with_actions(payload))
    extractor = GeminiIntentExtractor(
        api_key=None,
        structured_runnable=runnable,
        metadata_categories=categories,
        metadata_tags=tags,
    )

    intent = extractor.extract(message)

    assert intent.intent == "poi_recommendation"
    for field, value in expected.items():
        if field in {"semantic_interests", "family_required", "child_ages"}:
            if field == "semantic_interests":
                assert intent.field_actions.semantic_interests == "set"
                assert intent.semantic_preferences.interests == value
            elif field == "child_ages":
                assert intent.field_actions.child_ages == "set"
                assert intent.family_requirements.child_ages == value
            else:
                assert intent.field_actions.family_required == "set"
                assert intent.family_requirements.required is value
        elif field.startswith("filter_"):
            assert getattr(intent.field_actions, field) == value
        else:
            assert getattr(intent.hard_filters, field) == value

    for metadata_field, action_field in (
        ("category", "filter_category"),
        ("tags", "filter_tags"),
        ("setting", "filter_setting"),
        ("family_suitability_fixture", "filter_family_suitability"),
    ):
        filter_value = getattr(intent.hard_filters, metadata_field)
        action = getattr(intent.field_actions, action_field)
        if action == "set":
            assert filter_value is not None
        else:
            assert filter_value is None

    system_message = runnable.inputs[0][0]
    if categories or tags:
        assert "Allowed dataset categories:" in system_message.content
        assert "Allowed dataset tags:" in system_message.content
        assert "Do not put \"museum\" in category" in system_message.content
        assert "Category values often describe themes" in system_message.content
        assert "particularly interested in local history" in system_message.content
    else:
        assert "No dataset category/tag vocabulary was supplied" in system_message.content
        assert "do not populate category or tag hard filters" in system_message.content


def test_october_6_user_clock_window_stays_local_and_rejects_zulu_values() -> None:
    valid = payload_with_actions(
        {
            "intent": "itinerary_creation",
            "requested_date": "2026-10-06",
            "time_window": {"start_time": "11:00", "end_time": "15:00"},
        }
    )
    intent = UserIntent.model_validate(valid)

    assert intent.requested_date == date(2026, 10, 6)
    assert intent.time_window.start_time == "11:00"
    assert intent.time_window.end_time == "15:00"
    assert intent.effective_timezone == "Europe/Athens"

    invalid = payload_with_actions(
        {
            "intent": "itinerary_creation",
            "requested_date": "2026-10-06",
            "time_window": {"start_time": "11:00:00Z", "end_time": "15:00:00Z"},
        }
    )
    with pytest.raises(ValidationError):
        UserIntent.model_validate(invalid)


def test_next_24_hour_weather_query_preserves_horizon_and_precipitation_focus() -> None:
    payload = payload_with_actions(
        {
            "intent": "live_weather_query",
            "weather_request": {
                "relative_horizon": "next_24_hours",
                "duration_hours": 24,
                "aspects": ["precipitation"],
            },
        }
    )
    runnable = FakeStructuredRunnable({"parsed": payload, "parsing_error": None, "raw": object()})
    extractor = GeminiIntentExtractor(api_key=None, structured_runnable=runnable)

    intent = extractor.extract(
        "What's the weather forecast in Corfu for the next 24 hours? Will it rain?"
    )

    assert intent.intent == "live_weather_query"
    assert intent.weather_request.relative_horizon == "next_24_hours"
    assert intent.weather_request.duration_hours == 24
    assert intent.weather_request.aspects == ["precipitation"]
    assert intent.time_window is None
    assert intent.requested_date is None
    assert intent.field_actions.start_time == "preserve"
    assert intent.field_actions.end_time == "preserve"


def test_tomorrow_weather_scope_does_not_infer_a_calendar_date() -> None:
    payload = payload_with_actions(
        {
            "intent": "live_weather_query",
            "weather_request": {"relative_horizon": "tomorrow"},
        }
    )
    extractor = GeminiIntentExtractor(
        api_key=None,
        structured_runnable=FakeStructuredRunnable(payload),
    )

    intent = extractor.extract("Weather tomorrow in Corfu")

    assert intent.weather_request.relative_horizon == "tomorrow"
    assert intent.weather_request.forecast_date is None
    assert intent.requested_date is None
    assert intent.field_actions.requested_date == "preserve"


def test_weather_activity_window_and_unsupported_actions_are_request_scoped() -> None:
    payload = payload_with_actions(
        {
            "intent": "live_weather_query",
            "weather_request": {
                "relative_horizon": "tomorrow",
                "start_time": "09:00",
                "end_time": "12:00",
                "activity_recommendation": True,
                "aspects": ["precipitation"],
            },
            "unsupported_actions": ["booking", "email"],
        }
    )
    runnable = FakeStructuredRunnable({"parsed": payload, "parsing_error": None, "raw": object()})
    extractor = GeminiIntentExtractor(api_key=None, structured_runnable=runnable)

    intent = extractor.extract(
        "Should I sightsee outdoors tomorrow between 09:00 and 12:00? Book a museum and email me."
    )

    assert intent.weather_request.start_time == "09:00"
    assert intent.weather_request.end_time == "12:00"
    assert intent.weather_request.activity_recommendation is True
    assert intent.time_window is None
    assert intent.field_actions.start_time == "preserve"
    assert intent.unsupported_actions == ["booking", "email"]
    system_message = runnable.inputs[0][0]
    assert "without an existing itinerary" in system_message.content
    assert "Include every such action requested" in system_message.content


def test_itinerary_request_keeps_local_clock_time_window() -> None:
    payload = payload_with_actions(
        {
            "intent": "itinerary_creation",
            "semantic_preferences": {"interests": ["history"]},
            "travel_mode": "walking",
            "time_window": {"start_time": "11:00", "end_time": "15:00"},
        }
    )
    extractor = GeminiIntentExtractor(
        api_key=None,
        structured_runnable=FakeStructuredRunnable(payload),
    )

    intent = extractor.extract("Plan a historical walk from 11:00 to 15:00")

    assert intent.time_window.start_time == "11:00"
    assert intent.time_window.end_time == "15:00"
    assert intent.effective_timezone == "Europe/Athens"
    assert intent.weather_request is None
    assert intent.field_actions.start_time == "set"
    assert intent.field_actions.end_time == "set"


def test_itinerary_weather_replanning_preserves_existing_time_window() -> None:
    payload = payload_with_actions(
        {
            "intent": "weather_based_replanning",
            "weather_request": {
                "forecast_date": "2026-10-07",
                "aspects": ["precipitation"],
            },
        }
    )
    extractor = GeminiIntentExtractor(
        api_key=None,
        structured_runnable=FakeStructuredRunnable(payload),
    )

    intent = extractor.extract(
        "Check the weather for my itinerary on October 7 and adjust it if needed."
    )

    assert intent.weather_request.forecast_date == date(2026, 10, 7)
    assert intent.weather_request.aspects == ["precipitation"]
    assert intent.time_window is None
    assert intent.field_actions.start_time == "preserve"
    assert intent.field_actions.end_time == "preserve"


def test_weather_request_rejects_inconsistent_duration_and_horizon() -> None:
    with pytest.raises(ValidationError):
        UserIntent(
            intent="live_weather_query",
            weather_request={
                "relative_horizon": "next_24_hours",
                "duration_hours": 12,
            },
            field_actions=intent_actions(weather_request="set"),
        )


def test_user_intent_json_schema_declares_hhmm_without_timezone_suffix() -> None:
    schema = UserIntent.model_json_schema()
    time_window_schema = schema["$defs"]["IntentTimeWindow"]
    start_schema = time_window_schema["properties"]["start_time"]
    action_schema = schema["$defs"]["IntentUpdateActions"]
    weather_schema = schema["$defs"]["WeatherRequestScope"]

    assert start_schema["anyOf"][0]["pattern"] == r"^(?:[01]\d|2[0-3]):[0-5]\d$"
    assert set(action_schema["required"]) == set(action_schema["properties"])
    assert weather_schema["properties"]["relative_horizon"]["anyOf"][0]["enum"] == [
        "next_24_hours",
        "tomorrow",
        "today",
        "tonight",
    ]


def test_user_intent_clear_action_supports_explicit_null() -> None:
    intent = UserIntent(
        intent="poi_recommendation",
        field_actions=intent_actions(timezone="clear"),
    )

    assert intent.timezone is None
    assert intent.effective_timezone == "Europe/Athens"
    assert intent.field_actions.timezone == "clear"


def test_model_timezone_is_discarded_when_user_did_not_request_one() -> None:
    payload = payload_with_actions({
        "intent": "itinerary_creation",
        "timezone": "Europe/Invalid-Corfu-Alias",
    })
    with pytest.raises(ValidationError, match="Unknown IANA timezone"):
        UserIntent.model_validate(payload)
    extractor = GeminiIntentExtractor(
        api_key=None, structured_runnable=FakeStructuredRunnable(payload)
    )

    result = extractor.extract("Plan a Corfu itinerary tomorrow")

    assert result.timezone is None
    assert result.field_actions.timezone == "preserve"
    assert result.effective_timezone == "Europe/Athens"


def test_user_intent_accepts_explicit_valid_iana_timezone() -> None:
    payload = payload_with_actions({
        "intent": "itinerary_creation", "timezone": "Europe/Athens",
    })
    payload["field_actions"]["timezone"] = "set"
    value = UserIntent.model_validate(payload)

    assert value.timezone == "Europe/Athens"
    assert value.field_actions.timezone == "set"


def test_unrequested_valid_model_timezone_with_preserve_action_is_repaired() -> None:
    payload = payload_with_actions({
        "intent": "itinerary_creation", "timezone": "Europe/Athens",
    })
    with pytest.raises(ValidationError, match="action 'preserve' must not include a value"):
        UserIntent.model_validate(payload)
    extractor = GeminiIntentExtractor(
        api_key=None, structured_runnable=FakeStructuredRunnable(payload)
    )

    result = extractor.extract("Plan a Corfu itinerary tomorrow")

    assert result.timezone is None
    assert result.field_actions.timezone == "preserve"


def test_explicit_destination_iana_timezone_is_applied_deterministically() -> None:
    payload = payload_with_actions({"intent": "itinerary_creation"})
    extractor = GeminiIntentExtractor(
        api_key=None, structured_runnable=FakeStructuredRunnable(payload)
    )

    result = extractor.extract("Plan the visit using Europe/Athens timezone")

    assert result.timezone == "Europe/Athens"
    assert result.field_actions.timezone == "set"


def test_explicit_conflicting_iana_timezone_requests_targeted_clarification() -> None:
    payload = payload_with_actions({"intent": "itinerary_creation"})
    extractor = GeminiIntentExtractor(
        api_key=None, structured_runnable=FakeStructuredRunnable(payload)
    )

    result = extractor.extract("Plan the Corfu visit using Europe/Paris timezone")

    assert result.intent == "clarification"
    assert "Europe/Paris" in result.clarification_question
    assert "Europe/Athens" in result.clarification_question
    assert result.field_actions.timezone == "preserve"


def test_explicit_malformed_iana_timezone_requests_targeted_clarification() -> None:
    payload = payload_with_actions({"intent": "itinerary_creation"})
    extractor = GeminiIntentExtractor(
        api_key=None, structured_runnable=FakeStructuredRunnable(payload)
    )

    result = extractor.extract("Plan the visit in Europe/Corfu timezone")

    assert result.intent == "clarification"
    assert "valid IANA timezone" in result.clarification_question
    assert result.field_actions.timezone == "preserve"


def test_local_corfu_time_is_resolved_without_model_timezone_authority() -> None:
    payload = payload_with_actions({
        "intent": "itinerary_creation", "timezone": "EET",
    })
    extractor = GeminiIntentExtractor(
        api_key=None, structured_runnable=FakeStructuredRunnable(payload)
    )

    result = extractor.extract("Use local Corfu time for this itinerary")

    assert result.timezone == "Europe/Athens"
    assert result.field_actions.timezone == "set"


def test_multi_turn_invented_model_timezone_preserves_conversation_timezone() -> None:
    manager = ConversationStateManager()
    state = manager.create_conversation()
    payload = payload_with_actions({
        "intent": "poi_recommendation", "timezone": "Invalid/Provider-Alias",
        "semantic_preferences": {"interests": ["museum"]},
    })
    extractor = GeminiIntentExtractor(
        api_key=None, structured_runnable=FakeStructuredRunnable(payload)
    )

    extracted = extractor.extract(
        "I am interested in museums too", conversation_state=state
    )
    updated = manager.apply_intent(
        state.conversation_id, extracted, message="I am interested in museums too"
    ).state

    assert extracted.field_actions.timezone == "preserve"
    assert updated.timezone == "Europe/Athens"
    assert updated.active_request.timezone == "Europe/Athens"


def test_user_intent_rejects_values_marked_preserve() -> None:
    with pytest.raises(ValidationError, match="action 'preserve' must not include a value"):
        UserIntent(
            intent="itinerary_creation",
            requested_date="2026-10-06",
            field_actions=intent_actions(requested_date="preserve"),
        )


def test_extractor_rejects_utc_marked_provider_clock_values_without_shifting() -> None:
    invalid_output = payload_with_actions(
        {
            "intent": "itinerary_creation",
            "requested_date": "2026-10-06",
            "time_window": {"start_time": "11:00:00Z", "end_time": "15:00:00Z"},
        }
    )
    extractor = GeminiIntentExtractor(
        api_key=None,
        structured_output_retry_budget=0,
        structured_runnable=FakeStructuredRunnable(invalid_output),
    )

    with pytest.raises(InvalidStructuredOutputError, match="does not match"):
        extractor.extract(
            "Plan a Corfu itinerary on October 6, 2026 from 11:00 to 15:00 local time."
        )


def test_langchain_google_model_accepts_typed_structured_output_without_network() -> None:
    extractor = GeminiIntentExtractor(
        api_key=SecretStr("test-only-not-a-real-key"),
        model_name="gemini-3.5-flash-lite",
    )

    assert extractor._structured_runnable is not None


def test_gemini_adapter_uses_current_provider_configuration(monkeypatch) -> None:
    import langchain_google_genai

    captured: dict[str, object] = {}

    class FakeChatModel:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def with_structured_output(
            self, schema: object, **kwargs: object
        ) -> FakeStructuredRunnable:
            captured["schema"] = schema
            captured["structured_options"] = kwargs
            return FakeStructuredRunnable(
                payload_with_actions({"intent": "live_weather_query"})
            )

    monkeypatch.setattr(langchain_google_genai, "ChatGoogleGenerativeAI", FakeChatModel)

    GeminiIntentExtractor(
        api_key=SecretStr("test-only-not-a-real-key"),
        timeout_seconds=17,
    )

    assert captured["api_key"] == "test-only-not-a-real-key"
    assert captured["request_timeout"] == 17
    assert captured["retries"] == 1
    assert captured["temperature"] == 0
    assert isinstance(captured["schema"], dict)
    expected_schema = UserIntent.model_json_schema()
    expected_schema["properties"]["constraint_evidence"]["anyOf"][0].pop("maxItems")
    expected_schema["properties"]["timezone"] = {
        "type": "null",
        "description": "Always null; the application resolves timezone deterministically.",
    }
    expected_schema["$defs"]["IntentUpdateActions"]["properties"]["timezone"] = {
        "type": "string", "enum": ["preserve"],
        "description": "Always preserve; the application resolves timezone deterministically.",
    }
    expected_schema["properties"]["semantic_preferences"]["description"] = (
        "Semantic preference update. Supply interests only when "
        "field_actions.semantic_interests is set; otherwise return null."
    )
    expected_schema["$defs"]["IntentUpdateActions"]["properties"][
        "semantic_interests"
    ]["description"] = (
        "Use set with a non-null semantic_preferences.interests list; "
        "use preserve or clear only when semantic_preferences is null."
    )
    assert captured["schema"] == expected_schema
    assert captured["structured_options"] == {
        "method": "json_schema",
        "include_raw": True,
    }


def test_provider_schema_omits_rejected_array_bound_but_local_validation_keeps_it() -> None:
    entries = [{
        "field": "filter_tags", "value": "art", "kind": "required", "quote": "Only art",
    }] * 51
    response = {"parsed": payload_with_actions({
        "intent": "tourism_qa", "constraint_evidence": entries,
    })}
    extractor = GeminiIntentExtractor(
        api_key=None, structured_output_retry_budget=0,
        structured_runnable=FakeStructuredRunnable(response),
    )

    with pytest.raises(InvalidStructuredOutputError) as failure:
        extractor.extract("Only art")
    assert any("constraint_evidence" in issue for issue in failure.value.validation_issues)
