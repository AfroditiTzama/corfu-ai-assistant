from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from corfu_assistant.models import (
    ConversationState,
    Itinerary,
    ItineraryStop,
    ItineraryValidationResult,
    OpeningWindow,
    StatePatch,
    TravelPreferences,
    TravelPreferencesPatch,
    UserRequest,
    ValidationIssue,
)


def state_actions(**updates: str) -> dict[str, str]:
    fields = (
        "requested_date",
        "start_time",
        "end_time",
        "timezone",
        "interests",
        "excluded_categories",
        "excluded_tags",
        "preferred_setting",
        "family_required",
        "child_ages",
        "replace_stop",
    )
    actions = dict.fromkeys(fields, "preserve")
    actions.update(updates)
    return actions


def test_opening_window_parses_overnight_hours() -> None:
    window = OpeningWindow.model_validate("20:00-02:00")

    assert window.start_minute == 1200
    assert window.end_minute == 1560
    assert window.crosses_midnight


def test_opening_window_parses_end_of_day_fixture_value() -> None:
    window = OpeningWindow.model_validate("00:00-24:00")

    assert window.start_minute == 0
    assert window.end_minute == 1440
    assert not window.crosses_midnight


def test_opening_window_ending_at_midnight_is_overnight() -> None:
    window = OpeningWindow.model_validate("23:00-00:00")

    assert window.end_minute == 1440
    assert window.crosses_midnight


def test_opening_window_accepts_valid_dictionary_input() -> None:
    window = OpeningWindow.model_validate({"start_minute": 1200, "end_minute": 1560})

    assert window.start_minute == 1200
    assert window.end_minute == 1560
    assert window.crosses_midnight


@pytest.mark.parametrize(
    "value",
    [
        {"start_minute": 60, "end_minute": 60},
        {"start_minute": 120, "end_minute": 60},
        {"start_minute": 0, "end_minute": 1441},
        {"start_minute": 1440, "end_minute": 1500},
    ],
)
def test_opening_window_rejects_invalid_dictionary_input(value: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        OpeningWindow.model_validate(value)


@pytest.mark.parametrize("value", ["24:00-02:00", "09:00-24:30", "09:00-17:00:00", "09:00-09:00"])
def test_rejects_invalid_opening_window(value: str) -> None:
    with pytest.raises(ValidationError):
        OpeningWindow.model_validate(value)


def test_user_request_rejects_reversed_time_window() -> None:
    with pytest.raises(ValidationError, match="end_time must be later"):
        UserRequest(message="Plan a visit", start_time="17:00", end_time="13:00")


@pytest.mark.parametrize("value", ["11:00:00Z", "11:00Z", "11:00+02:00", "24:00", "9:00"])
def test_user_clock_fields_reject_utc_offsets_and_non_hhmm_values(value: str) -> None:
    with pytest.raises(ValidationError):
        UserRequest(message="Plan a visit", start_time=value)


def test_state_patch_supports_explicit_clear_and_set_empty_list() -> None:
    clear_timezone = StatePatch.model_validate(
        {
            "timezone": None,
            "actions": state_actions(timezone="clear"),
        }
    )
    clear_interests = StatePatch.model_validate(
        {
            "preferences": {"interests": []},
            "actions": state_actions(interests="set"),
        }
    )

    assert clear_timezone.actions.timezone == "clear"
    assert clear_interests.preferences.interests == []
    assert clear_interests.actions.interests == "set"


def test_state_patch_preserve_does_not_infer_from_populated_value() -> None:
    with pytest.raises(ValidationError, match="action 'preserve' must not include a value"):
        StatePatch.model_validate(
            {
                "requested_date": "2026-10-06",
                "actions": state_actions(requested_date="preserve"),
            }
        )


def test_state_patch_can_explicitly_clear_nested_nullable_preference() -> None:
    patch = StatePatch.model_validate(
        {
            "preferences": {"preferred_setting": None},
            "actions": state_actions(preferred_setting="clear"),
        }
    )

    assert patch.actions.preferred_setting == "clear"
    assert patch.preferences.preferred_setting is None


def test_state_patch_rejects_reversed_local_time_window() -> None:
    with pytest.raises(ValidationError, match="end_time must be later"):
        StatePatch(start_time="15:00", end_time="11:00")


def test_conversation_state_has_stable_identifier_and_timezone() -> None:
    state = ConversationState()

    assert state.conversation_id is not None
    assert state.timezone == "Europe/Athens"
    assert state.updated_at.tzinfo is not None


def test_travel_preferences_are_typed_and_validate_child_ages() -> None:
    preferences = TravelPreferences(
        interests=["history"],
        preferred_setting="outdoor",
        family_required=True,
        child_ages=[10],
    )

    assert preferences.interests == ["history"]
    assert preferences.preferred_setting == "outdoor"
    assert preferences.family_required is True
    assert preferences.child_ages == [10]

    with pytest.raises(ValidationError):
        TravelPreferences(preferred_setting="covered", child_ages=[18])


def test_state_patch_validates_typed_preference_updates() -> None:
    patch = StatePatch.model_validate(
        {
            "preferences": {
                "interests": ["art"],
                "preferred_setting": "indoor",
                "child_ages": [10],
            },
            "replace_stop": {"stop_index": 1, "preserve_other_stops": True},
        }
    )

    assert isinstance(patch.preferences, TravelPreferencesPatch)
    assert patch.preferences.interests == ["art"]
    assert patch.preferences.preferred_setting == "indoor"
    assert patch.replace_stop is not None and patch.replace_stop.stop_index == 1

    with pytest.raises(ValidationError):
        StatePatch.model_validate({"preferences": {"preferred_setting": "covered"}})
    with pytest.raises(ValidationError):
        StatePatch.model_validate({"replace_stop": {"stop_index": -1}})


def test_itinerary_validates_timezone_aware_non_overlapping_stops() -> None:
    start = datetime(2026, 10, 7, 13, tzinfo=timezone.utc)
    first_stop = ItineraryStop(
        poi_id="CF-001",
        starts_at=datetime(2026, 10, 7, 13, 30, tzinfo=timezone.utc),
        ends_at=datetime(2026, 10, 7, 14, 30, tzinfo=timezone.utc),
    )
    itinerary = Itinerary(
        starts_at=start,
        ends_at=datetime(2026, 10, 7, 16, tzinfo=timezone.utc),
        stops=[first_stop],
    )

    assert itinerary.stops == [first_stop]


def test_itinerary_rejects_overlapping_stops() -> None:
    start = datetime(2026, 10, 7, 13, tzinfo=timezone.utc)
    first_stop = ItineraryStop(
        poi_id="CF-001",
        starts_at=datetime(2026, 10, 7, 13, 30, tzinfo=timezone.utc),
        ends_at=datetime(2026, 10, 7, 14, 30, tzinfo=timezone.utc),
    )
    second_stop = ItineraryStop(
        poi_id="CF-002",
        starts_at=datetime(2026, 10, 7, 14, tzinfo=timezone.utc),
        ends_at=datetime(2026, 10, 7, 15, tzinfo=timezone.utc),
    )

    with pytest.raises(ValidationError, match="chronological and non-overlapping"):
        Itinerary(
            starts_at=start,
            ends_at=datetime(2026, 10, 7, 16, tzinfo=timezone.utc),
            stops=[first_stop, second_stop],
        )


def test_validation_result_carries_structured_issues() -> None:
    result = ItineraryValidationResult(
        is_valid=False,
        issues=[ValidationIssue(code="closed", message="Fixture marks the site closed", poi_id="CF-002")],
    )

    assert result.issues[0].poi_id == "CF-002"
