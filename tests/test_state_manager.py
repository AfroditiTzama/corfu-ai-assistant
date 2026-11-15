from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from uuid import UUID

import pytest
from pydantic import ValidationError

from corfu_assistant.models import ConversationState, Itinerary, ItineraryStop, UserIntent
from corfu_assistant.state_manager import (
    ConcurrentConversationUpdateError,
    ConversationNotFoundError,
    ConversationStateManager,
    InvalidConversationUpdateError,
    NoPendingModificationError,
)


_INTENT_ACTIONS = (
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


def make_intent(
    intent: str = "poi_recommendation",
    *,
    values: dict[str, object] | None = None,
    actions: dict[str, str] | None = None,
) -> UserIntent:
    payload = dict(values or {})
    field_actions = dict.fromkeys(_INTENT_ACTIONS, "preserve")
    field_actions.update(actions or {})
    payload.update(intent=intent, field_actions=field_actions)
    return UserIntent.model_validate(payload)


def itinerary(conversation_id: UUID | None = None) -> Itinerary:
    return Itinerary(
        conversation_id=conversation_id,
        starts_at=datetime(2026, 10, 6, 9, tzinfo=timezone.utc),
        ends_at=datetime(2026, 10, 6, 15, tzinfo=timezone.utc),
        stops=[
            ItineraryStop(
                poi_id=f"CF-00{index}",
                starts_at=datetime(2026, 10, 6, 9 + index, tzinfo=timezone.utc),
                ends_at=datetime(2026, 10, 6, 9 + index, 30, tzinfo=timezone.utc),
            )
            for index in range(1, 4)
        ],
    )


def test_create_conversation_assigns_uuid_and_get_returns_isolated_copy() -> None:
    manager = ConversationStateManager()

    created = manager.create_conversation()
    fetched = manager.get(str(created.conversation_id))
    fetched.preferences.interests.append("mutated outside")

    assert isinstance(created.conversation_id, UUID)
    assert fetched.preferences.interests == ["mutated outside"]
    assert manager.get(created.conversation_id).preferences.interests == []


def test_two_turns_merge_explicit_updates_and_preserve_unmentioned_values() -> None:
    manager = ConversationStateManager()
    state = manager.create_conversation()

    first = make_intent(
        values={
            "semantic_preferences": {"interests": ["local history"]},
            "hard_filters": {"tags": ["museum"]},
        },
        actions={"semantic_interests": "set", "filter_tags": "set"},
    )
    manager.apply_intent(state.conversation_id, first, message="I like local history and museums")

    second = make_intent(
        values={"travel_mode": "walking"},
        actions={"travel_mode": "set"},
    )
    updated = manager.apply_intent(
        state.conversation_id,
        second,
        message="I prefer to walk",
    ).state

    assert updated.preferences.interests == ["local history"]
    assert updated.hard_filters.tags == ["museum"]
    assert updated.travel_mode == "walking"
    assert updated.updated_at >= state.updated_at


def test_set_clear_preserve_and_explicit_empty_lists() -> None:
    manager = ConversationStateManager()
    state = manager.create_conversation()
    manager.apply_intent(
        state.conversation_id,
        make_intent(
            values={
                "semantic_preferences": {"interests": ["art"]},
                "hard_filters": {"category": "history", "tags": ["museum"]},
            },
            actions={
                "semantic_interests": "set",
                "filter_category": "set",
                "filter_tags": "set",
            },
        ),
        message="Art and museums",
    )

    cleared = manager.apply_intent(
        state.conversation_id,
        make_intent(
            values={
                "semantic_preferences": {"interests": []},
                "hard_filters": {"tags": []},
            },
            actions={"semantic_interests": "set", "filter_tags": "set"},
        ),
        message="Clear my interests and museum filter",
    ).state
    assert cleared.preferences.interests == []
    assert cleared.hard_filters.tags == []
    assert cleared.hard_filters.category == "history"

    cleared = manager.apply_intent(
        state.conversation_id,
        make_intent(
            actions={
                "filter_category": "clear",
                "filter_tags": "clear",
                "semantic_interests": "clear",
            }
        ),
        message="Remove those filters",
    ).state
    assert cleared.hard_filters.category is None
    assert cleared.hard_filters.tags is None
    assert cleared.preferences.interests == []


def test_exclusion_filters_merge_preserve_and_clear_across_turns() -> None:
    manager = ConversationStateManager()
    state = manager.create_conversation()
    manager.apply_intent(
        state.conversation_id,
        make_intent(
            values={
                "hard_filters": {
                    "excluded_tags": ["museum"],
                    "excluded_poi_ids": ["CF-022"],
                }
            },
            actions={
                "filter_excluded_tags": "set",
                "filter_excluded_poi_ids": "set",
            },
        ),
        message="Exclude museums and the Mon Repos museum",
    )

    preserved = manager.apply_intent(
        state.conversation_id,
        make_intent(values={"travel_mode": "walking"}, actions={"travel_mode": "set"}),
        message="I prefer walking",
    ).state
    assert preserved.hard_filters.excluded_tags == ["museum"]
    assert preserved.hard_filters.excluded_poi_ids == ["CF-022"]

    cleared = manager.apply_intent(
        state.conversation_id,
        make_intent(
            actions={
                "filter_excluded_tags": "clear",
                "filter_excluded_poi_ids": "clear",
            }
        ),
        message="Remove those exclusions",
    ).state
    assert cleared.hard_filters.excluded_tags is None
    assert cleared.hard_filters.excluded_poi_ids is None


def test_partial_local_time_updates_preserve_other_boundary() -> None:
    manager = ConversationStateManager()
    state = manager.create_conversation()
    first = manager.apply_intent(
        state.conversation_id,
        make_intent(
            "itinerary_creation",
            values={
                "time_window": {"start_time": "11:00"},
                "requested_date": "2026-10-06",
            },
            actions={"start_time": "set", "requested_date": "set"},
        ),
        message="Start at 11 on October 6",
    ).state
    assert first.active_request.start_time == "11:00"
    assert first.active_request.end_time is None
    assert first.active_request.timezone == "Europe/Athens"

    second = manager.apply_intent(
        state.conversation_id,
        make_intent(
            values={"time_window": {"end_time": "15:00"}},
            actions={"end_time": "set"},
        ),
        message="Finish at 15:00",
    ).state
    assert second.active_request.start_time == "11:00"
    assert second.active_request.end_time == "15:00"
    assert second.active_request.requested_date.isoformat() == "2026-10-06"

    third = manager.apply_intent(
        state.conversation_id,
        make_intent(
            values={"time_window": {"start_time": "12:00"}},
            actions={"start_time": "set"},
        ),
        message="Start later, at noon",
    ).state
    assert third.active_request.start_time == "12:00"
    assert third.active_request.end_time == "15:00"


def test_clear_resets_optional_time_date_and_timezone_values() -> None:
    manager = ConversationStateManager()
    state = manager.create_conversation()
    manager.apply_intent(
        state.conversation_id,
        make_intent(
            values={
                "requested_date": "2026-10-06",
                "time_window": {"start_time": "11:00", "end_time": "15:00"},
                "timezone": "Europe/Paris",
            },
            actions={
                "requested_date": "set",
                "start_time": "set",
                "end_time": "set",
                "timezone": "set",
            },
        ),
        message="Plan a Paris-time visit",
    )
    updated = manager.apply_intent(
        state.conversation_id,
        make_intent(
            actions={
                "requested_date": "clear",
                "start_time": "clear",
                "timezone": "clear",
            }
        ),
        message="Clear the date and start time",
    ).state

    assert updated.timezone == "Europe/Athens"
    assert updated.active_request.timezone == "Europe/Athens"
    assert updated.active_request.requested_date is None
    assert updated.active_request.start_time is None
    assert updated.active_request.end_time == "15:00"


def test_inconsistent_merged_time_window_is_rejected_atomically() -> None:
    manager = ConversationStateManager()
    state = manager.create_conversation()
    manager.apply_intent(
        state.conversation_id,
        make_intent(
            values={"time_window": {"start_time": "11:00", "end_time": "15:00"}},
            actions={"start_time": "set", "end_time": "set"},
        ),
        message="Plan from 11 to 15",
    )
    before = manager.get(state.conversation_id)

    with pytest.raises(InvalidConversationUpdateError, match="end_time must be later"):
        manager.apply_intent(
            state.conversation_id,
            make_intent(
                values={"time_window": {"start_time": "16:00"}},
                actions={"start_time": "set"},
            ),
            message="Start at 16",
        )

    after = manager.get(state.conversation_id)
    assert after.active_request.start_time == "11:00"
    assert after.active_request.end_time == "15:00"
    assert after.updated_at == before.updated_at


def test_transient_question_weather_and_clarification_do_not_replace_preferences() -> None:
    manager = ConversationStateManager()
    state = manager.create_conversation()
    manager.apply_intent(
        state.conversation_id,
        make_intent(
            values={"semantic_preferences": {"interests": ["architecture"]}},
            actions={"semantic_interests": "set"},
        ),
        message="I like architecture",
    )
    existing_itinerary = manager.update_itinerary(state.conversation_id, itinerary())

    qa = manager.apply_intent(
        state.conversation_id,
        make_intent(
            "tourism_qa",
            values={"question": "What is the Old Fortress?"},
            actions={"question": "set"},
        ),
        message="What is the Old Fortress?",
    )
    assert qa.question == "What is the Old Fortress?"
    assert qa.state.preferences.interests == ["architecture"]
    assert qa.state.active_request.message == "I like architecture"
    assert qa.state.itinerary == existing_itinerary.itinerary

    weather = manager.apply_intent(
        state.conversation_id,
        make_intent(
            "live_weather_query",
            values={"weather_request": {"relative_horizon": "tomorrow"}},
            actions={"weather_request": "set"},
        ),
        message="Weather tomorrow",
    )
    assert weather.weather_request.relative_horizon == "tomorrow"
    assert weather.state.preferences.interests == ["architecture"]
    assert weather.state.active_request.message == "I like architecture"
    assert weather.state.itinerary == existing_itinerary.itinerary

    clarification = manager.apply_intent(
        state.conversation_id,
        make_intent(
            "clarification",
            values={"clarification_question": "Which date do you mean?"},
            actions={"clarification_question": "set"},
        ),
        message="Plan a visit",
    )
    assert clarification.clarification_question == "Which date do you mean?"
    assert clarification.state.preferences.interests == ["architecture"]
    assert clarification.state.itinerary == existing_itinerary.itinerary


def test_weather_replanning_preserves_active_itinerary_and_planning_window() -> None:
    manager = ConversationStateManager()
    state = manager.create_conversation()
    manager.apply_intent(
        state.conversation_id,
        make_intent(
            "itinerary_creation",
            values={
                "requested_date": "2026-10-07",
                "time_window": {"start_time": "13:00", "end_time": "17:00"},
            },
            actions={"requested_date": "set", "start_time": "set", "end_time": "set"},
        ),
        message="Plan from 13:00 to 17:00",
    )
    current_itinerary = manager.update_itinerary(state.conversation_id, itinerary())
    replanning = manager.apply_intent(
        state.conversation_id,
        make_intent(
            "weather_based_replanning",
            values={
                "weather_request": {
                    "forecast_date": "2026-10-07",
                    "aspects": ["precipitation"],
                }
            },
            actions={"weather_request": "set"},
        ),
        message="Replan if it rains",
    )

    assert replanning.weather_request.forecast_date.isoformat() == "2026-10-07"
    assert replanning.state.active_request.start_time == "13:00"
    assert replanning.state.active_request.end_time == "17:00"
    assert replanning.state.itinerary == current_itinerary.itinerary


def test_replacing_second_stop_records_request_without_mutating_itinerary() -> None:
    manager = ConversationStateManager()
    state = manager.create_conversation()
    stored_itinerary = manager.update_itinerary(state.conversation_id, itinerary())
    before = stored_itinerary.itinerary.model_copy(deep=True)

    result = manager.apply_intent(
        state.conversation_id,
        make_intent(
            "itinerary_modification",
            values={
                "itinerary_modification": {
                    "action": "replace_stop",
                    "stop_index": 1,
                    "preserve_other_stops": True,
                }
            },
            actions={"itinerary_modification": "set"},
        ),
        message="Replace the second stop",
    )

    assert result.state.pending_modification.stop_index == 1
    assert result.state.pending_modification.action == "replace_stop"
    assert result.state.itinerary == before
    assert manager.get(state.conversation_id).itinerary == before


def test_replacement_family_requirements_are_scoped_not_global() -> None:
    manager = ConversationStateManager()
    state = manager.create_conversation()
    manager.apply_intent(
        state.conversation_id,
        make_intent(
            values={"semantic_preferences": {"interests": ["history"]}},
            actions={"semantic_interests": "set"},
        ),
        message="Plan a historical walk",
    )
    state = manager.get(state.conversation_id)

    updated = manager.apply_intent(
        state.conversation_id,
        make_intent(
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
        ),
        message="Replace the second stop for my 10-year-old",
    ).state

    assert updated.preferences.family_required is None
    assert updated.preferences.child_ages == []
    assert updated.replacement_family_requirements.required is True
    assert updated.replacement_family_requirements.child_ages == [10]
    assert updated.itinerary is None


def test_modification_stays_pending_until_planner_explicitly_resolves_it() -> None:
    manager = ConversationStateManager()
    state = manager.create_conversation()
    original_itinerary = manager.update_itinerary(state.conversation_id, itinerary())
    manager.apply_intent(
        state.conversation_id,
        make_intent(
            "itinerary_modification",
            values={
                "itinerary_modification": {
                    "action": "replace_stop",
                    "stop_index": 1,
                    "preserve_other_stops": True,
                }
            },
            actions={"itinerary_modification": "set"},
        ),
        message="Replace the second stop",
    )

    pending = manager.get(state.conversation_id)
    assert pending.pending_modification.stop_index == 1
    assert pending.itinerary == original_itinerary.itinerary
    with pytest.raises(InvalidConversationUpdateError, match="requires the planner"):
        manager.resolve_pending_modification(state.conversation_id, outcome="applied")

    rejected = manager.resolve_pending_modification(
        state.conversation_id,
        outcome="rejected",
    )
    assert rejected.pending_modification is None
    assert rejected.last_modification_outcome == "rejected"
    assert rejected.itinerary == original_itinerary.itinerary
    with pytest.raises(NoPendingModificationError):
        manager.resolve_pending_modification(state.conversation_id, outcome="rejected")


def test_planner_can_apply_modification_only_with_resulting_itinerary() -> None:
    manager = ConversationStateManager()
    state = manager.create_conversation()
    old_itinerary = manager.update_itinerary(state.conversation_id, itinerary())
    manager.apply_intent(
        state.conversation_id,
        make_intent(
            "itinerary_modification",
            values={
                "itinerary_modification": {
                    "action": "replace_stop",
                    "stop_index": 1,
                    "preserve_other_stops": True,
                }
            },
            actions={"itinerary_modification": "set"},
        ),
        message="Replace the second stop",
    )
    replacement = itinerary()
    replacement.stops[1].poi_id = "CF-050"

    applied = manager.resolve_pending_modification(
        state.conversation_id,
        outcome="applied",
        itinerary=replacement,
    )

    assert applied.pending_modification is None
    assert applied.last_modification_outcome == "applied"
    assert applied.itinerary.stops[1].poi_id == "CF-050"
    assert old_itinerary.itinerary.stops[1].poi_id != "CF-050"


def test_conversations_are_isolated() -> None:
    manager = ConversationStateManager()
    first = manager.create_conversation()
    second = manager.create_conversation()
    manager.apply_intent(
        first.conversation_id,
        make_intent(
            values={"semantic_preferences": {"interests": ["art"]}},
            actions={"semantic_interests": "set"},
        ),
        message="Art please",
    )

    assert manager.get(first.conversation_id).preferences.interests == ["art"]
    assert manager.get(second.conversation_id).preferences.interests == []


def test_invalid_action_value_combination_is_rejected_without_state_change() -> None:
    manager = ConversationStateManager()
    state = manager.create_conversation()
    malformed = make_intent(
        values={"semantic_preferences": {"interests": ["art"]}},
        actions={"semantic_interests": "set"},
    )
    malformed.field_actions.semantic_interests = "preserve"

    with pytest.raises(InvalidConversationUpdateError, match="Intent is invalid"):
        manager.apply_intent(state.conversation_id, malformed, message="Art")
    assert manager.get(state.conversation_id).preferences.interests == []

    with pytest.raises(ValidationError):
        make_intent(actions={"semantic_interests": "set"})


def test_unknown_ids_reset_and_delete_are_explicit() -> None:
    manager = ConversationStateManager()
    state = manager.create_conversation()

    with pytest.raises(ConversationNotFoundError):
        manager.get(UUID(int=999))
    with pytest.raises(ConversationNotFoundError):
        manager.apply_intent(
            UUID(int=999),
            make_intent(),
            message="Anything",
        )

    updated = manager.apply_intent(
        state.conversation_id,
        make_intent(
            values={"semantic_preferences": {"interests": ["history"]}},
            actions={"semantic_interests": "set"},
        ),
        message="History",
    ).state
    reset = manager.reset(state.conversation_id)
    assert reset.conversation_id == state.conversation_id
    assert reset.preferences.interests == []
    assert reset.itinerary is None
    assert reset.updated_at >= updated.updated_at

    assert manager.delete(state.conversation_id) is True
    assert manager.delete(state.conversation_id) is False
    with pytest.raises(ConversationNotFoundError):
        manager.get(state.conversation_id)


def test_concurrent_updates_do_not_lose_independent_fields() -> None:
    manager = ConversationStateManager()
    state = manager.create_conversation()
    interest_intent = make_intent(
        values={"semantic_preferences": {"interests": ["history"]}},
        actions={"semantic_interests": "set"},
    )
    setting_intent = make_intent(
        values={"hard_filters": {"setting": "indoor"}},
        actions={"filter_setting": "set"},
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                manager.apply_intent,
                state.conversation_id,
                interest_intent,
                message="History",
            ),
            executor.submit(
                manager.apply_intent,
                state.conversation_id,
                setting_intent,
                message="Indoor",
            ),
        ]
        [future.result() for future in futures]

    final = manager.get(state.conversation_id)
    assert final.preferences.interests == ["history"]
    assert final.hard_filters.setting == "indoor"
    assert final.preferences.preferred_setting == "indoor"


def test_planned_itinerary_and_pending_resolution_commit_atomically() -> None:
    manager = ConversationStateManager()
    state = manager.create_conversation()
    old = manager.update_itinerary(state.conversation_id, itinerary())
    pending = manager.apply_intent(
        state.conversation_id,
        make_intent(
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
        ),
        message="Replace the second stop for a child",
    ).state
    revised = itinerary()
    revised.stops[1].poi_id = "CF-050"

    committed = manager.commit_planned_itinerary(
        state.conversation_id,
        revised,
        expected_state=pending,
        expected_pending_modification=pending.pending_modification,
    )

    assert committed.itinerary.stops[1].poi_id == "CF-050"
    assert committed.pending_modification is None
    assert committed.replacement_family_requirements is None
    assert committed.last_modification_outcome == "applied"
    assert old.itinerary.stops[1].poi_id != committed.itinerary.stops[1].poi_id


def test_planned_commit_requires_the_pending_modification_from_expected_state() -> None:
    manager = ConversationStateManager()
    state = manager.create_conversation()
    pending = manager.apply_intent(
        state.conversation_id,
        make_intent(
            "itinerary_modification",
            values={"itinerary_modification": {
                "action": "change_constraints",
                "preserve_other_stops": True,
            }},
            actions={"itinerary_modification": "set"},
        ),
        message="Change the itinerary constraints",
    ).state

    with pytest.raises(
        ConcurrentConversationUpdateError,
        match="Pending modification changed while the itinerary was being planned",
    ):
        manager.commit_planned_itinerary(
            state.conversation_id,
            itinerary(),
            expected_state=pending,
        )


def test_concurrent_plan_commits_use_compare_and_swap() -> None:
    manager = ConversationStateManager()
    state = manager.create_conversation()
    first_plan = itinerary()
    second_plan = itinerary()
    second_plan.stops[1].poi_id = "CF-050"

    def commit(plan: Itinerary) -> str:
        try:
            manager.commit_planned_itinerary(
                state.conversation_id,
                plan,
                expected_state=state,
            )
        except ConcurrentConversationUpdateError:
            return "conflict"
        return "committed"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(commit, [first_plan, second_plan]))

    assert sorted(outcomes) == ["committed", "conflict"]
