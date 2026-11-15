from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from typing import Callable, Literal
from uuid import UUID

from pydantic import ValidationError

from corfu_assistant.models import (
    ConversationState,
    Itinerary,
    ItineraryModification,
    UserIntent,
    WeatherRequestScope,
)


class ConversationNotFoundError(KeyError):
    """Raised when an operation references an unknown conversation ID."""


class InvalidConversationUpdateError(ValueError):
    """Raised when an intent cannot be applied without violating state invariants."""


class NoPendingModificationError(ValueError):
    """Raised when a planner attempts to resolve a missing modification request."""


class ConcurrentConversationUpdateError(InvalidConversationUpdateError):
    """Raised when state changed after a plan was calculated from its snapshot."""


@dataclass(frozen=True)
class ConversationTurn:
    """Persistent state plus request-scoped values that are not conversation memory."""

    state: ConversationState
    question: str | None = None
    clarification_question: str | None = None
    weather_request: WeatherRequestScope | None = None


class ConversationStateManager:
    """Thread-safe in-memory conversation store with explicit intent merging.

    State is process-local and is lost on restart. Reads and returned turn values
    are deep copies so callers cannot mutate the manager's stored models.
    """

    def __init__(self, *, clock: Callable[[], datetime] | None = None) -> None:
        self._conversations: dict[UUID, ConversationState] = {}
        self._lock = RLock()
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def create_conversation(self) -> ConversationState:
        with self._lock:
            state = ConversationState(updated_at=self._now())
            self._conversations[state.conversation_id] = state
            return state.model_copy(deep=True)

    def get(self, conversation_id: UUID | str) -> ConversationState:
        identifier = self._normalize_id(conversation_id)
        with self._lock:
            state = self._conversations.get(identifier)
            if state is None:
                raise ConversationNotFoundError(str(identifier))
            return state.model_copy(deep=True)

    def apply_intent(
        self,
        conversation_id: UUID | str,
        intent: UserIntent,
        *,
        message: str,
    ) -> ConversationTurn:
        identifier = self._normalize_id(conversation_id)
        normalized_message = message.strip()
        if not normalized_message:
            raise ValueError("message must not be blank")
        if not isinstance(intent, UserIntent):
            raise InvalidConversationUpdateError("Intent must be a validated UserIntent")

        try:
            validated_intent = UserIntent.model_validate(intent.model_dump(mode="python"))
        except (ValidationError, TypeError, ValueError) as error:
            raise InvalidConversationUpdateError(
                "Intent is invalid and cannot be applied to conversation state"
            ) from error

        with self._lock:
            current = self._conversations.get(identifier)
            if current is None:
                raise ConversationNotFoundError(str(identifier))
            try:
                updated = self._merge(current, validated_intent, normalized_message)
            except (ValidationError, TypeError, ValueError) as error:
                raise InvalidConversationUpdateError(
                    f"Intent update would violate conversation state invariants: {error}"
                ) from error
            self._conversations[identifier] = updated
            return ConversationTurn(
                state=updated.model_copy(deep=True),
                question=(
                    validated_intent.question
                    if validated_intent.field_actions.question == "set"
                    else None
                ),
                clarification_question=(
                    validated_intent.clarification_question
                    if validated_intent.field_actions.clarification_question == "set"
                    else None
                ),
                weather_request=(
                    validated_intent.weather_request
                    if validated_intent.field_actions.weather_request == "set"
                    else None
                ),
            )

    def rollback_failed_turn(
        self, expected_state: ConversationState, previous_state: ConversationState | None
    ) -> bool:
        """Undo an uncommitted workflow failure without overwriting another writer."""
        identifier = expected_state.conversation_id
        if previous_state is not None and previous_state.conversation_id != identifier:
            raise ValueError("Rollback snapshots must refer to the same conversation")
        with self._lock:
            if self._conversations.get(identifier) != expected_state:
                return False
            if previous_state is None:
                del self._conversations[identifier]
            else:
                self._conversations[identifier] = previous_state.model_copy(deep=True)
            return True

    def update_itinerary(
        self,
        conversation_id: UUID | str,
        itinerary: Itinerary,
    ) -> ConversationState:
        """Store an itinerary supplied by a future planner without exposing it mutably."""
        identifier = self._normalize_id(conversation_id)
        with self._lock:
            current = self._conversations.get(identifier)
            if current is None:
                raise ConversationNotFoundError(str(identifier))
            itinerary_data = itinerary.model_dump(mode="python")
            if itinerary.conversation_id not in (None, identifier):
                raise InvalidConversationUpdateError(
                    "Itinerary conversation_id does not match the target conversation"
                )
            itinerary_data["conversation_id"] = identifier
            candidate_data = current.model_dump(mode="python")
            candidate_data["itinerary"] = itinerary_data
            candidate_data["updated_at"] = self._now()
            updated = ConversationState.model_validate(candidate_data)
            self._conversations[identifier] = updated
            return updated.model_copy(deep=True)

    def commit_planned_itinerary(
        self,
        conversation_id: UUID | str,
        itinerary: Itinerary,
        *,
        expected_state: ConversationState,
        expected_pending_modification: ItineraryModification | None = None,
    ) -> ConversationState:
        """Atomically persist an accepted plan and, when applicable, resolve its action."""
        identifier = self._normalize_id(conversation_id)
        with self._lock:
            current = self._conversations.get(identifier)
            if current is None:
                raise ConversationNotFoundError(str(identifier))
            if current != expected_state:
                raise ConcurrentConversationUpdateError(
                    "Conversation changed while the itinerary was being planned"
                )
            if current.pending_modification != expected_pending_modification:
                raise ConcurrentConversationUpdateError(
                    "Pending modification changed while the itinerary was being planned"
                )
            if itinerary.conversation_id not in (None, identifier):
                raise InvalidConversationUpdateError(
                    "Itinerary conversation_id does not match the target conversation"
                )

            itinerary_data = itinerary.model_dump(mode="python")
            itinerary_data["conversation_id"] = identifier
            data = current.model_dump(mode="python")
            data["itinerary"] = itinerary_data
            if expected_pending_modification is not None:
                data["pending_modification"] = None
                data["replacement_family_requirements"] = None
                data["last_modification_outcome"] = "applied"
            data["updated_at"] = self._now()
            updated = ConversationState.model_validate(data)
            self._conversations[identifier] = updated
            return updated.model_copy(deep=True)

    def resolve_pending_modification(
        self,
        conversation_id: UUID | str,
        *,
        outcome: Literal["applied", "rejected"],
        itinerary: Itinerary | None = None,
    ) -> ConversationState:
        """Resolve a pending request only when a planner explicitly reports an outcome.

        An applied modification must supply the planner's resulting itinerary;
        rejection leaves the stored itinerary untouched.
        """
        identifier = self._normalize_id(conversation_id)
        if outcome == "applied" and itinerary is None:
            raise InvalidConversationUpdateError(
                "An applied modification requires the planner's resulting itinerary"
            )
        if outcome == "rejected" and itinerary is not None:
            raise InvalidConversationUpdateError(
                "A rejected modification must not replace the stored itinerary"
            )

        with self._lock:
            current = self._conversations.get(identifier)
            if current is None:
                raise ConversationNotFoundError(str(identifier))
            if current.pending_modification is None:
                raise NoPendingModificationError(
                    f"Conversation {identifier} has no pending modification"
                )

            data = current.model_dump(mode="python")
            data["pending_modification"] = None
            data["replacement_family_requirements"] = None
            data["last_modification_outcome"] = outcome
            if itinerary is not None:
                itinerary_data = itinerary.model_dump(mode="python")
                if itinerary.conversation_id not in (None, identifier):
                    raise InvalidConversationUpdateError(
                        "Itinerary conversation_id does not match the target conversation"
                    )
                itinerary_data["conversation_id"] = identifier
                data["itinerary"] = itinerary_data
            data["updated_at"] = self._now()
            updated = ConversationState.model_validate(data)
            self._conversations[identifier] = updated
            return updated.model_copy(deep=True)

    def reset(self, conversation_id: UUID | str) -> ConversationState:
        """Clear all conversation data while retaining the same conversation ID."""
        identifier = self._normalize_id(conversation_id)
        with self._lock:
            if identifier not in self._conversations:
                raise ConversationNotFoundError(str(identifier))
            state = ConversationState(conversation_id=identifier, updated_at=self._now())
            self._conversations[identifier] = state
            return state.model_copy(deep=True)

    def delete(self, conversation_id: UUID | str) -> bool:
        """Delete a conversation; return False when it was already absent."""
        identifier = self._normalize_id(conversation_id)
        with self._lock:
            return self._conversations.pop(identifier, None) is not None

    def _merge(
        self,
        current: ConversationState,
        intent: UserIntent,
        message: str,
    ) -> ConversationState:
        data = current.model_dump(mode="python")
        actions = intent.field_actions.model_dump()

        if intent.intent == "clarification":
            data["pending_clarification"] = {
                "original_message": message,
                "question": intent.clarification_question,
            }
        else:
            data["pending_clarification"] = None

        timezone_value = current.timezone
        if actions["timezone"] == "set":
            timezone_value = intent.timezone
        elif actions["timezone"] == "clear":
            timezone_value = "Europe/Athens"
        data["timezone"] = timezone_value

        request = current.active_request
        request_data = (
            request.model_dump(mode="python")
            if request is not None
            else {
                "message": message,
                "requested_date": None,
                "start_time": None,
                "end_time": None,
                "timezone": timezone_value,
                "interests": [],
                "setting": None,
                "child_ages": [],
            }
        )
        request_data["timezone"] = timezone_value
        if actions["requested_date"] == "set":
            request_data["requested_date"] = intent.requested_date
        elif actions["requested_date"] == "clear":
            request_data["requested_date"] = None

        current_start = request_data["start_time"]
        current_end = request_data["end_time"]
        if actions["start_time"] == "set":
            current_start = intent.time_window.start_time
        elif actions["start_time"] == "clear":
            current_start = None
        if actions["end_time"] == "set":
            current_end = intent.time_window.end_time
        elif actions["end_time"] == "clear":
            current_end = None
        request_data["start_time"] = current_start
        request_data["end_time"] = current_end

        preferences = current.preferences.model_dump(mode="python")
        if actions["semantic_interests"] == "set":
            preferences["interests"] = intent.semantic_preferences.interests
        elif actions["semantic_interests"] == "clear":
            preferences["interests"] = []

        hard_filters = current.hard_filters.model_dump(mode="python")
        for action_name, value_name in (
            ("filter_category", "category"),
            ("filter_tags", "tags"),
            ("filter_setting", "setting"),
            ("filter_family_suitability", "family_suitability_fixture"),
            ("filter_excluded_categories", "excluded_categories"),
            ("filter_excluded_tags", "excluded_tags"),
            ("filter_excluded_poi_ids", "excluded_poi_ids"),
        ):
            if actions[action_name] == "set":
                hard_filters[value_name] = getattr(intent.hard_filters, value_name)
            elif actions[action_name] == "clear":
                hard_filters[value_name] = None

        if actions["filter_setting"] == "set":
            preferences["preferred_setting"] = intent.hard_filters.setting
        elif actions["filter_setting"] == "clear":
            preferences["preferred_setting"] = None

        if actions["travel_mode"] == "set":
            data["travel_mode"] = intent.travel_mode
        elif actions["travel_mode"] == "clear":
            data["travel_mode"] = None

        is_replacement_request = (
            intent.intent == "itinerary_modification"
            and actions["itinerary_modification"] == "set"
            and intent.itinerary_modification is not None
            and intent.itinerary_modification.action == "replace_stop"
        )
        replacement_family = current.replacement_family_requirements
        replacement_family_data = (
            replacement_family.model_dump(mode="python")
            if replacement_family is not None
            else {"required": None, "child_ages": None}
        )

        if is_replacement_request:
            replacement_family_data = {"required": None, "child_ages": None}
            for action_name, value_name in (
                ("family_required", "required"),
                ("child_ages", "child_ages"),
            ):
                if actions[action_name] == "set":
                    replacement_family_data[value_name] = getattr(
                        intent.family_requirements, value_name
                    )
                elif actions[action_name] == "clear":
                    replacement_family_data[value_name] = (
                        None if value_name == "required" else []
                    )
            if all(value is None for value in replacement_family_data.values()):
                replacement_family_data = None
        else:
            if actions["family_required"] == "set":
                preferences["family_required"] = intent.family_requirements.required
            elif actions["family_required"] == "clear":
                preferences["family_required"] = None
            if actions["child_ages"] == "set":
                preferences["child_ages"] = intent.family_requirements.child_ages
            elif actions["child_ages"] == "clear":
                preferences["child_ages"] = []

        if actions["itinerary_modification"] == "set":
            data["pending_modification"] = intent.itinerary_modification
            data["last_modification_outcome"] = None
            if is_replacement_request:
                data["replacement_family_requirements"] = replacement_family_data
            else:
                data["replacement_family_requirements"] = None
        elif actions["itinerary_modification"] == "clear":
            data["pending_modification"] = None
            data["replacement_family_requirements"] = None
        elif replacement_family_data != (
            replacement_family.model_dump(mode="python")
            if replacement_family is not None
            else {"required": None, "child_ages": None}
        ):
            data["replacement_family_requirements"] = replacement_family_data

        request_action_names = (
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
        )
        if any(actions[name] != "preserve" for name in request_action_names):
            request_data["message"] = message
            request_data["interests"] = preferences["interests"]
            request_data["setting"] = hard_filters["setting"]
            request_data["child_ages"] = preferences["child_ages"]
            data["active_request"] = request_data

        data["preferences"] = preferences
        data["hard_filters"] = hard_filters
        data["updated_at"] = self._now()
        return ConversationState.model_validate(data)

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Conversation clock must return a timezone-aware datetime")
        return value.astimezone(timezone.utc)

    @staticmethod
    def _normalize_id(conversation_id: UUID | str) -> UUID:
        try:
            return conversation_id if isinstance(conversation_id, UUID) else UUID(conversation_id)
        except (ValueError, TypeError, AttributeError) as error:
            raise ConversationNotFoundError(str(conversation_id)) from error
