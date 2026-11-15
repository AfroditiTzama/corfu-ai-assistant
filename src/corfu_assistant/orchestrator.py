from __future__ import annotations

import logging
import json
import re
import time as time_module
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from threading import RLock
from typing import Callable, Literal, Protocol, Sequence
from uuid import UUID
from zoneinfo import ZoneInfo

from pydantic import Field, ValidationError, field_validator

from corfu_assistant.intent_extractor import IntentExtractor
from corfu_assistant.intent_policy import normalize_thematic_filters
from corfu_assistant.models import (
    ConversationState,
    DomainModel,
    Itinerary,
    POI,
    POIMetadataFilters,
    ItineraryValidationResult,
    UserIntent,
    ValidationIssue,
    WeatherForecast,
    WeatherRequestScope,
)
from corfu_assistant.planner import (
    DeterministicItineraryPlanner,
    PlanningRequest,
    PlanningResult,
    WalkingTransfer,
)
from corfu_assistant.repository import InvalidMetadataFilter, POIRepository
from corfu_assistant.state_manager import (
    ConcurrentConversationUpdateError,
    ConversationNotFoundError,
    ConversationStateManager,
    ConversationTurn,
)
from corfu_assistant.weather import RainAssessment, WeatherProvider, WeatherProviderError, assess_rain


ResponseStatus = Literal[
    "completed",
    "no_evidence",
    "clarification_required",
    "not_implemented",
    "feasible",
    "uncertain",
    "infeasible",
    "conflict",
    "weather_unavailable",
    "unsupported_action",
]
EvidenceMethod = Literal["exact_name", "semantic"]
AnswerMode = Literal["generated", "deterministic", "none"]
CitationStatus = Literal["not_attempted", "valid", "invalid"]
AnswerFallbackReason = Literal[
    "evidence_insufficient",
    "generator_unavailable",
    "provider_error",
    "parsing_error",
    "structured_output_error",
    "citation_validation_error",
    "recommendation_selection_mismatch",
]
WeatherReplanStatus = Literal["feasible", "uncertain", "infeasible"]
_LOGGER = logging.getLogger(__name__)
_URL_IN_TEXT = re.compile(
    r"(?:\bhttps?://|\bwww\.)\S+|\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z]{2,}(?:/\S*)?",
    re.IGNORECASE,
)
_SENSITIVE_ACCESS_CLAIM = re.compile(
    r"\b(?:open|closed|available|availability|accessible|admission)\b",
    re.IGNORECASE,
)
_PRICE_AMOUNT = re.compile(r"(?:€\s*\d+(?:[.,]\d+)?|\b\d+(?:[.,]\d+)?\s*(?:EUR|euros?)\b)", re.IGNORECASE)
_FACT_GAP_DISCLAIMER = re.compile(
    r"\b(?:does not|doesn't|do not|don't|cannot|can't|not)\s+"
    r"(?:provide|specify|state|list|verify|confirm|include|available|given|shown)\b",
    re.IGNORECASE,
)
_SIGHTSEEING_DISCOVERY = re.compile(
    r"\b(?:visit|see|sightseeing|discover|explore|attractions?|landmarks?|"
    r"places? to go|things? to do|cultural activities)\b|"
    r"(?:επισκεφ|αξιοθέατ|ανακαλύψ|εξερευν)",
    re.IGNORECASE,
)
_HOSPITALITY_CONTEXT = re.compile(
    r"\b(?:food|eat|dining|restaurant|taverna|cafe|coffee|bar|drink|nightlife|"
    r"breakfast|lunch|dinner)\b|(?:φαγητ|εστιατόρ|ταβέρν|καφέ|μπαρ|ποτό|νυχτεριν)",
    re.IGNORECASE,
)
_HOSPITALITY_TAGS = frozenset({"food", "restaurant", "cafe", "coffee", "bar"})


class OrchestrationError(RuntimeError):
    """Base error for request orchestration failures."""


class InvalidOrchestrationRequestError(OrchestrationError, ValueError):
    """Raised when request fields or conversation identity are invalid."""


class UnknownConversationError(OrchestrationError, LookupError):
    """Raised when a valid conversation ID is not in the state store."""


class InvalidExtractedConstraintError(OrchestrationError, ValueError):
    """Raised when extracted hard metadata constraints do not match the dataset."""


class AnswerGenerationError(OrchestrationError):
    """Raised when an injected grounded-answer provider fails or violates its contract."""


class OrchestrationRequest(DomainModel):
    message: str = Field(min_length=1)
    conversation_id: UUID | None = None
    request_id: UUID | None = None
    top_k: int = Field(default=5, gt=0, le=50, strict=True)

    @field_validator("message")
    @classmethod
    def normalize_message(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("message must not be blank")
        return normalized


class POIEvidence(DomainModel):
    poi_id: str
    name: str
    description: str
    source_urls: list[str]
    description_provenance: str
    method: EvidenceMethod
    relevance_score: float | None = Field(default=None, ge=-1, le=1)


class GroundedCitation(DomainModel):
    poi_id: str = Field(min_length=1)
    supporting_quote: str = Field(min_length=8)


class GroundedClaim(DomainModel):
    text: str = Field(min_length=1)
    citations: list[GroundedCitation] = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Claim text must not be blank")
        return normalized


class GroundedAnswer(DomainModel):
    claims: list[GroundedClaim] = Field(min_length=1)
    recommended_poi_ids: list[str] = Field(default_factory=list, max_length=50)


class OrchestrationResult(DomainModel):
    conversation_id: UUID
    intent: UserIntent
    status: ResponseStatus
    response: str
    evidence: list[POIEvidence] = Field(default_factory=list)
    relevant_poi_ids: list[str] = Field(default_factory=list)
    recommended_poi_ids: list[str] = Field(default_factory=list)
    answer_mode: AnswerMode = "none"
    answer_attempted: bool = False
    citation_status: CitationStatus = "not_attempted"
    fallback_reason: AnswerFallbackReason | None = None
    planning_result: PlanningResult | None = None
    modification_applied: bool | None = None
    weather_forecast: WeatherForecast | None = None
    rain_assessment: RainAssessment | None = None
    weather_replanning: WeatherReplanningResult | None = None
    weather_error: str | None = None
    conversation_state: ConversationState


class WeatherStopChange(DomainModel):
    """A replacement; stop_index is zero-based in the revised itinerary."""

    stop_index: int
    previous_poi_id: str
    revised_poi_id: str


class WeatherReplanningResult(DomainModel):
    status: WeatherReplanStatus
    planning_result: PlanningResult | None = None
    original_poi_ids: list[str]
    revised_poi_ids: list[str]
    changed_stops: list[WeatherStopChange] = Field(default_factory=list)
    removed_poi_ids: list[str] = Field(default_factory=list)
    reordered: bool = False
    conditional_exposure_poi_ids: list[str] = Field(default_factory=list)
    explanations: list[str] = Field(default_factory=list)


class AnswerGenerator(Protocol):
    """Generate claims with verifiable citations to supplied evidence only."""

    def generate(
        self,
        question: str,
        evidence: Sequence[POIEvidence],
        recommendation_count: int | None = None,
    ) -> GroundedAnswer:
        """Return claims and an ordered, evidence-grounded recommendation selection."""


@dataclass(frozen=True)
class _IdempotentResult:
    fingerprint: tuple[UUID | None, str, int]
    result: OrchestrationResult


class Orchestrator:
    """Deterministic request routing over injected extraction, state, and retrieval."""

    def __init__(
        self,
        *,
        intent_extractor: IntentExtractor,
        state_manager: ConversationStateManager,
        poi_repository: POIRepository,
        planner: DeterministicItineraryPlanner | None = None,
        weather_provider: WeatherProvider | None = None,
        answer_generator: AnswerGenerator | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._intent_extractor = intent_extractor
        self._state_manager = state_manager
        self._poi_repository = poi_repository
        self._planner = planner
        self._weather_provider = weather_provider
        self._answer_generator = answer_generator
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._idempotency_lock = RLock()
        self._request_locks: dict[UUID, RLock] = {}
        self._conversation_locks: dict[UUID, RLock] = {}
        self._idempotent_results: dict[UUID, _IdempotentResult] = {}

    def process(
        self,
        request: OrchestrationRequest | dict[str, object],
    ) -> OrchestrationResult:
        try:
            validated_request = (
                request
                if isinstance(request, OrchestrationRequest)
                else OrchestrationRequest.model_validate(request)
            )
        except (ValidationError, TypeError, ValueError) as error:
            raise InvalidOrchestrationRequestError(
                f"Invalid orchestration request: {error}"
            ) from error

        idempotency_key = (
            validated_request.request_id
            if validated_request.request_id is not None
            else None
        )
        if idempotency_key is None:
            return self._process_serialized(validated_request)

        with self._idempotency_lock:
            request_lock = self._request_locks.setdefault(idempotency_key, RLock())

        with request_lock:
            fingerprint = (
                validated_request.conversation_id,
                validated_request.message,
                validated_request.top_k,
            )
            with self._idempotency_lock:
                cached = self._idempotent_results.get(idempotency_key)
            if cached is not None:
                is_new_conversation_retry = (
                    cached.fingerprint[0] is None
                    and validated_request.conversation_id == cached.result.conversation_id
                    and cached.fingerprint[1:] == fingerprint[1:]
                )
                if cached.fingerprint != fingerprint and not is_new_conversation_retry:
                    raise InvalidOrchestrationRequestError(
                        "request_id was already used with different request content"
                    )
                try:
                    self._state_manager.get(cached.result.conversation_id)
                except ConversationNotFoundError as error:
                    raise UnknownConversationError(
                        f"Unknown conversation ID: {cached.result.conversation_id}"
                    ) from error
                return cached.result.model_copy(deep=True)

            result = self._process_serialized(validated_request)
            with self._idempotency_lock:
                self._idempotent_results[idempotency_key] = _IdempotentResult(
                    fingerprint=(
                        validated_request.conversation_id,
                        validated_request.message,
                        validated_request.top_k,
                    ),
                    result=result.model_copy(deep=True),
                )
            return result

    def _process_serialized(self, request: OrchestrationRequest) -> OrchestrationResult:
        if request.conversation_id is None:
            return self._process(request)
        with self._idempotency_lock:
            lock = self._conversation_locks.setdefault(request.conversation_id, RLock())
        with lock:
            return self._process(request)

    def _process(self, request: OrchestrationRequest) -> OrchestrationResult:
        started = time_module.perf_counter()
        if request.conversation_id is None:
            prior_state = self._state_manager.create_conversation()
        else:
            try:
                prior_state = self._state_manager.get(request.conversation_id)
            except ConversationNotFoundError as error:
                raise UnknownConversationError(
                    f"Unknown conversation ID: {request.conversation_id}"
                ) from error

        intent_started = time_module.perf_counter()
        try:
            extracted = self._intent_extractor.extract(request.message, prior_state)
        except Exception:
            _LOGGER.info(
                "Intent extraction stage completed",
                extra={
                    "stage": "intent_extraction",
                    "outcome": "error",
                    "duration_ms": round((time_module.perf_counter() - intent_started) * 1000, 2),
                },
            )
            if request.conversation_id is None:
                self._state_manager.delete(prior_state.conversation_id)
            raise
        _LOGGER.info(
            "Intent extraction stage completed",
            extra={
                "stage": "intent_extraction",
                "outcome": "success",
                "duration_ms": round((time_module.perf_counter() - intent_started) * 1000, 2),
            },
        )

        try:
            intent = (
                extracted
                if isinstance(extracted, UserIntent)
                else UserIntent.model_validate(extracted)
            )
        except (ValidationError, TypeError, ValueError) as error:
            if request.conversation_id is None:
                self._state_manager.delete(prior_state.conversation_id)
            raise OrchestrationError(
                "Intent extractor returned output that does not match UserIntent"
            ) from error

        # Category-like themes can be represented as dataset tags too. Keep
        # them soft unless the user explicitly asked for literal tags.
        intent, demoted_tags = normalize_thematic_filters(
            intent,
            message=request.message,
            category_vocabulary=self._poi_repository.metadata_categories,
            tag_vocabulary=self._poi_repository.metadata_tags,
            prior_interests=prior_state.preferences.interests,
            poi_names={poi.id: poi.name for poi in self._poi_repository.get_all()},
        )
        if demoted_tags:
            _LOGGER.info(
                "Implicit category-like tags normalized",
                extra={"stage": "intent_normalization", "demoted_tag_count": len(demoted_tags)},
            )

        try:
            turn = self._state_manager.apply_intent(
                prior_state.conversation_id,
                intent,
                message=request.message,
            )
        except ConversationNotFoundError as error:
            raise UnknownConversationError(
                f"Unknown conversation ID: {prior_state.conversation_id}"
            ) from error

        try:
            result = self._route(request, intent, turn)
        except Exception:
            self._state_manager.rollback_failed_turn(
                turn.state, prior_state if request.conversation_id is not None else None
            )
            raise
        _LOGGER.info(
            "Chat orchestration completed",
            extra={
                "intent_kind": intent.intent,
                "response_status": result.status,
                "duration_ms": round((time_module.perf_counter() - started) * 1000, 2),
            },
        )
        return result

    def _route(
        self,
        request: OrchestrationRequest,
        intent: UserIntent,
        turn: ConversationTurn,
    ) -> OrchestrationResult:
        if intent.unsupported_actions:
            action_text = ", ".join(intent.unsupported_actions)
            transaction_note = (
                " No booking or transaction was made."
                if any(
                    action in {"booking", "payment", "purchase"}
                    for action in intent.unsupported_actions
                )
                else " No external action was performed."
            )
            return self._result(
                intent,
                turn.state,
                "unsupported_action",
                f"I cannot perform the requested {action_text} action(s)."
                f"{transaction_note} POI source links are background information, "
                "not booking or communication facilities.",
            )
        if intent.intent == "clarification":
            return self._result(
                intent,
                turn.state,
                "clarification_required",
                turn.clarification_question or "Please clarify your request.",
            )

        if intent.intent == "tourism_qa":
            question = turn.question or request.message
            evidence = self._qa_evidence(question, request.top_k)
            return self._answer_or_default(
                intent,
                turn.state,
                question,
                evidence,
                recommendation=False,
            )

        if intent.intent == "poi_recommendation":
            retrieval_limit = max(request.top_k, intent.recommendation_count or 1)
            weather_suitable = bool(
                turn.weather_request
                and turn.weather_request.activity_recommendation
                and "precipitation" in turn.weather_request.aspects
            )
            evidence = self._recommendation_evidence(
                request.message,
                turn.state,
                retrieval_limit,
                prefer_indoor=weather_suitable,
            )
            return self._answer_or_default(
                intent,
                turn.state,
                request.message,
                evidence,
                recommendation=True,
                recommendation_count=intent.recommendation_count,
            )

        if intent.intent in {"itinerary_creation", "itinerary_modification"}:
            if self._planner is None:
                return self._result(
                    intent,
                    turn.state,
                    "not_implemented",
                    self._unsupported_message(intent),
                )
            if intent.intent == "itinerary_creation":
                return self._create_itinerary(request, intent, turn)
            return self._modify_itinerary(request, intent, turn)

        if intent.intent == "live_weather_query":
            return self._live_weather_query(intent, turn)

        if intent.intent == "weather_based_replanning":
            # A provider can occasionally classify a standalone weather question
            # as a replan. Without a stored itinerary this is a forecast request,
            # not a reason to block the user on plan creation.
            if turn.state.itinerary is None:
                return self._live_weather_query(intent, turn)
            if self._weather_provider is None:
                return self._result(
                    intent,
                    turn.state,
                    "not_implemented",
                    self._unsupported_message(intent),
                )
            if self._planner is None:
                return self._result(
                    intent,
                    turn.state,
                    "not_implemented",
                    "Weather-aware replanning requires the deterministic planner.",
                )
            return self._weather_replan(intent, request, turn)

        raise OrchestrationError(f"Unsupported intent kind: {intent.intent}")

    def _live_weather_query(
        self,
        intent: UserIntent,
        turn: ConversationTurn,
    ) -> OrchestrationResult:
        if self._weather_provider is None:
            return self._result(
                intent,
                turn.state,
                "not_implemented",
                self._unsupported_message(intent),
            )
        scope = turn.weather_request or WeatherRequestScope()
        try:
            if scope.relative_horizon == "next_24_hours":
                forecast = self._weather_provider.next_24_hours()
                response = self._weather_summary(
                    forecast,
                    current=False,
                    period_label="the next 24 hours",
                )
                if scope.activity_recommendation:
                    response += " " + self._weather_activity_guidance(forecast)
            elif (
                scope.relative_horizon is None
                and scope.forecast_date is None
                and scope.duration_hours is None
            ):
                forecast = self._weather_provider.get_current()
                response = self._weather_summary(forecast, current=True)
            else:
                start, end = self._weather_request_window(scope)
                forecast = self._weather_provider.get_forecast(start, end)
                response = self._weather_summary(forecast, current=False)
                if scope.activity_recommendation:
                    response += " " + self._weather_activity_guidance(forecast)
        except WeatherProviderError as error:
            return self._result(
                intent,
                turn.state,
                "weather_unavailable",
                f"Weather data is unavailable ({type(error).__name__}); no synthetic fallback was used.",
                weather_error=type(error).__name__,
            )
        return self._result(
            intent,
            turn.state,
            "completed",
            response,
            weather_forecast=forecast,
        )

    def _weather_request_window(
        self,
        scope: WeatherRequestScope,
    ) -> tuple[datetime, datetime]:
        zone = ZoneInfo("Europe/Athens")
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise OrchestrationError("Weather clock must return a timezone-aware datetime")
        local_now = now.astimezone(zone)

        if scope.relative_horizon == "next_24_hours":
            start = local_now
            end = (local_now.astimezone(timezone.utc) + timedelta(hours=24)).astimezone(zone)
            return start, end
        if scope.relative_horizon == "tomorrow":
            target = local_now.date() + timedelta(days=1)
            start_clock = time.fromisoformat(scope.start_time) if scope.start_time else time.min
            end_clock = time.fromisoformat(scope.end_time) if scope.end_time else time.min
            start = datetime.combine(target, start_clock, tzinfo=zone)
            end_date = target + timedelta(days=1) if scope.end_time is None else target
            end = datetime.combine(end_date, end_clock, tzinfo=zone)
            return start, end
        if scope.relative_horizon == "today":
            start_clock = scope.start_time
            end_clock = scope.end_time
            start = (
                datetime.combine(local_now.date(), time.fromisoformat(start_clock), tzinfo=zone)
                if start_clock is not None
                else local_now
            )
            end = (
                datetime.combine(local_now.date(), time.fromisoformat(end_clock), tzinfo=zone)
                if end_clock is not None
                else datetime.combine(local_now.date() + timedelta(days=1), time.min, tzinfo=zone)
            )
            return start, end
        if scope.relative_horizon == "tonight":
            evening = datetime.combine(local_now.date(), time(18), tzinfo=zone)
            start = max(local_now, evening)
            end = datetime.combine(local_now.date() + timedelta(days=1), time.min, tzinfo=zone)
            if start >= end:
                raise OrchestrationError("The requested tonight forecast window has ended")
            return start, end
        if scope.forecast_date is not None:
            start_clock = time.fromisoformat(scope.start_time) if scope.start_time else time.min
            end_clock = time.fromisoformat(scope.end_time) if scope.end_time else time.min
            start = datetime.combine(scope.forecast_date, start_clock, tzinfo=zone)
            end_date = (
                scope.forecast_date + timedelta(days=1)
                if scope.end_time is None
                else scope.forecast_date
            )
            end = datetime.combine(end_date, end_clock, tzinfo=zone)
            return start, end
        if scope.start_time is not None and scope.end_time is not None:
            return (
                datetime.combine(local_now.date(), time.fromisoformat(scope.start_time), tzinfo=zone),
                datetime.combine(local_now.date(), time.fromisoformat(scope.end_time), tzinfo=zone),
            )
        duration = scope.duration_hours or 24
        start = local_now
        end = (local_now.astimezone(timezone.utc) + timedelta(hours=duration)).astimezone(zone)
        return start, end

    @staticmethod
    def _weather_summary(
        forecast: WeatherForecast,
        *,
        current: bool,
        period_label: str | None = None,
    ) -> str:
        attribution = (
            "synthetic exercise fixture"
            if forecast.source == "synthetic_fixture"
            else "live Open-Meteo data"
        )
        if current:
            hour = forecast.hourly[0]
            conditions = []
            if hour.temperature_c is not None:
                conditions.append(f"{hour.temperature_c:g}°C")
            if hour.precipitation_mm is not None:
                conditions.append(f"{hour.precipitation_mm:g} mm precipitation")
            if hour.weather_code is not None:
                conditions.append(f"WMO code {hour.weather_code}")
            facts = ", ".join(conditions) if conditions else "current values are unavailable"
            return (
                f"Current weather for {forecast.place} ({forecast.timezone}): {facts}. "
                f"Source: {attribution}; retrieved {forecast.retrieved_at}."
            )

        rain_summary = Orchestrator._forecast_precipitation_summary(forecast)
        period = f"For {period_label}: " if period_label else ""
        return (
            f"{period}{rain_summary} Forecast for {forecast.place} ({forecast.timezone}). "
            f"Source: {attribution}; "
            f"retrieved {forecast.retrieved_at}."
        )

    @staticmethod
    def _forecast_precipitation_summary(forecast: WeatherForecast) -> str:
        zone = ZoneInfo(forecast.timezone)
        missing = any(hour.precipitation_mm is None for hour in forecast.hourly)
        known = [hour for hour in forecast.hourly if hour.precipitation_mm is not None]
        if not known:
            return (
                "I can't determine whether rain is expected because precipitation "
                "data is unavailable for this forecast period."
            )

        rainy_intervals: list[tuple[datetime, datetime, float]] = []
        for hour in known:
            amount = hour.precipitation_mm
            assert amount is not None
            if amount <= 0:
                continue
            start = hour.precipitation_interval_start or hour.at
            end = hour.precipitation_interval_end or (
                hour.at.astimezone(timezone.utc) + timedelta(hours=1)
            ).astimezone(zone)
            rainy_intervals.append((start, end, amount))
        rainy_intervals.sort(key=lambda interval: interval[0].astimezone(timezone.utc))

        if not rainy_intervals:
            if missing:
                return (
                    "No rain is indicated in the available precipitation data, but "
                    "some forecast hours are missing, so rain cannot be ruled out "
                    "throughout the period."
                )
            return "No rain is indicated in the precipitation forecast for this period."

        periods: list[tuple[datetime, datetime, float]] = []
        for start, end, amount in rainy_intervals:
            if periods and start.astimezone(timezone.utc) <= periods[-1][1].astimezone(timezone.utc):
                previous_start, previous_end, previous_amount = periods[-1]
                merged_end = max(
                    previous_end.astimezone(timezone.utc),
                    end.astimezone(timezone.utc),
                ).astimezone(zone)
                periods[-1] = (previous_start, merged_end, previous_amount + amount)
            else:
                periods.append((start, end, amount))

        def format_period(period: tuple[datetime, datetime, float]) -> str:
            start, end, total = period
            start = start.astimezone(zone)
            end = end.astimezone(zone)
            total_mm = round(total, 2)
            if start.date() == end.date():
                timing = f"{start:%a %H:%M}–{end:%H:%M}"
            else:
                timing = f"{start:%a %H:%M}–{end:%a %H:%M}"
            return f"{timing} ({total_mm:g} mm reported)"

        periods_text = "; ".join(format_period(period) for period in periods)
        summary = (
            f"Rain is expected during {periods_text}. Plan for wet conditions then; "
            "rain protection or indoor options may be useful."
        )
        if missing:
            summary += (
                " Some hourly precipitation data is unavailable, so conditions "
                "outside the reported rainy intervals are uncertain."
            )
        return summary

    @staticmethod
    def _weather_activity_guidance(forecast: WeatherForecast) -> str:
        precipitation = [hour.precipitation_mm for hour in forecast.hourly]
        if not precipitation or any(value is None for value in precipitation):
            return (
                "The requested activity window has incomplete precipitation data, "
                "so I cannot reliably recommend indoor versus outdoor sightseeing."
            )
        if any(value > 0 for value in precipitation if value is not None):
            return (
                "The forecast reports precipitation during this window, so an indoor "
                "activity is the safer weather choice. This does not confirm that any "
                "venue is open."
            )
        return (
            "No precipitation is reported for the sampled hours, so outdoor sightseeing "
            "looks reasonable based on this forecast. Forecast conditions can change."
        )

    def _weather_replan(
        self,
        intent: UserIntent,
        request: OrchestrationRequest,
        turn: ConversationTurn,
    ) -> OrchestrationResult:
        itinerary = turn.state.itinerary
        if itinerary is None:
            return self._result(
                intent,
                turn.state,
                "clarification_required",
                "Create an itinerary before requesting weather-based replanning.",
            )
        if turn.state.active_request is None:
            return self._result(
                intent,
                turn.state,
                "clarification_required",
                "The itinerary's requested date and time window are unavailable.",
            )
        planning_request = self._planning_request(turn.state)
        if isinstance(planning_request, str):
            return self._result(
                intent,
                turn.state,
                "clarification_required",
                planning_request,
            )
        zone = ZoneInfo(itinerary.timezone)
        itinerary_start = itinerary.starts_at.astimezone(zone)
        itinerary_end = itinerary.ends_at.astimezone(zone)
        if (
            itinerary_start.date() != planning_request.requested_date
            or itinerary_start.strftime("%H:%M") != planning_request.start_time
            or itinerary_end.strftime("%H:%M") != planning_request.end_time
        ):
            return self._result(
                intent,
                turn.state,
                "clarification_required",
                "The stored itinerary does not match the active requested date and time window; no replanning was applied.",
            )
        if self._weather_provider is None or self._planner is None:
            return self._result(
                intent,
                turn.state,
                "not_implemented",
                "Weather-aware replanning requires both a weather provider and the deterministic planner.",
            )

        original_ids = [stop.poi_id for stop in itinerary.stops]
        try:
            forecast = self._weather_provider.get_forecast(itinerary.starts_at, itinerary.ends_at)
        except WeatherProviderError as error:
            details = WeatherReplanningResult(
                status="infeasible",
                planning_result=None,
                original_poi_ids=original_ids,
                revised_poi_ids=original_ids,
                explanations=[
                    f"Weather data is unavailable ({type(error).__name__}); the stored itinerary was preserved."
                ],
            )
            return self._result(
                intent,
                turn.state,
                "weather_unavailable",
                details.explanations[0],
                weather_replanning=details,
                weather_error=type(error).__name__,
            )

        rain = assess_rain(forecast, itinerary.starts_at, itinerary.ends_at)
        base_filters = planning_request.hard_filters.model_copy(
            update={"setting": None}
        )
        plan_request = planning_request.model_copy(
            update={"hard_filters": base_filters}
        )
        validation = self._planner.validate(plan_request, itinerary)
        original_assessment = self._itinerary_weather_exposure(forecast, itinerary)
        conditional_ids = original_assessment[1]
        if not validation.is_valid and not self._only_unknown_hours(validation):
            planning = PlanningResult(
                status="infeasible",
                itinerary=itinerary,
                validation=validation,
                explanations=[issue.message for issue in validation.issues],
                selected_poi_ids=original_ids,
            )
            return self._weather_replan_result(
                intent,
                turn.state,
                forecast,
                rain,
                planning,
                status="infeasible",
                original_ids=original_ids,
                revised_ids=original_ids,
                conditional_ids=conditional_ids,
                explanations=[
                    "The stored itinerary fails independent planning validation; it was not changed.",
                    *planning.explanations,
                ],
            )

        current_itinerary = itinerary
        current_planning = self._planning_from_existing(itinerary, validation)
        known_exposed, conditional_ids = self._itinerary_weather_exposure(
            forecast,
            current_itinerary,
        )
        explanations: list[str] = []

        if known_exposed:
            reordered_request = plan_request.model_copy(
                update={
                    "required_poi_ids": original_ids,
                    "ranked_candidate_ids": original_ids,
                    "preserved_stops": [],
                    "max_stops": len(original_ids),
                    "allow_duplicate_stops": False,
                }
            )
            reordered = self._plan_and_validate(
                reordered_request,
                candidates=self._poi_repository.get_all(),
            )
            if self._planning_is_accepted(reordered) and reordered.itinerary is not None:
                reordered_exposed, reordered_conditional = self._itinerary_weather_exposure(
                    forecast,
                    reordered.itinerary,
                )
                # Keep the existing order when a full reschedule does not
                # reduce known rain exposure; otherwise it can move safe stops
                # into rain and cause needless replacements.
                if len(reordered_exposed) < len(known_exposed):
                    current_itinerary = reordered.itinerary
                    current_planning = reordered
                    known_exposed = reordered_exposed
                    conditional_ids = reordered_conditional
                if not reordered_exposed and current_itinerary is reordered.itinerary:
                    reordered_ids = [
                        stop.poi_id for stop in current_itinerary.stops
                    ]
                    action = (
                        "reordered and rescheduled"
                        if reordered_ids != original_ids
                        else "rescheduled"
                    )
                    explanations.append(
                        f"Existing stops were {action} to avoid known rain intervals; no POI was replaced."
                    )

        if known_exposed:
            for stop_index in known_exposed:
                replacement_filters = base_filters.model_copy(
                    update={"setting": "indoor"}
                )
                if plan_request.family_required:
                    replacement_filters = replacement_filters.model_copy(
                        update={"family_suitability_fixture": True}
                    )
                candidates = self._rank_planning_candidates(
                    request.message,
                    turn.state.preferences.interests,
                    replacement_filters,
                )
                preferred_poi_ids = self._focused_candidate_ids(
                    request.message, turn.state.preferences.interests, candidates
                )
                trace = {
                    "normalized_semantic_interests": list(turn.state.preferences.interests)[:12],
                    "preferred_indoor_candidate_count": len(preferred_poi_ids),
                    "preferred_indoor_candidate_ids": preferred_poi_ids[:50],
                    "candidate_ranking_order": [poi.id for poi in candidates[:50]],
                }
                _LOGGER.warning(
                    "Weather replacement candidates ranked | trace=%s",
                    json.dumps(trace, separators=(",", ":")),
                    extra={
                        "stage": "weather_replacement_selection",
                        **trace,
                    },
                )
                replacement_request = plan_request.model_copy(
                    update={
                        "hard_filters": POIMetadataFilters(),
                        "family_required": False,
                        "preserved_stops": list(current_itinerary.stops),
                        "required_poi_ids": [],
                        "replacement_stop_index": stop_index,
                        "replacement_hard_filters": replacement_filters,
                        "replacement_family_required": plan_request.family_required,
                        "replacement_required_poi_ids": [],
                        "ranked_candidate_ids": [poi.id for poi in candidates],
                        "preferred_poi_ids": preferred_poi_ids,
                        "max_stops": len(current_itinerary.stops),
                        "allow_duplicate_stops": False,
                    }
                )
                replanned = self._plan_and_validate(
                    replacement_request,
                    candidates=candidates,
                )
                if not self._planning_is_accepted(replanned) or replanned.itinerary is None:
                    explanations.extend(
                        [
                            "No validated indoor replacement could be scheduled within the original time window; the stored itinerary was preserved.",
                            *replanned.explanations,
                        ]
                    )
                    return self._weather_replan_result(
                        intent,
                        turn.state,
                        forecast,
                        rain,
                        replanned,
                        status="infeasible",
                        original_ids=original_ids,
                        revised_ids=original_ids,
                        conditional_ids=conditional_ids,
                        explanations=explanations,
                    )
                current_itinerary = replanned.itinerary
                current_planning = replanned
                replacement_id = current_itinerary.stops[stop_index].poi_id
                if (
                    replacement_request.preferred_poi_ids
                    and replacement_id not in replacement_request.preferred_poi_ids
                ):
                    explanations.append(
                        "No preferred thematic replacement fitted the preserved schedule. "
                        "This indoor alternative prioritizes shelter over the original theme."
                    )
                explanations.append(
                    f"Stop {stop_index + 1} was replaced with an indoor-setting POI because its visit overlapped a rain interval."
                )
                known_exposed, conditional_ids = self._itinerary_weather_exposure(
                    forecast,
                    current_itinerary,
                )
                if not known_exposed:
                    break

        known_exposed, conditional_ids = self._itinerary_weather_exposure(
            forecast,
            current_itinerary,
        )
        if known_exposed:
            issue = ValidationIssue(
                code="known_rain_exposure",
                message=(
                    "The revised itinerary still contains outdoor or mixed-setting "
                    "visits during a known rain interval."
                ),
            )
            failed = PlanningResult(
                status="infeasible",
                itinerary=current_itinerary,
                validation=current_planning.validation.model_copy(
                    update={
                        "is_valid": False,
                        "issues": [*current_planning.validation.issues, issue],
                    }
                ),
                transfers=current_planning.transfers,
                explanations=[issue.message],
                selected_poi_ids=[stop.poi_id for stop in current_itinerary.stops],
            )
            return self._weather_replan_result(
                intent,
                turn.state,
                forecast,
                rain,
                failed,
                status="infeasible",
                original_ids=original_ids,
                revised_ids=original_ids,
                conditional_ids=conditional_ids,
                explanations=[
                    "Known rain exposure remains after replanning; the original itinerary was preserved."
                ],
            )

        final_validation = self._planner.validate(plan_request, current_itinerary)
        current_planning = self._planning_from_existing(
            current_itinerary,
            final_validation,
            transfers=current_planning.transfers,
        )
        if current_planning.status == "infeasible":
            return self._weather_replan_result(
                intent,
                turn.state,
                forecast,
                rain,
                current_planning,
                status="infeasible",
                original_ids=original_ids,
                revised_ids=original_ids,
                conditional_ids=conditional_ids,
                explanations=[
                    "Independent itinerary validation failed; the original itinerary was preserved.",
                    *current_planning.explanations,
                ],
            )

        status: WeatherReplanStatus = (
            "uncertain"
            if current_planning.status == "uncertain" or conditional_ids
            else "feasible"
        )
        revised_ids = [stop.poi_id for stop in current_itinerary.stops]
        if any(
            assess_rain(forecast, transfer.starts_at, transfer.ends_at).rain_forecast is True
            for transfer in current_planning.transfers
        ):
            explanations.append(
                "Rain overlaps walking transfers between stops. Indoor visits do not "
                "make the walking route fully sheltered; rain protection is still needed."
            )
        if not explanations:
            explanations.append(
                "No existing stop overlapped a known rain interval; the itinerary was left unchanged."
            )
        if conditional_ids:
            explanations.append(
                "Weather coverage is incomplete for these outdoor or mixed-setting stops; their exposure is unknown, not confirmed dry."
            )
        if status == "uncertain":
            current_planning = current_planning.model_copy(update={"status": "uncertain"})
        try:
            committed = self._state_manager.commit_planned_itinerary(
                turn.state.conversation_id,
                current_itinerary,
                expected_state=turn.state,
                expected_pending_modification=turn.state.pending_modification,
            )
        except ConcurrentConversationUpdateError:
            current_state = self._state_manager.get(turn.state.conversation_id)
            details = self._weather_replanning_details(
                status="infeasible",
                planning_result=current_planning,
                original_ids=original_ids,
                revised_ids=original_ids,
                conditional_ids=conditional_ids,
                explanations=[
                    "Conversation state changed during weather replanning; no itinerary update was committed."
                ],
            )
            return self._result(
                intent,
                current_state,
                "conflict",
                details.explanations[0],
                planning_result=current_planning,
                weather_forecast=forecast,
                rain_assessment=rain,
                weather_replanning=details,
            )

        current_planning = current_planning.model_copy(
            update={"itinerary": committed.itinerary}
        )
        return self._weather_replan_result(
            intent,
            committed,
            forecast,
            rain,
            current_planning,
            status=status,
            original_ids=original_ids,
            revised_ids=revised_ids,
            conditional_ids=conditional_ids,
            explanations=explanations,
        )

    def _itinerary_weather_exposure(
        self,
        forecast: WeatherForecast,
        itinerary: Itinerary,
    ) -> tuple[list[int], list[str]]:
        known_exposed: list[int] = []
        conditional: list[str] = []
        for index, stop in enumerate(itinerary.stops):
            poi = self._poi_repository.get_by_id(stop.poi_id)
            if poi is None or poi.setting == "indoor":
                continue
            assessment = assess_rain(forecast, stop.starts_at, stop.ends_at)
            if assessment.rain_forecast is True:
                known_exposed.append(index)
            elif assessment.rain_forecast is None:
                conditional.append(poi.id)
        return known_exposed, list(dict.fromkeys(conditional))

    @staticmethod
    def _only_unknown_hours(validation: ItineraryValidationResult) -> bool:
        return bool(validation.issues) and all(
            issue.code == "unknown_opening_hours"
            for issue in validation.issues
        )

    @staticmethod
    def _planning_from_existing(
        itinerary: Itinerary,
        validation: ItineraryValidationResult,
        *,
        transfers: list[WalkingTransfer] | None = None,
    ) -> PlanningResult:
        if validation.is_valid:
            status = "feasible"
        elif Orchestrator._only_unknown_hours(validation):
            status = "uncertain"
        else:
            status = "infeasible"
        return PlanningResult(
            status=status,
            itinerary=itinerary,
            transfers=transfers or [],
            validation=validation,
            explanations=[issue.message for issue in validation.issues],
            selected_poi_ids=[stop.poi_id for stop in itinerary.stops],
        )

    def _weather_replan_result(
        self,
        intent: UserIntent,
        state: ConversationState,
        forecast: WeatherForecast,
        rain: RainAssessment,
        planning: PlanningResult,
        *,
        status: WeatherReplanStatus,
        original_ids: list[str],
        revised_ids: list[str],
        conditional_ids: list[str],
        explanations: list[str],
    ) -> OrchestrationResult:
        details = self._weather_replanning_details(
            status=status,
            planning_result=planning,
            original_ids=original_ids,
            revised_ids=revised_ids,
            conditional_ids=conditional_ids,
            explanations=explanations,
        )
        response_status: ResponseStatus = status
        summary = " ".join(explanations)
        if status == "infeasible":
            summary += " The previous itinerary remains stored."
        elif status == "uncertain":
            summary += " This result is conditional, not a weather-safety guarantee."
        return self._result(
            intent,
            state,
            response_status,
            summary,
            planning_result=planning,
            weather_forecast=forecast,
            rain_assessment=rain,
            weather_replanning=details,
        )

    @staticmethod
    def _weather_replanning_details(
        *,
        status: WeatherReplanStatus,
        planning_result: PlanningResult | None,
        original_ids: list[str],
        revised_ids: list[str],
        conditional_ids: list[str],
        explanations: list[str],
    ) -> WeatherReplanningResult:
        changed: list[WeatherStopChange] = []
        if len(original_ids) == len(revised_ids):
            missing = set(original_ids) - set(revised_ids)
            additions = [
                (index, poi_id)
                for index, poi_id in enumerate(revised_ids)
                if poi_id not in original_ids
            ]
            for old_id in original_ids:
                if old_id in missing and additions:
                    revised_index, new_id = additions.pop(0)
                    changed.append(
                        WeatherStopChange(
                            # The UI presents a stop number in the revised route,
                            # so this index must refer to the replacement's slot.
                            stop_index=revised_index,
                            previous_poi_id=old_id,
                            revised_poi_id=new_id,
                        )
                    )
        original_common_order = [poi_id for poi_id in original_ids if poi_id in revised_ids]
        revised_common_order = [poi_id for poi_id in revised_ids if poi_id in original_ids]
        return WeatherReplanningResult(
            status=status,
            planning_result=planning_result,
            original_poi_ids=original_ids,
            revised_poi_ids=revised_ids,
            changed_stops=changed,
            removed_poi_ids=[
                poi_id for poi_id in original_ids if poi_id not in revised_ids
            ],
            reordered=original_common_order != revised_common_order,
            conditional_exposure_poi_ids=conditional_ids,
            explanations=explanations,
        )

    def _qa_evidence(self, question: str, top_k: int) -> list[POIEvidence]:
        started = time_module.perf_counter()
        exact_pois = self._exact_name_matches(question)
        if exact_pois:
            pois = exact_pois[:top_k]
            evidence = [self._evidence(poi, "exact_name") for poi in pois]
            method = "exact_name"
        else:
            retrieved = self._poi_repository.hybrid_search(question, top_k=top_k)
            evidence = [
                self._evidence(
                    self._required_poi(result.poi_id),
                    "semantic",
                    result.relevance_score,
                )
                for result in retrieved
            ]
            method = "semantic"
        _LOGGER.info(
            "Tourism QA retrieval completed",
            extra={
                "stage": "retrieval",
                "method": method,
                "candidate_count": len(evidence),
                "duration_ms": round((time_module.perf_counter() - started) * 1000, 2),
            },
        )
        return evidence

    def _recommendation_evidence(
        self,
        message: str,
        state: ConversationState,
        top_k: int,
        *,
        prefer_indoor: bool = False,
    ) -> list[POIEvidence]:
        hard_filters = state.hard_filters
        family_suitability = (
            True
            if state.preferences.family_required is True
            else hard_filters.family_suitability_fixture
        )
        filters = POIMetadataFilters(
            category=hard_filters.category,
            tags=hard_filters.tags or [],
            excluded_categories=hard_filters.excluded_categories or [],
            excluded_tags=hard_filters.excluded_tags or [],
            excluded_poi_ids=hard_filters.excluded_poi_ids or [],
            setting=hard_filters.setting,
            family_suitability_fixture=family_suitability,
        )
        started = time_module.perf_counter()
        query = " ".join(
            part
            for part in (message, *state.preferences.interests)
            if part.strip()
        )
        prefer_sightseeing = bool(
            _SIGHTSEEING_DISCOVERY.search(query)
            and not _HOSPITALITY_CONTEXT.search(query)
        )
        search_limit = self._poi_repository.count if (prefer_indoor or prefer_sightseeing) else top_k
        try:
            retrieved = self._poi_repository.hybrid_search(
                query,
                filters=filters,
                top_k=search_limit,
            )
            rank_by_id = {result.poi_id: index for index, result in enumerate(retrieved)}
            retrieved.sort(
                key=lambda result: (
                    prefer_sightseeing and self._is_hospitality_venue(result.poi_id),
                    prefer_indoor
                    and filters.setting is None
                    and self._required_poi(result.poi_id).setting != "indoor",
                    rank_by_id[result.poi_id],
                )
            )
            retrieved = retrieved[:top_k]
        except InvalidMetadataFilter as error:
            raise InvalidExtractedConstraintError(
                f"Extracted hard filter is not in the POI dataset vocabulary: {error}"
            ) from error
        _LOGGER.info(
            "POI recommendation retrieval completed",
            extra={
                "eligible_candidate_count": len(
                    self._poi_repository.filter_metadata(filters)
                ),
                "retrieved_candidate_count": len(retrieved),
                "indoor_preference_applied": prefer_indoor and filters.setting is None,
                "sightseeing_preference_applied": prefer_sightseeing,
                "duration_ms": round((time_module.perf_counter() - started) * 1000, 2),
            },
        )
        return [
            self._evidence(
                self._required_poi(result.poi_id),
                "semantic",
                result.relevance_score,
            )
            for result in retrieved
        ]

    def _is_hospitality_venue(self, poi_id: str) -> bool:
        poi = self._required_poi(poi_id)
        return bool(_HOSPITALITY_TAGS & {tag.casefold() for tag in poi.tags})

    def _create_itinerary(
        self,
        request: OrchestrationRequest,
        intent: UserIntent,
        turn: ConversationTurn,
    ) -> OrchestrationResult:
        if turn.state.pending_modification is not None:
            return self._result(
                intent,
                turn.state,
                "clarification_required",
                "A stop replacement is still pending. Complete or cancel it before creating a new itinerary.",
            )
        planning_request = self._planning_request(turn.state)
        if isinstance(planning_request, str):
            return self._result(
                intent,
                turn.state,
                "clarification_required",
                planning_request,
            )
        required_ids, ambiguities = self._resolve_named_pois(request.message)
        if ambiguities and not required_ids:
            return self._result(
                intent,
                turn.state,
                "clarification_required",
                self._ambiguity_message(ambiguities),
            )

        candidates = self._rank_planning_candidates(
            request.message,
            turn.state.preferences.interests,
            planning_request.hard_filters,
        )
        planning_request = planning_request.model_copy(
            update={
                "required_poi_ids": required_ids,
                "ranked_candidate_ids": [poi.id for poi in candidates],
                "preferred_poi_ids": self._focused_candidate_ids(
                    request.message, turn.state.preferences.interests, candidates
                ),
            }
        )
        assert self._planner is not None
        planning = self._plan_and_validate(planning_request, candidates=candidates)

        if ambiguities and planning.status != "infeasible":
            return self._result(
                intent,
                turn.state,
                "clarification_required",
                self._ambiguity_message(ambiguities),
            )
        if ambiguities:
            planning = self._add_ambiguity_issues(planning, ambiguities)
        if not self._planning_is_accepted(planning):
            return self._planning_result(intent, turn.state, planning)
        if planning.itinerary is None:
            raise OrchestrationError("Accepted planning result did not contain an itinerary")
        try:
            committed = self._state_manager.commit_planned_itinerary(
                turn.state.conversation_id,
                planning.itinerary,
                expected_state=turn.state,
            )
        except ConcurrentConversationUpdateError:
            return self._result(
                intent,
                self._state_manager.get(turn.state.conversation_id),
                "conflict",
                "Conversation state changed while planning. The itinerary was not saved; please retry.",
                planning_result=planning,
            )
        planning = planning.model_copy(update={"itinerary": committed.itinerary})
        return self._planning_result(intent, committed, planning)

    def _modify_itinerary(
        self,
        request: OrchestrationRequest,
        intent: UserIntent,
        turn: ConversationTurn,
    ) -> OrchestrationResult:
        modification = turn.state.pending_modification
        itinerary = turn.state.itinerary
        if modification is None:
            return self._result(
                intent,
                turn.state,
                "clarification_required",
                "No pending itinerary modification was found. Ask to replace a stop in the current itinerary.",
                modification_applied=False,
            )
        if modification.action != "replace_stop":
            return self._result(
                intent,
                turn.state,
                "not_implemented",
                "Only replacing one existing itinerary stop is supported in this phase.",
                modification_applied=False,
            )
        if itinerary is None:
            return self._result(
                intent,
                turn.state,
                "clarification_required",
                "There is no current itinerary to modify. Create an itinerary first.",
                modification_applied=False,
            )
        if modification.stop_index is None or modification.stop_index >= len(itinerary.stops):
            issue = ValidationIssue(
                code="invalid_replacement_index",
                message=(
                    f"Stop index {modification.stop_index!r} is outside the current "
                    f"itinerary, which has {len(itinerary.stops)} stops."
                ),
            )
            planning = self._infeasible_planning_result(issue)
            return self._planning_result(
                intent,
                turn.state,
                planning,
                modification_applied=False,
            )

        planning_request = self._planning_request(turn.state)
        if isinstance(planning_request, str):
            return self._result(
                intent,
                turn.state,
                "clarification_required",
                planning_request,
                modification_applied=False,
            )
        named_ids, ambiguities = self._resolve_named_pois(request.message)
        if ambiguities:
            return self._result(
                intent,
                turn.state,
                "clarification_required",
                self._ambiguity_message(ambiguities),
                modification_applied=False,
            )

        replacement_family = turn.state.replacement_family_requirements
        replacement_family_required = bool(
            replacement_family is not None
            and (
                replacement_family.required is True
                or bool(replacement_family.child_ages)
            )
        )
        replacement_filters = self._effective_hard_filters(
            turn.state,
            family_required=replacement_family_required,
        )
        candidates = self._rank_planning_candidates(
            request.message,
            turn.state.preferences.interests,
            replacement_filters,
        )
        planning_request = planning_request.model_copy(
            update={
                "hard_filters": POIMetadataFilters(),
                "family_required": False,
                "preserved_stops": list(itinerary.stops),
                "replacement_stop_index": modification.stop_index,
                "replacement_hard_filters": replacement_filters,
                "replacement_family_required": replacement_family_required,
                "replacement_required_poi_ids": named_ids,
                "ranked_candidate_ids": [poi.id for poi in candidates],
                "max_stops": len(itinerary.stops),
                "allow_duplicate_stops": False,
            }
        )
        assert self._planner is not None
        planning = self._plan_and_validate(planning_request, candidates=candidates)
        if not self._planning_is_accepted(planning):
            return self._planning_result(
                intent,
                turn.state,
                planning,
                modification_applied=False,
            )
        if planning.itinerary is None:
            raise OrchestrationError("Accepted replacement result did not contain an itinerary")
        try:
            committed = self._state_manager.commit_planned_itinerary(
                turn.state.conversation_id,
                planning.itinerary,
                expected_state=turn.state,
                expected_pending_modification=modification,
            )
        except ConcurrentConversationUpdateError:
            return self._result(
                intent,
                self._state_manager.get(turn.state.conversation_id),
                "conflict",
                "Conversation changed while replacing the stop. No changes were saved; please retry.",
                planning_result=planning,
                modification_applied=False,
            )
        planning = planning.model_copy(update={"itinerary": committed.itinerary})
        return self._planning_result(
            intent,
            committed,
            planning,
            modification_applied=True,
        )

    def _planning_request(
        self,
        state: ConversationState,
    ) -> PlanningRequest | str:
        active = state.active_request
        if active is None:
            return "Please provide a local date, start time, and end time for the itinerary."
        missing: list[str] = []
        if active.requested_date is None:
            missing.append("date")
        if active.start_time is None:
            missing.append("start time")
        if active.end_time is None:
            missing.append("end time")
        if missing:
            return (
                "Please provide the itinerary " + ", ".join(missing)
                + " (local Corfu time)."
            )
        filters = self._effective_hard_filters(
            state,
            family_required=state.preferences.family_required is True,
        )
        try:
            return PlanningRequest(
                requested_date=active.requested_date,
                start_time=active.start_time,
                end_time=active.end_time,
                timezone=state.timezone,
                travel_mode=state.travel_mode or "walking",
                hard_filters=filters,
                family_required=state.preferences.family_required is True,
                semantic_interests=list(state.preferences.interests),
                max_stops=6,
            )
        except (ValidationError, TypeError, ValueError) as error:
            return f"The merged itinerary constraints are invalid: {error}"

    @staticmethod
    def _effective_hard_filters(
        state: ConversationState,
        *,
        family_required: bool,
    ) -> POIMetadataFilters:
        return POIMetadataFilters(
            category=state.hard_filters.category,
            tags=state.hard_filters.tags or [],
            excluded_categories=state.hard_filters.excluded_categories or [],
            excluded_tags=state.hard_filters.excluded_tags or [],
            excluded_poi_ids=state.hard_filters.excluded_poi_ids or [],
            setting=state.hard_filters.setting,
            family_suitability_fixture=(
                True
                if family_required
                else state.hard_filters.family_suitability_fixture
            ),
        )

    def _rank_planning_candidates(
        self,
        message: str,
        interests: Sequence[str],
        filters: POIMetadataFilters,
    ) -> list[POI]:
        query = " ".join(part for part in (message, *interests) if part.strip())
        try:
            results = self._poi_repository.hybrid_search(
                query,
                filters=filters,
                top_k=self._poi_repository.count,
            )
        except InvalidMetadataFilter as error:
            raise InvalidExtractedConstraintError(
                f"Extracted hard filter is not in the POI dataset vocabulary: {error}"
            ) from error
        return [
            self._required_poi(result.poi_id)
            for result in results
        ]

    @staticmethod
    def _focused_candidate_ids(
        message: str, interests: Sequence[str], candidates: Sequence[POI]
    ) -> list[str]:
        """Resolve soft focus terms to supplied dataset tags/categories."""
        # Intent extraction owns language interpretation. Only positive,
        # structured preferences may influence this optional selection policy.
        text = " ".join(interests).casefold()
        aliases = {
            "museum": ("museum", "museums"),
            "art": ("art", "arts", "gallery", "galleries"),
            "history": ("history", "historical", "historic"),
            "nature": ("nature", "natural"),
            "architecture": ("architecture", "architectural"),
        }
        if not text.strip():
            return []
        requested = {
            key for key, forms in aliases.items()
            if any(re.search(r"\b" + re.escape(form) + r"\b", text) for form in forms)
        }
        if not requested:
            return []
        focused: list[str] = []
        for poi in candidates:
            tags = {tag.casefold() for tag in poi.tags}
            category = poi.category.casefold()
            matches = (
                ("museum" in requested and "museum" in tags)
                or ("art" in requested and bool(tags & {"art", "gallery", "painting", "studio"}))
                or ("history" in requested and category == "history")
                or ("nature" in requested and category == "nature")
                or ("architecture" in requested and "architecture" in tags)
            )
            if matches:
                focused.append(poi.id)
        return focused

    def _resolve_named_pois(self, message: str) -> tuple[list[str], list[str]]:
        normalized = message.casefold()
        hits: list[tuple[int, int, str, POI]] = []
        unresolved_ids: list[str] = []
        suffixes = (" of palaiopolis", " palace", " museum", " park")
        for poi in self._poi_repository.get_all():
            normalized_name = poi.name.casefold()
            phrases = {normalized_name}
            without_parenthetical = re.sub(r"\s*\([^)]*\)", "", normalized_name).strip()
            if without_parenthetical:
                phrases.add(without_parenthetical)
            conjunction_prefix = re.split(r"\s+and\s+", without_parenthetical, maxsplit=1)[0]
            if len(conjunction_prefix.split()) > 1:
                phrases.add(conjunction_prefix)
            if poi.name.casefold().startswith("gate of the "):
                fortress = normalized_name.removeprefix("gate of the ")
                phrases.add(f"{fortress} gate")
                phrases.add(f"gate of {fortress}")
            pending = list(phrases)
            while pending:
                phrase = pending.pop()
                for suffix in suffixes:
                    if phrase.endswith(suffix):
                        alias = phrase[: -len(suffix)].strip()
                        if len(alias.split()) > 1 and alias not in phrases:
                            phrases.add(alias)
                            pending.append(alias)
            for phrase in phrases:
                pattern = rf"(?<!\w){re.escape(phrase)}(?!\w)"
                for match in re.finditer(pattern, normalized):
                    hits.append((match.start(), match.end(), phrase, poi))
        for match in re.finditer(r"(?<!\w)CF-\d{3}(?!\w)", message, re.IGNORECASE):
            poi_id = match.group().upper()
            poi = self._poi_repository.get_by_id(poi_id)
            if poi is not None:
                hits.append((match.start(), match.end(), poi_id, poi))
            else:
                unresolved_ids.append(poi_id)

        resolved: list[str] = []
        ambiguities: list[str] = []
        consumed: list[tuple[int, int]] = []
        for start, end, phrase, poi in sorted(
            hits,
            key=lambda hit: (hit[0], -(hit[1] - hit[0]), hit[3].id),
        ):
            if any(start < prior_end and prior_start < end for prior_start, prior_end in consumed):
                continue
            same_span = {
                candidate.id
                for hit_start, hit_end, _, candidate in hits
                if hit_start == start and hit_end == end
            }
            if len(same_span) > 1:
                ambiguities.append(message[start:end])
            else:
                resolved.append(poi.id)
            consumed.append((start, end))
        return list(dict.fromkeys([*resolved, *unresolved_ids])), list(dict.fromkeys(ambiguities))

    @staticmethod
    def _ambiguity_message(ambiguities: Sequence[str]) -> str:
        names = ", ".join(f"“{name}”" for name in ambiguities)
        return (
            f"I found more than one matching dataset location for {names}. "
            "Please specify the exact venue, such as Mon Repos Museum or Mon Repos Park."
        )

    def _plan_and_validate(
        self,
        request: PlanningRequest,
        *,
        candidates: Sequence[POI] | None = None,
    ) -> PlanningResult:
        if self._planner is None:
            raise OrchestrationError("Planning requires the deterministic planner")
        started = time_module.perf_counter()
        try:
            planning = self._planner.plan(request, candidates=candidates)
            planning = self._independently_validate(request, planning)
        except Exception as error:
            _LOGGER.info(
                "Planning stage completed",
                extra={
                    "stage": "planner_validator",
                    "outcome": "error",
                    "error_type": type(error).__name__,
                    "duration_ms": round((time_module.perf_counter() - started) * 1000, 2),
                },
            )
            raise
        _LOGGER.info(
            "Planning stage completed",
            extra={
                "stage": "planner_validator",
                "outcome": planning.status,
                "stop_count": len(planning.selected_poi_ids),
                "constraint_violation_count": len(planning.validation.issues),
                "duration_ms": round((time_module.perf_counter() - started) * 1000, 2),
            },
        )
        return planning

    def _independently_validate(
        self,
        request: PlanningRequest,
        planning: PlanningResult,
    ) -> PlanningResult:
        if self._planner is None or planning.itinerary is None:
            return planning
        validation = self._planner.validate(request, planning.itinerary)
        if validation.is_valid:
            status = "feasible"
        elif validation.issues and all(
            issue.code == "unknown_opening_hours"
            for issue in validation.issues
        ):
            status = "uncertain"
        else:
            status = "infeasible"
        explanations = (
            [issue.message for issue in validation.issues]
            if status != "feasible"
            else planning.explanations
        )
        return planning.model_copy(
            update={
                "status": status,
                "validation": validation,
                "explanations": explanations,
            }
        )

    @staticmethod
    def _planning_is_accepted(planning: PlanningResult) -> bool:
        return planning.status in {"feasible", "uncertain"} and planning.itinerary is not None

    def _planning_result(
        self,
        intent: UserIntent,
        state: ConversationState,
        planning: PlanningResult,
        *,
        modification_applied: bool | None = None,
    ) -> OrchestrationResult:
        if planning.status == "feasible":
            status: ResponseStatus = "feasible"
            response = (
                "The itinerary is scheduled and validated against the supplied exercise "
                "fixtures. Walking transfer times are approximate estimates, not route directions; "
                "opening hours and visit durations are exercise assumptions."
            )
        elif planning.status == "uncertain":
            status = "uncertain"
            response = (
                "This itinerary is conditional, not fully confirmed: one or more stops have "
                "unknown opening hours in the exercise fixture. Walking times are approximate."
            )
        else:
            status = "infeasible"
            reasons = " ".join(planning.explanations)
            response = (
                "The requested itinerary could not be validated within the stated constraints. "
                + reasons
            ).strip()
        if status in {"feasible", "uncertain"}:
            completion_notes = [
                note for note in planning.explanations
                if note.startswith(("Scheduled visits finish", "No further candidate"))
            ]
            if completion_notes:
                response += " " + " ".join(completion_notes)
        return self._result(
            intent,
            state,
            status,
            response,
            planning_result=planning,
            modification_applied=modification_applied,
        )

    @staticmethod
    def _infeasible_planning_result(issue: ValidationIssue) -> PlanningResult:
        return PlanningResult(
            status="infeasible",
            validation=ItineraryValidationResult(is_valid=False, issues=[issue]),
            explanations=[issue.message],
        )

    @staticmethod
    def _add_ambiguity_issues(
        planning: PlanningResult,
        ambiguities: Sequence[str],
    ) -> PlanningResult:
        issues = [
            *planning.validation.issues,
            *[
                ValidationIssue(
                    code="ambiguous_named_poi",
                    message=(
                        f"Required destination {name!r} matches multiple POI records "
                        "and was not selected."
                    ),
                )
                for name in ambiguities
            ],
        ]
        return planning.model_copy(
            update={
                "validation": planning.validation.model_copy(
                    update={"is_valid": False, "issues": issues}
                ),
                "explanations": [
                    *planning.explanations,
                    *[f"Clarify which dataset location is meant by {name!r}." for name in ambiguities],
                ],
            }
        )

    def _answer_or_default(
        self,
        intent: UserIntent,
        state: ConversationState,
        question: str,
        evidence: list[POIEvidence],
        *,
        recommendation: bool,
        recommendation_count: int | None = None,
    ) -> OrchestrationResult:
        if not evidence:
            return self._result(
                intent,
                state,
                "no_evidence",
                "I could not find matching evidence in the supplied Corfu POI dataset.",
                answer_mode="none",
                answer_attempted=False,
                fallback_reason="evidence_insufficient",
            )

        answer_started = time_module.perf_counter()
        response: str
        recommended_poi_ids = (
            self._deterministic_recommendations(evidence, recommendation_count)
            if recommendation
            else []
        )
        selected_evidence = (
            tuple(
                item for item in evidence
                if item.poi_id in set(recommended_poi_ids)
            )
            if recommendation
            else tuple(evidence)
        )
        if recommendation:
            _LOGGER.info(
                "POI recommendation selection completed",
                extra={
                    "candidate_count": len(evidence),
                    "requested_count": recommendation_count,
                    "selected_count": len(recommended_poi_ids),
                    "selected_poi_ids": recommended_poi_ids,
                },
            )
        answer_mode: AnswerMode = "deterministic"
        fallback_reason: AnswerFallbackReason | None = "generator_unavailable"
        citation_status: CitationStatus = "not_attempted"
        if self._answer_generator is not None:
            try:
                generated = self._answer_generator.generate(
                    question,
                    selected_evidence,
                    len(recommended_poi_ids) if recommendation else None,
                )
                answer = GroundedAnswer.model_validate(generated)
                self._validate_grounded_answer(answer, selected_evidence)
                citation_status = "valid"
                if recommendation:
                    if answer.recommended_poi_ids != recommended_poi_ids:
                        raise ValueError(
                            "Generated recommendation selection differs from the authoritative selection"
                        )
                elif answer.recommended_poi_ids:
                    raise ValueError(
                        "Non-recommendation answer included recommended POI references"
                    )
                response = "\n\n".join(claim.text for claim in answer.claims)
                answer_mode = "generated"
                fallback_reason = None
            except Exception as error:
                failure_reason = self._answer_fallback_reason(error)
                if failure_reason == "citation_validation_error":
                    citation_status = "invalid"
                _LOGGER.warning(
                    "Grounded answer generation failed; using evidence-based fallback",
                    extra={
                        "fallback_reason": failure_reason,
                        "evidence_count": len(evidence),
                        "recommendation": recommendation,
                    },
                )
                fallback_reason = failure_reason
                response = self._deterministic_answer(
                    question,
                    selected_evidence,
                    recommendation=recommendation,
                    recommended_poi_ids=recommended_poi_ids,
                )
        else:
            response = self._deterministic_answer(
                question,
                selected_evidence,
                recommendation=recommendation,
                recommended_poi_ids=recommended_poi_ids,
            )
        if not recommendation and answer_mode == "deterministic":
            response = self._add_unsupported_fact_caveat(question, response)
        if (
            recommendation
            and recommendation_count is not None
            and len(recommended_poi_ids) < recommendation_count
        ):
            response = (
                f"I found only {len(recommended_poi_ids)} eligible matching "
                f"place{'s' if len(recommended_poi_ids) != 1 else ''}, fewer than "
                f"the {recommendation_count} requested. {response}"
            )

        _LOGGER.info(
            "Answer stage completed",
            extra={
                "stage": "answer_generation_and_validation",
                "outcome": answer_mode,
                "fallback_reason": fallback_reason,
                "citation_status": citation_status,
                "duration_ms": round((time_module.perf_counter() - answer_started) * 1000, 2),
            },
        )

        return self._result(
            intent,
            state,
            "completed",
            response,
            evidence,
            recommended_poi_ids=recommended_poi_ids,
            answer_mode=answer_mode,
            answer_attempted=self._answer_generator is not None,
            citation_status=citation_status,
            fallback_reason=fallback_reason,
        )

    @staticmethod
    def _answer_fallback_reason(error: Exception) -> AnswerFallbackReason:
        provider_reason = getattr(error, "reason", None)
        if provider_reason in {
            "provider_error",
            "parsing_error",
            "structured_output_error",
        }:
            return provider_reason
        if isinstance(error, ValidationError):
            return "structured_output_error"
        message = str(error).casefold()
        if "recommendation selection differs" in message or "included recommended poi" in message:
            return "recommendation_selection_mismatch"
        if isinstance(error, ValueError):
            return "citation_validation_error"
        return "provider_error"

    @staticmethod
    def _add_unsupported_fact_caveat(question: str, response: str) -> str:
        if not re.search(
            r"\b(?:price|cost|ticket|admission|opening hours?|open|closed|"
            r"available|availability|accessible|access today|currently|right now)\b",
            question,
            re.IGNORECASE,
        ):
            return response
        return (
            "The supplied dataset does not verify current prices, opening hours, "
            "ticket availability, or visitor access. "
            + response
        )

    def _validate_grounded_answer(
        self,
        answer: GroundedAnswer,
        evidence: Sequence[POIEvidence],
    ) -> None:
        evidence_by_id = {item.poi_id: item for item in evidence}
        all_pois = self._poi_repository.get_all()
        cited_poi_ids: set[str] = set()
        for claim in answer.claims:
            if re.search(r"\bCF-\d{3}\b", claim.text, re.IGNORECASE):
                raise ValueError("Generated prose must not expose internal POI IDs")
            if _URL_IN_TEXT.search(claim.text):
                raise ValueError("Generated prose must not contain URLs")
            evidenced_name_spans = [
                (match.start(), match.end())
                for poi in all_pois
                if poi.id in evidence_by_id
                for match in re.finditer(
                    rf"(?<!\w){re.escape(poi.name)}(?!\w)",
                    claim.text,
                    re.IGNORECASE,
                )
            ]
            for poi in all_pois:
                if poi.id in evidence_by_id:
                    continue
                for match in re.finditer(
                    rf"(?<!\w){re.escape(poi.name)}(?!\w)",
                    claim.text,
                    re.IGNORECASE,
                ):
                    nested_in_evidenced_name = any(
                        start <= match.start() and match.end() <= end
                        for start, end in evidenced_name_spans
                    )
                    if not nested_in_evidenced_name:
                        raise ValueError(
                            "Generated prose mentioned a POI outside selected evidence"
                        )

            supporting_quotes: list[str] = []
            for citation in claim.citations:
                item = evidence_by_id.get(citation.poi_id)
                if item is None:
                    raise ValueError("Generated answer cited a POI outside retrieved evidence")
                quote = " ".join(citation.supporting_quote.casefold().split())
                description = " ".join(item.description.casefold().split())
                if len(quote.split()) < 2 or quote not in description:
                    raise ValueError("Generated answer citation has no exact evidence quote")
                supporting_quotes.append(quote)
                cited_poi_ids.add(citation.poi_id)

            cited_text = " ".join(supporting_quotes)
            if not _FACT_GAP_DISCLAIMER.search(claim.text):
                for amount in _PRICE_AMOUNT.findall(claim.text):
                    if amount.casefold() not in cited_text:
                        raise ValueError("Generated answer contains an unsupported price")
                for access_term in _SENSITIVE_ACCESS_CLAIM.findall(claim.text):
                    if re.search(rf"\b{re.escape(access_term.casefold())}\b", cited_text) is None:
                        raise ValueError("Generated answer contains an unsupported access claim")

        if len(set(answer.recommended_poi_ids)) != len(answer.recommended_poi_ids):
            raise ValueError("Generated answer contains duplicate recommendation references")
        if any(poi_id not in evidence_by_id for poi_id in answer.recommended_poi_ids):
            raise ValueError("Generated answer selected a POI outside retrieved evidence")
        if any(poi_id not in cited_poi_ids for poi_id in answer.recommended_poi_ids):
            raise ValueError("Generated recommendation is not referenced by a grounded claim")

    @staticmethod
    def _recommendation_target_count(
        evidence: Sequence[POIEvidence],
        requested_count: int | None,
    ) -> int:
        desired_count = requested_count if requested_count is not None else 3
        return min(desired_count, len(evidence))

    @classmethod
    def _deterministic_recommendations(
        cls,
        evidence: Sequence[POIEvidence],
        requested_count: int | None,
    ) -> list[str]:
        target_count = cls._recommendation_target_count(evidence, requested_count)
        return [item.poi_id for item in evidence[:target_count]]

    @staticmethod
    def _deterministic_answer(
        question: str,
        evidence: Sequence[POIEvidence],
        *,
        recommendation: bool,
        recommended_poi_ids: Sequence[str] = (),
    ) -> str:
        if recommendation:
            recommended_ids = set(recommended_poi_ids)
            names = ", ".join(
                item.name for item in evidence if item.poi_id in recommended_ids
            )
            return (
                "Based on the available place descriptions, you could explore "
                f"{names}."
            )

        if len(evidence) > 1 and re.search(
            r"\b(compare|comparison|difference|different|versus|vs\.?)\b",
            question,
            re.IGNORECASE,
        ):
            details = " ".join(
                f"{item.name}: {Orchestrator._first_description_sentence(item.description)}"
                for item in evidence
            )
            return (
                f"The available descriptions distinguish these places as follows: "
                f"{details}"
            )

        return "\n\n".join(
            f"{item.name}: {Orchestrator._first_description_sentence(item.description)}"
            for item in evidence
        )

    @staticmethod
    def _first_description_sentence(description: str) -> str:
        return re.split(r"(?<=[.!?])\s+", description.strip(), maxsplit=1)[0]

    def _exact_name_matches(self, question: str) -> list[POI]:
        normalized_question = question.casefold()
        candidates = sorted(
            (
                poi
                for poi in self._poi_repository.get_all()
                if re.search(
                    rf"(?<!\w){re.escape(poi.name.casefold())}(?!\w)",
                    normalized_question,
                )
            ),
            key=lambda poi: (-len(poi.name), poi.id),
        )
        matches: list[POI] = []
        occupied: list[tuple[int, int]] = []
        for poi in candidates:
            match = re.search(
                rf"(?<!\w){re.escape(poi.name.casefold())}(?!\w)",
                normalized_question,
            )
            if match is None:
                continue
            interval = (match.start(), match.end())
            if any(interval[0] < end and start < interval[1] for start, end in occupied):
                continue
            matches.append(poi)
            occupied.append(interval)
        return matches

    def _required_poi(self, poi_id: str) -> POI:
        poi = self._poi_repository.get_by_id(poi_id)
        if poi is None:
            raise OrchestrationError(
                f"Repository returned an unknown POI ID: {poi_id}"
            )
        return poi

    @staticmethod
    def _evidence(
        poi: POI,
        method: EvidenceMethod,
        relevance_score: float | None = None,
    ) -> POIEvidence:
        return POIEvidence(
            poi_id=poi.id,
            name=poi.name,
            description=poi.description,
            source_urls=list(poi.source_urls),
            description_provenance=poi.description_provenance,
            method=method,
            relevance_score=relevance_score,
        )

    @staticmethod
    def _result(
        intent: UserIntent,
        state: ConversationState,
        status: ResponseStatus,
        response: str,
        evidence: list[POIEvidence] | None = None,
        *,
        recommended_poi_ids: list[str] | None = None,
        planning_result: PlanningResult | None = None,
        modification_applied: bool | None = None,
        weather_forecast: WeatherForecast | None = None,
        rain_assessment: RainAssessment | None = None,
        weather_replanning: WeatherReplanningResult | None = None,
        weather_error: str | None = None,
        answer_mode: AnswerMode = "none",
        answer_attempted: bool = False,
        citation_status: CitationStatus = "not_attempted",
        fallback_reason: AnswerFallbackReason | None = None,
    ) -> OrchestrationResult:
        relevant_evidence = evidence or []
        return OrchestrationResult(
            conversation_id=state.conversation_id,
            intent=intent.model_copy(deep=True),
            status=status,
            response=response,
            evidence=[item.model_copy(deep=True) for item in relevant_evidence],
            relevant_poi_ids=[item.poi_id for item in relevant_evidence],
            recommended_poi_ids=list(recommended_poi_ids or []),
            answer_mode=answer_mode,
            answer_attempted=answer_attempted,
            citation_status=citation_status,
            fallback_reason=fallback_reason,
            planning_result=(
                planning_result.model_copy(deep=True)
                if planning_result is not None
                else None
            ),
            modification_applied=modification_applied,
            weather_forecast=(
                weather_forecast.model_copy(deep=True)
                if weather_forecast is not None
                else None
            ),
            rain_assessment=(
                rain_assessment.model_copy(deep=True)
                if rain_assessment is not None
                else None
            ),
            weather_replanning=(
                weather_replanning.model_copy(deep=True)
                if weather_replanning is not None
                else None
            ),
            weather_error=weather_error,
            conversation_state=state.model_copy(deep=True),
        )

    @staticmethod
    def _unsupported_message(intent: UserIntent) -> str:
        descriptions = {
            "itinerary_creation": "Itinerary planning",
            "itinerary_modification": "Itinerary modification execution",
            "live_weather_query": "Live weather retrieval",
            "weather_based_replanning": "Weather-based itinerary replanning",
        }
        name = descriptions[intent.intent]
        if intent.intent == "itinerary_modification":
            return (
                f"{name} is not implemented yet. The typed modification remains pending "
                "for the Planner; no stops were changed."
            )
        return f"{name} is not implemented yet. No result was fabricated."
