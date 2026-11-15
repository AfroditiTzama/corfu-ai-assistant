from __future__ import annotations

import re
from datetime import date, datetime, timezone
from typing import Annotated, Literal
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator


class DomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _validate_timezone(value: str) -> str:
    try:
        ZoneInfo(value)
    except ZoneInfoNotFoundError as error:
        raise ValueError(f"Unknown IANA timezone: {value}") from error
    return value


def _validate_child_ages(ages: list[int]) -> list[int]:
    if any(age < 0 or age > 17 for age in ages):
        raise ValueError("Child ages must be between 0 and 17")
    return ages


Setting = Literal["indoor", "outdoor", "mixed"]
Weekday = Literal["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
FamilySuitabilityFilter = Literal[True, False, "unknown"]
PatchAction = Literal["set", "clear", "preserve"]
LocalClockTime = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$"),
]


class Location(DomainModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class OpeningWindow(DomainModel):
    # End minutes > 1440 are an offset into the next calendar day.
    start_minute: int = Field(ge=0, le=1439)
    end_minute: int = Field(ge=1, le=2879)

    @model_validator(mode="before")
    @classmethod
    def parse_fixture_range(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        match = re.fullmatch(r"(\d{2}):(\d{2})-(\d{2}):(\d{2})", value)
        if match is None:
            raise ValueError("Opening window must use HH:MM-HH:MM")

        start_hour, start_minute, end_hour, end_minute = map(int, match.groups())
        if start_hour > 23 or end_hour > 24 or start_minute > 59 or end_minute > 59:
            raise ValueError("Opening window contains an invalid time")
        if end_hour == 24 and end_minute != 0:
            raise ValueError("24:00 is the only supported time using hour 24")

        start = start_hour * 60 + start_minute
        end = end_hour * 60 + end_minute
        if end == start:
            raise ValueError("Opening window must have a positive duration")
        if end < start:
            end += 1440
        return {"start_minute": start, "end_minute": end}

    @model_validator(mode="after")
    def validate_interval(self) -> OpeningWindow:
        # Enforce invariants for all inputs, including direct dict/model construction.
        if not self.start_minute < self.end_minute <= self.start_minute + 1440:
            raise ValueError("Opening window duration must be > 0 and <= 24 hours")
        return self

    @property
    def crosses_midnight(self) -> bool:
        return self.end_minute > 1440 or (
            self.end_minute == 1440 and self.start_minute > 0
        )


class WeekdaySchedule(DomainModel):
    mon: OpeningWindow | None
    tue: OpeningWindow | None
    wed: OpeningWindow | None
    thu: OpeningWindow | None
    fri: OpeningWindow | None
    sat: OpeningWindow | None
    sun: OpeningWindow | None


class POI(DomainModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    area: str
    category: str
    tags: list[str]
    location: Location
    description: str
    setting: Setting
    suggested_visit_minutes: int = Field(gt=0)
    opening_hours_fixture: WeekdaySchedule | None
    source_urls: list[str]
    family_suitability_fixture: bool | None
    description_provenance: str


class POIMetadataFilters(DomainModel):
    category: str | None = Field(default=None, min_length=1)
    tags: list[str] = Field(default_factory=list)
    excluded_categories: list[str] = Field(default_factory=list)
    excluded_tags: list[str] = Field(default_factory=list)
    excluded_poi_ids: list[str] = Field(default_factory=list)
    setting: Setting | None = None
    family_suitability_fixture: FamilySuitabilityFilter | None = None

    @field_validator("category")
    @classmethod
    def normalize_category(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("category must not be blank")
        return normalized

    @field_validator("tags", "excluded_tags", "excluded_categories")
    @classmethod
    def normalize_filter_values(cls, value: list[str]) -> list[str]:
        normalized = [tag.strip() for tag in value]
        if any(not item for item in normalized):
            raise ValueError("Filter values must not contain blank values")
        if len({tag.casefold() for tag in normalized}) != len(normalized):
            raise ValueError("Filter values must be unique")
        return normalized

    @field_validator("excluded_poi_ids")
    @classmethod
    def validate_excluded_poi_ids(cls, value: list[str]) -> list[str]:
        normalized = [poi_id.strip() for poi_id in value]
        if any(not poi_id for poi_id in normalized):
            raise ValueError("excluded_poi_ids must not contain blank values")
        if len(set(normalized)) != len(normalized):
            raise ValueError("excluded_poi_ids must be unique")
        return normalized


class HybridSearchRequest(DomainModel):
    query: str = Field(min_length=1)
    filters: POIMetadataFilters = Field(default_factory=POIMetadataFilters)
    top_k: int = Field(default=5, gt=0, strict=True)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("query must not be blank")
        return normalized


TravelMode = Literal["walking", "driving", "cycling", "public_transit"]
IntentKind = Literal[
    "tourism_qa",
    "poi_recommendation",
    "itinerary_creation",
    "itinerary_modification",
    "live_weather_query",
    "weather_based_replanning",
    "clarification",
]


class IntentTimeWindow(DomainModel):
    start_time: LocalClockTime | None = None
    end_time: LocalClockTime | None = None

    @model_validator(mode="after")
    def validate_order(self) -> IntentTimeWindow:
        if (
            self.start_time is not None
            and self.end_time is not None
            and self.end_time <= self.start_time
        ):
            raise ValueError("end_time must be later than start_time")
        return self


class IntentUpdateActions(DomainModel):
    requested_date: PatchAction
    start_time: PatchAction
    end_time: PatchAction
    timezone: PatchAction
    semantic_interests: PatchAction
    filter_category: PatchAction
    filter_tags: PatchAction
    filter_setting: PatchAction
    filter_family_suitability: PatchAction
    filter_excluded_categories: PatchAction
    filter_excluded_tags: PatchAction
    filter_excluded_poi_ids: PatchAction
    travel_mode: PatchAction
    family_required: PatchAction
    child_ages: PatchAction
    itinerary_modification: PatchAction
    question: PatchAction
    clarification_question: PatchAction
    weather_request: PatchAction


class StatePatchActions(DomainModel):
    requested_date: PatchAction
    start_time: PatchAction
    end_time: PatchAction
    timezone: PatchAction
    interests: PatchAction
    excluded_categories: PatchAction
    excluded_tags: PatchAction
    preferred_setting: PatchAction
    family_required: PatchAction
    child_ages: PatchAction
    replace_stop: PatchAction


WeatherAspect = Literal[
    "precipitation",
    "temperature",
    "wind",
    "general_conditions",
]
RelativeWeatherHorizon = Literal["next_24_hours", "tomorrow", "today", "tonight"]


class WeatherRequestScope(DomainModel):
    """Requested forecast scope; relative ranges are resolved outside the LLM."""

    relative_horizon: RelativeWeatherHorizon | None = None
    duration_hours: int | None = Field(default=None, ge=1, le=168, strict=True)
    forecast_date: date | None = None
    aspects: list[WeatherAspect] = Field(default_factory=list)
    start_time: LocalClockTime | None = None
    end_time: LocalClockTime | None = None
    activity_recommendation: bool = False

    @field_validator("aspects")
    @classmethod
    def validate_unique_aspects(cls, value: list[WeatherAspect]) -> list[WeatherAspect]:
        if len(set(value)) != len(value):
            raise ValueError("weather aspects must be unique")
        return value

    @model_validator(mode="after")
    def validate_relative_duration(self) -> WeatherRequestScope:
        if self.relative_horizon == "next_24_hours" and self.duration_hours not in (None, 24):
            raise ValueError("next_24_hours horizon must have duration_hours 24")
        if self.relative_horizon == "tomorrow" and self.duration_hours is not None:
            raise ValueError("tomorrow horizon cannot specify a fixed duration_hours")
        if self.relative_horizon is not None and self.forecast_date is not None:
            raise ValueError("relative_horizon and forecast_date are separate alternatives")
        if (self.start_time is None) != (self.end_time is None):
            raise ValueError("weather start_time and end_time must be supplied together")
        if self.start_time is not None and self.end_time <= self.start_time:
            raise ValueError("weather end_time must be later than start_time")
        return self


class SemanticPreferenceUpdate(DomainModel):
    """Semantic interests; intentionally distinct from deterministic POI filters."""

    interests: list[str] | None = None

    @field_validator("interests")
    @classmethod
    def validate_interests(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        cleaned = [interest.strip() for interest in value]
        if any(not interest for interest in cleaned):
            raise ValueError("interests must not contain blank values")
        return cleaned


class IntentHardFilters(DomainModel):
    """Only explicitly extracted hard filters; omitted values remain absent."""

    category: str | None = None
    tags: list[str] | None = None
    setting: Setting | None = None
    family_suitability_fixture: FamilySuitabilityFilter | None = None
    excluded_categories: list[str] | None = None
    excluded_tags: list[str] | None = None
    excluded_poi_ids: list[str] | None = None

    @field_validator("category")
    @classmethod
    def normalize_category(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("category must not be blank")
        return normalized

    @field_validator("tags")
    @classmethod
    def validate_optional_tags(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        cleaned = [tag.strip() for tag in value]
        if any(not tag for tag in cleaned):
            raise ValueError("tags must not contain blank values")
        if len({tag.casefold() for tag in cleaned}) != len(cleaned):
            raise ValueError("tags must be unique")
        return cleaned

    @field_validator("excluded_categories", "excluded_tags", "excluded_poi_ids")
    @classmethod
    def validate_exclusions(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        cleaned = [item.strip() for item in value]
        if any(not item for item in cleaned):
            raise ValueError("Exclusion values must not contain blank values")
        if len({item.casefold() for item in cleaned}) != len(cleaned):
            raise ValueError("Exclusion values must be unique")
        return cleaned


class FamilyRequirementsUpdate(DomainModel):
    required: bool | None = None
    child_ages: list[int] | None = Field(default=None, max_length=20)

    @field_validator("child_ages")
    @classmethod
    def validate_child_ages(cls, value: list[int] | None) -> list[int] | None:
        return _validate_child_ages(value) if value is not None else None


class ItineraryModification(DomainModel):
    action: Literal[
        "replace_stop",
        "remove_stop",
        "add_stop",
        "change_order",
        "change_constraints",
    ]
    stop_index: int | None = Field(default=None, ge=0)
    requested_replacement: str | None = None
    preserve_other_stops: bool | None = None

    @field_validator("requested_replacement")
    @classmethod
    def normalize_requested_replacement(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("requested_replacement must not be blank")
        return normalized


class ConstraintEvidence(DomainModel):
    """Current-turn provenance, never evidence inferred from stored interests."""
    field: Literal[
        "filter_category", "filter_tags", "filter_setting", "filter_family_suitability",
        "filter_excluded_categories", "filter_excluded_tags", "filter_excluded_poi_ids",
    ]
    value: str = Field(min_length=1)
    kind: Literal["required", "literal_tag", "excluded", "clear", "ambiguous"]
    quote: str = Field(min_length=1, max_length=500)


class UserIntent(DomainModel):
    """Structured extraction only; never treat model_fields_set as user intent.

    ``field_actions`` is the explicit update contract. Set values carry a value
    (an empty list is a valid explicit set); clear and preserve carry null.
    """

    intent: IntentKind
    field_actions: IntentUpdateActions
    constraint_evidence: list[ConstraintEvidence] | None = Field(
        default=None, max_length=50,
        description="Current-request evidence for new metadata constraints. Return [] when none."
    )
    question: str | None = None
    recommendation_count: int | None = Field(default=None, ge=1, le=50, strict=True)
    requested_date: date | None = None
    time_window: IntentTimeWindow | None = None
    timezone: str | None = None
    semantic_preferences: SemanticPreferenceUpdate | None = None
    hard_filters: IntentHardFilters | None = None
    travel_mode: TravelMode | None = None
    family_requirements: FamilyRequirementsUpdate | None = None
    itinerary_modification: ItineraryModification | None = None
    clarification_question: str | None = None
    weather_request: WeatherRequestScope | None = None
    unsupported_actions: list[Literal["booking", "email", "payment", "purchase", "contact"]] = Field(default_factory=list)

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str | None) -> str | None:
        return _validate_timezone(value) if value is not None else None

    @field_validator("question", "clarification_question")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("Text fields must not be blank when supplied")
        return normalized

    @model_validator(mode="after")
    def validate_intent_specific_fields(self) -> UserIntent:
        if self.intent == "clarification" and self.clarification_question is None:
            raise ValueError("clarification intent requires clarification_question")
        if self.intent == "itinerary_modification" and self.itinerary_modification is None:
            raise ValueError("itinerary_modification intent requires a modification")
        if self.recommendation_count is not None and self.intent != "poi_recommendation":
            raise ValueError("recommendation_count is only valid for POI recommendations")
        self._validate_field_actions()
        return self

    def _validate_field_actions(self) -> None:
        values: dict[str, object | None] = {
            "requested_date": self.requested_date,
            "start_time": self.time_window.start_time if self.time_window else None,
            "end_time": self.time_window.end_time if self.time_window else None,
            "timezone": self.timezone,
            "semantic_interests": (
                self.semantic_preferences.interests if self.semantic_preferences else None
            ),
            "filter_category": self.hard_filters.category if self.hard_filters else None,
            "filter_tags": self.hard_filters.tags if self.hard_filters else None,
            "filter_setting": self.hard_filters.setting if self.hard_filters else None,
            "filter_family_suitability": (
                self.hard_filters.family_suitability_fixture if self.hard_filters else None
            ),
            "filter_excluded_categories": (
                self.hard_filters.excluded_categories if self.hard_filters else None
            ),
            "filter_excluded_tags": (
                self.hard_filters.excluded_tags if self.hard_filters else None
            ),
            "filter_excluded_poi_ids": (
                self.hard_filters.excluded_poi_ids if self.hard_filters else None
            ),
            "travel_mode": self.travel_mode,
            "family_required": (
                self.family_requirements.required if self.family_requirements else None
            ),
            "child_ages": (
                self.family_requirements.child_ages if self.family_requirements else None
            ),
            "itinerary_modification": self.itinerary_modification,
            "question": self.question,
            "clarification_question": self.clarification_question,
            "weather_request": self.weather_request,
        }
        for field_name, action in self.field_actions.model_dump().items():
            value = values[field_name]
            if action == "set" and value is None:
                raise ValueError(f"{field_name} action 'set' requires a value")
            if action != "set" and value is not None:
                raise ValueError(f"{field_name} action '{action}' must not include a value")

    @property
    def effective_timezone(self) -> str:
        return self.timezone or "Europe/Athens"


class RetrievalResult(DomainModel):
    poi_id: str = Field(min_length=1)
    relevance_score: float = Field(ge=-1, le=1)
    source_urls: list[str]
    description_provenance: str


class FixtureFieldNotes(DomainModel):
    opening_hours_fixture: str
    setting: str
    suggested_visit_minutes: str
    family_suitability_fixture: str
    source_urls: str
    area: str
    description_provenance: str


class CorfuDataset(DomainModel):
    dataset: str
    version: str
    notice: str
    timezone: str
    fields_note: FixtureFieldNotes
    pois: list[POI]

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        return _validate_timezone(value)

    @model_validator(mode="after")
    def validate_unique_poi_ids(self) -> CorfuDataset:
        poi_ids = [poi.id for poi in self.pois]
        if len(poi_ids) != len(set(poi_ids)):
            raise ValueError("POI ids must be unique")
        return self


class HourlyWeather(DomainModel):
    """Raw fixture hour: local wall-clock time without a UTC offset."""

    local_time: datetime
    temperature_c: float = Field(allow_inf_nan=False)
    precipitation_mm: float = Field(ge=0, allow_inf_nan=False)
    weather_code: int = Field(ge=0, le=99)

    @field_validator("local_time")
    @classmethod
    def require_local_naive_datetime(cls, value: datetime) -> datetime:
        if value.tzinfo is not None and value.utcoffset() is not None:
            raise ValueError("Fixture local_time must be a timezone-naive local timestamp")
        return value


class ForecastHour(DomainModel):
    """Provider-independent weather values and optional precipitation interval."""

    at: datetime
    temperature_c: float | None = Field(default=None, allow_inf_nan=False)
    precipitation_mm: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
    )
    precipitation_probability_percent: float | None = Field(
        default=None,
        ge=0,
        le=100,
        allow_inf_nan=False,
    )
    weather_code: int | None = Field(default=None, ge=0, le=99)
    precipitation_interval_start: datetime | None = None
    precipitation_interval_end: datetime | None = None

    @field_validator("at", "precipitation_interval_start", "precipitation_interval_end")
    @classmethod
    def require_aware_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("Forecast timestamps must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_precipitation_interval(self) -> ForecastHour:
        start = self.precipitation_interval_start
        end = self.precipitation_interval_end
        if (start is None) != (end is None):
            raise ValueError("Both precipitation interval boundaries must be supplied")
        if start is not None and end is not None:
            if end.astimezone(timezone.utc) <= start.astimezone(timezone.utc):
                raise ValueError("Precipitation interval end must be later than start")
        return self


class WeatherForecast(DomainModel):
    """Common contract emitted by either a real weather API or a fixture adapter."""

    place: str
    timezone: str
    source: Literal["live", "synthetic_fixture"]
    hourly: list[ForecastHour] = Field(min_length=1)
    retrieved_at: datetime | None = None

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        return _validate_timezone(value)

    @field_validator("retrieved_at")
    @classmethod
    def require_aware_retrieval_time(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("Forecast retrieval timestamp must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_hourly_order(self) -> WeatherForecast:
        times = [entry.at.astimezone(timezone.utc) for entry in self.hourly]
        if times != sorted(times):
            raise ValueError("Forecast hours must be chronological")
        if len(times) != len(set(times)):
            raise ValueError("Forecast hourly timestamps must be unique")
        return self


class WeatherFixture(DomainModel):
    """Strict schema of the supplied synthetic weather JSON, not the live API."""

    fixture_type: Literal["synthetic_weather_for_reproducible_test_only"]
    is_live: Literal[False]
    place: str
    timezone: str
    date: date
    scenario: str
    hourly: list[HourlyWeather] = Field(min_length=1)

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        validated = _validate_timezone(value)
        if validated != "Europe/Athens":
            raise ValueError("Synthetic Corfu weather fixture timezone must be Europe/Athens")
        return validated

    @model_validator(mode="after")
    def validate_hourly_order(self) -> WeatherFixture:
        times = [entry.local_time for entry in self.hourly]
        if times != sorted(times):
            raise ValueError("Hourly weather records must be chronological")
        if len(times) != len(set(times)):
            raise ValueError("Hourly weather timestamps must be unique")
        if any(entry.local_time.date() != self.date for entry in self.hourly):
            raise ValueError("Fixture hourly entries must match fixture date")
        return self

    def to_forecast(self, *, retrieved_at: datetime | None = None) -> WeatherForecast:
        """Convert the supplied test fixture to a normalized forecast.

        The supplied 2026-10-07 fixture is not in a DST transition. Other
        ambiguous/nonexistent local times require an explicit policy before
        a fixture for such dates could be normalized safely.
        """
        zone = ZoneInfo(self.timezone)
        return WeatherForecast(
            place=self.place,
            timezone=self.timezone,
            source="synthetic_fixture",
            hourly=[
                ForecastHour(
                    at=hour.local_time.replace(tzinfo=zone),
                    temperature_c=hour.temperature_c,
                    precipitation_mm=hour.precipitation_mm,
                    weather_code=hour.weather_code,
                )
                for hour in self.hourly
            ],
            retrieved_at=retrieved_at,
        )


class UserRequest(DomainModel):
    message: str = Field(min_length=1)
    requested_date: date | None = None
    start_time: LocalClockTime | None = None
    end_time: LocalClockTime | None = None
    timezone: str = "Europe/Athens"
    interests: list[str] = Field(default_factory=list)
    setting: Setting | None = None
    child_ages: list[int] = Field(default_factory=list, max_length=20)

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        return _validate_timezone(value)

    @field_validator("child_ages")
    @classmethod
    def validate_child_ages(cls, value: list[int]) -> list[int]:
        return _validate_child_ages(value)

    @model_validator(mode="after")
    def validate_time_window(self) -> UserRequest:
        if self.start_time is not None and self.end_time is not None:
            if self.end_time <= self.start_time:
                raise ValueError("end_time must be later than start_time")
        return self


class ItineraryStop(DomainModel):
    poi_id: str = Field(min_length=1)
    starts_at: datetime
    ends_at: datetime
    reason: str | None = None

    @model_validator(mode="after")
    def validate_interval(self) -> ItineraryStop:
        if self.starts_at.tzinfo is None or self.starts_at.utcoffset() is None:
            raise ValueError("Itinerary stop timestamps must include a timezone")
        if self.ends_at.tzinfo is None or self.ends_at.utcoffset() is None:
            raise ValueError("Itinerary stop timestamps must include a timezone")
        if self.ends_at <= self.starts_at:
            raise ValueError("Itinerary stop must have a positive duration")
        return self


class Itinerary(DomainModel):
    conversation_id: UUID | None = None
    timezone: str = "Europe/Athens"
    starts_at: datetime
    ends_at: datetime
    stops: list[ItineraryStop] = Field(default_factory=list)

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        return _validate_timezone(value)

    @model_validator(mode="after")
    def validate_schedule(self) -> Itinerary:
        if self.starts_at.tzinfo is None or self.starts_at.utcoffset() is None:
            raise ValueError("Itinerary timestamps must include a timezone")
        if self.ends_at.tzinfo is None or self.ends_at.utcoffset() is None:
            raise ValueError("Itinerary timestamps must include a timezone")
        if self.ends_at <= self.starts_at:
            raise ValueError("Itinerary must have a positive duration")

        previous_end = self.starts_at
        for stop in self.stops:
            if stop.starts_at < self.starts_at or stop.ends_at > self.ends_at:
                raise ValueError("Itinerary stop must fit within the itinerary window")
            if stop.starts_at < previous_end:
                raise ValueError("Itinerary stops must be chronological and non-overlapping")
            previous_end = stop.ends_at
        return self


class TravelPreferences(DomainModel):
    interests: list[str] = Field(default_factory=list)
    excluded_categories: list[str] = Field(default_factory=list)
    excluded_tags: list[str] = Field(default_factory=list)
    preferred_setting: Setting | None = None
    family_required: bool | None = None
    child_ages: list[int] = Field(default_factory=list, max_length=20)

    @field_validator("child_ages")
    @classmethod
    def validate_child_ages(cls, value: list[int]) -> list[int]:
        return _validate_child_ages(value)


class TravelPreferencesPatch(DomainModel):
    """Legacy values container; future merging must use StatePatch.actions."""

    interests: list[str] | None = None
    excluded_categories: list[str] | None = None
    excluded_tags: list[str] | None = None
    preferred_setting: Setting | None = None
    family_required: bool | None = None
    child_ages: list[int] | None = Field(default=None, max_length=20)

    @field_validator("child_ages")
    @classmethod
    def validate_child_ages(cls, value: list[int] | None) -> list[int] | None:
        return _validate_child_ages(value) if value is not None else None


class PendingClarification(DomainModel):
    """Session-local context needed to interpret a short clarification reply."""

    original_message: str = Field(min_length=1, max_length=4_000)
    question: str = Field(min_length=1, max_length=1_000)

    @field_validator("original_message", "question")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Clarification context must not be blank")
        return normalized


class StopReplacement(DomainModel):
    """Zero-based position; the replanner selects a replacement when none is specified."""

    stop_index: int = Field(ge=0)
    replacement_poi_id: str | None = None
    preserve_other_stops: bool = True


class StatePatch(DomainModel):
    """Legacy patch values plus explicit optional actions for future merging.

    When ``actions`` is provided, set requires a corresponding value, clear
    requires null, and preserve requires null. Empty lists may be set explicitly.
    Without actions, these legacy fields carry no merge semantics.
    """

    new_plan: bool = False
    preferences: TravelPreferencesPatch | None = None
    requested_date: date | None = None
    start_time: LocalClockTime | None = None
    end_time: LocalClockTime | None = None
    timezone: str | None = None
    replace_stop: StopReplacement | None = None
    actions: StatePatchActions | None = None

    @field_validator("timezone")
    @classmethod
    def validate_optional_timezone(cls, value: str | None) -> str | None:
        return _validate_timezone(value) if value is not None else None

    @model_validator(mode="after")
    def validate_time_window(self) -> StatePatch:
        if (
            self.start_time is not None
            and self.end_time is not None
            and self.end_time <= self.start_time
        ):
            raise ValueError("end_time must be later than start_time")
        return self

    @model_validator(mode="after")
    def validate_patch_actions(self) -> StatePatch:
        if self.actions is None:
            return self
        preference_values = self.preferences
        values: dict[str, object | None] = {
            "requested_date": self.requested_date,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "timezone": self.timezone,
            "interests": preference_values.interests if preference_values else None,
            "excluded_categories": (
                preference_values.excluded_categories if preference_values else None
            ),
            "excluded_tags": preference_values.excluded_tags if preference_values else None,
            "preferred_setting": (
                preference_values.preferred_setting if preference_values else None
            ),
            "family_required": (
                preference_values.family_required if preference_values else None
            ),
            "child_ages": preference_values.child_ages if preference_values else None,
            "replace_stop": self.replace_stop,
        }
        for field_name, action in self.actions.model_dump().items():
            value = values[field_name]
            if action == "set" and value is None:
                raise ValueError(f"{field_name} action 'set' requires a value")
            if action != "set" and value is not None:
                raise ValueError(f"{field_name} action '{action}' must not include a value")
        return self


class ConversationState(DomainModel):
    conversation_id: UUID = Field(default_factory=uuid4)
    timezone: str = "Europe/Athens"
    preferences: TravelPreferences = Field(default_factory=TravelPreferences)
    hard_filters: IntentHardFilters = Field(default_factory=IntentHardFilters)
    travel_mode: TravelMode | None = None
    active_request: UserRequest | None = None
    itinerary: Itinerary | None = None
    pending_modification: ItineraryModification | None = None
    pending_clarification: PendingClarification | None = None
    replacement_family_requirements: FamilyRequirementsUpdate | None = None
    last_modification_outcome: Literal["applied", "rejected"] | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        return _validate_timezone(value)

    @field_validator("updated_at")
    @classmethod
    def require_aware_updated_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("updated_at must include a timezone")
        return value


class ValidationIssue(DomainModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    poi_id: str | None = None


class ItineraryValidationResult(DomainModel):
    is_valid: bool
    issues: list[ValidationIssue] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class OpeningHoursEligibility(DomainModel):
    status: Literal["unknown", "closed", "scheduled"]
    opening_window: OpeningWindow | None = None

    @model_validator(mode="after")
    def validate_status_window(self) -> OpeningHoursEligibility:
        if (self.status == "scheduled") != (self.opening_window is not None):
            raise ValueError("Only scheduled opening hours may include an opening window")
        return self
