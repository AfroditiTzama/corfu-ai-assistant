from __future__ import annotations

import math
import logging
import json
from datetime import date, datetime, time, timedelta, timezone
from typing import Literal, Protocol, Sequence, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, field_validator, model_validator

from corfu_assistant.models import (
    DomainModel,
    Itinerary,
    ItineraryStop,
    ItineraryValidationResult,
    LocalClockTime,
    POI,
    POIMetadataFilters,
    TravelMode,
    ValidationIssue,
    Weekday,
)
from corfu_assistant.opening_hours import opening_hours_eligibility
from corfu_assistant.repository import InvalidMetadataFilter, POIRepository


PlanStatus = Literal["feasible", "infeasible", "uncertain"]
_LOGGER = logging.getLogger(__name__)
UnknownHoursPolicy = Literal["conditional", "exclude"]


class PlanningRequest(DomainModel):
    requested_date: date
    start_time: LocalClockTime
    end_time: LocalClockTime
    timezone: str = "Europe/Athens"
    travel_mode: TravelMode = "walking"
    hard_filters: POIMetadataFilters = Field(default_factory=POIMetadataFilters)
    family_required: bool = False
    semantic_interests: list[str] = Field(default_factory=list)
    ranked_candidate_ids: list[str] = Field(default_factory=list)
    preferred_poi_ids: list[str] = Field(default_factory=list)
    required_poi_ids: list[str] = Field(default_factory=list, max_length=8)
    preserved_stops: list[ItineraryStop] = Field(default_factory=list)
    replacement_stop_index: int | None = Field(default=None, ge=0)
    replacement_hard_filters: POIMetadataFilters | None = None
    replacement_family_required: bool = False
    replacement_required_poi_ids: list[str] = Field(default_factory=list, max_length=8)
    max_stops: int = Field(default=6, ge=1, le=20)
    unknown_hours_policy: UnknownHoursPolicy = "conditional"
    allow_duplicate_stops: bool = False

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as error:
            raise ValueError(f"Unknown IANA timezone: {value}") from error
        return value

    @field_validator("semantic_interests")
    @classmethod
    def validate_interests(cls, value: list[str]) -> list[str]:
        cleaned = [interest.strip() for interest in value]
        if any(not interest for interest in cleaned):
            raise ValueError("semantic_interests must not contain blank values")
        return cleaned

    @field_validator("required_poi_ids", "ranked_candidate_ids", "preferred_poi_ids")
    @classmethod
    def validate_unique_ids(cls, value: list[str]) -> list[str]:
        if any(not poi_id.strip() for poi_id in value):
            raise ValueError("POI IDs must not be blank")
        if len(set(value)) != len(value):
            raise ValueError("POI IDs must be unique")
        return value

    @model_validator(mode="after")
    def validate_time_window(self) -> PlanningRequest:
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be later than start_time")
        if self.family_required and self.hard_filters.family_suitability_fixture is False:
            raise ValueError(
                "family_required conflicts with family_suitability_fixture=False"
            )
        if self.replacement_stop_index is not None:
            if self.replacement_stop_index >= len(self.preserved_stops):
                raise ValueError("replacement_stop_index must identify a preserved stop")
            if self.required_poi_ids:
                raise ValueError("replacement requests must use replacement_required_poi_ids")
            if (
                self.replacement_family_required
                and self.replacement_hard_filters is not None
                and self.replacement_hard_filters.family_suitability_fixture is False
            ):
                raise ValueError(
                    "replacement family requirement conflicts with family_suitability_fixture=False"
                )
        elif self.replacement_hard_filters is not None or self.replacement_family_required:
            raise ValueError("replacement constraints require replacement_stop_index")
        return self


class TravelEstimate(DomainModel):
    straight_line_distance_km: float = Field(ge=0)
    estimated_walking_distance_km: float = Field(ge=0)
    minutes: int = Field(ge=0)


class WalkingTransfer(DomainModel):
    from_poi_id: str
    to_poi_id: str
    starts_at: datetime
    ends_at: datetime
    estimate: TravelEstimate


class WaitingPeriod(DomainModel):
    """Required wait before a POI can be visited within fixture opening hours."""

    poi_id: str
    starts_at: datetime
    ends_at: datetime
    reason: Literal["opening_hours"] = "opening_hours"

    @model_validator(mode="after")
    def validate_interval(self) -> WaitingPeriod:
        if self.starts_at.tzinfo is None or self.starts_at.utcoffset() is None:
            raise ValueError("Waiting-period timestamps must include a timezone")
        if self.ends_at.tzinfo is None or self.ends_at.utcoffset() is None:
            raise ValueError("Waiting-period timestamps must include a timezone")
        if self.ends_at <= self.starts_at:
            raise ValueError("Waiting period must have a positive duration")
        return self


class PlanningResult(DomainModel):
    status: PlanStatus
    itinerary: Itinerary | None = None
    transfers: list[WalkingTransfer] = Field(default_factory=list)
    waiting_periods: list[WaitingPeriod] = Field(default_factory=list)
    validation: ItineraryValidationResult
    explanations: list[str] = Field(default_factory=list)
    selected_poi_ids: list[str] = Field(default_factory=list)


class UnsupportedTravelModeError(ValueError):
    """Raised when an estimator cannot make a grounded estimate for a mode."""


class TravelTimeEstimator(Protocol):
    def estimate(self, origin: POI, destination: POI, mode: TravelMode) -> TravelEstimate:
        """Estimate transfer distance and duration for one travel mode."""


class HaversineWalkingTimeEstimator:
    """Approximate walking time; this is not pedestrian-route or map data."""

    def __init__(
        self,
        *,
        routing_multiplier: float = 1.4,
        walking_speed_kmh: float = 4.5,
    ) -> None:
        if routing_multiplier < 1:
            raise ValueError("routing_multiplier must be at least 1")
        if walking_speed_kmh <= 0:
            raise ValueError("walking_speed_kmh must be positive")
        self._routing_multiplier = routing_multiplier
        self._walking_speed_kmh = walking_speed_kmh

    def estimate(self, origin: POI, destination: POI, mode: TravelMode) -> TravelEstimate:
        if mode != "walking":
            raise UnsupportedTravelModeError(
                f"Approximate travel-time estimation is only implemented for walking, not {mode}"
            )
        distance = self._haversine_km(
            origin.location.latitude,
            origin.location.longitude,
            destination.location.latitude,
            destination.location.longitude,
        )
        walking_distance = distance * self._routing_multiplier
        minutes = math.ceil(walking_distance / self._walking_speed_kmh * 60)
        return TravelEstimate(
            straight_line_distance_km=distance,
            estimated_walking_distance_km=walking_distance,
            minutes=minutes,
        )

    @staticmethod
    def _haversine_km(
        latitude_a: float,
        longitude_a: float,
        latitude_b: float,
        longitude_b: float,
    ) -> float:
        earth_radius_km = 6371.0088
        lat_a, lat_b = math.radians(latitude_a), math.radians(latitude_b)
        latitude_delta = lat_b - lat_a
        longitude_delta = math.radians(longitude_b - longitude_a)
        value = (
            math.sin(latitude_delta / 2) ** 2
            + math.cos(lat_a)
            * math.cos(lat_b)
            * math.sin(longitude_delta / 2) ** 2
        )
        return earth_radius_km * 2 * math.asin(math.sqrt(min(1.0, value)))


class ItineraryValidator:
    """Recalculates itinerary constraints without trusting planner annotations."""

    def __init__(
        self,
        repository: POIRepository,
        travel_time_estimator: TravelTimeEstimator,
    ) -> None:
        self._repository = repository
        self._travel_time_estimator = travel_time_estimator

    def validate(
        self,
        request: PlanningRequest,
        itinerary: Itinerary,
    ) -> ItineraryValidationResult:
        issues: list[ValidationIssue] = []
        warnings: list[str] = []
        zone = ZoneInfo(request.timezone)
        try:
            expected_start = _local_datetime(request.requested_date, request.start_time, zone)
            expected_end = _local_datetime(request.requested_date, request.end_time, zone)
            _validate_constant_offset_window(expected_start, expected_end)
        except ValueError as error:
            return ItineraryValidationResult(
                is_valid=False,
                issues=[ValidationIssue(code="invalid_local_time", message=str(error))],
            )
        if (
            request.travel_mode != "walking"
            and isinstance(self._travel_time_estimator, HaversineWalkingTimeEstimator)
        ):
            issues.append(
                ValidationIssue(
                    code="unsupported_travel_mode",
                    message=(
                        "Approximate travel-time estimation is only implemented for walking, "
                        f"not {request.travel_mode}."
                    ),
                )
            )
        if itinerary.timezone != request.timezone:
            issues.append(
                ValidationIssue(
                    code="timezone_mismatch",
                    message="Itinerary timezone does not match the requested local timezone.",
                )
            )
        if itinerary.starts_at.astimezone(zone) != expected_start:
            issues.append(
                ValidationIssue(
                    code="window_start_mismatch",
                    message="Itinerary does not start at the requested local start time.",
                )
            )
        if itinerary.ends_at.astimezone(zone) != expected_end:
            issues.append(
                ValidationIssue(
                    code="window_end_mismatch",
                    message="Itinerary does not end at the requested local end time.",
                )
            )

        stops_by_id: dict[str, POI] = {}
        previous: ItineraryStop | None = None
        seen: set[str] = set()
        for index, stop in enumerate(itinerary.stops):
            poi = self._repository.get_by_id(stop.poi_id)
            if poi is None:
                issues.append(
                    ValidationIssue(
                        code="unknown_poi",
                        message=f"POI {stop.poi_id!r} is not present in the supplied dataset.",
                        poi_id=stop.poi_id,
                    )
                )
                previous = stop
                continue
            stops_by_id[stop.poi_id] = poi
            local_start = stop.starts_at.astimezone(zone)
            local_end = stop.ends_at.astimezone(zone)
            if any(
                value.second or value.microsecond
                for value in (local_start, local_end)
            ):
                issues.append(
                    ValidationIssue(
                        code="subminute_stop_boundary",
                        message=f"{poi.name} must use local minute-precision timestamps.",
                        poi_id=poi.id,
                    )
                )
            if stop.starts_at < itinerary.starts_at or stop.ends_at > itinerary.ends_at:
                issues.append(
                    ValidationIssue(
                        code="outside_time_window",
                        message=f"{poi.name} is scheduled outside the requested time window.",
                        poi_id=poi.id,
                    )
                )
            if local_end - local_start != timedelta(minutes=poi.suggested_visit_minutes):
                issues.append(
                    ValidationIssue(
                        code="visit_duration_mismatch",
                        message=(
                            f"{poi.name} must reserve its full suggested "
                            f"{poi.suggested_visit_minutes}-minute visit duration."
                        ),
                        poi_id=poi.id,
                    )
                )
            if poi.id in seen and not request.allow_duplicate_stops:
                issues.append(
                    ValidationIssue(
                        code="duplicate_stop",
                        message=f"{poi.name} appears more than once.",
                        poi_id=poi.id,
                    )
                )
            seen.add(poi.id)
            self._validate_metadata(request, poi, issues, stop_index=index)
            self._validate_opening_hours(request, poi, stop, issues, warnings)

            if previous is not None:
                if stop.starts_at < previous.ends_at:
                    issues.append(
                        ValidationIssue(
                            code="overlapping_stops",
                            message=f"{poi.name} overlaps the preceding itinerary stop.",
                            poi_id=poi.id,
                        )
                    )
                previous_poi = self._repository.get_by_id(previous.poi_id)
                if previous_poi is not None:
                    try:
                        estimate = self._travel_time_estimator.estimate(
                            previous_poi, poi, request.travel_mode
                        )
                    except UnsupportedTravelModeError as error:
                        issues.append(
                            ValidationIssue(code="unsupported_travel_mode", message=str(error))
                        )
                    else:
                        earliest_start = previous.ends_at + timedelta(minutes=estimate.minutes)
                        if stop.starts_at < earliest_start:
                            issues.append(
                                ValidationIssue(
                                    code="insufficient_travel_time",
                                    message=(
                                        f"At least {estimate.minutes} estimated walking minutes "
                                        f"are required between {previous_poi.name} and {poi.name}."
                                    ),
                                    poi_id=poi.id,
                                )
                            )
            previous = stop

        for required_id in request.required_poi_ids:
            if required_id not in seen:
                issues.append(
                    ValidationIssue(
                        code="required_stop_missing",
                        message=f"Required POI {required_id!r} is missing from the itinerary.",
                        poi_id=required_id,
                    )
                )

        if request.replacement_stop_index is not None:
            original_ids = [stop.poi_id for stop in request.preserved_stops]
            actual_ids = [stop.poi_id for stop in itinerary.stops]
            index = request.replacement_stop_index
            expected_ids = list(original_ids)
            if 0 <= index < len(expected_ids) and len(actual_ids) == len(expected_ids):
                if actual_ids[index] == original_ids[index]:
                    issues.append(
                        ValidationIssue(
                            code="replacement_not_applied",
                            message="The requested itinerary stop was not replaced.",
                            poi_id=actual_ids[index],
                        )
                    )
                expected_ids[index] = actual_ids[index]
                if actual_ids != expected_ids:
                    issues.append(
                        ValidationIssue(
                            code="unaffected_stop_changed",
                            message=(
                                "A stop outside the requested replacement position changed "
                                "identity, order, or count."
                            ),
                        )
                    )
            else:
                issues.append(
                    ValidationIssue(
                        code="replacement_sequence_invalid",
                        message="The revised itinerary does not preserve the original stop sequence.",
                    )
                )
            for required_id in request.replacement_required_poi_ids:
                if index >= len(actual_ids) or actual_ids[index] != required_id:
                    issues.append(
                        ValidationIssue(
                            code="required_replacement_missing",
                            message=(
                                f"Requested replacement POI {required_id!r} was not used "
                                "at the selected stop."
                            ),
                            poi_id=required_id,
                        )
                    )

        return ItineraryValidationResult(
            is_valid=not issues,
            issues=issues,
            warnings=warnings,
        )

    @staticmethod
    def _validate_metadata(
        request: PlanningRequest,
        poi: POI,
        issues: list[ValidationIssue],
        *,
        stop_index: int,
    ) -> None:
        is_replacement_stop = request.replacement_stop_index == stop_index
        if request.replacement_stop_index is not None and not is_replacement_stop:
            return
        filters = (
            request.replacement_hard_filters or POIMetadataFilters()
            if is_replacement_stop
            else request.hard_filters
        )
        if filters.category and poi.category.casefold() != filters.category.casefold():
            issues.append(
                ValidationIssue(
                    code="category_constraint_violation",
                    message=f"{poi.name} does not match the required category.",
                    poi_id=poi.id,
                )
            )
        poi_tags = {tag.casefold() for tag in poi.tags}
        if any(tag.casefold() not in poi_tags for tag in filters.tags):
            issues.append(
                ValidationIssue(
                    code="tag_constraint_violation",
                    message=f"{poi.name} does not match all required tags.",
                    poi_id=poi.id,
                )
            )
        if filters.setting and poi.setting != filters.setting:
            issues.append(
                ValidationIssue(
                    code="setting_constraint_violation",
                    message=f"{poi.name} does not match the required setting.",
                    poi_id=poi.id,
                )
            )
        family_filter = filters.family_suitability_fixture
        if family_filter == "unknown" and poi.family_suitability_fixture is not None:
            family_mismatch = True
        elif isinstance(family_filter, bool):
            family_mismatch = poi.family_suitability_fixture is not family_filter
        else:
            family_mismatch = False
        family_required = (
            request.replacement_family_required
            if is_replacement_stop
            else request.family_required
        )
        if family_required and poi.family_suitability_fixture is not True:
            family_mismatch = True
        if family_mismatch:
            issues.append(
                ValidationIssue(
                    code="family_suitability_not_confirmed",
                    message=(
                        f"{poi.name} does not meet the requested family-suitability "
                        "fixture constraint."
                    ),
                    poi_id=poi.id,
                )
            )

    @staticmethod
    def _validate_opening_hours(
        request: PlanningRequest,
        poi: POI,
        stop: ItineraryStop,
        issues: list[ValidationIssue],
        warnings: list[str],
    ) -> None:
        if poi.opening_hours_fixture is None:
            message = (
                f"Opening hours for {poi.name} are unknown in the exercise fixture; "
                "this stop is conditional, not confirmed open."
            )
            issues.append(
                ValidationIssue(
                    code="unknown_opening_hours",
                    message=message,
                    poi_id=poi.id,
                )
            )
            warnings.append(message)
            return

        zone = ZoneInfo(request.timezone)
        local_start = stop.starts_at.astimezone(zone)
        local_end = stop.ends_at.astimezone(zone)
        start_minute = _minutes_from_date(local_start, request.requested_date)
        end_minute = _minutes_from_date(local_end, request.requested_date)
        windows: list[tuple[int, int]] = []
        first_date = min(local_start.date(), local_end.date())
        last_date = max(local_start.date(), local_end.date())
        cursor = first_date
        while cursor <= last_date:
            windows.extend(_opening_intervals(poi, cursor, request.requested_date))
            cursor += timedelta(days=1)
        if not windows:
            weekday_names = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
            weekday = cast(Weekday, weekday_names[request.requested_date.weekday()])
            eligibility = opening_hours_eligibility(poi, weekday)
            if eligibility.status == "closed":
                issues.append(
                    ValidationIssue(
                        code="poi_closed",
                        message=f"{poi.name} is closed on the requested date in the fixture.",
                        poi_id=poi.id,
                    )
                )
            else:
                issues.append(
                    ValidationIssue(
                        code="outside_opening_hours",
                        message=f"{poi.name} has no opening interval covering this visit.",
                        poi_id=poi.id,
                    )
                )
            return
        if not any(start_minute >= opened and end_minute <= closed for opened, closed in windows):
            issues.append(
                ValidationIssue(
                    code="outside_opening_hours",
                    message=(
                        f"The full {poi.suggested_visit_minutes}-minute visit to {poi.name} "
                        "does not fit within an opening window."
                    ),
                    poi_id=poi.id,
                )
            )


class DeterministicItineraryPlanner:
    """Bounded scheduler over supplied or repository-filtered POI candidates."""

    def __init__(
        self,
        repository: POIRepository,
        *,
        travel_time_estimator: TravelTimeEstimator | None = None,
        validator: ItineraryValidator | None = None,
    ) -> None:
        self._repository = repository
        self._travel_time_estimator = travel_time_estimator or HaversineWalkingTimeEstimator()
        self._validator = validator or ItineraryValidator(
            repository,
            self._travel_time_estimator,
        )

    def plan(
        self,
        request: PlanningRequest | dict[str, object],
        candidates: Sequence[POI] | None = None,
    ) -> PlanningResult:
        validated = (
            request
            if isinstance(request, PlanningRequest)
            else PlanningRequest.model_validate(request)
        )
        window_start, window_end = _window_minutes(validated)
        base_issues = self._validate_request_candidates(validated)
        if base_issues:
            return self._infeasible(base_issues)
        required_duration = sum(
            poi.suggested_visit_minutes
            for poi_id in validated.required_poi_ids
            if (poi := self._repository.get_by_id(poi_id)) is not None
        )
        window_duration = window_end - window_start
        duration_issues = (
            [
                ValidationIssue(
                    code="insufficient_overall_time",
                    message=(
                        f"Required visit durations total {required_duration} minutes, "
                        f"exceeding the {window_duration}-minute window before travel."
                    ),
                )
            ]
            if required_duration > window_duration
            else []
        )

        if validated.max_stops < len(
            (
                {stop.poi_id for stop in validated.preserved_stops}
                if validated.replacement_stop_index is not None
                else set(validated.required_poi_ids)
                | {stop.poi_id for stop in validated.preserved_stops}
            )
        ):
            return self._infeasible(
                [
                    ValidationIssue(
                        code="max_stops_below_required_count",
                        message="max_stops is smaller than the required and preserved stop count.",
                    )
                ]
            )
        try:
            candidate_pool = self._candidate_pool(validated, candidates)
        except InvalidMetadataFilter as error:
            return self._infeasible(
                [ValidationIssue(code="invalid_hard_filter", message=str(error))]
            )

        missing_required = [
            poi_id
            for poi_id in validated.required_poi_ids
            if self._repository.get_by_id(poi_id) is None
        ]
        if missing_required:
            return self._infeasible(
                [
                    *duration_issues,
                    *(
                        ValidationIssue(
                            code="required_poi_missing",
                            message=f"Required POI {poi_id!r} is absent from the dataset.",
                            poi_id=poi_id,
                        )
                        for poi_id in missing_required
                    ),
                ]
            )
        if (
            not candidate_pool
            and not validated.required_poi_ids
            and not validated.preserved_stops
        ):
            return self._infeasible(
                [
                    *duration_issues,
                    ValidationIssue(
                        code="no_eligible_candidates",
                        message="No supplied POI candidates satisfy the planning constraints.",
                    ),
                ]
            )

        preserved_ids = {stop.poi_id for stop in validated.preserved_stops}
        required_ids = [poi_id for poi_id in validated.required_poi_ids if poi_id not in preserved_ids]
        required_pois: dict[str, POI] = {}
        required_issues: list[ValidationIssue] = []
        for poi_id in required_ids:
            poi = self._repository.get_by_id(poi_id)
            assert poi is not None
            if not self._matches_hard_constraints(validated, poi):
                required_issues.append(
                    ValidationIssue(
                        code="required_stop_violates_constraints",
                        message=f"Required POI {poi.name} violates an explicit hard constraint.",
                        poi_id=poi_id,
                    )
                )
            required_pois[poi_id] = poi
            if poi.opening_hours_fixture is None and validated.unknown_hours_policy == "exclude":
                required_issues.append(
                    ValidationIssue(
                        code="unknown_opening_hours",
                        message=(
                            f"Required POI {poi.name} has unknown fixture hours and the "
                            "request excludes stops with unknown hours."
                        ),
                        poi_id=poi.id,
                    )
                )
            if poi.opening_hours_fixture is not None and not _opening_intervals(
                poi, validated.requested_date, validated.requested_date
            ):
                required_issues.append(
                    ValidationIssue(
                        code="poi_closed",
                        message=f"Required POI {poi.name} is closed on the requested date.",
                        poi_id=poi.id,
                    )
                )
        if required_issues:
            return self._infeasible([*duration_issues, *required_issues])
        if duration_issues:
            return self._infeasible(duration_issues)

        zone = ZoneInfo(validated.timezone)
        try:
            itinerary_start = _local_datetime(
                validated.requested_date, validated.start_time, zone
            )
            itinerary_end = _local_datetime(
                validated.requested_date, validated.end_time, zone
            )
            _validate_constant_offset_window(itinerary_start, itinerary_end)
        except ValueError as error:
            return self._infeasible(
                [ValidationIssue(code="invalid_local_time", message=str(error))]
            )
        preserved_stops = sorted(validated.preserved_stops, key=lambda stop: stop.starts_at)
        preserved_issues = self._validate_preserved_stops(
            validated,
            preserved_stops,
            itinerary_start,
            itinerary_end,
        )
        if preserved_issues:
            return self._infeasible(preserved_issues)

        if (
            validated.travel_mode != "walking"
            and isinstance(self._travel_time_estimator, HaversineWalkingTimeEstimator)
        ):
            return self._infeasible(
                [
                    ValidationIssue(
                        code="unsupported_travel_mode",
                        message=(
                            "Approximate travel-time estimation is only implemented for "
                            f"walking, not {validated.travel_mode}."
                        ),
                    )
                ]
            )

        if validated.replacement_stop_index is not None:
            return self._plan_replacement(
                validated,
                candidate_pool,
                itinerary_start,
                itinerary_end,
            )

        used_ids = {stop.poi_id for stop in preserved_stops}
        ranked_pool = self._rank_candidates(validated, candidate_pool)
        required_remaining = dict(required_pois)
        cursor = window_start
        previous_poi: POI | None = None
        previous_stop_end: int | None = None
        if preserved_stops:
            last_stop = preserved_stops[-1]
            previous_poi = self._repository.get_by_id(last_stop.poi_id)
            previous_stop_end = _minutes_from_date(
                last_stop.ends_at.astimezone(zone), validated.requested_date
            )
            cursor = previous_stop_end

        required_schedule = self._schedule_required(
            validated,
            list(required_remaining.values()),
            cursor,
            previous_poi,
            window_end,
            _ranking_index(ranked_pool),
        )
        if required_schedule is None:
            duration_total = sum(poi.suggested_visit_minutes for poi in required_remaining.values())
            duration_window = window_end - window_start
            reasons = [
                ValidationIssue(
                    code="required_stops_do_not_fit",
                    message=(
                        f"All required stops cannot fit in the requested window. "
                        f"Required visit time alone is {duration_total} minutes "
                        f"for a {duration_window}-minute window, before travel and opening constraints."
                    ),
                )
            ]
            return self._infeasible(reasons)

        new_stops: list[ItineraryStop] = list(required_schedule)
        for stop in required_schedule:
            poi = self._repository.get_by_id(stop.poi_id)
            if poi is not None:
                used_ids.add(poi.id)
                previous_poi = poi
                previous_stop_end = _minutes_from_date(
                    stop.ends_at.astimezone(zone), validated.requested_date
                )
                cursor = previous_stop_end

        if previous_stop_end is not None and previous_poi is None:
            cursor = previous_stop_end

        total_stop_count = len(preserved_stops) + len(new_stops)
        stop_reason = "max_stops_reached"
        while total_stop_count < validated.max_stops:
            choices: list[tuple[int, int, str, POI, int, int, TravelEstimate | None]] = []
            for rank, poi in enumerate(ranked_pool):
                if poi.id in used_ids or not self._matches_hard_constraints(validated, poi):
                    continue
                estimate = (
                    self._travel_time_estimator.estimate(previous_poi, poi, validated.travel_mode)
                    if previous_poi is not None
                    else None
                )
                earliest = cursor + (estimate.minutes if estimate is not None else 0)
                slot = _find_slot(validated, poi, earliest, window_end)
                if slot is not None:
                    choices.append(
                        (
                            estimate.minutes if estimate is not None else 0,
                            rank,
                            poi.id,
                            poi,
                            slot[0],
                            slot[1],
                            estimate,
                        )
                    )
            if not choices:
                stop_reason = "no_schedulable_candidates"
                break
            if validated.preferred_poi_ids:
                preferred_ids = set(validated.preferred_poi_ids)
                focused = [choice for choice in choices if choice[3].id in preferred_ids]
                if focused:
                    choices = focused
                elif any(stop.poi_id in preferred_ids for stop in new_stops):
                    stop_reason = "focused_candidates_exhausted"
                    break
            _, _, _, poi, start_minute, end_minute, _ = min(
                choices,
                key=lambda choice: (
                    choice[3].opening_hours_fixture is None,
                    choice[1],
                    choice[0],
                    choice[2],
                ),
            )
            stop = _make_stop(
                poi,
                start_minute,
                end_minute,
                validated.requested_date,
                zone,
                reason=(
                    "Required destination."
                    if poi.id in validated.required_poi_ids
                    else "Selected from eligible candidates."
                ),
            )
            new_stops.append(stop)
            used_ids.add(poi.id)
            previous_poi = poi
            cursor = end_minute
            total_stop_count += 1

        all_stops = preserved_stops + new_stops
        _LOGGER.info(
            "Itinerary candidate selection completed",
            extra={
                "stage": "itinerary_selection",
                "candidate_count": len(candidate_pool),
                "selected_count": len(all_stops),
                "selected_poi_ids": [stop.poi_id for stop in all_stops],
                "stop_reason": stop_reason,
                "remaining_minutes": max(0, window_end - cursor),
            },
        )
        if not all_stops:
            return self._infeasible(
                [
                    ValidationIssue(
                        code="no_schedulable_candidates",
                        message=(
                            "No eligible candidate can be scheduled within the opening-hour "
                            "and time-window constraints."
                        ),
                    )
                ]
            )
        try:
            itinerary = Itinerary(
                conversation_id=None,
                timezone=validated.timezone,
                starts_at=itinerary_start,
                ends_at=itinerary_end,
                stops=all_stops,
            )
        except ValueError as error:
            return self._infeasible(
                [ValidationIssue(code="invalid_schedule", message=str(error))]
            )

        validation = self._validator.validate(validated, itinerary)
        if not validation.is_valid:
            unknown_only = all(issue.code == "unknown_opening_hours" for issue in validation.issues)
            status: PlanStatus = "uncertain" if unknown_only else "infeasible"
            if status == "infeasible":
                return PlanningResult(
                    status=status,
                    validation=validation,
                    explanations=[issue.message for issue in validation.issues],
                    selected_poi_ids=[stop.poi_id for stop in all_stops],
                )
        else:
            status = "feasible"
        transfers = self._transfers(validated, all_stops)
        explanations = (
            [issue.message for issue in validation.issues]
            if status == "uncertain"
            else [
                "Schedule times and visit durations fit the supplied fixture constraints.",
                "Walking transfers are approximate distance-based estimates, not route directions.",
            ]
        )
        unused_time = _unused_time_explanation(all_stops, validated.requested_date, window_end, zone)
        if unused_time is not None:
            explanations.append(unused_time)
        if stop_reason == "focused_candidates_exhausted":
            explanations.append(
                "No further candidate matching the recognized interests fits after the last "
                "stop under the visit-duration, walking and opening-hour constraints. "
                "Unrelated stops were not added to fill the remaining time."
            )
        return PlanningResult(
            status=status,
            itinerary=itinerary,
            transfers=transfers,
            validation=validation,
            explanations=explanations,
            selected_poi_ids=[stop.poi_id for stop in all_stops],
        )

    def _candidate_pool(
        self,
        request: PlanningRequest,
        candidates: Sequence[POI] | None,
    ) -> list[POI]:
        filters = (
            request.replacement_hard_filters or POIMetadataFilters()
            if request.replacement_stop_index is not None
            else request.hard_filters
        )
        eligible = self._repository.filter_metadata(filters)
        eligible_ids = {poi.id for poi in eligible}
        family_required = (
            request.replacement_family_required
            if request.replacement_stop_index is not None
            else request.family_required
        )
        if family_required:
            eligible_ids = {
                poi_id
                for poi_id in eligible_ids
                if (poi := self._repository.get_by_id(poi_id)) is not None
                and poi.family_suitability_fixture is True
            }
        source = candidates if candidates is not None else eligible
        selected: dict[str, POI] = {}
        for candidate in source:
            canonical = self._repository.get_by_id(candidate.id)
            if canonical is not None and canonical.id in eligible_ids:
                selected[canonical.id] = canonical
        for poi_id in request.required_poi_ids:
            canonical = self._repository.get_by_id(poi_id)
            if canonical is not None and self._matches_hard_constraints(request, canonical):
                selected[canonical.id] = canonical
        for poi_id in request.replacement_required_poi_ids:
            canonical = self._repository.get_by_id(poi_id)
            if canonical is not None and self._matches_hard_constraints(
                request, canonical, replacement=True
            ):
                selected[canonical.id] = canonical
        for stop in request.preserved_stops:
            canonical = self._repository.get_by_id(stop.poi_id)
            if canonical is not None:
                selected[canonical.id] = canonical
        return list(selected.values())

    def _plan_replacement(
        self,
        request: PlanningRequest,
        candidate_pool: Sequence[POI],
        itinerary_start: datetime,
        itinerary_end: datetime,
    ) -> PlanningResult:
        index = request.replacement_stop_index
        assert index is not None
        originals = list(request.preserved_stops)
        original_ids = [stop.poi_id for stop in originals]
        original_pois = [self._repository.get_by_id(poi_id) for poi_id in original_ids]
        if any(poi is None for poi in original_pois):
            return self._infeasible(
                [
                    ValidationIssue(
                        code="unknown_preserved_poi",
                        message="The stored itinerary contains a POI absent from the dataset.",
                    )
                ]
            )

        missing_required = [
            poi_id
            for poi_id in request.replacement_required_poi_ids
            if self._repository.get_by_id(poi_id) is None
        ]
        if missing_required:
            return self._infeasible(
                [
                    ValidationIssue(
                        code="required_replacement_missing",
                        message=f"Required replacement POI {poi_id!r} is absent from the dataset.",
                        poi_id=poi_id,
                    )
                    for poi_id in missing_required
                ]
            )

        existing_except_target = set(original_ids) - {original_ids[index]}
        ranked_candidates = [
            poi
            for poi in self._rank_candidates(request, candidate_pool)
            if poi.id not in existing_except_target
            and poi.id != original_ids[index]
            and self._matches_hard_constraints(request, poi, replacement=True)
            and (
                not request.replacement_required_poi_ids
                or poi.id in request.replacement_required_poi_ids
            )
        ]
        # Prefer candidates whose exercise hours can be checked. Keep unknown-
        # hours options as a fallback when no known-hours candidate can fit.
        preferred_ids = set(request.preferred_poi_ids)
        ranked_candidates.sort(key=lambda poi: (
            bool(preferred_ids) and poi.id not in preferred_ids,
            poi.opening_hours_fixture is None,
        ))
        if not ranked_candidates:
            return self._infeasible(
                [
                    ValidationIssue(
                        code="no_suitable_replacement",
                        message=(
                            "No distinct POI matches the replacement constraints without "
                            "duplicating an unaffected stop."
                        ),
                        poi_id=original_ids[index],
                    )
                ]
            )

        zone = ZoneInfo(request.timezone)
        window_start, window_end = _window_minutes(request)
        considered_issues: list[ValidationIssue] = []
        preferred_pool = [poi for poi in ranked_candidates if poi.id in preferred_ids]
        decision_trace: list[dict[str, object]] = []
        for candidate_rank, replacement in enumerate(ranked_candidates):
            candidate_trace: dict[str, object] = {
                "candidate_id": replacement.id,
                "rank": candidate_rank,
                "thematic_match": replacement.id in preferred_ids,
                "opening_hours_known": replacement.opening_hours_fixture is not None,
                "visit_duration_minutes": replacement.suggested_visit_minutes,
                "requested_end_time": request.end_time,
                "feasible": False,
                "feasibility_reason": "sequence_not_checked",
                "transfers": [],
            }
            sequence: list[POI] = []
            for stop_index, poi in enumerate(original_pois):
                if stop_index == index:
                    sequence.append(replacement)
                else:
                    assert poi is not None
                    sequence.append(poi)

            stops: list[ItineraryStop] = []
            waiting_periods: list[WaitingPeriod] = []
            previous_poi: POI | None = None
            cursor = window_start
            schedulable = True
            for stop_index, (poi, old_stop) in enumerate(zip(sequence, originals)):
                travel_minutes = 0
                if previous_poi is not None:
                    try:
                        travel_minutes = self._travel_time_estimator.estimate(
                            previous_poi, poi, request.travel_mode
                        ).minutes
                    except UnsupportedTravelModeError as error:
                        return self._infeasible(
                            [
                                ValidationIssue(
                                    code="unsupported_travel_mode",
                                    message=str(error),
                                )
                            ]
                        )
                transfer_trace = candidate_trace["transfers"]
                if previous_poi is not None and isinstance(transfer_trace, list):
                    transfer_trace.append({
                        "from_poi_id": previous_poi.id,
                        "to_poi_id": poi.id,
                        "estimated_minutes": travel_minutes,
                    })
                arrival = max(cursor + travel_minutes, window_start)
                earliest = arrival
                slot = _find_slot(request, poi, earliest, window_end)
                if slot is None:
                    schedulable = False
                    candidate_trace["feasibility_reason"] = (
                        "closed_on_requested_day"
                        if poi.opening_hours_fixture is not None and not _opening_intervals(
                            poi, request.requested_date, request.requested_date
                        )
                        else "opening_hours_or_end_time_conflict"
                    )
                    candidate_trace["first_unschedulable_poi_id"] = poi.id
                    candidate_trace["arrival_time"] = _local_datetime_from_minutes(
                        request.requested_date, arrival, zone
                    ).strftime("%H:%M")
                    candidate_trace["required_visit_minutes"] = poi.suggested_visit_minutes
                    candidate_trace["earliest_finish_time"] = _local_datetime_from_minutes(
                        request.requested_date,
                        arrival + poi.suggested_visit_minutes,
                        zone,
                    ).strftime("%H:%M")
                    if poi.opening_hours_fixture is not None and not _opening_intervals(
                        poi, request.requested_date, request.requested_date
                    ):
                        considered_issues.append(
                            ValidationIssue(
                                code="poi_closed",
                                message=f"{poi.name} is closed on the requested date.",
                                poi_id=poi.id,
                            )
                        )
                    else:
                        considered_issues.append(
                            ValidationIssue(
                                code="replacement_time_window_conflict",
                                message=(
                                    f"{poi.name} cannot fit its full visit duration, required "
                                    "transfer, and opening window before the requested end time."
                                ),
                                poi_id=poi.id,
                            )
                        )
                    break
                if slot[0] > arrival:
                    waiting_periods.append(
                        WaitingPeriod(
                            poi_id=poi.id,
                            starts_at=_local_datetime_from_minutes(
                                request.requested_date,
                                arrival,
                                zone,
                            ),
                            ends_at=_local_datetime_from_minutes(
                                request.requested_date,
                                slot[0],
                                zone,
                            ),
                        )
                    )
                starts_at = _local_datetime_from_minutes(
                    request.requested_date,
                    slot[0],
                    zone,
                )
                ends_at = _local_datetime_from_minutes(
                    request.requested_date,
                    slot[1],
                    zone,
                )
                stops.append(
                    ItineraryStop(
                        poi_id=poi.id,
                        starts_at=starts_at,
                        ends_at=ends_at,
                        reason=(
                            "Replacement selected for the requested stop."
                            if stop_index == index
                            else old_stop.reason
                        ),
                    )
                )
                cursor = slot[1]
                previous_poi = poi
            if not schedulable:
                decision_trace.append(candidate_trace)
                continue

            itinerary = Itinerary(
                conversation_id=None,
                timezone=request.timezone,
                starts_at=itinerary_start,
                ends_at=itinerary_end,
                stops=stops,
            )
            validation = self._validator.validate(request, itinerary)
            if not validation.is_valid:
                if all(
                    issue.code == "unknown_opening_hours"
                    for issue in validation.issues
                ):
                    status: PlanStatus = "uncertain"
                else:
                    considered_issues.extend(validation.issues)
                    candidate_trace["feasibility_reason"] = "independent_validation_failed"
                    candidate_trace["validation_issue_codes"] = [issue.code for issue in validation.issues[:8]]
                    decision_trace.append(candidate_trace)
                    continue
            else:
                status = "feasible"
            candidate_trace["feasible"] = True
            candidate_trace["feasibility_reason"] = "selected_first_feasible_candidate"
            candidate_trace["scheduled_replacement_start"] = next(
                stop.starts_at.astimezone(zone).strftime("%H:%M")
                for stop_index, stop in enumerate(stops)
                if stop_index == index
            )
            candidate_trace["scheduled_replacement_end"] = next(
                stop.ends_at.astimezone(zone).strftime("%H:%M")
                for stop_index, stop in enumerate(stops)
                if stop_index == index
            )
            decision_trace.append(candidate_trace)
            for remaining in ranked_candidates[candidate_rank + 1:]:
                decision_trace.append({
                    "candidate_id": remaining.id,
                    "rank": ranked_candidates.index(remaining),
                    "thematic_match": remaining.id in preferred_ids,
                    "opening_hours_known": remaining.opening_hours_fixture is not None,
                    "visit_duration_minutes": remaining.suggested_visit_minutes,
                    "requested_end_time": request.end_time,
                    "feasible": None,
                    "feasibility_reason": "not_evaluated_after_first_feasible_candidate",
                    "transfers": [],
                })
            fallback_selected = replacement.id not in preferred_ids
            decision = {
                "ranked_candidate_count": len(ranked_candidates),
                "preferred_candidate_count": len(preferred_pool),
                "selected_candidate_id": replacement.id,
                "selected_from_pool": "fallback" if fallback_selected else "preferred",
                "fallback_reason": (
                    "all_preferred_candidates_infeasible"
                    if fallback_selected and preferred_pool
                    else "no_preferred_candidates"
                    if fallback_selected
                    else None
                ),
                "candidate_decisions": decision_trace[:50],
            }
            _LOGGER.warning(
                "Weather replacement decision completed | trace=%s",
                json.dumps(decision, separators=(",", ":")),
                extra={
                    "stage": "weather_replacement_selection",
                    **decision,
                },
            )
            transfers = self._transfers(request, stops)
            explanations = (
                [issue.message for issue in validation.issues]
                if status == "uncertain"
                else [
                    "The requested stop was replaced; all other POI IDs and their order were preserved.",
                    "Walking transfers are approximate distance-based estimates, not route directions.",
                ]
            )
            unused_time = _unused_time_explanation(stops, request.requested_date, window_end, zone)
            if unused_time is not None:
                explanations.append(unused_time)
            return PlanningResult(
                status=status,
                itinerary=itinerary,
                transfers=transfers,
                waiting_periods=waiting_periods,
                validation=validation,
                explanations=explanations,
                selected_poi_ids=[stop.poi_id for stop in stops],
            )

        return self._infeasible(
            considered_issues
            or [
                ValidationIssue(
                    code="replacement_not_feasible",
                    message=(
                        "No matching replacement can be scheduled while preserving the "
                        "unaffected stops and requested end time."
                    ),
                    poi_id=original_ids[index],
                )
            ]
        )

    def _validate_request_candidates(self, request: PlanningRequest) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        filters = (
            request.replacement_hard_filters or POIMetadataFilters()
            if request.replacement_stop_index is not None
            else request.hard_filters
        )
        try:
            self._repository.filter_metadata(filters)
        except InvalidMetadataFilter as error:
            issues.append(ValidationIssue(code="invalid_hard_filter", message=str(error)))
        return issues

    def _validate_preserved_stops(
        self,
        request: PlanningRequest,
        stops: Sequence[ItineraryStop],
        itinerary_start: datetime,
        itinerary_end: datetime,
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        previous: ItineraryStop | None = None
        for stop in stops:
            if any(
                value.second or value.microsecond
                for value in (stop.starts_at, stop.ends_at)
            ):
                issues.append(
                    ValidationIssue(
                        code="preserved_stop_subminute_boundary",
                        message=f"Preserved stop {stop.poi_id!r} is not aligned to whole minutes.",
                        poi_id=stop.poi_id,
                    )
                )
            if stop.starts_at < itinerary_start or stop.ends_at > itinerary_end:
                issues.append(
                    ValidationIssue(
                        code="preserved_stop_outside_window",
                        message=f"Preserved stop {stop.poi_id!r} falls outside the requested window.",
                        poi_id=stop.poi_id,
                    )
                )
            if previous is not None and stop.starts_at < previous.ends_at:
                issues.append(
                    ValidationIssue(
                        code="preserved_stops_overlap",
                        message="Preserved itinerary stops overlap.",
                        poi_id=stop.poi_id,
                    )
                )
            poi = self._repository.get_by_id(stop.poi_id)
            if poi is None:
                issues.append(
                    ValidationIssue(
                        code="unknown_poi",
                        message=f"Preserved POI {stop.poi_id!r} is absent from the dataset.",
                        poi_id=stop.poi_id,
                    )
                )
            elif not self._matches_hard_constraints(request, poi):
                issues.append(
                    ValidationIssue(
                        code="preserved_stop_violates_constraints",
                        message=f"Preserved POI {poi.name} violates a hard constraint.",
                        poi_id=poi.id,
                    )
                )
            previous = stop
        return issues

    def _schedule_required(
        self,
        request: PlanningRequest,
        remaining: list[POI],
        cursor: int,
        previous_poi: POI | None,
        window_end: int,
        rank_by_id: dict[str, int],
    ) -> list[ItineraryStop] | None:
        if not remaining:
            return []
        options: list[tuple[int, int, str, POI, int, int]] = []
        for poi in remaining:
            estimate_minutes = (
                self._travel_time_estimator.estimate(
                    previous_poi, poi, request.travel_mode
                ).minutes
                if previous_poi is not None
                else 0
            )
            slot = _find_slot(request, poi, cursor + estimate_minutes, window_end)
            if slot is not None:
                options.append(
                    (
                        estimate_minutes,
                        rank_by_id.get(poi.id, math.inf),
                        poi.id,
                        poi,
                        slot[0],
                        slot[1],
                    )
                )
        for _, _, _, poi, start_minute, end_minute in sorted(
            options, key=lambda option: option[:3]
        ):
            rest = [item for item in remaining if item.id != poi.id]
            next_stops = self._schedule_required(
                request,
                rest,
                end_minute,
                poi,
                window_end,
                rank_by_id,
            )
            if next_stops is not None:
                return [
                    _make_stop(
                        poi,
                        start_minute,
                        end_minute,
                        request.requested_date,
                        ZoneInfo(request.timezone),
                        reason="Required destination.",
                    ),
                    *next_stops,
                ]
        return None

    @staticmethod
    def _matches_hard_constraints(
        request: PlanningRequest,
        poi: POI,
        *,
        replacement: bool = False,
    ) -> bool:
        filters = (
            request.replacement_hard_filters or POIMetadataFilters()
            if replacement
            else request.hard_filters
        )
        if filters.category and poi.category.casefold() != filters.category.casefold():
            return False
        if any(tag.casefold() not in {item.casefold() for item in poi.tags} for tag in filters.tags):
            return False
        if filters.setting and poi.setting != filters.setting:
            return False
        if filters.family_suitability_fixture == "unknown" and poi.family_suitability_fixture is not None:
            return False
        if isinstance(filters.family_suitability_fixture, bool):
            if poi.family_suitability_fixture is not filters.family_suitability_fixture:
                return False
        family_required = (
            request.replacement_family_required
            if replacement
            else request.family_required
        )
        if family_required and poi.family_suitability_fixture is not True:
            return False
        return True

    def validate(
        self,
        request: PlanningRequest,
        itinerary: Itinerary,
    ) -> ItineraryValidationResult:
        """Independently validate a proposed itinerary with this planner's validator."""
        return self._validator.validate(request, itinerary)

    @staticmethod
    def _rank_candidates(request: PlanningRequest, candidates: Sequence[POI]) -> list[POI]:
        rank = _ranking_index(candidates, request.ranked_candidate_ids)
        return sorted(candidates, key=lambda poi: (rank.get(poi.id, math.inf), poi.id))

    def _transfers(
        self,
        request: PlanningRequest,
        stops: Sequence[ItineraryStop],
    ) -> list[WalkingTransfer]:
        transfers: list[WalkingTransfer] = []
        for previous, current in zip(stops, stops[1:]):
            origin = self._repository.get_by_id(previous.poi_id)
            destination = self._repository.get_by_id(current.poi_id)
            if origin is None or destination is None:
                continue
            estimate = self._travel_time_estimator.estimate(
                origin, destination, request.travel_mode
            )
            transfers.append(
                WalkingTransfer(
                    from_poi_id=origin.id,
                    to_poi_id=destination.id,
                    starts_at=previous.ends_at,
                    ends_at=previous.ends_at + timedelta(minutes=estimate.minutes),
                    estimate=estimate,
                )
            )
        return transfers

    @staticmethod
    def _infeasible(issues: list[ValidationIssue]) -> PlanningResult:
        return PlanningResult(
            status="infeasible",
            validation=ItineraryValidationResult(is_valid=False, issues=issues),
            explanations=[issue.message for issue in issues],
        )


def _window_minutes(request: PlanningRequest) -> tuple[int, int]:
    start = _clock_minutes(request.start_time)
    end = _clock_minutes(request.end_time)
    return start, end


def _clock_minutes(value: str) -> int:
    hour, minute = map(int, value.split(":"))
    return hour * 60 + minute


def _local_datetime(day: date, clock: str, zone: ZoneInfo) -> datetime:
    hour, minute = map(int, clock.split(":"))
    local_naive = datetime.combine(day, time(hour, minute))
    local_aware = local_naive.replace(tzinfo=zone, fold=0)
    round_trip = local_aware.astimezone(timezone.utc).astimezone(zone).replace(tzinfo=None)
    if round_trip != local_naive:
        raise ValueError(f"{day.isoformat()} {clock} is not a valid local time in {zone.key}")
    alternate = local_naive.replace(tzinfo=zone, fold=1)
    if alternate.utcoffset() != local_aware.utcoffset():
        alternate_round_trip = (
            alternate.astimezone(timezone.utc).astimezone(zone).replace(tzinfo=None)
        )
        if alternate_round_trip == local_naive:
            raise ValueError(
                f"{day.isoformat()} {clock} is ambiguous during a daylight-saving "
                f"transition in {zone.key}"
            )
    return local_aware


def _validate_constant_offset_window(start: datetime, end: datetime) -> None:
    if start.utcoffset() != end.utcoffset():
        zone_name = getattr(start.tzinfo, "key", str(start.tzinfo))
        raise ValueError(
            "The requested time window crosses a daylight-saving transition in "
            f"{zone_name}; provide a window that does not cross the clock change."
        )


def _local_datetime_from_minutes(day: date, minutes: int, zone: ZoneInfo) -> datetime:
    return datetime.combine(day, time.min, tzinfo=zone) + timedelta(minutes=minutes)


def _minutes_from_date(value: datetime, origin: date) -> int:
    return (value.date() - origin).days * 1440 + value.hour * 60 + value.minute


def _unused_time_explanation(
    stops: Sequence[ItineraryStop],
    requested_date: date,
    window_end: int,
    zone: ZoneInfo,
) -> str | None:
    if not stops:
        return None
    last_stop_end = stops[-1].ends_at.astimezone(zone)
    unused_minutes = window_end - _minutes_from_date(last_stop_end, requested_date)
    if unused_minutes <= 0:
        return None
    return (
        f"Scheduled visits finish at {last_stop_end:%H:%M}, leaving "
        f"{unused_minutes} minutes unallocated in the requested time window."
    )


def _opening_intervals(poi: POI, day: date, origin: date) -> list[tuple[int, int]]:
    schedule = poi.opening_hours_fixture
    if schedule is None:
        return []
    weekday_names = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
    intervals: list[tuple[int, int]] = []
    today_offset = (day - origin).days * 1440
    today = opening_hours_eligibility(poi, weekday_names[day.weekday()])
    if today.status == "scheduled" and today.opening_window is not None:
        intervals.append(
            (
                today_offset + today.opening_window.start_minute,
                today_offset + today.opening_window.end_minute,
            )
        )
    previous_day = day - timedelta(days=1)
    previous = opening_hours_eligibility(poi, weekday_names[previous_day.weekday()])
    if (
        previous.status == "scheduled"
        and previous.opening_window is not None
        and previous.opening_window.end_minute > 1440
    ):
        intervals.append(
            (
                (previous_day - origin).days * 1440
                + previous.opening_window.start_minute,
                (previous_day - origin).days * 1440
                + previous.opening_window.end_minute,
            )
        )
    return intervals


def _find_slot(
    request: PlanningRequest,
    poi: POI,
    earliest_minute: int,
    window_end: int,
) -> tuple[int, int] | None:
    duration = poi.suggested_visit_minutes
    if poi.opening_hours_fixture is None:
        if request.unknown_hours_policy == "exclude":
            return None
        start = earliest_minute
        return (start, start + duration) if start + duration <= window_end else None
    intervals = _opening_intervals(poi, request.requested_date, request.requested_date)
    for opening_start, opening_end in sorted(intervals):
        start = max(earliest_minute, opening_start)
        end = start + duration
        if end <= opening_end and end <= window_end:
            return start, end
    return None


def _make_stop(
    poi: POI,
    start_minute: int,
    end_minute: int,
    requested_date: date,
    zone: ZoneInfo,
    *,
    reason: str,
) -> ItineraryStop:
    start = datetime.combine(requested_date, time.min, tzinfo=zone) + timedelta(
        minutes=start_minute
    )
    end = datetime.combine(requested_date, time.min, tzinfo=zone) + timedelta(
        minutes=end_minute
    )
    return ItineraryStop(
        poi_id=poi.id,
        starts_at=start,
        ends_at=end,
        reason=reason,
    )


def _ranking_index(
    candidates: Sequence[POI],
    ranked_ids: Sequence[str] | None = None,
) -> dict[str, int]:
    ranked = list(ranked_ids or ())
    rank = {poi_id: index for index, poi_id in enumerate(ranked)}
    fallback = len(rank)
    for index, poi in enumerate(candidates):
        rank.setdefault(poi.id, fallback + index)
    return rank
