from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from corfu_assistant.data import load_dataset
from corfu_assistant.models import (
    CorfuDataset,
    Itinerary,
    ItineraryStop,
    OpeningWindow,
    POI,
    POIMetadataFilters,
    TravelMode,
)
from corfu_assistant.planner import (
    DeterministicItineraryPlanner,
    HaversineWalkingTimeEstimator,
    ItineraryValidator,
    PlanningRequest,
    TravelEstimate,
)
from corfu_assistant.repository import POIRepository


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TZ = ZoneInfo("Europe/Athens")


class FixedTravelEstimator:
    def __init__(self, minutes: int = 0) -> None:
        self.minutes = minutes

    def estimate(
        self,
        _origin: POI,
        _destination: POI,
        _mode: TravelMode,
    ) -> TravelEstimate:
        return TravelEstimate(
            straight_line_distance_km=0.1,
            estimated_walking_distance_km=0.14,
            minutes=self.minutes,
        )


@pytest.fixture
def dataset() -> CorfuDataset:
    return load_dataset(PROJECT_ROOT / "corfu_dataset.json")


@pytest.fixture
def repository(dataset: CorfuDataset) -> POIRepository:
    return POIRepository(dataset)


def request(**updates: object) -> PlanningRequest:
    values: dict[str, object] = {
        "requested_date": date(2026, 10, 6),
        "start_time": "10:00",
        "end_time": "17:00",
        "travel_mode": "walking",
    }
    values.update(updates)
    return PlanningRequest.model_validate(values)


def test_required_five_stop_case_reports_time_and_tuesday_closure(
    repository: POIRepository,
) -> None:
    plan = DeterministicItineraryPlanner(
        repository, travel_time_estimator=FixedTravelEstimator()
    ).plan(
        request(
            start_time="11:00",
            end_time="12:30",
            required_poi_ids=["CF-001", "CF-002", "CF-021", "CF-022", "CF-017"],
        )
    )

    assert plan.status == "infeasible"
    codes = {issue.code for issue in plan.validation.issues}
    assert "insufficient_overall_time" in codes
    assert "poi_closed" in codes
    assert plan.itinerary is None
    assert any("New Fortress" in explanation for explanation in plan.explanations)
    issue_messages = {issue.message for issue in plan.validation.issues}
    assert issue_messages.issubset(set(plan.explanations))


def test_monday_museum_selection_respects_closure(
    repository: POIRepository,
) -> None:
    plan = DeterministicItineraryPlanner(
        repository, travel_time_estimator=FixedTravelEstimator()
    ).plan(
        request(
            requested_date=date(2026, 10, 5),
            start_time="13:00",
            end_time="17:00",
            hard_filters=POIMetadataFilters(tags=["museum"]),
        )
    )

    assert plan.status in {"feasible", "uncertain"}
    assert plan.itinerary is not None
    assert "CF-017" not in plan.selected_poi_ids


@pytest.mark.parametrize(
    "requested_date,start_time,end_time,expected_fragment",
    [
        (date(2026, 3, 29), "03:30", "05:00", "not a valid local time"),
        (date(2026, 10, 25), "03:30", "05:00", "ambiguous"),
        (date(2026, 3, 29), "02:30", "04:30", "crosses a daylight-saving transition"),
    ],
)
def test_dst_gap_overlap_and_transition_windows_are_explicitly_infeasible(
    repository: POIRepository,
    requested_date: date,
    start_time: str,
    end_time: str,
    expected_fragment: str,
) -> None:
    plan = DeterministicItineraryPlanner(
        repository, travel_time_estimator=FixedTravelEstimator()
    ).plan(request(
        requested_date=requested_date,
        start_time=start_time,
        end_time=end_time,
        timezone="Europe/Athens",
    ))

    assert plan.status == "infeasible"
    assert plan.itinerary is None
    assert [issue.code for issue in plan.validation.issues] == ["invalid_local_time"]
    assert expected_fragment in plan.validation.issues[0].message


def test_ranked_theme_candidates_are_preferred_over_proximity_ties(
    repository: POIRepository,
) -> None:
    ranked = ["CF-007", "CF-012", "CF-029", "CF-036"]
    candidates = [repository.get_by_id(poi_id) for poi_id in ranked]
    assert all(candidates)
    plan = DeterministicItineraryPlanner(
        repository, travel_time_estimator=FixedTravelEstimator()
    ).plan(
        request(
            requested_date=date(2026, 10, 5), start_time="13:00", end_time="17:00",
            ranked_candidate_ids=ranked,
        ),
        candidates=candidates,
    )
    assert plan.status == "feasible"
    assert plan.selected_poi_ids == ranked


def test_tuesday_new_fortress_closure_is_not_scheduled(
    repository: POIRepository,
) -> None:
    plan = DeterministicItineraryPlanner(
        repository, travel_time_estimator=FixedTravelEstimator()
    ).plan(request(required_poi_ids=["CF-002"]))

    assert plan.status == "infeasible"
    assert any(issue.code == "poi_closed" for issue in plan.validation.issues)


def test_tuesday_historical_route_skips_closed_new_fortress(
    repository: POIRepository,
) -> None:
    plan = DeterministicItineraryPlanner(
        repository, travel_time_estimator=FixedTravelEstimator(5)
    ).plan(
        request(
            start_time="11:00",
            end_time="15:00",
            hard_filters=POIMetadataFilters(category="history"),
        )
    )

    assert plan.itinerary is not None
    assert "CF-002" not in plan.selected_poi_ids
    assert all(
        repository.get_by_id(poi_id).category == "history"
        for poi_id in plan.selected_poi_ids
    )


@pytest.mark.parametrize(
    ("start", "end"),
    [("10:00", "10:55"), ("16:05", "17:00")],
)
def test_opening_time_boundaries_are_inclusive(
    repository: POIRepository,
    start: str,
    end: str,
) -> None:
    plan = DeterministicItineraryPlanner(
        repository, travel_time_estimator=FixedTravelEstimator()
    ).plan(
        request(
            start_time=start,
            end_time=end,
            required_poi_ids=["CF-007"],
        )
    )

    assert plan.status == "feasible"
    assert plan.itinerary is not None
    assert plan.itinerary.stops[0].starts_at.strftime("%H:%M") == start
    assert plan.itinerary.stops[0].ends_at.strftime("%H:%M") == end


def test_previous_day_overnight_window_covers_early_morning_visit(
    dataset: CorfuDataset,
) -> None:
    original = next(poi for poi in dataset.pois if poi.id == "CF-007")
    schedule = original.opening_hours_fixture
    assert schedule is not None
    updated_schedule = schedule.model_copy(
        update={
            "mon": OpeningWindow.model_validate("22:00-02:00"),
            "tue": None,
        }
    )
    updated_poi = POI.model_validate(
        {**original.model_dump(), "opening_hours_fixture": updated_schedule.model_dump()}
    )
    updated_dataset = dataset.model_copy(
        update={
            "pois": [
                updated_poi if poi.id == updated_poi.id else poi for poi in dataset.pois
            ]
        }
    )
    repository = POIRepository(updated_dataset)

    plan = DeterministicItineraryPlanner(
        repository, travel_time_estimator=FixedTravelEstimator()
    ).plan(
        request(
            start_time="00:00",
            end_time="03:00",
            required_poi_ids=["CF-007"],
        ),
        candidates=[updated_poi],
    )

    assert plan.status == "feasible"
    assert plan.itinerary is not None
    assert plan.itinerary.stops[0].starts_at.strftime("%H:%M") == "00:00"


def test_unknown_hours_are_conditional_or_excluded(repository: POIRepository) -> None:
    unknown_poi = next(
        poi for poi in repository.get_all() if poi.opening_hours_fixture is None
    )
    planner = DeterministicItineraryPlanner(
        repository, travel_time_estimator=FixedTravelEstimator()
    )

    conditional = planner.plan(request(required_poi_ids=[unknown_poi.id]))
    strict = planner.plan(
        request(
            required_poi_ids=[unknown_poi.id],
            unknown_hours_policy="exclude",
        )
    )

    assert conditional.status == "uncertain"
    assert conditional.itinerary is not None
    assert any(
        issue.code == "unknown_opening_hours"
        for issue in conditional.validation.issues
    )
    assert strict.status == "infeasible"
    assert strict.itinerary is None


def test_family_requirement_excludes_unknown_and_false_candidates(
    repository: POIRepository,
) -> None:
    candidates = [
        poi
        for poi in repository.get_all()
        if poi.setting == "indoor" and "museum" in poi.tags
    ]
    plan = DeterministicItineraryPlanner(
        repository, travel_time_estimator=FixedTravelEstimator()
    ).plan(
        request(
            hard_filters=POIMetadataFilters(tags=["museum"], setting="indoor"),
            family_required=True,
        ),
        candidates=candidates,
    )

    assert plan.itinerary is not None
    assert plan.selected_poi_ids
    assert all(
        repository.get_by_id(poi_id).family_suitability_fixture is True
        for poi_id in plan.selected_poi_ids
    )


def test_no_matching_candidates_is_an_explicit_infeasible_outcome(
    repository: POIRepository,
) -> None:
    plan = DeterministicItineraryPlanner(
        repository, travel_time_estimator=FixedTravelEstimator()
    ).plan(request(), candidates=[])

    assert plan.status == "infeasible"
    assert plan.itinerary is None
    assert plan.validation.issues[0].code == "no_eligible_candidates"


def test_required_visit_duration_cannot_be_shortened(repository: POIRepository) -> None:
    plan = DeterministicItineraryPlanner(
        repository, travel_time_estimator=FixedTravelEstimator()
    ).plan(
        request(
            start_time="11:00",
            end_time="11:30",
            required_poi_ids=["CF-001"],
        )
    )

    assert plan.status == "infeasible"
    assert plan.validation.issues[0].code == "insufficient_overall_time"


def test_metadata_filters_are_applied_before_candidate_ranking(
    repository: POIRepository,
) -> None:
    candidates = list(repository.get_all())
    plan = DeterministicItineraryPlanner(
        repository, travel_time_estimator=FixedTravelEstimator()
    ).plan(
        request(
            hard_filters=POIMetadataFilters(setting="indoor", tags=["museum"]),
            ranked_candidate_ids=["CF-001", "CF-007"],
        ),
        candidates=candidates,
    )

    assert plan.itinerary is not None
    assert all(
        repository.get_by_id(poi_id).setting == "indoor"
        and "museum" in repository.get_by_id(poi_id).tags
        for poi_id in plan.selected_poi_ids
    )


def test_distant_pois_take_longer_to_walk_than_same_location(
    repository: POIRepository,
) -> None:
    estimator = HaversineWalkingTimeEstimator()
    old_fortress = repository.get_by_id("CF-001")
    nearby = repository.get_by_id("CF-017")
    achilleion = repository.get_by_id("CF-021")
    assert old_fortress is not None and nearby is not None and achilleion is not None

    same = estimator.estimate(old_fortress, old_fortress, "walking")
    near = estimator.estimate(old_fortress, nearby, "walking")
    far = estimator.estimate(old_fortress, achilleion, "walking")

    assert same.minutes == 0
    assert far.minutes > near.minutes > 0


def test_planning_output_is_deterministic(repository: POIRepository) -> None:
    planner = DeterministicItineraryPlanner(
        repository, travel_time_estimator=FixedTravelEstimator(3)
    )
    request_value = request(
        required_poi_ids=["CF-001", "CF-017"],
        ranked_candidate_ids=["CF-017", "CF-001"],
    )

    first = planner.plan(request_value)
    second = planner.plan(request_value)

    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_planner_reports_unused_window_without_padding_with_extra_stops(
    repository: POIRepository,
) -> None:
    poi = repository.get_by_id("CF-032")
    assert poi is not None
    result = DeterministicItineraryPlanner(
        repository, travel_time_estimator=FixedTravelEstimator()
    ).plan(
        request(
            start_time="13:00",
            end_time="17:00",
            max_stops=1,
            ranked_candidate_ids=[poi.id],
        ),
        candidates=[poi],
    )

    assert result.status == "feasible"
    assert result.selected_poi_ids == [poi.id]
    assert any(
        "Scheduled visits finish at 13:25, leaving 215 minutes unallocated" in note
        for note in result.explanations
    )


def test_preserved_stops_keep_their_original_times(repository: POIRepository) -> None:
    original_stop = ItineraryStop(
        poi_id="CF-001",
        starts_at=datetime(2026, 10, 6, 11, 0, tzinfo=TZ),
        ends_at=datetime(2026, 10, 6, 12, 25, tzinfo=TZ),
    )
    planner = DeterministicItineraryPlanner(
        repository, travel_time_estimator=FixedTravelEstimator()
    )

    result = planner.plan(
        request(
            start_time="11:00",
            end_time="15:00",
            preserved_stops=[original_stop],
        )
    )

    assert result.itinerary is not None
    retained = next(stop for stop in result.itinerary.stops if stop.poi_id == "CF-001")
    assert retained.starts_at == original_stop.starts_at
    assert retained.ends_at == original_stop.ends_at
    assert original_stop.starts_at.strftime("%H:%M") == "11:00"


def test_validator_detects_travel_time_closure_and_missing_required(
    repository: POIRepository,
) -> None:
    request_value = request(
        start_time="10:00",
        end_time="14:00",
        required_poi_ids=["CF-017", "CF-001", "CF-002", "CF-021"],
    )
    itinerary = Itinerary(
        timezone="Europe/Athens",
        starts_at=datetime(2026, 10, 6, 10, 0, tzinfo=TZ),
        ends_at=datetime(2026, 10, 6, 14, 0, tzinfo=TZ),
        stops=[
            ItineraryStop(
                poi_id="CF-017",
                starts_at=datetime(2026, 10, 6, 10, 0, tzinfo=TZ),
                ends_at=datetime(2026, 10, 6, 11, 5, tzinfo=TZ),
            ),
            ItineraryStop(
                poi_id="CF-001",
                starts_at=datetime(2026, 10, 6, 11, 5, tzinfo=TZ),
                ends_at=datetime(2026, 10, 6, 12, 30, tzinfo=TZ),
            ),
            ItineraryStop(
                poi_id="CF-002",
                starts_at=datetime(2026, 10, 6, 12, 30, tzinfo=TZ),
                ends_at=datetime(2026, 10, 6, 13, 35, tzinfo=TZ),
            ),
        ],
    )
    result = ItineraryValidator(
        repository, FixedTravelEstimator(minutes=20)
    ).validate(request_value, itinerary)

    codes = {issue.code for issue in result.issues}
    assert not result.is_valid
    assert "insufficient_travel_time" in codes
    assert "poi_closed" in codes
    assert "required_stop_missing" in codes


def test_validator_detects_duplicates_and_metadata_violations(
    repository: POIRepository,
) -> None:
    request_value = request(
        start_time="10:00",
        end_time="14:00",
        hard_filters=POIMetadataFilters(setting="outdoor"),
    )
    itinerary = Itinerary(
        timezone="Europe/Athens",
        starts_at=datetime(2026, 10, 6, 10, 0, tzinfo=TZ),
        ends_at=datetime(2026, 10, 6, 14, 0, tzinfo=TZ),
        stops=[
            ItineraryStop(
                poi_id="CF-017",
                starts_at=datetime(2026, 10, 6, 10, 0, tzinfo=TZ),
                ends_at=datetime(2026, 10, 6, 11, 5, tzinfo=TZ),
            ),
            ItineraryStop(
                poi_id="CF-017",
                starts_at=datetime(2026, 10, 6, 11, 5, tzinfo=TZ),
                ends_at=datetime(2026, 10, 6, 12, 10, tzinfo=TZ),
            ),
        ],
    )
    result = ItineraryValidator(
        repository, FixedTravelEstimator()
    ).validate(request_value, itinerary)

    codes = {issue.code for issue in result.issues}
    assert "duplicate_stop" in codes
    assert "setting_constraint_violation" in codes


def test_validator_rejects_subminute_visit_past_closing_time(
    repository: POIRepository,
) -> None:
    request_value = request(
        start_time="16:05",
        end_time="18:00",
        required_poi_ids=["CF-007"],
    )
    itinerary = Itinerary(
        timezone="Europe/Athens",
        starts_at=datetime(2026, 10, 6, 16, 5, tzinfo=TZ),
        ends_at=datetime(2026, 10, 6, 18, 0, tzinfo=TZ),
        stops=[
            ItineraryStop(
                poi_id="CF-007",
                starts_at=datetime(2026, 10, 6, 16, 5, tzinfo=TZ),
                ends_at=datetime(2026, 10, 6, 17, 0, 30, tzinfo=TZ),
            )
        ],
    )

    result = ItineraryValidator(
        repository, FixedTravelEstimator()
    ).validate(request_value, itinerary)

    assert "subminute_stop_boundary" in {issue.code for issue in result.issues}


def test_replacement_mode_preserves_stop_order_and_scopes_family_filter(
    repository: POIRepository,
) -> None:
    originals = [
        ItineraryStop(
            poi_id=poi_id,
            starts_at=datetime(2026, 10, 6, hour, minute, tzinfo=TZ),
            ends_at=datetime(2026, 10, 6, end_hour, end_minute, tzinfo=TZ),
        )
        for poi_id, hour, minute, end_hour, end_minute in (
            ("CF-003", 11, 0, 11, 20),
            ("CF-006", 11, 25, 11, 40),
            ("CF-009", 11, 45, 12, 5),
        )
    ]
    replacement = repository.get_by_id("CF-007")
    assert replacement is not None
    plan = DeterministicItineraryPlanner(
        repository, travel_time_estimator=FixedTravelEstimator(2)
    ).plan(
        request(
            start_time="11:00",
            end_time="15:00",
            preserved_stops=originals,
            replacement_stop_index=1,
            replacement_family_required=True,
            replacement_hard_filters=POIMetadataFilters(category="history"),
            ranked_candidate_ids=["CF-007"],
            max_stops=3,
        ),
        candidates=[replacement],
    )

    assert plan.status == "feasible"
    assert plan.itinerary is not None
    assert [stop.poi_id for stop in plan.itinerary.stops] == [
        "CF-003",
        "CF-007",
        "CF-009",
    ]
    assert repository.get_by_id("CF-007").family_suitability_fixture is True
    assert plan.itinerary.stops[0].starts_at == originals[0].starts_at
    assert plan.itinerary.stops[2].starts_at > originals[2].starts_at
    assert len(plan.transfers) == 2


def test_replacement_prefers_schedulable_known_hours_before_unknown_hours(
    repository: POIRepository,
) -> None:
    originals = [
        ItineraryStop(
            poi_id=poi_id,
            starts_at=datetime(2026, 10, 6, hour, minute, tzinfo=TZ),
            ends_at=datetime(2026, 10, 6, end_hour, end_minute, tzinfo=TZ),
        )
        for poi_id, hour, minute, end_hour, end_minute in (
            ("CF-003", 10, 0, 10, 20),
            ("CF-006", 10, 25, 10, 40),
            ("CF-009", 10, 45, 11, 5),
        )
    ]
    planner = DeterministicItineraryPlanner(
        repository, travel_time_estimator=FixedTravelEstimator(2)
    )

    known_hours = planner.plan(
        request(
            start_time="10:00",
            end_time="14:00",
            preserved_stops=originals,
            replacement_stop_index=1,
            replacement_hard_filters=POIMetadataFilters(setting="indoor"),
            ranked_candidate_ids=["CF-004", "CF-007"],
            max_stops=3,
        ),
        candidates=[repository.get_by_id("CF-004"), repository.get_by_id("CF-007")],
    )
    unknown_only = planner.plan(
        request(
            start_time="10:00",
            end_time="14:00",
            preserved_stops=originals,
            replacement_stop_index=1,
            replacement_hard_filters=POIMetadataFilters(setting="indoor"),
            ranked_candidate_ids=["CF-004"],
            max_stops=3,
        ),
        candidates=[repository.get_by_id("CF-004")],
    )

    assert known_hours.status == "feasible"
    assert known_hours.selected_poi_ids[1] == "CF-007"
    assert unknown_only.status == "uncertain"
    assert unknown_only.selected_poi_ids[1] == "CF-004"


def test_stop_replacement_retimes_unaffected_stops_and_removes_stale_idle_gap(
    repository: POIRepository,
) -> None:
    originals = [
        ItineraryStop(
            poi_id=poi_id,
            starts_at=datetime(2026, 10, 6, hour, minute, tzinfo=TZ),
            ends_at=datetime(2026, 10, 6, end_hour, end_minute, tzinfo=TZ),
        )
        for poi_id, hour, minute, end_hour, end_minute in (
            ("CF-032", 11, 0, 11, 25),
            ("CF-005", 11, 37, 12, 12),
            ("CF-036", 12, 35, 12, 55),
        )
    ]
    planner = DeterministicItineraryPlanner(
        repository, travel_time_estimator=FixedTravelEstimator(12)
    )

    plan = planner.plan(
        request(
            start_time="11:00",
            end_time="15:00",
            preserved_stops=originals,
            replacement_stop_index=1,
            replacement_required_poi_ids=["CF-003"],
            ranked_candidate_ids=["CF-003"],
            max_stops=3,
        ),
        candidates=[repository.get_by_id("CF-003")],
    )

    assert plan.status == "feasible"
    assert plan.itinerary is not None
    assert [stop.poi_id for stop in plan.itinerary.stops] == ["CF-032", "CF-003", "CF-036"]
    assert plan.itinerary.stops[1].starts_at == datetime(2026, 10, 6, 11, 37, tzinfo=TZ)
    assert plan.itinerary.stops[2].starts_at == datetime(2026, 10, 6, 12, 9, tzinfo=TZ)
    assert plan.waiting_periods == []


def test_replacement_reports_wait_required_by_fixture_opening_window(
    repository: POIRepository,
) -> None:
    originals = [
        ItineraryStop(
            poi_id=poi_id,
            starts_at=datetime(2026, 10, 6, hour, minute, tzinfo=TZ),
            ends_at=datetime(2026, 10, 6, end_hour, end_minute, tzinfo=TZ),
        )
        for poi_id, hour, minute, end_hour, end_minute in (
            ("CF-032", 11, 0, 11, 25),
            ("CF-005", 11, 37, 12, 12),
            ("CF-036", 12, 35, 12, 55),
        )
    ]
    planner = DeterministicItineraryPlanner(
        repository, travel_time_estimator=FixedTravelEstimator(12)
    )

    plan = planner.plan(
        request(
            start_time="11:00",
            end_time="15:00",
            preserved_stops=originals,
            replacement_stop_index=1,
            replacement_required_poi_ids=["CF-042"],
            ranked_candidate_ids=["CF-042"],
            max_stops=3,
        ),
        candidates=[repository.get_by_id("CF-042")],
    )

    assert plan.status == "feasible"
    assert plan.itinerary is not None
    assert plan.itinerary.stops[1].starts_at == datetime(2026, 10, 6, 12, tzinfo=TZ)
    assert len(plan.waiting_periods) == 1
    wait = plan.waiting_periods[0]
    assert wait.poi_id == "CF-042"
    assert wait.starts_at == datetime(2026, 10, 6, 11, 37, tzinfo=TZ)
    assert wait.ends_at == datetime(2026, 10, 6, 12, tzinfo=TZ)
    assert wait.reason == "opening_hours"


def test_schedulable_semantic_rank_precedes_shorter_walking_transfer(
    repository: POIRepository,
) -> None:
    class UnequalTravelEstimator:
        def estimate(self, origin: POI, destination: POI, mode: TravelMode) -> TravelEstimate:
            minutes = 20 if destination.id == "CF-034" else 1
            return TravelEstimate(
                straight_line_distance_km=0.2,
                estimated_walking_distance_km=0.28,
                minutes=minutes,
            )

    preserved = ItineraryStop(
        poi_id="CF-036",
        starts_at=datetime(2026, 10, 6, 10, 0, tzinfo=TZ),
        ends_at=datetime(2026, 10, 6, 10, 20, tzinfo=TZ),
    )
    planner = DeterministicItineraryPlanner(
        repository, travel_time_estimator=UnequalTravelEstimator()
    )

    plan = planner.plan(
        request(
            start_time="10:00",
            end_time="12:00",
            preserved_stops=[preserved],
            ranked_candidate_ids=["CF-034", "CF-028"],
            max_stops=2,
        ),
        candidates=[repository.get_by_id("CF-028"), repository.get_by_id("CF-034")],
    )

    assert plan.status == "feasible"
    assert plan.selected_poi_ids == ["CF-036", "CF-034"]


def test_replacement_mode_rejects_closed_or_too_long_required_candidate(
    repository: POIRepository,
) -> None:
    base = datetime(2026, 10, 6, 11, 0, tzinfo=TZ)
    originals = [
        ItineraryStop(
            poi_id=poi_id,
            starts_at=base + timedelta(minutes=offset),
            ends_at=base + timedelta(minutes=offset + 15),
        )
        for poi_id, offset in (("CF-006", 0), ("CF-003", 20), ("CF-009", 40))
    ]
    planner = DeterministicItineraryPlanner(
        repository, travel_time_estimator=FixedTravelEstimator(3)
    )

    closed = planner.plan(
        request(
            start_time="11:00",
            end_time="13:00",
            preserved_stops=originals,
            replacement_stop_index=1,
            replacement_required_poi_ids=["CF-002"],
            max_stops=3,
        ),
        candidates=[repository.get_by_id("CF-002")],
    )
    too_long = planner.plan(
        request(
            start_time="11:00",
            end_time="12:00",
            preserved_stops=originals,
            replacement_stop_index=1,
            replacement_required_poi_ids=["CF-001"],
            max_stops=3,
        ),
        candidates=[repository.get_by_id("CF-001")],
    )

    assert closed.status == "infeasible"
    assert any(issue.code == "poi_closed" for issue in closed.validation.issues)
    assert too_long.status == "infeasible"
    assert any(
        issue.code == "replacement_time_window_conflict"
        for issue in too_long.validation.issues
    )
