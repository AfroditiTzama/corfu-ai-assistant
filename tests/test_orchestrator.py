"""New orchestration contract regressions (not a reconstruction of the lost file).

Uses the real Corfu dataset/planner and deterministic embedding fixture; external
Gemini/weather services are injected so these checks are repeatable.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pytest
from numpy.typing import NDArray

from corfu_assistant.data import load_dataset
from corfu_assistant.intent_extractor import IntentExtractionError
from corfu_assistant.intent_policy import normalize_thematic_filters
from corfu_assistant.models import ConversationState, UserIntent
from corfu_assistant.orchestrator import (
    GroundedAnswer, GroundedClaim, GroundedCitation, OrchestrationError,
    Orchestrator, POIEvidence,
)
from corfu_assistant.planner import DeterministicItineraryPlanner, HaversineWalkingTimeEstimator
from corfu_assistant.repository import POIRepository
from corfu_assistant.state_manager import ConversationStateManager
from corfu_assistant.weather import FixtureWeatherProvider, WeatherProviderError, assess_rain

ROOT = Path(__file__).resolve().parents[1]
ACTIONS = (
    "requested_date start_time end_time timezone semantic_interests filter_category filter_tags "
    "filter_setting filter_family_suitability filter_excluded_categories filter_excluded_tags "
    "filter_excluded_poi_ids travel_mode family_required child_ages itinerary_modification "
    "question clarification_question weather_request"
).split()
TZ = ZoneInfo("Europe/Athens")


class FixedEmbeddings:
    """Deterministic semantic stub; exercises repository filtering/ranking plumbing only."""
    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        rng = np.random.default_rng(17)
        return rng.normal(size=(len(texts), 12)).astype(np.float32)


class Extractor:
    def __init__(self, *items): self.items = list(items)
    def extract(self, message: str, conversation_state: ConversationState | None = None):
        item = self.items.pop(0)
        if isinstance(item, Exception): raise item
        return item


class Generator:
    def __init__(self, answer): self.answer = answer
    def generate(self, question, evidence, recommendation_count=None): return self.answer


def intent(kind="poi_recommendation", values=None, actions=None):
    data = dict(values or {})
    patch = dict.fromkeys(ACTIONS, "preserve")
    patch.update(actions or {})
    data.update(intent=kind, field_actions=patch)
    return UserIntent.model_validate(data)


def repo():
    return POIRepository(load_dataset(ROOT / "corfu_dataset.json"), embedding_model=FixedEmbeddings())


def make_orch(items, repository=None, manager=None, weather=None, generator=None):
    repository = repository or repo()
    manager = manager or ConversationStateManager()
    planner = DeterministicItineraryPlanner(repository, travel_time_estimator=HaversineWalkingTimeEstimator())
    return Orchestrator(intent_extractor=Extractor(*items), state_manager=manager,
                        poi_repository=repository, planner=planner,
                        weather_provider=weather, answer_generator=generator), repository, manager


def plan_intent(day="2026-10-07", start="13:00", end="17:00", filters=None, interests=None, family=False):
    values = {"requested_date": day, "time_window": {"start_time": start, "end_time": end},
              "travel_mode": "walking"}
    actions = {"requested_date":"set", "start_time":"set", "end_time":"set", "travel_mode":"set"}
    if filters is not None: values["hard_filters"] = filters; actions.update({"filter_"+k:"set" for k in filters})
    if interests is not None: values["semantic_preferences"] = {"interests": interests}; actions["semantic_interests"]="set"
    if family: values["family_requirements"]={"required":True}; actions["family_required"]="set"
    return intent("itinerary_creation", values, actions)


def test_demo_a_monday_theme_respects_fixture_closures_and_selects_museum_candidates():
    r=repo(); orch,_,_=make_orch([plan_intent(day="2026-10-05", interests=["art", "museums"])],r)
    result=orch.process({"message":"Plan a four-hour visit to Corfu focused on art and museums on Monday, October 5, 2026, from 13:00 to 17:00"})
    assert result.status in {"feasible","uncertain"}
    assert result.planning_result and result.planning_result.status in {"feasible", "uncertain"}
    assert result.intent.hard_filters is None or not result.intent.hard_filters.tags
    selected=[r.get_by_id(pid) for pid in result.planning_result.selected_poi_ids]
    assert len(selected) >= 2 and all(set(p.tags) & {"museum", "art", "gallery", "painting", "studio"} for p in selected)
    assert "CF-017" not in result.planning_result.selected_poi_ids  # Monday closed fixture


def test_demo_d_hard_setting_and_soft_history_do_not_require_sparse_history_tag():
    r=repo()
    assert len(r.filter_metadata({"setting":"outdoor"})) == 22
    assert len(r.filter_metadata({"setting":"outdoor","tags":["history"]})) == 2
    orch,_,_=make_orch([plan_intent(filters={"setting":"outdoor"},interests=["history"])],r)
    result=orch.process({"message":"Outdoor historical attractions Wednesday 13:00-17:00"})
    assert result.planning_result and result.planning_result.validation.is_valid
    assert all(r.get_by_id(pid).setting=="outdoor" for pid in result.planning_result.selected_poi_ids)


def test_open_ended_discovery_prefers_attractions_without_excluding_hospitality_globally():
    r = repo()
    discovery = intent(
        "poi_recommendation",
        {"semantic_preferences": {"interests": ["Corfu discovery"]}},
        {"semantic_interests": "set"},
    ).model_copy(update={"constraint_evidence": []})
    food = intent(
        "poi_recommendation",
        {
            "semantic_preferences": {"interests": ["local dining"]},
            "hard_filters": {"category": "food"},
        },
        {"semantic_interests": "set", "filter_category": "set"},
    )
    food = type(food).model_validate({
        **food.model_dump(),
        "constraint_evidence": [{
            "field": "filter_category", "value": "food", "kind": "required",
            "quote": "only food places",
        }],
    })
    orch, _, _ = make_orch([discovery, food], r)

    first = orch.process({"message": "Suggest places to explore and visit around Corfu"})
    hospitality_tags = {"food", "restaurant", "cafe", "coffee", "bar"}
    assert first.recommended_poi_ids
    assert all(not (set(r.get_by_id(poi_id).tags) & hospitality_tags)
               for poi_id in first.recommended_poi_ids)

    second = orch.process({"message": "Recommend only food places for local dining"})
    assert second.recommended_poi_ids
    assert all(r.get_by_id(poi_id).category == "food"
               for poi_id in second.recommended_poi_ids)


def test_implicit_history_tag_becomes_soft_interest_but_explicit_tag_stays_hard():
    r=repo()
    raw=intent("itinerary_creation", {"hard_filters":{"tags":["history"],"setting":"outdoor"}},
              {"filter_tags":"set","filter_setting":"set"})
    soft, demoted=normalize_thematic_filters(raw,message="focused on outdoor historical attractions",
        category_vocabulary=r.metadata_categories,tag_vocabulary=r.metadata_tags)
    assert demoted==("history",) and soft.hard_filters.tags is None
    assert soft.hard_filters.setting=="outdoor" and soft.semantic_preferences.interests==["history"]
    hard,_=normalize_thematic_filters(raw,message="only places tagged history outdoors",
        category_vocabulary=r.metadata_categories,tag_vocabulary=r.metadata_tags)
    assert hard.hard_filters.tags==["history"]
    exclusive,_=normalize_thematic_filters(raw,message="Only outdoor historical attractions",
        category_vocabulary=r.metadata_categories,tag_vocabulary=r.metadata_tags)
    assert exclusive.hard_filters.category=="history"
    assert exclusive.hard_filters.setting=="outdoor"
    assert exclusive.hard_filters.tags is None


def test_soft_art_museum_tags_from_gemini_do_not_become_hard_filters():
    r=repo()
    raw=intent("itinerary_creation", {"hard_filters":{"tags":["art","museum"],"setting":"indoor"}},
              {"filter_tags":"set","filter_setting":"set"})
    normalized, demoted=normalize_thematic_filters(raw,
        message="focused on art and museums",category_vocabulary=r.metadata_categories,
        tag_vocabulary=r.metadata_tags)
    assert set(demoted)=={"art","museum"}
    assert normalized.hard_filters.tags is None
    assert normalized.hard_filters.setting=="indoor"
    assert normalized.semantic_preferences.interests==["art","museum"]


def test_explicit_only_museums_remains_a_hard_constraint():
    r = repo()
    raw = intent("itinerary_creation", {"hard_filters": {"tags": ["museum"]}},
                 {"filter_tags": "set"})
    normalized, demoted = normalize_thematic_filters(
        raw, message="I am focused on only museums",
        category_vocabulary=r.metadata_categories, tag_vocabulary=r.metadata_tags,
    )
    assert not demoted
    assert normalized.hard_filters.tags == ["museum"]


def test_explicit_hard_category_and_setting_are_preserved():
    r=repo(); orch,_,_=make_orch([plan_intent(filters={"category":"history","setting":"outdoor"})],r)
    out=orch.process({"message":"Only outdoor history category attractions"})
    assert out.intent.hard_filters.category=="history"
    assert out.intent.hard_filters.setting=="outdoor"
    assert all(r.get_by_id(pid).category=="history" and r.get_by_id(pid).setting=="outdoor" for pid in out.planning_result.selected_poi_ids)


def test_infeasible_required_stops_are_rejected_with_validation_issue():
    r=repo(); required=["CF-001","CF-002","CF-021","CF-022","CF-017"]
    values={"requested_date":"2026-10-06","time_window":{"start_time":"11:00","end_time":"12:30"},"required_poi_ids":required}
    # Required IDs are extracted from exact names in the prompt by production orchestration.
    orch,_,_=make_orch([plan_intent(day="2026-10-06",start="11:00",end="12:30")],r)
    out=orch.process({"message":"Visit Corfu Museum, New Fortress, Achilleion Palace, Mon Repos Museum and Museum of Asian Art on Tuesday October 6 from 11:00 to 12:30"})
    assert out.status in {"infeasible","clarification_required"}
    if out.planning_result: assert not out.planning_result.validation.is_valid


def test_failed_intent_extraction_does_not_change_existing_state():
    first=plan_intent(interests=["history"])
    orch,r,manager=make_orch([first,IntentExtractionError("provider failed")])
    one=orch.process({"message":"Plan outdoor history Wednesday 13:00-17:00"})
    before=manager.get(one.conversation_id)
    with pytest.raises(IntentExtractionError): orch.process({"conversation_id":one.conversation_id,"message":"replace a stop"})
    assert manager.get(one.conversation_id)==before


def test_sessions_are_isolated():
    orch,_,_=make_orch([intent("clarification",{"clarification_question":"Which date?"},{"clarification_question":"set"}) for _ in range(2)])
    a=orch.process({"message":"first"}); b=orch.process({"message":"second"})
    assert a.conversation_id != b.conversation_id


def test_short_reply_resolves_stored_clarification_without_losing_original_request():
    clarification = intent(
        "clarification",
        {"clarification_question": "Should these be preferences or requirements?"},
        {"clarification_question": "set"},
    )
    resolved = intent(
        "poi_recommendation",
        {
            "semantic_preferences": {"interests": ["cultural attractions"]},
        },
        {"semantic_interests": "set"},
    ).model_copy(update={"constraint_evidence": []})

    class ClarificationAwareExtractor:
        def __init__(self): self.calls = 0
        def extract(self, message, conversation_state=None):
            self.calls += 1
            if self.calls == 1:
                return clarification
            assert message == "preferences"
            assert conversation_state is not None
            assert conversation_state.pending_clarification is not None
            assert conversation_state.pending_clarification.original_message == (
                "Suggest cultural attractions for an afternoon visit"
            )
            return resolved

    repository = repo()
    manager = ConversationStateManager()
    extractor = ClarificationAwareExtractor()
    orch = Orchestrator(
        intent_extractor=extractor,
        state_manager=manager,
        poi_repository=repository,
        planner=DeterministicItineraryPlanner(repository),
    )

    first = orch.process({"message": "Suggest cultural attractions for an afternoon visit"})
    assert first.status == "clarification_required"
    assert first.conversation_state.pending_clarification is not None

    second = orch.process({
        "conversation_id": first.conversation_id,
        "message": "preferences",
    })
    assert second.status == "completed"
    assert second.intent.intent == "poi_recommendation"
    assert second.conversation_state.preferences.interests == ["cultural attractions"]
    assert second.conversation_state.pending_clarification is None
    assert second.recommended_poi_ids


def test_grounded_answer_rejects_unsupported_citation_and_falls_back():
    r=repo(); poi=r.get_by_id("CF-001")
    evidence=POIEvidence(poi_id=poi.id,name=poi.name,description=poi.description,source_urls=poi.source_urls,
                         description_provenance=poi.description_provenance,method="exact_name")
    gen=Generator(GroundedAnswer(claims=[GroundedClaim(text="Unsupported fact",citations=[GroundedCitation(poi_id="CF-999",supporting_quote="made up quote")])]))
    orch,_,_=make_orch([intent("tourism_qa",{"question":"Tell me about Old Fortress"},{"question":"set"})],r,generator=gen)
    out=orch.process({"message":"Tell me about Old Fortress"})
    assert out.answer_mode=="deterministic" or out.status=="no_evidence"


def test_weather_fixture_source_and_replanning_are_explicit():
    r=repo(); manager=ConversationStateManager()
    create=plan_intent(filters={"setting":"outdoor"},interests=["history"])
    provider=FixtureWeatherProvider(ROOT/"corfu_synthetic_rain.json",clock=lambda:datetime(2026,10,7,12,tzinfo=TZ))
    orch,_,_=make_orch([create,intent("weather_based_replanning")],r,manager,provider)
    first=orch.process({"message":"Plan outdoor historical attractions Wednesday October 7 13:00-17:00"})
    second=orch.process({"conversation_id":first.conversation_id,"message":"Check weather and adjust the walk for rain"})
    assert second.weather_forecast and second.weather_forecast.source=="synthetic_fixture"
    assert second.weather_replanning is not None
    assert second.conversation_state.itinerary is not None
    assert second.conversation_state.preferences.interests == first.conversation_state.preferences.interests
    original_ids = {stop.poi_id for stop in first.conversation_state.itinerary.stops}
    replacements = [r.get_by_id(stop.poi_id) for stop in second.conversation_state.itinerary.stops
                    if stop.poi_id not in original_ids]
    assert replacements, "The fixture must exercise actual replacement selection"
    assert all(poi.setting == "indoor" and poi.category == "history" for poi in replacements)
    assert all(poi.category != "food" for poi in replacements)
    assert any(assess_rain(second.weather_forecast, transfer.starts_at, transfer.ends_at).rain_forecast
               for transfer in second.planning_result.transfers)
    assert any("walking transfers" in explanation
               for explanation in second.weather_replanning.explanations)


def test_textual_soft_history_focus_is_persisted_and_inherited_by_weather_replanning():
    r = repo()
    manager = ConversationStateManager()
    # Simulate valid structured extraction that omitted semantic interests.
    # The user message still explicitly supplies a soft theme.
    create = plan_intent(filters={"setting": "outdoor"})
    follow_up = intent("weather_based_replanning")
    provider = FixtureWeatherProvider(
        ROOT / "corfu_synthetic_rain.json",
        clock=lambda: datetime(2026, 10, 7, 12, tzinfo=TZ),
    )
    orch, _, _ = make_orch([create, follow_up], r, manager, provider)

    first = orch.process({
        "message": "Plan a walk focused on outdoor historical attractions on Wednesday, "
                   "October 7, 2026, from 13:00 to 17:00.",
    })

    assert first.conversation_state.preferences.interests == ["history"]
    assert first.conversation_state.active_request.interests == ["history"]
    assert first.intent.hard_filters.tags is None

    second = orch.process({
        "conversation_id": first.conversation_id,
        "message": "Check the weather and adapt my existing itinerary if needed.",
    })

    assert second.conversation_state.preferences.interests == ["history"]
    assert second.weather_forecast.source == "synthetic_fixture"
    assert second.weather_replanning is not None
    original_ids = set(first.planning_result.selected_poi_ids)
    replacements = [
        r.get_by_id(poi_id)
        for poi_id in second.planning_result.selected_poi_ids
        if poi_id not in original_ids
    ]
    assert replacements
    assert all(poi.setting == "indoor" and poi.category == "history" for poi in replacements)
    assert second.planning_result.validation.is_valid
    assert second.conversation_state.itinerary.ends_at <= datetime(
        2026, 10, 7, 17, tzinfo=TZ
    )


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Plan a visit focused on art and museums", {"art", "museum"}),
        ("Plan a walk focused on nature", {"nature"}),
        ("Plan a visit focused on food and dining", {"food"}),
    ],
)
def test_omitted_soft_focus_is_recovered_as_interest_not_hard_filter(message, expected):
    r = repo()
    raw = intent("poi_recommendation")  # extractor omitted the preference
    normalized, _ = normalize_thematic_filters(
        raw,
        message=message,
        category_vocabulary=r.metadata_categories,
        tag_vocabulary=r.metadata_tags,
    )
    assert expected <= {value.casefold() for value in normalized.semantic_preferences.interests}
    assert normalized.field_actions.semantic_interests == "set"
    assert normalized.hard_filters is None


def test_current_soft_focus_overrides_extractor_empty_clear_but_user_clear_remains_effective():
    r = repo()
    extractor_said_clear = intent("poi_recommendation", actions={"semantic_interests": "clear"})
    focused, _ = normalize_thematic_filters(
        extractor_said_clear,
        message="Now focus on nature",
        category_vocabulary=r.metadata_categories,
        tag_vocabulary=r.metadata_tags,
    )
    assert focused.semantic_preferences.interests == ["nature"]

    extractor_said_empty_set = intent(
        "poi_recommendation",
        {"semantic_preferences": {"interests": []}},
        {"semantic_interests": "set"},
    )
    focused_after_empty_set, _ = normalize_thematic_filters(
        extractor_said_empty_set,
        message="Now focus on history",
        category_vocabulary=r.metadata_categories,
        tag_vocabulary=r.metadata_tags,
    )
    assert focused_after_empty_set.semantic_preferences.interests == ["history"]

    manager = ConversationStateManager()
    initial = manager.create_conversation()
    set_history = intent(
        "poi_recommendation",
        {"semantic_preferences": {"interests": ["history"]}},
        {"semantic_interests": "set"},
    )
    with_history = manager.apply_intent(initial.conversation_id, set_history, message="History").state
    extractor_omitted_change = intent("poi_recommendation")
    change, _ = normalize_thematic_filters(
        extractor_omitted_change,
        message="Now focus on art",
        category_vocabulary=r.metadata_categories,
        tag_vocabulary=r.metadata_tags,
    )
    changed = manager.apply_intent(
        initial.conversation_id, change, message="Now focus on art"
    ).state
    clear = intent("poi_recommendation", actions={"semantic_interests": "clear"})
    cleared = manager.apply_intent(
        initial.conversation_id, clear, message="Clear my interests."
    ).state
    assert with_history.preferences.interests == ["history"]
    assert changed.preferences.interests == ["art"]
    assert changed.active_request.interests == ["art"]
    assert cleared.preferences.interests == []
    assert cleared.active_request.interests == []


def test_weather_replanning_commits_with_its_valid_pending_modification():
    r=repo(); manager=ConversationStateManager()
    create=plan_intent(filters={"setting":"outdoor"},interests=["history"])
    weather_change=intent(
        "weather_based_replanning",
        {"itinerary_modification": {
            "action": "change_constraints",
            "preserve_other_stops": True,
        }},
        {"itinerary_modification": "set"},
    )
    provider=FixtureWeatherProvider(
        ROOT/"corfu_synthetic_rain.json",
        clock=lambda:datetime(2026,10,7,12,tzinfo=TZ),
    )
    orch,_,_=make_orch([create,weather_change],r,manager,provider)
    first=orch.process({"message":"Plan outdoor historical attractions Wednesday October 7 13:00-17:00"})

    second=orch.process({
        "conversation_id":first.conversation_id,
        "message":"Check weather and adapt my existing itinerary",
    })

    assert second.status in {"feasible", "uncertain"}
    assert second.weather_replanning is not None
    assert second.conversation_state.pending_modification is None
    assert second.conversation_state.last_modification_outcome == "applied"


def test_genuine_concurrent_update_rejects_weather_commit_and_preserves_itinerary():
    r=repo(); manager=ConversationStateManager()

    class ConcurrentWeather(FixtureWeatherProvider):
        conversation_id = None

        def get_forecast(self, start, end):
            assert self.conversation_id is not None
            manager.apply_intent(
                self.conversation_id,
                intent(
                    "poi_recommendation",
                    {"semantic_preferences":{"interests":["art"]}},
                    {"semantic_interests":"set"},
                ),
                message="Concurrent preference update",
            )
            return super().get_forecast(start, end)

    provider=ConcurrentWeather(
        ROOT/"corfu_synthetic_rain.json",
        clock=lambda:datetime(2026,10,7,12,tzinfo=TZ),
    )
    orch,_,_=make_orch([
        plan_intent(filters={"setting":"outdoor"},interests=["history"]),
        intent("weather_based_replanning"),
    ],r,manager,provider)
    first=orch.process({"message":"Plan outdoor historical attractions Wednesday October 7 13:00-17:00"})
    provider.conversation_id=first.conversation_id
    original=first.conversation_state.itinerary.model_copy(deep=True)

    second=orch.process({
        "conversation_id":first.conversation_id,
        "message":"Check weather and adapt my existing itinerary",
    })

    assert second.status == "conflict"
    assert second.conversation_state.preferences.interests == ["art"]
    assert second.conversation_state.itinerary == original
    assert second.weather_replanning is not None
    assert second.weather_replanning.revised_poi_ids == second.weather_replanning.original_poi_ids


def test_live_weather_failure_does_not_fabricate_fixture_data():
    class Broken:
        def get_current(self): raise WeatherProviderError("network unavailable")
        def get_forecast(self,*args): raise WeatherProviderError("network unavailable")
        def next_24_hours(self): raise WeatherProviderError("network unavailable")
    orch,_,_=make_orch([intent("live_weather_query",{"weather_request":{"aspects":["precipitation"]}},{"weather_request":"set"})],weather=Broken())
    out=orch.process({"message":"What is the weather in Corfu now?"})
    assert out.status=="weather_unavailable"
    assert out.weather_forecast is None


def test_demo_c_mandatory_five_stops_tuesday_is_infeasible():
    r=repo(); planner=DeterministicItineraryPlanner(r,travel_time_estimator=HaversineWalkingTimeEstimator())
    from corfu_assistant.planner import PlanningRequest
    result=planner.plan(PlanningRequest(requested_date="2026-10-06",start_time="11:00",end_time="12:30",
        required_poi_ids=["CF-001","CF-002","CF-021","CF-022","CF-017"]))
    assert result.status=="infeasible"
    assert not result.validation.is_valid
