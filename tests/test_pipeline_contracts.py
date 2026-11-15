"""Cross-boundary regressions for semantic intent and failed-turn atomicity."""
import pytest
from tests.test_orchestrator import repo, make_orch, intent, plan_intent
from corfu_assistant.intent_policy import normalize_thematic_filters
from corfu_assistant.orchestrator import Orchestrator, InvalidExtractedConstraintError


def test_structured_preferences_are_independent_of_surface_wording():
    candidates = repo().get_all()
    expected = Orchestrator._focused_candidate_ids('focused on art and museums', ['art', 'museums'], candidates)
    assert expected
    for message in ('Make a cultural afternoon for me', 'I would enjoy galleries', 'Please build my walk'):
        assert Orchestrator._focused_candidate_ids(message, ['art', 'museums'], candidates) == expected


def test_negative_message_terms_do_not_override_positive_structured_interests():
    candidates = repo().get_all()
    selected = Orchestrator._focused_candidate_ids('Interested in architecture; no museums', ['architecture'], candidates)
    assert selected
    assert all('architecture' in p.tags for p in candidates if p.id in selected)


def test_normalization_does_not_mutate_extracted_intent():
    r = repo()
    raw = intent('itinerary_creation', {'hard_filters': {'tags': ['history']},
        'semantic_preferences': {'interests': ['architecture']}},
        {'filter_tags': 'set', 'semantic_interests': 'set'})
    before = raw.model_dump(mode='json')
    normalize_thematic_filters(raw, message='historical architecture',
        category_vocabulary=r.metadata_categories, tag_vocabulary=r.metadata_tags)
    assert raw.model_dump(mode='json') == before


def test_invalid_filter_failure_preserves_existing_conversation():
    orch, _, manager = make_orch([plan_intent(), plan_intent(filters={'category': 'nonexistent'})])
    first = orch.process({'message': 'A short afternoon walk'})
    before = manager.get(first.conversation_id)
    with pytest.raises(InvalidExtractedConstraintError):
        orch.process({'conversation_id': first.conversation_id, 'message': 'Change my interests'})
    assert manager.get(first.conversation_id) == before


def test_rollback_does_not_overwrite_a_concurrent_update():
    from corfu_assistant.state_manager import ConversationStateManager
    manager = ConversationStateManager()
    prior = manager.create_conversation()
    failed = manager.apply_intent(prior.conversation_id, plan_intent(), message='first').state
    newer = manager.apply_intent(prior.conversation_id, plan_intent(interests=['architecture']), message='second').state
    assert not manager.rollback_failed_turn(failed, prior)
    assert manager.get(prior.conversation_id) == newer


def test_duplicate_concurrent_requests_are_executed_once():
    from concurrent.futures import ThreadPoolExecutor
    from uuid import uuid4
    orch, _, _ = make_orch([plan_intent(interests=['architecture'])])
    request = {'message': 'Architecture afternoon', 'request_id': uuid4()}
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: orch.process(request), range(2)))
    assert results[0] == results[1]
    assert not orch._intent_extractor.items


def test_child_replacement_preserves_other_ids_and_deadline():
    replacement = intent('itinerary_modification', {
        'itinerary_modification': {'action': 'replace_stop', 'stop_index': 1, 'preserve_other_stops': True},
        'family_requirements': {'required': True, 'child_ages': [10]},
    }, {'itinerary_modification': 'set', 'family_required': 'set', 'child_ages': 'set'})
    orch, r, _ = make_orch([plan_intent(day='2026-10-06', start='11:00', end='15:00', filters={'category': 'history'}), replacement])
    first = orch.process({'message': 'Historical walk'})
    assert first.planning_result and len(first.planning_result.selected_poi_ids) >= 2
    second = orch.process({'conversation_id': first.conversation_id, 'message': 'Replace stop two for my child'})
    assert second.modification_applied
    before = first.conversation_state.itinerary
    after = second.conversation_state.itinerary
    assert [s.poi_id for i,s in enumerate(before.stops) if i != 1] == [s.poi_id for i,s in enumerate(after.stops) if i != 1]
    assert r.get_by_id(after.stops[1].poi_id).family_suitability_fixture is True
    assert after.stops[-1].ends_at <= before.ends_at


@pytest.mark.parametrize('message', ['Use the tag history', 'Use tags history', 'Use places tagged history'])
def test_literal_tag_forms_remain_mandatory(message):
    r = repo()
    raw = intent('itinerary_creation', {'hard_filters': {'tags': ['history']}}, {'filter_tags': 'set'})
    normalized, demoted = normalize_thematic_filters(raw, message=message,
        category_vocabulary=r.metadata_categories, tag_vocabulary=r.metadata_tags)
    assert normalized == raw
    assert not demoted


def test_early_completion_is_explained_in_chat_response():
    orch, _, _ = make_orch([plan_intent(filters={'tags': ['history'], 'setting': 'outdoor'})])
    out = orch.process({'message': 'Use only places tagged history for my outdoor afternoon'})
    assert out.status == 'feasible'
    assert 'minutes unallocated' in out.response
    assert out.conversation_state.itinerary.stops[-1].ends_at < out.conversation_state.itinerary.ends_at
