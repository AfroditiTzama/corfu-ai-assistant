"""Semantic contract tested through policy, state merge and real metadata filtering."""
import pytest
from tests.test_orchestrator import repo, intent, make_orch
from corfu_assistant.intent_policy import normalize_thematic_filters
from corfu_assistant.state_manager import ConversationStateManager


def grounded(tags, quote, kind='required', evidence=True):
    value = intent('poi_recommendation', {'hard_filters': {'tags': tags}}, {'filter_tags': 'set'})
    payload = value.model_dump(mode='python')
    payload['constraint_evidence'] = ([{'field':'filter_tags','value':tag,'kind':kind,'quote':quote} for tag in tags] if evidence else [])
    return type(value).model_validate(payload)


def soft_with_spurious_filter(theme, *, tags=None, category=None, evidence=None):
    filters = {}
    actions = {'semantic_interests': 'set'}
    if tags is not None:
        filters['tags'] = tags
        actions['filter_tags'] = 'set'
    if category is not None:
        filters['category'] = category
        actions['filter_category'] = 'set'
    value = intent('poi_recommendation', {
        'hard_filters': filters, 'semantic_preferences': {'interests': [theme]},
    }, actions)
    return type(value).model_validate({**value.model_dump(), 'constraint_evidence': evidence or []})


def candidates(value, message, manager=None, conversation_id=None):
    r=repo(); manager=manager or ConversationStateManager()
    state=manager.get(conversation_id) if conversation_id else manager.create_conversation()
    normalized,_=normalize_thematic_filters(value,message=message,category_vocabulary=r.metadata_categories,tag_vocabulary=r.metadata_tags)
    turn=manager.apply_intent(state.conversation_id,normalized,message=message)
    filters=turn.state.hard_filters.model_dump(exclude_none=True)
    return normalized, r.filter_metadata(filters), manager, turn.state


@pytest.mark.parametrize('theme', ['museum','art','architecture','garden','nightlife'])
def test_topic_alone_becomes_a_soft_interest_without_authorizing_a_hard_tag(theme):
    message=f'I enjoy {theme}'
    normalized,pois,_,state=candidates(grounded([theme],message),message)
    assert normalized.intent=='poi_recommendation'
    assert normalized.hard_filters.tags is None
    assert state.preferences.interests == [theme]
    assert len(pois)==50


@pytest.mark.parametrize('tag', ['museum','art','architecture','garden'])
def test_explicit_only_restriction_filters_real_records(tag):
    message=f'Only {tag} places please'
    normalized,pois,_,_=candidates(grounded([tag],message),message)
    assert normalized.intent=='poi_recommendation'
    assert pois and all(tag in p.tags for p in pois)


def test_literal_tags_and_combined_requirements():
    message='Use places tagged museum and art'
    _,pois,_,_=candidates(grounded(['museum','art'],message,'literal_tag'),message)
    assert pois and all({'museum','art'} <= set(p.tags) for p in pois)


def test_fabricated_or_wrong_subject_quote_requests_clarification():
    for quote in ('Only museums', 'Only outdoor places'):
        value=grounded(['museum'],quote)
        normalized,pois,_,_=candidates(value,'Only outdoor places')
        assert normalized.intent=='clarification'
        assert len(pois)==50


def test_inherited_constraint_survives_soft_followup_and_ambiguous_update():
    first,pois,manager,state=candidates(grounded(['museum'],'Only museums'),'Only museums')
    followup=intent('poi_recommendation', {'semantic_preferences':{'interests':['art']}},{'semantic_interests':'set'}).model_copy(update={'constraint_evidence':[]})
    _,remaining,_,state=candidates(followup,'I enjoy art too',manager,state.conversation_id)
    assert {p.id for p in remaining}=={p.id for p in pois}
    ambiguous=grounded(['art'],'I enjoy art')
    normalized,remaining,_,_=candidates(ambiguous,'I enjoy art',manager,state.conversation_id)
    assert normalized.intent=='poi_recommendation'
    assert normalized.hard_filters.tags is None
    assert {p.id for p in remaining}=={p.id for p in pois}


@pytest.mark.parametrize('theme,tag,message', [
    ('art', 'art', 'I would enjoy an art afternoon'),
    ('architecture', 'architecture', 'Please arrange an architecture walk'),
    ('garden', 'garden', 'I am interested in gardens'),
    ('music', 'music', 'A music themed outing sounds good'),
])
def test_unsupported_positive_tag_preserves_soft_interest_and_full_candidate_pool(theme, tag, message):
    normalized, pois, _, state = candidates(soft_with_spurious_filter(theme, tags=[tag]), message)
    assert normalized.intent == 'poi_recommendation'
    assert normalized.field_actions.filter_tags == 'preserve'
    assert normalized.semantic_preferences.interests == [theme]
    assert state.preferences.interests == [theme]
    assert len(pois) == 50


def test_unsupported_category_and_tag_are_dropped_without_weakening_supported_setting():
    message = 'Arrange an outdoor walk inspired by architecture and historic squares'
    value = soft_with_spurious_filter('architecture', tags=['history'], category='history', evidence=[
        {'field': 'filter_setting', 'value': 'outdoor', 'kind': 'required', 'quote': 'outdoor walk'},
    ])
    data = value.model_dump()
    data['hard_filters']['setting'] = 'outdoor'
    data['field_actions']['filter_setting'] = 'set'
    value = type(value).model_validate(data)
    normalized, pois, _, state = candidates(value, message)
    assert normalized.intent == 'poi_recommendation'
    assert normalized.hard_filters.setting == 'outdoor'
    assert normalized.hard_filters.tags is None and normalized.hard_filters.category is None
    assert state.preferences.interests == ['architecture']
    assert len(pois) == 22 and all(p.setting == 'outdoor' for p in pois)


def test_explicit_outdoor_requirement_does_not_make_nearby_art_interest_hard():
    message = 'Only outdoor, interested in art'
    value = soft_with_spurious_filter('art', tags=['art'], evidence=[
        {'field': 'filter_setting', 'value': 'outdoor', 'kind': 'required', 'quote': 'Only outdoor'},
    ])
    data = value.model_dump()
    data['hard_filters']['setting'] = 'outdoor'
    data['field_actions']['filter_setting'] = 'set'
    value = type(value).model_validate(data)
    normalized, pois, _, state = candidates(value, message)
    assert normalized.intent == 'poi_recommendation'
    assert normalized.hard_filters.setting == 'outdoor'
    assert normalized.hard_filters.tags is None
    assert state.preferences.interests == ['art']
    assert len(pois) == 22


def test_unsupported_model_filter_does_not_erase_inherited_requirement():
    _, original, manager, state = candidates(grounded(['museum'], 'Only museums'), 'Only museums')
    value = soft_with_spurious_filter('art', tags=['art'])
    normalized, pois, _, updated = candidates(value, 'I would also enjoy art', manager, state.conversation_id)
    assert normalized.intent == 'poi_recommendation'
    assert updated.preferences.interests == ['art']
    assert updated.hard_filters.tags == ['museum']
    assert {p.id for p in pois} == {p.id for p in original}


def test_explicit_museum_requirement_and_soft_art_interest_are_resolved_per_value():
    message = 'Only museums, and I would enjoy art too'
    value = soft_with_spurious_filter('art', tags=['museum', 'art'], evidence=[
        {'field': 'filter_tags', 'value': 'museum', 'kind': 'required', 'quote': 'Only museums'},
    ])
    normalized, pois, _, state = candidates(value, message)
    assert normalized.intent == 'poi_recommendation'
    assert normalized.hard_filters.tags == ['museum']
    assert state.preferences.interests == ['art']
    assert pois and all('museum' in poi.tags for poi in pois)


def test_broad_model_quote_cannot_promote_separate_art_preference():
    message = 'Only museums, and I would enjoy art too'
    value = soft_with_spurious_filter('art', tags=['museum', 'art'], evidence=[
        {'field': 'filter_tags', 'value': 'museum', 'kind': 'required', 'quote': message},
        {'field': 'filter_tags', 'value': 'art', 'kind': 'required', 'quote': message},
    ])
    normalized, pois, _, _ = candidates(value, message)
    assert normalized.intent == 'poi_recommendation'
    assert normalized.hard_filters.tags == ['museum']
    assert len(pois) == 8


def test_unrelated_category_requirement_and_tag_exclusion_ignore_bad_annotations():
    message = 'Every place must be in the nature category; leave out nightlife-tagged places'
    value = intent('poi_recommendation', {
        'hard_filters': {'category': 'nature', 'excluded_tags': ['nightlife']},
    }, {'filter_category': 'set', 'filter_excluded_tags': 'set'})
    value = type(value).model_validate({**value.model_dump(), 'constraint_evidence': [
        {'field': 'filter_category', 'value': 'nature', 'kind': 'ambiguous', 'quote': 'nature category'},
    ]})
    normalized, pois, _, _ = candidates(value, message)
    assert normalized.intent == 'poi_recommendation'
    assert pois and all(poi.category == 'nature' and 'nightlife' not in poi.tags for poi in pois)


def test_literal_dataset_tag_wins_over_same_named_category():
    value = intent('poi_recommendation', {'hard_filters': {'category': 'history'}},
                   {'filter_category': 'set'})
    value = type(value).model_validate({**value.model_dump(), 'constraint_evidence': []})
    normalized, pois, _, _ = candidates(value, 'Use only places tagged history')
    assert normalized.intent == 'poi_recommendation'
    assert normalized.hard_filters.category is None
    assert normalized.hard_filters.tags == ['history']
    assert pois and all('history' in poi.tags for poi in pois)


def test_conflicting_explicit_museum_require_and_exclude_clarifies():
    value = intent('poi_recommendation', {
        'hard_filters': {'tags': ['museum'], 'excluded_tags': ['museum']},
    }, {'filter_tags': 'set', 'filter_excluded_tags': 'set'})
    value = type(value).model_validate({**value.model_dump(), 'constraint_evidence': []})
    normalized, _, _, _ = candidates(value, 'Only museums, but no museums')
    assert normalized.intent == 'clarification'


def test_time_limit_word_only_does_not_turn_a_soft_interest_into_a_restriction():
    value = soft_with_spurious_filter('art', tags=['art'])
    normalized, pois, _, _ = candidates(value, 'I only have two hours and would enjoy art')
    assert normalized.intent == 'poi_recommendation'
    assert len(pois) == 50


@pytest.mark.parametrize('message', [
    'Only museums for this trip',
    'I require museums for every stop',
    'Use places tagged museum',
])
def test_explicit_user_restriction_is_enforced_even_if_model_omits_evidence(message):
    value = soft_with_spurious_filter('museum', tags=['museum'])
    normalized, pois, _, _ = candidates(value, message)
    assert normalized.intent == 'poi_recommendation'
    assert normalized.hard_filters.tags == ['museum']
    assert len(pois) == 8


def test_explicit_exclusion_is_enforced_even_without_model_evidence():
    value = intent('poi_recommendation', {
        'hard_filters': {'excluded_tags': ['museum']},
        'semantic_preferences': {'interests': ['art']},
    }, {'filter_excluded_tags': 'set', 'semantic_interests': 'set'})
    value = type(value).model_validate({**value.model_dump(), 'constraint_evidence': []})
    normalized, pois, _, _ = candidates(value, 'I enjoy art without museums')
    assert normalized.intent == 'poi_recommendation'
    assert all('museum' not in poi.tags for poi in pois)


def test_model_invented_exclusion_still_requires_clarification():
    value = intent('poi_recommendation', {
        'hard_filters': {'excluded_tags': ['museum']},
        'semantic_preferences': {'interests': ['art']},
    }, {'filter_excluded_tags': 'set', 'semantic_interests': 'set'})
    value = type(value).model_validate({**value.model_dump(), 'constraint_evidence': []})
    normalized, _, _, _ = candidates(value, 'I enjoy art')
    assert normalized.intent == 'clarification'


def test_live_shape_museum_requirement_and_nightlife_exclusion_reaches_planner():
    source = intent('itinerary_creation', {
        'requested_date': '2026-10-08',
        'time_window': {'start_time': '13:00', 'end_time': '17:00'},
        'hard_filters': {'tags': ['museum'], 'excluded_tags': ['nightlife']},
    }, {'requested_date': 'set', 'start_time': 'set', 'end_time': 'set',
        'filter_tags': 'set', 'filter_excluded_tags': 'set'})
    data = source.model_dump()
    data['constraint_evidence'] = [
        {'field': 'filter_tags', 'value': 'museum', 'kind': 'literal_tag',
         'quote': 'Every stop must be a museum'},
        {'field': 'filter_excluded_tags', 'value': 'nightlife', 'kind': 'excluded',
         'quote': 'exclude places tagged nightlife'},
    ]
    orch, repository, _ = make_orch([type(source).model_validate(data)])
    result = orch.process({'message': 'Plan a walking visit in Corfu on Thursday, October 8, 2026, '
                                     'from 13:00 to 17:00. Every stop must be a museum, '
                                     'and exclude places tagged nightlife.'})
    assert result.status in {'feasible', 'uncertain', 'infeasible'}
    assert result.status != 'clarification_required'
    assert result.intent.hard_filters.tags == ['museum']
    assert result.intent.hard_filters.excluded_tags == ['nightlife']
    if result.planning_result:
        assert all('museum' in repository.get_by_id(pid).tags and
                   'nightlife' not in repository.get_by_id(pid).tags
                   for pid in result.planning_result.selected_poi_ids)


def test_entity_type_mapped_from_wrong_category_to_dataset_tag():
    value = intent('poi_recommendation', {'hard_filters': {'category': 'museum'}},
                   {'filter_category': 'set'})
    value = type(value).model_validate({**value.model_dump(), 'constraint_evidence': []})
    normalized, pois, _, _ = candidates(value, 'Only museums')
    assert normalized.intent == 'poi_recommendation'
    assert normalized.hard_filters.category is None
    assert normalized.hard_filters.tags == ['museum']
    assert len(pois) == 8


def test_excluded_entity_type_mapped_from_wrong_category_to_dataset_tag():
    value = intent('poi_recommendation', {'hard_filters': {'excluded_categories': ['museum']}},
                   {'filter_excluded_categories': 'set'})
    value = type(value).model_validate({**value.model_dump(), 'constraint_evidence': []})
    normalized, pois, _, _ = candidates(value, 'Without museums')
    assert normalized.intent == 'poi_recommendation'
    assert normalized.hard_filters.excluded_categories is None
    assert normalized.hard_filters.excluded_tags == ['museum']
    assert len(pois) == 42


def test_unsupported_category_is_not_reinterpreted_as_a_tag():
    value = intent('poi_recommendation', {'hard_filters': {'category': 'imaginary'}},
                   {'filter_category': 'set'})
    value = type(value).model_validate({**value.model_dump(), 'constraint_evidence': []})
    repository = repo()
    normalized, _ = normalize_thematic_filters(
        value, message='Only imaginary places',
        category_vocabulary=repository.metadata_categories,
        tag_vocabulary=repository.metadata_tags,
    )
    assert normalized.intent == 'poi_recommendation'
    assert normalized.hard_filters.category == 'imaginary'


def test_clear_but_empty_filter_intersection_reports_infeasibility_not_ambiguity():
    value = intent('itinerary_creation', {
        'requested_date': '2026-10-08',
        'time_window': {'start_time': '13:00', 'end_time': '17:00'},
        'hard_filters': {'category': 'nightlife', 'tags': ['museum']},
    }, {'requested_date': 'set', 'start_time': 'set', 'end_time': 'set',
        'filter_category': 'set', 'filter_tags': 'set'})
    value = type(value).model_validate({**value.model_dump(), 'constraint_evidence': []})
    orch, repository, _ = make_orch([value])
    assert not repository.filter_metadata({'category': 'nightlife', 'tags': ['museum']})
    result = orch.process({'message': 'Only nightlife category places, and every stop must be a museum. '
                                      'Plan Thursday October 8, 2026 from 13:00 to 17:00.'})
    assert result.status == 'infeasible'
    assert result.intent.hard_filters.category == 'nightlife'
    assert result.intent.hard_filters.tags == ['museum']


def test_exact_failure_shape_resolves_plural_list_and_conditional_exclusion(caplog):
    message = (
        'Plan a walking itinerary in Corfu Town for Tuesday, October 6, 2026, '
        'from 11:00 to 15:00. I only want indoor museums. Do not include churches, '
        'parks, outdoor squares, or art galleries that are not museums.'
    )
    value = intent('itinerary_creation', {
        'requested_date': '2026-10-06',
        'time_window': {'start_time': '11:00', 'end_time': '15:00'},
        'travel_mode': 'walking',
        'hard_filters': {
            'tags': ['museum'], 'setting': 'indoor',
            'excluded_tags': ['church', 'park', 'square', 'art', 'gallery'],
        },
    }, {
        'requested_date': 'set', 'start_time': 'set', 'end_time': 'set',
        'travel_mode': 'set', 'filter_tags': 'set', 'filter_setting': 'set',
        'filter_excluded_tags': 'set',
    })
    value = type(value).model_validate({**value.model_dump(), 'constraint_evidence': [
        {'field': 'filter_tags', 'value': 'museum', 'kind': 'literal_tag',
         'quote': 'I only want indoor museums'},
        {'field': 'filter_setting', 'value': 'indoor', 'kind': 'required',
         'quote': 'indoor museums'},
        *[{'field': 'filter_excluded_tags', 'value': tag, 'kind': 'excluded',
           'quote': 'Do not include churches, parks, outdoor squares, or art galleries that are not museums'}
          for tag in ['church', 'park', 'square', 'art', 'gallery']],
    ]})
    repository = repo()
    with caplog.at_level('INFO', logger='corfu_assistant.intent_policy'):
        normalized, _ = normalize_thematic_filters(
            value, message=message, category_vocabulary=repository.metadata_categories,
            tag_vocabulary=repository.metadata_tags,
        )
    filters = normalized.hard_filters.model_dump(exclude_none=True)
    pois = repository.filter_metadata(filters)
    assert normalized.intent == 'itinerary_creation'
    assert filters == {
        'tags': ['museum'], 'setting': 'indoor',
        'excluded_tags': ['church', 'park', 'square'],
    }
    assert len(pois) == 7
    assert 'conditional_exclusion_drop_count' in caplog.records[0].__dict__


def test_exact_failure_shape_reaches_real_planner_without_clarification():
    message = (
        'Plan a walking itinerary in Corfu Town for Tuesday, October 6, 2026, '
        'from 11:00 to 15:00. I only want indoor museums. Do not include churches, '
        'parks, outdoor squares, or art galleries that are not museums.'
    )
    value = intent('itinerary_creation', {
        'requested_date': '2026-10-06',
        'time_window': {'start_time': '11:00', 'end_time': '15:00'},
        'travel_mode': 'walking',
        'hard_filters': {
            'tags': ['museum'], 'setting': 'indoor',
            'excluded_tags': ['church', 'park', 'square', 'gallery'],
        },
    }, {
        'requested_date': 'set', 'start_time': 'set', 'end_time': 'set',
        'travel_mode': 'set', 'filter_tags': 'set', 'filter_setting': 'set',
        'filter_excluded_tags': 'set',
    })
    value = type(value).model_validate({**value.model_dump(), 'constraint_evidence': []})
    orchestrator, repository, _ = make_orch([value])
    result = orchestrator.process({'message': message})
    assert result.status in {'feasible', 'uncertain'}
    assert result.status != 'clarification_required'
    assert result.planning_result and result.planning_result.selected_poi_ids
    for poi_id in result.planning_result.selected_poi_ids:
        poi = repository.get_by_id(poi_id)
        assert poi.setting == 'indoor' and 'museum' in poi.tags
        assert not {'church', 'park', 'square'} & set(poi.tags)


def test_soft_interest_with_spurious_tag_reaches_real_orchestration():
    source = soft_with_spurious_filter('art galleries', tags=['art', 'gallery'])
    payload = source.model_dump()
    payload['intent'] = 'itinerary_creation'
    payload['requested_date'] = '2026-10-08'
    payload['time_window'] = {'start_time': '13:00', 'end_time': '17:00'}
    payload['field_actions'].update(requested_date='set', start_time='set', end_time='set')
    value = type(source).model_validate(payload)
    orch, _, _ = make_orch([value])
    result = orch.process({'message': 'Put together a gallery and art afternoon in Corfu on October 8, 2026, 13:00–17:00.'})
    assert result.status in {'feasible', 'uncertain'}
    assert result.planning_result
    assert all(issue.code == 'unknown_opening_hours' for issue in result.planning_result.validation.issues)
    assert result.planning_result.selected_poi_ids
    assert result.intent.field_actions.filter_tags == 'preserve'
    assert result.conversation_state.preferences.interests == ['art galleries']


def test_exclusion_is_preserved_and_not_a_positive_topic():
    value=intent('poi_recommendation', {'hard_filters':{'excluded_tags':['museum']}},{'filter_excluded_tags':'set'})
    value=type(value).model_validate({**value.model_dump(), 'constraint_evidence':[{'field':'filter_excluded_tags','value':'museum','kind':'excluded','quote':'No museums'}]})
    normalized,pois,_,_=candidates(value,'No museums')
    assert normalized.intent=='poi_recommendation'
    assert pois and all('museum' not in p.tags for p in pois)


def test_missing_evidence_never_reaches_filtering_as_a_hard_update():
    orch,_,_=make_orch([grounded(['museum'],'museums',evidence=False)])
    result=orch.process({'message':'A cultural afternoon'})
    assert result.status=='completed'
    assert not result.conversation_state.hard_filters.tags


def test_negated_restriction_is_not_positive_evidence():
    value=grounded(['museum'],'not only museums')
    normalized,pois,_,_=candidates(value,'not only museums')
    assert normalized.intent=='poi_recommendation' and len(pois)==50
    assert normalized.hard_filters.tags is None


def test_model_adapter_never_defaults_to_trusted_legacy_intent():
    from corfu_assistant.intent_extractor import GeminiIntentExtractor
    raw=grounded(['museum'],'museum',evidence=False).model_dump(mode='json')
    raw.pop('constraint_evidence')
    class Runnable:
        def invoke(self, _messages): return raw
    extractor=GeminiIntentExtractor(api_key=None,structured_runnable=Runnable())
    result=extractor.extract('A cultural afternoon')
    assert result.constraint_evidence==[]
    normalized,pois,_,_=candidates(result,'A cultural afternoon')
    assert normalized.intent=='poi_recommendation' and len(pois)==50
    assert normalized.hard_filters.tags is None


def test_model_adapter_soft_interest_survives_unsupported_hard_filter():
    from corfu_assistant.intent_extractor import GeminiIntentExtractor
    value = soft_with_spurious_filter('architecture', tags=['architecture'])
    raw = value.model_dump(mode='json')
    raw.pop('constraint_evidence')
    class Runnable:
        def invoke(self, _messages): return raw
    extractor = GeminiIntentExtractor(api_key=None, structured_runnable=Runnable())
    parsed = extractor.extract('I would enjoy an architecture walk')
    normalized, pois, _, state = candidates(parsed, 'I would enjoy an architecture walk')
    assert normalized.intent == 'poi_recommendation'
    assert normalized.hard_filters.tags is None
    assert state.preferences.interests == ['architecture']
    assert len(pois) == 50


def test_hypothetical_weather_activity_request_routes_to_indoor_first_recommendations():
    value = intent('live_weather_query', {
        'semantic_preferences': {'interests': ['rainy afternoon']},
        'hard_filters': {'setting': 'indoor'},
        'weather_request': {
            'aspects': ['precipitation', 'general_conditions'],
            'activity_recommendation': True,
        },
    }, {
        'semantic_interests': 'set',
        'filter_setting': 'set',
        'weather_request': 'set',
    })
    value = type(value).model_validate({
        **value.model_dump(),
        'constraint_evidence': [{
            'field': 'filter_setting', 'value': 'indoor', 'kind': 'required',
            'quote': 'rainy afternoon',
        }],
    })
    orch, repository, _ = make_orch([value])

    result = orch.process({'message': 'Which attractions suit a wet-weather afternoon?'})

    assert result.status == 'completed'
    assert result.intent.intent == 'poi_recommendation'
    assert result.intent.hard_filters.setting is None
    assert result.conversation_state.hard_filters.setting is None
    assert result.conversation_state.preferences.interests == ['rainy afternoon']
    assert result.recommended_poi_ids
    assert all(repository.get_by_id(poi_id).setting == 'indoor'
               for poi_id in result.recommended_poi_ids)
    hospitality_tags = {'food', 'restaurant', 'cafe', 'coffee', 'bar'}
    assert all(not (set(repository.get_by_id(poi_id).tags) & hospitality_tags)
               for poi_id in result.recommended_poi_ids)
    assert result.weather_forecast is None


def test_explicit_clear_can_remove_an_inherited_restriction():
    _,_,manager,state=candidates(grounded(['museum'],'Only museums'),'Only museums')
    value=intent('poi_recommendation', actions={'filter_tags':'clear'})
    value=type(value).model_validate({**value.model_dump(),'constraint_evidence':[{'field':'filter_tags','value':'*','kind':'clear','quote':'Remove the museum restriction'}]})
    normalized,pois,_,_=candidates(value,'Remove the museum restriction',manager,state.conversation_id)
    assert normalized.intent=='poi_recommendation' and len(pois)==50


@pytest.mark.parametrize('quote,accepted', [('Avoid Old Fortress',True),('Avoid shopping',False)])
def test_named_exclusion_requires_matching_source_destination(quote,accepted):
    r=repo()
    value=intent('poi_recommendation', {'hard_filters':{'excluded_poi_ids':['CF-001']}},{'filter_excluded_poi_ids':'set'})
    value=type(value).model_validate({**value.model_dump(), 'constraint_evidence':[{'field':'filter_excluded_poi_ids','value':'CF-001','kind':'excluded','quote':quote}]})
    normalized,_=normalize_thematic_filters(value,message=quote,category_vocabulary=r.metadata_categories,
        tag_vocabulary=r.metadata_tags,poi_names={p.id:p.name for p in r.get_all()})
    assert (normalized.intent=='poi_recommendation') is accepted


@pytest.mark.parametrize('theme', ['music','nature','architecture','art','history'])
def test_positive_interests_do_not_reduce_metadata_eligibility(theme):
    value=intent('poi_recommendation', {'semantic_preferences':{'interests':[theme]}},{'semantic_interests':'set'})
    value=value.model_copy(update={'constraint_evidence':[]})
    normalized,pois,_,state=candidates(value,f'I enjoy {theme}')
    assert normalized.intent=='poi_recommendation'
    assert len(pois)==50 and state.preferences.interests==[theme]
