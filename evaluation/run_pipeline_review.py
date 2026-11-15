"""Requirements review: real retrieval/planning, controlled intents; optional 2 live chats."""
import argparse
import json
import time
from pathlib import Path
from fastapi.testclient import TestClient
from corfu_assistant.config import Settings
from corfu_assistant.data import load_dataset
from corfu_assistant.intent_extractor import GeminiIntentExtractor
from corfu_assistant.main import create_app
from corfu_assistant.repository import POIRepository
from corfu_assistant.services import ServiceOverrides
from evaluation.run_e2e_evaluation import make_intent

ROOT = Path(__file__).resolve().parents[1]

class Extractor:
    def __init__(self, value): self.value = value
    def extract(self, *args): return self.value


def planning(interests, filters=None, end='17:00'):
    values = {'requested_date':'2026-10-08','time_window':{'start_time':'13:00','end_time':end},
              'semantic_preferences':{'interests':interests},'travel_mode':'walking'}
    actions = dict.fromkeys(['requested_date','start_time','end_time','semantic_interests','travel_mode'],'set')
    if filters:
        values['hard_filters']=filters
        actions.update({'filter_'+key:'set' for key in filters})
    return make_intent('itinerary_creation',values,actions)


def summarize(response, elapsed):
    if response.status_code != 200:
        return {'http':response.status_code,'latency_seconds':round(elapsed,2)}
    d=response.json(); p=d.get('planning_result') or {}; it=p.get('itinerary') or {}
    return {'http':200,'status':d['status'],'latency_seconds':round(elapsed,2),
        'intent':d['intent'],'response':d['response'],
        'selected_ids':p.get('selected_poi_ids',d.get('relevant_poi_ids',[])),
        'stops':it.get('stops',[]),'validation':p.get('validation'),
        'explanations':p.get('explanations'), 'citation_status':d.get('citation_status')}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--live',action='store_true'); args=parser.parse_args()
    settings=Settings(); repository=POIRepository(load_dataset(settings.dataset_path),
        embedding_model_name=settings.embedding_model_name,embedding_cache_dir=settings.embedding_cache_dir)
    repository.initialize_semantic_index()
    cases=[
        ('art_paraphrase','Put together a gallery and art afternoon this Thursday, October 8, 2026, 13:00–17:00.',planning(['art','galleries']),'Relevant art, no mandatory category, valid or explicitly conditional'),
        ('same_semantics','I would enjoy a cultural afternoon in Corfu, October 8, 2026, 13:00–17:00.',planning(['art','galleries']),'Same structured focus independent of surface phrasing'),
        ('exclusive_museums','Only museums indoors on October 8 from 13:00 to 17:00.',planning(['art'],{'tags':['museum'],'setting':'indoor'}),'All selected stops indoor museum-tagged'),
        ('architecture_exclusion','I enjoy architecture; please avoid museums on October 8, 13:00–17:00.',planning(['architecture'],{'excluded_tags':['museum']}),'No museum-tagged stop; architectural relevance'),
        ('short_mandatory','Visit Achilleion Palace on foot October 8 from 13:00 to 13:20.',planning([],end='13:20'),'Reject required visit whose duration alone exceeds window'),
        ('missing_date','Organize a walk for me.',make_intent('itinerary_creation'),'Ask for date/time, no invented schedule'),
        ('qa','What is the Old Fortress?',make_intent('tourism_qa',{'question':'What is the Old Fortress?'},{'question':'set'}),'Dataset-grounded answer with POI references'),
        ('unsupported','Please book admission and guarantee entry at Casa Parlante.',make_intent('tourism_qa',{'unsupported_actions':['booking']}),'Explicit unsupported service, no fictional booking'),
    ]
    outputs=[]
    controlled=settings.model_copy(update={'google_api_key':None,'weather_fixture_enabled':False})
    for identifier,prompt,value,expected in cases:
        app=create_app(controlled,service_overrides=ServiceOverrides(poi_repository=repository,intent_extractor=Extractor(value)))
        with TestClient(app) as client:
            started=time.perf_counter(); response=client.post('/chat',json={'message':prompt})
            outputs.append({'id':identifier,'prompt':prompt,'expected':expected,**summarize(response,time.perf_counter()-started)})
    result={'controlled_real_retrieval':outputs,'live':[],'provider_invocations':0}
    if args.live and settings.google_api_key:
        extractor=GeminiIntentExtractor(api_key=settings.google_api_key,model_name=settings.gemini_model,
            timeout_seconds=15,retry_budget=0,structured_output_retry_budget=1,
            metadata_categories=repository.metadata_categories,metadata_tags=repository.metadata_tags,
            metadata_poi_names={p.id:p.name for p in repository.get_all()})
        underlying=extractor._structured_runnable
        class Counted:
            def invoke(self,messages):
                result['provider_invocations']+=1
                if result['provider_invocations']>4: raise RuntimeError('evaluation call budget exceeded')
                return underlying.invoke(messages)
        extractor._structured_runnable=Counted()
        app=create_app(settings,service_overrides=ServiceOverrides(poi_repository=repository,intent_extractor=extractor))
        with TestClient(app) as client:
            for prompt in [cases[0][1], 'Arrange an outdoor architecture and historic squares walk in Corfu on October 8, 2026, from 10:00 to 13:00.']:
                started=time.perf_counter(); response=client.post('/chat',json={'message':prompt})
                result['live'].append({'prompt':prompt,**summarize(response,time.perf_counter()-started)})
                if response.status_code>=500: break
    output=ROOT/'evaluation/results/pipeline-review.json'
    output.write_text(json.dumps(result,indent=2))
    print(json.dumps({'controlled':[(r['id'],r.get('status'),r['selected_ids']) for r in outputs],
        'live':[(r['http'],r.get('status'),r.get('selected_ids')) for r in result['live']],
        'provider_invocations':result['provider_invocations']}))

if __name__=='__main__': main()
