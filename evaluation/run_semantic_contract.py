"""Bounded live semantic-contract checks through the real /chat endpoint."""
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

ROOT=Path(__file__).resolve().parents[1]
settings=Settings()
output=ROOT/'evaluation/results/semantic-contract-live.json'
if not settings.google_api_key:
    output.write_text(json.dumps({'available':False,'reason':'credentials unavailable'})); raise SystemExit(0)
repository=POIRepository(load_dataset(settings.dataset_path),embedding_model_name=settings.embedding_model_name,embedding_cache_dir=settings.embedding_cache_dir)
repository.initialize_semantic_index()
extractor=GeminiIntentExtractor(api_key=settings.google_api_key,model_name=settings.gemini_model,
    timeout_seconds=15,retry_budget=0,structured_output_retry_budget=1,
    metadata_categories=repository.metadata_categories,metadata_tags=repository.metadata_tags,
    metadata_poi_names={p.id:p.name for p in repository.get_all()})
result={'available':True,'model':settings.gemini_model,'provider_invocations':0,'cases':[]}
underlying=extractor._structured_runnable
class Counted:
    def invoke(self,messages):
        result['provider_invocations']+=1
        if result['provider_invocations']>8: raise RuntimeError('evaluation budget exceeded')
        return underlying.invoke(messages)
extractor._structured_runnable=Counted()
class CapturedExtractor:
    latest=None
    def extract(self,message,conversation_state=None):
        value=extractor.extract(message,conversation_state)
        self.latest={
            'kind':value.intent,
            'interests':value.semantic_preferences.interests if value.semantic_preferences else None,
            'hard_filters':value.hard_filters.model_dump(exclude_none=True) if value.hard_filters else {},
            'filter_actions':{name:getattr(value.field_actions,name) for name in (
                'filter_category','filter_tags','filter_setting','filter_excluded_categories',
                'filter_excluded_tags','filter_excluded_poi_ids')},
            'evidence':[(item.field,item.value,item.kind) for item in value.constraint_evidence or []],
        }
        return value
captured=CapturedExtractor()
app=create_app(settings,service_overrides=ServiceOverrides(poi_repository=repository,intent_extractor=captured))
cases=[
 ('soft_culture','Put together a gallery and art afternoon in Corfu on October 8, 2026, 13:00–17:00.'),
 ('soft_architecture','Arrange an outdoor architecture and historic squares walk in Corfu on October 8, 2026, from 10:00 to 13:00.'),
 ('explicit','Plan a walk on October 8, 2026, 13:00–17:00. Only museums; every stop must be indoors.'),
 ('exclusion','Plan an outdoor walk on October 8, 2026, 10:00–13:00. I enjoy architecture, but no museums and no nightlife.'),
]
with TestClient(app) as client:
 for label,prompt in cases:
    captured.latest=None
    started=time.perf_counter(); response=client.post('/chat',json={'message':prompt})
    row={'id':label,'prompt':prompt,'http':response.status_code,'latency_seconds':round(time.perf_counter()-started,2),
         'parsed_intent':captured.latest}
    if response.status_code==200:
        d=response.json(); p=d.get('planning_result') or {}
        row.update(status=d['status'],intent=d['intent'],selected=p.get('selected_poi_ids',[]),
                   validation_issue_codes=[item['code'] for item in (p.get('validation') or {}).get('issues',[])])
    result['cases'].append(row)
    output.write_text(json.dumps(result,indent=2))
    print(label,row['http'],row.get('status'),row.get('selected'),flush=True)
    if response.status_code>=500: break
print('provider_invocations',result['provider_invocations'])
