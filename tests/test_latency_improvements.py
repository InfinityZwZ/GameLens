"""Latency contracts with local transports and fixed model payloads; no API calls."""
import copy
import json
from datetime import datetime, timezone

import httpx
import pytest

from gamelens.core import llm as llm_module
from gamelens.core.llm import OpenAICompatibleClient,LLMResponseError,OUTPUT_BUDGET
from gamelens.core.runtime import InvestigationAgent,LLMPlanner,ToolRegistry,PlannerDecision
from gamelens.core.final_repair import apply_repair,draft_signature
from gamelens.core.grounding import fact_catalog,resolve_fact_ids
from gamelens.memory.context import PlannerContextBuilder
from gamelens.storage.telemetry import summarize
from gamelens.storage.timing import turn_timing,model_status
from test_finish_repair import ready_state,PayloadClient


def test_default_output_budgets_are_finite_and_zero_means_safe_auto(monkeypatch):
    for name in ('LLM_MAX_OUTPUT_TOKENS','LLM_PLANNER_MAX_TOKENS','LLM_FINAL_MAX_TOKENS','LLM_REPAIR_MAX_TOKENS'):
        monkeypatch.setenv(name,'0')
    client=OpenAICompatibleClient('test',api_key='fake')
    planner=LLMPlanner(client)
    assert client.max_output_tokens==4096
    assert planner.planning_budget==3072 and planner.final_budget==4096 and planner.repair_budget==3072


def verified_ready(tmp_path):
    import hashlib
    state=ready_state()
    source=tmp_path/'source.json';source.write_text('{}')
    state['observations'][0]['data']['source']={'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    state['observations'].append({'step':3,'tool':'verify_investigation','ok':True,'data':{'verified_files':1}})
    state['step_count']=3
    state['artifacts']['final_report']=str(tmp_path/'report.md')
    return state


def test_generated_field_patch_preserves_unchanged_report_and_reruns_gates(tmp_path):
    state=verified_ready(tmp_path)
    original={'action':'finish','reason':'answer','final_answer':'本地有7个补丁。',
        'detailed_answer':'完整分析报告：仅代表缓存资料。',
        'claims':[{'label':'补丁数','source_step':2,'fact_path':['patch_count'],'value':8}]}
    fixed={'mode':'patch','reason':'按源数字修正',
        'changes':{'claims':[{'label':'补丁数','source_step':2,'fact_path':['patch_count'],'value':7}]}}
    client=PayloadClient([original,fixed])
    result=InvestigationAgent(LLMPlanner(client),ToolRegistry([])).invoke(state)
    assert result['status']=='completed' and result['validation_retries']==1
    assert result['final_detailed_answer']==original['detailed_answer']
    assert result['final_claims'][0]['value']==7
    assert result['repair_origin']=='field_patch' and len(client.messages)==2
    small=json.loads(client.messages[1][1]['content'])
    assert 'original_draft' in small and 'repair_schema' in small
    assert 'planner_output_schema' not in small


def test_patch_can_edit_unique_wrong_sentence_without_rewriting_report():
    draft=PlannerDecision(action='finish',reason='draft',final_answer='缓存有8条。',
        detailed_answer='第一段保留。缓存有8条。最后一段保留。').model_dump(mode='json')
    result=apply_repair(draft,{'mode':'patch','reason':'修正一处数字','text_edits':[
        {'field':'final_answer','old':'8条','new':'7条'},
        {'field':'detailed_answer','old':'8条','new':'7条'}]})
    assert result.detailed_answer=='第一段保留。缓存有7条。最后一段保留。'


@pytest.mark.parametrize('edit',[
    {'field':'final_answer','old':'missing','new':'correct'},
    {'field':'final_answer','old':'重复','new':'correct'}])
def test_invalid_or_ambiguous_text_patch_rejected(edit):
    draft=PlannerDecision(action='finish',reason='draft',final_answer='重复重复').model_dump(mode='json')
    with pytest.raises(ValueError,match='唯一'):
        apply_repair(draft,{'mode':'patch','reason':'invalid','text_edits':[edit]})


def test_dropping_bad_citation_without_disclosure_is_not_a_repair():
    draft=PlannerDecision(action='finish',reason='draft',final_answer='问题已修复',
        citations=[{'source_step':1,'review_id':'fake','quote':'fake'}]).model_dump(mode='json')
    with pytest.raises(ValueError,match='删除'):
        apply_repair(draft,{'mode':'patch','reason':'hide error','changes':{'citations':[]}})


def test_invalid_repair_still_has_two_shared_attempt_limit(tmp_path):
    bad={'action':'finish','reason':'bad','claims':[{'label':'wrong','source_step':2,'fact_path':['patch_count'],'value':9}]}
    patch={'mode':'patch','reason':'still wrong','changes':{'claims':bad['claims']}}
    result=InvestigationAgent(LLMPlanner(PayloadClient([bad,patch,patch])),ToolRegistry([])).invoke(verified_ready(tmp_path))
    assert result['status']=='failed' and result['validation_retries']==2


def test_patch_can_request_replan_but_not_bypass_tool_arguments():
    draft=PlannerDecision(action='finish',reason='draft').model_dump(mode='json')
    result=apply_repair(draft,{'mode':'replan','reason':'需要原文','action':'get_review',
        'arguments':{'appid':730,'review_id':'r4'}})
    assert result.action=='get_review' and result.arguments['review_id']=='r4'
    with pytest.raises(ValueError):
        apply_repair(draft,{'mode':'replan','reason':'fake','action':'finish'})


def test_changed_scope_changes_draft_identity():
    state=ready_state();first=draft_signature(state)
    state['goal']='不同问题'
    assert draft_signature(state)!=first
    state=ready_state();state['observations'][0]['data']['source']={'path':'data.json','sha256':'changed'}
    assert draft_signature(state)!=first


def test_small_fact_catalog_ids_resolve_and_equal_counts_keep_population_sources():
    state=ready_state()
    state['observations']=[{'step':s,'ok':True,'tool':'query_reviews','data':{
        'stats':{'count':7,'positive_rate':.5},'filters':{'period':'before' if s==1 else 'after'},
        'model':{f'config_{i}':'irrelevant' for i in range(100)}}} for s in (1,2)]
    before=copy.deepcopy(state)
    context=PlannerContextBuilder().build(state,[])
    catalog=context['verified_fact_catalog']
    assert len([f for f in catalog if f['field']=='stats.count'])==2
    assert not any('model.' in f['field'] for f in catalog)
    _,errors=resolve_fact_ids(state['observations'],[f['fact_id'] for f in catalog])
    assert not errors and state==before
    assert len(json.dumps(catalog))<len(json.dumps(fact_catalog(state['observations'])))*.3


def test_mid_document_selected_quote_is_not_lost_by_compression():
    text='开头'*1000+'崩溃已经消失'+'结尾'*1000
    observation={'step':1,'ok':True,'tool':'search_game_evidence','data':{
        'reviews':[{'review_id':'r1','text':text,'created_at_timestamp':100,'is_positive':True}],
        'hits':[{'kind':'review','source_id':'r1','evidence_role':'counterexample',
                 'evidence_quote':'崩溃已经消失','original':{'text':text},'embedding_text':text}]}}
    projected=PlannerContextBuilder._project_observation(observation)
    assert '崩溃已经消失' in projected['data']['reviews'][0]['text']
    assert projected['data']['reviews'][0]['text'] in text
    assert projected['data']['hits'][0]['evidence_quote']=='崩溃已经消失'
    assert observation['data']['reviews'][0]['text']==text


def client(monkeypatch,handler,events):
    factory=httpx.Client
    monkeypatch.setattr(llm_module.httpx,'Client',lambda **kw:factory(transport=httpx.MockTransport(handler),**kw))
    monkeypatch.setattr(llm_module.time,'sleep',lambda _:None)
    return OpenAICompatibleClient('test',api_key='fake',telemetry_sink=events.append)


def test_timeout_log_exposes_type_but_not_credentials(monkeypatch,capsys):
    events=[]
    def handler(request):raise httpx.ReadTimeout('private detail sk-FAKESECRET',request=request)
    c=client(monkeypatch,handler,events);c.max_retries=0
    with pytest.raises(httpx.ReadTimeout):c.generate_json([])
    out=capsys.readouterr().out
    assert 'ReadTimeout' in out and '读取响应超时' in out
    assert 'FAKESECRET' not in out and 'private detail' not in str(events)


def test_total_retry_wait_deadline_prevents_four_long_waits(monkeypatch):
    events=[];clock=[0.0];requests=[]
    def handler(request):
        requests.append(request);clock[0]+=182
        raise httpx.ReadTimeout('slow',request=request)
    c=client(monkeypatch,handler,events)
    c.request_deadline=180
    monkeypatch.setattr(llm_module.time,'monotonic',lambda:clock[0])
    with pytest.raises(LLMResponseError) as error:c.generate_json([])
    assert error.value.error_code=='retry_deadline_exceeded'
    assert len(requests)==1 and summarize(events)['failed_calls']==1


@pytest.mark.parametrize('status,expected',[(429,'限流'),(503,'服务暂不可用'),(403,'权限')])
def test_http_error_detail_is_specific(monkeypatch,capsys,status,expected):
    events=[]
    c=client(monkeypatch,lambda _:httpx.Response(status,json={}),events);c.max_retries=0
    with pytest.raises(LLMResponseError):c.generate_json([])
    assert expected in capsys.readouterr().out
    assert any(e.get('http_status')==status for e in events)


def test_per_turn_time_persists_when_finished_and_counts_do_not_double_add():
    run={'status':'completed','started_at':'2026-10-11T00:00:00Z',
        'finished_at':'2026-10-11T00:03:54Z','events':[{'duration_ms':60000}]}
    value=turn_timing(run,telemetry={'latency_ms':90000,'http_requests':2,'retries':1})
    assert value['elapsed_ms']==234000 and value['model_request_ms']==90000 and value['tool_ms']==60000
    assert turn_timing(run,now=datetime(2026,12,1,tzinfo=timezone.utc))['elapsed_ms']==234000
    assert '不能相加' in value['note']


def test_live_and_legacy_timing_unknown_is_not_zero():
    now=datetime(2026,10,11,0,1,tzinfo=timezone.utc)
    assert turn_timing({'status':'running','started_at':'2026-10-11T00:00:00Z'},now=now)['elapsed_ms']==60000
    assert turn_timing({'status':'completed'})['elapsed_ms'] is None
    assert turn_timing({'status':'completed','started_at':'bad','finished_at':'bad'})['elapsed_ms'] is None


def test_model_status_does_not_show_cancelled_request_as_running():
    events=[{'kind':'request','status':'failed','error_type':'ReadTimeout'},
            {'kind':'request','status':'pending','retry':True}]
    assert model_status(events,True)['pending'] and model_status(events,True)['retrying']
    assert not model_status(events,False)['pending']


def test_timing_in_api_payload_survives_record_roundtrip():
    from gamelens.api.app import RunState,RunRequest,run_to_payload,run_from_payload
    state=RunState(run_id='unit-time',request=RunRequest(appid=730,goal='统计缓存'),status='completed',
        started_at='2026-10-11T00:00:00Z',finished_at='2026-10-11T00:03:54Z')
    saved=run_to_payload(state)
    assert saved['timing']['elapsed_ms']==234000
    assert run_to_payload(run_from_payload(saved))['timing']['elapsed_ms']==234000


def test_bad_text_patch_can_be_corrected_within_shared_budget(tmp_path):
    bad={'action':'finish','reason':'draft','claims':[{'label':'count','source_step':2,'fact_path':['patch_count'],'value':9}],
         'final_answer':'缓存有9个补丁。'}
    wrong={'mode':'patch','reason':'wrong target','text_edits':[{'field':'final_answer','old':'not present','new':'7'}]}
    good={'mode':'patch','reason':'fix count','changes':{'claims':[{'label':'count','source_step':2,'fact_path':['patch_count'],'value':7}]},
          'text_edits':[{'field':'final_answer','old':'9个','new':'7个'}]}
    client=PayloadClient([bad,wrong,good])
    result=InvestigationAgent(LLMPlanner(client),ToolRegistry([])).invoke(verified_ready(tmp_path))
    assert result['status']=='completed' and result['validation_retries']==2
    assert result['final_answer']=='缓存有7个补丁。' and len(client.messages)==3


def test_source_edit_disables_old_pending_draft(tmp_path):
    state=verified_ready(tmp_path)
    bad=PlannerDecision(action='finish',reason='bad',claims=[{'label':'count','source_step':2,'fact_path':['patch_count'],'value':9}])
    first=PayloadClient([bad.model_dump(mode='json')])
    agent=InvestigationAgent(LLMPlanner(first),ToolRegistry([]))
    state.update(agent._decide(state))
    assert state['pending_final_decision']
    from pathlib import Path
    Path(state['observations'][0]['data']['source']['path']).write_text('{"changed":true}')
    second=PayloadClient([{'action':'finish','reason':'attempt','fact_ids':['S2F1']}])
    result=InvestigationAgent(LLMPlanner(second),ToolRegistry([]))._decide(state)
    context=json.loads(second.messages[0][1]['content'].split('\n',1)[1])
    assert 'original_draft' not in context
    assert result['status']=='repairing'


def test_quick_repair_context_drops_unrelated_raw_data(tmp_path):
    state=verified_ready(tmp_path)
    state['observations'][0]['data']['manifest']={'big':'x'*10000}
    state['pending_final_decision']=PlannerDecision(action='finish',reason='bad',
        claims=[{'label':'count','source_step':2,'fact_path':['patch_count'],'value':9}]).model_dump(mode='json')
    state['validation_errors']=['数字/事实不匹配：count']
    client=PayloadClient([{'mode':'patch','reason':'fix','changes':{'fact_ids':['S2F1']}}])
    LLMPlanner(client).repair_final(state,[])
    context=json.loads(client.messages[0][1]['content'])
    assert 'manifest' not in json.dumps(context)
    assert len(client.messages)==1


def test_screening_budget_change_invalidates_search_reuse_identity(tmp_path,monkeypatch):
    from gamelens.tools.query_reuse import ReadonlySnapshot
    from gamelens.rag.tools import SearchArgs
    path=tmp_path/'data/rag/730/index.sqlite';path.parent.mkdir(parents=True);path.write_bytes(b'unit')
    snapshot=ReadonlySnapshot(tmp_path,True)
    args=SearchArgs(appid=730,query='启动崩溃',filters={'period':'all'})
    first=snapshot(args,{'appid':730,'goal':'调查崩溃'})
    monkeypatch.setenv('LLM_SCREEN_MAX_TOKENS','8192')
    assert snapshot(args,{'appid':730,'goal':'调查崩溃'})!=first


def test_format_fallback_shares_retry_deadline(monkeypatch):
    clock=[0.0];events=[];calls=[]
    def handler(request):
        calls.append(request);clock[0]+=181
        return httpx.Response(400,json={})
    c=client(monkeypatch,handler,events);c.request_deadline=180
    monkeypatch.setattr(llm_module.time,'monotonic',lambda:clock[0])
    with pytest.raises(LLMResponseError) as error:c.generate_json([])
    assert error.value.error_code=='retry_deadline_exceeded' and len(calls)==1


@pytest.mark.parametrize('code,description', [('insufficient_quota','配额/余额'),('rate_limit_exceeded','速率限制')])
def test_provider_code_distinguishes_quota_from_rate_limit(monkeypatch,capsys,code,description):
    events=[]
    c=client(monkeypatch,lambda _:httpx.Response(429,json={'error':{'code':code}}),events);c.max_retries=0
    with pytest.raises(LLMResponseError):c.generate_json([])
    assert description in capsys.readouterr().out
    request=next(e for e in events if e.get('kind')=='request' and e.get('status')=='failed')
    assert request['provider_error_code']==code
