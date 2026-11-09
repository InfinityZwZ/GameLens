"""Offline regression tests for the exact failure modes in the live replay."""
import copy
import hashlib
import json

import httpx
import pytest
from pydantic import ValidationError

from gamelens.core import llm as llm_module
from gamelens.core.llm import OpenAICompatibleClient, LLMResponseError, LLMConfigurationError, CALL_PHASE
from gamelens.core.runtime import InvestigationAgent, LLMPlanner, AgentTool, ToolRegistry, ToolResult
from gamelens.core.synthesis import evidence_packet, expand_decision
from gamelens.memory.context import PlannerContextBuilder
from gamelens.rag.tools import SearchArgs
from gamelens.storage.telemetry import summarize
from gamelens.storage.timing import model_status
from gamelens.tools.argument_normalization import normalize_search_arguments
from gamelens.tools.registry import AppArgs
from test_finish_repair import PayloadClient, ready_state


@pytest.mark.parametrize('model,host,thinking,stream',[
    ('qwen3.8-max','https://dashscope.aliyuncs.com/compatible-mode/v1',False,True),
    ('qwen3.8-max-0902','https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1',False,True),
    ('qwen3.7-max-preview','https://dashscope.aliyuncs.com/compatible-mode/v1',None,False),
    ('qwen3.8-2.4t-a95b','https://dashscope.aliyuncs.com/compatible-mode/v1',None,False),
    ('qwen3.8-max','https://different-provider.test/v1',None,False),
    ('other-model','https://api.openai.com/v1',None,False)])
def test_auto_tuning_is_provider_and_model_specific(monkeypatch,model,host,thinking,stream):
    monkeypatch.setenv('LLM_ENABLE_THINKING','auto'); monkeypatch.setenv('LLM_STREAMING','auto')
    client=OpenAICompatibleClient(model,api_key='fake',base_url=host)
    assert client.enable_thinking is thinking and client.streaming is stream


def test_explicit_mode_and_invalid_configuration(monkeypatch):
    monkeypatch.setenv('LLM_ENABLE_THINKING','true');monkeypatch.setenv('LLM_STREAMING','false')
    client=OpenAICompatibleClient('test',api_key='fake')
    assert client.enable_thinking and not client.streaming
    monkeypatch.setenv('LLM_ENABLE_THINKING','perhaps')
    with pytest.raises(LLMConfigurationError):OpenAICompatibleClient('test',api_key='fake')


def sse(delta=None,finish=None,usage=None):
    value={'choices':[] if usage else [{'index':0,'delta':delta or {},'finish_reason':finish}]}
    if usage:value['usage']=usage
    return ('data: '+json.dumps(value,ensure_ascii=False)+'\n\n').encode()


def stream_client(monkeypatch,handler,events):
    factory=httpx.Client
    monkeypatch.setenv('LLM_ENABLE_THINKING','auto');monkeypatch.setenv('LLM_STREAMING','true')
    monkeypatch.setattr(llm_module.httpx,'Client',lambda **kw:factory(transport=httpx.MockTransport(handler),**kw))
    monkeypatch.setattr(llm_module.time,'sleep',lambda _:None)
    return OpenAICompatibleClient('qwen3.8-max',api_key='fake',base_url='https://dashscope.aliyuncs.com/v1',telemetry_sink=events.append)


def test_sse_assembly_counts_one_request_and_never_logs_reasoning(monkeypatch):
    events=[];payloads=[]
    def handler(request):
        payloads.append(json.loads(request.content))
        data=sse({'reasoning_content':'PRIVATE_THOUGHT'})+sse({'content':'{"value":'})+sse({'content':'"中文"}'},'stop')
        data+=sse(usage={'prompt_tokens':10,'completion_tokens':5,'completion_tokens_details':{'reasoning_tokens':0}})+b'data: [DONE]\n\n'
        return httpx.Response(200,content=data,headers={'content-type':'text/event-stream'})
    client=stream_client(monkeypatch,handler,events)
    assert client.generate_json([])=={'value':'中文'}
    assert payloads[0]['enable_thinking'] is False and payloads[0]['stream']
    assert 'PRIVATE_THOUGHT' not in json.dumps(events)
    requests={e['id']:e for e in events if e['kind']=='request'}
    assert len(requests)==1 and list(requests.values())[0]['reasoning_tokens']==0
    assert list(requests.values())[0]['completion_tokens']==5
    assert list(requests.values())[0]['first_content_ms']>=0
    assert model_status([{'kind':'request','status':'streaming','content_chars':20}],True)['pending']


@pytest.mark.parametrize('ending,code',[(b'', 'incomplete_stream'),(b'data: [DONE]\n\n','incomplete_stream'),
    (b'data: broken\n\n','invalid_stream')])
def test_incomplete_or_invalid_sse_is_not_accepted_or_retried(monkeypatch,ending,code):
    events=[];calls=[]
    def handler(request):
        calls.append(1)
        return httpx.Response(200,content=sse({'content':'{"ok":true}'})+ending,headers={'content-type':'text/event-stream'})
    client=stream_client(monkeypatch,handler,events)
    with pytest.raises(LLMResponseError) as error:client.generate_json([])
    assert error.value.error_code==code and len(calls)==1
    assert next(e for e in reversed(events) if e['kind']=='request')['status']=='failed'


def test_truncated_sse_json_is_rejected_even_if_json_parses(monkeypatch):
    events=[]
    data=sse({'content':'{"ok":true}'},'length')+b'data: [DONE]\n\n'
    client=stream_client(monkeypatch,lambda _:httpx.Response(200,content=data,headers={'content-type':'text/event-stream'}),events)
    with pytest.raises(LLMResponseError) as error:client.generate_json([])
    assert error.value.error_code=='output_truncated'


def test_provider_ignoring_stream_can_return_complete_json(monkeypatch):
    client=stream_client(monkeypatch,lambda _:httpx.Response(200,json={'choices':[{'message':{'content':'{"ok":true}'},'finish_reason':'stop'}]}),[])
    assert client.generate_json([])=={'ok':True}


def test_stall_after_partial_stream_does_not_regenerate_whole_answer(monkeypatch):
    events=[];calls=[]
    class StallingStream(httpx.SyncByteStream):
        def __iter__(self):
            yield sse({'content':'{"unfinished":'})
            raise httpx.ReadTimeout('stall')
    def handler(request):
        calls.append(1)
        return httpx.Response(200,stream=StallingStream(),headers={'content-type':'text/event-stream'})
    client=stream_client(monkeypatch,handler,events)
    with pytest.raises(httpx.ReadTimeout):client.generate_json([])
    assert len(calls)==1 and next(e for e in reversed(events) if e['kind']=='request')['content_chars']>0


def test_final_stage_timeout_does_not_repeat_full_generation(monkeypatch):
    events=[];calls=[]
    def handler(request):calls.append(1);raise httpx.ReadTimeout('stall before any content',request=request)
    client=stream_client(monkeypatch,handler,events)
    token=CALL_PHASE.set('synthesis')
    try:
        with pytest.raises(httpx.ReadTimeout):client.generate_json([])
    finally:CALL_PHASE.reset(token)
    assert len(calls)==1 and summarize(list({e['id']:e for e in events}.values()))['http_requests']==1


def test_stream_deadline_stops_continuous_chunks(monkeypatch):
    events=[];clock=[0.0]
    class EndlessStream(httpx.SyncByteStream):
        def __iter__(self):
            yield sse({'content':'{"a":'})
            clock[0]=300
            yield sse({'content':'1}'},'stop')
    client=stream_client(monkeypatch,lambda _:httpx.Response(200,stream=EndlessStream(),headers={'content-type':'text/event-stream'}),events)
    monkeypatch.setattr(llm_module.time,'monotonic',lambda:clock[0])
    with pytest.raises(LLMResponseError) as error:client.generate_json([])
    assert error.value.error_code=='retry_deadline_exceeded'


def source_state(tmp_path):
    state=ready_state();state.update(investigation_profile='focused',goal='核对缓存中的评论原文')
    source=tmp_path/'source.json';source.write_text('{}')
    report=tmp_path/'report.md';report.write_text('Verified source report',encoding='utf-8')
    state['artifacts']['final_report']=str(report)
    reference={'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'scope':'cached_reviews_only','cached_review_count':2}
    state['observations']=[{'step':1,'tool':'search_game_evidence','ok':True,'data':{
        'source':reference,'returned_count':2,'patch_news_id':'p1','filters':{'period':'after'},
        'reviews':[{'review_id':'a','text':'卡顿仍存在'},{'review_id':'b','text':'现在已经不卡顿了'}],
        'hits':[{'kind':'review','source_id':'a','evidence_role':'support','evidence_quote':'卡顿仍存在'},
                {'kind':'review','source_id':'b','evidence_role':'counterexample','evidence_quote':'现在已经不卡顿了'}]}},
        {'step':2,'tool':'verify_investigation','ok':True,'data':{'verified_files':1}}]
    state['step_count']=2
    return state


def test_final_packet_preserves_counterexample_quote_and_fact_ids(tmp_path):
    state=source_state(tmp_path);before=copy.deepcopy(state)
    packet,lookup=evidence_packet(state,[],PlannerContextBuilder())
    assert packet['evidence_catalog'][0]['role']=='counterexample'
    response={'action':'finish','reason':'回答','final_answer':'反馈有分歧，不能推断更新因果。',
        'fact_ids':[packet['fact_catalog'][0]['fact_id']],
        'review_evidence_ids':[e['evidence_id'] for e in packet['evidence_catalog']],
        'news_evidence_ids':[],'scope_limited':True}
    decision=expand_decision(response,lookup)
    assert {c.review_id for c in decision.citations}=={'a','b'} and state==before
    assert 'available_tools' in packet and not any(k in packet for k in ('current_plan','loop_control','earlier_tool_digest'))
    with pytest.raises(ValueError):expand_decision({**response,'review_evidence_ids':['FAKE']},lookup)
    with pytest.raises(ValueError):expand_decision({**response,'news_evidence_ids':[packet['evidence_catalog'][0]['evidence_id']]},lookup)


def test_real_final_planner_uses_synthesis_phase_and_small_schema(tmp_path):
    state=source_state(tmp_path)
    packet,lookup=evidence_packet(state,[],PlannerContextBuilder())
    client=PayloadClient([{'action':'finish','reason':'回答','fact_ids':[packet['fact_catalog'][0]['fact_id']],
        'review_evidence_ids':list(lookup),'news_evidence_ids':[],'scope_limited':True}])
    decision=LLMPlanner(client).decide(state,[])
    body=json.loads(client.messages[0][1]['content'])
    assert 'evidence_catalog' in body and 'planner_output_schema' not in body
    assert len(decision.citations)==2 and CALL_PHASE.get()=='json'


def test_stale_source_cannot_use_the_final_packet(tmp_path):
    state=source_state(tmp_path)
    (tmp_path/'source.json').write_text('changed')
    client=PayloadClient([{'action':'inspect_data','reason':'重新检查','arguments':{'appid':730}}])
    LLMPlanner(client).decide(state,[])
    assert '规划上下文' in client.messages[0][1]['content']


def test_exact_live_argument_failures_are_normalized_together():
    original={'appid':730,'query':'性能和崩溃','patch_news_id':'p1','lexical_terms':[str(i) for i in range(10)],'window_days':7}
    result,changes=normalize_search_arguments(original,{'goal':'调查最近一次更新后的性能和崩溃'})
    validated=SearchArgs.model_validate(result)
    assert validated.filters.period=='after' and len(validated.lexical_terms)==8 and len(changes)==2
    assert 'filters' not in original and len(original['lexical_terms'])==10
    assert validated.patch_news_id=='p1' and validated.window_days==7


@pytest.mark.parametrize('goal,filters',[
    ('比较更新前与更新后的反馈',{}),('查看全时段评论',{}),
    ('调查更新后的反馈',{'period':'before'}),
    ('调查更新后的反馈',{'period':'all','start_at':'2026-09-01T00:00:00Z','end_at':'2026-09-02T00:00:00Z'})])
def test_normalization_does_not_override_ambiguous_or_explicit_time(goal,filters):
    original={'appid':730,'query':'问题','patch_news_id':'p1','filters':filters}
    result,_=normalize_search_arguments(original,{'goal':goal})
    assert result['filters']==filters


def test_wrong_game_and_unsupported_constraints_are_not_fixed_away():
    original={'appid':999,'query':'问题','patch_news_id':'p1','filters':{'period':'after','language':'schinese'},'window_days':99}
    result,_=normalize_search_arguments(original,{'goal':'调查更新后的反馈','appid':730})
    assert result==original
    with pytest.raises(ValidationError):SearchArgs.model_validate(result)


def test_normalized_arguments_and_adjustments_survive_tool_boundary(tmp_path):
    def handler(args,state):return ToolResult(ok=True,summary='validated',data={'period':args.filters.period})
    tool=AgentTool('search_game_evidence','test',SearchArgs,handler,argument_normalizer=normalize_search_arguments)
    registry=ToolRegistry([tool],receipt_root=tmp_path/'receipts')
    arguments={'appid':730,'query':'问题','patch_news_id':'p1','lexical_terms':[str(i) for i in range(10)]}
    state={'thread_id':'normalization','appid':730,'step_count':0,'goal':'调查更新后的问题','observations':[]}
    result=registry.execute(tool.name,arguments,state)
    assert result.ok and result.data['period']=='after'
    assert len(result.data['execution']['argument_adjustments'])==2
    effective=registry.validated_arguments(tool.name,arguments,state)
    assert effective['filters']['period']=='after' and len(effective['lexical_terms'])==8


def test_auto_verification_is_only_source_check_not_auto_finish(tmp_path):
    state=source_state(tmp_path);state['observations']=state['observations'][:1]
    state['artifacts']={};state['step_count']=1
    client=PayloadClient([])
    registry=ToolRegistry([AgentTool('verify_investigation','核验',AppArgs,lambda a,s:ToolResult(ok=True,summary='核验'))])
    result=InvestigationAgent(LLMPlanner(client),registry)._decide(state)
    assert result['next_action']=='verify_investigation' and result['status']=='running'
    assert client.messages==[]


def test_partial_only_issue_evidence_requires_limited_answer(tmp_path):
    from gamelens.core.investigation_checks import assess_investigation_progress, progress_finish_errors
    state=source_state(tmp_path);state['goal']='调查卡顿问题'
    data=state['observations'][0]['data'];data['screening']={'enabled':True}
    for hit in data['hits']:hit['evidence_role']='partial'
    progress=assess_investigation_progress(state)
    assert any(g['requirement']=='review_evidence' and g['status']=='insufficient_evidence' for g in progress['gaps'])
    assert not progress['can_finalize'] and progress['can_report_limited']
    assert progress_finish_errors(progress,False) and not progress_finish_errors(progress,True)


def test_announcement_excerpt_retains_relevant_stability_clause(tmp_path):
    state=source_state(tmp_path);state['goal']='调查性能和崩溃'
    text='无关内容'*200+'Stability improvements.'+'后续内容'*200
    state['observations'][0]['data']['announcements']=[{'news_id':'p1','title':'更新','contents':text}]
    packet,_=evidence_packet(state,[],PlannerContextBuilder())
    announcement=next(e for e in packet['evidence_catalog'] if e['kind']=='announcement')
    assert 'Stability improvements.' in announcement['quote'] and announcement['quote'] in text
    assert announcement['excerpt_only']


def test_timeout_keeps_readable_verified_evidence_but_not_success(tmp_path):
    state=source_state(tmp_path)
    class TimeoutClient:
        def generate_json(self,messages):raise httpx.ReadTimeout('supplier stall')
    result=InvestigationAgent(LLMPlanner(TimeoutClient()),ToolRegistry([])).invoke(state)
    assert result['status']=='interrupted' and result['generation_pending']
    assert '调查未完成' in result['final_answer'] and '现在已经不卡顿了' in result['final_answer']
    assert {c['review_id'] for c in result['final_citations']}=={'a','b'}
    assert result['final_claims'] and 'partial_report' in result['artifacts']


def test_changed_source_cannot_be_presented_as_verified_fallback(tmp_path):
    from gamelens.core.synthesis import verified_evidence_partial
    state=source_state(tmp_path)
    (tmp_path/'source.json').write_text('changed')
    assert verified_evidence_partial(state) is None


@pytest.mark.parametrize('text',[
    '该更新于2026年9月22日发布。','该更新发布于2026-09-22。',
    '本次补丁发布时间为2026/09/22。',
    '设定的观察窗口内（起始于2026年9月22日），返回三条评论。'])
def test_cached_earliest_date_cannot_be_used_as_target_or_window(tmp_path,text):
    from gamelens.core.runtime import PlannerDecision
    from gamelens.core.synthesis import target_date_errors
    state=source_state(tmp_path)
    state['observations'][0]['data'].update(bounds={'start_inclusive':1790979853,'end_exclusive':1791584653},
        announcements=[{'news_id':'p1','title':'update','contents':'Stability improvements',
                        'published_at_timestamp':1790979853}])
    packet,_=evidence_packet(state,[],PlannerContextBuilder())
    assert packet['target_updates'][0]['published_on_local']=='2026-10-02'
    errors=target_date_errors(state,PlannerDecision(action='finish',reason='test',final_answer=text,scope_limited=True))
    assert errors and '2026-10-02' in errors[0]
    correct=text.replace('9月22','10月2').replace('09-22','10-02').replace('09/22','10/02')
    assert not target_date_errors(state,PlannerDecision(action='finish',reason='test',final_answer=correct))


@pytest.mark.parametrize('opening,closing',[('[',']'),('(',')'),('（','）')])
def test_internal_reference_cleanup_preserves_literal_controller_key(tmp_path,opening,closing):
    state=source_state(tmp_path);packet,lookup=evidence_packet(state,[],PlannerContextBuilder())
    identifier=next(iter(lookup));fact=packet['fact_catalog'][0]['fact_id']
    result=expand_decision({'action':'finish','reason':'test','final_answer':f'按[R2]键。结果{opening}{identifier}{closing}[{fact}]',
        'review_evidence_ids':[identifier],'news_evidence_ids':[],'fact_ids':[fact]},lookup)
    assert result.final_answer=='按[R2]键。结果'


def test_reference_cleanup_never_rewrites_a_literal_source_quote():
    from gamelens.core.synthesis import clean_evidence_references
    quote='按(EV_R1)键后游戏正常'
    assert clean_evidence_references(quote+'，参见(EV_R1)',{'EV_R1'},[quote])==quote+'，参见'


def test_cached_screening_identity_changes_with_thinking_configuration(tmp_path):
    from types import SimpleNamespace
    from gamelens.tools.registry import SteamInvestigationTools
    base={'model':'qwen3.8-max','base_url':'https://dashscope.aliyuncs.com/v1','max_output_tokens':4096,'streaming':True}
    a=SteamInvestigationTools(tmp_path,SimpleNamespace(**base,enable_thinking=True)).registry()
    b=SteamInvestigationTools(tmp_path,SimpleNamespace(**base,enable_thinking=False)).registry()
    assert a._tools['search_game_evidence'].readonly_snapshot.screening_model != b._tools['search_game_evidence'].readonly_snapshot.screening_model
