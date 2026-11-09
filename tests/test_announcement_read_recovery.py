"""Offline regressions for cached bodies hidden by missing planner arguments."""
import copy
import json
from pathlib import Path

import pytest

import gamelens.core.runtime as runtime
from gamelens.core.action_feedback import observation_feedback
from gamelens.core.investigation_checks import assess_investigation_progress
from gamelens.core.llm import CALL_PHASE
from gamelens.core.runtime import InvestigationAgent, LLMPlanner, PlannerDecision
from gamelens.core.synthesis import evidence_packet
from gamelens.memory.context import PlannerContextBuilder
from gamelens.storage.atomic_io import atomic_write_json
from gamelens.storage.runs import RunStore
from gamelens.tools.announcement_comments import PrepareDiscussionArgs, QueryDiscussionArgs
from gamelens.tools.investigation import VerifyInvestigationArgs, ReviewInvestigation
from gamelens.tools.registry import SteamInvestigationTools
from test_agent_runtime import SequencePlanner
from test_announcement_comments import corpus, snapshot, NEWS


def inline(monkeypatch):
    monkeypatch.setattr(runtime, 'run_isolated',
        lambda handler, args, state, timeout: ('result', handler(args,state).model_dump(mode='json')))


def observe(state, tool, result, arguments=None):
    state['step_count'] = max([state.get('step_count',0)] + [o['step'] for o in state['observations']]) + 1
    state['observations'].append({'step':state['step_count'], 'tool':tool, 'ok':result.ok,
        'summary':result.summary, 'arguments':arguments or {}, 'data':result.data})
    state['artifacts'].update(result.artifacts)


def prepared(corpus):
    tools, state = corpus
    saved = snapshot(tools,state)
    saved['comments'][0]['text'] = 'Please improve performance and fix crashes.'
    atomic_write_json(tools.path(PrepareDiscussionArgs(appid=730,news_id=NEWS)),saved)
    result = tools.prepare_announcement_comments(PrepareDiscussionArgs(appid=730,news_id=NEWS),state)
    observe(state,'prepare_announcement_comments',result)
    return tools,state,result


def test_missing_appid_uses_trusted_session_and_preserves_original_arguments(corpus,monkeypatch):
    inline(monkeypatch)
    tools,state,_ = prepared(corpus)
    registry = SteamInvestigationTools(tools.root).registry()
    arguments = {'news_id':NEWS}
    before = copy.deepcopy(arguments)
    assert registry.validated_arguments('query_announcement_comments',arguments,state)['appid'] == 730
    assert registry.approval_request('query_announcement_comments',arguments,state) is None
    result = registry.execute('query_announcement_comments',arguments,state)
    assert result.ok and result.data['returned_count'] == 4
    assert result.data['comments'][0]['text'].startswith('Please improve')
    assert result.data['execution']['argument_adjustments'] == [
        {'field':'appid','rule':'trusted_session_identity','effective_value':730}]
    assert arguments == before


@pytest.mark.parametrize('appid',[None,999])
def test_explicit_wrong_or_null_appid_is_not_repaired(corpus,monkeypatch,appid):
    tools,state = corpus
    registry = SteamInvestigationTools(tools.root).registry()
    monkeypatch.setattr(runtime,'run_isolated',lambda *_: pytest.fail('Handler must not run'))
    result = registry.execute('query_announcement_comments',{'appid':appid,'news_id':NEWS},state)
    assert not result.ok and result.data['error_code'] == 'invalid_arguments'
    assert result.data['argument_errors'][0]['loc'] in [('appid',),['appid']]
    assert result.data['execution']['handler_executed'] is False
    assert registry.validated_arguments('query_announcement_comments',{'appid':appid,'news_id':NEWS},state) is None


def test_other_required_ids_are_not_guessed_and_errors_survive_feedback(corpus):
    tools,state = corpus
    result = SteamInvestigationTools(tools.root).registry().execute('query_announcement_comments',{},state)
    assert not result.ok and result.data['argument_errors'][0]['loc'] == ('news_id',)
    assert 'input' not in result.data['argument_errors'][0]
    feedback = observation_feedback(result,{},set(),state)
    assert feedback['error_code'] == 'invalid_arguments'
    assert feedback['argument_errors'][0]['type'] == 'missing'


def test_equivalent_invalid_arguments_stop_even_when_limit_changes(corpus):
    tools,state = corpus
    registry = SteamInvestigationTools(tools.root).registry()
    for limit in (20,50):
        result = registry.execute('query_announcement_comments',{'limit':limit},state)
        assert result.data['error_code'] == 'invalid_arguments'
        observe(state,'query_announcement_comments',result,{'limit':limit})
    result = registry.execute('query_announcement_comments',{'limit':10},state)
    assert result.data['error_code'] == 'repeated_invalid_arguments'
    assert result.data['execution']['handler_executed'] is False
    # A real correction is still allowed, and does not inherit the invalid guard.
    assert not registry.argument_errors('query_announcement_comments',{'news_id':NEWS},state)
    assert registry.validated_arguments('query_announcement_comments',{'news_id':NEWS},state)


def test_agent_stops_before_a_third_equivalent_invalid_tool_call(corpus,monkeypatch):
    inline(monkeypatch)
    tools,state = corpus
    planner = SequencePlanner([PlannerDecision(action='query_announcement_comments',reason='读正文',
        arguments={'limit':limit}) for limit in (20,50,10)])
    result = InvestigationAgent(planner,SteamInvestigationTools(tools.root).registry()).invoke(state)
    assert result['status'] == 'failed' and result['step_count'] == 2
    assert '重复2次' in result['error'] and 'news_id' in result['error']
    assert all(o['data']['error_code'] == 'invalid_arguments' for o in result['observations'])


def test_prepared_bodies_are_pending_read_not_empty_or_ready_to_summarize(corpus):
    _,state,result = prepared(corpus)
    assert result.data['body_read_status'] == 'not_read'
    assert 'returned_count' not in result.data and 'comments' not in result.data
    assert result.data['collected_count'] == 4
    assert result.data['next_read_arguments'] == {'appid':730,'news_id':NEWS}
    progress = assess_investigation_progress(state)
    assert next(c for c in progress['checks'] if c['requirement']=='announcement_comments')['status'] == 'pending_read'
    assert not progress['can_finalize']


class CaptureClient:
    def __init__(self,payload): self.payload=payload; self.messages=[]; self.phases=[]
    def generate_json(self,messages):
        self.messages.append(messages);self.phases.append(CALL_PHASE.get())
        return self.payload


def test_old_premature_verification_cannot_hide_tool_schema(corpus):
    tools,state,_ = prepared(corpus)
    verification = ReviewInvestigation(tools.root).verify_investigation(VerifyInvestigationArgs(appid=730),state)
    observe(state,'verify_investigation',verification)
    state['investigation_progress'] = assess_investigation_progress(state)
    client = CaptureClient({'action':'query_announcement_comments','reason':'尚未读正文','arguments':{'news_id':NEWS}})
    LLMPlanner(client).decide(state,SteamInvestigationTools(tools.root).registry().specifications())
    content = client.messages[0][1]['content']
    assert 'planner_output_schema' in content and 'pending_read' in content
    assert 'query_announcement_comments' in content and 'news_id' in content
    assert client.phases == ['planning']


def test_parameter_error_returns_to_planning_and_remains_actionable(corpus):
    tools,state,_ = prepared(corpus)
    registry = SteamInvestigationTools(tools.root).registry()
    observe(state,'query_announcement_comments',tools.query_announcement_comments(QueryDiscussionArgs(appid=730,news_id=NEWS),state))
    observe(state,'verify_investigation',ReviewInvestigation(tools.root).verify_investigation(VerifyInvestigationArgs(appid=730),state))
    read = registry.execute('query_announcement_comments',{},state)
    observe(state,'query_announcement_comments',read)
    client = CaptureClient({'action':'query_announcement_comments','reason':'补必填字段','arguments':{'news_id':NEWS}})
    LLMPlanner(client).decide(state,registry.specifications())
    content = client.messages[0][1]['content']
    assert 'argument_errors' in content and 'news_id' in content and 'Field required' in content
    assert client.phases == ['planning']


def test_confirmed_empty_thread_is_not_pending_read(corpus):
    tools,state,_ = prepared(corpus)
    path=tools.path(PrepareDiscussionArgs(appid=730,news_id=NEWS))
    saved=json.loads(path.read_text(encoding='utf-8'))
    saved['comments']=[]
    saved['coverage'].update(reported_total=0,collected_count=0,complete=True,has_more=False)
    atomic_write_json(path,saved)
    state.update(observations=state['observations'][:1],artifacts={},step_count=1)
    result=tools.prepare_announcement_comments(PrepareDiscussionArgs(appid=730,news_id=NEWS),state)
    observe(state,'prepare_announcement_comments',result)
    check=next(c for c in assess_investigation_progress(state)['checks'] if c['requirement']=='announcement_comments')
    assert check['status']=='satisfied' and result.data['collected_count']==0


def test_report_distinguishes_cached_from_read_bodies(corpus):
    from gamelens.analysis.report_prose import readable_statistics, write_complete_report
    tools,state,_ = prepared(corpus)
    text='\n'.join(readable_statistics(state['observations']))
    assert '缓存采集4条' in text and '已保存留言正文' in text and '还需执行读取工具' in text
    assert '展示0条' not in text
    verification=ReviewInvestigation(tools.root).verify_investigation(VerifyInvestigationArgs(appid=730),state)
    observe(state,'verify_investigation',verification)
    state['scope_limited']=True
    report=write_complete_report(state,'尚未完成正文阅读，不归纳玩家观点。')
    assert report.read_text(encoding='utf-8').startswith('# 游鉴 GameLens · 有限范围调查报告')


def test_compact_replanning_packet_includes_specific_schema_and_error(corpus):
    tools,state,_ = prepared(corpus)
    registry = SteamInvestigationTools(tools.root).registry()
    observe(state,'query_announcement_comments',registry.execute('query_announcement_comments',{},state))
    state['investigation_progress'] = assess_investigation_progress(state)
    packet,_ = evidence_packet(state,registry.specifications(),PlannerContextBuilder())
    schema = packet['replan_tool_schemas']['query_announcement_comments']
    assert {'appid','news_id'} <= set(schema['required'])
    assert packet['tool_failures'][0]['error_code'] == 'invalid_arguments'
    assert packet['tool_failures'][0]['argument_errors'][0]['type'] == 'missing'
    assert len(packet['replan_tool_schemas']) <= 2
    assert packet['bound_identity']['appid'] == 730


@pytest.mark.parametrize('early_finish,omit_citations',[(False,False),(True,False),(False,True)])
def test_real_planner_flow_reads_english_before_automatic_verification(corpus,monkeypatch,early_finish,omit_citations):
    inline(monkeypatch)
    tools,state,_ = prepared(corpus)
    state.update(observations=[],artifacts={},step_count=0)
    class Client:
        def __init__(self): self.stages=[]
        def generate_json(self,messages):
            content=messages[1]['content']; self.stages.append(CALL_PHASE.get())
            if self.stages[-1] == 'repair':
                packet=json.loads(content)
                assert omit_citations
                candidate=next(c for c in packet['available_comment_citations'] if c['quote'].startswith('Please improve'))
                return {'mode':'patch','reason':'补充与反馈有关的原文',
                    'changes':{'comment_citations':[candidate]}}
            if self.stages[-1] == 'synthesis':
                packet=json.loads(content)
                comments=[e for e in packet['evidence_catalog'] if e['kind']=='announcement_comment']
                assert any(e['quote'].startswith('Please improve') for e in comments)
                return {'action':'finish','reason':'离线固定答复，不评估语义质量',
                    'final_answer':'**已读取的留言**\n\n- 缓存包含英文与中文正文。\n\n仅用于离线链路验证，不代表全部玩家。',
                    'fact_ids':[next(f['fact_id'] for f in packet['fact_catalog'] if f['field']=='returned_count')],
                    'comment_evidence_ids':[] if omit_citations else [e['evidence_id'] for e in comments], 'scope_limited':True}
            if len(self.stages)==1:
                return {'action':'list_announcements','reason':'定位','arguments':{'published_on':'2026-10-08'}}
            if len(self.stages)==2:
                return {'action':'prepare_announcement_comments','reason':'复用缓存','arguments':{'news_id':NEWS}}
            if early_finish:
                return {'action':'finish','reason':'过早总结','final_answer':'尚未读取正文','scope_limited':True}
            return {'action':'query_announcement_comments','reason':'读正文','arguments':{'news_id':NEWS}}
    client=Client()
    monkeypatch.setattr('gamelens.tools.steam_announcement_comments.fetch_announcement_comments',
        lambda *_:pytest.fail('Must reuse cached discussion, not fetch'))
    result = InvestigationAgent(LLMPlanner(client),SteamInvestigationTools(tools.root).registry()).invoke(state)
    assert result['status'] == 'completed',(result.get('error'),[(o['tool'],o['ok'],o.get('summary')) for o in result['observations']],result['investigation_progress'])
    names=[o['tool'] for o in result['observations']]
    assert names == ['inspect_data','list_announcements','prepare_announcement_comments','query_announcement_comments','verify_investigation']
    assert result['step_count'] == 5
    assert client.stages == ['planning']*3 + ['synthesis'] + (['repair'] if omit_citations else [])
    assert result['observations'][3]['data']['returned_count'] == 4
    assert len(result['final_comment_citations']) == (1 if omit_citations else 4)
    if early_finish:
        assert any(e.get('program_override') for e in result['decision_trace'])


def test_new_comment_read_invalidates_previous_verification(corpus,monkeypatch):
    inline(monkeypatch)
    tools,state,_ = prepared(corpus)
    registry = SteamInvestigationTools(tools.root).registry()
    agent = InvestigationAgent(SequencePlanner([]),registry)
    verification=ReviewInvestigation(tools.root).verify_investigation(VerifyInvestigationArgs(appid=730),state)
    observe(state,'verify_investigation',verification)
    assert state['artifacts']['evidence_assessed'] == 'true'
    state.update(next_action='query_announcement_comments',next_arguments={'news_id':NEWS})
    updates=agent._execute(state)
    assert 'final_report' not in updates['artifacts'] and 'evidence_assessed' not in updates['artifacts']


@pytest.mark.parametrize('outcome',['partial_result','budget_exhausted','needs_clarification',None])
def test_conversation_summary_exposes_outcome_without_rewriting_payload(tmp_path,outcome):
    store=RunStore(tmp_path/'runs.sqlite')
    payload={'run_id':'one','status':'completed','started_at':'2026-10-11T00:00:00Z',
        'updated_at':'2026-10-11T00:00:00Z','request':{'appid':730,'goal':'测试'}}
    if outcome: payload['investigation_progress']={'outcome':outcome}
    store.save(payload)
    before=store.get('one')
    row=store.list_conversations()[0]
    assert row['investigation_outcome'] == outcome and row['status'] == 'completed'
    assert store.get('one') == before
