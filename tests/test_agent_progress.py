"""P0 task coverage and audit regression; no provider or embedding model is called."""
import copy
import json
import sqlite3

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import ValidationError

from gamelens.core.decision_trace import safe_audit, append_trace, decision_metrics, observation_status
from gamelens.core.investigation_checks import assess_investigation_progress, progress_finish_errors
from gamelens.core.runtime import InvestigationAgent, PlannerDecision, AgentTool, ToolRegistry, ToolResult
from gamelens.evaluation.benchmark import initial_state
from gamelens.evaluation.business_cases import materialize_dataset
from gamelens.tools.registry import SteamInvestigationTools
from gamelens.tools.investigation import required_patch_count
from test_agent_runtime import SequencePlanner, EmptyArgs


def state(goal='调查启动闪退，官方公告是否提到修复？'):
    s = initial_state('p0', goal)
    s.update(investigation_profile='focused', rag_mode='auto')
    s['artifacts'] = {'final_report':'unit-test-only.md', 'evidence_assessed':'true'}
    return s


def observation(tool='search_game_evidence', **data):
    return {'step':1,'tool':tool,'ok':True,'data':data}


def statuses(progress):
    return {c['requirement']:c['status'] for c in progress['checks']}


def test_simple_statistics_needs_no_retrieval():
    s=state('全部缓存评论数量与推荐率')
    s['observations']=[observation('query_reviews', filters={'period':'all','is_positive':None},
        stats={'count':2,'positive_count':1,'negative_count':1,'positive_rate':.5,'negative_rate':.5})]
    p=assess_investigation_progress(s)
    assert p['required_evidence']==['statistics'] and p['can_finalize']
    assert not progress_finish_errors(p, False)


def test_known_review_id_does_not_require_counterexample_search():
    s=state('给我崩溃评论r20的原文')
    s['observations']=[observation('get_review',review={'review_id':'r20','text':'启动崩溃'})]
    p=assess_investigation_progress(s,citations=[{'source_step':1,'review_id':'r20'}])
    assert p['required_evidence']==['review_evidence'] and p['can_finalize']


def test_verified_statistics_do_not_answer_issue_or_official_question():
    s=state()
    s['observations']=[observation('query_reviews', filters={'period':'all'},
        stats={'count':2,'positive_count':1,'negative_count':1,'positive_rate':.5,'negative_rate':.5})]
    p=assess_investigation_progress(s, citations=[], news_citations=[])
    assert statuses(p)['announcement_check']=='not_checked'
    assert statuses(p)['review_evidence']=='insufficient_evidence'
    assert p['verified_sources_current'] and not p['can_finalize']
    assert progress_finish_errors(p, False) and not progress_finish_errors(p, True)


@pytest.mark.parametrize('coverage,expected', [
    ({'status':'partial_snapshot'},'no_hit'),
    ({'status':'outside_observed_range'},'no_data_coverage'),
    ({'status':'no_cached_reviews'},'no_data_coverage'),
    ({'status':'partial_snapshot','requested_start_not_observed':True},'insufficient_coverage')])
def test_empty_results_distinguish_coverage(coverage,expected):
    s=state('调查启动闪退')
    s['observations']=[observation(reviews=[],returned_count=0,review_coverage=coverage,filters={'is_positive':None})]
    p=assess_investigation_progress(s, citations=[])
    assert statuses(p)['review_evidence']==expected and not p['can_finalize']
    assert p['semantic_accuracy_verified'] is False


def test_tool_failure_is_not_no_hit():
    s=state('调查启动闪退')
    s['observations']=[{'step':1,'tool':'search_game_evidence','ok':False,'data':{'timed_out':True}}]
    assert statuses(assess_investigation_progress(s))['review_evidence']=='tool_failed'


def test_ambiguity_cannot_become_current_year():
    s=state('7月10日更新后是否有闪退')
    s['observations']=[observation('list_patches',requested_date='07-10',needs_year_clarification=True,
        lookup_status='ambiguous_year',matched_years=['2025','2026'])]
    p=assess_investigation_progress(s)
    assert p['needs_clarification'] and p['target']['requested_date']=='07-10'
    assert statuses(p)['target_update']=='ambiguous_target'


def test_invalid_calendar_date_stops_before_planning():
    s=state('调查2026年2月30日更新后的闪退')
    p=assess_investigation_progress(s,scope_limited=True)
    assert p['needs_clarification'] and p['hard_errors'] and not p['can_report_limited']
    result=InvestigationAgent(SequencePlanner([]),ToolRegistry([])).invoke(s)
    assert result['status']=='failed' and '未调用模型' in result['error']


@pytest.mark.parametrize('goal', ['比较两次更新后的反馈','对比p1与p2更新后反馈','两次版本的对比'])
def test_two_versions_not_treated_as_one(goal):
    assert required_patch_count(goal)==2
    p=assess_investigation_progress(state(goal))
    assert statuses(p)['version_coverage']=='insufficient_evidence'


def test_model_cannot_remove_explicit_requirements():
    s=state()
    p=assess_investigation_progress(s, required_evidence=['statistics'])
    assert set(p['required_evidence'])=={'statistics','review_evidence','announcement_check','counterexample_check'}
    s['investigation_progress']=p
    assert 'statistics' in assess_investigation_progress(s)['required_evidence']
    s['goal']='全部缓存评论数量与推荐率'
    assert assess_investigation_progress(s)['required_evidence']==['statistics']


def test_quote_coverage_is_not_semantic_self_confidence():
    s=state()
    s['observations']=[observation(filters={'is_positive':None},reviews=[{'review_id':'r1','text':'没有闪退'},
        {'review_id':'r2','text':'以前崩溃现在恢复正常'}],
        announcements=[{'news_id':'n1','contents':'Fixed stability'}],
        hits=[{'kind':'review','source_id':'r1','evidence_role':'counterexample'}])]
    p=assess_investigation_progress(s, citations=[{'source_step':1,'review_id':'r1'}],
        news_citations=[{'source_step':1,'news_id':'n1'}])
    assert p['can_finalize'] and p['semantic_accuracy_verified'] is False
    # This test checks coverage only; negation interpretation is not declared true.
    assert '置信度' not in json.dumps(p, ensure_ascii=False)


def test_missing_quotes_need_limitation_not_unchecked_completion():
    s=state()
    s['observations']=[observation(filters={'is_positive':None},reviews=[{'review_id':'r1','text':'闪退'}],
        announcements=[{'news_id':'n1','contents':'修复稳定性'}])]
    p=assess_investigation_progress(s, citations=[],news_citations=[])
    assert statuses(p)['review_evidence']=='insufficient_evidence'
    assert statuses(p)['announcement_check']=='insufficient_evidence'


def test_review_only_search_does_not_count_as_announcement_check():
    s=state()
    o=observation(reviews=[],returned_count=0,filters={'is_positive':None},
        evidence_coverage={'requested_kinds':['review']})
    s['observations']=[o]
    assert statuses(assess_investigation_progress(s))['announcement_check']=='not_checked'


def test_negative_only_sampling_is_not_counterexample_check():
    s=state('调查闪退是否还有反例')
    s['observations']=[observation(filters={'is_positive':False},reviews=[{'review_id':'r1','text':'闪退'}])]
    assert statuses(assess_investigation_progress(s))['counterexample_check']=='insufficient_evidence'


def test_known_counterexample_cannot_be_silently_omitted():
    s=state('调查启动闪退')
    s['observations']=[observation(filters={'is_positive':None},
        reviews=[{'review_id':'r1','text':'没有闪退'},{'review_id':'r2','text':'闪退'}],
        hits=[{'kind':'review','source_id':'r1','evidence_role':'counterexample'}])]
    p=assess_investigation_progress(s,citations=[{'source_step':1,'review_id':'r2'}])
    assert statuses(p)['counterexample_check']=='insufficient_evidence'


def test_explicit_cached_cohort_goal_keeps_range_warning_without_false_failure():
    s=state('本地缓存评论数量与推荐率')
    s['observations']=[observation('query_reviews',filters={'period':'all'},
        coverage={'status':'partial_snapshot','requested_start_not_observed':True},
        stats={'count':2,'positive_count':1,'negative_count':1,'positive_rate':.5,'negative_rate':.5})]
    p=assess_investigation_progress(s)
    assert p['can_finalize'] and p['effective_scopes'][0]['coverage']['requested_start_not_observed']


def test_budget_is_distinct_from_success():
    s=state();s['step_count']=s['max_steps']
    p=assess_investigation_progress(s)
    assert not p['can_continue'] and not p['can_finalize']
    assert p['outcome']=='budget_exhausted'
    assert assess_investigation_progress(s,scope_limited=True)['outcome']=='budget_exhausted'


def test_wrong_language_or_playtime_scope_cannot_supply_issue_evidence():
    s=state('调查更新后中文玩家评价时100小时以上的启动闪退')
    s['observations']=[observation(filters={'period':'after','language':'english','min_playtime_hours':100},
        patch_news_id='p1',reviews=[{'review_id':'r1','text':'闪退','language':'english'}])]
    p=assess_investigation_progress(s,citations=[{'source_step':1,'review_id':'r1'}])
    assert statuses(p)['review_evidence']=='not_checked'
    assert p['requested_player_conditions']=={'language':'schinese','min_playtime_hours':100,'playtime_basis':'at_review'}


def test_old_id_quote_cannot_answer_post_update_issue():
    s=state('p1更新后一天是否仍有启动闪退')
    s['observations']=[observation('query_reviews',filters={'period':'after','is_positive':None},
        patch_news_id='p1',bounds={'start_inclusive':100000,'end_exclusive':186400},reviews=[]),
        {'step':2,'tool':'get_review','ok':True,'data':{'review':{'review_id':'old','text':'闪退','created_at_timestamp':99999}}}]
    p=assess_investigation_progress(s,citations=[{'source_step':2,'review_id':'old'}])
    assert statuses(p)['review_evidence']=='insufficient_evidence' and not p['can_finalize']


def test_original_decision_payload_remains_valid():
    old=PlannerDecision.model_validate({'action':'query_reviews','reason':'统计','arguments':{'appid':730}})
    assert old.evidence_gap=='' and old.required_evidence==[]
    with pytest.raises(ValidationError):
        PlannerDecision(action='finish',reason='不可识别要求',required_evidence=['confidence_95'])


def test_audit_does_not_store_sensitive_values_or_long_raw_evidence():
    raw={'arguments':{'authorization':'Bearer abc','api_key':'private','email':'a@b.com'},
         'reason':'sk-secret_abc email person@example.com token=secret', 'evidence_gap':'x'*10000}
    safe=safe_audit(raw)
    assert 'private' not in json.dumps(safe) and 'person@example.com' not in json.dumps(safe)
    assert 'sk-secret' not in json.dumps(safe) and 'token=secret' not in json.dumps(safe)
    assert len(safe['evidence_gap'])==1000 and raw['arguments']['api_key']=='private'


def test_trace_replay_ids_are_stable():
    s=state()
    e={'event':'observation','action':'query_reviews','step':1,'next_decision':'continue'}
    s['decision_trace']=append_trace(s,e)
    assert len(append_trace(s,e))==1
    s['generation_recoveries']=1
    assert len(append_trace(s,e))==2


def test_observation_status_no_data_not_no_match():
    assert observation_status(ToolResult(ok=True,summary='空',data={'returned_count':0}))=='no_match'
    assert observation_status(ToolResult(ok=True,summary='缺资料',data={'returned_count':0,
        'coverage':{'status':'outside_observed_range'}}))=='no_data_coverage'
    assert observation_status(ToolResult(ok=False,summary='失败'))=='tool_failed'


@pytest.fixture
def corpus(tmp_path):
    materialize_dataset(tmp_path)
    s=initial_state('p0-runtime','p1更新后是否仍有启动闪退，官方公告是否提到修复？')
    s.update(investigation_profile='focused',rag_mode='auto')
    return tmp_path,s,SteamInvestigationTools(tmp_path).registry()


class PrematurePlanner:
    def __init__(self,limited=False):self.limited=limited
    def decide(self,s,tools):
        if not s['observations']:
            return PlannerDecision(action='query_reviews',reason='取得实际窗口统计',evidence_gap='尚无玩家问题证据',
                arguments={'appid':730,'patch_news_id':'p1','filters':{'period':'after'}})
        if s['observations'][-1]['tool']=='query_reviews':
            return PlannerDecision(action='verify_investigation',reason='核对文件哈希',arguments={'appid':730})
        q=s['observations'][0]
        return PlannerDecision(action='finish',reason='尝试凭计数结束',scope_limited=self.limited,
            final_answer='只能报告缓存统计。',claims=[{'label':'缓存条数','source_step':q['step'],
                'fact_path':['stats','count'],'value':q['data']['stats']['count']}])


def test_premature_finish_is_replaced_with_verified_limited_statistics(corpus):
    root,s,registry=corpus
    result=InvestigationAgent(PrematurePlanner(),registry).invoke(s)
    assert result['status']=='completed' and result['scope_limited']
    assert len(result['observations'])==2
    assert '反馈分析' in result['final_answer']
    assert '官方公告核对' in result['final_answer'] and '玩家问题证据' in result['final_answer']
    assert result['investigation_progress']['outcome']=='partial_result'
    assert decision_metrics(result)['completion_gate_rejection_count']==1
    assert all('event_id' in t for t in result['decision_trace'])


def test_limited_report_discloses_unanswered_aspects(corpus):
    root,s,registry=corpus
    result=InvestigationAgent(PrematurePlanner(True),registry).invoke(s)
    assert result['status']=='completed' and result['scope_limited']
    assert result['investigation_progress']['outcome']=='partial_result'
    assert '官方公告核对' in result['final_answer'] and '玩家问题证据' in result['final_answer']
    assert not result['investigation_progress']['can_finalize']


def test_source_change_invalidates_progress_and_finalize(corpus):
    root,s,registry=corpus
    result=InvestigationAgent(PrematurePlanner(True),registry).invoke(s)
    (root/'data/raw/730_reviews.json').write_text('[]',encoding='utf-8')
    assert not assess_investigation_progress(result)['verified_sources_current']
    final=InvestigationAgent._finalize(result)
    assert final['status']=='failed' and '资料已变化' in final['error']


def test_scope_limited_cannot_bypass_invalid_window():
    s=state('查询缓存评论数量与推荐率')
    s['observations']=[observation('query_reviews',bounds={'start_inclusive':20,'end_exclusive':10})]
    p=assess_investigation_progress(s,scope_limited=True)
    assert not p['can_report_limited'] and progress_finish_errors(p,True)


def test_invalid_arguments_error_does_not_echo_secret_input():
    registry=ToolRegistry([AgentTool('test','test',EmptyArgs,lambda a,s:ToolResult(ok=True,summary='ok'))])
    result=registry.execute('test',{'api_key':'sk-do-not-print'},state())
    assert not result.ok and result.data['error_code']=='invalid_arguments'
    assert 'sk-do-not-print' not in result.summary


def test_finalize_rechecks_goal_not_just_cached_progress(corpus):
    root,s,registry=corpus
    result=InvestigationAgent(PrematurePlanner(True),registry).invoke(s)
    result['scope_limited']=False
    result['investigation_progress']['gaps']=[]
    final=InvestigationAgent._finalize(result)
    assert final['status']=='failed' and 'announcement_check' in final['error']


def test_effective_arguments_and_new_ids_are_observable(corpus):
    root,s,registry=corpus
    result=InvestigationAgent(PrematurePlanner(True),registry).invoke(s)
    q=result['observations'][0]
    assert q['effective_arguments']['limit']==5 and q['duration_ms']>=0
    assert q['new_evidence_ids'] and q['evidence_gap']=='尚无玩家问题证据'
    assert len({t['event_id'] for t in result['decision_trace']})==len(result['decision_trace'])


def test_approval_restore_preserves_trace_and_does_not_repeat_side_effect(tmp_path):
    output=tmp_path/'count.txt'
    def handler(args,s):
        output.write_text('once')
        return ToolResult(ok=True,summary='已执行',artifacts={'final_report':'unit.md','evidence_assessed':'true'})
    registry=ToolRegistry([AgentTool('approved','审批',EmptyArgs,handler,lambda a,s:'确认？')])
    s=initial_state('legacy-checkpoint','检查已获取的证据')
    with sqlite3.connect(tmp_path/'checkpoint.sqlite',check_same_thread=False) as conn:
        a=InvestigationAgent(SequencePlanner([PlannerDecision(action='approved',reason='需要确认')]),registry,
            checkpointer=SqliteSaver(conn))
        list(a.stream(s))
        before=a.snapshot(s['thread_id']).values
        assert len(before['decision_trace'])==1 and not output.exists()
    with sqlite3.connect(tmp_path/'checkpoint.sqlite',check_same_thread=False) as conn:
        b=InvestigationAgent(SequencePlanner([PlannerDecision(action='finish',reason='结束')]),registry,
            checkpointer=SqliteSaver(conn))
        list(b.resume_stream(s['thread_id'],approved=True))
        result=b.snapshot(s['thread_id']).values
    assert output.read_text()=='once' and len(result['observations'])==1
    assert result['decision_trace'][0]['event_id']==before['decision_trace'][0]['event_id']
    assert len({t['event_id'] for t in result['decision_trace']})==len(result['decision_trace'])


def test_unknown_semantics_not_reported_as_zero_or_success():
    m=decision_metrics({'observations':[]})
    assert m['semantic_success'] is None and m['unnecessary_tool_calls'] is None
    assert m['has_structured_trace'] is False


def test_progress_context_does_not_duplicate_scope_or_trust_history():
    from gamelens.memory.context import PlannerContextBuilder
    s=state()
    s['investigation_progress']=assess_investigation_progress(s)
    s['conversation_memory']={'assistant_answer_unverified':'之前已经全部修复'}
    context=PlannerContextBuilder().build(s,[])
    p=context['investigation_progress']
    assert 'effective_scopes' not in p and 'goal' not in p and p['gaps']
    assert p['semantic_accuracy_verified'] is False


def test_web_payload_preserves_new_fields_and_accepts_old_records():
    from gamelens.api.app import RunRequest, RunState, run_to_payload, run_from_payload
    run=RunState(run_id='p0-web',request=RunRequest(appid=730),status='completed')
    run.decision_trace=[{'event_id':'test','next_decision':'partial_result'}]
    run.investigation_progress={'outcome':'partial_result','can_finalize':False}
    payload=run_to_payload(run)
    restored=run_from_payload(payload)
    assert restored.decision_trace==run.decision_trace and restored.investigation_progress==run.investigation_progress
    payload.pop('decision_trace');payload.pop('investigation_progress')
    restored=run_from_payload(payload)
    assert restored.decision_trace==[] and restored.investigation_progress=={}


def test_limited_delivery_not_scored_as_full_task_success(tmp_path):
    from test_business_evaluation import authored_run
    from gamelens.evaluation.business_cases import CASES
    from gamelens.evaluation.quality import evaluate_business_run
    case,s,dataset=authored_run(tmp_path,CASES[0])
    s['investigation_progress']={'can_finalize':False,'outcome':'partial_result'}
    row=evaluate_business_run(case,s,dataset)
    assert not row['checks']['progress_complete'] and not row['automatic_passed']
