"""P1 offline invariants; no provider, Steam, or embedding download."""
import copy
import json
from dataclasses import replace

import pytest

import gamelens.core.runtime as runtime
from gamelens.core.runtime import InvestigationAgent, ToolRegistry, AgentTool, ToolResult, PlannerDecision
from gamelens.core.action_feedback import observation_feedback, loop_signals, project_plan
from gamelens.core.decision_trace import decision_metrics
from gamelens.core.investigation_checks import assess_investigation_progress
from gamelens.evaluation.business_cases import materialize_dataset
from gamelens.evaluation.benchmark import initial_state
from gamelens.memory.context import PlannerContextBuilder
from gamelens.tools.registry import SteamInvestigationTools, AppArgs
from gamelens.tools.query_reuse import ReadonlySnapshot, call_identity, digest
from gamelens.domain.tool_results import classify_tool_exception
from test_agent_runtime import SequencePlanner, EmptyArgs


def setup(root):
    materialize_dataset(root)
    state = initial_state('p1', '全部缓存评论数量与推荐率')
    state.update(investigation_profile='focused',rag_mode='auto')
    return state, SteamInvestigationTools(root).registry()


def step(registry, state, action='query_reviews', arguments=None):
    state.update(next_action=action, next_arguments=arguments or {'appid':730,'filters':{'period':'all'}})
    state.update(InvestigationAgent(SequencePlanner([]), registry)._execute(state))
    return state['observations'][-1]


@pytest.fixture
def inline(monkeypatch):
    # Only for deterministic registry unit tests; integration below keeps isolation.
    calls = []
    def run(handler, args, state, timeout):
        calls.append((args.model_dump(mode='json'), state['step_count']))
        try:
            return 'result', handler(args, state).model_dump(mode='json')
        except BaseException as error:
            return 'error', classify_tool_exception(error).model_dump(mode='json')
    monkeypatch.setattr(runtime, 'run_isolated', run)
    return calls


def test_exact_defaults_reused_without_mutating_prior_observation(tmp_path, inline):
    state, registry = setup(tmp_path)
    first = copy.deepcopy(step(registry, state))
    expanded = registry.validated_arguments('query_reviews', {'appid':730,'filters':{'period':'all'}})
    second = step(registry, state, arguments=expanded)
    assert len(inline) == 1
    assert second['data']['execution']['reused_from_step'] == 1
    assert state['observations'][0] == first
    assert first['data']['stats'] == second['data']['stats']
    assert second['feedback']['new_evidence_ids'] == []
    assert not state['artifacts'].get('evidence_assessed')


@pytest.mark.parametrize('change', ['reviews','patches','window','patch_id','goal','profile',
                                    'query_args','audit_deleted','audit_changed','body_changed'])
def test_changed_dependency_or_result_does_not_reuse(tmp_path, inline, change):
    state, registry = setup(tmp_path)
    first = step(registry, state)
    args = {'appid':730,'filters':{'period':'all'}}
    if change in ('reviews','patches'):
        suffix = 'reviews' if change == 'reviews' else 'patch_candidates'
        path = tmp_path / 'data/raw' / f'730_{suffix}.json'
        path.write_text(path.read_text(encoding='utf-8')+'\n', encoding='utf-8')
    elif change == 'window': state['window_days'] = 3
    elif change == 'patch_id': state['patch_news_id'] = 'p2'
    elif change == 'goal': state['goal'] = '中文缓存评论推荐率'
    elif change == 'profile': state['investigation_profile'] = 'extended'
    elif change == 'query_args': args['limit'] = 6
    elif change == 'audit_deleted':
        from pathlib import Path
        Path(next(iter(first['artifacts'].values()))).unlink()
    elif change == 'audit_changed':
        from pathlib import Path
        Path(next(iter(first['artifacts'].values()))).write_text('{}')
    else: first['data']['stats']['count'] = 999
    second = step(registry, state, arguments=args)
    assert len(inline) == 2 and 'reused_from_step' not in second['data']['execution']


def test_resume_uses_observations_not_registry_memory(tmp_path, inline):
    state, registry = setup(tmp_path)
    step(registry, state)
    restored = json.loads(json.dumps(state))
    fresh_registry = SteamInvestigationTools(tmp_path).registry()
    row = step(fresh_registry, restored)
    assert len(inline) == 1 and row['data']['execution']['reused_from_step'] == 1


def test_old_readonly_receipt_and_old_observation_are_not_trusted(tmp_path, inline):
    state, registry = setup(tmp_path)
    step(registry, state)
    for receipt in (tmp_path / 'data/tool_receipts').rglob('*.json'):
        record = json.loads(receipt.read_text(encoding='utf-8'))
        record['data']['execution'].pop('reuse')
        receipt.write_text(json.dumps(record))
    state['observations'][0].pop('artifacts')
    state['step_count'] = 0
    step(registry, state)
    assert len(inline) == 2


def test_source_change_during_execution_disables_reuse(tmp_path, inline):
    state, registry = setup(tmp_path)
    tool = registry._tools['query_reviews']
    def changing(args, s):
        result = tool.handler(args, s)
        path = tmp_path / 'data/raw/730_reviews.json'
        path.write_text(path.read_text(encoding='utf-8')+'\n', encoding='utf-8')
        return result
    registry._tools['query_reviews'] = replace(tool, handler=changing)
    assert 'reuse' not in step(registry, state)['data']['execution']
    assert 'reuse' not in step(registry, state)['data']['execution']
    assert len(inline) == 2


def test_search_identity_distinguishes_query_model_and_index(tmp_path):
    state, registry = setup(tmp_path)
    tool = registry._tools['search_game_evidence']
    path = tmp_path / 'data/rag/730/index.sqlite'
    args = tool.args_model(appid=730, query='启动闪退', filters={'period':'all'})
    assert call_identity(tool, args, state) is None  # No index, no cache.
    path.parent.mkdir(parents=True)
    path.write_bytes(b'offline-unit-index')
    first = call_identity(tool, args, state)
    assert first != call_identity(tool, args.model_copy(update={'query':'主菜单崩溃'}), state)
    other_model = replace(tool, readonly_snapshot=ReadonlySnapshot(tmp_path, True, 'different-model'))
    assert first != call_identity(other_model, args, state)
    path.write_bytes(b'changed-index')
    assert first != call_identity(tool, args, state)


def test_refresh_build_verify_and_legacy_inspect_never_opt_in(tmp_path):
    state, registry = setup(tmp_path)
    for name in ('prepare_reviews','prepare_patches','build_game_index','verify_investigation'):
        assert registry._tools[name].readonly_snapshot is None
    extended = SteamInvestigationTools(tmp_path).registry('extended')
    assert extended._tools['inspect_data'].readonly_snapshot is None
    assert extended._tools['compare_versions'].readonly_snapshot is not None


def test_failed_results_never_reused_and_identical_failures_are_bounded(tmp_path, inline):
    state, registry = setup(tmp_path)
    args = {'appid':730,'review_id':'missing'}
    for _ in range(3):
        assert not step(registry, state, 'get_review', args)['ok']
    assert len(inline) == 3
    assert registry.repeated_readonly_call('get_review', args, state)
    # Even direct dispatch cannot execute a fourth time.
    result = registry.execute('get_review', args, state)
    assert result.data['error_code'] == 'repeated_readonly_call' and len(inline) == 3


def test_approval_precedes_reuse_and_refusal_records_no_execution(tmp_path, inline):
    state, registry = setup(tmp_path)
    step(registry, state)
    registry._tools['query_reviews'] = replace(registry._tools['query_reviews'], approval_policy=lambda a,s:'确认读取')
    state.update(next_arguments={'appid':730,'filters':{'period':'all'}})
    monkey = pytest.MonkeyPatch()
    monkey.setattr(runtime, 'interrupt', lambda *a,**kw:{'approved':False})
    try:
        row = step(registry, state)
    finally:
        monkey.undo()
    assert row['data']['approval_denied'] and len(inline) == 1
    assert registry.approval_request('query_reviews', state['next_arguments'], state)


def test_undeclared_tool_has_no_automatic_cache_or_loop_guard(tmp_path, inline):
    state, _ = setup(tmp_path)
    registry = ToolRegistry([AgentTool('query_reviews','generic',EmptyArgs,
        lambda a,s:ToolResult(ok=True,summary='test'))])
    for i in range(4):
        state.update(next_arguments={},next_action='query_reviews')
        state.update(InvestigationAgent(SequencePlanner([]),registry)._execute(state))
    assert len(inline) == 4 and not registry.repeated_readonly_call('query_reviews', {}, state)


def test_loop_guard_uses_shared_repairs_and_never_declares_success(tmp_path):
    state, registry = setup(tmp_path)
    decision = PlannerDecision(action='query_reviews',reason='repeat',arguments={'appid':730,'filters':{'period':'all'}})
    agent = InvestigationAgent(SequencePlanner([decision]*6),registry)
    result = agent.invoke(state)
    assert result['status'] == 'failed' and result['step_count'] == 3
    assert result['validation_retries'] == 2
    metrics = decision_metrics(result)
    assert metrics['handler_execution_count'] == 1 and metrics['readonly_reuse_count'] == 2
    assert metrics['semantic_success'] is None


def test_distinct_query_wording_not_equivalent_and_no_progress_is_only_a_signal(tmp_path, inline):
    state, registry = setup(tmp_path)
    snapshot = ReadonlySnapshot(tmp_path)
    from gamelens.rag.tools import SearchArgs
    registry = ToolRegistry([AgentTool('search_game_evidence','unit',SearchArgs,
        lambda a,s:ToolResult(ok=True,summary='no hit',data={'returned_count':0,'reviews':[],
            'filters':a.filters.model_dump(mode='json'),'review_coverage':{'status':'partial_snapshot'}}),
        readonly_snapshot=snapshot)])
    for q in ('启动闪退','主菜单崩溃','启动时无响应'):
        step(registry,state,'search_game_evidence',{'appid':730,'query':q,'filters':{'period':'all'}})
    assert len(inline) == 3
    assert state['loop_control']['warning'] and state['status'] == 'running'
    assert not registry.repeated_readonly_call('search_game_evidence',
        {'appid':730,'query':'加载卡死','filters':{'period':'all'}},state)


def test_feedback_roles_are_ids_not_confidence_and_keeps_missing_kind():
    result = ToolResult(ok=True,summary='selected',data={'query':'crash', 'returned_count':2,
        'candidate_summary':{'count':4}, 'evidence_coverage':{'missing_kinds':['announcement']},
        'screening':{'enabled':True,'verdicts':[
            {'kind':'review','source_id':str(i),'role':role,'reason':'untrusted '*200}
            for i,role in enumerate(('support','counterexample','partial','irrelevant'))]},
        'reviews':[{'review_id':'0'},{'review_id':'1'}], 'filters':{'period':'after'},
        'bounds':{'start_inclusive':100}, 'review_coverage':{'status':'partial_snapshot'}})
    feedback = observation_feedback(result, {}, {'review:0'})
    assert feedback['new_evidence_ids'] == ['review:1']
    assert feedback['role_ids']['counterexample'] == ['review:1']
    assert feedback['role_ids']['irrelevant'] == ['review:3']
    assert feedback['candidate_count'] == 4 and feedback['selected_count'] == 2
    assert feedback['missing_source_kinds'] == ['announcement']
    assert feedback['semantic_accuracy_verified'] is False
    assert 'untrusted' not in json.dumps(feedback)


@pytest.mark.parametrize('ok,data,status', [
    (False,{'timed_out':True},'tool_failed'),
    (True,{'review_coverage':{'status':'outside_observed_range'},'returned_count':0},'no_data_coverage'),
    (True,{'returned_count':0,'review_coverage':{'status':'partial_snapshot'}},'no_match'),
    (True,{'needs_year_clarification':True},'ambiguous_target'),
    (True,{'lookup_status':'not_found_in_cache'},'target_not_found')])
def test_feedback_statuses_not_conflated(ok,data,status):
    assert observation_feedback(ToolResult(ok=ok,summary='test',data=data),{},set())['status'] == status


def test_plan_omission_preserved_revision_change_and_sources_not_cleared(tmp_path):
    state, registry = setup(tmp_path)
    first = PlannerDecision(action='list_patches', reason='locate', arguments={'appid':730},
                            plan=['定位目标更新','统计缓存评论','核验来源'])
    second = PlannerDecision(action='query_reviews',reason='stats',arguments={'appid':730,'filters':{'period':'all'}})
    third = second.model_copy(update={'plan':['统计缓存评论','核验来源并说明限制']})
    agent = InvestigationAgent(SequencePlanner([first,second,third]), registry)
    state.update(agent._decide(state))
    original = copy.deepcopy(state['plan'])
    revision = state['plan_state']['revision']
    step(registry,state)
    observations = copy.deepcopy(state['observations'])
    state.update(agent._decide(state))
    assert state['plan'] == original and state['plan_state']['revision'] == revision
    assert state['observations'] == observations
    state.update(agent._decide(state))
    assert state['plan_state']['revision'] == revision+1
    assert state['observations'] == observations
    assert any(t.get('plan_changed') for t in state['decision_trace'])


def test_goal_change_clears_plan_and_old_loop_signal(tmp_path):
    state, registry = setup(tmp_path)
    state['plan'] = ['统计缓存评论']
    state['plan_state'] = project_plan(state,assess_investigation_progress(state))
    for _ in range(3): step(registry,state)
    assert state['loop_control']['warning']
    state['goal'] = '读取指定评论原文'
    agent = InvestigationAgent(SequencePlanner([PlannerDecision(action='get_review',reason='read',
        arguments={'appid':730,'review_id':'r1'})]),registry)
    result = agent._decide(state)
    assert result['plan'] == [] and not result['loop_control']['warning']


def test_plan_status_is_observable_projection_not_model_done(tmp_path):
    state, _ = setup(tmp_path)
    state['plan'] = ['统计缓存评论已完成','自由解释硬件原因','核验来源']
    p = project_plan(state,assess_investigation_progress(state))
    assert [i['status'] for i in p['items']] == ['pending','context_only','pending']
    assert p['semantic_completion_inferred'] is False


def test_old_state_and_legacy_decision_work_without_plan(tmp_path):
    state,registry = setup(tmp_path)
    decision = PlannerDecision.model_validate({'action':'query_reviews','reason':'query',
        'arguments':{'appid':730,'filters':{'period':'all'}}})
    result = InvestigationAgent(SequencePlanner([decision]),registry)._decide(state)
    assert result['plan'] == [] and result['next_action'] == 'query_reviews'


def test_context_is_bounded_and_not_raw_audit_dump(tmp_path, inline):
    state, registry = setup(tmp_path)
    row = step(registry,state)
    row['feedback']['new_evidence_ids'] = [str(i) for i in range(100)]
    row['feedback']['next_step_guidance'] = 'sk-secret-example user@site.com '+ 'x'*2000
    context = PlannerContextBuilder().build(state,registry.specifications())
    rendered = json.dumps(context,ensure_ascii=False)
    assert 'sk-secret-example' not in rendered and 'user@site.com' not in rendered
    recent = context['recent_observations'][0]
    assert 'artifacts' not in recent and 'execution' not in recent['data']
    assert 'information_fingerprint' not in recent['feedback']
    assert len(recent['feedback']['new_evidence_ids']) == 11
    assert context['plan_progress']['semantic_completion_inferred'] is False


def test_legacy_failed_observation_keeps_parameters_in_context(tmp_path):
    state,registry = setup(tmp_path)
    state['observations'] = [{'step':1,'tool':'get_review','ok':False,
        'arguments':{'appid':730,'review_id':'bad-id'},'data':{}}]
    context = PlannerContextBuilder().build(state,registry.specifications())
    assert context['recent_observations'][0]['arguments']['review_id'] == 'bad-id'


def test_reused_query_requires_fresh_verification(tmp_path, inline):
    from gamelens.core.answers import verified_sources_current
    state, registry = setup(tmp_path)
    step(registry,state)
    step(registry,state,'verify_investigation',{'appid':730})
    assert verified_sources_current(state)
    row = step(registry,state)
    assert row['data']['execution']['reused_from_step'] == 1
    assert not verified_sources_current(state)
    step(registry,state,'verify_investigation',{'appid':730})
    assert verified_sources_current(state)
    assert len(inline) == 3  # One query, two real verifications.


def test_changed_source_cannot_finish_even_if_plan_says_done(tmp_path, inline):
    state,registry = setup(tmp_path)
    step(registry,state)
    step(registry,state,'verify_investigation',{'appid':730})
    path = tmp_path/'data/raw/730_reviews.json'
    path.write_text(path.read_text(encoding='utf-8')+'\n',encoding='utf-8')
    state['plan'] = ['来源核验完成']
    progress = assess_investigation_progress(state)
    assert not progress['can_finalize']
    assert project_plan(state,progress)['items'][0]['status'] == 'pending'


def test_same_step_receipt_with_changed_source_executes_again(tmp_path, inline):
    state,registry = setup(tmp_path)
    args={'appid':730,'filters':{'period':'all'}}
    assert registry.execute('query_reviews',args,state).ok
    path=tmp_path/'data/raw/730_reviews.json'
    path.write_text(path.read_text(encoding='utf-8')+'\n',encoding='utf-8')
    result=registry.execute('query_reviews',args,state)
    assert result.ok and not result.data['execution'].get('from_receipt') and len(inline)==2


def test_same_step_receipt_preserves_identity_on_recovery(tmp_path, inline):
    state,registry=setup(tmp_path)
    args={'appid':730,'filters':{'period':'all'}}
    assert registry.execute('query_reviews',args,state).ok
    restored=SteamInvestigationTools(tmp_path).registry()
    result=restored.execute('query_reviews',args,state)
    assert result.data['execution']['from_receipt'] and len(inline)==1


def test_sqlite_checkpoint_reuses_prior_query_across_agent_restart(tmp_path):
    import sqlite3
    from langgraph.checkpoint.sqlite import SqliteSaver
    state,registry=setup(tmp_path)
    decision=PlannerDecision(action='query_reviews',reason='query',arguments={'appid':730,'filters':{'period':'all'}})
    db=tmp_path/'p1.sqlite'
    connection=sqlite3.connect(db,check_same_thread=False)
    agent=InvestigationAgent(SequencePlanner([decision]),registry,checkpointer=SqliteSaver(connection))
    stream=agent.stream(state)
    next(stream);next(stream);next(stream)
    stream.close()
    assert agent.snapshot(state['thread_id']).values['step_count']==1
    connection.close()
    connection=sqlite3.connect(db,check_same_thread=False)
    fresh=InvestigationAgent(SequencePlanner([decision]),SteamInvestigationTools(tmp_path).registry(),
                            checkpointer=SqliteSaver(connection))
    restored=fresh.snapshot(state['thread_id']).values
    row=step(fresh.registry,restored)
    connection.close()
    assert row['data']['execution']['reused_from_step']==1
    assert decision_metrics(restored)['handler_execution_count']==1


def test_api_optional_p1_fields_roundtrip_without_breaking_old_records():
    from gamelens.api.app import RunState,RunRequest,run_to_payload,run_from_payload
    state=RunState(run_id='p1',request=RunRequest(appid=730,goal='统计缓存'))
    state.plan_state={'revision':2,'items':[]}
    state.loop_control={'warning':True}
    payload=run_to_payload(state)
    assert run_from_payload(payload).plan_state==state.plan_state
    assert run_from_payload(payload).loop_control==state.loop_control
    payload.pop('plan_state');payload.pop('loop_control')
    restored=run_from_payload(payload)
    assert restored.plan_state=={} and restored.loop_control=={}


def test_cohort_feedback_includes_both_populations_and_coverage(tmp_path, inline):
    state,registry=setup(tmp_path)
    state['goal']='分别比较全部缓存中中文和英文评论的数量与推荐率'
    row=step(registry,state,'compare_cohorts',{'appid':730,
        'cohort_a':{'period':'all','language':'schinese'},
        'cohort_b':{'period':'all','language':'english'}})
    scope=row['feedback']['effective_scope']['cohorts']
    assert scope['cohort_a']['filters']['language']=='schinese'
    assert scope['cohort_b']['filters']['language']=='english'
    assert set(row['feedback']['coverage'])=={'cohort_a','cohort_b'}


def test_verify_feedback_reports_hash_check_not_semantic_success(tmp_path, inline):
    state,registry=setup(tmp_path)
    step(registry,state)
    row=step(registry,state,'verify_investigation',{'appid':730})
    verification=row['feedback']['verification']
    assert verification['artifact_count']==1 and verification['source_checks_passed']
    assert not verification['semantic_or_causal_truth_verified']
    assert row['feedback']['source_snapshot_id']


def test_inspect_feedback_distinguishes_missing_cached_sources(tmp_path, inline):
    state,registry=setup(tmp_path)
    (tmp_path/'data/raw/730_patch_candidates.json').unlink()
    row=step(registry,state,'inspect_data',{'appid':730})
    assert row['ok'] and row['feedback']['gaps']==['announcements_not_cached']
    assert row['feedback']['status']=='succeeded'


def test_feedback_driven_replan_preserves_sources_without_extra_planner_call(tmp_path, inline):
    state,registry=setup(tmp_path)
    state['goal']='调查启动闪退，官方公告有没有修复？'
    def search(args,s):
        return ToolResult(ok=True,summary='发现相反反馈',data={
            'source':{'path':'offline','sha256':'test'},
            'reviews':[{'review_id':'r4','text':'没有崩溃，启动正常。'}],
            'returned_count':1, 'filters':{'period':'all'},
            'reranking':{'mode':'off'},
            'hits':[{'kind':'review','source_id':'r4','evidence_role':'counterexample'}],
            'evidence_coverage':{'missing_kinds':['announcement']}})
    registry._tools['search_game_evidence']=replace(registry._tools['search_game_evidence'],handler=search)
    class FeedbackPlanner:
        calls=0
        def decide(self,s,tools):
            self.calls+=1
            if not s['observations']:
                return PlannerDecision(action='search_game_evidence',reason='查玩家反馈',
                    arguments={'appid':730,'query':'启动闪退','filters':{'period':'all'}},
                    plan=['调查玩家反馈','核对官方公告','核验来源'])
            feedback=s['observations'][-1]['feedback']
            assert feedback['role_ids']['counterexample']==['review:r4']
            assert 'counterexample_candidate_requires_review' in s['plan_state']['adjustment_signals']
            return PlannerDecision(action='get_review',reason='核对启动正常的相反材料',
                evidence_gap='玩家体验相反，须先核对原文再检查官方说明',
                arguments={'appid':730,'review_id':'r4'},
                plan=['核对启动正常的反例与原问题的关系','核对官方公告','核验来源'])
    planner=FeedbackPlanner()
    agent=InvestigationAgent(planner,registry)
    state.update(agent._decide(state))
    # No index/download needed: mock search observes a fixed offline result.
    step(registry,state,'search_game_evidence',state['next_arguments'])
    before=copy.deepcopy(state['observations'])
    state.update(agent._decide(state))
    assert planner.calls==2 and len(inline)==1
    assert state['next_action']=='get_review' and state['next_arguments']['review_id']=='r4'
    assert state['plan_state']['revision']==2 and state['observations']==before


def test_three_distinct_id_failures_are_not_same_scope_information():
    state=initial_state('unit','读取评论原文')
    rows=[]
    for i in range(3):
        feedback=observation_feedback(ToolResult(ok=False,summary='not found'),
            {'review_id':str(i)},set(),state)
        rows.append({'tool':'get_review','ok':False,'feedback':feedback})
    state['observations']=rows
    assert not loop_signals(state)['warning']
