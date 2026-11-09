"""Regression for 更新前后 misread as 更新前, without source/model calls."""
import copy
import json
from pathlib import Path

import pytest

import gamelens.core.runtime as runtime
from gamelens.core.answers import simple_statistics_goal
from gamelens.core.investigation_checks import assess_investigation_progress, statistics_alignment
from gamelens.core.runtime import InvestigationAgent, LLMPlanner, PlannerDecision
from gamelens.core.synthesis import evidence_packet
from gamelens.memory.context import PlannerContextBuilder
from gamelens.storage.atomic_io import atomic_write_json
from gamelens.tools.argument_normalization import normalize_search_arguments
from gamelens.tools.investigation import ConditionQueryArgs, ConditionCohortsArgs, ListPatchesArgs
from gamelens.tools.registry import SteamInvestigationTools
from gamelens.tools.time_scope import requested_directions, requested_windows, located_dated_patch
from test_agent_runtime import SequencePlanner
from test_historical_time_scope import dated, stamp

GOAL='调查2026-07-10更新前后评论的好评率变化，并用原文解释主要反馈与相反体验。'


@pytest.mark.parametrize('goal,directions',[
    ('调查2026-07-15更新前后评论',{'before','after'}),
    ('对比版本前与后',{'before','after'}),
    ('补丁后与前的反馈',{'before','after'}),
    ('更新之前与更新之后',{'before','after'}),
    ('比较p1前后好评率',{'before','after'}),
    ('这次更新的前后评论变化',{'before','after'}),
    ('更新之前之后的玩家反馈',{'before','after'}),
    ('更新前一天和更新后五天',{'before','after'}),
    ('2026-07-15更新后玩家反馈',{'after'}),
    ('7月15日更新前的评论',{'before'})])
def test_shared_direction_parser(goal,directions):
    assert requested_directions(goal)==directions


@pytest.mark.parametrize('text',[ '更新前后各7天','版本前后七天','补丁前后分别7天'])
def test_symmetric_duration_includes_both_windows(text):
    assert set(requested_windows(text))=={('前',7),('后',7)}


def setup(dated):
    domain,state,announcement=dated
    state.update(goal=GOAL,investigation_profile='focused',rag_mode='off',max_steps=12)
    result=domain.list_patches(ListPatchesArgs(appid=730,published_on='2026-07-10'),state)
    state['observations']=[{'step':1,'tool':'list_patches','ok':True,
        'arguments':{'appid':730,'published_on':'2026-07-10'},'data':result.data}]
    state['artifacts'].update(result.artifacts);state['step_count']=1
    return domain,state,announcement


def inline(monkeypatch):
    monkeypatch.setattr(runtime,'run_isolated',lambda handler,args,state,timeout:
        ('result',handler(args,state).model_dump(mode='json')))


def test_real_before_after_cohorts_pass_and_use_declared_symmetric_window(dated):
    domain,state,_=setup(dated)
    result=domain.compare_cohorts(ConditionCohortsArgs(appid=730,patch_news_id='old',window_days=7,
        cohort_a={'period':'before'},cohort_b={'period':'after'}),state)
    assert result.ok and result.data['patch_news_id']=='old'
    assert result.data['cohort_a']['bounds']=={'start_inclusive':stamp('2026-07-03'),'end_exclusive':stamp('2026-07-10')}
    assert result.data['cohort_b']['bounds']=={'start_inclusive':stamp('2026-07-10'),'end_exclusive':stamp('2026-07-17')}
    for period in ('before','after'):
        query=domain.query_reviews(ConditionQueryArgs(appid=730,patch_news_id='old',filters={'period':period}),state)
        assert query.ok and not statistics_alignment(state['goal'],query.data,state)
    assert not simple_statistics_goal('调查更新前后评论的好评率变化')


def test_two_after_populations_do_not_replace_before_after(dated):
    domain,state,_=setup(dated)
    with pytest.raises(ValueError,match='两组不同窗口'):
        domain.compare_cohorts(ConditionCohortsArgs(appid=730,patch_news_id='old',
            cohort_a={'period':'after'},cohort_b={'period':'after'}),state)


def test_unambiguous_current_locator_binds_id_not_session_default(dated,monkeypatch):
    inline(monkeypatch)
    domain,state,_=setup(dated)
    state['patch_news_id']='new-8'
    registry=SteamInvestigationTools(domain.root).registry()
    args={'cohort_a':{'period':'before'},'cohort_b':{'period':'after'}}
    original=copy.deepcopy(args)
    result=registry.execute('compare_cohorts',args,state)
    assert result.ok and result.data['patch_news_id']=='old'
    assert any(c['rule']=='current_unique_date_locator' for c in result.data['execution']['argument_adjustments'])
    assert args==original and state['patch_news_id']=='new-8'
    context=PlannerContextBuilder().build(state,registry.specifications())
    assert context['identity']['resolved_dated_patch']['news_id']=='old'
    packet,_=evidence_packet(state,registry.specifications(),PlannerContextBuilder())
    assert packet['target_patch_ids']==['old'] and packet['target_updates'][0]['news_id']=='old'


def test_explicit_wrong_id_is_rejected_before_handler(dated,monkeypatch):
    domain,state,_=setup(dated)
    monkeypatch.setattr(runtime,'run_isolated',lambda *_:pytest.fail('Wrong patch must not execute'))
    result=SteamInvestigationTools(domain.root).registry().execute('query_reviews',
        {'patch_news_id':'new-8','filters':{'period':'after'}},state)
    assert not result.ok and result.data['argument_errors'][0]['type']=='dated_target_mismatch'
    assert 'old' in result.summary


@pytest.mark.parametrize('change',['no_lookup','goal_changed','stale_source','same_day','filtered_lookup'])
def test_unsafe_locator_cannot_fill_missing_id(dated,change):
    domain,state,announcement=setup(dated)
    if change=='no_lookup':state['observations']=[]
    if change=='goal_changed':state['goal']=GOAL.replace('07-10','07-11')
    if change=='stale_source':
        # Mutate the authoritative file actually hashed by list_patches.  The
        # legacy patch-candidate cache is no longer trusted for date binding.
        path=Path(state['observations'][0]['data']['source']['path'])
        atomic_write_json(path,json.loads(path.read_text(encoding='utf-8'))+[announcement('other','2026-06-01')])
    if change=='same_day':state['observations'][0]['data']['matched_count']=2
    if change=='filtered_lookup':state['observations'][0]['arguments']['news_id']='old'
    assert located_dated_patch(state) is None
    errors=SteamInvestigationTools(domain.root).registry().argument_errors('query_reviews',{'filters':{'period':'after'}},state)
    assert errors[0]['type']=='dated_target_missing'


def test_search_never_guesses_one_side_of_a_comparison():
    args={'patch_news_id':'old','query':'玩家反馈'}
    normalized,changes=normalize_search_arguments(args,{'goal':GOAL})
    assert normalized==args and not changes


def test_comparison_requires_both_sides_not_two_before_queries(dated):
    domain,state,_=setup(dated)
    for step,days in ((2,7),(3,3)):
        result=domain.query_reviews(ConditionQueryArgs(appid=730,patch_news_id='old',window_days=days,filters={'period':'before'}),state)
        state['observations'].append({'step':step,'tool':'query_reviews','ok':True,'data':result.data})
    check=next(c for c in assess_investigation_progress(state)['checks'] if c['requirement']=='statistics')
    assert check['status']!='satisfied'


def test_missing_history_closes_after_exact_queries_without_repeated_search(dated,monkeypatch):
    inline(monkeypatch)
    domain,state,_=setup(dated)
    state.update(observations=[],artifacts={},step_count=0,patch_news_id='new-8')
    class Client:
        calls=0
        def generate_json(self,messages):
            self.calls+=1
            if self.calls==1:return {'action':'list_patches','reason':'定位','arguments':{'published_on':'2026-07-10'}}
            assert self.calls==2,'No model summary or repeated retrieval for unavailable historical bodies'
            assert 'resolved_dated_patch' in messages[1]['content']
            return {'action':'compare_cohorts','reason':'前后对照','arguments':{
                'cohort_a':{'period':'before'},'cohort_b':{'period':'after'}}}
    client=Client()
    result=InvestigationAgent(LLMPlanner(client),SteamInvestigationTools(domain.root).registry()).invoke(state)
    assert result['status']=='completed',result.get('error')
    assert [o['tool'] for o in result['observations']]==['inspect_data','list_patches','compare_cohorts','verify_investigation']
    assert result['scope_limited'] and client.calls==2 and result['step_count']==4
    assert '没有可用缓存评论' in result['final_answer'] and '历史评论' in result['final_answer']
    assert len(result['final_answer'])<=300 and '调查范围未完整满足' not in result['final_answer']
    assert not result['final_citations'] and result['investigation_progress']['outcome']=='partial_result'
    assert result['observations'][2]['data']['positive_rate_delta_pp'] is None


def test_available_history_computes_rates_and_retains_opposite_originals(dated,monkeypatch):
    inline(monkeypatch)
    domain,state,_=setup(dated)
    path=domain.root/'data/raw/730_reviews.json'
    base=json.loads(path.read_text(encoding='utf-8'))[0]
    records=[]
    for rid,day,positive,text in [('b1','2026-07-08',True,'更新前运行正常'),('b2','2026-07-09',False,'更新前卡顿'),
        ('a1','2026-07-11',True,'更新后流畅'),('a2','2026-07-12',True,'更新后体验不错')]:
        records.append({**base,'review_id':rid,'created_at':day+'T12:00:00Z','created_at_timestamp':stamp(day),
            'updated_at':day+'T12:00:00Z','updated_at_timestamp':stamp(day),'is_positive':positive,'text':text})
    atomic_write_json(path,records)
    state.update(observations=[],artifacts={},step_count=0)
    planner=SequencePlanner([
        PlannerDecision(action='list_patches',reason='定位',arguments={'published_on':'2026-07-10'}),
        PlannerDecision(action='compare_cohorts',reason='两侧统计',arguments={'cohort_a':{'period':'before'},'cohort_b':{'period':'after'}}),
        PlannerDecision(action='query_reviews',reason='更新前原文',arguments={'filters':{'period':'before'}}),
        PlannerDecision(action='query_reviews',reason='更新后原文',arguments={'filters':{'period':'after'}}),
        PlannerDecision(action='verify_investigation',reason='核验',arguments={}),
        PlannerDecision(action='finish',reason='固定离线测试答复，不评价模型语义质量',scope_limited=True,
            final_answer='**缓存对照**\n\n- 两侧均有原文，更新后也有流畅体验。\n\n只描述缓存，不代表全体玩家。',
            claims=[{'label':'缓存好评率变化','source_step':2,'fact_path':['positive_rate_delta_pp'],'value':50}],
            citations=[{'source_step':3,'review_id':'b2','quote':'更新前卡顿'},{'source_step':4,'review_id':'a1','quote':'更新后流畅'}])])
    result=InvestigationAgent(planner,SteamInvestigationTools(domain.root).registry()).invoke(state)
    assert result['status']=='completed',result.get('error')
    data=result['observations'][1]['data']
    assert data['cohort_a']['stats']['positive_rate']==.5 and data['cohort_b']['stats']['positive_rate']==1
    assert data['positive_rate_delta_pp']==50 and len(result['final_citations'])==2
    assert {'counterexample_check','review_evidence','statistics'} <= set(result['investigation_progress']['required_evidence'])
