"""Paraphrases, Cartesian filter combinations and full workflow contracts.

Mock source + mock planner only. These are execution regressions, not paid-model
semantic effectiveness measurements.
"""
import copy
import json
from datetime import datetime,timezone
from pathlib import Path

import pytest
import httpx

import gamelens.core.runtime as runtime
import gamelens.tools.steam_review_windows as api
from gamelens.core.query_contract import build_contract,contract_for,scope_errors
from gamelens.core.investigation_checks import assess_investigation_progress, statistics_alignment
from gamelens.core.answers import statistical_answer
from gamelens.core.runtime import InvestigationAgent,LLMPlanner,PlannerDecision
from gamelens.evaluation.benchmark import initial_state
from gamelens.storage.atomic_io import atomic_write_json
from gamelens.tools.investigation import ReviewInvestigation,ConditionQueryArgs,MetadataReviewFilter
from gamelens.tools.registry import SteamInvestigationTools
from gamelens.tools.review_coverage import ReviewCoverageTools,EnsureReviewCoverageArgs
from test_agent_runtime import SequencePlanner
from test_review_window_acquisition import raw,page,mock_client

ROOT=Path(__file__).resolve().parents[1]
REFERENCE='2026-10-11T12:00:00Z'


@pytest.mark.parametrize('month',['2026年6月份','2026年六月','2026-06','2026/06'])
@pytest.mark.parametrize('language,code',[('中文','schinese'),('英文','english'),('繁体中文','tchinese')])
@pytest.mark.parametrize('hours',['100h+','100 小时以上','至少100h','不少于100小时','100H+','>=100h','一百小时以上','100+小时'])
def test_cartesian_condition_paraphrases(month,language,code,hours,tmp_path):
    goal=f'调查{month}，游玩时长{hours}，{language}评论的好评率'
    state=initial_state('matrix',goal);state['request_started_at']=REFERENCE
    original={'appid':730,'filters':{}}
    registry=SteamInvestigationTools(tmp_path).registry()
    parsed=registry.validated_arguments('query_reviews',original,state)
    assert parsed and original=={'appid':730,'filters':{}}
    assert parsed['filters']['period']=='all' and parsed['filters']['language']==code
    assert parsed['filters']['min_playtime_hours']==100 and not parsed['filters']['min_playtime_exclusive']
    assert datetime.fromisoformat(parsed['filters']['start_at']).isoformat()=='2026-06-01T00:00:00+00:00'
    assert datetime.fromisoformat(parsed['filters']['end_at']).isoformat()=='2026-07-01T00:00:00+00:00'
    contract=contract_for(state)
    assert not contract['issues'] and not scope_errors(contract,parsed['filters'])


@pytest.mark.parametrize('words,field,value,strict',[
    ('超过100小时','min_playtime_hours',100,True),('大于100h','min_playtime_hours',100,True),
    ('至少100小时','min_playtime_hours',100,False),('不小于100小时','min_playtime_hours',100,False),
    ('不超过100小时','max_playtime_hours',100,False),('不大于100小时','max_playtime_hours',100,False),
    ('少于100小时','max_playtime_hours',100,True),('低于100h','max_playtime_hours',100,True),
    ('100h以内','max_playtime_hours',100,False),('0h+','min_playtime_hours',0,False)])
def test_hour_comparison_negation_and_inclusive_edges(words,field,value,strict):
    contract=build_contract('统计'+words+'中文评论的好评率',REFERENCE)
    assert not contract['issues'] and contract['filters'][field]==value
    exclusive=field.replace('_hours','_exclusive')
    assert contract['filters'].get(exclusive,False)==strict


@pytest.mark.parametrize('phrase,start,end',[
    ('2026年六月','2026-06-01','2026-07-01'),('2026年12月','2026-12-01','2027-01-01'),
    ('2024年2月','2024-02-01','2024-03-01'),('2026-06-01至2026-06-30','2026-06-01','2026-07-01'),
    ('2026年6月1日至2026年6月30日','2026-06-01','2026-07-01'),
    ('2026年6月1日至6月30日','2026-06-01','2026-07-01'),
    ('2026年6月15日','2026-06-15','2026-06-16'),('上个月','2026-09-01','2026-10-01'),
    ('昨天','2026-10-10','2026-10-11')])
def test_calendar_bounds_and_date_range_end_includes_the_last_day(phrase,start,end):
    contract=build_contract('统计'+phrase+'的中文评论数量',REFERENCE)
    assert not contract['issues']
    assert contract['calendar_window']['start_at'][:10]==start and contract['calendar_window']['end_at'][:10]==end


def test_this_month_cutoff_and_relative_clock_are_stable():
    state=initial_state('clock','统计本月中文评论好评率');state['request_started_at']=REFERENCE
    contract=contract_for(state);state['query_contract']=contract
    assert contract['calendar_window']['end_at']=='2026-10-11T12:00:00+00:00'
    assert contract_for(state)==contract
    seven=build_contract('过去七天评论数量',REFERENCE)
    assert seven['calendar_window']['start_at']=='2026-10-04T12:00:00+00:00'


@pytest.mark.parametrize('filters',[
    {'language':'english'},{'min_playtime_hours':200},{'min_playtime_exclusive':True},
    {'start_at':'2026-07-01','end_at':'2026-08-01'}, {'period':'after'}])
def test_contradictory_model_conditions_are_not_silently_repaired(filters,tmp_path):
    state=initial_state('wrong','统计2026年6月份，100h+，中文评论的好评率')
    registry=SteamInvestigationTools(tmp_path).registry()
    original={'filters':filters};copied=copy.deepcopy(original)
    errors=registry.argument_errors('query_reviews',original,state)
    assert errors and original==copied


@pytest.mark.parametrize('goal',['统计6月份中文评论好评率','统计2026年13月的好评率',
    '统计超过100小时且少于50小时的评论好评率'])
def test_ambiguous_or_invalid_constraints_ask_without_calls(goal,tmp_path):
    state=initial_state('clarify',goal);state['investigation_profile']='focused'
    result=InvestigationAgent(SequencePlanner([]),SteamInvestigationTools(tmp_path).registry()).invoke(state)
    assert result['status']=='completed' and result['investigation_progress']['needs_clarification']
    assert result['observations']==[] and '需要确认' in result['final_answer']


@pytest.fixture
def monthly(tmp_path):
    base=json.loads((ROOT/'evals/fixtures/730_reviews.json').read_text(encoding='utf-8'))[0]
    moment=int(datetime(2026,10,3,tzinfo=timezone.utc).timestamp())
    atomic_write_json(tmp_path/'data/raw/730_reviews.json',[{**base,'review_id':'outside-june',
        'created_at_timestamp':moment,'updated_at_timestamp':moment,'created_at':'2026-10-03T00:00:00Z','updated_at':'2026-10-03T00:00:00Z'}])
    state=initial_state('monthly','调查2026年6月份，游戏时长100h+，中文评论的好评率')
    state.update(investigation_profile='focused',rag_mode='off',patch_news_id=None,
        allow_review_acquisition=True,request_started_at=REFERENCE)
    monkey_root=ReviewInvestigation(tmp_path)
    return monkey_root,state


def fake_acquisition(monkeypatch, *, complete=False, error=None):
    calls=[];real=api.acquire_window
    def acquire(appid,start,end,language,**kwargs):
        calls.append((appid,start,end,language))
        bodies=[raw('min-equal',int(start)+100,language=language),raw('more',int(end)-10,positive=False,language=language),
            raw('less',int(end)-20,language=language)]
        for r,hours in zip(bodies,[100,101,99]):r['author']['playtime_at_review']=hours*60
        client,_=mock_client([error or page(bodies,total=3 if complete else 1000,positive=2 if complete else 900,cursor='*' if complete else 'next')])
        return real(appid,start,end,language,client=client,sleep=lambda _:None,**{**kwargs,'max_pages':1})
    monkeypatch.setattr(api,'acquire_window',acquire)
    monkeypatch.setattr(runtime,'run_isolated',lambda handler,args,state,timeout:('result',handler(args,state).model_dump(mode='json')))
    return calls


@pytest.mark.parametrize('complete',[False,True])
def test_full_query_workflow_separates_population_completeness_from_success(monthly,monkeypatch,complete):
    domain,state=monthly;calls=fake_acquisition(monkeypatch,complete=complete)
    class Client:
        calls=0
        def generate_json(self,messages):
            self.calls+=1
            assert self.calls==1,'Pure stats must not loop on a missing scope_limited bit'
            return {'action':'query_reviews','reason':'统计','arguments':{'filters':{}}}
    client=Client()
    result=InvestigationAgent(LLMPlanner(client),SteamInvestigationTools(domain.root).registry()).invoke(state)
    assert result['status']=='completed',result.get('error')
    assert [o['tool'] for o in result['observations']]==['inspect_data','ensure_review_coverage','query_reviews','verify_investigation']
    query=result['observations'][2]
    assert query['data']['stats']['count']==2 and query['data']['stats']['positive_rate']==.5
    assert query['data']['statistics_complete_for_scope']==complete
    assert result['scope_limited']==(not complete) and client.calls==1 and len(calls)==1
    assert '50.00%' in result['final_answer']
    if not complete:assert '不是整个月' in result['final_answer'] and result['investigation_progress']['outcome']=='partial_result'
    else:assert query['data']['statistics_basis']=='steam_window_complete_bodies'


def test_exact_hour_boundary_is_filtered_on_complete_body_population(monthly,monkeypatch):
    domain,state=monthly;state['goal']='统计2026年6月，超过100h的中文评论好评率'
    fake_acquisition(monkeypatch,complete=True)
    registry=SteamInvestigationTools(domain.root).registry()
    args=registry.validated_arguments('query_reviews',{'filters':{}},state)
    coverage=ReviewCoverageTools(domain.root);request=coverage.request_for('query_reviews',args,state)
    state['coverage_followup']={'windows':request['windows']}
    acquired=coverage.ensure_review_coverage(EnsureReviewCoverageArgs(**request),state)
    state['artifacts'].update(acquired.artifacts)
    query=domain.query_reviews(ConditionQueryArgs(**args),state)
    assert query.data['stats']['count']==1 and query.data['stats']['negative_rate']==1
    assert query.data['statistics_complete_for_scope'] and query.data['review_ids']==['more']


def test_bad_finish_complete_bit_cannot_fail_a_verified_sample(monthly,monkeypatch):
    domain,state=monthly;fake_acquisition(monkeypatch)
    decisions=[PlannerDecision(action='query_reviews',reason='统计',arguments={'filters':{}}),
        PlannerDecision(action='verify_investigation',reason='核验',arguments={'appid':730}),
        PlannerDecision(action='finish',reason='错误宣称完整',final_answer='全部玩家都满意，推荐率99%。',scope_limited=False)]
    result=InvestigationAgent(SequencePlanner(decisions),SteamInvestigationTools(domain.root).registry()).invoke(state)
    assert result['status']=='completed' and result['scope_limited']
    assert '99%' not in result['final_answer'] and '全部玩家都满意' not in result['final_answer']
    assert '50.00%' in result['final_answer']


def test_wrong_month_direct_handler_rejects_even_with_real_data(monthly):
    domain,state=monthly
    with pytest.raises(ValueError,match='日期区间'):
        domain.query_reviews(ConditionQueryArgs(appid=730,filters={'period':'all','language':'schinese','min_playtime_hours':100}),state)


def test_metric_paraphrase_and_negative_rate_are_not_lost(monthly,monkeypatch):
    domain,state=monthly;state['goal']='统计2026年6月份100h+中文评论的差评占比'
    fake_acquisition(monkeypatch)
    class Client:
        def generate_json(self,_):return {'action':'query_reviews','reason':'计算差评比例','arguments':{'filters':{}}}
    result=InvestigationAgent(LLMPlanner(Client()),SteamInvestigationTools(domain.root).registry()).invoke(state)
    assert result['status']=='completed' and '差评率50.00%' in result['final_answer']


def test_monthly_query_never_reuses_entire_cache_statistics(monthly,monkeypatch):
    domain,state=monthly;state['goal']='调查2026年10月份，游戏时长100h+，中文评论的好评率'
    registry=SteamInvestigationTools(domain.root).registry()
    args=registry.validated_arguments('query_reviews',{'filters':{'period':'all','language':'schinese','min_playtime_hours':100}},state)
    assert datetime.fromisoformat(args['filters']['start_at']).isoformat()=='2026-10-01T00:00:00+00:00'
    assert datetime.fromisoformat(args['filters']['end_at']).isoformat()==REFERENCE.replace('Z','+00:00')
    assert not statistics_alignment(state['goal'],{'filters':args['filters'],'bounds':{
        'start_inclusive':datetime.fromisoformat(args['filters']['start_at']).timestamp(),
        'end_exclusive':datetime.fromisoformat(args['filters']['end_at']).timestamp()}},state)


def test_corrupted_source_remains_a_failure_not_limited_success(monthly,monkeypatch):
    domain,state=monthly;fake_acquisition(monkeypatch)
    # A fake planner tries finish after a failed verification. Scope fallback
    # must never turn this integrity failure into a completed partial.
    class CorruptPlanner:
        stage=0
        def decide(self,current,tools):
            self.stage+=1
            if self.stage==1:return PlannerDecision(action='query_reviews',reason='统计',arguments={'filters':{}})
            if self.stage==2:
                path=Path(current['artifacts']['reviews']);atomic_write_json(path,[])
                return PlannerDecision(action='verify_investigation',reason='核验',arguments={'appid':730})
            pytest.fail('Failed source verification must stop before another model call')
    result=InvestigationAgent(CorruptPlanner(),SteamInvestigationTools(domain.root).registry()).invoke(state)
    assert result['status']=='failed' and not result.get('scope_limited')


@pytest.mark.parametrize('goal',['寻找无法启动的反馈','核对缓存中的评论原文','调查卡顿问题'])
def test_generic_semantic_investigation_does_not_force_default_time_as_user_scope(goal):
    contract=build_contract(goal,REFERENCE)
    assert 'period' not in contract['filters']
    assert not scope_errors(contract,{'period':'after'})


@pytest.mark.parametrize('phrase',['6月1日至6月30日','2026年12月30日至1月3日',
    '2026-06-01、2026-06-03、2026-06-05','2026年6月1日10:00至2026年6月30日12:00'])
def test_unclear_or_time_specific_ranges_never_silently_collapse(phrase):
    assert build_contract('统计'+phrase+'的评论数量',REFERENCE)['issues']


@pytest.mark.parametrize('goal,first,second',[
    ('比较超过100h和少于50h的好评率',{'max_playtime_hours':50,'max_playtime_exclusive':True},{'min_playtime_hours':100,'min_playtime_exclusive':True}),
    ('比较中文和英文评论的好评率',{'language':'english'},{'language':'schinese'})])
def test_reversed_comparison_groups_are_preserved_and_duplicate_groups_rejected(goal,first,second,tmp_path):
    state=initial_state('comparison',goal);state['request_started_at']=REFERENCE
    registry=SteamInvestigationTools(tmp_path).registry()
    arguments={'cohort_a':first,'cohort_b':second}
    effective=registry.validated_arguments('compare_cohorts',arguments,state)
    assert effective and not registry.argument_errors('compare_cohorts',arguments,state)
    for key,value in first.items():assert effective['cohort_a'][key]==value
    assert registry.argument_errors('compare_cohorts',{'cohort_a':first,'cohort_b':first},state)


@pytest.mark.parametrize('fields',[
    {'min_playtime_hours':'wrong'},{'start_at':'not-a-date'},
    {'max_playtime_hours':-1},{'min_playtime_hours':100,'max_playtime_hours':50},
    {'language':'schinese','unknown_condition':1}])
def test_invalid_structured_conditions_are_clarified_not_crashes(fields,tmp_path):
    goal='统计评论好评率，条件：'+json.dumps(fields)
    state=initial_state('invalid-json',goal);state['investigation_profile']='focused'
    result=InvestigationAgent(SequencePlanner([]),SteamInvestigationTools(tmp_path).registry()).invoke(state)
    assert result['status']=='completed' and result['investigation_progress']['needs_clarification']
    assert result['observations']==[] and 'JSON筛选' in result['final_answer']


@pytest.mark.parametrize('fields',[{'language':'english'},{'max_playtime_hours':200},{'is_positive':False}])
def test_unsolicited_narrower_denominator_cannot_be_used_for_statistics(fields,tmp_path):
    from gamelens.core.query_contract import statistical_population_errors
    contract=build_contract('统计100h+评论的差评比例',REFERENCE)
    assert statistical_population_errors(contract,{'min_playtime_hours':100,**fields})
    state=initial_state('narrower',contract['goal'])
    errors=SteamInvestigationTools(tmp_path).registry().argument_errors('query_reviews',{'filters':fields},state)
    assert errors and errors[0]['type']=='query_contract_mismatch'


def test_structured_max_only_population_does_not_fall_back_to_latest_patch(tmp_path):
    state=initial_state('max-json','统计评论数量，{"max_playtime_hours":50}')
    args=SteamInvestigationTools(tmp_path).registry().validated_arguments('query_reviews',{'filters':{}},state)
    assert args['filters']['period']=='all' and args['filters']['max_playtime_hours']==50


def test_full_population_and_subset_are_distinct_comparison_alternatives(tmp_path):
    from gamelens.core.query_contract import matching_alternatives
    goal='比较评论数量和好评率，{"period":"all"} 与 {"period":"all","is_positive":false}'
    state=initial_state('nested-populations',goal);contract=contract_for(state)
    assert matching_alternatives(contract,{'period':'all','is_positive':False})=={1}
    assert matching_alternatives(contract,{'period':'all','is_positive':None})=={0}
    registry=SteamInvestigationTools(tmp_path).registry()
    assert registry.argument_errors('compare_cohorts',{'cohort_a':{'period':'all','is_positive':False},'cohort_b':{'period':'all','is_positive':False}},state)


@pytest.mark.parametrize('basis,field',[('at_review','playtime_at_review'),('lifetime','playtime_forever')])
def test_complete_text_with_missing_playtime_is_not_complete_hour_population(tmp_path,monkeypatch,basis,field):
    record=raw('missing-hours',120);record['author'].pop(field)
    client,_=mock_client([page([record],total=1,positive=1,cursor='*')])
    row=api.acquire_window(730,100,200,client=client,sleep=lambda _:None)
    assert row['complete_bodies'] and row['stats_verified']
    assert not row['playtime_metadata_complete'][basis]
    coverage=ReviewCoverageTools(tmp_path)
    monkeypatch.setattr(coverage,'read',lambda *_:(row,{'path':'mock','sha256':'mock'}))
    result=coverage.window_statistics(730,{'start_inclusive':100,'end_exclusive':200},
        MetadataReviewFilter(period='all',min_playtime_hours=100,playtime_basis=basis))
    assert result is None


@pytest.mark.parametrize('reverse',[False,True])
@pytest.mark.parametrize('threshold,exclusive',[('超过100小时',True),('至少100小时',False)])
def test_comparison_keeps_each_groups_playtime_basis(tmp_path,reverse,threshold,exclusive):
    groups=[f'评价时长{threshold}',f'当前累计时长{threshold}']
    if reverse:groups.reverse()
    goal='比较冻结快照中中文、'+'与'.join(groups)+'两组玩家的评论数量和推荐率。'
    state=initial_state('basis-comparison',goal)
    registry=SteamInvestigationTools(tmp_path).registry()
    contract=contract_for(state)
    bases=['lifetime','at_review'] if reverse else ['at_review','lifetime']
    assert not contract['issues']
    assert [a['playtime_basis'] for a in contract['population_alternatives']]==bases
    assert 'playtime_basis' not in contract['filters']
    arguments={'cohort_a':{},'cohort_b':{}}
    effective=registry.validated_arguments('compare_cohorts',arguments,state)
    assert effective is not None,registry.argument_errors('compare_cohorts',arguments,state)
    for name,basis in zip(('cohort_a','cohort_b'),bases):
        assert effective[name]['playtime_basis']==basis
        assert effective[name]['min_playtime_hours']==100
        assert effective[name]['min_playtime_exclusive']==exclusive
        assert effective[name]['language']=='schinese'
    # Reversing explicit tool arguments must not change their identity.
    swapped={'cohort_a':effective['cohort_b'],'cohort_b':effective['cohort_a']}
    assert not registry.argument_errors('compare_cohorts',swapped,state)
    assert registry.argument_errors('compare_cohorts',{'cohort_a':effective['cohort_a'],'cohort_b':effective['cohort_a']},state)
    for field,value in [('min_playtime_hours',200),('min_playtime_exclusive',not exclusive),('is_positive',True),('language','english')]:
        wrong=copy.deepcopy(effective);wrong['cohort_a'][field]=value
        assert registry.argument_errors('compare_cohorts',wrong,state)


def test_comparison_unspecified_basis_stays_at_review():
    contract=build_contract('比较超过100h与少于50h的评论数量和推荐率',REFERENCE)
    assert not contract['issues']
    assert [a['playtime_basis'] for a in contract['population_alternatives']]==['at_review','at_review']


@pytest.mark.parametrize('goal',[
    '按当前累计时长比较超过100小时与少于50小时的评论数量和推荐率',
    '比较超过100小时与少于50小时的评论数量和推荐率，两组均按当前累计时长',
])
def test_shared_lifetime_basis_applies_to_both_comparison_groups(goal,tmp_path):
    state=initial_state('shared-basis',goal)
    contract=contract_for(state)
    assert [a['playtime_basis'] for a in contract['population_alternatives']]==['lifetime','lifetime']
    result=SteamInvestigationTools(tmp_path).registry().validated_arguments('compare_cohorts',{'cohort_a':{},'cohort_b':{}},state)
    assert result is not None
    assert result['cohort_a']['playtime_basis']==result['cohort_b']['playtime_basis']=='lifetime'


def test_explicit_group_basis_overrides_shared_prefix(tmp_path):
    state=initial_state('basis-precedence','按当前累计时长比较评价时长超过100小时与少于50小时的评论数量和推荐率')
    contract=contract_for(state)
    assert [a['playtime_basis'] for a in contract['population_alternatives']]==['at_review','lifetime']
    result=SteamInvestigationTools(tmp_path).registry().validated_arguments('compare_cohorts',{'cohort_a':{},'cohort_b':{}},state)
    assert result is not None
    assert result['cohort_a']['playtime_basis']=='at_review'
    assert result['cohort_b']['playtime_basis']=='lifetime'


@pytest.mark.parametrize('start,end,want_start,want_end',[
    ('2026-09-13T05:45:33+00:00','2026-10-13T07:36:07+00:00','2026-09-13T05:45:33+00:00','2026-10-13T07:36:07+00:00'),
    ('2026-09-13T13:45:33+08:00','2026-10-13T15:36:07+08:00','2026-09-13T05:45:33+00:00','2026-10-13T07:36:07+00:00'),
    ('2026-09-13T05:45:33Z','2026-10-13T07:36:07Z','2026-09-13T05:45:33+00:00','2026-10-13T07:36:07+00:00'),
    ('2026-09-13T05:45:33','2026-10-13T07:36:07','2026-09-13T05:45:33+00:00','2026-10-13T07:36:07+00:00'),
    ('2026-09-13T05:45:33.125Z','2026-10-13T07:36:07.500Z','2026-09-13T05:45:33.125000+00:00','2026-10-13T07:36:07.500000+00:00'),
])
def test_iso_time_window_accepts_exact_and_equivalent_tool_bounds(tmp_path,start,end,want_start,want_end):
    goal=f'统计已冻结评论在{start}至{end}（结束不含）内的数量与推荐率；空缓存窗口不代表Steam实际没有反馈。'
    state=initial_state('iso-window',goal);contract=contract_for(state)
    assert not contract['issues']
    assert contract['calendar_window']['start_at']==want_start
    assert contract['calendar_window']['end_at']==want_end
    registry=SteamInvestigationTools(tmp_path).registry()
    # Registry fills omitted endpoints only from an unambiguous request.
    effective=registry.validated_arguments('query_reviews',{'filters':{}},state)
    assert effective is not None
    assert not scope_errors(contract,{'period':'all','start_at':want_start,'end_at':want_end})
    for fields in ({'start_at':None,'end_at':want_end},
                   {'start_at':want_start,'end_at':None},
                   {'start_at':'2026-09-13T00:00:00Z','end_at':want_end},
                   {'start_at':want_start,'end_at':'2026-10-13T07:36:08Z'}):
        assert scope_errors(contract,{'period':'all',**fields})
    wrong={'filters':{'period':'all','start_at':'2026-09-13T00:00:00Z','end_at':want_end}}
    assert registry.argument_errors('query_reviews',wrong,state)
    equivalent={'filters':{'period':'all','start_at':want_start,'end_at':want_end}}
    assert not registry.argument_errors('query_reviews',equivalent,state)


@pytest.mark.parametrize('phrase',[
    '2026-09-13T05:45:33Z',
    '2026-09-13T05:45:33Z至2026-10-13',
    '2026-10-13T07:36:07Z至2026-09-13T05:45:33Z',
    '2026-09-13T05:45:33Z至2026-09-13T05:45:33Z',
    '2026-09-13T05:45:33Z、2026-10-13T07:36:07Z',
    '2026-09-13T05:45:33+25:00至2026-10-13T07:36:07Z',
    '2026-09-13T05:45:33+08:99至2026-10-13T07:36:07Z',
    '2026-09-13T至2026-10-13T',
    '2026-09-13t至2026-10-13t',
    '2026-09-13T05:45:33Z至2026-10-13T07:36:07Z至2026-10-14T07:36:07Z',
])
def test_ambiguous_or_invalid_iso_windows_never_become_all_time(phrase,tmp_path):
    state=initial_state('bad-iso','统计'+phrase+'的评论数量')
    assert contract_for(state)['issues']
    assert SteamInvestigationTools(tmp_path).registry().argument_errors('query_reviews',{'filters':{'period':'all'}},state)


def test_legacy_cached_contract_is_reinterpreted_without_mutating_checkpoint():
    state=initial_state('old-contract','统计2026-09-13T05:45:33Z至2026-10-13T07:36:07Z的评论数量')
    from gamelens.core.local_time import timezone_signature
    state['query_contract']={'version':1,'goal':state['goal'],'time_zone':timezone_signature(),
        'issues':['旧版不支持精确时间'],'filters':{},'calendar_window':{}}
    before=copy.deepcopy(state)
    assert not contract_for(state)['issues']
    assert state==before


def test_iso_query_filters_real_records_with_half_open_boundaries(tmp_path):
    base=json.loads((ROOT/'evals/fixtures/730_reviews.json').read_text(encoding='utf-8'))[0]
    bodies=[]
    for rid,instant,positive in [
        ('too-early','2026-09-13T05:45:32+00:00',True),
        ('at-start','2026-09-13T05:45:33+00:00',True),
        ('inside','2026-10-13T07:36:06+00:00',False),
        ('at-end','2026-10-13T07:36:07+00:00',True)]:
        stamp=int(datetime.fromisoformat(instant).timestamp())
        bodies.append({**base,'review_id':rid,'is_positive':positive,
            'created_at':instant,'updated_at':instant,
            'created_at_timestamp':stamp,'updated_at_timestamp':stamp})
    atomic_write_json(tmp_path/'data/raw/730_reviews.json',bodies)
    state=initial_state('iso-real','统计冻结评论在2026-09-13T05:45:33Z至2026-10-13T07:36:07Z的数量与推荐率')
    state['investigation_profile']='focused'
    registry=SteamInvestigationTools(tmp_path).registry()
    effective=registry.validated_arguments('query_reviews',{'filters':{}},state)
    result=ReviewInvestigation(tmp_path).query_reviews(ConditionQueryArgs(**effective),state)
    assert result.ok and result.data['stats']['count']==2
    assert result.data['stats']['positive_rate']==.5
    assert set(result.data['review_ids'])=={'at-start','inside'}


@pytest.mark.system_local_time
def test_naive_iso_window_uses_system_local_offset(monkeypatch):
    from datetime import timedelta
    from gamelens.core import local_time
    zone=timezone(timedelta(hours=8))
    monkeypatch.setattr(local_time,'_system_localize',lambda m:
        m.replace(tzinfo=zone) if m.tzinfo is None else m.astimezone(zone))
    contract=build_contract('统计2026-09-13T13:45:33至2026-10-13T15:36:07的评论数量',REFERENCE)
    assert not contract['issues']
    assert contract['calendar_window']['start_at']=='2026-09-13T13:45:33+08:00'
    assert not scope_errors(contract,{'period':'all','start_at':'2026-09-13T05:45:33Z','end_at':'2026-10-13T07:36:07Z'})
