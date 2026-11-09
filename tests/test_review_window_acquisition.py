"""Historical coverage regression: all HTTP and planner responses are offline."""
import copy
import hashlib
import json
import time
from datetime import datetime, timezone

import httpx
import pytest

import gamelens.core.runtime as runtime
import gamelens.tools.steam_review_windows as api
from gamelens.analysis.report_prose import readable_statistics
from gamelens.core.investigation_checks import assess_investigation_progress
from gamelens.core.runtime import InvestigationAgent, PlannerDecision
from gamelens.core.synthesis import evidence_packet
from gamelens.memory.context import PlannerContextBuilder
from gamelens.storage.atomic_io import atomic_write_json
from gamelens.tools.investigation import ConditionCohortsArgs, ConditionQueryArgs, ListPatchesArgs, MetadataReviewFilter
from gamelens.tools.query_reuse import ReadonlySnapshot
from gamelens.tools.registry import SteamInvestigationTools
from gamelens.tools.review_coverage import ReviewCoverageTools, EnsureReviewCoverageArgs, window_key, TTL
from test_agent_runtime import SequencePlanner
from test_historical_time_scope import dated, stamp

GOAL='调查2026-07-10更新前后评论的好评率变化，并用原文解释主要反馈与相反体验。'


def raw(rid, created, *, positive=True, language='english'):
    return {'recommendationid':rid,'author':{'steamid':'PRIVATE_ID','playtime_forever':1200,'playtime_at_review':600},
        'language':language,'review':'反馈 '+str(rid),'timestamp_created':created,'timestamp_updated':created,
        'voted_up':positive,'votes_up':3,'steam_purchase':True,'received_for_free':False,
        'written_during_early_access':False}


def page(reviews, *, total=10, positive=8, cursor='next'):
    return {'response':{'query_summary':{'total_positive':positive,'total_negative':total-positive,'total_reviews':total},
        'total_matching':total,'reviews':reviews,'cursor':cursor}}


def mock_client(responses):
    calls=[]
    def handler(request):
        calls.append(json.loads(request.url.params['input_json']))
        response=responses[len(calls)-1]
        if isinstance(response,Exception):raise response
        return httpx.Response(200,json=response)
    return httpx.Client(transport=httpx.MockTransport(handler)),calls


def test_exact_public_query_half_open_and_count_body_separation():
    client,calls=mock_client([page([raw('a',120)],cursor='page2'),page([raw('b',110)])])
    result=api.acquire_window(730,100,200,target_count=2,client=client,sleep=lambda _:None)
    assert result['stats_verified'] and result['stats']['count']==10 and result['body_count']==2
    assert result['stop_reason']=='sample_budget' and not result['complete_bodies']
    assert calls[0]=={'appid':730,'filter':1,'languages':['all'],'review_type':0,'purchase_type':1,
        'filter_offtopic_activity':False,'date_range_start':100,'date_range_end':199,'cursor':'*','num_per_page':2}
    assert calls[1]['cursor']=='page2'
    assert 'PRIVATE_ID' not in json.dumps(result) and 'steamid' not in json.dumps(result)


@pytest.mark.parametrize('reviews,total,positive', [([],0,0),([raw('a',120)],1,1)])
def test_verified_empty_or_exhausted_source(reviews,total,positive):
    client,_=mock_client([page(reviews,total=total,positive=positive,cursor='*')])
    result=api.acquire_window(730,100,200,client=client,sleep=lambda _:None)
    assert result['stats_verified'] and result['complete_bodies']
    assert result['stats']['positive_rate']==(1 if total else None)


@pytest.mark.parametrize('response', [{},{'response':{}},{'response':{'query_summary':[]}},
    {'response':{'query_summary':{}}},
    page([raw('a',120)],total=0,positive=0)])
def test_invalid_envelope_is_not_real_zero(response):
    client,_=mock_client([response])
    result=api.acquire_window(730,100,200,client=client,sleep=lambda _:None)
    assert not result['stats_verified'] and result['stats'] is None and result['error_code']=='invalid_review_response'


@pytest.mark.parametrize('change',[{'timestamp_created':200},{'timestamp_created':99},{'language':'schinese'},
    {'recommendationid':None},{'author':None}])
def test_server_filter_or_record_error_cannot_support_counts(change):
    client,_=mock_client([page([{**raw('a',120),**change}],total=1,positive=1)])
    result=api.acquire_window(730,100,200,'english',client=client,sleep=lambda _:None)
    assert not result['stats_verified'] and result['error_code']=='invalid_review_record_or_range'


@pytest.mark.parametrize('error,expected',[
    (httpx.ReadTimeout('timeout'),'request_timeout'),
    (httpx.ConnectError('offline'),'transport_error'),
    (httpx.HTTPStatusError('limited',request=httpx.Request('GET',api.ENDPOINT),
        response=httpx.Response(429,request=httpx.Request('GET',api.ENDPOINT))),'http_429')])
def test_partial_body_preserved_when_next_page_fails(error,expected):
    client,calls=mock_client([page([raw('a',120)]),error])
    result=api.acquire_window(730,100,200,target_count=5,client=client,sleep=lambda _:None)
    assert result['body_count']==1 and result['stats_verified'] and result['error_code']==expected
    assert len(calls)==2 and not result['complete_bodies']


@pytest.mark.parametrize('second,expected',[
    (page([raw('a',120)]),'repeated_page'),
    (page([raw('b',110)],total=11,positive=8),'counts_changed_during_fetch'),
    (page([raw('b',110)],cursor='next'),'cursor_missing_or_stalled')])
def test_pagination_stops_safely(second,expected):
    client,calls=mock_client([page([raw('a',120)]),second])
    result=api.acquire_window(730,100,200,target_count=20,client=client,sleep=lambda _:None)
    assert result['error_code']==expected and len(calls)==2
    if expected=='counts_changed_during_fetch':assert not result['stats_verified']


def test_budget_no_http_after_deadline_and_page_cap():
    client,calls=mock_client([])
    result=api.acquire_window(730,100,200,client=client,deadline=0,clock=lambda:1)
    assert not calls and result['stop_reason']=='time_budget'
    client,calls=mock_client([page([raw('a',120)])])
    result=api.acquire_window(730,100,200,max_pages=1,client=client)
    assert len(calls)==1 and result['stop_reason']=='page_budget'


def test_reported_population_with_missing_bodies_has_explicit_error():
    client,_=mock_client([page([],total=10,positive=8)])
    result=api.acquire_window(730,100,200,client=client)
    assert result['stats']['count']==10 and result['stats_verified']
    assert result['body_count']==0 and result['error_code']=='body_missing' and not result['complete_bodies']


def test_server_cannot_bypass_requested_body_cap():
    client,_=mock_client([page([raw('a',120),raw('b',121)])])
    result=api.acquire_window(730,100,200,target_count=1,client=client)
    assert result['body_count']==0 and result['error_code']=='invalid_review_response'


def setup(dated):
    domain,state,_=dated
    state.update(goal=GOAL,allow_review_acquisition=True,rag_mode='off')
    lookup=domain.list_patches(ListPatchesArgs(appid=730,published_on='2026-07-10'),state)
    state['observations']=[{'step':1,'tool':'list_patches','arguments':{'appid':730,'published_on':'2026-07-10'},**lookup.model_dump(mode='json')}]
    state['step_count']=1;state['artifacts'].update(lookup.artifacts)
    arguments={'appid':730,'patch_news_id':'old','cohort_a':{'period':'before'},'cohort_b':{'period':'after'}}
    coverage=ReviewCoverageTools(domain.root)
    request=coverage.request_for('compare_cohorts',arguments,state)
    state['coverage_followup']={'action':'compare_cohorts','arguments':arguments,'windows':request['windows']}
    return domain,state,coverage,request


def inject_api(monkeypatch, *, error=None):
    calls=[]
    real=api.acquire_window
    def acquire(appid,start,end,language,**kwargs):
        calls.append((appid,start,end,language))
        response=page([raw('id-'+str(int(start)),int(end)-1,language='english' if language=='all' else language)],total=10,positive=8)
        client,_=mock_client([error or response])
        return real(appid,start,end,language,client=client,sleep=lambda _:None,**{**kwargs,'max_pages':1})
    monkeypatch.setattr(api,'acquire_window',acquire)
    return calls


def add_observation(state,result,tool='ensure_review_coverage'):
    state['step_count']+=1
    state['observations'].append({'step':state['step_count'],'tool':tool,**result.model_dump(mode='json')})
    state['artifacts'].update(result.artifacts)


def test_union_acquisition_merge_is_immutable_and_reused(dated,monkeypatch):
    domain,state,coverage,request=setup(dated);calls=inject_api(monkeypatch)
    base=domain.root/'data/raw/730_reviews.json';original=base.read_bytes()
    request={**request,'target_count_per_window':1}
    result=coverage.ensure_review_coverage(EnsureReviewCoverageArgs(**request),state)
    assert len(calls)==2 and result.data['added_reviews']==2 and base.read_bytes()==original
    assert len(json.loads(open(result.artifacts['reviews'],encoding='utf-8').read()))==3
    for reference in [result.data['source'],*result.data['source']['reference_files']]:
        assert hashlib.sha256(open(reference['path'],'rb').read()).hexdigest()==reference['sha256']
    again=coverage.ensure_review_coverage(EnsureReviewCoverageArgs(**request),state)
    assert len(calls)==2 and all(w['reused'] for w in again.data['windows'])
    assert result.artifacts['reviews']==again.artifacts['reviews']
    add_observation(state,result)
    followup=coverage.request_for('compare_cohorts',state['coverage_followup']['arguments'],state)
    assert followup['target_count_per_window']==1000


def test_window_cache_is_reacquired_when_requested_body_target_increases(dated,monkeypatch):
    domain,state,coverage,request=setup(dated)
    calls=[]
    def acquire(appid,start,end,language,**kwargs):
        target=kwargs['target_count'];calls.append(target)
        comments=[]
        for index in range(target):
            created=min(int(end)-1,int(start)+index)
            comments.append(api.normalize_review(raw(f'{int(start)}-{index}',created,
                language='english' if language=='all' else language)).model_dump(mode='json'))
        return {'query':{'appid':appid,'filter':1,'languages':[language],'review_type':0,'purchase_type':1,
                'filter_offtopic_activity':False,'date_range_start':int(start),'date_range_end':int(end)-1},
            'comments':comments,'reported_total':10,'stats':{'count':10,'positive_count':8,'negative_count':2,
                'positive_rate':.8,'negative_rate':.2},'stats_verified':True,'body_count':len(comments),
            'pages_read':1,'complete_bodies':False,'playtime_metadata_complete':{'at_review':True,'lifetime':True},
            'error_code':None,'stop_reason':'sample_budget','sampling':'recent_first_within_exact_window',
            'invalid_records':0,'endpoint':api.ENDPOINT,'historical_text_available':False}
    monkeypatch.setattr(api,'acquire_window',acquire)
    first=coverage.ensure_review_coverage(EnsureReviewCoverageArgs(**{**request,'target_count_per_window':1}),state)
    second=coverage.ensure_review_coverage(EnsureReviewCoverageArgs(**{**request,'target_count_per_window':2}),state)
    assert calls==[1,1,2,2]
    assert all(not row['reused'] and row['body_count']==2 and row['cache_adequate'] for row in second.data['windows'])
    assert all(row['refresh_attempted'] for row in second.data['windows'])


def test_errored_larger_window_candidate_never_replaces_clean_cache(dated,monkeypatch):
    domain,state,coverage,request=setup(dated)
    phase={'error':None}
    def acquire(appid,start,end,language,**kwargs):
        target=kwargs['target_count'];comments=[]
        for index in range(target):
            comments.append(api.normalize_review(raw(f'{int(start)}-{index}',min(int(end)-1,int(start)+index),
                language='english' if language=='all' else language)).model_dump(mode='json'))
        return {'query':{'appid':appid,'filter':1,'languages':[language],'review_type':0,'purchase_type':1,
                'filter_offtopic_activity':False,'date_range_start':int(start),'date_range_end':int(end)-1},
            'comments':comments,'reported_total':target+100,'stats':None,'stats_verified':False,
            'body_count':len(comments),'pages_read':1,'complete_bodies':False,
            'playtime_metadata_complete':{'at_review':True,'lifetime':True},
            'error_code':phase['error'],'stop_reason':phase['error'] or 'sample_budget',
            'sampling':'recent_first_within_exact_window','invalid_records':0,
            'endpoint':api.ENDPOINT,'historical_text_available':False}
    monkeypatch.setattr(api,'acquire_window',acquire)
    first=coverage.ensure_review_coverage(EnsureReviewCoverageArgs(**{
        **request,'target_count_per_window':200}),state)
    phase['error']='counts_changed_during_fetch'
    second=coverage.ensure_review_coverage(EnsureReviewCoverageArgs(**{
        **request,'target_count_per_window':300}),state)
    assert all(row['body_count']==200 and row['retained_previous_snapshot'] for row in second.data['windows'])
    assert set(second.data['error_codes'])=={'counts_changed_during_fetch'}
    assert all(row['cache_adequate'] is False for row in second.data['windows'])


def test_platform_window_counts_and_source_refs_flow_to_statistics_and_packet(dated,monkeypatch):
    domain,state,coverage,request=setup(dated);inject_api(monkeypatch)
    result=coverage.ensure_review_coverage(EnsureReviewCoverageArgs(**request),state);add_observation(state,result)
    cohorts=domain.compare_cohorts(ConditionCohortsArgs.model_validate(state['coverage_followup']['arguments']),state)
    assert cohorts.data['cohort_a']['stats']['count']==10 and cohorts.data['cohort_a']['body_sample_count']==1
    assert cohorts.data['positive_rate_delta_pp']==0 and len(cohorts.data['source']['reference_files'])>=3
    add_observation(state,cohorts,'compare_cohorts')
    progress=assess_investigation_progress(state)
    assert next(c for c in progress['checks'] if c['requirement']=='statistics')['status']=='satisfied'
    packet,_=evidence_packet(state,[],PlannerContextBuilder())
    scopes=[s for s in packet['effective_scopes'] if s.get('cohort')]
    assert len(scopes)==2 and all(s['statistics_basis']=='steam_public_window_counts' for s in scopes)
    assert all(s['bounds_local']['start_inclusive'] for s in scopes)
    prose='\n'.join(readable_statistics(state['observations']))
    assert '全部匹配总数' in prose and '原文样例1条' in prose and '无法计算实际好评率' not in prose


@pytest.mark.parametrize('filters',[{'min_playtime_hours':10},{'max_playtime_hours':20}])
def test_unsupported_population_never_uses_platform_denominator(dated,monkeypatch,filters):
    domain,state,coverage,request=setup(dated);inject_api(monkeypatch)
    result=coverage.ensure_review_coverage(EnsureReviewCoverageArgs(**request),state);add_observation(state,result)
    data=domain.query_reviews(ConditionQueryArgs(appid=730,patch_news_id='old',filters={'period':'before',**filters}),state).data
    assert data['statistics_basis']=='cached_subset' and data['stats']['count']<=1


def test_language_specific_and_vote_specific_counts(dated,monkeypatch):
    domain,state,coverage,request=setup(dated)
    for w in request['windows']:w['language']='schinese'
    state['coverage_followup']['windows']=request['windows'];inject_api(monkeypatch)
    result=coverage.ensure_review_coverage(EnsureReviewCoverageArgs(**request),state);add_observation(state,result)
    data=domain.query_reviews(ConditionQueryArgs(appid=730,patch_news_id='old',filters={'period':'before','language':'schinese','is_positive':False}),state).data
    assert data['stats']['count']==2 and data['stats']['negative_rate']==1
    assert data['body_sample_count']==0 and data['returned_count']==0


@pytest.mark.parametrize('policy',['offline','cache_only','wrong_game','unbound_window'])
def test_network_scope_authorization(dated,monkeypatch,policy):
    domain,state,coverage,request=setup(dated)
    monkeypatch.setattr(api,'acquire_window',lambda *_a,**_k:pytest.fail('Unauthorized network'))
    if policy=='offline':state['allow_review_acquisition']=False
    if policy=='cache_only':state['goal']='仅使用本地缓存，'+GOAL
    if policy=='wrong_game':request['appid']=999
    if policy=='unbound_window':request['windows']=copy.deepcopy(request['windows']);request['windows'][0]['start_at']='2026-07-01T12:00:00Z'
    with pytest.raises(ValueError):coverage.ensure_review_coverage(EnsureReviewCoverageArgs(**request),state)
    if policy in {'offline','cache_only'}:assert coverage.request_for('compare_cohorts',state['coverage_followup']['arguments'],state) is None


def test_failure_retries_once_per_run_and_not_same_empty_as_zero(dated,monkeypatch):
    domain,state,coverage,request=setup(dated)
    calls=inject_api(monkeypatch,error=httpx.ReadTimeout('offline'))
    result=coverage.ensure_review_coverage(EnsureReviewCoverageArgs(**request),state);add_observation(state,result)
    assert len(calls)==2 and set(result.data['error_codes'])=={'request_timeout'}
    assert coverage.request_for('compare_cohorts',state['coverage_followup']['arguments'],state) is None
    data=domain.compare_cohorts(ConditionCohortsArgs.model_validate(state['coverage_followup']['arguments']),state)
    add_observation(state,data,'compare_cohorts')
    decision=InvestigationAgent._dated_review_closeout(state)
    assert decision and 'request_timeout' in decision.final_answer and decision.scope_limited


def test_inadequate_default_attempt_is_suppressed_only_for_current_run(dated,monkeypatch):
    domain,state,coverage,request=setup(dated);inject_api(monkeypatch)
    result=coverage.ensure_review_coverage(EnsureReviewCoverageArgs(**request),state)
    assert all(not row['cache_adequate'] for row in result.data['windows'])
    add_observation(state,result)
    arguments=state['coverage_followup']['arguments']
    assert coverage.request_for('compare_cohorts',arguments,state) is None
    next_run={**state,'observations':state['observations'][:1]}
    followup=coverage.request_for('compare_cohorts',arguments,next_run)
    assert followup and followup['target_count_per_window']==1000


def test_corrupt_or_expired_cache_denominator_never_reused(dated,monkeypatch):
    domain,state,coverage,request=setup(dated);inject_api(monkeypatch)
    result=coverage.ensure_review_coverage(EnsureReviewCoverageArgs(**request),state)
    key=result.data['window_keys'][0];row,ref=coverage.read(730,key)
    args=ConditionQueryArgs(appid=730,patch_news_id='old',filters={'period':'before'})
    before=ReadonlySnapshot(domain.root)(args,state)
    atomic_write_json(ref['path'],{**row,'fetched_at_timestamp':time.time()-TTL-1})
    atomic_write_json(coverage.ledger(730,key),{'archive':{'path':ref['path'],'sha256':hashlib.sha256(open(ref['path'],'rb').read()).hexdigest()}})
    assert coverage.read(730,key) is None and ReadonlySnapshot(domain.root)(args,state)!=before
    # A hash-valid artifact with the wrong query must still fail validation.
    atomic_write_json(ref['path'],{**row,'query':{**row['query'],'appid':999}})
    atomic_write_json(coverage.ledger(730,key),{'archive':{'path':ref['path'],'sha256':hashlib.sha256(open(ref['path'],'rb').read()).hexdigest()}})
    assert coverage.read(730,key) is None


def test_runtime_acquires_before_original_query_then_reads_and_verifies(dated,monkeypatch):
    domain,state,coverage,request=setup(dated);calls=inject_api(monkeypatch)
    state.pop('coverage_followup')
    monkeypatch.setattr(runtime,'run_isolated',lambda handler,args,state,timeout:('result',handler(args,state).model_dump(mode='json')))
    before_id='id-'+str(stamp('2026-07-03'));after_id='id-'+str(stamp('2026-07-10'))
    decisions=[PlannerDecision(action='compare_cohorts',reason='对照',arguments={'cohort_a':{'period':'before'},'cohort_b':{'period':'after'}}),
        PlannerDecision(action='query_reviews',reason='读取前原文',arguments={'filters':{'period':'before'}}),
        PlannerDecision(action='query_reviews',reason='读取后原文',arguments={'filters':{'period':'after'}}),
        PlannerDecision(action='verify_investigation',reason='核验',arguments={'appid':730}),
        PlannerDecision(action='finish',reason='只作有限观察',final_answer='已取得两个时间窗口的评论及原文样例，不能据此推断因果关系。',
            citations=[{'source_step':4,'review_id':before_id,'quote':'反馈 '+before_id},
                       {'source_step':5,'review_id':after_id,'quote':'反馈 '+after_id}],scope_limited=True)]
    class CheckedPlanner(SequencePlanner):
        def decide(self,state,tools):
            if not self.decisions:pytest.fail(str(state.get('validation_errors'))+' / '+str([(o['step'],o['tool'],o['ok']) for o in state['observations']]))
            decision=super().decide(state,tools)
            if decision.action=='finish':
                from gamelens.core.grounding import fact_catalog
                decision.fact_ids=[f['fact_id'] for f in fact_catalog(state['observations']) if f['fact_path']==['cohort_a','stats','count']]
            return decision
    result=InvestigationAgent(CheckedPlanner(decisions),SteamInvestigationTools(domain.root).registry()).invoke(state)
    assert result['status']=='completed',result.get('error')
    assert [o['tool'] for o in result['observations']]==['list_patches','ensure_review_coverage','compare_cohorts','query_reviews','query_reviews','verify_investigation']
    assert len(calls)==2 and result['coverage_followup']=={}
    assert result['observations'][2]['data']['cohort_a']['stats']['count']==10


def test_single_before_query_prepares_both_sides_and_future_is_not_count_complete(dated,monkeypatch):
    domain,state,coverage,_=setup(dated)
    request=coverage.request_for('query_reviews',{'appid':730,'patch_news_id':'old','filters':{'period':'before'}},state)
    assert len(request['windows'])==2
    future='2099-01-01T00:00:00Z'
    window={'start_at':'2026-07-10T12:00:00Z','end_at':future,'language':'all'}
    state['coverage_followup']={'windows':[window]};inject_api(monkeypatch)
    result=coverage.ensure_review_coverage(EnsureReviewCoverageArgs(appid=730,windows=[window]),state)
    assert result.data['windows'][0]['future_end_unobserved'] and not result.data['windows'][0]['stats_verified']


def test_early_finish_on_old_cache_is_redirected_to_exact_acquisition(dated,monkeypatch):
    domain,state,coverage,_=setup(dated);state.pop('coverage_followup')
    planner=SequencePlanner([PlannerDecision(action='finish',reason='旧缓存没有日期',final_answer='没有历史资料',scope_limited=True)])
    agent=InvestigationAgent(planner,SteamInvestigationTools(domain.root).registry())
    redirect=agent._decide_untraced(state,{})
    assert redirect['next_action']=='ensure_review_coverage'
    follow=redirect['coverage_followup']
    assert follow['arguments']['patch_news_id']=='old' and len(follow['windows'])==2
    assert follow['arguments']['cohort_a']['end_at']==follow['arguments']['cohort_b']['start_at']


def test_offline_evaluation_never_autofetches_and_all_cache_query_has_no_invented_window(dated):
    domain,state,coverage,_=setup(dated)
    state.update(allow_review_acquisition=False)
    assert coverage.request_for('query_reviews',{'appid':730,'patch_news_id':'old','filters':{'period':'before'}},state) is None
    state.update(allow_review_acquisition=True,goal='统计本地已采集评论数量')
    assert coverage.request_for('query_reviews',{'appid':730,'filters':{'period':'all'}},state) is None


def test_cached_api_zero_is_not_missing_history(dated,monkeypatch):
    domain,state,coverage,request=setup(dated)
    real=api.acquire_window
    def acquire(appid,start,end,language,**kwargs):
        client,_=mock_client([page([],total=0,positive=0,cursor='*')])
        return real(appid,start,end,language,client=client,sleep=lambda _:None,**kwargs)
    monkeypatch.setattr(api,'acquire_window',acquire)
    result=coverage.ensure_review_coverage(EnsureReviewCoverageArgs(**request),state);add_observation(state,result)
    cohorts=domain.compare_cohorts(ConditionCohortsArgs.model_validate(state['coverage_followup']['arguments']),state)
    add_observation(state,cohorts,'compare_cohorts')
    assert cohorts.data['cohort_a']['stats']['positive_rate'] is None and cohorts.data['cohort_a']['stats']['count']==0
    assert InvestigationAgent._dated_review_closeout(state) is None


def test_restored_followup_continues_without_another_planner_request(dated,monkeypatch):
    domain,state,coverage,request=setup(dated);inject_api(monkeypatch)
    result=coverage.ensure_review_coverage(EnsureReviewCoverageArgs(**request),state);add_observation(state,result)
    class NoPlanner:
        def decide(self,*_):pytest.fail('Restored acquisition must continue its original query')
    restored=json.loads(json.dumps(state))
    redirect=InvestigationAgent(NoPlanner(),SteamInvestigationTools(domain.root).registry())._decide_untraced(restored,{})
    assert redirect['next_action']=='compare_cohorts'


def test_denied_acquisition_is_not_repeated_by_changing_limit(dated):
    domain,state,coverage,request=setup(dated)
    state['observations'].append({'step':2,'tool':'ensure_review_coverage','ok':False,
        'arguments':request,'data':{'error_code':'approval_denied'}})
    assert coverage.request_for('compare_cohorts',state['coverage_followup']['arguments'],state) is None
