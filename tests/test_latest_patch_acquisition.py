"""Latest means the newest official patch, not a model-invented calendar day."""
import json
from pathlib import Path

import pytest

import gamelens.core.runtime as runtime
from gamelens.core.runtime import InvestigationAgent, LLMPlanner
from gamelens.core.synthesis import evidence_packet, target_date_errors
from gamelens.evaluation.benchmark import initial_state
from gamelens.memory.context import PlannerContextBuilder
from gamelens.storage.atomic_io import atomic_write_json
from gamelens.tools.investigation import ListPatchesArgs, ReviewInvestigation
from gamelens.tools.registry import PreparePatchesArgs, SteamInvestigationTools

GOAL = '整理最近一次更新后，游玩时长100h以上，中文玩家的问题，区分支持证据与相反反馈，并附上原始评论。'


def patch(gid='sept', date=1789441427):
    return {'gid':gid, 'title':'v1.0.5: Bug fixes', 'contents':'Fixed textures.',
        'date':date, 'feedname':'steam_community_announcements',
        'feedlabel':'Community Announcements',
        'url':'https://steamcommunity.com/games/1623730/announcements/detail/'+gid}


def state():
    value=initial_state('latest-acquisition', GOAL)
    value.update(appid=1623730, patch_news_id=None, investigation_profile='focused',
        allow_review_acquisition=True, allow_patch_acquisition=True, rag_mode='off')
    return value


class NoPlanning:
    def generate_json(self, messages):
        pytest.fail('Latest-patch discovery must finish before asking the planner to choose a date')


def record(s, tool, args, result):
    s['step_count']+=1
    s['observations'].append({'step':s['step_count'],'tool':tool,'ok':result.ok,
        'arguments':args,'data':result.data,'summary':result.summary})
    s['artifacts'].update(result.artifacts)


def test_missing_cache_is_fetched_then_latest_patch_is_located_before_planning(tmp_path, monkeypatch):
    from gamelens.tools import steam_news
    monkeypatch.setattr(steam_news,'fetch_news',lambda appid,count,**kw:[patch()])
    monkeypatch.setattr(runtime,'run_isolated',lambda handler,args,s,timeout:
        ('result',handler(args,s).model_dump(mode='json')))
    s=state(); registry=SteamInvestigationTools(tmp_path).registry()
    agent=InvestigationAgent(LLMPlanner(NoPlanning()),registry)
    for expected in ('inspect_data','prepare_patches','list_patches'):
        decision=agent._decide_untraced(s,{})
        assert decision['next_action']==expected
        args=decision['next_arguments']
        result=registry.execute(expected,args,s)
        assert result.ok
        record(s,expected,args,result)
    assert s['observations'][-1]['data']['patches'][0]['news_id']=='sept'
    assert s['observations'][-1]['data']['latest_check']['status']=='checked'


def test_existing_cache_is_refreshed_for_latest_without_refresh_approval(tmp_path, monkeypatch):
    from gamelens.tools import steam_news
    from gamelens.tools.steam_news import normalize_news
    old=normalize_news(patch('old',1784000000),1623730).model_dump(mode='json')
    atomic_write_json(tmp_path/'data/raw/1623730_news.json',[old])
    atomic_write_json(tmp_path/'data/raw/1623730_patch_candidates.json',[old])
    calls=[]
    def fetch(appid,count,**kw):
        calls.append((appid,count,kw));return [patch()]
    monkeypatch.setattr(steam_news,'fetch_news',fetch)
    s=state();registry=SteamInvestigationTools(tmp_path).registry()
    args={'appid':1623730,'check_latest':True}
    assert registry.approval_request('prepare_patches',args,s) is None
    result=SteamInvestigationTools(tmp_path).prepare_patches(PreparePatchesArgs(**args),s)
    assert result.ok and not result.data['reused']
    assert calls==[(1623730,100,{'feeds':'steam_community_announcements'})]
    assert {'old','sept'}=={p['news_id'] for p in json.loads((tmp_path/'data/raw/1623730_patch_candidates.json').read_text(encoding='utf-8'))}


def located_state(tmp_path,monkeypatch):
    from gamelens.tools import steam_news
    monkeypatch.setattr(steam_news,'fetch_news',lambda *a,**kw:[patch()])
    s=state()
    result=SteamInvestigationTools(tmp_path).prepare_patches(PreparePatchesArgs(appid=1623730,check_latest=True),s)
    record(s,'prepare_patches',{'appid':1623730,'check_latest':True},result)
    result=ReviewInvestigation(tmp_path).list_patches(ListPatchesArgs(appid=1623730),s)
    record(s,'list_patches',{'appid':1623730},result)
    return s


def test_latest_lookup_does_not_filter_by_invented_today_and_binds_queries(tmp_path,monkeypatch):
    s=located_state(tmp_path,monkeypatch)
    registry=SteamInvestigationTools(tmp_path).registry()
    args=registry.validated_arguments('list_patches',{'published_on':'10-14'},s)
    result=ReviewInvestigation(tmp_path).list_patches(ListPatchesArgs(**args),s)
    assert result.data['requested_date'] is None
    assert result.data['patches'][0]['news_id']=='sept'
    bound=registry.validated_arguments('query_reviews',{'filters':{'period':'after'}},s)
    assert bound['patch_news_id']=='sept'
    assert registry.argument_errors('query_reviews',{'patch_news_id':'wrong','filters':{'period':'after'}},s)
    packet,_=evidence_packet(s,[],PlannerContextBuilder())
    assert packet['target_updates'][0]['news_id']=='sept'
    assert packet['target_updates'][0]['published_on_local']=='2026-09-15'
    from gamelens.core.investigation_checks import assess_investigation_progress
    progress=assess_investigation_progress(s)
    assert next(c for c in progress['checks'] if c['requirement']=='target_update')['status']=='satisfied'
    # Requiring a complete all-time catalog must not invalidate an observed
    # newest patch in a freshly read, newest-first official response.
    assert s['observations'][0]['data']['complete'] is False


def test_latest_source_failure_attempts_once_and_offline_mode_never_fetches(tmp_path,monkeypatch):
    s=state();s['observations']=[{'step':1,'tool':'inspect_data','ok':True,'data':{}}];s['step_count']=1
    agent=InvestigationAgent(LLMPlanner(NoPlanning()),SteamInvestigationTools(tmp_path).registry())
    decision=agent._latest_patch_decision(s)
    assert decision.action=='prepare_patches'
    s['observations'].append({'step':2,'tool':'prepare_patches','ok':False,
        'arguments':decision.arguments,'data':{'error_code':'network_timeout'}})
    s['step_count']=2
    assert agent._latest_patch_decision(s).action=='list_patches'
    s['observations'].append({'step':3,'tool':'list_patches','ok':True,'arguments':{'appid':1623730},'data':{}})
    assert agent._latest_patch_decision(s) is None
    for updates in ({'allow_review_acquisition':False},{'allow_patch_acquisition':False},{'goal':GOAL+'只看本地缓存'}):
        offline={**state(),**updates,'observations':[{'tool':'inspect_data','ok':True}],'step_count':1}
        assert agent._latest_patch_decision(offline) is None


def test_known_latest_target_cannot_be_described_as_missing(tmp_path,monkeypatch):
    s=located_state(tmp_path,monkeypatch)
    decision=runtime.PlannerDecision(action='finish',reason='summary',
        final_answer='缓存中未找到有效补丁公告，无法确认最近一次更新。',scope_limited=True)
    assert target_date_errors(s,decision)


def test_bad_screening_quote_does_not_discard_other_valid_evidence():
    from gamelens.rag.screening import screen_candidates
    hits=[{'kind':'review','source_id':rid,'text':text,'original':{'text':text}}
          for rid,text in (('good','更新后不再闪退'),('bad','游戏偶尔闪退'))]
    class Client:
        def generate_json(self,messages):
            return {'verdicts':[
                {'kind':'review','source_id':'good','role':'counterexample','quote':'更新后不再闪退','reason':'相反体验'},
                {'kind':'review','source_id':'bad','role':'support','quote':'开发者确认引擎损坏','reason':'问题'},
            ]}
    selected,metadata=screen_candidates(Client(),GOAL,'更新后问题',hits)
    assert [h['source_id'] for h in selected]==['good']
    assert selected[0]['evidence_quote']=='更新后不再闪退'
    assert metadata['status']=='degraded'
    assert metadata['rejected_candidates'][0]['source_id']=='bad'
    assert metadata['error_code']=='screening_quote_validation_failed'


def test_empty_official_refresh_does_not_certify_stale_cached_patch(tmp_path,monkeypatch):
    from gamelens.tools import steam_news
    from gamelens.tools.steam_news import normalize_news
    from gamelens.tools.time_scope import located_latest_patch
    atomic_write_json(tmp_path/'data/raw/1623730_news.json',[
        normalize_news(patch('old'),1623730).model_dump(mode='json')])
    monkeypatch.setattr(steam_news,'fetch_news',lambda *a,**kw:[])
    s=state()
    result=SteamInvestigationTools(tmp_path).prepare_patches(PreparePatchesArgs(appid=1623730,check_latest=True),s)
    record(s,'prepare_patches',{'appid':1623730,'check_latest':True},result)
    result=ReviewInvestigation(tmp_path).list_patches(ListPatchesArgs(appid=1623730),s)
    record(s,'list_patches',{'appid':1623730},result)
    assert located_latest_patch(s) is None
    assert result.data['latest_check']['status']=='no_patch_in_response'
    errors=SteamInvestigationTools(tmp_path).registry().argument_errors('query_reviews',
        {'patch_news_id':'old','filters':{'period':'after'}},s)
    assert errors and errors[0]['type']=='latest_target_missing'


def test_changed_latest_source_cannot_bind_queries(tmp_path,monkeypatch):
    from gamelens.tools.time_scope import located_latest_patch
    s=located_state(tmp_path,monkeypatch)
    path=Path(s['observations'][-1]['data']['source']['path'])
    path.write_text('[]',encoding='utf-8')
    assert located_latest_patch(s) is None


def test_fresh_official_check_is_not_replaced_by_an_old_readonly_receipt(tmp_path,monkeypatch):
    from gamelens.tools import steam_news
    monkeypatch.setattr(steam_news,'fetch_news',lambda *a,**kw:[patch()])
    monkeypatch.setattr(runtime,'run_isolated',lambda handler,args,s,timeout:
        ('result',handler(args,s).model_dump(mode='json')))
    s=state();registry=SteamInvestigationTools(tmp_path).registry()
    for _ in range(2):
        result=registry.execute('prepare_patches',{'check_latest':True},s)
        record(s,'prepare_patches',{'check_latest':True},result)
        checked_at=result.data['latest_check']['checked_at']
        result=registry.execute('list_patches',{'limit':1},s)
        assert result.data['latest_check']['checked_at']==checked_at
        record(s,'list_patches',{'limit':1},result)

