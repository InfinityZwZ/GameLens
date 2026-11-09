"""Local-calendar integration, no OS mutation, no paid model or network calls."""
import copy
import hashlib
from datetime import datetime, timedelta, timezone

import pytest
from gamelens.core import local_time
from gamelens.core.answers import verified_sources_current
from gamelens.core.grounding import fact_catalog
from gamelens.core.investigation_checks import statistics_alignment, named_patch_time_alignment
from gamelens.core.synthesis import evidence_packet, target_date_errors
from gamelens.core.runtime import PlannerDecision
from gamelens.domain.models import SteamNewsItem
from gamelens.memory.context import PlannerContextBuilder
from gamelens.storage.atomic_io import atomic_write_json
from gamelens.tools.announcement_comments import AnnouncementDiscussionTools, QueryDiscussionArgs
from gamelens.tools.investigation import ListPatchesArgs, MetadataReviewFilter, ReviewInvestigation
from gamelens.tools.registry import PreparePatchesArgs, SteamInvestigationTools
from gamelens.tools.time_scope import patch_day
from gamelens.evaluation.benchmark import initial_state

pytestmark=pytest.mark.system_local_time


def fixed_clock(monkeypatch, hours):
    zone=timezone(timedelta(hours=hours))
    monkeypatch.setattr(local_time,'_system_localize',lambda moment:
        moment.replace(tzinfo=zone) if moment.tzinfo is None else moment.astimezone(zone))


def news():
    moment=datetime(2026,10,8,22,36,49,tzinfo=timezone.utc)
    return SteamNewsItem(news_id='1846018067931597',appid=730,title='Counter-Strike 2 Update',
        url='https://steamstore-a.akamaihd.net/news/externalpost/steam_community_announcements/1846018067931597',
        published_at=moment,published_at_timestamp=int(moment.timestamp()),is_patch_candidate=True)


@pytest.mark.parametrize('offset,expected',[(11,'2026-10-09'),(8,'2026-10-09'),(-7,'2026-10-08'),(0,'2026-10-08')])
def test_announcement_lookup_uses_computer_calendar_not_utc(tmp_path,monkeypatch,offset,expected):
    fixed_clock(monkeypatch,offset)
    item=news(); raw=item.model_dump(mode='json')
    atomic_write_json(tmp_path/'data/raw/730_news.json',[raw])
    before=(tmp_path/'data/raw/730_news.json').read_bytes()
    current=initial_state('local-lookup',f'调查玩家对{expected}更新版本的反馈，请读取公告留言。')
    result=AnnouncementDiscussionTools(tmp_path).list_announcements(ListPatchesArgs(appid=730,published_on=expected),current)
    assert result.data['matched_count']==1 and not result.data.get('needs_date_clarification')
    assert result.data['announcements'][0]['published_on_local']==expected
    assert patch_day(item)==expected
    assert (tmp_path/'data/raw/730_news.json').read_bytes()==before
    assert item.published_at_timestamp==raw['published_at_timestamp']


@pytest.mark.parametrize('model,field',[(MetadataReviewFilter,'start_at'),(QueryDiscussionArgs,'start_at'),(PreparePatchesArgs,'before_at')])
def test_naive_dates_are_local_and_explicit_offsets_keep_instants(monkeypatch,model,field):
    fixed_clock(monkeypatch,11)
    extra={'appid':730,'news_id':'1'} if model is QueryDiscussionArgs else {'appid':730} if model is PreparePatchesArgs else {}
    naive=model.model_validate({**extra,field:'2026-10-09T00:00:00'})
    absolute=model.model_validate({**extra,field:'2026-10-09T00:00:00Z'})
    assert getattr(naive,field).utcoffset()==timedelta(hours=11)
    assert getattr(absolute,field).timestamp()-getattr(naive,field).timestamp()==11*3600


def test_historical_cutoff_uses_next_local_midnight(tmp_path,monkeypatch):
    fixed_clock(monkeypatch,11)
    current=initial_state('local-cutoff','调查玩家对2026-10-09更新版本的反馈')
    captured=[]
    monkeypatch.setattr('gamelens.tools.steam_news.fetch_news',lambda appid,count,**kwargs: captured.append(kwargs) or [])
    tools=SteamInvestigationTools(tmp_path)
    tools.prepare_patches(PreparePatchesArgs(appid=730,before_at='2026-10-10T00:00:00'),current)
    assert captured[0]['enddate']==int(datetime(2026,10,9,13,tzinfo=timezone.utc).timestamp())
    assert captured[0]['feeds']=='steam_community_announcements'
    with pytest.raises(ValueError,match='北京时间零点'):
        tools.prepare_patches(PreparePatchesArgs(appid=730,before_at='2026-10-10T00:00:00Z'),current)


def sydney_dst(moment):
    """Synthetic system adapter for Sydney's 2026 spring-forward boundary."""
    ten,eleven=timezone(timedelta(hours=10)),timezone(timedelta(hours=11))
    if moment.tzinfo is None:
        return moment.replace(tzinfo=eleven if moment>=datetime(2026,10,4,3) else ten)
    transition=datetime(2026,10,3,16,tzinfo=timezone.utc)
    return moment.astimezone(eleven if moment>=transition else ten)


def test_local_calendar_dst_day_is_23_hours_and_all_gates_agree(tmp_path,monkeypatch):
    monkeypatch.setattr(local_time,'_system_localize',sydney_dst)
    start=local_time.local_midnight('2026-10-04').timestamp()
    end=local_time.local_midnight('2026-10-05').timestamp()
    assert end-start==23*3600
    item=news().model_copy(update={'published_at_timestamp':int(start)})
    filters=MetadataReviewFilter(period='after')
    _,bounds=ReviewInvestigation(tmp_path).select([],filters,item,1)
    assert bounds=={'start_inclusive':start,'end_exclusive':end}
    data={'bounds':bounds,'filters':filters.model_dump(mode='json')}
    assert not statistics_alignment('更新后1天中文评论', {**data,'filters':{**data['filters'],'language':'schinese'}},{})
    assert not named_patch_time_alignment('更新后1天',filters,item,1)
    assert named_patch_time_alignment('更新后1天',MetadataReviewFilter(period='all',start_at=local_time.local_iso(start),end_at=local_time.local_iso(start+86400)),item,1)
    with pytest.raises(ValueError,match='夏令时跳过'):
        MetadataReviewFilter(start_at='2026-10-04T02:30:00')


def test_source_timezone_change_invalidates_verified_flag(tmp_path,monkeypatch):
    fixed_clock(monkeypatch,11)
    source=tmp_path/'source.json';atomic_write_json(source,[])
    current={'artifacts':{'evidence_assessed':'true'},'observations':[
        {'step':1,'ok':True,'tool':'query_reviews','data':{'source':{'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'time_zone':{'basis':'another_clock'}}}},
        {'step':2,'ok':True,'tool':'verify_investigation','data':{}}]}
    assert not verified_sources_current(current)


def test_fall_back_day_is_25_hours_and_ambiguous_hour_requires_offset(monkeypatch):
    ten,eleven=timezone(timedelta(hours=10)),timezone(timedelta(hours=11))
    def fall_clock(moment):
        if moment.tzinfo is None:
            ambiguous=moment.date()==datetime(2026,4,5).date() and moment.hour==2
            zone=(ten if moment.fold else eleven) if ambiguous else eleven if moment<datetime(2026,4,5,3) else ten
            return moment.replace(tzinfo=zone)
        return moment.astimezone(ten if moment>=datetime(2026,4,4,16,tzinfo=timezone.utc) else eleven)
    monkeypatch.setattr(local_time,'_system_localize',fall_clock)
    assert local_time.local_midnight('2026-04-06').timestamp()-local_time.local_midnight('2026-04-05').timestamp()==25*3600
    with pytest.raises(ValueError,match='夏令时重复'):
        MetadataReviewFilter(start_at='2026-04-05T02:30:00')
    earlier=MetadataReviewFilter(start_at='2026-04-05T02:30:00+11:00')
    later=MetadataReviewFilter(start_at='2026-04-05T02:30:00+10:00')
    assert later.start_at.timestamp()-earlier.start_at.timestamp()==3600


def test_context_final_packet_and_date_gate_all_use_local_day(monkeypatch):
    fixed_clock(monkeypatch,11)
    current=initial_state('local-packet','调查2026-10-09更新后反馈')
    item=news()
    current['observations']=[{'step':1,'tool':'query_reviews','ok':True,'data':{'patch_news_id':item.news_id,'announcements':[item.model_dump(mode='json')]}}]
    packet,_=evidence_packet(current,[],PlannerContextBuilder())
    assert packet['target_updates'][0]['published_on_local']=='2026-10-09'
    context=PlannerContextBuilder().build(current,[])
    assert '北京时间' in context['time_standard']['date_rule']
    assert not target_date_errors(current,PlannerDecision(action='finish',reason='ok',final_answer='本次更新发布于2026-10-09'))
    assert target_date_errors(current,PlannerDecision(action='finish',reason='bad',final_answer='本次更新发布于2026-10-08'))


def test_api_display_conversion_never_mutates_original_records(monkeypatch):
    fixed_clock(monkeypatch,11)
    original={'started_at':'2026-10-08T22:36:49Z','events':[{'data':{'comments':[{'created_at':'2026-10-08T22:36:49Z'}]}}]}
    saved=copy.deepcopy(original)
    converted=local_time.display_view(original)
    assert original==saved
    assert converted['started_at_local']=='2026-10-09T09:36:49+11:00'
    assert converted['events'][0]['data']['comments'][0]['created_at_local']==converted['started_at_local']


def test_beijing_offset_is_used_without_new_dependencies(monkeypatch):
    monkeypatch.setenv('TZ','America/Los_Angeles')
    moment=datetime(2026,10,8,22,36,49,tzinfo=timezone.utc)
    assert local_time.local_datetime(moment).isoformat()=='2026-10-09T06:36:49+08:00'
    assert local_time.local_day(moment)=='2026-10-09'
    assert local_time.local_midnight('2026-10-09').timestamp()==datetime(2026,10,8,16,tzinfo=timezone.utc).timestamp()


def test_readonly_receipts_do_not_change_when_host_timezone_changes(tmp_path,monkeypatch):
    from gamelens.tools.query_reuse import ReadonlySnapshot, digest
    from gamelens.tools.registry import AppArgs
    snapshot=ReadonlySnapshot(tmp_path)
    current=initial_state('cache-clock','调查游戏反馈')
    before=snapshot(AppArgs(appid=730),current)
    import time
    monkeypatch.setattr(time,'tzname',('Other standard','Other daylight'))
    monkeypatch.setenv('TZ','America/Los_Angeles')
    after=snapshot(AppArgs(appid=730),current)
    assert digest(before)==digest(after)


def test_beijing_calendar_has_no_sydney_dst_and_preserves_explicit_instants(monkeypatch):
    monkeypatch.setenv('TZ','Australia/Sydney')
    assert local_time.timezone_signature()['basis']=='beijing'
    assert local_time.timezone_signature()['system_zone']=='Asia/Shanghai'
    assert local_time.local_midnight('2026-10-05').timestamp()-local_time.local_midnight('2026-10-04').timestamp()==86400
    naive=MetadataReviewFilter(start_at='2026-10-09T00:00:00')
    aware=MetadataReviewFilter(start_at='2026-10-08T16:00:00Z')
    assert naive.start_at==aware.start_at
    assert naive.start_at.utcoffset()==timedelta(hours=8)
    assert local_time.local_iso('2026-12-31T18:00:00Z')=='2027-01-01T02:00:00+08:00'
    assert '北京时间' in local_time.time_context()['date_rule']


def test_gold_filters_and_review_cards_use_beijing_naive_boundaries():
    from gamelens.evaluation.benchmark_score import eligible_records
    from gamelens.evaluation.benchmark_review import population,stamp
    rows=[{'review_id':'at-start','created_at_timestamp':int(datetime(2026,9,14,16,tzinfo=timezone.utc).timestamp())},
          {'review_id':'too-early','created_at_timestamp':int(datetime(2026,9,14,15,59,59,tzinfo=timezone.utc).timestamp())}]
    rows=[{**r,'language':'schinese','playtime_minutes':0,'playtime_at_review_minutes':0,'is_positive':True} for r in rows]
    filters={'start_at':'2026-09-15T00:00:00','end_at':'2026-09-16T00:00:00'}
    assert [r['review_id'] for r in eligible_records({'reviews':rows},filters)]==['at-start']
    assert [r['review_id'] for r in population(rows,filters)]==['at-start']
    assert stamp(rows[0]['created_at_timestamp'])=='2026-09-15 00:00:00 北京时间 UTC+8'


def test_api_returns_local_display_but_keeps_original_store(tmp_path,monkeypatch):
    fixed_clock(monkeypatch,11)
    from gamelens.api import app as api
    from gamelens.storage.runs import RunStore
    from gamelens.storage.telemetry import TelemetryStore
    store=RunStore(tmp_path/'runs.sqlite')
    monkeypatch.setattr(api,'RUN_STORE',store)
    monkeypatch.setattr(api,'TELEMETRY_STORE',TelemetryStore(tmp_path/'telemetry.sqlite'))
    monkeypatch.setattr(api,'RUNS',{})
    request=api.RunRequest(appid=730,goal='测试本地日期显示')
    run=api.RunState(run_id='local-api',request=request,status='completed',
        started_at='2026-10-08T22:36:49Z',updated_at='2026-10-08T23:00:00Z')
    api.RUNS[run.run_id]=run
    api.persist_run(run,touch=False)
    original=copy.deepcopy(store.get(run.run_id))
    result=api.get_run(run.run_id)
    assert result['started_at_local']=='2026-10-09T09:36:49+11:00'
    assert result['started_at']==original['started_at']
    assert store.get(run.run_id)==original
    assert api.list_runs()[0]['started_at_local']==result['started_at_local']


def test_oct9_local_feedback_workflow_reads_oct8_utc_announcement(tmp_path,monkeypatch):
    fixed_clock(monkeypatch,11)
    import httpx
    import gamelens.core.runtime as runtime
    from gamelens.core.runtime import InvestigationAgent
    from gamelens.tools.steam_announcement_comments import fetch_announcement_comments
    from test_announcement_comments import provider, NEWS
    from test_agent_runtime import SequencePlanner
    item=news()
    atomic_write_json(tmp_path/'data/raw/730_news.json',[item.model_dump(mode='json')])
    client,_=provider()
    def redirect(request):
        if request.url.host=='steamstore-a.akamaihd.net':
            return httpx.Response(302,headers={'location':'https://steamcommunity.com/games/CSGO/announcements/detail/'+NEWS})
        return client._transport.handle_request(request)
    saved=fetch_announcement_comments(item,4,client=httpx.Client(transport=httpx.MockTransport(redirect)),request_interval=0)
    monkeypatch.setattr('gamelens.tools.steam_announcement_comments.fetch_announcement_comments',lambda *_:copy.deepcopy(saved))
    monkeypatch.setattr(runtime,'run_isolated',lambda handler,args,current,timeout:('result',handler(args,current).model_dump(mode='json')))
    current=initial_state('local-oct9','调查玩家对2026-10-09更新版本的反馈内容，请读取公告下的玩家留言。')
    current.update(investigation_profile='focused',rag_mode='auto')
    planner=SequencePlanner([
        PlannerDecision(action='list_announcements',reason='本地日期定位',arguments={'appid':730,'published_on':'2026-10-09'}),
        PlannerDecision(action='prepare_announcement_comments',reason='公告讨论采集',arguments={'appid':730,'news_id':item.news_id}),
        PlannerDecision(action='query_announcement_comments',reason='读取留言',arguments={'appid':730,'news_id':item.news_id}),
        PlannerDecision(action='verify_investigation',reason='核验',arguments={'appid':730}),
        PlannerDecision(action='finish',reason='回答',final_answer='本次更新发布于2026-10-09，部分留言称卡顿改善，仍有闪退反馈。',
            claims=[{'source_step':3,'label':'展示留言','fact_path':['returned_count'],'value':4}],
            comment_citations=[{'source_step':3,'news_id':item.news_id,'comment_id':'100','quote':'卡顿已经改善'}])])
    result=InvestigationAgent(planner,SteamInvestigationTools(tmp_path).registry()).invoke(current)
    assert result['status']=='completed',result.get('error')
    assert result['final_comment_citations'] and not result['scope_limited']
    assert result['observations'][0]['data']['announcements'][0]['published_on_local']=='2026-10-09'
