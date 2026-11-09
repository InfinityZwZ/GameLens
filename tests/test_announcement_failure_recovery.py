"""Offline regressions for the actual Oct 9 failure: press, timezone, stale lookup."""
import copy
import hashlib
from dataclasses import replace
from datetime import datetime, timezone

import pytest

import gamelens.core.runtime as runtime
from gamelens.core.answers import verified_sources_current
from gamelens.core.grounding import fact_catalog, validate_grounding
from gamelens.core.investigation_checks import assess_investigation_progress
from gamelens.core.runtime import InvestigationAgent, PlannerDecision, ToolResult
from gamelens.core.source_freshness import superseded_locator_steps
from gamelens.domain.announcement_comments import announcement_only_goal
from gamelens.memory.context import PlannerContextBuilder
from gamelens.storage.atomic_io import atomic_write_json
from gamelens.tools.announcement_comments import AnnouncementDiscussionTools, PrepareDiscussionArgs
from gamelens.tools.investigation import ListPatchesArgs, ReviewInvestigation, VerifyInvestigationArgs
from gamelens.tools.registry import SteamInvestigationTools
from gamelens.tools.steam_news import normalize_news
from gamelens.tools.steam_announcement_comments import is_community_announcement
from gamelens.evaluation.benchmark import initial_state
from test_agent_runtime import SequencePlanner


GOAL = '调查玩家对 2026-10-09 更新版本的反馈内容。请读取该更新公告下的玩家留言，归纳主要赞同、质疑和具体问题，并说明采集范围。'


def record(gid, date, *, official=True):
    feed = 'steam_community_announcements' if official else 'PlayGround.ru - 官方新闻组'
    return normalize_news({'gid':gid, 'date':int(datetime.fromisoformat(date).replace(tzinfo=timezone.utc).timestamp()),
        'title':'Counter-Strike 2 Update' if official else 'Valve AUG reload update',
        'url':'https://steamstore-a.akamaihd.net/news/externalpost/'+feed+'/'+gid,
        'feedname':feed, 'contents':'Stability improvements.'},730)


def state():
    result=initial_state('oct9-replay',GOAL)
    result.update(investigation_profile='focused',rag_mode='auto',max_steps=12)
    return result


def write_news(root, records):
    atomic_write_json(root/'data/raw/730_news.json',[r.model_dump(mode='json') for r in records])


def observe(current, tool, args, result):
    current['step_count'] += 1
    current['observations'].append({'step':current['step_count'],'tool':tool,'arguments':args,'ok':result.ok,'data':result.data})
    current['artifacts'].update(result.artifacts)


def test_press_article_is_not_an_official_announcement_or_thread(tmp_path):
    tools=AnnouncementDiscussionTools(tmp_path); current=state()
    official=record('1846018067931597','2026-10-08T22:36:49')
    press=record('1846018067932423','2026-10-09T07:28:45',official=False)
    write_news(tmp_path,[official,press])
    result=tools.list_announcements(ListPatchesArgs(appid=730,published_on='2026-10-09'),current)
    assert result.data['matched_count']==0 and result.data['selected_news_id'] is None
    assert result.data['excluded_external_news_count']==1
    neighbor=result.data['nearby_announcements'][0]
    assert neighbor['published_on_local']=='2026-10-08'
    assert result.data['needs_date_clarification']
    assert not result.data['announcements']
    with pytest.raises(ValueError,match='外部新闻'):
        tools.target(PrepareDiscussionArgs(appid=730,news_id=press.news_id),current)


@pytest.mark.parametrize('url',[
    'https://example.com/announcements/detail/1',
    'https://steamcommunity.com.evil.test/ogg/730/announcements/detail/1',
    'https://evil@steamcommunity.com/ogg/730/announcements/detail/1',
    'https://steamstore-a.akamaihd.net/news/externalpost/PlayGround.ru/1',
    'https://steamcommunity.com:bad/ogg/730/announcements/detail/1'])
def test_unsupported_or_unsafe_news_are_excluded(url):
    item=record('1','2026-10-08T22:36:49'); item.url=url
    assert not is_community_announcement(item)


def test_old_exact_locator_is_superseded_without_rewriting_audit_history(tmp_path):
    tools=AnnouncementDiscussionTools(tmp_path); current=state()
    args={'appid':730,'published_on':'2026-10-09'}
    write_news(tmp_path,[])
    observe(current,'list_announcements',args,tools.list_announcements(ListPatchesArgs(**args),current))
    old=copy.deepcopy(current['observations'][0])
    write_news(tmp_path,[record('1','2026-10-09T06:00:00')])
    observe(current,'list_announcements',args,tools.list_announcements(ListPatchesArgs(**args),current))
    verification=ReviewInvestigation(tmp_path).verify_investigation(VerifyInvestigationArgs(appid=730),current)
    observe(current,'verify_investigation',{'appid':730},verification)
    assert verified_sources_current(current)
    assert superseded_locator_steps(current['observations'])=={1:2}
    assert current['observations'][0]==old
    assert not any(f['source_step']==1 for f in fact_catalog(current['observations']))
    assert validate_grounding(current['observations'],[{'label':'旧定位','source_step':1,'fact_path':['matched_count'],'value':0}],[])
    context=PlannerContextBuilder().build(current,[])
    assert context['superseded_locator_steps']=={1:2}
    assert all(o['step']!=1 for o in context['recent_observations'])
    write_news(tmp_path,[])  # Editing the ACTIVE snapshot is still rejected.
    assert not verified_sources_current(current)
    with pytest.raises(ValueError,match='重做该查询'):
        ReviewInvestigation(tmp_path).verify_investigation(VerifyInvestigationArgs(appid=730),current)


def test_different_locator_does_not_hide_stale_source(tmp_path):
    tools=AnnouncementDiscussionTools(tmp_path); current=state()
    args={'appid':730,'published_on':'2026-10-09','limit':5}
    write_news(tmp_path,[])
    observe(current,'list_announcements',args,tools.list_announcements(ListPatchesArgs(**args),current))
    write_news(tmp_path,[record('1','2026-10-09T06:00:00')])
    args={**args,'limit':10}
    observe(current,'list_announcements',args,tools.list_announcements(ListPatchesArgs(**args),current))
    assert not superseded_locator_steps(current['observations'])
    with pytest.raises(ValueError,match='步骤1'):
        ReviewInvestigation(tmp_path).verify_investigation(VerifyInvestigationArgs(appid=730),current)


def test_store_evidence_cannot_be_added_by_planner_to_announcement_only_task():
    current=state()
    progress=assess_investigation_progress(current,required_evidence=['review_evidence','statistics','counterexample_check'])
    assert progress['required_evidence']==['announcement_comments']
    current['goal'] += '同时结合商店评价中的具体问题证据。'
    assert not announcement_only_goal(current['goal'])
    progress=assess_investigation_progress(current,required_evidence=['review_evidence'])
    assert 'review_evidence' in progress['required_evidence']


def inline(monkeypatch):
    monkeypatch.setattr(runtime,'run_isolated',lambda handler,args,current,timeout:('result',handler(args,current).model_dump(mode='json')))


def test_oct9_exact_workflow_refreshes_once_then_clarifies_not_twelve_steps(tmp_path,monkeypatch):
    inline(monkeypatch)
    write_news(tmp_path,[])
    current=state();registry=SteamInvestigationTools(tmp_path).registry()
    official=record('1846018067931597','2026-10-08T22:36:49')
    press=record('1846018067932423','2026-10-09T07:28:45',official=False)
    def refresh(args,current):
        write_news(tmp_path,[official,press])
        return ToolResult(ok=True,summary='离线重放刷新',data={'fetched_count':2})
    registry._tools['prepare_patches']=replace(registry._tools['prepare_patches'],handler=refresh,approval_policy=None)
    planner=SequencePlanner([
        PlannerDecision(action='list_announcements',reason='定位',arguments={'appid':730,'published_on':'2026-10-09','limit':10}),
        PlannerDecision(action='prepare_patches',reason='补采',arguments={'appid':730,'before_at':'2026-10-10T00:00:00Z'}),
        PlannerDecision(action='list_announcements',reason='再次定位',arguments={'appid':730,'published_on':'2026-10-09','limit':10})])
    result=InvestigationAgent(planner,registry).invoke(current)
    assert result['status']=='completed',result.get('error')
    assert result['scope_limited'] and result['investigation_progress']['needs_clarification']
    assert result['answer_origin']=='deterministic_source_limitation'
    assert result['step_count']==4 and [o['tool'] for o in result['observations']].count('verify_investigation')==1
    assert '2026-10-08' in result['final_answer'] and '没有采集或分析' in result['final_answer']
    assert not result['final_citations'] and not result['final_comment_citations']
    assert verified_sources_current(result)
    assert len(planner.decisions)==0
    report=__import__('pathlib').Path(result['artifacts']['final_report']).read_text(encoding='utf-8')
    assert '玩家反馈内容调查未完成' in report


def test_unavailable_discussion_stops_without_store_search(tmp_path,monkeypatch):
    inline(monkeypatch)
    write_news(tmp_path,[record('1','2026-10-09T06:00:00')])
    from gamelens.tools.steam_announcement_comments import DiscussionError
    def unavailable(*_):raise DiscussionError('discussion_unavailable','公开入口不可读')
    monkeypatch.setattr('gamelens.tools.steam_announcement_comments.fetch_announcement_comments',unavailable)
    planner=SequencePlanner([
        PlannerDecision(action='list_announcements',reason='定位',arguments={'appid':730,'published_on':'2026-10-09'}),
        PlannerDecision(action='prepare_announcement_comments',reason='采集',arguments={'appid':730,'news_id':'1'})])
    result=InvestigationAgent(planner,SteamInvestigationTools(tmp_path).registry()).invoke(state())
    assert result['status']=='completed' and result['scope_limited'],result.get('error')
    assert result['step_count']==3
    assert 'discussion_unavailable' in result['final_answer'] and '没有用商店评价替代' in result['final_answer']
    assert all(o['tool']!='search_game_evidence' for o in result['observations'])


def test_wrong_source_action_is_rejected_before_execution(tmp_path):
    class WrongPlanner:
        def decide(self,current,tools):
            assert 'search_game_evidence' not in {t['name'] for t in tools}
            return PlannerDecision(action='search_game_evidence',reason='错误回退',arguments={'appid':730,'query':'feedback'})
    result=InvestigationAgent(WrongPlanner(),SteamInvestigationTools(tmp_path).registry()).invoke(state())
    assert result['status']=='failed' and result['step_count']==0
    assert '不允许商店评价' in result['error']


def test_discussion_fetch_filters_official_feed_but_preserves_other_news(tmp_path,monkeypatch):
    from gamelens.tools.registry import PreparePatchesArgs
    tools=SteamInvestigationTools(tmp_path);current=state()
    press=record('2','2026-10-09T07:28:45',official=False)
    write_news(tmp_path,[press]);captured=[]
    def fetch(appid,count,**kwargs):captured.append(kwargs);return []
    monkeypatch.setattr('gamelens.tools.steam_news.fetch_news',fetch)
    tools.prepare_patches(PreparePatchesArgs(appid=730,before_at='2026-10-10T00:00:00Z'),current)
    assert captured[0]['feeds']=='steam_community_announcements'
    assert (tmp_path/'data/raw/730_news.json').is_file()
    assert 'PlayGround.ru' in (tmp_path/'data/raw/730_news.json').read_text(encoding='utf-8')
