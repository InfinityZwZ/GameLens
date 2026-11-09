"""Offline regression for the July-date incident. No real API/model calls."""
import json
from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from gamelens.analysis.version import load_patch_candidates, load_reviews
from gamelens.evaluation.benchmark import initial_state
from gamelens.storage.atomic_io import atomic_write_json
from gamelens.tools.investigation import (ReviewInvestigation, ConditionQueryArgs,
    ConditionCohortsArgs, ListPatchesArgs, MetadataReviewFilter, VerifyInvestigationArgs, coverage_summary)
from gamelens.tools.registry import (SteamInvestigationTools, AppArgs, PreparePatchesArgs,
    patch_fetch_approval)
from gamelens.tools.time_scope import requested_patch_date
from gamelens.rag.tools import RagTools, SearchArgs
from gamelens.core.investigation_checks import investigation_finish_checks, statistics_alignment
from gamelens.analysis.report_prose import readable_statistics

ROOT=Path(__file__).resolve().parents[1]


def stamp(day):
    return int(datetime.fromisoformat(day+"T12:00:00+00:00").timestamp())


@pytest.fixture
def dated(tmp_path):
    patch_base=json.loads((ROOT/"evals/fixtures/730_patch_candidates.json").read_text(encoding="utf-8"))[0]
    review_base=json.loads((ROOT/"evals/fixtures/730_reviews.json").read_text(encoding="utf-8"))[0]
    def announcement(pid,day):
        return {**patch_base,"news_id":pid,"title":"Game Update","contents":"Stability improvements",
            "published_at":day+"T12:00:00Z","published_at_timestamp":stamp(day)}
    patches=[announcement("old","2026-07-10")]+[announcement(f"new-{i}",f"2026-10-{i:02d}") for i in range(1,9)]
    atomic_write_json(tmp_path/"data/raw/730_patch_candidates.json",patches)
    atomic_write_json(tmp_path/"data/raw/730_news.json",patches)
    atomic_write_json(tmp_path/"data/raw/730_reviews.json",[{**review_base,"review_id":"r1","text":"更新后卡顿",
        "created_at":"2026-10-03T12:00:00Z","updated_at":"2026-10-03T12:00:00Z",
        "created_at_timestamp":stamp("2026-10-03"),"updated_at_timestamp":stamp("2026-10-03")}])
    state=initial_state("dated","统计7.10日版本更新后，游戏时长10h以上，中文玩家提到的游戏性能问题")
    state.update(investigation_profile="focused",patch_news_id="new-8",rag_mode="auto")
    return ReviewInvestigation(tmp_path),state,announcement


@pytest.mark.parametrize("goal,expected",[
    ("7.10日版本更新后反馈","07-10"),("7月10日更新后反馈","07-10"),
    ("2025年7月10日更新后反馈","2025-07-10"),("2026-07-10更新后反馈","2026-07-10"),
    ("7.10版本有什么变化",None),("7月10日到7月17日的版本反馈",None),
    ("7月10日评论",None)])
def test_date_extraction_never_invents_year(goal,expected):
    assert requested_patch_date(goal)==expected


def test_historical_lookup_searches_all_cached_patches_and_paginates(dated):
    domain,state,_=dated
    generic={**state,"goal":"列出最近5次更新"}
    assert all(p["news_id"]!="old" for p in domain.list_patches(ListPatchesArgs(appid=730),generic).data["patches"])
    assert domain.list_patches(ListPatchesArgs(appid=730),state).data["patches"][0]["news_id"]=="old"
    result=domain.list_patches(ListPatchesArgs(appid=730,published_on="07-10"),state)
    assert result.data["patches"][0]["news_id"]=="old"
    assert result.data["matched_years"]==["2026"]
    assert result.data["lookup_status"]=="found_in_cache"
    assert "investigation_list_patches" in result.artifacts
    paged=domain.list_patches(ListPatchesArgs(appid=730,offset=5),generic)
    assert paged.data["matched_count"]==9 and paged.data["returned_count"]==4


def test_missing_and_ambiguous_date_are_not_absence_proofs(dated):
    domain,state,announcement=dated
    missing=domain.list_patches(ListPatchesArgs(appid=730,published_on="2024-07-10"),{**state,"goal":"2024年7月10日更新"})
    assert missing.data["lookup_status"]=="not_found_in_cache"
    assert "不代表官方" in missing.data["caution"]
    path=domain.root/"data/raw/730_patch_candidates.json"
    patches=json.loads(path.read_text(encoding="utf-8"))+[announcement("older","2025-07-10")]
    atomic_write_json(path,patches)
    atomic_write_json(domain.root/"data/raw/730_news.json",patches)
    ambiguous=domain.list_patches(ListPatchesArgs(appid=730,published_on="07-10"),state)
    assert ambiguous.data["needs_year_clarification"]
    args=ConditionQueryArgs(appid=730,patch_news_id="old",filters={"period":"after"})
    with pytest.raises(ValueError,match="确认年份"):domain.query_reviews(args,state)
    state["goal"]="2026年7月10日版本更新后玩家的反馈"
    assert domain.query_reviews(args,state).data["patch_news_id"]=="old"


@pytest.mark.parametrize("filters",[
    {"period":"after","start_at":"2026-07-10T00:00:00Z","end_at":"2026-07-17T23:59:59Z"},
    {"period":"all","start_at":"2026-07-10T00:00:00Z","end_at":"2026-07-17T23:59:59Z"}])
def test_dated_request_cannot_fall_back_to_latest_patch(dated,filters):
    domain,state,_=dated
    with pytest.raises(ValueError,match="不能回退最新"):
        domain.query_reviews(ConditionQueryArgs(appid=730,filters=filters),state)


def test_wrong_selected_patch_rejected(dated):
    domain,state,_=dated
    with pytest.raises(ValueError,match="不一致"):
        domain.query_reviews(ConditionQueryArgs(appid=730,patch_news_id="new-8"),state)


def test_original_inverted_window_rejected_after_intersection(dated):
    domain,state,_=dated
    filters=MetadataReviewFilter(start_at="2026-07-10T00:00:00Z",end_at="2026-07-17T23:59:59Z")
    patch=domain.patch(730,state)
    with pytest.raises(ValueError,match="最终查询时间区间无效"):
        domain.select(load_reviews(domain.root/"data/raw/730_reviews.json"),filters,patch,7)


@pytest.mark.parametrize("bounds",[
    {"start_inclusive":20,"end_exclusive":10},{"start_inclusive":10,"end_exclusive":10},
    {"start_inclusive":float("nan")},{"end_exclusive":float("inf")}])
def test_invalid_effective_bounds_never_pass_coverage(bounds):
    with pytest.raises(ValueError):coverage_summary([],bounds)


def test_unbounded_after_is_not_silently_limited_to_seven_days(dated):
    domain,state,_=dated
    state['goal']='统计7.10日版本更新后的玩家评论数量'
    result=domain.query_reviews(ConditionQueryArgs(appid=730,patch_news_id="old"),state)
    assert result.data["bounds"]=={"start_inclusive":stamp("2026-07-10"),"end_exclusive":None}
    assert result.data["stats"]["count"]==1
    assert result.data["coverage"]["requested_start_not_observed"]
    for args in [ConditionQueryArgs(appid=730,patch_news_id="old",window_days=7),
                 ConditionQueryArgs(appid=730,patch_news_id="old",filters={"end_at":"2026-07-17T12:00:00Z"}),
                 ConditionQueryArgs(appid=730,patch_news_id="old",filters={"start_at":"2026-09-01T00:00:00Z"})]:
        with pytest.raises(ValueError):domain.query_reviews(args,state)


def test_explicit_week_with_missing_reviews_reports_unavailable_not_true_zero(dated):
    domain,state,_=dated
    state["goal"]="2026年7月10日更新后7天的玩家反馈"
    result=domain.query_reviews(ConditionQueryArgs(appid=730,patch_news_id="old",window_days=7),state)
    assert result.data["bounds"]["end_exclusive"]==stamp("2026-07-17")
    assert result.data["statistics_status"]=="unavailable_window"
    text="\n".join(readable_statistics([{"tool":"query_reviews","data":result.data}]))
    assert "无法统计" in text and "匹配评论数为0" not in text
    assert not result.data["coverage"]["complete"]


def test_dated_explicit_week_rejects_wrong_days_and_trimmed_window(dated):
    domain,state,_=dated
    state["goal"]="7月10日更新后7天的玩家反馈"
    for args in [ConditionQueryArgs(appid=730,patch_news_id="old",window_days=1),
                 ConditionQueryArgs(appid=730,patch_news_id="old",window_days=7,filters={"start_at":"2026-07-11T00:00:00Z"})]:
        with pytest.raises(ValueError):domain.query_reviews(args,state)


def test_evidence_guard_runs_before_encoder_or_llm(dated):
    domain,state,_=dated
    class Forbidden:
        def encode(self,*args):raise AssertionError("Embedding call forbidden")
        def generate_json(self,*args):raise AssertionError("LLM call forbidden")
    tools=RagTools(domain.root,encoder=Forbidden(),client=Forbidden())
    args=SearchArgs(appid=730,query="performance issues",kind="reviews",filters={"period":"after","start_at":"2026-07-10T00:00:00Z","end_at":"2026-07-17T23:59:59Z"})
    with pytest.raises(ValueError,match="不能回退最新"):tools.search_game_evidence(args,state)


def test_cohort_query_uses_same_date_binding(dated):
    domain,state,_=dated
    args=ConditionCohortsArgs(appid=730,cohort_a={"period":"before"},cohort_b={"period":"after"})
    with pytest.raises(ValueError,match="显式传patch_news_id"):domain.compare_cohorts(args,state)


def test_dated_cohorts_can_compare_different_explicit_windows(dated):
    domain,state,_=dated
    state["goal"]="比较7月10日更新前一天和更新后五天的玩家反馈"
    args=ConditionCohortsArgs(appid=730,patch_news_id="old",cohort_a={"period":"all",
        "start_at":"2026-07-09T12:00:00Z","end_at":"2026-07-10T12:00:00Z"},
        cohort_b={"period":"all","start_at":"2026-07-10T12:00:00Z","end_at":"2026-07-15T12:00:00Z"})
    assert domain.compare_cohorts(args,state).ok


def test_empty_historical_page_does_not_erase_recent_cache(dated,monkeypatch):
    domain,state,_=dated
    state["goal"]="2024年7月10日版本更新后反馈"
    from gamelens.tools import steam_news
    monkeypatch.setattr(steam_news,"fetch_news",lambda *args,**kwargs:[])
    result=SteamInvestigationTools(domain.root).prepare_patches(
        PreparePatchesArgs(appid=730,before_at="2024-07-11T00:00:00Z"),state)
    assert result.data["fetched_count"]==0 and result.data["patch_count"]==9


def test_history_fetch_cannot_invent_year_when_date_is_absent(dated,monkeypatch):
    domain,state,_=dated
    state["goal"]="7月11日更新后的反馈"
    from gamelens.tools import steam_news
    monkeypatch.setattr(steam_news,"fetch_news",lambda *a,**k:pytest.fail("No request before year clarification"))
    with pytest.raises(ValueError,match="确认年份"):
        SteamInvestigationTools(domain.root).prepare_patches(
            PreparePatchesArgs(appid=730,before_at="2026-07-12T00:00:00Z"),state)


def test_history_fetch_cutoff_must_match_requested_day(dated,monkeypatch):
    domain,state,_=dated
    from gamelens.tools import steam_news
    monkeypatch.setattr(steam_news,"fetch_news",lambda *a,**k:pytest.fail("No request with wrong cutoff"))
    with pytest.raises(ValueError,match="截止时间"):
        SteamInvestigationTools(domain.root).prepare_patches(
            PreparePatchesArgs(appid=730,before_at="2026-10-09T00:00:00Z"),state)


def test_inspection_exposes_ranges_before_planning(dated):
    domain,state,_=dated
    result=SteamInvestigationTools(domain.root).inspect_data(AppArgs(appid=730),state)
    assert result.data["review_coverage"]["observed_start"]==stamp("2026-10-03")
    assert result.data["patch_coverage"]["first_date"]=="2026-07-10"


def test_historical_fetch_passes_cutoff_and_preserves_cache(dated,monkeypatch):
    domain,state,_=dated
    from gamelens.tools import steam_news
    calls=[]
    def fake_fetch(appid,count,**kwargs):
        calls.append((appid,count,kwargs))
        return [{"gid":"historical-new","title":"v0.9.0: Bug fixes","contents":"fixed","date":stamp("2026-07-09"),
                 "url":"https://example.com/announcement","feedlabel":"Steam","feedname":"steam_community_announcements"}]
    monkeypatch.setattr(steam_news,"fetch_news",fake_fetch)
    args=PreparePatchesArgs(appid=730,target_count=10,before_at="2026-07-11T00:00:00Z")
    assert patch_fetch_approval(args,state)
    result=SteamInvestigationTools(domain.root).prepare_patches(args,state)
    assert calls==[(730,10,{"enddate":int(args.before_at.timestamp())})]
    ids={p.news_id for p in load_patch_candidates(domain.root/"data/raw/730_patch_candidates.json")}
    assert {"old","new-8","historical-new"}<=ids and len(ids)==10
    assert result.data["fetched_count"]==1 and not result.data["complete"]


@pytest.mark.parametrize("date_value",["not-a-date","1960-01-01T00:00:00Z"])
def test_history_cutoff_requires_valid_datetime_and_supported_epoch(date_value):
    with pytest.raises(ValidationError):PreparePatchesArgs(appid=730,before_at=date_value)


def test_news_http_parameters_without_live_request(monkeypatch):
    from gamelens.tools import steam_news
    captured=[]
    class Response:
        def raise_for_status(self):pass
        def json(self):return {"appnews":{"newsitems":[]}}
    class Client:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def get(self,url,params):captured.append(params);return Response()
    monkeypatch.setattr(steam_news.httpx,"Client",Client)
    assert steam_news.fetch_news(730,10,enddate=stamp("2026-07-11"))==[]
    assert captured[0]["enddate"]==stamp("2026-07-11")
    steam_news.fetch_news(730,10)
    assert "enddate" not in captured[1]
    with pytest.raises(ValueError):steam_news.fetch_news(730,enddate=-1)


def test_old_invalid_artifact_cannot_be_verified_again(dated):
    domain,state,_=dated
    path=domain.root/"data/investigations/invalid.json"
    atomic_write_json(path,{"bounds":{"start_inclusive":20,"end_exclusive":10}})
    state["artifacts"]={"investigation_query_reviews":str(path)}
    with pytest.raises(ValueError,match="最终查询时间区间无效"):
        domain.verify_investigation(VerifyInvestigationArgs(appid=730),state)
    assert statistics_alignment(state["goal"],{"bounds":{"start_inclusive":20,"end_exclusive":10}},state)


def test_missing_target_can_only_finish_as_scope_limited():
    goal="统计7月10日更新后玩家的性能反馈"
    observations=[{"ok":True,"tool":"list_patches","data":{"requested_date":"07-10","lookup_status":"not_found_in_cache"}}]
    assert investigation_finish_checks(goal,observations,[],False)[0]
    assert not investigation_finish_checks(goal,observations,[],True)[0]


def test_dated_missing_reviews_cannot_finish_as_complete_investigation():
    goal="统计7月10日更新后7天性能反馈"
    observations=[{"ok":True,"tool":"list_patches","data":{"requested_date":"07-10","lookup_status":"found_in_cache"}},
        {"ok":True,"tool":"query_reviews","data":{"coverage":{"status":"outside_observed_range"}}}]
    assert investigation_finish_checks(goal,observations,[],False)[0]
    assert not investigation_finish_checks(goal,observations,[],True)[0]


def test_fake_planner_can_complete_missing_history_as_clarification(dated):
    from gamelens.core.runtime import InvestigationAgent, LLMPlanner
    domain,state,_=dated
    state["goal"]="调查7月11日更新后的性能问题"
    class FakeClient:
        def __init__(self):
            self.messages=[]
            self.responses=iter([
                {"action":"list_patches","reason":"定位历史日期","arguments":{"appid":730}},
                {"action":"verify_investigation","reason":"核对缺失快照","arguments":{"appid":730}},
                {"action":"finish","reason":"历史资料不足，需要澄清年份","scope_limited":True,
                 "final_answer":"当前缓存没有7月11日的公告，请确认年份；不能由此判断没有这次更新。",
                 "detailed_answer":"未取得目标更新和当时的评论，无法完成性能调查。",
                 "claims":[{"label":"缓存内匹配数","source_step":2,"fact_path":["matched_count"],"value":0}]}])
        def generate_json(self,messages):
            self.messages.append(messages)
            return next(self.responses)
    client=FakeClient()
    result=InvestigationAgent(LLMPlanner(client),SteamInvestigationTools(domain.root,client).registry()).invoke(state)
    assert result["status"]=="completed"
    assert [o["tool"] for o in result["observations"]]==["inspect_data","list_patches","verify_investigation"]
    assert "确认年份" in result["final_answer"] and len(client.messages)==3


def test_missing_announcement_lookup_can_produce_auditable_limitation_report(dated):
    domain,state,_=dated
    state["goal"]="2024年7月10日版本更新后反馈"
    lookup=domain.list_patches(ListPatchesArgs(appid=730,published_on="2024-07-10"),state)
    state["artifacts"].update(lookup.artifacts)
    result=domain.verify_investigation(VerifyInvestigationArgs(appid=730),state)
    assert result.ok
    text=Path(result.artifacts["final_report"]).read_text(encoding="utf-8")
    assert "目标更新定位" in text and "官方没有" in text
