import copy
import pytest

from gamelens.core.investigation_checks import statistics_alignment, fast_statistics_allowed, cited_window_audit, investigation_finish_checks, explicit_patch_alignment
from gamelens.tools.investigation import coverage_summary
from gamelens.evaluation.business_cases import materialize_dataset
from gamelens.analysis.version import load_reviews


def query(**changes):
    result = {"filters": {"period": "all", "is_positive": None, "playtime_basis": "at_review"}, "bounds": {}, "patch_news_id": None}
    result.update(changes)
    return result


@pytest.mark.parametrize("goal,data", [
    ("本地评论数量与好评率", query(filters={"period":"after"})),
    ("本地评论数量与好评率", query(filters={"period":"all", "is_positive":True})),
    ("本地评论数量与好评率", query(filters={"period":"all", "keywords":["bug"]})),
    ("p5更新后一天内评论数量与好评率", query(patch_news_id="p4", filters={"period":"after"}, bounds={"start_inclusive":0,"end_exclusive":86400})),
    ("p5更新后一天内评论数量与好评率", query(patch_news_id="p5", filters={"period":"after"}, bounds={"start_inclusive":0,"end_exclusive":7*86400})),
    ("评价时100小时以上玩家评论数量与推荐率", query(filters={"period":"all","min_playtime_hours":100,"playtime_basis":"lifetime"})),
    ("当前累计100小时以上玩家评论数量与推荐率", query(filters={"period":"all","min_playtime_hours":100,"playtime_basis":"at_review"})),
    ("中文评论数量与推荐率", query(filters={"period":"all","language":"english"})),
])
def test_wrong_conditions_never_fast_finish(goal, data):
    assert statistics_alignment(goal,data,{"appid":730})
    assert not fast_statistics_allowed(goal,data,{"appid":730})


@pytest.mark.parametrize("goal", ["超过100小时玩家的评论数量与推荐率", "昨天评论数量与好评率", "2026-10-01评论数量与好评率", "提到崩溃的评论数量"])
def test_unsupported_time_and_strict_inequalities_use_planner(goal):
    assert not fast_statistics_allowed(goal,query(),{})


def test_supported_patch_window_remains_fast():
    data=query(patch_news_id="p5", filters={"period":"after"}, bounds={"start_inclusive":0,"end_exclusive":86400})
    assert fast_statistics_allowed("p5更新后一天内，缓存评论数量和好评率是多少？", data, {})


def test_revised_text_is_not_historical_evidence(tmp_path):
    materialize_dataset(tmp_path)
    reviews=load_reviews(tmp_path/"data/raw/730_reviews.json")
    first=reviews[0].model_copy(update={"updated_at_timestamp":reviews[-1].created_at_timestamp})
    coverage=coverage_summary([first],{"start_inclusive":first.created_at_timestamp,"end_exclusive":first.created_at_timestamp+86400})
    assert coverage["revised_after_window_count"]==1
    assert coverage["complete"] is False and coverage["historical_text_available"] is False
    assert coverage_summary([], {})["status"]=="no_cached_reviews"
    assert coverage_summary(reviews,{"start_inclusive":reviews[-1].created_at_timestamp+86400})["status"]=="outside_observed_range"


def test_distinct_quotes_same_window_are_not_cross_version_evidence():
    observations=[{"step":1,"ok":True,"tool":"investigate_recent_versions","data":{"versions":[{"news_id":"p1","start_inclusive":0,"end_exclusive":100},{"news_id":"p2","start_inclusive":100,"end_exclusive":200}]}},
                  {"step":2,"ok":True,"tool":"query_reviews","data":{"filters":{"is_positive":None},"reviews":[{"review_id":"r1","text":"卡顿","created_at_timestamp":10},{"review_id":"r2","text":"卡顿","created_at_timestamp":20}]}}]
    citations=[{"source_step":2,"review_id":rid,"quote":"卡顿"} for rid in ("r1","r2")]
    errors,audit=investigation_finish_checks("最近五次更新一直存在的性能问题",observations,citations,False)
    assert errors and audit["cited_window_count"]==1 and audit["semantics_verified"] is False
    observations[1]["data"]["reviews"][1]["created_at_timestamp"]=100
    assert not investigation_finish_checks("最近五次更新一直存在的性能问题",observations,citations,False)[0]
    assert not investigation_finish_checks("最近五次更新一直存在的性能问题",observations,[],True)[0]


def test_missing_timestamp_or_invalid_quote_not_counted():
    observations=[{"step":1,"ok":True,"data":{"reviews":[{"review_id":"x","text":"真实原文"}]}}]
    assert cited_window_audit(observations,[{"source_step":1,"review_id":"x","quote":"虚构"}])["cited_window_count"]==0


def test_explicit_user_patch_not_overridden_by_selected_default():
    state={"patch_news_id":"p5"}
    assert explicit_patch_alignment("比较p1前后好评率", "query_reviews", {"filters":{"period":"after"}},state)
    assert not explicit_patch_alignment("比较p1前后好评率", "query_reviews", {"patch_news_id":"p1","filters":{"period":"after"}},state)
    assert not explicit_patch_alignment("比较p1前后好评率", "query_reviews", {"filters":{"period":"all","start_at":"2026-10-01T00:00:00Z"}},state)


@pytest.mark.parametrize("text", ["没有存档损坏，读档正常", "savegame is corrupted", "更新后读取进度失败"])
def test_topic_gate_preserves_contradictions_and_aliases(text):
    from gamelens.rag.relevance import topic_anchor, topic_supported
    topic=topic_anchor("存档损坏无法读取进度")
    assert topic_supported(topic,text)
    assert not topic_supported(topic,"匹配掉线又等了五分钟")


def test_unknown_or_multiple_topics_do_not_use_strict_anchor():
    from gamelens.rag.relevance import topic_anchor
    assert topic_anchor("冰箱温度") is None
    assert topic_anchor("掉帧和闪退") is None


def test_wrong_scope_not_exposed_as_partial_answer(tmp_path):
    from test_business_evaluation import authored_run
    from gamelens.evaluation.business_cases import CASES
    from gamelens.core.answers import statistical_answer
    case,state,_=authored_run(tmp_path,CASES[1])
    state["goal"]=CASES[0]["goal"]
    assert statistical_answer(state,partial=True) is None


def test_real_snapshot_hash_and_portable_files(tmp_path,monkeypatch):
    from gamelens.evaluation import real_snapshot as real
    from gamelens.storage.atomic_io import atomic_write_json
    materialize_dataset(tmp_path)
    source=tmp_path/"data/raw/730_reviews.json"
    records=load_reviews(source)
    monkeypatch.setattr(real,"PROJECT_ROOT",tmp_path)
    monkeypatch.setattr(real,"IDS",{730:[records[0].review_id,records[1].review_id]})
    root=tmp_path/"frozen"
    manifest=real.prepare(root)
    assert all("\\" not in name for name in manifest["files"])
    assert real.dataset(root,730)["source_hashes"]["reviews"]
    assert real.prepare(root)==manifest
    with (root/"data/raw/730_reviews.json").open("a",encoding="utf-8") as stream:stream.write(" ")
    with pytest.raises(ValueError,match="冻结快照"):
        real.validate_snapshot(root)


def test_absolute_dates_cannot_bypass_explicit_patch(tmp_path):
    from gamelens.tools.investigation import ReviewInvestigation, QueryReviewsArgs
    from gamelens.evaluation.benchmark import initial_state
    materialize_dataset(tmp_path)
    state=initial_state("absolute-patch","比较p1更新前一天和更新后五天评论的好评率")
    state.update(patch_news_id="p5",window_days=1)
    domain=ReviewInvestigation(tmp_path)
    wrong=QueryReviewsArgs(appid=730,filters={"period":"all","start_at":"2026-10-04T00:00:00Z","end_at":"2026-10-04T23:59:59Z"})
    with pytest.raises(ValueError,match="绝对日期"):
        domain.query_reviews(wrong,state)
    before=QueryReviewsArgs(appid=730,filters={"period":"all","start_at":"2026-09-30T00:00:00Z","end_at":"2026-10-01T00:00:00Z"})
    assert domain.query_reviews(before,state).data["stats"]["count"]==4
    after=QueryReviewsArgs(appid=730,patch_news_id="p1",window_days=5,filters={"period":"after"})
    assert domain.query_reviews(after,state).data["stats"]["count"]==11


def test_relative_window_days_must_match_question(tmp_path):
    from gamelens.tools.investigation import ReviewInvestigation,QueryReviewsArgs
    from gamelens.evaluation.benchmark import initial_state
    materialize_dataset(tmp_path)
    state=initial_state("days","p1更新后五天评论数量和推荐率")
    args=QueryReviewsArgs(appid=730,patch_news_id="p1",window_days=1,filters={"period":"after"})
    with pytest.raises(ValueError,match="窗口不一致"):
        ReviewInvestigation(tmp_path).query_reviews(args,state)


def test_patch_id_without_rag_period_is_rejected():
    from gamelens.rag.tools import SearchArgs
    from pydantic import ValidationError
    with pytest.raises(ValidationError,match="period=all"):
        SearchArgs(appid=730,query="掉帧",patch_news_id="p1")
    assert SearchArgs(appid=730,query="掉帧",patch_news_id="p1",filters={"period":"after"})


def test_version_assignment_rejects_wrong_window_and_duplicates():
    from gamelens.core.investigation_checks import validate_version_assignments
    observations=[{"step":1,"ok":True,"tool":"investigate_recent_versions","data":{"versions":[{"news_id":"p1","start_inclusive":100,"end_exclusive":200}]}},
        {"step":2,"ok":True,"data":{"reviews":[{"review_id":"in","text":"掉帧","created_at_timestamp":150},{"review_id":"out","text":"流畅","created_at_timestamp":50}]}}]
    citations=[{"source_step":2,"review_id":"in","quote":"掉帧"},{"source_step":2,"review_id":"out","quote":"流畅"}]
    good=[{"news_id":"p1","review_ids":["in"]},{"news_id":None,"review_ids":["out"]}]
    assert not validate_version_assignments(observations,citations,good)
    assert validate_version_assignments(observations,citations,[{"news_id":"p1","review_ids":["out","in"]}])
    assert validate_version_assignments(observations,citations,good+good)
    assert validate_version_assignments(observations,citations,[]) 


def test_citation_repair_feedback_survives_unnecessary_verify(tmp_path):
    from test_business_evaluation import authored_run
    from test_runtime_optimizations import FakeModel
    from gamelens.evaluation.business_cases import CASES
    from gamelens.evaluation.suite import business_registry
    from gamelens.core.runtime import InvestigationAgent,LLMPlanner
    from gamelens.core.grounding import fact_catalog
    case,state,_=authored_run(tmp_path,CASES[5])
    state["optimization_mode"]="off"
    fact=fact_catalog(state["observations"])[0]["fact_id"]
    bad={"action":"finish","reason":"完成","final_answer":"存在反馈，不能推断根因。","fact_ids":[fact],"news_citations":[{"source_step":1,"news_id":"p3","quote":"不在原文中的内容"}]}
    good=copy.deepcopy(bad)
    good["news_citations"][0]["quote"]="修复部分启动崩溃。"
    # This fixture deliberately only supplies an announcement. The citation
    # repair test must not also claim complete player/counterexample coverage.
    good["scope_limited"]=True
    client=FakeModel([bad,{"action":"verify_investigation","reason":"重复核验","arguments":{"appid":730}},good])
    result=InvestigationAgent(LLMPlanner(client),business_registry(tmp_path,None)).invoke(state)
    assert result["status"]=="completed"
    assert "公告引用不匹配" in client.messages[2][1]["content"]
    assert "exact_news_quote_examples" in client.messages[2][1]["content"]
