"""Offline regression for the small default tool surface and evidence screening."""
import argparse
import json
import shutil
from pathlib import Path

import httpx
import pytest

from gamelens.agents.investigation import build_initial_state, parse_args
from gamelens.core.runtime import InvestigationAgent, LLMPlanner
from gamelens.tools.registry import AppArgs, PrepareReviewsArgs, PreparePatchesArgs, SteamInvestigationTools
from gamelens.rag.tools import RagTools, SearchArgs
from gamelens.rag.screening import screen_candidates
from gamelens.evaluation.benchmark import initial_state
from test_rag import TestEncoder

ROOT = Path(__file__).resolve().parents[1]
FOCUSED_TOOLS = {"inspect_data", "prepare_reviews", "prepare_patches", "list_patches",
                 "query_reviews", "get_review", "compare_cohorts", "verify_investigation",
                 "build_game_index", "search_game_evidence"}
FOCUSED_TOOLS.update({'list_announcements','prepare_announcement_comments','query_announcement_comments','get_announcement_comment'})


@pytest.fixture
def corpus(tmp_path):
    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    for path in (ROOT / "evals/fixtures").glob("*.json"):
        shutil.copyfile(path, raw / path.name)
    state = initial_state(123, "调查本地全部缓存评论的好评率")
    state["investigation_profile"] = "focused"
    state["rag_mode"] = "off"
    return tmp_path, state


class FakeClient:
    def __init__(self, responses=()):
        self.responses = list(responses)
        self.messages = []

    def generate_json(self, messages):
        self.messages.append(messages)
        if not self.responses:
            raise AssertionError("Unexpected model call: no real API allowed")
        return self.responses.pop(0)


def hit(rid, text, positive=False, kind="review"):
    original = {"review_id": rid, "text": text, "is_positive": positive} if kind == "review" else {"news_id": rid, "contents": text}
    return {"kind": kind, "source_id": rid, "text": text, "original": original}


def verdict(rid, role, quote, kind="review"):
    return {"source_id": rid, "kind": kind, "role": role, "quote": quote, "reason": "测试相关性标签，不是模型效果测评"}


def test_small_default_and_explicit_compatibility(corpus):
    root, state = corpus
    tools = SteamInvestigationTools(root)
    assert tools.registry().names == FOCUSED_TOOLS | {'ensure_review_coverage'}
    assert {s['name'] for s in tools.registry().specifications()} == FOCUSED_TOOLS
    extended = tools.registry(profile="extended")
    assert len(extended.names) == 22  # Plus the internal window-acquisition gate.
    assert "classify_review_sample" in extended.names
    assert "investigate_recent_versions" in extended.names
    with pytest.raises(ValueError):
        tools.registry(profile="invalid")
    assert not tools.registry().execute("classify_review_sample", {"appid": 730}, state).ok


def test_inspection_does_not_load_previous_reports(corpus):
    root, state = corpus
    processed = root / "data/processed"
    processed.mkdir()
    (processed / "730_version_analysis_corrupt.json").write_text("not JSON", encoding="utf-8")
    result = SteamInvestigationTools(root).inspect_data(AppArgs(appid=730), state)
    assert set(result.artifacts) == {"reviews", "patches"}


def test_cached_preparation_without_pipeline_or_network(corpus, monkeypatch):
    root, state = corpus
    from gamelens.tools import steam_reviews, steam_news
    review_path = root / "data/raw/730_reviews.json"
    cached = json.loads(review_path.read_text(encoding="utf-8"))
    cached = [item for item in cached if item["language"] == "schinese"]
    review_path.write_text(json.dumps(cached, ensure_ascii=False), encoding="utf-8")
    def forbidden(*args, **kwargs):
        raise AssertionError("Network forbidden")
    monkeypatch.setattr(steam_reviews, "fetch_reviews_snapshot", forbidden)
    monkeypatch.setattr(steam_news, "fetch_news", forbidden)
    tools = SteamInvestigationTools(root)
    expected = len(cached)
    assert tools.prepare_reviews(PrepareReviewsArgs(appid=730,target_count=expected), state).data["review_count"] == expected
    assert tools.prepare_patches(PreparePatchesArgs(appid=730), state).ok
    with pytest.raises(ValueError, match="一致"):
        tools.inspect_data(AppArgs(appid=570), state)


def _raw_review(index):
    timestamp=1_700_000_000+index
    return {'recommendationid':str(90_000+index),'author':{'playtime_forever':1200,'playtime_at_review':600},
        'language':'schinese','review':'缓存充分性测试 '+str(index),'timestamp_created':timestamp,
        'timestamp_updated':timestamp,'voted_up':index%2==0,'votes_up':1,'steam_purchase':True,
        'received_for_free':False,'written_during_early_access':False}


def test_store_review_cache_refetches_when_target_grows_then_reuses_exhausted_source(corpus,monkeypatch):
    root,state=corpus
    calls=[]
    def fetched(**kwargs):
        calls.append(kwargs)
        return {'reviews':[_raw_review(i) for i in range(30)],'source_exhausted':True,
            'stop_reason':'source_exhausted'}
    monkeypatch.setattr('gamelens.tools.steam_reviews.fetch_reviews_snapshot',fetched)
    tools=SteamInvestigationTools(root)
    first=tools.prepare_reviews(PrepareReviewsArgs(appid=730,target_count=1000),state)
    second=tools.prepare_reviews(PrepareReviewsArgs(appid=730,target_count=1000),state)
    assert len(calls)==1 and first.data['review_count']==30 and not first.data['reused']
    assert first.data['refresh_attempted'] and first.data['source_exhausted']
    assert second.data['reused'] and second.data['cache_adequate']
    metadata=json.loads((root/'data/raw/730_reviews.coverage.json').read_text(encoding='utf-8'))
    assert metadata['collected_count']==30 and metadata['source_exhausted']


def test_store_review_failed_improvement_does_not_replace_better_cache(corpus,monkeypatch):
    root,state=corpus
    path=root/'data/raw/730_reviews.json'
    existing=[item for item in json.loads(path.read_text(encoding='utf-8')) if item['language']=='schinese']
    path.write_text(json.dumps(existing,ensure_ascii=False),encoding='utf-8')
    original=path.read_bytes();old_count=len(existing)
    monkeypatch.setattr('gamelens.tools.steam_reviews.fetch_reviews_snapshot',lambda **_:{
        'reviews':[_raw_review(i) for i in range(5)],'source_exhausted':False,'stop_reason':'cursor_stalled'})
    result=SteamInvestigationTools(root).prepare_reviews(PrepareReviewsArgs(appid=730,target_count=1000),state)
    assert result.data['review_count']==old_count and result.data['retained_previous_snapshot']
    assert result.data['refresh_error_code']=='cursor_stalled' and path.read_bytes()==original


def test_wrong_language_legacy_store_cache_is_not_used_as_failure_fallback(corpus,monkeypatch):
    root,state=corpus
    path=root/'data/raw/730_reviews.json';original=path.read_bytes()
    monkeypatch.setattr('gamelens.tools.steam_reviews.fetch_reviews_snapshot',
        lambda **_:(_ for _ in ()).throw(httpx.ReadTimeout('offline')))
    with pytest.raises(httpx.ReadTimeout):
        SteamInvestigationTools(root).prepare_reviews(
            PrepareReviewsArgs(appid=730,target_count=1),state)
    assert path.read_bytes()==original


def test_store_review_invalid_sidecar_forces_refresh(corpus,monkeypatch):
    root,state=corpus
    path=root/'data/raw/730_reviews.json'
    existing=[item for item in json.loads(path.read_text(encoding='utf-8')) if item['language']=='schinese']
    path.write_text(json.dumps(existing,ensure_ascii=False),encoding='utf-8')
    (root/'data/raw/730_reviews.coverage.json').write_text(json.dumps({
        'schema_version':1,'appid':730,'language':'schinese','collected_count':len(existing),
        'source_exhausted':True,'review_file_sha256':'wrong'}),encoding='utf-8')
    calls=[]
    def fetched(**kwargs):
        calls.append(kwargs)
        return {'reviews':[_raw_review(i) for i in range(30)],'source_exhausted':True,
            'stop_reason':'source_exhausted'}
    monkeypatch.setattr('gamelens.tools.steam_reviews.fetch_reviews_snapshot',fetched)
    result=SteamInvestigationTools(root).prepare_reviews(
        PrepareReviewsArgs(appid=730,target_count=max(1,len(existing))),state)
    assert len(calls)==1 and result.data['review_count']==30
    assert not result.data['reused'] and result.data['cache_adequate']


def test_store_review_refresh_exception_retains_valid_prior(corpus,monkeypatch):
    root,state=corpus
    monkeypatch.setattr('gamelens.tools.steam_reviews.fetch_reviews_snapshot',lambda **_:{
        'reviews':[_raw_review(i) for i in range(30)],'source_exhausted':True,
        'stop_reason':'source_exhausted'})
    tools=SteamInvestigationTools(root)
    first=tools.prepare_reviews(PrepareReviewsArgs(appid=730,target_count=1000),state)
    path=root/'data/raw/730_reviews.json';original=path.read_bytes()
    def timeout(**_):
        raise httpx.ReadTimeout('offline')
    monkeypatch.setattr('gamelens.tools.steam_reviews.fetch_reviews_snapshot',timeout)
    second=tools.prepare_reviews(PrepareReviewsArgs(appid=730,target_count=1000,refresh=True),state)
    assert first.data['review_count']==30 and second.data['review_count']==30
    assert second.data['reused'] and second.data['retained_previous_snapshot']
    assert second.data['refresh_error_code']=='request_timeout' and path.read_bytes()==original


def test_explicit_store_refresh_replaces_equal_sized_snapshot_when_source_grew(corpus,monkeypatch):
    root,state=corpus
    monkeypatch.setattr('gamelens.tools.steam_reviews.fetch_reviews_snapshot',lambda **_:{
        'reviews':[_raw_review(i) for i in range(30)],'source_exhausted':True,
        'stop_reason':'source_exhausted'})
    tools=SteamInvestigationTools(root)
    tools.prepare_reviews(PrepareReviewsArgs(appid=730,target_count=30),state)
    path=root/'data/raw/730_reviews.json';before=path.read_bytes()
    monkeypatch.setattr('gamelens.tools.steam_reviews.fetch_reviews_snapshot',lambda **_:{
        'reviews':[_raw_review(100+i) for i in range(30)],'source_exhausted':False,
        'stop_reason':'target_reached'})
    result=tools.prepare_reviews(PrepareReviewsArgs(appid=730,target_count=30,refresh=True),state)
    assert not result.data['reused'] and result.data['cache_adequate']
    assert path.read_bytes()!=before


def test_cache_candidate_prefers_larger_adequate_refresh_but_not_equal_rewrite():
    from gamelens.tools.cache_coverage import candidate_improves_cache
    assert candidate_improves_cache(100,500,500,existing_complete=True,candidate_complete=False)
    assert candidate_improves_cache(100,50,500,existing_complete=False,candidate_complete=True)
    assert not candidate_improves_cache(200,500,500,
        existing_error=None,candidate_error='counts_changed_during_fetch')
    assert not candidate_improves_cache(100,100,500)


def test_defaults_and_focused_memory_do_not_inherit_legacy_finish(monkeypatch):
    monkeypatch.setattr("sys.argv", ["investigation_agent.py", "--appid", "730", "--goal", "调查更新"])
    args = parse_args()
    assert args.profile == "focused"
    state = build_initial_state(args, {"inherited_artifacts": {"reviews": "r.json", "patches": "p.json",
        "classifications": "c.json", "impact_summary": "i.json", "final_report": "old.md", "evidence_assessed": "true"}})
    assert state["investigation_profile"] == "focused"
    assert state["artifacts"] == {"reviews": "r.json", "patches": "p.json"}
    from gamelens.api.app import RunRequest
    assert RunRequest(appid=730).investigation_profile == "focused"
    assert RunRequest(appid=730, investigation_profile="extended").investigation_profile == "extended"


def test_focused_memory_uses_patch_identity_not_old_report(corpus):
    root, _ = corpus
    old = root / "corrupt.json"
    old.write_text("invalid historical report", encoding="utf-8")
    args = argparse.Namespace(thread_id="followup", goal="这些闪退呢", appid=730, profile="focused",
        window_days=7, review_count=100, news_count=20, classify_count=10,
        patch_news_id=None, refresh_data=False, force_reclassify=False, max_steps=5)
    state = build_initial_state(args, {"selected_patch_news_id":"p1", "inherited_artifacts":{"version_analysis":str(old)}})
    assert state["patch_news_id"] == "p1" and not state["artifacts"]


def test_multi_version_request_not_silently_downgraded(corpus):
    root, state = corpus
    state["goal"] = "调查近五次更新一直存在的闪退"
    client = FakeClient()
    result = InvestigationAgent(LLMPlanner(client), SteamInvestigationTools(root, client).registry()).invoke(state)
    assert result["status"] == "failed"
    assert "extended" in result["error"]
    assert client.messages == []


def test_simple_stats_skip_rag_classification_and_extra_model_calls(corpus):
    root, state = corpus
    client = FakeClient([{"action": "query_reviews", "reason": "获取全部缓存统计",
                          "arguments": {"appid": 730, "filters": {"period": "all"}}}])
    result = InvestigationAgent(LLMPlanner(client), SteamInvestigationTools(root, client).registry()).invoke(state)
    assert result["status"] == "completed"
    assert result["answer_origin"] == "deterministic_statistics"
    assert [o["tool"] for o in result["observations"]] == ["inspect_data", "query_reviews", "verify_investigation"]
    assert len(client.messages) == 1
    context = json.loads(client.messages[0][1]["content"].split("\n", 1)[1])
    assert "classify_count" not in context["hard_constraints"]
    assert "version_evidence" not in context["planner_output_schema"]["properties"]
    assert set(t["name"] for t in context["available_tools"]) == FOCUSED_TOOLS - {
        "build_game_index", "search_game_evidence", "prepare_announcement_comments",
        "query_announcement_comments", "get_announcement_comment"}
    assert "```json" not in Path(result["artifacts"]["final_report"]).read_text(encoding="utf-8")


def test_screening_keeps_positive_complaint_counterexample_and_uncertainty():
    hits = [hit("a", "推荐，但更新后闪退", True), hit("b", "更新后不再闪退", True),
            hit("c", "有时会闪退，版本不明确"), hit("d", "皮肤太贵")]
    payload = {"verdicts": [verdict("a", "support", "更新后闪退"),
        verdict("b", "counterexample", "更新后不再闪退"), verdict("c", "partial", "有时会闪退"),
        verdict("d", "irrelevant", "")]}
    client = FakeClient([payload])
    selected, metadata = screen_candidates(client, "调查更新后的闪退", "crash", hits)
    assert [h["source_id"] for h in selected] == ["a", "b", "c"]
    assert [h["evidence_role"] for h in selected] == ["support", "counterexample", "partial"]
    assert metadata["candidate_count"] == 4 and metadata["selected_count"] == 3
    assert metadata["semantic_accuracy_verified"] is False
    assert len(client.messages) == 1


@pytest.mark.parametrize("bad", [
    {"verdicts": [verdict("unknown", "support", "闪退")]},
    {"verdicts": []},
    {"verdicts": [verdict("a", "support", "闪退"), verdict("a", "support", "闪退")]},
])
def test_invalid_selection_bounded_and_not_silently_accepted(bad):
    client = FakeClient([bad, bad])
    with pytest.raises(ValueError, match="连续两次"):
        screen_candidates(client, "闪退调查", "闪退", [hit("a", "闪退")])
    assert len(client.messages) == 2


@pytest.mark.parametrize('quote',['已经确认硬件损坏',''])
def test_invalid_quote_is_explicitly_rejected_without_fabricating_evidence(quote):
    bad={'verdicts':[verdict('a','support',quote)]}
    client=FakeClient([bad,bad])
    selected,metadata=screen_candidates(client,'闪退调查','闪退',[hit('a','闪退')])
    assert selected==[] and metadata['verdicts']==[]
    assert metadata['error_code']=='screening_quote_validation_failed'
    assert metadata['rejected_candidates'][0]['source_id']=='a'
    assert len(client.messages)==2


def test_empty_screening_has_no_call_and_news_quote_works():
    client = FakeClient()
    assert screen_candidates(client, "调查", "问题", [])[1]["attempts"] == 0
    assert not client.messages
    client = FakeClient([{"verdicts": [verdict("n1", "partial", "修复崩溃", "announcement")]}])
    selected, _ = screen_candidates(client, "玩家是否仍崩溃", "崩溃", [hit("n1", "本次修复崩溃", kind="announcement")])
    assert selected[0]["evidence_role"] == "partial"


def test_rag_search_runs_internal_screening_and_keeps_audit(corpus):
    root, state = corpus
    state.update(rag_mode="on", goal="调查全部缓存评论中的闪退")
    class EchoClient:
        def generate_json(self, messages):
            data = json.loads(messages[1]["content"])
            return {"verdicts": [verdict(c["source_id"], "partial", c["excerpt"][:100], c["kind"]) for c in data["candidates"]]}
    rag = RagTools(root, TestEncoder(), client=EchoClient())
    rag.index.build(730, state)
    result = rag.search_game_evidence(SearchArgs(appid=730, query="crash", kind="reviews"), state)
    assert result.data["screening"]["enabled"]
    assert result.data["screening"]["attempts"] == 1
    assert result.data["reviews"]
    assert all(h["evidence_role"] == "partial" for h in result.data["hits"])
    assert json.loads(Path(result.artifacts["investigation_search_game_evidence"]).read_text(encoding="utf-8"))["screening"]["enabled"]
