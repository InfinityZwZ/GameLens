"""Controlled channel/routing tests; not real-model relevance measurements."""
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from gamelens.core.runtime import InvestigationAgent, LLMPlanner, PlannerDecision
from gamelens.evaluation.benchmark import initial_state
from gamelens.memory.context import PlannerContextBuilder
from gamelens.rag.embedding import CONFIG
from gamelens.rag.index import GameIndex
from gamelens.rag.tools import RagTools, SearchArgs, MAX_EVIDENCE_SEARCHES
from gamelens.storage.atomic_io import atomic_write_json
from gamelens.tools.investigation import (
    MetadataReviewFilter, ReviewFilter, QueryReviewsArgs, ConditionQueryArgs,
    ConditionCohortsArgs, ReviewInvestigation,
)
from gamelens.tools.registry import SteamInvestigationTools
from gamelens.tools.argument_normalization import normalize_search_arguments

ROOT=Path(__file__).resolve().parents[1]


class ControlledEncoder:
    """Query mapping is hand-set, so tests only check algorithm eligibility."""
    def __init__(self,mapping):
        self.mapping=mapping
        self.calls=[]

    def encode(self,texts):
        self.calls.extend(texts)
        vectors=[]
        for text in texts:
            group=self.mapping.get(text,1)
            vector=[0.0]*CONFIG["dimension"]
            vector[group]=1.0
            vectors.append(vector)
        return vectors


def setup_index(root,reviews,news=(),mapping=None):
    review_base=json.loads((ROOT/"evals/fixtures/730_reviews.json").read_text(encoding="utf-8"))[0]
    news_base=json.loads((ROOT/"evals/fixtures/730_patch_candidates.json").read_text(encoding="utf-8"))[0]
    raw=root/"data/raw"
    atomic_write_json(raw/"730_reviews.json",[{**review_base,"review_id":rid,"text":text,
        "is_positive":positive} for rid,text,positive in reviews])
    announcements=[{**news_base,"news_id":nid,"title":text,"contents":text} for nid,text in news]
    atomic_write_json(raw/"730_patch_candidates.json",announcements)
    atomic_write_json(raw/"730_news.json",announcements)
    state=initial_state(89,"查找目标问题证据")
    state.update(investigation_profile="focused",rag_mode="auto")
    encoder=ControlledEncoder(mapping or {})
    index=GameIndex(root,encoder);index.build(730,state)
    return index,state,encoder


def test_lexical_candidate_survives_dense_threshold(tmp_path):
    index,state,_=setup_index(tmp_path,[("a","DXGI_ERROR_DEVICE_HUNG",False)],mapping={"DXGI_ERROR_DEVICE_HUNG":1,"查找设备错误":0})
    hits,_=index.search(SearchArgs(appid=730,query="查找设备错误",lexical_terms=["DXGI_ERROR_DEVICE_HUNG"],kind="reviews"),state)
    assert len(hits)==1 and hits[0]["dense_score"]==0
    assert hits[0]["retrieval_channels"]==["lexical"]


def test_semantic_candidate_needs_no_query_or_topic_words(tmp_path):
    text="一加载地图就回到桌面，游戏进程没了"
    index,state,_=setup_index(tmp_path,[("a",text,True)],mapping={text:0,"闪退":0})
    hits,_=index.search(SearchArgs(appid=730,query="闪退",kind="reviews"),state)
    assert len(hits)==1 and hits[0]["source_id"]=="a"
    assert hits[0]["retrieval_channels"]==["dense"]
    assert hits[0]["topic_hint_match"] is False


def test_low_dense_and_no_lexical_hit_still_abstains(tmp_path):
    index,state,_=setup_index(tmp_path,[("a","匹配队友等待太久",False)],mapping={"目标存档损坏":0})
    hits,_=index.search(SearchArgs(appid=730,query="目标存档损坏",kind="reviews"),state)
    assert hits==[]


def test_soft_term_expansion_does_not_hard_filter_other_hits(tmp_path):
    text="进图立刻回到桌面"
    index,state,_=setup_index(tmp_path,[("a",text,False)],mapping={text:0,"启动问题":0})
    hits,_=index.search(SearchArgs(appid=730,query="启动问题",lexical_terms=["crash"],kind="reviews"),state)
    assert hits[0]["source_id"]=="a" and hits[0]["lexical_score"]==0


def test_rule_query_expansion_is_bounded_and_preserves_hard_filters():
    arguments={"appid":730,"query":"更新后性能卡顿","kind":"reviews",
               "filters":{"period":"after","is_positive":False}}
    normalized,changes=normalize_search_arguments(arguments,{"goal":"调查更新后性能卡顿"})
    assert normalized["filters"]==arguments["filters"]
    assert 1<=len(normalized["lexical_terms"])<=8
    assert "stutter" in normalized["lexical_terms"]
    assert any(change["rule"]=="deterministic_topic_expansion" for change in changes)


def test_metadata_hard_filters_remain_before_channels(tmp_path):
    index,state,_=setup_index(tmp_path,[("a","crash",False),("b","crash",True)],mapping={"crash":0})
    hits,_=index.search(SearchArgs(appid=730,query="crash",kind="reviews",filters=MetadataReviewFilter(period="all",is_positive=True)),state)
    assert [h["source_id"] for h in hits]==["b"]


def test_both_source_kinds_have_candidate_allocation(tmp_path):
    index,state,_=setup_index(tmp_path,[("a","crash",False)],[(f"n{i}","crash fix") for i in range(8)],mapping={"crash":0})
    hits,_=index.search(SearchArgs(appid=730,query="crash",kind="both",top_k=4),state)
    assert len(hits)==4
    assert {h["kind"] for h in hits}=={"review","announcement"}
    assert len({(h["kind"],h["source_id"]) for h in hits})==4
    only_news,_=index.search(SearchArgs(appid=730,query="crash",kind="announcements"),state)
    assert all(h["kind"]=="announcement" for h in only_news)


def test_reranker_receives_hard_filtered_pool_before_top_k(tmp_path):
    reviews=[("negative-a","crash alpha",False),("negative-b","crash beta",False),
             ("positive","crash praise",True)]
    mapping={text:0 for _,text,_ in reviews}|{"crash":0}
    _,state,encoder=setup_index(tmp_path,reviews,mapping=mapping)

    class CaptureReranker:
        ready=True
        def rerank(self,query,hits):
            self.seen=list(hits)
            ordered=sorted(hits,key=lambda hit:hit["source_id"]!="negative-b")
            return [{**hit,"reranker_score":1.0-rank/10,"hybrid_rank":rank+1}
                    for rank,hit in enumerate(ordered)]

    reranker=CaptureReranker()
    state["investigation_profile"]="extended"
    tools=RagTools(tmp_path,encoder,reranker=reranker,reranker_mode="required")
    args=SearchArgs(appid=730,query="crash",kind="reviews",top_k=1,
                    filters=MetadataReviewFilter(period="all",is_positive=False))
    result=tools.search_game_evidence(args,state)

    assert {hit["source_id"] for hit in reranker.seen}=={"negative-a","negative-b"}
    assert result.data["returned_count"]==1
    assert result.data["reranking"]["status"]=="applied"
    assert result.data["hits"][0]["source_id"]=="negative-b"


def test_auto_reranker_fallback_is_visible_in_result_and_summary(tmp_path):
    _,state,encoder=setup_index(tmp_path,[("a","crash",False)],mapping={"crash":0})
    state["investigation_profile"]="extended"
    result=RagTools(tmp_path,encoder,reranker_mode="auto").search_game_evidence(
        SearchArgs(appid=730,query="crash",kind="reviews"),state)
    assert result.data["reranking"]["status"]=="hybrid_fallback"
    assert result.data["reranking"]["reason_code"]=="reranker_unavailable"
    assert "reranker_unavailable" in result.summary


@pytest.mark.parametrize("model,argument",[
    (ConditionQueryArgs,{"appid":730,"filters":{"keywords":["crash"]}}),
    (ConditionCohortsArgs,{"appid":730,"cohort_a":{"keywords":["crash"]},"cohort_b":{}}),
    (SearchArgs,{"appid":730,"query":"crash","filters":{"keywords":["crash"]}}),
])
def test_no_standalone_keyword_condition_in_focused_contract(model,argument):
    with pytest.raises(ValidationError,match="条件过滤"):
        model.model_validate(argument)


def test_legacy_literal_tool_preserved_but_focused_direct_call_rejected(tmp_path):
    _,state,_=setup_index(tmp_path,[("a","crash",False)])
    args=QueryReviewsArgs(appid=730,filters=ReviewFilter(period="all",keywords=["crash"]))
    domain=ReviewInvestigation(tmp_path)
    with pytest.raises(ValueError,match="统一使用"):
        domain.query_reviews(args,state)
    state["investigation_profile"]="extended"
    assert domain.query_reviews(args,state).data["stats"]["count"]==1


def test_empty_legacy_filter_fields_are_checkpoint_compatible():
    args=SearchArgs(appid=730,query="crash",filters=ReviewFilter(period="all"))
    assert "keywords" not in args.filters.model_dump()
    assert "keyword_mode" not in args.filters.model_dump()
    with pytest.raises(ValidationError):
        SearchArgs(appid=730,query="crash",lexical_terms=[" "])


def test_search_budget_blocks_tool_and_planner(tmp_path):
    _,state,encoder=setup_index(tmp_path,[("a","crash",False)])
    state["observations"]=[{"step":i+1,"tool":"search_game_evidence","ok":False,
        "arguments":{"query":"crash"},"data":{}} for i in range(MAX_EVIDENCE_SEARCHES)]
    with pytest.raises(ValueError,match="上限"):
        RagTools(tmp_path,encoder).search_game_evidence(SearchArgs(appid=730,query="crash"),state)
    class Planner:
        def decide(self,state,tools):
            return PlannerDecision(action="search_game_evidence",reason="再查一次",arguments={"appid":730,"query":"crash"})
    result=InvestigationAgent(Planner(),SteamInvestigationTools(tmp_path).registry())._decide(state)
    assert result["status"]=="repairing" and "预算已用尽" in result["validation_errors"][0]
    context=PlannerContextBuilder().build(state,[])
    assert context["search_progress"]["remaining_searches"]==0


def test_scoped_candidates_screened_and_all_rejections_recorded(tmp_path):
    index,state,encoder=setup_index(tmp_path,[("a","地图回到桌面",False)],mapping={"地图回到桌面":0,"闪退":0})
    class RejectClient:
        def generate_json(self,messages):
            inputs=json.loads(messages[1]["content"])["candidates"]
            return {"verdicts":[{"kind":i["kind"],"source_id":i["source_id"],"role":"irrelevant","quote":"","reason":"测试拒绝，不代表真实模型效果"} for i in inputs]}
    result=RagTools(tmp_path,encoder,client=RejectClient()).search_game_evidence(SearchArgs(appid=730,query="闪退",kind="reviews"),state)
    assert result.data["candidate_summary"]["count"]==1
    assert result.data["returned_count"]==0
    assert result.data["screening"]["verdicts"][0]["source_id"]=="a"
    assert result.data["evidence_coverage"]["missing_kinds"]==["review"]
    assert result.data["retrieval_policy"]["hard_topic_gate"] is False


def test_planner_context_exposes_unified_tools_and_search_history(tmp_path):
    _,state,_=setup_index(tmp_path,[("a","crash",False)])
    state["observations"]=[{"step":1,"tool":"search_game_evidence","ok":True,
        "arguments":{"query":"启动闪退","kind":"reviews","filters":{"period":"after"}},
        "data":{"evidence_coverage":{"missing_kinds":[],"roles":{"counterexample":0}}}}]
    class Capture:
        def generate_json(self,messages):
            self.messages=messages
            return {"action":"search_game_evidence","reason":"补充对局方面","arguments":{"appid":730,"query":"对局时闪退","kind":"reviews","lexical_terms":["crash"]}}
    capture=Capture();registry=SteamInvestigationTools(tmp_path).registry()
    decision=LLMPlanner(capture).decide(state,registry.specifications())
    context=json.loads(capture.messages[1]["content"].split("\n",1)[1])
    query_schema=next(t["arguments_schema"] for t in registry.specifications() if t["name"]=="query_reviews")
    assert "keywords" not in json.dumps(query_schema)
    assert context["search_progress"]["queries"][0]["kind"]=="reviews"
    assert context["search_progress"]["remaining_searches"]==3
    assert decision.arguments["lexical_terms"]==["crash"]
    assert "唯一" in next(t["description"] for t in registry.specifications() if t["name"]=="search_game_evidence")
