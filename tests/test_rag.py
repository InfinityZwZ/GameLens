import json
import math
import shutil
from pathlib import Path

import pytest
from gamelens.rag.embedding import CONFIG
from gamelens.rag.index import GameIndex, chunk
from gamelens.rag.text import clean_retrieval_text, sentence_chunks
from gamelens.rag.tools import RagTools, IndexArgs, SearchArgs
from gamelens.tools.investigation import ReviewFilter, VerifyInvestigationArgs, ReviewInvestigation
from gamelens.core.grounding import validate_news_citations, validate_grounding
from gamelens.core.runtime import AgentTool, ToolRegistry, InvestigationAgent, PlannerDecision
from gamelens.evaluation.benchmark import initial_state
from test_agent_runtime import SequencePlanner

ROOT=Path(__file__).resolve().parents[1]


class TestEncoder:
    """Deterministic test vectors; not a model quality benchmark."""
    __test__=False
    def __init__(self): self.calls=[]; self.fail=False
    def encode(self,texts):
        if self.fail: raise RuntimeError("simulated encoding interruption")
        self.calls.extend(texts)
        result=[]
        for text in texts:
            vector=[0.0]*CONFIG["dimension"]
            group=0 if any(t in text.casefold() for t in ["崩溃","闪退","crash","launch"]) else 1
            if "unrelated" in text: group=2
            vector[group]=1.0
            result.append(vector)
        return result


@pytest.fixture
def corpus(tmp_path):
    raw=tmp_path/"data"/"raw"; raw.mkdir(parents=True)
    for path in (ROOT/"evals"/"fixtures").glob("*.json"): shutil.copyfile(path,raw/path.name)
    return tmp_path,initial_state(99,"寻找无法启动的反馈"),TestEncoder()


def test_index_reuse_update_delete_and_metadata_only_change(corpus):
    root,state,encoder=corpus; index=GameIndex(root,encoder)
    assert index.build(730,state)["embedded_chunks"]>0
    count=len(encoder.calls)
    assert index.build(730,state)["reused"]
    assert len(encoder.calls)==count
    path=root/"data"/"raw"/"730_reviews.json"
    records=json.loads(path.read_text(encoding="utf-8"))
    records[0]["votes_up"]+=1
    path.write_text(json.dumps(records),encoding="utf-8")
    assert index.status(730,state)["status"]=="stale"
    assert index.build(730,state)["embedded_chunks"]==0
    records[0]["text"]="new launch failure content"
    path.write_text(json.dumps(records),encoding="utf-8")
    assert index.build(730,state)["embedded_chunks"]==1
    records.pop(0);path.write_text(json.dumps(records),encoding="utf-8")
    assert index.build(730,state)["embedded_chunks"]==0
    with index.connect(730) as con:
        assert not any(json.loads(r[0])["source_id"]=="r0" for r in con.execute("SELECT record FROM chunks"))


def test_game_isolation_missing_stale_and_filters(corpus):
    root,state,encoder=corpus; index=GameIndex(root,encoder)
    with pytest.raises(ValueError,match="索引缺失"): index.search(SearchArgs(appid=730,query="crash"),state)
    index.build(730,state)
    args=SearchArgs(appid=730,query="crash",kind="reviews",filters=ReviewFilter(period="after",language="schinese",min_playtime_hours=100))
    hits,bounds=index.search(args,state)
    for hit in hits:
        assert hit["original"]["language"]=="schinese"
        assert hit["original"]["playtime_at_review_minutes"]>=6000
        assert hit["original"]["created_at_timestamp"]>=bounds["start_inclusive"]
    with pytest.raises(ValueError,match="当前调查"): index.sources(999,state)
    assert not index.location(999).exists()
    path=root/"data"/"raw"/"730_reviews.json"; path.write_text(path.read_text(encoding="utf-8")+" ",encoding="utf-8")
    with pytest.raises(ValueError,match="过期"): index.search(args,state)


def test_semantic_paraphrase_is_not_blocked_by_topic_vocabulary(corpus):
    root,state,_=corpus
    path=root/"data/raw/730_reviews.json"
    records=json.loads(path.read_text(encoding="utf-8"))
    records[0]["text"]="一加载地图就回到桌面，游戏进程没了"
    path.write_text(json.dumps(records),encoding="utf-8")
    class ParaphraseEncoder(TestEncoder):
        def encode(self,texts):
            vectors=[]
            for text in texts:
                vector=[0.0]*CONFIG["dimension"]
                vector[0 if text=="闪退" or "回到桌面" in text else 1]=1.0
                vectors.append(vector)
            return vectors
    index=GameIndex(root,ParaphraseEncoder());index.build(730,state)
    hits,_=index.search(SearchArgs(appid=730,query="闪退",kind="reviews"),state)
    found=next(hit for hit in hits if "回到桌面" in hit["text"])
    assert found["retrieval_channels"]==["dense"]
    assert found["topic_hint_match"] is False


def test_failed_update_preserves_old_index_and_can_resume(corpus):
    root,state,encoder=corpus;index=GameIndex(root,encoder);index.build(730,state)
    with index.connect(730) as con: previous=index.manifest(con)
    path=root/"data"/"raw"/"730_reviews.json";records=json.loads(path.read_text(encoding="utf-8"))
    records[0]["text"]="brand new original";path.write_text(json.dumps(records),encoding="utf-8")
    encoder.fail=True
    with pytest.raises(RuntimeError):index.build(730,state)
    with index.connect(730) as con:assert index.manifest(con)==previous
    encoder.fail=False
    assert index.build(730,state)["embedded_chunks"]==1


def test_chunks_are_exact_substrings():
    text="公告原文"*500
    for offset,part in chunk(text): assert text[offset:offset+len(part)]==part


def test_retrieval_cleaning_preserves_original_and_quote_offsets(corpus):
    root,state,_=corpus
    path=root/"data/raw/730_reviews.json"
    records=json.loads(path.read_text(encoding="utf-8"))
    records[0]["text"]="<b>Crash</b>   [spoiler]回到桌面[/spoiler] &amp; freeze"
    path.write_text(json.dumps(records),encoding="utf-8")
    index=GameIndex(root,TestEncoder())
    chunks,_=index.corpus(730,state)
    record=next(item for item in chunks if item["source_id"]==records[0]["review_id"])
    assert clean_retrieval_text(records[0]["text"])=="Crash 回到桌面 & freeze"
    assert record["text"]==record["original"]["text"][record["offset"]:record["offset"]+len(record["text"])]
    assert "<b>" not in record["embedding_text"]


def test_sentence_chunks_are_exact_independent_source_spans():
    source="第一句崩溃。第二句仍然卡顿！Third sentence crashes? 最后一段。"
    parts=list(sentence_chunks(source,max_chars=16,overlap_sentences=1))
    assert parts
    assert all(source[offset:offset+len(part)]==part for offset,part in parts)
    assert all(len(part)<=16 for _,part in parts)


def test_tokenizer_limit_is_checked_before_embedding(corpus):
    root,state,_=corpus
    class OverflowEncoder(TestEncoder):
        def token_count(self,text): return CONFIG["max_seq_length"]+1
    encoder=OverflowEncoder()
    with pytest.raises(ValueError,match="tokenizer上限"):
        GameIndex(root,encoder).build(730,state)
    assert encoder.calls==[]


def test_completed_embedding_batches_are_reused_after_interruption(corpus):
    root,state,encoder=corpus;index=GameIndex(root,encoder);index.build(730,state)
    path=root/"data/raw/730_reviews.json";records=json.loads(path.read_text(encoding="utf-8"))
    for i in range(20):records[i]["text"]=f"brand new distinct content {i}"
    path.write_text(json.dumps(records),encoding="utf-8")
    class FailSecond(TestEncoder):
        def __init__(self):super().__init__();self.batches=0
        def encode(self,texts):
            self.batches+=1
            if self.batches==2:raise RuntimeError("stopped after first committed batch")
            return super().encode(texts)
    interrupted=FailSecond();index.encoder=interrupted
    with pytest.raises(RuntimeError):index.build(730,state)
    assert len(interrupted.calls)==16
    index.encoder=encoder
    assert index.build(730,state)["embedded_chunks"]==4


def test_approval_with_missing_data_defers_error_to_execution(tmp_path):
    rag=RagTools(tmp_path,TestEncoder())
    state=initial_state(1,"调查启动问题")
    assert rag.approval(IndexArgs(appid=730),state) is None
    with pytest.raises(ValueError,match="没有本地"):rag.build_game_index(IndexArgs(appid=730),state)


def test_no_hits_and_news_quote_verification(corpus):
    root,state,encoder=corpus;rag=RagTools(root,encoder);rag.build_game_index(IndexArgs(appid=730),state)
    result=rag.search_game_evidence(SearchArgs(appid=730,query="unrelated"),state)
    assert result.data["returned_count"]==0
    result=rag.search_game_evidence(SearchArgs(appid=730,query="update",kind="announcements"),state)
    assert result.data["announcements"]
    news=result.data["announcements"][0]
    observation={"step":1,"ok":True,"data":result.data}
    citation={"source_step":1,"news_id":news["news_id"],"quote":(news["contents"] or news["title"])[:30]}
    assert not validate_news_citations([observation],[citation])
    citation["quote"]="伪造开发者承认因果"
    assert validate_news_citations([observation],[citation])
    state["artifacts"].update(result.artifacts)
    assert ReviewInvestigation(root).verify_investigation(VerifyInvestigationArgs(appid=730),state).ok


def test_off_backend_guard_and_on_finish_gate(corpus):
    root,state,encoder=corpus;rag=RagTools(root,encoder)
    registry=ToolRegistry([AgentTool("build_game_index","build",IndexArgs,rag.build_game_index),AgentTool("search_game_evidence","search",SearchArgs,rag.search_game_evidence)])
    state["rag_mode"]="off"
    assert not registry.execute("build_game_index",{"appid":730},state).ok
    assert not rag.index.location(730).exists()
    state["rag_mode"]="on"
    state["artifacts"]={"final_report":"example","evidence_assessed":"true"}
    agent=InvestigationAgent(SequencePlanner([PlannerDecision(action="finish",reason="完成")]),registry)
    assert agent._decide(state)["next_action"]=="build_game_index"


def test_agent_retrieval_verify_finish_without_paid_model(corpus):
    root,state,encoder=corpus;rag=RagTools(root,encoder);domain=ReviewInvestigation(root)
    registry=ToolRegistry([AgentTool("build_game_index","build",IndexArgs,rag.build_game_index),
        AgentTool("search_game_evidence","search",SearchArgs,rag.search_game_evidence),
        AgentTool("verify_investigation","verify",VerifyInvestigationArgs,domain.verify_investigation)])
    planner=SequencePlanner([PlannerDecision(action="build_game_index",reason="语义反馈需要增强检索",arguments={"appid":730}),
        PlannerDecision(action="search_game_evidence",reason="检索启动问题",arguments={"appid":730,"query":"crash","kind":"reviews","top_k":1}),
        PlannerDecision(action="verify_investigation",reason="核验",arguments={"appid":730}),
        PlannerDecision(action="finish",reason="完成",final_answer="找到1条候选，非总体占比。",claims=[{"label":"候选数","source_step":2,"fact_path":["returned_count"],"value":1}])])
    result=InvestigationAgent(planner,registry).invoke(state)
    assert result["status"]=="completed"
    assert Path(result["artifacts"]["final_report"]).is_file()


def test_rag_mode_api_validation():
    from gamelens.api.app import RunRequest
    from pydantic import ValidationError
    assert RunRequest(appid=730).rag_mode=="auto"
    with pytest.raises(ValidationError):RunRequest(appid=730,rag_mode="invalid")
