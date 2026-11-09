"""No real API calls: exercise hybrid statistics and persistent generation recovery."""
import copy
import json
import sqlite3
from pathlib import Path

import httpx
import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from gamelens.core.answers import simple_statistics_goal, statistical_answer
from gamelens.core.grounding import fact_catalog
from gamelens.core.presentation import reader_text
from gamelens.core.runtime import InvestigationAgent, LLMPlanner, PlannerDecision
from gamelens.evaluation.business_cases import CASES, materialize_dataset
from gamelens.evaluation.quality import BusinessCase, evaluate_business_run
from gamelens.evaluation.suite import business_registry, evaluation_tags, load_dataset, task_state
from gamelens.memory.context import PlannerContextBuilder, compact_schema
from test_business_evaluation import authored_run


class FakeModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0
        self.messages = []

    def generate_json(self, messages):
        self.calls += 1
        self.messages.append(messages)
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        return result


def fixture(tmp_path, spec=CASES[0], execution_mode="hybrid"):
    materialize_dataset(tmp_path)
    dataset = load_dataset(tmp_path)
    case = BusinessCase.model_validate(copy.deepcopy(spec))
    state = task_state(case, "optimization", dataset, execution_mode=execution_mode)
    return case, state, dataset


def query_payload():
    return {"action": "query_reviews", "reason": "统计本地全集", "arguments": {"appid": 730, "filters": {"period": "all"}, "limit": 1}}


def test_pure_statistics_need_only_one_model_call(tmp_path):
    case, state, dataset = fixture(tmp_path)
    state["max_steps"] = 3  # Finalization is not an additional tool call.
    client = FakeModel([query_payload()])
    result = InvestigationAgent(LLMPlanner(client), business_registry(tmp_path, None)).invoke(state)
    assert client.calls == 1
    assert result["status"] == "completed"
    assert result["answer_origin"] == "deterministic_statistics"
    assert [o["tool"] for o in result["observations"]] == ["inspect_data", "query_reviews", "verify_investigation"]
    assert "15条" in result["final_answer"] and "53.33%" in result["final_answer"]
    result["evaluation"] = evaluation_tags(case, dataset)
    assert evaluate_business_run(case, result, dataset)["automatic_passed"]
    assert "程序根据已核验统计" in Path(result["artifacts"]["final_report"]).read_text(encoding="utf-8")


@pytest.mark.parametrize("goal", ["比较p1前后好评率，能否证明改善？", "近五次更新的评论数量和好评率", "本地评论数量和好评率是多少？请引用原文", "统计掉帧抱怨与已经不再掉帧的反馈", "统计p1和p2的评论数量"])
def test_ambiguous_or_complex_goals_do_not_fast_finish(goal):
    assert not simple_statistics_goal(goal)


def test_context_shares_schemas_preserves_constraints_and_raw_evidence(tmp_path):
    case, state, dataset = authored_run(tmp_path, CASES[5])
    tools = business_registry(tmp_path, None).specifications()
    original = copy.deepcopy(state)
    builder = PlannerContextBuilder()
    reference = builder.build(state, tools, compact=False)
    compact = builder.build(state, tools)
    compact["planner_output_schema"] = compact_schema(PlannerDecision.model_json_schema(), compact["$defs"])
    reference["planner_output_schema"] = PlannerDecision.model_json_schema()
    assert len(json.dumps(compact, ensure_ascii=False)) < len(json.dumps(reference, ensure_ascii=False)) * .85
    assert state == original
    assert len(compact["available_tools"]) == len(tools)
    assert compact["planner_output_schema"]["properties"]["fact_ids"]["maxItems"] == 64
    assert all("label" not in f for f in compact["verified_fact_catalog"])
    assert set(f["fact_id"] for f in compact["verified_fact_catalog"]) <= set(f["fact_id"] for f in fact_catalog(state["observations"]))

    def references(value):
        if isinstance(value, dict):
            if "$ref" in value:
                yield value["$ref"].split("/")[-1]
            for item in value.values():
                yield from references(item)
        elif isinstance(value, list):
            for item in value:
                yield from references(item)
    assert all(name in compact["$defs"] for name in references(compact))


def test_reader_cleanup_does_not_touch_structured_citations():
    text = "cached_reviews_only的stats.positive_rate为0.6666666666666666"
    assert "cached_reviews_only" not in reader_text(text)
    assert "0.6667" in reader_text(text)


def test_duplicate_claims_and_fact_ids_are_deduplicated():
    from test_finish_repair import ready_state
    state = ready_state()
    fact = fact_catalog(state["observations"])[0]
    claim = {k: v for k, v in fact.items() if k != "fact_id"}
    client = FakeModel([{"action": "finish", "reason": "结束", "claims": [claim], "fact_ids": [fact["fact_id"]]}])
    result = InvestigationAgent(LLMPlanner(client), business_registry(Path("."), None).subset(set())).invoke(state)
    assert len(result["final_claims"]) == 1


def test_transport_failure_has_verified_partial_result_and_sqlite_recovery(tmp_path):
    case, state, dataset = authored_run(tmp_path, CASES[2])
    state["optimization_mode"] = "off"
    state["thread_id"] = "persistent-generation"
    state["final_answer"] = ""
    state["final_claims"] = []
    connection = sqlite3.connect(tmp_path / "checkpoint.sqlite", check_same_thread=False)
    checker = SqliteSaver(connection)
    client = FakeModel([httpx.ReadTimeout("network timeout")])
    first = InvestigationAgent(LLMPlanner(client), business_registry(tmp_path, None), checkpointer=checker)
    result = first.invoke(state)
    assert result["status"] == "interrupted"
    assert result["generation_pending"]
    assert "调查未完成" in result["final_answer"]
    assert "50.00%" in result["final_answer"] and "54.55%" in result["final_answer"]
    assert result["final_claims"]
    assert "未完成调查" in Path(result["artifacts"]["final_report"]).read_text(encoding="utf-8")
    assert not first.snapshot(state["thread_id"]).next
    before_queries = [o for o in result["observations"] if o["tool"] == "query_reviews"]
    ids = [f["fact_id"] for f in fact_catalog(result["observations"]) if f["fact_path"][-1:] == ["positive_rate"]]
    resumed_client = FakeModel([{"action": "finish", "reason": "从已有证据生成", "final_answer": "本地好评率有所变化，但不能证明更新的因果效果。", "fact_ids": ids}])
    second = InvestigationAgent(LLMPlanner(resumed_client), business_registry(tmp_path, None), checkpointer=checker)
    recovered = list(second.recover_stream(state["thread_id"]))[-1]
    assert recovered["status"] == "completed"
    assert not recovered["generation_pending"] and recovered["generation_recoveries"] == 1
    assert [o for o in recovered["observations"] if o["tool"] == "query_reviews"] == before_queries
    assert recovered["observations"][-1]["tool"] == "verify_investigation"
    assert resumed_client.calls == 1
    connection.close()


def test_changed_source_not_displayed_as_verified_partial(tmp_path):
    case, state, _ = authored_run(tmp_path, CASES[0])
    path = tmp_path / "data/raw/730_reviews.json"
    path.write_text(path.read_text(encoding="utf-8")+" ", encoding="utf-8")
    assert statistical_answer(state, partial=True) is None


def test_planning_transport_failure_without_sources_does_not_invent_data(tmp_path):
    case, state, _ = fixture(tmp_path, execution_mode="full")
    client = FakeModel([httpx.ConnectError("offline")])
    result = InvestigationAgent(LLMPlanner(client), business_registry(tmp_path, None)).invoke(state)
    assert result["status"] == "interrupted"
    assert result["generation_stage"] == "planning"
    assert not result["final_claims"]
    assert "没有可重新核验" in result["final_answer"]


def test_permanent_auth_error_not_retried_as_generation(tmp_path):
    from gamelens.core.llm import LLMResponseError
    _, state, _ = fixture(tmp_path, execution_mode="full")
    client = FakeModel([LLMResponseError("permission denied", status_code=403)])
    result = InvestigationAgent(LLMPlanner(client), business_registry(tmp_path, None)).invoke(state)
    assert result["status"] == "failed"
    assert not result.get("generation_pending")


def test_recovery_rejects_changed_source_without_repeating_queries(tmp_path):
    _, state, _ = authored_run(tmp_path, CASES[2])
    state["optimization_mode"] = "off"
    client = FakeModel([httpx.ReadTimeout("timeout")])
    agent = InvestigationAgent(LLMPlanner(client), business_registry(tmp_path, None))
    interrupted = agent.invoke(state)
    assert interrupted["generation_pending"]
    path = tmp_path / "data/raw/730_reviews.json"
    path.write_text(path.read_text(encoding="utf-8")+" ", encoding="utf-8")
    restored = list(agent.recover_stream(state["thread_id"]))[-1]
    assert restored["status"] == "failed"
    assert client.calls == 1
    assert not restored["artifacts"].get("evidence_assessed")


def test_recovery_has_a_manual_attempt_cap(tmp_path):
    _, state, _ = fixture(tmp_path, execution_mode="full")
    client = FakeModel([httpx.ConnectError("offline") for _ in range(3)])
    agent = InvestigationAgent(LLMPlanner(client), business_registry(tmp_path, None))
    agent.invoke(state)
    for _ in range(2):
        last = list(agent.recover_stream(state["thread_id"]))[-1]
        assert last["status"] == "interrupted"
    with pytest.raises(RuntimeError, match="2次上限"):
        list(agent.recover_stream(state["thread_id"]))
    assert client.calls == 3


def test_rag_on_not_bypassed_by_statistics_optimization(tmp_path):
    _, state, _ = fixture(tmp_path)
    state["rag_mode"] = "on"
    client = FakeModel([query_payload(), httpx.ReadTimeout("next request")])
    result = InvestigationAgent(LLMPlanner(client), business_registry(tmp_path, None)).invoke(state)
    assert result["status"] == "interrupted"
    assert client.calls == 2
    assert "verify_investigation" not in [o["tool"] for o in result["observations"]]


def test_verified_original_without_stats_still_gets_partial_report(tmp_path):
    _, state, _ = authored_run(tmp_path, CASES[9])
    state["optimization_mode"] = "off"
    agent = InvestigationAgent(LLMPlanner(FakeModel([httpx.ReadTimeout("timeout")])), business_registry(tmp_path, None))
    result = agent.invoke(state)
    assert result["status"] == "interrupted"
    assert "尚未形成最终调查结论" in result["final_answer"]
    assert "未完成调查" in Path(result["artifacts"]["final_report"]).read_text(encoding="utf-8")
    assert not result["final_claims"]


@pytest.mark.parametrize("tool,goal,mode", [("investigate_recent_versions", "最近五次更新的反馈", "off"), ("search_game_evidence", "查看相关证据", "on")])
def test_finish_prerequisites_cannot_exceed_tool_budget(tool, goal, mode):
    from test_finish_repair import ready_state
    from gamelens.core.runtime import AgentTool, ToolRegistry
    from gamelens.tools.investigation import VerifyInvestigationArgs
    state = ready_state()
    state.update(goal=goal, rag_mode=mode, step_count=1, max_steps=1)
    def forbidden(args, state):
        raise AssertionError("Must not execute beyond the tool budget")
    client = FakeModel([{"action":"finish", "reason":"attempt finish"}])
    registry = ToolRegistry([AgentTool(tool, tool, VerifyInvestigationArgs, forbidden)])
    result = InvestigationAgent(LLMPlanner(client), registry).invoke(state)
    assert result["status"] == "failed"
    assert result["observations"] == state["observations"]
    assert "预算" in result["error"]
