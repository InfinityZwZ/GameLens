import argparse
import json
import sqlite3

from gamelens.memory.conversation import build_conversation_memory
from gamelens.memory.context import PlannerContextBuilder
from gamelens.agents.investigation import build_initial_state
from gamelens.storage.runs import RunStore
from test_agent_runtime import make_state, EmptyArgs, SequencePlanner
from gamelens.core.runtime import AgentTool, ToolRegistry, ToolResult, InvestigationAgent, PlannerDecision


def run_payload(index=1, conversation_id="session-a", status="completed"):
    return {
        "run_id": f"run-{index}", "conversation_id": conversation_id, "turn_index": index,
        "request": {"appid": 730, "goal": f"第{index}轮调查崩溃问题", "window_days": 7, "classify_count": 10},
        "status": status, "started_at": "2026-10-07T00:00:00Z", "updated_at": f"2026-10-07T00:{index:02d}:00Z",
        "final_answer": "此前回答提到崩溃，请重新核对证据。",
        "artifacts": {"final_report": "reports/test.md", "evidence_assessed": "true"},
        "events": [{"tool": "assess_evidence", "ok": True, "summary": "样本不足", "data": {"sample_count": 5}}],
    }


def test_session_rounds_are_durable_and_isolated(tmp_path):
    path = tmp_path / "runs.sqlite"
    store = RunStore(path)
    store.save(run_payload(1))
    store.save(run_payload(2))
    other = run_payload(3, "session-b")
    other["request"]["appid"] = 1172470
    store.save(other)
    restored = RunStore(path)
    memory = build_conversation_memory("session-a", restored.conversation_runs("session-a"))
    assert memory["previous_turn_count"] == 2
    assert {item["run_id"] for item in memory["recent_turns"]} == {"run-1", "run-2"}
    assert memory["inherited_artifacts"] == {"final_report": "reports/test.md"}
    assert memory["prior_evidence"][0]["source_run_id"] == "run-2"
    assert restored.list_conversations()[1]["turn_count"] == 2


def test_memory_bounds_history_and_excludes_failed_artifacts():
    runs = [run_payload(index) for index in range(1, 18)]
    runs[-1]["status"] = "failed"
    runs[-1]["artifacts"] = {"final_report": "corrupted.md"}
    memory = build_conversation_memory("session-a", runs)
    assert len(memory["recent_turns"]) == 4
    assert len(memory["older_turn_digest"]) == 8
    assert memory["omitted_older_turns"] == 5
    assert memory["prior_evidence"][0]["source_run_id"] == "run-16"
    assert memory["inherited_artifacts"]["final_report"] != "corrupted.md"


def test_legacy_tasks_migrate_without_losing_answers(tmp_path):
    path = tmp_path / "legacy.sqlite"
    payload = run_payload()
    payload.pop("conversation_id")
    payload.pop("turn_index")
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE web_runs(run_id TEXT PRIMARY KEY,status TEXT,started_at TEXT,updated_at TEXT,payload TEXT)")
        db.execute("INSERT INTO web_runs VALUES(?,?,?,?,?)", (payload["run_id"],payload["status"],payload["started_at"],payload["updated_at"],json.dumps(payload)))
    store = RunStore(path)
    assert store.conversation("run-1")["appid"] == 730
    assert store.conversation_runs("run-1")[0]["final_answer"] == payload["final_answer"]
    assert RunStore(path).list_conversations()[0]["turn_count"] == 1


def test_followup_agent_receives_memory_and_reassesses_evidence():
    memory = build_conversation_memory("session-a", [run_payload()])
    args = argparse.Namespace(thread_id="followup-run", goal="刚才的崩溃呢", appid=730,
        profile="extended",
        window_days=7, review_count=100, news_count=20, classify_count=10,
        patch_news_id=None, refresh_data=False, force_reclassify=False, max_steps=5)
    initial = build_initial_state(args, memory)
    assert initial["step_count"] == 0
    assert "evidence_assessed" not in initial["artifacts"]
    context = PlannerContextBuilder().build(initial, [])
    assert context["session_memory"]["recent_turns"][0]["user_question"] == "第1轮调查崩溃问题"
    assert "不能当作已验证事实" in context["session_memory"]["trust_note"]
    planner = SequencePlanner([
        PlannerDecision(action="assess_evidence", reason="对上一轮的崩溃证据重新核对"),
        PlannerDecision(action="finish", reason="已重新评估", scope_limited=True, final_answer="崩溃证据样本不足。"),
    ])
    registry = ToolRegistry([AgentTool("assess_evidence", "核对", EmptyArgs,
        lambda args,state: ToolResult(ok=True, summary="重新核对完成", artifacts={"evidence_assessed": "true"}))])
    result = InvestigationAgent(planner, registry).invoke(initial)
    assert result["status"] == "completed"
    assert result["conversation_memory"]["conversation_id"] == "session-a"
    assert result["observations"][0]["tool"] == "assess_evidence"


def test_earlier_tool_digest_survives_recent_window():
    state = make_state()
    state["observations"] = [{"tool": "inspect_data", "step": i, "ok": True, "summary": f"观察{i}"} for i in range(12)]
    context = PlannerContextBuilder().build(state, [])
    assert len(context["recent_observations"]) == 6
    assert context["earlier_tool_digest"][0]["summary"] == "观察0"
