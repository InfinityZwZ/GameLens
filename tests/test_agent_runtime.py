import sqlite3
import threading
import time
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, ConfigDict

from gamelens.core.runtime import (
    AgentTool,
    InvestigationAgent,
    InvestigationState,
    PlannerDecision,
    ToolExecutionPolicy,
    ToolRegistry,
    ToolResult,
)
from gamelens.memory.context import ContextPolicy, PlannerContextBuilder


class EmptyArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AppArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    appid: int


def make_state(max_steps: int = 5) -> InvestigationState:
    return {
        "thread_id": "test-thread",
        "goal": "调查测试游戏更新影响",
        "appid": 730,
        "window_days": 7,
        "review_count": 100,
        "news_count": 20,
        "classify_count": 10,
        "patch_news_id": None,
        "refresh_data": False,
        "force_reclassify": False,
        "max_steps": max_steps,
        "step_count": 0,
        "status": "running",
        "plan": [],
        "observations": [],
        "artifacts": {},
        "final_answer": "",
        "error": "",
    }


class SequencePlanner:
    def __init__(self, decisions: list[PlannerDecision]) -> None:
        self.decisions = decisions

    def decide(
        self,
        state: InvestigationState,
        tools: list[dict[str, Any]],
    ) -> PlannerDecision:
        return self.decisions.pop(0)


def test_agent_observes_tool_then_finishes() -> None:
    def create_report(args, state):
        return ToolResult(
            ok=True,
            summary="报告已生成",
            artifacts={
                "final_report": "reports/test.md",
                "evidence_assessed": "true",
            },
        )

    registry = ToolRegistry(
        [
            AgentTool(
                "create_report",
                "创建测试报告",
                EmptyArgs,
                create_report,
            )
        ]
    )
    planner = SequencePlanner(
        [
            PlannerDecision(
                action="create_report",
                reason="还没有报告",
                arguments={},
                plan=["创建报告", "结束"],
            ),
            PlannerDecision(
                action="finish",
                reason="报告已经生成",
                arguments={},
                plan=["已完成"],
                final_answer="调查完成",
            ),
        ]
    )

    result = InvestigationAgent(planner, registry).invoke(make_state())

    assert result["status"] == "completed"
    assert result["final_answer"] == "调查完成"
    assert result["artifacts"]["final_report"] == "reports/test.md"
    assert result["observations"][0]["tool"] == "create_report"


def test_agent_rejects_finish_without_report_and_recovers() -> None:
    def inspect(args, state):
        return ToolResult(
            ok=True,
            summary="检查完成",
            artifacts={
                "final_report": "reports/recovered.md",
                "evidence_assessed": "true",
            },
        )

    registry = ToolRegistry(
        [AgentTool("inspect_data", "检查数据", AppArgs, inspect)]
    )
    planner = SequencePlanner(
        [
            PlannerDecision(
                action="finish",
                reason="尝试提前结束",
                arguments={},
            ),
            PlannerDecision(
                action="finish",
                reason="现在已有报告",
                arguments={},
            ),
        ]
    )

    result = InvestigationAgent(planner, registry).invoke(make_state())

    assert result["status"] == "completed"
    assert result["observations"][0]["tool"] == "finish"
    assert result["observations"][0]["ok"] is False
    assert result["observations"][1]["tool"] == "inspect_data"


def test_agent_stops_at_step_budget() -> None:
    def noop(args, state):
        return ToolResult(ok=True, summary="继续")

    registry = ToolRegistry(
        [AgentTool("noop", "空操作", EmptyArgs, noop)]
    )
    planner = SequencePlanner(
        [
            PlannerDecision(action="noop", reason="测试", arguments={}),
            PlannerDecision(action="noop", reason="测试", arguments={}),
        ]
    )

    result = InvestigationAgent(planner, registry).invoke(
        make_state(max_steps=1)
    )

    assert result["status"] == "failed"
    assert "最大工具调用步数" in result["final_answer"]
    assert len(result["observations"]) == 1


def test_tool_runtime_retries_idempotent_failure(tmp_path) -> None:
    attempts = tmp_path / "attempts.txt"

    def flaky(args, state):
        count=int(attempts.read_text())+1 if attempts.exists() else 1
        attempts.write_text(str(count))
        if count == 1:
            raise ConnectionError("temporary network read failure")
        return ToolResult(ok=True, summary="retry succeeded")

    registry = ToolRegistry(
        [
            AgentTool(
                "flaky",
                "测试有限重试",
                EmptyArgs,
                flaky,
                execution_policy=ToolExecutionPolicy(
                    timeout_seconds=5,
                    max_attempts=2,
                    retry_backoff_seconds=0,
                    retry_mode="transient_read",
                ),
            )
        ]
    )

    result = registry.execute("flaky", {}, make_state())

    assert result.ok is True
    assert attempts.read_text() == "2"
    assert result.data["execution"]["attempts"] == 2


def test_tool_runtime_stops_waiting_after_timeout(tmp_path) -> None:
    output=tmp_path / "late-write.txt"
    def slow(args, state):
        time.sleep(3)
        output.write_text("must never appear")
        return ToolResult(ok=True, summary="too late")

    registry = ToolRegistry(
        [
            AgentTool(
                "slow",
                "测试工具超时",
                EmptyArgs,
                slow,
                execution_policy=ToolExecutionPolicy(
                    timeout_seconds=2,
                ),
            )
        ]
    )

    started = time.monotonic()
    result = registry.execute("slow", {}, make_state())

    assert result.ok is False
    assert result.data["timed_out"] is True
    assert result.data["worker_terminated"] is True
    assert time.monotonic() - started < 3
    time.sleep(3.2)
    assert not output.exists()


def test_tool_runtime_serializes_shared_concurrency_group(tmp_path) -> None:
    log=tmp_path / "serialized.txt"

    def serialized(args, state):
        with log.open("a") as stream: stream.write("start\n")
        time.sleep(0.1)
        with log.open("a") as stream: stream.write("end\n")
        return ToolResult(ok=True, summary="done")

    registry = ToolRegistry(
        [
            AgentTool(
                "serialized",
                "测试并发组",
                EmptyArgs,
                serialized,
                execution_policy=ToolExecutionPolicy(
                    timeout_seconds=10,
                    concurrency_key="shared-files",
                ),
            )
        ]
    )
    workers = [
        threading.Thread(
            target=registry.execute,
            args=("serialized", {}, make_state()),
        )
        for _ in range(2)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    assert log.read_text().splitlines() == ["start","end","start","end"]


def test_context_builder_redacts_and_bounds_untrusted_content() -> None:
    state = make_state()
    state["observations"] = [
        {
            "step": index,
            "tool": "test",
            "ok": True,
            "summary": "评" * 300,
            "data": {"api_key": "secret-value", "index": index},
        }
        for index in range(5)
    ]
    builder = PlannerContextBuilder(
        ContextPolicy(
            max_recent_observations=2,
            max_list_items=3,
            max_string_chars=120,
        )
    )

    context = builder.build(state, [{"name": "tool"}])

    assert len(context["recent_observations"]) == 2
    assert context["recent_observations"][0]["step"] == 3
    assert (
        context["recent_observations"][0]["data"]["api_key"]
        == "[REDACTED]"
    )
    assert "UNTRUSTED_EVIDENCE_ONLY" in context["trust_boundaries"][
        "steam_reviews_and_patch_text"
    ]
    assert "截断" in context["recent_observations"][0]["summary"]


def test_expensive_tool_interrupts_and_resumes_after_approval(tmp_path) -> None:
    executed = tmp_path / "approved.txt"

    def approval(args, state):
        return "该操作会产生额外费用，是否继续？"

    def expensive_tool(args, state):
        executed.write_text("approved")
        return ToolResult(
            ok=True,
            summary="高成本工具已执行",
            artifacts={
                "final_report": "reports/approved.md",
                "evidence_assessed": "true",
            },
        )

    registry = ToolRegistry(
        [
            AgentTool(
                "expensive_tool",
                "需要审批的工具",
                EmptyArgs,
                expensive_tool,
                approval,
            )
        ]
    )
    planner = SequencePlanner(
        [
            PlannerDecision(
                action="expensive_tool",
                reason="需要生成报告",
                arguments={},
            ),
            PlannerDecision(
                action="finish",
                reason="审批后工具已完成",
                arguments={},
            ),
        ]
    )
    agent = InvestigationAgent(planner, registry)
    state = make_state()
    state["thread_id"] = "approval-thread"

    list(agent.stream(state))
    waiting = agent.snapshot("approval-thread")

    assert not executed.exists()
    assert waiting.interrupts
    assert waiting.interrupts[0].value["tool"] == "expensive_tool"

    list(agent.resume_stream("approval-thread", approved=True))
    completed = agent.snapshot("approval-thread").values

    assert executed.read_text() == "approved"
    assert completed["status"] == "completed"
    assert completed["artifacts"]["final_report"] == "reports/approved.md"


def test_sqlite_checkpoint_resumes_with_a_new_agent_instance(tmp_path) -> None:
    executed = tmp_path / "persisted.txt"

    def approval(args, state):
        return "需要人工审批"

    def expensive_tool(args, state):
        executed.write_text("persisted")
        return ToolResult(
            ok=True,
            summary="跨进程恢复成功",
            artifacts={
                "final_report": "reports/persisted.md",
                "evidence_assessed": "true",
            },
        )

    registry = ToolRegistry(
        [
            AgentTool(
                "expensive_tool",
                "持久化审批测试",
                EmptyArgs,
                expensive_tool,
                approval,
            )
        ]
    )
    database = tmp_path / "checkpoints.sqlite"
    first_connection = sqlite3.connect(database, check_same_thread=False)
    first_agent = InvestigationAgent(
        SequencePlanner(
            [
                PlannerDecision(
                    action="expensive_tool",
                    reason="需要执行",
                    arguments={},
                )
            ]
        ),
        registry,
        checkpointer=SqliteSaver(first_connection),
    )
    state = make_state()
    state["thread_id"] = "persistent-thread"

    list(first_agent.stream(state))
    assert first_agent.snapshot("persistent-thread").interrupts
    first_connection.close()

    second_connection = sqlite3.connect(database, check_same_thread=False)
    second_agent = InvestigationAgent(
        SequencePlanner(
            [
                PlannerDecision(
                    action="finish",
                    reason="恢复后的工具已经完成",
                    arguments={},
                )
            ]
        ),
        registry,
        checkpointer=SqliteSaver(second_connection),
    )
    list(second_agent.resume_stream("persistent-thread", approved=True))
    completed = second_agent.snapshot("persistent-thread").values
    second_connection.close()

    assert executed.read_text() == "persisted"
    assert completed["status"] == "completed"
    assert completed["artifacts"]["final_report"] == "reports/persisted.md"
