import json
import multiprocessing
import os
import time

import httpx
import pytest
from pydantic import BaseModel, ConfigDict

from gamelens.core import llm as llm_client
from gamelens.core.action_feedback import project_plan
from gamelens.core.budget import BudgetExceeded, TaskBudgetStore
from gamelens.core.investigation_checks import assess_investigation_progress
from gamelens.core.llm import LLMResponseError, OpenAICompatibleClient
from gamelens.core.runtime import AgentTool, ToolExecutionPolicy, ToolRegistry, ToolResult
from gamelens.domain.tool_results import ReviewQueryPayload, ToolFailure
from test_agent_runtime import make_state


class EmptyArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _reserve_once(path, thread_id, ready, start, output):
    store = TaskBudgetStore(path, request_limit=1)
    ready.put(True)
    start.wait(10)
    try:
        output.put(("ok", store.reserve_llm_request(thread_id)["llm_requests_reserved"]))
    except BudgetExceeded as error:
        output.put(("blocked", error.code))


def test_unknown_tool_exception_is_structured_and_never_retried(tmp_path):
    attempts = tmp_path / "attempts.txt"

    def broken(args, state):
        attempts.write_text(str(int(attempts.read_text()) + 1 if attempts.exists() else 1))
        raise RuntimeError("unexpected internal failure")

    registry = ToolRegistry([AgentTool("broken", "broken", EmptyArgs, broken,
        execution_policy=ToolExecutionPolicy(timeout_seconds=5, max_attempts=3,
                                             retry_mode="transient_read"))])
    result = registry.execute("broken", {}, make_state())

    assert not result.ok
    assert attempts.read_text() == "1"
    assert result.failure == ToolFailure(error_type="RuntimeError", error_code="unknown_tool_error",
        message="unexpected internal failure", retryable=False, side_effects_started=False,
        status_code=None)
    assert result.data["execution"]["attempts"] == 1


def test_only_explicit_transient_read_failure_retries(tmp_path):
    attempts = tmp_path / "attempts.txt"

    def transient(args, state):
        count = int(attempts.read_text()) + 1 if attempts.exists() else 1
        attempts.write_text(str(count))
        if count == 1:
            raise ConnectionError("temporary network read")
        return ToolResult(ok=True, summary="recovered")

    registry = ToolRegistry([AgentTool("transient", "transient", EmptyArgs, transient,
        execution_policy=ToolExecutionPolicy(timeout_seconds=5, max_attempts=2,
                                             retry_backoff_seconds=0,
                                             retry_mode="transient_read"))])
    result = registry.execute("transient", {}, make_state())
    assert result.ok and attempts.read_text() == "2"
    assert result.data["execution"]["attempts"] == 2


def test_permission_error_does_not_retry_even_for_read_policy(tmp_path):
    attempts = tmp_path / "attempts.txt"

    def denied(args, state):
        attempts.write_text(str(int(attempts.read_text()) + 1 if attempts.exists() else 1))
        raise PermissionError("denied")

    registry = ToolRegistry([AgentTool("denied", "denied", EmptyArgs, denied,
        execution_policy=ToolExecutionPolicy(timeout_seconds=5, max_attempts=3,
                                             retry_mode="transient_read"))])
    result = registry.execute("denied", {}, make_state())
    assert not result.ok and attempts.read_text() == "1"
    assert result.failure.error_code == "permission_denied"
    assert result.failure.retryable is False


def test_dead_worker_is_structured_and_not_retried(tmp_path):
    def die(args, state):
        os._exit(7)

    registry = ToolRegistry([AgentTool("die", "die", EmptyArgs, die,
        execution_policy=ToolExecutionPolicy(timeout_seconds=5, max_attempts=2,
                                             retry_mode="transient_read"))])
    result = registry.execute("die", {}, make_state())
    assert not result.ok
    assert result.failure.error_code == "worker_process_died"
    assert result.failure.retryable is False


def test_sqlite_request_reservation_is_atomic_across_processes(tmp_path):
    context = multiprocessing.get_context("spawn")
    ready, output, start = context.Queue(), context.Queue(), context.Event()
    path = tmp_path / "budget.sqlite"
    workers = [context.Process(target=_reserve_once,
        args=(path, "same-task", ready, start, output)) for _ in range(2)]
    for worker in workers:
        worker.start()
    for _ in workers:
        assert ready.get(timeout=10)
    start.set()
    for worker in workers:
        worker.join(15)
        assert worker.exitcode == 0
    outcomes = sorted(output.get(timeout=5)[0] for _ in workers)
    assert outcomes == ["blocked", "ok"]
    assert TaskBudgetStore(path).snapshot("same-task")["llm_requests_reserved"] == 1


def test_every_http_attempt_including_retry_consumes_budget(monkeypatch, tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503, json={})

    factory = httpx.Client
    monkeypatch.setattr(llm_client.httpx, "Client",
        lambda **kwargs: factory(transport=httpx.MockTransport(handler), **kwargs))
    monkeypatch.setattr(llm_client.time, "sleep", lambda _: None)
    store = TaskBudgetStore(tmp_path / "budget.sqlite", request_limit=1)
    client = OpenAICompatibleClient("test", api_key="test", max_retries=2,
        request_budget=store, budget_thread_id="run")
    with pytest.raises(LLMResponseError) as captured:
        client.generate_json([])
    assert captured.value.error_code == "llm_request_budget_exhausted"
    assert len(calls) == 1
    assert store.snapshot("run")["llm_requests_reserved"] == 1


def test_response_format_fallback_also_consumes_budget(monkeypatch, tmp_path):
    calls = []
    factory = httpx.Client
    transport = httpx.MockTransport(lambda request: calls.append(request) or httpx.Response(400, json={}))
    monkeypatch.setattr(llm_client.httpx, "Client",
        lambda **kwargs: factory(transport=transport, **kwargs))
    store = TaskBudgetStore(tmp_path / "budget.sqlite", request_limit=1)
    client = OpenAICompatibleClient("test", api_key="test", max_retries=0,
        request_budget=store, budget_thread_id="run")
    with pytest.raises(LLMResponseError) as captured:
        client.generate_json([])
    assert captured.value.error_code == "llm_request_budget_exhausted"
    assert len(calls) == 1


def test_active_budget_pauses_offline_wait(tmp_path):
    store = TaskBudgetStore(tmp_path / "budget.sqlite", active_seconds_limit=0.2)
    store.resume_active("run")
    time.sleep(0.04)
    store.pause_active("run")
    first = store.snapshot("run")["active_elapsed_seconds"]
    time.sleep(0.08)
    second = store.snapshot("run")["active_elapsed_seconds"]
    assert first >= 0.02
    assert second == pytest.approx(first, abs=0.01)


def test_requirement_ids_do_not_depend_on_plan_wording():
    state = make_state()
    state["goal"] = "统计评论推荐率并提供玩家反馈"
    state["plan"] = ["统计缓存评论", "检查玩家反馈"]
    first = project_plan(state, assess_investigation_progress(state))
    state["plan"] = ["计算好评比例", "阅读用户体验"]
    second = project_plan(state, assess_investigation_progress(state), previous=first)
    first_steps = [(row["requirement_id"], row["status"]) for row in first["requirement_steps"]]
    second_steps = [(row["requirement_id"], row["status"]) for row in second["requirement_steps"]]
    assert second_steps == first_steps
    assert second["requirement_structure_id"] == first["requirement_structure_id"]


def test_task_specific_tool_visibility_is_conservative():
    tools = [AgentTool(name, name, EmptyArgs, lambda a, s: ToolResult(ok=True, summary="ok"))
             for name in ("inspect_data", "query_reviews", "compare_cohorts",
                          "prepare_announcement_comments", "query_announcement_comments",
                          "build_game_index", "search_game_evidence", "verify_investigation")]
    registry = ToolRegistry(tools)
    statistics = make_state()
    statistics["goal"] = "统计全部缓存评论数量与推荐率"
    visible = {row["name"] for row in registry.specifications_for(statistics)}
    assert {"inspect_data", "query_reviews", "compare_cohorts", "verify_investigation"} <= visible
    assert not {"prepare_announcement_comments", "query_announcement_comments",
                "build_game_index", "search_game_evidence"} & visible
    uncertain = make_state()
    uncertain["goal"] = "调查这次更新"
    assert {row["name"] for row in registry.specifications_for(uncertain)} == registry.names


def test_old_tool_receipt_without_failure_field_still_loads(tmp_path):
    payload = {"ok": True, "summary": "legacy", "data": {"fact_paths": ["stats.count"]},
               "artifacts": {}}
    restored = ToolResult.model_validate_json(json.dumps(payload))
    assert restored.ok and restored.failure is None
    assert restored.data["fact_paths"] == ["stats.count"]


def test_registered_critical_result_contract_preserves_extra_fact_paths():
    valid = ToolRegistry([AgentTool("query_reviews", "query", EmptyArgs,
        lambda a, s: ToolResult(ok=True, summary="ok", data={
            "source": {"path":"reviews.json", "sha256":"abc"}, "returned_count":0,
            "reviews":[], "stats":{"count":0}, "fact_paths":["stats.count"]}),
        result_model=ReviewQueryPayload)])
    result = valid.execute("query_reviews", {}, make_state())
    assert result.ok and result.data["fact_paths"] == ["stats.count"]

    invalid = ToolRegistry([AgentTool("query_reviews", "query", EmptyArgs,
        lambda a, s: ToolResult(ok=True, summary="bad", data={"returned_count":0}),
        result_model=ReviewQueryPayload)])
    rejected = invalid.execute("query_reviews", {}, make_state())
    assert not rejected.ok
    assert rejected.failure.error_code == "invalid_tool_result_contract"
