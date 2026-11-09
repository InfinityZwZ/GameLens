from fastapi.testclient import TestClient
import pytest

from gamelens.api import app as web_app
from gamelens.storage.runs import RunStore
from gamelens.storage.telemetry import TelemetryStore
from gamelens.api.app import RUNS, RUNS_LOCK, RunRequest, RunState, app


client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_run_store(tmp_path, monkeypatch):
    store = RunStore(tmp_path / "web_runs.sqlite")
    monkeypatch.setattr(web_app, "RUN_STORE", store)
    monkeypatch.setattr(web_app, "TELEMETRY_STORE", TelemetryStore(tmp_path / "telemetry.sqlite"))
    with RUNS_LOCK:
        RUNS.clear()
        web_app.ACTIVE_PROCESSES.clear()
    yield store
    with RUNS_LOCK:
        RUNS.clear()
        web_app.ACTIVE_PROCESSES.clear()


def test_health_and_frontend_shell() -> None:
    health = client.get("/api/health")
    index = client.get("/")

    assert health.status_code == 200
    assert health.json()['status'] == 'ok'
    assert health.json()['time_standard']['basis'] == 'beijing'
    assert health.json()['time_standard']['system_zone'] == 'Asia/Shanghai'
    assert '北京时间' in index.text
    assert index.status_code == 200
    assert "游鉴 GameLens" in index.text
    assert "调查目标" in index.text
    assert "工具执行" in index.text
    assert "需要你的确认" in index.text
    assert "历史调查" in index.text


def test_run_and_conversation_endpoints_expose_limited_outcome():
    run=RunState(run_id='limited-status',request=RunRequest(appid=730,goal='公告留言读取测试'),status='completed')
    run.investigation_progress={'outcome':'budget_exhausted'}
    with RUNS_LOCK:
        RUNS[run.run_id]=run
        web_app.persist_run(run)
    for response in (client.get('/api/runs/limited-status'),client.get('/api/runs'),client.get('/api/conversations')):
        assert response.status_code==200
        data=response.json()
        row=data[0] if isinstance(data,list) else data
        assert row['investigation_outcome']=='budget_exhausted'
        assert row['status']=='completed'


def test_analysis_catalog_and_dashboard_contract() -> None:
    catalog_response = client.get("/api/analyses")

    assert catalog_response.status_code == 200
    catalog = catalog_response.json()
    assert isinstance(catalog, list)
    if not catalog:
        return

    item = catalog[0]
    dashboard_response = client.get(
        f"/api/dashboard/{item['appid']}",
        params={"news_id": item["news_id"]},
    )
    assert dashboard_response.status_code == 200
    dashboard = dashboard_response.json()
    assert {
        "game",
        "full_window",
        "sample",
        "topic_changes",
        "evidence",
        "warnings",
        "report_url",
    } <= dashboard.keys()


def test_human_approval_endpoint_resumes_waiting_run(monkeypatch) -> None:
    started = []

    class FakeThread:
        def __init__(self, *, target, args, daemon):
            self.target = target
            self.args = args
            self.daemon = daemon

        def start(self):
            started.append(self.args)

    run_id = "approval-test"
    request = RunRequest(appid=730, goal="测试人工审批恢复流程")
    with RUNS_LOCK:
        RUNS[run_id] = RunState(
            run_id=run_id,
            request=request,
            status="awaiting_approval",
            pending_approval={
                "tool": "classify_review_sample",
                "message": "将产生额外费用",
            },
        )
    monkeypatch.setattr(web_app.threading, "Thread", FakeThread)

    response = client.post(
        f"/api/runs/{run_id}/approval",
        json={"approved": True, "comment": "测试批准"},
    )

    assert response.status_code == 202
    assert response.json()["status"] == "queued"
    assert started
    assert started[0][0] == run_id
    assert started[0][2].approved is True


def test_run_metadata_survives_memory_clear(isolated_run_store) -> None:
    state = RunState(
        run_id="persisted-run",
        request=RunRequest(appid=730, goal="验证Web运行记录持久化"),
        status="completed",
        final_answer="持久化成功",
        artifacts={"final_report": "reports/730_test_impact_report.md"},
    )
    web_app.persist_run(state)
    with RUNS_LOCK:
        RUNS.clear()

    response = client.get("/api/runs/persisted-run")
    listing = client.get("/api/runs")

    assert response.status_code == 200
    assert response.json()["final_answer"] == "持久化成功"
    assert response.json()["artifacts"]["final_report"] == "reports/730_test_impact_report.md"
    assert listing.status_code == 200
    assert listing.json()[0]["run_id"] == "persisted-run"


def test_cancel_waiting_approval_run() -> None:
    state = RunState(
        run_id="cancel-test",
        request=RunRequest(appid=730, goal="验证等待审批任务取消"),
        status="awaiting_approval",
        pending_approval={"tool": "prepare_reviews"},
    )
    with RUNS_LOCK:
        RUNS[state.run_id] = state
        web_app.persist_run(state)

    response = client.post("/api/runs/cancel-test/cancel", json={})
    persisted = client.get("/api/runs/cancel-test").json()

    assert response.status_code == 202
    assert response.json()["status"] == "cancelled"
    assert persisted["status"] == "cancelled"
    assert persisted["pending_approval"] is None


def test_duplicate_active_appid_is_rejected(monkeypatch) -> None:
    class FakeThread:
        def __init__(self, *, target, args, daemon):
            pass

        def start(self):
            pass

    monkeypatch.setattr(web_app.threading, "Thread", FakeThread)
    request = {"appid": 730, "goal": "检查同一个游戏的重复任务"}

    first = client.post("/api/runs", json=request)
    second = client.post("/api/runs", json=request)

    assert first.status_code == 202
    assert second.status_code == 409


def test_web_run_exposes_persisted_model_usage():
    state = RunState(run_id="usage-test", request=RunRequest(appid=730, goal="检查模型调用统计"), status="completed")
    web_app.persist_run(state)
    web_app.TELEMETRY_STORE.record(state.run_id, {
        "id": "request-1", "kind": "request", "call_id": "call-1",
        "model": "test-model", "status": "succeeded", "prompt_tokens": 100,
        "completion_tokens": 20, "estimated_cost": 0.001, "currency": "CNY",
        "latency_ms": 120,
    })
    with RUNS_LOCK:
        RUNS.clear()
    detail = client.get("/api/runs/usage-test").json()
    metrics = client.get("/api/runs/usage-test/telemetry").json()
    assert detail["telemetry"]["prompt_tokens"] == 100
    assert metrics["summary"]["estimated_cost"] == 0.001
    assert len(metrics["events"]) == 1
    assert client.get("/api/runs/nonexistent/telemetry").status_code == 404


def test_conversation_followup_and_refresh_persist(monkeypatch):
    class FakeThread:
        def __init__(self, **kwargs):
            pass
        def start(self):
            pass
    monkeypatch.setattr(web_app.threading, "Thread", FakeThread)
    first = client.post("/api/runs", json={"appid": 730, "goal": "调查游戏更新后的崩溃问题"}).json()
    cid = first["conversation_id"]
    blocked = client.post("/api/runs", json={"appid":730,"goal":"展开","conversation_id":cid})
    assert blocked.status_code == 409
    with RUNS_LOCK:
        state = RUNS[first["run_id"]]
        state.status = "completed"
        state.final_answer = "第一轮发现崩溃反馈，需进一步核对。"
        web_app.persist_run(state)
        RUNS.clear()
    second_response = client.post("/api/runs", json={"appid":730,"goal":"展开","conversation_id":cid})
    assert second_response.status_code == 202
    second = second_response.json()
    assert second["conversation_id"] == cid
    assert second["run_id"] != first["run_id"]
    assert second["turn_index"] == 2
    with RUNS_LOCK:
        RUNS.clear()
    detail = client.get(f"/api/conversations/{cid}").json()
    assert len(detail["turns"]) == 2
    assert detail["turns"][0]["final_answer"].startswith("第一轮")
    catalog = client.get("/api/conversations").json()
    assert len(catalog) == 1
    assert catalog[0]["turn_count"] == 2
    assert client.post("/api/runs", json={"appid":1172470,"goal":"换个游戏调查","conversation_id":cid}).status_code == 409
    assert client.post("/api/runs", json={"appid":730,"goal":"继续","conversation_id":"unknown"}).status_code == 404


def test_recover_api_dispatches_only_unfinished_checkpoints(tmp_path,monkeypatch):
    import sqlite3
    from langgraph.checkpoint.sqlite import SqliteSaver
    from gamelens.core.runtime import InvestigationAgent,PlannerDecision,ToolRegistry
    from test_agent_runtime import make_state,SequencePlanner
    monkeypatch.setattr(web_app,"PROJECT_ROOT",tmp_path)
    path=tmp_path/"data"/"agent_checkpoints.sqlite"
    path.parent.mkdir()
    connection=sqlite3.connect(path,check_same_thread=False)
    agent=InvestigationAgent(SequencePlanner([PlannerDecision(action="inspect_data",reason="检查数据",arguments={"appid":730})]),ToolRegistry([]),checkpointer=SqliteSaver(connection))
    state=make_state(); state["thread_id"]="recover-api"
    stream=agent.stream(state); next(stream); next(stream); stream.close(); connection.close()
    started=[]
    class FakeThread:
        def __init__(self,**kwargs): self.args=kwargs["args"]
        def start(self): started.append(self.args)
    monkeypatch.setattr(web_app.threading,"Thread",FakeThread)
    run=RunState(run_id="recover-api",request=RunRequest(appid=730,goal="恢复已中断调查"),status="interrupted")
    with RUNS_LOCK: RUNS[run.run_id]=run; web_app.persist_run(run)
    response=client.post("/api/runs/recover-api/recover",json={})
    assert response.status_code==202
    assert started[0][3] is True
    assert client.post("/api/runs/recover-api/recover",json={}).status_code==409


def test_recover_api_allows_pending_generation_after_graph_end(tmp_path, monkeypatch):
    import sqlite3
    from langgraph.checkpoint.sqlite import SqliteSaver
    from gamelens.core.runtime import InvestigationAgent, ToolRegistry
    from test_agent_runtime import make_state
    monkeypatch.setattr(web_app, "PROJECT_ROOT", tmp_path)
    path = tmp_path/"data/agent_checkpoints.sqlite"
    path.parent.mkdir()
    with sqlite3.connect(path, check_same_thread=False) as connection:
        agent = InvestigationAgent(None, ToolRegistry([]), checkpointer=SqliteSaver(connection))
        state = {**make_state(), "thread_id": "generation-api", "status": "interrupted",
                 "generation_pending": True, "generation_recoveries": 0}
        agent.graph.update_state({"configurable": {"thread_id": state["thread_id"]}}, state, as_node="partial")
        assert not agent.snapshot(state["thread_id"]).next
    started = []
    class FakeThread:
        def __init__(self, **kwargs): self.args = kwargs["args"]
        def start(self): started.append(self.args)
    monkeypatch.setattr(web_app.threading, "Thread", FakeThread)
    run = RunState(run_id="generation-api", request=RunRequest(appid=730), status="interrupted")
    RUNS[run.run_id] = run
    web_app.persist_run(run)
    response = client.post("/api/runs/generation-api/recover", json={})
    assert response.status_code == 202
    assert started[0][3] is True


def test_runner_keeps_interrupted_result_not_completed(monkeypatch):
    import json
    class Process:
        pid = 111
        stdout = iter(["AGENT_FINAL " + json.dumps({"status":"interrupted","answer":"调查未完成","artifacts":{}})])
        def wait(self): return 0
    monkeypatch.setattr(web_app, "require_environment", lambda *a, **k: {"python":"test"})
    monkeypatch.setattr(web_app.subprocess, "Popen", lambda *a, **k: Process())
    run = RunState(run_id="partial-run", request=RunRequest(appid=730))
    RUNS[run.run_id] = run
    web_app.persist_run(run)
    web_app.execute_agent(run.run_id, run.request)
    assert run.status == "interrupted"
    assert run.final_answer == "调查未完成"
