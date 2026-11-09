if __name__ == "__main__":
    from gamelens.bootstrap import ensure_environment
    ensure_environment()

import json
import os
import sqlite3
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict, field_validator

from gamelens.analysis.impact import load_classifications
from gamelens.analysis.report import TOPIC_LABELS, load_impact_summary
from gamelens.analysis.classifier import load_report
from gamelens.storage.runs import RunStore
from gamelens.storage.telemetry import TelemetryStore, summarize
from gamelens.storage.timing import turn_timing, model_status
from gamelens.storage.cleanup import delete_conversation
from gamelens.bootstrap import project_python, require_environment
from gamelens.core.local_time import local_iso, time_context, display_view


from gamelens.core.paths import PROJECT_ROOT
WEB_ROOT = PROJECT_ROOT / "web"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
REPORT_ROOT = PROJECT_ROOT / "reports"
RUN_DATABASE = PROJECT_ROOT / "data" / "web_runs.sqlite"


class RunRequest(BaseModel):
    investigation_profile: Literal["focused", "extended"] = "focused"
    rag_mode: Literal["off","auto","on"] = "auto"
    appid: int = Field(gt=0)
    conversation_id: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    goal: str = Field(
        default="分析最近一次游戏更新对玩家评价的影响",
        min_length=1,
        max_length=500,
    )
    window_days: int = Field(default=7, ge=1, le=30)
    review_count: int = Field(default=1000, ge=1, le=5000)
    news_count: int = Field(default=100, ge=1, le=1000)
    classify_count: int = Field(default=10, ge=1, le=100)
    max_steps: int = Field(default=12, ge=1, le=30)
    refresh_data: bool = False
    force_reclassify: bool = False


class ApprovalDecision(BaseModel):
    approved: bool
    comment: str = Field(default="", max_length=500)


class ConversationUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str | None = Field(default=None, min_length=1, max_length=100)
    pinned: bool | None = None
    archived: bool | None = None

    @field_validator("title")
    @classmethod
    def clean_title(cls, value):
        if value is None:
            return value
        value = value.strip()
        if not value:
            raise ValueError("标题不能为空")
        return value


@dataclass
class RunState:
    run_id: str
    request: RunRequest
    status: str = "queued"
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    finished_at: str | None = None
    return_code: int | None = None
    logs: list[str] = field(default_factory=list)
    plan: list[str] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    final_answer: str = ""
    artifacts: dict[str, str] = field(default_factory=dict)
    pending_approval: dict | None = None
    cancel_requested: bool = False
    conversation_id: str | None = None
    turn_index: int = 1
    agent_status: str | None = None
    decision_trace: list[dict] = field(default_factory=list)
    investigation_progress: dict = field(default_factory=dict)
    plan_state: dict = field(default_factory=dict)
    loop_control: dict = field(default_factory=dict)
    budget: dict = field(default_factory=dict)


@asynccontextmanager
async def lifespan(application):
    # The web process itself reads LangGraph checkpoints during approval/recovery.
    require_environment(sys.executable)
    yield


app = FastAPI(
    title="游鉴 GameLens：基于 LangGraph 的游戏版本影响调查 Agent",
    description="Steam版本评论影响分析Agent",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=WEB_ROOT), name="static")

RUNS: dict[str, RunState] = {}
RUNS_LOCK = threading.Lock()
ACTIVE_PROCESSES: dict[str, subprocess.Popen] = {}
RUN_STORE = RunStore(RUN_DATABASE)
TELEMETRY_STORE = TelemetryStore(PROJECT_ROOT / "data" / "telemetry.sqlite")
ACTIVE_STATUSES = {"queued", "running", "cancelling", "awaiting_approval"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_to_payload(state: RunState) -> dict:
    payload = {
        "run_id": state.run_id,
        "request": state.request.model_dump(mode="json"),
        "status": state.status,
        "started_at": state.started_at,
        "updated_at": state.updated_at,
        "finished_at": state.finished_at,
        "return_code": state.return_code,
        "logs": list(state.logs),
        "plan": list(state.plan),
        "events": list(state.events),
        "final_answer": state.final_answer,
        "artifacts": dict(state.artifacts),
        "pending_approval": state.pending_approval,
        "cancel_requested": state.cancel_requested,
        "conversation_id": state.conversation_id,
        "turn_index": state.turn_index,
        "agent_status": state.agent_status,
        "decision_trace": list(state.decision_trace),
        "investigation_progress": dict(state.investigation_progress),
        "plan_state": dict(state.plan_state),
        "loop_control": dict(state.loop_control),
        "budget": dict(state.budget),
    }
    payload['timing'] = turn_timing(payload)
    return payload


def run_from_payload(payload: dict) -> RunState:
    return RunState(
        run_id=payload["run_id"],
        request=RunRequest.model_validate(payload["request"]),
        status=payload.get("status", "interrupted"),
        started_at=payload.get("started_at", utc_now()),
        updated_at=payload.get("updated_at", utc_now()),
        finished_at=payload.get("finished_at"),
        return_code=payload.get("return_code"),
        logs=list(payload.get("logs", []))[-300:],
        plan=list(payload.get("plan", [])),
        events=list(payload.get("events", [])),
        final_answer=payload.get("final_answer", ""),
        artifacts=dict(payload.get("artifacts", {})),
        pending_approval=payload.get("pending_approval"),
        cancel_requested=bool(payload.get("cancel_requested", False)),
        conversation_id=payload.get("conversation_id") or payload["run_id"],
        turn_index=payload.get("turn_index", 1),
        agent_status=payload.get("agent_status"),
        decision_trace=list(payload.get("decision_trace", [])),
        investigation_progress=dict(payload.get("investigation_progress", {})),
        plan_state=dict(payload.get("plan_state", {})),
        loop_control=dict(payload.get("loop_control", {})),
        budget=dict(payload.get("budget", {})),
    )


def persist_run(state: RunState, *, touch: bool = True) -> None:
    state.conversation_id = state.conversation_id or state.run_id
    if touch:
        state.updated_at = utc_now()
    RUN_STORE.save(run_to_payload(state))


def load_run(run_id: str) -> RunState | None:
    state = RUNS.get(run_id)
    if state is not None:
        return state
    payload = RUN_STORE.get(run_id)
    if payload is None:
        return None
    state = run_from_payload(payload)
    RUNS[run_id] = state
    return state


def restore_runs() -> None:
    for payload in RUN_STORE.list_recent(limit=100):
        state = run_from_payload(payload)
        if state.status in {"queued", "running", "cancelling"}:
            state.status = "interrupted"
            state.finished_at = utc_now()
            state.logs.append("Web服务曾在任务运行期间退出，任务已标记为中断。")
            persist_run(state)
        RUNS[state.run_id] = state


restore_runs()


def find_reports(appid: int | None = None):
    pattern = (
        f"{appid}_version_analysis_*.json"
        if appid is not None
        else "*_version_analysis_*.json"
    )
    reports = []
    for path in PROCESSED_ROOT.glob(pattern):
        try:
            report = load_report(path)
        except (OSError, ValueError):
            continue
        impact_path = PROCESSED_ROOT / (
            f"{report.appid}_review_impact_{report.patch.news_id}.json"
        )
        classification_path = PROCESSED_ROOT / (
            f"{report.appid}_review_classifications_"
            f"{report.patch.news_id}.json"
        )
        if not impact_path.exists() or not classification_path.exists():
            continue
        reports.append((report, impact_path, classification_path))
    reports.sort(
        key=lambda item: item[0].patch.published_at_timestamp,
        reverse=True,
    )
    return reports


def get_report_bundle(appid: int, news_id: str | None):
    reports = find_reports(appid)
    if news_id is not None:
        reports = [
            item for item in reports if item[0].patch.news_id == news_id
        ]
    if not reports:
        raise HTTPException(status_code=404, detail="没有可展示的分析结果")
    return reports[0]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_ROOT / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "time_standard":time_context()}


@app.get("/api/analyses")
def list_analyses() -> list[dict]:
    return [
        {
            "appid": report.appid,
            "news_id": report.patch.news_id,
            "title": report.patch.title,
            "published_at": report.patch.published_at.isoformat(),
            "published_at_local":local_iso(report.patch.published_at_timestamp),
        }
        for report, _, _ in find_reports()
    ]


@app.get("/api/dashboard/{appid}")
def dashboard(appid: int, news_id: str | None = None) -> dict:
    report, impact_path, classification_path = get_report_bundle(
        appid,
        news_id,
    )
    impact = load_impact_summary(impact_path)
    records = load_classifications(classification_path)
    after_records = sorted(
        (record for record in records if record.period == "after"),
        key=lambda record: (
            record.analysis.is_actionable,
            record.analysis.update_related,
            record.analysis.severity,
            record.review.votes_up,
        ),
        reverse=True,
    )

    topic_changes = [
        {
            "topic": item.topic.value,
            "label": TOPIC_LABELS[item.topic],
            "before_count": item.before_count,
            "after_count": item.after_count,
            "before_rate": item.before_rate,
            "after_rate": item.after_rate,
            "change_pp": item.change_percentage_points,
        }
        for item in impact.topic_changes
        if item.before_count or item.after_count
    ]
    evidence = [
        {
            "review_id": record.review.review_id,
            "sentiment": record.analysis.sentiment,
            "topics": [
                TOPIC_LABELS[topic] for topic in record.analysis.topics
            ],
            "severity": record.analysis.severity,
            "actionable": record.analysis.is_actionable,
            "update_related": record.analysis.update_related,
            "summary": record.analysis.summary,
            "evidence": record.analysis.evidence,
            "confidence": record.analysis.confidence,
            "votes_up": record.review.votes_up,
        }
        for record in after_records[:8]
    ]

    return {
        "game": {
            "appid": report.appid,
            "patch_title": report.patch.title,
            "news_id": report.patch.news_id,
            "published_at": report.patch.published_at.isoformat(),
            "published_at_local":local_iso(report.patch.published_at_timestamp),
            "patch_url": report.patch.url,
            "window_days": report.window_days,
        },
        "full_window": {
            "before_total": report.before.total_reviews,
            "after_total": report.after.total_reviews,
            "review_count_change": report.review_count_change,
            "before_positive_rate": report.before.positive_rate,
            "after_positive_rate": report.after.positive_rate,
            "positive_rate_change_pp": (
                report.positive_rate_change_percentage_points
            ),
        },
        "sample": {
            "before_count": impact.before.review_count,
            "after_count": impact.after.review_count,
            "before_sentiments": impact.before.sentiment_counts,
            "after_sentiments": impact.after.sentiment_counts,
            "before_severity": impact.before.average_severity,
            "after_severity": impact.after.average_severity,
            "after_actionable_count": impact.after.actionable_count,
            "after_actionable_rate": impact.after.actionable_rate,
            "after_update_related_rate": impact.after.update_related_rate,
            "average_confidence": impact.after.average_confidence,
        },
        "topic_changes": topic_changes,
        "evidence": evidence,
        "warnings": impact.warnings,
        "report_url": (
            f"/api/reports/{report.appid}/{report.patch.news_id}"
        ),
    }


@app.get("/api/reports/{appid}/{news_id}")
def markdown_report(appid: int, news_id: str, download: bool = False):
    path = REPORT_ROOT / f"{appid}_{news_id}_impact_report.md"
    if not path.is_file() or not path.resolve().is_relative_to(REPORT_ROOT.resolve()):
        raise HTTPException(status_code=404, detail="Markdown报告不存在")
    if download:return FileResponse(path,media_type="text/markdown",filename=path.name)
    return PlainTextResponse(path.read_text(encoding="utf-8"))


def execute_agent(
    run_id: str,
    request: RunRequest,
    approval: ApprovalDecision | None = None,
    recover: bool = False,
) -> None:
    command = [
        project_python(PROJECT_ROOT),
        "-u",
        str(PROJECT_ROOT / "investigation_agent.py"),
        "--thread-id",
        run_id,
    ]
    if recover:
        command.append("--recover")
    elif approval is not None:
        command.extend(
            [
                "--resume",
                "approved" if approval.approved else "denied",
            ]
        )
        if approval.comment:
            command.extend(["--approval-comment", approval.comment])
    else:
        if request.conversation_id:
            command.extend(["--conversation-id", request.conversation_id])
        command.extend(
            [
                "--appid",
                str(request.appid),
                "--goal",
                request.goal,
                "--window-days",
                str(request.window_days),
                "--review-count",
                str(request.review_count),
                "--news-count",
                str(request.news_count),
                "--classify-count",
                str(request.classify_count),
                "--max-steps",
                str(request.max_steps),
                "--rag-mode",
                request.rag_mode,
                "--profile",
                request.investigation_profile,
            ]
        )
        if request.refresh_data:
            command.append("--refresh-data")
        if request.force_reclassify:
            command.append("--force-reclassify")

    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    creation_flags = (
        subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    )

    with RUNS_LOCK:
        state = load_run(run_id)
        if state is None or state.cancel_requested:
            return
        state.status = "running"
        state.pending_approval = None
        state.finished_at = None
        state.agent_status = None
        persist_run(state)
    process: subprocess.Popen | None = None
    try:
        runtime=require_environment(command[0],include_rag=request.rag_mode=="on")
        with RUNS_LOCK:
            state=RUNS[run_id]
            state.logs.append(f"运行环境检查通过；Agent Python：{runtime['python']}；RAG模式：{request.rag_mode}")
            persist_run(state)
            if state.cancel_requested:
                raise RuntimeError("任务在环境检查期间被取消")
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
            creationflags=creation_flags,
        )
        with RUNS_LOCK:
            ACTIVE_PROCESSES[run_id] = process
            state = RUNS[run_id]
            if state.cancel_requested:
                process.terminate()
        if process.stdout is not None:
            for line in process.stdout:
                clean_line = line.rstrip()
                with RUNS_LOCK:
                    state = RUNS[run_id]
                    if clean_line.startswith("AGENT_SNAPSHOT "):
                        try:
                            snapshot = json.loads(clean_line.removeprefix("AGENT_SNAPSHOT "))
                            state.plan = list(snapshot.get("plan",[]))
                            state.events = list(snapshot.get("observations",[]))
                            state.artifacts = dict(snapshot.get("artifacts",{}))
                            state.decision_trace = list(snapshot.get("decision_trace", []))
                            state.investigation_progress = dict(snapshot.get("investigation_progress", {}))
                            state.plan_state = dict(snapshot.get("plan_state", {}))
                            state.loop_control = dict(snapshot.get("loop_control", {}))
                        except (json.JSONDecodeError,TypeError): pass
                    elif clean_line.startswith("AGENT_PLAN "):
                        try:
                            state.plan = json.loads(
                                clean_line.removeprefix("AGENT_PLAN ")
                            )
                        except json.JSONDecodeError:
                            pass
                    elif clean_line.startswith("AGENT_EVENT "):
                        try:
                            event = json.loads(clean_line.removeprefix("AGENT_EVENT "))
                            state.events = [old for old in state.events if old.get("step") != event.get("step")]
                            state.events.append(event)
                        except json.JSONDecodeError:
                            pass
                    elif clean_line.startswith("AGENT_FINAL "):
                        try:
                            final_payload = json.loads(
                                clean_line.removeprefix("AGENT_FINAL ")
                            )
                            state.final_answer = final_payload.get(
                                "answer", ""
                            )
                            state.artifacts = dict(final_payload.get("artifacts", {}))
                            state.agent_status = final_payload.get("status")
                            state.decision_trace = list(final_payload.get("decision_trace", state.decision_trace))
                            state.investigation_progress = dict(final_payload.get("investigation_progress", state.investigation_progress))
                            state.plan_state = dict(final_payload.get("plan_state", state.plan_state))
                            state.loop_control = dict(final_payload.get("loop_control", state.loop_control))
                            state.budget = dict(final_payload.get("budget", state.budget))
                        except json.JSONDecodeError:
                            pass
                    elif clean_line.startswith("AGENT_INTERRUPT "):
                        try:
                            state.pending_approval = json.loads(
                                clean_line.removeprefix("AGENT_INTERRUPT ")
                            )
                        except json.JSONDecodeError:
                            pass
                    state.logs.append(clean_line)
                    state.logs = state.logs[-300:]
                    persist_run(state)
        return_code = process.wait()
    except Exception as error:  # 后台线程必须把异常转换成可查询状态
        with RUNS_LOCK:
            state = RUNS[run_id]
            state.logs.append(f"运行器异常：{error}")
            if process is None:state.final_answer=f"任务启动失败：{error}"
            persist_run(state)
        return_code = -1
    finally:
        with RUNS_LOCK:
            ACTIVE_PROCESSES.pop(run_id, None)

    with RUNS_LOCK:
        state = RUNS[run_id]
        state.return_code = return_code
        if state.cancel_requested:
            state.status = "cancelled"
            state.finished_at = utc_now()
            state.pending_approval = None
            state.logs.append("任务已由用户取消。")
        elif return_code != 0 or state.agent_status == "failed":
            state.status = "failed"
            state.finished_at = utc_now()
        elif state.pending_approval is not None:
            state.status = "awaiting_approval"
        elif state.agent_status == "interrupted":
            state.status = "interrupted"
            state.finished_at = utc_now()
        else:
            state.status = "completed"
            state.finished_at = utc_now()
        persist_run(state)


@app.post("/api/runs", status_code=202)
def start_run(request: RunRequest) -> dict:
    if not request.goal.strip():
        raise HTTPException(status_code=422, detail="调查问题不能为空")
    if request.conversation_id is None and len(request.goal.strip()) < 5:
        raise HTTPException(status_code=422, detail="首次调查目标至少5个字")
    run_id = uuid.uuid4().hex[:12]
    with RUNS_LOCK:
        cid = request.conversation_id
        previous_turns = []
        if cid:
            conversation = RUN_STORE.conversation(cid)
            if conversation is None:
                raise HTTPException(status_code=404, detail="会话不存在")
            if conversation["appid"] != request.appid:
                raise HTTPException(status_code=409, detail="同一会话不能切换游戏，请新建调查")
            if conversation["archived"]:
                raise HTTPException(status_code=409, detail="请先恢复归档会话再继续追问")
            previous_turns = RUN_STORE.conversation_runs(cid)
            if any(turn["status"] in ACTIVE_STATUSES for turn in previous_turns):
                raise HTTPException(status_code=409, detail="本轮尚未结束，请先完成审批或停止任务")
        duplicate = next(
            (
                state
                for state in RUNS.values()
                if state.request.appid == request.appid
                and state.status in ACTIVE_STATUSES
            ),
            None,
        )
        if duplicate is not None:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"AppID {request.appid}已有进行中的任务"
                    f"（{duplicate.run_id}）"
                ),
            )
        cid = cid or uuid.uuid4().hex[:12]
        request = request.model_copy(update={"conversation_id": cid})
        state = RunState(run_id=run_id, request=request, conversation_id=cid,
                         turn_index=max((turn.get("turn_index", 1) for turn in previous_turns), default=0)+1)
        RUNS[run_id] = state
        persist_run(state)
    thread = threading.Thread(
        target=execute_agent,
        args=(run_id, request),
        daemon=True,
    )
    thread.start()
    return {"run_id": run_id, "status": "queued", "conversation_id": cid, "turn_index": state.turn_index}


@app.post("/api/runs/{run_id}/approval", status_code=202)
def decide_approval(run_id: str, decision: ApprovalDecision) -> dict:
    with RUNS_LOCK:
        state = load_run(run_id)
        if state is None:
            raise HTTPException(status_code=404, detail="运行记录不存在")
        if state.status != "awaiting_approval":
            raise HTTPException(
                status_code=409,
                detail="当前任务不在等待审批状态",
            )
        state.status = "queued"
        state.cancel_requested = False
        request = state.request
        persist_run(state)
    thread = threading.Thread(
        target=execute_agent,
        args=(run_id, request, decision),
        daemon=True,
    )
    thread.start()
    return {"run_id": run_id, "status": "queued"}


@app.post("/api/runs/{run_id}/cancel", status_code=202)
def cancel_run(run_id: str) -> dict:
    with RUNS_LOCK:
        state = load_run(run_id)
        if state is None:
            raise HTTPException(status_code=404, detail="运行记录不存在")
        if state.status not in ACTIVE_STATUSES:
            raise HTTPException(
                status_code=409,
                detail="当前任务已结束，不能取消",
            )
        state.cancel_requested = True
        process = ACTIVE_PROCESSES.get(run_id)
        if process is None:
            state.status = "cancelled"
            state.finished_at = utc_now()
            state.pending_approval = None
        else:
            state.status = "cancelling"
        persist_run(state)

    if process is not None and process.poll() is None:
        try:
            process.terminate()
        except OSError:
            pass
    return {"run_id": run_id, "status": state.status}


@app.post("/api/runs/{run_id}/recover",status_code=202)
def recover_run(run_id: str) -> dict:
    from gamelens.core.runtime import InvestigationAgent,ToolRegistry
    from langgraph.checkpoint.sqlite import SqliteSaver
    with RUNS_LOCK:
        state = load_run(run_id)
        if state is None: raise HTTPException(status_code=404,detail="运行记录不存在")
        if state.status not in {"interrupted","failed"}:
            raise HTTPException(status_code=409,detail="任务不是可恢复的中断状态")
        path = PROJECT_ROOT/"data"/"agent_checkpoints.sqlite"
        if not path.is_file(): raise HTTPException(status_code=409,detail="没有持久化检查点，请新建调查")
        with sqlite3.connect(path,check_same_thread=False) as connection:
            agent = InvestigationAgent(None,ToolRegistry([]),checkpointer=SqliteSaver(connection))
            snapshot = agent.snapshot(run_id)
        pending_generation = bool(snapshot.values.get("generation_pending")) and snapshot.values.get("generation_recoveries", 0) < 2
        if not snapshot.values or (not snapshot.next and not pending_generation) or snapshot.interrupts:
            raise HTTPException(status_code=409,detail="没有未完成的执行节点；审批任务请批准/拒绝，已终结任务请追问或新建调查")
        if any(other.run_id != run_id and other.request.appid == state.request.appid and other.status in ACTIVE_STATUSES for other in RUNS.values()):
            raise HTTPException(status_code=409,detail="同一游戏已有其他活动任务")
        state.status="queued"
        state.cancel_requested=False
        persist_run(state)
        request=state.request
    threading.Thread(target=execute_agent,args=(run_id,request,None,True),daemon=True).start()
    return {"run_id":run_id,"status":"queued"}


@app.get("/api/runs/{run_id}/report")
def run_report(run_id: str, download: bool = False):
    with RUNS_LOCK:
        state = load_run(run_id)
        if state is None: raise HTTPException(status_code=404,detail="运行记录不存在")
        path=Path(state.artifacts.get("final_report",""))
    if not path.is_file() or not path.resolve().is_relative_to(REPORT_ROOT.resolve()):
        raise HTTPException(status_code=404,detail="当前任务报告不存在")
    if download:return FileResponse(path,media_type="text/markdown",filename=path.name)
    return PlainTextResponse(path.read_text(encoding="utf-8"))


@app.get("/api/runs")
def list_runs(limit: int = 20) -> list[dict]:
    if not 1 <= limit <= 100:
        raise HTTPException(status_code=422, detail="limit必须在1到100之间")
    payloads = RUN_STORE.list_recent(limit=limit)
    return display_view([
        {
            "run_id": payload["run_id"],
            "appid": payload["request"]["appid"],
            "goal": payload["request"]["goal"],
            "status": payload["status"],
            "investigation_outcome":payload.get('investigation_progress',{}).get('outcome'),
            "started_at": payload["started_at"],
            "updated_at": payload["updated_at"],
            "finished_at": payload.get("finished_at"),
            "event_count": len(payload.get("events", [])),
        }
        for payload in payloads
    ])


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    with RUNS_LOCK:
        state = load_run(run_id)
        if state is None:
            raise HTTPException(status_code=404, detail="运行记录不存在")
        model_events = TELEMETRY_STORE.events(run_id)
        telemetry = summarize(model_events)
        return display_view({
            "run_id": state.run_id,
            "status": state.status,
            "started_at": state.started_at,
            "updated_at": state.updated_at,
            "finished_at": state.finished_at,
            "return_code": state.return_code,
            "logs": list(state.logs),
            "plan": list(state.plan),
            "events": list(state.events),
            "decision_trace": list(state.decision_trace),
            "investigation_progress": dict(state.investigation_progress),
            "investigation_outcome":state.investigation_progress.get('outcome'),
            "plan_state": dict(state.plan_state),
            "loop_control": dict(state.loop_control),
            "budget": dict(state.budget),
            "final_answer": state.final_answer,
            "artifacts": dict(state.artifacts),
            "pending_approval": state.pending_approval,
            "cancel_requested": state.cancel_requested,
            "request": state.request.model_dump(mode="json"),
            "telemetry": telemetry,
            "timing":turn_timing(run_to_payload(state),telemetry=telemetry),
            "model_status":model_status(model_events,state.status in ACTIVE_STATUSES),
            "conversation_id": state.conversation_id or state.run_id,
            "turn_index": state.turn_index,
        })


@app.get("/api/conversations")
def list_conversations(limit: int = 30, view: str = "active", q: str = "") -> list[dict]:
    if not 1 <= limit <= 100:
        raise HTTPException(status_code=422, detail="limit必须在1到100之间")
    if view not in {"active", "archived", "all"} or len(q) > 100:
        raise HTTPException(status_code=422, detail="无效的会话筛选条件")
    return display_view(RUN_STORE.list_conversations(limit, view, q.strip()))


@app.patch("/api/conversations/{conversation_id}")
def update_conversation(conversation_id: str, update: ConversationUpdate) -> dict:
    changes = update.model_dump(exclude_none=True)
    if not changes:
        raise HTTPException(status_code=422, detail="至少提供一个更新字段")
    with RUNS_LOCK:
        try:
            return display_view(RUN_STORE.update_conversation(conversation_id, changes, utc_now()))
        except KeyError:
            raise HTTPException(status_code=404, detail="会话不存在")


@app.delete("/api/conversations/{conversation_id}")
def permanently_delete_conversation(conversation_id: str) -> dict:
    with RUNS_LOCK:
        if RUN_STORE.conversation(conversation_id) is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        turns = RUN_STORE.conversation_runs(conversation_id)
        if any(turn["status"] in ACTIVE_STATUSES or turn["run_id"] in ACTIVE_PROCESSES for turn in turns):
            raise HTTPException(status_code=409, detail="请先停止正在执行或等待审批的任务，再删除会话")
        try:
            run_ids = delete_conversation(RUN_STORE,conversation_id,
                PROJECT_ROOT / "data" / "agent_checkpoints.sqlite",
                TELEMETRY_STORE.path, PROJECT_ROOT / "data" / "runs")
        except (OSError, ValueError, sqlite3.Error) as error:
            raise HTTPException(status_code=500, detail=f"会话清理失败：{error}")
        for run_id in run_ids:
            RUNS.pop(run_id, None)
    return {"deleted": True, "conversation_id": conversation_id, "deleted_runs": len(run_ids)}


@app.get("/api/conversations/{conversation_id}")
def get_conversation(conversation_id: str) -> dict:
    with RUNS_LOCK:
        conversation = RUN_STORE.conversation(conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        turns = RUN_STORE.conversation_runs(conversation_id)
    events = []
    for turn in turns:
        per_turn = TELEMETRY_STORE.events(turn['run_id'])
        turn['telemetry'] = summarize(per_turn)
        turn['timing'] = turn_timing(turn,telemetry=turn['telemetry'])
        events.extend(per_turn)
    return display_view({**conversation, "turns": turns, "telemetry": summarize(events)})


@app.get("/api/runs/{run_id}/telemetry")
def get_run_telemetry(run_id: str) -> dict:
    with RUNS_LOCK:
        if load_run(run_id) is None:
            raise HTTPException(status_code=404, detail="运行记录不存在")
    events = TELEMETRY_STORE.events(run_id)
    return {"summary": summarize(events), "events": events}


if __name__ == "__main__":
    uvicorn.run(
        "gamelens.api.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
