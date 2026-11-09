"""CLI entry point for the LangGraph game investigation agent."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from gamelens.storage.atomic_io import atomic_write_json

from langgraph.checkpoint.sqlite import SqliteSaver

from gamelens.core.runtime import InvestigationAgent, InvestigationState, LLMPlanner
from gamelens.tools.registry import SteamInvestigationTools
from gamelens.core.llm import LLMConfigurationError, OpenAICompatibleClient
from gamelens.storage.telemetry import TelemetryStore, summarize
from gamelens.core.budget import TaskBudgetStore
from gamelens.storage.runs import RunStore
from gamelens.memory.conversation import build_conversation_memory
from gamelens.analysis.version import load_report


from gamelens.core.paths import PROJECT_ROOT


def save_run_record(
    thread_id: str,
    state: InvestigationState,
    *,
    status: str | None = None,
) -> Path:
    payload = dict(state)
    payload["telemetry"] = summarize(
        TelemetryStore(PROJECT_ROOT / "data" / "telemetry.sqlite").events(thread_id)
    )
    if status is not None:
        payload["status"] = status
    output = PROJECT_ROOT / "data" / "runs" / f"{thread_id}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output,payload)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="游鉴 GameLens：基于 LangGraph 的游戏版本影响调查 Agent"
    )
    parser.add_argument("--appid", type=int)
    parser.add_argument(
        "--goal",
        help="自然语言调查目标，例如：分析最近更新是否导致性能差评上升",
    )
    parser.add_argument("--window-days", type=int, default=7)
    parser.add_argument("--review-count", type=int, default=1000)
    parser.add_argument("--news-count", type=int, default=100)
    parser.add_argument("--classify-count", type=int, default=10)
    parser.add_argument("--patch-news-id")
    parser.add_argument("--profile", choices=["focused", "extended"], default="focused",
                        help="默认单次更新精简调查；extended显式启用旧管线与跨版本工具")
    parser.add_argument("--rag-mode",choices=["off","auto","on"],default="auto",help="增强检索：关闭/自动/启用")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--cache-only", action="store_true", help="仅使用已采集评论，禁止按窗口联网补采")
    parser.add_argument("--force-reclassify", action="store_true")
    parser.add_argument("--thread-id")
    parser.add_argument("--conversation-id")
    parser.add_argument("--recover",action="store_true",help="从持久化图状态恢复中断任务")
    parser.add_argument(
        "--resume",
        choices=("approved", "denied"),
        help="恢复一个等待人工审批的checkpoint",
    )
    parser.add_argument("--approval-comment", default="")
    args = parser.parse_args()
    if args.recover and args.resume is not None:
        parser.error("recover和resume不能同时使用")
    if args.resume is None and not args.recover and (args.appid is None or not args.goal):
        parser.error("新任务必须提供--appid和--goal")
    if args.appid is not None and args.appid <= 0:
        parser.error("appid必须是正整数")
    if not 1 <= args.window_days <= 30:
        parser.error("window-days必须在1到30之间")
    if not 1 <= args.review_count <= 5000:
        parser.error("review-count必须在1到5000之间")
    if not 1 <= args.news_count <= 1000:
        parser.error("news-count必须在1到1000之间")
    if not 1 <= args.classify_count <= 100:
        parser.error("classify-count必须在1到100之间")
    if not 1 <= args.max_steps <= 30:
        parser.error("max-steps必须在1到30之间")
    return args


def build_initial_state(args: argparse.Namespace, memory: dict | None = None) -> InvestigationState:
    memory = memory or {}
    inherited = dict(memory.get("inherited_artifacts", {}))
    previous_config = memory.get("inherited_config", {})
    profile = getattr(args, "profile", "focused")
    selected_patch = args.patch_news_id
    version_path = inherited.get("version_analysis")
    if profile == "focused" and selected_patch is None and not args.refresh_data:
        selected_patch = memory.get("selected_patch_news_id") or previous_config.get("patch_news_id")
    if profile == "extended" and selected_patch is None and version_path and Path(version_path).is_file() and not args.refresh_data:
        previous_report = load_report(Path(version_path))
        if previous_report.appid == args.appid:
            selected_patch = previous_report.patch.news_id
    if args.refresh_data:
        inherited.clear()
    elif previous_config.get("window_days", args.window_days) != args.window_days:
        inherited = {key:value for key,value in inherited.items() if key in {"reviews", "patches"}}
    elif args.force_reclassify or previous_config.get("classify_count", args.classify_count) != args.classify_count:
        inherited = {key:value for key,value in inherited.items() if key not in {"classifications", "impact_summary", "final_report"}}
    inherited.pop("evidence_assessed", None)
    if profile == "focused":
        # Reuse source files, not legacy generated conclusions or stale finish gates.
        inherited = {key: value for key, value in inherited.items() if key in {"reviews", "patches"}}
    return {
        "investigation_profile": profile,
        "thread_id": args.thread_id or uuid.uuid4().hex,
        "goal": args.goal,
        "request_started_at": datetime.now(timezone.utc).isoformat(),
        "rag_mode": getattr(args,"rag_mode","auto"),
        "appid": args.appid,
        "window_days": args.window_days,
        "review_count": args.review_count,
        "news_count": args.news_count,
        "classify_count": args.classify_count,
        "patch_news_id": selected_patch,
        "refresh_data": args.refresh_data,
        "allow_review_acquisition": not getattr(args,"cache_only",False),
        "force_reclassify": args.force_reclassify,
        "max_steps": args.max_steps,
        "step_count": 0,
        "status": "running",
        "plan": [],
        "observations": [],
        "artifacts": inherited,
        "conversation_memory": memory,
        "final_answer": "",
        "error": "",
    }


def main() -> None:
    args = parse_args()
    if (args.resume is not None or args.recover) and not args.thread_id:
        raise SystemExit("恢复任务必须提供--thread-id")
    args.thread_id = args.thread_id or uuid.uuid4().hex
    os.chdir(PROJECT_ROOT)
    telemetry = TelemetryStore(PROJECT_ROOT / "data" / "telemetry.sqlite")
    try:
        try:
            request_limit = int(os.getenv("AGENT_LLM_REQUEST_LIMIT", "32"))
            active_limit = float(os.getenv("AGENT_ACTIVE_SECONDS_LIMIT", "1800"))
        except ValueError as error:
            raise LLMConfigurationError("AGENT_LLM_REQUEST_LIMIT须为整数，AGENT_ACTIVE_SECONDS_LIMIT须为数字") from error
        task_budget = TaskBudgetStore(PROJECT_ROOT / "data" / "telemetry.sqlite",
                                      request_limit=request_limit,
                                      active_seconds_limit=active_limit)
        client = OpenAICompatibleClient(
            model=os.getenv("LLM_MODEL", "").strip(),
            telemetry_sink=lambda event: telemetry.record(args.thread_id, event),
            request_budget=task_budget,
            budget_thread_id=args.thread_id,
        )
        checkpoint_path = (
            PROJECT_ROOT / "data" / "agent_checkpoints.sqlite"
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            checkpoint_path,
            check_same_thread=False,
        )
        checkpointer = SqliteSaver(connection)
        profile = args.profile
        if args.resume is not None or args.recover:
            checkpoint = checkpointer.get_tuple({"configurable": {"thread_id": args.thread_id}})
            if checkpoint is not None:
                # Old checkpoints predate the focused contract and keep their tool set.
                profile = checkpoint.checkpoint.get("channel_values", {}).get("investigation_profile", "extended")
        registry = SteamInvestigationTools(PROJECT_ROOT, client).registry(profile=profile)
        agent = InvestigationAgent(
            LLMPlanner(client),
            registry,
            checkpointer=checkpointer,
            task_budget=task_budget,
            budget_thread_id=args.thread_id,
        )
        result = None
        if args.resume is not None or args.recover:
            if not args.thread_id:
                raise RuntimeError("恢复任务必须提供--thread-id")
            before_resume = agent.snapshot(args.thread_id)
            if not before_resume.values:
                raise RuntimeError("找不到指定thread_id的checkpoint")
            observation_count = len(
                before_resume.values.get("observations", [])
            )
            previous_plan = list(before_resume.values.get("plan", []))
            print("AGENT_SNAPSHOT "+json.dumps(dict(before_resume.values),ensure_ascii=False),flush=True)
            stream = agent.recover_stream(args.thread_id) if args.recover else agent.resume_stream(
                args.thread_id,approved=args.resume=="approved",comment=args.approval_comment)
            thread_id = args.thread_id
        else:
            memory = {}
            if args.conversation_id:
                store = RunStore(PROJECT_ROOT / "data" / "web_runs.sqlite")
                conversation = store.conversation(args.conversation_id)
                if conversation is None or conversation["appid"] != args.appid:
                    raise RuntimeError("会话不存在或游戏AppID不匹配")
                history = [run for run in store.conversation_runs(args.conversation_id) if run["run_id"] != args.thread_id]
                memory = build_conversation_memory(args.conversation_id, history)
            initial_state = build_initial_state(args, memory)
            observation_count = 0
            previous_plan: list[str] = []
            stream = agent.stream(initial_state)
            thread_id = initial_state["thread_id"]

        for snapshot in stream:
            result = snapshot
            if snapshot.get("status")=="repairing":
                print("AGENT_REPAIR "+json.dumps({"attempt":snapshot.get("validation_retries",0),
                    "errors":snapshot.get("validation_errors",[])},ensure_ascii=False),flush=True)
            plan = snapshot.get("plan", [])
            if plan and plan != previous_plan:
                print(
                    "AGENT_PLAN "
                    + json.dumps(plan, ensure_ascii=False),
                    flush=True,
                )
                previous_plan = list(plan)
            observations = snapshot.get("observations", [])
            for observation in observations[observation_count:]:
                print(
                    "AGENT_EVENT "
                    + json.dumps(observation, ensure_ascii=False),
                    flush=True,
                )
            observation_count = len(observations)
        persisted = agent.snapshot(thread_id)
        result = persisted.values or result
        if result is None:
            raise RuntimeError("Agent没有返回运行状态")
        if persisted.interrupts:
            approval = persisted.interrupts[0].value
            result = {**result, "task_budget":agent.budget_snapshot(thread_id)}
            run_record = save_run_record(
                thread_id,
                result,
                status="awaiting_approval",
            )
            print(
                "AGENT_INTERRUPT "
                + json.dumps(approval, ensure_ascii=False),
                flush=True,
            )
            print(
                "AGENT_FINAL "
                + json.dumps(
                    {
                        "status": "awaiting_approval",
                        "answer": "",
                        "artifacts": result.get("artifacts", {}),
                        "thread_id": thread_id,
                        "run_record": str(run_record),
                        "decision_trace": result.get("decision_trace", []),
                        "investigation_progress": result.get("investigation_progress", {}),
                        "plan_state": result.get("plan_state", {}),
                        "loop_control": result.get("loop_control", {}),
                        "budget": agent.budget_snapshot(thread_id),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            connection.close()
            return
    except LLMConfigurationError as error:
        raise SystemExit(f"Agent配置错误：{error}") from error

    print("\n=== Agent执行轨迹 ===")
    for observation in result.get("observations", []):
        marker = "OK" if observation["ok"] else "FAIL"
        print(
            f"[{marker}] [{observation['step']}] "
            f"{observation['tool']}: {observation['summary']}"
        )
    print(f"\n状态：{result['status']}")
    print(result.get("final_answer") or result.get("error", ""))
    result = {**result, "task_budget":agent.budget_snapshot(thread_id)}
    run_record = save_run_record(thread_id, result)
    print(
        "AGENT_FINAL "
        + json.dumps(
            {
                "status": result["status"],
                "answer": result.get("final_answer", ""),
                "artifacts": result.get("artifacts", {}),
                "thread_id": thread_id,
                "run_record": str(run_record),
                "decision_trace": result.get("decision_trace", []),
                "investigation_progress": result.get("investigation_progress", {}),
                "plan_state": result.get("plan_state", {}),
                "loop_control": result.get("loop_control", {}),
                "budget": agent.budget_snapshot(thread_id),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    connection.close()


if __name__ == "__main__":
    main()
