"""Reproducible offline baselines, opt-in live planning, and no-API replay scoring."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from gamelens.core.paths import PROJECT_ROOT
from gamelens.evaluation.business_cases import CASES, DATASET_ID, materialize_dataset
from gamelens.evaluation.quality import (
    BusinessCase, aggregate_business_results, digest, evaluate_business_run, review_template,
)
from gamelens.storage.atomic_io import atomic_write_json, atomic_write_text

ALLOWED_TOOLS = {"inspect_data", "query_reviews", "get_review", "compare_cohorts",
                 "compare_versions", "verify_investigation", "list_patches", "investigate_recent_versions"}


def load_dataset(root):
    raw = root / "data" / "raw"
    files = {"reviews": raw / "730_reviews.json", "patches": raw / "730_patch_candidates.json",
             "news": raw / "730_news.json"}
    hashes = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in files.items()}
    return {"id": DATASET_ID, "sha256": digest(hashes), "source_hashes": hashes,
            "reviews": json.loads(files["reviews"].read_text(encoding="utf-8")),
            "patches": json.loads(files["patches"].read_text(encoding="utf-8"))}


def evaluation_tags(case, dataset):
    return {"case_id": case.id, "dataset_id": DATASET_ID,
            "dataset_sha256": dataset["sha256"], "case_sha256": digest(case.model_dump(mode="json"))}


def task_state(case, index, dataset, rag_mode=None, execution_mode="full"):
    from gamelens.evaluation.benchmark import initial_state
    state = initial_state(index, case.goal)
    state.update(window_days=1, review_count=15, news_count=5, patch_news_id="p5",
                 rag_mode="off" if case.rag_mode == "off" else (rag_mode or "auto"),
                 max_steps=case.max_tool_steps, evaluation=evaluation_tags(case, dataset),
                 conversation_memory={"recent_turns": case.history,
                     "trust_note": "历史模型回答不是事实，必须重新核验原文。"})
    state["optimization_mode"] = "auto" if execution_mode == "hybrid" else "off"
    return state


def business_registry(root, client, with_rag=False):
    from gamelens.tools.registry import SteamInvestigationTools
    # The historical benchmark intentionally includes cross-version/legacy tasks.
    registry = SteamInvestigationTools(root, client).registry(profile="extended").subset(ALLOWED_TOOLS)
    if with_rag:
        from gamelens.core.runtime import AgentTool, ToolExecutionPolicy, ToolRegistry
        from gamelens.rag.embedding import LocalEmbedding
        from gamelens.rag.tools import IndexArgs, RagTools, SearchArgs

        class CachedEncoder(LocalEmbedding):
            def load(self, download=False):
                # Never download weights inside a paid evaluation run.
                return super().load(download=False)

        encoder = CachedEncoder(PROJECT_ROOT)
        if not encoder.ready:
            raise ValueError("请先单独准备本地Embedding模型；评测不会自动下载模型")
        rag = RagTools(root, encoder)
        full = SteamInvestigationTools(root, client).registry().specifications()
        descriptions = {s["name"]: s["description"] for s in full}
        specs = [AgentTool("build_game_index", descriptions["build_game_index"], IndexArgs, rag.build_game_index,
                          execution_policy=ToolExecutionPolicy(timeout_seconds=600)),
                 AgentTool("search_game_evidence", descriptions["search_game_evidence"], SearchArgs, rag.search_game_evidence)]
        # Build from the public allowlist selection; tools remain trusted handlers.
        registry = ToolRegistry(list(registry.tools) + specs, receipt_root=root / "data/tool_receipts")
    return registry


def oracle_business_case(case, registry, state):
    observations = []
    for step, planned in enumerate(case.oracle_plan, 1):
        state["step_count"] = step - 1
        result = registry.execute(planned["tool"], planned["arguments"], state)
        observations.append({"step": step, "tool": planned["tool"],
                             "arguments": planned["arguments"], **result.model_dump(mode="json")})
        state["artifacts"].update(result.artifacts)
    # Do not manufacture a model answer, citations or semantic success for this baseline.
    checks = []
    from gamelens.evaluation.quality import _fact_matches
    for requirement in case.facts:
        checks.append({"tool": requirement.tool, "path": requirement.path, "expected": requirement.value,
                       "passed": any(_fact_matches(o, requirement) for o in observations)})
    return {"case_id": case.id, "category": case.category, "split": case.split,
            "passed": all(c["passed"] for c in checks) and all(o["ok"] for o in observations),
            "checks": checks, "mode": "tool_oracle", "tests_answer_semantics": False}


def component_baselines():
    from gamelens.evaluation.benchmark import fixture_registry, initial_state, reference_run
    from gamelens.evaluation.effects import score_effect_case
    cases = json.loads((PROJECT_ROOT / "evals/investigation_cases.json").read_text(encoding="utf-8"))["cases"]
    results = []
    with tempfile.TemporaryDirectory(prefix="gamelens-component-eval-") as directory:
        root = Path(directory)
        raw = root / "data/raw"
        raw.mkdir(parents=True)
        for path in (PROJECT_ROOT / "evals/fixtures").glob("*.json"):
            shutil.copyfile(path, raw / path.name)
        corpus = json.loads((raw / "730_reviews.json").read_text(encoding="utf-8"))
        registry = fixture_registry(root)
        for mode in ("tool_oracle", "fixed"):
            scores = []
            for index, case in enumerate(cases):
                state = initial_state(index, case["goal"])
                result = reference_run(registry, case, state, fixed=mode == "fixed")
                scores.append(score_effect_case(case, result, corpus))
            results.append({"mode": mode, "cases": len(scores), "passed": sum(r["passed"] for r in scores),
                            "rate": sum(r["passed"] for r in scores) / len(scores), "results": scores,
                            "measures": "固定参数下的工具与结构化事实；不是模型回答准确率"})
    return results


def review_for(review_dir, run_name):
    path = review_dir / f"{run_name}.json" if review_dir else None
    return json.loads(path.read_text(encoding="utf-8")) if path and path.is_file() else None


def percentile(values, percent):
    return sorted(values)[max(0, math.ceil(len(values) * percent) - 1)] if values else None


def runtime_summary(results):
    times = [r["elapsed_seconds"] for r in results if r.get("elapsed_seconds") is not None]
    telemetry = [r.get("telemetry", {}) for r in results]
    complete = bool(telemetry) and all(t.get("estimated_cost") is not None for t in telemetry)
    currencies = {t.get("currency") for t in telemetry}
    return {"latency_cases": len(times), "latency_p50_seconds": percentile(times, .5),
            "latency_p95_seconds": percentile(times, .95),
            "estimated_cost": sum(t["estimated_cost"] for t in telemetry) if complete and len(currencies) == 1 else None,
            "currency": next(iter(currencies)) if complete and len(currencies) == 1 else None,
            "note": "缺失数据不是零；耗时仅针对本次固定合成资料任务，费用不是供应商账单。"}


def condition_summary(results):
    """Descriptive ablations only, not a causal improvement claim on mixed runs."""
    groups = {}
    for row in results:
        if row["execution_mode"] == "agent":
            key = row.get("condition", "unknown")
            groups.setdefault(key, []).append(row)
    return [{"condition": key, "runs": len(rows),
             "automatic_passed": sum(r["automatic_passed"] for r in rows),
             "accepted": sum(r["accepted"] for r in rows),
             "reviewed": sum(r["human_review"]["status"] == "reviewed" for r in rows),
             "case_ids": sorted({r["case_id"] for r in rows}),
             **runtime_summary(rows)} for key, rows in sorted(groups.items())]


def render_report(payload):
    summary = payload["summary"]
    lines = ["# GameLens 系统评测报告", "", f"生成时间：{payload['created_at']}",
             f"评测资料：{DATASET_ID}，15条人工编写评论、5次虚构更新。不是实际Steam游戏事实。", "",
             "## 结论与证据边界", "",
             "本报告区分工具契约、模型任务执行和人工语义验收。离线工具通过率不能作为Agent准确率；真实Agent运行和人工审核未完成时，最终回答效果尚未得到验证。", "",
             "## 离线工具与对照基线", "",
             "| 模式 | 案例数 | 通过数 | 通过率 |", "|---|---:|---:|---:|"]
    for baseline in payload["component_baselines"]:
        lines.append(f"| {baseline['mode']} | {baseline['cases']} | {baseline['passed']} | {baseline['rate']:.1%} |")
    oracle = payload["business_oracle"]
    if oracle:
        lines.append(f"| 业务任务的预设工具计划 | {len(oracle)} | {sum(r['passed'] for r in oracle)} | {sum(r['passed'] for r in oracle)/len(oracle):.1%} |")
    lines += ["", "tool_oracle使用人工指定参数；fixed仅保留时间窗口过滤，是能力受限的参考执行，并不是原有完整流水线，也不是LLM-only基线。两者的差异不能证明Agent或RAG带来的提升。", "",
              "## 真实Agent与人工验收", "",
              f"计划问题：{summary['planned_cases']}；真实Agent运行：{summary['evaluated_agent_runs']}；人工已审核：{summary['human_reviewed_runs']}；完整验收通过：{summary['accepted_runs']}。",
              "已执行记录的最终验收率：" + (f"{summary['fully_reviewed_acceptance_rate']:.1%}" if summary['fully_reviewed_acceptance_rate'] is not None else "尚不可报告（未运行或审核未完成）"), "",
              "| 问题 | 类别 | 状态 |", "|---|---|---|"]
    by_case = {}
    for row in payload["business_results"]:
        by_case.setdefault(row["case_id"], []).append(row)
    for case in payload["cases"]:
        rows = by_case.get(case["id"], [])
        status = "未进行真实模型评测" if not rows else "; ".join(r["status"] for r in rows)
        lines.append(f"| {case['id']} | {case['category']} | {status} |")
    lines += ["", "## RAG条件对照", "",
              "| 条件（模型 / RAG） | 运行数 | 自动通过 | 完整验收 | 人工已审 | P50秒 |",
              "|---|---:|---:|---:|---:|---:|"]
    for group in payload.get("conditions", []):
        lines.append(f"| {group['condition']} | {group['runs']} | {group['automatic_passed']} | {group['accepted']} | {group['reviewed']} | {group['latency_p50_seconds']} |")
    lines += ["", "没有运行结果时不报告对照收益。比较时须使用相同资料、问题、模型配置和重复次数，检查双方题目覆盖一致；这里只做描述性汇总，不自动证明RAG提升。", ""]
    lines += ["", "## 耗时与费用", ""]
    timing = payload["runtime"]
    for label, key in (("任务P50耗时（秒）", "latency_p50_seconds"), ("任务P95耗时（秒）", "latency_p95_seconds"), ("模型费用估算", "estimated_cost")):
        lines.append(f"- {label}：{timing[key] if timing[key] is not None else '未知 / 未测量'}")
    lines += ["", "P95采用最近秩方法；样本少时应同时查看逐任务耗时。费用不包含人工、CPU、Embedding时间或第三方实际账单。", "",
              "## 失败分析与下一步", ""]
    failures = [r for r in payload["business_results"] if not r["accepted"]]
    if not failures:
        lines.append("当前没有已评分的失败记录，不代表系统没有错误。下一步需要真实模型运行、人工审核和真实评论标注集。")
    for row in failures:
        checks = [k for k, passed in row["checks"].items() if not passed]
        lines.append(f"- {row['run_name']}：{row['status']}；自动检查失败项：{', '.join(checks) or '无'}；人工状态：{row['human_review']['status']}。")
    lines += ["", "## 尚未证明的实际价值", "",
              "目标是减少人工翻阅评论、对照公告和整理证据的工作量。当前不能声称提升真实业务效率或准确率：需要真实游戏问题、人工完成时间、答案质量以及同资料同配置对照实验。合成集的holdout划分仅用于流程演练，不是外部独立盲测。", "",
              "本项目保持本地单用户定位，不包含公开上线、账号或多用户生产服务验证。"]
    return "\n".join(lines) + "\n"


def write_outputs(output, payload):
    atomic_write_json(output / "summary.json", payload)
    atomic_write_text(output / "report.md", render_report(payload))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="GameLens任务级评测：默认离线、不调用模型")
    parser.add_argument("--mode", choices=("offline", "agent", "replay"), default="offline")
    parser.add_argument("--allow-paid-model", action="store_true", help="明确允许调用.env配置的供应商模型")
    parser.add_argument("--limit", type=int, default=3, help="真实模型默认只选3道题")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--case-ids", nargs="+", help="只运行指定题目，不能修改标准答案")
    parser.add_argument("--split", choices=("all", "development", "holdout"), default="all")
    parser.add_argument("--rag-mode", choices=("off", "auto", "on"), default="off")
    parser.add_argument("--execution-mode", choices=("full", "hybrid"), default="full", help="full保留全模型规划基线；hybrid启用统计快路径")
    parser.add_argument("--request-timeout", type=float, default=90, help="评测请求超时，独立于.env中的应用设置")
    parser.add_argument("--max-model-calls", type=int, default=30)
    parser.add_argument("--runs-manifest", type=Path)
    parser.add_argument("--review-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.case_ids and (len(set(args.case_ids)) != len(args.case_ids) or set(args.case_ids)-{c["id"] for c in CASES}):
        parser.error("case-ids必须是唯一且存在的题目ID")
    if not 1 <= args.limit <= len(CASES) or not 1 <= args.repeats <= 3 or not 1 <= args.max_model_calls <= 200:
        parser.error("limit范围1..12，repeats范围1..3，max-model-calls范围1..200")
    if not math.isfinite(args.request_timeout) or not 1 <= args.request_timeout <= 600:
        parser.error("request-timeout必须在1..600秒")
    if args.mode == "agent" and not args.allow_paid_model:
        parser.error("真实模型评测会产生费用，必须显式传入--allow-paid-model")
    if args.mode == "replay" and not args.runs_manifest:
        parser.error("replay需要--runs-manifest")
    if args.mode != "agent" and args.rag_mode != "off":
        parser.error("rag-mode仅用于agent模式；离线和replay不会执行Embedding")
    return args


def main(argv=None):
    args = parse_args(argv)
    created = datetime.now(timezone.utc)
    output = args.output_dir or PROJECT_ROOT / "evals/results/quality" / created.strftime("%Y%m%d_%H%M%S_%f")
    if (output / "summary.json").exists():
        raise SystemExit("输出目录已有评测报告，请换一个目录，避免覆盖历史基线")
    cases = [BusinessCase.model_validate(c) for c in CASES if (args.split == "all" or c["split"] == args.split) and (not args.case_ids or c["id"] in args.case_ids)]
    output.mkdir(parents=True, exist_ok=True)
    materialize_dataset(output / "fixture")
    dataset = load_dataset(output / "fixture")
    payload = {"created_at": created.isoformat(), "mode": args.mode, "dataset_id": DATASET_ID,
               "dataset_sha256": dataset["sha256"], "cases": [c.model_dump(mode="json") for c in cases],
               "component_baselines": [], "business_oracle": [], "business_results": [],
               "model_configuration": None,
               "limitations": ["人工合成资料", "holdout不是外部独立盲测", "不验证线上Steam事实", "不验证生产负载"]}
    # Never create a configured model client for default/replay paths.
    if args.mode == "offline":
        print("运行34项工具基线和时间窗口参考对照，不调用模型……", flush=True)
        payload["component_baselines"] = component_baselines()
        registry = business_registry(output / "fixture", client=None)
        for index, case in enumerate(cases):
            result = oracle_business_case(case, registry, task_state(case, index, dataset))
            payload["business_oracle"].append(result)
    elif args.mode == "agent":
        from gamelens.core.llm import OpenAICompatibleClient
        from gamelens.core.runtime import InvestigationAgent, LLMPlanner
        from gamelens.storage.telemetry import summarize

        if args.rag_mode != "off":
            import sys
            from gamelens.bootstrap import require_environment
            require_environment(sys.executable, include_rag=True)

        class BudgetedClient(OpenAICompatibleClient):
            calls = 0
            requests = 0

            def _emit(self, event):
                if event.get("kind") == "request" and event.get("status") == "pending":
                    if self.requests >= args.max_model_calls:
                        raise RuntimeError("评测HTTP请求次数预算已用尽（包含格式回退）")
                    self.requests += 1
                return super()._emit(event)

            def generate_json(self, messages):
                if self.calls >= args.max_model_calls:
                    raise RuntimeError("评测模型调用次数预算已用尽")
                self.calls += 1
                return super().generate_json(messages)

        events = {}
        client = BudgetedClient(os.getenv("LLM_MODEL", ""), timeout=args.request_timeout, max_retries=0,
                                telemetry_sink=lambda e: events.__setitem__(e["id"], e))
        # Only model name/runtime versions; never serialize keys or the .env file.
        payload["model_configuration"] = {"model": client.model, "rag_mode": args.rag_mode,
            "max_model_calls": args.max_model_calls, "http_retries": 0, "temperature": 0,
            "execution_mode": args.execution_mode, "request_timeout": args.request_timeout,
            "langgraph_version": importlib.metadata.version("langgraph"),
            "pydantic_version": importlib.metadata.version("pydantic")}
        selected = cases[:args.limit]
        for case in selected:
            for repeat in range(args.repeats):
                run_name = f"{case.id}_{repeat+1}"
                with tempfile.TemporaryDirectory(prefix="gamelens-task-eval-") as directory:
                    root = Path(directory)
                    materialize_dataset(root)
                    registry = business_registry(root, client, args.rag_mode != "off")
                    state = task_state(case, run_name, dataset, args.rag_mode, args.execution_mode)
                    before_ids = set(events)
                    start = time.monotonic()
                    print(f"真实模型任务：{run_name}", flush=True)
                    result = state
                    try:
                        for snapshot in InvestigationAgent(LLMPlanner(client), registry).stream(state):
                            result = dict(snapshot)
                    except Exception as error:
                        result = {**result, "status": "failed", "error": f"{type(error).__name__}: {error}"}
                    result["evaluation"] = evaluation_tags(case, dataset)
                    result["evaluation"]["model"] = client.model
                    result["evaluation"]["execution_mode"] = args.execution_mode
                    elapsed = round(time.monotonic() - start, 3)
                    result["telemetry"] = summarize([v for k, v in events.items() if k not in before_ids])
                    report_path = result.get("artifacts", {}).get("final_report")
                    if report_path and Path(report_path).is_file():
                        saved_report = output / "agent_reports" / f"{run_name}.md"
                        atomic_write_text(saved_report, Path(report_path).read_text(encoding="utf-8"))
                        result["artifacts"]["final_report"] = str(saved_report.resolve())
                        if result.get("generation_pending"):
                            result["artifacts"]["partial_report"] = str(saved_report.resolve())
                    atomic_write_json(output / "runs" / f"{run_name}.json", result)
                    row = evaluate_business_run(case, result, dataset, review_for(args.review_dir, run_name))
                    row.update(run_name=run_name, elapsed_seconds=elapsed, telemetry=result["telemetry"],
                               condition=f"{client.model} / RAG-{state['rag_mode']} / {args.execution_mode}")
                    payload["business_results"].append(row)
                    atomic_write_json(output / "reviews" / f"{run_name}.json", review_template(case, result, dataset["sha256"]))
    else:
        manifest = json.loads(args.runs_manifest.read_text(encoding="utf-8"))
        if manifest.get("dataset_sha256") != dataset["sha256"]:
            raise SystemExit("运行清单的数据集指纹不匹配，不能将实际游戏结果套用合成标准答案")
        lookup = {c.id: c for c in cases}
        names = set()
        for item in manifest["runs"]:
            case = lookup[item["case_id"]]
            run_name = item["run_name"]
            import re
            if not re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", run_name) or run_name in names:
                raise ValueError("run_name必须唯一且不含路径字符")
            names.add(run_name)
            path = Path(item["path"])
            if not path.is_absolute():
                path = args.runs_manifest.parent / path
            state = json.loads(path.read_text(encoding="utf-8"))
            row = evaluate_business_run(case, state, dataset, review_for(args.review_dir, run_name),
                                        execution_mode=item.get("execution_mode", "agent"))
            row.update(run_name=run_name, elapsed_seconds=item.get("elapsed_seconds"), telemetry=state.get("telemetry", {}),
                       condition=f"{state.get('evaluation', {}).get('model', 'unknown')} / RAG-{state.get('rag_mode', 'unknown')} / {state.get('evaluation', {}).get('execution_mode', 'full')}")
            payload["business_results"].append(row)
            atomic_write_json(output / "reviews" / f"{run_name}.json", review_template(case, state, dataset["sha256"]))
    payload["summary"] = aggregate_business_results(payload["business_results"], len(cases))
    payload["runtime"] = runtime_summary(payload["business_results"])
    payload["conditions"] = condition_summary(payload["business_results"])
    write_outputs(output, payload)
    if args.mode == "agent":
        atomic_write_json(output / "runs_manifest.json", {"dataset_sha256": dataset["sha256"], "runs": [
            {"case_id": row["case_id"], "run_name": row["run_name"], "path": f"runs/{row['run_name']}.json",
             "execution_mode": "agent", "elapsed_seconds": row["elapsed_seconds"]} for row in payload["business_results"]]})
    print(f"评测报告：{output / 'report.md'}", flush=True)
    if args.mode == "offline":
        # The deliberately limited fixed baseline is allowed to fail.
        oracle_ok = all(b["passed"] == b["cases"] for b in payload["component_baselines"] if b["mode"] == "tool_oracle")
        return 0 if oracle_ok and all(r["passed"] for r in payload["business_oracle"]) else 1
    if any(not r["automatic_passed"] or r["human_review"]["passed"] is False for r in payload["business_results"]):
        return 1
    return 0 if payload["business_results"] and all(r["accepted"] for r in payload["business_results"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
