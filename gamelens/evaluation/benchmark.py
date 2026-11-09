"""Run fixed ground-truth cases. Agent mode is explicitly opt-in and uses API tokens."""
import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

from gamelens.core.runtime import InvestigationAgent,LLMPlanner,ToolRegistry,AgentTool,ToolResult
from gamelens.tools.investigation import (ReviewInvestigation,QueryReviewsArgs,GetReviewArgs,CompareCohortsArgs,CompareVersionsArgs,VerifyInvestigationArgs)
from gamelens.core.llm import OpenAICompatibleClient
from gamelens.storage.atomic_io import atomic_write_json
from gamelens.evaluation.effects import score_effect_case
from gamelens.storage.telemetry import summarize

from gamelens.core.paths import PROJECT_ROOT as ROOT


def fixture_registry(root):
    domain=ReviewInvestigation(root)
    specs=[("query_reviews",QueryReviewsArgs),("get_review",GetReviewArgs),("compare_cohorts",CompareCohortsArgs),("compare_versions",CompareVersionsArgs),("verify_investigation",VerifyInvestigationArgs)]
    return ToolRegistry([AgentTool(name,name,model,getattr(domain,name)) for name,model in specs],receipt_root=root/"data"/"tool_receipts")


def initial_state(index,goal):
    return {"thread_id":f"eval-{index}","goal":goal,"appid":730,"window_days":7,"review_count":24,
        "news_count":2,"classify_count":10,"refresh_data":False,"force_reclassify":False,
        "allow_review_acquisition":False,
        "patch_news_id":"p1","max_steps":12,"step_count":0,"status":"running","plan":[],
        "observations":[],"artifacts":{},"final_answer":"","error":""}


def reference_run(registry,case,state,fixed=False):
    arguments=dict(case["arguments"])
    if fixed and case["tool"]=="query_reviews":
        # Baseline: time-window statistics only, without advanced cohort/keyword filters.
        arguments["filters"]={"period":arguments.get("filters",{}).get("period","after")}
    result=registry.execute(case["tool"],arguments,state)
    state["observations"]=[{"step":1,"tool":case["tool"],**result.model_dump(mode="json")}]
    state["status"]="completed" if result.ok else "failed"
    state["final_answer"]="固定参考执行；"+result.summary
    claims=[]
    data=result.data
    paths=[]
    if "stats" in data: paths=[["stats","count"],["stats","positive_rate"]]
    if "cohort_a" in data: paths=[[group,"stats",key] for group in ("cohort_a","cohort_b") for key in ("count","positive_rate")]
    if "versions" in data: paths=[["versions",index,"stats",key] for index in (0,1) for key in ("count","positive_rate")]
    if "review" in data: paths=[["review","review_id"]]
    from gamelens.core.grounding import resolve_path
    for path in paths: claims.append({"label":"/".join(map(str,path)),"source_step":1,"fact_path":path,"value":resolve_path(data,path)})
    reviews=data.get("reviews",[]) + ([data["review"]] if "review" in data else [])
    state["final_claims"]=claims
    state["final_citations"]=[{"source_step":1,"review_id":reviews[0]["review_id"],"quote":reviews[0]["text"][:150]}] if reviews else []
    return state


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=("tool_oracle","fixed","agent"),default="tool_oracle")
    parser.add_argument("--limit",type=int,default=34)
    parser.add_argument("--output",type=Path,default=ROOT/"evals"/"results"/"latest.json")
    args=parser.parse_args()
    if not 1<=args.limit<=34: parser.error("limit必须为1..34")
    cases=json.loads((ROOT/"evals"/"investigation_cases.json").read_text(encoding="utf-8"))["cases"][:args.limit]
    corpus=json.loads((ROOT/"evals"/"fixtures"/"730_reviews.json").read_text(encoding="utf-8"))
    events=[]
    results=[]
    started=time.monotonic()
    with tempfile.TemporaryDirectory(prefix="gamelens-eval-") as directory:
        root=Path(directory)
        raw=root/"data"/"raw"
        raw.mkdir(parents=True)
        for path in (ROOT/"evals"/"fixtures").glob("*.json"): shutil.copyfile(path,raw/path.name)
        registry=fixture_registry(root)
        if args.mode=="agent":
            import os
            client=OpenAICompatibleClient(os.getenv("LLM_MODEL",""),telemetry_sink=lambda event:events.append(event))
            agent=InvestigationAgent(LLMPlanner(client),registry)
        for index,case in enumerate(cases):
            before=time.monotonic()
            state=initial_state(index,case["goal"])
            state["conversation_memory"]={"recent_turns":case.get("history",[]),"trust_note":"历史回答仅用于理解指代，需重新核验工具事实。"}
            try:
                if args.mode=="agent": state=agent.invoke(state)
                else: state=reference_run(registry,case,state,fixed=args.mode=="fixed")
                score=score_effect_case(case,state,corpus)
                atomic_write_json(args.output.parent/"runs"/f"{args.mode}_{case['name']}.json",state)
            except Exception as error:
                score={"name":case["name"],"passed":False,"error":f"{type(error).__name__}: {error}"}
            score["elapsed_ms"]=round((time.monotonic()-before)*1000)
            results.append(score)
    # Upsert telemetry events because each HTTP request has pending and completed versions.
    usage=summarize(list({event["id"]:event for event in events}.values()))
    payload={"mode":args.mode,"cases":len(results),"passed":sum(r["passed"] for r in results),
        "success_rate":sum(r["passed"] for r in results)/len(results),"elapsed_seconds":round(time.monotonic()-started,2),
        "telemetry":usage,"results":results,"note":"tool_oracle不测LLM规划；fixed是仅时间窗口过滤的参考执行，不是完整线上流水线；agent才调用真实模型。自然语言因果表述仍需人工复核。"}
    atomic_write_json(args.output,payload)
    print(json.dumps({k:v for k,v in payload.items() if k!="results"},ensure_ascii=False,indent=2))
    raise SystemExit(0 if payload["passed"]==payload["cases"] else 1)


if __name__=="__main__": main()
