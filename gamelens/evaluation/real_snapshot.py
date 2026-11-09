"""Frozen, purposefully selected real cached evidence; never an external blind test.

No Steam requests. Preparation copies only explicitly listed public review IDs.
Annotations were read by the coding assistant, not independently human-reviewed.
"""
import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from gamelens.core.paths import PROJECT_ROOT
from gamelens.storage.atomic_io import atomic_write_json, atomic_write_text
from gamelens.evaluation.quality import BusinessCase, digest, evaluate_business_run, review_template


class SnapshotUnavailableError(RuntimeError):
    """The frozen real-data snapshot cannot be prepared from local files."""

IDS = {
    730: ["236619456", "236581306", "236549111", "236394705", "236383290", "236240052", "236201057", "236155620", "236132158", "235945546", "235887886", "235593328", "235071339", "235069522", "235028337", "234850885"],
    1172470: ["236854265", "236836265", "236774671", "236771530", "236703035", "236678299", "236624850", "236619902", "236607732", "236606001"],
}


def group(name, ids):
    return {"name":name,"kind":"review","ids":ids,"min_distinct":len(ids)}


def case(id, appid, goal, groups, rubric, rag="auto", plan=None):
    return {"appid":appid,"rag":rag,"case":{"id":id,"category":"真实缓存调查","split":"development", "goal":goal,
        "oracle_plan":plan or [{"tool":"get_review","arguments":{"appid":appid,"review_id":rid}}
                                 for g in groups for rid in g["ids"]],
        "facts":[],"evidence_groups":groups,"rubric":rubric,"max_tool_steps":12}}


TASKS = [
    case("real_cross_versions",730,"调查最近五次更新窗口的掉帧/卡顿反馈，找不同窗口原文及流畅的反例，不能直接认定同一缺陷一直存在。",[group("不同窗口反馈",["236619456","236383290","236155620"]),group("流畅反例",["236201057"])],"对照真实窗口；区分反复反馈与同一缺陷持续；9月28日评论9月30日改过，正文不证明9月28日的状态。",rag="on"),
    case("real_positive_complaints",730,"找出点了推荐却抱怨性能或服务器的原文，解释为什么不能把推荐态度当成没有问题。",[group("正面投票中的抱怨",["236581306","235945546"])],"服务器崩溃及没有之前流畅都是负面内容；不推断全部玩家。"),
    case("real_historical_edit",730,"核对评论236383290的创建、更新时间及原文。能否用现在的正文证明9月28日的具体性能状态？",[group("后来修改的评论",["236383290"])],"创建9月28日，修改9月30日；缺少历史正文，不可声称当前文本等于当时文本。",rag="off"),
    case("real_distinct_crashes",1172470,"找启动阶段闪退和对局决赛圈闪退的原文，区分故障场景。这些反馈能否证明同一个技术原因？",[group("启动场景",["236606001"]),group("对局场景",["236703035"])],"反作弊加载后与决赛圈不同；没有日志或复现不能证明同因。",rag="on"),
    case("real_match_scenarios",1172470,"区分匹配等待时间长和进入匹配后掉线的反馈，给出两类的原文，不要把它们混为同一种问题。",[group("等待时间",["236836265"]),group("进入后掉线",["236771530"])],"等待超过5分钟与进入后连续掉线是不同描述；不能由推荐态度推断场景。"),
    case("real_absence",1172470,"缓存中有明确包含“存档损坏”的评论吗？没有关键词命中能否证明游戏没有存档问题？",[],"字面匹配0，不代表故障不存在，也不代表语义检索必定没有不同措辞。",rag="off",plan=[{"tool":"query_reviews","arguments":{"appid":1172470,"filters":{"period":"all","keywords":["存档损坏"]}}}]),
    case("real_sarcasm",730,"读取评论236132158，解释推荐投票与这段讽刺文本的区别；不要照字面说玩家赞同更新。",[group("讽刺",["236132158"])],"调不回准心、新手改坏习惯等是讽刺线索，模型解释需人工审核。"),
    case("real_counterexample",730,"比较236155620和235069522的性能描述，能否据此说所有玩家都掉帧？",[group("抱怨",["236155620"]),group("正常反例",["235069522"])],"有掉帧与老电脑流畅两种反馈；不同时间和硬件，不能全体推广。"),
    case("real_network_punishment",1172470,"查找反作弊掉线后被扣分或罚禁赛的反馈，给原文，区分玩家报告与已验证的系统责任。",[group("反馈",["236854265","236607732"])],"是玩家描述，无法确认触发原因和真实责任。"),
    case("real_reconnection",1172470,"评论236703035说了哪些重连和惩罚问题？哪些是直接原文，哪些原因还不能确认？",[group("重连反馈",["236703035"])],"无法重连、弃赛和10分钟惩罚是原文；服务器为根因未验证。"),
    case("real_playtime_basis",730,"统计本地评价时100小时以上玩家的评论数量和推荐率，不能使用当前累计时长。",[],"必须at_review，结果仅适用选取的16条评论，不能当成Steam总体。",rag="off",plan=[{"tool":"query_reviews","arguments":{"appid":730,"filters":{"period":"all","min_playtime_hours":100,"playtime_basis":"at_review"}}}]),
    case("real_zero_population",1172470,"查询2100年1月1日至1月2日UTC的缓存评论。空结果是什么含义？",[],"在观察时间范围之外；无覆盖不等于当时无玩家或无缺陷。",rag="off",plan=[{"tool":"query_reviews","arguments":{"appid":1172470,"filters":{"period":"all","start_at":"2100-01-01T00:00:00Z","end_at":"2100-01-02T00:00:00Z"}}}]),
]

RETRIEVAL_CASES = [
    (730,"开枪时卡顿",["236619456"],["235069522"]),
    (730,"玩着玩着闪退，更新后一直退出",["236549111","236394705","234850885"],[]),
    (730,"老电脑也流畅，优化挺好",["235069522","236201057"],["235945546","236155620"]),
    (730,"更新后没有之前流畅，性能退步",["235945546","236383290"],["235069522"]),
    (730,"更新准心调不回，讽刺玩家最希望的改动",["236132158"],[]),
    (730,"电脑画面变糊键鼠不受控制，更新掉帧",["236383290"],[]),
    (1172470,"反作弊加载完成就闪退无法启动",["236606001","236624850"],["236703035"]),
    (1172470,"决赛圈闪退不能重连，还被弃赛处罚",["236703035"],["236606001"]),
    (1172470,"一进入匹配就连续掉线",["236771530"],[]),
    (1172470,"匹配要等很久超过五分钟",["236836265","236678299"],[]),
    (1172470,"反作弊异常掉线扣分罚禁赛",["236854265","236607732"],[]),
    (1172470,"存档损坏无法读取进度",[],IDS[1172470]),
]


def prepare(root):
    if (root/"manifest.json").exists():
        validate_snapshot(root)
        return json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    required = [PROJECT_ROOT/f"data/raw/{appid}_{suffix}.json"
                for appid in IDS for suffix in ("reviews", "patch_candidates", "news")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SnapshotUnavailableError(
            "真实缓存快照缺少本地来源；离线评测不会自动获取或替换数据：" + ", ".join(missing))
    for appid, selected in IDS.items():
        review_path = PROJECT_ROOT/f"data/raw/{appid}_reviews.json"
        values = json.loads(review_path.read_text(encoding="utf-8"))
        available = {row.get("review_id") for row in values}
        if set(selected) - available:
            raise SnapshotUnavailableError(
                f"AppID {appid}的选定评论不在本地缓存；不会自动获取或替换标注")
    files, upstream = {}, {}
    for appid, selected in IDS.items():
        for suffix in ("reviews","patch_candidates","news"):
            path=PROJECT_ROOT/f"data/raw/{appid}_{suffix}.json"
            values=json.loads(path.read_text(encoding="utf-8"))
            upstream[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
            if suffix=="reviews":
                by_id={r["review_id"]:r for r in values}
                values=[by_id[rid] for rid in selected]
            else:
                values=sorted(values,key=lambda r:r["published_at_timestamp"],reverse=True)[:5]
            out=root/"data/raw"/path.name
            atomic_write_json(out,values)
            files[out.relative_to(root).as_posix()]=hashlib.sha256(out.read_bytes()).hexdigest()
    atomic_write_json(root/"tasks.json",TASKS)
    retrieval_labels=[{"appid":a,"query":q,"relevant_ids":r,"contrast_ids":n}
                      for a,q,r,n in RETRIEVAL_CASES]
    atomic_write_json(root/"retrieval_labels.json",retrieval_labels)
    for name in ("tasks.json","retrieval_labels.json"):
        files[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
    manifest={"id":"real-cache-curated-v1","files":files,"upstream_hashes":upstream,
        "selected_review_count":sum(map(len,IDS.values())),"games":list(IDS),
        "dataset_sha256":digest(files),
        "task_set_sha256":files["tasks.json"],
        "retrieval_label_set_sha256":files["retrieval_labels.json"],
        "evaluation_implementation_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "independent_human_annotation":False,
        "note":"现有缓存中选取的真实原文；助理阅读后标注，选样偏向问题，不是完整Steam数据、总体估计或独立盲测。"}
    # The manifest is the commit marker: interrupted preparation is not ready.
    atomic_write_json(root/"manifest.json",manifest)
    return manifest


def validate_snapshot(root):
    manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        if hashlib.sha256((root/name.replace("\\", "/")).read_bytes()).hexdigest()!=expected:
            raise ValueError("冻结快照发生变化，不能继续使用旧标准")


def dataset(root, appid):
    manifest=prepare(root)
    raw=root/"data/raw"
    files={name.replace("\\", "/"):value for name,value in manifest["files"].items()}
    return {"id":manifest["id"],"appid":appid,"sha256":manifest["dataset_sha256"],
            "source_hashes": {label:files[f"data/raw/{appid}_{suffix}.json"] for label,suffix in (("reviews","reviews"),("patches","patch_candidates"),("news","news"))},
            "reviews":json.loads((raw/f"{appid}_reviews.json").read_text(encoding="utf-8")),
            "patches":json.loads((raw/f"{appid}_patch_candidates.json").read_text(encoding="utf-8"))}


def retrieval(root):
    from gamelens.rag.embedding import LocalEmbedding
    from gamelens.rag.index import GameIndex
    from gamelens.rag.tools import SearchArgs
    from gamelens.evaluation.retrieval import evaluate_ranking, summarize_mode
    manifest = prepare(root)
    runtime_implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    retrieval_cases=json.loads((root/"retrieval_labels.json").read_text(encoding="utf-8"))
    encoder=LocalEmbedding(PROJECT_ROOT)
    encoder.load(download=False)
    index=GameIndex(root,encoder)
    start=time.monotonic(); builds={}
    for appid in IDS:
        state={"appid":appid,"artifacts":{},"window_days":7}
        first=index.build(appid,state); second=index.build(appid,state)
        builds[appid]={"embedded_chunks":first["embedded_chunks"],"reuse_embedded_chunks":second["embedded_chunks"]}
    modes={}
    for mode in ("bm25","dense","hybrid"):
        results=[]
        for case_index,item in enumerate(retrieval_cases):
            appid=item["appid"];query=item["query"]
            relevant=item["relevant_ids"];contrast=item["contrast_ids"]
            state={"appid":appid,"artifacts":{},"window_days":7}
            args=SearchArgs(appid=appid,query=query,kind="reviews",top_k=10)
            index.search(args,state,mode=mode,use_topic_boost=False)  # One warmup, excluded from latency.
            rankings=[]; measurements=[]
            for _ in range(5):
                measured=time.perf_counter()
                hits,_=index.search(args,state,mode=mode,use_topic_boost=False)
                measurements.append((time.perf_counter()-measured)*1000)
                rankings.append([h["source_id"] for h in hits])
            if any(ids!=rankings[0] for ids in rankings[1:]):
                raise RuntimeError(f"真实快照{mode}/{case_index}重复运行排名不一致")
            evaluation_case={"id":f"real_{case_index}","labels":[
                {"source_id":source_id,"relevance":3,"role":"support"} for source_id in relevant]}
            results.append({"appid":appid,"query":query,"relevant_ids":relevant,
                "retrieved_ids":rankings[0],"contrast_hits":sorted(set(contrast)&set(rankings[0])),
                "latency_ms":sum(measurements)/len(measurements),"measurements_ms":measurements,
                **evaluate_ranking(evaluation_case,rankings[0])})
        modes[mode]={"exploratory_summary":summarize_mode(results),"results":results}
    payload={"manifest":manifest,"model":first["manifest"]["config"],"builds":builds,"modes":modes,
        "formal_metrics":None,"annotation_status":"pending_human_review","exploratory_only":True,
        "runtime_implementation_sha256":runtime_implementation_sha256,
        "implementation_matches_snapshot":runtime_implementation_sha256==manifest.get("evaluation_implementation_sha256"),
        "elapsed_seconds":round(time.monotonic()-start,2),"human_review_pending":True,
        "note":"真实缓存标签由助理选取且未独立人工复核；逐题结果仅供探索，不进入正式指标。"}
    atomic_write_json(root/"retrieval_result.json",payload)
    lines=["# 真实缓存小样本检索评测（探索性）","",manifest["note"],"",payload["note"],"",
           "|模式|Recall@5|Recall@10|MRR@10|nDCG@10|无答案准确率|","|---|---:|---:|---:|---:|---:|"]
    for mode,result in modes.items():
        summary=result["exploratory_summary"]
        show=lambda value:"N/A" if value is None else f"{value:.4f}"
        lines.append(f"|{mode}|{show(summary['mean_recall_at_5'])}|{show(summary['mean_recall_at_10'])}|"
                     f"{show(summary['mrr_at_10'])}|{show(summary['ndcg_at_10'])}|{show(summary['no_answer_accuracy'])}|")
    lines += ["","这些数值不是正式金标结果。无答案问题仍可能返回相似候选，候选不代表问题存在。",""]
    atomic_write_text(root/"retrieval_report.md","\n".join(lines))
    return payload


def run_live(root, output, limit, budget, case_ids=None):
    from gamelens.core.llm import OpenAICompatibleClient
    from gamelens.core.runtime import InvestigationAgent, LLMPlanner
    from gamelens.evaluation.benchmark import initial_state
    from gamelens.evaluation.suite import business_registry
    from gamelens.storage.telemetry import summarize
    from gamelens.core.budget import TaskBudgetStore
    if output.exists():raise ValueError("输出目录已存在，不覆盖历史记录")
    output.mkdir(parents=True)
    events={}
    budget_thread=f"evaluation-{output.name}"[-128:]
    budget_store=TaskBudgetStore(output/"request_budget.sqlite",request_limit=budget)
    client=OpenAICompatibleClient(os.getenv("LLM_MODEL",""),timeout=90,max_retries=0,
        telemetry_sink=lambda e:events.__setitem__(e["id"],e),request_budget=budget_store,
        budget_thread_id=budget_thread)
    scores=[]
    selected=[t for t in TASKS if not case_ids or t["case"]["id"] in case_ids][:limit]
    for task in selected:
        validate_snapshot(root)
        spec=BusinessCase.model_validate(task["case"]); data=dataset(root,task["appid"])
        state=initial_state(spec.id,spec.goal)
        state["thread_id"]=f"real-{output.name}-{spec.id}"[-128:]
        state.update(appid=task["appid"],window_days=7,patch_news_id=None,rag_mode=task["rag"],max_steps=12,optimization_mode="auto",artifacts={},
                     evaluation={"case_id":spec.id,"case_sha256":digest(spec.model_dump(mode="json")),"dataset_id":data["id"],"dataset_sha256":data["sha256"],"model":client.model})
        before=set(events); start=time.monotonic(); result=state
        print("真实缓存调查："+spec.id,flush=True)
        try:
            agent=InvestigationAgent(LLMPlanner(client),business_registry(root,client,task["rag"]!="off"),
                task_budget=budget_store,budget_thread_id=budget_thread)
            for update in agent.stream(state):result=dict(update)
        except Exception as error:
            result={**result,"status":"failed","error":f"{type(error).__name__}: {error}"}
        # Evaluation metadata is runner-owned; LangGraph drops undeclared keys.
        result["evaluation"]=state["evaluation"]
        result["telemetry"]=summarize([e for k,e in events.items() if k not in before])
        score=evaluate_business_run(spec,result,data)
        score.update(elapsed_seconds=round(time.monotonic()-start,3),telemetry=result["telemetry"])
        atomic_write_json(output/"runs"/f"{spec.id}.json",result)
        atomic_write_json(output/"reviews"/f"{spec.id}.json",review_template(spec,result,data["sha256"]))
        report=result.get("artifacts",{}).get("final_report")
        if report and Path(report).is_file():shutil.copyfile(report,output/f"{spec.id}.md")
        scores.append(score)
        totals=summarize(list(events.values()))
        atomic_write_json(output/"summary.json",{"model":client.model,"max_requests":budget,
            "calls":totals["model_calls"],"http_requests":totals["http_requests"],
            "budget":budget_store.snapshot(budget_thread),"manifest":prepare(root),
            "results":scores,"semantic_acceptance":"pending_human_review"})
        print(f"  {result.get('status')}；自动检查={score['automatic_passed']}；{score['elapsed_seconds']}秒",flush=True)
        if budget_store.snapshot(budget_thread)["llm_requests_remaining"]<=0:break
    return {"tasks":len(scores),"calls":client.calls,"automatic_passed":sum(r["automatic_passed"] for r in scores),"human_review_pending":True}


def replay(root, run_dir, output):
    """Correct runner-owned binding in a separate replay; never rewrite old runs."""
    if output.exists():raise ValueError("回放目录已存在，不覆盖历史")
    validate_snapshot(root)
    model=json.loads((run_dir/"summary.json").read_text(encoding="utf-8"))["model"]
    lookup={t["case"]["id"]:t for t in TASKS};results=[]
    for path in sorted((run_dir/"runs").glob("*.json")):
        task=lookup[path.stem];spec=BusinessCase.model_validate(task["case"]);data=dataset(root,task["appid"])
        state=json.loads(path.read_text(encoding="utf-8"))
        if state.get("goal")!=spec.goal or state.get("appid")!=data["appid"]:raise ValueError("运行身份不匹配，不能修正绑定")
        sources=[o["data"]["source"] for o in state.get("observations",[]) if o.get("ok") and o.get("data",{}).get("source")]
        if not sources or any(s["sha256"]!=data["source_hashes"]["reviews"] for s in sources):raise ValueError("来源不匹配，不能修正绑定")
        restored=not bool(state.get("evaluation"))
        state["evaluation"]={"case_id":spec.id,"case_sha256":digest(spec.model_dump(mode="json")),"dataset_id":data["id"],"dataset_sha256":data["sha256"],"model":model}
        row=evaluate_business_run(spec,state,data)
        row["runner_binding_restored"]=restored
        guards=[]
        from gamelens.rag.tools import SearchArgs
        from gamelens.core.investigation_checks import validate_version_assignments
        from gamelens.tools.investigation import required_patch_count
        for observation in state.get("observations",[]):
            if observation.get("tool")=="search_game_evidence" and observation.get("ok"):
                try:SearchArgs.model_validate(observation["arguments"])
                except ValueError:guards.append("旧运行指定补丁检索却未明确时间窗口；当前参数门禁会拒绝")
        if required_patch_count(spec.goal)>1 and state.get("final_citations"):
            guards.extend(validate_version_assignments(state["observations"],state["final_citations"],state.get("final_version_evidence",[])))
        row["current_runtime_guard_errors"]=list(dict.fromkeys(guards))
        row["passes_current_runtime_gates"]=not guards
        atomic_write_json(output/"runs"/path.name,state)
        atomic_write_json(output/"reviews"/path.name,review_template(spec,state,data["sha256"]))
        results.append(row)
    payload={"model":model,"results":results,"model_calls":0,"note":"修正LangGraph丢弃运行器身份标签的导出错误；原回答、事实、引用及历史运行不改动。当前门禁单独回放，不把旧答案当作修复后收费验证。","human_review_pending":True}
    atomic_write_json(output/"summary.json",payload)
    atomic_write_text(output/"report.md","# 真实缓存调查回放\n\n"+payload["note"]+"\n\n|任务|原客观检查|当前新增门禁|独立语义验收|\n|---|---|---|---|\n"+"\n".join(f"|{r['case_id']}|{r['automatic_passed']}|{r['passes_current_runtime_gates']}|待人工审核|" for r in results)+"\n\n"+"\n\n".join(r["case_id"]+"："+"；".join(r["current_runtime_guard_errors"]) for r in results if r["current_runtime_guard_errors"]))
    return {"runs":len(results),"objective_passed":sum(r["automatic_passed"] for r in results),"current_gates_passed":sum(r["passes_current_runtime_gates"] for r in results),"model_calls":0}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=("prepare","oracle","retrieval","agent","replay"),default="prepare")
    parser.add_argument("--snapshot-dir",type=Path,default=PROJECT_ROOT/"evals/results/real_snapshot_v1")
    parser.add_argument("--output-dir",type=Path,default=PROJECT_ROOT/"evals/results/quality/20261008_real_complex6")
    parser.add_argument("--limit",type=int,default=6)
    parser.add_argument("--max-model-calls",type=int,default=48)
    parser.add_argument("--allow-paid-model",action="store_true")
    parser.add_argument("--case-ids",nargs="+")
    parser.add_argument("--run-dir",type=Path)
    args=parser.parse_args()
    if args.case_ids and (len(set(args.case_ids))!=len(args.case_ids) or set(args.case_ids)-{t["case"]["id"] for t in TASKS}):parser.error("未知或重复case-id")
    if not 1<=args.limit<=12 or not 1<=args.max_model_calls<=48:parser.error("limit 1..12，模型请求上限1..48")
    if args.mode=="agent" and not args.allow_paid_model:parser.error("真实模型调用需要显式--allow-paid-model")
    if args.mode=="replay" and not args.run_dir:parser.error("replay需要--run-dir")
    result=prepare(args.snapshot_dir)
    if args.mode=="oracle":
        from gamelens.evaluation.validation_run_tests import oracle_snapshot
        rows=oracle_snapshot(args.snapshot_dir)
        result={"cases":len(rows),"passed":sum(r["passed"] for r in rows),"results":rows,"model_calls":0,"tests_semantics":False}
        atomic_write_json(args.snapshot_dir/"oracle_result.json",result)
    elif args.mode=="retrieval":result=retrieval(args.snapshot_dir)
    elif args.mode=="agent":result=run_live(args.snapshot_dir,args.output_dir,args.limit,args.max_model_calls,args.case_ids)
    elif args.mode=="replay":result=replay(args.snapshot_dir,args.run_dir,args.output_dir)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
