import json
import shutil
from pathlib import Path
from gamelens.core.runtime import InvestigationAgent, PlannerDecision, ToolRegistry, AgentTool, LLMPlanner
from gamelens.core.grounding import fact_catalog, resolve_fact_ids
from gamelens.tools.investigation import ReviewInvestigation, RecentVersionsArgs, VerifyInvestigationArgs, required_patch_count
from gamelens.evaluation.benchmark import initial_state
from test_agent_runtime import SequencePlanner


def ready_state():
    state=initial_state(201,"检查已获取的证据")
    state["artifacts"]={"final_report":"example.md","evidence_assessed":"true"}
    state["observations"]=[{"step":2,"tool":"prepare_patches","ok":True,"data":{"patch_count":7}}]
    return state


def test_730_paths_are_corrected_by_fact_identifier():
    state=ready_state()
    id=fact_catalog(state["observations"])[0]["fact_id"]
    planner=SequencePlanner([PlannerDecision(action="finish",reason="结束",claims=[{"label":"补丁数","source_step":2,"fact_path":[730,730],"value":7}]),
        PlannerDecision(action="finish",reason="改用提供的事实编号",fact_ids=[id],final_answer="本地有7个候选")])
    result=InvestigationAgent(planner,ToolRegistry([])).invoke(state)
    assert result["status"]=="completed"
    assert result["validation_retries"]==1
    assert result["final_claims"][0]["fact_path"]==["patch_count"]
    assert result["final_claims"][0]["value"]==7


def test_unknown_fact_ids_stop_after_two_repairs():
    planner=SequencePlanner([PlannerDecision(action="finish",reason="错误引用",fact_ids=["S999F1"]) for _ in range(3)])
    result=InvestigationAgent(planner,ToolRegistry([])).invoke(ready_state())
    assert result["status"]=="failed"
    assert result["validation_retries"]==2
    assert "不存在的事实编号" in result["error"]


def test_fact_catalog_preserves_lists_and_redacts_secrets():
    data={"stats":{"count":13},"versions":[{"count":2}],"api_key":"private","source":{"path":"private/path","scope":"cached"}}
    catalog=fact_catalog([{"step":1,"ok":True,"data":data}])
    assert "private" not in json.dumps(catalog)
    fact=next(f for f in catalog if f["fact_path"]==["versions",0,"count"])
    claims,errors=resolve_fact_ids([{"step":1,"ok":True,"data":data}],[fact["fact_id"]])
    assert not errors and claims[0]["value"]==2


def test_multiversion_request_requires_actual_coverage(tmp_path):
    raw=tmp_path/"data/raw";raw.mkdir(parents=True)
    fixture=Path(__file__).resolve().parents[1]/"evals/fixtures"
    for file in fixture.glob("*.json"):shutil.copyfile(file,raw/file.name)
    domain=ReviewInvestigation(tmp_path)
    registry=ToolRegistry([AgentTool("investigate_recent_versions","覆盖",RecentVersionsArgs,domain.investigate_recent_versions),
                          AgentTool("verify_investigation","核验",VerifyInvestigationArgs,domain.verify_investigation)])
    state=initial_state(202,"分析近五次更新以来的性能问题")
    planner=SequencePlanner([PlannerDecision(action="finish",reason="试图结束"),
        PlannerDecision(action="verify_investigation",reason="核验有限范围",arguments={"appid":730}),
        PlannerDecision(action="finish",reason="覆盖不足",scope_limited=True,fact_ids=["S1F1"],final_answer="只能作有限分析")])
    result=InvestigationAgent(planner,registry).invoke(state)
    assert result["status"]=="completed"
    assert result["observations"][0]["tool"]=="investigate_recent_versions"
    assert result["observations"][0]["data"]["available_count"]==2
    assert "不能确认问题贯穿全部版本" in result["final_answer"]


def test_recent_patch_count_detection():
    assert required_patch_count("分析近五次更新版本以来的问题")==5
    assert required_patch_count("最近5个版本的问题")==5
    assert required_patch_count("last 3 patches")==3
    assert required_patch_count("更新后5天的反馈")==1


class PayloadClient:
    """Raw mock responses exercise the real planner validation boundary."""
    def __init__(self,payloads):self.payloads=payloads;self.messages=[]
    def generate_json(self,messages):self.messages.append(messages);return self.payloads.pop(0)


def test_eleven_citations_and_31_facts_do_not_fail_capacity():
    state=ready_state()
    state["observations"]=[{"step":2,"tool":"query_reviews","ok":True,"data":{
        "metrics":{f"metric_{i}":i for i in range(31)},
        "reviews":[{"review_id":f"r{i}","text":f"原文证据{i}"} for i in range(11)]}}]
    ids=[fact["fact_id"] for fact in fact_catalog(state["observations"])]
    payload={"action":"finish","reason":"充分引用","final_answer":"31条可核验事实和11条原文引用",
        "fact_ids":ids,"citations":[{"source_step":2,"review_id":f"r{i}","quote":f"原文证据{i}"} for i in range(11)]}
    client=PayloadClient([payload])
    result=InvestigationAgent(LLMPlanner(client),ToolRegistry([])).invoke(state)
    assert result["status"]=="completed"
    assert len(result["final_claims"])==31 and len(result["final_citations"])==11
    context=json.loads(client.messages[0][1]["content"].split("\n",1)[1])
    assert context["planner_output_schema"]["properties"]["fact_ids"]["maxItems"]==64


def test_structural_overflow_is_repaired_without_rerunning_tools():
    bad={"action":"finish","reason":"过量输出","fact_ids":["S2F1"]*65}
    good={"action":"finish","reason":"缩减输出","fact_ids":["S2F1"]}
    client=PayloadClient([bad,good]);state=ready_state()
    result=InvestigationAgent(LLMPlanner(client),ToolRegistry([])).invoke(state)
    assert result["status"]=="completed" and result["validation_retries"]==1
    assert result["observations"]==state["observations"]
    context=json.loads(client.messages[1][1]["content"].split("\n",1)[1])
    assert "fact_ids" in " ".join(context["validation_feedback"])


def test_structural_output_errors_have_a_finite_shared_retry_budget():
    bad={"action":"finish","reason":"无效输出","fact_ids":["S2F1"]*65}
    result=InvestigationAgent(LLMPlanner(PayloadClient([bad,bad,bad])),ToolRegistry([])).invoke(ready_state())
    assert result["status"]=="failed" and result["validation_retries"]==2
    assert "fact_ids" in result["error"]


def test_capacity_increase_does_not_accept_fabricated_quotes():
    state=ready_state()
    bad={"action":"finish","reason":"引用","fact_ids":["S2F1"],
         "citations":[{"source_step":2,"review_id":"missing","quote":"伪造原文"}]}
    result=InvestigationAgent(LLMPlanner(PayloadClient([bad,bad,bad])),ToolRegistry([])).invoke(state)
    assert result["status"]=="failed"
    assert "原文引用不匹配" in result["error"]
