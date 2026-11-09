import json
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError
from gamelens.tools.investigation import ReviewInvestigation,QueryReviewsArgs,VerifyInvestigationArgs,ReviewFilter
from gamelens.evaluation.benchmark import initial_state,reference_run
from gamelens.evaluation.effects import score_effect_case
from gamelens.core.grounding import validate_grounding

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=json.loads((ROOT/"evals"/"investigation_cases.json").read_text(encoding="utf-8"))


@pytest.fixture
def dataset(tmp_path):
    raw=tmp_path/"data"/"raw"
    raw.mkdir(parents=True)
    for path in (ROOT/"evals"/"fixtures").glob("*.json"): shutil.copyfile(path,raw/path.name)
    return ReviewInvestigation(tmp_path),json.loads((raw/"730_reviews.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case",MANIFEST["cases"],ids=lambda case:case["name"])
def test_fixed_ground_truth_cases_without_model(dataset,case):
    domain,corpus=dataset
    from gamelens.tools.investigation import QueryReviewsArgs,GetReviewArgs,CompareCohortsArgs,CompareVersionsArgs
    models={"query_reviews":QueryReviewsArgs,"get_review":GetReviewArgs,"compare_cohorts":CompareCohortsArgs,"compare_versions":CompareVersionsArgs}
    state=initial_state(1,case["goal"])
    try:
        result=getattr(domain,case["tool"])(models[case["tool"]].model_validate(case["arguments"]),state)
    except (ValueError,ValidationError):
        assert case.get("expect_error")
        return
    assert not case.get("expect_error")
    from gamelens.evaluation.effects import expected_matches
    assert expected_matches(case["expected"],result.data)


def test_fabricated_numeric_claim_and_quote_are_rejected(dataset):
    domain,corpus=dataset
    state=initial_state(2,"核验数字与引用")
    result=domain.query_reviews(QueryReviewsArgs(appid=730,filters=ReviewFilter(period="all")),state)
    observation={"step":1,"tool":"query_reviews",**result.model_dump(mode="json")}
    bad_claim={"label":"评论数","source_step":1,"fact_path":["stats","count"],"value":999}
    bad_quote={"source_step":1,"review_id":"r0","quote":"开发商承认更新导致崩溃"}
    assert len(validate_grounding([observation],[bad_claim],[bad_quote]))==2
    state.update(observations=[observation],status="completed",final_answer="完美回答",final_claims=[bad_claim],final_citations=[bad_quote])
    score=score_effect_case(MANIFEST["cases"][0],state,corpus)
    assert score["passed"] is False
    assert score["requested_facts_correct"] is False


def test_verification_rejects_changed_snapshot(dataset):
    domain,_=dataset
    state=initial_state(3,"检查数据变化")
    result=domain.query_reviews(QueryReviewsArgs(appid=730,filters=ReviewFilter(period="all")),state)
    state["artifacts"].update(result.artifacts)
    path=Path(result.data["source"]["path"])
    path.write_text(path.read_text(encoding="utf-8")+" ",encoding="utf-8")
    with pytest.raises(ValueError,match="快照已经变化"):
        domain.verify_investigation(VerifyInvestigationArgs(appid=730),state)


def test_all_tool_specs_remain_visible():
    from gamelens.memory.context import PlannerContextBuilder
    state=initial_state(4,"检查工具目录")
    context=PlannerContextBuilder().build(state,[{"name":f"tool-{i}"} for i in range(15)])
    assert len(context["available_tools"])==15


def test_registered_agent_query_quote_verify_and_finish(dataset):
    domain,_=dataset
    from gamelens.tools.registry import SteamInvestigationTools
    from gamelens.core.llm import OpenAICompatibleClient
    from gamelens.core.runtime import InvestigationAgent,PlannerDecision
    from test_agent_runtime import SequencePlanner
    client=OpenAICompatibleClient("test",api_key="not-a-real-key")
    registry=SteamInvestigationTools(domain.root,client).registry()
    planner=SequencePlanner([
        PlannerDecision(action="query_reviews",reason="按条件筛选",arguments={"appid":730,"filters":{"period":"all","is_positive":False},"limit":5}),
        PlannerDecision(action="get_review",reason="核对原文",arguments={"appid":730,"review_id":"r0"}),
        PlannerDecision(action="verify_investigation",reason="核验快照与报告",arguments={"appid":730}),
        PlannerDecision(action="finish",reason="证据已核验",final_answer="本地快照中有6条Steam差评，不能推断总体或因果。",
            claims=[{"label":"差评数","source_step":1,"fact_path":["stats","count"],"value":6}],
            citations=[{"source_step":2,"review_id":"r0","quote":"更新后崩溃 crash，启动时闪退。"}])])
    result=InvestigationAgent(planner,registry).invoke(initial_state(10,"调查差评并给原文证据"))
    assert result["status"]=="completed"
    assert Path(result["artifacts"]["final_report"]).is_file()
    assert "原文引用" not in result["final_answer"]
    assert "r0" in Path(result["artifacts"]["final_report"]).read_text(encoding="utf-8")
    assert result["observations"][0]["data"]["stats"]["count"]==6
