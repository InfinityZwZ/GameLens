"""Scorer tests use authored outputs; they are not live Agent effectiveness results."""
import copy
import json
from pathlib import Path

import pytest

from gamelens.core.grounding import resolve_path
from gamelens.evaluation.business_cases import CASES, materialize_dataset
from gamelens.evaluation.quality import (
    BusinessCase, HumanReview, RUBRIC_KEYS, aggregate_business_results,
    evaluate_business_run, review_template,
)
from gamelens.evaluation.suite import (
    evaluation_tags, load_dataset, parse_args, percentile, render_report,
    runtime_summary, task_state,
)
from gamelens.tools.investigation import (
    ReviewInvestigation, QueryReviewsArgs, GetReviewArgs, CompareCohortsArgs,
    RecentVersionsArgs, ListPatchesArgs,
    VerifyInvestigationArgs,
)

MODELS = {"query_reviews": QueryReviewsArgs, "get_review": GetReviewArgs,
          "compare_cohorts": CompareCohortsArgs, "investigate_recent_versions": RecentVersionsArgs,
          "list_patches": ListPatchesArgs}


def authored_run(root, spec):
    materialize_dataset(root)
    dataset = load_dataset(root)
    case = BusinessCase.model_validate(copy.deepcopy(spec))
    state = task_state(case, case.id, dataset)
    domain = ReviewInvestigation(root)
    observations = []
    for step, planned in enumerate(case.oracle_plan, 1):
        state["step_count"] = step - 1
        result = getattr(domain, planned["tool"])(MODELS[planned["tool"]].model_validate(planned["arguments"]), state)
        observations.append({"step": step, "tool": planned["tool"], "arguments": copy.deepcopy(planned["arguments"]), **result.model_dump(mode="json")})
        state["artifacts"].update(result.artifacts)
    claims = []
    from gamelens.evaluation.quality import _fact_matches
    for requirement in case.facts:
        match = next(o for o in observations if _fact_matches(o, requirement))
        claims.append({"source_step": match["step"], "label": str(requirement.path),
                       "fact_path": requirement.path, "value": requirement.value})
    citations, news = [], []
    for group in case.evidence_groups:
        for source_id in group.ids:
            for observation in observations:
                data = observation["data"]
                rows = data.get("announcements", []) if group.kind == "news" else data.get("reviews", []) + ([data["review"]] if "review" in data else [])
                key = "news_id" if group.kind == "news" else "review_id"
                record = next((r for r in rows if r[key] == source_id), None)
                if record:
                    target = news if group.kind == "news" else citations
                    target.append({"source_step": observation["step"], key: source_id,
                                   "quote": record["contents" if group.kind == "news" else "text"]})
                    break
    result = domain.verify_investigation(VerifyInvestigationArgs(appid=730), state)
    state["artifacts"].update(result.artifacts)
    observations.append({"step": len(observations)+1, "tool": "verify_investigation", **result.model_dump(mode="json")})
    state.update(observations=observations, final_claims=claims, final_citations=citations,
                 final_news_citations=news, status="completed",
                 step_count=len(observations),
                 final_answer="评分器单元测试的手工候选，不是真实模型回答。")
    return case, state, dataset


def human(case, state, dataset):
    review = review_template(case, state, dataset["sha256"])
    review.update(completed=True, reviewer="unit-test-only", scores={k: 2 for k in RUBRIC_KEYS},
                  notes="这是验证评分绑定的测试样本，不是效果评测。")
    return review


@pytest.mark.parametrize("spec", CASES, ids=lambda s: s["id"])
def test_all_hand_authored_gold_matches_real_tools(tmp_path, spec):
    case, state, dataset = authored_run(tmp_path, spec)
    result = evaluate_business_run(case, state, dataset)
    assert result["automatic_passed"], result
    assert result["accepted"] is False
    assert result["status"] == "awaiting_human_review"


def test_review_binds_run_case_and_dataset(tmp_path):
    case, state, dataset = authored_run(tmp_path, CASES[0])
    review = human(case, state, dataset)
    assert evaluate_business_run(case, state, dataset, review)["accepted"]
    changed = copy.deepcopy(state)
    changed["final_answer"] = "修改过的结论"
    assert evaluate_business_run(case, changed, dataset, review)["human_review"]["status"] == "invalid"
    review["dataset_sha256"] = "different"
    assert evaluate_business_run(case, state, dataset, review)["accepted"] is False


def test_missing_claims_not_just_correct_tool_is_failure(tmp_path):
    case, state, dataset = authored_run(tmp_path, CASES[0])
    state["final_claims"] = []
    result = evaluate_business_run(case, state, dataset)
    assert result["checks"]["gold_facts"] is False
    assert result["automatic_passed"] is False


def test_wrong_numbers_and_quote_fabrication_fail(tmp_path):
    case, state, dataset = authored_run(tmp_path, CASES[6])
    state["final_claims"][0]["value"] = 999
    state["final_citations"][0]["quote"] = "不存在的结论"
    result = evaluate_business_run(case, state, dataset)
    assert not result["checks"]["gold_facts"]
    assert not result["checks"]["citation_integrity"]


def test_forged_observation_cannot_make_forged_quote_true(tmp_path):
    case, state, dataset = authored_run(tmp_path, CASES[9])
    state["observations"][0]["data"]["review"]["text"] = "作者承认游戏已完全修复"
    state["final_citations"][0]["quote"] = "作者承认游戏已完全修复"
    assert not evaluate_business_run(case, state, dataset)["checks"]["citation_integrity"]


def test_repeated_same_quote_not_cross_version_coverage(tmp_path):
    case, state, dataset = authored_run(tmp_path, CASES[4])
    one = next(c for c in state["final_citations"] if c["review_id"] == "r1")
    state["final_citations"] = [one] * 5
    result = evaluate_business_run(case, state, dataset)
    assert not result["checks"]["evidence_coverage"]
    assert result["evidence_groups"][0]["matched_ids"] == ["r1"]


def test_missing_counterexample_fails_coverage(tmp_path):
    case, state, dataset = authored_run(tmp_path, CASES[4])
    state["final_citations"] = [c for c in state["final_citations"] if c["review_id"] != "r11"]
    assert not evaluate_business_run(case, state, dataset)["checks"]["evidence_coverage"]


def test_human_rejects_semantically_unsupported_answer(tmp_path):
    case, state, dataset = authored_run(tmp_path, CASES[4])
    state["final_answer"] = "五次更新中这个缺陷始终存在，已证明所有玩家都受影响。"
    review = human(case, state, dataset)
    review["scores"]["evidence_support"] = 0
    review["critical_errors"] = ["证据不支持五个窗口连续存在或所有玩家受到影响"]
    result = evaluate_business_run(case, state, dataset, review)
    assert result["automatic_passed"]  # Honest limitation of deterministic evidence checks.
    assert not result["accepted"]
    assert result["status"] == "human_failed"


def test_low_score_and_bool_score_rejected(tmp_path):
    case, state, dataset = authored_run(tmp_path, CASES[0])
    review = human(case, state, dataset)
    review["scores"] = {k: 1 for k in RUBRIC_KEYS}
    assert not evaluate_business_run(case, state, dataset, review)["accepted"]
    review["scores"]["readability"] = True
    with pytest.raises(ValueError):
        HumanReview.model_validate(review)


def test_offline_baseline_never_counts_as_agent_success(tmp_path):
    case, state, dataset = authored_run(tmp_path, CASES[0])
    result = evaluate_business_run(case, state, dataset, human(case, state, dataset), execution_mode="tool_oracle")
    assert not result["accepted"]
    assert aggregate_business_results([result], 12)["evaluated_agent_runs"] == 0


def test_no_rag_and_dataset_hash_enforced(tmp_path):
    case, state, dataset = authored_run(tmp_path, CASES[8])
    state["observations"].append({"step": 3, "tool": "search_game_evidence", "ok": False, "data": {}})
    state["observations"][0]["data"]["source"]["sha256"] = "changed"
    result = evaluate_business_run(case, state, dataset)
    assert not result["checks"]["rag_policy"]
    assert not result["checks"]["dataset_identity"]


def test_equivalent_timezones_match_but_wrong_window_fails(tmp_path):
    case, state, dataset = authored_run(tmp_path, CASES[2])
    state["observations"][0]["arguments"]["filters"]["start_at"] = "2026-09-30T00:00:00+00:00"
    assert evaluate_business_run(case, state, dataset)["checks"]["gold_facts"]
    state["observations"][0]["arguments"]["filters"]["start_at"] = "2026-09-29T00:00:00Z"
    assert not evaluate_business_run(case, state, dataset)["checks"]["gold_facts"]


def test_partial_review_never_reports_final_accuracy(tmp_path):
    case, state, dataset = authored_run(tmp_path, CASES[0])
    approved = evaluate_business_run(case, state, dataset, human(case, state, dataset))
    pending = evaluate_business_run(case, state, dataset)
    aggregate = aggregate_business_results([approved, pending], 12)
    assert aggregate["fully_reviewed_acceptance_rate"] is None
    assert aggregate["accepted_fraction_including_pending"] == .5


def test_paid_mode_needs_explicit_optin():
    with pytest.raises(SystemExit):
        parse_args(["--mode", "agent"])
    assert parse_args([]).mode == "offline"
    assert parse_args(["--mode", "agent", "--allow-paid-model"]).limit == 3


def test_empty_runtime_metrics_not_zero_and_nearest_rank():
    assert runtime_summary([])["estimated_cost"] is None
    assert runtime_summary([])["latency_p95_seconds"] is None
    assert percentile([1, 2, 3], .95) == 3


def test_report_is_readable_and_does_not_claim_unmeasured_accuracy():
    payload = {"created_at": "test", "component_baselines": [], "business_oracle": [],
               "business_results": [], "cases": CASES,
               "summary": aggregate_business_results([], 12), "runtime": runtime_summary([])}
    text = render_report(payload)
    assert "尚不可报告" in text
    assert "不是模型回答准确率" in text or "不能作为Agent准确率" in text
    assert "```json" not in text


def test_replay_roundtrip_without_api(tmp_path):
    from gamelens.evaluation.suite import main
    case, state, dataset = authored_run(tmp_path / "data_source", CASES[0])
    run = tmp_path / "saved.json"
    run.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"dataset_sha256": dataset["sha256"], "runs": [
        {"case_id": case.id, "run_name": "example_1", "path": "saved.json", "execution_mode": "agent"}]}), encoding="utf-8")
    output = tmp_path / "out"
    assert main(["--mode", "replay", "--runs-manifest", str(manifest), "--output-dir", str(output)]) == 2
    assert (output / "report.md").is_file()
    template = json.loads((output / "reviews/example_1.json").read_text(encoding="utf-8"))
    assert template["completed"] is False


def test_real_graph_with_scripted_planner_is_not_live_effectiveness(tmp_path):
    from gamelens.core.runtime import InvestigationAgent, PlannerDecision
    from gamelens.evaluation.suite import business_registry
    from test_agent_runtime import SequencePlanner
    materialize_dataset(tmp_path)
    dataset = load_dataset(tmp_path)
    case = BusinessCase.model_validate(CASES[0])
    planner = SequencePlanner([
        PlannerDecision(action="query_reviews", reason="核对本地统计", arguments={"appid": 730, "filters": {"period": "all"}}),
        PlannerDecision(action="verify_investigation", reason="核验资料", arguments={"appid": 730}),
        PlannerDecision(action="finish", reason="完成离线契约验证", final_answer="本地15条评论中有7条差评，好评率约53.33%；仅代表本地样本。",
                        claims=[{"label": str(f.path), "source_step": 1, "fact_path": f.path, "value": f.value} for f in case.facts]),
    ])
    state = InvestigationAgent(planner, business_registry(tmp_path, None)).invoke(task_state(case, 1, dataset))
    state["evaluation"] = evaluation_tags(case, dataset)
    result = evaluate_business_run(case, state, dataset, execution_mode="scripted_test")
    assert result["automatic_passed"], result
    assert not result["accepted"]
    assert state["observations"][0]["arguments"]["filters"]["period"] == "all"


def test_tool_allowlist_cannot_fetch_steam_or_classify_paid(tmp_path):
    from gamelens.evaluation.suite import business_registry
    registry = business_registry(tmp_path, None)
    assert "verify_investigation" in registry.names
    assert not registry.names & {"prepare_reviews", "prepare_patches", "classify_review_sample"}
    with pytest.raises(ValueError):
        registry.subset({"unknown"})


@pytest.mark.parametrize("budget,expected_code", [(3, 2), (2, 1)])
def test_agent_cli_export_and_budget_with_fake_model_only(tmp_path, monkeypatch, budget, expected_code):
    """Smoke test the real runner, never a measurement of real LLM performance."""
    import gamelens.core.llm as llm
    from gamelens.evaluation.suite import main
    case = BusinessCase.model_validate(CASES[0])

    class FakeClient:
        def __init__(self, model, **kwargs):
            self.model = "fake-model-unit-test"
            self.decisions = iter([
                {"action": "query_reviews", "reason": "读取本地数据", "arguments": {"appid": 730, "filters": {"period": "all"}}},
                {"action": "verify_investigation", "reason": "核验来源", "arguments": {"appid": 730}},
                {"action": "finish", "reason": "测试完成", "final_answer": "本地样本15条、差评7条，好评率约53.33%。",
                 "claims": [{"label": str(f.path), "source_step": 1, "fact_path": f.path, "value": f.value} for f in case.facts]},
            ])

        def generate_json(self, messages):
            return next(self.decisions)

    monkeypatch.setattr(llm, "OpenAICompatibleClient", FakeClient)
    monkeypatch.setattr(llm.httpx, "Client", lambda *a, **k: pytest.fail("No HTTP allowed in this test"))
    output = tmp_path / "fake_model_test"
    assert main(["--mode", "agent", "--allow-paid-model", "--limit", "1",
                 "--max-model-calls", str(budget), "--output-dir", str(output)]) == expected_code
    payload = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    state = json.loads((output / "runs/overall_scope_1.json").read_text(encoding="utf-8"))
    assert payload["model_configuration"]["model"] == "fake-model-unit-test"
    assert state["observations"][0]["tool"] == "query_reviews"
    assert (output / "runs_manifest.json").is_file()
    if budget == 3:
        assert payload["business_results"][0]["automatic_passed"]
        assert Path(state["artifacts"]["final_report"]).is_file()
        assert payload["summary"]["fully_reviewed_acceptance_rate"] is None
    else:
        assert state["status"] == "failed"
        assert "预算" in state["error"]


def cohort_equivalent_run(root):
    case, state, dataset = authored_run(root, CASES[2])
    args = {"appid": 730,
            "cohort_a": {"period": "all", "start_at": "2026-09-30T00:00:00Z", "end_at": "2026-10-01T00:00:00Z"},
            "cohort_b": {"period": "all", "start_at": "2026-10-01T00:00:00Z", "end_at": "2026-10-06T00:00:00Z"}}
    result = ReviewInvestigation(root).compare_cohorts(CompareCohortsArgs.model_validate(args), state)
    state["observations"] = [{"step": 1, "tool": "compare_cohorts", "arguments": args, **result.model_dump(mode="json")},
                             {"step": 2, "tool": "verify_investigation", "ok": True, "data": {}}]
    state["final_claims"] = [{"source_step": 1, "label": group+key, "fact_path": [group, "stats", key], "value": value}
                            for group, key, value in (("cohort_a", "count", 4), ("cohort_a", "positive_rate", .5),
                                                      ("cohort_b", "count", 11), ("cohort_b", "positive_rate", 6/11))]
    return case, state, dataset


def test_equivalent_cohort_strategy_is_not_penalized(tmp_path):
    case, state, dataset = cohort_equivalent_run(tmp_path)
    result = evaluate_business_run(case, state, dataset)
    assert result["checks"]["gold_facts"]
    assert result["facts"][-1]["supporting_sources"][0]["path"] == ["cohort_b", "stats", "positive_rate"]


def test_same_numbers_wrong_window_still_fail(tmp_path):
    case, state, dataset = cohort_equivalent_run(tmp_path)
    item = state["observations"][0]
    item["arguments"]["cohort_b"]["end_at"] = "2026-10-07T00:00:00Z"
    item["data"]["cohort_b"]["filters"]["end_at"] = "2026-10-07T00:00:00Z"
    item["data"]["cohort_b"]["bounds"]["end_exclusive"] = 1791331200
    result = evaluate_business_run(case, state, dataset)
    assert not result["facts"][-1]["tool_correct"]


def test_implicit_patch_and_explicit_time_window_are_equivalent(tmp_path):
    case, state, dataset = authored_run(tmp_path, CASES[1])
    item = state["observations"][0]
    item["arguments"].pop("patch_news_id")  # p5 remains the selected patch in state.
    assert evaluate_business_run(case, state, dataset)["checks"]["gold_facts"]
    item["arguments"]["filters"] = {"period": "all", "start_at": "2026-10-05T00:00:00Z", "end_at": "2026-10-06T00:00:00Z"}
    item["data"]["filters"].update(item["arguments"]["filters"])
    item["data"]["patch_news_id"] = None
    assert evaluate_business_run(case, state, dataset)["checks"]["gold_facts"]
