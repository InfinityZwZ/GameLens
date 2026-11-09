"""Task acceptance: objective evidence checks plus explicitly bound human review.

There is intentionally no keyword-based semantic accuracy or LLM self-confidence
score. Grounded quotations do not automatically establish a narrative conclusion.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from gamelens.core.grounding import (
    resolve_path, same_value, validate_grounding, validate_news_citations, validate_comment_citations,
)
from gamelens.evaluation.effects import expected_matches
from gamelens.core.decision_trace import decision_metrics

RUBRIC_KEYS = {"relevance", "evidence_support", "scope_handling", "completeness", "readability"}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


class FactRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: str
    path: list[str | int] = Field(min_length=1)
    value: Any
    arguments: dict = Field(default_factory=dict)


class EvidenceGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    kind: Literal["review", "news"]
    ids: list[str] = Field(min_length=1)
    min_distinct: int = Field(ge=1)

    @model_validator(mode="after")
    def check_count(self):
        if self.min_distinct > len(set(self.ids)):
            raise ValueError("证据要求超过不同来源的标注数量")
        return self


class BusinessCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z0-9_-]+$")
    category: str
    split: Literal["development", "holdout"]
    goal: str
    oracle_plan: list[dict]
    facts: list[FactRequirement]
    evidence_groups: list[EvidenceGroup]
    rubric: str
    history: list[dict] = Field(default_factory=list)
    rag_mode: Literal["off", "auto", "on"] = "auto"
    max_tool_steps: int = Field(default=12, ge=1, le=30)


class HumanReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    completed: Literal[True]
    case_id: str
    case_sha256: str
    dataset_sha256: str
    run_sha256: str
    reviewer: str = Field(min_length=1)
    scores: dict[str, StrictInt]
    critical_errors: list[str]
    notes: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_scores(self):
        if set(self.scores) != RUBRIC_KEYS or any(v not in (0, 1, 2) for v in self.scores.values()):
            raise ValueError("人工评分必须包含五项0/1/2分，不能缺项")
        if not self.reviewer.strip() or not self.notes.strip():
            raise ValueError("审核人及备注不能只包含空白")
        return self


def review_template(case: BusinessCase, state: dict, dataset_sha256: str) -> dict:
    return {
        "completed": False, "case_id": case.id,
        "case_sha256": digest(case.model_dump(mode="json")),
        "dataset_sha256": dataset_sha256, "run_sha256": digest(state),
        "reviewer": "", "scores": {key: None for key in sorted(RUBRIC_KEYS)},
        "critical_errors": [], "notes": "",
    }


def _review_status(case, state, dataset_hash, review):
    if not review or review.get("completed") is False:
        return {"status": "pending", "passed": None, "detail": "尚未进行人工语义审核"}
    try:
        parsed = HumanReview.model_validate(review)
        bindings = review_template(case, state, dataset_hash)
        if any(getattr(parsed, key) != bindings[key] for key in
               ("case_id", "case_sha256", "dataset_sha256", "run_sha256")):
            raise ValueError("人工评分与当前问题、资料或运行记录不匹配")
    except ValueError as error:
        return {"status": "invalid", "passed": False, "detail": str(error)}
    total = sum(parsed.scores.values())
    passed = not parsed.critical_errors and min(parsed.scores.values()) >= 1 and total >= 8
    return {"status": "reviewed", "passed": passed, "score": total / 10,
            "scores": parsed.scores, "critical_errors": parsed.critical_errors,
            "reviewer": parsed.reviewer, "notes": parsed.notes}


def _fact_matches(observation, requirement):
    if not observation.get("ok") or observation.get("tool") != requirement.tool:
        return False
    if not _arguments_match(requirement.arguments, observation.get("arguments", {})):
        return False
    try:
        return same_value(requirement.value, resolve_path(observation.get("data", {}), requirement.path))
    except (KeyError, IndexError, TypeError):
        return False


def _arguments_match(expected, actual):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(k in actual and _arguments_match(v, actual[k]) for k, v in expected.items())
    if isinstance(expected, str) and isinstance(actual, str) and "T" in expected:
        try:
            return datetime.fromisoformat(expected.replace("Z", "+00:00")) == datetime.fromisoformat(actual.replace("Z", "+00:00"))
        except ValueError:
            pass
    return expected_matches(expected, actual)


def _filter_signature(filters):
    """Non-time restrictions; equivalent spellings must not change the population."""
    from gamelens.tools.investigation import ReviewFilter
    parsed = ReviewFilter.model_validate(filters)
    result = parsed.model_dump(mode="json")
    for key in ("period", "start_at", "end_at"):
        result.pop(key)
    result["keywords"] = sorted({s.casefold() for s in result["keywords"]})
    if len(result["keywords"]) <= 1:
        result["keyword_mode"] = "any"
    if result["min_playtime_hours"] == 0 and result["max_playtime_hours"] is None:
        result["playtime_basis"] = "unrestricted"
    return result


def _query_signature(arguments, state, dataset):
    """Canonical time interval and population, without computing any gold counts."""
    from gamelens.tools.investigation import QueryReviewsArgs
    args = QueryReviewsArgs.model_validate(arguments)
    if args.appid != state["appid"]:
        raise ValueError("查询AppID与任务不匹配")
    filters = args.filters
    start = filters.start_at.timestamp() if filters.start_at else None
    end = filters.end_at.timestamp() if filters.end_at else None
    if filters.period != "all":
        pid = args.patch_news_id or state.get("patch_news_id")
        patch = next(p for p in dataset["patches"] if p["news_id"] == pid)
        pivot = patch["published_at_timestamp"]
        days = args.window_days or state["window_days"]
        from gamelens.core.local_time import shift_calendar_days
        left, right = (shift_calendar_days(pivot,-days), pivot) if filters.period == "before" else (pivot, shift_calendar_days(pivot,days))
        start = max(start, left) if start is not None else left
        end = min(end, right) if end is not None else right
    return {"bounds": {"start_inclusive": start, "end_exclusive": end},
            "filters": _filter_signature(filters.model_dump(mode="json"))}


def _equivalent_fact_paths(observation, requirement, case, state, dataset):
    """Accept query/compare_cohorts when scopes are identical, not just numbers.

The original oracle plan supplies an independent definition of the requested
population. Different legal tool strategies can report the same factual answer.
"""
    if requirement.tool != "query_reviews" or requirement.path[:1] != ["stats"]:
        return [requirement.path] if _fact_matches(observation, requirement) else []
    if not observation.get("ok"):
        return []
    references = [p["arguments"] for p in case.oracle_plan if p["tool"] == "query_reviews"
                  and _arguments_match(requirement.arguments, p["arguments"])]
    data, arguments = observation.get("data", {}), observation.get("arguments", {})
    candidates = []
    if observation.get("tool") == "query_reviews":
        candidates.append(([], arguments, data))
    elif observation.get("tool") == "compare_cohorts":
        for group in ("cohort_a", "cohort_b"):
            group_args = {key: arguments[key] for key in ("appid", "patch_news_id", "window_days") if key in arguments}
            group_args["filters"] = arguments.get(group, {})
            candidates.append(([group], group_args, data.get(group, {})))
    matched = []
    for prefix, actual_args, actual_data in candidates:
        try:
            actual = _query_signature(actual_args, state, dataset)
            returned = {"bounds": actual_data["bounds"], "filters": _filter_signature(actual_data["filters"])}
            # Detect inconsistent argument/output records as well as wrong windows.
            if not expected_matches(actual, returned):
                continue
            if not any(expected_matches(_query_signature(reference, state, dataset), actual) for reference in references):
                continue
            path = prefix + requirement.path
            if same_value(requirement.value, resolve_path(data, path)):
                matched.append(path)
        except (ValueError, KeyError, IndexError, TypeError, StopIteration):
            continue
    return matched


def evaluate_business_run(case: BusinessCase, state: dict, dataset: dict,
                          review: dict | None = None, *, execution_mode="agent") -> dict:
    """Replay scoring without reading artifact paths or invoking any models."""
    observations = state.get("observations", [])
    claims = state.get("final_claims", [])
    citations = state.get("final_citations", [])
    news = state.get("final_news_citations", [])
    if not all(isinstance(v, list) for v in (observations, claims, citations, news)):
        raise ValueError("运行记录的观察、事实和引用必须是列表")
    steps = [o.get("step") for o in observations if o.get("tool") != "finish"]
    unique_steps = all(type(s) is int and s > 0 for s in steps) and len(set(steps)) == len(steps)
    provenance_errors = []
    try:
        provenance_errors = validate_grounding(observations, claims, citations)
        provenance_errors += validate_news_citations(observations, news)
        provenance_errors += validate_comment_citations(observations, state.get('final_comment_citations', []))
    except (KeyError, TypeError, IndexError, AttributeError):
        provenance_errors.append("结构化事实或引用格式错误")
    # Also check the trusted fixed corpus, not just whatever the candidate returned.
    review_map = {r["review_id"]: r for r in dataset["reviews"]}
    news_map = {r["news_id"]: r for r in dataset["patches"]}
    for items, source_map, id_key, text_key in (
        (citations, review_map, "review_id", "text"),
        (news, news_map, "news_id", "contents"),
    ):
        for item in items:
            original = source_map.get(item.get(id_key))
            quote = item.get("quote")
            if not original or not isinstance(quote, str) or not quote.strip() or quote not in original[text_key]:
                provenance_errors.append(f"引用不在固定原始资料中：{item.get(id_key)}")
    comment_map = {(c['news_id'], c['comment_id']): c for c in dataset.get('announcement_comments', [])}
    for citation in state.get('final_comment_citations', []):
        original = comment_map.get((citation.get('news_id'), citation.get('comment_id')))
        if not original or citation.get('quote', '') not in original['text']:
            provenance_errors.append('公告留言引用不在固定原始资料中：'+str(citation.get('comment_id')))
    fact_results = []
    for requirement in case.facts:
        matching = [(o, path) for o in observations for path in
                    _equivalent_fact_paths(o, requirement, case, state, dataset)]
        reported = any(c.get("source_step") == o.get("step") and
                       c.get("fact_path") == path and same_value(c.get("value"), requirement.value)
                       for o, path in matching for c in claims)
        fact_results.append({"tool": requirement.tool, "path": requirement.path,
                             "expected": requirement.value, "tool_correct": bool(matching),
                             "reported_and_grounded": reported,
                             "supporting_sources": [{"step": o["step"], "tool": o["tool"], "path": path} for o, path in matching]})
    groups = []
    for group in case.evidence_groups:
        cited = citations if group.kind == "review" else news
        key = "review_id" if group.kind == "review" else "news_id"
        found = sorted({c.get(key) for c in cited} & set(group.ids))
        groups.append({"name": group.name, "matched_ids": found,
                       "required_distinct": group.min_distinct, "passed": len(found) >= group.min_distinct})
    expected_hashes = dataset["source_hashes"]
    sources = [o.get("data", {}).get("source") for o in observations if o.get("ok")]
    sources = [s for s in sources if s]
    # A candidate needs at least one auditable review snapshot; identity tags alone are insufficient.
    # Announcement lookup now has its own auditable source; it must bind to the
    # fixed patch corpus, not be mistaken for a review snapshot. Still require a
    # real review snapshot for these existing business cases.
    snapshot_ok = any(s.get("sha256")==expected_hashes["reviews"] for s in sources)
    for observation in observations:
        source=observation.get("data",{}).get("source") if observation.get("ok") else None
        if not source:continue
        expected="patches" if observation.get("tool")=="list_patches" and source.get("scope")=="cached_announcements_only" else "reviews"
        snapshot_ok = snapshot_ok and source.get("sha256")==expected_hashes[expected]
    artifact_sources_ok = True
    for source in sources:
        for reference in source.get("reference_files", []):
            if reference.get("sha256") not in set(expected_hashes.values()):
                artifact_sources_ok = False
    tagged = state.get("evaluation", {})
    identity_ok = (tagged.get("case_id") == case.id and tagged.get("dataset_sha256") == dataset["sha256"]
                   and tagged.get("case_sha256") == digest(case.model_dump(mode="json")))
    checks = {
        # Existing cases require a complete investigation. A delivered limited
        # report is not a successful complete-task run; older records still use
        # their original gold facts/coverage checks instead of invented progress.
        "progress_complete": not state.get("investigation_progress") or
            state["investigation_progress"].get("can_finalize") is True,
        "question_identity": state.get("goal") == case.goal and state.get("appid") == dataset.get("appid", 730),
        "completed": state.get("status") == "completed" and isinstance(state.get("final_answer"), str) and bool(state["final_answer"].strip()),
        "verification_gate": bool(state.get("artifacts", {}).get("final_report")) and
            bool(state.get("artifacts", {}).get("evidence_assessed")) and
            any(o.get("ok") and o.get("tool") == "verify_investigation" for o in observations),
        "dataset_identity": identity_ok and snapshot_ok and artifact_sources_ok,
        "unique_observation_steps": unique_steps,
        "gold_facts": all(f["tool_correct"] and f["reported_and_grounded"] for f in fact_results),
        "citation_integrity": not provenance_errors,
        "evidence_coverage": all(g["passed"] for g in groups),
        "step_budget": len(observations) <= case.max_tool_steps,
        "rag_policy": not ((case.rag_mode == "off" or state.get("rag_mode")=="off") and any(o.get("tool") in
                         {"build_game_index", "search_game_evidence"} for o in observations)) and
                         (state.get("rag_mode")!="on" or any(o.get("tool")=="search_game_evidence" for o in observations)),
    }
    human = _review_status(case, state, dataset["sha256"], review)
    automatic_passed = all(checks.values())
    # Offline baselines are never reclassified as successful live Agent answers.
    accepted = execution_mode == "agent" and automatic_passed and human["passed"] is True
    return {
        "case_id": case.id, "category": case.category, "split": case.split,
        "decision_metrics": decision_metrics(state),
        "execution_mode": execution_mode, "automatic_passed": automatic_passed,
        "accepted": accepted, "checks": checks, "facts": fact_results,
        "evidence_groups": groups, "provenance_errors": provenance_errors,
        "human_review": human,
        "status": "accepted" if accepted else ("automatic_failed" if not automatic_passed else
                  "human_failed" if human["passed"] is False else "awaiting_human_review"),
        "tool_calls": len(observations), "run_sha256": digest(state),
        "answer_semantics_automatically_verified": False,
    }


def aggregate_business_results(results: list[dict], planned_cases: int) -> dict:
    """Failures and absent runs remain visible; unknown timing/cost is not zero."""
    agent = [r for r in results if r["execution_mode"] == "agent"]
    reviewed = [r for r in agent if r["human_review"]["status"] == "reviewed"]
    accepted = sum(r["accepted"] for r in agent)
    return {
        "planned_cases": planned_cases, "evaluated_agent_runs": len(agent),
        "evaluated_unique_cases": len({r["case_id"] for r in agent}),
        "planned_case_coverage": len({r["case_id"] for r in agent}) / planned_cases if planned_cases else None,
        "automatic_passed_runs": sum(r["automatic_passed"] for r in agent),
        "human_reviewed_runs": len(reviewed), "accepted_runs": accepted,
        "automatic_pass_rate": sum(r["automatic_passed"] for r in agent) / len(agent) if agent else None,
        "fully_reviewed_acceptance_rate": accepted / len(agent) if agent and len(reviewed) == len(agent) else None,
        "accepted_fraction_including_pending": accepted / len(agent) if agent else None,
        "failure_checks": dict(Counter(k for r in agent for k, passed in r["checks"].items() if not passed)),
        "note": "没有真实Agent运行或人工审核未完成时，不报告最终回答准确率。",
    }
