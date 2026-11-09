"""Deterministic evaluation metrics for saved Agent run states."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    expected_tools_in_order: list[str] = Field(default_factory=list)
    required_artifacts: list[str] = Field(default_factory=list)
    required_answer_fragments: list[str] = Field(default_factory=list)
    forbidden_answer_fragments: list[str] = Field(default_factory=list)
    max_tool_steps: int | None = Field(default=None, ge=1)
    require_completed: bool = True


class MetricResult(BaseModel):
    passed: bool
    score: float = Field(ge=0, le=1)
    detail: str


class EvaluationResult(BaseModel):
    case: str
    passed: bool
    overall_score: float = Field(ge=0, le=1)
    metrics: dict[str, MetricResult]


def _is_ordered_subsequence(expected: list[str], actual: list[str]) -> bool:
    if not expected:
        return True
    position = 0
    for item in actual:
        if item == expected[position]:
            position += 1
            if position == len(expected):
                return True
    return False


def evaluate_run(
    state: dict[str, Any],
    case: EvaluationCase,
) -> EvaluationResult:
    observations = list(state.get("observations", []))
    actual_tools = [item.get("tool", "") for item in observations]
    artifacts = state.get("artifacts", {})
    answer = str(state.get("final_answer", ""))

    ordered = _is_ordered_subsequence(
        case.expected_tools_in_order,
        actual_tools,
    )
    missing_artifacts = [
        item for item in case.required_artifacts if not artifacts.get(item)
    ]
    missing_fragments = [
        item for item in case.required_answer_fragments if item not in answer
    ]
    forbidden_hits = [
        item for item in case.forbidden_answer_fragments if item in answer
    ]
    completed = (
        state.get("status") == "completed"
        if case.require_completed
        else True
    )
    within_budget = (
        len(observations) <= case.max_tool_steps
        if case.max_tool_steps is not None
        else True
    )

    metrics = {
        "tool_selection": MetricResult(
            passed=ordered,
            score=1.0 if ordered else 0.0,
            detail=(
                f"实际工具序列：{actual_tools}"
                if ordered
                else (
                    f"期望顺序子序列{case.expected_tools_in_order}，"
                    f"实际为{actual_tools}"
                )
            ),
        ),
        "completion_gate": MetricResult(
            passed=completed and not missing_artifacts,
            score=1.0 if completed and not missing_artifacts else 0.0,
            detail=(
                "完成状态与必要产物齐全"
                if completed and not missing_artifacts
                else f"缺少产物：{missing_artifacts}；状态：{state.get('status')}"
            ),
        ),
        "answer_fidelity": MetricResult(
            passed=not missing_fragments and not forbidden_hits,
            score=(
                1.0 if not missing_fragments and not forbidden_hits else 0.0
            ),
            detail=(
                "结论满足必备事实且未命中禁用断言"
                if not missing_fragments and not forbidden_hits
                else (
                    f"缺少：{missing_fragments}；"
                    f"命中禁用断言：{forbidden_hits}"
                )
            ),
        ),
        "step_budget": MetricResult(
            passed=within_budget,
            score=1.0 if within_budget else 0.0,
            detail=f"工具调用{len(observations)}步",
        ),
    }
    overall = sum(item.score for item in metrics.values()) / len(metrics)
    return EvaluationResult(
        case=case.name,
        passed=all(item.passed for item in metrics.values()),
        overall_score=overall,
        metrics=metrics,
    )


def load_evaluation_case(path: Path) -> EvaluationCase:
    return EvaluationCase.model_validate_json(path.read_text(encoding="utf-8"))


def load_run_state(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("运行记录最外层必须是JSON对象")
    return payload
