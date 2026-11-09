from gamelens.evaluation.contracts import EvaluationCase, evaluate_run


def test_trace_evaluator_accepts_faithful_bounded_run() -> None:
    state = {
        "status": "completed",
        "observations": [
            {"tool": "inspect_data"},
            {"tool": "assess_evidence"},
        ],
        "artifacts": {
            "final_report": "reports/test.md",
            "evidence_assessed": "true",
        },
        "final_answer": "完整窗口与分类样本存在差异。",
    }
    case = EvaluationCase(
        name="faithful",
        expected_tools_in_order=["inspect_data", "assess_evidence"],
        required_artifacts=["final_report", "evidence_assessed"],
        required_answer_fragments=["完整", "样本"],
        forbidden_answer_fragments=["性能问题确定上升"],
        max_tool_steps=3,
    )

    result = evaluate_run(state, case)

    assert result.passed is True
    assert result.overall_score == 1.0


def test_trace_evaluator_rejects_unsupported_claim() -> None:
    state = {
        "status": "completed",
        "observations": [{"tool": "inspect_data"}],
        "artifacts": {"final_report": "reports/test.md"},
        "final_answer": "性能问题确定上升",
    }
    case = EvaluationCase(
        name="unsupported",
        expected_tools_in_order=["inspect_data", "assess_evidence"],
        required_artifacts=["final_report", "evidence_assessed"],
        required_answer_fragments=["完整", "样本"],
        forbidden_answer_fragments=["性能问题确定上升"],
        max_tool_steps=3,
    )

    result = evaluate_run(state, case)

    assert result.passed is False
    assert result.metrics["tool_selection"].passed is False
    assert result.metrics["completion_gate"].passed is False
    assert result.metrics["answer_fidelity"].passed is False
