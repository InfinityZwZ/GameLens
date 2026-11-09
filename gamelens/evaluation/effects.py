"""Ground-truth task scoring with explicit numerical and quotation provenance."""
from gamelens.core.grounding import same_value, resolve_path, validate_grounding


def expected_matches(expected,actual):
    if isinstance(expected,dict):
        return isinstance(actual,dict) and all(key in actual and expected_matches(value,actual[key]) for key,value in expected.items())
    if isinstance(expected,list):
        return isinstance(actual,list) and len(actual)==len(expected) and all(expected_matches(a,b) for a,b in zip(expected,actual))
    return same_value(expected,actual)


def score_effect_case(case,state,corpus):
    observations=state.get("observations",[])
    relevant=[o for o in observations if o.get("tool")==case["tool"]]
    claims=state.get("final_claims",[])
    citations=state.get("final_citations",[])
    grounding_errors=validate_grounding(observations,claims,citations)
    corpus_map={review["review_id"]:review for review in corpus}
    bad_citations=[c["review_id"] for c in citations if c["review_id"] not in corpus_map or c["quote"] not in corpus_map[c["review_id"]]["text"]]
    if case.get("expect_error"):
        correct=any(not o.get("ok") for o in relevant) and not citations and not claims and not grounding_errors and bool(state.get("final_answer"))
        return {"name":case["name"],"passed":correct,"expected_failure_handled":correct,"grounding_errors":grounding_errors}
    successful=[o for o in relevant if o.get("ok")]
    output_correct=bool(successful) and expected_matches(case["expected"],successful[-1].get("data",{}))
    answer_grounded=bool(claims) and not grounding_errors
    # A correct tool invocation alone must not earn a correct-answer score.
    requested_paths=[]
    if "stats" in case["expected"]: requested_paths=[["stats","count"],["stats","positive_rate"]]
    if "cohort_a" in case["expected"]: requested_paths=[[group,"stats",key] for group in ("cohort_a","cohort_b") for key in ("count","positive_rate")]
    if "versions" in case["expected"]: requested_paths=[["versions",index,"stats",key] for index in (0,1) for key in ("count","positive_rate")]
    if "review" in case["expected"]: requested_paths=[["review","review_id"]]
    if successful:
        source_step=successful[-1]["step"]
        answer_grounded=answer_grounded and all(any(c["source_step"]==source_step and c["fact_path"]==path and same_value(c["value"],resolve_path(case["expected"],path)) for c in claims) for path in requested_paths)
    citations_correct=not bad_citations and not grounding_errors and (bool(citations) if case.get("require_citation") else True)
    # This grader does not claim to prove the semantics of arbitrary narrative prose.
    passed=output_correct and answer_grounded and citations_correct and state.get("status")=="completed"
    return {"name":case["name"],"passed":passed,"tool_output_correct":output_correct,
            "requested_facts_correct":answer_grounded,"citations_correct":citations_correct,
            "grounding_errors":grounding_errors,"bad_citations":bad_citations,
            "tool_calls":len(observations),"narrative_requires_human_review":True}
