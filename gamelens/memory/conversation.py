"""Bounded session memory: keep provenance and never treat old answers as facts."""

from gamelens.memory.context import ContextPolicy, PlannerContextBuilder


def build_conversation_memory(conversation_id: str, runs: list[dict]) -> dict:
    compact = PlannerContextBuilder(ContextPolicy(max_string_chars=1600))._compact
    recent = runs[-4:]
    older = runs[:-4]
    memory = {
        "conversation_id": conversation_id,
        "previous_turn_count": len(runs),
        "recent_turns": [
            {"run_id": run["run_id"], "turn_index": run.get("turn_index", index+1),
             "user_question": run["request"]["goal"],
             "assistant_answer_unverified": run.get("final_answer", ""),
             "status": run["status"]}
            for index,run in enumerate(recent)
        ],
        "older_turn_digest": [
            {"run_id": run["run_id"], "question": run["request"]["goal"][:300],
             "answer_unverified": run.get("final_answer", "")[:400], "status": run["status"]}
            for run in older[-8:]
        ],
        "omitted_older_turns": max(0,len(older)-8),
        "trust_note": "历史问题用于理解指代，历史回答是模型输出，不能当作已验证事实；需通过工具重新核对证据。",
        "inherited_artifacts": {},
        "inherited_config": {},
        "prior_evidence": [],
    }
    completed = [run for run in runs if run["status"] == "completed"]
    if completed:
        previous = completed[-1]
        memory["inherited_config"] = previous["request"]
        patch_ids = {event.get("data", {}).get("patch_news_id") or event.get("arguments", {}).get("patch_news_id")
                     for event in previous.get("events", []) if event.get("ok") and
                     event.get("tool") in {"query_reviews", "compare_cohorts", "search_game_evidence"}}
        patch_ids.discard(None)
        memory["selected_patch_news_id"] = next(iter(patch_ids)) if len(patch_ids) == 1 else None
        # Require a fresh evidence assessment in every round.
        memory["inherited_artifacts"] = {key:value for key,value in previous.get("artifacts",{}).items() if key != "evidence_assessed"}
        memory["prior_evidence"] = [
            {"source_run_id": previous["run_id"], "tool": event["tool"],
             "summary": event["summary"], "data": event.get("data",{})}
            for event in previous.get("events",[])
            if event.get("ok") and event.get("tool") == "assess_evidence"
        ][-1:]
    return compact(memory)
