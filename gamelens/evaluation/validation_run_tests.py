"""Offline validation helpers for frozen real-cache tasks."""
from gamelens.evaluation.benchmark import initial_state
from gamelens.evaluation.real_snapshot import TASKS


def oracle_snapshot(root):
    """Prescribed source-reading plans, never a synthetic model answer."""
    from gamelens.evaluation.suite import business_registry
    registry=business_registry(root,None)
    results=[]
    for task in TASKS:
        state=initial_state(task["case"]["id"],task["case"]["goal"])
        state["thread_id"]="oracle-"+task["case"]["id"]
        state.update(appid=task["appid"],artifacts={},patch_news_id=None)
        observations=[]
        for planned in task["case"]["oracle_plan"]:
            result=registry.execute(planned["tool"],planned["arguments"],state)
            observations.append({"tool":planned["tool"],"ok":result.ok})
            state["step_count"]+=1
            state["artifacts"].update(result.artifacts)
        results.append({"id":task["case"]["id"],"passed":all(o["ok"] for o in observations),"checks":observations,"tests_semantics":False})
    return results
