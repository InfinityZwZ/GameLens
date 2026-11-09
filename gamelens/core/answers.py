"""Conservative deterministic statistics and explicitly unfinished results."""
import hashlib
import json
import math
import re
from pathlib import Path

from gamelens.core.grounding import fact_catalog, validate_grounding
from gamelens.core.source_freshness import active_observations
from gamelens.core.local_time import timezone_signature


def simple_statistics_goal(goal):
    """A narrow optimization, not a general intent classifier."""
    from gamelens.tools.investigation import required_patch_count
    from gamelens.tools.time_scope import requested_directions
    if requested_directions(goal)=={'before','after'}:return False
    if required_patch_count(goal) > 1 or len(set(re.findall(r"(?<![A-Za-z0-9])p\d+(?![A-Za-z0-9])", goal, re.I))) > 1:
        return False
    if re.search(r"为什么|原因|导致|改善|能否|是否|趋势|哪些|修复|持续|反例|对比|比较|公告|崩溃|掉帧|卡顿|性能|bug|刚才|上次|原文|引用|证据|举例|样例|分别|分组", goal, re.I):
        return False
    from gamelens.core.query_contract import requested_metrics
    return bool(re.search(r"评论|评价|样本", goal) and requested_metrics(goal))


def verified_sources_current(state):
    """Do not re-use a verified flag after a later query or edited snapshot."""
    observations = state.get("observations", [])
    sources = [o for o in active_observations(observations) if o.get("ok") and o.get("data", {}).get("source")]
    checks = [o for o in observations if o.get("tool") == "verify_investigation"]
    if not sources or not checks or not checks[-1].get("ok"):
        return False
    if checks[-1].get("step", 0) <= max(o.get("step", 0) for o in sources):
        return False
    if not state.get("artifacts", {}).get("evidence_assessed"):
        return False
    try:
        for observation in sources:
            source = observation["data"]["source"]
            if source.get('time_zone') and source['time_zone'] != timezone_signature():
                return False
            for item in [source, *source.get("reference_files", [])]:
                if hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() != item["sha256"]:
                    return False
    except (OSError, KeyError, TypeError):
        return False
    return True


def safe_stats(stats):
    keys = ("count", "positive_count", "negative_count")
    if any(type(stats.get(k)) is not int or stats[k] < 0 for k in keys):
        return False
    count = stats["count"]
    if stats["positive_count"] + stats["negative_count"] != count:
        return False
    for key, numerator in (("positive_rate", "positive_count"), ("negative_rate", "negative_count")):
        value = stats.get(key)
        if count == 0:
            if value is not None:
                return False
        elif type(value) not in (int, float) or not math.isfinite(value) or not math.isclose(value, stats[numerator]/count, abs_tol=1e-6):
            return False
    return True


def _line(stats,metrics=()):
    if not stats["count"]:
        return "没有匹配的缓存评论，无法计算好评率；不能据此证明问题不存在。"
    rates=[]
    if 'negative_rate' in metrics:rates.append(f"差评率{stats['negative_rate']*100:.2f}%")
    if 'positive_rate' in metrics or not rates:rates.insert(0,f"好评率{stats['positive_rate']*100:.2f}%")
    return (f"{stats['count']}条评论，{stats['positive_count']}条好评、"
            f"{stats['negative_count']}条差评，"+'，'.join(rates)+'。')


def statistical_answer(state, *, partial=False):
    """Only render registered, independently verified observations, never inference."""
    if not verified_sources_current(state):
        return None
    eligible = [o for o in active_observations(state.get("observations", [])) if o.get("ok") and
                o.get("tool") in {"query_reviews", "compare_cohorts", "compare_versions"}]
    if not eligible:
        return None
    from gamelens.core.query_contract import contract_for,scope_errors,cohort_scope_errors,statistical_population_errors
    contract=contract_for(state)
    def aligned(item):
        groups=[item] if 'stats' in item else [item.get(k,{}) for k in ('cohort_a','cohort_b')] if 'cohort_a' in item else []
        return (not any(scope_errors(contract,p.get('filters',{}),p.get('bounds',{})) or
                statistical_population_errors(contract,p.get('filters',{})) for p in groups) and
                not (len(groups)>1 and cohort_scope_errors(contract,[p.get('filters',{}) for p in groups])))
    eligible=[o for o in eligible if aligned(o['data'])]
    if not eligible:return None
    last=eligible[-1]
    from gamelens.core.query_contract import requested_metrics
    metrics=requested_metrics(state['goal'])
    data = last["data"]
    paths, lines = [], []
    selected, paths_by_step = [last], {}
    if last["tool"] == "query_reviews":
        if partial:
            selected, seen = [], set()
            for observation in reversed(eligible):
                if observation["tool"] != "query_reviews":
                    continue
                item = observation["data"]
                if not aligned(item):continue
                key = json.dumps({k: item.get(k) for k in ("filters", "bounds", "patch_news_id")}, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    selected.append(observation)
                if len(selected) == 3:
                    break
            selected.reverse()
        from gamelens.analysis.report_prose import filter_description, date
        for observation in selected:
            data = observation["data"]
            if data.get("statistics_status")=="unavailable_window":
                return None  # Let the planner explain unavailable coverage, not report a real zero.
            if partial and simple_statistics_goal(state["goal"]):
                from gamelens.core.investigation_checks import statistics_alignment
                if statistics_alignment(state["goal"], data, state):
                    continue
            if not safe_stats(data.get("stats", {})):
                return None
            bounds = data.get("bounds", {})
            from gamelens.tools.time_scope import validate_bounds
            try:validate_bounds(bounds)
            except ValueError:return None
            label = filter_description(data.get("filters", {}))
            if bounds.get("start_inclusive") is not None and bounds.get("end_exclusive") is not None:
                label += f"；{date(bounds['start_inclusive'])}至{date(bounds['end_exclusive'])} 北京时间时间，结束时刻不含"
            lines.append(f"{label}：{_line(data['stats'],metrics)}")
            paths_by_step[observation["step"]] = [["stats", key] for key in ("count", "positive_count", "negative_count", "positive_rate", "negative_rate")]
        if not lines:
            return None
    else:
        groups = [(name, data.get(name, {}), label) for name, label in (("cohort_a", "群体A"), ("cohort_b", "群体B"))] if last["tool"] == "compare_cohorts" else [(i, v, v.get("title", "版本")) for i, v in enumerate(data.get("versions", []))]
        if not groups or any(not safe_stats(group.get("stats", {})) for _, group, _ in groups):
            return None
        for name, group, label in groups:
            lines.append(f"{label}：{_line(group['stats'],metrics)}")
            prefix = [name] if isinstance(name, str) else ["versions", name]
            paths += [prefix+["stats", key] for key in ("count", "positive_count", "negative_count", "positive_rate", "negative_rate")]
        lines.append("这些差异按各组标明的计数口径描述，不能证明更新导致了口碑或故障变化。")
        paths_by_step[last["step"]] = paths
    claims = [{k: v for k, v in fact.items() if k != "fact_id"} for fact in fact_catalog(selected)
              if fact["fact_path"] in paths_by_step.get(fact["source_step"], [])]
    if validate_grounding(state["observations"], claims, []):
        return None
    public_counts=any(o['data'].get('statistics_basis')=='steam_public_window_counts' or
        any(o['data'].get('cohort_'+s,{}).get('statistics_basis')=='steam_public_window_counts' for s in ('a','b')) for o in selected)
    prefix = "**调查未完成**：模型请求中断。以下仅是已核验的部分统计，不是最终调查结论。\n\n" if partial else ("按已核验的评论窗口计数：\n\n" if public_counts else "按已核验的本地评论样本统计：\n\n")
    limitation=('平台计数对应采集时公开接口对精确窗口的全部匹配结果；原文是有限样例，当前推荐态度/文本不等于历史当时快照，不代表全部玩家。' if public_counts else '仅代表当前缓存，不代表Steam总体。')
    answer = prefix + "\n".join('- ' + line for line in lines) + "\n\n**范围与限制**\n\n"+limitation
    if partial:
        answer += "\n\n可点击恢复任务继续生成；恢复可能产生模型调用费用。"
    from gamelens.core.query_contract import contract_for
    def population_complete(population):
        if 'statistics_complete_for_scope' in population:return population['statistics_complete_for_scope']
        if population.get('statistics_basis')=='cached_subset' and not contract_for(state)['explicit_cached_scope']:return False
        return True  # Legacy offline payloads without a declared sampling basis.
    complete=all(population_complete(p) for o in selected for p in
        ([o['data']] if 'stats' in o['data'] else [o['data']['cohort_a'],o['data']['cohort_b']] if 'cohort_a' in o['data'] else []))
    if not partial and not complete:
        answer='**已获取样本的统计结果**\n\n'+"\n".join('- '+line for line in lines)+'\n\n**范围与限制**\n\n尚未取得该时间和人群条件的完整数据。上述比例仅针对已获取的符合条件样本，不是整个月或全部符合条件玩家的好评率。有限采集达到数量/分页上限，不是接口没有更多评论。'
    return {"answer": answer, "claims": claims, "scope_limited":not complete,
            "details": "以下是程序根据已核验的筛选统计整理的结果，不包含模型推论。\n\n"+"\n\n".join(lines)}
