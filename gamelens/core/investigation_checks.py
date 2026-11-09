"""Conservative request/evidence gates, not an automatic semantic judge."""
import re
from gamelens.core.local_time import local_day, local_datetime, shift_calendar_days


def statistics_alignment(goal, data, state):
    """Return explicit disagreements; never silently translate ambiguous dates."""
    filters = data.get("filters", {})
    errors = []
    from gamelens.core.query_contract import contract_for, scope_errors,statistical_population_errors
    contract=contract_for({**state,'goal':goal})
    errors.extend(scope_errors(contract,filters,data.get('bounds',{})))
    errors.extend(statistical_population_errors(contract,filters))
    from gamelens.tools.time_scope import validate_bounds
    try:validate_bounds(data.get("bounds",{}))
    except ValueError as error:errors.append(str(error))
    patch_ids = set(re.findall(r"(?<![A-Za-z0-9])p\d+(?![A-Za-z0-9])", goal, re.I))
    if len(patch_ids) == 1 and (data.get("patch_news_id") or "").lower() != next(iter(patch_ids)).lower():
        errors.append("查询补丁与问题指定的补丁不一致")
    from gamelens.tools.time_scope import requested_directions
    directions=requested_directions(goal)
    after='after' in directions
    before='before' in directions
    if after != before and filters.get("period") != ("after" if after else "before"):
        errors.append("查询前后窗口与问题不一致")
    days = re.search(r"(?:更新|补丁|版本)[前后]\s*([一二三四五六七八九十\d]+)\s*天", goal)
    if days:
        value = days[1]
        expected = int(value) if value.isdigit() else "一二三四五六七八九十".index(value)+1
        bounds = data.get("bounds", {})
        if bounds.get("start_inclusive") is None or bounds.get("end_exclusive") is None or shift_calendar_days(bounds['start_inclusive'],expected) != bounds['end_exclusive']:
            errors.append("查询时间跨度与问题要求的天数不一致")
    # Playtime basis and rate denominators use the same contract as tool
    # validation; a second regex here previously contradicted JSON conditions.
    contract=contract_for({**state,'goal':goal})
    if not contract.get('has_structured_conditions') and not contract['calendar_window'] and not re.search(r"更新[前后]|补丁[前后]|版本[前后]|\d{4}-\d{2}-\d{2}|时长|小时|中文|英文|关键词|包含|提到|只看|只统计|仅", goal):
        if filters.get("period", "all") != "all" or filters.get("start_at") or filters.get("end_at"):
            errors.append("问题要求本地全集，查询却限制了时间窗口")
        if filters.get("keywords") or filters.get("language") or filters.get("min_playtime_hours", 0) or filters.get("max_playtime_hours") is not None:
            errors.append("问题要求本地全集，查询却增加了群体/关键词筛选")
    return list(dict.fromkeys(errors))


def fast_statistics_allowed(goal, data, state):
    """Only optimize supported, unambiguous conditions, otherwise use the planner."""
    if re.search(r"昨天|今天|本周|最近\s*\d+\s*天|关键词|包含|提到", goal):
        return False
    return not statistics_alignment(goal, data, state)


def explicit_patch_alignment(goal, action, arguments, state):
    """An explicit user patch overrides the session's default selected patch."""
    ids=set(re.findall(r"(?<![A-Za-z0-9])p\d+(?![A-Za-z0-9])",goal,re.I))
    if len(ids)!=1 or action not in {"query_reviews","compare_cohorts","search_game_evidence"}:
        return []
    filters=[arguments.get("filters",{})] if action!="compare_cohorts" else [arguments.get(k,{}) for k in ("cohort_a","cohort_b")]
    default="all" if action=="search_game_evidence" else "after"
    if not any(f.get("period",default)!="all" for f in filters):return []
    actual=arguments.get("patch_news_id") or state.get("patch_news_id")
    if (actual or "").lower()!=next(iter(ids)).lower():
        return ["问题指定的补丁优先于会话默认补丁，请显式patch_news_id="+next(iter(ids))]
    return []


def named_patch_time_alignment(goal, filters, patch, days):
    """Verify absolute as well as relative windows for explicit, bounded requests.

    An absolute timestamp must not bypass an explicitly named patch. Multiple
    requested windows are alternatives for an individual query, not a union.
    """
    intervals=[]
    from gamelens.tools.time_scope import requested_windows
    for direction,number in requested_windows(goal):
        pivot=patch.published_at_timestamp
        intervals.append((shift_calendar_days(pivot,-number),pivot) if direction=="前" else (pivot,shift_calendar_days(pivot,number)))
    if not intervals:return []
    left=filters.start_at.timestamp() if filters.start_at else None
    right=filters.end_at.timestamp() if filters.end_at else None
    if filters.period!="all":
        pivot=patch.published_at_timestamp
        begin,end=(shift_calendar_days(pivot,-days),pivot) if filters.period=="before" else (pivot,shift_calendar_days(pivot,days))
        left=max(left,begin) if left is not None else begin
        right=min(right,end) if right is not None else end
    if (left,right) not in intervals:
        return ["查询时间与用户指定补丁的前/后窗口不一致；绝对日期也必须对齐该补丁的发布时间，结束边界不包含"]
    return []


def cited_window_audit(observations, citations):
    """Map verified review quotes to disjoint patch windows without inferring defects."""
    coverage = next((o.get("data", {}) for o in reversed(observations)
                     if o.get("ok") and o.get("tool") == "investigate_recent_versions"), {})
    windows = coverage.get("versions", [])
    by_step = {o.get("step"): o.get("data", {}) for o in observations if o.get("ok")}
    members = {}
    for citation in citations:
        data = by_step.get(citation["source_step"], {})
        records = data.get("reviews", []) + ([data["review"]] if "review" in data else [])
        record = next((r for r in records if r["review_id"] == citation["review_id"]), None)
        if not record or citation["quote"] not in record["text"]:
            continue
        moment = record.get("created_at_timestamp")
        if moment is None:
            continue
        for window in windows:
            if window["start_inclusive"] <= moment < window["end_exclusive"]:
                members.setdefault(window["news_id"], set()).add(record["review_id"])
    return {"cited_window_count": len(members), "review_ids_by_window": {k: sorted(v) for k, v in members.items()},
            "semantics_verified": False,
            "note": "原文分布核验不证明同一缺陷持续存在；反馈可能描述不同原因，仍需分析反例与复现。"}


def investigation_finish_checks(goal, observations, citations, scope_limited):
    from gamelens.tools.investigation import required_patch_count
    audit = cited_window_audit(observations, citations)
    errors = []
    from gamelens.tools.time_scope import requested_patch_date
    if requested_patch_date(goal):
        lookups=[o.get("data",{}) for o in observations if o.get("ok") and o.get("tool") in {"list_patches", "list_announcements"} and o.get("data",{}).get("requested_date")]
        if (not lookups or lookups[-1].get("lookup_status")!="found_in_cache") and not scope_limited:
            errors.append("历史目标更新尚未唯一定位；须补充公告或scope_limited=true说明缺失/询问年份，不能声称完整调查")
        coverages=[o.get("data",{}).get("coverage",o.get("data",{}).get("review_coverage",{})) for o in observations if o.get("ok")]
        if any(c.get("status")=="outside_observed_range" or c.get("requested_start_not_observed") for c in coverages) and not scope_limited:
            errors.append("目标更新后的评论历史覆盖不足；须scope_limited=true说明缺失，仅报告实际缓存，不能把空结果当作实际0条反馈")
    if required_patch_count(goal) > 1 and re.search(r"bug|性能|崩溃|掉帧|卡顿|问题|修好", goal, re.I):
        if audit["cited_window_count"] < 2 and not scope_limited:
            errors.append("跨版本问题调查缺少至少两个不同版本窗口的原文；补查原文，或scope_limited=true明确证据不足")
        if re.search(r"反例|不再|修好|一直|持续", goal):
            broad = any(o.get("ok") and o.get("tool") in {"query_reviews", "search_game_evidence"}
                        and o.get("data", {}).get("filters", {}).get("is_positive") is None
                        and o.get("data", {}).get("reviews") for o in observations)
            if not broad and not scope_limited:
                errors.append("仅看差评不能核查恢复正常的反例；须检索不限推荐态度的原文，或明确限制")
    return errors, audit


def population_claim_errors(observations, answer):
    """Reject the specific but dangerous count-category error seen in production.

    Retrieval/screening may select zero issue evidence even though many reviews
    satisfy the deterministic player filters.  A zero in the former category
    must never be presented as a zero in the latter category.
    """
    if not answer:
        return []
    eligible_counts = []
    for observation in observations:
        if not observation.get("ok") or observation.get("tool") != "search_game_evidence":
            continue
        data = observation.get("data", {})
        eligible = data.get("population", {}).get("eligible_review_count")
        selected = data.get("retrieval", {}).get("selected_evidence_count", data.get("returned_count"))
        if type(eligible) is int and eligible > 0 and selected == 0:
            eligible_counts.append(eligible)
    if not eligible_counts:
        return []

    explicit_zero_population_claims = (
        r"(?:符合(?:筛选|条件)?|满足(?:筛选|条件)?)[^。；！？\n]{0,45}(?:评论|玩家)[^。；！？\n]{0,25}(?:为|是|：)?\s*(?:0|零)\s*条",
        r"(?:评论|玩家)[^。；！？\n]{0,45}(?:符合|满足)[^。；！？\n]{0,25}(?:为|是|：)?\s*(?:0|零)\s*条",
        r"\b(?:no|zero)\s+(?:matching|eligible|qualifying)[^.!?\n]{0,30}(?:reviews?|players?)\b",
    )
    absent_population_claims = (
        r"(?:没有|不存在|未(?:采集|找到|检索|返回)到?)(?:任何|一条)?[^。；！？\n]{0,60}(?:符合(?:筛选|条件)?[^。；！？\n]{0,25})?(?:玩家)?评论",
        r"\bthere\s+(?:are|were)\s+no\s+player\s+reviews?[^.!?\n]{0,60}(?:matching|that match|meeting)[^.!?\n]{0,40}(?:filters?|conditions?)\b",
    )
    for clause in re.split(r"[。；！？\n]+", answer):
        if any(re.search(pattern, clause, re.I) for pattern in explicit_zero_population_claims):
            return ["语义证据筛选为0不能表述为符合硬条件的人群评论为0；须分别报告eligible_review_count与selected_evidence_count"]
        evidence_only = bool(re.search(
            r"(?:评论|原文|直接支持|直接提及|问题语义)[^。；！？\n]{0,16}证据|evidence",
            clause, re.I,
        ))
        if not evidence_only and any(re.search(pattern, clause, re.I) for pattern in absent_population_claims):
            return ["语义证据筛选为0不能表述为符合硬条件的人群评论为0；须分别报告eligible_review_count与selected_evidence_count"]
    return []


def validate_version_assignments(observations, citations, assignments):
    """Check declared version membership, including quotes outside the windows."""
    audit=cited_window_audit(observations,citations)
    actual={rid:pid for pid,ids in audit["review_ids_by_window"].items() for rid in ids}
    cited={c["review_id"] for c in citations}
    seen=set();errors=[]
    for assignment in assignments:
        for rid in assignment["review_ids"]:
            if rid not in cited:
                errors.append("版本归属中出现未逐字引用的评论："+rid)
            elif assignment["news_id"] != actual.get(rid):
                errors.append("评论创建时间与声明版本窗口不符："+rid)
            if rid in seen:errors.append("同一评论重复分配到多个版本："+rid)
            seen.add(rid)
    if seen!=cited:errors.append("跨版本回答必须逐项声明所有引用的版本归属，窗口外评论用news_id=null")
    return errors


EVIDENCE_REQUIREMENTS = ('statistics', 'review_evidence', 'announcement_check', 'counterexample_check', 'announcement_comments')


def assess_investigation_progress(state, *, required_evidence=(), scope_limited=False,
                                  citations=None, news_citations=None, comment_citations=None):
    """Deterministic coverage checks, NOT a semantic accuracy/confidence judge.

    Existing source/date/quote validators retain authority. Model-proposed aspects
    can add requirements, never remove explicit ones. Recompute from observations;
    never trust a persisted 'sufficient' bit or an old assistant answer.
    """
    from gamelens.tools.investigation import required_patch_count
    from gamelens.tools.time_scope import requested_patch_date, requested_windows, validate_bounds
    from gamelens.core.answers import verified_sources_current, safe_stats
    from gamelens.core.source_freshness import active_observations
    observations = active_observations(state.get('observations', []))
    goal = state.get('goal', '')
    from gamelens.domain.announcement_comments import wants_announcement_feedback, announcement_only_goal
    discussion_goal = wants_announcement_feedback(goal)
    enabled = state.get('investigation_profile') in {'focused', 'extended'} or any(
        o.get('data', {}).get('source') for o in observations)
    exact_review_lookup = bool(re.search(r'原文|完整评论|评论\s*ID', goal, re.I) and
        re.search(r'(?<![A-Za-z0-9])r\d+(?![A-Za-z0-9])|(?:评论\s*ID|review[_ ]?id)\s*[:：]?\s*\d+', goal, re.I))
    requirements = set()
    if enabled:
        requirements.update(r for r in required_evidence if r in EVIDENCE_REQUIREMENTS)
        prior = state.get('investigation_progress', {})
        if prior.get('goal') == goal and prior.get('appid') == state.get('appid'):
            requirements.update(r for r in prior.get('required_evidence', []) if r in EVIDENCE_REQUIREMENTS)
        if discussion_goal:
            requirements.add('announcement_comments')
        if announcement_only_goal(goal):
            requirements.difference_update({'review_evidence', 'statistics', 'counterexample_check'})
        from gamelens.core.query_contract import requested_metrics
        if not discussion_goal and requested_metrics(goal):
            requirements.add('statistics')
        if not discussion_goal and re.search(r'闪退|崩溃|卡顿|掉帧|bug|故障|报错|性能问题|反例|原文|评论\s*(?:ID|id)|\br\d+\b', goal, re.I):
            requirements.add('review_evidence')
        if re.search(r'公告|官方|补丁说明|修复|修好', goal) and (not discussion_goal or re.search(r'官方|公告正文|补丁说明|官方修复', goal)):
            requirements.add('announcement_check')
        if not discussion_goal and not exact_review_lookup and re.search(r'闪退|崩溃|卡顿|掉帧|bug|故障|反例|相反体验|相反反馈', goal, re.I):
            requirements.add('counterexample_check')
    successful = [o for o in observations if o.get('ok')]
    failures = [o for o in observations if o.get('ok') is False]
    from gamelens.core.query_contract import contract_for, scope_errors
    contract=contract_for(state)
    requested_players={k:v for k,v in contract['filters'].items() if k not in {'is_positive','period'}}
    records, announcement_records, scopes, hard_errors = [], [], [], []
    valid_observations = []
    for o in successful:
        data = o.get('data', {})
        try:
            validate_bounds(data.get('bounds', {}))
        except ValueError as error:
            hard_errors.append(f"步骤{o.get('step')}的时间范围无效：{error}")
            continue
        if o.get('tool') in {'query_reviews', 'search_game_evidence'} and any(
                data.get('filters', {}).get(k, 'at_review' if k == 'playtime_basis' else None) != v
                for k, v in requested_players.items()):
            continue  # Wrong player scope cannot contribute evidence, even with real quotes.
        if o.get('tool') in {'query_reviews','search_game_evidence'} and scope_errors(contract,data.get('filters',{}),data.get('bounds',{})):
            continue
        if o.get('tool')=='compare_cohorts' and any(scope_errors(contract,data.get(k,{}).get('filters',{}),data.get(k,{}).get('bounds',{})) for k in ('cohort_a','cohort_b')):
            continue
        if o.get('tool') == 'get_review' and requested_players:
            r = data.get('review', {})
            minutes = r.get('playtime_minutes' if requested_players.get('playtime_basis') == 'lifetime' else 'playtime_at_review_minutes')
            if (requested_players.get('language') and r.get('language') != requested_players['language'] or
                    requested_players.get('min_playtime_hours') is not None and (minutes is None or minutes < requested_players['min_playtime_hours'] * 60) or
                    requested_players.get('max_playtime_hours') is not None and (minutes is None or minutes > requested_players['max_playtime_hours'] * 60)):
                continue
        valid_observations.append(o)
        scopes.append({'step': o.get('step'), 'tool': o.get('tool'),
                       'patch_news_id': data.get('patch_news_id'), 'bounds': data.get('bounds', {}),
                       'filters': data.get('filters', {}),
                       'coverage': data.get('coverage', data.get('review_coverage', {}))})
        for r in data.get('reviews', []) + ([data['review']] if 'review' in data else []):
            if r.get('appid', state.get('appid')) == state.get('appid'):
                records.append((o, r))
        for n in data.get('announcements', []):
            if n.get('appid', state.get('appid')) == state.get('appid') and n.get('contents'):
                announcement_records.append((o, n))
    checks = []

    def add(name, status, steps=(), detail=''):
        checks.append({'requirement': name, 'status': status,
                       'source_steps': sorted(set(s for s in steps if s is not None)), 'detail': detail})

    invalid_requested_date = False
    try:
        target_date = requested_patch_date(goal)
    except ValueError as error:
        target_date = None
        invalid_requested_date = True
        hard_errors.append('用户指定的更新日期无效：' + str(error))
    ids = sorted(set(re.findall(r'(?<![A-Za-z0-9])p\d+(?![A-Za-z0-9])', goal, re.I)))
    missing_discussion_date = discussion_goal and bool(re.search(r'YYYY-MM-DD|某日期', goal, re.I))
    target_needed = enabled and bool(missing_discussion_date or target_date or ids or re.search(r'最近一次更新|最新(?:版本|补丁|更新)|(?:更新|版本|补丁)[前后]', goal))
    lookups = [o for o in valid_observations if o.get('tool') in ({'list_announcements'} if discussion_goal else {'list_patches', 'list_announcements'})]
    if target_date:
        lookups = [o for o in lookups if o.get('data', {}).get('requested_date') == target_date]
    from gamelens.tools.time_scope import latest_patch_requested, located_latest_patch
    latest_target=latest_patch_requested(goal)
    if latest_target:
        lookups=[o for o in lookups if o.get('data',{}).get('latest_lookup')]
    ambiguous = missing_discussion_date or any(o.get('data', {}).get('needs_date_clarification') or o.get('data', {}).get('needs_year_clarification') or
        o.get('data', {}).get('needs_announcement_clarification') for o in lookups[-1:])
    requested_count = required_patch_count(goal)
    if target_needed and requested_count == 1:
        found = [o for o in lookups if o.get('data', {}).get('lookup_status') == 'found_in_cache']
        bound = [o for o in valid_observations if o.get('data', {}).get('patch_news_id')
                 and (not ids or o['data']['patch_news_id'].lower() in {v.lower() for v in ids})]
        located = not ambiguous and (bool(found) if target_date else bool(found or bound))
        if latest_target:
            located=bool(located_latest_patch(state))
        add('target_update', 'satisfied' if located else 'ambiguous_target' if ambiguous else 'target_not_found',
            [o.get('step') for o in found + bound], '必须定位实际更新，不能只使用会话默认补丁')
    if enabled and requested_count > 1:
        multi = [o for o in valid_observations if o.get('tool') in {'compare_versions', 'investigate_recent_versions'}]
        covered = any(len(o['data'].get('versions', [])) >= requested_count for o in multi)
        add('version_coverage', 'satisfied' if covered else 'insufficient_evidence',
            [o.get('step') for o in multi], f'要求{requested_count}个更新，单版本结果不满足该要求')
    # ID lookup does not itself apply a window. Do not let a genuine old quote
    # answer an explicitly post-update question just because a separate count
    # was scoped correctly. Infer a pivot only from actual returned metadata.
    from gamelens.tools.time_scope import requested_directions
    directions=requested_directions(goal)
    after='after' in directions
    before='before' in directions
    pivot = None
    if target_needed and requested_count == 1 and after != before:
        # Prefer exact announcement metadata over a query's clamped bounds.
        from datetime import datetime, timezone
        for o in lookups:
            for n in o.get('data', {}).get('patches', []):
                moment = n.get('published_at_timestamp')
                if moment is None or ids and str(n.get('news_id', '')).lower() not in {v.lower() for v in ids}:
                    continue
                day = local_day(moment)
                if target_date and not (day == target_date if len(target_date) == 10 else day[5:] == target_date):
                    continue
                if ids or target_date:
                    pivot = moment
                    break
            if pivot is not None:
                break
        for o in valid_observations:
            if pivot is not None:
                break
            d = o.get('data', {})
            if d.get('patch_news_id') and (not ids or d['patch_news_id'].lower() in {v.lower() for v in ids}):
                direction = d.get('filters', {}).get('period')
                if direction == 'after':
                    pivot = d.get('bounds', {}).get('start_inclusive')
                elif direction == 'before':
                    pivot = d.get('bounds', {}).get('end_exclusive')
                if pivot is not None:
                    break
        if pivot is not None:
            windows = requested_windows(goal)
            days = next((n for d, n in windows if d == ('后' if after else '前')), None)
            start = pivot if after else (shift_calendar_days(pivot, -days) if days else None)
            end = pivot if before else (shift_calendar_days(pivot, days) if days else None)
            records = [(o, r) for o, r in records if r.get('created_at_timestamp') is not None and
                (start is None or r['created_at_timestamp'] >= start) and
                (end is None or r['created_at_timestamp'] < end)]
    for requirement in sorted(requirements):
        relevant_tools = {'statistics': {'query_reviews', 'compare_cohorts', 'compare_versions'},
                          'review_evidence': {'query_reviews', 'get_review', 'search_game_evidence'},
                          'announcement_check': {'list_patches', 'list_announcements', 'prepare_announcement_comments', 'search_game_evidence', 'investigate_recent_versions'},
                          'announcement_comments': {'prepare_announcement_comments', 'query_announcement_comments', 'get_announcement_comment'},
                          'counterexample_check': {'query_reviews', 'search_game_evidence', 'get_review'}}[requirement]
        relevant = [o for o in valid_observations if o.get('tool') in relevant_tools]
        if requirement in {'review_evidence', 'counterexample_check', 'announcement_check'}:
            kind = 'announcement' if requirement == 'announcement_check' else 'review'
            def searched_kind(o):
                if o.get('tool') != 'search_game_evidence':
                    return True
                requested = o.get('data', {}).get('evidence_coverage', {}).get('requested_kinds')
                if requested is not None:
                    return kind in requested
                argument_kind = o.get('effective_arguments', o.get('arguments', {})) or {}
                value = argument_kind.get('kind', 'both')
                return value == 'both' or value == ('announcements' if kind == 'announcement' else 'reviews')
            relevant = [o for o in relevant if searched_kind(o)]
        status, detail, steps = 'not_checked', '该方面尚未得到可核对的调查结果', []
        if relevant:
            status = 'insufficient_evidence'
            if requirement == 'announcement_comments':
                missing_read_citations = False
                readable = [o for o in relevant if o.get('data', {}).get('binding_verified') and
                    (o.get('data', {}).get('comments') or o.get('data', {}).get('coverage', {}).get('complete') and
                     o['data']['coverage'].get('reported_total') == 0)]
                if readable:
                    steps = [o['step'] for o in readable]
                    status, detail = 'satisfied', '已读取独立公告留言；来源关联不保证语义正确'
                    if any(not o['data'].get('coverage', {}).get('complete') for o in readable):
                        status, detail = 'insufficient_coverage', '只采集了该讨论主题的部分留言；不能代表完整讨论区或全部玩家'
                    missing_read_citations = comment_citations is not None and not comment_citations and any(
                        str(c.get('text') or '').strip() for o in readable for c in o['data'].get('comments', []))
                    if missing_read_citations:
                        status, detail = 'insufficient_evidence', '最终反馈分析必须提供该公告留言的逐字引用，不能用商店评价替代'
                elif any(o.get('data', {}).get('coverage', {}).get('status') == 'unavailable' for o in relevant):
                    status, detail = 'source_unavailable', '公告讨论区不可访问或没有受支持的讨论入口；未采集不等于实际0条反馈'
                elif any(o.get('tool') == 'prepare_announcement_comments' and o.get('data',{}).get('binding_verified') and
                         o.get('data',{}).get('collected_count',0) > 0 for o in relevant):
                    status, detail = 'pending_read', '留言正文已缓存，但尚未成功读取：调用query_announcement_comments；不能把未展示当成0条正文或进入总结'
                    read_failures = [o for o in failures if o.get('tool') in {'query_announcement_comments','get_announcement_comment'}]
                    if read_failures:
                        detail += '。最近读取失败：' + str(read_failures[-1].get('summary','核对工具参数'))[:300]
                if relevant[-1].get('data', {}).get('coverage', {}).get('status') == 'unavailable':
                    status, detail = 'source_unavailable', '本轮最后一次读取/刷新未成功，旧留言不能冒充新采集结果；须明确来源限制'
                if requested_players or re.search(r'好评率|推荐率|差评数|只看差评|只看好评', goal):
                    status, detail = 'unsupported_population', '公告留言没有官方语言、推荐投票与游玩时长字段，不能保证这些人群/好差评条件'
                if missing_read_citations and status == 'insufficient_evidence':
                    hard_errors.append(detail)
            elif requirement == 'statistics':
                comparison = directions=={'before','after'}
                valid_statistics = []
                for o in relevant:
                    d = o['data']
                    populations = [d] if 'stats' in d else ([d['cohort_a'], d['cohort_b']] if 'cohort_a' in d else d.get('versions', []))
                    if populations and all(safe_stats(p.get('stats', {})) for p in populations):
                        errors = statistics_alignment(goal, d, state) if o['tool'] == 'query_reviews' else []
                        from gamelens.core.query_contract import statistical_population_errors
                        errors.extend(error for p in populations for error in statistical_population_errors(contract,p.get('filters',{})))
                        if comparison:
                            # Each query covers ONE of the requested alternatives. The
                            # old single-population helper must not apply the first
                            # duration (e.g. before1d) to the other (after5d).
                            errors = [e for e in errors if '时间跨度' not in e and
                                not ('查询补丁' in e and not d.get('patch_news_id'))]
                        if not errors:
                            valid_statistics.append(o)
                if valid_statistics:
                    enough = not comparison
                    if contract.get('allowed_languages'):
                        languages={p.get('filters',{}).get('language') for o in valid_statistics for p in
                            ([o['data']] if 'stats' in o['data'] else [o['data']['cohort_a'],o['data']['cohort_b']] if 'cohort_a' in o['data'] else [])}
                        enough=set(contract['allowed_languages'])<=languages
                    if contract.get('population_alternatives'):
                        from gamelens.core.query_contract import matching_alternatives
                        matches=set().union(*(matching_alternatives(contract,p.get('filters',{})) for o in valid_statistics for p in
                            ([o['data']] if 'stats' in o['data'] else [o['data']['cohort_a'],o['data']['cohort_b']] if 'cohort_a' in o['data'] else [])))
                        enough=set(range(len(contract['population_alternatives'])))<=matches
                    if comparison:
                        moments={n['published_at_timestamp'] for o in lookups for n in o['data'].get('patches',[])
                            if n.get('published_at_timestamp') is not None and
                            (not ids or str(n.get('news_id','')).lower() in {v.lower() for v in ids})}
                        pivot=next(iter(moments)) if len(moments)==1 else None
                        sides=set();windows=set()
                        for o in valid_statistics:
                            d=o['data']
                            for p in ([d] if 'stats' in d else [d['cohort_a'],d['cohort_b']] if 'cohort_a' in d else d.get('versions',[])):
                                b=p.get('bounds',{});left=b.get('start_inclusive');right=b.get('end_exclusive')
                                windows.add((left,right))
                                side=p.get('filters',{}).get('period')
                                if side not in {'before','after'} and pivot is not None:
                                    side='before' if right is not None and right<=pivot else 'after' if left is not None and left>=pivot else None
                                if side in {'before','after'}:sides.add(side)
                        enough = {'before','after'} <= sides and len(windows)>=2
                        if not enough and pivot is None and len(windows)==2:
                            # Older verified absolute-window checkpoints may not
                            # carry locator metadata. Accept only the requested
                            # durations meeting at one shared update boundary.
                            durations=dict(requested_windows(goal))
                            if set(durations)=={'前','后'}:
                                enough=any(a[1] is not None and a[1]==b[0] and a[0] is not None and b[1] is not None and
                                    shift_calendar_days(a[0],durations['前'])==a[1] and
                                    shift_calendar_days(b[0],durations['后'])==b[1]
                                    for a in windows for b in windows if a!=b)
                    if enough:
                        status, steps = 'satisfied', [o['step'] for o in valid_statistics]
            elif requirement == 'review_evidence':
                selected = [(o, r) for o, r in records if o in relevant and r.get('text')]
                if selected:
                    status, steps = 'satisfied', [o['step'] for o, _ in selected]
                    if state.get('investigation_profile') == 'focused' and re.search(r'崩溃|闪退|卡顿|掉帧|性能|bug|故障',goal,re.I):
                        judged = [(o,next((h for h in o.get('data',{}).get('hits',[]) if h.get('kind')=='review' and h.get('source_id')==r.get('review_id')),{})) for o,r in selected]
                        if all(o.get('data',{}).get('screening',{}).get('enabled') and h.get('evidence_role')=='partial' for o,h in judged):
                            status, detail = 'insufficient_evidence', '只有部分相关反馈，不能确认具体故障；补查更直接原文或明确证据不足'
                    # A count or popular sample alone must not stand in for issue evidence.
                    if citations is not None and not any(c.get('review_id') == r.get('review_id') and c.get('source_step') == o.get('step') for c in citations for o, r in selected):
                        status, detail = 'insufficient_evidence', '最终回答未关联本轮玩家原文；统计数量不能代替问题证据'
                elif any(o.get('data', {}).get('returned_count') == 0 for o in relevant):
                    if any(o.get('data',{}).get('screening',{}).get('status')=='degraded' for o in relevant):
                        status,detail='insufficient_evidence','模型候选引文未通过逐字核验；读取被拒绝候选的原文或说明筛选失败，不能当作无相关评论'
                    else:
                        status, detail = 'no_hit', '有效查询未返回原文；不等于问题不存在'
            elif requirement == 'announcement_check':
                selected = [(o, n) for o, n in announcement_records if o in relevant]
                if selected:
                    status, steps = 'satisfied', [o['step'] for o, _ in selected]
                    if news_citations is not None and not any(c.get('news_id') == n.get('news_id') and c.get('source_step') == o.get('step') for c in news_citations for o, n in selected):
                        status, detail = 'insufficient_evidence', '官方核对没有关联本轮公告原文；需引用或明确未确认'
                elif any(o.get('data', {}).get('returned_count') == 0 for o in relevant):
                    status, detail = 'no_hit', '当前公告查询无命中，不代表官方没有修复声明'
            else:
                broad = [o for o in relevant if o.get('tool') != 'get_review' and o.get('data', {}).get('filters', {}).get('is_positive') is None]
                if broad:
                    status, steps = 'satisfied', [o['step'] for o in broad]
                    detail = '已检查不限推荐态度的反馈；仅证明检查范围，不自动证明否定句与反例解释正确'
                    counterexamples = [(o, h) for o in broad for h in o.get('data', {}).get('hits', [])
                        if h.get('kind') == 'review' and h.get('evidence_role') == 'counterexample']
                    if citations is not None and counterexamples and not any(
                            c.get('source_step') == o.get('step') and c.get('review_id') == h.get('source_id')
                            for c in citations for o, h in counterexamples):
                        status, detail = 'insufficient_evidence', '已有反例候选未关联最终引用；需核对并保留相反材料或明确限制'
                elif relevant:
                    detail = '只读取指定原文或只看差评，未检查可包含正常反馈的范围'
            coverage = [c for o in relevant for c in
                ([o['data'].get('coverage_a',{}),o['data'].get('coverage_b',{})] if 'cohort_a' in o.get('data',{}) else
                 [o.get('data',{}).get('coverage',o.get('data',{}).get('review_coverage',{}))])]
            if requirement == 'statistics':
                # A capped body sample does not invalidate an independently
                # checked endpoint denominator for this exact population.
                coverage=[c for c in coverage if not c.get('counts_complete')]
            if requirement != 'announcement_check' and coverage and all(c.get('status') in {'outside_observed_range', 'no_cached_reviews'} for c in coverage):
                status, detail = 'no_data_coverage', '目标时段没有可用缓存；空结果不是该时段实际零反馈'
            elif requirement == 'statistics' and any(c.get('status') in {'outside_observed_range','no_cached_reviews'} for c in coverage):
                status, detail = 'insufficient_coverage', '至少一侧窗口没有可用缓存，不能计算前后好评率变化；缺失侧不是0%'
            elif requirement=='statistics' and any(p.get('statistics_complete_for_scope') is False for o in relevant for p in
                    ([o['data']] if 'stats' in o['data'] else [o['data']['cohort_a'],o['data']['cohort_b']] if 'cohort_a' in o['data'] else [])):
                status,detail='insufficient_coverage','只能统计已获取的符合条件样本，尚未获得请求人群的完整分母；可报告样本结果，不能冒充全量统计'
            elif requirement != 'announcement_check' and not re.search(r'缓存|本地|已采集|已获取', goal) and any(c.get('requested_start_not_observed') for c in coverage):
                status, detail = 'insufficient_coverage', '只能核对已采集部分，目标起点之前的反馈未知'
        elif any(o.get('tool') in relevant_tools for o in failures):
            status, detail = 'tool_failed', '相关工具失败，不能当作无命中'
        if status == 'satisfied' and detail == '该方面尚未得到可核对的调查结果':
            detail = '已取得本轮可核对的结果；这不自动证明语义或因果判断正确'
        add(requirement, status, steps, detail)
    gaps = [c for c in checks if c['status'] != 'satisfied']
    has_sources = any(o.get('data', {}).get('source') for o in successful)
    verified = verified_sources_current(state) if has_sources else bool(
        state.get('artifacts', {}).get('final_report') and state.get('artifacts', {}).get('evidence_assessed'))
    remaining = max(0, state.get('max_steps', 12) - state.get('step_count', 0))
    needs_clarification = bool(contract['issues']) or missing_discussion_date or invalid_requested_date or ambiguous or bool(target_date and len(target_date) == 5 and gaps and any(c['requirement'] == 'target_update' for c in gaps))
    return {'version': 1, 'goal': goal, 'appid': state.get('appid'), 'enabled': enabled,
            'target': {'requested_date': target_date, 'requested_ids': ids,
                       'session_patch_is_default': state.get('patch_news_id'), 'requested_count': requested_count},
            'required_evidence': sorted(requirements), 'checks': checks, 'gaps': gaps,
            'requested_player_conditions': requested_players,
            'requested_windows': [{'direction': d, 'days': n} for d, n in requested_windows(goal)],
            'subquestions': [{'aspect': r, 'status': next(c['status'] for c in checks if c['requirement'] == r)} for r in sorted(requirements)],
            'effective_scopes': scopes[-12:], 'verified_sources_current': verified,
            'can_continue': remaining > 0, 'remaining_steps': remaining,
            'budget_exhausted': remaining == 0, 'needs_clarification': needs_clarification,
            'hard_errors': hard_errors,
            'invalid_requested_date': invalid_requested_date,
            'query_contract':contract,
            'can_finalize': not gaps and not hard_errors and verified, 'can_report_limited': not hard_errors and verified,
            'outcome': 'needs_clarification' if needs_clarification else 'budget_exhausted' if gaps and remaining == 0
                else 'partial_result' if gaps and scope_limited and verified else 'insufficient_evidence' if gaps else 'complete' if verified else 'needs_verification',
            'semantic_accuracy_verified': False}


def progress_finish_errors(progress, scope_limited):
    hard_errors = progress.get('hard_errors', [])
    if not progress.get('enabled') or scope_limited:
        return hard_errors
    return hard_errors + ['调查方面未满足：' + g['requirement'] + '（' + g['status'] + '）；' + g['detail']
            for g in progress['gaps']]


def progress_limitation(progress):
    if not progress.get('enabled') or not progress.get('gaps'):
        return ''
    labels = {'target_update': '目标更新定位', 'version_coverage': '多版本覆盖', 'statistics': '精确统计',
              'review_evidence': '玩家问题证据', 'announcement_check': '官方公告核对', 'counterexample_check': '反例检查',
              'announcement_comments': '公告讨论区留言'}
    summary = '\n'.join('- **' + labels.get(g['requirement'], g['requirement']) + '**：' + g['detail'] for g in progress['gaps'])
    return '\n\n**调查范围未完整满足**\n\n' + summary + '\n\n以上仅为已核对资料的有限结果，未知不等于不存在。' + (
        '工具步数预算已用尽，上述方面尚未完成。' if progress.get('budget_exhausted') else '') + (
        '请确认目标更新年份或具体公告ID。' if progress.get('needs_clarification') else '')
