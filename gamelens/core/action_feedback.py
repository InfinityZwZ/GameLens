"""Bounded feedback and lightweight plan projections; no model calls or scheduler."""
from gamelens.core.decision_trace import evidence_ids, observation_status, safe_audit
from gamelens.tools.query_reuse import digest

QUERY_ACTIONS = frozenset({'inspect_data', 'list_patches', 'query_reviews',
    'get_review', 'compare_cohorts', 'compare_versions',
    'investigate_recent_versions', 'search_game_evidence', 'list_announcements',
    'query_announcement_comments', 'get_announcement_comment'})


def observation_feedback(result, arguments, known_ids, state=None):
    data = result.data
    roles = {role:[] for role in ('support', 'counterexample', 'partial', 'irrelevant')}
    verdicts = data.get('screening', {}).get('verdicts', [])
    for hit in verdicts or data.get('hits', []):
        role = hit.get('role', hit.get('evidence_role'))
        if role in roles and hit.get('source_id') is not None:
            kind = 'announcement' if hit.get('kind') == 'announcement' else 'review'
            roles[role].append(kind + ':' + str(hit['source_id']))
    scope = {k:data.get(k, arguments.get(k)) for k in
             ('filters', 'bounds', 'patch_news_id', 'window_days', 'kind', 'requested_date',
              'review_id','news_id','published_on','offset')
             if k in data or k in arguments}
    coverage = data.get('review_coverage', data.get('coverage', data.get('patch_coverage', {})))
    if 'cohort_a' in data:
        scope = {**scope, 'cohorts':{k:{f:data[k].get(f) for f in ('filters','bounds','statistics_basis','body_sample_count')}
                                   for k in ('cohort_a','cohort_b')}}
        coverage = {'cohort_a':data.get('coverage_a', {}), 'cohort_b':data.get('coverage_b', {})}
    if 'versions' in data:
        scope['versions'] = [{k:v.get(k) for k in ('news_id','patch_news_id','bounds','filters')}
                             for v in data['versions']]
    missing = data.get('evidence_coverage', {}).get('missing_source_kinds',
                data.get('evidence_coverage', {}).get('missing_kinds', []))
    status = observation_status(result)
    gaps = []
    if 'available' in data:
        if 'reviews' not in data['available']:
            gaps.append('reviews_not_cached')
        if 'patches' not in data['available']:
            gaps.append('announcements_not_cached')
    if status != 'succeeded':
        gaps.append(status)
    if coverage.get('requested_start_not_observed') and not coverage.get('counts_complete'):
        gaps.append('requested_start_not_observed')
    if missing:
        gaps.append('missing_source_kind')
    if data.get('screening', {}).get('enabled') and not roles['counterexample']:
        gaps.append('no_selected_counterexample_not_proof_of_absence')
    if not result.ok:
        guidance = '核对失败类型与有效参数；无命中和工具失败不同，不得据此否定问题。'
    elif status == 'ambiguous_target':
        guidance = '需要用户澄清年份或目标，不能替换成最新补丁。'
    elif data.get('statistics_complete_for_scope') is False:
        guidance = '该人群只有有限原文样本，不是整月/全部符合条件玩家的统计。保留可核验的样本计算并明确有限结果；不要仅因覆盖缺口反复重生成答案或失败。'
    elif coverage.get('counts_complete'):
        guidance = '精确窗口平台计数已核对，原文仅为有限样例；不要把原文样例数当分母。需要问题解释/反例时继续读取或检索这些实际原文，不重复补采同一窗口。'
    elif status == 'no_data_coverage' or coverage.get('requested_start_not_observed'):
        guidance = '缺少目标范围资料；改写问题或增加候选量不会填补时间覆盖。'
    elif missing:
        guidance = '尚缺请求的来源类型；保持目标条件补查或明确范围限制。'
    elif 'available' in data and gaps:
        guidance = '缓存资料缺失；按当前任务需要准备相应来源，不把未采集当作零反馈。'
    elif gaps:
        guidance = '候选不足不等于事实不存在；如需补查，说明具体缺口并保留相反材料。'
    else:
        guidance = '结合任务缺口选择下一步；来源可追溯不代表语义或因果判断正确。'
    identity = data.get('execution', {}).get('reuse', {})
    source_id = identity.get('snapshot_id') or (digest(data['source']) if data.get('source') else None)
    verification = None
    if 'verified_files' in data:
        sources = [o['data']['source'] for o in (state or {}).get('observations', [])
                   if o.get('ok') and o.get('data', {}).get('source')]
        source_id = digest(sources) if sources else None
        verification = {'artifact_count':data['verified_files'],
                        'source_checks_passed':result.ok,
                        'semantic_or_causal_truth_verified':False}
    # Only observable information, excluding query wording/step/file output paths.
    information = {'status':status, 'scope':scope, 'source_snapshot_id':source_id,
        'evidence_ids':sorted(evidence_ids(data)), 'roles':roles, 'coverage':coverage,
        'lookup_status':data.get('lookup_status'), 'missing':missing,
        'facts':{k:data[k] for k in ('stats','cohort_a','cohort_b','versions',
                                  'available','review_count','patch_count','error_code','timed_out') if k in data}}
    # Query tools may include samples in cohorts/versions; still bounded when presented.
    feedback = {'status':status, 'effective_scope':scope, 'coverage':coverage,
        'error_code':data.get('error_code'), 'argument_errors':data.get('argument_errors',[]),
        'goal_id':digest({'goal':state.get('goal'), 'appid':state.get('appid')}) if state else None,
        'source_snapshot_id':source_id,
        'new_evidence_ids':sorted(evidence_ids(data) - known_ids), 'role_ids':roles,
        'candidate_count':data.get('candidate_summary', {}).get('count'),
        'selected_count':data.get('returned_count'),
        'eligible_review_count':data.get('population', {}).get('eligible_review_count'),
        'retrieval_counts':data.get('retrieval', {}), 'missing_source_kinds':missing,
        'verification':verification,
        'gaps':gaps, 'next_step_guidance':guidance,
        'detail_location':'本步原始观察/审计产物；按来源ID读取原文',
        'semantic_accuracy_verified':False,
        'information_fingerprint':digest(information)}
    return safe_audit(feedback)


def loop_signals(state):
    seen, streak, total = set(), 0, 0
    goal_id = digest({'goal':state.get('goal'), 'appid':state.get('appid')})
    for row in state.get('observations', []):
        if row.get('tool') not in QUERY_ACTIONS:
            continue
        fingerprint = row.get('feedback', {}).get('information_fingerprint')
        if row.get('feedback', {}).get('goal_id') not in (None, goal_id):
            continue
        if not fingerprint:
            continue  # Old records cannot be retroactively judged equivalent.
        same = fingerprint in seen and not row['feedback'].get('new_evidence_ids')
        streak = streak + 1 if same else 0
        total += int(same)
        seen.add(fingerprint)
    return {'consecutive_no_new_information':streak,
            'repeated_information_observations':total,
            'warning':streak >= 2,
            'guidance':('连续查询未增加可观察信息：不要仅改写措辞；说明新的具体缺口、澄清或核验后输出有限结果。'
                        if streak >= 2 else ''),
            'semantic_completion_inferred':False}


def project_plan(state, progress, previous=None):
    """Track supplied high-level goals against program checks, not model self-ratings."""
    previous = state.get('plan_state', {}) if previous is None else previous
    goal_id = digest({'goal':state.get('goal'), 'appid':state.get('appid')})
    plan = list(state.get('plan', []))
    plan_id = digest(plan)
    changed = bool(previous or plan) and (previous.get('goal_id') != goal_id or previous.get('plan_id') != plan_id)
    checks = {c['requirement']:c for c in progress.get('checks', [])}
    requirement_steps = []
    for key in sorted(checks):
        check = checks[key]
        requirement_steps.append({
            'requirement_id':'req:'+goal_id[:12]+':'+key,
            'requirement':key,
            'status':check['status'],
            'source_steps':list(check.get('source_steps', [])),
        })
    requirement_structure_id = digest([row['requirement_id'] for row in requirement_steps])
    mapping = {'target_update':('定位','目标更新','版本范围'),
        'version_coverage':('逐版本','多版本','版本覆盖'),
        'statistics':('统计','推荐率','群体比较'),
        'review_evidence':('反馈','玩家','评论'),
        'announcement_check':('公告','官方','补丁说明'),
        'announcement_comments':('留言','讨论区','公告反馈'),
        'counterexample_check':('反例','相反','否定')}
    items = []
    for index, label in enumerate(plan[:12]):
        matched = [key for key, terms in mapping.items() if key in checks and any(t in label for t in terms)]
        verification = any(t in label for t in ('核验','核对来源','验证来源'))
        status = 'context_only'
        if matched or verification:
            status = 'covered' if (all(checks[k]['status'] == 'satisfied' for k in matched) and
                                  (not verification or progress.get('verified_sources_current'))) else 'pending'
        steps = sorted({s for k in matched for s in checks[k].get('source_steps', [])})
        items.append({'index':index+1, 'requirements':matched, 'status':status, 'source_steps':steps})
    latest = state.get('observations', [])[-1:] or [{}]
    feedback = latest[0].get('feedback', {})
    triggers = list(feedback.get('gaps', []))
    if feedback.get('new_evidence_ids'):
        triggers.append('new_sources')
    if feedback.get('role_ids', {}).get('counterexample'):
        triggers.append('counterexample_candidate_requires_review')
    if state.get('validation_errors'):
        triggers.append('validation_feedback')
    prior_checks = previous.get('check_statuses', {})
    if prior_checks and prior_checks != {k:c['status'] for k,c in checks.items()}:
        triggers.append('requirement_status_changed')
    return {'goal_id':goal_id, 'plan_id':plan_id,
            'requirement_structure_id':requirement_structure_id,
            'requirement_steps':requirement_steps,
            'revision':previous.get('revision', 0)+int(changed),
            'items':items, 'adjustment_signals':sorted(set(triggers)),
            'check_statuses':{k:c['status'] for k,c in checks.items()},
            'semantic_completion_inferred':False}
