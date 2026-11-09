"""Small audit projections over existing decisions and observations, not a second log store."""
import hashlib
import json
import re

SENSITIVE = ('api_key', 'apikey', 'authorization', 'password', 'secret', 'token',
             'email', 'phone', 'address', 'author', 'steamid')


def safe_audit(value):
    """Bound/redact new audit fields. Never changes original evidence or effective execution."""
    if isinstance(value, dict):
        return {str(k): '[REDACTED]' if any(s in str(k).lower().replace('-', '_') for s in SENSITIVE)
                else safe_audit(v) for k, v in list(value.items())[:40]}
    if isinstance(value, (list, tuple)):
        return [safe_audit(v) for v in value[:40]]
    if isinstance(value, str):
        value = re.sub(r'\bsk-[A-Za-z0-9_.-]+', '[REDACTED]', value)
        value = re.sub(r'(?i)bearer\s+\S+', 'Bearer [REDACTED]', value)
        value = re.sub(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}', '[REDACTED]', value)
        value = re.sub(r'(?i)((?:api[_-]?key|password|secret|token)\s*[=:]\s*)[^\s,;]+', r'\1[REDACTED]', value)
        return value[:1000]
    return value if value is None or isinstance(value, (bool, int, float)) else safe_audit(str(value))


def evidence_ids(data):
    ids = {'review:' + str(r['review_id']) for r in data.get('reviews', []) if r.get('review_id')}
    if data.get('review', {}).get('review_id'):
        ids.add('review:' + str(data['review']['review_id']))
    ids.update('announcement:' + str(n['news_id']) for n in data.get('announcements', []) if n.get('news_id'))
    ids.update('announcement_comment:' + str(c['news_id']) + ':' + str(c['comment_id'])
        for c in data.get('comments', []) if c.get('news_id') and c.get('comment_id'))
    return ids


def observation_status(result):
    if not result.ok:
        return 'tool_failed'
    data = result.data
    coverage = data.get('coverage', data.get('review_coverage', {}))
    if coverage.get('status') in {'outside_observed_range', 'no_cached_reviews'}:
        return 'no_data_coverage'
    if data.get('needs_date_clarification') or data.get('needs_year_clarification') or data.get('needs_announcement_clarification'):
        return 'ambiguous_target'
    if coverage.get('status') == 'unavailable':
        return 'source_unavailable'
    if data.get('lookup_status') == 'not_found_in_cache':
        return 'target_not_found'
    if data.get('status') in {'collected', 'reused'} and data.get('binding_verified'):
        return 'succeeded'
    if data.get('returned_count') == 0 or data.get('stats', {}).get('count') == 0:
        return 'no_match'
    return 'succeeded'


def append_trace(state, event):
    """Stable event IDs and replacement on replay; step numbers remain the tool identity."""
    event = safe_audit(event)
    identity = {k: event.get(k) for k in ('event', 'action', 'step', 'next_decision', 'approval')}
    identity.update(thread_id=state.get('thread_id'),
                    validation_retries=state.get('validation_retries', 0),
                    generation_recoveries=state.get('generation_recoveries', 0))
    event['event_id'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    rows = list(state.get('decision_trace', []))
    index = next((i for i, r in enumerate(rows) if r.get('event_id') == event['event_id']), None)
    if index is None:
        rows.append(event)
    else:
        rows[index] = event
    return rows


def decision_metrics(state):
    """Counts of observable behavior only; no fabricated quality or unnecessary-call score."""
    observations = [o for o in state.get('observations', []) if o.get('tool') != 'finish']
    trace = state.get('decision_trace', [])
    return {'tool_call_count': len(observations),
            'readonly_reuse_count': sum(o.get('data', {}).get('execution', {}).get('reused_from_step') is not None for o in observations),
            'handler_execution_count': sum(o['data']['execution'].get('attempts', 1)
                for o in observations if o.get('data', {}).get('execution', {}).get('handler_executed') is True)
                if all('handler_executed' in o.get('data', {}).get('execution', {}) or
                       o.get('data', {}).get('error_code') in {'unknown_tool','invalid_arguments','repeated_readonly_call'} or
                       o.get('data', {}).get('approval_denied') for o in observations) else None,
            'no_new_information_warnings': sum(t.get('no_new_information_warning') is True for t in trace),
            'failed_tool_call_count': sum(o.get('ok') is False for o in observations),
            'approval_denied_count': sum(o.get('data', {}).get('approval_denied') is True for o in observations),
            'invalid_tool_call_count': sum(o.get('data', {}).get('error_code') in {'unknown_tool','invalid_arguments'} for o in observations) if trace else None,
            'completion_gate_rejection_count': sum(t.get('event') == 'decision' and t.get('requested_action') == 'finish'
                and t.get('next_decision') in {'repair', 'fail'} for t in trace) if trace else None,
            'repair_count': sum(t.get('event') == 'decision' and t.get('next_decision') == 'repair' for t in trace) if trace else None,
            'trace_event_count': len(trace), 'has_structured_trace': bool(trace),
            'unnecessary_tool_calls': None, 'semantic_success': None,
            'note': '缺少追踪的旧记录不可反推决策依据；调用合理性与最终语义需要独立标注。'}
