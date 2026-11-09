"""Per-turn wall time, separate from overlapping model/tool durations."""
from datetime import datetime, timezone


def instant(value):
    if not isinstance(value,str):
        return None
    try:
        result = datetime.fromisoformat(value.replace('Z','+00:00'))
        return result if result.tzinfo else None
    except ValueError:
        return None


def turn_timing(payload, *, now=None, telemetry=None):
    start = instant(payload.get('started_at'))
    terminal = payload.get('status') not in {'queued','running','cancelling','awaiting_approval'}
    end = instant(payload.get('finished_at')) if terminal else (now or datetime.now(timezone.utc))
    if terminal and end is None:
        end = instant(payload.get('updated_at'))
    duration = max(0,round((end-start).total_seconds()*1000)) if start and end else None
    metric = telemetry or {}
    tools = payload.get('events',[])
    return {'elapsed_ms':duration, 'started_at':payload.get('started_at'),
            'finished_at':payload.get('finished_at'), 'is_running':not terminal,
            'model_request_ms':metric.get('latency_ms'),
            'tool_ms':sum(o.get('duration_ms',o.get('data',{}).get('execution',{}).get('elapsed_ms',0)) or 0 for o in tools),
            'http_requests':metric.get('http_requests'), 'retries':metric.get('retries'),
            'method':'wall_clock',
            'note':'总用时含排队、审批等待和恢复间隔；工具可能包含模型请求，分项不能相加。'}


def model_status(events, active):
    requests = [e for e in events if e.get('kind') == 'request']
    latest = requests[-1] if requests else {}
    failed = next((e for e in reversed(events) if e.get('status') == 'failed'),{})
    return {'pending':bool(active and latest.get('status') in {'pending','streaming'}),
            'streaming':bool(active and latest.get('status')=='streaming'),
            'response_progress':{k:latest.get(k) for k in ('first_byte_ms','first_content_ms','content_chars','elapsed_ms')},
            'last_error':{k:failed.get(k) for k in ('error_type','error_code','provider_error_code','http_status','error_detail','latency_ms')} if failed else None,
            'latest_status':latest.get('status'), 'retrying':bool(active and latest.get('retry'))}
