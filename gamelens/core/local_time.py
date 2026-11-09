"""Project calendar: Beijing standard time, UTC+08:00, independent of host.

Unix instants and UTC source storage stay unchanged. The private adapter name
is retained for historical fixture injection; production never reads OS zones.
"""
from datetime import date, datetime, timedelta, timezone

TIMEZONE_NAME='Asia/Shanghai'
BEIJING=timezone(timedelta(hours=8),name='Asia/Shanghai')


def _system_localize(moment):
    return moment.replace(tzinfo=BEIJING) if moment.tzinfo is None else moment.astimezone(BEIJING)


def localize_naive(moment):
    candidates = []
    for fold in (0, 1):
        converted = _system_localize(moment.replace(fold=fold))
        round_trip = _system_localize(converted.astimezone(timezone.utc))
        if round_trip.replace(tzinfo=None) == moment.replace(fold=fold):
            candidates.append(converted)
    if not candidates:
        raise ValueError('该本地时间位于夏令时跳过的时段，请使用有效时间或明确的UTC偏移')
    if len({c.timestamp() for c in candidates}) > 1:
        raise ValueError('该本地时间因夏令时重复，请明确提供UTC偏移以选择真实时刻')
    return candidates[0]


def local_datetime(value):
    if isinstance(value, (int, float)):
        moment = datetime.fromtimestamp(value, timezone.utc)
    elif isinstance(value, str):
        moment = datetime.fromisoformat(value.replace('Z', '+00:00'))
    else:
        moment = value
    if not isinstance(moment, datetime):
        raise TypeError('需要有效的时间戳或日期时间')
    return localize_naive(moment) if moment.tzinfo is None else _system_localize(moment)


def local_iso(value):
    return local_datetime(value).isoformat(timespec='seconds')


def local_day(value):
    return local_datetime(value).date().isoformat()


def local_midnight(day, *, days=0):
    day = date.fromisoformat(day) if isinstance(day, str) else day
    return localize_naive(datetime.combine(day + timedelta(days=days), datetime.min.time()))


def shift_calendar_days(timestamp, days):
    # Beijing calendar days have 24 hours; keep wall-clock time unchanged.
    wall_time = local_datetime(timestamp).replace(tzinfo=None, fold=0)
    return int(localize_naive(wall_time + timedelta(days=days)).timestamp())


def timezone_signature():
    return {'basis':'beijing','system_zone':TIMEZONE_NAME,
            'names':['China Standard Time'],'standard_offset_seconds':28800,
            'daylight_offset_seconds':28800}


def time_context():
    now = local_datetime(datetime.now(timezone.utc))
    return {**timezone_signature(), 'current_offset':now.strftime('%z'),
            'date_rule':'未标注时区的日期/时间按北京时间UTC+8；明确Z/偏移保留真实时刻再转换为北京时间，不随电脑时区变化。',
            'day_rule':'北京时间日历天，固定24小时，不采用夏令时；结束边界不含。'}


def display_view(payload):
    """Add derived local ISO fields for UI only; never mutate stored evidence."""
    if isinstance(payload, list):
        return [display_view(item) for item in payload]
    if not isinstance(payload, dict):
        return payload
    result = {k:display_view(v) for k,v in payload.items()}
    for key in ('started_at','finished_at','updated_at','created_at','published_at','fetched_at'):
        if payload.get(key):
            try:
                result[key+'_local'] = local_iso(payload[key])
            except (ValueError, TypeError, OSError):
                pass
    return result
