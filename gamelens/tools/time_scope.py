"""Date binding and window checks shared by statistics and evidence retrieval."""
import math
import re
import hashlib
from pathlib import Path
from datetime import date, datetime, timezone
from gamelens.core.local_time import local_day, shift_calendar_days


def normalize_patch_date(value):
    """Accept an explicit ISO day or a month/day, never invent a year."""
    if not re.fullmatch(r"(?:\d{4}-)?\d{2}-\d{2}", value):
        raise ValueError("公告日期须为YYYY-MM-DD或MM-DD；缺少年份时不能擅自补当前年")
    date.fromisoformat(value if len(value) == 10 else "2000-" + value)
    return value


def requested_patch_date(goal):
    if not re.search(r"更新|版本|补丁|活动|公告", goal):
        return None
    dates = re.findall(r"(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)", goal)
    if dates:
        values = {f"{int(y):04d}-{int(m):02d}-{int(d):02d}" for y, m, d in dates}
    else:
        dates = re.findall(r"(?:(\d{4})年)?(\d{1,2})月(\d{1,2})(?:日|号)?", goal)
        dates += re.findall(r"(?<![\d.])(?:(\d{4})[./])?(\d{1,2})[./](\d{1,2})(?:日|号)", goal)
        values = {(f"{int(y):04d}-" if y else "") + f"{int(m):02d}-{int(d):02d}" for y, m, d in dates}
    if len(values) != 1:
        return None  # A date range is not a unique announcement identity.
    return normalize_patch_date(values.pop())


def patch_day(patch):
    return local_day(patch.published_at_timestamp)


def date_matches(patch, target):
    actual = patch_day(patch)
    return actual == target if len(target) == 10 else actual[5:] == target


def validate_bounds(bounds):
    start, end = bounds.get("start_inclusive"), bounds.get("end_exclusive")
    for value in (start, end):
        if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise ValueError("查询时间边界无效")
    if start is not None and end is not None and start >= end:
        raise ValueError("最终查询时间区间无效：开始必须早于结束；绝对日期与补丁窗口无交集，不能当作0条评论")


def requested_windows(goal):
    windows=[]
    both = list(re.finditer(r'(?:更新|补丁|版本)?前后\s*(?:各|分别)?\s*([一二三四五六七八九十]|\d+)\s*天',goal))
    for match in both:
        value=match[1]
        days=int(value) if value.isdigit() else '一二三四五六七八九十'.index(value)+1
        windows.extend([('前',days),('后',days)])
    for direction,value in re.findall(r"(?:更新|补丁|版本)?([前后])\s*([一二三四五六七八九十]|\d+)\s*天",goal):
        days=int(value) if value.isdigit() else "一二三四五六七八九十".index(value)+1
        windows.append((direction,days))
    return list(dict.fromkeys(windows))


def requested_directions(goal):
    """Treat 前后 as two populations, never as the substring 更新前 alone."""
    directions={direction for direction,_ in requested_windows(goal)}
    prefix=r'(?:更新|版本|补丁|p\d+)'
    gap=r'\s*(?:的\s*)?(?:之)?'
    for label in ('前','后'):
        if re.search(prefix+gap+label,goal,re.I):
            directions.add(label)
    if re.search(prefix+gap+r'(?:前\s*(?:与|和|及|、|/)?\s*(?:之)?后|后\s*(?:与|和|及|、|/)?\s*(?:之)?前)',goal,re.I):
        directions.update({'前','后'})
    return {'before' if d=='前' else 'after' for d in directions}


def located_dated_patch(state):
    """One current exact-date locator may bind omissions; defaults never may."""
    target=requested_patch_date(state.get('goal',''))
    if not target:return None
    from gamelens.core.source_freshness import active_observations
    from gamelens.core.local_time import timezone_signature
    lookups=[o for o in active_observations(state.get('observations',[]))
        if o.get('ok') and o.get('tool')=='list_patches' and o.get('data',{}).get('requested_date')==target]
    if not lookups:return None
    latest=lookups[-1];data=latest['data']
    args=latest.get('effective_arguments') or latest.get('arguments',{})
    records=data.get('patches',[])
    if (data.get('lookup_status')!='found_in_cache' or data.get('matched_count')!=1 or len(records)!=1 or
            data.get('needs_year_clarification') or args.get('news_id') or args.get('title_query')):
        return None
    source=data.get('source',{})
    if not source.get('path') or source.get('time_zone') and source['time_zone']!=timezone_signature():return None
    try:
        for item in [source,*source.get('reference_files',[])]:
            if hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()!=item['sha256']:return None
    except (OSError,KeyError,TypeError):return None
    record=records[0]
    if record.get('appid')!=state.get('appid') or type(record.get('published_at_timestamp')) is not int:return None
    day=local_day(record['published_at_timestamp'])
    if (day!=target if len(target)==10 else day[5:]!=target):return None
    return {k:record[k] for k in ('news_id','published_at_timestamp','title') if k in record} | {'source_step':latest['step'],'published_on_local':day}


def latest_patch_requested(goal):
    """A relative latest-version request never means the current calendar day."""
    relative=re.search(r'(?:最近|最新)(?:的|一(?:次|个))?\s*(?:更新|补丁|版本)|latest\s+(?:update|patch|version)', goal, re.I)
    if not relative:return False
    try:
        return requested_patch_date(goal) is None
    except ValueError:
        return False  # The existing date validator reports invalid/ambiguous dates.


def patch_acquisition_allowed(state):
    from gamelens.tools.review_coverage import cache_only
    return (state.get('allow_patch_acquisition', True) and state.get('allow_review_acquisition', True)
            and not cache_only(state.get('goal', '')))


def located_latest_patch(state):
    if not latest_patch_requested(state.get('goal', '')):
        return None
    from gamelens.core.source_freshness import active_observations
    from gamelens.core.local_time import timezone_signature
    lookups=[o for o in active_observations(state.get('observations', []))
        if o.get('ok') and o.get('tool')=='list_patches' and o.get('data', {}).get('latest_lookup')]
    if not lookups:
        return None
    latest=lookups[-1]; data=latest['data']; source=data.get('source', {})
    records=data.get('patches', [])
    if not records or data.get('lookup_status')!='found_in_cache':
        return None
    if patch_acquisition_allowed(state) and data.get('latest_check', {}).get('status')!='checked':
        return None
    try:
        if source.get('time_zone') and source['time_zone']!=timezone_signature():return None
        if hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()!=source['sha256']:return None
    except (KeyError, OSError, TypeError):
        return None
    item=records[0]
    if item.get('appid')!=state.get('appid') or type(item.get('published_at_timestamp')) is not int:return None
    if patch_acquisition_allowed(state) and item.get('news_id')!=data.get('latest_check',{}).get('latest_patch_id'):return None
    return {k:item[k] for k in ('news_id','published_at_timestamp','title')} | {
        'source_step':latest['step'], 'published_on_local':local_day(item['published_at_timestamp']),
        'latest_check':data.get('latest_check', {})}


def selection_days(args, state):
    # A dated 'after the update' request is open-ended unless the user bounded it.
    if (requested_patch_date(state.get("goal", "")) and not requested_windows(state["goal"]) and
            requested_directions(state['goal']) != {'before','after'}):
        return None
    return args.window_days or state["window_days"]


def validate_dated_request(domain, args, state, populations):
    target = requested_patch_date(state.get("goal", ""))
    if not target:
        return
    if not args.patch_news_id:
        raise ValueError(f"问题指定{target}更新：先list_patches(published_on='{target}')定位历史公告，再显式传patch_news_id；不能回退最新补丁")
    patch = domain.patch(args.appid, state, args.patch_news_id)
    if not date_matches(patch, target):
        raise ValueError(f"所选补丁日期{patch_day(patch)}与用户指定{target}不一致（北京时间日期）")
    if len(target) == 5:
        matches = domain.recent_patches(args.appid, state, 100000)
        years = {patch_day(p)[:4] for p in matches if date_matches(p, target)}
        if len(years) > 1:
            raise ValueError(f"{target}存在多个年份的更新，须先向用户确认年份，不能自行选当前年")
    windows=requested_windows(state["goal"])
    days={n for _,n in windows}
    if len(days)==1 and any(f.period!="all" for f in populations) and (args.window_days or state["window_days"]) != next(iter(days)):
        raise ValueError("查询窗口天数与问题要求不一致，请显式传window_days")
    directions=requested_directions(state['goal'])
    after='after' in directions
    before='before' in directions
    direction=("after" if after else "before") if after!=before else None
    for filters in populations:
        if direction and filters.period != direction and not (windows and filters.period=="all" and filters.start_at and filters.end_at):
            raise ValueError("日期指定更新的前后方向不一致；请使用明确before/after并绑定目标补丁")
        if not windows and direction == "after" and (filters.end_at is not None or args.window_days is not None):
            raise ValueError("用户只说该日期更新后，未指定截止日；不能擅自截断成7天，使用end_at=null、window_days=null或先澄清")
        if not windows and direction == "after" and filters.start_at and filters.start_at.timestamp() > patch.published_at_timestamp:
            raise ValueError("查询起点晚于目标更新发布时间，不能静默缩小用户要求的更新后范围")
        if windows:
            _,bounds=domain.select([],filters,patch,args.window_days or state["window_days"])
            expected=[(shift_calendar_days(patch.published_at_timestamp,-n),patch.published_at_timestamp) if d=="前" else (
                patch.published_at_timestamp,shift_calendar_days(patch.published_at_timestamp,n)) for d,n in windows]
            if (bounds["start_inclusive"],bounds["end_exclusive"]) not in expected:
                raise ValueError("查询绝对日期与目标更新的指定天数窗口不一致")
        elif directions=={'before','after'} and filters.period in {'before','after'}:
            # Midnight boundaries proposed by a model must not remove the
            # morning of the actual release or silently shorten a default day.
            _,bounds=domain.select([],filters,patch,args.window_days or state['window_days'])
            pivot=patch.published_at_timestamp;days=args.window_days or state['window_days']
            expected=(shift_calendar_days(pivot,-days),pivot) if filters.period=='before' else (pivot,shift_calendar_days(pivot,days))
            if (bounds['start_inclusive'],bounds['end_exclusive'])!=expected:
                raise ValueError('更新前后窗口须以实际发布时间为分界，不能用日历零点截断；去掉额外绝对日期或使用精确发布时间')
    if directions=={'before','after'} and len(populations)==2:
        sides=set()
        for filters in populations:
            if filters.period in {'before','after'}:sides.add(filters.period)
            elif filters.end_at and filters.end_at.timestamp()<=patch.published_at_timestamp:sides.add('before')
            elif filters.start_at and filters.start_at.timestamp()>=patch.published_at_timestamp:sides.add('after')
        if sides!={'before','after'}:
            raise ValueError('更新前后对照必须包含更新前、更新后两组不同窗口，不能以两组同侧数据替代')
