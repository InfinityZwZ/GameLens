"""Shared, conservative interpretation of explicit review query conditions.

This is not a general NLU classifier. The planner still chooses actions and
semantic queries. Unambiguous user constraints are filled once, checked at every
boundary and never changed to make a result look complete.
"""
import copy
import re
import json
from datetime import date, datetime, timezone, timedelta

from gamelens.core.local_time import local_datetime, local_iso, local_midnight, shift_calendar_days, timezone_signature

NUMBER=r'(?:\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百]+)'
UNIT=r'(?:小时|小時|hours?|hrs?|h)(?![a-z])'
CONTRACT_VERSION=2


def requested_metrics(goal):
    result=[]
    if re.search(r'好评率|推荐率|好评(?:占比|比例)|推荐(?:占比|比例)',goal):result.append('positive_rate')
    if re.search(r'差评率|差评(?:占比|比例)',goal):result.append('negative_rate')
    if re.search(r'差评(?:数|数量)|(?:多少|几)条差评',goal):result.append('negative_count')
    if re.search(r'好评(?:数|数量)|(?:多少|几)条好评',goal):result.append('positive_count')
    if re.search(r'评论数量|评论数|多少条|数量与|评价数量',goal):result.append('count')
    return result


def number(value):
    if re.fullmatch(r'\d+(?:\.\d+)?',value):return float(value)
    digits={'零':0,'〇':0,'一':1,'二':2,'两':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9}
    total=current=0
    for char in value:
        if char in digits:current=digits[char]
        elif char in {'十','百'}:total+=(current or 1)*({'十':10,'百':100}[char]);current=0
        else:raise ValueError('无法识别数字')
    return float(total+current)


def _explicit_playtime_basis(goal):
    if re.search(r'当前累计|现在累计|总游玩|累计游玩|当前时长|总时长',goal):return 'lifetime'
    if re.search(r'评价(?:发表)?时|发表时|评论时|发布时',goal):return 'at_review'
    return None


def _hour_conditions(goal):
    minimum=[];maximum=[]
    for lo,hi in re.findall('('+NUMBER+r')\s*(?:-|—|~|～|到|至)\s*('+NUMBER+r')\s*'+UNIT,goal,re.I):
        minimum.append((number(lo),False));maximum.append((number(hi),False))
    expressions=[('min',False,r'(?:至少|不少于|不低于|不小于|>=|≥)\s*('+NUMBER+r')\s*'+UNIT),
        ('min',True,r'(?:超过|大于|>)\s*('+NUMBER+r')\s*'+UNIT),
        ('max',False,r'(?:至多|不超过|不多于|不大于|<=|≤)\s*('+NUMBER+r')\s*'+UNIT),
        ('max',True,r'(?:少于|小于|低于|<)\s*('+NUMBER+r')\s*'+UNIT),
        ('min',False,'('+NUMBER+r')\s*'+UNIT+r'\s*(?:及以上|以上|起|\+)'),
        ('max',False,'('+NUMBER+r')\s*'+UNIT+r'\s*(?:及以下|以下|以内|之内)'),
        ('min',False,'('+NUMBER+r')\s*\+\s*'+UNIT)]
    for direction,strict,pattern in expressions:
        for match in re.finditer(pattern,goal,re.I):
            # >= / <= must not be interpreted again as > / <.
            if strict and match.start()>0 and goal[match.start()-1] in '<>不':continue
            (minimum if direction=='min' else maximum).append((number(match[1]),strict))
    result={}
    if minimum:
        lo=max(minimum);result['min_playtime_hours']=lo[0]
        if lo[1]:result['min_playtime_exclusive']=True
    if maximum:
        hi=min(maximum,key=lambda pair:(pair[0],not pair[1]));result['max_playtime_hours']=hi[0]
        if hi[1]:result['max_playtime_exclusive']=True
    if re.search(r'游玩|时长|小[时時]|\d\s*(?:h|hours?|hrs?)\b',goal,re.I) and (minimum or maximum or re.search(r'时长|小[时時]',goal)):
        result['playtime_basis']=_explicit_playtime_basis(goal) or 'at_review'
    return result


def _calendar_window(goal,now):
    from gamelens.tools.time_scope import requested_patch_date
    # Parse exact ISO instants before date-only/month/patch interpretation.
    # Partial or ambiguous timestamps must never fall through to midnight.
    if re.search(r'\d{4}-\d{2}-\d{2}(?:[Tt]| \d)',goal):
        stamps=list(re.finditer(
            r'(?<!\d)\d{4}-\d{2}-\d{2}[Tt ]\d{2}:\d{2}'
            r'(?::\d{2}(?:\.\d{1,6})?)?(?:Z|[+-](?:[01]\d|2[0-3]):[0-5]\d)?'
            r'(?![0-9A-Za-z_:+-]|\.\d)',goal))
        dates=re.findall(r'(?<!\d)\d{4}[-/]\d{1,2}[-/]\d{1,2}(?!\d)',goal)
        if len(stamps)!=2 or len(dates)!=2:
            return {},['精确时间区间须提供完整且唯一的ISO起点和终点，不能省略或降为整日']
        if not re.fullmatch(r'\s*(?:至|到|~|～|—|–|-|to)\s*',goal[stamps[0].end():stamps[1].start()],re.I):
            return {},['精确时间区间关系不明确，请明确起点至终点']
        try:
            start,end=(local_datetime(m[0].replace('t','T')) for m in stamps)
        except ValueError:
            return {},['精确时间或时区偏移无效，请确认完整ISO时间']
        if end.timestamp()<=start.timestamp():
            return {},['结束时刻必须晚于开始时刻']
        return {'start_at':start.isoformat(),'end_at':end.isoformat(),
            'period':'all','kind':'exact_iso_range'},[]
    # A single dated update is an announcement identity, not a midnight cutoff.
    if requested_patch_date(goal):return {},[]
    matches=list(re.finditer(r'(?<!\d)(\d{4})年\s*([一二三四五六七八九十零〇两\d]+)月(?:份)?(?!\s*\d+[日号])|(?<!\d)(\d{4})[-/](\d{1,2})(?![-/\d])',goal))
    values=[]
    for m in matches:
        if m[1]:values.append((int(m[1]),int(number(m[2]))))
        else:values.append((int(m[3]),int(m[4])))
    if len(set(values))>1:return {},['多个不同月份不能合并为单个人群，请说明分别比较还是连续区间']
    if values:
        year,month=values[0]
        start=local_midnight(date(year,month,1))
        end=local_midnight(date(year+1,1,1) if month==12 else date(year,month+1,1))
        requested_end=end
        if start<now<end:end=now
        return {'start_at':local_iso(start),'end_at':local_iso(end),
            'requested_end_at':local_iso(requested_end),'period':'all','kind':'calendar_month',
            'as_of_now':end!=requested_end},[]
    iso=list(re.finditer(r'(?<!\d)(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?!\d)',goal))
    chinese=list(re.finditer(r'(?:(\d{4})年)?(\d{1,2})月(\d{1,2})[日号]',goal))
    dates=iso or chinese
    if len(dates)>2:return {},['多个日期不能自动合并，请明确一个起止区间或分别比较']
    if len(dates) in {1,2}:
        if not dates[0][1]:return {},['日期没有明确年份，请补充年份，避免查询错误的历史区间']
        year=int(dates[0][1])
        days=[date(int(m[1]) if m[1] else year,int(m[2]),int(m[3])) for m in dates]
        if any(re.match(r'\s*(?:T\d|\d{1,2}(?:[:：]\d|点|时))',goal[m.end():]) for m in dates):
            return {},['问题包含精确时刻，请用JSON的start_at/end_at指定完整时间；不会自动忽略时分']
        if len(days)==1:
            start=local_midnight(days[0]);end=local_midnight(days[0],days=1)
        elif re.search(r'到|至|~|～|—|-',goal[dates[0].end():dates[1].start()]):
            start=local_midnight(days[0]);end=local_midnight(days[1],days=0 if re.search(r'不含(?:结束|截止|最后)',goal) else 1)
        else:return {},['多个日期的关系不明确，请说明起止区间或目标更新']
        if end<=start:return {},['结束日期不能早于开始日期']
        return {'start_at':local_iso(start),'end_at':local_iso(end),'period':'all','kind':'calendar_days'},[]
    if re.search(r'(?<!\d)(?:\d{1,2}|[一二三四五六七八九十]+)月(?:份)?(?!\s*\d+[日号])',goal):
        return {},['月份没有明确年份，请补充年份，避免查询错误的历史区间']
    if not re.search(r'更新|补丁|版本',goal):
        if re.search(r'本月|这个月|上月|上个月',goal):
            year,month=now.year,now.month
            if re.search(r'上月|上个月',goal):year,month=(year-1,12) if month==1 else (year,month-1)
            start=local_midnight(date(year,month,1))
            end=local_midnight(date(year+1,1,1) if month==12 else date(year,month+1,1))
            end=min(end,now)
        elif re.search(r'今天|今日|昨天|昨日|前天',goal):
            back=2 if '前天' in goal else 1 if re.search(r'昨天|昨日',goal) else 0
            start=local_midnight(now.date(),days=-back);end=min(local_midnight(now.date(),days=1-back),now)
        elif re.search(r'本周|这周|上周',goal):
            monday=now.date()-timedelta(days=now.weekday()+(7 if '上周' in goal else 0))
            start=local_midnight(monday);end=min(local_midnight(monday,days=7),now)
        else:
            recent=re.search(r'(?:最近|近|过去)\s*('+NUMBER+r')\s*天',goal)
            if not recent:return {},[]
            days=number(recent[1])
            if not days.is_integer() or not 1<=days<=366:return {},['最近天数须为1至366的整数']
            end=now;start=local_datetime(shift_calendar_days(now.timestamp(),-int(days)))
        if end<=start:return {},['当前时间区间尚未开始，请稍后查询或指定历史区间']
        return {'start_at':local_iso(start),'end_at':local_iso(end),'period':'all','kind':'relative_calendar'},[]
    return {},[]


def build_contract(goal,reference_time=None):
    now=local_datetime(reference_time or datetime.now(timezone.utc))
    structured=[];structured_issues=[]
    for match in re.finditer(r'\{[^{}]+\}',goal):
        try:
            item=json.loads(match[0])
            allowed={'period','language','is_positive','min_playtime_hours','max_playtime_hours','playtime_basis',
                     'min_playtime_exclusive','max_playtime_exclusive','start_at','end_at'}
            if isinstance(item,dict) and any(k in item for k in allowed):
                from gamelens.tools.investigation import MetadataReviewFilter,ReviewFilter
                from pydantic import ValidationError
                try:
                    # Legacy literal filters remain valid in extended fixtures
                    # and checkpoints; this contract interprets metadata only.
                    model=ReviewFilter if {'keywords','keyword_mode'} & item.keys() else MetadataReviewFilter
                    typed=model.model_validate(item).model_dump(mode='json')
                    structured.append({k:typed[k] for k in item if k in allowed})
                except ValidationError:
                    structured_issues.append('JSON筛选条件的字段、类型或边界无效，请确认支持的参数；不会忽略无效条件')
        except ValueError:pass
    prose=re.sub(r'\{[^{}]+\}','',goal) if structured or structured_issues else goal
    filters=_hour_conditions(prose);issues=list(structured_issues);alternatives=[]
    if re.search(r'比较|对比|分别|对照',prose) and re.search(r'和|与|vs|、',prose,re.I):
        # Only clearly shared declarations apply to unspecified groups.
        # Explicit per-group bases take precedence over that common default.
        prefix=re.split(r'比较|对比|分别|对照',prose,maxsplit=1)[0]
        shared=re.findall(r'(?:两组)?(?:均|都|统一)\s*(?:按|使用|采用|以)([^，。；]+)',prose)
        basis=_explicit_playtime_basis(' '.join([prefix,*shared])) or 'at_review'
        comparison=re.sub(r'(?:两组)?(?:均|都|统一)\s*(?:按|使用|采用|以)[^，。；]+','',prose[len(prefix):])
        pieces=re.split(r'和|与|vs\.?|、',comparison,flags=re.I)
        for piece in pieces:
            population=_hour_conditions(piece)
            if any(k in population for k in ('min_playtime_hours','max_playtime_hours')):
                population['playtime_basis']=_explicit_playtime_basis(piece) or basis
                alternatives.append(population)
        if len(alternatives)>1:
            # Basis belongs to each population, just like its threshold.
            # A lifetime phrase in one group must not overwrite another group.
            filters={k:v for k,v in filters.items() if k not in {'min_playtime_hours','max_playtime_hours','min_playtime_exclusive','max_playtime_exclusive','playtime_basis'}}
        else:alternatives=[]
    labels=[]
    if re.search(r'繁体中文|繁體中文',prose):labels.append('tchinese')
    elif re.search(r'中文|简体|簡體',prose):labels.append('schinese')
    if re.search(r'英文|英语|英語|english',prose,re.I):labels.append('english')
    if len(labels)==1:filters['language']=labels[0]
    elif len(labels)>1 and not re.search(r'分别|比较|对比',goal):issues.append('多种语言须说明合并统计还是分别比较；不能只查其中一种')
    restricted=re.findall(r'(?:只看|只统计|只查|仅看|仅统计|仅查)\s*(好评|差评)',goal)
    if len(set(restricted))==1:filters['is_positive']=restricted[0]=='好评'
    try:window,calendar_issues=_calendar_window(prose,now);issues.extend(calendar_issues)
    except ValueError:window={};issues.append('日期或月份无效，请确认真实日历日期')
    if structured:
        if len(structured)==1:
            filters.update({k:v for k,v in structured[0].items() if k not in {'start_at','end_at'}})
            if structured[0].get('start_at') or structured[0].get('end_at'):
                window={'period':structured[0].get('period','all'),'kind':'explicit_structured',
                    **{k:local_iso(v) if v else None for k,v in structured[0].items() if k in {'start_at','end_at'}}}
                window.setdefault('start_at',None);window.setdefault('end_at',None)
        else:
            alternatives=[{k:v for k,v in f.items() if k not in {'start_at','end_at'}} for f in structured]
            if any(f.get('start_at') or f.get('end_at') for f in structured):
                issues.append('多个JSON时间人群须显式通过cohort_a/cohort_b参数比较，不能合并为单个时间窗口')
    from gamelens.tools.time_scope import requested_directions, requested_patch_date
    # Generic semantic investigations may deliberately inspect different
    # windows. Only a numerical population request defaults to all-time:
    # default parameters must not masquerade as explicit user restrictions.
    if not any('period' in item for item in structured) and requested_metrics(goal) and not requested_directions(prose) and not requested_patch_date(prose) and not re.search(r'更新|版本|补丁|刚才|上次',prose):
        filters['period']='all'
    low=filters.get('min_playtime_hours',0);high=filters.get('max_playtime_hours')
    if high is not None and (low>high or low==high and (filters.get('min_playtime_exclusive') or filters.get('max_playtime_exclusive'))):
        issues.append('游玩时长上下界没有交集，请确认筛选条件')
    return {'version':CONTRACT_VERSION,'goal':goal,'reference_time':local_iso(now),'time_zone':timezone_signature(),
        'filters':filters,'calendar_window':window,'issues':issues,
        'allowed_languages':labels if len(labels)>1 else [],'requested_metrics':requested_metrics(goal),
        'population_alternatives':alternatives,'has_structured_conditions':bool(structured),
        'explicit_cached_scope':bool(re.search(r'本地|缓存|已采集|已获取',goal))}


def contract_for(state):
    old=state.get('query_contract',{})
    if old.get('version')==CONTRACT_VERSION and old.get('goal')==state.get('goal') and old.get('time_zone')==timezone_signature():return old
    return build_contract(state.get('goal',''),state.get('request_started_at'))


def complete_arguments(action,arguments,state):
    if action not in {'query_reviews','compare_cohorts','search_game_evidence'}:return arguments,[]
    contract=contract_for(state)
    if contract['issues']:return arguments,[]
    result=copy.deepcopy(arguments);changes=[]
    names=('cohort_a','cohort_b') if action=='compare_cohorts' else ('filters',)
    for index,name in enumerate(names):
        if name not in result:result[name]={}
        filters=result[name]
        if not isinstance(filters,dict):continue
        # Comparisons may deliberately vary language/playtime; the completion
        # logic applies only common, explicitly stated conditions.
        expected=contract['filters']
        if action=='compare_cohorts' and len(contract.get('population_alternatives',[]))==2:
            alternatives=contract['population_alternatives']
            defaults={'min_playtime_hours':0,'max_playtime_hours':None,'min_playtime_exclusive':False,'max_playtime_exclusive':False,'playtime_basis':'at_review','language':None,'is_positive':None,'period':'all'}
            keys=set().union(*(set(a) for a in alternatives))
            compatible=[a for a in alternatives if all(filters[k]==a.get(k,defaults.get(k)) for k in keys if k in filters and filters[k] is not None)]
            expected={**expected,**(compatible[0] if len(compatible)==1 else alternatives[index])}
        for key,value in expected.items():
            if key not in filters or filters[key] is None and key in {'language','min_playtime_hours','max_playtime_hours','playtime_basis'}:
                filters[key]=value;changes.append({'field':name+'.'+key,'rule':'explicit_query_contract','effective_value':value})
        window=contract['calendar_window']
        if window:
            for key in ('start_at','end_at','period'):
                if key not in filters or filters[key] is None:
                    filters[key]=window[key];changes.append({'field':name+'.'+key,'rule':'explicit_calendar_window','effective_value':window[key]})
    return result,changes


def scope_errors(contract,filters,bounds=None):
    errors=list(contract['issues'])
    if contract.get('allowed_languages') and filters.get('language') not in contract['allowed_languages']:
        errors.append('分组查询须使用用户要求的语言之一，不能查询其他语言或省略')
    if contract.get('population_alternatives') and not matching_alternatives(contract,filters):
        errors.append('人群条件未匹配用户要求的任何一个比较分组')
    for key,value in contract['filters'].items():
        default='all' if key=='period' else 'at_review' if key=='playtime_basis' else False if key.endswith('_exclusive') else 0 if key=='min_playtime_hours' else None
        if filters.get(key,default)!=value:errors.append('查询条件与用户要求不一致：'+key)
    # A stricter condition not requested by the user is not an equivalent query.
    if not contract.get('population_alternatives'):
        for key in ('min_playtime_exclusive','max_playtime_exclusive'):
            if filters.get(key,False)!=contract['filters'].get(key,False):errors.append('时长边界是否包含与用户要求不一致：'+key)
    window=contract['calendar_window']
    if window:
        expected={k:local_datetime(window[v]).timestamp() if window.get(v) else None for k,v in [('start_inclusive','start_at'),('end_exclusive','end_at')]}
        if bounds is None:
            actual={k:local_datetime(filters[v]).timestamp() if filters.get(v) else None for k,v in [('start_inclusive','start_at'),('end_exclusive','end_at')]}
        else:actual=bounds
        if filters.get('period','all')!=window.get('period','all') or any(actual.get(k)!=v for k,v in expected.items()):
            errors.append('查询月份/日期区间与用户要求不一致，不能省略边界或替换为全部缓存')
    return list(dict.fromkeys(errors))


def matching_alternatives(contract,filters):
    alternatives=contract.get('population_alternatives',[])
    keys=set().union(*(set(a) for a in alternatives)) | {'min_playtime_exclusive','max_playtime_exclusive'}
    defaults={'min_playtime_hours':0,'max_playtime_hours':None,'min_playtime_exclusive':False,
        'max_playtime_exclusive':False,'playtime_basis':'at_review','language':None,'is_positive':None,'period':'all'}
    # A full-population alternative must not also match its narrower subset;
    # compare every distinguishing field, including omitted/default values.
    return {i for i,a in enumerate(contract.get('population_alternatives',[])) if all(
        filters.get(k,defaults.get(k))==a.get(k,defaults.get(k)) for k in keys)}


def cohort_scope_errors(contract,populations):
    errors=[]
    if contract.get('population_alternatives'):
        matched=set().union(*(matching_alternatives(contract,f) for f in populations))
        if matched!=set(range(len(contract['population_alternatives']))):errors.append('比较工具必须覆盖用户要求的不同人群，不能用两组相同人群代替')
    if contract.get('allowed_languages') and {f.get('language') for f in populations}!=set(contract['allowed_languages']):
        errors.append('比较工具必须分别覆盖所有指定语言')
    return errors


def statistical_population_errors(contract,filters):
    """Do not use an unsolicited narrower population as a requested denominator.

    Evidence search may intentionally explore subgroups, so this guard applies
    only to numerical populations, not every semantic retrieval observation.
    """
    if not contract['requested_metrics']:return []
    expected=dict(contract['filters'])
    matches=matching_alternatives(contract,filters)
    if len(matches)==1:expected.update(contract['population_alternatives'][next(iter(matches))])
    defaults={'language':None,'min_playtime_hours':0,'max_playtime_hours':None}
    errors=[]
    for key,default in defaults.items():
        if key in expected:continue  # Explicit disagreements are checked by scope_errors.
        if key=='language' and filters.get(key) in contract.get('allowed_languages',[]):continue
        if filters.get(key,default)!=default:errors.append('统计不能增加用户未要求的人群筛选：'+key)
    if ({'positive_rate','negative_rate'} & set(contract['requested_metrics']) and
            'is_positive' not in expected and filters.get('is_positive') is not None):
        errors.append('推荐/差评率分母必须同时包含好评和差评，不能预先筛选态度')
    return errors
