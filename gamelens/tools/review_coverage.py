"""Time-window acquisition cache, immutable evidence snapshots and planning gate."""
import hashlib
import json
import math
import re
import time
from datetime import datetime,timezone
from pathlib import Path

from pydantic import BaseModel,ConfigDict,Field,model_validator

from gamelens.analysis.version import load_reviews
from gamelens.core.local_time import local_datetime,local_iso,shift_calendar_days
from gamelens.domain.models import SteamReview
from gamelens.storage.atomic_io import atomic_write_json
from gamelens.tools.query_reuse import digest
from gamelens.tools.time_scope import selection_days,requested_directions,requested_windows
from gamelens.tools.cache_coverage import cache_satisfies_target,candidate_improves_cache

TTL=3600
FAILURE_TTL=60


class AcquisitionWindow(BaseModel):
    model_config=ConfigDict(extra='forbid')
    start_at:datetime
    end_at:datetime
    language:str=Field(default='all',pattern=r'^[a-z_]{2,32}$')

    @model_validator(mode='after')
    def dates(self):
        self.start_at=local_datetime(self.start_at);self.end_at=local_datetime(self.end_at)
        if not 0<self.start_at.timestamp()<self.end_at.timestamp()<=4294967296:
            raise ValueError('补采区间须为有效的Steam时间范围，结束时刻不含')
        return self


class EnsureReviewCoverageArgs(BaseModel):
    model_config=ConfigDict(extra='forbid')
    appid:int=Field(gt=0)
    windows:list[AcquisitionWindow]=Field(min_length=1,max_length=4)
    target_count_per_window:int=Field(default=1000,ge=1,le=2000)
    max_pages_per_window:int=Field(default=10,ge=1,le=20)
    max_seconds:int=Field(default=90,ge=5,le=150)
    refresh:bool=False


def window_key(appid,start,end,language):
    return digest({'appid':appid,'start':math.ceil(start),'end':math.ceil(end),'language':language,
        'purchase_type':'all','review_type':'all','offtopic_included':True})[:24]


def cache_only(goal):
    return bool(re.search(r'(?:只|仅).{0,8}(?:本地|缓存|已采集)|(?:不要|不|无需)(?:联网|补采|抓取)',goal))


class ReviewCoverageTools:
    def __init__(self,root):self.root=Path(root).resolve()

    def ledger(self,appid,key):return self.root/'data/review_windows'/str(appid)/(key+'.json')

    def read(self,appid,key):
        path=self.ledger(appid,key)
        if not path.is_file():return None
        try:
            saved=json.loads(path.read_text(encoding='utf-8'))
            artifact=Path(saved['archive']['path']).resolve()
            if not artifact.is_relative_to((self.root/'data/review_windows').resolve()):return None
            if hashlib.sha256(artifact.read_bytes()).hexdigest()!=saved['archive']['sha256']:return None
            row=json.loads(artifact.read_text(encoding='utf-8'))
            if row['appid']!=appid or row['key']!=key:return None
            if window_key(appid,row['start'],row['end'],row['language'])!=key:return None
            if not 0<row['start']<row['counted_end']<=row['end']:return None
            comments=[SteamReview.model_validate(r) for r in row.get('comments',[])]
            if len(comments)!=row.get('body_count') or len({r.review_id for r in comments})!=len(comments):return None
            if any(not row['start']<=r.created_at_timestamp<row['counted_end'] or
                   row['language']!='all' and r.language!=row['language'] for r in comments):return None
            if row.get('stats_verified'):
                from gamelens.tools.steam_review_windows import counter
                stats=row['stats'];count=counter(stats['count'])
                positive=counter(stats['positive_count']);negative=counter(stats['negative_count'])
                if positive+negative!=count or count!=counter(row['reported_total']) or row.get('future_end_unobserved'):return None
                if stats['positive_rate']!=(positive/count if count else None) or stats['negative_rate']!=(negative/count if count else None):return None
                query=row['query']
                if query!={'appid':appid,'filter':1,'languages':[row['language']],'review_type':0,
                    'purchase_type':1,'filter_offtopic_activity':False,'date_range_start':row['start'],
                    'date_range_end':row['counted_end']-1}:return None
            age=datetime.now(timezone.utc).timestamp()-row['fetched_at_timestamp']
            ttl=TTL if row.get('stats_verified') or row.get('comments') else FAILURE_TTL
            if age<0 or age>ttl:return None
            return row,saved['archive']
        except (OSError,ValueError,KeyError,TypeError):return None

    def request_for(self,action,args,state):
        """Only actual registered query scope; no model-guessed dates or root cause."""
        if not state.get('allow_review_acquisition',True) or cache_only(state.get('goal','')):return None
        from gamelens.tools.investigation import ConditionQueryArgs,ConditionCohortsArgs,ReviewInvestigation
        from gamelens.rag.tools import SearchArgs
        models={'query_reviews':ConditionQueryArgs,'compare_cohorts':ConditionCohortsArgs,'search_game_evidence':SearchArgs}
        if action not in models or action=='search_game_evidence' and args.get('kind')=='announcements':return None
        parsed=models[action].model_validate(args)
        domain=ReviewInvestigation(self.root)
        populations=[parsed.cohort_a,parsed.cohort_b] if action=='compare_cohorts' else [parsed.filters]
        domain.check_requested_window(parsed,state,populations)
        patch=domain.patch(parsed.appid,state,parsed.patch_news_id) if parsed.patch_news_id or any(f.period!='all' for f in populations) else None
        now=datetime.now(timezone.utc).timestamp();windows=[]
        for filters in populations:
            _,bounds=domain.select([],filters,patch,selection_days(parsed,state))
            start=bounds['start_inclusive'];end=bounds['end_exclusive']
            if start is None and end is None:continue  # No invented range for an all-cache query.
            start=max(1,start if start is not None else 1);end=end if end is not None else now
            if start>=now:continue
            windows.append({'start_at':local_iso(start),'end_at':local_iso(end),'language':filters.language or 'all'})
        # Preparing both requested sides at once avoids changing the dataset
        # between before/after queries, and never substitutes a different patch.
        if patch and requested_directions(state.get('goal',''))=={'before','after'} and action!='compare_cohorts':
            durations=dict(requested_windows(state['goal']));days=parsed.window_days or state['window_days']
            pivot=patch.published_at_timestamp;language=populations[0].language or 'all'
            windows=[{'start_at':local_iso(shift_calendar_days(pivot,-durations.get('前',days))),
                'end_at':local_iso(pivot),'language':language},
                {'start_at':local_iso(pivot),'end_at':local_iso(shift_calendar_days(pivot,durations.get('后',days))),
                 'language':language}]
        windows=list({window_key(state['appid'],local_datetime(w['start_at']).timestamp(),
            local_datetime(w['end_at']).timestamp(),w['language']):w for w in windows}.values())
        if not windows:return None
        keys={window_key(state['appid'],local_datetime(w['start_at']).timestamp(),local_datetime(w['end_at']).timestamp(),w['language']) for w in windows}
        desired_target=EnsureReviewCoverageArgs.model_fields['target_count_per_window'].default
        attempted_targets={}
        for observation in state.get('observations',[]):
            if observation.get('tool')!='ensure_review_coverage':continue
            supplied=observation.get('effective_arguments') or observation.get('arguments') or {}
            recorded_targets=[row.get('requested_target_count') for row in
                observation.get('data',{}).get('windows',[]) if type(row.get('requested_target_count')) is int]
            target=supplied.get('target_count_per_window',max(recorded_targets) if recorded_targets else desired_target)
            if type(target) is not int or target<1:target=desired_target
            for key in observation.get('data',{}).get('window_keys',[]):
                attempted_targets[key]=max(attempted_targets.get(key,0),target)
        # Failed validation/denied approval is also an attempt: changing a
        # sample limit must not trigger an endless acquisition/approval loop.
        for observation in state.get('observations',[]):
            if observation.get('tool')!='ensure_review_coverage':continue
            for window in (observation.get('effective_arguments') or observation.get('arguments') or {}).get('windows',[]):
                try:
                    w=AcquisitionWindow.model_validate(window)
                    key=window_key(state['appid'],w.start_at.timestamp(),w.end_at.timestamp(),w.language)
                    target=(observation.get('effective_arguments') or observation.get('arguments') or {}).get(
                        'target_count_per_window',desired_target)
                    if type(target) is not int or target<1:target=desired_target
                    attempted_targets[key]=max(attempted_targets.get(key,0),target)
                except (ValueError,TypeError):pass
        if all(attempted_targets.get(key,0)>=desired_target for key in keys):return None
        path=Path(state.get('artifacts',{}).get('reviews',self.root/'data/raw'/f"{state['appid']}_reviews.json"))
        reviews=load_reviews(path) if path.is_file() else []
        from gamelens.tools.investigation import coverage_summary
        needed=False
        for window in windows:
            start=local_datetime(window['start_at']).timestamp();end=local_datetime(window['end_at']).timestamp()
            coverage=coverage_summary(reviews,{'start_inclusive':start,'end_exclusive':end})
            relevant=[r for r in reviews if start<=r.created_at_timestamp<end and (window['language']=='all' or r.language==window['language'])]
            # A persisted acquisition also needs attaching to this new run's
            # merged snapshot, even if the original recent JSON remains old.
            persisted=self.read(state['appid'],window_key(state['appid'],start,end,window['language']))
            # A valid persisted window still has to be attached to this run's
            # immutable merged dataset. The attempted-target gate above keeps
            # this from looping after an equal-or-larger request in one run.
            needed|=bool(persisted or not persisted and (
                len(relevant)<desired_target or coverage['requested_start_not_observed'] or
                coverage['requested_end_not_observed']))
        return {'appid':state['appid'],'windows':windows,
            'target_count_per_window':desired_target} if needed else None

    def ensure_review_coverage(self,args,state):
        from gamelens.core.runtime import ToolResult
        from gamelens.tools.steam_review_windows import acquire_window
        if args.appid!=state['appid']:raise ValueError('工具AppID必须与调查一致')
        if not state.get('allow_review_acquisition',True):raise ValueError('当前离线/固定快照模式禁止联网补采')
        if cache_only(state['goal']):raise ValueError('用户要求只看缓存，不允许联网补采')
        # Only the runtime may authorize the derived range. The model cannot
        # acquire arbitrary dates instead of the validated original query.
        authorized=state.get('coverage_followup',{}).get('windows',[])
        authorized_keys={window_key(args.appid,w.start_at.timestamp(),w.end_at.timestamp(),w.language)
                         for w in map(AcquisitionWindow.model_validate,authorized)}
        requested_keys={window_key(args.appid,w.start_at.timestamp(),w.end_at.timestamp(),w.language) for w in args.windows}
        if not authorized_keys or requested_keys!=authorized_keys:
            raise ValueError('补采范围须由有效的原始统计/证据查询绑定；先调用query_reviews、compare_cohorts或search_game_evidence，程序自动检查覆盖')
        deadline=time.monotonic()+args.max_seconds;now=datetime.now(timezone.utc).timestamp()
        rows=[];refs=[];keys=[]
        for window in args.windows:
            start=window.start_at.timestamp();end=window.end_at.timestamp()
            key=window_key(args.appid,start,end,window.language);keys.append(key)
            cached=self.read(args.appid,key)
            prior_row,prior_reference=cached if cached else (None,None)
            prior_adequate=bool(prior_row and cache_satisfies_target(
                prior_row.get('body_count',0),args.target_count_per_window,
                complete=bool(prior_row.get('complete_bodies'))))
            if prior_row and prior_adequate and not args.refresh:
                rows.append({**prior_row,'reused':True,'cache_adequate':True,
                    'requested_target_count':args.target_count_per_window,
                    'refresh_attempted':False,'retained_previous_snapshot':False,
                    'refresh_error_code':None})
                refs.append(prior_reference);continue
            actual_end=min(end,now)
            if start>=actual_end:
                raw={'comments':[],'stats':None,'stats_verified':False,'error_code':'future_window',
                     'body_count':0,'pages_read':0,'complete_bodies':False,'stop_reason':'future_window'}
            else:
                raw=acquire_window(args.appid,start,actual_end,window.language,target_count=args.target_count_per_window,
                    max_pages=args.max_pages_per_window,deadline=deadline)
            row={**raw,'appid':args.appid,'key':key,'start':math.ceil(start),'end':math.ceil(end),
                'counted_end':math.ceil(actual_end),'language':window.language,'future_end_unobserved':end>now,
                'fetched_at_timestamp':datetime.now(timezone.utc).timestamp(),'reused':False,
                'requested_target_count':args.target_count_per_window,'refresh_attempted':True,
                'retained_previous_snapshot':False,'refresh_error_code':None}
            if row['future_end_unobserved']:row['stats_verified']=False
            row['cache_adequate']=cache_satisfies_target(row.get('body_count',0),args.target_count_per_window,
                complete=bool(row.get('complete_bodies')))
            explicit_refresh_replacement = bool(args.refresh and not row.get('error_code') and row['cache_adequate'])
            if prior_row and not explicit_refresh_replacement and not candidate_improves_cache(
                    prior_row.get('body_count',0),row.get('body_count',0),args.target_count_per_window,
                    existing_complete=bool(prior_row.get('complete_bodies')),
                    candidate_complete=bool(row.get('complete_bodies')),
                    existing_error=prior_row.get('error_code'),candidate_error=row.get('error_code')):
                refresh_error=row.get('error_code') or 'candidate_not_better'
                row={**prior_row,'reused':True,
                    'cache_adequate':cache_satisfies_target(prior_row.get('body_count',0),
                        args.target_count_per_window,complete=bool(prior_row.get('complete_bodies'))),
                    'requested_target_count':args.target_count_per_window,'refresh_attempted':True,
                    'retained_previous_snapshot':True,'refresh_error_code':refresh_error}
                rows.append(row);refs.append(prior_reference);continue
            encoded=json.dumps(row,sort_keys=True,ensure_ascii=False).encode('utf-8')
            archive=self.root/'data/review_windows'/str(args.appid)/'snapshots'/(key+'_'+hashlib.sha256(encoded).hexdigest()[:16]+'.json')
            atomic_write_json(archive,row)
            reference={'path':str(archive),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}
            atomic_write_json(self.ledger(args.appid,key),{'archive':reference})
            rows.append(row);refs.append(reference)
        base=Path(state.get('artifacts',{}).get('reviews',self.root/'data/raw'/f'{args.appid}_reviews.json'))
        if base.is_file():refs.append({'path':str(base.resolve()),'sha256':hashlib.sha256(base.read_bytes()).hexdigest()})
        records={r.review_id:r.model_dump(mode='json') for r in load_reviews(base)} if base.is_file() else {}
        original=len(records)
        for row in rows:
            for item in row.get('comments',[]):
                review=SteamReview.model_validate(item).model_dump(mode='json');rid=review['review_id']
                if rid not in records or review['updated_at_timestamp']>=records[rid]['updated_at_timestamp']:records[rid]=review
        corpus=sorted(records.values(),key=lambda r:r['review_id']);generation=digest(corpus)
        snapshot=self.root/'data/review_windows'/str(args.appid)/'datasets'/(generation+'.json')
        if not snapshot.is_file():atomic_write_json(snapshot,corpus)
        summary_rows=[{k:v for k,v in row.items() if k not in {'comments','query'}} for row in rows]
        errors=[r.get('refresh_error_code') or r.get('error_code') for r in rows
                if r.get('refresh_error_code') or r.get('error_code')]
        suffix='；来源限制：'+','.join(sorted(set(errors))) if errors else ''
        return ToolResult(ok=True,summary=f"已按目标时段检查/补采{len(rows)}个窗口，合并新增{len(records)-original}条原文；全量计数与原文样例分别核对"+suffix,
            data={'window_keys':keys,'windows':summary_rows,'added_reviews':len(records)-original,
                'error_codes':errors,
                'source':{'path':str(snapshot),'sha256':hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                    'reference_files':refs,'scope':'time_window_acquisition_snapshot'},
                'scope_limited':any(not r.get('complete_bodies') for r in rows)},
            artifacts={'reviews':str(snapshot)})

    def window_statistics(self,appid,bounds,filters):
        start=bounds.get('start_inclusive');end=bounds.get('end_exclusive')
        if start is None or end is None:return None
        key=window_key(appid,start,end,filters.language or 'all');cached=self.read(appid,key)
        if not cached:return None
        row,reference=cached
        if not row.get('stats_verified') or not row.get('stats'):return None
        restricted=bool(filters.min_playtime_hours or filters.max_playtime_hours is not None or
            filters.min_playtime_exclusive or filters.max_playtime_exclusive or getattr(filters,'keywords',[]))
        if restricted:
            if not row.get('complete_bodies'):return None
            hours_restricted=bool(filters.min_playtime_hours or filters.max_playtime_hours is not None or
                filters.min_playtime_exclusive or filters.max_playtime_exclusive)
            if hours_restricted and not row.get('playtime_metadata_complete',{}).get(filters.playtime_basis):
                return None  # Missing playtime is unknown, not a proven zero-hour player.
            from gamelens.tools.investigation import ReviewInvestigation,stats as count_stats
            population,_=ReviewInvestigation(self.root).select([SteamReview.model_validate(r) for r in row['comments']],
                filters.model_copy(update={'period':'all'}),None,None)
            return count_stats(population),dict(reference),{**row,'statistics_basis':'steam_window_complete_bodies',
                'population_reviews':population}
        stats=dict(row['stats'])
        if filters.is_positive is not None:
            count=stats['positive_count' if filters.is_positive else 'negative_count']
            stats={'count':count,'positive_count':count if filters.is_positive else 0,
                'negative_count':0 if filters.is_positive else count,
                'positive_rate':(1. if filters.is_positive else 0.) if count else None,
                'negative_rate':(0. if filters.is_positive else 1.) if count else None}
        return stats,{'path':reference['path'],'sha256':reference['sha256']},{**row,'statistics_basis':'steam_public_window_counts'}
