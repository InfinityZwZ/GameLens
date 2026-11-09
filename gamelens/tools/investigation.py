"""Deterministic review investigations over a clearly identified local snapshot."""
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator
from gamelens.storage.atomic_io import atomic_write_json, atomic_write_text
from gamelens.analysis.version import load_reviews, load_patch_candidates
from gamelens.analysis.version import load_report
from gamelens.core.local_time import local_datetime, local_iso, shift_calendar_days, time_context, timezone_signature
from gamelens.tools.time_scope import (normalize_patch_date, date_matches, patch_day,
    validate_bounds, validate_dated_request, selection_days, requested_patch_date)


class MetadataReviewFilter(BaseModel):
    """Exact population conditions; deliberately excludes textual relevance."""
    model_config = ConfigDict(extra="forbid",allow_inf_nan=False,revalidate_instances="always")
    period: Literal["before","after","all"] = "after"
    language: str | None = None
    is_positive: bool | None = None
    min_playtime_hours: float = Field(default=0,ge=0)
    max_playtime_hours: float | None = Field(default=None,ge=0)
    min_playtime_exclusive: bool = False
    max_playtime_exclusive: bool = False
    playtime_basis: Literal["at_review","lifetime"] = "at_review"
    start_at: datetime | None = Field(default=None, description='开始时刻包含；未带时区按北京时间时间，明确偏移保留真实时刻。')
    end_at: datetime | None = Field(default=None, description='结束时刻不含；未带时区按北京时间时间，日期表示当天零点。')

    @model_validator(mode="before")
    @classmethod
    def metadata_only(cls,value):
        if cls is not MetadataReviewFilter:
            return value  # Extended ReviewFilter still supports literal filtering.
        if isinstance(value,BaseModel):value=value.model_dump()
        if isinstance(value,dict):
            value=dict(value)
            if value.get("keywords"):
                raise ValueError("条件过滤不接受关键词；证据检索使用query和lexical_terms，不做字面硬过滤")
            # Empty historical keyword fields imposed no restriction; allow old checkpoints.
            value.pop("keywords",None)
            if value.get("keyword_mode") in {"any","all"}:value.pop("keyword_mode")
        return value

    @model_validator(mode="after")
    def validate_ranges(self):
        if self.max_playtime_hours is not None and self.max_playtime_hours < self.min_playtime_hours:
            raise ValueError("游玩时长上下界相反")
        if self.max_playtime_hours == self.min_playtime_hours and (self.min_playtime_exclusive or self.max_playtime_exclusive):
            raise ValueError('严格时长边界之间没有交集')
        for name in ('start_at','end_at'):
            moment = getattr(self, name)
            if moment is not None:
                setattr(self, name, local_datetime(moment))
        if self.start_at and self.end_at and self.end_at <= self.start_at:
            raise ValueError("结束时间必须晚于开始时间")
        return self


class ReviewFilter(MetadataReviewFilter):
    """Legacy literal filtering, retained for extended tasks and old fixtures."""
    keywords: list[str] = Field(default_factory=list,max_length=10)
    keyword_mode: Literal["any","all"] = "any"

    @model_validator(mode="after")
    def validate_keywords(self):
        self.keywords = [word.strip() for word in self.keywords]
        if any(not word or len(word)>100 for word in self.keywords):
            raise ValueError("关键词不能为空或超过100字符")
        return self


class QueryReviewsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    appid: int = Field(gt=0)
    filters: ReviewFilter = Field(default_factory=ReviewFilter)
    patch_news_id: str | None = None
    window_days: int | None = Field(default=None,ge=1,le=30)
    limit: int = Field(default=5,ge=1,le=20)
    sort_by: Literal["recent","votes"] = "votes"


class GetReviewArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    appid: int = Field(gt=0)
    review_id: str = Field(min_length=1,max_length=100)


class CompareCohortsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    appid: int = Field(gt=0)
    cohort_a: ReviewFilter
    cohort_b: ReviewFilter
    patch_news_id: str | None = None
    window_days: int | None = Field(default=None,ge=1,le=30)


class ConditionQueryArgs(QueryReviewsArgs):
    filters: MetadataReviewFilter = Field(default_factory=MetadataReviewFilter,
        description="仅精确人口/时间条件。找具体问题证据用search_game_evidence，不接受关键词。")


class ConditionCohortsArgs(CompareCohortsArgs):
    cohort_a: MetadataReviewFilter
    cohort_b: MetadataReviewFilter


class CompareVersionsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    appid: int = Field(gt=0)
    patch_a: str
    patch_b: str
    window_days: int = Field(default=7,ge=1,le=30)
    filters: ReviewFilter = Field(default_factory=ReviewFilter)


class VerifyInvestigationArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    appid: int = Field(gt=0)


class RecentVersionsArgs(VerifyInvestigationArgs):
    count: int = Field(default=5,ge=1,le=10)
    window_days: int | None = Field(default=None,ge=1,le=30)


class ListPatchesArgs(VerifyInvestigationArgs):
    limit: int = Field(default=5,ge=1,le=10)
    published_on: str | None = Field(default=None,description="按北京时间发布日期定位历史更新：YYYY-MM-DD或MM-DD；缺年匹配多个年份须澄清。")
    news_id: str | None = Field(default=None,min_length=1,max_length=100)
    title_query: str | None = Field(default=None,min_length=1,max_length=100)
    offset: int = Field(default=0,ge=0,le=10000)

    @field_validator("published_on")
    @classmethod
    def valid_day(cls,value):
        return normalize_patch_date(value) if value is not None else None


def required_patch_count(goal):
    explicit = set(re.findall(r"(?<![A-Za-z0-9])p\d+(?![A-Za-z0-9])", goal, re.I))
    if len(explicit) > 1:
        return len(explicit)
    if re.search(r"(?:比较|对比|对照).{0,12}(?:两|二|2)\s*(?:次|个)?\s*(?:更新|版本|补丁)|(?:两|二|2)\s*(?:次|个)?\s*(?:更新|版本|补丁).{0,12}(?:比较|对比|对照)", goal):
        return 2
    match=re.search(r"(?:最近|近|过去)\s*([一二三四五六七八九十\d]+)\s*(?:次|个)\s*(?:更新|版本|补丁)",goal)
    if match:
        value=match.group(1)
        return int(value) if value.isdigit() else {"一":1,"二":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,"十":10}.get(value,1)
    match=re.search(r"(?:last|recent)\s+(\d+)\s+(?:updates|versions|patches)",goal,re.I)
    return int(match.group(1)) if match else 1


def stats(reviews):
    positive = sum(review.is_positive for review in reviews)
    return {"count":len(reviews),"positive_count":positive,"negative_count":len(reviews)-positive,
            "positive_rate":positive/len(reviews) if reviews else None,
            "negative_rate":(len(reviews)-positive)/len(reviews) if reviews else None}


def coverage_summary(reviews, bounds):
    """Observed extents are not acquisition coverage or historical text versions."""
    validate_bounds(bounds)
    moments = [r.created_at_timestamp for r in reviews]
    start, end = bounds.get("start_inclusive"), bounds.get("end_exclusive")
    outside = bool(moments and ((start is not None and start > max(moments)) or
                               (end is not None and end <= min(moments))))
    selected = [r for r in reviews if (start is None or r.created_at_timestamp >= start)
                and (end is None or r.created_at_timestamp < end)]
    revised = sum(end is not None and r.updated_at_timestamp >= end and
                  r.updated_at_timestamp > r.created_at_timestamp for r in selected)
    return {"status": "outside_observed_range" if outside else ("no_cached_reviews" if not selected else "partial_snapshot"),
            "observed_start": min(moments) if moments else None,
            "observed_end": max(moments) if moments else None,
            "complete": False, "historical_text_available": False,
            "requested_start_not_observed": bool(moments and start is not None and start < min(moments)),
            "requested_end_not_observed": bool(moments and end is not None and end > max(moments)+1),
            "requested_window_observed": not outside and bool(selected),
            "time_basis": "review_created_at", "revised_after_window_count": revised,
            "warning": "时间按评论创建时间分窗，正文是抓取时的最新版本，不是当时的原始正文；缓存起止日期不代表区间完整覆盖。"}


class ReviewInvestigation:
    def __init__(self,root: Path):
        self.root = root.resolve()

    def dataset(self,appid,state):
        if appid != state["appid"]:
            raise ValueError("工具AppID必须与当前调查一致")
        path = Path(state.get("artifacts",{}).get("reviews",self.root/"data"/"raw"/f"{appid}_reviews.json"))
        reviews = load_reviews(path)
        reviews = list({review.review_id:review for review in reviews}.values())
        source = {"path":str(path.resolve()),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                  "cached_review_count":len(reviews),"scope":"cached_reviews_only",
                  "warning":"这是本地已获取评论，不保证覆盖Steam全部评论；关键词命中不等于已验证软件缺陷。"}
        source["coverage"] = coverage_summary(reviews, {})
        references=[]
        for candidate in (state.get("artifacts",{}).get("version_analysis"),state.get("artifacts",{}).get("patches",self.root/"data"/"raw"/f"{appid}_patch_candidates.json")):
            if candidate and Path(candidate).is_file(): references.append({"path":str(Path(candidate).resolve()),"sha256":hashlib.sha256(Path(candidate).read_bytes()).hexdigest()})
        source["reference_files"]=references
        return reviews,source

    def patch_dataset(self, appid, state):
        """Load and reclassify the authoritative news snapshot on every read.

        Older caches may contain external media stories that happened to use the
        word "update".  They are never eligible to become a patch boundary.
        """
        news_path = Path(state.get("artifacts", {}).get(
            "news", self.root / "data/raw" / f"{appid}_news.json"))
        patch_path = Path(state.get("artifacts", {}).get(
            "patches", self.root / "data/raw" / f"{appid}_patch_candidates.json"))
        source_path = news_path if news_path.is_file() else patch_path
        items = load_patch_candidates(source_path) if source_path.is_file() else []
        from gamelens.tools.steam_news import retained_patch_candidate
        patches = [item.model_copy(update={"is_patch_candidate": True}) for item in items
                   if item.appid == appid and retained_patch_candidate(item)]
        return sorted(patches, key=lambda item:(item.published_at_timestamp, item.news_id), reverse=True), source_path

    def patch(self,appid,state,id=None):
        artifact = state.get("artifacts",{}).get("version_analysis")
        if artifact and not id:
            report = load_report(Path(artifact))
            if report.appid == appid: return report.patch
        patches,_ = self.patch_dataset(appid,state)
        id = id or state.get("patch_news_id")
        if id:
            for patch in patches:
                if patch.news_id == id and patch.appid == appid: return patch
            raise ValueError(f"未找到补丁{id}")
        if not patches: raise ValueError("缺少补丁数据")
        return patches[0]

    def recent_patches(self,appid,state,count):
        if appid!=state["appid"]:raise ValueError("AppID不匹配")
        patches,_=self.patch_dataset(appid,state)
        return patches[:count]

    def list_patches(self,args: ListPatchesArgs,state):
        from gamelens.core.runtime import ToolResult
        cached,path=self.patch_dataset(args.appid,state)
        goal_date=requested_patch_date(state.get("goal",""))
        from gamelens.tools.time_scope import latest_patch_requested
        latest=latest_patch_requested(state.get('goal',''))
        target=None if latest else goal_date or args.published_on
        if goal_date and args.published_on and (args.published_on!=goal_date if len(goal_date)==10 else args.published_on[-5:]!=goal_date):
            raise ValueError("公告定位日期与用户指定更新日期不一致")
        patches=[p for p in cached if (not target or date_matches(p,target))
                 and (latest or not args.news_id or p.news_id==args.news_id)
                 and (latest or not args.title_query or args.title_query.casefold() in p.title.casefold())]
        years=sorted({patch_day(p)[:4] for p in patches})
        ambiguous=bool(target and len(target)==5 and len(years)>1)
        offset=0 if latest else args.offset
        selected=patches[offset:offset+args.limit]
        source={"path":str(path.resolve()),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"scope":"cached_announcements_only"} if path.is_file() else {}
        latest_check={'status':'not_checked'}
        if latest:
            for o in reversed(state.get('observations', [])):
                check=o.get('data', {}).get('latest_check', {})
                if o.get('tool')=='prepare_patches' and o.get('ok') and check.get('status') in {'checked','no_patch_in_response'}:
                    if check.get('source_path')==source.get('path') and check.get('source_sha256')==source.get('sha256'):
                        latest_check=check
                    break
        if latest and latest_check.get('latest_patch_id'):
            selected=[p for p in cached if p.news_id==latest_check['latest_patch_id']]
        data={"source":source,"latest_lookup":latest,"latest_check":latest_check,
              "requested_date":target,"date_basis":"beijing publication day",
              "cached_patch_count":len(cached),"matched_count":len(patches),"returned_count":len(selected),
              "offset":offset,"has_more":offset+len(selected)<len(patches),
              "needs_year_clarification":ambiguous,"matched_years":years,
              "lookup_status":"ambiguous_year" if ambiguous else ("found_in_cache" if patches else "not_found_in_cache"),
              "cached_first_date":patch_day(cached[-1]) if cached else None,
              "cached_last_date":patch_day(cached[0]) if cached else None,
              "patches":[{**p.model_dump(mode="json"), 'published_at_local':local_iso(p.published_at_timestamp), 'published_on_local':patch_day(p)} for p in selected],
              "announcements":[{**p.model_dump(mode="json"), 'published_at_local':local_iso(p.published_at_timestamp), 'published_on_local':patch_day(p)} for p in selected],
              "caution":("最近一次按新读取的官方补丁发布时间定位，不按今天的日期过滤；完整历史目录与最新目标确认是不同要求。" if latest else
                         "只搜索本地快照；未命中不代表官方没有该更新。缺历史公告时明确年份，审批后prepare_patches(before_at=目标日次日北京时间零点)补采。")}
        result=self.save("list_patches",data,state)
        result.summary=(f"最近补丁定位：{selected[0].title if selected else '暂无可定位补丁'}；官方核对状态：{latest_check['status']}" if latest else
                        f"历史公告定位：{data['lookup_status']}，缓存内匹配{len(patches)}条；不代表完整更新历史")
        return result

    def investigate_recent_versions(self,args: RecentVersionsArgs,state):
        reviews,source=self.dataset(args.appid,state)
        patches=self.recent_patches(args.appid,state,args.count)
        versions=[]
        days=args.window_days or state["window_days"]
        for index,patch in enumerate(patches):
            start=patch.published_at_timestamp
            end=min(shift_calendar_days(start,days),patches[index-1].published_at_timestamp) if index else shift_calendar_days(start,days)
            selected=[r for r in reviews if start<=r.created_at_timestamp<end]
            versions.append({"news_id":patch.news_id,"title":patch.title,"start_inclusive":start,"end_exclusive":end,
                "stats":stats(selected),"coverage":coverage_summary(reviews, {"start_inclusive":start,"end_exclusive":end}),"has_cached_reviews":bool(selected),"search_filters":{"period":"all","start_at":local_iso(start),"end_at":local_iso(end)}})
        return self.save("investigate_recent_versions",{"source":source,"requested_count":args.count,
            "available_count":len(patches),"versions_with_reviews":sum(v["has_cached_reviews"] for v in versions),
            "versions":versions,"announcements":[p.model_dump(mode="json") for p in patches],
            "caution":"窗口截止下一个补丁或指定天数，避免把同条评论重复算进多个版本。只统计缓存覆盖；逐窗口有评论不证明同一个缺陷持续存在，应分别检索原文。"},state)

    def select(self,reviews,filters,patch,days):
        start = filters.start_at.timestamp() if filters.start_at else None
        end = filters.end_at.timestamp() if filters.end_at else None
        if filters.period != "all":
            if patch is None: raise ValueError("before/after筛选需要补丁")
            pivot = patch.published_at_timestamp
            period_start = shift_calendar_days(pivot,-days) if filters.period=="before" and days is not None else (None if filters.period=="before" else pivot)
            period_end = pivot if filters.period=="before" else (shift_calendar_days(pivot,days) if days is not None else None)
            if period_start is not None:start = max(start,period_start) if start is not None else period_start
            if period_end is not None:end = min(end,period_end) if end is not None else period_end
        validate_bounds({"start_inclusive":start,"end_exclusive":end})
        result = []
        for review in reviews:
            if start is not None and review.created_at_timestamp < start: continue
            if end is not None and review.created_at_timestamp >= end: continue
            if filters.language and review.language != filters.language: continue
            if filters.is_positive is not None and review.is_positive != filters.is_positive: continue
            minutes = review.playtime_at_review_minutes if filters.playtime_basis=="at_review" else review.playtime_minutes
            if minutes < filters.min_playtime_hours*60 or filters.min_playtime_exclusive and minutes == filters.min_playtime_hours*60: continue
            if filters.max_playtime_hours is not None and (minutes > filters.max_playtime_hours*60 or filters.max_playtime_exclusive and minutes == filters.max_playtime_hours*60): continue
            keywords = getattr(filters,"keywords",[])
            if keywords:
                matches = [word.casefold() in review.text.casefold() for word in keywords]
                if not (all(matches) if filters.keyword_mode=="all" else any(matches)): continue
            result.append(review)
        return result,{"start_inclusive":start,"end_exclusive":end}

    def save(self,tool,data,state):
        from gamelens.core.runtime import ToolResult
        thread_id = state["thread_id"]
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,128}",thread_id): raise ValueError("非法任务ID")
        file = self.root/"data"/"investigations"/thread_id/f"{state['step_count']+1}_{tool}.json"
        data['time_standard'] = time_context()
        if data.get('source'):
            data['source']['time_zone'] = timezone_signature()
        atomic_write_json(file,data)
        return ToolResult(ok=True,summary=f"{tool}完成；统计仅针对可审计的本地评论快照",data=data,
                          artifacts={f"investigation_{tool}":str(file)})

    def query_reviews(self,args: QueryReviewsArgs,state):
        self.check_condition_only(state,[args.filters])
        self.check_requested_window(args,state,[args.filters])
        reviews,source = self.dataset(args.appid,state)
        patch = self.patch(args.appid,state,args.patch_news_id) if args.patch_news_id or args.filters.period!="all" else None
        selected,bounds = self.select(reviews,args.filters,patch,selection_days(args,state))
        ordered = sorted(selected,key=lambda r:(r.votes_up,r.created_at_timestamp) if args.sort_by=="votes" else (r.created_at_timestamp,r.votes_up),reverse=True)
        data = {"source":source,"filters":args.filters.model_dump(mode="json"),"bounds":bounds,
                "patch_news_id":patch.news_id if patch else None,"stats":stats(selected),
                "returned_count":min(len(ordered),args.limit),"limit":args.limit,
                "review_ids":[r.review_id for r in ordered[:args.limit]],
                "reviews":[r.model_dump(mode="json") for r in ordered[:args.limit]]}
        data["coverage"] = coverage_summary(reviews, bounds)
        from gamelens.tools.review_coverage import ReviewCoverageTools
        remote=ReviewCoverageTools(self.root).window_statistics(args.appid,bounds,args.filters)
        if remote and remote[2].get('population_reviews') is not None:
            selected=remote[2]['population_reviews']
            ordered=sorted(selected,key=lambda r:(r.votes_up,r.created_at_timestamp) if args.sort_by=='votes' else (r.created_at_timestamp,r.votes_up),reverse=True)
            data.update(returned_count=min(len(ordered),args.limit),review_ids=[r.review_id for r in ordered[:args.limit]],
                reviews=[r.model_dump(mode='json') for r in ordered[:args.limit]])
        data['body_sample_count']=len(selected)
        data['statistics_basis']='cached_subset'
        from gamelens.core.query_contract import contract_for
        data['statistics_complete_for_scope']=bool(contract_for(state)['explicit_cached_scope'])
        if remote:
            data['stats'],reference,row=remote
            source['reference_files'].append(reference)
            data.update(statistics_basis=row['statistics_basis'],statistics_fetched_at=local_iso(row['fetched_at_timestamp']),statistics_complete_for_scope=True)
            data['coverage'].update(platform_counts_verified=True,counts_complete=True)
        data["statistics_status"] = 'public_window_counts_verified' if remote else ("unavailable_window" if data["coverage"]["status"]=="outside_observed_range" else "cached_subset_only")
        data["interpretation"] = ('计数来自采集时公开接口对精确创建时间/语言范围的全量匹配总数，原文为有限样例；不是历史当时的文本/投票快照，也不代表全部玩家' if remote else
            "缓存不覆盖请求时段，0仅为空缓存筛选结果，不是该时段实际评论数" if data["statistics_status"]=="unavailable_window" else "仅统计符合条件的缓存评论；未观测区间未知")
        result=self.save("query_reviews",data,state)
        if remote:result.summary='已核对公开接口的目标窗口计数，并读取有限原文样例；不把样例数当全量计数'
        return result

    def check_requested_window(self,args,state,filters):
        validate_dated_request(self,args,state,filters)
        from gamelens.core.query_contract import contract_for, scope_errors,cohort_scope_errors,statistical_population_errors
        from gamelens.core.answers import simple_statistics_goal
        contract=contract_for(state)
        if len(filters)>1:
            errors=cohort_scope_errors(contract,[p.model_dump(mode='json') for p in filters])
            if errors:raise ValueError('；'.join(errors))
        for population in filters:
            errors=scope_errors(contract,population.model_dump(mode='json'))
            if not hasattr(args,'query') and (len(filters)>1 or simple_statistics_goal(state.get('goal',''))):
                errors.extend(statistical_population_errors(contract,population.model_dump(mode='json')))
            if errors:raise ValueError('；'.join(errors))
        from gamelens.core.investigation_checks import named_patch_time_alignment, explicit_patch_alignment
        goal=state.get("goal", "")
        ids=set(re.findall(r"(?<![A-Za-z0-9])p\d+(?![A-Za-z0-9])",goal,re.I))
        if len(ids)!=1:return
        action="compare_cohorts" if isinstance(args,CompareCohortsArgs) else "query_reviews"
        errors=explicit_patch_alignment(goal,action,args.model_dump(mode="json"),state)
        if errors:raise ValueError("；".join(errors))
        if not re.search(r"[前后]\s*([一二三四五六七八九十]|\d+)\s*天",goal):return
        requested=self.patch(args.appid,state,next(iter(ids)))
        for population in filters:
            errors=named_patch_time_alignment(goal,population,requested,args.window_days or state["window_days"])
            if errors:raise ValueError("；".join(errors))

    @staticmethod
    def check_condition_only(state,filters):
        if state.get("investigation_profile")=="focused" and any(getattr(f,"keywords",[]) for f in filters):
            raise ValueError("条件查询不接受关键词检索；具体问题证据统一使用search_game_evidence混合召回")

    def get_review(self,args: GetReviewArgs,state):
        reviews,source = self.dataset(args.appid,state)
        review = next((r for r in reviews if r.review_id==args.review_id),None)
        if review is None: raise ValueError("指定评论不在本地快照中，不补造原文")
        return self.save("get_review",{"source":source,"review":review.model_dump(mode="json"),"text_length":len(review.text)},state)

    def compare_cohorts(self,args: CompareCohortsArgs,state):
        self.check_condition_only(state,[args.cohort_a,args.cohort_b])
        self.check_requested_window(args,state,[args.cohort_a,args.cohort_b])
        reviews,source = self.dataset(args.appid,state)
        patch = self.patch(args.appid,state,args.patch_news_id) if args.patch_news_id or args.cohort_a.period!="all" or args.cohort_b.period!="all" else None
        a,ab = self.select(reviews,args.cohort_a,patch,selection_days(args,state))
        b,bb = self.select(reviews,args.cohort_b,patch,selection_days(args,state))
        sa,sb = stats(a),stats(b)
        from gamelens.tools.review_coverage import ReviewCoverageTools
        coverage_tools=ReviewCoverageTools(self.root)
        ra=coverage_tools.window_statistics(args.appid,ab,args.cohort_a)
        rb=coverage_tools.window_statistics(args.appid,bb,args.cohort_b)
        if ra and ra[2].get('population_reviews') is not None:a=ra[2]['population_reviews']
        if rb and rb[2].get('population_reviews') is not None:b=rb[2]['population_reviews']
        from gamelens.core.query_contract import contract_for
        cached_scope=contract_for(state)['explicit_cached_scope']
        if ra:sa=ra[0];source['reference_files'].append(ra[1])
        if rb:sb=rb[0];source['reference_files'].append(rb[1])
        delta = (sb["positive_rate"]-sa["positive_rate"])*100 if sa["positive_rate"] is not None and sb["positive_rate"] is not None else None
        return self.save("compare_cohorts",{"source":source,"patch_news_id":patch.news_id if patch else None,
            "cohort_a":{"filters":args.cohort_a.model_dump(mode="json"),"bounds":ab,"stats":sa,
                "statistics_basis":ra[2]['statistics_basis'] if ra else "cached_subset","body_sample_count":len(a),
                "statistics_complete_for_scope":bool(ra or cached_scope),
                "statistics_fetched_at":local_iso(ra[2]['fetched_at_timestamp']) if ra else None},
            "cohort_b":{"filters":args.cohort_b.model_dump(mode="json"),"bounds":bb,"stats":sb,
                "statistics_basis":rb[2]['statistics_basis'] if rb else "cached_subset","body_sample_count":len(b),
                "statistics_complete_for_scope":bool(rb or cached_scope),
                "statistics_fetched_at":local_iso(rb[2]['fetched_at_timestamp']) if rb else None},
            "coverage_a":{**coverage_summary(reviews,ab),'platform_counts_verified':bool(ra),'counts_complete':bool(ra)},
            "coverage_b":{**coverage_summary(reviews,bb),'platform_counts_verified':bool(rb),'counts_complete':bool(rb)},
            "overlap_count":len({r.review_id for r in a}&{r.review_id for r in b}),
            "positive_rate_delta_pp":delta,"caution":"分别报告平台匹配计数或缓存子集口径；原文仅为样例。采集时的当前评价不等于历史当时快照；群体可重叠，差异不构成更新导致问题的因果证据。"},state)

    def compare_versions(self,args: CompareVersionsArgs,state):
        reviews,source = self.dataset(args.appid,state)
        versions = []
        ids = []
        for id in (args.patch_a,args.patch_b):
            patch = self.patch(args.appid,state,id)
            selected,bounds = self.select(reviews,args.filters,patch,args.window_days)
            ids.append({r.review_id for r in selected})
            versions.append({"news_id":id,"title":patch.title,"bounds":bounds,"stats":stats(selected)})
        return self.save("compare_versions",{"source":source,"filters":args.filters.model_dump(mode="json"),
            "versions":versions,"overlap_count":len(ids[0]&ids[1]),"caution":"窗口可能重叠且覆盖不完整；不能据此声称因果关系。"},state)

    def verify_investigation(self,args: VerifyInvestigationArgs,state):
        from gamelens.core.runtime import ToolResult
        if args.appid != state["appid"]: raise ValueError("AppID不匹配")
        artifacts = state.get("artifacts",{})
        files = [Path(path) for key,path in artifacts.items() if key.startswith("investigation_")]
        if not files: raise ValueError("尚未生成调查结果")
        from gamelens.core.source_freshness import active_observations, superseded_locator_steps
        active = [o for o in active_observations(state.get('observations', [])) if o.get('ok') and o.get('data', {}).get('source')]
        for observation in active:
            source = observation['data']['source']
            if source.get('time_zone') and source['time_zone'] != timezone_signature():
                raise ValueError('系统时区已变化，请重新执行时间筛选查询再核验，不能混用旧时区结果')
            for reference in [source, *source.get('reference_files', [])]:
                if hashlib.sha256(Path(reference['path']).read_bytes()).hexdigest() != reference['sha256']:
                    raise ValueError(f"步骤{observation['step']}（{observation['tool']}）的来源快照已变化：须重做该查询；重复verify_investigation不能修复旧证据")
        results = []
        for path in files:
            data = json.loads(path.read_text(encoding="utf-8"))
            validate_bounds(data.get("bounds",{}))
            for group in (data.get("cohort_a",{}),data.get("cohort_b",{}),*data.get("versions",[])):
                validate_bounds(group.get("bounds",group if "start_inclusive" in group else {}))
            source = data["source"]
            if hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest()!=source["sha256"]:
                raise ValueError("原始评论快照已经变化，请重新调查")
            for reference in source.get("reference_files",[]):
                if hashlib.sha256(Path(reference["path"]).read_bytes()).hexdigest()!=reference["sha256"]:
                    raise ValueError("补丁/窗口来源已变化，请重新调查")
            results.append({"artifact":path.name,"result":data})
        output = self.root/"reports"/f"{args.appid}_{state['thread_id']}_investigation_report.md"
        lines = ["# 可追溯反馈调查",f"\n目标：{state['goal']}","\n范围：本地已获取资料，各来源分别报告，不保证Steam总体覆盖；无因果推断。"]
        # Even the provisional report is readable prose; raw payloads stay in
        # data/investigations for audit and are not another user-facing report.
        from gamelens.analysis.report_prose import readable_statistics
        observations = [{"ok":True, "tool":item["artifact"].rsplit(".",1)[0].split("_",1)[-1],
                         "data":item["result"]} for item in results]
        lines.extend(readable_statistics(observations))
        lines.append("\n本文件为来源核验后的中间报告，最终文字分析在调查完成后生成。原始记录保留在任务证据文件中。")
        atomic_write_text(output,"\n".join(lines))
        return ToolResult(ok=True,summary="已核对调查快照哈希并生成原文/数字可追溯报告",data={"verified_files":len(files),"scope":"verified_local_evidence",
            "verified_observation_steps":[o['step'] for o in active],
            "superseded_locator_steps":superseded_locator_steps(state.get('observations', []))},
            artifacts={"final_report":str(output),"evidence_assessed":"true"})
