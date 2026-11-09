"""Small default tool surface; the legacy pipeline is an explicit opt-in."""
from pathlib import Path
from datetime import datetime, timedelta, timezone

from pydantic import BaseModel, ConfigDict, Field, field_validator

from gamelens.core.runtime import AgentTool, ToolExecutionPolicy, ToolRegistry, ToolResult
from gamelens.domain.tool_results import (InspectDataPayload, ReviewQueryPayload,
    EvidenceSearchPayload, VerificationPayload)
from gamelens.storage.atomic_io import atomic_write_json
from gamelens.analysis.version import load_reviews, load_patch_candidates
from gamelens.tools.time_scope import patch_day, requested_patch_date, date_matches
from gamelens.core.local_time import local_datetime, local_midnight, time_context
from gamelens.tools.investigation import (
    ReviewInvestigation, ConditionQueryArgs, GetReviewArgs, ConditionCohortsArgs,
    VerifyInvestigationArgs, ListPatchesArgs,
)


class AppArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    appid: int = Field(gt=0)


class PrepareReviewsArgs(AppArgs):
    target_count: int = Field(default=1000, ge=1, le=5000)
    refresh: bool = False


class PreparePatchesArgs(AppArgs):
    target_count: int = Field(default=100, ge=1, le=1000)
    refresh: bool = False
    check_latest: bool = Field(default=False, description='为最近一次更新重新读取最新官方公告；不复用旧快照，不代表完整历史目录。')
    before_at: datetime | None = Field(default=None,description="历史公告补采：目标日次日北京时间零点；无偏移按北京时间时间，明确偏移保留真实时刻。例如2026-07-11T00:00:00。与旧缓存按ID合并，需审批。")

    @field_validator("before_at")
    @classmethod
    def timestamp_with_zone(cls,value):
        if value is not None:
            value = local_datetime(value)
            if not 0 < value.timestamp() <= 4294967295:
                raise ValueError("历史截止时间须为Steam支持的正uint32时间戳")
        return value


def review_fetch_approval(args, state):
    if args.refresh or args.target_count > 2000:
        return f"即将刷新或大量获取最多{args.target_count}条Steam评论，可能覆盖原始快照，是否继续？"
    return None


def patch_fetch_approval(args, state):
    if args.before_at is not None:
        return f"即将补采{args.before_at.isoformat()}之前最多{args.target_count}条历史Steam公告，并按ID合并缓存（不保证完整覆盖）；是否继续？"
    if args.refresh or args.target_count > 300:
        return f"即将刷新最多{args.target_count}条Steam公告，可能覆盖本地公告快照，是否继续？"
    return None


class SteamInvestigationTools:
    def __init__(self, project_root: Path, llm_client=None):
        self.root = project_root.resolve()
        self.llm_client = llm_client

    def registry(self, profile="focused") -> ToolRegistry:
        if profile == "extended":
            from gamelens.tools.legacy import LegacyInvestigationTools
            legacy = LegacyInvestigationTools(self.root, self.llm_client).registry()
            focused = self.registry('focused')
            additions = {'list_announcements','prepare_announcement_comments','query_announcement_comments','get_announcement_comment','prepare_patches'}
            additions.add('ensure_review_coverage')
            return ToolRegistry([*[t for t in legacy.tools if t.name != 'prepare_patches'],
                *[t for t in focused.tools if t.name in additions]], receipt_root=legacy.receipt_root)
        if profile != "focused":
            raise ValueError("调查模式必须是focused或extended")
        from gamelens.rag.tools import IndexArgs, SearchArgs, RagTools
        from gamelens.tools.argument_normalization import normalize_search_arguments
        domain = ReviewInvestigation(self.root)
        from gamelens.tools.announcement_comments import (AnnouncementDiscussionTools, PrepareDiscussionArgs,
            QueryDiscussionArgs, GetDiscussionArgs, discussion_fetch_approval)
        discussion = AnnouncementDiscussionTools(self.root)
        from gamelens.tools.review_coverage import ReviewCoverageTools,EnsureReviewCoverageArgs
        coverage=ReviewCoverageTools(self.root)
        rag = RagTools(self.root, client=self.llm_client)
        files = ToolExecutionPolicy(timeout_seconds=90, concurrency_key="project_files")
        fetch = ToolExecutionPolicy(timeout_seconds=180, max_attempts=2,
                                    retry_backoff_seconds=1, concurrency_key="project_files",
                                    retry_mode="transient_read")
        tools = [
            AgentTool("inspect_data", "检查评论与公告快照；不自动载入旧分类、聚合或报告。", AppArgs, self.inspect_data,
                      execution_policy=files, result_model=InspectDataPayload),
            AgentTool("prepare_reviews", "获取或复用Steam评论快照。", PrepareReviewsArgs, self.prepare_reviews, review_fetch_approval, fetch),
            AgentTool('ensure_review_coverage','目标评论时间/语言覆盖不足时，按精确窗口调用公开接口补采并合并，不覆盖旧缓存；分别核对平台计数与有限原文。程序可在查询前自动调用。',
                EnsureReviewCoverageArgs,coverage.ensure_review_coverage,
                lambda args,state: ('将按目标窗口采集较多公开评论，是否继续？' if args.target_count_per_window*len(args.windows)>2000 or args.refresh else None),
                ToolExecutionPolicy(timeout_seconds=180,max_attempts=1,concurrency_key='project_files'),planner_visible=False),
            AgentTool("prepare_patches", "获取或复用公告及补丁候选。历史缺失用带时区before_at补采，需审批；按ID合并保留原缓存，数量不是完整覆盖。", PreparePatchesArgs, self.prepare_patches, patch_fetch_approval, fetch),
            AgentTool("list_patches", "按published_on日期、news_id或标题定位全部本地历史公告，可offset分页；无条件才返回最新几条。缺年份跨年匹配须澄清，不得默认最新补丁。", ListPatchesArgs, domain.list_patches, execution_policy=files),
            AgentTool('list_announcements', '按北京时间日期/ID/标题定位Steam社区官方公告（补丁与活动），排除媒体新闻；无匹配返回邻近日候选供澄清，不自动换日期。', ListPatchesArgs, discussion.list_announcements, execution_policy=files),
            AgentTool('prepare_announcement_comments', '匿名采集具体公告下的独立Event讨论留言，核对公告/活动/主题ID；默认最多500条、上限1000，不是商店评价。没有公开入口/失败不是0条留言。', PrepareDiscussionArgs,
                discussion.prepare_announcement_comments, discussion_fetch_approval,
                ToolExecutionPolicy(timeout_seconds=180, max_attempts=1, concurrency_key='project_files')),
            AgentTool('query_announcement_comments', '按已定位news_id读取公告讨论留言和覆盖信息；默认均匀展示20条，最多50条，可分页或按留言创建时间筛选；没有好差评、游玩时长或官方语言标签，不推算观点占比。', QueryDiscussionArgs, discussion.query_announcement_comments, execution_policy=files),
            AgentTool('get_announcement_comment', '按news_id+comment_id读取该公告下一条完整留言，提供逐字引用；不是get_review。', GetDiscussionArgs, discussion.get_announcement_comment, execution_policy=files),
            AgentTool("query_reviews", "按时间/语言/推荐态度/游玩时长筛选；范围不足时程序先自动精确补采。支持的窗口计数由公开接口核对，其他条件仅统计已采集子集，原文样例与全量计数分开；具体问题用search_game_evidence，不接受关键词。", ConditionQueryArgs, domain.query_reviews,
                      execution_policy=files, result_model=ReviewQueryPayload),
            AgentTool("get_review", "按ID查询完整评论，核对原文及元数据。", GetReviewArgs, domain.get_review, execution_policy=files),
            AgentTool("compare_cohorts", "以精确人口和时间条件比较两组玩家，不以关键词近似问题分类；报告交集，不做因果推断。", ConditionCohortsArgs, domain.compare_cohorts, execution_policy=files),
            AgentTool("build_game_index", "仅需要语义证据时建立或增量更新按AppID缓存的本地索引；首次模型下载需审批。", IndexArgs, rag.build_game_index, rag.approval,
                      ToolExecutionPolicy(timeout_seconds=1800, concurrency_key="project_files")),
            AgentTool("search_game_evidence", "问题证据的唯一召回入口：关键词和向量独立召回、融合排序、LLM筛选；支持评论/公告分开查及词项扩展。本轮最多4次搜索，结果不是总体统计。缺失/过期先build_game_index。", SearchArgs, rag.search_game_evidence,
                      execution_policy=ToolExecutionPolicy(timeout_seconds=600, concurrency_key="project_files"),
                      argument_normalizer=normalize_search_arguments,
                      result_model=EvidenceSearchPayload),
            AgentTool("verify_investigation", "查询后核验快照与证据，生成本轮报告；finish前必须调用，不需要旧LLM分类链。", VerifyInvestigationArgs, domain.verify_investigation,
                      execution_policy=files, result_model=VerificationPayload),
        ]
        from dataclasses import replace
        from gamelens.tools.query_reuse import ReadonlySnapshot, digest
        query_snapshot = ReadonlySnapshot(self.root)
        search_snapshot = ReadonlySnapshot(self.root, include_index=True,
            screening_model=digest({key:getattr(self.llm_client, key, None)
                for key in ('model','base_url','max_output_tokens','enable_thinking','streaming')}))
        readonly = {'inspect_data','list_patches','list_announcements','query_announcement_comments','get_announcement_comment','query_reviews','get_review','compare_cohorts'}
        tools = [replace(tool, readonly_snapshot=search_snapshot if tool.name == 'search_game_evidence'
                         else query_snapshot if tool.name in readonly else None) for tool in tools]
        return ToolRegistry(tools, receipt_root=self.root / "data" / "tool_receipts")

    def _check(self, args, state):
        if args.appid != state["appid"]:
            raise ValueError("工具AppID必须与调查一致")

    def inspect_data(self, args, state):
        self._check(args, state)
        paths = {"reviews": self.root / "data/raw" / f"{args.appid}_reviews.json",
                 "patches": self.root / "data/raw" / f"{args.appid}_patch_candidates.json",
                 "news": self.root / "data/raw" / f"{args.appid}_news.json"}
        artifacts = {key: str(path) for key, path in paths.items() if path.is_file()}
        data={"available":sorted(artifacts)}
        data['announcement_comment_snapshots'] = len(list((self.root/'data/raw/announcement_comments'/str(args.appid)).glob('*.json')))
        if "reviews" in artifacts:
            from gamelens.tools.investigation import coverage_summary
            reviews=load_reviews(paths["reviews"])
            data["review_count"]=len(reviews)
            data["review_coverage"]=coverage_summary(reviews,{})
        if "patches" in artifacts:
            patches,_=ReviewInvestigation(self.root).patch_dataset(args.appid,{**state,"artifacts":{**state.get("artifacts",{}),**artifacts}})
            data["patch_coverage"]={"cached_count":len(patches),"first_date":patch_day(patches[-1]) if patches else None,
                "last_date":patch_day(patches[0]) if patches else None,"complete":False,"date_basis":"beijing"}
        data['time_standard'] = time_context()
        if state.get('rag_mode','auto') != 'off':
            import sqlite3
            from gamelens.rag.index import GameIndex
            from gamelens.rag.embedding import LocalEmbedding
            try:
                status = GameIndex(self.root).status(args.appid,{**state,'artifacts':{**state.get('artifacts',{}),**artifacts}})
                data['index_status'] = status['status']
            except (sqlite3.Error,OSError,ValueError) as error:
                data['index_status'] = 'unavailable'
                data['index_error_type'] = type(error).__name__
            data['embedding_ready'] = LocalEmbedding(self.root).ready
        return ToolResult(ok=True, summary="已检查原始快照与观测时间范围；现有资料：" + ("、".join(artifacts) or "无"),
                          data=data, artifacts=artifacts)

    def prepare_reviews(self, args, state):
        self._check(args, state)
        path = self.root / "data/raw" / f"{args.appid}_reviews.json"
        from gamelens.tools.steam_reviews import prepare_review_snapshot
        prepared=prepare_review_snapshot(path,args.appid,language="schinese",
            target_count=args.target_count,refresh=args.refresh)
        reviews=prepared.pop("reviews")
        reused=prepared["reused"]
        from gamelens.tools.investigation import coverage_summary
        status="复用" if reused and not prepared["refresh_attempted"] else "保留旧快照" if prepared["retained_previous_snapshot"] else "重新采集"
        return ToolResult(ok=True, summary=f"评论快照{status}完成，共{len(reviews)}条；数量不保证历史覆盖",
                          data={"review_count": len(reviews),**prepared,"coverage":coverage_summary(reviews,{})}, artifacts={"reviews": str(path)})

    def prepare_patches(self, args, state):
        self._check(args, state)
        from gamelens.tools.time_scope import patch_acquisition_allowed
        if args.check_latest and (args.before_at is not None or not patch_acquisition_allowed(state)):
            raise ValueError('最新公告核对不能带历史截止日，也不能在只看缓存/离线模式中联网')
        raw = self.root / "data/raw"
        path = raw / f"{args.appid}_patch_candidates.json"
        target=requested_patch_date(state.get("goal",""))
        if target and args.before_at is not None:
            if len(target)==5:
                news_path=raw / f"{args.appid}_news.json"
                cached=load_patch_candidates(news_path) if news_path.is_file() else load_patch_candidates(path) if path.is_file() else []
                years={patch_day(p)[:4] for p in cached if date_matches(p,target)}
                if len(years)!=1:
                    raise ValueError("历史补采前须先向用户确认年份；未找到唯一缓存年份，不能擅自补当前年")
                target=next(iter(years))+"-"+target
            cutoff=local_midnight(target,days=1)
            if args.before_at.timestamp()!=cutoff.timestamp():
                raise ValueError("历史补采截止时间与目标本地公告日期不一致；须使用目标日次日北京时间零点："+cutoff.isoformat())
        reused=path.is_file() and not args.refresh and not args.check_latest and args.before_at is None
        fetched=[]
        if reused:
            news_path=raw / f"{args.appid}_news.json"
            source=news_path if news_path.is_file() else path
            from gamelens.tools.steam_news import retained_patch_candidate
            items=load_patch_candidates(source)
            patches=sorted((item.model_copy(update={"is_patch_candidate":True}) for item in items
                if item.appid==args.appid and retained_patch_candidate(item)),
                key=lambda item:item.published_at_timestamp,reverse=True)
            atomic_write_json(path,[item.model_dump(mode="json") for item in patches])
        else:
            from gamelens.tools.steam_news import fetch_news, normalize_news, retained_patch_candidate
            fetch_args={"enddate":int(args.before_at.timestamp())} if args.before_at is not None else {}
            from gamelens.domain.announcement_comments import wants_announcement_feedback
            if args.check_latest or wants_announcement_feedback(state.get('goal', '')):
                fetch_args['feeds'] = 'steam_community_announcements'
            fetched = [normalize_news(item, args.appid) for item in fetch_news(args.appid, args.target_count, **fetch_args)]
            news_path=raw / f"{args.appid}_news.json"
            existing=load_patch_candidates(news_path) if news_path.is_file() else []
            if path.is_file():existing+=load_patch_candidates(path)
            merged={item.news_id:item for item in existing if item.appid==args.appid}
            merged.update({item.news_id:item for item in fetched})
            news=sorted(merged.values(),key=lambda item:item.published_at_timestamp,reverse=True)
            patches = sorted((item.model_copy(update={"is_patch_candidate":True}) for item in news
                              if retained_patch_candidate(item)),
                             key=lambda item: item.published_at_timestamp, reverse=True)
            atomic_write_json(raw / f"{args.appid}_news.json", [item.model_dump(mode="json") for item in news])
            atomic_write_json(path, [item.model_dump(mode="json") for item in patches])
        latest_check={'status':'not_checked'}
        if args.check_latest:
            import hashlib
            fetched_patches=sorted((n for n in fetched if retained_patch_candidate(n)),
                key=lambda n:(n.published_at_timestamp,n.news_id),reverse=True)
            latest_check={'status':'checked' if fetched_patches else 'no_patch_in_response', 'requested_count':args.target_count,
                'source_path':str((raw / f'{args.appid}_news.json').resolve()),
                'source_sha256':hashlib.sha256((raw / f'{args.appid}_news.json').read_bytes()).hexdigest(),
                'checked_at':datetime.now(timezone.utc).isoformat(),
                'official_feed':True, 'fetched_patch_count':len(fetched_patches),
                'latest_patch_id':fetched_patches[0].news_id if fetched_patches else None}
        return ToolResult(ok=True, summary=f"补丁快照{'复用' if reused else '合并'}完成，共{len(patches)}条；历史覆盖未保证",
                          data={"patch_count": len(patches),"reused":bool(reused),"fetched_count":len(fetched),
                                "latest_check":latest_check,
                                "before_at":args.before_at.isoformat() if args.before_at else None, "time_standard":time_context(),
                                "first_date":patch_day(patches[-1]) if patches else None,
                                "last_date":patch_day(patches[0]) if patches else None,"complete":False},
                          artifacts={"patches":str(path), "news":str(raw / f'{args.appid}_news.json')}
                          if (raw / f'{args.appid}_news.json').is_file() else {"patches":str(path)})
