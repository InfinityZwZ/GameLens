"""Steam game investigation tools used by the LangGraph agent."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel, ConfigDict, Field

from gamelens.agents.pipeline import AgentConfig, ensure_patch_candidates, ensure_reviews, select_patch
from gamelens.core.runtime import (
    AgentTool,
    InvestigationState,
    ToolExecutionPolicy,
    ToolRegistry,
    ToolResult,
)
from gamelens.analysis.impact import (
    build_impact_summary,
    load_classifications,
    save_impact_summary,
)
from gamelens.core.llm import OpenAICompatibleClient
from gamelens.analysis.report import generate_markdown, load_impact_summary, save_markdown_report
from gamelens.analysis.classifier import (
    classify_reviews,
    load_existing_classifications,
    load_report,
    save_classifications,
)
from gamelens.analysis.version import (
    build_version_impact_report,
    load_patch_candidates,
    load_reviews,
    save_report,
)
from gamelens.tools.investigation import (ReviewInvestigation, QueryReviewsArgs, GetReviewArgs,
    CompareCohortsArgs, CompareVersionsArgs, VerifyInvestigationArgs)
from gamelens.rag.tools import RagTools, IndexArgs, SearchArgs
from gamelens.tools.investigation import ListPatchesArgs, RecentVersionsArgs


class StrictArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AppArgs(StrictArgs):
    appid: int = Field(gt=0)


class PrepareReviewsArgs(AppArgs):
    target_count: int = Field(default=1000, ge=1, le=5000)
    refresh: bool = False


class PreparePatchesArgs(AppArgs):
    target_count: int = Field(default=100, ge=1, le=1000)
    refresh: bool = False


class AnalyzePatchArgs(AppArgs):
    window_days: int = Field(default=7, ge=1, le=30)
    patch_news_id: str | None = None


class ClassifyArgs(AppArgs):
    per_window_count: int = Field(default=10, ge=1, le=100)
    force: bool = False


def review_fetch_approval(
    args: BaseModel,
    state: InvestigationState,
) -> str | None:
    request = PrepareReviewsArgs.model_validate(args.model_dump())
    if request.refresh or request.target_count > 2000:
        return (
            f"即将从Steam重新/大量获取最多{request.target_count}条评论，"
            "可能耗时并覆盖本地原始数据。是否继续？"
        )
    return None


def patch_fetch_approval(
    args: BaseModel,
    state: InvestigationState,
) -> str | None:
    request = PreparePatchesArgs.model_validate(args.model_dump())
    if request.refresh or request.target_count > 300:
        return (
            f"即将刷新最多{request.target_count}条Steam公告，"
            "可能覆盖本地公告数据。是否继续？"
        )
    return None


def classification_approval(
    args: BaseModel,
    state: InvestigationState,
) -> str | None:
    request = ClassifyArgs.model_validate(args.model_dump())
    if request.force or request.per_window_count > 30:
        total = request.per_window_count * 2
        return (
            f"即将{'强制重新' if request.force else ''}调用模型分类"
            f"最多{total}条评论，可能产生额外API费用。是否继续？"
        )
    return None


@contextmanager
def project_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class LegacyInvestigationTools:
    def __init__(
        self,
        project_root: Path,
        llm_client: OpenAICompatibleClient,
    ) -> None:
        self.root = project_root.resolve()
        self.llm_client = llm_client

    def registry(self) -> ToolRegistry:
        investigation = ReviewInvestigation(self.root)
        rag = RagTools(self.root)
        registry = ToolRegistry(
            [
                AgentTool(
                    "inspect_data",
                    "检查该AppID已有的评论、补丁、分类、聚合和报告产物；调查第一步应优先调用。",
                    AppArgs,
                    self.inspect_data,
                    execution_policy=ToolExecutionPolicy(
                        timeout_seconds=30,
                        concurrency_key="project_files",
                    ),
                ),
                AgentTool(
                    "prepare_reviews",
                    "获取或复用Steam评论，输出规范化评论文件。",
                    PrepareReviewsArgs,
                    self.prepare_reviews,
                    review_fetch_approval,
                    ToolExecutionPolicy(
                        timeout_seconds=180,
                        max_attempts=2,
                        retry_backoff_seconds=1,
                        concurrency_key="project_files",
                        retry_mode="transient_read",
                    ),
                ),
                AgentTool(
                    "prepare_patches",
                    "获取或复用Steam公告，并筛选可能的版本补丁。",
                    PreparePatchesArgs,
                    self.prepare_patches,
                    patch_fetch_approval,
                    ToolExecutionPolicy(
                        timeout_seconds=180,
                        max_attempts=2,
                        retry_backoff_seconds=1,
                        concurrency_key="project_files",
                        retry_mode="transient_read",
                    ),
                ),
                AgentTool(
                    "analyze_patch",
                    "基于评论覆盖范围选择补丁并建立更新前后统计窗口；需要评论和补丁文件。",
                    AnalyzePatchArgs,
                    self.analyze_patch,
                    execution_policy=ToolExecutionPolicy(
                        timeout_seconds=90,
                        concurrency_key="project_files",
                    ),
                ),
                AgentTool(
                    "classify_review_sample",
                    "调用LLM分类补丁前后评论样本；需要版本窗口分析。",
                    ClassifyArgs,
                    self.classify_review_sample,
                    classification_approval,
                    ToolExecutionPolicy(
                        timeout_seconds=900,
                        max_attempts=1,
                        concurrency_key="project_files",
                    ),
                ),
                AgentTool(
                    "aggregate_impact",
                    "聚合分类结果，计算情绪、主题、严重度和可操作性变化。",
                    AppArgs,
                    self.aggregate_impact,
                    execution_policy=ToolExecutionPolicy(
                        timeout_seconds=90,
                        concurrency_key="project_files",
                    ),
                ),
                AgentTool(
                    "generate_report",
                    "根据统计、分类和聚合结果生成可审计的Markdown报告。",
                    AppArgs,
                    self.generate_report,
                    execution_policy=ToolExecutionPolicy(
                        timeout_seconds=90,
                        concurrency_key="project_files",
                    ),
                ),
                AgentTool(
                    "assess_evidence",
                    "检查报告样本量、覆盖警告和平均置信度，结束调查前必须调用。",
                    AppArgs,
                    self.assess_evidence,
                    execution_policy=ToolExecutionPolicy(
                        timeout_seconds=90,
                        concurrency_key="project_files",
                    ),
                ),
                AgentTool("query_reviews","按时间、语言、好差评、评论时游玩时长和关键词筛选原始评论，返回全部匹配统计和带ID原文样例；只代表本地快照。",QueryReviewsArgs,investigation.query_reviews),
                AgentTool("get_review","根据review_id读取完整原文和元数据，核对引用；超长原文在模型上下文中可能截断。",GetReviewArgs,investigation.get_review),
                AgentTool("compare_cohorts","按两组独立筛选条件比较玩家群体，返回各组统计与重叠数，不做因果推断。",CompareCohortsArgs,investigation.compare_cohorts),
                AgentTool("compare_versions","按补丁ID比较两个版本的相同筛选窗口，返回样本数、统计和窗口重叠数。",CompareVersionsArgs,investigation.compare_versions),
                AgentTool("verify_investigation","筛选、检索或对比后必须调用：核对快照哈希，生成专属证据报告并通过完成门禁；不需要LLM分类。",VerifyInvestigationArgs,investigation.verify_investigation),
                AgentTool("list_patches","列出本地最近补丁ID、日期、标题和公告原文；多版本调查先确认范围，缺失时先prepare_patches。",ListPatchesArgs,investigation.list_patches),
                AgentTool("investigate_recent_versions","检查近N次补丁及各版本的缓存评论覆盖，返回不重叠窗口和可供RAG逐版本使用的时间筛选条件。不能证明同一问题持续存在，之后需原文调查并verify_investigation。",RecentVersionsArgs,investigation.investigate_recent_versions),
                AgentTool("build_game_index","按AppID建立或增量更新免费本地RAG索引；资料未变化时复用；先获取评论/公告。首次模型下载需要审批。",IndexArgs,rag.build_game_index,
                          approval_policy=rag.approval,execution_policy=ToolExecutionPolicy(timeout_seconds=1800,concurrency_key="project_files")),
                AgentTool("search_game_evidence","对已有且最新的索引做语义+关键词混合检索。灵活语义、跨版本原文查找时使用；缺失/过期先build_game_index。候选不是统计样本，不能计算总体占比。",SearchArgs,rag.search_game_evidence,
                          execution_policy=ToolExecutionPolicy(timeout_seconds=180,concurrency_key="project_files")),
            ],
            receipt_root=self.root / "data" / "tool_receipts",
        )
        from dataclasses import replace
        from gamelens.tools.query_reuse import ReadonlySnapshot
        readonly = {'list_patches','query_reviews','get_review','compare_cohorts',
                    'compare_versions','investigate_recent_versions'}
        query_snapshot = ReadonlySnapshot(self.root)
        search_snapshot = ReadonlySnapshot(self.root, include_index=True)
        tools = [replace(tool, readonly_snapshot=search_snapshot if tool.name == 'search_game_evidence'
                         else query_snapshot if tool.name in readonly else None) for tool in registry.tools]
        return ToolRegistry(tools, receipt_root=registry.receipt_root)

    def _config(self, state: InvestigationState, **overrides) -> AgentConfig:
        values = {
            "appid": state["appid"],
            "window_days": state["window_days"],
            "review_count": state["review_count"],
            "news_count": state["news_count"],
            "classify_count": state["classify_count"],
            "patch_news_id": state.get("patch_news_id"),
            "refresh_data": state["refresh_data"],
            "force_reclassify": state["force_reclassify"],
        }
        values.update(overrides)
        return AgentConfig(**values)

    def inspect_data(
        self,
        args: AppArgs,
        state: InvestigationState,
    ) -> ToolResult:
        raw = self.root / "data" / "raw"
        processed = self.root / "data" / "processed"
        artifacts: dict[str, str] = {}
        candidates = {
            "reviews": raw / f"{args.appid}_reviews.json",
            "patches": raw / f"{args.appid}_patch_candidates.json",
        }
        for name, path in candidates.items():
            if path.exists():
                artifacts[name] = str(path)

        reports = sorted(
            processed.glob(f"{args.appid}_version_analysis_*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        inherited_report = state.get("artifacts", {}).get("version_analysis")
        if inherited_report and Path(inherited_report).is_file():
            inherited = load_report(Path(inherited_report))
            if inherited.appid == args.appid:
                reports = [Path(inherited_report)]
        if reports:
            report_path = reports[0]
            report = load_report(report_path)
            news_id = report.patch.news_id
            artifacts["version_analysis"] = str(report_path)
            downstream = {
                "classifications": processed
                / f"{args.appid}_review_classifications_{news_id}.json",
                "impact_summary": processed
                / f"{args.appid}_review_impact_{news_id}.json",
                "final_report": self.root
                / "reports"
                / f"{args.appid}_{news_id}_impact_report.md",
            }
            for name, path in downstream.items():
                if path.exists():
                    artifacts[name] = str(path)

        return ToolResult(
            ok=True,
            summary=(
                "已检查本地数据；现有产物："
                + ("、".join(sorted(artifacts)) or "无")
            ),
            data={"available": sorted(artifacts)},
            artifacts=artifacts,
        )

    def prepare_reviews(
        self,
        args: PrepareReviewsArgs,
        state: InvestigationState,
    ) -> ToolResult:
        config = self._config(
            state,
            review_count=args.target_count,
            refresh_data=args.refresh,
        )
        with project_directory(self.root):
            reviews = ensure_reviews(config)
        path = self.root / "data" / "raw" / f"{args.appid}_reviews.json"
        return ToolResult(
            ok=True,
            summary=f"评论数据准备完成，共{len(reviews)}条",
            data={"review_count": len(reviews)},
            artifacts={"reviews": str(path)},
        )

    def prepare_patches(
        self,
        args: PreparePatchesArgs,
        state: InvestigationState,
    ) -> ToolResult:
        config = self._config(
            state,
            news_count=args.target_count,
            refresh_data=args.refresh,
        )
        with project_directory(self.root):
            patches = ensure_patch_candidates(config)
        path = (
            self.root
            / "data"
            / "raw"
            / f"{args.appid}_patch_candidates.json"
        )
        return ToolResult(
            ok=True,
            summary=f"补丁候选准备完成，共{len(patches)}条",
            data={"patch_count": len(patches)},
            artifacts={"patches": str(path)},
        )

    def analyze_patch(
        self,
        args: AnalyzePatchArgs,
        state: InvestigationState,
    ) -> ToolResult:
        artifacts = state.get("artifacts", {})
        if "reviews" not in artifacts or "patches" not in artifacts:
            return ToolResult(
                ok=False,
                summary="缺少评论或补丁文件，请先准备两类数据",
            )
        reviews = load_reviews(Path(artifacts["reviews"]))
        patches = load_patch_candidates(Path(artifacts["patches"]))
        config = self._config(
            state,
            window_days=args.window_days,
            patch_news_id=args.patch_news_id or state.get("patch_news_id"),
        )
        patch = select_patch(config, reviews, patches)
        report = build_version_impact_report(
            reviews,
            patch,
            args.window_days,
        )
        with project_directory(self.root):
            path = save_report(report).resolve()
        return ToolResult(
            ok=True,
            summary=(
                f"已选择补丁“{patch.title}”，建立前后各"
                f"{args.window_days}天的分析窗口"
            ),
            data={
                "news_id": patch.news_id,
                "patch_title": patch.title,
                "before_reviews": report.before.total_reviews,
                "after_reviews": report.after.total_reviews,
                "warnings": report.warnings,
            },
            artifacts={"version_analysis": str(path)},
        )

    def classify_review_sample(
        self,
        args: ClassifyArgs,
        state: InvestigationState,
    ) -> ToolResult:
        artifacts = state.get("artifacts", {})
        if "reviews" not in artifacts or "version_analysis" not in artifacts:
            return ToolResult(
                ok=False,
                summary="缺少评论或版本窗口分析，无法执行分类",
            )
        reviews = load_reviews(Path(artifacts["reviews"]))
        report = load_report(Path(artifacts["version_analysis"]))
        existing = [] if args.force else load_existing_classifications(report)
        with project_directory(self.root):
            records = classify_reviews(
                self.llm_client,
                reviews,
                report,
                args.per_window_count,
                existing_records=existing,
            )
            path = save_classifications(report, records).resolve()
        return ToolResult(
            ok=True,
            summary=f"评论分类完成，共得到{len(records)}条结构化结果",
            data={"classified_count": len(records)},
            artifacts={"classifications": str(path)},
        )

    def aggregate_impact(
        self,
        args: AppArgs,
        state: InvestigationState,
    ) -> ToolResult:
        artifacts = state.get("artifacts", {})
        if (
            "version_analysis" not in artifacts
            or "classifications" not in artifacts
        ):
            return ToolResult(
                ok=False,
                summary="缺少版本分析或分类结果，无法聚合",
            )
        report = load_report(Path(artifacts["version_analysis"]))
        records = load_classifications(Path(artifacts["classifications"]))
        impact = build_impact_summary(report, records)
        with project_directory(self.root):
            path = save_impact_summary(impact).resolve()
        return ToolResult(
            ok=True,
            summary="版本影响聚合完成",
            data={
                "before_sample": impact.before.review_count,
                "after_sample": impact.after.review_count,
                "warnings": impact.warnings,
            },
            artifacts={"impact_summary": str(path)},
        )

    def generate_report(
        self,
        args: AppArgs,
        state: InvestigationState,
    ) -> ToolResult:
        artifacts = state.get("artifacts", {})
        required = {"version_analysis", "classifications", "impact_summary"}
        missing = sorted(required - set(artifacts))
        if missing:
            return ToolResult(
                ok=False,
                summary=f"生成报告缺少产物：{', '.join(missing)}",
            )
        report = load_report(Path(artifacts["version_analysis"]))
        records = load_classifications(Path(artifacts["classifications"]))
        impact = load_impact_summary(Path(artifacts["impact_summary"]))
        markdown = generate_markdown(report, impact, records)
        with project_directory(self.root):
            path = save_markdown_report(report, markdown).resolve()
        return ToolResult(
            ok=True,
            summary=f"Markdown调查报告已生成：{path.name}",
            artifacts={"final_report": str(path)},
        )

    def assess_evidence(
        self,
        args: AppArgs,
        state: InvestigationState,
    ) -> ToolResult:
        artifacts = state.get("artifacts", {})
        required = {
            "version_analysis",
            "classifications",
            "impact_summary",
            "final_report",
        }
        missing = sorted(required - set(artifacts))
        if missing:
            return ToolResult(
                ok=False,
                summary=f"证据评估缺少产物：{', '.join(missing)}",
            )
        report = load_report(Path(artifacts["version_analysis"]))
        records = load_classifications(Path(artifacts["classifications"]))
        impact = load_impact_summary(Path(artifacts["impact_summary"]))
        total = impact.before.review_count + impact.after.review_count
        confidence = impact.after.average_confidence
        sufficient = total >= 20 and (confidence is None or confidence >= 0.7)
        topic_changes = [
            {
                "topic": item.topic.value,
                "before_count": item.before_count,
                "after_count": item.after_count,
                "change_percentage_points": round(
                    item.change_percentage_points,
                    2,
                ),
            }
            for item in impact.topic_changes
            if item.before_count or item.after_count
        ][:8]
        representative_evidence = [
            {
                "review_id": record.review.review_id,
                "sentiment": record.analysis.sentiment,
                "topics": [topic.value for topic in record.analysis.topics],
                "summary": record.analysis.summary,
                "evidence": record.analysis.evidence,
                "severity": record.analysis.severity,
            }
            for record in records
            if record.period == "after"
            and (
                record.analysis.update_related
                or record.analysis.is_actionable
            )
        ][:5]
        return ToolResult(
            ok=True,
            summary=(
                f"证据质量检查完成：分类样本{total}条，"
                f"更新后平均置信度"
                f"{confidence if confidence is not None else 'N/A'}；"
                f"{'达到' if sufficient else '未达到'}样本/自评启发式阈值（不是正确率）"
            ),
            data={
                "sample_count": total,
                "average_confidence": confidence,
                "sufficient": sufficient,
                "assessment_kind": "sample_and_self_confidence_heuristic",
                "semantic_accuracy_verified": False,
                "warnings": impact.warnings,
                "facts": {
                    "patch_title": report.patch.title,
                    "full_steam_window": {
                        "before_review_count": report.before.total_reviews,
                        "after_review_count": report.after.total_reviews,
                        "before_positive_rate": report.before.positive_rate,
                        "after_positive_rate": report.after.positive_rate,
                        "positive_rate_change_percentage_points": (
                            report.positive_rate_change_percentage_points
                        ),
                        "negative_rate_change_percentage_points": (
                            report.negative_rate_change_percentage_points
                        ),
                    },
                    "classified_sample": {
                        "before_sentiment_rates": (
                            impact.before.sentiment_rates
                        ),
                        "after_sentiment_rates": (
                            impact.after.sentiment_rates
                        ),
                        "topic_changes": topic_changes,
                    },
                    "representative_after_evidence": (
                        representative_evidence
                    ),
                },
            },
            artifacts={"evidence_assessed": "true"},
        )
