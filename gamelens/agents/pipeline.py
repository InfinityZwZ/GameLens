import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import httpx
from pydantic import ValidationError

from gamelens.analysis.impact import (
    build_impact_summary,
    load_classifications,
    save_impact_summary,
)
from gamelens.core.llm import LLMConfigurationError, OpenAICompatibleClient
from gamelens.domain.models import ClassifiedReview, SteamNewsItem, SteamReview
from gamelens.analysis.report import generate_markdown, save_markdown_report
from gamelens.analysis.classifier import (
    LLMResponseError,
    classification_output_path,
    classify_reviews,
    load_existing_classifications,
    save_classifications,
    select_window_reviews,
)
from gamelens.tools.steam_news import fetch_news, normalize_news, save_news
from gamelens.tools.steam_reviews import prepare_review_snapshot
from gamelens.analysis.version import (
    build_version_impact_report,
    load_patch_candidates,
    save_report,
)


from gamelens.core.paths import PROJECT_ROOT


@dataclass(frozen=True)
class AgentConfig:
    appid: int
    window_days: int
    review_count: int
    news_count: int
    classify_count: int
    patch_news_id: str | None
    refresh_data: bool
    force_reclassify: bool


def parse_args() -> AgentConfig:
    parser = argparse.ArgumentParser(
        description=(
            "游鉴 GameLens 确定性分析管线：自动抓取、选择补丁、"
            "分类、聚合并生成Markdown报告。"
        )
    )
    parser.add_argument("--appid", type=int, help="Steam游戏AppID")
    parser.add_argument("--window-days", type=int, default=7)
    parser.add_argument("--review-count", type=int, default=5000)
    parser.add_argument("--news-count", type=int, default=100)
    parser.add_argument("--classify-count", type=int, default=10)
    parser.add_argument(
        "--patch-news-id",
        help="指定Steam公告news_id；省略时自动选择最近的完整窗口补丁",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="忽略已有原始数据，重新请求Steam接口",
    )
    parser.add_argument(
        "--force-reclassify",
        action="store_true",
        help="忽略已有分类检查点，重新调用LLM",
    )
    args = parser.parse_args()

    appid = args.appid
    if appid is None:
        appid_text = input("请输入Steam游戏AppID：").strip()
        if not appid_text.isdigit():
            parser.error("AppID必须是正整数")
        appid = int(appid_text)

    if appid <= 0:
        parser.error("AppID必须是正整数")
    if args.window_days <= 0:
        parser.error("window-days必须是正整数")
    if not 1 <= args.review_count <= 5000:
        parser.error("review-count必须在1到5000之间")
    if not 1 <= args.news_count <= 1000:
        parser.error("news-count必须在1到1000之间")
    if args.classify_count <= 0:
        parser.error("classify-count必须是正整数")

    return AgentConfig(
        appid=appid,
        window_days=args.window_days,
        review_count=args.review_count,
        news_count=args.news_count,
        classify_count=args.classify_count,
        patch_news_id=args.patch_news_id,
        refresh_data=args.refresh_data,
        force_reclassify=args.force_reclassify,
    )


def ensure_reviews(config: AgentConfig) -> list[SteamReview]:
    path = Path("data") / "raw" / f"{config.appid}_reviews.json"
    prepared=prepare_review_snapshot(path,config.appid,language="schinese",
        target_count=config.review_count,refresh=config.refresh_data)
    reviews=prepared["reviews"]
    if prepared["reused"] and not prepared["refresh_attempted"]:
        print(f"[1/6] 复用充分的评论数据：{len(reviews)}条")
    elif prepared["retained_previous_snapshot"]:
        print(f"[1/6] 新采集未改善缓存，保留原评论数据：{len(reviews)}条")
    else:
        print(f"[1/6] 评论数据已重新采集：{path.resolve()}（{len(reviews)}条）")
    return reviews


def ensure_patch_candidates(config: AgentConfig) -> list[SteamNewsItem]:
    path = (
        Path("data")
        / "raw"
        / f"{config.appid}_patch_candidates.json"
    )
    if path.exists() and not config.refresh_data:
        patches = load_patch_candidates(path)
        print(f"[2/6] 复用补丁候选：{len(patches)}条")
        return patches

    print(f"[2/6] 正在从Steam获取{config.news_count}条公告……")
    raw_news = fetch_news(config.appid, config.news_count)
    news_items = [
        normalize_news(item, config.appid) for item in raw_news
    ]
    _, path = save_news(config.appid, news_items)
    patches = [item for item in news_items if item.is_patch_candidate]
    patches.sort(
        key=lambda item: item.published_at_timestamp,
        reverse=True,
    )
    print(f"[2/6] 补丁候选已保存：{path.resolve()}")
    return patches


def select_patch(
    config: AgentConfig,
    reviews: list[SteamReview],
    patches: list[SteamNewsItem],
) -> SteamNewsItem:
    if not patches:
        raise ValueError("没有补丁候选可供分析")

    if config.patch_news_id:
        for patch in patches:
            if patch.news_id == config.patch_news_id:
                print(f"[3/6] 使用指定补丁：{patch.title}")
                return patch
        raise ValueError(
            f"没有找到news_id={config.patch_news_id}的补丁"
        )

    minimum_timestamp = min(
        review.created_at_timestamp for review in reviews
    )
    maximum_timestamp = max(
        review.created_at_timestamp for review in reviews
    )
    window_seconds = config.window_days * 24 * 60 * 60
    complete_patches = [
        patch
        for patch in patches
        if minimum_timestamp
        <= patch.published_at_timestamp - window_seconds
        and maximum_timestamp
        >= patch.published_at_timestamp + window_seconds
    ]

    patch = complete_patches[0] if complete_patches else patches[0]
    reason = (
        "最近且评论窗口完整"
        if complete_patches
        else "没有完整窗口，回退到最近补丁"
    )
    print(
        f"[3/6] Agent自动选择补丁：{patch.title} "
        f"({patch.published_at.date()}，{reason})"
    )
    return patch


def pending_classification_count(
    records: list[ClassifiedReview],
    reviews: list[SteamReview],
    report,
    classify_count: int,
) -> int:
    completed = {
        (record.period, record.review.review_id)
        for record in records
    }
    pending = 0
    for period, seed in (("before", 42), ("after", 43)):
        selected = select_window_reviews(
            reviews,
            report,
            period,
            classify_count,
            seed,
        )
        pending += sum(
            (period, review.review_id) not in completed
            for review in selected
        )
    return pending


def run_agent(config: AgentConfig) -> Path:
    reviews = ensure_reviews(config)
    patches = ensure_patch_candidates(config)
    patch = select_patch(config, reviews, patches)

    report = build_version_impact_report(
        reviews,
        patch,
        config.window_days,
    )
    report_path = save_report(report)
    print(f"[3/6] 版本窗口分析已保存：{report_path.resolve()}")

    if config.force_reclassify:
        existing_records: list[ClassifiedReview] = []
    else:
        existing_records = load_existing_classifications(report)
    pending_count = pending_classification_count(
        existing_records,
        reviews,
        report,
        config.classify_count,
    )

    if pending_count:
        model = os.getenv("LLM_MODEL", "").strip()
        client = OpenAICompatibleClient(model=model)
        print(
            f"[4/6] 需要分类{pending_count}条评论，"
            f"使用模型：{model}"
        )
        records = classify_reviews(
            client,
            reviews,
            report,
            config.classify_count,
            existing_records=existing_records,
        )
        classification_path = save_classifications(report, records)
        print(f"[4/6] 分类结果已保存：{classification_path.resolve()}")
    else:
        records = existing_records
        classification_path = classification_output_path(report)
        print(
            f"[4/6] 分类检查点完整，跳过LLM调用："
            f"{classification_path.resolve()}"
        )

    impact = build_impact_summary(report, records)
    impact_path = save_impact_summary(impact)
    print(f"[5/6] 聚合结果已保存：{impact_path.resolve()}")

    markdown = generate_markdown(report, impact, records)
    markdown_path = save_markdown_report(report, markdown)
    print(f"[6/6] Markdown报告已生成：{markdown_path.resolve()}")
    return markdown_path


def main() -> None:
    os.chdir(PROJECT_ROOT)
    try:
        config = parse_args()
        output_path = run_agent(config)
    except (
        OSError,
        ValueError,
        RuntimeError,
        ValidationError,
        httpx.HTTPError,
        LLMConfigurationError,
        LLMResponseError,
    ) as error:
        print(f"Agent运行失败：{error}")
        raise SystemExit(1) from error

    print("\nAgent工作流完成。")
    print(f"最终报告：{output_path.resolve()}")


if __name__ == "__main__":
    main()
