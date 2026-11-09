import json
from datetime import datetime, timezone
from pathlib import Path
from gamelens.storage.atomic_io import atomic_write_json
from gamelens.core.local_time import local_datetime, shift_calendar_days

from pydantic import ValidationError

from gamelens.domain.models import (
    ReviewWindowStats,
    SteamNewsItem,
    SteamReview,
    VersionImpactReport,
)


def timestamp_to_datetime(timestamp: int) -> datetime:
    return local_datetime(timestamp)


def load_report(path: Path) -> VersionImpactReport:
    """Window metadata loading does not depend on the optional LLM classifier."""
    return VersionImpactReport.model_validate_json(path.read_text(encoding="utf-8"))


def load_reviews(path: Path) -> list[SteamReview]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("评论JSON最外层必须是列表")

    return [SteamReview.model_validate(item) for item in data]


def load_patch_candidates(path: Path) -> list[SteamNewsItem]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("补丁候选JSON最外层必须是列表")

    candidates = [
        SteamNewsItem.model_validate(item)
        for item in data
    ]
    return sorted(
        candidates,
        key=lambda item: item.published_at_timestamp,
        reverse=True,
    )


def calculate_window_stats(
    reviews: list[SteamReview],
    start_timestamp: int,
    end_timestamp: int,
    coverage_complete: bool,
) -> ReviewWindowStats:
    selected_reviews = [
        review
        for review in reviews
        if start_timestamp
        <= review.created_at_timestamp
        < end_timestamp
    ]

    total_reviews = len(selected_reviews)
    positive_reviews = sum(
        review.is_positive for review in selected_reviews
    )
    negative_reviews = total_reviews - positive_reviews

    if total_reviews:
        positive_rate = positive_reviews / total_reviews
        negative_rate = negative_reviews / total_reviews
        average_playtime_hours = (
            sum(
                review.playtime_at_review_minutes
                for review in selected_reviews
            )
            / total_reviews
            / 60
        )
    else:
        positive_rate = None
        negative_rate = None
        average_playtime_hours = None

    return ReviewWindowStats(
        start_at=timestamp_to_datetime(start_timestamp),
        end_at=timestamp_to_datetime(end_timestamp),
        total_reviews=total_reviews,
        positive_reviews=positive_reviews,
        negative_reviews=negative_reviews,
        positive_rate=positive_rate,
        negative_rate=negative_rate,
        average_playtime_hours=average_playtime_hours,
        coverage_complete=coverage_complete,
    )


def build_version_impact_report(
    reviews: list[SteamReview],
    patch: SteamNewsItem,
    window_days: int,
) -> VersionImpactReport:
    if not reviews:
        raise ValueError("评论列表不能为空")

    if window_days <= 0:
        raise ValueError("window_days必须是正整数")

    patch_timestamp = patch.published_at_timestamp

    before_start = shift_calendar_days(patch_timestamp, -window_days)
    before_end = patch_timestamp
    after_start = patch_timestamp
    after_end = shift_calendar_days(patch_timestamp, window_days)

    minimum_review_timestamp = min(
        review.created_at_timestamp for review in reviews
    )
    maximum_review_timestamp = max(
        review.created_at_timestamp for review in reviews
    )

    before_coverage_complete = (
        minimum_review_timestamp <= before_start
        and maximum_review_timestamp >= before_end
    )
    after_coverage_complete = maximum_review_timestamp >= after_end

    before = calculate_window_stats(
        reviews,
        before_start,
        before_end,
        before_coverage_complete,
    )
    after = calculate_window_stats(
        reviews,
        after_start,
        after_end,
        after_coverage_complete,
    )

    if before.positive_rate is None or after.positive_rate is None:
        positive_rate_change = None
        negative_rate_change = None
    else:
        positive_rate_change = round(
            (after.positive_rate - before.positive_rate) * 100,
            2,
        )
        negative_rate_change = round(
            (after.negative_rate - before.negative_rate) * 100,
            2,
        )

    warnings: list[str] = []
    if not before.coverage_complete:
        warnings.append("评论数据未完整覆盖更新前时间窗口")
    if not after.coverage_complete:
        warnings.append("评论数据未完整覆盖更新后时间窗口")
    if before.total_reviews < 30:
        warnings.append("更新前样本少于30条，统计波动可能较大")
    if after.total_reviews < 30:
        warnings.append("更新后样本少于30条，统计波动可能较大")

    return VersionImpactReport(
        appid=patch.appid,
        patch=patch,
        window_days=window_days,
        before=before,
        after=after,
        review_count_change=after.total_reviews - before.total_reviews,
        positive_rate_change_percentage_points=positive_rate_change,
        negative_rate_change_percentage_points=negative_rate_change,
        warnings=warnings,
    )


def save_report(report: VersionImpactReport) -> Path:
    output_directory = Path("data") / "processed"
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / (
        f"{report.appid}_version_analysis_{report.patch.news_id}.json"
    )

    atomic_write_json(output_path,report.model_dump(mode="json"))

    return output_path


def format_rate(rate: float | None) -> str:
    if rate is None:
        return "N/A"
    return f"{rate * 100:.2f}%"


def main() -> None:
    appid_text = input("请输入Steam游戏AppID：").strip()
    if not appid_text.isdigit() or int(appid_text) <= 0:
        print("AppID必须是正整数")
        return

    window_text = input(
        "请输入分析窗口天数（直接回车默认7天）："
    ).strip()
    if window_text:
        if not window_text.isdigit() or int(window_text) <= 0:
            print("窗口天数必须是正整数")
            return
        window_days = int(window_text)
    else:
        window_days = 7

    appid = int(appid_text)
    raw_directory = Path("data") / "raw"
    reviews_path = raw_directory / f"{appid}_reviews.json"
    patches_path = raw_directory / f"{appid}_patch_candidates.json"

    missing_paths = [
        path for path in (reviews_path, patches_path) if not path.exists()
    ]
    if missing_paths:
        print("缺少以下数据文件：")
        for path in missing_paths:
            print(f"- {path.resolve()}")
        return

    try:
        reviews = load_reviews(reviews_path)
        patches = load_patch_candidates(patches_path)
    except (OSError, ValueError, ValidationError) as error:
        print(f"读取数据失败：{error}")
        return

    if not patches:
        print("没有找到补丁候选，请先运行 python -m gamelens.tools.steam_news。")
        return

    minimum_review_timestamp = min(
        review.created_at_timestamp for review in reviews
    )
    maximum_review_timestamp = max(
        review.created_at_timestamp for review in reviews
    )
    window_seconds = window_days * 24 * 60 * 60

    print("\n可选择的补丁候选：")
    for index, patch in enumerate(patches[:20], start=1):
        complete = (
            minimum_review_timestamp
            <= patch.published_at_timestamp - window_seconds
            and maximum_review_timestamp
            >= patch.published_at_timestamp + window_seconds
        )
        status = "完整窗口" if complete else "数据可能不完整"
        print(
            f"[{index}] {patch.published_at.date()} | "
            f"{patch.title} | {status}"
        )

    selection_text = input("请选择补丁编号：").strip()
    if not selection_text.isdigit():
        print("补丁编号必须是数字")
        return

    selection = int(selection_text)
    visible_patches = patches[:20]
    if not 1 <= selection <= len(visible_patches):
        print("补丁编号超出范围")
        return

    patch = visible_patches[selection - 1]
    report = build_version_impact_report(
        reviews,
        patch,
        window_days,
    )
    output_path = save_report(report)

    print(f"\n补丁：{report.patch.title}")
    print(f"发布时间：{report.patch.published_at}")
    print(f"分析窗口：更新前后各{report.window_days}天")
    print("\n更新前：")
    print(f"  评论数：{report.before.total_reviews}")
    print(f"  好评率：{format_rate(report.before.positive_rate)}")
    print(f"  差评率：{format_rate(report.before.negative_rate)}")
    print("更新后：")
    print(f"  评论数：{report.after.total_reviews}")
    print(f"  好评率：{format_rate(report.after.positive_rate)}")
    print(f"  差评率：{format_rate(report.after.negative_rate)}")
    print("变化：")
    print(f"  评论数变化：{report.review_count_change:+d}")

    positive_change = report.positive_rate_change_percentage_points
    negative_change = report.negative_rate_change_percentage_points
    if positive_change is not None and negative_change is not None:
        print(f"  好评率变化：{positive_change:+.2f}个百分点")
        print(f"  差评率变化：{negative_change:+.2f}个百分点")
    else:
        print("  好差评率变化：N/A")

    if report.warnings:
        print("\n注意：")
        for warning in report.warnings:
            print(f"- {warning}")

    print(f"\n报告已保存：{output_path.resolve()}")


if __name__ == "__main__":
    main()
