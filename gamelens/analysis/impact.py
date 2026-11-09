import json
from collections import Counter
from pathlib import Path
from gamelens.storage.atomic_io import atomic_write_json
from statistics import fmean

from pydantic import ValidationError

from gamelens.domain.models import (
    ClassifiedPeriodStats,
    ClassifiedReview,
    ReviewTopic,
    TopicChange,
    VersionImpactReport,
    VersionReviewImpactSummary,
)
from gamelens.analysis.classifier import load_report


SENTIMENTS = ("positive", "neutral", "negative", "mixed")


def load_classifications(path: Path) -> list[ClassifiedReview]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError("评论分类文件的最外层必须是数组")
    return [ClassifiedReview.model_validate(item) for item in payload]


def safe_rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def summarize_period(
    records: list[ClassifiedReview],
) -> ClassifiedPeriodStats:
    review_count = len(records)
    sentiment_counter = Counter(
        record.analysis.sentiment for record in records
    )
    topic_counter = Counter(
        topic
        for record in records
        for topic in record.analysis.topics
    )
    actionable_count = sum(
        record.analysis.is_actionable for record in records
    )
    update_related_count = sum(
        record.analysis.update_related for record in records
    )

    return ClassifiedPeriodStats(
        review_count=review_count,
        sentiment_counts={
            sentiment: sentiment_counter[sentiment]
            for sentiment in SENTIMENTS
        },
        sentiment_rates={
            sentiment: safe_rate(
                sentiment_counter[sentiment],
                review_count,
            )
            for sentiment in SENTIMENTS
        },
        topic_counts={
            topic: topic_counter[topic]
            for topic in ReviewTopic
        },
        topic_rates={
            topic: safe_rate(topic_counter[topic], review_count)
            for topic in ReviewTopic
        },
        average_severity=(
            fmean(record.analysis.severity for record in records)
            if records
            else None
        ),
        actionable_count=actionable_count,
        actionable_rate=(
            safe_rate(actionable_count, review_count)
            if records
            else None
        ),
        update_related_count=update_related_count,
        update_related_rate=(
            safe_rate(update_related_count, review_count)
            if records
            else None
        ),
        average_confidence=(
            fmean(record.analysis.confidence for record in records)
            if records
            else None
        ),
    )


def build_impact_summary(
    report: VersionImpactReport,
    records: list[ClassifiedReview],
) -> VersionReviewImpactSummary:
    record_keys = [
        (record.period, record.review.review_id)
        for record in records
    ]
    if len(record_keys) != len(set(record_keys)):
        raise ValueError("评论分类结果中存在重复的时期与review_id")

    before_records = [
        record for record in records if record.period == "before"
    ]
    after_records = [
        record for record in records if record.period == "after"
    ]
    before = summarize_period(before_records)
    after = summarize_period(after_records)

    topic_changes = [
        TopicChange(
            topic=topic,
            before_count=before.topic_counts[topic],
            after_count=after.topic_counts[topic],
            before_rate=before.topic_rates[topic],
            after_rate=after.topic_rates[topic],
            change_percentage_points=(
                after.topic_rates[topic]
                - before.topic_rates[topic]
            )
            * 100,
        )
        for topic in ReviewTopic
    ]
    topic_changes.sort(
        key=lambda item: abs(item.change_percentage_points),
        reverse=True,
    )
    leading_increases = [
        item.topic
        for item in sorted(
            topic_changes,
            key=lambda item: item.change_percentage_points,
            reverse=True,
        )
        if item.change_percentage_points > 0
    ][:3]
    leading_decreases = [
        item.topic
        for item in sorted(
            topic_changes,
            key=lambda item: item.change_percentage_points,
        )
        if item.change_percentage_points < 0
    ][:3]

    warnings = list(report.warnings)
    minimum_period_size = min(
        before.review_count,
        after.review_count,
    )
    if minimum_period_size < 30:
        warnings.append(
            "LLM分类样本少于每个窗口30条，主题变化只可视为探索性信号。"
        )
    if before.review_count == 0 or after.review_count == 0:
        warnings.append(
            "至少一个时间窗口没有分类样本，无法可靠比较更新前后变化。"
        )

    return VersionReviewImpactSummary(
        appid=report.appid,
        news_id=report.patch.news_id,
        patch_title=report.patch.title,
        patch_published_at=report.patch.published_at,
        classified_review_count=len(records),
        before=before,
        after=after,
        topic_changes=topic_changes,
        leading_topic_increases=leading_increases,
        leading_topic_decreases=leading_decreases,
        warnings=warnings,
        notes=[
            "主题允许多标签，因此同一窗口的主题比例之和可能超过100%。",
            "主题变化以各窗口分类样本数为分母，单位为百分点。",
            "完整窗口好评率仍以版本分析报告中的Steam推荐数据为准。",
        ],
    )


def save_impact_summary(
    summary: VersionReviewImpactSummary,
) -> Path:
    output_path = Path("data") / "processed" / (
        f"{summary.appid}_review_impact_{summary.news_id}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_path,summary.model_dump(mode="json"))
    return output_path


def main() -> None:
    appid_text = input("请输入Steam游戏AppID：").strip()
    if not appid_text.isdigit() or int(appid_text) <= 0:
        print("AppID必须是正整数")
        return
    appid = int(appid_text)

    report_files = sorted(
        (Path("data") / "processed").glob(
            f"{appid}_version_analysis_*.json"
        )
    )
    if not report_files:
        print("没有版本分析报告，请先运行 python -m gamelens.analysis.version。")
        return

    print("可选择的版本分析报告：")
    for index, path in enumerate(report_files, start=1):
        report = load_report(path)
        print(
            f"[{index}] {report.patch.published_at.date()} | "
            f"{report.patch.title}"
        )

    selection_text = input("请选择报告编号：").strip()
    if not selection_text.isdigit():
        print("报告编号必须是数字")
        return
    selection = int(selection_text)
    if not 1 <= selection <= len(report_files):
        print("报告编号超出范围")
        return

    try:
        report = load_report(report_files[selection - 1])
        classification_path = Path("data") / "processed" / (
            f"{appid}_review_classifications_"
            f"{report.patch.news_id}.json"
        )
        if not classification_path.exists():
            print("没有评论分类结果，请先运行 python -m gamelens.analysis.classifier。")
            return

        records = load_classifications(classification_path)
        summary = build_impact_summary(report, records)
        output_path = save_impact_summary(summary)
    except (OSError, ValueError, ValidationError) as error:
        print(f"影响聚合失败：{error}")
        return

    print(f"已聚合{summary.classified_review_count}条分类结果")
    print(
        "更新后相关评论比例："
        f"{(summary.after.update_related_rate or 0) * 100:.1f}%"
    )
    print(
        "上升主题："
        + (", ".join(summary.leading_topic_increases) or "无")
    )
    print(
        "下降主题："
        + (", ".join(summary.leading_topic_decreases) or "无")
    )
    print(f"结果已保存：{output_path.resolve()}")


if __name__ == "__main__":
    main()
