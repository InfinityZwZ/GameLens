import json
from pathlib import Path
from gamelens.storage.atomic_io import atomic_write_text
from gamelens.core.local_time import local_iso

from pydantic import ValidationError

from gamelens.analysis.impact import load_classifications
from gamelens.domain.models import (
    ClassifiedReview,
    ReviewTopic,
    VersionImpactReport,
    VersionReviewImpactSummary,
)
from gamelens.analysis.classifier import load_report


TOPIC_LABELS = {
    ReviewTopic.PERFORMANCE: "性能与优化",
    ReviewTopic.CRASH_BUG: "崩溃与缺陷",
    ReviewTopic.SERVER_NETWORK: "服务器与网络",
    ReviewTopic.MATCHMAKING: "匹配机制",
    ReviewTopic.CHEATING: "外挂与反作弊",
    ReviewTopic.BALANCE: "平衡性",
    ReviewTopic.MONETIZATION: "商业化",
    ReviewTopic.UI_UX: "界面与交互",
    ReviewTopic.CONTENT: "内容",
    ReviewTopic.LOCALIZATION: "本地化",
    ReviewTopic.GAMEPLAY: "核心玩法",
    ReviewTopic.PRAISE: "正面评价",
    ReviewTopic.NOISE: "低信息内容",
    ReviewTopic.OTHER: "其他",
}

SENTIMENT_LABELS = {
    "positive": "正面",
    "neutral": "中性",
    "negative": "负面",
    "mixed": "褒贬混合",
}


def load_impact_summary(path: Path) -> VersionReviewImpactSummary:
    with path.open("r", encoding="utf-8") as file:
        return VersionReviewImpactSummary.model_validate(json.load(file))


def format_percent(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def format_pp(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:+.1f} pp"


def table_text(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def topic_names(topics: list[ReviewTopic]) -> str:
    return "、".join(TOPIC_LABELS[topic] for topic in topics)


def select_representative_reviews(
    records: list[ClassifiedReview],
    *,
    positive: bool,
    limit: int,
) -> list[ClassifiedReview]:
    after_records = [record for record in records if record.period == "after"]
    if positive:
        candidates = [
            record
            for record in after_records
            if record.analysis.sentiment == "positive"
        ]
    else:
        candidates = [
            record
            for record in after_records
            if record.analysis.sentiment in {"negative", "mixed"}
            or record.analysis.is_actionable
        ]

    return sorted(
        candidates,
        key=lambda record: (
            record.analysis.update_related,
            record.analysis.is_actionable,
            record.analysis.severity,
            record.review.votes_up,
            record.analysis.confidence,
        ),
        reverse=True,
    )[:limit]


def build_executive_summary(
    report: VersionImpactReport,
    impact: VersionReviewImpactSummary,
) -> list[str]:
    statements = [
        (
            "完整窗口的Steam好评率从"
            f"{format_percent(report.before.positive_rate)}变化至"
            f"{format_percent(report.after.positive_rate)}，"
            "变化"
            f"{format_pp(report.positive_rate_change_percentage_points)}。"
        ),
        (
            "LLM抽样中，负面评论由"
            f"{impact.before.sentiment_counts['negative']}条变化至"
            f"{impact.after.sentiment_counts['negative']}条，"
            "平均问题严重度由"
            f"{impact.before.average_severity or 0:.2f}变化至"
            f"{impact.after.average_severity or 0:.2f}。"
        ),
        (
            "更新后抽样评论中，"
            f"{format_percent(impact.after.update_related_rate)}"
            "被标记为与本次更新直接相关，"
            f"{impact.after.actionable_count}条包含可执行问题。"
        ),
    ]

    if impact.leading_topic_increases:
        statements.append(
            "上升最明显的主题为："
            f"{topic_names(impact.leading_topic_increases)}。"
        )
    return statements


def build_review_section(
    title: str,
    records: list[ClassifiedReview],
) -> list[str]:
    lines = [f"## {title}", ""]
    if not records:
        lines.extend(["本次抽样中没有符合条件的评论。", ""])
        return lines

    for index, record in enumerate(records, start=1):
        analysis = record.analysis
        lines.extend(
            [
                f"### {index}. {table_text(analysis.summary)}",
                "",
                (
                    f"- 评论ID：`{record.review.review_id}`"
                    f"；情绪：{SENTIMENT_LABELS[analysis.sentiment]}"
                    f"；主题：{topic_names(analysis.topics)}"
                ),
                (
                    f"- 严重度：{analysis.severity}/5"
                    f"；可执行：{'是' if analysis.is_actionable else '否'}"
                    f"；更新相关：{'是' if analysis.update_related else '否'}"
                    f"；置信度：{analysis.confidence:.2f}"
                ),
                "",
                f"> {analysis.evidence.replace(chr(10), ' ').strip()}",
                "",
            ]
        )
    return lines


def generate_markdown(
    report: VersionImpactReport,
    impact: VersionReviewImpactSummary,
    records: list[ClassifiedReview],
) -> str:
    lines = [
        f"# {report.patch.title}：玩家评论影响报告",
        "",
        "## 报告信息",
        "",
        f"- Steam AppID：`{report.appid}`",
        f"- 补丁发布时间（北京时间时间）：{local_iso(report.patch.published_at_timestamp)}",
        f"- 分析窗口：更新前后各 {report.window_days} 天",
        (
            f"- 完整窗口评论量：更新前 {report.before.total_reviews} 条，"
            f"更新后 {report.after.total_reviews} 条"
        ),
        (
            f"- LLM分类样本：更新前 {impact.before.review_count} 条，"
            f"更新后 {impact.after.review_count} 条"
        ),
        f"- [Steam补丁原文]({report.patch.url})",
        "",
        "## 执行摘要",
        "",
    ]
    lines.extend(
        f"- {statement}"
        for statement in build_executive_summary(report, impact)
    )

    lines.extend(
        [
            "",
            "## 核心指标",
            "",
            "| 指标 | 更新前 | 更新后 |",
            "|---|---:|---:|",
            (
                "| Steam完整窗口评论量 | "
                f"{report.before.total_reviews} | {report.after.total_reviews} |"
            ),
            (
                "| Steam好评率 | "
                f"{format_percent(report.before.positive_rate)} | "
                f"{format_percent(report.after.positive_rate)} |"
            ),
            (
                "| 抽样正面评论 | "
                f"{impact.before.sentiment_counts['positive']} | "
                f"{impact.after.sentiment_counts['positive']} |"
            ),
            (
                "| 抽样负面评论 | "
                f"{impact.before.sentiment_counts['negative']} | "
                f"{impact.after.sentiment_counts['negative']} |"
            ),
            (
                "| 平均严重度 | "
                f"{impact.before.average_severity or 0:.2f} | "
                f"{impact.after.average_severity or 0:.2f} |"
            ),
            (
                "| 可执行问题 | "
                f"{impact.before.actionable_count} | "
                f"{impact.after.actionable_count} |"
            ),
            "",
            "## 主题变化",
            "",
            "主题允许多标签，因此各主题比例之和可能超过100%。",
            "",
            "| 主题 | 更新前 | 更新后 | 变化 |",
            "|---|---:|---:|---:|",
        ]
    )

    changed_topics = [
        item
        for item in impact.topic_changes
        if item.before_count or item.after_count
    ]
    for item in changed_topics:
        lines.append(
            f"| {TOPIC_LABELS[item.topic]} | "
            f"{item.before_count} ({format_percent(item.before_rate)}) | "
            f"{item.after_count} ({format_percent(item.after_rate)}) | "
            f"{format_pp(item.change_percentage_points)} |"
        )

    lines.extend([""])
    lines.extend(
        build_review_section(
            "更新后重点问题",
            select_representative_reviews(
                records,
                positive=False,
                limit=5,
            ),
        )
    )
    lines.extend(
        build_review_section(
            "更新后积极反馈",
            select_representative_reviews(
                records,
                positive=True,
                limit=3,
            ),
        )
    )

    lines.extend(["## 限制与说明", ""])
    for warning in impact.warnings:
        lines.append(f"- ⚠️ {warning}")
    for note in impact.notes:
        lines.append(f"- {note}")
    lines.extend(
        [
            "- 本报告仅依据输入的Steam补丁、评论和分类结果生成，不补充外部事实。",
            "- 原文证据用于审计模型判断，不代表报告作者认同评论内容。",
            "",
        ]
    )
    return "\n".join(lines)


def save_markdown_report(
    report: VersionImpactReport,
    markdown: str,
) -> Path:
    output_path = Path("reports") / (
        f"{report.appid}_{report.patch.news_id}_impact_report.md"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output_path,markdown)
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
        impact_path = Path("data") / "processed" / (
            f"{appid}_review_impact_{report.patch.news_id}.json"
        )
        if not classification_path.exists() or not impact_path.exists():
            print(
                "缺少评论分类或影响聚合结果，请先运行"
                "python -m gamelens.analysis.classifier 和 python -m gamelens.analysis.impact。"
            )
            return

        records = load_classifications(classification_path)
        impact = load_impact_summary(impact_path)
        markdown = generate_markdown(report, impact, records)
        output_path = save_markdown_report(report, markdown)
    except (OSError, ValueError, ValidationError) as error:
        print(f"报告生成失败：{error}")
        return

    print(f"Markdown报告已生成：{output_path.resolve()}")


if __name__ == "__main__":
    main()
