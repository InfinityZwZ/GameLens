import json
import os
import random
from pathlib import Path
from gamelens.storage.atomic_io import atomic_write_json
from gamelens.core.local_time import local_iso

import httpx
from pydantic import ValidationError

from gamelens.core.llm import (
    LLMConfigurationError,
    LLMResponseError,
    OpenAICompatibleClient,
)
from gamelens.domain.models import (
    ClassifiedReview,
    ReviewAnalysis,
    ReviewAnalysisBatch,
    SteamReview,
    VersionImpactReport,
)
from gamelens.analysis.version import load_reviews


SYSTEM_PROMPT = """你是严格的游戏玩家评论标注器。
只能依据输入的评论与补丁内容判断，不得补充外部事实。
必须返回纯JSON对象，禁止Markdown和解释文字。

主题枚举：
- performance：帧率、卡顿、优化、硬件占用
- crash_bug：崩溃、闪退、无法启动、明确的软件缺陷
- server_network：服务器、延迟、掉线、连接问题
- matchmaking：匹配质量、排位或队友匹配
- cheating：外挂、作弊、反作弊
- balance：角色、武器、数值和平衡性
- monetization：价格、内购、付费、抽卡
- ui_ux：界面、操作、可用性
- content：地图、模式、剧情、活动等内容量
- localization：翻译、配音、本地化
- gameplay：核心玩法和操作体验
- praise：明确表扬且没有可执行问题
- noise：玩梗、刷屏、无实际信息
- other：无法归入以上类别

严重度规则：
1=轻微偏好；2=小问题；3=明显影响体验；
4=严重影响主要功能；5=无法游玩、频繁崩溃或重大故障。

evidence必须逐字摘自原评论，不能改写。
evidence只选最能支撑判断的一小段，不要复制整条评论，长度不得超过200个字符。
每条分析必须包含以下全部字段，不得省略：review_id、sentiment、topics、
severity、is_actionable、update_related、summary、evidence、confidence。
summary必须填写，是对该条评论的简短客观概括，长度不得超过200个字符。
topics最多3个。confidence范围为0到1。
更新前评论的update_related必须为false。
"""

MAX_SUMMARY_LENGTH = 200
MAX_EVIDENCE_LENGTH = 300


from gamelens.analysis.version import load_report  # Backward-compatible import.


def select_window_reviews(
    reviews: list[SteamReview],
    report: VersionImpactReport,
    period: str,
    count: int,
    seed: int = 42,
) -> list[SteamReview]:
    window = report.before if period == "before" else report.after
    start_timestamp = int(window.start_at.timestamp())
    end_timestamp = int(window.end_at.timestamp())
    candidates = [
        review
        for review in reviews
        if review.text.strip()
        and start_timestamp
        <= review.created_at_timestamp
        < end_timestamp
    ]

    if len(candidates) <= count:
        return candidates

    random_generator = random.Random(seed)
    return random_generator.sample(candidates, count)


def build_user_prompt(
    reviews: list[SteamReview],
    report: VersionImpactReport,
    period: str,
) -> str:
    review_payload = [
        {
            "review_id": review.review_id,
            "text": review.text,
            "is_positive": review.is_positive,
            "playtime_at_review_minutes": (
                review.playtime_at_review_minutes
            ),
        }
        for review in reviews
    ]

    output_example = {
        "analyses": [
            {
                "review_id": "必须与输入ID一致",
                "sentiment": "negative",
                "topics": ["crash_bug"],
                "severity": 4,
                "is_actionable": True,
                "update_related": period == "after",
                "summary": "简短客观总结",
                "evidence": "逐字摘录原评论",
                "confidence": 0.9,
            }
        ]
    }

    return (
        f"补丁标题：{report.patch.title}\n"
        f"补丁发布时间（北京时间时间）：{local_iso(report.patch.published_at_timestamp)}\n"
        f"补丁内容：{report.patch.contents}\n"
        f"评论时期：{period}\n\n"
        "请逐条分析下面的评论，每个输入ID必须且只能输出一次。\n"
        f"评论：{json.dumps(review_payload, ensure_ascii=False)}\n\n"
        "输出结构示例："
        f"{json.dumps(output_example, ensure_ascii=False)}"
    )


def validate_batch_result(
    raw_result: dict,
    reviews: list[SteamReview],
    period: str,
) -> list[ClassifiedReview]:
    normalized_result = normalize_batch_result(raw_result)
    batch = ReviewAnalysisBatch.model_validate(normalized_result)
    review_by_id = {review.review_id: review for review in reviews}
    returned_ids = [item.review_id for item in batch.analyses]

    if len(returned_ids) != len(set(returned_ids)):
        raise ValueError("LLM返回了重复的review_id")
    if set(returned_ids) != set(review_by_id):
        raise ValueError("LLM返回的review_id与输入不一致")

    records: list[ClassifiedReview] = []
    for analysis in batch.analyses:
        review = review_by_id[analysis.review_id]
        if analysis.evidence not in review.text:
            raise ValueError(
                f"评论{review.review_id}的evidence不是原文片段"
            )

        if period == "before" and analysis.update_related:
            analysis = analysis.model_copy(
                update={"update_related": False}
            )

        records.append(
            ClassifiedReview(
                period=period,
                review=review,
                analysis=analysis,
            )
        )

    return records


def normalize_batch_result(raw_result: dict) -> dict:
    """清理不影响语义的LLM格式偏差，再交给Pydantic严格校验。"""
    analyses = raw_result.get("analyses")
    if not isinstance(analyses, list):
        return raw_result

    normalized_analyses = []
    for item in analyses:
        if not isinstance(item, dict):
            normalized_analyses.append(item)
            continue

        normalized_item = item.copy()
        evidence = normalized_item.get("evidence")
        if isinstance(evidence, str):
            normalized_evidence = evidence.strip()[
                :MAX_EVIDENCE_LENGTH
            ]
            normalized_item["evidence"] = normalized_evidence

            summary = normalized_item.get("summary")
            if not isinstance(summary, str) or not summary.strip():
                # 模型偶尔会漏掉summary；使用原文证据作为提取式保底摘要。
                normalized_item["summary"] = normalized_evidence[
                    :MAX_SUMMARY_LENGTH
                ]

        summary = normalized_item.get("summary")
        if isinstance(summary, str):
            normalized_item["summary"] = summary.strip()[
                :MAX_SUMMARY_LENGTH
            ]
        normalized_analyses.append(normalized_item)

    normalized_result = raw_result.copy()
    normalized_result["analyses"] = normalized_analyses
    return normalized_result


def classify_batch(
    client: OpenAICompatibleClient,
    reviews: list[SteamReview],
    report: VersionImpactReport,
    period: str,
) -> list[ClassifiedReview]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt(reviews, report, period),
        },
    ]
    last_error: Exception | None = None

    max_attempts = 2
    for attempt in range(max_attempts):
        try:
            raw_result = client.generate_json(messages)
            return validate_batch_result(raw_result, reviews, period)
        except (ValidationError, ValueError, LLMResponseError) as error:
            last_error = error
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "上一次输出未通过校验："
                        f"{error}。请重新返回完整且严格有效的JSON。"
                    ),
                }
            )
            if attempt + 1 < max_attempts:
                print(
                    f"第{attempt + 1}次输出校验失败，正在重试……"
                )

    raise LLMResponseError(
        f"LLM连续两次输出无效：{last_error}"
    )


def classify_reviews(
    client: OpenAICompatibleClient,
    reviews: list[SteamReview],
    report: VersionImpactReport,
    per_window_count: int,
    batch_size: int = 5,
    existing_records: list[ClassifiedReview] | None = None,
) -> list[ClassifiedReview]:
    record_by_key = {
        (record.period, record.review.review_id): record
        for record in (existing_records or [])
    }
    all_records = list(record_by_key.values())

    for period, seed in (("before", 42), ("after", 43)):
        selected = select_window_reviews(
            reviews,
            report,
            period,
            per_window_count,
            seed,
        )
        print(f"{period}窗口选择了{len(selected)}条评论")
        pending = [
            review
            for review in selected
            if (period, review.review_id) not in record_by_key
        ]
        skipped_count = len(selected) - len(pending)
        if skipped_count:
            print(
                f"已从检查点恢复{skipped_count}条，"
                f"剩余{len(pending)}条待分析"
            )

        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            print(
                f"正在分析{period}窗口剩余评论 "
                f"{start + 1}-{start + len(batch)}条……"
            )
            batch_records = classify_batch(
                client,
                batch,
                report,
                period,
            )
            all_records.extend(batch_records)
            for record in batch_records:
                record_by_key[
                    (record.period, record.review.review_id)
                ] = record
            checkpoint_path = save_classifications(
                report,
                all_records,
            )
            print(
                "已保存阶段性结果："
                f"{checkpoint_path.resolve()}"
            )

    return all_records


def classification_output_path(report: VersionImpactReport) -> Path:
    return Path("data") / "processed" / (
        f"{report.appid}_review_classifications_"
        f"{report.patch.news_id}.json"
    )


def load_existing_classifications(
    report: VersionImpactReport,
) -> list[ClassifiedReview]:
    output_path = classification_output_path(report)
    if not output_path.exists():
        return []

    with output_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError("已有评论分类文件的最外层必须是数组")

    return [
        ClassifiedReview.model_validate(item)
        for item in payload
    ]


def save_classifications(
    report: VersionImpactReport,
    records: list[ClassifiedReview],
) -> Path:
    output_path = classification_output_path(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    atomic_write_json(output_path,[record.model_dump(mode="json") for record in records])

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

    count_text = input(
        "每个时间窗口分析多少条评论（直接回车默认10）："
    ).strip()
    if count_text:
        if not count_text.isdigit() or int(count_text) <= 0:
            print("评论数量必须是正整数")
            return
        per_window_count = int(count_text)
    else:
        per_window_count = 10

    model = os.getenv("LLM_MODEL") or input(
        "请输入模型名称（也可设置LLM_MODEL环境变量）："
    ).strip()

    try:
        client = OpenAICompatibleClient(model=model)
        report = load_report(report_files[selection - 1])
        reviews = load_reviews(
            Path("data") / "raw" / f"{appid}_reviews.json"
        )
        existing_records = load_existing_classifications(report)
        if existing_records:
            print(
                f"发现检查点：{len(existing_records)}条已完成结果"
            )
        records = classify_reviews(
            client,
            reviews,
            report,
            per_window_count,
            existing_records=existing_records,
        )
        output_path = save_classifications(report, records)
    except (
        OSError,
        ValueError,
        ValidationError,
        httpx.HTTPError,
        LLMConfigurationError,
        LLMResponseError,
    ) as error:
        print(f"评论分类失败：{error}")
        return

    print(f"完成分类：{len(records)}条")
    print(f"结果已保存：{output_path.resolve()}")


if __name__ == "__main__":
    main()
