from datetime import datetime, timezone

import pytest

from gamelens.agents.pipeline import AgentConfig, select_patch
from gamelens.analysis.impact import build_impact_summary
from gamelens.domain.models import (
    ClassifiedReview,
    ReviewAnalysis,
    SteamNewsItem,
    SteamReview,
)
from gamelens.analysis.report import generate_markdown
from gamelens.analysis.classifier import classify_reviews, validate_batch_result
from gamelens.analysis.version import build_version_impact_report


PATCH_TIMESTAMP = 1_800_000_000
SECONDS_PER_DAY = 24 * 60 * 60


def make_review(
    review_id: str,
    timestamp: int,
    *,
    text: str = "测试评论",
    positive: bool = True,
) -> SteamReview:
    moment = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return SteamReview(
        review_id=review_id,
        text=text,
        language="schinese",
        is_positive=positive,
        votes_up=1,
        created_at=moment,
        updated_at=moment,
        created_at_timestamp=timestamp,
        updated_at_timestamp=timestamp,
        playtime_minutes=120,
        playtime_at_review_minutes=60,
        steam_purchase=True,
        received_for_free=False,
        early_access=False,
    )


def make_patch(
    news_id: str = "patch-1",
    timestamp: int = PATCH_TIMESTAMP,
    title: str = "Test Update",
) -> SteamNewsItem:
    return SteamNewsItem(
        news_id=news_id,
        appid=730,
        title=title,
        url="https://example.com/patch",
        author="Valve",
        contents="Fixed a test issue.",
        feed_label="Community Announcements",
        feed_name="steam_community_announcements",
        published_at=datetime.fromtimestamp(
            timestamp,
            tz=timezone.utc,
        ),
        published_at_timestamp=timestamp,
        is_patch_candidate=True,
    )


def make_classified(
    period: str,
    review: SteamReview,
    *,
    sentiment: str,
    topic: str,
    severity: int,
    actionable: bool,
    update_related: bool,
) -> ClassifiedReview:
    return ClassifiedReview(
        period=period,
        review=review,
        analysis=ReviewAnalysis(
            review_id=review.review_id,
            sentiment=sentiment,
            topics=[topic],
            severity=severity,
            is_actionable=actionable,
            update_related=update_related,
            summary=review.text,
            evidence=review.text,
            confidence=0.9,
        ),
    )


def test_version_window_boundaries_are_half_open() -> None:
    patch = make_patch()
    reviews = [
        make_review(
            "before-start",
            PATCH_TIMESTAMP - SECONDS_PER_DAY,
        ),
        make_review(
            "before-end",
            PATCH_TIMESTAMP - 1,
            positive=False,
        ),
        make_review("after-start", PATCH_TIMESTAMP),
        make_review(
            "after-end-marker",
            PATCH_TIMESTAMP + SECONDS_PER_DAY,
        ),
    ]

    report = build_version_impact_report(reviews, patch, 1)

    assert report.before.total_reviews == 2
    assert report.after.total_reviews == 1
    assert report.before.positive_rate == pytest.approx(0.5)
    assert report.after.positive_rate == pytest.approx(1.0)
    assert report.before.coverage_complete is True
    assert report.after.coverage_complete is True


def test_missing_summary_and_long_evidence_are_normalized() -> None:
    text = "测" * 400
    review = make_review("review-1", PATCH_TIMESTAMP - 1, text=text)
    raw_result = {
        "analyses": [
            {
                "review_id": review.review_id,
                "sentiment": "negative",
                "topics": ["other"],
                "severity": 2,
                "is_actionable": False,
                "update_related": False,
                "evidence": text,
                "confidence": 0.8,
            }
        ]
    }

    result = validate_batch_result(raw_result, [review], "before")

    assert len(result[0].analysis.evidence) == 300
    assert len(result[0].analysis.summary) == 200
    assert result[0].analysis.evidence in review.text


def test_classification_resume_skips_completed_reviews(tmp_path) -> None:
    patch = make_patch()
    before_review = make_review("before", PATCH_TIMESTAMP - 1)
    after_review = make_review("after", PATCH_TIMESTAMP)
    reviews = [before_review, after_review]
    report = build_version_impact_report(reviews, patch, 1)
    existing = [
        make_classified(
            "before",
            before_review,
            sentiment="positive",
            topic="praise",
            severity=1,
            actionable=False,
            update_related=False,
        ),
        make_classified(
            "after",
            after_review,
            sentiment="negative",
            topic="crash_bug",
            severity=4,
            actionable=True,
            update_related=True,
        ),
    ]

    class ExplodingClient:
        def generate_json(self, messages):
            raise AssertionError("已完成的评论不应再次调用LLM")

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.chdir(tmp_path)
        result = classify_reviews(
            ExplodingClient(),
            reviews,
            report,
            per_window_count=1,
            existing_records=existing,
        )

    assert len(result) == 2


def test_impact_summary_and_markdown_report() -> None:
    patch = make_patch()
    before_review = make_review(
        "before",
        PATCH_TIMESTAMP - 1,
        text="更新前体验很好",
    )
    after_review = make_review(
        "after",
        PATCH_TIMESTAMP,
        text="更新后出现严重崩溃",
        positive=False,
    )
    report = build_version_impact_report(
        [before_review, after_review],
        patch,
        1,
    )
    records = [
        make_classified(
            "before",
            before_review,
            sentiment="positive",
            topic="praise",
            severity=1,
            actionable=False,
            update_related=False,
        ),
        make_classified(
            "after",
            after_review,
            sentiment="negative",
            topic="crash_bug",
            severity=5,
            actionable=True,
            update_related=True,
        ),
    ]

    impact = build_impact_summary(report, records)
    markdown = generate_markdown(report, impact, records)

    crash_change = next(
        item
        for item in impact.topic_changes
        if item.topic.value == "crash_bug"
    )
    assert crash_change.change_percentage_points == pytest.approx(100)
    assert impact.after.actionable_rate == pytest.approx(1.0)
    assert "## 更新后重点问题" in markdown
    assert "更新后出现严重崩溃" in markdown
    assert "探索性信号" in markdown


def test_agent_selects_latest_patch_with_complete_window() -> None:
    config = AgentConfig(
        appid=730,
        window_days=1,
        review_count=100,
        news_count=20,
        classify_count=5,
        patch_news_id=None,
        refresh_data=False,
        force_reclassify=False,
    )
    reviews = [
        make_review(
            "oldest",
            PATCH_TIMESTAMP - 3 * SECONDS_PER_DAY,
        ),
        make_review(
            "newest",
            PATCH_TIMESTAMP + SECONDS_PER_DAY,
        ),
    ]
    incomplete_latest = make_patch(
        "latest",
        PATCH_TIMESTAMP + SECONDS_PER_DAY,
        "Latest Incomplete Update",
    )
    complete_older = make_patch(
        "complete",
        PATCH_TIMESTAMP,
        "Complete Update",
    )

    selected = select_patch(
        config,
        reviews,
        [incomplete_latest, complete_older],
    )

    assert selected.news_id == "complete"
