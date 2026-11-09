from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SteamReview(BaseModel):
    """清洗并校验后的单条 Steam 评论。"""

    model_config = ConfigDict(extra="forbid")

    review_id: str = Field(min_length=1)
    text: str
    language: str = Field(min_length=1)
    is_positive: bool
    votes_up: int = Field(ge=0)
    created_at: datetime
    updated_at: datetime
    created_at_timestamp: int = Field(ge=0)
    updated_at_timestamp: int = Field(ge=0)
    playtime_minutes: int = Field(ge=0)
    playtime_at_review_minutes: int = Field(ge=0)
    steam_purchase: bool
    received_for_free: bool
    early_access: bool

    @model_validator(mode="after")
    def validate_time_order(self) -> "SteamReview":
        if self.updated_at_timestamp < self.created_at_timestamp:
            raise ValueError("评论更新时间不能早于创建时间")

        return self


class SteamNewsItem(BaseModel):
    """清洗并校验后的单条 Steam 新闻或公告。"""

    model_config = ConfigDict(extra="forbid")

    news_id: str = Field(min_length=1)
    appid: int = Field(gt=0)
    title: str = Field(min_length=1)
    url: str
    author: str = ""
    contents: str = ""
    feed_label: str = ""
    feed_name: str = ""
    published_at: datetime
    published_at_timestamp: int = Field(ge=0)
    is_patch_candidate: bool = False


class ReviewWindowStats(BaseModel):
    """某个时间窗口内的评论统计。"""

    start_at: datetime
    end_at: datetime
    total_reviews: int = Field(ge=0)
    positive_reviews: int = Field(ge=0)
    negative_reviews: int = Field(ge=0)
    positive_rate: float | None = Field(default=None, ge=0, le=1)
    negative_rate: float | None = Field(default=None, ge=0, le=1)
    average_playtime_hours: float | None = Field(default=None, ge=0)
    coverage_complete: bool


class VersionImpactReport(BaseModel):
    """某次版本公告前后的评论对比报告。"""

    appid: int = Field(gt=0)
    patch: SteamNewsItem
    window_days: int = Field(gt=0)
    before: ReviewWindowStats
    after: ReviewWindowStats
    review_count_change: int
    positive_rate_change_percentage_points: float | None
    negative_rate_change_percentage_points: float | None
    warnings: list[str] = Field(default_factory=list)


class ReviewTopic(StrEnum):
    PERFORMANCE = "performance"
    CRASH_BUG = "crash_bug"
    SERVER_NETWORK = "server_network"
    MATCHMAKING = "matchmaking"
    CHEATING = "cheating"
    BALANCE = "balance"
    MONETIZATION = "monetization"
    UI_UX = "ui_ux"
    CONTENT = "content"
    LOCALIZATION = "localization"
    GAMEPLAY = "gameplay"
    PRAISE = "praise"
    NOISE = "noise"
    OTHER = "other"


class ReviewAnalysis(BaseModel):
    """LLM对单条玩家评论的结构化分析。"""

    model_config = ConfigDict(extra="forbid")

    review_id: str = Field(min_length=1)
    sentiment: Literal["positive", "neutral", "negative", "mixed"]
    topics: list[ReviewTopic] = Field(min_length=1, max_length=3)
    severity: int = Field(ge=1, le=5)
    is_actionable: bool
    update_related: bool
    summary: str = Field(min_length=1, max_length=200)
    evidence: str = Field(min_length=1, max_length=300)
    confidence: float = Field(ge=0, le=1)


class ReviewAnalysisBatch(BaseModel):
    analyses: list[ReviewAnalysis] = Field(min_length=1)


class ClassifiedReview(BaseModel):
    """保留原始事实与LLM判断，避免两者混淆。"""

    period: Literal["before", "after"]
    review: SteamReview
    analysis: ReviewAnalysis


class ClassifiedPeriodStats(BaseModel):
    """一个时间窗口内的LLM分类聚合统计。"""

    review_count: int = Field(ge=0)
    sentiment_counts: dict[str, int]
    sentiment_rates: dict[str, float]
    topic_counts: dict[ReviewTopic, int]
    topic_rates: dict[ReviewTopic, float]
    average_severity: float | None = Field(default=None, ge=1, le=5)
    actionable_count: int = Field(ge=0)
    actionable_rate: float | None = Field(default=None, ge=0, le=1)
    update_related_count: int = Field(ge=0)
    update_related_rate: float | None = Field(default=None, ge=0, le=1)
    average_confidence: float | None = Field(default=None, ge=0, le=1)


class TopicChange(BaseModel):
    """某个主题在更新前后样本中的变化。"""

    topic: ReviewTopic
    before_count: int = Field(ge=0)
    after_count: int = Field(ge=0)
    before_rate: float = Field(ge=0, le=1)
    after_rate: float = Field(ge=0, le=1)
    change_percentage_points: float


class VersionReviewImpactSummary(BaseModel):
    """评论分类结果形成的版本影响聚合报告。"""

    appid: int = Field(gt=0)
    news_id: str = Field(min_length=1)
    patch_title: str = Field(min_length=1)
    patch_published_at: datetime
    classified_review_count: int = Field(ge=0)
    before: ClassifiedPeriodStats
    after: ClassifiedPeriodStats
    topic_changes: list[TopicChange]
    leading_topic_increases: list[ReviewTopic]
    leading_topic_decreases: list[ReviewTopic]
    warnings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
