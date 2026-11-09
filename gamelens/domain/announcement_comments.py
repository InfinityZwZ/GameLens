"""Announcement discussion messages are NOT Steam store recommendations."""
from datetime import datetime
import re
from pydantic import BaseModel, ConfigDict, Field


def announcement_only_goal(goal):
    """Explicitly requested shop reviews keep mixed-source investigations available."""
    return wants_announcement_feedback(goal) and not re.search(
        r'(?:同时|也|并|结合|对比|比较|以及|和|与).{0,18}(?:商店|游戏整体|游戏总|整个游戏).{0,5}(?:评价|评论)|'
        r'(?:商店|游戏整体|游戏总|整个游戏).{0,5}(?:评价|评论).{0,18}(?:同时|也|并|结合|对比|比较|以及|和|与)|'
        r'store\s+reviews', goal, re.I)


class AnnouncementComment(BaseModel):
    model_config = ConfigDict(extra='forbid')
    appid: int = Field(gt=0)
    news_id: str = Field(min_length=1, max_length=100)
    announcement_gid: str = Field(pattern=r'^\d+$')
    topic_id: str = Field(pattern=r'^\d+$')
    comment_id: str = Field(pattern=r'^\d+$')
    text: str = Field(min_length=1, max_length=60000)
    author: str = Field(default='', max_length=300)
    created_at_timestamp: int = Field(gt=0)
    created_at: datetime
    source_url: str = Field(pattern=r'^https://steamcommunity\.com/app/\d+/eventcomments/\d+/#c\d+$')
    contains_quoted_text: bool = False


def wants_announcement_feedback(goal):
    """Explicit source, or the dated-update-feedback template; not every bug task."""
    import re
    if re.search(r'(?:仅|只)(?:看|查|读|分析|统计).{0,6}(?:商店|游戏整体)评[价论]|(?:不要|不读|不看|不用).{0,10}(?:公告|活动).{0,10}(?:评论|留言|讨论)', goal):
        return False
    explicit = re.search(r'(?:公告|补丁|活动).{0,12}(?:评论区|讨论区|留言|讨论回复|(?:下|里|中的).{0,5}评论)|announcement.{0,15}(?:comments|discussion)', goal, re.I)
    template = re.search(r'玩家对.{0,35}(?:更新|版本|活动|公告).{0,15}(?:反馈|评价|看法)', goal)
    return bool(explicit or template)
