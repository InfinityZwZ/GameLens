import json
import re
from urllib.parse import urlparse
from pathlib import Path
from gamelens.storage.atomic_io import atomic_write_json

import httpx
from pydantic import ValidationError

from gamelens.domain.models import SteamNewsItem
from gamelens.core.utils import unix_timestamp_to_iso


STEAM_NEWS_URL = (
    "https://api.steampowered.com/ISteamNews/GetNewsForApp/v2/"
)

STRONG_PATCH_KEYWORDS = (
    "patch",
    "hotfix",
    "changelog",
    "版本",
    "补丁",
    "热修",
    "修复",
)

OFFICIAL_ANNOUNCEMENT_FEED = "steam_community_announcements"
PATCH_TITLE_PATTERNS = (
    re.compile(r"\bbug\s+fix(?:es)?\b", re.I),
    re.compile(r"\bfix(?:es|ed)?\b", re.I),
    re.compile(r"\bv?\d+(?:\.\d+){1,3}\b", re.I),
)
PRODUCT_VERSION_UPDATE_PATTERN = re.compile(
    r"^[A-Z0-9][A-Za-z0-9®™:'’\- ]*\d[A-Za-z0-9®™:'’\- ]*\s+Update\s*$",
    re.I,
)
NON_PATCH_ACTIVITY_PATTERNS = (
    re.compile(r"\bcommunity\s+update\b", re.I),
    re.compile(r"\bdeveloper\s+update\b", re.I),
    re.compile(r"\bdevelopment\s+update\b", re.I),
    re.compile(r"\broadmap\s+update\b", re.I),
    re.compile(r"\b(?:event|livestream|contest|giveaway|sale|bundle|poll)\b", re.I),
    re.compile(r"\bmeet\s+the\s+team\b", re.I),
    re.compile(r"\b(?:server\s+)?maintenance(?:\s+notice)?\b", re.I),
    re.compile(r"(?:社区|开发者|活动|直播|赛事|促销|维护)(?:公告|更新)?"),
)


def fetch_news(appid: int, count: int = 100, *, enddate: int | None = None, feeds: str | None = None) -> list[dict]:
    """获取指定 Steam 游戏的公开新闻与公告。"""
    if appid <= 0:
        raise ValueError("appid 必须是正整数")

    if not 1 <= count <= 1000:
        raise ValueError("count 必须在 1 到 1000 之间")
    if enddate is not None and (type(enddate) is not int or not 0 < enddate <= 4294967295):
        raise ValueError("enddate必须是有效的uint32 Unix秒时间戳")
    if feeds not in {None, 'steam_community_announcements'}:
        raise ValueError('不支持的新闻来源筛选')

    params = {
        "appid": appid,
        "count": count,
        "maxlength": 0,
        "format": "json",
    }
    if enddate is not None:
        params["enddate"] = enddate
    if feeds is not None:
        params['feeds'] = feeds

    with httpx.Client(timeout=30, follow_redirects=True) as client:
        response = client.get(STEAM_NEWS_URL, params=params)
        response.raise_for_status()
        data = response.json()

    app_news = data.get("appnews")
    if not isinstance(app_news, dict):
        raise RuntimeError("Steam News API 返回格式异常")

    return app_news.get("newsitems", [])


def _news_value(news, field, default=""):
    if isinstance(news, dict):
        return news.get(field, default)
    return getattr(news, field, default)


def is_official_announcement(news) -> bool:
    """Return whether the item is published through Steam's official app feed."""
    feed = str(_news_value(news, "feedname", _news_value(news, "feed_name", ""))).strip().casefold()
    url = str(_news_value(news, "url", "")).casefold()
    parsed = urlparse(url)
    direct_community_post = (parsed.scheme == "https" and parsed.hostname in
        {"steamcommunity.com", "www.steamcommunity.com"} and
        "/announcements/detail/" in parsed.path)
    return (feed == OFFICIAL_ANNOUNCEMENT_FEED or
            f"/{OFFICIAL_ANNOUNCEMENT_FEED}/" in parsed.path or direct_community_post)


def detect_patch_candidate(news) -> bool:
    """Identify official patch/version notes, excluding media and activity posts."""
    if not is_official_announcement(news):
        return False
    title = str(_news_value(news, "title", ""))
    title_folded = title.casefold()
    has_strong_patch_signal = (
        any(keyword in title_folded for keyword in STRONG_PATCH_KEYWORDS) or
        any(pattern.search(title) for pattern in PATCH_TITLE_PATTERNS)
    )
    if any(pattern.search(title) for pattern in NON_PATCH_ACTIVITY_PATTERNS) and not has_strong_patch_signal:
        return False
    # A generic word such as "update" is not sufficient: roadmap, community,
    # developer and event posts frequently use it.  Keep the common official
    # "<product name containing a version number> Update" form (for example
    # "Counter-Strike 2 Update") without opening that broad false-positive path.
    return has_strong_patch_signal or bool(PRODUCT_VERSION_UPDATE_PATTERN.fullmatch(title.strip()))


def retained_patch_candidate(news) -> bool:
    """Re-evaluate cached records while retaining explicit synthetic fixtures."""
    if is_official_announcement(news):
        return detect_patch_candidate(news)
    feed = str(_news_value(news, "feedname", _news_value(news, "feed_name", ""))).strip().casefold()
    url = str(_news_value(news, "url", "")).casefold()
    explicitly_marked = bool(_news_value(news, "is_patch_candidate", False))
    return explicitly_marked and (feed == "fixture" or ".invalid/" in url)


def normalize_news(news: dict, appid: int) -> SteamNewsItem:
    """将 Steam 原始新闻转换为统一的数据模型。"""
    published_timestamp = news.get("date")

    return SteamNewsItem(
        news_id=str(news.get("gid", "")),
        appid=appid,
        title=news.get("title", ""),
        url=news.get("url", ""),
        author=news.get("author", ""),
        contents=news.get("contents", ""),
        feed_label=news.get("feedlabel", ""),
        feed_name=news.get("feedname", ""),
        published_at=unix_timestamp_to_iso(published_timestamp),
        published_at_timestamp=published_timestamp,
        is_patch_candidate=detect_patch_candidate(news),
    )


def save_news(
    appid: int,
    news_items: list[SteamNewsItem],
) -> tuple[Path, Path]:
    """保存全部新闻和补丁候选项。"""
    output_directory = Path("data") / "raw"
    output_directory.mkdir(parents=True, exist_ok=True)

    all_news_path = output_directory / f"{appid}_news.json"
    patch_candidates_path = (
        output_directory / f"{appid}_patch_candidates.json"
    )

    all_news_data = [
        item.model_dump(mode="json")
        for item in news_items
    ]
    patch_candidates_data = [
        item.model_dump(mode="json")
        for item in news_items
        if item.is_patch_candidate
    ]

    atomic_write_json(all_news_path,all_news_data)
    atomic_write_json(patch_candidates_path,patch_candidates_data)

    return all_news_path, patch_candidates_path


def main() -> None:
    appid_text = input("请输入Steam游戏AppID：").strip()
    if not appid_text.isdigit() or int(appid_text) <= 0:
        print("AppID必须是正整数")
        return

    count_text = input(
        "请输入希望获取的公告数量（直接回车默认100）："
    ).strip()
    if count_text:
        if not count_text.isdigit():
            print("公告数量必须是数字")
            return
        count = int(count_text)
    else:
        count = 100

    if not 1 <= count <= 1000:
        print("公告数量必须在1到1000之间")
        return

    appid = int(appid_text)
    print("正在获取Steam公告……")

    try:
        raw_news = fetch_news(appid=appid, count=count)
        news_items = [
            normalize_news(item, appid)
            for item in raw_news
        ]
    except httpx.HTTPError as error:
        print(f"网络请求失败：{error}")
        return
    except (RuntimeError, ValidationError) as error:
        print(f"公告数据处理失败：{error}")
        return

    all_news_path, patch_candidates_path = save_news(
        appid,
        news_items,
    )
    patch_candidates = [
        item for item in news_items if item.is_patch_candidate
    ]

    print(f"获取公告数量：{len(news_items)}")
    print(f"补丁候选数量：{len(patch_candidates)}")
    print(f"全部公告：{all_news_path.resolve()}")
    print(f"补丁候选：{patch_candidates_path.resolve()}")

    if patch_candidates:
        print("\n最近的补丁候选：")
        for item in patch_candidates[:5]:
            print(f"- {item.published_at.date()} | {item.title}")


if __name__ == "__main__":
    main()
