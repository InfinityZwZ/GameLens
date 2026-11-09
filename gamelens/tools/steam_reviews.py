import json
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from gamelens.storage.atomic_io import atomic_write_json

import httpx
from pydantic import ValidationError

from gamelens.domain.models import SteamReview
from gamelens.core.utils import unix_timestamp_to_iso


STEAM_REVIEW_URL = "https://store.steampowered.com/appreviews/{appid}"


def fetch_reviews_snapshot(
    appid: int,
    language: str = "schinese",
    target_count: int = 500,
    request_interval: float = 0.5,
) -> dict:
    """Fetch a bounded review snapshot and retain why pagination stopped."""
    if not 1 <= target_count <= 5000:
        raise ValueError("target_count 必须在 1 到 5000 之间")
    if request_interval < 0:
        raise ValueError("request_interval 不能小于 0")

    url = STEAM_REVIEW_URL.format(appid=appid)
    cursor = "*"
    reviews: list[dict] = []
    seen_review_ids: set[str] = set()
    page_number = 0
    stop_reason = "target_reached"

    with httpx.Client(timeout=30) as client:
        while len(reviews) < target_count:
            page_size = min(100, target_count - len(reviews))
            params = {"json": 1, "filter": "recent", "language": language,
                "review_type": "all", "purchase_type": "all",
                "num_per_page": page_size, "cursor": cursor}
            response = client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            if data.get("success") != 1:
                raise RuntimeError("Steam API 返回失败")
            page_reviews = data.get("reviews", [])
            if not page_reviews:
                stop_reason = "source_exhausted"
                break
            new_review_count = 0
            for review in page_reviews:
                review_id = review.get("recommendationid")
                if not review_id or review_id in seen_review_ids:
                    continue
                seen_review_ids.add(review_id)
                reviews.append(review)
                new_review_count += 1
            page_number += 1
            print(f"已获取第 {page_number} 页：新增 {new_review_count} 条，累计 {len(reviews)} 条")
            if len(reviews) >= target_count:
                stop_reason = "target_reached"
                break
            next_cursor = data.get("cursor")
            if not next_cursor or next_cursor == cursor:
                stop_reason = "cursor_stalled"
                break
            if new_review_count == 0:
                stop_reason = "duplicate_page"
                break
            cursor = next_cursor
            if len(reviews) < target_count:
                time.sleep(request_interval)
    return {"reviews": reviews[:target_count], "source_exhausted": stop_reason == "source_exhausted",
        "stop_reason": stop_reason, "pages_read": page_number}


def fetch_reviews(
    appid: int,
    language: str = "schinese",
    target_count: int = 500,
    request_interval: float = 0.5,
) -> list[dict]:
    """
    使用 cursor 分页获取 Steam 游戏评论。

    Args:
        appid: Steam 游戏的 AppID。
        language: 评论语言，例如 schinese、english、all。
        target_count: 希望获取的评论总数，最多 5000 条。
        request_interval: 两次请求之间的等待秒数。

    Returns:
        去重后的 Steam 评论列表。
    """
    return fetch_reviews_snapshot(appid, language, target_count, request_interval)["reviews"]


def review_coverage_path(review_path: Path) -> Path:
    return review_path.with_name(review_path.stem + ".coverage.json")


def _load_review_metadata(path: Path, review_path: Path, appid: int, language: str, count: int) -> dict | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != 1 or data.get("appid") != appid or data.get("language") != language:
            return None
        if data.get("collected_count") != count or type(data.get("source_exhausted")) is not bool:
            return None
        if data.get("review_file_sha256") != hashlib.sha256(review_path.read_bytes()).hexdigest():
            return None
        return data
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _review_error_code(error: Exception) -> str:
    if isinstance(error, httpx.TimeoutException):
        return "request_timeout"
    if isinstance(error, httpx.HTTPStatusError):
        return "http_" + str(error.response.status_code)
    if isinstance(error, httpx.TransportError):
        return "transport_error"
    if isinstance(error, ValidationError):
        return "validation_error"
    if isinstance(error, RuntimeError):
        return "steam_api_error"
    return "invalid_response"


def prepare_review_snapshot(review_path: Path, appid: int, *, language: str = "schinese",
                            target_count: int = 1000, refresh: bool = False) -> dict:
    """Apply one adequacy policy to focused and legacy store-review collectors."""
    from gamelens.tools.cache_coverage import cache_satisfies_target, candidate_improves_cache

    review_path = Path(review_path)
    metadata_path = review_coverage_path(review_path)
    has_existing = review_path.is_file()
    existing = []
    metadata = None
    metadata_exists = metadata_path.is_file()
    if has_existing:
        existing = [SteamReview.model_validate(item) for item in json.loads(review_path.read_text(encoding="utf-8"))]
        if len({item.review_id for item in existing}) != len(existing):
            raise ValueError("评论缓存包含重复review_id")
        metadata = _load_review_metadata(metadata_path, review_path, appid, language, len(existing))
    scope_valid = bool(has_existing and (language == "all" or all(item.language == language for item in existing)))
    metadata_valid = metadata is not None
    legacy_cache = bool(has_existing and not metadata_exists)
    existing_complete = bool(metadata and metadata["source_exhausted"])
    adequate = bool(scope_valid and (metadata_valid or legacy_cache) and
        cache_satisfies_target(len(existing), target_count, complete=existing_complete))
    if adequate and not refresh:
        return {"reviews": existing, "reused": True, "cache_adequate": True,
            "refresh_attempted": False, "retained_previous_snapshot": False,
            "source_exhausted": existing_complete, "requested_count": target_count,
            "stop_reason": metadata.get("stop_reason") if metadata else "target_reached",
            "refresh_error_code": None, "coverage_path": str(metadata_path)}

    try:
        fetched = fetch_reviews_snapshot(appid=appid, language=language, target_count=target_count)
        candidate = [normalize_review(item) for item in fetched["reviews"]]
        if len({item.review_id for item in candidate}) != len(candidate):
            raise ValueError("新采集评论包含重复review_id")
    except (httpx.HTTPError, RuntimeError, ValidationError, ValueError, TypeError, KeyError) as error:
        if not scope_valid:
            raise
        error_code = _review_error_code(error)
        return {"reviews": existing, "reused": True, "cache_adequate": adequate,
            "refresh_attempted": True, "retained_previous_snapshot": True,
            "source_exhausted": existing_complete, "requested_count": target_count,
            "stop_reason": metadata.get("stop_reason") if metadata else "legacy_cache",
            "refresh_error_code": error_code, "coverage_path": str(metadata_path)}
    candidate_complete = bool(fetched.get("source_exhausted"))
    candidate_error = None if candidate_complete or fetched.get("stop_reason") == "target_reached" else fetched.get("stop_reason")
    existing_error = (metadata.get("error_code") if metadata else
        "invalid_coverage_metadata" if metadata_exists else None)
    candidate_adequate = cache_satisfies_target(len(candidate), target_count, complete=candidate_complete)
    replace = (not scope_valid or
        refresh and candidate_error is None and candidate_adequate or
        candidate_improves_cache(len(existing), len(candidate), target_count,
            existing_complete=existing_complete, candidate_complete=candidate_complete,
            existing_error=existing_error, candidate_error=candidate_error))
    if not replace:
        return {"reviews": existing, "reused": True,
            "cache_adequate": adequate,
            "refresh_attempted": True, "retained_previous_snapshot": True,
            "source_exhausted": existing_complete, "requested_count": target_count,
            "stop_reason": metadata.get("stop_reason") if metadata else "legacy_cache",
            "refresh_error_code": candidate_error or "candidate_not_better", "coverage_path": str(metadata_path)}

    encoded = [review.model_dump(mode="json") for review in candidate]
    atomic_write_json(review_path, encoded)
    metadata = {"schema_version": 1, "appid": appid, "language": language,
        "requested_count": target_count, "collected_count": len(candidate),
        "source_exhausted": candidate_complete, "stop_reason": fetched.get("stop_reason"),
        "error_code": candidate_error, "fetched_at": datetime.now(timezone.utc).isoformat(),
        "review_file_sha256": hashlib.sha256(review_path.read_bytes()).hexdigest()}
    atomic_write_json(metadata_path, metadata)
    return {"reviews": candidate, "reused": False,
        "cache_adequate": cache_satisfies_target(len(candidate), target_count, complete=candidate_complete),
        "refresh_attempted": True, "retained_previous_snapshot": False,
        "source_exhausted": candidate_complete, "requested_count": target_count,
        "stop_reason": fetched.get("stop_reason"), "refresh_error_code": candidate_error,
        "coverage_path": str(metadata_path)}


def normalize_review(review: dict) -> SteamReview:
    """
    从Steam原始数据中提取项目真正需要的字段。

    不保存steamid，避免存储不必要的用户身份信息。
    """
    author = review.get("author", {})
    created_timestamp = review.get("timestamp_created")
    updated_timestamp = review.get("timestamp_updated")

    return SteamReview(
        review_id=review.get("recommendationid"),
        text=review.get("review", ""),
        language=review.get("language"),
        is_positive=review.get("voted_up"),
        votes_up=review.get("votes_up", 0),
        created_at=unix_timestamp_to_iso(created_timestamp),
        updated_at=unix_timestamp_to_iso(updated_timestamp),
        created_at_timestamp=created_timestamp,
        updated_at_timestamp=updated_timestamp,
        playtime_minutes=author.get("playtime_forever", 0),
        playtime_at_review_minutes=author.get(
            "playtime_at_review", 0
        ),
        steam_purchase=review.get("steam_purchase"),
        received_for_free=review.get("received_for_free"),
        early_access=review.get(
            "written_during_early_access"
        ),
    )


def save_reviews(appid: int, reviews: list[SteamReview]) -> Path:
    output_directory = Path("data") / "raw"
    output_directory.mkdir(parents=True, exist_ok=True)

    output_path = output_directory / f"{appid}_reviews.json"

    atomic_write_json(output_path,[review.model_dump(mode="json") for review in reviews])

    return output_path


def main() -> None:
    appid_text = input("请输入Steam游戏AppID：").strip()

    if not appid_text.isdigit():
        print("AppID必须是数字")
        return

    appid = int(appid_text)

    target_count_text = input(
        "请输入希望获取的评论数量（直接回车默认500，最多5000）："
    ).strip()

    if target_count_text:
        if not target_count_text.isdigit():
            print("评论数量必须是数字")
            return
        target_count = int(target_count_text)
    else:
        target_count = 500

    if not 1 <= target_count <= 5000:
        print("评论数量必须在1到5000之间")
        return

    try:
        output_path = Path("data") / "raw" / f"{appid}_reviews.json"
        prepared = prepare_review_snapshot(output_path, appid, language="schinese",
            target_count=target_count, refresh=True)
        reviews = prepared["reviews"]
    except httpx.HTTPError as error:
        print(f"网络请求失败：{error}")
        return
    except RuntimeError as error:
        print(f"接口请求失败：{error}")
        return

    except ValidationError as error:
        print(f"评论数据校验失败：\n{error}")
        return

    positive_count = sum(
        review.is_positive is True
        for review in reviews
    )
    negative_count = len(reviews) - positive_count

    print(f"获取评论数量：{len(reviews)}")
    print(f"好评数量：{positive_count}")
    print(f"差评数量：{negative_count}")
    print(f"数据已保存到：{output_path.resolve()}")

    if reviews:
        print("\n第一条评论：")
        print(json.dumps(
            reviews[0].model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        ))


if __name__ == "__main__":
    main()
