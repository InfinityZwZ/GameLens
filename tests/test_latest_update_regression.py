"""Regressions for latest-patch routing, full bounded windows and count semantics."""
import json
from datetime import datetime, timezone

import httpx

import gamelens.core.investigation_checks as investigation_checks
from gamelens.analysis.report_prose import readable_statistics
from gamelens.evaluation.benchmark import initial_state
from gamelens.rag.embedding import CONFIG
from gamelens.rag.tools import IndexArgs, RagTools, SearchArgs
from gamelens.storage.atomic_io import atomic_write_json
from gamelens.tools.investigation import ListPatchesArgs, MetadataReviewFilter, ReviewInvestigation
from gamelens.tools.steam_news import normalize_news, retained_patch_candidate
from gamelens.tools.steam_review_windows import acquire_window


def stamp(day: str) -> int:
    return int(datetime.fromisoformat(day + "T12:00:00+00:00").timestamp())


def raw_news(gid: str, title: str, day: str, feed: str) -> dict:
    return {
        "gid": gid,
        "title": title,
        "date": stamp(day),
        "contents": title,
        "feedname": feed,
        "feedlabel": "Community Announcements" if feed == "steam_community_announcements" else feed,
        "url": f"https://steamstore-a.akamaihd.net/news/externalpost/{feed}/{gid}",
    }


def test_latest_patch_uses_official_patch_semantics_and_repairs_stale_candidates(tmp_path):
    external = normalize_news(raw_news(
        "media", "Pocketpair is planning a massive update for Palworld 1.1", "2026-10-01", "PC Gamer"), 1623730)
    activity = normalize_news(raw_news(
        "bundle", "Palworld Steam Bundle is Out Now!", "2026-09-28", "steam_community_announcements"), 1623730)
    latest = normalize_news(raw_news(
        "patch-105", "v1.0.5: Bug fixes", "2026-09-15", "steam_community_announcements"), 1623730)
    older = normalize_news(raw_news(
        "patch-104", "v1.0.4: Balance Adjustments & Bug Fixes", "2026-09-10", "steam_community_announcements"), 1623730)

    assert not external.is_patch_candidate
    assert not activity.is_patch_candidate
    assert latest.is_patch_candidate and older.is_patch_candidate

    raw = tmp_path / "data/raw"
    atomic_write_json(raw / "1623730_news.json", [
        item.model_dump(mode="json") for item in (external, activity, latest, older)
    ])
    # Reproduce the old poisoned cache: an external media item was marked as a patch.
    atomic_write_json(raw / "1623730_patch_candidates.json", [
        external.model_copy(update={"is_patch_candidate": True}).model_dump(mode="json")
    ])
    state = initial_state("latest-patch", "整理最近一次更新后的玩家反馈")
    state.update(appid=1623730, artifacts={
        "patches": str(raw / "1623730_patch_candidates.json"),
        "news": str(raw / "1623730_news.json"),
    })

    result = ReviewInvestigation(tmp_path).list_patches(ListPatchesArgs(appid=1623730), state)
    assert result.data["patches"][0]["news_id"] == "patch-105"
    assert result.data["patches"][0]["published_on_local"] == "2026-09-15"
    assert {item["news_id"] for item in result.data["patches"]} == {"patch-105", "patch-104"}
    assert result.data["source"]["path"].endswith("1623730_news.json")


def test_official_activity_updates_and_unverifiable_legacy_media_are_not_patches():
    activities = (
        "Community Update: Halloween Event",
        "Developer Update: Meet the Team",
        "Development Update: Meet the Art Team",
        "Roadmap Update",
        "Server Maintenance Notice",
        "Summer Event Update and Livestream",
    )
    for index, title in enumerate(activities):
        item = normalize_news(raw_news(str(index), title, "2026-10-01", "steam_community_announcements"), 730)
        assert not item.is_patch_candidate, title
    assert normalize_news(raw_news(
        "real-patch", "Counter-Strike 2 Update", "2026-10-01", "steam_community_announcements"), 730).is_patch_candidate
    unverifiable = {
        "title": "Massive update planned for Game 1.1", "feed_name": "", "feed_label": "",
        "url": "https://example.com/media/story", "is_patch_candidate": True,
    }
    assert not retained_patch_candidate(unverifiable)


def review(rid: int, created: int = 150, *, language: str = "schinese", hours: int = 120) -> dict:
    minutes = hours * 60
    return {
        "recommendationid": str(rid),
        "author": {"steamid": "PRIVATE", "playtime_forever": minutes, "playtime_at_review": minutes},
        "language": language,
        "review": f"review {rid}",
        "timestamp_created": created,
        "timestamp_updated": created,
        "voted_up": rid % 2 == 0,
        "votes_up": 1,
        "steam_purchase": True,
        "received_for_free": False,
        "written_during_early_access": False,
    }


def page(records: list[dict], total: int, cursor: str) -> dict:
    positive = total // 2
    return {"response": {
        "query_summary": {"total_positive": positive, "total_negative": total - positive, "total_reviews": total},
        "total_matching": total,
        "reviews": records,
        "cursor": cursor,
    }}


def paged_client(total: int):
    calls = []

    def handler(request):
        query = json.loads(request.url.params["input_json"])
        calls.append(query)
        offset = 0 if query["cursor"] == "*" else int(query["cursor"])
        size = min(query["num_per_page"], total - offset)
        records = [review(offset + index, created=100 + offset + index) for index in range(size)]
        next_offset = offset + size
        return httpx.Response(200, json=page(records, total, str(next_offset)))

    return httpx.Client(transport=httpx.MockTransport(handler)), calls


def test_exact_window_default_collects_all_455_reported_bodies():
    client, calls = paged_client(455)
    result = acquire_window(1623730, 100, 1000, "schinese", client=client, sleep=lambda _: None)
    assert result["reported_total"] == 455
    assert result["body_count"] == 455
    assert result["complete_bodies"] is True
    assert result["stop_reason"] == "endpoint_exhausted"
    assert len(calls) == 5


def test_exact_window_default_has_explicit_1000_body_hard_cap():
    client, calls = paged_client(1200)
    result = acquire_window(1623730, 100, 2000, "schinese", client=client, sleep=lambda _: None)
    assert result["reported_total"] == 1200
    assert result["body_count"] == 1000
    assert result["complete_bodies"] is False
    assert result["stop_reason"] == "sample_budget"
    assert len(calls) == 10


class Encoder:
    def encode(self, texts):
        result = []
        for _ in texts:
            vector = [0.0] * CONFIG["dimension"]
            vector[0] = 1.0
            result.append(vector)
        return result

    def token_count(self, text):
        return len(text)


class RejectAll:
    def generate_json(self, messages):
        candidates = json.loads(messages[1]["content"])["candidates"]
        return {"verdicts": [{
            "kind": item["kind"], "source_id": item["source_id"], "role": "irrelevant",
            "quote": "", "reason": "没有直接提及目标问题",
        } for item in candidates]}


def test_search_distinguishes_hard_filter_population_from_selected_evidence(tmp_path):
    fixture = json.loads((__import__("pathlib").Path(__file__).parents[1] /
        "evals/fixtures/730_reviews.json").read_text(encoding="utf-8"))
    rows = []
    for index, (hours, language) in enumerate(((120, "schinese"), (200, "schinese"), (20, "schinese"))):
        rows.append({**fixture[index], "review_id": f"r{index}", "text": "普通游戏体验，没有目标词",
            "language": language, "playtime_at_review_minutes": hours * 60, "playtime_minutes": hours * 60})
    atomic_write_json(tmp_path / "data/raw/730_reviews.json", rows)
    state = initial_state("count-semantics", "整理中文100小时以上玩家的帆布问题反馈")
    state.update(investigation_profile="focused", rag_mode="auto")
    tools = RagTools(tmp_path, Encoder(), client=RejectAll(), reranker_mode="off")
    tools.build_game_index(IndexArgs(appid=730), state)

    result = tools.search_game_evidence(SearchArgs(appid=730, query="帆布问题", kind="reviews",
        filters=MetadataReviewFilter(period="all", language="schinese", min_playtime_hours=100)), state)

    assert result.data["population"]["eligible_review_count"] == 2
    assert result.data["retrieval"]["candidate_count"] == 2
    assert result.data["retrieval"]["selected_evidence_count"] == 0
    assert result.data["returned_count"] == 0
    assert "硬条件" in result.data["caution"] and "玩家评论为0" in result.data["caution"]
    prose = "\n".join(readable_statistics([{"tool": "search_game_evidence", "ok": True,
        "data": result.data}]))
    assert "硬条件的已采集评论有2条" in prose
    assert "语义证据0条" in prose


def test_finish_guard_rejects_zero_population_claim_when_only_semantic_evidence_is_zero():
    observations = [{"step": 1, "tool": "search_game_evidence", "ok": True, "data": {
        "population": {"eligible_review_count": 113},
        "retrieval": {"candidate_count": 20, "selected_evidence_count": 0},
        "returned_count": 0,
    }}]
    errors = investigation_checks.population_claim_errors(
        observations,
        "未采集到任何符合中文语言且游玩时长100小时以上的玩家评论。",
    )
    assert errors and "语义证据" in errors[0] and "人群" in errors[0]
    assert investigation_checks.population_claim_errors(
        observations,
        "已采集的符合硬条件评论有113条，但本次语义筛选没有保留直接提及帆布问题的证据。",
    ) == []
    for valid_answer in (
        "未找到符合问题语义的玩家评论证据。",
        "没有筛选出直接支持该问题的玩家评论证据。",
    ):
        assert investigation_checks.population_claim_errors(observations, valid_answer) == []


def test_finish_guard_catches_common_and_contradictory_zero_population_phrasing():
    observations = [{"step": 1, "tool": "search_game_evidence", "ok": True, "data": {
        "population": {"eligible_review_count": 113},
        "retrieval": {"candidate_count": 20, "selected_evidence_count": 0},
        "returned_count": 0,
    }}]
    bad_answers = (
        "本次没有符合条件的玩家评论。",
        "符合条件的玩家评论为零条。",
        "There are no player reviews matching the filters.",
        "硬条件的已采集评论有113条，也有证据记录。但本次没有符合条件的玩家评论。",
    )
    for answer in bad_answers:
        assert investigation_checks.population_claim_errors(observations, answer), answer
