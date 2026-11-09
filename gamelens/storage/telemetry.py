"""Durable, content-free model request telemetry and explicitly priced estimates."""

import json
import math
import sqlite3
from pathlib import Path


class TelemetryStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, thread_id TEXT NOT NULL, payload TEXT NOT NULL)")
            db.execute("CREATE INDEX IF NOT EXISTS events_thread ON events(thread_id)")

    def connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def record(self, thread_id: str, event: dict):
        with self.connect() as db:
            db.execute("INSERT INTO events VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
                       (event["id"], thread_id, json.dumps(event, ensure_ascii=False)))

    def events(self, thread_id: str) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("SELECT payload FROM events WHERE thread_id=? ORDER BY rowid", (thread_id,)).fetchall()
        return [json.loads(row[0]) for row in rows]


def token_count(value):
    return value if type(value) is int and value >= 0 else None


def summarize(events: list[dict]) -> dict:
    requests = [e for e in events if e["kind"] == "request"]
    calls = [e for e in events if e["kind"] == "call"]
    known = [e for e in requests if e.get("prompt_tokens") is not None and e.get("completion_tokens") is not None]
    priced = [e for e in requests if e.get("estimated_cost") is not None]
    currencies = sorted({e["currency"] for e in priced})
    complete_cost = bool(requests) and len(priced) == len(requests) and len(currencies) == 1
    context_sizes = [e["prompt_chars"] for e in calls if isinstance(e.get("prompt_chars"), int)]
    return {
        "model_calls": len({e["call_id"] for e in events}),
        "http_requests": len(requests),
        "retries": sum(e.get("retry", False) for e in requests),
        "fallbacks": sum(e.get("fallback", False) for e in requests),
        "failed_requests": sum(e.get("status") == "failed" for e in requests),
        "failed_calls": sum(e.get("status") == "failed" for e in calls),
        "pending_requests": sum(e.get("status") in {"pending", "streaming"} for e in requests),
        "prompt_tokens": sum(e["prompt_tokens"] for e in known) if known else None,
        "completion_tokens": sum(e["completion_tokens"] for e in known) if known else None,
        "usage_known_requests": len(known),
        "usage_complete": bool(requests) and len(known) == len(requests),
        "estimated_cost": round(sum(e["estimated_cost"] for e in priced), 8) if complete_cost else None,
        "known_cost_subtotal": round(sum(e["estimated_cost"] for e in priced), 8) if priced and len(currencies) == 1 else None,
        "currency": currencies[0] if len(currencies) == 1 else None,
        "latency_ms": round(sum(e.get("latency_ms", 0) for e in requests), 1),
        "models": sorted({e["model"] for e in requests}),
        "prompt_chars_total": sum(context_sizes) if context_sizes else None,
        "prompt_chars_peak": max(context_sizes) if context_sizes else None,
    }


def price_from_env(environment) -> dict:
    def rate(key):
        raw = environment.get(key, "").strip()
        if not raw:
            return None
        value = float(raw)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key}必须是有限非负数字")
        return value
    return {
        "input_per_million": rate("LLM_INPUT_PRICE_PER_MILLION"),
        "output_per_million": rate("LLM_OUTPUT_PRICE_PER_MILLION"),
        "currency": environment.get("LLM_PRICE_CURRENCY", "CNY").strip() or "CNY",
    }
