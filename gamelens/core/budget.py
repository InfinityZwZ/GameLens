"""Durable per-task execution budgets with cross-process atomic reservation."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path


class BudgetExceeded(RuntimeError):
    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


class TaskBudgetStore:
    """SQLite budget ledger. Existing rows retain the limits they started with."""

    def __init__(self, path: Path, *, request_limit: int = 32,
                 active_seconds_limit: float = 1800) -> None:
        if request_limit < 1:
            raise ValueError("LLM请求预算必须大于0")
        if active_seconds_limit <= 0:
            raise ValueError("活跃执行时间预算必须大于0")
        self.path = Path(path)
        self.request_limit = request_limit
        self.active_seconds_limit = float(active_seconds_limit)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as database:
            database.execute("""
                CREATE TABLE IF NOT EXISTS task_budgets (
                    thread_id TEXT PRIMARY KEY,
                    llm_request_limit INTEGER NOT NULL,
                    llm_requests_reserved INTEGER NOT NULL DEFAULT 0,
                    active_seconds_limit REAL NOT NULL,
                    active_elapsed_seconds REAL NOT NULL DEFAULT 0,
                    active_started_at REAL,
                    updated_at REAL NOT NULL
                )
            """)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure(self, database: sqlite3.Connection, thread_id: str, now: float) -> None:
        if not thread_id or len(thread_id) > 128:
            raise ValueError("任务ID无效")
        database.execute("""
            INSERT OR IGNORE INTO task_budgets
            (thread_id,llm_request_limit,llm_requests_reserved,active_seconds_limit,
             active_elapsed_seconds,active_started_at,updated_at)
            VALUES (?,?,0,?,0,NULL,?)
        """, (thread_id, self.request_limit, self.active_seconds_limit, now))

    @staticmethod
    def _payload(row: sqlite3.Row, now: float | None = None) -> dict:
        elapsed = float(row["active_elapsed_seconds"])
        if now is not None and row["active_started_at"] is not None:
            elapsed += max(0.0, now - float(row["active_started_at"]))
        return {
            "thread_id": row["thread_id"],
            "llm_request_limit": int(row["llm_request_limit"]),
            "llm_requests_reserved": int(row["llm_requests_reserved"]),
            "llm_requests_remaining": max(0, int(row["llm_request_limit"]) - int(row["llm_requests_reserved"])),
            "active_seconds_limit": float(row["active_seconds_limit"]),
            "active_elapsed_seconds": round(elapsed, 3),
            "active_seconds_remaining": round(max(0.0, float(row["active_seconds_limit"]) - elapsed), 3),
            "active": row["active_started_at"] is not None,
        }

    def reserve_llm_request(self, thread_id: str) -> dict:
        now = time.time()
        with self._connect() as database:
            database.execute("BEGIN IMMEDIATE")
            self._ensure(database, thread_id, now)
            row = database.execute("SELECT * FROM task_budgets WHERE thread_id=?", (thread_id,)).fetchone()
            if row["llm_requests_reserved"] >= row["llm_request_limit"]:
                database.rollback()
                raise BudgetExceeded("本任务LLM HTTP请求预算已耗尽", code="llm_request_budget_exhausted")
            database.execute("""
                UPDATE task_budgets SET llm_requests_reserved=llm_requests_reserved+1,updated_at=?
                WHERE thread_id=?
            """, (now, thread_id))
            database.commit()
        return self.snapshot(thread_id)

    def resume_active(self, thread_id: str) -> dict:
        now = time.time()
        with self._connect() as database:
            database.execute("BEGIN IMMEDIATE")
            self._ensure(database, thread_id, now)
            # A stale marker belongs to a terminated process. Resetting it makes
            # offline/crash recovery time explicitly non-billable active time.
            database.execute("UPDATE task_budgets SET active_started_at=?,updated_at=? WHERE thread_id=?",
                             (now, now, thread_id))
            database.commit()
        return self.snapshot(thread_id)

    def checkpoint_active(self, thread_id: str) -> dict:
        now = time.time()
        with self._connect() as database:
            database.execute("BEGIN IMMEDIATE")
            self._ensure(database, thread_id, now)
            row = database.execute("SELECT * FROM task_budgets WHERE thread_id=?", (thread_id,)).fetchone()
            elapsed = float(row["active_elapsed_seconds"])
            if row["active_started_at"] is not None:
                elapsed += max(0.0, now - float(row["active_started_at"]))
            database.execute("UPDATE task_budgets SET active_elapsed_seconds=?,active_started_at=?,updated_at=? WHERE thread_id=?",
                             (elapsed, now if row["active_started_at"] is not None else None, now, thread_id))
            database.commit()
            limit = float(row["active_seconds_limit"])
        if elapsed >= limit:
            raise BudgetExceeded("本任务活跃执行时间预算已耗尽", code="active_time_budget_exhausted")
        return self.snapshot(thread_id)

    def pause_active(self, thread_id: str) -> dict:
        now = time.time()
        with self._connect() as database:
            database.execute("BEGIN IMMEDIATE")
            self._ensure(database, thread_id, now)
            row = database.execute("SELECT * FROM task_budgets WHERE thread_id=?", (thread_id,)).fetchone()
            elapsed = float(row["active_elapsed_seconds"])
            if row["active_started_at"] is not None:
                elapsed += max(0.0, now - float(row["active_started_at"]))
            database.execute("UPDATE task_budgets SET active_elapsed_seconds=?,active_started_at=NULL,updated_at=? WHERE thread_id=?",
                             (elapsed, now, thread_id))
            database.commit()
        return self.snapshot(thread_id)

    def snapshot(self, thread_id: str) -> dict:
        now = time.time()
        with self._connect() as database:
            self._ensure(database, thread_id, now)
            row = database.execute("SELECT * FROM task_budgets WHERE thread_id=?", (thread_id,)).fetchone()
        return self._payload(row, now)

