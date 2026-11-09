"""SQLite-backed persistence for Web Agent run metadata."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class RunStore:
    """Persist JSON-compatible run snapshots with one connection per call."""

    def __init__(self, database: Path) -> None:
        self.database = database
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS web_runs (
                    run_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_web_runs_updated_at
                ON web_runs(updated_at DESC)
                """
            )
            connection.execute("""CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY, appid INTEGER NOT NULL,
                title TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            )""")
            conversation_columns = {row[1] for row in connection.execute("PRAGMA table_info(conversations)")}
            for column in ("pinned", "archived"):
                if column not in conversation_columns:
                    connection.execute(f"ALTER TABLE conversations ADD COLUMN {column} INTEGER NOT NULL DEFAULT 0")
            columns = {row[1] for row in connection.execute("PRAGMA table_info(web_runs)")}
            if "conversation_id" not in columns:
                connection.execute("ALTER TABLE web_runs ADD COLUMN conversation_id TEXT")
            if "turn_index" not in columns:
                connection.execute("ALTER TABLE web_runs ADD COLUMN turn_index INTEGER DEFAULT 1")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_web_runs_conversation ON web_runs(conversation_id, turn_index)")
            # Migrate old independent tasks into one-turn conversations, preserving IDs.
            for row in connection.execute("SELECT run_id,payload FROM web_runs WHERE conversation_id IS NULL").fetchall():
                payload = json.loads(row["payload"])
                cid = payload.get("conversation_id") or row["run_id"]
                payload.update(conversation_id=cid, turn_index=1)
                connection.execute("INSERT OR IGNORE INTO conversations(conversation_id,appid,title,created_at,updated_at) VALUES (?,?,?,?,?)",
                    (cid,payload["request"]["appid"],payload["request"]["goal"],payload["started_at"],payload["updated_at"]))
                connection.execute("UPDATE web_runs SET conversation_id=?,turn_index=1,payload=? WHERE run_id=?",
                    (cid,json.dumps(payload,ensure_ascii=False),row["run_id"]))

    def save(self, payload: dict[str, Any]) -> None:
        payload = dict(payload)
        cid = payload.get("conversation_id") or payload["run_id"]
        turn_index = payload.get("turn_index", 1)
        payload.update(conversation_id=cid, turn_index=turn_index)
        encoded = json.dumps(payload, ensure_ascii=False)
        with self._connect() as connection:
            connection.execute("INSERT OR IGNORE INTO conversations(conversation_id,appid,title,created_at,updated_at) VALUES (?,?,?,?,?)",
                (cid,payload["request"]["appid"],payload["request"]["goal"],payload["started_at"],payload["updated_at"]))
            connection.execute(
                """
                INSERT INTO web_runs (
                    run_id, status, started_at, updated_at, payload, conversation_id, turn_index
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    status = excluded.status,
                    updated_at = excluded.updated_at,
                    payload = excluded.payload,
                    conversation_id = excluded.conversation_id,
                    turn_index = excluded.turn_index
                """,
                (
                    payload["run_id"],
                    payload["status"],
                    payload["started_at"],
                    payload["updated_at"],
                    encoded,
                    cid,
                    turn_index,
                ),
            )
            connection.execute("UPDATE conversations SET updated_at=? WHERE conversation_id=?", (payload["updated_at"],cid))

    def conversation(self, conversation_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM conversations WHERE conversation_id=?", (conversation_id,)).fetchone()
        return dict(row) if row is not None else None

    def conversation_runs(self, conversation_id: str) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute("SELECT payload FROM web_runs WHERE conversation_id=? ORDER BY turn_index, started_at", (conversation_id,)).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def list_conversations(self, limit: int = 30, view: str = "active", query: str = "") -> list[dict]:
        clause = "AND c.archived=0" if view == "active" else "AND c.archived=1" if view == "archived" else ""
        pattern = "%" + query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        with self._connect() as connection:
            rows = connection.execute(f"""SELECT c.*, r.status, r.run_id AS last_run_id,
                json_extract(r.payload, '$.investigation_progress.outcome') AS investigation_outcome,
                r.turn_index AS turn_count FROM conversations c JOIN web_runs r
                ON r.conversation_id=c.conversation_id WHERE r.turn_index=(
                    SELECT MAX(turn_index) FROM web_runs WHERE conversation_id=c.conversation_id
                ) {clause} AND (c.title LIKE ? ESCAPE '\\' OR CAST(c.appid AS TEXT) LIKE ? ESCAPE '\\')
                ORDER BY c.pinned DESC,c.updated_at DESC LIMIT ?""", (pattern,pattern,limit)).fetchall()
        return [dict(row) for row in rows]

    def update_conversation(self, conversation_id: str, changes: dict, updated_at: str) -> dict:
        allowed = {"title", "pinned", "archived"}
        if not changes or not set(changes) <= allowed:
            raise ValueError("不支持的会话更新字段")
        columns = [f"{key}=?" for key in changes]
        with self._connect() as connection:
            cursor = connection.execute(f"UPDATE conversations SET {','.join(columns)},updated_at=? WHERE conversation_id=?",
                (*changes.values(),updated_at,conversation_id))
            if not cursor.rowcount:
                raise KeyError(conversation_id)
        return self.conversation(conversation_id)

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM web_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return json.loads(row["payload"]) if row is not None else None

    def list_recent(self, limit: int = 30) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload
                FROM web_runs
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def delete(self, run_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM web_runs WHERE run_id = ?",
                (run_id,),
            )
