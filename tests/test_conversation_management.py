import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gamelens.api import app as web_app
from gamelens.storage.runs import RunStore
from gamelens.storage.telemetry import TelemetryStore
from gamelens.storage.cleanup import delete_conversation


def payload(id="run-a",cid="session-a",status="completed"):
    return {"run_id":id,"conversation_id":cid,"turn_index":1,"status":status,
            "request":{"appid":730,"goal":"游戏版本反馈调查"},"started_at":"2026-10-07T00:00:00Z",
            "updated_at":"2026-10-07T00:00:00Z","final_answer":"样本证据不足","artifacts":{},"events":[]}


def test_rename_pin_archive_survive_reload_and_later_run_save(tmp_path):
    store = RunStore(tmp_path / "web.sqlite")
    store.save(payload())
    store.save(payload("run-b","session-b"))
    store.update_conversation("session-a",{"title":"CS2 崩溃问题","pinned":True},"2026-10-07T00:00:01Z")
    store.save(payload())
    reloaded = RunStore(store.database)
    assert reloaded.conversation("session-a")["title"] == "CS2 崩溃问题"
    assert reloaded.list_conversations()[0]["conversation_id"] == "session-a"
    reloaded.update_conversation("session-a",{"archived":True},"2026-10-07T00:00:02Z")
    assert len(reloaded.list_conversations(view="active")) == 1
    assert reloaded.list_conversations(view="archived",query="崩溃")[0]["title"] == "CS2 崩溃问题"
    assert len(reloaded.list_conversations(view="all",query="730")) == 2
    reloaded.update_conversation("session-a",{"archived":False,"pinned":False},"2026-10-07T00:00:03Z")
    assert len(reloaded.list_conversations()) == 2


def test_permanent_delete_cleans_only_session_owned_records(tmp_path):
    store = RunStore(tmp_path / "web.sqlite")
    store.save(payload())
    store.save(payload("run-b","session-b"))
    telemetry = TelemetryStore(tmp_path / "telemetry.sqlite")
    for id in ("run-a","run-b"):
        telemetry.record(id,{"id":id,"kind":"call","call_id":id,"model":"test","status":"succeeded"})
    checkpoints = tmp_path / "checkpoints.sqlite"
    with sqlite3.connect(checkpoints) as db:
        for table in ("checkpoints","writes"):
            db.execute(f"CREATE TABLE {table}(thread_id TEXT, value TEXT)")
            db.executemany(f"INSERT INTO {table} VALUES (?,?)",[("run-a","private-a"),("run-b","private-b")])
    records = tmp_path / "runs"
    records.mkdir()
    for id in ("run-a","run-b"):
        (records/f"{id}.json").write_text("{}",encoding="utf-8")
    shared = tmp_path / "reports" / "shared.md"
    shared.parent.mkdir()
    shared.write_text("shared report",encoding="utf-8")
    assert delete_conversation(store,"session-a",checkpoints,telemetry.path,records) == ["run-a"]
    assert store.conversation("session-a") is None
    assert store.get("run-a") is None
    assert telemetry.events("run-a") == []
    assert not (records/"run-a.json").exists()
    assert store.get("run-b") is not None
    assert telemetry.events("run-b")
    assert (records/"run-b.json").exists()
    assert shared.read_text() == "shared report"
    with sqlite3.connect(checkpoints) as db:
        assert db.execute("SELECT thread_id FROM checkpoints").fetchall() == [("run-b",)]
        assert db.execute("SELECT thread_id FROM writes").fetchall() == [("run-b",)]


def test_management_api_guards_and_permanent_delete(tmp_path,monkeypatch):
    store = RunStore(tmp_path/"web.sqlite")
    monkeypatch.setattr(web_app,"RUN_STORE",store)
    monkeypatch.setattr(web_app,"TELEMETRY_STORE",TelemetryStore(tmp_path/"telemetry.sqlite"))
    monkeypatch.setattr(web_app,"PROJECT_ROOT",tmp_path)
    monkeypatch.setattr(web_app,"RUNS",{})
    store.save(payload(status="awaiting_approval"))
    client = TestClient(web_app.app)
    assert client.delete("/api/conversations/session-a").status_code == 409
    assert client.patch("/api/conversations/session-a",json={"title":"  "}).status_code == 422
    assert client.patch("/api/conversations/session-a",json={"title":None}).status_code == 422
    assert client.patch("/api/conversations/session-a",json={"unknown":True}).status_code == 422
    assert client.patch("/api/conversations/session-a",json={"title":"新标题","pinned":True}).json()["title"] == "新标题"
    client.patch("/api/conversations/session-a",json={"archived":True})
    assert client.get("/api/conversations").json() == []
    assert len(client.get("/api/conversations?view=archived").json()) == 1
    assert client.post("/api/runs",json={"appid":730,"goal":"继续这次调查","conversation_id":"session-a"}).status_code == 409
    store.save(payload(status="completed"))
    assert client.delete("/api/conversations/session-a").json()["deleted"] is True
    assert client.get("/api/conversations/session-a").status_code == 404
    assert client.get("/api/runs/run-a").status_code == 404
    assert client.delete("/api/conversations/session-a").status_code == 404
