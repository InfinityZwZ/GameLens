import sqlite3
from pathlib import Path
import subprocess
import sys
import time

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from gamelens.core.runtime import AgentTool,ToolRegistry,ToolResult,InvestigationAgent,PlannerDecision
from gamelens.storage.atomic_io import atomic_write_text
from test_agent_runtime import make_state,EmptyArgs,SequencePlanner


def test_atomic_write_failure_preserves_existing_file(tmp_path,monkeypatch):
    from gamelens.storage import atomic_io as atomic_io
    file=tmp_path/"existing.json"
    file.write_text('{"valid":true}',encoding="utf-8")
    def fail(*args): raise OSError("simulated replace failure")
    monkeypatch.setattr(atomic_io.os,"replace",fail)
    with pytest.raises(OSError): atomic_write_text(file,"broken")
    assert file.read_text()== '{"valid":true}'
    assert list(tmp_path.glob("*.tmp"))==[]


def test_recovery_replays_completed_receipt_without_repeating_side_effect(tmp_path):
    counter=tmp_path/"counter.txt"
    report=tmp_path/"report.md"
    def handler(args,state):
        value=int(counter.read_text())+1 if counter.exists() else 1
        atomic_write_text(counter,str(value))
        atomic_write_text(report,"proof")
        return ToolResult(ok=True,summary="done",artifacts={"final_report":str(report),"evidence_assessed":"true"})
    registry=ToolRegistry([AgentTool("create_report","生成",EmptyArgs,handler)],receipt_root=tmp_path/"receipts")
    database=tmp_path/"checkpoint.sqlite"
    connection=sqlite3.connect(database,check_same_thread=False)
    first=InvestigationAgent(SequencePlanner([PlannerDecision(action="create_report",reason="执行报告")]),registry,checkpointer=SqliteSaver(connection))
    stream=first.stream(make_state())
    next(stream)
    next(stream)  # Persist next=execute, then simulate a crash before that node commits.
    stream.close()
    snapshot=first.snapshot("test-thread")
    assert snapshot.next==("execute",)
    result=registry.execute("create_report",{},snapshot.values)
    assert result.ok
    connection.close()
    second_connection=sqlite3.connect(database,check_same_thread=False)
    restored_registry=ToolRegistry([AgentTool("create_report","生成",EmptyArgs,handler)],receipt_root=tmp_path/"receipts")
    second=InvestigationAgent(SequencePlanner([PlannerDecision(action="finish",reason="恢复后完成")]),restored_registry,checkpointer=SqliteSaver(second_connection))
    list(second.recover_stream("test-thread"))
    completed=second.snapshot("test-thread").values
    second_connection.close()
    assert completed["status"]=="completed"
    assert counter.read_text()=="1"
    assert completed["observations"][0]["data"]["execution"]["from_receipt"] is True


def test_terminated_parent_leaves_no_tool_worker_writes(tmp_path):
    process=subprocess.Popen([sys.executable,str(Path(__file__).with_name("parent_death_fixture.py")),str(tmp_path)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        deadline=time.monotonic()+15
        while not (tmp_path/"ready.txt").exists() and time.monotonic()<deadline:
            time.sleep(0.05)
        assert (tmp_path/"ready.txt").exists()
        process.terminate()
        process.wait(timeout=5)
        time.sleep(3.3)
        assert not (tmp_path/"late.txt").exists()
    finally:
        if process.poll() is None: process.kill(); process.wait()
