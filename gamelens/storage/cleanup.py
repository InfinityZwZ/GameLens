"""Delete only the selected session's own state; shared Steam artifacts survive."""
import re
from pathlib import Path


def delete_conversation(store, conversation_id: str, checkpoint_path: Path,
                        telemetry_path: Path, records_root: Path) -> list[str]:
    turns = store.conversation_runs(conversation_id)
    run_ids = [turn["run_id"] for turn in turns]
    files = []
    root = records_root.resolve()
    for run_id in run_ids:
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", run_id):
            raise ValueError("任务ID异常，未执行删除")
        path = records_root / f"{run_id}.json"
        if path.is_symlink() or path.resolve().parent != root:
            raise ValueError("任务记录路径异常，未执行删除")
        files.append(path)
        for suffix in ("investigation_report","complete_report","partial_report"):
            report=records_root.parent.parent/"reports"/f"{turns[run_ids.index(run_id)]['request']['appid']}_{run_id}_{suffix}.md"
            if report.is_symlink() or report.resolve().parent != (records_root.parent.parent/"reports").resolve():
                raise ValueError("任务报告路径异常")
            files.append(report)
        for directory in (records_root.parent/"tool_receipts"/run_id,records_root.parent/"investigations"/run_id):
            if directory.is_symlink() or not directory.resolve().is_relative_to(records_root.parent.resolve()):
                raise ValueError("任务专属目录异常，未执行删除")
            if directory.is_dir():
                for artifact in directory.glob("*.json"):
                    if artifact.is_symlink() or artifact.resolve().parent != directory.resolve():
                        raise ValueError("任务产物路径异常，未执行删除")
                    files.append(artifact)
    with store._connect() as db:
        attached = []
        for alias,path in (("cp",checkpoint_path),("tm",telemetry_path)):
            if path.is_file():
                db.execute(f"ATTACH DATABASE ? AS {alias}", (str(path),))
                attached.append(alias)
        # Only parameterized thread IDs and explicitly named tables are deleted.
        if "cp" in attached:
            tables = {row[0] for row in db.execute("SELECT name FROM cp.sqlite_master WHERE type='table'")}
            for table in ("checkpoints", "writes"):
                if table in tables:
                    db.executemany(f"DELETE FROM cp.{table} WHERE thread_id=?", [(id,) for id in run_ids])
        if "tm" in attached:
            db.executemany("DELETE FROM tm.events WHERE thread_id=?", [(id,) for id in run_ids])
        db.execute("DELETE FROM web_runs WHERE conversation_id=?", (conversation_id,))
        db.execute("DELETE FROM conversations WHERE conversation_id=?", (conversation_id,))
        # On an export-file error keep database history/checkpoints retriable.
        for path in files:
            path.unlink(missing_ok=True)
    return run_ids
