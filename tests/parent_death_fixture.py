"""Used only by the parent-death test; no model or network calls."""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pydantic import BaseModel
from gamelens.core.runtime import AgentTool,ToolRegistry,ToolResult


class Args(BaseModel): pass


def handler(args,state):
    root=Path(state["test_root"])
    (root/"ready.txt").write_text(str(os.getpid()))
    time.sleep(3)
    (root/"late.txt").write_text("orphan write")
    return ToolResult(ok=True,summary="done")


if __name__=="__main__":
    ToolRegistry([AgentTool("slow","test",Args,handler)]).execute("slow",{},
        {"thread_id":"parent-death","step_count":0,"test_root":sys.argv[1]})
