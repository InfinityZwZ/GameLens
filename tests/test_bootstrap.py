import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from gamelens import bootstrap


def test_project_python_prefers_local_venv(tmp_path):
    candidate=tmp_path/".venv"/("Scripts/python.exe" if os.name=="nt" else "bin/python")
    candidate.parent.mkdir(parents=True);candidate.touch()
    assert bootstrap.project_python(tmp_path)==str(candidate)


def test_missing_dependency_message_is_actionable(monkeypatch):
    monkeypatch.setattr(bootstrap,"probe_environment",lambda *args:{"python":"project-python","errors":{"cloudpickle":"ModuleNotFoundError"}})
    with pytest.raises(RuntimeError,match="cloudpickle") as error:bootstrap.require_environment("project-python")
    assert "pip install -r requirements.txt" in str(error.value)


@pytest.fixture
def runner(tmp_path,monkeypatch):
    from gamelens.api import app as module
    from gamelens.storage.runs import RunStore
    state=module.RunState(run_id="environment-test",request=module.RunRequest(appid=730,rag_mode="on"))
    monkeypatch.setattr(module,"RUN_STORE",RunStore(tmp_path/"runs.sqlite"))
    monkeypatch.setattr(module,"RUNS",{state.run_id:state})
    monkeypatch.setattr(module,"ACTIVE_PROCESSES",{})
    return module,state


def test_task_does_not_inherit_global_web_interpreter(runner,monkeypatch):
    module,state=runner;commands=[]
    monkeypatch.setattr(module,"project_python",lambda root:"project-venv-python")
    checks=[]
    monkeypatch.setattr(module,"require_environment",lambda python,include_rag:checks.append((python,include_rag)) or {"python":python})
    class Process:
        stdout=['AGENT_FINAL '+json.dumps({"status":"completed","answer":"offline process test","artifacts":{}})+'\n']
        def wait(self):return 0
    monkeypatch.setattr(module.subprocess,"Popen",lambda command,**kwargs:commands.append(command) or Process())
    module.execute_agent(state.run_id,state.request)
    assert commands[0][0]=="project-venv-python"
    assert checks==[("project-venv-python",True)]
    assert state.status=="completed"
    assert "project-venv-python" in " ".join(state.logs)


def test_failed_preflight_never_launches_agent(runner,monkeypatch):
    module,state=runner
    def fail(*args,**kwargs):raise RuntimeError("missing cloudpickle")
    monkeypatch.setattr(module,"require_environment",fail)
    monkeypatch.setattr(module.subprocess,"Popen",lambda *args,**kwargs:pytest.fail("must not launch"))
    module.execute_agent(state.run_id,state.request)
    assert state.status=="failed" and "cloudpickle" in state.final_answer


def test_cancel_during_preflight_finishes_cancelled(runner,monkeypatch):
    module,state=runner
    def check(*args,**kwargs):state.cancel_requested=True;return {"python":"project-python"}
    monkeypatch.setattr(module,"require_environment",check)
    monkeypatch.setattr(module.subprocess,"Popen",lambda *args,**kwargs:pytest.fail("must not launch"))
    module.execute_agent(state.run_id,state.request)
    assert state.status=="cancelled" and state.finished_at


def test_actual_legacy_python_launcher_switches_without_api_call():
    root=Path(__file__).resolve().parents[1]
    if not (root/".venv").is_dir():pytest.skip("project venv not configured")
    environment={**os.environ,"PYTHONIOENCODING":"utf-8"}
    result=subprocess.run([getattr(sys,"_base_executable",sys.executable),str(root/"investigation_agent.py"),"--help"],
        cwd=root,env=environment,capture_output=True,text=True,encoding="utf-8",timeout=30)
    assert result.returncode==0,result.stderr
    assert "--rag-mode" in result.stdout


@pytest.mark.parametrize("console",["", "debugger attached\nnot json",'{"python":"wrong-console"}\nIDE warning\n'])
def test_probe_ignores_empty_or_polluted_console(monkeypatch,console):
    bootstrap.probe_environment.cache_clear()
    def run(command,**kwargs):
        output=Path(command[command.index("--output")+1])
        output.write_text(json.dumps({"protocol":1,"python":"correct-python","prefix":"correct-env","errors":{}}),encoding="utf-8")
        return subprocess.CompletedProcess(command,0,stdout=console,stderr="unrelated warning")
    monkeypatch.setattr(bootstrap.subprocess,"run",run)
    report=bootstrap.probe_environment("test-python",True)
    assert report["python"]=="correct-python" and report["errors"]=={}
    bootstrap.probe_environment.cache_clear()


def test_probe_missing_report_never_counts_as_success(monkeypatch):
    bootstrap.probe_environment.cache_clear()
    monkeypatch.setattr(bootstrap.subprocess,"run",lambda *args,**kwargs:subprocess.CompletedProcess(args[0],0,stdout='{"errors":{}}',stderr=""))
    with pytest.raises(RuntimeError,match="诊断文件"):bootstrap.probe_environment("test-python")
    bootstrap.probe_environment.cache_clear()
