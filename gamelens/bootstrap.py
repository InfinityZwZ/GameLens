"""Standard-library-only startup and environment diagnostics."""
import json
import os
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_MODULES = ("httpx","pydantic","dotenv","fastapi","uvicorn","langgraph",
                "langgraph.checkpoint.sqlite","cloudpickle","filelock")
RAG_MODULES = ("sentence_transformers","torch","numpy")


def project_python(root=ROOT):
    root=Path(root)
    path=root/".venv"/("Scripts/python.exe" if os.name=="nt" else "bin/python")
    return str(path) if path.is_file() else sys.executable


@lru_cache(maxsize=8)
def probe_environment(python,include_rag=False):
    # Actual imports, not merely installed distribution names. No .env/model/API access.
    modules=BASE_MODULES+(RAG_MODULES if include_rag else ())
    with tempfile.TemporaryDirectory(prefix="gamelens-env-") as directory:
        report_file=Path(directory)/"report.json"
        environment={**os.environ,"PYTHONIOENCODING":"utf-8"}
        result=subprocess.run([python,str(ROOT/"gamelens/_environment_probe.py"),"--output",str(report_file),"--modules",json.dumps(modules)],
            capture_output=True,text=True,encoding="utf-8",errors="replace",env=environment,
            timeout=60,creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
        if result.returncode!=0:raise RuntimeError(f"Python环境检查失败：{python}，退出码{result.returncode}")
        try:
            report=json.loads(report_file.read_text(encoding="utf-8"))
            valid=(isinstance(report,dict) and report.get("protocol")==1 and isinstance(report.get("python"),str)
                   and isinstance(report.get("prefix"),str) and isinstance(report.get("errors"),dict)
                   and all(isinstance(k,str) and isinstance(v,str) for k,v in report["errors"].items()))
            if not valid:raise ValueError("invalid report schema")
            return report
        except (OSError,ValueError) as error:
            raise RuntimeError(f"环境诊断文件未生成或格式无效：{python}。这不等于解释器选错，请保留运行日志排查。") from error


def require_environment(python,include_rag=False):
    report=probe_environment(python,include_rag)
    if report["errors"]:
        requirement="requirements-rag.txt" if include_rag else "requirements.txt"
        missing="、".join(f"{name} ({error})" for name,error in report["errors"].items())
        raise RuntimeError(f"运行环境缺少或无法加载依赖：{missing}。实际Python：{report['python']}。请执行：& \"{python}\" -m pip install -r {requirement}")
    return report


def ensure_environment():
    preferred=project_python()
    if os.path.normcase(os.path.abspath(sys.executable))!=os.path.normcase(os.path.abspath(preferred)):
        print(f"GameLens 自动使用项目虚拟环境：{preferred}",flush=True)
        original=getattr(sys,"orig_argv",[])
        arguments=original[1:] if "-m" in original[:3] else sys.argv
        result=subprocess.run([preferred,*arguments],stdin=sys.stdin,stdout=sys.stdout,stderr=sys.stderr,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
        raise SystemExit(result.returncode)
    try:require_environment(preferred)
    except (RuntimeError,subprocess.TimeoutExpired,OSError) as error:raise SystemExit(str(error)) from error


def main():
    import argparse
    parser=argparse.ArgumentParser(description="GameLens环境检查，不调用模型API")
    parser.add_argument("--rag",action="store_true")
    args=parser.parse_args()
    report=require_environment(project_python(),args.rag)
    print(json.dumps({"launcher_python":sys.executable,"task_environment":report},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
