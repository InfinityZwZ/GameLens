"""Standalone import probe; report transport is independent of console output."""
import argparse
import importlib
import json
import sys
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--modules",required=True)
    args=parser.parse_args()
    errors={}
    for name in json.loads(args.modules):
        try:importlib.import_module(name)
        except Exception as error:errors[name]=type(error).__name__
    report={"protocol":1,"python":sys.executable,"prefix":sys.prefix,"errors":errors}
    # Third-party imports and IDE debuggers may write unrelated stdout/stderr.
    args.output.write_text(json.dumps(report,ensure_ascii=False),encoding="utf-8")


if __name__=="__main__":main()
