"""Explicit local model preparation and per-game index maintenance."""
import argparse
import json
from gamelens.core.paths import PROJECT_ROOT
from gamelens.rag.embedding import LocalEmbedding
from gamelens.rag.index import GameIndex


def main():
    parser=argparse.ArgumentParser(description="游鉴本地RAG管理；不调用收费API")
    parser.add_argument("action",choices=["download","build","status"])
    parser.add_argument("--appid",type=int)
    args=parser.parse_args()
    if args.action=="download":
        LocalEmbedding(PROJECT_ROOT).load(download=True)
        print("免费本地BGE-M3模型已就绪")
        return
    if args.appid is None or args.appid<=0: parser.error("需要正整数--appid")
    index=GameIndex(PROJECT_ROOT)
    state={"appid":args.appid,"artifacts":{}}
    print(json.dumps(index.build(args.appid,state) if args.action=="build" else index.status(args.appid,state),ensure_ascii=False,indent=2))


if __name__=="__main__": main()
