"""Unified CLI; old benchmark.py --mode commands remain compatible."""
import argparse
import json
from pathlib import Path

from gamelens.core.paths import PROJECT_ROOT
from gamelens.evaluation.benchmark_data import prepare,validate,export_review_pack,import_reviews,read,migrate_timezone
from gamelens.evaluation.benchmark_score import compare
from gamelens.evaluation.benchmark_runner import run_oracle,run_retrieval,replay,run_agent
from gamelens.storage.atomic_io import atomic_write_json,atomic_write_text

COMMANDS={'prepare','timezone-migrate','validate','review-export','review-import','retrieval','oracle','agent','fixed-workflow','replay','compare','public-export','judge-export','judge','suite','status'}


def main(argv=None):
    parser=argparse.ArgumentParser(description='GameLens Benchmark v1：默认冻结资料/离线评测')
    subs=parser.add_subparsers(dest='command',required=True)
    p=subs.add_parser('status');p.add_argument('--bundle',type=Path,required=True)
    p=subs.add_parser('suite');p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-paid',action='store_true');p.add_argument('--agent-max-http',type=int);p.add_argument('--judge-max-http',type=int)
    p.add_argument('--judge-output-tokens',type=int,help='完整批次评委输出预算512..8192；默认沿用协议配置')
    p.add_argument('--resume-from',type=Path,help='从已完成离线阶段的原批次续跑；原账本上限不变')
    p=subs.add_parser('prepare');p.add_argument('--project',type=Path,default=PROJECT_ROOT);p.add_argument('--output',type=Path,required=True)
    p=subs.add_parser('timezone-migrate');p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=subs.add_parser('judge-export');p.add_argument('--bundle',type=Path,required=True);p.add_argument('--runs-manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--withdraw-human-review-dir',type=Path);p.add_argument('--withdrawal-reason')
    p.add_argument('--reference-policy',choices=['full','source_catalog'],default='full',help='source_catalog只精简来源分类题，指定补丁/事实/引文题保持全文')
    p.add_argument('--protocol-version',choices=['v2','v3'],default='v3')
    p=subs.add_parser('judge');p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--judge-model');p.add_argument('--allow-paid',action='store_true');p.add_argument('--max-http',type=int)
    p.add_argument('--budget-ledger',type=Path);p.add_argument('--max-attempts-per-answer',type=int,choices=[1,2],default=2)
    p.add_argument('--output-tokens',type=int,help='评委单次输出预算512..8192；未指定时保持协议默认，不改变生产配置')
    for name in COMMANDS-{'prepare','timezone-migrate','compare','public-export','judge-export','judge','suite','status'}:
        p=subs.add_parser(name);p.add_argument('--bundle',type=Path,required=True)
        if name in {'retrieval','oracle','agent','fixed-workflow','replay'}:p.add_argument('--output',type=Path,required=True)
        if name in {'retrieval','oracle','agent','fixed-workflow'}:p.add_argument('--split',choices=['all','development','holdout'],default='development' if name in {'agent','fixed-workflow'} else 'all')
        if name in {'oracle','agent','fixed-workflow'}:p.add_argument('--limit',type=int,default=3 if name in {'agent','fixed-workflow'} else None)
        if name in {'agent','fixed-workflow'}:p.add_argument('--allow-paid',action='store_true');p.add_argument('--max-http',type=int)
        if name=='retrieval':
            p.add_argument('--modes',nargs='+',default=['bm25','dense','hybrid','hybrid_reranker','hybrid_reranker_screening'])
            p.add_argument('--exploratory',action='store_true')
            p.add_argument('--allow-paid',action='store_true');p.add_argument('--max-http',type=int)
        if name=='review-import':p.add_argument('--reviews',type=Path,required=True)
        if name=='replay':
            p.add_argument('--runs-manifest',type=Path,required=True);p.add_argument('--review-dir',type=Path)
            p.add_argument('--ai-results',type=Path);p.add_argument('--withdrawals',type=Path,help='当前有效性需传入撤回记录；未提供时只重放历史评分')
    p=subs.add_parser('compare');p.add_argument('--base',type=Path,required=True);p.add_argument('--new',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=subs.add_parser('public-export');p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if hasattr(args,'limit') and args.limit is not None and args.limit<1:parser.error('--limit必须为正数')
    try:
        if args.command in {'suite','status'}:
            from gamelens.evaluation.benchmark_suite import run_suite,benchmark_status
            value=benchmark_status(args.bundle) if args.command=='status' else run_suite(args.bundle,args.output,PROJECT_ROOT,allow_paid=args.allow_paid,agent_max_http=args.agent_max_http,judge_max_http=args.judge_max_http,**({'resume_from':args.resume_from} if args.resume_from else {}),**({'judge_output_tokens':args.judge_output_tokens} if args.judge_output_tokens is not None else {}))
        elif args.command=='judge-export':
            from gamelens.evaluation.benchmark_judge import export_judge
            value=export_judge(args.bundle,args.runs_manifest,args.output,withdraw_human_review_dir=args.withdraw_human_review_dir,withdrawal_reason=args.withdrawal_reason,reference_policy=args.reference_policy,protocol_version=args.protocol_version)
        elif args.command=='judge':
            from gamelens.evaluation.benchmark_judge import run_judge
            value=run_judge(args.inputs,args.output,allow_paid=args.allow_paid,max_http=args.max_http,judge_model=args.judge_model,
                            budget_ledger=args.budget_ledger,max_attempts_per_answer=args.max_attempts_per_answer,
                            **({'output_tokens_override':args.output_tokens} if args.output_tokens is not None else {}))
        elif args.command=='prepare':value={'bundle':str(prepare(args.project,args.output))}
        elif args.command=='timezone-migrate':value={'bundle':str(migrate_timezone(args.bundle,args.output))}
        elif args.command=='validate':value=validate(args.bundle)
        elif args.command=='review-export':export_review_pack(args.bundle);value={'review_pack':str(args.bundle/'review_pack')}
        elif args.command=='review-import':value=import_reviews(args.bundle,args.reviews)
        elif args.command=='oracle':value=run_oracle(args.bundle,args.output,split=args.split,limit=args.limit)
        elif args.command=='retrieval':value=run_retrieval(args.bundle,args.output,PROJECT_ROOT,modes=args.modes,split=args.split,exploratory=args.exploratory,allow_paid=args.allow_paid,max_http=args.max_http)
        elif args.command in {'agent','fixed-workflow'}:value=run_agent(args.bundle,args.output,PROJECT_ROOT,mode=args.command.replace('-','_'),split=args.split,limit=args.limit,max_http=args.max_http,allow_paid=args.allow_paid)
        elif args.command=='replay':value=replay(args.bundle,args.runs_manifest,args.output,args.review_dir,ai_results=args.ai_results,withdrawals_path=args.withdrawals)
        elif args.command=='compare':
            value=compare(read(args.base),read(args.new))
            if args.output.exists():raise ValueError('禁止覆盖对比报告')
            atomic_write_json(args.output/'comparison.json',value)
            atomic_write_text(args.output/'report.md','# GameLens Benchmark 对比\n\n'+json.dumps(value,ensure_ascii=False,indent=2))
        else:
            import shutil
            validate(args.bundle)
            if args.output.exists():raise ValueError('公开导出目录已存在')
            args.output.mkdir(parents=True)
            for name in ('corpus','review_pack'):
                if (args.bundle/name).exists():shutil.copytree(args.bundle/name,args.output/name)
            for name in ('manifest.json','retrieval.json','investigations.json','fixtures.json'):
                shutil.copyfile(args.bundle/name,args.output/name)
            value={'public_bundle':str(args.output),'excluded':'private/source-map.json以及全部运行私有数据'}
        print(json.dumps(value.get('summary',value),ensure_ascii=False,indent=2,default=str))
        return 0
    except (ValueError,FileNotFoundError,KeyError) as error:
        print('Benchmark blocked: '+str(error));return 2


if __name__=='__main__':raise SystemExit(main())
