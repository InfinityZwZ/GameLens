"""Thin, immutable benchmark orchestration; no business Agent implementation."""
from pathlib import Path
import time
from datetime import datetime,timezone,timedelta
from gamelens.evaluation.benchmark_data import load,validate,read,file_hash
from gamelens.evaluation.benchmark_runner import run_oracle,run_retrieval,run_agent,replay,output_directory,agent_cases
from gamelens.evaluation.benchmark_judge import export_judge,run_judge
from gamelens.storage.atomic_io import atomic_write_json,atomic_write_text

def benchmark_status(bundle_root):
    b=load(bundle_root);status=validate(bundle_root)
    active=[c for c in b['investigations'] if c['review']['status']!='rejected']
    status.update(bundle_root=str(Path(bundle_root).resolve()),oracle_planned=sum(bool(c.get('oracle_plan')) for c in active),
        semantic_case_ids=[c['id'] for c in agent_cases(b,split='all',limit=None)],
        scripted_only_cases=[c['id'] for c in active if c.get('response_scenario')=='generation_interruption' and c.get('recovery_fixture')],
        oracle_without_plan=[c['id'] for c in active if not c.get('oracle_plan')],
        split_counts={group:{split:sum(c['review']['status']=='reviewed' and c['split']==split for c in b[group]) for split in ('development','holdout')} for group in ('retrieval','investigations')})
    return status

def suite_report(output,payload):
    lines=['# GameLens Benchmark 完整批次','', '## 本批测量范围','',
        f"有效题：检索{payload['inventory']['formal_retrieval_count']}道、调查{payload['inventory']['formal_investigation_count']}道。冻结资料哈希：{payload['inventory']['dataset_sha256']}。",
        f"工具Oracle计划{payload['inventory']['oracle_planned']}道；缺计划题不计通过或业务失败。",
        f"真实LLM语义任务{len(payload['inventory']['semantic_case_ids'])}道；独立脚本恢复夹具{len(payload['inventory']['scripted_only_cases'])}道，不混入模型准确率。",
        '', '| 阶段 | 状态 | 结果入口／原因 |','|---|---|---|']
    for name,stage in payload['stages'].items():lines.append(f"| {name} | {stage['status']} | {stage.get('path',stage.get('reason',''))} |")
    if payload.get('continuation'):
        lines+=['','本目录为续跑：已完成阶段沿用原记录；原批次已保存答案及AI评分单独保留，不重新请求。',
            '原结果入口：'+payload['continuation']['source_suite'],
            '本目录Agent/AI统计只包含剩余题，不将局部完成率当作全24题完成率。']
    for name,entry in payload['stages'].items():
        for condition,reason in entry.get('unavailable_conditions',{}).items():lines.append(f"- {name}/{condition} 未测：{reason}")
    lines+=['', '## 关键结果','']
    oracle=payload['stages'].get('oracle',{}).get('summary')
    if oracle:lines.append(f"- Oracle：执行{oracle.get('executed')}道，通过{oracle.get('tool_oracle_passed')}道；这是工具执行，不是答案准确率。")
    retrieval=payload['stages'].get('retrieval',{}).get('summary',{})
    if retrieval:
        lines+=['', '| 检索方式 | 正式题数 | Recall@10 | nDCG@10 | P95毫秒 |','|---|---|---|---|---|']
        for mode,row in retrieval.items():lines.append(f"| {mode} | {row.get('formal_cases')} | {row.get('recall_at_10')} | {row.get('ndcg_at_10')} | {row.get('p95_ms')} |")
    split_metrics=payload['stages'].get('retrieval',{}).get('metrics_by_split',{})
    for split,modes in split_metrics.items():
        lines+=['', '### '+split,'', '| 方式 | 题数 | Recall@10 | nDCG@10 |','|---|---|---|---|']
        for mode,row in modes.items():lines.append(f"| {mode} | {row['cases']} | {row['recall_at_10']} | {row['ndcg_at_10']} |")
    scored=payload['stages'].get('judge',{}).get('summary')
    if scored:lines+=['', '- AI评分完成：'+str(scored.get('ai_scored_cases'))+'/'+str(scored.get('ai_requested_cases')),
        '- AI语义通过率：'+str(scored.get('ai_semantic_pass_rate'))+'；AI辅助综合通过率：'+str(scored.get('ai_assisted_task_success_rate'))+'。不是人工确认的答案准确率。']
    lines+=['', '## 未测量与解释边界','',
        '- 固定工作流付费对比、检索层付费Screening本批不测；没有用降级结果冒充相应成绩。',
        '- 5道撤下题不恢复；6道缺Oracle计划仍可进入已审核的真实Agent测评，但不能编造Oracle通过记录。',
        '- 失败、缺模型、输入超限、预算用尽均保留；未知不是0，未跑完不称完整成功。',
        '- 已有留出题属于内部留出配置，不声称未见过的独立盲测。不同资料/提示词/校验版本不直接宣称能力提升。',
        '- 题目和标准答案没有被本批修改；模型可能偏置，原文逐字校验不等于语义真值。',
        '', '逐题分数、引用、用量、耗时和失败原因见各阶段summary.json、failures.json及私有运行记录；请勿将private目录公开提交。']
    atomic_write_text(output/'REPORT.md','\n'.join(lines))

def run_suite(bundle_root,output,project,*,allow_paid=False,agent_max_http=None,judge_max_http=None,resume_from=None,judge_output_tokens=None):
    if judge_output_tokens is not None and (type(judge_output_tokens) is not int or not 512<=judge_output_tokens<=8192):
        raise ValueError('评委输出预算须为512..8192整数')
    if allow_paid and (type(agent_max_http) is not int or agent_max_http<1 or type(judge_max_http) is not int or judge_max_http<1):raise ValueError('完整付费批次须明确Agent和评委各自HTTP预算')
    inventory=benchmark_status(bundle_root);prior=None;source=None;origin=None
    if resume_from:
        if not allow_paid:raise ValueError('续跑付费批次须显式授权，不能在离线模式继续')
        source=Path(resume_from).resolve();prior=read(source/'suite.json')
        origin=source;ancestor=prior;seen=set()
        while ancestor.get('continuation'):
            if origin in seen or len(seen)>=32:raise ValueError('续跑批次循环或层数超限')
            seen.add(origin);link=ancestor['continuation'];origin=Path(link['source_suite']).resolve()
            if file_hash(origin/'suite.json')!=link['source_suite_sha256']:raise ValueError('原批次清单哈希不匹配')
            ancestor=read(origin/'suite.json')
        if prior['inventory']['dataset_sha256']!=inventory['dataset_sha256']:raise ValueError('续跑资料包不匹配')
        if prior['budgets']!={'agent':agent_max_http,'judge':judge_max_http}:raise ValueError('续跑预算与原批次不符，禁止扩容')
        if prior.get('judge_output_tokens')!=judge_output_tokens:raise ValueError('续跑评委输出预算与原批次不符；请另建独立批次')
        if prior['stages'].get('retrieval',{}).get('status')!='completed' or prior['stages'].get('oracle',{}).get('status') not in {'partial','completed'}:
            raise ValueError('原离线阶段未完成，不支持跳过')
        if not (source/'agent/runs_manifest.json').is_file():raise ValueError('缺少原Agent运行清单')
    output=output_directory(output)
    payload={'version':'gamelens-suite-v1','inventory':inventory,'paid_authorized':allow_paid,
        'budgets':{'agent':agent_max_http if allow_paid else 0,'judge':judge_max_http if allow_paid else 0},'stages':{}}
    if judge_output_tokens is not None:payload['judge_output_tokens']=judge_output_tokens
    if prior:
        payload['continuation']={'source_suite':str(source),'source_suite_sha256':file_hash(source/'suite.json'),
            'previous_judge':prior['stages'].get('judge',{})}
    def save():
        atomic_write_json(output/'suite.json',payload);suite_report(output,payload)
    def stage(name,call):
        started=time.monotonic();payload['stages'][name]={'status':'running'};save()
        try:
            result=call();entry={'status':'completed','path':str(output/name),'summary':result.get('summary',{}),'elapsed_seconds':time.monotonic()-started,
                                'unavailable_conditions':result.get('unavailable_conditions',{})}
            if result.get('unavailable_conditions') or any(r.get('blocked') for r in result.get('results',[])):entry['status']='partial'
            if name=='judge' and result.get('summary',{}).get('ai_scored_cases')!=result.get('summary',{}).get('ai_requested_cases'):entry['status']='partial'
        except Exception as error:
            from gamelens.core.decision_trace import safe_audit
            result=None;entry={'status':'failed','error_type':type(error).__name__,'reason':safe_audit(str(error)),'elapsed_seconds':time.monotonic()-started}
        payload['stages'][name]=entry;save();print(f"Benchmark suite {name}: {entry['status']}",flush=True);return result
    if prior:
        import copy
        for name in ('oracle','retrieval'):
            payload['stages'][name]=copy.deepcopy(prior['stages'][name]);payload['stages'][name]['reused_from']=str(source)
        save();retrieved=None
    else:
        stage('oracle',lambda:run_oracle(bundle_root,output/'oracle',split='all'))
        retrieved=stage('retrieval',lambda:run_retrieval(bundle_root,output/'retrieval',project,modes=('bm25','dense','hybrid','hybrid_reranker'),split='all'))
    if retrieved:
        bundle=load(bundle_root);splits={c['id']:c['split'] for c in bundle['retrieval']};groups={}
        for split in ('development','holdout','all'):
            groups[split]={}
            for mode in ('bm25','dense','hybrid','hybrid_reranker'):
                rows=[r for r in retrieved.get('results',[]) if r.get('condition')==mode and r.get('metrics') is not None and (split=='all' or splits.get(r['case_id'])==split)]
                metrics={}
                for key in ('recall_at_5','recall_at_10','mrr_at_10','ndcg_at_10','support_coverage_at_10','counterexample_coverage_at_10','no_answer_candidate_false_positive'):
                    values=[r['metrics'][key] for r in rows if r['metrics'].get(key) is not None];metrics[key]=sum(values)/len(values) if values else None
                groups[split][mode]={'cases':len(rows),**metrics}
        payload['stages']['retrieval']['metrics_by_split']=groups
        payload['stages']['retrieval']['ranking_inconsistent']=[{'case_id':r['case_id'],'condition':r.get('condition')} for r in retrieved.get('results',[]) if r.get('ranking_consistent') is False]
        atomic_write_json(output/'retrieval/metrics-by-split.json',groups);save()
    if not allow_paid:
        for name in ('agent','judge_export','judge','replay'):payload['stages'][name]={'status':'not_measured','reason':'paid_not_authorized'}
        save();return payload
    if not inventory['formal_ready']:
        for name in ('agent','judge_export','judge','replay'):payload['stages'][name]={'status':'blocked','reason':'dataset_not_formally_ready'}
        save();return payload
    agent_options={'resume_from':source/'agent/runs_manifest.json','budget_ledger':origin/'agent/private/http-budget.sqlite'} if prior else {}
    agent=stage('agent',lambda:run_agent(bundle_root,output/'agent',project,split='all',limit=None,max_http=agent_max_http,allow_paid=True,active_budget_seconds=7200,**agent_options))
    runs=output/'agent/runs_manifest.json'
    if runs.is_file() and read(runs).get('runs'):
        exported=stage('judge_export',lambda:export_judge(bundle_root,runs,output/'judge_export',reference_policy='source_catalog',protocol_version='v3'))
        if exported is not None:
            judge_options={'budget_ledger':origin/'judge/private/http-budget.sqlite'} if prior else {}
            judged=stage('judge',lambda:run_judge(output/'judge_export',output/'judge',allow_paid=True,max_http=judge_max_http,max_attempts_per_answer=1,
                **judge_options,**({'output_tokens_override':judge_output_tokens} if judge_output_tokens is not None else {})))
            if judged is not None:stage('replay',lambda:replay(bundle_root,runs,output/'replay',ai_results=output/'judge'))
    for name in ('judge_export','judge','replay'):
        if name not in payload['stages']:payload['stages'][name]={'status':'blocked','reason':'upstream_stage_unavailable'}
    save();return payload
