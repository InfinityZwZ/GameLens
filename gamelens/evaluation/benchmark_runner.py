"""Isolated frozen-data runners, explicit paid opt-in, and immutable reports."""
from __future__ import annotations

import json
import os
import platform
import shutil
import tempfile
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from gamelens.evaluation.benchmark_data import load, validate, case_hash, digest, read, file_hash
from gamelens.evaluation.benchmark_score import ranking_score, score_run, summary, review_template
from gamelens.evaluation.retrieval import percentile
from gamelens.storage.atomic_io import atomic_write_json, atomic_write_text


def identity(bundle,cases,configuration):
    package=Path(__file__).resolve().parents[1]
    fingerprints={p.relative_to(package).as_posix():file_hash(p) for p in package.rglob('*.py')}
    from gamelens.rag.embedding import CONFIG
    from gamelens.rag.reranker import MODEL_ID, MODEL_REVISION
    return {'benchmark_version':bundle['manifest']['version'],
        'manifest_sha256':file_hash(bundle['root']/'manifest.json'),
        'dataset_sha256':bundle['manifest']['dataset_sha256'],
        'case_hashes':{c['id']:case_hash(c) for c in cases},'configuration':configuration,
        'environment':{'python':platform.python_version(),'platform':platform.platform(),
            'processor':platform.processor(),'clock':bundle['manifest']['clock'],
            'timezone':bundle['manifest']['timezone_iana'],'time_standard':bundle['manifest']['timezone']},
        'code_sha256':digest(fingerprints),'embedding':CONFIG,
        'reranker':{'model':MODEL_ID,'revision':MODEL_REVISION}}


def output_directory(path):
    path=Path(path)
    if path.exists():raise ValueError('结果目录已存在；禁止覆盖历史测量')
    path.mkdir(parents=True)
    return path


def materialize(bundle,root,case=None):
    for appid,game in bundle['corpus'].items():
        raw=root/'data/raw'
        for kind in ('reviews','news'):
            source=bundle['root']/'corpus'/f'{appid}_{kind}.json'
            if source.exists():
                (raw).mkdir(parents=True,exist_ok=True);shutil.copyfile(source,raw/source.name)
        from gamelens.tools.steam_news import retained_patch_candidate
        atomic_write_json(raw/f'{appid}_patch_candidates.json',[n for n in game['news'] if retained_patch_candidate(n)])
        for p in (bundle['root']/'corpus/announcement_comments'/str(appid)).glob('*.json'):
            target=raw/'announcement_comments'/str(appid)/p.name
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    if case and case.get('initial_conditions',{}).get('news_cache_absent'):
        # Only materialized copies are removed, never the frozen evidence.
        for suffix in ('news','patch_candidates'):
            (root/'data/raw'/f"{case['appid']}_{suffix}.json").unlink(missing_ok=True)
    if case and case.get('initial_conditions',{}).get('reviews_cache_absent'):
        (root/'data/raw'/f"{case['appid']}_reviews.json").unlink(missing_ok=True)
        (root/'data/raw'/f"{case['appid']}_reviews.coverage.json").unlink(missing_ok=True)


def oracle_step_passed(call,observation):
    expected=call.get('expected_result',{'ok':True})
    if not isinstance(expected,dict) or type(expected.get('ok')) is not bool:return False
    if observation.get('ok') is not expected['ok']:return False
    if not expected['ok']:
        code=expected.get('error_code')
        actual=(observation.get('failure') or {}).get('error_code') or observation.get('data',{}).get('error_code')
        return bool(code) and actual==code
    return True


@contextmanager
def frozen_environment(bundle,scenario='normal'):
    """Applied in each tool worker as well as in the planner process."""
    game_by_id=bundle['corpus'];moment=datetime.fromisoformat(bundle['manifest']['clock'])
    from gamelens.core.local_time import timezone_signature
    saved_signature={k:v for k,v in bundle['manifest']['timezone'].items() if k in
        {'basis','system_zone','names','standard_offset_seconds','daylight_offset_seconds'}}
    if timezone_signature()!=saved_signature:
        raise ValueError('项目时间标准与冻结快照不同；请用timezone-migrate建立北京时间新包，禁止静默更改旧包')
    class FrozenDateTime(datetime):
        @classmethod
        def now(cls,tz=None):
            from gamelens.core.local_time import local_datetime
            return moment.astimezone(tz) if tz else local_datetime(moment).replace(tzinfo=None)
    def blocked_get(*args,**kwargs):raise RuntimeError('Benchmark禁止真实Steam GET；只能使用冻结响应')
    import httpx
    original_send=httpx.Client.send
    allowed_endpoint=os.getenv('LLM_BASE_URL','https://api.openai.com/v1').rstrip('/')+'/chat/completions'
    def frozen_send(client,request,*args,**kwargs):
        if request.method!='POST' or str(request.url)!=allowed_endpoint:
            raise RuntimeError('Benchmark禁止非LLM网络请求；只能使用冻结响应')
        return original_send(client,request,*args,**kwargs)
    def discussion(news,target_count=500,**kwargs):
        import copy
        from gamelens.tools.steam_announcement_comments import DiscussionError
        if scenario=='timeout':raise DiscussionError('request_timeout','frozen discussion timeout')
        path=bundle['root']/'corpus/announcement_comments'/str(news.appid)/(str(news.news_id)+'.json')
        if not path.is_file():raise DiscussionError('discussion_unavailable','冻结资料未包含该公告讨论区，不进行在线采集')
        snapshot=copy.deepcopy(read(path));snapshot['comments']=snapshot.get('comments',[])[:target_count]
        original_count=len(read(path).get('comments',[]))
        if len(snapshot['comments'])<original_count:
            snapshot['coverage']={**snapshot.get('coverage',{}),'complete':False,'source_exhausted':False,'collected_count':len(snapshot['comments'])}
        return snapshot
    def news(appid,count=100,**kwargs):
        import httpx
        if scenario=='timeout':raise httpx.ReadTimeout('frozen_timeout_fixture')
        from gamelens.tools.steam_news import is_official_announcement
        rows=sorted(game_by_id.get(appid,{}).get('news',[]),key=lambda n:n['published_at_timestamp'],reverse=True)
        rows=[n for n in rows if n['published_at_timestamp']<=moment.timestamp() and (not kwargs.get('feeds') or is_official_announcement(n)) and
            (not kwargs.get('enddate') or n['published_at_timestamp']<kwargs['enddate'])]
        return [{'gid':n['news_id'],'title':n['title'],'contents':n['contents'],'date':n['published_at_timestamp'],
            'url':n['url'],'feedname':n['feed_name'],'feedlabel':n['feed_label'],'author':''} for n in rows[:count]]
    def acquire(appid,start,end,language='all',target_count=1000,**kwargs):
        rows=[r for r in game_by_id.get(appid,{}).get('reviews',[]) if start<=r['created_at_timestamp']<end and (language=='all' or r['language']==language)]
        rows=sorted(rows,key=lambda r:r['created_at_timestamp'],reverse=True)
        count=len(rows);positive=sum(r['is_positive'] for r in rows)
        error='request_timeout' if scenario=='timeout' else None
        sample=[] if error else rows[:target_count]
        return {'comments':sample,'reported_total':count if not error else None,
            'stats':{'count':count,'positive_count':positive,'negative_count':count-positive,
                'positive_rate':positive/count if count else None,'negative_rate':(count-positive)/count if count else None} if not error else None,
            'stats_verified':not error,'body_count':len(sample),'pages_read':1,
            'complete_bodies':not error and len(sample)==count,'playtime_metadata_complete':{'at_review':True,'lifetime':True},
            'error_code':error,'stop_reason':error or 'endpoint_exhausted' if len(sample)==count else 'sample_budget',
            'sampling':'frozen_fixture_only','historical_text_available':False}
    def reviews(appid,target_count=1000,**kwargs):return game_by_id.get(appid,{}).get('reviews',[])[:target_count]
    with ExitStack() as stack:
        stack.enter_context(patch('httpx.Client.get',blocked_get))
        stack.enter_context(patch('httpx.Client.send',frozen_send))
        import importlib
        for module in ('gamelens.core.local_time','gamelens.tools.registry','gamelens.tools.investigation','gamelens.core.synthesis'):
            target=importlib.import_module(module)
            if hasattr(target,'time_context'):stack.enter_context(patch.object(target,'time_context',lambda:bundle['manifest']['timezone']))
        for module in ('gamelens.tools.registry','gamelens.tools.review_coverage','gamelens.core.query_contract'):
            stack.enter_context(patch(module+'.datetime',FrozenDateTime))
        stack.enter_context(patch('gamelens.tools.steam_news.fetch_news',news))
        stack.enter_context(patch('gamelens.tools.steam_review_windows.acquire_window',acquire))
        stack.enter_context(patch('gamelens.tools.steam_announcement_comments.fetch_announcement_comments',discussion))
        # Production tools may explicitly request download=True. Benchmark
        # overrides that request and permits only already cached model weights.
        from gamelens.rag.embedding import LocalEmbedding
        from gamelens.rag.reranker import BGEReranker
        embedding_load=LocalEmbedding.load;reranker_load=BGEReranker.load
        stack.enter_context(patch.object(LocalEmbedding,'load',lambda self,download=False:embedding_load(self,download=False)))
        stack.enter_context(patch.object(BGEReranker,'load',lambda self,download=False:reranker_load(self,download=False)))
        yield


@dataclass
class FrozenHandler:
    handler:object
    bundle_root:str
    scenario:str

    @property
    def __self__(self):
        # Runtime asks the registered coverage owner for a pure request plan.
        # Keep that bound-method contract without bypassing frozen execution.
        return self.handler.__self__

    def __call__(self,args,state):
        bundle=load(self.bundle_root)
        with frozen_environment(bundle,self.scenario):return self.handler(args,state)


def registry_for(bundle,root,case,client=None,model_root=None):
    from gamelens.tools.registry import SteamInvestigationTools
    from gamelens.core.runtime import ToolRegistry
    registry=SteamInvestigationTools(root,client).registry()
    if model_root is not None:
        from gamelens.rag.embedding import LocalEmbedding
        from gamelens.rag.reranker import BGEReranker
        rag=next(t.handler.__self__ for t in registry.tools if t.name=='search_game_evidence')
        rag.index.encoder=LocalEmbedding(Path(model_root));rag.reranker=BGEReranker(Path(model_root))
    # Embedding is always shared from an already prepared model cache. Loading
    # remains local-only even when the production build tool requests download.
    return ToolRegistry([replace(t,handler=FrozenHandler(t.handler,str(bundle['root'].resolve()),case['response_scenario']),
        approval_policy=None if t.name in {'prepare_patches','ensure_review_coverage'} else t.approval_policy)
        for t in registry.tools if t.name not in {'prepare_reviews','prepare_announcement_comments'}],
        receipt_root=root/'data/tool_receipts')


def state_for(bundle,case):
    from gamelens.evaluation.benchmark import initial_state
    s=initial_state(case['id'],case['goal'])
    s.update(appid=case['appid'],thread_id=case['id'],investigation_profile='focused',patch_news_id=None,
        rag_mode='auto',optimization_mode='auto',request_started_at=bundle['manifest']['clock'],
        allow_review_acquisition=True,allow_patch_acquisition=True,
        conversation_memory={'recent_turns':case.get('history',[]),'trust_note':'历史是未验证的上下文。'},
        evaluation={'case_id':case['id'],'case_sha256':case_hash(case),'dataset_sha256':bundle['manifest']['dataset_sha256']})
    if case.get('initial_conditions',{}).get('coverage_followup'):
        import copy
        s['coverage_followup']=copy.deepcopy(case['initial_conditions']['coverage_followup'])
    return s


def run_generation_recovery_fixture(registry,state,root,*,quote=None,answer=None):
    """Actual graph/SQLite recovery, scripted model failure, zero model requests."""
    import copy,sqlite3,httpx
    from langgraph.checkpoint.sqlite import SqliteSaver
    from gamelens.core.runtime import InvestigationAgent,PlannerDecision
    from gamelens.core.grounding import validate_grounding
    source=next(o for o in state['observations'] if o.get('ok') and o['tool']=='get_review')
    record=source['data']['review']
    quote=quote or record['text'].split('\n\n')[-1]
    if not quote.strip() or quote not in record['text'] or len(quote)>500:
        raise ValueError('中断夹具须提供不超过500字的独立原文片段，不能拼接或补造')
    original=copy.deepcopy(state['observations'])
    state=copy.deepcopy(state);state['step_count']=max(o['step'] for o in original)
    state['status']='running'
    class InterruptedPlanner:
        def decide(self,current,tools):raise httpx.ReadTimeout('benchmark_injected_generation_interruption')
    class RecoveryPlanner:
        def decide(self,current,tools):
            return PlannerDecision(action='finish',reason='脚本化离线恢复夹具，无真实模型调用',
                final_answer=answer or '已从检查点恢复并读取指定评论原文。仅能确认该条玩家反馈，不能据此外推所有玩家或认定统一故障原因。',
                claims=[{'source_step':source['step'],'label':'已读取的评论ID','fact_path':['review','review_id'],'value':record['review_id']}],
                citations=[{'source_step':source['step'],'review_id':record['review_id'],'quote':quote}],scope_limited=True)
    connection=sqlite3.connect(Path(root)/'generation-recovery.sqlite',check_same_thread=False)
    try:
        first=InvestigationAgent(InterruptedPlanner(),registry,checkpointer=SqliteSaver(connection))
        interrupted=first.invoke(state)
        atomic_write_json(Path(root)/'generation-interrupted.json',interrupted)
        recovery=InvestigationAgent(RecoveryPlanner(),registry,checkpointer=SqliteSaver(connection))
        recovered=interrupted
        for snapshot in recovery.recover_stream(state['thread_id']):recovered=dict(snapshot)
        atomic_write_json(Path(root)/'generation-recovered.json',recovered)
    finally:connection.close()
    checks={'interrupted_not_completed':interrupted.get('status')=='interrupted' and bool(interrupted.get('generation_pending')),
        'checkpoint_recovered':recovered.get('status')=='completed' and recovered.get('generation_recoveries')==1,
        'observations_preserved':recovered.get('observations',[])[:len(original)]==original,
        'no_repeated_source_read':sum(o['tool']=='get_review' for o in recovered.get('observations',[]))==1,
        'citation_after_recovery':bool(recovered.get('final_citations')) and not validate_grounding(
            recovered.get('observations',[]),recovered.get('final_claims',[]),recovered.get('final_citations',[]))}
    recovered['evaluation_fixture']={'type':'scripted_generation_recovery','model_http_requests':0,'checks':checks}
    return recovered,checks


def report(output,payload):
    atomic_write_json(output/'summary.json',payload)
    atomic_write_json(output/'failures.json',[r for r in payload.get('results',[]) if r.get('blocked') or r.get('failed_checks') or r.get('tool_oracle_passed') is False])
    lines=['# GameLens Benchmark v1','',f"模式：{payload['mode']}",f"资料哈希：{payload['identity']['dataset_sha256']}",'',
        '## 测量范围','',payload.get('note',''),'',
        '未审核题、缺失模型与缺资料条件不产生正式Agent答案准确率。工具Oracle只核对工具执行。','',
        '## 结果','', '```json',json.dumps(payload.get('summary',{}),ensure_ascii=False,indent=2),'```','',
        '## 未测量与阻塞','']
    for r in payload.get('results',[]):
        if r.get('blocked'):lines.append('- '+r['case_id']+'：'+r['blocked'])
        elif r.get('tool_oracle_passed') is False:
            lines.append('- '+r['case_id']+'：工具Oracle执行失败；'+', '.join(r.get('failed_checks',[])))
    for mode,reason in payload.get('unavailable_conditions',{}).items():lines.append('- '+mode+'：'+reason)
    atomic_write_text(output/'report.md','\n'.join(lines))


def run_oracle(bundle_root,output,*,split='all',limit=None):
    b=load(bundle_root);validate(bundle_root)
    cases=[c for c in b['investigations'] if c['review']['status']!='rejected' and (split=='all' or c['split']==split)][:limit]
    output=output_directory(output);rows=[]
    with tempfile.TemporaryDirectory(prefix='gl-oracle-') as directory:
        for c in cases:
            if c['blocking'] or not c['oracle_plan']:
                rows.append({'case_id':c['id'],'blocked':'missing_source' if c['blocking'] else 'oracle_plan_pending_human_annotation'});continue
            root=Path(directory)/c['id'];materialize(b,root,c)
            news_path=root/'data/raw'/f"{c['appid']}_news.json"
            news_before=file_hash(news_path) if news_path.is_file() else None
            review_path=root/'data/raw'/f"{c['appid']}_reviews.json"
            reviews_before=review_path.is_file()
            state=state_for(b,c);registry=registry_for(b,root,c);before=time.perf_counter()
            for step,call in enumerate(c['oracle_plan'],1):
                state['step_count']=step-1
                result=registry.execute(call['tool'],call['arguments'],state)
                state['observations'].append({'step':step,'tool':call['tool'],'arguments':call['arguments'],**result.model_dump(mode='json')})
                state['artifacts'].update(result.artifacts)
            state.update(status='oracle_executed',final_answer='')
            atomic_write_json(output/'runs'/(c['id']+'.json'),state)
            facts=[]
            from gamelens.core.grounding import resolve_path,same_value
            for fact in c['expectations']['facts']:
                matches=[]
                for o in state['observations']:
                    try:matches.append(o['ok'] and same_value(resolve_path(o['data'],fact['path']),fact['value']))
                    except (KeyError,TypeError,IndexError):pass
                facts.append(any(matches))
            fixture_checks={}
            if c.get('initial_conditions',{}).get('reviews_cache_absent'):
                fixture_checks['reviews_cache_absent_at_start']=not reviews_before
                snapshot=Path(state.get('artifacts',{}).get('reviews','')).resolve()
                fixture_checks['reviews_cache_restored']=(snapshot.is_file() and snapshot.is_relative_to(root.resolve()) and
                    any(o.get('ok') and o['tool']=='query_reviews' and
                        Path(o.get('data',{}).get('source',{}).get('path','')).resolve()==snapshot and
                        o['data']['source'].get('sha256')==file_hash(snapshot) for o in state['observations']))
            if c.get('response_scenario')=='generation_interruption':
                state,recovery_checks=run_generation_recovery_fixture(registry,state,root,**c.get('recovery_fixture',{}))
                fixture_checks.update(recovery_checks)
                atomic_write_json(output/'runs'/(c['id']+'.json'),state)
            if c.get('initial_conditions',{}).get('news_cache_absent'):
                fixture_checks['news_cache_absent_at_start']=news_before is None
                fixture_checks['news_cache_restored']=news_path.is_file()
            if c.get('oracle_assertions',{}).get('news_cache_preserved'):
                fixture_checks['news_cache_preserved']=bool(news_before) and news_path.is_file() and file_hash(news_path)==news_before
            step_checks=[{'tool':call['tool'],'passed':oracle_step_passed(call,o),
                'expected':call.get('expected_result',{'ok':True}),'actual_ok':o['ok'],
                'summary':o.get('summary','')} for call,o in zip(c['oracle_plan'],state['observations'])]
            failures=['tool:'+s['tool'] for s in step_checks if not s['passed']]
            if not all(facts):failures.append('gold_facts')
            failures += ['fixture:'+k for k,v in fixture_checks.items() if not v]
            rows.append({'case_id':c['id'],'mode':'oracle','formal_eligible':c['review']['status']=='reviewed',
                'fixture_checks':fixture_checks,
                'step_checks':step_checks,'failed_checks':failures,
                'tool_oracle_passed':not failures,
                'elapsed_seconds':time.perf_counter()-before,'accepted':None,'automatic_passed':None})
    payload={'mode':'oracle','identity':identity(b,cases,{'split':split,'seed':0}),
        'results':rows,'summary':{'executed':sum('tool_oracle_passed' in r for r in rows),
            'tool_oracle_passed':sum(r.get('tool_oracle_passed',False) for r in rows),'task_success_rate':None},
        'note':'固定人工候选工具参数的探索性执行；没有调用LLM，不测答案语义。'}
    report(output,payload);return payload


def run_retrieval(bundle_root,output,project,*,modes=('bm25','dense','hybrid','hybrid_reranker','hybrid_reranker_screening'),split='all',exploratory=False,client=None,allow_paid=False,max_http=None):
    from gamelens.rag.index import GameIndex
    from gamelens.rag.embedding import LocalEmbedding
    from gamelens.rag.tools import SearchArgs
    from gamelens.rag.reranker import BGEReranker,apply_reranker,RerankerError
    from gamelens.rag.screening import screen_candidates
    b=load(bundle_root);validate(bundle_root)
    cases=[c for c in b['retrieval'] if c['review']['status']!='rejected' and (split=='all' or c['split']==split) and not c['blocking'] and
        (exploratory or c['review']['status']=='reviewed')]
    if allow_paid and (not max_http or max_http<1):raise ValueError('Screening必须明确HTTP预算')
    output=output_directory(output);rows=[];unavailable={};model_times={};telemetry=None
    if allow_paid and cases and 'hybrid_reranker_screening' in modes and client is None:
        from gamelens.core.llm import OpenAICompatibleClient
        from gamelens.core.budget import TaskBudgetStore
        from gamelens.storage.telemetry import TelemetryStore
        budget=TaskBudgetStore(output/'private/http-budget.sqlite',request_limit=max_http)
        telemetry=TelemetryStore(output/'private/telemetry.sqlite')
        client=OpenAICompatibleClient(os.getenv('LLM_MODEL',''),request_budget=budget,budget_thread_id='screening-batch',
            telemetry_sink=lambda event:telemetry.record('screening-batch',event))
    with tempfile.TemporaryDirectory(prefix='gl-retrieval-') as directory:
        root=Path(directory);materialize(b,root)
        encoder=LocalEmbedding(project);reranker=BGEReranker(project)
        loaded=False;indexes={}
        for mode in modes:
            if mode not in {'bm25','dense','hybrid','hybrid_reranker','hybrid_reranker_screening'}:raise ValueError('未知检索条件')
            if mode!='bm25' and not encoder.ready:unavailable[mode]='embedding_model_unavailable';continue
            if 'reranker' in mode and not reranker.ready:unavailable[mode]='reranker_model_unavailable';continue
            if mode.endswith('screening') and client is None:unavailable[mode]='paid_screening_not_authorized';continue
            for c in cases:
                args=SearchArgs(appid=c['appid'],query=c['query'],kind=c['kind'],filters=c['filters'],top_k=10)
                state={'appid':c['appid'],'goal':c['query'],'artifacts':{},'window_days':7}
                key=(c['appid'],mode=='bm25')
                if key not in indexes:
                    index_root=root/('lexical' if mode=='bm25' else 'vector');materialize(b,index_root)
                    indexes[key]=GameIndex(index_root,encoder=object() if mode=='bm25' else encoder)
                    start=time.perf_counter()
                    if mode=='bm25':indexes[key].build_lexical(c['appid'],state)
                    else:
                        if not loaded:encoder.load(download=False);loaded=True
                        indexes[key].build(c['appid'],state)
                    model_times[str(key)]=1000*(time.perf_counter()-start)
                index=indexes[key];measures=[];stages=[];hits=[];screening={};rankings=[]
                try:
                    # Screening is paid and not repeated for timing; retrieval
                    # itself gets one warmup and five measured repetitions.
                    for repeat in range(6):
                        start=time.perf_counter()
                        candidates,_=index.search(args,state,mode='bm25' if mode=='bm25' else 'dense' if mode=='dense' else 'hybrid',finalize=False,candidate_limit=20)
                        recall_ms=1000*(time.perf_counter()-start);ranked=candidates;reorder={}
                        if 'reranker' in mode:ranked,reorder=apply_reranker(c['query'],candidates,mode='required',reranker=reranker)
                        hits=index.select_hits(ranked,args)
                        rankings.append([(h['kind'],str(h['source_id'])) for h in hits])
                        elapsed=1000*(time.perf_counter()-start)
                        if repeat:measures.append(elapsed);stages.append({'recall_ms':recall_ms,'reranker_ms':reorder.get('inference_ms')})
                    if mode.endswith('screening'):
                        start=time.perf_counter();hits,screening=screen_candidates(client,c['query'],c['query'],hits)
                        screening['latency_ms']=1000*(time.perf_counter()-start)
                except (RerankerError,ValueError,RuntimeError) as error:
                    rows.append({'case_id':c['id'],'condition':mode,'blocked':str(error),'formal_eligible':False});continue
                ids=[(h['kind'],str(h['source_id'])) for h in hits]
                # Score only gold sources that satisfy this query's hard scope.
                from gamelens.evaluation.benchmark_score import eligible_records
                allowed_reviews={str(r['review_id']) for r in eligible_records(b['corpus'][c['appid']],c['filters'])}
                from gamelens.core.local_time import local_datetime
                left=local_datetime(c['filters']['start_at']).timestamp() if c['filters'].get('start_at') else None
                right=local_datetime(c['filters']['end_at']).timestamp() if c['filters'].get('end_at') else None
                from gamelens.tools.steam_news import is_official_announcement
                records={('review',rid) for rid in allowed_reviews} | {('announcement',str(n['news_id'])) for n in b['corpus'][c['appid']]['news']
                    if (left is None or n['published_at_timestamp']>=left) and (right is None or n['published_at_timestamp']<right)
                    and (c.get('source_constraint')!='official' or is_official_announcement(n))}
                eligible_case={**c,'labels':[l for l in c['labels'] if (l['kind'],str(l['source_id'])) in records]}
                formal=c['review']['status']=='reviewed'
                metrics=ranking_score(eligible_case,ids) if formal else None
                row={'case_id':c['id'],'condition':mode,'formal_eligible':formal,'metrics':metrics,
                    'hard_scope_violations':[i for i in ids if i not in records],
                    'ranked_ids':ids,'measurements_ms':measures,'stages':stages,'screening':screening,
                    'ranking_repetitions':rankings,'ranking_consistent':all(r==rankings[0] for r in rankings)}
                if screening:
                    truth={(l['kind'],str(l['source_id'])):l['role'] for l in c['labels'] if l['relevance']>0}
                    judged=[v for v in screening['verdicts'] if (v['kind'],str(v['source_id'])) in truth]
                    row['screening_metrics']={'role_accuracy':sum(v['role']==truth[(v['kind'],str(v['source_id']))] for v in judged)/len(judged) if formal and judged else None,
                        'quote_rejection_rate':len(screening.get('rejected_candidates',[]))/screening['candidate_count'] if screening['candidate_count'] else None,
                        'retained_precision':sum(i in truth for i in ids)/len(ids) if formal and ids else None}
                rows.append(row)
    aggregate={}
    for mode in modes:
        valid=[r for r in rows if r.get('condition')==mode and r.get('metrics') is not None]
        keys=('recall_at_5','recall_at_10','mrr_at_10','ndcg_at_10','no_answer_candidate_false_positive','support_coverage_at_10','counterexample_coverage_at_10')
        values={k:sum(r['metrics'][k] for r in valid if r['metrics'].get(k) is not None)/sum(r['metrics'].get(k) is not None for r in valid)
            if any(r['metrics'].get(k) is not None for r in valid) else None for k in keys}
        times=[t for r in rows if r.get('condition')==mode for t in r.get('measurements_ms',[])]
        aggregate[mode]={'formal_cases':len(valid),**values,'p50_ms':percentile(times,.5),'p95_ms':percentile(times,.95)}
    payload={'mode':'retrieval','identity':identity(b,cases,{'split':split,'seed':0,'warmups':1,'repeats':5}),
        'results':rows,'summary':aggregate,'unavailable_conditions':unavailable,'build_and_load_ms':model_times,
        'note':'pending_review仅保存探索性候选与实测耗时，相关性指标为null；无答案仅指冻结语料。'}
    if telemetry:
        from gamelens.storage.telemetry import summarize
        payload['telemetry']=summarize(telemetry.events('screening-batch'))
        payload['screening_model']=client.model
    report(output,payload);return payload


def replay(bundle_root,runs_manifest,output,review_dir=None,*,ai_results=None,withdrawals_path=None):
    b=load(bundle_root);validate(bundle_root);manifest=read(runs_manifest)
    lookup={c['id']:c for c in b['investigations']};rows=[];cases=[];seen=set()
    if manifest['identity']['dataset_sha256']!=b['manifest']['dataset_sha256']:raise ValueError('回放资料指纹不同')
    ai_payload=read(Path(ai_results)/'summary.json') if ai_results else None
    if ai_payload and (ai_payload.get('mode')!='ai_assisted' or ai_payload.get('identity')!=manifest['identity']):
        raise ValueError('AI回放结果与原运行身份不同')
    if ai_results and withdrawals_path is None:withdrawals_path=Path(ai_results)/'review-withdrawals.json'
    withdrawals=read(withdrawals_path) if withdrawals_path else []
    if not isinstance(withdrawals,list) or len({w['case_id'] for w in withdrawals})!=len(withdrawals):
        raise ValueError('撤回清单无效/重复')
    output=output_directory(output)
    for item in manifest['runs']:
        if item['case_id'] in seen:raise ValueError('回放题目重复')
        seen.add(item['case_id']);case=lookup[item['case_id']];cases.append(case)
        path=Path(item['path']);path=path if path.is_absolute() else Path(runs_manifest).parent/path
        run=read(path)
        if item['run_sha256']!=digest(run):raise ValueError('回放运行哈希不匹配')
        human=Path(review_dir)/(case['id']+'.json') if review_dir else None
        ai=read(Path(ai_results)/'ai-reviews'/(case['id']+'.json')) if ai_results else None
        if ai and ai.get('status') in {'reviewed','unable_to_judge'}:
            config=ai.get('model_config',{});batch=ai_payload['judge_identity']
            if ({k:v for k,v in config.items() if k!='provider_models'}!={k:v for k,v in batch.items() if k!='provider_models'}
                or not isinstance(config.get('provider_models'),list)
                or not set(config['provider_models']).issubset(batch.get('provider_models',[]))):
                raise ValueError('judge逐题配置与批次身份不一致')
        withdrawal=next((w for w in withdrawals if w['case_id']==case['id']),None)
        row=score_case(b,case,run,read(human) if human and human.is_file() else None,
            mode='ai_assisted' if ai_results else manifest['mode'],ai_review=ai,review_withdrawal=withdrawal)
        row.update(elapsed_seconds=item.get('elapsed_seconds'),telemetry=run.get('telemetry',{}));rows.append(row)
        atomic_write_json(output/'reviews'/(case['id']+'.json'),review_template(b,case,run))
    payload={'mode':'ai_assisted' if ai_results else manifest['mode'],'identity':manifest['identity'],'results':rows,'summary':summary(rows),
        'evaluator_sha256':file_hash(Path(__file__).with_name('benchmark_score.py')),
        'note':'原执行身份不变；当前评分器对保存记录离线重评分，无模型请求。评分器修订不是Agent效果提升。未提供撤回记录的人工评分仅为历史回放，不代表当前有效。'}
    if ai_payload:payload['judge_identity']=ai_payload['judge_identity']
    report(output,payload);return payload


def score_case(bundle,case,run,review=None,*,mode='agent',ai_review=None,review_withdrawal=None):
    """Keep evaluator failures distinct from wrong business answers."""
    error=run.get('evaluation_error')
    if not error:
        try:return score_run(bundle,case,run,review,mode=mode,ai_review=ai_review,review_withdrawal=review_withdrawal)
        except Exception as exc:
            from gamelens.core.decision_trace import safe_audit
            error={'type':type(exc).__name__,'reason':safe_audit(str(exc))}
    row={'case_id':case['id'],'split':case['split'],'category':case['category'],'mode':mode,'formal_eligible':False,
        'automatic_passed':False,'accepted':None,'checks':{'evaluation':False},'failed_checks':['evaluation_error'],
        'evaluation_error':error,'provenance_errors':[],'facts':[],'human_review':{'status':'pending'},
        'run_sha256':digest(run),'semantic_accuracy_automatically_verified':False}
    if mode=='ai_assisted':row['ai_review']={'status':'blocked','passed':None,'total':None,'reason':'evaluation_error'}
    return row


def serializable_snapshot(value):
    """Encode the known graph interrupt type, without lossy default=str."""
    from langgraph.types import Interrupt
    from dataclasses import asdict
    if isinstance(value,Interrupt):return serializable_snapshot(asdict(value))
    if isinstance(value,dict):return {k:serializable_snapshot(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serializable_snapshot(v) for v in value]
    return value


def archive_agent_result(bundle,case,result,root,output,mode,elapsed,manifest,batch_identity):
    """Save expensive answer before optional receipts/reports/scoring can fail."""
    result=serializable_snapshot(result)
    if result.get('__interrupt__'):
        result.update(status='interrupted',approval_pending=True)
    path=output/'private/runs'/(case['id']+'.json')
    atomic_write_json(path,result)
    manifest.append({'case_id':case['id'],'path':'private/runs/'+case['id']+'.json','run_sha256':digest(result),'elapsed_seconds':elapsed})
    atomic_write_json(output/'runs_manifest.json',{'mode':mode,'identity':batch_identity,'runs':manifest})
    errors=[];receipts={}
    for observation in result.get('observations',[]):
        source=observation.get('data',{}).get('source',{})
        for reference in [source,*source.get('reference_files',[])]:
            try:
                original=Path(reference.get('path',''))
                if original.is_file() and original.resolve().is_relative_to(root.resolve()) and original.stat().st_size<8_000_000:
                    receipts[file_hash(original)]={'sha256':file_hash(original),'content':original.read_text(encoding='utf-8')}
            except (OSError,UnicodeError,ValueError) as exc:errors.append({'stage':'receipt','type':type(exc).__name__})
    result['source_receipts']=list(receipts.values())
    try:
        report_path=result.get('artifacts',{}).get('final_report')
        if report_path and Path(report_path).is_file():atomic_write_text(output/'agent_reports'/(case['id']+'.md'),Path(report_path).read_text(encoding='utf-8'))
        atomic_write_json(output/'reviews'/(case['id']+'.json'),review_template(bundle,case,result))
    except Exception as exc:errors.append({'stage':'auxiliary','type':type(exc).__name__})
    if errors:result['archive_warnings']=errors
    row=score_case(bundle,case,result,mode=mode)
    if row.get('evaluation_error'):result['evaluation_error']=row['evaluation_error']
    atomic_write_json(path,result)
    manifest[-1]['run_sha256']=digest(result)
    atomic_write_json(output/'runs_manifest.json',{'mode':mode,'identity':batch_identity,'runs':manifest})
    row['run_sha256']=digest(result)
    row.update(elapsed_seconds=elapsed,telemetry=result['telemetry'],execution_status=result.get('status'))
    return row


def agent_cases(bundle,*,split='development',limit=3):
    if split not in {'all','development','holdout'}:raise ValueError('未知题目分组')
    if limit is not None and (type(limit) is not int or limit<1):raise ValueError('题目上限须为正整数')
    return [c for c in bundle['investigations'] if (split=='all' or c['split']==split) and c['review']['status']=='reviewed' and not c['blocking'] and
            not (c.get('response_scenario')=='generation_interruption' and c.get('recovery_fixture'))][:limit]


def resume_history(bundle_root,manifest_path):
    """Validate every immutable predecessor; never trust a cached list of IDs."""
    from gamelens.evaluation.benchmark_judge import _saved_runs
    history=[];seen=set();path=Path(manifest_path).resolve()
    while True:
        if path in seen or len(history)>=32:raise ValueError('续跑清单循环或层数超限')
        seen.add(path)
        _,prior,saved=_saved_runs(bundle_root,path);history.append((path,prior,saved))
        parent=prior['identity'].get('continuation')
        if not parent:break
        path=Path(parent['runs_manifest']).resolve()
        if file_hash(path)!=parent['sha256']:raise ValueError('原续跑清单哈希不匹配')
    return history


def run_agent(bundle_root,output,project,*,mode='agent',split='development',limit=3,max_http=None,allow_paid=False,active_budget_seconds=1800,resume_from=None,budget_ledger=None):
    if not allow_paid or not max_http or max_http<1:raise ValueError('真实模型需要--allow-paid及明确--max-http预算')
    b=load(bundle_root);validate(bundle_root)
    cases=agent_cases(b,split=split,limit=limit)
    if not cases:raise ValueError('没有已审核且资料齐全的调查题；不发送HTTP请求')
    if mode=='fixed_workflow' and any(not c['oracle_plan'] for c in cases):raise ValueError('固定工作流计划尚未人工确认')
    from gamelens.core.budget import TaskBudgetStore
    from gamelens.core.llm import OpenAICompatibleClient
    from gamelens.core.runtime import InvestigationAgent,LLMPlanner
    from gamelens.storage.telemetry import summarize,TelemetryStore
    predecessor=None
    if bool(resume_from)!=bool(budget_ledger):raise ValueError('续跑必须同时提供原运行清单和原预算账本')
    if resume_from:
        import sqlite3
        history=resume_history(bundle_root,resume_from)
        _,prior,_=history[0]
        expected={'split':split,'seed':0,'limit':limit,'max_http':max_http,'active_budget_seconds':active_budget_seconds}
        if any(p['mode']!=mode or any(p['identity']['configuration'].get(k)!=v for k,v in expected.items()) for _,p,_ in history):raise ValueError('续跑配置或预算与原批次不符')
        allowed={c['id'] for c in cases};done={c['id'] for _,_,saved in history for c,r,i in saved}
        if not done.issubset(allowed):raise ValueError('续跑题目与原批次不符')
        budget_ledger=Path(budget_ledger).resolve()
        original_ledger=history[-1][0].parent/'private/http-budget.sqlite'
        if budget_ledger!=original_ledger or not budget_ledger.is_file():raise ValueError('必须复用原批次预算账本')
        with sqlite3.connect(budget_ledger.as_uri()+'?mode=ro',uri=True) as db:
            row=db.execute('SELECT llm_request_limit,active_seconds_limit FROM task_budgets WHERE thread_id=?',('benchmark-batch',)).fetchone()
        if row is None or row!=(max_http,float(active_budget_seconds)):raise ValueError('预算上限与原账本不符，禁止扩容')
        predecessor={'runs_manifest':str(Path(resume_from).resolve()),'sha256':file_hash(resume_from),
            'code_sha256':prior['identity']['code_sha256'],'saved_case_ids':sorted(done)}
        cases=[c for c in cases if c['id'] not in done]
        if not cases:raise ValueError('没有未执行的调查题')
    output=output_directory(output);budget=TaskBudgetStore(budget_ledger or output/'private/http-budget.sqlite',request_limit=max_http,active_seconds_limit=active_budget_seconds)
    budget_before=budget.snapshot('benchmark-batch')
    telemetry=TelemetryStore(output/'private/telemetry.sqlite')
    client=OpenAICompatibleClient(os.getenv('LLM_MODEL',''),request_budget=budget,budget_thread_id='benchmark-batch',
        telemetry_sink=lambda event:telemetry.record('benchmark-batch',event))
    rows=[];manifest=[]
    batch_identity=identity(b,cases,{'split':split,'seed':0,'limit':limit,'max_http':max_http,'active_budget_seconds':active_budget_seconds})
    if predecessor:batch_identity['continuation']=predecessor
    atomic_write_json(output/'runs_manifest.json',{'mode':mode,'identity':batch_identity,'runs':manifest})
    for case in cases:
        if budget.snapshot('benchmark-batch')['llm_requests_remaining']==0:
            missing=state_for(b,case)
            missing.update(status='not_executed',error='batch_http_budget_exhausted',telemetry={'http_requests':0})
            atomic_write_json(output/'private/runs'/(case['id']+'.json'),missing)
            row=score_case(b,case,missing,mode=mode)
            row.update(blocked='batch_http_budget_exhausted',elapsed_seconds=None,telemetry=missing['telemetry']);rows.append(row)
            manifest.append({'case_id':case['id'],'path':'private/runs/'+case['id']+'.json','run_sha256':digest(missing),'elapsed_seconds':None})
            atomic_write_json(output/'runs_manifest.json',{'mode':mode,'identity':batch_identity,'runs':manifest})
            continue
        with tempfile.TemporaryDirectory(prefix='gl-agent-') as directory:
            root=Path(directory);materialize(b,root,case);registry=registry_for(b,root,case,client,project)
            state=state_for(b,case);before={e['id'] for e in telemetry.events('benchmark-batch')};start=time.perf_counter()
            with frozen_environment(b,case['response_scenario']):
                agent=InvestigationAgent(LLMPlanner(client),registry,task_budget=budget,budget_thread_id='benchmark-batch')
                result=state
                try:
                    if mode=='fixed_workflow':
                        for i,call in enumerate(case['oracle_plan'],1):
                            result['step_count']=i-1;tool=registry.execute(call['tool'],call['arguments'],result)
                            result['observations'].append({'step':i,'tool':call['tool'],'arguments':call['arguments'],**tool.model_dump(mode='json')})
                            result['artifacts'].update(tool.artifacts)
                        result['step_count']=len(case['oracle_plan']);result['max_steps']=result['step_count']
                    for snapshot in agent.stream(result):
                        result=serializable_snapshot(dict(snapshot))
                        atomic_write_json(output/'private/progress'/(case['id']+'.json'),result)
                except Exception as error:result={**result,'status':'failed','error':type(error).__name__+': '+str(error)}
            result['evaluation']=state['evaluation'];result['telemetry']=summarize([e for e in telemetry.events('benchmark-batch') if e['id'] not in before])
            elapsed=time.perf_counter()-start
            row=archive_agent_result(b,case,result,root,output,mode,elapsed,manifest,batch_identity);rows.append(row)
            print(f"Benchmark Agent: {len(manifest)}/{len(cases)} {case['id']} {row.get('execution_status')} HTTP={result['telemetry'].get('http_requests')}",flush=True)
    payload={'mode':mode,'identity':batch_identity,
        'model':client.model,'results':rows,'summary':summary(rows),
        'request_budget_before':budget_before,'request_budget':budget.snapshot('benchmark-batch'),
        'budget_source':{'ledger':str(budget.path),'external':budget_ledger is not None},
        'note':'人工答案评分待填写；Steam读取仅用冻结夹具。'}
    report(output,payload);atomic_write_json(output/'runs_manifest.json',{'mode':mode,'identity':payload['identity'],'runs':manifest})
    return payload
