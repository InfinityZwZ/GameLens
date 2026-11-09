"""Evidence-based scoring; no keyword proxy for answer semantics."""
from __future__ import annotations

import math
from collections import Counter

from gamelens.evaluation.benchmark_data import case_hash, digest
from gamelens.evaluation.quality import RUBRIC_KEYS, _arguments_match
from gamelens.evaluation.retrieval import percentile
from gamelens.core.grounding import resolve_path, same_value, validate_grounding, validate_news_citations, validate_comment_citations


def eligible_records(game,filters,bounds=None):
    """Independent gold eligibility, not a call to the production query tool."""
    bounds=bounds or {};selected=[]
    from gamelens.core.local_time import local_datetime
    def instant(value):return local_datetime(value).timestamp() if value else None
    left=bounds.get('start_inclusive',instant(filters.get('start_at')))
    right=bounds.get('end_exclusive',instant(filters.get('end_at')))
    for r in game['reviews']:
        time=r['created_at_timestamp']
        if left is not None and time<left or right is not None and time>=right:continue
        if filters.get('language') not in {None,'all'} and r['language']!=filters['language']:continue
        if filters.get('is_positive') is not None and r['is_positive']!=filters['is_positive']:continue
        minutes=r['playtime_minutes'] if filters.get('playtime_basis')=='lifetime' else r['playtime_at_review_minutes']
        minimum=filters.get('min_playtime_hours',0)*60;maximum=filters.get('max_playtime_hours')
        if minutes<minimum or filters.get('min_playtime_exclusive') and minutes==minimum:continue
        if maximum is not None and (minutes>maximum*60 or filters.get('max_playtime_exclusive') and minutes==maximum*60):continue
        selected.append(r)
    return selected


def ranking_score(case, retrieved):
    grades={(l['kind'],str(l['source_id'])):l['relevance'] for l in case['labels']}
    relevant={k for k,v in grades.items() if v>0}
    ordered=list(dict.fromkeys(retrieved))
    if not relevant:
        return {'is_no_answer':True,'no_answer_candidate_false_positive':bool(ordered),
            'recall_at_5':None,'recall_at_10':None,'mrr_at_10':None,'ndcg_at_10':None}
    dcg=sum((2**grades.get(k,0)-1)/math.log2(i+1) for i,k in enumerate(ordered[:10],1))
    ideal=sum((2**v-1)/math.log2(i+1) for i,v in enumerate(sorted((grades[k] for k in relevant),reverse=True)[:10],1))
    result={'is_no_answer':False,'no_answer_candidate_false_positive':None,
        'recall_at_5':len(relevant&set(ordered[:5]))/len(relevant),
        'recall_at_10':len(relevant&set(ordered[:10]))/len(relevant),
        'mrr_at_10':next((1/i for i,k in enumerate(ordered[:10],1) if k in relevant),0),
        'ndcg_at_10':dcg/ideal}
    for role in ('support','counterexample'):
        ids={(l['kind'],str(l['source_id'])) for l in case['labels'] if l['relevance']>0 and l['role']==role}
        result[role+'_coverage_at_10']=len(ids&set(ordered[:10]))/len(ids) if ids else None
    return result


def review_template(bundle,case,run):
    return {'completed':False,'case_id':case['id'],'case_sha256':case_hash(case),
        'dataset_sha256':bundle['manifest']['dataset_sha256'],'run_sha256':digest(run),
        'reviewer':'','reviewed_at':'','notes':'','critical_errors':[],
        'scores':{key:None for key in sorted(RUBRIC_KEYS)}}


def _human(bundle,case,run,review):
    if not review or not review.get('completed'):return {'status':'pending','passed':None}
    template=review_template(bundle,case,run)
    if any(review.get(k)!=template[k] for k in ('case_id','case_sha256','dataset_sha256','run_sha256')):
        return {'status':'invalid','passed':False,'reason':'人工评分绑定不匹配'}
    scores=review.get('scores',{})
    if (set(scores)!=RUBRIC_KEYS or any(type(v) is not int or v not in (0,1,2) for v in scores.values())
        or not all(str(review.get(k,'')).strip() for k in ('reviewer','reviewed_at','notes'))
        or not isinstance(review.get('critical_errors'),list)):
        return {'status':'invalid','passed':False,'reason':'评分字段不完整'}
    from datetime import datetime
    try:datetime.fromisoformat(review['reviewed_at'].replace('Z','+00:00'))
    except ValueError:return {'status':'invalid','passed':False,'reason':'评分时间无效'}
    passed=(not review['critical_errors'] and sum(scores.values())>=8
        and scores['evidence_support']==2 and scores['scope_handling']==2)
    return {'status':'reviewed','passed':passed,'total':sum(scores.values()),'scores':scores,
        'reviewer':review['reviewer'],'notes':review['notes'],'critical_errors':review['critical_errors']}


def score_run(bundle,case,run,review=None,*,mode='agent',ai_review=None,review_withdrawal=None):
    eligible=case['review']['status']=='reviewed' and not case.get('blocking')
    game=bundle['corpus'][case['appid']];expected=case['expectations']
    observations=run.get('observations',[]);claims=run.get('final_claims',[])
    citations=run.get('final_citations',[]);news=run.get('final_news_citations',[]);comments=run.get('final_comment_citations',[])
    errors=[]
    try:
        errors=validate_grounding(observations,claims,citations)+validate_news_citations(observations,news)+validate_comment_citations(observations,comments)
    except (KeyError,TypeError,IndexError,AttributeError,ValueError):errors.append('引用结构无效')
    originals={'review':{str(r['review_id']):r['text'] for r in game['reviews']},
        'news':{str(r['news_id']):r.get('contents') or r.get('title','') for r in game['news']},
        'comment':{(str(r['news_id']),str(r['comment_id'])):r['text'] for r in game['comments']}}
    for kind,items,key in (('review',citations,'review_id'),('news',news,'news_id'),('comment',comments,'comment_id')):
        for item in items:
            identity=(str(item.get('news_id')),str(item.get(key))) if kind=='comment' else str(item.get(key))
            text=originals[kind].get(identity);quote=item.get('quote')
            if not text or not isinstance(quote,str) or not quote.strip() or quote not in text:errors.append('引用不在冻结原文：'+str(identity))
    successful=[o for o in observations if o.get('ok')]
    scopes=[o['data'] for o in successful if o.get('tool') in {'query_reviews','search_game_evidence'}]
    scopes += [o['data'][side] for o in successful if o.get('tool')=='compare_cohorts' for side in ('cohort_a','cohort_b') if side in o.get('data',{})]
    required_filters=expected.get('filters',{})
    filter_check=not scopes or all(_arguments_match(required_filters,s.get('filters',{})) for s in scopes)
    facts=[]
    for fact in expected.get('facts',[]):
        matches=[]
        for o in successful:
            data=o.get('data',{})
            for prefix in ([],['cohort_a'],['cohort_b']):
                try:
                    scope=resolve_path(data,prefix) if prefix else data
                    if not _arguments_match(fact.get('filters',{}),scope.get('filters',{})):continue
                    path=prefix+fact['path']
                    if same_value(resolve_path(data,path),fact['value']):
                        matches.append(any(c.get('source_step')==o.get('step') and c.get('fact_path')==path and same_value(c.get('value'),fact['value']) for c in claims))
                except (KeyError,IndexError,TypeError):continue
        facts.append({'requirement':fact,'passed':any(matches)})
    sources=[o['data']['source'] for o in successful if o.get('data',{}).get('source')]
    hashes=set(bundle['manifest']['files'].values())
    # Generated window/dataset files are accepted only when their archived
    # bytes match their hash and every record matches the trusted frozen corpus.
    import hashlib,json
    for receipt in run.get('source_receipts',[]):
        content=receipt.get('content','')
        if hashlib.sha256(content.encode('utf-8')).hexdigest()!=receipt.get('sha256'):continue
        try:
            data=json.loads(content)
            records=data if isinstance(data,list) else data.get('comments',[])
            if isinstance(data,dict) and ('comments' not in data or 'start' not in data):continue
            trusted={str(r['review_id']):r for r in game['reviews']}
            trusted_news={str(n['news_id']):n for n in game['news']}
            if not all(r==trusted.get(str(r.get('review_id'))) if 'review_id' in r else r==trusted_news.get(str(r.get('news_id'))) for r in records):continue
            if isinstance(data,dict):
                filtered=eligible_records(game,{'language':data.get('language')},{'start_inclusive':data['start'],'end_exclusive':data.get('counted_end',data['end'])})
                if data.get('reported_total')!=len(filtered):continue
            hashes.add(receipt['sha256'])
        except (ValueError,TypeError,KeyError):continue
    source_ok=bool(sources) and all(s.get('sha256') in hashes and
        all(r.get('sha256') in hashes for r in s.get('reference_files',[])) for s in sources)
    tags=run.get('evaluation',{})
    outcome=expected['outcome'];completed=run.get('status')=='completed' and bool(run.get('final_answer','').strip())
    outcome_ok=completed and (run.get('scope_limited') is True if outcome=='limited' else
        bool(run.get('investigation_progress',{}).get('needs_clarification') or run.get('answer_origin')=='clarification') if outcome=='clarification' else True)
    target=expected.get('target_news_id')
    target_ok=not target or any(o.get('data',{}).get('patch_news_id')==target or
        any(str(n.get('news_id'))==target for n in o.get('data',{}).get('patches',[])) for o in successful)
    evidence=expected.get('evidence',[])
    evidence_ok=all(any(str(c.get('review_id' if e['kind']=='review' else 'news_id'))==str(e['source_id'])
        for c in (citations if e['kind']=='review' else news)) for e in evidence if e.get('required',False))
    checks={'question_identity':run.get('appid')==case['appid'] and run.get('goal')==case['goal'],
        'dataset_identity':tags.get('case_id')==case['id'] and tags.get('case_sha256')==case_hash(case) and tags.get('dataset_sha256')==bundle['manifest']['dataset_sha256'],
        'outcome':outcome_ok,'player_scope':filter_check,'gold_facts':all(f['passed'] for f in facts),
        'target_update':target_ok,'citation_integrity':not errors,'source_identity':source_ok or outcome=='clarification',
        'evidence_coverage':evidence_ok and (bool(citations or comments) if expected.get('requires_citation') else True),
        'verification':any(o.get('tool')=='verify_investigation' for o in successful) or outcome=='clarification',
        'step_budget':len(observations)<=12,
        'required_behavior':all(any(o.get('tool')==t for o in observations) for t in expected.get('required_events',[])),
        'discussion_source':not expected.get('discussion_news_id') or
            all(c.get('news_id')==expected['discussion_news_id'] for c in comments) and not citations}
    truthful=True
    for scope in scopes:
        if 'stats' not in scope:continue
        records=eligible_records(game,scope.get('filters',{}),scope.get('bounds',{}));positive=sum(r['is_positive'] for r in records)
        gold={'count':len(records),'positive_count':positive,'negative_count':len(records)-positive,
            'positive_rate':positive/len(records) if records else None,'negative_rate':(len(records)-positive)/len(records) if records else None}
        truthful=truthful and all(k not in scope['stats'] or same_value(scope['stats'][k],v) for k,v in gold.items())
    checks['statistics_truth']=truthful
    human=_human(bundle,case,run,review)
    automatic=all(checks.values())
    accepted=(False if not automatic else human['passed']) if eligible and mode in {'agent','fixed_workflow'} else None
    if review_withdrawal is not None:
        from gamelens.evaluation.benchmark_judge import binding
        expected_binding=binding(bundle,case,run)
        if (review_withdrawal.get('status')!='withdrawn' or not review_withdrawal.get('reason')
            or not review_withdrawal.get('review_sha256') or not review_withdrawal.get('withdrawn_at')
            or any(review_withdrawal.get(k)!=v for k,v in expected_binding.items())):
            raise ValueError('人工评分撤回记录绑定不匹配')
        human={'status':'withdrawn','passed':None,'previous_review_sha256':review_withdrawal['review_sha256'],
            'reason':review_withdrawal['reason']}
        accepted=None
    ai=None
    if mode=='ai_assisted':
        from gamelens.evaluation.benchmark_judge import verify_review
        ai=verify_review(bundle,case,run,ai_review or {'status':'not_executed'})
        accepted=bool(automatic and ai['passed']) if eligible and ai['status']=='reviewed' else None
    result={'case_id':case['id'],'split':case['split'],'category':case['category'],'mode':mode,
        'formal_eligible':eligible,'automatic_passed':automatic,'checks':checks,
        'failed_checks':[k for k,v in checks.items() if not v],'provenance_errors':errors,'facts':facts,
        'human_review':human,'accepted':accepted,'run_sha256':digest(run),
        'semantic_accuracy_automatically_verified':False}
    if ai is not None:result['ai_review']=ai
    return result


def summary(results):
    formal=[r for r in results if r.get('formal_eligible')]
    reviewed=[r for r in formal if r.get('human_review',{}).get('status')=='reviewed']
    latencies=[r['elapsed_seconds'] for r in results if r.get('elapsed_seconds') is not None]
    result={'executed_cases':len(results),'formal_cases':len(formal),'human_scored_cases':len(reviewed),
        'task_success_rate':sum(r['accepted'] for r in reviewed)/len(formal) if formal and len(reviewed)==len(formal) else None,
        'automatic_check_pass_rate':sum(r['automatic_passed'] for r in formal)/len(formal) if formal else None,
        'latency_p50_seconds':percentile(latencies,.5),'latency_p95_seconds':percentile(latencies,.95),
        'failure_checks':dict(Counter(k for r in results for k in r.get('failed_checks',[]))),
        'http_requests':sum(r.get('telemetry',{}).get('http_requests',0) for r in results) if results and all('http_requests' in r.get('telemetry',{}) for r in results) else None,
        'note':'工具Oracle与未审核题不产生Agent答案准确率；缺失用量不是0。'}
    ai_rows=[r for r in formal if r.get('mode')=='ai_assisted']
    if ai_rows:
        scored=[r for r in ai_rows if r.get('ai_review',{}).get('status')=='reviewed']
        result.update(ai_requested_cases=len(ai_rows),ai_scored_cases=len(scored),
            ai_completion_rate=len(scored)/len(ai_rows),
            ai_semantic_pass_rate=sum(r['ai_review']['passed'] for r in scored)/len(scored) if scored else None,
            ai_assisted_task_success_rate=sum(r['accepted'] for r in scored)/len(ai_rows) if len(scored)==len(ai_rows) else None)
        result['task_success_rate']=None
        result['note']='AI评分与程序核对分别报告；AI辅助综合通过率不是人工确认的答案准确率。'
    return result


def compare(base,other):
    if base.get('mode')=='ai_assisted' or other.get('mode')=='ai_assisted':
        if base.get('mode')!=other.get('mode') or base.get('judge_identity')!=other.get('judge_identity'):
            raise ValueError('不可比：judge评分模式/模型/提示词/配置不同')
        other_cases={r['case_id']:r for r in other['results']}
        for row in base['results']:
            counterpart=other_cases.get(row['case_id'])
            if counterpart and row.get('ai_review',{}).get('model_config')!=counterpart.get('ai_review',{}).get('model_config'):
                raise ValueError('不可比：judge逐题供应商模型/配置不同')
            if counterpart and row.get('ai_review',{}).get('validation_version','gamelens-ai-judge-validation-v1')!=counterpart.get('ai_review',{}).get('validation_version','gamelens-ai-judge-validation-v1'):
                raise ValueError('不可比：judge校验版本不同；离线纠正不等于Agent能力提升')
    for key in ('benchmark_version','manifest_sha256','dataset_sha256','case_hashes','environment','configuration'):
        if base['identity'].get(key)!=other['identity'].get(key):raise ValueError('不可比：'+key+'不同')
    if base.get('mode')=='oracle' or other.get('mode')=='oracle':raise ValueError('Oracle不能比较为Agent答案准确率')
    left={r['case_id']:r for r in base['results']};right={r['case_id']:r for r in other['results']}
    if not left or set(left)!=set(right):raise ValueError('不可比：题目覆盖不同')
    if set(left)!=set(base['identity']['case_hashes']):raise ValueError('不可比：计划题目未全部执行')
    review_field='ai_review' if base.get('mode')=='ai_assisted' else 'human_review'
    if any(not r.get('formal_eligible') or r.get('accepted') is None or r.get(review_field,{}).get('status')!='reviewed'
           for r in [*left.values(),*right.values()]):
        raise ValueError('不可比：题目/答案审核未完成')
    return {'comparable':True,'base':summary(list(left.values())),'new':summary(list(right.values())),
        'regressions':[k for k in left if left[k]['accepted'] and not right[k]['accepted']],
        'improvements':[k for k in left if not left[k]['accepted'] and right[k]['accepted']],
        'success_rate_delta':sum(right[k]['accepted']-left[k]['accepted'] for k in left)/len(left)}
