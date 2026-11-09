"""Frozen public evidence and human-review inputs for GameLens Benchmark v1."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

from gamelens.evaluation.quality import digest
from gamelens.storage.atomic_io import atomic_write_json, atomic_write_text

VERSION='gamelens-benchmark-v1'
APPIDS=(1623730,251570,730)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def case_hash(case):
    return digest({k:v for k,v in case.items() if k!='review'})


def scrub(value):
    """Keep source text intact; remove structured author identity and host paths."""
    if isinstance(value,list):return [scrub(v) for v in value]
    if isinstance(value,dict):
        return {k:scrub(v) for k,v in value.items() if k not in
            {'author','steamid','steam_id','avatar','personaname','profile_url','path','artifact_path'}}
    return value


def _record(case):
    case['review']={'status':'pending_review','reviewer':'','reviewed_at':'','notes':'',
        'case_sha256':case_hash(case)}
    return case


def _labels(records, terms, kind, role='support', maximum=8):
    identity='review_id' if kind=='review' else 'news_id'
    text='text' if kind=='review' else 'contents'
    return [{'kind':kind,'source_id':str(r[identity]),'relevance':2,'role':role}
        for r in records if any(t.casefold() in (r.get(text,'')+' '+r.get('title','')).casefold() for t in terms)][:maximum]


def candidates(corpus, clock):
    topics=[
        ('performance',1623730,'reviews','性能与画面卡顿',['卡顿','掉帧','fps','stutter'],6,'development'),
        ('building',1623730,'reviews','建造与地图体验',['建造','建筑','地图','building'],6,'development'),
        ('progress',1623730,'reviews','存档与进度体验',['存档','进度','save'],6,'development'),
        ('official_fixes',251570,'announcements','官方修复与稳定性说明',['fix','stability','修复'],6,'development'),
        ('negation',1623730,'reviews','流畅运行的相反体验',['流畅','不卡','不再','smooth'],4,'holdout'),
        ('network',730,'reviews','掉线与连接问题',['掉线','延迟','disconnect','network'],4,'holdout'),
        ('activity',251570,'announcements','活动内容与发布时间',['event','周年','活动','sale'],4,'holdout'),
        ('no_answer',1623730,'both','当前快照能否证明内核驱动泄漏',['内核驱动泄漏','kernel driver leak'],4,'holdout'),
    ]
    retrieval=[]
    for group,appid,kind,topic,terms,n,split in topics:
        game=corpus.get(appid,{})
        labels=[]
        if kind in {'reviews','both'}:labels+=_labels(game.get('reviews',[]),terms,'review','counterexample' if group=='negation' else 'support')
        if kind in {'announcements','both'}:
            from gamelens.tools.steam_news import is_official_announcement
            eligible_news=[n for n in game.get('news',[]) if group not in {'official_fixes','activity'} or is_official_announcement(n)]
            labels+=_labels(eligible_news,terms,'announcement')
        english={'performance':'poor performance and frame stuttering','building':'building and map experience',
            'progress':'saved progress and save files','official_fixes':'official fixes and stability improvements',
            'negation':'smooth performance without crashes','network':'disconnects and network connection problems',
            'activity':'official events and announcement dates','no_answer':'a confirmed kernel driver memory leak'}[group]
        queries=[f'查找{topic}的直接原文',f'Find direct evidence about {english}.',
            f'{topic} feedback：保留相关上下文',f'有没有关于{topic}的明确描述？',
            f'Which sources describe {english}? Do not infer the content from recommendation votes.',
            f'{topic} feedback：哪些结论仍不能确认？']
        for i in range(n):
            needs=['reviews'] if kind=='reviews' else ['news'] if kind=='announcements' else ['reviews','news']
            retrieval.append(_record({'id':f'ret-{group}-{i+1}','group_id':f'ret-{group}',
                'appid':appid,'query':queries[i],'kind':kind,'split':split,
                'language':'zh' if i%3==0 else 'en' if i%3==1 else 'mixed',
                'filters':{'period':'all'},'tags':[group,kind]+(['long_announcement'] if kind=='announcements' else []),
                'source_constraint':'official' if group in {'official_fixes','activity'} else 'any',
                'labels':labels,'needs':needs,'blocking':[] if all(game.get(k) for k in needs) else ['missing_source'],
                'annotation_note':'规则预选候选，仅供人工核对；无标签不自动表示无答案。',
                'expected_no_answer':None}))
    groups=[
        ('locate',5,3, ['最近一次更新对应哪个官方补丁？给出标题、发布时间和原文来源。','定位指定补丁公告并说明改动。',
            '区分官方补丁与媒体新闻，列出可核对的版本。','寻找活动公告，说明它为何不能代替补丁。','同一天若有多份公告，应如何确定用户指的是哪一份？']),
        ('statistics',6,4,['统计冻结快照中文评论的数量与好评率。','统计评价时长超过100小时的中文评论。',
            '对比评价时长与当前累计时长100小时的两组统计。','统计指定日期区间的缓存评论与推荐率。',
            '比较目标更新前后各7天的缓存好评率。','空窗口能否计算好评率？说明分母和数据覆盖。']),
        ('evidence',7,5,['查找卡顿原文与流畅的相反体验。','推荐游戏但抱怨问题的评论有哪些？',
            '区分启动闪退与游玩过程退出，是否能认定同一原因？','解释否定句中的正常体验，保留原文。',
            '找建造或地图问题的原文，哪些原因仍无法确认？','官方承诺修复能否证明所有玩家问题消失？',
            '后来修改的评论能否证明更新当日的状态？']),
        ('discussion',4,3,['调查该公告下玩家的赞同、质疑和具体反馈。','读取该公告留言原文，区分正文与引用他人的内容。',
            '公告有558条回复但只采集500条，如何说明反馈范围？','公告留言能否按评价时长100小时筛选，并计算好评率？']),
        ('conversation',4,3,['继续上一轮目标更新，仅看中文玩家原文。','继承上一轮目标，但把时长条件改为评价时超过100小时。',
            '这次改看英文反馈，应如何覆盖上一轮中文条件？','用户说那个版本但历史有两个版本，应如何处理？']),
        ('reliability',4,2,['缓存中没有最新补丁，应先读取冻结的官方接口响应再定位。','缓存不足时补采指定窗口，不能把空缓存当零反馈。',
            '官方接口超时，应保留资料并说明尚未确认的部分。','模型生成中断后恢复任务，保留可核验的证据，不冒充完成。']),
    ]
    investigations=[]
    for category,n,development,goals in groups:
        for i in range(n):
            appid=251570 if category=='discussion' or category=='locate' and i==3 else 730 if category=='reliability' and i in (1,3) else 1623730
            game=corpus.get(appid,{})
            news=game.get('news',[])
            from gamelens.tools.steam_news import retained_patch_candidate
            patches=sorted([r for r in news if retained_patch_candidate(r)],key=lambda r:(r['published_at_timestamp'],r['news_id']),reverse=True)
            target=patches[0] if patches else None
            comments=game.get('comments',[])
            discussion_id=comments[0]['news_id'] if comments else None
            needs=['news'] if category=='locate' else ['news','comments'] if category=='discussion' else ['reviews','news']
            filters={'period':'all'}
            if category=='statistics' and i in (0,1,2):filters['language']='schinese'
            if category=='statistics' and i==1:filters.update(min_playtime_hours=100,min_playtime_exclusive=True,playtime_basis='at_review')
            facts=[];plan=[];evidence=[]
            if category=='statistics' and i in (0,1):
                rows=[r for r in game.get('reviews',[]) if r.get('language')=='schinese' and
                    (i==0 or (r.get('playtime_at_review_minutes') or 0)>6000)]
                facts=[{'path':['stats','count'],'value':len(rows),'filters':filters},
                    {'path':['stats','positive_count'],'value':sum(bool(r['is_positive']) for r in rows),'filters':filters}]
                plan=[{'tool':'query_reviews','arguments':{'appid':appid,'filters':filters}},
                      {'tool':'verify_investigation','arguments':{'appid':appid}}]
            elif category=='locate':
                plan=[{'tool':'list_patches','arguments':{'appid':appid,'limit':5}}]
                if i==0:plan=[{'tool':'prepare_patches','arguments':{'appid':appid,'check_latest':True}},
                    {'tool':'list_patches','arguments':{'appid':appid,'limit':1}},
                    {'tool':'verify_investigation','arguments':{'appid':appid}}]
            elif category=='discussion' and discussion_id:
                plan=[{'tool':'query_announcement_comments','arguments':{'appid':appid,'news_id':discussion_id,'limit':20}},
                      {'tool':'verify_investigation','arguments':{'appid':appid}}]
            if category=='evidence':evidence=_labels(game.get('reviews',[]),['卡顿','闪退','地图','建造'],'review')
            outcome='clarification' if category=='conversation' and i==3 or category=='locate' and i==4 else 'limited' if category=='reliability' and i>=2 or category=='discussion' and i==3 else 'answer'
            case={'id':f'task-{category}-{i+1}','group_id':f'task-{category}-{i+1}',
                'appid':appid,'category':category,'goal':goals[i],
                'split':'development' if i<development else 'holdout','needs':needs,
                'blocking':[] if all(game.get(k) for k in needs) else ['missing_source'],
                'oracle_plan':plan,'history':[],'response_scenario':'timeout' if category=='reliability' and i==2 else 'normal',
                'expectations':{'outcome':outcome,'filters':filters,'facts':facts,'evidence':evidence,
                    'target_news_id':target['news_id'] if target and category=='locate' and i==0 else None,
                    'discussion_news_id':discussion_id if category=='discussion' else None,
                    'requires_citation':category in {'evidence','discussion'},
                    'allowed_limits':['只描述冻结资料','推荐态度不等于问题是否存在','当前正文不是历史时点原文'],
                    'must_not_claim':['由少量候选推断总体比例','未采集等于不存在','官方修复承诺等于玩家问题已消失']},
                'annotation_note':'候选任务；审核时须补足日期、证据组、事实及必要行为，未审核不能测正式效果。'}
            if category=='conversation':
                case['history']=[{'goal':'调查版本'+str(target['news_id'] if target else '待定位'),
                    'target_news_id':target['news_id'] if target else None,'filters':{'language':'schinese'},
                    'verified':False}]
                if i in (0,1):case['expectations']['filters']={'language':'schinese'}
                if i==1:case['expectations']['filters'].update(min_playtime_hours=100,min_playtime_exclusive=True,playtime_basis='at_review')
                if i==2:
                    case['goal']='继续上一轮的同一款游戏和目标更新，这次只看英文玩家反馈。'
                    case['history'][0]['appid']=appid
                    case['expectations']['filters']={'language':'english'}
            if target and category=='locate' and i==1:
                from gamelens.core.local_time import local_day
                day=local_day(target['published_at_timestamp'])
                case['goal']=f'定位{day}的更新公告并说明官方改动，保留原文来源。'
                case['oracle_plan']=[{'tool':'list_patches','arguments':{'appid':appid,'published_on':day}}]
                case['expectations']['target_news_id']=target['news_id']
            if category=='statistics' and game.get('reviews'):
                rows=game['reviews']
                if i==2:
                    a={'period':'all','language':'schinese','min_playtime_hours':100,'min_playtime_exclusive':True,'playtime_basis':'at_review'}
                    z={**a,'playtime_basis':'lifetime'}
                    case['expectations']['filters']={'language':'schinese','min_playtime_hours':100,'min_playtime_exclusive':True}
                    case['goal']='比较冻结快照中中文、评价时长超过100小时与当前累计时长超过100小时两组玩家的评论数量和推荐率。'
                    case['oracle_plan']=[{'tool':'compare_cohorts','arguments':{'appid':appid,'cohort_a':a,'cohort_b':z}},
                        {'tool':'verify_investigation','arguments':{'appid':appid}}]
                    case['expectations']['facts']=[{'path':[group,'stats','count'],
                        'value':sum(r['language']=='schinese' and r[field]>6000 for r in rows),'filters':{}}
                        for group,field in (('cohort_a','playtime_at_review_minutes'),('cohort_b','playtime_minutes'))]
                if i in (3,5):
                    start=min(r['created_at_timestamp'] for r in rows) if i==3 else max(r['created_at_timestamp'] for r in rows)+1
                    end=max(r['created_at_timestamp'] for r in rows)+1 if i==3 else start+86400
                    iso=lambda t:datetime.fromtimestamp(t,timezone.utc).isoformat()
                    f={'period':'all','start_at':iso(start),'end_at':iso(end)}
                    case['goal']=f"统计已冻结评论在{iso(start)}至{iso(end)}（结束不含）内的数量与推荐率；空缓存窗口不代表Steam实际没有反馈。"
                    case['expectations']['filters']=f
                    case['expectations']['facts']=[{'path':['stats','count'],'value':len(rows) if i==3 else 0,'filters':f}]
                    case['oracle_plan']=[{'tool':'query_reviews','arguments':{'appid':appid,'filters':f}},
                        {'tool':'verify_investigation','arguments':{'appid':appid}}]
                if i==4 and target:
                    from gamelens.core.local_time import local_day,shift_calendar_days
                    pivot=target['published_at_timestamp'];day=local_day(pivot)
                    case['goal']=f'比较{day}更新前后各7天的已采集评论好评率；用原文解释样本限制。'
                    case['oracle_plan']=[{'tool':'list_patches','arguments':{'appid':appid,'published_on':day}}]
                    case['expectations']['facts']=[];case['expectations']['filters']={}
                    case['expectations']['target_news_id']=target['news_id']
                    for direction,left,right in (('before',shift_calendar_days(pivot,-7),pivot),('after',pivot,shift_calendar_days(pivot,7))):
                        case['oracle_plan'].append({'tool':'query_reviews','arguments':{'appid':appid,'patch_news_id':target['news_id'],'window_days':7,'filters':{'period':direction}}})
                        case['expectations']['facts'].append({'path':['stats','count'],'value':sum(left<=r['created_at_timestamp']<right for r in rows),'filters':{'period':direction}})
                    case['oracle_plan'].append({'tool':'verify_investigation','arguments':{'appid':appid}})
            if category=='discussion' and discussion_id:
                case['goal']=f"公告ID {discussion_id}："+case['goal']
                snapshot=(next((n for n in news if n['news_id']==discussion_id),None))
                if snapshot:case['expectations']['announcement_title']=snapshot['title']
            investigations.append(_record(case))
    return retrieval,investigations


def prepare(project, output):
    project=Path(project);output=Path(output)
    if output.exists():raise ValueError('冻结目录已存在，不能覆盖；请选择新版本目录')
    output.mkdir(parents=True)
    corpus={};files={};mapping=[];missing=[];inventory=[]
    for appid in APPIDS:
        game={};corpus[appid]=game
        for kind,suffix in (('reviews','reviews'),('news','news')):
            source=project/'data/raw'/f'{appid}_{suffix}.json'
            if not source.is_file():missing.append(f'{appid}:{kind}');continue
            value=scrub(read(source));target=output/'corpus'/source.name
            if kind=='news':
                from gamelens.domain.models import SteamNewsItem
                from gamelens.tools.steam_news import retained_patch_candidate
                value=[SteamNewsItem.model_validate(n).model_copy(update={'author':'','is_patch_candidate':retained_patch_candidate(n)}).model_dump(mode='json') for n in value]
                value=sorted(value,key=lambda n:n['published_at_timestamp'],reverse=True)
            atomic_write_json(target,value);files[str(target.relative_to(output)).replace('\\','/')]=file_hash(target)
            game[kind]=value;mapping.append({'file':str(target.relative_to(output)),'original_path':str(source.resolve()),'original_sha256':file_hash(source)})
            inventory.append({'appid':appid,'kind':kind,'count':len(value),'source_file':source.name})
        comments=[]
        for source in sorted((project/'data/raw/announcement_comments'/str(appid)).glob('*.json')):
            value=scrub(read(source));comments+=value.get('comments',[])
            target=output/'corpus/announcement_comments'/str(appid)/source.name
            atomic_write_json(target,value);files[target.relative_to(output).as_posix()]=file_hash(target)
            mapping.append({'file':target.relative_to(output).as_posix(),'original_path':str(source.resolve()),'original_sha256':file_hash(source)})
            inventory.append({'appid':appid,'kind':'comments','news_id':value.get('news_id'),'coverage':value.get('coverage',{}),'fetched_at':value.get('fetched_at')})
        game['comments']=comments
    from gamelens.core.local_time import time_context
    clock=datetime.now(timezone.utc).isoformat()
    # These fixtures are derived from cached source records, not represented as
    # archived HTTP responses or evidence of a fresh network collection.
    fixtures={str(appid):{'provenance':'derived_from_frozen_cache','news':game.get('news',[])} for appid,game in corpus.items()}
    atomic_write_json(output/'fixtures.json',fixtures);files['fixtures.json']=file_hash(output/'fixtures.json')
    manifest={'version':VERSION,'created_at':clock,'clock':clock,'timezone':time_context(),
        'timezone_iana':'Asia/Shanghai','appids':list(APPIDS),'files':files,
        'dataset_sha256':digest(files),'missing_sources':missing,'inventory':inventory,
        'minimum_games':3,'response_provenance':'Derived fixtures; no live Steam collection during preparation.'}
    atomic_write_json(output/'manifest.json',manifest)
    atomic_write_json(output/'private/source-map.json',mapping)
    retrieval,tasks=candidates(corpus,clock)
    atomic_write_json(output/'retrieval.json',retrieval);atomic_write_json(output/'investigations.json',tasks)
    export_review_pack(output)
    return output


def load(root):
    root=Path(root);manifest=read(root/'manifest.json')
    if digest(manifest['files'])!=manifest['dataset_sha256']:raise ValueError('manifest hash mismatch')
    for relative,expected in manifest['files'].items():
        path=(root/relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or file_hash(path)!=expected:
            raise ValueError('冻结来源 hash mismatch: '+relative)
    corpus={}
    for appid in manifest['appids']:
        game={}
        for kind in ('reviews','news'):
            p=root/'corpus'/f'{appid}_{kind}.json';game[kind]=read(p) if p.exists() else []
        game['comments']=[c for p in (root/'corpus/announcement_comments'/str(appid)).glob('*.json') for c in read(p).get('comments',[])]
        corpus[appid]=game
    return {'root':root,'manifest':manifest,'corpus':corpus,'retrieval':read(root/'retrieval.json'),
            'investigations':read(root/'investigations.json')}


def migrate_timezone(source,output):
    """Create a new time-policy revision; never mutate old evidence or approvals.

    Approval transfers only when case content is unchanged and no naive date
    restriction or potential referenced-announcement day shift is detected.
    Uncertain cases return to pending review, never receive machine approval.
    """
    import shutil
    from gamelens.core.local_time import time_context, TIMEZONE_NAME
    source=Path(source).resolve();output=Path(output).resolve()
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('迁移必须使用独立且不存在的新目录，不能覆盖或嵌套旧评测包')
    b=load(source);validate(source)
    if b['manifest']['timezone'].get('basis')=='beijing':raise ValueError('该包已经使用北京时间，无需重复迁移')
    before={p.relative_to(source).as_posix():file_hash(p) for p in source.rglob('*') if p.is_file()}
    old=b['manifest']['timezone']
    offsets={old['standard_offset_seconds'],old['daylight_offset_seconds']}
    decisions=[]
    for group in ('retrieval','investigations'):
        for case in b[group]:
            reasons=[]
            def visit(value):
                if isinstance(value,dict):
                    for key,item in value.items():
                        if key in {'start_at','end_at'} and isinstance(item,str) and datetime.fromisoformat(item.replace('Z','+00:00')).tzinfo is None:
                            reasons.append('无时区边界的解释发生变化')
                        visit(item)
                elif isinstance(value,list):
                    for item in value:visit(item)
            visit({k:v for k,v in case.items() if k!='review'})
            goal=case.get('goal',case.get('query',''))
            from gamelens.core.query_contract import build_contract
            for text in [goal,*[h.get('goal','') for h in case.get('history',[])]]:
                window=build_contract(text,b['manifest']['clock'])['calendar_window']
                if window.get('kind') in {'relative_calendar','calendar_month','calendar_days'}:
                    reasons.append('自然语言日历条件须按北京时间重新核对')
            expects=case.get('expectations',{})
            ids={str(expects.get(k)) for k in ('target_news_id','discussion_news_id') if expects.get(k)}
            ids.update(str(h['target_news_id']) for h in case.get('history',[]) if h.get('target_news_id'))
            if re.search(r'日期|发布|更新|date|recent|latest',goal,re.I):
                ids.update(str(e['source_id']) for e in case.get('labels',expects.get('evidence',[])) if e.get('kind')=='announcement')
                for item in b['corpus'].get(case['appid'],{}).get('news',[]):
                    if str(item['news_id']) not in ids:continue
                    moment=datetime.fromtimestamp(item['published_at_timestamp'],timezone.utc)
                    day=moment.astimezone(timezone(timedelta(hours=8))).date()
                    if any(moment.astimezone(timezone(timedelta(seconds=offset))).date()!=day for offset in offsets):
                        reasons.append('相关公告可能跨北京时间日期边界：'+str(item['news_id']))
            previous=dict(case['review'])
            if reasons and previous['status']=='reviewed':
                case['review']={**previous,'status':'pending_review','notes':'时区迁移待核对：'+'；'.join(sorted(set(reasons)))}
            decisions.append({'case_id':case['id'],'case_sha256':case_hash(case),
                'previous_review':previous,'status':case['review']['status'],'reasons':sorted(set(reasons))})
    shutil.copytree(source,output)
    clock=b['manifest']['clock']
    manifest={**b['manifest'],'version':b['manifest']['version']+'-beijing-v1',
        'timezone':time_context(),'timezone_iana':TIMEZONE_NAME,
        'migration':{'source_version':b['manifest']['version'],'source_manifest_sha256':file_hash(source/'manifest.json'),
                     'policy':'new_revision_raw_instants_and_case_content_unchanged'}}
    # Keep the frozen clock instant, not the migration machine's current date.
    from gamelens.core.local_time import local_datetime
    manifest['timezone']['current_offset']=local_datetime(clock).strftime('%z')
    atomic_write_json(output/'manifest.json',manifest)
    for group in ('retrieval','investigations'):atomic_write_json(output/(group+'.json'),b[group])
    atomic_write_json(output/'timezone-migration.json',{'old_timezone':old,'new_timezone':manifest['timezone'],
        'source_manifest_sha256':file_hash(source/'manifest.json'),'case_decisions':decisions})
    validate(output);export_review_pack(output)
    if before!={p.relative_to(source).as_posix():file_hash(p) for p in source.rglob('*') if p.is_file()}:
        raise ValueError('迁移过程中旧包发生变化')
    return output


def validate(root,*,_bundle=None):
    b=load(root) if _bundle is None else _bundle
    groups={};ids=set();reviewed={'retrieval':0,'investigations':0}
    for name,wanted,splits in (('retrieval',40,{'development':24,'holdout':16}),('investigations',30,{'development':20,'holdout':10})):
        cases=b[name]
        if len(cases)!=wanted or any(sum(c['split']==s for c in cases)!=n for s,n in splits.items()):raise ValueError('题目数量/分组不匹配')
        for c in cases:
            if not re.fullmatch(r'[a-z0-9_-]+',c['id']) or c['id'] in ids:raise ValueError('重复/非法题目ID')
            ids.add(c['id']);groups.setdefault(c['group_id'],set()).add(c['split'])
            review=c['review']
            if review['status'] not in {'reviewed','pending_review','rejected'}:raise ValueError('审核状态无效')
            if review['status']=='reviewed':
                if c.get('category')=='conversation' and any(h.get('appid',c['appid'])!=c['appid'] for h in c.get('history',[])):
                    raise ValueError('同一会话不能切换游戏；此题不符合会话交互约束')
                if not all(str(review.get(k,'')).strip() for k in ('reviewer','reviewed_at','notes')):raise ValueError('审核身份/时间/理由缺失')
                datetime.fromisoformat(review['reviewed_at'].replace('Z','+00:00'))
                if review.get('case_sha256')!=case_hash(c):raise ValueError('审核后题目已变化')
                if c.get('blocking'):raise ValueError('阻塞题不能批准为正式题')
                if name=='retrieval' and not any(l['relevance']>0 for l in c.get('labels',[])) and c.get('expected_no_answer') is not True:raise ValueError('无正相关标签须人工确认无答案')
                reviewed[name]+=1
            source=b['corpus'].get(c['appid'],{})
            for label in c.get('labels',c.get('expectations',{}).get('evidence',[])):
                kind=label['kind'];key='review_id' if kind=='review' else 'news_id'
                values=source.get('reviews' if kind=='review' else 'news',[])
                if label['source_id'] not in {str(r[key]) for r in values}:raise ValueError('证据标签不在绑定语料中')
                if type(label.get('relevance')) is not int or label['relevance'] not in range(4):raise ValueError('相关性须0..3')
                if label.get('role') not in {'support','counterexample','partial'}:raise ValueError('证据角色无效')
    if any(len(s)>1 for s in groups.values()):raise ValueError('同一场景跨开发/留出分组')
    active={name:sum(c['review']['status']!='rejected' for c in b[name]) for name in reviewed}
    return {'version':b['manifest']['version'],'dataset_sha256':b['manifest']['dataset_sha256'],
        'candidate_counts':{'retrieval':40,'investigations':30},'formal_retrieval_count':reviewed['retrieval'],
        'active_counts':active,'withdrawn_cases':[c['id'] for name in reviewed for c in b[name] if c['review']['status']=='rejected'],
        'formal_investigation_count':reviewed['investigations'],'missing_sources':b['manifest']['missing_sources'],
        'blocked_cases':[c['id'] for name in reviewed for c in b[name] if c.get('blocking')],
        'formal_ready':reviewed==active and any(active.values()) and not b['manifest']['missing_sources']}


def export_review_pack(root):
    from gamelens.evaluation.benchmark_review import render_card, audit_cards
    b=load(root)
    for name in ('retrieval','investigations'):
        for c in b[name]:
            text,_=render_card(b,c,name)
            folder=root/'review_pack'
            if c['review']['status']=='rejected':
                folder=folder/'withdrawn'
                old=root/'review_pack'/name/(c['id']+'.md')
                target=folder/name/(c['id']+'.md')
                if old.exists():
                    target.parent.mkdir(parents=True,exist_ok=True)
                    old.replace(target)
            atomic_write_text(folder/name/(c['id']+'.md'),text)
    atomic_write_json(root/'review_pack/approval_template.json',{'dataset_sha256':b['manifest']['dataset_sha256'],
        'benchmark_version':b['manifest']['version'],'manifest_sha256':file_hash(root/'manifest.json'),
        'reviews':[{'case_id':c['id'],'case_sha256':case_hash(c),'status':'pending_review','reviewer':'','reviewed_at':'','notes':''}
                   for name in ('retrieval','investigations') for c in b[name] if c['review']['status']!='rejected']})
    audit=audit_cards(b,root)
    atomic_write_json(root/'review_pack/material_check.json',audit)
    status={m['case_id']:m for m in audit['cards']}
    approved=sum(c['review']['status']=='reviewed' for name in ('retrieval','investigations') for c in b[name])
    pending=[c for name in ('investigations','retrieval') for c in b[name]
        if c['review']['status']=='pending_review' and not status[c['id']]['gaps']]
    lines=['# GameLens Benchmark：从这里开始审核','',
        '你只需要判断问题、原文和答案要点；不需要看哈希、改JSON或运行代码。','',
        '每题的“标准答案”是待确认要点，不是系统已生成的实际答案。请先看完整调查题，再看检索题。','',
        f"当前 {audit['count']} 道，已撤下 {len(audit['withdrawn'])} 道；已审核通过 {approved} 道，本轮待你审核 {len(pending)} 道，材料待补齐 {len(audit['needs_material'])} 道。未确认的题不会计入正式成绩。",'',
        '**“材料待补齐”是我还没有把资料、对话背景或异常回放准备完整，不是让你去补数据或修程序。这些题先不要审核，由我补齐后再交给你。**','',
        '[本轮只需审核这些题](pending_review.md)。已通过题不必重复审核。','',
        '回复方式：`题目ID：通过`、`题目ID：修改，原因……` 或 `题目ID：材料不足`。我负责记录审核。','',
        f"## 完整调查题（{sum(c['review']['status']!='rejected' for c in b['investigations'])}道）",'']
    for name in ('investigations','retrieval'):
        if name=='retrieval':lines += ['','## 检索题（40道）','']
        for c in b[name]:
            if c['review']['status']=='rejected':continue
            if status[c['id']]['gaps']:continue
            state='已审核通过' if c['review']['status']=='reviewed' else '待你审核'
            lines.append(f"- [{c['id']}]({name}/{c['id']}.md)｜{state}｜{c.get('goal',c.get('query',''))}")
    lines += ['','## 待补齐项（不是交给你修程序）','']
    for row in audit['needs_material']:
        lines += ['- '+row['case_id']+'：'+'；'.join(row['gaps'])]
    if audit['withdrawn']:
        lines += ['','## 已撤下（不参与审核与评测）','']
        lines += ['- '+c['id']+'：'+c['review'].get('notes','用户撤下') for name in ('retrieval','investigations') for c in b[name] if c['review']['status']=='rejected']
    missing=b['manifest']['missing_sources']
    lines += ['',f"资料清单当前列出 {len(missing)} 种缺失来源"+('：'+'、'.join(missing) if missing else '；原文已经齐备，新标注仍需人工审核。'),
        '打开留出题后，你已看到题目和证据；以后不得宣称这些题对你仍是未见过的盲测。','']
    atomic_write_text(root/'review_pack/README.md','\n'.join(lines))
    batch=['# 本轮待你审核','',f'只需要审核下面 {len(pending)} 道；此前通过的 {approved} 道不必重复看。', '',
        '每题核对：题目与背景是否明确、原文是否相关、预拟答案与限制是否成立。修订稿没有自动批准。', '']
    for c in pending:
        folder='investigations' if c['id'].startswith('task-') else 'retrieval'
        batch.append(f"- [{c['id']}]({folder}/{c['id']}.md)：{c.get('goal',c.get('query',''))}")
    batch += ['',f'另外 {len(audit["needs_material"])} 道仍由我准备，不要求你补数据。审核材料导出本身不联网或调用付费模型；实际来源采集与模型试测历史另有记录。','']
    atomic_write_text(root/'review_pack/pending_review.md','\n'.join(batch))


def import_reviews(root,path):
    b=load(root);payload=read(path)
    if payload['dataset_sha256']!=b['manifest']['dataset_sha256']:raise ValueError('审核资料哈希不匹配')
    lookup={c['id']:c for n in ('retrieval','investigations') for c in b[n]};seen=set()
    for row in payload['reviews']:
        if row['case_id'] in seen or row['case_id'] not in lookup:raise ValueError('审核ID重复/不存在')
        seen.add(row['case_id']);c=lookup[row['case_id']]
        if row['case_sha256']!=case_hash(c):raise ValueError('审核题目哈希不匹配')
        c['review']={k:v for k,v in row.items() if k!='case_id'}
    if b['manifest']['timezone'].get('basis')=='beijing':
        if payload.get('benchmark_version')!=b['manifest']['version'] or payload.get('manifest_sha256')!=file_hash(Path(root)/'manifest.json'):
            raise ValueError('审批版本/时间标准绑定不匹配，请使用本包新导出的approval_template；旧manifest审批不可重放')
    # Validate the proposed result in memory using the same checks before writing.
    for c in lookup.values():
        r=c['review']
        if r['status'] not in {'reviewed','pending_review','rejected'}:raise ValueError('审核状态无效')
        if r['status']=='reviewed' and (c.get('blocking') or not all(str(r.get(k,'')).strip() for k in ('reviewer','reviewed_at','notes'))):
            raise ValueError('阻塞或审核字段缺失')
        if r['status']=='reviewed':datetime.fromisoformat(r['reviewed_at'].replace('Z','+00:00'))
        if r['status']=='reviewed' and 'labels' in c and not any(l['relevance']>0 for l in c['labels']) and c.get('expected_no_answer') is not True:raise ValueError('须人工确认无答案')
    result=validate(root,_bundle=b)
    for name in ('retrieval','investigations'):atomic_write_json(Path(root)/(name+'.json'),b[name])
    return result
