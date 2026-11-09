"""Acceptance tests for benchmark identity, review gates and evidence integrity."""
import json
from pathlib import Path

import pytest

from gamelens.evaluation.benchmark_data import prepare, load, validate, case_hash, digest
from gamelens.evaluation.benchmark_score import score_run, compare, ranking_score


@pytest.fixture
def bundle(tmp_path):
    raw=tmp_path/'project/data/raw';raw.mkdir(parents=True)
    base=Path(__file__).resolve().parents[1]/'evals/fixtures'
    reviews=json.loads((base/'730_reviews.json').read_text(encoding='utf-8'))
    news=json.loads((base/'730_patch_candidates.json').read_text(encoding='utf-8'))
    for appid in (1623730,251570,730):
        (raw/f'{appid}_reviews.json').write_text(json.dumps(reviews),encoding='utf-8')
        (raw/f'{appid}_news.json').write_text(json.dumps([{**n,'appid':appid,
            'published_at':'2026-09-15T03:03:47Z','published_at_timestamp':1789441427,
            'title':'Patch '+n['news_id'],'feed_name':'steam_community_announcements',
            'feed_label':'Community Announcements'} for n in news]),encoding='utf-8')
    return prepare(tmp_path/'project',tmp_path/'bundle')


def test_seventy_pending_cases_and_grouped_split(bundle):
    b=load(bundle)
    assert len(b['retrieval'])==40 and len(b['investigations'])==30
    for name,counts in (('retrieval',{'development':24,'holdout':16}),('investigations',{'development':20,'holdout':10})):
        cases=b[name]
        assert {s:sum(c['split']==s for c in cases) for s in counts}==counts
        assert all(c['review']['status']=='pending_review' for c in cases)
        groups={}
        for c in cases:groups.setdefault(c['group_id'],set()).add(c['split'])
        assert all(len(s)==1 for s in groups.values())
    assert validate(bundle)['formal_ready'] is False


@pytest.mark.parametrize('ready',[True,False])
def test_frozen_rag_approval_uses_configured_encoder(bundle,tmp_path,ready):
    from gamelens.evaluation.benchmark_runner import materialize,registry_for,state_for
    from gamelens.rag.embedding import MODEL_REVISION
    b=load(bundle);case=b['investigations'][0];root=tmp_path/'scratch'
    model_root=tmp_path/'shared-model'
    if ready:
        cache=model_root/'data/rag/models';cache.mkdir(parents=True)
        (cache/f'{MODEL_REVISION}.ready').touch()
    materialize(b,root,case)
    registry=registry_for(b,root,case,model_root=model_root)
    build=next(t for t in registry.tools if t.name=='build_game_index')
    args=build.args_model(appid=case['appid'])
    prompt=build.approval_policy(args,state_for(b,case))
    assert (prompt is None) if ready else ('下载' in prompt)


def test_frozen_registry_uses_scoped_alternatives_without_removing_production_tools(bundle,tmp_path):
    from gamelens.evaluation.benchmark_runner import registry_for
    from gamelens.tools.registry import SteamInvestigationTools
    b=load(bundle);case=b['investigations'][0]
    frozen={t.name for t in registry_for(b,tmp_path,case).tools}
    production={t.name:t for t in SteamInvestigationTools(tmp_path).registry().tools}
    for name in ('prepare_reviews','prepare_announcement_comments'):
        assert name not in frozen
        assert production[name].approval_policy is not None
    assert {'ensure_review_coverage','query_announcement_comments'}<=frozen


def test_new_benchmark_freezes_beijing_policy(bundle):
    manifest=load(bundle)['manifest']
    assert manifest['timezone_iana']=='Asia/Shanghai'
    assert manifest['timezone']['basis']=='beijing'


def test_review_index_reports_actual_source_inventory_not_old_missing_claim(bundle):
    b=load(bundle)
    readme=(bundle/'review_pack/README.md').read_text(encoding='utf-8')
    pending=(bundle/'review_pack/pending_review.md').read_text(encoding='utf-8')
    assert f"资料清单当前列出 {len(b['manifest']['missing_sources'])} 种缺失来源" in readme
    assert '审核材料导出本身不联网' in pending


@pytest.mark.system_local_time
def test_review_card_displays_frozen_clock_in_beijing_not_raw_utc(bundle):
    from gamelens.evaluation.benchmark_review import render_card
    b=load(bundle);b['manifest']['clock']='2026-10-14T13:17:46.102606+00:00'
    card,_=render_card(b,b['retrieval'][0],'retrieval')
    line=next(line for line in card.splitlines() if line.startswith('资料和评测时钟固定在：'))
    assert '2026-10-14T21:17:46+08:00' in line


def test_wrapped_frozen_tools_support_agent_coverage_planning(bundle,tmp_path):
    from gamelens.evaluation.benchmark_runner import materialize,registry_for,state_for,frozen_environment
    from gamelens.core.runtime import InvestigationAgent,PlannerDecision
    b=load(bundle);case=b['investigations'][0];root=tmp_path/'wrapped-agent'
    materialize(b,root,case);registry=registry_for(b,root,case)
    class ReachedModelBoundary(Exception):pass
    class Planner:
        calls=0
        def decide(self,state,tools):
            self.calls+=1
            if self.calls==1:return PlannerDecision(action='inspect_data',reason='Inspect frozen sources',arguments={'appid':case['appid']})
            assert any(o['tool']=='inspect_data' for o in state['observations'])
            raise ReachedModelBoundary()
    # Exercise the real graph's deterministic preflight and coverage interception
    # before stopping at the external model boundary. Old wrappers raise
    # AttributeError before reaching this boundary.
    state=state_for(b,case)
    with frozen_environment(b):
        with pytest.raises(ReachedModelBoundary):
            InvestigationAgent(Planner(),registry).invoke(state)


def test_missing_review_cache_fixture_removes_only_target_and_preserves_coverage_context(bundle,tmp_path):
    from gamelens.evaluation.benchmark_runner import materialize,state_for
    b=load(bundle);case=json.loads(json.dumps(b['investigations'][0]));case['appid']=730
    windows=[{'start_at':'2027-01-12T00:00:00Z','end_at':'2027-01-18T00:00:00Z','language':'all'}]
    case['initial_conditions']={'reviews_cache_absent':True,'coverage_followup':{'windows':windows}}
    root=tmp_path/'missing-reviews';materialize(b,root,case)
    assert not (root/'data/raw/730_reviews.json').exists()
    assert (root/'data/raw/730_news.json').exists()
    assert (root/'data/raw/1623730_reviews.json').exists()
    assert (bundle/'corpus/730_reviews.json').exists()
    state=state_for(b,case)
    assert state['coverage_followup']['windows']==windows
    state['coverage_followup']['windows'][0]['language']='english'
    assert case['initial_conditions']['coverage_followup']['windows'][0]['language']=='all'


def test_oracle_workspace_is_not_nested_under_long_report_directory(bundle,tmp_path):
    from gamelens.evaluation.benchmark_runner import run_oracle
    output=tmp_path/('report-'+'x'*45)
    result=run_oracle(bundle,output,limit=1)
    case_id=result['results'][0]['case_id']
    saved=json.loads((output/'runs'/f'{case_id}.json').read_text(encoding='utf-8'))
    sources=[o['data']['source']['path'] for o in saved['observations'] if o.get('data',{}).get('source',{}).get('path')]
    assert sources
    assert all(not Path(p).is_relative_to(output) for p in sources)


def test_oracle_accepts_restored_immutable_review_snapshot_not_raw_overwrite(bundle,tmp_path):
    from datetime import datetime,timezone
    from gamelens.evaluation.benchmark_runner import run_oracle
    b=load(bundle);records=b['corpus'][730]['reviews']
    start=min(r['created_at_timestamp'] for r in records)-1
    end=max(r['created_at_timestamp'] for r in records)+1
    iso=lambda t:datetime.fromtimestamp(t,timezone.utc).isoformat()
    manifest=b['manifest'];manifest['clock']=iso(end+86400)
    (bundle/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    case=next(c for c in b['investigations'] if c['id']=='task-reliability-2')
    for c in b['investigations']:c['review']['status']='rejected'
    case['review']['status']='pending_review';case['blocking']=[]
    case['goal']=f'统计提供的已采集评论在{iso(start)}至{iso(end)}的数量。'
    filters={'period':'all','start_at':iso(start),'end_at':iso(end)}
    windows=[{'start_at':iso(start),'end_at':iso(end),'language':'all'}]
    case['initial_conditions']={'reviews_cache_absent':True,'coverage_followup':{'windows':windows}}
    case['oracle_plan']=[{'tool':'ensure_review_coverage','arguments':{'appid':730,'windows':windows}},
        {'tool':'query_reviews','arguments':{'appid':730,'filters':filters}},
        {'tool':'verify_investigation','arguments':{'appid':730}}]
    case['expectations']['facts']=[{'path':['stats','count'],'value':len(records),'filters':filters}]
    (bundle/'investigations.json').write_text(json.dumps(b['investigations']),encoding='utf-8')
    result=run_oracle(bundle,tmp_path/'snapshot-oracle')
    row=result['results'][0]
    assert row['tool_oracle_passed'],row
    assert row['fixture_checks']['reviews_cache_absent_at_start']
    assert row['fixture_checks']['reviews_cache_restored']


def test_generation_recovery_replay_uses_checkpoint_and_preserves_original_evidence(bundle,tmp_path):
    from gamelens.evaluation import benchmark_runner as runner
    b=load(bundle);case=json.loads(json.dumps(b['investigations'][0]));case['appid']=730
    case['goal']='读取商店评论r0原文，说明该条评论的样本边界。'
    root=tmp_path/'generation-replay';runner.materialize(b,root,case)
    registry=runner.registry_for(b,root,case);state=runner.state_for(b,case)
    for step,(name,args) in enumerate([('get_review',{'appid':730,'review_id':'r0'}),('verify_investigation',{'appid':730})],1):
        state['step_count']=step-1;result=registry.execute(name,args,state)
        assert result.ok,result.summary
        state['observations'].append({'step':step,'tool':name,'arguments':args,**result.model_dump(mode='json')})
        state['artifacts'].update(result.artifacts)
    assert hasattr(runner,'run_generation_recovery_fixture')
    final,checks=runner.run_generation_recovery_fixture(registry,state,root)
    assert all(checks.values()),checks
    assert final['status']=='completed'
    assert final['generation_recoveries']==1
    assert len([o for o in final['observations'] if o['tool']=='get_review'])==1
    assert final['final_citations'][0]['review_id']=='r0'


def test_pending_human_failure_is_not_reported_as_human_scored_or_accuracy():
    from gamelens.evaluation.benchmark_score import summary
    rows=[{'case_id':'x','formal_eligible':True,'automatic_passed':False,'accepted':False,
           'human_review':{'status':'pending','passed':None}}]
    result=summary(rows)
    assert result['human_scored_cases']==0
    assert result['task_success_rate'] is None
    assert result['automatic_check_pass_rate']==0
    rows[0]['human_review']={'status':'reviewed','passed':False}
    assert summary(rows)['human_scored_cases']==1
    assert summary(rows)['task_success_rate']==0


def test_gold_all_language_archive_is_not_an_empty_population():
    from gamelens.evaluation.benchmark_score import eligible_records
    records=[{'review_id':rid,'language':language,'created_at_timestamp':100,
        'playtime_minutes':60,'playtime_at_review_minutes':60,'is_positive':True}
        for rid,language in [('en','english'),('zh','schinese')]]
    assert [r['review_id'] for r in eligible_records({'reviews':records},{'language':'all'})]==['en','zh']
    assert [r['review_id'] for r in eligible_records({'reviews':records},{'language':'english'})]==['en']


def test_comparison_refuses_automatically_failed_but_unreviewed_answers():
    base={'identity':{'dataset_sha256':'same','case_hashes':{'x':'1'},'environment':{},'configuration':{}},
          'mode':'agent','results':[{'case_id':'x','formal_eligible':True,'automatic_passed':False,
          'accepted':False,'human_review':{'status':'pending','passed':None}}]}
    with pytest.raises(ValueError,match='审核'):compare(base,json.loads(json.dumps(base)))


def test_timezone_migration_preserves_sources_and_approval_but_changes_identity(bundle,tmp_path):
    from gamelens.evaluation.benchmark_data import migrate_timezone
    from gamelens.core.local_time import time_context
    original=load(bundle)
    # Emulate a historical bundle without altering its sources or case content.
    manifest=original['manifest']
    manifest['timezone']={'basis':'system_local','system_zone':'AUS Eastern Standard Time','names':['AEST','AEDT'],
        'standard_offset_seconds':36000,'daylight_offset_seconds':39600}
    manifest['timezone_iana']='Australia/Sydney'
    (bundle/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    c=original['retrieval'][0]
    c['review']={'status':'reviewed','reviewer':'human','reviewed_at':'2026-10-25','notes':'original checked','case_sha256':case_hash(c)}
    (bundle/'retrieval.json').write_text(json.dumps(original['retrieval']),encoding='utf-8')
    before={p.relative_to(bundle).as_posix():p.read_bytes() for p in bundle.rglob('*') if p.is_file()}
    destination=tmp_path/'beijing'
    migrate_timezone(bundle,destination)
    after=load(destination)
    assert before=={p.relative_to(bundle).as_posix():p.read_bytes() for p in bundle.rglob('*') if p.is_file()}
    assert after['manifest']['timezone_iana']=='Asia/Shanghai'
    assert after['manifest']['timezone']['basis']=='beijing'
    assert after['manifest']['version']!=manifest['version']
    assert after['manifest']['clock']==manifest['clock']
    assert after['manifest']['dataset_sha256']==manifest['dataset_sha256']
    assert after['retrieval']==original['retrieval']
    assert (destination/'timezone-migration.json').is_file()
    assert all((destination/p).read_bytes()==data for p,data in before.items() if p.startswith('corpus/'))
    with pytest.raises(ValueError):migrate_timezone(bundle,destination)


@pytest.mark.parametrize('goal',['这周的英文商店评论好评率','过去7天的中文评论数量','最近七天的评论数量'])
def test_timezone_migration_demotes_relative_calendar_and_refuses_old_approval(bundle,tmp_path,goal):
    from gamelens.evaluation.benchmark_data import migrate_timezone,import_reviews,file_hash
    b=load(bundle);c=b['retrieval'][0];c['query']=goal
    c['review']={'status':'reviewed','reviewer':'human','reviewed_at':'2026-10-25','notes':'old calendar checked','case_sha256':case_hash(c)}
    (bundle/'retrieval.json').write_text(json.dumps(b['retrieval']),encoding='utf-8')
    manifest=b['manifest'];manifest['timezone']={'basis':'system_local','system_zone':'AUS Eastern Standard Time','names':['AEST','AEDT'],
        'standard_offset_seconds':36000,'daylight_offset_seconds':39600};manifest['timezone_iana']='Australia/Sydney'
    (bundle/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    new=tmp_path/'calendar-beijing';migrate_timezone(bundle,new)
    revised=load(new);case=revised['retrieval'][0]
    assert case['review']['status']=='pending_review'
    old_payload={'dataset_sha256':manifest['dataset_sha256'],'reviews':[{'case_id':c['id'],**c['review']}]}
    approval=tmp_path/'old-approval.json';approval.write_text(json.dumps(old_payload),encoding='utf-8')
    with pytest.raises(ValueError,match='版本|时间标准|manifest'):import_reviews(new,approval)
    assert load(new)['retrieval'][0]['review']['status']=='pending_review'
    fresh={**old_payload,'benchmark_version':revised['manifest']['version'],'manifest_sha256':file_hash(new/'manifest.json')}
    fresh['reviews'][0]['notes']='Human checked Beijing calendar boundaries'
    approval.write_text(json.dumps(fresh),encoding='utf-8')
    import_reviews(new,approval)
    assert load(new)['retrieval'][0]['review']['status']=='reviewed'


def test_language_followup_candidate_never_switches_game_in_same_session(bundle):
    b=load(bundle); c=next(c for c in b['investigations'] if c['id']=='task-conversation-3')
    assert all(h.get('appid',c['appid'])==c['appid'] for h in c['history'])
    assert '251570' not in c['goal']
    assert c['expectations']['filters']['language']=='english'


def test_human_approval_cannot_accept_cross_game_single_session_case(bundle):
    b=load(bundle); c=next(c for c in b['investigations'] if c['id']=='task-conversation-3')
    c['history'][0]['appid']=251570
    c['review']={'status':'reviewed','reviewer':'human','reviewed_at':'2026-10-25',
        'notes':'checked','case_sha256':case_hash(c)}
    (bundle/'investigations.json').write_text(json.dumps(b['investigations']),encoding='utf-8')
    with pytest.raises(ValueError,match='同一会话不能切换游戏'):validate(bundle)


def test_withdrawn_case_is_archived_and_not_run_by_oracle(bundle,tmp_path):
    from gamelens.evaluation.benchmark_data import export_review_pack
    from gamelens.evaluation.benchmark_runner import run_oracle
    b=load(bundle); c=b['investigations'][0]
    c['review']['status']='rejected'; c['review']['notes']='用户撤下'
    (bundle/'investigations.json').write_text(json.dumps(b['investigations']),encoding='utf-8')
    export_review_pack(bundle)
    assert not (bundle/'review_pack/investigations'/f"{c['id']}.md").exists()
    assert (bundle/'review_pack/withdrawn/investigations'/f"{c['id']}.md").exists()
    readme=(bundle/'review_pack/README.md').read_text(encoding='utf-8')
    assert f"(investigations/{c['id']}.md)" not in readme
    report=run_oracle(bundle,tmp_path/'oracle-withdrawn',limit=1)
    assert all(r['case_id']!=c['id'] for r in report['results'])


def test_hash_change_and_fake_review_are_rejected(bundle):
    b=load(bundle);c=b['retrieval'][0]
    c['review']={'status':'reviewed','reviewer':'','reviewed_at':'','notes':''}
    (bundle/'retrieval.json').write_text(json.dumps(b['retrieval']),encoding='utf-8')
    with pytest.raises(ValueError):validate(bundle)
    source=next((bundle/'corpus').glob('*reviews.json'))
    source.write_text('[]',encoding='utf-8')
    with pytest.raises(ValueError,match='hash|哈希'):load(bundle)


def test_unreviewed_case_has_no_formal_score(bundle):
    b=load(bundle);c=b['investigations'][0]
    result=score_run(b,c,{'goal':c['goal'],'appid':c['appid'],'status':'completed','final_answer':'ok'})
    assert result['accepted'] is None and result['formal_eligible'] is False


def test_false_quote_and_wrong_player_scope_fail(bundle):
    b=load(bundle);c=b['investigations'][5]
    c['review']={'status':'reviewed','reviewer':'human','reviewed_at':'2026-10-14T00:00:00Z','notes':'checked'}
    c['expectations']['filters']={'language':'schinese','min_playtime_hours':100,'playtime_basis':'at_review'}
    run={'appid':c['appid'],'goal':c['goal'],'status':'completed','final_answer':'调查结果',
        'observations':[{'step':1,'tool':'query_reviews','ok':True,'data':{'filters':{'language':'english'},'reviews':[]}}],
        'final_citations':[{'source_step':1,'review_id':'absent','quote':'虚构原文'}]}
    run['evaluation']={'case_id':c['id'],'case_sha256':case_hash(c),'dataset_sha256':b['manifest']['dataset_sha256']}
    result=score_run(b,c,run)
    assert not result['automatic_passed']
    assert 'citation_integrity' in result['failed_checks'] and 'player_scope' in result['failed_checks']


def test_grade_zero_is_not_relevant():
    c={'labels':[{'source_id':'bad','kind':'review','relevance':0,'role':'partial'},
                 {'source_id':'good','kind':'review','relevance':3,'role':'counterexample'}]}
    result=ranking_score(c,[('review','bad'),('review','good')])
    assert result['recall_at_5']==1 and result['mrr_at_10']==0.5
    assert result['counterexample_coverage_at_10']==1


def test_comparison_refuses_mismatched_data_and_partial_case_coverage():
    base={'identity':{'dataset_sha256':'a','case_hashes':{'x':'1'},'environment':{},'configuration':{'split':'holdout','seed':0}},
          'mode':'agent','results':[{'case_id':'x','accepted':True,'automatic_passed':True,'formal_eligible':True}]}
    other=json.loads(json.dumps(base));other['identity']['dataset_sha256']='b'
    with pytest.raises(ValueError):compare(base,other)


def test_comparison_refuses_changed_benchmark_time_revision():
    base={'identity':{'benchmark_version':'v1-sydney','manifest_sha256':'old',
          'dataset_sha256':'same','case_hashes':{'x':'1'},'environment':{},'configuration':{}},
          'mode':'agent','results':[{'case_id':'x','accepted':True,'automatic_passed':True,'formal_eligible':True}]}
    newer=json.loads(json.dumps(base));newer['identity'].update(benchmark_version='v1-beijing',manifest_sha256='new')
    with pytest.raises(ValueError,match='不可比'):compare(base,newer)
    other=json.loads(json.dumps(base));other['results']=[]
    with pytest.raises(ValueError):compare(base,other)


def test_manifest_reports_missing_third_game(tmp_path):
    root=prepare(tmp_path/'empty-project',tmp_path/'out')
    result=validate(root)
    assert result['formal_ready'] is False
    assert len(result['missing_sources'])>=3
    assert result['formal_investigation_count']==0


def test_frozen_acquisition_really_fills_missing_cache_and_never_reads_network(bundle,tmp_path):
    from gamelens.evaluation.benchmark_runner import materialize,registry_for,state_for,frozen_environment
    from gamelens.tools.registry import PreparePatchesArgs
    b=load(bundle);case=b['investigations'][0];root=tmp_path/'task';materialize(b,root)
    news=root/'data/raw'/f"{case['appid']}_news.json"
    news.unlink();(root/'data/raw'/f"{case['appid']}_patch_candidates.json").unlink()
    registry=registry_for(b,root,case);tool=next(t for t in registry.tools if t.name=='prepare_patches')
    result=tool.handler(PreparePatchesArgs(appid=case['appid'],check_latest=True),state_for(b,case))
    assert result.ok and news.is_file() and result.data['fetched_count']>0
    import httpx
    with frozen_environment(b):
        with pytest.raises(RuntimeError,match='禁止真实'):httpx.Client().get('https://example.com')


def test_case_initial_conditions_remove_only_isolated_target_news_cache(bundle,tmp_path):
    from gamelens.evaluation.benchmark_runner import materialize
    b=load(bundle); c=b['investigations'][0]
    c['initial_conditions']={'news_cache_absent':True}
    original=(bundle/'corpus/1623730_news.json').read_bytes()
    root=tmp_path/'isolated'; materialize(b,root,c)
    assert not (root/'data/raw/1623730_news.json').exists()
    assert not (root/'data/raw/1623730_patch_candidates.json').exists()
    assert (root/'data/raw/251570_news.json').exists()
    assert (bundle/'corpus/1623730_news.json').read_bytes()==original


def test_expected_timeout_oracle_requires_exact_error_not_any_failure():
    from gamelens.evaluation.benchmark_runner import oracle_step_passed
    expected={'expected_result':{'ok':False,'error_code':'request_timeout'}}
    assert oracle_step_passed(expected,{'ok':False,'failure':{'error_code':'request_timeout'}})
    assert not oracle_step_passed(expected,{'ok':False,'failure':{'error_code':'permission_denied'}})
    assert not oracle_step_passed(expected,{'ok':True,'data':{}})
    assert not oracle_step_passed({'expected_result':{'ok':False}},{'ok':False})


def test_oracle_timeout_preserves_cache_and_is_scenario_success_not_agent_success(bundle,tmp_path):
    from gamelens.evaluation.benchmark_runner import run_oracle
    b=load(bundle); c=b['investigations'][0]
    c['response_scenario']='timeout'
    c['oracle_plan']=[{'tool':'prepare_patches','arguments':{'appid':1623730,'check_latest':True},
                      'expected_result':{'ok':False,'error_code':'request_timeout'}}]
    c['oracle_assertions']={'news_cache_preserved':True}
    c['expectations']['facts']=[]
    (bundle/'investigations.json').write_text(json.dumps(b['investigations']),encoding='utf-8')
    result=run_oracle(bundle,tmp_path/'timeout-oracle',limit=1)
    row=result['results'][0]
    assert row['tool_oracle_passed'] is True
    assert row['fixture_checks']['news_cache_preserved'] is True
    assert result['summary']['task_success_rate'] is None


def test_review_export_separates_already_approved_from_next_review_batch(bundle):
    from gamelens.evaluation.benchmark_data import export_review_pack
    b=load(bundle); c=b['investigations'][0]
    c['review']['status']='reviewed'
    (bundle/'investigations.json').write_text(json.dumps(b['investigations']),encoding='utf-8')
    export_review_pack(bundle)
    index=(bundle/'review_pack/README.md').read_text(encoding='utf-8')
    assert '已审核通过 1 道' in index
    batch=(bundle/'review_pack/pending_review.md').read_text(encoding='utf-8')
    assert c['id'] not in batch
    assert 'task-locate-2' in batch


def test_oracle_execution_failure_is_in_failure_artifact_and_readable_report(tmp_path):
    from gamelens.evaluation.benchmark_runner import report
    report(tmp_path,{'mode':'oracle','identity':{'dataset_sha256':'test'},'summary':{},
        'results':[{'case_id':'failed-case','tool_oracle_passed':False},
                   {'case_id':'successful-case','tool_oracle_passed':True}]})
    assert [row['case_id'] for row in json.loads((tmp_path/'failures.json').read_text(encoding='utf-8'))]==['failed-case']
    assert 'failed-case' in (tmp_path/'report.md').read_text(encoding='utf-8')


def test_oracle_run_has_no_agent_answer_accuracy(bundle,tmp_path):
    from gamelens.evaluation.benchmark_runner import run_oracle
    result=run_oracle(bundle,tmp_path/'oracle',limit=1)
    assert result['mode']=='oracle' and result['summary']['task_success_rate'] is None
    assert (tmp_path/'oracle/report.md').is_file()


def test_paid_runner_refuses_unreviewed_cases_before_creating_client(bundle,tmp_path):
    from gamelens.evaluation.benchmark_runner import run_agent
    with pytest.raises(ValueError,match='审核'):
        run_agent(bundle,tmp_path/'agent',tmp_path,allow_paid=True,max_http=3)
    assert not (tmp_path/'agent').exists()


def test_human_score_requires_support_and_scope_two(bundle):
    from gamelens.evaluation.benchmark_score import _human,review_template
    b=load(bundle);c=b['investigations'][0];run={}
    human=review_template(b,c,run)
    human.update(completed=True,reviewer='human',reviewed_at='2026-10-14T00:00:00Z',notes='reviewed',
        scores={'relevance':2,'evidence_support':1,'scope_handling':2,'completeness':2,'readability':2})
    assert _human(b,c,run,human)['passed'] is False


def test_changed_source_after_annotation_cannot_be_formal(bundle):
    from gamelens.evaluation.benchmark_data import import_reviews
    b=load(bundle);c=b['retrieval'][0];old=case_hash(c)
    c['query']='changed question'
    (bundle/'retrieval.json').write_text(json.dumps(b['retrieval']),encoding='utf-8')
    approval={'dataset_sha256':b['manifest']['dataset_sha256'],'reviews':[{
        'case_id':c['id'],'case_sha256':old,'status':'reviewed','reviewer':'human',
        'reviewed_at':'2026-10-14T00:00:00Z','notes':'checked'}]}
    file=bundle/'approval.json';file.write_text(json.dumps(approval),encoding='utf-8')
    with pytest.raises(ValueError,match='哈希'):import_reviews(bundle,file)


def test_pending_bm25_preview_keeps_metrics_unknown(bundle,tmp_path):
    from gamelens.evaluation.benchmark_runner import run_retrieval
    result=run_retrieval(bundle,tmp_path/'retrieval',tmp_path,modes=['bm25'],exploratory=True)
    assert result['results'] and all(r['metrics'] is None for r in result['results'])
    assert result['summary']['bm25']['ndcg_at_10'] is None
    assert result['summary']['bm25']['p50_ms'] is not None


def test_private_mapping_does_not_leak_into_public_bundle(bundle,tmp_path):
    from gamelens.evaluation.benchmark_v1 import main
    target=tmp_path/'public'
    assert main(['public-export','--bundle',str(bundle),'--output',str(target)])==0
    assert not (target/'private').exists()
    for path in target.rglob('*.json'):
        value=path.read_text(encoding='utf-8')
        assert 'original_path' not in value and 'steamid' not in value


def test_bad_review_import_does_not_overwrite_pending_cases(bundle):
    from gamelens.evaluation.benchmark_data import import_reviews
    b=load(bundle);c=b['retrieval'][0];original=(bundle/'retrieval.json').read_bytes()
    payload={'dataset_sha256':b['manifest']['dataset_sha256'],'reviews':[{
        'case_id':c['id'],'case_sha256':case_hash(c),'status':'bogus','reviewer':'human',
        'reviewed_at':'2026-10-14T00:00:00Z','notes':'checked'}]}
    file=bundle/'invalid.json';file.write_text(json.dumps(payload),encoding='utf-8')
    with pytest.raises(ValueError):import_reviews(bundle,file)
    assert (bundle/'retrieval.json').read_bytes()==original


def test_all_zero_grade_labels_need_explicit_no_answer_confirmation(bundle):
    b=load(bundle);c=b['retrieval'][0]
    if not c['labels']:
        source=b['corpus'][c['appid']]['reviews'][0]
        c['labels']=[{'kind':'review','source_id':source['review_id'],'relevance':0,'role':'partial'}]
    for label in c['labels']:label['relevance']=0
    c['review']={'status':'reviewed','reviewer':'human','reviewed_at':'2026-10-14T00:00:00Z','notes':'checked','case_sha256':case_hash(c)}
    (bundle/'retrieval.json').write_text(json.dumps(b['retrieval']),encoding='utf-8')
    with pytest.raises(ValueError,match='无答案'):validate(bundle)

def test_agent_all_split_selects_both_groups_and_excludes_unapproved():
    from gamelens.evaluation.benchmark_runner import agent_cases
    cases=[{'id':'dev','split':'development','review':{'status':'reviewed'},'blocking':[]},
           {'id':'test','split':'holdout','review':{'status':'reviewed'},'blocking':[]},
           {'id':'old','split':'holdout','review':{'status':'rejected'},'blocking':[]},
           {'id':'script','split':'holdout','review':{'status':'reviewed'},'blocking':[],'response_scenario':'generation_interruption','recovery_fixture':{'quote':'frozen original'}}]
    assert [c['id'] for c in agent_cases({'investigations':cases},split='all',limit=None)]==['dev','test']
    assert [c['id'] for c in agent_cases({'investigations':cases},split='holdout',limit=None)]==['test']

def test_suite_offline_continues_after_stage_failure_without_paid_calls(bundle,tmp_path,monkeypatch):
    from gamelens.evaluation import benchmark_suite as suite
    calls=[]
    def broken(*a,**k):calls.append('oracle');raise RuntimeError('fixture stage failure')
    def retrieval(*a,**k):calls.append('retrieval');return {'summary':{},'results':[],'unavailable_conditions':{}}
    def paid(*a,**k):pytest.fail('offline attempted paid stage')
    monkeypatch.setattr(suite,'run_oracle',broken);monkeypatch.setattr(suite,'run_retrieval',retrieval)
    monkeypatch.setattr(suite,'run_agent',paid);monkeypatch.setattr(suite,'run_judge',paid)
    result=suite.run_suite(bundle,tmp_path/'suite',tmp_path/'project')
    assert calls==['oracle','retrieval']
    assert result['stages']['oracle']['status']=='failed'
    assert result['stages']['agent']['status']=='not_measured'
    assert (tmp_path/'suite/REPORT.md').is_file()

def test_suite_rejects_paid_missing_budgets_before_output(bundle,tmp_path):
    from gamelens.evaluation.benchmark_suite import run_suite
    target=tmp_path/'suite'
    with pytest.raises(ValueError):run_suite(bundle,target,tmp_path/'project',allow_paid=True,agent_max_http=10)
    assert not target.exists()

def test_retrieval_persists_all_six_rankings_and_consistency(bundle,tmp_path):
    from gamelens.evaluation.benchmark_runner import run_retrieval
    result=run_retrieval(bundle,tmp_path/'rankings',tmp_path/'project',modes=['bm25'],exploratory=True)
    for row in result['results']:
        assert len(row['ranking_repetitions'])==6 and len(row['measurements_ms'])==5
        assert row['ranking_consistent'] is True

def test_frozen_environment_blocks_non_llm_post_and_stream(bundle,monkeypatch):
    import httpx
    from gamelens.evaluation.benchmark_runner import frozen_environment
    monkeypatch.setenv('LLM_BASE_URL','https://example.invalid/v1');calls=[]
    def handle(req):calls.append(str(req.url));return httpx.Response(200,json={'ok':True})
    with frozen_environment(load(bundle)),httpx.Client(transport=httpx.MockTransport(handle)) as client:
        assert client.post('https://example.invalid/v1/chat/completions').status_code==200
        with pytest.raises(RuntimeError):client.post('https://steamcommunity.com/comment/ForumTopic/render/1/2/')
        with pytest.raises(RuntimeError):
            with client.stream('POST','https://steamcommunity.com/comment/ForumTopic/render/1/2/'):pass
    assert len(calls)==1

@pytest.mark.parametrize('exhausted',[False,True])
def test_agent_scoring_error_keeps_paid_answer_and_continues_next_case(bundle,tmp_path,monkeypatch,exhausted):
    from gamelens.evaluation import benchmark_runner as runner
    b=load(bundle);b['investigations']=b['investigations'][:2]
    for c in b['investigations']:c['review']['status']='reviewed';c['blocking']=[]
    monkeypatch.setattr(runner,'load',lambda p:b);monkeypatch.setattr(runner,'validate',lambda p:None)
    monkeypatch.setattr(runner,'registry_for',lambda *a,**kw:object())
    monkeypatch.setenv('LLM_API_KEY','test');monkeypatch.setenv('LLM_MODEL','test')
    if exhausted:monkeypatch.setattr('gamelens.core.budget.TaskBudgetStore.snapshot',lambda *a:{'llm_requests_remaining':0})
    class Agent:
        def __init__(self,*a,**kw):pass
        def stream(self,state):yield {**state,'status':'completed','final_answer':'saved '+state['thread_id']}
    monkeypatch.setattr('gamelens.core.runtime.InvestigationAgent',Agent)
    score=runner.score_run;calls=[]
    def fail_first(*a,**kw):
        calls.append(1)
        if len(calls)==1:raise ValueError('injected evaluator failure')
        return score(*a,**kw)
    monkeypatch.setattr(runner,'score_run',fail_first)
    out=tmp_path/'agent';result=runner.run_agent(bundle,out,tmp_path,split='all',limit=None,allow_paid=True,max_http=2)
    manifest=json.loads((out/'runs_manifest.json').read_text(encoding='utf-8'))
    assert len(manifest['runs'])==2 and len(result['results'])==2
    first=json.loads((out/manifest['runs'][0]['path']).read_text(encoding='utf-8'))
    if exhausted:assert first['status']=='not_executed' and result['results'][0].get('evaluation_error')
    else:assert first['final_answer'].startswith('saved ') and first.get('evaluation_error')


def test_interrupted_snapshot_is_saved_and_next_case_runs(bundle,tmp_path,monkeypatch):
    from gamelens.evaluation import benchmark_runner as runner
    from langgraph.types import Interrupt
    b=load(bundle);b['investigations']=b['investigations'][:2]
    for c in b['investigations']:c['review']['status']='reviewed';c['blocking']=[]
    monkeypatch.setattr(runner,'load',lambda p:b);monkeypatch.setattr(runner,'validate',lambda p:None)
    monkeypatch.setattr(runner,'registry_for',lambda *a,**kw:object())
    monkeypatch.setenv('LLM_API_KEY','test');monkeypatch.setenv('LLM_MODEL','test')
    first_id=b['investigations'][0]['id']
    class Agent:
        def __init__(self,*a,**kw):pass
        def stream(self,state):
            if state['thread_id']==first_id:
                yield {**state,'__interrupt__':(Interrupt(value={'tool':'approval_only','reason':'approval_required'},id='saved-interrupt'),)}
            else:yield {**state,'status':'completed','final_answer':'next case survived'}
    monkeypatch.setattr('gamelens.core.runtime.InvestigationAgent',Agent)
    out=tmp_path/'agent';result=runner.run_agent(bundle,out,tmp_path,split='all',limit=None,allow_paid=True,max_http=2)
    manifest=read_json(out/'runs_manifest.json')
    saved=read_json(out/manifest['runs'][0]['path'])
    assert saved['status']=='interrupted' and saved['approval_pending'] is True
    assert saved['__interrupt__'][0]['value']['tool']=='approval_only'
    assert saved['__interrupt__'][0]['id']=='saved-interrupt'
    assert len(result['results'])==2
    assert read_json(out/manifest['runs'][1]['path'])['final_answer']=='next case survived'


def read_json(path):return json.loads(path.read_text(encoding='utf-8'))


def test_agent_resume_skips_saved_case_and_reuses_original_budget(bundle,tmp_path,monkeypatch):
    from gamelens.evaluation import benchmark_runner as runner
    from gamelens.core.budget import TaskBudgetStore
    b=load(bundle);b['investigations']=b['investigations'][:2]
    for c in b['investigations']:c['review']['status']='reviewed';c['blocking']=[]
    monkeypatch.setattr(runner,'load',lambda p:b);monkeypatch.setattr(runner,'validate',lambda p:None)
    monkeypatch.setattr(runner,'registry_for',lambda *a,**kw:object())
    monkeypatch.setenv('LLM_API_KEY','test');monkeypatch.setenv('LLM_MODEL','test')
    prior=tmp_path/'prior';runner.output_directory(prior)
    ledger=prior/'private/http-budget.sqlite';budget=TaskBudgetStore(ledger,request_limit=3,active_seconds_limit=1800)
    budget.reserve_llm_request('benchmark-batch')
    first=b['investigations'][0];state=runner.state_for(b,first)
    state.update(status='completed',final_answer='already saved',telemetry={'http_requests':1})
    manifest=[];ident=runner.identity(b,b['investigations'],{'split':'all','seed':0,'limit':None,'max_http':3,'active_budget_seconds':1800})
    runner.archive_agent_result(b,first,state,tmp_path,prior,'agent',1,manifest,ident)
    original=(prior/'runs_manifest.json').read_bytes()
    class Agent:
        def __init__(self,*a,**kw):self.budget=kw['task_budget']
        def stream(self,state):
            assert state['thread_id']==b['investigations'][1]['id']
            self.budget.reserve_llm_request('benchmark-batch')
            yield {**state,'status':'completed','final_answer':'remaining case'}
    monkeypatch.setattr('gamelens.core.runtime.InvestigationAgent',Agent)
    out=tmp_path/'continued'
    result=runner.run_agent(bundle,out,tmp_path,split='all',limit=None,allow_paid=True,max_http=3,
        resume_from=prior/'runs_manifest.json',budget_ledger=ledger)
    assert [r['case_id'] for r in result['results']]==[b['investigations'][1]['id']]
    assert budget.snapshot('benchmark-batch')['llm_requests_reserved']==2
    assert not (out/'private/http-budget.sqlite').exists()
    assert (prior/'runs_manifest.json').read_bytes()==original
    assert result['request_budget_before']['llm_requests_reserved']==1
    with pytest.raises(ValueError,match='没有未执行'):
        runner.run_agent(bundle,tmp_path/'second-continuation',tmp_path,split='all',limit=None,allow_paid=True,max_http=3,
            resume_from=out/'runs_manifest.json',budget_ledger=ledger)
    assert budget.snapshot('benchmark-batch')['llm_requests_reserved']==2
    with pytest.raises(ValueError,match='预算'):
        runner.run_agent(bundle,tmp_path/'wrong',tmp_path,split='all',limit=None,allow_paid=True,max_http=4,
            resume_from=prior/'runs_manifest.json',budget_ledger=ledger)
    assert not (tmp_path/'wrong').exists()


@pytest.mark.parametrize('judge_tokens',[None,8192])
def test_suite_continuation_reuses_finished_stages_and_both_ledgers(bundle,tmp_path,monkeypatch,judge_tokens):
    from gamelens.evaluation import benchmark_suite as suite
    prior=tmp_path/'prior';prior.mkdir();(prior/'agent').mkdir()
    inventory=suite.benchmark_status(bundle);inventory['formal_ready']=True
    monkeypatch.setattr(suite,'benchmark_status',lambda p:inventory)
    (prior/'suite.json').write_text(json.dumps({'inventory':inventory,'budgets':{'agent':3,'judge':2},'judge_output_tokens':judge_tokens,
        'stages':{'oracle':{'status':'partial','path':str(prior/'oracle'),'summary':{'executed':19}},
                  'retrieval':{'status':'completed','path':str(prior/'retrieval'),'summary':{}},
                  'judge':{'status':'completed','path':str(prior/'judge'),'summary':{'ai_scored_cases':1}}}}),encoding='utf-8')
    (prior/'agent/runs_manifest.json').write_text('{}',encoding='utf-8')
    monkeypatch.setattr(suite,'run_oracle',lambda *a,**kw:pytest.fail('repeated Oracle'))
    monkeypatch.setattr(suite,'run_retrieval',lambda *a,**kw:pytest.fail('repeated retrieval'))
    def agent(b,out,p,**kw):
        assert kw['budget_ledger']==prior/'agent/private/http-budget.sqlite'
        assert kw['resume_from']==prior/'agent/runs_manifest.json'
        out.mkdir();(out/'runs_manifest.json').write_text('{"runs":[{}]}',encoding='utf-8')
        return {'summary':{}}
    def judge(inp,out,**kw):
        assert kw['budget_ledger']==prior/'judge/private/http-budget.sqlite'
        assert kw['max_http']==2
        assert kw.get('output_tokens_override')==judge_tokens
        return {'summary':{'ai_scored_cases':1,'ai_requested_cases':1}}
    monkeypatch.setattr(suite,'run_agent',agent);monkeypatch.setattr(suite,'run_judge',judge)
    monkeypatch.setattr(suite,'export_judge',lambda *a,**kw:{})
    monkeypatch.setattr(suite,'replay',lambda *a,**kw:{'summary':{}})
    options={} if judge_tokens is None else {'judge_output_tokens':judge_tokens}
    result=suite.run_suite(bundle,tmp_path/'continued',tmp_path,allow_paid=True,agent_max_http=3,judge_max_http=2,resume_from=prior,**options)
    assert result['stages']['retrieval']['status']=='completed'
    assert result['continuation']['source_suite_sha256']
    assert result.get('judge_output_tokens')==judge_tokens
    if judge_tokens is not None:
        with pytest.raises(ValueError,match='输出预算'):
            suite.run_suite(bundle,tmp_path/'changed-budget',tmp_path,allow_paid=True,agent_max_http=3,
                judge_max_http=2,resume_from=prior,judge_output_tokens=4096)
        assert not (tmp_path/'changed-budget').exists()


@pytest.mark.parametrize('invalid',[511,8193,True])
def test_suite_rejects_invalid_judge_tokens_before_any_stage(tmp_path,invalid):
    from gamelens.evaluation.benchmark_suite import run_suite
    with pytest.raises(ValueError,match='512..8192'):
        run_suite(tmp_path/'absent',tmp_path/'result',tmp_path,judge_output_tokens=invalid)
    assert not (tmp_path/'result').exists()


def test_suite_cli_rejects_invalid_judge_tokens_without_starting_batch(tmp_path):
    from gamelens.evaluation.benchmark_v1 import main
    assert main(['suite','--bundle',str(tmp_path/'absent'),'--output',str(tmp_path/'result'),
        '--judge-output-tokens','8193'])==2
    assert not (tmp_path/'result').exists()
