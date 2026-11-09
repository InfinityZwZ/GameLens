"""Independent announcement discussions: identity, pagination, gates and routing."""
import copy
import hashlib
import html
import json
from datetime import datetime, timezone
import httpx
import pytest

import gamelens.core.runtime as runtime
from gamelens.core.runtime import InvestigationAgent, PlannerDecision
from gamelens.core.grounding import validate_comment_citations, validate_grounding
from gamelens.core.investigation_checks import assess_investigation_progress, progress_finish_errors
from gamelens.core.synthesis import evidence_packet, expand_decision
from gamelens.memory.context import PlannerContextBuilder
from gamelens.tools.steam_news import normalize_news
from gamelens.tools.steam_announcement_comments import fetch_announcement_comments, DiscussionError, CommentHTML
from gamelens.tools.announcement_comments import (AnnouncementDiscussionTools, PrepareDiscussionArgs,
    QueryDiscussionArgs, GetDiscussionArgs, discussion_fetch_approval)
from gamelens.tools.investigation import ListPatchesArgs
from gamelens.tools.registry import SteamInvestigationTools
from gamelens.storage.atomic_io import atomic_write_json
from gamelens.evaluation.benchmark import initial_state
from test_agent_runtime import SequencePlanner

DATE = 1791499009
NEWS = '675134162359814147'
TOPIC = '587313969629570500'
OWNER = '103582791432902485'
FORUM = '1634166237646390927'


def news(gid=NEWS, timestamp=DATE, title='Counter-Strike 2 Update'):
    return normalize_news({'gid': gid, 'title': title, 'date': timestamp, 'contents': 'Stability improvements.',
        'url': 'https://steamcommunity.com/ogg/730/announcements/detail/'+gid}, 730)


def event_page(*, appid=730, gid=NEWS, topic=TOPIC):
    event = {'gid': '675134162359814146', 'appid': appid, 'clan_steamid': OWNER,
        'forum_topic_id': topic, 'gidfeature': FORUM, 'comment_type': 'ForumTopic',
        'announcement_body': {'gid': gid, 'posttime': DATE}}
    return '<div data-partnereventstore="'+html.escape(json.dumps([event]), quote=True)+'"></div>'


def comment_markup(index, text=None):
    return f'<div class="commentthread_comment" id="comment_{100+index}"><div class="commentthread_comment_text">'+(
        text or ['卡顿已经改善', '更新后仍然闪退', '这个活动不错', '反作弊还需要改进'][index % 4])+(
        f'</div><div class="commentthread_comment_timestamp" data-timestamp="{DATE+index+1}"></div></div>')


def thread_page(total=4, *, topic=TOPIC, backlink=NEWS):
    size = min(total, 2)
    config = {'owner': OWNER, 'feature': FORUM, 'feature2': topic, 'total_count': total, 'pagesize': size,
        'extended_data': json.dumps({'forum_appid':730, 'forum_type':'Event'}), 'comments_raw': {}}
    return (f'<meta name="Description" content="https://steamcommunity.com/ogg/730/announcements/detail/{backlink}">'+
        '<script>g_sessionID = "abc123"; InitializeCommentThread("ForumTopic", "fixture", '+json.dumps(config)+');</script>'+
        ''.join(comment_markup(i) for i in range(size)))


def provider(*, total=4, event=None, thread=None, failure=False, wrong_offset=False,
             strings=False, duplicate_second=False):
    requests = []
    def handler(request):
        requests.append(request)
        if '/announcements/detail/' in request.url.path:
            return httpx.Response(200, text=event or event_page())
        if '/eventcomments/' in request.url.path:
            return httpx.Response(200, text=thread if thread is not None else thread_page(total))
        if request.method == 'POST' and '/comment/ForumTopic/render/' in request.url.path:
            if failure:
                return httpx.Response(403)
            from urllib.parse import parse_qs
            data = parse_qs(request.content.decode())
            assert data['feature2'] == [TOPIC] and data['oldestfirst'] == ['true']
            start, count = int(data['start'][0]), int(data['count'][0])
            count = min(count, total-start)
            markup = (comment_markup(start-1)+comment_markup(start) if duplicate_second and start else
                ''.join(comment_markup(i) for i in range(start,start+count)))
            return httpx.Response(200, json={'success':True, 'start':start+1 if wrong_offset else start,
                'name':'ForumTopic_'+OWNER+'_'+FORUM+'_'+TOPIC,
                'pagesize':str(count) if strings else count, 'total_count':str(total) if strings else total,
                'comments_html':markup})
        raise AssertionError('Unexpected network endpoint: '+str(request.url))
    return httpx.Client(transport=httpx.MockTransport(handler)), requests


@pytest.mark.parametrize('limit,complete,expected', [(4,True,4),(3,False,3),(1,False,1)])
def test_bounded_pagination_and_separate_message_fields(limit, complete, expected):
    client, requests = provider()
    result = fetch_announcement_comments(news(), limit, client=client, request_interval=0)
    assert result['coverage']['complete'] is complete
    assert result['coverage']['reported_total'] == 4
    assert len(result['comments']) == expected
    assert result['mapping']['topic_id'] != result['mapping']['announcement_gid']
    assert all(c['news_id'] == NEWS and c['topic_id'] == TOPIC for c in result['comments'])
    assert not any(key in c for c in result['comments'] for key in ('is_positive','playtime_minutes','language'))
    assert all(r.url.host == 'steamcommunity.com' for r in requests)


def test_default_discussion_budget_is_500_without_large_fetch_approval():
    args=PrepareDiscussionArgs(appid=730,news_id=NEWS)
    assert args.target_count==500
    assert discussion_fetch_approval(args,{}) is None


def test_default_fetch_collects_500_when_thread_has_more_messages():
    client,_=provider(total=600)
    result=fetch_announcement_comments(news(),client=client,request_interval=0)
    assert len(result['comments'])==500
    assert result['coverage']['target_count']==500
    assert not result['coverage']['complete']


def test_maximum_discussion_budget_collects_1000_messages():
    args=PrepareDiscussionArgs(appid=730,news_id=NEWS,target_count=1000)
    client,_=provider(total=1000)
    result=fetch_announcement_comments(news(),args.target_count,client=client,request_interval=0)
    assert len(result['comments'])==1000
    assert result['coverage']['target_count']==1000
    assert result['coverage']['complete']


def test_comment_query_offset_covers_the_full_1000_message_budget():
    assert QueryDiscussionArgs(appid=730,news_id=NEWS,offset=950).offset==950
    with pytest.raises(ValueError):
        QueryDiscussionArgs(appid=730,news_id=NEWS,offset=1001)


def test_actual_empty_thread_not_confused_with_unavailable():
    client, _ = provider(total=0)
    result = fetch_announcement_comments(news(), client=client, request_interval=0)
    assert result['comments'] == [] and result['coverage']['complete']
    assert result['coverage']['reported_total'] == 0


def test_public_endpoint_numeric_strings_are_supported():
    client, _ = provider(strings=True)
    result=fetch_announcement_comments(news(),4,client=client,request_interval=0)
    assert result['coverage']['complete'] and len(result['comments'])==4


def test_real_news_feed_redirect_preserves_distinct_news_and_announcement_ids():
    record=news('1845383656387875')
    record.url='https://steamstore-a.akamaihd.net/news/externalpost/steam_community_announcements/'+record.news_id
    base,_=provider()
    def handler(request):
        if request.url.host=='steamstore-a.akamaihd.net':
            return httpx.Response(302,headers={'location':'https://steamcommunity.com/games/CSGO/announcements/detail/'+NEWS})
        return base._transport.handle_request(request)
    result=fetch_announcement_comments(record,4,client=httpx.Client(transport=httpx.MockTransport(handler)),request_interval=0)
    assert result['mapping']['news_id'] != result['mapping']['announcement_gid']
    assert all(c['news_id']==record.news_id and c['announcement_gid']==NEWS for c in result['comments'])


@pytest.mark.parametrize('value',[True,-1,1.2,'-1','bad'])
def test_invalid_pagination_values_are_not_silently_coerced(value):
    from gamelens.tools.steam_announcement_comments import page_integer
    with pytest.raises(DiscussionError):page_integer(value,'pagesize')


@pytest.mark.parametrize('option,error', [({'event':event_page(appid=999)},'identity_mismatch'),
    ({'event':event_page(gid='999')},'event_mapping_unavailable'),
    ({'thread':thread_page(topic='999')},'identity_mismatch'),
    ({'thread':thread_page(backlink='999')},'backlink_mismatch'),
    ({'thread':'Please sign in'},'discussion_unavailable')])
def test_mapping_and_privacy_fail_closed(option, error):
    client, _ = provider(**option)
    with pytest.raises(DiscussionError) as caught:
        fetch_announcement_comments(news(), client=client, request_interval=0)
    assert caught.value.code == error


@pytest.mark.parametrize('option,code', [({'failure':True},'HTTPStatusError'),({'wrong_offset':True},'invalid_page')])
def test_partial_page_failure_preserves_known_records(option, code):
    client, _ = provider(**option)
    result = fetch_announcement_comments(news(), client=client, request_interval=0)
    assert len(result['comments']) == 2 and not result['coverage']['complete']
    assert result['coverage']['page_error_code'] == code


def test_cross_page_duplicate_is_unstable_not_source_exhausted():
    client,_=provider(duplicate_second=True)
    result=fetch_announcement_comments(news(),4,client=client,request_interval=0)
    assert result['coverage']['page_error_code']=='unstable_pagination'
    assert not result['coverage']['source_exhausted'] and not result['coverage']['complete']
    assert result['coverage']['duplicate_message_count']==1


def test_quote_blocks_are_flagged_without_inventing_quote_text():
    records = CommentHTML(comment_markup(0, '<blockquote class="bb_blockquote">对方说正常</blockquote>但我仍闪退<br>需要修复')).records
    assert records[0]['contains_quoted_text']
    assert records[0]['text'] == '对方说正常\n但我仍闪退\n需要修复'


def test_image_only_message_does_not_destroy_other_valid_messages():
    thread=thread_page().replace('卡顿已经改善','<img src="https://example.com/meme.png">')
    client,requests=provider(thread=thread)
    result=fetch_announcement_comments(news(),4,client=client,request_interval=0)
    assert len(result['comments'])==3 and result['coverage']['unreadable_message_count']==1
    assert not result['coverage']['complete']
    assert result['coverage']['source_exhausted'] and not result['coverage']['has_more']
    assert not any(r.url.host=='example.com' for r in requests)


@pytest.fixture
def corpus(tmp_path):
    path = tmp_path/'data/raw/730_news.json'
    atomic_write_json(path, [news().model_dump(mode='json')])
    state = initial_state('comments-test', '调查玩家对2026-10-08更新版本的反馈内容，请读取公告评论区留言。')
    state.update(investigation_profile='focused', rag_mode='off', max_steps=10)
    return AnnouncementDiscussionTools(tmp_path), state


def lookup(tools, state, **kwargs):
    result = tools.list_announcements(ListPatchesArgs(appid=730,published_on='2026-10-08',**kwargs), state)
    state['observations'].append({'step':1,'tool':'list_announcements','ok':True,'data':result.data})
    state['artifacts'].update(result.artifacts)
    return result


def snapshot(tools, state):
    lookup(tools, state)
    client, _ = provider()
    result = fetch_announcement_comments(news(), 4, client=client, request_interval=0)
    atomic_write_json(tools.path(PrepareDiscussionArgs(appid=730,news_id=NEWS)), result)
    return result


def test_activity_announcements_are_not_filtered_out(corpus):
    tools, state = corpus
    activity = news('10002', title='周年活动')
    atomic_write_json(tools.root/'data/raw/730_news.json', [activity.model_dump(mode='json')])
    state['goal'] = '调查2026-10-08活动公告下的留言'
    result = lookup(tools, state)
    assert result.data['selected_news_id'] == '10002'
    assert not result.data['announcements'][0]['is_patch_candidate']


@pytest.mark.parametrize('goal', ['调查玩家对YYYY-MM-DD更新版本的反馈内容', '调查玩家对某日期更新版本的反馈内容'])
def test_date_placeholder_is_not_latest_update(corpus, goal):
    tools, state = corpus; state['goal'] = goal
    with pytest.raises(ValueError,match='具体日期'):
        tools.list_announcements(ListPatchesArgs(appid=730), state)
    assert assess_investigation_progress(state)['needs_clarification']


def test_wrong_game_date_and_missing_lookup_are_rejected(corpus):
    tools, state = corpus
    with pytest.raises(ValueError,match='先list_announcements'):
        tools.target(PrepareDiscussionArgs(appid=730,news_id=NEWS), state)
    with pytest.raises(ValueError,match='AppID'):
        tools.sources(999,state)
    lookup(tools,state)
    state['goal'] = '调查玩家对2026-07-10更新版本的反馈内容'
    with pytest.raises(ValueError,match='日期不一致'):
        tools.target(PrepareDiscussionArgs(appid=730,news_id=NEWS), state)


def test_same_day_multiple_updates_cannot_be_bypassed_by_model_id(corpus):
    tools, state = corpus
    atomic_write_json(tools.root/'data/raw/730_news.json', [news().model_dump(mode='json'), news('10002').model_dump(mode='json')])
    lookup(tools,state,news_id=NEWS)
    with pytest.raises(ValueError,match='同一天'):
        tools.target(PrepareDiscussionArgs(appid=730,news_id=NEWS), state)


def test_missing_year_across_years_requires_clarification(corpus):
    tools,state=corpus
    old=news('10002',timestamp=int(datetime(2025,10,8,tzinfo=timezone.utc).timestamp()))
    atomic_write_json(tools.root/'data/raw/730_news.json',[news().model_dump(mode='json'),old.model_dump(mode='json')])
    state['goal']='调查玩家对10月8日更新版本的反馈内容'
    result=tools.list_announcements(ListPatchesArgs(appid=730,published_on='10-08'),state)
    assert result.data['needs_year_clarification']
    with pytest.raises(ValueError,match='年份'):
        tools.target(PrepareDiscussionArgs(appid=730,news_id=NEWS),state)


@pytest.mark.parametrize('change',[{'collected_count':99},{'reported_total':99},{'has_more':True}])
def test_corrupt_coverage_cannot_claim_complete_thread(corpus,change):
    tools,state=corpus;saved=snapshot(tools,state)
    saved['coverage'].update(change)
    atomic_write_json(tools.path(PrepareDiscussionArgs(appid=730,news_id=NEWS)),saved)
    with pytest.raises(ValueError,match='覆盖|数量'):
        tools.query_announcement_comments(QueryDiscussionArgs(appid=730,news_id=NEWS),state)


def test_last_sequential_page_is_not_reported_as_more(corpus):
    tools,state=corpus;snapshot(tools,state)
    result=tools.query_announcement_comments(QueryDiscussionArgs(appid=730,news_id=NEWS,order='oldest',offset=3,limit=2),state)
    assert len(result.data['comments'])==1 and not result.data['has_more']
    assert result.data['next_offset']==4


@pytest.mark.parametrize('goal',['调查玩家对2026-10-08更新版本的反馈，只看商店评价','不要读公告评论区，只分析游戏商店评价'])
def test_user_can_explicitly_choose_store_source(goal):
    from gamelens.domain.announcement_comments import wants_announcement_feedback
    assert not wants_announcement_feedback(goal)


def test_query_full_record_citations_and_final_packet(corpus):
    tools, state = corpus; saved = snapshot(tools,state)
    result = tools.query_announcement_comments(QueryDiscussionArgs(appid=730,news_id=NEWS),state)
    state['observations'].append({'step':2,'ok':True,'tool':'query_announcement_comments','data':result.data})
    state['artifacts'].update(result.artifacts)
    record = result.data['comments'][0]
    citation = {'source_step':2,'news_id':NEWS,'comment_id':record['comment_id'],'quote':record['text']}
    assert not validate_comment_citations(state['observations'],[citation])
    assert validate_comment_citations(state['observations'],[{**citation,'news_id':'wrong'}])
    assert validate_comment_citations(state['observations'],[{**citation,'quote':'并不存在的文本'}])
    packet, evidence = evidence_packet(state,[],PlannerContextBuilder())
    ids = [key for key, value in evidence.items() if value['kind'] == 'announcement_comment']
    assert ids and packet['target_patch_ids'] == [NEWS]
    decision = expand_decision({'action':'finish','reason':'answer','comment_evidence_ids':[ids[0]],
        'final_answer':'只能描述该公告讨论的已采集反馈。'}, evidence)
    assert decision.comment_citations and not decision.citations
    assert not validate_comment_citations(state['observations'],[c.model_dump() for c in decision.comment_citations])
    full = tools.get_announcement_comment(GetDiscussionArgs(appid=730,news_id=NEWS,comment_id=record['comment_id']),state)
    assert full.data['comments'][0]['text'] == record['text']
    assert saved['coverage']['complete']


def test_store_reviews_cannot_satisfy_discussion_requirement(corpus):
    tools, state = corpus; lookup(tools,state)
    state['observations'].append({'step':2,'ok':True,'tool':'query_reviews','data':{'reviews':[{'review_id':'r1','text':'开枪卡顿'}]}})
    progress = assess_investigation_progress(state,comment_citations=[])
    assert 'announcement_comments' in progress['required_evidence']
    assert any(c['requirement']=='announcement_comments' and c['status']!='satisfied' for c in progress['checks'])
    assert progress_finish_errors(progress,False)


def test_partial_discussion_requires_explicit_scope_limit(corpus):
    tools, state = corpus; saved = snapshot(tools,state)
    saved['coverage']['complete']=False
    atomic_write_json(tools.path(PrepareDiscussionArgs(appid=730,news_id=NEWS)),saved)
    result=tools.query_announcement_comments(QueryDiscussionArgs(appid=730,news_id=NEWS),state)
    state['observations'].append({'step':2,'ok':True,'tool':'query_announcement_comments','data':result.data})
    progress = assess_investigation_progress(state)
    assert any(c['requirement']=='announcement_comments' and c['status']=='insufficient_coverage' for c in progress['checks'])
    assert progress_finish_errors(progress,False)


def test_unavailable_does_not_replace_prior_snapshot(corpus,monkeypatch):
    tools, state = corpus; snapshot(tools,state)
    path = tools.path(PrepareDiscussionArgs(appid=730,news_id=NEWS))
    original=path.read_bytes()
    def unavailable(*_):raise DiscussionError('discussion_unavailable','cannot read')
    monkeypatch.setattr('gamelens.tools.steam_announcement_comments.fetch_announcement_comments',unavailable)
    result=tools.prepare_announcement_comments(PrepareDiscussionArgs(appid=730,news_id=NEWS,refresh=True),state)
    assert result.data['coverage']['collected_count']==4
    assert result.data['status']=='reused_after_refresh_failure'
    assert result.data['retained_previous_snapshot'] and result.data['error_code']=='discussion_unavailable'
    assert path.read_bytes()==original


def test_inadequate_discussion_cache_is_refetched_from_the_first_page(corpus,monkeypatch):
    tools,state=corpus;lookup(tools,state)
    old_client,_=provider(total=600)
    old=fetch_announcement_comments(news(),200,client=old_client,request_interval=0)
    path=tools.path(PrepareDiscussionArgs(appid=730,news_id=NEWS))
    atomic_write_json(path,old)
    new_client,_=provider(total=600)
    refreshed=fetch_announcement_comments(news(),500,client=new_client,request_interval=0)
    monkeypatch.setattr('gamelens.tools.steam_announcement_comments.fetch_announcement_comments',
        lambda *_:copy.deepcopy(refreshed))
    result=tools.prepare_announcement_comments(PrepareDiscussionArgs(appid=730,news_id=NEWS),state)
    assert result.data['collected_count']==500 and not result.data['reused']
    assert result.data['refresh_attempted'] and result.data['cache_adequate']
    assert json.loads(path.read_text(encoding='utf-8'))['coverage']['collected_count']==500


def test_complete_discussion_cache_is_reused_even_below_new_target(corpus,monkeypatch):
    tools,state=corpus;snapshot(tools,state)
    monkeypatch.setattr('gamelens.tools.steam_announcement_comments.fetch_announcement_comments',
        lambda *_:pytest.fail('A complete source must not be fetched again'))
    result=tools.prepare_announcement_comments(PrepareDiscussionArgs(appid=730,news_id=NEWS),state)
    assert result.data['reused'] and result.data['collected_count']==4
    assert result.data['cache_adequate'] and not result.data['refresh_attempted']


def test_exhausted_discussion_with_unreadable_body_is_reused(corpus,monkeypatch):
    tools,state=corpus;lookup(tools,state)
    thread=thread_page().replace('卡顿已经改善','<img src="https://example.com/meme.png">')
    client,_=provider(thread=thread)
    saved=fetch_announcement_comments(news(),500,client=client,request_interval=0)
    atomic_write_json(tools.path(PrepareDiscussionArgs(appid=730,news_id=NEWS)),saved)
    monkeypatch.setattr('gamelens.tools.steam_announcement_comments.fetch_announcement_comments',
        lambda *_:pytest.fail('An exhausted source must not be fetched again'))
    result=tools.prepare_announcement_comments(PrepareDiscussionArgs(appid=730,news_id=NEWS),state)
    assert result.data['reused'] and result.data['cache_adequate']
    assert result.data['coverage']['source_exhausted'] and not result.data['coverage']['complete']


def test_explicit_refresh_can_replace_small_complete_discussion_with_larger_snapshot(corpus,monkeypatch):
    tools,state=corpus;snapshot(tools,state)
    client,_=provider(total=600)
    larger=fetch_announcement_comments(news(),500,client=client,request_interval=0)
    monkeypatch.setattr('gamelens.tools.steam_announcement_comments.fetch_announcement_comments',
        lambda *_:copy.deepcopy(larger))
    result=tools.prepare_announcement_comments(
        PrepareDiscussionArgs(appid=730,news_id=NEWS,target_count=500,refresh=True),state)
    assert result.data['collected_count']==500 and not result.data['reused']
    assert result.data['cache_adequate'] and result.data['status']=='collected'


def test_explicit_refresh_replaces_equal_sized_complete_discussion_after_growth(corpus,monkeypatch):
    tools,state=corpus;lookup(tools,state)
    old_client,_=provider(total=500)
    old=fetch_announcement_comments(news(),500,client=old_client,request_interval=0)
    path=tools.path(PrepareDiscussionArgs(appid=730,news_id=NEWS));atomic_write_json(path,old)
    new_client,_=provider(total=600)
    candidate=fetch_announcement_comments(news(),500,client=new_client,request_interval=0)
    monkeypatch.setattr('gamelens.tools.steam_announcement_comments.fetch_announcement_comments',
        lambda *_:copy.deepcopy(candidate))
    result=tools.prepare_announcement_comments(
        PrepareDiscussionArgs(appid=730,news_id=NEWS,target_count=500,refresh=True),state)
    assert not result.data['reused'] and result.data['collected_count']==500
    assert result.data['coverage']['reported_total']==600


def test_partial_new_discussion_surfaces_collection_error(corpus,monkeypatch):
    tools,state=corpus;lookup(tools,state)
    client,_=provider(failure=True)
    partial=fetch_announcement_comments(news(),500,client=client,request_interval=0)
    monkeypatch.setattr('gamelens.tools.steam_announcement_comments.fetch_announcement_comments',
        lambda *_:copy.deepcopy(partial))
    result=tools.prepare_announcement_comments(PrepareDiscussionArgs(appid=730,news_id=NEWS),state)
    assert result.data['status']=='collected_partial'
    assert result.data['refresh_error_code']=='HTTPStatusError'
    assert result.data['error_code']=='HTTPStatusError' and not result.data['cache_adequate']


def test_discussion_parse_failure_preserves_prior_snapshot(corpus,monkeypatch):
    tools,state=corpus;snapshot(tools,state)
    path=tools.path(PrepareDiscussionArgs(appid=730,news_id=NEWS));original=path.read_bytes()
    monkeypatch.setattr('gamelens.tools.steam_announcement_comments.fetch_announcement_comments',
        lambda *_:(_ for _ in ()).throw(ValueError('bad payload')))
    result=tools.prepare_announcement_comments(
        PrepareDiscussionArgs(appid=730,news_id=NEWS,refresh=True),state)
    assert result.data['status']=='reused_after_refresh_failure'
    assert result.data['retained_previous_snapshot'] and path.read_bytes()==original


def test_counts_changed_candidate_does_not_replace_trusted_discussion(corpus,monkeypatch):
    tools,state=corpus;snapshot(tools,state)
    path=tools.path(PrepareDiscussionArgs(appid=730,news_id=NEWS));original=path.read_bytes()
    candidate=json.loads(original)
    candidate['coverage'].update(complete=False,source_exhausted=False,
        counts_changed_during_fetch=True,status='partial_thread_snapshot')
    monkeypatch.setattr('gamelens.tools.steam_announcement_comments.fetch_announcement_comments',
        lambda *_:copy.deepcopy(candidate))
    result=tools.prepare_announcement_comments(
        PrepareDiscussionArgs(appid=730,news_id=NEWS,refresh=True),state)
    assert result.data['reused'] and result.data['refresh_error_code']=='counts_changed_during_fetch'
    assert path.read_bytes()==original


def test_larger_counts_changed_discussion_does_not_replace_clean_partial_cache(corpus,monkeypatch):
    tools,state=corpus;lookup(tools,state)
    old_client,_=provider(total=600)
    old=fetch_announcement_comments(news(),200,client=old_client,request_interval=0)
    path=tools.path(PrepareDiscussionArgs(appid=730,news_id=NEWS));atomic_write_json(path,old)
    original=path.read_bytes()
    new_client,_=provider(total=600)
    candidate=fetch_announcement_comments(news(),500,client=new_client,request_interval=0)
    candidate['coverage']['counts_changed_during_fetch']=True
    monkeypatch.setattr('gamelens.tools.steam_announcement_comments.fetch_announcement_comments',
        lambda *_:copy.deepcopy(candidate))
    result=tools.prepare_announcement_comments(PrepareDiscussionArgs(appid=730,news_id=NEWS),state)
    assert result.data['reused'] and result.data['collected_count']==200
    assert result.data['refresh_error_code']=='counts_changed_during_fetch'
    assert path.read_bytes()==original


def test_worse_partial_discussion_refresh_keeps_better_cache(corpus,monkeypatch):
    tools,state=corpus;lookup(tools,state)
    old_client,_=provider(total=600)
    old=fetch_announcement_comments(news(),200,client=old_client,request_interval=0)
    path=tools.path(PrepareDiscussionArgs(appid=730,news_id=NEWS))
    atomic_write_json(path,old);original=path.read_bytes()
    worse_client,_=provider(total=600)
    worse=fetch_announcement_comments(news(),100,client=worse_client,request_interval=0)
    worse['coverage']['page_error_code']='request_timeout'
    monkeypatch.setattr('gamelens.tools.steam_announcement_comments.fetch_announcement_comments',
        lambda *_:copy.deepcopy(worse))
    result=tools.prepare_announcement_comments(PrepareDiscussionArgs(appid=730,news_id=NEWS),state)
    assert result.data['status']=='reused_after_refresh_shortfall'
    assert result.data['collected_count']==200 and result.data['retained_previous_snapshot']
    assert result.data['refresh_error_code']=='request_timeout'
    assert path.read_bytes()==original


def test_agent_can_finish_with_typed_comment_quote_and_readable_report(corpus,monkeypatch):
    tools,state=corpus
    client,_=provider()
    saved=fetch_announcement_comments(news(),4,client=client,request_interval=0)
    monkeypatch.setattr('gamelens.tools.steam_announcement_comments.fetch_announcement_comments',lambda *_:copy.deepcopy(saved))
    monkeypatch.setattr(runtime,'run_isolated',lambda handler,args,state,timeout:('result',handler(args,state).model_dump(mode='json')))
    class CheckedPlanner(SequencePlanner):
        def decide(self, state, specifications):
            assert self.decisions, state.get('validation_errors') or state.get('error')
            return super().decide(state, specifications)
    planner=CheckedPlanner([
        PlannerDecision(action='list_announcements',reason='按指定日期定位',arguments={'appid':730,'published_on':'2026-10-08'}),
        PlannerDecision(action='prepare_announcement_comments',reason='精确采集公告讨论',arguments={'appid':730,'news_id':NEWS}),
        PlannerDecision(action='query_announcement_comments',reason='读取玩家留言',arguments={'appid':730,'news_id':NEWS}),
        PlannerDecision(action='verify_investigation',reason='核验来源',arguments={'appid':730}),
        PlannerDecision(action='finish',reason='已取得独立留言',final_answer='**主要发现**\n\n- 有反馈称卡顿已改善，也有玩家仍遇到闪退。\n\n仅描述已核对留言。',
            claims=[{'source_step':3,'label':'返回留言数','fact_path':['returned_count'],'value':4}],
            comment_citations=[{'source_step':3,'news_id':NEWS,'comment_id':'100','quote':'卡顿已经改善'}])])
    result=InvestigationAgent(planner,SteamInvestigationTools(tools.root).registry()).invoke(state)
    assert result['status']=='completed',result.get('error')
    assert result['final_comment_citations'][0]['comment_id']=='100'
    assert not result['final_citations']
    assert not validate_grounding(result['observations'],result['final_claims'],[])
    report=__import__('pathlib').Path(result['artifacts']['final_report']).read_text(encoding='utf-8')
    assert '公告讨论区玩家留言' in report and '卡顿已经改善' in report and '#c100' in report


def test_registry_profiles_expose_independent_source_tools(tmp_path):
    for profile in ('focused','extended'):
        registry=SteamInvestigationTools(tmp_path).registry(profile)
        assert {'list_announcements','prepare_announcement_comments','query_announcement_comments','get_announcement_comment'} <= registry.names
        assert '商店' in next(t.description for t in registry.tools if t.name=='query_announcement_comments') or '好差评' in next(t.description for t in registry.tools if t.name=='query_announcement_comments')
