from gamelens.evaluation.benchmark_review import render_card, audit_cards


def bundle():
    return {'manifest': {'clock': '2026-10-14T00:00:00Z', 'dataset_sha256': 'hash',
            'timezone': {}, 'inventory': []},
        'corpus': {1623730: {'reviews': [{'review_id': 'r1', 'text': '不卡，但昨天闪退。',
            'language': 'schinese', 'is_positive': True, 'created_at_timestamp': 1,
            'playtime_at_review_minutes': 6001, 'playtime_minutes': 6100}], 'news': [], 'comments': []}}}


def test_conversation_shows_history_and_missing_target_instead_of_empty_evidence():
    c = {'id': 'task-conversation-1', 'appid': 1623730, 'category': 'conversation',
        'goal': '继续上一轮目标更新，仅看中文玩家原文。', 'history': [
            {'goal': '调查版本missing', 'target_news_id': 'missing', 'verified': False}],
        'expectations': {'filters': {'language': 'schinese'}, 'evidence': []},
        'review': {'status': 'pending_review'}, 'split': 'development', 'blocking': []}
    text, meta = render_card(bundle(), c, 'investigations')
    assert '调查版本missing' in text
    assert '未核验' in text and '中文' in text
    assert meta['gaps'] and '材料待补齐' in text
    assert '修改源题JSON' not in text


def test_statistics_gives_hand_calculated_population_and_original_not_vote_inference():
    c = {'id': 'task-statistics-2', 'appid': 1623730, 'category': 'statistics',
        'goal': '统计超过100小时的中文评论', 'expectations': {'filters': {
            'language': 'schinese', 'min_playtime_hours': 100,
            'min_playtime_exclusive': True, 'playtime_basis': 'at_review'}, 'facts': []},
        'review': {'status': 'pending_review'}, 'split': 'development', 'blocking': []}
    text, _ = render_card(bundle(), c, 'investigations')
    assert '1 条' in text and '100.00%' in text
    assert '不卡，但昨天闪退。' in text and '6001' in text
    assert '推荐不代表没有问题' in text


def test_missing_corpus_is_not_presented_as_no_answer_gold():
    c = {'id': 'ret-network-1', 'appid': 730, 'query': '断线', 'labels': [],
        'filters': {}, 'kind': 'reviews', 'blocking': ['missing_source'],
        'review': {'status': 'pending_review'}, 'split': 'holdout'}
    text, meta = render_card(bundle(), c, 'retrieval')
    assert meta['gaps'] and '不能据此确认无答案' in text
    assert '请勿批准' in text


def test_audit_rejects_missing_or_altered_card(tmp_path):
    from gamelens.storage.atomic_io import atomic_write_text
    c = {'id': 'ret-performance-1', 'appid': 1623730, 'query': '性能',
        'labels': [{'kind': 'review', 'source_id': 'r1', 'relevance': 2, 'role': 'support'}],
        'filters': {}, 'kind': 'reviews', 'blocking': [],
        'review': {'status': 'pending_review'}, 'split': 'development'}
    b = bundle(); b.update(retrieval=[c], investigations=[])
    folder = tmp_path/'review_pack/retrieval'; folder.mkdir(parents=True)
    path = folder/(c['id']+'.md')
    path.write_text('empty', encoding='utf-8')
    assert audit_cards(b, tmp_path)['ok'] is False
    text, _ = render_card(b, c, 'retrieval'); atomic_write_text(path,text)
    assert audit_cards(b, tmp_path)['ok'] is True


def test_original_crlf_survives_export_and_file_read(tmp_path):
    from gamelens.storage.atomic_io import atomic_write_text
    b = bundle(); b['corpus'][1623730]['reviews'][0]['text'] = '不卡\r\n但是闪退。'
    c = {'id': 'ret-performance-1', 'appid': 1623730, 'query': '性能',
        'labels': [{'kind': 'review', 'source_id': 'r1', 'relevance': 2, 'role': 'support'}],
        'filters': {}, 'kind': 'reviews', 'blocking': [],
        'review': {'status': 'pending_review'}, 'split': 'development'}
    b.update(retrieval=[c], investigations=[])
    text, _ = render_card(b,c,'retrieval')
    atomic_write_text(tmp_path/'review_pack/retrieval/ret-performance-1.md',text)
    assert audit_cards(b,tmp_path)['ok'] is True


def test_reading_view_strips_steam_markup_but_keeps_negation():
    from gamelens.evaluation.benchmark_review import display_text
    assert display_text('[p][b]Not fixed[/b][/p]<p>No crashes</p>') == 'Not fixed\n\nNo crashes'


def test_long_announcement_has_short_view_and_full_original():
    from gamelens.evaluation.benchmark_review import record_card
    raw = '<p>Not fixed.</p>'*1000
    text='\n'.join(record_card('announcement',{'news_id':'n','contents':raw},'candidate'))
    assert len(text.split('<details>',1)[0]) < 3000
    assert raw in text and 'Not fixed.' in text
    assert '节选' in text


def test_investigation_shows_actual_scored_labels_not_only_keyword_candidates():
    b=bundle()
    c={'id':'task-evidence-4','appid':1623730,'category':'evidence','goal':'否定',
       'expectations':{'filters':{},'evidence':[{'kind':'review','source_id':'r1','role':'partial','relevance':1}]},
       'split':'development','review':{'status':'pending_review'},'blocking':[]}
    text,meta=render_card(b,c,'investigations')
    assert '相关性预标 1/3' in text and '部分相关' in text
    assert ('review','r1') in meta['evidence_ids']


def test_timeout_without_execution_plan_not_ready_for_scenario_approval():
    c={'id':'task-reliability-3','appid':1623730,'category':'reliability','goal':'超时',
       'oracle_plan':[],'response_scenario':'timeout','expectations':{'filters':{}},
       'split':'holdout','review':{'status':'pending_review'},'blocking':[]}
    _,meta=render_card(bundle(),c,'investigations')
    assert meta['gaps']


def test_positive_filter_is_visible_and_excludes_negative_population():
    from gamelens.evaluation.benchmark_review import population,filter_words
    rows=[{'is_positive':True},{'is_positive':False}]
    assert population(rows,{'is_positive':True})==[{'is_positive':True}]
    assert any('推荐' in text for text in filter_words({'is_positive':True}))


def test_positive_complaint_card_does_not_add_unlabelled_keyword_matches():
    b=bundle(); b['corpus'][1623730]['reviews'].append({**b['corpus'][1623730]['reviews'][0],
        'review_id':'extra','text':'没有问题，很好玩。'})
    c={'id':'task-evidence-2','appid':1623730,'category':'evidence','goal':'推荐但抱怨',
       'expectations':{'filters':{'is_positive':True},'evidence':[
           {'kind':'review','source_id':'r1','relevance':2,'role':'support'}]},
       'split':'development','review':{'status':'pending_review'},'blocking':[]}
    _,meta=render_card(b,c,'investigations')
    assert meta['evidence_ids']==[('review','r1')]


def test_cross_game_history_is_not_review_ready_even_with_valid_source():
    b=bundle(); b['corpus'][251570]={'reviews':[],'comments':[],'news':[{'news_id':'old','title':'old patch','contents':'fix','published_at_timestamp':1}]}
    c={'id':'task-conversation-3','appid':1623730,'category':'conversation','goal':'改查英文冻结评论',
       'history':[{'appid':251570,'target_news_id':'old','goal':'旧游戏','filters':{'language':'schinese'}}],
       'expectations':{'filters':{'language':'english'},'outcome':'limited'},
       'split':'development','review':{'status':'pending_review'},'blocking':[]}
    text,meta=render_card(b,c,'investigations')
    assert any('同一会话不能切换游戏' in gap for gap in meta['gaps'])
    assert 'old patch' in text and '0 条' in text and '不能' in text


def test_conversation_originals_must_stay_inside_explicit_window():
    b=bundle(); b['corpus'][1623730]['news']=[{'news_id':'target','title':'patch','contents':'fix','published_at_timestamp':1}]
    b['corpus'][1623730]['reviews'].append({**b['corpus'][1623730]['reviews'][0],
        'review_id':'outside','created_at_timestamp':100})
    c={'id':'task-conversation-1','appid':1623730,'category':'conversation','goal':'更新后一秒',
       'history':[{'target_news_id':'target','goal':'目标','filters':{}}],
       'expectations':{'outcome':'answer','filters':{'language':'schinese','start_at':'1970-01-01T00:00:01+00:00','end_at':'1970-01-01T00:00:02+00:00'}},
       'split':'development','review':{'status':'pending_review'},'blocking':[]}
    _,meta=render_card(b,c,'investigations')
    assert not meta['gaps']
    assert ('review','r1') in meta['evidence_ids'] and ('review','outside') not in meta['evidence_ids']


def test_explicit_absence_review_is_not_just_zero_keyword_hits():
    c={'id':'ret-no_answer-1','appid':1623730,'query':'有明确驱动泄漏吗','kind':'both','filters':{},'labels':[],
       'expected_no_answer':True,'absence_review':{'dataset_sha256':'hash','document_count':1,
           'scope':'frozen_corpus_only','notes':['已检查近似候选，不能将一般卡顿当成驱动泄漏。']},
       'split':'holdout','review':{'status':'pending_review'},'blocking':[]}
    text,meta=render_card(bundle(),c,'retrieval')
    assert not meta['gaps'] and '不能将一般卡顿当成驱动泄漏' in text
    c['absence_review']['dataset_sha256']='wrong'
    _,meta=render_card(bundle(),c,'retrieval')
    assert meta['gaps']


def test_ambiguous_version_card_shows_candidates_not_unbound_player_feedback():
    b=bundle(); b['corpus'][1623730]['news']=[{'news_id':i,'title':i,'contents':'fix','published_at_timestamp':1} for i in ('a','b')]
    c={'id':'task-conversation-4','appid':1623730,'category':'conversation','goal':'那个版本？',
       'history':[{'target_news_id':i,'goal':'未选定的候选','filters':{}} for i in ('a','b')],
       'expectations':{'outcome':'clarification','filters':{}},'split':'holdout',
       'review':{'status':'pending_review'},'blocking':[]}
    _,meta=render_card(b,c,'investigations')
    assert not any(kind=='review' for kind,rid in meta['evidence_ids'])
    assert len(meta['evidence_ids'])==2
