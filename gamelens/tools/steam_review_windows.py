"""Bounded anonymous GetAppReviews acquisition; never assumes a full archive."""
import json
import math
import time

import httpx
from pydantic import ValidationError

from gamelens.tools.steam_reviews import normalize_review

ENDPOINT='https://api.steampowered.com/IUserReviewsService/GetAppReviews/v1/'


def counter(value):
    if isinstance(value,str) and value.isdigit():value=int(value)
    if type(value) is not int or value<0:raise ValueError('invalid_review_counter')
    return value


def public_page(client, query, timeout):
    response=client.get(ENDPOINT,params={'input_json':json.dumps(query,separators=(',',':'))},timeout=timeout)
    response.raise_for_status()
    envelope=response.json()
    body=envelope.get('response') if isinstance(envelope,dict) else None
    if not isinstance(body,dict) or not isinstance(body.get('query_summary'),dict):
        raise ValueError('invalid_review_response')
    if 'total_matching' not in body and 'total_reviews' not in body['query_summary']:
        raise ValueError('missing_review_counts')
    # Protocol zero fields may be omitted, but only within a valid response.
    total=counter(body.get('total_matching',0))
    records=body.get('reviews',[])
    if not isinstance(records,list) or total<len(records):raise ValueError('invalid_review_response')
    return body,records,total


def acquire_window(appid, start, end, language='all', *, target_count=1000,max_pages=10,
                   deadline=None,client=None,clock=time.monotonic,sleep=time.sleep):
    """start/end are positive UTC seconds, end exclusive; server end is inclusive."""
    start,end=math.ceil(start),math.ceil(end)
    if not 0<start<end<=4294967296:raise ValueError('invalid_acquisition_bounds')
    if not 1<=target_count<=2000 or not 1<=max_pages<=20:raise ValueError('invalid_acquisition_budget')
    deadline=deadline if deadline is not None else clock()+90
    if client is None:
        with httpx.Client(timeout=15,follow_redirects=False) as session:
            return acquire_window(appid,start,end,language,target_count=target_count,max_pages=max_pages,
                deadline=deadline,client=session,clock=clock,sleep=sleep)
    records=[];seen=set();cursors=set();cursor='*';pages=0;total=None;stats=None
    error=None;exhausted=False;invalid=0
    playtime_metadata_complete={'at_review':True,'lifetime':True}
    query={'appid':appid,'filter':1,'languages':[language],'review_type':0,'purchase_type':1,
        'filter_offtopic_activity':False,'date_range_start':start,'date_range_end':end-1}
    try:
        while pages<max_pages and len(records)<target_count:
            remaining=deadline-clock()
            if remaining<=0:error='time_budget';break
            if cursor in cursors:error='repeated_cursor';break
            cursors.add(cursor)
            page_size=min(100,target_count-len(records))
            body,page,reported=public_page(client,{**query,'cursor':cursor,
                'num_per_page':page_size},min(15,remaining))
            pages+=1
            if len(page)>page_size:raise ValueError('server_page_budget_mismatch')
            if total is not None and total!=reported:error='counts_changed_during_fetch'
            if total is None:
                total=reported;summary=body['query_summary']
                try:
                    positive=counter(summary.get('total_positive',0));negative=counter(summary.get('total_negative',0))
                    if counter(summary.get('total_reviews',0))!=total or positive+negative!=total:
                        raise ValueError('inconsistent_review_counts')
                    stats={'count':total,'positive_count':positive,'negative_count':negative,
                        'positive_rate':positive/total if total else None,
                        'negative_rate':negative/total if total else None}
                except ValueError:
                    stats=None;error='inconsistent_review_counts'
            new=0
            for raw in page:
                try:
                    if not isinstance(raw,dict):raise ValueError('invalid_review_record')
                    if not isinstance(raw.get('author'),dict):raise ValueError('invalid_review_author')
                    for basis,field in (('at_review','playtime_at_review'),('lifetime','playtime_forever')):
                        value=raw['author'].get(field)
                        if type(value) is not int or value<0:playtime_metadata_complete[basis]=False
                    if not raw.get('recommendationid'):raise ValueError('missing_review_id')
                    item=normalize_review({**raw,'recommendationid':str(raw['recommendationid'])})
                    if not start<=item.created_at_timestamp<end or language!='all' and item.language!=language:
                        raise ValueError('server_filter_mismatch')
                    if item.review_id not in seen:
                        seen.add(item.review_id);records.append(item.model_dump(mode='json'));new+=1
                except (ValidationError,ValueError,TypeError):
                    invalid+=1;error='invalid_review_record_or_range'
            if error:break
            if not page:
                exhausted=len(records)==total
                if not exhausted:error='body_missing'
                break
            next_cursor=body.get('cursor')
            if not new:error='repeated_page';break
            if not isinstance(next_cursor,str) or not next_cursor or next_cursor==cursor:
                exhausted=len(records)==total
                if not exhausted:error='cursor_missing_or_stalled'
                break
            cursor=next_cursor
            if len(records)==total:exhausted=True;break
            if pages<max_pages and len(records)<target_count:sleep(min(.25,max(0,deadline-clock())))
    except httpx.HTTPStatusError as exc:error='http_'+str(exc.response.status_code)
    except httpx.TimeoutException:error='request_timeout'
    except httpx.TransportError:error='transport_error'
    except (ValueError,TypeError,json.JSONDecodeError):error='invalid_review_response'
    if error in {'counts_changed_during_fetch','invalid_review_record_or_range','invalid_review_response'}:stats=None
    stop_reason=error or ('endpoint_exhausted' if exhausted else
                         'sample_budget' if len(records)>=target_count else 'page_budget')
    return {'query':query,'comments':records,'reported_total':total,'stats':stats,
        'stats_verified':stats is not None,'body_count':len(records),'pages_read':pages,
        'complete_bodies':bool(exhausted and error is None and invalid==0 and len(records)==total),
        'playtime_metadata_complete':playtime_metadata_complete,
        'error_code':error,'stop_reason':stop_reason,
        'sampling':'recent_first_within_exact_window','invalid_records':invalid,
        'endpoint':ENDPOINT,'historical_text_available':False}
