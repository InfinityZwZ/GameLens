"""Anonymous bounded read-only adapter for Steam Event forum discussions.

This is a community web endpoint, not the documented Steam News/review API.
Verify app -> announcement -> event -> forum -> message identities at each hop.
No login cookies, posting, JavaScript execution, or user-controlled URL requests.
"""
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import urlparse

import httpx
from gamelens.domain.announcement_comments import AnnouncementComment


DEFAULT_COMMENT_TARGET = 500
MAX_COMMENT_TARGET = 1000
COMMENT_PAGE_SIZE = 50
MAX_COMMENT_PAGES = (MAX_COMMENT_TARGET + COMMENT_PAGE_SIZE - 1) // COMMENT_PAGE_SIZE + 1


class DiscussionError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def page_integer(value, field):
    # Steam render currently returns pagesize as a numeric string; do not accept
    # floats, booleans, negative values, or arbitrary coercion of bad responses.
    if type(value) is int and value >= 0:
        return value
    if isinstance(value, str) and re.fullmatch(r'\d{1,10}', value):
        return int(value)
    raise DiscussionError('invalid_page', '分页字段无效：'+field)


class PageMetadata(HTMLParser):
    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.attributes, self.meta = {}, {}
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        for key in ('data-partnereventstore', 'data-community'):
            if key in attrs:
                self.attributes[key] = json.loads(attrs[key])
        if tag == 'meta':
            self.meta[(attrs.get('name') or attrs.get('property') or '').lower()] = attrs.get('content', '')


class CommentHTML(HTMLParser):
    """Extract only message bodies and their timestamps; exclude controls/OP."""
    VOID = {'br', 'hr', 'img', 'input', 'meta', 'link', 'wbr', 'source', 'area', 'embed', 'param', 'track', 'col'}

    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.stack, self.records, self.current = [], [], None
        self.outer, self.body = None, None
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = set(attrs.get('class', '').split())
        if tag not in self.VOID:
            self.stack.append(tag)
        if 'commentthread_comment' in classes:
            if self.current is not None:
                raise DiscussionError('invalid_page', '留言HTML嵌套异常')
            match = re.fullmatch(r'comment_(\d+)', attrs.get('id', ''))
            if not match:
                raise DiscussionError('invalid_page', '留言ID缺失')
            self.current = {'comment_id': match[1], 'parts': [], 'contains_quoted_text': False}
            self.outer = len(self.stack)
        if self.current is not None:
            if 'data-timestamp' in attrs and 'commentthread_comment_timestamp' in classes:
                self.current['created_at_timestamp'] = int(attrs['data-timestamp'])
            if 'commentthread_comment_text' in classes:
                self.body = len(self.stack)
            if self.body is not None and (tag == 'blockquote' or 'bb_blockquote' in classes):
                self.current['contains_quoted_text'] = True
                self.current['parts'].append('\n')
            if self.body is not None and tag in {'br', 'p'}:
                self.current['parts'].append('\n')

    def handle_data(self, text):
        if self.current is not None and self.body is not None:
            self.current['parts'].append(text)

    def handle_endtag(self, tag):
        if tag not in self.stack:
            return
        position = len(self.stack) - 1 - self.stack[::-1].index(tag)
        if self.current is not None:
            if self.body is not None and position + 1 <= self.body:
                self.body = None
            elif self.body is not None and tag in {'p', 'div', 'blockquote'}:
                self.current['parts'].append('\n')
            if position + 1 <= self.outer:
                self.current['text'] = ''.join(self.current.pop('parts')).strip()
                self.records.append(self.current)
                self.current, self.outer, self.body = None, None, None
        del self.stack[position:]


def is_news_bridge(news):
    parsed = urlparse(news.url)
    return parsed.scheme == 'https' and parsed.hostname == 'steamstore-a.akamaihd.net' and parsed.username is None and parsed.port in {None,443} and (
        parsed.path.rstrip('/') == '/news/externalpost/steam_community_announcements/'+news.news_id)


def is_community_announcement(news):
    """Steam's news API also includes press RSS. Those are not Event threads."""
    try:
        if is_news_bridge(news):
            return True
        parsed = urlparse(news.url)
        return (parsed.scheme == 'https' and parsed.hostname == 'steamcommunity.com' and
                parsed.username is None and parsed.port in {None, 443} and bool(re.fullmatch(
                    r'/(?:ogg/\d+|games/[^/]+|groups/[^/]+)/announcements/detail/\d+/?', parsed.path)))
    except ValueError:
        return False


def announcement_id(news, *, resolved_url=None):
    parsed = urlparse(resolved_url or news.url)
    match = re.search(r'/announcements/detail/(\d+)(?:/|$)', parsed.path)
    if parsed.scheme != 'https' or parsed.hostname != 'steamcommunity.com' or not match:
        raise DiscussionError('unsupported_announcement_url', '该新闻不是可定位的Steam社区公告，不能猜测讨论主题ID')
    return match[1]


def _get(client, url, *, allow_bridge=False):
    for _ in range(5):
        parsed = urlparse(str(url))
        bridge = allow_bridge and parsed.hostname == 'steamstore-a.akamaihd.net' and bool(re.fullmatch(r'/news/externalpost/steam_community_announcements/\d+/?', parsed.path))
        if parsed.scheme != 'https' or parsed.username is not None or (parsed.hostname not in {'steamcommunity.com', 'store.steampowered.com'} and not bridge) or parsed.port not in {None, 443}:
            raise DiscussionError('unsafe_redirect', 'Steam响应重定向到非允许地址')
        response = client.get(url, follow_redirects=False)
        if response.is_redirect:
            url = str(response.url.join(response.headers['location']))
            continue
        response.raise_for_status()
        if len(response.content) > 8_000_000:
            raise DiscussionError('page_too_large', '公开页面超出读取上限')
        return response
    raise DiscussionError('redirect_limit', 'Steam重定向次数超限')


def resolve_event(client, news):
    if is_news_bridge(news):
        page = _get(client, news.url, allow_bridge=True)
        gid = announcement_id(news, resolved_url=str(page.url))
    else:
        gid = announcement_id(news)
        page = _get(client, f'https://steamcommunity.com/ogg/{news.appid}/announcements/detail/{gid}?l=english')
        if announcement_id(news, resolved_url=str(page.url)) != gid:
            raise DiscussionError('identity_mismatch', '公告直链重定向到了另一篇公告')
    metadata = PageMetadata(page.text)
    events = metadata.attributes.get('data-partnereventstore', [])
    event = next((e for e in events if str(e.get('announcement_body', {}).get('gid')) == gid), None)
    community = metadata.attributes.get('data-community', {})
    if event is None and community.get('APPID') == news.appid and community.get('CLANACCOUNTID'):
        # Verified read-only store endpoint fallback for pages missing embedded events.
        response = _get(client, 'https://store.steampowered.com/events/ajaxgetpartnerevent?' +
            f'clan_accountid={int(community["CLANACCOUNTID"])}&announcement_gid={gid}&lang_list=0')
        payload = response.json()
        if payload.get('success') == 1:
            event = payload.get('event')
    if not event:
        raise DiscussionError('event_mapping_unavailable', '公告页面未提供可核对的活动/讨论映射')
    body = event.get('announcement_body', {})
    if event.get('appid') != news.appid or str(body.get('gid')) != gid:
        raise DiscussionError('identity_mismatch', '公告详情的游戏或公告ID与目标不一致')
    timestamp = body.get('posttime')
    if not timestamp or datetime.fromtimestamp(timestamp, timezone.utc).date() != news.published_at.astimezone(timezone.utc).date():
        raise DiscussionError('publication_mismatch', '公告详情的UTC发布日期与已定位公告不一致')
    topic = str(event.get('forum_topic_id') or body.get('forum_topic_id') or '0')
    if topic == '0' or event.get('comment_type', 'ForumTopic') != 'ForumTopic':
        raise DiscussionError('no_supported_discussion', '该公告没有公开的Event讨论主题，或讨论类型暂不支持')
    mapping = {'announcement_gid': gid, 'event_gid': str(event['gid']), 'topic_id': topic,
        'news_id': news.news_id, 'news_feed_url': news.url,
        'owner': str(event['clan_steamid']), 'forum_id': str(event['gidfeature']),
        'published_at_timestamp': timestamp, 'announcement_url': str(page.url),
        'thread_url': f'https://steamcommunity.com/app/{news.appid}/eventcomments/{topic}/'}
    if any(not str(mapping[k]).isdigit() or int(mapping[k]) <= 0 for k in ('event_gid', 'topic_id', 'owner', 'forum_id')):
        raise DiscussionError('invalid_mapping', '活动或论坛标识无效')
    return mapping


def thread_configuration(html, news, mapping):
    marker = re.search(r'InitializeCommentThread\(\s*"ForumTopic"\s*,\s*"[^"]+"\s*,\s*', html)
    if not marker:
        raise DiscussionError('discussion_unavailable', '讨论页面不可匿名读取，或Steam页面结构已变化')
    config, _ = json.JSONDecoder().raw_decode(html[marker.end():])
    extended = json.loads(config.get('extended_data', '{}'))
    if (str(config.get('feature2')) != mapping['topic_id'] or str(config.get('owner')) != mapping['owner'] or
            str(config.get('feature')) != mapping['forum_id'] or extended.get('forum_appid') != news.appid or
            extended.get('forum_type') != 'Event'):
        raise DiscussionError('identity_mismatch', '讨论主题/游戏/论坛与目标公告不一致')
    description = PageMetadata(html).meta.get('description', '')
    if not re.search(r'/announcements/detail/' + re.escape(mapping['announcement_gid']) + r'(?:\s|[/?#]|$)', description):
        raise DiscussionError('backlink_mismatch', '讨论主题未明确回链到目标公告')
    session = re.search(r'g_sessionID\s*=\s*"([a-f0-9]+)"', html)
    if not session:
        raise DiscussionError('anonymous_session_unavailable', '缺少匿名分页上下文')
    return config, session[1]


def fetch_announcement_comments(news, target_count=DEFAULT_COMMENT_TARGET, *, client=None, request_interval=.3):
    if not 1 <= target_count <= MAX_COMMENT_TARGET:
        raise ValueError(f'公告留言采集上限必须在1..{MAX_COMMENT_TARGET}')
    if client is None:
        with httpx.Client(timeout=25, headers={'User-Agent': 'GameLens/1.0 (public discussion reader)'}) as anonymous:
            return fetch_announcement_comments(news, target_count, client=anonymous, request_interval=request_interval)
    started = time.monotonic()
    mapping = resolve_event(client, news)
    page = _get(client, mapping['thread_url'] + '?l=english')
    config, session = thread_configuration(page.text, news, mapping)
    totals, hashes, records, seen = [], [], [], set()
    start, pages, failed, unreadable, duplicates = 0, 0, None, 0, 0
    html, total, page_size = page.text, page_integer(config['total_count'],'total_count'), page_integer(config['pagesize'],'pagesize')
    while pages < MAX_COMMENT_PAGES:
        if type(total) is not int or total < 0 or type(page_size) is not int or page_size < 0:
            raise DiscussionError('invalid_page', '分页总数或页大小无效')
        parsed = CommentHTML(html).records
        if len(parsed) != min(page_size, max(0, total - start)):
            raise DiscussionError('invalid_page', '返回的留言数量与分页声明不符，不能视为完整采集')
        totals.append(total); hashes.append(hashlib.sha256(html.encode('utf-8')).hexdigest()); pages += 1
        raw = config.get('comments_raw', {}) if start == 0 else {}
        for item in parsed:
            cid = item['comment_id']
            if cid in seen:
                duplicates += 1
                continue
            seen.add(cid)
            moment = item.get('created_at_timestamp')
            if not item['text'] or not moment:
                unreadable += 1  # Image-only/removed messages are not invented text.
                continue
            record = AnnouncementComment(appid=news.appid, news_id=news.news_id, **{k:mapping[k] for k in ('announcement_gid','topic_id')},
                **item, author=raw.get(cid, {}).get('author', ''),
                created_at=datetime.fromtimestamp(moment, timezone.utc), source_url=mapping['thread_url']+'#c'+cid)
            records.append(record.model_dump(mode='json'))
        start += len(parsed)
        if duplicates:
            failed = 'unstable_pagination'
            break
        if len(records) >= target_count or start >= total:
            break
        try:
            if time.monotonic() - started > 130:
                failed = 'collection_deadline'
                break
            if request_interval:
                time.sleep(request_interval)
            response = client.post(f'https://steamcommunity.com/comment/ForumTopic/render/{mapping["owner"]}/{mapping["forum_id"]}/',
                data={'sessionid': session, 'feature2': mapping['topic_id'], 'extended_data': config['extended_data'],
                      'start': start, 'count': min(COMMENT_PAGE_SIZE, target_count-len(records)), 'oldestfirst': 'true'}, follow_redirects=False)
            response.raise_for_status()
            if response.is_redirect or len(response.content) > 8_000_000:
                raise DiscussionError('invalid_page', '分页跳转或响应过大')
            payload = response.json()
            expected_name = 'ForumTopic_'+mapping['owner']+'_'+mapping['forum_id']+'_'+mapping['topic_id']
            if payload.get('success') not in (True, 1) or page_integer(payload.get('start'),'start') != start or payload.get('name') != expected_name:
                raise DiscussionError('invalid_page', '讨论分页失败或偏移不符')
            html, total, page_size = payload['comments_html'], page_integer(payload['total_count'],'total_count'), page_integer(payload['pagesize'],'pagesize')
            if page_size <= 0:
                raise DiscussionError('pagination_stalled', '仍有留言但分页未返回新内容')
        except (httpx.HTTPError, ValueError, KeyError) as error:
            failed = getattr(error, 'code', type(error).__name__)
            break
    records = records[:target_count]
    source_exhausted = not failed and len(set(totals)) == 1 and start >= totals[-1]
    complete = source_exhausted and unreadable == 0 and len(records) == totals[-1]
    return {'schema_version': 1, 'appid': news.appid, 'news_id': news.news_id,
        'mapping': mapping, 'binding_verified': True, 'comments': records,
        'fetched_at': datetime.now(timezone.utc).isoformat(),
        'coverage': {'status': 'complete_thread_snapshot' if complete else 'partial_thread_snapshot',
            'complete': complete, 'source_exhausted': source_exhausted,
            'all_bodies_readable': unreadable == 0,
            'scope': 'announcement_discussion_only', 'sample_order': 'oldest_first',
            'reported_total': totals[-1], 'collected_count': len(records), 'target_count': target_count,
            'pages_read': pages, 'has_more': not source_exhausted, 'counts_changed_during_fetch': len(set(totals)) > 1,
            'unreadable_message_count': unreadable,
            'duplicate_message_count': duplicates,
            'page_error_code': failed, 'page_content_hashes': hashes,
            'warning': '仅为公开讨论区采集时快照；留言可能删除/编辑，且不代表全部玩家。主题关联不证明每条留言描述本次更新。'}}
