"""Exact announcement lookup and independent discussion collection/query tools."""
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

from gamelens.analysis.version import load_patch_candidates
from gamelens.domain.announcement_comments import AnnouncementComment
from gamelens.storage.atomic_io import atomic_write_json
from gamelens.tools.investigation import ListPatchesArgs, ReviewInvestigation
from gamelens.tools.time_scope import requested_patch_date, date_matches, patch_day
from gamelens.tools.steam_announcement_comments import (
    DEFAULT_COMMENT_TARGET, MAX_COMMENT_TARGET, is_community_announcement,
)
from gamelens.tools.cache_coverage import cache_satisfies_target, candidate_improves_cache
from gamelens.core.local_time import local_datetime, local_iso, local_day, local_midnight


class DiscussionArgs(BaseModel):
    model_config = ConfigDict(extra='forbid')
    appid: int = Field(gt=0)
    news_id: str = Field(pattern=r'^\d+$', max_length=100, description='list_announcements返回的news_id，必须明确选择公告；不是活动ID/主题ID。')


class PrepareDiscussionArgs(DiscussionArgs):
    target_count: int = Field(default=DEFAULT_COMMENT_TARGET, ge=1, le=MAX_COMMENT_TARGET)
    refresh: bool = False


class QueryDiscussionArgs(DiscussionArgs):
    offset: int = Field(default=0, ge=0, le=MAX_COMMENT_TARGET)
    limit: int = Field(default=20, ge=1, le=50)
    order: Literal['spread', 'oldest', 'recent'] = 'spread'
    start_at: datetime | None = None
    end_at: datetime | None = None

    @model_validator(mode='after')
    def dates(self):
        for name in ('start_at', 'end_at'):
            moment = getattr(self,name)
            if moment is not None:
                setattr(self,name,local_datetime(moment))
        if self.start_at and self.end_at and self.start_at >= self.end_at:
            raise ValueError('结束时间必须晚于开始时间')
        return self


class GetDiscussionArgs(DiscussionArgs):
    comment_id: str = Field(pattern=r'^\d+$', max_length=100)


def discussion_fetch_approval(args, state):
    if args.refresh or args.target_count > DEFAULT_COMMENT_TARGET:
        return f'将匿名刷新或采集该公告讨论区最多{args.target_count}条留言，不保证完整覆盖；是否继续？'
    return None


class AnnouncementDiscussionTools:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def sources(self, appid, state):
        if appid != state['appid']:
            raise ValueError('工具AppID必须与当前调查一致')
        paths = [self.root/'data/raw'/f'{appid}_news.json', self.root/'data/raw'/f'{appid}_patch_candidates.json']
        records, existing = {}, []
        for path in paths:
            if path.is_file():
                existing.append(path)
                records.update({n.news_id: n for n in load_patch_candidates(path) if n.appid == appid})
        if not existing:
            raise ValueError('缺少公告快照，请先prepare_patches')
        return records, existing

    def target(self, args, state):
        records, paths = self.sources(args.appid, state)
        if args.news_id not in records:
            raise ValueError('所选公告不在该游戏的已定位公告快照中')
        news = records[args.news_id]
        if not is_community_announcement(news):
            raise ValueError('所选记录是外部新闻，不是可定位留言的Steam官方社区公告；须重新list_announcements定位')
        import re
        goal = state.get('goal', '')
        if re.search(r'YYYY-MM-DD|某日期', goal, re.I):
            raise ValueError('请先填写具体公告日期，不能把日期占位符当成最近一次更新')
        target = requested_patch_date(state.get('goal', ''))
        if target:
            matches = [n for n in records.values() if is_community_announcement(n) and date_matches(n, target)]
            if len({patch_day(n)[:4] for n in matches}) > 1:
                raise ValueError('该日期跨多个年份，必须先向用户澄清年份')
            if not date_matches(news, target):
                raise ValueError('采集目标公告的北京时间发布日期与用户指定日期不一致')
            self.check_choice(matches, news, goal)
            # Do not silently choose one of several announcements on the same day.
            selected = [o.get('data', {}) for o in state.get('observations', []) if o.get('ok') and
                o.get('tool') == 'list_announcements' and o.get('data', {}).get('requested_date') == target]
            exact = any(d.get('selected_news_id') == args.news_id or
                d.get('matched_count') == 1 and any(n.get('news_id') == args.news_id for n in d.get('announcements', [])) for d in selected)
            if not exact:
                raise ValueError('须先list_announcements按日期定位；同日多条公告须用明确news_id或标题进一步选择/询问用户')
        return news, paths

    @staticmethod
    def check_choice(matches, news, goal):
        if len(matches) <= 1:
            return
        import re
        if re.search(r'(?<!\d)' + re.escape(news.news_id) + r'(?!\d)', goal) or news.title.casefold() in goal.casefold():
            return
        patches = [n for n in matches if n.is_patch_candidate]
        if re.search(r'更新|版本|补丁', goal) and not re.search(r'活动', goal) and len(patches) == 1 and patches[0].news_id == news.news_id:
            return
        raise ValueError('同一天存在多条目标公告，请先让用户明确公告标题或ID，不得擅自选择第一条')

    @staticmethod
    def source(path, paths=()):
        return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'scope': 'announcement_discussion_only',
            'reference_files': [{'path': str(p.resolve()), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths if p != path]}

    def save(self, tool, data, state):
        result = ReviewInvestigation(self.root).save(tool, data, state)
        if tool == 'list_announcements':
            result.summary = f'公告定位：{data.get("lookup_status")}，匹配{data.get("matched_count",0)}条公告；按北京时间日期/真实ID核对'
        elif data.get('status') in {'reused_after_refresh_failure','reused_after_refresh_shortfall'}:
            result.summary = (f'公告讨论区补采未改善缓存，已保留原{data.get("collected_count",0)}条留言；'
                f'原因：{data.get("refresh_error_code") or "candidate_not_better"}')
        elif data.get('status') in {'collected','reused'}:
            result.summary = f'公告讨论区已采集/复用{data.get("collected_count",0)}条留言；下一步读取原文，不是商店评价'
        elif data.get('status') == 'collected_partial':
            result.summary = (f'公告讨论区仅采集到{data.get("collected_count",0)}条留言，覆盖不完整；'
                f'原因：{data.get("refresh_error_code") or "unknown"}')
        else:
            result.summary = f'公告讨论区读取：{data.get("status", "ready")}，返回{data.get("returned_count", 0)}条留言；不是商店评价'
        return result

    def list_announcements(self, args: ListPatchesArgs, state):
        records, paths = self.sources(args.appid, state)
        goal_date = requested_patch_date(state.get('goal', ''))
        import re
        if re.search(r'YYYY-MM-DD|某日期', state.get('goal', ''), re.I):
            raise ValueError('请用户填写具体日期后再定位公告')
        target = goal_date or args.published_on
        if goal_date and args.published_on and (args.published_on != goal_date if len(goal_date) == 10 else args.published_on[-5:] != goal_date):
            raise ValueError('公告定位日期与用户要求不一致')
        eligible = [n for n in records.values() if is_community_announcement(n)]
        external = [n for n in records.values() if not is_community_announcement(n) and
            (not target or date_matches(n, target))]
        matches = sorted((n for n in eligible if (not target or date_matches(n, target)) and
            (not args.news_id or n.news_id == args.news_id) and
            (not args.title_query or args.title_query.casefold() in n.title.casefold())),
            key=lambda n:(n.published_at_timestamp,n.news_id), reverse=True)
        years = sorted({patch_day(n)[:4] for n in matches})
        ambiguous_year = bool(target and len(target) == 5 and len(years) > 1)
        selected = matches[args.offset:args.offset+args.limit]
        data = {'source': self.source(paths[0], paths[1:]), 'requested_date': target,
            'date_basis': 'beijing publication day', 'matched_count': len(matches), 'returned_count': len(selected),
            'offset': args.offset, 'has_more': args.offset+len(selected) < len(matches),
            'needs_year_clarification': ambiguous_year, 'needs_announcement_clarification': bool(target and len(matches) > 1),
            'selected_news_id': matches[0].news_id if len(matches) == 1 and not ambiguous_year else None,
            'matched_years': years, 'lookup_status': 'ambiguous_year' if ambiguous_year else
                'ambiguous_announcement' if len(matches) > 1 and target else 'found_in_cache' if matches else 'not_found_in_cache',
            'announcements': [{**n.model_dump(mode='json'), 'published_at_local':local_iso(n.published_at_timestamp), 'published_on_local':patch_day(n)} for n in selected],
            'excluded_external_news_count': len(external),
            'caution': '只定位Steam社区公告（含补丁与活动），媒体新闻不是官方公告；缓存缺失不代表不存在。按北京时间日历定位，同日多公告不要擅自选第一条。'}
        # Expose adjacent official dates as clarification choices, NEVER select one
        # for a different requested day or bind its discussion automatically.
        if not matches and target and len(target) == 10:
            day = datetime.fromisoformat(target).date()
            nearby = sorted((n for n in eligible if abs((local_datetime(n.published_at_timestamp).date()-day).days) <= 1),
                            key=lambda n: abs(n.published_at_timestamp-int(local_midnight(target).timestamp())))[:4]
            data['nearby_announcements'] = [{'news_id': n.news_id, 'title': n.title,
                'published_at_local': local_iso(n.published_at_timestamp),
                'published_on_local': patch_day(n),
                'url': n.url} for n in nearby]
            data['needs_date_clarification'] = bool(nearby)
            data['next_step'] = ('请确认邻近官方公告的北京时间日期/标题或news_id；不得自动改用邻近日公告或外部新闻。' if nearby else
                '可审批后按目标日期有限补采官方公告；仍缺失则说明未定位，不能当作没有留言。')
        data['source']['scope'] = 'cached_announcements_only'
        return self.save('list_announcements', data, state)

    def path(self, args):
        return self.root/'data/raw/announcement_comments'/str(args.appid)/f'{args.news_id}.json'

    def prepare_announcement_comments(self, args, state):
        from gamelens.tools.steam_announcement_comments import fetch_announcement_comments, DiscussionError
        import httpx
        news, paths = self.target(args, state)
        path = self.path(args)
        prior = self.load(args, state)[0] if path.is_file() else None
        prior_coverage = prior.get('coverage', {}) if prior else {}
        prior_count = len(prior.get('comments', [])) if prior else 0
        prior_complete = bool(prior_coverage.get('source_exhausted', prior_coverage.get('complete')))
        cache_adequate = bool(prior and cache_satisfies_target(
            prior_count, args.target_count, complete=prior_complete))
        refresh_attempted = bool(args.refresh or prior is None or not cache_adequate)
        retained = False
        refresh_error = None
        status = 'reused'
        try:
            if prior is not None and not refresh_attempted:
                snapshot = prior
            else:
                candidate = fetch_announcement_comments(news, args.target_count)
                candidate_coverage = candidate.get('coverage', {})
                candidate_count = len(candidate.get('comments', []))
                candidate_complete = bool(candidate_coverage.get('source_exhausted', candidate_coverage.get('complete')))
                candidate_error = (candidate_coverage.get('page_error_code') or
                    ('counts_changed_during_fetch' if candidate_coverage.get('counts_changed_during_fetch') else None))
                candidate_adequate = cache_satisfies_target(
                    candidate_count, args.target_count, complete=candidate_complete)
                replace = (prior is None or
                    args.refresh and candidate_error is None and candidate_adequate or
                    candidate_improves_cache(prior_count, candidate_count, args.target_count,
                        existing_complete=prior_complete, candidate_complete=candidate_complete,
                        existing_error=prior_coverage.get('page_error_code'), candidate_error=candidate_error))
                if replace:
                    snapshot = candidate
                    atomic_write_json(path, snapshot)
                    refresh_error = candidate_error
                    status = 'collected_partial' if candidate_error else 'collected'
                else:
                    snapshot = prior
                    retained = True
                    refresh_error = candidate_error or 'candidate_not_better'
                    status = 'reused_after_refresh_shortfall'
        except (DiscussionError, httpx.HTTPError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            error_code = getattr(error, 'code', type(error).__name__)
            if prior is not None:
                snapshot = prior
                retained = True
                refresh_error = error_code
                status = 'reused_after_refresh_failure'
            else:
                data = {'source': self.source(paths[0], paths[1:]), 'status': error_code, 'news_id': args.news_id,
                'announcements': [news.model_dump(mode='json')], 'body_read_status':'unavailable',
                'coverage': {'status': 'unavailable', 'complete': False, 'collected_count': None},
                    'error_code': error_code, 'refresh_attempted': True,
                    'retained_previous_snapshot': False,
                    'caution': '未能读取公告留言，不代表0条留言；不得用商店评价替代该来源。'}
                result = self.save('prepare_announcement_comments', data, state)
                result.summary = '公告讨论区不可用：' + error_code + '；未采集不等于没有留言'
                return result  # A traceable limitation; planner can verify/report unknown.
        reused = snapshot is prior
        selected_coverage = snapshot['coverage']
        selected_count = len(snapshot['comments'])
        cache_adequate = cache_satisfies_target(selected_count, args.target_count,
            complete=bool(selected_coverage.get('source_exhausted', selected_coverage.get('complete'))))
        data = {'source': self.source(path, paths), 'status': status,
            'news_id': args.news_id, 'binding_verified': snapshot['binding_verified'],
            'mapping': snapshot['mapping'], 'coverage': selected_coverage, 'reused': reused,
            'fetched_at': snapshot.get('fetched_at'),
            'body_read_status':'not_read', 'collected_count': selected_count,
            'requested_count': args.target_count, 'cache_adequate': cache_adequate,
            'refresh_attempted': refresh_attempted, 'retained_previous_snapshot': retained,
            'refresh_error_code': refresh_error, 'error_code': refresh_error,
            'announcements': [news.model_dump(mode='json')],
            'next_step': 'query_announcement_comments读取独立留言原文，不能将采集成功当成反馈分析完成'}
        data['next_read_arguments'] = {'appid':args.appid,'news_id':args.news_id}
        return self.save('prepare_announcement_comments', data, state)

    def load(self, args, state):
        news, paths = self.target(args, state)
        path = self.path(args)
        if not path.is_file():
            raise ValueError('该公告尚未采集讨论留言，请prepare_announcement_comments')
        snapshot = json.loads(path.read_text(encoding='utf-8'))
        if snapshot.get('appid') != args.appid or snapshot.get('news_id') != args.news_id or snapshot.get('binding_verified') is not True:
            raise ValueError('公告留言缓存身份/关联无效')
        from gamelens.tools.steam_announcement_comments import announcement_id, is_news_bridge
        mapping = snapshot.get('mapping', {})
        resolved = mapping.get('announcement_url') if is_news_bridge(news) else None
        if mapping.get('news_feed_url') != news.url or mapping.get('news_id') != news.news_id or mapping.get('announcement_gid') != announcement_id(news, resolved_url=resolved):
            raise ValueError('公告缓存映射与原始公告不一致')
        coverage = snapshot.get('coverage', {})
        count = len(snapshot.get('comments', []))
        if type(coverage.get('collected_count')) is not int or coverage['collected_count'] != count or type(coverage.get('reported_total')) is not int:
            raise ValueError('留言缓存数量与覆盖元数据不一致')
        if coverage.get('complete') and (coverage['reported_total'] != count or coverage.get('page_error_code') or coverage.get('counts_changed_during_fetch') or coverage.get('has_more')):
            raise ValueError('完整留言快照声明与采集覆盖不一致')
        source_exhausted = coverage.get('source_exhausted', bool(coverage.get('complete')))
        if type(source_exhausted) is not bool:
            raise ValueError('留言缓存来源遍历状态无效')
        if source_exhausted and (coverage.get('page_error_code') or
                coverage.get('counts_changed_during_fetch') or coverage.get('has_more') or
                coverage['reported_total'] != count + int(coverage.get('unreadable_message_count', 0))):
            raise ValueError('已遍历留言快照声明与采集覆盖不一致')
        return snapshot, paths, path

    def query_announcement_comments(self, args, state):
        snapshot, paths, path = self.load(args, state)
        records = [AnnouncementComment.model_validate(item).model_dump(mode='json') for item in snapshot['comments']]
        if len({r['comment_id'] for r in records}) != len(records) or any(r['news_id'] != args.news_id or
                r['appid'] != args.appid or r['topic_id'] != snapshot['mapping']['topic_id'] for r in records):
            raise ValueError('留言所属游戏/公告/主题不一致，或缓存包含重复留言ID')
        records = [r for r in records if (args.start_at is None or r['created_at_timestamp'] >= args.start_at.timestamp()) and
            (args.end_at is None or r['created_at_timestamp'] < args.end_at.timestamp())]
        records.sort(key=lambda r:(r['created_at_timestamp'],r['comment_id']), reverse=args.order == 'recent')
        if args.order == 'spread' and args.offset == 0 and len(records) > args.limit:
            indexes = [round(i*(len(records)-1)/max(1,args.limit-1)) for i in range(args.limit)]
            selected = [records[i] for i in indexes]
        else:
            selected = records[args.offset:args.offset+args.limit]
        data = {'source': self.source(path, paths), 'status': 'ready', 'news_id': args.news_id,
            'binding_verified': True, 'mapping': snapshot['mapping'], 'coverage': snapshot['coverage'],
            'fetched_at': snapshot.get('fetched_at'),
            'filters': {'start_at': args.start_at.isoformat() if args.start_at else None, 'end_at': args.end_at.isoformat() if args.end_at else None},
            'matched_count': len(records), 'returned_count': len(selected), 'offset': args.offset, 'order': args.order,
            'has_more': len(records) > len(selected) if args.order == 'spread' and args.offset == 0 else args.offset+len(selected) < len(records),
            'next_offset': None if args.order == 'spread' and args.offset == 0 else args.offset+len(selected),
            'comments': selected,
            'caution': '按公告关联精确定位；均匀位置展示不是随机或语义抽样，也不是顺序第一页。要完整分页用order=oldest/recent与next_offset。不能推算观点占比。没有游玩时长、好差评投票或官方语言标签。引用块不是发言者本人观点。'}
        return self.save('query_announcement_comments', data, state)

    def get_announcement_comment(self, args, state):
        snapshot, paths, path = self.load(args, state)
        record = next((r for r in snapshot['comments'] if r['comment_id'] == args.comment_id), None)
        if record is None:
            raise ValueError('该公告缓存没有指定留言ID')
        record = AnnouncementComment.model_validate(record).model_dump(mode='json')
        if record['news_id'] != args.news_id or record['appid'] != args.appid or record['topic_id'] != snapshot['mapping']['topic_id']:
            raise ValueError('指定留言的公告/主题归属错误')
        return self.save('get_announcement_comment', {'source': self.source(path, paths), 'news_id': args.news_id,
            'binding_verified': True, 'mapping': snapshot['mapping'], 'coverage': snapshot['coverage'],
            'returned_count': 1, 'comments': [record]}, state)
