"""Readable human review cards; never approve labels or invoke models.

Only frozen records are used. Suggested snippets do NOT become gold labels.
Unspecified or contradictory task inputs are reported, not silently repaired.
"""
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from gamelens.core.local_time import local_iso
import json
import re

GAMES = {1623730: '幻兽帕鲁', 251570: '七日杀', 730: 'Counter-Strike 2'}
ROLES = {'support': '支持证据（预标注，需你确认）', 'counterexample': '相反体验（预标注，需你确认）',
         'partial': '部分相关（预标注，需你确认）'}
LANGUAGES = {'schinese': '简体中文', 'tchinese': '繁体中文', 'english': '英文'}
TERMS = {
    'performance': ['卡顿', '掉帧', 'fps', 'stutter'],
    'building': ['建造', '建筑', '地图', 'building'],
    'progress': ['存档', '进度', 'save'],
    'negation': ['流畅', '不卡', '不再', 'smooth'],
    'network': ['掉线', '延迟', 'disconnect', 'network'],
    'official_fixes': ['fix', 'stability', '修复'],
    'activity': ['event', '周年', '活动', 'sale'],
    'no_answer': ['内核驱动泄漏', 'kernel driver leak'],
}
POINTS = {
 'locate': [
  ['在固定评测时钟下定位最近的官方补丁，说明标题、发布时间和公告 ID。', '媒体报道和促销活动不能代替补丁；这里的“最近”不是审核当天的最新消息。'],
  ['按问题中的日期找官方更新，并根据该公告正文说明改动。', '若同日存在多份更新，不可不经说明随便选一份。'],
  ['分别指出官方补丁和媒体新闻的来源，保留标题、时间和来源链接。', '媒体提到某版本，不代表它就是官方补丁公告。'],
  ['定位官方活动公告，说明其正文是活动或促销内容，而非软件修复。', '不能用活动发布时间作为最近补丁的时间。'],
  ['当用户没有指定同一天的哪份公告时，应列出候选并询问用户。', '如果该日实际上只有一份公告，则此题缺少“多份公告”的成立前提。']],
 'statistics': [
  ['以全部符合语言条件的冻结评论为分母，给出数量、推荐数和推荐率。', '推荐率是快照中的推荐票比例，不是全 Steam 玩家好评率。'],
  ['只统计简体中文且评价发表时游玩超过100小时的评论；100小时整不算。', '不能用当前累计时长替代评价时长。'],
  ['分别报告“评价时超过100小时”和“当前累计超过100小时”的两组统计。', '同一评论可以同时属于两组；这不是两个互斥人群。'],
  ['严格使用题目中的开始时间和结束不含的时间边界，报告样本量和推荐率。', '不能丢弃时间中的时分秒，也不能扩大到整天。'],
  ['绑定指定更新，分别计算发布时刻之前7天和之后7天的已采集评论推荐率。', '两侧覆盖不足必须说明，不能把零样本率写成0%，不能断言更新导致变化。'],
  ['明确冻结窗口内样本量为0，推荐率无法计算。', '空缓存窗口不证明 Steam 实际没有玩家反馈。']],
 'evidence': [
  ['分别列出性能问题原文和运行流畅的相反体验，保留否定与转折。', '若没有足够反例，明确说明缺失，不能把普通好评当成流畅证据。'],
  ['给出推荐票为推荐、但正文明确抱怨问题的评论。', '推荐不代表没有问题，不能只读推荐票判断体验。'],
  ['按原文区分启动阶段闪退、联机闪退与游玩过程退出。', '这些描述不支持认定为同一个根因；缺少启动证据时必须说明。'],
  ['解释“不再崩溃”“不卡”等否定表达实际指向什么体验。', '不能因为出现“崩溃”一词就把整句判为发生崩溃。'],
  ['引用建造或地图相关的具体体验，区分喜好、问题与原因推测。', '评论没有给出原因时不能补写技术根因。'],
  ['对照官方修复说明与玩家评论，区分承诺、修复声明和个人体验。', '官方宣布修复不证明所有玩家的问题都消失。'],
  ['检查评论创建和修改时间，明确当前正文能证明什么。', '若评论后来修改且无历史正文，不能用当前文字证明更新当日状态。']],
 'discussion': [
  ['只分析指定公告讨论区留言，区分赞同、质疑、具体问题和无关留言。', '所有归纳要有对应留言原文，不可混入商店评价。'],
  ['保留留言原文；有引用他人文字时区分引用和留言者本人的观点。', '不能把被引用的观点直接归属于留言作者。'],
  ['说明采集500条、讨论页报告558条，属于不完整的讨论快照。', '558是页面报告回复数，不是独立玩家数；无法代表全部留言立场比例。'],
  ['说明公告留言没有商店评价的推荐票与评价时长，不能据此筛选100小时或计算好评率。', '应说明能力限制，不能编造对应字段或悄悄改用商店评论。']],
 'conversation': [
  ['继承上一轮的目标更新，当前轮只保留中文评论；先核验历史绑定。', '历史模型文字不能直接当事实，不得偷偷改成最新版本。'],
  ['继承上一轮版本，使用评价发表时超过100小时的新条件。', '当前明确条件覆盖旧条件；不可替换为当前累计时长。'],
  ['保留上一轮的游戏、目标更新和时间窗口，仅把评论语言改为英文。', '当前明确语言条件覆盖上一轮语言；冻结窗口没有英文样本时，说明限制，不用其他语言评论代替。'],
  ['如果历史确有两个版本而用户只说“那个版本”，询问是哪一个。', '必须在测试输入中提供两个候选版本；不能只有文字声称有两个。']],
 'reliability': [
  ['模拟初始缓存缺失，读取被冻结的官方响应，再定位更新并绑定正确目标。', '不真实联网；来源响应由已有缓存派生，不能说这是新采集的实时官方结果。'],
  ['模拟目标评论窗口缓存不足，尝试读取冻结采集响应并检查覆盖。', '不得把缺少资料解释为零玩家反馈；必须有指定窗口与补采响应。'],
  ['模拟官方接口超时，保留已有资料并解释哪些部分尚未确认。', '旧公告不能冒充已经核验的最新补丁，失败不等于没有补丁。'],
  ['模拟模型生成中断后恢复，复用可核验的证据并报告恢复或中断状态。', '未生成完整答案不可冒充完成；必须有中断状态及恢复输入。']],
}

# Editorial pre-annotations tied to exact frozen records, not generated facts.
# If another dataset lacks the ID or anchor phrase, the note is not emitted.
REVIEW_NOTES = {
 'task-evidence-1': [('review','237472418','优化卡顿','该玩家推荐游戏，但仍希望优化卡顿；可作性能问题支持证据。'),
  ('review','236007901','装完这几个后据点一百只帕鲁也不会卡','另一位玩家称装模组后据点不会卡。这是有额外条件的相反体验，不证明未装模组的官方版本流畅。')],
 'task-evidence-2': [('review','237472418','优化卡顿','“好玩”与“优化卡顿现象”出现在同一条推荐评论中，说明推荐票与问题抱怨可以同时成立。')],
 'task-evidence-3': [('review','237463593','联机一直闪退','该原文明确说联机闪退，不能归为启动闪退。'),
  ('review','237557982','一直UE崩溃','该原文称持续UE崩溃，但没说启动阶段。仅这两条不能确认同一原因，也不能填补启动闪退证据。')],
 'task-evidence-4': [('review','236007901','不会卡','“不会卡”是安装模组后的体验，否定的是卡顿发生；不能省略模组条件。'),
  ('review','237256769','以前是在4399上玩的','“不流畅”说的是以前在4399玩的体验，并非明确称当前游戏运行流畅；不宜直接当当前性能的强反例。')],
 'task-evidence-5': [('review','237254625','后悔一开始没选好建造地点','该评论认可建造玩法，但后悔最初选址，这是玩法决策体验，不是已确认的建筑功能故障。'),
  ('review','237230256','地图的模型和材质','该评论批评地图模型、材质和建造动作表现；不能由此确认某个底层技术根因。')],
 'task-evidence-6': [('announcement','1843481262703464','reduced resolution','该补丁声明修复部分玩家纹理低分辨率和重生后地图筛选图标问题；声明的范围有限，不是所有问题都已消失。'),
  ('review','237472418','优化卡顿','较晚的评论仍提到卡顿。但它未证明发生的就是该补丁修复的纹理问题，不应直接声称补丁修复失败。')],
 'task-evidence-7': [('review','237254625','后悔一开始','此评论有不同的发表与修改时间；当前正文能展示当前快照内容，不能重建修改之前的历史正文。')],
 'task-discussion-1': [('comment','801217828677039710','having a dope time','赞同样例：玩家说近一周玩得很开心；不能推定其支持公告的每项改动。'),
  ('comment','801217828677038715','micro transactions','质疑样例：玩家猜测微交易、付费DLC并担心创意工坊。这是玩家猜测，不是官方已确认的收费计划。'),
  ('comment','801217828677040207','discord auto-connecting','具体问题样例：询问如何关闭启动游戏时Discord自动连接。问题原文不能证明所有玩家都会发生此现象。'),
  ('comment','801217828677040699','zombie types','具体反馈样例：玩家说第28天仍出现基础僵尸，并问是否修复；这是个人体验，不是官方故障确认。')],
 'task-discussion-2': [('comment','801217828677041410','move to a tougher biome','前半段是引用其他玩家关于僵尸类型的提问；留言者自己的回复是“去更难的生物群系”。不能把引用中的体验归给回复者。')],
}


class PlainHTML(HTMLParser):
    def __init__(self):
        super().__init__(); self.parts = []; self.hidden = 0
    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'): self.hidden += 1
        if tag in ('p', 'div', 'br', 'li', 'h1', 'h2', 'h3'): self.parts.append('\n')
    def handle_endtag(self, tag):
        if tag in ('script', 'style'): self.hidden = max(0, self.hidden - 1)
        if tag in ('p', 'div', 'li', 'h1', 'h2', 'h3'): self.parts.append('\n')
    def handle_data(self, data):
        if not self.hidden: self.parts.append(data)


def display_text(raw):
    parser = PlainHTML(); parser.feed(raw)
    text = unescape(''.join(parser.parts))
    text = re.sub(r'\[/?(?:p|h[1-6]|list|olist)\]', '\n', text, flags=re.I)
    text = re.sub(r'\[/?(?:b|i|u|s|url|quote|code|img|previewyoutube|table|tr|td)(?:=[^\]]*)?\]', '', text, flags=re.I)
    text = text.replace('[*]', '\n- ')
    return text.strip()


def stamp(value):
    if not value: return '未记录'
    from gamelens.core.local_time import local_datetime
    return local_datetime(value).strftime('%Y-%m-%d %H:%M:%S 北京时间 UTC+8')


def filter_words(filters):
    words = []
    if filters.get('is_positive') is not None: words.append('推荐票：'+('只看推荐评论' if filters['is_positive'] else '只看不推荐评论'))
    if filters.get('language'): words.append('评论语言：' + LANGUAGES.get(filters['language'], filters['language']))
    if filters.get('min_playtime_hours') is not None:
        basis = '评价发表时' if filters.get('playtime_basis') == 'at_review' else '当前累计' if filters.get('playtime_basis') == 'lifetime' else '口径待确定'
        words.append(f"游玩时长：{basis}，{'严格超过' if filters.get('min_playtime_exclusive') else '不少于'}{filters['min_playtime_hours']}小时")
    if filters.get('start_at'): words.append('开始时间（包含）：' + filters['start_at'])
    if filters.get('end_at'): words.append('结束时间（不包含）：' + filters['end_at'])
    if filters.get('period') in ('before', 'after'): words.append('窗口：更新' + ('前' if filters['period']=='before' else '后'))
    if not words: words.append('没有指定额外语言、时长或日期条件；不能擅自增加条件。')
    return words


def population(rows, filters):
    from gamelens.core.local_time import local_datetime
    def epoch(v): return local_datetime(v).timestamp()
    result = []
    for r in rows:
        if filters.get('is_positive') is not None and r.get('is_positive')!=filters['is_positive']: continue
        if filters.get('language') and r.get('language') != filters['language']: continue
        if filters.get('start_at') and r['created_at_timestamp'] < epoch(filters['start_at']): continue
        if filters.get('end_at') and r['created_at_timestamp'] >= epoch(filters['end_at']): continue
        if filters.get('min_playtime_hours') is not None:
            field = 'playtime_at_review_minutes' if filters.get('playtime_basis')=='at_review' else 'playtime_minutes'
            minutes = r.get(field)
            if minutes is None: continue
            limit = filters['min_playtime_hours']*60
            if minutes < limit or filters.get('min_playtime_exclusive') and minutes == limit: continue
        result.append(r)
    return result


def stat_line(rows, title):
    count = len(rows); good = sum(bool(r['is_positive']) for r in rows)
    rate = f'{good/count*100:.2f}%' if count else '无法计算（分母为0）'
    return f'{title}：{count} 条，推荐 {good} 条，不推荐 {count-good} 条；推荐率 {rate}。'


def record_card(kind, r, note):
    key = 'review_id' if kind=='review' else 'comment_id' if kind=='comment' else 'news_id'
    raw = r.get('text', r.get('contents', ''))
    lines = [f"### 原文 {r.get(key, '未记录')} — {note}", '']
    if kind=='review':
        lines += [f"商店评论｜{LANGUAGES.get(r.get('language'), r.get('language','未记录'))}｜{'推荐' if r.get('is_positive') else '不推荐'}", '',
          f"发表：{stamp(r.get('created_at_timestamp'))}；修改：{stamp(r.get('updated_at_timestamp'))}。", '',
          f"评价时长：{r.get('playtime_at_review_minutes', '未知')}分钟；当前累计：{r.get('playtime_minutes','未知')}分钟。", '']
    elif kind=='announcement':
        lines += ['标题：' + r.get('title', '未记录'), '', '发布：' + stamp(r.get('published_at_timestamp')), '',
          '来源：' + r.get('feed_name', '未记录'), '', '来源链接：' + r.get('url', '未记录'), '']
    else: lines += ['公告讨论留言；公告ID：' + str(r.get('news_id')), '', '发表：'+stamp(r.get('created_at_timestamp')), '',
                    '含他人引用：' + ('是，需区分说话者' if r.get('contains_quoted_text') else '接口未标记'), '']
    if kind=='announcement':
        reading=display_text(raw)
        if len(reading)>2200:
            # End at a paragraph boundary; the complete original remains below.
            boundary=reading.rfind('\n',0,2200)
            reading=reading[:boundary] if boundary>0 else ''
            reading += '\n\n（长公告阅读节选；完整正文请展开下方。节选不是全部改动。）'
        lines += ['以下是去除HTML/BBCode标签后的阅读视图，不是译文；英文原文需按其实际含义核对。', '', reading, '',
          '<details><summary>展开完整原始正文（保留HTML）</summary>', '', '```text', raw, '```', '', '</details>', '']
    else:
        # Fence longer than any fence within the original avoids broken Markdown.
        fence = '`' * max(3, max((len(s) for s in raw.split() if s and set(s)=={'`'}), default=0)+1)
        lines += [fence+'text', raw, fence, '']
    return lines


def render_card(bundle, case, kind):
    c = case; game = bundle['corpus'].get(c['appid'], {'reviews': [], 'news': [], 'comments': []})
    rows = game.get('reviews', []); news = game.get('news', []); gaps = []; originals = []
    expectation = c.get('expectations', {}); filters = expectation.get('filters', c.get('filters', {}))
    category = c.get('category'); number = int(c['id'].rsplit('-', 1)[-1])
    if c.get('blocking'): gaps.append('绑定的冻结来源缺失；不能据此确认无答案。需补齐来源后重新制作证据。')
    lines = [f"# {c['id']}｜{'完整调查题' if kind=='investigations' else '检索题'}", '',
      '## 1. 用户问什么', '', c.get('goal', c.get('query', '')), '',
      f"游戏：{GAMES.get(c['appid'], '待确认游戏')}（AppID {c['appid']}）。", '',
      '这是一份预拟审核稿，不是Agent已经生成的答案。你不用检查哈希、编辑JSON或运行程序。', '',
      '## 2. 问题背景与筛选条件', '',
      '资料和评测时钟固定在：'+local_iso(bundle['manifest']['clock'])+'。“最近”按这个时钟理解，不按今天理解。', '',
      '日期与时间统一显示为北京时间UTC+8；原始时间戳不变。旧时区冻结包保留历史依据，不可与新包混用。', '']
    if c.get('review',{}).get('status')=='rejected':
        lines[2:2]=['**此题已按用户意见撤下，不再审核或运行；仅保留历史记录。**', '',
                    '撤下原因：'+c['review'].get('notes','用户撤下'), '']
    elif c.get('review',{}).get('status')=='reviewed':
        lines[2:2]=['**此题已由用户审核通过，无需重复审核。**', '']
    lines += ['- '+s for s in filter_words(filters)]
    if kind=='retrieval':
        source = {'reviews':'商店评论', 'announcements':'公告', 'both':'商店评论与公告'}.get(c.get('kind'), '题目指定来源')
        lines += ['- 检索来源：'+source + ('，只接受官方公告。' if c.get('source_constraint')=='official' else '。')]
    else:
        lines += ['- 来源：' + ('指定公告的讨论区，不是商店评论。' if category=='discussion' else '按本题要求使用商店评论或官方公告，不可互相冒充。')]
    if category=='conversation':
        lines += ['', '### 上一轮给系统的背景', '']
        history = c.get('history', [])
        if not history: gaps.append('多轮题缺少上一轮输入。')
        for h in history:
            lines += ['- 上一轮问题：'+h.get('goal','未提供'), '- 上一轮游戏：'+str(h.get('appid',c['appid'])),
              ('- 上一轮候选公告ID：' if h.get('role')=='candidate' else '- 上一轮目标公告ID：')+str(h.get('target_news_id')), '- 历史来源状态：'+('已核验' if h.get('verified') else '未核验，不能直接当事实'),
              *['- 上一轮'+s for s in filter_words(h.get('filters',{}))]]
            history_appid=h.get('appid',c['appid'])
            if history_appid!=c['appid']:
                gaps.append('同一会话不能切换游戏；请改为同游戏条件追问，或拆分成另一个新会话题。')
            historical_news=bundle['corpus'].get(history_appid,{}).get('news',[])
            target = next((n for n in historical_news if n['news_id']==h.get('target_news_id')), None)
            if target and history_appid!=c['appid']:
                lines += ['- 历史公告核验：'+target['title']+'；'+stamp(target.get('published_at_timestamp'))+'。本轮已换游戏，不能继承此公告。']
            elif target: originals.append(('announcement',target,'历史目标核验材料'))
            else: gaps.append('历史目标公告不在绑定资料中，无法核验目标。')
        if number==4 and len({h.get('target_news_id') for h in history})<2:
            gaps.append('题干说历史有两个版本，但实际输入只有一个；需补充第二轮及第二个版本。')
        if c.get('history_origin'):lines += ['', '以上对话为评测构造的背景，版本ID和原文来自真实冻结资料；不是实际用户聊天记录。']
    if category=='locate' and number==5:
        gaps.append('题干未给出具体日期，也未绑定该日的两份公告；目前只能审核澄清规则，不能批准为完整场景。')
    if category=='reliability':
        if c.get('scenario_description'):lines += ['', '### 本题的离线输入与异常场景', '', c['scenario_description']]
        if c.get('initial_conditions',{}).get('reviews_cache_absent'):
            lines += ['', '只清空临时测试目录内本游戏的评论缓存，再从冻结语料派生的回放响应恢复；不发起真实补采。这是缓存恢复行为检查，不证明已读取全部Steam评论。']
        if c.get('response_scenario')=='generation_interruption':
            lines += ['', '脚本规划器在已核验原文后注入生成超时，再通过真实SQLite Checkpoint恢复。模型HTTP请求为0；结果只证明流程恢复，不是模型语义准确率。']
        if c.get('initial_conditions',{}).get('news_cache_absent'):
            lines += ['', '离线初始场景：只清空临时测试目录中的本游戏公告缓存；冻结原始资料不会删除。随后读取冻结响应，检查缓存能否恢复。']
        if c.get('response_scenario')=='timeout':
            lines += ['', '离线异常场景：官方公告接口返回预设超时；必须识别为请求超时，并保留已有公告，不能确认实时最新版本。']
        if not c.get('oracle_plan'):
            gaps.append('异常场景尚未配置可执行的初始缺失/中断状态及完整回放步骤；可以审核行为规则，但不可批准运行场景。')
        if number==2 and not filters.get('start_at'): gaps.append('补采题尚未指定目标时间窗口。')
    lines += ['', '## 3. 预拟标准答案：应该回答哪些要点', '']
    points = POINTS.get(category, [[]])[number-1] if category in POINTS else []
    if kind=='retrieval':
        points = ['返回与问题直接相关的原文及来源ID；不要仅靠命中关键词判定相关。',
          '逐条确认相关程度：0=无关，1=弱相关，2=明确相关，3=直接回答；确认属于支持、相反或部分相关。',
          '保留否定与转折；不能从推荐票推断某个问题不存在。']
        if 'no_answer' in c['id']: points += ['“无答案”仍待你确认：需要检查冻结语料，而不是凭候选为空判定游戏没有这个问题。']
    if c.get('answer_points'):points=c['answer_points']
    lines += ['- '+p for p in points]
    absence=c.get('absence_review',{})
    absence_valid=(c.get('expected_no_answer') is True and absence.get('dataset_sha256')==bundle['manifest']['dataset_sha256']
        and absence.get('scope')=='frozen_corpus_only' and absence.get('document_count')==len(rows)+len(news)
        and bool(absence.get('notes')))
    if absence:
        lines += ['', '**无答案预核对记录（仍需你确认语义）：**', '',
                  f"本题绑定资料共 {len(rows)} 条商店评论、{len(news)} 份新闻/公告；只讨论这份资料中有没有直接证据。", '',
                  *['- '+note for note in absence.get('notes',[])], '',
                  '关键词扫描用于找近似候选，不是语义证明；下方的0分候选是容易混淆的材料，不是本题正相关答案。']
    notes = REVIEW_NOTES.get(c['id'], [])
    if kind=='retrieval' and 'negation' in c['id']: notes = REVIEW_NOTES['task-evidence-4']
    emitted=[]
    for rk,rid,anchor,meaning in notes:
        collection=rows if rk=='review' else news if rk=='announcement' else game.get('comments',[])
        key='review_id' if rk=='review' else 'news_id' if rk=='announcement' else 'comment_id'
        record=next((r for r in collection if str(r[key])==rid and anchor in r.get('text',r.get('contents',''))),None)
        if record:
            emitted.append(f'- 原文 {rid}：{meaning}')
            originals.append((rk,record,'答案要点对应原文；仍待人工确认'))
    if emitted: lines += ['', '**结合现有原文的具体预拟判断：**', '', *emitted]
    if category=='statistics':
        if number in (1,2,4,6):
            eligible = population(rows,filters)
            lines += ['', '**从冻结原始记录独立计算的待确认数字：**', '', stat_line(eligible,'符合本题条件的样本')]
            originals += [('review',r,'统计分母中的原文示例，不是全部样本') for r in eligible[:5]]
        elif number==3:
            for basis in ('at_review','lifetime'):
                f={**filters,'playtime_basis':basis}; eligible=population(rows,f)
                lines += ['', stat_line(eligible,'评价发表时超过100小时' if basis=='at_review' else '当前累计超过100小时')]
                originals += [('review',r,'对应时长组的原文示例') for r in eligible[:3]]
        elif number==5:
            from gamelens.core.local_time import shift_calendar_days
            target=next((n for n in news if n['news_id']==expectation.get('target_news_id')),None)
            if target:
                pivot=target['published_at_timestamp']; originals.append(('announcement',target,'比较窗口的目标更新'))
                for label,left,right in [('更新前',shift_calendar_days(pivot,-7),pivot),('更新后',pivot,shift_calendar_days(pivot,7))]:
                    eligible=[r for r in rows if left<=r['created_at_timestamp']<right]
                    lines += ['', f'{label}窗口：{stamp(left)} 至 {stamp(right)}（结束不含）。', '', stat_line(eligible,label)]
                    originals += [('review',r,label+'原文示例') for r in eligible[:3]]
            else: gaps.append('缺少目标更新，不能确定前后窗口。')
        lines += ['', '数字依据：逐条读取下方来源文件，按题目条件筛选；原文示例只帮助阅读，不代替全部统计分母。']
    labels=c.get('labels',expectation.get('evidence',[]))
    # Every scored label must be visible, even when a more useful topic-specific
    # candidate is shown as supplementary reading. Neither is auto-approved.
    if labels:
        for lab in labels:
            records=rows if lab['kind']=='review' else news; key='review_id' if lab['kind']=='review' else 'news_id'
            r=next((r for r in records if str(r[key])==lab['source_id']),None)
            if r: originals.append((lab['kind'],r,ROLES[lab['role']]+f"；相关性预标 {lab['relevance']}/3"))
            else: gaps.append('预标注来源不存在：'+lab['source_id'])
    if kind=='retrieval':
        if (not labels or not any(l.get('relevance',0)>0 for l in labels)) and not absence_valid and not c.get('blocking'):
            gaps.append('没有已标注的相关证据；不能据此确认无答案，需要补查并确认无答案边界。')
    elif category=='locate':
        from gamelens.tools.steam_news import retained_patch_candidate, is_official_announcement
        patches=sorted([n for n in news if retained_patch_candidate(n)], key=lambda n:(n['published_at_timestamp'],n['news_id']),reverse=True)
        target=next((n for n in news if n['news_id']==expectation.get('target_news_id')),None)
        selected=[target] if target else patches[:2]
        if number in (3,4):
            if number==3:
                selected=patches[:1]+[n for n in news if not is_official_announcement(n)][:1]
            else: selected=[n for n in news if is_official_announcement(n) and not retained_patch_candidate(n)][:2]
        originals += [('announcement',n,'候选公告，需核对类别及正文') for n in selected if n]
        if target: lines += ['', f"预拟目标：{target['title']}；{stamp(target['published_at_timestamp'])}；公告ID {target['news_id']}。"]
        if not selected: gaps.append('没有可供核对的公告原文。')
    elif category=='evidence':
        terms={1:TERMS['performance']+TERMS['negation'],2:['卡顿','bug','问题','闪退'],3:['启动','闪退','崩溃','crash'],
               4:TERMS['negation'],5:TERMS['building'],6:['bug','修复','问题'],7:[]}[number]
        selected=[r for r in rows if any(t.casefold() in r['text'].casefold() for t in terms)]
        if number==2:
            # This conjunction needs semantic pre-annotation, not the mere
            # occurrence of e.g. "问题" in "没有问题".
            labelled={l['source_id'] for l in labels if l['kind']=='review'}
            selected=[r for r in rows if r['is_positive'] and r['review_id'] in labelled]
        if number==7: selected=[r for r in rows if r.get('updated_at_timestamp',0)>r.get('created_at_timestamp',0)]
        # Sample both normal and problem words, not just the first matched records.
        if number==1:
            normal=[r for r in rows if any(t.casefold() in r['text'].casefold() for t in TERMS['negation'])]
            selected=selected[:3]+normal[:3]
        originals += [('review',r,'人工待判读的候选；关键词命中不能保证相关') for r in selected[:8]]
        if number==6:
            from gamelens.tools.steam_news import retained_patch_candidate
            originals += [('announcement',n,'官方修复声明，不能代表全部玩家体验') for n in news if retained_patch_candidate(n)][:1]
        if not selected: gaps.append('该主题尚无候选原文，需补查资料，不能编造观点。')
        lines += ['', '下列原文只是一组可审核候选，尚未按语义确认支持或反例；请删除不相关候选并指出必须保留的证据。']
    elif category=='discussion':
        target_id=expectation.get('discussion_news_id')
        target=next((n for n in news if n['news_id']==target_id),None)
        if target: originals.append(('announcement',target,'讨论区对应的公告正文，不是玩家留言'))
        comments=[r for r in game.get('comments',[]) if r.get('news_id')==target_id]
        meaningful=[r for r in comments if len(r.get('text',''))>=40]
        quotes=[r for r in comments if r.get('contains_quoted_text')]
        selected=(quotes[:3]+meaningful[:12]) if number==2 else meaningful[:12]
        originals += [('comment',r,'讨论区留言候选，立场需人工判断') for r in selected]
        coverage=next((r for r in bundle['manifest'].get('inventory',[]) if r.get('kind')=='comments' and r.get('news_id')==target_id),{}).get('coverage',{})
        lines += ['', f'已冻结留言记录 {len(comments)} 条；来源限定公告ID {target_id}。', '',
          f"讨论页报告回复 {coverage.get('reported_total','未知')} 条；本次采集 {coverage.get('collected_count',len(comments))} 条。", '',
          '采集顺序：'+('从早到晚' if coverage.get('sample_order')=='oldest_first' else '未记录或其他顺序')+'；'+('完整讨论快照。' if coverage.get('complete') else '仅部分讨论快照，仍有未采集内容。'), '',
          f"已读 {coverage.get('pages_read','未知')} 页；无法读取正文的记录 {coverage.get('unreadable_message_count','未知')} 条。", '',
          '这里展示的是待判读的部分原文；不能将其直接当作所有留言的观点分布。']
        if number==2 and not quotes: gaps.append('没有被接口标为含引用的留言，需人工确认引用样例后再批准此场景。')
    elif category=='conversation':
        eligible=population(rows,filters)
        bounded=bool(filters.get('start_at') and filters.get('end_at'))
        if expectation.get('outcome')=='clarification':
            lines += ['', '尚未选定目标版本，应先澄清。下方只展示候选公告，不提供未绑定版本的玩家反馈。']
            examples=[]
        else:
            lines += ['',stat_line(eligible,'本轮条件下的冻结样本'),'', '这里只描述冻结快照，不能推断全部玩家情况。']
            examples=[r for r in eligible if len(r.get('text',''))>=20] or eligible
        originals += [('review',r,'符合当前轮明确条件的原文示例'+('；未限定日期，不能归因于某次更新' if not bounded else '')) for r in examples[:5]]
        if number in (1,2) and not bounded: gaps.append('题目未定义目标更新后的观察窗口；下列原文未被绑定为该版本反馈，需先确定窗口。')
        if not eligible and expectation.get('outcome')!='limited': gaps.append('当前轮条件下没有冻结评论，不能以其他语言评论代替；题目预期回答方式须改为有限回答或补齐资料。')
    elif category=='reliability':
        originals += [('announcement',n,'已有冻结来源示例，不代表实时采集或恢复已经成功') for n in news if n.get('is_patch_candidate')][:1]
    lines += ['', '## 4. 对照证据原文', '']
    seen=set(); identities=[]
    for record_kind,record,note in originals:
        key='review_id' if record_kind=='review' else 'comment_id' if record_kind=='comment' else 'news_id'
        ident=(record_kind,str(record[key]))
        if ident in seen: continue
        annotations=list(dict.fromkeys(other_note for other_kind,other,other_note in originals
            if other_kind==record_kind and str(other[key])==ident[1]))
        seen.add(ident); identities.append(ident)
        lines += record_card(record_kind,record,'；'.join(annotations))
    if not originals:
        lines += ['本题没有可展示的相关原文。', '',
          '如果是空统计窗口，核对的是完整冻结来源与筛选边界，不是寻找一条不存在的示例。其他题须补充原文；请勿批准为完整证据题。', '']
    lines += ['## 5. 不能得出的结论', '',
      '- 不能由少量候选评论推断全体玩家比例。',
      '- 推荐不代表没有问题；没采集到不代表不存在。',
      '- 当前评论文字不一定是历史发表时的文字；官方修复声明不证明所有玩家体验一致。', '',
      '## 6. 你需要确认什么', '',
      '- 问题及背景是否明确，条件是否符合你的意思？',
      '- 上面的原文是否真的相关，支持与相反体验是否标对？',
      '- 答案要点或数字是否由这些冻结资料支持，回答边界是否恰当？', '',
      '直接在聊天里回复：`题目ID：通过`，或 `题目ID：修改，原因……`，或 `题目ID：材料不足`。', '',
      '你不用修改JSON、填写审核时间或检查哈希；这些由我处理。未得到你的明确确认不会批准。', '',
      '## 7. 当前材料检查', '']
    if gaps: lines += ['**材料待补齐：请勿批准为正式题。**', '', *['- '+g for g in gaps]]
    else: lines += ['阅读材料已整理，可开始人工审核。语义正确性及标准答案仍须你确认。']
    source_files=[p for p in bundle['manifest'].get('files',{}) if p.startswith('corpus/') and (str(c['appid'])+'_' in p or '/'+str(c['appid'])+'/' in p)]
    lines += ['', '<details><summary>程序追踪信息（无需人工审核）</summary>', '',
      '- 题目ID：'+c['id'], '- 分组：'+c.get('split',''),
      '- 状态：'+c.get('review',{}).get('status','pending_review'),
      '- 冻结资料指纹：'+bundle['manifest']['dataset_sha256'],
      '- 题目指纹：'+c.get('review',{}).get('case_sha256','未记录'), '',
      '绑定原始文件（相对本资料包）：', '', *['- `'+p+'`' for p in source_files], '', '</details>', '']
    return '\n'.join(lines), {'case_id':c['id'],'kind':kind,'gaps':gaps,'evidence_ids':identities,
      'human_review_status':c.get('review',{}).get('status','pending_review')}


def audit_cards(bundle, root):
    errors=[]; metadata=[]
    for kind in ('retrieval','investigations'):
        for c in bundle[kind]:
            expected,meta=render_card(bundle,c,kind); metadata.append(meta)
            folder=Path(root)/'review_pack'
            if c.get('review',{}).get('status')=='rejected': folder=folder/'withdrawn'
            path=folder/kind/(c['id']+'.md')
            if not path.exists() or path.read_bytes().decode('utf-8')!=expected:
                errors.append(c['id']+': 文档缺失或与冻结输入不一致')
    active=[m for m in metadata if m['human_review_status']!='rejected']
    return {'ok':not errors,'count':len(active),'historical_count':len(metadata),'errors':errors,
      'withdrawn':[m['case_id'] for m in metadata if m['human_review_status']=='rejected'],
      'ready_for_human_review':[m['case_id'] for m in active if not m['gaps']],
      'needs_material':[{'case_id':m['case_id'],'gaps':m['gaps']} for m in active if m['gaps']],
      'cards':metadata,'note':'此验证只确认生成和来源一致性，不代替人工语义审核。'}
