"""Reader-facing analysis; raw machine traces remain in run records."""
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from gamelens.storage.atomic_io import atomic_write_text
from gamelens.core.presentation import reader_text
from gamelens.core.local_time import local_datetime, time_context

TOPICS={"performance":"性能与优化","crash_bug":"崩溃与缺陷","server_network":"服务器与网络",
        "matchmaking":"匹配机制","cheating":"外挂与反作弊","balance":"平衡性","monetization":"商业化",
        "ui_ux":"界面与操作","content":"内容","localization":"本地化","gameplay":"核心玩法",
        "praise":"正面反馈","noise":"低信息反馈","other":"其他"}


def cell(value):return str(value).replace("|","\\|").replace("\n"," ").strip()


def percent(value):
    return f"{value*100:.1f}%" if isinstance(value,(int,float)) and not isinstance(value,bool) else "未提供"


def date(value):
    try:
        return local_datetime(value).strftime("%Y-%m-%d %H:%M %z")
    except (ValueError,TypeError,AttributeError,OSError):return "未提供"


def clean_narrative(text):
    def fenced(match):
        info,body=match.group(2).strip().lower(),match.group(3).strip()
        if info in {"json","jsonc"}:return ""
        try:
            if isinstance(json.loads(body),(dict,list)):return ""
        except ValueError:pass
        return match.group(0)
    text=re.sub(r"(?ms)^(`{3,}|~{3,})([^\n]*)\n(.*?)^\1\s*$",fenced,text or "")
    paragraphs=[]
    for paragraph in re.split(r"\n\s*\n",text):
        try:
            if isinstance(json.loads(paragraph.strip()),(dict,list)):continue
        except ValueError:pass
        paragraphs.append(paragraph)
    text=reader_text("\n\n".join(paragraphs)).replace("本地评论样本","本地缓存评论")
    return re.sub(r"(?m)^(#{1,6})\s+",lambda m:"#"*min(len(m.group(1))+2,6)+" ",text).strip()


def filter_description(filters):
    parts=[]
    period={"before":"更新前","after":"更新后","all":"全部缓存范围"}.get(filters.get("period"))
    if period:parts.append(period)
    language=filters.get("language")
    if language:parts.append({"schinese":"简体中文","tchinese":"繁体中文","english":"英文"}.get(language,language))
    if filters.get("is_positive") is not None:parts.append("Steam好评" if filters["is_positive"] else "Steam差评")
    minimum=filters.get("min_playtime_hours",0)
    if minimum or filters.get('min_playtime_exclusive'):parts.append(f"游玩{'超过' if filters.get('min_playtime_exclusive') else '至少'}{minimum:g}小时")
    if filters.get("max_playtime_hours") is not None:parts.append(f"游玩{'少于' if filters.get('max_playtime_exclusive') else '不超过'}{filters['max_playtime_hours']:g}小时")
    keywords=filters.get("keywords",[])
    if keywords:parts.append(("同时包含" if filters.get("keyword_mode")=="all" else "至少包含一个")+"关键词："+"、".join(keywords))
    return "；".join(parts) or "未额外限制"


def readable_statistics(observations):
    # Repeated preparations may change the same window; show its last result,
    # not intermediate zero counts as if they were independent findings.
    seen=set();effective=[]
    for observation in reversed(observations):
        data=observation.get("data",{});tool=observation.get("tool")
        if tool=="analyze_patch":key=(tool,data.get("news_id"))
        elif tool=="query_reviews":key=(tool,json.dumps({k:data.get(k) for k in ("filters","bounds","patch_news_id")},sort_keys=True))
        elif tool in {"investigate_recent_versions","assess_evidence"}:key=(tool,)
        else:key=(tool,observation.get("step"))
        if key not in seen:effective.append(observation);seen.add(key)
    observations=list(reversed(effective))
    lines=[]
    for observation in observations:
        data=observation.get("data",{});tool=observation.get("tool")
        if tool=="investigate_recent_versions":
            lines.extend(["### 逐版本观察",f"计划检查最近{data.get('requested_count')}次更新，实际有{data.get('available_count')}个补丁候选，其中{data.get('versions_with_reviews')}个窗口有缓存评论。",
                "窗口截止下一次更新或设定天数，避免重复计数。有评论不等于同一缺陷在每个版本都存在。"])
            table=["| 版本 | 观察区间（北京时间时间） | 缓存评论数 | Steam好评率 |","| --- | --- | --- | --- |"]
            for version in reversed(data.get("versions",[])):
                table.append(f"| {cell(version.get('title','未提供'))} | {date(version.get('start_inclusive'))} 至 {date(version.get('end_exclusive'))} | {version.get('stats',{}).get('count','未提供')} | {percent(version.get('stats',{}).get('positive_rate'))} |")
            lines.append("\n".join(table))
        elif tool=="query_reviews":
            stats=data.get("stats",{});bounds=data.get("bounds",{})
            if data.get("statistics_status")=="unavailable_window":
                lines.extend(["### 评论时间覆盖不足",f"请求时段：{date(bounds.get('start_inclusive'))} 至 {date(bounds.get('end_exclusive'))} 北京时间时间。",
                    "本地评论快照没有覆盖该时段，无法统计该时段的实际反馈；空缓存筛选结果不能解释为0条反馈。"])
                continue
            lines.extend(["### 评论筛选观察",f"筛选条件：{filter_description(data.get('filters',{}))}。",
                f"匹配评论数为{stats.get('count','未提供')}条，其中Steam好评{stats.get('positive_count','未提供')}条、差评{stats.get('negative_count','未提供')}条，好评率为{percent(stats.get('positive_rate'))}。",
                f"本次展示{data.get('returned_count',0)}条原文样例。样例只是匹配集合的一部分，关键词命中不等于已确认软件缺陷。"])
            if bounds.get("start_inclusive") is not None or bounds.get("end_exclusive") is not None:
                lines.append(f"观察区间为{date(bounds.get('start_inclusive'))}至{date(bounds.get('end_exclusive'))}（北京时间时间，结束时刻不含）。")
            if data.get('statistics_basis')=='steam_public_window_counts':
                lines.append(f"计数口径：采集时Steam公开接口对上述创建时间/语言条件的全部匹配计数，采集于{data.get('statistics_fetched_at')}。本地有{data.get('body_sample_count',0)}条符合条件的原文样例，样例数量不是统计分母。当前公开文本和推荐态度不等于历史当时快照，不作因果判断。")
            elif data.get('statistics_basis')=='steam_window_complete_bodies':
                lines.append('计数口径：公开接口窗口全部原文已读完，并在完整原文上执行了游玩时长等条件筛选；不是借用未按时长筛选的平台分母。当前文本与投票仍不是历史当时快照。')
            elif data.get('statistics_complete_for_scope') is False:
                lines.append('统计覆盖未完整：这里只是已获取的符合条件样本，不能代表指定月份/人群的全量好评率；来源哈希通过不意味着采集完整。')
            if stats.get("count")==0:lines.append("本轮条件下没有匹配评论；这不能证明没有相关问题，应先检查筛选条件和资料覆盖。")
            usual={"performance","crash","fps","lag","stutter","freeze","崩溃","卡顿","掉帧","闪退"}
            suspicious=[word for word in data.get("filters",{}).get("keywords",[]) if word.startswith("f") and word[1:].casefold() in usual]
            if suspicious:lines.append("筛选条件提醒：本轮实际词项中出现“"+"、".join(suspicious)+"”这类额外前缀，与常用问题词不同，可能导致漏检。应核对这些实际参数，不能只凭零命中判断没有反馈。")
        elif tool=="analyze_patch":
            lines.extend(["### 更新前后窗口",f"所选更新为“{data.get('patch_title','未提供')}”。缓存中的更新前评论有{data.get('before_reviews','未提供')}条，更新后有{data.get('after_reviews','未提供')}条。评论量变化本身不能说明缺陷增加。"])
        elif tool=="search_game_evidence":
            lines.extend(["### 语义证据检索",f"检索问题为“{data.get('query','未提供')}”，返回{data.get('returned_count',0)}条相关性候选。",
                "检索用于发现不同措辞的相关材料，结果按相关性选取，不是随机抽样。候选数量和检索分数不能解释为投诉比例或事实置信度。"])
            population=data.get('population',{})
            retrieval=data.get('retrieval',{})
            eligible=population.get('eligible_review_count')
            if type(eligible) is int:
                lines.append(f"硬条件的已采集评论有{eligible}条；混合检索得到{retrieval.get('candidate_count',0)}条候选，最终保留语义证据{retrieval.get('selected_evidence_count',data.get('returned_count',0))}条。三者是不同计数。")
            if not data.get("returned_count"):
                lines.append("没有返回满足问题语义的证据，不能用这次检索支持问题存在；若硬条件评论数大于0，也不能改写成符合条件的玩家评论为0。")
            if data.get("review_coverage",{}).get("status")=="outside_observed_range":
                lines.append("评论缓存不覆盖请求时段；本次空结果首先是历史资料缺失，不能据此判断当时没有相关反馈。")
            screening=data.get("screening",{})
            if screening.get("enabled"):
                lines.append(f"模型按本轮问题筛选了{screening.get('candidate_count',0)}条候选，保留{screening.get('selected_count',0)}条支持、反例或部分相关材料；相关性判断未经过独立人工验收。")
                labels={"support":"支持材料","counterexample":"相反反馈","partial":"部分相关","irrelevant":"不相关"}
                for verdict in screening.get("verdicts",[]):
                    lines.append(f"- {labels[verdict['role']]}（{cell(verdict['source_id'])}）：{cell(verdict['reason'])}")
        elif tool=="compare_cohorts":
            rows=["| 群体 | 条件 | 评论数 | Steam好评率 |","| --- | --- | --- | --- |"]
            for group,label in (("cohort_a","群体A"),("cohort_b","群体B")):
                cohort=data.get(group,{})
                rows.append(f"| {label} | {cell(filter_description(cohort.get('filters',{})))} | {cohort.get('stats',{}).get('count','未提供')} | {percent(cohort.get('stats',{}).get('positive_rate'))} |")
            lines.extend(["### 人群对比","\n".join(rows),f"两组有{data.get('overlap_count',0)}条评论重叠；不能视为独立样本或直接推断因果关系。"])
            for side,label in (('a','群体A'),('b','群体B')):
                cohort=data.get('cohort_'+side,{})
                bounds=cohort.get('bounds',{})
                lines.append(label+'观察区间：'+date(bounds.get('start_inclusive'))+' 至 '+date(bounds.get('end_exclusive'))+' 北京时间时间（结束时刻不含）。')
                if cohort.get('statistics_basis')=='steam_public_window_counts':
                    lines.append(label+f"计数来自Steam精确窗口全部匹配总数（采集于{cohort.get('statistics_fetched_at')}）；原文样例{cohort.get('body_sample_count',0)}条，与分母分开。当前推荐态度/文本不等于历史当时快照。")
                elif data.get('coverage_'+side,{}).get('status') in {'outside_observed_range','no_cached_reviews'}:
                    lines.append(label+'缺少该时段的缓存评论，无法计算实际好评率；空缓存匹配不是当时0条反馈或0%好评率。')
            if data.get('positive_rate_delta_pp') is not None:
                lines.append(f"群体B相对群体A的好评率变化为{data['positive_rate_delta_pp']:+.2f}个百分点；不是相对增长百分比。")
        elif tool=='ensure_review_coverage':
            lines.append('### 目标窗口补采与原文覆盖')
            for w in data.get('windows',[]):
                lines.append(f"- {date(w.get('start'))}至{date(w.get('end'))}：原文{w.get('body_count',0)}条，计数{'已核对' if w.get('stats_verified') else '未核对'}；停止原因：{w.get('stop_reason','未知')}。")
            lines.append('按评论创建时间补采，原文受分页、数量和时间预算限制；保留旧文件，资料不足或来源错误不会解释为当时零反馈。')
        elif tool in {'prepare_announcement_comments','query_announcement_comments','get_announcement_comment'}:
            coverage=data.get('coverage',{})
            lines.extend(['### 公告讨论区反馈来源',
                f"所选公告ID：{data.get('news_id','未提供')}；这是公告下的独立讨论，不是游戏商店评价。"])
            if coverage.get('status') == 'unavailable':
                lines.append(f"该讨论区未能读取（{data.get('error_code','原因未提供')}）。不能解释为0条留言，也不能用商店评价冒充这次公告的回复。")
            else:
                body = ('已保存留言正文，采集工具不展示原文；还需执行读取工具，不能据此称正文为0条。' if tool=='prepare_announcement_comments' else
                        f"本次成功读取/展示{data.get('returned_count',0)}条原文。")
                lines.extend([f"采集时讨论页报告{coverage.get('reported_total','未知')}条回复；缓存采集{coverage.get('collected_count','未知')}条。"+body,
                    '已覆盖当时公开主题的全部回复。' if coverage.get('complete') else '只覆盖该主题的部分留言；采样和分页上限可能遗漏其他观点，不能计算全体玩家或整个讨论区的观点占比。',
                    '回复数不等于独立玩家数。留言没有商店推荐投票、游玩时长或官方语言标签；引用块和玩梗需要单独判断，主题关联不证明反馈针对本次新增改动。'])
                if coverage.get('unreadable_message_count'):
                    lines.append(f"另有{coverage['unreadable_message_count']}条留言没有可用的文字或创建时间，未用于原文分析；不会为图片类留言编造文字。")
            if data.get('mapping',{}).get('thread_url'):
                lines.append('讨论来源：'+data['mapping']['thread_url'])
        elif tool in {'list_patches','list_announcements'}:
            lines.extend(["### 目标更新定位",f"目标日期：{data.get('requested_date') or '未指定（列出最近公告）'}；按北京时间发布时间检索。",
                f"缓存内匹配{data.get('matched_count',0)}条公告。查询状态：{data.get('lookup_status','未提供')}。",
                "本地未命中不能证明官方没有该更新；缺历史资料时应补采，跨年匹配应先确认年份。"])
        elif tool=="compare_versions":
            rows=["| 版本 | 评论数 | Steam好评率 |","| --- | --- | --- |"]
            for version in data.get("versions",[]):rows.append(f"| {cell(version.get('title','未提供'))} | {version.get('stats',{}).get('count','未提供')} | {percent(version.get('stats',{}).get('positive_rate'))} |")
            lines.extend(["### 版本对比","\n".join(rows),f"两个窗口有{data.get('overlap_count',0)}条评论重叠；结果只描述本地集合差异。"])
        elif tool=="assess_evidence":
            facts=data.get("facts",{});window=facts.get("full_steam_window",{});sample=facts.get("classified_sample",{})
            lines.extend(["### 分类样本与窗口统计",f"本轮引用的分类样本有{data.get('sample_count','未提供')}条。分类样本不是全部窗口评论，不能混用两种分母。"])
            if window:lines.append(f"缓存窗口好评率由{percent(window.get('before_positive_rate'))}变为{percent(window.get('after_positive_rate'))}；对应评论数为{window.get('before_review_count','未提供')}条和{window.get('after_review_count','未提供')}条。")
            changes=sample.get("topic_changes",[])
            if changes:
                rows=["| 分类主题 | 更新前样本数 | 更新后样本数 |","| --- | --- | --- |"]
                for item in changes:rows.append(f"| {TOPICS.get(item.get('topic'),'其他')} | {item.get('before_count','未提供')} | {item.get('after_count','未提供')} |")
                lines.append("\n".join(rows))
    return lines


def write_complete_report(state,summary):
    previous=Path(state.get("artifacts",{}).get("final_report",""))
    if not previous.is_file():return None
    run_id=state["thread_id"]
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,128}",run_id):raise ValueError("非法任务ID")
    pending = bool(state.get("generation_pending"))
    output=previous.parent/f"{state['appid']}_{run_id}_{'partial' if pending else 'complete'}_report.md"
    from gamelens.core.source_freshness import active_observations, superseded_locator_steps
    observations=[o for o in active_observations(state.get("observations",[])) if o.get("ok")]
    source=next((o.get("data",{}).get("source") for o in reversed(observations) if o.get("data",{}).get("source")),{})
    lines=["# 游鉴 GameLens · " + ("未完成调查的部分记录" if pending else "有限范围调查报告" if state.get('scope_limited') else "完整调查分析报告"),"## 一、" + ("已保留的部分结果" if pending else "核心结论"),reader_text(summary),"## 二、调查范围与证据基础",
        f"调查问题：{state['goal']}",f"调查对象：Steam游戏 {state['appid']}。"]
    clock = time_context()
    lines.append(f"时间标准：北京时间（UTC+8） {clock['system_zone']}；日期后附该时刻的UTC偏移，固定UTC+8，不随电脑时区变化。原始Unix时间戳保持不变。")
    if state.get("answer_origin") in {"deterministic_statistics", "verified_partial"}:
        lines.append("回答方式：程序根据已核验统计整理，未追加模型综合推论。")
    elif state.get('answer_origin') == 'verified_evidence_partial':
        lines.append('回答方式：程序保留已核验的原文与范围；最终模型综合未完成，不将片段展示当作调查成功。')
    elif state.get('answer_origin') == 'deterministic_source_limitation':
        lines.append('状态：仅生成来源/日期限制说明，玩家反馈内容调查未完成；未调用模型编造或归纳未读取的留言。')
    elif state.get('answer_origin') == 'deterministic_cache_limitation':
        lines.append('状态：目标更新已定位、前后窗口已查询，但缺少历史评论。限制说明由程序按已核验查询生成，未追加模型综合；好评率变化与玩家观点尚不能确认。')
    superseded = superseded_locator_steps(state.get('observations', []))
    if superseded:
        lines.append('重复定位已采用同一查询的最新快照；旧定位仍保留在运行审计历史，不用于本报告事实或引用。')
    if pending:
        lines.append("状态：调查未完成。模型请求中断，本文件只保留已有观察，不代表最终调查结论。")
    if source.get("cached_review_count") is not None:lines.append(f"本轮可用缓存包含{source['cached_review_count']}条评论；这是调查执行时的资料快照，并非Steam全部评论。")
    coverage = source.get("coverage", {})
    if coverage:
        lines.append(f"缓存评论的创建时间范围：{date(coverage.get('observed_start'))}至{date(coverage.get('observed_end'))} 北京时间时间。起止日期不表示中间资料完整，采集覆盖率未知。")
        lines.append(coverage["warning"])
    revised = max((o.get("data", {}).get("coverage", {}).get("revised_after_window_count", 0) for o in observations), default=0)
    if revised:
        lines.append(f"本轮查询时间窗口内有{revised}条评论在窗口结束后修改过；当前正文不能当作当时版本的原始反馈。")
    lines.extend(["本报告综合更新窗口、玩家评论与可用公告。Steam好差评表示推荐态度，不等同于某个问题的分类结果。同一条件被重复检查时，下方采用最后一次成功统计。","## 三、问题表现与证据解释"])
    lines.append(clean_narrative(state.get("final_detailed_answer","")) or "本轮没有额外的模型综合分析。以下只呈现工具直接支持的观察与原文，不补造问题类型、持续性或原因。")
    lines.extend(["## 四、数据表现与版本变化",*readable_statistics(observations),"## 五、原文证据与公告关联"])
    from gamelens.core.investigation_checks import cited_window_audit
    audit=cited_window_audit(observations,state.get("final_citations",[]))
    if audit["review_ids_by_window"]:
        lines.append("### 按创建时间核对的引用归属")
        for pid,ids in audit["review_ids_by_window"].items():
            lines.append(f"版本 {pid} 的实际引用：评论{ '、'.join(ids) }。归属按不重叠窗口计算，不按检索步骤或推荐投票判断。")
    reviews={};news={};comments={}
    for observation in observations:
        data=observation.get("data",{})
        for r in data.get("reviews",[])+([data["review"]] if "review" in data else []):reviews[r["review_id"]]=r
        for n in data.get("announcements",[]):news[n["news_id"]]=n
        for c in data.get('comments',[]):comments[(c['news_id'],c['comment_id'])]=c
    citations=state.get("final_citations",[]) or [{"review_id":r["review_id"],"quote":r["text"][:500]} for r in list(reviews.values())[:8]]
    if citations:
        lines.append("### 玩家评论")
        for citation in citations:
            review=reviews.get(citation["review_id"],{})
            lines.extend([f"**评论 {citation['review_id']}**"+(f" · {date(review.get('created_at'))} 北京时间时间" if review.get("created_at") else ""),
                          "\n".join("> "+line for line in citation["quote"].splitlines())])
        lines.append("原文用于支持或检验正文判断。反讽、否定句和笼统抱怨必须结合上下文阅读，不能按关键词直接当作已确认缺陷。")
    elif not comments:lines.append("本轮没有返回可直接引用的玩家原文，无法据此确认具体问题表现。")
    comment_citations=state.get('final_comment_citations',[])
    if comment_citations:
        lines.append('### 公告讨论区玩家留言')
        for citation in comment_citations:
            item=comments.get((citation['news_id'],citation['comment_id']),{})
            lines.extend([f"**公告 {citation['news_id']} · 留言 {citation['comment_id']}** · {date(item.get('created_at_timestamp'))} 北京时间时间",
                '\n'.join('> '+line for line in citation['quote'].splitlines())])
            if item.get('source_url'):lines.append('原始留言：'+item['source_url'])
            if item.get('contains_quoted_text'):lines.append('此留言包含引用其他人的文字，请区分引用内容与发言者本人的表态。')
    news_citations=state.get("final_news_citations",[])
    if news_citations:
        lines.append("### 更新公告")
        for citation in news_citations:
            item=news.get(citation["news_id"],{})
            lines.extend([f"**{item.get('title','公告')}（公告 {citation['news_id']}）**",
                          "\n".join("> "+line for line in citation["quote"].splitlines())])
        lines.append("公告改动提供时间背景，不证明评论描述的一定是同一故障，也不证明故障已对所有玩家解决。")
    elif news:lines.append("已取得公告背景，但本轮未提供逐字核验的公告引用，不额外归纳具体修复承诺。")
    has_base=False
    if previous!=output and not previous.name.endswith(("_investigation_report.md","_complete_report.md","_partial_report.md")):
        base=clean_narrative(previous.read_text(encoding="utf-8"))
        if base:lines.extend(["## 六、补充分析",base]);has_base=True
    lines.extend([f"## {'七' if has_base else '六'}、结论边界与进一步验证",
        "- 当前结论只适用于已取得的资料。评论少、窗口不完整或检索无结果，不能直接解释为问题不存在。\n"
        "- 多版本出现相似描述，可支持“反复出现的反馈”；要确认同一缺陷一直存在，还需逐版本原文及复现资料。\n"
        "- 如需进一步确认，应补充缺失时间段、核对设备和运行环境，对照实际日志或复现过程。本报告不猜测未验证的技术原因。"])
    warnings=list(dict.fromkeys(w for o in observations for w in o.get("data",{}).get("warnings",[]) if isinstance(w,str)))
    if warnings:lines.append("\n".join("- "+w for w in warnings))
    lines.append("原始数据与机器核验记录保留在系统审计存储中，不作为本报告的阅读正文。")
    atomic_write_text(output,"\n\n".join(lines)+"\n")
    return output


def refresh_existing_reports(root):
    root=Path(root).resolve();updated=[]
    for file in (root/"data/runs").glob("*.json"):
        state=json.loads(file.read_text(encoding="utf-8"));report=Path(state.get("artifacts",{}).get("final_report",""))
        if state.get("status")!="completed" or not report.is_file() or not report.name.endswith("_complete_report.md"):continue
        if not report.resolve().is_relative_to((root/"reports").resolve()):continue
        from gamelens.core.grounding import validate_grounding,validate_news_citations,validate_comment_citations
        errors=validate_grounding(state.get("observations",[]),state.get("final_claims",[]),state.get("final_citations",[]))
        errors.extend(validate_news_citations(state.get("observations",[]),state.get("final_news_citations",[])))
        errors.extend(validate_comment_citations(state.get('observations',[]),state.get('final_comment_citations',[])))
        if not errors:updated.append(str(write_complete_report(state,state.get("final_answer","调查已完成。"))))
    return updated
