"""Reader-facing wording only. Never rewrite structured facts or exact citations."""
import re


ANSWER_FORMAT_INSTRUCTION = """final_answer使用简洁Markdown排版：先一小段直接结论；有多个发现时空一行，再用“**主要发现**”及分条列表，每条只讲一个要点；必要限制放在独立的“**范围与限制**”段落。标题和列表前后留空行，重点可加粗。简单问题只需短段落或少量列表，不强凑章节，不用表格和大标题。仍返回JSON，换行保存在字符串中，不用代码围栏包住答案；不为排版增加内容或重复核验细节。"""


def reader_text(text: str) -> str:
    substitutions = {
        "cached_reviews_only": "本地评论样本",
        "cached_retrieval_candidates_only": "检索候选资料",
        "partial_snapshot": "部分缓存资料", "outside_observed_range": "超出缓存可见时间范围",
        "no_cached_reviews": "没有该时段的缓存评论",
        "stats.positive_rate": "好评率", "stats.negative_rate": "差评率",
        "stats.positive_count": "好评数量", "stats.negative_count": "差评数量",
        "stats.count": "评论数量", "source.scope": "统计范围",
        "query_reviews": "评论筛选工具",
    }
    for key, value in substitutions.items():
        text = (text or "").replace(key, value)
    # Display rounding only. Original full precision stays in grounded claims.
    return re.sub(r"(?<![A-Za-z0-9_.])(-?\d+\.\d{7,})(?![A-Za-z0-9_.])",
                  lambda m: f"{float(m.group(1)):.4f}".rstrip("0").rstrip("."), text)
