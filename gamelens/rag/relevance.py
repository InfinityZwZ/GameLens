"""Weak topic ranking hints only; never a gate for semantic candidates."""
import re

TOPICS = {
    "save_data": r"存档|存[储檔]|读档|进度丢|丢失进度|save\s*(?:game|file|data)|savegame|corrupt(?:ed)?\s*save|读取进度",
    "performance": r"掉帧|卡顿|帧数|帧率|流畅|性能|优化|fps|frame\s*rate|stutter|laggy|performance",
    "crash": r"闪退|崩溃|crash|启动不了|无法启动|自动退出|launch\s*(?:fail|error)",
    "matchmaking": r"匹配|matchmaking|queue\s*time",
    "anticheat": r"反作弊|外挂|cheat|vac",
    "crosshair": r"准[心星]|crosshair",
}

TERM_EXPANSIONS = {
    "performance": ("fps", "stutter", "lag", "掉帧", "卡顿"),
    "crash": ("crash", "闪退", "崩溃", "回到桌面"),
    "save_data": ("save", "存档", "进度丢失"),
    "matchmaking": ("matchmaking", "queue time", "匹配"),
    "anticheat": ("cheat", "vac", "外挂", "反作弊"),
    "crosshair": ("crosshair", "准星", "准心"),
}


def topic_anchor(query):
    matched = [name for name, pattern in TOPICS.items() if re.search(pattern, query, re.I)]
    return matched[0] if len(matched) == 1 else None


def topic_supported(topic, text):
    return topic is None or bool(re.search(TOPICS[topic], text, re.I))


def expand_lexical_terms(query, supplied=(), limit=8):
    """Small deterministic query expansion; it never changes hard filters."""
    values = [value.strip() for value in supplied if value and value.strip()]
    matched = [name for name, pattern in TOPICS.items() if re.search(pattern, query, re.I)]
    if len(matched) == 1:
        values.extend(TERM_EXPANSIONS.get(matched[0], ()))
    return list(dict.fromkeys(values))[:limit]
