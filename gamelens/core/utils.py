from datetime import datetime, timezone


def unix_timestamp_to_iso(timestamp: int | None) -> str | None:
    """将 Unix 秒级时间戳转换成 UTC ISO 8601 字符串。"""
    if timestamp is None:
        return None

    return (
        datetime.fromtimestamp(timestamp, tz=timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )
