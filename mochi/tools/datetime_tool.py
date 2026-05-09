"""日期時間工具 — 取得目前時間與時區轉換。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from typing import Any

from mochi.tools.base import BaseTool, ToolResult

# zoneinfo 在 Windows 需要 tzdata 套件；我們提供常用時區的 UTC offset fallback
try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    _zoneinfo_available = True
except ImportError:
    _zoneinfo_available = False

    class ZoneInfoNotFoundError(KeyError):  # type: ignore[no-redef]
        """Stub for missing zoneinfo."""

# 常用 IANA 時區 → UTC offset (hours) 的 fallback 對照表
_FALLBACK_OFFSETS: dict[str, float] = {
    "Asia/Taipei": 8, "Asia/Shanghai": 8, "Asia/Hong_Kong": 8, "Asia/Singapore": 8,
    "Asia/Tokyo": 9, "Asia/Seoul": 9,
    "Asia/Kolkata": 5.5, "Asia/Bangkok": 7,
    "US/Eastern": -5, "America/New_York": -5,
    "US/Central": -6, "America/Chicago": -6,
    "US/Mountain": -7, "America/Denver": -7,
    "US/Pacific": -8, "America/Los_Angeles": -8,
    "Europe/London": 0, "Europe/Berlin": 1, "Europe/Paris": 1,
    "Europe/Moscow": 3,
    "Australia/Sydney": 11, "Pacific/Auckland": 12,
}


def _resolve_timezone(tz_name: str) -> timezone | Any | None:
    """解析時區名稱，支援 zoneinfo → fallback offset → None。"""
    if tz_name.upper() == "UTC":
        return UTC

    # 嘗試 zoneinfo（需要 tzdata）
    if _zoneinfo_available:
        try:
            return ZoneInfo(tz_name)
        except (ZoneInfoNotFoundError, KeyError):
            pass

    # Fallback: 常用時區 offset 對照表
    offset = _FALLBACK_OFFSETS.get(tz_name)
    if offset is not None:
        return timezone(timedelta(hours=offset), name=tz_name)

    return None


class DateTimeTool(BaseTool):
    """取得目前日期時間或執行時間運算。"""

    @property
    def name(self) -> str:
        return "get_current_time"

    @property
    def description(self) -> str:
        return (
            "Get the current date and time in a specified timezone, or perform "
            "date arithmetic. Use when you need to know 'what time is it now', "
            "'what day is today', 'what is the current date', or calculate "
            "time differences and future/past dates."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "default": "UTC",
                    "description": (
                        "IANA timezone name (e.g. 'Asia/Taipei', 'US/Eastern', "
                        "'Europe/London', 'UTC'). Defaults to UTC."
                    ),
                },
                "format": {
                    "type": "string",
                    "default": "iso",
                    "description": (
                        "Output format: 'iso' for ISO 8601, 'human' for human-readable, "
                        "or a strftime pattern like '%Y-%m-%d %H:%M'."
                    ),
                },
                "offset_days": {
                    "type": "integer",
                    "default": 0,
                    "description": (
                        "Add or subtract days from the current time. "
                        "E.g. -1 for yesterday, 7 for one week from now."
                    ),
                },
                "offset_hours": {
                    "type": "integer",
                    "default": 0,
                    "description": "Add or subtract hours from the current time.",
                },
            },
            "additionalProperties": False,
        }

    async def execute(self, **kwargs: Any) -> ToolResult:
        """取得時間。"""
        tz_name = str(kwargs.get("timezone", "UTC")).strip()
        fmt = str(kwargs.get("format", "iso")).strip()
        offset_days = int(kwargs.get("offset_days", 0))
        offset_hours = int(kwargs.get("offset_hours", 0))

        # 解析時區
        tz = _resolve_timezone(tz_name)
        if tz is None:
            return ToolResult(
                error=f"Unknown timezone: '{tz_name}'.",
                suggestion=(
                    "Use a valid IANA timezone name like 'Asia/Taipei', "
                    "'US/Eastern', 'Europe/London', or 'UTC'."
                ),
            )

        now = datetime.now(tz)
        if offset_days != 0 or offset_hours != 0:
            now = now + timedelta(days=offset_days, hours=offset_hours)

        # 格式化
        if fmt == "iso":
            formatted = now.isoformat()
        elif fmt == "human":
            formatted = now.strftime("%A, %B %d, %Y at %H:%M:%S %Z")
        else:
            try:
                formatted = now.strftime(fmt)
            except (ValueError, TypeError) as exc:
                return ToolResult(
                    error=f"Invalid strftime format: {exc}",
                    suggestion="Use a valid Python strftime pattern like '%Y-%m-%d %H:%M'.",
                )

        return ToolResult(
            output={
                "datetime": formatted,
                "timezone": tz_name,
                "unix_timestamp": int(now.timestamp()),
                "iso": now.isoformat(),
                "weekday": now.strftime("%A"),
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S"),
            },
            metadata={"timezone": tz_name, "format": fmt},
        )
