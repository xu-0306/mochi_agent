"""日期時間工具 — 取得目前時間與時區轉換。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, tzinfo
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from mochi.tools.base import BaseTool, ToolExecutionContext, ToolResult


def _resolve_timezone(tz_name: str) -> tzinfo | None:
    """Resolve one IANA timezone through the platform/tzdata database."""
    if tz_name.upper() == "UTC":
        return UTC

    try:
        return ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, KeyError, ValueError):
        return None


def _local_timezone() -> tuple[str, tzinfo]:
    """Return the backend system timezone without assuming a named region."""
    local_now = datetime.now().astimezone()
    local_tz = local_now.tzinfo
    if local_tz is None:
        return "UTC", UTC
    zone_key = getattr(local_tz, "key", None)
    if isinstance(zone_key, str) and zone_key.strip():
        return zone_key.strip(), local_tz
    return local_now.tzname() or "UTC", local_tz


class DateTimeTool(BaseTool):
    """取得目前日期時間或執行時間運算。"""

    def __init__(self, *, default_timezone: str | None = "auto") -> None:
        self._default_timezone = default_timezone

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
    def is_read_only(self) -> bool:
        return True

    @property
    def tool_capabilities(self) -> dict[str, Any]:
        return {
            **super().tool_capabilities,
            "capabilities": ["temporal_lookup"],
        }

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": (
                        "Optional IANA timezone name (e.g. 'Asia/Taipei', "
                        "'Africa/Nairobi', or 'UTC'). Omit it to use the "
                        "current user's timezone."
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

    async def execute(
        self,
        context: ToolExecutionContext | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """取得時間。"""
        fmt = str(kwargs.get("format", "iso")).strip()
        offset_days = int(kwargs.get("offset_days", 0))
        offset_hours = int(kwargs.get("offset_hours", 0))

        timezone_source = "tool_argument"
        if "timezone" in kwargs:
            tz_name = str(kwargs["timezone"]).strip()
            tz = _resolve_timezone(tz_name)
        else:
            timezone_source = "client"
            client_timezone = (
                context.client_timezone.strip()
                if context is not None
                and isinstance(context.client_timezone, str)
                and context.client_timezone.strip()
                else None
            )
            tz_name = client_timezone or ""
            tz = _resolve_timezone(tz_name) if tz_name else None

            configured_timezone = (
                self._default_timezone.strip()
                if isinstance(self._default_timezone, str)
                and self._default_timezone.strip()
                and self._default_timezone.strip().casefold() != "auto"
                else None
            )
            if tz is None and client_timezone is None and configured_timezone is not None:
                timezone_source = "configured_default"
                tz_name = configured_timezone
                tz = _resolve_timezone(tz_name)
            if tz is None and client_timezone is None:
                timezone_source = "system"
                tz_name, tz = _local_timezone()

        if tz is None:
            return ToolResult(
                error=f"Unknown timezone: '{tz_name}'.",
                suggestion=(
                    "Use a valid IANA timezone name like 'Asia/Taipei', "
                    "'Africa/Nairobi', 'Europe/London', or 'UTC'."
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
            metadata={
                "timezone": tz_name,
                "timezone_source": timezone_source,
                "format": fmt,
            },
        )
