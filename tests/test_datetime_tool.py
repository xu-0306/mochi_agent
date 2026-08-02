"""datetime_tool \u5de5\u5177\u6e2c\u8a66\u3002"""

from __future__ import annotations

from datetime import datetime

import pytest

from mochi.tools.base import ToolExecutionContext
from mochi.tools.datetime_tool import DateTimeTool, _resolve_timezone


def test_datetime_declares_read_only_temporal_lookup_capability() -> None:
    tool = DateTimeTool()

    assert tool.is_read_only is True
    assert tool.tool_capabilities["capabilities"] == ["temporal_lookup"]
    assert tool.tool_capabilities["read_only"] is True
    assert tool.tool_capabilities["destructive"] is False
    assert tool.tool_capabilities["open_world"] is False


@pytest.mark.asyncio
async def test_datetime_uses_system_timezone_by_default() -> None:
    """Without request context, the tool should inherit the backend system timezone."""
    tool = DateTimeTool()
    result = await tool.execute()
    assert result.error is None
    output = result.output
    assert isinstance(output["timezone"], str)
    assert output["timezone"]
    assert result.metadata["timezone_source"] == "system"
    assert len(output["date"]) == 10  # YYYY-MM-DD
    assert ":" in output["time"]


@pytest.mark.asyncio
async def test_datetime_with_timezone() -> None:
    """\u6307\u5b9a\u6642\u5340\u3002"""
    tool = DateTimeTool()
    result = await tool.execute(timezone="Asia/Taipei")
    assert result.error is None
    assert result.output["timezone"] == "Asia/Taipei"


@pytest.mark.asyncio
async def test_datetime_accepts_etc_utc_and_unlisted_iana_timezone() -> None:
    """IANA support must come from tzdata, not the old regional offset table."""
    tool = DateTimeTool()

    utc_result = await tool.execute(timezone="Etc/UTC")
    nairobi_result = await tool.execute(timezone="Africa/Nairobi")

    assert utc_result.error is None
    assert utc_result.output["timezone"] == "Etc/UTC"
    assert nairobi_result.error is None
    assert nairobi_result.output["timezone"] == "Africa/Nairobi"


@pytest.mark.asyncio
async def test_datetime_uses_request_timezone_when_argument_is_omitted() -> None:
    tool = DateTimeTool(default_timezone="UTC")
    context = ToolExecutionContext(client_timezone="Asia/Taipei")

    result = await tool.execute(context=context)

    assert result.error is None
    assert result.output["timezone"] == "Asia/Taipei"
    assert result.metadata["timezone_source"] == "client"


@pytest.mark.asyncio
async def test_datetime_explicit_timezone_overrides_isolated_request_contexts() -> None:
    tool = DateTimeTool(default_timezone="UTC")
    taipei_context = ToolExecutionContext(client_timezone="Asia/Taipei")
    nairobi_context = ToolExecutionContext(client_timezone="Africa/Nairobi")

    taipei_result = await tool.execute(context=taipei_context)
    nairobi_result = await tool.execute(context=nairobi_context)
    explicit_result = await tool.execute(
        context=taipei_context,
        timezone="America/Toronto",
    )

    assert taipei_result.output["timezone"] == "Asia/Taipei"
    assert nairobi_result.output["timezone"] == "Africa/Nairobi"
    assert explicit_result.output["timezone"] == "America/Toronto"
    assert explicit_result.metadata["timezone_source"] == "tool_argument"
    assert taipei_context.client_timezone == "Asia/Taipei"
    assert nairobi_context.client_timezone == "Africa/Nairobi"


def test_datetime_iana_timezone_preserves_dst_rules() -> None:
    timezone = _resolve_timezone("America/Toronto")

    assert timezone is not None
    winter_offset = datetime(2026, 1, 15, tzinfo=timezone).utcoffset()
    summer_offset = datetime(2026, 7, 15, tzinfo=timezone).utcoffset()
    assert winter_offset != summer_offset


@pytest.mark.asyncio
async def test_datetime_invalid_timezone() -> None:
    """\u7121\u6548\u6642\u5340\u61c9\u56de\u50b3\u932f\u8aa4\u8207\u5efa\u8b70\u3002"""
    tool = DateTimeTool()
    result = await tool.execute(timezone="Mars/Olympus_Mons")
    assert result.error is not None
    assert "Unknown timezone" in result.error
    assert result.suggestion is not None


@pytest.mark.asyncio
async def test_datetime_human_format() -> None:
    """human \u683c\u5f0f\u3002"""
    tool = DateTimeTool()
    result = await tool.execute(format="human")
    assert result.error is None
    # Human format contains weekday and month name
    assert "," in result.output["datetime"]


@pytest.mark.asyncio
async def test_datetime_strftime_format() -> None:
    """\u81ea\u8a02 strftime \u683c\u5f0f\u3002"""
    tool = DateTimeTool()
    result = await tool.execute(format="%Y/%m/%d")
    assert result.error is None
    assert "/" in result.output["datetime"]


@pytest.mark.asyncio
async def test_datetime_offset_days() -> None:
    """\u65e5\u671f\u504f\u79fb\u3002"""
    tool = DateTimeTool()
    today = await tool.execute()
    yesterday = await tool.execute(offset_days=-1)

    assert today.error is None
    assert yesterday.error is None
    assert today.output["unix_timestamp"] > yesterday.output["unix_timestamp"]


@pytest.mark.asyncio
async def test_datetime_output_fields() -> None:
    """\u8f38\u51fa\u61c9\u5305\u542b\u6240\u6709\u9810\u671f\u6b04\u4f4d\u3002"""
    tool = DateTimeTool()
    result = await tool.execute()
    assert result.error is None
    output = result.output
    expected_keys = {"datetime", "timezone", "unix_timestamp", "iso", "weekday", "date", "time"}
    assert set(output.keys()) == expected_keys
