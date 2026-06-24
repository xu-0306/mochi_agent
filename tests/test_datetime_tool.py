"""datetime_tool \u5de5\u5177\u6e2c\u8a66\u3002"""

from __future__ import annotations

import pytest

from mochi.tools.datetime_tool import DateTimeTool


@pytest.mark.asyncio
async def test_datetime_returns_utc_by_default() -> None:
    """\u9810\u8a2d\u61c9\u56de\u50b3 UTC \u6642\u9593\u3002"""
    tool = DateTimeTool()
    result = await tool.execute()
    assert result.error is None
    output = result.output
    assert output["timezone"] == "UTC"
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
