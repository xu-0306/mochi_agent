"""SSE（Server-Sent Events）工具函式。"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any


async def sse_stream(events: AsyncIterator[dict[str, Any]]) -> AsyncIterator[str]:
    """將事件字典序列轉為 SSE 格式字串串流。

    Args:
        events: 事件字典非同步迭代器。

    Yields:
        SSE 格式字串（data: {...}\\n\\n）。
    """
    async for event in events:
        named_event = isinstance(event, dict) and "_sse_data" in event
        event_name = event.get("_sse_event") if named_event else None
        event_id = event.get("_sse_id") if named_event else None
        payload = event.get("_sse_data") if named_event else event
        data = json.dumps(payload, ensure_ascii=False)
        prefix = ""
        if isinstance(event_id, str) and event_id:
            prefix += f"id: {event_id}\n"
        if isinstance(event_name, str) and event_name:
            prefix += f"event: {event_name}\n"
        yield f"{prefix}data: {data}\n\n"
