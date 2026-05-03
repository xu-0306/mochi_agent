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
        data = json.dumps(event, ensure_ascii=False)
        yield f"data: {data}\n\n"
