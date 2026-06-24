from __future__ import annotations

import httpx
import pytest

from mochi.tools.collector_adapter import BaseCollectorAdapter, CollectorRequestPolicy


@pytest.mark.asyncio
async def test_base_collector_adapter_retries_retryable_statuses() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(
                429,
                headers={"retry-after": "0"},
                request=request,
            )
        return httpx.Response(200, json={"status": "ok"}, request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = BaseCollectorAdapter(
        client=client,
        request_policy=CollectorRequestPolicy(
            max_retries=1,
            backoff_base=0.0,
            backoff_max=0.0,
        ),
    )

    try:
        payload = await adapter.request_json("GET", "https://collector.example/shard")
    finally:
        await client.aclose()

    assert payload == {"status": "ok"}
    assert attempts == 2
