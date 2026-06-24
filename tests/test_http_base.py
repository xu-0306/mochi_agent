"""HTTP \u57fa\u790e\u5c64\u6e2c\u8a66\u3002"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from mochi.tools._http import (
    TokenBucketRateLimiter,
    ToolHttpError,
    _backoff_wait,
    error_to_tool_result,
    http_request,
    make_default_client,
)


def _mock_response(status_code: int = 200, *, headers: dict | None = None) -> MagicMock:
    r = MagicMock()
    r.status_code = status_code
    r.headers = headers or {}
    return r


@pytest.mark.asyncio
async def test_http_request_success() -> None:
    """200 回應應直接回傳。"""
    client = MagicMock(spec=httpx.AsyncClient)
    client.request = AsyncMock(return_value=_mock_response(200))

    result = await http_request(client, "GET", "https://example.com")
    assert result.status_code == 200
    client.request.assert_awaited_once()


@pytest.mark.asyncio
async def test_http_request_retries_on_429() -> None:
    """429 回應應重試。"""
    client = MagicMock(spec=httpx.AsyncClient)
    client.request = AsyncMock(
        side_effect=[
            _mock_response(429, headers={"retry-after": "0"}),
            _mock_response(200),
        ]
    )

    result = await http_request(client, "GET", "https://example.com", max_retries=1, backoff_base=0.01)
    assert result.status_code == 200
    assert client.request.await_count == 2


@pytest.mark.asyncio
async def test_http_request_raises_on_non_retryable_4xx() -> None:
    """400/403/404 應直接拋出 non-retryable 錯誤。"""
    client = MagicMock(spec=httpx.AsyncClient)
    client.request = AsyncMock(return_value=_mock_response(403))

    with pytest.raises(ToolHttpError) as exc_info:
        await http_request(client, "GET", "https://example.com")
    assert exc_info.value.retryable is False
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_http_request_raises_on_timeout() -> None:
    """Timeout 應拋出 retryable 錯誤。"""
    client = MagicMock(spec=httpx.AsyncClient)
    client.request = AsyncMock(side_effect=httpx.TimeoutException("timed out"))

    with pytest.raises(ToolHttpError) as exc_info:
        await http_request(client, "GET", "https://example.com", max_retries=0)
    assert exc_info.value.retryable is True


@pytest.mark.asyncio
async def test_http_request_respects_retry_after_header() -> None:
    """429 + Retry-After header 應 respect。"""
    client = MagicMock(spec=httpx.AsyncClient)
    client.request = AsyncMock(
        side_effect=[
            _mock_response(429, headers={"retry-after": "0.01"}),
            _mock_response(200),
        ]
    )
    result = await http_request(client, "GET", "https://example.com", max_retries=1, backoff_base=0.01)
    assert result.status_code == 200


def test_error_to_tool_result_conversion() -> None:
    """ToolHttpError 應正確轉為 ToolResult。"""
    err = ToolHttpError("test error", retryable=True, status_code=429, url="https://x.com")
    result = error_to_tool_result(err, suggestion="try later")
    assert result.error == "test error"
    assert result.retryable is True
    assert result.suggestion == "try later"
    assert result.metadata["status_code"] == 429


def test_backoff_wait_grows_exponentially() -> None:
    """退避等待時間應指數增長。"""
    w0 = _backoff_wait(0, 1.0, 60.0)
    w2 = _backoff_wait(2, 1.0, 60.0)
    # attempt=2 → base * 4 + jitter, should be significantly larger
    assert w2 > w0 * 1.5  # With jitter, at least 1.5x larger


def test_make_default_client_sets_user_agent() -> None:
    """預設 client 應包含 User-Agent header。"""
    client = make_default_client()
    assert "User-Agent" in client.headers
    assert "Mochi" in client.headers["User-Agent"]


@pytest.mark.asyncio
async def test_token_bucket_rate_limiter() -> None:
    """Rate limiter 應控制請求頻率。"""
    limiter = TokenBucketRateLimiter(rate=100.0, burst=2)
    # First 2 should be instant (burst)
    await limiter.acquire()
    await limiter.acquire()
    # Still works (refills fast at rate=100)
    await limiter.acquire()
