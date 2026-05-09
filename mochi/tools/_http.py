"""HTTP \u5de5\u5177\u5171\u7528\u57fa\u790e\u5c64 \u2014 retry / backoff / error wrapping / rate limiting\u3002"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any

import httpx
from loguru import logger

from mochi.tools.base import ToolResult

# ---------------------------------------------------------------------------
# \u5171\u7528\u5e38\u6578
# ---------------------------------------------------------------------------

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (compatible; Mochi/1.0; +https://github.com/mochi-agent)"
)

_RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})

# ---------------------------------------------------------------------------
# \u7d50\u69cb\u5316 HTTP \u932f\u8aa4
# ---------------------------------------------------------------------------


class ToolHttpError(Exception):
    """\u5c01\u88dd HTTP \u5de5\u5177\u7684\u7d50\u69cb\u5316\u932f\u8aa4\u3002"""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        status_code: int | None = None,
        url: str = "",
    ) -> None:
        super().__init__(message)
        self.message = message
        self.retryable = retryable
        self.status_code = status_code
        self.url = url


# ---------------------------------------------------------------------------
# Token-bucket rate limiter
# ---------------------------------------------------------------------------


class TokenBucketRateLimiter:
    """\u7c21\u6613 token bucket rate limiter\uff0c\u63a7\u5236 HTTP \u8acb\u6c42\u983b\u7387\u3002"""

    def __init__(
        self,
        *,
        rate: float = 10.0,
        burst: int | None = None,
    ) -> None:
        """\u521d\u59cb\u5316 rate limiter\u3002

        Args:
            rate: \u6bcf\u79d2\u5141\u8a31\u7684\u8acb\u6c42\u6578\u3002
            burst: \u7a81\u767c\u5bb9\u91cf\u4e0a\u9650\uff1b\u9810\u8a2d\u7b49\u65bc rate\u3002
        """
        self._rate = rate
        self._burst = burst if burst is not None else max(1, int(rate))
        self._tokens = float(self._burst)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """\u7b49\u5f85\u76f4\u5230\u53ef\u53d6\u5f97\u4e00\u500b token\u3002"""
        async with self._lock:
            self._refill()
            while self._tokens < 1.0:
                wait = (1.0 - self._tokens) / self._rate
                await asyncio.sleep(wait)
                self._refill()
            self._tokens -= 1.0

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
        self._last_refill = now


# ---------------------------------------------------------------------------
# \u5e36 retry / backoff / error wrapping \u7684 HTTP \u8acb\u6c42\u5668
# ---------------------------------------------------------------------------


async def http_request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    max_retries: int = 3,
    backoff_base: float = 1.0,
    backoff_max: float = 30.0,
    rate_limiter: TokenBucketRateLimiter | None = None,
    **httpx_kwargs: Any,
) -> httpx.Response:
    """\u5e36 retry / backoff / error wrapping \u7684 HTTP \u8acb\u6c42\u3002

    Args:
        client: httpx.AsyncClient \u5be6\u4f8b\u3002
        method: HTTP method (GET/POST/...)\u3002
        url: \u8acb\u6c42 URL\u3002
        max_retries: \u6700\u5927\u91cd\u8a66\u6b21\u6578\uff08\u4e0d\u542b\u9996\u6b21\u8acb\u6c42\uff09\u3002
        backoff_base: \u6307\u6578\u9000\u907f\u57fa\u6578\uff08\u79d2\uff09\u3002
        backoff_max: \u9000\u907f\u6700\u5927\u7b49\u5f85\u79d2\u6578\u3002
        rate_limiter: \u53ef\u9078 rate limiter\u3002
        **httpx_kwargs: \u50b3\u7d66 httpx \u7684\u984d\u5916\u53c3\u6578\u3002

    Returns:
        httpx.Response\u3002

    Raises:
        ToolHttpError: HTTP \u8acb\u6c42\u5931\u6557\u4e14\u91cd\u8a66\u8017\u76e1\u3002
    """
    last_error: Exception | None = None
    attempts = 1 + max_retries

    for attempt in range(attempts):
        if rate_limiter is not None:
            await rate_limiter.acquire()

        try:
            response = await client.request(method, url, **httpx_kwargs)
        except httpx.TimeoutException as exc:
            last_error = exc
            if attempt < attempts - 1:
                wait = _backoff_wait(attempt, backoff_base, backoff_max)
                logger.debug(
                    f"HTTP timeout on {url} (attempt {attempt + 1}/{attempts}), "
                    f"retrying in {wait:.1f}s"
                )
                await asyncio.sleep(wait)
                continue
            raise ToolHttpError(
                f"Request timed out after {attempts} attempts: {url}",
                retryable=True,
                url=url,
            ) from exc
        except httpx.HTTPError as exc:
            raise ToolHttpError(
                f"HTTP request failed: {exc}",
                retryable=False,
                url=url,
            ) from exc

        # \u6210\u529f\u5f97\u5230\u56de\u61c9\u2014\u2014\u6aa2\u67e5\u662f\u5426\u9700\u8981\u91cd\u8a66
        if response.status_code not in _RETRYABLE_STATUS_CODES:
            if response.status_code >= 400:
                raise ToolHttpError(
                    f"HTTP {response.status_code} from {url}",
                    retryable=False,
                    status_code=response.status_code,
                    url=url,
                )
            return response

        # Retryable status code
        if attempt < attempts - 1:
            # Respect Retry-After header
            retry_after = response.headers.get("retry-after")
            if retry_after is not None:
                try:
                    wait = min(float(retry_after), backoff_max)
                except ValueError:
                    wait = _backoff_wait(attempt, backoff_base, backoff_max)
            else:
                wait = _backoff_wait(attempt, backoff_base, backoff_max)
            logger.debug(
                f"HTTP {response.status_code} on {url} (attempt {attempt + 1}/{attempts}), "
                f"retrying in {wait:.1f}s"
            )
            await asyncio.sleep(wait)
            continue

        # \u6240\u6709\u91cd\u8a66\u8017\u76e1
        raise ToolHttpError(
            f"HTTP {response.status_code} after {attempts} attempts: {url}",
            retryable=True,
            status_code=response.status_code,
            url=url,
        )

    # \u4e0d\u61c9\u5230\u9054\uff0c\u4f46\u9632\u79a6\u6027\u8655\u7406
    raise ToolHttpError(
        f"HTTP request failed after {attempts} attempts: {url}",
        retryable=True,
        url=url,
    ) from last_error


def make_default_client(
    *,
    timeout: float = 20.0,
    user_agent: str = DEFAULT_USER_AGENT,
) -> httpx.AsyncClient:
    """\u5efa\u7acb\u9810\u8a2d\u7684\u5171\u7528 httpx.AsyncClient\u3002"""
    return httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        headers={
            "User-Agent": user_agent,
            "Accept-Language": "en,zh-TW;q=0.9,zh;q=0.8",
        },
    )


def error_to_tool_result(
    error: ToolHttpError,
    *,
    extra_metadata: dict[str, Any] | None = None,
    suggestion: str | None = None,
) -> ToolResult:
    """\u5c07 ToolHttpError \u8f49\u70ba ToolResult\u3002"""
    metadata: dict[str, Any] = {}
    if error.status_code is not None:
        metadata["status_code"] = error.status_code
    if error.url:
        metadata["url"] = error.url
    if extra_metadata:
        metadata.update(extra_metadata)

    return ToolResult(
        error=error.message,
        retryable=error.retryable,
        suggestion=suggestion,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# \u5167\u90e8\u8f14\u52a9
# ---------------------------------------------------------------------------


def _backoff_wait(attempt: int, base: float, maximum: float) -> float:
    """\u6307\u6578\u9000\u907f + jitter\u3002"""
    wait = min(base * (2**attempt), maximum)
    jitter = random.uniform(0, wait * 0.5)  # noqa: S311
    return wait + jitter
