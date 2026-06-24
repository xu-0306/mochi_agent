"""Shared foundations for source-specific dataset collector adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import httpx

from mochi.tools._http import TokenBucketRateLimiter, http_request, make_default_client


@dataclass(slots=True)
class CollectorRequestPolicy:
    """HTTP execution policy for remote collector adapters."""

    timeout: float = 20.0
    max_retries: int = 3
    backoff_base: float = 1.0
    backoff_max: float = 30.0
    requests_per_second: float | None = None
    burst: int | None = None


class CollectorAdapter(Protocol):
    """Protocol for source-specific dataset collector adapters."""

    @property
    def name(self) -> str:
        ...

    async def collect_shard(self, shard: Mapping[str, Any]) -> dict[str, Any] | list[dict[str, Any]]:
        ...


class BaseCollectorAdapter:
    """Reusable HTTP wrapper for source-specific remote collectors."""

    def __init__(
        self,
        *,
        request_policy: CollectorRequestPolicy | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._request_policy = request_policy or CollectorRequestPolicy()
        self._client = client or make_default_client(timeout=self._request_policy.timeout)
        self._owns_client = client is None
        self._rate_limiter = (
            TokenBucketRateLimiter(
                rate=self._request_policy.requests_per_second,
                burst=self._request_policy.burst,
            )
            if self._request_policy.requests_per_second is not None
            else None
        )

    @property
    def request_policy(self) -> CollectorRequestPolicy:
        return self._request_policy

    async def request(
        self,
        method: str,
        url: str,
        **httpx_kwargs: Any,
    ) -> httpx.Response:
        return await http_request(
            self._client,
            method,
            url,
            max_retries=self._request_policy.max_retries,
            backoff_base=self._request_policy.backoff_base,
            backoff_max=self._request_policy.backoff_max,
            rate_limiter=self._rate_limiter,
            **httpx_kwargs,
        )

    async def request_json(
        self,
        method: str,
        url: str,
        **httpx_kwargs: Any,
    ) -> Any:
        response = await self.request(method, url, **httpx_kwargs)
        return response.json()

    async def request_text(
        self,
        method: str,
        url: str,
        **httpx_kwargs: Any,
    ) -> str:
        response = await self.request(method, url, **httpx_kwargs)
        return response.text

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
