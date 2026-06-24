"""web_crawl tool tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mochi.tools.base import ToolExecutionContext
from mochi.tools.web_crawl import WebCrawlTool


def _response(
    content: bytes,
    *,
    content_type: str = "text/html; charset=utf-8",
    url: str,
    status_code: int = 200,
) -> MagicMock:
    response = MagicMock()
    response.content = content
    response.text = content.decode("utf-8", errors="replace")
    response.headers = {"content-type": content_type}
    response.encoding = "utf-8"
    response.url = url
    response.status_code = status_code
    response.raise_for_status = MagicMock()
    return response


@pytest.mark.asyncio
async def test_web_crawl_rejects_blocked_start_url() -> None:
    tool = WebCrawlTool()

    result = await tool.execute(
        url="https://sub.example.com./start",
        context=ToolExecutionContext(
            permission_policy={"blocked_web_domains": ["example.com"]},
        ),
    )

    assert result.error == "Access to this URL is blocked by policy."
    assert result.metadata["blocked_domains"] == ["example.com"]

    await tool.close()


@pytest.mark.asyncio
async def test_web_crawl_skips_blocked_discovered_links() -> None:
    tool = WebCrawlTool()
    start_html = b"""
    <html>
      <body>
        <a href="https://blocked.example.com/secret">Blocked</a>
        <a href="https://allowed.test/page-2">Allowed</a>
      </body>
    </html>
    """
    allowed_html = b"<html><body><p>Allowed page</p></body></html>"

    async def _request_side_effect(method: str, url: str, **kwargs: Any) -> MagicMock:  # noqa: ARG001
        if url == "https://blocked.example.com/secret":
            raise AssertionError("blocked domain should not be fetched")
        if url == "https://start.test/index":
            return _response(start_html, url=url)
        if url == "https://allowed.test/page-2":
            return _response(allowed_html, url=url)
        raise AssertionError(f"unexpected url: {url}")

    with patch.object(
        tool._client,
        "request",
        new_callable=AsyncMock,
        side_effect=_request_side_effect,
    ):
        result = await tool.execute(
            url="https://start.test/index",
            same_domain_only=False,
            context=ToolExecutionContext(
                permission_policy={"blocked_web_domains": ["example.com"]},
            ),
        )

    assert result.error is None
    assert result.metadata["visited_urls"] == [
        "https://start.test/index",
        "https://allowed.test/page-2",
    ]
    assert result.metadata["blocked_domains"] == ["example.com"]
    assert result.metadata["blocked_urls"] == ["https://blocked.example.com/secret"]
    assert [page["url"] for page in result.output["pages"]] == [
        "https://start.test/index",
        "https://allowed.test/page-2",
    ]

    await tool.close()
