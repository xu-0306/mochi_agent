"""web_fetch 工具測試。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mochi.tools.base import ToolExecutionContext
from mochi.tools.web_fetch import WebFetchTool


def _response(
    content: bytes,
    *,
    content_type: str,
    url: str = "https://example.com/page",
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
async def test_web_fetch_extracts_readable_html_text() -> None:
    """HTML 頁面應轉成可讀純文字，並忽略 script/style。"""
    html = b"""
    <html>
      <head><style>.x { color: red; }</style></head>
      <body>
        <h1>Title</h1>
        <script>alert('x')</script>
        <p>Hello <strong>world</strong>.</p>
      </body>
    </html>
    """
    tool = WebFetchTool()

    with patch.object(
        tool._client,
        "request",
        new_callable=AsyncMock,
        return_value=_response(html, content_type="text/html; charset=utf-8"),
    ):
        result = await tool.execute(url="https://example.com/page")

    assert result.error is None
    # Both trafilatura and HTMLParser should strip script/style
    assert "Title" in result.output
    assert "Hello" in result.output
    assert "world" in result.output
    assert "alert" not in result.output
    assert result.metadata["content_type"] == "text/html"
    assert result.metadata["truncated"] is False

    await tool.close()


@pytest.mark.asyncio
async def test_web_fetch_rejects_non_http_urls() -> None:
    """web_fetch 只允許 HTTP/HTTPS URL。"""
    tool = WebFetchTool()

    result = await tool.execute(url="file:///etc/passwd")

    assert result.error == "`url` must be an HTTP or HTTPS URL."

    await tool.close()


@pytest.mark.asyncio
async def test_web_fetch_rejects_blocked_policy_domain() -> None:
    tool = WebFetchTool()

    result = await tool.execute(
        url="https://sub.example.com./page",
        context=ToolExecutionContext(
            permission_policy={"blocked_web_domains": ["example.com"]},
        ),
    )

    assert result.error == "Access to this URL is blocked by policy."
    assert result.metadata["blocked_domains"] == ["example.com"]

    await tool.close()


@pytest.mark.asyncio
async def test_web_fetch_rejects_binary_content_type() -> None:
    """非文字型內容不應當成可讀正文回傳。"""
    tool = WebFetchTool()

    with patch.object(
        tool._client,
        "request",
        new_callable=AsyncMock,
        return_value=_response(b"%PDF", content_type="application/pdf"),
    ):
        result = await tool.execute(url="https://example.com/paper.pdf")

    assert result.error == "Unsupported content type: application/pdf"

    await tool.close()


@pytest.mark.asyncio
async def test_web_fetch_truncates_to_max_bytes() -> None:
    """max_bytes 應限制讀取內容長度並標記 truncated。"""
    tool = WebFetchTool()

    with patch.object(
        tool._client,
        "request",
        new_callable=AsyncMock,
        return_value=_response(b"abcdef", content_type="text/plain"),
    ):
        result = await tool.execute(url="https://example.com/text.txt", max_bytes=3)

    assert result.error is None
    assert result.output == "abc"
    assert result.metadata["bytes_read"] == 3
    assert result.metadata["truncated"] is True

    await tool.close()


@pytest.mark.asyncio
async def test_web_fetch_retryable_and_suggestion() -> None:
    """HTTP 工具錯誤應包含 retryable 和 suggestion 欄位。"""
    tool = WebFetchTool()

    result = await tool.execute(url="")
    assert result.error == "`url` must not be empty."
    # Non-input errors get retryable/suggestion
    assert result.retryable is False

    await tool.close()
