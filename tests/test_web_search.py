"""web_search 工具測試。"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mochi.tools.base import ToolExecutionContext
from mochi.tools.web_search import WebSearchTool


def _mock_response(
    html: str = "",
    *,
    json_payload: dict[str, Any] | None = None,
    status_code: int = 200,
) -> MagicMock:
    """建立 HTTP response mock。"""
    response = MagicMock()
    response.text = html
    response.content = html.encode("utf-8") if isinstance(html, str) else html
    response.status_code = status_code
    response.raise_for_status = MagicMock()
    response.json = MagicMock(return_value=json_payload or {})
    response.headers = {}
    return response


@pytest.mark.asyncio
async def test_web_search_parses_duckduckgo_results() -> None:
    """DuckDuckGo HTML 結果頁應可解析為結構化資料。"""
    html = """
    <html>
      <body>
        <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Falpha">Alpha</a>
        <td class="result-snippet">Alpha snippet</td>
        <a class="result__a" href="https://example.com/beta">Beta</a>
        <td class="result-snippet">Beta snippet</td>
      </body>
    </html>
    """
    tool = WebSearchTool(engine="duckduckgo_html")

    with patch.object(
        tool._client,
        "request",
        new_callable=AsyncMock,
        return_value=_mock_response(html),
    ):
        result = await tool.execute(query="mochi", top_k=2)

    assert result.error is None
    assert result.metadata["engine"] == "duckduckgo_html"
    assert result.metadata["count"] == 2
    assert result.output == {
        "query": "mochi",
        "provider": "duckduckgo_html",
        "results": [
            {
                "title": "Alpha",
                "url": "https://example.com/alpha",
                "snippet": "Alpha snippet",
            },
            {
                "title": "Beta",
                "url": "https://example.com/beta",
                "snippet": "Beta snippet",
            },
        ],
        "warnings": [],
        "attempted_providers": ["duckduckgo_html"],
    }

    await tool.close()


@pytest.mark.asyncio
async def test_web_search_requires_query() -> None:
    """空查詢應直接回傳錯誤。"""
    tool = WebSearchTool(engine="duckduckgo_html")
    result = await tool.execute(query=" ")
    assert result.error == "`query` must not be empty."
    await tool.close()


@pytest.mark.asyncio
async def test_web_search_duckduckgo_challenge_returns_error() -> None:
    """DuckDuckGo anomaly 頁面應回傳明確錯誤，不可偽裝成空成功。"""
    tool = WebSearchTool(engine="duckduckgo_html")
    html = "<html><script src='/dist/anomaly.js'></script><form id='challenge-form'></form></html>"

    with patch.object(
        tool._client,
        "request",
        new_callable=AsyncMock,
        return_value=_mock_response(html, status_code=202),
    ):
        result = await tool.execute(query="台中 南區 天氣")

    assert result.output is None
    assert result.error is not None
    assert "anti-bot challenge" in result.error
    assert result.metadata["blocked"] is True
    assert result.metadata["status_code"] == 202
    await tool.close()


@pytest.mark.asyncio
async def test_web_search_empty_duckduckgo_page_returns_error() -> None:
    """不可解析結果時應回傳 error，避免 ReAct loop 誤判工具成功。"""
    tool = WebSearchTool(engine="duckduckgo_html")

    with patch.object(
        tool._client,
        "request",
        new_callable=AsyncMock,
        return_value=_mock_response("<html><body>No results here</body></html>"),
    ):
        result = await tool.execute(query="mochi")

    assert result.output is None
    assert result.error == "Search returned no parseable results."
    assert result.metadata["count"] == 0
    await tool.close()


@pytest.mark.asyncio
async def test_web_search_searxng_json_provider() -> None:
    """SearXNG JSON API 結果應轉成共用搜尋結果格式。"""
    tool = WebSearchTool(engine="searxng", searxng_base_url="https://search.example.test")
    payload = {
        "results": [
            {
                "title": "Alpha",
                "url": "https://example.com/alpha",
                "content": "Alpha <b>snippet</b>",
            },
            {
                "title": "Missing URL",
                "content": "ignored",
            },
        ]
    }

    with patch.object(
        tool._client,
        "request",
        new_callable=AsyncMock,
        return_value=_mock_response(json_payload=payload),
    ):
        result = await tool.execute(query="mochi", top_k=3)

    assert result.error is None
    assert result.metadata["engine"] == "searxng"
    assert result.output == {
        "query": "mochi",
        "provider": "searxng",
        "results": [
            {
                "title": "Alpha",
                "url": "https://example.com/alpha",
                "snippet": "Alpha snippet",
            }
        ],
        "warnings": [],
        "attempted_providers": ["searxng"],
    }
    await tool.close()


@pytest.mark.asyncio
async def test_web_search_brave_json_provider() -> None:
    """Brave Search API 結果應轉成共用搜尋結果格式。"""
    tool = WebSearchTool(engine="brave", brave_api_key="test-key")
    payload = {
        "web": {
            "results": [
                {
                    "title": "Beta",
                    "url": "https://example.com/beta",
                    "description": "Beta <strong>snippet</strong>",
                }
            ]
        }
    }

    with patch.object(
        tool._client,
        "request",
        new_callable=AsyncMock,
        return_value=_mock_response(json_payload=payload),
    ):
        result = await tool.execute(query="mochi", top_k=1)

    assert result.error is None
    assert result.metadata["engine"] == "brave"
    assert result.output == {
        "query": "mochi",
        "provider": "brave",
        "results": [
            {
                "title": "Beta",
                "url": "https://example.com/beta",
                "snippet": "Beta snippet",
            }
        ],
        "warnings": [],
        "attempted_providers": ["brave"],
    }
    await tool.close()


@pytest.mark.asyncio
async def test_web_search_filters_blocked_domains_from_context_policy() -> None:
    tool = WebSearchTool(engine="brave", brave_api_key="test-key")
    payload = {
        "web": {
            "results": [
                {
                    "title": "Blocked",
                    "url": "https://example.com./blocked",
                    "description": "Should be filtered",
                },
                {
                    "title": "Allowed",
                    "url": "https://allowed.test/result",
                    "description": "Should remain",
                },
                {
                    "title": "Explicitly Blocked",
                    "url": "https://other.test/blocked",
                    "description": "Should also be filtered",
                },
            ]
        }
    }

    with patch.object(
        tool._client,
        "request",
        new_callable=AsyncMock,
        return_value=_mock_response(json_payload=payload),
    ):
        result = await tool.execute(
            query="mochi",
            blocked_domains=["other.test"],
            context=ToolExecutionContext(
                permission_policy={"blocked_web_domains": ["example.com"]},
            ),
        )

    assert result.error is None
    assert result.output["results"] == [
        {
            "title": "Allowed",
            "url": "https://allowed.test/result",
            "snippet": "Should remain",
        }
    ]
    assert result.metadata["blocked_domains"] == ["other.test", "example.com"]
    await tool.close()


@pytest.mark.asyncio
async def test_web_search_tavily_provider() -> None:
    """Tavily API 結果應正確解析。"""
    tool = WebSearchTool(engine="tavily", tavily_api_key="test-tavily-key")
    payload = {
        "results": [
            {
                "title": "Tavily Alpha",
                "url": "https://example.com/tavily",
                "content": "Agent-native search result",
            }
        ]
    }

    with patch.object(
        tool._client,
        "request",
        new_callable=AsyncMock,
        return_value=_mock_response(json_payload=payload),
    ):
        result = await tool.execute(query="AI agents", top_k=1)

    assert result.error is None
    assert result.metadata["engine"] == "tavily"
    assert result.output["query"] == "AI agents"
    assert result.output["provider"] == "tavily"
    assert result.output["attempted_providers"] == ["tavily"]
    assert len(result.output["results"]) == 1
    assert result.output["results"][0]["title"] == "Tavily Alpha"
    assert result.output["results"][0]["snippet"] == "Agent-native search result"
    await tool.close()


@pytest.mark.asyncio
async def test_web_search_exa_provider() -> None:
    """Exa API 結果應正確解析，並在需要時回傳 content。"""
    tool = WebSearchTool(engine="exa", exa_api_key="test-exa-key")
    payload = {
        "results": [
            {
                "title": "Exa Alpha",
                "url": "https://example.com/exa",
                "highlights": ["Exa <b>snippet</b>"],
                "summary": "Fallback summary",
                "text": "Full markdown text",
            }
        ]
    }

    with patch.object(
        tool._client,
        "request",
        new_callable=AsyncMock,
        return_value=_mock_response(json_payload=payload),
    ):
        result = await tool.execute(query="AI agents", top_k=1, include_content=True)

    assert result.error is None
    assert result.metadata["engine"] == "exa"
    assert result.output == {
        "query": "AI agents",
        "provider": "exa",
        "results": [
            {
                "title": "Exa Alpha",
                "url": "https://example.com/exa",
                "snippet": "Exa snippet",
                "content": "Full markdown text",
            }
        ],
        "warnings": [],
        "attempted_providers": ["exa"],
    }
    await tool.close()


@pytest.mark.asyncio
async def test_web_search_fallback_chain() -> None:
    """主引擎失敗時應自動 fallback 到下一個 engine。"""
    # Tavily unconfigured (no key), brave has key
    tool = WebSearchTool(
        engine="tavily",
        fallback_engines=["brave"],
        brave_api_key="test-key",
    )
    payload = {
        "web": {
            "results": [
                {
                    "title": "Brave Fallback",
                    "url": "https://example.com/fallback",
                    "description": "This came from brave",
                }
            ]
        }
    }

    with patch.object(
        tool._client,
        "request",
        new_callable=AsyncMock,
        return_value=_mock_response(json_payload=payload),
    ):
        result = await tool.execute(query="mochi")

    assert result.error is None
    assert result.metadata["engine"] == "brave"
    await tool.close()


@pytest.mark.asyncio
async def test_web_search_no_configured_engine() -> None:
    """所有 engine 都未配置時應回傳明確錯誤。"""
    tool = WebSearchTool(engine="tavily", fallback_engines=["serper", "exa"])
    payload = {
        "data": [
            {
                "title": "Jina Alpha",
                "url": "https://example.com/jina",
                "description": "Jina fallback result",
            }
        ]
    }

    with patch.object(
        tool._client,
        "request",
        new_callable=AsyncMock,
        return_value=_mock_response(json_payload=payload),
    ):
        result = await tool.execute(query="mochi")

    assert result.error is None
    assert result.output["provider"] == "jina"
    assert result.output["attempted_providers"] == ["jina"]
    assert any(
        warning["provider"] == "tavily" and warning["status"] == "skipped_missing_key"
        for warning in result.output["warnings"]
    )
    await tool.close()


@pytest.mark.asyncio
async def test_web_search_retryable_flag() -> None:
    """DuckDuckGo challenge 應設 retryable=True。"""
    tool = WebSearchTool(engine="duckduckgo_html")
    html = "<html><script src='/dist/anomaly.js'></script></html>"

    with patch.object(
        tool._client,
        "request",
        new_callable=AsyncMock,
        return_value=_mock_response(html),
    ):
        result = await tool.execute(query="test")

    assert result.retryable is True
    await tool.close()


@pytest.mark.asyncio
async def test_web_search_metadata_distinguishes_missing_key_and_request_failure() -> None:
    """Provider diagnostics should separate skipped_missing_key from request_failed."""
    tool = WebSearchTool(engine="tavily", fallback_engines=["brave"])

    async def _request_side_effect(method: str, url: str, **kwargs: Any) -> MagicMock:  # noqa: ARG001
        if "s.jina.ai" in url:
            return _mock_response(json_payload={"error": "temporary upstream issue"}, status_code=503)
        return _mock_response(
            """
            <html>
              <body>
                <a class="result__a" href="https://example.com/ddg">DDG Alpha</a>
                <td class="result-snippet">DuckDuckGo fallback result</td>
              </body>
            </html>
            """
        )

    with patch.object(
        tool._client,
        "request",
        new_callable=AsyncMock,
        side_effect=_request_side_effect,
    ):
        result = await tool.execute(query="mochi")

    assert result.error is None
    assert result.output["provider"] == "duckduckgo_html"
    assert result.output["attempted_providers"] == ["jina", "duckduckgo_html"]
    diagnostics = result.metadata["provider_attempts"]
    assert any(
        item["provider"] == "tavily" and item["status"] == "skipped_missing_key"
        for item in diagnostics
    )
    assert any(
        item["provider"] == "brave" and item["status"] == "skipped_missing_key"
        for item in diagnostics
    )
    assert any(
        item["provider"] == "jina" and item["status"] == "request_failed"
        for item in diagnostics
    )
    assert any(
        item["provider"] == "duckduckgo_html" and item["status"] == "succeeded"
        for item in diagnostics
    )
    fallback_diagnostics = result.metadata["fallback_diagnostics"]
    assert any(
        item["name"] == "search_provider_fallback"
        and item["to"] == "duckduckgo_html"
        for item in fallback_diagnostics
    )
    await tool.close()


@pytest.mark.asyncio
async def test_web_search_falls_back_after_non_retryable_provider_http_error() -> None:
    """Provider 4xx errors should still fall back to the next configured engine."""
    tool = WebSearchTool(
        engine="brave",
        fallback_engines=["duckduckgo_html"],
        brave_api_key="test-key",
    )

    async def _request_side_effect(method: str, url: str, **kwargs: Any) -> MagicMock:  # noqa: ARG001
        if "api.search.brave.com" in url:
            return _mock_response(json_payload={"error": "unprocessable"}, status_code=422)
        return _mock_response(
            """
            <html>
              <body>
                <a class="result__a" href="https://example.com/ddg-fallback">DDG Fallback</a>
                <td class="result-snippet">Recovered via DuckDuckGo HTML fallback.</td>
              </body>
            </html>
            """
        )

    with patch.object(
        tool._client,
        "request",
        new_callable=AsyncMock,
        side_effect=_request_side_effect,
    ):
        result = await tool.execute(query="taichung weather tomorrow")

    assert result.error is None
    assert result.output["provider"] == "duckduckgo_html"
    assert result.output["attempted_providers"] == ["brave", "duckduckgo_html"]
    diagnostics = result.metadata["provider_attempts"]
    assert any(
        item["provider"] == "brave" and item["status"] == "request_failed"
        for item in diagnostics
    )
    assert any(
        item["provider"] == "duckduckgo_html" and item["status"] == "succeeded"
        for item in diagnostics
    )
    await tool.close()
