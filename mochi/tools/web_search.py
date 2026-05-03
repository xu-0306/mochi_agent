"""網頁搜尋工具。"""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from mochi.tools.base import BaseTool, ToolResult


@dataclass
class SearchResult:
    """搜尋結果項目。"""

    title: str
    url: str
    snippet: str = ""


class _DuckDuckGoHTMLParser(HTMLParser):
    """解析 DuckDuckGo HTML 搜尋頁結果。"""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[SearchResult] = []
        self._in_title = False
        self._in_snippet = False
        self._current_href = ""
        self._current_title: list[str] = []
        self._current_snippet: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attrs_dict = dict(attrs)
        class_name = attrs_dict.get("class", "")

        if tag == "a" and "result__a" in class_name:
            self._in_title = True
            self._current_href = attrs_dict.get("href", "") or ""
            self._current_title = []
            self._current_snippet = []
            return

        if tag == "a" and "result-link" in class_name:
            self._in_title = True
            self._current_href = attrs_dict.get("href", "") or ""
            self._current_title = []
            self._current_snippet = []
            return

        if tag == "a" and attrs_dict.get("rel") == "nofollow":
            self._in_title = True
            self._current_href = attrs_dict.get("href", "") or ""
            self._current_title = []
            self._current_snippet = []
            return

        if tag == "a" and attrs_dict.get("class") == "result-snippet":
            self._in_snippet = True
            return

        if tag == "td" and "result-snippet" in class_name:
            self._in_snippet = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_title:
            self._in_title = False
            title = "".join(self._current_title).strip()
            if title:
                self.results.append(
                    SearchResult(
                        title=title,
                        url=_resolve_duckduckgo_url(self._current_href),
                    )
                )
            return

        if tag in {"a", "td"} and self._in_snippet:
            self._in_snippet = False
            snippet = " ".join("".join(self._current_snippet).split()).strip()
            if snippet and self.results:
                self.results[-1].snippet = snippet

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._current_title.append(data)
        elif self._in_snippet:
            self._current_snippet.append(data)


def _resolve_duckduckgo_url(raw_url: str) -> str:
    """將 DuckDuckGo redirect URL 還原為原始網址。"""
    if not raw_url:
        return raw_url

    parsed = urlparse(raw_url)
    if parsed.path.startswith("/l/"):
        query = parse_qs(parsed.query)
        uddg = query.get("uddg")
        if uddg:
            return unquote(uddg[0])
    return raw_url


class WebSearchTool(BaseTool):
    """使用 DuckDuckGo HTML 搜尋頁進行網頁搜尋。"""

    def __init__(
        self,
        engine: str = "duckduckgo",
        timeout: float = 15.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._engine = engine
        self._timeout = timeout
        self._client = client or httpx.AsyncClient(timeout=timeout, follow_redirects=True)
        self._owns_client = client is None

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web and return titles, URLs, and snippets."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords or question.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of results to return.",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        }

    async def execute(self, **kwargs: Any) -> ToolResult:
        """執行網頁搜尋。"""
        query = str(kwargs.get("query", "")).strip()
        top_k = int(kwargs.get("top_k", 5))

        if not query:
            return ToolResult(error="`query` must not be empty.")

        if self._engine != "duckduckgo":
            return ToolResult(error=f"Unsupported search engine: {self._engine}")

        params = {"q": query}
        response = await self._client.get("https://html.duckduckgo.com/html/", params=params)
        response.raise_for_status()

        parser = _DuckDuckGoHTMLParser()
        parser.feed(response.text)
        results = parser.results[: max(1, min(top_k, 10))]

        return ToolResult(
            output=[
                {
                    "title": item.title,
                    "url": item.url,
                    "snippet": item.snippet,
                }
                for item in results
            ],
            metadata={"query": query, "engine": self._engine, "count": len(results)},
        )

    async def close(self) -> None:
        """關閉內部 HTTP client。"""
        if self._owns_client:
            await self._client.aclose()
