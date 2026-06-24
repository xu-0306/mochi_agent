"""Simple same-domain web crawler built on the shared HTTP tooling."""

from __future__ import annotations

from collections import deque
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

from mochi.tools._http import ToolHttpError, error_to_tool_result, http_request, make_default_client
from mochi.tools.base import BaseTool, ToolExecutionContext, ToolResult
from mochi.tools.web_fetch import (
    _blocked_domains_from_context,
    _blocked_url_tool_result,
    _extract_with_htmlparser,
    _extract_with_trafilatura,
    _is_supported_url,
    _url_matches_blocked_domain,
)


class _LinkExtractor(HTMLParser):
    """Collect anchor href links from HTML."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        for key, value in attrs:
            if key == "href" and isinstance(value, str) and value.strip():
                self.links.append(value.strip())
                return


class WebCrawlTool(BaseTool):
    """Crawl a small set of linked pages starting from one URL."""

    _TEXT_TYPES = (
        "text/",
        "application/xhtml+xml",
    )

    def __init__(
        self,
        timeout: float = 20.0,
        max_bytes: int = 512 * 1024,
        max_pages_default: int = 5,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._timeout = timeout
        self._max_bytes = max_bytes
        self._max_pages_default = max(1, int(max_pages_default))
        self._client = client or make_default_client(timeout=timeout)
        self._owns_client = client is None

    @property
    def name(self) -> str:
        return "web_crawl"

    @property
    def description(self) -> str:
        return (
            "Crawl a small set of linked pages from a starting URL, usually within the same domain, "
            "and return extracted text for each visited page."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Starting HTTP or HTTPS URL."},
                "max_pages": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "default": self._max_pages_default,
                    "description": "Maximum number of pages to crawl.",
                },
                "max_depth": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 5,
                    "default": 1,
                    "description": "Maximum link depth from the starting page.",
                },
                "same_domain_only": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to keep crawling on the same domain only.",
                },
                "max_bytes": {
                    "type": "integer",
                    "minimum": 1024,
                    "maximum": 1048576,
                    "default": self._max_bytes,
                    "description": "Maximum response bytes to read per page.",
                },
            },
            "required": ["url"],
            "additionalProperties": False,
        }

    @property
    def is_read_only(self) -> bool:
        return True

    @property
    def is_open_world(self) -> bool:
        return True

    @property
    def tool_capabilities(self) -> dict[str, Any]:
        return {
            "domains": ["web"],
            "retrieval_modes": ["crawl"],
            "preference_tags": [
                "open_web",
                "multi_page_retrieval",
                "site_crawl",
            ],
            "read_only": self.is_read_only,
            "destructive": self.is_destructive,
            "open_world": self.is_open_world,
        }

    @property
    def is_concurrency_safe(self) -> bool:
        return True

    @property
    def search_hint(self) -> str | None:
        return "Use this to collect a few related pages from one site before summarizing or extracting evidence."

    async def execute(
        self,
        *,
        context: ToolExecutionContext | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        url = str(kwargs.get("url", "")).strip()
        if not url:
            return ToolResult(error="`url` must not be empty.")
        if not _is_supported_url(url):
            return ToolResult(error="`url` must be an HTTP or HTTPS URL.")
        blocked_domains = _blocked_domains_from_context(context)
        if _url_matches_blocked_domain(url, blocked_domains):
            return _blocked_url_tool_result(url, blocked_domains)

        max_pages = int(kwargs.get("max_pages", self._max_pages_default))
        max_depth = int(kwargs.get("max_depth", 1))
        same_domain_only = bool(kwargs.get("same_domain_only", True))
        max_bytes = int(kwargs.get("max_bytes", self._max_bytes))
        if max_pages <= 0:
            return ToolResult(error="`max_pages` must be greater than 0.")
        if max_depth < 0:
            return ToolResult(error="`max_depth` must be greater than or equal to 0.")
        if max_bytes <= 0:
            return ToolResult(error="`max_bytes` must be greater than 0.")

        origin_host = (urlparse(url).hostname or "").lower()
        queue: deque[tuple[str, int]] = deque([(url, 0)])
        visited: set[str] = set()
        pages: list[dict[str, Any]] = []
        truncated = False
        blocked_urls: list[str] = []

        while queue and len(pages) < max_pages:
            current_url, depth = queue.popleft()
            if current_url in visited:
                continue
            visited.add(current_url)
            if _url_matches_blocked_domain(current_url, blocked_domains):
                blocked_urls.append(current_url)
                continue

            fetched = await self._fetch_page(current_url, max_bytes=max_bytes)
            if fetched.error is not None:
                if not pages:
                    return fetched
                continue

            page_output = fetched.output if isinstance(fetched.output, dict) else {}
            pages.append(
                {
                    "url": current_url,
                    "text": str(page_output.get("text", "")),
                }
            )
            if depth >= max_depth:
                continue

            links = page_output.get("links", [])
            if not isinstance(links, list):
                continue
            for link in links:
                if not isinstance(link, str) or not _is_supported_url(link):
                    continue
                if same_domain_only and (urlparse(link).hostname or "").lower() != origin_host:
                    continue
                if _url_matches_blocked_domain(link, blocked_domains):
                    blocked_urls.append(link)
                    continue
                if link in visited:
                    continue
                queue.append((link, depth + 1))

        if queue:
            truncated = True

        return ToolResult(
            output={"pages": pages},
            metadata={
                "start_url": url,
                "pages_crawled": len(pages),
                "visited_urls": [page["url"] for page in pages],
                "max_depth": max_depth,
                "same_domain_only": same_domain_only,
                "truncated": truncated,
                "blocked_domains": blocked_domains,
                "blocked_urls": blocked_urls,
            },
        )

    async def _fetch_page(self, url: str, *, max_bytes: int) -> ToolResult:
        try:
            response = await http_request(self._client, "GET", url, max_retries=2)
        except ToolHttpError as exc:
            return error_to_tool_result(
                exc,
                extra_metadata={"url": url},
                suggestion="Check the URL or reduce crawl scope.",
            )

        content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
        if content_type and not content_type.startswith(self._TEXT_TYPES):
            return ToolResult(
                error=f"Unsupported content type: {content_type}",
                metadata={"url": str(response.url), "content_type": content_type},
            )

        body = response.content[:max_bytes]
        encoding = response.encoding or "utf-8"
        html = body.decode(encoding, errors="replace")
        extracted = _extract_with_trafilatura(html, output_format="text", url=str(response.url))
        if extracted is None:
            extracted = _extract_with_htmlparser(html)

        parser = _LinkExtractor()
        parser.feed(html)
        links = [
            normalized
            for normalized in (
                urljoin(str(response.url), href)
                for href in parser.links
            )
            if _is_supported_url(normalized)
        ]

        return ToolResult(
            output={
                "text": extracted,
                "links": links,
            },
            metadata={
                "url": str(response.url),
                "content_type": content_type,
                "bytes_read": len(body),
                "truncated": len(response.content) > max_bytes,
            },
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()
