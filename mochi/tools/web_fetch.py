"""\u7db2\u9801\u5167\u5bb9\u64f7\u53d6\u5de5\u5177 \u2014 trafilatura + Jina Reader fallback\u3002"""

from __future__ import annotations

from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlparse

import httpx
try:
    from loguru import logger
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal test envs
    import logging

    logger = logging.getLogger(__name__)

from mochi.tools._http import (
    ToolHttpError,
    error_to_tool_result,
    http_request,
    make_default_client,
)
from mochi.tools.base import BaseTool, ToolResult

# ---------------------------------------------------------------------------
# trafilatura optional import
# ---------------------------------------------------------------------------

_trafilatura_available = False
try:
    import trafilatura  # type: ignore[import-untyped]

    _trafilatura_available = True
except ImportError:
    trafilatura = None

# ---------------------------------------------------------------------------
# Stdlib HTML \u2192 text fallback (\u7e7c\u627f\u820a\u5be6\u4f5c)
# ---------------------------------------------------------------------------


class _ReadableHTMLParser(HTMLParser):
    """\u5c07 HTML \u8f49\u6210\u9069\u5408\u6a21\u578b\u95b1\u8b80\u7684\u7d14\u6587\u5b57\u3002"""

    _SKIP_TAGS = {"script", "style", "noscript", "template", "svg"}
    _BLOCK_TAGS = {
        "article", "aside", "blockquote", "br", "dd", "div", "dl", "dt",
        "figcaption", "footer", "h1", "h2", "h3", "h4", "h5", "h6",
        "header", "li", "main", "nav", "ol", "p", "pre", "section",
        "table", "td", "th", "tr", "ul",
    }

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth == 0 and tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if self._skip_depth == 0 and tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def text(self) -> str:
        """\u56de\u50b3\u58d3\u7e2e\u7a7a\u767d\u5f8c\u7684\u7d14\u6587\u5b57\u3002"""
        lines = [" ".join(line.split()) for line in "".join(self._parts).splitlines()]
        return "\n".join(line for line in lines if line).strip()


# ---------------------------------------------------------------------------
# \u63d0\u53d6\u7b56\u7565
# ---------------------------------------------------------------------------


def _extract_with_trafilatura(
    html: str,
    *,
    output_format: str = "text",
    url: str = "",
) -> str | None:
    """\u7528 trafilatura \u63d0\u53d6\u7db2\u9801\u6b63\u6587\u3002"""
    if not _trafilatura_available or trafilatura is None:
        return None

    try:
        kwargs: dict[str, Any] = {
            "include_comments": False,
            "include_tables": True,
        }
        if output_format == "markdown":
            kwargs["output_format"] = "markdown"

        result = trafilatura.extract(html, url=url, **kwargs)
        return result if isinstance(result, str) and result.strip() else None
    except Exception as exc:  # pragma: no cover - trafilatura \u5167\u90e8\u932f\u8aa4
        logger.debug(f"trafilatura extraction failed: {exc}")
        return None


def _extract_with_htmlparser(html: str) -> str:
    """\u7528 stdlib HTMLParser fallback \u63d0\u53d6\u3002"""
    parser = _ReadableHTMLParser()
    parser.feed(html)
    return parser.text()


# ---------------------------------------------------------------------------
# URL \u9a57\u8b49
# ---------------------------------------------------------------------------


def _is_supported_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


# ---------------------------------------------------------------------------
# WebFetchTool
# ---------------------------------------------------------------------------


class WebFetchTool(BaseTool):
    """\u64f7\u53d6\u7db2\u9801\u4e26\u8f49\u63db\u70ba\u53ef\u8b80\u6587\u5b57\u6216 Markdown\u3002"""

    _TEXT_TYPES = (
        "text/",
        "application/json",
        "application/xml",
        "application/atom+xml",
        "application/rss+xml",
        "application/xhtml+xml",
    )

    def __init__(
        self,
        timeout: float = 20.0,
        max_bytes: int = 512 * 1024,
        client: httpx.AsyncClient | None = None,
        *,
        jina_api_key: str | None = None,
        extractor: str = "trafilatura",
    ) -> None:
        self._timeout = timeout
        self._max_bytes = max_bytes
        self._client = client or make_default_client(timeout=timeout)
        self._owns_client = client is None
        self._jina_api_key = jina_api_key
        self._extractor = extractor  # "trafilatura" | "jina_reader" | "htmlparser"

    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return (
            "Fetch a web page and extract its main content as readable text or Markdown. "
            "Use after web_search to read a specific page, or when you have a known URL "
            "to inspect. Handles HTML, JSON, XML, and plain text. "
            "For PDF files, use a dedicated PDF tool instead."
        )

    @property
    def is_read_only(self) -> bool:
        return True

    @property
    def is_open_world(self) -> bool:
        return True

    @property
    def is_concurrency_safe(self) -> bool:
        return True

    @property
    def search_hint(self) -> str | None:
        return "Use this after search when you need the content of one known URL."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "HTTP or HTTPS URL to fetch.",
                },
                "output_format": {
                    "type": "string",
                    "enum": ["text", "markdown"],
                    "default": "text",
                    "description": (
                        "Output format. 'markdown' preserves headings, lists, and links."
                    ),
                },
                "max_bytes": {
                    "type": "integer",
                    "minimum": 1024,
                    "maximum": 1048576,
                    "default": self._max_bytes,
                    "description": "Maximum response bytes to read before truncating.",
                },
            },
            "required": ["url"],
            "additionalProperties": False,
        }

    async def execute(self, **kwargs: Any) -> ToolResult:
        """\u64f7\u53d6\u7db2\u5740\u5167\u5bb9\u3002"""
        url = str(kwargs.get("url", ""))
        output_format = str(kwargs.get("output_format", "text"))
        max_bytes_arg = kwargs.get("max_bytes")
        clean_url = url.strip()
        if not clean_url:
            return ToolResult(error="`url` must not be empty.")
        if not _is_supported_url(clean_url):
            return ToolResult(error="`url` must be an HTTP or HTTPS URL.")

        effective_max_bytes = (
            int(max_bytes_arg) if max_bytes_arg is not None else self._max_bytes
        )
        if effective_max_bytes <= 0:
            return ToolResult(error="`max_bytes` must be greater than 0.")

        # Layer 1: \u76f4\u63a5 HTTP \u64f7\u53d6 + trafilatura/htmlparser
        result = await self._fetch_direct(
            clean_url,
            output_format=output_format,
            max_bytes=effective_max_bytes,
        )
        if result.error is None and result.output:
            return result

        # Layer 2: Jina Reader API fallback (\u53ef\u8655\u7406 JS-rendered \u9801\u9762)
        if self._jina_api_key or self._extractor == "jina_reader":
            jina_result = await self._fetch_via_jina(clean_url)
            if jina_result.error is None:
                return jina_result

        # Layer 3: \u82e5\u76f4\u63a5\u64f7\u53d6\u6709\u5167\u5bb9\u4f46\u54c1\u8cea\u5dee\uff0c\u4ecd\u56de\u50b3
        if result.error is None:
            return result

        return result

    async def _fetch_direct(
        self,
        url: str,
        *,
        output_format: str,
        max_bytes: int,
    ) -> ToolResult:
        """\u76f4\u63a5 HTTP GET + \u672c\u5730\u63d0\u53d6\u3002"""
        try:
            response = await http_request(
                self._client, "GET", url, max_retries=2,
            )
        except ToolHttpError as exc:
            return error_to_tool_result(
                exc,
                extra_metadata={"url": url},
                suggestion="Check the URL is correct, or try again later.",
            )

        content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
        body = response.content[:max_bytes]
        truncated = len(response.content) > max_bytes

        if content_type and not content_type.startswith(self._TEXT_TYPES):
            return ToolResult(
                error=f"Unsupported content type: {content_type}",
                metadata={"url": str(response.url), "content_type": content_type},
                suggestion=f"This URL returned {content_type}. Use a specialized tool for this content type.",
            )

        encoding = response.encoding or "utf-8"
        text = body.decode(encoding, errors="replace")

        is_html = "html" in content_type or "<html" in text[:500].lower()
        if is_html:
            # \u5617\u8a66 trafilatura \u2192 HTMLParser fallback
            extracted = _extract_with_trafilatura(
                text, output_format=output_format, url=str(response.url),
            )
            if extracted is None:
                extracted = _extract_with_htmlparser(text)
            text = extracted

        return ToolResult(
            output=text,
            metadata={
                "url": str(response.url),
                "status_code": response.status_code,
                "content_type": content_type,
                "bytes_read": len(body),
                "truncated": truncated,
                "extractor": "trafilatura" if _trafilatura_available and is_html else "htmlparser",
            },
        )

    async def _fetch_via_jina(self, url: str) -> ToolResult:
        """\u900f\u904e Jina Reader API (r.jina.ai) \u64f7\u53d6 Markdown\u3002"""
        jina_url = f"https://r.jina.ai/{url}"
        headers: dict[str, str] = {"Accept": "text/markdown"}
        if self._jina_api_key:
            headers["Authorization"] = f"Bearer {self._jina_api_key}"

        try:
            response = await http_request(
                self._client, "GET", jina_url, max_retries=1, headers=headers,
            )
        except ToolHttpError as exc:
            return error_to_tool_result(
                exc,
                extra_metadata={"url": url, "extractor": "jina_reader"},
                suggestion="Jina Reader fallback failed. The page may not be accessible.",
            )

        text = response.text.strip()
        if not text:
            return ToolResult(
                error="Jina Reader returned empty content.",
                metadata={"url": url, "extractor": "jina_reader"},
            )

        return ToolResult(
            output=text,
            metadata={
                "url": url,
                "status_code": response.status_code,
                "extractor": "jina_reader",
                "bytes_read": len(response.content),
                "truncated": False,
            },
        )

    async def close(self) -> None:
        """\u95dc\u9589\u5167\u90e8 HTTP client\u3002"""
        if self._owns_client:
            await self._client.aclose()
