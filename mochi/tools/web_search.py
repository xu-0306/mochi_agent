"""\u7db2\u9801\u641c\u5c0b\u5de5\u5177 \u2014 \u591a provider + fallback chain\u3002"""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import httpx

from mochi.diagnostics.fallbacks import append_fallback_diagnostic
from mochi.tools._http import (
    DEFAULT_USER_AGENT,
    ToolHttpError,
    error_to_tool_result,
    http_request,
    make_default_client,
)
from mochi.tools.base import BaseTool, ToolExecutionContext, ToolResult
from mochi.tools.web_search_providers import (
    get_web_search_provider_spec,
    iter_web_search_provider_specs,
    normalize_web_search_provider,
)

# ---------------------------------------------------------------------------
# \u5171\u7528\u5e38\u6578\u8207\u578b\u5225
# ---------------------------------------------------------------------------

_SUPPORTED_ENGINES = frozenset(
    spec.canonical_name
    for spec in iter_web_search_provider_specs()
)
_EMERGENCY_FALLBACK_ENGINES = ("jina", "duckduckgo_html")


@dataclass
class SearchResult:
    """\u641c\u5c0b\u7d50\u679c\u9805\u76ee\u3002"""

    title: str
    url: str
    snippet: str = ""
    content: str = ""


# ---------------------------------------------------------------------------
# DuckDuckGo HTML parser (\u7e7c\u627f\u820a\u5be6\u4f5c)
# ---------------------------------------------------------------------------


class _DuckDuckGoHTMLParser(HTMLParser):
    """\u89e3\u6790 DuckDuckGo HTML \u641c\u5c0b\u9801\u7d50\u679c\u3002"""

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
            self._start_title(attrs_dict.get("href", "") or "")
            return

        if tag == "a" and "result-link" in class_name:
            self._start_title(attrs_dict.get("href", "") or "")
            return

        if tag == "a" and attrs_dict.get("rel") == "nofollow":
            self._start_title(attrs_dict.get("href", "") or "")
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

    def _start_title(self, href: str) -> None:
        self._in_title = True
        self._current_href = href
        self._current_title = []
        self._current_snippet = []


class _TextExtractor(HTMLParser):
    """\u5c07\u7c21\u55ae HTML \u7247\u6bb5\u8f49\u70ba\u7d14\u6587\u5b57\u3002"""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def text(self) -> str:
        return " ".join(" ".join(self.parts).split())


def _resolve_duckduckgo_url(raw_url: str) -> str:
    """\u5c07 DuckDuckGo redirect URL \u9084\u539f\u70ba\u539f\u59cb\u7db2\u5740\u3002"""
    if not raw_url:
        return raw_url

    parsed = urlparse(raw_url)
    if parsed.path.startswith("/l/"):
        query = parse_qs(parsed.query)
        uddg = query.get("uddg")
        if uddg:
            return unquote(uddg[0])
    return raw_url


def _is_duckduckgo_challenge(html: str) -> bool:
    """\u5224\u65b7 DuckDuckGo \u662f\u5426\u56de\u50b3 anomaly/bot challenge\u3002"""
    lowered = html.lower()
    return "anomaly.js" in lowered or "challenge-form" in lowered or "botnet" in lowered


def _strip_html(value: str) -> str:
    if not value:
        return ""
    parser = _TextExtractor()
    parser.feed(value)
    return parser.text()


def _truncate_text(value: str, max_chars: int = 300) -> str:
    text = " ".join(value.split()).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _exa_snippet(item: dict[str, Any]) -> str:
    highlights = item.get("highlights")
    if isinstance(highlights, list):
        for entry in highlights:
            if isinstance(entry, str) and entry.strip():
                return _strip_html(entry)
            if isinstance(entry, dict):
                candidate = entry.get("text") or entry.get("highlight") or entry.get("content")
                if isinstance(candidate, str) and candidate.strip():
                    return _strip_html(candidate)

    summary = item.get("summary")
    if isinstance(summary, str) and summary.strip():
        return _strip_html(summary)
    if isinstance(summary, list):
        for entry in summary:
            if isinstance(entry, str) and entry.strip():
                return _strip_html(entry)

    text_value = item.get("text")
    if isinstance(text_value, str) and text_value.strip():
        return _truncate_text(_strip_html(text_value))
    return ""


# ---------------------------------------------------------------------------
# WebSearchTool
# ---------------------------------------------------------------------------


class WebSearchTool(BaseTool):
    """\u4f7f\u7528\u53ef\u66ff\u63db provider \u9032\u884c\u7db2\u9801\u641c\u5c0b\uff0c\u652f\u63f4 fallback chain\u3002"""

    def __init__(
        self,
        engine: str = "tavily",
        timeout: float = 15.0,
        client: httpx.AsyncClient | None = None,
        *,
        fallback_engines: list[str] | None = None,
        searxng_base_url: str | None = None,
        brave_api_key: str | None = None,
        tavily_api_key: str | None = None,
        serper_api_key: str | None = None,
        jina_api_key: str | None = None,
        exa_api_key: str | None = None,
        language: str | None = None,
        region: str | None = None,
    ) -> None:
        self._engine = _normalize_engine(engine)
        self._fallback_engines = [
            _normalize_engine(e)
            for e in (fallback_engines or [])
        ]
        self._timeout = timeout
        self._client = client or make_default_client(timeout=timeout)
        self._owns_client = client is None
        self._searxng_base_url = searxng_base_url.rstrip("/") if searxng_base_url else None
        self._brave_api_key = brave_api_key
        self._tavily_api_key = tavily_api_key
        self._serper_api_key = serper_api_key
        self._jina_api_key = jina_api_key
        self._exa_api_key = exa_api_key
        self._language = language
        self._region = region

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web for current information. Returns titles, URLs, snippets, "
            "and optionally full page content. Use this tool when you need to find "
            "up-to-date information, verify facts, or discover relevant sources. "
            "For reading a specific known URL, use web_fetch instead."
        )

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
            "retrieval_modes": ["search"],
            "preference_tags": [
                "open_web",
                "current_information",
                "source_discovery",
            ],
            "read_only": self.is_read_only,
            "destructive": self.is_destructive,
            "open_world": self.is_open_world,
        }

    @property
    def search_hint(self) -> str | None:
        return "Use this tool for current information or to discover sources before fetching a page."

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
                "search_depth": {
                    "type": "string",
                    "enum": ["basic", "advanced"],
                    "default": "basic",
                    "description": (
                        "Search depth. 'advanced' returns more detailed results "
                        "(Tavily only, other engines ignore this)."
                    ),
                },
                "include_content": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "If true, return full page content with results "
                        "(when supported by the search engine)."
                    ),
                },
                "allowed_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional allowlist of result domains.",
                },
                "blocked_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional denylist of result domains.",
                },
                "language": {
                    "type": "string",
                    "description": "Optional search language override.",
                },
                "region": {
                    "type": "string",
                    "description": "Optional search region override.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        *,
        context: ToolExecutionContext | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """\u57f7\u884c\u7db2\u9801\u641c\u5c0b\uff0c\u5931\u6557\u6642\u81ea\u52d5 fallback\u3002"""
        query = str(kwargs.get("query", "")).strip()
        top_k_result = _coerce_top_k(kwargs.get("top_k", 5))
        if isinstance(top_k_result, ToolResult):
            return top_k_result
        top_k = top_k_result
        search_depth = str(kwargs.get("search_depth", "basic"))
        include_content = bool(kwargs.get("include_content", False))
        allowed_domains = _coerce_domain_list(kwargs.get("allowed_domains"))
        blocked_domains = _merge_domain_lists(
            _coerce_domain_list(kwargs.get("blocked_domains")),
            _blocked_domains_from_context(context),
        )
        language = str(kwargs.get("language", "")).strip() or self._language
        region = str(kwargs.get("region", "")).strip() or self._region

        if not query:
            return ToolResult(error="`query` must not be empty.")

        # \u5617\u8a66\u4e3b\u8981\u5f15\u64ce + fallback chain + no-key emergency fallback
        engines_to_try = self._provider_chain()
        last_error: ToolResult | None = None
        attempted_providers: list[str] = []
        provider_attempts: list[dict[str, Any]] = []
        warnings: list[dict[str, str]] = []
        fallback_diagnostics: list[dict[str, Any]] = []

        for index, engine in enumerate(engines_to_try):
            if engine not in _SUPPORTED_ENGINES:
                continue
            provider_state = self._provider_state(engine)
            next_engine = _next_provider_in_chain(engines_to_try, start=index + 1)
            if not provider_state["configured"]:
                warning = {
                    "provider": engine,
                    "status": str(provider_state["status"]),
                    "reason": str(provider_state["reason"]),
                }
                provider_attempts.append(dict(warning))
                warnings.append(warning)
                append_fallback_diagnostic(
                    fallback_diagnostics,
                    category="provider_chain",
                    name="search_provider_skipped",
                    reason=str(provider_state["status"]),
                    kind="skip",
                    severity="info",
                    from_state=engine,
                    to_state=next_engine,
                    metadata={"detail": str(provider_state["reason"])},
                )
                continue
            attempted_providers.append(engine)

            result = await self._search_with_engine(
                engine=engine,
                query=query,
                top_k=top_k,
                search_depth=search_depth,
                include_content=include_content,
                allowed_domains=allowed_domains,
                blocked_domains=blocked_domains,
                language=language,
                region=region,
            )
            if result.error is None:
                provider_attempts.append({
                    "provider": engine,
                    "status": "succeeded",
                })
                return _normalize_search_tool_result(
                    result=result,
                    query=query,
                    provider=engine,
                    attempted_providers=attempted_providers,
                    warnings=warnings,
                    provider_attempts=provider_attempts,
                    fallback_diagnostics=fallback_diagnostics,
                )
            provider_attempts.append({
                "provider": engine,
                "status": "request_failed",
                "reason": result.error or "request failed",
            })
            append_fallback_diagnostic(
                fallback_diagnostics,
                category="provider_chain",
                name="search_provider_fallback",
                reason="request_failed",
                kind="fallback",
                severity="warning",
                from_state=engine,
                to_state=next_engine,
                metadata={
                    "error": result.error or "request failed",
                    "retryable": result.retryable,
                },
            )
            last_error = result

        if last_error is not None:
            last_error.metadata.setdefault("provider_attempts", provider_attempts)
            last_error.metadata.setdefault("warnings", warnings)
            last_error.metadata.setdefault("attempted_providers", attempted_providers)
            last_error.metadata.setdefault("fallback_diagnostics", fallback_diagnostics)
            return last_error

        return ToolResult(
            error="No configured search engine is available.",
            metadata={
                "attempted_providers": attempted_providers,
                "provider_attempts": provider_attempts,
                "warnings": warnings,
                "fallback_diagnostics": fallback_diagnostics,
            },
            suggestion=(
                "Configure at least one search engine API key in settings. "
                "Recommended: set MOCHI_TAVILY_API_KEY for reliable web search."
            ),
        )

    def _provider_chain(self) -> list[str]:
        primary_chain: list[str] = []
        for engine in [self._engine, *self._fallback_engines]:
            normalized = _normalize_engine(engine)
            if normalized in _SUPPORTED_ENGINES and normalized not in primary_chain:
                primary_chain.append(normalized)

        if any(self._provider_state(engine)["configured"] for engine in primary_chain):
            return primary_chain

        ordered = list(primary_chain)
        for engine in _EMERGENCY_FALLBACK_ENGINES:
            normalized = _normalize_engine(engine)
            if normalized in _SUPPORTED_ENGINES and normalized not in ordered:
                ordered.append(normalized)
        return ordered

    def _provider_state(self, engine: str) -> dict[str, Any]:
        """Describe whether one provider is usable or why it is skipped."""
        spec = get_web_search_provider_spec(engine)
        if spec is None:
            return {
                "configured": False,
                "status": "skipped_unsupported",
                "reason": "unsupported provider",
            }
        if spec.base_url_config_field:
            if isinstance(self._searxng_base_url, str) and self._searxng_base_url.strip():
                return {"configured": True, "status": "configured", "reason": ""}
            return {
                "configured": False,
                "status": "skipped_missing_config",
                "reason": f"{spec.base_url_config_field} is not configured",
            }
        secret = self._provider_secret_value(engine)
        if secret:
            return {"configured": True, "status": "configured", "reason": ""}
        if spec.no_key_supported:
            return {
                "configured": True,
                "status": "configured_no_key",
                "reason": "provider supports no-key access",
            }
        if spec.key_config_field:
            return {
                "configured": False,
                "status": "skipped_missing_key",
                "reason": f"{spec.key_config_field} is not configured",
            }
        return {
            "configured": False,
            "status": "skipped_missing_config",
            "reason": "provider is not configured",
        }

    def _provider_secret_value(self, engine: str) -> str | None:
        if engine == "tavily":
            return self._tavily_api_key
        if engine == "serper":
            return self._serper_api_key
        if engine == "brave":
            return self._brave_api_key
        if engine == "jina":
            return self._jina_api_key
        if engine == "exa":
            return self._exa_api_key
        return None

    async def _search_with_engine(
        self,
        *,
        engine: str,
        query: str,
        top_k: int,
        search_depth: str,
        include_content: bool,
        allowed_domains: list[str],
        blocked_domains: list[str],
        language: str | None,
        region: str | None,
    ) -> ToolResult:
        """\u4f7f\u7528\u6307\u5b9a engine \u57f7\u884c\u641c\u5c0b\u3002"""
        try:
            if engine == "tavily":
                return await self._search_tavily(
                    query=query, top_k=top_k,
                    search_depth=search_depth,
                    include_content=include_content,
                    allowed_domains=allowed_domains,
                    blocked_domains=blocked_domains,
                )
            if engine == "serper":
                return await self._search_serper(
                    query=query,
                    top_k=top_k,
                    allowed_domains=allowed_domains,
                    blocked_domains=blocked_domains,
                    language=language,
                    region=region,
                )
            if engine == "jina":
                return await self._search_jina(
                    query=query,
                    top_k=top_k,
                    allowed_domains=allowed_domains,
                    blocked_domains=blocked_domains,
                )
            if engine == "exa":
                return await self._search_exa(
                    query=query,
                    top_k=top_k,
                    include_content=include_content,
                    allowed_domains=allowed_domains,
                    blocked_domains=blocked_domains,
                )
            if engine == "brave":
                return await self._search_brave(
                    query=query,
                    top_k=top_k,
                    allowed_domains=allowed_domains,
                    blocked_domains=blocked_domains,
                    language=language,
                    region=region,
                )
            if engine == "searxng":
                return await self._search_searxng(
                    query=query,
                    top_k=top_k,
                    allowed_domains=allowed_domains,
                    blocked_domains=blocked_domains,
                )
            if engine == "duckduckgo_html":
                return await self._search_duckduckgo_html(
                    query=query,
                    top_k=top_k,
                    allowed_domains=allowed_domains,
                    blocked_domains=blocked_domains,
                )
            return ToolResult(error=f"Unsupported search engine: {engine}")
        except ToolHttpError as exc:
            return error_to_tool_result(
                exc,
                extra_metadata={"query": query, "engine": engine},
                suggestion="Try again later or switch to a different search engine.",
            )

    # -----------------------------------------------------------------------
    # Tavily (agent-native)
    # -----------------------------------------------------------------------

    async def _search_tavily(
        self,
        *,
        query: str,
        top_k: int,
        search_depth: str,
        include_content: bool,
        allowed_domains: list[str],
        blocked_domains: list[str],
    ) -> ToolResult:
        """\u4f7f\u7528 Tavily Search API\u3002"""
        body: dict[str, Any] = {
            "query": query,
            "max_results": top_k,
            "search_depth": search_depth if search_depth in {"basic", "advanced"} else "basic",
            "include_raw_content": include_content,
        }
        if allowed_domains:
            body["include_domains"] = allowed_domains
        if blocked_domains:
            body["exclude_domains"] = blocked_domains

        response = await http_request(
            self._client,
            "POST",
            "https://api.tavily.com/search",
            json=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._tavily_api_key}",
            },
        )
        payload = response.json()
        if not isinstance(payload, dict):
            return ToolResult(
                error="Tavily returned an invalid response.",
                metadata={"query": query, "engine": "tavily"},
            )

        raw_results = payload.get("results", [])
        results: list[SearchResult] = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            if not title or not url:
                continue
            results.append(SearchResult(
                title=title,
                url=url,
                snippet=str(item.get("content") or ""),
                content=str(item.get("raw_content") or "") if include_content else "",
            ))
            if len(results) >= top_k:
                break

        return _results_to_tool_result(
            results=results,
            query=query,
            engine="tavily",
            status_code=response.status_code,
            include_content=include_content,
            allowed_domains=allowed_domains,
            blocked_domains=blocked_domains,
        )

    # -----------------------------------------------------------------------
    # Serper (Google SERP)
    # -----------------------------------------------------------------------

    async def _search_serper(
        self,
        *,
        query: str,
        top_k: int,
        allowed_domains: list[str],
        blocked_domains: list[str],
        language: str | None,
        region: str | None,
    ) -> ToolResult:
        """\u4f7f\u7528 Serper.dev Google Search API\u3002"""
        body: dict[str, Any] = {"q": query, "num": top_k}
        if language:
            body["hl"] = language
        if region:
            body["gl"] = region

        response = await http_request(
            self._client,
            "POST",
            "https://google.serper.dev/search",
            json=body,
            headers={
                "X-API-KEY": self._serper_api_key or "",
                "Content-Type": "application/json",
            },
        )
        payload = response.json()
        if not isinstance(payload, dict):
            return ToolResult(
                error="Serper returned an invalid response.",
                metadata={"query": query, "engine": "serper"},
            )

        raw_organic = payload.get("organic", [])
        results: list[SearchResult] = []
        for item in raw_organic:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            url = str(item.get("link") or "").strip()
            if not title or not url:
                continue
            results.append(SearchResult(
                title=title,
                url=url,
                snippet=str(item.get("snippet") or ""),
            ))
            if len(results) >= top_k:
                break

        return _results_to_tool_result(
            results=results,
            query=query,
            engine="serper",
            status_code=response.status_code,
            allowed_domains=allowed_domains,
            blocked_domains=blocked_domains,
        )

    # -----------------------------------------------------------------------
    # Jina s.jina.ai
    # -----------------------------------------------------------------------

    async def _search_jina(
        self,
        *,
        query: str,
        top_k: int,
        allowed_domains: list[str],
        blocked_domains: list[str],
    ) -> ToolResult:
        """\u4f7f\u7528 Jina AI s.jina.ai \u641c\u5c0b API\u3002"""
        headers: dict[str, str] = {"Accept": "application/json"}
        if self._jina_api_key:
            headers["Authorization"] = f"Bearer {self._jina_api_key}"

        response = await http_request(
            self._client,
            "GET",
            f"https://s.jina.ai/{query}",
            headers=headers,
        )
        payload = response.json()
        if not isinstance(payload, dict):
            return ToolResult(
                error="Jina search returned an invalid response.",
                metadata={"query": query, "engine": "jina"},
            )

        raw_data = payload.get("data", [])
        results: list[SearchResult] = []
        for item in raw_data:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            if not title or not url:
                continue
            results.append(SearchResult(
                title=title,
                url=url,
                snippet=str(item.get("description") or item.get("content") or ""),
            ))
            if len(results) >= top_k:
                break

        return _results_to_tool_result(
            results=results,
            query=query,
            engine="jina",
            status_code=response.status_code,
            allowed_domains=allowed_domains,
            blocked_domains=blocked_domains,
        )

    # -----------------------------------------------------------------------
    # Exa Search API
    # -----------------------------------------------------------------------

    async def _search_exa(
        self,
        *,
        query: str,
        top_k: int,
        include_content: bool,
        allowed_domains: list[str],
        blocked_domains: list[str],
    ) -> ToolResult:
        """使用 Exa Search API 搜尋。"""
        contents: dict[str, Any] = {
            "highlights": True,
            "summary": True,
        }
        if include_content:
            contents["text"] = True

        response = await http_request(
            self._client,
            "POST",
            "https://api.exa.ai/search",
            json={
                "query": query,
                "numResults": top_k,
                "contents": contents,
            },
            headers={
                "Content-Type": "application/json",
                "x-api-key": self._exa_api_key or "",
            },
        )
        payload = response.json()
        if not isinstance(payload, dict):
            return ToolResult(
                error="Exa returned an invalid response.",
                metadata={"query": query, "engine": "exa"},
            )

        raw_results = payload.get("results", [])
        if not isinstance(raw_results, list):
            raw_results = []

        results: list[SearchResult] = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            if not title or not url:
                continue
            text_value = item.get("text")
            content = text_value.strip() if include_content and isinstance(text_value, str) else ""
            results.append(SearchResult(
                title=title,
                url=url,
                snippet=_exa_snippet(item),
                content=content,
            ))
            if len(results) >= top_k:
                break

        return _results_to_tool_result(
            results=results,
            query=query,
            engine="exa",
            status_code=response.status_code,
            include_content=include_content,
            allowed_domains=allowed_domains,
            blocked_domains=blocked_domains,
        )

    # -----------------------------------------------------------------------
    # Brave Search API (\u7e7c\u627f\u820a\u5be6\u4f5c + retry)
    # -----------------------------------------------------------------------

    async def _search_brave(
        self,
        *,
        query: str,
        top_k: int,
        allowed_domains: list[str],
        blocked_domains: list[str],
        language: str | None,
        region: str | None,
    ) -> ToolResult:
        """\u4f7f\u7528 Brave Search API \u641c\u5c0b\u3002"""
        params: dict[str, Any] = {"q": query, "count": top_k}
        if language:
            params["search_lang"] = language
        if region:
            params["country"] = region

        response = await http_request(
            self._client,
            "GET",
            "https://api.search.brave.com/res/v1/web/search",
            params=params,
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": self._brave_api_key or "",
            },
        )
        payload = response.json()
        if not isinstance(payload, dict):
            return ToolResult(
                error="Brave returned an invalid JSON payload.",
                metadata={"query": query, "engine": "brave"},
            )

        web_payload = payload.get("web", {})
        raw_results: Any = []
        if isinstance(web_payload, dict):
            raw_results = web_payload.get("results", [])
        if not isinstance(raw_results, list):
            raw_results = []

        results: list[SearchResult] = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            if not title or not url:
                continue
            results.append(SearchResult(
                title=title,
                url=url,
                snippet=_strip_html(str(item.get("description") or "")),
            ))
            if len(results) >= top_k:
                break

        return _results_to_tool_result(
            results=results,
            query=query,
            engine="brave",
            status_code=response.status_code,
            allowed_domains=allowed_domains,
            blocked_domains=blocked_domains,
        )

    # -----------------------------------------------------------------------
    # SearXNG JSON API (\u7e7c\u627f\u820a\u5be6\u4f5c + retry)
    # -----------------------------------------------------------------------

    async def _search_searxng(
        self,
        *,
        query: str,
        top_k: int,
        allowed_domains: list[str],
        blocked_domains: list[str],
    ) -> ToolResult:
        """\u4f7f\u7528 SearXNG JSON API \u641c\u5c0b\u3002"""
        if not self._searxng_base_url:
            return ToolResult(
                error="SearXNG search requires `tools.web_search_searxng_base_url`.",
                metadata={"query": query, "engine": "searxng", "missing_config": True},
            )

        response = await http_request(
            self._client,
            "GET",
            urljoin(f"{self._searxng_base_url}/", "search"),
            params={"q": query, "format": "json"},
            headers={"Accept": "application/json"},
        )
        payload = response.json()
        if not isinstance(payload, dict):
            return ToolResult(
                error="SearXNG returned an invalid JSON payload.",
                metadata={"query": query, "engine": "searxng"},
            )

        raw_results = payload.get("results", [])
        if not isinstance(raw_results, list):
            raw_results = []

        results: list[SearchResult] = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            if not title or not url:
                continue
            results.append(SearchResult(
                title=title,
                url=url,
                snippet=_strip_html(str(item.get("content") or "")),
            ))
            if len(results) >= top_k:
                break

        return _results_to_tool_result(
            results=results,
            query=query,
            engine="searxng",
            status_code=response.status_code,
            allowed_domains=allowed_domains,
            blocked_domains=blocked_domains,
        )

    # -----------------------------------------------------------------------
    # DuckDuckGo HTML (\u7e7c\u627f\u820a\u5be6\u4f5c + retry)
    # -----------------------------------------------------------------------

    async def _search_duckduckgo_html(
        self,
        *,
        query: str,
        top_k: int,
        allowed_domains: list[str],
        blocked_domains: list[str],
    ) -> ToolResult:
        """\u4f7f\u7528 DuckDuckGo HTML \u9801\u9762\u505a best-effort \u641c\u5c0b\u3002"""
        try:
            response = await http_request(
                self._client,
                "GET",
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers={
                    "User-Agent": DEFAULT_USER_AGENT,
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
                },
                max_retries=1,
            )
        except ToolHttpError as exc:
            return error_to_tool_result(
                exc,
                extra_metadata={"query": query, "engine": "duckduckgo_html"},
            )

        status_code = response.status_code
        if _is_duckduckgo_challenge(response.text):
            return ToolResult(
                error=(
                    "DuckDuckGo returned an anti-bot challenge instead of search results. "
                    "Configure a Tavily, Serper, or Brave search backend for reliable web search."
                ),
                retryable=True,
                metadata={
                    "query": query,
                    "engine": "duckduckgo_html",
                    "status_code": status_code,
                    "blocked": True,
                },
                suggestion="Set MOCHI_TAVILY_API_KEY for reliable search.",
            )

        parser = _DuckDuckGoHTMLParser()
        parser.feed(response.text)
        return _results_to_tool_result(
            results=parser.results[:top_k],
            query=query,
            engine="duckduckgo_html",
            status_code=status_code,
            allowed_domains=allowed_domains,
            blocked_domains=blocked_domains,
        )

    async def close(self) -> None:
        """\u95dc\u9589\u5167\u90e8 HTTP client\u3002"""
        if self._owns_client:
            await self._client.aclose()

    def format_result_for_model(
        self,
        result: ToolResult,
        *,
        max_chars: int = 2000,
    ) -> str:
        if result.error is not None or not isinstance(result.output, dict):
            return super().format_result_for_model(result, max_chars=max_chars)

        results = result.output.get("results")
        if not isinstance(results, list):
            return super().format_result_for_model(result, max_chars=max_chars)

        citations: list[str] = []
        for index, item in enumerate(results, start=1):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip() or f"Result {index}"
            url = str(item.get("url") or "").strip()
            snippet = str(item.get("snippet") or "").strip()
            citations.append(f"[{index}] {title} - {url} - {snippet}")

        payload = ToolResult(
            output={
                **result.output,
                "citations": citations,
            },
            error=None,
            metadata=result.metadata,
            retryable=result.retryable,
            suggestion=result.suggestion,
        )
        return super().format_result_for_model(payload, max_chars=max_chars)


# ---------------------------------------------------------------------------
# \u5167\u90e8\u8f14\u52a9
# ---------------------------------------------------------------------------


def _normalize_engine(engine: str) -> str:
    return normalize_web_search_provider(engine)


def _coerce_top_k(value: Any) -> int | ToolResult:
    try:
        top_k = int(value)
    except (TypeError, ValueError):
        return ToolResult(error="`top_k` must be an integer.")
    return max(1, min(top_k, 10))


def _coerce_domain_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    domains: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        domain = item.strip().lower().lstrip(".").rstrip(".")
        if domain:
            domains.append(domain)
    return domains


def _merge_domain_lists(*domain_lists: list[str]) -> list[str]:
    merged: list[str] = []
    for domain_list in domain_lists:
        for domain in domain_list:
            if domain and domain not in merged:
                merged.append(domain)
    return merged


def _blocked_domains_from_context(context: ToolExecutionContext | None) -> list[str]:
    if context is None:
        return []
    return _coerce_domain_list(context.permission_policy.get("blocked_web_domains"))


def _domain_allowed(url: str, allowed_domains: list[str], blocked_domains: list[str]) -> bool:
    host = (urlparse(url).hostname or "").lower().rstrip(".")
    if not host:
        return False
    if allowed_domains and not any(host == domain or host.endswith(f".{domain}") for domain in allowed_domains):
        return False
    if any(host == domain or host.endswith(f".{domain}") for domain in blocked_domains):
        return False
    return True


def _results_to_tool_result(
    *,
    results: list[SearchResult],
    query: str,
    engine: str,
    status_code: int | None = None,
    include_content: bool = False,
    allowed_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
) -> ToolResult:
    allowed_domains = allowed_domains or []
    blocked_domains = blocked_domains or []
    results = [
        item for item in results
        if _domain_allowed(item.url, allowed_domains, blocked_domains)
    ]
    if not results:
        return ToolResult(
            error="Search returned no parseable results.",
            retryable=True,
            metadata={
                "query": query,
                "engine": engine,
                "status_code": status_code,
                "count": 0,
            },
            suggestion="Try different or more specific search keywords.",
        )

    output = []
    for item in results:
        entry: dict[str, str] = {
            "title": item.title,
            "url": item.url,
            "snippet": item.snippet,
        }
        if include_content and item.content:
            entry["content"] = item.content
        output.append(entry)

    return ToolResult(
        output=output,
        metadata={
            "query": query,
            "engine": engine,
            "status_code": status_code,
            "count": len(results),
            "allowed_domains": allowed_domains,
            "blocked_domains": blocked_domains,
        },
    )


def _normalize_search_tool_result(
    *,
    result: ToolResult,
    query: str,
    provider: str,
    attempted_providers: list[str],
    warnings: list[dict[str, str]],
    provider_attempts: list[dict[str, Any]],
    fallback_diagnostics: list[dict[str, Any]],
) -> ToolResult:
    raw_results = result.output if isinstance(result.output, list) else []
    normalized_results: list[dict[str, Any]] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        entry = {
            "title": str(item.get("title") or "").strip(),
            "url": str(item.get("url") or "").strip(),
            "snippet": str(item.get("snippet") or "").strip(),
        }
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            entry["content"] = content
        if entry["title"] and entry["url"]:
            normalized_results.append(entry)

    result.output = {
        "query": query,
        "provider": provider,
        "results": normalized_results,
        "warnings": warnings,
        "attempted_providers": attempted_providers,
    }
    result.metadata["provider"] = provider
    result.metadata["attempted_providers"] = attempted_providers
    result.metadata["provider_attempts"] = provider_attempts
    result.metadata["warnings"] = warnings
    result.metadata["fallback_diagnostics"] = fallback_diagnostics
    return result


def _next_provider_in_chain(engines: list[str], *, start: int) -> str | None:
    for engine in engines[start:]:
        if engine in _SUPPORTED_ENGINES:
            return engine
    return None
