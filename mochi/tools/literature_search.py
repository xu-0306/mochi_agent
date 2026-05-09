"""文獻搜尋工具。"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from email.utils import parseaddr
from typing import Any, cast

import httpx

from mochi.tools._http import (
    TokenBucketRateLimiter,
    ToolHttpError,
    error_to_tool_result,
    http_request,
    make_default_client,
)
from mochi.tools.base import BaseTool, ToolResult

_ARXIV_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def _clean_text(value: str | None) -> str:
    return " ".join((value or "").split())


def _first_text(element: ET.Element, path: str, ns: dict[str, str] | None = None) -> str:
    found = element.find(path, ns or {})
    return _clean_text(found.text if found is not None else "")


def _validate_query(query: str) -> str | None:
    if not query.strip():
        return "`query` must not be empty."
    return None


def _clamp_limit(value: int, *, minimum: int = 1, maximum: int = 20) -> int:
    return max(minimum, min(value, maximum))


def _optional_email(value: str | None) -> str | None:
    if value is None:
        return None
    address = parseaddr(value.strip())[1]
    return address or None


def _as_dict(value: Any) -> dict[str, Any]:
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return cast(list[Any], value) if isinstance(value, list) else []


def _as_str(value: Any) -> str:
    return value if isinstance(value, str) else ""


class ArxivSearchTool(BaseTool):
    """Search arXiv papers through the official arXiv API."""

    def __init__(
        self,
        timeout: float = 20.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._client = client or make_default_client(timeout=timeout)
        self._owns_client = client is None
        self._rate_limiter = TokenBucketRateLimiter(rate=1.0 / 3.0, burst=1)

    @property
    def name(self) -> str:
        return "arxiv_search"

    @property
    def description(self) -> str:
        return (
            "Search arXiv papers using the official arXiv API. Returns paper metadata, "
            "abstracts, and PDF links."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "arXiv search query, such as all:retrieval augmented generation, "
                        "ti:transformer, au:vaswani, or cat:cs.CL."
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5,
                    "description": "Maximum number of papers to return.",
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["relevance", "lastUpdatedDate", "submittedDate"],
                    "default": "relevance",
                    "description": "arXiv sortBy parameter.",
                },
                "sort_order": {
                    "type": "string",
                    "enum": ["ascending", "descending"],
                    "default": "descending",
                    "description": "arXiv sortOrder parameter.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        }

    async def execute(self, **kwargs: Any) -> ToolResult:
        """執行 arXiv 搜尋。"""
        query = str(kwargs.get("query", ""))
        top_k = int(kwargs.get("top_k", 5))
        sort_by = str(kwargs.get("sort_by", "relevance"))
        sort_order = str(kwargs.get("sort_order", "descending"))
        if error := _validate_query(query):
            return ToolResult(error=error)
        if sort_by not in {"relevance", "lastUpdatedDate", "submittedDate"}:
            return ToolResult(error="`sort_by` must be relevance, lastUpdatedDate, or submittedDate.")
        if sort_order not in {"ascending", "descending"}:
            return ToolResult(error="`sort_order` must be ascending or descending.")

        limit = _clamp_limit(top_k)
        try:
            response = await http_request(
                self._client,
                "GET",
                "https://export.arxiv.org/api/query",
                rate_limiter=self._rate_limiter,
                params={
                    "search_query": query.strip(),
                    "start": 0,
                    "max_results": limit,
                    "sortBy": sort_by,
                    "sortOrder": sort_order,
                },
                max_retries=2,
            )
        except ToolHttpError as exc:
            return error_to_tool_result(
                exc,
                extra_metadata={"query": query.strip(), "source": "arxiv"},
                suggestion="arXiv API may be temporarily unavailable. Try again later.",
            )
        root = ET.fromstring(response.text)
        results = [self._parse_entry(entry) for entry in root.findall("atom:entry", _ARXIV_NS)]

        return ToolResult(
            output=results,
            metadata={
                "query": query.strip(),
                "source": "arxiv",
                "count": len(results),
                "top_k": limit,
            },
        )

    @staticmethod
    def _parse_entry(entry: ET.Element) -> dict[str, Any]:
        paper_id = _first_text(entry, "atom:id", _ARXIV_NS)
        pdf_url = ""
        html_url = paper_id
        for link in entry.findall("atom:link", _ARXIV_NS):
            href = link.attrib.get("href", "")
            rel = link.attrib.get("rel", "")
            title = link.attrib.get("title", "")
            link_type = link.attrib.get("type", "")
            if title == "pdf" or link_type == "application/pdf":
                pdf_url = href
            elif rel == "alternate":
                html_url = href

        categories = [
            category.attrib.get("term", "")
            for category in entry.findall("atom:category", _ARXIV_NS)
            if category.attrib.get("term")
        ]
        primary_category = ""
        primary = entry.find("arxiv:primary_category", _ARXIV_NS)
        if primary is not None:
            primary_category = primary.attrib.get("term", "")

        return {
            "id": paper_id.rsplit("/", 1)[-1] if paper_id else "",
            "title": _first_text(entry, "atom:title", _ARXIV_NS),
            "authors": [
                _first_text(author, "atom:name", _ARXIV_NS)
                for author in entry.findall("atom:author", _ARXIV_NS)
            ],
            "summary": _first_text(entry, "atom:summary", _ARXIV_NS),
            "published": _first_text(entry, "atom:published", _ARXIV_NS),
            "updated": _first_text(entry, "atom:updated", _ARXIV_NS),
            "categories": categories,
            "primary_category": primary_category,
            "url": html_url,
            "pdf_url": pdf_url,
        }

    async def close(self) -> None:
        """關閉內部 HTTP client。"""
        if self._owns_client:
            await self._client.aclose()


class SemanticScholarSearchTool(BaseTool):
    """Search papers through the Semantic Scholar Graph API."""

    _FIELDS = (
        "paperId,title,abstract,authors,year,venue,publicationDate,"
        "citationCount,influentialCitationCount,url,openAccessPdf,externalIds"
    )

    def __init__(
        self,
        timeout: float = 20.0,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._client = client or make_default_client(timeout=timeout)
        self._owns_client = client is None
        self._rate_limiter = TokenBucketRateLimiter(rate=2.0, burst=2)

    @property
    def name(self) -> str:
        return "semantic_scholar_search"

    @property
    def description(self) -> str:
        return (
            "Search academic papers using the Semantic Scholar Graph API. Returns abstracts, "
            "authors, citation counts, DOI/arXiv IDs, and open access PDF links when available."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Paper search keywords."},
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5,
                    "description": "Maximum number of papers to return.",
                },
                "year": {
                    "type": "string",
                    "description": "Optional year filter accepted by Semantic Scholar, e.g. 2020-2024.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        }

    async def execute(self, **kwargs: Any) -> ToolResult:
        """執行 Semantic Scholar 搜尋。"""
        query = str(kwargs.get("query", ""))
        top_k = int(kwargs.get("top_k", 5))
        year_arg = kwargs.get("year")
        year = str(year_arg) if year_arg is not None else None
        if error := _validate_query(query):
            return ToolResult(error=error)

        limit = _clamp_limit(top_k)
        params: dict[str, Any] = {
            "query": query.strip(),
            "limit": limit,
            "fields": self._FIELDS,
        }
        if year and year.strip():
            params["year"] = year.strip()

        headers = {"x-api-key": self._api_key} if self._api_key else None
        try:
            response = await http_request(
                self._client,
                "GET",
                "https://api.semanticscholar.org/graph/v1/paper/search",
                rate_limiter=self._rate_limiter,
                params=params,
                headers=headers,
                max_retries=2,
            )
        except ToolHttpError as exc:
            return error_to_tool_result(
                exc,
                extra_metadata={"query": query.strip(), "source": "semantic_scholar"},
                suggestion="Semantic Scholar API may be rate-limited. Try again in a few seconds.",
            )
        payload = _as_dict(response.json())
        papers = _as_list(payload.get("data"))

        results = [self._parse_paper(_as_dict(paper)) for paper in papers if isinstance(paper, dict)]
        return ToolResult(
            output=results,
            metadata={
                "query": query.strip(),
                "source": "semantic_scholar",
                "count": len(results),
                "top_k": limit,
                "total": payload.get("total"),
            },
        )

    @staticmethod
    def _parse_paper(paper: dict[str, Any]) -> dict[str, Any]:
        external_ids = _as_dict(paper.get("externalIds"))
        open_access_pdf = _as_dict(paper.get("openAccessPdf"))
        authors = [
            {"name": _as_str(author.get("name")), "author_id": _as_str(author.get("authorId"))}
            for author in (_as_dict(item) for item in _as_list(paper.get("authors")))
            if author
        ]
        return {
            "paper_id": _as_str(paper.get("paperId")),
            "title": _as_str(paper.get("title")),
            "authors": authors,
            "abstract": _as_str(paper.get("abstract")),
            "year": paper.get("year"),
            "venue": _as_str(paper.get("venue")),
            "publication_date": paper.get("publicationDate"),
            "citation_count": paper.get("citationCount"),
            "influential_citation_count": paper.get("influentialCitationCount"),
            "url": _as_str(paper.get("url")),
            "pdf_url": _as_str(open_access_pdf.get("url")),
            "doi": _as_str(external_ids.get("DOI")),
            "arxiv_id": _as_str(external_ids.get("ArXiv")),
            "external_ids": external_ids,
        }

    async def close(self) -> None:
        """關閉內部 HTTP client。"""
        if self._owns_client:
            await self._client.aclose()


class CrossrefSearchTool(BaseTool):
    """Search scholarly works through the Crossref REST API."""

    def __init__(
        self,
        timeout: float = 20.0,
        mailto: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._mailto = _optional_email(mailto)
        self._client = client or make_default_client(timeout=timeout)
        self._owns_client = client is None
        self._rate_limiter = TokenBucketRateLimiter(rate=3.0, burst=3)

    @property
    def name(self) -> str:
        return "crossref_search"

    @property
    def description(self) -> str:
        return (
            "Search scholarly metadata using the Crossref REST API. Use for DOI, journal, "
            "publisher, author, date, and reference metadata."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Bibliographic search keywords."},
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5,
                    "description": "Maximum number of works to return.",
                },
                "filter": {
                    "type": "string",
                    "description": "Optional Crossref filter string, e.g. from-pub-date:2020,type:journal-article.",
                },
                "mailto": {
                    "type": "string",
                    "description": "Optional email for Crossref polite pool routing.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        }

    async def execute(self, **kwargs: Any) -> ToolResult:
        """執行 Crossref 搜尋。"""
        query = str(kwargs.get("query", ""))
        top_k = int(kwargs.get("top_k", 5))
        filter_arg = kwargs.get("filter")
        mailto_arg = kwargs.get("mailto")
        filter = str(filter_arg) if filter_arg is not None else None
        mailto = str(mailto_arg) if mailto_arg is not None else None
        if error := _validate_query(query):
            return ToolResult(error=error)

        limit = _clamp_limit(top_k)
        params: dict[str, Any] = {
            "query": query.strip(),
            "rows": limit,
            "select": (
                "DOI,title,author,container-title,published-print,published-online,"
                "published,type,publisher,is-referenced-by-count,URL,abstract"
            ),
        }
        if filter and filter.strip():
            params["filter"] = filter.strip()
        email = _optional_email(mailto) or self._mailto
        if email:
            params["mailto"] = email

        response = await http_request(
            self._client, "GET", "https://api.crossref.org/works",
            params=params,
            rate_limiter=self._rate_limiter,
            max_retries=2,
        )
        payload = _as_dict(response.json())
        message = _as_dict(payload.get("message"))
        items = _as_list(message.get("items"))

        results = [self._parse_work(_as_dict(item)) for item in items if isinstance(item, dict)]
        return ToolResult(
            output=results,
            metadata={
                "query": query.strip(),
                "source": "crossref",
                "count": len(results),
                "top_k": limit,
                "total": message.get("total-results"),
            },
        )

    @staticmethod
    def _parse_work(item: dict[str, Any]) -> dict[str, Any]:
        titles = _as_list(item.get("title"))
        container_titles = _as_list(item.get("container-title"))
        authors = [
            _format_crossref_author(_as_dict(author))
            for author in _as_list(item.get("author"))
            if isinstance(author, dict)
        ]
        return {
            "doi": _as_str(item.get("DOI")),
            "title": str(titles[0]) if titles else "",
            "authors": authors,
            "container_title": str(container_titles[0]) if container_titles else "",
            "published": _crossref_date(item),
            "type": _as_str(item.get("type")),
            "publisher": _as_str(item.get("publisher")),
            "citation_count": item.get("is-referenced-by-count"),
            "url": _as_str(item.get("URL")),
            "abstract": _strip_crossref_abstract(_as_str(item.get("abstract"))),
        }

    async def close(self) -> None:
        """關閉內部 HTTP client。"""
        if self._owns_client:
            await self._client.aclose()


class PubMedSearchTool(BaseTool):
    """Search PubMed through the official NCBI E-utilities API."""

    def __init__(
        self,
        timeout: float = 20.0,
        tool_name: str = "mochi",
        email: str | None = None,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._tool_name = tool_name
        self._email = _optional_email(email)
        self._api_key = api_key
        self._client = client or make_default_client(timeout=timeout)
        self._owns_client = client is None
        self._rate_limiter = TokenBucketRateLimiter(
            rate=10.0 if self._api_key else 3.0,
            burst=10 if self._api_key else 3,
        )

    @property
    def name(self) -> str:
        return "pubmed_search"

    @property
    def description(self) -> str:
        return (
            "Search PubMed using NCBI E-utilities. Use for biomedical literature and return "
            "PMID, title, journal, authors, publication date, DOI, and PubMed URL."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "PubMed search query."},
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5,
                    "description": "Maximum number of articles to return.",
                },
                "include_abstract": {
                    "type": "boolean",
                    "default": True,
                    "description": (
                        "If true, fetch full abstracts via efetch (adds latency). "
                        "If false, return metadata only."
                    ),
                },
                "email": {
                    "type": "string",
                    "description": "Optional email passed to NCBI E-utilities.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        }

    async def execute(self, **kwargs: Any) -> ToolResult:
        """執行 PubMed 搜尋。"""
        query = str(kwargs.get("query", ""))
        top_k = int(kwargs.get("top_k", 5))
        email_arg = kwargs.get("email")
        email = str(email_arg) if email_arg is not None else None
        include_abstract = bool(kwargs.get("include_abstract", True))
        if error := _validate_query(query):
            return ToolResult(error=error)

        limit = _clamp_limit(top_k)
        params: dict[str, Any] = {
            "db": "pubmed",
            "term": query.strip(),
            "retmode": "json",
            "retmax": limit,
            "tool": self._tool_name,
        }
        active_email = _optional_email(email) or self._email
        if active_email:
            params["email"] = active_email
        if self._api_key:
            params["api_key"] = self._api_key

        try:
            search_response = await http_request(
                self._client, "GET",
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params=params,
                rate_limiter=self._rate_limiter,
                max_retries=2,
            )
        except ToolHttpError as exc:
            return error_to_tool_result(
                exc,
                extra_metadata={"query": query.strip(), "source": "pubmed"},
                suggestion="NCBI E-utilities may be temporarily unavailable.",
            )
        search_payload = _as_dict(search_response.json())
        search_result = _as_dict(search_payload.get("esearchresult"))
        id_list = [str(item) for item in _as_list(search_result.get("idlist"))]

        if not id_list:
            return ToolResult(
                output=[],
                metadata={"query": query.strip(), "source": "pubmed", "count": 0, "top_k": limit},
            )

        # Step 2: esummary for metadata
        summary_params: dict[str, Any] = {
            "db": "pubmed",
            "id": ",".join(id_list),
            "retmode": "json",
            "tool": self._tool_name,
        }
        if active_email:
            summary_params["email"] = active_email
        if self._api_key:
            summary_params["api_key"] = self._api_key

        try:
            summary_response = await http_request(
                self._client, "GET",
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                params=summary_params,
                rate_limiter=self._rate_limiter,
                max_retries=2,
            )
        except ToolHttpError as exc:
            return error_to_tool_result(
                exc,
                extra_metadata={"query": query.strip(), "source": "pubmed"},
            )
        summary_payload = _as_dict(summary_response.json())
        result_map = _as_dict(summary_payload.get("result"))
        order = [str(uid) for uid in _as_list(result_map.get("uids"))] or id_list

        # Step 3: efetch for abstracts (optional)
        abstract_map: dict[str, str] = {}
        if include_abstract:
            abstract_map = await self._fetch_abstracts(id_list, active_email)

        results: list[dict[str, Any]] = []
        for uid in order:
            summary = _as_dict(result_map.get(uid))
            if summary:
                parsed = self._parse_summary(uid, summary)
                if include_abstract:
                    parsed["abstract"] = abstract_map.get(uid, "")
                results.append(parsed)
        return ToolResult(
            output=results,
            metadata={
                "query": query.strip(),
                "source": "pubmed",
                "count": len(results),
                "top_k": limit,
                "total": search_result.get("count"),
                "include_abstract": include_abstract,
            },
        )

    async def _fetch_abstracts(
        self,
        id_list: list[str],
        email: str | None,
    ) -> dict[str, str]:
        """透過 efetch XML 取得 PubMed 文章摘要。"""
        efetch_params: dict[str, Any] = {
            "db": "pubmed",
            "id": ",".join(id_list),
            "retmode": "xml",
            "rettype": "abstract",
            "tool": self._tool_name,
        }
        if email:
            efetch_params["email"] = email
        if self._api_key:
            efetch_params["api_key"] = self._api_key

        try:
            response = await http_request(
                self._client, "GET",
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params=efetch_params,
                rate_limiter=self._rate_limiter,
                max_retries=1,
            )
        except ToolHttpError:
            return {}  # efetch 失敗不阻塞整體結果

        return _parse_efetch_abstracts(response.text)

    @staticmethod
    def _parse_summary(uid: str, item: dict[str, Any]) -> dict[str, Any]:
        article_ids = _as_list(item.get("articleids"))
        doi = ""
        for article_id_value in article_ids:
            article_id = _as_dict(article_id_value)
            if article_id.get("idtype") == "doi":
                doi = _as_str(article_id.get("value"))
                break

        authors = [
            _as_str(author.get("name"))
            for author in (_as_dict(item_value) for item_value in _as_list(item.get("authors")))
            if author.get("name")
        ]

        return {
            "pmid": uid,
            "title": _as_str(item.get("title")),
            "authors": authors,
            "journal": _as_str(item.get("fulljournalname")) or _as_str(item.get("source")),
            "pubdate": _as_str(item.get("pubdate")),
            "epubdate": _as_str(item.get("epubdate")),
            "doi": doi,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{uid}/",
        }

    async def close(self) -> None:
        """關閉內部 HTTP client。"""
        if self._owns_client:
            await self._client.aclose()


def _format_crossref_author(author: dict[str, Any]) -> str:
    given = _clean_text(str(author.get("given", "")))
    family = _clean_text(str(author.get("family", "")))
    name = " ".join(part for part in [given, family] if part)
    return name or _clean_text(str(author.get("name", "")))


def _crossref_date(item: dict[str, Any]) -> str:
    for key in ("published-print", "published-online", "published"):
        value = _as_dict(item.get(key))
        date_parts = _as_list(value.get("date-parts"))
        if not date_parts:
            continue
        first = _as_list(date_parts[0])
        if first:
            parts = [str(part) if index == 0 else f"{int(part):02d}" for index, part in enumerate(first)]
            return "-".join(parts)
    return ""


def _strip_crossref_abstract(value: str) -> str:
    if not value:
        return ""
    return _clean_text(re.sub(r"<[^>]+>", " ", value))


def _parse_efetch_abstracts(xml_text: str) -> dict[str, str]:
    """解析 efetch XML 回應，提取各 PMID 的 AbstractText。"""
    result: dict[str, str] = {}
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return result

    for article in root.iter("PubmedArticle"):
        pmid_elem = article.find(".//PMID")
        if pmid_elem is None or not pmid_elem.text:
            continue
        pmid = pmid_elem.text.strip()

        abstract_parts: list[str] = []
        for abstract_text in article.iter("AbstractText"):
            label = abstract_text.get("Label", "")
            text = _clean_text(abstract_text.text)
            # 包含結構化 abstract 標籤內的子元素文字
            tail_parts = []
            for child in abstract_text:
                if child.text:
                    tail_parts.append(_clean_text(child.text))
                if child.tail:
                    tail_parts.append(_clean_text(child.tail))
            if tail_parts:
                text = (text + " " + " ".join(tail_parts)).strip()
            if label and text:
                abstract_parts.append(f"{label}: {text}")
            elif text:
                abstract_parts.append(text)

        if abstract_parts:
            result[pmid] = " ".join(abstract_parts)

    return result
