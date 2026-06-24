"""文獻搜尋工具測試。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mochi.tools.literature_search import (
    ArxivSearchTool,
    CrossrefSearchTool,
    PubMedSearchTool,
    SemanticScholarSearchTool,
)


def _text_response(text: str, *, status_code: int = 200) -> MagicMock:
    response = MagicMock()
    response.text = text
    response.status_code = status_code
    response.headers = {}
    response.raise_for_status = MagicMock()
    return response


def _json_response(payload: dict, *, status_code: int = 200) -> MagicMock:
    response = MagicMock()
    response.json = MagicMock(return_value=payload)
    response.status_code = status_code
    response.headers = {}
    response.raise_for_status = MagicMock()
    return response


@pytest.mark.asyncio
async def test_arxiv_search_parses_atom_results() -> None:
    """arXiv Atom API 回應應解析為論文 metadata。"""
    atom = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom"
          xmlns:arxiv="http://arxiv.org/schemas/atom">
      <entry>
        <id>http://arxiv.org/abs/1706.03762v7</id>
        <updated>2023-08-02T00:00:00Z</updated>
        <published>2017-06-12T17:57:34Z</published>
        <title>Attention Is All You Need</title>
        <summary>We propose a new simple network architecture.</summary>
        <author><name>Ashish Vaswani</name></author>
        <author><name>Noam Shazeer</name></author>
        <arxiv:primary_category term="cs.CL" />
        <category term="cs.CL" />
        <category term="cs.LG" />
        <link href="http://arxiv.org/abs/1706.03762v7" rel="alternate" type="text/html" />
        <link title="pdf" href="http://arxiv.org/pdf/1706.03762v7" rel="related" type="application/pdf" />
      </entry>
    </feed>
    """
    tool = ArxivSearchTool()

    with patch.object(
        tool._client,
        "request",
        new_callable=AsyncMock,
        return_value=_text_response(atom),
    ) as get_mock:
        result = await tool.execute(query="ti:attention", top_k=3)

    assert result.error is None
    get_mock.assert_awaited_once()
    assert result.metadata["source"] == "arxiv"
    assert result.output == [
        {
            "id": "1706.03762v7",
            "title": "Attention Is All You Need",
            "authors": ["Ashish Vaswani", "Noam Shazeer"],
            "summary": "We propose a new simple network architecture.",
            "published": "2017-06-12T17:57:34Z",
            "updated": "2023-08-02T00:00:00Z",
            "categories": ["cs.CL", "cs.LG"],
            "primary_category": "cs.CL",
            "url": "http://arxiv.org/abs/1706.03762v7",
            "pdf_url": "http://arxiv.org/pdf/1706.03762v7",
        }
    ]

    await tool.close()


@pytest.mark.asyncio
async def test_semantic_scholar_search_parses_graph_results() -> None:
    """Semantic Scholar Graph API JSON 應解析為論文結果。"""
    payload = {
        "total": 1,
        "data": [
            {
                "paperId": "abc123",
                "title": "Retrieval-Augmented Generation",
                "abstract": "RAG combines retrieval with generation.",
                "authors": [{"name": "Patrick Lewis", "authorId": "1"}],
                "year": 2020,
                "venue": "NeurIPS",
                "publicationDate": "2020-12-01",
                "citationCount": 1234,
                "influentialCitationCount": 99,
                "url": "https://www.semanticscholar.org/paper/abc123",
                "openAccessPdf": {"url": "https://example.com/rag.pdf"},
                "externalIds": {"DOI": "10.5555/rag", "ArXiv": "2005.11401"},
            }
        ],
    }
    tool = SemanticScholarSearchTool(api_key="secret")

    with patch.object(
        tool._client,
        "request",
        new_callable=AsyncMock,
        return_value=_json_response(payload),
    ) as get_mock:
        result = await tool.execute(query="retrieval augmented generation", top_k=1, year="2020")

    assert result.error is None
    _, kwargs = get_mock.await_args
    assert kwargs["headers"] == {"x-api-key": "secret"}
    assert kwargs["params"]["fields"].startswith("paperId,title")
    assert result.output[0]["paper_id"] == "abc123"
    assert result.output[0]["doi"] == "10.5555/rag"
    assert result.output[0]["pdf_url"] == "https://example.com/rag.pdf"

    await tool.close()


@pytest.mark.asyncio
async def test_crossref_search_parses_work_metadata_and_mailto() -> None:
    """Crossref /works JSON 應解析 DOI、作者與日期。"""
    payload = {
        "message": {
            "total-results": 1,
            "items": [
                {
                    "DOI": "10.1038/nature12373",
                    "title": ["Direct observations of the cosmic web"],
                    "author": [{"given": "J.", "family": "Smith"}],
                    "container-title": ["Nature"],
                    "published-print": {"date-parts": [[2014, 1, 2]]},
                    "type": "journal-article",
                    "publisher": "Springer Science and Business Media LLC",
                    "is-referenced-by-count": 42,
                    "URL": "https://doi.org/10.1038/nature12373",
                    "abstract": "<jats:p>Observed structure.</jats:p>",
                }
            ],
        }
    }
    tool = CrossrefSearchTool(mailto="Mochi <mochi@example.com>")

    with patch.object(
        tool._client,
        "request",
        new_callable=AsyncMock,
        return_value=_json_response(payload),
    ) as get_mock:
        result = await tool.execute(query="cosmic web", top_k=1)

    assert result.error is None
    _, kwargs = get_mock.await_args
    assert kwargs["params"]["mailto"] == "mochi@example.com"
    assert result.output == [
        {
            "doi": "10.1038/nature12373",
            "title": "Direct observations of the cosmic web",
            "authors": ["J. Smith"],
            "container_title": "Nature",
            "published": "2014-01-02",
            "type": "journal-article",
            "publisher": "Springer Science and Business Media LLC",
            "citation_count": 42,
            "url": "https://doi.org/10.1038/nature12373",
            "abstract": "Observed structure.",
        }
    ]

    await tool.close()


@pytest.mark.asyncio
async def test_pubmed_search_uses_esearch_then_esummary() -> None:
    """PubMed 工具應先 ESearch 取 PMID，再 ESummary 取文章 metadata，再 EFetch 取 abstract。"""
    search_payload = {"esearchresult": {"count": "1", "idlist": ["31452104"]}}
    summary_payload = {
        "result": {
            "uids": ["31452104"],
            "31452104": {
                "title": "A biomedical article",
                "authors": [{"name": "Lee A"}, {"name": "Chen B"}],
                "fulljournalname": "Journal of Tests",
                "pubdate": "2019 Oct",
                "epubdate": "2019 Sep 1",
                "articleids": [{"idtype": "doi", "value": "10.1000/test"}],
            },
        }
    }
    efetch_xml = """<?xml version="1.0"?>
    <PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation>
          <PMID>31452104</PMID>
          <Article>
            <Abstract>
              <AbstractText>This is the abstract.</AbstractText>
            </Abstract>
          </Article>
        </MedlineCitation>
      </PubmedArticle>
    </PubmedArticleSet>
    """
    tool = PubMedSearchTool(email="Mochi <mochi@example.com>")

    with patch.object(
        tool._client,
        "request",
        new_callable=AsyncMock,
        side_effect=[
            _json_response(search_payload),
            _json_response(summary_payload),
            _text_response(efetch_xml),
        ],
    ) as request_mock:
        result = await tool.execute(query="cancer immunotherapy", top_k=1)

    assert result.error is None
    assert request_mock.await_count == 3
    # request(method, url, **kwargs) — url is 2nd positional arg
    first_call = request_mock.await_args_list[0]
    second_call = request_mock.await_args_list[1]
    third_call = request_mock.await_args_list[2]
    assert "esearch.fcgi" in str(first_call)
    assert "esummary.fcgi" in str(second_call)
    assert "efetch.fcgi" in str(third_call)
    assert result.output[0]["title"] == "A biomedical article"
    assert result.output[0]["abstract"] == "This is the abstract."
    assert result.metadata["include_abstract"] is True

    await tool.close()


@pytest.mark.asyncio
async def test_literature_search_tools_reject_empty_queries() -> None:
    """文獻搜尋工具應拒絕空查詢。"""
    tools = [
        ArxivSearchTool(),
        SemanticScholarSearchTool(),
        CrossrefSearchTool(),
        PubMedSearchTool(),
    ]

    try:
        for tool in tools:
            result = await tool.execute(query=" ")
            assert result.error == "`query` must not be empty."
    finally:
        for tool in tools:
            await tool.close()
