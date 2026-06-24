from __future__ import annotations

import pytest

from mochi.agents.multi_agent.evidence_collector import collect_evidence_packets
from mochi.tools.base import ToolResult


class _SearchToolStub:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def execute(self, **kwargs: object) -> ToolResult:
        self.calls.append(dict(kwargs))
        return ToolResult(
            output={
                "query": str(kwargs["query"]),
                "provider": "stub-search",
                "results": [
                    {
                        "title": "Primary source",
                        "url": "https://example.com/source-1",
                        "snippet": "Primary source snippet.",
                    },
                    {
                        "title": "Secondary source",
                        "url": "https://example.com/source-2",
                        "snippet": "Secondary source snippet.",
                    },
                ],
            }
        )


class _FetchToolStub:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def execute(self, **kwargs: object) -> ToolResult:
        self.calls.append(dict(kwargs))
        url = str(kwargs["url"])
        if url.endswith("source-2"):
            return ToolResult(error="fetch failed")
        return ToolResult(
            output=f"Fetched content for {url}",
            metadata={"url": url, "extractor": "stub-fetch"},
        )


class _MemorySearchToolStub:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def execute(self, **kwargs: object) -> ToolResult:
        self.calls.append(dict(kwargs))
        return ToolResult(
            output=[
                {
                    "id": "mem-1",
                    "content": "Local memory says the approved deployment note matches the student answer.",
                    "category": "deployment",
                    "metadata": {"source": "memory"},
                }
            ]
        )


class _McpListResourcesToolStub:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def execute(self, **kwargs: object) -> ToolResult:
        self.calls.append(dict(kwargs))
        return ToolResult(
            output=[
                {"uri": "memo://deployment-note", "name": "Deployment note"},
                {"uri": "memo://changelog", "name": "Changelog"},
            ]
        )


class _McpReadResourceToolStub:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def execute(self, **kwargs: object) -> ToolResult:
        self.calls.append(dict(kwargs))
        uri = str(kwargs["uri"])
        if uri.endswith("deployment-note"):
            return ToolResult(output={"uri": uri, "text": "MCP resource confirms the student answer is approved."})
        return ToolResult(output={"uri": uri, "text": "Generic changelog entry."})


class _LegacyToolMustNotRun:
    async def execute(self, **kwargs: object) -> ToolResult:  # noqa: ARG002
        raise AssertionError("legacy raw tool path should not be used when execute_tool is provided")


@pytest.mark.asyncio
async def test_collect_evidence_packets_from_search_and_fetch() -> None:
    search_tool = _SearchToolStub()
    fetch_tool = _FetchToolStub()

    packets, summary = await collect_evidence_packets(
        queries=["deployment approval note"],
        search_tool=search_tool,
        fetch_tool=fetch_tool,
        max_results_per_query=2,
        max_fetch_per_query=2,
        max_content_chars=120,
    )

    assert len(packets) == 2
    assert packets[0]["title"] == "Primary source"
    assert packets[0]["url"] == "https://example.com/source-1"
    assert packets[0]["source_type"] == "web_fetch"
    assert "Fetched content" in packets[0]["content"]
    assert packets[1]["source_type"] == "web_search_snippet"
    assert packets[1]["content"] == "Secondary source snippet."
    assert summary["query_count"] == 1
    assert summary["collected_packet_count"] == 2
    assert summary["provider_counts"] == {"stub-search": 2}
    assert summary["retrieval_path_counts"] == {"web": 2}
    assert len(summary["queries"]) == 1
    assert len(fetch_tool.calls) == 2


@pytest.mark.asyncio
async def test_collect_evidence_packets_prefers_registry_executor_over_legacy_tools() -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    async def execute_tool(name: str, args: dict[str, object]) -> ToolResult:
        calls.append((name, dict(args)))
        if name == "web_search":
            return ToolResult(
                output={
                    "provider": "registry-search",
                    "results": [
                        {
                            "title": "Primary source",
                            "url": "https://example.com/source-1",
                            "snippet": "Primary source snippet.",
                        },
                        {
                            "title": "Secondary source",
                            "url": "https://example.com/source-2",
                            "snippet": "Secondary source snippet.",
                        },
                    ],
                }
            )
        if name == "web_fetch":
            return ToolResult(
                output=f"Fetched content for {args['url']}",
                metadata={"extractor": "registry-fetch"},
            )
        raise AssertionError(f"unexpected tool call: {name}")

    packets, summary = await collect_evidence_packets(
        queries=["deployment approval note"],
        execute_tool=execute_tool,
        search_tool=_LegacyToolMustNotRun(),
        fetch_tool=_LegacyToolMustNotRun(),
        mode="web",
        max_results_per_query=2,
        max_fetch_per_query=1,
        max_content_chars=120,
    )

    assert len(packets) == 2
    assert packets[0]["source_type"] == "web_fetch"
    assert packets[0]["provider"] == "registry-search"
    assert packets[0]["extractor"] == "registry-fetch"
    assert packets[1]["source_type"] == "web_search_snippet"
    assert summary["provider_counts"] == {"registry-search": 2}
    assert calls == [
        ("web_search", {"query": "deployment approval note", "top_k": 2}),
        ("web_fetch", {"url": "https://example.com/source-1", "output_format": "text"}),
    ]


@pytest.mark.asyncio
async def test_collect_evidence_packets_rag_mode_uses_memory_search_only() -> None:
    search_tool = _SearchToolStub()
    fetch_tool = _FetchToolStub()
    memory_search_tool = _MemorySearchToolStub()

    packets, summary = await collect_evidence_packets(
        queries=["deployment approval note"],
        search_tool=search_tool,
        fetch_tool=fetch_tool,
        memory_search_tool=memory_search_tool,
        mode="rag",
        max_results_per_query=2,
        max_fetch_per_query=2,
        max_content_chars=120,
    )

    assert len(packets) == 1
    assert packets[0]["source_type"] == "memory_search"
    assert packets[0]["provider"] == "memory_search"
    assert summary["mode"] == "rag"
    assert summary["retrieval_path_counts"] == {"rag": 1}
    assert len(memory_search_tool.calls) == 1
    assert search_tool.calls == []
    assert fetch_tool.calls == []


@pytest.mark.asyncio
async def test_collect_evidence_packets_hybrid_combines_rag_and_web_results() -> None:
    search_tool = _SearchToolStub()
    fetch_tool = _FetchToolStub()
    memory_search_tool = _MemorySearchToolStub()

    packets, summary = await collect_evidence_packets(
        queries=["deployment approval note"],
        search_tool=search_tool,
        fetch_tool=fetch_tool,
        memory_search_tool=memory_search_tool,
        mode="hybrid",
        max_results_per_query=1,
        max_fetch_per_query=1,
        max_content_chars=120,
    )

    assert len(packets) == 2
    assert packets[0]["source_type"] == "memory_search"
    assert packets[1]["source_type"] == "web_fetch"
    assert summary["mode"] == "hybrid"
    assert summary["retrieval_path_counts"] == {"rag": 1, "web": 1}
    assert len(memory_search_tool.calls) == 1
    assert len(search_tool.calls) == 1


@pytest.mark.asyncio
async def test_collect_evidence_packets_rag_mcp_resource_provider_reads_resources() -> None:
    search_tool = _SearchToolStub()
    fetch_tool = _FetchToolStub()
    memory_search_tool = _MemorySearchToolStub()
    mcp_list_resources_tool = _McpListResourcesToolStub()
    mcp_read_resource_tool = _McpReadResourceToolStub()

    packets, summary = await collect_evidence_packets(
        queries=["deployment approved"],
        search_tool=search_tool,
        fetch_tool=fetch_tool,
        memory_search_tool=memory_search_tool,
        mcp_list_resources_tool=mcp_list_resources_tool,
        mcp_read_resource_tool=mcp_read_resource_tool,
        mode="rag",
        rag_provider="mcp_resource",
        rag_mcp_servers=["docs"],
        max_results_per_query=1,
        max_fetch_per_query=1,
        max_content_chars=120,
    )

    assert len(packets) == 1
    assert packets[0]["source_type"] == "mcp_resource"
    assert packets[0]["provider"] == "mcp_resource:docs"
    assert summary["rag_provider"] == "mcp_resource"
    assert summary["retrieval_path_counts"] == {"rag": 1}
    assert len(mcp_list_resources_tool.calls) == 1
    assert len(mcp_read_resource_tool.calls) == 1
    assert memory_search_tool.calls == []
    assert search_tool.calls == []
    assert fetch_tool.calls == []
