"""Evidence collection helpers for multi-agent runs."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from mochi.tools.base import ToolResult

ToolExecutor = Callable[[str, dict[str, Any]], Awaitable[ToolResult]]


async def collect_evidence_packets(
    *,
    queries: list[str],
    execute_tool: ToolExecutor | None = None,
    search_tool: Any,
    fetch_tool: Any | None = None,
    memory_search_tool: Any | None = None,
    mcp_list_resources_tool: Any | None = None,
    mcp_read_resource_tool: Any | None = None,
    rag_provider: str = "memory",
    rag_mcp_servers: list[str] | None = None,
    mode: str = "hybrid",
    max_results_per_query: int = 3,
    max_fetch_per_query: int = 2,
    max_content_chars: int = 2000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect normalized evidence packets from search and optional page fetches."""
    packets: list[dict[str, Any]] = []
    query_summaries: list[dict[str, Any]] = []
    provider_counts: dict[str, int] = {}
    retrieval_path_counts: dict[str, int] = {}

    normalized_queries = [item.strip() for item in queries if isinstance(item, str) and item.strip()]
    normalized_mode = (mode or "hybrid").strip().lower() or "hybrid"
    normalized_rag_provider = (rag_provider or "memory").strip().lower() or "memory"
    normalized_rag_mcp_servers = [
        item.strip()
        for item in (rag_mcp_servers or [])
        if isinstance(item, str) and item.strip()
    ]
    for query_index, query in enumerate(normalized_queries, start=1):
        query_summary: dict[str, Any] = {
            "query": query,
            "packet_count": 0,
        }

        if normalized_mode in {"rag", "hybrid"}:
            rag_packets = await _collect_rag_packets(
                query=query,
                query_index=query_index,
                execute_tool=execute_tool,
                memory_search_tool=memory_search_tool,
                mcp_list_resources_tool=mcp_list_resources_tool,
                mcp_read_resource_tool=mcp_read_resource_tool,
                rag_provider=normalized_rag_provider,
                rag_mcp_servers=normalized_rag_mcp_servers,
                max_results_per_query=max_results_per_query,
                max_content_chars=max_content_chars,
            )
            for packet in rag_packets:
                packets.append(packet)
                provider = str(packet.get("provider") or "memory_search")
                provider_counts[provider] = provider_counts.get(provider, 0) + 1
                retrieval_path_counts["rag"] = retrieval_path_counts.get("rag", 0) + 1
                query_summary["packet_count"] = int(query_summary["packet_count"]) + 1

        if normalized_mode in {"web", "hybrid"}:
            web_packets, provider, error = await _collect_web_packets(
                query=query,
                query_index=query_index,
                execute_tool=execute_tool,
                search_tool=search_tool,
                fetch_tool=fetch_tool,
                max_results_per_query=max_results_per_query,
                max_fetch_per_query=max_fetch_per_query,
                max_content_chars=max_content_chars,
            )
            if provider is not None:
                query_summary["provider"] = provider
            if error and int(query_summary["packet_count"]) == 0:
                query_summary["error"] = error
            for packet in web_packets:
                packets.append(packet)
                provider_name = str(packet.get("provider") or "unknown")
                provider_counts[provider_name] = provider_counts.get(provider_name, 0) + 1
                retrieval_path_counts["web"] = retrieval_path_counts.get("web", 0) + 1
                query_summary["packet_count"] = int(query_summary["packet_count"]) + 1

        query_summaries.append(query_summary)

    return packets, {
        "mode": mode,
        "rag_provider": normalized_rag_provider,
        "query_count": len(normalized_queries),
        "collected_packet_count": len(packets),
        "provider_counts": provider_counts,
        "retrieval_path_counts": retrieval_path_counts,
        "queries": query_summaries,
    }


async def _collect_rag_packets(
    *,
    query: str,
    query_index: int,
    execute_tool: ToolExecutor | None,
    memory_search_tool: Any | None,
    mcp_list_resources_tool: Any | None,
    mcp_read_resource_tool: Any | None,
    rag_provider: str,
    rag_mcp_servers: list[str],
    max_results_per_query: int,
    max_content_chars: int,
) -> list[dict[str, Any]]:
    provider = rag_provider if rag_provider in {"memory", "mcp_resource", "auto"} else "memory"
    if provider == "memory":
        return await _collect_memory_packets(
            query=query,
            query_index=query_index,
            execute_tool=execute_tool,
            memory_search_tool=memory_search_tool,
            max_results_per_query=max_results_per_query,
            max_content_chars=max_content_chars,
        )
    if provider == "mcp_resource":
        return await _collect_mcp_resource_packets(
            query=query,
            query_index=query_index,
            execute_tool=execute_tool,
            mcp_list_resources_tool=mcp_list_resources_tool,
            mcp_read_resource_tool=mcp_read_resource_tool,
            rag_mcp_servers=rag_mcp_servers,
            max_results_per_query=max_results_per_query,
            max_content_chars=max_content_chars,
        )

    memory_packets = await _collect_memory_packets(
        query=query,
        query_index=query_index,
        execute_tool=execute_tool,
        memory_search_tool=memory_search_tool,
        max_results_per_query=max_results_per_query,
        max_content_chars=max_content_chars,
    )
    if memory_packets:
        return memory_packets
    return await _collect_mcp_resource_packets(
        query=query,
        query_index=query_index,
        execute_tool=execute_tool,
        mcp_list_resources_tool=mcp_list_resources_tool,
        mcp_read_resource_tool=mcp_read_resource_tool,
        rag_mcp_servers=rag_mcp_servers,
        max_results_per_query=max_results_per_query,
        max_content_chars=max_content_chars,
    )


async def _collect_memory_packets(
    *,
    query: str,
    query_index: int,
    execute_tool: ToolExecutor | None,
    memory_search_tool: Any | None,
    max_results_per_query: int,
    max_content_chars: int,
) -> list[dict[str, Any]]:
    if execute_tool is None and memory_search_tool is None:
        return []
    result = await _execute_tool(
        "memory_search",
        {"query": query, "top_k": max_results_per_query},
        execute_tool=execute_tool,
        legacy_tool=memory_search_tool,
    )
    if result.error is not None or not isinstance(result.output, list):
        return []

    packets: list[dict[str, Any]] = []
    for result_index, item in enumerate(result.output[:max_results_per_query], start=1):
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        packets.append(
            {
                "evidence_id": str(item.get("id") or f"memory-{query_index}-{result_index}"),
                "title": str(item.get("category") or "Memory Evidence"),
                "content": _truncate_text(content, max_content_chars),
                "url": None,
                "source_type": "memory_search",
                "provider": "memory_search",
                "query": query,
                "rank": result_index,
                "metadata": item.get("metadata") if isinstance(item.get("metadata"), dict) else None,
            }
        )
    return packets


async def _collect_mcp_resource_packets(
    *,
    query: str,
    query_index: int,
    execute_tool: ToolExecutor | None,
    mcp_list_resources_tool: Any | None,
    mcp_read_resource_tool: Any | None,
    rag_mcp_servers: list[str],
    max_results_per_query: int,
    max_content_chars: int,
) -> list[dict[str, Any]]:
    if execute_tool is None and (mcp_list_resources_tool is None or mcp_read_resource_tool is None):
        return []
    if not rag_mcp_servers:
        return []

    packets: list[dict[str, Any]] = []
    for server in rag_mcp_servers:
        list_result = await _execute_tool(
            "mcp_list_resources",
            {"server": server},
            execute_tool=execute_tool,
            legacy_tool=mcp_list_resources_tool,
        )
        if list_result.error is not None or not isinstance(list_result.output, list):
            continue
        ranked_resources = _rank_mcp_resources(query=query, resources=list_result.output)
        for result_index, resource in enumerate(ranked_resources[:max_results_per_query], start=1):
            if not isinstance(resource, dict):
                continue
            uri = str(resource.get("uri") or "").strip()
            if not uri:
                continue
            read_result = await _execute_tool(
                "mcp_read_resource",
                {"server": server, "uri": uri},
                execute_tool=execute_tool,
                legacy_tool=mcp_read_resource_tool,
            )
            content = _coerce_resource_text(read_result)
            if not content:
                continue
            packets.append(
                {
                    "evidence_id": f"mcp-{server}-{query_index}-{result_index}",
                    "title": str(resource.get("name") or resource.get("title") or uri),
                    "content": _truncate_text(content, max_content_chars),
                    "url": uri,
                    "source_type": "mcp_resource",
                    "provider": f"mcp_resource:{server}",
                    "query": query,
                    "rank": result_index,
                    "server": server,
                }
            )
            if len(packets) >= max_results_per_query:
                return packets
    return packets


async def _collect_web_packets(
    *,
    query: str,
    query_index: int,
    execute_tool: ToolExecutor | None,
    search_tool: Any,
    fetch_tool: Any | None,
    max_results_per_query: int,
    max_fetch_per_query: int,
    max_content_chars: int,
) -> tuple[list[dict[str, Any]], str | None, str | None]:
    search_result = await _execute_tool(
        "web_search",
        {"query": query, "top_k": max_results_per_query},
        execute_tool=execute_tool,
        legacy_tool=search_tool,
    )
    provider, results = _extract_search_payload(search_result)
    if search_result.error is not None:
        return [], provider, search_result.error

    packets: list[dict[str, Any]] = []
    for result_index, item in enumerate(results[:max_results_per_query], start=1):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip() or f"Evidence {query_index}-{result_index}"
        url = str(item.get("url") or "").strip()
        snippet = str(item.get("snippet") or "").strip()
        inline_content = str(item.get("content") or "").strip()
        content = ""
        source_type = "web_search_snippet"
        extractor = None

        if (
            (execute_tool is not None or fetch_tool is not None)
            and url
            and result_index <= max_fetch_per_query
        ):
            fetch_result = await _execute_tool(
                "web_fetch",
                {"url": url, "output_format": "text"},
                execute_tool=execute_tool,
                legacy_tool=fetch_tool,
            )
            fetched_content = _coerce_text(fetch_result)
            if fetched_content:
                content = _truncate_text(fetched_content, max_content_chars)
                source_type = "web_fetch"
                extractor = fetch_result.metadata.get("extractor")
            elif snippet:
                content = _truncate_text(snippet, max_content_chars)
            elif inline_content:
                content = _truncate_text(inline_content, max_content_chars)
        elif inline_content:
            content = _truncate_text(inline_content, max_content_chars)
        elif snippet:
            content = _truncate_text(snippet, max_content_chars)

        if not content:
            continue

        packets.append(
            {
                "evidence_id": f"evidence-{query_index}-{result_index}",
                "title": title,
                "content": content,
                "url": url or None,
                "source_type": source_type,
                "provider": provider,
                "query": query,
                "rank": result_index,
                "extractor": extractor,
            }
        )
    return packets, provider, None


async def _execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    execute_tool: ToolExecutor | None,
    legacy_tool: Any | None,
) -> ToolResult:
    if execute_tool is not None:
        return await execute_tool(tool_name, arguments)
    if legacy_tool is None:
        return ToolResult(error=f"{tool_name} tool is not available.")
    return await legacy_tool.execute(**arguments)


def _extract_search_payload(result: ToolResult) -> tuple[str, list[dict[str, Any]]]:
    output = result.output
    if isinstance(output, dict):
        provider = str(output.get("provider") or "unknown").strip() or "unknown"
        results = output.get("results")
        if isinstance(results, list):
            return provider, [item for item in results if isinstance(item, dict)]
        return provider, []
    if isinstance(output, list):
        return "unknown", [item for item in output if isinstance(item, dict)]
    return "unknown", []


def _coerce_text(result: ToolResult) -> str:
    if result.error is not None:
        return ""
    if isinstance(result.output, str):
        return result.output.strip()
    return ""


def _coerce_resource_text(result: ToolResult) -> str:
    if result.error is not None:
        return ""
    if isinstance(result.output, str):
        return result.output.strip()
    if isinstance(result.output, dict):
        for key in ("text", "content", "markdown"):
            value = result.output.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _rank_mcp_resources(*, query: str, resources: list[Any]) -> list[dict[str, Any]]:
    query_terms = [item for item in query.lower().split() if item]
    ranked: list[tuple[int, dict[str, Any]]] = []
    for item in resources:
        if not isinstance(item, dict):
            continue
        haystack = " ".join(
            [
                str(item.get("name") or ""),
                str(item.get("title") or ""),
                str(item.get("uri") or ""),
            ]
        ).lower()
        score = sum(1 for term in query_terms if term in haystack)
        ranked.append((score, dict(item)))
    ranked.sort(key=lambda entry: entry[0], reverse=True)
    return [item for _, item in ranked]


def _truncate_text(value: str, max_chars: int) -> str:
    text = " ".join(value.split()).strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."
