from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from mochi.tools.base import BaseTool, ToolExecutionContext, ToolResult
from mochi.tools.file_ops import FileReadTool, FileWriteTool
from mochi.tools.mcp_client import (
    McpCatalogRefreshError,
    McpCatalogTimeoutError,
    McpRuntimeManager,
    McpServerConfig,
)
from mochi.tools.registry import ToolRegistry
from mochi.tools.tool_search import ToolSearchTool


@dataclass
class _FakeTool(BaseTool):
    tool_name: str
    tool_description: str
    hint: str | None = None

    @property
    def name(self) -> str:
        return self.tool_name

    @property
    def description(self) -> str:
        return self.tool_description

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Workspace path."},
            },
        }

    @property
    def search_hint(self) -> str | None:
        return self.hint

    async def execute(self, **kwargs: Any) -> ToolResult:
        del kwargs
        return ToolResult(output="unused")


class _FakeMcpAdapter:
    def __init__(self) -> None:
        self.tools: list[Any] = [
            {
                "name": "search_docs",
                "description": "Search local docs.",
                "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
                "annotations": {"readOnlyHint": True},
            }
        ]
        self.resources: list[Any] = [{"uri": "memo://welcome", "name": "welcome"}]
        self.raise_timeout = False
        self.resource_error: Exception | None = None

    def list_tools(self) -> list[Any]:
        if self.raise_timeout:
            raise TimeoutError("timed out")
        return list(self.tools)

    def list_resources(self) -> list[Any]:
        if self.resource_error is not None:
            raise self.resource_error
        return list(self.resources)


@pytest.mark.asyncio
async def test_tool_search_returns_rank_score_fingerprint_and_bounded_top_k(tmp_path: Path) -> None:
    registry = ToolRegistry(discover_builtin=False)
    registry.register(ToolSearchTool(catalog_provider=registry.list_tools))
    registry.register(FileReadTool(workspace_dir=tmp_path))
    registry.register(FileWriteTool(workspace_dir=tmp_path, require_approval=False))

    view = registry.create_view(
        ["tool_search", "file_read"],
        tool_search_catalog_names=["tool_search", "file_read", "file_write"],
    )

    result = await view.execute("tool_search", {"query": "file_write", "top_k": 99})

    assert result.error is None
    assert result.metadata["top_k"] == 10
    assert result.metadata["catalog_status"] == "ready"
    match = next(item for item in result.output if item["name"] == "file_write")
    assert match["rank"] == 1
    assert match["score"] > 0.0
    assert isinstance(match["catalog_fingerprint"], str) and match["catalog_fingerprint"]
    assert match["callable_this_turn"] is False
    assert match["activation_required"] is True
    assert match["activation_authorizes_tool_call"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "expected_status"),
    [
        (lambda: [], "ready"),
        (lambda: (_ for _ in ()).throw(TimeoutError("too slow")), "timeout"),
        (lambda: [object()], "malformed_catalog"),
        (lambda: (_ for _ in ()).throw(McpCatalogRefreshError("refresh failed")), "refresh_failed"),
    ],
)
async def test_tool_search_fail_closed_states(
    provider: Any,
    expected_status: str,
) -> None:
    tool = ToolSearchTool(catalog_provider=provider)

    result = await tool.execute(query="save file")

    assert result.error is None
    assert result.output == []
    assert result.metadata["catalog_status"] == expected_status


@pytest.mark.asyncio
async def test_tool_search_discovery_hook_receives_bounded_match_metadata() -> None:
    seen: list[dict[str, Any]] = []
    tool = ToolSearchTool(
        catalog_provider=lambda: [
            _FakeTool("file_write", "Write text into a workspace file.", hint="寫入檔案並保存內容。"),
            _FakeTool("web_search", "Search the web."),
        ],
        catalog_generation_provider=lambda: 3,
        discovery_hook=lambda payload: seen.append(payload),
    )
    context = ToolExecutionContext(session_id="session-1", state={"turn_id": "turn-7"})

    result = await tool.execute(query="寫入 檔案", context=context)

    assert result.error is None
    assert len(seen) == 1
    assert seen[0]["session_id"] == "session-1"
    assert seen[0]["turn_id"] == "turn-7"
    assert seen[0]["catalog_generation"] == 3
    assert seen[0]["matches"][0]["tool_name"] == "file_write"
    assert seen[0]["matches"][0]["score"] > 0.0
    assert isinstance(seen[0]["source_query_hash"], str) and seen[0]["source_query_hash"]


@pytest.mark.asyncio
async def test_mcp_runtime_refresh_tracks_generation_fingerprint_and_failures() -> None:
    adapter = _FakeMcpAdapter()
    runtime = McpRuntimeManager(
        servers={"demo": McpServerConfig(name="demo", transport="in_memory")},
        adapters={"demo": adapter},
    )

    await runtime.refresh_server("demo")
    first_state = runtime.get_catalog_state("demo")
    assert first_state.generation == 1
    assert first_state.status == "ready"
    assert first_state.fingerprint is not None

    adapter.tools.append(
        {
            "name": "read_docs",
            "description": "Read docs.",
            "input_schema": {"type": "object", "properties": {"uri": {"type": "string"}}},
            "annotations": {"readOnlyHint": True},
        }
    )
    await runtime.refresh_server("demo")
    second_state = runtime.get_catalog_state("demo")
    assert second_state.generation == 2
    assert second_state.fingerprint != first_state.fingerprint

    adapter.tools = [object()]
    with pytest.raises(McpCatalogRefreshError, match="malformed data"):
        await runtime.refresh_server("demo")
    malformed_state = runtime.get_catalog_state("demo")
    assert malformed_state.generation == 2
    assert malformed_state.status == "malformed_catalog"
    assert malformed_state.fingerprint == second_state.fingerprint

    adapter.tools = [
        {
            "name": "search_docs",
            "description": "Search local docs.",
            "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
            "annotations": {"readOnlyHint": True},
        }
    ]
    adapter.resource_error = RuntimeError("resources unavailable")
    with pytest.raises(McpCatalogRefreshError, match="resources unavailable"):
        await runtime.refresh_server("demo")
    refresh_failed_state = runtime.get_catalog_state("demo")
    assert refresh_failed_state.generation == 2
    assert refresh_failed_state.status == "refresh_failed"

    adapter.resource_error = None
    adapter.raise_timeout = True
    with pytest.raises(McpCatalogTimeoutError, match="timed out"):
        await runtime.refresh_server("demo")
    timeout_state = runtime.get_catalog_state("demo")
    assert timeout_state.generation == 2
    assert timeout_state.status == "timeout"
