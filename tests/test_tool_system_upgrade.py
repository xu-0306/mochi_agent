"""Tool system upgrade tests for phases 1-4."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from mochi.config.schema import MochiConfig
from mochi.memory.store import MemoryStore
from mochi.tools.base import BaseTool, ToolExecutionContext, ToolResult
from mochi.tools.file_ops import ApplyPatchTool, FileEditTool, FileReadTool, FileWriteTool
from mochi.tools.glob_search import GlobSearchTool
from mochi.tools.grep_search import GrepSearchTool
from mochi.tools.csv_read import CsvReadTool
from mochi.tools.execute_code_v2 import ExecuteCodeV2Tool
from mochi.tools.mcp_client import (
    MCPCallTool,
    McpDynamicTool,
    McpListResourcesTool,
    McpReadResourceTool,
    McpRuntimeManager,
    McpServerConfig,
    McpToolDefinition,
)
from mochi.tools.memory_delete import MemoryDeleteTool
from mochi.tools.memory_export import MemoryExportTool
from mochi.tools.memory_update import MemoryUpdateTool
from mochi.tools.notebook_read import NotebookReadTool
from mochi.tools.pdf_read import PdfReadTool
from mochi.tools.registry import ToolRegistry
from mochi.tools.registry_factory import ToolRegistryFactory
from mochi.tools.web_search import WebSearchTool
from mochi.tools.tool_search import ToolSearchTool
from mochi.tools.web_crawl import WebCrawlTool


class _EchoTool(BaseTool):
    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echoes the provided text."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        *,
        text: str,
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        return ToolResult(
            output={
                "text": text,
                "workspace": context.workspace_dir if context is not None else None,
                "session": context.session_id if context is not None else None,
            }
        )


class _GuardedTool(_EchoTool):
    @property
    def name(self) -> str:
        return "guarded"

    def validate_input(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> ToolResult | None:
        del context
        if not str(arguments.get("text", "")).strip():
            return ToolResult(error="text is required")
        return None

    def check_permissions(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> ToolResult | None:
        del arguments
        if context is not None and not context.permission_policy.get("allow_guarded", False):
            return ToolResult(error="guarded tool denied")
        return None


class _FakeMcpAdapter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    async def list_tools(self) -> list[McpToolDefinition]:
        return [
            McpToolDefinition(
                name="search_docs",
                description="Search docs from MCP.",
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
                annotations={
                    "readOnlyHint": True,
                    "openWorldHint": True,
                    "searchHint": "Use before fetching a specific URL.",
                },
            )
        ]

    async def list_resources(self) -> list[dict[str, Any]]:
        return [{"uri": "memo://welcome", "name": "welcome"}]

    async def read_resource(self, uri: str) -> dict[str, Any]:
        return {"uri": uri, "text": "hello from resource"}

    async def call_tool(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("demo", tool, arguments))
        return {"tool": tool, "arguments": arguments, "ok": True}


@pytest.mark.asyncio
async def test_base_tool_defaults_and_registry_factory_context(tmp_path: Path) -> None:
    registry = ToolRegistry(discover_builtin=False)
    registry.register_factory("echo", lambda: _EchoTool())
    context = ToolExecutionContext(
        workspace_dir=str(tmp_path),
        session_id="session-1",
    )

    tool = registry.get("echo")
    assert tool is not None
    assert tool.is_read_only is False
    assert tool.is_destructive is False
    assert tool.is_concurrency_safe is False
    assert tool.is_open_world is False
    assert tool.search_hint is None

    result = await registry.execute("echo", {"text": "hi"}, context=context)

    assert result.error is None
    assert result.output == {
        "text": "hi",
        "workspace": str(tmp_path),
        "session": "session-1",
    }


@pytest.mark.asyncio
async def test_registry_runs_validation_and_permission_checks(tmp_path: Path) -> None:
    registry = ToolRegistry(discover_builtin=False)
    registry.register(_GuardedTool())

    denied = await registry.execute(
        "guarded",
        {"text": "ok"},
        context=ToolExecutionContext(
            workspace_dir=str(tmp_path),
            permission_policy={"allow_guarded": False},
        ),
    )
    assert denied.error == "guarded tool denied"

    invalid = await registry.execute("guarded", {"text": "   "})
    assert invalid.error == "text is required"


@pytest.mark.asyncio
async def test_file_edit_requires_read_and_returns_diff_metadata(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")
    context = ToolExecutionContext(workspace_dir=str(tmp_path))
    reader = FileReadTool(workspace_dir=tmp_path)
    editor = FileEditTool(workspace_dir=tmp_path, require_approval=False)

    blocked = await editor.execute(
        path="notes.txt",
        old_string="beta",
        new_string="gamma",
        context=context,
    )
    assert blocked.error is not None
    assert "read" in blocked.error.lower()

    read_result = await reader.execute(path="notes.txt", context=context)
    assert read_result.error is None

    edited = await editor.execute(
        path="notes.txt",
        old_string="beta",
        new_string="gamma",
        context=context,
    )

    assert edited.error is None
    assert target.read_text(encoding="utf-8") == "alpha\ngamma\n"
    assert edited.metadata["undo_available"] is True
    assert edited.metadata["diff_available"] is True
    assert "-beta" in str(edited.metadata["diff"])
    assert "+gamma" in str(edited.metadata["diff"])


@pytest.mark.asyncio
async def test_file_write_rejects_stale_read_state(tmp_path: Path) -> None:
    target = tmp_path / "stale.txt"
    target.write_text("before", encoding="utf-8")
    context = ToolExecutionContext(workspace_dir=str(tmp_path))
    reader = FileReadTool(workspace_dir=tmp_path)
    writer = FileWriteTool(workspace_dir=tmp_path, require_approval=False)

    read_result = await reader.execute(path="stale.txt", context=context)
    assert read_result.error is None

    target.write_text("outside change", encoding="utf-8")

    stale = await writer.execute(
        path="stale.txt",
        content="new value",
        context=context,
    )

    assert stale.error is not None
    assert "changed" in stale.error.lower() or "stale" in stale.error.lower()


@pytest.mark.asyncio
async def test_file_write_new_file_uses_delete_undo_action(tmp_path: Path) -> None:
    context = ToolExecutionContext(workspace_dir=str(tmp_path))
    writer = FileWriteTool(workspace_dir=tmp_path, require_approval=False)

    result = await writer.execute(
        path="created.txt",
        content="hello",
        context=context,
    )

    assert result.error is None
    assert result.metadata["undo_available"] is True
    assert result.metadata["undo_action"] == "delete"


@pytest.mark.asyncio
async def test_file_write_approval_returns_normalized_file_change_metadata(tmp_path: Path) -> None:
    writer = FileWriteTool(workspace_dir=tmp_path, require_approval=True)

    result = await writer.execute(
        path="created.txt",
        content="hello",
        context=ToolExecutionContext(workspace_dir=str(tmp_path)),
    )

    assert result.error is not None
    assert result.metadata["requires_approval"] is True
    assert result.metadata["change_count"] == 1
    assert result.metadata["file_changes"][0]["change_type"] == "add"
    assert result.metadata["undo_action"] == "delete"


@pytest.mark.asyncio
async def test_apply_patch_requires_read_and_returns_shared_metadata(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")
    context = ToolExecutionContext(workspace_dir=str(tmp_path))
    tool = ApplyPatchTool(workspace_dir=tmp_path, require_approval=False)
    patch_text = "\n".join(
        [
            "*** Begin Patch",
            "*** Update File: notes.txt",
            "@@",
            " alpha",
            "-beta",
            "+gamma",
            "*** End Patch",
        ]
    )

    blocked = await tool.execute(patch=patch_text, context=context)
    assert blocked.error is not None
    assert "read" in blocked.error.lower()

    reader = FileReadTool(workspace_dir=tmp_path)
    read_result = await reader.execute(path="notes.txt", context=context)
    assert read_result.error is None

    result = await tool.execute(patch=patch_text, context=context)

    assert result.error is None
    assert target.read_text(encoding="utf-8") == "alpha\ngamma\n"
    assert result.metadata["change_count"] == 1
    assert result.metadata["file_changes"][0]["change_type"] == "update"
    assert "-beta" in str(result.metadata["diff"])
    assert "+gamma" in str(result.metadata["diff"])


@pytest.mark.asyncio
async def test_apply_patch_rejects_protected_paths(tmp_path: Path) -> None:
    tool = ApplyPatchTool(workspace_dir=tmp_path, require_approval=False)
    patch_text = "\n".join(
        [
            "*** Begin Patch",
            "*** Add File: .git/config",
            "+blocked",
            "*** End Patch",
        ]
    )

    result = await tool.execute(patch=patch_text)

    assert result.error is not None
    assert "protected path" in result.error.lower()


@pytest.mark.asyncio
async def test_apply_patch_uses_task_sandbox_workspace(tmp_path: Path) -> None:
    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    tool = ApplyPatchTool(workspace_dir=tmp_path, require_approval=False)
    patch_text = "\n".join(
        [
            "*** Begin Patch",
            "*** Add File: sandbox.txt",
            "+hello sandbox",
            "*** End Patch",
        ]
    )

    result = await tool.execute(
        patch=patch_text,
        context=ToolExecutionContext(
            workspace_dir=str(tmp_path),
            task_sandbox_dir=str(sandbox_dir),
        ),
    )

    assert result.error is None
    assert (sandbox_dir / "sandbox.txt").exists()
    assert not (tmp_path / "sandbox.txt").exists()


@pytest.mark.asyncio
async def test_mcp_runtime_surfaces_dynamic_tools_and_resources(tmp_path: Path) -> None:
    adapter = _FakeMcpAdapter()
    runtime = McpRuntimeManager(
        servers={
            "demo": McpServerConfig(name="demo", transport="in_memory"),
        },
        adapters={"demo": adapter},
    )
    await runtime.refresh_server("demo")

    tools = runtime.materialize_tools()
    dynamic_tool = next(tool for tool in tools if tool.name == "mcp__demo__search_docs")
    assert isinstance(dynamic_tool, McpDynamicTool)
    assert dynamic_tool.is_read_only is True
    assert dynamic_tool.is_open_world is True
    assert dynamic_tool.search_hint == "Use before fetching a specific URL."

    tool_result = await dynamic_tool.execute(query="mochi")
    assert tool_result.error is None
    assert tool_result.output == {"tool": "search_docs", "arguments": {"query": "mochi"}, "ok": True}

    list_tool = McpListResourcesTool(runtime=runtime)
    list_result = await list_tool.execute(server="demo")
    assert list_result.error is None
    assert list_result.output == [{"uri": "memo://welcome", "name": "welcome"}]

    read_tool = McpReadResourceTool(runtime=runtime)
    read_result = await read_tool.execute(server="demo", uri="memo://welcome")
    assert read_result.error is None
    assert read_result.output == {"uri": "memo://welcome", "text": "hello from resource"}


def test_registry_can_register_workspace_search_tools() -> None:
    registry = ToolRegistry(discover_builtin=False)
    glob_tool = GlobSearchTool(workspace_dir=".")
    grep_tool = GrepSearchTool(workspace_dir=".")
    csv_tool = CsvReadTool(workspace_dir=".")
    pdf_tool = PdfReadTool(workspace_dir=".")
    notebook_tool = NotebookReadTool(workspace_dir=".")
    memory_update_tool = MemoryUpdateTool(workspace_dir=".")
    memory_delete_tool = MemoryDeleteTool(workspace_dir=".")
    memory_export_tool = MemoryExportTool(workspace_dir=".")
    tool_search_tool = ToolSearchTool(catalog_provider=lambda: [])
    web_crawl_tool = WebCrawlTool()
    execute_code_v2_tool = ExecuteCodeV2Tool(workspace_dir=".")

    registry.register(glob_tool)
    registry.register(grep_tool)
    registry.register(csv_tool)
    registry.register(pdf_tool)
    registry.register(notebook_tool)
    registry.register(memory_update_tool)
    registry.register(memory_delete_tool)
    registry.register(memory_export_tool)
    registry.register(tool_search_tool)
    registry.register(web_crawl_tool)
    registry.register(execute_code_v2_tool)

    assert registry.get("glob_search") is glob_tool
    assert registry.get("grep_search") is grep_tool
    assert registry.get("csv_read") is csv_tool
    assert registry.get("pdf_read") is pdf_tool
    assert registry.get("notebook_read") is notebook_tool
    assert registry.get("memory_update") is memory_update_tool
    assert registry.get("memory_delete") is memory_delete_tool
    assert registry.get("memory_export") is memory_export_tool
    assert registry.get("tool_search") is tool_search_tool
    assert registry.get("web_crawl") is web_crawl_tool
    assert registry.get("execute_code_v2") is execute_code_v2_tool


def test_registry_factory_registers_tool_search(tmp_path: Path) -> None:
    config = MochiConfig.model_validate(
        {
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db")},
        }
    )
    factory = ToolRegistryFactory(
        config,
        memory_store=MemoryStore(db_path=tmp_path / "memory.db"),
    )

    registry = factory.create_registry(str(tmp_path))

    assert registry.get("tool_search") is not None
    assert registry.get("apply_patch") is not None
    assert registry.get("shell") is None
    assert "tool_search" in factory.tool_groups["workspace"]
    assert "apply_patch" in factory.tool_groups["workspace"]
    assert "shell" not in factory.tool_groups["workspace"]
    assert "web_crawl" in factory.tool_groups["web"]


@pytest.mark.asyncio
async def test_mcp_call_uses_runtime_when_available() -> None:
    adapter = _FakeMcpAdapter()
    runtime = McpRuntimeManager(
        servers={
            "demo": McpServerConfig(name="demo", transport="in_memory"),
        },
        adapters={"demo": adapter},
    )
    await runtime.refresh_server("demo")
    tool = MCPCallTool(runtime=runtime)

    result = await tool.execute(server="demo", tool="search_docs", arguments={"query": "mochi"})

    assert result.error is None
    assert result.output == {
        "tool": "search_docs",
        "arguments": {"query": "mochi"},
        "ok": True,
    }


@pytest.mark.asyncio
async def test_web_search_returns_normalized_payload_and_domain_filters() -> None:
    tool = WebSearchTool(engine="tavily", tavily_api_key="test-key")
    payload = {
        "results": [
            {
                "title": "Alpha",
                "url": "https://docs.example.com/alpha",
                "content": "Alpha snippet",
                "raw_content": "Full alpha page",
            }
        ]
    }

    with patch.object(
        tool._client,
        "request",
        new_callable=AsyncMock,
        return_value=_mock_search_response(json_payload=payload),
    ) as request_mock:
        result = await tool.execute(
            query="mochi docs",
            top_k=2,
            include_content=True,
            allowed_domains=["docs.example.com"],
            blocked_domains=["ads.example.com"],
        )

    assert result.error is None
    assert result.output == {
        "query": "mochi docs",
        "provider": "tavily",
        "results": [
            {
                "title": "Alpha",
                "url": "https://docs.example.com/alpha",
                "snippet": "Alpha snippet",
                "content": "Full alpha page",
            }
        ],
        "warnings": [],
        "attempted_providers": ["tavily"],
    }

    assert request_mock.await_args.kwargs["json"]["include_domains"] == ["docs.example.com"]
    assert request_mock.await_args.kwargs["json"]["exclude_domains"] == ["ads.example.com"]
    await tool.close()


def _mock_search_response(*, json_payload: dict[str, Any], status_code: int = 200) -> Any:
    class _Response:
        def __init__(self) -> None:
            self.status_code = status_code
            self.headers: dict[str, str] = {}

        def json(self) -> dict[str, Any]:
            return json_payload

    return _Response()
