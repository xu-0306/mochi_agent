"""Tool system upgrade tests for phases 1-4."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from mochi.config.schema import MochiConfig
from mochi.memory.store import MemoryStore
from mochi.tools.base import BaseTool, ToolExecutionContext, ToolResult
from mochi.tools.csv_read import CsvReadTool
from mochi.tools.execute_code_v2 import ExecuteCodeV2Tool
from mochi.tools.file_ops import ApplyPatchTool, FileEditTool, FileReadTool, FileWriteTool
from mochi.tools.glob_search import GlobSearchTool
from mochi.tools.grep_search import GrepSearchTool
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
from mochi.tools.tool_search import ToolSearchTool
from mochi.tools.web_crawl import WebCrawlTool
from mochi.tools.web_search import WebSearchTool


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
async def test_denied_file_write_call_returns_tool_denied_metadata(tmp_path: Path) -> None:
    registry = ToolRegistry(discover_builtin=False)
    registry.register(FileWriteTool(workspace_dir=tmp_path, require_approval=False))
    arguments = {"path": "created.txt", "content": "hello"}
    context = ToolExecutionContext(
        workspace_dir=str(tmp_path),
        permission_policy={
            "denied_tool_calls": [
                {"tool_name": "file_write", "arguments": arguments},
            ],
        },
    )

    result = await registry.execute("file_write", arguments, context=context)

    assert result.error is not None
    assert result.error.startswith("tool_denied:")
    assert result.metadata["runtime_category"] == "permission"
    assert result.metadata["error_type"] == "tool_denied"
    assert result.metadata["recoverability"] == "requires_changed_tool_call"
    assert result.metadata["status"] == "tool_denied"
    assert result.metadata["code"] == "tool_denied"
    assert result.metadata["tool_name"] == "file_write"
    assert result.metadata["arguments"] == arguments
    assert result.metadata["requires_approval"] is False
    assert result.retryable is False
    assert not (tmp_path / "created.txt").exists()


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
    assert result.metadata["error_type"] == "file_path_denied"
    assert result.metadata["runtime_category"] == "permission"
    assert result.metadata["approval_scope"] == "protected_path"


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


def test_registry_factory_caches_registries_per_workspace(tmp_path: Path) -> None:
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

    primary = factory.create_registry(str(tmp_path))
    primary_again = factory.create_registry(str(tmp_path))
    secondary_workspace = tmp_path / "other"
    secondary_workspace.mkdir()
    secondary = factory.create_registry(str(secondary_workspace))

    assert primary is primary_again
    assert secondary is not primary
    assert factory.list_cached_registries() == [primary, secondary]


def test_registry_get_schemas_for_names_returns_requested_schemas_only() -> None:
    registry = ToolRegistry(discover_builtin=False)
    registry.register(_EchoTool())
    registry.register(_GuardedTool())

    schemas = registry.get_schemas_for_names(["guarded", "missing", "echo"])

    assert [schema["function"]["name"] for schema in schemas] == ["guarded", "echo"]


def test_registry_create_view_skips_registration_debug_logs_for_view_population() -> None:
    registry = ToolRegistry(discover_builtin=False)
    registry.register(_EchoTool())
    registry.register(ToolSearchTool(catalog_provider=registry.list_tools))

    with patch("mochi.tools.registry.logger.debug") as debug_mock:
        view = registry.create_view(["echo", "tool_search"])

    assert view.get("echo") is not None
    assert view.get("tool_search") is not None
    debug_mock.assert_not_called()


@pytest.mark.asyncio
async def test_tool_search_marks_discoverable_but_not_callable_tools(tmp_path: Path) -> None:
    registry = ToolRegistry(discover_builtin=False)
    registry.register(ToolSearchTool(catalog_provider=registry.list_tools))
    registry.register(FileReadTool(workspace_dir=tmp_path))
    registry.register(FileWriteTool(workspace_dir=tmp_path, require_approval=False))

    view = registry.create_view(
        ["tool_search", "file_read"],
        tool_search_catalog_names=["tool_search", "file_read", "file_write"],
    )

    result = await view.execute("tool_search", {"query": "file_write", "top_k": 10})

    assert result.error is None
    assert isinstance(result.output, list)
    match = next(item for item in result.output if item["name"] == "file_write")
    assert view.get("file_write") is None
    assert match["callable_this_turn"] is False
    assert match["activation_required"] is True
    assert "not exposed" in match["activation_reason"].lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "tool_factory"),
    [
        ("file_write", lambda workspace: FileWriteTool(workspace_dir=workspace, require_approval=False)),
        ("file_edit", lambda workspace: FileEditTool(workspace_dir=workspace, require_approval=False)),
        ("apply_patch", lambda workspace: ApplyPatchTool(workspace_dir=workspace, require_approval=False)),
    ],
)
async def test_tool_search_hidden_core_mutation_tools_include_activation_request(
    tmp_path: Path,
    tool_name: str,
    tool_factory: Any,
) -> None:
    registry = ToolRegistry(discover_builtin=False)
    registry.register(ToolSearchTool(catalog_provider=registry.list_tools))
    registry.register(tool_factory(tmp_path))

    view = registry.create_view(
        ["tool_search"],
        tool_search_catalog_names=["tool_search", tool_name],
    )

    result = await view.execute("tool_search", {"query": tool_name, "top_k": 10})

    assert result.error is None
    match = next(item for item in result.output if item["name"] == tool_name)
    assert match["callable_this_turn"] is False
    assert match["activation_required"] is True
    assert match["activation_request"] == {
        "tool_name": tool_name,
        "required_intent": "workspace_write",
        "policy_check": "required",
    }


@pytest.mark.asyncio
async def test_tool_search_callable_file_write_omits_activation_request(tmp_path: Path) -> None:
    registry = ToolRegistry(discover_builtin=False)
    registry.register(ToolSearchTool(catalog_provider=registry.list_tools))
    registry.register(FileWriteTool(workspace_dir=tmp_path, require_approval=False))

    view = registry.create_view(
        ["tool_search", "file_write"],
        tool_search_catalog_names=["tool_search", "file_write"],
    )

    result = await view.execute("tool_search", {"query": "file_write", "top_k": 10})

    assert result.error is None
    match = next(item for item in result.output if item["name"] == "file_write")
    assert match["callable_this_turn"] is True
    assert "activation_request" not in match


@pytest.mark.asyncio
async def test_tool_search_hidden_read_only_tool_omits_activation_request(tmp_path: Path) -> None:
    registry = ToolRegistry(discover_builtin=False)
    registry.register(ToolSearchTool(catalog_provider=registry.list_tools))
    registry.register(FileReadTool(workspace_dir=tmp_path))

    view = registry.create_view(
        ["tool_search"],
        tool_search_catalog_names=["tool_search", "file_read"],
    )

    result = await view.execute("tool_search", {"query": "file_read", "top_k": 10})

    assert result.error is None
    match = next(item for item in result.output if item["name"] == "file_read")
    assert match["callable_this_turn"] is False
    assert "activation_request" not in match


@pytest.mark.asyncio
async def test_tool_search_execute_does_not_change_registry_callable_state(tmp_path: Path) -> None:
    registry = ToolRegistry(discover_builtin=False)
    registry.register(ToolSearchTool(catalog_provider=registry.list_tools))
    registry.register(FileReadTool(workspace_dir=tmp_path))
    registry.register(FileWriteTool(workspace_dir=tmp_path, require_approval=False))

    view = registry.create_view(
        ["tool_search", "file_read"],
        tool_search_catalog_names=["tool_search", "file_read", "file_write"],
    )

    tool_search = view.get("tool_search")
    assert isinstance(tool_search, ToolSearchTool)

    file_write_before = view.get("file_write")
    schemas_before = view.get_schemas()
    callable_names_before = tool_search._callable_name_provider()

    assert file_write_before is None

    result = await view.execute("tool_search", {"query": "file_write", "top_k": 10})

    assert result.error is None
    assert view.get("file_write") is None
    assert view.get_schemas() == schemas_before
    assert tool_search._callable_name_provider() == callable_names_before

def _make_tool_activation_workspace() -> Path:
    root = Path(".tmp") / "tool-activation-tests"
    root.mkdir(parents=True, exist_ok=True)
    workspace = root / uuid4().hex
    workspace.mkdir(parents=True, exist_ok=False)
    return workspace.resolve()
def _tool_activation_context(
    *,
    workspace: Path,
    routed_intent: str = "workspace_write",
    execution_profile: str = "chat",
    tool_mode: str = "auto",
    discoverable_tool_names: list[str] | None = None,
    tool_allowlist: list[str] | None = None,
    tool_denylist: list[str] | None = None,
    permission_policy: dict[str, Any] | None = None,
) -> ToolExecutionContext:
    return ToolExecutionContext(
        workspace_dir=str(workspace),
        permission_policy={
            "autonomy_mode": "auto_review",
            "require_approval_for_file_write": False,
            "file_ops_scope": "workspace",
            **(permission_policy or {}),
        },
        state={
            "tool_activation_policy": {
                "routed_intent": routed_intent,
                "execution_profile": execution_profile,
                "tool_mode": tool_mode,
                "discoverable_tool_names": list(discoverable_tool_names or []),
                "tool_allowlist": list(tool_allowlist) if tool_allowlist is not None else None,
                "tool_denylist": list(tool_denylist) if tool_denylist is not None else None,
            }
        },
    )


def test_registry_view_workspace_write_activation_promotes_hidden_file_write() -> None:
    tmp_path = _make_tool_activation_workspace()
    registry = ToolRegistry(discover_builtin=False)
    registry.register(FileWriteTool(workspace_dir=tmp_path, require_approval=False))
    view = registry.create_view([], tool_search_catalog_names=["file_write"])
    context = _tool_activation_context(
        workspace=tmp_path,
        discoverable_tool_names=["file_write"],
    )

    assert view.get("file_write") is None

    result = view.request_tool_activation("file_write", context=context)

    assert result.error is None
    assert result.metadata["status"] == "tool_activated"
    assert result.metadata["requested_tool"] == "file_write"
    assert view.get("file_write") is not None
    assert "file_write" in {
        schema["function"]["name"]
        for schema in view.get_schemas()
    }


@pytest.mark.parametrize(
    ("policy_kwargs", "expected_reason"),
    [
        (
            {"execution_profile": "subagent_readonly", "discoverable_tool_names": []},
            "execution_profile_disallows_activation",
        ),
        (
            {"tool_allowlist": ["tool_search"], "discoverable_tool_names": []},
            "allowlist_excluded",
        ),
        (
            {"tool_denylist": ["file_write"], "discoverable_tool_names": []},
            "denylist_blocked",
        ),
        (
            {
                "discoverable_tool_names": ["file_write"],
                "permission_policy": {"require_approval_for_file_write": True},
            },
            "approval_required",
        ),
    ],
)
def test_registry_view_tool_activation_denials_return_structured_metadata(
    policy_kwargs: dict[str, Any],
    expected_reason: str,
) -> None:
    tmp_path = _make_tool_activation_workspace()
    registry = ToolRegistry(discover_builtin=False)
    registry.register(FileWriteTool(workspace_dir=tmp_path, require_approval=False))
    view = registry.create_view([], tool_search_catalog_names=["file_write"])
    context = _tool_activation_context(workspace=tmp_path, **policy_kwargs)

    result = view.request_tool_activation("file_write", context=context)

    assert result.error is not None
    assert result.metadata["error_type"] == "tool_activation_denied"
    assert result.metadata["requested_tool"] == "file_write"
    assert result.metadata["reason"] == expected_reason
    assert result.metadata["recoverability"]
    assert view.get("file_write") is None


def test_registry_view_file_write_activation_accepts_existing_approval() -> None:
    tmp_path = _make_tool_activation_workspace()
    arguments = {"path": "approved.txt", "content": "approved"}
    registry = ToolRegistry(discover_builtin=False)
    registry.register(FileWriteTool(workspace_dir=tmp_path, require_approval=False))
    view = registry.create_view([], tool_search_catalog_names=["file_write"])
    context = _tool_activation_context(
        workspace=tmp_path,
        discoverable_tool_names=["file_write"],
        permission_policy={
            "require_approval_for_file_write": True,
            "approved_tool_calls": [
                {"tool_name": "file_write", "arguments": arguments},
            ],
        },
    )

    result = view.request_tool_activation("file_write", context=context)

    assert result.error is None
    assert result.metadata["status"] == "tool_activated"
    assert view.get("file_write") is not None

@pytest.mark.asyncio
async def test_registry_approved_file_write_is_argument_scoped() -> None:
    workspace = _make_tool_activation_workspace()
    approved_arguments = {"path": "approved.txt", "content": "approved"}
    unapproved_arguments = {"path": "other.txt", "content": "other"}
    registry = ToolRegistry(discover_builtin=False)
    registry.register(FileWriteTool(workspace_dir=workspace, require_approval=False))
    view = registry.create_view([], tool_search_catalog_names=["file_write"])
    context = _tool_activation_context(
        workspace=workspace,
        discoverable_tool_names=["file_write"],
        permission_policy={
            "require_approval_for_file_write": True,
            "approved_tool_calls": [
                {"tool_name": "file_write", "arguments": approved_arguments},
            ],
        },
    )

    activation = view.request_tool_activation("file_write", context=context)
    approved = await view.execute("file_write", approved_arguments, context=context)
    unapproved = await view.execute("file_write", unapproved_arguments, context=context)

    assert activation.error is None
    assert approved.error is None
    assert (workspace / "approved.txt").read_text() == "approved"
    assert unapproved.error == "File write requires approval."
    assert unapproved.metadata["approval_kind"] == "file_write"
    assert not (workspace / "other.txt").exists()


def test_registry_denied_activation_replay_is_structured_and_not_promoted() -> None:
    workspace = _make_tool_activation_workspace()
    registry = ToolRegistry(discover_builtin=False)
    registry.register(FileWriteTool(workspace_dir=workspace, require_approval=False))
    view = registry.create_view([], tool_search_catalog_names=["file_write"])
    context = _tool_activation_context(
        workspace=workspace,
        discoverable_tool_names=["file_write"],
        tool_allowlist=["tool_search"],
    )

    first = view.request_tool_activation("file_write", context=context)
    second = view.request_tool_activation("file_write", context=context)

    assert first.metadata["reason"] == "allowlist_excluded"
    assert second.metadata["reason"] == "activation_denied_replay"
    assert second.metadata["error_type"] == "tool_activation_denied"
    assert second.metadata["runtime_category"] == "tool_activation"
    assert second.metadata["recoverability"]
    assert second.retryable is False
    assert view.get("file_write") is None


def test_activation_replay_isolated_between_sources_with_same_catalog() -> None:
    workspace = _make_tool_activation_workspace()
    missing_source = ToolRegistry(discover_builtin=False)
    populated_source = ToolRegistry(discover_builtin=False)
    populated_source.register(FileWriteTool(workspace_dir=workspace, require_approval=False))
    missing_view = missing_source.create_view([], tool_search_catalog_names=["file_write"])
    populated_view = populated_source.create_view([], tool_search_catalog_names=["file_write"])
    context = _tool_activation_context(
        workspace=workspace,
        discoverable_tool_names=["file_write"],
    )

    missing = missing_view.request_tool_activation("file_write", context=context)
    promoted = populated_view.request_tool_activation("file_write", context=context)

    assert missing.metadata["reason"] == "tool_not_found"
    assert promoted.error is None
    assert promoted.metadata["status"] == "tool_activated"
    assert populated_view.get("file_write") is not None

def test_activation_denial_re_evaluates_after_workspace_fix() -> None:
    workspace = _make_tool_activation_workspace()
    registry = ToolRegistry(discover_builtin=False)
    registry.register(FileWriteTool(workspace_dir=workspace, require_approval=False))
    view = registry.create_view([], tool_search_catalog_names=["file_write"])
    wrong_workspace = workspace / "wrong"
    context = _tool_activation_context(
        workspace=wrong_workspace,
        discoverable_tool_names=["file_write"],
    )

    denied = view.request_tool_activation("file_write", context=context)
    context.workspace_dir = str(workspace)
    promoted = view.request_tool_activation("file_write", context=context)

    assert denied.metadata["reason"] == "workspace_security_rejected"
    assert promoted.error is None
    assert promoted.metadata["status"] == "tool_activated"
    assert view.get("file_write") is not None

@pytest.mark.parametrize(
    ("tool_name", "tool_factory", "approval_kind"),
    [
        (
            "file_write",
            lambda workspace: FileWriteTool(workspace_dir=workspace, require_approval=False),
            "file_write",
        ),
        (
            "file_edit",
            lambda workspace: FileEditTool(workspace_dir=workspace, require_approval=False),
            "file_edit",
        ),
        (
            "apply_patch",
            lambda workspace: ApplyPatchTool(workspace_dir=workspace, require_approval=False),
            "apply_patch",
        ),
    ],
)
def test_activation_approval_kind_matches_tool(
    tool_name: str,
    tool_factory: Any,
    approval_kind: str,
) -> None:
    workspace = _make_tool_activation_workspace()
    registry = ToolRegistry(discover_builtin=False)
    registry.register(tool_factory(workspace))
    view = registry.create_view([], tool_search_catalog_names=[tool_name])
    context = _tool_activation_context(
        workspace=workspace,
        discoverable_tool_names=[tool_name],
        permission_policy={"require_approval_for_file_write": True},
    )

    result = view.request_tool_activation(tool_name, context=context)

    assert result.error is not None
    assert result.metadata["error_type"] == "tool_activation_denied"
    assert result.metadata["approval_kind"] == approval_kind
    assert result.metadata["runtime_category"] == "tool_activation"

def test_activation_denial_isolated_between_registry_views() -> None:
    workspace = _make_tool_activation_workspace()
    registry = ToolRegistry(discover_builtin=False)
    registry.register(FileWriteTool(workspace_dir=workspace, require_approval=False))
    denied_view = registry.create_view([], tool_search_catalog_names=["file_read"])
    allowed_view = registry.create_view([], tool_search_catalog_names=["file_write"])
    context = _tool_activation_context(workspace=workspace, discoverable_tool_names=[])

    denied = denied_view.request_tool_activation("file_write", context=context)
    promoted = allowed_view.request_tool_activation("file_write", context=context)

    assert denied.metadata["reason"] == "not_discoverable"
    assert promoted.error is None
    assert promoted.metadata["status"] == "tool_activated"
    assert denied_view.get("file_write") is None
    assert allowed_view.get("file_write") is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_factory", "arguments"),
    [
        (
            lambda workspace: FileWriteTool(workspace_dir=workspace, require_approval=False),
            {"path": ".git/config", "content": "blocked"},
        ),
        (
            lambda workspace: FileEditTool(workspace_dir=workspace, require_approval=False),
            {"path": ".git/config", "old_string": "old", "new_string": "new"},
        ),
    ],
)
async def test_file_mutation_tools_return_structured_path_denial(
    tool_factory: Any,
    arguments: dict[str, str],
) -> None:
    workspace = _make_tool_activation_workspace()
    tool = tool_factory(workspace)

    result = await tool.execute(
        **arguments,
        context=ToolExecutionContext(workspace_dir=str(workspace)),
    )

    assert result.error is not None
    assert result.metadata["error_type"] == "file_path_denied"
    assert result.metadata["runtime_category"] == "permission"
    assert result.metadata["approval_scope"] == "protected_path"
    assert not (workspace / ".git" / "config").exists()


@pytest.mark.asyncio
async def test_file_write_allows_explicit_workspace_nested_under_mochi_state_root(
    tmp_path: Path,
) -> None:
    """A configured workspace child under the Mochi state root is writable without exposing state files."""
    workspace = tmp_path / ".mochi" / "workspace"
    workspace.mkdir(parents=True)
    tool = FileWriteTool(workspace_dir=workspace, require_approval=False)

    result = await tool.execute(
        path="test.txt",
        content="hi",
        context=ToolExecutionContext(workspace_dir=str(workspace)),
    )

    assert result.error is None
    assert (workspace / "test.txt").read_text(encoding="utf-8") == "hi"
    assert result.metadata["workspace_dir"] == str(workspace.resolve())
    assert result.metadata["resolved_path"] == str((workspace / "test.txt").resolve())


@pytest.mark.asyncio
async def test_apply_patch_protected_path_returns_structured_path_denial() -> None:
    workspace = _make_tool_activation_workspace()
    tool = ApplyPatchTool(workspace_dir=workspace, require_approval=False)
    patch_text = "\n".join(
        [
            "*** Begin Patch",
            "*** Add File: .git/config",
            "+blocked",
            "*** End Patch",
        ]
    )

    result = await tool.execute(
        patch=patch_text,
        context=ToolExecutionContext(workspace_dir=str(workspace)),
    )

    assert result.error is not None
    assert result.metadata["error_type"] == "file_path_denied"
    assert result.metadata["runtime_category"] == "permission"
    assert result.metadata["approval_scope"] == "protected_path"
    assert result.metadata["workspace_dir"] == str(workspace.resolve())
    assert result.metadata["requested_path"] == ".git/config"
    assert result.metadata["path_scope"] == "workspace"
    assert result.metadata["resolved_path"] == str((workspace / ".git" / "config").resolve())
    assert not (workspace / ".git" / "config").exists()

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
