"""Phase 2 工具與安全模組測試。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from mochi.tools.base import BaseTool, ToolExecutionContext, ToolResult
from mochi.tools.execute_code import ExecuteCodeTool
from mochi.tools.file_ops import FileReadTool, FileWriteTool
from mochi.tools.literature_search import (
    ArxivSearchTool,
    CrossrefSearchTool,
    PubMedSearchTool,
    SemanticScholarSearchTool,
)
from mochi.tools.mcp_client import MCPCallTool
from mochi.tools.memory_save import MemorySaveTool
from mochi.tools.memory_search import MemorySearchTool
from mochi.tools.registry import ToolRegistry
from mochi.tools.shell import ShellTool
from mochi.tools.web_fetch import WebFetchTool
from mochi.tools.web_search import WebSearchTool
from mochi.utils.security import (
    is_path_within_workspace,
    is_safe_command,
    is_within_write_size_limit,
    resolve_path_in_workspace,
)

_HAN_RE = re.compile(r"[\u4e00-\u9fff]")


def _assert_no_han_text(value: Any) -> None:
    if isinstance(value, str):
        assert _HAN_RE.search(value) is None
        return
    if isinstance(value, dict):
        for nested in value.values():
            _assert_no_han_text(nested)
        return
    if isinstance(value, list):
        for nested in value:
            _assert_no_han_text(nested)


def test_is_safe_command_with_allowlist_and_shell_syntax() -> None:
    """命令安全判斷應同時檢查白名單與危險 shell 語法。"""
    allowlist = ["echo", "ls"]

    assert is_safe_command("echo hello", allowlist) is True
    assert is_safe_command("/bin/ls -la", allowlist) is True
    assert is_safe_command("echo hello && ls", allowlist) is False
    assert is_safe_command("rm -rf /", allowlist) is False


def test_workspace_path_restriction(tmp_path: Path) -> None:
    """路徑應被限制於 workspace 內。"""
    inside = resolve_path_in_workspace("notes/a.txt", tmp_path)
    assert inside == (tmp_path / "notes" / "a.txt").resolve(strict=False)
    assert is_path_within_workspace(inside, tmp_path) is True

    outside = tmp_path.parent / "outside.txt"
    assert is_path_within_workspace(outside, tmp_path) is False
    with pytest.raises(ValueError):
        resolve_path_in_workspace(outside, tmp_path)


def test_write_size_limit_check() -> None:
    """寫入大小限制應正確判斷。"""
    assert is_within_write_size_limit("hello", max_size_mb=0.00001) is True
    assert is_within_write_size_limit("x" * 200, max_size_mb=0.00001) is False


@pytest.mark.asyncio
async def test_shell_tool_allowlist_and_approval(tmp_path: Path) -> None:
    """Shell 工具應套用 allowlist 並尊重 approval 設定。"""
    tool = ShellTool(
        allowlist=["echo"],
        workspace_dir=tmp_path,
        require_approval=True,
    )

    approval_needed = await tool.execute(command="echo hi")
    assert approval_needed.error is not None
    assert "approval" in approval_needed.error.lower()

    denied = await tool.execute(command="pwd", approved=True)
    assert denied.error is not None
    assert "denied" in denied.error.lower()

    ok = await tool.execute(command="echo hi", approved=True)
    assert ok.error is None
    assert "hi" in str(ok.output)


@pytest.mark.asyncio
async def test_shell_tool_supports_injected_runner(tmp_path: Path) -> None:
    """Shell 工具應可注入 runner。"""
    captured: dict[str, Any] = {}

    async def fake_runner(command: str, cwd: Path, timeout_sec: int) -> tuple[int, str, str]:
        captured["command"] = command
        captured["cwd"] = cwd
        captured["timeout_sec"] = timeout_sec
        return 0, "ok", ""

    tool = ShellTool(
        allowlist=["echo"],
        workspace_dir=tmp_path,
        require_approval=False,
        runner=fake_runner,
    )
    result = await tool.execute(command="echo injected")
    assert result.error is None
    assert result.output == "ok"
    assert captured["command"] == "echo injected"
    assert captured["cwd"] == tmp_path.resolve(strict=False)


@pytest.mark.asyncio
async def test_shell_tool_prefers_task_sandbox_from_context(tmp_path: Path) -> None:
    """Shell tool should default cwd to context task sandbox when provided."""
    captured: dict[str, Any] = {}
    sandbox_dir = tmp_path / "task-sandbox"
    sandbox_dir.mkdir(parents=True, exist_ok=True)

    async def fake_runner(command: str, cwd: Path, timeout_sec: int) -> tuple[int, str, str]:
        captured["command"] = command
        captured["cwd"] = cwd
        captured["timeout_sec"] = timeout_sec
        return 0, "ok", ""

    tool = ShellTool(
        allowlist=["echo"],
        workspace_dir=tmp_path,
        require_approval=False,
        runner=fake_runner,
    )
    result = await tool.execute(
        command="echo sandbox",
        context=ToolExecutionContext(
            workspace_dir=str(tmp_path),
            task_sandbox_dir=str(sandbox_dir),
        ),
    )
    assert result.error is None
    assert captured["cwd"] == sandbox_dir.resolve(strict=False)


@pytest.mark.asyncio
async def test_file_tools_security_and_write_size(tmp_path: Path) -> None:
    """file_read/file_write 應限制 workspace、檢查大小與審批。"""
    writer = FileWriteTool(
        workspace_dir=tmp_path,
        require_approval=True,
        max_write_size_mb=0.001,
    )
    reader = FileReadTool(workspace_dir=tmp_path)

    approval_needed = await writer.execute(path="memo.txt", content="hello")
    assert approval_needed.error is not None
    assert "approval" in approval_needed.error.lower()

    write_ok = await writer.execute(path="memo.txt", content="hello", approved=True)
    assert write_ok.error is None
    assert (tmp_path / "memo.txt").exists()

    read_ok = await reader.execute(path="memo.txt")
    assert read_ok.error is None
    assert read_ok.output == "hello"

    outside = tmp_path.parent / "outside.txt"
    write_outside = await writer.execute(path=str(outside), content="x", approved=True)
    assert write_outside.error is not None
    assert "outside workspace" in write_outside.error.lower()

    tiny_limit_writer = FileWriteTool(
        workspace_dir=tmp_path,
        require_approval=False,
        max_write_size_mb=0.00001,
    )
    too_large = await tiny_limit_writer.execute(path="big.txt", content="x" * 256)
    assert too_large.error is not None
    assert "too large" in too_large.error.lower()


@pytest.mark.asyncio
async def test_memory_tools_default_jsonl_store(tmp_path: Path) -> None:
    """memory_save 與 memory_search 預設 JSONL store 應可協作。"""
    save_tool = MemorySaveTool(workspace_dir=tmp_path)
    search_tool = MemorySearchTool(workspace_dir=tmp_path)

    save_1 = await save_tool.execute(
        content="Mochi uses async tools and ReAct loop.",
        category="architecture",
        metadata={"source": "unit-test"},
    )
    assert save_1.error is None
    assert "memory_id" in save_1.metadata

    save_2 = await save_tool.execute(content="Voice pipeline has VAD/STT/TTS.", category="voice")
    assert save_2.error is None

    found = await search_tool.execute(query="async", top_k=5)
    assert found.error is None
    assert isinstance(found.output, list)
    assert any("async" in item.get("content", "").lower() for item in found.output)


@pytest.mark.asyncio
async def test_memory_tools_support_dependency_injection() -> None:
    """memory tools 應支援 constructor 注入記憶依賴。"""

    class FakeMemoryStore:
        def __init__(self) -> None:
            self._items: list[dict[str, Any]] = []

        async def save(self, content: str, category: str, metadata: dict[str, Any]) -> str:
            memory_id = f"id-{len(self._items) + 1}"
            self._items.append(
                {
                    "id": memory_id,
                    "content": content,
                    "category": category,
                    "metadata": metadata,
                }
            )
            return memory_id

        async def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
            query_lc = query.lower()
            matched = [item for item in self._items if query_lc in item["content"].lower()]
            return matched[:top_k]

    store = FakeMemoryStore()
    save_tool = MemorySaveTool(memory_store=store)
    search_tool = MemorySearchTool(memory_store=store)

    saved = await save_tool.execute(content="custom dependency injection works")
    assert saved.error is None

    found = await search_tool.execute(query="injection")
    assert found.error is None
    assert len(found.output) == 1
    assert found.output[0]["id"] == "id-1"


@pytest.mark.asyncio
async def test_builtin_tool_descriptions_are_english_default(tmp_path: Path) -> None:
    """Built-in tool descriptions should not inject Chinese into LLM-facing schemas."""
    web_search = WebSearchTool()
    web_fetch = WebFetchTool()
    arxiv_search = ArxivSearchTool()
    semantic_scholar_search = SemanticScholarSearchTool()
    crossref_search = CrossrefSearchTool()
    pubmed_search = PubMedSearchTool()
    tools = [
        ShellTool(workspace_dir=tmp_path),
        FileReadTool(workspace_dir=tmp_path),
        FileWriteTool(workspace_dir=tmp_path),
        MemorySaveTool(workspace_dir=tmp_path),
        MemorySearchTool(workspace_dir=tmp_path),
        ExecuteCodeTool(workspace_dir=tmp_path),
        MCPCallTool(),
        web_search,
        web_fetch,
        arxiv_search,
        semantic_scholar_search,
        crossref_search,
        pubmed_search,
    ]

    try:
        for tool in tools:
            _assert_no_han_text(tool.description)
            _assert_no_han_text(tool.parameters_schema)
    finally:
        await web_search.close()
        await web_fetch.close()
        await arxiv_search.close()
        await semantic_scholar_search.close()
        await crossref_search.close()
        await pubmed_search.close()


@pytest.mark.asyncio
async def test_registry_auto_injects_approved_for_exact_tool_call(tmp_path: Path) -> None:
    """Registry should inject approved=True only for an exact tool+arguments match."""

    class _ApprovalProbeTool(BaseTool):
        @property
        def name(self) -> str:
            return "approval_probe"

        @property
        def description(self) -> str:
            return "probe"

        @property
        def parameters_schema(self) -> dict[str, Any]:
            return {
                "type": "object",
                "properties": {"value": {"type": "integer"}, "approved": {"type": "boolean"}},
                "required": ["value"],
                "additionalProperties": False,
            }

        async def execute(self, *, value: int, approved: bool = False) -> ToolResult:
            if not approved:
                return ToolResult(error="requires approval")
            return ToolResult(output={"value": value, "approved": approved})

    registry = ToolRegistry(discover_builtin=False)
    registry.register(_ApprovalProbeTool())

    denied = await registry.execute("approval_probe", {"value": 3})
    assert denied.error == "requires approval"

    context = ToolExecutionContext(
        workspace_dir=str(tmp_path),
        permission_policy={
            "approved_tool_calls": [
                {"tool_name": "approval_probe", "arguments": {"value": 7}},
            ]
        },
    )
    approved = await registry.execute("approval_probe", {"value": 7}, context=context)
    assert approved.error is None
    assert approved.output == {"value": 7, "approved": True}

    not_matched = await registry.execute("approval_probe", {"value": 8}, context=context)
    assert not_matched.error == "requires approval"
