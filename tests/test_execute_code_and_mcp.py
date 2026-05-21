"""execute_code 與 mcp_call 工具測試。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from mochi.tools.execute_code import ExecuteCodeTool
from mochi.tools.mcp_client import MCPCallTool
from mochi.tools.process_service import ProcessService


def test_execute_code_requires_approval_by_default(tmp_path: Path) -> None:
    """execute_code 預設應要求審批。"""
    tool = ExecuteCodeTool(workspace_dir=tmp_path)

    result = asyncio.run(tool.execute(code="print('hello')"))

    assert result.error is not None
    assert "approval" in result.error.lower()
    assert result.metadata.get("requires_approval") is True


def test_execute_code_supports_injected_runner(tmp_path: Path) -> None:
    """execute_code 應可注入 runner。"""
    captured: dict[str, Any] = {}
    workdir = tmp_path / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    async def fake_runner(
        code: str,
        cwd: Path,
        timeout_sec: int,
        python_executable: str,
    ) -> tuple[int, str, str]:
        captured["code"] = code
        captured["cwd"] = cwd
        captured["timeout_sec"] = timeout_sec
        captured["python_executable"] = python_executable
        return 0, "ok", ""

    tool = ExecuteCodeTool(
        workspace_dir=tmp_path,
        require_approval=False,
        runner=fake_runner,
    )

    result = asyncio.run(
        tool.execute(
            code="print('injected')",
            cwd="work",
            timeout_sec=7,
        )
    )

    assert result.error is None
    assert result.output == "ok"
    assert captured["code"] == "print('injected')"
    assert captured["cwd"] == workdir.resolve(strict=False)
    assert captured["timeout_sec"] == 7


def test_execute_code_rejects_path_outside_workspace(tmp_path: Path) -> None:
    """execute_code 應拒絕 workspace 外路徑。"""
    tool = ExecuteCodeTool(workspace_dir=tmp_path, require_approval=False)
    outside = tmp_path.parent

    result = asyncio.run(tool.execute(code="print('x')", cwd=str(outside)))

    assert result.error is not None
    assert "outside workspace" in result.error.lower()


def test_execute_code_default_runner_executes_python(tmp_path: Path) -> None:
    """預設 runner 應可執行 Python 程式碼。"""
    tool = ExecuteCodeTool(workspace_dir=tmp_path, require_approval=False)

    result = asyncio.run(tool.execute(code="print('hello from execute_code')"))

    assert result.error is None
    assert "hello from execute_code" in str(result.output)


def test_execute_code_background_returns_process_metadata(tmp_path: Path) -> None:
    """execute_code should return running process metadata in background mode."""
    async def _run() -> None:
        service = ProcessService()
        tool = ExecuteCodeTool(
            workspace_dir=tmp_path,
            require_approval=False,
            process_service=service,
        )
        result = await tool.execute(
            code="import time; time.sleep(5)",
            background=True,
            process_label="bg-python",
        )
        assert result.error is None
        assert result.metadata["background"] is True
        assert result.metadata["status"] == "running"
        assert result.metadata["process_id"].startswith("proc-")
        assert result.metadata["label"] == "bg-python"
        stopped = await service.stop(result.metadata["process_id"])
        assert stopped is not None

    asyncio.run(_run())


def test_mcp_call_without_injected_backend_returns_error() -> None:
    """未注入 caller/adapter 時應回傳可預期錯誤。"""
    tool = MCPCallTool()

    result = asyncio.run(tool.execute(server="local", tool="search", arguments={"q": "mochi"}))

    assert result.error is not None
    assert "not configured" in result.error.lower()


def test_mcp_call_supports_sync_callable() -> None:
    """mcp_call 應支援同步 callable。"""

    def fake_caller(server: str, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return {"server": server, "tool": tool, "arguments": arguments, "ok": True}

    tool = MCPCallTool(caller=fake_caller)

    result = asyncio.run(tool.execute(server="local", tool="search", arguments={"q": "mochi"}))

    assert result.error is None
    assert result.output == {
        "server": "local",
        "tool": "search",
        "arguments": {"q": "mochi"},
        "ok": True,
    }


def test_mcp_call_supports_async_adapter() -> None:
    """mcp_call 應支援 async adapter。"""

    class FakeAdapter:
        async def call(self, server: str, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
            return {"source": "adapter", "server": server, "tool": tool, "arguments": arguments}

    tool = MCPCallTool(adapter=FakeAdapter())

    result = asyncio.run(tool.execute(server="svc", tool="ping", arguments={"n": 1}))

    assert result.error is None
    assert result.output == {
        "source": "adapter",
        "server": "svc",
        "tool": "ping",
        "arguments": {"n": 1},
    }
