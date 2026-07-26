"""execute_code 與 mcp_call 工具測試。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from mochi.tools.base import ActiveToolController, ToolExecutionContext
from mochi.tools.execute_code import ExecuteCodeTool
from mochi.tools.execute_code_v2 import ExecuteCodeV2Tool
from mochi.tools.mcp_client import MCPCallTool
from mochi.tools.process_service import ProcessService


def _effective_exec_policy(
    *,
    require_approval: bool,
    hard_denies: list[str] | None = None,
    suffix: str,
) -> dict[str, object]:
    return {
        "policy_snapshot_id": f"policy-{suffix}",
        "policy_version": f"effective-policy-v1:{suffix}",
        "source_chain": ["security_config", "session_override"],
        "autonomy_mode": "strict" if require_approval else "auto_review",
        "require_approval_for_file_write": True,
        "require_approval_for_exec": require_approval,
        "file_read_scope": "workspace",
        "file_write_scope": "workspace",
        "hard_denies": list(hard_denies or []),
    }


def test_cached_execute_code_tool_uses_each_call_policy_snapshot(tmp_path: Path) -> None:
    calls: list[str] = []

    async def fake_runner(
        code: str,
        cwd: Path,
        timeout_sec: int,
        python_executable: str,
    ) -> tuple[int, str, str]:
        del cwd, timeout_sec, python_executable
        calls.append(code)
        return 0, "ok", ""

    tool = ExecuteCodeTool(
        workspace_dir=tmp_path,
        require_approval=True,
        runner=fake_runner,
    )
    allowed = asyncio.run(
        tool.execute(
            code="print('allowed')",
            context=ToolExecutionContext(
                permission_policy=_effective_exec_policy(
                    require_approval=False,
                    suffix="allow",
                )
            ),
        )
    )
    pending = asyncio.run(
        tool.execute(
            code="print('pending')",
            context=ToolExecutionContext(
                permission_policy=_effective_exec_policy(
                    require_approval=True,
                    suffix="approval",
                )
            ),
        )
    )
    denied = asyncio.run(
        tool.execute(
            code="print('denied')",
            approved=True,
            context=ToolExecutionContext(
                permission_policy=_effective_exec_policy(
                    require_approval=False,
                    hard_denies=["tool:execute_code"],
                    suffix="deny",
                )
            ),
        )
    )

    assert allowed.error is None
    assert allowed.metadata["policy_snapshot_id"] == "policy-allow"
    assert pending.metadata["requires_approval"] is True
    assert pending.metadata["policy_snapshot_id"] == "policy-approval"
    assert denied.metadata["security_decision"] == "deny"
    assert denied.metadata["hard_deny"] == "tool:execute_code"
    assert calls == ["print('allowed')"]


def test_execute_code_requires_approval_by_default(tmp_path: Path) -> None:
    """execute_code 預設應要求審批。"""
    tool = ExecuteCodeTool(workspace_dir=tmp_path)

    result = asyncio.run(tool.execute(code="print('hello')"))

    assert result.error is not None
    assert "approval" in result.error.lower()
    assert result.metadata.get("requires_approval") is True
    assert result.metadata.get("security_decision") == "require_approval"
    assert result.metadata.get("approval_scope") == "dangerous_command"
    assert result.metadata.get("policy_source") == "execute_code_policy"


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


def test_execute_code_prefers_task_sandbox_from_context(tmp_path: Path) -> None:
    """execute_code should default cwd to context task sandbox when present."""
    captured: dict[str, Any] = {}
    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir(parents=True, exist_ok=True)

    async def fake_runner(
        code: str,
        cwd: Path,
        timeout_sec: int,
        python_executable: str,
    ) -> tuple[int, str, str]:
        captured["cwd"] = cwd
        return 0, "ok", ""

    tool = ExecuteCodeTool(
        workspace_dir=tmp_path,
        require_approval=False,
        runner=fake_runner,
    )
    result = asyncio.run(
        tool.execute(
            code="print('x')",
            context=ToolExecutionContext(
                workspace_dir=str(tmp_path),
                task_sandbox_dir=str(sandbox_dir),
            ),
        )
    )
    assert result.error is None
    assert captured["cwd"] == sandbox_dir.resolve(strict=False)


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


def test_execute_code_foreground_cancellation_reports_cancelled_status(tmp_path: Path) -> None:
    async def _run() -> None:
        tool = ExecuteCodeTool(workspace_dir=tmp_path, require_approval=False)
        controller = ActiveToolController()
        context = ToolExecutionContext(
            active_tool_controller=controller,
            permission_policy=_effective_exec_policy(
                require_approval=False,
                suffix="cancel",
            ),
        )
        await controller.activate_tool(
            tool_call_id="tool-call-execute-code-cancel",
            tool_name="execute_code",
            cancellable=False,
        )

        task = asyncio.create_task(
            tool.execute(
                code="import time; print('started', flush=True); time.sleep(10)",
                context=context,
            )
        )
        try:
            for _ in range(100):
                snapshot = await controller.snapshot()
                if snapshot["active"] and snapshot["cancellable"]:
                    break
                await asyncio.sleep(0.02)
            else:
                raise AssertionError("controller never observed a cancellable execute_code run")

            cancel_result = await controller.request_cancel()
            assert cancel_result.cancelled is True
            result = await task
        finally:
            if not task.done():
                task.cancel()

        assert result.error is None
        assert result.metadata["status"] == "cancelled"
        assert result.metadata["cancelled"] is True
        assert result.metadata["policy_snapshot_id"] == "policy-cancel"
        assert result.metadata["effective_policy_version"] == "effective-policy-v1:cancel"
        assert isinstance(result.output, str)

    asyncio.run(_run())


def test_execute_code_v2_requires_approval_by_default(tmp_path: Path) -> None:
    tool = ExecuteCodeV2Tool(workspace_dir=tmp_path)

    result = asyncio.run(tool.execute(code="result = 1"))

    assert result.error is not None
    assert "approval" in result.error.lower()


def test_cached_execute_code_v2_tool_uses_each_call_policy_snapshot(tmp_path: Path) -> None:
    calls: list[str] = []

    async def fake_runner(
        code: str,
        cwd: Path,
        timeout_sec: int,
        python_executable: str,
        workspace_dir: Path,
        allowed_tools: list[str],
    ) -> dict[str, Any]:
        del cwd, timeout_sec, python_executable, workspace_dir, allowed_tools
        calls.append(code)
        return {"stdout": "ok", "result": None, "tool_calls": []}

    tool = ExecuteCodeV2Tool(
        workspace_dir=tmp_path,
        require_approval=True,
        runner=fake_runner,
    )

    allowed = asyncio.run(
        tool.execute(
            code="result = 'allowed'",
            context=ToolExecutionContext(
                permission_policy=_effective_exec_policy(
                    require_approval=False,
                    suffix="v2-allow",
                )
            ),
        )
    )
    pending = asyncio.run(
        tool.execute(
            code="result = 'pending'",
            context=ToolExecutionContext(
                permission_policy=_effective_exec_policy(
                    require_approval=True,
                    suffix="v2-approval",
                )
            ),
        )
    )
    denied = asyncio.run(
        tool.execute(
            code="result = 'denied'",
            approved=True,
            context=ToolExecutionContext(
                permission_policy=_effective_exec_policy(
                    require_approval=False,
                    hard_denies=["tool:execute_code_v2"],
                    suffix="v2-deny",
                )
            ),
        )
    )

    assert allowed.error is None
    assert allowed.metadata["policy_snapshot_id"] == "policy-v2-allow"
    assert pending.metadata["requires_approval"] is True
    assert pending.metadata["policy_snapshot_id"] == "policy-v2-approval"
    assert denied.metadata["security_decision"] == "deny"
    assert denied.metadata["hard_deny"] == "tool:execute_code_v2"
    assert denied.metadata["policy_snapshot_id"] == "policy-v2-deny"
    assert calls == ["result = 'allowed'"]


def test_execute_code_v2_foreground_cancellation_reports_cancelled_status(tmp_path: Path) -> None:
    async def _run() -> None:
        tool = ExecuteCodeV2Tool(workspace_dir=tmp_path, require_approval=False)
        controller = ActiveToolController()
        context = ToolExecutionContext(
            active_tool_controller=controller,
            permission_policy=_effective_exec_policy(
                require_approval=False,
                suffix="v2-cancel",
            ),
        )
        await controller.activate_tool(
            tool_call_id="tool-call-execute-code-v2-cancel",
            tool_name="execute_code_v2",
            cancellable=False,
        )

        task = asyncio.create_task(
            tool.execute(
                code="import time; print('started', flush=True); time.sleep(10)",
                context=context,
            )
        )
        try:
            for _ in range(100):
                snapshot = await controller.snapshot()
                if snapshot["active"] and snapshot["cancellable"]:
                    break
                await asyncio.sleep(0.02)
            else:
                raise AssertionError("controller never observed a cancellable execute_code_v2 run")

            cancel_result = await controller.request_cancel()
            assert cancel_result.cancelled is True
            result = await task
        finally:
            if not task.done():
                task.cancel()

        assert result.error is None
        assert result.metadata["status"] == "cancelled"
        assert result.metadata["cancelled"] is True
        assert result.metadata["policy_snapshot_id"] == "policy-v2-cancel"
        assert result.metadata["effective_policy_version"] == "effective-policy-v1:v2-cancel"
        assert result.output["result"] is None
        assert result.output["tool_calls"] == []

    asyncio.run(_run())


def test_execute_code_v2_supports_injected_runner(tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    async def fake_runner(
        code: str,
        cwd: Path,
        timeout_sec: int,
        python_executable: str,
        workspace_dir: Path,
        allowed_tools: list[str],
    ) -> dict[str, Any]:
        captured["code"] = code
        captured["cwd"] = cwd
        captured["timeout_sec"] = timeout_sec
        captured["python_executable"] = python_executable
        captured["workspace_dir"] = workspace_dir
        captured["allowed_tools"] = allowed_tools
        return {"stdout": "", "result": {"ok": True}, "tool_calls": []}

    tool = ExecuteCodeV2Tool(
        workspace_dir=tmp_path,
        require_approval=False,
        runner=fake_runner,
    )

    result = asyncio.run(
        tool.execute(
            code="result = {'ok': True}",
            timeout_sec=9,
            allowed_tools=["file_read", "glob_search"],
        )
    )

    assert result.error is None
    assert result.output["result"] == {"ok": True}
    assert captured["timeout_sec"] == 9
    assert captured["workspace_dir"] == tmp_path.resolve(strict=False)
    assert captured["allowed_tools"] == ["file_read", "glob_search"]


def test_execute_code_v2_default_runner_can_call_tool_helpers(tmp_path: Path) -> None:
    target = tmp_path / "note.txt"
    target.write_bytes(b"hello from helper\n")
    tool = ExecuteCodeV2Tool(workspace_dir=tmp_path, require_approval=False)

    result = asyncio.run(
        tool.execute(
            code="result = file_read(path='note.txt', line_numbers=False)",
            allowed_tools=["file_read"],
        )
    )

    assert result.error is None
    assert result.output["result"] == "hello from helper\n"
    assert result.output["tool_calls"][0]["tool_name"] == "file_read"
    assert result.output["tool_calls"][0]["ok"] is True


def test_execute_code_v2_rejects_unknown_allowed_tools(tmp_path: Path) -> None:
    tool = ExecuteCodeV2Tool(workspace_dir=tmp_path, require_approval=False)

    result = asyncio.run(tool.execute(code="result = 1", allowed_tools=["exec_command"]))

    assert result.error is not None
    assert "allowed_tools" in result.error


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
