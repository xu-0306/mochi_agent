"""Exec tool family tests for Task 3."""

from __future__ import annotations

import sys

import pytest

from mochi.runtime.approvals import InMemoryApprovalStore
from mochi.runtime.exec_runtime import ExecRuntime
from mochi.tools.exec_command import ExecCommandTool
from mochi.tools.kill_session import KillSessionTool
from mochi.tools.list_sessions import ListSessionsTool
from mochi.tools.read_session import ReadSessionTool
from mochi.tools.write_stdin import WriteStdinTool
from mochi.utils.shell_providers import BaseShellProvider, SubprocessSpec


def _allow_rule(*tokens: str, shells: list[str] | None = None) -> dict[str, object]:
    return {
        "tokens": list(tokens),
        "decision": "allow",
        "match": "exact",
        "shells": list(shells or []),
    }


class _PythonDirectProvider(BaseShellProvider):
    @property
    def canonical_name(self) -> str:
        return "test"

    @property
    def aliases(self) -> tuple[str, ...]:
        return ("test",)

    def build_subprocess_spec(self, command: str, *, tty: bool = False) -> SubprocessSpec:
        del tty
        script = _SCRIPT_BY_COMMAND.get(command)
        if script is None:
            raise ValueError(f"Unsupported test command: {command}")
        return SubprocessSpec(executable=sys.executable, args=("-c", script))


_SCRIPT_BY_COMMAND = {
    "fg": "import sys;print('hello');sys.stderr.write('warn\\n')",
    "bg": "import time;print('started', flush=True);time.sleep(5)",
    "interactive": (
        "import sys,time;"
        "print('ready', flush=True);"
        "line=sys.stdin.readline().strip();"
        "print('echo:'+line, flush=True);"
        "time.sleep(5)"
    ),
}


@pytest.mark.asyncio
async def test_exec_command_foreground_success() -> None:
    runtime = ExecRuntime(
        providers={"test": _PythonDirectProvider()},
        default_shell="test",
    )
    tool = ExecCommandTool(
        runtime=runtime,
        require_approval=False,
        workspace_dir="H:/_python/agent_mochi",
        command_rules=[_allow_rule("fg", shells=["test"])],
    )

    result = await tool.execute(
        command="fg",
        shell="test",
    )

    assert result.error is None
    assert result.metadata["status"] == "completed"
    assert result.metadata["timed_out"] is False
    assert result.metadata["approval_id"] is None
    assert result.metadata["policy_state"] == "allow"
    assert result.metadata["policy_reason"] == "Command is allowed by a persisted command rule."
    assert result.metadata["rule_id"] == "persisted_command_rule"
    assert result.metadata["suggested_rule"]["tokens"] == ["fg"]
    assert isinstance(result.output, dict)
    assert "hello" in result.output["stdout"]
    assert "warn" in result.output["stderr"]


@pytest.mark.asyncio
async def test_exec_command_background_returns_session_id() -> None:
    runtime = ExecRuntime(
        providers={"test": _PythonDirectProvider()},
        default_shell="test",
    )
    tool = ExecCommandTool(
        runtime=runtime,
        require_approval=False,
        workspace_dir="H:/_python/agent_mochi",
        command_rules=[_allow_rule("bg", shells=["test"])],
    )

    result = await tool.execute(
        command="bg",
        shell="test",
        background=True,
    )

    assert result.error is None
    assert result.metadata["status"] == "running"
    assert isinstance(result.metadata["session_id"], str)
    await runtime.kill_session(result.metadata["session_id"])


@pytest.mark.asyncio
async def test_exec_command_returns_approval_pending_metadata() -> None:
    runtime = ExecRuntime(
        providers={"test": _PythonDirectProvider()},
        default_shell="test",
    )
    approvals = InMemoryApprovalStore()
    tool = ExecCommandTool(
        runtime=runtime,
        approval_store=approvals,
        require_approval=False,
        workspace_dir="H:/_python/agent_mochi",
    )

    result = await tool.execute(command="cmd /c more notes.txt", shell="cmd")

    assert result.error is not None
    assert result.metadata["status"] == "approval_pending"
    assert result.metadata["requires_approval"] is True
    assert result.metadata["policy_state"] == "ask"
    assert "requires approval" in result.metadata["policy_reason"]
    assert result.metadata["rule_id"] == "cmd_c_requires_approval"
    assert result.metadata["suggested_rule"]["tokens"] == ["cmd", "/c", "more", "notes.txt"]
    approval_id = result.metadata["approval_id"]
    assert isinstance(approval_id, str)
    stored = approvals.get(approval_id)
    assert stored is not None
    assert stored.status == "pending"


@pytest.mark.asyncio
async def test_exec_command_does_not_use_legacy_shell_allowlist_for_primary_path() -> None:
    runtime = ExecRuntime(
        providers={"test": _PythonDirectProvider()},
        default_shell="test",
    )
    tool = ExecCommandTool(
        runtime=runtime,
        require_approval=False,
        workspace_dir="H:/_python/agent_mochi",
        command_rules=[_allow_rule("fg", shells=["test"])],
    )

    result = await tool.execute(command="fg", shell="test")

    assert result.error is None
    assert result.metadata["policy_state"] == "allow"
    assert result.metadata["policy_reason"] == "Command is allowed by a persisted command rule."


@pytest.mark.asyncio
async def test_write_read_kill_and_list_delegate_to_runtime() -> None:
    runtime = ExecRuntime(
        providers={"test": _PythonDirectProvider()},
        default_shell="test",
    )
    exec_tool = ExecCommandTool(
        runtime=runtime,
        require_approval=False,
        workspace_dir="H:/_python/agent_mochi",
        command_rules=[_allow_rule("interactive", shells=["test"])],
    )
    write_tool = WriteStdinTool(runtime=runtime)
    read_tool = ReadSessionTool(runtime=runtime)
    kill_tool = KillSessionTool(runtime=runtime)
    list_tool = ListSessionsTool(runtime=runtime)

    started = await exec_tool.execute(
        command="interactive",
        shell="test",
        background=True,
    )
    assert started.error is None
    session_id = started.metadata["session_id"]

    first = await read_tool.execute(session_id=session_id, yield_time_ms=120)
    assert first.error is None
    assert "ready" in first.output["stdout"]

    wrote = await write_tool.execute(session_id=session_id, chars="abc\n", yield_time_ms=120)
    assert wrote.error is None
    assert "echo:abc" in wrote.output["stdout"]

    listed = await list_tool.execute()
    assert listed.error is None
    assert any(item["session_id"] == session_id for item in listed.output)

    killed = await kill_tool.execute(session_id=session_id)
    assert killed.error is None
    assert killed.metadata["status"] == "killed"
