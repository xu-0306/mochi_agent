"""Exec tool family tests for Task 3."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from mochi.runtime.approvals import (
    APPROVAL_OWNER_TASK_ID_KEY,
    InMemoryApprovalStore,
)
from mochi.runtime.exec_runtime import ExecRuntime
from mochi.runtime.sandbox import SandboxPlan
from mochi.runtime.sandbox.base import HostSandboxBackend
from mochi.tools.base import ActiveToolController, ToolExecutionContext
from mochi.tools.exec_command import ExecCommandTool
from mochi.tools.kill_session import KillSessionTool
from mochi.tools.list_sessions import ListSessionsTool
from mochi.tools.read_session import ReadSessionTool
from mochi.tools.write_stdin import WriteStdinTool
from mochi.utils.shell_providers import BaseShellProvider, CmdProvider, SubprocessSpec


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
    "slow": "import time;print('started', flush=True);time.sleep(10)",
    "interactive": (
        "import sys,time;"
        "print('ready', flush=True);"
        "line=sys.stdin.readline().strip();"
        "print('echo:'+line, flush=True);"
        "time.sleep(5)"
    ),
}


def _effective_exec_policy(
    *,
    require_approval: bool,
    hard_denies: list[str] | None = None,
    snapshot_id: str = "policy-test-exec",
    policy_version: str = "effective-policy-v1:test-exec",
) -> dict[str, object]:
    return {
        "policy_snapshot_id": snapshot_id,
        "policy_version": policy_version,
        "source_chain": ["security_config", "session_override"],
        "autonomy_mode": "strict" if require_approval else "auto_review",
        "require_approval_for_file_write": True,
        "require_approval_for_exec": require_approval,
        "file_read_scope": "workspace",
        "file_write_scope": "workspace",
        "hard_denies": list(hard_denies or []),
    }


@pytest.mark.asyncio
async def test_exec_command_hard_deny_prevents_execution() -> None:
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
        context=ToolExecutionContext(
            permission_policy=_effective_exec_policy(
                require_approval=False,
                hard_denies=["exec"],
            )
        ),
    )

    assert result.error is not None
    assert result.metadata["status"] == "denied"
    assert result.metadata["hard_deny"] == "exec"
    assert result.metadata["policy_snapshot_id"] == "policy-test-exec"
    assert result.metadata["effective_policy_version"] == "effective-policy-v1:test-exec"
    assert runtime.list_sessions() == []


@pytest.mark.asyncio
async def test_cached_exec_command_uses_each_call_policy_snapshot(tmp_path: Path) -> None:
    runtime = ExecRuntime(
        providers={"test": _PythonDirectProvider()},
        default_shell="test",
    )
    approvals = InMemoryApprovalStore()
    tool = ExecCommandTool(
        runtime=runtime,
        approval_store=approvals,
        require_approval=True,
        workspace_dir=tmp_path,
        command_rules=[_allow_rule("fg", shells=["test"])],
    )

    allowed = await tool.execute(
        command="fg",
        shell="test",
        context=ToolExecutionContext(
            permission_policy=_effective_exec_policy(
                require_approval=False,
                snapshot_id="policy-exec-allow",
                policy_version="effective-policy-v1:exec-allow",
            )
        ),
    )
    pending = await tool.execute(
        command="fg",
        shell="test",
        context=ToolExecutionContext(
            permission_policy=_effective_exec_policy(
                require_approval=True,
                snapshot_id="policy-exec-approval",
                policy_version="effective-policy-v1:exec-approval",
            )
        ),
    )
    sessions_before_deny = list(runtime.list_sessions())
    approvals_before_deny = approvals.list(status="pending")
    denied = await tool.execute(
        command="fg",
        shell="test",
        context=ToolExecutionContext(
            permission_policy=_effective_exec_policy(
                require_approval=False,
                hard_denies=["exec"],
                snapshot_id="policy-exec-deny",
                policy_version="effective-policy-v1:exec-deny",
            )
        ),
    )

    assert allowed.error is None
    assert allowed.metadata["policy_snapshot_id"] == "policy-exec-allow"
    assert pending.metadata["requires_approval"] is True
    assert pending.metadata["policy_snapshot_id"] == "policy-exec-approval"
    assert denied.metadata["status"] == "denied"
    assert denied.metadata["security_decision"] == "deny"
    assert denied.metadata["hard_deny"] == "exec"
    assert denied.metadata["policy_snapshot_id"] == "policy-exec-deny"
    assert runtime.list_sessions() == sessions_before_deny
    assert approvals.list(status="pending") == approvals_before_deny


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
async def test_exec_command_foreground_cancellation_reports_cancelled_status() -> None:
    runtime = ExecRuntime(
        providers={"test": _PythonDirectProvider()},
        default_shell="test",
    )
    tool = ExecCommandTool(
        runtime=runtime,
        require_approval=False,
        workspace_dir="H:/_python/agent_mochi",
        command_rules=[_allow_rule("slow", shells=["test"])],
    )
    controller = ActiveToolController()
    context = ToolExecutionContext(active_tool_controller=controller)
    await controller.activate_tool(
        tool_call_id="tool-call-cancelled",
        tool_name="exec_command",
        cancellable=False,
    )

    async def _run_tool() -> object:
        return await tool.execute(
            command="slow",
            shell="test",
            context=context,
        )

    task = asyncio.create_task(_run_tool())
    try:
        for _ in range(100):
            snapshot = await controller.snapshot()
            if snapshot["active"] and snapshot["cancellable"]:
                break
            await asyncio.sleep(0.02)
        else:
            raise AssertionError("controller never observed a cancellable active exec session")

        cancel_result = await controller.request_cancel()
        assert cancel_result.cancelled is True
        result = await task
    finally:
        if not task.done():
            task.cancel()
        await runtime.close()

    assert result.error is None
    assert result.metadata["status"] == "cancelled"
    assert result.metadata["runtime_status"] == "killed"
    assert result.metadata["cancelled"] is True


@pytest.mark.asyncio
async def test_exec_command_returns_approval_pending_metadata() -> None:
    runtime = ExecRuntime(
        providers={"test": _PythonDirectProvider(), "cmd": CmdProvider()},
        default_shell="test",
    )
    approvals = InMemoryApprovalStore()
    tool = ExecCommandTool(
        runtime=runtime,
        approval_store=approvals,
        require_approval=False,
        workspace_dir="H:/_python/agent_mochi",
    )

    result = await tool.execute(
        command="cmd /c more notes.txt",
        shell="cmd",
        context=ToolExecutionContext(
            permission_policy=_effective_exec_policy(
                require_approval=True,
                snapshot_id="policy-approval",
                policy_version="effective-policy-v1:approval",
            )
        ),
    )

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
    assert stored.requester_id == "runtime-service"
    assert len(stored.request_digest) == 64
    assert len(stored.context_digest) == 64
    assert set(stored.request_digest) <= set("0123456789abcdef")
    assert set(stored.context_digest) <= set("0123456789abcdef")
    assert isinstance(stored.command_payload, dict)
    assert stored.metadata["tool_name"] == "exec_command"
    assert stored.metadata["policy_snapshot_id"] == "policy-approval"
    assert stored.metadata["effective_policy_version"] == "effective-policy-v1:approval"
    assert stored.command_payload["policy_snapshot_id"] == "policy-approval"
    assert stored.command_payload["effective_policy_version"] == "effective-policy-v1:approval"
    sandbox_plan = stored.command_payload.get("sandbox_plan")
    assert isinstance(sandbox_plan, dict)
    assert SandboxPlan.from_dict(sandbox_plan).digest == sandbox_plan["plan_digest"]


@pytest.mark.asyncio
async def test_exec_command_preferred_mode_reports_host_degradation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = ExecRuntime(
        providers={"test": _PythonDirectProvider()},
        default_shell="test",
    )
    monkeypatch.setattr(
        "mochi.runtime.exec_runtime.select_sandbox_backend",
        lambda mode: HostSandboxBackend(degraded_reason=f"{mode}_backend_unavailable"),
    )
    tool = ExecCommandTool(
        runtime=runtime,
        workspace_dir=tmp_path,
        command_rules=[_allow_rule("fg", shells=["test"])],
        sandbox_mode="preferred",
    )

    result = await tool.execute(command="fg", shell="test")

    assert result.error is None
    assert result.metadata["sandbox_backend"] == "host"
    assert result.metadata["sandbox_degraded"] is True
    assert result.metadata["sandbox_degraded_reason"] == "preferred_backend_unavailable"


@pytest.mark.asyncio
async def test_exec_command_required_mode_blocks_when_backend_is_incomplete(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = ExecRuntime(
        providers={"test": _PythonDirectProvider()},
        default_shell="test",
    )
    monkeypatch.setattr(
        "mochi.runtime.exec_runtime.select_sandbox_backend",
        lambda mode: HostSandboxBackend(degraded_reason=f"{mode}_backend_unavailable"),
    )
    tool = ExecCommandTool(
        runtime=runtime,
        workspace_dir=tmp_path,
        command_rules=[_allow_rule("fg", shells=["test"])],
        sandbox_mode="required",
    )

    result = await tool.execute(command="fg", shell="test")

    assert result.error is not None
    assert result.metadata["status"] == "denied"
    assert result.metadata["sandbox_enforcement_unavailable"] is True


@pytest.mark.asyncio
async def test_exec_command_auto_review_allows_policy_ask_without_manual_approval() -> None:
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

    result = await tool.execute(
        command="fg",
        shell="test",
        context=ToolExecutionContext(
            permission_policy={
                "autonomy_mode": "auto_review",
                "require_approval_for_exec": False,
            }
        ),
    )

    assert result.error is None
    assert result.metadata["status"] == "completed"
    assert result.metadata["policy_state"] == "allow"
    assert result.metadata["auto_reviewed_policy_ask"] is True
    assert result.metadata["auto_review_decision"] == "allow"
    assert result.metadata["auto_review_source"] == "reviewed_allow"
    assert result.metadata["auto_review_risk_factors"] == []
    assert result.metadata["auto_review_reason_codes"] == ["reviewed_allow"]
    assert result.metadata["auto_review_reviewer_version"] == "deterministic-v1"
    assert len(result.metadata["auto_review_input_digest"]) == 64
    assert result.metadata["auto_review_execution_verified"] is True
    assert result.metadata["approval_id"] is None
    assert approvals.list(status="pending") == []
    assert isinstance(result.output, dict)
    assert "hello" in result.output["stdout"]


@pytest.mark.asyncio
async def test_exec_command_auto_review_still_requests_manual_approval_for_escalation() -> None:
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

    result = await tool.execute(
        command="fg",
        shell="test",
        sandbox_permissions="require_escalated",
        context=ToolExecutionContext(
            permission_policy={
                "autonomy_mode": "auto_review",
                "require_approval_for_exec": False,
                APPROVAL_OWNER_TASK_ID_KEY: "runtime-task-1",
            }
        ),
    )

    assert result.error is not None
    assert result.metadata["status"] == "approval_pending"
    assert result.metadata["requires_approval"] is True
    assert result.metadata["approval_id"] is not None
    pending = approvals.list(status="pending")
    assert pending != []
    assert pending[0].command_payload is not None
    assert pending[0].command_payload["sandbox_permissions"] == "require_escalated"
    assert pending[0].metadata[APPROVAL_OWNER_TASK_ID_KEY] == "runtime-task-1"
    assert pending[0].metadata["auto_review_decision"] == "require_approval"
    assert pending[0].metadata["auto_review_risk_factors"] == ["require_escalated"]
    assert pending[0].metadata["auto_review_reviewer_version"] == "deterministic-v1"


@pytest.mark.asyncio
async def test_exec_command_auto_review_requires_approval_for_network_credential_exposure() -> None:
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
        allowed_env_vars=["API_TOKEN"],
    )

    result = await tool.execute(
        command="fg",
        shell="test",
        env={"API_TOKEN": "known-secret"},
        context=ToolExecutionContext(
            session_id="session-1",
            permission_policy={
                "autonomy_mode": "auto_review",
                "require_approval_for_exec": False,
            },
        ),
    )

    assert result.error == "Exec command requires approval."
    assert result.metadata["auto_review_decision"] == "require_approval"
    assert result.metadata["auto_review_risk_factors"] == [
        "network_credential_exposure"
    ]
    pending = approvals.list(status="pending")
    assert len(pending) == 1
    assert pending[0].metadata["auto_review_input_digest"] == result.metadata[
        "auto_review_input_digest"
    ]


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
