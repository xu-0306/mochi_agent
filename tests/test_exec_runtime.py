"""Exec runtime skeleton tests."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mochi.config.schema import MochiConfig
from mochi.runtime.exec_runtime import ExecRuntime
from mochi.runtime.exec_sessions import ExecSessionStatus
from mochi.runtime.sandbox import SandboxPlanMismatch, SandboxUnavailableError
from mochi.runtime.sandbox.base import HostSandboxBackend
from mochi.runtime.service import RuntimeService
from mochi.utils.shell_providers import (
    BaseShellProvider,
    BashProvider,
    CmdProvider,
    PowerShellProvider,
    SubprocessSpec,
)


class _PythonDirectProvider(BaseShellProvider):
    @property
    def canonical_name(self) -> str:
        return "test"

    @property
    def aliases(self) -> tuple[str, ...]:
        return ("test",)

    def build_subprocess_spec(
        self, command: str, *, tty: bool = False
    ) -> SubprocessSpec:
        del tty
        return SubprocessSpec(executable=sys.executable, args=("-c", command))


def test_shell_provider_command_packaging_shape() -> None:
    ps_spec = PowerShellProvider().build_subprocess_spec("Get-ChildItem", tty=False)
    assert ps_spec.executable == "pwsh"
    assert ps_spec.args[:3] == ("-NoLogo", "-NoProfile", "-NonInteractive")
    assert ps_spec.args[-2:] == ("-Command", "Get-ChildItem")

    bash_non_tty = BashProvider().build_subprocess_spec("echo hi", tty=False)
    assert bash_non_tty.argv == ("bash", "-lc", "echo hi")

    bash_tty = BashProvider().build_subprocess_spec("echo hi", tty=True)
    assert bash_tty.argv == ("bash", "-ic", "echo hi")

    cmd_spec = CmdProvider().build_subprocess_spec("dir", tty=False)
    assert cmd_spec.argv == ("cmd.exe", "/d", "/s", "/c", "dir")


@pytest.mark.asyncio
async def test_exec_runtime_foreground_execution() -> None:
    runtime = ExecRuntime(
        providers={"test": _PythonDirectProvider()},
        default_shell="test",
    )

    result = await runtime.start_command(
        command="import sys; print('hello'); sys.stderr.write('warn\\n')",
        shell="test",
        background=False,
    )
    assert result.status == ExecSessionStatus.COMPLETED
    assert result.exit_code == 0
    assert "hello" in result.stdout
    assert "warn" in result.stderr

    sessions = runtime.list_sessions()
    assert len(sessions) == 1
    assert sessions[0].status == ExecSessionStatus.COMPLETED


@pytest.mark.asyncio
async def test_exec_runtime_background_incremental_read_and_write_stdin() -> None:
    runtime = ExecRuntime(
        providers={"test": _PythonDirectProvider()},
        default_shell="test",
    )
    command = (
        "import sys, time; "
        "print('ready', flush=True); "
        "line = sys.stdin.readline().strip(); "
        "print('echo:' + line, flush=True); "
        "time.sleep(0.15); "
        "print('done', flush=True)"
    )
    started = await runtime.start_command(command=command, background=True)
    assert started.status == ExecSessionStatus.RUNNING

    first = await runtime.read_session(started.session_id, yield_time_ms=120)
    assert first is not None
    assert "ready" in first.stdout

    wrote = await runtime.write_stdin(
        started.session_id, chars="abc\n", yield_time_ms=120
    )
    assert wrote is not None
    assert "echo:abc" in wrote.stdout

    second = await runtime.read_session(started.session_id, yield_time_ms=180)
    assert second is not None
    assert "done" in second.stdout
    assert second.status == ExecSessionStatus.COMPLETED
    assert second.exit_code == 0


@pytest.mark.asyncio
async def test_exec_runtime_inspection_does_not_consume_incremental_output() -> None:
    runtime = ExecRuntime(
        providers={"test": _PythonDirectProvider()},
        default_shell="test",
    )
    started = await runtime.start_command(
        command="import time; print('ready', flush=True); time.sleep(10)",
        background=True,
    )

    inspected = await runtime.inspect_session(started.session_id)
    assert inspected is not None
    assert inspected.status == ExecSessionStatus.RUNNING
    assert inspected.stdout == ""
    assert inspected.stderr == ""

    read = await runtime.read_session(started.session_id, yield_time_ms=120)
    assert read is not None
    assert "ready" in read.stdout
    await runtime.kill_session(started.session_id)


@pytest.mark.asyncio
async def test_exec_runtime_kill_session() -> None:
    runtime = ExecRuntime(
        providers={"test": _PythonDirectProvider()},
        default_shell="test",
    )
    command = "import time; print('alive', flush=True); time.sleep(10)"
    started = await runtime.start_command(command=command, background=True)

    before_kill = await runtime.read_session(started.session_id, yield_time_ms=120)
    assert before_kill is not None
    assert "alive" in before_kill.stdout

    killed = await runtime.kill_session(started.session_id)
    assert killed is not None
    assert killed.status == ExecSessionStatus.KILLED
    assert killed.exit_code is not None


@pytest.mark.asyncio
async def test_exec_runtime_timeout_marks_timed_out() -> None:
    runtime = ExecRuntime(
        providers={"test": _PythonDirectProvider()},
        default_shell="test",
    )

    result = await runtime.start_command(
        command="import time; print('start', flush=True); time.sleep(1)",
        background=False,
        timeout_sec=0.1,
    )
    assert result.status == ExecSessionStatus.TIMED_OUT
    assert result.timed_out is True


@pytest.mark.asyncio
async def test_exec_runtime_close_releases_sessions() -> None:
    runtime = ExecRuntime(
        providers={"test": _PythonDirectProvider()},
        default_shell="test",
    )
    started = await runtime.start_command(
        command="import time; print('alive', flush=True); time.sleep(10)",
        shell="test",
        background=True,
    )
    assert started.status == ExecSessionStatus.RUNNING
    assert runtime.list_sessions()

    await runtime.close()

    assert runtime.list_sessions() == []


@pytest.mark.asyncio
async def test_exec_runtime_revalidates_and_launches_approved_sandbox_plan(
    tmp_path: Path,
) -> None:
    runtime = ExecRuntime(
        providers={"test": _PythonDirectProvider()},
        default_shell="test",
    )
    command = "print('sandbox-plan')"
    plan = runtime.build_sandbox_plan(
        command=command,
        mode="off",
        shell="test",
        cwd=tmp_path,
        env=None,
        timeout_sec=5,
        requested_escalation="use_default",
        workspace_root=tmp_path,
        background=False,
        tty=False,
    )

    result = await runtime.start_command(
        command=command,
        shell="test",
        cwd=tmp_path,
        timeout_sec=5,
        sandbox_plan=plan.to_dict(),
    )

    assert result.status == ExecSessionStatus.COMPLETED
    assert "sandbox-plan" in result.stdout


@pytest.mark.asyncio
async def test_exec_runtime_rejects_tampered_sandbox_replay(tmp_path: Path) -> None:
    runtime = ExecRuntime(
        providers={"test": _PythonDirectProvider()},
        default_shell="test",
    )
    plan = runtime.build_sandbox_plan(
        command="print('approved')",
        mode="off",
        shell="test",
        cwd=tmp_path,
        env=None,
        timeout_sec=5,
        requested_escalation="use_default",
        workspace_root=tmp_path,
        background=False,
        tty=False,
    ).to_dict()
    plan["argv"] = ["-c", "print('tampered')"]

    with pytest.raises(SandboxPlanMismatch):
        await runtime.start_command(
            command="print('approved')",
            shell="test",
            cwd=tmp_path,
            timeout_sec=5,
            sandbox_plan=plan,
        )


@pytest.mark.asyncio
async def test_exec_runtime_rejects_valid_plan_replayed_for_different_command(
    tmp_path: Path,
) -> None:
    runtime = ExecRuntime(
        providers={"test": _PythonDirectProvider()},
        default_shell="test",
    )
    plan = runtime.build_sandbox_plan(
        command="print('approved')",
        mode="off",
        shell="test",
        cwd=tmp_path,
        env=None,
        timeout_sec=5,
        requested_escalation="use_default",
        workspace_root=tmp_path,
        background=False,
        tty=False,
    )

    with pytest.raises(SandboxPlanMismatch, match="no longer matches"):
        await runtime.start_command(
            command="print('different')",
            shell="test",
            cwd=tmp_path,
            timeout_sec=5,
            sandbox_plan=plan,
        )


def test_exec_runtime_required_mode_fails_closed_without_complete_backend(
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

    with pytest.raises(SandboxUnavailableError, match="required_backend_unavailable"):
        runtime.build_sandbox_plan(
            command="print('blocked')",
            mode="required",
            shell="test",
            cwd=tmp_path,
            env=None,
            timeout_sec=5,
            requested_escalation="use_default",
            workspace_root=tmp_path,
            background=False,
            tty=False,
        )


@pytest.mark.parametrize("change_contract_mode", ["observe", "enforce"])
@pytest.mark.parametrize("sandbox_mode", ["off", "preferred", "required"])
def test_file_rollout_and_exec_sandbox_axes_are_orthogonal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    change_contract_mode: str,
    sandbox_mode: str,
) -> None:
    config = MochiConfig.model_validate(
        {
            "workspace_dir": str(tmp_path),
            "security": {
                "change_contract_mode": change_contract_mode,
            },
            "sandbox": {"mode": sandbox_mode},
        }
    )
    runtime = ExecRuntime(
        providers={"test": _PythonDirectProvider()},
        default_shell="test",
    )
    monkeypatch.setattr(
        "mochi.runtime.exec_runtime.select_sandbox_backend",
        lambda mode: HostSandboxBackend(degraded_reason=f"{mode}_backend_unavailable"),
    )

    assert config.security.change_contract_mode == change_contract_mode
    assert config.sandbox.mode == sandbox_mode
    if sandbox_mode == "required":
        with pytest.raises(
            SandboxUnavailableError, match="required_backend_unavailable"
        ):
            runtime.build_sandbox_plan(
                command="print('blocked')",
                mode=config.sandbox.mode,
                shell="test",
                cwd=tmp_path,
                env=None,
                timeout_sec=5,
                requested_escalation="use_default",
                workspace_root=tmp_path,
                background=False,
                tty=False,
            )
    else:
        plan = runtime.build_sandbox_plan(
            command="print('allowed')",
            mode=config.sandbox.mode,
            shell="test",
            cwd=tmp_path,
            env=None,
            timeout_sec=5,
            requested_escalation="use_default",
            workspace_root=tmp_path,
            background=False,
            tty=False,
        )
        assert plan.mode == sandbox_mode


@pytest.mark.asyncio
async def test_approved_exec_replay_uses_serialized_sandbox_plan(
    tmp_path: Path,
) -> None:
    runtime = ExecRuntime(
        providers={"test": _PythonDirectProvider()},
        default_shell="test",
    )
    command = "print('approved-replay')"
    plan = runtime.build_sandbox_plan(
        command=command,
        mode="off",
        shell="test",
        cwd=tmp_path,
        env=None,
        timeout_sec=5,
        requested_escalation="use_default",
        workspace_root=tmp_path,
        background=False,
        tty=False,
    )
    service = object.__new__(RuntimeService)
    service._exec_runtime = runtime
    approval = SimpleNamespace(
        command_payload={
            "command": command,
            "shell": "test",
            "workdir": str(tmp_path),
            "env": None,
            "timeout_sec": 5,
            "background": False,
            "tty": False,
            "approval_state": "approved",
            "sandbox_plan": plan.to_dict(),
        }
    )

    result = await service._execute_approved_exec_request(approval)

    assert result["status"] == "completed"
    assert result["sandbox_plan_digest"] == plan.digest
    assert result["sandbox_backend"] == "host"
    assert "approved-replay" in result["stdout"]
