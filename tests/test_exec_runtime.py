"""Exec runtime skeleton tests."""

from __future__ import annotations

import sys

import pytest

from mochi.runtime.exec_runtime import ExecRuntime
from mochi.runtime.exec_sessions import ExecSessionStatus
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

    def build_subprocess_spec(self, command: str, *, tty: bool = False) -> SubprocessSpec:
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

    wrote = await runtime.write_stdin(started.session_id, chars="abc\n", yield_time_ms=120)
    assert wrote is not None
    assert "echo:abc" in wrote.stdout

    second = await runtime.read_session(started.session_id, yield_time_ms=180)
    assert second is not None
    assert "done" in second.stdout
    assert second.status == ExecSessionStatus.COMPLETED
    assert second.exit_code == 0


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
