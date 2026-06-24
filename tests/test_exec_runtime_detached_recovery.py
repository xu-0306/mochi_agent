"""Detached exec runtime recovery tests."""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
from pathlib import Path

import pytest

from mochi.runtime.exec_runtime import ExecRuntime
from mochi.runtime.exec_sessions import ExecSessionStatus
from mochi.utils.shell_providers import BaseShellProvider, SubprocessSpec


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


@pytest.mark.asyncio
async def test_detached_session_persists_and_recovers_after_runtime_restart(tmp_path: Path) -> None:
    provider = {"test": _PythonDirectProvider()}
    state_root = tmp_path / "exec-state"
    runtime = ExecRuntime(providers=provider, default_shell="test", state_root=state_root)
    recovered_runtime: ExecRuntime | None = None
    session_id: str | None = None

    try:
        started = await runtime.start_command(
            command="import time; print('detached-ready', flush=True); time.sleep(30)",
            shell="test",
            background=True,
        )
        session_id = started.session_id
        manifest_path = state_root / session_id / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["session_id"] == session_id
        assert manifest["background"] is True
        assert manifest["detached_persisted"] is True

        await runtime.close()

        recovered_runtime = ExecRuntime(providers=provider, default_shell="test", state_root=state_root)
        recovered_sessions = {session.session_id: session for session in recovered_runtime.list_sessions()}
        assert session_id in recovered_sessions
        recovered_session = recovered_sessions[session_id]
        assert recovered_session.detached_persisted is True
        assert recovered_session.recovered is True

        poll = await recovered_runtime.read_session(session_id, yield_time_ms=300)
        assert poll is not None
        assert poll.status == ExecSessionStatus.RUNNING
        assert poll.detached is True
        assert poll.restored is True
        assert poll.supports_stdin is False
        assert "detached-ready" in poll.stdout

        with pytest.raises(RuntimeError, match="stdin is unavailable"):
            await recovered_runtime.write_stdin(session_id, chars="abc\n")

        killed = await recovered_runtime.kill_session(session_id)
        assert killed is not None
        assert killed.status == ExecSessionStatus.KILLED
        assert killed.detached is True
        assert killed.restored is True
        assert killed.supports_stdin is False

        updated_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert updated_manifest["status"] == ExecSessionStatus.KILLED.value
    finally:
        await runtime.close()
        if recovered_runtime is not None:
            if session_id is not None:
                with contextlib.suppress(Exception):
                    await recovered_runtime.kill_session(session_id)
            await recovered_runtime.close()


@pytest.mark.asyncio
async def test_recovered_detached_session_reads_log_tail_after_process_exit(tmp_path: Path) -> None:
    provider = {"test": _PythonDirectProvider()}
    state_root = tmp_path / "exec-state"
    runtime = ExecRuntime(
        providers=provider,
        default_shell="test",
        state_root=state_root,
        output_tail_limit=512,
    )

    started = await runtime.start_command(
        command=(
            "import time; "
            "print('first', flush=True); "
            "time.sleep(0.15); "
            "print('last', flush=True)"
        ),
        shell="test",
        background=True,
    )

    await runtime.close()
    await asyncio.sleep(0.35)

    recovered_runtime = ExecRuntime(
        providers=provider,
        default_shell="test",
        state_root=state_root,
        output_tail_limit=512,
    )
    try:
        poll = await recovered_runtime.read_session(started.session_id, yield_time_ms=120)
        assert poll is not None
        assert poll.status == ExecSessionStatus.COMPLETED
        assert poll.detached is True
        assert poll.restored is True
        assert poll.supports_stdin is False
        assert "first" in poll.stdout
        assert "last" in poll.stdout
    finally:
        await recovered_runtime.close()
