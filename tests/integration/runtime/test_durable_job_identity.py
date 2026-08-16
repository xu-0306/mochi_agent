"""Integration coverage for durable detached-process identity recovery."""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

import pytest

from mochi.runtime.exec_runtime import ExecRuntime
from mochi.runtime.exec_sessions import ExecSessionStatus
from mochi.runtime.process_identity import ProcessVerification
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


def _manifest_payload(*, manifest_version: int, identity: dict[str, object] | None) -> dict[str, object]:
    payload: dict[str, object] = {
        "manifest_version": manifest_version,
        "session_id": "exec-1",
        "shell": "test",
        "command": "import time; time.sleep(30)",
        "cwd": str(Path.cwd()),
        "pid": 12345,
        "status": "running",
        "background": True,
        "tty": False,
        "started_at": "2026-08-06T00:00:00+00:00",
        "last_activity_at": "2026-08-06T00:00:00+00:00",
        "exit_code": None,
        "timed_out": False,
        "approval_state": "not_required",
        "log_path": None,
        "checkpoint_dir": None,
        "detached_persisted": True,
    }
    if manifest_version == 2:
        payload["durable_process_identity"] = identity
        payload["identity_state"] = "verified"
    return payload


def _write_manifest(
    state_root: Path,
    *,
    manifest_version: int,
    identity: dict[str, object] | None,
) -> Path:
    manifest_path = state_root / "exec-1" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(_manifest_payload(manifest_version=manifest_version, identity=identity)),
        encoding="utf-8",
    )
    return manifest_path


@pytest.mark.asyncio
async def test_legacy_manifest_is_migrated_to_orphaned_and_cannot_be_terminated(tmp_path: Path) -> None:
    state_root = tmp_path / "exec-state"
    manifest_path = _write_manifest(state_root, manifest_version=1, identity=None)
    runtime = ExecRuntime(providers={"test": _PythonDirectProvider()}, state_root=state_root)
    try:
        poll = await runtime.inspect_session("exec-1")
        assert poll is not None
        assert poll.status is ExecSessionStatus.ORPHANED
        killed = await runtime.kill_session("exec-1")
        assert killed is not None
        assert killed.status is ExecSessionStatus.ORPHANED

        migrated = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert migrated["manifest_version"] == 2
        assert migrated["identity_state"] == "legacy"
        assert migrated["status"] == ExecSessionStatus.ORPHANED.value
    finally:
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("verification_state", ["mismatch", "unknown"])
async def test_recycled_pid_or_query_failure_is_never_reattached_or_terminated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    verification_state: str,
) -> None:
    state_root = tmp_path / "exec-state"
    _write_manifest(
        state_root,
        manifest_version=2,
        identity={
            "platform": "windows",
            "pid": 12345,
            "windows_process_creation_filetime": 99,
            "linux_boot_id": None,
            "linux_proc_start_time": None,
        },
    )
    monkeypatch.setattr(
        "mochi.runtime.exec_runtime.open_verified_process",
        lambda _: ProcessVerification(verification_state, None, verification_state),
    )
    runtime = ExecRuntime(providers={"test": _PythonDirectProvider()}, state_root=state_root)
    try:
        session = runtime.list_sessions()[0]
        assert session.status is ExecSessionStatus.ORPHANED
        assert session.verified_process_handle is None
        killed = await runtime.kill_session(session.session_id)
        assert killed is not None
        assert killed.status is ExecSessionStatus.ORPHANED
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_verified_detached_identity_is_persisted_as_v2_and_terminated_by_handle(
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "exec-state"
    provider = {"test": _PythonDirectProvider()}
    runtime = ExecRuntime(providers=provider, default_shell="test", state_root=state_root)
    recovered_runtime: ExecRuntime | None = None
    session_id: str | None = None
    try:
        started = await runtime.start_command(
            command="import time; time.sleep(30)",
            shell="test",
            background=True,
        )
        session_id = started.session_id
        manifest_path = state_root / session_id / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["manifest_version"] == 2
        assert manifest["identity_state"] == "verified"
        assert manifest["durable_process_identity"]["pid"] == started.pid

        await runtime.close()
        recovered_runtime = ExecRuntime(providers=provider, default_shell="test", state_root=state_root)
        recovered = recovered_runtime.list_sessions()[0]
        assert recovered.status is ExecSessionStatus.RUNNING
        assert recovered.verified_process_handle is not None

        killed = await recovered_runtime.kill_session(session_id)
        assert killed is not None
        assert killed.status is ExecSessionStatus.KILLED
        assert recovered.verified_process_handle is None
    finally:
        await runtime.close()
        if recovered_runtime is not None:
            if session_id is not None:
                with contextlib.suppress(Exception):
                    await recovered_runtime.kill_session(session_id)
            await recovered_runtime.close()
