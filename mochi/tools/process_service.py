"""In-memory background process runtime for tool execution."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import count
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class ManagedProcess:
    process_id: str
    label: str | None
    command: str
    cwd: str
    mode: str
    process: asyncio.subprocess.Process
    created_at: str = field(default_factory=_utc_now_iso)
    stopped: bool = False
    stopped_at: str | None = None
    stop_signal: str | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    stdout_task: asyncio.Task[None] | None = None
    stderr_task: asyncio.Task[None] | None = None


_TAIL_LIMIT = 8000


class ProcessService:
    """Tracks and controls background subprocesses in-memory."""

    def __init__(self) -> None:
        self._seq = count(1)
        self._processes: dict[str, ManagedProcess] = {}

    async def start_shell(
        self,
        *,
        command: str,
        cwd: Path,
        label: str | None = None,
    ) -> dict[str, Any]:
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        managed = self._register(
            command=command,
            cwd=cwd,
            label=label,
            mode="shell",
            process=process,
        )
        return self._status_payload(managed)

    async def start_python(
        self,
        *,
        code: str,
        cwd: Path,
        python_executable: str,
        label: str | None = None,
    ) -> dict[str, Any]:
        process = await asyncio.create_subprocess_exec(
            python_executable,
            "-I",
            "-c",
            code,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        managed = self._register(
            command=code,
            cwd=cwd,
            label=label,
            mode="python",
            process=process,
        )
        return self._status_payload(managed)

    async def poll(self, process_id: str) -> dict[str, Any] | None:
        managed = self._processes.get(process_id)
        if managed is None:
            return None
        self._refresh_state(managed)
        if managed.process.returncode is not None:
            await self._await_drain_tasks(managed)
        return self._status_payload(managed)

    async def stop(self, process_id: str) -> dict[str, Any] | None:
        managed = self._processes.get(process_id)
        if managed is None:
            return None

        self._refresh_state(managed)
        if managed.process.returncode is None:
            managed.process.terminate()
            try:
                await asyncio.wait_for(managed.process.wait(), timeout=2)
                managed.stop_signal = "terminate"
            except TimeoutError:
                managed.process.kill()
                await managed.process.wait()
                managed.stop_signal = "kill"

        self._refresh_state(managed)
        await self._await_drain_tasks(managed)
        managed.stopped = True
        managed.stopped_at = managed.stopped_at or _utc_now_iso()
        return self._status_payload(managed)

    def _register(
        self,
        *,
        command: str,
        cwd: Path,
        label: str | None,
        mode: str,
        process: asyncio.subprocess.Process,
    ) -> ManagedProcess:
        process_id = f"proc-{next(self._seq)}"
        managed = ManagedProcess(
            process_id=process_id,
            label=label,
            command=command,
            cwd=str(cwd),
            mode=mode,
            process=process,
        )
        if process.stdout is not None:
            managed.stdout_task = asyncio.create_task(
                self._drain_stream(managed, process.stdout, "stdout")
            )
        if process.stderr is not None:
            managed.stderr_task = asyncio.create_task(
                self._drain_stream(managed, process.stderr, "stderr")
            )
        self._processes[process_id] = managed
        return managed

    def _refresh_state(self, managed: ManagedProcess) -> None:
        if managed.process.returncode is not None and managed.stopped_at is None:
            managed.stopped_at = _utc_now_iso()

    def _status_payload(self, managed: ManagedProcess) -> dict[str, Any]:
        returncode = managed.process.returncode
        status = "running" if returncode is None else "exited"
        return {
            "process_id": managed.process_id,
            "pid": managed.process.pid,
            "status": status,
            "returncode": returncode,
            "background": True,
            "mode": managed.mode,
            "cwd": managed.cwd,
            "label": managed.label,
            "created_at": managed.created_at,
            "stopped": managed.stopped,
            "stopped_at": managed.stopped_at,
            "stop_signal": managed.stop_signal,
            "stdout_tail": managed.stdout_tail,
            "stderr_tail": managed.stderr_tail,
        }

    async def _await_drain_tasks(self, managed: ManagedProcess) -> None:
        tasks = [task for task in (managed.stdout_task, managed.stderr_task) if task is not None]
        if not tasks:
            return
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _drain_stream(
        self,
        managed: ManagedProcess,
        stream: asyncio.StreamReader,
        target: str,
    ) -> None:
        while True:
            chunk = await stream.read(1024)
            if not chunk:
                return
            text = chunk.decode("utf-8", errors="replace")
            if target == "stdout":
                managed.stdout_tail = _append_tail(managed.stdout_tail, text)
            else:
                managed.stderr_tail = _append_tail(managed.stderr_tail, text)


def _append_tail(existing: str, chunk: str) -> str:
    combined = existing + chunk
    if len(combined) <= _TAIL_LIMIT:
        return combined
    return combined[-_TAIL_LIMIT:]
