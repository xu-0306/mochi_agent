"""Exec runtime service for foreground/background shell execution."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import subprocess
import sys
from collections.abc import Awaitable, Callable
from itertools import count
from pathlib import Path

from mochi.runtime.exec_sessions import (
    ExecSession,
    ExecSessionSnapshot,
    ExecSessionStatus,
    SessionPollResult,
    utc_now,
)
from mochi.utils.shell_providers import BaseShellProvider, default_shell_providers

ProcessLauncher = Callable[..., Awaitable[asyncio.subprocess.Process]]


class ExecRuntime:
    """Manage foreground exec sessions and file-backed detached sessions."""

    def __init__(
        self,
        *,
        providers: dict[str, BaseShellProvider] | None = None,
        default_shell: str = "powershell",
        output_tail_limit: int = 8000,
        process_launcher: ProcessLauncher | None = None,
        state_root: str | Path | None = None,
    ) -> None:
        self._providers = providers or default_shell_providers()
        self._default_shell = default_shell.strip().lower()
        self._output_tail_limit = max(256, int(output_tail_limit))
        self._process_launcher = process_launcher or asyncio.create_subprocess_exec
        self._state_root = Path(state_root).resolve() if state_root is not None else None
        self._sessions: dict[str, ExecSession] = {}
        if self._state_root is not None:
            self._state_root.mkdir(parents=True, exist_ok=True)
            self._recover_detached_sessions()
        self._session_seq = count(self._next_session_sequence_start())

    async def start_command(
        self,
        *,
        command: str,
        shell: str | None = None,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: float | None = None,
        background: bool = False,
        tty: bool = False,
        approval_state: str = "not_required",
        log_path: str | Path | None = None,
        checkpoint_dir: str | Path | None = None,
    ) -> SessionPollResult:
        """Start a command and optionally keep it as a detached background session."""
        normalized_command = command.strip()
        if not normalized_command:
            raise ValueError("`command` must not be empty.")

        session_id = f"exec-{next(self._session_seq)}"
        provider = self._resolve_provider(shell)
        spec = provider.build_subprocess_spec(normalized_command, tty=tty)
        persisted_detached = background and self._state_root is not None
        resolved_log_path = self._resolve_log_path(
            session_id=session_id,
            requested_log_path=log_path,
            persisted_detached=persisted_detached,
        )
        resolved_checkpoint_dir = self._resolve_checkpoint_dir(
            session_id=session_id,
            requested_checkpoint_dir=checkpoint_dir,
            persisted_detached=persisted_detached,
        )
        state_dir = self._session_state_dir(session_id) if persisted_detached else None

        if resolved_log_path is not None:
            log_file = Path(resolved_log_path)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_file.touch(exist_ok=True)
        if resolved_checkpoint_dir is not None:
            Path(resolved_checkpoint_dir).mkdir(parents=True, exist_ok=True)

        process: asyncio.subprocess.Process | None = None
        pid: int | None = None
        if persisted_detached:
            if resolved_log_path is None:
                raise RuntimeError("Detached persisted sessions require a log path.")
            log_handle = Path(resolved_log_path).open("ab")
            try:
                popen_kwargs: dict[str, object] = {
                    "stdin": subprocess.DEVNULL,
                    "stdout": log_handle,
                    "stderr": log_handle,
                    "cwd": str(cwd) if cwd is not None else None,
                    "env": env,
                }
                if sys.platform == "win32":
                    popen_kwargs["creationflags"] = (
                        subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
                    )
                else:
                    popen_kwargs["start_new_session"] = True
                detached_process = subprocess.Popen(
                    [spec.executable, *spec.args],
                    **popen_kwargs,
                )
            finally:
                log_handle.close()
            pid = detached_process.pid
        else:
            process = await self._process_launcher(
                spec.executable,
                *spec.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd) if cwd is not None else None,
                env=env,
            )
            pid = process.pid

        session = ExecSession(
            session_id=session_id,
            shell=provider.canonical_name,
            command=normalized_command,
            cwd=str(cwd) if cwd is not None else None,
            pid=pid,
            status=ExecSessionStatus.RUNNING,
            background=background,
            tty=tty,
            started_at=utc_now(),
            last_activity_at=utc_now(),
            exit_code=None,
            timed_out=False,
            approval_state=approval_state,
            log_path=resolved_log_path,
            checkpoint_dir=resolved_checkpoint_dir,
            detached_persisted=persisted_detached,
            recovered=False,
            state_dir=str(state_dir.resolve()) if state_dir is not None else None,
            manifest_path=(
                str(self._session_manifest_path(session_id).resolve()) if persisted_detached else None
            ),
            process=process,
            tail_limit=self._output_tail_limit,
        )
        if not persisted_detached:
            session.stdout_task = self._start_stream_task(session, "stdout")
            session.stderr_task = self._start_stream_task(session, "stderr")
            session.wait_task = asyncio.create_task(
                self._watch_session(session.session_id, timeout_sec=timeout_sec)
            )
        elif timeout_sec is not None and timeout_sec > 0:
            session.wait_task = asyncio.create_task(
                self._watch_detached_session(session.session_id, timeout_sec=timeout_sec)
            )
        self._sessions[session.session_id] = session
        self._persist_session_state(session)

        if background:
            return self._build_poll_result(session, stdout="", stderr="")

        if session.wait_task is not None:
            await session.wait_task
        poll = await self.read_session(session.session_id)
        if poll is None:
            raise RuntimeError(f"Session disappeared unexpectedly: {session.session_id}")
        return poll

    async def read_session(
        self,
        session_id: str,
        *,
        yield_time_ms: int | None = None,
    ) -> SessionPollResult | None:
        """Poll incremental session output."""
        session = self._sessions.get(session_id)
        if session is None:
            return None

        if yield_time_ms and yield_time_ms > 0:
            deadline = asyncio.get_running_loop().time() + (yield_time_ms / 1000.0)
            collected_stdout = ""
            collected_stderr = ""
            while True:
                poll = await self._collect_session_output(session)
                if poll is None:
                    return None
                collected_stdout += poll.stdout
                collected_stderr += poll.stderr
                if poll.status is not ExecSessionStatus.RUNNING:
                    return self._merge_poll_output(
                        poll,
                        stdout=collected_stdout,
                        stderr=collected_stderr,
                    )
                if asyncio.get_running_loop().time() >= deadline:
                    return self._merge_poll_output(
                        poll,
                        stdout=collected_stdout,
                        stderr=collected_stderr,
                    )
                await asyncio.sleep(0.02)

        return await self._collect_session_output(session)

    async def write_stdin(
        self,
        session_id: str,
        *,
        chars: str = "",
        yield_time_ms: int | None = None,
    ) -> SessionPollResult | None:
        """Write chars to session stdin and return incremental output."""
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if chars and session.detached_persisted and session.process is None:
            raise RuntimeError(f"stdin is unavailable for detached persisted session: {session_id}")

        process = session.process
        if process is not None and process.returncode is None and process.stdin is not None and chars:
            process.stdin.write(chars.encode("utf-8"))
            await process.stdin.drain()
            session.mark_activity()
            self._persist_session_state(session)

        return await self.read_session(session_id, yield_time_ms=yield_time_ms)

    async def kill_session(self, session_id: str) -> SessionPollResult | None:
        """Terminate a running session."""
        session = self._sessions.get(session_id)
        if session is None:
            return None

        if session.detached_persisted and session.process is None:
            await self._kill_recovered_detached_session(session)
            return await self.read_session(session_id)

        process = session.process
        if process is not None and process.returncode is None:
            session.status = ExecSessionStatus.KILLED
            session.mark_activity()
            self._persist_session_state(session)
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=2)
            except TimeoutError:
                process.kill()
                await process.wait()

        if session.wait_task is not None and not session.wait_task.done():
            await session.wait_task

        return await self.read_session(session_id)

    async def close(self, *, preserve_detached: bool = True) -> None:
        """Release local runtime resources."""
        for session_id, session in list(self._sessions.items()):
            if session.detached_persisted and preserve_detached:
                await self._detach_session_from_runtime(session)
                continue
            if session.detached_persisted:
                await self.kill_session(session_id)
                await self._detach_session_from_runtime(session)
                continue
            process = session.process
            if process is not None and process.returncode is None:
                await self.kill_session(session_id)
            else:
                if session.wait_task is not None and not session.wait_task.done():
                    await session.wait_task
                await self._release_session_process(session)
        self._sessions.clear()

    def list_sessions(self) -> list[ExecSession]:
        """List sessions known to this runtime."""
        return list(self._sessions.values())

    def _resolve_provider(self, shell: str | None) -> BaseShellProvider:
        resolved = (shell or self._default_shell).strip().lower()
        provider = self._providers.get(resolved)
        if provider is None:
            available = ", ".join(sorted(set(self._providers)))
            raise ValueError(f"Unsupported shell '{resolved}'. Available: {available}")
        return provider

    def _start_stream_task(self, session: ExecSession, stream_name: str) -> asyncio.Task[None] | None:
        process = session.process
        if process is None:
            return None
        stream = getattr(process, stream_name)
        if stream is None:
            return None
        return asyncio.create_task(self._drain_stream(session.session_id, stream_name=stream_name))

    async def _drain_stream(self, session_id: str, *, stream_name: str) -> None:
        session = self._sessions.get(session_id)
        if session is None or session.process is None:
            return

        stream: asyncio.StreamReader | None = getattr(session.process, stream_name)
        if stream is None:
            return

        while True:
            chunk = await stream.read(1024)
            if not chunk:
                return
            text = chunk.decode("utf-8", errors="replace")
            async with session.lock:
                if stream_name == "stdout":
                    session.append_stdout(text)
                else:
                    session.append_stderr(text)
                self._append_session_log(session, text)

    async def _watch_session(self, session_id: str, *, timeout_sec: float | None) -> None:
        session = self._sessions.get(session_id)
        if session is None or session.process is None:
            return
        process = session.process

        try:
            if timeout_sec is not None and timeout_sec > 0:
                await asyncio.wait_for(process.wait(), timeout=timeout_sec)
            else:
                await process.wait()
        except TimeoutError:
            session.timed_out = True
            session.status = ExecSessionStatus.TIMED_OUT
            session.mark_activity()
            self._persist_session_state(session)
            process.kill()
            await process.wait()
        except asyncio.CancelledError:
            return

        await self._await_stream_tasks(session)
        await self._refresh_session_state(session)

    async def _refresh_session_state(self, session: ExecSession) -> None:
        process = session.process
        if session.detached_persisted and process is None:
            await self._refresh_recovered_detached_session_state(session)
            return
        if process is None or process.returncode is None:
            return

        await self._await_stream_tasks(session)
        async with session.lock:
            if session.status in {ExecSessionStatus.TIMED_OUT, ExecSessionStatus.KILLED}:
                session.exit_code = process.returncode
            else:
                session.exit_code = process.returncode
                session.status = (
                    ExecSessionStatus.COMPLETED
                    if process.returncode == 0
                    else ExecSessionStatus.FAILED
                )
            session.mark_activity()
        self._persist_session_state(session)
        await self._release_session_process(session)

    async def _await_stream_tasks(self, session: ExecSession) -> None:
        tasks = [task for task in (session.stdout_task, session.stderr_task) if task is not None]
        if not tasks:
            return
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _release_session_process(self, session: ExecSession) -> None:
        process = session.process
        if process is None:
            self._close_log_handle(session)
            return
        stdin = process.stdin
        if stdin is not None:
            with contextlib.suppress(Exception):
                stdin.close()
            wait_closed = getattr(stdin, "wait_closed", None)
            if callable(wait_closed):
                with contextlib.suppress(Exception):
                    await wait_closed()
        transport = getattr(process, "_transport", None)
        close = getattr(transport, "close", None)
        if callable(close):
            with contextlib.suppress(Exception):
                close()
        self._close_log_handle(session)
        session.process = None

    @staticmethod
    def _append_session_log(session: ExecSession, text: str) -> None:
        if not text or not session.log_path:
            return
        log_file = Path(session.log_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(text)

    async def _collect_session_output(self, session: ExecSession) -> SessionPollResult | None:
        if session.session_id not in self._sessions:
            return None
        if session.detached_persisted:
            return await self._collect_detached_session_output(session)

        await self._refresh_session_state(session)
        if session.wait_task is not None and session.wait_task.done():
            await self._refresh_session_state(session)
        async with session.lock:
            stdout = session.consume_stdout_delta()
            stderr = session.consume_stderr_delta()
            return self._build_poll_result(session, stdout=stdout, stderr=stderr)

    @staticmethod
    def _build_poll_result(
        session: ExecSession,
        *,
        stdout: str,
        stderr: str,
    ) -> SessionPollResult:
        return SessionPollResult(
            session_id=session.session_id,
            shell=session.shell,
            status=session.status,
            background=session.background,
            tty=session.tty,
            pid=session.pid,
            exit_code=session.exit_code,
            timed_out=session.timed_out,
            approval_state=session.approval_state,
            stdout=stdout,
            stderr=stderr,
            detached=session.detached_persisted,
            restored=session.recovered,
            supports_stdin=not session.detached_persisted,
        )

    @staticmethod
    def _merge_poll_output(
        poll: SessionPollResult,
        *,
        stdout: str,
        stderr: str,
    ) -> SessionPollResult:
        return SessionPollResult(
            session_id=poll.session_id,
            shell=poll.shell,
            status=poll.status,
            background=poll.background,
            tty=poll.tty,
            pid=poll.pid,
            exit_code=poll.exit_code,
            timed_out=poll.timed_out,
            approval_state=poll.approval_state,
            stdout=stdout,
            stderr=stderr,
            detached=poll.detached,
            restored=poll.restored,
            supports_stdin=poll.supports_stdin,
        )

    def _resolve_log_path(
        self,
        *,
        session_id: str,
        requested_log_path: str | Path | None,
        persisted_detached: bool,
    ) -> str | None:
        if requested_log_path is not None:
            return str(Path(requested_log_path).resolve())
        if not persisted_detached:
            return None
        return str((self._session_state_dir(session_id) / "session.log").resolve())

    def _resolve_checkpoint_dir(
        self,
        *,
        session_id: str,
        requested_checkpoint_dir: str | Path | None,
        persisted_detached: bool,
    ) -> str | None:
        if requested_checkpoint_dir is not None:
            return str(Path(requested_checkpoint_dir).resolve())
        if not persisted_detached:
            return None
        return str((self._session_state_dir(session_id) / "checkpoints").resolve())

    def _session_state_dir(self, session_id: str) -> Path:
        if self._state_root is None:
            raise RuntimeError("Detached session state root is not configured.")
        return self._state_root / session_id

    def _session_manifest_path(self, session_id: str) -> Path:
        return self._session_state_dir(session_id) / "manifest.json"

    def _persist_session_state(self, session: ExecSession) -> None:
        if not session.detached_persisted or not session.manifest_path:
            return
        manifest_path = Path(session.manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot = ExecSessionSnapshot.from_session(session)
        tmp_path = manifest_path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(snapshot.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(manifest_path)

    async def recover_detached_sessions(self) -> None:
        if self._state_root is None:
            return
        self._recover_detached_sessions()

    def _recover_detached_sessions(self) -> None:
        if self._state_root is None:
            return
        for manifest_path in sorted(self._state_root.glob("*/manifest.json")):
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            snapshot = ExecSessionSnapshot.from_dict(payload)
            if snapshot is None or not snapshot.background or not snapshot.detached_persisted:
                continue
            session = snapshot.to_session(
                tail_limit=self._output_tail_limit,
                manifest_path=manifest_path,
                recovered=True,
            )
            if session.log_path:
                with contextlib.suppress(OSError):
                    session.log_read_offset = max(
                        0,
                        Path(session.log_path).stat().st_size - self._output_tail_limit,
                    )
            self._sessions[session.session_id] = session

    async def _watch_detached_session(self, session_id: str, *, timeout_sec: float | None) -> None:
        session = self._sessions.get(session_id)
        if session is None or not session.detached_persisted:
            return

        loop = asyncio.get_running_loop()
        deadline = (
            loop.time() + timeout_sec
            if timeout_sec is not None and timeout_sec > 0
            else None
        )
        try:
            while True:
                if session.status is not ExecSessionStatus.RUNNING:
                    return
                if not await self._is_pid_running(session.pid):
                    await self._refresh_recovered_detached_session_state(session)
                    return
                if deadline is not None and loop.time() >= deadline:
                    async with session.lock:
                        session.timed_out = True
                        session.status = ExecSessionStatus.TIMED_OUT
                        session.mark_activity()
                    self._persist_session_state(session)
                    if session.pid is not None:
                        await self._terminate_pid(session.pid)
                    return
                await asyncio.sleep(0.2)
        except asyncio.CancelledError:
            return

    def _next_session_sequence_start(self) -> int:
        next_index = 1
        for session_id in self._sessions:
            if not session_id.startswith("exec-"):
                continue
            with contextlib.suppress(ValueError):
                next_index = max(next_index, int(session_id.split("-", 1)[1]) + 1)
        return next_index

    async def _collect_detached_session_output(self, session: ExecSession) -> SessionPollResult | None:
        if session.session_id not in self._sessions:
            return None
        await self._refresh_session_state(session)
        if session.wait_task is not None and session.wait_task.done():
            await self._refresh_session_state(session)
        async with session.lock:
            stdout = self._read_detached_log_delta(session)
            return self._build_poll_result(session, stdout=stdout, stderr="")

    def _read_detached_log_delta(self, session: ExecSession) -> str:
        if not session.log_path:
            return ""
        log_file = Path(session.log_path)
        if not log_file.exists():
            return ""
        size = log_file.stat().st_size
        if session.log_read_offset > size:
            session.log_read_offset = 0
        with log_file.open("rb") as handle:
            handle.seek(session.log_read_offset)
            chunk = handle.read()
            session.log_read_offset = handle.tell()
        return chunk.decode("utf-8", errors="replace") if chunk else ""

    async def _refresh_recovered_detached_session_state(self, session: ExecSession) -> None:
        if session.status is not ExecSessionStatus.RUNNING:
            return
        if await self._is_pid_running(session.pid):
            return
        async with session.lock:
            if session.exit_code is None or session.exit_code == 0:
                session.status = ExecSessionStatus.COMPLETED
            else:
                session.status = ExecSessionStatus.FAILED
            session.mark_activity()
        self._persist_session_state(session)

    async def _kill_recovered_detached_session(self, session: ExecSession) -> None:
        if session.pid is None:
            async with session.lock:
                session.status = ExecSessionStatus.KILLED
                session.mark_activity()
            self._persist_session_state(session)
            return
        if not await self._is_pid_running(session.pid):
            await self._refresh_recovered_detached_session_state(session)
            return
        await self._terminate_pid(session.pid)
        async with session.lock:
            session.status = ExecSessionStatus.KILLED
            session.mark_activity()
        self._persist_session_state(session)

    async def _detach_session_from_runtime(self, session: ExecSession) -> None:
        process = session.process
        if process is not None and process.returncode is None:
            return
        wait_task = session.wait_task
        if wait_task is not None and not wait_task.done():
            wait_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await wait_task
        session.wait_task = None
        session.stdout_task = None
        session.stderr_task = None
        if process is not None and process.stdin is not None:
            with contextlib.suppress(Exception):
                process.stdin.close()
        self._close_log_handle(session)
        session.process = None

    def _close_log_handle(self, session: ExecSession) -> None:
        log_handle = session.log_handle
        if log_handle is not None:
            with contextlib.suppress(Exception):
                log_handle.close()
            session.log_handle = None

    async def _is_pid_running(self, pid: int | None) -> bool:
        if pid is None:
            return False
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            process_handle = kernel32.OpenProcess(
                0x1000,  # PROCESS_QUERY_LIMITED_INFORMATION
                False,
                pid,
            )
            if not process_handle:
                return False
            try:
                exit_code = wintypes.DWORD()
                if not kernel32.GetExitCodeProcess(process_handle, ctypes.byref(exit_code)):
                    return False
                return exit_code.value == 259  # STILL_ACTIVE
            finally:
                kernel32.CloseHandle(process_handle)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    async def _terminate_pid(self, pid: int) -> None:
        if sys.platform == "win32":
            import ctypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            process_handle = kernel32.OpenProcess(
                0x0001 | 0x0400,  # PROCESS_TERMINATE | PROCESS_QUERY_INFORMATION
                False,
                pid,
            )
            if process_handle:
                try:
                    kernel32.TerminateProcess(process_handle, 1)
                finally:
                    kernel32.CloseHandle(process_handle)
            return
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGTERM)
        for _ in range(20):
            if not await self._is_pid_running(pid):
                return
            await asyncio.sleep(0.1)
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)
