"""Exec runtime service for foreground/background shell execution."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from itertools import count
from pathlib import Path

from mochi.runtime.exec_sessions import ExecSession, ExecSessionStatus, SessionPollResult, utc_now
from mochi.utils.shell_providers import BaseShellProvider, default_shell_providers

ProcessLauncher = Callable[..., Awaitable[asyncio.subprocess.Process]]


class ExecRuntime:
    """管理可互動 shell session 的執行時。"""

    def __init__(
        self,
        *,
        providers: dict[str, BaseShellProvider] | None = None,
        default_shell: str = "powershell",
        output_tail_limit: int = 8000,
        process_launcher: ProcessLauncher | None = None,
    ) -> None:
        self._providers = providers or default_shell_providers()
        self._default_shell = default_shell.strip().lower()
        self._output_tail_limit = max(256, int(output_tail_limit))
        self._process_launcher = process_launcher or asyncio.create_subprocess_exec
        self._sessions: dict[str, ExecSession] = {}
        self._session_seq = count(1)

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
    ) -> SessionPollResult:
        """啟動命令；foreground 等待完成，background 立即回傳 session。"""
        normalized_command = command.strip()
        if not normalized_command:
            raise ValueError("`command` must not be empty.")

        provider = self._resolve_provider(shell)
        spec = provider.build_subprocess_spec(normalized_command, tty=tty)
        process = await self._process_launcher(
            spec.executable,
            *spec.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
        )
        session = ExecSession(
            session_id=f"exec-{next(self._session_seq)}",
            shell=provider.canonical_name,
            command=normalized_command,
            cwd=str(cwd) if cwd is not None else None,
            pid=process.pid,
            status=ExecSessionStatus.RUNNING,
            background=background,
            tty=tty,
            started_at=utc_now(),
            last_activity_at=utc_now(),
            exit_code=None,
            timed_out=False,
            approval_state=approval_state,
            process=process,
            tail_limit=self._output_tail_limit,
        )
        session.stdout_task = self._start_stream_task(session, "stdout")
        session.stderr_task = self._start_stream_task(session, "stderr")
        session.wait_task = asyncio.create_task(
            self._watch_session(session.session_id, timeout_sec=timeout_sec)
        )
        self._sessions[session.session_id] = session

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
        """輪詢 session 增量輸出。"""
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
        """向 session stdin 寫入字串，並回傳最新增量輸出。"""
        session = self._sessions.get(session_id)
        if session is None:
            return None

        process = session.process
        if process is not None and process.returncode is None and process.stdin is not None and chars:
            process.stdin.write(chars.encode("utf-8"))
            await process.stdin.drain()
            session.mark_activity()

        return await self.read_session(session_id, yield_time_ms=yield_time_ms)

    async def kill_session(self, session_id: str) -> SessionPollResult | None:
        """終止一個執行中的 session。"""
        session = self._sessions.get(session_id)
        if session is None:
            return None

        process = session.process
        if process is not None and process.returncode is None:
            session.status = ExecSessionStatus.KILLED
            session.mark_activity()
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=2)
            except TimeoutError:
                process.kill()
                await process.wait()

        if session.wait_task is not None and not session.wait_task.done():
            await session.wait_task

        return await self.read_session(session_id)

    def list_sessions(self) -> list[ExecSession]:
        """列出目前 runtime 管理的 session。"""
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
            process.kill()
            await process.wait()

        await self._await_stream_tasks(session)
        await self._refresh_session_state(session)

    async def _refresh_session_state(self, session: ExecSession) -> None:
        process = session.process
        if process is None:
            return

        if process.returncode is None:
            return

        await self._await_stream_tasks(session)
        async with session.lock:
            if session.status == ExecSessionStatus.TIMED_OUT:
                session.exit_code = process.returncode
            elif session.status == ExecSessionStatus.KILLED:
                session.exit_code = process.returncode
            else:
                session.exit_code = process.returncode
                if process.returncode == 0:
                    session.status = ExecSessionStatus.COMPLETED
                else:
                    session.status = ExecSessionStatus.FAILED
            session.mark_activity()

    async def _await_stream_tasks(self, session: ExecSession) -> None:
        tasks = [task for task in (session.stdout_task, session.stderr_task) if task is not None]
        if not tasks:
            return
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _collect_session_output(self, session: ExecSession) -> SessionPollResult | None:
        if session.session_id not in self._sessions:
            return None
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
        )
