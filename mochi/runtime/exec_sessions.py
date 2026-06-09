"""Exec runtime session models."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import IO, Any


def utc_now() -> datetime:
    """取得 UTC 現在時間。"""
    return datetime.now(UTC)


class ExecSessionStatus(str, Enum):
    """Exec session 狀態。"""

    PENDING_APPROVAL = "pending_approval"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"
    TIMED_OUT = "timed_out"


@dataclass
class ExecSession:
    """一個執行命令的 session。"""

    session_id: str
    shell: str
    command: str
    cwd: str | None
    pid: int | None
    status: ExecSessionStatus
    background: bool
    tty: bool
    started_at: datetime
    last_activity_at: datetime
    exit_code: int | None
    timed_out: bool
    approval_state: str
    log_path: str | None = None
    checkpoint_dir: str | None = None
    detached_persisted: bool = False
    recovered: bool = False
    state_dir: str | None = None
    manifest_path: str | None = None
    log_read_offset: int = 0
    process: asyncio.subprocess.Process | None = field(default=None, repr=False)
    log_handle: IO[bytes] | None = field(default=None, repr=False)
    stdout_tail: str = ""
    stderr_tail: str = ""
    stdout_total_chars: int = 0
    stderr_total_chars: int = 0
    stdout_read_cursor: int = 0
    stderr_read_cursor: int = 0
    tail_limit: int = 8000
    stdout_task: asyncio.Task[None] | None = field(default=None, repr=False)
    stderr_task: asyncio.Task[None] | None = field(default=None, repr=False)
    wait_task: asyncio.Task[None] | None = field(default=None, repr=False)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def mark_activity(self) -> None:
        """更新最後活動時間。"""
        self.last_activity_at = utc_now()

    def append_stdout(self, text: str) -> None:
        """追加 stdout tail。"""
        self.stdout_total_chars += len(text)
        self.stdout_tail = _append_tail(self.stdout_tail, text, self.tail_limit)
        self.mark_activity()

    def append_stderr(self, text: str) -> None:
        """追加 stderr tail。"""
        self.stderr_total_chars += len(text)
        self.stderr_tail = _append_tail(self.stderr_tail, text, self.tail_limit)
        self.mark_activity()

    def consume_stdout_delta(self) -> str:
        """取得自上次讀取後的 stdout 增量。"""
        delta, cursor = _consume_delta(
            tail=self.stdout_tail,
            total_chars=self.stdout_total_chars,
            read_cursor=self.stdout_read_cursor,
        )
        self.stdout_read_cursor = cursor
        return delta

    def consume_stderr_delta(self) -> str:
        """取得自上次讀取後的 stderr 增量。"""
        delta, cursor = _consume_delta(
            tail=self.stderr_tail,
            total_chars=self.stderr_total_chars,
            read_cursor=self.stderr_read_cursor,
        )
        self.stderr_read_cursor = cursor
        return delta


@dataclass(frozen=True)
class SessionPollResult:
    """對外回傳的 session 輪詢結果。"""

    session_id: str
    shell: str
    status: ExecSessionStatus
    background: bool
    tty: bool
    pid: int | None
    exit_code: int | None
    timed_out: bool
    approval_state: str
    stdout: str
    stderr: str
    detached: bool = False
    restored: bool = False
    supports_stdin: bool = True


@dataclass(frozen=True)
class ExecSessionSnapshot:
    """Persistent detached session snapshot stored on disk."""

    manifest_version: int
    session_id: str
    shell: str
    command: str
    cwd: str | None
    pid: int | None
    status: str
    background: bool
    tty: bool
    started_at: str
    last_activity_at: str
    exit_code: int | None
    timed_out: bool
    approval_state: str
    log_path: str | None
    checkpoint_dir: str | None
    detached_persisted: bool

    @classmethod
    def from_session(cls, session: ExecSession) -> ExecSessionSnapshot:
        return cls(
            manifest_version=1,
            session_id=session.session_id,
            shell=session.shell,
            command=session.command,
            cwd=session.cwd,
            pid=session.pid,
            status=session.status.value,
            background=session.background,
            tty=session.tty,
            started_at=session.started_at.isoformat(),
            last_activity_at=session.last_activity_at.isoformat(),
            exit_code=session.exit_code,
            timed_out=session.timed_out,
            approval_state=session.approval_state,
            log_path=session.log_path,
            checkpoint_dir=session.checkpoint_dir,
            detached_persisted=session.detached_persisted,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExecSessionSnapshot | None:
        try:
            return cls(
                manifest_version=int(payload.get("manifest_version") or 1),
                session_id=str(payload["session_id"]),
                shell=str(payload["shell"]),
                command=str(payload["command"]),
                cwd=str(payload["cwd"]) if isinstance(payload.get("cwd"), str) else None,
                pid=int(payload["pid"]) if payload.get("pid") is not None else None,
                status=str(payload["status"]),
                background=bool(payload.get("background", False)),
                tty=bool(payload.get("tty", False)),
                started_at=str(payload["started_at"]),
                last_activity_at=str(payload["last_activity_at"]),
                exit_code=int(payload["exit_code"]) if payload.get("exit_code") is not None else None,
                timed_out=bool(payload.get("timed_out", False)),
                approval_state=str(payload.get("approval_state") or "not_required"),
                log_path=str(payload["log_path"]) if isinstance(payload.get("log_path"), str) else None,
                checkpoint_dir=(
                    str(payload["checkpoint_dir"]) if isinstance(payload.get("checkpoint_dir"), str) else None
                ),
                detached_persisted=bool(payload.get("detached_persisted", False)),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "session_id": self.session_id,
            "shell": self.shell,
            "command": self.command,
            "cwd": self.cwd,
            "pid": self.pid,
            "status": self.status,
            "background": self.background,
            "tty": self.tty,
            "started_at": self.started_at,
            "last_activity_at": self.last_activity_at,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "approval_state": self.approval_state,
            "log_path": self.log_path,
            "checkpoint_dir": self.checkpoint_dir,
            "detached_persisted": self.detached_persisted,
        }

    def to_session(
        self,
        *,
        tail_limit: int,
        manifest_path: str | Path,
        recovered: bool,
    ) -> ExecSession:
        return ExecSession(
            session_id=self.session_id,
            shell=self.shell,
            command=self.command,
            cwd=self.cwd,
            pid=self.pid,
            status=_parse_status(self.status),
            background=self.background,
            tty=self.tty,
            started_at=_parse_datetime(self.started_at),
            last_activity_at=_parse_datetime(self.last_activity_at),
            exit_code=self.exit_code,
            timed_out=self.timed_out,
            approval_state=self.approval_state,
            log_path=self.log_path,
            checkpoint_dir=self.checkpoint_dir,
            detached_persisted=self.detached_persisted,
            recovered=recovered,
            state_dir=str(Path(manifest_path).resolve().parent),
            manifest_path=str(Path(manifest_path).resolve()),
            tail_limit=tail_limit,
        )


def _append_tail(existing: str, chunk: str, limit: int) -> str:
    if not chunk:
        return existing
    combined = existing + chunk
    if len(combined) <= limit:
        return combined
    return combined[-limit:]


def _consume_delta(*, tail: str, total_chars: int, read_cursor: int) -> tuple[str, int]:
    if total_chars <= 0:
        return "", 0

    retained_start = max(0, total_chars - len(tail))
    effective_start = max(read_cursor, retained_start)
    start_index = max(0, effective_start - retained_start)
    return tail[start_index:], total_chars


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_status(value: str) -> ExecSessionStatus:
    try:
        return ExecSessionStatus(value)
    except ValueError:
        return ExecSessionStatus.RUNNING
