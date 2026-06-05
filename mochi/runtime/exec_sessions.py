"""Exec runtime session models."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum


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
    process: asyncio.subprocess.Process | None = field(default=None, repr=False)
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
