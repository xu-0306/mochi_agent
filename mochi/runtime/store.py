"""SQLite-backed runtime store."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class RuntimeStore:
    """Persist task runs, task events, and approval requests."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self) -> None:
        async with self._lock:
            if self._initialized:
                return
            await asyncio.to_thread(self._init_db)
            self._initialized = True

    def _init_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS task_runs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    input TEXT NOT NULL,
                    session_id TEXT,
                    project_id TEXT,
                    workspace_dir TEXT,
                    inference_overrides_json TEXT NOT NULL,
                    permission_override_json TEXT,
                    final_answer TEXT,
                    error TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS task_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    event_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(task_id) REFERENCES task_runs(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS approval_requests (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    call_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    arguments_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reason TEXT,
                    resolved_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(task_id) REFERENCES task_runs(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_task_events_task_id_seq ON task_events(task_id, seq)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_approvals_task_id ON approval_requests(task_id)"
            )
            _ensure_column(conn, "task_runs", "final_answer", "TEXT")
            _ensure_column(conn, "task_runs", "started_at", "TEXT")
            _ensure_column(conn, "task_runs", "finished_at", "TEXT")
            conn.commit()

    async def create_task_run(
        self,
        *,
        task_id: str,
        input_text: str,
        session_id: str | None,
        project_id: str | None,
        workspace_dir: str | None,
        inference_overrides: dict[str, Any] | None = None,
        permission_override: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        await self.initialize()
        now = _now_iso()

        def _op() -> None:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO task_runs (
                        id, status, input, session_id, project_id, workspace_dir,
                        inference_overrides_json, permission_override_json, final_answer,
                        error, started_at, finished_at, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task_id,
                        "queued",
                        input_text,
                        session_id,
                        project_id,
                        workspace_dir,
                        json.dumps(inference_overrides or {}, ensure_ascii=False),
                        json.dumps(permission_override or {}, ensure_ascii=False),
                        None,
                        None,
                        None,
                        None,
                        now,
                        now,
                    ),
                )
                conn.commit()

        await asyncio.to_thread(_op)
        return await self.get_task_run(task_id) or {}

    async def update_task_status(
        self,
        task_id: str,
        status: str,
        *,
        error: str | None = None,
        permission_override: dict[str, Any] | None = None,
        final_answer: str | None = None,
    ) -> None:
        await self.initialize()
        now = _now_iso()

        def _op() -> None:
            started_at = now if status in {"running", "resumed"} else None
            finished_at = now if status in {"succeeded", "failed", "cancelled"} else None
            with sqlite3.connect(self._db_path) as conn:
                existing = conn.execute(
                    "SELECT started_at, finished_at, final_answer FROM task_runs WHERE id=?",
                    (task_id,),
                ).fetchone()
                current_started_at = existing[0] if existing else None
                current_finished_at = existing[1] if existing else None
                current_final_answer = existing[2] if existing else None
                conn.execute(
                    """
                    UPDATE task_runs
                    SET status=?,
                        error=?,
                        permission_override_json=?,
                        final_answer=?,
                        started_at=?,
                        finished_at=?,
                        updated_at=?
                    WHERE id=?
                    """,
                    (
                        status,
                        error,
                        json.dumps(permission_override, ensure_ascii=False)
                        if permission_override is not None
                        else None,
                        final_answer if final_answer is not None else current_final_answer,
                        current_started_at or started_at,
                        finished_at or current_finished_at,
                        now,
                        task_id,
                    ),
                )
                conn.commit()

        await asyncio.to_thread(_op)

    async def append_task_event(self, task_id: str, event: dict[str, Any]) -> None:
        await self.initialize()
        now = _now_iso()

        def _op() -> None:
            with sqlite3.connect(self._db_path) as conn:
                row = conn.execute(
                    "SELECT COALESCE(MAX(seq), 0) + 1 FROM task_events WHERE task_id=?",
                    (task_id,),
                ).fetchone()
                seq = int(row[0]) if row else 1
                conn.execute(
                    "INSERT INTO task_events(task_id, seq, event_json, created_at) VALUES (?, ?, ?, ?)",
                    (task_id, seq, json.dumps(event, ensure_ascii=False), now),
                )
                conn.commit()

        await asyncio.to_thread(_op)

    async def get_task_events(self, task_id: str) -> list[dict[str, Any]]:
        await self.initialize()

        def _op() -> list[dict[str, Any]]:
            with sqlite3.connect(self._db_path) as conn:
                rows = conn.execute(
                    "SELECT event_json FROM task_events WHERE task_id=? ORDER BY seq ASC",
                    (task_id,),
                ).fetchall()
            return [json.loads(str(row[0])) for row in rows]

        return await asyncio.to_thread(_op)

    async def get_task_run(self, task_id: str) -> dict[str, Any] | None:
        await self.initialize()

        def _op() -> dict[str, Any] | None:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute("SELECT * FROM task_runs WHERE id=?", (task_id,)).fetchone()
            if row is None:
                return None
            payload = dict(row)
            payload["inference_overrides"] = json.loads(payload.pop("inference_overrides_json") or "{}")
            payload["permission_override"] = json.loads(payload.pop("permission_override_json") or "{}")
            return payload

        return await asyncio.to_thread(_op)

    async def list_task_runs(self) -> list[dict[str, Any]]:
        await self.initialize()

        def _op() -> list[dict[str, Any]]:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM task_runs ORDER BY datetime(created_at) DESC"
                ).fetchall()
            output: list[dict[str, Any]] = []
            for row in rows:
                payload = dict(row)
                payload["inference_overrides"] = json.loads(
                    payload.pop("inference_overrides_json") or "{}"
                )
                payload["permission_override"] = json.loads(
                    payload.pop("permission_override_json") or "{}"
                )
                output.append(payload)
            return output

        return await asyncio.to_thread(_op)

    async def create_approval_request(
        self,
        *,
        approval_id: str,
        task_id: str,
        call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        await self.initialize()
        now = _now_iso()

        def _op() -> None:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO approval_requests(
                        id, task_id, call_id, tool_name, arguments_json, status,
                        reason, resolved_at, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        approval_id,
                        task_id,
                        call_id,
                        tool_name,
                        json.dumps(arguments, ensure_ascii=False),
                        "pending",
                        None,
                        None,
                        now,
                        now,
                    ),
                )
                conn.commit()

        await asyncio.to_thread(_op)
        return await self.get_approval_request(approval_id) or {}

    async def get_approval_request(self, approval_id: str) -> dict[str, Any] | None:
        await self.initialize()

        def _op() -> dict[str, Any] | None:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT * FROM approval_requests WHERE id=?",
                    (approval_id,),
                ).fetchone()
            if row is None:
                return None
            payload = dict(row)
            payload["arguments"] = json.loads(payload.pop("arguments_json") or "{}")
            return payload

        return await asyncio.to_thread(_op)

    async def list_approval_requests(self, *, status: str | None = None) -> list[dict[str, Any]]:
        await self.initialize()

        def _op() -> list[dict[str, Any]]:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                if status:
                    rows = conn.execute(
                        "SELECT * FROM approval_requests WHERE status=? ORDER BY datetime(created_at) DESC",
                        (status,),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM approval_requests ORDER BY datetime(created_at) DESC"
                    ).fetchall()
            payloads: list[dict[str, Any]] = []
            for row in rows:
                item = dict(row)
                item["arguments"] = json.loads(item.pop("arguments_json") or "{}")
                payloads.append(item)
            return payloads

        return await asyncio.to_thread(_op)

    async def resolve_approval_request(
        self,
        approval_id: str,
        *,
        approved: bool,
        reason: str | None = None,
    ) -> dict[str, Any] | None:
        await self.initialize()
        now = _now_iso()
        status = "approved" if approved else "rejected"

        def _op() -> None:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """
                    UPDATE approval_requests
                    SET status=?, reason=?, resolved_at=?, updated_at=?
                    WHERE id=?
                    """,
                    (status, reason, now, now, approval_id),
                )
                conn.commit()

        await asyncio.to_thread(_op)
        return await self.get_approval_request(approval_id)

    async def get_pending_approval_for_task(self, task_id: str) -> dict[str, Any] | None:
        await self.initialize()

        def _op() -> dict[str, Any] | None:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    """
                    SELECT * FROM approval_requests
                    WHERE task_id=? AND status='pending'
                    ORDER BY datetime(created_at) DESC
                    LIMIT 1
                    """,
                    (task_id,),
                ).fetchone()
            if row is None:
                return None
            payload = dict(row)
            payload["arguments"] = json.loads(payload.pop("arguments_json") or "{}")
            return payload

        return await asyncio.to_thread(_op)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column in columns:
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
