"""SQLite-backed runtime store."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_UNSET = object()


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
                    project_workspace_dir TEXT,
                    task_workspace_dir TEXT,
                    task_type TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
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
                    metadata_json TEXT,
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
                """
                CREATE TABLE IF NOT EXISTS agent_runs (
                    id TEXT PRIMARY KEY,
                    protocol_id TEXT NOT NULL,
                    title TEXT,
                    topic TEXT,
                    project_id TEXT,
                    workspace_dir TEXT,
                    status TEXT NOT NULL,
                    selected_models_roles_json TEXT NOT NULL,
                    evaluation_policy_json TEXT NOT NULL,
                    run_policy_json TEXT NOT NULL DEFAULT '{}',
                    schedule_json TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    latest_error TEXT,
                    evidence_status_json TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_run_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    event_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES agent_runs(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_run_artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    artifact_id TEXT,
                    artifact_type TEXT NOT NULL,
                    title TEXT,
                    uri TEXT,
                    mime_type TEXT,
                    size_bytes INTEGER,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES agent_runs(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_task_events_task_id_seq ON task_events(task_id, seq)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_approvals_task_id ON approval_requests(task_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_agent_runs_created_at ON agent_runs(created_at)"
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_agent_run_events_run_id_seq
                ON agent_run_events(run_id, seq)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_agent_run_artifacts_run_id
                ON agent_run_artifacts(run_id)
                """
            )
            _ensure_column(conn, "task_runs", "final_answer", "TEXT")
            _ensure_column(conn, "task_runs", "started_at", "TEXT")
            _ensure_column(conn, "task_runs", "finished_at", "TEXT")
            _ensure_column(conn, "task_runs", "project_workspace_dir", "TEXT")
            _ensure_column(conn, "task_runs", "task_workspace_dir", "TEXT")
            _ensure_column(conn, "task_runs", "task_type", "TEXT")
            _ensure_column(conn, "task_runs", "metadata_json", "TEXT NOT NULL DEFAULT '{}'")
            _ensure_column(conn, "approval_requests", "metadata_json", "TEXT")
            _ensure_column(conn, "agent_runs", "title", "TEXT")
            _ensure_column(conn, "agent_runs", "topic", "TEXT")
            _ensure_column(conn, "agent_runs", "project_id", "TEXT")
            _ensure_column(conn, "agent_runs", "workspace_dir", "TEXT")
            _ensure_column(conn, "agent_runs", "latest_error", "TEXT")
            _ensure_column(conn, "agent_runs", "evidence_status_json", "TEXT NOT NULL DEFAULT '{}'")
            _ensure_column(conn, "agent_runs", "run_policy_json", "TEXT NOT NULL DEFAULT '{}'")
            _ensure_column(conn, "agent_run_artifacts", "artifact_id", "TEXT")
            conn.commit()

    async def create_task_run(
        self,
        *,
        task_id: str,
        input_text: str,
        session_id: str | None,
        project_id: str | None,
        workspace_dir: str | None,
        project_workspace_dir: str | None,
        task_workspace_dir: str | None,
        task_type: str | None = None,
        metadata: dict[str, Any] | None = None,
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
                        project_workspace_dir, task_workspace_dir, task_type, metadata_json,
                        inference_overrides_json, permission_override_json, final_answer,
                        error, started_at, finished_at, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task_id,
                        "queued",
                        input_text,
                        session_id,
                        project_id,
                        workspace_dir,
                        project_workspace_dir,
                        task_workspace_dir,
                        task_type,
                        json.dumps(metadata or {}, ensure_ascii=False),
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
            payload["metadata"] = json.loads(payload.pop("metadata_json") or "{}")
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
                payload["metadata"] = json.loads(payload.pop("metadata_json") or "{}")
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
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        await self.initialize()
        now = _now_iso()

        def _op() -> None:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO approval_requests(
                        id, task_id, call_id, tool_name, arguments_json, metadata_json,
                        status, reason, resolved_at, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        approval_id,
                        task_id,
                        call_id,
                        tool_name,
                        json.dumps(arguments, ensure_ascii=False),
                        json.dumps(metadata or {}, ensure_ascii=False),
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
            payload["metadata"] = json.loads(payload.pop("metadata_json") or "{}")
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
                item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
                payloads.append(item)
            return payloads

        return await asyncio.to_thread(_op)

    async def resolve_approval_request(
        self,
        approval_id: str,
        *,
        decision: str,
        reason: str | None = None,
    ) -> dict[str, Any] | None:
        await self.initialize()
        now = _now_iso()
        status = (
            "approved_once"
            if decision == "approve_once"
            else "approved_and_saved_rule"
            if decision == "approve_and_save_rule"
            else "rejected"
        )

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

    async def create_agent_run(
        self,
        *,
        run_id: str,
        protocol_id: str,
        title: str | None,
        topic: str | None,
        project_id: str | None = None,
        workspace_dir: str | None = None,
        selected_models_roles: dict[str, Any] | None = None,
        evaluation_policy: dict[str, Any] | None = None,
        run_policy: dict[str, Any] | None = None,
        schedule: dict[str, Any] | None = None,
        summary: dict[str, Any] | None = None,
        latest_error: str | None = None,
        evidence_status: dict[str, Any] | None = None,
        artifacts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        await self.initialize()
        now = _now_iso()

        def _op() -> None:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO agent_runs (
                        id, protocol_id, title, topic, project_id, workspace_dir, status, selected_models_roles_json,
                        evaluation_policy_json, run_policy_json, schedule_json, summary_json, latest_error,
                        evidence_status_json, started_at, finished_at, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        protocol_id,
                        title,
                        topic,
                        project_id,
                        workspace_dir,
                        "created",
                        json.dumps(selected_models_roles or {}, ensure_ascii=False),
                        json.dumps(evaluation_policy or {}, ensure_ascii=False),
                        json.dumps(run_policy or {}, ensure_ascii=False),
                        json.dumps(schedule or {}, ensure_ascii=False),
                        json.dumps(summary or {}, ensure_ascii=False),
                        latest_error,
                        json.dumps(evidence_status or {}, ensure_ascii=False),
                        None,
                        None,
                        now,
                        now,
                    ),
                )
                for artifact in artifacts or []:
                    conn.execute(
                        """
                        INSERT INTO agent_run_artifacts(
                            run_id, artifact_id, artifact_type, title, uri, mime_type,
                            size_bytes, metadata_json, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            run_id,
                            artifact.get("artifact_id"),
                            str(artifact.get("artifact_type") or ""),
                            artifact.get("title"),
                            artifact.get("uri"),
                            artifact.get("mime_type"),
                            artifact.get("size_bytes"),
                            json.dumps(artifact.get("metadata") or {}, ensure_ascii=False),
                            now,
                            now,
                        ),
                    )
                conn.commit()

        await asyncio.to_thread(_op)
        return await self.get_agent_run(run_id) or {}

    async def update_agent_run_status(
        self,
        run_id: str,
        status: str,
        *,
        latest_error: str | None | object = _UNSET,
        reset_started_at: bool = False,
        reset_finished_at: bool = False,
    ) -> None:
        await self.initialize()
        now = _now_iso()

        def _op() -> None:
            with sqlite3.connect(self._db_path) as conn:
                existing = conn.execute(
                    "SELECT started_at, finished_at, latest_error FROM agent_runs WHERE id=?",
                    (run_id,),
                ).fetchone()
                if existing is None:
                    return
                current_started_at = None if reset_started_at else existing[0]
                current_finished_at = None if reset_finished_at else existing[1]
                current_latest_error = existing[2]
                started_at = now if status == "running" else current_started_at
                finished_at = (
                    now
                    if status in {"cancelled", "failed", "succeeded", "partial"}
                    else None
                )
                conn.execute(
                    """
                    UPDATE agent_runs
                    SET status=?,
                        latest_error=?,
                        started_at=?,
                        finished_at=?,
                        updated_at=?
                    WHERE id=?
                    """,
                    (
                        status,
                        current_latest_error if latest_error is _UNSET else latest_error,
                        started_at,
                        finished_at if finished_at is not None else current_finished_at,
                        now,
                        run_id,
                    ),
                )
                conn.commit()

        await asyncio.to_thread(_op)

    async def update_agent_run_schedule(
        self,
        run_id: str,
        schedule: dict[str, Any],
    ) -> None:
        await self.initialize()
        now = _now_iso()

        def _op() -> None:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """
                    UPDATE agent_runs
                    SET schedule_json=?,
                        updated_at=?
                    WHERE id=?
                    """,
                    (
                        json.dumps(schedule, ensure_ascii=False),
                        now,
                        run_id,
                    ),
                )
                conn.commit()

        await asyncio.to_thread(_op)

    async def update_agent_run_metadata(
        self,
        run_id: str,
        *,
        summary: dict[str, Any] | None = None,
        evidence_status: dict[str, Any] | None = None,
        project_id: str | None | object = _UNSET,
        workspace_dir: str | None | object = _UNSET,
        latest_error: str | None | object = _UNSET,
    ) -> None:
        await self.initialize()
        now = _now_iso()

        def _op() -> None:
            with sqlite3.connect(self._db_path) as conn:
                existing = conn.execute(
                    """
                    SELECT summary_json, evidence_status_json, latest_error, project_id, workspace_dir
                    FROM agent_runs
                    WHERE id=?
                    """,
                    (run_id,),
                ).fetchone()
                if existing is None:
                    return
                current_summary = json.loads(str(existing[0] or "{}"))
                current_evidence_status = json.loads(str(existing[1] or "{}"))
                current_latest_error = existing[2]
                current_project_id = existing[3]
                current_workspace_dir = existing[4]
                next_summary = current_summary if summary is None else summary
                next_evidence_status = (
                    current_evidence_status if evidence_status is None else evidence_status
                )
                next_latest_error = current_latest_error if latest_error is _UNSET else latest_error
                next_project_id = current_project_id if project_id is _UNSET else project_id
                next_workspace_dir = current_workspace_dir if workspace_dir is _UNSET else workspace_dir
                conn.execute(
                    """
                    UPDATE agent_runs
                    SET summary_json=?,
                        evidence_status_json=?,
                        project_id=?,
                        workspace_dir=?,
                        latest_error=?,
                        updated_at=?
                    WHERE id=?
                    """,
                    (
                        json.dumps(next_summary, ensure_ascii=False),
                        json.dumps(next_evidence_status, ensure_ascii=False),
                        next_project_id,
                        next_workspace_dir,
                        next_latest_error,
                        now,
                        run_id,
                    ),
                )
                conn.commit()

        await asyncio.to_thread(_op)

    async def acquire_agent_run_resume_lease(
        self,
        run_id: str,
        *,
        expected_statuses: set[str],
        lease: dict[str, Any],
    ) -> str:
        await self.initialize()
        now = _now_iso()

        def _op() -> str:
            with sqlite3.connect(self._db_path) as conn:
                row = conn.execute(
                    """
                    SELECT status, summary_json
                    FROM agent_runs
                    WHERE id=?
                    """,
                    (run_id,),
                ).fetchone()
                if row is None:
                    return "run_not_found"
                current_status = str(row[0] or "created")
                summary = json.loads(str(row[1] or "{}"))
                recovery_state = (
                    dict(summary.get("recovery_state"))
                    if isinstance(summary.get("recovery_state"), dict)
                    else {}
                )
                resume_runtime = (
                    dict(recovery_state.get("resume_runtime"))
                    if isinstance(recovery_state.get("resume_runtime"), dict)
                    else {}
                )
                active_lease_id = resume_runtime.get("lease_id")
                active_status = str(resume_runtime.get("status") or "").strip().lower()
                if current_status not in expected_statuses:
                    if current_status == "running" and isinstance(active_lease_id, str) and active_lease_id.strip():
                        return "already_running"
                    return "invalid_status"
                if isinstance(active_lease_id, str) and active_lease_id.strip() and active_status == "active":
                    return "lease_conflict"
                recovery_state["resume_runtime"] = dict(lease)
                summary["recovery_state"] = recovery_state
                conn.execute(
                    """
                    UPDATE agent_runs
                    SET status=?,
                        latest_error=?,
                        summary_json=?,
                        started_at=COALESCE(started_at, ?),
                        updated_at=?
                    WHERE id=?
                    """,
                    (
                        "running",
                        None,
                        json.dumps(summary, ensure_ascii=False),
                        now,
                        now,
                        run_id,
                    ),
                )
                conn.commit()
                return "acquired"

        return await asyncio.to_thread(_op)

    async def release_agent_run_resume_lease(
        self,
        run_id: str,
        *,
        lease_id: str,
        status: str,
    ) -> None:
        await self.initialize()
        now = _now_iso()

        def _op() -> None:
            with sqlite3.connect(self._db_path) as conn:
                row = conn.execute(
                    """
                    SELECT summary_json
                    FROM agent_runs
                    WHERE id=?
                    """,
                    (run_id,),
                ).fetchone()
                if row is None:
                    return
                summary = json.loads(str(row[0] or "{}"))
                recovery_state = (
                    dict(summary.get("recovery_state"))
                    if isinstance(summary.get("recovery_state"), dict)
                    else {}
                )
                resume_runtime = (
                    dict(recovery_state.get("resume_runtime"))
                    if isinstance(recovery_state.get("resume_runtime"), dict)
                    else {}
                )
                if str(resume_runtime.get("lease_id") or "").strip() != lease_id:
                    return
                resume_runtime["status"] = status
                resume_runtime["released_at"] = now
                recovery_state["resume_runtime"] = resume_runtime
                summary["recovery_state"] = recovery_state
                conn.execute(
                    """
                    UPDATE agent_runs
                    SET summary_json=?,
                        updated_at=?
                    WHERE id=?
                    """,
                    (
                        json.dumps(summary, ensure_ascii=False),
                        now,
                        run_id,
                    ),
                )
                conn.commit()

        await asyncio.to_thread(_op)

    async def append_agent_run_event(self, run_id: str, event: dict[str, Any]) -> None:
        await self.initialize()
        now = _now_iso()

        def _op() -> None:
            with sqlite3.connect(self._db_path) as conn:
                row = conn.execute(
                    "SELECT COALESCE(MAX(seq), 0) + 1 FROM agent_run_events WHERE run_id=?",
                    (run_id,),
                ).fetchone()
                seq = int(row[0]) if row else 1
                conn.execute(
                    """
                    INSERT INTO agent_run_events(run_id, seq, event_json, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (run_id, seq, json.dumps(event, ensure_ascii=False), now),
                )
                conn.execute(
                    "UPDATE agent_runs SET updated_at=? WHERE id=?",
                    (now, run_id),
                )
                conn.commit()

        await asyncio.to_thread(_op)

    async def get_agent_run_events(self, run_id: str) -> list[dict[str, Any]]:
        await self.initialize()

        def _op() -> list[dict[str, Any]]:
            with sqlite3.connect(self._db_path) as conn:
                rows = conn.execute(
                    """
                    SELECT event_json
                    FROM agent_run_events
                    WHERE run_id=?
                    ORDER BY seq ASC
                    """,
                    (run_id,),
                ).fetchall()
            return [json.loads(str(row[0])) for row in rows]

        return await asyncio.to_thread(_op)

    async def append_agent_run_artifact(
        self,
        run_id: str,
        *,
        artifact_id: str | None,
        artifact_type: str,
        title: str | None = None,
        uri: str | None = None,
        mime_type: str | None = None,
        size_bytes: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        await self.initialize()
        now = _now_iso()

        def _op() -> None:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO agent_run_artifacts(
                        run_id, artifact_id, artifact_type, title, uri, mime_type,
                        size_bytes, metadata_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        artifact_id,
                        artifact_type,
                        title,
                        uri,
                        mime_type,
                        size_bytes,
                        json.dumps(metadata or {}, ensure_ascii=False),
                        now,
                        now,
                    ),
                )
                conn.execute(
                    "UPDATE agent_runs SET updated_at=? WHERE id=?",
                    (now, run_id),
                )
                conn.commit()

        await asyncio.to_thread(_op)

    async def get_agent_run(self, run_id: str) -> dict[str, Any] | None:
        await self.initialize()

        def _op() -> dict[str, Any] | None:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute("SELECT * FROM agent_runs WHERE id=?", (run_id,)).fetchone()
                if row is None:
                    return None
                payload = dict(row)
                payload["selected_models_roles"] = json.loads(
                    payload.pop("selected_models_roles_json") or "{}"
                )
                payload["evaluation_policy"] = json.loads(
                    payload.pop("evaluation_policy_json") or "{}"
                )
                payload["run_policy"] = json.loads(payload.pop("run_policy_json") or "{}")
                payload["schedule"] = json.loads(payload.pop("schedule_json") or "{}")
                payload["summary"] = json.loads(payload.pop("summary_json") or "{}")
                payload["evidence_status"] = json.loads(payload.pop("evidence_status_json") or "{}")
                payload["artifacts"] = _load_agent_run_artifacts(conn, run_id)
                return payload

        return await asyncio.to_thread(_op)

    async def list_agent_runs(self) -> list[dict[str, Any]]:
        await self.initialize()

        def _op() -> list[dict[str, Any]]:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM agent_runs ORDER BY datetime(created_at) DESC"
                ).fetchall()
                output: list[dict[str, Any]] = []
                for row in rows:
                    payload = dict(row)
                    payload["selected_models_roles"] = json.loads(
                        payload.pop("selected_models_roles_json") or "{}"
                    )
                    payload["evaluation_policy"] = json.loads(
                        payload.pop("evaluation_policy_json") or "{}"
                    )
                    payload["run_policy"] = json.loads(payload.pop("run_policy_json") or "{}")
                    payload["schedule"] = json.loads(payload.pop("schedule_json") or "{}")
                    payload["summary"] = json.loads(payload.pop("summary_json") or "{}")
                    payload["evidence_status"] = json.loads(
                        payload.pop("evidence_status_json") or "{}"
                    )
                    payload["artifacts"] = _load_agent_run_artifacts(conn, str(payload["id"]))
                    output.append(payload)
            return output

        return await asyncio.to_thread(_op)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _load_agent_run_artifacts(conn: sqlite3.Connection, run_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT artifact_id, artifact_type, title, uri, mime_type, size_bytes, metadata_json
        FROM agent_run_artifacts
        WHERE run_id=?
        ORDER BY id ASC
        """,
        (run_id,),
    ).fetchall()
    artifacts: list[dict[str, Any]] = []
    for row in rows:
        artifacts.append(
            {
                "artifact_id": row[0],
                "artifact_type": row[1],
                "title": row[2],
                "uri": row[3],
                "mime_type": row[4],
                "size_bytes": row[5],
                "metadata": json.loads(str(row[6] or "{}")),
            }
        )
    return artifacts


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column in columns:
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
