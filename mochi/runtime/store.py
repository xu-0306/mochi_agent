"""SQLite-backed runtime store."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_UNSET = object()
_DEFAULT_GOAL_EXECUTION_MODE = "workflow"
_DEFAULT_SINGLE_AGENT_GOAL_PROTOCOL = "teacher_student_distill"
_GOAL_EXECUTION_MODES = {"single_agent", _DEFAULT_GOAL_EXECUTION_MODE}
_GOAL_WORKER_GENERATION_TERMINAL_STATUSES = {
    "cancelled",
    "completed",
    "failed",
    "rolled_over",
    "superseded",
    "succeeded",
}


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
                """
                CREATE TABLE IF NOT EXISTS goals (
                    id TEXT PRIMARY KEY,
                    objective TEXT NOT NULL,
                    title TEXT,
                    goal_type TEXT,
                    execution_mode TEXT NOT NULL DEFAULT 'workflow',
                    protocol_id TEXT,
                    topic TEXT,
                    project_id TEXT,
                    workspace_dir TEXT,
                    status TEXT NOT NULL,
                    current_attempt_id TEXT,
                    run_policy_json TEXT NOT NULL DEFAULT '{}',
                    capability_policy_json TEXT NOT NULL DEFAULT '{}',
                    source_manifest_json TEXT NOT NULL DEFAULT '{}',
                    summary_json TEXT NOT NULL DEFAULT '{}',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    latest_error TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS goal_attempts (
                    id TEXT PRIMARY KEY,
                    goal_id TEXT NOT NULL,
                    attempt_index INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    trigger TEXT,
                    agent_run_id TEXT,
                    summary_json TEXT NOT NULL DEFAULT '{}',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    latest_error TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(goal_id) REFERENCES goals(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS goal_leases (
                    goal_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    acquired_at TEXT NOT NULL,
                    heartbeat_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    takeover_count INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(goal_id) REFERENCES goals(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS goal_audit_findings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    goal_id TEXT NOT NULL,
                    finding_code TEXT NOT NULL,
                    status TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    details_json TEXT NOT NULL DEFAULT '{}',
                    resolved_at TEXT,
                    closed_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(goal_id) REFERENCES goals(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS goal_operator_controls (
                    scope TEXT PRIMARY KEY,
                    stop_all_goals INTEGER NOT NULL DEFAULT 0,
                    blocked_tools_json TEXT NOT NULL DEFAULT '[]',
                    blocked_domains_json TEXT NOT NULL DEFAULT '[]',
                    block_network_usage INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS goal_operator_audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    subject_type TEXT NOT NULL,
                    subject_id TEXT,
                    action TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    details_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS goal_checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    goal_id TEXT NOT NULL,
                    attempt_id TEXT,
                    agent_run_id TEXT,
                    checkpoint_index INTEGER,
                    stage TEXT,
                    source TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    captured_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(goal_id) REFERENCES goals(id) ON DELETE CASCADE,
                    FOREIGN KEY(attempt_id) REFERENCES goal_attempts(id) ON DELETE SET NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS goal_memory_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    goal_id TEXT NOT NULL,
                    attempt_id TEXT,
                    checkpoint_id INTEGER,
                    snapshot_kind TEXT NOT NULL,
                    snapshot_json TEXT NOT NULL DEFAULT '{}',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    captured_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(goal_id) REFERENCES goals(id) ON DELETE CASCADE,
                    FOREIGN KEY(attempt_id) REFERENCES goal_attempts(id) ON DELETE SET NULL,
                    FOREIGN KEY(checkpoint_id) REFERENCES goal_checkpoints(id) ON DELETE SET NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS goal_worker_generations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    goal_id TEXT NOT NULL,
                    attempt_id TEXT,
                    agent_run_id TEXT,
                    generation_index INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    rollover_reason TEXT,
                    parent_generation_id INTEGER,
                    resume_source_snapshot_id INTEGER,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    started_at TEXT,
                    finished_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(goal_id) REFERENCES goals(id) ON DELETE CASCADE,
                    FOREIGN KEY(attempt_id) REFERENCES goal_attempts(id) ON DELETE SET NULL,
                    FOREIGN KEY(parent_generation_id) REFERENCES goal_worker_generations(id) ON DELETE SET NULL,
                    FOREIGN KEY(resume_source_snapshot_id) REFERENCES goal_memory_snapshots(id) ON DELETE SET NULL
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
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_goals_created_at ON goals(created_at)"
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_goal_attempts_goal_id_attempt_index
                ON goal_attempts(goal_id, attempt_index)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_goal_leases_owner_id
                ON goal_leases(owner_id)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_goal_audit_findings_goal_id_status
                ON goal_audit_findings(goal_id, status)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_goal_operator_audit_log_event_type_created_at
                ON goal_operator_audit_log(event_type, created_at DESC, id DESC)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_goal_checkpoints_goal_id_captured_at
                ON goal_checkpoints(goal_id, captured_at DESC, id DESC)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_goal_checkpoints_attempt_id_captured_at
                ON goal_checkpoints(attempt_id, captured_at DESC, id DESC)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_goal_memory_snapshots_goal_id_captured_at
                ON goal_memory_snapshots(goal_id, captured_at DESC, id DESC)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_goal_memory_snapshots_attempt_id_captured_at
                ON goal_memory_snapshots(attempt_id, captured_at DESC, id DESC)
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
            _ensure_column(conn, "goals", "title", "TEXT")
            _ensure_column(conn, "goals", "goal_type", "TEXT")
            _ensure_column(
                conn,
                "goals",
                "execution_mode",
                "TEXT NOT NULL DEFAULT 'workflow'",
            )
            _ensure_column(conn, "goals", "protocol_id", "TEXT")
            _ensure_column(conn, "goals", "topic", "TEXT")
            _ensure_column(conn, "goals", "project_id", "TEXT")
            _ensure_column(conn, "goals", "workspace_dir", "TEXT")
            _ensure_column(conn, "goals", "current_attempt_id", "TEXT")
            _ensure_column(conn, "goals", "run_policy_json", "TEXT NOT NULL DEFAULT '{}'")
            _ensure_column(conn, "goals", "capability_policy_json", "TEXT NOT NULL DEFAULT '{}'")
            _ensure_column(conn, "goals", "source_manifest_json", "TEXT NOT NULL DEFAULT '{}'")
            _ensure_column(conn, "goals", "summary_json", "TEXT NOT NULL DEFAULT '{}'")
            _ensure_column(conn, "goals", "metadata_json", "TEXT NOT NULL DEFAULT '{}'")
            _ensure_column(conn, "goals", "latest_error", "TEXT")
            _ensure_column(conn, "goal_attempts", "trigger", "TEXT")
            _ensure_column(conn, "goal_attempts", "agent_run_id", "TEXT")
            _ensure_column(conn, "goal_attempts", "summary_json", "TEXT NOT NULL DEFAULT '{}'")
            _ensure_column(conn, "goal_attempts", "metadata_json", "TEXT NOT NULL DEFAULT '{}'")
            _ensure_column(conn, "goal_attempts", "latest_error", "TEXT")
            _ensure_column(conn, "goal_attempts", "started_at", "TEXT")
            _ensure_column(conn, "goal_attempts", "finished_at", "TEXT")
            _ensure_column(conn, "goal_leases", "takeover_count", "INTEGER NOT NULL DEFAULT 0")
            _ensure_column(conn, "goal_leases", "metadata_json", "TEXT NOT NULL DEFAULT '{}'")
            _ensure_column(
                conn,
                "goal_operator_controls",
                "blocked_domains_json",
                "TEXT NOT NULL DEFAULT '[]'",
            )
            _ensure_column(conn, "goal_audit_findings", "details_json", "TEXT NOT NULL DEFAULT '{}'")
            _ensure_column(conn, "goal_audit_findings", "resolved_at", "TEXT")
            _ensure_column(conn, "goal_audit_findings", "closed_at", "TEXT")
            _ensure_column(conn, "goal_worker_generations", "attempt_id", "TEXT")
            _ensure_column(conn, "goal_worker_generations", "agent_run_id", "TEXT")
            _ensure_column(conn, "goal_worker_generations", "generation_index", "INTEGER NOT NULL DEFAULT 0")
            _ensure_column(conn, "goal_worker_generations", "status", "TEXT NOT NULL DEFAULT 'created'")
            _ensure_column(conn, "goal_worker_generations", "rollover_reason", "TEXT")
            _ensure_column(conn, "goal_worker_generations", "parent_generation_id", "INTEGER")
            _ensure_column(conn, "goal_worker_generations", "resume_source_snapshot_id", "INTEGER")
            _ensure_column(conn, "goal_worker_generations", "metadata_json", "TEXT NOT NULL DEFAULT '{}'")
            _ensure_column(conn, "goal_worker_generations", "started_at", "TEXT")
            _ensure_column(conn, "goal_worker_generations", "finished_at", "TEXT")
            _ensure_column(conn, "goal_worker_generations", "created_at", "TEXT")
            _ensure_column(conn, "goal_worker_generations", "updated_at", "TEXT")
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_goal_worker_generations_goal_id_created_at
                ON goal_worker_generations(goal_id, created_at DESC, id DESC)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_goal_worker_generations_attempt_id_generation_index
                ON goal_worker_generations(attempt_id, generation_index DESC, id DESC)
                """
            )
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
                    "SELECT started_at, finished_at, final_answer, metadata_json FROM task_runs WHERE id=?",
                    (task_id,),
                ).fetchone()
                current_started_at = existing[0] if existing else None
                current_finished_at = existing[1] if existing else None
                current_final_answer = existing[2] if existing else None
                current_metadata_json = existing[3] if existing else None
                loaded_metadata = json.loads(current_metadata_json or "{}")
                metadata = loaded_metadata if isinstance(loaded_metadata, dict) else {}
                if isinstance(metadata.get("delegated_subagent"), dict):
                    delegated_subagent = dict(metadata["delegated_subagent"])
                    delegated_subagent["status"] = status
                    metadata["delegated_subagent"] = delegated_subagent
                conn.execute(
                    """
                    UPDATE task_runs
                    SET status=?,
                        error=?,
                        permission_override_json=?,
                        metadata_json=?,
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
                        json.dumps(metadata, ensure_ascii=False),
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

    async def update_approval_request_metadata(
        self,
        approval_id: str,
        *,
        metadata: dict[str, Any],
    ) -> dict[str, Any] | None:
        await self.initialize()
        now = _now_iso()

        def _op() -> None:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """
                    UPDATE approval_requests
                    SET metadata_json=?, updated_at=?
                    WHERE id=?
                    """,
                    (
                        json.dumps(metadata or {}, ensure_ascii=False),
                        now,
                        approval_id,
                    ),
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

    async def create_goal(
        self,
        *,
        goal_id: str,
        objective: str,
        title: str | None = None,
        goal_type: str | None = None,
        execution_mode: str = _DEFAULT_GOAL_EXECUTION_MODE,
        protocol_id: str | None = None,
        topic: str | None = None,
        project_id: str | None = None,
        workspace_dir: str | None = None,
        run_policy: dict[str, Any] | None = None,
        capability_policy: dict[str, Any] | None = None,
        source_manifest: dict[str, Any] | None = None,
        summary: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        latest_error: str | None = None,
    ) -> dict[str, Any]:
        await self.initialize()
        now = _now_iso()
        normalized_execution_mode = _normalize_goal_execution_mode(execution_mode)
        normalized_protocol_id = _normalize_goal_protocol_id(
            normalized_execution_mode,
            protocol_id,
        )

        def _op() -> None:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO goals (
                        id, objective, title, goal_type, execution_mode, protocol_id, topic, project_id,
                        workspace_dir, status, current_attempt_id, run_policy_json,
                        capability_policy_json, source_manifest_json, summary_json,
                        metadata_json, latest_error, started_at, finished_at, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        goal_id,
                        objective,
                        title,
                        goal_type,
                        normalized_execution_mode,
                        normalized_protocol_id,
                        topic,
                        project_id,
                        workspace_dir,
                        "created",
                        None,
                        json.dumps(run_policy or {}, ensure_ascii=False),
                        json.dumps(capability_policy or {}, ensure_ascii=False),
                        json.dumps(source_manifest or {}, ensure_ascii=False),
                        json.dumps(summary or {}, ensure_ascii=False),
                        json.dumps(metadata or {}, ensure_ascii=False),
                        latest_error,
                        None,
                        None,
                        now,
                        now,
                    ),
                )
                conn.commit()

        await asyncio.to_thread(_op)
        return await self.get_goal(goal_id) or {}

    async def create_goal_attempt(
        self,
        *,
        attempt_id: str,
        goal_id: str,
        attempt_index: int,
        status: str,
        trigger: str | None = None,
        agent_run_id: str | None = None,
        summary: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        latest_error: str | None = None,
    ) -> dict[str, Any]:
        await self.initialize()
        now = _now_iso()

        def _op() -> None:
            started_at = now if status in {"queued", "running"} else None
            finished_at = now if status in {"completed", "failed", "cancelled"} else None
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO goal_attempts (
                        id, goal_id, attempt_index, status, trigger, agent_run_id,
                        summary_json, metadata_json, latest_error, started_at, finished_at,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        attempt_id,
                        goal_id,
                        attempt_index,
                        status,
                        trigger,
                        agent_run_id,
                        json.dumps(summary or {}, ensure_ascii=False),
                        json.dumps(metadata or {}, ensure_ascii=False),
                        latest_error,
                        started_at,
                        finished_at,
                        now,
                        now,
                    ),
                )
                conn.commit()

        await asyncio.to_thread(_op)
        return await self.get_goal_attempt(attempt_id) or {}

    async def get_goal_attempt(self, attempt_id: str) -> dict[str, Any] | None:
        await self.initialize()

        def _op() -> dict[str, Any] | None:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT * FROM goal_attempts WHERE id=?",
                    (attempt_id,),
                ).fetchone()
            return _row_to_goal_attempt_payload(row)

        return await asyncio.to_thread(_op)

    async def update_goal_status(
        self,
        goal_id: str,
        status: str,
        *,
        latest_error: str | None | object = _UNSET,
        current_attempt_id: str | None | object = _UNSET,
        reset_started_at: bool = False,
        reset_finished_at: bool = False,
    ) -> None:
        await self.initialize()
        now = _now_iso()

        def _op() -> None:
            with sqlite3.connect(self._db_path) as conn:
                existing = conn.execute(
                    """
                    SELECT started_at, finished_at, latest_error, current_attempt_id
                    FROM goals
                    WHERE id=?
                    """,
                    (goal_id,),
                ).fetchone()
                if existing is None:
                    return
                current_started_at = None if reset_started_at else existing[0]
                current_finished_at = None if reset_finished_at else existing[1]
                current_latest_error = existing[2]
                current_current_attempt_id = existing[3]
                started_at = now if status in {"queued", "running"} and not current_started_at else current_started_at
                finished_at = (
                    now
                    if status in {"completed", "failed", "cancelled"}
                    else None
                )
                conn.execute(
                    """
                    UPDATE goals
                    SET status=?,
                        latest_error=?,
                        current_attempt_id=?,
                        started_at=?,
                        finished_at=?,
                        updated_at=?
                    WHERE id=?
                    """,
                    (
                        status,
                        current_latest_error if latest_error is _UNSET else latest_error,
                        current_current_attempt_id if current_attempt_id is _UNSET else current_attempt_id,
                        started_at,
                        finished_at if finished_at is not None else current_finished_at,
                        now,
                        goal_id,
                    ),
                )
                conn.commit()

        await asyncio.to_thread(_op)

    async def update_goal_attempt_status(
        self,
        attempt_id: str,
        status: str,
        *,
        latest_error: str | None | object = _UNSET,
        agent_run_id: str | None | object = _UNSET,
        summary: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        reset_started_at: bool = False,
        reset_finished_at: bool = False,
    ) -> None:
        await self.initialize()
        now = _now_iso()

        def _op() -> None:
            with sqlite3.connect(self._db_path) as conn:
                existing = conn.execute(
                    """
                    SELECT started_at, finished_at, latest_error, agent_run_id, summary_json, metadata_json
                    FROM goal_attempts
                    WHERE id=?
                    """,
                    (attempt_id,),
                ).fetchone()
                if existing is None:
                    return
                current_started_at = None if reset_started_at else existing[0]
                current_finished_at = None if reset_finished_at else existing[1]
                current_latest_error = existing[2]
                current_agent_run_id = existing[3]
                current_summary = json.loads(str(existing[4] or "{}"))
                current_metadata = json.loads(str(existing[5] or "{}"))
                started_at = now if status in {"queued", "running"} and not current_started_at else current_started_at
                finished_at = (
                    now
                    if status in {"completed", "failed", "cancelled", "paused"}
                    else None
                )
                conn.execute(
                    """
                    UPDATE goal_attempts
                    SET status=?,
                        agent_run_id=?,
                        summary_json=?,
                        metadata_json=?,
                        latest_error=?,
                        started_at=?,
                        finished_at=?,
                        updated_at=?
                    WHERE id=?
                    """,
                    (
                        status,
                        current_agent_run_id if agent_run_id is _UNSET else agent_run_id,
                        json.dumps(current_summary if summary is None else summary, ensure_ascii=False),
                        json.dumps(current_metadata if metadata is None else metadata, ensure_ascii=False),
                        current_latest_error if latest_error is _UNSET else latest_error,
                        started_at,
                        finished_at if finished_at is not None else current_finished_at,
                        now,
                        attempt_id,
                    ),
                )
                conn.commit()

        await asyncio.to_thread(_op)

    async def update_goal_projection(
        self,
        *,
        goal_id: str,
        goal_status: str,
        attempt_id: str,
        attempt_status: str,
        latest_error: str | None | object = _UNSET,
        current_attempt_id: str | None | object = _UNSET,
        agent_run_id: str | None | object = _UNSET,
        attempt_summary: dict[str, Any] | None = None,
        attempt_metadata: dict[str, Any] | None = None,
        reset_goal_started_at: bool = False,
        reset_goal_finished_at: bool = False,
        reset_attempt_started_at: bool = False,
        reset_attempt_finished_at: bool = False,
    ) -> None:
        await self.initialize()
        now = _now_iso()

        def _op() -> None:
            with sqlite3.connect(self._db_path) as conn:
                goal_existing = conn.execute(
                    """
                    SELECT started_at, finished_at, latest_error, current_attempt_id
                    FROM goals
                    WHERE id=?
                    """,
                    (goal_id,),
                ).fetchone()
                attempt_existing = conn.execute(
                    """
                    SELECT started_at, finished_at, latest_error, agent_run_id, summary_json, metadata_json
                    FROM goal_attempts
                    WHERE id=?
                    """,
                    (attempt_id,),
                ).fetchone()
                if goal_existing is None or attempt_existing is None:
                    return

                goal_started_at = None if reset_goal_started_at else goal_existing[0]
                goal_finished_at = None if reset_goal_finished_at else goal_existing[1]
                goal_latest_error = goal_existing[2]
                goal_current_attempt_id = goal_existing[3]
                next_goal_started_at = (
                    now
                    if goal_status in {"queued", "running"} and not goal_started_at
                    else goal_started_at
                )
                next_goal_finished_at = (
                    now
                    if goal_status in {"completed", "failed", "cancelled"}
                    else goal_finished_at
                )

                attempt_started_at = None if reset_attempt_started_at else attempt_existing[0]
                attempt_finished_at = None if reset_attempt_finished_at else attempt_existing[1]
                attempt_latest_error = attempt_existing[2]
                attempt_agent_run_id = attempt_existing[3]
                attempt_current_summary = json.loads(str(attempt_existing[4] or "{}"))
                attempt_current_metadata = json.loads(str(attempt_existing[5] or "{}"))
                next_attempt_started_at = (
                    now
                    if attempt_status in {"queued", "running"} and not attempt_started_at
                    else attempt_started_at
                )
                next_attempt_finished_at = (
                    now
                    if attempt_status in {"completed", "failed", "cancelled", "paused"}
                    else attempt_finished_at
                )

                conn.execute(
                    """
                    UPDATE goal_attempts
                    SET status=?,
                        agent_run_id=?,
                        summary_json=?,
                        metadata_json=?,
                        latest_error=?,
                        started_at=?,
                        finished_at=?,
                        updated_at=?
                    WHERE id=?
                    """,
                    (
                        attempt_status,
                        attempt_agent_run_id if agent_run_id is _UNSET else agent_run_id,
                        json.dumps(
                            attempt_current_summary if attempt_summary is None else attempt_summary,
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            attempt_current_metadata if attempt_metadata is None else attempt_metadata,
                            ensure_ascii=False,
                        ),
                        attempt_latest_error if latest_error is _UNSET else latest_error,
                        next_attempt_started_at,
                        next_attempt_finished_at,
                        now,
                        attempt_id,
                    ),
                )
                conn.execute(
                    """
                    UPDATE goals
                    SET status=?,
                        latest_error=?,
                        current_attempt_id=?,
                        started_at=?,
                        finished_at=?,
                        updated_at=?
                    WHERE id=?
                    """,
                    (
                        goal_status,
                        goal_latest_error if latest_error is _UNSET else latest_error,
                        goal_current_attempt_id if current_attempt_id is _UNSET else current_attempt_id,
                        next_goal_started_at,
                        next_goal_finished_at,
                        now,
                        goal_id,
                    ),
                )
                conn.commit()

        await asyncio.to_thread(_op)

    async def claim_goal_attempt_agent_run_id(
        self,
        attempt_id: str,
        agent_run_id: str,
    ) -> dict[str, Any] | None:
        await self.initialize()
        now = _now_iso()

        def _op() -> None:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                existing = conn.execute(
                    """
                    SELECT agent_run_id, summary_json
                    FROM goal_attempts
                    WHERE id=?
                    """,
                    (attempt_id,),
                ).fetchone()
                if existing is None:
                    return
                current_agent_run_id = str(existing["agent_run_id"] or "").strip()
                if current_agent_run_id:
                    return
                summary = json.loads(str(existing["summary_json"] or "{}"))
                summary["agent_run_id"] = agent_run_id
                cursor = conn.execute(
                    """
                    UPDATE goal_attempts
                    SET agent_run_id=?,
                        summary_json=?,
                        updated_at=?
                    WHERE id=?
                      AND (agent_run_id IS NULL OR agent_run_id='')
                    """,
                    (
                        agent_run_id,
                        json.dumps(summary, ensure_ascii=False),
                        now,
                        attempt_id,
                    ),
                )
                if cursor.rowcount:
                    conn.commit()

        await asyncio.to_thread(_op)
        return await self.get_goal_attempt(attempt_id)

    async def update_goal_metadata(
        self,
        goal_id: str,
        *,
        summary: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        project_id: str | None | object = _UNSET,
        workspace_dir: str | None | object = _UNSET,
        latest_error: str | None | object = _UNSET,
        current_attempt_id: str | None | object = _UNSET,
    ) -> None:
        await self.initialize()
        now = _now_iso()

        def _op() -> None:
            with sqlite3.connect(self._db_path) as conn:
                existing = conn.execute(
                    """
                    SELECT summary_json, metadata_json, latest_error, project_id, workspace_dir, current_attempt_id
                    FROM goals
                    WHERE id=?
                    """,
                    (goal_id,),
                ).fetchone()
                if existing is None:
                    return
                current_summary = json.loads(str(existing[0] or "{}"))
                current_metadata = json.loads(str(existing[1] or "{}"))
                current_latest_error = existing[2]
                current_project_id = existing[3]
                current_workspace_dir = existing[4]
                current_current_attempt_id = existing[5]
                conn.execute(
                    """
                    UPDATE goals
                    SET summary_json=?,
                        metadata_json=?,
                        project_id=?,
                        workspace_dir=?,
                        latest_error=?,
                        current_attempt_id=?,
                        updated_at=?
                    WHERE id=?
                    """,
                    (
                        json.dumps(current_summary if summary is None else summary, ensure_ascii=False),
                        json.dumps(current_metadata if metadata is None else metadata, ensure_ascii=False),
                        current_project_id if project_id is _UNSET else project_id,
                        current_workspace_dir if workspace_dir is _UNSET else workspace_dir,
                        current_latest_error if latest_error is _UNSET else latest_error,
                        current_current_attempt_id if current_attempt_id is _UNSET else current_attempt_id,
                        now,
                        goal_id,
                    ),
                )
                conn.commit()

        await asyncio.to_thread(_op)

    async def get_goal(self, goal_id: str) -> dict[str, Any] | None:
        await self.initialize()

        def _op() -> dict[str, Any] | None:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute("SELECT * FROM goals WHERE id=?", (goal_id,)).fetchone()
                if row is None:
                    return None
                payload = _row_to_goal_payload(row)
                payload["attempts"] = _load_goal_attempts(conn, goal_id)
                return payload

        return await asyncio.to_thread(_op)

    async def list_goals(self) -> list[dict[str, Any]]:
        await self.initialize()

        def _op() -> list[dict[str, Any]]:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM goals ORDER BY datetime(created_at) DESC"
                ).fetchall()
                output: list[dict[str, Any]] = []
                for row in rows:
                    payload = _row_to_goal_payload(row)
                    payload["attempts"] = _load_goal_attempts(conn, str(payload["id"]))
                    output.append(payload)
                return output

        return await asyncio.to_thread(_op)

    async def create_goal_checkpoint(
        self,
        *,
        goal_id: str,
        attempt_id: str | None = None,
        agent_run_id: str | None = None,
        checkpoint_index: int | None = None,
        stage: str | None = None,
        source: str,
        payload: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        captured_at: str | None = None,
    ) -> dict[str, Any]:
        await self.initialize()
        now = _now_iso()
        effective_captured_at = captured_at or now

        def _op() -> int:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO goal_checkpoints (
                        goal_id, attempt_id, agent_run_id, checkpoint_index, stage, source,
                        payload_json, metadata_json, captured_at, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        goal_id,
                        attempt_id,
                        agent_run_id,
                        checkpoint_index,
                        stage,
                        source,
                        json.dumps(payload or {}, ensure_ascii=False),
                        json.dumps(metadata or {}, ensure_ascii=False),
                        effective_captured_at,
                        now,
                        now,
                    ),
                )
                conn.commit()
                return int(cursor.lastrowid)

        checkpoint_id = await asyncio.to_thread(_op)
        return await self.get_goal_checkpoint(checkpoint_id) or {}

    async def get_goal_checkpoint(self, checkpoint_id: int) -> dict[str, Any] | None:
        await self.initialize()

        def _op() -> dict[str, Any] | None:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT * FROM goal_checkpoints WHERE id=?",
                    (checkpoint_id,),
                ).fetchone()
            return _row_to_goal_checkpoint_payload(row)

        return await asyncio.to_thread(_op)

    async def list_goal_checkpoints(
        self,
        goal_id: str,
        *,
        attempt_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        await self.initialize()
        effective_limit = int(limit) if isinstance(limit, int) and limit > 0 else None

        def _op() -> list[dict[str, Any]]:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                params: list[Any] = [goal_id]
                query_parts = ["SELECT * FROM goal_checkpoints WHERE goal_id=?"]
                if attempt_id is not None:
                    query_parts.append("AND attempt_id=?")
                    params.append(attempt_id)
                query_parts.append("ORDER BY datetime(captured_at) DESC, id DESC")
                if effective_limit is not None:
                    query_parts.append("LIMIT ?")
                    params.append(effective_limit)
                rows = conn.execute(" ".join(query_parts), tuple(params)).fetchall()
            return [
                payload
                for payload in (_row_to_goal_checkpoint_payload(row) for row in rows)
                if payload is not None
            ]

        return await asyncio.to_thread(_op)

    async def get_latest_goal_checkpoint(
        self,
        goal_id: str,
        *,
        attempt_id: str | None = None,
    ) -> dict[str, Any] | None:
        checkpoints = await self.list_goal_checkpoints(goal_id, attempt_id=attempt_id, limit=1)
        return checkpoints[0] if checkpoints else None

    async def create_goal_memory_snapshot(
        self,
        *,
        goal_id: str,
        attempt_id: str | None = None,
        checkpoint_id: int | None = None,
        snapshot_kind: str,
        snapshot: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        captured_at: str | None = None,
    ) -> dict[str, Any]:
        await self.initialize()
        now = _now_iso()
        effective_captured_at = captured_at or now

        def _op() -> int:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO goal_memory_snapshots (
                        goal_id, attempt_id, checkpoint_id, snapshot_kind, snapshot_json,
                        metadata_json, captured_at, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        goal_id,
                        attempt_id,
                        checkpoint_id,
                        snapshot_kind,
                        json.dumps(snapshot or {}, ensure_ascii=False),
                        json.dumps(metadata or {}, ensure_ascii=False),
                        effective_captured_at,
                        now,
                        now,
                    ),
                )
                conn.commit()
                return int(cursor.lastrowid)

        snapshot_id = await asyncio.to_thread(_op)
        return await self.get_goal_memory_snapshot(snapshot_id) or {}

    async def get_goal_memory_snapshot(self, snapshot_id: int) -> dict[str, Any] | None:
        await self.initialize()

        def _op() -> dict[str, Any] | None:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT * FROM goal_memory_snapshots WHERE id=?",
                    (snapshot_id,),
                ).fetchone()
            return _row_to_goal_memory_snapshot_payload(row)

        return await asyncio.to_thread(_op)

    async def list_goal_memory_snapshots(
        self,
        goal_id: str,
        *,
        attempt_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        await self.initialize()
        effective_limit = int(limit) if isinstance(limit, int) and limit > 0 else None

        def _op() -> list[dict[str, Any]]:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                params: list[Any] = [goal_id]
                query_parts = ["SELECT * FROM goal_memory_snapshots WHERE goal_id=?"]
                if attempt_id is not None:
                    query_parts.append("AND attempt_id=?")
                    params.append(attempt_id)
                query_parts.append("ORDER BY datetime(captured_at) DESC, id DESC")
                if effective_limit is not None:
                    query_parts.append("LIMIT ?")
                    params.append(effective_limit)
                rows = conn.execute(" ".join(query_parts), tuple(params)).fetchall()
            return [
                payload
                for payload in (_row_to_goal_memory_snapshot_payload(row) for row in rows)
                if payload is not None
            ]

        return await asyncio.to_thread(_op)

    async def get_latest_goal_memory_snapshot(
        self,
        goal_id: str,
        *,
        attempt_id: str | None = None,
    ) -> dict[str, Any] | None:
        snapshots = await self.list_goal_memory_snapshots(
            goal_id,
            attempt_id=attempt_id,
            limit=1,
        )
        return snapshots[0] if snapshots else None

    async def create_goal_worker_generation(
        self,
        *,
        goal_id: str,
        attempt_id: str | None = None,
        agent_run_id: str | None = None,
        generation_index: int,
        status: str,
        rollover_reason: str | None = None,
        parent_generation_id: int | None = None,
        resume_source_snapshot_id: int | None = None,
        metadata: dict[str, Any] | None = None,
        started_at: str | None = None,
        finished_at: str | None = None,
    ) -> dict[str, Any]:
        await self.initialize()
        now = _now_iso()
        effective_started_at = started_at or now
        effective_finished_at = (
            finished_at
            if finished_at is not None
            else now
            if status in _GOAL_WORKER_GENERATION_TERMINAL_STATUSES
            else None
        )

        def _op() -> int:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO goal_worker_generations (
                        goal_id, attempt_id, agent_run_id, generation_index, status,
                        rollover_reason, parent_generation_id, resume_source_snapshot_id,
                        metadata_json, started_at, finished_at, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        goal_id,
                        attempt_id,
                        agent_run_id,
                        generation_index,
                        status,
                        rollover_reason,
                        parent_generation_id,
                        resume_source_snapshot_id,
                        json.dumps(metadata or {}, ensure_ascii=False),
                        effective_started_at,
                        effective_finished_at,
                        now,
                        now,
                    ),
                )
                conn.commit()
                return int(cursor.lastrowid)

        generation_id = await asyncio.to_thread(_op)
        return await self.get_goal_worker_generation(generation_id) or {}

    async def update_goal_worker_generation(
        self,
        generation_id: int,
        *,
        status: str | object = _UNSET,
        agent_run_id: str | None | object = _UNSET,
        rollover_reason: str | None | object = _UNSET,
        parent_generation_id: int | None | object = _UNSET,
        resume_source_snapshot_id: int | None | object = _UNSET,
        metadata: dict[str, Any] | None = None,
        started_at: str | None | object = _UNSET,
        finished_at: str | None | object = _UNSET,
    ) -> None:
        await self.initialize()
        now = _now_iso()

        def _op() -> None:
            with sqlite3.connect(self._db_path) as conn:
                existing = conn.execute(
                    """
                    SELECT status, agent_run_id, rollover_reason, parent_generation_id,
                           resume_source_snapshot_id, metadata_json, started_at, finished_at
                    FROM goal_worker_generations
                    WHERE id=?
                    """,
                    (generation_id,),
                ).fetchone()
                if existing is None:
                    return
                current_status = existing[0]
                current_agent_run_id = existing[1]
                current_rollover_reason = existing[2]
                current_parent_generation_id = existing[3]
                current_resume_source_snapshot_id = existing[4]
                current_metadata = json.loads(str(existing[5] or "{}"))
                current_started_at = existing[6]
                current_finished_at = existing[7]
                next_status = current_status if status is _UNSET else status
                next_started_at = current_started_at if started_at is _UNSET else started_at
                if next_status == "running" and not next_started_at:
                    next_started_at = now
                if finished_at is _UNSET:
                    next_finished_at = (
                        now
                        if next_status in _GOAL_WORKER_GENERATION_TERMINAL_STATUSES
                        and not current_finished_at
                        else current_finished_at
                    )
                else:
                    next_finished_at = finished_at
                conn.execute(
                    """
                    UPDATE goal_worker_generations
                    SET status=?,
                        agent_run_id=?,
                        rollover_reason=?,
                        parent_generation_id=?,
                        resume_source_snapshot_id=?,
                        metadata_json=?,
                        started_at=?,
                        finished_at=?,
                        updated_at=?
                    WHERE id=?
                    """,
                    (
                        next_status,
                        current_agent_run_id if agent_run_id is _UNSET else agent_run_id,
                        current_rollover_reason if rollover_reason is _UNSET else rollover_reason,
                        (
                            current_parent_generation_id
                            if parent_generation_id is _UNSET
                            else parent_generation_id
                        ),
                        (
                            current_resume_source_snapshot_id
                            if resume_source_snapshot_id is _UNSET
                            else resume_source_snapshot_id
                        ),
                        json.dumps(current_metadata if metadata is None else metadata, ensure_ascii=False),
                        next_started_at,
                        next_finished_at,
                        now,
                        generation_id,
                    ),
                )
                conn.commit()

        await asyncio.to_thread(_op)

    async def get_goal_worker_generation(
        self,
        generation_id: int,
    ) -> dict[str, Any] | None:
        await self.initialize()

        def _op() -> dict[str, Any] | None:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT * FROM goal_worker_generations WHERE id=?",
                    (generation_id,),
                ).fetchone()
            return _row_to_goal_worker_generation_payload(row)

        return await asyncio.to_thread(_op)

    async def list_goal_worker_generations(
        self,
        goal_id: str,
        *,
        attempt_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        await self.initialize()
        effective_limit = int(limit) if isinstance(limit, int) and limit > 0 else None

        def _op() -> list[dict[str, Any]]:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                params: list[Any] = [goal_id]
                query_parts = ["SELECT * FROM goal_worker_generations WHERE goal_id=?"]
                if attempt_id is not None:
                    query_parts.append("AND attempt_id=?")
                    params.append(attempt_id)
                    query_parts.append("ORDER BY generation_index DESC, id DESC")
                else:
                    query_parts.append("ORDER BY datetime(created_at) DESC, id DESC")
                if effective_limit is not None:
                    query_parts.append("LIMIT ?")
                    params.append(effective_limit)
                rows = conn.execute(" ".join(query_parts), tuple(params)).fetchall()
            return [
                payload
                for payload in (_row_to_goal_worker_generation_payload(row) for row in rows)
                if payload is not None
            ]

        return await asyncio.to_thread(_op)

    async def get_latest_goal_worker_generation(
        self,
        goal_id: str,
        *,
        attempt_id: str | None = None,
    ) -> dict[str, Any] | None:
        generations = await self.list_goal_worker_generations(
            goal_id,
            attempt_id=attempt_id,
            limit=1,
        )
        return generations[0] if generations else None

    async def upsert_goal_lease(
        self,
        *,
        goal_id: str,
        owner_id: str,
        metadata: dict[str, Any] | None = None,
        acquired_at: str | None = None,
        heartbeat_at: str | None = None,
        expires_at: str | None = None,
        force_takeover: bool = False,
    ) -> dict[str, Any] | None:
        await self.initialize()
        now = _now_iso()
        effective_acquired_at = acquired_at or now
        effective_heartbeat_at = heartbeat_at or now
        effective_expires_at = expires_at or now

        def _op() -> None:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                existing = conn.execute(
                    """
                    SELECT owner_id, acquired_at, expires_at, takeover_count, metadata_json
                    FROM goal_leases
                    WHERE goal_id=?
                    """,
                    (goal_id,),
                ).fetchone()
                if existing is None:
                    conn.execute(
                        """
                        INSERT INTO goal_leases (
                            goal_id, owner_id, acquired_at, heartbeat_at, expires_at,
                            takeover_count, metadata_json, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            goal_id,
                            owner_id,
                            effective_acquired_at,
                            effective_heartbeat_at,
                            effective_expires_at,
                            0,
                            json.dumps(metadata or {}, ensure_ascii=False),
                            now,
                        ),
                    )
                    conn.commit()
                    return

                previous_owner_id = str(existing["owner_id"] or "")
                previous_acquired_at = str(existing["acquired_at"] or effective_acquired_at)
                previous_expires_at = str(existing["expires_at"] or "")
                previous_takeover_count = int(existing["takeover_count"] or 0)
                if (
                    previous_owner_id
                    and previous_owner_id != owner_id
                    and not force_takeover
                    and not _goal_lease_is_stale(previous_expires_at, now=now)
                ):
                    return
                takeover_count = previous_takeover_count
                next_acquired_at = previous_acquired_at
                if force_takeover or previous_owner_id != owner_id:
                    takeover_count += 1 if previous_owner_id else 0
                    next_acquired_at = effective_acquired_at
                existing_metadata = json.loads(str(existing["metadata_json"] or "{}"))
                next_metadata = metadata if metadata is not None else existing_metadata
                conn.execute(
                    """
                    UPDATE goal_leases
                    SET owner_id=?,
                        acquired_at=?,
                        heartbeat_at=?,
                        expires_at=?,
                        takeover_count=?,
                        metadata_json=?,
                        updated_at=?
                    WHERE goal_id=?
                    """,
                    (
                        owner_id,
                        next_acquired_at,
                        effective_heartbeat_at,
                        effective_expires_at,
                        takeover_count,
                        json.dumps(next_metadata or {}, ensure_ascii=False),
                        now,
                        goal_id,
                    ),
                )
                conn.commit()

        await asyncio.to_thread(_op)
        return await self.get_goal_lease(goal_id)

    async def get_goal_lease(self, goal_id: str) -> dict[str, Any] | None:
        await self.initialize()

        def _op() -> dict[str, Any] | None:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT * FROM goal_leases WHERE goal_id=?",
                    (goal_id,),
                ).fetchone()
            return _row_to_goal_lease_payload(row)

        return await asyncio.to_thread(_op)

    async def list_goal_leases(self, *, owner_id: str | None = None) -> list[dict[str, Any]]:
        await self.initialize()

        def _op() -> list[dict[str, Any]]:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                if owner_id is None:
                    rows = conn.execute(
                        "SELECT * FROM goal_leases ORDER BY datetime(updated_at) DESC"
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT * FROM goal_leases
                        WHERE owner_id=?
                        ORDER BY datetime(updated_at) DESC
                        """,
                        (owner_id,),
                    ).fetchall()
            leases: list[dict[str, Any]] = []
            for row in rows:
                payload = _row_to_goal_lease_payload(row)
                if payload is not None:
                    leases.append(payload)
            return leases

        return await asyncio.to_thread(_op)

    async def delete_goal_lease(self, goal_id: str) -> None:
        await self.initialize()

        def _op() -> None:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("DELETE FROM goal_leases WHERE goal_id=?", (goal_id,))
                conn.commit()

        await asyncio.to_thread(_op)

    async def delete_goal_leases_for_owner(self, owner_id: str) -> None:
        await self.initialize()

        def _op() -> None:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("DELETE FROM goal_leases WHERE owner_id=?", (owner_id,))
                conn.commit()

        await asyncio.to_thread(_op)

    async def upsert_goal_audit_finding(
        self,
        *,
        goal_id: str,
        finding_code: str,
        summary: str,
        severity: str = "warning",
        details: dict[str, Any] | None = None,
        status: str = "open",
    ) -> dict[str, Any]:
        await self.initialize()
        now = _now_iso()
        normalized_status = str(status or "open").strip().lower() or "open"

        def _op() -> int:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                existing = conn.execute(
                    """
                    SELECT id, resolved_at, closed_at
                    FROM goal_audit_findings
                    WHERE goal_id=? AND finding_code=? AND status=?
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (goal_id, finding_code, normalized_status),
                ).fetchone()
                resolved_at, closed_at = _goal_audit_finding_resolution_timestamps(
                    status=normalized_status,
                    now=now,
                    resolved_at=(
                        str(existing["resolved_at"])
                        if existing is not None and existing["resolved_at"] is not None
                        else None
                    ),
                    closed_at=(
                        str(existing["closed_at"])
                        if existing is not None and existing["closed_at"] is not None
                        else None
                    ),
                )
                if existing is None:
                    cursor = conn.execute(
                        """
                        INSERT INTO goal_audit_findings (
                            goal_id, finding_code, status, severity, summary, details_json,
                            resolved_at, closed_at, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            goal_id,
                            finding_code,
                            normalized_status,
                            severity,
                            summary,
                            json.dumps(details or {}, ensure_ascii=False),
                            resolved_at,
                            closed_at,
                            now,
                            now,
                        ),
                    )
                    conn.commit()
                    return int(cursor.lastrowid)

                finding_id = int(existing["id"])
                conn.execute(
                    """
                    UPDATE goal_audit_findings
                    SET severity=?,
                        summary=?,
                        details_json=?,
                        resolved_at=?,
                        closed_at=?,
                        updated_at=?
                    WHERE id=?
                    """,
                    (
                        severity,
                        summary,
                        json.dumps(details or {}, ensure_ascii=False),
                        resolved_at,
                        closed_at,
                        now,
                        finding_id,
                    ),
                )
                conn.commit()
                return finding_id

        finding_id = await asyncio.to_thread(_op)
        result = await self.get_goal_audit_finding(finding_id)
        return result or {}

    async def update_goal_audit_finding_status(
        self,
        finding_id: int,
        *,
        status: str,
    ) -> dict[str, Any] | None:
        await self.initialize()
        now = _now_iso()
        normalized_status = str(status or "open").strip().lower() or "open"

        def _op() -> bool:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                existing = conn.execute(
                    """
                    SELECT resolved_at, closed_at
                    FROM goal_audit_findings
                    WHERE id=?
                    """,
                    (finding_id,),
                ).fetchone()
                if existing is None:
                    return False
                resolved_at, closed_at = _goal_audit_finding_resolution_timestamps(
                    status=normalized_status,
                    now=now,
                    resolved_at=(
                        str(existing["resolved_at"])
                        if existing["resolved_at"] is not None
                        else None
                    ),
                    closed_at=(
                        str(existing["closed_at"])
                        if existing["closed_at"] is not None
                        else None
                    ),
                )
                conn.execute(
                    """
                    UPDATE goal_audit_findings
                    SET status=?,
                        resolved_at=?,
                        closed_at=?,
                        updated_at=?
                    WHERE id=?
                    """,
                    (
                        normalized_status,
                        resolved_at,
                        closed_at,
                        now,
                        finding_id,
                    ),
                )
                conn.commit()
                return True

        updated = await asyncio.to_thread(_op)
        if not updated:
            return None
        return await self.get_goal_audit_finding(finding_id)

    async def resolve_goal_audit_finding(self, finding_id: int) -> dict[str, Any] | None:
        return await self.update_goal_audit_finding_status(
            finding_id,
            status="resolved",
        )

    async def close_goal_audit_finding(self, finding_id: int) -> dict[str, Any] | None:
        return await self.update_goal_audit_finding_status(
            finding_id,
            status="closed",
        )

    async def get_goal_audit_finding(self, finding_id: int) -> dict[str, Any] | None:
        await self.initialize()

        def _op() -> dict[str, Any] | None:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT * FROM goal_audit_findings WHERE id=?",
                    (finding_id,),
                ).fetchone()
            return _row_to_goal_audit_finding_payload(row)

        return await asyncio.to_thread(_op)

    async def list_goal_audit_findings(
        self,
        goal_id: str,
        *,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        await self.initialize()

        def _op() -> list[dict[str, Any]]:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                if status is None:
                    rows = conn.execute(
                        """
                        SELECT * FROM goal_audit_findings
                        WHERE goal_id=?
                        ORDER BY id DESC
                        """,
                        (goal_id,),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT * FROM goal_audit_findings
                        WHERE goal_id=? AND status=?
                        ORDER BY id DESC
                        """,
                        (goal_id, status),
                    ).fetchall()
            findings: list[dict[str, Any]] = []
            for row in rows:
                payload = _row_to_goal_audit_finding_payload(row)
                if payload is not None:
                    findings.append(payload)
            return findings

        return await asyncio.to_thread(_op)

    async def get_goal_operator_controls(self, *, scope: str = "global") -> dict[str, Any] | None:
        await self.initialize()

        def _op() -> dict[str, Any] | None:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT * FROM goal_operator_controls WHERE scope=?",
                    (scope,),
                ).fetchone()
            return _row_to_goal_operator_controls_payload(row)

        return await asyncio.to_thread(_op)

    async def upsert_goal_operator_controls(
        self,
        *,
        scope: str = "global",
        stop_all_goals: bool,
        blocked_tools: list[str] | None = None,
        blocked_domains: list[str] | None = None,
        block_network_usage: bool,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        await self.initialize()
        now = _now_iso()

        def _op() -> None:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                existing = conn.execute(
                    """
                    SELECT created_at
                    FROM goal_operator_controls
                    WHERE scope=?
                    """,
                    (scope,),
                ).fetchone()
                if existing is None:
                    conn.execute(
                        """
                        INSERT INTO goal_operator_controls (
                            scope, stop_all_goals, blocked_tools_json, blocked_domains_json, block_network_usage,
                            metadata_json, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            scope,
                            1 if stop_all_goals else 0,
                            json.dumps(blocked_tools or [], ensure_ascii=False),
                            json.dumps(blocked_domains or [], ensure_ascii=False),
                            1 if block_network_usage else 0,
                            json.dumps(metadata or {}, ensure_ascii=False),
                            now,
                            now,
                        ),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE goal_operator_controls
                        SET stop_all_goals=?,
                            blocked_tools_json=?,
                            blocked_domains_json=?,
                            block_network_usage=?,
                            metadata_json=?,
                            updated_at=?
                        WHERE scope=?
                        """,
                        (
                            1 if stop_all_goals else 0,
                            json.dumps(blocked_tools or [], ensure_ascii=False),
                            json.dumps(blocked_domains or [], ensure_ascii=False),
                            1 if block_network_usage else 0,
                            json.dumps(metadata or {}, ensure_ascii=False),
                            now,
                            scope,
                        ),
                    )
                conn.commit()

        await asyncio.to_thread(_op)
        return await self.get_goal_operator_controls(scope=scope) or {}

    async def append_goal_operator_audit_log(
        self,
        *,
        event_type: str,
        subject_type: str,
        subject_id: str | None,
        action: str,
        summary: str,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        await self.initialize()
        now = _now_iso()

        def _op() -> int:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO goal_operator_audit_log (
                        event_type, subject_type, subject_id, action, summary, details_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_type,
                        subject_type,
                        subject_id,
                        action,
                        summary,
                        json.dumps(details or {}, ensure_ascii=False),
                        now,
                    ),
                )
                conn.commit()
                return int(cursor.lastrowid)

        audit_id = await asyncio.to_thread(_op)
        result = await self.get_goal_operator_audit_log_entry(audit_id)
        return result or {}

    async def get_goal_operator_audit_log_entry(self, audit_id: int) -> dict[str, Any] | None:
        await self.initialize()

        def _op() -> dict[str, Any] | None:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT * FROM goal_operator_audit_log WHERE id=?",
                    (audit_id,),
                ).fetchone()
            return _row_to_goal_operator_audit_log_payload(row)

        return await asyncio.to_thread(_op)

    async def list_goal_operator_audit_log(
        self,
        *,
        event_type: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        await self.initialize()

        def _op() -> list[dict[str, Any]]:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                if event_type is None and limit is None:
                    rows = conn.execute(
                        """
                        SELECT * FROM goal_operator_audit_log
                        ORDER BY datetime(created_at) DESC, id DESC
                        """
                    ).fetchall()
                elif event_type is None:
                    rows = conn.execute(
                        """
                        SELECT * FROM goal_operator_audit_log
                        ORDER BY datetime(created_at) DESC, id DESC
                        LIMIT ?
                        """,
                        (limit or 100,),
                    ).fetchall()
                elif limit is None:
                    rows = conn.execute(
                        """
                        SELECT * FROM goal_operator_audit_log
                        WHERE event_type=?
                        ORDER BY datetime(created_at) DESC, id DESC
                        """,
                        (event_type,),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT * FROM goal_operator_audit_log
                        WHERE event_type=?
                        ORDER BY datetime(created_at) DESC, id DESC
                        LIMIT ?
                        """,
                        (event_type, limit or 100),
                    ).fetchall()
            payloads: list[dict[str, Any]] = []
            for row in rows:
                payload = _row_to_goal_operator_audit_log_payload(row)
                if payload is not None:
                    payloads.append(payload)
            return payloads

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


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _goal_lease_is_stale(expires_at: str | None, *, now: str | None = None) -> bool:
    expires_at_dt = _parse_iso_datetime(expires_at)
    if expires_at_dt is None:
        return True
    effective_now = _parse_iso_datetime(now) if now is not None else datetime.now(UTC)
    if effective_now is None:
        effective_now = datetime.now(UTC)
    return expires_at_dt <= effective_now


def _load_goal_attempts(conn: sqlite3.Connection, goal_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT * FROM goal_attempts
        WHERE goal_id=?
        ORDER BY attempt_index ASC, datetime(created_at) ASC
        """,
        (goal_id,),
    ).fetchall()
    attempts: list[dict[str, Any]] = []
    for row in rows:
        payload = _row_to_goal_attempt_payload(row)
        if payload is not None:
            attempts.append(payload)
    return attempts


def _row_to_goal_payload(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    payload = dict(row)
    payload["execution_mode"] = _normalize_goal_execution_mode(payload.get("execution_mode"))
    payload["protocol_id"] = _normalize_goal_protocol_id(
        payload["execution_mode"],
        payload.get("protocol_id"),
    )
    payload["run_policy"] = json.loads(payload.pop("run_policy_json") or "{}")
    payload["capability_policy"] = json.loads(payload.pop("capability_policy_json") or "{}")
    payload["source_manifest"] = json.loads(payload.pop("source_manifest_json") or "{}")
    payload["summary"] = json.loads(payload.pop("summary_json") or "{}")
    payload["metadata"] = json.loads(payload.pop("metadata_json") or "{}")
    return payload


def _row_to_goal_attempt_payload(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    payload = dict(row)
    payload["summary"] = json.loads(payload.pop("summary_json") or "{}")
    payload["metadata"] = json.loads(payload.pop("metadata_json") or "{}")
    return payload


def _row_to_goal_checkpoint_payload(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    payload = dict(row)
    payload["payload"] = json.loads(payload.pop("payload_json") or "{}")
    payload["metadata"] = json.loads(payload.pop("metadata_json") or "{}")
    checkpoint_index = payload.get("checkpoint_index")
    try:
        payload["checkpoint_index"] = (
            int(checkpoint_index) if checkpoint_index is not None else None
        )
    except (TypeError, ValueError):
        payload["checkpoint_index"] = None
    return payload


def _row_to_goal_memory_snapshot_payload(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    payload = dict(row)
    payload["snapshot"] = json.loads(payload.pop("snapshot_json") or "{}")
    payload["metadata"] = json.loads(payload.pop("metadata_json") or "{}")
    return payload


def _row_to_goal_worker_generation_payload(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    payload = dict(row)
    payload["metadata"] = json.loads(payload.pop("metadata_json") or "{}")
    generation_index = payload.get("generation_index")
    try:
        payload["generation_index"] = (
            int(generation_index) if generation_index is not None else None
        )
    except (TypeError, ValueError):
        payload["generation_index"] = None
    return payload


def _row_to_goal_lease_payload(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    payload = dict(row)
    payload["takeover_count"] = int(payload.get("takeover_count") or 0)
    payload["metadata"] = json.loads(payload.pop("metadata_json") or "{}")
    return payload


def _row_to_goal_audit_finding_payload(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    payload = dict(row)
    payload["details"] = json.loads(payload.pop("details_json") or "{}")
    payload["resolved_at"] = payload.get("resolved_at")
    payload["closed_at"] = payload.get("closed_at")
    return payload


def _row_to_goal_operator_controls_payload(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    payload = dict(row)
    payload["stop_all_goals"] = bool(payload.get("stop_all_goals"))
    payload["blocked_tools"] = json.loads(payload.pop("blocked_tools_json") or "[]")
    payload["blocked_domains"] = json.loads(payload.pop("blocked_domains_json") or "[]")
    payload["block_network_usage"] = bool(payload.get("block_network_usage"))
    payload["metadata"] = json.loads(payload.pop("metadata_json") or "{}")
    return payload


def _row_to_goal_operator_audit_log_payload(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    payload = dict(row)
    payload["details"] = json.loads(payload.pop("details_json") or "{}")
    return payload


def _goal_audit_finding_resolution_timestamps(
    *,
    status: str,
    now: str,
    resolved_at: str | None = None,
    closed_at: str | None = None,
) -> tuple[str | None, str | None]:
    normalized_status = str(status or "open").strip().lower()
    if normalized_status == "resolved":
        return resolved_at or now, None
    if normalized_status == "closed":
        return resolved_at or now, closed_at or now
    return None, None


def _load_agent_run_artifacts(conn: sqlite3.Connection, run_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT artifact_id, artifact_type, title, uri, mime_type, size_bytes, metadata_json, created_at, updated_at
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
                "created_at": row[7],
                "updated_at": row[8],
            }
        )
    return artifacts


def _normalize_goal_execution_mode(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in _GOAL_EXECUTION_MODES:
        return normalized
    return _DEFAULT_GOAL_EXECUTION_MODE


def _normalize_goal_protocol_id(execution_mode: Any, protocol_id: Any) -> str | None:
    if _normalize_goal_execution_mode(execution_mode) == "single_agent":
        return _DEFAULT_SINGLE_AGENT_GOAL_PROTOCOL
    normalized = str(protocol_id or "").strip()
    return normalized or None


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column in columns:
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
