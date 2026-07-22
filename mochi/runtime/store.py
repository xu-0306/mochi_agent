"""SQLite-backed runtime store."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from mochi.runtime.goal_strategy_registry import DEFAULT_GOAL_STRATEGY_ID, get_goal_strategy_entry
from mochi.runtime.runtime_approval_lifecycle import (
    RuntimeApprovalLifecycleMixin,
    initialize_runtime_approval_schema,
)
from mochi.runtime.security_audit import (
    SecurityAuditEvent,
    redact_for_persistence,
    security_audit_projection,
)

_UNSET = object()
_DEFAULT_GOAL_EXECUTION_MODE = "single_agent"
_DEFAULT_SINGLE_AGENT_GOAL_PROTOCOL = "autonomous_single_agent"
_SECURITY_AUDIT_RETENTION_DAYS = 90
_SECURITY_AUDIT_MAX_EVENTS = 10_000
_GOAL_EXECUTION_MODES = {"single_agent", "workflow"}
_GOAL_WORKER_GENERATION_TERMINAL_STATUSES = {
    "cancelled",
    "completed",
    "failed",
    "rolled_over",
    "superseded",
    "succeeded",
}


class RuntimeStore(RuntimeApprovalLifecycleMixin):
    """Persist task runs, task events, and approval requests."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._lock = asyncio.Lock()
        self._initialized = False

    @property
    def database_path(self) -> Path:
        """Return the exact SQLite path used by durable runtime facades."""

        return self._db_path

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
            _initialize_change_set_schema(conn)
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
                CREATE TABLE IF NOT EXISTS subagent_transcripts (
                    id TEXT PRIMARY KEY,
                    parent_type TEXT NOT NULL,
                    parent_id TEXT NOT NULL,
                    session_id TEXT,
                    goal_id TEXT,
                    agent_run_id TEXT,
                    parent_turn_id TEXT,
                    role_id TEXT,
                    title TEXT,
                    model_id TEXT,
                    status TEXT NOT NULL,
                    system_prompt TEXT,
                    user_prompt TEXT,
                    prompt_preview TEXT,
                    summary TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS subagent_transcript_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subagent_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    event_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(subagent_id) REFERENCES subagent_transcripts(id) ON DELETE CASCADE
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
                    execution_mode TEXT NOT NULL DEFAULT 'single_agent',
                    strategy_id TEXT,
                    selection_source TEXT,
                    selection_reason TEXT,
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
                CREATE TABLE IF NOT EXISTS security_audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    subject_type TEXT NOT NULL,
                    subject_id TEXT,
                    request_digest TEXT,
                    outcome TEXT,
                    details_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_security_audit_event_created
                ON security_audit_events(event_type, created_at DESC)
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
                """
                CREATE INDEX IF NOT EXISTS idx_subagent_transcripts_parent
                ON subagent_transcripts(parent_type, parent_id, updated_at DESC)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_subagent_transcripts_session_id
                ON subagent_transcripts(session_id, updated_at DESC)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_subagent_transcripts_goal_id
                ON subagent_transcripts(goal_id, updated_at DESC)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_subagent_transcripts_agent_run_id
                ON subagent_transcripts(agent_run_id, updated_at DESC)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_subagent_transcript_events_subagent_id_seq
                ON subagent_transcript_events(subagent_id, seq)
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
            initialize_runtime_approval_schema(conn)
            _ensure_column(conn, "agent_runs", "title", "TEXT")
            _ensure_column(conn, "agent_runs", "topic", "TEXT")
            _ensure_column(conn, "agent_runs", "project_id", "TEXT")
            _ensure_column(conn, "agent_runs", "workspace_dir", "TEXT")
            _ensure_column(conn, "agent_runs", "latest_error", "TEXT")
            _ensure_column(conn, "agent_runs", "evidence_status_json", "TEXT NOT NULL DEFAULT '{}'")
            _ensure_column(conn, "agent_runs", "run_policy_json", "TEXT NOT NULL DEFAULT '{}'")
            _ensure_column(conn, "agent_run_artifacts", "artifact_id", "TEXT")
            _ensure_column(conn, "subagent_transcripts", "session_id", "TEXT")
            _ensure_column(conn, "subagent_transcripts", "goal_id", "TEXT")
            _ensure_column(conn, "subagent_transcripts", "agent_run_id", "TEXT")
            _ensure_column(conn, "subagent_transcripts", "parent_turn_id", "TEXT")
            _ensure_column(conn, "subagent_transcripts", "role_id", "TEXT")
            _ensure_column(conn, "subagent_transcripts", "title", "TEXT")
            _ensure_column(conn, "subagent_transcripts", "model_id", "TEXT")
            _ensure_column(conn, "subagent_transcripts", "system_prompt", "TEXT")
            _ensure_column(conn, "subagent_transcripts", "user_prompt", "TEXT")
            _ensure_column(conn, "subagent_transcripts", "prompt_preview", "TEXT")
            _ensure_column(conn, "subagent_transcripts", "summary", "TEXT")
            _ensure_column(
                conn,
                "subagent_transcripts",
                "metadata_json",
                "TEXT NOT NULL DEFAULT '{}'",
            )
            _ensure_column(conn, "goals", "title", "TEXT")
            _ensure_column(conn, "goals", "goal_type", "TEXT")
            _ensure_column(
                conn,
                "goals",
                "execution_mode",
                "TEXT NOT NULL DEFAULT 'single_agent'",
            )
            _ensure_column(conn, "goals", "strategy_id", "TEXT")
            _ensure_column(conn, "goals", "selection_source", "TEXT")
            _ensure_column(conn, "goals", "selection_reason", "TEXT")
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
        projected = redact_for_persistence(event)
        safe_event = projected if isinstance(projected, dict) else {}

        def _op() -> None:
            with sqlite3.connect(self._db_path) as conn:
                row = conn.execute(
                    "SELECT COALESCE(MAX(seq), 0) + 1 FROM task_events WHERE task_id=?",
                    (task_id,),
                ).fetchone()
                seq = int(row[0]) if row else 1
                conn.execute(
                    "INSERT INTO task_events(task_id, seq, event_json, created_at) VALUES (?, ?, ?, ?)",
                    (task_id, seq, json.dumps(safe_event, ensure_ascii=False), now),
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

    async def _legacy_create_approval_request(
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

    async def _legacy_get_approval_request(self, approval_id: str) -> dict[str, Any] | None:
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

    async def _legacy_list_approval_requests(self, *, status: str | None = None) -> list[dict[str, Any]]:
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

    async def _legacy_resolve_approval_request(
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

    async def _legacy_update_approval_request_metadata(
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

    async def _legacy_get_pending_approval_for_task(self, task_id: str) -> dict[str, Any] | None:
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
        strategy_id: str | None = None,
        selection_source: str | None = None,
        selection_reason: str | None = None,
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
        explicit_strategy_hint = str(strategy_id or "").strip()
        explicit_protocol_hint = str(protocol_id or "").strip()
        _validate_goal_strategy_protocol_inputs(
            strategy_id=explicit_strategy_hint or None,
            protocol_id=explicit_protocol_hint or None,
        )
        _validate_goal_selection_source_input(selection_source)
        normalized_strategy_id = _normalize_goal_strategy_id(
            strategy_id=strategy_id,
            protocol_id=protocol_id,
            selection_source="explicit_override" if explicit_protocol_hint else None,
        )
        normalized_protocol_id = _normalize_goal_protocol_id(
            normalized_strategy_id,
            protocol_id,
        )
        normalized_selection_source = _normalize_goal_selection_source(
            selection_source=selection_source,
            strategy_id=explicit_strategy_hint,
            protocol_id=explicit_protocol_hint,
            treat_protocol_as_explicit=bool(explicit_protocol_hint),
        )
        normalized_selection_reason = _normalize_goal_selection_reason(
            selection_reason=selection_reason,
            selection_source=normalized_selection_source,
            strategy_id=normalized_strategy_id,
        )

        def _op() -> None:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO goals (
                        id, objective, title, goal_type, execution_mode, strategy_id, selection_source,
                        selection_reason, protocol_id, topic, project_id,
                        workspace_dir, status, current_attempt_id, run_policy_json,
                        capability_policy_json, source_manifest_json, summary_json,
                        metadata_json, latest_error, started_at, finished_at, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        goal_id,
                        objective,
                        title,
                        goal_type,
                        normalized_execution_mode,
                        normalized_strategy_id,
                        normalized_selection_source,
                        normalized_selection_reason,
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
        goal_summary: dict[str, Any] | None = None,
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
                    SELECT started_at, finished_at, latest_error, current_attempt_id, summary_json
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
                goal_current_summary = json.loads(str(goal_existing[4] or "{}"))
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
                        summary_json=?,
                        latest_error=?,
                        current_attempt_id=?,
                        started_at=?,
                        finished_at=?,
                        updated_at=?
                    WHERE id=?
                    """,
                    (
                        goal_status,
                        json.dumps(
                            goal_current_summary if goal_summary is None else goal_summary,
                            ensure_ascii=False,
                        ),
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
        strategy_id: str | None | object = _UNSET,
        selection_source: str | None | object = _UNSET,
        selection_reason: str | None | object = _UNSET,
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
                    SELECT summary_json, metadata_json, strategy_id, selection_source, selection_reason,
                           latest_error, project_id, workspace_dir, current_attempt_id
                    FROM goals
                    WHERE id=?
                    """,
                    (goal_id,),
                ).fetchone()
                if existing is None:
                    return
                current_summary = json.loads(str(existing[0] or "{}"))
                current_metadata = json.loads(str(existing[1] or "{}"))
                current_strategy_id = existing[2]
                current_selection_source = existing[3]
                current_selection_reason = existing[4]
                current_latest_error = existing[5]
                current_project_id = existing[6]
                current_workspace_dir = existing[7]
                current_current_attempt_id = existing[8]
                conn.execute(
                    """
                    UPDATE goals
                    SET summary_json=?,
                        metadata_json=?,
                        strategy_id=?,
                        selection_source=?,
                        selection_reason=?,
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
                        current_strategy_id if strategy_id is _UNSET else strategy_id,
                        current_selection_source if selection_source is _UNSET else selection_source,
                        current_selection_reason if selection_reason is _UNSET else selection_reason,
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
                # Serialize the read/insert decision so concurrent supervisors
                # cannot both observe a missing lease and insert it.
                conn.execute("BEGIN IMMEDIATE")
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

    async def append_security_audit_event(
        self,
        event: SecurityAuditEvent,
    ) -> dict[str, Any]:
        """Persist an allowlisted, centrally redacted security audit event."""

        await self.initialize()
        now = _now_iso()
        projection = security_audit_projection(event)

        def _op() -> int:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO security_audit_events (
                        event_type, subject_type, subject_id, request_digest,
                        outcome, details_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        projection["event_type"],
                        projection["subject_type"],
                        projection["subject_id"],
                        projection["request_digest"],
                        projection["outcome"],
                        json.dumps(projection["details"], ensure_ascii=False),
                        now,
                    ),
                )
                cutoff = (datetime.now(UTC) - timedelta(days=_SECURITY_AUDIT_RETENTION_DAYS)).isoformat()
                conn.execute(
                    "DELETE FROM security_audit_events WHERE created_at<?",
                    (cutoff,),
                )
                conn.execute(
                    """
                    DELETE FROM security_audit_events
                    WHERE id NOT IN (
                        SELECT id FROM security_audit_events
                        ORDER BY id DESC LIMIT ?
                    )
                    """,
                    (_SECURITY_AUDIT_MAX_EVENTS,),
                )
                conn.commit()
                return int(cursor.lastrowid)

        event_id = await asyncio.to_thread(_op)
        rows = await self.list_security_audit_events(limit=1, event_id=event_id)
        return rows[0] if rows else {}

    async def list_security_audit_events(
        self,
        *,
        event_type: str | None = None,
        event_id: int | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return only the public security-audit projection."""

        await self.initialize()
        bounded_limit = max(1, min(int(limit), 1000))

        def _op() -> list[dict[str, Any]]:
            clauses: list[str] = []
            params: list[Any] = []
            if event_type is not None:
                clauses.append("event_type=?")
                params.append(event_type)
            if event_id is not None:
                clauses.append("id=?")
                params.append(event_id)
            where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
            sql = (
                "SELECT id,event_type,subject_type,subject_id,request_digest,"
                "outcome,details_json,created_at FROM security_audit_events"
                f"{where} ORDER BY id DESC LIMIT ?"
            )
            params.append(bounded_limit)
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(sql, tuple(params)).fetchall()
            output: list[dict[str, Any]] = []
            for row in rows:
                item = dict(row)
                item["details"] = json.loads(item.pop("details_json") or "{}")
                output.append(item)
            return output

        return await asyncio.to_thread(_op)

    async def prune_security_audit_events(
        self,
        *,
        retention_days: int = _SECURITY_AUDIT_RETENTION_DAYS,
        max_events: int = _SECURITY_AUDIT_MAX_EVENTS,
    ) -> int:
        """Delete expired/overflow audit rows and return the removed count."""

        await self.initialize()
        bounded_days = max(1, int(retention_days))
        bounded_max = max(1, int(max_events))
        cutoff = (datetime.now(UTC) - timedelta(days=bounded_days)).isoformat()

        def _op() -> int:
            with sqlite3.connect(self._db_path) as conn:
                before = int(
                    conn.execute("SELECT COUNT(*) FROM security_audit_events").fetchone()[0]
                )
                conn.execute(
                    "DELETE FROM security_audit_events WHERE created_at<?",
                    (cutoff,),
                )
                conn.execute(
                    """
                    DELETE FROM security_audit_events
                    WHERE id NOT IN (
                        SELECT id FROM security_audit_events
                        ORDER BY id DESC LIMIT ?
                    )
                    """,
                    (bounded_max,),
                )
                after = int(
                    conn.execute("SELECT COUNT(*) FROM security_audit_events").fetchone()[0]
                )
                conn.commit()
            return before - after

        return await asyncio.to_thread(_op)

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

    async def upsert_subagent_transcript(
        self,
        *,
        subagent_id: str,
        parent_type: str,
        parent_id: str,
        session_id: str | None = None,
        goal_id: str | None = None,
        agent_run_id: str | None = None,
        parent_turn_id: str | None = None,
        role_id: str | None = None,
        title: str | None = None,
        model_id: str | None = None,
        status: str = "running",
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        prompt_preview: str | None = None,
        summary: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        await self.initialize()
        now = _now_iso()

        def _op() -> dict[str, Any] | None:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                conn.execute(
                    """
                    INSERT OR IGNORE INTO subagent_transcripts(
                        id, parent_type, parent_id, session_id, goal_id, agent_run_id,
                        parent_turn_id, role_id, title, model_id, status, system_prompt,
                        user_prompt, prompt_preview, summary, metadata_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        subagent_id,
                        parent_type,
                        parent_id,
                        session_id,
                        goal_id,
                        agent_run_id,
                        parent_turn_id,
                        role_id,
                        title,
                        model_id,
                        status,
                        system_prompt,
                        user_prompt,
                        prompt_preview,
                        summary,
                        json.dumps(metadata or {}, ensure_ascii=False),
                        now,
                        now,
                    ),
                )
                existing = conn.execute(
                    "SELECT * FROM subagent_transcripts WHERE id=?",
                    (subagent_id,),
                ).fetchone()
                payload = _row_to_subagent_transcript_payload(existing, conn=conn)
                if payload is not None:
                    next_metadata = payload["metadata"] if metadata is None else metadata
                    conn.execute(
                        """
                        UPDATE subagent_transcripts
                        SET parent_type=?,
                            parent_id=?,
                            session_id=?,
                            goal_id=?,
                            agent_run_id=?,
                            parent_turn_id=?,
                            role_id=?,
                            title=?,
                            model_id=?,
                            status=?,
                            system_prompt=?,
                            user_prompt=?,
                            prompt_preview=?,
                            summary=?,
                            metadata_json=?,
                            updated_at=?
                        WHERE id=?
                        """,
                        (
                            parent_type or payload["parent_type"],
                            parent_id or payload["parent_id"],
                            payload["session_id"] if session_id is None else session_id,
                            payload["goal_id"] if goal_id is None else goal_id,
                            payload["agent_run_id"] if agent_run_id is None else agent_run_id,
                            payload["parent_turn_id"]
                            if parent_turn_id is None
                            else parent_turn_id,
                            payload["role_id"] if role_id is None else role_id,
                            payload["title"] if title is None else title,
                            payload["model_id"] if model_id is None else model_id,
                            status or payload["status"],
                            payload["system_prompt"]
                            if system_prompt is None
                            else system_prompt,
                            payload["user_prompt"] if user_prompt is None else user_prompt,
                            payload["prompt_preview"]
                            if prompt_preview is None
                            else prompt_preview,
                            payload["summary"] if summary is None else summary,
                            json.dumps(next_metadata, ensure_ascii=False),
                            now,
                            subagent_id,
                        ),
                    )
                conn.commit()
                row = conn.execute(
                    "SELECT * FROM subagent_transcripts WHERE id=?",
                    (subagent_id,),
                ).fetchone()
                return _row_to_subagent_transcript_payload(row, conn=conn)

        return await asyncio.to_thread(_op) or {}

    async def append_subagent_transcript_event(
        self,
        subagent_id: str,
        event: dict[str, Any],
    ) -> None:
        await self.initialize()
        now = _now_iso()

        def _op() -> None:
            with sqlite3.connect(self._db_path) as conn:
                row = conn.execute(
                    """
                    SELECT COALESCE(MAX(seq), 0) + 1
                    FROM subagent_transcript_events
                    WHERE subagent_id=?
                    """,
                    (subagent_id,),
                ).fetchone()
                seq = int(row[0]) if row else 1
                conn.execute(
                    """
                    INSERT INTO subagent_transcript_events(subagent_id, seq, event_json, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (subagent_id, seq, json.dumps(event, ensure_ascii=False), now),
                )
                conn.execute(
                    "UPDATE subagent_transcripts SET updated_at=? WHERE id=?",
                    (now, subagent_id),
                )
                conn.commit()

        await asyncio.to_thread(_op)

    async def list_subagent_transcripts(
        self,
        *,
        parent_type: str | None = None,
        parent_id: str | None = None,
        session_id: str | None = None,
        goal_id: str | None = None,
        agent_run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        await self.initialize()

        def _op() -> list[dict[str, Any]]:
            clauses: list[str] = []
            params: list[Any] = []
            if parent_type is not None:
                clauses.append("t.parent_type=?")
                params.append(parent_type)
            if parent_id is not None:
                clauses.append("t.parent_id=?")
                params.append(parent_id)
            if session_id is not None:
                clauses.append("t.session_id=?")
                params.append(session_id)
            if goal_id is not None:
                clauses.append("t.goal_id=?")
                params.append(goal_id)
            if agent_run_id is not None:
                clauses.append("t.agent_run_id=?")
                params.append(agent_run_id)
            where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            query = f"""
                SELECT t.*, COALESCE(COUNT(e.id), 0) AS event_count
                FROM subagent_transcripts AS t
                LEFT JOIN subagent_transcript_events AS e
                    ON e.subagent_id = t.id
                {where_sql}
                GROUP BY t.id
                ORDER BY datetime(t.updated_at) DESC, t.id ASC
            """
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(query, tuple(params)).fetchall()
                return [_row_to_subagent_transcript_summary_payload(row) for row in rows]

        return await asyncio.to_thread(_op)

    async def get_subagent_transcript(self, subagent_id: str) -> dict[str, Any] | None:
        await self.initialize()

        def _op() -> dict[str, Any] | None:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    """
                    SELECT t.*, COALESCE(COUNT(e.id), 0) AS event_count
                    FROM subagent_transcripts AS t
                    LEFT JOIN subagent_transcript_events AS e
                        ON e.subagent_id = t.id
                    WHERE t.id=?
                    GROUP BY t.id
                    """,
                    (subagent_id,),
                ).fetchone()
                if row is None:
                    return None
                payload = _row_to_subagent_transcript_payload(row, conn=conn)
                payload["event_count"] = int(row["event_count"] or 0)
                payload["events"] = _load_subagent_transcript_events(conn, subagent_id)
                return payload

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
    raw_execution_mode = payload.get("execution_mode")
    payload["execution_mode"] = _normalize_goal_execution_mode(raw_execution_mode)
    summary = json.loads(payload.pop("summary_json") or "{}")
    metadata = json.loads(payload.pop("metadata_json") or "{}")
    raw_strategy_id = payload.get("strategy_id")
    raw_protocol_id = payload.get("protocol_id")
    normalized_strategy_id = _normalize_goal_strategy_id(
        strategy_id=raw_strategy_id,
        protocol_id=raw_protocol_id,
        selection_source=payload.get("selection_source"),
        summary=summary,
        metadata=metadata,
    )
    payload["strategy_id"] = normalized_strategy_id
    payload["protocol_id"] = _normalize_goal_protocol_id(
        normalized_strategy_id,
        raw_protocol_id,
    )
    normalized_selection_source = _normalize_goal_selection_source(
        selection_source=payload.get("selection_source"),
        execution_mode=raw_execution_mode,
        strategy_id=raw_strategy_id,
        protocol_id=raw_protocol_id,
        summary=summary,
        metadata=metadata,
    )
    payload["selection_source"] = normalized_selection_source
    payload["selection_reason"] = _normalize_goal_selection_reason(
        selection_reason=payload.get("selection_reason"),
        selection_source=normalized_selection_source,
        strategy_id=normalized_strategy_id,
        summary=summary,
        metadata=metadata,
    )
    payload["run_policy"] = json.loads(payload.pop("run_policy_json") or "{}")
    payload["capability_policy"] = json.loads(payload.pop("capability_policy_json") or "{}")
    payload["source_manifest"] = json.loads(payload.pop("source_manifest_json") or "{}")
    payload["summary"] = summary
    payload["metadata"] = metadata
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


def _load_subagent_transcript_events(
    conn: sqlite3.Connection,
    subagent_id: str,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT seq, event_json, created_at
        FROM subagent_transcript_events
        WHERE subagent_id=?
        ORDER BY seq ASC
        """,
        (subagent_id,),
    ).fetchall()
    events: list[dict[str, Any]] = []
    for row in rows:
        payload = json.loads(str(row[1] or "{}"))
        if "seq" not in payload:
            payload["seq"] = int(row[0])
        if "created_at" not in payload:
            payload["created_at"] = row[2]
        events.append(_hydrate_subagent_message_delivery_fields(payload))
    return events


def _hydrate_subagent_message_delivery_fields(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("type") == "subagent_tool_cancelled" and "status" not in payload:
        payload["status"] = "cancelled"
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        return payload
    for key in (
        "message_id",
        "delivery_mode",
        "delivery_status",
        "delivery_reason",
        "interrupt",
        "cancel_current_tool",
    ):
        if key not in payload and key in metadata:
            payload[key] = metadata[key]
    return payload


def _row_to_subagent_transcript_payload(
    row: sqlite3.Row | None,
    *,
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any] | None:
    if row is None:
        return None
    payload = dict(row)
    payload["subagent_id"] = payload.pop("id")
    payload["metadata"] = json.loads(payload.pop("metadata_json") or "{}")
    event_count = payload.get("event_count")
    if event_count is not None:
        try:
            payload["event_count"] = int(event_count)
        except (TypeError, ValueError):
            payload["event_count"] = 0
    elif conn is not None:
        payload["event_count"] = int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM subagent_transcript_events
                WHERE subagent_id=?
                """,
                (payload["subagent_id"],),
            ).fetchone()[0]
        )
    else:
        payload["event_count"] = 0
    return payload


def _row_to_subagent_transcript_summary_payload(row: sqlite3.Row | None) -> dict[str, Any]:
    payload = _row_to_subagent_transcript_payload(row)
    if payload is None:
        return {}
    payload.pop("system_prompt", None)
    payload.pop("user_prompt", None)
    return payload


def _normalize_goal_execution_mode(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in _GOAL_EXECUTION_MODES:
        return normalized
    return _DEFAULT_GOAL_EXECUTION_MODE


def _goal_text_from_sources(
    summary: Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None,
    *keys: str,
) -> str:
    for source in (summary or {}, metadata or {}):
        for key in keys:
            value = str(source.get(key) or "").strip()
            if value:
                return value
    return ""


def _goal_has_legacy_route_state(
    summary: Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None,
) -> bool:
    return bool(
        _goal_text_from_sources(
            summary,
            metadata,
            "interaction_mode",
            "execution_topology",
            "execution_mode",
            "default_route",
        )
    )


def _validate_goal_strategy_protocol_inputs(
    *,
    strategy_id: str | None,
    protocol_id: str | None,
) -> None:
    normalized_strategy_id = str(strategy_id or "").strip()
    normalized_protocol_id = str(protocol_id or "").strip()
    if not normalized_strategy_id or not normalized_protocol_id:
        return
    entry = get_goal_strategy_entry(normalized_strategy_id)
    expected_protocol_id = (
        str((entry.protocol_id if entry is not None else normalized_strategy_id) or "").strip()
        or normalized_strategy_id
    )
    if expected_protocol_id != normalized_protocol_id:
        raise ValueError(
            f"Conflicting strategy_id/protocol_id: strategy {normalized_strategy_id} requires protocol {expected_protocol_id}, not {normalized_protocol_id}."
        )


def _validate_goal_selection_source_input(selection_source: Any) -> None:
    normalized = str(selection_source or "").strip()
    if not normalized:
        return
    if normalized in {"explicit_override", "semantic_registry_selector", "safe_default", "legacy_migration"}:
        return
    raise ValueError(f"Invalid selection_source for create_goal: {normalized}.")


def _normalize_goal_protocol_id(strategy_id: Any, protocol_id: Any) -> str:
    normalized = str(protocol_id or "").strip()
    if normalized:
        return normalized
    normalized_strategy_id = str(strategy_id or "").strip()
    if normalized_strategy_id:
        entry = get_goal_strategy_entry(normalized_strategy_id)
        if entry is not None and str(entry.protocol_id or "").strip():
            return str(entry.protocol_id or "").strip()
        return normalized_strategy_id
    return _DEFAULT_SINGLE_AGENT_GOAL_PROTOCOL


def _normalize_goal_strategy_id(
    *,
    strategy_id: Any,
    protocol_id: Any,
    selection_source: Any = None,
    summary: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    normalized = str(strategy_id or "").strip()
    if normalized:
        return normalized
    protocol = str(protocol_id or "").strip()
    if protocol:
        return protocol
    legacy_strategy = _goal_text_from_sources(summary, metadata, "strategy_id", "protocol_selection")
    if legacy_strategy:
        return legacy_strategy
    normalized_selection_source = str(selection_source or "").strip()
    if normalized_selection_source == "explicit_override":
        return _DEFAULT_SINGLE_AGENT_GOAL_PROTOCOL
    return DEFAULT_GOAL_STRATEGY_ID


def _normalize_goal_selection_source(
    *,
    selection_source: Any,
    execution_mode: Any = None,
    strategy_id: Any,
    protocol_id: Any,
    summary: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    treat_protocol_as_explicit: bool = False,
) -> str:
    normalized = str(selection_source or "").strip()
    if normalized in {"explicit_override", "semantic_registry_selector", "safe_default", "legacy_migration"}:
        return normalized
    if str(strategy_id or "").strip():
        return "explicit_override"
    if treat_protocol_as_explicit and str(protocol_id or "").strip():
        return "explicit_override"
    for source in (summary or {}, metadata or {}):
        summary_selection_source = str(source.get("selection_source") or "").strip()
        if summary_selection_source in {
            "explicit_override",
            "semantic_registry_selector",
            "safe_default",
            "legacy_migration",
        }:
            return summary_selection_source
    if str(protocol_id or "").strip():
        return "legacy_migration"
    if _goal_text_from_sources(
        summary,
        metadata,
        "strategy_id",
        "protocol_selection",
        "selection_reason",
        "selection_rationale",
    ):
        return "legacy_migration"
    if _goal_has_legacy_route_state(summary, metadata):
        return "legacy_migration"
    if _normalize_goal_execution_mode(execution_mode) == "workflow":
        return "legacy_migration"
    if normalized:
        return "legacy_migration"
    return "safe_default"


def _normalize_goal_selection_reason(
    *,
    selection_reason: Any,
    selection_source: Any,
    strategy_id: Any,
    summary: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    normalized = str(selection_reason or "").strip()
    if normalized:
        return normalized
    legacy_reason = _goal_text_from_sources(summary, metadata, "selection_reason", "selection_rationale")
    if legacy_reason:
        return legacy_reason
    resolved_strategy_id = _normalize_goal_strategy_id(
        strategy_id=strategy_id,
        protocol_id=None,
    )
    if str(selection_source or "").strip() == "explicit_override":
        return f"Strategy explicitly set to {resolved_strategy_id}."
    if str(selection_source or "").strip() == "legacy_migration":
        return f"Legacy goal metadata mapped to {resolved_strategy_id} during strategy migration."
    return f"Defaulted to {resolved_strategy_id} because no explicit strategy was provided."


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column in columns:
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

def _initialize_change_set_schema(conn: sqlite3.Connection) -> None:
    """Create the durable Task 3 schema without mutating existing rows."""

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS change_sets (
            id TEXT PRIMARY KEY,
            schema_version INTEGER NOT NULL,
            requester_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            task_id TEXT,
            workspace_root TEXT NOT NULL,
            workspace_identity_json TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            intent TEXT NOT NULL,
            request_digest TEXT NOT NULL,
            authorization_envelope_json TEXT NOT NULL,
            patch_sha256 TEXT,
            policy_version TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            applied_at TEXT,
            updated_at TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS change_entries (
            id TEXT PRIMARY KEY,
            change_set_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            relative_path TEXT NOT NULL,
            operation TEXT NOT NULL,
            base_sha256 TEXT,
            after_sha256 TEXT,
            base_identity_json TEXT,
            before_blob_id TEXT,
            after_blob_id TEXT,
            mode_before INTEGER,
            mode_after INTEGER,
            base_metadata_blob_id TEXT,
            after_metadata_blob_id TEXT,
            rename_source TEXT,
            dependency_group TEXT,
            UNIQUE(change_set_id, ordinal),
            UNIQUE(change_set_id, id),
            FOREIGN KEY(change_set_id) REFERENCES change_sets(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS applied_change_entries (
            change_set_id TEXT NOT NULL,
            entry_id TEXT NOT NULL,
            applied_sha256 TEXT,
            applied_identity_json TEXT,
            applied_metadata_sha256 TEXT,
            applied_at TEXT NOT NULL,
            PRIMARY KEY(change_set_id, entry_id),
            FOREIGN KEY(change_set_id, entry_id)
                REFERENCES change_entries(change_set_id, id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS change_blobs (
            id TEXT PRIMARY KEY,
            sha256 TEXT NOT NULL UNIQUE,
            size_bytes INTEGER NOT NULL,
            content BLOB NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS blob_references (
            blob_id TEXT NOT NULL,
            owner_type TEXT NOT NULL,
            owner_id TEXT NOT NULL,
            purpose TEXT NOT NULL,
            retained_until TEXT,
            state TEXT NOT NULL,
            PRIMARY KEY(blob_id, owner_type, owner_id, purpose),
            FOREIGN KEY(blob_id) REFERENCES change_blobs(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS undo_retention (
            change_set_id TEXT NOT NULL,
            entry_id TEXT NOT NULL,
            status TEXT NOT NULL,
            retained_until TEXT,
            expired_at TEXT,
            PRIMARY KEY(change_set_id, entry_id),
            FOREIGN KEY(change_set_id, entry_id)
                REFERENCES change_entries(change_set_id, id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS file_transaction_journal (
            id TEXT PRIMARY KEY,
            change_set_id TEXT NOT NULL,
            status TEXT NOT NULL,
            phase TEXT NOT NULL,
            error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(change_set_id) REFERENCES change_sets(id) ON DELETE RESTRICT
        );

        CREATE TABLE IF NOT EXISTS file_transaction_entries (
            journal_id TEXT NOT NULL,
            entry_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            state TEXT NOT NULL,
            base_sha256 TEXT,
            after_sha256 TEXT,
            base_identity_json TEXT,
            staged_name TEXT,
            staged_identity_json TEXT,
            rollback_blob_id TEXT,
            rollback_staged_name TEXT,
            rollback_staged_identity_json TEXT,
            rollback_successor_identity_json TEXT,
            base_metadata_blob_id TEXT,
            last_error TEXT,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(journal_id, entry_id),
            UNIQUE(journal_id, ordinal),
            FOREIGN KEY(journal_id)
                REFERENCES file_transaction_journal(id) ON DELETE CASCADE
        );

        CREATE UNIQUE INDEX IF NOT EXISTS change_set_idempotency
        ON change_sets(
            schema_version,
            requester_id,
            session_id,
            IFNULL(task_id, ''),
            workspace_identity_json,
            request_digest
        );

        CREATE INDEX IF NOT EXISTS idx_change_entries_change_set_ordinal
        ON change_entries(change_set_id, ordinal);

        CREATE INDEX IF NOT EXISTS idx_blob_references_state_retained
        ON blob_references(state, retained_until);

        CREATE INDEX IF NOT EXISTS idx_blob_references_owner_purpose_state
        ON blob_references(owner_type, owner_id, purpose, state);

        CREATE INDEX IF NOT EXISTS idx_undo_retention_status_retained
        ON undo_retention(status, retained_until);

        CREATE INDEX IF NOT EXISTS idx_file_transaction_journal_status_updated
        ON file_transaction_journal(status, updated_at);
        """
    )
