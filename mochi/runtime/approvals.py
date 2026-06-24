"""Exec approval request primitives."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Lock
from typing import Any, Literal, Protocol

ApprovalStatus = Literal["pending", "approved_once", "approved_and_saved_rule", "rejected"]
ApprovalDecision = Literal["approve_once", "approve_and_save_rule", "reject"]


def utc_now_iso() -> str:
    """回傳 UTC ISO8601 時間字串。"""
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class ExecApprovalRequest:
    """Exec approval request model."""

    approval_id: str
    status: ApprovalStatus
    reason: str | None
    command: str
    shell: str
    scope: str
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)
    command_payload: dict[str, Any] | None = None
    execution_result: dict[str, Any] | None = None
    resolved_at: str | None = None


class ApprovalStore(Protocol):
    """Sync approval-store contract shared by exec approval backends."""

    def create(
        self,
        *,
        approval_id: str,
        command: str,
        shell: str,
        scope: str,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
        command_payload: dict[str, Any] | None = None,
    ) -> ExecApprovalRequest: ...

    def get(self, approval_id: str) -> ExecApprovalRequest | None: ...

    def list(self, *, status: ApprovalStatus | None = None) -> list[ExecApprovalRequest]: ...

    def resolve(
        self,
        approval_id: str,
        *,
        decision: ApprovalDecision,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
        execution_result: dict[str, Any] | None = None,
    ) -> ExecApprovalRequest | None: ...


class InMemoryApprovalStore:
    """In-memory approval request store primitive."""

    def __init__(self) -> None:
        self._items: dict[str, ExecApprovalRequest] = {}

    def create(
        self,
        *,
        approval_id: str,
        command: str,
        shell: str,
        scope: str,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
        command_payload: dict[str, Any] | None = None,
    ) -> ExecApprovalRequest:
        request = ExecApprovalRequest(
            approval_id=approval_id,
            status="pending",
            reason=reason,
            command=command,
            shell=shell,
            scope=scope,
            created_at=utc_now_iso(),
            metadata=dict(metadata) if isinstance(metadata, dict) else {},
            command_payload=dict(command_payload) if isinstance(command_payload, dict) else None,
            execution_result=None,
            resolved_at=None,
        )
        self._items[approval_id] = request
        return request

    def get(self, approval_id: str) -> ExecApprovalRequest | None:
        return self._items.get(approval_id)

    def list(self, *, status: ApprovalStatus | None = None) -> list[ExecApprovalRequest]:
        items = list(self._items.values())
        if status is None:
            return sorted(items, key=lambda item: item.created_at, reverse=True)
        filtered = [item for item in items if item.status == status]
        return sorted(filtered, key=lambda item: item.created_at, reverse=True)

    def resolve(
        self,
        approval_id: str,
        *,
        decision: ApprovalDecision,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
        execution_result: dict[str, Any] | None = None,
    ) -> ExecApprovalRequest | None:
        current = self._items.get(approval_id)
        if current is None:
            return None
        next_status: ApprovalStatus = (
            "approved_once"
            if decision == "approve_once"
            else "approved_and_saved_rule"
            if decision == "approve_and_save_rule"
            else "rejected"
        )
        resolved = ExecApprovalRequest(
            approval_id=current.approval_id,
            status=next_status,
            reason=reason if reason is not None else current.reason,
            command=current.command,
            shell=current.shell,
            scope=current.scope,
            created_at=current.created_at,
            metadata=(
                {**current.metadata, **metadata}
                if isinstance(metadata, dict)
                else dict(current.metadata)
            ),
            command_payload=current.command_payload,
            execution_result=(
                dict(execution_result) if isinstance(execution_result, dict) else current.execution_result
            ),
            resolved_at=utc_now_iso(),
        )
        self._items[approval_id] = resolved
        return resolved


class PersistentApprovalStore:
    """SQLite-backed exec approval store for restart-safe standalone approvals."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._lock = Lock()
        self._init_db()

    def create(
        self,
        *,
        approval_id: str,
        command: str,
        shell: str,
        scope: str,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
        command_payload: dict[str, Any] | None = None,
    ) -> ExecApprovalRequest:
        request = ExecApprovalRequest(
            approval_id=approval_id,
            status="pending",
            reason=reason,
            command=command,
            shell=shell,
            scope=scope,
            created_at=utc_now_iso(),
            metadata=dict(metadata) if isinstance(metadata, dict) else {},
            command_payload=dict(command_payload) if isinstance(command_payload, dict) else None,
            execution_result=None,
            resolved_at=None,
        )
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO exec_approval_requests(
                        approval_id,
                        status,
                        reason,
                        command,
                        shell,
                        scope,
                        created_at,
                        metadata_json,
                        command_payload_json,
                        execution_result_json,
                        resolved_at,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        request.approval_id,
                        request.status,
                        request.reason,
                        request.command,
                        request.shell,
                        request.scope,
                        request.created_at,
                        json.dumps(request.metadata, ensure_ascii=False),
                        (
                            json.dumps(request.command_payload, ensure_ascii=False)
                            if request.command_payload is not None
                            else None
                        ),
                        None,
                        None,
                        request.created_at,
                    ),
                )
                conn.commit()
        return request

    def get(self, approval_id: str) -> ExecApprovalRequest | None:
        with self._lock:
            with self._connect() as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT * FROM exec_approval_requests WHERE approval_id=?",
                    (approval_id,),
                ).fetchone()
        if row is None:
            return None
        return self._row_to_request(row)

    def list(self, *, status: ApprovalStatus | None = None) -> list[ExecApprovalRequest]:
        with self._lock:
            with self._connect() as conn:
                conn.row_factory = sqlite3.Row
                if status is None:
                    rows = conn.execute(
                        """
                        SELECT * FROM exec_approval_requests
                        ORDER BY datetime(created_at) DESC, approval_id DESC
                        """
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT * FROM exec_approval_requests
                        WHERE status=?
                        ORDER BY datetime(created_at) DESC, approval_id DESC
                        """,
                        (status,),
                    ).fetchall()
        return [self._row_to_request(row) for row in rows]

    def resolve(
        self,
        approval_id: str,
        *,
        decision: ApprovalDecision,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
        execution_result: dict[str, Any] | None = None,
    ) -> ExecApprovalRequest | None:
        with self._lock:
            with self._connect() as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT * FROM exec_approval_requests WHERE approval_id=?",
                    (approval_id,),
                ).fetchone()
                if row is None:
                    return None
                current = self._row_to_request(row)
                next_status: ApprovalStatus = (
                    "approved_once"
                    if decision == "approve_once"
                    else "approved_and_saved_rule"
                    if decision == "approve_and_save_rule"
                    else "rejected"
                )
                resolved = ExecApprovalRequest(
                    approval_id=current.approval_id,
                    status=next_status,
                    reason=reason if reason is not None else current.reason,
                    command=current.command,
                    shell=current.shell,
                    scope=current.scope,
                    created_at=current.created_at,
                    metadata=(
                        {**current.metadata, **metadata}
                        if isinstance(metadata, dict)
                        else dict(current.metadata)
                    ),
                    command_payload=current.command_payload,
                    execution_result=(
                        dict(execution_result)
                        if isinstance(execution_result, dict)
                        else current.execution_result
                    ),
                    resolved_at=utc_now_iso(),
                )
                conn.execute(
                    """
                    UPDATE exec_approval_requests
                    SET status=?,
                        reason=?,
                        metadata_json=?,
                        command_payload_json=?,
                        execution_result_json=?,
                        resolved_at=?,
                        updated_at=?
                    WHERE approval_id=?
                    """,
                    (
                        resolved.status,
                        resolved.reason,
                        json.dumps(resolved.metadata, ensure_ascii=False),
                        (
                            json.dumps(resolved.command_payload, ensure_ascii=False)
                            if resolved.command_payload is not None
                            else None
                        ),
                        (
                            json.dumps(resolved.execution_result, ensure_ascii=False)
                            if resolved.execution_result is not None
                            else None
                        ),
                        resolved.resolved_at,
                        resolved.resolved_at,
                        resolved.approval_id,
                    ),
                )
                conn.commit()
        return resolved

    def _init_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS exec_approval_requests (
                    approval_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    reason TEXT,
                    command TEXT NOT NULL,
                    shell TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    command_payload_json TEXT,
                    execution_result_json TEXT,
                    resolved_at TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path, timeout=5.0)

    @staticmethod
    def _row_to_request(row: sqlite3.Row) -> ExecApprovalRequest:
        payload = dict(row)
        return ExecApprovalRequest(
            approval_id=str(payload.get("approval_id") or ""),
            status=str(payload.get("status") or "pending"),
            reason=payload.get("reason") if isinstance(payload.get("reason"), str) else None,
            command=str(payload.get("command") or ""),
            shell=str(payload.get("shell") or "auto"),
            scope=str(payload.get("scope") or "dangerous_command"),
            created_at=str(payload.get("created_at") or ""),
            metadata=(
                json.loads(payload.get("metadata_json") or "{}")
                if isinstance(payload.get("metadata_json"), str)
                else {}
            ),
            command_payload=(
                json.loads(payload.get("command_payload_json"))
                if isinstance(payload.get("command_payload_json"), str)
                and payload.get("command_payload_json")
                else None
            ),
            execution_result=(
                json.loads(payload.get("execution_result_json"))
                if isinstance(payload.get("execution_result_json"), str)
                and payload.get("execution_result_json")
                else None
            ),
            resolved_at=(
                payload.get("resolved_at")
                if isinstance(payload.get("resolved_at"), str) and payload.get("resolved_at")
                else None
            ),
        )
