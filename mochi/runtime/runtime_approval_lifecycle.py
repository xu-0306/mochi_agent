"""RuntimeStore mixin for durable approval lifecycle transitions."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Protocol, cast

from mochi.runtime.approval_lifecycle import build_rule_side_effect
from mochi.runtime.approval_state_machine import (
    DEFAULT_APPROVAL_TTL_SECONDS as _DEFAULT_TTL_SECONDS,
)
from mochi.runtime.approval_state_machine import (
    DEFAULT_CONSUME_LEASE_SECONDS as _DEFAULT_LEASE_SECONDS,
)
from mochi.runtime.approval_state_machine import (
    ApprovalConflict,
    ApprovalExpired,
    ApprovalLifecycleState,
    ApprovalStatus,
    ResolutionKind,
    claim_approval,
    complete_approval,
    normalize_approval_decision,
    recover_approval,
    recovery_outcome_for_task_status,
    resolve_approval,
    supersede_approval,
    validate_ttl,
)
from mochi.runtime.approval_state_machine import (
    future_iso as _future_iso,
)
from mochi.runtime.approval_state_machine import (
    utc_now_iso as _now_iso,
)


class _RuntimeStoreShape(Protocol):
    @property
    def database_path(self) -> Path: ...

    async def initialize(self) -> None: ...

    async def get_approval_request(
        self,
        approval_id: str,
    ) -> dict[str, Any] | None: ...


def _digest(tool_name: str, arguments: dict[str, Any], metadata: dict[str, Any]) -> str:
    payload = json.dumps(
        {"tool_name": tool_name, "arguments": arguments, "metadata": metadata},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def initialize_runtime_approval_schema(conn: sqlite3.Connection) -> None:
    """Upgrade the runtime approval table without invalidating existing rows."""
    definitions = (
        ("requester_id", "TEXT NOT NULL DEFAULT 'legacy'"),
        ("request_digest", "TEXT NOT NULL DEFAULT ''"),
        ("context_digest", "TEXT NOT NULL DEFAULT ''"),
        ("expires_at", "TEXT NOT NULL DEFAULT ''"),
        ("resolution_kind", "TEXT"),
        ("execution_idempotency_key", "TEXT"),
        ("consume_lease_owner", "TEXT"),
        ("consume_lease_token", "TEXT"),
        ("consume_lease_expires_at", "TEXT"),
        ("consumed_at", "TEXT"),
        ("execution_result_json", "TEXT"),
    )
    columns = {
        str(row[1])
        for row in conn.execute("PRAGMA table_info(approval_requests)").fetchall()
    }
    for column, definition in definitions:
        if column not in columns:
            conn.execute(f"ALTER TABLE approval_requests ADD COLUMN {column} {definition}")
    conn.execute(
        "UPDATE approval_requests SET expires_at=? WHERE expires_at=''",
        (_future_iso(_DEFAULT_TTL_SECONDS),),
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS runtime_approval_execution_idempotency
        ON approval_requests(execution_idempotency_key)
        WHERE execution_idempotency_key IS NOT NULL
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS approval_side_effects(
            side_effect_id TEXT PRIMARY KEY,
            approval_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            payload_digest TEXT NOT NULL,
            target_config_path TEXT NOT NULL,
            status TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            lease_owner TEXT,
            lease_expires_at TEXT,
            last_error TEXT,
            created_at TEXT NOT NULL,
            delivered_at TEXT,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(approval_id) REFERENCES approval_requests(id) ON DELETE CASCADE,
            UNIQUE(approval_id,kind,payload_digest)
        )
        """
    )


def _decode(row: sqlite3.Row, *, rule_status: str | None = None) -> dict[str, Any]:
    item = dict(row)
    item["arguments"] = json.loads(item.pop("arguments_json") or "{}")
    item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
    result = item.pop("execution_result_json", None)
    item["execution_result"] = json.loads(result) if isinstance(result, str) and result else None
    item["rule_persistence_status"] = rule_status or (
        "not_requested" if item.get("resolution_kind") != "approve_and_save_rule" else "pending"
    )
    return item


def _lifecycle_state(row: sqlite3.Row) -> ApprovalLifecycleState:
    return ApprovalLifecycleState(
        approval_id=str(row["id"]),
        status=cast(ApprovalStatus, str(row["status"])),
        requester_id=str(row["requester_id"]),
        request_digest=str(row["request_digest"]),
        context_digest=str(row["context_digest"]),
        expires_at=str(row["expires_at"]),
        resolution_kind=cast(ResolutionKind | None, row["resolution_kind"]),
        resolved_at=cast(str | None, row["resolved_at"]),
        execution_idempotency_key=cast(
            str | None,
            row["execution_idempotency_key"],
        ),
        consume_lease_owner=cast(str | None, row["consume_lease_owner"]),
        consume_lease_token=cast(str | None, row["consume_lease_token"]),
        consume_lease_expires_at=cast(
            str | None,
            row["consume_lease_expires_at"],
        ),
        consumed_at=cast(str | None, row["consumed_at"]),
    )



class RuntimeApprovalLifecycleMixin:
    """Methods inherited by RuntimeStore after legacy approval methods are retired."""

    async def create_approval_request(
        self: _RuntimeStoreShape,
        *,
        approval_id: str,
        task_id: str,
        call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        requester_id: str | None = None,
        request_digest: str | None = None,
        context_digest: str | None = None,
        expires_at: str | None = None,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
    ) -> dict[str, Any]:
        await self.initialize()
        validate_ttl(ttl_seconds, expires_at=expires_at)
        now = _now_iso()
        stored_metadata = dict(metadata or {})
        stored_digest = request_digest or _digest(tool_name, arguments, stored_metadata)

        def _op() -> None:
            with sqlite3.connect(self.database_path) as conn:
                try:
                    conn.execute(
                        """
                        INSERT INTO approval_requests(
                            id,task_id,call_id,tool_name,arguments_json,metadata_json,
                            status,reason,resolved_at,created_at,updated_at,
                            requester_id,request_digest,context_digest,expires_at
                        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            approval_id,
                            task_id,
                            call_id,
                            tool_name,
                            json.dumps(arguments, ensure_ascii=False),
                            json.dumps(stored_metadata, ensure_ascii=False),
                            "pending",
                            None,
                            None,
                            now,
                            now,
                            requester_id or "legacy",
                            stored_digest,
                            context_digest or stored_digest,
                            expires_at or _future_iso(ttl_seconds),
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    raise ApprovalConflict(f"Approval {approval_id} already exists.") from exc
                conn.commit()

        await asyncio.to_thread(_op)
        return await self.get_approval_request(approval_id) or {}

    async def get_approval_request(
        self: _RuntimeStoreShape,
        approval_id: str,
    ) -> dict[str, Any] | None:
        await self.initialize()

        def _op() -> dict[str, Any] | None:
            with sqlite3.connect(self.database_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT * FROM approval_requests WHERE id=?",
                    (approval_id,),
                ).fetchone()
                if row is None:
                    return None
                status_row = conn.execute(
                    "SELECT status FROM approval_side_effects WHERE approval_id=? ORDER BY created_at DESC LIMIT 1",
                    (approval_id,),
                ).fetchone()
            return _decode(row, rule_status=str(status_row[0]) if status_row else None)

        return await asyncio.to_thread(_op)
    async def list_approval_requests(
        self: _RuntimeStoreShape,
        *,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        await self.initialize()

        def _op() -> list[dict[str, Any]]:
            with sqlite3.connect(self.database_path) as conn:
                conn.row_factory = sqlite3.Row
                sql = "SELECT * FROM approval_requests"
                params: tuple[object, ...] = ()
                if status:
                    sql += " WHERE status=?"
                    params = (status,)
                sql += " ORDER BY datetime(created_at) DESC"
                rows = conn.execute(sql, params).fetchall()
                effects = {
                    str(row[0]): str(row[1])
                    for row in conn.execute(
                        "SELECT approval_id,status FROM approval_side_effects ORDER BY created_at"
                    ).fetchall()
                }
            return [_decode(row, rule_status=effects.get(str(row["id"]))) for row in rows]

        return await asyncio.to_thread(_op)
    async def resolve_approval_request(
        self: _RuntimeStoreShape,
        approval_id: str,
        *,
        decision: str,
        reason: str | None = None,
        requester_id: str | None = None,
        request_digest: str | None = None,
        context_digest: str | None = None,
        rule_side_effect: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        await self.initialize()
        normalized_decision = normalize_approval_decision(decision)
        effect = (
            build_rule_side_effect(approval_id, rule_side_effect)
            if normalized_decision == "approve_and_save_rule"
            else None
        )

        def _op() -> bool | None:
            with sqlite3.connect(self.database_path) as conn:
                conn.row_factory = sqlite3.Row
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT * FROM approval_requests WHERE id=?",
                    (approval_id,),
                ).fetchone()
                if row is None:
                    conn.rollback()
                    return None
                try:
                    transition = resolve_approval(
                        _lifecycle_state(row),
                        decision=normalized_decision,
                        requester_id=requester_id,
                        request_digest=request_digest,
                        context_digest=context_digest,
                    )
                except ApprovalExpired:
                    conn.execute(
                        "UPDATE approval_requests SET status='expired',updated_at=? WHERE id=? AND status IN ('pending','approved_once')",
                        (_now_iso(), approval_id),
                    )
                    conn.commit()
                    raise
                now = transition.resolved_at or _now_iso()
                cursor = conn.execute(
                    """
                    UPDATE approval_requests SET status=?,reason=?,resolved_at=?,
                        resolution_kind=?,updated_at=?
                    WHERE id=? AND status='pending' AND expires_at>?
                    """,
                    (
                        transition.status,
                        reason,
                        transition.resolved_at,
                        transition.resolution_kind,
                        now,
                        approval_id,
                        now,
                    ),
                )
                if cursor.rowcount != 1:
                    conn.rollback()
                    raise ApprovalConflict("Approval resolve lost its compare-and-swap race.")
                if effect is not None:
                    conn.execute(
                        """
                        INSERT INTO approval_side_effects(
                            side_effect_id,approval_id,kind,payload_json,payload_digest,
                            target_config_path,status,attempts,created_at,updated_at
                        ) VALUES (?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            effect["side_effect_id"],
                            approval_id,
                            effect["kind"],
                            json.dumps(effect["payload"], ensure_ascii=False),
                            effect["payload_digest"],
                            effect["target_config_path"],
                            "pending",
                            0,
                            effect["created_at"],
                            effect["updated_at"],
                        ),
                    )
                conn.commit()
                return True

        changed = await asyncio.to_thread(_op)
        return None if changed is None else await self.get_approval_request(approval_id)

    async def consume_approval_request(
        self: _RuntimeStoreShape,
        approval_id: str,
        *,
        execution_idempotency_key: str,
        lease_owner: str,
        requester_id: str | None = None,
        request_digest: str | None = None,
        context_digest: str | None = None,
        lease_seconds: int = _DEFAULT_LEASE_SECONDS,
    ) -> dict[str, Any]:
        await self.initialize()

        def _op() -> None:
            with sqlite3.connect(self.database_path) as conn:
                conn.row_factory = sqlite3.Row
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT * FROM approval_requests WHERE id=?",
                    (approval_id,),
                ).fetchone()
                if row is None:
                    conn.rollback()
                    raise ApprovalConflict(f"Approval {approval_id} does not exist.")
                try:
                    transition = claim_approval(
                        _lifecycle_state(row),
                        execution_idempotency_key=execution_idempotency_key,
                        lease_owner=lease_owner,
                        requester_id=requester_id,
                        request_digest=request_digest,
                        context_digest=context_digest,
                        lease_seconds=lease_seconds,
                    )
                except ApprovalExpired:
                    conn.execute(
                        "UPDATE approval_requests SET status='expired',updated_at=? WHERE id=? AND status='approved_once'",
                        (_now_iso(), approval_id),
                    )
                    conn.commit()
                    raise
                now = _now_iso()
                try:
                    cursor = conn.execute(
                        """
                        UPDATE approval_requests SET status='consuming',
                            execution_idempotency_key=?,consume_lease_owner=?,
                            consume_lease_token=?,consume_lease_expires_at=?,updated_at=?
                        WHERE id=? AND status='approved_once' AND expires_at>?
                        """,
                        (
                            transition.execution_idempotency_key,
                            transition.consume_lease_owner,
                            transition.consume_lease_token,
                            transition.consume_lease_expires_at,
                            now,
                            approval_id,
                            now,
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    conn.rollback()
                    raise ApprovalConflict("Execution idempotency key is already in use.") from exc
                if cursor.rowcount != 1:
                    conn.rollback()
                    raise ApprovalConflict("Approval consume lost its compare-and-swap race.")
                conn.commit()

        await asyncio.to_thread(_op)
        return cast(dict[str, Any], await self.get_approval_request(approval_id))

    async def complete_approval_consumption(
        self: _RuntimeStoreShape,
        approval_id: str,
        *,
        execution_idempotency_key: str,
        lease_owner: str,
        lease_token: str,
        execution_result: dict[str, Any],
        succeeded: bool = True,
    ) -> dict[str, Any]:
        await self.initialize()

        def _op() -> None:
            with sqlite3.connect(self.database_path) as conn:
                conn.row_factory = sqlite3.Row
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT * FROM approval_requests WHERE id=?",
                    (approval_id,),
                ).fetchone()
                if row is None:
                    conn.rollback()
                    raise ApprovalConflict("Approval consumption is not owned by this lease.")
                transition = complete_approval(
                    _lifecycle_state(row),
                    execution_idempotency_key=execution_idempotency_key,
                    lease_owner=lease_owner,
                    lease_token=lease_token,
                    succeeded=succeeded,
                )
                now = transition.consumed_at or _now_iso()
                cursor = conn.execute(
                    """
                    UPDATE approval_requests SET status=?,execution_result_json=?,
                        consume_lease_owner=NULL,consume_lease_token=NULL,
                        consume_lease_expires_at=NULL,consumed_at=?,updated_at=?
                    WHERE id=? AND status='consuming' AND execution_idempotency_key=?
                      AND consume_lease_owner=? AND consume_lease_token=?
                    """,
                    (
                        transition.status,
                        json.dumps(execution_result, ensure_ascii=False),
                        transition.consumed_at,
                        now,
                        approval_id,
                        execution_idempotency_key,
                        lease_owner,
                        lease_token,
                    ),
                )
                if cursor.rowcount != 1:
                    conn.rollback()
                    raise ApprovalConflict("Approval consumption is not owned by this lease.")
                conn.commit()

        await asyncio.to_thread(_op)
        return cast(dict[str, Any], await self.get_approval_request(approval_id))

    async def supersede_approval_request(
        self: _RuntimeStoreShape,
        approval_id: str,
        *,
        reason: str | None = None,
    ) -> dict[str, Any]:
        await self.initialize()

        def _op() -> None:
            with sqlite3.connect(self.database_path) as conn:
                conn.row_factory = sqlite3.Row
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT * FROM approval_requests WHERE id=?",
                    (approval_id,),
                ).fetchone()
                if row is None:
                    conn.rollback()
                    raise ApprovalConflict("Only pending approvals can be superseded.")
                try:
                    transition = supersede_approval(_lifecycle_state(row))
                except ApprovalExpired:
                    conn.execute(
                        "UPDATE approval_requests SET status='expired',updated_at=? WHERE id=? AND status='pending'",
                        (_now_iso(), approval_id),
                    )
                    conn.commit()
                    raise
                now = transition.resolved_at or _now_iso()
                cursor = conn.execute(
                    "UPDATE approval_requests SET status=?,reason=?,resolved_at=?,updated_at=? WHERE id=? AND status='pending' AND expires_at>?",
                    (
                        transition.status,
                        reason,
                        transition.resolved_at,
                        now,
                        approval_id,
                        now,
                    ),
                )
                if cursor.rowcount != 1:
                    conn.rollback()
                    raise ApprovalConflict(
                        "Approval supersede lost its compare-and-swap race."
                    )
                conn.commit()

        await asyncio.to_thread(_op)
        return cast(dict[str, Any], await self.get_approval_request(approval_id))

    async def recover_stale_approval_consumptions(
        self: _RuntimeStoreShape,
    ) -> int:
        await self.initialize()

        def _op() -> int:
            now = _now_iso()
            recovered = 0
            with sqlite3.connect(self.database_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT approval_requests.*,task_runs.status AS task_status
                    FROM approval_requests
                    LEFT JOIN task_runs ON task_runs.id=approval_requests.task_id
                    WHERE approval_requests.status='consuming'
                      AND (
                          approval_requests.consume_lease_expires_at IS NULL
                          OR approval_requests.consume_lease_expires_at<=?
                      )
                    """,
                    (now,),
                ).fetchall()
                for row in rows:
                    task_status = str(row["task_status"] or "unknown")
                    transition = recover_approval(
                        _lifecycle_state(row),
                        outcome=recovery_outcome_for_task_status(task_status),
                    )
                    recovered_result = json.dumps(
                        {
                            "status": task_status,
                            "recovered_from_stale_lease": True,
                        },
                        ensure_ascii=False,
                    )
                    cursor = conn.execute(
                        """
                        UPDATE approval_requests
                        SET status=?,execution_result_json=COALESCE(execution_result_json,?),
                            consume_lease_owner=NULL,consume_lease_token=NULL,
                            consume_lease_expires_at=NULL,consumed_at=?,updated_at=?
                        WHERE id=? AND status='consuming'
                          AND (
                              consume_lease_expires_at IS NULL
                              OR consume_lease_expires_at<=?
                          )
                        """,
                        (
                            transition.status,
                            recovered_result,
                            transition.consumed_at,
                            now,
                            str(row["id"]),
                            now,
                        ),
                    )
                    recovered += int(cursor.rowcount)
                conn.commit()
            return recovered

        return await asyncio.to_thread(_op)

    async def update_approval_request_metadata(
        self: _RuntimeStoreShape,
        approval_id: str,
        *,
        metadata: dict[str, Any],
    ) -> dict[str, Any] | None:
        await self.initialize()

        def _op() -> None:
            with sqlite3.connect(self.database_path) as conn:
                conn.execute(
                    "UPDATE approval_requests SET metadata_json=?,updated_at=? WHERE id=?",
                    (json.dumps(metadata or {}, ensure_ascii=False), _now_iso(), approval_id),
                )
                conn.commit()

        await asyncio.to_thread(_op)
        return await self.get_approval_request(approval_id)

    async def get_pending_approval_for_task(
        self: _RuntimeStoreShape,
        task_id: str,
    ) -> dict[str, Any] | None:
        await self.initialize()

        def _op() -> dict[str, Any] | None:
            with sqlite3.connect(self.database_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    """
                    SELECT * FROM approval_requests
                    WHERE task_id=? AND status='pending' AND expires_at>?
                    ORDER BY datetime(created_at) DESC LIMIT 1
                    """,
                    (task_id, _now_iso()),
                ).fetchone()
            return None if row is None else _decode(row)

        return await asyncio.to_thread(_op)
