"""Request-bound, consume-once approval lifecycle implementations."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Lock
from typing import Any, Mapping, Protocol, cast
from uuid import uuid4

from mochi.runtime.approval_state_machine import (
    APPROVAL_OWNER_TASK_ID_KEY,
    DEFAULT_APPROVAL_TTL_SECONDS,
    DEFAULT_CONSUME_LEASE_SECONDS,
    ApprovalConflict,
    ApprovalDecision,
    ApprovalError,
    ApprovalExpired,
    ApprovalLifecycleState,
    ApprovalRequesterMismatch,
    ApprovalStatus,
    ConsumeRecoveryOutcome,
    ResolutionKind,
    can_record_execution_result,
    claim_approval,
    complete_approval,
    recover_approval,
    resolve_approval,
    supersede_approval,
    utc_now_iso,
    validate_ttl,
)
from mochi.runtime.approval_state_machine import (
    future_iso as _future_iso,
)
from mochi.runtime.approval_state_machine import (
    is_expired as _is_expired,
)
from mochi.runtime.security_audit import audit_details


def _canonical_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _default_request_digest(
    command: str,
    shell: str,
    scope: str,
    command_payload: dict[str, Any] | None,
) -> str:
    return _canonical_digest(
        {
            "command_utf8_sha256": hashlib.sha256(command.encode("utf-8")).hexdigest(),
            "shell": shell,
            "scope": scope,
            "command_payload": command_payload,
        }
    )


@dataclass(frozen=True)
class ExecApprovalRequest:
    approval_id: str
    status: ApprovalStatus
    reason: str | None
    command: str
    shell: str
    scope: str
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])
    command_payload: dict[str, Any] | None = None
    execution_result: dict[str, Any] | None = None
    resolved_at: str | None = None
    requester_id: str = "legacy"
    request_digest: str = ""
    context_digest: str = ""
    expires_at: str = ""
    resolution_kind: ResolutionKind | None = None
    execution_idempotency_key: str | None = None
    consume_lease_owner: str | None = None
    consume_lease_token: str | None = None
    consume_lease_expires_at: str | None = None
    consumed_at: str | None = None
    # Source revision for projections.  This is deliberately independent from
    # updated_at so equal timestamps cannot collapse two lifecycle transitions.
    approval_revision: int | None = 1


@dataclass(frozen=True)
class ContinuationReconciliationClaim:
    approval_id: str
    lease_token: str
    lease_expires_at: str


@dataclass(frozen=True)
class ContinuationReconciliationCandidate:
    """One durably applied ordinary-Chat continuation eligible for recovery."""

    approval: ExecApprovalRequest
    status: str
    schema_version: int


class ApprovalStore(Protocol):
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
        requester_id: str | None = None,
        request_digest: str | None = None,
        context_digest: str | None = None,
        expires_at: str | None = None,
        ttl_seconds: int = DEFAULT_APPROVAL_TTL_SECONDS,
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
        requester_id: str | None = None,
        request_digest: str | None = None,
        context_digest: str | None = None,
        rule_side_effect: dict[str, Any] | None = None,
    ) -> ExecApprovalRequest | None: ...

    def consume(
        self,
        approval_id: str,
        *,
        execution_idempotency_key: str,
        lease_owner: str,
        requester_id: str | None = None,
        request_digest: str | None = None,
        context_digest: str | None = None,
        lease_seconds: int = DEFAULT_CONSUME_LEASE_SECONDS,
    ) -> ExecApprovalRequest: ...

    def complete_consumption(
        self,
        approval_id: str,
        *,
        execution_idempotency_key: str,
        lease_owner: str,
        lease_token: str,
        execution_result: dict[str, Any],
        succeeded: bool = True,
    ) -> ExecApprovalRequest: ...

    def recover_consumption(
        self,
        approval_id: str,
        *,
        outcome: ConsumeRecoveryOutcome,
        execution_result: dict[str, Any] | None = None,
    ) -> ExecApprovalRequest: ...

    def record_execution_result(
        self,
        approval_id: str,
        *,
        execution_result: dict[str, Any],
    ) -> ExecApprovalRequest | None: ...

    def claim_continuation_reconciliation(
        self, approval_id: str, *, lease_owner: str, lease_seconds: int = DEFAULT_CONSUME_LEASE_SECONDS,
    ) -> ContinuationReconciliationClaim: ...
    def complete_continuation_reconciliation(
        self, approval_id: str, *, lease_token: str, execution_result: dict[str, Any],
    ) -> ExecApprovalRequest: ...
    def fail_continuation_reconciliation(
        self, approval_id: str, *, lease_token: str, reason: str,
    ) -> ExecApprovalRequest: ...

    def list_continuation_reconciliation_candidates(
        self, *, limit: int,
    ) -> list[ContinuationReconciliationCandidate]: ...

    def supersede(
        self,
        approval_id: str,
        *,
        reason: str | None = None,
    ) -> ExecApprovalRequest: ...

    def recover_stale_consumptions(self) -> list[ExecApprovalRequest]: ...
    def list_side_effects(self, approval_id: str) -> list[dict[str, Any]]: ...


def _new_request(
    *,
    approval_id: str,
    command: str,
    shell: str,
    scope: str,
    reason: str | None,
    metadata: dict[str, Any] | None,
    command_payload: dict[str, Any] | None,
    requester_id: str | None,
    request_digest: str | None,
    context_digest: str | None,
    expires_at: str | None,
    ttl_seconds: int,
) -> ExecApprovalRequest:
    validate_ttl(ttl_seconds, expires_at=expires_at)
    payload = dict(command_payload) if isinstance(command_payload, dict) else None
    digest = request_digest or _default_request_digest(command, shell, scope, payload)
    return ExecApprovalRequest(
        approval_id=approval_id,
        status="pending",
        reason=reason,
        command=command,
        shell=shell,
        scope=scope,
        created_at=utc_now_iso(),
        metadata=dict(metadata) if isinstance(metadata, dict) else {},
        command_payload=payload,
        requester_id=requester_id or "legacy",
        request_digest=digest,
        context_digest=context_digest or digest,
        expires_at=expires_at or _future_iso(ttl_seconds),
        # New rows always carry an explicit revision.  ``None`` is reserved
        # for rows that genuinely predate this column.
        approval_revision=1,
    )


def _next_approval_revision(value: int | None) -> int:
    """Advance a durable source revision without inventing a legacy value."""

    return (value if type(value) is int and value > 0 else 0) + 1


def _lifecycle_state(request: ExecApprovalRequest) -> ApprovalLifecycleState:
    return ApprovalLifecycleState(
        approval_id=request.approval_id,
        status=request.status,
        requester_id=request.requester_id,
        request_digest=request.request_digest,
        context_digest=request.context_digest,
        expires_at=request.expires_at,
        resolution_kind=request.resolution_kind,
        resolved_at=request.resolved_at,
        execution_idempotency_key=request.execution_idempotency_key,
        consume_lease_owner=request.consume_lease_owner,
        consume_lease_token=request.consume_lease_token,
        consume_lease_expires_at=request.consume_lease_expires_at,
        consumed_at=request.consumed_at,
    )


def build_rule_side_effect(
    approval_id: str,
    value: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("approve_and_save_rule requires a durable rule side effect payload.")
    payload = value.get("payload")
    target = value.get("target_config_path")
    if not isinstance(payload, dict) or not isinstance(target, str) or not target.strip():
        raise ValueError("rule side effect requires payload and target_config_path.")
    typed_payload = cast(dict[str, Any], payload)
    now = utc_now_iso()
    return {
        "side_effect_id": str(uuid4()),
        "approval_id": approval_id,
        "kind": "save_command_rule",
        "payload": deepcopy(typed_payload),
        "payload_digest": _canonical_digest(typed_payload),
        "target_config_path": target,
        "status": "pending",
        "attempts": 0,
        "lease_owner": None,
        "lease_expires_at": None,
        "last_error": None,
        "created_at": now,
        "delivered_at": None,
        "updated_at": now,
    }


class InMemoryApprovalStore:
    def __init__(self) -> None:
        self._items: dict[str, ExecApprovalRequest] = {}
        self._effects: dict[str, list[dict[str, Any]]] = {}
        self._execution_keys: set[str] = set()
        self._lock = Lock()

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
        requester_id: str | None = None,
        request_digest: str | None = None,
        context_digest: str | None = None,
        expires_at: str | None = None,
        ttl_seconds: int = DEFAULT_APPROVAL_TTL_SECONDS,
    ) -> ExecApprovalRequest:
        item = _new_request(
            approval_id=approval_id,
            command=command,
            shell=shell,
            scope=scope,
            reason=reason,
            metadata=metadata,
            command_payload=command_payload,
            requester_id=requester_id,
            request_digest=request_digest,
            context_digest=context_digest,
            expires_at=expires_at,
            ttl_seconds=ttl_seconds,
        )
        with self._lock:
            if approval_id in self._items:
                raise ApprovalConflict(f"Approval {approval_id} already exists.")
            stored = deepcopy(item)
            self._items[approval_id] = stored
        return deepcopy(stored)

    def get(self, approval_id: str) -> ExecApprovalRequest | None:
        with self._lock:
            item = self._items.get(approval_id)
            return deepcopy(item) if item is not None else None

    def list(self, *, status: ApprovalStatus | None = None) -> list[ExecApprovalRequest]:
        with self._lock:
            items = [deepcopy(item) for item in self._items.values()]
        if status is not None:
            items = [item for item in items if item.status == status]
        return sorted(items, key=lambda item: item.created_at, reverse=True)

    def resolve(
        self,
        approval_id: str,
        *,
        decision: ApprovalDecision,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
        execution_result: dict[str, Any] | None = None,
        requester_id: str | None = None,
        request_digest: str | None = None,
        context_digest: str | None = None,
        rule_side_effect: dict[str, Any] | None = None,
    ) -> ExecApprovalRequest | None:
        effect = (
            build_rule_side_effect(approval_id, rule_side_effect)
            if decision == "approve_and_save_rule"
            else None
        )
        with self._lock:
            current = self._items.get(approval_id)
            if current is None:
                return None
            try:
                transition = resolve_approval(
                    _lifecycle_state(current),
                    decision=decision,
                    requester_id=requester_id,
                    request_digest=request_digest,
                    context_digest=context_digest,
                )
            except ApprovalExpired:
                self._items[approval_id] = replace(
                    current,
                    status="expired",
                    approval_revision=_next_approval_revision(current.approval_revision),
                )
                raise
            if effect is not None:
                self._effects.setdefault(approval_id, []).append(effect)
            merged = {**current.metadata, **metadata} if isinstance(metadata, dict) else current.metadata
            resolved = replace(
                current,
                status=transition.status,
                reason=reason if reason is not None else current.reason,
                metadata=dict(merged),
                execution_result=(
                    audit_details(execution_result)
                    if isinstance(execution_result, dict)
                    else current.execution_result
                ),
                resolved_at=transition.resolved_at,
                resolution_kind=transition.resolution_kind,
                approval_revision=_next_approval_revision(current.approval_revision),
            )
            self._items[approval_id] = deepcopy(resolved)
            return deepcopy(resolved)

    def consume(
        self,
        approval_id: str,
        *,
        execution_idempotency_key: str,
        lease_owner: str,
        requester_id: str | None = None,
        request_digest: str | None = None,
        context_digest: str | None = None,
        lease_seconds: int = DEFAULT_CONSUME_LEASE_SECONDS,
    ) -> ExecApprovalRequest:
        with self._lock:
            current = self._items.get(approval_id)
            if current is None:
                raise ApprovalConflict(f"Approval {approval_id} does not exist.")
            try:
                transition = claim_approval(
                    _lifecycle_state(current),
                    execution_idempotency_key=execution_idempotency_key,
                    lease_owner=lease_owner,
                    requester_id=requester_id,
                    request_digest=request_digest,
                    context_digest=context_digest,
                    lease_seconds=lease_seconds,
                )
            except ApprovalExpired:
                self._items[approval_id] = replace(
                    current,
                    status="expired",
                    approval_revision=_next_approval_revision(current.approval_revision),
                )
                raise
            if (
                execution_idempotency_key in self._execution_keys
                and current.execution_idempotency_key != execution_idempotency_key
            ):
                raise ApprovalConflict("Execution idempotency key is already in use.")
            self._execution_keys.add(execution_idempotency_key)
            claimed = replace(
                current,
                status=transition.status,
                execution_idempotency_key=transition.execution_idempotency_key,
                consume_lease_owner=transition.consume_lease_owner,
                consume_lease_token=transition.consume_lease_token,
                consume_lease_expires_at=transition.consume_lease_expires_at,
                approval_revision=_next_approval_revision(current.approval_revision),
            )
            self._items[approval_id] = deepcopy(claimed)
            return deepcopy(claimed)

    def complete_consumption(
        self,
        approval_id: str,
        *,
        execution_idempotency_key: str,
        lease_owner: str,
        lease_token: str,
        execution_result: dict[str, Any],
        succeeded: bool = True,
    ) -> ExecApprovalRequest:
        with self._lock:
            current = self._items.get(approval_id)
            if current is None:
                raise ApprovalConflict("Approval consumption is not owned by this lease.")
            transition = complete_approval(
                _lifecycle_state(current),
                execution_idempotency_key=execution_idempotency_key,
                lease_owner=lease_owner,
                lease_token=lease_token,
                succeeded=succeeded,
            )
            completed = replace(
                current,
                status=transition.status,
                execution_result=audit_details(execution_result),
                consume_lease_owner=transition.consume_lease_owner,
                consume_lease_token=transition.consume_lease_token,
                consume_lease_expires_at=transition.consume_lease_expires_at,
                consumed_at=transition.consumed_at,
                approval_revision=_next_approval_revision(current.approval_revision),
            )
            self._items[approval_id] = deepcopy(completed)
            return deepcopy(completed)

    def recover_consumption(
        self,
        approval_id: str,
        *,
        outcome: ConsumeRecoveryOutcome,
        execution_result: dict[str, Any] | None = None,
    ) -> ExecApprovalRequest:
        with self._lock:
            current = self._items.get(approval_id)
            if current is None:
                raise ApprovalConflict("Approval has no consuming lease to recover.")
            transition = recover_approval(
                _lifecycle_state(current),
                outcome=outcome,
            )
            recovered = replace(
                current,
                status=transition.status,
                execution_result=(
                    audit_details(execution_result)
                    if isinstance(execution_result, dict)
                    else current.execution_result
                ),
                consume_lease_owner=transition.consume_lease_owner,
                consume_lease_token=transition.consume_lease_token,
                consume_lease_expires_at=transition.consume_lease_expires_at,
                consumed_at=transition.consumed_at,
                approval_revision=_next_approval_revision(current.approval_revision),
            )
            self._items[approval_id] = deepcopy(recovered)
            return deepcopy(recovered)

    def record_execution_result(
        self,
        approval_id: str,
        *,
        execution_result: dict[str, Any],
    ) -> ExecApprovalRequest | None:
        with self._lock:
            current = self._items.get(approval_id)
            if current is None or not can_record_execution_result(
                _lifecycle_state(current)
            ):
                return None
            updated = replace(
                current,
                execution_result=audit_details(execution_result),
                approval_revision=_next_approval_revision(current.approval_revision),
            )
            self._items[approval_id] = deepcopy(updated)
            return deepcopy(updated)

    def supersede(
        self,
        approval_id: str,
        *,
        reason: str | None = None,
    ) -> ExecApprovalRequest:
        with self._lock:
            current = self._items.get(approval_id)
            if current is None:
                raise ApprovalConflict("Only pending approvals can be superseded.")
            try:
                transition = supersede_approval(_lifecycle_state(current))
            except ApprovalExpired:
                self._items[approval_id] = replace(
                    current,
                    status="expired",
                    approval_revision=_next_approval_revision(current.approval_revision),
                )
                raise
            updated = replace(
                current,
                status=transition.status,
                reason=reason if reason is not None else current.reason,
                resolved_at=transition.resolved_at,
                approval_revision=_next_approval_revision(current.approval_revision),
            )
            self._items[approval_id] = deepcopy(updated)
            return deepcopy(updated)

    def recover_stale_consumptions(self) -> list[ExecApprovalRequest]:
        with self._lock:
            stale_ids = [
                item.approval_id
                for item in self._items.values()
                if item.status == 'consuming'
                and (
                    item.consume_lease_expires_at is None
                    or _is_expired(item.consume_lease_expires_at)
                )
            ]
        recovered: list[ExecApprovalRequest] = []
        for item_id in stale_ids:
            result = self._items[item_id].execution_result
            recovered.append(
                self.recover_consumption(
                    item_id,
                    outcome=(
                        'applied'
                        if isinstance(result, dict)
                        and result.get('status') in {'succeeded', 'completed'}
                        else 'unknown'
                    ),
                )
            )
        return recovered

    def list_side_effects(self, approval_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return deepcopy(self._effects.get(approval_id, []))


class PersistentApprovalStore:
    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._lock = Lock()
        self._initialize()

    @property
    def database_path(self) -> Path:
        return self._db_path

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
        requester_id: str | None = None,
        request_digest: str | None = None,
        context_digest: str | None = None,
        expires_at: str | None = None,
        ttl_seconds: int = DEFAULT_APPROVAL_TTL_SECONDS,
    ) -> ExecApprovalRequest:
        item = _new_request(
            approval_id=approval_id,
            command=command,
            shell=shell,
            scope=scope,
            reason=reason,
            metadata=metadata,
            command_payload=command_payload,
            requester_id=requester_id,
            request_digest=request_digest,
            context_digest=context_digest,
            expires_at=expires_at,
            ttl_seconds=ttl_seconds,
        )
        with self._lock, self._connect() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO exec_approval_requests(
                        approval_id,status,reason,command,shell,scope,created_at,
                        metadata_json,command_payload_json,execution_result_json,
                        resolved_at,updated_at,requester_id,request_digest,context_digest,
                        expires_at,resolution_kind,execution_idempotency_key,
                        consume_lease_owner,consume_lease_token,consume_lease_expires_at,consumed_at
                        ,approval_revision
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    _row_values(item, updated_at=item.created_at),
                )
            except sqlite3.IntegrityError as exc:
                raise ApprovalConflict(f"Approval {approval_id} already exists.") from exc
            conn.commit()
        return item

    def get(self, approval_id: str) -> ExecApprovalRequest | None:
        with self._lock, self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM exec_approval_requests WHERE approval_id=?",
                (approval_id,),
            ).fetchone()
        return None if row is None else _from_row(row)

    def list(self, *, status: ApprovalStatus | None = None) -> list[ExecApprovalRequest]:
        with self._lock, self._connect() as conn:
            conn.row_factory = sqlite3.Row
            sql = "SELECT * FROM exec_approval_requests"
            params: tuple[object, ...] = ()
            if status is not None:
                sql += " WHERE status=?"
                params = (status,)
            sql += " ORDER BY datetime(created_at) DESC, approval_id DESC"
            rows = conn.execute(sql, params).fetchall()
        return [_from_row(row) for row in rows]

    def resolve(
        self,
        approval_id: str,
        *,
        decision: ApprovalDecision,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
        execution_result: dict[str, Any] | None = None,
        requester_id: str | None = None,
        request_digest: str | None = None,
        context_digest: str | None = None,
        rule_side_effect: dict[str, Any] | None = None,
    ) -> ExecApprovalRequest | None:
        effect = (
            build_rule_side_effect(approval_id, rule_side_effect)
            if decision == "approve_and_save_rule"
            else None
        )
        with self._lock, self._connect() as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM exec_approval_requests WHERE approval_id=?",
                (approval_id,),
            ).fetchone()
            if row is None:
                conn.rollback()
                return None
            current = _from_row(row)
            try:
                transition = resolve_approval(
                    _lifecycle_state(current),
                    decision=decision,
                    requester_id=requester_id,
                    request_digest=request_digest,
                    context_digest=context_digest,
                )
            except ApprovalExpired:
                conn.execute(
                    "UPDATE exec_approval_requests SET status='expired',updated_at=?,approval_revision=COALESCE(approval_revision,0)+1 WHERE approval_id=? AND status IN ('pending','approved_once')",
                    (utc_now_iso(), approval_id),
                )
                conn.commit()
                raise
            now = transition.resolved_at or utc_now_iso()
            merged = {**current.metadata, **metadata} if isinstance(metadata, dict) else current.metadata
            cursor = conn.execute(
                """
                UPDATE exec_approval_requests
                SET status=?,reason=?,metadata_json=?,execution_result_json=?,
                    resolved_at=?,resolution_kind=?,updated_at=?,approval_revision=COALESCE(approval_revision,0)+1
                WHERE approval_id=? AND status='pending' AND expires_at>?
                """,
                (
                    transition.status,
                    reason if reason is not None else current.reason,
                    json.dumps(merged, ensure_ascii=False),
                    json.dumps(audit_details(execution_result), ensure_ascii=False)
                    if isinstance(execution_result, dict)
                    else None,
                    now,
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
                        target_config_path,status,attempts,lease_owner,lease_expires_at,
                        last_error,created_at,delivered_at,updated_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        effect["side_effect_id"], approval_id, effect["kind"],
                        json.dumps(effect["payload"], ensure_ascii=False),
                        effect["payload_digest"], effect["target_config_path"], "pending", 0,
                        None, None, None, effect["created_at"], None, effect["updated_at"],
                    ),
                )
            conn.commit()
        return self._required(approval_id)

    def consume(
        self,
        approval_id: str,
        *,
        execution_idempotency_key: str,
        lease_owner: str,
        requester_id: str | None = None,
        request_digest: str | None = None,
        context_digest: str | None = None,
        lease_seconds: int = DEFAULT_CONSUME_LEASE_SECONDS,
    ) -> ExecApprovalRequest:
        with self._lock, self._connect() as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM exec_approval_requests WHERE approval_id=?",
                (approval_id,),
            ).fetchone()
            if row is None:
                conn.rollback()
                raise ApprovalConflict(f"Approval {approval_id} does not exist.")
            current = _from_row(row)
            try:
                transition = claim_approval(
                    _lifecycle_state(current),
                    execution_idempotency_key=execution_idempotency_key,
                    lease_owner=lease_owner,
                    requester_id=requester_id,
                    request_digest=request_digest,
                    context_digest=context_digest,
                    lease_seconds=lease_seconds,
                )
            except ApprovalExpired:
                conn.execute(
                    "UPDATE exec_approval_requests SET status='expired',updated_at=?,approval_revision=COALESCE(approval_revision,0)+1 WHERE approval_id=? AND status='approved_once'",
                    (utc_now_iso(), approval_id),
                )
                conn.commit()
                raise
            now = utc_now_iso()
            try:
                cursor = conn.execute(
                    """
                    UPDATE exec_approval_requests
                    SET status='consuming',execution_idempotency_key=?,consume_lease_owner=?,
                        consume_lease_token=?,consume_lease_expires_at=?,updated_at=?,approval_revision=COALESCE(approval_revision,0)+1
                    WHERE approval_id=? AND status='approved_once' AND expires_at>?
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
        return self._required(approval_id)

    def complete_consumption(
        self,
        approval_id: str,
        *,
        execution_idempotency_key: str,
        lease_owner: str,
        lease_token: str,
        execution_result: dict[str, Any],
        succeeded: bool = True,
    ) -> ExecApprovalRequest:
        with self._lock, self._connect() as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM exec_approval_requests WHERE approval_id=?",
                (approval_id,),
            ).fetchone()
            if row is None:
                conn.rollback()
                raise ApprovalConflict("Approval consumption is not owned by this lease.")
            transition = complete_approval(
                _lifecycle_state(_from_row(row)),
                execution_idempotency_key=execution_idempotency_key,
                lease_owner=lease_owner,
                lease_token=lease_token,
                succeeded=succeeded,
            )
            now = transition.consumed_at or utc_now_iso()
            cursor = conn.execute(
                """
                UPDATE exec_approval_requests
                    SET status=?,execution_result_json=?,consume_lease_owner=NULL,
                    consume_lease_token=NULL,consume_lease_expires_at=NULL,
                    consumed_at=?,updated_at=?,approval_revision=COALESCE(approval_revision,0)+1
                WHERE approval_id=? AND status='consuming'
                  AND execution_idempotency_key=? AND consume_lease_owner=?
                  AND consume_lease_token=?
                """,
                (
                    transition.status,
                    json.dumps(audit_details(execution_result), ensure_ascii=False),
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
        return self._required(approval_id)

    def recover_consumption(
        self,
        approval_id: str,
        *,
        outcome: ConsumeRecoveryOutcome,
        execution_result: dict[str, Any] | None = None,
    ) -> ExecApprovalRequest:
        with self._lock, self._connect() as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM exec_approval_requests WHERE approval_id=?",
                (approval_id,),
            ).fetchone()
            if row is None:
                conn.rollback()
                raise ApprovalConflict(f"Approval {approval_id} does not exist.")
            current = _from_row(row)
            transition = recover_approval(
                _lifecycle_state(current),
                outcome=outcome,
            )
            now = utc_now_iso()
            cursor = conn.execute(
                """
                UPDATE exec_approval_requests
                    SET status=?,execution_result_json=?,consume_lease_owner=NULL,
                    consume_lease_token=NULL,consume_lease_expires_at=NULL,
                    consumed_at=?,updated_at=?,approval_revision=COALESCE(approval_revision,0)+1
                WHERE approval_id=? AND status='consuming'
                  AND (consume_lease_expires_at IS NULL OR consume_lease_expires_at<=?)
                """,
                (
                    transition.status,
                    json.dumps(audit_details(execution_result), ensure_ascii=False)
                    if isinstance(execution_result, dict)
                    else current.execution_result
                    and json.dumps(current.execution_result, ensure_ascii=False),
                    transition.consumed_at,
                    now,
                    approval_id,
                    now,
                ),
            )
            if cursor.rowcount != 1:
                conn.rollback()
                raise ApprovalConflict("Approval recovery lost its compare-and-swap race.")
            conn.commit()
        return self._required(approval_id)

    def record_execution_result(
        self,
        approval_id: str,
        *,
        execution_result: dict[str, Any],
    ) -> ExecApprovalRequest | None:
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE exec_approval_requests SET execution_result_json=?,updated_at=?,approval_revision=COALESCE(approval_revision,0)+1
                WHERE approval_id=? AND status IN ('consuming','consumed','execution_failed')
                """,
                (
                    json.dumps(audit_details(execution_result), ensure_ascii=False),
                    utc_now_iso(),
                    approval_id,
                ),
            )
            conn.commit()
        return self.get(approval_id) if cursor.rowcount == 1 else None

    def list_continuation_reconciliation_candidates(
        self,
        *,
        limit: int,
    ) -> list[ContinuationReconciliationCandidate]:
        """Materialize and enumerate a bounded set of safe continuation work.

        ``execution_result.recovery_required`` predates the durable
        continuation table.  Treat it as a reader migration signal: the first
        dispatcher pass records a v1 ``pending`` row atomically.  Records with
        untrusted execution evidence are terminalized as ``unknown`` rather
        than becoming a possible replay path.
        """
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer.")
        candidates: list[ContinuationReconciliationCandidate] = []
        with self._lock, self._connect() as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """
                SELECT approval.*
                FROM exec_approval_requests AS approval
                LEFT JOIN approval_continuation_reconciliations AS continuation
                  ON continuation.approval_id=approval.approval_id
                WHERE approval.status='consumed'
                  AND json_valid(approval.execution_result_json)=1
                  AND json_extract(
                      approval.execution_result_json,
                      '$.recovery_required'
                  )=?
                  AND (
                      continuation.status IS NULL
                      OR continuation.status='pending'
                      OR (
                          continuation.status='reconciling'
                          AND (
                              continuation.lease_expires_at IS NULL
                              OR continuation.lease_expires_at<=?
                          )
                      )
                  )
                ORDER BY datetime(approval.updated_at) ASC, approval.approval_id ASC
                LIMIT ?
                """,
                ("ordinary_chat_continuation", utc_now_iso(), limit),
            ).fetchall()
            now = utc_now_iso()
            for row in rows:
                approval = _from_row(row)
                result = approval.execution_result or {}
                if result.get("recovery_required") != "ordinary_chat_continuation":
                    continue
                current = conn.execute(
                    "SELECT status,schema_version FROM approval_continuation_reconciliations WHERE approval_id=?",
                    (approval.approval_id,),
                ).fetchone()
                if not _is_trusted_ordinary_chat_continuation(approval):
                    if current is None:
                        conn.execute(
                            """
                            INSERT INTO approval_continuation_reconciliations(
                                approval_id,status,schema_version,lease_owner,lease_token,
                                lease_expires_at,reason,created_at,updated_at
                            ) VALUES (?,?,?,?,?,?,?,?,?)
                            """,
                            (
                                approval.approval_id, "unknown", 1, None, None,
                                None, "applied_execution_evidence_missing", now, now,
                            ),
                        )
                    continue
                if current is None:
                    conn.execute(
                        """
                        INSERT INTO approval_continuation_reconciliations(
                            approval_id,status,schema_version,lease_owner,lease_token,
                            lease_expires_at,reason,created_at,updated_at
                        ) VALUES (?,?,?,?,?,?,?,?,?)
                        """,
                        (approval.approval_id, "pending", 1, None, None, None, None, now, now),
                    )
                    status, schema_version = "pending", 1
                else:
                    status = str(current["status"])
                    schema_version = int(current["schema_version"] or 1)
                if status in {"pending", "reconciling"}:
                    candidates.append(
                        ContinuationReconciliationCandidate(
                            approval=approval,
                            status=status,
                            schema_version=schema_version,
                        )
                    )
            conn.commit()
        return candidates

    def claim_continuation_reconciliation(
        self,
        approval_id: str,
        *,
        lease_owner: str,
        lease_seconds: int = DEFAULT_CONSUME_LEASE_SECONDS,
    ) -> ContinuationReconciliationClaim:
        """Claim the one-shot post-crash ReAct continuation in SQLite."""
        with self._lock, self._connect() as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
            approval = conn.execute(
                "SELECT status,execution_result_json FROM exec_approval_requests WHERE approval_id=?",
                (approval_id,),
            ).fetchone()
            if approval is None or approval["status"] != "consumed":
                conn.rollback()
                raise ApprovalConflict("Applied approval evidence is unavailable for reconciliation.")
            result = _json_object(approval["execution_result_json"])
            if result.get("recovery_required") != "ordinary_chat_continuation":
                conn.rollback()
                raise ApprovalConflict("Approval is not awaiting continuation reconciliation.")
            current = conn.execute(
                "SELECT status,lease_expires_at FROM approval_continuation_reconciliations WHERE approval_id=?",
                (approval_id,),
            ).fetchone()
            if current is not None:
                status = str(current["status"])
                if status == "pending":
                    token = f"continuation-{uuid4().hex}"
                    expires_at = _future_iso(lease_seconds)
                    now = utc_now_iso()
                    cursor = conn.execute(
                        """
                        UPDATE approval_continuation_reconciliations
                        SET status='reconciling',lease_owner=?,lease_token=?,
                            lease_expires_at=?,reason=NULL,updated_at=?
                        WHERE approval_id=? AND status='pending'
                        """,
                        (lease_owner, token, expires_at, now, approval_id),
                    )
                    if cursor.rowcount != 1:
                        conn.rollback()
                        raise ApprovalConflict("Continuation reconciliation lost its compare-and-swap race.")
                    conn.commit()
                    return ContinuationReconciliationClaim(approval_id, token, expires_at)
                if status == "reconciling":
                    expiry = str(current["lease_expires_at"] or "")
                    if not expiry or _is_expired(expiry):
                        conn.execute(
                            "UPDATE approval_continuation_reconciliations SET status='unknown',updated_at=? WHERE approval_id=? AND status='reconciling'",
                            (utc_now_iso(), approval_id),
                        )
                        conn.commit()
                        raise ApprovalConflict("Continuation lease expired with unknown outcome; manual recovery is required.")
                    conn.rollback()
                    raise ApprovalConflict("Continuation reconciliation is already in progress.")
                conn.rollback()
                raise ApprovalConflict(f"Continuation reconciliation is terminal: {status}.")
            token = f"continuation-{uuid4().hex}"
            expires_at = _future_iso(lease_seconds)
            now = utc_now_iso()
            conn.execute(
                """
                INSERT INTO approval_continuation_reconciliations(
                    approval_id,status,schema_version,lease_owner,lease_token,
                    lease_expires_at,reason,created_at,updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (approval_id, "reconciling", 1, lease_owner, token, expires_at, None, now, now),
            )
            conn.commit()
        return ContinuationReconciliationClaim(approval_id, token, expires_at)

    def complete_continuation_reconciliation(
        self,
        approval_id: str,
        *,
        lease_token: str,
        execution_result: dict[str, Any],
    ) -> ExecApprovalRequest:
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            now = utc_now_iso()
            cursor = conn.execute(
                "UPDATE approval_continuation_reconciliations SET status='continued',lease_owner=NULL,lease_token=NULL,lease_expires_at=NULL,updated_at=? WHERE approval_id=? AND status='reconciling' AND lease_token=?",
                (now, approval_id, lease_token),
            )
            if cursor.rowcount != 1:
                conn.rollback()
                raise ApprovalConflict("Continuation completion lost its lease compare-and-swap race.")
            result_cursor = conn.execute(
                "UPDATE exec_approval_requests SET execution_result_json=?,updated_at=?,approval_revision=COALESCE(approval_revision,0)+1 WHERE approval_id=? AND status='consumed'",
                (json.dumps(audit_details(execution_result), ensure_ascii=False), now, approval_id),
            )
            if result_cursor.rowcount != 1:
                conn.rollback()
                raise ApprovalConflict("Approval disappeared before continuation completion.")
            conn.commit()
        return self._required(approval_id)

    def fail_continuation_reconciliation(
        self,
        approval_id: str,
        *,
        lease_token: str,
        reason: str,
    ) -> ExecApprovalRequest:
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                "UPDATE approval_continuation_reconciliations SET status='unknown',lease_owner=NULL,lease_token=NULL,lease_expires_at=NULL,reason=?,updated_at=? WHERE approval_id=? AND status='reconciling' AND lease_token=?",
                (reason, utc_now_iso(), approval_id, lease_token),
            )
            if cursor.rowcount != 1:
                conn.rollback()
                raise ApprovalConflict("Continuation failure lost its lease compare-and-swap race.")
            conn.commit()
        return self._required(approval_id)

    def supersede(
        self,
        approval_id: str,
        *,
        reason: str | None = None,
    ) -> ExecApprovalRequest:
        with self._lock, self._connect() as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM exec_approval_requests WHERE approval_id=?",
                (approval_id,),
            ).fetchone()
            if row is None:
                conn.rollback()
                raise ApprovalConflict("Only pending approvals can be superseded.")
            current = _from_row(row)
            try:
                transition = supersede_approval(_lifecycle_state(current))
            except ApprovalExpired:
                conn.execute(
                    "UPDATE exec_approval_requests SET status='expired',updated_at=?,approval_revision=COALESCE(approval_revision,0)+1 WHERE approval_id=? AND status='pending'",
                    (utc_now_iso(), approval_id),
                )
                conn.commit()
                raise
            now = transition.resolved_at or utc_now_iso()
            cursor = conn.execute(
                "UPDATE exec_approval_requests SET status=?,reason=?,resolved_at=?,updated_at=?,approval_revision=COALESCE(approval_revision,0)+1 WHERE approval_id=? AND status='pending' AND expires_at>?",
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
                raise ApprovalConflict("Approval supersede lost its compare-and-swap race.")
            conn.commit()
        return self._required(approval_id)

    def recover_stale_consumptions(self) -> list[ExecApprovalRequest]:
        """Atomically recover stale exact calls and mark Chat continuation work.

        A previous implementation changed ``consuming`` to ``consumed`` and
        then let RuntimeService add ``recovery_required`` in a separate write.
        A crash in that gap made an applied Chat mutation undiscoverable.
        """
        recovered: list[ExecApprovalRequest] = []
        with self._lock, self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT approval_id FROM exec_approval_requests WHERE status='consuming' AND (consume_lease_expires_at IS NULL OR consume_lease_expires_at<=?)",
                (utc_now_iso(),),
            ).fetchall()
        for row in rows:
            approval_id = str(row["approval_id"])
            with self._lock, self._connect() as conn:
                conn.row_factory = sqlite3.Row
                conn.execute("BEGIN IMMEDIATE")
                current_row = conn.execute(
                    "SELECT * FROM exec_approval_requests WHERE approval_id=?",
                    (approval_id,),
                ).fetchone()
                if current_row is None:
                    conn.rollback()
                    continue
                current = _from_row(current_row)
                if (
                    current.status != "consuming"
                    or (
                        current.consume_lease_expires_at is not None
                        and not _is_expired(current.consume_lease_expires_at)
                    )
                ):
                    # Another process completed recovery or renewed the lease.
                    conn.rollback()
                    continue
                result = dict(current.execution_result or {})
                applied = result.get("status") in {"succeeded", "completed"}
                status = (
                    "expired"
                    if _is_expired(current.expires_at)
                    else "consumed" if applied else "execution_failed"
                )
                if status == "consumed" and _has_ordinary_chat_checkpoint(current.command_payload):
                    continuation = result.get("react_continuation")
                    if result.get("continuation_started") is True and not isinstance(continuation, Mapping):
                        result["continuation_outcome"] = "unknown"
                        result.pop("recovery_required", None)
                    elif not isinstance(continuation, Mapping):
                        result["recovery_required"] = "ordinary_chat_continuation"
                now = utc_now_iso()
                cursor = conn.execute(
                    "UPDATE exec_approval_requests SET status=?,execution_result_json=?,consume_lease_owner=NULL,consume_lease_token=NULL,consume_lease_expires_at=NULL,consumed_at=?,updated_at=?,approval_revision=COALESCE(approval_revision,0)+1 WHERE approval_id=? AND status='consuming' AND (consume_lease_expires_at IS NULL OR consume_lease_expires_at<=?)",
                    (
                        status,
                        json.dumps(audit_details(result), ensure_ascii=False) if result else None,
                        now,
                        now,
                        approval_id,
                        now,
                    ),
                )
                if cursor.rowcount != 1:
                    conn.rollback()
                    continue
                conn.commit()
            recovered.append(self._required(approval_id))
        return recovered

    def list_side_effects(self, approval_id: str) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM approval_side_effects WHERE approval_id=? ORDER BY created_at",
                (approval_id,),
            ).fetchall()
        output: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            raw_payload = item.pop("payload_json", None)
            try:
                item["payload"] = json.loads(raw_payload or "{}")
            except (json.JSONDecodeError, TypeError):
                item["payload"] = None
            output.append(item)
        return output

    def _required(self, approval_id: str) -> ExecApprovalRequest:
        item = self.get(approval_id)
        if item is None:
            raise ApprovalConflict("Approval disappeared after a committed transition.")
        return item

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=5.0)
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _initialize(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS exec_approval_requests(
                    approval_id TEXT PRIMARY KEY,status TEXT NOT NULL,reason TEXT,
                    command TEXT NOT NULL,shell TEXT NOT NULL,scope TEXT NOT NULL,
                    created_at TEXT NOT NULL,metadata_json TEXT NOT NULL DEFAULT '{}',
                    command_payload_json TEXT,execution_result_json TEXT,resolved_at TEXT,
                    updated_at TEXT NOT NULL,requester_id TEXT NOT NULL DEFAULT 'legacy',
                    request_digest TEXT NOT NULL DEFAULT '',context_digest TEXT NOT NULL DEFAULT '',
                    expires_at TEXT NOT NULL DEFAULT '',resolution_kind TEXT,
                    execution_idempotency_key TEXT,consume_lease_owner TEXT,
                    consume_lease_token TEXT,consume_lease_expires_at TEXT,consumed_at TEXT
                    ,approval_revision INTEGER
                )
                """
            )
            for column, definition in (
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
                # Keep pre-column rows distinguishable from new source rows.
                # ``create`` explicitly writes revision 1; a migration must
                # not fabricate that monotonic proof for existing data.
                ("approval_revision", "INTEGER"),
            ):
                _ensure_column(conn, "exec_approval_requests", column, definition)
            conn.execute(
                "UPDATE exec_approval_requests SET expires_at=? WHERE expires_at=''",
                (_future_iso(DEFAULT_APPROVAL_TTL_SECONDS),),
            )
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS exec_approval_execution_idempotency
                ON exec_approval_requests(execution_idempotency_key)
                WHERE execution_idempotency_key IS NOT NULL
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS approval_side_effects(
                    side_effect_id TEXT PRIMARY KEY,approval_id TEXT NOT NULL,
                    kind TEXT NOT NULL,payload_json TEXT NOT NULL,payload_digest TEXT NOT NULL,
                    target_config_path TEXT NOT NULL,status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,lease_owner TEXT,lease_expires_at TEXT,
                    last_error TEXT,created_at TEXT NOT NULL,delivered_at TEXT,updated_at TEXT NOT NULL,
                    FOREIGN KEY(approval_id) REFERENCES exec_approval_requests(approval_id) ON DELETE CASCADE,
                    UNIQUE(approval_id,kind,payload_digest)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS approval_continuation_reconciliations(
                    approval_id TEXT PRIMARY KEY,status TEXT NOT NULL,
                    schema_version INTEGER NOT NULL DEFAULT 1,
                    lease_owner TEXT,lease_token TEXT,lease_expires_at TEXT,
                    reason TEXT,created_at TEXT NOT NULL,updated_at TEXT NOT NULL,
                    FOREIGN KEY(approval_id) REFERENCES exec_approval_requests(approval_id)
                        ON DELETE CASCADE
                )
                """
            )
            _ensure_column(
                conn,
                "approval_continuation_reconciliations",
                "schema_version",
                "INTEGER NOT NULL DEFAULT 1",
            )
            conn.execute(
                "UPDATE approval_continuation_reconciliations SET schema_version=1 WHERE schema_version IS NULL OR schema_version<1"
            )
            conn.commit()


def _ensure_column(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    definition: str,
) -> None:
    columns = {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _json_object(value: object) -> dict[str, Any]:
    if not isinstance(value, str) or not value:
        return {}
    loaded: object = json.loads(value)
    return cast(dict[str, Any], loaded) if isinstance(loaded, dict) else {}


def _optional_json_object(value: object) -> dict[str, Any] | None:
    loaded = _json_object(value)
    return loaded or None


def _has_ordinary_chat_checkpoint(payload: Mapping[str, Any] | None) -> bool:
    checkpoint = payload.get("ordinary_chat_checkpoint") if isinstance(payload, Mapping) else None
    return isinstance(checkpoint, Mapping) and checkpoint.get("source") == "ordinary_chat"


def _is_trusted_ordinary_chat_continuation(approval: ExecApprovalRequest) -> bool:
    """Only a durable, successfully applied ordinary-Chat result can resume."""
    result = approval.execution_result
    if (
        approval.status != "consumed"
        or not isinstance(result, Mapping)
        or result.get("recovery_required") != "ordinary_chat_continuation"
        or result.get("status") not in {"succeeded", "completed"}
        or result.get("continuation_started") is True
    ):
        return False
    metadata = approval.metadata
    payload = approval.command_payload
    checkpoint = (
        payload.get("ordinary_chat_checkpoint")
        if isinstance(payload, Mapping)
        else None
    )
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("approval_source") != "ordinary_chat"
        or not isinstance(payload, Mapping)
        or not isinstance(checkpoint, Mapping)
        or checkpoint.get("source") != "ordinary_chat"
        or not isinstance(checkpoint.get("react_continuation"), Mapping)
    ):
        return False
    tool_name = payload.get("tool_name")
    operation_id = payload.get("operation_id")
    arguments_digest = payload.get("arguments_digest")
    return (
        isinstance(tool_name, str)
        and bool(tool_name)
        and isinstance(operation_id, str)
        and bool(operation_id)
        and isinstance(arguments_digest, str)
        and bool(arguments_digest)
        and checkpoint.get("tool_name") == tool_name
        and checkpoint.get("operation_id") == operation_id
        and checkpoint.get("arguments_digest") == arguments_digest
        and metadata.get("operation_id") == operation_id
        and metadata.get("arguments_digest") == arguments_digest
    )


def _from_row(row: sqlite3.Row) -> ExecApprovalRequest:
    value = dict(row)
    return ExecApprovalRequest(
        approval_id=str(value.get("approval_id") or ""),
        status=str(value.get("status") or "pending"),  # type: ignore[arg-type]
        reason=_optional_string(value.get("reason")),
        command=str(value.get("command") or ""),
        shell=str(value.get("shell") or "auto"),
        scope=str(value.get("scope") or "dangerous_command"),
        created_at=str(value.get("created_at") or ""),
        metadata=_json_object(value.get("metadata_json")),
        command_payload=_optional_json_object(value.get("command_payload_json")),
        execution_result=_optional_json_object(value.get("execution_result_json")),
        resolved_at=_optional_string(value.get("resolved_at")),
        requester_id=str(value.get("requester_id") or "legacy"),
        request_digest=str(value.get("request_digest") or ""),
        context_digest=str(value.get("context_digest") or ""),
        expires_at=str(value.get("expires_at") or _future_iso(DEFAULT_APPROVAL_TTL_SECONDS)),
        resolution_kind=_optional_string(value.get("resolution_kind")),  # type: ignore[arg-type]
        execution_idempotency_key=_optional_string(value.get("execution_idempotency_key")),
        consume_lease_owner=_optional_string(value.get("consume_lease_owner")),
        consume_lease_token=_optional_string(value.get("consume_lease_token")),
        consume_lease_expires_at=_optional_string(value.get("consume_lease_expires_at")),
        consumed_at=_optional_string(value.get("consumed_at")),
        approval_revision=(
            raw_revision
            if type(raw_revision := value.get("approval_revision")) is int
            and raw_revision > 0
            else None
        ),
    )


def _row_values(item: ExecApprovalRequest, *, updated_at: str) -> tuple[Any, ...]:
    return (
        item.approval_id,
        item.status,
        item.reason,
        item.command,
        item.shell,
        item.scope,
        item.created_at,
        json.dumps(item.metadata, ensure_ascii=False),
        json.dumps(item.command_payload, ensure_ascii=False)
        if item.command_payload is not None
        else None,
        json.dumps(item.execution_result, ensure_ascii=False)
        if item.execution_result is not None
        else None,
        item.resolved_at,
        updated_at,
        item.requester_id,
        item.request_digest,
        item.context_digest,
        item.expires_at,
        item.resolution_kind,
        item.execution_idempotency_key,
        item.consume_lease_owner,
        item.consume_lease_token,
        item.consume_lease_expires_at,
        item.consumed_at,
        item.approval_revision,
    )


__all__ = [
    "APPROVAL_OWNER_TASK_ID_KEY",
    "ApprovalConflict",
    "ApprovalDecision",
    "ApprovalError",
    "ApprovalExpired",
    "ApprovalRequesterMismatch",
    "ApprovalStatus",
    "ConsumeRecoveryOutcome",
    "ApprovalStore",
    "ContinuationReconciliationCandidate",
    "ExecApprovalRequest",
    "InMemoryApprovalStore",
    "PersistentApprovalStore",
    "utc_now_iso",
]
