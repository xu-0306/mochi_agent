"""Shared approval lifecycle policy independent of persistence adapters."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from uuid import uuid4

ApprovalStatus = Literal[
    "pending",
    "approved_once",
    "rejected",
    "expired",
    "superseded",
    "consuming",
    "consumed",
    "execution_failed",
]
ApprovalDecision = Literal["approve_once", "approve_and_save_rule", "reject"]
ResolutionKind = ApprovalDecision
ConsumeRecoveryOutcome = Literal["applied", "not_started", "unknown"]

DEFAULT_APPROVAL_TTL_SECONDS = 900
DEFAULT_CONSUME_LEASE_SECONDS = 30
APPROVAL_OWNER_TASK_ID_KEY = "approval_owner_task_id"

_VALID_DECISIONS: frozenset[str] = frozenset(
    {"approve_once", "approve_and_save_rule", "reject"}
)
_TERMINAL_EXECUTION_STATUSES: frozenset[ApprovalStatus] = frozenset(
    {"consumed", "execution_failed"}
)


class ApprovalError(RuntimeError):
    """Base class for typed approval lifecycle failures."""


class ApprovalExpired(ApprovalError):
    """The approval TTL elapsed before resolve or consume."""


class ApprovalConflict(ApprovalError):
    """The approval state or bound digest did not permit the transition."""


class ApprovalRequesterMismatch(ApprovalError):
    """The caller did not match the requester bound at creation."""


@dataclass(frozen=True)
class ApprovalLifecycleState:
    """Persistence-neutral fields required to decide lifecycle transitions."""

    approval_id: str
    status: ApprovalStatus
    requester_id: str
    request_digest: str
    context_digest: str
    expires_at: str
    resolution_kind: ResolutionKind | None = None
    resolved_at: str | None = None
    execution_idempotency_key: str | None = None
    consume_lease_owner: str | None = None
    consume_lease_token: str | None = None
    consume_lease_expires_at: str | None = None
    consumed_at: str | None = None


def _aware_now(now: datetime | None = None) -> datetime:
    value = now or datetime.now(UTC)
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def utc_now_iso(now: datetime | None = None) -> str:
    return _aware_now(now).isoformat()


def future_iso(seconds: int, *, now: datetime | None = None) -> str:
    return (_aware_now(now) + timedelta(seconds=seconds)).isoformat()


def is_expired(value: str, *, now: datetime | None = None) -> bool:
    return parse_timestamp(value) <= _aware_now(now)


def validate_ttl(ttl_seconds: int, *, expires_at: str | None) -> None:
    if ttl_seconds <= 0 and expires_at is None:
        raise ValueError("ttl_seconds must be positive")


def normalize_approval_decision(value: str) -> ApprovalDecision:
    if value not in _VALID_DECISIONS:
        raise ValueError(f"Unsupported approval decision: {value}")
    return value  # type: ignore[return-value]


def derive_approval_binding(
    *,
    requester_id: str,
    request: Mapping[str, Any],
    authorization_context: Mapping[str, Any],
) -> tuple[str, str, str]:
    """Build stable binding values from a canonical server-side request context."""
    request_payload = json.dumps(
        dict(request),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    request_digest = hashlib.sha256(request_payload).hexdigest()
    context_payload = json.dumps(
        {
            "request_digest": request_digest,
            "requester_id": requester_id,
            "authorization_context": dict(authorization_context),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return requester_id, request_digest, hashlib.sha256(context_payload).hexdigest()


def validate_binding(
    state: ApprovalLifecycleState,
    *,
    requester_id: str | None,
    request_digest: str | None,
    context_digest: str | None,
) -> None:
    if requester_id is not None and requester_id != state.requester_id:
        raise ApprovalRequesterMismatch(
            "Approval requester does not match the bound requester."
        )
    if request_digest is not None and request_digest != state.request_digest:
        raise ApprovalConflict("Approval request digest does not match the bound request.")
    if context_digest is not None and context_digest != state.context_digest:
        raise ApprovalConflict("Approval context digest does not match the bound context.")


def resolve_approval(
    state: ApprovalLifecycleState,
    *,
    decision: ApprovalDecision,
    requester_id: str | None = None,
    request_digest: str | None = None,
    context_digest: str | None = None,
    now: datetime | None = None,
) -> ApprovalLifecycleState:
    decision = normalize_approval_decision(decision)
    validate_binding(
        state,
        requester_id=requester_id,
        request_digest=request_digest,
        context_digest=context_digest,
    )
    if is_expired(state.expires_at, now=now):
        raise ApprovalExpired(f"Approval {state.approval_id} has expired.")
    if state.status != "pending":
        raise ApprovalConflict(
            f"Approval {state.approval_id} cannot resolve from {state.status}."
        )
    timestamp = utc_now_iso(now)
    return replace(
        state,
        status="rejected" if decision == "reject" else "approved_once",
        resolution_kind=decision,
        resolved_at=timestamp,
    )


def claim_approval(
    state: ApprovalLifecycleState,
    *,
    execution_idempotency_key: str,
    lease_owner: str,
    requester_id: str | None = None,
    request_digest: str | None = None,
    context_digest: str | None = None,
    lease_seconds: int = DEFAULT_CONSUME_LEASE_SECONDS,
    lease_token: str | None = None,
    now: datetime | None = None,
) -> ApprovalLifecycleState:
    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive")
    validate_binding(
        state,
        requester_id=requester_id,
        request_digest=request_digest,
        context_digest=context_digest,
    )
    if is_expired(state.expires_at, now=now):
        raise ApprovalExpired(f"Approval {state.approval_id} has expired.")
    if state.status != "approved_once":
        raise ApprovalConflict(
            f"Approval {state.approval_id} cannot consume from {state.status}."
        )
    if state.execution_idempotency_key not in {None, execution_idempotency_key}:
        raise ApprovalConflict("Recovered consumption must reuse its execution key.")
    return replace(
        state,
        status="consuming",
        execution_idempotency_key=execution_idempotency_key,
        consume_lease_owner=lease_owner,
        consume_lease_token=lease_token or str(uuid4()),
        consume_lease_expires_at=future_iso(lease_seconds, now=now),
    )


def complete_approval(
    state: ApprovalLifecycleState,
    *,
    execution_idempotency_key: str,
    lease_owner: str,
    lease_token: str,
    succeeded: bool = True,
    now: datetime | None = None,
) -> ApprovalLifecycleState:
    if (
        state.status != "consuming"
        or state.execution_idempotency_key != execution_idempotency_key
        or state.consume_lease_owner != lease_owner
        or state.consume_lease_token != lease_token
    ):
        raise ApprovalConflict("Approval consumption is not owned by this lease.")
    return replace(
        state,
        status="consumed" if succeeded else "execution_failed",
        consume_lease_owner=None,
        consume_lease_token=None,
        consume_lease_expires_at=None,
        consumed_at=utc_now_iso(now),
    )


def recover_approval(
    state: ApprovalLifecycleState,
    *,
    outcome: ConsumeRecoveryOutcome,
    now: datetime | None = None,
) -> ApprovalLifecycleState:
    if state.status != "consuming":
        raise ApprovalConflict("Approval has no consuming lease to recover.")
    if state.consume_lease_expires_at and not is_expired(
        state.consume_lease_expires_at,
        now=now,
    ):
        raise ApprovalConflict("Approval consuming lease is still active.")
    if is_expired(state.expires_at, now=now):
        status: ApprovalStatus = "expired"
    elif outcome == "applied":
        status = "consumed"
    elif outcome == "not_started":
        status = "approved_once"
    else:
        status = "execution_failed"
    return replace(
        state,
        status=status,
        consume_lease_owner=None,
        consume_lease_token=None,
        consume_lease_expires_at=None,
        consumed_at=(
            utc_now_iso(now) if status in _TERMINAL_EXECUTION_STATUSES else None
        ),
    )


def supersede_approval(
    state: ApprovalLifecycleState,
    *,
    now: datetime | None = None,
) -> ApprovalLifecycleState:
    if state.status != "pending":
        raise ApprovalConflict("Only pending approvals can be superseded.")
    if is_expired(state.expires_at, now=now):
        raise ApprovalExpired(f"Approval {state.approval_id} has expired.")
    return replace(
        state,
        status="superseded",
        resolved_at=utc_now_iso(now),
    )


def can_record_execution_result(state: ApprovalLifecycleState) -> bool:
    # Persist the concrete result while the consume lease is still held.  This
    # closes the crash window between a mutation and its ReAct continuation.
    return state.status == "consuming" or state.status in _TERMINAL_EXECUTION_STATUSES


def recovery_outcome_for_task_status(task_status: str) -> ConsumeRecoveryOutcome:
    return "applied" if task_status in {"succeeded", "completed"} else "unknown"
