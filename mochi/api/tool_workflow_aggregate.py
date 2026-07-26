"""Pure v1 tool-workflow aggregate reduction and strict payload reader.

Phase 0 deliberately has no repository, outbox, SSE, feature flag, or UI
concerns.  Callers provide a durable-source snapshot and the delivery sequence
that a later outbox writer allocated.  This module never writes a source and
never treats raw turn transcripts as execution or verification evidence.

`canonical_json_subset_v1` is intentionally *not* a full RFC 8785
implementation.  It accepts JSON null/bool, JavaScript-safe integers, Unicode
strings without surrogate code points, arrays, and string-keyed objects.  It
rejects floats and non-JSON values, so callers fail closed until a verified full
RFC 8785 implementation is introduced.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from mochi.agents.artifact_verifier import ArtifactReceipt, tool_arguments_digest
from mochi.agents.conversation_state_store import TurnCheckpoint
from mochi.sessions.turn_timeline import (
    SessionTurnTimeline,
    TimelinePayloadError,
    TimelineUnsupportedVersionError,
)

AGGREGATE_TYPE = "tool_workflow_aggregate"
AGGREGATE_SCHEMA_VERSION = 1

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_BARE_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EVENT_ID_RE = re.compile(r"^twa:v1:[A-Za-z0-9_-]{43}$")
_TURN_STATUSES = frozenset(
    {
        "queued",
        "running",
        "awaiting_approval",
        "executing",
        "verifying",
        "completed",
        "blocked",
        "cancelled",
        "unknown",
    }
)
_INTEGRITIES = frozenset({"complete", "partial", "unsupported"})
_ACTIVATION_STATUSES = frozenset(
    {"not_observed", "requested", "activated", "rejected", "failed", "unknown"}
)
_REVIEW_STATUSES = frozenset(
    {"not_observed", "pending", "approved", "rejected", "expired", "consuming", "consumed", "unknown"}
)
_EXECUTION_STATUSES = frozenset(
    {"not_started", "precommitted", "started", "succeeded", "failed", "abandoned", "cancelled", "unknown"}
)
_VERIFICATION_STATUSES = frozenset(
    {"not_required", "pending", "verified", "failed", "unknown", "not_observed"}
)
_APPROVAL_STATUS_MAP = {
    "pending": "pending",
    "approved_once": "approved",
    "rejected": "rejected",
    "expired": "expired",
    "superseded": "unknown",
    "consuming": "consuming",
    "consumed": "consumed",
    "execution_failed": "consumed",
}
_UNRESOLVED_REVIEWS = frozenset({"pending", "approved", "consuming", "unknown"})
_MAX_SAFE_INTEGER = 9_007_199_254_740_991


class AggregatePayloadError(ValueError):
    """The aggregate payload or supported canonical JSON subset is invalid."""


class UnsupportedAggregateSourceError(AggregatePayloadError):
    """A durable source has a future or unsupported schema/version."""


def canonical_json_subset_v1(value: Any) -> str:
    """Return deterministic JSON for the documented non-float RFC 8785 subset."""

    return _canonical_json(value)


def canonical_sha256_subset_v1(value: Any) -> str:
    """Return a namespaced SHA-256 digest of `canonical_json_subset_v1(value)`."""

    return "sha256:" + hashlib.sha256(
        canonical_json_subset_v1(value).encode("utf-8")
    ).hexdigest()


def build_tool_workflow_idempotency_key_v1(
    *, source_refs: Mapping[str, Any], state: Mapping[str, Any]
) -> str:
    """Build the v1 key from source state only, excluding delivery fields."""

    return canonical_sha256_subset_v1(
        {
            "schema_version": AGGREGATE_SCHEMA_VERSION,
            "source_refs": dict(source_refs),
            "state": dict(state),
        }
    )


def build_tool_workflow_event_id_v1(*, session_id: str, turn_id: str, seq: int) -> str:
    """Build the opaque event ID without embedding or splitting logical IDs."""

    payload = canonical_json_subset_v1(
        {
            "schema_version": AGGREGATE_SCHEMA_VERSION,
            "session_id": _text(session_id, "session_id"),
            "turn_id": _text(turn_id, "turn_id"),
            "seq": _positive_int(seq, "seq"),
        }
    ).encode("utf-8")
    token = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode("ascii")
    return "twa:v1:" + token.rstrip("=")


def reduce_tool_workflow_aggregate_v1(
    *,
    session_id: str,
    turn_id: str,
    seq: int,
    occurred_at: str,
    timeline: Mapping[str, Any],
    timeline_source_position: int,
    checkpoint: Mapping[str, Any] | None,
    approvals: Sequence[Mapping[str, Any]] = (),
    receipts: Sequence[Mapping[str, Any]] = (),
    allow_legacy_approval_rows: bool = False,
) -> dict[str, Any]:
    """Reduce one validated durable-source snapshot without mutating any source."""

    session_id = _text(session_id, "session_id")
    turn_id = _text(turn_id, "turn_id")
    seq = _positive_int(seq, "seq")
    occurred_at = _timestamp(occurred_at, "occurred_at")
    timeline_position = _positive_int(timeline_source_position, "timeline_source_position")
    raw_timeline = _mapping(timeline, "timeline")

    try:
        parsed_timeline = SessionTurnTimeline.from_dict(raw_timeline)
    except TimelineUnsupportedVersionError as exc:
        raise UnsupportedAggregateSourceError(str(exc)) from exc
    except TimelinePayloadError as exc:
        raise AggregatePayloadError(f"invalid timeline source: {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise AggregatePayloadError(f"invalid timeline source: {exc}") from exc

    turn = next((item for item in parsed_timeline.turns if item.turn_id == turn_id), None)
    partial = parsed_timeline.session_id != session_id or turn is None
    blockers: list[str] = []
    if partial:
        blockers.append("timeline_identity_mismatch")

    timeline_ref = {
        # The parsed object normalizes legacy payloads to the current model;
        # source_refs must retain the actual durable source version.
        "timeline_version": _text(raw_timeline.get("timeline_version"), "timeline_version"),
        "turn_sequence": turn.sequence if turn is not None else None,
        "events": [
            {
                "source_position": timeline_position,
                "kind": "session_turn_timeline",
                "digest": canonical_sha256_subset_v1(raw_timeline),
            }
        ],
    }

    parsed_checkpoint: TurnCheckpoint | None = None
    checkpoint_ref: dict[str, Any] = {"checkpoint_revision": None, "digest": None}
    if checkpoint is None:
        partial = True
        blockers.append("checkpoint_not_observed")
    else:
        raw_checkpoint = _mapping(checkpoint, "checkpoint")
        try:
            parsed_checkpoint = TurnCheckpoint.from_dict(raw_checkpoint)
        except ValueError as exc:
            if "unsupported checkpoint version" in str(exc).lower():
                raise UnsupportedAggregateSourceError(str(exc)) from exc
            raise AggregatePayloadError(f"invalid checkpoint source: {exc}") from exc
        checkpoint_ref = {
            "checkpoint_revision": parsed_checkpoint.revision,
            "digest": canonical_sha256_subset_v1(raw_checkpoint),
        }
        if parsed_checkpoint.session_id != session_id or parsed_checkpoint.turn_id != turn_id:
            parsed_checkpoint = None
            partial = True
            blockers.append("checkpoint_identity_mismatch")

    calls: dict[str, dict[str, Any]] = {}
    if turn is not None:
        if turn.legacy_operation_ids:
            partial = True
            blockers.append("legacy_operation_identity_unjoined")
        for descriptor in turn.operation_descriptors:
            calls[descriptor.call_id] = _new_call(
                call_id=descriptor.call_id,
                operation_id=descriptor.operation_id,
                tool_name=descriptor.tool_name,
                arguments_digest=descriptor.arguments_digest,
                execution_status=descriptor.status,
                receipt_reference=descriptor.receipt_reference,
            )

    required_receipt = _artifact_receipt_required(parsed_checkpoint)
    _apply_checkpoint_call_evidence(calls, parsed_checkpoint, blockers)
    partial = partial or bool(blockers)

    approval_refs, approval_partial = _apply_approvals(
        approvals=approvals,
        calls=calls,
        session_id=session_id,
        turn_id=turn_id,
        blockers=blockers,
        allow_legacy_approval_rows=allow_legacy_approval_rows,
    )
    partial = partial or approval_partial
    receipt_refs, receipt_partial = _apply_receipts(
        receipts=receipts,
        calls=calls,
        session_id=session_id,
        turn_id=turn_id,
        blockers=blockers,
    )
    partial = partial or receipt_partial

    missing_required_receipt = False
    for call in calls.values():
        if call["_receipt_conflict"]:
            call["verification_status"] = "unknown"
        elif call["_receipt_verification"] is not None:
            call["verification_status"] = call.pop("_receipt_verification")
        elif required_receipt:
            call["verification_status"] = "not_observed"
            if call["execution_status"] == "succeeded":
                missing_required_receipt = True
        else:
            call["verification_status"] = "not_required"
        call.pop("_receipt_verification", None)
        if call["execution_status"] == "unknown":
            call["blocker"] = call["blocker"] or "operation_outcome_unknown"

    if missing_required_receipt:
        partial = True
        blockers.append("required_receipt_not_observed")
    _apply_terminal_cancellation(turn, calls)
    turn_status = _turn_status(
        turn=turn,
        checkpoint=parsed_checkpoint,
        calls=calls,
        partial=partial,
        missing_required_receipt=missing_required_receipt,
    )
    if turn_status == "blocked" and not blockers:
        blockers.append("terminal_execution_or_verification_blocked")

    normalized_calls = [_public_call(calls[call_id]) for call_id in sorted(calls)]
    state = {
        "turn_status": turn_status,
        "integrity": "partial" if partial else "complete",
        "policy": dict(parsed_checkpoint.policy_snapshot) if parsed_checkpoint else None,
        "inventory": dict(parsed_checkpoint.inventory_snapshot) if parsed_checkpoint else None,
        "calls": normalized_calls,
        "blocker": blockers[0] if blockers else None,
    }
    source_refs = {
        "timeline": timeline_ref,
        "checkpoint": checkpoint_ref,
        "approvals": approval_refs,
        "receipts": receipt_refs,
    }
    return _aggregate(
        session_id=session_id,
        turn_id=turn_id,
        seq=seq,
        occurred_at=occurred_at,
        source_refs=source_refs,
        state=state,
    )


def adapt_legacy_turn_events_v1(
    *,
    session_id: str,
    turn_id: str,
    seq: int,
    occurred_at: str,
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Read known raw `turn_event` v1 only as display enrichment.

    The adapter preserves call labels but cannot create an operation identity,
    approval state, result evidence, or verification result from a transcript.
    """

    session_id = _text(session_id, "session_id")
    turn_id = _text(turn_id, "turn_id")
    seq = _positive_int(seq, "seq")
    occurred_at = _timestamp(occurred_at, "occurred_at")
    if not _sequence(events):
        raise AggregatePayloadError("events must be a JSON array")

    calls: dict[str, dict[str, Any]] = {}
    previous_seq = 0
    for raw_event in events:
        event = _mapping(raw_event, "legacy turn_event")
        _require_exact_keys(
            event,
            {
                "type",
                "schema_version",
                "turn_id",
                "event_id",
                "seq",
                "phase",
                "timestamp",
                "payload",
            },
            "legacy turn_event",
        )
        if event["type"] != "turn_event" or event["schema_version"] != 1:
            raise UnsupportedAggregateSourceError("unsupported legacy turn_event version")
        if _text(event["turn_id"], "turn_event.turn_id") != turn_id:
            raise AggregatePayloadError("legacy turn_event turn_id does not match")
        event_seq = _positive_int(event["seq"], "turn_event.seq")
        if event_seq <= previous_seq:
            raise AggregatePayloadError("legacy turn_event seq must increase strictly")
        previous_seq = event_seq
        _text(event["event_id"], "turn_event.event_id")
        _timestamp(event["timestamp"], "turn_event.timestamp")
        phase = _text(event["phase"], "turn_event.phase")
        payload = _mapping(event["payload"], "turn_event.payload")
        if phase not in {"tool_call_request", "tool_call_created", "tool_call_result", "tool_call_completed"}:
            continue
        call_id = _text(payload.get("call_id"), "turn_event.payload.call_id")
        tool_name = _text(payload.get("tool_name"), "turn_event.payload.tool_name")
        prior = calls.get(call_id)
        if prior is not None and prior["tool_name"] != tool_name:
            raise AggregatePayloadError("legacy turn_event call_id has conflicting tool_name")
        calls[call_id] = _new_call(
            call_id=call_id,
            operation_id=None,
            tool_name=tool_name,
            arguments_digest=None,
            execution_status="not_started",
            receipt_reference=None,
        )

    state = {
        "turn_status": "unknown",
        "integrity": "partial",
        "policy": None,
        "inventory": None,
        "calls": [_public_call(calls[call_id]) for call_id in sorted(calls)],
        "blocker": "legacy_transcript_not_authoritative",
    }
    source_refs = {
        "timeline": {"timeline_version": None, "turn_sequence": None, "events": []},
        "checkpoint": {"checkpoint_revision": None, "digest": None},
        "approvals": [],
        "receipts": [],
    }
    return _aggregate(
        session_id=session_id,
        turn_id=turn_id,
        seq=seq,
        occurred_at=occurred_at,
        source_refs=source_refs,
        state=state,
    )


def parse_tool_workflow_aggregate_v1(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Strictly validate and return a v1 aggregate without partial application."""

    value = _mapping(payload, "aggregate")
    _require_exact_keys(
        value,
        {
            "type",
            "schema_version",
            "event_id",
            "seq",
            "idempotency_key",
            "session_id",
            "turn_id",
            "occurred_at",
            "source_refs",
            "state",
        },
        "aggregate",
    )
    if value["type"] != AGGREGATE_TYPE or value["schema_version"] != AGGREGATE_SCHEMA_VERSION:
        raise AggregatePayloadError("unsupported tool_workflow_aggregate schema version")
    session_id = _text(value["session_id"], "session_id")
    turn_id = _text(value["turn_id"], "turn_id")
    seq = _positive_int(value["seq"], "seq")
    _timestamp(value["occurred_at"], "occurred_at")
    expected_event_id = build_tool_workflow_event_id_v1(
        session_id=session_id, turn_id=turn_id, seq=seq
    )
    if not isinstance(value["event_id"], str) or not _EVENT_ID_RE.fullmatch(value["event_id"]):
        raise AggregatePayloadError("invalid aggregate event_id")
    if value["event_id"] != expected_event_id:
        raise AggregatePayloadError("aggregate event_id does not match its identity")
    source_refs = _parse_source_refs(value["source_refs"])
    state = _parse_state(value["state"])
    _validate_state_consistency(state)
    expected_key = build_tool_workflow_idempotency_key_v1(
        source_refs=source_refs, state=state
    )
    if value["idempotency_key"] != expected_key or not _SHA256_RE.fullmatch(str(value["idempotency_key"])):
        raise AggregatePayloadError("aggregate idempotency_key does not match source state")
    return {
        "type": AGGREGATE_TYPE,
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "event_id": value["event_id"],
        "seq": seq,
        "idempotency_key": expected_key,
        "session_id": session_id,
        "turn_id": turn_id,
        "occurred_at": value["occurred_at"],
        "source_refs": source_refs,
        "state": state,
    }


def _aggregate(
    *,
    session_id: str,
    turn_id: str,
    seq: int,
    occurred_at: str,
    source_refs: Mapping[str, Any],
    state: Mapping[str, Any],
) -> dict[str, Any]:
    normalized_refs = _parse_source_refs(source_refs)
    normalized_state = _parse_state(state)
    return {
        "type": AGGREGATE_TYPE,
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "event_id": build_tool_workflow_event_id_v1(
            session_id=session_id, turn_id=turn_id, seq=seq
        ),
        "seq": seq,
        "idempotency_key": build_tool_workflow_idempotency_key_v1(
            source_refs=normalized_refs, state=normalized_state
        ),
        "session_id": session_id,
        "turn_id": turn_id,
        "occurred_at": occurred_at,
        "source_refs": normalized_refs,
        "state": normalized_state,
    }


def _new_call(
    *,
    call_id: str,
    operation_id: str | None,
    tool_name: str,
    arguments_digest: str | None,
    execution_status: str,
    receipt_reference: str | None,
) -> dict[str, Any]:
    return {
        "call_id": call_id,
        "operation_id": operation_id,
        "tool_name": tool_name,
        "arguments_digest": arguments_digest,
        "target": None,
        "activation_status": "not_observed",
        "review_status": "not_observed",
        "approval_id": None,
        "execution_status": execution_status,
        "verification_status": "not_observed",
        # A timeline descriptor reference identifies its tool-result evidence.
        # The public field is reserved for a separately joined artifact receipt.
        "receipt_reference": None,
        "changed_paths": [],
        "blocker": None,
        "_timeline_result_reference": receipt_reference,
        "_receipt_conflict": False,
        "_receipt_verification": None,
    }


def _apply_checkpoint_call_evidence(
    calls: dict[str, dict[str, Any]], checkpoint: TurnCheckpoint | None, blockers: list[str]
) -> None:
    if checkpoint is None:
        return
    pending = checkpoint.pending_tool_call
    if not isinstance(pending, Mapping):
        return
    call_id = pending.get("call_id")
    tool_name = pending.get("tool_name")
    arguments = pending.get("arguments")
    if not isinstance(call_id, str) or not call_id or not isinstance(tool_name, str) or not tool_name:
        blockers.append("checkpoint_call_identity_invalid")
        return
    if not isinstance(arguments, Mapping):
        blockers.append("checkpoint_call_arguments_missing")
        return
    digest = tool_arguments_digest(tool_name=tool_name, arguments=arguments)
    call = calls.get(call_id)
    if call is None:
        calls[call_id] = _new_call(
            call_id=call_id,
            operation_id=None,
            tool_name=tool_name,
            arguments_digest=digest,
            execution_status="not_started",
            receipt_reference=None,
        )
        return
    if call["tool_name"] != tool_name or call["arguments_digest"] != digest:
        call["blocker"] = "source_join_mismatch"
        blockers.append("checkpoint_operation_join_mismatch")


def _apply_approvals(
    *,
    approvals: Sequence[Mapping[str, Any]],
    calls: dict[str, dict[str, Any]],
    session_id: str,
    turn_id: str,
    blockers: list[str],
    allow_legacy_approval_rows: bool,
) -> tuple[list[dict[str, Any]], bool]:
    if not _sequence(approvals):
        raise AggregatePayloadError("approvals must be a JSON array")
    refs: list[dict[str, Any]] = []
    partial = False
    seen: dict[str, str] = {}
    for raw in approvals:
        row = _mapping(raw, "approval")
        metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
        get = lambda key: row.get(key, metadata.get(key))
        version = row.get("schema_version", 1)
        if version != 1:
            raise UnsupportedAggregateSourceError("unsupported approval source version")
        approval_id = _text(get("approval_id"), "approval_id")
        status = _text(get("status"), "approval.status")
        review_status = _APPROVAL_STATUS_MAP.get(status)
        if review_status is None:
            raise UnsupportedAggregateSourceError(
                f"unsupported approval source status: {status!r}"
            )
        row_session = _text(get("session_id"), "approval.session_id")
        row_turn = _text(get("turn_id"), "approval.turn_id")
        call_id = _text(get("call_id"), "approval.call_id")
        revision = get("approval_revision")
        request_digest = _digest_or_empty(get("request_digest"), "approval.request_digest")
        context_digest = _digest_or_empty(get("context_digest"), "approval.context_digest")
        if revision is None:
            if not allow_legacy_approval_rows:
                raise AggregatePayloadError(
                    "current approval source requires a positive approval_revision"
                )
            supplied_legacy_digest = get("legacy_digest")
            ref: dict[str, Any] = {
                "approval_id": approval_id,
                "approval_revision": None,
                "status": status,
                "request_digest": request_digest,
                "context_digest": context_digest,
                # A durable adapter may provide the exact canonical digest for
                # a pre-revision source.  Raw legacy inputs retain the
                # canonical whole-row fallback for backwards compatibility.
                "legacy_digest": (
                    _sha256(supplied_legacy_digest, "approval.legacy_digest")
                    if supplied_legacy_digest is not None
                    else canonical_sha256_subset_v1(row)
                ),
            }
        else:
            ref = {
                "approval_id": approval_id,
                "approval_revision": _positive_int(revision, "approval_revision"),
                "status": status,
                "request_digest": request_digest,
                "context_digest": context_digest,
                "legacy_digest": None,
            }
        fingerprint = canonical_json_subset_v1(ref)
        prior = seen.get(approval_id)
        if prior is not None:
            if prior == fingerprint:
                continue
            raise UnsupportedAggregateSourceError(
                f"conflicting duplicate approval source: {approval_id!r}"
            )
        seen[approval_id] = fingerprint
        refs.append(ref)
        if row_session != session_id or row_turn != turn_id:
            partial = True
            blockers.append("approval_identity_mismatch")
            continue
        call = calls.get(call_id)
        if call is None:
            partial = True
            blockers.append("approval_call_mismatch")
            continue
        operation_id = get("operation_id")
        arguments_digest = get("arguments_digest")
        if not isinstance(operation_id, str) or not operation_id:
            partial = True
            call["blocker"] = "source_join_mismatch"
            blockers.append("approval_operation_identity_missing")
            continue
        if not isinstance(arguments_digest, str) or not _BARE_SHA256_RE.fullmatch(arguments_digest):
            partial = True
            call["blocker"] = "source_join_mismatch"
            blockers.append("approval_arguments_digest_missing")
            continue
        if operation_id != call["operation_id"]:
            partial = True
            call["blocker"] = "source_join_mismatch"
            blockers.append("approval_operation_join_mismatch")
            continue
        if arguments_digest != call["arguments_digest"]:
            partial = True
            call["blocker"] = "source_join_mismatch"
            blockers.append("approval_arguments_join_mismatch")
            continue
        if call["approval_id"] not in {None, approval_id}:
            partial = True
            call["blocker"] = "source_join_mismatch"
            blockers.append("approval_call_conflict")
            continue
        call["approval_id"] = approval_id
        call["review_status"] = review_status
    return sorted(refs, key=lambda ref: (ref["approval_id"], ref["approval_revision"] or 0)), partial


def _apply_receipts(
    *,
    receipts: Sequence[Mapping[str, Any]],
    calls: dict[str, dict[str, Any]],
    session_id: str,
    turn_id: str,
    blockers: list[str],
) -> tuple[list[dict[str, Any]], bool]:
    if not _sequence(receipts):
        raise AggregatePayloadError("receipts must be a JSON array")
    refs: list[dict[str, Any]] = []
    partial = False
    seen: dict[str, tuple[str, dict[str, Any]]] = {}
    for raw in receipts:
        envelope = _mapping(raw, "receipt envelope")
        _require_exact_keys(
            envelope,
            {"source_position", "session_id", "receipt_reference", "receipt"},
            "receipt envelope",
        )
        position = _positive_int(envelope["source_position"], "receipt.source_position")
        receipt_reference = _text(envelope["receipt_reference"], "receipt_reference")
        receipt_session_id = _text(envelope["session_id"], "receipt.session_id")
        raw_receipt = _mapping(envelope["receipt"], "receipt")
        try:
            receipt = ArtifactReceipt.from_dict(raw_receipt)
        except ValueError as exc:
            if "unsupported artifact receipt schema" in str(exc).lower():
                raise UnsupportedAggregateSourceError(str(exc)) from exc
            raise AggregatePayloadError(f"invalid receipt source: {exc}") from exc
        digest = canonical_sha256_subset_v1(raw_receipt)
        raw_version = raw_receipt.get("schema_version", 1)
        if type(raw_version) is not int or raw_version not in {1, 2, 3}:
            raise UnsupportedAggregateSourceError("unsupported artifact receipt source version")
        ref = {
            "kind": "artifact_receipt",
            "schema_version": raw_version,
            "source_position": position,
            "operation_id": receipt.operation_id,
            "receipt_reference": receipt_reference,
            "digest": digest,
            "verification_status": receipt.verification_status,
        }
        fingerprint = canonical_json_subset_v1(
            {key: value for key, value in ref.items() if key != "source_position"}
        )
        prior = seen.get(receipt.operation_id)
        if prior is not None:
            if prior[0] != fingerprint:
                raise UnsupportedAggregateSourceError(
                    f"conflicting duplicate receipt source: {receipt.operation_id!r}"
                )
            if ref["source_position"] < prior[1]["source_position"]:
                refs.remove(prior[1])
                refs.append(ref)
                seen[receipt.operation_id] = (fingerprint, ref)
            continue
        seen[receipt.operation_id] = (fingerprint, ref)
        refs.append(ref)
        if receipt_session_id != session_id:
            partial = True
            blockers.append("receipt_session_mismatch")
            continue
        candidates = [
            call
            for call in calls.values()
            if call["operation_id"] == receipt.operation_id
        ]
        if receipt.turn_id != turn_id or len(candidates) != 1:
            partial = True
            blockers.append("receipt_identity_or_operation_mismatch")
            continue
        call = candidates[0]
        if call["call_id"] not in receipt.tool_call_ids:
            partial = True
            call["blocker"] = "source_join_mismatch"
            blockers.append("receipt_call_join_mismatch")
            continue
        if call["receipt_reference"] not in {None, receipt_reference}:
            partial = True
            call["blocker"] = "source_join_mismatch"
            blockers.append("receipt_reference_join_mismatch")
            continue
        if receipt.execution_status != call["execution_status"]:
            partial = True
            call["blocker"] = "source_join_mismatch"
            call["_receipt_conflict"] = True
            blockers.append("receipt_execution_conflict")
            continue
        call["receipt_reference"] = receipt_reference
        call["changed_paths"] = sorted(receipt.changed_paths)
        call["_receipt_verification"] = {
            "verified": "verified",
            "failed": "failed",
            "partial": "unknown",
            "not_run": "pending",
        }[receipt.verification_status]
    return sorted(refs, key=lambda ref: (ref["operation_id"], ref["receipt_reference"])), partial


def _apply_terminal_cancellation(turn: Any, calls: Mapping[str, dict[str, Any]]) -> None:
    if turn is None or turn.terminal_outcome != "cancelled":
        return
    for call in calls.values():
        if call["execution_status"] in {"unknown", "abandoned"}:
            continue
        if call["execution_status"] in {"not_started", "precommitted", "started"}:
            call["execution_status"] = "cancelled"


def _turn_status(
    *,
    turn: Any,
    checkpoint: TurnCheckpoint | None,
    calls: Mapping[str, Mapping[str, Any]],
    partial: bool,
    missing_required_receipt: bool,
) -> str:
    statuses = {str(call["execution_status"]) for call in calls.values()}
    reviews = {str(call["review_status"]) for call in calls.values()}
    if "unknown" in statuses or (turn is not None and turn.terminal_outcome == "unknown"):
        return "unknown"
    if turn is None:
        return "unknown"
    if turn.status != "terminal":
        if checkpoint is not None:
            return {
                "awaiting_approval": "awaiting_approval",
                "executing": "executing",
                "verifying": "verifying",
            }.get(checkpoint.stage, "running")
        return "queued" if turn.status == "queued" else "running"
    if turn.terminal_outcome == "cancelled":
        return "cancelled"
    if turn.terminal_outcome == "blocked":
        return "blocked"
    if turn.terminal_outcome != "completed":
        return "unknown"
    if (
        partial
        or missing_required_receipt
        or bool(statuses - {"succeeded"})
        or bool(reviews.intersection(_UNRESOLVED_REVIEWS))
        or any(call["verification_status"] in {"failed", "pending", "unknown", "not_observed"} for call in calls.values())
    ):
        return "blocked"
    return "completed"


def _artifact_receipt_required(checkpoint: TurnCheckpoint | None) -> bool:
    if checkpoint is None:
        return False
    value = checkpoint.capability_plan.get("artifact_obligation")
    return bool(isinstance(value, Mapping) and value.get("required") and value.get("ready"))


def _public_call(call: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in call.items() if not key.startswith("_")}


def _parse_source_refs(value: Any) -> dict[str, Any]:
    refs = _mapping(value, "source_refs")
    _require_exact_keys(refs, {"timeline", "checkpoint", "approvals", "receipts"}, "source_refs")
    timeline = _mapping(refs["timeline"], "source_refs.timeline")
    _require_exact_keys(timeline, {"timeline_version", "turn_sequence", "events"}, "source_refs.timeline")
    version = timeline["timeline_version"]
    if version is not None and not isinstance(version, str):
        raise AggregatePayloadError("timeline_version must be string or null")
    turn_sequence = timeline["turn_sequence"]
    if turn_sequence is not None:
        _positive_int(turn_sequence, "turn_sequence")
    timeline_events = _records(timeline["events"], "source_refs.timeline.events")
    positions: set[int] = set()
    normalized_events: list[dict[str, Any]] = []
    for event in timeline_events:
        _require_exact_keys(event, {"source_position", "kind", "digest"}, "timeline source ref")
        position = _positive_int(event["source_position"], "source_position")
        if position in positions:
            raise AggregatePayloadError("duplicate timeline source position")
        positions.add(position)
        normalized_events.append(
            {
                "source_position": position,
                "kind": _text(event["kind"], "timeline source kind"),
                "digest": _sha256(event["digest"], "timeline source digest"),
            }
        )
    checkpoint = _mapping(refs["checkpoint"], "source_refs.checkpoint")
    _require_exact_keys(checkpoint, {"checkpoint_revision", "digest"}, "source_refs.checkpoint")
    revision = checkpoint["checkpoint_revision"]
    digest = checkpoint["digest"]
    if (revision is None) != (digest is None):
        raise AggregatePayloadError("checkpoint source reference must be fully present or null")
    normalized_checkpoint = {
        "checkpoint_revision": None if revision is None else _non_negative_int(revision, "checkpoint_revision"),
        "digest": None if digest is None else _sha256(digest, "checkpoint digest"),
    }
    approvals = [_parse_approval_ref(item) for item in _records(refs["approvals"], "source_refs.approvals")]
    receipts = [_parse_receipt_ref(item) for item in _records(refs["receipts"], "source_refs.receipts")]
    if approvals != sorted(approvals, key=lambda item: (item["approval_id"], item["approval_revision"] or 0)):
        raise AggregatePayloadError("approval source refs must be sorted")
    if receipts != sorted(receipts, key=lambda item: (item["operation_id"], item["receipt_reference"])):
        raise AggregatePayloadError("receipt source refs must be sorted")
    if normalized_events != sorted(normalized_events, key=lambda item: item["source_position"]):
        raise AggregatePayloadError("timeline source refs must be sorted")
    return {
        "timeline": {
            "timeline_version": version,
            "turn_sequence": turn_sequence,
            "events": normalized_events,
        },
        "checkpoint": normalized_checkpoint,
        "approvals": approvals,
        "receipts": receipts,
    }


def _parse_approval_ref(value: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_keys(
        value,
        {"approval_id", "approval_revision", "status", "request_digest", "context_digest", "legacy_digest"},
        "approval source ref",
    )
    revision = value["approval_revision"]
    legacy = value["legacy_digest"]
    if revision is None and legacy is None:
        raise AggregatePayloadError("legacy approval source ref requires a digest")
    if revision is not None and legacy is not None:
        raise AggregatePayloadError("current approval source ref cannot include legacy digest")
    status = _text(value["status"], "approval status")
    if status not in _APPROVAL_STATUS_MAP:
        raise AggregatePayloadError("unsupported approval source status")
    return {
        "approval_id": _text(value["approval_id"], "approval_id"),
        "approval_revision": None if revision is None else _positive_int(revision, "approval_revision"),
        "status": status,
        "request_digest": _digest_or_empty(value["request_digest"], "request_digest"),
        "context_digest": _digest_or_empty(value["context_digest"], "context_digest"),
        "legacy_digest": None if legacy is None else _sha256(legacy, "legacy approval digest"),
    }


def _parse_receipt_ref(value: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_keys(
        value,
        {"kind", "schema_version", "source_position", "operation_id", "receipt_reference", "digest", "verification_status"},
        "receipt source ref",
    )
    if value["kind"] != "artifact_receipt":
        raise AggregatePayloadError("unsupported receipt source kind")
    if type(value["schema_version"]) is not int or value["schema_version"] not in {1, 2, 3}:
        raise AggregatePayloadError("unsupported receipt source version")
    verification = _text(value["verification_status"], "receipt verification_status")
    if verification not in {"verified", "failed", "partial", "not_run"}:
        raise AggregatePayloadError("unsupported receipt verification_status")
    return {
        "kind": "artifact_receipt",
        "schema_version": value["schema_version"],
        "source_position": _positive_int(value["source_position"], "receipt source_position"),
        "operation_id": _text(value["operation_id"], "receipt operation_id"),
        "receipt_reference": _text(value["receipt_reference"], "receipt_reference"),
        "digest": _sha256(value["digest"], "receipt digest"),
        "verification_status": verification,
    }


def _parse_state(value: Any) -> dict[str, Any]:
    state = _mapping(value, "state")
    _require_exact_keys(state, {"turn_status", "integrity", "policy", "inventory", "calls", "blocker"}, "state")
    turn_status = _text(state["turn_status"], "turn_status")
    integrity = _text(state["integrity"], "integrity")
    if turn_status not in _TURN_STATUSES or integrity not in _INTEGRITIES:
        raise AggregatePayloadError("unsupported aggregate state vocabulary")
    policy = state["policy"]
    inventory = state["inventory"]
    if policy is not None and not isinstance(policy, Mapping):
        raise AggregatePayloadError("state.policy must be object or null")
    if inventory is not None and not isinstance(inventory, Mapping):
        raise AggregatePayloadError("state.inventory must be object or null")
    calls = [_parse_call(item) for item in _records(state["calls"], "state.calls")]
    if calls != sorted(calls, key=lambda item: item["call_id"]):
        raise AggregatePayloadError("state calls must be sorted by call_id")
    if len({item["call_id"] for item in calls}) != len(calls):
        raise AggregatePayloadError("state calls must have unique call_id")
    blocker = state["blocker"]
    if blocker is not None:
        blocker = _text(blocker, "state.blocker")
    return {
        "turn_status": turn_status,
        "integrity": integrity,
        "policy": None if policy is None else dict(policy),
        "inventory": None if inventory is None else dict(inventory),
        "calls": calls,
        "blocker": blocker,
    }


def _validate_state_consistency(state: Mapping[str, Any]) -> None:
    calls = state["calls"]
    execution_statuses = {call["execution_status"] for call in calls}
    if "unknown" in execution_statuses and state["turn_status"] != "unknown":
        raise AggregatePayloadError("unknown operation must make the turn unknown")
    if state["turn_status"] == "cancelled" and "unknown" in execution_statuses:
        raise AggregatePayloadError("unknown operation cannot be cancelled away")
    if state["turn_status"] != "completed":
        return
    if state["integrity"] != "complete":
        raise AggregatePayloadError("completed turn requires complete integrity")
    if any(call["execution_status"] != "succeeded" for call in calls):
        raise AggregatePayloadError("completed turn requires succeeded execution evidence")
    if any(call["review_status"] in _UNRESOLVED_REVIEWS for call in calls):
        raise AggregatePayloadError("completed turn cannot retain unresolved approval")
    if any(
        call["verification_status"] not in {"verified", "not_required"}
        for call in calls
    ):
        raise AggregatePayloadError("completed turn requires terminal verification evidence")


def _parse_call(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "call_id", "operation_id", "tool_name", "arguments_digest", "target", "activation_status",
        "review_status", "approval_id", "execution_status", "verification_status", "receipt_reference",
        "changed_paths", "blocker",
    }
    _require_exact_keys(value, fields, "aggregate call")
    operation_id = value["operation_id"]
    arguments_digest = value["arguments_digest"]
    target = value["target"]
    approval_id = value["approval_id"]
    receipt_reference = value["receipt_reference"]
    blocker = value["blocker"]
    for name, status, allowed in (
        ("activation_status", value["activation_status"], _ACTIVATION_STATUSES),
        ("review_status", value["review_status"], _REVIEW_STATUSES),
        ("execution_status", value["execution_status"], _EXECUTION_STATUSES),
        ("verification_status", value["verification_status"], _VERIFICATION_STATUSES),
    ):
        if _text(status, name) not in allowed:
            raise AggregatePayloadError(f"unsupported {name}")
    if target is not None and not isinstance(target, Mapping):
        raise AggregatePayloadError("call target must be object or null")
    return {
        "call_id": _text(value["call_id"], "call_id"),
        "operation_id": None if operation_id is None else _text(operation_id, "operation_id"),
        "tool_name": _text(value["tool_name"], "tool_name"),
        "arguments_digest": None if arguments_digest is None else _bare_sha256(arguments_digest, "arguments_digest"),
        "target": None if target is None else dict(target),
        "activation_status": value["activation_status"],
        "review_status": value["review_status"],
        "approval_id": None if approval_id is None else _text(approval_id, "approval_id"),
        "execution_status": value["execution_status"],
        "verification_status": value["verification_status"],
        "receipt_reference": None if receipt_reference is None else _text(receipt_reference, "receipt_reference"),
        "changed_paths": sorted(_strings(value["changed_paths"], "changed_paths")),
        "blocker": None if blocker is None else _text(blocker, "call.blocker"),
    }


def _canonical_json(value: Any) -> str:
    if value is None:
        return "null"
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) is int:
        if not -_MAX_SAFE_INTEGER <= value <= _MAX_SAFE_INTEGER:
            raise AggregatePayloadError("canonical JSON subset integer exceeds JavaScript safe range")
        return str(value)
    if isinstance(value, str):
        if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
            raise AggregatePayloadError("canonical JSON subset rejects surrogate code points")
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, list):
        return "[" + ",".join(_canonical_json(item) for item in value) + "]"
    if isinstance(value, Mapping):
        normalized: list[tuple[str, Any]] = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise AggregatePayloadError("canonical JSON subset object keys must be strings")
            normalized.append((key, item))
        normalized.sort(key=lambda item: item[0].encode("utf-16-be", "surrogatepass"))
        return "{" + ",".join(
            _canonical_json(key) + ":" + _canonical_json(item) for key, item in normalized
        ) + "}"
    raise AggregatePayloadError(f"canonical JSON subset rejects {type(value).__name__}")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AggregatePayloadError(f"{name} must be an object")
    return dict(value)


def _sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _records(value: Any, name: str) -> list[dict[str, Any]]:
    if not _sequence(value):
        raise AggregatePayloadError(f"{name} must be a JSON array")
    return [_mapping(item, name) for item in value]


def _strings(value: Any, name: str) -> list[str]:
    if not _sequence(value) or any(not isinstance(item, str) or not item for item in value):
        raise AggregatePayloadError(f"{name} must be a JSON array of non-empty strings")
    return list(value)


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise AggregatePayloadError(
            f"{name} fields do not match v1 contract: missing={sorted(expected - actual)!r}; unexpected={sorted(actual - expected)!r}"
        )


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AggregatePayloadError(f"{name} must be a non-empty string")
    return value.strip()


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise AggregatePayloadError(f"{name} must be a positive integer")
    return value


def _non_negative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise AggregatePayloadError(f"{name} must be a non-negative integer")
    return value


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise AggregatePayloadError(f"{name} must be a sha256 digest")
    return value


def _bare_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _BARE_SHA256_RE.fullmatch(value):
        raise AggregatePayloadError(f"{name} must be a bare sha256 digest")
    return value


def _digest_or_empty(value: Any, name: str) -> str:
    if value in {None, ""}:
        return ""
    return _bare_sha256(value, name)


def _timestamp(value: Any, name: str) -> str:
    raw = _text(value, name)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AggregatePayloadError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise AggregatePayloadError(f"{name} must include a timezone")
    return raw


__all__ = [
    "AGGREGATE_SCHEMA_VERSION",
    "AGGREGATE_TYPE",
    "AggregatePayloadError",
    "UnsupportedAggregateSourceError",
    "adapt_legacy_turn_events_v1",
    "build_tool_workflow_event_id_v1",
    "build_tool_workflow_idempotency_key_v1",
    "canonical_json_subset_v1",
    "canonical_sha256_subset_v1",
    "parse_tool_workflow_aggregate_v1",
    "reduce_tool_workflow_aggregate_v1",
]
