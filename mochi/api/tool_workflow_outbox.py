"""Durable, rebuildable delivery cache for tool-workflow aggregates.

The session JSONL remains the authority for every record in this module.  An
outbox entry is only a cached reduction of timeline, checkpoint, receipt, and
approval-observation source records.  It is safe to delete and rebuild; it is
never read as lifecycle evidence.

Approval rows live in the approval database, so they cross the storage
boundary through a small, idempotent session observation.  The reconciler in
this module only writes that observation and its aggregate cache entry.  It
does not invoke an engine, a tool registry, or a continuation callback.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock
from typing import Any, TYPE_CHECKING

from mochi.agents.conversation_state_store import TURN_CHECKPOINT_EVENT, TurnCheckpoint
from mochi.api.tool_workflow_aggregate import (
    AggregatePayloadError,
    UnsupportedAggregateSourceError,
    canonical_json_subset_v1,
    canonical_sha256_subset_v1,
    parse_tool_workflow_aggregate_v1,
    reduce_tool_workflow_aggregate_v1,
)
from mochi.sessions.turn_timeline import (
    SESSION_TURN_TIMELINE_EVENT,
    SessionTurnTimeline,
    TimelinePayloadError,
    TimelineUnsupportedVersionError,
)

if TYPE_CHECKING:
    from mochi.runtime.approval_lifecycle import ApprovalStore, ExecApprovalRequest
    from mochi.sessions.store import (
        DurableSessionSnapshot,
        SessionStore,
        ToolWorkflowPublicationGate,
    )


TOOL_WORKFLOW_OUTBOX_EVENT = "tool_workflow_aggregate_outbox"
TOOL_WORKFLOW_OUTBOX_EVENT_VERSION = 1
TOOL_WORKFLOW_APPROVAL_OBSERVATION_EVENT = "tool_workflow_approval_observation"
TOOL_WORKFLOW_APPROVAL_OBSERVATION_VERSION = 2

_OUTBOX_EVENT_FIELDS = frozenset(
    {
        "type",
        "event",
        "schema_version",
        "session_id",
        "turn_id",
        "aggregate",
        "timestamp",
    }
)
_OBSERVATION_EVENT_FIELDS = frozenset(
    {
        "type",
        "event",
        "schema_version",
        "session_id",
        "turn_id",
        "approval_observation",
        "timestamp",
    }
)
_OBSERVATION_FIELDS_V1 = frozenset(
    {
        "approval_id",
        "approval_revision",
        "status",
        "request_digest",
        "context_digest",
        "call_id",
        "operation_id",
        "arguments_digest",
    }
)
_OBSERVATION_FIELDS_V2 = _OBSERVATION_FIELDS_V1 | frozenset({"legacy_digest"})
_RECEIPT_EVENT = "artifact_verification_receipt"


class ToolWorkflowOutboxError(ValueError):
    """A source/outbox invariant prevents aggregate delivery."""


class ToolWorkflowOutboxUnsupportedError(ToolWorkflowOutboxError):
    """A future or conflicting outbox source was encountered."""


@dataclass(frozen=True)
class ToolWorkflowOutboxReconcileResult:
    """One no-tool restart reconciliation result."""

    approval_id: str
    session_id: str | None
    turn_id: str | None
    status: str
    message: str | None = None


@dataclass(frozen=True)
class ToolWorkflowOutboxVerificationResult:
    """Read-only durable replay result for one session's outbox cache."""

    session_id: str
    checked: int = 0
    matched: int = 0
    source_mismatch: int = 0
    duplicate: int = 0
    gap: int = 0
    unsupported: int = 0
    messages: tuple[str, ...] = ()
    # (counter_name, content-qualified durable identity) tuples let a runtime
    # report each immutable finding once without treating a replaced/restored
    # record at the same physical position as the previous payload.
    findings: tuple[tuple[str, str], ...] = ()

    def counters(self) -> dict[str, int]:
        return {
            "source_mismatch": self.source_mismatch,
            "duplicate": self.duplicate,
            "gap": self.gap,
            "unsupported": self.unsupported,
        }


class ToolWorkflowOutboxVerifierDiagnostics:
    """Process-local, de-duplicated counters for durable verifier findings."""

    def __init__(self) -> None:
        self._counters = {
            "source_mismatch": 0,
            "duplicate": 0,
            "gap": 0,
            "unsupported": 0,
        }
        self._findings: set[tuple[str, str, str]] = set()
        self._lock = Lock()

    def record(self, verification: ToolWorkflowOutboxVerificationResult) -> None:
        with self._lock:
            for name, identity in verification.findings:
                key = (verification.session_id, name, identity)
                if key not in self._findings:
                    self._findings.add(key)
                    self._counters[name] += 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counters)


def build_outbox_companion_events_v1(
    *,
    session_id: str,
    before_events: Sequence[Mapping[str, Any]],
    source_events: Sequence[Mapping[str, Any]],
    force_turn_ids: Iterable[str] = (),
) -> tuple[dict[str, Any], ...]:
    """Return cache entries for a prospective strict SessionStore mutation.

    ``source_events`` must be the authoritative records being appended in the
    same CAS.  The function excludes prior outbox records from reduction input,
    so replaying or rebuilding an entry cannot create a sequence loop.
    """

    normalized_session_id = _text(session_id, "session_id")
    before = tuple(_object(event, "existing session event") for event in before_events)
    source = tuple(_object(event, "source event") for event in source_events)
    candidate = (*before, *source)
    turn_ids = {turn_id for turn_id in force_turn_ids if isinstance(turn_id, str) and turn_id}
    turn_ids.update(_affected_turn_ids(normalized_session_id, before, candidate, source))
    if not turn_ids:
        return ()

    # Include any explicitly supplied companion records.  This permits callers
    # that already formed a strict source+outbox batch to pass through the same
    # SessionStore hook without the hook appending a second cache record.
    existing = _outbox_by_key(normalized_session_id, candidate)
    events: list[dict[str, Any]] = []
    for turn_id in sorted(turn_ids):
        aggregate = _reduce_from_events(
            session_id=normalized_session_id,
            turn_id=turn_id,
            events=candidate,
            seq=_next_sequence(normalized_session_id, turn_id, candidate),
        )
        key = str(aggregate["idempotency_key"])
        prior = existing.get((turn_id, key))
        if prior is not None:
            # The existing cache record is already fully parsed by _outbox_by_key.
            continue
        events.append(
            {
                "type": "session_meta",
                "event": TOOL_WORKFLOW_OUTBOX_EVENT,
                "schema_version": TOOL_WORKFLOW_OUTBOX_EVENT_VERSION,
                "session_id": normalized_session_id,
                "turn_id": turn_id,
                "aggregate": aggregate,
                "timestamp": aggregate["occurred_at"],
            }
        )
    return tuple(events)


def verify_tool_workflow_outbox_v1(
    session_id: str,
    events: Sequence[Mapping[str, Any]],
    *,
    start_position: int = 1,
) -> ToolWorkflowOutboxVerificationResult:
    """Verify cache records against the source prefix visible at each append.

    Outbox entries are deliberately not trusted as input to this replay.  The
    verifier only reports mismatches; it never repairs a source, replays a
    tool, or mutates the approval store.
    """

    normalized_session_id = _text(session_id, "session_id")
    if type(start_position) is not int or start_position <= 0:
        raise ValueError("start_position must be a positive integer")
    # Keep the full durable prefix, including prior cache envelopes.  Reducers
    # ignore outbox entries as lifecycle evidence, but timeline source
    # positions are physical SessionStore positions and must not be compressed.
    prefix_events: list[Mapping[str, Any]] = []
    last_seq: dict[str, int] = {}
    seen_keys: dict[tuple[str, str], dict[str, Any]] = {}
    seen_sequences: dict[tuple[str, int], dict[str, Any]] = {}
    checked = matched = source_mismatch = duplicate = gap = unsupported = 0
    messages: list[str] = []
    findings: list[tuple[str, str]] = []

    for position, raw_event in enumerate(events, start=1):
        if raw_event.get("event") != TOOL_WORKFLOW_OUTBOX_EVENT:
            prefix_events.append(raw_event)
            continue
        is_target = position >= start_position
        if is_target:
            checked += 1
        try:
            aggregate = _parse_outbox_event(normalized_session_id, raw_event)
        except (ToolWorkflowOutboxError, AggregatePayloadError) as exc:
            if is_target:
                unsupported += 1
                messages.append(f"position={position}: unsupported outbox record ({type(exc).__name__})")
                findings.append(("unsupported", _outbox_finding_identity(position, raw_event)))
            prefix_events.append(raw_event)
            continue

        turn_id = str(aggregate["turn_id"])
        seq = int(aggregate["seq"])
        key = (turn_id, str(aggregate["idempotency_key"]))
        sequence_key = (turn_id, seq)
        duplicate_found = key in seen_keys or sequence_key in seen_sequences
        expected_seq = last_seq.get(turn_id, 0) + 1
        if is_target and seq > expected_seq:
            gap += 1
            messages.append(
                f"position={position}: sequence gap for turn {turn_id!r}; "
                f"expected {expected_seq}, got {seq}"
            )
            findings.append(("gap", _outbox_finding_identity(position, raw_event)))
        if duplicate_found and is_target:
            duplicate += 1
            messages.append(f"position={position}: duplicate outbox identity")
            findings.append(("duplicate", _outbox_finding_identity(position, raw_event)))
        else:
            last_seq[turn_id] = max(last_seq.get(turn_id, 0), seq)
        seen_keys.setdefault(key, aggregate)
        seen_sequences.setdefault(sequence_key, aggregate)

        if not is_target:
            prefix_events.append(raw_event)
            continue

        try:
            expected = _reduce_from_events(
                session_id=normalized_session_id,
                turn_id=turn_id,
                events=prefix_events,
                seq=seq,
            )
        except ToolWorkflowOutboxUnsupportedError as exc:
            unsupported += 1
            messages.append(f"position={position}: unsupported source ({type(exc).__name__})")
            findings.append(("unsupported", _outbox_finding_identity(position, raw_event)))
            prefix_events.append(raw_event)
            continue
        except ToolWorkflowOutboxError as exc:
            source_mismatch += 1
            messages.append(f"position={position}: invalid source ({type(exc).__name__})")
            findings.append(("source_mismatch", _outbox_finding_identity(position, raw_event)))
            prefix_events.append(raw_event)
            continue

        if canonical_json_subset_v1(aggregate) != canonical_json_subset_v1(expected):
            source_mismatch += 1
            messages.append(f"position={position}: aggregate does not match durable sources")
            findings.append(("source_mismatch", _outbox_finding_identity(position, raw_event)))
        else:
            matched += 1
        # The current entry becomes part of the next source prefix only after
        # its comparison, preserving append positions without self-input.
        prefix_events.append(raw_event)

    return ToolWorkflowOutboxVerificationResult(
        session_id=normalized_session_id,
        checked=checked,
        matched=matched,
        source_mismatch=source_mismatch,
        duplicate=duplicate,
        gap=gap,
        unsupported=unsupported,
        messages=tuple(messages),
        findings=tuple(findings),
    )


def _outbox_finding_identity(position: int, raw_event: Mapping[str, Any]) -> str:
    """Bind a verifier finding to both its locator and durable raw content.

    Session histories are normally append-only, but restore and migration
    paths can replace a record at the same position.  Position alone would
    then suppress a distinct finding forever.  Hash the full raw envelope so
    valid and malformed outbox payloads use the same fail-closed identity.
    """

    try:
        raw_digest = canonical_sha256_subset_v1(raw_event)
    except Exception:
        # Strict SessionStore JSONL records are JSON values, but this verifier
        # is also intentionally useful against damaged histories.  Keep the
        # fallback deterministic without exposing the raw payload in metrics.
        try:
            encoded = json.dumps(
                _clone_json(raw_event),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=repr,
            ).encode("utf-8")
        except Exception:
            encoded = repr(raw_event).encode("utf-8", errors="backslashreplace")
        raw_digest = "sha256:" + hashlib.sha256(encoded).hexdigest()
    return f"position={position};raw={raw_digest}"


class ToolWorkflowOutboxRepository:
    """SessionStore-backed aggregate cache and approval-observation repairer."""

    def __init__(
        self,
        session_store: SessionStore,
        *,
        enabled: bool = False,
        publication_gate: ToolWorkflowPublicationGate | None = None,
    ) -> None:
        self._session_store = session_store
        self._enabled = bool(enabled)
        self._publication_gate = publication_gate

    @property
    def enabled(self) -> bool:
        gate = self._publication_gate
        return self._enabled and (gate.enabled if gate is not None else True)

    async def list(self, session_id: str, *, turn_id: str | None = None) -> tuple[dict[str, Any], ...]:
        """Read validated entries even while publication is rolled back/off."""

        snapshot = await self._session_store.load_strict_snapshot(session_id)
        records = _outbox_records(session_id, snapshot.events)
        if turn_id is not None:
            records = [record for record in records if record["turn_id"] == turn_id]
        return tuple(copy.deepcopy(record) for record in records)

    async def verify_session(self, session_id: str) -> ToolWorkflowOutboxVerificationResult:
        """Replay durable sources and compare every cached aggregate read-only."""

        snapshot = await self._session_store.load_strict_snapshot(session_id)
        return verify_tool_workflow_outbox_v1(session_id, snapshot.events)

    async def rebuild_turn(self, session_id: str, turn_id: str) -> dict[str, Any] | None:
        """Repair one missing delivery entry without changing source state."""

        if not self.enabled:
            return None
        result_aggregate: dict[str, Any] | None = None
        for _ in range(8):
            snapshot = await self._session_store.load_strict_snapshot(session_id)

            def build(snapshot_under_lock: DurableSessionSnapshot) -> Sequence[Mapping[str, Any]] | None:
                nonlocal result_aggregate
                if not self.enabled:
                    return None
                companions = build_outbox_companion_events_v1(
                    session_id=session_id,
                    before_events=snapshot_under_lock.events,
                    source_events=(),
                    force_turn_ids=(turn_id,),
                )
                if not companions:
                    latest = _latest_outbox_for_turn(session_id, turn_id, snapshot_under_lock.events)
                    result_aggregate = latest
                    return None
                result_aggregate = copy.deepcopy(companions[-1]["aggregate"])
                return companions

            result = await self._session_store.mutate_strict_snapshot(
                session_id,
                expected_history_revision=snapshot.history_revision,
                build_events=build,
            )
            if result.status != "rebase_required":
                return result_aggregate
        raise ToolWorkflowOutboxError("outbox rebuild repeatedly lost the SessionStore CAS")

    async def observe_approval(self, approval: ExecApprovalRequest) -> ToolWorkflowOutboxReconcileResult:
        """Persist an idempotent ordinary-Chat approval observation then reduce.

        Task-runtime approvals are explicitly outside this aggregate stream;
        they have no ordinary-Chat session/timeline identity and are returned as
        ``not_ordinary_chat`` rather than being silently projected.
        """

        if not self.enabled:
            return ToolWorkflowOutboxReconcileResult(
                approval_id=_approval_id(approval),
                session_id=None,
                turn_id=None,
                status="disabled",
            )
        observation = approval_observation_from_request(approval)
        if observation is None:
            return ToolWorkflowOutboxReconcileResult(
                approval_id=_approval_id(approval),
                session_id=None,
                turn_id=None,
                status="not_ordinary_chat",
            )
        session_id = observation.pop("session_id")
        turn_id = observation.pop("turn_id")
        observation_event = _approval_observation_event(
            session_id=session_id,
            turn_id=turn_id,
            observation=observation,
        )
        disabled_during_mutation = False
        for _ in range(8):
            snapshot = await self._session_store.load_strict_snapshot(session_id)

            def build(snapshot_under_lock: DurableSessionSnapshot) -> Sequence[Mapping[str, Any]] | None:
                nonlocal disabled_during_mutation
                if not self.enabled:
                    disabled_during_mutation = True
                    return None
                existing = _observations_for_approval(
                    session_id,
                    turn_id,
                    snapshot_under_lock.events,
                    str(observation["approval_id"]),
                )
                source_events: tuple[Mapping[str, Any], ...]
                if existing:
                    highest = existing[0]
                    if _approval_revision_key(highest) > _approval_revision_key(observation):
                        source_events = ()
                    elif _approval_revision_key(highest) == _approval_revision_key(observation):
                        if highest != observation:
                            raise ToolWorkflowOutboxUnsupportedError(
                                "conflicting approval observation revision"
                            )
                        source_events = ()
                    else:
                        source_events = (observation_event,)
                else:
                    source_events = (observation_event,)
                companions = build_outbox_companion_events_v1(
                    session_id=session_id,
                    before_events=snapshot_under_lock.events,
                    source_events=source_events,
                    force_turn_ids=(turn_id,),
                )
                combined = (*source_events, *companions)
                return combined or None

            result = await self._session_store.mutate_strict_snapshot(
                session_id,
                expected_history_revision=snapshot.history_revision,
                build_events=build,
            )
            if result.status != "rebase_required":
                return ToolWorkflowOutboxReconcileResult(
                    approval_id=str(observation["approval_id"]),
                    session_id=session_id,
                    turn_id=turn_id,
                    status=(
                        "disabled"
                        if disabled_during_mutation
                        else "observed" if result.status == "appended" else "already_observed"
                    ),
                )
        raise ToolWorkflowOutboxError("approval observation repeatedly lost the SessionStore CAS")

    async def reconcile_checkpoint_approvals(
        self,
        session_id: str,
        approval_store: ApprovalStore,
    ) -> tuple[ToolWorkflowOutboxReconcileResult, ...]:
        """Restart repair for checkpoint-linked approvals; never execute a tool."""

        snapshot = await self._session_store.load_strict_snapshot(session_id)
        results: list[ToolWorkflowOutboxReconcileResult] = []
        for checkpoint in _latest_checkpoints(session_id, snapshot.events).values():
            approval_record = checkpoint.get("approval_record")
            if not isinstance(approval_record, Mapping):
                continue
            approval_id = approval_record.get("approval_id")
            if not isinstance(approval_id, str) or not approval_id:
                results.append(
                    ToolWorkflowOutboxReconcileResult(
                        approval_id="",
                        session_id=session_id,
                        turn_id=str(checkpoint["turn_id"]),
                        status="unsupported",
                        message="checkpoint approval reference is missing approval_id",
                    )
                )
                continue
            approval = approval_store.get(approval_id)
            if approval is None:
                results.append(
                    ToolWorkflowOutboxReconcileResult(
                        approval_id=approval_id,
                        session_id=session_id,
                        turn_id=str(checkpoint["turn_id"]),
                        status="missing_approval",
                    )
                )
                continue
            result = await self.observe_approval(approval)
            if result.status == "not_ordinary_chat":
                result = ToolWorkflowOutboxReconcileResult(
                    approval_id=approval_id,
                    session_id=session_id,
                    turn_id=str(checkpoint["turn_id"]),
                    status="unsupported",
                    message="task-runtime approval is outside ordinary-Chat aggregate scope",
                )
            results.append(result)
        return tuple(results)


def approval_observation_from_request(approval: ExecApprovalRequest) -> dict[str, Any] | None:
    """Extract only reducer identity/digest fields from an ordinary-Chat row."""

    metadata = getattr(approval, "metadata", None)
    if not isinstance(metadata, Mapping) or metadata.get("approval_source") != "ordinary_chat":
        return None
    payload = getattr(approval, "command_payload", None)
    if not isinstance(payload, Mapping):
        raise ToolWorkflowOutboxUnsupportedError("ordinary-Chat approval payload is missing")
    checkpoint = payload.get("ordinary_chat_checkpoint")
    if not isinstance(checkpoint, Mapping) or checkpoint.get("source") != "ordinary_chat":
        raise ToolWorkflowOutboxUnsupportedError("ordinary-Chat approval checkpoint is invalid")
    session_id = _text(checkpoint.get("session_id"), "approval.session_id")
    turn_id = _text(checkpoint.get("turn_id"), "approval.turn_id")
    operation_id = _text(checkpoint.get("operation_id"), "approval.operation_id")
    call_id = _text(checkpoint.get("timeline_call_id"), "approval.call_id")
    arguments_digest = _bare_digest(checkpoint.get("arguments_digest"), "approval.arguments_digest")
    if (
        payload.get("session_id") != session_id
        or payload.get("operation_id") != operation_id
        or payload.get("timeline_call_id") != call_id
        or payload.get("arguments_digest") != arguments_digest
    ):
        raise ToolWorkflowOutboxUnsupportedError("ordinary-Chat approval identity copies conflict")
    revision = getattr(approval, "approval_revision", None)
    observation = {
        "session_id": session_id,
        "turn_id": turn_id,
        "approval_id": _approval_id(approval),
        "approval_revision": revision if type(revision) is int and revision > 0 else None,
        "status": _text(getattr(approval, "status", None), "approval.status"),
        "request_digest": _bare_digest(getattr(approval, "request_digest", None), "approval.request_digest"),
        "context_digest": _bare_digest(getattr(approval, "context_digest", None), "approval.context_digest"),
        "call_id": call_id,
        "operation_id": operation_id,
        "arguments_digest": arguments_digest,
    }
    if observation["approval_revision"] is None:
        # A pre-column row has no trustworthy monotonic revision.  Preserve a
        # canonical fingerprint of the legacy source instead of inventing one.
        observation["legacy_digest"] = canonical_sha256_subset_v1(observation)
    return observation


def _affected_turn_ids(
    session_id: str,
    before_events: Sequence[Mapping[str, Any]],
    candidate_events: Sequence[Mapping[str, Any]],
    source_events: Sequence[Mapping[str, Any]],
) -> set[str]:
    turn_ids: set[str] = set()
    source_kinds = {event.get("event") for event in source_events}
    if SESSION_TURN_TIMELINE_EVENT in source_kinds:
        before = _latest_timeline(session_id, before_events)
        after = _latest_timeline(session_id, candidate_events)
        before_turns = (
            {}
            if before is None
            else {
                turn.turn_id: turn.to_dict()
                for turn in SessionTurnTimeline.from_dict(before[0]).turns
            }
        )
        if after is None:
            raise ToolWorkflowOutboxError("timeline source transition did not produce a timeline")
        for turn in SessionTurnTimeline.from_dict(after[0]).turns:
            if before_turns.get(turn.turn_id) != turn.to_dict():
                turn_ids.add(turn.turn_id)
    for event in source_events:
        kind = event.get("event")
        if kind in {TURN_CHECKPOINT_EVENT, TOOL_WORKFLOW_APPROVAL_OBSERVATION_EVENT, _RECEIPT_EVENT}:
            turn_id = event.get("turn_id")
            if not isinstance(turn_id, str) or not turn_id:
                raise ToolWorkflowOutboxError(f"{kind} source event has no turn_id")
            turn_ids.add(turn_id)
    return turn_ids


def _reduce_from_events(
    *,
    session_id: str,
    turn_id: str,
    events: Sequence[Mapping[str, Any]],
    seq: int,
) -> dict[str, Any]:
    timeline = _latest_timeline(session_id, events)
    if timeline is None:
        raise ToolWorkflowOutboxError("aggregate source has no session turn timeline")
    timeline_value, timeline_position, timeline_occurred_at = timeline
    checkpoint = _latest_checkpoint(session_id, turn_id, events)
    approvals = [
        {
            "schema_version": 1,
            "session_id": session_id,
            "turn_id": turn_id,
            **observation,
        }
        for observation in _latest_approval_observations(session_id, turn_id, events)
    ]
    receipts = _receipt_envelopes(session_id, turn_id, events)
    try:
        return reduce_tool_workflow_aggregate_v1(
            session_id=session_id,
            turn_id=turn_id,
            seq=seq,
            occurred_at=_latest_authoritative_timestamp(
                session_id=session_id,
                turn_id=turn_id,
                events=events,
                fallback=timeline_occurred_at,
            ),
            timeline=timeline_value,
            timeline_source_position=timeline_position,
            checkpoint=checkpoint,
            approvals=approvals,
            receipts=receipts,
            # Session observations are the explicit legacy adapter for
            # approval rows that predate durable revisions.
            allow_legacy_approval_rows=True,
        )
    except UnsupportedAggregateSourceError as exc:
        raise ToolWorkflowOutboxUnsupportedError(str(exc)) from exc
    except AggregatePayloadError as exc:
        raise ToolWorkflowOutboxError(str(exc)) from exc


def _latest_timeline(
    session_id: str,
    events: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], int, str] | None:
    latest: tuple[dict[str, Any], int, str] | None = None
    for index, event in enumerate(events):
        if event.get("event") != SESSION_TURN_TIMELINE_EVENT:
            continue
        if event.get("type") != "session_meta" or event.get("session_id") != session_id:
            raise ToolWorkflowOutboxError("timeline source envelope identity is invalid")
        if event.get("schema_version") != 1:
            raise ToolWorkflowOutboxUnsupportedError("unsupported timeline source event version")
        raw = event.get("timeline")
        if not isinstance(raw, Mapping):
            raise ToolWorkflowOutboxError("timeline source payload is missing")
        try:
            parsed = SessionTurnTimeline.from_dict(raw)
        except TimelineUnsupportedVersionError as exc:
            raise ToolWorkflowOutboxUnsupportedError(str(exc)) from exc
        except TimelinePayloadError as exc:
            raise ToolWorkflowOutboxError(str(exc)) from exc
        if parsed.session_id != session_id or parsed.history_current_revision != index + 1:
            raise ToolWorkflowOutboxError("timeline source position or identity is invalid")
        latest = (_clone_json(raw), index + 1, _timestamp(event.get("timestamp")))
    return latest


def _latest_checkpoint(
    session_id: str,
    turn_id: str,
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    result: dict[str, Any] | None = None
    for event in events:
        if event.get("event") != TURN_CHECKPOINT_EVENT or event.get("turn_id") != turn_id:
            continue
        if event.get("type") != "session_meta" or event.get("session_id") != session_id:
            raise ToolWorkflowOutboxError("checkpoint source envelope identity is invalid")
        raw = event.get("checkpoint")
        if not isinstance(raw, Mapping):
            raise ToolWorkflowOutboxError("checkpoint source payload is missing")
        try:
            parsed = TurnCheckpoint.from_dict(raw)
        except ValueError as exc:
            if "unsupported checkpoint version" in str(exc).lower():
                raise ToolWorkflowOutboxUnsupportedError(str(exc)) from exc
            raise ToolWorkflowOutboxError(str(exc)) from exc
        if parsed.session_id != session_id or parsed.turn_id != turn_id:
            raise ToolWorkflowOutboxError("checkpoint source identity is invalid")
        result = _clone_json(raw)
    return result


def _latest_checkpoints(session_id: str, events: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    checkpoints: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.get("event") != TURN_CHECKPOINT_EVENT:
            continue
        turn_id = event.get("turn_id")
        if not isinstance(turn_id, str) or not turn_id:
            raise ToolWorkflowOutboxError("checkpoint source event has no turn_id")
        checkpoint = _latest_checkpoint(session_id, turn_id, (event,))
        if checkpoint is not None:
            checkpoints[turn_id] = checkpoint
    return checkpoints


def _latest_approval_observations(
    session_id: str,
    turn_id: str,
    events: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_revision: dict[tuple[str, int | None], dict[str, Any]] = {}
    for raw_event in events:
        if raw_event.get("event") != TOOL_WORKFLOW_APPROVAL_OBSERVATION_EVENT:
            continue
        event = _parse_observation_event(raw_event, session_id=session_id)
        if event["turn_id"] != turn_id:
            continue
        observation = event["approval_observation"]
        approval_id = str(observation["approval_id"])
        revision_key = (approval_id, observation["approval_revision"])
        prior = by_revision.get(revision_key)
        if prior is not None:
            # Validate *every* historical (approval_id, revision) pair before
            # selecting the newest row.  Otherwise a conflicting older source
            # can be hidden by a later valid transition.
            if prior != observation:
                raise ToolWorkflowOutboxUnsupportedError(
                    "conflicting approval observation revision"
                )
            continue
        by_revision[revision_key] = observation
    latest: dict[str, dict[str, Any]] = {}
    for (approval_id, _), observation in by_revision.items():
        prior = latest.get(approval_id)
        if prior is None or _approval_revision_key(observation) > _approval_revision_key(prior):
            latest[approval_id] = observation
    return [copy.deepcopy(latest[key]) for key in sorted(latest)]


def _observations_for_approval(
    session_id: str,
    turn_id: str,
    events: Sequence[Mapping[str, Any]],
    approval_id: str,
) -> list[dict[str, Any]]:
    return [
        value
        for value in _latest_approval_observations(session_id, turn_id, events)
        if value["approval_id"] == approval_id
    ]


def _receipt_envelopes(
    session_id: str,
    turn_id: str,
    events: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    latest_timeline = _latest_timeline(session_id, events)
    if latest_timeline is None:
        raise ToolWorkflowOutboxError("artifact receipt source has no session turn timeline")
    try:
        timeline = SessionTurnTimeline.from_dict(latest_timeline[0])
    except TimelineUnsupportedVersionError as exc:
        raise ToolWorkflowOutboxUnsupportedError(str(exc)) from exc
    except TimelinePayloadError as exc:
        raise ToolWorkflowOutboxError(str(exc)) from exc
    turn = next((item for item in timeline.turns if item.turn_id == turn_id), None)
    if turn is None:
        raise ToolWorkflowOutboxError("artifact receipt source names an unknown timeline turn")
    receipts: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        if event.get("event") != _RECEIPT_EVENT or event.get("turn_id") != turn_id:
            continue
        if event.get("type") != "session_meta" or event.get("session_id") != session_id:
            raise ToolWorkflowOutboxError("artifact receipt envelope identity is invalid")
        if event.get("schema_version") != 1:
            raise ToolWorkflowOutboxUnsupportedError("unsupported artifact receipt event version")
        receipt = event.get("artifact_receipt")
        if not isinstance(receipt, Mapping):
            raise ToolWorkflowOutboxError("artifact receipt source payload is missing")
        operation_id = _text(receipt.get("operation_id"), "artifact receipt operation_id")
        raw_call_ids = receipt.get("tool_call_ids")
        if not isinstance(raw_call_ids, Sequence) or isinstance(raw_call_ids, (str, bytes)):
            raise ToolWorkflowOutboxError("artifact receipt tool_call_ids is invalid")
        call_ids = {
            _text(value, "artifact receipt tool_call_id")
            for value in raw_call_ids
        }
        matches = [
            descriptor
            for descriptor in turn.operation_descriptors
            if descriptor.operation_id == operation_id and descriptor.call_id in call_ids
        ]
        if len(matches) != 1 or matches[0].receipt_reference is None:
            raise ToolWorkflowOutboxError(
                "artifact receipt cannot be correlated to one timeline tool-result reference"
            )
        receipts.append(
            {
                "source_position": index + 1,
                "session_id": session_id,
                # The tool-result event ID is the authoritative correlation
                # reference.  The independently appended artifact receipt
                # remains a distinct source, identified by source_position.
                "receipt_reference": matches[0].receipt_reference,
                "receipt": _clone_json(receipt),
            }
        )
    return receipts


def _latest_authoritative_timestamp(
    *,
    session_id: str,
    turn_id: str,
    events: Sequence[Mapping[str, Any]],
    fallback: str,
) -> str:
    """Use append order, not clock ordering, for an aggregate occurrence time."""

    latest = fallback
    for event in events:
        kind = event.get("event")
        if kind == SESSION_TURN_TIMELINE_EVENT:
            raw = event.get("timeline")
            if event.get("session_id") != session_id or not isinstance(raw, Mapping):
                raise ToolWorkflowOutboxError("timeline source envelope identity is invalid")
            try:
                parsed = SessionTurnTimeline.from_dict(raw)
            except TimelineUnsupportedVersionError as exc:
                raise ToolWorkflowOutboxUnsupportedError(str(exc)) from exc
            except TimelinePayloadError as exc:
                raise ToolWorkflowOutboxError(str(exc)) from exc
            if parsed.session_id == session_id and any(
                item.turn_id == turn_id for item in parsed.turns
            ):
                latest = _timestamp(event.get("timestamp"))
        elif (
            kind in {TURN_CHECKPOINT_EVENT, TOOL_WORKFLOW_APPROVAL_OBSERVATION_EVENT, _RECEIPT_EVENT}
            and event.get("session_id") == session_id
            and event.get("turn_id") == turn_id
        ):
            latest = _timestamp(event.get("timestamp"))
    return latest


def _outbox_by_key(
    session_id: str,
    events: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    values: dict[tuple[str, str], dict[str, Any]] = {}
    sequences: dict[tuple[str, int], dict[str, Any]] = {}
    for aggregate in _outbox_records(session_id, events):
        key = (str(aggregate["turn_id"]), str(aggregate["idempotency_key"]))
        prior = values.get(key)
        if prior is not None and prior != aggregate:
            raise ToolWorkflowOutboxUnsupportedError("conflicting aggregate idempotency key")
        seq_key = (str(aggregate["turn_id"]), int(aggregate["seq"]))
        prior_sequence = sequences.get(seq_key)
        if prior_sequence is not None and (
            prior_sequence["event_id"] != aggregate["event_id"]
            or prior_sequence["idempotency_key"] != aggregate["idempotency_key"]
        ):
            raise ToolWorkflowOutboxUnsupportedError("conflicting aggregate sequence")
        values[key] = aggregate
        sequences[seq_key] = aggregate
    return values


def _outbox_records(session_id: str, events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw in events:
        if raw.get("event") != TOOL_WORKFLOW_OUTBOX_EVENT:
            continue
        records.append(_parse_outbox_event(session_id, raw))
    _outbox_by_key_no_recursion(records)
    return records


def _parse_outbox_event(
    session_id: str,
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    if set(raw) != _OUTBOX_EVENT_FIELDS:
        raise ToolWorkflowOutboxError("outbox event has unsupported fields")
    if raw.get("type") != "session_meta" or raw.get("session_id") != session_id:
        raise ToolWorkflowOutboxError("outbox event envelope identity is invalid")
    if raw.get("schema_version") != TOOL_WORKFLOW_OUTBOX_EVENT_VERSION:
        raise ToolWorkflowOutboxUnsupportedError("unsupported outbox event version")
    aggregate = raw.get("aggregate")
    if not isinstance(aggregate, Mapping):
        raise ToolWorkflowOutboxError("outbox aggregate payload is missing")
    if raw.get("turn_id") != aggregate.get("turn_id"):
        raise ToolWorkflowOutboxError("outbox turn identity is invalid")
    try:
        parsed = parse_tool_workflow_aggregate_v1(aggregate)
    except AggregatePayloadError as exc:
        raise ToolWorkflowOutboxError(str(exc)) from exc
    if parsed["session_id"] != session_id:
        raise ToolWorkflowOutboxError("outbox aggregate session identity is invalid")
    return parsed


def _outbox_by_key_no_recursion(records: Sequence[Mapping[str, Any]]) -> None:
    seen_keys: dict[tuple[str, str], Mapping[str, Any]] = {}
    seen_sequences: dict[tuple[str, int], Mapping[str, Any]] = {}
    for aggregate in records:
        key = (str(aggregate["turn_id"]), str(aggregate["idempotency_key"]))
        sequence = (str(aggregate["turn_id"]), int(aggregate["seq"]))
        prior_key = seen_keys.get(key)
        if prior_key is not None and prior_key != aggregate:
            raise ToolWorkflowOutboxUnsupportedError("conflicting aggregate idempotency key")
        prior_sequence = seen_sequences.get(sequence)
        if prior_sequence is not None and (
            prior_sequence["event_id"] != aggregate["event_id"]
            or prior_sequence["idempotency_key"] != aggregate["idempotency_key"]
        ):
            raise ToolWorkflowOutboxUnsupportedError("conflicting aggregate sequence")
        seen_keys[key] = aggregate
        seen_sequences[sequence] = aggregate


def _next_sequence(session_id: str, turn_id: str, events: Sequence[Mapping[str, Any]]) -> int:
    records = _outbox_records(session_id, events)
    return max((int(record["seq"]) for record in records if record["turn_id"] == turn_id), default=0) + 1


def _latest_outbox_for_turn(
    session_id: str,
    turn_id: str,
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    records = [record for record in _outbox_records(session_id, events) if record["turn_id"] == turn_id]
    return copy.deepcopy(max(records, key=lambda item: int(item["seq"]))) if records else None


def _approval_observation_event(
    *,
    session_id: str,
    turn_id: str,
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    version = (
        2
        if observation.get("approval_revision") is None
        else 1
    )
    _parse_observation_payload(observation, version=version)
    return {
        "type": "session_meta",
        "event": TOOL_WORKFLOW_APPROVAL_OBSERVATION_EVENT,
        "schema_version": version,
        "session_id": session_id,
        "turn_id": turn_id,
        "approval_observation": copy.deepcopy(dict(observation)),
        "timestamp": datetime.now(tz=UTC).isoformat(),
    }


def _parse_observation_event(raw: Mapping[str, Any], *, session_id: str) -> dict[str, Any]:
    if set(raw) != _OBSERVATION_EVENT_FIELDS:
        raise ToolWorkflowOutboxError("approval observation has unsupported fields")
    if raw.get("type") != "session_meta" or raw.get("session_id") != session_id:
        raise ToolWorkflowOutboxError("approval observation envelope identity is invalid")
    version = raw.get("schema_version")
    if version not in {1, TOOL_WORKFLOW_APPROVAL_OBSERVATION_VERSION}:
        raise ToolWorkflowOutboxUnsupportedError("unsupported approval observation version")
    turn_id = _text(raw.get("turn_id"), "approval observation turn_id")
    observation = _parse_observation_payload(raw.get("approval_observation"), version=version)
    if version == TOOL_WORKFLOW_APPROVAL_OBSERVATION_VERSION:
        legacy_digest = observation.get("legacy_digest")
        if observation.get("approval_revision") is None:
            canonical_legacy_source = {
                "session_id": session_id,
                "turn_id": turn_id,
                **{
                    key: value
                    for key, value in observation.items()
                    if key != "legacy_digest"
                },
            }
            if legacy_digest != canonical_sha256_subset_v1(canonical_legacy_source):
                raise ToolWorkflowOutboxError("legacy approval observation digest is invalid")
    _timestamp(raw.get("timestamp"))
    return {"turn_id": turn_id, "approval_observation": observation}


def _parse_observation_payload(value: Any, *, version: int) -> dict[str, Any]:
    observation = _object(value, "approval observation")
    fields = _OBSERVATION_FIELDS_V1 if version == 1 else _OBSERVATION_FIELDS_V2
    if set(observation) != fields:
        raise ToolWorkflowOutboxError("approval observation has unsupported fields")
    if version == 1:
        revision = _positive_int(observation.get("approval_revision"), "approval_revision")
        legacy_digest = None
    else:
        # Version 2 is a deliberately narrow adapter for rows created before
        # approval_revision existed.  A revision-bearing row must use v1 so a
        # future v2 meaning cannot silently enter the reducer.
        if observation.get("approval_revision") is not None:
            raise ToolWorkflowOutboxError(
                "legacy approval observation must not include a revision"
            )
        revision = None
        legacy_digest = _canonical_digest(observation.get("legacy_digest"), "legacy_digest")
    parsed = {
        "approval_id": _text(observation.get("approval_id"), "approval_id"),
        "approval_revision": revision,
        "status": _text(observation.get("status"), "approval.status"),
        "request_digest": _bare_digest(observation.get("request_digest"), "request_digest"),
        "context_digest": _bare_digest(observation.get("context_digest"), "context_digest"),
        "call_id": _text(observation.get("call_id"), "call_id"),
        "operation_id": _text(observation.get("operation_id"), "operation_id"),
        "arguments_digest": _bare_digest(observation.get("arguments_digest"), "arguments_digest"),
    }
    if version == 2:
        parsed["legacy_digest"] = legacy_digest
    return parsed


def _approval_revision_key(observation: Mapping[str, Any]) -> int:
    revision = observation.get("approval_revision")
    return revision if type(revision) is int and revision > 0 else 0


def _approval_id(approval: Any) -> str:
    return _text(getattr(approval, "approval_id", None), "approval_id")


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ToolWorkflowOutboxError(f"{name} must be an object")
    cloned = _clone_json(value)
    if not isinstance(cloned, dict):  # pragma: no cover - mapping invariant.
        raise ToolWorkflowOutboxError(f"{name} must be an object")
    return cloned


def _clone_json(value: Any) -> Any:
    """Thaw strict SessionStore MappingProxy snapshots without pickle/deepcopy."""

    if isinstance(value, Mapping):
        return {str(key): _clone_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clone_json(item) for item in value]
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ToolWorkflowOutboxError(f"{name} must be a non-empty string")
    return value


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ToolWorkflowOutboxError(f"{name} must be a positive integer")
    return value


def _bare_digest(value: Any, name: str) -> str:
    text = _text(value, name)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ToolWorkflowOutboxError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _canonical_digest(value: Any, name: str) -> str:
    text = _text(value, name)
    if not text.startswith("sha256:"):
        raise ToolWorkflowOutboxError(f"{name} must be a namespaced SHA-256 digest")
    _bare_digest(text.removeprefix("sha256:"), name)
    return text


def _timestamp(value: Any) -> str:
    text = _text(value, "timestamp")
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ToolWorkflowOutboxError("timestamp must be ISO-8601") from exc
    return text


__all__ = [
    "TOOL_WORKFLOW_APPROVAL_OBSERVATION_EVENT",
    "TOOL_WORKFLOW_APPROVAL_OBSERVATION_VERSION",
    "TOOL_WORKFLOW_OUTBOX_EVENT",
    "TOOL_WORKFLOW_OUTBOX_EVENT_VERSION",
    "ToolWorkflowOutboxError",
    "ToolWorkflowOutboxReconcileResult",
    "ToolWorkflowOutboxRepository",
    "ToolWorkflowOutboxUnsupportedError",
    "ToolWorkflowOutboxVerificationResult",
    "ToolWorkflowOutboxVerifierDiagnostics",
    "approval_observation_from_request",
    "build_outbox_companion_events_v1",
    "verify_tool_workflow_outbox_v1",
]
