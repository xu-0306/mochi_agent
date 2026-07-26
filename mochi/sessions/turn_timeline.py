"""Strict, durable admission and lane state for ordinary Chat turns.

This module is deliberately a persistence foundation only.  It never executes
models or tools and it never keeps the session lock across an await.  A later
runtime integration is responsible for binding this FIFO lane to prompt
construction, cancellation delivery, approval reconciliation, and tool calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, Callable, Literal, Mapping, Sequence

from mochi.sessions.store import (
    DurableSessionSnapshot,
    SessionStore,
    StrictSessionSnapshotError,
)

SESSION_TURN_TIMELINE_EVENT = "session_turn_timeline"
SESSION_TURN_TIMELINE_EVENT_VERSION = 1
_LEGACY_SESSION_TURN_TIMELINE_VERSION = "session-turn-timeline-v1"
_LEGACY_V2_SESSION_TURN_TIMELINE_VERSION = "session-turn-timeline-v2"
_LEGACY_V3_SESSION_TURN_TIMELINE_VERSION = "session-turn-timeline-v3"
SESSION_TURN_TIMELINE_VERSION = "session-turn-timeline-v4"

TimelineLoadStatus = Literal["loaded", "missing", "invalid", "unsupported_version"]
TimelineMutationStatus = Literal[
    "admitted",
    "claimed",
    "precommitted",
    "operation_abandoned",
    "operation_result",
    "admission_lease_renewed",
    "lease_renewed",
    "recovery_cancelled",
    "recovery_pending_approval",
    "recovery_admission_cancelled",
    "recovery_unknown",
    "terminal",
    "boundary_updated",
    "duplicate",
    "queue_empty",
    "lane_busy",
    "admission_busy",
    "admission_stale",
    "admission_invalid",
    "lease_stale",
    "lease_invalid",
    "already_terminal",
    "rebase_required",
    "missing",
    "invalid",
    "unsupported_version",
]
TurnTimelineStatus = Literal["queued", "running", "terminal"]
SideEffectBoundary = Literal["not_started", "started", "unknown"]
TerminalOutcome = Literal["completed", "cancelled", "blocked", "unknown"]
CancellationOutcome = Literal["cancelled_queued", "cancelled_running"]
RecoveryReason = Literal["admission_owner_expired_before_claim"]

_EVENT_FIELDS = frozenset(
    {
        "type",
        "event",
        "schema_version",
        "session_id",
        "timeline",
        "timestamp",
    }
)
_TIMELINE_FIELDS = frozenset(
    {
        "timeline_version",
        "session_id",
        "history_base_revision",
        "history_current_revision",
        "turns",
        "lane_turn_id",
        "lane_owner",
        "lane_token",
        "lane_lease_expires_at",
    }
)
_TURN_FIELDS = frozenset(
    {
        "sequence",
        "turn_id",
        "status",
        "side_effect_boundary",
        "operation_descriptors",
        "legacy_operation_ids",
        "cancellation_outcome",
        "terminal_outcome",
        "admission_owner",
        "admission_token",
        "admission_lease_expires_at",
        "recovery_reason",
    }
)
_LEGACY_V3_TURN_FIELDS = frozenset(
    {
        "sequence",
        "turn_id",
        "status",
        "side_effect_boundary",
        "operation_descriptors",
        "legacy_operation_ids",
        "cancellation_outcome",
        "terminal_outcome",
    }
)
_LEGACY_TURN_FIELDS = frozenset(
    {
        "sequence",
        "turn_id",
        "status",
        "side_effect_boundary",
        "operation_ids",
        "cancellation_outcome",
        "terminal_outcome",
    }
)
_OPERATION_DESCRIPTOR_FIELDS = frozenset(
    {
        "operation_id",
        "tool_name",
        "arguments_digest",
        "call_id",
        "status",
        "precommit_boundary",
        "result_digest",
        "receipt_reference",
    }
)
_V2_OPERATION_DESCRIPTOR_FIELDS = frozenset(
    {
        "operation_id",
        "tool_name",
        "arguments_digest",
        "call_id",
        "status",
        "precommit_boundary",
    }
)


class TimelinePayloadError(ValueError):
    """The timeline payload is malformed and must not be recovered from."""


class TimelineUnsupportedVersionError(TimelinePayloadError):
    """A future timeline envelope or state version was encountered."""


@dataclass(frozen=True)
class OperationDescriptor:
    """Exact, non-secret evidence for one operation selected after admission."""

    operation_id: str
    tool_name: str
    arguments_digest: str
    call_id: str
    status: Literal[
        "precommitted",
        "started",
        "succeeded",
        "failed",
        "unknown",
        "abandoned",
    ] = "precommitted"
    precommit_boundary: SideEffectBoundary = "not_started"
    result_digest: str | None = None
    receipt_reference: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation_id", _require_text(self.operation_id, "operation_id"))
        object.__setattr__(self, "tool_name", _require_text(self.tool_name, "tool_name"))
        object.__setattr__(self, "call_id", _require_text(self.call_id, "call_id"))
        object.__setattr__(
            self,
            "arguments_digest",
            _require_bare_sha256_digest(self.arguments_digest, "arguments_digest"),
        )
        if self.result_digest is not None:
            object.__setattr__(
                self,
                "result_digest",
                _require_namespaced_sha256_digest(self.result_digest, "result_digest"),
            )
        if self.receipt_reference is not None:
            object.__setattr__(
                self,
                "receipt_reference",
                _require_text(self.receipt_reference, "receipt_reference"),
            )
        allowed = {
            ("precommitted", "not_started"),
            ("started", "started"),
            ("succeeded", "started"),
            ("failed", "started"),
            ("unknown", "unknown"),
            ("abandoned", "not_started"),
        }
        if (self.status, self.precommit_boundary) not in allowed:
            raise TimelinePayloadError(
                "operation descriptor status and precommit_boundary are inconsistent"
            )
        if self.status in {"precommitted", "started", "unknown"} and (
            self.result_digest is not None or self.receipt_reference is not None
        ):
            raise TimelinePayloadError(
                "only known operation results may include result evidence"
            )

    def to_dict(self) -> dict[str, str | None]:
        return {
            "operation_id": self.operation_id,
            "tool_name": self.tool_name,
            "arguments_digest": self.arguments_digest,
            "call_id": self.call_id,
            "status": self.status,
            "precommit_boundary": self.precommit_boundary,
            "result_digest": self.result_digest,
            "receipt_reference": self.receipt_reference,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OperationDescriptor:
        _require_exact_fields(value, _OPERATION_DESCRIPTOR_FIELDS, "operation descriptor")
        return cls(
            operation_id=_require_text(value.get("operation_id"), "operation_id"),
            tool_name=_require_text(value.get("tool_name"), "tool_name"),
            arguments_digest=_require_bare_sha256_digest(
                value.get("arguments_digest"), "arguments_digest"
            ),
            call_id=_require_text(value.get("call_id"), "call_id"),
            status=_require_text(value.get("status"), "status"),  # type: ignore[arg-type]
            precommit_boundary=_require_text(
                value.get("precommit_boundary"), "precommit_boundary"
            ),  # type: ignore[arg-type]
            result_digest=_optional_namespaced_digest(value.get("result_digest"), "result_digest"),
            receipt_reference=_optional_text(
                value.get("receipt_reference"), "receipt_reference"
            ),
        )

    @classmethod
    def from_v2_dict(cls, value: Mapping[str, Any]) -> OperationDescriptor:
        """Migrate v2 descriptor lifecycle names without inventing results."""
        _require_exact_fields(value, _V2_OPERATION_DESCRIPTOR_FIELDS, "v2 operation descriptor")
        raw_status = _require_text(value.get("status"), "status")
        raw_boundary = _require_text(value.get("precommit_boundary"), "precommit_boundary")
        migrated_status = {
            "precommitted": "precommitted",
            "side_effect_started": "started",
            "unknown": "unknown",
        }.get(raw_status)
        if migrated_status is None:
            raise TimelinePayloadError(f"unsupported v2 operation descriptor status: {raw_status!r}")
        return cls(
            operation_id=_require_text(value.get("operation_id"), "operation_id"),
            tool_name=_require_text(value.get("tool_name"), "tool_name"),
            arguments_digest=_legacy_namespaced_arguments_digest(
                value.get("arguments_digest"), "arguments_digest"
            ),
            call_id=_require_text(value.get("call_id"), "call_id"),
            status=migrated_status,  # type: ignore[arg-type]
            precommit_boundary=raw_boundary,  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class TimelineTurn:
    """One admitted turn and its later-discovered exact operation descriptors."""

    sequence: int
    turn_id: str
    status: TurnTimelineStatus
    side_effect_boundary: SideEffectBoundary
    operation_descriptors: tuple[OperationDescriptor, ...] = ()
    legacy_operation_ids: tuple[str, ...] = ()
    cancellation_outcome: CancellationOutcome | None = None
    terminal_outcome: TerminalOutcome | None = None
    admission_owner: str | None = None
    admission_token: str | None = None
    admission_lease_expires_at: str | None = None
    recovery_reason: RecoveryReason | None = None

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or self.sequence <= 0:
            raise TimelinePayloadError("turn sequence must be a positive integer")
        object.__setattr__(self, "turn_id", _require_text(self.turn_id, "turn_id"))
        if self.status not in {"queued", "running", "terminal"}:
            raise TimelinePayloadError(f"unsupported turn status: {self.status!r}")
        if self.side_effect_boundary not in {"not_started", "started", "unknown"}:
            raise TimelinePayloadError(
                f"unsupported side_effect_boundary: {self.side_effect_boundary!r}"
            )
        descriptors = tuple(self.operation_descriptors)
        if any(not isinstance(descriptor, OperationDescriptor) for descriptor in descriptors):
            raise TimelinePayloadError("operation_descriptors must contain OperationDescriptor objects")
        descriptor_ids = tuple(descriptor.operation_id for descriptor in descriptors)
        descriptor_calls = tuple(descriptor.call_id for descriptor in descriptors)
        if len(set(descriptor_ids)) != len(descriptor_ids):
            raise TimelinePayloadError("operation descriptor ids must be unique within a turn")
        if len(set(descriptor_calls)) != len(descriptor_calls):
            raise TimelinePayloadError("operation descriptor call ids must be unique within a turn")
        legacy_operation_ids = _normalize_identifiers(
            self.legacy_operation_ids,
            "legacy_operation_ids",
        )
        if set(descriptor_ids).intersection(legacy_operation_ids):
            raise TimelinePayloadError("legacy operation ids cannot become exact descriptors")
        object.__setattr__(self, "operation_descriptors", descriptors)
        object.__setattr__(self, "legacy_operation_ids", legacy_operation_ids)
        if self.cancellation_outcome not in {
            None,
            "cancelled_queued",
            "cancelled_running",
        }:
            raise TimelinePayloadError(
                f"unsupported cancellation_outcome: {self.cancellation_outcome!r}"
            )
        if self.terminal_outcome not in {None, "completed", "cancelled", "blocked", "unknown"}:
            raise TimelinePayloadError(
                f"unsupported terminal_outcome: {self.terminal_outcome!r}"
            )
        if self.recovery_reason not in {None, "admission_owner_expired_before_claim"}:
            raise TimelinePayloadError(f"unsupported recovery_reason: {self.recovery_reason!r}")
        admission_fields = (
            self.admission_owner,
            self.admission_token,
            self.admission_lease_expires_at,
        )
        if any(value is None for value in admission_fields):
            if any(value is not None for value in admission_fields):
                raise TimelinePayloadError(
                    "admission identity must be either complete or absent"
                )
        else:
            object.__setattr__(
                self,
                "admission_owner",
                _require_text(self.admission_owner, "admission_owner"),
            )
            object.__setattr__(
                self,
                "admission_token",
                _require_text(self.admission_token, "admission_token"),
            )
            object.__setattr__(
                self,
                "admission_lease_expires_at",
                _normalize_timestamp(
                    self.admission_lease_expires_at,
                    "admission_lease_expires_at",
                ),
            )
        if self.status == "queued":
            if (
                self.side_effect_boundary != "not_started"
                or self.operation_descriptors
                or self.cancellation_outcome is not None
                or self.terminal_outcome is not None
                or self.recovery_reason is not None
            ):
                raise TimelinePayloadError("queued turn must have no side effect or terminal outcome")
        elif self.status == "running":
            if any(value is not None for value in admission_fields):
                raise TimelinePayloadError("running turn cannot retain admission identity")
            if (
                self.cancellation_outcome is not None
                or self.terminal_outcome is not None
                or self.recovery_reason is not None
            ):
                raise TimelinePayloadError("running turn must have no terminal outcome")
        else:
            if any(value is not None for value in admission_fields):
                raise TimelinePayloadError("terminal turn cannot retain admission identity")
            if self.terminal_outcome is None:
                raise TimelinePayloadError("terminal turn requires terminal_outcome")
            if self.recovery_reason is not None and (
                self.terminal_outcome != "cancelled"
                or self.cancellation_outcome != "cancelled_queued"
                or self.side_effect_boundary != "not_started"
            ):
                raise TimelinePayloadError(
                    "admission recovery reason requires a queued no-effect cancellation"
                )
            if self.cancellation_outcome is not None:
                if self.terminal_outcome != "cancelled":
                    raise TimelinePayloadError(
                        "cancellation_outcome requires cancelled terminal_outcome"
                    )
                if (
                    self.cancellation_outcome == "cancelled_queued"
                    and self.side_effect_boundary != "not_started"
                ):
                    raise TimelinePayloadError(
                        "queued cancellation cannot cross the side-effect boundary"
                    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "turn_id": self.turn_id,
            "status": self.status,
            "side_effect_boundary": self.side_effect_boundary,
            "operation_descriptors": [
                descriptor.to_dict() for descriptor in self.operation_descriptors
            ],
            "legacy_operation_ids": list(self.legacy_operation_ids),
            "cancellation_outcome": self.cancellation_outcome,
            "terminal_outcome": self.terminal_outcome,
            "admission_owner": self.admission_owner,
            "admission_token": self.admission_token,
            "admission_lease_expires_at": self.admission_lease_expires_at,
            "recovery_reason": self.recovery_reason,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TimelineTurn:
        _require_exact_fields(value, _TURN_FIELDS, "timeline turn")
        descriptors = value.get("operation_descriptors")
        legacy_operation_ids = value.get("legacy_operation_ids")
        if not isinstance(descriptors, (list, tuple)):
            raise TimelinePayloadError("timeline turn operation_descriptors must be a JSON array")
        if not isinstance(legacy_operation_ids, (list, tuple)):
            raise TimelinePayloadError("timeline turn legacy_operation_ids must be a JSON array")
        if any(not isinstance(item, Mapping) for item in descriptors):
            raise TimelinePayloadError("timeline operation_descriptors must contain objects")
        status = value.get("status")
        boundary = value.get("side_effect_boundary")
        terminal_outcome = value.get("terminal_outcome")
        cancellation_outcome = value.get("cancellation_outcome")
        if not isinstance(status, str) or not isinstance(boundary, str):
            raise TimelinePayloadError("timeline turn status fields must be strings")
        if terminal_outcome is not None and not isinstance(terminal_outcome, str):
            raise TimelinePayloadError("timeline turn terminal_outcome must be a string or null")
        if cancellation_outcome is not None and not isinstance(cancellation_outcome, str):
            raise TimelinePayloadError(
                "timeline turn cancellation_outcome must be a string or null"
            )
        return cls(
            sequence=_require_positive_int(value.get("sequence"), "sequence"),
            turn_id=_require_text(value.get("turn_id"), "turn_id"),
            status=status,  # type: ignore[arg-type]
            side_effect_boundary=boundary,  # type: ignore[arg-type]
            operation_descriptors=tuple(
                OperationDescriptor.from_dict(item) for item in descriptors
            ),
            legacy_operation_ids=tuple(legacy_operation_ids),
            cancellation_outcome=cancellation_outcome,  # type: ignore[arg-type]
            terminal_outcome=terminal_outcome,  # type: ignore[arg-type]
            admission_owner=_optional_text(value.get("admission_owner"), "admission_owner"),
            admission_token=_optional_text(value.get("admission_token"), "admission_token"),
            admission_lease_expires_at=_optional_text(
                value.get("admission_lease_expires_at"),
                "admission_lease_expires_at",
            ),
            recovery_reason=value.get("recovery_reason"),  # type: ignore[arg-type]
        )

    @classmethod
    def from_v3_dict(cls, value: Mapping[str, Any]) -> TimelineTurn:
        """Read v3 turns without fabricating an admission owner or deadline."""
        _require_exact_fields(value, _LEGACY_V3_TURN_FIELDS, "v3 timeline turn")
        migrated = dict(value)
        migrated.update(
            {
                "admission_owner": None,
                "admission_token": None,
                "admission_lease_expires_at": None,
                "recovery_reason": None,
            }
        )
        return cls.from_dict(migrated)

    @classmethod
    def from_v1_dict(cls, value: Mapping[str, Any]) -> TimelineTurn:
        """Read v1 IDs without fabricating tool, call, or argument evidence."""
        _require_exact_fields(value, _LEGACY_TURN_FIELDS, "legacy timeline turn")
        operation_ids = value.get("operation_ids")
        if not isinstance(operation_ids, (list, tuple)):
            raise TimelinePayloadError("legacy timeline turn operation_ids must be a JSON array")
        status = value.get("status")
        boundary = value.get("side_effect_boundary")
        terminal_outcome = value.get("terminal_outcome")
        cancellation_outcome = value.get("cancellation_outcome")
        if not isinstance(status, str) or not isinstance(boundary, str):
            raise TimelinePayloadError("legacy timeline turn status fields must be strings")
        if terminal_outcome is not None and not isinstance(terminal_outcome, str):
            raise TimelinePayloadError("legacy timeline terminal_outcome must be a string or null")
        if cancellation_outcome is not None and not isinstance(cancellation_outcome, str):
            raise TimelinePayloadError(
                "legacy timeline cancellation_outcome must be a string or null"
            )
        return cls(
            sequence=_require_positive_int(value.get("sequence"), "sequence"),
            turn_id=_require_text(value.get("turn_id"), "turn_id"),
            status=status,  # type: ignore[arg-type]
            side_effect_boundary=boundary,  # type: ignore[arg-type]
            legacy_operation_ids=tuple(operation_ids),
            cancellation_outcome=cancellation_outcome,  # type: ignore[arg-type]
            terminal_outcome=terminal_outcome,  # type: ignore[arg-type]
        )

    @classmethod
    def from_v2_dict(cls, value: Mapping[str, Any]) -> TimelineTurn:
        """Read v2 turns while upgrading descriptor lifecycle labels to v3."""
        _require_exact_fields(value, _LEGACY_V3_TURN_FIELDS, "v2 timeline turn")
        descriptors = value.get("operation_descriptors")
        legacy_operation_ids = value.get("legacy_operation_ids")
        if not isinstance(descriptors, (list, tuple)):
            raise TimelinePayloadError("v2 operation_descriptors must be a JSON array")
        if not isinstance(legacy_operation_ids, (list, tuple)):
            raise TimelinePayloadError("v2 legacy_operation_ids must be a JSON array")
        if any(not isinstance(item, Mapping) for item in descriptors):
            raise TimelinePayloadError("v2 operation_descriptors must contain objects")
        status = value.get("status")
        boundary = value.get("side_effect_boundary")
        terminal_outcome = value.get("terminal_outcome")
        cancellation_outcome = value.get("cancellation_outcome")
        if not isinstance(status, str) or not isinstance(boundary, str):
            raise TimelinePayloadError("v2 timeline turn status fields must be strings")
        if terminal_outcome is not None and not isinstance(terminal_outcome, str):
            raise TimelinePayloadError("v2 timeline terminal_outcome must be a string or null")
        if cancellation_outcome is not None and not isinstance(cancellation_outcome, str):
            raise TimelinePayloadError(
                "v2 timeline cancellation_outcome must be a string or null"
            )
        return cls(
            sequence=_require_positive_int(value.get("sequence"), "sequence"),
            turn_id=_require_text(value.get("turn_id"), "turn_id"),
            status=status,  # type: ignore[arg-type]
            side_effect_boundary=boundary,  # type: ignore[arg-type]
            operation_descriptors=tuple(
                OperationDescriptor.from_v2_dict(item) for item in descriptors
            ),
            legacy_operation_ids=tuple(legacy_operation_ids),
            cancellation_outcome=cancellation_outcome,  # type: ignore[arg-type]
            terminal_outcome=terminal_outcome,  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class SessionTurnTimeline:
    """Versioned state for one session's FIFO logical execution lane."""

    session_id: str
    history_base_revision: int
    history_current_revision: int
    turns: tuple[TimelineTurn, ...]
    lane_turn_id: str | None = None
    lane_owner: str | None = None
    lane_token: str | None = None
    lane_lease_expires_at: str | None = None
    timeline_version: str = SESSION_TURN_TIMELINE_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", _require_text(self.session_id, "session_id"))
        if self.timeline_version != SESSION_TURN_TIMELINE_VERSION:
            raise TimelineUnsupportedVersionError(
                f"unsupported timeline version: {self.timeline_version!r}"
            )
        if type(self.history_base_revision) is not int or self.history_base_revision < 0:
            raise TimelinePayloadError("history_base_revision must be a non-negative integer")
        if type(self.history_current_revision) is not int or self.history_current_revision <= 0:
            raise TimelinePayloadError("history_current_revision must be a positive integer")
        if self.history_current_revision <= self.history_base_revision:
            raise TimelinePayloadError("history_current_revision must follow history_base_revision")
        turns = tuple(self.turns)
        if any(not isinstance(turn, TimelineTurn) for turn in turns):
            raise TimelinePayloadError("turns must contain TimelineTurn objects")
        if tuple(turn.sequence for turn in turns) != tuple(range(1, len(turns) + 1)):
            raise TimelinePayloadError("turn sequences must be contiguous and ordered")
        turn_ids = tuple(turn.turn_id for turn in turns)
        if len(set(turn_ids)) != len(turn_ids):
            raise TimelinePayloadError("turn ids must be unique")
        all_operation_ids = tuple(
            operation_id
            for turn in turns
            for operation_id in (
                *(descriptor.operation_id for descriptor in turn.operation_descriptors),
                *turn.legacy_operation_ids,
            )
        )
        if len(set(all_operation_ids)) != len(all_operation_ids):
            raise TimelinePayloadError("operation ids must not be reused across timeline turns")
        all_call_ids = tuple(
            descriptor.call_id
            for turn in turns
            for descriptor in turn.operation_descriptors
        )
        if len(set(all_call_ids)) != len(all_call_ids):
            raise TimelinePayloadError("operation call ids must not be reused across timeline turns")
        object.__setattr__(self, "turns", turns)

        running = tuple(turn for turn in turns if turn.status == "running")
        if len(running) > 1:
            raise TimelinePayloadError("only one turn may be running")
        lane_fields = (
            self.lane_turn_id,
            self.lane_owner,
            self.lane_token,
            self.lane_lease_expires_at,
        )
        if any(value is None for value in lane_fields):
            if any(value is not None for value in lane_fields):
                raise TimelinePayloadError("lane identity must be either complete or absent")
            if running:
                raise TimelinePayloadError("running turn requires a lane claim")
        else:
            lane_turn_id = _require_text(self.lane_turn_id, "lane_turn_id")
            object.__setattr__(self, "lane_turn_id", lane_turn_id)
            object.__setattr__(self, "lane_owner", _require_text(self.lane_owner, "lane_owner"))
            object.__setattr__(self, "lane_token", _require_text(self.lane_token, "lane_token"))
            object.__setattr__(
                self,
                "lane_lease_expires_at",
                _normalize_timestamp(self.lane_lease_expires_at, "lane_lease_expires_at"),
            )
            if len(running) != 1 or running[0].turn_id != lane_turn_id:
                raise TimelinePayloadError("lane claim must name the single running turn")
        if running:
            running_sequence = running[0].sequence
            if any(turn.status == "queued" and turn.sequence < running_sequence for turn in turns):
                raise TimelinePayloadError("running turn must be the FIFO queue head")

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeline_version": self.timeline_version,
            "session_id": self.session_id,
            "history_base_revision": self.history_base_revision,
            "history_current_revision": self.history_current_revision,
            "turns": [turn.to_dict() for turn in self.turns],
            "lane_turn_id": self.lane_turn_id,
            "lane_owner": self.lane_owner,
            "lane_token": self.lane_token,
            "lane_lease_expires_at": self.lane_lease_expires_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SessionTurnTimeline:
        _require_exact_fields(value, _TIMELINE_FIELDS, "session turn timeline")
        version = value.get("timeline_version")
        if version not in {
            _LEGACY_SESSION_TURN_TIMELINE_VERSION,
            _LEGACY_V2_SESSION_TURN_TIMELINE_VERSION,
            _LEGACY_V3_SESSION_TURN_TIMELINE_VERSION,
            SESSION_TURN_TIMELINE_VERSION,
        }:
            raise TimelineUnsupportedVersionError(f"unsupported timeline version: {version!r}")
        raw_turns = value.get("turns")
        if not isinstance(raw_turns, (list, tuple)):
            raise TimelinePayloadError("timeline turns must be a JSON array")
        if any(not isinstance(turn, Mapping) for turn in raw_turns):
            raise TimelinePayloadError("timeline turns must contain objects")
        return cls(
            session_id=_require_text(value.get("session_id"), "session_id"),
            history_base_revision=_require_non_negative_int(
                value.get("history_base_revision"), "history_base_revision"
            ),
            history_current_revision=_require_positive_int(
                value.get("history_current_revision"), "history_current_revision"
            ),
            turns=tuple(
                (
                    TimelineTurn.from_v1_dict(turn)
                    if version == _LEGACY_SESSION_TURN_TIMELINE_VERSION
                    else (
                        TimelineTurn.from_v2_dict(turn)
                        if version == _LEGACY_V2_SESSION_TURN_TIMELINE_VERSION
                        else (
                            TimelineTurn.from_v3_dict(turn)
                            if version == _LEGACY_V3_SESSION_TURN_TIMELINE_VERSION
                            else TimelineTurn.from_dict(turn)
                        )
                    )
                )
                for turn in raw_turns
            ),
            lane_turn_id=_optional_text(value.get("lane_turn_id"), "lane_turn_id"),
            lane_owner=_optional_text(value.get("lane_owner"), "lane_owner"),
            lane_token=_optional_text(value.get("lane_token"), "lane_token"),
            lane_lease_expires_at=_optional_text(
                value.get("lane_lease_expires_at"), "lane_lease_expires_at"
            ),
        )


@dataclass(frozen=True)
class TimelineLoadResult:
    status: TimelineLoadStatus
    timeline: SessionTurnTimeline | None
    history_revision: str | None
    event_count: int = 0
    message: str | None = None


@dataclass(frozen=True)
class TimelineMutationResult:
    status: TimelineMutationStatus
    timeline: SessionTurnTimeline | None
    history_revision: str | None
    message: str | None = None


@dataclass(frozen=True)
class _Reduction:
    status: TimelineMutationStatus
    timeline: SessionTurnTimeline | None
    message: str | None = None


class SessionTurnTimelineRepository:
    """Strict history-CAS repository for admission and one logical lane.

    A session with no timeline event is a valid ``missing`` state and the first
    admission anchors ``history_base_revision`` to the accepted event count.
    The v1 reader preserves historical operation IDs as non-replayable legacy
    identities.  The v2 reader upgrades its descriptor lifecycle names without
    inventing result evidence.  All new writes use v4 descriptors.  Corrupt and
    future payloads are rejected.
    """

    def __init__(self, session_store: SessionStore) -> None:
        self._session_store = session_store

    async def load(self, session_id: str) -> TimelineLoadResult:
        try:
            snapshot = await self._session_store.load_strict_snapshot(session_id)
        except StrictSessionSnapshotError as exc:
            return TimelineLoadResult("invalid", None, None, message=str(exc))
        return self._load_from_snapshot(session_id, snapshot)

    async def admit(
        self,
        session_id: str,
        *,
        turn_id: str,
        expected_history_revision: str,
        companion_events: Sequence[Mapping[str, Any]] = (),
        admission_owner: str | None = None,
        admission_token: str | None = None,
        admission_lease_expires_at: str | None = None,
        now: datetime | None = None,
    ) -> TimelineMutationResult:
        turn_id = _require_text(turn_id, "turn_id")
        admission_values = (
            admission_owner,
            admission_token,
            admission_lease_expires_at,
        )
        if any(value is None for value in admission_values):
            if any(value is not None for value in admission_values):
                raise ValueError("admission identity must be supplied completely")
            normalized_admission_owner = None
            normalized_admission_token = None
            normalized_admission_expiry = None
        else:
            normalized_admission_owner = _require_text(admission_owner, "admission_owner")
            normalized_admission_token = _require_text(admission_token, "admission_token")
            admission_now = _coerce_now(now)
            expiry = _parse_timestamp(admission_lease_expires_at, "admission_lease_expires_at")
            if expiry <= admission_now:
                raise ValueError("admission_lease_expires_at must be in the future")
            normalized_admission_expiry = expiry.isoformat()

        def reduce(
            current: SessionTurnTimeline | None,
            snapshot: DurableSessionSnapshot,
        ) -> _Reduction:
            existing = _find_turn(current, turn_id) if current is not None else None
            if existing is not None:
                if (
                    existing.status == "queued"
                    and existing.admission_owner == normalized_admission_owner
                    and existing.admission_token == normalized_admission_token
                    and existing.admission_lease_expires_at == normalized_admission_expiry
                ):
                    return _Reduction("duplicate", current, "exact queued admission is already durable")
                return _Reduction(
                    "admission_invalid",
                    current,
                    "turn_id is already admitted by a different admission identity",
                )
            if current is None:
                next_timeline = SessionTurnTimeline(
                    session_id=session_id,
                    history_base_revision=snapshot.event_count,
                    history_current_revision=snapshot.event_count + 1,
                    turns=(
                        TimelineTurn(
                            sequence=1,
                            turn_id=turn_id,
                            status="queued",
                            side_effect_boundary="not_started",
                            admission_owner=normalized_admission_owner,
                            admission_token=normalized_admission_token,
                            admission_lease_expires_at=normalized_admission_expiry,
                        ),
                    ),
                )
            else:
                next_timeline = self._next_timeline(
                    current,
                    snapshot=snapshot,
                    turns=(
                        *current.turns,
                        TimelineTurn(
                            sequence=len(current.turns) + 1,
                            turn_id=turn_id,
                            status="queued",
                            side_effect_boundary="not_started",
                            admission_owner=normalized_admission_owner,
                            admission_token=normalized_admission_token,
                            admission_lease_expires_at=normalized_admission_expiry,
                        ),
                    ),
                )
            return _Reduction("admitted", next_timeline)

        return await self._apply(
            session_id,
            expected_history_revision,
            reduce,
            companion_events=companion_events,
        )

    async def claim_next(
        self,
        session_id: str,
        *,
        expected_history_revision: str,
        owner: str,
        token: str,
        lease_expires_at: str,
        now: datetime | None = None,
    ) -> TimelineMutationResult:
        owner = _require_text(owner, "owner")
        token = _require_text(token, "token")
        now_utc = _coerce_now(now)
        lease = _parse_timestamp(lease_expires_at, "lease_expires_at")
        if lease <= now_utc:
            raise ValueError("lease_expires_at must be in the future")
        lease_text = lease.isoformat()

        def reduce(
            current: SessionTurnTimeline | None,
            snapshot: DurableSessionSnapshot,
        ) -> _Reduction:
            if current is None:
                return _Reduction("missing", None, "timeline has no admitted turns")
            if current.lane_turn_id is not None:
                current_expiry = _parse_timestamp(
                    current.lane_lease_expires_at,
                    "lane_lease_expires_at",
                )
                if current_expiry <= now_utc:
                    return _Reduction("lease_stale", current, "existing lane lease is stale")
                return _Reduction("lane_busy", current, "another lane claim is active")
            next_turn = next((turn for turn in current.turns if turn.status == "queued"), None)
            if next_turn is None:
                return _Reduction("queue_empty", current, "no queued turn is eligible for the lane")
            if next_turn.admission_owner is not None:
                assert next_turn.admission_token is not None
                assert next_turn.admission_lease_expires_at is not None
                admission_expiry = _parse_timestamp(
                    next_turn.admission_lease_expires_at,
                    "admission_lease_expires_at",
                )
                if admission_expiry <= now_utc:
                    return _Reduction(
                        "admission_stale",
                        current,
                        "queued admission deadline is stale",
                    )
                if (
                    next_turn.admission_owner != owner
                    or next_turn.admission_token != token
                ):
                    return _Reduction(
                        "admission_busy",
                        current,
                        "queued admission belongs to another live owner",
                    )
            running = replace(
                next_turn,
                status="running",
                admission_owner=None,
                admission_token=None,
                admission_lease_expires_at=None,
            )
            turns = tuple(running if turn.turn_id == next_turn.turn_id else turn for turn in current.turns)
            return _Reduction(
                "claimed",
                self._next_timeline(
                    current,
                    snapshot=snapshot,
                    turns=turns,
                    lane_turn_id=next_turn.turn_id,
                    lane_owner=owner,
                    lane_token=token,
                    lane_lease_expires_at=lease_text,
                ),
            )

        return await self._apply(session_id, expected_history_revision, reduce)

    async def renew_lease(
        self,
        session_id: str,
        *,
        turn_id: str,
        expected_history_revision: str,
        owner: str,
        token: str,
        lease_expires_at: str,
        now: datetime | None = None,
    ) -> TimelineMutationResult:
        """Extend the active lane lease without holding it across external work."""
        turn_id = _require_text(turn_id, "turn_id")
        owner = _require_text(owner, "owner")
        token = _require_text(token, "token")
        now_utc = _coerce_now(now)
        next_expiry = _parse_timestamp(lease_expires_at, "lease_expires_at")
        if next_expiry <= now_utc:
            raise ValueError("lease_expires_at must be in the future")

        def reduce(
            current: SessionTurnTimeline | None,
            snapshot: DurableSessionSnapshot,
        ) -> _Reduction:
            lease_status = _lease_status(current, turn_id, owner, token, now_utc)
            if lease_status is not None:
                return _Reduction(lease_status, current, "lane identity is not usable")
            assert current is not None and current.lane_lease_expires_at is not None
            current_expiry = _parse_timestamp(
                current.lane_lease_expires_at,
                "lane_lease_expires_at",
            )
            if next_expiry <= current_expiry:
                return _Reduction(
                    "invalid",
                    current,
                    "renewed lease must extend the current expiry",
                )
            return _Reduction(
                "lease_renewed",
                self._next_timeline(
                    current,
                    snapshot=snapshot,
                    turns=current.turns,
                    lane_lease_expires_at=next_expiry.isoformat(),
                ),
            )

        return await self._apply(session_id, expected_history_revision, reduce)

    async def renew_queued_admission(
        self,
        session_id: str,
        *,
        turn_id: str,
        expected_history_revision: str,
        owner: str,
        token: str,
        admission_lease_expires_at: str,
        now: datetime | None = None,
    ) -> TimelineMutationResult:
        """Extend a live queued admission while it waits behind the FIFO lane."""
        turn_id = _require_text(turn_id, "turn_id")
        owner = _require_text(owner, "owner")
        token = _require_text(token, "token")
        now_utc = _coerce_now(now)
        next_expiry = _parse_timestamp(
            admission_lease_expires_at,
            "admission_lease_expires_at",
        )
        if next_expiry <= now_utc:
            raise ValueError("admission_lease_expires_at must be in the future")

        def reduce(
            current: SessionTurnTimeline | None,
            snapshot: DurableSessionSnapshot,
        ) -> _Reduction:
            if current is None:
                return _Reduction("missing", None, "timeline has no admitted turns")
            turn = _find_turn(current, turn_id)
            if turn is None or turn.status != "queued":
                return _Reduction("admission_invalid", current, "turn is not a queued admission")
            if (
                turn.admission_owner != owner
                or turn.admission_token != token
                or turn.admission_lease_expires_at is None
            ):
                return _Reduction(
                    "admission_invalid",
                    current,
                    "queued admission identity is not usable",
                )
            current_expiry = _parse_timestamp(
                turn.admission_lease_expires_at,
                "admission_lease_expires_at",
            )
            if current_expiry <= now_utc:
                return _Reduction("admission_stale", current, "queued admission deadline is stale")
            if next_expiry <= current_expiry:
                return _Reduction(
                    "admission_invalid",
                    current,
                    "renewed admission lease must extend the current expiry",
                )
            renewed_turn = replace(
                turn,
                admission_lease_expires_at=next_expiry.isoformat(),
            )
            turns = tuple(
                renewed_turn if item.turn_id == turn_id else item for item in current.turns
            )
            return _Reduction(
                "admission_lease_renewed",
                self._next_timeline(current, snapshot=snapshot, turns=turns),
            )

        return await self._apply(session_id, expected_history_revision, reduce)

    async def recover_expired_queued_admission(
        self,
        session_id: str,
        *,
        turn_id: str,
        expected_history_revision: str,
        now: datetime | None = None,
    ) -> TimelineMutationResult:
        """Cancel a queued head only after its durable admission deadline expires.

        A queued turn has not crossed a model or tool boundary, so a matching
        expired admission lease proves a known no-effect orphan.  Legacy rows
        without this identity deliberately remain unrecoverable here.
        """
        turn_id = _require_text(turn_id, "turn_id")
        now_utc = _coerce_now(now)

        def reduce(
            current: SessionTurnTimeline | None,
            snapshot: DurableSessionSnapshot,
        ) -> _Reduction:
            if current is None:
                return _Reduction("missing", None, "timeline has no admitted turns")
            if current.lane_turn_id is not None:
                return _Reduction(
                    "admission_invalid",
                    current,
                    "an active lane must recover before a queued admission",
                )
            queued_head = next((turn for turn in current.turns if turn.status == "queued"), None)
            if queued_head is None or queued_head.turn_id != turn_id:
                return _Reduction(
                    "admission_invalid",
                    current,
                    "turn is not the durable queued head",
                )
            if (
                queued_head.admission_owner is None
                or queued_head.admission_token is None
                or queued_head.admission_lease_expires_at is None
            ):
                return _Reduction(
                    "admission_invalid",
                    current,
                    "queued turn has no durable admission ownership or deadline",
                )
            expiry = _parse_timestamp(
                queued_head.admission_lease_expires_at,
                "admission_lease_expires_at",
            )
            if expiry > now_utc:
                return _Reduction(
                    "admission_invalid",
                    current,
                    "queued admission deadline is still live",
                )
            terminal_turn = replace(
                queued_head,
                status="terminal",
                terminal_outcome="cancelled",
                cancellation_outcome="cancelled_queued",
                admission_owner=None,
                admission_token=None,
                admission_lease_expires_at=None,
                recovery_reason="admission_owner_expired_before_claim",
            )
            turns = tuple(
                terminal_turn if item.turn_id == turn_id else item for item in current.turns
            )
            return _Reduction(
                "recovery_admission_cancelled",
                self._next_timeline(current, snapshot=snapshot, turns=turns),
            )

        return await self._apply(session_id, expected_history_revision, reduce)

    async def record_operation_precommit(
        self,
        session_id: str,
        *,
        turn_id: str,
        expected_history_revision: str,
        owner: str,
        token: str,
        descriptor: OperationDescriptor,
        now: datetime | None = None,
    ) -> TimelineMutationResult:
        """Durably bind one exact operation before any side effect can begin.

        The descriptor intentionally stores only the canonical argument digest,
        never the raw arguments.  Repeating the same descriptor is idempotent;
        reusing an operation ID or call ID with different evidence fails closed.
        """
        turn_id = _require_text(turn_id, "turn_id")
        owner = _require_text(owner, "owner")
        token = _require_text(token, "token")
        if not isinstance(descriptor, OperationDescriptor):
            raise TypeError("descriptor must be an OperationDescriptor")
        if (
            descriptor.status != "precommitted"
            or descriptor.precommit_boundary != "not_started"
        ):
            raise ValueError("new operation descriptor must be precommitted at not_started")
        now_utc = _coerce_now(now)

        def reduce(
            current: SessionTurnTimeline | None,
            snapshot: DurableSessionSnapshot,
        ) -> _Reduction:
            lease_status = _lease_status(current, turn_id, owner, token, now_utc)
            if lease_status is not None:
                return _Reduction(lease_status, current, "lane identity is not usable")
            assert current is not None
            turn = _find_turn(current, turn_id)
            assert turn is not None
            if any(
                item.side_effect_boundary == "unknown"
                or item.terminal_outcome == "unknown"
                or any(descriptor.status == "unknown" for descriptor in item.operation_descriptors)
                for item in current.turns
            ):
                return _Reduction(
                    "invalid",
                    current,
                    "an unknown prior operation quarantines new session side effects",
                )
            if any(
                item.sequence < turn.sequence
                and item.status == "terminal"
                and item.terminal_outcome == "blocked"
                and any(
                    prior.status in {"precommitted", "started"}
                    for prior in item.operation_descriptors
                )
                for item in current.turns
            ):
                return _Reduction(
                    "invalid",
                    current,
                    "a pending ordinary-Chat continuation blocks later session side effects",
                )
            if turn.side_effect_boundary == "unknown":
                return _Reduction(
                    "invalid",
                    current,
                    "unknown side-effect state blocks new operation precommits",
                )
            local_by_operation = {
                item.operation_id: item for item in turn.operation_descriptors
            }
            local_by_call = {item.call_id: item for item in turn.operation_descriptors}
            prior = local_by_operation.get(descriptor.operation_id) or local_by_call.get(
                descriptor.call_id
            )
            if prior is not None:
                if prior == descriptor:
                    return _Reduction(
                        "precommitted",
                        current,
                        "exact operation descriptor is already precommitted",
                    )
                return _Reduction(
                    "invalid",
                    current,
                    "operation or call identity conflicts with existing descriptor",
                )
            if any(
                item.status not in {"succeeded", "failed", "abandoned"}
                for item in turn.operation_descriptors
            ):
                return _Reduction(
                    "invalid",
                    current,
                    "all earlier operations must have a known result before the next precommit",
                )
            all_operation_ids = {
                operation_id
                for item in current.turns
                for operation_id in (
                    *(entry.operation_id for entry in item.operation_descriptors),
                    *item.legacy_operation_ids,
                )
            }
            all_call_ids = {
                entry.call_id
                for item in current.turns
                for entry in item.operation_descriptors
            }
            if descriptor.operation_id in all_operation_ids or descriptor.call_id in all_call_ids:
                return _Reduction(
                    "invalid",
                    current,
                    "operation_id or call_id is already bound by another turn",
                )
            updated_turn = replace(
                turn,
                operation_descriptors=(*turn.operation_descriptors, descriptor),
            )
            turns = tuple(
                updated_turn if item.turn_id == turn_id else item for item in current.turns
            )
            return _Reduction(
                "precommitted",
                self._next_timeline(current, snapshot=snapshot, turns=turns),
            )

        return await self._apply(session_id, expected_history_revision, reduce)

    async def record_operation_result(
        self,
        session_id: str,
        *,
        turn_id: str,
        expected_history_revision: str,
        owner: str,
        token: str,
        operation_id: str,
        status: Literal["succeeded", "failed", "unknown"],
        result_digest: str | None = None,
        receipt_reference: str | None = None,
        companion_events: Sequence[Mapping[str, Any]] = (),
        now: datetime | None = None,
    ) -> TimelineMutationResult:
        """Record the known terminal result for the currently started operation."""
        turn_id = _require_text(turn_id, "turn_id")
        owner = _require_text(owner, "owner")
        token = _require_text(token, "token")
        operation_id = _require_text(operation_id, "operation_id")
        if status not in {"succeeded", "failed", "unknown"}:
            raise ValueError(f"unsupported operation result status: {status!r}")
        candidate_digest = _optional_namespaced_digest(result_digest, "result_digest")
        candidate_receipt = _optional_text(receipt_reference, "receipt_reference")
        if status == "unknown" and (candidate_digest is not None or candidate_receipt is not None):
            raise ValueError("unknown operation result cannot claim result evidence")
        if status in {"succeeded", "failed"} and (
            candidate_digest is None and candidate_receipt is None
        ):
            raise ValueError(
                "known operation result requires result_digest or receipt_reference"
            )
        now_utc = _coerce_now(now)

        def reduce(
            current: SessionTurnTimeline | None,
            snapshot: DurableSessionSnapshot,
        ) -> _Reduction:
            lease_status = _lease_status(current, turn_id, owner, token, now_utc)
            if lease_status is not None:
                return _Reduction(lease_status, current, "lane identity is not usable")
            assert current is not None
            turn = _find_turn(current, turn_id)
            assert turn is not None
            descriptor = next(
                (
                    item
                    for item in turn.operation_descriptors
                    if item.operation_id == operation_id
                ),
                None,
            )
            if descriptor is None:
                return _Reduction("invalid", current, "operation_id has no precommitted descriptor")
            replacement = replace(
                descriptor,
                status=status,
                precommit_boundary="unknown" if status == "unknown" else "started",
                result_digest=candidate_digest,
                receipt_reference=candidate_receipt,
            )
            if descriptor == replacement:
                return _Reduction(
                    "operation_result",
                    current,
                    "exact operation result is already recorded",
                )
            if descriptor.status != "started" or descriptor.precommit_boundary != "started":
                return _Reduction(
                    "invalid",
                    current,
                    "only a started operation may record a terminal result",
                )
            if turn.side_effect_boundary == "unknown":
                return _Reduction("invalid", current, "unknown side-effect state cannot be reconciled here")
            descriptors = tuple(
                replacement if item.operation_id == operation_id else item
                for item in turn.operation_descriptors
            )
            updated_turn = replace(
                turn,
                side_effect_boundary=(
                    "unknown" if status == "unknown" else turn.side_effect_boundary
                ),
                operation_descriptors=descriptors,
            )
            turns = tuple(
                updated_turn if item.turn_id == turn_id else item for item in current.turns
            )
            return _Reduction(
                "operation_result",
                self._next_timeline(current, snapshot=snapshot, turns=turns),
            )

        return await self._apply(
            session_id,
            expected_history_revision,
            reduce,
            companion_events=companion_events,
        )

    async def record_operation_approval_pending(
        self,
        session_id: str,
        *,
        turn_id: str,
        expected_history_revision: str,
        owner: str,
        token: str,
        operation_id: str,
        companion_events: Sequence[Mapping[str, Any]],
        now: datetime | None = None,
    ) -> TimelineMutationResult:
        """Persist a durable approval interrupt without consuming the operation.

        The descriptor remains ``precommitted/not_started``.  This event is a
        replayable proof that a separate durable approval record owns the
        interruption, rather than an execution result for the operation.
        """
        turn_id = _require_text(turn_id, "turn_id")
        owner = _require_text(owner, "owner")
        token = _require_text(token, "token")
        operation_id = _require_text(operation_id, "operation_id")
        now_utc = _coerce_now(now)

        def reduce(
            current: SessionTurnTimeline | None,
            snapshot: DurableSessionSnapshot,
        ) -> _Reduction:
            lease_status = _lease_status(current, turn_id, owner, token, now_utc)
            if lease_status is not None:
                return _Reduction(lease_status, current, "lane identity is not usable")
            assert current is not None
            turn = _find_turn(current, turn_id)
            assert turn is not None
            descriptor = next(
                (item for item in turn.operation_descriptors if item.operation_id == operation_id),
                None,
            )
            if descriptor is None:
                return _Reduction("invalid", current, "operation_id has no precommitted descriptor")
            if descriptor.status != "precommitted" or descriptor.precommit_boundary != "not_started":
                return _Reduction(
                    "invalid",
                    current,
                    "only a precommitted operation may enter approval pending",
                )
            # Advance the timeline even though the descriptor stays unchanged,
            # so the companion event and the latest history position commit in
            # one strict session mutation before it reaches callbacks.
            return _Reduction(
                "precommitted",
                self._next_timeline(current, snapshot=snapshot, turns=current.turns),
            )

        return await self._apply(
            session_id,
            expected_history_revision,
            reduce,
            companion_events=companion_events,
        )

    async def abandon_operation_pre_effect(
        self,
        session_id: str,
        *,
        turn_id: str,
        expected_history_revision: str,
        owner: str,
        token: str,
        operation_id: str,
        result_digest: str,
        receipt_reference: str,
        companion_events: Sequence[Mapping[str, Any]] = (),
        now: datetime | None = None,
    ) -> TimelineMutationResult:
        """Record a known no-effect result for an exact precommitted operation."""
        turn_id = _require_text(turn_id, "turn_id")
        owner = _require_text(owner, "owner")
        token = _require_text(token, "token")
        operation_id = _require_text(operation_id, "operation_id")
        candidate_digest = _require_namespaced_sha256_digest(result_digest, "result_digest")
        candidate_receipt = _require_text(receipt_reference, "receipt_reference")
        now_utc = _coerce_now(now)

        def reduce(
            current: SessionTurnTimeline | None,
            snapshot: DurableSessionSnapshot,
        ) -> _Reduction:
            lease_status = _lease_status(current, turn_id, owner, token, now_utc)
            if lease_status is not None:
                return _Reduction(lease_status, current, "lane identity is not usable")
            assert current is not None
            turn = _find_turn(current, turn_id)
            assert turn is not None
            descriptor = next(
                (item for item in turn.operation_descriptors if item.operation_id == operation_id),
                None,
            )
            if descriptor is None:
                return _Reduction("invalid", current, "operation_id has no precommitted descriptor")
            replacement = replace(
                descriptor,
                status="abandoned",
                precommit_boundary="not_started",
                result_digest=candidate_digest,
                receipt_reference=candidate_receipt,
            )
            if descriptor == replacement:
                return _Reduction(
                    "operation_abandoned",
                    current,
                    "exact pre-effect abandonment is already recorded",
                )
            if descriptor.status != "precommitted" or descriptor.precommit_boundary != "not_started":
                return _Reduction(
                    "invalid",
                    current,
                    "only a precommitted operation may be abandoned before an effect",
                )
            descriptors = tuple(
                replacement if item.operation_id == operation_id else item
                for item in turn.operation_descriptors
            )
            turns = tuple(
                replace(turn, operation_descriptors=descriptors)
                if turn.turn_id == turn_id
                else turn
                for turn in current.turns
            )
            return _Reduction(
                "operation_abandoned",
                self._next_timeline(current, snapshot=snapshot, turns=turns),
            )

        return await self._apply(
            session_id,
            expected_history_revision,
            reduce,
            companion_events=companion_events,
        )

    async def mark_terminal_precommitted_operation_started(
        self,
        session_id: str,
        *,
        turn_id: str,
        expected_history_revision: str,
        operation_id: str,
        call_id: str,
        arguments_digest: str,
    ) -> TimelineMutationResult:
        """Cross the boundary for an approved operation after its Chat turn paused.

        The original lane has already been terminally blocked and released, so
        this transition intentionally has no lane owner. The approval service
        may use it only after its exact call/operation binding has been
        validated and consume-once claimed.
        """
        turn_id = _require_text(turn_id, "turn_id")
        operation_id = _require_text(operation_id, "operation_id")
        call_id = _require_text(call_id, "call_id")
        arguments_digest = _require_bare_sha256_digest(arguments_digest, "arguments_digest")

        def reduce(
            current: SessionTurnTimeline | None,
            snapshot: DurableSessionSnapshot,
        ) -> _Reduction:
            if current is None:
                return _Reduction("missing", None, "timeline has no admitted turns")
            turn = _find_turn(current, turn_id)
            if (
                turn is None
                or turn.status != "terminal"
                or turn.terminal_outcome != "blocked"
            ):
                return _Reduction(
                    "invalid",
                    current,
                    "approved continuation does not own a blocked terminal turn",
                )
            descriptor = next(
                (item for item in turn.operation_descriptors if item.operation_id == operation_id),
                None,
            )
            if descriptor is None:
                return _Reduction("invalid", current, "operation_id has no precommitted descriptor")
            if descriptor.call_id != call_id or descriptor.arguments_digest != arguments_digest:
                return _Reduction(
                    "invalid",
                    current,
                    "approved continuation call identity does not match its descriptor",
                )
            replacement = replace(
                descriptor,
                status="started",
                precommit_boundary="started",
            )
            if descriptor == replacement:
                return _Reduction(
                    "already_started",
                    current,
                    "approved operation effect boundary was already claimed",
                )
            if descriptor.status != "precommitted" or descriptor.precommit_boundary != "not_started":
                return _Reduction(
                    "invalid",
                    current,
                    "only a precommitted blocked operation may begin approval continuation",
                )
            descriptors = tuple(
                replacement if item.operation_id == operation_id else item
                for item in turn.operation_descriptors
            )
            turns = tuple(
                replace(turn, operation_descriptors=descriptors)
                if item.turn_id == turn_id
                else item
                for item in current.turns
            )
            return _Reduction(
                "boundary_updated",
                self._next_timeline(current, snapshot=snapshot, turns=turns),
            )

        return await self._apply(session_id, expected_history_revision, reduce)

    async def record_terminal_continuation_result(
        self,
        session_id: str,
        *,
        turn_id: str,
        expected_history_revision: str,
        operation_id: str,
        call_id: str,
        arguments_digest: str,
        status: Literal["succeeded", "failed", "unknown"],
        result_digest: str | None = None,
        receipt_reference: str | None = None,
    ) -> TimelineMutationResult:
        """Record a post-approval result before any ReAct continuation callback."""
        turn_id = _require_text(turn_id, "turn_id")
        operation_id = _require_text(operation_id, "operation_id")
        call_id = _require_text(call_id, "call_id")
        arguments_digest = _require_bare_sha256_digest(arguments_digest, "arguments_digest")
        if status not in {"succeeded", "failed", "unknown"}:
            raise ValueError(f"unsupported operation result status: {status!r}")
        candidate_digest = _optional_namespaced_digest(result_digest, "result_digest")
        candidate_receipt = _optional_text(receipt_reference, "receipt_reference")
        if status == "unknown" and (candidate_digest is not None or candidate_receipt is not None):
            raise ValueError("unknown operation result cannot claim result evidence")
        if status in {"succeeded", "failed"} and (
            candidate_digest is None and candidate_receipt is None
        ):
            raise ValueError("known operation result requires result_digest or receipt_reference")

        def reduce(
            current: SessionTurnTimeline | None,
            snapshot: DurableSessionSnapshot,
        ) -> _Reduction:
            if current is None:
                return _Reduction("missing", None, "timeline has no admitted turns")
            turn = _find_turn(current, turn_id)
            if (
                turn is None
                or turn.status != "terminal"
                or turn.terminal_outcome != "blocked"
            ):
                return _Reduction(
                    "invalid",
                    current,
                    "approved continuation does not own a blocked terminal turn",
                )
            descriptor = next(
                (item for item in turn.operation_descriptors if item.operation_id == operation_id),
                None,
            )
            if descriptor is None:
                return _Reduction("invalid", current, "operation_id has no precommitted descriptor")
            if descriptor.call_id != call_id or descriptor.arguments_digest != arguments_digest:
                return _Reduction(
                    "invalid",
                    current,
                    "approved continuation call identity does not match its descriptor",
                )
            replacement = replace(
                descriptor,
                status=status,
                precommit_boundary="unknown" if status == "unknown" else "started",
                result_digest=candidate_digest,
                receipt_reference=candidate_receipt,
            )
            if descriptor == replacement:
                return _Reduction("operation_result", current, "exact continuation result is already recorded")
            if descriptor.status != "started" or descriptor.precommit_boundary != "started":
                return _Reduction(
                    "invalid",
                    current,
                    "only a started approval continuation may record a result",
                )
            descriptors = tuple(
                replacement if item.operation_id == operation_id else item
                for item in turn.operation_descriptors
            )
            turns = tuple(
                replace(
                    turn,
                    side_effect_boundary=(
                        "unknown" if status == "unknown" else turn.side_effect_boundary
                    ),
                    operation_descriptors=descriptors,
                )
                if item.turn_id == turn_id
                else item
                for item in current.turns
            )
            return _Reduction(
                "operation_result",
                self._next_timeline(current, snapshot=snapshot, turns=turns),
            )

        return await self._apply(session_id, expected_history_revision, reduce)

    async def abandon_terminal_precommitted_operation(
        self,
        session_id: str,
        *,
        turn_id: str,
        expected_history_revision: str,
        operation_id: str,
        call_id: str,
        arguments_digest: str,
        result_digest: str,
        receipt_reference: str,
    ) -> TimelineMutationResult:
        """Release a rejected, expired, drifted, or pre-effect failed approval."""
        turn_id = _require_text(turn_id, "turn_id")
        operation_id = _require_text(operation_id, "operation_id")
        call_id = _require_text(call_id, "call_id")
        arguments_digest = _require_bare_sha256_digest(arguments_digest, "arguments_digest")
        candidate_digest = _require_namespaced_sha256_digest(result_digest, "result_digest")
        candidate_receipt = _require_text(receipt_reference, "receipt_reference")

        def reduce(
            current: SessionTurnTimeline | None,
            snapshot: DurableSessionSnapshot,
        ) -> _Reduction:
            if current is None:
                return _Reduction("missing", None, "timeline has no admitted turns")
            turn = _find_turn(current, turn_id)
            if (
                turn is None
                or turn.status != "terminal"
                or turn.terminal_outcome != "blocked"
            ):
                return _Reduction(
                    "invalid",
                    current,
                    "approval abandonment does not own a blocked terminal turn",
                )
            descriptor = next(
                (item for item in turn.operation_descriptors if item.operation_id == operation_id),
                None,
            )
            if descriptor is None:
                return _Reduction("invalid", current, "operation_id has no precommitted descriptor")
            if descriptor.call_id != call_id or descriptor.arguments_digest != arguments_digest:
                return _Reduction(
                    "invalid",
                    current,
                    "approved continuation call identity does not match its descriptor",
                )
            replacement = replace(
                descriptor,
                status="abandoned",
                precommit_boundary="not_started",
                result_digest=candidate_digest,
                receipt_reference=candidate_receipt,
            )
            if descriptor == replacement:
                return _Reduction(
                    "operation_abandoned",
                    current,
                    "exact approval abandonment is already recorded",
                )
            if descriptor.status != "precommitted" or descriptor.precommit_boundary != "not_started":
                return _Reduction(
                    "invalid",
                    current,
                    "only a precommitted blocked operation may be abandoned",
                )
            descriptors = tuple(
                replacement if item.operation_id == operation_id else item
                for item in turn.operation_descriptors
            )
            turns = tuple(
                replace(turn, operation_descriptors=descriptors)
                if item.turn_id == turn_id
                else item
                for item in current.turns
            )
            return _Reduction(
                "operation_abandoned",
                self._next_timeline(current, snapshot=snapshot, turns=turns),
            )

        return await self._apply(session_id, expected_history_revision, reduce)

    async def recover_stale_unstarted_turn(
        self,
        session_id: str,
        *,
        turn_id: str,
        expected_history_revision: str,
        now: datetime | None = None,
    ) -> TimelineMutationResult:
        """Release a stale lane only when durable state proves no effect began.

        The abandoned turn is terminally cancelled instead of replayed. A
        caller may admit a fresh turn explicitly, while the original operation
        and call identities remain durable and non-reusable.
        """
        turn_id = _require_text(turn_id, "turn_id")
        now_utc = _coerce_now(now)

        def reduce(
            current: SessionTurnTimeline | None,
            snapshot: DurableSessionSnapshot,
        ) -> _Reduction:
            if current is None or current.lane_turn_id != turn_id:
                return _Reduction("lease_invalid", current, "turn does not own the active lane")
            assert current.lane_lease_expires_at is not None
            if _parse_timestamp(current.lane_lease_expires_at, "lane_lease_expires_at") > now_utc:
                return _Reduction("lease_invalid", current, "lane lease is not stale")
            turn = _find_turn(current, turn_id)
            if turn is None or turn.status != "running":
                return _Reduction("lease_invalid", current, "stale lane does not own a running turn")
            if turn.side_effect_boundary != "not_started" or any(
                item.status != "precommitted" or item.precommit_boundary != "not_started"
                for item in turn.operation_descriptors
            ):
                return _Reduction(
                    "invalid",
                    current,
                    "unstarted recovery cannot conceal a started or unknown operation",
                )
            pending_operation_ids = _pending_approval_operation_ids(snapshot, turn_id)
            if any(
                item.operation_id in pending_operation_ids
                for item in turn.operation_descriptors
            ):
                updated_turn = replace(
                    turn,
                    status="terminal",
                    terminal_outcome="blocked",
                    cancellation_outcome=None,
                )
                turns = tuple(
                    updated_turn if item.turn_id == turn_id else item for item in current.turns
                )
                return _Reduction(
                    "recovery_pending_approval",
                    self._next_timeline(
                        current,
                        snapshot=snapshot,
                        turns=turns,
                        lane_turn_id=None,
                        lane_owner=None,
                        lane_token=None,
                        lane_lease_expires_at=None,
                    ),
                )
            updated_turn = replace(
                turn,
                status="terminal",
                terminal_outcome="cancelled",
                cancellation_outcome="cancelled_running",
            )
            turns = tuple(
                updated_turn if item.turn_id == turn_id else item for item in current.turns
            )
            return _Reduction(
                "recovery_cancelled",
                self._next_timeline(
                    current,
                    snapshot=snapshot,
                    turns=turns,
                    lane_turn_id=None,
                    lane_owner=None,
                    lane_token=None,
                    lane_lease_expires_at=None,
                ),
            )

        return await self._apply(session_id, expected_history_revision, reduce)

    async def recover_stale_started_operation(
        self,
        session_id: str,
        *,
        turn_id: str,
        expected_history_revision: str,
        now: datetime | None = None,
    ) -> TimelineMutationResult:
        """Terminally quarantine a started operation after its lane lease expires.

        This is a recovery transition, not a lease takeover: it turns started
        evidence into ``unknown`` and terminalizes the affected turn.  The
        original turn can therefore never be reclaimed or replayed.
        """
        turn_id = _require_text(turn_id, "turn_id")
        now_utc = _coerce_now(now)

        def reduce(
            current: SessionTurnTimeline | None,
            snapshot: DurableSessionSnapshot,
        ) -> _Reduction:
            if current is None or current.lane_turn_id != turn_id:
                return _Reduction("lease_invalid", current, "turn does not own the active lane")
            assert current.lane_lease_expires_at is not None
            if _parse_timestamp(current.lane_lease_expires_at, "lane_lease_expires_at") > now_utc:
                return _Reduction("lease_invalid", current, "lane lease is not stale")
            turn = _find_turn(current, turn_id)
            if turn is None or turn.status != "running":
                return _Reduction("lease_invalid", current, "stale lane does not own a running turn")
            started = [
                item for item in turn.operation_descriptors if item.status == "started"
            ]
            if not started:
                return _Reduction(
                    "invalid",
                    current,
                    "stale recovery only quarantines a started operation",
                )
            descriptors = tuple(
                replace(item, status="unknown", precommit_boundary="unknown")
                if item.status == "started"
                else item
                for item in turn.operation_descriptors
            )
            updated_turn = replace(
                turn,
                status="terminal",
                side_effect_boundary="unknown",
                operation_descriptors=descriptors,
                terminal_outcome="unknown",
                cancellation_outcome=None,
            )
            turns = tuple(
                updated_turn if item.turn_id == turn_id else item for item in current.turns
            )
            return _Reduction(
                "recovery_unknown",
                self._next_timeline(
                    current,
                    snapshot=snapshot,
                    turns=turns,
                    lane_turn_id=None,
                    lane_owner=None,
                    lane_token=None,
                    lane_lease_expires_at=None,
                ),
            )

        return await self._apply(session_id, expected_history_revision, reduce)

    async def mark_side_effect_boundary(
        self,
        session_id: str,
        *,
        turn_id: str,
        expected_history_revision: str,
        owner: str,
        token: str,
        boundary: SideEffectBoundary,
        operation_id: str | None = None,
        now: datetime | None = None,
    ) -> TimelineMutationResult:
        turn_id = _require_text(turn_id, "turn_id")
        owner = _require_text(owner, "owner")
        token = _require_text(token, "token")
        if boundary not in {"not_started", "started", "unknown"}:
            raise ValueError(f"unsupported side-effect boundary: {boundary!r}")
        if boundary in {"started", "unknown"}:
            operation_id = _require_text(operation_id, "operation_id")
        elif operation_id is not None:
            operation_id = _require_text(operation_id, "operation_id")
        now_utc = _coerce_now(now)

        def reduce(
            current: SessionTurnTimeline | None,
            snapshot: DurableSessionSnapshot,
        ) -> _Reduction:
            lease_status = _lease_status(current, turn_id, owner, token, now_utc)
            if lease_status is not None:
                return _Reduction(lease_status, current, "lane identity is not usable")
            assert current is not None
            turn = _find_turn(current, turn_id)
            assert turn is not None  # The matching lane invariant already establishes this.
            if not _boundary_transition_allowed(turn.side_effect_boundary, boundary):
                return _Reduction("invalid", current, "side-effect boundary cannot move backwards")
            if boundary == "not_started":
                if turn.side_effect_boundary != "not_started":
                    return _Reduction("invalid", current, "side-effect boundary cannot move backwards")
                return _Reduction("boundary_updated", current, "side-effect boundary already recorded")
            assert operation_id is not None
            descriptor = next(
                (
                    item
                    for item in turn.operation_descriptors
                    if item.operation_id == operation_id
                ),
                None,
            )
            if descriptor is None:
                return _Reduction(
                    "invalid",
                    current,
                    "side-effect boundary requires an exact precommitted operation",
                )
            if boundary == "started":
                if turn.side_effect_boundary == "unknown":
                    return _Reduction("invalid", current, "unknown side-effect state cannot start work")
                if (
                    descriptor.status == "started"
                    and descriptor.precommit_boundary == "started"
                ):
                    return _Reduction(
                        "boundary_updated",
                        current,
                        "side-effect boundary already recorded",
                    )
                if (
                    descriptor.status != "precommitted"
                    or descriptor.precommit_boundary != "not_started"
                ):
                    return _Reduction(
                        "invalid",
                        current,
                        "operation descriptor is not eligible to start",
                    )
                if any(
                    item.operation_id != operation_id
                    and item.status not in {"succeeded", "failed", "abandoned"}
                    for item in turn.operation_descriptors
                ):
                    return _Reduction(
                        "invalid",
                        current,
                        "an earlier operation is still unresolved",
                    )
                next_status = "started"
                next_turn_boundary = "started"
            else:
                if turn.side_effect_boundary == "unknown":
                    if (
                        descriptor.status == "unknown"
                        and descriptor.precommit_boundary == "unknown"
                    ):
                        return _Reduction(
                            "boundary_updated",
                            current,
                            "unknown boundary is already recorded",
                        )
                    return _Reduction(
                        "invalid",
                        current,
                        "unknown boundary is bound to a different operation",
                    )
                if (
                    descriptor.status != "started"
                    or descriptor.precommit_boundary != "started"
                ):
                    return _Reduction(
                        "invalid",
                        current,
                        "only a started operation may become unknown",
                    )
                next_status = "unknown"
                next_turn_boundary = "unknown"
            updated_descriptors = tuple(
                replace(
                    item,
                    status=next_status,
                    precommit_boundary=boundary,
                )
                if item.operation_id == operation_id
                else item
                for item in turn.operation_descriptors
            )
            turns = tuple(
                replace(
                    turn,
                    side_effect_boundary=next_turn_boundary,
                    operation_descriptors=updated_descriptors,
                )
                if turn.turn_id == turn_id
                else turn
                for turn in current.turns
            )
            return _Reduction(
                "boundary_updated",
                self._next_timeline(current, snapshot=snapshot, turns=turns),
            )

        return await self._apply(session_id, expected_history_revision, reduce)

    async def terminal(
        self,
        session_id: str,
        *,
        turn_id: str,
        expected_history_revision: str,
        owner: str,
        token: str,
        outcome: TerminalOutcome,
        side_effect_boundary: SideEffectBoundary | None = None,
        cancellation_outcome: CancellationOutcome | None = None,
        companion_events: Sequence[Mapping[str, Any]] = (),
        now: datetime | None = None,
    ) -> TimelineMutationResult:
        turn_id = _require_text(turn_id, "turn_id")
        owner = _require_text(owner, "owner")
        token = _require_text(token, "token")
        if outcome not in {"completed", "cancelled", "blocked", "unknown"}:
            raise ValueError(f"unsupported terminal outcome: {outcome!r}")
        if cancellation_outcome not in {None, "cancelled_queued", "cancelled_running"}:
            raise ValueError(f"unsupported cancellation outcome: {cancellation_outcome!r}")
        if (outcome == "cancelled") != (cancellation_outcome is not None):
            raise ValueError("cancelled terminal outcome requires matching cancellation_outcome")
        if cancellation_outcome == "cancelled_queued":
            raise ValueError("queued cancellation must use cancel(), not terminal()")
        if side_effect_boundary is not None and side_effect_boundary not in {
            "not_started",
            "started",
            "unknown",
        }:
            raise ValueError(f"unsupported side-effect boundary: {side_effect_boundary!r}")
        now_utc = _coerce_now(now)

        def reduce(
            current: SessionTurnTimeline | None,
            snapshot: DurableSessionSnapshot,
        ) -> _Reduction:
            lease_status = _lease_status(current, turn_id, owner, token, now_utc)
            if lease_status is not None:
                return _Reduction(lease_status, current, "lane identity is not usable")
            assert current is not None
            turn = _find_turn(current, turn_id)
            assert turn is not None
            if (
                side_effect_boundary is not None
                and side_effect_boundary != turn.side_effect_boundary
            ):
                return _Reduction(
                    "invalid",
                    current,
                    "side-effect boundary must be recorded through an exact operation precommit",
                )
            operation_statuses = {item.status for item in turn.operation_descriptors}
            unresolved = operation_statuses.intersection({"precommitted", "started"})
            has_unknown = "unknown" in operation_statuses
            if outcome == "completed" and (unresolved or has_unknown):
                return _Reduction(
                    "invalid",
                    current,
                    "completed turn cannot contain precommitted, started, or unknown operations",
                )
            if outcome == "blocked" and (
                operation_statuses.intersection({"started", "unknown"})
                or has_unknown
            ):
                return _Reduction(
                    "invalid",
                    current,
                    "blocked turn cannot conceal started or unknown operations",
                )
            if outcome == "unknown" and (unresolved or not has_unknown):
                return _Reduction(
                    "invalid",
                    current,
                    "unknown terminal outcome requires a recorded unknown operation",
                )
            if outcome == "cancelled" and operation_statuses.intersection({"started", "unknown"}):
                return _Reduction(
                    "invalid",
                    current,
                    "cancelled turn cannot conceal started or unknown operations",
                )
            terminal_turn = replace(
                turn,
                status="terminal",
                terminal_outcome=outcome,
                cancellation_outcome=cancellation_outcome,
            )
            turns = tuple(
                terminal_turn if item.turn_id == turn_id else item for item in current.turns
            )
            # A terminal transition is the only lane release in v3.  Releasing
            # an unfinished running turn would permit a concurrent replay.
            return _Reduction(
                "terminal",
                self._next_timeline(
                    current,
                    snapshot=snapshot,
                    turns=turns,
                    lane_turn_id=None,
                    lane_owner=None,
                    lane_token=None,
                    lane_lease_expires_at=None,
                ),
            )

        return await self._apply(
            session_id,
            expected_history_revision,
            reduce,
            companion_events=companion_events,
        )

    async def cancel(
        self,
        session_id: str,
        *,
        turn_id: str,
        expected_history_revision: str,
        owner: str | None = None,
        token: str | None = None,
        now: datetime | None = None,
    ) -> TimelineMutationResult:
        turn_id = _require_text(turn_id, "turn_id")
        if (owner is None) != (token is None):
            raise ValueError("owner and token must be supplied together")
        if owner is not None:
            owner = _require_text(owner, "owner")
            token = _require_text(token, "token")
        now_utc = _coerce_now(now)

        def reduce(
            current: SessionTurnTimeline | None,
            snapshot: DurableSessionSnapshot,
        ) -> _Reduction:
            if current is None:
                return _Reduction("missing", None, "timeline has no admitted turns")
            turn = _find_turn(current, turn_id)
            if turn is None:
                return _Reduction("missing", current, "turn_id is not admitted")
            if turn.status == "terminal":
                return _Reduction("already_terminal", current, "turn is already terminal")
            if turn.status == "queued":
                terminal_turn = replace(
                    turn,
                    status="terminal",
                    terminal_outcome="cancelled",
                    cancellation_outcome="cancelled_queued",
                    admission_owner=None,
                    admission_token=None,
                    admission_lease_expires_at=None,
                )
                turns = tuple(
                    terminal_turn if item.turn_id == turn_id else item for item in current.turns
                )
                return _Reduction(
                    "terminal",
                    self._next_timeline(current, snapshot=snapshot, turns=turns),
                )
            if owner is None or token is None:
                return _Reduction("lease_invalid", current, "running cancellation requires lane identity")
            lease_status = _lease_status(current, turn_id, owner, token, now_utc)
            if lease_status is not None:
                return _Reduction(lease_status, current, "lane identity is not usable")
            if any(
                item.status in {"started", "unknown"}
                for item in turn.operation_descriptors
            ):
                return _Reduction(
                    "invalid",
                    current,
                    "running cancellation cannot conceal started or unknown operations",
                )
            terminal_turn = replace(
                turn,
                status="terminal",
                terminal_outcome="cancelled",
                cancellation_outcome="cancelled_running",
            )
            turns = tuple(
                terminal_turn if item.turn_id == turn_id else item for item in current.turns
            )
            return _Reduction(
                "terminal",
                self._next_timeline(
                    current,
                    snapshot=snapshot,
                    turns=turns,
                    lane_turn_id=None,
                    lane_owner=None,
                    lane_token=None,
                    lane_lease_expires_at=None,
                ),
            )

        return await self._apply(session_id, expected_history_revision, reduce)

    async def _apply(
        self,
        session_id: str,
        expected_history_revision: str,
        reduce: Callable[[SessionTurnTimeline | None, DurableSessionSnapshot], _Reduction],
        *,
        companion_events: Sequence[Mapping[str, Any]] = (),
    ) -> TimelineMutationResult:
        outcome: _Reduction | None = None
        normalized_companion_events = _normalize_companion_events(
            session_id,
            companion_events,
        )

        def build_events(snapshot: DurableSessionSnapshot) -> tuple[Mapping[str, Any], ...] | None:
            nonlocal outcome
            loaded = self._load_from_snapshot(session_id, snapshot)
            if loaded.status in {"invalid", "unsupported_version"}:
                outcome = _Reduction(loaded.status, None, loaded.message)
                return None
            outcome = reduce(loaded.timeline, snapshot)
            if outcome.timeline == loaded.timeline:
                return None
            if outcome.timeline is None or outcome.status not in {
                "admitted",
                "claimed",
                "precommitted",
                "operation_abandoned",
                "operation_result",
                "admission_lease_renewed",
                "lease_renewed",
                "recovery_cancelled",
                "recovery_admission_cancelled",
                "recovery_pending_approval",
                "recovery_unknown",
                "terminal",
                "boundary_updated",
            }:
                return None
            expected_timeline_position = (
                snapshot.event_count + len(normalized_companion_events) + 1
            )
            if outcome.timeline.history_current_revision != expected_timeline_position:
                outcome = replace(
                    outcome,
                    timeline=replace(
                        outcome.timeline,
                        history_current_revision=expected_timeline_position,
                    ),
                )
            return (
                *normalized_companion_events,
                self._event(session_id, outcome.timeline),
            )

        try:
            result = await self._session_store.mutate_strict_snapshot(
                session_id,
                expected_history_revision=expected_history_revision,
                build_events=build_events,
            )
        except StrictSessionSnapshotError as exc:
            return TimelineMutationResult("invalid", None, None, str(exc))
        if result.status == "rebase_required":
            return TimelineMutationResult(
                "rebase_required",
                None,
                result.before.history_revision,
                "durable session history changed before transition",
            )
        if outcome is None:  # pragma: no cover - defensive store callback invariant.
            return TimelineMutationResult("invalid", None, result.before.history_revision)
        history_revision = (
            result.after.history_revision if result.status == "appended" else result.before.history_revision
        )
        return TimelineMutationResult(
            outcome.status,
            outcome.timeline,
            history_revision,
            outcome.message,
        )

    @staticmethod
    def _event(session_id: str, timeline: SessionTurnTimeline) -> dict[str, Any]:
        return {
            "type": "session_meta",
            "event": SESSION_TURN_TIMELINE_EVENT,
            "schema_version": SESSION_TURN_TIMELINE_EVENT_VERSION,
            "session_id": session_id,
            "timeline": timeline.to_dict(),
            "timestamp": datetime.now(tz=UTC).isoformat(),
        }

    @staticmethod
    def _next_timeline(
        current: SessionTurnTimeline,
        *,
        snapshot: DurableSessionSnapshot,
        turns: tuple[TimelineTurn, ...],
        lane_turn_id: str | None | object = ...,
        lane_owner: str | None | object = ...,
        lane_token: str | None | object = ...,
        lane_lease_expires_at: str | None | object = ...,
    ) -> SessionTurnTimeline:
        values: dict[str, Any] = {
            "turns": turns,
            "history_current_revision": snapshot.event_count + 1,
        }
        for field_name, value in (
            ("lane_turn_id", lane_turn_id),
            ("lane_owner", lane_owner),
            ("lane_token", lane_token),
            ("lane_lease_expires_at", lane_lease_expires_at),
        ):
            if value is not ...:
                values[field_name] = value
        return replace(current, **values)

    @staticmethod
    def _load_from_snapshot(
        session_id: str,
        snapshot: DurableSessionSnapshot,
    ) -> TimelineLoadResult:
        latest: SessionTurnTimeline | None = None
        for event_index, event in enumerate(snapshot.events):
            if event.get("event") != SESSION_TURN_TIMELINE_EVENT:
                continue
            try:
                parsed = _parse_event(session_id, event, event_index)
            except TimelineUnsupportedVersionError as exc:
                return TimelineLoadResult(
                    "unsupported_version",
                    None,
                    snapshot.history_revision,
                    snapshot.event_count,
                    str(exc),
                )
            except TimelinePayloadError as exc:
                return TimelineLoadResult(
                    "invalid",
                    None,
                    snapshot.history_revision,
                    snapshot.event_count,
                    str(exc),
                )
            latest = parsed
        if latest is None:
            return TimelineLoadResult(
                "missing",
                None,
                snapshot.history_revision,
                snapshot.event_count,
            )
        return TimelineLoadResult(
            "loaded",
            latest,
            snapshot.history_revision,
            snapshot.event_count,
        )


def _parse_event(
    session_id: str,
    event: Mapping[str, Any],
    event_index: int,
) -> SessionTurnTimeline:
    _require_exact_fields(event, _EVENT_FIELDS, "timeline event")
    version = event.get("schema_version")
    if type(version) is not int:
        raise TimelinePayloadError("timeline event schema_version must be an integer")
    if version != SESSION_TURN_TIMELINE_EVENT_VERSION:
        raise TimelineUnsupportedVersionError(
            f"unsupported timeline event schema version: {version!r}"
        )
    if event.get("type") != "session_meta" or event.get("event") != SESSION_TURN_TIMELINE_EVENT:
        raise TimelinePayloadError("timeline event envelope is invalid")
    if event.get("session_id") != session_id:
        raise TimelinePayloadError("timeline event session_id does not match requested session")
    _normalize_timestamp(event.get("timestamp"), "timeline event timestamp")
    payload = event.get("timeline")
    if not isinstance(payload, Mapping):
        raise TimelinePayloadError("timeline event payload must be an object")
    timeline = SessionTurnTimeline.from_dict(payload)
    if timeline.session_id != session_id:
        raise TimelinePayloadError("timeline payload session_id does not match envelope")
    if timeline.history_current_revision != event_index + 1:
        raise TimelinePayloadError("timeline history_current_revision does not match event position")
    return timeline


def _lease_status(
    current: SessionTurnTimeline | None,
    turn_id: str,
    owner: str,
    token: str,
    now: datetime,
) -> Literal["lease_stale", "lease_invalid"] | None:
    if current is None:
        return "lease_invalid"
    if (
        current.lane_turn_id != turn_id
        or current.lane_owner != owner
        or current.lane_token != token
        or current.lane_lease_expires_at is None
    ):
        return "lease_invalid"
    if _parse_timestamp(current.lane_lease_expires_at, "lane_lease_expires_at") <= now:
        return "lease_stale"
    return None


def _find_turn(timeline: SessionTurnTimeline, turn_id: str) -> TimelineTurn | None:
    return next((turn for turn in timeline.turns if turn.turn_id == turn_id), None)


def _boundary_transition_allowed(
    current: SideEffectBoundary,
    target: SideEffectBoundary,
) -> bool:
    return {
        "not_started": {"not_started", "started", "unknown"},
        "started": {"started", "unknown"},
        "unknown": {"unknown"},
    }[current].__contains__(target)


def _normalize_companion_events(
    session_id: str,
    events: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    if isinstance(events, (str, bytes)) or not isinstance(events, Sequence):
        raise TypeError("companion_events must be a sequence of event objects")
    normalized: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        if not isinstance(event, Mapping) or not event:
            raise TypeError(f"companion event {index} must be a non-empty object")
        event_session_id = event.get("session_id")
        if event_session_id != session_id:
            raise TimelinePayloadError(
                f"companion event {index} session_id does not match requested session"
            )
        normalized.append(dict(event))
    return tuple(normalized)


def _pending_approval_operation_ids(
    snapshot: DurableSessionSnapshot,
    turn_id: str,
) -> frozenset[str]:
    """Read only persisted approval interrupts from the strict session snapshot."""
    operation_ids: set[str] = set()
    for event in snapshot.events:
        if event.get("type") != "turn_event" or event.get("turn_id") != turn_id:
            continue
        if event.get("phase") != "tool_call_result":
            continue
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            continue
        metadata = payload.get("metadata")
        if not isinstance(metadata, Mapping) or metadata.get("timeline_approval_pending") is not True:
            continue
        operation_id = metadata.get("timeline_operation_id")
        if isinstance(operation_id, str) and operation_id.strip():
            operation_ids.add(operation_id.strip())
    return frozenset(operation_ids)


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    label: str,
) -> None:
    missing = sorted(expected - set(value))
    unexpected = sorted(set(value) - expected)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing fields: {missing}")
        if unexpected:
            details.append(f"unexpected fields: {unexpected}")
        raise TimelinePayloadError(f"{label} " + "; ".join(details))


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TimelinePayloadError(f"{field_name} must be a non-empty string")
    return value.strip()


def _require_bare_sha256_digest(value: Any, field_name: str) -> str:
    digest = _require_text(value, field_name)
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise TimelinePayloadError(f"{field_name} must be a lowercase sha256 digest")
    return digest


def _require_namespaced_sha256_digest(value: Any, field_name: str) -> str:
    digest = _require_text(value, field_name)
    if re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
        raise TimelinePayloadError(f"{field_name} must be a lowercase sha256 digest")
    return digest


def _legacy_namespaced_arguments_digest(value: Any, field_name: str) -> str:
    digest = _require_namespaced_sha256_digest(value, field_name)
    return digest.removeprefix("sha256:")


def _optional_namespaced_digest(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_namespaced_sha256_digest(value, field_name)


def _optional_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_text(value, field_name)


def _require_non_negative_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise TimelinePayloadError(f"{field_name} must be a non-negative integer")
    return value


def _require_positive_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise TimelinePayloadError(f"{field_name} must be a positive integer")
    return value


def _normalize_identifiers(value: Any, field_name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise TimelinePayloadError(f"{field_name} must be a sequence of strings")
    try:
        normalized = tuple(_require_text(item, field_name) for item in value)
    except TypeError as exc:
        raise TimelinePayloadError(f"{field_name} must be a sequence of strings") from exc
    if len(set(normalized)) != len(normalized):
        raise TimelinePayloadError(f"{field_name} must not contain duplicates")
    return normalized


def _parse_timestamp(value: Any, field_name: str) -> datetime:
    text = _require_text(value, field_name)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TimelinePayloadError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise TimelinePayloadError(f"{field_name} must include a timezone")
    return parsed.astimezone(UTC)


def _normalize_timestamp(value: Any, field_name: str) -> str:
    return _parse_timestamp(value, field_name).isoformat()


def _coerce_now(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(tz=UTC)
    if value.tzinfo is None:
        raise ValueError("now must include a timezone")
    return value.astimezone(UTC)
