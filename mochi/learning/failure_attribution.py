"""Strict user-session attribution for background failure-learning telemetry.

The global outbox remains an internal processing detail.  These additive
events are the only failure-learning records projected into an ordinary user
session.  They contain bounded identifiers and enums only: no prompt, model
output, tool arguments, paths, failure text, or hidden reasoning.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, Protocol, cast


FAILURE_ATTRIBUTION_EVENT = "failure_learning_attribution_recorded"
FAILURE_ATTRIBUTION_SCHEMA_VERSION = 1

FailureAttributionTransition = Literal[
    "candidate",
    "processed",
    "rejected",
    "hint_selected",
]
FailureAttributionStatus = Literal[
    "pending",
    "processed",
    "rejected",
    "selected",
]
FailureAttributionReason = Literal[
    "candidate_enqueued",
    "worker_acked",
    "worker_rejected",
    "verified_hint_injected",
]

_TRANSITION_STATUS = {
    "candidate": "pending",
    "processed": "processed",
    "rejected": "rejected",
    "hint_selected": "selected",
}
_TRANSITION_REASON = {
    "candidate": "candidate_enqueued",
    "processed": "worker_acked",
    "rejected": "worker_rejected",
    "hint_selected": "verified_hint_injected",
}
_EVENT_FIELDS = frozenset(
    {
        "type",
        "event",
        "schema_version",
        "candidate_id",
        "turn_id",
        "transition",
        "status",
        "reason_code",
        "timestamp",
        "idempotency_key",
    }
)
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")


class FailureAttributionError(ValueError):
    """Raised when an attribution record is malformed or conflicts."""


class FailureAttributionSessionStore(Protocol):
    async def append_event_if(
        self,
        session_id: str,
        event: dict[str, Any],
        predicate: Callable[[list[dict[str, Any]]], bool],
    ) -> bool: ...


def _identifier(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise FailureAttributionError(f"{field_name} must be a bounded identifier")
    return value


def _timestamp(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > 64:
        raise FailureAttributionError("timestamp must be a bounded ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise FailureAttributionError(
            "timestamp must be a valid ISO timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise FailureAttributionError("timestamp must include a UTC offset")
    return value


def _require_exact_keys(value: Mapping[str, Any]) -> None:
    actual = frozenset(value)
    if actual != _EVENT_FIELDS:
        raise FailureAttributionError("attribution event must have exact v1 keys")


def attribution_idempotency_key(
    *,
    candidate_id: str,
    transition: FailureAttributionTransition,
    turn_id: str,
) -> str:
    candidate = _identifier(candidate_id, "candidate_id")
    turn = _identifier(turn_id, "turn_id")
    suffix = f":{turn}" if transition == "hint_selected" else ""
    return f"failure-attribution:v1:{candidate}:{transition}{suffix}"


@dataclass(frozen=True)
class FailureAttributionRecord:
    """One redacted failure-learning transition attributed to a user turn."""

    candidate_id: str
    turn_id: str
    transition: FailureAttributionTransition
    status: FailureAttributionStatus
    reason_code: FailureAttributionReason
    timestamp: str
    idempotency_key: str

    def __post_init__(self) -> None:
        candidate_id = _identifier(self.candidate_id, "candidate_id")
        turn_id = _identifier(self.turn_id, "turn_id")
        if self.transition not in _TRANSITION_STATUS:
            raise FailureAttributionError("unsupported attribution transition")
        expected_status = _TRANSITION_STATUS[self.transition]
        expected_reason = _TRANSITION_REASON[self.transition]
        if self.status != expected_status:
            raise FailureAttributionError("status does not match transition")
        if self.reason_code != expected_reason:
            raise FailureAttributionError("reason_code does not match transition")
        timestamp = _timestamp(self.timestamp)
        expected_key = attribution_idempotency_key(
            candidate_id=candidate_id,
            transition=self.transition,
            turn_id=turn_id,
        )
        if self.idempotency_key != expected_key:
            raise FailureAttributionError(
                "idempotency_key does not match attribution identity"
            )
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "turn_id", turn_id)
        object.__setattr__(self, "timestamp", timestamp)

    @classmethod
    def create(
        cls,
        *,
        candidate_id: str,
        turn_id: str,
        transition: FailureAttributionTransition,
        timestamp: str | None = None,
    ) -> "FailureAttributionRecord":
        if transition not in _TRANSITION_STATUS:
            raise FailureAttributionError("unsupported attribution transition")
        return cls(
            candidate_id=candidate_id,
            turn_id=turn_id,
            transition=transition,
            status=cast(FailureAttributionStatus, _TRANSITION_STATUS[transition]),
            reason_code=cast(
                FailureAttributionReason,
                _TRANSITION_REASON[transition],
            ),
            timestamp=timestamp or datetime.now(tz=UTC).isoformat(),
            idempotency_key=attribution_idempotency_key(
                candidate_id=candidate_id,
                transition=transition,
                turn_id=turn_id,
            ),
        )

    def to_event(self) -> dict[str, Any]:
        return {
            "type": "session_meta",
            "event": FAILURE_ATTRIBUTION_EVENT,
            "schema_version": FAILURE_ATTRIBUTION_SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "turn_id": self.turn_id,
            "transition": self.transition,
            "status": self.status,
            "reason_code": self.reason_code,
            "timestamp": self.timestamp,
            "idempotency_key": self.idempotency_key,
        }

    @classmethod
    def from_event(
        cls,
        value: Mapping[str, Any],
    ) -> "FailureAttributionRecord":
        if not isinstance(value, Mapping):
            raise FailureAttributionError("attribution event must be an object")
        _require_exact_keys(value)
        if (
            value["type"] != "session_meta"
            or value["event"] != FAILURE_ATTRIBUTION_EVENT
            or type(value["schema_version"]) is not int
            or value["schema_version"] != FAILURE_ATTRIBUTION_SCHEMA_VERSION
        ):
            raise FailureAttributionError("unsupported attribution event")
        return cls(
            candidate_id=value["candidate_id"],
            turn_id=value["turn_id"],
            transition=value["transition"],
            status=value["status"],
            reason_code=value["reason_code"],
            timestamp=value["timestamp"],
            idempotency_key=value["idempotency_key"],
        )


class FailureAttributionRepository:
    """CAS-safe append-only attribution writer for one explicit user session."""

    def __init__(self, session_store: FailureAttributionSessionStore) -> None:
        self._session_store = session_store

    async def append(
        self,
        session_id: str,
        record: FailureAttributionRecord,
    ) -> bool:
        session_id = _identifier(session_id, "session_id")
        event = record.to_event()
        outcome: str | None = None

        def predicate(events: list[dict[str, Any]]) -> bool:
            nonlocal outcome
            matches = [
                item
                for item in events
                if item.get("event") == FAILURE_ATTRIBUTION_EVENT
                and item.get("idempotency_key") == record.idempotency_key
            ]
            if not matches:
                outcome = "append"
                return True
            if len(matches) != 1:
                outcome = "conflict"
                return False
            try:
                existing = FailureAttributionRecord.from_event(matches[0])
            except FailureAttributionError:
                outcome = "conflict"
                return False
            existing_semantics = (
                existing.candidate_id,
                existing.turn_id,
                existing.transition,
                existing.status,
                existing.reason_code,
            )
            incoming_semantics = (
                record.candidate_id,
                record.turn_id,
                record.transition,
                record.status,
                record.reason_code,
            )
            outcome = (
                "existing" if existing_semantics == incoming_semantics else "conflict"
            )
            return False

        appended = await self._session_store.append_event_if(
            session_id,
            event,
            predicate,
        )
        if appended:
            return True
        if outcome == "existing":
            return False
        raise FailureAttributionError(
            "attribution idempotency key conflicts with durable history"
        )


__all__ = [
    "FAILURE_ATTRIBUTION_EVENT",
    "FAILURE_ATTRIBUTION_SCHEMA_VERSION",
    "FailureAttributionError",
    "FailureAttributionRecord",
    "FailureAttributionRepository",
    "FailureAttributionTransition",
    "attribution_idempotency_key",
]
