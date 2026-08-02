"""Strict, redacted v1 ordinary-Chat adaptive diagnostics contract.

This module deliberately contains only bounded counters and trusted identifiers.
It must never carry prompts, generated text, tool arguments, artifact contents,
paths, or any other unredacted execution payload.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
import re
from types import MappingProxyType
from typing import Any, Literal, Protocol, cast


DIAGNOSTICS_EVENT = "ordinary_chat_adaptive_diagnostics_recorded"
DIAGNOSTICS_VERSION = 1
DIAGNOSTICS_CONTEXT_KEY = "_ordinary_chat_adaptive_diagnostics_v1"
DIAGNOSTICS_CONTEXT_TURN_KEY = "_ordinary_chat_adaptive_diagnostics_turn_id"

DiagnosticsClassification = Literal["simple", "complex", "unknown"]

_CLASSIFICATIONS = frozenset({"simple", "complex", "unknown"})
_COUNTER_FIELDS = (
    "model_calls",
    "model_usage_observed_calls",
    "model_wall_observed_calls",
    "tool_calls",
    "input_tokens",
    "output_tokens",
    "model_wall_ms",
    "tool_wall_ms",
    "effectful_plan_guard_blocks",
    "retrieval_search_queries",
    "retrieval_zero_match_queries",
    "retrieval_candidates",
    "retrieval_activations",
    "retrieval_schema_count_before_total",
    "retrieval_schema_count_after_total",
    "retrieval_schema_token_estimate_before_total",
    "retrieval_schema_token_estimate_after_total",
    "recovery_attempts",
    "recovery_blocked",
    "recovery_budget_exhaustions",
    "recovery_model_calls",
    "recovery_tool_calls",
    "recovery_input_tokens",
    "recovery_output_tokens",
    "recovery_model_wall_ms",
    "recovery_tool_wall_ms",
)
DIAGNOSTICS_COUNTER_FIELDS = _COUNTER_FIELDS
_EVENT_FIELDS = frozenset(
    {
        "type",
        "event",
        "schema_version",
        "session_id",
        "turn_id",
        "idempotency_key",
        "classification",
        "counters",
        "timestamp",
    }
)
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_TIMESTAMP_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})"
)
_MAX_COUNTER = 1_000_000
_RECOVERY_SUBSET_PAIRS = (
    ("recovery_model_calls", "model_calls"),
    ("recovery_tool_calls", "tool_calls"),
    ("recovery_input_tokens", "input_tokens"),
    ("recovery_output_tokens", "output_tokens"),
    ("recovery_model_wall_ms", "model_wall_ms"),
    ("recovery_tool_wall_ms", "tool_wall_ms"),
    ("recovery_blocked", "recovery_attempts"),
    ("recovery_budget_exhaustions", "recovery_attempts"),
    ("model_usage_observed_calls", "model_calls"),
    ("model_wall_observed_calls", "model_calls"),
)


class AdaptiveDiagnosticsError(ValueError):
    """Raised when a diagnostics value is malformed or unsafe to persist."""


class AdaptiveDiagnosticsConflictError(AdaptiveDiagnosticsError):
    """A replay key exists but names a materially different diagnostics event."""


class AdaptiveDiagnosticsSessionStore(Protocol):
    async def append_event_if(
        self,
        session_id: str,
        event: dict[str, Any],
        predicate: Callable[[list[dict[str, Any]]], bool],
    ) -> bool: ...


def _count(value: Any, name: str) -> int:
    if type(value) is not int or value < 0 or value > _MAX_COUNTER:
        raise AdaptiveDiagnosticsError(f"{name} must be a bounded non-negative integer")
    return value


def _add_count(current: int, increment: Any, name: str) -> int:
    return _count(_count(current, name) + _count(increment, name), name)


def _validate_counter_snapshot(values: Mapping[str, Any]) -> dict[str, int]:
    checked = {name: _count(values[name], name) for name in _COUNTER_FIELDS}
    if any(checked[left] > checked[right] for left, right in _RECOVERY_SUBSET_PAIRS):
        raise AdaptiveDiagnosticsError("recovery counters exceed total counters")
    return checked


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise AdaptiveDiagnosticsError(f"{name} must be a bounded identifier")
    return value


def _classification(value: Any) -> DiagnosticsClassification:
    if not isinstance(value, str) or value not in _CLASSIFICATIONS:
        raise AdaptiveDiagnosticsError("classification is invalid")
    return cast(DiagnosticsClassification, value)


def _timestamp(value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) > 64
        or not _TIMESTAMP_RE.fullmatch(value)
    ):
        raise AdaptiveDiagnosticsError(
            "timestamp must be an ISO-8601 UTC-offset timestamp"
        )
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AdaptiveDiagnosticsError("timestamp is malformed") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise AdaptiveDiagnosticsError("timestamp must include a UTC offset")
    return value


def _idempotency_key(turn_id: str) -> str:
    return f"adaptive-diagnostics:{turn_id}:v1"


def _validate_idempotency_key(value: Any, turn_id: str) -> str:
    expected = _idempotency_key(turn_id)
    if not isinstance(value, str) or len(value) > 256 or value != expected:
        raise AdaptiveDiagnosticsError("idempotency_key does not match turn_id")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], name: str
) -> None:
    actual = frozenset(value)
    if actual != expected:
        unexpected = sorted(actual - expected)
        missing = sorted(expected - actual)
        details: list[str] = []
        if unexpected:
            details.append(f"unexpected keys: {unexpected}")
        if missing:
            details.append(f"missing keys: {missing}")
        raise AdaptiveDiagnosticsError(
            f"{name} must have exact keys ({'; '.join(details)})"
        )


@dataclass
class AdaptiveDiagnosticsAccumulator:
    """Mutable, bounded counters for one ordinary-Chat turn.

    ``add_model_call`` and ``add_tool_call`` are the only mutation helpers
    needed by runtime instrumentation. They accept numeric measurements only;
    callers must convert durations to integer milliseconds before recording.
    """

    model_calls: int = 0
    model_usage_observed_calls: int = 0
    model_wall_observed_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    model_wall_ms: int = 0
    tool_wall_ms: int = 0
    effectful_plan_guard_blocks: int = 0
    retrieval_search_queries: int = 0
    retrieval_zero_match_queries: int = 0
    retrieval_candidates: int = 0
    retrieval_activations: int = 0
    retrieval_schema_count_before_total: int = 0
    retrieval_schema_count_after_total: int = 0
    retrieval_schema_token_estimate_before_total: int = 0
    retrieval_schema_token_estimate_after_total: int = 0
    recovery_attempts: int = 0
    recovery_blocked: int = 0
    recovery_budget_exhaustions: int = 0
    recovery_model_calls: int = 0
    recovery_tool_calls: int = 0
    recovery_input_tokens: int = 0
    recovery_output_tokens: int = 0
    recovery_model_wall_ms: int = 0
    recovery_tool_wall_ms: int = 0

    def __post_init__(self) -> None:
        self.snapshot()

    def _apply_increments_atomic(self, increments: Mapping[str, Any]) -> None:
        updated = self.snapshot()
        for key, value in increments.items():
            if key not in _COUNTER_FIELDS:
                raise AdaptiveDiagnosticsError(f"unknown counter {key}")
            updated[key] = _add_count(updated[key], value, key)
        for key, value in _validate_counter_snapshot(updated).items():
            setattr(self, key, value)

    def add_model_call(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        wall_ms: int = 0,
        recovery: bool = False,
        usage_observed: bool = False,
        wall_observed: bool = False,
    ) -> None:
        if (
            type(recovery) is not bool
            or type(usage_observed) is not bool
            or type(wall_observed) is not bool
        ):
            raise AdaptiveDiagnosticsError("model-call flags must be bool")
        increments = {
            "model_calls": 1,
            "model_usage_observed_calls": int(usage_observed),
            "model_wall_observed_calls": int(wall_observed),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model_wall_ms": wall_ms,
        }
        if recovery:
            increments.update(
                {
                    "recovery_model_calls": 1,
                    "recovery_input_tokens": input_tokens,
                    "recovery_output_tokens": output_tokens,
                    "recovery_model_wall_ms": wall_ms,
                }
            )
        self._apply_increments_atomic(increments)

    def add_tool_call(self, *, wall_ms: int = 0, recovery: bool = False) -> None:
        if type(recovery) is not bool:
            raise AdaptiveDiagnosticsError("recovery must be bool")
        increments = {"tool_calls": 1, "tool_wall_ms": wall_ms}
        if recovery:
            increments.update(
                {"recovery_tool_calls": 1, "recovery_tool_wall_ms": wall_ms}
            )
        self._apply_increments_atomic(increments)

    def record_plan_guard_block(self) -> None:
        self._apply_increments_atomic({"effectful_plan_guard_blocks": 1})

    def record_tool_search(self, *, candidates: int | None) -> None:
        values: dict[str, int] = {"retrieval_search_queries": 1}
        if candidates is not None:
            if type(candidates) is not int:
                raise AdaptiveDiagnosticsError("candidates must be an integer or None")
            values["retrieval_candidates"] = candidates
            values["retrieval_zero_match_queries"] = int(candidates == 0)
        self._apply_increments_atomic(values)

    def record_activation(
        self,
        *,
        schema_count_before: int,
        schema_count_after: int,
        schema_token_estimate_before: int,
        schema_token_estimate_after: int,
    ) -> None:
        self._apply_increments_atomic(
            {
                "retrieval_activations": 1,
                "retrieval_schema_count_before_total": schema_count_before,
                "retrieval_schema_count_after_total": schema_count_after,
                "retrieval_schema_token_estimate_before_total": schema_token_estimate_before,
                "retrieval_schema_token_estimate_after_total": schema_token_estimate_after,
            }
        )

    def record_recovery_attempt(
        self, *, blocked: bool = False, budget_exhausted: bool = False
    ) -> None:
        if type(blocked) is not bool or type(budget_exhausted) is not bool:
            raise AdaptiveDiagnosticsError("recovery flags must be bool")
        values = {
            "recovery_attempts": 1,
            "recovery_blocked": int(blocked),
            "recovery_budget_exhaustions": int(budget_exhausted),
        }
        self._apply_increments_atomic(values)

    def snapshot(self) -> dict[str, int]:
        """Return a deterministic, validated copy suitable for serialization."""
        return _validate_counter_snapshot(
            {name: getattr(self, name) for name in _COUNTER_FIELDS}
        )

    def to_dict(self) -> dict[str, int]:
        """Serialize the redacted counter snapshot with a stable field order."""
        return self.snapshot()


def get_context_diagnostics_accumulator(
    context: Any,
    *,
    reset: bool = False,
) -> AdaptiveDiagnosticsAccumulator | None:
    """Get the turn-local accumulator without making a context mandatory.

    ``ToolExecutionContext`` instances are cached by ``AgentEngine`` across
    ordinary-Chat turns.  Callers starting a new unmarked turn must therefore
    pass ``reset=True``.  A future Engine finalization hook can set the turn
    marker before invoking ReAct, allowing recovery/approval continuations to
    retain one accumulator for their entire turn.
    """
    state = getattr(context, "state", None)
    if not isinstance(state, dict):
        return None
    accumulator = state.get(DIAGNOSTICS_CONTEXT_KEY)
    if reset or not isinstance(accumulator, AdaptiveDiagnosticsAccumulator):
        accumulator = AdaptiveDiagnosticsAccumulator()
        state[DIAGNOSTICS_CONTEXT_KEY] = accumulator
    return accumulator


def initialize_turn_diagnostics_accumulator(
    context: Any,
    *,
    turn_id: str,
) -> AdaptiveDiagnosticsAccumulator | None:
    """Create or retrieve a diagnostics accumulator bound to one trusted turn."""
    state = getattr(context, "state", None)
    if not isinstance(state, dict):
        return None
    normalized_turn_id = _identifier(turn_id, "turn_id")
    reset = state.get(DIAGNOSTICS_CONTEXT_TURN_KEY) != normalized_turn_id
    accumulator = get_context_diagnostics_accumulator(context, reset=reset)
    state[DIAGNOSTICS_CONTEXT_TURN_KEY] = normalized_turn_id
    return accumulator


@dataclass(frozen=True)
class AdaptiveDiagnosticsRecord:
    """The complete, redacted diagnostics event payload for a single turn."""

    session_id: str
    turn_id: str
    classification: DiagnosticsClassification
    counters: Mapping[str, int]
    timestamp: str
    idempotency_key: str

    def __post_init__(self) -> None:
        session_id = _identifier(self.session_id, "session_id")
        turn_id = _identifier(self.turn_id, "turn_id")
        classification = _classification(self.classification)
        if not isinstance(self.counters, Mapping):
            raise AdaptiveDiagnosticsError("counters must be an object")
        _require_exact_keys(self.counters, frozenset(_COUNTER_FIELDS), "counters")
        counters = {name: _count(self.counters[name], name) for name in _COUNTER_FIELDS}
        timestamp = _timestamp(self.timestamp)
        idempotency_key = _validate_idempotency_key(self.idempotency_key, turn_id)
        object.__setattr__(self, "session_id", session_id)
        object.__setattr__(self, "turn_id", turn_id)
        object.__setattr__(self, "classification", classification)
        object.__setattr__(self, "counters", MappingProxyType(counters))
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "idempotency_key", idempotency_key)

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        turn_id: str,
        classification: DiagnosticsClassification,
        accumulator: AdaptiveDiagnosticsAccumulator,
        timestamp: str,
    ) -> "AdaptiveDiagnosticsRecord":
        """Build a record with the deterministic v1 idempotency key."""
        return cls(
            session_id=session_id,
            turn_id=turn_id,
            classification=classification,
            counters=accumulator.snapshot(),
            timestamp=timestamp,
            idempotency_key=_idempotency_key(turn_id),
        )

    def to_event(self) -> dict[str, Any]:
        """Serialize a strict SessionStore-compatible v1 event envelope."""
        return {
            "type": "session_meta",
            "event": DIAGNOSTICS_EVENT,
            "schema_version": DIAGNOSTICS_VERSION,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "idempotency_key": self.idempotency_key,
            "classification": self.classification,
            "counters": dict(self.counters),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_event(cls, value: Mapping[str, Any]) -> "AdaptiveDiagnosticsRecord":
        """Parse only a complete, exact-key, v1 diagnostics event."""
        if not isinstance(value, Mapping):
            raise AdaptiveDiagnosticsError("diagnostics event must be an object")
        _require_exact_keys(value, _EVENT_FIELDS, "diagnostics event")
        if value["type"] != "session_meta" or value["event"] != DIAGNOSTICS_EVENT:
            raise AdaptiveDiagnosticsError("unsupported diagnostics event")
        if type(value["schema_version"]) is not int:
            raise AdaptiveDiagnosticsError("schema_version must be an integer")
        if value["schema_version"] != DIAGNOSTICS_VERSION:
            raise AdaptiveDiagnosticsError("unsupported diagnostics event version")
        counters = value["counters"]
        if not isinstance(counters, Mapping):
            raise AdaptiveDiagnosticsError("counters must be an object")
        _require_exact_keys(counters, frozenset(_COUNTER_FIELDS), "counters")
        return cls(
            session_id=_identifier(value["session_id"], "session_id"),
            turn_id=_identifier(value["turn_id"], "turn_id"),
            classification=_classification(value["classification"]),
            counters={name: _count(counters[name], name) for name in _COUNTER_FIELDS},
            timestamp=_timestamp(value["timestamp"]),
            idempotency_key=_validate_idempotency_key(
                value["idempotency_key"], _identifier(value["turn_id"], "turn_id")
            ),
        )


class AdaptiveDiagnosticsRepository:
    """Atomically append exactly one immutable diagnostics record per turn."""

    def __init__(self, session_store: AdaptiveDiagnosticsSessionStore) -> None:
        self._session_store = session_store

    async def append(self, record: AdaptiveDiagnosticsRecord) -> bool:
        """Append once; return False for a byte-equivalent durable replay.

        ``append_event_if`` executes its predicate beneath SessionStore's
        cross-process sidecar lock, making the matching-key check and append a
        single CAS operation rather than a load-then-append race.
        """
        event = record.to_event()

        # Production SessionStore exposes strict immutable snapshots.  Prefer
        # its revision-CAS mutation API; the append_event_if fallback preserves
        # atomic idempotency for narrow test doubles and older adapters.
        strict_load = getattr(self._session_store, "load_strict_snapshot", None)
        strict_mutate = getattr(self._session_store, "mutate_strict_snapshot", None)
        if callable(strict_load) and callable(strict_mutate):
            for _attempt in range(3):
                snapshot = await strict_load(record.session_id)
                outcome: str | None = None

                def build_events(current: Any) -> tuple[Mapping[str, Any], ...] | None:
                    nonlocal outcome
                    matching = [
                        item
                        for item in current.events
                        if item.get("idempotency_key") == record.idempotency_key
                    ]
                    if not matching:
                        outcome = "append"
                        return (event,)
                    if len(matching) != 1:
                        outcome = "conflict"
                        return None
                    try:
                        existing = AdaptiveDiagnosticsRecord.from_event(matching[0])
                    except AdaptiveDiagnosticsError:
                        outcome = "conflict"
                        return None
                    outcome = "existing" if existing.to_event() == event else "conflict"
                    return None

                result = await strict_mutate(
                    record.session_id,
                    expected_history_revision=snapshot.history_revision,
                    build_events=build_events,
                )
                if result.status == "rebase_required":
                    continue
                if result.status == "appended":
                    return True
                if outcome == "existing":
                    return False
                raise AdaptiveDiagnosticsConflictError(
                    "diagnostics idempotency key conflicts with durable history"
                )
            raise AdaptiveDiagnosticsConflictError(
                "diagnostics CAS rebase limit exceeded"
            )

        outcome: str | None = None

        def can_append(events: list[dict[str, Any]]) -> bool:
            nonlocal outcome
            matching = [
                item
                for item in events
                if item.get("idempotency_key") == record.idempotency_key
            ]
            if not matching:
                outcome = "append"
                return True
            if len(matching) != 1:
                outcome = "conflict"
                return False
            try:
                existing = AdaptiveDiagnosticsRecord.from_event(matching[0])
            except AdaptiveDiagnosticsError:
                outcome = "conflict"
                return False
            outcome = "existing" if existing.to_event() == event else "conflict"
            return False

        appended = await self._session_store.append_event_if(
            record.session_id,
            event,
            can_append,
        )
        if appended:
            return True
        if outcome == "existing":
            return False
        raise AdaptiveDiagnosticsConflictError(
            "diagnostics idempotency key conflicts with durable history"
        )
