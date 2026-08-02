from __future__ import annotations

from copy import deepcopy

import pytest

from mochi.agents.adaptive_diagnostics import (
    DIAGNOSTICS_EVENT,
    DIAGNOSTICS_VERSION,
    AdaptiveDiagnosticsAccumulator,
    AdaptiveDiagnosticsConflictError,
    AdaptiveDiagnosticsError,
    AdaptiveDiagnosticsRepository,
    AdaptiveDiagnosticsRecord,
)
from mochi.sessions.store import SessionStore


def _zero_counters(**overrides: int) -> dict[str, int]:
    counters = AdaptiveDiagnosticsAccumulator().snapshot()
    counters.update(overrides)
    return counters


def _record() -> AdaptiveDiagnosticsRecord:
    return AdaptiveDiagnosticsRecord.create(
        session_id="session-1",
        turn_id="turn-1",
        classification="simple",
        accumulator=AdaptiveDiagnosticsAccumulator(model_calls=1, input_tokens=12),
        timestamp="2026-07-28T12:34:56+00:00",
    )


def test_diagnostics_v1_round_trip_is_exact_and_deterministic() -> None:
    record = _record()

    assert record.to_event() == {
        "type": "session_meta",
        "event": DIAGNOSTICS_EVENT,
        "schema_version": DIAGNOSTICS_VERSION,
        "session_id": "session-1",
        "turn_id": "turn-1",
        "idempotency_key": "adaptive-diagnostics:turn-1:v1",
        "classification": "simple",
        "counters": _zero_counters(model_calls=1, input_tokens=12),
        "timestamp": "2026-07-28T12:34:56+00:00",
    }
    assert AdaptiveDiagnosticsRecord.from_event(record.to_event()) == record


def test_idempotency_key_remains_valid_for_the_longest_turn_identifier() -> None:
    turn_id = "t" * 128
    record = AdaptiveDiagnosticsRecord.create(
        session_id="session-1",
        turn_id=turn_id,
        classification="complex",
        accumulator=AdaptiveDiagnosticsAccumulator(),
        timestamp="2026-07-28T12:34:56+00:00",
    )

    assert AdaptiveDiagnosticsRecord.from_event(record.to_event()) == record


def test_record_copies_counter_snapshot_before_the_accumulator_can_mutate() -> None:
    accumulator = AdaptiveDiagnosticsAccumulator(model_calls=1)
    record = AdaptiveDiagnosticsRecord.create(
        session_id="session-1",
        turn_id="turn-1",
        classification="simple",
        accumulator=accumulator,
        timestamp="2026-07-28T12:34:56+00:00",
    )

    accumulator.add_model_call(input_tokens=99)

    assert record.to_event()["counters"] == _zero_counters(model_calls=1)
    with pytest.raises(TypeError):
        record.counters["model_calls"] = 99  # type: ignore[index]


class _AtomicStore:
    def __init__(self) -> None:
        self.events: dict[str, list[dict[str, object]]] = {}

    async def append_event_if(self, session_id, event, predicate):  # type: ignore[no-untyped-def]
        history = self.events.setdefault(session_id, [])
        if not predicate(history):
            return False
        history.append(dict(event))
        return True


@pytest.mark.asyncio
async def test_repository_appends_once_and_rejects_conflicting_replay() -> None:
    store = _AtomicStore()
    repository = AdaptiveDiagnosticsRepository(store)
    record = _record()

    assert await repository.append(record) is True
    assert await repository.append(record) is False
    assert len(store.events["session-1"]) == 1

    conflicting = AdaptiveDiagnosticsRecord.create(
        session_id="session-1",
        turn_id="turn-1",
        classification="complex",
        accumulator=AdaptiveDiagnosticsAccumulator(model_calls=3),
        timestamp="2026-07-28T12:34:56+00:00",
    )
    with pytest.raises(AdaptiveDiagnosticsConflictError):
        await repository.append(conflicting)


@pytest.mark.asyncio
async def test_repository_uses_strict_session_store_cas_and_keeps_old_sessions_readable(
    tmp_path,
) -> None:
    store = SessionStore(tmp_path / "sessions")
    await store.save_event("session-1", {"type": "message", "content": "old"})
    repository = AdaptiveDiagnosticsRepository(store)

    assert await repository.append(_record()) is True
    assert await repository.append(_record()) is False
    events = await store.load_session("session-1")
    assert [event["type"] for event in events] == ["message", "session_meta"]


def test_extended_diagnostics_helpers_accumulate_verified_subsets() -> None:
    counters = AdaptiveDiagnosticsAccumulator()
    counters.record_plan_guard_block()
    counters.record_tool_search(candidates=2)
    counters.record_tool_search(candidates=0)
    counters.record_activation(schema_count_before=11, schema_count_after=6, schema_token_estimate_before=110, schema_token_estimate_after=60)
    counters.record_recovery_attempt(blocked=True, budget_exhausted=True)
    counters.add_model_call(
        input_tokens=4,
        output_tokens=2,
        wall_ms=6,
        recovery=True,
        usage_observed=True,
        wall_observed=True,
    )
    counters.add_tool_call(wall_ms=5, recovery=True)
    snapshot = counters.snapshot()
    assert snapshot["effectful_plan_guard_blocks"] == snapshot["retrieval_activations"] == 1
    assert (snapshot["retrieval_search_queries"], snapshot["retrieval_zero_match_queries"], snapshot["retrieval_candidates"]) == (2, 1, 2)
    assert (snapshot["retrieval_schema_count_before_total"], snapshot["retrieval_schema_count_after_total"], snapshot["retrieval_schema_token_estimate_before_total"], snapshot["retrieval_schema_token_estimate_after_total"]) == (11, 6, 110, 60)
    assert (snapshot["recovery_attempts"], snapshot["recovery_blocked"], snapshot["recovery_budget_exhaustions"]) == (1, 1, 1)
    assert (snapshot["recovery_model_calls"], snapshot["recovery_tool_calls"], snapshot["recovery_input_tokens"], snapshot["recovery_output_tokens"], snapshot["recovery_model_wall_ms"], snapshot["recovery_tool_wall_ms"]) == (1, 1, 4, 2, 6, 5)
    assert (
        snapshot["model_usage_observed_calls"],
        snapshot["model_wall_observed_calls"],
    ) == (1, 1)


def test_constructor_rejects_recovery_subset_larger_than_total() -> None:
    with pytest.raises(AdaptiveDiagnosticsError):
        AdaptiveDiagnosticsAccumulator(model_calls=0, recovery_model_calls=1)
    with pytest.raises(AdaptiveDiagnosticsError):
        AdaptiveDiagnosticsAccumulator(
            model_calls=0,
            model_usage_observed_calls=1,
        )


@pytest.mark.parametrize("action", [
    lambda value: AdaptiveDiagnosticsAccumulator().add_model_call(input_tokens=value),
    lambda value: AdaptiveDiagnosticsAccumulator().add_tool_call(wall_ms=value),
    lambda value: AdaptiveDiagnosticsAccumulator().record_tool_search(candidates=value),
])
def test_public_helper_invalid_measurement_is_atomic(action) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(AdaptiveDiagnosticsError): action(-1)


def test_public_helper_invalid_flags_are_rejected_without_mutation() -> None:
    counters = AdaptiveDiagnosticsAccumulator(); before = counters.snapshot()
    with pytest.raises(AdaptiveDiagnosticsError): counters.add_model_call(recovery=1)  # type: ignore[arg-type]
    with pytest.raises(AdaptiveDiagnosticsError): counters.add_model_call(usage_observed=1)  # type: ignore[arg-type]
    with pytest.raises(AdaptiveDiagnosticsError): counters.add_model_call(wall_observed=1)  # type: ignore[arg-type]
    with pytest.raises(AdaptiveDiagnosticsError): counters.record_tool_search(candidates=True)  # type: ignore[arg-type]
    with pytest.raises(AdaptiveDiagnosticsError): counters.record_recovery_attempt(blocked=1)  # type: ignore[arg-type]
    assert counters.snapshot() == before


def test_accumulator_records_only_bounded_numeric_measurements() -> None:
    accumulator = AdaptiveDiagnosticsAccumulator()

    accumulator.add_model_call(input_tokens=10, output_tokens=4, wall_ms=13)
    accumulator.add_tool_call(wall_ms=7)

    assert accumulator.snapshot() == _zero_counters(model_calls=1, tool_calls=1, input_tokens=10, output_tokens=4, model_wall_ms=13, tool_wall_ms=7)
    with pytest.raises(AdaptiveDiagnosticsError, match="input_tokens"):
        accumulator.add_model_call(input_tokens=True)
    with pytest.raises(AdaptiveDiagnosticsError, match="tool_wall_ms"):
        accumulator.add_tool_call(wall_ms=-1)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("session_id", "C:/workspace/secret.txt"),
        ("session_id", 1),
        ("turn_id", "turn with whitespace"),
        ("classification", "model-answer"),
        ("timestamp", "2026-07-28T12:34:56"),
        ("timestamp", "not-a-timestamp"),
        ("idempotency_key", "wrong"),
    ],
)
def test_record_rejects_unredacted_or_malformed_metadata(
    field: str, value: object
) -> None:
    fields = {
        "session_id": "session-1",
        "turn_id": "turn-1",
        "classification": "simple",
        "counters": AdaptiveDiagnosticsAccumulator().snapshot(),
        "timestamp": "2026-07-28T12:34:56+00:00",
        "idempotency_key": "adaptive-diagnostics:turn-1:v1",
    }
    fields[field] = value

    with pytest.raises(AdaptiveDiagnosticsError):
        AdaptiveDiagnosticsRecord(**fields)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda event: event.__setitem__("extra", "value"),
        lambda event: event.pop("timestamp"),
        lambda event: event.__setitem__("schema_version", True),
        lambda event: event.__setitem__("schema_version", DIAGNOSTICS_VERSION + 1),
        lambda event: event.__setitem__("type", "assistant_message"),
        lambda event: event.__setitem__("classification", "prompt text"),
        lambda event: event.__setitem__("idempotency_key", "adaptive-diagnostics:other:v1"),
        lambda event: event["counters"].__setitem__("model_calls", True),
        lambda event: event["counters"].__setitem__("model_wall_ms", float("inf")),
        lambda event: event["counters"].__setitem__("tool_calls", -1),
        lambda event: event["counters"].pop("tool_wall_ms"),
        lambda event: event["counters"].__setitem__("raw_prompt", "secret"),
    ],
)
def test_parser_fails_closed_for_any_malformed_or_unredacted_event(mutation: object) -> None:
    event = deepcopy(_record().to_event())
    mutation(event)  # type: ignore[operator]

    with pytest.raises(AdaptiveDiagnosticsError):
        AdaptiveDiagnosticsRecord.from_event(event)
