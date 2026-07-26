from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime, timedelta

import pytest

from mochi.sessions.store import SessionStore, StrictSessionSnapshotError
from mochi.sessions.turn_timeline import (
    OperationDescriptor,
    SESSION_TURN_TIMELINE_EVENT,
    SessionTurnTimelineRepository,
)


def _now() -> datetime:
    return datetime(2026, 7, 25, 9, 0, tzinfo=UTC)


async def _admit(
    repository: SessionTurnTimelineRepository,
    session_id: str,
    turn_id: str,
):
    loaded = await repository.load(session_id)
    assert loaded.history_revision is not None
    result = await repository.admit(
        session_id,
        turn_id=turn_id,
        expected_history_revision=loaded.history_revision,
    )
    assert result.status == "admitted"
    assert result.history_revision is not None
    return result


def _descriptor(operation_id: str, *, tool_name: str = "file_write", call_id: str | None = None):
    return OperationDescriptor(
        operation_id=operation_id,
        tool_name=tool_name,
        arguments_digest=hashlib.sha256(operation_id.encode()).hexdigest(),
        call_id=call_id or f"call-{operation_id}",
    )


@pytest.mark.asyncio
async def test_strict_snapshot_is_immutable_and_rejects_empty_malformed_and_mismatched_history(
    tmp_path,
) -> None:
    sessions_dir = tmp_path / "sessions"
    store = SessionStore(sessions_dir)

    missing = await store.load_strict_snapshot("missing")
    assert missing.exists is False
    assert missing.events == ()

    empty_path = sessions_dir / "empty.jsonl"
    empty_path.parent.mkdir(parents=True, exist_ok=True)
    empty_path.write_text("", encoding="utf-8")
    with pytest.raises(StrictSessionSnapshotError, match="empty"):
        await store.load_strict_snapshot("empty")

    empty_object_path = sessions_dir / "empty-object.jsonl"
    empty_object_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(StrictSessionSnapshotError, match="empty object"):
        await store.load_strict_snapshot("empty-object")

    malformed_path = sessions_dir / "malformed.jsonl"
    malformed_path.write_text('{"type":"message"}\n{broken}\n', encoding="utf-8")
    with pytest.raises(StrictSessionSnapshotError, match="malformed JSON"):
        await store.load_strict_snapshot("malformed")

    await store.save_event(
        "mismatch",
        {"type": "session_meta", "session_id": "another-session"},
    )
    with pytest.raises(StrictSessionSnapshotError, match="does not match"):
        await store.load_strict_snapshot("mismatch")

    await store.save_event("immutable", {"type": "message", "role": "user", "content": "hi"})
    snapshot = await store.load_strict_snapshot("immutable")
    with pytest.raises(TypeError):
        snapshot.events[0]["role"] = "assistant"  # type: ignore[index]


@pytest.mark.asyncio
async def test_strict_batch_is_atomic_and_has_deterministic_cross_instance_revision(tmp_path) -> None:
    sessions_dir = tmp_path / "sessions"
    first = SessionStore(sessions_dir)
    second = SessionStore(sessions_dir)

    initial = await first.load_strict_snapshot("batch")
    same_initial = await second.load_strict_snapshot("batch")
    assert initial.history_revision == same_initial.history_revision
    appended = await first.append_strict_batch_if_revision(
        "batch",
        expected_history_revision=initial.history_revision,
        events=(
            {"type": "message", "role": "user", "content": "one"},
            {"type": "message", "role": "assistant", "content": "two"},
        ),
    )
    assert appended.status == "appended"
    assert appended.after.event_count == 2

    with pytest.raises(TypeError, match="must be an object"):
        await second.append_strict_batch_if_revision(
            "batch",
            expected_history_revision=appended.after.history_revision,
            events=({"type": "message", "role": "user"}, 42),  # type: ignore[arg-type]
        )
    unchanged = await second.load_strict_snapshot("batch")
    assert unchanged.history_revision == appended.after.history_revision
    assert unchanged.event_count == 2

    stale = await second.append_strict_batch_if_revision(
        "batch",
        expected_history_revision=initial.history_revision,
        events=({"type": "message", "role": "user", "content": "late"},),
    )
    assert stale.status == "rebase_required"
    assert stale.after.event_count == 2


@pytest.mark.asyncio
async def test_timeline_fails_closed_for_malformed_middle_json_and_future_payload(tmp_path) -> None:
    sessions_dir = tmp_path / "sessions"
    store = SessionStore(sessions_dir)
    repository = SessionTurnTimelineRepository(store)
    await store.save_event("broken", {"type": "message", "role": "user", "content": "first"})
    path = store._session_path("broken")  # noqa: SLF001
    with path.open("a", encoding="utf-8") as fh:
        fh.write("{not-json}\n")
        fh.write('{"type":"message","role":"assistant","content":"third"}\n')
    broken = await repository.load("broken")
    assert broken.status == "invalid"
    assert broken.timeline is None

    future_snapshot = await store.load_strict_snapshot("future")
    future = await store.append_strict_batch_if_revision(
        "future",
        expected_history_revision=future_snapshot.history_revision,
        events=(
            {
                "type": "session_meta",
                "event": SESSION_TURN_TIMELINE_EVENT,
                "schema_version": 2,
                "session_id": "future",
                "timeline": {},
                "timestamp": _now().isoformat(),
            },
        ),
    )
    assert future.status == "appended"
    loaded = await repository.load("future")
    assert loaded.status == "unsupported_version"
    assert loaded.timeline is None


@pytest.mark.asyncio
async def test_two_repository_instances_have_one_admission_cas_winner_and_no_duplicate_event(tmp_path) -> None:
    sessions_dir = tmp_path / "sessions"
    first = SessionTurnTimelineRepository(SessionStore(sessions_dir))
    second = SessionTurnTimelineRepository(SessionStore(sessions_dir))
    first_loaded, second_loaded = await asyncio.gather(
        first.load("shared"),
        second.load("shared"),
    )
    assert first_loaded.history_revision == second_loaded.history_revision
    assert first_loaded.history_revision is not None

    first_result, second_result = await asyncio.gather(
        first.admit(
            "shared",
            turn_id="turn-one",
            expected_history_revision=first_loaded.history_revision,
        ),
        second.admit(
            "shared",
            turn_id="turn-two",
            expected_history_revision=second_loaded.history_revision,
        ),
    )
    assert sorted((first_result.status, second_result.status)) == ["admitted", "rebase_required"]

    loaded = await first.load("shared")
    assert loaded.status == "loaded"
    assert loaded.timeline is not None
    assert len(loaded.timeline.turns) == 1
    snapshot = await SessionStore(sessions_dir).load_strict_snapshot("shared")
    assert sum(event.get("event") == SESSION_TURN_TIMELINE_EVENT for event in snapshot.events) == 1

    assert loaded.history_revision is not None
    duplicate = await second.admit(
        "shared",
        turn_id=loaded.timeline.turns[0].turn_id,
        expected_history_revision=loaded.history_revision,
    )
    assert duplicate.status == "duplicate"
    still_one = await SessionStore(sessions_dir).load_strict_snapshot("shared")
    assert still_one.event_count == snapshot.event_count


@pytest.mark.asyncio
async def test_fifo_lane_claim_terminal_release_and_queued_cancellation(tmp_path) -> None:
    repository = SessionTurnTimelineRepository(SessionStore(tmp_path / "sessions"))
    first = await _admit(repository, "fifo", "turn-one")
    assert first.history_revision is not None
    second = await repository.admit(
        "fifo",
        turn_id="turn-two",
        expected_history_revision=first.history_revision,
    )
    assert second.status == "admitted"
    assert second.history_revision is not None

    claim = await repository.claim_next(
        "fifo",
        expected_history_revision=second.history_revision,
        owner="worker-a",
        token="lease-a",
        lease_expires_at=(_now() + timedelta(minutes=5)).isoformat(),
        now=_now(),
    )
    assert claim.status == "claimed"
    assert claim.timeline is not None
    assert claim.timeline.lane_turn_id == "turn-one"
    assert claim.history_revision is not None

    cancelled = await repository.cancel(
        "fifo",
        turn_id="turn-two",
        expected_history_revision=claim.history_revision,
    )
    assert cancelled.status == "terminal"
    assert cancelled.timeline is not None
    assert cancelled.timeline.turns[1].cancellation_outcome == "cancelled_queued"
    assert cancelled.history_revision is not None

    terminal = await repository.terminal(
        "fifo",
        turn_id="turn-one",
        expected_history_revision=cancelled.history_revision,
        owner="worker-a",
        token="lease-a",
        outcome="completed",
        now=_now(),
    )
    assert terminal.status == "terminal"
    assert terminal.timeline is not None
    assert terminal.timeline.lane_turn_id is None
    assert terminal.history_revision is not None
    empty = await repository.claim_next(
        "fifo",
        expected_history_revision=terminal.history_revision,
        owner="worker-b",
        token="lease-b",
        lease_expires_at=(_now() + timedelta(minutes=5)).isoformat(),
        now=_now(),
    )
    assert empty.status == "queue_empty"


@pytest.mark.asyncio
async def test_running_final_and_cancel_race_has_one_cas_winner(tmp_path) -> None:
    repository = SessionTurnTimelineRepository(SessionStore(tmp_path / "sessions"))
    admitted = await _admit(repository, "race", "turn-race")
    assert admitted.history_revision is not None
    claim = await repository.claim_next(
        "race",
        expected_history_revision=admitted.history_revision,
        owner="worker",
        token="lease",
        lease_expires_at=(_now() + timedelta(minutes=5)).isoformat(),
        now=_now(),
    )
    assert claim.history_revision is not None
    final, cancel = await asyncio.gather(
        repository.terminal(
            "race",
            turn_id="turn-race",
            expected_history_revision=claim.history_revision,
            owner="worker",
            token="lease",
            outcome="completed",
            now=_now(),
        ),
        repository.cancel(
            "race",
            turn_id="turn-race",
            expected_history_revision=claim.history_revision,
            owner="worker",
            token="lease",
            now=_now(),
        ),
    )
    assert [final.status, cancel.status].count("terminal") == 1
    assert [final.status, cancel.status].count("rebase_required") == 1
    loaded = await repository.load("race")
    assert loaded.timeline is not None
    turn = loaded.timeline.turns[0]
    assert turn.status == "terminal"
    assert turn.terminal_outcome in {"completed", "cancelled"}


@pytest.mark.asyncio
async def test_lease_identity_and_stale_lease_fail_closed_without_replay(tmp_path) -> None:
    repository = SessionTurnTimelineRepository(SessionStore(tmp_path / "sessions"))
    first = await _admit(repository, "lease", "turn-one")
    assert first.history_revision is not None
    second = await repository.admit(
        "lease",
        turn_id="turn-two",
        expected_history_revision=first.history_revision,
    )
    assert second.history_revision is not None
    expiry = _now() + timedelta(minutes=1)
    claim = await repository.claim_next(
        "lease",
        expected_history_revision=second.history_revision,
        owner="worker-a",
        token="token-a",
        lease_expires_at=expiry.isoformat(),
        now=_now(),
    )
    assert claim.history_revision is not None

    wrong_owner = await repository.mark_side_effect_boundary(
        "lease",
        turn_id="turn-one",
        expected_history_revision=claim.history_revision,
        owner="worker-b",
        token="token-a",
        boundary="started",
        operation_id="unbound-operation",
        now=_now(),
    )
    assert wrong_owner.status == "lease_invalid"
    assert wrong_owner.history_revision == claim.history_revision

    stale = await repository.claim_next(
        "lease",
        expected_history_revision=claim.history_revision,
        owner="worker-b",
        token="token-b",
        lease_expires_at=(_now() + timedelta(minutes=10)).isoformat(),
        now=expiry + timedelta(seconds=1),
    )
    assert stale.status == "lease_stale"
    loaded = await repository.load("lease")
    assert loaded.timeline is not None
    assert loaded.timeline.lane_turn_id == "turn-one"
    assert loaded.timeline.turns[0].status == "running"


@pytest.mark.asyncio
async def test_late_operation_precommit_must_precede_exact_side_effect_boundary(tmp_path) -> None:
    repository = SessionTurnTimelineRepository(SessionStore(tmp_path / "sessions"))
    admitted = await _admit(repository, "precommit", "turn-one")
    assert admitted.history_revision is not None
    claimed = await repository.claim_next(
        "precommit",
        expected_history_revision=admitted.history_revision,
        owner="worker",
        token="token",
        lease_expires_at=(_now() + timedelta(minutes=5)).isoformat(),
        now=_now(),
    )
    assert claimed.history_revision is not None

    no_precommit = await repository.mark_side_effect_boundary(
        "precommit",
        turn_id="turn-one",
        expected_history_revision=claimed.history_revision,
        owner="worker",
        token="token",
        boundary="started",
        operation_id="missing-operation",
        now=_now(),
    )
    assert no_precommit.status == "invalid"

    descriptor = _descriptor("operation-one")
    precommitted = await repository.record_operation_precommit(
        "precommit",
        turn_id="turn-one",
        expected_history_revision=claimed.history_revision,
        owner="worker",
        token="token",
        descriptor=descriptor,
        now=_now(),
    )
    assert precommitted.status == "precommitted"
    assert precommitted.timeline is not None
    assert precommitted.history_revision is not None
    assert precommitted.timeline.turns[0].operation_descriptors == (descriptor,)

    idempotent = await repository.record_operation_precommit(
        "precommit",
        turn_id="turn-one",
        expected_history_revision=precommitted.history_revision,
        owner="worker",
        token="token",
        descriptor=descriptor,
        now=_now(),
    )
    assert idempotent.status == "precommitted"
    assert idempotent.history_revision == precommitted.history_revision

    started = await repository.mark_side_effect_boundary(
        "precommit",
        turn_id="turn-one",
        expected_history_revision=idempotent.history_revision,
        owner="worker",
        token="token",
        boundary="started",
        operation_id=descriptor.operation_id,
        now=_now(),
    )
    assert started.status == "boundary_updated"
    assert started.timeline is not None
    assert started.history_revision is not None
    recorded = started.timeline.turns[0].operation_descriptors[0]
    assert recorded.status == "started"
    assert recorded.precommit_boundary == "started"
    reloaded = await repository.load("precommit")
    assert reloaded.status == "loaded"
    assert reloaded.timeline is not None
    assert reloaded.timeline.turns[0].operation_descriptors[0] == recorded

    too_late = await repository.record_operation_precommit(
        "precommit",
        turn_id="turn-one",
        expected_history_revision=started.history_revision,
        owner="worker",
        token="token",
        descriptor=_descriptor("operation-two"),
        now=_now(),
    )
    assert too_late.status == "invalid"


@pytest.mark.asyncio
async def test_operation_identity_mismatch_cross_turn_and_stale_precommit_fail_closed(tmp_path) -> None:
    repository = SessionTurnTimelineRepository(SessionStore(tmp_path / "sessions"))
    first = await _admit(repository, "operations", "turn-one")
    assert first.history_revision is not None
    second = await repository.admit(
        "operations",
        turn_id="turn-two",
        expected_history_revision=first.history_revision,
    )
    assert second.history_revision is not None
    expiry = _now() + timedelta(minutes=1)
    claimed_first = await repository.claim_next(
        "operations",
        expected_history_revision=second.history_revision,
        owner="worker-one",
        token="token-one",
        lease_expires_at=expiry.isoformat(),
        now=_now(),
    )
    assert claimed_first.history_revision is not None

    stale = await repository.record_operation_precommit(
        "operations",
        turn_id="turn-one",
        expected_history_revision=claimed_first.history_revision,
        owner="worker-one",
        token="token-one",
        descriptor=_descriptor("stale-operation"),
        now=expiry + timedelta(seconds=1),
    )
    assert stale.status == "lease_stale"

    descriptor = _descriptor("operation-shared", call_id="call-shared")
    precommitted = await repository.record_operation_precommit(
        "operations",
        turn_id="turn-one",
        expected_history_revision=claimed_first.history_revision,
        owner="worker-one",
        token="token-one",
        descriptor=descriptor,
        now=_now(),
    )
    assert precommitted.history_revision is not None
    local_mismatch = OperationDescriptor(
        operation_id="operation-shared",
        tool_name="file_edit",
        arguments_digest=hashlib.sha256(b'local-mismatch').hexdigest(),
        call_id="call-local-mismatch",
    )
    mismatch_same_turn = await repository.record_operation_precommit(
        "operations",
        turn_id="turn-one",
        expected_history_revision=precommitted.history_revision,
        owner="worker-one",
        token="token-one",
        descriptor=local_mismatch,
        now=_now(),
    )
    assert mismatch_same_turn.status == "invalid"
    assert mismatch_same_turn.history_revision == precommitted.history_revision
    started = await repository.mark_side_effect_boundary(
        "operations",
        turn_id="turn-one",
        expected_history_revision=precommitted.history_revision,
        owner="worker-one",
        token="token-one",
        boundary="started",
        operation_id=descriptor.operation_id,
        now=_now(),
    )
    assert started.history_revision is not None
    failed = await repository.record_operation_result(
        "operations",
        turn_id="turn-one",
        expected_history_revision=started.history_revision,
        owner="worker-one",
        token="token-one",
        operation_id=descriptor.operation_id,
        status="failed",
        result_digest=f"sha256:{hashlib.sha256(b'failure').hexdigest()}",
        receipt_reference="receipt-operation-shared",
        now=_now(),
    )
    assert failed.history_revision is not None
    terminal = await repository.terminal(
        "operations",
        turn_id="turn-one",
        expected_history_revision=failed.history_revision,
        owner="worker-one",
        token="token-one",
        outcome="blocked",
        now=_now(),
    )
    assert terminal.history_revision is not None
    claimed_second = await repository.claim_next(
        "operations",
        expected_history_revision=terminal.history_revision,
        owner="worker-two",
        token="token-two",
        lease_expires_at=(_now() + timedelta(minutes=5)).isoformat(),
        now=_now(),
    )
    assert claimed_second.history_revision is not None

    mismatch = OperationDescriptor(
        operation_id="operation-shared",
        tool_name="file_edit",
        arguments_digest=hashlib.sha256(b'different').hexdigest(),
        call_id="call-shared",
    )
    conflict = await repository.record_operation_precommit(
        "operations",
        turn_id="turn-two",
        expected_history_revision=claimed_second.history_revision,
        owner="worker-two",
        token="token-two",
        descriptor=mismatch,
        now=_now(),
    )
    assert conflict.status == "invalid"
    assert conflict.history_revision == claimed_second.history_revision


@pytest.mark.asyncio
async def test_v1_timeline_migrates_without_fabricating_operation_evidence_and_future_v4_rejects(
    tmp_path,
) -> None:
    store = SessionStore(tmp_path / "sessions")
    repository = SessionTurnTimelineRepository(store)
    initial = await store.load_strict_snapshot("legacy")
    migrated_event = {
        "type": "session_meta",
        "event": SESSION_TURN_TIMELINE_EVENT,
        "schema_version": 1,
        "session_id": "legacy",
        "timeline": {
            "timeline_version": "session-turn-timeline-v1",
            "session_id": "legacy",
            "history_base_revision": 0,
            "history_current_revision": 1,
            "turns": [
                {
                    "sequence": 1,
                    "turn_id": "legacy-turn",
                    "status": "queued",
                    "side_effect_boundary": "not_started",
                    "operation_ids": ["legacy-operation"],
                    "cancellation_outcome": None,
                    "terminal_outcome": None,
                },
            ],
            "lane_turn_id": None,
            "lane_owner": None,
            "lane_token": None,
            "lane_lease_expires_at": None,
        },
        "timestamp": _now().isoformat(),
    }
    appended = await store.append_strict_batch_if_revision(
        "legacy",
        expected_history_revision=initial.history_revision,
        events=(migrated_event,),
    )
    assert appended.status == "appended"
    loaded = await repository.load("legacy")
    assert loaded.status == "loaded"
    assert loaded.timeline is not None
    assert loaded.timeline.timeline_version == "session-turn-timeline-v4"
    assert loaded.timeline.turns[0].operation_descriptors == ()
    assert loaded.timeline.turns[0].legacy_operation_ids == ("legacy-operation",)
    assert loaded.history_revision is not None

    claimed = await repository.claim_next(
        "legacy",
        expected_history_revision=loaded.history_revision,
        owner="worker",
        token="token",
        lease_expires_at=(_now() + timedelta(minutes=5)).isoformat(),
        now=_now(),
    )
    assert claimed.history_revision is not None
    v2_reloaded = await repository.load("legacy")
    assert v2_reloaded.status == "loaded"
    assert v2_reloaded.timeline is not None
    assert v2_reloaded.timeline.timeline_version == "session-turn-timeline-v4"
    assert v2_reloaded.timeline.turns[0].legacy_operation_ids == ("legacy-operation",)
    legacy_replay = await repository.record_operation_precommit(
        "legacy",
        turn_id="legacy-turn",
        expected_history_revision=claimed.history_revision,
        owner="worker",
        token="token",
        descriptor=_descriptor("legacy-operation"),
        now=_now(),
    )
    assert legacy_replay.status == "invalid"

    future = await store.load_strict_snapshot("future-v4")
    future_event = dict(migrated_event)
    future_event["session_id"] = "future-v4"
    future_event["timeline"] = dict(migrated_event["timeline"])
    future_event["timeline"]["session_id"] = "future-v4"
    future_event["timeline"]["timeline_version"] = "session-turn-timeline-v5"
    appended_future = await store.append_strict_batch_if_revision(
        "future-v4",
        expected_history_revision=future.history_revision,
        events=(future_event,),
    )
    assert appended_future.status == "appended"
    assert (await repository.load("future-v4")).status == "unsupported_version"


@pytest.mark.asyncio
async def test_sequential_operations_require_durable_known_results_before_replan(tmp_path) -> None:
    repository = SessionTurnTimelineRepository(SessionStore(tmp_path / "sessions"))
    admitted = await _admit(repository, "sequential", "turn-one")
    assert admitted.history_revision is not None
    claimed = await repository.claim_next(
        "sequential",
        expected_history_revision=admitted.history_revision,
        owner="worker",
        token="token",
        lease_expires_at=(_now() + timedelta(minutes=5)).isoformat(),
        now=_now(),
    )
    assert claimed.history_revision is not None
    first = _descriptor("operation-first")
    first_precommit = await repository.record_operation_precommit(
        "sequential",
        turn_id="turn-one",
        expected_history_revision=claimed.history_revision,
        owner="worker",
        token="token",
        descriptor=first,
        now=_now(),
    )
    assert first_precommit.history_revision is not None
    second = _descriptor("operation-second")
    unresolved_second = await repository.record_operation_precommit(
        "sequential",
        turn_id="turn-one",
        expected_history_revision=first_precommit.history_revision,
        owner="worker",
        token="token",
        descriptor=second,
        now=_now(),
    )
    assert unresolved_second.status == "invalid"
    assert unresolved_second.history_revision == first_precommit.history_revision

    first_started = await repository.mark_side_effect_boundary(
        "sequential",
        turn_id="turn-one",
        expected_history_revision=first_precommit.history_revision,
        owner="worker",
        token="token",
        boundary="started",
        operation_id=first.operation_id,
        now=_now(),
    )
    assert first_started.history_revision is not None
    with pytest.raises(ValueError, match="requires result_digest or receipt_reference"):
        await repository.record_operation_result(
            "sequential",
            turn_id="turn-one",
            expected_history_revision=first_started.history_revision,
            owner="worker",
            token="token",
            operation_id=first.operation_id,
            status="succeeded",
            now=_now(),
        )
    first_result_digest = f"sha256:{hashlib.sha256(b'first-result').hexdigest()}"
    first_succeeded = await repository.record_operation_result(
        "sequential",
        turn_id="turn-one",
        expected_history_revision=first_started.history_revision,
        owner="worker",
        token="token",
        operation_id=first.operation_id,
        status="succeeded",
        result_digest=first_result_digest,
        receipt_reference="receipt-first",
        now=_now(),
    )
    assert first_succeeded.history_revision is not None
    result_idempotent = await repository.record_operation_result(
        "sequential",
        turn_id="turn-one",
        expected_history_revision=first_succeeded.history_revision,
        owner="worker",
        token="token",
        operation_id=first.operation_id,
        status="succeeded",
        result_digest=first_result_digest,
        receipt_reference="receipt-first",
        now=_now(),
    )
    assert result_idempotent.status == "operation_result"
    assert result_idempotent.history_revision == first_succeeded.history_revision
    result_mismatch = await repository.record_operation_result(
        "sequential",
        turn_id="turn-one",
        expected_history_revision=result_idempotent.history_revision,
        owner="worker",
        token="token",
        operation_id=first.operation_id,
        status="failed",
        result_digest=f"sha256:{hashlib.sha256(b'mismatch').hexdigest()}",
        now=_now(),
    )
    assert result_mismatch.status == "invalid"
    assert result_mismatch.history_revision == first_succeeded.history_revision

    second_precommit = await repository.record_operation_precommit(
        "sequential",
        turn_id="turn-one",
        expected_history_revision=first_succeeded.history_revision,
        owner="worker",
        token="token",
        descriptor=second,
        now=_now(),
    )
    assert second_precommit.history_revision is not None
    assert second_precommit.timeline is not None
    assert second_precommit.timeline.turns[0].side_effect_boundary == "started"
    second_started = await repository.mark_side_effect_boundary(
        "sequential",
        turn_id="turn-one",
        expected_history_revision=second_precommit.history_revision,
        owner="worker",
        token="token",
        boundary="started",
        operation_id=second.operation_id,
        now=_now(),
    )
    assert second_started.history_revision is not None
    second_failed = await repository.record_operation_result(
        "sequential",
        turn_id="turn-one",
        expected_history_revision=second_started.history_revision,
        owner="worker",
        token="token",
        operation_id=second.operation_id,
        status="failed",
        result_digest=f"sha256:{hashlib.sha256(b'second-failure').hexdigest()}",
        now=_now(),
    )
    assert second_failed.history_revision is not None

    third = _descriptor("operation-replan")
    replan = await repository.record_operation_precommit(
        "sequential",
        turn_id="turn-one",
        expected_history_revision=second_failed.history_revision,
        owner="worker",
        token="token",
        descriptor=third,
        now=_now(),
    )
    assert replan.status == "precommitted"
    assert replan.history_revision is not None
    premature_complete = await repository.terminal(
        "sequential",
        turn_id="turn-one",
        expected_history_revision=replan.history_revision,
        owner="worker",
        token="token",
        outcome="completed",
        now=_now(),
    )
    assert premature_complete.status == "invalid"

    third_started = await repository.mark_side_effect_boundary(
        "sequential",
        turn_id="turn-one",
        expected_history_revision=replan.history_revision,
        owner="worker",
        token="token",
        boundary="started",
        operation_id=third.operation_id,
        now=_now(),
    )
    assert third_started.history_revision is not None
    third_succeeded = await repository.record_operation_result(
        "sequential",
        turn_id="turn-one",
        expected_history_revision=third_started.history_revision,
        owner="worker",
        token="token",
        operation_id=third.operation_id,
        status="succeeded",
        result_digest=f"sha256:{hashlib.sha256(b'third-result').hexdigest()}",
        now=_now(),
    )
    assert third_succeeded.history_revision is not None
    completed = await repository.terminal(
        "sequential",
        turn_id="turn-one",
        expected_history_revision=third_succeeded.history_revision,
        owner="worker",
        token="token",
        outcome="completed",
        now=_now(),
    )
    assert completed.status == "terminal"


@pytest.mark.asyncio
async def test_unknown_operation_blocks_later_precommit_and_requires_unknown_terminal(tmp_path) -> None:
    repository = SessionTurnTimelineRepository(SessionStore(tmp_path / "sessions"))
    admitted = await _admit(repository, "unknown", "turn-one")
    assert admitted.history_revision is not None
    claimed = await repository.claim_next(
        "unknown",
        expected_history_revision=admitted.history_revision,
        owner="worker",
        token="token",
        lease_expires_at=(_now() + timedelta(minutes=5)).isoformat(),
        now=_now(),
    )
    assert claimed.history_revision is not None
    first = _descriptor("operation-unknown")
    precommitted = await repository.record_operation_precommit(
        "unknown",
        turn_id="turn-one",
        expected_history_revision=claimed.history_revision,
        owner="worker",
        token="token",
        descriptor=first,
        now=_now(),
    )
    assert precommitted.history_revision is not None
    started = await repository.mark_side_effect_boundary(
        "unknown",
        turn_id="turn-one",
        expected_history_revision=precommitted.history_revision,
        owner="worker",
        token="token",
        boundary="started",
        operation_id=first.operation_id,
        now=_now(),
    )
    assert started.history_revision is not None
    unknown_result = await repository.record_operation_result(
        "unknown",
        turn_id="turn-one",
        expected_history_revision=started.history_revision,
        owner="worker",
        token="token",
        operation_id=first.operation_id,
        status="unknown",
        now=_now(),
    )
    assert unknown_result.history_revision is not None
    assert unknown_result.timeline is not None
    assert unknown_result.timeline.turns[0].side_effect_boundary == "unknown"
    blocked_next = await repository.record_operation_precommit(
        "unknown",
        turn_id="turn-one",
        expected_history_revision=unknown_result.history_revision,
        owner="worker",
        token="token",
        descriptor=_descriptor("operation-after-unknown"),
        now=_now(),
    )
    assert blocked_next.status == "invalid"
    complete = await repository.terminal(
        "unknown",
        turn_id="turn-one",
        expected_history_revision=unknown_result.history_revision,
        owner="worker",
        token="token",
        outcome="completed",
        now=_now(),
    )
    assert complete.status == "invalid"
    unknown_terminal = await repository.terminal(
        "unknown",
        turn_id="turn-one",
        expected_history_revision=unknown_result.history_revision,
        owner="worker",
        token="token",
        outcome="unknown",
        now=_now(),
    )
    assert unknown_terminal.status == "terminal"


@pytest.mark.asyncio
async def test_stale_started_operation_is_durably_quarantined_unknown_not_reclaimed(tmp_path) -> None:
    repository = SessionTurnTimelineRepository(SessionStore(tmp_path / "sessions"))
    first = await _admit(repository, "stale-recovery", "turn-one")
    assert first.history_revision is not None
    second = await repository.admit(
        "stale-recovery",
        turn_id="turn-two",
        expected_history_revision=first.history_revision,
    )
    assert second.history_revision is not None
    expiry = _now() + timedelta(minutes=1)
    claimed = await repository.claim_next(
        "stale-recovery",
        expected_history_revision=second.history_revision,
        owner="worker-one",
        token="token-one",
        lease_expires_at=expiry.isoformat(),
        now=_now(),
    )
    assert claimed.history_revision is not None
    descriptor = _descriptor("operation-stale")
    precommitted = await repository.record_operation_precommit(
        "stale-recovery",
        turn_id="turn-one",
        expected_history_revision=claimed.history_revision,
        owner="worker-one",
        token="token-one",
        descriptor=descriptor,
        now=_now(),
    )
    assert precommitted.history_revision is not None
    started = await repository.mark_side_effect_boundary(
        "stale-recovery",
        turn_id="turn-one",
        expected_history_revision=precommitted.history_revision,
        owner="worker-one",
        token="token-one",
        boundary="started",
        operation_id=descriptor.operation_id,
        now=_now(),
    )
    assert started.history_revision is not None
    blocked_claim = await repository.claim_next(
        "stale-recovery",
        expected_history_revision=started.history_revision,
        owner="worker-two",
        token="token-two",
        lease_expires_at=(_now() + timedelta(minutes=10)).isoformat(),
        now=expiry + timedelta(seconds=1),
    )
    assert blocked_claim.status == "lease_stale"
    quarantined = await repository.recover_stale_started_operation(
        "stale-recovery",
        turn_id="turn-one",
        expected_history_revision=started.history_revision,
        now=expiry + timedelta(seconds=1),
    )
    assert quarantined.status == "recovery_unknown"
    assert quarantined.history_revision is not None
    assert quarantined.timeline is not None
    terminal_turn = quarantined.timeline.turns[0]
    assert terminal_turn.status == "terminal"
    assert terminal_turn.terminal_outcome == "unknown"
    assert terminal_turn.operation_descriptors[0].status == "unknown"
    assert quarantined.timeline.lane_turn_id is None
    next_claim = await repository.claim_next(
        "stale-recovery",
        expected_history_revision=quarantined.history_revision,
        owner="worker-two",
        token="token-two",
        lease_expires_at=(_now() + timedelta(minutes=10)).isoformat(),
        now=expiry + timedelta(seconds=1),
    )
    assert next_claim.status == "claimed"
    assert next_claim.timeline is not None
    assert next_claim.timeline.lane_turn_id == "turn-two"


@pytest.mark.asyncio
async def test_v2_descriptor_reader_migrates_to_v4_without_result_evidence(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    repository = SessionTurnTimelineRepository(store)
    initial = await store.load_strict_snapshot("v2")
    descriptor = _descriptor("operation-v2", call_id="call-v2")
    v2_event = {
        "type": "session_meta",
        "event": SESSION_TURN_TIMELINE_EVENT,
        "schema_version": 1,
        "session_id": "v2",
        "timeline": {
            "timeline_version": "session-turn-timeline-v2",
            "session_id": "v2",
            "history_base_revision": 0,
            "history_current_revision": 1,
            "turns": [
                {
                    "sequence": 1,
                    "turn_id": "turn-v2",
                    "status": "running",
                    "side_effect_boundary": "started",
                    "operation_descriptors": [
                        {
                            "operation_id": descriptor.operation_id,
                            "tool_name": descriptor.tool_name,
                                "arguments_digest": f"sha256:{descriptor.arguments_digest}",
                            "call_id": descriptor.call_id,
                            "status": "side_effect_started",
                            "precommit_boundary": "started",
                        },
                    ],
                    "legacy_operation_ids": [],
                    "cancellation_outcome": None,
                    "terminal_outcome": None,
                },
            ],
            "lane_turn_id": "turn-v2",
            "lane_owner": "legacy-worker",
            "lane_token": "legacy-token",
            "lane_lease_expires_at": (_now() + timedelta(minutes=5)).isoformat(),
        },
        "timestamp": _now().isoformat(),
    }
    appended = await store.append_strict_batch_if_revision(
        "v2",
        expected_history_revision=initial.history_revision,
        events=(v2_event,),
    )
    assert appended.status == "appended"
    loaded = await repository.load("v2")
    assert loaded.status == "loaded"
    assert loaded.timeline is not None
    assert loaded.timeline.timeline_version == "session-turn-timeline-v4"
    migrated = loaded.timeline.turns[0].operation_descriptors[0]
    assert migrated.status == "started"
    assert migrated.result_digest is None
    assert migrated.receipt_reference is None
    assert loaded.history_revision is not None
    result = await repository.record_operation_result(
        "v2",
        turn_id="turn-v2",
        expected_history_revision=loaded.history_revision,
        owner="legacy-worker",
        token="legacy-token",
        operation_id=descriptor.operation_id,
        status="succeeded",
        receipt_reference="receipt-v2",
        now=_now(),
    )
    assert result.status == "operation_result"
    assert result.history_revision is not None
    reloaded = await repository.load("v2")
    assert reloaded.timeline is not None
    assert reloaded.timeline.timeline_version == "session-turn-timeline-v4"
    assert reloaded.timeline.turns[0].operation_descriptors[0].status == "succeeded"


@pytest.mark.asyncio
async def test_lease_renewal_requires_exact_live_identity_and_extends_expiry(tmp_path) -> None:
    repository = SessionTurnTimelineRepository(SessionStore(tmp_path / "sessions"))
    admitted = await _admit(repository, "renew", "turn-one")
    assert admitted.history_revision is not None
    original_expiry = _now() + timedelta(minutes=1)
    claimed = await repository.claim_next(
        "renew",
        expected_history_revision=admitted.history_revision,
        owner="worker",
        token="token",
        lease_expires_at=original_expiry.isoformat(),
        now=_now(),
    )
    assert claimed.history_revision is not None
    wrong_identity = await repository.renew_lease(
        "renew",
        turn_id="turn-one",
        expected_history_revision=claimed.history_revision,
        owner="other-worker",
        token="token",
        lease_expires_at=(_now() + timedelta(minutes=5)).isoformat(),
        now=_now(),
    )
    assert wrong_identity.status == "lease_invalid"
    renewed = await repository.renew_lease(
        "renew",
        turn_id="turn-one",
        expected_history_revision=claimed.history_revision,
        owner="worker",
        token="token",
        lease_expires_at=(_now() + timedelta(minutes=5)).isoformat(),
        now=_now(),
    )
    assert renewed.status == "lease_renewed"
    assert renewed.timeline is not None
    assert renewed.timeline.lane_lease_expires_at == (
        _now() + timedelta(minutes=5)
    ).isoformat()


@pytest.mark.asyncio
@pytest.mark.parametrize("with_precommit", [False, True])
async def test_stale_unstarted_lane_is_cancelled_without_replay(
    tmp_path,
    with_precommit: bool,
) -> None:
    repository = SessionTurnTimelineRepository(SessionStore(tmp_path / "sessions"))
    admitted = await _admit(repository, "stale-unstarted", "turn-one")
    assert admitted.history_revision is not None
    second = await repository.admit(
        "stale-unstarted",
        turn_id="turn-two",
        expected_history_revision=admitted.history_revision,
    )
    assert second.history_revision is not None
    expiry = _now() + timedelta(minutes=1)
    claimed = await repository.claim_next(
        "stale-unstarted",
        expected_history_revision=second.history_revision,
        owner="worker",
        token="token",
        lease_expires_at=expiry.isoformat(),
        now=_now(),
    )
    assert claimed.history_revision is not None
    revision = claimed.history_revision
    if with_precommit:
        precommitted = await repository.record_operation_precommit(
            "stale-unstarted",
            turn_id="turn-one",
            expected_history_revision=revision,
            owner="worker",
            token="token",
            descriptor=_descriptor("abandoned-operation"),
            now=_now(),
        )
        assert precommitted.history_revision is not None
        revision = precommitted.history_revision
    recovered = await repository.recover_stale_unstarted_turn(
        "stale-unstarted",
        turn_id="turn-one",
        expected_history_revision=revision,
        now=expiry + timedelta(seconds=1),
    )
    assert recovered.status == "recovery_cancelled"
    assert recovered.timeline is not None
    assert recovered.timeline.turns[0].terminal_outcome == "cancelled"
    assert recovered.timeline.turns[0].cancellation_outcome == "cancelled_running"
    assert recovered.timeline.lane_turn_id is None
    assert recovered.history_revision is not None
    next_claim = await repository.claim_next(
        "stale-unstarted",
        expected_history_revision=recovered.history_revision,
        owner="next-worker",
        token="next-token",
        lease_expires_at=(_now() + timedelta(minutes=10)).isoformat(),
        now=expiry + timedelta(seconds=1),
    )
    assert next_claim.status == "claimed"
    assert next_claim.timeline is not None
    assert next_claim.timeline.lane_turn_id == "turn-two"


@pytest.mark.asyncio
async def test_prior_unknown_quarantines_later_session_side_effects(tmp_path) -> None:
    repository = SessionTurnTimelineRepository(SessionStore(tmp_path / "sessions"))
    admitted = await _admit(repository, "quarantine", "turn-one")
    assert admitted.history_revision is not None
    claimed = await repository.claim_next(
        "quarantine",
        expected_history_revision=admitted.history_revision,
        owner="worker-one",
        token="token-one",
        lease_expires_at=(_now() + timedelta(minutes=5)).isoformat(),
        now=_now(),
    )
    assert claimed.history_revision is not None
    descriptor = _descriptor("unknown-operation")
    precommitted = await repository.record_operation_precommit(
        "quarantine",
        turn_id="turn-one",
        expected_history_revision=claimed.history_revision,
        owner="worker-one",
        token="token-one",
        descriptor=descriptor,
        now=_now(),
    )
    assert precommitted.history_revision is not None
    started = await repository.mark_side_effect_boundary(
        "quarantine",
        turn_id="turn-one",
        expected_history_revision=precommitted.history_revision,
        owner="worker-one",
        token="token-one",
        boundary="started",
        operation_id=descriptor.operation_id,
        now=_now(),
    )
    assert started.history_revision is not None
    unknown = await repository.record_operation_result(
        "quarantine",
        turn_id="turn-one",
        expected_history_revision=started.history_revision,
        owner="worker-one",
        token="token-one",
        operation_id=descriptor.operation_id,
        status="unknown",
        now=_now(),
    )
    assert unknown.history_revision is not None
    terminal = await repository.terminal(
        "quarantine",
        turn_id="turn-one",
        expected_history_revision=unknown.history_revision,
        owner="worker-one",
        token="token-one",
        outcome="unknown",
        now=_now(),
    )
    assert terminal.history_revision is not None
    admitted_second = await repository.admit(
        "quarantine",
        turn_id="turn-two",
        expected_history_revision=terminal.history_revision,
    )
    assert admitted_second.history_revision is not None
    claimed_second = await repository.claim_next(
        "quarantine",
        expected_history_revision=admitted_second.history_revision,
        owner="worker-two",
        token="token-two",
        lease_expires_at=(_now() + timedelta(minutes=5)).isoformat(),
        now=_now(),
    )
    assert claimed_second.history_revision is not None
    rejected = await repository.record_operation_precommit(
        "quarantine",
        turn_id="turn-two",
        expected_history_revision=claimed_second.history_revision,
        owner="worker-two",
        token="token-two",
        descriptor=_descriptor("operation-after-unknown"),
        now=_now(),
    )
    assert rejected.status == "invalid"
    assert "quarantines" in str(rejected.message)


@pytest.mark.asyncio
async def test_companion_events_share_the_timeline_cas_and_reject_cross_session(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    repository = SessionTurnTimelineRepository(store)
    loaded = await repository.load("atomic")
    assert loaded.history_revision is not None
    admitted = await repository.admit(
        "atomic",
        turn_id="turn-one",
        expected_history_revision=loaded.history_revision,
        companion_events=(
            {
                "type": "message",
                "session_id": "atomic",
                "turn_id": "turn-one",
                "role": "user",
                "content": "write the report",
            },
        ),
    )
    assert admitted.status == "admitted"
    assert admitted.timeline is not None
    assert admitted.timeline.history_current_revision == 2
    snapshot = await store.load_strict_snapshot("atomic")
    assert snapshot.event_count == 2
    assert snapshot.events[0]["type"] == "message"
    assert snapshot.events[1]["event"] == SESSION_TURN_TIMELINE_EVENT

    other = await repository.load("other")
    assert other.history_revision is not None
    with pytest.raises(ValueError, match="does not match"):
        await repository.admit(
            "other",
            turn_id="turn-other",
            expected_history_revision=other.history_revision,
            companion_events=(
                {"type": "message", "session_id": "different", "content": "bad"},
            ),
        )


@pytest.mark.asyncio
async def test_operation_result_and_tool_result_are_one_atomic_batch(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    repository = SessionTurnTimelineRepository(store)
    admitted = await _admit(repository, "atomic-result", "turn-one")
    assert admitted.history_revision is not None
    claimed = await repository.claim_next(
        "atomic-result",
        expected_history_revision=admitted.history_revision,
        owner="worker",
        token="token",
        lease_expires_at=(_now() + timedelta(minutes=5)).isoformat(),
        now=_now(),
    )
    assert claimed.history_revision is not None
    descriptor = _descriptor("atomic-operation")
    precommitted = await repository.record_operation_precommit(
        "atomic-result",
        turn_id="turn-one",
        expected_history_revision=claimed.history_revision,
        owner="worker",
        token="token",
        descriptor=descriptor,
        now=_now(),
    )
    assert precommitted.history_revision is not None
    started = await repository.mark_side_effect_boundary(
        "atomic-result",
        turn_id="turn-one",
        expected_history_revision=precommitted.history_revision,
        owner="worker",
        token="token",
        boundary="started",
        operation_id=descriptor.operation_id,
        now=_now(),
    )
    assert started.history_revision is not None
    before = await store.load_strict_snapshot("atomic-result")
    digest = f"sha256:{hashlib.sha256(b'tool-result').hexdigest()}"
    result = await repository.record_operation_result(
        "atomic-result",
        turn_id="turn-one",
        expected_history_revision=started.history_revision,
        owner="worker",
        token="token",
        operation_id=descriptor.operation_id,
        status="succeeded",
        result_digest=digest,
        companion_events=(
            {
                "type": "turn_event",
                "session_id": "atomic-result",
                "turn_id": "turn-one",
                "event_id": "turn-one:tool-result",
                "phase": "tool_call_result",
                "operation_id": descriptor.operation_id,
                "result_digest": digest,
            },
        ),
        now=_now(),
    )
    assert result.status == "operation_result"
    after = await store.load_strict_snapshot("atomic-result")
    assert after.event_count == before.event_count + 2
    assert after.events[-2]["operation_id"] == descriptor.operation_id
    assert after.events[-1]["event"] == SESSION_TURN_TIMELINE_EVENT
    assert result.timeline is not None
    assert result.timeline.history_current_revision == after.event_count


@pytest.mark.asyncio
async def test_terminal_and_final_transcript_are_one_atomic_batch(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    repository = SessionTurnTimelineRepository(store)
    admitted = await _admit(repository, "atomic-terminal", "turn-one")
    assert admitted.history_revision is not None
    claimed = await repository.claim_next(
        "atomic-terminal",
        expected_history_revision=admitted.history_revision,
        owner="worker",
        token="token",
        lease_expires_at=(_now() + timedelta(minutes=5)).isoformat(),
        now=_now(),
    )
    assert claimed.history_revision is not None
    before = await store.load_strict_snapshot("atomic-terminal")
    terminal = await repository.terminal(
        "atomic-terminal",
        turn_id="turn-one",
        expected_history_revision=claimed.history_revision,
        owner="worker",
        token="token",
        outcome="completed",
        companion_events=(
            {
                "type": "message",
                "session_id": "atomic-terminal",
                "turn_id": "turn-one",
                "role": "assistant",
                "content": "completed safely",
            },
        ),
        now=_now(),
    )
    assert terminal.status == "terminal"
    after = await store.load_strict_snapshot("atomic-terminal")
    assert after.event_count == before.event_count + 2
    assert after.events[-2]["content"] == "completed safely"
    assert after.events[-1]["event"] == SESSION_TURN_TIMELINE_EVENT
    assert terminal.timeline is not None
    assert terminal.timeline.lane_turn_id is None
    assert terminal.timeline.history_current_revision == after.event_count
