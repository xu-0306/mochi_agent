from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from mochi.sessions.store import SessionStore
from mochi.sessions.timeline_coordinator import TimelineCoordinator
from mochi.sessions.turn_timeline import (
    SESSION_TURN_TIMELINE_EVENT,
    SessionTurnTimelineRepository,
)


def _now() -> datetime:
    return datetime(2026, 7, 25, 9, 0, tzinfo=UTC)


async def _admit_owned(
    repository: SessionTurnTimelineRepository,
    session_id: str,
    turn_id: str,
    *,
    owner: str,
    token: str,
    expires_at: datetime,
) -> str:
    loaded = await repository.load(session_id)
    assert loaded.history_revision is not None
    result = await repository.admit(
        session_id,
        turn_id=turn_id,
        expected_history_revision=loaded.history_revision,
        admission_owner=owner,
        admission_token=token,
        admission_lease_expires_at=expires_at.isoformat(),
        now=_now(),
    )
    assert result.status == "admitted"
    assert result.history_revision is not None
    return result.history_revision


@pytest.mark.asyncio
async def test_live_queued_admission_cannot_be_claimed_or_recovered_by_another_owner(tmp_path) -> None:
    repository = SessionTurnTimelineRepository(SessionStore(tmp_path / "sessions"))
    revision = await _admit_owned(
        repository,
        "live-admission",
        "turn-one",
        owner="admitter-one",
        token="admission-token-one",
        expires_at=_now() + timedelta(minutes=1),
    )
    duplicate = await repository.admit(
        "live-admission",
        turn_id="turn-one",
        expected_history_revision=revision,
        admission_owner="admitter-one",
        admission_token="admission-token-one",
        admission_lease_expires_at=(_now() + timedelta(minutes=1)).isoformat(),
        now=_now(),
    )
    assert duplicate.status == "duplicate"
    mismatched = await repository.admit(
        "live-admission",
        turn_id="turn-one",
        expected_history_revision=revision,
        admission_owner="another-admitter",
        admission_token="another-token",
        admission_lease_expires_at=(_now() + timedelta(minutes=1)).isoformat(),
        now=_now(),
    )
    assert mismatched.status == "admission_invalid"

    stolen = await repository.claim_next(
        "live-admission",
        expected_history_revision=revision,
        owner="worker-two",
        token="worker-token-two",
        lease_expires_at=(_now() + timedelta(minutes=2)).isoformat(),
        now=_now(),
    )
    assert stolen.status == "admission_busy"
    assert stolen.history_revision == revision

    recovery = await repository.recover_expired_queued_admission(
        "live-admission",
        turn_id="turn-one",
        expected_history_revision=revision,
        now=_now(),
    )
    assert recovery.status == "admission_invalid"
    assert recovery.history_revision == revision


@pytest.mark.asyncio
async def test_expired_queued_admission_is_cancelled_and_following_turn_can_progress(tmp_path) -> None:
    repository = SessionTurnTimelineRepository(SessionStore(tmp_path / "sessions"))
    expired_at = _now() + timedelta(seconds=1)
    first_revision = await _admit_owned(
        repository,
        "orphaned-admission",
        "turn-crashed",
        owner="crashed-admitter",
        token="crashed-token",
        expires_at=expired_at,
    )
    second_revision = await _admit_owned(
        repository,
        "orphaned-admission",
        "turn-following",
        owner="following-admitter",
        token="following-token",
        expires_at=_now() + timedelta(minutes=1),
    )
    assert second_revision != first_revision

    recovered = await repository.recover_expired_queued_admission(
        "orphaned-admission",
        turn_id="turn-crashed",
        expected_history_revision=second_revision,
        now=expired_at + timedelta(seconds=1),
    )
    assert recovered.status == "recovery_admission_cancelled"
    assert recovered.history_revision is not None
    assert recovered.timeline is not None
    assert recovered.timeline.turns[0].terminal_outcome == "cancelled"
    assert recovered.timeline.turns[0].cancellation_outcome == "cancelled_queued"
    assert (
        recovered.timeline.turns[0].recovery_reason
        == "admission_owner_expired_before_claim"
    )

    claimed = await repository.claim_next(
        "orphaned-admission",
        expected_history_revision=recovered.history_revision,
        owner="following-admitter",
        token="following-token",
        lease_expires_at=(expired_at + timedelta(minutes=2)).isoformat(),
        now=expired_at + timedelta(seconds=1),
    )
    assert claimed.status == "claimed"
    assert claimed.timeline is not None
    assert claimed.timeline.lane_turn_id == "turn-following"


@pytest.mark.asyncio
async def test_concurrent_expired_admission_recovery_has_one_cas_winner(tmp_path) -> None:
    sessions = tmp_path / "sessions"
    first = SessionTurnTimelineRepository(SessionStore(sessions))
    second = SessionTurnTimelineRepository(SessionStore(sessions))
    revision = await _admit_owned(
        first,
        "admission-cas",
        "turn-crashed",
        owner="crashed-admitter",
        token="crashed-token",
        expires_at=_now() + timedelta(seconds=1),
    )
    recovery_now = _now() + timedelta(seconds=2)
    first_result, second_result = await asyncio.gather(
        first.recover_expired_queued_admission(
            "admission-cas",
            turn_id="turn-crashed",
            expected_history_revision=revision,
            now=recovery_now,
        ),
        second.recover_expired_queued_admission(
            "admission-cas",
            turn_id="turn-crashed",
            expected_history_revision=revision,
            now=recovery_now,
        ),
    )
    assert sorted((first_result.status, second_result.status)) == [
        "rebase_required",
        "recovery_admission_cancelled",
    ]
    loaded = await first.load("admission-cas")
    assert loaded.timeline is not None
    assert loaded.timeline.turns[0].status == "terminal"


@pytest.mark.asyncio
async def test_restart_reconciles_only_expired_admission_without_replaying_orphan_history(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mochi.sessions.timeline_coordinator as coordinator_module

    class _Clock:
        current = _now()

        @classmethod
        def now(cls, tz: object | None = None) -> datetime:
            assert tz is UTC
            return cls.current

        @staticmethod
        def fromisoformat(value: str) -> datetime:
            return datetime.fromisoformat(value)

    monkeypatch.setattr(coordinator_module, "datetime", _Clock)
    store = SessionStore(tmp_path / "sessions")
    completed = TimelineCoordinator(
        session_store=store,
        session_id="restart-admission",
        turn_id="turn-completed",
        lease_seconds=6,
    )
    await completed.admit_user_message(
        {
            "type": "message",
            "session_id": "restart-admission",
            "turn_id": "turn-completed",
            "role": "user",
            "content": "completed predecessor request",
        }
    )
    await completed.claim()
    await completed.finish(
        companion_events=(
            {
                "type": "message",
                "session_id": "restart-admission",
                "turn_id": "turn-completed",
                "role": "assistant",
                "content": "completed predecessor response",
            },
        )
    )
    crashed = TimelineCoordinator(
        session_store=store,
        session_id="restart-admission",
        turn_id="turn-crashed",
        lease_seconds=6,
    )
    await crashed.admit_user_message(
        {
            "type": "message",
            "session_id": "restart-admission",
            "turn_id": "turn-crashed",
            "role": "user",
            "content": "request lost before claim",
        }
    )
    _Clock.current = _now() + timedelta(seconds=7)
    following = TimelineCoordinator(
        session_store=store,
        session_id="restart-admission",
        turn_id="turn-following",
        lease_seconds=6,
    )
    await following.admit_user_message(
        {
            "type": "message",
            "session_id": "restart-admission",
            "turn_id": "turn-following",
            "role": "user",
            "content": "follow-up request",
        }
    )

    history = await asyncio.wait_for(following.claim(), timeout=1)
    history_contents = {str(event.get("content")) for event in history}
    assert "completed predecessor request" in history_contents
    assert "completed predecessor response" in history_contents
    assert "request lost before claim" not in history_contents
    loaded = await SessionTurnTimelineRepository(store).load("restart-admission")
    assert loaded.timeline is not None
    assert loaded.timeline.turns[1].terminal_outcome == "cancelled"
    assert (
        loaded.timeline.turns[1].recovery_reason
        == "admission_owner_expired_before_claim"
    )
    assert loaded.timeline.lane_turn_id == "turn-following"
    await following.finish()


@pytest.mark.asyncio
async def test_active_lane_blocks_expired_queued_admission_recovery(tmp_path) -> None:
    repository = SessionTurnTimelineRepository(SessionStore(tmp_path / "sessions"))
    first_revision = await _admit_owned(
        repository,
        "active-lane-admission",
        "turn-running",
        owner="running-admitter",
        token="running-token",
        expires_at=_now() + timedelta(minutes=1),
    )
    claimed = await repository.claim_next(
        "active-lane-admission",
        expected_history_revision=first_revision,
        owner="running-admitter",
        token="running-token",
        lease_expires_at=(_now() + timedelta(minutes=2)).isoformat(),
        now=_now(),
    )
    assert claimed.history_revision is not None
    queued_revision = await _admit_owned(
        repository,
        "active-lane-admission",
        "turn-queued",
        owner="queued-admitter",
        token="queued-token",
        expires_at=_now() + timedelta(seconds=1),
    )
    assert queued_revision != claimed.history_revision

    recovery = await repository.recover_expired_queued_admission(
        "active-lane-admission",
        turn_id="turn-queued",
        expected_history_revision=queued_revision,
        now=_now() + timedelta(seconds=2),
    )
    assert recovery.status == "admission_invalid"
    assert recovery.history_revision == queued_revision
    loaded = await repository.load("active-lane-admission")
    assert loaded.timeline is not None
    assert loaded.timeline.lane_turn_id == "turn-running"
    assert loaded.timeline.turns[1].status == "queued"


@pytest.mark.asyncio
async def test_claim_wait_renews_its_queued_admission_before_the_deadline(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mochi.sessions.timeline_coordinator as coordinator_module

    class _Clock:
        current = _now()

        @classmethod
        def now(cls, tz: object | None = None) -> datetime:
            assert tz is UTC
            return cls.current

        @staticmethod
        def fromisoformat(value: str) -> datetime:
            return datetime.fromisoformat(value)

    monkeypatch.setattr(coordinator_module, "datetime", _Clock)
    store = SessionStore(tmp_path / "sessions")
    first = TimelineCoordinator(
        session_store=store,
        session_id="admission-heartbeat",
        turn_id="turn-running",
        lease_seconds=6,
    )
    await first.admit_user_message(
        {
            "type": "message",
            "session_id": "admission-heartbeat",
            "turn_id": "turn-running",
            "role": "user",
            "content": "running request",
        }
    )
    await first.claim()
    waiting = TimelineCoordinator(
        session_store=store,
        session_id="admission-heartbeat",
        turn_id="turn-waiting",
        lease_seconds=6,
        poll_seconds=0.01,
    )
    await waiting.admit_user_message(
        {
            "type": "message",
            "session_id": "admission-heartbeat",
            "turn_id": "turn-waiting",
            "role": "user",
            "content": "queued request",
        }
    )
    before = await SessionTurnTimelineRepository(store).load("admission-heartbeat")
    assert before.timeline is not None
    original_expiry = before.timeline.turns[1].admission_lease_expires_at
    _Clock.current = _now() + timedelta(seconds=5)

    claim_task = asyncio.create_task(waiting.claim())
    await asyncio.sleep(0.05)
    claim_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await claim_task
    after = await SessionTurnTimelineRepository(store).load("admission-heartbeat")
    assert after.timeline is not None
    renewed_expiry = after.timeline.turns[1].admission_lease_expires_at
    assert renewed_expiry is not None
    assert original_expiry is not None
    assert datetime.fromisoformat(renewed_expiry) > datetime.fromisoformat(original_expiry)
    assert after.timeline.turns[1].status == "queued"
    await waiting.request_cancel()
    await first.finish()


@pytest.mark.asyncio
async def test_v3_queued_rows_remain_readable_but_are_not_age_recovered_without_admission_proof(
    tmp_path,
) -> None:
    store = SessionStore(tmp_path / "sessions")
    snapshot = await store.load_strict_snapshot("legacy-admission")
    appended = await store.append_strict_batch_if_revision(
        "legacy-admission",
        expected_history_revision=snapshot.history_revision,
        events=(
            {
                "type": "session_meta",
                "event": SESSION_TURN_TIMELINE_EVENT,
                "schema_version": 1,
                "session_id": "legacy-admission",
                "timeline": {
                    "timeline_version": "session-turn-timeline-v3",
                    "session_id": "legacy-admission",
                    "history_base_revision": 0,
                    "history_current_revision": 1,
                    "turns": [
                        {
                            "sequence": 1,
                            "turn_id": "legacy-queued",
                            "status": "queued",
                            "side_effect_boundary": "not_started",
                            "operation_descriptors": [],
                            "legacy_operation_ids": [],
                            "cancellation_outcome": None,
                            "terminal_outcome": None,
                        }
                    ],
                    "lane_turn_id": None,
                    "lane_owner": None,
                    "lane_token": None,
                    "lane_lease_expires_at": None,
                },
                "timestamp": _now().isoformat(),
            },
        ),
    )
    assert appended.status == "appended"
    repository = SessionTurnTimelineRepository(store)
    loaded = await repository.load("legacy-admission")
    assert loaded.timeline is not None
    legacy_turn = loaded.timeline.turns[0]
    assert legacy_turn.admission_owner is None
    assert loaded.history_revision is not None

    recovery = await repository.recover_expired_queued_admission(
        "legacy-admission",
        turn_id="legacy-queued",
        expected_history_revision=loaded.history_revision,
        now=_now() + timedelta(days=365),
    )
    assert recovery.status == "admission_invalid"
    assert recovery.history_revision == loaded.history_revision
