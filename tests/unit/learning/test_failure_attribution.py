from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from mochi.learning.failure_attribution import (
    FAILURE_ATTRIBUTION_EVENT,
    FailureAttributionError,
    FailureAttributionRecord,
    FailureAttributionRepository,
)
from mochi.learning.failure_episode import FailureEpisode
from mochi.learning.failure_outbox import (
    FAILURE_OUTBOX_SESSION_ID,
    FailureOutboxRepository,
)
from mochi.learning.failure_store import FailureStore
from mochi.learning.failure_worker import FailureWorker
from mochi.learning.runtime import FailureAdvisoryHint, LearningRuntime
from mochi.sessions.store import SessionStore


def _episode(
    *,
    session_id: str = "user-session-a",
    episode_id: str = "candidate-1",
    created_at: str = "2026-07-28T12:00:00+00:00",
) -> FailureEpisode:
    return FailureEpisode.candidate(
        session_id=session_id,
        turn_id="turn-1",
        capability_tags=("ordinary_chat", "verification"),
        tool_name="file_write",
        failure_signature="verification failed",
        reason_codes=("verification_failed",),
        verifier_feedback=("bounded feedback",),
        correction_attempted=False,
        correction_verified=False,
        episode_id=episode_id,
        idempotency_key=f"failure:{episode_id}",
        created_at=created_at,
    )


def test_failure_attribution_event_is_strict_versioned_and_redacted() -> None:
    record = FailureAttributionRecord.create(
        candidate_id="candidate-1",
        turn_id="turn-1",
        transition="candidate",
        timestamp="2026-07-28T12:00:00+00:00",
    )

    assert FailureAttributionRecord.from_event(record.to_event()) == record
    assert set(record.to_event()) == {
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
    with pytest.raises(FailureAttributionError, match="exact v1 keys"):
        FailureAttributionRecord.from_event(
            {
                **record.to_event(),
                "raw_prompt": "secret=must-not-persist",
            }
        )
    with pytest.raises(FailureAttributionError, match="unsupported"):
        FailureAttributionRecord.from_event(
            {**record.to_event(), "schema_version": 999}
        )


async def test_restart_processes_candidate_in_original_session_once(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions")
    episode = _episode()
    first_runtime = LearningRuntime(
        FailureOutboxRepository(store),
        FailureStore(store),
    )

    assert await first_runtime.submit(
        episode,
        attribution_session_id="user-session-a",
    )
    assert (
        await first_runtime.submit(
            episode,
            attribution_session_id="user-session-a",
        )
        is False
    )

    # Recreate every learning component to prove the original session target
    # is recovered from durable outbox state rather than process memory.
    restarted = LearningRuntime(
        FailureOutboxRepository(store),
        FailureStore(store),
    )
    result = await restarted.process_once(now=datetime(2026, 7, 28, 13, tzinfo=UTC))
    assert result.acked == 1
    assert (
        await restarted.process_once(now=datetime(2026, 7, 28, 14, tzinfo=UTC))
    ).claimed == 0

    user_events = await store.load_session("user-session-a")
    transitions = [
        event["transition"]
        for event in user_events
        if event.get("event") == FAILURE_ATTRIBUTION_EVENT
    ]
    assert transitions == ["candidate", "processed"]
    global_events = await store.load_session(FAILURE_OUTBOX_SESSION_ID)
    assert global_events[0]["attribution_session_id"] == "user-session-a"


async def test_retry_emits_only_terminal_rejection_and_stays_session_scoped(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions")
    outbox = FailureOutboxRepository(store)
    attribution = FailureAttributionRepository(store)
    runtime = LearningRuntime(
        outbox,
        FailureStore(store),
        attribution_repository=attribution,
    )
    episode = _episode(session_id="user-session-b")
    assert await runtime.submit(
        episode,
        attribution_session_id="user-session-b",
    )

    async def reject(_: FailureEpisode) -> None:
        raise RuntimeError("poison")

    worker = FailureWorker(
        outbox,
        FailureStore(store),
        worker_id="rejecting-worker",
        batch_size=1,
        max_attempts=2,
        backoff_base_seconds=0,
        processor=reject,
        attribution_repository=attribution,
    )
    now = datetime(2026, 7, 28, 13, tzinfo=UTC)
    assert (await worker.process_once(now=now)).retried == 1
    assert (await worker.process_once(now=now)).rejected == 1

    transitions = [
        event["transition"]
        for event in await store.load_session("user-session-b")
        if event.get("event") == FAILURE_ATTRIBUTION_EVENT
    ]
    assert transitions == ["candidate", "rejected"]
    assert await store.load_session("user-session-a") == []


async def test_attribution_failure_is_observational_and_counted(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions")

    class BrokenAttribution:
        async def append(self, *_: Any, **__: Any) -> bool:
            raise OSError("telemetry unavailable")

    runtime = LearningRuntime(
        FailureOutboxRepository(store),
        FailureStore(store),
        attribution_repository=BrokenAttribution(),  # type: ignore[arg-type]
    )
    assert await runtime.submit(
        _episode(),
        attribution_session_id="user-session-a",
    )
    assert runtime.attribution_failure_count == 1
    assert len(await FailureOutboxRepository(store).list_records()) == 1


async def test_invalid_attribution_target_preserves_global_candidate(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions")
    runtime = LearningRuntime(
        FailureOutboxRepository(store),
        FailureStore(store),
    )

    assert await runtime.submit(
        _episode(session_id="user-session-a"),
        attribution_session_id="user-session-b",
    )
    assert runtime.attribution_failure_count == 1
    assert len(await FailureOutboxRepository(store).list_records()) == 1
    assert await store.load_session("user-session-b") == []


async def test_hint_selection_attribution_is_idempotent_per_candidate_turn(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions")
    runtime = LearningRuntime(
        FailureOutboxRepository(store),
        FailureStore(store),
    )
    selections = (
        FailureAdvisoryHint(
            candidate_id="candidate-1",
            text="bounded verified correction",
        ),
    )

    assert (
        await runtime.record_hint_selections(
            session_id="hint-session",
            turn_id="turn-hint",
            selections=selections,
        )
        == 1
    )
    assert (
        await runtime.record_hint_selections(
            session_id="hint-session",
            turn_id="turn-hint",
            selections=selections,
        )
        == 0
    )
    events = [
        event
        for event in await store.load_session("hint-session")
        if event.get("event") == FAILURE_ATTRIBUTION_EVENT
    ]
    assert len(events) == 1
    assert events[0]["transition"] == "hint_selected"
