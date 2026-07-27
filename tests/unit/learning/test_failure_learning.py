from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from mochi.learning.failure_episode import (
    FAILURE_EPISODE_VERSION,
    FailureEpisode,
    FailureEpisodeError,
)
from mochi.learning.failure_outbox import FailureOutboxRepository
from mochi.learning.failure_store import FailureStore
from mochi.learning.failure_worker import FailureWorker
from mochi.learning.runtime import LearningRuntime


class _MemorySessionStore:
    def __init__(self) -> None:
        self.events: dict[str, list[dict[str, Any]]] = {}

    async def load_session(self, session_id: str) -> list[dict[str, Any]]:
        return [dict(event) for event in self.events.get(session_id, [])]

    async def append_event_if(
        self,
        session_id: str,
        event: dict[str, Any],
        predicate: Callable[[list[dict[str, Any]]], bool],
    ) -> bool:
        current = self.events.setdefault(session_id, [])
        if not predicate([dict(item) for item in current]):
            return False
        current.append(dict(event))
        return True


def _episode(
    *,
    episode_id: str = "episode-1",
    idempotency_key: str | None = None,
    created_at: str = "2026-07-27T10:00:00+00:00",
    correction_attempted: bool = False,
    correction_verified: bool = False,
    feedback: tuple[str, ...] = ("expected output differs",),
) -> FailureEpisode:
    return FailureEpisode.candidate(
        session_id="session/with/private@example.com",
        turn_id="turn-1",
        capability_tags=("file_mutation", "verification"),
        tool_name="file_edit",
        failure_signature="expected output differs for api_key=super-secret",
        reason_codes=("target_mismatch",),
        verifier_feedback=feedback,
        correction_attempted=correction_attempted,
        correction_verified=correction_verified,
        episode_id=episode_id,
        idempotency_key=idempotency_key or f"failure:{episode_id}",
        created_at=created_at,
    )


def test_failure_episode_is_redacted_hashed_and_versioned() -> None:
    episode = _episode(feedback=("contact alice@example.com token=secret-value",))

    assert episode.episode_version == FAILURE_EPISODE_VERSION
    assert len(episode.session_id_hash) == 64
    assert episode.failure_signature.startswith("failure:v1:")
    assert "alice@example.com" not in episode.verifier_feedback[0]
    assert "secret-value" not in episode.verifier_feedback[0]
    assert "api_key" not in episode.failure_signature
    assert FailureEpisode.from_dict(episode.to_dict()) == episode


def test_failure_episode_rejects_future_fields_and_raw_signature() -> None:
    episode = _episode()

    with pytest.raises(FailureEpisodeError, match="unexpected fields"):
        FailureEpisode.from_dict({**episode.to_dict(), "future": True})
    with pytest.raises(FailureEpisodeError, match="normalized digest"):
        FailureEpisode(
            **{**episode.to_dict(), "failure_signature": "raw failure"}  # type: ignore[arg-type]
        )


async def test_outbox_is_idempotent_and_acknowledges_a_claim() -> None:
    store = _MemorySessionStore()
    outbox = FailureOutboxRepository(store)
    episode = _episode()

    assert await outbox.append_candidate(episode) is True
    assert await outbox.append_candidate(episode) is False
    records = await outbox.list_records()
    assert len(records) == 1
    assert records[0].status == "pending"

    claimed = await outbox.claim(
        worker_id="worker-1",
        now=datetime(2026, 7, 28, tzinfo=UTC),
    )
    assert claimed is not None
    assert claimed.status == "claimed"
    assert claimed.attempts == 1
    assert await outbox.ack(episode.episode_id, worker_id="worker-1") is True
    assert (await outbox.list_records())[0].status == "acked"
    assert await outbox.ack(episode.episode_id, worker_id="worker-1") is False


async def test_expired_lease_is_replayable_after_worker_crash() -> None:
    store = _MemorySessionStore()
    outbox = FailureOutboxRepository(store)
    episode = _episode()
    started = datetime(2026, 7, 27, 10, 0, tzinfo=UTC)
    await outbox.append_candidate(episode)

    claimed = await outbox.claim(
        worker_id="crashed-worker",
        lease_seconds=1,
        now=started,
    )
    assert claimed is not None
    reclaimed = await outbox.claim(
        worker_id="replacement-worker",
        lease_seconds=1,
        now=started + timedelta(seconds=2),
    )

    assert reclaimed is not None
    assert reclaimed.lease_owner == "replacement-worker"
    assert reclaimed.attempts == 2


async def test_worker_retries_then_processes_without_a_request_path_model_call() -> None:
    session_store = _MemorySessionStore()
    outbox = FailureOutboxRepository(session_store)
    failure_store = FailureStore(session_store)
    episode = _episode()
    await outbox.append_candidate(episode)
    calls = 0

    async def flaky_processor(candidate: FailureEpisode) -> None:
        nonlocal calls
        assert candidate == episode
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary worker failure")
        await failure_store.record(candidate)

    worker = FailureWorker(
        outbox,
        failure_store,
        worker_id="worker-1",
        batch_size=1,
        backoff_base_seconds=0,
        max_attempts=3,
        processor=flaky_processor,
    )
    run_at = datetime(2026, 7, 28, tzinfo=UTC)
    first = await worker.process_once(now=run_at)
    second = await worker.process_once(now=run_at)

    assert first.retried == 1
    assert second.acked == 1
    assert calls == 2
    assert (await outbox.list_records())[0].status == "acked"
    assert (await failure_store.aggregates())[0].occurrence_count == 1


async def test_poison_candidate_is_rejected_after_bounded_attempts() -> None:
    session_store = _MemorySessionStore()
    outbox = FailureOutboxRepository(session_store)
    failure_store = FailureStore(session_store)
    await outbox.append_candidate(_episode())

    async def broken_processor(_: FailureEpisode) -> None:
        raise ValueError("malformed candidate")

    worker = FailureWorker(
        outbox,
        failure_store,
        worker_id="worker-1",
        max_attempts=1,
        processor=broken_processor,
    )
    result = await worker.process_once(now=datetime(2026, 7, 28, tzinfo=UTC))

    assert result.rejected == 1
    assert (await outbox.list_records())[0].status == "rejected"
    assert await failure_store.aggregates() == ()


async def test_repeated_verified_correction_becomes_hint_eligible() -> None:
    session_store = _MemorySessionStore()
    failure_store = FailureStore(session_store)
    first = _episode(
        episode_id="episode-1",
        correction_attempted=True,
        correction_verified=True,
        feedback=("corrected target",),
    )
    second = _episode(
        episode_id="episode-2",
        idempotency_key="failure:episode-2",
        created_at="2026-07-27T11:00:00+00:00",
        correction_attempted=False,
        correction_verified=False,
    )
    assert await failure_store.record(first) is True
    assert await failure_store.record(second) is True
    assert await failure_store.record(first) is False

    aggregate = (await failure_store.aggregates())[0]
    hints = await failure_store.hint_candidates(min_occurrences=2)

    assert aggregate.occurrence_count == 2
    assert aggregate.verified_correction_count == 1
    assert len(hints) == 1
    assert hints[0].verified_correction_summary == "corrected target"


async def test_learning_runtime_persists_pending_work_and_has_owned_lifecycle() -> None:
    session_store = _MemorySessionStore()
    outbox = FailureOutboxRepository(session_store)
    failure_store = FailureStore(session_store)
    runtime = LearningRuntime(outbox, failure_store)

    assert await runtime.submit(_episode()) is True
    assert runtime.worker.running is False
    await runtime.start()
    await runtime.stop()
    assert runtime.worker.running is False
    assert (await outbox.list_records())[0].status in {"acked", "pending"}
