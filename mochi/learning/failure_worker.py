"""Owned background worker for the failure-learning outbox."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from mochi.learning.failure_attribution import (
    FailureAttributionRecord,
    FailureAttributionRepository,
)
from mochi.learning.failure_episode import FailureEpisode
from mochi.learning.failure_outbox import FailureOutboxRepository
from mochi.learning.failure_store import FailureStore

FailureProcessor = Callable[[FailureEpisode], Awaitable[None] | None]


@dataclass(frozen=True)
class FailureWorkerBatchResult:
    claimed: int = 0
    acked: int = 0
    retried: int = 0
    rejected: int = 0


class FailureWorker:
    """Process candidates with leases, retry backoff, and poison rejection."""

    def __init__(
        self,
        outbox: FailureOutboxRepository,
        store: FailureStore,
        *,
        worker_id: str | None = None,
        batch_size: int = 10,
        poll_interval_seconds: float = 1.0,
        max_attempts: int = 3,
        backoff_base_seconds: float = 1.0,
        processor: FailureProcessor | None = None,
        attribution_repository: FailureAttributionRepository | None = None,
    ) -> None:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be positive")
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if poll_interval_seconds <= 0 or backoff_base_seconds < 0:
            raise ValueError("worker timing values are invalid")
        self._outbox = outbox
        self._store = store
        self._worker_id = worker_id or f"failure-worker-{uuid4().hex}"
        self._batch_size = batch_size
        self._poll_interval_seconds = poll_interval_seconds
        self._max_attempts = max_attempts
        self._backoff_base_seconds = backoff_base_seconds
        self._processor = processor
        self._attribution_repository = attribution_repository
        self._attribution_failure_count = 0
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    @property
    def worker_id(self) -> str:
        return self._worker_id

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def attribution_failure_count(self) -> int:
        """Best-effort attribution failures that never alter worker state."""

        return self._attribution_failure_count

    async def _attribute_terminal(
        self,
        record: object,
        *,
        transition: str,
        timestamp: str,
    ) -> None:
        repository = self._attribution_repository
        session_id = getattr(record, "attribution_session_id", None)
        episode = getattr(record, "episode", None)
        if (
            repository is None
            or not isinstance(session_id, str)
            or not isinstance(episode, FailureEpisode)
        ):
            return
        try:
            await repository.append(
                session_id,
                FailureAttributionRecord.create(
                    candidate_id=episode.episode_id,
                    turn_id=episode.turn_id,
                    transition=transition,  # type: ignore[arg-type]
                    timestamp=timestamp,
                ),
            )
        except Exception:
            self._attribution_failure_count += 1

    async def process_once(self, *, now: datetime | None = None) -> FailureWorkerBatchResult:
        result = FailureWorkerBatchResult()
        current = now or datetime.now(tz=UTC)
        for _ in range(self._batch_size):
            claimed = await self._outbox.claim(worker_id=self._worker_id, now=current)
            if claimed is None:
                break
            result = FailureWorkerBatchResult(
                claimed=result.claimed + 1,
                acked=result.acked,
                retried=result.retried,
                rejected=result.rejected,
            )
            try:
                processor = self._processor or self._store.record
                processed = processor(claimed.episode)
                if inspect.isawaitable(processed):
                    await processed
            except Exception as exc:
                if claimed.attempts >= self._max_attempts:
                    rejected = await self._outbox.reject(
                        claimed.episode.episode_id,
                        worker_id=self._worker_id,
                        reason=f"processor_failed:{type(exc).__name__}",
                        now=current,
                    )
                    if rejected:
                        await self._attribute_terminal(
                            claimed,
                            transition="rejected",
                            timestamp=current.isoformat(),
                        )
                    result = FailureWorkerBatchResult(
                        claimed=result.claimed,
                        acked=result.acked,
                        retried=result.retried,
                        rejected=result.rejected + int(rejected),
                    )
                else:
                    backoff = self._backoff_base_seconds * (2 ** max(0, claimed.attempts - 1))
                    await self._outbox.retry(
                        claimed.episode.episode_id,
                        worker_id=self._worker_id,
                        error=f"processor_failed:{type(exc).__name__}",
                        backoff_seconds=backoff,
                        now=current,
                    )
                    result = FailureWorkerBatchResult(
                        claimed=result.claimed,
                        acked=result.acked,
                        retried=result.retried + 1,
                        rejected=result.rejected,
                    )
            else:
                acked = await self._outbox.ack(
                    claimed.episode.episode_id,
                    worker_id=self._worker_id,
                    now=current,
                )
                if acked:
                    await self._attribute_terminal(
                        claimed,
                        transition="processed",
                        timestamp=current.isoformat(),
                    )
                result = FailureWorkerBatchResult(
                    claimed=result.claimed,
                    acked=result.acked + int(acked),
                    retried=result.retried,
                    rejected=result.rejected,
                )
        return result

    async def run(self) -> None:
        while not self._stop_event.is_set():
            await self.process_once()
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._poll_interval_seconds,
                )
            except TimeoutError:
                continue

    async def start(self) -> None:
        if self.running:
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self.run(), name=f"{self._worker_id}-loop")

    async def stop(self) -> None:
        task = self._task
        if task is None:
            return
        self._stop_event.set()
        try:
            await task
        finally:
            self._task = None
