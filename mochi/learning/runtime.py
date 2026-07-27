"""Application-scoped lifecycle for background failure learning."""

from __future__ import annotations

from datetime import datetime

from mochi.learning.failure_episode import FailureEpisode
from mochi.learning.failure_outbox import FailureOutboxRepository
from mochi.learning.failure_store import FailureStore
from mochi.learning.failure_worker import FailureWorker, FailureWorkerBatchResult


class LearningRuntime:
    """Durable producer plus an explicitly owned optional worker."""

    def __init__(
        self,
        outbox: FailureOutboxRepository,
        store: FailureStore,
        *,
        enabled: bool = True,
        worker: FailureWorker | None = None,
    ) -> None:
        self._outbox = outbox
        self._store = store
        self._enabled = bool(enabled)
        self._worker = worker or FailureWorker(outbox, store)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def worker(self) -> FailureWorker:
        return self._worker

    async def submit(self, episode: FailureEpisode) -> bool:
        if not self._enabled:
            return False
        return await self._outbox.append_candidate(episode)

    async def process_once(self, *, now: datetime | None = None) -> FailureWorkerBatchResult:
        if not self._enabled:
            return FailureWorkerBatchResult()
        return await self._worker.process_once(now=now)

    async def start(self) -> None:
        if self._enabled:
            await self._worker.start()

    async def stop(self) -> None:
        await self._worker.stop()
