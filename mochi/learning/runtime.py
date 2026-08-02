"""Application-scoped lifecycle for background failure learning."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from mochi.learning.failure_attribution import (
    FailureAttributionRecord,
    FailureAttributionRepository,
)
from mochi.learning.failure_episode import FailureEpisode
from mochi.learning.failure_outbox import FailureOutboxRepository
from mochi.learning.failure_outbox import FailureOutboxError
from mochi.learning.failure_store import FailureStore
from mochi.learning.failure_worker import FailureWorker, FailureWorkerBatchResult


@dataclass(frozen=True)
class FailureAdvisoryHint:
    """A bounded verified hint plus its redacted source candidate identity."""

    candidate_id: str
    text: str


class LearningRuntime:
    """Durable producer plus an explicitly owned optional worker."""

    def __init__(
        self,
        outbox: FailureOutboxRepository,
        store: FailureStore,
        *,
        enabled: bool = True,
        worker: FailureWorker | None = None,
        attribution_repository: FailureAttributionRepository | None = None,
    ) -> None:
        self._outbox = outbox
        self._store = store
        self._enabled = bool(enabled)
        self._attribution_repository = (
            attribution_repository
            or FailureAttributionRepository(outbox.session_store)
        )
        self._worker = worker or FailureWorker(
            outbox,
            store,
            attribution_repository=self._attribution_repository,
        )
        self._attribution_failure_count = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def worker(self) -> FailureWorker:
        return self._worker

    @property
    def attribution_failure_count(self) -> int:
        return (
            self._attribution_failure_count
            + self._worker.attribution_failure_count
        )

    async def submit(
        self,
        episode: FailureEpisode,
        *,
        attribution_session_id: str | None = None,
    ) -> bool:
        if not self._enabled:
            return False
        try:
            appended = await self._outbox.append_candidate(
                episode,
                attribution_session_id=attribution_session_id,
            )
        except FailureOutboxError:
            if attribution_session_id is None:
                raise
            # Preserve durable work if only the attribution target is invalid.
            # The worker can still process the redacted episode, while the
            # failure counter exposes the missing observational join.
            self._attribution_failure_count += 1
            appended = await self._outbox.append_candidate(episode)
            return appended
        if attribution_session_id is not None:
            try:
                await self._attribution_repository.append(
                    attribution_session_id,
                    FailureAttributionRecord.create(
                        candidate_id=episode.episode_id,
                        turn_id=episode.turn_id,
                        transition="candidate",
                        timestamp=episode.created_at,
                    ),
                )
            except Exception:
                # The durable global candidate remains authoritative work.
                # Attribution is observational and must not change Chat or
                # worker processing semantics.
                self._attribution_failure_count += 1
        return appended

    async def process_once(self, *, now: datetime | None = None) -> FailureWorkerBatchResult:
        if not self._enabled:
            return FailureWorkerBatchResult()
        return await self._worker.process_once(now=now)

    async def advisory_hints(
        self,
        *,
        enabled: bool = False,
        min_occurrences: int = 2,
        max_hints: int = 2,
        max_hint_chars: int = 800,
        available_tool_names: set[str] | None = None,
    ) -> tuple[str, ...]:
        """Return bounded telemetry hints only for an explicit rollout.

        Hints are plain advisory text: this method neither exposes tools nor
        interacts with the SkillLibrary.
        """
        return tuple(
            item.text
            for item in await self.advisory_hint_selections(
                enabled=enabled,
                min_occurrences=min_occurrences,
                max_hints=max_hints,
                max_hint_chars=max_hint_chars,
                available_tool_names=available_tool_names,
            )
        )

    async def advisory_hint_selections(
        self,
        *,
        enabled: bool = False,
        min_occurrences: int = 2,
        max_hints: int = 2,
        max_hint_chars: int = 800,
        available_tool_names: set[str] | None = None,
    ) -> tuple[FailureAdvisoryHint, ...]:
        """Select structured hints without recording an injection."""

        if not self._enabled or not enabled:
            return ()
        candidates = await self._store.hint_candidates(
            min_occurrences=min_occurrences,
            max_hints=max_hints,
            max_hint_chars=max_hint_chars,
            available_tool_names=available_tool_names,
        )
        return tuple(
            FailureAdvisoryHint(
                candidate_id=item.hint_candidate_id,
                text=item.verified_correction_summary,
            )
            for item in candidates
            if item.verified_correction_summary and item.hint_candidate_id
        )

    async def record_hint_selections(
        self,
        *,
        session_id: str,
        turn_id: str,
        selections: tuple[FailureAdvisoryHint, ...],
    ) -> int:
        """Record only hints that the Engine successfully injected."""

        recorded = 0
        for selection in selections:
            try:
                appended = await self._attribution_repository.append(
                    session_id,
                    FailureAttributionRecord.create(
                        candidate_id=selection.candidate_id,
                        turn_id=turn_id,
                        transition="hint_selected",
                    ),
                )
                recorded += int(appended)
            except Exception:
                self._attribution_failure_count += 1
        return recorded

    async def start(self) -> None:
        if self._enabled:
            await self._worker.start()

    async def stop(self) -> None:
        await self._worker.stop()


__all__ = [
    "FailureAdvisoryHint",
    "LearningRuntime",
]
