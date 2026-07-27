"""Durable telemetry aggregation for processed failure episodes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, cast

from mochi.learning.failure_episode import FailureEpisode
from mochi.learning.failure_outbox import FAILURE_OUTBOX_SCHEMA_VERSION

FAILURE_STORE_SESSION_ID = "__mochi_failure_learning_store__"
FAILURE_OBSERVATION_EVENT = "failure_learning_observed"


class FailureStoreError(ValueError):
    """Invalid failure-learning telemetry state."""


class FailureStoreSessionStore(Protocol):
    async def load_session(self, session_id: str) -> list[dict[str, Any]]: ...

    async def append_event_if(
        self,
        session_id: str,
        event: dict[str, Any],
        predicate: Callable[[list[dict[str, Any]]], bool],
    ) -> bool: ...


@dataclass(frozen=True)
class FailureAggregate:
    failure_signature: str
    occurrence_count: int
    verified_correction_count: int
    last_occurrence: str
    capability_tags: tuple[str, ...]
    tool_name: str | None
    verified_correction_summary: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_signature": self.failure_signature,
            "occurrence_count": self.occurrence_count,
            "verified_correction_count": self.verified_correction_count,
            "last_occurrence": self.last_occurrence,
            "capability_tags": list(self.capability_tags),
            "tool_name": self.tool_name,
            "verified_correction_summary": self.verified_correction_summary,
        }


class FailureStore:
    """Append observations and derive redacted signature aggregates."""

    def __init__(
        self,
        session_store: FailureStoreSessionStore,
        *,
        session_id: str = FAILURE_STORE_SESSION_ID,
    ) -> None:
        self._session_store = session_store
        self._session_id = session_id

    async def record(self, episode: FailureEpisode) -> bool:
        if type(episode) is not FailureEpisode:
            raise FailureStoreError("episode must be a FailureEpisode")
        event = {
            "type": "session_meta",
            "event": FAILURE_OBSERVATION_EVENT,
            "schema_version": FAILURE_OUTBOX_SCHEMA_VERSION,
            "idempotency_key": episode.idempotency_key,
            "failure_episode": episode.to_dict(),
            "timestamp": episode.created_at,
        }

        def predicate(events: list[dict[str, Any]]) -> bool:
            return not any(
                item.get("event") == FAILURE_OBSERVATION_EVENT
                and item.get("idempotency_key") == episode.idempotency_key
                for item in events
            )

        return await self._session_store.append_event_if(self._session_id, event, predicate)

    async def aggregates(self) -> tuple[FailureAggregate, ...]:
        events = await self._session_store.load_session(self._session_id)
        by_signature: dict[str, list[FailureEpisode]] = {}
        for event in events:
            if event.get("event") != FAILURE_OBSERVATION_EVENT:
                continue
            if event.get("schema_version") != FAILURE_OUTBOX_SCHEMA_VERSION:
                raise FailureStoreError("unsupported failure store schema")
            payload = event.get("failure_episode")
            if not isinstance(payload, Mapping):
                raise FailureStoreError("failure store episode must be an object")
            episode = FailureEpisode.from_dict(cast(Mapping[str, Any], payload))
            by_signature.setdefault(episode.failure_signature, []).append(episode)
        aggregates: list[FailureAggregate] = []
        for signature, episodes in by_signature.items():
            latest = max(episodes, key=lambda episode: episode.created_at)
            verified = [episode for episode in episodes if episode.correction_verified]
            summary = next(
                (
                    feedback
                    for episode in verified
                    for feedback in episode.verifier_feedback
                    if feedback
                ),
                None,
            )
            aggregates.append(
                FailureAggregate(
                    failure_signature=signature,
                    occurrence_count=len(episodes),
                    verified_correction_count=len(verified),
                    last_occurrence=latest.created_at,
                    capability_tags=latest.capability_tags,
                    tool_name=latest.tool_name,
                    verified_correction_summary=summary,
                )
            )
        return tuple(sorted(aggregates, key=lambda item: item.failure_signature))

    async def hint_candidates(
        self,
        *,
        min_occurrences: int = 2,
        max_hints: int = 2,
        max_hint_chars: int = 800,
    ) -> tuple[FailureAggregate, ...]:
        if type(min_occurrences) is not int or min_occurrences < 1:
            raise FailureStoreError("min_occurrences must be positive")
        if type(max_hints) is not int or max_hints < 0:
            raise FailureStoreError("max_hints must be non-negative")
        if type(max_hint_chars) is not int or max_hint_chars < 0:
            raise FailureStoreError("max_hint_chars must be non-negative")
        candidates = [
            aggregate
            for aggregate in await self.aggregates()
            if aggregate.occurrence_count >= min_occurrences
            and aggregate.verified_correction_count > 0
            and aggregate.verified_correction_summary
        ]
        return tuple(
            FailureAggregate(
                failure_signature=aggregate.failure_signature,
                occurrence_count=aggregate.occurrence_count,
                verified_correction_count=aggregate.verified_correction_count,
                last_occurrence=aggregate.last_occurrence,
                capability_tags=aggregate.capability_tags,
                tool_name=aggregate.tool_name,
                verified_correction_summary=(
                    aggregate.verified_correction_summary or ""
                )[:max_hint_chars],
            )
            for aggregate in candidates[:max_hints]
        )
