"""Durable append-only outbox for redacted failure-learning candidates."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, Protocol, cast

from mochi.learning.failure_episode import FailureEpisode

FAILURE_OUTBOX_SESSION_ID = "__mochi_failure_learning_outbox__"
FAILURE_OUTBOX_EVENT = "failure_learning_candidate"
FAILURE_OUTBOX_PROCESSED_EVENT = "failure_learning_processed"
FAILURE_OUTBOX_SCHEMA_VERSION = 1

OutboxStatus = Literal["pending", "claimed", "acked", "rejected"]


class FailureOutboxError(ValueError):
    """Invalid or inconsistent outbox data."""


class FailureOutboxSessionStore(Protocol):
    async def load_session(self, session_id: str) -> list[dict[str, Any]]: ...

    async def append_event_if(
        self,
        session_id: str,
        event: dict[str, Any],
        predicate: Callable[[list[dict[str, Any]]], bool],
    ) -> bool: ...


def _now(value: datetime | None = None) -> datetime:
    current = value or datetime.now(tz=UTC)
    if current.tzinfo is None:
        raise FailureOutboxError("timestamps must be timezone-aware")
    return current.astimezone(UTC)


def _timestamp(value: datetime | None = None) -> str:
    return _now(value).isoformat()


def _parse_timestamp(value: object, *, field_name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise FailureOutboxError(f"{field_name} must be an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise FailureOutboxError(f"{field_name} must be an ISO timestamp") from exc
    if parsed.tzinfo is None:
        raise FailureOutboxError(f"{field_name} must include a timezone")
    return parsed.astimezone(UTC)


def _clean_text(value: object, *, field_name: str, max_chars: int = 128) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FailureOutboxError(f"{field_name} must be a non-empty string")
    value = value.strip()
    if len(value) > max_chars:
        raise FailureOutboxError(f"{field_name} exceeds {max_chars} characters")
    return value


def _clean_attribution_session_id(
    value: object,
    *,
    episode: FailureEpisode,
) -> str:
    session_id = _clean_text(
        value,
        field_name="attribution_session_id",
        max_chars=256,
    )
    if session_id.startswith("__mochi_"):
        raise FailureOutboxError(
            "attribution_session_id must identify a user session"
        )
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    if digest != episode.session_id_hash:
        raise FailureOutboxError(
            "attribution_session_id does not match episode session hash"
        )
    return session_id


def _clean_non_negative_int(value: object, *, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise FailureOutboxError(f"{field_name} must be a non-negative integer")
    return value


def _clean_status(value: object) -> OutboxStatus:
    if value not in {"pending", "claimed", "acked", "rejected"}:
        raise FailureOutboxError(f"unsupported outbox status: {value!r}")
    return value  # type: ignore[return-value]


@dataclass(frozen=True)
class FailureOutboxRecord:
    episode: FailureEpisode
    status: OutboxStatus
    attempts: int
    next_attempt_at: str
    attribution_session_id: str | None = None
    lease_owner: str | None = None
    lease_expires_at: str | None = None
    last_error: str | None = None

    def __post_init__(self) -> None:
        if type(self.episode) is not FailureEpisode:
            raise FailureOutboxError("episode must be a FailureEpisode")
        object.__setattr__(self, "status", _clean_status(self.status))
        object.__setattr__(
            self,
            "attempts",
            _clean_non_negative_int(self.attempts, field_name="attempts"),
        )
        _parse_timestamp(self.next_attempt_at, field_name="next_attempt_at")
        if self.attribution_session_id is not None:
            object.__setattr__(
                self,
                "attribution_session_id",
                _clean_attribution_session_id(
                    self.attribution_session_id,
                    episode=self.episode,
                ),
            )
        if self.lease_owner is not None:
            object.__setattr__(
                self,
                "lease_owner",
                _clean_text(self.lease_owner, field_name="lease_owner"),
            )
        if self.lease_expires_at is not None:
            _parse_timestamp(self.lease_expires_at, field_name="lease_expires_at")
        if self.last_error is not None:
            object.__setattr__(
                self,
                "last_error",
                _clean_text(self.last_error, field_name="last_error", max_chars=400),
            )
        if self.status == "claimed" and (
            self.lease_owner is None or self.lease_expires_at is None
        ):
            raise FailureOutboxError("claimed record requires an active lease")
        if self.status != "claimed" and (
            self.lease_owner is not None or self.lease_expires_at is not None
        ):
            raise FailureOutboxError("only claimed records may carry a lease")


class FailureOutboxRepository:
    """CAS/idempotent candidate and lease operations over SessionStore."""

    def __init__(
        self,
        session_store: FailureOutboxSessionStore,
        *,
        session_id: str = FAILURE_OUTBOX_SESSION_ID,
        retention_days: int | None = None,
    ) -> None:
        self._session_store = session_store
        self._session_id = _clean_text(session_id, field_name="session_id", max_chars=256)
        if retention_days is not None and (type(retention_days) is not int or retention_days < 1):
            raise FailureOutboxError("retention_days must be positive")
        self._retention_days = retention_days

    @property
    def session_store(self) -> FailureOutboxSessionStore:
        """Return the shared store used for user-session attribution writes."""

        return self._session_store

    async def append_candidate(
        self,
        episode: FailureEpisode,
        *,
        attribution_session_id: str | None = None,
    ) -> bool:
        if type(episode) is not FailureEpisode:
            raise FailureOutboxError("episode must be a FailureEpisode")
        attribution_target = (
            _clean_attribution_session_id(
                attribution_session_id,
                episode=episode,
            )
            if attribution_session_id is not None
            else None
        )
        created = episode.created_at
        event = {
            "type": "session_meta",
            "event": FAILURE_OUTBOX_EVENT,
            "schema_version": FAILURE_OUTBOX_SCHEMA_VERSION,
            "idempotency_key": episode.idempotency_key,
            "candidate_id": episode.episode_id,
            "failure_episode": episode.to_dict(),
            # Internal durable routing target only.  Public APIs explicitly
            # deny the global outbox session and project only the redacted
            # attribution event written to this target.
            "attribution_session_id": attribution_target,
            "status": "pending",
            "attempts": 0,
            "next_attempt_at": created,
            "timestamp": created,
        }

        def predicate(events: list[dict[str, Any]]) -> bool:
            return not any(
                event.get("event") == FAILURE_OUTBOX_EVENT
                and event.get("idempotency_key") == episode.idempotency_key
                for event in events
            )

        return await self._session_store.append_event_if(
            self._session_id,
            event,
            predicate,
        )

    async def list_records(self, *, now: datetime | None = None) -> tuple[FailureOutboxRecord, ...]:
        events = await self._session_store.load_session(self._session_id)
        candidates: dict[str, FailureEpisode] = {}
        states: dict[str, FailureOutboxRecord] = {}
        for event in events:
            if event.get("event") == FAILURE_OUTBOX_EVENT:
                if event.get("schema_version") != FAILURE_OUTBOX_SCHEMA_VERSION:
                    raise FailureOutboxError("unsupported failure outbox schema")
                candidate_id = _clean_text(event.get("candidate_id"), field_name="candidate_id")
                episode_payload = event.get("failure_episode")
                if not isinstance(episode_payload, Mapping):
                    raise FailureOutboxError("failure outbox episode must be an object")
                episode = FailureEpisode.from_dict(cast(Mapping[str, Any], episode_payload))
                if episode.episode_id != candidate_id:
                    raise FailureOutboxError("failure outbox candidate identity mismatch")
                candidates[candidate_id] = episode
                states[candidate_id] = FailureOutboxRecord(
                    episode=episode,
                    status="pending",
                    attempts=0,
                    next_attempt_at=_clean_text(
                        event.get("next_attempt_at"),
                        field_name="next_attempt_at",
                    ),
                    attribution_session_id=event.get(
                        "attribution_session_id"
                    ),
                )
            elif event.get("event") == FAILURE_OUTBOX_PROCESSED_EVENT:
                if event.get("schema_version") != FAILURE_OUTBOX_SCHEMA_VERSION:
                    raise FailureOutboxError("unsupported processed outbox schema")
                candidate_id = _clean_text(event.get("candidate_id"), field_name="candidate_id")
                if candidate_id not in candidates:
                    raise FailureOutboxError("processed outbox event has no candidate")
                states[candidate_id] = self._record_from_transition(
                    candidates[candidate_id],
                    event,
                    attribution_session_id=states[
                        candidate_id
                    ].attribution_session_id,
                )
        records = tuple(states[candidate_id] for candidate_id in sorted(states))
        if self._retention_days is None or now is None:
            return records
        cutoff = _now(now) - timedelta(days=self._retention_days)
        # Append-only audit is retained; expiry only bounds worker processing.
        return tuple(record for record in records if _parse_timestamp(record.episode.created_at, field_name="created_at") >= cutoff)

    @staticmethod
    def _record_from_transition(
        episode: FailureEpisode,
        event: Mapping[str, Any],
        *,
        attribution_session_id: str | None,
    ) -> FailureOutboxRecord:
        status = _clean_status(event.get("status"))
        return FailureOutboxRecord(
            episode=episode,
            status=status,
            attempts=_clean_non_negative_int(event.get("attempts"), field_name="attempts"),
            next_attempt_at=_clean_text(
                event.get("next_attempt_at"),
                field_name="next_attempt_at",
            ),
            attribution_session_id=attribution_session_id,
            lease_owner=event.get("lease_owner"),
            lease_expires_at=event.get("lease_expires_at"),
            last_error=event.get("last_error"),
        )

    async def claim(
        self,
        *,
        worker_id: str,
        lease_seconds: float = 30.0,
        now: datetime | None = None,
    ) -> FailureOutboxRecord | None:
        owner = _clean_text(worker_id, field_name="worker_id")
        if isinstance(lease_seconds, bool) or lease_seconds <= 0:
            raise FailureOutboxError("lease_seconds must be positive")
        current_time = _now(now)
        for record in await self.list_records(now=current_time):
            eligible = record.status == "pending" and _parse_timestamp(
                record.next_attempt_at,
                field_name="next_attempt_at",
            ) <= current_time
            if record.status == "claimed" and record.lease_expires_at is not None:
                eligible = _parse_timestamp(
                    record.lease_expires_at,
                    field_name="lease_expires_at",
                ) <= current_time
            if not eligible:
                continue
            attempts = record.attempts + 1
            event = self._transition_event(
                record,
                status="claimed",
                attempts=attempts,
                lease_owner=owner,
                lease_expires_at=_timestamp(current_time + timedelta(seconds=lease_seconds)),
                next_attempt_at=record.next_attempt_at,
                last_error=None,
                timestamp=_timestamp(current_time),
            )
            if await self._append_transition(record, event):
                return await self._get_record(record.episode.episode_id)
        return None

    async def ack(
        self,
        candidate_id: str,
        *,
        worker_id: str,
        now: datetime | None = None,
    ) -> bool:
        return await self._finish_claim(
            candidate_id,
            worker_id=worker_id,
            status="acked",
            last_error=None,
            now=now,
        )

    async def retry(
        self,
        candidate_id: str,
        *,
        worker_id: str,
        error: str,
        backoff_seconds: float,
        now: datetime | None = None,
    ) -> bool:
        if isinstance(backoff_seconds, bool) or backoff_seconds < 0:
            raise FailureOutboxError("backoff_seconds must be non-negative")
        current = _now(now)
        return await self._finish_claim(
            candidate_id,
            worker_id=worker_id,
            status="pending",
            last_error=_clean_text(error, field_name="last_error", max_chars=400),
            next_attempt_at=_timestamp(current + timedelta(seconds=backoff_seconds)),
            now=current,
        )

    async def reject(
        self,
        candidate_id: str,
        *,
        worker_id: str,
        reason: str,
        now: datetime | None = None,
    ) -> bool:
        return await self._finish_claim(
            candidate_id,
            worker_id=worker_id,
            status="rejected",
            last_error=_clean_text(reason, field_name="last_error", max_chars=400),
            now=_now(now),
        )

    async def _get_record(self, candidate_id: str) -> FailureOutboxRecord | None:
        candidate_id = _clean_text(candidate_id, field_name="candidate_id")
        return next(
            (record for record in await self.list_records() if record.episode.episode_id == candidate_id),
            None,
        )

    @staticmethod
    def _transition_event(
        record: FailureOutboxRecord,
        *,
        status: OutboxStatus,
        attempts: int,
        lease_owner: str | None,
        lease_expires_at: str | None,
        next_attempt_at: str,
        last_error: str | None,
        timestamp: str,
    ) -> dict[str, Any]:
        transition_key = (
            f"failure-learning:{record.episode.episode_id}:{status}:{attempts}"
        )
        return {
            "type": "session_meta",
            "event": FAILURE_OUTBOX_PROCESSED_EVENT,
            "schema_version": FAILURE_OUTBOX_SCHEMA_VERSION,
            "idempotency_key": transition_key,
            "candidate_id": record.episode.episode_id,
            "status": status,
            "attempts": attempts,
            "lease_owner": lease_owner,
            "lease_expires_at": lease_expires_at,
            "next_attempt_at": next_attempt_at,
            "last_error": last_error,
            "timestamp": timestamp,
        }

    async def _append_transition(
        self,
        record: FailureOutboxRecord,
        event: dict[str, Any],
    ) -> bool:
        transition_key = event["idempotency_key"]

        def predicate(events: list[dict[str, Any]]) -> bool:
            if any(
                event.get("event") == FAILURE_OUTBOX_PROCESSED_EVENT
                and event.get("idempotency_key") == transition_key
                for event in events
            ):
                # append_event_if() appends when its predicate is true.  A
                # transition key is therefore a compare-and-swap conflict,
                # not an idempotent success: accepting it here would append
                # a second claim and let competing workers process one item.
                return False
            current = self._record_from_events_for_candidate(
                events,
                record.episode.episode_id,
            )
            return current == record

        return await self._session_store.append_event_if(
            self._session_id,
            event,
            predicate,
        )

    @staticmethod
    def _record_from_events_for_candidate(
        events: list[dict[str, Any]], candidate_id: str
    ) -> FailureOutboxRecord:
        candidate: FailureEpisode | None = None
        current: FailureOutboxRecord | None = None
        for event in events:
            if event.get("event") == FAILURE_OUTBOX_EVENT and event.get("candidate_id") == candidate_id:
                payload = event.get("failure_episode")
                if not isinstance(payload, Mapping):
                    raise FailureOutboxError("failure outbox episode must be an object")
                candidate = FailureEpisode.from_dict(cast(Mapping[str, Any], payload))
                current = FailureOutboxRecord(
                    episode=candidate,
                    status="pending",
                    attempts=0,
                    next_attempt_at=_clean_text(
                        event.get("next_attempt_at"),
                        field_name="next_attempt_at",
                    ),
                    attribution_session_id=event.get(
                        "attribution_session_id"
                    ),
                )
            elif (
                event.get("event") == FAILURE_OUTBOX_PROCESSED_EVENT
                and event.get("candidate_id") == candidate_id
            ):
                if candidate is None:
                    raise FailureOutboxError("processed outbox event has no candidate")
                current = FailureOutboxRepository._record_from_transition(
                    candidate,
                    event,
                    attribution_session_id=(
                        current.attribution_session_id
                        if current is not None
                        else None
                    ),
                )
        if current is None:
            raise FailureOutboxError("candidate not found")
        return current

    async def _finish_claim(
        self,
        candidate_id: str,
        *,
        worker_id: str,
        status: Literal["pending", "acked", "rejected"],
        last_error: str | None,
        next_attempt_at: str | None = None,
        now: datetime | None = None,
    ) -> bool:
        candidate_id = _clean_text(candidate_id, field_name="candidate_id")
        owner = _clean_text(worker_id, field_name="worker_id")
        record = await self._get_record(candidate_id)
        if record is None:
            return False
        if record.status != "claimed" or record.lease_owner != owner:
            return False
        current = _now(now)
        event = self._transition_event(
            record,
            status=status,
            attempts=record.attempts,
            lease_owner=None,
            lease_expires_at=None,
            next_attempt_at=next_attempt_at or _timestamp(current),
            last_error=last_error,
            timestamp=_timestamp(current),
        )
        return await self._append_transition(record, event)
