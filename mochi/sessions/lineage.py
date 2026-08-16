"""Canonical, fail-closed session-lineage records and resolution policy.

Lineage is persisted in ``session_meta.created.lineage`` on the JSONL source.
It deliberately remains outside the rebuildable FTS index.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

LINEAGE_SCHEMA_VERSION = 1

_LINEAGE_REASONS = frozenset(
    {
        "malformed_id",
        "record_not_found",
        "dangling_parent",
        "self_parent",
        "cycle",
        "cross_store_parent",
        "conflicting_record",
        "invalid_resolution",
        "resolver_unavailable",
    }
)


class SessionLineageError(RuntimeError):
    """A canonical lineage chain cannot safely yield a root."""

    code = "session_lineage_error"

    def __init__(self, reason: str) -> None:
        if reason not in _LINEAGE_REASONS:
            raise ValueError("unsupported session lineage error reason")
        self.reason = reason
        super().__init__(f"session lineage resolution failed: {reason}")


@dataclass(frozen=True)
class SessionLineageEnvelope:
    """The immutable v1 parent pointer persisted with created metadata."""

    storage_id: str
    session_id: str
    parent_session_id: str | None
    parent_storage_id: str | None
    fork_until_turn_id: str | None


@dataclass(frozen=True)
class SessionLineageResolution:
    """A stable root reference for one session within one canonical store."""

    storage_id: str
    session_id: str
    root_session_id: str


LineageEnvelopeLoader = Callable[
    [str],
    SessionLineageEnvelope | None | Awaitable[SessionLineageEnvelope | None],
]


def build_session_lineage_envelope(
    *,
    storage_id: str,
    session_id: str,
    parent_session_id: str | None = None,
    parent_storage_id: str | None = None,
    fork_until_turn_id: str | None = None,
) -> dict[str, object]:
    """Build the only v1 JSON-compatible canonical lineage envelope."""

    envelope = SessionLineageEnvelope(
        storage_id=_require_persisted_identifier(storage_id),
        session_id=_require_persisted_identifier(session_id),
        parent_session_id=_require_optional_persisted_identifier(parent_session_id),
        parent_storage_id=_require_optional_persisted_identifier(parent_storage_id),
        fork_until_turn_id=_require_optional_persisted_identifier(fork_until_turn_id),
    )
    _validate_envelope(envelope, expected_storage_id=envelope.storage_id, expected_session_id=envelope.session_id)
    return {
        "schema_version": LINEAGE_SCHEMA_VERSION,
        "storage_id": envelope.storage_id,
        "session_id": envelope.session_id,
        "parent_session_id": envelope.parent_session_id,
        "parent_storage_id": envelope.parent_storage_id,
        "fork_until_turn_id": envelope.fork_until_turn_id,
    }


def lineage_envelope_from_events(
    events: Sequence[Mapping[str, Any]],
    *,
    expected_storage_id: str,
    expected_session_id: str,
) -> SessionLineageEnvelope | None:
    """Parse one immutable created-envelope or identify a legacy history.

    No envelope means a pre-v1 legacy root. A malformed or repeated envelope is
    never downgraded to legacy compatibility.
    """

    candidates: list[Mapping[str, Any]] = []
    for event in events:
        if not isinstance(event, Mapping):
            raise SessionLineageError("invalid_resolution")
        if event.get("type") != "session_meta" or event.get("event") != "created":
            continue
        if "lineage" not in event:
            continue
        candidate = event["lineage"]
        if not isinstance(candidate, Mapping):
            raise SessionLineageError("invalid_resolution")
        candidates.append(candidate)

    if not candidates:
        return None
    if len(candidates) != 1:
        raise SessionLineageError("conflicting_record")

    payload = candidates[0]
    if payload.get("schema_version") != LINEAGE_SCHEMA_VERSION:
        raise SessionLineageError("invalid_resolution")
    envelope = SessionLineageEnvelope(
        storage_id=_require_persisted_identifier(payload.get("storage_id")),
        session_id=_require_persisted_identifier(payload.get("session_id")),
        parent_session_id=_require_optional_persisted_identifier(payload.get("parent_session_id")),
        parent_storage_id=_require_optional_persisted_identifier(payload.get("parent_storage_id")),
        fork_until_turn_id=_require_optional_persisted_identifier(payload.get("fork_until_turn_id")),
    )
    _validate_envelope(
        envelope,
        expected_storage_id=expected_storage_id,
        expected_session_id=expected_session_id,
    )
    return envelope


class SessionLineageResolver:
    """Resolve roots through a supplied same-store canonical-envelope loader."""

    def __init__(self, storage_id: str, envelope_loader: LineageEnvelopeLoader) -> None:
        self._storage_id = _require_persisted_identifier(storage_id)
        if not callable(envelope_loader):
            raise TypeError("envelope_loader must be callable")
        self._envelope_loader = envelope_loader

    async def resolve(self, session_id: str) -> SessionLineageResolution:
        requested_session_id = _require_persisted_identifier(session_id)
        current_session_id = requested_session_id
        visited: set[str] = set()
        initial = True

        while True:
            if current_session_id in visited:
                raise SessionLineageError("cycle")
            visited.add(current_session_id)
            try:
                envelope = self._envelope_loader(current_session_id)
                if inspect.isawaitable(envelope):
                    envelope = await envelope
            except SessionLineageError as exc:
                if exc.reason == "record_not_found" and not initial:
                    raise SessionLineageError("dangling_parent") from exc
                raise
            except Exception as exc:
                raise SessionLineageError("resolver_unavailable") from exc
            initial = False

            if envelope is None:
                return SessionLineageResolution(
                    storage_id=self._storage_id,
                    session_id=requested_session_id,
                    root_session_id=current_session_id,
                )
            if not isinstance(envelope, SessionLineageEnvelope):
                raise SessionLineageError("invalid_resolution")
            _validate_envelope(
                envelope,
                expected_storage_id=self._storage_id,
                expected_session_id=current_session_id,
            )
            if envelope.parent_session_id is None:
                return SessionLineageResolution(
                    storage_id=self._storage_id,
                    session_id=requested_session_id,
                    root_session_id=current_session_id,
                )
            if envelope.parent_storage_id != self._storage_id:
                raise SessionLineageError("cross_store_parent")
            if envelope.parent_session_id == current_session_id:
                raise SessionLineageError("self_parent")
            current_session_id = envelope.parent_session_id

    async def resolve_root(self, session_id: str) -> str:
        """Return only the canonical root for consumer callback injection."""

        return (await self.resolve(session_id)).root_session_id


def _validate_envelope(
    envelope: SessionLineageEnvelope,
    *,
    expected_storage_id: str,
    expected_session_id: str,
) -> None:
    if envelope.storage_id != expected_storage_id or envelope.session_id != expected_session_id:
        raise SessionLineageError("invalid_resolution")
    if envelope.parent_session_id is None:
        if envelope.parent_storage_id is not None or envelope.fork_until_turn_id is not None:
            raise SessionLineageError("invalid_resolution")
        return
    if envelope.parent_storage_id != expected_storage_id:
        raise SessionLineageError("cross_store_parent")
    if envelope.fork_until_turn_id is None:
        raise SessionLineageError("invalid_resolution")


def _require_persisted_identifier(value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise SessionLineageError("malformed_id")
    return value


def _require_optional_persisted_identifier(value: Any) -> str | None:
    if value is None:
        return None
    return _require_persisted_identifier(value)
