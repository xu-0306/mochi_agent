"""Safe, lineage-aware query layer over the rebuildable session FTS index.

The index deliberately knows only about canonical session events.  Lineage is
an application concern, so callers must supply an authoritative root resolver
(or a complete mapping of session ids to their final roots) rather than adding
metadata to JSONL or the derived SQLite schema.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from mochi.sessions.index import (
    MAX_INDEXED_PREVIEW_CHARS,
    MAX_QUERY_RESULTS,
    SessionSearchDocument,
    SessionSearchIndex,
)

MAX_ARTIFACT_REFERENCE_CHARS = 2048
MAX_SOURCE_KIND_CHARS = 128
MAX_TIMESTAMP_CHARS = 256
MAX_TURN_ID_CHARS = 256

_CONTENT_KINDS = frozenset({"bounded_text", "bounded_text_and_artifact", "artifact_only"})


class SessionSearchLineageError(RuntimeError):
    """Lineage could not be resolved safely enough to enforce exclusion."""


class SessionSearchResultSafetyError(RuntimeError):
    """An index response violates the bounded document contract."""


RootSessionResolver = Callable[[str], str | Awaitable[str]]


@dataclass(frozen=True)
class SessionSearchResult:
    """One bounded, stable reference selected from a distinct lineage."""

    session_id: str
    root_session_id: str
    event_index: int
    turn_id: str
    timestamp: str | None
    source_kind: str
    bounded_preview: str
    artifact_ref: str | None
    score: float | None


class SessionSearchService:
    """Query a :class:`SessionSearchIndex` without exposing raw event payloads.

    ``lineage_resolver`` returns the final root session id for a session.  A
    mapping is acceptable only when it is complete for every examined session
    and its roots map to themselves.  Any missing, cyclic, or non-idempotent
    resolution fails closed rather than risking inclusion of an excluded
    ancestor or descendant.
    """

    def __init__(
        self,
        index: SessionSearchIndex,
        *,
        lineage_resolver: RootSessionResolver | Mapping[str, str],
    ) -> None:
        if not isinstance(index, SessionSearchIndex):
            raise TypeError("index must be a SessionSearchIndex")
        if not callable(lineage_resolver) and not isinstance(lineage_resolver, Mapping):
            raise TypeError("lineage_resolver must be a callable or complete mapping")
        self._index = index
        self._lineage_resolver = lineage_resolver

    async def search(
        self,
        query: str,
        *,
        limit: int = 20,
        current_session_id: str | None = None,
        exclude_lineage_ids: Iterable[str] = (),
    ) -> tuple[SessionSearchResult, ...]:
        """Return at most one bounded result per non-excluded lineage.

        The index error contract is intentionally preserved: unavailable,
        corrupt, or incompatible indexes raise their explicit index exception
        instead of being transformed into an empty result set.
        """

        _require_query(query)
        bounded_limit = _require_limit(limit)
        excluded_roots = await self._excluded_roots(
            current_session_id=current_session_id,
            exclude_lineage_ids=exclude_lineage_ids,
        )

        # Index ordering is relevance-first and deterministic.  Scan its
        # bounded maximum so exclusion and lineage de-duplication can still
        # fill the requested number of distinct roots when possible.
        documents = await self._index.search(query, limit=MAX_QUERY_RESULTS)
        selected: list[SessionSearchResult] = []
        seen_roots: set[str] = set()
        for document in documents:
            result = _bounded_result(document, await self._root_for(document.session_id))
            if result.root_session_id in excluded_roots or result.root_session_id in seen_roots:
                continue
            selected.append(result)
            seen_roots.add(result.root_session_id)
            if len(selected) == bounded_limit:
                break
        return tuple(selected)

    async def _excluded_roots(
        self,
        *,
        current_session_id: str | None,
        exclude_lineage_ids: Iterable[str],
    ) -> set[str]:
        roots: set[str] = set()
        if current_session_id is not None:
            roots.add(await self._root_for(_require_identifier(current_session_id, "current_session_id")))
        if isinstance(exclude_lineage_ids, (str, bytes)):
            raise TypeError("exclude_lineage_ids must be an iterable of session ids")
        try:
            requested = tuple(exclude_lineage_ids)
        except TypeError as exc:
            raise TypeError("exclude_lineage_ids must be an iterable of session ids") from exc
        for session_id in requested:
            roots.add(await self._root_for(_require_identifier(session_id, "exclude_lineage_ids item")))
        return roots

    async def _root_for(self, session_id: str) -> str:
        resolver = self._lineage_resolver
        try:
            if isinstance(resolver, Mapping):
                if session_id not in resolver:
                    raise SessionSearchLineageError("session lineage mapping is incomplete")
                resolved: Any = resolver[session_id]
            else:
                resolved = resolver(session_id)
                if inspect.isawaitable(resolved):
                    resolved = await resolved
        except SessionSearchLineageError:
            raise
        except Exception as exc:
            raise SessionSearchLineageError("session lineage cannot be resolved") from exc
        try:
            root_session_id = _require_identifier(resolved, "resolved root_session_id")
        except (TypeError, ValueError) as exc:
            raise SessionSearchLineageError("session lineage resolver returned an invalid root") from exc
        try:
            if isinstance(resolver, Mapping):
                if root_session_id not in resolver:
                    raise SessionSearchLineageError("session lineage mapping is incomplete")
                resolved_root: Any = resolver[root_session_id]
            else:
                resolved_root = resolver(root_session_id)
                if inspect.isawaitable(resolved_root):
                    resolved_root = await resolved_root
            canonical_root = _require_identifier(resolved_root, "canonical root_session_id")
        except SessionSearchLineageError:
            raise
        except (TypeError, ValueError) as exc:
            raise SessionSearchLineageError("session lineage resolver returned an invalid root") from exc
        except Exception as exc:
            raise SessionSearchLineageError("session lineage cannot be resolved") from exc
        if canonical_root != root_session_id:
            raise SessionSearchLineageError("session lineage resolver is not idempotent")
        return root_session_id


def _bounded_result(document: SessionSearchDocument, root_session_id: str) -> SessionSearchResult:
    """Validate the index's public bounded shape before returning it."""

    if not isinstance(document, SessionSearchDocument):
        raise SessionSearchResultSafetyError("index returned an invalid search document")
    session_id = _checked_text(document.session_id, "session_id")
    turn_id = _checked_text(document.turn_id, "turn_id", maximum=MAX_TURN_ID_CHARS)
    timestamp = _checked_optional_text(document.timestamp, "timestamp", maximum=MAX_TIMESTAMP_CHARS)
    source_kind = _checked_text(document.source, "source", maximum=MAX_SOURCE_KIND_CHARS)
    bounded_preview = _checked_text(
        document.preview,
        "preview",
        allow_empty=True,
        maximum=MAX_INDEXED_PREVIEW_CHARS,
    )
    artifact_ref = _checked_optional_text(
        document.artifact_ref,
        "artifact_ref",
        maximum=MAX_ARTIFACT_REFERENCE_CHARS,
    )
    if document.content_kind not in _CONTENT_KINDS:
        raise SessionSearchResultSafetyError("index returned an unknown content kind")
    if document.content_kind == "artifact_only" and bounded_preview:
        raise SessionSearchResultSafetyError("artifact-only index document contains a preview")
    score = _checked_score(document.score)
    return SessionSearchResult(
        session_id=session_id,
        root_session_id=root_session_id,
        event_index=_checked_event_index(document.event_index),
        turn_id=turn_id,
        timestamp=timestamp,
        source_kind=source_kind,
        bounded_preview=bounded_preview,
        artifact_ref=artifact_ref,
        score=score,
    )


def _require_query(query: str) -> None:
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if not query.strip():
        raise ValueError("query must contain searchable text")


def _require_limit(limit: int) -> int:
    if type(limit) is not int or not 1 <= limit <= MAX_QUERY_RESULTS:
        raise ValueError(f"limit must be an integer from 1 through {MAX_QUERY_RESULTS}")
    return limit


def _require_identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _checked_text(value: Any, name: str, *, allow_empty: bool = False, maximum: int | None = None) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise SessionSearchResultSafetyError(f"index returned an invalid {name}")
    if maximum is not None and len(value) > maximum:
        raise SessionSearchResultSafetyError(f"index returned an unbounded {name}")
    return value


def _checked_optional_text(value: Any, name: str, *, maximum: int) -> str | None:
    if value is None:
        return None
    return _checked_text(value, name, maximum=maximum)


def _checked_event_index(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise SessionSearchResultSafetyError("index returned an invalid event_index")
    return value


def _checked_score(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SessionSearchResultSafetyError("index returned an invalid score")
    return float(value)
