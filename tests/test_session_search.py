"""Focused coverage for the lineage-aware bounded session search service."""

from __future__ import annotations

import asyncio

import pytest

from mochi.sessions.index import (
    MAX_INDEXED_PREVIEW_CHARS,
    SessionIndexSnapshot,
    SessionSearchDocument,
    SessionSearchIndex,
    SessionSearchIndexUnavailableError,
)
from mochi.sessions.search import (
    SessionSearchLineageError,
    SessionSearchResultSafetyError,
    SessionSearchService,
)


def _snapshot(session_id: str, *contents: str) -> SessionIndexSnapshot:
    return SessionIndexSnapshot(
        session_id=session_id,
        events=tuple(
            {
                "type": "assistant",
                "turn_id": f"{session_id}-turn-{event_index}",
                "timestamp": f"2026-08-0{event_index + 1}T00:00:00Z",
                "content": content,
            }
            for event_index, content in enumerate(contents)
        ),
        history_revision=f"sha256:{session_id}",
        source_modified_ns=1,
    )


def _service(tmp_path, snapshots: tuple[SessionIndexSnapshot, ...], lineage: dict[str, str]) -> SessionSearchService:
    index = SessionSearchIndex(tmp_path / "sessions")
    asyncio.run(index.rebuild_from_supplier(lambda: snapshots))
    complete_lineage = {snapshot.session_id: snapshot.session_id for snapshot in snapshots}
    complete_lineage.update(lineage)
    return SessionSearchService(index, lineage_resolver=complete_lineage)


def test_search_groups_hits_by_root_lineage_and_keeps_stable_event_references(tmp_path) -> None:
    service = _service(
        tmp_path,
        (
            _snapshot("alpha-root", "needle root match"),
            _snapshot("alpha-child", "needle child match"),
            _snapshot("beta-root", "needle beta match"),
        ),
        {"alpha-child": "alpha-root"},
    )

    results = asyncio.run(service.search("needle", limit=3))

    assert [(result.session_id, result.root_session_id, result.event_index, result.turn_id) for result in results] == [
        ("alpha-child", "alpha-root", 0, "alpha-child-turn-0"),
        ("beta-root", "beta-root", 0, "beta-root-turn-0"),
    ]
    assert all(not hasattr(result, "content") for result in results)


def test_current_session_excludes_its_entire_lineage_not_only_the_current_id(tmp_path) -> None:
    service = _service(
        tmp_path,
        (
            _snapshot("root", "needle parent match"),
            _snapshot("child", "needle descendant match"),
            _snapshot("other", "needle unrelated match"),
        ),
        {"child": "root"},
    )

    results = asyncio.run(service.search("needle", current_session_id="child"))

    assert [(result.session_id, result.root_session_id) for result in results] == [("other", "other")]


def test_explicit_lineage_exclusions_remove_every_member_of_each_root(tmp_path) -> None:
    service = _service(
        tmp_path,
        (
            _snapshot("alpha-root", "needle alpha root"),
            _snapshot("alpha-child", "needle alpha child"),
            _snapshot("beta-root", "needle beta root"),
            _snapshot("beta-child", "needle beta child"),
            _snapshot("other", "needle unrelated"),
        ),
        {"alpha-child": "alpha-root", "beta-child": "beta-root"},
    )

    results = asyncio.run(
        service.search("needle", exclude_lineage_ids=("alpha-root", "beta-child"))
    )

    assert [(result.session_id, result.root_session_id) for result in results] == [("other", "other")]


def test_service_returns_only_index_bounded_preview_and_artifact_reference(tmp_path) -> None:
    oversized = "needle " + ("private raw payload " * 100)
    service = _service(
        tmp_path,
        (_snapshot("external", oversized),),
        {},
    )

    (result,) = asyncio.run(service.search("needle"))

    assert result.bounded_preview == oversized[:MAX_INDEXED_PREVIEW_CHARS]
    assert len(result.bounded_preview) == MAX_INDEXED_PREVIEW_CHARS
    assert result.artifact_ref == "session://external/events/0"
    assert result.source_kind == "assistant"
    assert result.timestamp == "2026-08-01T00:00:00Z"
    assert result.__dict__.keys() == {
        "session_id",
        "root_session_id",
        "event_index",
        "turn_id",
        "timestamp",
        "source_kind",
        "bounded_preview",
        "artifact_ref",
        "score",
    }


def test_result_limit_is_bounded_after_lineage_grouping(tmp_path) -> None:
    service = _service(
        tmp_path,
        (
            _snapshot("alpha", "needle alpha"),
            _snapshot("beta", "needle beta"),
            _snapshot("gamma", "needle gamma"),
        ),
        {},
    )

    results = asyncio.run(service.search("needle", limit=2))

    assert len(results) == 2
    assert [result.root_session_id for result in results] == ["alpha", "beta"]
    with pytest.raises(ValueError, match="1 through 100"):
        asyncio.run(service.search("needle", limit=101))


def test_unavailable_index_is_explicit_not_an_empty_result(tmp_path) -> None:
    service = SessionSearchService(
        SessionSearchIndex(tmp_path / "sessions"),
        lineage_resolver=lambda session_id: session_id,
    )

    with pytest.raises(SessionSearchIndexUnavailableError):
        asyncio.run(service.search("needle"))


def test_lineage_and_document_contract_fail_closed(tmp_path) -> None:
    index = SessionSearchIndex(tmp_path / "sessions")
    asyncio.run(index.rebuild_from_supplier(lambda: (_snapshot("safe", "needle"),)))

    bad_lineage = SessionSearchService(index, lineage_resolver=lambda _session_id: "")
    with pytest.raises(SessionSearchLineageError):
        asyncio.run(bad_lineage.search("needle"))

    class UnsafeIndex(SessionSearchIndex):
        async def search(self, query: str, *, limit: int = 20):
            return (
                SessionSearchDocument(
                    session_id="unsafe",
                    event_index=0,
                    turn_id="unsafe-turn",
                    timestamp=None,
                    source="assistant",
                    preview="x" * (MAX_INDEXED_PREVIEW_CHARS + 1),
                    artifact_ref=None,
                    content_kind="bounded_text",
                ),
            )

    unsafe = SessionSearchService(UnsafeIndex(tmp_path / "unsafe"), lineage_resolver=lambda session_id: session_id)
    with pytest.raises(SessionSearchResultSafetyError):
        asyncio.run(unsafe.search("needle"))


def test_result_retains_original_nonzero_event_index_and_rejects_invalid_index_reference(tmp_path) -> None:
    snapshot = SessionIndexSnapshot(
        session_id="safe",
        events=(
            {"type": "session_meta", "content": {"ignored": "metadata"}},
            {
                "type": "assistant",
                "turn_id": "safe-turn-1",
                "timestamp": "2026-08-01T00:00:00Z",
                "content": "needle",
            },
        ),
        history_revision="sha256:safe",
        source_modified_ns=1,
    )
    index = SessionSearchIndex(tmp_path / "sessions")
    asyncio.run(index.rebuild_from_supplier(lambda: (snapshot,)))

    (result,) = asyncio.run(SessionSearchService(index, lineage_resolver=lambda session_id: session_id).search("needle"))

    assert (result.session_id, result.event_index, result.turn_id) == ("safe", 1, "safe-turn-1")

    class InvalidReferenceIndex(SessionSearchIndex):
        async def search(self, query: str, *, limit: int = 20):
            return (
                SessionSearchDocument(
                    session_id="unsafe",
                    event_index=-1,
                    turn_id="unsafe-turn",
                    timestamp=None,
                    source="assistant",
                    preview="needle",
                    artifact_ref=None,
                    content_kind="bounded_text",
                ),
            )

    unsafe = SessionSearchService(
        InvalidReferenceIndex(tmp_path / "unsafe"),
        lineage_resolver=lambda session_id: session_id,
    )
    with pytest.raises(SessionSearchResultSafetyError, match="event_index"):
        asyncio.run(unsafe.search("needle"))


def test_lineage_resolution_requires_complete_idempotent_authority(tmp_path) -> None:
    index = SessionSearchIndex(tmp_path / "sessions")
    asyncio.run(index.rebuild_from_supplier(lambda: (_snapshot("child", "needle"),)))

    with pytest.raises(TypeError, match="callable or complete mapping"):
        SessionSearchService(index, lineage_resolver=None)  # type: ignore[arg-type]

    incomplete = SessionSearchService(index, lineage_resolver={"child": "root"})
    with pytest.raises(SessionSearchLineageError, match="incomplete"):
        asyncio.run(incomplete.search("needle", current_session_id="child"))

    cyclic = SessionSearchService(index, lineage_resolver={"child": "root", "root": "child"})
    with pytest.raises(SessionSearchLineageError, match="not idempotent"):
        asyncio.run(cyclic.search("needle", current_session_id="child"))


def test_async_authoritative_resolver_can_exclude_a_full_lineage(tmp_path) -> None:
    snapshots = (
        _snapshot("root", "needle root"),
        _snapshot("child", "needle child"),
        _snapshot("other", "needle other"),
    )
    index = SessionSearchIndex(tmp_path / "sessions")
    asyncio.run(index.rebuild_from_supplier(lambda: snapshots))

    async def resolve(session_id: str) -> str:
        return {"root": "root", "child": "root", "other": "other"}[session_id]

    results = asyncio.run(SessionSearchService(index, lineage_resolver=resolve).search("needle", current_session_id="child"))

    assert [(result.session_id, result.root_session_id) for result in results] == [("other", "other")]
