"""Focused tests for the rebuildable JSONL-derived session FTS index."""

from __future__ import annotations

import asyncio
import base64
import sqlite3

import pytest

from mochi.sessions.index import (
    MAX_INDEXED_PREVIEW_CHARS,
    SessionIndexSnapshot,
    SessionSearchIndex,
    SessionSearchIndexUnavailableError,
)


def _snapshot(
    session_id: str,
    events: list[dict],
    *,
    revision: str | None = None,
    modified_ns: int = 1,
) -> SessionIndexSnapshot:
    return SessionIndexSnapshot(
        session_id=session_id,
        events=tuple(events),
        history_revision=revision or f"sha256:{session_id}:{len(events)}",
        source_modified_ns=modified_ns,
    )


def test_rebuild_is_deterministic_and_searches_bounded_source_previews(tmp_path) -> None:
    index = SessionSearchIndex(tmp_path / "sessions")
    alpha = _snapshot(
        "alpha",
        [
            {
                "type": "assistant",
                "turn_id": "turn-alpha",
                "timestamp": "2026-08-05T00:00:00Z",
                "content": "The glacier is blue.",
            }
        ],
        modified_ns=10,
    )
    beta = _snapshot(
        "beta",
        [{"type": "user", "turn_id": "turn-beta", "content": "Look up nebula details."}],
        modified_ns=20,
    )

    first = asyncio.run(index.rebuild_from_supplier(lambda: (beta, alpha)))
    second = asyncio.run(index.rebuild_from_supplier(lambda: (alpha, beta)))

    assert first == second
    assert first.session_count == 2
    assert first.document_count == 2
    hits = asyncio.run(index.search("glacier"))
    assert [(hit.session_id, hit.turn_id, hit.preview) for hit in hits] == [
        ("alpha", "turn-alpha", "The glacier is blue."),
    ]


def test_replacement_and_removal_replace_derived_rows_without_source_aliasing(tmp_path) -> None:
    index = SessionSearchIndex(tmp_path / "sessions")
    asyncio.run(
        index.replace_snapshot(
            _snapshot("session-a", [{"type": "user", "content": "old-needle"}], modified_ns=10)
        )
    )
    assert [hit.preview for hit in asyncio.run(index.search("old needle"))] == ["old-needle"]

    asyncio.run(
        index.replace_snapshot(
            _snapshot("session-a", [{"type": "assistant", "content": "new-needle"}], modified_ns=20)
        )
    )
    assert asyncio.run(index.search("old needle")) == ()
    assert [hit.preview for hit in asyncio.run(index.search("new needle"))] == ["new-needle"]

    asyncio.run(index.remove_session("session-a", source_modified_ns=30))
    assert asyncio.run(index.search("new needle")) == ()
    assert asyncio.run(index.documents_for_session("session-a")) == ()


def test_binary_and_oversized_payloads_keep_only_references_and_bounded_previews(tmp_path) -> None:
    index = SessionSearchIndex(tmp_path / "sessions")
    binary = base64.b64encode(b"\0" * 512).decode("ascii")
    oversized = "ordinary searchable text " * 80
    asyncio.run(
        index.replace_snapshot(
            _snapshot(
                "safe-session",
                [
                    {
                        "type": "tool_result",
                        "turn_id": "tool-turn",
                        "content": binary,
                        "artifact_ref": "artifact://tool-output-1",
                    },
                    {"type": "assistant", "turn_id": "assistant-turn", "content": oversized},
                ],
                modified_ns=10,
            )
        )
    )

    documents = asyncio.run(index.documents_for_session("safe-session"))
    binary_document, oversized_document = documents
    assert binary_document.preview == ""
    assert binary_document.content_kind == "artifact_only"
    assert binary_document.artifact_ref == "artifact://tool-output-1"
    assert binary not in binary_document.preview
    assert len(oversized_document.preview) == MAX_INDEXED_PREVIEW_CHARS
    assert oversized_document.content_kind == "bounded_text_and_artifact"
    assert oversized_document.artifact_ref == "session://safe-session/events/1"
    assert oversized not in oversized_document.preview


def test_unavailable_or_incompatible_index_fails_explicitly_and_rebuild_recovers(tmp_path) -> None:
    index = SessionSearchIndex(tmp_path / "sessions")
    with pytest.raises(SessionSearchIndexUnavailableError):
        asyncio.run(index.search("anything"))

    index.database_path.parent.mkdir(parents=True)
    conn = sqlite3.connect(index.database_path)
    try:
        conn.execute("CREATE TABLE obsolete_schema (value TEXT)")
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(SessionSearchIndexUnavailableError):
        asyncio.run(index.search("anything"))

    report = asyncio.run(
        index.rebuild_from_supplier(
            lambda: (_snapshot("recovered", [{"type": "user", "content": "rebuild works"}]),)
        )
    )
    assert report.session_count == 1
    assert [hit.session_id for hit in asyncio.run(index.search("rebuild"))] == ["recovered"]
