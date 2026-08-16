"""Integration coverage for the bounded lineage-aware session search route."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from mochi.api.server import create_app
from mochi.config.schema import MochiConfig
from mochi.sessions.store import SessionStore

from ._support import _create_test_app


def _search_app(tmp_path: Path) -> tuple[object, SessionStore]:
    sessions_dir = tmp_path / "sessions"
    config = MochiConfig.model_validate({"sessions_dir": str(sessions_dir)})
    store = SessionStore(sessions_dir)
    return _create_test_app(config=config, session_store=store), store


def _create_session(client: TestClient, session_id: str) -> None:
    response = client.post("/v1/sessions", json={"session_id": session_id})
    assert response.status_code == 200


def _append_message(
    client: TestClient,
    session_id: str,
    *,
    turn_id: str,
    content: str,
) -> None:
    response = client.post(
        f"/v1/sessions/{session_id}/events",
        json={
            "events": [
                {
                    "type": "message",
                    "role": "assistant",
                    "turn_id": turn_id,
                    "timestamp": "2026-08-08T00:00:00Z",
                    "content": content,
                }
            ]
        },
    )
    assert response.status_code == 200


def test_session_search_returns_stable_bounded_turn_references(tmp_path: Path) -> None:
    app, _store = _search_app(tmp_path)

    with TestClient(app) as client:
        _create_session(client, "reference-session")
        _append_message(
            client,
            "reference-session",
            turn_id="turn-reference",
            content="Find the glacier reference.",
        )

        response = client.get("/v1/sessions/search", params={"query": "glacier"})

    assert response.status_code == 200
    assert response.json() == {
        "type": "session_search",
        "query": "glacier",
        "limit": 20,
        "current_session_id": None,
        "exclude_lineage_ids": [],
        "items": [
            {
                "session_id": "reference-session",
                "root_session_id": "reference-session",
                "event_index": 1,
                "turn_id": "turn-reference",
                "timestamp": "2026-08-08T00:00:00Z",
                "source_kind": "message",
                "bounded_preview": "Find the glacier reference.",
                "artifact_ref": None,
                "score": response.json()["items"][0]["score"],
            }
        ],
    }
    assert isinstance(response.json()["items"][0]["score"], float)


def test_session_search_filters_current_and_explicitly_excluded_lineages(tmp_path: Path) -> None:
    app, _store = _search_app(tmp_path)

    with TestClient(app) as client:
        _create_session(client, "root")
        _append_message(
            client,
            "root",
            turn_id="root-turn",
            content="lineage needle from the root",
        )
        fork = client.post(
            "/v1/sessions",
            json={
                "session_id": "child",
                "fork_from_session_id": "root",
                "fork_until_turn_id": "root-turn",
            },
        )
        assert fork.status_code == 200
        _create_session(client, "other")
        _append_message(
            client,
            "other",
            turn_id="other-turn",
            content="lineage needle from another session",
        )

        current_filtered = client.get(
            "/v1/sessions/search",
            params={"query": "lineage needle", "current_session_id": "child"},
        )
        explicitly_filtered = client.get(
            "/v1/sessions/search",
            params=[("query", "lineage needle"), ("exclude_lineage_ids", "root")],
        )

    assert current_filtered.status_code == 200
    assert explicitly_filtered.status_code == 200
    assert [item["session_id"] for item in current_filtered.json()["items"]] == ["other"]
    assert [item["root_session_id"] for item in current_filtered.json()["items"]] == ["other"]
    assert [item["session_id"] for item in explicitly_filtered.json()["items"]] == ["other"]


def test_session_search_returns_empty_results_and_never_serializes_oversized_content(tmp_path: Path) -> None:
    app, _store = _search_app(tmp_path)
    oversized_content = "artifact needle " + ("safe text " * 100) + "private-tail"

    with TestClient(app) as client:
        _create_session(client, "oversized")
        _append_message(
            client,
            "oversized",
            turn_id="oversized-turn",
            content=oversized_content,
        )
        artifact_response = client.get("/v1/sessions/search", params={"query": "artifact needle"})
        empty_response = client.get("/v1/sessions/search", params={"query": "not-indexed"})

    assert artifact_response.status_code == 200
    artifact = artifact_response.json()["items"][0]
    assert artifact["event_index"] == 1
    assert artifact["turn_id"] == "oversized-turn"
    assert len(artifact["bounded_preview"]) == 512
    assert artifact["artifact_ref"] == "session://oversized/events/1"
    assert "private-tail" not in artifact_response.text
    assert empty_response.status_code == 200
    assert empty_response.json()["items"] == []


def test_session_search_rejects_invalid_filters_before_store_resolution() -> None:
    app = create_app()

    with TestClient(app) as client:
        blank_query = client.get("/v1/sessions/search", params={"query": "   "})
        invalid_limit = client.get("/v1/sessions/search", params={"query": "needle", "limit": 101})
        blank_lineage_id = client.get(
            "/v1/sessions/search",
            params=[("query", "needle"), ("exclude_lineage_ids", " ")],
        )

    assert blank_query.status_code == 422
    assert invalid_limit.status_code == 422
    assert blank_lineage_id.status_code == 422


def test_session_search_reports_an_unavailable_index_explicitly(tmp_path: Path) -> None:
    app, _store = _search_app(tmp_path)

    with TestClient(app) as client:
        response = client.get("/v1/sessions/search", params={"query": "needle"})

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "session_search_index_unavailable"
