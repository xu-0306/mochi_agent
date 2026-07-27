from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from mochi.config.schema import MochiConfig
from mochi.sessions.store import SessionStore

from ._support import _create_test_app


def test_adaptive_runtime_projection_and_replay_stream_are_bounded(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    config = MochiConfig.model_validate({"sessions_dir": str(sessions_dir)})
    store = SessionStore(sessions_dir)
    app = _create_test_app(config=config, session_store=store)

    with TestClient(app) as client:
        assert client.post("/v1/sessions", json={"session_id": "adaptive"}).status_code == 200
        asyncio.run(
            store.save_event(
                "adaptive",
                {
                    "type": "session_meta",
                    "event": "ordinary_chat_plan_ledger_updated",
                    "schema_version": 1,
                    "session_id": "adaptive",
                    "goal_id": "goal:turn-1",
                    "ledger_id": "plan:1",
                    "ledger_revision": 1,
                    "turn_id": "turn-1",
                    "idempotency_key": "plan-update:1",
                    "plan_ledger": {
                        "ledger_id": "plan:1",
                        "revision": 1,
                        "status": "active",
                        "objective": "bounded objective",
                        "reason_codes": [],
                        "items": [],
                    },
                    "timestamp": "2026-07-27T00:00:00+00:00",
                },
            )
        )

        snapshot = client.get("/v1/sessions/adaptive/adaptive-runtime")
        assert snapshot.status_code == 200
        payload = snapshot.json()
        assert payload["session_id"] == "adaptive"
        assert payload["turns"][0]["plan"]["ledger_id"] == "plan:1"
        assert "objective" in payload["turns"][0]["plan"]

        detail = client.get("/v1/sessions/adaptive?include_adaptive_runtime=true")
        assert detail.status_code == 200
        assert detail.json()["adaptive_runtime"]["session_id"] == "adaptive"

        event_range = client.get("/v1/sessions/adaptive/adaptive-runtime/range?limit=1")
        assert event_range.status_code == 200
        assert len(event_range.json()["events"]) == 1

        stream = client.get("/v1/sessions/adaptive/adaptive-runtime/stream")
        assert stream.status_code == 200
        assert "event: ordinary_chat_adaptive_runtime" in stream.text
        event_id = payload["events"][0]["event_id"]
        replayed = client.get(
            "/v1/sessions/adaptive/adaptive-runtime/stream",
            headers={"Last-Event-ID": event_id},
        )
        assert replayed.status_code == 200
        assert replayed.text == ""


def test_adaptive_runtime_stream_requires_snapshot_for_unknown_cursor(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    config = MochiConfig.model_validate({"sessions_dir": str(sessions_dir)})
    app = _create_test_app(config=config, session_store=SessionStore(sessions_dir))

    with TestClient(app) as client:
        assert client.post("/v1/sessions", json={"session_id": "empty"}).status_code == 200
        response = client.get(
            "/v1/sessions/empty/adaptive-runtime/stream",
            headers={"Last-Event-ID": "missing"},
        )
        assert response.status_code == 409
        assert response.json()["detail"]["requires_snapshot"] is True
