from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from mochi.agents.adaptive_diagnostics import (
    DIAGNOSTICS_EVENT,
    AdaptiveDiagnosticsAccumulator,
    AdaptiveDiagnosticsRecord,
    AdaptiveDiagnosticsRepository,
)
from mochi.config.schema import MochiConfig
from mochi.learning.failure_attribution import FAILURE_ATTRIBUTION_EVENT
from mochi.learning.failure_episode import FailureEpisode
from mochi.learning.failure_outbox import (
    FAILURE_OUTBOX_SESSION_ID,
    FailureOutboxRepository,
)
from mochi.learning.failure_store import FailureStore
from mochi.learning.runtime import LearningRuntime
from mochi.sessions.store import SessionStore

from ._support import _create_test_app


def test_api_engine_learning_runtime_lifecycle_starts_once_and_stops(
    tmp_path: Path,
) -> None:
    class Runtime:
        def __init__(self) -> None:
            self.starts = 0
            self.stops = 0

        async def start(self) -> None:
            self.starts += 1

        async def stop(self) -> None:
            self.stops += 1

    class Engine:
        def __init__(self) -> None:
            self.learning_runtime = Runtime()

        async def get_voice_runtime_status(self) -> dict[str, object]:
            return {"type": "voice_runtime_status"}

        async def close(self) -> None:
            return None

    config = MochiConfig.model_validate({"sessions_dir": str(tmp_path / "sessions")})
    engine = Engine()
    app = _create_test_app(config=config)
    app.state.engine_factory = lambda: engine
    with TestClient(app) as client:
        assert client.get("/v1/voice/status").status_code == 200
        assert client.get("/v1/voice/status").status_code == 200
        assert engine.learning_runtime.starts == 1
    assert engine.learning_runtime.stops == 1


def test_api_owned_learning_runtime_drains_pending_candidate_and_stops(
    tmp_path: Path,
) -> None:
    sessions_dir = tmp_path / "sessions"
    store = SessionStore(sessions_dir)
    runtime = LearningRuntime(FailureOutboxRepository(store), FailureStore(store))
    episode = FailureEpisode.candidate(
        session_id="session-1",
        turn_id="turn-1",
        capability_tags=("file_mutation",),
        tool_name="file_write",
        failure_signature="verification failed",
        reason_codes=("verification_failed",),
        verifier_feedback=("fixed",),
        correction_attempted=False,
        correction_verified=False,
        episode_id="pending-api",
        idempotency_key="failure:pending-api",
    )
    assert asyncio.run(runtime.submit(episode))

    class Engine:
        learning_runtime = runtime

        async def get_voice_runtime_status(self) -> dict[str, object]:
            return {"type": "voice_runtime_status"}

        async def close(self) -> None:
            return None

    config = MochiConfig.model_validate({"sessions_dir": str(sessions_dir)})
    app = _create_test_app(config=config, session_store=store)
    app.state.engine_factory = Engine
    with TestClient(app) as client:
        assert client.get("/v1/voice/status").status_code == 200
        for _ in range(30):
            records = asyncio.run(FailureOutboxRepository(store).list_records())
            if records and records[0].status == "acked":
                break
            __import__("time").sleep(0.05)
        assert records[0].status == "acked"
        assert runtime.worker.running is True
    assert runtime.worker.running is False


def test_adaptive_runtime_projection_and_replay_stream_are_bounded(
    tmp_path: Path,
) -> None:
    sessions_dir = tmp_path / "sessions"
    config = MochiConfig.model_validate({"sessions_dir": str(sessions_dir)})
    store = SessionStore(sessions_dir)
    app = _create_test_app(config=config, session_store=store)

    with TestClient(app) as client:
        assert (
            client.post("/v1/sessions", json={"session_id": "adaptive"}).status_code
            == 200
        )
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


def test_adaptive_runtime_routes_project_redacted_terminal_status_hint(
    tmp_path: Path,
) -> None:
    sessions_dir = tmp_path / "sessions"
    config = MochiConfig.model_validate({"sessions_dir": str(sessions_dir)})
    store = SessionStore(sessions_dir)
    app = _create_test_app(config=config, session_store=store)

    with TestClient(app) as client:
        assert (
            client.post(
                "/v1/sessions", json={"session_id": "adaptive-terminal"}
            ).status_code
            == 200
        )
        events = [
            {
                "type": "session_meta",
                "event": "ordinary_chat_plan_ledger_updated",
                "session_id": "adaptive-terminal",
                "turn_id": "turn-1",
                "ledger_revision": 1,
                "sequence": 10,
                "plan_ledger": {
                    "ledger_id": "plan:1",
                    "revision": 1,
                    "status": "active",
                    "objective": "bounded objective",
                    "reason_codes": [],
                    "items": [],
                },
            },
            {
                "type": "turn_event",
                "phase": "final_answer",
                "turn_id": "turn-1",
                "seq": 20,
                "payload": {
                    "content": "private final content",
                    "hidden_reasoning": "private chain of thought",
                    "metadata": {
                        "error_type": "plan_finalization_required",
                        "recoverability": "partial",
                        "reason": "plan_incomplete_at_finalization",
                    },
                },
            },
            {
                "type": "session_meta",
                "event": "session_turn_timeline",
                "session_id": "adaptive-terminal",
                "sequence": 30,
                "timeline": {
                    "history_current_revision": 3,
                    "turns": [
                        {
                            "turn_id": "turn-1",
                            "status": "terminal",
                            "terminal_outcome": "completed",
                        }
                    ],
                },
            },
        ]
        for event in events:
            asyncio.run(store.save_event("adaptive-terminal", event))

        snapshot = client.get(
            "/v1/sessions/adaptive-terminal/adaptive-runtime"
        )
        assert snapshot.status_code == 200
        payload = snapshot.json()
        assert payload["turns"][0]["status"] == "partial"
        assert "plan_finalization_required" in payload["turns"][0]["blockers"]
        serialized = json.dumps(payload, ensure_ascii=False)
        assert "private final content" not in serialized
        assert "private chain of thought" not in serialized

        event_range = client.get(
            "/v1/sessions/adaptive-terminal/adaptive-runtime/range?limit=10"
        )
        assert event_range.status_code == 200
        assert any(
            event["event"] == "turn_status_hint"
            for event in event_range.json()["events"]
        )

        stream = client.get(
            "/v1/sessions/adaptive-terminal/adaptive-runtime/stream"
        )
        assert stream.status_code == 200
        assert "turn_status_hint" in stream.text
        assert "private final content" not in stream.text
        assert "private chain of thought" not in stream.text


def test_adaptive_runtime_stream_requires_snapshot_for_unknown_cursor(
    tmp_path: Path,
) -> None:
    sessions_dir = tmp_path / "sessions"
    config = MochiConfig.model_validate({"sessions_dir": str(sessions_dir)})
    app = _create_test_app(config=config, session_store=SessionStore(sessions_dir))

    with TestClient(app) as client:
        assert (
            client.post("/v1/sessions", json={"session_id": "empty"}).status_code == 200
        )
        response = client.get(
            "/v1/sessions/empty/adaptive-runtime/stream",
            headers={"Last-Event-ID": "missing"},
        )
        assert response.status_code == 409
        assert response.json()["detail"]["requires_snapshot"] is True


def test_adaptive_runtime_routes_replay_strict_redacted_diagnostics(
    tmp_path: Path,
) -> None:
    sessions_dir = tmp_path / "sessions"
    config = MochiConfig.model_validate({"sessions_dir": str(sessions_dir)})
    store = SessionStore(sessions_dir)
    app = _create_test_app(config=config, session_store=store)
    accumulator = AdaptiveDiagnosticsAccumulator(
        model_calls=2,
        tool_calls=1,
        input_tokens=21,
        output_tokens=8,
        model_wall_ms=34,
        tool_wall_ms=5,
    )
    record = AdaptiveDiagnosticsRecord.create(
        session_id="diagnostics",
        turn_id="turn-1",
        classification="complex",
        accumulator=accumulator,
        timestamp="2026-07-28T12:34:56+00:00",
    )

    with TestClient(app) as client:
        assert (
            client.post(
                "/v1/sessions",
                json={"session_id": "diagnostics"},
            ).status_code
            == 200
        )
        assert asyncio.run(AdaptiveDiagnosticsRepository(store).append(record))

        snapshot = client.get("/v1/sessions/diagnostics/adaptive-runtime")
        assert snapshot.status_code == 200
        snapshot_payload = snapshot.json()
        diagnostic_event = next(
            event
            for event in snapshot_payload["events"]
            if event["event"] == DIAGNOSTICS_EVENT
        )
        assert diagnostic_event["payload"] == {
            "classification": "complex",
            "counters": accumulator.to_dict(),
        }
        assert snapshot_payload["turns"][0]["diagnostics"] == (
            diagnostic_event["payload"]
        )

        event_range = client.get(
            "/v1/sessions/diagnostics/adaptive-runtime/range?limit=10"
        )
        assert event_range.status_code == 200
        assert diagnostic_event in event_range.json()["events"]

        stream = client.get("/v1/sessions/diagnostics/adaptive-runtime/stream")
        assert stream.status_code == 200
        assert "event: ordinary_chat_adaptive_runtime" in stream.text
        assert diagnostic_event["event_id"] in stream.text
        assert DIAGNOSTICS_EVENT in stream.text

        serialized = json.dumps(
            {
                "snapshot": snapshot_payload,
                "range": event_range.json(),
                "stream": stream.text,
            },
            ensure_ascii=False,
        )
        for forbidden in (
            "raw_prompt",
            "tool_arguments",
            "artifact_path",
            "retrieval_query",
            "hidden_reasoning",
        ):
            assert forbidden not in serialized


def test_failure_learning_transitions_replay_in_original_session_only(
    tmp_path: Path,
) -> None:
    sessions_dir = tmp_path / "sessions"
    config = MochiConfig.model_validate({"sessions_dir": str(sessions_dir)})
    store = SessionStore(sessions_dir)
    app = _create_test_app(config=config, session_store=store)
    episode = FailureEpisode.candidate(
        session_id="learning-a",
        turn_id="turn-1",
        capability_tags=("ordinary_chat", "verification"),
        tool_name="file_write",
        failure_signature="verification failed secret=must-not-leak",
        reason_codes=("verification_failed",),
        verifier_feedback=("raw output and C:/private/path",),
        correction_attempted=False,
        correction_verified=False,
        episode_id="candidate-route",
        idempotency_key="failure:candidate-route",
        created_at="2026-07-28T12:00:00+00:00",
    )
    runtime = LearningRuntime(
        FailureOutboxRepository(store),
        FailureStore(store),
    )

    with TestClient(app) as client:
        for session_id in ("learning-a", "learning-b"):
            assert (
                client.post(
                    "/v1/sessions",
                    json={"session_id": session_id},
                ).status_code
                == 200
            )
        assert asyncio.run(
            runtime.submit(
                episode,
                attribution_session_id="learning-a",
            )
        )
        assert (
            asyncio.run(
                runtime.submit(
                    episode,
                    attribution_session_id="learning-a",
                )
            )
            is False
        )

        first_snapshot = client.get(
            "/v1/sessions/learning-a/adaptive-runtime"
        ).json()
        candidate_event = next(
            event
            for event in first_snapshot["events"]
            if event["event"] == FAILURE_ATTRIBUTION_EVENT
        )
        assert candidate_event["payload"]["transition"] == "candidate"
        assert (
            client.get(
                "/v1/sessions/learning-b/adaptive-runtime"
            ).json()["events"]
            == []
        )
        ranged = client.get(
            "/v1/sessions/learning-a/adaptive-runtime/range?limit=10"
        )
        assert candidate_event in ranged.json()["events"]

        restarted = LearningRuntime(
            FailureOutboxRepository(store),
            FailureStore(store),
        )
        result = asyncio.run(
            restarted.process_once(
                now=datetime(2026, 7, 28, 13, tzinfo=UTC)
            )
        )
        assert result.acked == 1
        replayed = client.get(
            "/v1/sessions/learning-a/adaptive-runtime/stream",
            headers={"Last-Event-ID": candidate_event["event_id"]},
        )
        assert replayed.status_code == 200
        assert FAILURE_ATTRIBUTION_EVENT in replayed.text
        assert '"transition":"processed"' in replayed.text.replace(" ", "")

        snapshot = client.get(
            "/v1/sessions/learning-a/adaptive-runtime"
        ).json()
        assert snapshot["metrics"]["failure_learning"]["candidates"] == 1
        assert snapshot["metrics"]["failure_learning"]["processed"] == 1
        serialized = json.dumps(snapshot, ensure_ascii=False)
        for forbidden in (
            "must-not-leak",
            "raw output",
            "private/path",
            "failure_signature",
            "verifier_feedback",
            "tool_name",
        ):
            assert forbidden not in serialized

        assert (
            client.get(
                f"/v1/sessions/{FAILURE_OUTBOX_SESSION_ID}"
            ).status_code
            == 404
        )
        assert (
            client.get(
                f"/v1/sessions/{FAILURE_OUTBOX_SESSION_ID}/adaptive-runtime"
            ).status_code
            == 404
        )
        listed_ids = {
            item["session_id"]
            for item in client.get("/v1/sessions").json()["items"]
        }
        assert FAILURE_OUTBOX_SESSION_ID not in listed_ids


def test_adaptive_runtime_exact_cursor_keeps_same_sequence_siblings(
    tmp_path: Path,
) -> None:
    sessions_dir = tmp_path / "sessions"
    config = MochiConfig.model_validate({"sessions_dir": str(sessions_dir)})
    store = SessionStore(sessions_dir)
    app = _create_test_app(config=config, session_store=store)
    checkpoint = {
        "turn_id": "turn-1",
        "revision": 1,
        "stage": "running",
        "complexity_decision": {"kind": "plan_required"},
        "inventory_snapshot": {"eligible_tools": ["file_write"]},
    }
    with TestClient(app) as client:
        assert (
            client.post("/v1/sessions", json={"session_id": "siblings"}).status_code
            == 200
        )
        asyncio.run(
            store.save_event(
                "siblings",
                {
                    "type": "session_meta",
                    "event": "turn_execution_checkpoint",
                    "session_id": "siblings",
                    "sequence": 1,
                    "checkpoint": checkpoint,
                },
            )
        )
        first = client.get(
            "/v1/sessions/siblings/adaptive-runtime/range?limit=1"
        ).json()
        assert len(first["events"]) == 1
        second = client.get(
            f"/v1/sessions/siblings/adaptive-runtime/range?limit=1&after_event_id={first['next_event_id']}"
        ).json()
        assert len(second["events"]) == 1
        assert second["events"][0]["event_id"] != first["events"][0]["event_id"]
        stream = client.get(
            "/v1/sessions/siblings/adaptive-runtime/stream",
            headers={"Last-Event-ID": first["next_event_id"]},
        )
        assert second["events"][0]["event_id"] in stream.text
