from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mochi.agents.engine import AgentEngine
from mochi.api.server import create_app
from mochi.config.manager import config_revision, save_config
from mochi.config.schema import MochiConfig
from mochi.runtime.service import RuntimeService
from mochi.runtime.store import RuntimeStore
from mochi.sessions.store import (
    SessionStore,
    SessionsDirectoryRestartRequired,
    StorageIdentityError,
    ToolWorkflowPublicationGate,
    canonical_sessions_dir,
    ensure_sessions_dir_unchanged,
)


def _config(sessions_dir: Path) -> MochiConfig:
    return MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "sessions_dir": str(sessions_dir),
        }
    )


def test_sessions_root_comparison_normalizes_relative_and_case_variants(
    tmp_path: Path,
) -> None:
    base = tmp_path / "Parent"
    root = base / "Sessions"
    equivalent = base / "child" / ".." / "SESSIONS"

    assert canonical_sessions_dir(root) == canonical_sessions_dir(equivalent)
    ensure_sessions_dir_unchanged(root, equivalent)


def test_sessions_root_storage_id_is_durable_and_shared_by_store_instances(
    tmp_path: Path,
) -> None:
    first = SessionStore(tmp_path / "sessions")
    second = SessionStore(tmp_path / "sessions")
    marker = tmp_path / "sessions" / ".mochi-storage.json"

    assert first.storage_id == second.storage_id
    assert json.loads(marker.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "storage_id": first.storage_id,
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"schema_version": 2, "storage_id": "storage:v1:" + "0" * 32},
        {"schema_version": 1, "storage_id": "invalid"},
        {"schema_version": 1, "storage_id": "storage:v1:" + "0" * 32, "extra": True},
    ],
)
def test_sessions_root_storage_marker_fails_closed(
    tmp_path: Path,
    payload: dict[str, object],
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    (root / ".mochi-storage.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(StorageIdentityError):
        SessionStore(root)


@pytest.mark.asyncio
async def test_engine_rejects_sessions_root_before_live_mutation(tmp_path: Path) -> None:
    current = _config(tmp_path / "sessions-old")
    requested = current.model_copy(update={"sessions_dir": str(tmp_path / "sessions-new")})
    engine = AgentEngine.__new__(AgentEngine)
    engine._config = current  # noqa: SLF001

    with pytest.raises(SessionsDirectoryRestartRequired) as raised:
        await engine.apply_config(requested)

    assert raised.value.code == "sessions_dir_restart_required"
    assert engine._config is current  # noqa: SLF001


class _RecordingEngine:
    def __init__(self, session_store: SessionStore) -> None:
        self._session_store = session_store
        self.apply_config_calls = 0

    async def apply_config(self, config: MochiConfig, *, reload_voice: bool = False) -> None:
        del config, reload_voice
        self.apply_config_calls += 1


def test_settings_rejects_sessions_root_before_persist_or_runtime_apply(
    tmp_path: Path,
) -> None:
    old_root = tmp_path / "sessions-old"
    new_root = tmp_path / "sessions-new"
    config = _config(old_root)
    config_path = tmp_path / "config.yaml"
    save_config(config, config_path, expected_revision=config_revision(config_path))
    revision = config_revision(config_path)
    store = SessionStore(old_root)
    engine = _RecordingEngine(store)
    app = create_app()
    app.state.config = config
    app.state.config_path = config_path
    app.state.config_revision = revision
    app.state.session_store = store
    app.state.engine = engine
    before = config_path.read_bytes()

    with TestClient(app) as client:
        response = client.patch(
            "/v1/settings",
            json={"paths": {"sessions_dir": str(new_root)}},
        )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "sessions_dir_restart_required"
    assert config_path.read_bytes() == before
    assert config_revision(config_path) == revision
    assert app.state.config is config
    assert app.state.session_store is store
    assert engine.apply_config_calls == 0
    assert not new_root.exists()


def test_external_sessions_root_reload_is_kept_pending_until_restart(
    tmp_path: Path,
) -> None:
    old_root = tmp_path / "sessions-old"
    new_root = tmp_path / "sessions-new"
    config = _config(old_root)
    config_path = tmp_path / "config.yaml"
    save_config(config, config_path, expected_revision=config_revision(config_path))
    applied_revision = config_revision(config_path)
    store = SessionStore(old_root)
    app = create_app()
    app.state.config = config
    app.state.config_path = config_path
    app.state.config_revision = applied_revision
    app.state.session_store = store
    app.state.engine = _RecordingEngine(store)

    candidate = config.model_copy(update={"sessions_dir": str(new_root)})
    save_config(candidate, config_path, expected_revision=applied_revision)
    pending_revision = config_revision(config_path)

    with TestClient(app) as client:
        response = client.get("/v1/settings")

    assert response.status_code == 200
    payload = response.json()
    assert payload["paths"]["sessions_dir"] == str(old_root)
    assert payload["revision"] == applied_revision
    assert payload["config_reload"] == {
        "status": "pending_restart",
        "code": "sessions_dir_restart_required",
        "pending_revision": pending_revision,
        "pending_sessions_dir": str(new_root),
    }
    assert app.state.config is config
    assert app.state.config_revision == applied_revision
    assert app.state.session_store is store


@pytest.mark.asyncio
async def test_tool_workflow_snapshot_range_and_sse_are_storage_scoped(
    tmp_path: Path,
) -> None:
    fixture_path = Path(__file__).parents[2] / "fixtures" / "tool_workflow_aggregate" / "v1_cases.json"
    case = copy.deepcopy(json.loads(fixture_path.read_text(encoding="utf-8"))["complete_verified"])
    session_id = str(case["session_id"])
    turn_id = str(case["turn_id"])
    timeline = case["timeline"]
    assert isinstance(timeline, dict)
    timeline["history_current_revision"] = 1
    store = SessionStore(tmp_path / "sessions", tool_observability_v1=True)
    await store.save_event(
        session_id,
        {
            "type": "session_meta",
            "event": "session_turn_timeline",
            "schema_version": 1,
            "session_id": session_id,
            "timeline": timeline,
            "timestamp": case["occurred_at"],
        },
    )
    replay_timeline = copy.deepcopy(timeline)
    # The first strict append also writes its aggregate outbox companion at
    # physical position 2, so the next timeline source occupies position 3.
    replay_timeline["history_current_revision"] = 3
    replay_turn = replay_timeline["turns"][0]
    replay_turn["terminal_outcome"] = "cancelled"
    replay_turn["cancellation_outcome"] = "cancelled_running"
    await store.save_event(
        session_id,
        {
            "type": "session_meta",
            "event": "session_turn_timeline",
            "schema_version": 1,
            "session_id": session_id,
            "timeline": replay_timeline,
            "timestamp": case["occurred_at"],
        },
    )
    durable_before_replay = await store.load_strict_snapshot(session_id)
    config = _config(store.sessions_dir)
    config.agent.tool_observability_v1 = True
    app = create_app()
    app.state.config_factory = lambda: config
    app.state.session_store = store

    with TestClient(app) as client:
        snapshot = client.get(f"/v1/sessions/{session_id}/turns/{turn_id}/tool-workflow")
        range_response = client.get(
            f"/v1/sessions/{session_id}/turns/{turn_id}/tool-workflow/range",
            params={"after_seq": 0, "storage_id": store.storage_id},
        )
        stream = client.get(
            f"/v1/sessions/{session_id}/turns/{turn_id}/tool-workflow/stream",
            params={"storage_id": store.storage_id},
        )
        first_event_id = range_response.json()["events"][0]["event_id"]
        replay = client.get(
            f"/v1/sessions/{session_id}/turns/{turn_id}/tool-workflow/stream",
            params={"storage_id": store.storage_id},
            headers={"Last-Event-ID": first_event_id},
        )
        unknown_cursor = client.get(
            f"/v1/sessions/{session_id}/turns/{turn_id}/tool-workflow/stream",
            params={"storage_id": store.storage_id},
            headers={"Last-Event-ID": "twa:v1:missing"},
        )
        mismatch = client.get(
            f"/v1/sessions/{session_id}/turns/{turn_id}/tool-workflow/range",
            params={"storage_id": "storage:v1:" + "f" * 32},
        )

    assert snapshot.status_code == 200
    assert snapshot.json()["storage_id"] == store.storage_id
    assert snapshot.json()["aggregate"]["seq"] == 2
    assert range_response.status_code == 200
    assert range_response.json()["contiguous"] is True
    assert [item["seq"] for item in range_response.json()["events"]] == [1, 2]
    assert stream.status_code == 200
    assert "event: tool_workflow_aggregate" in stream.text
    assert "id: twa:v1:" in stream.text
    assert store.storage_id in stream.text
    data_lines = [line for line in stream.text.splitlines() if line.startswith("data: ")]
    assert len(data_lines) == 2
    stream_payload = json.loads(data_lines[-1].removeprefix("data: "))
    assert stream_payload == {
        "type": "tool_workflow_aggregate",
        "schema_version": 1,
        "storage_id": store.storage_id,
        "session_id": session_id,
        "turn_id": turn_id,
        "aggregate": snapshot.json()["aggregate"],
        "publication_enabled": True,
        "authoritative": True,
    }
    assert replay.status_code == 200
    replay_data_lines = [line for line in replay.text.splitlines() if line.startswith("data: ")]
    assert len(replay_data_lines) == 1
    replay_payload = json.loads(replay_data_lines[0].removeprefix("data: "))
    assert replay_payload["aggregate"]["seq"] == 2
    assert replay_payload["aggregate"]["event_id"] == snapshot.json()["aggregate"]["event_id"]
    assert unknown_cursor.status_code == 409
    assert unknown_cursor.json()["detail"] == {
        "code": "tool_workflow_cursor_not_found",
        "requires_snapshot": True,
        "storage_id": store.storage_id,
    }
    assert mismatch.status_code == 409
    assert mismatch.json()["detail"]["code"] == "tool_workflow_storage_scope_mismatch"
    durable_after_replay = await store.load_strict_snapshot(session_id)
    assert durable_after_replay == durable_before_replay


def test_runtime_service_rejects_rebinding_to_another_sessions_root(tmp_path: Path) -> None:
    old_root = tmp_path / "sessions-old"
    current = _config(old_root)
    requested = _config(tmp_path / "sessions-new")
    service = RuntimeService(
        engine=object(),
        store=RuntimeStore(old_root / "runtime.db"),
    )
    service.bind_app_config(config=current, config_path=None)

    with pytest.raises(SessionsDirectoryRestartRequired):
        service.bind_app_config(config=requested, config_path=None)

    assert service._bound_config is current  # noqa: SLF001

    injected_store = SessionStore(tmp_path / "sessions-injected")
    first_bind_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(old_root / "runtime-injected.db"),
        ordinary_chat_session_store=injected_store,
    )
    with pytest.raises(SessionsDirectoryRestartRequired):
        first_bind_service.bind_app_config(config=current, config_path=None)
    assert first_bind_service._bound_config is None  # noqa: SLF001


def test_runtime_service_preserves_storage_scope_and_publication_gate_for_injected_store(
    tmp_path: Path,
) -> None:
    root = tmp_path / "sessions"
    gate = ToolWorkflowPublicationGate(enabled=True)
    engine_store = SessionStore(
        root,
        tool_observability_v1=True,
        tool_workflow_publication_gate=gate,
    )
    injected_store = SessionStore(root, tool_observability_v1=False)
    engine = SimpleNamespace(
        _session_store=engine_store,
        tool_workflow_publication_gate=gate,
    )
    config = _config(root)
    config.agent.tool_observability_v1 = True
    service = RuntimeService(
        engine=engine,
        store=RuntimeStore(tmp_path / "runtime.db"),
        ordinary_chat_session_store=injected_store,
    )

    service.bind_app_config(config=config, config_path=None)

    assert engine_store.storage_id == injected_store.storage_id
    assert injected_store._tool_workflow_publication_gate is gate  # noqa: SLF001
    assert service._tool_workflow_outbox._session_store is injected_store  # noqa: SLF001
    assert service._tool_workflow_outbox._publication_gate is gate  # noqa: SLF001
