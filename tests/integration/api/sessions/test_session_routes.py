from __future__ import annotations

import asyncio
import importlib.util
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mochi.config.schema import MochiConfig
from mochi.sessions.store import SessionStore

from ._support import _create_test_app


def test_sessions_create_list_get_round_trip(tmp_path: Path) -> None:
    """`/v1/sessions` should consistently expose protected-workspace rollout status."""
    sessions_dir = tmp_path / "sessions"
    config = MochiConfig.model_validate({"sessions_dir": str(sessions_dir)})
    app = _create_test_app(config=config, session_store=SessionStore(sessions_dir))

    with TestClient(app) as client:
        create_response = client.post("/v1/sessions", json={"session_id": "alpha"})
        assert create_response.status_code == 200
        create_payload = create_response.json()
        assert create_payload["type"] == "session"
        assert create_payload["session_id"] == "alpha"
        alpha_projection = create_payload["protected_workspace"]
        assert alpha_projection["session_id"] == "alpha"
        assert alpha_projection["change_contract"]["enforcement_active"] is False
        assert alpha_projection["sandbox"]["effective_exec_behavior"] == "host_execution_available"

        alpha_path = app.state.session_store._session_path("alpha")  # noqa: SLF001
        old_timestamp = 1_700_000_000
        os.utime(alpha_path, (old_timestamp, old_timestamp))

        create_auto_response = client.post("/v1/sessions")
        assert create_auto_response.status_code == 200
        auto_payload = create_auto_response.json()
        assert auto_payload["type"] == "session"
        assert auto_payload["session_id"]
        assert auto_payload["session_id"] != "alpha"
        assert auto_payload["protected_workspace"]["session_id"] == auto_payload["session_id"]

        list_response = client.get("/v1/sessions")
        assert list_response.status_code == 200
        list_payload = list_response.json()
        assert list_payload["type"] == "sessions"
        assert [item["session_id"] for item in list_payload["items"]] == [
            auto_payload["session_id"],
            "alpha",
        ]
        assert list_payload["items"][0]["event_count"] == 1
        assert list_payload["items"][1]["event_count"] == 1
        assert list_payload["items"][1]["title"] == "alpha"
        assert list_payload["items"][1]["project_id"] is None
        assert list_payload["items"][1]["goal"] is None
        assert list_payload["items"][0]["protected_workspace"] == auto_payload["protected_workspace"]
        assert list_payload["items"][1]["protected_workspace"] == alpha_projection

        get_response = client.get("/v1/sessions/alpha")
        assert get_response.status_code == 200
        assert get_response.json() == {
            "type": "session",
            "session_id": "alpha",
            "title": "alpha",
            "project_id": None,
            "workflow": None,
            "goal": None,
            "security_override": None,
            "protected_workspace": alpha_projection,
            "events": [
                {
                    "type": "session_meta",
                    "event": "created",
                    "session_id": "alpha",
                    "timestamp": get_response.json()["events"][0]["timestamp"],
                }
            ],
        }

        update_response = client.patch("/v1/sessions/alpha", json={"title": "Alpha"})
        assert update_response.status_code == 200
        assert update_response.json()["protected_workspace"] == alpha_projection

        assert getattr(app.state, "runtime_service", None) is None


def test_sessions_list_uses_logical_special_ids_not_encoded_filenames(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    config = MochiConfig.model_validate({"sessions_dir": str(sessions_dir)})
    app = _create_test_app(config=config, session_store=SessionStore(sessions_dir))

    with TestClient(app) as client:
        for session_id in ("a:b", "a?b"):
            response = client.post("/v1/sessions", json={"session_id": session_id})
            assert response.status_code == 200

        listed = client.get("/v1/sessions")
        assert listed.status_code == 200
        assert {item["session_id"] for item in listed.json()["items"]} == {"a:b", "a?b"}


def test_session_create_persists_security_override_before_returning(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    config = MochiConfig.model_validate({"sessions_dir": str(sessions_dir)})
    store = SessionStore(sessions_dir)
    app = _create_test_app(config=config, session_store=store)

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/sessions",
            json={
                "session_id": "auto-review-draft",
                "security_override": {"autonomy_mode": "auto_review"},
            },
        )

    assert create_response.status_code == 200
    assert create_response.json()["security_override"] == {
        "autonomy_mode": "auto_review"
    }

    events = asyncio.run(store.load_session("auto-review-draft"))
    assert [event.get("event") for event in events] == ["created"]
    assert events[0]["security_override"] == {"autonomy_mode": "auto_review"}

    reloaded_app = _create_test_app(
        config=config,
        session_store=SessionStore(sessions_dir),
    )
    with TestClient(reloaded_app) as reloaded_client:
        detail = reloaded_client.get("/v1/sessions/auto-review-draft")
        summaries = reloaded_client.get("/v1/sessions")

    assert detail.status_code == 200
    assert detail.json()["security_override"] == {"autonomy_mode": "auto_review"}
    assert summaries.status_code == 200
    assert summaries.json()["items"][0]["security_override"] == {
        "autonomy_mode": "auto_review"
    }


@pytest.mark.parametrize(
    "security_override",
    [
        {},
        {"autonomy_mode": "unrestricted"},
        {"autonomy_mode": "auto_review", "unexpected": True},
    ],
)
def test_session_create_rejects_invalid_security_override(
    tmp_path: Path,
    security_override: dict[str, object],
) -> None:
    sessions_dir = tmp_path / "sessions"
    config = MochiConfig.model_validate({"sessions_dir": str(sessions_dir)})
    app = _create_test_app(config=config, session_store=SessionStore(sessions_dir))

    with TestClient(app) as client:
        response = client.post(
            "/v1/sessions",
            json={
                "session_id": "invalid-security-override",
                "security_override": security_override,
            },
        )

    assert response.status_code == 422
    assert not (sessions_dir / "invalid-security-override.jsonl").exists()




@pytest.mark.parametrize("sandbox_mode", ["off", "preferred", "required"])
def test_session_and_settings_share_sandbox_rollout_projection(
    tmp_path: Path,
    sandbox_mode: str,
) -> None:
    sessions_dir = tmp_path / "sessions"
    config = MochiConfig.model_validate(
        {
            "sessions_dir": str(sessions_dir),
            "sandbox": {"mode": sandbox_mode},
        }
    )
    app = _create_test_app(config=config, session_store=SessionStore(sessions_dir))

    with TestClient(app) as client:
        session_response = client.post(
            "/v1/sessions",
            json={"session_id": f"sandbox-{sandbox_mode}"},
        )
        settings_response = client.get("/v1/settings")

    assert session_response.status_code == 200
    assert settings_response.status_code == 200
    assert (
        session_response.json()["protected_workspace"]["sandbox"]
        == settings_response.json()["sandbox"]
    )
    assert importlib.util.find_spec("mochi.security.rollout") is not None
    assert getattr(app.state, "runtime_service", None) is None




def test_get_missing_session_returns_empty_events(tmp_path: Path) -> None:
    """不存在 session 時應回傳空 events。"""
    sessions_dir = tmp_path / "sessions"
    config = MochiConfig.model_validate({"sessions_dir": str(sessions_dir)})
    app = _create_test_app(config=config, session_store=SessionStore(sessions_dir))

    with TestClient(app) as client:
        response = client.get("/v1/sessions/missing")

    assert response.status_code == 200
    payload = response.json()
    projection = payload.pop("protected_workspace")
    assert projection["session_id"] == "missing"
    assert projection["change_contract"]["effective_file_behavior"] == "legacy_mutation_allowed"
    assert projection["sandbox"]["enforcement_active"] is False
    assert payload == {
        "type": "session",
        "session_id": "missing",
        "title": "missing",
        "project_id": None,
        "workflow": None,
        "goal": None,
        "security_override": None,
        "events": [],
    }




def test_sessions_can_assign_and_clear_project(tmp_path: Path) -> None:
    """Session summaries and details expose project_id and allow reassignment."""
    from mochi.projects.store import ProjectStore

    sessions_dir = tmp_path / "sessions"
    projects_path = tmp_path / "projects.json"
    config = MochiConfig.model_validate({"sessions_dir": str(sessions_dir)})
    app = _create_test_app(config=config, session_store=SessionStore(sessions_dir))
    app.state.project_store = ProjectStore(projects_path)

    with TestClient(app) as client:
        project = client.post(
            "/v1/projects",
            json={
                "name": "Alpha",
                "workspace_dir": str(tmp_path / "workspace-alpha"),
            },
        ).json()
        create_response = client.post("/v1/sessions", json={"session_id": "alpha"})
        assert create_response.status_code == 200

        assign_response = client.patch(
            "/v1/sessions/alpha/project",
            json={"project_id": project["id"]},
        )
        assert assign_response.status_code == 200
        assert assign_response.json()["project_id"] == project["id"]

        list_response = client.get("/v1/sessions")
        assert list_response.status_code == 200
        assert list_response.json()["items"][0]["project_id"] == project["id"]

        detail_response = client.get("/v1/sessions/alpha")
        assert detail_response.status_code == 200
        assert detail_response.json()["project_id"] == project["id"]

        clear_response = client.patch(
            "/v1/sessions/alpha/project",
            json={"project_id": None},
        )
        assert clear_response.status_code == 200
        assert clear_response.json()["project_id"] is None

        cleared_detail = client.get("/v1/sessions/alpha")
        assert cleared_detail.status_code == 200
        assert cleared_detail.json()["project_id"] is None




def test_sessions_can_fork_from_turn_and_preserve_project(tmp_path: Path) -> None:
    """Forked sessions keep history up to the selected turn and preserve project assignment."""
    from mochi.projects.store import ProjectStore

    sessions_dir = tmp_path / "sessions"
    projects_path = tmp_path / "projects.json"
    config = MochiConfig.model_validate({"sessions_dir": str(sessions_dir)})
    store = SessionStore(sessions_dir)
    app = _create_test_app(config=config, session_store=store)
    app.state.project_store = ProjectStore(projects_path)

    with TestClient(app) as client:
        project = client.post(
            "/v1/projects",
            json={
                "name": "Alpha",
                "workspace_dir": str(tmp_path / "workspace-alpha"),
            },
        ).json()

        assert client.post(
            "/v1/sessions",
            json={"session_id": "alpha", "project_id": project["id"]},
        ).status_code == 200

        asyncio.run(
            store.save_event(
                "alpha",
                {
                    "type": "message",
                    "role": "user",
                    "content": "first question",
                    "turn_id": "turn-1",
                    "timestamp": "2026-05-18T10:00:00+00:00",
                },
            )
        )
        asyncio.run(
            store.save_event(
                "alpha",
                {
                    "type": "message",
                    "role": "assistant",
                    "content": "first answer",
                    "turn_id": "turn-1",
                    "timestamp": "2026-05-18T10:00:01+00:00",
                },
            )
        )
        asyncio.run(
            store.save_event(
                "alpha",
                {
                    "type": "turn_event",
                    "phase": "final_answer",
                    "turn_id": "turn-1",
                    "timestamp": "2026-05-18T10:00:01+00:00",
                    "payload": {"content": "first answer"},
                },
            )
        )
        asyncio.run(
            store.save_event(
                "alpha",
                {
                    "type": "message",
                    "role": "user",
                    "content": "second question",
                    "turn_id": "turn-2",
                    "timestamp": "2026-05-18T10:01:00+00:00",
                },
            )
        )
        asyncio.run(
            store.save_event(
                "alpha",
                {
                    "type": "message",
                    "role": "assistant",
                    "content": "second answer",
                    "turn_id": "turn-2",
                    "timestamp": "2026-05-18T10:01:01+00:00",
                },
            )
        )

        fork_response = client.post(
            "/v1/sessions",
            json={"fork_from_session_id": "alpha", "fork_until_turn_id": "turn-1"},
        )

        assert fork_response.status_code == 200
        forked_session_id = fork_response.json()["session_id"]
        assert forked_session_id != "alpha"

        forked_detail = client.get(f"/v1/sessions/{forked_session_id}")
        assert forked_detail.status_code == 200
        payload = forked_detail.json()
        assert payload["project_id"] == project["id"]
        assert [
            (event.get("type"), event.get("role"), event.get("content"), event.get("turn_id"))
            for event in payload["events"]
            if event.get("type") == "message"
        ] == [
            ("message", "user", "first question", "turn-1"),
            ("message", "assistant", "first answer", "turn-1"),
        ]

        list_payload = client.get("/v1/sessions").json()
        forked_summary = next(
            item for item in list_payload["items"] if item["session_id"] == forked_session_id
        )
        assert forked_summary["project_id"] == project["id"]


def test_session_fork_uses_explicit_create_override_without_inheriting_metadata(
    tmp_path: Path,
) -> None:
    sessions_dir = tmp_path / "sessions"
    config = MochiConfig.model_validate({"sessions_dir": str(sessions_dir)})
    store = SessionStore(sessions_dir)
    app = _create_test_app(config=config, session_store=store)

    with TestClient(app) as client:
        source = client.post(
            "/v1/sessions",
            json={
                "session_id": "source-with-override",
                "security_override": {"autonomy_mode": "auto_review"},
            },
        )
        assert source.status_code == 200
        asyncio.run(
            store.save_event(
                "source-with-override",
                {
                    "type": "message",
                    "role": "assistant",
                    "content": "fork point",
                    "turn_id": "turn-1",
                    "timestamp": "2026-07-23T08:00:00+00:00",
                },
            )
        )

        without_override = client.post(
            "/v1/sessions",
            json={
                "fork_from_session_id": "source-with-override",
                "fork_until_turn_id": "turn-1",
            },
        )
        with_override = client.post(
            "/v1/sessions",
            json={
                "fork_from_session_id": "source-with-override",
                "fork_until_turn_id": "turn-1",
                "security_override": {"autonomy_mode": "strict"},
            },
        )

        assert without_override.status_code == 200
        assert with_override.status_code == 200
        without_detail = client.get(
            f"/v1/sessions/{without_override.json()['session_id']}"
        )
        with_detail = client.get(f"/v1/sessions/{with_override.json()['session_id']}")

    assert without_detail.json()["security_override"] is None
    assert with_detail.json()["security_override"] == {"autonomy_mode": "strict"}




def test_sessions_can_rewrite_from_turn_in_place(tmp_path: Path) -> None:
    """Editing and resending should be able to trim one existing session from a user turn onward."""
    sessions_dir = tmp_path / "sessions"
    config = MochiConfig.model_validate({"sessions_dir": str(sessions_dir)})
    store = SessionStore(sessions_dir)
    app = _create_test_app(config=config, session_store=store)

    with TestClient(app) as client:
        assert client.post("/v1/sessions", json={"session_id": "alpha"}).status_code == 200

        for event in [
            {
                "type": "message",
                "role": "user",
                "content": "first question",
                "turn_id": "turn-1",
                "timestamp": "2026-05-18T10:00:00+00:00",
            },
            {
                "type": "message",
                "role": "assistant",
                "content": "first answer",
                "turn_id": "turn-1",
                "timestamp": "2026-05-18T10:00:01+00:00",
            },
            {
                "type": "message",
                "role": "user",
                "content": "second question",
                "turn_id": "turn-2",
                "timestamp": "2026-05-18T10:01:00+00:00",
            },
            {
                "type": "message",
                "role": "assistant",
                "content": "second answer",
                "turn_id": "turn-2",
                "timestamp": "2026-05-18T10:01:01+00:00",
            },
            {
                "type": "message",
                "role": "user",
                "content": "third question",
                "turn_id": "turn-3",
                "timestamp": "2026-05-18T10:02:00+00:00",
            },
        ]:
            asyncio.run(store.save_event("alpha", event))

        rewrite_response = client.post(
            "/v1/sessions/alpha/rewrite-from-turn",
            json={"from_turn_id": "turn-2"},
        )

        assert rewrite_response.status_code == 200
        payload = rewrite_response.json()
        assert [
            (event.get("type"), event.get("role"), event.get("content"), event.get("turn_id"))
            for event in payload["events"]
            if event.get("type") == "message"
        ] == [
            ("message", "user", "first question", "turn-1"),
            ("message", "assistant", "first answer", "turn-1"),
        ]

        reloaded_detail = client.get("/v1/sessions/alpha")
        assert reloaded_detail.status_code == 200
        assert [
            (event.get("type"), event.get("role"), event.get("content"), event.get("turn_id"))
            for event in reloaded_detail.json()["events"]
            if event.get("type") == "message"
        ] == [
            ("message", "user", "first question", "turn-1"),
            ("message", "assistant", "first answer", "turn-1"),
        ]




def test_sessions_can_update_goal_metadata_separately_from_workflow(tmp_path: Path) -> None:
    """Session PATCH should expose goal metadata separately from workflow state."""
    sessions_dir = tmp_path / "sessions"
    config = MochiConfig.model_validate({"sessions_dir": str(sessions_dir)})
    app = _create_test_app(config=config, session_store=SessionStore(sessions_dir))
    expected_goal = {
        "active_goal_id": "goal-1",
        "active_goal_status": "running",
        "execution_mode": "single_agent",
        "interaction_mode": "goal",
        "execution_topology": "single_agent",
        "bound_run_id": None,
        "protocol_selection": None,
        "selection_rationale": None,
        "default_route": "goal",
        "last_goal_summary": None,
        "pending_proposal": None,
    }

    with TestClient(app) as client:
        assert client.post("/v1/sessions", json={"session_id": "alpha"}).status_code == 200

        update_response = client.patch(
            "/v1/sessions/alpha",
            json={
                "workflow": {"enabled": True, "bound_run_id": "run-1"},
                "goal": {
                    "goal_id": "goal-1",
                    "status": "running",
                    "execution_mode": "single_agent",
                    "default_route": "continue",
                },
            },
        )

        assert update_response.status_code == 200
        updated = update_response.json()
        assert updated["workflow"] == {"enabled": True, "bound_run_id": "run-1"}
        assert updated["goal"] == expected_goal
        assert updated["events"][-2]["event"] == "workflow_state_updated"
        assert updated["events"][-1]["event"] == "goal_state_updated"
        assert updated["events"][-1]["goal"] == expected_goal

        detail_response = client.get("/v1/sessions/alpha")
        assert detail_response.status_code == 200
        assert detail_response.json()["workflow"] == {"enabled": True, "bound_run_id": "run-1"}
        assert detail_response.json()["goal"] == expected_goal

        list_response = client.get("/v1/sessions")
        assert list_response.status_code == 200
        assert list_response.json()["items"][0]["workflow"] == {
            "enabled": True,
            "bound_run_id": "run-1",
        }
        assert list_response.json()["items"][0]["goal"] == expected_goal




def test_sessions_goal_only_patch_preserves_prior_workflow_metadata(tmp_path: Path) -> None:
    """Goal-only PATCH should not clear previously persisted workflow state."""
    sessions_dir = tmp_path / "sessions"
    config = MochiConfig.model_validate({"sessions_dir": str(sessions_dir)})
    app = _create_test_app(config=config, session_store=SessionStore(sessions_dir))

    with TestClient(app) as client:
        assert client.post("/v1/sessions", json={"session_id": "alpha"}).status_code == 200

        workflow = {"enabled": True, "bound_run_id": "run-1"}
        goal = {
            "goal_id": "goal-1",
            "status": "running",
            "execution_mode": "single_agent",
            "default_route": "continue",
        }
        expected_goal = {
            "active_goal_id": "goal-1",
            "active_goal_status": "running",
            "execution_mode": "single_agent",
            "interaction_mode": "goal",
            "execution_topology": "single_agent",
            "bound_run_id": None,
            "protocol_selection": None,
            "selection_rationale": None,
            "default_route": "goal",
            "last_goal_summary": None,
            "pending_proposal": None,
        }

        workflow_response = client.patch("/v1/sessions/alpha", json={"workflow": workflow})
        assert workflow_response.status_code == 200
        assert workflow_response.json()["workflow"] == workflow
        assert workflow_response.json()["goal"] is None

        goal_response = client.patch("/v1/sessions/alpha", json={"goal": goal})
        assert goal_response.status_code == 200
        goal_payload = goal_response.json()
        assert goal_payload["workflow"] == workflow
        assert goal_payload["goal"] == expected_goal
        assert goal_payload["events"][-1]["event"] == "goal_state_updated"
        assert goal_payload["events"][-1]["goal"] == expected_goal

        detail_response = client.get("/v1/sessions/alpha")
        assert detail_response.status_code == 200
        assert detail_response.json()["workflow"] == workflow
        assert detail_response.json()["goal"] == expected_goal

        list_response = client.get("/v1/sessions")
        assert list_response.status_code == 200
        assert list_response.json()["items"][0]["workflow"] == workflow
        assert list_response.json()["items"][0]["goal"] == expected_goal




def test_sessions_workflow_only_patch_preserves_prior_goal_metadata(tmp_path: Path) -> None:
    """Workflow-only PATCH should not clear previously persisted goal state."""
    sessions_dir = tmp_path / "sessions"
    config = MochiConfig.model_validate({"sessions_dir": str(sessions_dir)})
    app = _create_test_app(config=config, session_store=SessionStore(sessions_dir))

    with TestClient(app) as client:
        assert client.post("/v1/sessions", json={"session_id": "alpha"}).status_code == 200

        goal = {
            "goal_id": "goal-1",
            "status": "running",
            "execution_mode": "single_agent",
            "default_route": "continue",
        }
        expected_goal = {
            "active_goal_id": "goal-1",
            "active_goal_status": "running",
            "execution_mode": "single_agent",
            "interaction_mode": "goal",
            "execution_topology": "single_agent",
            "bound_run_id": None,
            "protocol_selection": None,
            "selection_rationale": None,
            "default_route": "goal",
            "last_goal_summary": None,
            "pending_proposal": None,
        }
        workflow = {"enabled": True, "bound_run_id": "run-1"}

        goal_response = client.patch("/v1/sessions/alpha", json={"goal": goal})
        assert goal_response.status_code == 200
        assert goal_response.json()["goal"] == expected_goal
        assert goal_response.json()["workflow"] is None

        workflow_response = client.patch("/v1/sessions/alpha", json={"workflow": workflow})
        assert workflow_response.status_code == 200
        workflow_payload = workflow_response.json()
        assert workflow_payload["workflow"] == workflow
        assert workflow_payload["goal"] == expected_goal
        assert workflow_payload["events"][-1]["event"] == "workflow_state_updated"

        detail_response = client.get("/v1/sessions/alpha")
        assert detail_response.status_code == 200
        assert detail_response.json()["workflow"] == workflow
        assert detail_response.json()["goal"] == expected_goal

        list_response = client.get("/v1/sessions")
        assert list_response.status_code == 200
        assert list_response.json()["items"][0]["workflow"] == workflow
        assert list_response.json()["items"][0]["goal"] == expected_goal




def test_sessions_goal_and_workflow_round_trip_across_fresh_app_instance(
    tmp_path: Path,
) -> None:
    """Goal and workflow metadata should survive reload through a fresh app/store instance."""
    sessions_dir = tmp_path / "sessions"
    config = MochiConfig.model_validate({"sessions_dir": str(sessions_dir)})
    workflow = {"enabled": True, "bound_run_id": "run-1"}
    goal = {
        "goal_id": "goal-1",
        "status": "running",
        "execution_mode": "single_agent",
        "default_route": "continue",
    }
    expected_goal = {
        "active_goal_id": "goal-1",
        "active_goal_status": "running",
        "execution_mode": "single_agent",
        "interaction_mode": "goal",
        "execution_topology": "single_agent",
        "bound_run_id": None,
        "protocol_selection": None,
        "selection_rationale": None,
        "default_route": "goal",
        "last_goal_summary": None,
        "pending_proposal": None,
    }

    first_app = _create_test_app(config=config, session_store=SessionStore(sessions_dir))
    with TestClient(first_app) as client:
        assert client.post("/v1/sessions", json={"session_id": "alpha"}).status_code == 200
        update_response = client.patch(
            "/v1/sessions/alpha",
            json={"workflow": workflow, "goal": goal},
        )
        assert update_response.status_code == 200
        assert update_response.json()["workflow"] == workflow
        assert update_response.json()["goal"] == expected_goal

    reloaded_app = _create_test_app(config=config, session_store=SessionStore(sessions_dir))
    with TestClient(reloaded_app) as reloaded_client:
        detail_response = reloaded_client.get("/v1/sessions/alpha")
        assert detail_response.status_code == 200
        assert detail_response.json()["workflow"] == workflow
        assert detail_response.json()["goal"] == expected_goal

        list_response = reloaded_client.get("/v1/sessions")
        assert list_response.status_code == 200
        assert list_response.json()["items"][0]["workflow"] == workflow
        assert list_response.json()["items"][0]["goal"] == expected_goal




def test_sessions_goal_state_round_trips_completed_summary_and_pending_proposal(
    tmp_path: Path,
) -> None:
    """Completed goal continuity and pending proposal state should survive a fresh app reload."""
    sessions_dir = tmp_path / "sessions"
    config = MochiConfig.model_validate({"sessions_dir": str(sessions_dir)})
    goal = {
        "active_goal_id": None,
        "active_goal_status": None,
        "execution_mode": "workflow",
        "default_route": "goal",
        "last_goal_summary": {
            "goal_id": "goal-completed-1",
            "objective": "Finish the existing migration plan.",
            "execution_mode": "single_agent",
            "protocol_id": None,
            "models": ["gpt-5"],
            "role_summary": "Primary agent continues the task directly with the current chat tools.",
            "runtime_mode": "Single-agent long-running execution",
            "risk_note": None,
            "status": "completed",
        },
        "pending_proposal": {
            "proposal_id": "goal-proposal-2",
            "goal_id": None,
            "objective": "Draft a follow-up validation workflow for the migration.",
            "execution_mode": "workflow",
            "protocol_id": "controlled_subagent_execution",
            "models": ["gpt-5", "claude-sonnet-4-6"],
            "role_summary": "planner, executor, controller, evaluator",
            "runtime_mode": "Workflow run starts immediately",
            "risk_note": "Riskier runtime actions may still require approval.",
            "status": None,
            "revision_index": 1,
            "updated_at": "2026-06-25T12:00:00Z",
        },
    }

    first_app = _create_test_app(config=config, session_store=SessionStore(sessions_dir))
    with TestClient(first_app) as client:
        assert client.post("/v1/sessions", json={"session_id": "alpha"}).status_code == 200
        update_response = client.patch(
            "/v1/sessions/alpha",
            json={"goal": goal},
        )
        assert update_response.status_code == 200
        normalized_goal = update_response.json()["goal"]
        assert normalized_goal["execution_mode"] == "workflow"
        assert normalized_goal["interaction_mode"] == "workflow"
        assert normalized_goal["execution_topology"] == "multi_agent"
        assert normalized_goal["protocol_selection"] == "controlled_subagent_execution"
        assert normalized_goal["selection_rationale"] == (
            "Routed to controlled_subagent_execution for operator-gated execution steps."
        )
        assert normalized_goal["default_route"] == "goal"
        assert normalized_goal["last_goal_summary"]["interaction_mode"] == "goal"
        assert normalized_goal["last_goal_summary"]["execution_topology"] == "single_agent"
        assert normalized_goal["pending_proposal"]["interaction_mode"] == "workflow"
        assert normalized_goal["pending_proposal"]["execution_topology"] == "multi_agent"
        assert normalized_goal["pending_proposal"]["protocol_selection"] == "controlled_subagent_execution"
        assert normalized_goal["pending_proposal"]["selection_rationale"] == (
            "Routed to controlled_subagent_execution for operator-gated execution steps."
        )

    reloaded_app = _create_test_app(config=config, session_store=SessionStore(sessions_dir))
    with TestClient(reloaded_app) as reloaded_client:
        detail_response = reloaded_client.get("/v1/sessions/alpha")
        assert detail_response.status_code == 200
        detail_goal = detail_response.json()["goal"]
        assert detail_goal["execution_mode"] == "workflow"
        assert detail_goal["interaction_mode"] == "workflow"
        assert detail_goal["execution_topology"] == "multi_agent"
        assert detail_goal["protocol_selection"] == "controlled_subagent_execution"
        assert detail_goal["selection_rationale"] == (
            "Routed to controlled_subagent_execution for operator-gated execution steps."
        )
        assert detail_goal["default_route"] == "goal"
        assert detail_goal["last_goal_summary"]["interaction_mode"] == "goal"
        assert detail_goal["last_goal_summary"]["execution_topology"] == "single_agent"
        assert detail_goal["pending_proposal"]["interaction_mode"] == "workflow"
        assert detail_goal["pending_proposal"]["execution_topology"] == "multi_agent"
        assert detail_goal["pending_proposal"]["protocol_selection"] == "controlled_subagent_execution"
        assert detail_goal["pending_proposal"]["selection_rationale"] == (
            "Routed to controlled_subagent_execution for operator-gated execution steps."
        )

        list_response = reloaded_client.get("/v1/sessions")
        assert list_response.status_code == 200
        listed_goal = list_response.json()["items"][0]["goal"]
        assert listed_goal["execution_mode"] == "workflow"
        assert listed_goal["interaction_mode"] == "workflow"
        assert listed_goal["execution_topology"] == "multi_agent"
        assert listed_goal["protocol_selection"] == "controlled_subagent_execution"
        assert listed_goal["selection_rationale"] == (
            "Routed to controlled_subagent_execution for operator-gated execution steps."
        )
        assert listed_goal["default_route"] == "goal"
        assert listed_goal["last_goal_summary"]["interaction_mode"] == "goal"
        assert listed_goal["last_goal_summary"]["execution_topology"] == "single_agent"
        assert listed_goal["pending_proposal"]["interaction_mode"] == "workflow"
        assert listed_goal["pending_proposal"]["execution_topology"] == "multi_agent"
        assert listed_goal["pending_proposal"]["protocol_selection"] == "controlled_subagent_execution"
        assert listed_goal["pending_proposal"]["selection_rationale"] == (
            "Routed to controlled_subagent_execution for operator-gated execution steps."
        )




def test_sessions_goal_patch_normalizes_legacy_protocol_only_state(
    tmp_path: Path,
) -> None:
    sessions_dir = tmp_path / "sessions"
    config = MochiConfig.model_validate({"sessions_dir": str(sessions_dir)})
    app = _create_test_app(config=config, session_store=SessionStore(sessions_dir))

    with TestClient(app) as client:
        assert client.post("/v1/sessions", json={"session_id": "alpha"}).status_code == 200

        response = client.patch(
            "/v1/sessions/alpha",
            json={
                "goal": {
                    "goal_id": "goal-legacy-1",
                    "status": "running",
                    "protocol_id": "autonomous_single_agent",
                    "default_route": "continue",
                }
            },
        )

    assert response.status_code == 200
    goal = response.json()["goal"]
    assert goal["active_goal_id"] == "goal-legacy-1"
    assert goal["active_goal_status"] == "running"
    assert goal["execution_mode"] == "single_agent"
    assert goal["interaction_mode"] == "goal"
    assert goal["execution_topology"] == "single_agent"
    assert goal["protocol_selection"] == "autonomous_single_agent"
    assert goal["selection_rationale"] == (
        "Routed to autonomous_single_agent for direct long-running execution."
    )
    assert goal["default_route"] == "goal"




def test_sessions_can_rename_and_delete(tmp_path: Path) -> None:
    """session 應可更新顯示名稱並刪除。"""
    sessions_dir = tmp_path / "sessions"
    config = MochiConfig.model_validate({"sessions_dir": str(sessions_dir)})
    app = _create_test_app(config=config, session_store=SessionStore(sessions_dir))

    with TestClient(app) as client:
        create_response = client.post("/v1/sessions", json={"session_id": "alpha"})
        assert create_response.status_code == 200

        rename_response = client.patch("/v1/sessions/alpha", json={"title": "研究筆記"})
        assert rename_response.status_code == 200
        rename_payload = rename_response.json()
        assert rename_payload["type"] == "session"
        assert rename_payload["session_id"] == "alpha"
        assert rename_payload["title"] == "研究筆記"
        assert rename_payload["project_id"] is None
        assert rename_payload["workflow"] is None
        assert rename_payload["goal"] is None
        assert isinstance(rename_payload["events"], list)

        get_response = client.get("/v1/sessions/alpha")
        assert get_response.status_code == 200
        payload = get_response.json()
        assert payload["title"] == "研究筆記"
        assert payload["events"][-1]["event"] == "renamed"

        list_response = client.get("/v1/sessions")
        assert list_response.status_code == 200
        assert list_response.json()["items"][0]["title"] == "研究筆記"

        delete_response = client.delete("/v1/sessions/alpha")
        assert delete_response.status_code == 200
        assert delete_response.json() == {
            "type": "session",
            "session_id": "alpha",
            "deleted": True,
        }
        assert client.get("/v1/sessions").json()["items"] == []




def test_sessions_rename_delete_missing_returns_404(tmp_path: Path) -> None:
    """rename/delete missing session 應回 404。"""
    sessions_dir = tmp_path / "sessions"
    config = MochiConfig.model_validate({"sessions_dir": str(sessions_dir)})
    app = _create_test_app(config=config, session_store=SessionStore(sessions_dir))

    with TestClient(app) as client:
        rename_response = client.patch("/v1/sessions/missing", json={"title": "x"})
        delete_response = client.delete("/v1/sessions/missing")

    assert rename_response.status_code == 404
    assert rename_response.json() == {"detail": "Session not found"}
    assert delete_response.status_code == 404
    assert delete_response.json() == {"detail": "Session not found"}




def test_sessions_routes_fall_back_to_config_sessions_dir(tmp_path: Path) -> None:
    """未注入 app.state.session_store 時應使用 config.sessions_dir。"""
    sessions_dir = tmp_path / "sessions"
    config = MochiConfig.model_validate({"sessions_dir": str(sessions_dir)})
    app = _create_test_app(config=config)

    with TestClient(app) as client:
        response = client.post("/v1/sessions", json={"session_id": "from-config"})

    assert response.status_code == 200
    payload = response.json()
    projection = payload.pop("protected_workspace")
    assert projection["session_id"] == "from-config"
    assert payload == {"type": "session", "session_id": "from-config"}
    events = asyncio.run(SessionStore(sessions_dir).load_session("from-config"))
    assert len(events) == 1
    assert events[0]["type"] == "session_meta"


def test_session_event_endpoint_rejects_authoritative_tool_workflow_events(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    config = MochiConfig.model_validate({"sessions_dir": str(sessions_dir)})
    app = _create_test_app(config=config, session_store=SessionStore(sessions_dir))

    with TestClient(app) as client:
        assert client.post("/v1/sessions", json={"session_id": "reserved"}).status_code == 200
        response = client.post(
            "/v1/sessions/reserved/events",
            json={
                "events": [
                    {
                        "type": "session_meta",
                        "event": "tool_workflow_aggregate_outbox",
                        "aggregate": {},
                    }
                ]
            },
        )

    assert response.status_code == 403
    events = asyncio.run(SessionStore(sessions_dir).load_session("reserved"))
    assert [event.get("event") for event in events] == ["created"]
