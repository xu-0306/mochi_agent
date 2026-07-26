"""Session-scoped permission policy wiring for ordinary chat routes."""

from __future__ import annotations

import pytest

from ._support import MochiConfig, Path, SessionStore, TestClient, _build_app


def _build_policy_app(
    tmp_path: Path,
    *,
    global_mode: str,
):
    sessions_dir = tmp_path / "sessions"
    security: dict[str, object] = {"autonomy_mode": global_mode}
    if global_mode == "high_autonomy":
        security.update({"file_read_scope": "any", "file_write_scope": "any"})
    config = MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(sessions_dir),
            "security": security,
        }
    )
    app, engine = _build_app(workspace_dir=tmp_path)
    app.state.config_factory = lambda: config
    app.state.session_store = SessionStore(sessions_dir)
    return app, engine


@pytest.mark.parametrize("endpoint", ["/v1/chat", "/v1/chat/stream"])
@pytest.mark.parametrize(
    (
        "global_mode",
        "session_mode",
        "expected_write_approval",
        "expected_exec_approval",
    ),
    [
        ("strict", "auto_review", False, False),
        ("high_autonomy", "strict", True, True),
    ],
)
def test_chat_routes_pass_complete_effective_session_policy_to_engine(
    tmp_path: Path,
    endpoint: str,
    global_mode: str,
    session_mode: str,
    expected_write_approval: bool,
    expected_exec_approval: bool,
) -> None:
    app, engine = _build_policy_app(tmp_path, global_mode=global_mode)

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/sessions",
            json={
                "session_id": "policy-session",
                "security_override": {"autonomy_mode": session_mode},
            },
        )
        assert create_response.status_code == 200

        response = client.post(
            endpoint,
            json={"message": "apply the session policy", "session_id": "policy-session"},
        )

    assert response.status_code == 200
    policy = engine.chat_permission_policy_calls[-1]
    assert policy is not None
    assert set(policy) == {
        "policy_snapshot_id",
        "policy_version",
        "source_chain",
        "autonomy_mode",
        "require_approval_for_file_write",
        "require_approval_for_exec",
        "file_read_scope",
        "file_write_scope",
        "hard_denies",
    }
    assert policy["autonomy_mode"] == session_mode
    assert policy["require_approval_for_file_write"] is expected_write_approval
    assert policy["require_approval_for_exec"] is expected_exec_approval
    assert policy["file_read_scope"] == "workspace"
    assert policy["file_write_scope"] == "workspace"
    assert policy["source_chain"] == ["security_config", "session_override"]
    assert str(policy["policy_snapshot_id"]).startswith("policy-")
    assert str(policy["policy_version"]).startswith("effective-policy-v1:")
    assert policy["hard_denies"] == []


def test_chat_route_reloads_session_override_on_the_next_turn(tmp_path: Path) -> None:
    app, engine = _build_policy_app(tmp_path, global_mode="strict")

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/sessions",
            json={
                "session_id": "updated-policy-session",
                "security_override": {"autonomy_mode": "auto_review"},
            },
        )
        assert create_response.status_code == 200

        first_turn = client.post(
            "/v1/chat",
            json={"message": "first turn", "session_id": "updated-policy-session"},
        )
        update_response = client.patch(
            "/v1/sessions/updated-policy-session",
            json={"security_override": {"autonomy_mode": "strict"}},
        )
        second_turn = client.post(
            "/v1/chat",
            json={"message": "second turn", "session_id": "updated-policy-session"},
        )

    assert first_turn.status_code == 200
    assert update_response.status_code == 200
    assert second_turn.status_code == 200
    first_policy, second_policy = engine.chat_permission_policy_calls[-2:]
    assert first_policy is not None
    assert second_policy is not None
    assert first_policy["autonomy_mode"] == "auto_review"
    assert first_policy["require_approval_for_exec"] is False
    assert second_policy["autonomy_mode"] == "strict"
    assert second_policy["require_approval_for_exec"] is True
    assert first_policy["policy_snapshot_id"] != second_policy["policy_snapshot_id"]


def test_chat_route_uses_global_policy_when_session_has_no_override(tmp_path: Path) -> None:
    app, engine = _build_policy_app(tmp_path, global_mode="strict")

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat",
            json={"message": "global fallback", "session_id": "missing-session"},
        )

    assert response.status_code == 200
    policy = engine.chat_permission_policy_calls[-1]
    assert policy is not None
    assert policy["autonomy_mode"] == "strict"
    assert policy["source_chain"] == ["security_config"]
