"""OpenAI Codex native browser OAuth flow tests."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

from mochi.api.server import create_app
from mochi.auth import openai_codex as openai_codex_auth
from mochi.auth.openai_codex import (
    OPENAI_CODEX_DEFAULT_PROFILE_ID,
    OpenAICodexAuthService,
)
from mochi.config.schema import MochiConfig


def _jwt(claims: dict[str, object]) -> str:
    header = {"alg": "none", "typ": "JWT"}
    return ".".join(
        (
            base64.urlsafe_b64encode(json.dumps(header).encode("utf-8")).decode("utf-8").rstrip("="),
            base64.urlsafe_b64encode(json.dumps(claims).encode("utf-8")).decode("utf-8").rstrip("="),
            "signature",
        )
    )


def _build_app(workspace_dir: Path):
    app = create_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "workspace_dir": str(workspace_dir),
        }
    )
    return app


@pytest.fixture
def fake_oauth_tokens(monkeypatch: pytest.MonkeyPatch):
    def _fake_exchange(self, *, code: str, redirect_uri: str, code_verifier: str) -> dict[str, str]:
        assert code
        assert redirect_uri
        assert code_verifier
        return {
            "access_token": _jwt(
                {
                    "email": "codex@example.com",
                    "name": "Codex Browser User",
                    "exp": 4_100_000_000,
                    "https://api.openai.com/auth": {
                        "chatgpt_account_id": "acct-browser-123",
                    },
                }
            ),
            "refresh_token": "refresh-browser-token",
            "id_token": _jwt(
                {
                    "email": "codex@example.com",
                    "name": "Codex Browser User",
                    "https://api.openai.com/auth": {
                        "chatgpt_account_id": "acct-browser-123",
                    },
                }
            ),
        }

    monkeypatch.setattr(
        OpenAICodexAuthService,
        "_request_authorization_code_tokens",
        _fake_exchange,
    )


def test_openai_codex_login_start_persists_pending_flow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openai_codex_auth, "_ensure_local_callback_server", lambda workspace_dir: True)
    app = _build_app(tmp_path / "workspace")

    with TestClient(app) as client:
        response = client.post(
            "/v1/model-auth/openai-codex/login",
            json={"frontend_origin": "http://localhost:3000"},
            headers={"origin": "http://localhost:3000"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "openai_codex_auth_login_start"
    assert payload["callback_url"] == "http://localhost:1455/auth/callback"
    assert payload["flow_id"]
    assert payload["expires_at"] > 0
    assert payload["callback_ready"] is True
    assert len(payload["guidance"]) == 3

    auth_url = urlparse(payload["auth_url"])
    query = parse_qs(auth_url.query)
    assert auth_url.scheme == "https"
    assert auth_url.netloc == "auth.openai.com"
    assert auth_url.path == "/oauth/authorize"
    assert query["redirect_uri"] == [payload["callback_url"]]
    assert query["response_type"] == ["code"]
    assert query["code_challenge_method"] == ["S256"]
    assert query["scope"] == [
        "openid profile email offline_access api.connectors.read api.connectors.invoke"
    ]
    assert query["id_token_add_organizations"] == ["true"]
    assert query["codex_cli_simplified_flow"] == ["true"]
    assert query["state"][0]
    assert query["code_challenge"][0]

    service = OpenAICodexAuthService(str(tmp_path / "workspace"))
    pending_flows = service._store.list_openai_codex_pending_flows()
    assert len(pending_flows) == 1
    assert pending_flows[0].flow_id == payload["flow_id"]
    assert pending_flows[0].redirect_uri == payload["callback_url"]
    assert pending_flows[0].frontend_origin == "http://localhost:3000"


def test_openai_codex_complete_route_saves_browser_profile_and_clears_flow(
    tmp_path: Path,
    fake_oauth_tokens,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openai_codex_auth, "_ensure_local_callback_server", lambda workspace_dir: True)
    workspace_dir = tmp_path / "workspace"
    app = _build_app(workspace_dir)

    with TestClient(app) as client:
        start_response = client.post(
            "/v1/model-auth/openai-codex/login",
            json={"frontend_origin": "http://localhost:3000"},
            headers={"origin": "http://localhost:3000"},
        )
        start_payload = start_response.json()
        state = parse_qs(urlparse(start_payload["auth_url"]).query)["state"][0]

        complete_response = client.post(
            "/v1/model-auth/openai-codex/complete",
            json={
                "callback_url": f"{start_payload['callback_url']}?code=test-code&state={state}",
            },
        )
        status_response = client.get("/v1/model-auth/openai-codex/status")

    assert complete_response.status_code == 200
    complete_payload = complete_response.json()
    assert complete_payload["profile"]["profile_id"] == OPENAI_CODEX_DEFAULT_PROFILE_ID
    assert complete_payload["profile"]["source"] == "browser_oauth"
    assert "access_token" not in complete_response.text
    assert "refresh_token" not in complete_response.text

    assert status_response.status_code == 200
    status_payload = status_response.json()
    assert status_payload["configured"] is True
    assert status_payload["active_profile_id"] == OPENAI_CODEX_DEFAULT_PROFILE_ID
    assert status_payload["profiles"][0]["source"] == "browser_oauth"

    service = OpenAICodexAuthService(str(workspace_dir))
    profile = service.get_profile(OPENAI_CODEX_DEFAULT_PROFILE_ID)
    assert profile is not None
    assert profile.source == "browser_oauth"
    assert profile.account_id == "acct-browser-123"
    assert profile.email == "codex@example.com"
    assert service._store.list_openai_codex_pending_flows() == []


def test_openai_codex_callback_returns_popup_html_and_updates_profile(
    tmp_path: Path,
    fake_oauth_tokens,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openai_codex_auth, "_ensure_local_callback_server", lambda workspace_dir: True)
    workspace_dir = tmp_path / "workspace"
    app = _build_app(workspace_dir)

    with TestClient(app) as client:
        start_response = client.post(
            "/v1/model-auth/openai-codex/login",
            json={"frontend_origin": "http://localhost:3000"},
            headers={"origin": "http://localhost:3000"},
        )
        start_payload = start_response.json()
        state = parse_qs(urlparse(start_payload["auth_url"]).query)["state"][0]

        callback_response = client.get(
            f"/v1/model-auth/openai-codex/callback?code=test-code&state={state}"
        )
        status_response = client.get("/v1/model-auth/openai-codex/status")

    assert callback_response.status_code == 200
    assert "mochi-openai-codex-auth-callback" in callback_response.text
    assert "window.close()" in callback_response.text
    assert "http://localhost:3000/settings?tab=model" in callback_response.text

    assert status_response.status_code == 200
    assert status_response.json()["active_profile_id"] == OPENAI_CODEX_DEFAULT_PROFILE_ID


def test_openai_codex_complete_route_rejects_missing_or_expired_state(
    tmp_path: Path,
    fake_oauth_tokens,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openai_codex_auth, "_ensure_local_callback_server", lambda workspace_dir: True)
    app = _build_app(tmp_path / "workspace")

    with TestClient(app) as client:
        response = client.post(
            "/v1/model-auth/openai-codex/complete",
            json={"code": "test-code", "state": "missing-state"},
        )

    assert response.status_code == 400
    assert "missing or has expired" in response.json()["detail"]
