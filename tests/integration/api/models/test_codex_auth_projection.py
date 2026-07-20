"""Codex authentication projection integration tests."""

from __future__ import annotations

from ._support import *  # noqa: F401,F403


def test_openai_codex_import_route_stores_cli_login_under_mochi_state_root(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """OpenAI Codex CLI import should populate the separate auth store."""
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "auth.json").write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": "header.eyJleHAiOjE5MDAwMDAwMDAsImVtYWlsIjoiY29kZXhAZXhhbXBsZS5jb20ifQ.sig",
                    "refresh_token": "refresh-token",
                    "account_id": "acct_123",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    workspace_dir = tmp_path / "workspace"
    app, _engine = _build_app(workspace_dir=workspace_dir)

    with TestClient(app) as client:
        response = client.post("/v1/model-auth/openai-codex/import-codex-cli")

    assert response.status_code == 200
    payload = response.json()
    assert payload["profile"]["profile_id"] == "openai_codex:default"
    assert "store_path" not in payload
    assert "source_path" not in payload["profile"]
    store_path = workspace_dir / ".mochi" / "auth.json"
    assert store_path.is_file()
    raw = store_path.read_text(encoding="utf-8")
    assert "refresh-token" in raw
    assert "codex@example.com" in raw

def test_openai_codex_status_route_redacts_tokens_and_paths(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Auth status should expose safe profile metadata only."""
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "auth.json").write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": "header.eyJleHAiOjE5MDAwMDAwMDAsImVtYWlsIjoiY29kZXhAZXhhbXBsZS5jb20ifQ.sig",
                    "refresh_token": "refresh-token",
                    "account_id": "acct_123",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    workspace_dir = tmp_path / "workspace"
    app, _engine = _build_app(workspace_dir=workspace_dir)

    with TestClient(app) as client:
        import_response = client.post("/v1/model-auth/openai-codex/import-codex-cli")
        status_response = client.get("/v1/model-auth/openai-codex/status")

    assert import_response.status_code == 200
    assert status_response.status_code == 200
    payload = status_response.json()
    assert payload["configured"] is True
    assert payload["active_profile_id"] == "openai_codex:default"
    assert payload["profiles"][0]["email"] == "codex@example.com"
    assert "access_token" not in status_response.text
    assert "refresh-token" not in status_response.text
    assert "store_path" not in status_response.text
    assert "source_path" not in status_response.text

def test_openai_codex_service_refreshes_expired_access_token_on_resolve(tmp_path: Path, monkeypatch: Any) -> None:
    """Expired access tokens should refresh automatically on access-token resolution."""
    workspace_dir = tmp_path / "workspace"
    service = OpenAICodexAuthService(str(workspace_dir))
    expired_token = _fake_jwt(1_700_000_000)
    refreshed_token = _fake_jwt(4_100_000_000, name="Codex User")
    service._store.upsert_openai_codex_profile(  # noqa: SLF001
        OpenAICodexAuthProfile(
            profile_id="openai_codex:default",
            access_token=expired_token,
            refresh_token="refresh-token",
            email="codex@example.com",
            display_name="Codex User",
            expires_at=1_700_000_000,
            source_path=None,
        )
    )

    monkeypatch.setattr(
        service,
        "_request_token_refresh",
        lambda refresh_token: {
            "access_token": refreshed_token,
            "refresh_token": f"{refresh_token}-next",
        },
    )

    resolved = service.resolve_access_token("openai_codex:default")
    saved = service.get_profile("openai_codex:default")

    assert resolved == refreshed_token
    assert saved is not None
    assert saved.refresh_token.get_secret_value() == "refresh-token-next"
    assert saved.last_refresh_error is None
    assert service.get_profile_summary("openai_codex:default").status == "ready"  # type: ignore[union-attr]

def test_openai_codex_service_records_refresh_failure_status(tmp_path: Path, monkeypatch: Any) -> None:
    """Refresh failures should be persisted as auth diagnostics instead of surfacing only as backend 401s."""
    workspace_dir = tmp_path / "workspace"
    service = OpenAICodexAuthService(str(workspace_dir))
    expired_token = _fake_jwt(1_700_000_000)
    service._store.upsert_openai_codex_profile(  # noqa: SLF001
        OpenAICodexAuthProfile(
            profile_id="openai_codex:default",
            access_token=expired_token,
            refresh_token="refresh-token",
            email="codex@example.com",
            display_name="Codex User",
            expires_at=1_700_000_000,
            source_path=None,
        )
    )

    def _raise_refresh_error(_refresh_token: str) -> dict[str, Any]:
        raise RuntimeError("invalid_grant")

    monkeypatch.setattr(service, "_request_token_refresh", _raise_refresh_error)

    with pytest.raises(RuntimeError, match="OpenAI Codex auth refresh failed"):
        service.resolve_access_token("openai_codex:default")

    profile = service.get_profile("openai_codex:default")
    summary = service.get_profile_summary("openai_codex:default")
    assert profile is not None
    assert profile.last_refresh_error == "invalid_grant"
    assert summary is not None
    assert summary.status == "refresh_failed"
    assert summary.last_refresh_error == "invalid_grant"

def test_openai_codex_refresh_access_token_recovers_stale_file_lock(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A stale cross-process refresh lock should not block token refresh forever."""
    workspace_dir = tmp_path / "workspace"
    service = OpenAICodexAuthService(str(workspace_dir))
    expired_token = _fake_jwt(1_700_000_000)
    refreshed_token = _fake_jwt(4_100_000_000)
    service._store.upsert_openai_codex_profile(  # noqa: SLF001
        OpenAICodexAuthProfile(
            profile_id="openai_codex:default",
            access_token=expired_token,
            refresh_token="refresh-token",
            email="codex@example.com",
            display_name="Codex User",
            expires_at=1_700_000_000,
            source_path=None,
        )
    )
    lock_path = _profile_refresh_lock_path(service.store_path, "openai_codex:default")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("stale", encoding="utf-8")
    stale_age = time.time() - (OPENAI_CODEX_REFRESH_LOCK_STALE_SECONDS + 5.0)
    os.utime(lock_path, (stale_age, stale_age))

    monkeypatch.setattr(
        service,
        "_request_token_refresh",
        lambda refresh_token: {
            "access_token": refreshed_token,
            "refresh_token": refresh_token,
        },
    )

    resolved = service.resolve_access_token("openai_codex:default")

    assert resolved == refreshed_token
    assert not lock_path.exists()

def test_openai_codex_refresh_access_token_times_out_on_live_file_lock(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A live cross-process refresh lock should fail fast with a bounded timeout."""
    workspace_dir = tmp_path / "workspace"
    service = OpenAICodexAuthService(str(workspace_dir))
    expired_token = _fake_jwt(1_700_000_000)
    service._store.upsert_openai_codex_profile(  # noqa: SLF001
        OpenAICodexAuthProfile(
            profile_id="openai_codex:default",
            access_token=expired_token,
            refresh_token="refresh-token",
            email="codex@example.com",
            display_name="Codex User",
            expires_at=1_700_000_000,
            source_path=None,
        )
    )
    lock_path = _profile_refresh_lock_path(service.store_path, "openai_codex:default")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("live", encoding="utf-8")
    monkeypatch.setattr(
        "mochi.auth.openai_codex.OPENAI_CODEX_REFRESH_LOCK_TIMEOUT_SECONDS",
        0.05,
    )
    monkeypatch.setattr(
        "mochi.auth.openai_codex.OPENAI_CODEX_REFRESH_LOCK_POLL_SECONDS",
        0.01,
    )

    with pytest.raises(RuntimeError, match="Timed out waiting for OpenAI Codex refresh lock"):
        service.refresh_access_token("openai_codex:default", force=True)

def test_openai_codex_import_route_returns_400_when_cli_login_missing(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Import should fail clearly when the local Codex CLI auth file does not exist."""
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    workspace_dir = tmp_path / "workspace"
    app, _engine = _build_app(workspace_dir=workspace_dir)

    with TestClient(app) as client:
        response = client.post("/v1/model-auth/openai-codex/import-codex-cli")

    assert response.status_code == 400
    assert "was not found" in response.json()["detail"]

def test_openai_codex_import_route_rejects_apikey_cli_state_with_actionable_message(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Import should explain that API-key Codex CLI state is not importable as ChatGPT OAuth."""
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "auth.json").write_text(
        json.dumps(
            {
                "auth_mode": "apikey",
                "OPENAI_API_KEY": "sk-test",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    workspace_dir = tmp_path / "workspace"
    app, _engine = _build_app(workspace_dir=workspace_dir)

    with TestClient(app) as client:
        response = client.post("/v1/model-auth/openai-codex/import-codex-cli")

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "API key mode" in detail
    assert "Connect ChatGPT" in detail

def test_openai_codex_status_route_reports_cli_auth_diagnostics_without_leaking_paths(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Status should expose safe CLI diagnostics even when no Mochi auth profile is saved yet."""
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "auth.json").write_text(
        json.dumps(
            {
                "auth_mode": "apikey",
                "OPENAI_API_KEY": "sk-test",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    workspace_dir = tmp_path / "workspace"
    app, _engine = _build_app(workspace_dir=workspace_dir)

    with TestClient(app) as client:
        response = client.get("/v1/model-auth/openai-codex/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["configured"] is False
    assert payload["cli_auth_state"] == "apikey"
    assert payload["cli_auth_mode"] == "apikey"
    assert payload["cli_auth_can_import"] is False
    assert "API key mode" in payload["cli_auth_message"]
    assert str(codex_home) not in response.text
    assert "sk-test" not in response.text

def test_openai_codex_refresh_route_updates_status_without_reimport(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Refresh route should renew an expired imported profile instead of requiring a new CLI import."""
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "auth.json").write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": _fake_jwt(1_700_000_000),
                    "refresh_token": "refresh-token",
                    "account_id": "acct_123",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setattr(
        OpenAICodexAuthService,
        "_request_token_refresh",
        lambda self, refresh_token: {
            "access_token": _fake_jwt(4_100_000_000, name="Codex User"),
            "refresh_token": f"{refresh_token}-next",
        },
    )
    workspace_dir = tmp_path / "workspace"
    app, _engine = _build_app(workspace_dir=workspace_dir)

    with TestClient(app) as client:
        import_response = client.post("/v1/model-auth/openai-codex/import-codex-cli")
        refresh_response = client.post("/v1/model-auth/openai-codex/refresh")
        status_response = client.get("/v1/model-auth/openai-codex/status")

    assert import_response.status_code == 200
    assert refresh_response.status_code == 200
    assert status_response.status_code == 200
    refresh_payload = refresh_response.json()
    status_payload = status_response.json()
    assert refresh_payload["profile"]["status"] == "ready"
    assert status_payload["status"] == "ready"
    assert status_payload["last_refresh_error"] is None

def test_openai_codex_status_route_surfaces_refresh_failure_diagnostics(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Failed refresh attempts should be visible in auth status diagnostics."""
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "auth.json").write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": _fake_jwt(1_700_000_000),
                    "refresh_token": "refresh-token",
                    "account_id": "acct_123",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    def _raise_refresh_error(self, _refresh_token: str) -> dict[str, Any]:
        raise RuntimeError("invalid_grant")

    monkeypatch.setattr(OpenAICodexAuthService, "_request_token_refresh", _raise_refresh_error)
    workspace_dir = tmp_path / "workspace"
    app, _engine = _build_app(workspace_dir=workspace_dir)

    with TestClient(app) as client:
        import_response = client.post("/v1/model-auth/openai-codex/import-codex-cli")
        refresh_response = client.post("/v1/model-auth/openai-codex/refresh")
        status_response = client.get("/v1/model-auth/openai-codex/status")

    assert import_response.status_code == 200
    assert refresh_response.status_code == 503
    assert "refresh failed" in refresh_response.json()["detail"].lower()
    assert status_response.status_code == 200
    status_payload = status_response.json()
    assert status_payload["status"] == "refresh_failed"
    assert status_payload["last_refresh_error"] == "invalid_grant"

def test_stale_openai_codex_profile_id_is_not_reported_as_configured(tmp_path: Path) -> None:
    """Missing auth profiles should not keep Codex selected in status or settings payloads."""
    workspace_dir = tmp_path / "workspace"
    app = create_app()
    fake_engine = _FakeEngine()
    app.state.engine_factory = lambda: fake_engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "https://chatgpt.com/backend-api",
            "workspace_dir": str(workspace_dir),
            "openai_codex": {
                "base_url": "https://chatgpt.com/backend-api",
                "model": "gpt-5.4",
                "auth_profile_id": "missing-profile",
            },
        }
    )

    with TestClient(app) as client:
        status_response = client.get("/v1/model-auth/openai-codex/status")
        settings_response = client.get("/v1/settings")
        models_response = client.get("/v1/models")

    assert status_response.status_code == 200
    assert settings_response.status_code == 200
    assert models_response.status_code == 200

    status_payload = status_response.json()
    settings_payload = settings_response.json()
    models_payload = models_response.json()

    assert status_payload["configured"] is False
    assert status_payload["active_profile_id"] is None
    assert settings_payload["model_config"]["provider"] == "openai_compat"
    assert settings_payload["model_config"]["openai_codex_auth_profile_id"] is None
    assert settings_payload["model_config"]["openai_codex_auth_configured"] is False
    assert models_payload["configured_remote_provider"] == "openai_compat"

def test_openai_codex_logout_route_removes_active_profile(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Logout should remove the imported profile and clear active auth state."""
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "auth.json").write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": "header.eyJleHAiOjE5MDAwMDAwMDAsImVtYWlsIjoiY29kZXhAZXhhbXBsZS5jb20ifQ.sig",
                    "refresh_token": "refresh-token",
                    "account_id": "acct_123",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    workspace_dir = tmp_path / "workspace"
    app, _engine = _build_app(workspace_dir=workspace_dir)

    with TestClient(app) as client:
        import_response = client.post("/v1/model-auth/openai-codex/import-codex-cli")
        logout_response = client.post("/v1/model-auth/openai-codex/logout")
        status_response = client.get("/v1/model-auth/openai-codex/status")

    assert import_response.status_code == 200
    assert logout_response.status_code == 200
    assert status_response.status_code == 200
    logout_payload = logout_response.json()
    status_payload = status_response.json()
    assert logout_payload["deleted"] is True
    assert logout_payload["active_profile_id"] is None
    assert "store_path" not in logout_response.text
    assert status_payload["configured"] is False
    assert status_payload["active_profile_id"] is None
    assert status_payload["profiles"] == []
