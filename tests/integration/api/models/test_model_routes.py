"""Model API route integration tests."""

from __future__ import annotations

from ._support import *  # noqa: F401,F403

def test_models_route_returns_active_model_without_leaking_secrets() -> None:
    """`GET /v1/models` 應只回傳非敏感模型資訊。"""
    app, _engine = _build_app()

    with TestClient(app) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "models_status"
    assert payload["configured_model"] == "ollama:configured"
    assert payload["active_model"] == {
        "name": "ollama:test",
        "backend_type": "ollama",
        "context_length": 8192,
        "supports_tool_calling": True,
        "metadata": {"provider": "fake"},
    }
    assert payload["available_models"] == [
        {
            "id": "ollama:configured",
            "provider": "ollama",
            "model": "configured",
            "model_spec": "ollama:configured",
            "base_url": "http://localhost:11434",
            "label": "configured",
            "backend_type": "ollama",
            "api_key_configured": False,
        }
    ]
    assert [item["pattern"] for item in payload["supported_model_spec_formats"]] == [
        "ollama:<model>",
        "/path/to/model.gguf",
        "/path/to/model_dir/",
        "https://host/v1",
    ]
    assert "secret-token" not in response.text

def test_models_switch_route_calls_engine_switch_model() -> None:
    """`POST /v1/models/switch` 應委派給 engine.switch_model。"""
    app, engine = _build_app()

    with TestClient(app) as client:
        response = client.post("/v1/models/switch", json={"model": "/models/demo.gguf"})

    assert response.status_code == 200
    assert engine.switch_calls == ["/models/demo.gguf"]
    assert response.json() == {
        "type": "model_switch",
        "active_model": {
            "name": "/models/demo.gguf",
            "backend_type": "gguf",
            "context_length": 4096,
            "supports_tool_calling": False,
            "metadata": {"switched": True},
        },
    }

def test_models_configure_route_supports_ollama_without_leaking_key() -> None:
    """`POST /v1/models/configure` 應支援 Ollama base_url/model 設定。"""
    app, engine = _build_app()

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/configure",
            json={
                "provider": "ollama",
                "base_url": "http://localhost:11434",
                "model": "qwen2.5",
            },
        )

    assert response.status_code == 200
    assert engine.ollama_switch_calls == [("qwen2.5", "http://localhost:11434")]
    assert response.json()["provider"] == "ollama"
    assert response.json()["api_key_configured"] is False
    assert response.json()["active_model"]["name"] == "qwen2.5"
    assert response.json()["available_models"][0]["id"] == "ollama:qwen2.5"
    assert response.json()["available_models"][0]["model"] == "qwen2.5"

def test_models_configure_route_supports_openai_compat_without_returning_api_key() -> None:
    """`POST /v1/models/configure` 應接收 API key 但不得回傳原文。"""
    app, engine = _build_app()

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/configure",
            json={
                "provider": "openai_compat",
                "base_url": "https://api.example.com/v1",
                "model": "gpt-test",
                "api_key": "sk-secret-value",
            },
        )

    assert response.status_code == 200
    assert engine.openai_switch_calls == [
        ("https://api.example.com/v1", "gpt-test", "sk-secret-value", "openai_compat")
    ]
    assert response.json()["provider"] == "openai_compat"
    assert response.json()["api_key_configured"] is True
    assert response.json()["active_model"]["name"] == "gpt-test"
    assert response.json()["available_models"][0]["id"] == (
        "openai_compat:https://api.example.com/v1:gpt-test"
    )
    assert response.json()["available_models"][0]["model_spec"] == "https://api.example.com/v1"
    assert "sk-secret-value" not in response.text

def test_models_configure_route_supports_openai_codex_without_leaking_oauth_tokens(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """OpenAI Codex configure should use auth_profile_id and keep tokens out of config.yaml."""
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
    config_path = tmp_path / "config.yaml"
    app, engine = _build_app(workspace_dir=workspace_dir, config_path=config_path)

    with TestClient(app) as client:
        import_response = client.post("/v1/model-auth/openai-codex/import-codex-cli")
        response = client.post(
            "/v1/models/configure",
            json={
                "provider": "openai_codex",
                "base_url": "https://chatgpt.com/backend-api",
                "model": "gpt-5.4",
            },
        )

    assert import_response.status_code == 200
    assert response.status_code == 200
    assert engine.openai_codex_switch_calls == [
        ("https://chatgpt.com/backend-api", "gpt-5.4", "openai_codex:default")
    ]
    payload = response.json()
    assert payload["provider"] == "openai_codex"
    assert payload["api_key_configured"] is False
    assert payload["available_models"][0]["auth_profile_id"] == "openai_codex:default"
    assert payload["available_models"][0]["auth_mode"] == "oauth"
    saved = config_path.read_text(encoding="utf-8")
    assert "auth_profile_id: openai_codex:default" in saved
    assert "refresh-token" not in saved
    assert "access_token" not in saved

def test_models_configure_route_rejects_non_official_openai_codex_base_url(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """OpenAI Codex OAuth tokens must never be routed to arbitrary hosts."""
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
    app, engine = _build_app(workspace_dir=workspace_dir)

    with TestClient(app) as client:
        import_response = client.post("/v1/model-auth/openai-codex/import-codex-cli")
        response = client.post(
            "/v1/models/configure",
            json={
                "provider": "openai_codex",
                "base_url": "https://example.invalid/backend-api",
                "model": "gpt-5.4",
            },
        )

    assert import_response.status_code == 200
    assert response.status_code == 400
    assert "official ChatGPT backend endpoint" in response.json()["detail"]
    assert engine.openai_codex_switch_calls == []

def test_models_configure_route_appends_to_available_models() -> None:
    """多次成功設定後 `/v1/models` 應回傳可供聊天頁選擇的模型列表。"""
    app, engine = _build_app()

    with TestClient(app) as client:
        first_response = client.post(
            "/v1/models/configure",
            json={
                "provider": "ollama",
                "base_url": "http://localhost:11434",
                "model": "qwen2.5",
            },
        )
        second_response = client.post(
            "/v1/models/configure",
            json={
                "provider": "openai_compat",
                "base_url": "https://api.example.com/v1",
                "model": "gpt-test",
                "api_key": "sk-secret-value",
            },
        )
        models_response = client.get("/v1/models")

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert models_response.status_code == 200
    assert engine.ollama_switch_calls == [("qwen2.5", "http://localhost:11434")]
    assert engine.openai_switch_calls == [
        ("https://api.example.com/v1", "gpt-test", "sk-secret-value", "openai_compat")
    ]
    assert [model["id"] for model in models_response.json()["available_models"]] == [
        "openai_compat:https://api.example.com/v1:gpt-test",
        "ollama:qwen2.5",
    ]
    assert "sk-secret-value" not in models_response.text

def test_models_switch_route_accepts_available_model_id() -> None:
    """聊天頁模型下拉以 available model id 切換時應還原 provider/model/base_url。"""
    config = MochiConfig.model_validate(
        {
            "model": "ollama:qwen2.5",
            "model_setup": {
                "configured_models": [
                    {
                        "id": "openai_compat:https://api.example.com/v1:gpt-test",
                        "provider": "openai_compat",
                        "model": "gpt-test",
                        "model_spec": "https://api.example.com/v1",
                        "base_url": "https://api.example.com/v1",
                        "label": "gpt-test (openai_compat)",
                        "backend_type": "openai_compat",
                    },
                    {
                        "id": "ollama:qwen2.5",
                        "provider": "ollama",
                        "model": "qwen2.5",
                        "model_spec": "ollama:qwen2.5",
                        "base_url": "http://localhost:11434",
                        "label": "qwen2.5",
                        "backend_type": "ollama",
                    },
                ]
            },
            "openai_compat": {
                "provider": "openai_compat",
                "base_url": "https://api.example.com/v1",
                "model": "gpt-test",
                "api_key": "sk-secret-value",
            },
        }
    )
    app, engine = _build_app()
    app.state.config_factory = lambda: config

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/switch",
            json={"model": "openai_compat:https://api.example.com/v1:gpt-test"},
        )

    assert response.status_code == 200
    assert engine.openai_switch_calls == [
        ("https://api.example.com/v1", "gpt-test", "sk-secret-value", "openai_compat")
    ]
    assert response.json()["active_model"]["name"] == "gpt-test"
    assert "sk-secret-value" not in response.text

def test_models_switch_route_accepts_bare_ollama_model_name() -> None:
    """Ollama active model 回傳裸模型名時，switch API 應還原到已設定清單項目。"""
    config = MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "ollama": {"base_url": "http://localhost:11434"},
            "model_setup": {
                "configured_models": [
                    {
                        "id": "ollama:qwen2.5",
                        "provider": "ollama",
                        "model": "qwen2.5",
                        "model_spec": "ollama:qwen2.5",
                        "base_url": "http://localhost:11434",
                        "label": "qwen2.5",
                        "backend_type": "ollama",
                    },
                ]
            },
        }
    )
    app, engine = _build_app()
    app.state.config_factory = lambda: config

    with TestClient(app) as client:
        response = client.post("/v1/models/switch", json={"model": "qwen2.5"})

    assert response.status_code == 200
    assert engine.ollama_switch_calls == [("qwen2.5", "http://localhost:11434")]
    assert engine.switch_calls == []
    assert response.json()["active_model"]["name"] == "qwen2.5"

def test_models_switch_route_accepts_bare_active_ollama_model_name() -> None:
    """已 active 的 Ollama 模型以裸名切換時不應落到通用 switch_model。"""
    config = MochiConfig.model_validate(
        {
            "model": "ollama:qwen2.5",
            "ollama": {"base_url": "http://localhost:11434"},
            "model_setup": {
                "configured_models": [
                    {
                        "id": "ollama:qwen2.5",
                        "provider": "ollama",
                        "model": "qwen2.5",
                        "model_spec": "ollama:qwen2.5",
                        "base_url": "http://localhost:11434",
                        "label": "qwen2.5",
                        "backend_type": "ollama",
                    },
                ]
            },
        }
    )
    app, engine = _build_app()
    app.state.config_factory = lambda: config

    with TestClient(app) as client:
        response = client.post("/v1/models/switch", json={"model": "qwen2.5"})

    assert response.status_code == 200
    assert engine.ollama_switch_calls == []
    assert engine.switch_calls == []

def test_models_configure_route_persists_ollama_selection(tmp_path: Path) -> None:
    """成功切換模型後應保存到指定 YAML，避免重啟後回到預設模型。"""
    config_path = tmp_path / "config.yaml"
    app, engine = _build_app(config_path=config_path)

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/configure",
            json={
                "provider": "ollama",
                "base_url": "http://localhost:11434",
                "model": "qwen2.5",
            },
        )

    assert response.status_code == 200
    assert engine.ollama_switch_calls == [("qwen2.5", "http://localhost:11434")]
    assert response.json()["persisted"] is True
    assert "qwen2.5" in config_path.read_text(encoding="utf-8")
    assert "ollama:qwen2.5" in config_path.read_text(encoding="utf-8")

def test_models_configure_route_persists_multiple_available_models(tmp_path: Path) -> None:
    """連續新增模型後，重載 YAML 應保留完整可用模型清單。"""
    config_path = tmp_path / "config.yaml"
    app, engine = _build_app(config_path=config_path)

    with TestClient(app) as client:
        first_response = client.post(
            "/v1/models/configure",
            json={
                "provider": "ollama",
                "base_url": "http://localhost:11434",
                "model": "qwen2.5",
            },
        )
        second_response = client.post(
            "/v1/models/configure",
            json={
                "provider": "openai_compat",
                "base_url": "https://api.example.com/v1",
                "model": "gpt-test",
                "api_key": "sk-secret-value",
            },
        )
        models_response = client.get("/v1/models")

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert models_response.status_code == 200
    assert engine.ollama_switch_calls == [("qwen2.5", "http://localhost:11434")]
    assert engine.openai_switch_calls == [
        ("https://api.example.com/v1", "gpt-test", "sk-secret-value", "openai_compat")
    ]

    expected_ids = [
        "openai_compat:https://api.example.com/v1:gpt-test",
        "ollama:qwen2.5",
    ]
    assert [model["id"] for model in second_response.json()["available_models"]] == expected_ids
    assert [model["id"] for model in models_response.json()["available_models"]] == expected_ids

    saved_config = load_config(config_path)
    assert [model.id for model in saved_config.model_setup.configured_models] == expected_ids
    assert saved_config.model == "https://api.example.com/v1"
    assert saved_config.openai_compat.model == "gpt-test"
    assert "sk-secret-value" not in models_response.text

def test_models_configure_route_supports_gemini_preset_without_leaking_key() -> None:
    """Gemini provider preset 應走 OpenAI-compatible backend 並保存 provider。"""
    app, engine = _build_app()

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/configure",
            json={
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "api_key": "gemini-secret",
            },
        )

    assert response.status_code == 200
    assert engine.openai_switch_calls == [
        (
            "https://generativelanguage.googleapis.com/v1beta/openai",
            "gemini-2.5-flash",
            "gemini-secret",
            "gemini",
        )
    ]
    assert response.json()["provider"] == "gemini"
    assert response.json()["active_model"]["name"] == "gemini-2.5-flash"
    assert "gemini-secret" not in response.text

def test_models_configure_route_supports_vllm_preset_without_leaking_key() -> None:
    """vLLM provider preset 應走 OpenAI-compatible backend 並保存 provider。"""
    app, engine = _build_app()

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/configure",
            json={
                "provider": "vllm",
                "model": "qwen2.5-7b-instruct",
                "api_key": "vllm-secret",
            },
        )

    assert response.status_code == 200
    assert engine.openai_switch_calls == [
        (
            "http://localhost:8000/v1",
            "qwen2.5-7b-instruct",
            "vllm-secret",
            "vllm",
        )
    ]
    payload = response.json()
    assert payload["provider"] == "vllm"
    assert payload["active_model"]["name"] == "qwen2.5-7b-instruct"
    assert payload["available_models"][0]["provider"] == "vllm"
    assert payload["available_models"][0]["backend_type"] == "openai_compat"
    assert payload["available_models"][0]["id"] == (
        "vllm:http://localhost:8000/v1:qwen2.5-7b-instruct"
    )
    assert "vllm-secret" not in response.text

def test_models_configured_patch_updates_vllm_remote_entry_without_leaking_api_key() -> None:
    """`PATCH /v1/models/configured/{id}` 應支援更新 vLLM remote entry。"""
    config = MochiConfig.model_validate(
        {
            "model": "http://localhost:8000/v1",
            "openai_compat": {
                "provider": "vllm",
                "base_url": "http://localhost:8000/v1",
                "model": "qwen2.5-7b-instruct",
                "api_key": "vllm-old-secret",
            },
            "model_setup": {
                "configured_models": [
                    {
                        "id": "vllm:http://localhost:8000/v1:qwen2.5-7b-instruct",
                        "provider": "vllm",
                        "model": "qwen2.5-7b-instruct",
                        "model_spec": "http://localhost:8000/v1",
                        "base_url": "http://localhost:8000/v1",
                        "label": "qwen2.5-7b-instruct (vllm)",
                        "backend_type": "openai_compat",
                    }
                ]
            },
        }
    )
    app, _engine = _build_app()
    app.state.config_factory = lambda: config

    with TestClient(app) as client:
        response = client.patch(
            "/v1/models/configured/vllm%3Ahttp%3A%2F%2Flocalhost%3A8000%2Fv1%3Aqwen2.5-7b-instruct",
            json={
                "provider": "vllm",
                "model": "qwen3-8b",
                "model_spec": "http://localhost:9000/v1",
                "base_url": "http://localhost:9000/v1",
                "api_key": "vllm-new-secret",
                "persist": False,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "model_entry_update"
    assert payload["updated_model"]["provider"] == "vllm"
    assert payload["updated_model"]["model"] == "qwen3-8b"
    assert payload["updated_model"]["model_spec"] == "http://localhost:9000/v1"
    assert payload["updated_model"]["base_url"] == "http://localhost:9000/v1"
    assert payload["updated_model"]["id"] == "vllm:http://localhost:9000/v1:qwen3-8b"
    assert payload["updated_model"]["backend_type"] == "openai_compat"
    assert payload["updated_model"]["launch_mode"] == "external"
    assert payload["api_key_configured"] is True
    assert payload["configured_model"] == "http://localhost:9000/v1"
    assert "vllm-new-secret" not in response.text
    assert "vllm-old-secret" not in response.text

def test_models_probe_tool_calling_returns_probe_payload() -> None:
    app, fake_engine = _build_app()
    fake_engine.model_info = ModelInfo(
        name="google/gemma-4-26B-A4B-it",
        provider="vllm",
        backend_type="openai_compat",
        supports_tool_calling=False,
        metadata={
            "tool_call_mode": "simulated_fallback",
            "native_tool_calling_status": "rejected_missing_parser",
        },
    )
    fake_engine.tool_probe_result = {
        "status": "rejected_missing_parser",
        "message": "vLLM rejected native auto tool choice.",
    }

    with TestClient(app) as client:
        response = client.post("/v1/models/probe-tool-calling")

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "tool_calling_probe"
    assert payload["probe"]["status"] == "rejected_missing_parser"
    assert payload["active_model"]["metadata"]["tool_call_mode"] == "simulated_fallback"

def test_models_probe_tool_calling_returns_post_probe_active_model_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = MochiConfig.model_validate(
        {
            "model": "ollama:qwen2.5",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db")},
        }
    )
    app, _fake_engine = _build_app()
    real_engine = AgentEngine(config)

    class _ProbeBackend:
        def __init__(self) -> None:
            self.metadata = {
                "tool_call_mode": "simulated_fallback",
                "native_tool_calling_status": "native_tool_calls_missing",
            }
            self.probe_calls = 0
            self.closed = False
            self.close_calls = 0

        async def probe_tool_calling(self) -> dict[str, Any] | None:
            self.probe_calls += 1
            self.metadata.update(
                {
                    "tool_call_mode": "native",
                    "native_tool_calling_status": "supported",
                }
            )
            return {
                "status": "supported",
                "message": "native structured tool calling succeeded",
                "metadata": dict(self.metadata),
            }

        def get_model_info(self) -> ModelInfo:
            supports_tool_calling = not (
                self.metadata.get("tool_call_mode") == "unavailable"
                or self.metadata.get("tool_calling_blocked") is True
            )
            return ModelInfo(
                name="qwen2.5",
                provider="ollama",
                backend_type="ollama",
                supports_tool_calling=supports_tool_calling,
                metadata=dict(self.metadata),
            )

        async def close(self) -> None:
            self.close_calls += 1
            self.closed = True

    class _StaleInfoBackend:
        def __init__(self) -> None:
            self.get_model_info_calls = 0

        def get_model_info(self) -> ModelInfo:
            self.get_model_info_calls += 1
            return ModelInfo(
                name="qwen2.5",
                provider="ollama",
                backend_type="ollama",
                supports_tool_calling=True,
                metadata={
                    "tool_call_mode": "simulated_fallback",
                    "native_tool_calling_status": "native_tool_calls_missing",
                },
            )

    probe_backend = _ProbeBackend()
    stale_info_backend = _StaleInfoBackend()
    resolve_calls = 0

    async def fake_acquire_temporary_backend(*, model_spec: str, **kwargs: Any) -> _ProbeBackend:
        del model_spec, kwargs
        return probe_backend

    def fake_resolve(model_spec: str, **kwargs: Any) -> _StaleInfoBackend:
        nonlocal resolve_calls
        del model_spec, kwargs
        resolve_calls += 1
        return stale_info_backend

    monkeypatch.setattr(real_engine._router, "acquire_temporary_backend", fake_acquire_temporary_backend)  # noqa: SLF001
    monkeypatch.setattr(real_engine._router, "_resolve", fake_resolve)  # noqa: SLF001
    app.state.engine = real_engine

    with TestClient(app) as client:
        response = client.post("/v1/models/probe-tool-calling")

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "tool_calling_probe"
    assert payload["probe"]["status"] == "supported"
    assert payload["active_model"]["metadata"]["tool_call_mode"] == "native"
    assert payload["active_model"]["metadata"]["native_tool_calling_status"] == "supported"
    assert payload["active_model"]["supports_tool_calling"] is True
    assert probe_backend.probe_calls == 1
    assert probe_backend.close_calls == 1
    assert probe_backend.closed is True
    assert resolve_calls == 0
    assert stale_info_backend.get_model_info_calls == 0

def test_models_test_connection_route_validates_explicit_remote_payload_without_switching() -> None:
    app, fake_engine = _build_app()

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/test-connection",
            json={
                "provider": "openai_compat",
                "model": "gpt-4.1-mini",
                "base_url": "https://api.example.com/v1",
                "api_key": "test-secret",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "model_connection_test"
    assert payload["provider"] == "openai_compat"
    assert payload["tested_model"]["model_spec"] == "https://api.example.com/v1"
    assert payload["tested_model"]["base_url"] == "https://api.example.com/v1"
    assert payload["tested_model"]["metadata"]["tested"] is True
    assert fake_engine.test_connection_calls == [
        {
            "provider": "openai_compat",
            "model": "gpt-4.1-mini",
            "base_url": "https://api.example.com/v1",
            "api_key": "test-secret",
            "auth_profile_id": None,
        }
    ]
    assert fake_engine.switch_calls == []
    assert fake_engine.openai_switch_calls == []

def test_models_test_connection_route_supports_saved_model_id() -> None:
    config = MochiConfig.model_validate(
        {
            "model": "https://api.example.com/v1",
            "openai_compat": {
                "provider": "openai_compat",
                "base_url": "https://api.example.com/v1",
                "model": "gpt-4.1-mini",
                "api_key": "saved-secret",
            },
            "model_setup": {
                "configured_models": [
                    {
                        "id": "openai_compat:https://api.example.com/v1:gpt-4.1-mini",
                        "provider": "openai_compat",
                        "model": "gpt-4.1-mini",
                        "model_spec": "https://api.example.com/v1",
                        "base_url": "https://api.example.com/v1",
                        "label": "gpt-4.1-mini (openai_compat)",
                        "backend_type": "openai_compat",
                    }
                ]
            },
        }
    )
    app, fake_engine = _build_app()
    app.state.config_factory = lambda: config

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/test-connection",
            json={
                "model_id": "openai_compat:https://api.example.com/v1:gpt-4.1-mini",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "openai_compat"
    assert payload["tested_model"]["id"] == "openai_compat:https://api.example.com/v1:gpt-4.1-mini"
    assert payload["tested_model"]["base_url"] == "https://api.example.com/v1"
    assert fake_engine.test_connection_calls == [
        {
            "provider": "openai_compat",
            "model": "gpt-4.1-mini",
            "base_url": "https://api.example.com/v1",
            "api_key": "saved-secret",
            "auth_profile_id": None,
        }
    ]
    assert fake_engine.switch_calls == []
    assert fake_engine.openai_switch_calls == []

def test_models_test_connection_route_prefers_saved_model_api_key_over_global_runtime_key() -> None:
    config = MochiConfig.model_validate(
        {
            "model": "https://active.example.com/v1",
            "openai_compat": {
                "provider": "openai_compat",
                "base_url": "https://active.example.com/v1",
                "model": "active-model",
                "api_key": "runtime-active-secret",
            },
            "model_setup": {
                "configured_models": [
                    {
                        "id": "openai_compat:https://api.example.com/v1:gpt-4.1-mini",
                        "provider": "openai_compat",
                        "model": "gpt-4.1-mini",
                        "model_spec": "https://api.example.com/v1",
                        "base_url": "https://api.example.com/v1",
                        "label": "gpt-4.1-mini (openai_compat)",
                        "backend_type": "openai_compat",
                        "api_key": "saved-model-secret",
                    }
                ]
            },
        }
    )
    app, fake_engine = _build_app()
    app.state.config_factory = lambda: config

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/test-connection",
            json={
                "model_id": "openai_compat:https://api.example.com/v1:gpt-4.1-mini",
            },
        )

    assert response.status_code == 200
    assert fake_engine.test_connection_calls == [
        {
            "provider": "openai_compat",
            "model": "gpt-4.1-mini",
            "base_url": "https://api.example.com/v1",
            "api_key": "saved-model-secret",
            "auth_profile_id": None,
        }
    ]
    assert "saved-model-secret" not in response.text
    assert "runtime-active-secret" not in response.text

def test_models_configured_patch_updates_remote_entry_without_leaking_api_key() -> None:
    """`PATCH /v1/models/configured/{id}` 應可更新 remote entry 並保留 secret 規則。"""
    config = MochiConfig.model_validate(
        {
            "model": "https://api.example.com/v1",
            "openai_compat": {
                "provider": "openai_compat",
                "base_url": "https://api.example.com/v1",
                "model": "gpt-test",
                "api_key": "sk-old-secret",
            },
            "model_setup": {
                "configured_models": [
                    {
                        "id": "openai_compat:https://api.example.com/v1:gpt-test",
                        "provider": "openai_compat",
                        "model": "gpt-test",
                        "model_spec": "https://api.example.com/v1",
                        "base_url": "https://api.example.com/v1",
                        "label": "gpt-test (openai_compat)",
                        "backend_type": "openai_compat",
                    }
                ]
            },
        }
    )
    app, _engine = _build_app()
    app.state.config_factory = lambda: config

    with TestClient(app) as client:
        response = client.patch(
            "/v1/models/configured/openai_compat%3Ahttps%3A%2F%2Fapi.example.com%2Fv1%3Agpt-test",
            json={
                "provider": "openai_compat",
                "model": "gpt-new",
                "model_spec": "https://api.new-example.com/v1",
                "base_url": "https://api.new-example.com/v1",
                "api_key": "sk-new-secret",
                "persist": False,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "model_entry_update"
    assert payload["updated_model"]["model"] == "gpt-new"
    assert payload["updated_model"]["model_spec"] == "https://api.new-example.com/v1"
    assert payload["updated_model"]["base_url"] == "https://api.new-example.com/v1"
    assert payload["api_key_configured"] is True
    assert payload["configured_model"] == "https://api.new-example.com/v1"
    assert "sk-new-secret" not in response.text
    assert "sk-old-secret" not in response.text

def test_models_configured_patch_updates_local_entry_path(tmp_path: Path) -> None:
    """`PATCH /v1/models/configured/{id}` 應可更新 local entry 路徑。"""
    first = tmp_path / "first.gguf"
    second = tmp_path / "second.gguf"
    first.write_text("gguf", encoding="utf-8")
    second.write_text("gguf", encoding="utf-8")

    config = MochiConfig.model_validate(
        {
            "model": str(first.resolve()),
            "local_models": {"roots": [str(tmp_path)]},
            "model_setup": {
                "configured_models": [
                    {
                        "id": str(first.resolve()),
                        "provider": "local",
                        "model": first.name,
                        "model_spec": str(first.resolve()),
                        "label": first.name,
                        "backend_type": "gguf",
                    }
                ]
            },
        }
    )
    app, _engine = _build_app()
    app.state.config_factory = lambda: config

    with TestClient(app) as client:
        response = client.patch(
            f"/v1/models/configured/{first.resolve()}",
            json={
                "provider": "local",
                "model": second.name,
                "model_spec": str(second.resolve()),
                "persist": False,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["updated_model"]["provider"] == "local"
    assert payload["updated_model"]["model_spec"] == str(second.resolve())
    assert payload["configured_model"] == str(second.resolve())
    assert payload["api_key_configured"] is False

def test_models_configured_delete_removes_entry_and_falls_back_to_remaining_model() -> None:
    """`DELETE /v1/models/configured/{id}` 應刪除指定 entry 並維持可用 configured model。"""
    config = MochiConfig.model_validate(
        {
            "model": "ollama:qwen2.5",
            "ollama": {"base_url": "http://localhost:11434"},
            "model_setup": {
                "configured_models": [
                    {
                        "id": "ollama:qwen2.5",
                        "provider": "ollama",
                        "model": "qwen2.5",
                        "model_spec": "ollama:qwen2.5",
                        "base_url": "http://localhost:11434",
                        "label": "qwen2.5",
                        "backend_type": "ollama",
                    },
                    {
                        "id": "openai_compat:https://api.example.com/v1:gpt-test",
                        "provider": "openai_compat",
                        "model": "gpt-test",
                        "model_spec": "https://api.example.com/v1",
                        "base_url": "https://api.example.com/v1",
                        "label": "gpt-test (openai_compat)",
                        "backend_type": "openai_compat",
                    },
                ]
            },
        }
    )
    app, _engine = _build_app()
    app.state.config_factory = lambda: config

    with TestClient(app) as client:
        response = client.request(
            "DELETE",
            "/v1/models/configured/ollama%3Aqwen2.5",
            json={"persist": False},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "model_entry_delete"
    assert payload["deleted_model_id"] == "ollama:qwen2.5"
    assert payload["configured_model"] == "https://api.example.com/v1"
    assert [item["id"] for item in payload["available_models"]] == [
        "openai_compat:https://api.example.com/v1:gpt-test"
    ]

def test_models_configured_delete_accepts_wsl_alias_for_windows_local_model() -> None:
    """本地模型刪除應接受對應的 WSL 路徑識別。"""
    config = MochiConfig.model_validate(
        {
            "model": r"J:\_models\Qwen3.5-9B",
            "model_setup": {
                "configured_models": [
                    {
                        "id": r"J:\_models\Qwen3.5-9B",
                        "provider": "local",
                        "model": "Qwen3.5-9B",
                        "model_spec": r"J:\_models\Qwen3.5-9B",
                        "label": "Qwen3.5-9B",
                        "backend_type": "safetensors",
                    }
                ]
            },
        }
    )
    app, _engine = _build_app()
    app.state.config_factory = lambda: config

    with TestClient(app) as client:
        response = client.request(
            "DELETE",
            "/v1/models/configured/%2Fmnt%2Fj%2F_models%2FQwen3.5-9B",
            json={"persist": False},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["deleted_model_id"] == r"J:\_models\Qwen3.5-9B"
    assert payload["available_models"] == []

def test_models_configured_delete_active_last_remote_resets_to_default_model() -> None:
    """Deleting the active last saved remote model should stop it from being rehydrated as a saved entry."""
    config = MochiConfig.model_validate(
        {
            "model": "https://co.yes.vg/v1",
            "openai_compat": {
                "provider": "openai_compat",
                "base_url": "https://co.yes.vg/v1",
                "model": "gpt-5.4",
            },
            "model_setup": {
                "default_provider": "ollama",
                "default_model": "llama3.2",
                "default_model_spec": "ollama:llama3.2",
                "configured_models": [
                    {
                        "id": "openai_compat:https://co.yes.vg/v1:gpt-5.4",
                        "provider": "openai_compat",
                        "model": "gpt-5.4",
                        "model_spec": "https://co.yes.vg/v1",
                        "base_url": "https://co.yes.vg/v1",
                        "label": "gpt-5.4 (openai_compat)",
                        "backend_type": "openai_compat",
                    }
                ],
            },
        }
    )
    app, engine = _build_app()
    app.state.config_factory = lambda: config

    with TestClient(app) as client:
        response = client.request(
            "DELETE",
            "/v1/models/configured/openai_compat%3Ahttps%3A%2F%2Fco.yes.vg%2Fv1%3Agpt-5.4",
            json={"persist": False},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["deleted_model_id"] == "openai_compat:https://co.yes.vg/v1:gpt-5.4"
        assert payload["available_models"] == []
        assert payload["configured_model"] == "ollama:llama3.2"
        assert any(
            item["name"] == "active_configured_model_deleted"
            and item["reason"] == "deleted_active_model_switched_to_default_model"
            and item["from"] == "openai_compat:https://co.yes.vg/v1:gpt-5.4"
            and item["to"] == "ollama:llama3.2"
            for item in payload["diagnostics"]
        )

        models_response = client.get("/v1/models")

    assert models_response.status_code == 200
    models_payload = models_response.json()
    assert models_payload["configured_model"] == "ollama:llama3.2"
    assert all(
        item["id"] != "openai_compat:https://co.yes.vg/v1:gpt-5.4"
        for item in models_payload["available_models"]
    )
    assert engine.apply_config_calls[-1] == ("ollama:llama3.2", False)

def test_models_route_does_not_duplicate_local_model_when_wsl_path_alias_matches_windows_config() -> None:
    """`GET /v1/models` 不應因 Windows/WSL 路徑別名再補一筆 local fallback。"""
    config = MochiConfig.model_validate(
        {
            "model": "/mnt/j/_models/Qwen3.5-9B",
            "model_setup": {
                "configured_models": [
                    {
                        "id": r"J:\_models\Qwen3.5-9B",
                        "provider": "local",
                        "model": "Qwen3.5-9B",
                        "model_spec": r"J:\_models\Qwen3.5-9B",
                        "label": "Qwen3.5-9B",
                        "backend_type": "safetensors",
                    }
                ]
            },
        }
    )
    app, _engine = _build_app()
    app.state.config_factory = lambda: config

    with TestClient(app) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["available_models"]) == 1
    assert payload["available_models"][0]["id"] == r"J:\_models\Qwen3.5-9B"

def test_dump_saved_configured_models_returns_only_explicit_entries() -> None:
    """edit/delete response 應只回傳實際保存的 configured models。"""
    from mochi.api.routes.models import _dump_saved_configured_models

    config = MochiConfig.model_validate(
        {
            "model": "/mnt/j/_models/Qwen3.5-9B",
            "model_setup": {
                "configured_models": [
                    {
                        "id": r"J:\_models\Qwen3.5-9B",
                        "provider": "local",
                        "model": "Qwen3.5-9B",
                        "model_spec": r"J:\_models\Qwen3.5-9B",
                        "label": "Qwen3.5-9B",
                        "backend_type": "safetensors",
                    }
                ]
            },
        }
    )

    payload = _dump_saved_configured_models(config)
    assert len(payload) == 1
    assert payload[0]["id"] == r"J:\_models\Qwen3.5-9B"

def test_models_ollama_discovery_returns_model_names(monkeypatch) -> None:
    """`GET /v1/models/ollama` 應解析 Ollama `/api/tags` model names。"""
    app, _engine = _build_app()

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "models": [
                    {"name": "qwen2.5:latest"},
                    {"name": "llama3.2"},
                    {"name": ""},
                    {"id": "ignored"},
                ]
            }

    class _FakeAsyncClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.calls: list[str] = []

        async def __aenter__(self) -> _FakeAsyncClient:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def get(self, url: str) -> _FakeResponse:
            self.calls.append(url)
            return _FakeResponse()

    monkeypatch.setattr("mochi.api.routes.models.httpx.AsyncClient", _FakeAsyncClient)

    with TestClient(app) as client:
        response = client.get("/v1/models/ollama", params={"base_url": "http://localhost:11434"})

    assert response.status_code == 200
    assert response.json() == {
        "type": "ollama_models",
        "base_url": "http://localhost:11434",
        "models": ["llama3.2", "qwen2.5:latest"],
    }

def test_models_configure_route_supports_local_provider(tmp_path: Path) -> None:
    """`POST /v1/models/configure` 應支援 local provider。"""
    gguf = tmp_path / "demo.gguf"
    gguf.write_text("gguf", encoding="utf-8")
    app, engine = _build_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "local_models": {
                "roots": [str(tmp_path)],
            },
        }
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/configure",
            json={
                "provider": "local",
                "model": str(gguf),
            },
        )

    assert response.status_code == 200
    assert engine.switch_calls == [str(gguf.resolve())]
    payload = response.json()
    assert payload["provider"] == "local"
    assert payload["api_key_configured"] is False
    assert payload["available_models"][0]["provider"] == "local"
    assert payload["available_models"][0]["model_spec"] == str(gguf.resolve())

def test_models_configure_route_returns_503_for_local_runtime_failure(tmp_path: Path) -> None:
    """local provider runtime 無法啟動時，應回傳可讀 API 錯誤而非 500。"""

    class _FailingLocalEngine(_FakeEngine):
        async def switch_model(self, model: str) -> ModelInfo:
            raise RuntimeError(
                f"Backend switch rejected unhealthy backend for '{model}': "
                "Missing dependencies: transformers, accelerate. Install with `uv sync --extra hf`."
            )

    hf_dir = tmp_path / "Qwen3.5-9B"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text("{}", encoding="utf-8")
    (hf_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (hf_dir / "model.safetensors").write_text("x", encoding="utf-8")
    app, _engine = _build_app(engine=_FailingLocalEngine())
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "local_models": {
                "roots": [str(tmp_path)],
            },
        }
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/configure",
            json={
                "provider": "local",
                "model": str(hf_dir),
            },
        )

    assert response.status_code == 503
    assert "Missing dependencies: transformers, accelerate" in response.json()["detail"]

def test_models_route_serializes_saved_local_entries(tmp_path: Path) -> None:
    """`GET /v1/models` 應正確序列化 provider=local 的已保存模型。"""
    gguf = tmp_path / "demo.gguf"
    gguf.write_text("gguf", encoding="utf-8")
    config = MochiConfig.model_validate(
        {
            "model": str(gguf.resolve()),
            "model_setup": {
                "configured_models": [
                    {
                        "id": str(gguf.resolve()),
                        "provider": "local",
                        "model": gguf.name,
                        "model_spec": str(gguf.resolve()),
                        "label": gguf.name,
                        "backend_type": "gguf",
                    }
                ]
            },
        }
    )
    app, _engine = _build_app()
    app.state.config_factory = lambda: config

    with TestClient(app) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["available_models"][0]["provider"] == "local"
    assert payload["available_models"][0]["model_spec"] == str(gguf.resolve())
