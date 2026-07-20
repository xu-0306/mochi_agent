"""Model runtime selection integration tests."""

from __future__ import annotations

from ._support import *  # noqa: F401,F403

@pytest.mark.parametrize(('provider', 'default_base_url'), [('sglang', 'http://localhost:30000/v1'), ('tensorrt_llm', 'http://localhost:8000/v1')])
def test_models_configure_route_supports_external_openai_compat_presets_without_managed_vllm_path(
    provider: str,
    default_base_url: str,
) -> None:
    manager = _FakeManagedVLLMRuntimeManager()
    app, engine = _build_app(vllm_runtime_manager=manager)
    requested_model = "Qwen/Qwen2.5-7B-Instruct"

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/configure",
            json={
                "provider": provider,
                "model": requested_model,
                "api_key": "sk-provider-secret",
            },
        )

    assert response.status_code == 200
    assert manager.start_calls == []
    assert engine.openai_switch_calls == [
        (default_base_url, requested_model, "sk-provider-secret", provider)
    ]
    payload = response.json()
    assert payload["provider"] == provider
    assert payload["api_key_configured"] is True
    assert payload["active_model"]["name"] == requested_model
    assert payload["active_model"]["backend_type"] == "openai_compat"
    assert payload["available_models"][0]["provider"] == provider
    assert payload["available_models"][0]["base_url"] == default_base_url
    assert payload["available_models"][0]["model_spec"] == default_base_url
    assert payload["available_models"][0]["launch_mode"] == "external"
    assert payload["available_models"][0]["backend_type"] == "openai_compat"
    assert "sk-provider-secret" not in response.text

def test_models_vllm_runtime_status_start_stop_with_managed_entry() -> None:
    """vLLM managed runtime endpoints 應回報狀態並可啟停單一 instance。"""
    manager = _FakeManagedVLLMRuntimeManager()
    config = MochiConfig.model_validate(
        {
            "model": "ollama:qwen2.5",
            "model_setup": {
                "configured_models": [
                    {
                        "id": "vllm-managed-qwen",
                        "provider": "vllm",
                        "model": "Qwen/Qwen2.5-7B-Instruct",
                        "model_spec": "Qwen/Qwen2.5-7B-Instruct",
                        "base_url": "http://localhost:8000/v1",
                        "label": "Qwen/Qwen2.5-7B-Instruct (vllm managed)",
                        "backend_type": "openai_compat",
                    }
                ]
            },
        }
    )
    app, _engine = _build_app(vllm_runtime_manager=manager)
    app.state.config_factory = lambda: config

    with TestClient(app) as client:
        status_before = client.get("/v1/models/vllm/runtime")
        started = client.post(
            "/v1/models/vllm/runtime/start",
            json={"model_id": "vllm-managed-qwen"},
        )
        stopped = client.post("/v1/models/vllm/runtime/stop")

    assert status_before.status_code == 200
    assert status_before.json()["running"] is False

    assert started.status_code == 200
    start_payload = started.json()
    assert start_payload["action"] == "start"
    assert start_payload["runtime_status"]["running"] is True
    assert start_payload["runtime_status"]["active_model_spec"] == "Qwen/Qwen2.5-7B-Instruct"
    assert manager.start_calls == [
        ("vllm-managed-qwen", "Qwen/Qwen2.5-7B-Instruct", "http://localhost:8000/v1", "managed")
    ]

    assert stopped.status_code == 200
    stop_payload = stopped.json()
    assert stop_payload["action"] == "stop"
    assert stop_payload["runtime_status"]["running"] is False
    assert manager.stop_calls == 2

def test_models_vllm_runtime_start_rejects_managed_gguf_model_spec() -> None:
    """managed vLLM start 應拒絕 .gguf target。"""
    manager = _FakeManagedVLLMRuntimeManager()
    config = MochiConfig.model_validate(
        {
            "model": "ollama:qwen2.5",
            "model_setup": {
                "configured_models": [
                    {
                        "id": "vllm-managed-gguf",
                        "provider": "vllm",
                        "model": "demo.gguf",
                        "model_spec": "demo.gguf",
                        "base_url": "http://localhost:8000/v1",
                        "label": "demo.gguf (vllm managed)",
                        "backend_type": "openai_compat",
                    }
                ]
            },
        }
    )
    app, _engine = _build_app(vllm_runtime_manager=manager)
    app.state.config_factory = lambda: config

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/vllm/runtime/start",
            json={"model_id": "vllm-managed-gguf"},
        )

    assert response.status_code == 400
    assert ".gguf" in response.json()["detail"]
    assert manager.start_calls == []

def test_models_configure_route_supports_managed_vllm_target_and_starts_runtime() -> None:
    """vLLM managed configure 路徑應啟動 runtime 並走 openai_compat backend。"""
    manager = _FakeManagedVLLMRuntimeManager()
    app, engine = _build_app(vllm_runtime_manager=manager)

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/configure",
            json={
                "provider": "vllm",
                "model": "Qwen/Qwen2.5-7B-Instruct",
            },
        )

    assert response.status_code == 200
    assert manager.start_calls == [
        (None, "Qwen/Qwen2.5-7B-Instruct", "http://localhost:8000/v1", "managed")
    ]
    assert engine.openai_switch_calls == [
        (
            "http://localhost:8000/v1",
            "Qwen/Qwen2.5-7B-Instruct",
            "",
            "vllm",
        )
    ]

    payload = response.json()
    assert payload["provider"] == "vllm"
    assert payload["active_model"]["name"] == "Qwen/Qwen2.5-7B-Instruct"
    assert payload["available_models"][0]["provider"] == "vllm"
    assert payload["available_models"][0]["model_spec"] == "Qwen/Qwen2.5-7B-Instruct"
    assert payload["available_models"][0]["backend_type"] == "openai_compat"

def test_models_switch_route_accepts_available_vllm_model_id() -> None:
    """`POST /v1/models/switch` 應可透過 vLLM configured model id 還原 provider。"""
    config = MochiConfig.model_validate(
        {
            "model": "ollama:qwen2.5",
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
                "provider": "vllm",
                "base_url": "http://localhost:8000/v1",
                "model": "qwen2.5-7b-instruct",
                "api_key": "vllm-secret",
            },
        }
    )
    app, engine = _build_app()
    app.state.config_factory = lambda: config

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/switch",
            json={"model": "vllm:http://localhost:8000/v1:qwen2.5-7b-instruct"},
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
    assert response.json()["active_model"]["name"] == "qwen2.5-7b-instruct"
    assert response.json()["active_model"]["id"] == "vllm:http://localhost:8000/v1:qwen2.5-7b-instruct"
    assert "vllm-secret" not in response.text

@pytest.mark.parametrize('provider', ['sglang', 'tensorrt_llm'])
def test_models_switch_route_keeps_external_provider_out_of_managed_vllm_path(
    provider: str,
) -> None:
    base_url = "http://remote.example.test/v1"
    model_name = "qwen2.5-7b-instruct"
    api_key = "sk-provider-secret"
    config = MochiConfig.model_validate(
        {
            "model": "ollama:qwen2.5",
            "model_setup": {
                "configured_models": [
                    {
                        "id": f"{provider}:{base_url}:{model_name}",
                        "provider": provider,
                        "model": model_name,
                        "model_spec": base_url,
                        "base_url": base_url,
                        "label": f"{model_name} ({provider})",
                        "backend_type": "openai_compat",
                        "launch_mode": "managed",
                    }
                ]
            },
            "openai_compat": {
                "provider": provider,
                "base_url": base_url,
                "model": model_name,
                "api_key": api_key,
            },
        }
    )
    manager = _FakeManagedVLLMRuntimeManager()
    app, engine = _build_app(vllm_runtime_manager=manager)
    app.state.config_factory = lambda: config

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/switch",
            json={"model": f"{provider}:{base_url}:{model_name}"},
        )

    assert response.status_code == 200
    assert manager.start_calls == []
    assert engine.openai_switch_calls == [
        (base_url, model_name, api_key, provider)
    ]
    payload = response.json()
    assert payload["active_model"]["id"] == f"{provider}:{base_url}:{model_name}"
    assert payload["active_model"]["provider"] == provider
    assert payload["active_model"]["backend_type"] == "openai_compat"
    assert "sk-provider-secret" not in response.text

def test_models_switch_route_starts_managed_vllm_runtime_for_managed_entry() -> None:
    """切換到 managed vLLM configured model 時應先啟動 runtime。"""
    manager = _FakeManagedVLLMRuntimeManager()
    config = MochiConfig.model_validate(
        {
            "model": "ollama:qwen2.5",
            "model_setup": {
                "configured_models": [
                    {
                        "id": "vllm-managed-qwen",
                        "provider": "vllm",
                        "model": "Qwen/Qwen2.5-7B-Instruct",
                        "model_spec": "Qwen/Qwen2.5-7B-Instruct",
                        "base_url": "http://localhost:8000/v1",
                        "label": "Qwen/Qwen2.5-7B-Instruct (vllm managed)",
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
                "provider": "vllm",
                "base_url": "http://localhost:8000/v1",
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "api_key": "vllm-secret",
            },
        }
    )
    app, engine = _build_app(vllm_runtime_manager=manager)
    app.state.config_factory = lambda: config

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/switch",
            json={"model": "vllm-managed-qwen"},
        )

    assert response.status_code == 200
    assert manager.start_calls == [
        ("vllm-managed-qwen", "Qwen/Qwen2.5-7B-Instruct", "http://localhost:8000/v1", "managed")
    ]
    assert engine.openai_switch_calls == [
        (
            "http://localhost:8000/v1",
            "Qwen/Qwen2.5-7B-Instruct",
            "vllm-secret",
            "vllm",
        )
    ]
    assert response.json()["active_model"]["name"] == "Qwen/Qwen2.5-7B-Instruct"
    assert response.json()["active_model"]["id"] == "vllm-managed-qwen"

def test_models_status_preserves_configured_vllm_active_model_id() -> None:
    """`GET /v1/models` should keep the configured vLLM model id on the active model payload."""
    config = MochiConfig.model_validate(
        {
            "model": "http://127.0.0.1:18000/v1",
            "openai_compat": {
                "provider": "vllm",
                "base_url": "http://127.0.0.1:18000/v1",
                "model": "google/gemma-4-26B-A4B-it",
                "api_key": "",
            },
            "model_setup": {
                "configured_models": [
                    {
                        "id": "vllm:http://127.0.0.1:18000/v1:google/gemma-4-26B-A4B-it",
                        "provider": "vllm",
                        "model": "google/gemma-4-26B-A4B-it",
                        "model_spec": "http://127.0.0.1:18000/v1",
                        "base_url": "http://127.0.0.1:18000/v1",
                        "label": "google/gemma-4-26B-A4B-it (vllm)",
                        "backend_type": "openai_compat",
                    }
                ]
            },
        }
    )
    app, engine = _build_app()
    engine.model_info = ModelInfo(
        name="google/gemma-4-26B-A4B-it",
        provider="vllm",
        backend_type="openai_compat",
        supports_tool_calling=True,
        metadata={"base_url": "http://127.0.0.1:18000/v1"},
    )
    app.state.config_factory = lambda: config

    with TestClient(app) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["active_model"]["id"] == "vllm:http://127.0.0.1:18000/v1:google/gemma-4-26B-A4B-it"
    assert payload["active_model"]["model_spec"] == "http://127.0.0.1:18000/v1"
    assert payload["active_model"]["base_url"] == "http://127.0.0.1:18000/v1"

def test_models_configured_patch_preserves_managed_vllm_entry_model_spec() -> None:
    """PATCH managed vLLM configured entry 時應保留 managed model_spec 與 launch_mode。"""
    config = MochiConfig.model_validate(
        {
            "model": "http://localhost:8000/v1",
            "openai_compat": {
                "provider": "vllm",
                "base_url": "http://localhost:8000/v1",
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "api_key": "vllm-old-secret",
            },
            "model_setup": {
                "configured_models": [
                    {
                        "id": "vllm-managed-qwen",
                        "provider": "vllm",
                        "model": "Qwen/Qwen2.5-7B-Instruct",
                        "model_spec": "Qwen/Qwen2.5-7B-Instruct",
                        "base_url": "http://localhost:8000/v1",
                        "label": "Qwen/Qwen2.5-7B-Instruct (vllm managed)",
                        "backend_type": "openai_compat",
                        "launch_mode": "managed",
                    }
                ]
            },
        }
    )
    app, _engine = _build_app()
    app.state.config_factory = lambda: config

    with TestClient(app) as client:
        response = client.patch(
            "/v1/models/configured/vllm-managed-qwen",
            json={
                "api_key": "vllm-new-secret",
                "persist": False,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "model_entry_update"
    assert payload["updated_model"]["provider"] == "vllm"
    assert payload["updated_model"]["id"] == "vllm-managed-qwen"
    assert payload["updated_model"]["model_spec"] == "Qwen/Qwen2.5-7B-Instruct"
    assert payload["updated_model"]["launch_mode"] == "managed"
    assert payload["available_models"][0]["id"] == "vllm-managed-qwen"
    assert payload["available_models"][0]["model_spec"] == "Qwen/Qwen2.5-7B-Instruct"
    assert payload["available_models"][0]["launch_mode"] == "managed"
    assert payload["api_key_configured"] is True
    assert payload["configured_model"] == "http://localhost:8000/v1"
    assert "vllm-new-secret" not in response.text
    assert "vllm-old-secret" not in response.text

def test_models_configured_patch_updates_managed_vllm_entry_from_model_field() -> None:
    """managed vLLM PATCH 應允許以 model 欄位更新 managed target 並保留 managed 模式。"""
    config = MochiConfig.model_validate(
        {
            "model": "http://localhost:8000/v1",
            "openai_compat": {
                "provider": "vllm",
                "base_url": "http://localhost:8000/v1",
                "model": "Qwen/Qwen2.5-7B-Instruct",
            },
            "model_setup": {
                "configured_models": [
                    {
                        "id": "vllm-managed-qwen",
                        "provider": "vllm",
                        "model": "Qwen/Qwen2.5-7B-Instruct",
                        "model_spec": "Qwen/Qwen2.5-7B-Instruct",
                        "base_url": "http://localhost:8000/v1",
                        "label": "Qwen/Qwen2.5-7B-Instruct (vllm managed)",
                        "backend_type": "openai_compat",
                        "launch_mode": "managed",
                    }
                ]
            },
        }
    )
    app, _engine = _build_app()
    app.state.config_factory = lambda: config

    with TestClient(app) as client:
        response = client.patch(
            "/v1/models/configured/vllm-managed-qwen",
            json={
                "model": "Qwen/Qwen3-8B",
                "persist": False,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["updated_model"]["model"] == "Qwen/Qwen3-8B"
    assert payload["updated_model"]["model_spec"] == "Qwen/Qwen3-8B"
    assert payload["updated_model"]["launch_mode"] == "managed"
    assert payload["available_models"][0]["model"] == "Qwen/Qwen3-8B"
    assert payload["available_models"][0]["model_spec"] == "Qwen/Qwen3-8B"
    assert payload["available_models"][0]["launch_mode"] == "managed"

def test_models_local_discovery_returns_gguf_and_hf_candidates(tmp_path: Path) -> None:
    """`GET /v1/models/local` 應回傳可辨識的本地模型候選。"""
    gguf = tmp_path / "demo.gguf"
    gguf.write_text("gguf", encoding="utf-8")
    hf_dir = tmp_path / "Qwen2.5-7B-Instruct"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text("{}", encoding="utf-8")
    (hf_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (hf_dir / "model.safetensors").write_text("x", encoding="utf-8")

    app, _engine = _build_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "local_models": {
                "roots": [str(tmp_path)],
                "scan_max_depth": 3,
                "scan_max_entries": 100,
            },
        }
    )

    with TestClient(app) as client:
        response = client.get("/v1/models/local", params={"root": str(tmp_path)})

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "local_models"
    assert payload["root"] == str(tmp_path.resolve())
    specs = {item["model_spec"] for item in payload["models"]}
    assert str(gguf.resolve()) in specs
    assert str(hf_dir.resolve()) in specs

def test_models_switch_route_accepts_saved_local_model_entry(tmp_path: Path) -> None:
    """`/v1/models/switch` 應可切換已保存 local entry。"""
    gguf = tmp_path / "demo.gguf"
    gguf.write_text("gguf", encoding="utf-8")
    config = MochiConfig.model_validate(
        {
            "model": "ollama:configured",
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
    app, engine = _build_app()
    app.state.config_factory = lambda: config

    with TestClient(app) as client:
        response = client.post("/v1/models/switch", json={"model": str(gguf.resolve())})

    assert response.status_code == 200
    assert engine.switch_calls == [str(gguf.resolve())]
    assert response.json()["active_model"]["name"] == str(gguf.resolve())

def test_models_local_capabilities_returns_gguf_and_hardware_summary(tmp_path: Path) -> None:
    """`GET /v1/models/local/capabilities` 應回傳 GGUF 量化能力摘要。"""
    hf_dir = tmp_path / "Qwen2.5-7B-Instruct"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text('{"model_type":"qwen2"}', encoding="utf-8")
    (hf_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (hf_dir / "model.safetensors").write_text("x", encoding="utf-8")

    app, _engine = _build_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "local_models": {
                "roots": [str(tmp_path)],
            },
        }
    )

    with TestClient(app) as client:
        response = client.get(
            "/v1/models/local/capabilities",
            params={"model_spec": str(hf_dir)},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "local_model_quantization_capabilities"
    assert payload["model_spec"] == str(hf_dir.resolve())
    assert payload["model_dir"] == str(hf_dir.resolve())
    assert payload["model_family"] == "qwen2"
    by_format = {item["format_id"]: item for item in payload["formats"]}
    assert set(by_format.keys()) == {"gguf"}
    assert by_format["gguf"]["supported"] is True
    assert by_format["gguf"]["priority"] == "primary"
    assert by_format["gguf"]["suggested_default_quantization"] in {
        "Q3_K_M",
        "Q4_K_M",
        "Q5_K_M",
        "Q6_K",
        "Q8_0",
    }
    option_ids = {item["id"] for item in by_format["gguf"]["quantization_options"]}
    assert {"Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16", "BF16"} <= option_ids
    assert payload["hardware"] is not None

def test_models_local_capabilities_rejects_non_hf_or_missing_paths(tmp_path: Path) -> None:
    """非 HF 目錄或不存在路徑應回傳可讀 4xx 錯誤。"""
    gguf = tmp_path / "demo.gguf"
    gguf.write_text("gguf", encoding="utf-8")
    broken_dir = tmp_path / "broken-hf"
    broken_dir.mkdir()
    (broken_dir / "config.json").write_text("{}", encoding="utf-8")

    app, _engine = _build_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "local_models": {
                "roots": [str(tmp_path)],
            },
        }
    )

    with TestClient(app) as client:
        as_file = client.get("/v1/models/local/capabilities", params={"model_spec": str(gguf)})
        missing = client.get(
            "/v1/models/local/capabilities",
            params={"model_spec": str(tmp_path / "missing-model")},
        )
        broken = client.get(
            "/v1/models/local/capabilities",
            params={"model_spec": str(broken_dir)},
        )

    assert as_file.status_code == 400
    assert "HuggingFace model directories only" in as_file.json()["detail"]
    assert missing.status_code == 404
    assert "does not exist" in missing.json()["detail"]
    assert broken.status_code == 400
    assert "not a valid HuggingFace safetensors directory" in broken.json()["detail"]

def test_models_local_convert_persists_converted_gguf_to_available_models(tmp_path: Path) -> None:
    """`POST /v1/models/local/convert` 成功且 persist=true 時應寫入 configured_models。"""
    hf_dir = tmp_path / "Qwen2.5-7B-Instruct"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text('{"model_type":"qwen2"}', encoding="utf-8")
    (hf_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (hf_dir / "model.safetensors").write_text("x", encoding="utf-8")
    output_path = tmp_path / "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    config_path = tmp_path / "mochi.yaml"
    app, _engine = _build_app(config_path=config_path)
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "local_models": {
                "roots": [str(tmp_path)],
            },
        }
    )

    class _FakeConverter:
        async def convert(self, request: Any) -> LocalModelConvertExecutionResult:
            assert request.target_format == "gguf"
            assert request.quantization == "Q4_K_M"
            return LocalModelConvertExecutionResult(
                target_format="gguf",
                quantization="Q4_K_M",
                source_model_dir=str(hf_dir.resolve()),
                output_model_path=str(output_path.resolve()),
                converted=True,
                message="fake converter done",
            )

    app.state.local_model_converter = _FakeConverter()

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/local/convert",
            json={
                "source_model_dir": str(hf_dir),
                "target_format": "gguf",
                "quantization": "Q4_K_M",
                "persist": True,
            },
        )
        models_response = client.get("/v1/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "local_model_convert"
    assert payload["target_format"] == "gguf"
    assert payload["quantization"] == "Q4_K_M"
    assert payload["provider"] == "local"
    assert payload["source_model_dir"] == str(hf_dir.resolve())
    assert payload["output_model_path"] == str(output_path.resolve())
    assert payload["converted"] is True
    assert payload["persisted"] is True
    assert payload["active_model"]["model_spec"] == str(output_path.resolve())
    assert payload["config_path"] == str(config_path)
    assert payload["warnings"] == []
    assert payload["saved_as_model"]["provider"] == "local"
    assert payload["saved_as_model"]["backend_type"] == "gguf"
    assert payload["saved_as_model"]["model_spec"] == str(output_path.resolve())
    assert isinstance(payload["available_models"], list)
    assert payload["available_models"][0]["model_spec"] == str(output_path.resolve())
    assert payload["available_models"][0]["backend_type"] == "gguf"
    assert models_response.status_code == 200
    assert models_response.json()["available_models"][0]["model_spec"] == str(output_path.resolve())

    saved_config = load_config(config_path)
    assert saved_config.model_setup.configured_models[0].model_spec == str(output_path.resolve())
    assert saved_config.model_setup.configured_models[0].backend_type == "gguf"

def test_models_local_convert_rejects_invalid_quantization(tmp_path: Path) -> None:
    """不支援的 GGUF 量化值應回傳 400。"""
    hf_dir = tmp_path / "Qwen2.5-7B-Instruct"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text("{}", encoding="utf-8")
    (hf_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (hf_dir / "model.safetensors").write_text("x", encoding="utf-8")
    app, _engine = _build_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "local_models": {"roots": [str(tmp_path)]},
        }
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/local/convert",
            json={
                "source_model_dir": str(hf_dir),
                "target_format": "gguf",
                "quantization": "Q9_FAKE",
                "persist": False,
            },
        )

    assert response.status_code == 400
    assert "Unsupported GGUF quantization" in response.json()["detail"]

def test_models_local_convert_rejects_non_hf_source_dir(tmp_path: Path) -> None:
    """非 HF safetensors 目錄應回傳 400。"""
    broken_dir = tmp_path / "broken-hf"
    broken_dir.mkdir()
    (broken_dir / "config.json").write_text("{}", encoding="utf-8")
    app, _engine = _build_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "local_models": {"roots": [str(tmp_path)]},
        }
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/local/convert",
            json={
                "source_model_dir": str(broken_dir),
                "target_format": "gguf",
                "quantization": "Q4_K_M",
                "persist": False,
            },
        )

    assert response.status_code == 400
    assert "not a valid HuggingFace safetensors directory" in response.json()["detail"]

def test_models_local_convert_returns_503_when_converter_runtime_unavailable(tmp_path: Path) -> None:
    """預設 placeholder converter 在缺 runtime 時應回傳 503。"""
    hf_dir = tmp_path / "Qwen2.5-7B-Instruct"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text("{}", encoding="utf-8")
    (hf_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (hf_dir / "model.safetensors").write_text("x", encoding="utf-8")
    app, _engine = _build_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "local_models": {"roots": [str(tmp_path)]},
        }
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/local/convert",
            json={
                "source_model_dir": str(hf_dir),
                "target_format": "gguf",
                "quantization": "Q4_K_M",
                "persist": False,
            },
        )

    assert response.status_code == 503
    assert "runtime is unavailable" in response.json()["detail"]

def test_models_local_convert_runtime_unavailable_error_preserves_actionable_tooling_hint(
    tmp_path: Path,
) -> None:
    """runtime unavailable 錯誤應保留可操作訊息（例如缺少 llama.cpp 工具）。"""
    hf_dir = tmp_path / "Qwen2.5-7B-Instruct"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text("{}", encoding="utf-8")
    (hf_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (hf_dir / "model.safetensors").write_text("x", encoding="utf-8")
    app, _engine = _build_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "local_models": {"roots": [str(tmp_path)]},
        }
    )

    class _MissingToolConverter:
        async def convert(self, request: Any) -> LocalModelConvertExecutionResult:
            raise RuntimeError("unexpected")

    class _ActionableUnavailableConverter:
        async def convert(self, request: Any) -> LocalModelConvertExecutionResult:
            from mochi.backends.local_models import LocalModelConversionRuntimeUnavailableError

            raise LocalModelConversionRuntimeUnavailableError(
                "GGUF llama.cpp tools/runtime is unavailable: missing `llama-quantize` in PATH."
            )

    app.state.local_model_converter = _ActionableUnavailableConverter()

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/local/convert",
            json={
                "source_model_dir": str(hf_dir),
                "target_format": "gguf",
                "quantization": "Q4_K_M",
                "persist": False,
            },
        )

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "runtime is unavailable" in detail.lower()
    assert "llama-quantize" in detail

def test_models_local_convert_success_without_persist_does_not_mutate_available_models(
    tmp_path: Path,
) -> None:
    """persist=false 成功轉換時，不應寫入 configured_models/available_models。"""
    hf_dir = tmp_path / "Qwen2.5-7B-Instruct"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text("{}", encoding="utf-8")
    (hf_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (hf_dir / "model.safetensors").write_text("x", encoding="utf-8")
    output_path = tmp_path / "Qwen2.5-7B-Instruct-F16.gguf"

    app, _engine = _build_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "local_models": {"roots": [str(tmp_path)]},
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
                    }
                ]
            },
        }
    )

    class _FakeConverter:
        async def convert(self, request: Any) -> LocalModelConvertExecutionResult:
            assert request.target_format == "gguf"
            assert request.quantization == "F16"
            return LocalModelConvertExecutionResult(
                target_format="gguf",
                quantization="F16",
                source_model_dir=str(hf_dir.resolve()),
                output_model_path=str(output_path.resolve()),
                converted=True,
                message="fake converter done",
            )

    app.state.local_model_converter = _FakeConverter()

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/local/convert",
            json={
                "source_model_dir": str(hf_dir),
                "target_format": "gguf",
                "quantization": "F16",
                "persist": False,
            },
        )
        models_response = client.get("/v1/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["converted"] is True
    assert payload["persisted"] is False
    assert payload["config_path"] is None
    assert payload["saved_as_model"] is None
    assert payload["available_models"] is None
    assert payload["active_model"] is None
    assert payload["output_model_path"] == str(output_path.resolve())

    assert models_response.status_code == 200
    listed = models_response.json()["available_models"]
    assert any(item["model_spec"] == "ollama:qwen2.5" for item in listed)
    assert all(item["model_spec"] != str(output_path.resolve()) for item in listed)

def test_models_local_convert_rejects_duplicate_in_progress_conversion(tmp_path: Path) -> None:
    """同一 source model 重複轉換時應回傳 409，避免共享中間檔競爭。"""
    hf_dir = tmp_path / "Qwen2.5-7B-Instruct"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text("{}", encoding="utf-8")
    (hf_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (hf_dir / "model.safetensors").write_text("x", encoding="utf-8")

    app, _engine = _build_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "local_models": {"roots": [str(tmp_path)]},
        }
    )
    app.state.local_model_conversion_in_progress = {str(hf_dir.resolve())}

    class _UnexpectedConverter:
        async def convert(self, request: Any) -> LocalModelConvertExecutionResult:
            raise AssertionError("converter should not run when the source model is already converting")

    app.state.local_model_converter = _UnexpectedConverter()

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/local/convert",
            json={
                "source_model_dir": str(hf_dir),
                "target_format": "gguf",
                "quantization": "Q4_K_M",
                "persist": False,
            },
        )

    assert response.status_code == 409
    assert "already in progress" in response.json()["detail"]

def test_models_local_runtime_status_reports_missing_runtime_actions(tmp_path: Path) -> None:
    """`GET /v1/models/local/runtime` 在未發現 runtime 時應回傳可操作狀態。"""
    app, _engine = _build_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "workspace_dir": str(tmp_path / "workspace"),
            "local_models": {"roots": [str(tmp_path)]},
        }
    )

    with TestClient(app) as client:
        response = client.get("/v1/models/local/runtime")

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "local_model_runtime_status"
    assert payload["runtime"] == "llama.cpp"
    assert payload["readiness"] in {"missing", "degraded"}
    assert "register_existing_path" in payload["actions"]
    assert "prepare_managed_runtime" in payload["actions"]
    assert isinstance(payload["missing_components"], list)
    assert payload["install_dir"] == str((tmp_path / "workspace" / "runtimes" / "llama.cpp" / "b9058").resolve())

def test_models_local_runtime_status_includes_hardware_recommendation(tmp_path: Path, monkeypatch) -> None:
    """`GET /v1/models/local/runtime` should include hardware-based backend recommendation."""
    from mochi.backends.local_models import HardwareSummary

    monkeypatch.setattr(
        "mochi.api.routes.models._detect_hardware_summary",
        lambda: HardwareSummary(
            provider="torch",
            cuda_available=False,
            gpu_count=1,
            gpu_vendor="amd",
            primary_gpu_name="AMD Radeon RX 7900 XTX",
            total_vram_gb=24.0,
            recommended_runtime_backend="hip",
            recommended_runtime_label="HIP",
            warnings=[],
        ),
    )

    app, _engine = _build_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "workspace_dir": str(tmp_path / "workspace"),
            "local_models": {"roots": [str(tmp_path)]},
        }
    )

    with TestClient(app) as client:
        response = client.get("/v1/models/local/runtime")

    assert response.status_code == 200
    payload = response.json()
    assert payload["hardware"] is not None
    assert payload["hardware"]["gpu_vendor"] == "amd"
    assert payload["hardware"]["recommended_runtime_backend"] == "hip"
    assert payload["hardware"]["recommended_runtime_label"] == "HIP"

def test_models_route_aligns_active_gguf_runtime_root_with_runtime_status(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    runtime_root = workspace_dir / "runtimes" / "llama.cpp" / "b9058"
    build_bin = runtime_root / "build" / "bin"
    build_bin.mkdir(parents=True, exist_ok=True)
    (runtime_root / "convert_hf_to_gguf.py").write_text("#!/usr/bin/env python3", encoding="utf-8")
    (build_bin / "llama-quantize").write_text("bin", encoding="utf-8")
    (build_bin / "llama-server").write_text("bin", encoding="utf-8")
    model_path = tmp_path / "demo.gguf"
    model_path.write_text("gguf", encoding="utf-8")

    config = MochiConfig.model_validate(
        {
            "model": str(model_path.resolve()),
            "workspace_dir": str(workspace_dir),
            "sessions_dir": str(tmp_path / "sessions"),
            "skills_dir": str(tmp_path / "skills"),
            "plugins_dir": str(tmp_path / "plugins"),
            "memory": {"db_path": str(tmp_path / "memory.db")},
            "local_models": {
                "roots": [str(tmp_path)],
                "llama_cpp": {
                    "source": "managed",
                    "python_executable": "/usr/bin/python3",
                    "version": "b9058",
                },
            },
        }
    )

    class _RouterBackedEngine:
        def __init__(self, runtime_config: MochiConfig) -> None:
            self._config = runtime_config
            self._router = BackendRouter(
                ollama_base_url=runtime_config.ollama.base_url,
                openai_default_model=runtime_config.openai_compat.model,
                openai_api_key="",
                gguf_config=runtime_config.gguf,
                huggingface_config=runtime_config.huggingface,
                llama_cpp_runtime=runtime_config.local_models.llama_cpp,
                workspace_dir=runtime_config.workspace_dir,
            )
            self._loaded = False

        async def get_model_info(self) -> ModelInfo:
            if not self._loaded:
                await self._router.load(self._config.model)
                self._loaded = True
            return self._router.active.get_model_info()

    app, _engine = _build_app()
    app.state.config_factory = lambda: config
    app.state.engine_factory = lambda: _RouterBackedEngine(config)

    with TestClient(app) as client:
        runtime_response = client.get("/v1/models/local/runtime")
        models_response = client.get("/v1/models")

    assert runtime_response.status_code == 200
    assert models_response.status_code == 200

    runtime_payload = runtime_response.json()
    models_payload = models_response.json()

    assert runtime_payload["readiness"] == "ready"
    assert runtime_payload["root_dir"] == str(runtime_root.resolve())
    assert models_payload["active_model"]["backend_type"] == "gguf"
    assert models_payload["active_model"]["metadata"]["runtime_root"] == str(runtime_root.resolve())

def test_models_active_local_runtime_status_reports_loaded_active_model(tmp_path: Path) -> None:
    """`GET /v1/models/local/active-runtime` 應回報目前 active local model 載入狀態。"""
    app, engine = _build_app()
    model_path = tmp_path / "demo.gguf"
    model_path.write_text("gguf", encoding="utf-8")
    engine.model_info = ModelInfo(
        name=str(model_path.resolve()),
        backend_type="gguf",
        context_length=4096,
        supports_tool_calling=False,
        metadata={
            "loaded": True,
            "idle_unloaded": False,
        },
    )

    with TestClient(app) as client:
        response = client.get("/v1/models/local/active-runtime")

    assert response.status_code == 200
    assert response.json() == {
        "type": "local_active_model_runtime_status",
        "has_active_local_model": True,
        "model_spec": str(model_path.resolve()),
        "backend_type": "gguf",
        "loaded": True,
        "idle_unloaded": False,
        "can_unload": True,
    }

def test_models_active_local_runtime_unload_unloads_current_local_model(tmp_path: Path) -> None:
    """`POST /v1/models/local/active-runtime/unload` 應釋放目前 active local model。"""
    app, engine = _build_app()
    model_path = tmp_path / "demo.gguf"
    model_path.write_text("gguf", encoding="utf-8")
    engine.model_info = ModelInfo(
        name=str(model_path.resolve()),
        backend_type="gguf",
        context_length=4096,
        supports_tool_calling=False,
        metadata={
            "loaded": True,
            "idle_unloaded": False,
        },
    )

    with TestClient(app) as client:
        response = client.post("/v1/models/local/active-runtime/unload")

    assert response.status_code == 200
    assert engine.unload_active_local_model_calls == 1
    assert response.json() == {
        "type": "local_active_model_runtime_unload",
        "unloaded": True,
        "active_runtime": {
            "type": "local_active_model_runtime_status",
            "has_active_local_model": True,
            "model_spec": str(model_path.resolve()),
            "backend_type": "gguf",
            "loaded": False,
            "idle_unloaded": False,
            "can_unload": True,
        },
    }

def test_models_local_runtime_install_prepare_managed_persists_runtime_metadata(tmp_path: Path) -> None:
    """`POST /v1/models/local/runtime/install` prepare_managed 應執行安裝並保存 metadata。"""
    config_path = tmp_path / "mochi.yaml"
    app, _engine = _build_app(config_path=config_path)
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "workspace_dir": str(tmp_path / "workspace"),
            "local_models": {"roots": [str(tmp_path)]},
        }
    )
    runtime_dir = tmp_path / "workspace" / "runtimes" / "llama.cpp" / "b9058"

    from mochi.api.routes import models as models_route

    async def _fake_install_managed_llama_cpp_runtime(**_: object) -> object:
        runtime_dir.mkdir(parents=True, exist_ok=True)
        (runtime_dir / "convert_hf_to_gguf.py").write_text("#!/usr/bin/env python3", encoding="utf-8")
        build_bin = runtime_dir / "build" / "bin"
        build_bin.mkdir(parents=True, exist_ok=True)
        (build_bin / "llama-quantize").write_text("bin", encoding="utf-8")

        class _Result:
            state = "installed"
            source = "managed"
            action = "install"
            version = "b9058"
            root_dir = str(runtime_dir.resolve())
            python_executable = "/usr/bin/python3"
            warnings: list[str] = []
            message = "Installed managed llama.cpp runtime b9058."

        return _Result()

    original_install = models_route.install_managed_llama_cpp_runtime
    models_route.install_managed_llama_cpp_runtime = _fake_install_managed_llama_cpp_runtime

    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/models/local/runtime/install",
                json={"action": "prepare_managed", "persist": True},
            )
    finally:
        models_route.install_managed_llama_cpp_runtime = original_install

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "local_model_runtime_install"
    assert payload["runtime"] == "llama.cpp"
    assert payload["action"] == "prepare_managed"
    assert payload["source"] == "managed"
    assert payload["persisted"] is True
    assert payload["config_path"] == str(config_path)
    assert payload["message"] == "Installed managed llama.cpp runtime b9058."
    assert payload["version"] == "b9058"
    assert payload["runtime_status"]["source"] == "managed"
    assert payload["runtime_status"]["readiness"] == "ready"
    assert payload["runtime_status"]["root_dir"] == str(runtime_dir.resolve())

    saved_config = load_config(config_path)
    assert saved_config.local_models.llama_cpp.source == "managed"
    assert saved_config.local_models.llama_cpp.root_dir == runtime_dir.resolve()
    assert saved_config.local_models.llama_cpp.version == "b9058"
    assert saved_config.local_models.llama_cpp.python_executable == "/usr/bin/python3"

def test_models_local_runtime_install_prepare_managed_surfaces_installer_failure(tmp_path: Path) -> None:
    """managed installer 失敗時，API 應映射成穩定 HTTP error。"""
    app, _engine = _build_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "workspace_dir": str(tmp_path / "workspace"),
            "local_models": {"roots": [str(tmp_path)]},
        }
    )

    from mochi.api.routes import models as models_route
    from mochi.backends.local_models import ManagedLlamaCppInstallNetworkError

    async def _fake_install_failure(**_: object) -> object:
        raise ManagedLlamaCppInstallNetworkError("download failed")

    original_install = models_route.install_managed_llama_cpp_runtime
    models_route.install_managed_llama_cpp_runtime = _fake_install_failure

    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/models/local/runtime/install",
                json={"action": "prepare_managed", "persist": False},
            )
    finally:
        models_route.install_managed_llama_cpp_runtime = original_install

    assert response.status_code == 503
    assert response.json()["detail"] == "download failed"

def test_models_local_runtime_install_register_existing_path_persists_existing_runtime(
    tmp_path: Path,
) -> None:
    """`POST /v1/models/local/runtime/install` register_existing_path 應保存既有路徑。"""
    runtime_dir = tmp_path / "llama.cpp"
    runtime_dir.mkdir()
    (runtime_dir / "convert_hf_to_gguf.py").write_text("#!/usr/bin/env python3", encoding="utf-8")
    build_bin = runtime_dir / "build" / "bin"
    build_bin.mkdir(parents=True)
    (build_bin / "llama-quantize").write_text("bin", encoding="utf-8")

    app, _engine = _build_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "workspace_dir": str(tmp_path / "workspace"),
            "local_models": {"roots": [str(tmp_path)]},
        }
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/local/runtime/install",
            json={
                "action": "register_existing_path",
                "existing_path": str(runtime_dir),
                "persist": False,
            },
        )
        status = client.get("/v1/models/local/runtime")

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "existing_path"
    assert payload["root_dir"] == str(runtime_dir.resolve())
    assert payload["runtime_status"]["readiness"] == "ready"
    assert payload["runtime_status"]["convert_script"] == str((runtime_dir / "convert_hf_to_gguf.py").resolve())
    assert payload["runtime_status"]["quantize_binary"] == str((build_bin / "llama-quantize").resolve())
    assert status.status_code == 200
    assert status.json()["source"] == "existing_path"
    assert status.json()["readiness"] == "ready"

def test_models_local_runtime_install_applies_updated_config_to_existing_engine(tmp_path: Path) -> None:
    """Runtime install/register should refresh the existing engine config, not only app.state.config."""
    runtime_dir = tmp_path / "llama.cpp"
    runtime_dir.mkdir()
    (runtime_dir / "convert_hf_to_gguf.py").write_text("#!/usr/bin/env python3", encoding="utf-8")
    build_bin = runtime_dir / "build" / "bin"
    build_bin.mkdir(parents=True)
    (build_bin / "llama-quantize").write_text("bin", encoding="utf-8")

    app, _engine = _build_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "workspace_dir": str(tmp_path / "workspace"),
            "local_models": {"roots": [str(tmp_path)]},
        }
    )

    class _FakeEngine:
        def __init__(self) -> None:
            self.received_config: MochiConfig | None = None

        async def apply_config(self, config: MochiConfig, *, reload_voice: bool = False) -> None:
            self.received_config = config

    fake_engine = _FakeEngine()
    app.state.engine = fake_engine

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/local/runtime/install",
            json={
                "action": "register_existing_path",
                "existing_path": str(runtime_dir),
                "persist": False,
            },
        )

    assert response.status_code == 200
    assert fake_engine.received_config is not None
    assert fake_engine.received_config.local_models.llama_cpp.root_dir == runtime_dir.resolve()
    assert fake_engine.received_config.local_models.llama_cpp.source == "existing_path"

def test_models_local_runtime_install_prepare_managed_maps_ready_status_when_tools_ready(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """`prepare_managed` 成功路徑：若 runtime discovery ready，API 應映射 readiness=ready。"""
    app, _engine = _build_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "workspace_dir": str(tmp_path / "workspace"),
            "local_models": {"roots": [str(tmp_path)]},
        }
    )

    runtime_dir = tmp_path / "workspace" / "runtimes" / "llama.cpp" / "b9058"

    async def _fake_install_managed_llama_cpp_runtime(**_: object) -> object:
        runtime_dir.mkdir(parents=True, exist_ok=True)
        (runtime_dir / "convert_hf_to_gguf.py").write_text("#!/usr/bin/env python3", encoding="utf-8")
        build_bin = runtime_dir / "build" / "bin"
        build_bin.mkdir(parents=True, exist_ok=True)
        (build_bin / "llama-quantize").write_text("bin", encoding="utf-8")

        class _Result:
            state = "installed"
            source = "managed"
            action = "install"
            version = "b9058"
            root_dir = str(runtime_dir.resolve())
            python_executable = "/usr/bin/python3"
            warnings: list[str] = []
            message = "Installed managed llama.cpp runtime b9058."

        return _Result()

    monkeypatch.setattr(
        "mochi.api.routes.models.install_managed_llama_cpp_runtime",
        _fake_install_managed_llama_cpp_runtime,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/local/runtime/install",
            json={"action": "prepare_managed", "persist": False},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["action"] == "prepare_managed"
    assert payload["source"] == "managed"
    assert payload["state"] == "ready"
    assert payload["runtime_status"]["readiness"] == "ready"
    assert payload["runtime_status"]["installed"] is True
    assert payload["runtime_status"]["actions"] == ["ready_for_conversion"]
    assert payload["runtime_status"]["version"] == "b9058"

def test_models_local_runtime_install_prepare_managed_failure_mapping_preserves_http_error(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """`prepare_managed` 失敗路徑：installer failure 應映射為穩定 HTTP error。"""
    app, _engine = _build_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "workspace_dir": str(tmp_path / "workspace"),
            "local_models": {"roots": [str(tmp_path)]},
        }
    )

    from mochi.backends.local_models import ManagedLlamaCppInstallNetworkError

    async def _fake_install_failure(**_: object) -> object:
        raise ManagedLlamaCppInstallNetworkError("Managed installer backend unavailable.")

    monkeypatch.setattr(
        "mochi.api.routes.models.install_managed_llama_cpp_runtime",
        _fake_install_failure,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/local/runtime/install",
            json={"action": "prepare_managed", "persist": False},
        )

    assert response.status_code == 503
    assert "Managed installer backend unavailable." in response.json()["detail"]
