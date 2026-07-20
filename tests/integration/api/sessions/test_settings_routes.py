from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from mochi.config.schema import MochiConfig, VoiceConfig

from ._support import _create_test_app


def test_settings_hides_secrets_and_returns_bounded_summary(tmp_path: Path) -> None:
    """`/v1/settings` 不得回傳 token 或 API key。"""
    config = MochiConfig.model_validate(
        {
            "model": "ollama:qwen2.5",
            "memory": {"db_path": str(tmp_path / "memory.db")},
            "voice": {
                "enabled": True,
                "stt_backend": "external-api",
                "stt_openai_base_url": "http://stt.example.com/v1",
                "stt_openai_api_key": SecretStr("sk-voice-secret"),
                "tts_backend": "external-api",
                "tts_model": "gpt-4o-mini-tts",
                "tts_voice": "alloy",
                "tts_openai_base_url": "http://tts.example.com/v1",
                "tts_openai_api_key": SecretStr("sk-tts-secret"),
            },
            "channels": {
                "discord": {
                    "enabled": True,
                    "bot_token": SecretStr("discord-secret-token"),
                    "allowed_channel_ids": [123],
                    "allowed_user_ids": [789],
                    "rate_limit_per_user": 5,
                },
                "telegram": {
                    "enabled": True,
                    "bot_token": SecretStr("telegram-secret-token"),
                    "allowed_chat_ids": [456],
                    "rate_limit_per_user": 7,
                },
            },
            "web": {"host": "127.0.0.1", "port": 9000},
            "locale_defaults": {
                "region_profile": "global",
                "ui_locale": "auto",
                "ui_locale_fallback": "en-US",
                "response_language": "same_as_user",
                "default_tts_voice": "en-US-AriaNeural",
                "timezone": "auto",
            },
        }
    )
    app = _create_test_app(config=config)

    with TestClient(app) as client:
        response = client.get("/v1/settings")

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "settings"
    assert payload["model"] == "ollama:qwen2.5"
    assert payload["model_setup"] == {
        "mode": "configured_or_setup",
        "default_provider": "ollama",
        "default_model": "llama3.2",
        "default_model_spec": "ollama:llama3.2",
        "setup_required": True,
        "configured_models": [],
        "fallback_chain": [
            "user_config",
            "ollama_tags",
            "openai_compatible_provider",
        ],
    }
    assert payload["locale_defaults"] == {
        "region_profile": "global",
        "ui_locale": "auto",
        "ui_locale_fallback": "en-US",
        "response_language": "same_as_user",
        "default_tts_voice": "en-US-AriaNeural",
        "timezone": "auto",
    }
    assert payload["agent"] == {
        "system_prompt": config.agent.system_prompt,
        "temperature": 0.7,
        "max_tokens": None,
        "reserve_output_tokens": None,
        "top_p": 1.0,
        "min_p": 0.0,
        "top_k": 0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "repeat_penalty": 1.0,
        "reasoning_effort": None,
        "show_token_stats": False,
        "presets": [
            {
                "name": "default",
                "system_prompt": "",
                "temperature": 0.7,
                "max_tokens": None,
                "reserve_output_tokens": None,
                "top_p": 1.0,
                "min_p": 0.0,
                "top_k": 0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "repeat_penalty": 1.0,
                "reasoning_effort": None,
            }
        ],
        "active_preset": "default",
    }
    assert payload["voice"]["enabled"] is True
    assert payload["voice"]["stt_backend"] == "external-api"
    assert payload["voice"]["stt_openai_base_url"] == "http://stt.example.com/v1"
    assert payload["voice"]["stt_openai_api_key_configured"] is True
    assert payload["voice"]["stt_openai_timeout"] == 60.0
    assert payload["voice"]["tts_backend"] == "external-api"
    assert payload["voice"]["tts_model"] == "gpt-4o-mini-tts"
    assert payload["voice"]["tts_voice"] == "alloy"
    assert payload["voice"]["tts_openai_base_url"] == "http://tts.example.com/v1"
    assert payload["voice"]["tts_openai_api_key_configured"] is True
    assert payload["voice"]["tts_openai_timeout"] == 60.0
    assert payload["voice"]["reply_model_mode"] == "inherit_active"
    assert payload["voice"]["reply_model_id"] is None
    assert payload["voice"]["session_mode"] == "append_current"
    assert payload["voice"]["registered_tts_voices"] == []
    assert "large-v3" in payload["voice"]["supported_stt_models_by_backend"]["faster-whisper"]
    assert "medium" in payload["voice"]["supported_stt_models_by_backend"]["openai-whisper"]
    assert "whisper-1" in payload["voice"]["supported_stt_models_by_backend"]["external-api"]
    assert "gpt-4o-mini-tts" in payload["voice"]["supported_tts_models_by_backend"]["external-api"]
    assert payload["voice"]["voice_pack_dir"]
    assert payload["memory"]["db_path"] == str(tmp_path / "memory.db")
    assert payload["channels"]["discord"] == {
        "enabled": True,
        "text_enabled": True,
        "voice_enabled": False,
        "bot_token_configured": True,
        "allowed_guild_ids": [],
        "allowed_channel_ids": [123],
        "allowed_voice_channel_ids": [],
        "allowed_user_ids": [789],
        "admin_user_ids": [],
        "rate_limit_per_user": 5,
        "message_mode": "mentions_only",
        "auto_join_policy": "manual_only",
        "voice_auto_reply": True,
        "voice_stt_enabled": True,
        "voice_tts_enabled": True,
    }
    assert payload["channels"]["telegram"] == {
        "enabled": True,
        "allowed_chat_ids": [456],
        "allowed_user_ids": [],
        "rate_limit_per_user": 7,
    }
    assert payload["web"] == {"host": "127.0.0.1", "port": 9000}
    assert payload["security"] == {
        "autonomy_mode": "trusted_workspace",
        "change_contract_mode": "observe",
        "require_approval_for_file_write": False,
        "require_approval_for_exec": True,
        "command_rules": [],
        "exec_allowed_env_vars": [],
        "exec_default_shell": "auto",
        "agent_run_default_max_wall_clock_sec": None,
        "agent_run_default_heartbeat_timeout_sec": None,
        "agent_run_default_checkpoint_interval_steps": 1,
        "agent_run_default_max_subagent_failures_per_role": 2,
        "agent_run_default_on_budget_exhausted": "pause",
        "agent_run_default_on_subagent_disconnect": "retry_then_degrade",
        "exec_default_timeout_sec": 30,
        "exec_session_output_limit": 8000,
        "max_file_write_size_mb": 10.0,
        "file_ops_scope": "workspace",
        "file_undo_max_size_mb": 2.0,
    }
    assert "discord-secret-token" not in response.text
    assert "telegram-secret-token" not in response.text
    assert "sk-voice-secret" not in response.text
    assert "sk-tts-secret" not in response.text




def test_settings_returns_default_discord_channel_summary_without_secrets() -> None:
    """未設定 Discord token 時，settings 仍應回傳穩定的非敏感預設摘要。"""
    config = MochiConfig.model_validate({})
    app = _create_test_app(config=config)

    with TestClient(app) as client:
        response = client.get("/v1/settings")

    assert response.status_code == 200
    payload = response.json()
    assert payload["channels"]["discord"] == {
        "enabled": False,
        "text_enabled": True,
        "voice_enabled": False,
        "bot_token_configured": False,
        "allowed_guild_ids": [],
        "allowed_channel_ids": [],
        "allowed_voice_channel_ids": [],
        "allowed_user_ids": [],
        "admin_user_ids": [],
        "rate_limit_per_user": 10,
        "message_mode": "mentions_only",
        "auto_join_policy": "manual_only",
        "voice_auto_reply": True,
        "voice_stt_enabled": True,
        "voice_tts_enabled": True,
    }
    assert payload["channels"]["telegram"] == {
        "enabled": False,
        "allowed_chat_ids": [],
        "allowed_user_ids": [],
        "rate_limit_per_user": 10,
    }
    assert "bot_token" not in payload["channels"]["discord"]




def test_settings_route_treats_non_path_bare_model_name_as_ollama_provider() -> None:
    """非路徑型裸模型名不應被誤判成 local，避免前端誤觸量化能力探測。"""
    config = MochiConfig.model_validate(
        {
            "model": "gemma4:e4b",
            "ollama": {"base_url": "http://localhost:11434"},
        }
    )
    app = _create_test_app(config=config)

    with TestClient(app) as client:
        response = client.get("/v1/settings")

    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "gemma4:e4b"
    assert payload["model_config"]["provider"] == "ollama"
    assert payload["model_config"]["ollama_model"] == "gemma4:e4b"
    assert payload["model_config"]["local_model_path"] == ""




def test_settings_route_reports_vllm_provider_for_openai_compat_model() -> None:
    """HTTP model spec + openai_compat.provider=vllm 時，settings 應回傳 vllm。"""
    config = MochiConfig.model_validate(
        {
            "model": "https://vllm.example.com/v1",
            "openai_compat": {
                "provider": "vllm",
                "base_url": "https://vllm.example.com/v1",
                "model": "Qwen/Qwen3-8B",
                "api_key": SecretStr("sk-vllm-secret"),
            },
        }
    )
    app = _create_test_app(config=config)

    with TestClient(app) as client:
        response = client.get("/v1/settings")

    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "https://vllm.example.com/v1"
    assert payload["model_config"]["provider"] == "vllm"
    assert payload["model_config"]["openai_compat_provider"] == "vllm"
    assert payload["model_config"]["vllm_launch_mode"] == "external"
    assert payload["model_config"]["openai_compat_base_url"] == "https://vllm.example.com/v1"
    assert payload["model_config"]["openai_compat_model"] == "Qwen/Qwen3-8B"
    assert payload["model_config"]["openai_compat_api_key_configured"] is True
    assert "sk-vllm-secret" not in response.text


@pytest.mark.parametrize(
    ("provider", "base_url", "model_name"),
    [
        ("sglang", "http://localhost:30000/v1", "Qwen/Qwen2.5-7B-Instruct"),
        ("tensorrt_llm", "http://localhost:8000/v1", "meta/llama-3.1-8b-instruct"),
    ],


)
def test_settings_route_reports_external_openai_compat_provider_presets(
    provider: str,
    base_url: str,
    model_name: str,
) -> None:
    config = MochiConfig.model_validate(
        {
            "model": base_url,
            "openai_compat": {
                "provider": provider,
                "base_url": base_url,
                "model": model_name,
                "api_key": SecretStr("sk-provider-secret"),
            },
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
                        "launch_mode": "external",
                    }
                ]
            },
        }
    )
    app = _create_test_app(config=config)

    with TestClient(app) as client:
        response = client.get("/v1/settings")

    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == base_url
    assert payload["model_config"]["provider"] == provider
    assert payload["model_config"]["openai_compat_provider"] == provider
    assert payload["model_config"]["openai_compat_base_url"] == base_url
    assert payload["model_config"]["openai_compat_model"] == model_name
    assert payload["model_config"]["openai_compat_api_key_configured"] is True
    assert payload["model_setup"]["configured_models"][0]["provider"] == provider
    assert payload["model_setup"]["configured_models"][0]["launch_mode"] == "external"
    assert "sk-provider-secret" not in response.text




def test_settings_route_reports_managed_vllm_launch_mode() -> None:
    config = MochiConfig.model_validate(
        {
            "model": "http://localhost:8000/v1",
            "openai_compat": {
                "provider": "vllm",
                "base_url": "http://localhost:8000/v1",
                "model": "google/gemma-4-26B-A4B-it",
            },
            "model_setup": {
                "configured_models": [
                    {
                        "id": "vllm:managed:gemma4",
                        "provider": "vllm",
                        "model": "google/gemma-4-26B-A4B-it",
                        "model_spec": "google/gemma-4-26B-A4B-it",
                        "base_url": "http://localhost:8000/v1",
                        "launch_mode": "managed",
                    }
                ]
            },
        }
    )
    app = _create_test_app(config=config)

    with TestClient(app) as client:
        response = client.get("/v1/settings")

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_config"]["provider"] == "vllm"
    assert payload["model_config"]["openai_compat_provider"] == "vllm"
    assert payload["model_config"]["vllm_launch_mode"] == "managed"




def test_settings_get_and_patch_support_ollama_gguf_and_vllm_context_length(tmp_path: Path) -> None:
    """`/v1/settings` should expose and patch provider-specific context length settings."""
    config = MochiConfig.model_validate(
        {
            "workspace_dir": str(tmp_path / "workspace"),
            "model": "ollama:qwen2.5",
            "ollama": {
                "num_ctx": 16384,
            },
            "gguf": {
                "n_ctx": 4096,
            },
            "vllm": {
                "max_model_len": 8192,
            },
        }
    )
    app = _create_test_app(config=config)

    with TestClient(app) as client:
        before = client.get("/v1/settings")
        assert before.status_code == 200
        assert before.json()["ollama"] == {"num_ctx": 16384}
        assert before.json()["gguf"] == {"n_ctx": 4096}
        assert before.json()["vllm"] == {"max_model_len": 8192}

        response = client.patch(
            "/v1/settings",
            json={
                "ollama": {
                    "num_ctx": 32768,
                },
                "gguf": {
                    "n_ctx": 16384,
                },
                "vllm": {
                    "max_model_len": 32768,
                },
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["ollama"] == {"num_ctx": 32768}
        assert payload["gguf"] == {"n_ctx": 16384}
        assert payload["vllm"] == {"max_model_len": 32768}

        followup = client.get("/v1/settings")

    assert followup.status_code == 200
    assert followup.json()["ollama"] == {"num_ctx": 32768}
    assert followup.json()["gguf"] == {"n_ctx": 16384}
    assert followup.json()["vllm"] == {"max_model_len": 32768}




def test_settings_patch_updates_voice_memory_learning_and_paths(tmp_path: Path) -> None:
    """`PATCH /v1/settings` 應更新 WebGUI 可編輯設定並建立資料目錄。"""
    config = MochiConfig.model_validate(
        {
            "workspace_dir": str(tmp_path / "workspace"),
            "sessions_dir": str(tmp_path / "sessions"),
            "skills_dir": str(tmp_path / "skills"),
            "plugins_dir": str(tmp_path / "plugins"),
            "memory": {"db_path": str(tmp_path / "memory-old.db")},
        }
    )
    app = _create_test_app(config=config)

    with TestClient(app) as client:
        response = client.patch(
            "/v1/settings",
            json={
                "voice": {
                    "enabled": True,
                    "stt_backend": "openai-whisper",
                    "stt_model": "base",
                    "stt_language": "zh",
                    "stt_model_cache_dir": str(tmp_path / "voice-models"),
                    "tts_backend": "piper",
                    "tts_voice": str(tmp_path / "voices" / "zh.onnx"),
                    "tts_speed": 1.2,
                    "reply_model_mode": "configured_model",
                    "reply_model_id": "voice-openai",
                    "session_mode": "isolated_voice",
                    "voice_pack_dir": str(tmp_path / "voice-packs"),
                },
                "memory": {
                    "db_path": str(tmp_path / "memory-new.db"),
                    "max_short_term_messages": 80,
                    "fts_top_k": 8,
                },
                "learning": {
                    "enabled": False,
                    "auto_extract_skills": False,
                    "auto_sync_filesystem_skills": False,
                    "min_steps_for_extraction": 5,
                    "min_tool_calls_for_extraction": 3,
                    "trajectory_retention_days": 90,
                    "skill_improvement_threshold": 0.8,
                    "max_skills": 900,
                },
                "locale_defaults": {
                    "region_profile": "us",
                    "ui_locale": "en-US",
                    "ui_locale_fallback": "en-US",
                    "response_language": "same_as_user",
                    "default_tts_voice": "en-US-JennyNeural",
                    "timezone": "America/New_York",
                },
                "paths": {
                    "workspace_dir": str(tmp_path / "workspace-new"),
                    "sessions_dir": str(tmp_path / "sessions-new"),
                    "skills_dir": str(tmp_path / "skills-new"),
                    "plugins_dir": str(tmp_path / "plugins-new"),
                },
                "download_missing_models": False,
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["voice"]["enabled"] is True
        assert payload["voice"]["stt_backend"] == "openai-whisper"
        assert payload["voice"]["stt_model_cache_dir"] == str(tmp_path / "voice-models")
        assert payload["voice"]["tts_backend"] == "piper"
        assert payload["voice"]["reply_model_mode"] == "configured_model"
        assert payload["voice"]["reply_model_id"] == "voice-openai"
        assert payload["voice"]["session_mode"] == "isolated_voice"
        assert payload["voice"]["voice_pack_dir"] == str(tmp_path / "voice-packs")
        assert payload["memory"]["db_path"] == str(tmp_path / "memory-new.db")
        assert payload["learning"]["enabled"] is False
        assert payload["learning"]["auto_sync_filesystem_skills"] is False
        assert payload["learning"]["min_tool_calls_for_extraction"] == 3
        assert payload["learning"]["trajectory_retention_days"] == 90
        assert payload["locale_defaults"] == {
            "region_profile": "us",
            "ui_locale": "en-US",
            "ui_locale_fallback": "en-US",
            "response_language": "same_as_user",
            "default_tts_voice": "en-US-JennyNeural",
            "timezone": "America/New_York",
        }
        assert payload["paths"]["workspace_dir"] == str(tmp_path / "workspace-new")
        assert payload["update"]["download"] == {"requested": False, "status": "skipped"}

        followup = client.get("/v1/settings")

    assert followup.status_code == 200
    assert followup.json()["memory"]["db_path"] == str(tmp_path / "memory-new.db")
    assert (tmp_path / "workspace-new").is_dir()
    assert (tmp_path / "sessions-new").is_dir()
    assert (tmp_path / "skills-new").is_dir()
    assert (tmp_path / "plugins-new").is_dir()
    assert (tmp_path / "voice-models").is_dir()
    assert (tmp_path / "voice-packs").is_dir()




def test_settings_patch_updates_voice_reply_modes_and_discord_admins(tmp_path: Path) -> None:
    """`PATCH /v1/settings` should update shared voice routing fields and Discord admins."""
    config = MochiConfig.model_validate(
        {
            "workspace_dir": str(tmp_path / "workspace"),
            "model_setup": {
                "configured_models": [
                    {
                        "id": "voice-openai",
                        "provider": "openai_compat",
                        "model": "gpt-4o-mini",
                        "model_spec": "https://example.invalid/v1",
                        "base_url": "https://example.invalid/v1",
                        "label": "Voice OpenAI",
                        "backend_type": "openai_compat",
                    }
                ]
            },
        }
    )
    app = _create_test_app(config=config)

    with TestClient(app) as client:
        response = client.patch(
            "/v1/settings",
            json={
                "voice": {
                    "reply_model_mode": "configured_model",
                    "reply_model_id": "voice-openai",
                    "session_mode": "isolated_voice",
                },
                "channels": {
                    "discord": {
                        "admin_user_ids": [1001, 1002],
                    }
                },
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["voice"]["reply_model_mode"] == "configured_model"
        assert payload["voice"]["reply_model_id"] == "voice-openai"
        assert payload["voice"]["session_mode"] == "isolated_voice"
        assert payload["channels"]["discord"]["admin_user_ids"] == [1001, 1002]

        followup = client.get("/v1/settings")

    assert followup.status_code == 200
    assert followup.json()["voice"]["reply_model_mode"] == "configured_model"
    assert followup.json()["voice"]["reply_model_id"] == "voice-openai"
    assert followup.json()["voice"]["session_mode"] == "isolated_voice"
    assert followup.json()["channels"]["discord"]["admin_user_ids"] == [1001, 1002]




def test_settings_patch_updates_agent_presets_and_security(tmp_path: Path) -> None:
    """`PATCH /v1/settings` 應更新 inference presets 與 security 設定。"""
    config = MochiConfig.model_validate(
        {
            "workspace_dir": str(tmp_path / "workspace"),
            "sessions_dir": str(tmp_path / "sessions"),
            "skills_dir": str(tmp_path / "skills"),
            "plugins_dir": str(tmp_path / "plugins"),
        }
    )
    app = _create_test_app(config=config)

    with TestClient(app) as client:
        response = client.patch(
            "/v1/settings",
            json={
                "agent": {
                    "system_prompt": "你是 Mochi inference 測試代理。",
                    "temperature": 0.25,
                    "max_tokens": 8192,
                    "reserve_output_tokens": 1536,
                    "top_p": 0.9,
                    "min_p": 0.05,
                    "top_k": 32,
                    "frequency_penalty": 0.2,
                    "presence_penalty": 0.3,
                    "repeat_penalty": 1.1,
                    "reasoning_effort": "medium",
                    "show_token_stats": True,
                    "presets": [
                        {
                            "name": "default",
                            "system_prompt": "default prompt",
                            "temperature": 0.4,
                            "max_tokens": 4096,
                            "reserve_output_tokens": 1024,
                            "top_p": 0.95,
                            "min_p": 0.0,
                            "top_k": 0,
                            "frequency_penalty": 0.0,
                            "presence_penalty": 0.0,
                            "repeat_penalty": 1.0,
                            "reasoning_effort": None,
                        },
                        {
                            "name": "focused",
                            "system_prompt": "focused prompt",
                            "temperature": 0.2,
                            "max_tokens": 2048,
                            "reserve_output_tokens": 768,
                            "top_p": 0.8,
                            "min_p": 0.1,
                            "top_k": 24,
                            "frequency_penalty": 0.1,
                            "presence_penalty": 0.2,
                            "repeat_penalty": 1.05,
                            "reasoning_effort": "high",
                        },
                    ],
                    "active_preset": "focused",
                },
                "security": {
                    "autonomy_mode": "auto_review",
                    "require_approval_for_file_write": False,
                    "require_approval_for_exec": False,
                    "command_rules": [],
                    "exec_allowed_env_vars": [],
                    "exec_default_shell": "auto",
                    "agent_run_default_max_wall_clock_sec": 1800,
                    "agent_run_default_heartbeat_timeout_sec": 120,
                    "agent_run_default_checkpoint_interval_steps": 4,
                    "agent_run_default_max_subagent_failures_per_role": 5,
                    "agent_run_default_on_budget_exhausted": "finalize_partial",
                    "agent_run_default_on_subagent_disconnect": "pause",
                    "exec_default_timeout_sec": 90,
                    "exec_session_output_limit": 16384,
                    "max_file_write_size_mb": 3.5,
                    "file_ops_scope": "any",
                    "file_undo_max_size_mb": 1.5,
                },
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["agent"] == {
            "system_prompt": "你是 Mochi inference 測試代理。",
            "temperature": 0.25,
            "max_tokens": 8192,
            "reserve_output_tokens": 1536,
            "top_p": 0.9,
            "min_p": 0.05,
            "top_k": 32,
            "frequency_penalty": 0.2,
            "presence_penalty": 0.3,
            "repeat_penalty": 1.1,
            "reasoning_effort": "medium",
            "show_token_stats": True,
            "presets": [
                {
                    "name": "default",
                    "system_prompt": "default prompt",
                    "temperature": 0.4,
                    "max_tokens": 4096,
                    "reserve_output_tokens": 1024,
                    "top_p": 0.95,
                    "min_p": 0.0,
                    "top_k": 0,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0,
                    "repeat_penalty": 1.0,
                    "reasoning_effort": None,
                },
                {
                    "name": "focused",
                    "system_prompt": "focused prompt",
                    "temperature": 0.2,
                    "max_tokens": 2048,
                    "reserve_output_tokens": 768,
                    "top_p": 0.8,
                    "min_p": 0.1,
                    "top_k": 24,
                    "frequency_penalty": 0.1,
                    "presence_penalty": 0.2,
                    "repeat_penalty": 1.05,
                    "reasoning_effort": "high",
                },
            ],
            "active_preset": "focused",
        }
        assert payload["security"] == {
            "autonomy_mode": "auto_review",
            "change_contract_mode": "observe",
            "require_approval_for_file_write": False,
            "require_approval_for_exec": False,
            "command_rules": [],
            "exec_allowed_env_vars": [],
            "exec_default_shell": "auto",
            "agent_run_default_max_wall_clock_sec": 1800,
            "agent_run_default_heartbeat_timeout_sec": 120,
            "agent_run_default_checkpoint_interval_steps": 4,
            "agent_run_default_max_subagent_failures_per_role": 5,
            "agent_run_default_on_budget_exhausted": "finalize_partial",
            "agent_run_default_on_subagent_disconnect": "pause",
            "exec_default_timeout_sec": 90,
            "exec_session_output_limit": 16384,
            "max_file_write_size_mb": 3.5,
            "file_ops_scope": "any",
            "file_undo_max_size_mb": 1.5,
        }

        followup = client.get("/v1/settings")

    assert followup.status_code == 200
    followup_payload = followup.json()
    assert followup_payload["agent"]["active_preset"] == "focused"
    assert followup_payload["agent"]["reasoning_effort"] == "medium"
    assert followup_payload["agent"]["presets"][1]["reasoning_effort"] == "high"
    assert followup_payload["agent"]["show_token_stats"] is True
    assert followup_payload["security"]["autonomy_mode"] == "auto_review"
    assert followup_payload["security"]["require_approval_for_exec"] is False
    assert followup_payload["security"]["command_rules"] == []
    assert followup_payload["security"]["exec_allowed_env_vars"] == []
    assert followup_payload["security"]["exec_default_shell"] == "auto"
    assert followup_payload["security"]["agent_run_default_max_wall_clock_sec"] == 1800
    assert followup_payload["security"]["agent_run_default_heartbeat_timeout_sec"] == 120
    assert followup_payload["security"]["agent_run_default_checkpoint_interval_steps"] == 4
    assert followup_payload["security"]["agent_run_default_max_subagent_failures_per_role"] == 5
    assert followup_payload["security"]["agent_run_default_on_budget_exhausted"] == "finalize_partial"
    assert followup_payload["security"]["agent_run_default_on_subagent_disconnect"] == "pause"
    assert followup_payload["security"]["exec_default_timeout_sec"] == 90
    assert followup_payload["security"]["exec_session_output_limit"] == 16384
    assert followup_payload["security"]["file_ops_scope"] == "any"
    assert followup_payload["security"]["file_undo_max_size_mb"] == 1.5




def test_settings_round_trip_exposes_independent_protected_workspace_axes(tmp_path: Path) -> None:
    config = MochiConfig.model_validate(
        {
            "workspace_dir": str(tmp_path / "workspace"),
            "security": {"change_contract_mode": "enforce"},
            "sandbox": {"mode": "preferred"},
        }
    )
    app = _create_test_app(config=config)

    with TestClient(app) as client:
        initial = client.get("/v1/settings")
        updated = client.patch(
            "/v1/settings",
            json={
                "security": {"change_contract_mode": "observe"},
                "sandbox": {"mode": "required"},
                "persist": False,
            },
        )
        followup = client.get("/v1/settings")

    assert initial.status_code == 200
    assert initial.json()["security"]["change_contract_mode"] == "enforce"
    assert initial.json()["sandbox"] == {
        "mode": "preferred",
        "backend": None,
        "backend_available": False,
        "capabilities": {"exec_containment": False},
        "enforcement_active": False,
        "configured_policy_decision": "prefer_sandbox_backend",
        "effective_exec_behavior": "host_execution_available",
        "status": "configured_unavailable",
        "degraded": True,
        "degraded_reason": "sandbox_backend_unavailable_and_pipeline_not_connected",
        "host_execution_allowed": True,
    }
    assert updated.status_code == 200
    assert updated.json()["security"]["change_contract_mode"] == "observe"
    assert updated.json()["sandbox"] == {
        "mode": "required",
        "backend": None,
        "backend_available": False,
        "capabilities": {"exec_containment": False},
        "enforcement_active": False,
        "configured_policy_decision": "reject_backend_unavailable",
        "effective_exec_behavior": "host_execution_available",
        "status": "configured_unavailable",
        "degraded": True,
        "degraded_reason": "sandbox_backend_unavailable_and_pipeline_not_connected",
        "host_execution_allowed": True,
    }
    assert followup.status_code == 200
    assert followup.json()["security"]["change_contract_mode"] == "observe"
    assert followup.json()["sandbox"]["mode"] == "required"


@pytest.mark.parametrize(
    "patch",
    [
        {"security": {"change_contract_mode": "hybrid"}},
        {"sandbox": {"mode": "observe"}},
    ],


)
def test_settings_rejects_invalid_protected_workspace_modes(patch: dict[str, object]) -> None:
    app = _create_test_app(config=MochiConfig())

    with TestClient(app) as client:
        response = client.patch("/v1/settings", json={**patch, "persist": False})

    assert response.status_code == 422




def test_settings_patch_persists_auto_output_token_mode(tmp_path: Path) -> None:
    config = MochiConfig.model_validate(
        {
            "workspace_dir": str(tmp_path / "workspace"),
            "sessions_dir": str(tmp_path / "sessions"),
            "skills_dir": str(tmp_path / "skills"),
            "plugins_dir": str(tmp_path / "plugins"),
            "agent": {
                "max_tokens": 8192,
                "reserve_output_tokens": 2048,
                "presets": [
                    {
                        "name": "default",
                        "system_prompt": "",
                        "temperature": 0.7,
                        "max_tokens": 8192,
                        "reserve_output_tokens": 2048,
                        "top_p": 1.0,
                        "min_p": 0.0,
                        "top_k": 0,
                        "frequency_penalty": 0.0,
                        "presence_penalty": 0.0,
                        "repeat_penalty": 1.0,
                        "reasoning_effort": None,
                    }
                ],
            },
        }
    )
    app = _create_test_app(config=config)

    with TestClient(app) as client:
        response = client.patch(
            "/v1/settings",
            json={
                "agent": {
                    "max_tokens": None,
                    "reserve_output_tokens": None,
                    "presets": [
                        {
                            "name": "default",
                            "system_prompt": "",
                            "temperature": 0.7,
                            "max_tokens": None,
                            "reserve_output_tokens": None,
                            "top_p": 1.0,
                            "min_p": 0.0,
                            "top_k": 0,
                            "frequency_penalty": 0.0,
                            "presence_penalty": 0.0,
                            "repeat_penalty": 1.0,
                            "reasoning_effort": None,
                        }
                    ],
                }
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["agent"]["max_tokens"] is None
        assert payload["agent"]["reserve_output_tokens"] is None
        assert payload["agent"]["presets"][0]["max_tokens"] is None
        assert payload["agent"]["presets"][0]["reserve_output_tokens"] is None

        followup = client.get("/v1/settings")

    assert followup.status_code == 200
    assert followup.json()["agent"]["max_tokens"] is None
    assert followup.json()["agent"]["reserve_output_tokens"] is None




def test_settings_patch_applies_autonomy_mode_defaults(tmp_path: Path) -> None:
    """`PATCH /v1/settings` autonomy mode 單獨切換時，應帶入對應預設策略。"""
    config = MochiConfig.model_validate(
        {
            "workspace_dir": str(tmp_path / "workspace"),
            "sessions_dir": str(tmp_path / "sessions"),
            "skills_dir": str(tmp_path / "skills"),
            "plugins_dir": str(tmp_path / "plugins"),
        }
    )
    app = _create_test_app(config=config)

    with TestClient(app) as client:
        response = client.patch(
            "/v1/settings",
            json={
                "security": {
                    "autonomy_mode": "high_autonomy",
                },
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["security"] == {
        "autonomy_mode": "high_autonomy",
        "change_contract_mode": "observe",
        "require_approval_for_file_write": False,
        "require_approval_for_exec": False,
        "command_rules": [],
        "exec_allowed_env_vars": [],
        "exec_default_shell": "auto",
        "agent_run_default_max_wall_clock_sec": None,
        "agent_run_default_heartbeat_timeout_sec": None,
        "agent_run_default_checkpoint_interval_steps": 1,
        "agent_run_default_max_subagent_failures_per_role": 2,
        "agent_run_default_on_budget_exhausted": "pause",
        "agent_run_default_on_subagent_disconnect": "retry_then_degrade",
        "exec_default_timeout_sec": 30,
        "exec_session_output_limit": 8000,
        "max_file_write_size_mb": 10.0,
        "file_ops_scope": "any",
        "file_undo_max_size_mb": 2.0,
    }




def test_settings_patch_updates_tools_web_search_and_fetch_config(tmp_path: Path) -> None:
    """`PATCH /v1/settings` 應更新 Web search / fetch 工具設定並隱藏 secrets。"""
    config = MochiConfig.model_validate(
        {
            "workspace_dir": str(tmp_path / "workspace"),
            "tools": {
                "web_search_engine": "duckduckgo",
                "web_search_fallback_engines": ["duckduckgo_html"],
                "web_fetch_extractor": "trafilatura",
            },
        }
    )
    app = _create_test_app(config=config)

    with TestClient(app) as client:
        response = client.patch(
            "/v1/settings",
            json={
                "tools": {
                    "web_search_engine": "tavily",
                    "web_search_fallback_engines": ["brave", "duckduckgo_html"],
                    "web_search_tavily_api_key": "tvly-test-key",
                    "web_search_serper_api_key": "serper-test-key",
                    "web_search_brave_api_key": "brave-test-key",
                    "web_search_jina_api_key": "jina-search-key",
                    "web_search_exa_api_key": "exa-test-key",
                    "web_search_searxng_base_url": "https://search.example.test",
                    "web_search_language": "zh-TW",
                    "web_search_region": "tw",
                    "web_fetch_extractor": "jina_reader",
                    "web_fetch_jina_api_key": "jina-fetch-key",
                }
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["tools"] == {
            "web_search_engine": "tavily",
            "web_search_fallback_engines": ["brave", "duckduckgo_html"],
            "web_search_searxng_base_url": "https://search.example.test",
            "web_search_language": "zh-TW",
            "web_search_region": "tw",
            "web_search_tavily_api_key_configured": True,
            "web_search_serper_api_key_configured": True,
            "web_search_brave_api_key_configured": True,
            "web_search_jina_configured": True,
            "web_search_jina_api_key_configured": True,
            "web_search_exa_api_key_configured": True,
            "web_search_searxng_configured": True,
            "web_search_duckduckgo_html_configured": True,
            "web_fetch_extractor": "jina_reader",
            "web_fetch_jina_api_key_configured": True,
        }

        followup = client.get("/v1/settings")

    assert followup.status_code == 200
    assert followup.json()["tools"] == {
        "web_search_engine": "tavily",
        "web_search_fallback_engines": ["brave", "duckduckgo_html"],
        "web_search_searxng_base_url": "https://search.example.test",
        "web_search_language": "zh-TW",
        "web_search_region": "tw",
        "web_search_tavily_api_key_configured": True,
        "web_search_serper_api_key_configured": True,
        "web_search_brave_api_key_configured": True,
        "web_search_jina_configured": True,
        "web_search_jina_api_key_configured": True,
        "web_search_exa_api_key_configured": True,
        "web_search_searxng_configured": True,
        "web_search_duckduckgo_html_configured": True,
        "web_fetch_extractor": "jina_reader",
        "web_fetch_jina_api_key_configured": True,
    }




def test_settings_patch_can_clear_web_provider_keys(tmp_path: Path) -> None:
    """Explicit null values should clear persisted tool secrets and configured flags."""
    config = MochiConfig.model_validate(
        {
            "workspace_dir": str(tmp_path / "workspace"),
            "tools": {
                "web_search_engine": "tavily",
                "web_search_tavily_api_key": "tvly-test-key",
                "web_search_serper_api_key": "serper-test-key",
                "web_search_exa_api_key": "exa-test-key",
                "web_fetch_jina_api_key": "jina-fetch-key",
            },
        }
    )
    app = _create_test_app(config=config)

    with TestClient(app) as client:
        response = client.patch(
            "/v1/settings",
            json={
                "tools": {
                    "web_search_tavily_api_key": None,
                    "web_search_serper_api_key": None,
                    "web_search_exa_api_key": None,
                    "web_fetch_jina_api_key": None,
                }
            },
        )

        assert response.status_code == 200
        assert response.json()["tools"]["web_search_tavily_api_key_configured"] is False
        assert response.json()["tools"]["web_search_serper_api_key_configured"] is False
        assert response.json()["tools"]["web_search_exa_api_key_configured"] is False
        assert response.json()["tools"]["web_fetch_jina_api_key_configured"] is False




def test_settings_get_and_patch_support_local_model_idle_unload_controls(tmp_path: Path) -> None:
    """`/v1/settings` 應回傳並更新本地模型閒置卸載控制欄位。"""
    config = MochiConfig.model_validate(
        {
            "workspace_dir": str(tmp_path / "workspace"),
            "local_models": {
                "idle_unload_enabled": False,
                "idle_unload_seconds": 300,
            },
        }
    )
    app = _create_test_app(config=config)

    with TestClient(app) as client:
        before = client.get("/v1/settings")
        assert before.status_code == 200
        assert before.json()["local_models"] == {
            "idle_unload_enabled": False,
            "idle_unload_seconds": 300,
        }

        response = client.patch(
            "/v1/settings",
            json={
                "local_models": {
                    "idle_unload_enabled": True,
                    "idle_unload_seconds": 900,
                }
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["local_models"] == {
            "idle_unload_enabled": True,
            "idle_unload_seconds": 900,
        }

        followup = client.get("/v1/settings")

    assert followup.status_code == 200
    assert followup.json()["local_models"] == {
        "idle_unload_enabled": True,
        "idle_unload_seconds": 900,
    }




def test_settings_patch_reports_faster_whisper_runtime_download(tmp_path: Path) -> None:
    """faster-whisper 的模型下載由 runtime 首次使用時寫入 cache dir。"""
    config = MochiConfig.model_validate({})
    app = _create_test_app(config=config)

    with TestClient(app) as client:
        response = client.patch(
            "/v1/settings",
            json={
                "voice": {
                    "stt_backend": "faster-whisper",
                    "stt_model": "small",
                    "stt_model_cache_dir": str(tmp_path / "fw-cache"),
                },
                "download_missing_models": True,
            },
        )

    assert response.status_code == 200
    assert response.json()["update"]["download"] == {
        "requested": True,
        "status": "runtime_download_on_first_use",
        "stt": {
            "requested": True,
            "status": "runtime_download_on_first_use",
            "model_cache_dir": str(tmp_path / "fw-cache"),
        },
        "tts": {
            "requested": False,
            "backend": "kokoro-tts",
            "status": "skipped",
        },
    }




def test_settings_patch_reports_tts_prepare_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """更新 TTS 設定時，若本地模型預熱/下載失敗，API 應回傳 attention_required 而不是 crash。"""
    config = MochiConfig.model_validate({})
    app = _create_test_app(config=config)

    async def _fake_prepare_tts(voice: VoiceConfig) -> dict[str, object]:  # noqa: ARG001
        return {
            "requested": True,
            "backend": "kokoro-tts",
            "status": "prepare_failed",
            "error": "download failed",
        }

    monkeypatch.setattr(
        "mochi.api.routes.settings.ensure_tts_runtime_available",
        _fake_prepare_tts,
    )

    with TestClient(app) as client:
        response = client.patch(
            "/v1/settings",
            json={
                "voice": {
                    "tts_backend": "kokoro-tts",
                    "tts_voice": "af_heart",
                },
                "download_missing_models": True,
            },
        )

    assert response.status_code == 200
    assert response.json()["update"]["download"] == {
        "requested": True,
        "status": "attention_required",
        "message": (
            "TTS backend kokoro-tts could not be prepared automatically: "
            "download failed"
        ),
        "stt": {
            "requested": True,
            "status": "runtime_download_on_first_use",
            "model_cache_dir": str(config.voice.stt_model_cache_dir),
        },
        "tts": {
            "requested": True,
            "backend": "kokoro-tts",
            "status": "prepare_failed",
            "error": "download failed",
        },
    }
