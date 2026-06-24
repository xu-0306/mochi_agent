"""設定 Schema 解析測試。"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mochi.config import defaults
from mochi.config import manager as config_manager
from mochi.config.manager import load_config, save_config
from mochi.config.schema import MochiConfig


def _config_without_paths(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model: ollama:test\n", encoding="utf-8")
    return config_path


def test_default_config_is_valid() -> None:
    """預設設定應可無錯誤實例化。"""
    cfg = MochiConfig()
    assert cfg.model == "ollama:llama3.2"
    assert cfg.log_level == "INFO"
    assert cfg.agent.max_react_iterations == 10
    assert cfg.agent.system_prompt
    assert cfg.agent.temperature == 0.7
    assert cfg.agent.max_tokens == 4096
    assert cfg.agent.top_p == 1.0
    assert cfg.agent.min_p == 0.0
    assert cfg.agent.top_k == 0
    assert cfg.agent.frequency_penalty == 0.0
    assert cfg.agent.presence_penalty == 0.0
    assert cfg.agent.repeat_penalty == 1.0
    assert cfg.agent.show_token_stats is False
    assert len(cfg.agent.presets) == 1
    assert cfg.agent.presets[0].name == "default"
    assert cfg.agent.active_preset == "default"
    assert cfg.model_setup.mode == "configured_or_setup"
    assert cfg.model_setup.default_provider == "ollama"
    assert cfg.model_setup.default_model == "llama3.2"
    assert cfg.model_setup.default_model_spec == "ollama:llama3.2"
    assert cfg.model_setup.setup_required is True
    assert cfg.model_setup.configured_models == []
    assert cfg.model_setup.fallback_chain == [
        "user_config",
        "ollama_tags",
        "openai_compatible_provider",
    ]
    assert cfg.local_models.roots == []
    assert cfg.local_models.scan_max_depth == 3
    assert cfg.local_models.scan_max_entries == 500
    assert cfg.local_models.runtime == "inprocess"
    assert cfg.local_models.idle_unload_enabled is False
    assert cfg.local_models.idle_unload_seconds == 300
    assert cfg.local_models.llama_cpp.source is None
    assert cfg.local_models.llama_cpp.root_dir is None
    assert cfg.local_models.llama_cpp.python_executable is None
    assert cfg.local_models.llama_cpp.version is None
    assert cfg.vllm.enabled is False
    assert cfg.vllm.host == "127.0.0.1"
    assert cfg.vllm.port is None
    assert cfg.vllm.api_key is None
    assert cfg.vllm.tensor_parallel_size == 1
    assert cfg.vllm.dtype == "auto"
    assert cfg.vllm.gpu_memory_utilization == 0.9
    assert cfg.vllm.max_model_len is None
    assert cfg.vllm.trust_remote_code is False
    assert cfg.vllm.quantization is None
    assert cfg.vllm.startup_timeout_seconds == 180
    assert cfg.vllm.cuda_visible_devices is None
    assert cfg.gguf.n_batch == 512
    assert cfg.gguf.n_ubatch == 512
    assert cfg.gguf.n_threads_batch is None
    assert cfg.gguf.flash_attn is False
    assert cfg.gguf.offload_kqv is True
    assert cfg.gguf.use_mmap is True
    assert cfg.gguf.use_mlock is False
    assert cfg.locale_defaults.region_profile == "global"
    assert cfg.locale_defaults.ui_locale == "auto"
    assert cfg.locale_defaults.ui_locale_fallback == "en-US"
    assert cfg.locale_defaults.response_language == "same_as_user"
    assert cfg.locale_defaults.default_tts_voice == "af_heart"
    assert cfg.locale_defaults.timezone == "auto"
    assert cfg.voice.tts_backend == "kokoro-tts"
    assert cfg.voice.tts_voice == cfg.locale_defaults.default_tts_voice
    assert cfg.voice.voice_input_channel_policy == {
        "mode": "mono-only",
        "configured_channels": 1,
        "supported_channels": [1],
        "validation_message": (
            "voice.channels must be 1 because /v1/voice only accepts mono "
            "PCM16 input (base64-encoded s16le)."
        ),
    }
    assert cfg.voice.voice_input_contract == {
        "transport": "websocket",
        "message_type": "audio_chunk",
        "message_field": "data",
        "payload_encoding": "base64",
        "encoding": "pcm16",
        "sample_format": "s16le",
        "endianness": "little",
        "channels": 1,
        "channel_layout": "mono",
        "sample_rate_hz": 16000,
        "pcm_input": True,
    }
    assert cfg.channels.discord.enabled is False
    assert cfg.channels.discord.allowed_channel_ids == []
    assert cfg.channels.discord.allowed_user_ids == []
    assert cfg.channels.discord.rate_limit_per_user == 10
    assert cfg.channels.telegram.enabled is False
    assert cfg.channels.telegram.allowed_chat_ids == []
    assert cfg.channels.telegram.allowed_user_ids == []
    assert cfg.channels.telegram.rate_limit_per_user == 10
    assert cfg.tools.web_search_engine == "tavily"
    assert cfg.tools.web_search_searxng_base_url is None
    assert cfg.tools.web_search_brave_api_key is None
    assert cfg.security.autonomy_mode == "trusted_workspace"
    assert cfg.security.require_approval_for_exec is True
    assert cfg.security.require_approval_for_file_write is False
    assert cfg.security.file_ops_scope == "workspace"
    assert cfg.security.file_undo_max_size_mb == 2.0
    assert cfg.workspace_dir == defaults.default_workspace_dir()
    assert cfg.sessions_dir == defaults.default_sessions_dir()
    assert cfg.skills_dir == defaults.default_skills_dir()
    assert cfg.plugins_dir == defaults.default_plugins_dir()
    assert cfg.memory.db_path == defaults.default_memory_db_path()


def test_default_yaml_parseable() -> None:
    """configs/default.yaml 應可被 Pydantic 正確解析。"""
    yaml_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    assert yaml_path.exists(), "configs/default.yaml 不存在"

    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    cfg = MochiConfig.model_validate(raw)

    assert cfg.model == "ollama:llama3.2"
    assert cfg.agent.temperature == 0.7
    assert cfg.agent.max_tokens == 4096
    assert cfg.agent.top_p == 1.0
    assert cfg.agent.min_p == 0.0
    assert cfg.agent.top_k == 0
    assert cfg.agent.frequency_penalty == 0.0
    assert cfg.agent.presence_penalty == 0.0
    assert cfg.agent.repeat_penalty == 1.0
    assert cfg.agent.show_token_stats is False
    assert [preset.name for preset in cfg.agent.presets] == ["default"]
    assert cfg.agent.active_preset == "default"
    assert cfg.model_setup.mode == "configured_or_setup"
    assert cfg.model_setup.default_provider == "ollama"
    assert cfg.model_setup.default_model == "llama3.2"
    assert cfg.model_setup.default_model_spec == "ollama:llama3.2"
    assert cfg.model_setup.setup_required is True
    assert cfg.model_setup.configured_models == []
    assert cfg.model_setup.fallback_chain == [
        "user_config",
        "ollama_tags",
        "openai_compatible_provider",
    ]
    assert cfg.local_models.roots == []
    assert cfg.local_models.scan_max_depth == 3
    assert cfg.local_models.scan_max_entries == 500
    assert cfg.local_models.runtime == "inprocess"
    assert cfg.local_models.idle_unload_enabled is False
    assert cfg.local_models.idle_unload_seconds == 300
    assert cfg.local_models.llama_cpp.source is None
    assert cfg.local_models.llama_cpp.root_dir is None
    assert cfg.local_models.llama_cpp.python_executable is None
    assert cfg.local_models.llama_cpp.version is None
    assert cfg.vllm.enabled is False
    assert cfg.vllm.host == "127.0.0.1"
    assert cfg.vllm.port is None
    assert cfg.vllm.api_key is None
    assert cfg.vllm.tensor_parallel_size == 1
    assert cfg.vllm.dtype == "auto"
    assert cfg.vllm.gpu_memory_utilization == 0.9
    assert cfg.vllm.max_model_len is None
    assert cfg.vllm.trust_remote_code is False
    assert cfg.vllm.quantization is None
    assert cfg.vllm.startup_timeout_seconds == 180
    assert cfg.vllm.cuda_visible_devices is None
    assert cfg.gguf.n_batch == 512
    assert cfg.gguf.n_ubatch == 512
    assert cfg.gguf.n_threads_batch is None
    assert cfg.gguf.flash_attn is False
    assert cfg.gguf.offload_kqv is True
    assert cfg.gguf.use_mmap is True
    assert cfg.gguf.use_mlock is False
    assert cfg.locale_defaults.region_profile == "global"
    assert cfg.locale_defaults.ui_locale == "auto"
    assert cfg.locale_defaults.ui_locale_fallback == "en-US"
    assert cfg.locale_defaults.response_language == "same_as_user"
    assert cfg.locale_defaults.default_tts_voice == "af_heart"
    assert cfg.locale_defaults.timezone == "auto"
    assert cfg.voice.enabled is False
    assert cfg.voice.tts_backend == "kokoro-tts"
    assert cfg.voice.tts_voice == "af_heart"
    assert cfg.voice.stt_openai_base_url is None
    assert cfg.voice.stt_openai_api_key is None
    assert cfg.voice.stt_openai_timeout == 60.0
    assert cfg.voice.tts_model is None
    assert cfg.voice.tts_language is None
    assert cfg.voice.tts_use_gpu is False
    assert cfg.voice.tts_kokoro_lang_code == "a"
    assert cfg.voice.tts_split_pattern == r"\n+"
    assert cfg.voice.tts_openai_base_url is None
    assert cfg.voice.tts_openai_api_key is None
    assert cfg.voice.tts_openai_timeout == 60.0
    assert cfg.voice.tts_openai_response_format == "pcm"
    assert cfg.channels.discord.enabled is False
    assert cfg.channels.discord.allowed_channel_ids == []
    assert cfg.channels.discord.allowed_user_ids == []
    assert cfg.channels.discord.rate_limit_per_user == 10
    assert cfg.channels.telegram.enabled is False
    assert cfg.channels.telegram.allowed_chat_ids == []
    assert cfg.channels.telegram.allowed_user_ids == []
    assert cfg.channels.telegram.rate_limit_per_user == 10
    assert cfg.tools.web_search_engine == "tavily"
    assert cfg.tools.web_search_searxng_base_url is None
    assert cfg.tools.web_search_brave_api_key is None
    assert cfg.security.autonomy_mode == "trusted_workspace"
    assert cfg.security.require_approval_for_exec is True
    assert cfg.security.require_approval_for_file_write is False
    assert cfg.security.file_ops_scope == "workspace"
    assert cfg.security.file_undo_max_size_mb == 2.0


def test_default_paths_are_project_local_on_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows 預設 runtime path 應放在專案內，避免額外建立 `~/.mochi`。"""
    monkeypatch.setattr(defaults, "running_on_windows", lambda: True)

    cfg = load_config(config_path=_config_without_paths(tmp_path))

    assert cfg.workspace_dir == ".mochi"
    assert cfg.sessions_dir == ".mochi/sessions"
    assert cfg.skills_dir == ".mochi/skills"
    assert cfg.plugins_dir == ".mochi/plugins"
    assert cfg.memory.db_path == Path(".mochi/memory.db")
    assert defaults.default_config_path() == Path(".mochi/config.yaml")


def test_load_config_normalizes_legacy_windows_mochi_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows 應自動把 legacy `~/.mochi/...` 路徑正規化成專案內 `.mochi/...`。"""
    monkeypatch.setattr(defaults, "running_on_windows", lambda: True)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join([
            "model: ollama:test",
            "workspace_dir: ~/.mochi",
            "sessions_dir: ~/.mochi/sessions",
            "skills_dir: ~/.mochi/skills",
            "plugins_dir: ~/.mochi/plugins",
            "memory:",
            "  db_path: ~/.mochi/memory.db",
        ]),
        encoding="utf-8",
    )

    cfg = load_config(config_path=config_path)

    assert cfg.workspace_dir == ".mochi"
    assert cfg.sessions_dir == ".mochi/sessions"
    assert cfg.skills_dir == ".mochi/skills"
    assert cfg.plugins_dir == ".mochi/plugins"
    assert cfg.memory.db_path == Path(".mochi/memory.db")


def test_load_config_uses_project_local_windows_config_and_reads_legacy_home_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows 應以專案 `.mochi/config.yaml` 為主，必要時讀取 legacy home config。"""
    monkeypatch.setattr(defaults, "running_on_windows", lambda: True)
    home = tmp_path / "home"
    project = tmp_path / "project"
    (home / ".mochi").mkdir(parents=True)
    project.mkdir()
    (home / ".mochi" / "config.yaml").write_text(
        "\n".join([
            "model: ollama:legacy-home",
            "skills_dir: ~/.mochi/skills",
        ]),
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)

    cfg = load_config()

    assert cfg.model == "ollama:legacy-home"
    assert cfg.skills_dir == ".mochi/skills"
    assert config_manager.user_config_path() == Path(".mochi/config.yaml")
    assert (project / ".mochi" / "config.yaml").is_file()


def test_default_paths_use_home_on_posix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linux/macOS 預設 runtime path 應維持 `~/.mochi`。"""
    monkeypatch.setattr(defaults, "running_on_windows", lambda: False)

    cfg = load_config(config_path=_config_without_paths(tmp_path))

    assert cfg.workspace_dir == "~/.mochi"
    assert cfg.sessions_dir == "~/.mochi/sessions"
    assert cfg.skills_dir == "~/.mochi/skills"
    assert cfg.plugins_dir == "~/.mochi/plugins"
    assert cfg.memory.db_path == Path.home() / ".mochi" / "memory.db"
    assert defaults.default_config_path() == Path.home() / ".mochi" / "config.yaml"


def test_discord_channel_config_fields_parse() -> None:
    """Discord channel 設定欄位應可被正確解析。"""
    cfg = MochiConfig.model_validate(
        {
            "channels": {
                "discord": {
                    "enabled": True,
                    "bot_token": "discord-test-token",
                    "allowed_channel_ids": [100, 101],
                    "allowed_user_ids": [200, 201],
                    "rate_limit_per_user": 3,
                }
            }
        }
    )

    assert cfg.channels.discord.enabled is True
    assert cfg.channels.discord.bot_token is not None
    assert cfg.channels.discord.bot_token.get_secret_value() == "discord-test-token"
    assert cfg.channels.discord.allowed_channel_ids == [100, 101]
    assert cfg.channels.discord.allowed_user_ids == [200, 201]
    assert cfg.channels.discord.rate_limit_per_user == 3


def test_web_search_config_fields_parse() -> None:
    """web_search provider 設定應可被正確解析。"""
    cfg = MochiConfig.model_validate(
        {
            "tools": {
                "web_search_engine": "brave",
                "web_search_searxng_base_url": "https://search.example.com",
                "web_search_brave_api_key": "brave-test-key",
            }
        }
    )

    assert cfg.tools.web_search_engine == "brave"
    assert cfg.tools.web_search_searxng_base_url == "https://search.example.com"
    assert cfg.tools.web_search_brave_api_key is not None
    assert cfg.tools.web_search_brave_api_key.get_secret_value() == "brave-test-key"


def test_voice_config_stt_backend_validation() -> None:
    """STT backend 應只接受合法的 Literal 值。"""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        MochiConfig.model_validate({"voice": {"stt_backend": "invalid-backend"}})


def test_voice_config_external_api_stt_fields_parse() -> None:
    """`external-api` STT 設定應可被正確解析，並兼容舊 alias。"""
    cfg = MochiConfig.model_validate(
        {
            "voice": {
                "stt_backend": "external-api",
                "stt_model": "whisper-1",
                "stt_openai_base_url": "http://api.example.com/v1",
                "stt_openai_api_key": "sk-test",
                "stt_openai_timeout": 12.5,
            }
        }
    )

    assert cfg.voice.stt_backend == "external-api"
    assert cfg.voice.stt_model == "whisper-1"
    assert cfg.voice.stt_openai_base_url == "http://api.example.com/v1"
    assert cfg.voice.stt_openai_api_key is not None
    assert cfg.voice.stt_openai_api_key.get_secret_value() == "sk-test"
    assert cfg.voice.stt_openai_timeout == 12.5

    legacy = MochiConfig.model_validate(
        {
            "voice": {
                "stt_backend": "openai-api",
                "stt_model": "whisper-1",
            }
        }
    )
    assert legacy.voice.stt_backend == "openai-api"


def test_voice_config_extended_stt_backend_literals_parse() -> None:
    """新增 STT backend literal 應可被正確解析。"""
    cfg = MochiConfig.model_validate(
        {
            "voice": {
                "stt_backend": "qwen-asr",
                "stt_model": "qwen3-asr-0.6b",
            }
        }
    )

    assert cfg.voice.stt_backend == "qwen-asr"
    assert cfg.voice.stt_model == "qwen3-asr-0.6b"


def test_voice_config_tts_backend_validation() -> None:
    """TTS backend 應只接受合法的 Literal 值。"""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        MochiConfig.model_validate({"voice": {"tts_backend": "invalid-backend"}})


def test_voice_config_rejects_non_mono_api_input_channels() -> None:
    """`/v1/voice` 契約固定 mono，channels=1 以外應拒絕。"""
    from pydantic import ValidationError

    with pytest.raises(
        ValidationError,
        match=(
            r"voice\.channels must be 1 because /v1/voice only accepts mono "
            r"PCM16 input \(base64-encoded s16le\)\."
        ),
    ):
        MochiConfig.model_validate({"voice": {"channels": 2}})


def test_voice_config_external_api_tts_fields_parse() -> None:
    """`external-api` TTS 設定應可被正確解析，並兼容舊 alias。"""
    cfg = MochiConfig.model_validate(
        {
            "voice": {
                "tts_backend": "external-api",
                "tts_model": "gpt-4o-mini-tts",
                "tts_voice": "alloy",
                "tts_language": "en",
                "tts_use_gpu": True,
                "tts_kokoro_lang_code": "z",
                "tts_split_pattern": r"[.!?]+\s+",
                "tts_openai_base_url": "http://api.example.com/v1",
                "tts_openai_api_key": "sk-test-tts",
                "tts_openai_timeout": 20.0,
                "tts_openai_response_format": "wav",
            }
        }
    )

    assert cfg.voice.tts_backend == "external-api"
    assert cfg.voice.tts_model == "gpt-4o-mini-tts"
    assert cfg.voice.tts_voice == "alloy"
    assert cfg.voice.tts_language == "en"
    assert cfg.voice.tts_use_gpu is True
    assert cfg.voice.tts_kokoro_lang_code == "z"
    assert cfg.voice.tts_split_pattern == r"[.!?]+\s+"
    assert cfg.voice.tts_openai_base_url == "http://api.example.com/v1"
    assert cfg.voice.tts_openai_api_key is not None
    assert cfg.voice.tts_openai_api_key.get_secret_value() == "sk-test-tts"
    assert cfg.voice.tts_openai_timeout == 20.0
    assert cfg.voice.tts_openai_response_format == "wav"

    legacy = MochiConfig.model_validate(
        {
            "voice": {
                "tts_backend": "openai-tts",
                "tts_model": "gpt-4o-mini-tts",
            }
        }
    )
    assert legacy.voice.tts_backend == "openai-tts"


def test_config_from_partial_yaml() -> None:
    """只覆寫部分欄位時，其餘應保留預設值。"""
    partial = {"model": "ollama:qwen2.5", "log_level": "DEBUG"}
    cfg = MochiConfig.model_validate(partial)
    assert cfg.model == "ollama:qwen2.5"
    assert cfg.log_level == "DEBUG"
    assert cfg.agent.max_react_iterations == 10  # 預設值保留


def test_local_models_llama_cpp_runtime_fields_parse() -> None:
    """llama.cpp runtime 設定欄位應可被正確解析。"""
    cfg = MochiConfig.model_validate(
        {
            "local_models": {
                "llama_cpp": {
                    "source": "existing_path",
                    "root_dir": "/models/llama.cpp",
                    "python_executable": "/usr/bin/python3",
                    "version": "b9999",
                }
            }
        }
    )

    assert cfg.local_models.llama_cpp.source == "existing_path"
    assert cfg.local_models.llama_cpp.root_dir == Path("/models/llama.cpp")
    assert cfg.local_models.llama_cpp.python_executable == "/usr/bin/python3"
    assert cfg.local_models.llama_cpp.version == "b9999"


def test_local_models_idle_unload_legacy_positive_seconds_enable_feature() -> None:
    """舊設定若僅有正整數 idle_unload_seconds，升級後仍應保留啟用狀態。"""
    cfg = MochiConfig.model_validate(
        {
            "local_models": {
                "idle_unload_seconds": 300,
            }
        }
    )

    assert cfg.local_models.idle_unload_enabled is True
    assert cfg.local_models.idle_unload_seconds == 300


def test_local_models_idle_unload_legacy_zero_seconds_disable_feature() -> None:
    """舊設定若 idle_unload_seconds 為 0，升級後應視為停用。"""
    cfg = MochiConfig.model_validate(
        {
            "local_models": {
                "idle_unload_seconds": 0,
            }
        }
    )

    assert cfg.local_models.idle_unload_enabled is False
    assert cfg.local_models.idle_unload_seconds == 0


def test_load_config_prefers_user_config_over_project_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """未指定 config path 時，使用者設定應覆蓋 repo default。"""
    home = tmp_path / "home"
    project = tmp_path / "project"
    (home / ".mochi").mkdir(parents=True)
    (project / "configs").mkdir(parents=True)
    (home / ".mochi" / "config.yaml").write_text(
        "model: ollama:user-model\n",
        encoding="utf-8",
    )
    (project / "configs" / "default.yaml").write_text(
        "model: ollama:project-default\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)

    cfg = load_config()

    assert cfg.model == "ollama:user-model"


def test_load_config_ignores_console_encoding_errors_in_debug_logging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Console encoding failures in debug logging must not break config loading."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model: ollama:test\n", encoding="utf-8")

    def _boom(message: str, *args: object) -> None:  # noqa: ARG001
        raise UnicodeEncodeError("cp950", "x", 0, 1, "illegal multibyte sequence")

    monkeypatch.setattr(config_manager.logger, "debug", _boom)

    cfg = load_config(config_path=config_path)

    assert cfg.model == "ollama:test"


def test_load_config_applies_env_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """環境變數應可覆蓋 web/ollama 常用部署設定。"""
    home = tmp_path / "home"
    project = tmp_path / "project"
    (home / ".mochi").mkdir(parents=True)
    (project / "configs").mkdir(parents=True)
    (project / "configs" / "default.yaml").write_text(
        "\n".join([
            "web:",
            "  host: \"0.0.0.0\"",
            "  port: 8000",
            "  cors_origins:",
            "    - \"http://localhost:3000\"",
            "ollama:",
            "  base_url: \"http://localhost:11434\"",
            "locale_defaults:",
            "  region_profile: \"global\"",
            "  ui_locale: \"auto\"",
            "  ui_locale_fallback: \"en-US\"",
            "  response_language: \"same_as_user\"",
            "  default_tts_voice: \"af_heart\"",
            "  timezone: \"auto\"",
            "voice:",
            "  tts_voice: \"af_heart\"",
        ]),
        encoding="utf-8",
    )

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)
    monkeypatch.setenv("MOCHI_WEB_HOST", "127.0.0.1")
    monkeypatch.setenv("MOCHI_WEB_PORT", "19090")
    monkeypatch.setenv(
        "MOCHI_WEB_CORS_ORIGINS",
        "https://ui.example.com, https://admin.example.com",
    )
    monkeypatch.setenv("MOCHI_OLLAMA_BASE_URL", "http://ollama.internal:11434")
    monkeypatch.setenv("MOCHI_REGION_PROFILE", "us")
    monkeypatch.setenv("MOCHI_LOCALE", "en-US")
    monkeypatch.setenv("MOCHI_UI_LOCALE_FALLBACK", "en-GB")
    monkeypatch.setenv("MOCHI_TIMEZONE", "America/New_York")
    monkeypatch.setenv("MOCHI_RESPONSE_LANGUAGE", "en-US")
    monkeypatch.setenv("MOCHI_DEFAULT_TTS_VOICE", "en-US-JennyNeural")
    monkeypatch.setenv("MOCHI_TTS_VOICE", "en-US-GuyNeural")
    monkeypatch.setenv("MOCHI_WORKSPACE_DIR", "/data/mochi/workspace")
    monkeypatch.setenv("MOCHI_SESSIONS_DIR", "/data/mochi/sessions")
    monkeypatch.setenv("MOCHI_SKILLS_DIR", "/data/mochi/skills")
    monkeypatch.setenv("MOCHI_PLUGINS_DIR", "/data/mochi/plugins")
    monkeypatch.setenv("MOCHI_SERPER_API_KEY", "serper-env-key")
    monkeypatch.setenv("MOCHI_EXA_API_KEY", "exa-env-key")
    monkeypatch.setenv("MOCHI_WEB_FETCH_JINA_API_KEY", "jina-fetch-env-key")

    cfg = load_config()

    assert cfg.web.host == "127.0.0.1"
    assert cfg.web.port == 19090
    assert cfg.web.cors_origins == [
        "https://ui.example.com",
        "https://admin.example.com",
    ]
    assert cfg.ollama.base_url == "http://ollama.internal:11434"
    assert cfg.locale_defaults.region_profile == "us"
    assert cfg.locale_defaults.ui_locale == "en-US"
    assert cfg.locale_defaults.ui_locale_fallback == "en-GB"
    assert cfg.locale_defaults.response_language == "en-US"
    assert cfg.locale_defaults.default_tts_voice == "en-US-JennyNeural"
    assert cfg.locale_defaults.timezone == "America/New_York"
    assert cfg.voice.tts_voice == "en-US-GuyNeural"
    assert cfg.workspace_dir == "/data/mochi/workspace"
    assert cfg.sessions_dir == "/data/mochi/sessions"
    assert cfg.skills_dir == "/data/mochi/skills"
    assert cfg.plugins_dir == "/data/mochi/plugins"
    assert cfg.tools.web_search_serper_api_key is not None
    assert cfg.tools.web_search_serper_api_key.get_secret_value() == "serper-env-key"
    assert cfg.tools.web_search_exa_api_key is not None
    assert cfg.tools.web_search_exa_api_key.get_secret_value() == "exa-env-key"
    assert cfg.tools.web_fetch_jina_api_key is not None
    assert cfg.tools.web_fetch_jina_api_key.get_secret_value() == "jina-fetch-env-key"


def test_save_config_preserves_secret_values(tmp_path: Path) -> None:
    """保存本機 YAML 時應寫入 SecretStr 原始值，而不是遮罩字串。"""
    config_path = tmp_path / "config.yaml"
    cfg = MochiConfig.model_validate(
        {
            "model": "https://api.example.com/v1",
            "openai_compat": {
                "base_url": "https://api.example.com/v1",
                "model": "gpt-test",
                "api_key": "sk-local-secret",
            },
            "channels": {
                "discord": {
                    "enabled": True,
                    "bot_token": "discord-local-secret",
                }
            },
        }
    )

    save_config(cfg, config_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert raw["openai_compat"]["api_key"] == "sk-local-secret"
    assert raw["channels"]["discord"]["bot_token"] == "discord-local-secret"
    assert "**********" not in config_path.read_text(encoding="utf-8")


def test_openai_compat_provider_accepts_vllm() -> None:
    """openai_compat.provider 應接受 vllm。"""
    cfg = MochiConfig.model_validate(
        {
            "model": "https://vllm.example.com/v1",
            "openai_compat": {
                "provider": "vllm",
                "base_url": "https://vllm.example.com/v1",
                "model": "Qwen/Qwen3-8B",
            },
        }
    )

    assert cfg.openai_compat.provider == "vllm"
    assert cfg.openai_compat.base_url == "https://vllm.example.com/v1"
    assert cfg.openai_compat.model == "Qwen/Qwen3-8B"


def test_model_setup_configured_model_provider_accepts_vllm() -> None:
    """model_setup.configured_models.provider 應接受 vllm。"""
    cfg = MochiConfig.model_validate(
        {
            "model_setup": {
                "configured_models": [
                    {
                        "id": "vllm-prod",
                        "provider": "vllm",
                        "model": "Qwen/Qwen3-8B",
                        "model_spec": "https://vllm.example.com/v1",
                        "base_url": "https://vllm.example.com/v1",
                        "backend_type": "openai_compat",
                    }
                ]
            }
        }
    )

    assert len(cfg.model_setup.configured_models) == 1
    assert cfg.model_setup.configured_models[0].provider == "vllm"


def test_openai_codex_provider_accepts_oauth_metadata() -> None:
    """openai_codex provider metadata should validate without API keys in YAML."""
    cfg = MochiConfig.model_validate(
        {
            "model": "https://chatgpt.com/backend-api",
            "openai_codex": {
                "base_url": "https://chatgpt.com/backend-api",
                "model": "gpt-5.4",
                "auth_profile_id": "openai_codex:default",
            },
            "model_setup": {
                "configured_models": [
                    {
                        "id": "openai_codex:https://chatgpt.com/backend-api:gpt-5.4",
                        "provider": "openai_codex",
                        "model": "gpt-5.4",
                        "model_spec": "https://chatgpt.com/backend-api",
                        "base_url": "https://chatgpt.com/backend-api",
                        "backend_type": "openai_codex",
                        "auth_profile_id": "openai_codex:default",
                        "auth_mode": "oauth",
                    }
                ]
            },
        }
    )

    assert cfg.openai_codex.auth_profile_id == "openai_codex:default"
    assert cfg.model_setup.configured_models[0].provider == "openai_codex"
    assert cfg.model_setup.configured_models[0].auth_mode == "oauth"


def test_model_setup_configured_model_launch_mode_accepts_managed() -> None:
    """configured_models.launch_mode 應接受 managed/external。"""
    cfg = MochiConfig.model_validate(
        {
            "model_setup": {
                "configured_models": [
                    {
                        "id": "vllm-managed",
                        "provider": "vllm",
                        "model": "Qwen/Qwen3-8B",
                        "model_spec": "http://127.0.0.1:8000/v1",
                        "launch_mode": "managed",
                    },
                    {
                        "id": "vllm-external",
                        "provider": "vllm",
                        "model": "Qwen/Qwen3-8B",
                        "model_spec": "https://vllm.example.com/v1",
                        "launch_mode": "external",
                    },
                ]
            }
        }
    )

    assert cfg.model_setup.configured_models[0].launch_mode == "managed"
    assert cfg.model_setup.configured_models[1].launch_mode == "external"


def test_model_setup_configured_model_launch_mode_rejects_invalid_value() -> None:
    """configured_models.launch_mode 應拒絕非 external/managed。"""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        MochiConfig.model_validate(
            {
                "model_setup": {
                    "configured_models": [
                        {
                            "id": "vllm-invalid",
                            "provider": "vllm",
                            "model": "Qwen/Qwen3-8B",
                            "model_spec": "http://127.0.0.1:8000/v1",
                            "launch_mode": "embedded",
                        }
                    ]
                }
            }
        )


def test_vllm_config_fields_parse() -> None:
    """vllm schema 欄位應可正確解析。"""
    cfg = MochiConfig.model_validate(
        {
            "vllm": {
                "enabled": True,
                "host": "0.0.0.0",
                "port": 18000,
                "api_key": "vllm-test-key",
                "tensor_parallel_size": 2,
                "dtype": "bfloat16",
                "gpu_memory_utilization": 0.85,
                "max_model_len": 8192,
                "trust_remote_code": True,
                "quantization": "awq",
                "startup_timeout_seconds": 300,
                "cuda_visible_devices": "0,1",
            }
        }
    )

    assert cfg.vllm.enabled is True
    assert cfg.vllm.host == "0.0.0.0"
    assert cfg.vllm.port == 18000
    assert cfg.vllm.api_key is not None
    assert cfg.vllm.api_key.get_secret_value() == "vllm-test-key"
    assert cfg.vllm.tensor_parallel_size == 2
    assert cfg.vllm.dtype == "bfloat16"
    assert cfg.vllm.gpu_memory_utilization == 0.85
    assert cfg.vllm.max_model_len == 8192
    assert cfg.vllm.trust_remote_code is True
    assert cfg.vllm.quantization == "awq"
    assert cfg.vllm.startup_timeout_seconds == 300
    assert cfg.vllm.cuda_visible_devices == "0,1"


def test_vllm_config_rejects_invalid_gpu_memory_utilization() -> None:
    """vllm.gpu_memory_utilization 應限制在 0.0~1.0。"""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        MochiConfig.model_validate(
            {
                "vllm": {
                    "gpu_memory_utilization": 1.5,
                }
            }
        )


def test_model_setup_configured_model_launch_mode_defaults_to_none() -> None:
    """configured_models.launch_mode 預設應為 None。"""
    cfg = MochiConfig.model_validate(
        {
            "model_setup": {
                "configured_models": [
                    {
                        "id": "vllm-no-launch-mode",
                        "provider": "vllm",
                        "model": "Qwen/Qwen3-8B",
                        "model_spec": "https://vllm.example.com/v1",
                    }
                ]
            }
        }
    )

    assert cfg.model_setup.configured_models[0].launch_mode is None
