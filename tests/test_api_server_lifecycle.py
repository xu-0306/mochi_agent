"""API 生命週期與語音 session 釋放測試。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from fastapi.testclient import TestClient
from pydantic import SecretStr

from mochi.api.server import (
    _attach_voice_bridge_diagnostics,
    _get_or_create_engine,
    _merge_voice_bridge_diagnostics,
    _rebuild_channel_manager,
    _shutdown_engine,
    create_app,
)
from mochi.config.schema import MochiConfig


@dataclass
class _StubChannel:
    name: str
    starts: int = 0
    stops: int = 0
    handler: Callable[..., Awaitable[None]] | None = None

    def set_event_handler(self, handler: Callable[..., Awaitable[None]]) -> None:
        self.handler = handler

    async def start(self) -> None:
        self.starts += 1

    async def stop(self) -> None:
        self.stops += 1


class _FakeEngine:
    def __init__(self) -> None:
        self.release_calls: list[str | None] = []
        self.closed = False

    async def release_voice_session(self, session_id: str | None = None) -> bool:
        self.release_calls.append(session_id)
        return True

    async def close(self) -> None:
        self.closed = True


class _FakeVLLMManager:
    def __init__(self) -> None:
        self.stop_calls = 0

    async def stop(self) -> dict[str, str]:
        self.stop_calls += 1
        return {"state": "stopped"}


def test_engine_factory_supports_zero_arg_callable() -> None:
    """engine_factory 支援 0 參數 callable。"""
    engine = _FakeEngine()
    app = create_app()
    app.state.engine_factory = lambda: engine
    resolved = asyncio.run(_get_or_create_engine(app))
    assert resolved is engine
    assert app.state.engine is engine


def test_engine_factory_accepts_app_argument() -> None:
    """engine_factory 可接收 app 參數建立 engine。"""
    engine = _FakeEngine()
    app = create_app()
    expected_app = app
    call_count = 0

    def factory(app) -> _FakeEngine:
        nonlocal call_count
        call_count += 1
        assert app is expected_app
        return engine

    app.state.engine_factory = factory
    resolved = asyncio.run(_get_or_create_engine(app))
    resolved_again = asyncio.run(_get_or_create_engine(app))
    assert resolved is engine
    assert resolved_again is engine
    assert call_count == 1


def test_get_or_create_engine_injects_shared_vllm_runtime_manager(monkeypatch) -> None:
    """_get_or_create_engine 預設建立 AgentEngine 時要注入 app.state vLLM manager。"""
    app = create_app()
    app.state.config_factory = lambda: MochiConfig()
    shared_manager = object()
    app.state.vllm_runtime_manager = shared_manager

    class _ProbeEngine:
        def __init__(self, config: MochiConfig, *, vllm_runtime_manager: object | None = None) -> None:
            self.config = config
            self.vllm_runtime_manager = vllm_runtime_manager

    monkeypatch.setattr("mochi.agents.engine.AgentEngine", _ProbeEngine)

    resolved = asyncio.run(_get_or_create_engine(app))

    assert isinstance(resolved, _ProbeEngine)
    assert resolved.vllm_runtime_manager is shared_manager
    assert app.state.engine is resolved


def test_shutdown_engine_closes_existing_engine() -> None:
    """shutdown 會呼叫 engine.close 並清空 state.engine。"""
    engine = _FakeEngine()
    app = create_app()
    app.state.engine = engine
    asyncio.run(_shutdown_engine(app))
    assert engine.closed is True
    assert app.state.engine is None


def test_shutdown_engine_stops_vllm_runtime_manager_without_engine() -> None:
    manager = _FakeVLLMManager()
    app = create_app()
    app.state.vllm_runtime_manager = manager

    asyncio.run(_shutdown_engine(app))

    assert manager.stop_calls == 1
    assert app.state.vllm_runtime_manager is None


def test_merge_voice_bridge_diagnostics_accumulates_counts_and_replaces_last_failure() -> None:
    """bridge diagnostics 應累積計數並以最新 failure 覆蓋 last_preview_failure。"""
    app = create_app()

    _merge_voice_bridge_diagnostics(
        app,
        {
            "preview_append_failures": 1,
            "preview_flush_failures": 0,
            "preview_degraded_turns": 1,
            "last_preview_failure": {
                "stage": "append",
                "error_type": "RuntimeError",
                "message": "append boom",
            },
        },
    )
    _merge_voice_bridge_diagnostics(
        app,
        {
            "preview_append_failures": 2,
            "preview_flush_failures": 3,
            "preview_degraded_turns": 4,
            "last_preview_failure": {
                "stage": "flush",
                "error_type": "RuntimeError",
                "message": "flush boom",
            },
        },
    )

    assert app.state.voice_bridge_diagnostics == {
        "preview_append_failures": 3,
        "preview_flush_failures": 3,
        "preview_degraded_turns": 5,
        "last_preview_failure": {
            "stage": "flush",
            "error_type": "RuntimeError",
            "message": "flush boom",
        },
    }


def test_attach_voice_bridge_diagnostics_returns_copy_of_state() -> None:
    """status payload 應攜帶 bridge diagnostics，但不應回傳 state 的可變引用。"""
    app = create_app()
    app.state.voice_bridge_diagnostics = {
        "preview_append_failures": 1,
        "preview_flush_failures": 2,
        "preview_degraded_turns": 3,
        "last_preview_failure": {
            "stage": "flush",
            "message": "boom",
        },
    }

    payload = _attach_voice_bridge_diagnostics(app, {"type": "voice_runtime_status"})
    payload["bridge_diagnostics"]["preview_append_failures"] = 99
    payload["bridge_diagnostics"]["last_preview_failure"]["message"] = "mutated"

    assert app.state.voice_bridge_diagnostics == {
        "preview_append_failures": 1,
        "preview_flush_failures": 2,
        "preview_degraded_turns": 3,
        "last_preview_failure": {
            "stage": "flush",
            "message": "boom",
        },
    }


def test_channels_status_reports_supported_channels_without_tokens() -> None:
    """`/v1/channels` 應揭露 bounded status 且不洩漏 bot token。"""
    app = create_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "channels": {
                "discord": {
                    "enabled": True,
                    "bot_token": SecretStr("discord-secret-token"),
                    "allowed_channel_ids": [123],
                    "allowed_user_ids": [456],
                    "rate_limit_per_user": 5,
                },
                "telegram": {
                    "enabled": False,
                    "bot_token": SecretStr("telegram-secret-token"),
                },
            }
        }
    )

    with TestClient(app) as client:
        response = client.get("/v1/channels")

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "channels_status"
    assert payload["phase"] == "bounded"
    assert payload["supported_channels"] == ["discord", "telegram"]
    assert payload["channels"]["discord"] == {
        "supported": True,
        "enabled": True,
        "registered": False,
        "running": False,
        "text_enabled": True,
        "voice_enabled": False,
        "bot_token_configured": True,
        "allowed_guild_ids": [],
        "allowed_channel_ids": [123],
        "allowed_voice_channel_ids": [],
        "allowed_user_ids": [456],
        "rate_limit_per_user": 5,
        "message_mode": "mentions_only",
        "auto_join_policy": "manual_only",
        "voice_auto_reply": True,
        "voice_stt_enabled": True,
        "voice_tts_enabled": True,
    }
    assert payload["channels"]["telegram"] == {
        "supported": True,
        "enabled": False,
        "registered": False,
        "running": False,
        "allowed_chat_ids": [],
        "allowed_user_ids": [],
        "rate_limit_per_user": 10,
        "bot_token_configured": True,
    }
    assert "secret-token" not in response.text


def test_channels_status_marks_registered_without_running_for_registered_discord_channel() -> None:
    """已註冊但未啟動的 Discord channel 應反映 registered=true, running=false。"""
    from mochi.channels.manager import ChannelManager

    app = create_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "channels": {
                "discord": {
                    "enabled": True,
                    "bot_token": SecretStr("discord-secret-token"),
                }
            }
        }
    )
    manager = ChannelManager(engine=_FakeEngine())
    manager.register(_StubChannel(name="discord"))
    app.state.channel_manager = manager

    with TestClient(app) as client:
        response = client.get("/v1/channels")

    assert response.status_code == 200
    assert response.json()["channels"]["discord"] == {
        "supported": True,
        "enabled": True,
        "registered": True,
        "running": False,
        "text_enabled": True,
        "voice_enabled": False,
        "bot_token_configured": True,
        "allowed_guild_ids": [],
        "allowed_channel_ids": [],
        "allowed_voice_channel_ids": [],
        "allowed_user_ids": [],
        "rate_limit_per_user": 10,
        "message_mode": "mentions_only",
        "auto_join_policy": "manual_only",
        "voice_auto_reply": True,
        "voice_stt_enabled": True,
        "voice_tts_enabled": True,
    }


def test_channels_start_stop_single_and_all_with_non_sensitive_status() -> None:
    """`/v1/channels` 管理 API 可啟停單一/全部頻道，且維持非敏感回傳。"""
    from mochi.channels.manager import ChannelManager

    app = create_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "channels": {
                "discord": {
                    "enabled": True,
                    "bot_token": SecretStr("discord-secret-token"),
                },
                "telegram": {
                    "enabled": True,
                    "bot_token": SecretStr("telegram-secret-token"),
                },
            }
        }
    )
    manager = ChannelManager(engine=_FakeEngine())
    discord = _StubChannel(name="discord")
    telegram = _StubChannel(name="telegram")
    manager.register(discord)
    manager.register(telegram)
    app.state.channel_manager = manager

    with TestClient(app) as client:
        start_one = client.post("/v1/channels/discord/start")
        assert start_one.status_code == 200
        assert start_one.json() == {
            "type": "channels_control",
            "action": "start",
            "scope": "single",
            "channel": "discord",
            "running": True,
            "running_channels": ["discord"],
        }

        status_one = client.get("/v1/channels")
        assert status_one.status_code == 200
        assert status_one.json()["channels"]["discord"]["running"] is True
        assert status_one.json()["channels"]["telegram"]["running"] is False

        start_all = client.post("/v1/channels/start")
        assert start_all.status_code == 200
        assert start_all.json() == {
            "type": "channels_control",
            "action": "start",
            "scope": "all",
            "running_channels": ["discord", "telegram"],
        }

        stop_one = client.post("/v1/channels/telegram/stop")
        assert stop_one.status_code == 200
        assert stop_one.json() == {
            "type": "channels_control",
            "action": "stop",
            "scope": "single",
            "channel": "telegram",
            "running": False,
            "running_channels": ["discord"],
        }

        stop_all = client.post("/v1/channels/stop")
        assert stop_all.status_code == 200
        assert stop_all.json() == {
            "type": "channels_control",
            "action": "stop",
            "scope": "all",
            "running_channels": [],
        }

        status_final = client.get("/v1/channels")
        assert status_final.status_code == 200
        payload = status_final.json()
        assert payload["channels"]["discord"]["running"] is False
        assert payload["channels"]["telegram"]["running"] is False
        assert "secret-token" not in status_final.text

    assert discord.starts == 1
    assert discord.stops == 1
    assert telegram.starts == 1
    assert telegram.stops == 1


def test_channels_status_without_running_channels_method_stays_conservative() -> None:
    """缺少 `running_channels()` 時，status 應保守回報 registered=true、running=false。"""
    app = create_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "channels": {
                "discord": {
                    "enabled": True,
                    "bot_token": SecretStr("discord-secret-token"),
                },
                "telegram": {
                    "enabled": True,
                    "bot_token": SecretStr("telegram-secret-token"),
                },
            }
        }
    )

    class _LegacyManager:
        _running = True

        def list_channels(self) -> list[str]:
            return ["discord", "telegram"]

    app.state.channel_manager = _LegacyManager()

    with TestClient(app) as client:
        response = client.get("/v1/channels")

    assert response.status_code == 200
    payload = response.json()
    assert payload["channels"]["discord"]["registered"] is True
    assert payload["channels"]["discord"]["running"] is False
    assert payload["channels"]["telegram"]["registered"] is True
    assert payload["channels"]["telegram"]["running"] is False


def test_channels_control_returns_404_for_unknown_channel() -> None:
    """未註冊 channel 的單一啟停請求應回 404。"""
    from mochi.channels.manager import ChannelManager

    app = create_app()
    manager = ChannelManager(engine=_FakeEngine())
    manager.register(_StubChannel(name="discord"))
    app.state.channel_manager = manager

    with TestClient(app) as client:
        start_missing = client.post("/v1/channels/telegram/start")
        stop_missing = client.post("/v1/channels/telegram/stop")

    assert start_missing.status_code == 404
    assert start_missing.json() == {"detail": "Channel not registered"}
    assert stop_missing.status_code == 404
    assert stop_missing.json() == {"detail": "Channel not registered"}


def test_channels_control_returns_503_when_manager_is_missing() -> None:
    """未注入 channel manager 時，管理 API 應回 503。"""
    app = create_app()
    with TestClient(app) as client:
        response = client.post("/v1/channels/start")
    assert response.status_code == 503
    assert response.json() == {"detail": "Channel manager is not configured"}


def test_channels_start_lazily_builds_manager_from_current_config(monkeypatch) -> None:
    """`/v1/channels/{name}/start` 應可用目前 config lazy 建立 manager。"""
    from mochi.channels.manager import ChannelManager
    import mochi.api.server as server_module

    app = create_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "channels": {
                "discord": {
                    "enabled": True,
                    "bot_token": SecretStr("discord-secret-token"),
                }
            }
        }
    )
    app.state.engine_factory = lambda: _FakeEngine()

    stub = _StubChannel(name="discord")

    async def fake_builder(app) -> ChannelManager:  # noqa: ANN001
        config = await server_module._get_config(app)
        engine = await server_module._get_or_create_engine(app)
        assert config.channels.discord.enabled is True
        manager = ChannelManager(engine=engine)
        manager.register(stub)
        return manager

    monkeypatch.setattr(server_module, "_build_channel_manager_for_app", fake_builder)

    with TestClient(app) as client:
        start_response = client.post("/v1/channels/discord/start")
        status_response = client.get("/v1/channels")

    assert start_response.status_code == 200
    assert start_response.json() == {
        "type": "channels_control",
        "action": "start",
        "scope": "single",
        "channel": "discord",
        "running": True,
        "running_channels": ["discord"],
    }
    assert status_response.status_code == 200
    assert status_response.json()["channels"]["discord"]["registered"] is True
    assert status_response.json()["channels"]["discord"]["running"] is True
    assert stub.starts == 1


def test_rebuild_channel_manager_restarts_previously_running_channels() -> None:
    """rebuild manager 後，先前 running 的 channel 應嘗試在新 manager 上恢復。"""
    from mochi.channels.manager import ChannelManager

    app = create_app()
    old_manager = ChannelManager(engine=_FakeEngine())
    old_channel = _StubChannel(name="discord")
    old_manager.register(old_channel)
    asyncio.run(old_manager.start_channel("discord"))
    app.state.channel_manager = old_manager

    new_manager = ChannelManager(engine=_FakeEngine())
    new_channel = _StubChannel(name="discord")
    new_manager.register(new_channel)

    async def fake_builder(app) -> ChannelManager:  # noqa: ANN001
        return new_manager

    app.state.engine_factory = lambda: _FakeEngine()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "channels": {
                "discord": {
                    "enabled": True,
                    "bot_token": SecretStr("discord-secret-token"),
                }
            }
        }
    )

    import mochi.api.server as server_module

    original_builder = server_module._build_channel_manager_for_app
    server_module._build_channel_manager_for_app = fake_builder  # type: ignore[assignment]
    try:
        rebuilt = asyncio.run(_rebuild_channel_manager(app))
    finally:
        server_module._build_channel_manager_for_app = original_builder  # type: ignore[assignment]

    assert rebuilt is new_manager
    assert app.state.channel_manager is new_manager
    assert old_channel.stops == 1
    assert new_channel.starts == 1


def test_create_app_applies_cors_origins_from_env(monkeypatch) -> None:
    """create_app 應支援以環境變數覆蓋 CORS origins。"""
    monkeypatch.setenv("MOCHI_WEB_CORS_ORIGINS", "https://ui.example.com")
    app = create_app()

    with TestClient(app) as client:
        response = client.options(
            "/health",
            headers={
                "Origin": "https://ui.example.com",
                "Access-Control-Request-Method": "GET",
            },
        )

    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "https://ui.example.com"


def test_create_app_keeps_env_cors_when_config_load_fails(monkeypatch) -> None:
    """config 暫時壞掉時，app bootstrap 仍應保留明確 env CORS 設定。"""
    import mochi.config.manager as config_manager

    def raise_config_error():
        raise RuntimeError("broken config")

    monkeypatch.setenv("MOCHI_WEB_CORS_ORIGINS", "https://ui.example.com")
    monkeypatch.setattr(config_manager, "load_config", raise_config_error)
    app = create_app()

    with TestClient(app) as client:
        response = client.options(
            "/health",
            headers={
                "Origin": "https://ui.example.com",
                "Access-Control-Request-Method": "GET",
            },
        )

    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "https://ui.example.com"
