from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import SecretStr

from mochi.config.schema import MochiConfig

from ._support import _create_test_app


def test_discord_setup_persists_secret_without_exposing_it(tmp_path: Path) -> None:
    """`POST /v1/setup/discord` 應保存 token，但回應不得回傳 secret。"""
    config_path = tmp_path / "config.yaml"
    config = MochiConfig.model_validate({})
    app = _create_test_app(config=config)
    app.state.config_path = config_path

    with TestClient(app) as client:
        response = client.post(
            "/v1/setup/discord",
            json={
                "bot_token": "discord-super-secret-token",
                "enabled": True,
                "text_enabled": True,
                "voice_enabled": True,
                "allowed_guild_ids": [1234],
                "allowed_channel_ids": [5678],
                "allowed_voice_channel_ids": [9012],
                "allowed_user_ids": [3456],
                "message_mode": "mentions_only",
                "voice_auto_reply": True,
                "voice_stt_enabled": True,
                "voice_tts_enabled": True,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["channels"]["discord"] == {
        "enabled": True,
        "text_enabled": True,
        "voice_enabled": True,
        "bot_token_configured": True,
        "allowed_guild_ids": [1234],
        "allowed_channel_ids": [5678],
        "allowed_voice_channel_ids": [9012],
        "allowed_user_ids": [3456],
        "admin_user_ids": [],
        "rate_limit_per_user": 10,
        "message_mode": "mentions_only",
        "auto_join_policy": "manual_only",
        "voice_auto_reply": True,
        "voice_stt_enabled": True,
        "voice_tts_enabled": True,
    }
    assert payload["update"] == {
        "type": "discord_setup",
        "persisted": True,
        "config_path": str(config_path),
        "discord": {
            "configured": True,
            "enabled": True,
            "text_enabled": True,
            "voice_enabled": True,
        },
    }
    assert "discord-super-secret-token" not in response.text

    saved_text = config_path.read_text(encoding="utf-8")
    assert "discord-super-secret-token" in saved_text




def test_discord_setup_skips_persist_when_config_factory_is_injected() -> None:
    """測試模式下若使用 config_factory，setup 不應假裝已持久化。"""
    config = MochiConfig.model_validate({})
    app = _create_test_app(config=config)

    with TestClient(app) as client:
        response = client.post(
            "/v1/setup/discord",
            json={
                "bot_token": "discord-inline-secret",
                "enabled": True,
                "voice_enabled": False,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["channels"]["discord"]["bot_token_configured"] is True
    assert payload["update"]["persisted"] is False
    assert payload["update"]["config_path"] is None
    assert "discord-inline-secret" not in response.text




def test_discord_setup_rejects_missing_initial_token() -> None:
    """首次 setup 若沒有既有 token，應清楚拒絕。"""
    config = MochiConfig.model_validate({})
    app = _create_test_app(config=config)

    with TestClient(app) as client:
        response = client.post(
            "/v1/setup/discord",
            json={
                "enabled": True,
                "text_enabled": True,
            },
        )

    assert response.status_code == 400
    assert response.json() == {"detail": "Discord bot token is required for initial setup."}




def test_discord_setup_allows_followup_updates_without_resending_token() -> None:
    """若 config 已有 token，後續可只更新 Discord 非敏感欄位。"""
    config = MochiConfig.model_validate(
        {
            "channels": {
                "discord": {
                    "enabled": True,
                    "bot_token": SecretStr("existing-discord-token"),
                    "message_mode": "mentions_only",
                }
            }
        }
    )
    app = _create_test_app(config=config)

    with TestClient(app) as client:
        response = client.post(
            "/v1/setup/discord",
            json={
                "voice_enabled": False,
                "message_mode": "slash_only",
                "allowed_channel_ids": [999],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["channels"]["discord"]["bot_token_configured"] is True
    assert payload["channels"]["discord"]["voice_enabled"] is False
    assert payload["channels"]["discord"]["message_mode"] == "slash_only"
    assert payload["channels"]["discord"]["allowed_channel_ids"] == [999]
    assert "existing-discord-token" not in response.text
