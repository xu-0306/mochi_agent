"""channels CLI 測試。"""

from __future__ import annotations

from typer.testing import CliRunner

from mochi.main import app

runner = CliRunner()


def test_channels_run_command_uses_async_helper(monkeypatch) -> None:
    """channels run 應呼叫 async helper。"""
    called: dict[str, object] = {}

    async def fake_channels_run_async(config_path: str | None) -> None:
        called["config_path"] = config_path

    monkeypatch.setattr("mochi.main._channels_run_async", fake_channels_run_async)
    result = runner.invoke(app, ["channels", "run", "--config", "cfg.yaml"])

    assert result.exit_code == 0
    assert called == {"config_path": "cfg.yaml"}


def test_channels_guide_discord_prints_setup_steps() -> None:
    """channels guide discord 應輸出 onboarding 指引。"""
    result = runner.invoke(app, ["channels", "guide", "discord"])

    assert result.exit_code == 0
    assert "Discord Setup Guide" in result.stdout
    assert "DISCORD_BOT_TOKEN" in result.stdout
    assert "uv sync --extra channels --active" in result.stdout
    assert "voice_enabled: true" in result.stdout


def test_channels_guide_unknown_platform_fails() -> None:
    """未知平台 guide 應清楚失敗。"""
    result = runner.invoke(app, ["channels", "guide", "slack"])

    assert result.exit_code == 1
    assert "Unsupported channel guide" in result.stdout


def test_channels_voice_settings_command_uses_async_helper(monkeypatch) -> None:
    called: dict[str, object] = {}

    async def fake_channels_voice_settings_async(
        *,
        config_path: str | None,
        tts_voice: str | None,
        session_mode: str | None,
        reply_model_mode: str | None,
        reply_model: str | None,
    ) -> None:
        called["config_path"] = config_path
        called["tts_voice"] = tts_voice
        called["session_mode"] = session_mode
        called["reply_model_mode"] = reply_model_mode
        called["reply_model"] = reply_model

    monkeypatch.setattr(
        "mochi.main._channels_voice_settings_async",
        fake_channels_voice_settings_async,
    )
    result = runner.invoke(
        app,
        [
            "channels",
            "voice-settings",
            "--config",
            "cfg.yaml",
            "--tts-voice",
            "en-US-JennyNeural",
            "--session-mode",
            "shared",
            "--reply-model-mode",
            "agent-default",
            "--reply-model",
            "ollama:qwen2.5",
        ],
    )

    assert result.exit_code == 0
    assert called == {
        "config_path": "cfg.yaml",
        "tts_voice": "en-US-JennyNeural",
        "session_mode": "shared",
        "reply_model_mode": "agent-default",
        "reply_model": "ollama:qwen2.5",
    }
