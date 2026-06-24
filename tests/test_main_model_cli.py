"""模型管理 CLI 測試。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from mochi.main import app

runner = CliRunner()


async def _fake_doctor_inspect(
    *,
    backend_type: str,
    is_ready: bool,
    error: str | None,
):
    return (
        SimpleNamespace(
            name="configured-model",
            backend_type=backend_type,
            context_length=4096,
            supports_tool_calling=backend_type != "ollama",
            metadata={},
        ),
        is_ready,
        error,
    )


async def _fake_doctor_unresolved_inspect(error: str):
    return None, False, error


def test_model_list_shows_current_config(monkeypatch) -> None:
    """model list 應顯示目前設定模型與支援格式。"""
    def fake_load_config(config_path=None):
        return SimpleNamespace(model="ollama:qwen2.5")

    monkeypatch.setattr("mochi.config.manager.load_config", fake_load_config)
    result = runner.invoke(app, ["model", "list"])

    assert result.exit_code == 0
    assert "ollama:qwen2.5" in result.stdout
    assert "/path/to/model.gguf" in result.stdout


def test_model_info_command_uses_async_helper(monkeypatch) -> None:
    """model info 應呼叫對應 helper。"""
    called = {"config_path": None}

    async def fake_info(config_path: str | None) -> None:
        called["config_path"] = config_path

    monkeypatch.setattr("mochi.main._model_info_async", fake_info)
    result = runner.invoke(app, ["model", "info"])

    assert result.exit_code == 0
    assert called["config_path"] is None


def test_model_switch_command_uses_async_helper(monkeypatch) -> None:
    """model switch 應呼叫對應 helper。"""
    called = {"model_spec": None, "config_path": None}

    async def fake_switch(model_spec: str, config_path: str | None) -> None:
        called["model_spec"] = model_spec
        called["config_path"] = config_path

    monkeypatch.setattr("mochi.main._model_switch_async", fake_switch)
    result = runner.invoke(app, ["model", "switch", "ollama:qwen2.5"])

    assert result.exit_code == 0
    assert called["model_spec"] == "ollama:qwen2.5"
    assert called["config_path"] is None


def test_model_switch_async_reports_switch_failure(monkeypatch) -> None:
    """切換失敗時 CLI 應輸出錯誤並以 exit code 1 結束。"""
    class _FakeEngine:
        def __init__(self, config) -> None:  # noqa: D401, ANN001
            self.config = config

        async def initialize(self) -> None:
            return None

        async def switch_model(self, model_spec: str) -> None:  # noqa: ARG002
            raise RuntimeError("Backend switch rejected unhealthy backend for '/models/bad.gguf'.")

        async def close(self) -> None:
            return None

    def fake_load_config(config_path=None):
        return SimpleNamespace(model="ollama:old")

    monkeypatch.setattr("mochi.config.manager.load_config", fake_load_config)
    monkeypatch.setattr("mochi.agents.engine.AgentEngine", _FakeEngine)

    result = runner.invoke(app, ["model", "switch", "/models/bad.gguf"])

    assert result.exit_code == 1
    assert "Model switch failed" in result.stdout
    assert "rejected unhealthy backend" in result.stdout


def test_model_info_reports_configured_model_available_state(monkeypatch) -> None:
    """model info 應回報 configured model 可用狀態。"""
    def fake_load_config(config_path=None):
        return SimpleNamespace(
            model="ollama:qwen2.5",
            ollama=SimpleNamespace(base_url="http://localhost:11434"),
        )

    async def fake_inspect(model_spec: str, ollama_base_url: str):
        return (
            SimpleNamespace(
                name="ollama:qwen2.5",
                backend_type="ollama",
                context_length=32768,
                supports_tool_calling=True,
                metadata={},
            ),
            True,
            None,
        )

    monkeypatch.setattr("mochi.config.manager.load_config", fake_load_config)
    monkeypatch.setattr("mochi.main._inspect_configured_model", fake_inspect)

    result = runner.invoke(app, ["model", "info"])

    assert result.exit_code == 0
    assert "Current Model Information" in result.stdout
    assert "configured: " in result.stdout
    assert "ollama:qwen2.5" in result.stdout
    assert "backend: ollama" in result.stdout
    assert "ready" in result.stdout


def test_model_info_reports_configured_model_unavailable_state(monkeypatch) -> None:
    """configured backend 不可用時，model info 應回報 unavailable。"""
    def fake_load_config(config_path=None):
        return SimpleNamespace(
            model="ollama:qwen2.5",
            ollama=SimpleNamespace(base_url="http://localhost:11434"),
        )

    async def fake_inspect(model_spec: str, ollama_base_url: str):
        return (
            SimpleNamespace(
                name="ollama:qwen2.5",
                backend_type="ollama",
                context_length=32768,
                supports_tool_calling=True,
                metadata={},
            ),
            False,
            "Configured backend unavailable.",
        )

    monkeypatch.setattr("mochi.config.manager.load_config", fake_load_config)
    monkeypatch.setattr("mochi.main._inspect_configured_model", fake_inspect)

    result = runner.invoke(app, ["model", "info"])

    assert result.exit_code == 0
    assert "configured: " in result.stdout
    assert "unavailable" in result.stdout.lower()
    assert "Configured backend unavailable." in result.stdout


def test_doctor_reports_configured_model_status_when_backend_unavailable(monkeypatch) -> None:
    """doctor 應在 configured backend 不可用時回報模型診斷狀態。"""
    class _FakeOllamaBackend:
        def __init__(self, model: str, base_url: str) -> None:
            self.model = model
            self.base_url = base_url

        async def health_check(self) -> bool:
            return False

        async def close(self) -> None:
            return None

    def fake_load_config(config_path=None):
        return SimpleNamespace(
            model="ollama:qwen2.5",
            ollama=SimpleNamespace(base_url="http://localhost:11434"),
        )

    monkeypatch.setattr("mochi.config.manager.load_config", fake_load_config)
    monkeypatch.setattr("mochi.backends.ollama.OllamaBackend", _FakeOllamaBackend)
    monkeypatch.setattr(
        "mochi.main._inspect_configured_model",
        lambda model_spec, ollama_base_url: _fake_doctor_inspect(  # noqa: ARG005
            backend_type="ollama",
            is_ready=False,
            error="ollama backend is not reachable",
        ),
    )
    monkeypatch.setattr("os.path.exists", lambda _: True)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "ollama:qwen2.5" in result.stdout
    assert any(token in result.stdout.lower() for token in ("unavailable", "not available"))


def test_doctor_reports_configured_model_status_when_backend_unresolvable(monkeypatch) -> None:
    """doctor 應在 configured backend 無法解析時回報模型診斷狀態。"""
    class _FakeOllamaBackend:
        def __init__(self, model: str, base_url: str) -> None:
            self.model = model
            self.base_url = base_url

        async def health_check(self) -> bool:
            raise RuntimeError("Cannot resolve configured backend host.")

        async def close(self) -> None:
            return None

    def fake_load_config(config_path=None):
        return SimpleNamespace(
            model="ollama:qwen2.5",
            ollama=SimpleNamespace(base_url="http://unresolvable-host:11434"),
        )

    monkeypatch.setattr("mochi.config.manager.load_config", fake_load_config)
    monkeypatch.setattr("mochi.backends.ollama.OllamaBackend", _FakeOllamaBackend)
    monkeypatch.setattr(
        "mochi.main._inspect_configured_model",
        lambda model_spec, ollama_base_url: _fake_doctor_unresolved_inspect(  # noqa: ARG005
            "Cannot resolve configured backend host."
        ),
    )
    monkeypatch.setattr("os.path.exists", lambda _: True)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "ollama:qwen2.5" in result.stdout
    assert any(token in result.stdout.lower() for token in ("unresolvable", "cannot resolve"))


@pytest.mark.asyncio
async def test_model_info_async_shows_unavailable_configured_model(monkeypatch, capsys) -> None:
    """model info 應清楚標示 configured model 不可用。"""
    def fake_load_config(config_path=None):
        return SimpleNamespace(
            model="/models/demo.gguf",
            ollama=SimpleNamespace(base_url="http://localhost:11434"),
        )

    async def fake_inspect(model_spec: str, ollama_base_url: str):
        return (
            SimpleNamespace(
                name="/models/demo.gguf",
                backend_type="gguf",
                context_length=4096,
                supports_tool_calling=False,
                metadata={"dependency_ready": False, "model_path": "/models/demo.gguf"},
            ),
            False,
            None,
        )

    monkeypatch.setattr("mochi.config.manager.load_config", fake_load_config)
    monkeypatch.setattr("mochi.main._inspect_configured_model", fake_inspect)

    from mochi.main import _model_info_async

    await _model_info_async(None)
    output = capsys.readouterr().out

    assert "configured: " in output
    assert "/models/demo.gguf" in output
    assert "status: " in output
    assert "unavailable" in output
    assert "runtime dependency is not ready" in output
    assert "dependency_ready: False" in output


def test_model_info_command_reports_unresolved_model(monkeypatch) -> None:
    """model info 對無法解析的 configured model 應輸出 unresolved 狀態。"""
    def fake_load_config(config_path=None):
        return SimpleNamespace(
            model="bad-spec",
            ollama=SimpleNamespace(base_url="http://localhost:11434"),
        )

    async def fake_inspect(model_spec: str, ollama_base_url: str):
        return None, False, "Cannot resolve model_spec 'bad-spec'."

    monkeypatch.setattr("mochi.config.manager.load_config", fake_load_config)
    monkeypatch.setattr("mochi.main._inspect_configured_model", fake_inspect)

    result = runner.invoke(app, ["model", "info"])

    assert result.exit_code == 0
    assert "configured: " in result.stdout
    assert "bad-spec" in result.stdout
    assert "unresolved" in result.stdout
    assert "Cannot resolve model_spec" in result.stdout


def test_doctor_reports_configured_model_diagnostic_status(monkeypatch) -> None:
    """doctor 應顯示 configured model 的可用性診斷。"""
    class _FakeOllamaBackend:
        def __init__(self, model: str, base_url: str) -> None:  # noqa: D401, ARG002
            self.base_url = base_url

        async def health_check(self) -> bool:
            return True

        async def close(self) -> None:
            return None

    def fake_load_config(config_path=None):
        return SimpleNamespace(
            model="http://api.example.com/v1",
            ollama=SimpleNamespace(base_url="http://localhost:11434"),
        )

    async def fake_inspect(model_spec: str, ollama_base_url: str):
        return (
            SimpleNamespace(
                name="auto",
                backend_type="openai_compat",
                context_length=4096,
                supports_tool_calling=True,
                metadata={},
            ),
            False,
            "connection refused",
        )

    monkeypatch.setattr("mochi.config.manager.load_config", fake_load_config)
    monkeypatch.setattr("mochi.backends.ollama.OllamaBackend", _FakeOllamaBackend)
    monkeypatch.setattr("mochi.main._inspect_configured_model", fake_inspect)
    monkeypatch.setattr("os.path.exists", lambda path: True)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "Configured model http://api.example.com/v1 (openai_compat)" in result.stdout
    assert "unavailable" in result.stdout
    assert "connection refused" in result.stdout
