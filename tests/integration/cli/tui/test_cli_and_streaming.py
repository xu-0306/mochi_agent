"""Chat TUI CLI tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from mochi.agents.events import FinalAnswerEvent, TextChunkEvent
from mochi.main import DEFAULT_TUI_MAX_TURNS, DEFAULT_TUI_SESSION_ID, app

runner = CliRunner()

def test_root_without_args_enters_tui(monkeypatch) -> None:
    """`mochi` without args should enter the TUI."""
    called: dict[str, object] = {}

    async def fake_chat_tui_async(
        *,
        model: str | None,
        config_path: str | None,
        session_id: str,
        max_turns: int,
    ) -> None:
        called["model"] = model
        called["config_path"] = config_path
        called["session_id"] = session_id
        called["max_turns"] = max_turns

    monkeypatch.setattr("mochi.main._chat_tui_async", fake_chat_tui_async)
    result = runner.invoke(app, [])

    assert result.exit_code == 0
    assert called == {
        "model": None,
        "config_path": None,
        "session_id": DEFAULT_TUI_SESSION_ID,
        "max_turns": DEFAULT_TUI_MAX_TURNS,
    }



def test_tui_command_calls_async_helper(monkeypatch) -> None:
    """`mochi tui` should call the async helper."""
    called: dict[str, object] = {}

    async def fake_chat_tui_async(
        *,
        model: str | None,
        config_path: str | None,
        session_id: str,
        max_turns: int,
    ) -> None:
        called["model"] = model
        called["config_path"] = config_path
        called["session_id"] = session_id
        called["max_turns"] = max_turns

    monkeypatch.setattr("mochi.main._chat_tui_async", fake_chat_tui_async)
    result = runner.invoke(
        app,
        [
            "tui",
            "--model",
            "ollama:qwen2.5",
            "--config",
            "cfg.yaml",
            "--session-id",
            "s-tui",
            "--max-turns",
            "7",
        ],
    )

    assert result.exit_code == 0
    assert called == {
        "model": "ollama:qwen2.5",
        "config_path": "cfg.yaml",
        "session_id": "s-tui",
        "max_turns": 7,
    }



def test_chat_command_calls_terminal_async_helper(monkeypatch) -> None:
    called: dict[str, object] = {}

    async def fake_chat_async_terminal(
        text: str,
        model: str | None,
        config_path: str | None,
        session_id: str,
    ) -> None:
        called["text"] = text
        called["model"] = model
        called["config_path"] = config_path
        called["session_id"] = session_id

    monkeypatch.setattr("mochi.main._chat_async_terminal", fake_chat_async_terminal)
    result = runner.invoke(
        app,
        [
            "chat",
            "--model",
            "ollama:qwen3",
            "--config",
            "cfg.yaml",
            "--session-id",
            "cli-s1",
            "hello from cli",
        ],
    )

    assert result.exit_code == 0
    assert called == {
        "text": "hello from cli",
        "model": "ollama:qwen3",
        "config_path": "cfg.yaml",
        "session_id": "cli-s1",
    }



@pytest.mark.asyncio
async def test_chat_tui_async_streaming_and_session_switch(monkeypatch, capsys) -> None:
    """TUI should support slash commands, session switching, and streaming output."""
    from mochi.main import _chat_tui_async

    class _FakeEngine:
        def __init__(self, config) -> None:  # noqa: ANN001
            self.config = config
            self.calls: list[tuple[str, str | None]] = []
            self.closed = False

        async def initialize(self) -> None:
            return None

        async def switch_model(self, model_spec: str) -> SimpleNamespace:
            self.config.model = model_spec
            return SimpleNamespace(name=model_spec, backend_type="ollama")

        async def chat(
            self,
            message: str,
            session_id: str | None = None,
        ) -> AsyncIterator[object]:
            self.calls.append((message, session_id))
            yield TextChunkEvent(content="hello ")
            yield TextChunkEvent(content="world")

        async def close(self) -> None:
            self.closed = True

    fake_engine_ref: dict[str, _FakeEngine] = {}
    inputs = iter(
        [
            "/help",
            "/session s2",
            "/session",
            "/model",
            "/model ollama:new",
            "hi mochi",
            "/exit",
        ]
    )

    def fake_load_config(config_path=None):  # noqa: ARG001
        return SimpleNamespace(model="ollama:base")

    def fake_engine_factory(config) -> _FakeEngine:  # noqa: ANN001
        engine = _FakeEngine(config)
        fake_engine_ref["engine"] = engine
        return engine

    monkeypatch.setattr("mochi.config.manager.load_config", fake_load_config)
    monkeypatch.setattr("mochi.agents.engine.AgentEngine", fake_engine_factory)
    monkeypatch.setattr("mochi.main.console.input", lambda prompt="": next(inputs))  # noqa: ARG005

    await _chat_tui_async(
        model=None,
        config_path=None,
        session_id="s1",
        max_turns=3,
    )
    captured = capsys.readouterr().out

    assert "Slash Commands" in captured
    assert "Current session: s2" in captured
    assert "Current model: ollama:base" in captured
    assert "Model switched:" in captured
    assert "hello world" in captured
    assert fake_engine_ref["engine"].calls == [("hi mochi", "s2")]
    assert fake_engine_ref["engine"].closed is True



@pytest.mark.asyncio
async def test_chat_tui_async_rejects_non_positive_max_turns() -> None:
    """max_turns <= 0 should exit with an error."""
    from mochi.main import _chat_tui_async

    with pytest.raises(SystemExit) as exc_info:
        await _chat_tui_async(
            model=None,
            config_path=None,
            session_id="s1",
            max_turns=0,
        )

    assert exc_info.value.code == 1



@pytest.mark.asyncio
async def test_chat_tui_async_supports_final_answer_event_fallback(monkeypatch, capsys) -> None:
    """FinalAnswerEvent should still render in the TUI fallback path."""
    from mochi.main import _chat_tui_async

    class _FakeEngine:
        def __init__(self, config) -> None:  # noqa: ANN001, ARG002
            self.closed = False

        async def initialize(self) -> None:
            return None

        async def switch_model(self, model_spec: str) -> SimpleNamespace:  # noqa: ARG002
            return SimpleNamespace(name="x", backend_type="test")

        async def chat(
            self,
            message: str,
            session_id: str | None = None,
        ) -> AsyncIterator[object]:  # noqa: ARG002
            yield FinalAnswerEvent(content="fallback answer")

        async def close(self) -> None:
            self.closed = True

    inputs = iter(["hi", "/exit"])

    def fake_load_config(config_path=None):  # noqa: ARG001
        return SimpleNamespace(model="ollama:base")

    fake_engine = _FakeEngine(None)
    monkeypatch.setattr("mochi.config.manager.load_config", fake_load_config)
    monkeypatch.setattr("mochi.agents.engine.AgentEngine", lambda config: fake_engine)  # noqa: ARG005
    monkeypatch.setattr("mochi.main.console.input", lambda prompt="": next(inputs))  # noqa: ARG005

    await _chat_tui_async(
        model=None,
        config_path=None,
        session_id="s1",
        max_turns=2,
    )
    captured = capsys.readouterr().out

    assert "fallback answer" in captured
    assert fake_engine.closed is True
