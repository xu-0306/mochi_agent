"""Chat TUI CLI tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from mochi.agents.events import FinalAnswerEvent, TextChunkEvent, ToolCallResultEvent
from mochi.config.schema import SecurityConfig
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


@pytest.mark.asyncio
async def test_chat_tui_async_tools_commands_show_and_update_web_search_settings(
    monkeypatch,
    capsys,
    tmp_path,
) -> None:
    """`/tools` commands should expose and persist web search settings from the TUI."""
    from mochi.main import _chat_tui_async

    class _FakeEngine:
        def __init__(self, config) -> None:  # noqa: ANN001
            self.config = config
            self.closed = False

        async def initialize(self) -> None:
            return None

        async def close(self) -> None:
            self.closed = True

    fake_config = SimpleNamespace(
        model="ollama:base",
        sessions_dir=str(tmp_path / "sessions"),
        tools=SimpleNamespace(
            web_search_engine="duckduckgo",
            web_search_fallback_engines=["duckduckgo_html"],
            web_fetch_extractor="trafilatura",
            web_search_tavily_api_key=None,
            web_search_serper_api_key=None,
            web_search_jina_api_key=None,
            web_search_exa_api_key=None,
            web_search_brave_api_key=None,
            web_fetch_jina_api_key=None,
        ),
    )
    saved = {"path": None, "engine": None, "fallback": None, "extractor": None}
    inputs = iter(
        [
            "/tools",
            "/tools search-engine tavily",
            "/tools fallback brave duckduckgo_html",
            "/tools fetch-extractor jina_reader",
            "/tools",
            "/exit",
        ]
    )

    monkeypatch.setattr("mochi.config.manager.load_config", lambda config_path=None: fake_config)  # noqa: ARG005
    monkeypatch.setattr("mochi.agents.engine.AgentEngine", _FakeEngine)
    monkeypatch.setattr("mochi.main.console.input", lambda prompt="": next(inputs))  # noqa: ARG005

    def fake_save_config(config, config_path=None):  # noqa: ANN001, ARG001
        saved["path"] = config_path
        saved["engine"] = config.tools.web_search_engine
        saved["fallback"] = list(config.tools.web_search_fallback_engines)
        saved["extractor"] = config.tools.web_fetch_extractor
        return tmp_path / "saved-config.yaml"

    monkeypatch.setattr("mochi.config.manager.save_config", fake_save_config)

    await _chat_tui_async(
        model=None,
        config_path="config.yaml",
        session_id="s1",
        max_turns=2,
    )
    captured = capsys.readouterr().out

    assert "Web search engine: duckduckgo" in captured
    assert "Web search engine updated: tavily" in captured
    assert "Web search fallback updated: brave, duckduckgo_html" in captured
    assert "Web fetch extractor updated: jina_reader" in captured
    assert "Web search engine: tavily" in captured
    assert saved == {
        "path": "config.yaml",
        "engine": "tavily",
        "fallback": ["brave", "duckduckgo_html"],
        "extractor": "jina_reader",
    }


@pytest.mark.asyncio
async def test_chat_tui_async_tools_key_commands_mask_and_clear_secrets(
    monkeypatch,
    capsys,
    tmp_path,
) -> None:
    """`/tools key*` commands should manage secrets without echoing them."""
    from mochi.main import _chat_tui_async

    class _FakeEngine:
        def __init__(self, config) -> None:  # noqa: ANN001
            self.config = config
            self.closed = False

        async def initialize(self) -> None:
            return None

        async def close(self) -> None:
            self.closed = True

    fake_config = SimpleNamespace(
        model="ollama:base",
        sessions_dir=str(tmp_path / "sessions"),
        tools=SimpleNamespace(
            web_search_engine="jina",
            web_search_fallback_engines=["duckduckgo_html"],
            web_fetch_extractor="trafilatura",
            web_search_tavily_api_key=None,
            web_search_serper_api_key=None,
            web_search_jina_api_key=None,
            web_search_exa_api_key=None,
            web_search_brave_api_key=None,
            web_fetch_jina_api_key=None,
        ),
    )
    saved_keys: list[tuple[object, object]] = []
    inputs = iter(
        [
            "/tools key-status",
            "/tools key tavily",
            "tvly-secret-value",
            "/tools key-status",
            "/tools key-clear tavily",
            "/tools key-status",
            "/exit",
        ]
    )

    monkeypatch.setattr("mochi.config.manager.load_config", lambda config_path=None: fake_config)  # noqa: ARG005
    monkeypatch.setattr("mochi.agents.engine.AgentEngine", _FakeEngine)

    def fake_console_input(prompt="", **kwargs):  # noqa: ANN001, ARG001
        return next(inputs)

    monkeypatch.setattr("mochi.main.console.input", fake_console_input)

    def fake_save_config(config, config_path=None):  # noqa: ANN001, ARG001
        tools = config.tools
        saved_keys.append(
            (
                getattr(tools, "web_search_tavily_api_key", None),
                getattr(tools, "web_fetch_jina_api_key", None),
            )
        )
        return tmp_path / "saved-config.yaml"

    monkeypatch.setattr("mochi.config.manager.save_config", fake_save_config)

    await _chat_tui_async(
        model=None,
        config_path="config.yaml",
        session_id="s1",
        max_turns=2,
    )
    captured = capsys.readouterr().out

    assert "tavily: not configured" in captured
    assert "tavily: configured" in captured
    assert "Cleared key for tavily" in captured
    assert "tvly-secret-value" not in captured
    assert len(saved_keys) == 2


@pytest.mark.asyncio
async def test_chat_tui_async_supports_approval_and_safety_commands(
    monkeypatch,
    capsys,
    tmp_path,
) -> None:
    """`/approvals`, approval actions, and `/safety` should use the shared runtime/config flow."""
    from mochi.main import _chat_tui_async

    class _FakeEngine:
        def __init__(self, config) -> None:  # noqa: ANN001
            self.config = config
            self.closed = False

        async def initialize(self) -> None:
            return None

        async def close(self) -> None:
            self.closed = True

    class _FakeRuntimeService:
        created: list["_FakeRuntimeService"] = []

        def __init__(self, *, engine, store) -> None:  # noqa: ANN001, ARG002
            self.engine = engine
            self.security = None
            self.bound_config = None
            self.closed = False
            self.__class__.created.append(self)

        def update_security_config(self, security) -> None:  # noqa: ANN001
            self.security = security

        def bind_app_config(self, *, config, config_path) -> None:  # noqa: ANN001, ARG002
            self.bound_config = config

        def set_runtime_tasks_root(self, root_dir) -> None:  # noqa: ANN001
            self.root_dir = root_dir

        async def start(self) -> None:
            return None

        async def close(self) -> None:
            self.closed = True

        async def list_approvals(self) -> list[dict[str, object]]:
            return [
                {
                    "approval_id": "ap-1",
                    "status": "pending",
                    "approval_kind": "exec",
                    "approval_scope": "workspace",
                    "command": "echo hi",
                    "policy_reason": "Needs approval.",
                    "exec_session_id": "sess-1",
                    "exec_status": "pending",
                },
                {
                    "approval_id": "ap-2",
                    "status": "pending",
                    "approval_kind": "apply_patch",
                    "approval_scope": "workspace",
                    "tool_name": "apply_patch",
                    "file_changes": [
                        {
                            "relative_path": "notes.py",
                            "added_lines": 3,
                            "deleted_lines": 1,
                        }
                    ],
                },
            ]

        async def resolve_approval(self, approval_id: str, *, decision: str) -> dict[str, object] | None:
            if approval_id == "missing":
                return None
            status_map = {
                "approve_once": "approved_once",
                "approve_and_save_rule": "approved_and_saved_rule",
                "reject": "rejected",
            }
            payload: dict[str, object] = {
                "approval_id": approval_id,
                "status": status_map[decision],
            }
            if decision != "reject":
                payload["execution_result"] = {"session_id": f"session-for-{approval_id}"}
            return payload

    fake_config = SimpleNamespace(
        model="ollama:base",
        sessions_dir=str(tmp_path / "sessions"),
        security=SecurityConfig(),
        tools=SimpleNamespace(
            web_search_engine="duckduckgo",
            web_search_fallback_engines=["duckduckgo_html"],
            web_fetch_extractor="trafilatura",
            web_search_tavily_api_key=None,
            web_search_serper_api_key=None,
            web_search_jina_api_key=None,
            web_search_exa_api_key=None,
            web_search_brave_api_key=None,
            web_fetch_jina_api_key=None,
        ),
    )
    saved_modes: list[str] = []
    inputs = iter(
        [
            "/approvals",
            "/approve ap-1",
            "/approve-save ap-2",
            "/reject ap-3",
            "/safety",
            "/safety auto_review",
            "/safety",
            "/exit",
        ]
    )

    async def fake_runtime_service_factory(*, engine, config, config_path):  # noqa: ANN001, ARG001
        service = _FakeRuntimeService(engine=engine, store=None)
        service.update_security_config(config.security)
        service.bind_app_config(config=config, config_path=config_path)
        await service.start()
        return service

    monkeypatch.setattr("mochi.config.manager.load_config", lambda config_path=None: fake_config)  # noqa: ARG005
    monkeypatch.setattr("mochi.agents.engine.AgentEngine", _FakeEngine)
    monkeypatch.setattr("mochi.main._create_tui_runtime_service", fake_runtime_service_factory)
    monkeypatch.setattr("mochi.main.console.input", lambda prompt="": next(inputs))  # noqa: ARG005

    def fake_save_config(config, config_path=None):  # noqa: ANN001, ARG001
        saved_modes.append(config.security.autonomy_mode)
        return tmp_path / "saved-config.yaml"

    monkeypatch.setattr("mochi.config.manager.save_config", fake_save_config)

    await _chat_tui_async(
        model=None,
        config_path="config.yaml",
        session_id="s1",
        max_turns=2,
    )
    captured = capsys.readouterr().out

    assert "ap-1 [pending] exec/workspace cmd=echo hi" in captured
    assert "ap-2 [pending] apply_patch/workspace tool=apply_patch" in captured
    assert "file changes=1" in captured
    assert "notes.py (+3/-1)" in captured
    assert "Approval updated: ap-1 -> approved_once" in captured
    assert "Approval updated: ap-2 -> approved_and_saved_rule" in captured
    assert "Approval updated: ap-3 -> rejected" in captured
    assert "Safety mode: trusted_workspace" in captured
    assert "Safety mode updated: auto_review" in captured
    assert "Safety mode: auto_review" in captured
    assert saved_modes == ["auto_review"]
    assert len(_FakeRuntimeService.created) >= 2


@pytest.mark.asyncio
async def test_chat_tui_async_supports_exec_session_commands(monkeypatch, capsys, tmp_path) -> None:
    """`/exec-read` and `/exec-stop` should surface approval-bound exec session output."""
    from mochi.main import _chat_tui_async

    class _FakeEngine:
        def __init__(self, config) -> None:  # noqa: ANN001, ARG002
            self.closed = False

        async def initialize(self) -> None:
            return None

        async def close(self) -> None:
            self.closed = True

    class _FakeRuntimeService:
        def __init__(self, *, engine, store) -> None:  # noqa: ANN001, ARG002
            self.closed = False

        def update_security_config(self, security) -> None:  # noqa: ANN001, ARG002
            return None

        def bind_app_config(self, *, config, config_path) -> None:  # noqa: ANN001, ARG002
            return None

        def set_runtime_tasks_root(self, root_dir) -> None:  # noqa: ANN001, ARG002
            return None

        async def start(self) -> None:
            return None

        async def close(self) -> None:
            self.closed = True

        async def get_approval_exec_session(self, approval_id: str) -> dict[str, object] | tuple[str, None]:
            if approval_id == "missing":
                return ("session_unavailable", None)
            return {
                "session": {
                    "session_id": "sess-7",
                    "status": "running",
                    "shell": "powershell",
                    "exit_code": None,
                    "stdout": "line one\nline two\n",
                    "stderr": "",
                }
            }

        async def stop_approval_exec_session(self, approval_id: str) -> dict[str, object] | tuple[str, None]:
            if approval_id == "missing":
                return ("session_unavailable", None)
            return {
                "stop_status": "killed",
                "session": {
                    "session_id": "sess-7",
                    "status": "killed",
                    "shell": "powershell",
                    "exit_code": 137,
                    "stdout": "line one\n",
                    "stderr": "stopped\n",
                },
            }

    fake_config = SimpleNamespace(
        model="ollama:base",
        sessions_dir=str(tmp_path / "sessions"),
        security=SecurityConfig(),
        tools=SimpleNamespace(
            web_search_engine="duckduckgo",
            web_search_fallback_engines=["duckduckgo_html"],
            web_fetch_extractor="trafilatura",
            web_search_tavily_api_key=None,
            web_search_serper_api_key=None,
            web_search_jina_api_key=None,
            web_search_exa_api_key=None,
            web_search_brave_api_key=None,
            web_fetch_jina_api_key=None,
        ),
    )
    inputs = iter(
        [
            "/exec-read ap-1",
            "/exec-stop ap-1",
            "/exec-read missing",
            "/exit",
        ]
    )

    async def fake_runtime_service_factory(*, engine, config, config_path):  # noqa: ANN001, ARG001
        service = _FakeRuntimeService(engine=engine, store=None)
        service.update_security_config(config.security)
        service.bind_app_config(config=config, config_path=config_path)
        await service.start()
        return service

    monkeypatch.setattr("mochi.config.manager.load_config", lambda config_path=None: fake_config)  # noqa: ARG005
    monkeypatch.setattr("mochi.agents.engine.AgentEngine", _FakeEngine)
    monkeypatch.setattr("mochi.main._create_tui_runtime_service", fake_runtime_service_factory)
    monkeypatch.setattr("mochi.main.console.input", lambda prompt="": next(inputs))  # noqa: ARG005

    await _chat_tui_async(
        model=None,
        config_path=None,
        session_id="s1",
        max_turns=2,
    )
    captured = capsys.readouterr().out

    assert "Exec session: sess-7 status=running shell=powershell" in captured
    assert "line one" in captured
    assert "Exec session stop requested: killed" in captured
    assert "Exit code: 137" in captured
    assert "stopped" in captured
    assert "session unavailable" in captured


@pytest.mark.asyncio
async def test_chat_tui_async_clear_resets_session_history(monkeypatch, capsys) -> None:
    """`/clear` should reset the session and recreate the engine."""
    from mochi.main import _chat_tui_async

    class _FakeEngine:
        def __init__(self, config) -> None:  # noqa: ANN001, ARG002
            self.closed = False
            self.calls: list[tuple[str, str | None]] = []

        async def initialize(self) -> None:
            return None

        async def switch_model(self, model_spec: str) -> SimpleNamespace:  # noqa: ARG002
            return SimpleNamespace(name="x", backend_type="test")

        async def chat(
            self,
            message: str,
            session_id: str | None = None,
        ) -> AsyncIterator[object]:
            self.calls.append((message, session_id))
            yield FinalAnswerEvent(content="after clear")

        async def close(self) -> None:
            self.closed = True

    class _FakeSessionStore:
        def __init__(self, sessions_dir) -> None:  # noqa: ANN001, ARG002
            self.deleted: list[str] = []
            self.events: dict[str, list[dict[str, object]]] = {}

        async def save_event(self, session_id: str, event: dict[str, object]) -> None:
            self.events.setdefault(session_id, []).append(dict(event))

        async def load_session(self, session_id: str) -> list[dict[str, object]]:
            return list(self.events.get(session_id, []))

        async def delete_session(self, session_id: str) -> bool:
            self.deleted.append(session_id)
            self.events.pop(session_id, None)
            return True

    inputs = iter(["/clear", "hi", "/exit"])
    engines: list[_FakeEngine] = []
    fake_store = _FakeSessionStore(None)

    def fake_load_config(config_path=None):  # noqa: ARG001
        return SimpleNamespace(model="ollama:base", sessions_dir="/tmp/mochi-sessions")

    def fake_engine_factory(config) -> _FakeEngine:  # noqa: ANN001
        engine = _FakeEngine(config)
        engines.append(engine)
        return engine

    monkeypatch.setattr("mochi.config.manager.load_config", fake_load_config)
    monkeypatch.setattr("mochi.agents.engine.AgentEngine", fake_engine_factory)
    monkeypatch.setattr("mochi.sessions.store.SessionStore", lambda sessions_dir: fake_store)  # noqa: ARG005
    monkeypatch.setattr("mochi.main.console.input", lambda prompt="": next(inputs))  # noqa: ARG005

    await _chat_tui_async(
        model=None,
        config_path=None,
        session_id="clear-me",
        max_turns=2,
    )
    captured = capsys.readouterr().out

    assert "Cleared session:" in captured
    assert fake_store.deleted == ["clear-me"]
    assert len(engines) == 2
    assert engines[0].closed is True
    assert engines[1].calls == [("hi", "clear-me")]


@pytest.mark.asyncio
async def test_chat_tui_async_prints_tool_errors(monkeypatch, capsys) -> None:
    """Tool failures should still be printed in the TUI."""
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
            yield ToolCallResultEvent(
                call_id="c1",
                tool_name="exec_command",
                result="",
                error="approval required",
                metadata={
                    "approval_id": "exec-approval-1",
                    "requires_approval": True,
                    "approval_kind": "exec",
                    "approval_scope": "workspace",
                },
            )
            yield FinalAnswerEvent(content="done")

        async def close(self) -> None:
            self.closed = True

    inputs = iter(["run tool", "/exit"])

    def fake_load_config(config_path=None):  # noqa: ARG001
        return SimpleNamespace(model="ollama:base", sessions_dir="/tmp/mochi-sessions")

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

    assert "Tool exec_command failed: approval required" in captured
    assert "Approval pending: id=exec-approval-1 kind=exec scope=workspace" in captured
    assert "/approve exec-approval-1" in captured
    assert "/approve-save exec-approval-1" in captured
    assert "/reject exec-approval-1" in captured
    assert "done" in captured


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_chat_tui_async_goal_lifecycle_commands_use_terminal_goal_flow(
    monkeypatch,
    capsys,
) -> None:
    from mochi.main import _chat_tui_async

    class _FakeEngine:
        def __init__(self, config) -> None:  # noqa: ANN001, ARG002
            self.calls: list[tuple[str, str | None]] = []
            self.closed = False
            self.invocations: list[object] = []

        async def initialize(self) -> None:
            return None

        async def chat(
            self,
            message: str,
            session_id: str | None = None,
        ) -> AsyncIterator[object]:
            self.calls.append((message, session_id))
            yield FinalAnswerEvent(content="unexpected")

        async def invoke(self, request):  # noqa: ANN001
            self.invocations.append(request)
            if str(getattr(request, "session_id", "")).startswith("goal-proposal-copy:"):
                return SimpleNamespace(
                    content=(
                        "I framed your request as a goal proposal that we can launch directly. "
                        "This scope fits a single-agent long-running run best."
                    )
                )
            assert '"user_follow_up": "start it"' in request.message
            return SimpleNamespace(
                content='{"intent":"confirm_start","confidence":0.93,"rationale":"The user clearly wants to launch the pending goal now."}'
            )

        async def close(self) -> None:
            self.closed = True

    class _FakeSessionStore:
        def __init__(self, sessions_dir) -> None:  # noqa: ANN001, ARG002
            self.by_session: dict[str, list[dict[str, object]]] = {}

        async def save_event(self, session_id: str, event: dict[str, object]) -> None:
            self.by_session.setdefault(session_id, []).append(dict(event))

        async def load_session(self, session_id: str) -> list[dict[str, object]]:
            return list(self.by_session.get(session_id, []))

        async def delete_session(self, session_id: str) -> bool:
            self.by_session.pop(session_id, None)
            return True

    class _FakeRuntimeService:
        def __init__(self) -> None:
            self.goal: dict[str, object] | None = None

        async def create_goal(self, payload) -> dict[str, object]:  # noqa: ANN001
            self.goal = {
                "goal_id": "goal-1",
                "objective": payload.objective,
                "execution_mode": payload.execution_mode,
                "protocol_id": payload.protocol_id,
                "status": "created",
                "attempts": [],
                "current_attempt_id": None,
                "project_id": payload.project_id,
                "workspace_dir": payload.workspace_dir,
                "latest_error": None,
            }
            return dict(self.goal)

        async def start_goal(self, goal_id: str) -> dict[str, object]:
            assert goal_id == "goal-1"
            assert self.goal is not None
            self.goal.update(
                {
                    "status": "running",
                    "current_attempt_id": "attempt-1",
                    "attempts": [
                        {
                            "attempt_id": "attempt-1",
                            "agent_run_id": "run-1",
                        }
                    ],
                }
            )
            return dict(self.goal)

        async def get_goal(self, goal_id: str) -> dict[str, object] | None:
            assert goal_id == "goal-1"
            return dict(self.goal) if self.goal is not None else None

        async def pause_goal(self, goal_id: str) -> dict[str, object]:
            assert goal_id == "goal-1"
            assert self.goal is not None
            self.goal["status"] = "paused"
            return dict(self.goal)

        async def resume_goal(self, goal_id: str) -> dict[str, object]:
            assert goal_id == "goal-1"
            assert self.goal is not None
            self.goal["status"] = "running"
            return dict(self.goal)

        async def cancel_goal(self, goal_id: str) -> dict[str, object]:
            assert goal_id == "goal-1"
            assert self.goal is not None
            self.goal["status"] = "cancelled"
            return dict(self.goal)

        async def close(self) -> None:
            return None

    fake_store = _FakeSessionStore(None)
    fake_runtime = _FakeRuntimeService()
    fake_engine = _FakeEngine(None)
    inputs = iter(
        [
            "/goal keep working on this for 30 minutes",
            "start it",
            "/goal status",
            "/goal pause",
            "/goal resume",
            "/goal stop",
            "/exit",
        ]
    )

    monkeypatch.setattr(
        "mochi.config.manager.load_config",
        lambda config_path=None: SimpleNamespace(  # noqa: ARG005
            model="ollama:base",
            sessions_dir="/tmp/mochi-sessions",
            security=SecurityConfig(),
        ),
    )
    monkeypatch.setattr("mochi.agents.engine.AgentEngine", lambda config: fake_engine)  # noqa: ARG005
    monkeypatch.setattr("mochi.sessions.store.SessionStore", lambda sessions_dir: fake_store)  # noqa: ARG005

    async def fake_runtime_service_factory(**kwargs):  # noqa: ANN003, ARG001
        return fake_runtime

    monkeypatch.setattr("mochi.main._create_tui_runtime_service", fake_runtime_service_factory)
    monkeypatch.setattr("mochi.main.console.input", lambda prompt="": next(inputs))  # noqa: ARG005

    await _chat_tui_async(
        model=None,
        config_path=None,
        session_id="goal-session",
        max_turns=2,
    )
    captured = capsys.readouterr().out

    assert "I framed your request as a goal proposal that we can launch directly." in captured
    assert "Next step:" in captured
    assert "Launch: Send a short confirmation when you want execution to begin." in captured
    assert "Goal started." in captured
    assert "Fetched the latest goal status." in captured
    assert "Paused the active goal." in captured
    assert "Resumed the active goal." in captured
    assert "Stopped the active goal." in captured
    assert fake_engine.calls == []
    assert len(fake_engine.invocations) == 1

    session_events = fake_store.by_session["goal-session"]
    workflow_updates = [
        event
        for event in session_events
        if event.get("type") == "session_meta" and event.get("event") == "workflow_state_updated"
    ]
    goal_updates = [
        event
        for event in session_events
        if event.get("type") == "session_meta" and event.get("event") == "goal_state_updated"
    ]
    assert workflow_updates[-1]["workflow"]["enabled"] is False
    assert workflow_updates[-1]["workflow"]["bound_run_id"] is None
    assert goal_updates[-1]["goal"]["active_goal_id"] is None
    assert goal_updates[-1]["goal"]["active_goal_status"] == "cancelled"
    initial_pending_proposal = goal_updates[0]["goal"]["pending_proposal"]
    assert initial_pending_proposal["assistant_explanation"] == (
        "I framed your request as a goal proposal that we can launch directly. "
        "This scope fits a single-agent long-running run best."
    )


@pytest.mark.asyncio
async def test_chat_tui_async_routes_active_goal_followups_and_chat_escape(
    monkeypatch,
    capsys,
) -> None:
    from mochi.main import _chat_tui_async

    class _FakeEngine:
        def __init__(self, config) -> None:  # noqa: ANN001, ARG002
            self.calls: list[tuple[str, str | None]] = []
            self.closed = False

        async def initialize(self) -> None:
            return None

        async def chat(
            self,
            message: str,
            session_id: str | None = None,
        ) -> AsyncIterator[object]:
            self.calls.append((message, session_id))
            yield FinalAnswerEvent(content="plain chat reply")

        async def close(self) -> None:
            self.closed = True

    class _FakeSessionStore:
        def __init__(self, sessions_dir) -> None:  # noqa: ANN001, ARG002
            self.by_session: dict[str, list[dict[str, object]]] = {
                "goal-session": [
                    {
                        "type": "session_meta",
                        "event": "goal_state_updated",
                        "session_id": "goal-session",
                        "goal": {
                            "active_goal_id": "goal-1",
                            "active_goal_status": "running",
                            "execution_mode": "single_agent",
                            "default_route": "goal",
                            "last_goal_summary": {
                                "goal_id": "goal-1",
                                "objective": "Ship the task",
                                "execution_mode": "single_agent",
                                "protocol_id": None,
                                "models": ["ollama:base"],
                                "role_summary": "Primary agent continues the task directly with the current chat tools.",
                                "runtime_mode": "Single-agent long-running execution",
                                "risk_note": None,
                                "status": "running",
                            },
                            "pending_proposal": None,
                        },
                        "timestamp": "2026-06-25T00:00:00+00:00",
                    }
                ]
            }

        async def save_event(self, session_id: str, event: dict[str, object]) -> None:
            self.by_session.setdefault(session_id, []).append(dict(event))

        async def load_session(self, session_id: str) -> list[dict[str, object]]:
            return list(self.by_session.get(session_id, []))

        async def delete_session(self, session_id: str) -> bool:
            self.by_session.pop(session_id, None)
            return True

    class _FakeRuntimeService:
        def __init__(self) -> None:
            self.appended_messages: list[str] = []

        async def get_goal_health(self, goal_id: str) -> dict[str, object]:
            assert goal_id == "goal-1"
            return {
                "goal_id": "goal-1",
                "status": "running",
                "current_attempt": {"agent_run_id": "run-1"},
                "linked_agent_run": {"run_id": "run-1", "status": "running"},
                "recommended_next_action": {
                    "action": "monitor",
                    "summary": "The goal can continue with more guidance.",
                },
            }

        async def get_goal(self, goal_id: str) -> dict[str, object]:
            assert goal_id == "goal-1"
            return {
                "goal_id": "goal-1",
                "objective": "Ship the task",
                "execution_mode": "single_agent",
                "protocol_id": None,
                "status": "running",
                "current_attempt_id": "attempt-1",
                "attempts": [{"attempt_id": "attempt-1", "agent_run_id": "run-1"}],
                "project_id": None,
                "workspace_dir": None,
                "latest_error": None,
            }

        async def append_agent_run_message(self, run_id: str, payload) -> dict[str, object]:  # noqa: ANN001
            assert run_id == "run-1"
            self.appended_messages.append(payload.content)
            return {"run_id": run_id}

        async def close(self) -> None:
            return None

    fake_store = _FakeSessionStore(None)
    fake_runtime = _FakeRuntimeService()
    fake_engine = _FakeEngine(None)
    inputs = iter(
        [
            "please focus on tests",
            "/chat what time is it",
            "/exit",
        ]
    )

    monkeypatch.setattr(
        "mochi.config.manager.load_config",
        lambda config_path=None: SimpleNamespace(  # noqa: ARG005
            model="ollama:base",
            sessions_dir="/tmp/mochi-sessions",
            security=SecurityConfig(),
        ),
    )
    monkeypatch.setattr("mochi.agents.engine.AgentEngine", lambda config: fake_engine)  # noqa: ARG005
    monkeypatch.setattr("mochi.sessions.store.SessionStore", lambda sessions_dir: fake_store)  # noqa: ARG005
    async def fake_runtime_service_factory(**kwargs):  # noqa: ANN003, ARG001
        return fake_runtime

    monkeypatch.setattr("mochi.main._create_tui_runtime_service", fake_runtime_service_factory)
    monkeypatch.setattr("mochi.main.console.input", lambda prompt="": next(inputs))  # noqa: ARG005

    await _chat_tui_async(
        model=None,
        config_path=None,
        session_id="goal-session",
        max_turns=2,
    )
    captured = capsys.readouterr().out

    assert "Forwarded your guidance to the active goal." in captured
    assert fake_runtime.appended_messages == ["please focus on tests"]
    assert fake_engine.calls == [("what time is it", "goal-session")]
    assert "plain chat reply" in captured


@pytest.mark.asyncio
async def test_chat_tui_async_shows_active_goal_summary_on_start_and_session_query(
    monkeypatch,
    capsys,
) -> None:
    from mochi.main import _chat_tui_async

    class _FakeEngine:
        def __init__(self, config) -> None:  # noqa: ANN001, ARG002
            self.closed = False

        async def initialize(self) -> None:
            return None

        async def close(self) -> None:
            self.closed = True

    class _FakeSessionStore:
        def __init__(self, sessions_dir) -> None:  # noqa: ANN001, ARG002
            self.by_session: dict[str, list[dict[str, object]]] = {
                "goal-session": [
                    {
                        "type": "session_meta",
                        "event": "goal_state_updated",
                        "session_id": "goal-session",
                        "goal": {
                            "active_goal_id": "goal-9",
                            "active_goal_status": "running",
                            "execution_mode": "single_agent",
                            "default_route": "goal",
                            "last_goal_summary": {
                                "goal_id": "goal-9",
                                "objective": "Finalize the release checklist",
                                "execution_mode": "single_agent",
                                "protocol_id": None,
                                "models": ["ollama:base"],
                                "role_summary": "Primary agent continues the task directly with the current chat tools.",
                                "runtime_mode": "Single-agent long-running execution",
                                "risk_note": None,
                                "status": "running",
                            },
                            "pending_proposal": None,
                        },
                        "timestamp": "2026-06-25T00:00:00+00:00",
                    }
                ]
            }

        async def save_event(self, session_id: str, event: dict[str, object]) -> None:
            self.by_session.setdefault(session_id, []).append(dict(event))

        async def load_session(self, session_id: str) -> list[dict[str, object]]:
            return list(self.by_session.get(session_id, []))

        async def delete_session(self, session_id: str) -> bool:
            self.by_session.pop(session_id, None)
            return True

    fake_store = _FakeSessionStore(None)
    inputs = iter(["/session", "/exit"])

    monkeypatch.setattr(
        "mochi.config.manager.load_config",
        lambda config_path=None: SimpleNamespace(  # noqa: ARG005
            model="ollama:base",
            sessions_dir="/tmp/mochi-sessions",
            security=SecurityConfig(),
        ),
    )
    monkeypatch.setattr("mochi.agents.engine.AgentEngine", lambda config: _FakeEngine(config))  # noqa: ARG005
    monkeypatch.setattr("mochi.sessions.store.SessionStore", lambda sessions_dir: fake_store)  # noqa: ARG005
    monkeypatch.setattr("mochi.main.console.input", lambda prompt="": next(inputs))  # noqa: ARG005

    await _chat_tui_async(
        model=None,
        config_path=None,
        session_id="goal-session",
        max_turns=2,
    )
    captured = capsys.readouterr().out

    assert captured.count("Active goal summary for this session.") >= 2
    assert "Active goal" in captured
    assert "Finalize the release checklist" in captured
