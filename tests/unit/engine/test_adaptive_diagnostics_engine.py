from __future__ import annotations

import asyncio

import pytest

from mochi.agents.adaptive_diagnostics import DIAGNOSTICS_EVENT
from mochi.agents.conversation_resolver import (
    BoundedConversationContext,
    ConversationResolver,
    IntentInterpretation,
)
from mochi.agents.engine import AgentEngine
from mochi.agents.events import ToolCallRequestEvent
from mochi.agents.invocation import AgentInvocationRequest
from mochi.backends.types import GenerationResult, ToolCall
from mochi.config.schema import MochiConfig
from tests.unit.engine._support import FakeBackend


def _config(tmp_path) -> MochiConfig:  # type: ignore[no-untyped-def]
    return MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db")},
        }
    )


class _OperationInterpreter:
    def __init__(self, *operations: str) -> None:
        self._operations = frozenset(operations)

    async def interpret(
        self,
        context: BoundedConversationContext,
    ) -> IntentInterpretation:
        return IntentInterpretation(
            current_speech_act="request_execution" if self._operations else "request_information",
            task_relation="standalone",
            objective=context.current_turn.content,
            operations=self._operations,  # type: ignore[arg-type]
            confidence=0.99,
        )


def _engine(tmp_path, *operations: str) -> AgentEngine:  # type: ignore[no-untyped-def]
    return AgentEngine(
        _config(tmp_path),
        conversation_resolver_factory=lambda backend: ConversationResolver(
            interpreter=_OperationInterpreter(*operations)
        ),
    )


def _backend_metadata() -> dict[str, object]:
    return {
        "effective_context_length": 32768,
        "effective_context_length_source": "test",
    }


class _ReplyBackend(FakeBackend):
    async def generate(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(messages)
        return GenerationResult(content="done", input_tokens=4, output_tokens=2)


class _FailingBackend(FakeBackend):
    async def generate(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("backend failed")


class _ToolThenBlockingBackend(FakeBackend):
    def __init__(self) -> None:
        super().__init__(metadata=_backend_metadata())
        self.second_call_started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def generate(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(messages)
        if len(self.calls) == 1:
            return GenerationResult(
                content="",
                input_tokens=5,
                output_tokens=1,
                tool_calls=[
                    ToolCall(
                        id="datetime-call",
                        name="get_current_time",
                        arguments={"timezone": "UTC"},
                    )
                ],
            )
        self.second_call_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        raise AssertionError("unreachable")


async def _diagnostics(engine: AgentEngine, session_id: str):
    return [
        event
        for event in await engine._session_store.load_session(session_id)  # noqa: SLF001
        if event.get("event") == DIAGNOSTICS_EVENT
    ]


@pytest.mark.asyncio
async def test_engine_persists_one_diagnostics_event_and_replay_is_idempotent(tmp_path) -> None:
    engine = _engine(tmp_path)
    request = AgentInvocationRequest(
        message="hello",
        session_id="diag-engine",
        turn_id="diag-turn",
        persist_session=True,
        backend_override=_ReplyBackend(metadata=_backend_metadata()),
    )
    await engine.invoke(request)
    await engine.invoke(request)

    events = await _diagnostics(engine, "diag-engine")
    assert len(events) == 1
    assert events[0]["turn_id"] == "diag-turn"
    assert events[0]["classification"] in {"simple", "complex", "unknown"}
    assert events[0]["counters"]["model_calls"] >= 1
    await engine.close()


@pytest.mark.asyncio
async def test_engine_persists_backend_error_diagnostics_but_not_nonpersistent_turn(tmp_path) -> None:
    engine = _engine(tmp_path)
    await engine.invoke(
        AgentInvocationRequest(
            message="fail",
            session_id="diag-error",
            turn_id="error-turn",
            persist_session=True,
            backend_override=_FailingBackend(metadata=_backend_metadata()),
        )
    )
    assert len(await _diagnostics(engine, "diag-error")) == 1

    await engine.invoke(
        AgentInvocationRequest(
            message="ephemeral",
            session_id="diag-ephemeral",
            turn_id="ephemeral-turn",
            persist_session=False,
            backend_override=_ReplyBackend(metadata=_backend_metadata()),
        )
    )
    assert await _diagnostics(engine, "diag-ephemeral") == []
    await engine.close()


@pytest.mark.asyncio
async def test_async_generator_close_finalizes_partial_turn_diagnostics_once(
    tmp_path,
) -> None:
    engine = _engine(tmp_path, "execution")
    backend = _ToolThenBlockingBackend()
    persist_calls: list[str] = []
    persist_contexts = []
    original_persist = engine._persist_adaptive_diagnostics  # noqa: SLF001

    async def _track_persist(**kwargs):  # type: ignore[no-untyped-def]
        persist_calls.append(str(kwargs["turn_id"]))
        persist_contexts.append(kwargs["tool_execution_context"])
        await original_persist(**kwargs)

    engine._persist_adaptive_diagnostics = _track_persist  # type: ignore[method-assign]  # noqa: SLF001
    stream = engine._run_chat(  # noqa: SLF001
        AgentInvocationRequest(
            message="get the current time in UTC",
            session_id="diag-stream-close",
            turn_id="diag-stream-close-turn",
            persist_session=True,
            backend_override=backend,
            execution_profile="chat",
            tool_mode="required",
            tool_names_override=["get_current_time"],
            tool_allowlist=["get_current_time"],
        )
    )

    for _ in range(10):
        event = await asyncio.wait_for(anext(stream), timeout=2)
        if isinstance(event, ToolCallRequestEvent):
            break
    else:
        raise AssertionError("tool execution boundary was not reached")
    await asyncio.wait_for(backend.second_call_started.wait(), timeout=2)
    await stream.aclose()

    await asyncio.wait_for(backend.cancelled.wait(), timeout=2)
    session_events = await engine._session_store.load_session(  # noqa: SLF001
        "diag-stream-close"
    )
    events = [
        event
        for event in session_events
        if event.get("event") == DIAGNOSTICS_EVENT
    ]
    assert persist_calls == ["diag-stream-close-turn"]
    assert [
        context.state.get("adaptive_diagnostics_persist_error")
        for context in persist_contexts
    ] == [None]
    assert len(events) == 1, session_events
    assert events[0]["turn_id"] == "diag-stream-close-turn"
    assert events[0]["counters"]["model_calls"] == 2
    assert events[0]["counters"]["tool_calls"] == 1
    await engine.close()
