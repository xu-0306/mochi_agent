"""AgentEngine Phase 2 整合測試。"""

from __future__ import annotations

import pytest

from mochi.agents.events import (
    AssistantTruncatedEvent,
    FinalAnswerEvent,
    StatusEvent,
)
from mochi.backends.types import (
    GenerationResult,
)
from mochi.tools.registry import ToolRegistry
from tests.unit.engine._support import (
    EchoTool,
    FakeBackend,
    FileWriteProbeTool,
)


@pytest.mark.asyncio
async def test_react_loop_rescues_final_text_tool_call_markup() -> None:
    from mochi.agents.events import ToolCallRequestEvent, ToolCallResultEvent
    from mochi.agents.react_loop import AsyncReActLoop

    class _RawToolMarkupBackend(FakeBackend):
        async def generate(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append(messages)
            self.generation_kwargs.append(dict(kwargs))
            if len(self.calls) == 1:
                return GenerationResult(
                    content=(
                        'I need to use a tool. '
                        '<tool_call>{"name":"echo_tool","arguments":{"value":"rescued"}}</tool_call>'
                    ),
                    finish_reason="stop",
                )
            assert messages[-1].role == "tool"
            assert messages[-1].name == "echo_tool"
            return GenerationResult(content="tool output consumed", finish_reason="stop")

    registry = ToolRegistry(discover_builtin=False)
    registry.register(EchoTool())
    loop = AsyncReActLoop(backend=_RawToolMarkupBackend(), tool_registry=registry, max_iterations=3)

    events = [event async for event in loop.run("system", [], "use a tool")]

    statuses = [
        event
        for event in events
        if isinstance(event, StatusEvent)
        and event.metadata.get("reason") == "final_text_tool_call_rescue"
    ]
    requests = [event for event in events if isinstance(event, ToolCallRequestEvent)]
    results = [event for event in events if isinstance(event, ToolCallResultEvent)]
    finals = [event for event in events if isinstance(event, FinalAnswerEvent)]
    assert len(statuses) == 1
    assert statuses[0].metadata["tool_names"] == ["echo_tool"]
    assert len(requests) == 1
    assert requests[0].tool_name == "echo_tool"
    assert requests[0].arguments == {"value": "rescued"}
    assert len(results) == 1
    assert results[0].result == {"echo": "rescued"}
    assert len(finals) == 1
    assert finals[0].content == "tool output consumed"


@pytest.mark.asyncio
async def test_react_loop_rescues_thinking_tool_call_markup_without_leaking_it() -> None:
    from mochi.agents.events import ThinkingEvent, ToolCallRequestEvent, ToolCallResultEvent
    from mochi.agents.react_loop import AsyncReActLoop

    class _ThinkingToolMarkupBackend(FakeBackend):
        async def generate(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append(messages)
            self.generation_kwargs.append(dict(kwargs))
            if len(self.calls) == 1:
                return GenerationResult(
                    content="I should search before answering.",
                    thinking=(
                        "Need evidence.\n"
                        "<tool_call>\n"
                        "<function=echo_tool>\n"
                        "<parameter=value>rescued from thinking</parameter>\n"
                        "</function>\n"
                        "</tool_call>"
                    ),
                    finish_reason="stop",
                )
            assert messages[-1].role == "tool"
            assert messages[-1].name == "echo_tool"
            return GenerationResult(content="tool output consumed", finish_reason="stop")

    registry = ToolRegistry(discover_builtin=False)
    registry.register(EchoTool())
    loop = AsyncReActLoop(backend=_ThinkingToolMarkupBackend(), tool_registry=registry, max_iterations=3)

    events = [event async for event in loop.run("system", [], "use a tool")]

    statuses = [
        event
        for event in events
        if isinstance(event, StatusEvent)
        and event.metadata.get("reason") == "thinking_tool_call_rescue"
    ]
    thinking_events = [event for event in events if isinstance(event, ThinkingEvent)]
    requests = [event for event in events if isinstance(event, ToolCallRequestEvent)]
    results = [event for event in events if isinstance(event, ToolCallResultEvent)]
    finals = [event for event in events if isinstance(event, FinalAnswerEvent)]

    assert len(statuses) == 1
    assert statuses[0].metadata["tool_names"] == ["echo_tool"]
    assert all("<tool_call>" not in event.content for event in thinking_events)
    assert len(requests) == 1
    assert requests[0].tool_name == "echo_tool"
    assert requests[0].arguments == {"value": "rescued from thinking"}
    assert len(results) == 1
    assert results[0].result == {"echo": "rescued from thinking"}
    assert len(finals) == 1
    assert finals[0].content == "tool output consumed"

@pytest.mark.asyncio
async def test_react_loop_recovers_once_from_length_limited_final_answer() -> None:
    from mochi.agents.react_loop import AsyncReActLoop

    class _LengthBackend(FakeBackend):
        def __init__(self) -> None:
            super().__init__(backend_type="ollama")
            self.count = 0

        async def generate(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append(messages)
            self.generation_kwargs.append(dict(kwargs))
            self.count += 1
            if self.count == 1:
                return GenerationResult(content="partial answer", finish_reason="length")
            assert "Continue exactly where it stopped" in messages[-1].content
            assert messages[-2].role == "assistant"
            assert messages[-2].content == "partial answer"
            return GenerationResult(content=" completed", finish_reason="stop")

    backend = _LengthBackend()
    loop = AsyncReActLoop(backend=backend, tool_registry=None, max_iterations=3)

    events = [event async for event in loop.run("system", [], "user request")]

    truncation_events = [event for event in events if isinstance(event, AssistantTruncatedEvent)]
    statuses = [
        event
        for event in events
        if isinstance(event, StatusEvent)
        and event.metadata.get("reason") == "finish_reason_length"
    ]
    finals = [event for event in events if isinstance(event, FinalAnswerEvent)]
    assert len(truncation_events) == 1
    assert truncation_events[0].finish_reason == "length"
    assert truncation_events[0].recovery_attempt == 1
    assert truncation_events[0].partial_output_chars == len("partial answer")
    assert truncation_events[0].metadata["error_type"] == "output_truncated"
    assert len(statuses) == 1
    assert statuses[0].metadata["reason"] == "finish_reason_length"
    assert statuses[0].metadata["runtime_category"] == "truncation"
    assert statuses[0].metadata["error_type"] == "output_truncated"
    assert statuses[0].metadata["recoverability"] == "retrying"
    assert len(finals) == 1
    assert finals[0].content == "completed"
    assert finals[0].finish_reason == "stop"
    assert finals[0].metadata == {
        "runtime_category": "truncation",
        "error_type": "output_truncated",
        "recoverability": "recovered",
        "truncated": True,
        "recovery_attempts": 1,
    }
    assert backend.count == 2


@pytest.mark.asyncio
async def test_react_loop_recovers_truncated_final_text_tool_call_markup() -> None:
    from mochi.agents.events import ToolCallRequestEvent, ToolCallResultEvent
    from mochi.agents.react_loop import AsyncReActLoop

    class _SplitToolMarkupBackend(FakeBackend):
        async def generate(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append(messages)
            self.generation_kwargs.append(dict(kwargs))
            if len(self.calls) == 1:
                return GenerationResult(
                    content='<tool_call>{"name":"echo_tool","arguments":{"value":"split',
                    finish_reason="length",
                )
            if len(self.calls) == 2:
                assert "Continue exactly where it stopped" in messages[-1].content
                assert messages[-2].role == "assistant"
                return GenerationResult(content=' rescued"}}</tool_call>', finish_reason="stop")
            assert messages[-1].role == "tool"
            assert messages[-1].name == "echo_tool"
            return GenerationResult(content="tool output consumed", finish_reason="stop")

    registry = ToolRegistry(discover_builtin=False)
    registry.register(EchoTool())
    backend = _SplitToolMarkupBackend()
    loop = AsyncReActLoop(backend=backend, tool_registry=registry, max_iterations=4)

    events = [event async for event in loop.run("system", [], "use a tool")]

    truncation_statuses = [
        event
        for event in events
        if isinstance(event, StatusEvent)
        and event.metadata.get("reason") == "finish_reason_length"
    ]
    rescue_statuses = [
        event
        for event in events
        if isinstance(event, StatusEvent)
        and event.metadata.get("reason") == "truncated_final_text_tool_call_rescue"
    ]
    requests = [event for event in events if isinstance(event, ToolCallRequestEvent)]
    results = [event for event in events if isinstance(event, ToolCallResultEvent)]
    finals = [event for event in events if isinstance(event, FinalAnswerEvent)]

    assert len(truncation_statuses) == 1
    assert len(rescue_statuses) == 1
    assert rescue_statuses[0].metadata["tool_names"] == ["echo_tool"]
    assert len(requests) == 1
    assert requests[0].arguments == {"value": "split rescued"}
    assert len(results) == 1
    assert results[0].result == {"echo": "split rescued"}
    assert len(finals) == 1
    assert finals[0].content == "tool output consumed"
    assert any(message.role == "tool" and message.name == "echo_tool" for message in backend.calls[-1])


@pytest.mark.asyncio
async def test_react_loop_enforces_file_artifact_obligation_before_final_answer() -> None:
    from mochi.agents.events import ToolCallRequestEvent, ToolCallResultEvent
    from mochi.agents.react_loop import AsyncReActLoop
    from mochi.backends.types import ToolCall

    class _FileArtifactBackend(FakeBackend):
        async def generate(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append(messages)
            self.generation_kwargs.append(dict(kwargs))
            if len(self.calls) == 1:
                return GenerationResult(content="Saved report.md", finish_reason="stop")
            if len(self.calls) == 2:
                assert "no successful file mutation" in messages[-1].content
                return GenerationResult(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            name="file_write",
                            arguments={"path": "report.md", "content": "# Report\n"},
                        )
                    ],
                    finish_reason="tool_calls",
                )
            assert messages[-1].role == "tool"
            assert messages[-1].name == "file_write"
            return GenerationResult(content="Saved report.md", finish_reason="stop")

    registry = ToolRegistry(discover_builtin=False)
    registry.register(FileWriteProbeTool())
    backend = _FileArtifactBackend()
    loop = AsyncReActLoop(
        backend=backend,
        tool_registry=registry,
        max_iterations=4,
        requires_file_mutation=True,
    )

    events = [event async for event in loop.run("system", [], "save a report file")]

    guard_statuses = [
        event
        for event in events
        if isinstance(event, StatusEvent)
        and event.metadata.get("reason") == "file_artifact_missing"
    ]
    requests = [event for event in events if isinstance(event, ToolCallRequestEvent)]
    results = [event for event in events if isinstance(event, ToolCallResultEvent)]
    finals = [event for event in events if isinstance(event, FinalAnswerEvent)]

    assert len(guard_statuses) == 1
    assert guard_statuses[0].metadata["available_file_mutation_tools"] == ["file_write"]
    assert len(requests) == 1
    assert requests[0].tool_name == "file_write"
    assert len(results) == 1
    assert results[0].error is None
    assert len(finals) == 1
    assert finals[0].content == "Saved report.md"
    assert any(
        message.role == "user" and "no successful file mutation" in message.content
        for message in backend.calls[1]
    )


@pytest.mark.asyncio
async def test_react_loop_does_not_count_failed_file_preview_as_file_mutation() -> None:
    from mochi.agents.events import ToolCallResultEvent
    from mochi.agents.react_loop import AsyncReActLoop
    from mochi.backends.types import ToolCall

    class _FileArtifactApprovalBackend(FakeBackend):
        async def generate(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append(messages)
            self.generation_kwargs.append(dict(kwargs))
            if len(self.calls) == 1:
                return GenerationResult(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            name="file_write",
                            arguments={"path": "report.md", "content": "# Report\n"},
                        )
                    ],
                    finish_reason="tool_calls",
                )
            if len(self.calls) == 2:
                return GenerationResult(content="Saved report.md", finish_reason="stop")
            assert "previous file mutation attempt failed" in messages[-1].content
            return GenerationResult(content="Blocked: file write requires approval.", finish_reason="stop")

    registry = ToolRegistry(discover_builtin=False)
    registry.register(FileWriteProbeTool(error="File write requires approval."))
    backend = _FileArtifactApprovalBackend()
    loop = AsyncReActLoop(
        backend=backend,
        tool_registry=registry,
        max_iterations=4,
        requires_file_mutation=True,
    )

    events = [event async for event in loop.run("system", [], "save a report file")]

    guard_statuses = [
        event
        for event in events
        if isinstance(event, StatusEvent)
        and event.metadata.get("reason") == "file_artifact_missing"
    ]
    results = [event for event in events if isinstance(event, ToolCallResultEvent)]
    finals = [event for event in events if isinstance(event, FinalAnswerEvent)]

    assert len(results) == 1
    assert results[0].error == "File write requires approval."
    assert len(guard_statuses) == 2
    assert guard_statuses[-1].metadata["last_file_mutation_error"] == "File write requires approval."
    assert len(finals) == 1
    assert finals[0].content == "Blocked: file write requires approval."


@pytest.mark.asyncio
async def test_react_loop_marks_final_answer_when_length_recovery_also_truncates() -> None:
    from mochi.agents.react_loop import AsyncReActLoop

    class _StillLengthBackend(FakeBackend):
        async def generate(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append(messages)
            self.generation_kwargs.append(dict(kwargs))
            return GenerationResult(content=f"part {len(self.calls)}", finish_reason="length")

    backend = _StillLengthBackend(backend_type="ollama")
    loop = AsyncReActLoop(backend=backend, tool_registry=None, max_iterations=3)

    events = [event async for event in loop.run("system", [], "user request")]

    truncation_statuses = [
        event
        for event in events
        if isinstance(event, StatusEvent)
        and event.metadata.get("reason") == "finish_reason_length"
    ]
    assert len(truncation_statuses) == 1
    finals = [event for event in events if isinstance(event, FinalAnswerEvent)]
    assert len(finals) == 1
    assert finals[0].content == "part 2"
    assert finals[0].finish_reason == "length"
    assert finals[0].metadata == {
        "runtime_category": "truncation",
        "error_type": "output_truncated",
        "recoverability": "partial",
        "truncated": True,
        "recovery_attempts": 1,
    }
    assert len(backend.calls) == 2
