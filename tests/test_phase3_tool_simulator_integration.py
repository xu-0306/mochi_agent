from __future__ import annotations

from pathlib import Path

import pytest

from mochi.agents.events import FinalAnswerEvent, ThinkingEvent, ToolCallResultEvent
from mochi.agents.react_loop import AsyncReActLoop
from mochi.backends.base import BaseLLMBackend
from mochi.backends.types import GenerationResult, Message, ModelInfo, ToolCall, ToolSchema
from mochi.tools.base import ToolExecutionContext
from mochi.tools.file_ops import FileReadTool
from mochi.tools.registry import ToolRegistry


class _JsonLookingFileReadBackend(BaseLLMBackend):
    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: float = 1.0,
        min_p: float = 0.0,
        top_k: int = 0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repeat_penalty: float = 1.0,
        stream: bool = False,
    ) -> GenerationResult:
        del tools, temperature, max_tokens, top_p, min_p, top_k
        del frequency_penalty, presence_penalty, repeat_penalty, stream
        self.calls.append(messages)
        tool_message = next((message for message in messages if message.role == "tool"), None)
        if tool_message is not None:
            assert tool_message.content == '{"name": "mochi"}'
            assert not tool_message.content.startswith('{"ok":')
            return GenerationResult(content="done")
        return GenerationResult(
            content="",
            tool_calls=[
                ToolCall(
                    id="call-file-read-json",
                    name="file_read",
                    arguments={"path": "sample.json", "line_numbers": False},
                )
            ],
            finish_reason="tool_calls",
        )

    def supports_tool_calling(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name="json-file-read-backend", backend_type="test")

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class _LargeFileReadBackend(BaseLLMBackend):
    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: float = 1.0,
        min_p: float = 0.0,
        top_k: int = 0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repeat_penalty: float = 1.0,
        stream: bool = False,
    ) -> GenerationResult:
        del tools, temperature, max_tokens, top_p, min_p, top_k
        del frequency_penalty, presence_penalty, repeat_penalty, stream
        self.calls.append(messages)
        tool_messages = [message for message in messages if message.role == "tool"]
        tool_message = tool_messages[-1] if tool_messages else None
        if tool_message is not None:
            if 'tool-result://' in tool_message.content:
                reference_id = tool_message.content.split('tool-result://', 1)[1].split('"', 1)[0]
                return GenerationResult(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call-file-read-continue",
                            name="file_read",
                            arguments={"path": f"tool-result://{reference_id}", "offset": 61, "limit": 3, "line_numbers": True},
                        )
                    ],
                    finish_reason="tool_calls",
                )
            assert tool_message.content == "61: line 61\n62: line 62\n63: line 63"
            return GenerationResult(content="done")
        return GenerationResult(
            content="",
            tool_calls=[
                ToolCall(
                    id="call-file-read-large",
                    name="file_read",
                    arguments={"path": "large.txt", "offset": 1, "limit": 60, "line_numbers": True},
                )
            ],
            finish_reason="tool_calls",
        )

    def supports_tool_calling(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name="large-file-read-backend", backend_type="test")

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class _UnavailableToolBackend(BaseLLMBackend):
    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: float = 1.0,
        min_p: float = 0.0,
        top_k: int = 0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repeat_penalty: float = 1.0,
        stream: bool = False,
    ) -> GenerationResult:
        del tools, temperature, max_tokens, top_p, min_p, top_k
        del frequency_penalty, presence_penalty, repeat_penalty, stream
        self.calls.append(messages)
        tool_message = next((message for message in messages if message.role == "tool"), None)
        if tool_message is not None:
            assert "exec_command" in tool_message.content
            assert "not available in this turn" in tool_message.content
            assert "file_read" in tool_message.content
            return GenerationResult(content="Recovered with visible tools only.")
        return GenerationResult(
            content="",
            tool_calls=[
                ToolCall(
                    id="call-hidden-exec",
                    name="exec_command",
                    arguments={"cmd": "python transform.py"},
                )
            ],
            finish_reason="tool_calls",
        )

    def supports_tool_calling(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name="unavailable-tool-backend", backend_type="test")

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class _ChannelMarkerBackend(BaseLLMBackend):
    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: float = 1.0,
        min_p: float = 0.0,
        top_k: int = 0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repeat_penalty: float = 1.0,
        stream: bool = False,
    ) -> GenerationResult:
        del messages, tools, temperature, max_tokens, top_p, min_p, top_k
        del frequency_penalty, presence_penalty, repeat_penalty, stream
        return GenerationResult(content="<|channel|>thought<channel|>Visible final answer.")

    def supports_tool_calling(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name="channel-marker-backend", backend_type="test")

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class _AnalysisTagBackend(BaseLLMBackend):
    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: float = 1.0,
        min_p: float = 0.0,
        top_k: int = 0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repeat_penalty: float = 1.0,
        stream: bool = False,
    ) -> GenerationResult:
        del messages, tools, temperature, max_tokens, top_p, min_p, top_k
        del frequency_penalty, presence_penalty, repeat_penalty, stream
        return GenerationResult(content="<analysis>Inspect attachment schema first.</analysis>Visible final answer.")

    def supports_tool_calling(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name="analysis-tag-backend", backend_type="test")

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class _RoleSentinelBackend(BaseLLMBackend):
    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: float = 1.0,
        min_p: float = 0.0,
        top_k: int = 0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repeat_penalty: float = 1.0,
        stream: bool = False,
    ) -> GenerationResult:
        del messages, tools, temperature, max_tokens, top_p, min_p, top_k
        del frequency_penalty, presence_penalty, repeat_penalty, stream
        return GenerationResult(
            content="<|im_start|>assistant<|im_end|>\nVisible final answer.",
            thinking="<｜Assistant｜>Review the attached records first.",
        )

    def supports_tool_calling(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name="role-sentinel-backend", backend_type="test")

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_react_loop_preserves_small_json_looking_file_read_text(
    tmp_path: Path,
) -> None:
    (tmp_path / "sample.json").write_text('{"name": "mochi"}', encoding="utf-8")

    backend = _JsonLookingFileReadBackend()
    registry = ToolRegistry(discover_builtin=False)
    registry.register(FileReadTool(workspace_dir=tmp_path))
    react_loop = AsyncReActLoop(
        backend=backend,
        tool_registry=registry,
        tool_execution_context=ToolExecutionContext(workspace_dir=str(tmp_path)),
        max_iterations=3,
    )

    events = [
        event
        async for event in react_loop.run(
            system_prompt="system",
            history=[],
            user_message="read the json file",
        )
    ]

    result_event = next(event for event in events if isinstance(event, ToolCallResultEvent))
    final_event = next(event for event in events if isinstance(event, FinalAnswerEvent))

    assert result_event.result == '{"name": "mochi"}'
    assert final_event.content == "done"


@pytest.mark.asyncio
async def test_react_loop_persists_oversized_file_read_before_dead_end_preview(
    tmp_path: Path,
) -> None:
    (tmp_path / "large.txt").write_text(
        "".join(f"line {idx}\n" for idx in range(1, 80)),
        encoding="utf-8",
    )

    backend = _LargeFileReadBackend()
    context = ToolExecutionContext(
        workspace_dir=str(tmp_path),
        session_id="session-file-read-large",
        tool_result_store_dir=str(tmp_path / "tool-results"),
    )
    registry = ToolRegistry(discover_builtin=False)
    registry.register(FileReadTool(workspace_dir=tmp_path))
    react_loop = AsyncReActLoop(
        backend=backend,
        tool_registry=registry,
        tool_execution_context=context,
        max_iterations=3,
        max_tool_message_chars=220,
    )

    events = [
        event
        async for event in react_loop.run(
            system_prompt="system",
            history=[],
            user_message="read the large file",
        )
    ]

    result_event = next(event for event in events if isinstance(event, ToolCallResultEvent))
    final_event = next(event for event in events if isinstance(event, FinalAnswerEvent))
    transport = result_event.metadata["transport"]
    reference_id = transport["reference_id"]

    assert reference_id
    assert transport["overflow_persisted"] is True
    assert context.tool_result_references[reference_id]["source_path"] == str(
        (tmp_path / "large.txt").resolve(strict=False)
    )
    assert context.tool_result_references[reference_id]["reference_id"] == reference_id
    assert final_event.content == "done"


@pytest.mark.asyncio
async def test_react_loop_converts_hidden_tool_calls_into_recoverable_tool_errors() -> None:
    backend = _UnavailableToolBackend()
    registry = ToolRegistry(discover_builtin=False)
    registry.register(FileReadTool(workspace_dir="."))
    react_loop = AsyncReActLoop(
        backend=backend,
        tool_registry=registry,
        tool_execution_context=ToolExecutionContext(workspace_dir="."),
        max_iterations=3,
    )

    events = [
        event
        async for event in react_loop.run(
            system_prompt="system",
            history=[],
            user_message="Process the attached file.",
        )
    ]

    result_event = next(event for event in events if isinstance(event, ToolCallResultEvent))
    final_event = next(event for event in events if isinstance(event, FinalAnswerEvent))

    assert result_event.error is not None
    assert "not available in this turn" in result_event.error
    assert "file_read" in result_event.error
    assert final_event.content == "Recovered with visible tools only."


@pytest.mark.asyncio
async def test_react_loop_strips_channel_reasoning_markers_from_final_answer() -> None:
    backend = _ChannelMarkerBackend()
    react_loop = AsyncReActLoop(
        backend=backend,
        tool_registry=ToolRegistry(discover_builtin=False),
        tool_execution_context=ToolExecutionContext(workspace_dir="."),
        max_iterations=1,
    )

    events = [
        event
        async for event in react_loop.run(
            system_prompt="system",
            history=[],
            user_message="Answer directly.",
        )
    ]

    final_event = next(event for event in events if isinstance(event, FinalAnswerEvent))

    assert final_event.content == "Visible final answer."


@pytest.mark.asyncio
async def test_react_loop_extracts_analysis_tags_into_reasoning_events() -> None:
    backend = _AnalysisTagBackend()
    react_loop = AsyncReActLoop(
        backend=backend,
        tool_registry=ToolRegistry(discover_builtin=False),
        tool_execution_context=ToolExecutionContext(workspace_dir="."),
        max_iterations=1,
    )

    events = [
        event
        async for event in react_loop.run(
            system_prompt="system",
            history=[],
            user_message="Answer directly.",
        )
    ]

    thinking_event = next(event for event in events if isinstance(event, ThinkingEvent))
    final_event = next(event for event in events if isinstance(event, FinalAnswerEvent))

    assert thinking_event.content == "Inspect attachment schema first."
    assert final_event.content == "Visible final answer."


@pytest.mark.asyncio
async def test_react_loop_strips_role_sentinels_from_visible_and_reasoning_text() -> None:
    backend = _RoleSentinelBackend()
    react_loop = AsyncReActLoop(
        backend=backend,
        tool_registry=ToolRegistry(discover_builtin=False),
        tool_execution_context=ToolExecutionContext(workspace_dir="."),
        max_iterations=1,
    )

    events = [
        event
        async for event in react_loop.run(
            system_prompt="system",
            history=[],
            user_message="Answer directly.",
        )
    ]

    thinking_event = next(event for event in events if isinstance(event, ThinkingEvent))
    final_event = next(event for event in events if isinstance(event, FinalAnswerEvent))

    assert thinking_event.content == "Review the attached records first."
    assert final_event.content == "Visible final answer."
