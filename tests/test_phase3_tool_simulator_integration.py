from __future__ import annotations

from pathlib import Path

import pytest

from mochi.agents.events import FinalAnswerEvent, ToolCallResultEvent
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
        tool_message = next((message for message in messages if message.role == "tool"), None)
        if tool_message is not None:
            assert "Tool file_read result preview" in tool_message.content
            assert 'file_read(path="tool-result://' in tool_message.content
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
    assert context.tool_result_references[reference_id]["reference_id"] == reference_id
    assert final_event.content == "done"
