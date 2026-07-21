from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from mochi.agents.events import FinalAnswerEvent, StatusEvent, ToolCallResultEvent
from mochi.agents.react_loop import AsyncReActLoop
from mochi.backends.base import BaseLLMBackend
from mochi.backends.types import GenerationResult, Message, ModelInfo, StreamChunk, ToolCall
from mochi.tools.base import ToolExecutionContext
from mochi.tools.file_ops import FileWriteTool
from mochi.tools.registry import ToolRegistry


class _FakeBackend(BaseLLMBackend):
    def __init__(self) -> None:
        self.calls: list[list[Message]] = []
        self.generation_kwargs: list[dict[str, Any]] = []

    async def generate(
        self,
        messages: list[Message],
        tools: list | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: float = 1.0,
        min_p: float = 0.0,
        top_k: int = 0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repeat_penalty: float = 1.0,
        reasoning_effort: str | None = None,
        stream: bool = False,
    ) -> GenerationResult | AsyncIterator[StreamChunk]:
        self.calls.append(messages)
        self.generation_kwargs.append(
            {
                "tools": [tool.name for tool in tools or []],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "min_p": min_p,
                "top_k": top_k,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "repeat_penalty": repeat_penalty,
                "reasoning_effort": reasoning_effort,
                "stream": stream,
            }
        )
        return GenerationResult(content="ok")

    def supports_tool_calling(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name="fake", backend_type="test", metadata={})

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


def _claims_saved(content: str) -> bool:
    normalized = content.strip().lower()
    return "saved report.md" in normalized or "已保存" in normalized or "已儲存" in normalized


@pytest.mark.asyncio
async def test_hidden_file_write_tool_not_exposed_is_activation_contract_failure() -> None:
    class _HiddenFileWriteBackend(_FakeBackend):
        async def generate(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append(messages)
            self.generation_kwargs.append(dict(kwargs))
            if len(self.calls) in {1, 3}:
                return GenerationResult(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id=f"call-{len(self.calls)}",
                            name="file_write",
                            arguments={"path": "report.md", "content": "# Report\n"},
                        )
                    ],
                    finish_reason="tool_calls",
                )
            return GenerationResult(content="Saved report.md", finish_reason="stop")

    registry = ToolRegistry(discover_builtin=False)
    loop = AsyncReActLoop(
        backend=_HiddenFileWriteBackend(),
        tool_registry=registry,
        max_iterations=5,
        requires_file_mutation=True,
    )

    events = [event async for event in loop.run("system", [], "save report.md")]

    results = [event for event in events if isinstance(event, ToolCallResultEvent)]
    finals = [event for event in events if isinstance(event, FinalAnswerEvent)]

    assert [event.tool_name for event in results] == ["file_write", "file_write"]
    assert all(event.error for event in results)
    assert all("tool_not_exposed" in event.metadata["guard"] for event in results)
    assert all(event.metadata["runtime_category"] == "tool_activation" for event in results)
    assert results[0].metadata["error_type"] == "mutation_tool_not_callable"
    assert results[1].metadata["error_type"] == "repeated_unavailable_mutation_tool"
    assert all(
        event.metadata["recoverability"] == "requires_replanning_or_activation"
        for event in results
    )
    assert all(event.metadata["callable_this_turn"] is False for event in results)
    assert all(event.metadata["activation_required"] is True for event in results)
    assert results[1].metadata["retryable"] is False

    assert len(finals) == 1
    assert finals[0].content == (
        "I could not save the file because the required write tool was not callable in this turn."
    )
    assert not _claims_saved(finals[0].content)
    assert finals[0].metadata["reason"] == "file_artifact_not_mutated"
    assert finals[0].metadata["error_type"] == "file_artifact_not_mutated"
    assert finals[0].metadata["runtime_category"] == "deliverable_guard"
    assert finals[0].metadata["recoverability"] == "requires_replanning_or_activation"
    assert finals[0].metadata["last_file_mutation_tool"] == "file_write"


@pytest.mark.asyncio
async def test_write_obligation_without_successful_mutation_blocks_saved_final() -> None:
    class _PrematureSavedBackend(_FakeBackend):
        async def generate(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append(messages)
            self.generation_kwargs.append(dict(kwargs))
            return GenerationResult(content="Saved report.md", finish_reason="stop")

    registry = ToolRegistry(discover_builtin=False)
    backend = _PrematureSavedBackend()
    loop = AsyncReActLoop(
        backend=backend,
        tool_registry=registry,
        max_iterations=3,
        requires_file_mutation=True,
    )

    events = [event async for event in loop.run("system", [], "save report.md")]

    results = [event for event in events if isinstance(event, ToolCallResultEvent)]
    guard_statuses = [
        event
        for event in events
        if isinstance(event, StatusEvent) and event.metadata.get("reason") == "file_artifact_missing"
    ]
    finals = [event for event in events if isinstance(event, FinalAnswerEvent)]

    assert results == []
    assert len(guard_statuses) == 2
    assert len(finals) == 1
    assert finals[0].content == (
        "I could not save the file because the required write tool was not callable in this turn."
    )
    assert not _claims_saved(finals[0].content)
    assert finals[0].metadata["available_file_mutation_tools"] == []
    assert finals[0].metadata["last_file_mutation_tool"] is None
    assert len(backend.calls) == 3


def _make_activation_workspace() -> Path:
    root = Path(".tmp") / "tool-activation-contract-tests"
    root.mkdir(parents=True, exist_ok=True)
    workspace = root / uuid4().hex
    workspace.mkdir(parents=True, exist_ok=False)
    return workspace.resolve()
def _activation_context(workspace_dir: str) -> ToolExecutionContext:
    return ToolExecutionContext(
        workspace_dir=workspace_dir,
        permission_policy={
            "autonomy_mode": "auto_review",
            "require_approval_for_file_write": False,
            "file_read_scope": "workspace",
            "file_write_scope": "workspace",
        },
        state={
            "tool_activation_policy": {
                "routed_intent": "workspace_write",
                "execution_profile": "chat",
                "tool_mode": "auto",
                "discoverable_tool_names": ["file_write"],
                "tool_allowlist": None,
                "tool_denylist": None,
            }
        },
    )


@pytest.mark.asyncio
async def test_hidden_file_write_activation_promotes_tool_for_next_iteration() -> None:
    tmp_path = _make_activation_workspace()
    class _ActivatingBackend(_FakeBackend):
        async def generate(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append(messages)
            self.generation_kwargs.append(dict(kwargs))
            if len(self.calls) in {1, 2}:
                return GenerationResult(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id=f"call-{len(self.calls)}",
                            name="file_write",
                            arguments={"path": "report.md", "content": "# Report\n"},
                        )
                    ],
                    finish_reason="tool_calls",
                )
            return GenerationResult(content="Saved report.md", finish_reason="stop")

    source_registry = ToolRegistry(discover_builtin=False)
    source_registry.register(FileWriteTool(workspace_dir=tmp_path, require_approval=False))
    registry = source_registry.create_view([], tool_search_catalog_names=["file_write"])
    loop = AsyncReActLoop(
        backend=_ActivatingBackend(),
        tool_registry=registry,
        tool_execution_context=_activation_context(str(tmp_path)),
        max_iterations=5,
        requires_file_mutation=True,
    )

    events = [event async for event in loop.run("system", [], "save report.md")]

    results = [event for event in events if isinstance(event, ToolCallResultEvent)]
    finals = [event for event in events if isinstance(event, FinalAnswerEvent)]

    assert [event.tool_name for event in results] == ["file_write", "file_write"]
    assert results[0].error is None
    assert results[0].metadata["status"] == "tool_activated"
    assert results[1].error is None
    assert (tmp_path / "report.md").read_text(encoding="utf-8") == "# Report\n"
    assert len(finals) == 1
    assert finals[0].content == "Saved report.md"


@pytest.mark.asyncio
async def test_activation_without_actual_mutation_still_blocks_saved_final() -> None:
    tmp_path = _make_activation_workspace()
    class _ActivatedButNoWriteBackend(_FakeBackend):
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
            return GenerationResult(content="Saved report.md", finish_reason="stop")

    source_registry = ToolRegistry(discover_builtin=False)
    source_registry.register(FileWriteTool(workspace_dir=tmp_path, require_approval=False))
    registry = source_registry.create_view([], tool_search_catalog_names=["file_write"])
    backend = _ActivatedButNoWriteBackend()
    loop = AsyncReActLoop(
        backend=backend,
        tool_registry=registry,
        tool_execution_context=_activation_context(str(tmp_path)),
        max_iterations=4,
        requires_file_mutation=True,
    )

    events = [event async for event in loop.run("system", [], "save report.md")]

    results = [event for event in events if isinstance(event, ToolCallResultEvent)]
    finals = [event for event in events if isinstance(event, FinalAnswerEvent)]

    assert len(results) == 1
    assert results[0].metadata["status"] == "tool_activated"
    assert not (tmp_path / "report.md").exists()
    assert len(finals) == 1
    assert finals[0].metadata["error_type"] == "file_artifact_not_mutated"
    assert not _claims_saved(finals[0].content)


@pytest.mark.asyncio
async def test_denied_hidden_file_write_activation_is_not_replayed_forever() -> None:
    tmp_path = _make_activation_workspace()
    class _DeniedActivationBackend(_FakeBackend):
        async def generate(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append(messages)
            self.generation_kwargs.append(dict(kwargs))
            if len(self.calls) <= 3:
                return GenerationResult(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id=f"call-{len(self.calls)}",
                            name="file_write",
                            arguments={"path": "report.md", "content": "# Report\n"},
                        )
                    ],
                    finish_reason="tool_calls",
                )
            return GenerationResult(content="Saved report.md", finish_reason="stop")

    source_registry = ToolRegistry(discover_builtin=False)
    source_registry.register(FileWriteTool(workspace_dir=tmp_path, require_approval=False))
    registry = source_registry.create_view([], tool_search_catalog_names=["file_write"])
    context = ToolExecutionContext(
        workspace_dir=str(tmp_path),
        permission_policy={
            "autonomy_mode": "strict",
            "require_approval_for_file_write": True,
            "file_read_scope": "workspace",
            "file_write_scope": "workspace",
        },
        state={
            "tool_activation_policy": {
                "routed_intent": "workspace_write",
                "execution_profile": "chat",
                "tool_mode": "auto",
                "discoverable_tool_names": ["file_write"],
                "tool_allowlist": None,
                "tool_denylist": None,
            }
        },
    )
    loop = AsyncReActLoop(
        backend=_DeniedActivationBackend(),
        tool_registry=registry,
        tool_execution_context=context,
        max_iterations=5,
        requires_file_mutation=True,
    )

    events = [event async for event in loop.run("system", [], "save report.md")]

    results = [event for event in events if isinstance(event, ToolCallResultEvent)]
    finals = [event for event in events if isinstance(event, FinalAnswerEvent)]

    assert [event.metadata["error_type"] for event in results[:2]] == [
        "tool_activation_denied",
        "repeated_unavailable_mutation_tool",
    ]
    assert results[0].metadata["reason"] == "approval_required"
    assert len(finals) == 1
    assert finals[0].metadata["error_type"] == "file_artifact_not_mutated"

@pytest.mark.asyncio
async def test_discovery_request_does_not_call_hidden_file_write_tool() -> None:
    from mochi.tools.tool_search import ToolSearchTool

    workspace = _make_activation_workspace()
    file_tool = FileWriteTool(workspace_dir=workspace, require_approval=False)
    source_registry = ToolRegistry(discover_builtin=False)
    source_registry.register(file_tool)
    source_registry.register(
        ToolSearchTool(
            catalog_provider=lambda: [file_tool],
            callable_name_provider=lambda: {"tool_search"},
        )
    )
    registry = source_registry.create_view(
        ["tool_search"],
        tool_search_catalog_names=["file_write"],
    )

    class _DiscoveryBackend(_FakeBackend):
        async def generate(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append(messages)
            self.generation_kwargs.append(dict(kwargs))
            if len(self.calls) == 1:
                return GenerationResult(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="discover-1",
                            name="tool_search",
                            arguments={"query": "saving files"},
                        )
                    ],
                    finish_reason="tool_calls",
                )
            return GenerationResult(
                content="file_write is discoverable but not callable this turn.",
                finish_reason="stop",
            )

    backend = _DiscoveryBackend()
    loop = AsyncReActLoop(
        backend=backend,
        tool_registry=registry,
        max_iterations=3,
        requires_file_mutation=False,
    )

    events = [
        event
        async for event in loop.run(
            "system",
            [],
            "你有沒有保存檔案的工具？",
        )
    ]

    results = [event for event in events if isinstance(event, ToolCallResultEvent)]
    finals = [event for event in events if isinstance(event, FinalAnswerEvent)]

    assert [event.tool_name for event in results] == ["tool_search"]
    assert results[0].error is None
    matches = results[0].result
    assert isinstance(matches, list)
    file_match = next(match for match in matches if match["name"] == "file_write")
    assert file_match["callable_this_turn"] is False
    assert file_match["activation_required"] is True
    assert registry.get("file_write") is None
    assert len(finals) == 1
    assert "not callable" in finals[0].content
    assert not (workspace / "discovery-only.txt").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "path"),
    [
        ("請把上一段程式存成 train_med_vlm_lora.py", "train_med_vlm_lora.py"),
        ("帮我生成 requirements.txt 并保存", "requirements.txt"),
        ("save the previous script as train.py", "train.py"),
        ("把剛剛的內容另存為 infer_med_vlm_lora.py", "infer_med_vlm_lora.py"),
    ],
)
async def test_multilingual_workspace_write_produces_file(
    message: str,
    path: str,
) -> None:
    workspace = _make_activation_workspace()
    content = f"generated for: {message}\n"
    source_registry = ToolRegistry(discover_builtin=False)
    source_registry.register(FileWriteTool(workspace_dir=workspace, require_approval=False))
    registry = source_registry.create_view(
        ["file_write"],
        tool_search_catalog_names=["file_write"],
    )

    class _WriteBackend(_FakeBackend):
        async def generate(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append(messages)
            self.generation_kwargs.append(dict(kwargs))
            if len(self.calls) == 1:
                return GenerationResult(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="write-1",
                            name="file_write",
                            arguments={"path": path, "content": content},
                        )
                    ],
                    finish_reason="tool_calls",
                )
            return GenerationResult(content=f"Saved {path}", finish_reason="stop")

    backend = _WriteBackend()
    loop = AsyncReActLoop(
        backend=backend,
        tool_registry=registry,
        tool_execution_context=_activation_context(str(workspace)),
        max_iterations=3,
        requires_file_mutation=True,
    )

    events = [
        event
        async for event in loop.run(
            "system",
            [],
            message,
        )
    ]

    results = [event for event in events if isinstance(event, ToolCallResultEvent)]
    finals = [event for event in events if isinstance(event, FinalAnswerEvent)]

    assert [event.tool_name for event in results] == ["file_write"]
    assert results[0].error is None
    assert (workspace / path).read_text(encoding="utf-8") == content
    assert len(finals) == 1
    assert finals[0].content == f"Saved {path}"
