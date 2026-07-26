from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from mochi.agents.events import FinalAnswerEvent, StatusEvent, ToolCallResultEvent
from mochi.agents.react_loop import AsyncReActLoop
from mochi.backends.base import BaseLLMBackend
from mochi.backends.types import GenerationResult, Message, ModelInfo, StreamChunk, ToolCall
from mochi.tools.base import BaseTool, ToolExecutionContext, ToolResult
from mochi.tools.file_ops import FileWriteTool
from mochi.tools.registry import ToolRegistry
from mochi.tools.tool_search import ToolSearchTool


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


class _NamedTestTool(BaseTool):
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Test tool {self._name}."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }

    async def execute(self) -> ToolResult:
        return ToolResult(output={"tool": self._name})


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
                "capability_enforcement_mode": "enforce",
                "mutation_requirement": "required",
                "requested_operations": ["workspace_write"],
                "required_capabilities": ["workspace_write"],
                "activation_allowed_tool_names": ["file_write"],
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
async def test_explicit_activation_broker_refreshes_native_tool_schema() -> None:
    workspace = _make_activation_workspace()
    file_tool = FileWriteTool(workspace_dir=workspace, require_approval=False)
    source_registry = ToolRegistry(discover_builtin=False)
    source_registry.register(file_tool)
    source_registry.register(ToolSearchTool(catalog_provider=source_registry.list_tools))
    registry = source_registry.create_view(
        ["tool_search"],
        tool_search_catalog_names=["file_write"],
    )

    class _NativeActivationBackend(_FakeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.schema_history: list[list[str]] = []

        async def generate(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append(messages)
            tools = kwargs.get("tools") or []
            tool_names = [tool.name for tool in tools]
            self.schema_history.append(tool_names)
            self.generation_kwargs.append(dict(kwargs))
            if len(self.calls) == 1:
                assert tool_names == ["tool_search", "tool_activate"]
                return GenerationResult(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="search-1",
                            name="tool_search",
                            arguments={"query": "file_write"},
                        )
                    ],
                    finish_reason="tool_calls",
                )
            if len(self.calls) == 2:
                assert tool_names == ["tool_search", "tool_activate"]
                return GenerationResult(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="activate-1",
                            name="tool_activate",
                            arguments={"tool_name": "file_write"},
                        )
                    ],
                    finish_reason="tool_calls",
                )
            if len(self.calls) == 3:
                assert tool_names == ["tool_search", "file_write"]
                return GenerationResult(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="write-1",
                            name="file_write",
                            arguments={
                                "path": "broker-report.md",
                                "content": "# Broker report\n",
                            },
                        )
                    ],
                    finish_reason="tool_calls",
                )
            return GenerationResult(content="Saved broker-report.md", finish_reason="stop")

    backend = _NativeActivationBackend()
    context = _activation_context(str(workspace))
    loop = AsyncReActLoop(
        backend=backend,
        tool_registry=registry,
        tool_execution_context=context,
        max_iterations=5,
        requires_file_mutation=True,
    )

    events = [
        event
        async for event in loop.run("system", [], "save broker-report.md")
    ]

    results = [event for event in events if isinstance(event, ToolCallResultEvent)]
    finals = [event for event in events if isinstance(event, FinalAnswerEvent)]
    assert [event.tool_name for event in results] == [
        "tool_search",
        "tool_activate",
        "file_write",
    ]
    search_match = next(
        match for match in results[0].result if match["name"] == "file_write"
    )
    assert search_match["activation_request"] == {
        "tool_name": "file_write",
        "policy_check": "required",
        "required_intent": "workspace_write",
        "activation_tool": "tool_activate",
        "arguments": {"tool_name": "file_write"},
    }
    assert results[1].metadata["status"] == "tool_activated"
    assert results[1].metadata["activation_authorizes_tool_call"] is False
    assert results[2].error is None
    assert (workspace / "broker-report.md").read_text(encoding="utf-8") == (
        "# Broker report\n"
    )
    assert backend.schema_history[0] == ["tool_search", "tool_activate"]
    assert "file_write" not in backend.schema_history[1]
    assert "file_write" in backend.schema_history[2]
    assert "tool_activate" not in backend.schema_history[2]
    assert [event.content for event in finals] == ["Saved broker-report.md"]


@pytest.mark.asyncio
async def test_explicit_activation_broker_defers_strict_approval_to_file_call() -> None:
    workspace = _make_activation_workspace()
    source_registry = ToolRegistry(discover_builtin=False)
    source_registry.register(
        FileWriteTool(workspace_dir=workspace, require_approval=False)
    )
    registry = source_registry.create_view([], tool_search_catalog_names=["file_write"])
    context = _activation_context(str(workspace))
    context.permission_policy["autonomy_mode"] = "strict"
    context.permission_policy["require_approval_for_file_write"] = True

    activation = await registry.execute(
        "tool_activate",
        {"tool_name": "file_write"},
        context=context,
    )
    call_result = await registry.execute(
        "file_write",
        {"path": "strict.txt", "content": "blocked"},
        context=context,
    )

    assert activation.error is None
    assert activation.metadata["status"] == "tool_activated"
    assert activation.metadata["activation_authorizes_tool_call"] is False
    assert activation.metadata["authorization_required_for_call"] is True
    assert call_result.error == "File write requires approval."
    assert call_result.metadata["requires_approval"] is True
    assert not (workspace / "strict.txt").exists()


@pytest.mark.asyncio
async def test_explicit_activation_broker_preserves_contract_denial() -> None:
    workspace = _make_activation_workspace()
    source_registry = ToolRegistry(discover_builtin=False)
    source_registry.register(
        FileWriteTool(workspace_dir=workspace, require_approval=False)
    )
    registry = source_registry.create_view([], tool_search_catalog_names=["file_write"])
    context = _activation_context(str(workspace))
    context.state["tool_activation_policy"].update(
        {
            "mutation_requirement": "forbidden",
            "requested_operations": ["capability_inquiry"],
            "required_capabilities": ["tool_catalog_read"],
        }
    )

    result = await registry.execute(
        "tool_activate",
        {"tool_name": "file_write"},
        context=context,
    )

    assert result.error is not None
    assert result.metadata["reason"] == "mutation_forbidden_by_contract"
    assert registry.get("file_write") is None
    assert registry.get("tool_activate") is not None


@pytest.mark.asyncio
async def test_explicit_activation_broker_rejects_unknown_tool() -> None:
    workspace = _make_activation_workspace()
    source_registry = ToolRegistry(discover_builtin=False)
    source_registry.register(
        FileWriteTool(workspace_dir=workspace, require_approval=False)
    )
    registry = source_registry.create_view([], tool_search_catalog_names=["file_write"])
    context = _activation_context(str(workspace))

    result = await registry.execute(
        "tool_activate",
        {"tool_name": "not_a_real_tool"},
        context=context,
    )

    assert result.error is not None
    assert result.metadata["reason"] == "not_discoverable"
    assert registry.get("not_a_real_tool") is None


def test_activation_broker_schema_is_serializable_without_runtime_references() -> None:
    workspace = _make_activation_workspace()
    source_registry = ToolRegistry(discover_builtin=False)
    source_registry.register(
        FileWriteTool(workspace_dir=workspace, require_approval=False)
    )
    registry = source_registry.create_view([], tool_search_catalog_names=["file_write"])

    schemas = registry.get_schemas()
    serialized = json.dumps(schemas, ensure_ascii=False)

    assert [schema["function"]["name"] for schema in schemas] == ["tool_activate"]
    assert "request_activation" not in serialized
    assert "ToolRegistry" not in serialized


def test_activation_broker_is_absent_without_deferred_tools() -> None:
    workspace = _make_activation_workspace()
    source_registry = ToolRegistry(discover_builtin=False)
    source_registry.register(
        FileWriteTool(workspace_dir=workspace, require_approval=False)
    )

    registry = source_registry.create_view(
        ["file_write"],
        tool_search_catalog_names=["file_write"],
    )

    assert registry.get("tool_activate") is None


def test_activation_replaces_broker_when_last_deferred_tool_is_activated() -> None:
    source_registry = ToolRegistry(discover_builtin=False)
    for name in ("tool_search", "file_write"):
        source_registry.register(_NamedTestTool(name))
    registry = source_registry.create_view(
        ["tool_search"],
        tool_search_catalog_names=["tool_search", "file_write"],
        schema_limit=2,
    )
    context = _activation_context(str(_make_activation_workspace()))

    assert len(registry.get_schemas()) == 2
    result = registry.request_tool_activation("file_write", context=context)
    schema_names = [
        schema["function"]["name"] for schema in registry.get_schemas()
    ]

    assert result.error is None
    assert result.metadata["activation_broker_retained"] is False
    assert "file_write" in schema_names
    assert "tool_activate" not in schema_names
    assert len(schema_names) <= 2


def test_activation_evicts_nonessential_callable_when_deferred_tools_remain() -> None:
    source_registry = ToolRegistry(discover_builtin=False)
    for name in ("tool_search", "file_read", "file_write", "file_edit"):
        source_registry.register(_NamedTestTool(name))
    registry = source_registry.create_view(
        ["tool_search", "file_read"],
        tool_search_catalog_names=[
            "tool_search",
            "file_read",
            "file_write",
            "file_edit",
        ],
        schema_limit=3,
    )
    context = _activation_context(str(_make_activation_workspace()))

    assert len(registry.get_schemas()) == 3
    result = registry.request_tool_activation("file_write", context=context)
    schema_names = [
        schema["function"]["name"] for schema in registry.get_schemas()
    ]

    assert result.error is None
    assert result.metadata["activation_broker_retained"] is True
    assert result.metadata["activation_schema_evicted_tools"] == ["file_read"]
    assert "file_write" in schema_names
    assert "tool_activate" in schema_names
    assert len(schema_names) <= 3


def test_enforced_capability_plan_denies_ineligible_exec_activation() -> None:
    source_registry = ToolRegistry(discover_builtin=False)
    source_registry.register(_NamedTestTool("exec_command"))
    registry = source_registry.create_view(
        [],
        tool_search_catalog_names=["exec_command"],
    )
    context = _activation_context(str(_make_activation_workspace()))
    context.state["tool_activation_policy"].update(
        {
            "discoverable_tool_names": ["exec_command"],
            "capability_enforcement_mode": "enforce",
            "activation_allowed_tool_names": ["file_read"],
            "requested_operations": ["workspace_read"],
            "required_capabilities": ["workspace_read"],
        }
    )

    result = registry.request_tool_activation("exec_command", context=context)

    assert result.error is not None
    assert result.metadata["reason"] == "contract_capability_mismatch"
    assert result.metadata["eligibility_source"] == (
        "capability_plan.eligible_tools"
    )
    assert registry.get("exec_command") is None


def test_enforced_capability_plan_allows_required_exec_activation() -> None:
    source_registry = ToolRegistry(discover_builtin=False)
    source_registry.register(_NamedTestTool("exec_command"))
    registry = source_registry.create_view(
        [],
        tool_search_catalog_names=["exec_command"],
    )
    context = _activation_context(str(_make_activation_workspace()))
    context.state["tool_activation_policy"].update(
        {
            "discoverable_tool_names": ["exec_command"],
            "capability_enforcement_mode": "enforce",
            "activation_allowed_tool_names": ["exec_command"],
            "requested_operations": ["execution"],
            "required_capabilities": ["execution"],
        }
    )

    result = registry.request_tool_activation("exec_command", context=context)

    assert result.error is None
    assert result.metadata["status"] == "tool_activated"
    assert result.metadata["activation_authorizes_tool_call"] is False
    assert result.metadata["capability_plan_eligibility"] == "eligible"
    assert registry.get("exec_command") is not None


def test_hard_denylist_precedes_capability_activation_eligibility() -> None:
    source_registry = ToolRegistry(discover_builtin=False)
    source_registry.register(_NamedTestTool("exec_command"))
    registry = source_registry.create_view(
        [],
        tool_search_catalog_names=["exec_command"],
    )
    context = _activation_context(str(_make_activation_workspace()))
    context.state["tool_activation_policy"].update(
        {
            "discoverable_tool_names": ["exec_command"],
            "capability_enforcement_mode": "enforce",
            "activation_allowed_tool_names": ["exec_command"],
            "requested_operations": ["execution"],
            "required_capabilities": ["execution"],
            "tool_denylist": ["exec_command"],
        }
    )

    result = registry.request_tool_activation("exec_command", context=context)

    assert result.error is not None
    assert result.metadata["reason"] == "denylist_blocked"
    assert registry.get("exec_command") is None


def test_hard_denylist_precedes_legacy_mutation_eligibility() -> None:
    workspace = _make_activation_workspace()
    source_registry = ToolRegistry(discover_builtin=False)
    source_registry.register(
        FileWriteTool(workspace_dir=workspace, require_approval=False)
    )
    registry = source_registry.create_view(
        [],
        tool_search_catalog_names=["file_write"],
    )
    context = _activation_context(str(workspace))
    context.state["tool_activation_policy"].update(
        {
            "discoverable_tool_names": ["file_write"],
            "mutation_requirement": "forbidden",
            "requested_operations": ["workspace_read"],
            "required_capabilities": ["workspace_read"],
            "tool_denylist": ["file_write"],
        }
    )

    result = registry.request_tool_activation("file_write", context=context)

    assert result.error is not None
    assert result.metadata["reason"] == "denylist_blocked"
    assert registry.get("file_write") is None


def test_mutation_activation_without_intent_contract_fails_closed() -> None:
    workspace = _make_activation_workspace()
    source_registry = ToolRegistry(discover_builtin=False)
    source_registry.register(
        FileWriteTool(workspace_dir=workspace, require_approval=False)
    )
    registry = source_registry.create_view([], tool_search_catalog_names=["file_write"])
    context = _activation_context(str(workspace))
    context.state["tool_activation_policy"] = {
        "capability_enforcement_mode": "enforce",
        "activation_allowed_tool_names": ["file_write"],
        "execution_profile": "chat",
        "tool_mode": "auto",
        "discoverable_tool_names": ["file_write"],
    }

    result = registry.request_tool_activation("file_write", context=context)

    assert result.error is not None
    assert result.metadata["reason"] == "missing_intent_contract_eligibility"
    assert result.metadata["mutation_eligibility_source"] == "intent_contract"
    assert registry.get("file_write") is None


def test_contract_workspace_write_allows_capability_planned_activation() -> None:
    workspace = _make_activation_workspace()
    source_registry = ToolRegistry(discover_builtin=False)
    source_registry.register(
        FileWriteTool(workspace_dir=workspace, require_approval=False)
    )
    registry = source_registry.create_view([], tool_search_catalog_names=["file_write"])
    context = _activation_context(str(workspace))
    activation_policy = context.state["tool_activation_policy"]
    activation_policy.update(
        {
            "mutation_requirement": "required",
            "requested_operations": ["workspace_write"],
            "required_capabilities": ["workspace_write"],
        }
    )

    result = registry.request_tool_activation("file_write", context=context)

    assert result.error is None
    assert result.metadata["status"] == "tool_activated"
    assert result.metadata["mutation_eligibility_source"] == (
        "intent_contract.required_capabilities"
    )
    assert result.metadata["eligibility_source"] == "capability_plan.eligible_tools"
    assert result.metadata["mutation_eligibility"] == "eligible"
    assert result.metadata["activation_authorizes_tool_call"] is False
    assert registry.get("file_write") is not None


def test_contract_mutation_forbidden_blocks_write_activation() -> None:
    workspace = _make_activation_workspace()
    source_registry = ToolRegistry(discover_builtin=False)
    source_registry.register(
        FileWriteTool(workspace_dir=workspace, require_approval=False)
    )
    registry = source_registry.create_view([], tool_search_catalog_names=["file_write"])
    context = _activation_context(str(workspace))
    context.state["tool_activation_policy"].update(
        {
            "mutation_requirement": "forbidden",
            "requested_operations": ["workspace_write"],
        }
    )

    result = registry.request_tool_activation("file_write", context=context)

    assert result.error is not None
    assert result.metadata["reason"] == "mutation_forbidden_by_contract"
    assert result.metadata["mutation_eligibility_source"] == "intent_contract"
    assert result.metadata["mutation_eligibility"] == "forbidden"
    assert registry.get("file_write") is None


def test_capability_inquiry_contract_does_not_activate_mutation_tool() -> None:
    workspace = _make_activation_workspace()
    source_registry = ToolRegistry(discover_builtin=False)
    source_registry.register(
        FileWriteTool(workspace_dir=workspace, require_approval=False)
    )
    registry = source_registry.create_view([], tool_search_catalog_names=["file_write"])
    context = _activation_context(str(workspace))
    context.state["tool_activation_policy"].update(
        {
            "mutation_requirement": "unknown",
            "requested_operations": ["capability_inquiry"],
            "required_capabilities": ["tool_catalog_read"],
        }
    )

    result = registry.request_tool_activation("file_write", context=context)

    assert result.error is not None
    assert result.metadata["reason"] == "contract_disallows_mutation_activation"
    assert result.metadata["mutation_eligibility_source"] == "intent_contract"
    assert registry.get("file_write") is None


def test_session_auto_review_overrides_cached_strict_activation_hint() -> None:
    workspace = _make_activation_workspace()
    source_registry = ToolRegistry(discover_builtin=False)
    source_registry.register(
        FileWriteTool(workspace_dir=workspace, require_approval=True)
    )
    registry = source_registry.create_view([], tool_search_catalog_names=["file_write"])
    context = _activation_context(str(workspace))
    context.state["tool_activation_policy"].update(
        {
            "mutation_requirement": "required",
            "required_capabilities": ["workspace_write"],
        }
    )

    result = registry.request_tool_activation("file_write", context=context)

    assert result.error is None
    assert result.metadata["authorization_required_for_call"] is False
    assert result.metadata["authorization_policy_source"] == "execution_context"
    assert result.metadata["authorization_state"] == "deferred_to_tool_call"
    assert registry.get("file_write") is not None


def test_activation_approval_hint_falls_back_to_cached_tool_when_context_omits_it() -> None:
    workspace = _make_activation_workspace()
    source_registry = ToolRegistry(discover_builtin=False)
    source_registry.register(
        FileWriteTool(workspace_dir=workspace, require_approval=True)
    )
    registry = source_registry.create_view([], tool_search_catalog_names=["file_write"])
    context = _activation_context(str(workspace))
    context.permission_policy.pop("require_approval_for_file_write")

    result = registry.request_tool_activation("file_write", context=context)

    assert result.error is None
    assert result.metadata["authorization_required_for_call"] is True
    assert result.metadata["authorization_policy_source"] == "cached_tool_fallback"
    assert registry.get("file_write") is not None


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
async def test_strict_hidden_file_write_activation_defers_approval_to_call() -> None:
    tmp_path = _make_activation_workspace()
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
                    "capability_enforcement_mode": "enforce",
                    "mutation_requirement": "required",
                    "requested_operations": ["workspace_write"],
                    "required_capabilities": ["workspace_write"],
                    "activation_allowed_tool_names": ["file_write"],
                    "execution_profile": "chat",
                "tool_mode": "auto",
                "discoverable_tool_names": ["file_write"],
                "tool_allowlist": None,
                "tool_denylist": None,
            }
        },
    )
    activation = registry.request_tool_activation("file_write", context=context)
    call_result = await registry.execute(
        "file_write",
        {"path": "report.md", "content": "# Report\n"},
        context=context,
    )

    assert activation.error is None
    assert activation.metadata["status"] == "tool_activated"
    assert activation.metadata["activation_authorizes_tool_call"] is False
    assert activation.metadata["authorization_required_for_call"] is True
    assert activation.metadata["authorization_policy_source"] == "execution_context"
    assert call_result.error == "File write requires approval."
    assert call_result.metadata["requires_approval"] is True
    assert not (tmp_path / "report.md").exists()

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
