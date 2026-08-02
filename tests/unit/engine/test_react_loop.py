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
async def test_react_loop_stops_at_a_durable_approval_interrupt() -> None:
    from mochi.agents.events import ToolCallCompletedEvent, ToolCallResultEvent
    from mochi.agents.react_loop import AsyncReActLoop
    from mochi.backends.types import ToolCall
    from mochi.tools.base import BaseTool, ToolExecutionContext, ToolResult

    class _ApprovalTool(BaseTool):
        @property
        def name(self) -> str:
            return "file_write"

        @property
        def description(self) -> str:
            return "Creates a durable approval request."

        @property
        def parameters_schema(self) -> dict:
            return {"type": "object", "properties": {}, "additionalProperties": False}

        async def execute(self, **_: object) -> ToolResult:
            return ToolResult(
                error="File write requires approval.",
                metadata={
                    "status": "approval_pending",
                    "approval_id": "tool-approval-1",
                    "operation_id": "file-operation-1",
                    "arguments_digest": "digest-1",
                },
            )

    class _Backend(FakeBackend):
        async def generate(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append(messages)
            self.generation_kwargs.append(dict(kwargs))
            return GenerationResult(
                content="",
                tool_calls=[ToolCall(id="call-1", name="file_write", arguments={})],
                finish_reason="tool_calls",
            )

    registry = ToolRegistry(discover_builtin=False)
    registry.register(_ApprovalTool())
    context = ToolExecutionContext(
        workspace_dir=".",
        state={
            "ordinary_chat_approval_context": {
                "source": "ordinary_chat",
                "resume_cursor": {"turn_id": "turn-1"},
                "tool_registry_view": {
                    "tool_search_catalog_names": ["file_write"],
                    "schema_limit": 4,
                },
            }
        },
    )
    backend = _Backend()
    loop = AsyncReActLoop(
        backend=backend,
        tool_registry=registry,
        tool_execution_context=context,
        max_iterations=3,
    )

    events = [event async for event in loop.run("system", [], "write a file")]

    results = [event for event in events if isinstance(event, ToolCallResultEvent)]
    finals = [event for event in events if isinstance(event, FinalAnswerEvent)]
    assert len(results) == 1
    assert results[0].metadata["approval_id"] == "tool-approval-1"
    assert len(finals) == 1
    assert finals[0].finish_reason == "approval_required"
    assert finals[0].metadata["approval_id"] == "tool-approval-1"
    assert len(backend.calls) == 1
    assert context.state["ordinary_chat_approval_context"]["resume_cursor"] == {
        "turn_id": "turn-1",
        "tool_call_id": "call-1",
        "tool_name": "file_write",
    }
    continuation = context.state["ordinary_chat_approval_context"]["react_continuation"]
    assert continuation["callable_tool_names"] == ["file_write"]
    assert continuation["tool_registry_view"] == {
        "tool_search_catalog_names": ["file_write"],
        "schema_limit": 4,
    }
    assert [message["role"] for message in continuation["messages"]] == [
        "system",
        "user",
        "assistant",
    ]
    assert continuation["messages"][1]["native_tool_protocol_active"] is True


@pytest.mark.asyncio
async def test_react_loop_continues_from_the_original_approval_transcript() -> None:
    from mochi.agents.events import ToolCallResultEvent
    from mochi.agents.react_loop import AsyncReActLoop
    from mochi.backends.types import ToolCall
    from mochi.tools.base import BaseTool, ToolExecutionContext, ToolResult

    class _ApprovalTool(BaseTool):
        @property
        def name(self) -> str:
            return "file_write"

        @property
        def description(self) -> str:
            return "Creates a durable approval request."

        @property
        def parameters_schema(self) -> dict:
            return {"type": "object", "properties": {}, "additionalProperties": False}

        async def execute(self, **_: object) -> ToolResult:
            return ToolResult(
                error="File write requires approval.",
                metadata={"status": "approval_pending", "approval_id": "tool-approval-1"},
            )

    class _Backend(FakeBackend):
        async def generate(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append(messages)
            self.generation_kwargs.append(dict(kwargs))
            if len(self.calls) == 1:
                return GenerationResult(
                    content="",
                    tool_calls=[ToolCall(id="call-1", name="file_write", arguments={})],
                    finish_reason="tool_calls",
                )
            assert messages[-1].role == "tool"
            assert messages[-1].tool_call_id == "call-1"
            assert messages[-1].name == "file_write"
            assert any(
                message.role == "user" and message.native_tool_protocol_active
                for message in messages
            )
            assert any(
                message.role == "assistant" and message.tool_calls[0].id == "call-1"
                for message in messages
            )
            return GenerationResult(content="Saved report.md", finish_reason="stop")

    registry = ToolRegistry(discover_builtin=False)
    registry.register(_ApprovalTool())
    context = ToolExecutionContext(
        workspace_dir=".",
        state={
            "ordinary_chat_approval_context": {
                "source": "ordinary_chat",
                "session_id": "session-1",
                "turn_id": "turn-1",
                "resume_cursor": {"turn_id": "turn-1"},
            }
        },
    )
    backend = _Backend()
    initial_loop = AsyncReActLoop(
        backend=backend,
        tool_registry=registry,
        tool_execution_context=context,
        max_iterations=3,
    )
    _ = [event async for event in initial_loop.run("system", [], "write a file")]
    approval_context = context.state["ordinary_chat_approval_context"]
    checkpoint = {
        "source": "ordinary_chat",
        "tool_name": "file_write",
        "resume_cursor": dict(approval_context["resume_cursor"]),
        "react_continuation": approval_context["react_continuation"],
    }

    resumed_loop = AsyncReActLoop(
        backend=backend,
        tool_registry=registry,
        tool_execution_context=context,
        max_iterations=3,
    )
    events = [
        event
        async for event in resumed_loop.resume_from_ordinary_chat_approval(
            checkpoint=checkpoint,
            tool_result=ToolResult(output={"path": "report.md"}, metadata={"status": "completed"}),
        )
    ]

    results = [event for event in events if isinstance(event, ToolCallResultEvent)]
    finals = [event for event in events if isinstance(event, FinalAnswerEvent)]
    assert len(backend.calls) == 2
    assert len(results) == 1
    assert results[0].metadata["approval_continuation"] is True
    assert len(finals) == 1
    assert finals[0].content == "Saved report.md"
    assert [message.role for message in resumed_loop.turn_messages] == ["tool", "assistant"]


@pytest.mark.asyncio
async def test_react_loop_marks_each_observed_tool_execution_with_an_operation_id() -> None:
    from mochi.agents.events import ToolCallCompletedEvent, ToolCallResultEvent
    from mochi.agents.react_loop import AsyncReActLoop
    from mochi.backends.types import ToolCall
    from mochi.tools.base import BaseTool, ToolExecutionContext, ToolResult
    from mochi.tools.registry import ToolRegistry

    class _ExecTool(BaseTool):
        @property
        def name(self) -> str:
            return "exec_command"

        @property
        def description(self) -> str:
            return "Test-only command runner."

        @property
        def parameters_schema(self) -> dict:
            return {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            }

        async def execute(self, **_: object) -> ToolResult:
            return ToolResult(
                output={"exit_code": 0},
                metadata={"status": "completed"},
            )

    class _Backend(FakeBackend):
        async def generate(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append(messages)
            self.generation_kwargs.append(dict(kwargs))
            if len(self.calls) == 1:
                return GenerationResult(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="exec-call-1",
                            name="exec_command",
                            arguments={"command": "pytest -q"},
                        )
                    ],
                    finish_reason="tool_calls",
                )
            return GenerationResult(content="tests completed", finish_reason="stop")

    registry = ToolRegistry(discover_builtin=False)
    registry.register(_ExecTool())
    events = [
        event
        async for event in AsyncReActLoop(
            backend=_Backend(),
            tool_registry=registry,
            tool_execution_context=ToolExecutionContext(workspace_dir="."),
            max_iterations=3,
        ).run("system", [], "run tests")
    ]

    result = next(event for event in events if isinstance(event, ToolCallResultEvent))
    assert result.metadata["execution_observed"] is True
    assert str(result.metadata["operation_id"]).startswith("tool-execution-")
    assert events.index(result) < next(
        index
        for index, event in enumerate(events)
        if isinstance(event, ToolCallCompletedEvent)
        and event.call_id == result.call_id
    )


@pytest.mark.asyncio
async def test_engine_binds_normal_turn_validation_evidence_before_completion(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    from mochi.agents.engine import AgentEngine
    from mochi.agents.events import ToolCallRequestEvent, ToolCallResultEvent
    from mochi.agents.turn_intent_contract import ActiveTaskState, DeliverableContract
    from mochi.config.schema import MochiConfig

    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db")},
        }
    )
    engine = AgentEngine(config)
    target = tmp_path / "report.md"
    target.write_text("# Report\n", encoding="utf-8")
    active_task = ActiveTaskState(
        goal_id="goal-normal-evidence",
        objective="write and test report",
        operations=frozenset({"workspace_write", "execution"}),
        mutation_requirement="required",
        deliverables=(
            DeliverableContract(
                kind="workspace_artifact",
                target_hint="report.md",
                required=True,
                acceptance_criteria=(
                    "contains:Report",
                    {
                        "schema_version": 1,
                        "kind": "tool_execution",
                        "check": "test",
                        "profile_id": "pytest",
                        "tool_name": "exec_command",
                    },
                ),
                source_turn_ids=("turn-normal-evidence",),
            ),
        ),
        source_turn_ids=("turn-normal-evidence",),
        updated_turn_id="turn-normal-evidence",
    )
    saved = await engine._conversation_state_repository.save(  # noqa: SLF001
        "session-normal-evidence",
        active_task=active_task,
        expected_revision=0,
    )
    assert saved.status == "saved"
    state = await engine._conversation_state_repository.load(  # noqa: SLF001
        "session-normal-evidence"
    )
    mutation_request = ToolCallRequestEvent(
        call_id="write-call",
        tool_name="file_write",
        arguments={"path": "report.md", "content": "# Report\n"},
    )
    mutation_result = ToolCallResultEvent(
        call_id="write-call",
        tool_name="file_write",
        metadata={"resolved_path": str(target)},
    )
    test_request = ToolCallRequestEvent(
        call_id="pytest-call",
        tool_name="exec_command",
        arguments={"command": "pytest -q", "shell": "powershell"},
    )
    test_result = ToolCallResultEvent(
        call_id="pytest-call",
        tool_name="exec_command",
        result={"exit_code": 0},
        metadata={"status": "completed", "operation_id": "pytest-operation-1"},
    )

    receipt, completion_error = await engine._verify_and_complete_active_task(  # noqa: SLF001
        session_id="session-normal-evidence",
        turn_id="turn-normal-evidence",
        workspace_dir=str(tmp_path),
        active_task=active_task,
        state_revision=state.state_revision,
        requests=[mutation_request],
        results=[mutation_result],
        evidence_requests=[mutation_request, test_request],
        evidence_results=[mutation_result, test_result],
    )

    assert completion_error is None
    assert receipt["verification_status"] == "verified"
    assert receipt["targets"][0]["acceptance_checks"][-1]["evidence"][
        "call_id"
    ] == "pytest-call"
    completed = await engine._conversation_state_repository.load(  # noqa: SLF001
        "session-normal-evidence"
    )
    assert completed.active_task is not None
    assert completed.active_task.status == "completed"
    await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "reported_paths",
        "expected_checkpoint_status",
        "expected_task_status",
        "expected_verification_status",
        "acceptance_criteria",
        "profile_registry",
        "runtime_broker",
    ),
    [
        (
            ["report.md"],
            "completed",
            "completed",
            "verified",
            ("contains:Report",),
            None,
            False,
        ),
        (
            ["report.md", "sibling.md"],
            "blocked",
            "active",
            "failed",
            ("contains:Report",),
            None,
            False,
        ),
        (
            ["report.md"],
            "completed",
            "completed",
            "verified",
            (
                "contains:Report",
                {
                    "schema_version": 1,
                    "kind": "tool_execution",
                    "check": "test",
                    "profile_id": "approved-file-write",
                    "tool_name": "file_write",
                },
            ),
            "approved-file-write",
            False,
        ),
        (
            ["report.md"],
            "completed",
            "completed",
            "verified",
            ("contains:Report",),
            None,
            True,
        ),
    ],
    ids=[
        "authorized-target",
        "unexpected-sibling",
        "approval-exact-evidence",
        "runtime-activation-broker",
    ],
)
async def test_engine_resumes_an_ordinary_chat_approval_without_a_new_user_turn(
    tmp_path,
    reported_paths,
    expected_checkpoint_status,
    expected_task_status,
    expected_verification_status,
    acceptance_criteria,
    profile_registry,
    runtime_broker,
) -> None:  # type: ignore[no-untyped-def]
    from dataclasses import replace

    from mochi.agents.conversation_state_store import TurnCheckpoint
    from mochi.agents.artifact_verifier import ValidationProfileRegistry
    from mochi.agents.engine import AgentEngine
    from mochi.agents.turn_intent_contract import ActiveTaskState, DeliverableContract
    from mochi.backends.types import ToolCall
    from mochi.config.schema import MochiConfig
    from mochi.security.policy import EffectivePolicyResolver

    class _Backend(FakeBackend):
        async def generate(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append(messages)
            self.generation_kwargs.append(dict(kwargs))
            assert messages[-1].role == "tool"
            assert messages[-1].tool_call_id == "call-1"
            assert sum(message.role == "user" for message in messages) == 1
            return GenerationResult(content="Saved report.md", finish_reason="stop")

    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db")},
        }
    )
    backend = _Backend(metadata={"effective_context_length": 32768})
    validation_profiles = (
        ValidationProfileRegistry(
            {
                "approved-file-write": lambda tool_name, arguments: (
                    tool_name == "file_write"
                    and arguments.get("path") == "report.md"
                ),
            }
        )
        if profile_registry is not None
        else None
    )
    engine = AgentEngine(config, validation_profile_registry=validation_profiles)

    async def _load(_: str) -> _Backend:
        engine._router._active = backend  # noqa: SLF001
        return backend

    engine._router.load = _load  # type: ignore[method-assign]
    policy = EffectivePolicyResolver().resolve(config.security).to_dict()
    initial_checkpoint = TurnCheckpoint(
        session_id="chat-session-1",
        turn_id="turn-1",
        revision=0,
        stage="contract_resolved",
        turn_intent_contract={"turn_id": "turn-1"},
        capability_plan={
            "artifact_obligation": {"required": True, "ready": True},
        },
        active_goal_id="goal-report",
        policy_snapshot=policy,
        inventory_snapshot={"inventory_version": "inventory-test"},
        activation_state={"activation_allowed_tool_names": ["file_write"]},
        pending_tool_call={
            "call_id": "call-1",
            "tool_name": "file_write",
            "arguments": {"path": "report.md", "content": "# Report\n"},
        },
        resume_cursor={"turn_id": "turn-1", "phase": "approval"},
    )
    active_task = ActiveTaskState(
        goal_id="goal-report",
        objective="write report",
        operations=frozenset({"workspace_write"}),
        mutation_requirement="required",
        deliverables=(
            DeliverableContract(
                kind="workspace_artifact",
                target_hint="report.md",
                required=True,
                acceptance_criteria=acceptance_criteria,
                source_turn_ids=("turn-1",),
            ),
        ),
        source_turn_ids=("turn-1",),
        updated_turn_id="turn-1",
    )
    state_saved = await engine._conversation_state_repository.save(  # noqa: SLF001
        "chat-session-1", active_task=active_task, expected_revision=0,
    )
    assert state_saved.status == "saved"
    saved_initial = await engine._turn_checkpoint_repository.save(  # noqa: SLF001
        initial_checkpoint,
        expected_revision=0,
    )
    assert saved_initial.checkpoint is not None
    saved_pending = await engine._turn_checkpoint_repository.save(  # noqa: SLF001
        replace(
            saved_initial.checkpoint,
            stage="awaiting_approval",
            approval_record={
                "approval_id": "tool-approval-1",
                "status": "pending",
                "tool_name": "file_write",
            },
        ),
        expected_revision=1,
    )
    assert saved_pending.status == "saved"
    payload = {
        "ordinary_chat_checkpoint": {
            "schema_version": 1,
            "source": "ordinary_chat",
            "session_id": "chat-session-1",
            "turn_id": "turn-1",
            "resolved_workspace_dir": str(tmp_path.resolve()),
            "tool_name": "file_write",
            "normalized_arguments": {"path": "report.md", "content": "# Report\n"},
            "operation_id": "artifact-op-test",
            "resume_cursor": {
                "turn_id": "turn-1",
                "tool_call_id": "call-1",
                "tool_name": "file_write",
            },
            "react_continuation": {
                "schema_version": 1,
                "messages": [
                    {"role": "system", "content": "system", "tool_calls": []},
                    {"role": "user", "content": "write report.md", "tool_calls": []},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "name": "file_write",
                                "arguments": {"path": "report.md", "content": "# Report\n"},
                            }
                        ],
                    },
                ],
                "callable_tool_names": (
                    ["file_write", "tool_activate"]
                    if runtime_broker
                    else ["file_write"]
                ),
                "tool_registry_view": (
                    {
                        "tool_search_catalog_names": [
                            "file_write",
                            "exec_command",
                        ],
                        "schema_limit": 8,
                    }
                    if runtime_broker
                    else None
                ),
                "max_iterations": 3,
                "requires_file_mutation": True,
                "tool_activation_policy": (
                    {
                        "activation_allowed_tool_names": ["exec_command"],
                        "discoverable_tool_names": [
                            "file_write",
                            "exec_command",
                        ],
                    }
                    if runtime_broker
                    else None
                ),
                "generation": {"temperature": 0.7, "max_tokens": 128},
            },
        }
    }
    (tmp_path / "report.md").write_text("# Report\n", encoding="utf-8")
    result = await engine.resume_ordinary_chat_approval(
        approval_id="tool-approval-1",
        approval_payload=payload,
        execution_result={
            "status": "completed",
            "tool_name": "file_write",
            "output": {
                "path": "report.md",
                **({"exit_code": 0} if profile_registry is not None else {}),
            },
            "metadata": {"file_changes": [{"path": path} for path in reported_paths]},
        },
        current_permission_policy=policy,
    )

    assert result["status"] == "continued"
    assert result["content"] == "Saved report.md"
    assert result["turn_checkpoint_status"] == expected_checkpoint_status
    assert result["turn_checkpoint_error"] is None
    task_after = await engine._conversation_state_repository.load("chat-session-1")  # noqa: SLF001
    assert task_after.active_task is not None
    assert task_after.active_task.status == expected_task_status
    assert task_after.active_task.deliverables[0].status == (
        "satisfied" if expected_task_status == "completed" else "pending"
    )
    assert len(backend.calls) == 1
    checkpoint_events = await engine._session_store.load_session("chat-session-1")  # noqa: SLF001
    turn_checkpoints = [
        event["checkpoint"]["stage"]
        for event in checkpoint_events
        if event.get("event") == "turn_execution_checkpoint"
        and event.get("turn_id") == "turn-1"
    ]
    assert turn_checkpoints == [
        "contract_resolved",
        "awaiting_approval",
        "executing",
        "verifying",
        expected_checkpoint_status,
    ]
    terminal_checkpoint = next(
        event["checkpoint"]
        for event in reversed(checkpoint_events)
        if event.get("event") == "turn_execution_checkpoint"
        and event.get("turn_id") == "turn-1"
    )
    assert (
        terminal_checkpoint["verification_result"]["verification_status"]
        == expected_verification_status
    )
    assert any(
        event.get("event") == "artifact_verification_receipt"
        for event in checkpoint_events
    )
    await engine.close()


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
