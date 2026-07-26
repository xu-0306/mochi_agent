from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any

import pytest

from mochi.agents.conversation_resolver import (
    BoundedConversationContext,
    ConversationResolver,
    ConversationSummary,
    ConversationTurn,
)
from mochi.agents.model_conversation_interpreter import (
    INTERPRETATION_JSON_SCHEMA,
    ModelConversationInterpreter,
)
from mochi.agents.turn_intent_contract import (
    ActiveTaskState,
    DeliverableContract,
    TurnIntentContract,
)
from mochi.backends.base import BaseLLMBackend
from mochi.backends.types import (
    GenerationResult,
    Message,
    ModelInfo,
    StreamChunk,
    ToolSchema,
)


class _FakeBackend(BaseLLMBackend):
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[list[Message]] = []
        self.tools_seen: list[list[ToolSchema] | None] = []
        self.kwargs: list[dict[str, Any]] = []

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
        reasoning_effort: str | None = None,
        stream: bool = False,
    ) -> GenerationResult | AsyncIterator[StreamChunk]:
        self.calls.append(messages)
        self.tools_seen.append(tools)
        self.kwargs.append(
            {
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
        return GenerationResult(content=self.content, model="fake-interpreter")

    def supports_tool_calling(self) -> bool:
        return False

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name="fake", backend_type="test")

    async def health_check(self) -> bool:
        return True


def _response(**updates: Any) -> str:
    payload: dict[str, Any] = {
        "current_speech_act": "request_information",
        "task_relation": "standalone",
        "objective": None,
        "operations": ["conversation"],
        "deliverables": [],
        "resolved_references": [],
        "positive_constraints": [],
        "negative_constraints": [],
        "mutation_requirement": "unknown",
        "clarification": None,
        "confidence": 0.9,
        "evidence": [
            {
                "statement": "The current turn directly states the request.",
                "source": "current_turn",
                "source_turn_ids": ["turn-now"],
            }
        ],
    }
    payload.update(updates)
    return json.dumps(payload, ensure_ascii=False)


def _active_task() -> ActiveTaskState:
    return ActiveTaskState(
        goal_id="goal-existing",
        objective="Implement the selected design",
        operations=frozenset({"workspace_read", "workspace_write"}),
        mutation_requirement="required",
        deliverables=(
            DeliverableContract(
                kind="workspace_artifact",
                target_hint="src/feature.py",
                source_turn_ids=("turn-choice",),
            ),
        ),
        decisions=("Use design B",),
        source_turn_ids=("turn-choice",),
        updated_turn_id="turn-choice",
    )


def _context(
    *, active_task: ActiveTaskState | None = None
) -> BoundedConversationContext:
    return BoundedConversationContext(
        current_turn=ConversationTurn("turn-now", "user", "current request"),
        recent_history=(
            ConversationTurn("turn-choice", "assistant", "Design B was selected."),
        ),
        summary=ConversationSummary(
            "The user compared two designs.", source_turn_ids=("turn-summary",)
        ),
        active_task=active_task,
        omitted_history_count=3,
        truncated_fields=("summary.content",),
    )


@pytest.mark.asyncio
async def test_model_interpreter_uses_bounded_history_summary_and_active_task() -> None:
    backend = _FakeBackend(
        _response(
            current_speech_act="request_execution",
            task_relation="continue",
            objective=None,
            operations=["execution"],
            resolved_references=[
                {
                    "surface": "that one",
                    "resolved_to": "design B",
                    "status": "resolved",
                    "source_turn_ids": ["turn-choice"],
                }
            ],
            evidence=[
                {
                    "statement": "The active decision resolves the abbreviated request.",
                    "source": "active_task",
                    "source_turn_ids": ["turn-choice"],
                }
            ],
        )
    )
    resolver = ConversationResolver(interpreter=ModelConversationInterpreter(backend))

    resolution = await resolver.resolve(
        current_turn=ConversationTurn("turn-now", "user", "Proceed with that one"),
        recent_history=[
            ConversationTurn("turn-choice", "assistant", "Design B was selected."),
        ],
        summary=ConversationSummary(
            "The user compared two designs.", source_turn_ids=("turn-summary",)
        ),
        active_task=_active_task(),
    )

    assert resolution.diagnostics["interpreter_status"] == "accepted"
    assert resolution.contract.operations == frozenset(
        {"workspace_read", "workspace_write", "execution"}
    )
    assert resolution.contract.deliverables == _active_task().deliverables
    assert resolution.contract.resolved_references[0].resolved_to == "design B"

    assert len(backend.calls) == 1
    messages = backend.calls[0]
    assert [message.role for message in messages] == ["system", "user"]
    assert "JSON Schema" in messages[0].content
    payload = json.loads(messages[1].content)
    assert payload["current_turn"]["content"] == "Proceed with that one"
    assert payload["recent_history"][0]["turn_id"] == "turn-choice"
    assert payload["summary"]["source_turn_ids"] == ["turn-summary"]
    assert payload["active_task"]["decisions"] == ["Use design B"]
    assert "advisories" not in messages[1].content
    assert backend.tools_seen == [None]
    assert backend.kwargs[0]["temperature"] == 0.0
    assert backend.kwargs[0]["stream"] is False


@pytest.mark.asyncio
async def test_model_interpreter_preserves_composite_operations_and_deliverable() -> (
    None
):
    backend = _FakeBackend(
        _response(
            current_speech_act="request_execution",
            task_relation="start",
            objective="Inspect, repair, and verify the workspace",
            operations=["workspace_read", "workspace_write", "execution"],
            mutation_requirement="required",
            deliverables=[
                {
                    "kind": "workspace_patch",
                    "target_hint": "src/feature.py",
                    "required": True,
                    "acceptance_criteria": ["tests pass"],
                    "status": "pending",
                    "source_turn_ids": ["turn-now"],
                }
            ],
        )
    )

    interpretation = await ModelConversationInterpreter(backend).interpret(_context())

    assert interpretation.operations == frozenset(
        {"workspace_read", "workspace_write", "execution"}
    )
    assert interpretation.mutation_requirement == "required"
    assert interpretation.deliverables[0].target_hint == "src/feature.py"
    assert interpretation.deliverables[0].acceptance_criteria == ("tests pass",)


@pytest.mark.asyncio
async def test_model_interpreter_carries_structured_validation_to_turn_contract() -> None:
    criterion = {
        "schema_version": 1,
        "kind": "tool_execution",
        "check": "test",
        "tool_name": "exec_command",
        "profile_id": "pytest",
        "expected_exit_code": 0,
    }
    backend = _FakeBackend(
        _response(
            current_speech_act="request_execution",
            task_relation="start",
            objective="Write and test the report",
            operations=["workspace_write", "execution"],
            mutation_requirement="required",
            deliverables=[
                {
                    "kind": "workspace_artifact",
                    "target_hint": "report.md",
                    "required": True,
                    "acceptance_criteria": [criterion],
                    "status": "pending",
                    "source_turn_ids": ["turn-now"],
                }
            ],
        )
    )
    resolver = ConversationResolver(interpreter=ModelConversationInterpreter(backend))

    resolution = await resolver.resolve(
        current_turn=ConversationTurn("turn-now", "user", "Write and test report.md")
    )

    assert resolution.diagnostics["interpreter_status"] == "accepted"
    actual = resolution.contract.deliverables[0].acceptance_criteria[0]
    assert dict(actual) == criterion
    assert TurnIntentContract.from_dict(resolution.contract.to_dict()) == resolution.contract
    schema_items = INTERPRETATION_JSON_SCHEMA["properties"]["deliverables"]["items"][
        "properties"
    ]["acceptance_criteria"]["items"]
    assert len(schema_items["oneOf"]) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("criterion", "expected_error"),
    [
        (
            {
                "schema_version": 1,
                "kind": "tool_execution",
                "check": "test",
                "tool_name": "exec_command",
                "profile_id": "pytest",
                "command": "pytest -q",
            },
            "exact fields",
        ),
        (
            {
                "schema_version": 2,
                "kind": "tool_execution",
                "check": "test",
                "tool_name": "exec_command",
                "profile_id": "pytest",
            },
            "unsupported",
        ),
        (
            {
                "schema_version": 1,
                "kind": "network_execution",
                "check": "test",
                "tool_name": "exec_command",
                "profile_id": "pytest",
            },
            "unsupported",
        ),
    ],
    ids=["extra-command-field", "future-schema", "unknown-kind"],
)
async def test_model_interpreter_rejects_invalid_structured_acceptance_criterion(
    criterion: dict[str, object],
    expected_error: str,
) -> None:
    backend = _FakeBackend(
        _response(
            current_speech_act="request_execution",
            task_relation="start",
            objective="Write and test the report",
            operations=["workspace_write", "execution"],
            mutation_requirement="required",
            deliverables=[
                {
                    "kind": "workspace_artifact",
                    "target_hint": "report.md",
                    "required": True,
                    "acceptance_criteria": [criterion],
                    "status": "pending",
                    "source_turn_ids": ["turn-now"],
                }
            ],
        )
    )
    resolver = ConversationResolver(interpreter=ModelConversationInterpreter(backend))

    resolution = await resolver.resolve(
        current_turn=ConversationTurn("turn-now", "user", "Write and test report.md")
    )

    assert resolution.diagnostics["interpreter_status"] == "rejected"
    assert expected_error in resolution.diagnostics["interpreter_error"]
    assert resolution.contract.clarification_needed is True


@pytest.mark.asyncio
async def test_model_interpreter_reopens_current_turn_satisfied_write_deliverable() -> None:
    backend = _FakeBackend(
        _response(
            current_speech_act="request_execution",
            task_relation="start",
            objective="Create the requested report",
            operations=["workspace_write"],
            mutation_requirement="required",
            deliverables=[
                {
                    "kind": "workspace_artifact",
                    "target_hint": "report.md",
                    "required": True,
                    "acceptance_criteria": ["report exists"],
                    "status": "satisfied",
                    "source_turn_ids": ["turn-now"],
                }
            ],
        )
    )

    interpretation = await ModelConversationInterpreter(backend).interpret(_context())

    assert interpretation.deliverables[0].status == "pending"


@pytest.mark.asyncio
async def test_side_question_preserves_completed_durable_task_without_reopening_it() -> None:
    backend = _FakeBackend(
        _response(
            current_speech_act="side_question",
            task_relation="side_question",
            objective="Describe the completed output",
            operations=["workspace_read"],
            mutation_requirement="forbidden",
        )
    )
    pending = _active_task()
    completed = replace(
        pending,
        status="completed",
        deliverables=(replace(pending.deliverables[0], status="satisfied"),),
    )
    resolver = ConversationResolver(interpreter=ModelConversationInterpreter(backend))

    resolution = await resolver.resolve(
        current_turn=ConversationTurn("turn-now", "user", "What was produced?"),
        active_task=completed,
    )

    assert resolution.contract.deliverables == ()
    assert resolution.next_active_task is completed
    assert resolution.next_active_task.status == "completed"
    assert resolution.next_active_task.deliverables[0].status == "satisfied"


@pytest.mark.asyncio
async def test_side_question_does_not_inherit_active_task_into_current_contract() -> (
    None
):
    backend = _FakeBackend(
        _response(
            current_speech_act="side_question",
            task_relation="side_question",
            objective="Explain the available catalog",
            operations=["tool_discovery"],
            mutation_requirement="forbidden",
        )
    )
    active = _active_task()
    resolver = ConversationResolver(interpreter=ModelConversationInterpreter(backend))

    resolution = await resolver.resolve(
        current_turn=ConversationTurn("turn-now", "user", "What is available?"),
        recent_history=[
            ConversationTurn("turn-choice", "assistant", "Design B was selected."),
        ],
        active_task=active,
    )

    assert resolution.contract.current_speech_act == "side_question"
    assert resolution.contract.operations == frozenset({"tool_discovery"})
    assert resolution.contract.deliverables == ()
    assert resolution.contract.mutation_requirement == "forbidden"
    assert resolution.next_active_task == active


@pytest.mark.asyncio
async def test_negative_constraint_is_structured_without_keyword_routing() -> None:
    backend = _FakeBackend(
        _response(
            current_speech_act="constraint",
            task_relation="standalone",
            objective="Analyze the workspace without changing it",
            operations=["workspace_read"],
            mutation_requirement="forbidden",
            negative_constraints=[
                {
                    "text": "Do not modify workspace files",
                    "source_turn_ids": ["turn-now"],
                }
            ],
        )
    )

    interpretation = await ModelConversationInterpreter(backend).interpret(_context())

    assert interpretation.operations == frozenset({"workspace_read"})
    assert interpretation.mutation_requirement == "forbidden"
    assert (
        interpretation.negative_constraints[0].text == "Do not modify workspace files"
    )


@pytest.mark.asyncio
async def test_invalid_json_reaches_resolver_fail_closed_path() -> None:
    backend = _FakeBackend("not valid JSON")
    active = _active_task()
    resolver = ConversationResolver(interpreter=ModelConversationInterpreter(backend))

    resolution = await resolver.resolve(
        current_turn=ConversationTurn("turn-now", "user", "Do it"),
        recent_history=[
            ConversationTurn("turn-choice", "assistant", "Design B was selected."),
        ],
        active_task=active,
    )

    assert resolution.diagnostics["interpreter_status"] == "rejected"
    assert "valid JSON object" in resolution.diagnostics["interpreter_error"]
    assert resolution.contract.operations == frozenset()
    assert resolution.contract.clarification is not None
    assert (
        resolution.contract.clarification.reason_code
        == "semantic_interpretation_unavailable"
    )
    assert resolution.next_active_task == active


@pytest.mark.asyncio
async def test_schema_validation_failure_also_fails_closed() -> None:
    payload = json.loads(_response())
    payload["operations"] = ["workspace_write", "invented_operation"]
    backend = _FakeBackend(json.dumps(payload))
    resolver = ConversationResolver(interpreter=ModelConversationInterpreter(backend))

    resolution = await resolver.resolve(
        current_turn=ConversationTurn("turn-now", "user", "Do something"),
    )

    assert resolution.diagnostics["interpreter_status"] == "rejected"
    assert "unsupported operations item" in resolution.diagnostics["interpreter_error"]
    assert resolution.contract.operations == frozenset()
    assert resolution.contract.clarification_needed is True
