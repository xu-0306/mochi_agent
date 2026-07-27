from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from mochi.agents import engine as engine_module
from mochi.agents.conversation_resolver import (
    BoundedConversationContext,
    ConversationResolver,
    IntentInterpretation,
)
from mochi.agents.engine import AgentEngine
from mochi.agents.events import (
    AgentEvent,
    FinalAnswerEvent,
    StatusEvent,
    ToolCallRequestEvent,
    ToolCallResultEvent,
)
from mochi.agents.plan_ledger import PLAN_LEDGER_VERSION, PlanItem, PlanLedger
from mochi.agents.conversation_state_store import TurnCheckpoint
from mochi.agents.invocation import AgentInvocationRequest
from mochi.agents.tool_exposure import ToolExposurePlan
from mochi.agents.turn_intent_contract import DeliverableContract
from mochi.config.schema import MochiConfig
from mochi.tools.base import ToolExecutionContext
from mochi.tools.registry import ToolRegistry
from tests.unit.engine._support import FakeBackend


class _Interpreter:
    def __init__(self, interpretation: IntentInterpretation) -> None:
        self.interpretation = interpretation
        self.calls = 0

    async def interpret(
        self,
        context: BoundedConversationContext,
    ) -> IntentInterpretation:
        self.calls += 1
        deliverables = tuple(
            replace(
                deliverable,
                source_turn_ids=(context.current_turn.turn_id,),
            )
            if deliverable.source_turn_ids == ("__current__",)
            else deliverable
            for deliverable in self.interpretation.deliverables
        )
        return replace(self.interpretation, deliverables=deliverables)


def _config(tmp_path: Path, *, mode: str) -> MochiConfig:
    return MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
            "agent": {"turn_contract_mode": mode},
        }
    )


def _write_interpretation() -> IntentInterpretation:
    return IntentInterpretation(
        current_speech_act="request_execution",
        task_relation="standalone",
        objective="Write the requested workspace artifact",
        operations=frozenset({"workspace_write"}),
        deliverables=(
            DeliverableContract(
                kind="workspace_file",
                target_hint="report.md",
                source_turn_ids=("__current__",),
            ),
        ),
        mutation_requirement="required",
        confidence=0.99,
    )


async def _bind_backend(engine: AgentEngine, backend: FakeBackend) -> None:
    async def fake_load(model_spec: str) -> FakeBackend:
        del model_spec
        engine._router._active = backend  # noqa: SLF001
        return backend

    engine._router.load = fake_load  # type: ignore[method-assign]


class _RecordingTimeline:
    def __init__(self) -> None:
        self.results: list[dict[str, object]] = []

    async def persist_tool_result(self, **kwargs: object) -> bool:
        self.results.append(dict(kwargs))
        return True

    async def abandon_pre_effect_operation(self, **kwargs: object) -> None:
        self.results.append(dict(kwargs))


@pytest.mark.asyncio
async def test_resolver_failure_is_fail_closed_without_baseline_generation(
    tmp_path: Path,
) -> None:
    backend = FakeBackend(metadata={"effective_context_length": 32768})

    def failing_factory(backend_arg: object) -> ConversationResolver:
        del backend_arg
        raise RuntimeError("resolver unavailable")

    engine = AgentEngine(
        _config(tmp_path, mode="enforce"),
        conversation_resolver_factory=failing_factory,  # type: ignore[arg-type]
    )
    await _bind_backend(engine, backend)

    with pytest.raises(RuntimeError, match="resolver unavailable"):
        await engine.invoke(
            AgentInvocationRequest(
                message="Which file tools exist?",
                tool_mode="auto",
                execution_profile="chat",
                persist_session=False,
            )
        )

    await engine.close()


@pytest.mark.asyncio
async def test_enforce_contract_write_uses_default_workspace_and_required_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeBackend(metadata={"effective_context_length": 32768})
    interpreter = _Interpreter(_write_interpretation())
    captured: dict[str, Any] = {}

    class SpyReActLoop:
        def __init__(
            self,
            *args: object,
            tool_registry: ToolRegistry,
            tool_execution_context: ToolExecutionContext,
            requires_file_mutation: bool,
            **kwargs: object,
        ) -> None:
            del args, kwargs
            captured["tools"] = [
                schema["function"]["name"] for schema in tool_registry.get_schemas()
            ]
            captured["complexity_decision"] = dict(
                tool_execution_context.state.get("complexity_decision", {})
            )
            captured["policy"] = dict(
                tool_execution_context.state["tool_activation_policy"]
            )
            captured["requires_file_mutation"] = requires_file_mutation

        async def run(
            self,
            *args: object,
            **kwargs: object,
        ) -> AsyncIterator[AgentEvent]:
            del args, kwargs
            yield FinalAnswerEvent(content="spy reply")

    engine = AgentEngine(
        _config(tmp_path, mode="enforce"),
        conversation_resolver_factory=lambda backend_arg: ConversationResolver(
            interpreter=interpreter
        ),
    )
    await _bind_backend(engine, backend)
    monkeypatch.setattr(engine_module, "AsyncReActLoop", SpyReActLoop)

    result = await engine.invoke(
        AgentInvocationRequest(
            message="Create report.md",
            tool_mode="auto",
            execution_profile="chat",
            persist_session=False,
        )
    )

    assert result.content == "spy reply"
    assert interpreter.calls == 1
    assert {"file_write", "file_edit", "apply_patch"} & set(captured["tools"])
    assert captured["complexity_decision"]["decision_version"] == "complexity-decision-v1"
    assert captured["complexity_decision"]["turn_id"]
    assert captured["requires_file_mutation"] is True
    assert captured["policy"]["capability_enforcement_mode"] == "enforce"
    assert captured["policy"]["mutation_requirement"] == "required"
    assert captured["policy"]["activation_allowed_tool_names"]
    await engine.close()


@pytest.mark.asyncio
async def test_enforce_mutation_forbidden_excludes_write_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeBackend(metadata={"effective_context_length": 32768})
    interpreter = _Interpreter(
        IntentInterpretation(
            current_speech_act="request_execution",
            task_relation="standalone",
            objective="Explain without changing files",
            operations=frozenset({"workspace_read"}),
            mutation_requirement="forbidden",
            confidence=0.99,
        )
    )
    captured: dict[str, Any] = {}

    class SpyReActLoop:
        def __init__(
            self,
            *args: object,
            tool_registry: ToolRegistry,
            tool_execution_context: ToolExecutionContext,
            requires_file_mutation: bool,
            **kwargs: object,
        ) -> None:
            del args, kwargs
            captured["tools"] = [
                schema["function"]["name"] for schema in tool_registry.get_schemas()
            ]
            captured["policy"] = dict(
                tool_execution_context.state["tool_activation_policy"]
            )
            captured["requires_file_mutation"] = requires_file_mutation

        async def run(
            self,
            *args: object,
            **kwargs: object,
        ) -> AsyncIterator[AgentEvent]:
            del args, kwargs
            yield FinalAnswerEvent(content="spy reply")

    engine = AgentEngine(
        _config(tmp_path, mode="enforce"),
        conversation_resolver_factory=lambda backend_arg: ConversationResolver(
            interpreter=interpreter
        ),
    )
    await _bind_backend(engine, backend)
    monkeypatch.setattr(engine_module, "AsyncReActLoop", SpyReActLoop)

    result = await engine.invoke(
        AgentInvocationRequest(
            message="Do not edit anything; only explain.",
            tool_mode="auto",
            execution_profile="chat",
            persist_session=False,
        )
    )

    risky_writes = {"file_write", "file_edit", "apply_patch"}
    assert result.content == "spy reply"
    assert not (risky_writes & set(captured["tools"]))
    assert captured["requires_file_mutation"] is False
    assert captured["policy"]["mutation_requirement"] == "forbidden"
    assert not (
        risky_writes & set(captured["policy"]["activation_allowed_tool_names"])
    )
    await engine.close()


@pytest.mark.asyncio
async def test_enforce_tool_override_cannot_add_contract_external_tool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeBackend(metadata={"effective_context_length": 32768})
    interpreter = _Interpreter(_write_interpretation())
    captured_tools: list[str] = []

    class SpyReActLoop:
        def __init__(
            self,
            *args: object,
            tool_registry: ToolRegistry,
            **kwargs: object,
        ) -> None:
            del args, kwargs
            captured_tools.extend(
                schema["function"]["name"] for schema in tool_registry.get_schemas()
            )

        async def run(
            self,
            *args: object,
            **kwargs: object,
        ) -> AsyncIterator[AgentEvent]:
            del args, kwargs
            yield FinalAnswerEvent(content="spy reply")

    engine = AgentEngine(
        _config(tmp_path, mode="enforce"),
        conversation_resolver_factory=lambda backend_arg: ConversationResolver(
            interpreter=interpreter
        ),
    )
    await _bind_backend(engine, backend)
    monkeypatch.setattr(engine_module, "AsyncReActLoop", SpyReActLoop)

    await engine.invoke(
        AgentInvocationRequest(
            message="Create report.md",
            tool_mode="auto",
            tool_names_override=[
                "file_write",
                "file_edit",
                "apply_patch",
                "web_search",
            ],
            execution_profile="chat",
            persist_session=False,
        )
    )

    assert {"file_write", "file_edit", "apply_patch"} & set(captured_tools)
    assert "web_search" not in captured_tools
    await engine.close()


@pytest.mark.asyncio
async def test_strict_policy_ceiling_does_not_expose_exec_for_execution_contract(
    tmp_path: Path,
) -> None:
    backend = FakeBackend(metadata={"effective_context_length": 32768})
    interpreter = _Interpreter(
        IntentInterpretation(
            current_speech_act="request_execution",
            task_relation="standalone",
            objective="Run the requested command",
            operations=frozenset({"execution"}),
            mutation_requirement="forbidden",
            confidence=0.99,
        )
    )
    engine = AgentEngine(
        _config(tmp_path, mode="enforce"),
        conversation_resolver_factory=lambda backend_arg: ConversationResolver(
            interpreter=interpreter
        ),
    )
    await _bind_backend(engine, backend)

    result = await engine.invoke(
        AgentInvocationRequest(
            message="Run tests",
            tool_mode="auto",
            permission_policy={"autonomy_mode": "strict"},
            execution_profile="chat",
            persist_session=False,
        )
    )

    assert "exec_command" not in result.diagnostics.exposed_tools
    assert result.diagnostics.fallback_reason == (
        "turn_contract_required_capability_unavailable"
    )
    await engine.close()


@pytest.mark.asyncio
async def test_successful_enforce_artifact_turn_persists_completed_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeBackend(metadata={"effective_context_length": 32768})
    start_interpretation = replace(_write_interpretation(), task_relation="start")
    interpreter = _Interpreter(start_interpretation)
    target = tmp_path / "report.md"

    class SpyReActLoop:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            self.turn_messages: list = []

        async def run(
            self,
            *args: object,
            **kwargs: object,
        ) -> AsyncIterator[AgentEvent]:
            del args, kwargs
            # Keep the on-disk fixture byte-identical to the declared tool input.
            # Text-mode writes translate LF to CRLF on Windows.
            target.write_bytes(b"# Verified report\n")
            yield ToolCallRequestEvent(
                call_id="write-1",
                tool_name="file_write",
                arguments={"path": "report.md", "content": "# Verified report\n"},
            )
            yield ToolCallResultEvent(
                call_id="write-1",
                tool_name="file_write",
                result="report.md",
                metadata={
                    "file_changes": [{"path": "report.md"}],
                    "change_count": 1,
                },
            )
            yield FinalAnswerEvent(content="Saved report.md")

    engine = AgentEngine(
        _config(tmp_path, mode="enforce"),
        conversation_resolver_factory=lambda backend_arg: ConversationResolver(
            interpreter=interpreter
        ),
    )
    await _bind_backend(engine, backend)
    monkeypatch.setattr(engine_module, "AsyncReActLoop", SpyReActLoop)

    result = await engine.invoke(
        AgentInvocationRequest(
            message="Create report.md",
            session_id="durable-artifact",
            tool_mode="auto",
            execution_profile="chat",
            persist_session=True,
        )
    )

    assert result.content == "Saved report.md"
    reloaded = await engine._conversation_state_repository.load(  # noqa: SLF001
        "durable-artifact"
    )
    assert reloaded.active_task is not None
    assert reloaded.active_task.status == "completed"
    assert all(
        deliverable.status == "satisfied"
        for deliverable in reloaded.active_task.deliverables
        if deliverable.required
    )
    events = await engine._session_store.load_session("durable-artifact")  # noqa: SLF001
    receipt_index = next(
        index
        for index, event in enumerate(events)
        if event.get("event") == "artifact_verification_receipt"
    )
    state_index = next(
        index
        for index, event in enumerate(events)
        if event.get("event") == "active_task_state_updated"
        and event.get("active_task_state", {}).get("status") == "completed"
    )
    assert receipt_index < state_index
    checkpoints = [
        event["checkpoint"]
        for event in events
        if event.get("event") == "turn_execution_checkpoint"
    ]
    assert [checkpoint["stage"] for checkpoint in checkpoints] == [
        "contract_resolved",
        "executing",
        "verifying",
        "completed",
    ]
    assert [checkpoint["revision"] for checkpoint in checkpoints] == [1, 2, 3, 4]
    assert checkpoints[0]["turn_intent_contract"]["turn_id"] == result.events[-1].turn_id
    assert checkpoints[0]["complexity_decision"]["decision_version"] == "complexity-decision-v1"
    assert checkpoints[0]["complexity_decision"]["turn_id"] == result.events[-1].turn_id
    assert checkpoints[0]["verification_plan"]["criteria"][0]["kind"] == "artifact"
    assert checkpoints[0]["verification_plan"]["criteria"][0]["payload"]["check"] == "exists"
    assert checkpoints[-1]["verification_result"]["verification_status"] == "verified"
    await engine.close()


@pytest.mark.asyncio
async def test_unverified_mutation_keeps_required_deliverable_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeBackend(metadata={"effective_context_length": 32768})
    interpreter = _Interpreter(replace(_write_interpretation(), task_relation="start"))

    class SpyReActLoop:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            self.turn_messages: list = []

        async def run(
            self,
            *args: object,
            **kwargs: object,
        ) -> AsyncIterator[AgentEvent]:
            del args, kwargs
            yield ToolCallRequestEvent(
                call_id="write-1",
                tool_name="file_write",
                arguments={"path": "missing.md", "content": "claimed only"},
            )
            yield ToolCallResultEvent(
                call_id="write-1",
                tool_name="file_write",
                result="missing.md",
                metadata={"file_changes": [{"path": "missing.md"}]},
            )
            yield FinalAnswerEvent(content="Saved missing.md")

    engine = AgentEngine(
        _config(tmp_path, mode="enforce"),
        conversation_resolver_factory=lambda backend_arg: ConversationResolver(
            interpreter=interpreter
        ),
    )
    await _bind_backend(engine, backend)
    monkeypatch.setattr(engine_module, "AsyncReActLoop", SpyReActLoop)

    result = await engine.invoke(
        AgentInvocationRequest(
            message="Create missing.md",
            session_id="unverified-artifact",
            tool_mode="auto",
            execution_profile="chat",
            persist_session=True,
        )
    )

    final = next(event for event in result.events if isinstance(event, FinalAnswerEvent))
    assert final.metadata["artifact_verification_status"] == "failed"
    reloaded = await engine._conversation_state_repository.load(  # noqa: SLF001
        "unverified-artifact"
    )
    assert reloaded.active_task is not None
    assert reloaded.active_task.status == "active"
    assert all(
        deliverable.status == "pending"
        for deliverable in reloaded.active_task.deliverables
        if deliverable.required
    )
    events = await engine._session_store.load_session("unverified-artifact")  # noqa: SLF001
    receipt_event = next(
        event
        for event in events
        if event.get("event") == "artifact_verification_receipt"
    )
    receipt = receipt_event["artifact_receipt"]
    assert receipt["verification_status"] == "failed"
    assert receipt["retry_disposition"] == "requires_replan"
    await engine.close()


@pytest.mark.asyncio
async def test_timeline_artifact_failure_runs_one_durable_corrective_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeBackend(metadata={"effective_context_length": 32768})
    interpreter = _Interpreter(replace(_write_interpretation(), task_relation="start"))
    timeline = _RecordingTimeline()
    callback_order: list[str] = []
    receipt_seen_before_recovery_status: list[bool] = []
    target = tmp_path / "report.md"

    class SpyReActLoop:
        calls = 0

        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            self.turn_messages: list = []

        async def run(
            self,
            *args: object,
            **kwargs: object,
        ) -> AsyncIterator[AgentEvent]:
            del args, kwargs
            type(self).calls += 1
            if type(self).calls == 1:
                yield ToolCallRequestEvent(
                    call_id="write-first",
                    tool_name="file_write",
                    arguments={"path": "report.md", "content": "first"},
                )
                yield ToolCallResultEvent(
                    call_id="write-first",
                    tool_name="file_write",
                    result="report.md",
                    metadata={
                        "file_changes": [{"path": "report.md"}],
                        "timeline_operation_id": "operation-first",
                        "timeline_result_disposition": "succeeded",
                    },
                )
                yield FinalAnswerEvent(content="Saved report.md")
                return
            assert type(self).calls == 2
            target.write_text("corrected", encoding="utf-8")
            yield ToolCallRequestEvent(
                call_id="write-corrective",
                tool_name="file_write",
                arguments={"path": "report.md", "content": "corrected"},
            )
            yield ToolCallResultEvent(
                call_id="write-corrective",
                tool_name="file_write",
                result="report.md",
                metadata={
                    "file_changes": [{"path": "report.md"}],
                    "timeline_operation_id": "operation-corrective",
                    "timeline_result_disposition": "succeeded",
                },
            )
            yield FinalAnswerEvent(content="Corrected report.md")

    engine = AgentEngine(
        _config(tmp_path, mode="enforce"),
        conversation_resolver_factory=lambda backend_arg: ConversationResolver(
            interpreter=interpreter
        ),
    )
    await _bind_backend(engine, backend)
    monkeypatch.setattr(engine_module, "AsyncReActLoop", SpyReActLoop)

    async def observe(event: AgentEvent) -> None:
        if isinstance(event, StatusEvent) and event.metadata.get("reason") == "controlled_recovery_reserved":
            persisted = await engine._session_store.load_session("controlled-recovery")  # noqa: SLF001
            receipt_seen_before_recovery_status.append(
                any(item.get("event") == "artifact_verification_receipt" for item in persisted)
            )
            callback_order.append("recovery_status")
        elif isinstance(event, ToolCallRequestEvent):
            callback_order.append(event.call_id)

    result = await engine._invoke_shared_runtime(  # noqa: SLF001
        AgentInvocationRequest(
            message="Create report.md",
            session_id="controlled-recovery",
            turn_id="controlled-turn",
            tool_mode="auto",
            execution_profile="chat",
            persist_session=True,
            timeline_coordinator=timeline,
        ),
        event_callback=observe,
    )

    assert result.content == "Corrected report.md"
    assert SpyReActLoop.calls == 2
    assert receipt_seen_before_recovery_status == [True]
    assert callback_order.index("recovery_status") < callback_order.index("write-corrective")
    assert [item["operation_id"] for item in timeline.results] == [
        "operation-first",
        "operation-corrective",
    ]
    assert (
        timeline.results[1]["payload"]["metadata"]["controlled_recovery"][
            "predecessor_operation_id"
        ]
        == "operation-first"
    )
    persisted = await engine._session_store.load_session("controlled-recovery")  # noqa: SLF001
    receipt_indexes = [
        index
        for index, item in enumerate(persisted)
        if item.get("event") == "artifact_verification_receipt"
    ]
    checkpoints = [
        item["checkpoint"]
        for item in persisted
        if item.get("event") == "turn_execution_checkpoint"
    ]
    reserved_index = next(
        index
        for index, item in enumerate(persisted)
        if item.get("event") == "turn_execution_checkpoint"
        and (item["checkpoint"]["execution_receipt"] or {}).get(
            "controlled_recovery", {}
        ).get("status")
        == "reserved"
    )
    reserved_checkpoint = next(
        item
        for item in checkpoints
        if (item["execution_receipt"] or {}).get("controlled_recovery", {}).get(
            "status"
        )
        == "reserved"
    )
    assert receipt_indexes[0] < reserved_index < receipt_indexes[1]
    assert [item["stage"] for item in checkpoints] == [
        "contract_resolved",
        "executing",
        "verifying",
        "verifying",
        "completed",
    ]
    assert reserved_checkpoint["recovery_budget"]["remaining_attempts"] == 0
    recovery = checkpoints[-1]["execution_receipt"]["controlled_recovery"]
    assert checkpoints[-1]["recovery_budget"]["remaining_attempts"] == 0
    assert recovery["status"] == "completed"
    assert recovery["predecessor_operation_id"] == "operation-first"
    assert recovery["successor_operation_id"] == "operation-corrective"
    await engine.close()


@pytest.mark.asyncio
async def test_controlled_recovery_budget_exhaustion_blocks_a_second_model_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeBackend(metadata={"effective_context_length": 32768})
    interpreter = _Interpreter(replace(_write_interpretation(), task_relation="start"))
    timeline = _RecordingTimeline()

    class SpyReActLoop:
        calls = 0

        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            self.turn_messages: list = []

        async def run(self, *args: object, **kwargs: object) -> AsyncIterator[AgentEvent]:
            del args, kwargs
            type(self).calls += 1
            suffix = "first" if type(self).calls == 1 else "corrective"
            yield ToolCallRequestEvent(
                call_id=f"write-{suffix}",
                tool_name="file_write",
                arguments={"path": "report.md", "content": suffix},
            )
            yield ToolCallResultEvent(
                call_id=f"write-{suffix}",
                tool_name="file_write",
                result="report.md",
                metadata={
                    "file_changes": [{"path": "report.md"}],
                    "timeline_operation_id": f"operation-{suffix}",
                    "timeline_result_disposition": "succeeded",
                },
            )
            yield FinalAnswerEvent(content=f"{suffix} answer")

    engine = AgentEngine(
        _config(tmp_path, mode="enforce"),
        conversation_resolver_factory=lambda backend_arg: ConversationResolver(
            interpreter=interpreter
        ),
    )
    await _bind_backend(engine, backend)
    monkeypatch.setattr(engine_module, "AsyncReActLoop", SpyReActLoop)

    await engine.invoke(
        AgentInvocationRequest(
            message="Create report.md",
            session_id="recovery-budget",
            turn_id="budget-turn",
            tool_mode="auto",
            execution_profile="chat",
            persist_session=True,
            timeline_coordinator=timeline,
        )
    )

    assert SpyReActLoop.calls == 2
    loaded = await engine._turn_checkpoint_repository.load("recovery-budget", "budget-turn")  # noqa: SLF001
    assert loaded.checkpoint is not None
    assert loaded.checkpoint.stage == "blocked"
    assert loaded.checkpoint.blocker_reason == "controlled_recovery_budget_exhausted"
    assert loaded.checkpoint.recovery_budget["remaining_attempts"] == 0
    recovery = loaded.checkpoint.execution_receipt["controlled_recovery"]
    assert recovery["status"] == "blocked"
    assert recovery["replans_used"] == 1
    await engine.close()


@pytest.mark.asyncio
async def test_reserved_controlled_recovery_reentry_blocks_without_model_or_tool(
    tmp_path: Path,
) -> None:
    backend = FakeBackend(metadata={"effective_context_length": 32768})
    interpreter = _Interpreter(_write_interpretation())
    engine = AgentEngine(
        _config(tmp_path, mode="enforce"),
        conversation_resolver_factory=lambda backend_arg: ConversationResolver(
            interpreter=interpreter
        ),
    )
    await _bind_backend(engine, backend)
    initial = TurnCheckpoint(
        session_id="recovery-reentry",
        turn_id="reserved-turn",
        revision=0,
        stage="contract_resolved",
        turn_intent_contract={},
        capability_plan={},
    )
    saved = await engine._turn_checkpoint_repository.save(initial, expected_revision=0)  # noqa: SLF001
    assert saved.checkpoint is not None
    executing, error = await engine._transition_turn_checkpoint(  # noqa: SLF001
        saved.checkpoint,
        stage="executing",
    )
    assert error is None and executing is not None
    verifying, error = await engine._transition_turn_checkpoint(  # noqa: SLF001
        executing,
        stage="verifying",
        execution_receipt={
            "controlled_recovery": {
                "schema_version": 1,
                "max_replans": 1,
                "replans_used": 1,
                "status": "reserved",
                "predecessor_operation_id": "operation-first",
            }
        },
    )
    assert error is None and verifying is not None

    result = await engine.invoke(
        AgentInvocationRequest(
            message="Create report.md",
            session_id="recovery-reentry",
            turn_id="reserved-turn",
            tool_mode="auto",
            execution_profile="chat",
            persist_session=True,
        )
    )

    assert backend.calls == []
    assert result.events[-1].finish_reason == (
        "controlled_recovery_reservation_requires_manual_replan"
    )
    loaded = await engine._turn_checkpoint_repository.load("recovery-reentry", "reserved-turn")  # noqa: SLF001
    assert loaded.checkpoint is not None
    assert loaded.checkpoint.stage == "blocked"
    await engine.close()


@pytest.mark.asyncio
async def test_turn_checkpoint_round_trips_plan_ledger_snapshot_from_public_repository(
    tmp_path: Path,
) -> None:
    engine = AgentEngine(_config(tmp_path, mode="enforce"))
    ledger = PlanLedger(
        ledger_version=PLAN_LEDGER_VERSION,
        ledger_id="plan-goal-1",
        session_id="checkpoint-plan-session",
        goal_id="goal-1",
        revision=1,
        status="active",
        objective="Build the selected artifact",
        reason_codes=("multiple_deliverables",),
        items=(
            PlanItem(
                item_id="item-1",
                title="Inspect inputs",
                status="completed",
                dependencies=(),
                success_criteria=("done",),
                source_turn_ids=("turn-1",),
                evidence_refs=("receipt-1",),
            ),
            PlanItem(
                item_id="item-2",
                title="Apply the fix",
                status="in_progress",
                dependencies=("item-1",),
                success_criteria=("done",),
                source_turn_ids=("turn-1",),
            ),
        ),
        created_turn_id="turn-1",
        updated_turn_id="turn-2",
    )
    checkpoint = TurnCheckpoint(
        session_id="checkpoint-plan-session",
        turn_id="turn-2",
        revision=0,
        stage="contract_resolved",
        turn_intent_contract={"turn_id": "turn-2"},
        capability_plan={"plan_version": "capability-plan-v1"},
        plan_ledger_snapshot=ledger.to_dict(),
        verification_plan={"criteria": [{"kind": "artifact"}]},
    )

    saved = await engine._turn_checkpoint_repository.save(  # noqa: SLF001
        checkpoint,
        expected_revision=0,
    )

    assert saved.status == "saved"
    loaded = await engine._turn_checkpoint_repository.load(  # noqa: SLF001
        "checkpoint-plan-session",
        "turn-2",
    )
    assert loaded.diagnostics.status == "loaded"
    assert loaded.checkpoint is not None
    assert loaded.checkpoint.plan_ledger_snapshot == ledger.to_dict()
    assert loaded.checkpoint.plan_ledger_snapshot["items"][1]["status"] == "in_progress"
    assert loaded.checkpoint.plan_ledger_snapshot["reason_codes"] == ["multiple_deliverables"]
    await engine.close()


@pytest.mark.asyncio
async def test_unknown_timeline_predecessor_never_runs_corrective_model_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeBackend(metadata={"effective_context_length": 32768})
    interpreter = _Interpreter(replace(_write_interpretation(), task_relation="start"))
    timeline = _RecordingTimeline()

    class SpyReActLoop:
        calls = 0

        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            self.turn_messages: list = []

        async def run(self, *args: object, **kwargs: object) -> AsyncIterator[AgentEvent]:
            del args, kwargs
            type(self).calls += 1
            yield ToolCallRequestEvent(
                call_id="write-first",
                tool_name="file_write",
                arguments={"path": "report.md", "content": "first"},
            )
            yield ToolCallResultEvent(
                call_id="write-first",
                tool_name="file_write",
                result="report.md",
                metadata={
                    "file_changes": [{"path": "report.md"}],
                    "timeline_operation_id": "operation-unknown",
                    "timeline_result_disposition": "unknown",
                    "timeline_result_unknown": True,
                },
            )
            yield FinalAnswerEvent(content="Saved report.md")

    engine = AgentEngine(
        _config(tmp_path, mode="enforce"),
        conversation_resolver_factory=lambda backend_arg: ConversationResolver(
            interpreter=interpreter
        ),
    )
    await _bind_backend(engine, backend)
    monkeypatch.setattr(engine_module, "AsyncReActLoop", SpyReActLoop)

    await engine.invoke(
        AgentInvocationRequest(
            message="Create report.md",
            session_id="unknown-recovery",
            turn_id="unknown-turn",
            tool_mode="auto",
            execution_profile="chat",
            persist_session=True,
            timeline_coordinator=timeline,
        )
    )

    assert SpyReActLoop.calls == 1
    loaded = await engine._turn_checkpoint_repository.load("unknown-recovery", "unknown-turn")  # noqa: SLF001
    assert loaded.checkpoint is not None
    assert loaded.checkpoint.blocker_reason == "side_effect_outcome_unknown"
    await engine.close()


def test_missing_timeline_result_disposition_is_not_treated_as_success() -> None:
    operation, error = AgentEngine._recovery_operation_from_events(  # noqa: SLF001
        [
            ToolCallResultEvent(
                call_id="write-first",
                tool_name="file_write",
                result="report.md",
                metadata={"timeline_operation_id": "operation-first"},
            )
        ]
    )

    assert operation is None
    assert error == "timeline_result_disposition_missing"


@pytest.mark.asyncio
async def test_unexpected_reported_changed_path_keeps_required_deliverable_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeBackend(metadata={"effective_context_length": 32768})
    interpreter = _Interpreter(replace(_write_interpretation(), task_relation="start"))
    target = tmp_path / "report.md"
    sibling = tmp_path / "sibling.md"

    class SpyReActLoop:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            self.turn_messages: list = []

        async def run(
            self,
            *args: object,
            **kwargs: object,
        ) -> AsyncIterator[AgentEvent]:
            del args, kwargs
            target.write_bytes(b"# Verified report\n")
            sibling.write_bytes(b"unrelated\n")
            yield ToolCallRequestEvent(
                call_id="write-1",
                tool_name="file_write",
                arguments={"path": "report.md", "content": "# Verified report\n"},
            )
            yield ToolCallResultEvent(
                call_id="write-1",
                tool_name="file_write",
                result="report.md",
                metadata={
                    "file_changes": [
                        {"path": "report.md"},
                        {"path": "sibling.md"},
                    ]
                },
            )
            yield FinalAnswerEvent(content="Saved report.md")

    engine = AgentEngine(
        _config(tmp_path, mode="enforce"),
        conversation_resolver_factory=lambda backend_arg: ConversationResolver(
            interpreter=interpreter
        ),
    )
    await _bind_backend(engine, backend)
    monkeypatch.setattr(engine_module, "AsyncReActLoop", SpyReActLoop)

    result = await engine.invoke(
        AgentInvocationRequest(
            message="Create report.md",
            session_id="unexpected-artifact-scope",
            tool_mode="auto",
            execution_profile="chat",
            persist_session=True,
        )
    )

    final = next(event for event in result.events if isinstance(event, FinalAnswerEvent))
    assert final.metadata["artifact_verification_status"] == "failed"
    reloaded = await engine._conversation_state_repository.load(  # noqa: SLF001
        "unexpected-artifact-scope"
    )
    assert reloaded.active_task is not None
    assert reloaded.active_task.status == "active"
    assert all(
        deliverable.status == "pending"
        for deliverable in reloaded.active_task.deliverables
        if deliverable.required
    )
    await engine.close()


@pytest.mark.asyncio
async def test_concurrent_durable_turns_keep_isolated_checkpoints_and_reject_stale_task_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeBackend(metadata={"effective_context_length": 32768})

    class BarrierInterpreter:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.calls = 0

        async def interpret(
            self,
            context: BoundedConversationContext,
        ) -> IntentInterpretation:
            self.calls += 1
            if self.calls == 2:
                self.started.set()
            await self.started.wait()
            await self.release.wait()
            return replace(
                _write_interpretation(),
                task_relation="start",
                objective=f"Write {context.current_turn.content}",
                deliverables=(
                    DeliverableContract(
                        kind="workspace_file",
                        target_hint=context.current_turn.content,
                        source_turn_ids=(context.current_turn.turn_id,),
                    ),
                ),
            )

    class SpyReActLoop:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            self.turn_messages: list = []

        async def run(
            self,
            *args: object,
            **kwargs: object,
        ) -> AsyncIterator[AgentEvent]:
            del args, kwargs
            yield FinalAnswerEvent(content="No mutation performed.")

    interpreter = BarrierInterpreter()
    engine = AgentEngine(
        _config(tmp_path, mode="enforce"),
        conversation_resolver_factory=lambda backend_arg: ConversationResolver(
            interpreter=interpreter
        ),
    )
    await _bind_backend(engine, backend)
    monkeypatch.setattr(engine_module, "AsyncReActLoop", SpyReActLoop)

    first = asyncio.create_task(
        engine.invoke(
            AgentInvocationRequest(
                message="first.md",
                session_id="concurrent-turns",
                turn_id="turn-one",
                tool_mode="auto",
                execution_profile="chat",
                persist_session=True,
            )
        )
    )
    second = asyncio.create_task(
        engine.invoke(
            AgentInvocationRequest(
                message="second.md",
                session_id="concurrent-turns",
                turn_id="turn-two",
                tool_mode="auto",
                execution_profile="chat",
                persist_session=True,
            )
        )
    )
    await asyncio.wait_for(interpreter.started.wait(), timeout=2)
    interpreter.release.set()
    first_result, second_result = await asyncio.gather(first, second)

    assert interpreter.calls == 2
    assert {
        first_result.content,
        second_result.content,
    } & {"No mutation performed."}
    assert any(
        event.metadata.get("error_type") == "turn_contract_state_persist_failed"
        for result in (first_result, second_result)
        for event in result.events
        if isinstance(event, FinalAnswerEvent)
    )
    events = await engine._session_store.load_session("concurrent-turns")  # noqa: SLF001
    checkpoints_by_turn: dict[str, list[str]] = {}
    for event in events:
        if event.get("event") != "turn_execution_checkpoint":
            continue
        turn_id = event["turn_id"]
        checkpoints_by_turn.setdefault(turn_id, []).append(event["checkpoint"]["stage"])
    assert checkpoints_by_turn["turn-one"][0:2] == ["contract_resolved", "executing"]
    assert checkpoints_by_turn["turn-two"][0:2] == ["contract_resolved", "executing"]
    assert any(stages[-1] == "blocked" for stages in checkpoints_by_turn.values())
    reloaded = await engine._conversation_state_repository.load(  # noqa: SLF001
        "concurrent-turns"
    )
    assert reloaded.active_task is not None
    assert reloaded.active_task.updated_turn_id in {"turn-one", "turn-two"}
    await engine.close()


def test_continuation_reader_does_not_overflow_broker_schema_budget() -> None:
    tool_names = [f"tool_{index}" for index in range(9)]
    plan = ToolExposurePlan(
        tool_names=tool_names,
        matched_groups=[],
        limit=10,
        discoverable_tool_names=[
            *tool_names,
            "tool_result_read",
            "another_deferred_tool",
        ],
    )
    context = ToolExecutionContext(
        tool_result_references={"result-1": {"reference_id": "result-1"}}
    )
    engine = object.__new__(AgentEngine)

    result = engine._preserve_tool_result_read_for_continuation(  # noqa: SLF001
        plan,
        available_tool_names=[*plan.discoverable_tool_names],
        tool_execution_context=context,
    )

    assert "tool_result_read" in result.tool_names
    assert len(result.tool_names) < len(tool_names) + 1
    deferred = set(result.discoverable_tool_names) - set(result.tool_names)
    assert len(result.tool_names) + int(bool(deferred)) <= result.limit


@pytest.mark.asyncio
async def test_side_question_preserves_durable_task_without_exposing_write_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_tools: list[list[str]] = []

    class SpyReActLoop:
        def __init__(
            self,
            *args: object,
            tool_registry: ToolRegistry,
            **kwargs: object,
        ) -> None:
            del args, kwargs
            self.turn_messages: list = []
            captured_tools.append(
                [
                    schema["function"]["name"]
                    for schema in tool_registry.get_schemas()
                ]
            )

        async def run(
            self,
            *args: object,
            **kwargs: object,
        ) -> AsyncIterator[AgentEvent]:
            del args, kwargs
            yield FinalAnswerEvent(
                content="No file was written yet.",
                metadata={"error_type": "file_artifact_not_mutated"},
            )

    monkeypatch.setattr(engine_module, "AsyncReActLoop", SpyReActLoop)
    config = _config(tmp_path, mode="enforce")
    first_backend = FakeBackend(metadata={"effective_context_length": 32768})
    first_interpreter = _Interpreter(
        replace(_write_interpretation(), task_relation="start")
    )
    first_engine = AgentEngine(
        config,
        conversation_resolver_factory=lambda backend_arg: ConversationResolver(
            interpreter=first_interpreter
        ),
    )
    await _bind_backend(first_engine, first_backend)
    await first_engine.invoke(
        AgentInvocationRequest(
            message="Create report.md",
            session_id="side-question-reload",
            tool_mode="auto",
            execution_profile="chat",
            persist_session=True,
        )
    )
    await first_engine.close()

    second_backend = FakeBackend(metadata={"effective_context_length": 32768})
    second_interpreter = _Interpreter(
        IntentInterpretation(
            current_speech_act="side_question",
            task_relation="side_question",
            objective="List the available tools",
            operations=frozenset({"tool_discovery"}),
            mutation_requirement="forbidden",
            confidence=0.99,
        )
    )
    second_engine = AgentEngine(
        config,
        conversation_resolver_factory=lambda backend_arg: ConversationResolver(
            interpreter=second_interpreter
        ),
    )
    await _bind_backend(second_engine, second_backend)
    await second_engine.invoke(
        AgentInvocationRequest(
            message="Which tools are available?",
            session_id="side-question-reload",
            tool_mode="auto",
            execution_profile="chat",
            persist_session=True,
        )
    )

    risky_writes = {"file_write", "file_edit", "apply_patch"}
    assert not (risky_writes & set(captured_tools[-1]))
    reloaded = await second_engine._conversation_state_repository.load(  # noqa: SLF001
        "side-question-reload"
    )
    assert reloaded.active_task is not None
    assert reloaded.active_task.status == "active"
    assert reloaded.active_task.mutation_requirement == "required"
    assert any(
        deliverable.required and deliverable.status == "pending"
        for deliverable in reloaded.active_task.deliverables
    )
    await second_engine.close()
