"""Package A adversarial gates for the Wave 2 ordinary-Chat runtime."""

from __future__ import annotations

from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from mochi.agents.complexity_gate import (
    ComplexityActivePlanSummary,
    ComplexityCapabilitySummary,
    ComplexityGate,
    ComplexityGateRequest,
)
from mochi.agents.engine import AgentEngine
from mochi.agents.conversation_state_store import TurnCheckpoint
from mochi.agents.react_loop import AsyncReActLoop
from mochi.agents.tool_exposure import ToolExposurePlan
from mochi.agents.turn_intent_contract import (
    DeliverableContract,
    IntentEvidence,
    TurnIntentContract,
)
from mochi.backends.types import GenerationResult, ToolCall
from mochi.config.schema import MochiConfig
from mochi.tools.base import BaseTool, ToolExecutionContext, ToolResult
from mochi.tools.registry import ToolRegistry
from tests.unit.engine._support import FakeBackend, FileWriteProbeTool


def _config(tmp_path: Path, *, recovery: dict[str, object]) -> MochiConfig:
    return MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db")},
            "agent": {
                "ordinary_chat_adaptive_runtime": {
                    "recovery": recovery,
                },
            },
        }
    )


def test_weak_model_effectful_call_is_blocked_until_plan_and_then_stays_bounded() -> None:
    registry = ToolRegistry(discover_builtin=False)
    write_tool = FileWriteProbeTool()
    registry.register(write_tool)
    context = ToolExecutionContext(
        state={
            "plan_runtime": {
                "enabled": True,
                "required": True,
                "state": "required",
                "preplan_read_calls_used": 0,
                "max_preplan_read_calls": 2,
                "plan_corrections_used": 0,
                "max_plan_prompt_corrections": 1,
            }
        }
    )
    loop = AsyncReActLoop(
        backend=FakeBackend(),
        tool_registry=registry,
        tool_execution_context=context,
    )
    call = ToolCall(
        id="weak-model-write",
        name="file_write",
        arguments={"path": "report.md", "content": "unsafe"},
    )

    first = loop._build_plan_guarded_tool_result(  # pyright: ignore[reportPrivateUsage] # noqa: SLF001
        tool_call=call,
        batch_plan_guard_state={},
    )
    second = loop._build_plan_guarded_tool_result(  # pyright: ignore[reportPrivateUsage] # noqa: SLF001
        tool_call=call,
        batch_plan_guard_state={},
    )

    assert first is not None and first.error
    assert first.metadata["error_type"] == "plan_required_before_effect"
    assert first.retryable is True
    assert second is not None and second.error
    assert second.metadata["error_type"] == "plan_required_before_effect"
    assert second.retryable is False
    assert context.state["plan_runtime"]["plan_corrections_used"] == 1


@pytest.mark.asyncio
async def test_checkpoint_uses_configured_recovery_budget(tmp_path: Path) -> None:
    engine = AgentEngine(
        _config(
            tmp_path,
            recovery={
                "max_attempts": 2,
                "max_extra_model_calls": 2,
                "max_extra_tool_calls": 7,
                "max_extra_wall_seconds": 13.5,
            },
        )
    )
    contract = _equivalent_contract(objective="Update report.md and verify it.")
    rollout = SimpleNamespace(
        capability_plan=SimpleNamespace(
            eligible_tools=("file_write",),
            to_dict=lambda: {"eligible_tools": ["file_write"]},
        ),
        resolution=SimpleNamespace(
            contract=contract,
            next_active_task=None,
        ),
    )
    checkpoint = engine._build_turn_checkpoint(  # noqa: SLF001
        session_id="budget-session",
        turn_id="budget-turn",
        rollout=rollout,
        exposure_plan=ToolExposurePlan(
            tool_names=["file_write"],
            matched_groups=["workspace"],
            limit=1,
            discoverable_tool_names=["file_write"],
        ),
        tool_execution_context=ToolExecutionContext(
            state={"tool_activation_policy": {"activation_allowed_tool_names": []}},
        ),
    )

    assert checkpoint.recovery_budget == {
        "budget_version": "recovery-budget-v1",
        "remaining_attempts": 2,
        "remaining_extra_model_calls": 2,
        "remaining_extra_tool_calls": 7,
        "remaining_extra_wall_seconds": 13.5,
    }
    await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("signal_key", "expected_reason", "cleared_after_persist"),
    [
        ("dynamic_complexity_read_observed", "dynamic_read_to_effectful", False),
        ("dynamic_complexity_verifier_failed", "dynamic_verifier_failed", True),
    ],
)
async def test_engine_dynamic_recheck_persists_before_react_plan_guard(
    tmp_path: Path,
    signal_key: str,
    expected_reason: str,
    cleared_after_persist: bool,
) -> None:
    class UpdatePlanProbe(BaseTool):
        @property
        def name(self) -> str:
            return "update_plan"

        @property
        def description(self) -> str:
            return "Create the durable task plan."

        @property
        def parameters_schema(self) -> dict[str, object]:
            return {"type": "object", "properties": {}}

        async def execute(self, **_: object) -> ToolResult:
            return ToolResult(output={"status": "saved"})

    engine = AgentEngine(_config(tmp_path, recovery={}))
    contract = TurnIntentContract(
        turn_id="dynamic-turn",
        active_goal_id="dynamic-goal",
        objective="Update report.md",
        current_speech_act="request_execution",
        operations=frozenset({"workspace_write"}),
        deliverables=(
            DeliverableContract(
                kind="workspace_file",
                target_hint="report.md",
                acceptance_criteria=("exists",),
                source_turn_ids=("dynamic-turn",),
            ),
        ),
        resolved_references=(),
        positive_constraints=(),
        negative_constraints=(),
        mutation_requirement="required",
        clarification=None,
        supersedes_previous_goal=False,
        cancels_active_goal=False,
        modifies_active_task=True,
        confidence=0.99,
        evidence=(),
        advisories=(),
    )
    rollout = SimpleNamespace(
        capability_plan=SimpleNamespace(eligible_tools=("file_write",)),
        resolution=SimpleNamespace(
            contract=contract,
            context=SimpleNamespace(active_task=None),
            next_active_task=SimpleNamespace(goal_id="dynamic-goal"),
        ),
    )
    initial = ComplexityGate().evaluate_deterministic(
        ComplexityGateRequest(
            turn_intent=contract,
            task_relation="start",
            capability_summary=ComplexityCapabilitySummary(effectful_tool_count=1),
        )
    )
    persisted: list[dict[str, object]] = []
    persist_outcomes = [False, True]

    async def persist(decision: dict[str, object]) -> bool:
        persisted.append(decision)
        return persist_outcomes.pop(0)

    context = ToolExecutionContext(
        state={
            "complexity_decision": initial.to_dict(),
            "dynamic_complexity_checkpoint_persist": persist,
            "plan_runtime": {
                "enabled": False,
                "state": "inactive",
                "required": False,
                "exposed": False,
                "mutable": False,
                "goal_id": "dynamic-goal",
                "ledger_id": None,
            },
        }
    )
    engine._install_dynamic_complexity_recheck(  # noqa: SLF001
        session_id="dynamic-session",
        turn_id="dynamic-turn",
        rollout=rollout,
        available_tools=[FileWriteProbeTool(), UpdatePlanProbe()],
        exposure_plan=ToolExposurePlan(
            tool_names=["file_write"],
            matched_groups=["workspace"],
            limit=2,
            discoverable_tool_names=[],
        ),
        active_plan_summary=None,
        complexity_decision=initial.to_dict(),
        tool_execution_context=context,
    )

    registry = ToolRegistry(discover_builtin=False)
    registry.register(FileWriteProbeTool())
    loop = AsyncReActLoop(
        backend=FakeBackend(),
        tool_registry=registry,
        tool_execution_context=context,
    )
    first_effect = await loop._dynamic_complexity_recheck(  # noqa: SLF001
        tool_call=ToolCall(
            id="first-safe-write",
            name="file_write",
            arguments={"path": "report.md", "content": "safe"},
        ),
        completed_iterations=0,
    )
    assert first_effect is None
    assert persisted == []
    assert context.state["plan_runtime"]["required"] is False
    context.state[signal_key] = True
    failed_persist = await loop._dynamic_complexity_recheck(  # noqa: SLF001
        tool_call=ToolCall(
            id="dynamic-write",
            name="file_write",
            arguments={"path": "report.md", "content": "unsafe"},
        ),
        completed_iterations=1,
    )
    assert failed_persist is not None
    assert failed_persist.metadata["error_type"] == "dynamic_plan_recheck_persist_failed"
    assert context.state[signal_key] is True
    dynamic_result = await loop._dynamic_complexity_recheck(  # noqa: SLF001
        tool_call=ToolCall(
            id="dynamic-write-retry",
            name="file_write",
            arguments={"path": "report.md", "content": "unsafe"},
        ),
        completed_iterations=1,
    )
    assert dynamic_result is None
    assert persisted and persisted[-1]["kind"] == "plan_required"
    assert expected_reason in persisted[-1]["soft_reason_codes"]
    assert context.state["complexity_decision"]["kind"] == "plan_required"
    assert (
        signal_key not in context.state
        if cleared_after_persist
        else context.state[signal_key] is True
    )
    assert context.state["plan_runtime"]["required"] is True
    assert "update_plan" in {schema.name for schema in loop._collect_tool_schemas()}  # noqa: SLF001
    blocked = loop._build_plan_guarded_tool_result(  # noqa: SLF001
        tool_call=ToolCall(
            id="dynamic-write",
            name="file_write",
            arguments={"path": "report.md", "content": "unsafe"},
        ),
        batch_plan_guard_state={},
    )
    assert blocked is not None
    assert blocked.metadata["error_type"] == "plan_required_before_effect"
    await engine.close()


@pytest.mark.asyncio
async def test_restart_restores_durably_upgraded_dynamic_plan_decision(tmp_path: Path) -> None:
    engine = AgentEngine(_config(tmp_path, recovery={}))
    contract = _equivalent_contract(objective="Update report.md and verify it.")
    upgraded = ComplexityGate().evaluate_deterministic(
        ComplexityGateRequest(
            turn_intent=contract,
            task_relation="start",
            capability_summary=ComplexityCapabilitySummary(effectful_tool_count=2),
        )
    )
    checkpoint = TurnCheckpoint(
        session_id="restart-session",
        turn_id="restart-turn",
        revision=2,
        stage="executing",
        turn_intent_contract=contract.to_dict(),
        capability_plan={"eligible_tools": ["file_write"]},
        active_goal_id="goal-1",
        complexity_decision=upgraded.to_dict(),
        resume_cursor={"turn_id": "restart-turn", "phase": "react"},
    )
    context = ToolExecutionContext(state={})

    engine._restore_plan_runtime_from_checkpoint(  # noqa: SLF001
        checkpoint=checkpoint,
        turn_id="restart-turn",
        tool_execution_context=context,
    )

    assert context.state["complexity_decision"]["kind"] == "plan_required"
    assert context.state["plan_runtime"]["required"] is True
    assert context.state["plan_runtime"]["state"] == "required"
    await engine.close()


@pytest.mark.asyncio
async def test_dynamic_checkpoint_cas_failure_blocks_before_registry_execution() -> None:
    class Probe(FileWriteProbeTool):
        calls = 0

        async def execute(self, **kwargs: object) -> ToolResult:
            type(self).calls += 1
            return await super().execute(**kwargs)

    initial = ComplexityGate().evaluate_deterministic(
        ComplexityGateRequest(
            turn_intent=TurnIntentContract(
                turn_id="cas-turn",
                active_goal_id="cas-goal",
                objective="Update report.md",
                current_speech_act="request_execution",
                operations=frozenset({"workspace_write"}),
                deliverables=(
                    DeliverableContract(
                        kind="workspace_file",
                        target_hint="report.md",
                        acceptance_criteria=("exists",),
                        source_turn_ids=("cas-turn",),
                    ),
                ),
                resolved_references=(),
                positive_constraints=(),
                negative_constraints=(),
                mutation_requirement="required",
                clarification=None,
                supersedes_previous_goal=False,
                cancels_active_goal=False,
                modifies_active_task=True,
                confidence=1.0,
                evidence=(),
                advisories=(),
            ),
            task_relation="start",
            capability_summary=ComplexityCapabilitySummary(effectful_tool_count=1),
        )
    )

    async def cas_conflict(**_: object) -> dict[str, object]:
        return {"status": "persistence_failed"}

    probe = Probe()
    registry = ToolRegistry(discover_builtin=False)
    registry.register(probe)
    context = ToolExecutionContext(
        state={
            "complexity_decision": initial.to_dict(),
            "dynamic_complexity_recheck": cas_conflict,
            "dynamic_complexity_read_observed": True,
            "dynamic_complexity_plan_event": {
                "kind": "stale_revision",
                "ledger_id": "ledger-1",
                "observed_revision": 2,
            },
        }
    )
    loop = AsyncReActLoop(
        backend=FakeBackend(), tool_registry=registry, tool_execution_context=context
    )
    result = await loop._dynamic_complexity_recheck(  # noqa: SLF001
        tool_call=ToolCall(
            id="cas-write",
            name="file_write",
            arguments={"path": "report.md", "content": "unsafe"},
        ),
        completed_iterations=1,
    )

    assert result is not None
    assert result.metadata["error_type"] == "dynamic_plan_recheck_persist_failed"
    assert context.state["dynamic_complexity_plan_event"]["kind"] == "stale_revision"
    assert Probe.calls == 0


@pytest.mark.asyncio
async def test_react_run_refreshes_update_plan_schema_after_dynamic_upgrade() -> None:
    class UpdatePlanProbe(BaseTool):
        @property
        def name(self) -> str:
            return "update_plan"

        @property
        def description(self) -> str:
            return "Create a durable plan."

        @property
        def parameters_schema(self) -> dict[str, object]:
            return {"type": "object", "properties": {}}

        async def execute(self, **_: object) -> ToolResult:
            return ToolResult(output={"status": "saved"})

    class Backend(FakeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.schema_names: list[list[str]] = []

        async def generate(self, messages, tools=None, **kwargs):  # type: ignore[no-untyped-def]
            self.schema_names.append([tool.name for tool in tools or []])
            if len(self.schema_names) == 1:
                return GenerationResult(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="dynamic-write",
                            name="file_write",
                            arguments={"path": "report.md", "content": "unsafe"},
                        )
                    ],
                )
            return GenerationResult(content="blocked until plan")

    initial = ComplexityGate().evaluate_deterministic(
        ComplexityGateRequest(
            turn_intent=TurnIntentContract(
                turn_id="refresh-turn",
                active_goal_id="refresh-goal",
                objective="Update report.md",
                current_speech_act="request_execution",
                operations=frozenset({"workspace_write"}),
                deliverables=(
                    DeliverableContract(
                        kind="workspace_file",
                        target_hint="report.md",
                        acceptance_criteria=("exists",),
                        source_turn_ids=("refresh-turn",),
                    ),
                ),
                resolved_references=(),
                positive_constraints=(),
                negative_constraints=(),
                mutation_requirement="required",
                clarification=None,
                supersedes_previous_goal=False,
                cancels_active_goal=False,
                modifies_active_task=True,
                confidence=1.0,
                evidence=(),
                advisories=(),
            ),
            task_relation="start",
            capability_summary=ComplexityCapabilitySummary(effectful_tool_count=1),
        )
    )
    update_plan = UpdatePlanProbe()
    context = ToolExecutionContext(
        state={
            "complexity_decision": initial.to_dict(),
            "dynamic_complexity_read_observed": True,
            "dynamic_complexity_update_plan_tool": update_plan,
            "plan_runtime": {"enabled": False, "required": False, "exposed": False},
        }
    )

    async def durable_upgrade(**_: object) -> dict[str, object]:
        context.state["plan_runtime"].update(
            {"enabled": True, "required": True, "exposed": True, "state": "required"}
        )
        return {"status": "saved"}

    context.state["dynamic_complexity_recheck"] = durable_upgrade
    registry = ToolRegistry(discover_builtin=False)
    registry.register(FileWriteProbeTool())
    backend = Backend()
    loop = AsyncReActLoop(
        backend=backend, tool_registry=registry, tool_execution_context=context, max_iterations=2
    )

    _ = [event async for event in loop.run("system", [], "update report")]

    assert backend.schema_names[0] == ["file_write"]
    assert "update_plan" in backend.schema_names[1]


@pytest.mark.asyncio
async def test_stale_plan_revision_is_durably_replanned_before_next_iteration_effect(
    tmp_path: Path,
) -> None:
    class StaleUpdatePlanProbe(BaseTool):
        @property
        def name(self) -> str:
            return "update_plan"

        @property
        def description(self) -> str:
            return "Refresh the durable task plan."

        @property
        def parameters_schema(self) -> dict[str, object]:
            return {"type": "object", "properties": {}}

        async def execute(self, **_: object) -> ToolResult:
            return ToolResult(
                error="plan revision changed",
                metadata={
                    "error_type": "stale_plan_revision",
                    "current_revision": 2,
                },
            )

    class Probe(FileWriteProbeTool):
        calls = 0

        async def execute(self, **kwargs: object) -> ToolResult:
            type(self).calls += 1
            return await super().execute(**kwargs)

    class Backend(FakeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        async def generate(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            self.calls += 1
            if self.calls == 1:
                return GenerationResult(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="stale-plan",
                            name="update_plan",
                            arguments={},
                        )
                    ],
                )
            if self.calls == 2:
                return GenerationResult(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="blocked-write",
                            name="file_write",
                            arguments={"path": "report.md", "content": "unsafe"},
                        )
                    ],
                )
            return GenerationResult(content="replan required")

    engine = AgentEngine(_config(tmp_path, recovery={}))
    contract = _equivalent_contract(objective="Update report.md and verify it.")
    active_plan = ComplexityActivePlanSummary(
        ledger_id="ledger-1", status="active", revision=1
    )
    initial = ComplexityGate().evaluate_deterministic(
        ComplexityGateRequest(
            turn_intent=contract,
            task_relation="continue",
            capability_summary=ComplexityCapabilitySummary(effectful_tool_count=1),
            active_plan=active_plan,
        )
    )
    assert initial.kind == "continue_existing_plan"
    persisted: list[dict[str, object]] = []

    async def persist(decision: dict[str, object]) -> bool:
        persisted.append(decision)
        return True

    rollout = SimpleNamespace(
        capability_plan=SimpleNamespace(eligible_tools=("file_write", "update_plan")),
        resolution=SimpleNamespace(
            contract=contract,
            context=SimpleNamespace(active_task=SimpleNamespace(goal_id="goal-1")),
            next_active_task=SimpleNamespace(goal_id="goal-1"),
        ),
    )
    context = ToolExecutionContext(
        state={
            "complexity_decision": initial.to_dict(),
            "dynamic_complexity_checkpoint_persist": persist,
            "plan_runtime": {
                "enabled": True,
                "state": "active",
                "required": True,
                "exposed": True,
                "mutable": True,
                "goal_id": "goal-1",
                "ledger_id": "ledger-1",
                "current_revision": 1,
                "plan_corrections_used": 0,
                "max_plan_prompt_corrections": 1,
            },
            "plan_ledger_snapshot": {
                "ledger_id": "ledger-1",
                "revision": 1,
                "items": [],
            },
        }
    )
    update_plan = StaleUpdatePlanProbe()
    write_probe = Probe()
    engine._install_dynamic_complexity_recheck(  # noqa: SLF001
        session_id="stale-session",
        turn_id=contract.turn_id,
        rollout=rollout,
        available_tools=[write_probe, update_plan],
        exposure_plan=ToolExposurePlan(
            tool_names=["file_write", "update_plan"],
            matched_groups=["workspace"],
            limit=2,
            discoverable_tool_names=[],
        ),
        active_plan_summary=active_plan,
        complexity_decision=initial.to_dict(),
        tool_execution_context=context,
    )
    registry = ToolRegistry(discover_builtin=False)
    registry.register(write_probe)
    registry.register(update_plan)
    events = [
        event
        async for event in AsyncReActLoop(
            backend=Backend(),
            tool_registry=registry,
            tool_execution_context=context,
            max_iterations=3,
        ).run("system", [], "update report")
    ]

    assert persisted and persisted[0]["kind"] == "plan_required"
    assert "active_plan_invalidated" in persisted[0]["hard_reason_codes"]
    assert context.state["complexity_decision"]["kind"] == "plan_required"
    assert "dynamic_complexity_plan_event" not in context.state
    assert Probe.calls == 0
    blocked = [
        event
        for event in events
        if getattr(event, "metadata", {}).get("error_type")
        == "plan_required_before_effect"
    ]
    assert blocked
    await engine.close()


def test_recovery_runtime_guard_enforces_model_tool_and_wall_limits() -> None:
    context = ToolExecutionContext(
        state={
            "controlled_recovery_budget_runtime": {
                "started_at": time.perf_counter(),
                "model_calls_limit": 1,
                "model_calls_used": 0,
                "tool_calls_limit": 1,
                "tool_calls_used": 0,
                "wall_seconds_limit": 30.0,
            }
        }
    )
    loop = AsyncReActLoop(
        backend=FakeBackend(),
        tool_execution_context=context,
    )
    call = ToolCall(
        id="recovery-tool",
        name="file_write",
        arguments={"path": "report.md", "content": "ready"},
    )

    assert loop._consume_recovery_model_call() is None  # noqa: SLF001
    assert loop._consume_recovery_model_call() == (
        "controlled_recovery_model_budget_exhausted"
    )  # noqa: SLF001
    assert loop._consume_recovery_tool_call(call) is None  # noqa: SLF001
    blocked = loop._consume_recovery_tool_call(call)  # noqa: SLF001
    assert blocked is not None
    assert blocked.metadata["error_type"] == "controlled_recovery_tool_budget_exhausted"

    context.state["controlled_recovery_budget_runtime"]["started_at"] = (
        time.perf_counter() - 31.0
    )
    assert loop._consume_recovery_model_call() == (
        "controlled_recovery_wall_budget_exhausted"
    )  # noqa: SLF001


def _equivalent_contract(*, objective: str) -> TurnIntentContract:
    return TurnIntentContract(
        turn_id="multilingual-turn",
        active_goal_id="goal-1",
        objective=objective,
        current_speech_act="request_execution",
        operations=frozenset({"workspace_write", "execution"}),
        deliverables=(
            DeliverableContract(
                kind="workspace_file",
                target_hint="report.md",
                acceptance_criteria=("exists", "contains:ready"),
                source_turn_ids=("multilingual-turn",),
            ),
        ),
        resolved_references=(),
        positive_constraints=(),
        negative_constraints=(),
        mutation_requirement="required",
        clarification=None,
        supersedes_previous_goal=False,
        cancels_active_goal=False,
        modifies_active_task=True,
        confidence=0.99,
        evidence=(
            IntentEvidence(
                statement="The resolver returned a validated execution contract.",
                source="current_turn",
                source_turn_ids=("multilingual-turn",),
            ),
        ),
        advisories=(),
    )


def test_equivalent_multilingual_contracts_have_the_same_deterministic_gate_result() -> None:
    objectives = (
        "Update report.md and run the validation; it must contain ready.",
        "更新 report.md 並執行驗證；內容必須包含 ready。",
        "更新 report.md 并运行验证；内容必须包含 ready。",
        "report.md を更新して検証を実行し、ready を含める。",
    )
    decisions = [
        ComplexityGate().evaluate_deterministic(
            ComplexityGateRequest(
                turn_intent=_equivalent_contract(objective=objective),
                capability_summary=ComplexityCapabilitySummary(
                    effectful_tool_count=2,
                ),
            )
        )
        for objective in objectives
    ]

    fingerprints = [
        (
            decision.kind,
            decision.score,
            decision.hard_reason_codes,
            decision.soft_reason_codes,
            decision.effectful_action_requires_plan,
        )
        for decision in decisions
    ]
    assert len(set(fingerprints)) == 1
    assert decisions[0].kind == "plan_required"


def test_recovery_scope_escape_is_rejected_before_registry_execution(tmp_path: Path) -> None:
    registry = ToolRegistry(discover_builtin=False)
    probe = FileWriteProbeTool()
    registry.register(probe)
    context = ToolExecutionContext(
        workspace_dir=str(tmp_path),
        state={"controlled_recovery_allowed_targets": ["report.md"]},
    )
    loop = AsyncReActLoop(backend=FakeBackend(), tool_registry=registry, tool_execution_context=context)
    call = ToolCall(id="escape", name="file_write", arguments={"path": "unrelated.py", "content": "bad"})
    result = loop._recovery_scope_tool_result(call, probe)  # noqa: SLF001
    assert result is not None
    assert result.metadata["error_type"] == "controlled_recovery_scope_violation"
    assert not (tmp_path / "unrelated.py").exists()


@pytest.mark.asyncio
async def test_react_loop_recovery_scope_blocks_before_probe_execution(tmp_path: Path) -> None:
    class Probe(FileWriteProbeTool):
        calls = 0
        async def execute(self, **kwargs: object) -> ToolResult:
            type(self).calls += 1
            return await super().execute(**kwargs)
    class Backend(FakeBackend):
        calls = 0
        async def generate(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append(messages); type(self).calls += 1
            if type(self).calls == 1:
                return GenerationResult(content="", tool_calls=[ToolCall(id="escape", name="file_write", arguments={"path": "unrelated.py", "content": "bad"})], finish_reason="tool_calls")
            return GenerationResult(content="blocked")
    probe, backend = Probe(), Backend()
    registry = ToolRegistry(discover_builtin=False); registry.register(probe)
    context = ToolExecutionContext(workspace_dir=str(tmp_path), state={"controlled_recovery_allowed_targets": ["report.md"], "controlled_recovery_budget_runtime": {"tool_calls_used": 0, "tool_calls_limit": 1, "model_calls_used": 0, "model_calls_limit": 2, "started_at": time.perf_counter(), "wall_seconds_limit": 30}})
    loop = AsyncReActLoop(backend=backend, tool_registry=registry, tool_execution_context=context, max_iterations=2)
    from mochi.agents.events import ToolCallResultEvent
    events = [event async for event in loop.run("system", [], "repair")]
    result = next(event for event in events if isinstance(event, ToolCallResultEvent))
    assert result.metadata["error_type"] == "controlled_recovery_scope_violation"
    assert Probe.calls == 0 and not (tmp_path / "unrelated.py").exists()
    assert context.state["controlled_recovery_budget_runtime"]["tool_calls_used"] == 1
