from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mochi.agents import engine as engine_module
from mochi.agents.adaptive_diagnostics import AdaptiveDiagnosticsAccumulator
from mochi.agents.complexity_gate import COMPLEXITY_ADVISOR_RESPONSE_VERSION
from mochi.agents.conversation_resolver import (
    BoundedConversationContext,
    ConversationResolver,
    IntentInterpretation,
)
from mochi.agents.engine import AgentEngine
from mochi.agents.events import (
    AgentEvent,
    FinalAnswerEvent,
    ToolCallRequestEvent,
    ToolCallResultEvent,
)
from mochi.agents.outcome_verifier import VerificationCriterion
from mochi.agents.plan_ledger import PLAN_LEDGER_EVENT, PLAN_LEDGER_VERSION, PlanItem, PlanLedger
from mochi.agents.tool_exposure import ToolExposurePlan
from mochi.agents.turn_intent_contract import (
    ActiveTaskState,
    DeliverableContract,
    TurnIntentContract,
)
from mochi.agents.invocation import AgentInvocationRequest
from mochi.backends.types import GenerationResult
from mochi.config.schema import MochiConfig
from mochi.tools.base import BaseTool, ToolExecutionContext, ToolResult
from mochi.tools.file_ops import FileWriteTool
from tests.unit.engine._support import EchoTool, FakeBackend


def _config(
    tmp_path: Path,
    *,
    adaptive_runtime: dict[str, Any] | None = None,
) -> MochiConfig:
    return MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db")},
            "agent": {
                "ordinary_chat_adaptive_runtime": adaptive_runtime or {},
            },
        }
    )


def _contract(
    *,
    turn_id: str = "turn-1",
    deliverables: tuple[DeliverableContract, ...] = (),
    operations: frozenset[str] | None = None,
) -> TurnIntentContract:
    resolved_operations = operations or frozenset({"workspace_write"})
    return TurnIntentContract(
        turn_id=turn_id,
        active_goal_id=None,
        objective="Write one requested artifact",
        current_speech_act="request_execution",
        operations=resolved_operations,
        deliverables=deliverables,
        resolved_references=(),
        positive_constraints=(),
        negative_constraints=(),
        mutation_requirement=(
            "required" if "workspace_write" in resolved_operations else "forbidden"
        ),
        clarification=None,
        supersedes_previous_goal=False,
        cancels_active_goal=False,
        modifies_active_task=True,
        confidence=0.99,
        evidence=(),
    )


def _rollout(
    contract: TurnIntentContract,
    *,
    eligible_tools: tuple[str, ...] = (),
    goal_id: str = "goal-1",
) -> Any:
    return SimpleNamespace(
        mode="enforce",
        capability_plan=SimpleNamespace(eligible_tools=eligible_tools),
        resolution=SimpleNamespace(
            contract=contract,
            context=SimpleNamespace(active_task=None),
            next_active_task=SimpleNamespace(goal_id=goal_id),
        ),
    )


def _plan_item(
    item_id: str,
    status: str,
    *,
    dependencies: tuple[str, ...] = (),
    evidence_refs: tuple[str, ...] = (),
) -> PlanItem:
    return PlanItem(
        item_id=item_id,
        title=f"Item {item_id}",
        status=status,  # type: ignore[arg-type]
        dependencies=dependencies,
        success_criteria=("done",),
        source_turn_ids=("turn-1",),
        evidence_refs=evidence_refs,
    )


def _ledger(
    *,
    session_id: str,
    status: str = "active",
    items: tuple[PlanItem, ...],
) -> PlanLedger:
    durably_terminal = status in {"completed", "cancelled"}
    return PlanLedger(
        ledger_version=PLAN_LEDGER_VERSION,
        ledger_id="plan-goal-1",
        session_id=session_id,
        goal_id="goal-1",
        revision=1 if durably_terminal else 0,
        status=status,  # type: ignore[arg-type]
        objective="Write one requested artifact",
        reason_codes=("multiple_deliverables",),
        items=items,
        created_turn_id="turn-0",
        updated_turn_id="turn-1" if durably_terminal else "turn-0",
    )


class _ApprovalEffectTool(BaseTool):
    def __init__(
        self,
        name: str,
        *,
        boundary_known: bool,
        destructive: bool = False,
    ) -> None:
        self._name = name
        self._boundary_known = boundary_known
        self._destructive = destructive

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "Approval-gated workspace mutation."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    @property
    def requires_approval(self) -> bool:
        return True

    @property
    def is_destructive(self) -> bool:
        return self._destructive

    @property
    def supports_timeline_side_effect_boundary(self) -> bool:
        return self._boundary_known

    @property
    def tool_capabilities(self) -> dict[str, Any]:
        return {
            "capabilities": ["workspace_write"],
            "destructive": self._destructive,
        }

    async def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(output=kwargs)


class _JudgeBackend(FakeBackend):
    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__()
        self._payload = payload
        self.raw_tools: list[Any] = []

    async def generate(self, messages, tools=None, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(messages)
        self.raw_tools.append(tools)
        self.generation_kwargs.append(dict(kwargs))
        return GenerationResult(content=json.dumps(self._payload), model="judge")


class _SemanticWriteInterpreter:
    async def interpret(
        self,
        context: BoundedConversationContext,
    ) -> IntentInterpretation:
        return IntentInterpretation(
            current_speech_act="request_execution",
            task_relation="standalone",
            objective="Write and explain report.md",
            operations=frozenset({"workspace_write"}),
            deliverables=(
                DeliverableContract(
                    kind="workspace_file",
                    target_hint="report.md",
                    acceptance_criteria=("exists", "explain correctness"),
                    source_turn_ids=(context.current_turn.turn_id,),
                ),
            ),
            mutation_requirement="required",
            confidence=0.99,
        )


def _semantic_plan(*, configured_model_id: str | None = None) -> dict[str, Any]:
    criterion = VerificationCriterion(
        criterion_id="semantic-1",
        kind="semantic",
        required=True,
        description="The response satisfies the requested rubric",
        source_turn_ids=("turn-1",),
        verifier_id="semantic_judge",
        payload={"rubric": "state that the report is complete"},
    )
    plan: dict[str, Any] = {"criteria": [criterion.to_dict()]}
    if configured_model_id is not None:
        plan["semantic_judge_model_id"] = configured_model_id
    return plan


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "adaptive_runtime",
    [
        {"enabled": False},
        {"enabled": True, "retrieval": {"enabled": False}},
    ],
    ids=["parent-disabled", "retrieval-disabled"],
)
async def test_tool_discovery_persistence_respects_kill_switches(
    tmp_path: Path,
    adaptive_runtime: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = AgentEngine(_config(tmp_path, adaptive_runtime=adaptive_runtime))

    async def unexpected_record(**kwargs: Any) -> Any:
        raise AssertionError(f"discovery persistence must be a no-op: {kwargs}")

    monkeypatch.setattr(
        engine._tool_discovery_state_repository,  # noqa: SLF001
        "record_observations",
        unexpected_record,
    )
    await engine._record_tool_search_discovery(  # noqa: SLF001
        {
            "session_id": "session-1",
            "turn_id": "turn-1",
            "source_query_hash": "query-hash",
            "catalog_fingerprint": "catalog-hash",
            "catalog_generation": 1,
            "matches": [
                {"tool_name": "file_write", "capability_risk_class": "elevated"}
            ],
        }
    )
    await engine.close()


@pytest.mark.parametrize(
    "adaptive_runtime",
    [
        {"enabled": False},
        {"enabled": True, "retrieval": {"enabled": False}},
    ],
    ids=["parent-disabled", "retrieval-disabled"],
)
def test_retrieval_kill_switch_removes_jit_broker_and_deferred_activation(
    tmp_path: Path,
    adaptive_runtime: dict[str, Any],
) -> None:
    engine = AgentEngine(_config(tmp_path, adaptive_runtime=adaptive_runtime))
    disabled = engine._apply_adaptive_retrieval_switch(  # noqa: SLF001
        ToolExposurePlan(
            tool_names=["echo_tool", "tool_search"],
            matched_groups=["workspace"],
            limit=4,
            discoverable_tool_names=["echo_tool", "file_write", "tool_search"],
        )
    )

    assert disabled.tool_names == ["echo_tool"]
    assert disabled.discoverable_tool_names == ["echo_tool"]
    registry = engine._tool_registry.create_view(  # noqa: SLF001
        disabled.tool_names,
        tool_search_catalog_names=disabled.discoverable_tool_names,
        schema_limit=disabled.limit,
    )
    assert registry.get("tool_search") is None
    assert registry.get("tool_activate") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "adaptive_runtime",
    [
        {"enabled": False},
        {"enabled": True, "plan": {"enabled": False}},
    ],
    ids=["parent-disabled", "plan-disabled"],
)
async def test_plan_runtime_kill_switch_is_noop_before_ledger_load(
    tmp_path: Path,
    adaptive_runtime: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = AgentEngine(_config(tmp_path, adaptive_runtime=adaptive_runtime))

    async def unexpected_load(**kwargs: Any) -> Any:
        raise AssertionError(f"ledger load must be a no-op: {kwargs}")

    monkeypatch.setattr(engine, "_load_active_plan_ledger", unexpected_load)
    exposure = ToolExposurePlan(
        tool_names=["echo_tool"],
        matched_groups=[],
        limit=4,
        discoverable_tool_names=["echo_tool"],
    )
    context = ToolExecutionContext(session_id="session-1")
    result = await engine._configure_plan_runtime(  # noqa: SLF001
        session_id="session-1",
        turn_id="turn-1",
        request=AgentInvocationRequest(message="write", persist_session=True),
        rollout=_rollout(_contract()),
        available_tools=[EchoTool()],
        exposure_plan=exposure,
        tool_execution_context=context,
    )
    assert result == (exposure, {}, None)
    assert "plan_runtime" not in context.state
    await engine.close()


@pytest.mark.asyncio
async def test_shadow_mode_observes_complexity_without_exposing_or_enforcing_plan(
    tmp_path: Path,
) -> None:
    engine = AgentEngine(
        _config(tmp_path, adaptive_runtime={"complexity": {"mode": "shadow"}})
    )
    exposure = ToolExposurePlan(
        tool_names=["echo_tool"],
        matched_groups=[],
        limit=4,
        discoverable_tool_names=["echo_tool"],
    )
    context = ToolExecutionContext(session_id="session-shadow")
    deliverables = (
        DeliverableContract(
            kind="workspace_file",
            target_hint="report.md",
            acceptance_criteria=("report exists",),
            source_turn_ids=("turn-shadow",),
        ),
        DeliverableContract(
            kind="workspace_file",
            target_hint="tests.txt",
            acceptance_criteria=("tests summarized",),
            source_turn_ids=("turn-shadow",),
        ),
    )

    configured_exposure, decision, task_plan_context = await engine._configure_plan_runtime(  # noqa: SLF001
        session_id="session-shadow",
        turn_id="turn-shadow",
        request=AgentInvocationRequest(message="write", persist_session=True),
        rollout=_rollout(
            _contract(
                turn_id="turn-shadow",
                deliverables=deliverables,
                operations=frozenset({"workspace_write", "execution"}),
            )
        ),
        available_tools=[EchoTool()],
        exposure_plan=exposure,
        tool_execution_context=context,
    )

    assert decision["kind"] == "plan_required"
    assert configured_exposure == exposure
    assert task_plan_context is None
    plan_runtime = context.state["plan_runtime"]
    assert plan_runtime["enabled"] is False
    assert plan_runtime["required"] is False
    assert plan_runtime["exposed"] is False
    assert plan_runtime["mutable"] is False
    assert plan_runtime["state"] == "inactive"
    assert plan_runtime["unavailable_reason"] == "planning_shadow_mode"
    assert "update_plan_runtime" not in context.state
    await engine.close()


def test_verification_compile_and_run_respect_parent_and_component_switches(
    tmp_path: Path,
) -> None:
    deliverable = DeliverableContract(
        kind="workspace_file",
        target_hint="report.md",
        acceptance_criteria=("explain correctness",),
        source_turn_ids=("turn-1",),
    )
    for adaptive_runtime in (
        {"enabled": False},
        {"enabled": True, "verification": {"enabled": False}},
    ):
        engine = AgentEngine(_config(tmp_path, adaptive_runtime=adaptive_runtime))
        assert (
            engine._build_verification_plan(  # noqa: SLF001
                _rollout(_contract(deliverables=(deliverable,))),
                semantic_fallback_enabled=True,
            )
            is None
        )


@pytest.mark.asyncio
async def test_single_routine_file_write_approval_does_not_force_plan(
    tmp_path: Path,
) -> None:
    engine = AgentEngine(
        _config(
            tmp_path,
            adaptive_runtime={"complexity": {"mode": "enforce"}},
        )
    )
    tool = FileWriteTool(workspace_dir=tmp_path, require_approval=True)
    deliverable = DeliverableContract(
        kind="workspace_file",
        target_hint="report.md",
        source_turn_ids=("turn-1",),
    )
    decision = await engine._resolve_complexity_decision(  # noqa: SLF001
        rollout=_rollout(
            _contract(deliverables=(deliverable,)),
            eligible_tools=(tool.name,),
        ),
        available_tools=[tool],
        exposure_plan=ToolExposurePlan(
            tool_names=[tool.name],
            matched_groups=["workspace"],
            limit=4,
        ),
    )
    assert decision["kind"] == "no_plan"
    assert "approval_likely" not in decision["hard_reason_codes"]


@pytest.mark.parametrize(
    ("tools", "expected_reason"),
    [
        (
            [_ApprovalEffectTool("destructive", boundary_known=True, destructive=True)],
            "destructive_tool_available",
        ),
        (
            [
                _ApprovalEffectTool("write-a", boundary_known=True),
                _ApprovalEffectTool("write-b", boundary_known=True),
            ],
            "approval_likely",
        ),
        ([_ApprovalEffectTool("unknown-boundary", boundary_known=False)], "approval_likely"),
    ],
    ids=["destructive", "multiple-approvals", "unknown-boundary"],
)
@pytest.mark.asyncio
async def test_complexity_gate_fails_closed_for_risky_approval_shapes(
    tmp_path: Path,
    tools: list[BaseTool],
    expected_reason: str,
) -> None:
    engine = AgentEngine(
        _config(
            tmp_path,
            adaptive_runtime={"complexity": {"mode": "enforce"}},
        )
    )
    decision = await engine._resolve_complexity_decision(  # noqa: SLF001
        rollout=_rollout(
            _contract(),
            eligible_tools=tuple(tool.name for tool in tools),
        ),
        available_tools=tools,
        exposure_plan=ToolExposurePlan(
            tool_names=[tool.name for tool in tools],
            matched_groups=["workspace"],
            limit=4,
        ),
    )
    assert decision["kind"] == "plan_required"
    assert expected_reason in decision["hard_reason_codes"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "delay", "enabled", "expected_reason", "expected_calls"),
    [
        (
            {
                "response_version": COMPLEXITY_ADVISOR_RESPONSE_VERSION,
                "plan_recommended": True,
                "estimated_distinct_actions": 4,
                "dependency_count": 2,
                "reason_codes": ["cross_tool_dependency"],
                "confidence": 0.8,
            },
            0.0,
            True,
            "cross_tool_dependency",
            1,
        ),
        ({"unexpected": True}, 0.0, True, "advisor_malformed", 1),
        ({}, 0.05, True, "advisor_timeout", 1),
        ({}, 0.0, False, None, 0),
    ],
    ids=["success", "malformed", "timeout", "disabled"],
)
async def test_engine_complexity_advisor_producer_and_diagnostics(
    tmp_path: Path,
    payload: dict[str, Any],
    delay: float,
    enabled: bool,
    expected_reason: str | None,
    expected_calls: int,
) -> None:
    class AdvisorBackend(FakeBackend):
        async def generate(self, messages, tools=None, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append(messages)
            assert tools is None
            assert kwargs["temperature"] == 0.0
            if delay:
                await asyncio.sleep(delay)
            return GenerationResult(
                content=json.dumps(payload),
                input_tokens=11,
                output_tokens=7,
            )

    engine = AgentEngine(
        _config(
            tmp_path,
            adaptive_runtime={
                "complexity": {
                    "mode": "enforce",
                    "model_advisor_enabled": enabled,
                    "advisor_timeout_seconds": 0.01,
                }
            },
        )
    )
    backend = AdvisorBackend()
    diagnostics = AdaptiveDiagnosticsAccumulator()
    deliverables = (
        DeliverableContract(
            kind="workspace_file",
            target_hint="one.md",
            source_turn_ids=("turn-1",),
        ),
        DeliverableContract(
            kind="workspace_file",
            target_hint="two.md",
            source_turn_ids=("turn-1",),
        ),
    )

    decision = await engine._resolve_complexity_decision(  # noqa: SLF001
        active_backend=backend,
        rollout=_rollout(_contract(deliverables=deliverables)),
        available_tools=[],
        exposure_plan=ToolExposurePlan(tool_names=[], matched_groups=[], limit=4),
        diagnostics=diagnostics,
    )

    assert len(backend.calls) == expected_calls
    assert decision["advisor_used"] is enabled
    if expected_reason is not None:
        assert expected_reason in decision["soft_reason_codes"]
    snapshot = diagnostics.snapshot()
    assert snapshot["model_calls"] == expected_calls
    assert snapshot["model_wall_observed_calls"] == expected_calls
    assert snapshot["model_usage_observed_calls"] == (
        1 if expected_calls and delay == 0 else 0
    )
    await engine.close()


@pytest.mark.asyncio
async def test_incomplete_plan_persists_current_item_then_blocks_and_replays(
    tmp_path: Path,
) -> None:
    engine = AgentEngine(_config(tmp_path))
    seed = await engine._plan_ledger_repository.save(  # noqa: SLF001
        _ledger(
            session_id="session-plan",
            items=(
                _plan_item("item-1", "in_progress"),
                _plan_item("item-2", "pending", dependencies=("item-1",)),
            ),
        ),
        expected_revision=0,
        turn_id="turn-0",
        idempotency_key="seed-plan",
    )
    assert seed.status == "saved" and seed.ledger is not None

    first, first_error = await engine._complete_verified_plan_ledger(  # noqa: SLF001
        turn_id="turn-1",
        plan_ledger_snapshot=seed.ledger.to_dict(),
        recognized_evidence_refs=("write-call",),
    )
    assert first is not None
    assert first["status"] == "active"
    assert [item["status"] for item in first["items"]] == ["completed", "pending"]
    assert first_error == "plan ledger remains incomplete after completing the current item"

    replay, replay_error = await engine._complete_verified_plan_ledger(  # noqa: SLF001
        turn_id="turn-1",
        plan_ledger_snapshot=seed.ledger.to_dict(),
        recognized_evidence_refs=("write-call",),
    )
    assert replay == first
    assert replay_error == first_error
    await engine.close()


@pytest.mark.asyncio
async def test_active_plan_without_in_progress_item_blocks_completion(tmp_path: Path) -> None:
    engine = AgentEngine(_config(tmp_path))
    ledger = _ledger(
        session_id="session-plan",
        items=(_plan_item("item-1", "pending"),),
    )
    result, error = await engine._complete_verified_plan_ledger(  # noqa: SLF001
        turn_id="turn-1",
        plan_ledger_snapshot=ledger.to_dict(),
        recognized_evidence_refs=("write-call",),
    )
    assert result == ledger.to_dict()
    assert error == "active plan ledger has pending work but no in-progress item"
    await engine.close()


@pytest.mark.asyncio
async def test_active_task_caller_independently_requires_completed_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = AgentEngine(_config(tmp_path))
    target = tmp_path / "report.md"
    target.write_text("# Report\n", encoding="utf-8")
    active_task = ActiveTaskState(
        goal_id="goal-1",
        objective="Write one requested artifact",
        operations=frozenset({"workspace_write"}),
        mutation_requirement="required",
        deliverables=(
            DeliverableContract(
                kind="workspace_file",
                target_hint="report.md",
                acceptance_criteria=("exists",),
                source_turn_ids=("turn-1",),
            ),
        ),
        source_turn_ids=("turn-1",),
        updated_turn_id="turn-1",
    )
    saved = await engine._conversation_state_repository.save(  # noqa: SLF001
        "session-caller-invariant",
        active_task=active_task,
        expected_revision=0,
    )
    assert saved.status == "saved"
    state = await engine._conversation_state_repository.load(  # noqa: SLF001
        "session-caller-invariant"
    )

    async def incomplete_ledger(**kwargs: Any) -> tuple[dict[str, Any], None]:
        return {"status": "active", "items": []}, None

    monkeypatch.setattr(engine, "_complete_verified_plan_ledger", incomplete_ledger)
    receipt, error = await engine._verify_and_complete_active_task(  # noqa: SLF001
        session_id="session-caller-invariant",
        turn_id="turn-1",
        workspace_dir=str(tmp_path),
        active_task=active_task,
        state_revision=state.state_revision,
        requests=[
            ToolCallRequestEvent(
                call_id="write-call",
                tool_name="file_write",
                arguments={"path": "report.md", "content": "# Report\n"},
            )
        ],
        results=[
            ToolCallResultEvent(
                call_id="write-call",
                tool_name="file_write",
                metadata={"resolved_path": str(target)},
            )
        ],
        plan_ledger_snapshot={"ledger_version": PLAN_LEDGER_VERSION},
        recognized_evidence_refs=("write-call",),
    )
    assert receipt["plan_ledger"]["status"] == "active"
    assert error == "plan ledger is not completed"
    reloaded = await engine._conversation_state_repository.load(  # noqa: SLF001
        "session-caller-invariant"
    )
    assert reloaded.active_task is not None
    assert reloaded.active_task.status == "active"
    await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["blocked", "cancelled"])
async def test_terminal_noncompleted_plan_blocks_task_completion(
    tmp_path: Path,
    status: str,
) -> None:
    engine = AgentEngine(_config(tmp_path))
    ledger = _ledger(
        session_id="session-plan",
        status=status,
        items=(_plan_item("item-1", "cancelled"),),
    )
    result, error = await engine._complete_verified_plan_ledger(  # noqa: SLF001
        turn_id="turn-1",
        plan_ledger_snapshot=ledger.to_dict(),
        recognized_evidence_refs=("write-call",),
    )
    assert result == ledger.to_dict()
    assert error == f"plan ledger status {status} blocks active-task completion"
    await engine.close()


@pytest.mark.asyncio
async def test_semantic_judge_is_toolless_bounded_and_schema_driven(tmp_path: Path) -> None:
    engine = AgentEngine(
        _config(
            tmp_path,
            adaptive_runtime={
                "verification": {
                    "judge_max_tokens": 321,
                    "max_evidence_chars": 1000,
                }
            },
        )
    )
    backend = _JudgeBackend(
        {
            "verdict": "verified",
            "evidence_refs": ["response"],
            "reason_code": "rubric_satisfied",
            "retry_disposition": "none",
            "confidence": 0.9,
        }
    )
    receipt = await engine._build_aggregate_verification_receipt(  # noqa: SLF001
        turn_id="turn-1",
        goal_id="goal-1",
        active_task={"status": "active"},
        verification_plan=_semantic_plan(),
        artifact_verification=None,
        requests=(),
        results=(),
        final_response_text="The report is complete. " + ("x" * 4000),
        semantic_judge_backend=backend,
    )
    assert receipt is not None
    assert receipt["verdict"] == "verified"
    assert backend.raw_tools == [None]
    assert backend.generation_kwargs[0]["temperature"] == 0.0
    assert backend.generation_kwargs[0]["max_tokens"] == 321
    prompt = backend.calls[0][1].content
    assert "AUTHORITATIVE_RUBRIC_JSON" in prompt
    assert "UNTRUSTED_EVIDENCE_JSON" in prompt
    rendered_evidence = prompt.split("UNTRUSTED_EVIDENCE_JSON\n", 1)[1].split(
        "\nEND_", 1
    )[0]
    assert len(rendered_evidence) <= 1000
    assert json.loads(rendered_evidence)["truncated"] is True
    bounded = engine_module._BackendSemanticJudge(  # noqa: SLF001
        engine=engine,
        backend=backend,
        configured_model_id=None,
        max_tokens=321,
        max_evidence_chars=1000,
    )._bounded_evidence_json(  # noqa: SLF001
        {
            "recognized_evidence_refs": [
                "response",
                *(f"evidence-{index}-" + ("x" * 200) for index in range(20)),
            ],
            "response_text": "y" * 4000,
        }
    )
    assert len(bounded) <= 1000
    assert json.loads(bounded)["truncated"] is True
    await engine.close()


@pytest.mark.asyncio
async def test_semantic_judge_timeout_and_malformed_schema_fail_unverified(
    tmp_path: Path,
) -> None:
    class SlowJudge(_JudgeBackend):
        async def generate(self, messages, tools=None, **kwargs):  # type: ignore[no-untyped-def]
            await asyncio.sleep(0.05)
            return await super().generate(messages, tools=tools, **kwargs)

    engine = AgentEngine(
        _config(
            tmp_path,
            adaptive_runtime={"verification": {"judge_timeout_seconds": 0.01}},
        )
    )
    timeout_diagnostics = AdaptiveDiagnosticsAccumulator()
    timeout_receipt = await engine._build_aggregate_verification_receipt(  # noqa: SLF001
        turn_id="turn-timeout",
        goal_id="goal-1",
        active_task={"status": "active"},
        verification_plan=_semantic_plan(),
        artifact_verification=None,
        requests=(),
        results=(),
        final_response_text="The report is complete.",
        semantic_judge_backend=SlowJudge({}),
        diagnostics=timeout_diagnostics,
    )
    malformed_diagnostics = AdaptiveDiagnosticsAccumulator()
    malformed_receipt = await engine._build_aggregate_verification_receipt(  # noqa: SLF001
        turn_id="turn-malformed",
        goal_id="goal-1",
        active_task={"status": "active"},
        verification_plan=_semantic_plan(),
        artifact_verification=None,
        requests=(),
        results=(),
        final_response_text="The report is complete.",
        semantic_judge_backend=_JudgeBackend({"verdict": "verified"}),
        diagnostics=malformed_diagnostics,
    )
    assert timeout_receipt is not None
    assert timeout_receipt["verdict"] == "unverified"
    assert timeout_receipt["criteria"][0]["reason_code"] == "semantic_judge_timeout"
    assert malformed_receipt is not None
    assert malformed_receipt["verdict"] == "unverified"
    assert malformed_receipt["criteria"][0]["reason_code"] == "semantic_judge_malformed"
    assert (
        timeout_diagnostics.snapshot()["model_calls"],
        timeout_diagnostics.snapshot()["model_wall_observed_calls"],
        timeout_diagnostics.snapshot()["model_usage_observed_calls"],
        timeout_diagnostics.snapshot()["recovery_model_calls"],
    ) == (1, 1, 0, 0)
    assert (
        malformed_diagnostics.snapshot()["model_calls"],
        malformed_diagnostics.snapshot()["model_wall_observed_calls"],
        malformed_diagnostics.snapshot()["model_usage_observed_calls"],
        malformed_diagnostics.snapshot()["recovery_model_calls"],
    ) == (1, 1, 0, 0)
    await engine.close()


@pytest.mark.asyncio
async def test_semantic_judge_cancellation_records_direct_model_attempt(
    tmp_path: Path,
) -> None:
    class BlockingJudge(_JudgeBackend):
        def __init__(self) -> None:
            super().__init__({})
            self.started = asyncio.Event()

        async def generate(self, messages, tools=None, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append(messages)
            self.raw_tools.append(tools)
            self.generation_kwargs.append(dict(kwargs))
            self.started.set()
            await asyncio.Future()
            raise AssertionError("unreachable")

    engine = AgentEngine(_config(tmp_path))
    backend = BlockingJudge()
    diagnostics = AdaptiveDiagnosticsAccumulator()
    task = asyncio.create_task(
        engine._build_aggregate_verification_receipt(  # noqa: SLF001
            turn_id="turn-cancelled",
            goal_id="goal-1",
            active_task={"status": "active"},
            verification_plan=_semantic_plan(),
            artifact_verification=None,
            requests=(),
            results=(),
            final_response_text="The report is complete.",
            semantic_judge_backend=backend,
            diagnostics=diagnostics,
        )
    )
    await asyncio.wait_for(backend.started.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    snapshot = diagnostics.snapshot()
    assert (
        snapshot["model_calls"],
        snapshot["model_wall_observed_calls"],
        snapshot["model_usage_observed_calls"],
        snapshot["recovery_model_calls"],
    ) == (1, 1, 0, 0)
    await engine.close()


@pytest.mark.asyncio
async def test_semantic_judge_unavailable_blocks_instead_of_passing(tmp_path: Path) -> None:
    engine = AgentEngine(_config(tmp_path))
    receipt = await engine._build_aggregate_verification_receipt(  # noqa: SLF001
        turn_id="turn-1",
        goal_id="goal-1",
        active_task={"status": "active"},
        verification_plan=_semantic_plan(),
        artifact_verification=None,
        requests=(),
        results=(),
        final_response_text="The report is complete.",
    )
    assert receipt is not None
    assert receipt["verdict"] == "unverified"
    assert receipt["criteria"][0]["reason_code"] == "unsupported_criterion"
    await engine.close()


@pytest.mark.asyncio
async def test_unavailable_semantic_judge_keeps_active_task_open(tmp_path: Path) -> None:
    engine = AgentEngine(_config(tmp_path))
    target = tmp_path / "report.md"
    target.write_text("# Report\n", encoding="utf-8")
    active_task = ActiveTaskState(
        goal_id="goal-1",
        objective="Write one requested artifact",
        operations=frozenset({"workspace_write"}),
        mutation_requirement="required",
        deliverables=(
            DeliverableContract(
                kind="workspace_file",
                target_hint="report.md",
                acceptance_criteria=("exists",),
                source_turn_ids=("turn-1",),
            ),
        ),
        source_turn_ids=("turn-1",),
        updated_turn_id="turn-1",
    )
    saved = await engine._conversation_state_repository.save(  # noqa: SLF001
        "session-semantic",
        active_task=active_task,
        expected_revision=0,
    )
    assert saved.status == "saved"
    state = await engine._conversation_state_repository.load("session-semantic")  # noqa: SLF001
    request = ToolCallRequestEvent(
        call_id="write-call",
        tool_name="file_write",
        arguments={"path": "report.md", "content": "# Report\n"},
    )
    result = ToolCallResultEvent(
        call_id="write-call",
        tool_name="file_write",
        metadata={"resolved_path": str(target)},
    )
    receipt, error = await engine._verify_and_complete_active_task(  # noqa: SLF001
        session_id="session-semantic",
        turn_id="turn-1",
        workspace_dir=str(tmp_path),
        active_task=active_task,
        state_revision=state.state_revision,
        requests=[request],
        results=[result],
        verification_plan=_semantic_plan(),
        final_response_text="The report is complete.",
    )
    assert receipt["aggregate_verdict"] == "unverified"
    assert error == "aggregate verification blocked finalization: unverified"
    reloaded = await engine._conversation_state_repository.load("session-semantic")  # noqa: SLF001
    assert reloaded.active_task is not None
    assert reloaded.active_task.status == "active"
    await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("aggregate_verdict", "expected_prefix", "expected_error_type"),
    [
        (
            "unverified",
            "The requested operation ran, but independent semantic",
            "semantic_verification_unverified",
        ),
        (
            "failed",
            "The requested operation ran, but independent verification found",
            "required_verification_failed",
        ),
    ],
)
async def test_react_success_claim_is_replaced_when_required_verification_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    aggregate_verdict: str,
    expected_prefix: str,
    expected_error_type: str,
) -> None:
    backend = FakeBackend(metadata={"effective_context_length": 32768})
    target = tmp_path / "report.md"

    class SpyReActLoop:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.turn_messages: list[Any] = []

        async def run(self, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
            target.write_text("# Report\n", encoding="utf-8")
            yield ToolCallRequestEvent(
                call_id="write-call",
                tool_name="file_write",
                arguments={"path": "report.md", "content": "# Report\n"},
            )
            yield ToolCallResultEvent(
                call_id="write-call",
                tool_name="file_write",
                metadata={"resolved_path": str(target)},
            )
            yield FinalAnswerEvent(content="Saved report.md successfully")

    engine = AgentEngine(
        _config(tmp_path),
        conversation_resolver_factory=lambda backend_arg: ConversationResolver(
            interpreter=_SemanticWriteInterpreter()
        ),
    )

    async def fake_load(model_spec: str) -> FakeBackend:
        engine._router._active = backend  # noqa: SLF001
        return backend

    engine._router.load = fake_load  # type: ignore[method-assign]

    async def semantic_block(**kwargs: Any) -> str:
        final_event = next(
            event
            for event in reversed(kwargs["events"])
            if isinstance(event, FinalAnswerEvent)
        )
        final_event.metadata["artifact_verification"] = {
            "verification_status": "verified",
            "aggregate_verdict": aggregate_verdict,
            "aggregate_retry_disposition": "requires_replan",
            "aggregate_verification_receipt": {
                "verdict": aggregate_verdict,
            },
        }
        return f"aggregate verification blocked finalization: {aggregate_verdict}"

    monkeypatch.setattr(
        engine,
        "_complete_turn_contract_task_if_satisfied",
        semantic_block,
    )
    monkeypatch.setattr(engine_module, "AsyncReActLoop", SpyReActLoop)
    result = await engine.invoke(
        AgentInvocationRequest(
            message="Create report.md and explain correctness",
            session_id="session-blocked-final",
            persist_session=True,
        )
    )
    final = next(
        event for event in reversed(result.events) if isinstance(event, FinalAnswerEvent)
    )
    assert result.content.startswith(expected_prefix), final.metadata
    assert final.finish_reason == "verification_blocked"
    assert final.metadata["error_type"] == expected_error_type
    await engine.close()


@pytest.mark.asyncio
async def test_configured_semantic_judge_reacquires_model_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = AgentEngine(_config(tmp_path))
    calls: list[dict[str, Any]] = []

    async def configured_generate(**kwargs: Any) -> GenerationResult:
        calls.append(kwargs)
        return GenerationResult(
            content=json.dumps(
                {
                    "verdict": "verified",
                    "evidence_refs": ["response"],
                    "reason_code": "rubric_satisfied",
                    "retry_disposition": "none",
                    "confidence": 0.9,
                }
            )
        )

    monkeypatch.setattr(engine, "generate_with_configured_model", configured_generate)
    receipt = await engine._build_aggregate_verification_receipt(  # noqa: SLF001
        turn_id="turn-1",
        goal_id="goal-1",
        active_task={"status": "active"},
        verification_plan=_semantic_plan(configured_model_id="configured-judge"),
        artifact_verification=None,
        requests=(),
        results=(),
        final_response_text="The report is complete.",
        semantic_judge_backend=FakeBackend(),
    )
    assert receipt is not None and receipt["verdict"] == "verified"
    assert calls[0]["model_id"] == "configured-judge"
    assert calls[0]["temperature"] == 0.0
    assert calls[0]["max_tokens"] == 800
    await engine.close()


@pytest.mark.asyncio
async def test_semantic_success_cannot_override_deterministic_failure(tmp_path: Path) -> None:
    semantic = VerificationCriterion(
        criterion_id="semantic-1",
        kind="semantic",
        required=True,
        description="semantic",
        source_turn_ids=("turn-1",),
        verifier_id="semantic_judge",
        payload={"rubric": "be correct"},
    )
    response_shape = VerificationCriterion(
        criterion_id="shape-1",
        kind="response_shape",
        required=True,
        description="required section",
        source_turn_ids=("turn-1",),
        verifier_id="response_shape",
        payload={"required_sections": ["Required Section"]},
    )
    engine = AgentEngine(_config(tmp_path))
    backend = _JudgeBackend(
        {
            "verdict": "verified",
            "evidence_refs": ["response"],
            "reason_code": "semantic_ok",
            "retry_disposition": "none",
            "confidence": 0.99,
        }
    )
    receipt = await engine._build_aggregate_verification_receipt(  # noqa: SLF001
        turn_id="turn-1",
        goal_id="goal-1",
        active_task={"status": "active"},
        verification_plan={"criteria": [response_shape.to_dict(), semantic.to_dict()]},
        artifact_verification=None,
        requests=(),
        results=(),
        final_response_text="No matching heading.",
        semantic_judge_backend=backend,
    )
    assert receipt is not None
    assert receipt["verdict"] == "failed"
    assert receipt["hard_failure"] is True
    await engine.close()


@pytest.mark.asyncio
async def test_aggregate_receipt_is_durable_before_plan_and_task_completion(
    tmp_path: Path,
) -> None:
    engine = AgentEngine(_config(tmp_path))
    target = tmp_path / "report.md"
    target.write_text("# Report\n", encoding="utf-8")
    active_task = ActiveTaskState(
        goal_id="goal-1",
        objective="Write one requested artifact",
        operations=frozenset({"workspace_write"}),
        mutation_requirement="required",
        deliverables=(
            DeliverableContract(
                kind="workspace_file",
                target_hint="report.md",
                acceptance_criteria=("exists",),
                source_turn_ids=("turn-1",),
            ),
        ),
        source_turn_ids=("turn-1",),
        updated_turn_id="turn-1",
    )
    state_saved = await engine._conversation_state_repository.save(  # noqa: SLF001
        "session-order",
        active_task=active_task,
        expected_revision=0,
    )
    assert state_saved.status == "saved"
    state = await engine._conversation_state_repository.load("session-order")  # noqa: SLF001
    ledger_saved = await engine._plan_ledger_repository.save(  # noqa: SLF001
        _ledger(
            session_id="session-order",
            items=(_plan_item("item-1", "in_progress"),),
        ),
        expected_revision=0,
        turn_id="turn-0",
        idempotency_key="seed-order-plan",
    )
    assert ledger_saved.status == "saved" and ledger_saved.ledger is not None
    request = ToolCallRequestEvent(
        call_id="write-call",
        tool_name="file_write",
        arguments={"path": "report.md", "content": "# Report\n"},
    )
    result = ToolCallResultEvent(
        call_id="write-call",
        tool_name="file_write",
        metadata={"resolved_path": str(target), "operation_id": "write-operation"},
    )
    plan = {
        "criteria": [
            VerificationCriterion(
                criterion_id="artifact-1",
                kind="artifact",
                required=True,
                description="artifact exists",
                source_turn_ids=("turn-1",),
                verifier_id="artifact",
                payload={"check": "exists", "target_hint": "report.md"},
            ).to_dict()
        ]
    }
    receipt, error = await engine._verify_and_complete_active_task(  # noqa: SLF001
        session_id="session-order",
        turn_id="turn-1",
        workspace_dir=str(tmp_path),
        active_task=active_task,
        state_revision=state.state_revision,
        requests=[request],
        results=[result],
        verification_plan=plan,
        final_response_text="Saved report.md",
        plan_ledger_snapshot=ledger_saved.ledger.to_dict(),
        recognized_evidence_refs=("write-call",),
    )
    assert error is None
    assert receipt["aggregate_verdict"] == "verified"
    events = await engine._session_store.load_session("session-order")  # noqa: SLF001
    aggregate_index = next(
        index
        for index, event in enumerate(events)
        if event.get("event") == "ordinary_chat_verification_receipt_recorded"
    )
    completed_plan_index = next(
        index
        for index, event in enumerate(events)
        if event.get("event") == PLAN_LEDGER_EVENT
        and event.get("plan_ledger", {}).get("status") == "completed"
    )
    completed_task_index = next(
        index
        for index, event in enumerate(events)
        if event.get("event") == "active_task_state_updated"
        and event.get("active_task_state", {}).get("status") == "completed"
    )
    assert aggregate_index < completed_plan_index < completed_task_index
    await engine.close()
