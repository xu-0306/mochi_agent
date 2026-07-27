"""Package A adversarial gates for the Wave 2 ordinary-Chat runtime."""

from __future__ import annotations

from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from mochi.agents.complexity_gate import (
    ComplexityCapabilitySummary,
    ComplexityGate,
    ComplexityGateRequest,
)
from mochi.agents.engine import AgentEngine
from mochi.agents.react_loop import AsyncReActLoop
from mochi.agents.tool_exposure import ToolExposurePlan
from mochi.agents.turn_intent_contract import (
    DeliverableContract,
    IntentEvidence,
    TurnIntentContract,
)
from mochi.backends.types import ToolCall
from mochi.config.schema import MochiConfig
from mochi.tools.base import ToolExecutionContext
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
