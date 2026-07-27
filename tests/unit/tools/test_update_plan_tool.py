from __future__ import annotations

from typing import Any

import pytest

from mochi.agents.plan_ledger import PlanItem, PlanLedgerRepository
from mochi.sessions.store import SessionStore
from mochi.tools.base import ToolExecutionContext
from mochi.tools.update_plan import (
    ScopedPlanController,
    UpdatePlanRuntimeContext,
    UpdatePlanTool,
)


def _plan_item_dict(
    *,
    item_id: str = "item-1",
    status: str = "pending",
    dependencies: tuple[str, ...] = (),
    evidence_refs: tuple[str, ...] = (),
    blocker_reason: str | None = None,
) -> dict[str, Any]:
    return PlanItem(
        item_id=item_id,
        title=f"Title for {item_id}",
        status=status,  # type: ignore[arg-type]
        dependencies=dependencies,
        success_criteria=("done",),
        source_turn_ids=("turn-1",),
        evidence_refs=evidence_refs,
        blocker_reason=blocker_reason,
    ).to_dict()


def _runtime_context() -> UpdatePlanRuntimeContext:
    return UpdatePlanRuntimeContext(
        session_id="session-tool",
        goal_id="goal-tool",
        ledger_id="plan-tool",
        turn_id="turn-tool",
        objective="Build the selected artifact",
        reason_codes=("multiple_deliverables",),
        recognized_evidence_refs=frozenset({"receipt-1"}),
    )


def _tool_context(tmp_path) -> tuple[SessionStore, PlanLedgerRepository, ToolExecutionContext]:
    store = SessionStore(tmp_path / "sessions")
    repository = PlanLedgerRepository(store)
    controller = ScopedPlanController(
        repository=repository,
        runtime_context=_runtime_context(),
    )
    context = ToolExecutionContext(
        session_id="session-tool",
        state={"update_plan_controller": controller},
    )
    return store, repository, context


@pytest.mark.asyncio
async def test_update_plan_requires_scoped_controller_in_context() -> None:
    tool = UpdatePlanTool()

    result = await tool.execute(
        action="view",
        expected_revision=0,
        items=[],
        item_id=None,
        status=None,
        evidence_refs=[],
        blocker_reason=None,
        context=ToolExecutionContext(),
    )

    assert result.error is not None
    assert result.retryable is True
    assert result.metadata["error_type"] == "plan_controller_missing"


@pytest.mark.asyncio
async def test_view_ignores_mutation_fields(tmp_path) -> None:
    _, _, context = _tool_context(tmp_path)
    tool = UpdatePlanTool()

    result = await tool.execute(
        action="view",
        expected_revision=0,
        items="not-a-list",
        item_id=123,
        status="definitely-not-a-status",
        evidence_refs="not-a-list",
        blocker_reason=999,
        context=context,
    )

    assert result.error is None
    assert result.output["status"] == "missing"
    assert result.output["current_revision"] == 0


@pytest.mark.asyncio
async def test_create_or_replace_requires_items_and_set_status_requires_item_and_status(tmp_path) -> None:
    _, _, context = _tool_context(tmp_path)
    tool = UpdatePlanTool()

    missing_items = await tool.execute(
        action="create_or_replace",
        expected_revision=0,
        items=[],
        item_id=None,
        status=None,
        evidence_refs=[],
        blocker_reason=None,
        context=context,
    )
    missing_status = await tool.execute(
        action="set_status",
        expected_revision=0,
        items=[],
        item_id=None,
        status=None,
        evidence_refs=[],
        blocker_reason=None,
        context=context,
    )

    assert "create_or_replace requires items" in (missing_items.error or "")
    assert "set_status requires item_id and status" in (missing_status.error or "")


@pytest.mark.asyncio
async def test_create_or_replace_uses_trusted_runtime_ids_and_is_idempotent(tmp_path) -> None:
    store, repository, context = _tool_context(tmp_path)
    tool = UpdatePlanTool()
    items = [_plan_item_dict(item_id="item-1"), _plan_item_dict(item_id="item-2", dependencies=("item-1",))]

    first = await tool.execute(
        action="create_or_replace",
        expected_revision=0,
        items=items,
        item_id=None,
        status=None,
        evidence_refs=[],
        blocker_reason=None,
        context=context,
    )
    second = await tool.execute(
        action="create_or_replace",
        expected_revision=0,
        items=items,
        item_id=None,
        status=None,
        evidence_refs=[],
        blocker_reason=None,
        context=context,
    )

    assert first.error is None
    assert first.output["ledger"]["session_id"] == "session-tool"
    assert first.output["ledger"]["goal_id"] == "goal-tool"
    assert first.output["ledger"]["ledger_id"] == "plan-tool"
    assert second.error is None
    assert second.output["idempotent_replay"] is True
    events = await store.load_session("session-tool")
    assert len(events) == 1

    loaded = await repository.load("session-tool", "goal-tool", ledger_id="plan-tool")
    assert loaded.status == "loaded"
    assert loaded.ledger is not None
    assert loaded.ledger.revision == 1


@pytest.mark.asyncio
async def test_set_status_rejects_model_invented_evidence_refs(tmp_path) -> None:
    _, _, context = _tool_context(tmp_path)
    tool = UpdatePlanTool()
    create = await tool.execute(
        action="create_or_replace",
        expected_revision=0,
        items=[_plan_item_dict(item_id="item-1")],
        item_id=None,
        status=None,
        evidence_refs=[],
        blocker_reason=None,
        context=context,
    )
    assert create.error is None

    result = await tool.execute(
        action="set_status",
        expected_revision=1,
        items=[],
        item_id="item-1",
        status="completed",
        evidence_refs=["invented-ref"],
        blocker_reason=None,
        context=context,
    )

    assert result.error is not None
    assert result.retryable is False
    assert result.metadata["error_type"] == "plan_transition_invalid"


@pytest.mark.asyncio
async def test_set_status_completed_persists_trusted_evidence_and_saved_revision(tmp_path) -> None:
    _, repository, context = _tool_context(tmp_path)
    tool = UpdatePlanTool()
    create = await tool.execute(
        action="create_or_replace",
        expected_revision=0,
        items=[_plan_item_dict(item_id="item-1")],
        item_id=None,
        status=None,
        evidence_refs=[],
        blocker_reason=None,
        context=context,
    )
    assert create.error is None

    result = await tool.execute(
        action="set_status",
        expected_revision=1,
        items=[],
        item_id="item-1",
        status="completed",
        evidence_refs=["receipt-1"],
        blocker_reason=None,
        context=context,
    )

    assert result.error is None
    assert result.output["status"] == "saved"
    assert result.output["saved_revision"] == 2
    assert result.metadata["save_status"] == "saved"
    assert result.metadata["ledger_revision"] == 2
    loaded = await repository.load("session-tool", "goal-tool", ledger_id="plan-tool")
    assert loaded.status == "loaded"
    assert loaded.ledger is not None
    assert loaded.ledger.revision == 2
    assert loaded.ledger.items[0].status == "completed"
    assert loaded.ledger.items[0].evidence_refs == ("receipt-1",)
