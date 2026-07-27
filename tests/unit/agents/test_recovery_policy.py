from __future__ import annotations

import pytest

from mochi.agents.controlled_recovery import (
    ApprovalContinuationState,
    TimelineOperationState,
)
from mochi.agents.outcome_verifier import (
    CriterionReceipt,
    VerificationReceipt,
)
from mochi.agents.plan_ledger import PlanItem, PlanLedger
from mochi.agents.recovery_policy import (
    RECOVERY_BUDGET_VERSION,
    RECOVERY_DECISION_VERSION,
    RecoveryBudget,
    RecoveryBudgetExhausted,
    RecoveryPolicy,
    RecoveryPolicyError,
)


def _receipt(
    *,
    verdict: str = "failed",
    retry_disposition: str = "requires_replan",
    hard_failure: bool = True,
) -> VerificationReceipt:
    return VerificationReceipt(
        receipt_version="verification-receipt-v1",
        receipt_id="receipt-1",
        turn_id="turn-1",
        goal_id=None,
        verdict=verdict,  # type: ignore[arg-type]
        criteria=(
            CriterionReceipt(
                criterion_id="criterion-1",
                verdict=verdict,  # type: ignore[arg-type]
                verifier_id="artifact",
                evidence_refs=("evidence-1",),
                reason_code="target_mismatch",
                retry_disposition=retry_disposition,
            ),
        ),
        hard_failure=hard_failure,
        retry_disposition=retry_disposition,
    )


def _operation(status: str, operation_id: str = "operation-1") -> TimelineOperationState:
    boundary = {
        "precommitted": "not_started",
        "started": "started",
        "succeeded": "started",
        "failed": "started",
        "unknown": "unknown",
    }[status]
    return TimelineOperationState(
        operation_id=operation_id,
        status=status,  # type: ignore[arg-type]
        side_effect_boundary=boundary,  # type: ignore[arg-type]
    )


def _ledger() -> PlanLedger:
    return PlanLedger(
        ledger_version="plan-ledger-v1",
        ledger_id="ledger-1",
        session_id="session-1",
        goal_id="goal:turn-1",
        revision=1,
        status="active",
        objective="repair the declared artifact",
        reason_codes=("multiple_deliverables",),
        items=(
            PlanItem(
                item_id="item-1",
                title="repair the artifact",
                status="in_progress",
                dependencies=(),
                success_criteria=("criterion-1",),
                source_turn_ids=("turn-1",),
                attempts=1,
            ),
        ),
        created_turn_id="turn-1",
        updated_turn_id="turn-1",
    )


def test_budget_round_trip_and_strict_schema() -> None:
    budget = RecoveryBudget.initial()

    assert RecoveryBudget.from_dict(budget.to_dict()) == budget
    assert budget.to_dict()["budget_version"] == RECOVERY_BUDGET_VERSION
    with pytest.raises(RecoveryPolicyError, match="unexpected fields"):
        RecoveryBudget.from_dict({**budget.to_dict(), "extra": True})


def test_budget_consumption_is_immutable_and_fail_closed() -> None:
    budget = RecoveryBudget.initial(
        max_attempts=1,
        max_extra_model_calls=1,
        max_extra_tool_calls=2,
        max_extra_wall_seconds=10,
    )

    remaining = budget.reserve_recovery(tool_calls=1, wall_seconds=2.5)

    assert budget.remaining_attempts == 1
    assert remaining.remaining_attempts == 0
    assert remaining.remaining_extra_tool_calls == 1
    assert remaining.remaining_extra_wall_seconds == 7.5
    with pytest.raises(RecoveryBudgetExhausted):
        remaining.consume(tool_calls=2)


def test_pre_effect_failure_can_replan_without_reusing_an_operation() -> None:
    decision = RecoveryPolicy.decide(
        receipt=_receipt(),
        operation=None,
        plan_ledger=_ledger(),
        budget=RecoveryBudget.initial(),
    )

    assert decision.action == "model_replan"
    assert decision.reason_code == "known_pre_side_effect_failure"
    assert decision.operation_id is None
    assert decision.remaining_budget.remaining_attempts == 0


def test_known_failed_operation_requires_a_fresh_replacement_operation() -> None:
    decision = RecoveryPolicy.decide(
        receipt=_receipt(),
        operation=_operation("failed"),
        plan_ledger=_ledger(),
        budget=RecoveryBudget.initial(),
        fresh_operation_id="operation-2",
    )

    assert decision.action == "new_operation"
    assert decision.operation_id == "operation-2"
    assert decision.supersedes_operation_id == "operation-1"
    assert decision.remaining_budget.remaining_attempts == 0


def test_replacement_without_fresh_id_is_blocked_without_consuming_budget() -> None:
    decision = RecoveryPolicy.decide(
        receipt=_receipt(),
        operation=_operation("failed"),
        plan_ledger=_ledger(),
        budget=RecoveryBudget.initial(),
    )

    assert decision.action == "blocked"
    assert decision.reason_code == "fresh_operation_id_required"
    assert decision.remaining_budget == RecoveryBudget.initial()


def test_known_success_verification_failure_corrects_only_failed_criteria() -> None:
    decision = RecoveryPolicy.decide(
        receipt=_receipt(),
        operation=_operation("succeeded"),
        plan_ledger=_ledger(),
        budget=RecoveryBudget.initial(),
        fresh_operation_id="operation-2",
    )
    context = RecoveryPolicy.build_corrective_context(
        receipt=_receipt(),
        decision=decision,
        plan_ledger=_ledger(),
        allowed_targets=("README.md",),
    )

    assert decision.action == "corrective_replan"
    assert decision.failed_criterion_ids == ("criterion-1",)
    assert context["allowed_targets"] == ["README.md"]
    assert context["failed_criteria"] == [
        {
            "criterion_id": "criterion-1",
            "reason_code": "target_mismatch",
            "evidence_refs": ["evidence-1"],
        }
    ]
    assert "operation-1" in context["prohibited_repeats"]
    assert "expand the task scope" in context["instruction"]


def test_unknown_side_effect_is_blocked_even_when_receipt_requests_replan() -> None:
    decision = RecoveryPolicy.decide(
        receipt=_receipt(),
        operation=_operation("unknown"),
        plan_ledger=_ledger(),
        budget=RecoveryBudget.initial(),
        fresh_operation_id="operation-2",
    )

    assert decision.action == "blocked"
    assert decision.reason_code == "side_effect_outcome_unknown"
    assert decision.remaining_budget == RecoveryBudget.initial()


def test_pending_approval_returns_wait_state_without_recovery() -> None:
    decision = RecoveryPolicy.decide(
        receipt=_receipt(),
        operation=_operation("precommitted"),
        plan_ledger=_ledger(),
        budget=RecoveryBudget.initial(),
        approval=ApprovalContinuationState(approval_state="pending"),
    )

    assert decision.action == "await_approval"
    assert decision.reason_code == "approval_pending"
    assert decision.remaining_budget == RecoveryBudget.initial()


def test_budget_exhaustion_returns_partial_and_preserves_failed_criteria() -> None:
    decision = RecoveryPolicy.decide(
        receipt=_receipt(),
        operation=_operation("failed"),
        plan_ledger=_ledger(),
        budget=RecoveryBudget.initial(max_attempts=0),
        fresh_operation_id="operation-2",
    )

    assert decision.action == "partial"
    assert decision.reason_code == "recovery_budget_exhausted"
    assert decision.failed_criterion_ids == ("criterion-1",)


def test_verified_or_optional_failure_is_terminal_without_extra_model_work() -> None:
    verified = RecoveryPolicy.decide(
        receipt=_receipt(verdict="verified", retry_disposition="none", hard_failure=False),
        operation=_operation("succeeded"),
        plan_ledger=None,
        budget=RecoveryBudget.initial(),
    )
    optional = RecoveryPolicy.decide(
        receipt=_receipt(verdict="unverified", retry_disposition="none", hard_failure=False),
        operation=_operation("succeeded"),
        plan_ledger=None,
        budget=RecoveryBudget.initial(),
    )

    assert verified.action == "terminal"
    assert verified.reason_code == "verification_satisfied"
    assert optional.action == "terminal"
    assert optional.reason_code == "verification_non_blocking"


def test_decision_round_trip_is_versioned_and_exact() -> None:
    decision = RecoveryPolicy.decide(
        receipt=_receipt(),
        operation=_operation("failed"),
        plan_ledger=_ledger(),
        budget=RecoveryBudget.initial(),
        fresh_operation_id="operation-2",
    )

    loaded = decision.from_dict(decision.to_dict())

    assert loaded == decision
    assert decision.to_dict()["decision_version"] == RECOVERY_DECISION_VERSION
    with pytest.raises(RecoveryPolicyError, match="unexpected fields"):
        decision.from_dict({**decision.to_dict(), "future": True})


def test_corrective_context_rejects_non_recovery_decisions() -> None:
    decision = RecoveryPolicy.decide(
        receipt=_receipt(verdict="verified", retry_disposition="none", hard_failure=False),
        operation=_operation("succeeded"),
        plan_ledger=None,
        budget=RecoveryBudget.initial(),
    )

    with pytest.raises(RecoveryPolicyError, match="requires a recovery decision"):
        RecoveryPolicy.build_corrective_context(
            receipt=_receipt(),
            decision=decision,
            plan_ledger=None,
        )
