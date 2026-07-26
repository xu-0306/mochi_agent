from __future__ import annotations

import pytest

from mochi.agents.controlled_recovery import (
    ApprovalContinuationState,
    ArtifactReceiptState,
    ControlledRecoveryCoordinator,
    TimelineOperationState,
)


def _receipt(
    execution_status: str = "failed",
    retry_disposition: str = "retryable",
) -> ArtifactReceiptState:
    return ArtifactReceiptState(  # type: ignore[arg-type]
        execution_status=execution_status,
        retry_disposition=retry_disposition,
    )


def _operation(
    status: str,
    *,
    operation_id: str = "operation-1",
) -> TimelineOperationState:
    boundary = {
        "precommitted": "not_started",
        "started": "started",
        "succeeded": "started",
        "failed": "started",
        "unknown": "unknown",
    }[status]
    return TimelineOperationState(  # type: ignore[arg-type]
        operation_id=operation_id,
        status=status,
        side_effect_boundary=boundary,
    )


@pytest.mark.parametrize("status", ["started", "unknown"])
def test_started_or_unknown_operation_is_never_retried(status: str) -> None:
    decision = ControlledRecoveryCoordinator.decide(
        operation=_operation(status),
        receipt=_receipt(),
    )

    assert decision.action == "blocked_unknown"
    assert decision.reason_code == "side_effect_outcome_unknown"


def test_unknown_receipt_is_never_retried_without_an_operation() -> None:
    decision = ControlledRecoveryCoordinator.decide(
        operation=None,
        receipt=_receipt(execution_status="unknown", retry_disposition="retryable"),
    )

    assert decision.action == "blocked_unknown"
    assert decision.reason_code == "execution_outcome_unknown"


@pytest.mark.parametrize("approval_state", ["rejected", "expired", "drifted"])
def test_unknown_or_started_operation_precedes_unapplied_approval(
    approval_state: str,
) -> None:
    decision = ControlledRecoveryCoordinator.decide(
        operation=_operation("started"),
        receipt=_receipt(),
        approval=ApprovalContinuationState(approval_state=approval_state),  # type: ignore[arg-type]
    )

    assert decision.action == "blocked_unknown"
    assert decision.reason_code == "side_effect_outcome_unknown"


@pytest.mark.parametrize("approval_state", ["rejected", "expired", "drifted"])
def test_unknown_receipt_precedes_unapplied_approval(
    approval_state: str,
) -> None:
    decision = ControlledRecoveryCoordinator.decide(
        operation=_operation("precommitted"),
        receipt=_receipt(execution_status="unknown", retry_disposition="retryable"),
        approval=ApprovalContinuationState(approval_state=approval_state),  # type: ignore[arg-type]
    )

    assert decision.action == "blocked_unknown"
    assert decision.reason_code == "execution_outcome_unknown"


def test_model_replan_is_allowed_before_a_side_effect_starts() -> None:
    decision = ControlledRecoveryCoordinator.decide(
        operation=_operation("precommitted"),
        receipt=_receipt(),
    )

    assert decision.action == "model_replan"
    assert decision.operation_id == "operation-1"
    assert decision.supersedes_operation_id is None


def test_pre_side_effect_failure_without_an_operation_replans_the_model() -> None:
    decision = ControlledRecoveryCoordinator.decide(
        operation=None,
        receipt=_receipt(execution_status="failed", retry_disposition="retryable"),
    )

    assert decision.action == "model_replan"
    assert decision.reason_code == "known_pre_side_effect_failure"


@pytest.mark.parametrize("retry_disposition", ["retryable", "requires_replan"])
def test_known_failed_operation_requires_a_new_operation(
    retry_disposition: str,
) -> None:
    decision = ControlledRecoveryCoordinator.decide(
        operation=_operation("failed"),
        receipt=_receipt(retry_disposition=retry_disposition),
    )

    assert decision.action == "new_operation"
    assert decision.operation_id == "operation-1"
    assert decision.supersedes_operation_id == "operation-1"


def test_known_partial_failure_requires_a_new_operation() -> None:
    decision = ControlledRecoveryCoordinator.decide(
        operation=_operation("failed"),
        receipt=_receipt(execution_status="partial", retry_disposition="requires_replan"),
    )

    assert decision.action == "new_operation"


@pytest.mark.parametrize("retry_disposition", ["terminal", "requires_approval"])
def test_terminal_receipt_never_creates_a_replacement_operation(
    retry_disposition: str,
) -> None:
    decision = ControlledRecoveryCoordinator.decide(
        operation=_operation("failed"),
        receipt=_receipt(retry_disposition=retry_disposition),
    )

    assert decision.action == "terminal"
    assert decision.supersedes_operation_id is None


def test_successful_operation_requires_a_corrective_replan_for_verification_failure() -> None:
    decision = ControlledRecoveryCoordinator.decide(
        operation=_operation("succeeded"),
        receipt=ArtifactReceiptState(
            execution_status="succeeded",
            retry_disposition="requires_replan",
        ),
    )

    assert decision.action == "corrective_replan"
    assert decision.reason_code == "receipt_requires_replan_after_known_success"
    assert decision.operation_id == "operation-1"
    assert decision.supersedes_operation_id == "operation-1"


@pytest.mark.parametrize("continuation_state", ["not_started", "continuing"])
def test_applied_approval_only_continues_the_exact_result(
    continuation_state: str,
) -> None:
    decision = ControlledRecoveryCoordinator.decide(
        operation=_operation("succeeded"),
        receipt=_receipt(execution_status="succeeded", retry_disposition="none"),
        approval=ApprovalContinuationState(  # type: ignore[arg-type]
            approval_state="applied",
            continuation_state=continuation_state,
        ),
    )

    assert decision.action == "approval_continuation"
    assert decision.operation_id == "operation-1"


def test_unknown_receipt_precedes_applied_approval_continuation() -> None:
    decision = ControlledRecoveryCoordinator.decide(
        operation=_operation("succeeded"),
        receipt=_receipt(execution_status="unknown", retry_disposition="none"),
        approval=ApprovalContinuationState(
            approval_state="applied",
            continuation_state="not_started",
        ),
    )

    assert decision.action == "blocked_unknown"
    assert decision.reason_code == "execution_outcome_unknown"


@pytest.mark.parametrize(
    ("operation", "expected_action", "expected_reason"),
    [
        (None, "terminal", "approval_applied_operation_missing"),
        (_operation("precommitted"), "terminal", "approval_applied_operation_not_completed"),
        (_operation("started"), "blocked_unknown", "side_effect_outcome_unknown"),
        (_operation("unknown"), "blocked_unknown", "side_effect_outcome_unknown"),
    ],
)
def test_applied_approval_requires_an_exact_known_operation(
    operation: TimelineOperationState | None,
    expected_action: str,
    expected_reason: str,
) -> None:
    decision = ControlledRecoveryCoordinator.decide(
        operation=operation,
        receipt=_receipt(execution_status="failed", retry_disposition="none"),
        approval=ApprovalContinuationState(
            approval_state="applied",
            continuation_state="not_started",
        ),
    )

    assert decision.action == expected_action
    assert decision.reason_code == expected_reason


def test_applied_approval_can_continue_an_exact_known_failure() -> None:
    decision = ControlledRecoveryCoordinator.decide(
        operation=_operation("failed"),
        receipt=_receipt(execution_status="failed", retry_disposition="none"),
        approval=ApprovalContinuationState(
            approval_state="applied",
            continuation_state="not_started",
        ),
    )

    assert decision.action == "approval_continuation"
    assert decision.operation_id == "operation-1"


def test_completed_approval_continuation_never_replays_the_operation() -> None:
    decision = ControlledRecoveryCoordinator.decide(
        operation=_operation("succeeded"),
        receipt=_receipt(execution_status="succeeded", retry_disposition="none"),
        approval=ApprovalContinuationState(
            approval_state="applied",
            continuation_state="completed",
        ),
    )

    assert decision.action == "terminal"
    assert decision.reason_code == "approval_result_already_continued"


def test_unknown_approval_continuation_is_blocked() -> None:
    decision = ControlledRecoveryCoordinator.decide(
        operation=_operation("succeeded"),
        receipt=_receipt(execution_status="succeeded", retry_disposition="none"),
        approval=ApprovalContinuationState(
            approval_state="applied",
            continuation_state="unknown",
        ),
    )

    assert decision.action == "blocked_unknown"
    assert decision.reason_code == "approval_continuation_outcome_unknown"


def test_pending_approval_stops_recovery() -> None:
    decision = ControlledRecoveryCoordinator.decide(
        operation=_operation("precommitted"),
        receipt=_receipt(),
        approval=ApprovalContinuationState(approval_state="pending"),
    )

    assert decision.action == "terminal"
    assert decision.reason_code == "approval_pending"


@pytest.mark.parametrize("approval_state", ["rejected", "expired", "drifted"])
def test_unapplied_approval_can_replan_only_before_side_effect(
    approval_state: str,
) -> None:
    decision = ControlledRecoveryCoordinator.decide(
        operation=_operation("precommitted"),
        receipt=_receipt(),
        approval=ApprovalContinuationState(approval_state=approval_state),  # type: ignore[arg-type]
    )

    assert decision.action == "model_replan"
    assert decision.reason_code == f"approval_{approval_state}_before_side_effect"


def test_unapplied_approval_cannot_replan_after_side_effect() -> None:
    decision = ControlledRecoveryCoordinator.decide(
        operation=_operation("failed"),
        receipt=_receipt(),
        approval=ApprovalContinuationState(approval_state="rejected"),
    )

    assert decision.action == "terminal"
    assert decision.reason_code == "approval_rejected_after_side_effect"


def test_operation_state_rejects_a_boundary_that_cannot_prove_safety() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        TimelineOperationState(
            operation_id="operation-1",
            status="failed",
            side_effect_boundary="not_started",
        )


def test_operation_state_normalizes_its_operation_id() -> None:
    operation = TimelineOperationState(
        operation_id="  operation-1  ",
        status="precommitted",
        side_effect_boundary="not_started",
    )

    assert operation.operation_id == "operation-1"


@pytest.mark.parametrize(
    ("kwargs", "field_name"),
    [
        (
            {
                "operation_id": "operation-1",
                "status": "invalid",
                "side_effect_boundary": "not_started",
            },
            "operation status",
        ),
        (
            {
                "operation_id": "operation-1",
                "status": "precommitted",
                "side_effect_boundary": "invalid",
            },
            "side_effect_boundary",
        ),
    ],
)
def test_operation_state_rejects_unknown_literals(
    kwargs: dict[str, str],
    field_name: str,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        TimelineOperationState(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("execution_status", "retry_disposition", "field_name"),
    [
        ("invalid", "none", "execution_status"),
        ("succeeded", "invalid", "retry_disposition"),
    ],
)
def test_receipt_state_rejects_unknown_literals(
    execution_status: str,
    retry_disposition: str,
    field_name: str,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        ArtifactReceiptState(  # type: ignore[arg-type]
            execution_status=execution_status,
            retry_disposition=retry_disposition,
        )


@pytest.mark.parametrize(
    ("approval_state", "continuation_state", "field_name"),
    [
        ("invalid", "not_required", "approval_state"),
        ("not_required", "invalid", "continuation_state"),
        ("pending", "not_started", "non-applied"),
        ("applied", "not_required", "applied approval"),
    ],
)
def test_approval_state_rejects_invalid_literals_and_cross_field_combinations(
    approval_state: str,
    continuation_state: str,
    field_name: str,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        ApprovalContinuationState(  # type: ignore[arg-type]
            approval_state=approval_state,
            continuation_state=continuation_state,
        )


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        (
            {"action": "invalid", "reason_code": "reason"},
            "recovery action",
        ),
        (
            {"action": "terminal", "reason_code": "  "},
            "reason_code",
        ),
        (
            {
                "action": "new_operation",
                "reason_code": "reason",
                "operation_id": "operation-1",
            },
            "replacement decisions",
        ),
        (
            {
                "action": "corrective_replan",
                "reason_code": "reason",
                "operation_id": "operation-1",
                "supersedes_operation_id": "operation-2",
            },
            "lineage",
        ),
        (
            {
                "action": "approval_continuation",
                "reason_code": "reason",
            },
            "approval_continuation",
        ),
    ],
)
def test_decision_rejects_invalid_action_reason_and_lineage(
    kwargs: dict[str, str],
    error: str,
) -> None:
    from mochi.agents.controlled_recovery import ControlledRecoveryDecision

    with pytest.raises(ValueError, match=error):
        ControlledRecoveryDecision(**kwargs)  # type: ignore[arg-type]


def test_decision_normalizes_lineage_identifiers() -> None:
    from mochi.agents.controlled_recovery import ControlledRecoveryDecision

    decision = ControlledRecoveryDecision(
        action="new_operation",
        reason_code="  known_operation_failure  ",
        operation_id="  operation-1  ",
        supersedes_operation_id="  operation-1  ",
    )

    assert decision.reason_code == "known_operation_failure"
    assert decision.operation_id == "operation-1"
    assert decision.supersedes_operation_id == "operation-1"
