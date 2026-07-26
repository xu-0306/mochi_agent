"""Pure recovery decisions for durable tool-operation outcomes.

This module deliberately owns no persistence or execution.  Callers provide the
durable timeline state, artifact receipt summary, and approval continuation
state, then apply the returned decision through their own CAS-protected stores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mochi.agents.artifact_verifier import ExecutionStatus, RetryDisposition


TimelineOperationStatus = Literal[
    "precommitted",
    "started",
    "succeeded",
    "failed",
    "unknown",
]
SideEffectBoundary = Literal["not_started", "started", "unknown"]
ApprovalState = Literal[
    "not_required",
    "pending",
    "applied",
    "rejected",
    "expired",
    "drifted",
    "unknown",
]
ContinuationState = Literal[
    "not_required",
    "not_started",
    "continuing",
    "completed",
    "failed",
    "unknown",
]
RecoveryAction = Literal[
    "model_replan",
    "corrective_replan",
    "new_operation",
    "approval_continuation",
    "blocked_unknown",
    "terminal",
]

_TIMELINE_OPERATION_STATUSES = frozenset(
    {"precommitted", "started", "succeeded", "failed", "unknown"}
)
_SIDE_EFFECT_BOUNDARIES = frozenset({"not_started", "started", "unknown"})
_EXECUTION_STATUSES = frozenset({"succeeded", "failed", "partial", "unknown"})
_RETRY_DISPOSITIONS = frozenset(
    {"none", "retryable", "requires_replan", "requires_approval", "terminal"}
)
_APPROVAL_STATES = frozenset(
    {"not_required", "pending", "applied", "rejected", "expired", "drifted", "unknown"}
)
_CONTINUATION_STATES = frozenset(
    {"not_required", "not_started", "continuing", "completed", "failed", "unknown"}
)
_RECOVERY_ACTIONS = frozenset(
    {
        "model_replan",
        "corrective_replan",
        "new_operation",
        "approval_continuation",
        "blocked_unknown",
        "terminal",
    }
)


@dataclass(frozen=True)
class TimelineOperationState:
    """The durable state of one operation descriptor from the turn timeline."""

    operation_id: str
    status: TimelineOperationStatus
    side_effect_boundary: SideEffectBoundary

    def __post_init__(self) -> None:
        if not isinstance(self.operation_id, str) or not self.operation_id.strip():
            raise ValueError("operation_id must be non-empty")
        object.__setattr__(self, "operation_id", self.operation_id.strip())
        _require_literal(
            self.status,
            allowed=_TIMELINE_OPERATION_STATUSES,
            field_name="operation status",
        )
        _require_literal(
            self.side_effect_boundary,
            allowed=_SIDE_EFFECT_BOUNDARIES,
            field_name="side_effect_boundary",
        )
        expected_boundary = {
            "precommitted": "not_started",
            "started": "started",
            "succeeded": "started",
            "failed": "started",
            "unknown": "unknown",
        }[self.status]
        if self.side_effect_boundary != expected_boundary:
            raise ValueError(
                "operation status and side-effect boundary are inconsistent"
            )


@dataclass(frozen=True)
class ArtifactReceiptState:
    """The recovery-relevant subset of an artifact verification receipt."""

    execution_status: ExecutionStatus
    retry_disposition: RetryDisposition

    def __post_init__(self) -> None:
        _require_literal(
            self.execution_status,
            allowed=_EXECUTION_STATUSES,
            field_name="execution_status",
        )
        _require_literal(
            self.retry_disposition,
            allowed=_RETRY_DISPOSITIONS,
            field_name="retry_disposition",
        )


@dataclass(frozen=True)
class ApprovalContinuationState:
    """Whether an approval operation was applied and how its continuation ended."""

    approval_state: ApprovalState = "not_required"
    continuation_state: ContinuationState = "not_required"

    def __post_init__(self) -> None:
        _require_literal(
            self.approval_state,
            allowed=_APPROVAL_STATES,
            field_name="approval_state",
        )
        _require_literal(
            self.continuation_state,
            allowed=_CONTINUATION_STATES,
            field_name="continuation_state",
        )
        if self.approval_state == "applied":
            if self.continuation_state == "not_required":
                raise ValueError(
                    "applied approval requires an explicit continuation_state"
                )
        elif self.continuation_state != "not_required":
            raise ValueError(
                "non-applied approval requires continuation_state='not_required'"
            )


@dataclass(frozen=True)
class ControlledRecoveryDecision:
    """A side-effect-free directive for the runtime owner."""

    action: RecoveryAction
    reason_code: str
    operation_id: str | None = None
    supersedes_operation_id: str | None = None

    def __post_init__(self) -> None:
        _require_literal(
            self.action,
            allowed=_RECOVERY_ACTIONS,
            field_name="recovery action",
        )
        if not isinstance(self.reason_code, str) or not self.reason_code.strip():
            raise ValueError("reason_code must be non-empty")
        object.__setattr__(self, "reason_code", self.reason_code.strip())
        operation_id = _normalize_optional_operation_id(
            self.operation_id,
            field_name="operation_id",
        )
        supersedes_operation_id = _normalize_optional_operation_id(
            self.supersedes_operation_id,
            field_name="supersedes_operation_id",
        )
        object.__setattr__(self, "operation_id", operation_id)
        object.__setattr__(self, "supersedes_operation_id", supersedes_operation_id)
        if self.action in {"new_operation", "corrective_replan"}:
            if operation_id is None or supersedes_operation_id is None:
                raise ValueError(
                    "replacement decisions require operation_id and supersedes_operation_id"
                )
            if operation_id != supersedes_operation_id:
                raise ValueError(
                    "replacement decision lineage must bind the exact prior operation"
                )
        elif supersedes_operation_id is not None:
            raise ValueError("only replacement decisions may declare supersedes_operation_id")
        if self.action == "approval_continuation" and operation_id is None:
            raise ValueError("approval_continuation requires an exact operation_id")


class ControlledRecoveryCoordinator:
    """Classify recovery without authorizing a replay of an old operation."""

    @staticmethod
    def decide(
        *,
        operation: TimelineOperationState | None,
        receipt: ArtifactReceiptState,
        approval: ApprovalContinuationState | None = None,
    ) -> ControlledRecoveryDecision:
        """Return the only safe next step for a tool-operation recovery.

        A model may replan only before an operation crosses its side-effect
        boundary.  A known failed operation can be replaced, never replayed;
        the caller must mint the replacement ID and bind it to
        ``supersedes_operation_id``.  Any uncertain side effect is terminal for
        automation.  An applied approval is represented by its exact result and
        may only continue the original transcript.
        """

        approval_state = approval or ApprovalContinuationState()

        if operation is not None and operation.status in {"started", "unknown"}:
            return ControlledRecoveryCoordinator._blocked_unknown(
                operation,
                "side_effect_outcome_unknown",
            )

        if receipt.execution_status == "unknown":
            return ControlledRecoveryCoordinator._blocked_unknown(
                operation,
                "execution_outcome_unknown",
            )

        if approval_state.approval_state == "unknown":
            return ControlledRecoveryCoordinator._blocked_unknown(
                operation,
                "approval_outcome_unknown",
            )

        if approval_state.approval_state == "applied":
            return ControlledRecoveryCoordinator._applied_approval_decision(
                operation=operation,
                continuation_state=approval_state.continuation_state,
            )

        if approval_state.approval_state == "pending":
            return ControlledRecoveryCoordinator._terminal(
                operation,
                "approval_pending",
            )

        if approval_state.approval_state in {"rejected", "expired", "drifted"}:
            if operation is None or operation.status == "precommitted":
                return ControlledRecoveryCoordinator._model_replan(
                    operation,
                    f"approval_{approval_state.approval_state}_before_side_effect",
                )
            return ControlledRecoveryCoordinator._terminal(
                operation,
                f"approval_{approval_state.approval_state}_after_side_effect",
            )

        if receipt.retry_disposition == "terminal":
            return ControlledRecoveryCoordinator._terminal(
                operation,
                "receipt_terminal",
            )
        if receipt.retry_disposition == "requires_approval":
            return ControlledRecoveryCoordinator._terminal(
                operation,
                "receipt_requires_approval",
            )

        if operation is None:
            if receipt.execution_status == "failed":
                return ControlledRecoveryCoordinator._model_replan(
                    None,
                    "known_pre_side_effect_failure",
                )
            return ControlledRecoveryCoordinator._terminal(
                None,
                "operation_state_missing",
            )

        if operation.status == "precommitted":
            return ControlledRecoveryCoordinator._model_replan(
                operation,
                "operation_not_started",
            )

        if operation.status == "failed":
            if receipt.execution_status in {"failed", "partial"}:
                return ControlledRecoveryCoordinator._new_operation(
                    operation,
                    "known_operation_failure",
                )
            return ControlledRecoveryCoordinator._terminal(
                operation,
                "operation_failure_receipt_inconsistent",
            )

        if operation.status == "succeeded":
            if receipt.execution_status != "succeeded":
                return ControlledRecoveryCoordinator._terminal(
                    operation,
                    "operation_success_receipt_inconsistent",
                )
            if receipt.retry_disposition == "requires_replan":
                return ControlledRecoveryCoordinator._corrective_replan(operation)

        return ControlledRecoveryCoordinator._terminal(
            operation,
            "operation_already_succeeded",
        )

    @staticmethod
    def _applied_approval_decision(
        *,
        operation: TimelineOperationState | None,
        continuation_state: ContinuationState,
    ) -> ControlledRecoveryDecision:
        if operation is None:
            return ControlledRecoveryCoordinator._terminal(
                None,
                "approval_applied_operation_missing",
            )
        if operation.status in {"started", "unknown"}:
            return ControlledRecoveryCoordinator._blocked_unknown(
                operation,
                "approval_applied_operation_outcome_unknown",
            )
        if operation.status == "precommitted":
            return ControlledRecoveryCoordinator._terminal(
                operation,
                "approval_applied_operation_not_completed",
            )
        if continuation_state in {"not_started", "continuing"}:
            return ControlledRecoveryDecision(
                action="approval_continuation",
                reason_code="approval_applied_continue_exact_result",
                operation_id=operation.operation_id,
            )
        if continuation_state == "unknown":
            return ControlledRecoveryCoordinator._blocked_unknown(
                operation,
                "approval_continuation_outcome_unknown",
            )
        return ControlledRecoveryCoordinator._terminal(
            operation,
            "approval_result_already_continued"
            if continuation_state == "completed"
            else "approval_continuation_failed",
        )

    @staticmethod
    def _corrective_replan(
        operation: TimelineOperationState,
    ) -> ControlledRecoveryDecision:
        return ControlledRecoveryDecision(
            action="corrective_replan",
            reason_code="receipt_requires_replan_after_known_success",
            operation_id=operation.operation_id,
            supersedes_operation_id=operation.operation_id,
        )

    @staticmethod
    def _model_replan(
        operation: TimelineOperationState | None,
        reason_code: str,
    ) -> ControlledRecoveryDecision:
        return ControlledRecoveryDecision(
            action="model_replan",
            reason_code=reason_code,
            operation_id=operation.operation_id if operation is not None else None,
        )

    @staticmethod
    def _new_operation(
        operation: TimelineOperationState,
        reason_code: str,
    ) -> ControlledRecoveryDecision:
        return ControlledRecoveryDecision(
            action="new_operation",
            reason_code=reason_code,
            operation_id=operation.operation_id,
            supersedes_operation_id=operation.operation_id,
        )

    @staticmethod
    def _blocked_unknown(
        operation: TimelineOperationState | None,
        reason_code: str,
    ) -> ControlledRecoveryDecision:
        return ControlledRecoveryDecision(
            action="blocked_unknown",
            reason_code=reason_code,
            operation_id=operation.operation_id if operation is not None else None,
        )

    @staticmethod
    def _terminal(
        operation: TimelineOperationState | None,
        reason_code: str,
    ) -> ControlledRecoveryDecision:
        return ControlledRecoveryDecision(
            action="terminal",
            reason_code=reason_code,
            operation_id=operation.operation_id if operation is not None else None,
        )


def _require_literal(value: object, *, allowed: frozenset[str], field_name: str) -> None:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"unsupported {field_name}: {value!r}")


def _normalize_optional_operation_id(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string when provided")
    return value.strip()
