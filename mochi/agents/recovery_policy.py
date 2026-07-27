"""Pure, budgeted recovery policy for ordinary-Chat verification failures.

The policy deliberately sits above :mod:`controlled_recovery`.  The existing
coordinator remains the authority that classifies side-effect safety; this
module adds the ordinary-Chat concerns that are orthogonal to execution:
budgets, failed-criterion scope, operation lineage, and bounded corrective
context.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, cast

from mochi.agents.artifact_verifier import RetryDisposition
from mochi.agents.controlled_recovery import (
    ApprovalContinuationState,
    ArtifactReceiptState,
    ControlledRecoveryCoordinator,
    TimelineOperationState,
)
from mochi.agents.outcome_verifier import (
    VerificationReceipt,
)
from mochi.agents.plan_ledger import PlanLedger

RECOVERY_BUDGET_VERSION = "recovery-budget-v1"
RECOVERY_DECISION_VERSION = "recovery-decision-v1"
RECOVERY_CONTEXT_VERSION = "recovery-context-v1"

RecoveryAction = Literal[
    "model_replan",
    "new_operation",
    "corrective_replan",
    "approval_continuation",
    "await_approval",
    "blocked",
    "partial",
    "terminal",
]

_RECOVERY_ACTIONS = frozenset(
    {
        "model_replan",
        "new_operation",
        "corrective_replan",
        "approval_continuation",
        "await_approval",
        "blocked",
        "partial",
        "terminal",
    }
)
_REPLACEMENT_ACTIONS = frozenset({"new_operation", "corrective_replan"})
_RECOVERY_ACTIONS_USING_BUDGET = frozenset(
    {"model_replan", "new_operation", "corrective_replan"}
)
_TERMINAL_VERDICTS = frozenset({"verified", "not_applicable"})
_FAILED_VERDICTS = frozenset({"failed", "unverified"})


class RecoveryPolicyError(ValueError):
    """Invalid recovery contract or an unsafe policy input."""


class RecoveryBudgetExhausted(RecoveryPolicyError):
    """A recovery action would exceed one of the configured budgets."""


def _clean_text(value: object, *, field_name: str, max_chars: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecoveryPolicyError(f"{field_name} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > max_chars:
        raise RecoveryPolicyError(f"{field_name} exceeds {max_chars} characters")
    return normalized


def _clean_optional_text(value: object, *, field_name: str, max_chars: int) -> str | None:
    if value is None:
        return None
    return _clean_text(value, field_name=field_name, max_chars=max_chars)


def _clean_non_negative_int(value: object, *, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise RecoveryPolicyError(f"{field_name} must be a non-negative integer")
    return value


def _clean_non_negative_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise RecoveryPolicyError(f"{field_name} must be a non-negative number")
    return float(value)


def _clean_text_tuple(
    values: object,
    *,
    field_name: str,
    max_items: int,
    max_chars: int,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise RecoveryPolicyError(f"{field_name} must be a sequence")
    items = cast(Sequence[object], values)
    if len(items) > max_items:
        raise RecoveryPolicyError(f"{field_name} exceeds {max_items} items")
    normalized = tuple(
        _clean_text(item, field_name=field_name, max_chars=max_chars) for item in items
    )
    if len(set(normalized)) != len(normalized):
        raise RecoveryPolicyError(f"{field_name} must contain unique values")
    return normalized


def _require_exact_keys(
    payload: Mapping[str, Any], *, expected: frozenset[str], field_name: str
) -> None:
    actual = set(payload)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing fields: {missing}")
        if unexpected:
            details.append(f"unexpected fields: {unexpected}")
        raise RecoveryPolicyError(f"{field_name} " + "; ".join(details))


@dataclass(frozen=True)
class RecoveryBudget:
    """Remaining retry resources for one executing turn.

    A budget is immutable.  The runtime owns the returned value from
    :meth:`reserve_recovery` and persists it in the turn checkpoint.
    """

    budget_version: str
    remaining_attempts: int
    remaining_extra_model_calls: int
    remaining_extra_tool_calls: int
    remaining_extra_wall_seconds: float

    def __post_init__(self) -> None:
        if self.budget_version != RECOVERY_BUDGET_VERSION:
            raise RecoveryPolicyError(f"unsupported budget_version: {self.budget_version!r}")
        for field_name in (
            "remaining_attempts",
            "remaining_extra_model_calls",
            "remaining_extra_tool_calls",
        ):
            object.__setattr__(
                self,
                field_name,
                _clean_non_negative_int(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(
            self,
            "remaining_extra_wall_seconds",
            _clean_non_negative_float(
                self.remaining_extra_wall_seconds,
                field_name="remaining_extra_wall_seconds",
            ),
        )

    @classmethod
    def initial(
        cls,
        *,
        max_attempts: int = 1,
        max_extra_model_calls: int = 1,
        max_extra_tool_calls: int = 4,
        max_extra_wall_seconds: float = 120.0,
    ) -> RecoveryBudget:
        return cls(
            budget_version=RECOVERY_BUDGET_VERSION,
            remaining_attempts=max_attempts,
            remaining_extra_model_calls=max_extra_model_calls,
            remaining_extra_tool_calls=max_extra_tool_calls,
            remaining_extra_wall_seconds=max_extra_wall_seconds,
        )

    @classmethod
    def from_legacy_checkpoint(cls, payload: Mapping[str, Any]) -> RecoveryBudget:
        """Migrate the pre-policy checkpoint shape explicitly."""

        expected = frozenset(
            {
                "remaining_attempts",
                "remaining_extra_model_calls",
                "remaining_extra_tool_calls",
                "remaining_extra_wall_seconds",
            }
        )
        _require_exact_keys(
            payload,
            expected=expected,
            field_name="legacy recovery budget",
        )
        return cls(
            budget_version=RECOVERY_BUDGET_VERSION,
            remaining_attempts=payload["remaining_attempts"],
            remaining_extra_model_calls=payload["remaining_extra_model_calls"],
            remaining_extra_tool_calls=payload["remaining_extra_tool_calls"],
            remaining_extra_wall_seconds=payload["remaining_extra_wall_seconds"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget_version": self.budget_version,
            "remaining_attempts": self.remaining_attempts,
            "remaining_extra_model_calls": self.remaining_extra_model_calls,
            "remaining_extra_tool_calls": self.remaining_extra_tool_calls,
            "remaining_extra_wall_seconds": self.remaining_extra_wall_seconds,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RecoveryBudget:
        expected = frozenset(
            {
                "budget_version",
                "remaining_attempts",
                "remaining_extra_model_calls",
                "remaining_extra_tool_calls",
                "remaining_extra_wall_seconds",
            }
        )
        _require_exact_keys(payload, expected=expected, field_name="recovery budget")
        return cls(
            budget_version=payload["budget_version"],
            remaining_attempts=payload["remaining_attempts"],
            remaining_extra_model_calls=payload["remaining_extra_model_calls"],
            remaining_extra_tool_calls=payload["remaining_extra_tool_calls"],
            remaining_extra_wall_seconds=payload["remaining_extra_wall_seconds"],
        )

    def consume(
        self,
        *,
        attempts: int = 0,
        model_calls: int = 0,
        tool_calls: int = 0,
        wall_seconds: float = 0.0,
    ) -> RecoveryBudget:
        attempts_cost = _clean_non_negative_int(attempts, field_name="attempts")
        model_calls_cost = _clean_non_negative_int(
            model_calls,
            field_name="model_calls",
        )
        tool_calls_cost = _clean_non_negative_int(tool_calls, field_name="tool_calls")
        wall_seconds_cost = _clean_non_negative_float(
            wall_seconds,
            field_name="wall_seconds",
        )
        if attempts_cost > self.remaining_attempts:
            raise RecoveryBudgetExhausted("recovery budget exhausted for attempts")
        if model_calls_cost > self.remaining_extra_model_calls:
            raise RecoveryBudgetExhausted("recovery budget exhausted for model_calls")
        if tool_calls_cost > self.remaining_extra_tool_calls:
            raise RecoveryBudgetExhausted("recovery budget exhausted for tool_calls")
        if wall_seconds_cost > self.remaining_extra_wall_seconds:
            raise RecoveryBudgetExhausted("recovery budget exhausted for wall_seconds")
        return RecoveryBudget(
            budget_version=self.budget_version,
            remaining_attempts=self.remaining_attempts - attempts_cost,
            remaining_extra_model_calls=self.remaining_extra_model_calls - model_calls_cost,
            remaining_extra_tool_calls=self.remaining_extra_tool_calls - tool_calls_cost,
            remaining_extra_wall_seconds=self.remaining_extra_wall_seconds - wall_seconds_cost,
        )

    def reserve_recovery(
        self,
        *,
        model_calls: int = 1,
        tool_calls: int = 0,
        wall_seconds: float = 0.0,
    ) -> RecoveryBudget:
        return self.consume(
            attempts=1,
            model_calls=model_calls,
            tool_calls=tool_calls,
            wall_seconds=wall_seconds,
        )


@dataclass(frozen=True)
class RecoveryDecision:
    """A pure directive for the runtime owner to apply durably."""

    decision_version: str
    action: RecoveryAction
    reason_code: str
    failed_criterion_ids: tuple[str, ...]
    operation_id: str | None
    supersedes_operation_id: str | None
    remaining_budget: RecoveryBudget

    def __post_init__(self) -> None:
        if self.decision_version != RECOVERY_DECISION_VERSION:
            raise RecoveryPolicyError(
                f"unsupported decision_version: {self.decision_version!r}"
            )
        if self.action not in _RECOVERY_ACTIONS:
            raise RecoveryPolicyError(f"unsupported recovery action: {self.action!r}")
        object.__setattr__(
            self,
            "reason_code",
            _clean_text(self.reason_code, field_name="reason_code", max_chars=128),
        )
        object.__setattr__(
            self,
            "failed_criterion_ids",
            _clean_text_tuple(
                self.failed_criterion_ids,
                field_name="failed_criterion_ids",
                max_items=16,
                max_chars=128,
            ),
        )
        for field_name in ("operation_id", "supersedes_operation_id"):
            object.__setattr__(
                self,
                field_name,
                _clean_optional_text(
                    getattr(self, field_name),
                    field_name=field_name,
                    max_chars=128,
                ),
            )
        if type(self.remaining_budget) is not RecoveryBudget:
            raise RecoveryPolicyError("remaining_budget must be a RecoveryBudget")
        if self.action in _REPLACEMENT_ACTIONS:
            if self.operation_id is None or self.supersedes_operation_id is None:
                raise RecoveryPolicyError(
                    "replacement decisions require fresh operation lineage"
                )
            if self.operation_id == self.supersedes_operation_id:
                raise RecoveryPolicyError("replacement operation ID must be fresh")
        elif self.supersedes_operation_id is not None:
            raise RecoveryPolicyError(
                "only replacement decisions may declare supersedes_operation_id"
            )
        if self.action == "approval_continuation" and self.operation_id is None:
            raise RecoveryPolicyError("approval continuation requires operation_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_version": self.decision_version,
            "action": self.action,
            "reason_code": self.reason_code,
            "failed_criterion_ids": list(self.failed_criterion_ids),
            "operation_id": self.operation_id,
            "supersedes_operation_id": self.supersedes_operation_id,
            "remaining_budget": self.remaining_budget.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RecoveryDecision:
        expected = frozenset(
            {
                "decision_version",
                "action",
                "reason_code",
                "failed_criterion_ids",
                "operation_id",
                "supersedes_operation_id",
                "remaining_budget",
            }
        )
        _require_exact_keys(payload, expected=expected, field_name="recovery decision")
        raw_budget = payload["remaining_budget"]
        if not isinstance(raw_budget, Mapping):
            raise RecoveryPolicyError("remaining_budget must be an object")
        budget_payload = cast(Mapping[str, Any], raw_budget)
        return cls(
            decision_version=payload["decision_version"],
            action=cast(RecoveryAction, payload["action"]),
            reason_code=payload["reason_code"],
            failed_criterion_ids=tuple(payload["failed_criterion_ids"]),
            operation_id=payload["operation_id"],
            supersedes_operation_id=payload["supersedes_operation_id"],
            remaining_budget=RecoveryBudget.from_dict(budget_payload),
        )


def _operation_execution_status(
    operation: TimelineOperationState | None,
    *,
    receipt: VerificationReceipt,
) -> Literal["succeeded", "failed", "partial", "unknown"]:
    if operation is None:
        return "failed" if receipt.verdict in _FAILED_VERDICTS else "succeeded"
    if operation.status in {"started", "unknown"}:
        return "unknown"
    if operation.status == "succeeded":
        return "succeeded"
    return "failed"


def _coerce_ledger(value: object) -> PlanLedger | None:
    if value is None or isinstance(value, PlanLedger):
        return value
    if isinstance(value, Mapping):
        return PlanLedger.from_dict(cast(Mapping[str, Any], value))
    raise RecoveryPolicyError("plan_ledger must be a PlanLedger or object")


def _failed_criterion_ids(receipt: VerificationReceipt) -> tuple[str, ...]:
    return tuple(
        criterion.criterion_id
        for criterion in receipt.criteria
        if criterion.verdict in _FAILED_VERDICTS
    )


class RecoveryPolicy:
    """Classify one verification failure without performing recovery."""

    @staticmethod
    def decide(
        *,
        receipt: VerificationReceipt,
        operation: TimelineOperationState | None,
        plan_ledger: PlanLedger | Mapping[str, Any] | None,
        budget: RecoveryBudget,
        approval: ApprovalContinuationState | None = None,
        fresh_operation_id: str | None = None,
    ) -> RecoveryDecision:
        if type(receipt) is not VerificationReceipt:
            raise RecoveryPolicyError("receipt must be a VerificationReceipt")
        if type(budget) is not RecoveryBudget:
            raise RecoveryPolicyError("budget must be a RecoveryBudget")
        _coerce_ledger(plan_ledger)
        failed_ids = _failed_criterion_ids(receipt)

        if receipt.verdict in _TERMINAL_VERDICTS and not receipt.hard_failure:
            return _decision(
                action="terminal",
                reason_code="verification_satisfied",
                failed_criterion_ids=failed_ids,
                operation_id=operation.operation_id if operation else None,
                budget=budget,
            )
        if not receipt.hard_failure:
            return _decision(
                action="terminal",
                reason_code="verification_non_blocking",
                failed_criterion_ids=failed_ids,
                operation_id=operation.operation_id if operation else None,
                budget=budget,
            )

        controlled = ControlledRecoveryCoordinator.decide(
            operation=operation,
            receipt=ArtifactReceiptState(
                execution_status=_operation_execution_status(operation, receipt=receipt),
                retry_disposition=cast(RetryDisposition, receipt.retry_disposition),
            ),
            approval=approval,
        )
        if controlled.action == "blocked_unknown":
            return _decision(
                action="blocked",
                reason_code=controlled.reason_code,
                failed_criterion_ids=failed_ids,
                operation_id=controlled.operation_id,
                budget=budget,
            )
        if controlled.action == "approval_continuation":
            return _decision(
                action="approval_continuation",
                reason_code=controlled.reason_code,
                failed_criterion_ids=failed_ids,
                operation_id=controlled.operation_id,
                budget=budget,
            )
        if controlled.reason_code in {"approval_pending", "receipt_requires_approval"}:
            return _decision(
                action="await_approval",
                reason_code=controlled.reason_code,
                failed_criterion_ids=failed_ids,
                operation_id=controlled.operation_id,
                budget=budget,
            )
        if controlled.action == "terminal":
            return _decision(
                action="terminal",
                reason_code=controlled.reason_code,
                failed_criterion_ids=failed_ids,
                operation_id=controlled.operation_id,
                budget=budget,
            )

        action = cast(RecoveryAction, controlled.action)
        if action not in _RECOVERY_ACTIONS_USING_BUDGET:
            raise RecoveryPolicyError(
                f"unsupported coordinator action for policy: {controlled.action!r}"
            )
        next_budget: RecoveryBudget
        try:
            next_budget = budget.reserve_recovery()
        except RecoveryBudgetExhausted:
            return _decision(
                action="partial",
                reason_code="recovery_budget_exhausted",
                failed_criterion_ids=failed_ids,
                operation_id=controlled.operation_id,
                budget=budget,
            )

        if action in _REPLACEMENT_ACTIONS:
            prior_operation_id = controlled.supersedes_operation_id
            normalized_fresh_id = _clean_optional_text(
                fresh_operation_id,
                field_name="fresh_operation_id",
                max_chars=128,
            )
            if normalized_fresh_id is None:
                return _decision(
                    action="blocked",
                    reason_code="fresh_operation_id_required",
                    failed_criterion_ids=failed_ids,
                    operation_id=controlled.operation_id,
                    budget=budget,
                )
            if prior_operation_id is None or normalized_fresh_id == prior_operation_id:
                return _decision(
                    action="blocked",
                    reason_code="fresh_operation_id_reused",
                    failed_criterion_ids=failed_ids,
                    operation_id=controlled.operation_id,
                    budget=budget,
                )
            return _decision(
                action=action,
                reason_code=controlled.reason_code,
                failed_criterion_ids=failed_ids,
                operation_id=normalized_fresh_id,
                supersedes_operation_id=prior_operation_id,
                budget=next_budget,
            )

        return _decision(
            action=action,
            reason_code=controlled.reason_code,
            failed_criterion_ids=failed_ids,
            operation_id=controlled.operation_id,
            budget=next_budget,
        )

    @staticmethod
    def build_corrective_context(
        *,
        receipt: VerificationReceipt,
        decision: RecoveryDecision,
        plan_ledger: PlanLedger | Mapping[str, Any] | None,
        allowed_targets: Sequence[str] = (),
        max_chars: int = 4_000,
    ) -> Mapping[str, Any]:
        if type(max_chars) is not int or max_chars < 500 or max_chars > 20_000:
            raise RecoveryPolicyError("max_chars must be between 500 and 20000")
        if decision.action not in {"model_replan", "new_operation", "corrective_replan"}:
            raise RecoveryPolicyError("corrective context requires a recovery decision")
        ledger = _coerce_ledger(plan_ledger)
        targets = _clean_text_tuple(
            allowed_targets,
            field_name="allowed_targets",
            max_items=16,
            max_chars=240,
        )
        failed = set(decision.failed_criterion_ids)
        criterion_payload = [
            {
                "criterion_id": criterion.criterion_id,
                "reason_code": criterion.reason_code,
                "evidence_refs": list(criterion.evidence_refs),
            }
            for criterion in receipt.criteria
            if criterion.criterion_id in failed
        ]
        active_item: Mapping[str, Any] | None = None
        if ledger is not None:
            current = next(
                (item for item in ledger.items if item.status == "in_progress"),
                None,
            )
            if current is not None:
                active_item = {
                    "item_id": current.item_id,
                    "title": current.title,
                    "success_criteria": list(current.success_criteria),
                    "attempts": current.attempts,
                }
        context: dict[str, Any] = {
            "context_version": RECOVERY_CONTEXT_VERSION,
            "failed_criteria": criterion_payload,
            "prior_operation_id": decision.supersedes_operation_id or decision.operation_id,
            "allowed_targets": list(targets),
            "active_plan_item": active_item,
            "remaining_budget": decision.remaining_budget.to_dict(),
            "prohibited_repeats": [
                decision.supersedes_operation_id or decision.operation_id
            ],
            "scope_rule": "Correct only failed criteria within the existing task scope.",
            "instruction": (
                "Mint a fresh corrective operation. Do not replay the prior operation, "
                "claim success without new evidence, or expand the task scope."
            ),
        }
        if len(str(context)) > max_chars:
            raise RecoveryPolicyError("corrective context exceeds max_chars")
        return MappingProxyType(context)


def _decision(
    *,
    action: RecoveryAction,
    reason_code: str,
    failed_criterion_ids: tuple[str, ...],
    operation_id: str | None,
    budget: RecoveryBudget,
    supersedes_operation_id: str | None = None,
) -> RecoveryDecision:
    return RecoveryDecision(
        decision_version=RECOVERY_DECISION_VERSION,
        action=action,
        reason_code=reason_code,
        failed_criterion_ids=failed_criterion_ids,
        operation_id=operation_id,
        supersedes_operation_id=supersedes_operation_id,
        remaining_budget=budget,
    )
