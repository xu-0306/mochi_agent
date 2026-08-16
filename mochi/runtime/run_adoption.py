"""Pure restart-adoption decisions for durable agent-run snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from mochi.runtime.cancellation import CommitOutcome


class RunAdoptionClassification(StrEnum):
    """The only actions a startup supervisor may derive from a run snapshot."""

    ADOPT = "adopt"
    PROJECT_COMPLETION = "project_completion"
    CANCELLED = "cancelled"
    TERMINAL = "terminal"
    OPERATOR_REQUIRED = "operator_required"
    UNKNOWN = "unknown"


class LeaseOwnership(StrEnum):
    """Whether the caller can prove it owns the durable run lease."""

    OWNED = "owned"
    STALE = "stale"
    MISSING = "missing"
    UNKNOWN = "unknown"


class WorkerIdentityState(StrEnum):
    """Whether a durable worker identity can be safely rebound after restart."""

    VERIFIED = "verified"
    MISSING = "missing"
    UNVERIFIABLE = "unverifiable"
    UNKNOWN = "unknown"


_ACTIVE_STATUSES = frozenset({"queued", "running", "cancelling", "recoverable"})
_TERMINAL_STATUSES = frozenset({"completed", "succeeded", "failed", "cancelled", "partial"})


@dataclass(frozen=True)
class RunAdoptionSnapshot:
    """The durable, handle-free state needed to classify one agent run."""

    run_id: str
    status: str
    owner_lease: LeaseOwnership
    worker_identity: WorkerIdentityState
    checkpoint_revision: int
    projection_sequence: int
    durable_outcome: CommitOutcome | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        if not isinstance(self.status, str) or not self.status.strip():
            raise ValueError("status must be a non-empty string")
        _require_non_negative_int(self.checkpoint_revision, name="checkpoint_revision")
        _require_non_negative_int(self.projection_sequence, name="projection_sequence")


@dataclass(frozen=True)
class RunAdoptionDecision:
    """A deterministic decision that leaves scheduling and handles to the caller."""

    run_id: str
    classification: RunAdoptionClassification
    operator_action: str | None
    reason: str

    @property
    def may_schedule(self) -> bool:
        return self.classification is RunAdoptionClassification.ADOPT


class RunAdoptionReconciler:
    """Classify durable run state without performing I/O or acquiring ownership."""

    def classify(self, snapshot: RunAdoptionSnapshot) -> RunAdoptionDecision:
        """Return one restart-safe outcome for a durable run snapshot.

        A missing or unverifiable identity never becomes active after a restart.
        Likewise, a committed effect is projected or escalated instead of rerun.
        """

        status = snapshot.status.strip().lower()
        if snapshot.durable_outcome is CommitOutcome.CANCELLED_PRE_COMMIT:
            return _decision(
                snapshot,
                RunAdoptionClassification.CANCELLED,
                reason="cancelled_pre_commit",
            )
        if snapshot.durable_outcome is CommitOutcome.ALREADY_COMMITTED:
            return _decision(
                snapshot,
                RunAdoptionClassification.PROJECT_COMPLETION,
                reason="already_committed_requires_projection",
            )
        if status in _TERMINAL_STATUSES:
            if snapshot.checkpoint_revision > snapshot.projection_sequence:
                return _decision(
                    snapshot,
                    RunAdoptionClassification.PROJECT_COMPLETION,
                    reason="terminal_checkpoint_not_projected",
                )
            return _decision(snapshot, RunAdoptionClassification.TERMINAL, reason="terminal_run")
        if status not in _ACTIVE_STATUSES:
            return _decision(snapshot, RunAdoptionClassification.UNKNOWN, reason="unknown_run_status")
        if snapshot.owner_lease is not LeaseOwnership.OWNED:
            return _decision(
                snapshot,
                RunAdoptionClassification.OPERATOR_REQUIRED,
                operator_action="reacquire_or_confirm_lease",
                reason="run_lease_not_owned",
            )
        if snapshot.worker_identity is not WorkerIdentityState.VERIFIED:
            return _decision(
                snapshot,
                RunAdoptionClassification.OPERATOR_REQUIRED,
                operator_action="verify_worker_identity",
                reason="worker_identity_not_verified",
            )
        return _decision(snapshot, RunAdoptionClassification.ADOPT, reason="verified_active_run")


def _decision(
    snapshot: RunAdoptionSnapshot,
    classification: RunAdoptionClassification,
    *,
    reason: str,
    operator_action: str | None = None,
) -> RunAdoptionDecision:
    return RunAdoptionDecision(
        run_id=snapshot.run_id,
        classification=classification,
        operator_action=operator_action,
        reason=reason,
    )


def _require_non_negative_int(value: object, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
