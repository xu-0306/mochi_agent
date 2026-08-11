from __future__ import annotations

import inspect

import pytest

import mochi.runtime.run_adoption as run_adoption
from mochi.runtime.cancellation import CommitOutcome
from mochi.runtime.run_adoption import (
    LeaseOwnership,
    RunAdoptionClassification,
    RunAdoptionReconciler,
    RunAdoptionSnapshot,
    WorkerIdentityState,
)


def _snapshot(**overrides: object) -> RunAdoptionSnapshot:
    values: dict[str, object] = {
        "run_id": "run-1",
        "status": "running",
        "owner_lease": LeaseOwnership.OWNED,
        "worker_identity": WorkerIdentityState.VERIFIED,
        "checkpoint_revision": 4,
        "projection_sequence": 4,
    }
    values.update(overrides)
    return RunAdoptionSnapshot(**values)  # type: ignore[arg-type]


def test_verified_active_run_is_the_only_snapshot_that_may_be_adopted() -> None:
    decision = RunAdoptionReconciler().classify(_snapshot())

    assert decision.classification is RunAdoptionClassification.ADOPT
    assert decision.may_schedule is True
    assert decision.operator_action is None


@pytest.mark.parametrize(
    ("owner_lease", "worker_identity", "operator_action"),
    [
        (LeaseOwnership.STALE, WorkerIdentityState.VERIFIED, "reacquire_or_confirm_lease"),
        (LeaseOwnership.OWNED, WorkerIdentityState.UNVERIFIABLE, "verify_worker_identity"),
    ],
)
def test_unverifiable_active_snapshots_require_an_operator(
    owner_lease: LeaseOwnership,
    worker_identity: WorkerIdentityState,
    operator_action: str,
) -> None:
    decision = RunAdoptionReconciler().classify(
        _snapshot(owner_lease=owner_lease, worker_identity=worker_identity)
    )

    assert decision.classification is RunAdoptionClassification.OPERATOR_REQUIRED
    assert decision.operator_action == operator_action
    assert decision.may_schedule is False


def test_committed_or_unprojected_work_is_projected_instead_of_rerun() -> None:
    reconciler = RunAdoptionReconciler()

    committed = reconciler.classify(
        _snapshot(durable_outcome=CommitOutcome.ALREADY_COMMITTED)
    )
    unprojected = reconciler.classify(
        _snapshot(status="succeeded", checkpoint_revision=5, projection_sequence=4)
    )

    assert committed.classification is RunAdoptionClassification.PROJECT_COMPLETION
    assert unprojected.classification is RunAdoptionClassification.PROJECT_COMPLETION
    assert not committed.may_schedule
    assert not unprojected.may_schedule


def test_precommit_cancellation_never_reactivates_the_run() -> None:
    decision = RunAdoptionReconciler().classify(
        _snapshot(durable_outcome=CommitOutcome.CANCELLED_PRE_COMMIT)
    )

    assert decision.classification is RunAdoptionClassification.CANCELLED
    assert decision.may_schedule is False


@pytest.mark.parametrize("field", ["checkpoint_revision", "projection_sequence"])
def test_snapshot_rejects_boolean_or_negative_revisions(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        _snapshot(**{field: True})
    with pytest.raises(ValueError, match=field):
        _snapshot(**{field: -1})


def test_seam_has_no_service_or_exec_session_dependency() -> None:
    source = inspect.getsource(run_adoption)

    assert "mochi.runtime.service" not in source
    assert "mochi.runtime.exec_sessions" not in source
