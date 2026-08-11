from __future__ import annotations

import inspect

import pytest

import mochi.runtime.cancellation as cancellation
from mochi.runtime.cancellation import (
    CancellationCapability,
    CancellationCapabilityRegistry,
    CommitFence,
    CommitOutcome,
    CommitState,
)


def test_registry_reports_immediate_deferred_and_unknown_capabilities() -> None:
    registry = CancellationCapabilityRegistry(
        {
            "generation": CancellationCapability.IMMEDIATE,
            "tool": "deferred",
        }
    )

    immediate = registry.resolve("generation", requested_at=7, safe_point="during_model")
    deferred = registry.resolve("tool", safe_point="after_tool")
    unsupported = registry.resolve("exec", safe_point="before_commit")

    assert immediate.capability is CancellationCapability.IMMEDIATE
    assert immediate.requested_at == 7
    assert immediate.reason == "immediate_cancellation_supported"
    assert deferred.capability is CancellationCapability.DEFERRED
    assert deferred.reason == "safe_point_required"
    assert unsupported.capability is CancellationCapability.UNSUPPORTED
    assert unsupported.reason == "unknown_cancellation_target"


def test_pre_commit_cancellation_blocks_the_effect() -> None:
    fence = CommitFence()

    cancelled = fence.request_cancellation(safe_point="before_commit")
    commit = fence.try_commit(safe_point="commit")

    assert cancelled.commit_state is CommitState.CANCELLATION_REQUESTED
    assert cancelled.durable_outcome is CommitOutcome.CANCELLED_PRE_COMMIT
    assert cancelled.commit_permitted is False
    assert commit.durable_outcome is CommitOutcome.CANCELLED_PRE_COMMIT
    assert commit.commit_permitted is False
    assert fence.state is CommitState.CANCELLATION_REQUESTED


def test_post_commit_cancellation_preserves_one_durable_effect() -> None:
    fence = CommitFence()

    committed = fence.try_commit(safe_point="commit")
    cancelled = fence.request_cancellation(safe_point="after_commit")

    assert committed.durable_outcome is CommitOutcome.COMMITTED
    assert committed.commit_permitted is True
    assert cancelled.commit_state is CommitState.COMMITTED
    assert cancelled.durable_outcome is CommitOutcome.ALREADY_COMMITTED
    assert cancelled.commit_permitted is False


def test_commit_fence_cannot_commit_twice() -> None:
    fence = CommitFence()

    first = fence.try_commit(safe_point="commit")
    second = fence.try_commit(safe_point="retry")

    assert first.durable_outcome is CommitOutcome.COMMITTED
    assert second.durable_outcome is CommitOutcome.ALREADY_COMMITTED
    assert second.commit_permitted is False
    assert fence.state is CommitState.COMMITTED


@pytest.mark.parametrize("target", ["", "   ", None, 4])
def test_registry_rejects_invalid_target_names(target: object) -> None:
    registry = CancellationCapabilityRegistry({"generation": "immediate"})

    with pytest.raises(ValueError, match="target"):
        registry.resolve(target)


def test_registry_rejects_invalid_capability_and_request_timestamp() -> None:
    with pytest.raises(ValueError, match="unsupported cancellation capability"):
        CancellationCapabilityRegistry({"generation": "best_effort"})

    registry = CancellationCapabilityRegistry({"generation": "immediate"})
    with pytest.raises(ValueError, match="requested_at"):
        registry.resolve("generation", requested_at=True)


def test_seam_has_no_engine_or_runtime_service_dependency() -> None:
    source = inspect.getsource(cancellation)

    assert "mochi.agents.engine" not in source
    assert "mochi.runtime.service" not in source
