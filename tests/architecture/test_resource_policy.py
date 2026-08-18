from __future__ import annotations

import inspect

import pytest

import mochi.runtime.resource_policy as resource_policy
from mochi.runtime.resource_policy import ResourceAction, ResourcePolicy


def test_resource_policy_admits_when_required_capacity_is_available() -> None:
    decision = ResourcePolicy(max_active_runs=2, max_detached_execs=1).decide(
        active_runs=1,
        detached_execs=0,
        requires_detached_exec=True,
    )

    assert decision.action is ResourceAction.ADMIT
    assert decision.admitted is True


def test_resource_policy_defers_at_run_or_detached_exec_capacity() -> None:
    policy = ResourcePolicy(max_active_runs=2, max_detached_execs=1)

    run_limited = policy.decide(
        active_runs=2,
        detached_execs=0,
        requires_detached_exec=False,
    )
    exec_limited = policy.decide(
        active_runs=1,
        detached_execs=1,
        requires_detached_exec=True,
    )

    assert (run_limited.action, run_limited.reason) == (
        ResourceAction.DEFER,
        "active_run_capacity_exhausted",
    )
    assert (exec_limited.action, exec_limited.reason) == (
        ResourceAction.DEFER,
        "detached_exec_capacity_exhausted",
    )


@pytest.mark.parametrize("value", [0, -1, True])
def test_resource_policy_rejects_invalid_limits(value: object) -> None:
    with pytest.raises(ValueError, match="max_active_runs"):
        ResourcePolicy(max_active_runs=value, max_detached_execs=1)  # type: ignore[arg-type]


def test_policy_has_no_service_or_process_handle_dependency() -> None:
    source = inspect.getsource(resource_policy)

    assert "mochi.runtime.service" not in source
    assert "subprocess" not in source
