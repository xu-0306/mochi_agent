"""Pure resource-admission policy for agent-run recovery."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ResourceAction(StrEnum):
    """A scheduler-facing admission outcome without process-handle ownership."""

    ADMIT = "admit"
    DEFER = "defer"


@dataclass(frozen=True)
class ResourceDecision:
    """One bounded admission decision with a stable reason for projection."""

    action: ResourceAction
    reason: str

    @property
    def admitted(self) -> bool:
        return self.action is ResourceAction.ADMIT


@dataclass(frozen=True)
class ResourcePolicy:
    """Bound concurrent run and detached-exec admission before scheduling."""

    max_active_runs: int
    max_detached_execs: int

    def __post_init__(self) -> None:
        _require_positive_int(self.max_active_runs, name="max_active_runs")
        _require_positive_int(self.max_detached_execs, name="max_detached_execs")

    def decide(
        self,
        *,
        active_runs: int,
        detached_execs: int,
        requires_detached_exec: bool,
    ) -> ResourceDecision:
        """Admit only when all required durable capacity is available."""

        _require_non_negative_int(active_runs, name="active_runs")
        _require_non_negative_int(detached_execs, name="detached_execs")
        if not isinstance(requires_detached_exec, bool):
            raise ValueError("requires_detached_exec must be a boolean")
        if active_runs >= self.max_active_runs:
            return ResourceDecision(ResourceAction.DEFER, "active_run_capacity_exhausted")
        if requires_detached_exec and detached_execs >= self.max_detached_execs:
            return ResourceDecision(ResourceAction.DEFER, "detached_exec_capacity_exhausted")
        return ResourceDecision(ResourceAction.ADMIT, "capacity_available")


def _require_positive_int(value: object, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_non_negative_int(value: object, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
