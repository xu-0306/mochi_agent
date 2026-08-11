"""Pure capability and commit-boundary policy for runtime cancellation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from threading import Lock


class CancellationCapability(StrEnum):
    """Truthful cancellation behavior available at a named runtime boundary."""

    IMMEDIATE = "immediate"
    DEFERRED = "deferred"
    UNSUPPORTED = "unsupported"


class CommitState(StrEnum):
    """The only states a side-effect commit fence can expose."""

    OPEN = "open"
    CANCELLATION_REQUESTED = "cancellation_requested"
    COMMITTED = "committed"


class CommitOutcome(StrEnum):
    """Observable outcome for a cancellation or commit request."""

    COMMITTED = "committed"
    CANCELLED_PRE_COMMIT = "cancelled_pre_commit"
    ALREADY_COMMITTED = "already_committed"


@dataclass(frozen=True)
class CancellationDecision:
    """The capability reported for one cancellation request."""

    target: str
    capability: CancellationCapability
    requested_at: int | None
    safe_point: str
    reason: str


@dataclass(frozen=True)
class CommitFenceDecision:
    """A stable decision around the one-way side-effect commit boundary."""

    commit_state: CommitState
    durable_outcome: CommitOutcome
    safe_point: str
    cancellation_requested: bool
    commit_permitted: bool


class CancellationCapabilityRegistry:
    """Resolve named runtime cancellation capabilities without orchestration."""

    def __init__(self, capabilities: Mapping[str, CancellationCapability | str]) -> None:
        normalized: dict[str, CancellationCapability] = {}
        for target, capability in capabilities.items():
            normalized[_normalized_name(target, field="target")] = _coerce_capability(
                capability
            )
        self._capabilities = normalized

    def resolve(
        self,
        target: object,
        *,
        requested_at: int | None = None,
        safe_point: object = "unknown",
    ) -> CancellationDecision:
        """Return unsupported for unknown targets instead of claiming a cancel."""

        normalized_target = _normalized_name(target, field="target")
        normalized_safe_point = _normalized_name(safe_point, field="safe_point")
        _validate_requested_at(requested_at)
        capability = self._capabilities.get(
            normalized_target,
            CancellationCapability.UNSUPPORTED,
        )
        reason = (
            "immediate_cancellation_supported"
            if capability is CancellationCapability.IMMEDIATE
            else "safe_point_required"
            if capability is CancellationCapability.DEFERRED
            else "unknown_cancellation_target"
        )
        return CancellationDecision(
            target=normalized_target,
            capability=capability,
            requested_at=requested_at,
            safe_point=normalized_safe_point,
            reason=reason,
        )


class CommitFence:
    """Allow one commit or one pre-commit cancellation, never both."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._state = CommitState.OPEN

    @property
    def state(self) -> CommitState:
        with self._lock:
            return self._state

    def request_cancellation(self, *, safe_point: object) -> CommitFenceDecision:
        """Close an open fence before a side effect can become durable."""

        normalized_safe_point = _normalized_name(safe_point, field="safe_point")
        with self._lock:
            if self._state is CommitState.COMMITTED:
                return _decision(
                    self._state,
                    CommitOutcome.ALREADY_COMMITTED,
                    normalized_safe_point,
                )
            self._state = CommitState.CANCELLATION_REQUESTED
            return _decision(
                self._state,
                CommitOutcome.CANCELLED_PRE_COMMIT,
                normalized_safe_point,
            )

    def try_commit(self, *, safe_point: object) -> CommitFenceDecision:
        """Commit exactly once unless cancellation already closed the fence."""

        normalized_safe_point = _normalized_name(safe_point, field="safe_point")
        with self._lock:
            if self._state is CommitState.OPEN:
                self._state = CommitState.COMMITTED
                return _decision(
                    self._state,
                    CommitOutcome.COMMITTED,
                    normalized_safe_point,
                )
            if self._state is CommitState.CANCELLATION_REQUESTED:
                return _decision(
                    self._state,
                    CommitOutcome.CANCELLED_PRE_COMMIT,
                    normalized_safe_point,
                )
            return _decision(
                self._state,
                CommitOutcome.ALREADY_COMMITTED,
                normalized_safe_point,
            )


def _decision(
    state: CommitState,
    outcome: CommitOutcome,
    safe_point: str,
) -> CommitFenceDecision:
    return CommitFenceDecision(
        commit_state=state,
        durable_outcome=outcome,
        safe_point=safe_point,
        cancellation_requested=state is CommitState.CANCELLATION_REQUESTED,
        commit_permitted=outcome is CommitOutcome.COMMITTED,
    )


def _coerce_capability(value: CancellationCapability | str) -> CancellationCapability:
    if isinstance(value, CancellationCapability):
        return value
    if not isinstance(value, str):
        raise ValueError("capability must be a CancellationCapability or string")
    try:
        return CancellationCapability(value.strip().lower())
    except ValueError as exc:
        raise ValueError(f"unsupported cancellation capability: {value!r}") from exc


def _normalized_name(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not (normalized := value.strip().lower()):
        raise ValueError(f"{field} must be a non-empty string")
    return normalized


def _validate_requested_at(value: int | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("requested_at must be a non-negative integer or None")
