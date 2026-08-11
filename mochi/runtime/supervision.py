"""Runtime-owned helpers for restart-safe standalone AgentRun supervision."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from mochi.runtime.run_adoption import WorkerIdentityState


@dataclass(frozen=True)
class StandaloneAgentRunLease:
    """The fencing token a runtime must retain while it mutates a run."""

    run_id: str
    owner_id: str
    lease_epoch: int


@dataclass(frozen=True)
class DurableResourceCounters:
    """Counts derived solely from persisted AgentRun snapshots."""

    active_runs: int
    detached_execs: int


def is_goal_backed_agent_run(run: Mapping[str, Any]) -> bool:
    """Return whether Goal supervision, rather than standalone leasing, owns a run."""

    summary = run.get("summary")
    if not isinstance(summary, Mapping):
        return False
    return bool(str(cast(Mapping[str, Any], summary).get("goal_id") or "").strip())


def worker_identity_state(
    run: Mapping[str, Any],
    *,
    verified_session_ids: Iterable[str],
) -> WorkerIdentityState:
    """Require every active persisted detached job to have a verified live session."""

    verified = set(verified_session_ids)
    if _has_unverifiable_detached_job(run):
        return WorkerIdentityState.UNVERIFIABLE
    active_session_ids = _active_detached_session_ids(run)
    if not active_session_ids:
        return WorkerIdentityState.VERIFIED
    if all(session_id in verified for session_id in active_session_ids):
        return WorkerIdentityState.VERIFIED
    return WorkerIdentityState.UNVERIFIABLE


def durable_resource_counters(runs: Iterable[Mapping[str, Any]]) -> DurableResourceCounters:
    """Count persisted active runs and detached jobs without consulting PID liveness."""

    snapshots = list(runs)
    detached_session_ids = {
        session_id
        for run in snapshots
        for session_id in _active_detached_session_ids(run)
    }
    return DurableResourceCounters(
        active_runs=sum(
            1
            for run in snapshots
            if str(run.get("status") or "").strip().lower() == "running"
        ),
        detached_execs=len(detached_session_ids),
    )


def committed_completion_status(events: Iterable[Mapping[str, Any]]) -> str | None:
    """Return a terminal completion that needs durable status projection."""

    completion_status: str | None = None
    for event in events:
        if event.get("type") != "run_completion_committed":
            continue
        durable_outcome = str(event.get("durable_outcome") or "").strip().lower()
        status = str(event.get("completion_status") or "").strip().lower()
        if durable_outcome in {"committed", "already_committed"} and status in {
            "failed",
            "succeeded",
        }:
            completion_status = status
    return completion_status


def _active_detached_session_ids(run: Mapping[str, Any]) -> set[str]:
    session_ids: set[str] = set()
    for item in _detached_job_items(run):
        status = str(item.get("status") or "").strip().lower()
        session_id = str(item.get("session_id") or "").strip()
        if session_id and status in {"running", "pending"}:
            session_ids.add(session_id)
    return session_ids


def _has_unverifiable_detached_job(run: Mapping[str, Any]) -> bool:
    for item in _detached_job_items(run):
        status = str(item.get("status") or "").strip().lower()
        identity_state = str(item.get("identity_state") or "").strip().lower()
        if status == "orphaned" or identity_state in {"unknown", "mismatch", "legacy"}:
            return True
    return False


def _detached_job_items(run: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    artifacts_value = run.get("artifacts")
    if not isinstance(artifacts_value, list):
        return []
    artifacts: list[Any] = cast(list[Any], artifacts_value)
    items: list[Mapping[str, Any]] = []
    for raw_artifact in artifacts:
        if not isinstance(raw_artifact, Mapping):
            continue
        artifact = cast(Mapping[str, Any], raw_artifact)
        if artifact.get("artifact_type") != "detached_exec_jobs":
            continue
        metadata_value = artifact.get("metadata")
        if not isinstance(metadata_value, Mapping):
            continue
        metadata = cast(Mapping[str, Any], metadata_value)
        content_value = metadata.get("content")
        if not isinstance(content_value, Mapping):
            continue
        content = cast(Mapping[str, Any], content_value)
        item_values = content.get("items")
        if not isinstance(item_values, list):
            continue
        for raw_item in cast(list[Any], item_values):
            if isinstance(raw_item, Mapping):
                items.append(cast(Mapping[str, Any], raw_item))
    return items
