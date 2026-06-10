"""Background runtime service for tasks and approvals."""

from __future__ import annotations

import asyncio
import inspect
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from mochi.agents.multi_agent.execution_policy import (
    execution_policy_to_dict,
    parse_subagent_execution_policy,
)
from mochi.agents.multi_agent.orchestrator import MultiAgentOrchestrator, MultiAgentRunRequest
from mochi.api.routes.chat import _serialize_event
from mochi.backends.inference_capabilities import ReasoningEffort
from mochi.config.schema import SecurityConfig
from mochi.learning.dataset_exporter import export_run_to_dataset_records
from mochi.runtime.agent_run_packages import (
    attempt_id_exists,
    build_attempt_bundle,
    build_dataset_package,
    summarize_package_materialization,
)
from mochi.runtime.approvals import ApprovalStatus, InMemoryApprovalStore
from mochi.runtime.delegate import set_delegate_subagent_task_launcher
from mochi.runtime.exec_runtime import ExecRuntime
from mochi.runtime.models import AgentRunCreateRequest, AgentRunGuidanceRequest, TaskCreateRequest
from mochi.runtime.recovery import (
    build_resource_exhaustion_report,
    classify_agent_run_recovery_issue,
)
from mochi.runtime.store import RuntimeStore
from mochi.security import SecurityDecision
from mochi.security.policy import build_runtime_permission_policy_dict, resolve_runtime_permission_policy
from mochi.tools.exec_command import get_shared_exec_approval_store, get_shared_exec_runtime

TASK_STATUS_RUNNING = {"queued", "running", "resumed"}
AGENT_RUN_TERMINAL_STATUS = {"cancelled", "failed", "succeeded"}
AGENT_RUN_ACTIVE_STATUS = {"running"}
AGENT_RUN_OPERATOR_FINALIZEABLE_STATUS = {"awaiting_resources", "stalled"}
AGENT_RUN_RESUMABLE_STATUS = {"paused", "awaiting_resources", "partial", "stalled"}
DELEGATED_MULTI_AGENT_TASK_TYPE = "delegated_multi_agent"
AGENT_RUN_SUPPORTED_PROTOCOLS = {
    "teacher_student_distill",
    "multi_agent_debate",
    "dr_zero_self_evolve",
    "controlled_subagent_execution",
}
SCHEDULE_RUNTIME_KEYS = {
    "enabled",
    "interval_seconds",
    "run_at",
    "next_run_at",
    "start_immediately",
    "schedule_status",
    "last_scheduled_at",
    "last_completed_at",
    "last_completion_status",
    "attempt_count",
    "recent_attempts",
    "max_runs",
    "auto_pause_on_failure",
    "failure_streak",
    "health_status",
    "current_attempt_id",
    "last_package_error",
    "last_package_completed_at",
}


class RuntimeService:
    """Coordinate background task execution and approval workflow."""

    _DEFAULT_SCHEDULER_POLL_INTERVAL_SECONDS = 30.0

    def __init__(
        self,
        *,
        engine: Any,
        store: RuntimeStore,
        exec_approval_store: InMemoryApprovalStore | None = None,
        exec_runtime: ExecRuntime | None = None,
    ) -> None:
        self._engine = engine
        self._store = store
        self._exec_approval_store = exec_approval_store or get_shared_exec_approval_store()
        self._exec_runtime = exec_runtime or get_shared_exec_runtime()
        self._runtime_tasks_root = Path("sessions") / "runtime-tasks"
        self._active_jobs: dict[str, asyncio.Task[None]] = {}
        self._active_agent_run_jobs: dict[str, asyncio.Task[None]] = {}
        self._security_config = SecurityConfig()
        self._scheduler_poll_interval_seconds = self._DEFAULT_SCHEDULER_POLL_INTERVAL_SECONDS
        self._scheduler_stop_event = asyncio.Event()
        self._scheduler_task: asyncio.Task[None] | None = None
        set_delegate_subagent_task_launcher(self.create_delegated_subagent_task)

    def set_runtime_tasks_root(self, root_dir: Path) -> None:
        self._runtime_tasks_root = Path(root_dir)

    def update_security_config(self, security: SecurityConfig) -> None:
        self._security_config = security

    def set_scheduler_poll_interval(self, seconds: float) -> None:
        self._scheduler_poll_interval_seconds = max(0.05, float(seconds))

    async def start(self) -> None:
        recover_detached = getattr(self._exec_runtime, "recover_detached_sessions", None)
        if callable(recover_detached):
            await recover_detached()
        active = self._scheduler_task
        if active is not None and not active.done():
            return
        self._scheduler_stop_event = asyncio.Event()
        self._scheduler_task = asyncio.create_task(
            self._scheduler_loop(),
            name="runtime-agent-run-scheduler",
        )

    async def close(self) -> None:
        current_loop = asyncio.get_running_loop()

        def _awaitable_in_current_loop(task: asyncio.Task[Any] | None) -> bool:
            if task is None:
                return False
            get_loop = getattr(task, "get_loop", None)
            if not callable(get_loop):
                return True
            try:
                return get_loop() is current_loop
            except RuntimeError:
                return False

        self._scheduler_stop_event.set()
        scheduler_task = self._scheduler_task
        if scheduler_task is not None and not scheduler_task.done():
            scheduler_task.cancel()
            if _awaitable_in_current_loop(scheduler_task):
                try:
                    await scheduler_task
                except asyncio.CancelledError:
                    pass
        self._scheduler_task = None

        for job in list(self._active_jobs.values()):
            if not job.done():
                job.cancel()
        for job in list(self._active_agent_run_jobs.values()):
            if not job.done():
                job.cancel()

        for job in list(self._active_jobs.values()):
            if _awaitable_in_current_loop(job):
                try:
                    await job
                except asyncio.CancelledError:
                    pass
        for job in list(self._active_agent_run_jobs.values()):
            if _awaitable_in_current_loop(job):
                try:
                    await job
                except asyncio.CancelledError:
                    pass

        self._active_jobs.clear()
        self._active_agent_run_jobs.clear()
        close_exec_runtime = getattr(self._exec_runtime, "close", None)
        if callable(close_exec_runtime):
            try:
                signature = inspect.signature(close_exec_runtime)
            except (TypeError, ValueError):
                signature = None
            if signature is not None and "preserve_detached" in signature.parameters:
                await close_exec_runtime(preserve_detached=True)
            else:
                await close_exec_runtime()

    async def create_task(self, payload: TaskCreateRequest) -> dict[str, Any]:
        task_id = str(uuid4())
        project_workspace_dir = payload.project_workspace_dir or payload.workspace_dir
        task_workspace_dir = (self._runtime_tasks_root / task_id / "workspace").resolve()
        task_workspace_dir.mkdir(parents=True, exist_ok=True)
        summary = await self._store.create_task_run(
            task_id=task_id,
            input_text=payload.input,
            session_id=payload.session_id,
            project_id=payload.project_id,
            workspace_dir=project_workspace_dir,
            project_workspace_dir=project_workspace_dir,
            task_workspace_dir=str(task_workspace_dir),
            task_type=payload.task_type,
            metadata=payload.metadata,
            inference_overrides=payload.inference_overrides,
        )
        self._active_jobs[task_id] = asyncio.create_task(
            self._run_task(task_id=task_id),
            name=f"runtime-task-{task_id}",
        )
        summary["pending_approval"] = None
        return _task_summary(summary)

    async def create_delegated_subagent_task(
        self,
        *,
        objective: str,
        protocol: str | None = None,
        protocol_config: dict[str, Any] | None = None,
        session_id: str | None = None,
        project_id: str | None = None,
        workspace_dir: str | None = None,
        suggested_roles: list[str] | None = None,
        suggested_models: dict[str, str] | None = None,
        execution_budget: dict[str, Any] | None = None,
        execution_policy: dict[str, Any] | None = None,
        expected_artifacts: list[str] | None = None,
    ) -> dict[str, Any]:
        effective_protocol = str(protocol or "teacher_student_distill").strip() or "teacher_student_distill"
        raw_protocol_config = dict(protocol_config or {})
        raw_execution_policy = dict(execution_policy or {})
        if effective_protocol == "controlled_subagent_execution" and not raw_protocol_config:
            raw_protocol_config = _controlled_execution_protocol_config(execution_budget or {})
        if not raw_execution_policy and execution_budget:
            raw_execution_policy = {
                "mode": "controlled",
                **_controlled_execution_protocol_config(execution_budget),
            }
        parsed_execution_policy = parse_subagent_execution_policy(
            raw_execution_policy or (raw_protocol_config if effective_protocol == "controlled_subagent_execution" else None),
            legacy_protocol=effective_protocol,
        )
        selected_models_roles = {
            "by_role": dict(suggested_models or {}),
            "subagents": [
                {"role": str(role), "model_id": str((suggested_models or {}).get(str(role), ""))}
                for role in (suggested_roles or [])
                if str(role).strip()
            ],
        }
        payload = TaskCreateRequest(
            input_message=objective,
            session_id=session_id,
            project_id=project_id,
            workspace_dir=workspace_dir,
            task_type=DELEGATED_MULTI_AGENT_TASK_TYPE,
            metadata={
                "protocol": effective_protocol,
                "protocol_config": raw_protocol_config,
                "execution_policy": execution_policy_to_dict(parsed_execution_policy),
                "selected_models_roles": selected_models_roles,
                "expected_artifacts": list(expected_artifacts or []),
            },
        )
        return await self.create_task(payload)

    async def create_agent_run(self, payload: AgentRunCreateRequest) -> dict[str, Any]:
        run_id = str(uuid4())
        schedule = _normalize_agent_run_schedule(payload.schedule)
        run_policy = _normalize_agent_run_policy(
            payload.run_policy,
            defaults=_default_agent_run_policy_from_security(self._security_config),
        )
        summary = dict(payload.summary)
        if payload.reasoning_effort is not None:
            summary["reasoning_effort"] = payload.reasoning_effort
        run = await self._store.create_agent_run(
            run_id=run_id,
            protocol_id=payload.protocol_id,
            title=payload.title,
            topic=payload.topic,
            selected_models_roles=payload.selected_models_roles,
            evaluation_policy=payload.evaluation_policy,
            run_policy=run_policy,
            schedule=schedule,
            summary=summary,
            latest_error=payload.latest_error,
            evidence_status=payload.evidence_status,
            artifacts=[artifact.model_dump() for artifact in payload.artifacts],
        )
        await self._store.append_agent_run_event(
            run_id,
            {
                "type": "run_created",
                "protocol_id": payload.protocol_id,
            },
        )
        return _agent_run_summary(run)

    async def list_agent_runs(self) -> list[dict[str, Any]]:
        runs = await self._store.list_agent_runs()
        return [_agent_run_summary(run) for run in runs]

    async def get_agent_run(self, run_id: str) -> dict[str, Any] | None:
        run = await self._store.get_agent_run(run_id)
        if run is None:
            return None
        payload = _agent_run_summary(run)
        payload["events"] = await self._store.get_agent_run_events(run_id)
        return payload

    async def get_agent_run_attempt_package(
        self,
        run_id: str,
        attempt_id: str,
    ) -> dict[str, Any] | tuple[str, None]:
        run = await self.get_agent_run(run_id)
        if run is None:
            return ("run_not_found", None)
        if not attempt_id_exists(run, attempt_id):
            return ("attempt_not_found", None)
        return build_attempt_bundle(
            run,
            attempt_id=attempt_id,
            selected_scope=attempt_id,
        )

    async def get_agent_run_dataset_package(self, run_id: str) -> dict[str, Any] | None:
        run = await self.get_agent_run(run_id)
        if run is None:
            return None
        return build_dataset_package(run)

    async def get_agent_run_exec_session(
        self,
        run_id: str,
        session_id: str,
        *,
        yield_time_ms: int | None = None,
    ) -> dict[str, Any] | tuple[str, None]:
        run = await self.get_agent_run(run_id)
        if run is None:
            return ("run_not_found", None)
        lease = _find_agent_run_exec_lease(run, session_id)
        if lease is None:
            return ("session_not_found", None)
        poll = await self._exec_runtime.read_session(session_id, yield_time_ms=yield_time_ms)
        return _build_agent_run_exec_session_payload(
            run_id=run_id,
            session_id=session_id,
            lease=lease,
            poll=poll,
        )

    async def reattach_agent_run_exec_session(
        self,
        run_id: str,
        session_id: str,
        *,
        yield_time_ms: int | None = None,
    ) -> dict[str, Any] | tuple[str, None]:
        run = await self.get_agent_run(run_id)
        if run is None:
            return ("run_not_found", None)
        lease = _find_agent_run_exec_lease(run, session_id)
        if lease is None:
            return ("session_not_found", None)
        poll = await self._exec_runtime.read_session(session_id, yield_time_ms=yield_time_ms)
        await self._store.append_agent_run_event(
            run_id,
            {
                "type": "detached_exec_reattached",
                "session_id": session_id,
                "request_id": lease.get("request_id"),
                "live_status": "available" if poll is not None else "unavailable",
                "log_path": lease.get("log_path"),
                "checkpoint_dir": lease.get("checkpoint_dir"),
            },
        )
        payload = _build_agent_run_exec_session_payload(
            run_id=run_id,
            session_id=session_id,
            lease=lease,
            poll=poll,
        )
        payload["reattached"] = True
        payload["reattach_status"] = payload["live_status"]
        return payload

    async def stop_agent_run_exec_session(
        self,
        run_id: str,
        session_id: str,
    ) -> dict[str, Any] | tuple[str, None]:
        run = await self.get_agent_run(run_id)
        if run is None:
            return ("run_not_found", None)
        lease = _find_agent_run_exec_lease(run, session_id)
        if lease is None:
            return ("session_not_found", None)
        poll = await self._exec_runtime.kill_session(session_id)
        stop_status = poll.status.value if poll is not None else "unavailable"
        await self._store.append_agent_run_event(
            run_id,
            {
                "type": "detached_exec_stop",
                "session_id": session_id,
                "request_id": lease.get("request_id"),
                "status": stop_status,
                "log_path": lease.get("log_path"),
                "checkpoint_dir": lease.get("checkpoint_dir"),
            },
        )
        payload = _build_agent_run_exec_session_payload(
            run_id=run_id,
            session_id=session_id,
            lease=lease,
            poll=poll,
        )
        payload["stop_status"] = stop_status
        return payload

    async def start_agent_run(self, run_id: str) -> dict[str, Any] | None:
        run = await self._store.get_agent_run(run_id)
        if run is None:
            return None
        current_status = str(run.get("status") or "created")
        if current_status in AGENT_RUN_TERMINAL_STATUS:
            return _agent_run_summary(run)
        if current_status not in {"created", "paused", "running", "partial", "awaiting_resources", "stalled"}:
            return _agent_run_summary(run)

        if current_status in {"created", "paused", "partial", "awaiting_resources", "stalled"}:
            if current_status in {"created", "partial", "awaiting_resources", "stalled"}:
                await self._begin_agent_run_attempt(run_id, source="manual")
            await self._store.update_agent_run_status(run_id, "running", latest_error=None)
            attempt_id = await self._get_current_attempt_id(run_id)
            await self._store.append_agent_run_event(
                run_id,
                {"type": "run_started", "source": "manual", "attempt_id": attempt_id},
            )

        await self._ensure_agent_run_job(run_id, job_name=f"runtime-agent-run-{run_id}")
        updated = await self._store.get_agent_run(run_id)
        return _agent_run_summary(updated or run)

    async def pause_agent_run(self, run_id: str) -> dict[str, Any] | None:
        run = await self._store.get_agent_run(run_id)
        if run is None:
            return None
        current_status = str(run.get("status") or "created")
        if current_status in AGENT_RUN_TERMINAL_STATUS:
            return _agent_run_summary(run)
        if current_status != "running":
            return _agent_run_summary(run)

        await self._store.update_agent_run_status(
            run_id,
            "paused",
            latest_error=run.get("latest_error"),
        )
        await self._store.append_agent_run_event(run_id, {"type": "run_paused"})

        active = self._active_agent_run_jobs.get(run_id)
        if active is not None and not active.done():
            active.cancel()
            try:
                await active
            except asyncio.CancelledError:
                pass

        updated = await self._store.get_agent_run(run_id)
        return _agent_run_summary(updated or run)

    async def resume_agent_run(
        self,
        run_id: str,
        *,
        strategy: str | None = None,
    ) -> dict[str, Any] | None:
        run = await self._store.get_agent_run(run_id)
        if run is None:
            return None
        current_status = str(run.get("status") or "created")
        if current_status in AGENT_RUN_TERMINAL_STATUS:
            return _agent_run_summary(run)
        if current_status not in AGENT_RUN_RESUMABLE_STATUS:
            return _agent_run_summary(run)

        effective_strategy = _resolve_agent_run_resume_strategy(strategy, run)
        lease = _build_agent_run_resume_lease(
            run_id=run_id,
            strategy=effective_strategy,
            checkpoint_index=_recovery_checkpoint_index_from_run(run),
            source="manual_resume",
        )
        lease_status = await self._store.acquire_agent_run_resume_lease(
            run_id,
            expected_statuses=set(AGENT_RUN_RESUMABLE_STATUS),
            lease=lease,
        )
        if lease_status == "run_not_found":
            return None
        if lease_status in {"already_running", "lease_conflict", "invalid_status"}:
            updated = await self._store.get_agent_run(run_id)
            return _agent_run_summary(updated or run)

        attempt_id = await self._get_current_attempt_id(run_id)
        if current_status in {"awaiting_resources", "partial", "stalled"}:
            attempt_id = await self._begin_agent_run_attempt(run_id, source="resume")
        await self._store.append_agent_run_event(
            run_id,
            {
                "type": "run_resumed",
                "resume_strategy": effective_strategy,
                "resume_lease_id": lease["lease_id"],
                "attempt_id": attempt_id,
            },
        )
        try:
            await self._ensure_agent_run_job(run_id, job_name=f"runtime-agent-run-resume-{run_id}")
        except Exception:
            await self._store.release_agent_run_resume_lease(
                run_id,
                lease_id=str(lease["lease_id"]),
                status="failed_to_start",
            )
            raise
        updated = await self._store.get_agent_run(run_id)
        return _agent_run_summary(updated or run)

    async def cancel_agent_run(self, run_id: str) -> dict[str, Any] | None:
        run = await self._store.get_agent_run(run_id)
        if run is None:
            return None
        current_status = str(run.get("status") or "created")
        if current_status in AGENT_RUN_TERMINAL_STATUS:
            return _agent_run_summary(run)
        active = self._active_agent_run_jobs.get(run_id)
        if active is not None and not active.done():
            active.cancel()
        await self._store.update_agent_run_status(
            run_id,
            "cancelled",
            latest_error=run.get("latest_error"),
        )
        await self._store.append_agent_run_event(
            run_id,
            {"type": "run_cancelled"},
        )
        updated = await self._store.get_agent_run(run_id)
        return _agent_run_summary(updated or run)

    async def append_agent_run_guidance(
        self,
        run_id: str,
        payload: AgentRunGuidanceRequest,
    ) -> dict[str, Any] | None:
        run = await self._store.get_agent_run(run_id)
        if run is None:
            return None
        await self._store.append_agent_run_event(
            run_id,
            {
                "type": "guidance",
                "guidance": payload.guidance,
                "author": payload.author,
                "metadata": payload.metadata,
            },
        )
        updated = await self._store.get_agent_run(run_id)
        if updated is None:
            return None
        response = _agent_run_summary(updated)
        response["events"] = await self._store.get_agent_run_events(run_id)
        return response

    async def finalize_agent_run_partial(self, run_id: str) -> dict[str, Any] | tuple[str, str | None]:
        run = await self._store.get_agent_run(run_id)
        if run is None:
            return ("run_not_found", None)

        current_status = str(run.get("status") or "created")
        if current_status == "partial":
            payload = _agent_run_summary(run)
            payload["events"] = await self._store.get_agent_run_events(run_id)
            return payload
        if current_status not in AGENT_RUN_OPERATOR_FINALIZEABLE_STATUS:
            return ("invalid_status", current_status)

        summary = dict(run.get("summary") or {})
        recovery_state = (
            dict(summary.get("recovery_state"))
            if isinstance(summary.get("recovery_state"), dict)
            else {}
        )
        if recovery_state:
            recovery_state["status"] = "partial"
            recovery_state["action"] = "finalize_partial"
        summary["recovery_state"] = recovery_state

        partial_summary = _build_agent_run_partial_summary(run=run, summary=summary)
        await self._store.update_agent_run_metadata(
            run_id,
            summary=summary,
            latest_error=run.get("latest_error"),
        )
        await self._append_operator_partial_summary_artifact(
            run_id,
            partial_summary=partial_summary,
            recovery_state=recovery_state,
        )
        await self._finalize_agent_run_schedule_partial(run_id)
        await self._store.update_agent_run_status(
            run_id,
            "partial",
            latest_error=run.get("latest_error"),
        )
        await self._store.append_agent_run_event(
            run_id,
            {
                "type": "run_finalized_partial",
                "from_status": current_status,
                "artifact_count": len(run.get("artifacts") or []),
            },
        )

        updated = await self.get_agent_run(run_id)
        if updated is None:
            return ("run_not_found", None)
        return updated

    async def get_agent_run_health(self, run_id: str) -> dict[str, Any] | None:
        run = await self.get_agent_run(run_id)
        if run is None:
            return None
        summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
        recovery_state = (
            run.get("recovery_state")
            if isinstance(run.get("recovery_state"), dict)
            else {}
        )
        recovery_status = str(recovery_state.get("status") or run.get("status") or "created")
        return {
            "run_id": run_id,
            "status": str(run.get("status") or "created"),
            "degraded": bool(run.get("degraded", False)),
            "latest_error": run.get("latest_error"),
            "schedule_health_status": (
                run.get("schedule", {}).get("health_status")
                if isinstance(run.get("schedule"), dict)
                else None
            ),
            "recovery_state": {
                "status": recovery_state.get("status"),
                "action": recovery_state.get("action"),
                "reason": recovery_state.get("reason"),
                "stage": recovery_state.get("stage"),
                "finalize_partial_ready": recovery_status in AGENT_RUN_OPERATOR_FINALIZEABLE_STATUS,
                "recommended_resume_conditions": list(
                    recovery_state.get("recommended_resume_conditions") or []
                ),
                "checkpoint_index": (
                    recovery_state.get("checkpoint", {}).get("checkpoint_index")
                    if isinstance(recovery_state.get("checkpoint"), dict)
                    else None
                ),
            },
            "candidate_count": int(summary.get("candidate_count") or 0),
            "subagent_health_snapshot": _build_operator_subagent_health_summary(run),
            "role_task_snapshot": _build_operator_role_task_summary(run),
            "detached_exec_jobs": _build_operator_detached_exec_summary(run),
        }

    async def _scheduler_loop(self) -> None:
        while True:
            try:
                await self._process_due_agent_runs()
                await asyncio.wait_for(
                    self._scheduler_stop_event.wait(),
                    timeout=self._scheduler_poll_interval_seconds,
                )
                return
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                raise
            except Exception:
                try:
                    await asyncio.wait_for(
                        self._scheduler_stop_event.wait(),
                        timeout=self._scheduler_poll_interval_seconds,
                    )
                    return
                except asyncio.TimeoutError:
                    continue

    async def _process_due_agent_runs(self) -> None:
        runs = await self._store.list_agent_runs()
        now = _utcnow()
        for run in runs:
            schedule = run.get("schedule") if isinstance(run.get("schedule"), dict) else {}
            normalized_schedule = _normalize_agent_run_schedule(schedule)
            if normalized_schedule != schedule:
                await self._store.update_agent_run_schedule(run["id"], normalized_schedule)
                run["schedule"] = normalized_schedule
                schedule = normalized_schedule
            if not _is_agent_run_schedule_due(schedule, now=now):
                continue
            current_status = str(run.get("status") or "created")
            if current_status in {"running", "paused", "cancelled", "stalled"}:
                continue
            await self._trigger_scheduled_agent_run(run, schedule=schedule, now=now)

    async def _trigger_scheduled_agent_run(
        self,
        run: dict[str, Any],
        *,
        schedule: dict[str, Any],
        now: datetime,
    ) -> None:
        run_id = str(run["id"])
        scheduled_for = schedule.get("next_run_at")
        next_schedule = _advance_agent_run_schedule(schedule, triggered_at=now)
        next_schedule["current_attempt_id"] = _new_attempt_id(run_id)
        await self._store.update_agent_run_schedule(run_id, next_schedule)
        await self._store.update_agent_run_status(
            run_id,
            "running",
            latest_error=None,
            reset_started_at=True,
            reset_finished_at=True,
        )
        await self._store.append_agent_run_event(
            run_id,
            {
                "type": "run_scheduled",
                "source": "scheduler",
                "attempt_id": next_schedule["current_attempt_id"],
                "scheduled_for": scheduled_for,
                "triggered_at": _to_iso_utc(now),
                "next_run_at": next_schedule.get("next_run_at"),
            },
        )
        await self._store.append_agent_run_event(
            run_id,
            {
                "type": "run_started",
                "source": "scheduler",
                "attempt_id": next_schedule["current_attempt_id"],
            },
        )
        await self._ensure_agent_run_job(run_id, job_name=f"runtime-agent-run-scheduled-{run_id}")

    async def _begin_agent_run_attempt(self, run_id: str, *, source: str) -> str | None:
        run = await self._store.get_agent_run(run_id)
        if run is None:
            return None
        schedule = run.get("schedule") if isinstance(run.get("schedule"), dict) else {}
        if not _has_runtime_schedule_controls(schedule):
            return None
        normalized_schedule = _normalize_agent_run_schedule(schedule)
        attempt_id = _new_attempt_id(run_id)
        normalized_schedule["current_attempt_id"] = attempt_id
        await self._store.update_agent_run_schedule(run_id, normalized_schedule)
        await self._store.append_agent_run_event(
            run_id,
            {"type": "attempt_created", "source": source, "attempt_id": attempt_id},
        )
        return attempt_id

    async def _get_current_attempt_id(self, run_id: str) -> str | None:
        run = await self._store.get_agent_run(run_id)
        if run is None:
            return None
        schedule = run.get("schedule") if isinstance(run.get("schedule"), dict) else {}
        attempt_id = schedule.get("current_attempt_id")
        return str(attempt_id) if isinstance(attempt_id, str) and attempt_id else None

    async def _ensure_agent_run_job(self, run_id: str, *, job_name: str) -> None:
        active = self._active_agent_run_jobs.get(run_id)
        if active is None or active.done():
            self._active_agent_run_jobs[run_id] = asyncio.create_task(
                self._run_agent_run(run_id=run_id),
                name=job_name,
            )

    async def _update_agent_run_schedule_completion(
        self,
        run_id: str,
        *,
        completion_status: str,
        attempt_id: str | None,
        package_summary: dict[str, Any] | None = None,
    ) -> None:
        run = await self._store.get_agent_run(run_id)
        if run is None:
            return
        schedule = run.get("schedule") if isinstance(run.get("schedule"), dict) else {}
        if not _has_runtime_schedule_controls(schedule):
            return
        next_schedule = dict(schedule)
        completed_at = _now_iso_utc()
        next_schedule["last_completed_at"] = completed_at
        next_schedule["last_completion_status"] = completion_status
        if package_summary is not None:
            next_schedule["last_package_error"] = package_summary.get("last_package_error")
            next_schedule["last_package_completed_at"] = package_summary.get("last_package_completed_at")
        next_schedule["attempt_count"] = int(next_schedule.get("attempt_count", 0) or 0) + 1
        failure_streak = int(next_schedule.get("failure_streak", 0) or 0)
        if completion_status == "succeeded":
            next_schedule["failure_streak"] = 0
            next_schedule["health_status"] = "healthy"
        elif completion_status == "awaiting_resources":
            next_schedule["failure_streak"] = failure_streak
            next_schedule["health_status"] = "awaiting_resources"
        elif completion_status == "partial":
            next_schedule["failure_streak"] = failure_streak
            next_schedule["health_status"] = "partial"
        elif completion_status == "stalled":
            next_schedule["failure_streak"] = failure_streak + 1
            next_schedule["health_status"] = "stalled"
        else:
            next_schedule["failure_streak"] = failure_streak + 1
            next_schedule["health_status"] = "failing"
        next_schedule["recent_attempts"] = _append_schedule_attempt(
            next_schedule,
            run=run,
            completion_status=completion_status,
            completed_at=completed_at,
            attempt_id=attempt_id,
            package_summary=package_summary,
        )
        next_schedule["current_attempt_id"] = None
        max_runs = _parse_optional_positive_int(next_schedule.get("max_runs"))
        if max_runs is not None and int(next_schedule["attempt_count"]) >= max_runs:
            next_schedule["enabled"] = False
            next_schedule["next_run_at"] = None
            next_schedule["schedule_status"] = "max_runs_reached"
            next_schedule["health_status"] = "completed"
        elif completion_status == "awaiting_resources":
            next_schedule["enabled"] = False
            next_schedule["next_run_at"] = None
            next_schedule["schedule_status"] = "awaiting_resources"
        elif completion_status == "stalled":
            next_schedule["enabled"] = False
            next_schedule["next_run_at"] = None
            next_schedule["schedule_status"] = "stalled"
        elif completion_status != "succeeded" and bool(next_schedule.get("auto_pause_on_failure")):
            next_schedule["enabled"] = False
            next_schedule["next_run_at"] = None
            next_schedule["schedule_status"] = "paused_after_failure"
            next_schedule["health_status"] = "paused_after_failure"
        elif not bool(next_schedule.get("enabled")) and next_schedule.get("schedule_status") != "completed":
            next_schedule["schedule_status"] = "disabled"
        elif next_schedule.get("next_run_at"):
            next_schedule["schedule_status"] = "scheduled"
        else:
            next_schedule["schedule_status"] = "completed"
        await self._store.update_agent_run_schedule(run_id, next_schedule)

    async def _run_agent_run(self, *, run_id: str) -> None:
        run = await self.get_agent_run(run_id)
        if run is None:
            return
        schedule = run.get("schedule") if isinstance(run.get("schedule"), dict) else {}
        active_resume_runtime = _resume_runtime_from_run(run)
        active_resume_lease_id = (
            str(active_resume_runtime.get("lease_id"))
            if isinstance(active_resume_runtime.get("lease_id"), str) and active_resume_runtime.get("lease_id")
            else None
        )
        active_resume_strategy = _resolve_agent_run_resume_strategy(None, run)
        active_resume_payload = (
            _resume_payload_from_run(run)
            if active_resume_strategy == "continue_from_checkpoint"
            else None
        )
        attempt_id = (
            str(schedule.get("current_attempt_id"))
            if isinstance(schedule.get("current_attempt_id"), str) and schedule.get("current_attempt_id")
            else None
        )

        try:
            orchestrator = MultiAgentOrchestrator(
                engine=self._engine,
                exec_runtime=self._exec_runtime,
                exec_approval_store=self._exec_approval_store,
                controlled_exec_require_approval=resolve_runtime_permission_policy(
                    self._security_config
                ).require_approval_for_exec,
            )
            protocol_payload = _resolve_protocol_payload(run)
            task_workspace_dir = None
            if run.get("protocol_id") == "controlled_subagent_execution":
                task_workspace_dir = (self._runtime_tasks_root / f"agent-run-{run_id}" / "workspace").resolve()
                task_workspace_dir.mkdir(parents=True, exist_ok=True)
            request = MultiAgentRunRequest(
                run_id=run_id,
                task_input=_agent_run_task_input(run),
                protocol=protocol_payload,
                reasoning_effort=_agent_run_reasoning_effort(run),
                guidance_messages=await self._collect_guidance_messages(run_id),
                run_policy=run.get("run_policy") if isinstance(run.get("run_policy"), dict) else {},
                metadata={
                    "title": run.get("title"),
                    "topic": run.get("topic"),
                    "reasoning_effort": _agent_run_reasoning_effort(run),
                    "selected_models_roles": run.get("selected_models_roles") or {},
                    "evaluation_policy": run.get("evaluation_policy") or {},
                    "schedule": run.get("schedule") or {},
                    "summary": run.get("summary") or {},
                    "workspace_dir": run.get("summary", {}).get("workspace_dir") if isinstance(run.get("summary"), dict) else None,
                    "task_workspace_dir": str(task_workspace_dir) if task_workspace_dir is not None else None,
                    "resume_strategy": active_resume_strategy,
                    "resume_payload": dict(active_resume_payload or {}),
                    "resume_runtime": dict(active_resume_runtime or {}),
                },
            )
            result = await orchestrator.run(request)
            for event in result.events:
                event_payload = event.to_dict()
                if attempt_id is not None and not event_payload.get("attempt_id"):
                    event_payload["attempt_id"] = attempt_id
                await self._store.append_agent_run_event(run_id, event_payload)

            summary = _build_agent_run_summary_payload(run, result)
            summary = _ensure_agent_run_resume_payload(
                run=run,
                summary=summary,
                strategy=active_resume_strategy,
            )
            evidence_status = _build_evidence_status_payload(result)
            await self._store.update_agent_run_metadata(
                run_id,
                summary=summary,
                evidence_status=evidence_status,
                latest_error=None,
            )
            await self._persist_agent_run_artifacts(run_id, result, attempt_id=attempt_id)
            package_summary = await self._materialize_agent_run_packages(
                run_id,
                attempt_id=attempt_id,
            )
            completion_status = _resolve_agent_run_completion_status(result)
            await self._update_agent_run_schedule_completion(
                run_id,
                completion_status=completion_status,
                attempt_id=attempt_id,
                package_summary=package_summary,
            )
            await self._store.update_agent_run_status(
                run_id,
                completion_status,
                latest_error=_agent_run_latest_error(result),
            )
        except asyncio.CancelledError:
            current_run = await self._store.get_agent_run(run_id)
            current_status = str(current_run.get("status") or "cancelled") if current_run else "cancelled"
            if current_status != "paused":
                package_summary = await self._materialize_agent_run_packages(
                    run_id,
                    attempt_id=attempt_id,
                )
                await self._update_agent_run_schedule_completion(
                    run_id,
                    completion_status="cancelled",
                    attempt_id=attempt_id,
                    package_summary=package_summary,
                )
                await self._store.update_agent_run_status(
                    run_id,
                    "cancelled",
                    latest_error="Cancelled by user",
                )
            raise
        except Exception as exc:
            classified = classify_agent_run_recovery_issue(exc)
            if classified is not None:
                recovery_summary, latest_error = _build_runtime_resource_recovery_summary(
                    run=run,
                    run_id=run_id,
                    attempt_id=attempt_id,
                    issue=classified,
                    exception=exc,
                )
                await self._store.append_agent_run_event(
                    run_id,
                    {
                        "type": "run_error",
                        "error": str(exc),
                        "attempt_id": attempt_id,
                        "recovery_status": "awaiting_resources",
                        "recovery_classification": classified.kind,
                    },
                )
                await self._store.update_agent_run_metadata(
                    run_id,
                    summary=recovery_summary,
                    latest_error=latest_error,
                )
                await self._persist_runtime_resource_recovery_artifacts(
                    run_id,
                    run=run,
                    attempt_id=attempt_id,
                    recovery_summary=recovery_summary,
                )
                package_summary = await self._materialize_agent_run_packages(
                    run_id,
                    attempt_id=attempt_id,
                )
                await self._update_agent_run_schedule_completion(
                    run_id,
                    completion_status="awaiting_resources",
                    attempt_id=attempt_id,
                    package_summary=package_summary,
                )
                await self._store.update_agent_run_status(
                    run_id,
                    "awaiting_resources",
                    latest_error=latest_error,
                )
                return
            await self._store.append_agent_run_event(
                run_id,
                {
                    "type": "run_error",
                    "error": str(exc),
                    "attempt_id": attempt_id,
                },
            )
            await self._store.update_agent_run_metadata(
                run_id,
                latest_error=str(exc),
            )
            package_summary = await self._materialize_agent_run_packages(
                run_id,
                attempt_id=attempt_id,
            )
            await self._update_agent_run_schedule_completion(
                run_id,
                completion_status="failed",
                attempt_id=attempt_id,
                package_summary=package_summary,
            )
            await self._store.update_agent_run_status(
                run_id,
                "failed",
                latest_error=str(exc),
            )
        finally:
            if active_resume_lease_id is not None:
                release_status = (
                    "completed"
                    if str((await self._store.get_agent_run(run_id) or {}).get("status") or "") in AGENT_RUN_TERMINAL_STATUS | {"awaiting_resources", "stalled"}
                    else "released"
                )
                await self._store.release_agent_run_resume_lease(
                    run_id,
                    lease_id=active_resume_lease_id,
                    status=release_status,
                )
            active = self._active_agent_run_jobs.get(run_id)
            current = asyncio.current_task()
            if active is not None and (active is current or active.done()):
                self._active_agent_run_jobs.pop(run_id, None)

    async def _persist_agent_run_artifacts(
        self,
        run_id: str,
        result: Any,
        *,
        attempt_id: str | None,
    ) -> None:
        attempt_key = attempt_id or "latest"
        attempt_metadata = {"attempt_id": attempt_id} if attempt_id else {}
        await self._store.append_agent_run_artifact(
            run_id,
            artifact_id=f"{run_id}:attempt:{attempt_key}:run_summary",
            artifact_type="run_summary",
            title="Run Summary",
            uri=f"agent-run://{run_id}/artifacts/{attempt_key}/run_summary",
            mime_type="application/json",
            metadata={**attempt_metadata, "content": dict(result.artifacts or {})},
        )
        verification_summary = (
            result.artifacts.get("verification")
            if isinstance(result.artifacts, dict)
            else None
        )
        evidence_summary = (
            result.artifacts.get("evidence_collection")
            if isinstance(result.artifacts, dict)
            else None
        )
        if isinstance(evidence_summary, dict):
            await self._store.append_agent_run_artifact(
                run_id,
                artifact_id=f"{run_id}:attempt:{attempt_key}:evidence_summary",
                artifact_type="evidence_summary",
                title="Evidence Summary",
                uri=f"agent-run://{run_id}/artifacts/{attempt_key}/evidence_summary",
                mime_type="application/json",
                metadata={**attempt_metadata, "content": evidence_summary},
            )
        if isinstance(verification_summary, dict):
            await self._store.append_agent_run_artifact(
                run_id,
                artifact_id=f"{run_id}:attempt:{attempt_key}:verification_summary",
                artifact_type="verification_summary",
                title="Verification Summary",
                uri=f"agent-run://{run_id}/artifacts/{attempt_key}/verification_summary",
                mime_type="application/json",
                metadata={**attempt_metadata, "content": verification_summary},
            )
            chunk_summary = verification_summary.get("chunk_summaries")
            if isinstance(chunk_summary, list):
                await self._store.append_agent_run_artifact(
                    run_id,
                    artifact_id=f"{run_id}:attempt:{attempt_key}:verification_chunk_summary",
                    artifact_type="verification_chunk_summary",
                    title="Verification Chunk Summary",
                    uri=f"agent-run://{run_id}/artifacts/{attempt_key}/verification_chunk_summary",
                    mime_type="application/json",
                    metadata={**attempt_metadata, "content": {"chunk_summaries": chunk_summary}},
                )
        debate_state = result.artifacts.get("debate_state") if isinstance(result.artifacts, dict) else None
        if isinstance(debate_state, dict):
            await self._store.append_agent_run_artifact(
                run_id,
                artifact_id=f"{run_id}:attempt:{attempt_key}:debate_state",
                artifact_type="debate_state",
                title="Debate State",
                uri=f"agent-run://{run_id}/artifacts/{attempt_key}/debate_state",
                mime_type="application/json",
                metadata={**attempt_metadata, "content": debate_state},
            )
        debate_context_snapshot = (
            result.artifacts.get("debate_context_snapshot") if isinstance(result.artifacts, dict) else None
        )
        if isinstance(debate_context_snapshot, dict):
            await self._store.append_agent_run_artifact(
                run_id,
                artifact_id=f"{run_id}:attempt:{attempt_key}:debate_context_snapshot",
                artifact_type="debate_context_snapshot",
                title="Debate Context Snapshot",
                uri=f"agent-run://{run_id}/artifacts/{attempt_key}/debate_context_snapshot",
                mime_type="application/json",
                metadata={**attempt_metadata, "content": debate_context_snapshot},
            )
        for artifact_type, title in (
            ("run_guard_policy", "Run Guard Policy"),
            ("recovery_checkpoint", "Recovery Checkpoint"),
            ("partial_summary", "Partial Summary"),
            ("resource_exhaustion_report", "Resource Exhaustion Report"),
            ("subagent_health_snapshot", "Subagent Health Snapshot"),
            ("research_plan", "Research Plan"),
            ("source_quality_table", "Source Quality Table"),
            ("claim_evidence_map", "Claim Evidence Map"),
            ("research_brief", "Research Brief"),
            ("subagent_runtime", "Subagent Runtime Trace"),
            ("synthetic_tasks", "Synthetic Tasks"),
            ("solver_rollouts", "Solver Rollouts"),
            ("reward_summary", "Reward Summary"),
            ("curriculum_state", "Curriculum State"),
            ("drzero_iteration_summary", "Dr.Zero Iteration Summary"),
            ("execution_plan", "Execution Plan"),
            ("execution_requests", "Execution Requests"),
            ("controller_decisions", "Controller Decisions"),
            ("execution_results", "Execution Results"),
            ("detached_exec_jobs", "Detached Exec Jobs"),
            ("produced_artifacts", "Produced Artifacts"),
            ("evaluation_summary", "Evaluation Summary"),
            ("controlled_execution_runtime", "Controlled Execution Runtime"),
        ):
            artifact_content = result.artifacts.get(artifact_type) if isinstance(result.artifacts, dict) else None
            if not isinstance(artifact_content, dict):
                continue
            await self._store.append_agent_run_artifact(
                run_id,
                artifact_id=f"{run_id}:attempt:{attempt_key}:{artifact_type}",
                artifact_type=artifact_type,
                title=title,
                uri=f"agent-run://{run_id}/artifacts/{attempt_key}/{artifact_type}",
                mime_type="application/json",
                metadata={**attempt_metadata, "content": artifact_content},
            )
        dataset_records = export_run_to_dataset_records(result)
        for index, record in enumerate(dataset_records, start=1):
            await self._store.append_agent_run_artifact(
                run_id,
                artifact_id=f"{run_id}:attempt:{attempt_key}:dataset:{index}",
                artifact_type="dataset_record",
                title=f"Dataset Record {index}",
                uri=f"agent-run://{run_id}/artifacts/{attempt_key}/dataset-record-{index}",
                mime_type="application/json",
                metadata={**attempt_metadata, "record": record},
            )

    async def _persist_runtime_resource_recovery_artifacts(
        self,
        run_id: str,
        *,
        run: dict[str, Any],
        attempt_id: str | None,
        recovery_summary: dict[str, Any],
    ) -> None:
        attempt_key = attempt_id or "latest"
        attempt_metadata = {"attempt_id": attempt_id} if attempt_id else {}
        run_policy = run.get("run_policy") if isinstance(run.get("run_policy"), dict) else {}
        recovery_state = (
            recovery_summary.get("recovery_state")
            if isinstance(recovery_summary.get("recovery_state"), dict)
            else {}
        )
        checkpoint = (
            recovery_state.get("checkpoint")
            if isinstance(recovery_state.get("checkpoint"), dict)
            else {}
        )
        partial_summary = {
            "status": recovery_state.get("status"),
            "reason": recovery_state.get("reason"),
            "stage": recovery_state.get("stage"),
            "checkpoint_index": checkpoint.get("checkpoint_index"),
            "candidate_count": int(recovery_summary.get("candidate_count") or 0),
            "artifact_keys": [],
            "unfinished_steps": list(recovery_state.get("unfinished_steps") or []),
        }
        resource_report = (
            recovery_state.get("resource_exhaustion_report")
            if isinstance(recovery_state.get("resource_exhaustion_report"), dict)
            else {}
        )
        for artifact_type, title, content in (
            ("run_guard_policy", "Run Guard Policy", run_policy),
            ("recovery_checkpoint", "Recovery Checkpoint", checkpoint),
            ("partial_summary", "Partial Summary", partial_summary),
            ("resource_exhaustion_report", "Resource Exhaustion Report", resource_report),
        ):
            if not isinstance(content, dict) or not content:
                continue
            await self._store.append_agent_run_artifact(
                run_id,
                artifact_id=f"{run_id}:attempt:{attempt_key}:{artifact_type}",
                artifact_type=artifact_type,
                title=title,
                uri=f"agent-run://{run_id}/artifacts/{attempt_key}/{artifact_type}",
                mime_type="application/json",
                metadata={**attempt_metadata, "content": content},
            )

    async def _materialize_agent_run_packages(
        self,
        run_id: str,
        *,
        attempt_id: str | None,
    ) -> dict[str, Any]:
        try:
            run = await self.get_agent_run(run_id)
            if run is None:
                raise RuntimeError("Agent run not found during package materialization")
            attempt_bundle = build_attempt_bundle(
                run,
                attempt_id=attempt_id,
                selected_scope=attempt_id or "latest",
            )
            dataset_package = build_dataset_package(run)
            attempt_key = attempt_id or "latest"
            await self._store.append_agent_run_artifact(
                run_id,
                artifact_id=f"{run_id}:attempt:{attempt_key}:attempt-bundle",
                artifact_type="attempt_bundle",
                title="Attempt Bundle",
                uri=f"agent-run://{run_id}/artifacts/{attempt_key}/attempt-bundle",
                mime_type="application/json",
                metadata={
                    "attempt_id": attempt_id,
                    "manifest_version": attempt_bundle["manifest_version"],
                    "package_type": attempt_bundle["package_type"],
                    "artifact_count": attempt_bundle.get("artifact_count", 0),
                    "record_count": len(attempt_bundle.get("dataset_records") or []),
                    "package_ready": True,
                    "content": attempt_bundle,
                },
            )
            training_ready_count = _dataset_training_ready_count(
                dataset_package,
                attempt_id=attempt_id,
            )
            await self._store.append_agent_run_artifact(
                run_id,
                artifact_id=f"{run_id}:attempt:{attempt_key}:dataset-package-snapshot",
                artifact_type="dataset_package_snapshot",
                title="Dataset Package Snapshot",
                uri=f"agent-run://{run_id}/artifacts/{attempt_key}/dataset-package-snapshot",
                mime_type="application/json",
                metadata={
                    "attempt_id": attempt_id,
                    "manifest_version": dataset_package["manifest_version"],
                    "package_type": "dataset_package_snapshot",
                    "artifact_count": len(dataset_package.get("attempts") or []),
                    "record_count": dataset_package.get("dataset_record_count", 0),
                    "training_ready_count": training_ready_count,
                    "package_ready": True,
                    "content": dataset_package,
                },
            )
            return summarize_package_materialization(
                attempt_bundle,
                dataset_package,
                attempt_id=attempt_id,
            )
        except Exception as exc:
            await self._store.append_agent_run_event(
                run_id,
                {
                    "type": "package_error",
                    "error": str(exc),
                    "attempt_id": attempt_id,
                },
            )
            return {
                "attempt_id": attempt_id or "unscoped",
                "artifact_count": 0,
                "dataset_record_count": 0,
                "training_ready_count": 0,
                "package_ready": False,
                "last_package_error": str(exc),
                "last_package_completed_at": None,
            }

    async def _collect_guidance_messages(self, run_id: str) -> list[str]:
        events = await self._store.get_agent_run_events(run_id)
        messages: list[str] = []
        for event in events:
            event_type = str(event.get("type", ""))
            if event_type != "guidance":
                continue
            text = event.get("guidance")
            if isinstance(text, str) and text.strip():
                messages.append(text.strip())
                continue
            payload = event.get("payload")
            if isinstance(payload, dict):
                payload_text = payload.get("content")
                if isinstance(payload_text, str) and payload_text.strip():
                    messages.append(payload_text.strip())
        return messages

    async def list_tasks(self) -> list[dict[str, Any]]:
        tasks = await self._store.list_task_runs()
        items: list[dict[str, Any]] = []
        for task in tasks:
            task["pending_approval"] = await self._store.get_pending_approval_for_task(task["id"])
            items.append(_task_summary(task))
        return items

    async def get_task(self, task_id: str) -> dict[str, Any] | None:
        task = await self._store.get_task_run(task_id)
        if task is None:
            return None
        events = await self._store.get_task_events(task_id)
        pending_approval = await self._store.get_pending_approval_for_task(task_id)
        task["pending_approval"] = pending_approval
        payload = _task_summary(task)
        payload["events"] = events
        payload["pending_approval"] = pending_approval
        return payload

    async def cancel_task(self, task_id: str) -> dict[str, Any] | None:
        task = await self._store.get_task_run(task_id)
        if task is None:
            return None
        if task["status"] not in TASK_STATUS_RUNNING and task["status"] != "awaiting_approval":
            return _task_summary(task)

        if task["status"] in TASK_STATUS_RUNNING:
            active = self._active_jobs.get(task_id)
            if active is not None and not active.done():
                active.cancel()
        await self._store.update_task_status(task_id, "cancelled", error="Cancelled by user")
        return _task_summary((await self._store.get_task_run(task_id)) or task)

    async def resume_task(
        self,
        task_id: str,
        *,
        approved: bool,
        reason: str | None = None,
    ) -> dict[str, Any] | None:
        task = await self._store.get_task_run(task_id)
        if task is None:
            return None

        approval = await self._store.get_pending_approval_for_task(task_id)
        if approval is None:
            return _task_summary(task)

        resolved = await self.resolve_approval(
            approval["id"],
            approved=approved,
            reason=reason,
        )
        if resolved is None:
            return None
        return _task_summary((await self._store.get_task_run(task_id)) or task)

    async def list_approvals(self, *, status: str | None = None) -> list[dict[str, Any]]:
        approvals = await self._store.list_approval_requests(status=status)
        linked_exec_approval_ids = {
            linked_id
            for linked_id in (_linked_exec_approval_id(item.get("metadata")) for item in approvals)
            if linked_id
        }

        summaries = [_approval_summary(item) for item in approvals]
        exec_status = _to_exec_approval_status(status)
        if status is None or exec_status is not None:
            for exec_approval in self._exec_approval_store.list(status=exec_status):
                if exec_approval.approval_id in linked_exec_approval_ids:
                    continue
                summaries.append(_exec_approval_summary(exec_approval))

        summaries.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
        return summaries

    async def resolve_approval(
        self,
        approval_id: str,
        *,
        approved: bool,
        reason: str | None = None,
    ) -> dict[str, Any] | None:
        current = await self._store.get_approval_request(approval_id)
        if current is None:
            existing_exec = self._exec_approval_store.get(approval_id)
            if existing_exec is None:
                return None
            execution_result: dict[str, Any] | None = None
            if approved:
                execution_result = await self._execute_approved_exec_request(existing_exec)
            resolved_exec = self._exec_approval_store.resolve(
                approval_id,
                approved=approved,
                reason=reason,
                execution_result=execution_result,
            )
            if resolved_exec is None:
                return None
            return _exec_approval_summary(resolved_exec)
        approval = await self._store.resolve_approval_request(
            approval_id,
            approved=approved,
            reason=reason,
        )
        if approval is None:
            return None
        linked_exec_approval_id = _linked_exec_approval_id(current.get("metadata"))
        if linked_exec_approval_id:
            self._exec_approval_store.resolve(
                linked_exec_approval_id,
                approved=approved,
                reason=reason,
            )
        if not approved:
            await self._store.update_task_status(
                approval["task_id"],
                "cancelled",
                error="Approval rejected",
            )
            return self._task_approval_summary(approval)
        if await self._try_complete_task_with_linked_exec(
            approval,
            current=current,
            linked_exec_approval_id=linked_exec_approval_id,
            reason=reason,
        ):
            return self._task_approval_summary(approval)
        override = _approval_override(current)
        await self._store.update_task_status(
            approval["task_id"],
            "resumed",
            error=None,
            permission_override=override,
        )
        self._active_jobs[approval["task_id"]] = asyncio.create_task(
            self._run_task(task_id=approval["task_id"], permission_override=override),
            name=f"runtime-task-resume-{approval['task_id']}",
        )
        return self._task_approval_summary(approval)

    async def _execute_approved_exec_request(self, approval: Any) -> dict[str, Any]:
        payload = getattr(approval, "command_payload", None)
        if not isinstance(payload, dict):
            return {
                "status": "execution_unavailable",
                "error": "Approved exec request did not include a replay payload.",
            }

        workdir = payload.get("workdir")
        try:
            poll = await self._exec_runtime.start_command(
                command=str(payload.get("command") or ""),
                shell=str(payload.get("shell") or "auto"),
                cwd=str(workdir) if isinstance(workdir, str) and workdir else None,
                env=payload.get("env") if isinstance(payload.get("env"), dict) else None,
                timeout_sec=float(payload.get("timeout_sec")) if payload.get("timeout_sec") is not None else None,
                background=bool(payload.get("background", False)),
                tty=bool(payload.get("tty", False)),
                approval_state=str(payload.get("approval_state") or "approved"),
                log_path=str(payload.get("log_path")) if isinstance(payload.get("log_path"), str) else None,
                checkpoint_dir=(
                    str(payload.get("checkpoint_dir"))
                    if isinstance(payload.get("checkpoint_dir"), str)
                    else None
                ),
            )
        except Exception as exc:
            return {
                "status": "execution_failed",
                "error": str(exc),
            }

        return {
            "status": poll.status.value,
            "session_id": poll.session_id,
            "shell": poll.shell,
            "background": poll.background,
            "tty": poll.tty,
            "pid": poll.pid,
            "exit_code": poll.exit_code,
            "timed_out": poll.timed_out,
            "approval_state": poll.approval_state,
            "stdout": poll.stdout,
            "stderr": poll.stderr,
            "log_path": str(payload.get("log_path")) if isinstance(payload.get("log_path"), str) else None,
            "checkpoint_dir": (
                str(payload.get("checkpoint_dir"))
                if isinstance(payload.get("checkpoint_dir"), str)
                else None
            ),
        }

    def _task_approval_summary(self, approval: dict[str, Any]) -> dict[str, Any]:
        summary = _approval_summary(approval)
        linked_exec_approval_id = summary.get("exec_approval_id")
        if not isinstance(linked_exec_approval_id, str) or not linked_exec_approval_id:
            return summary
        linked = self._exec_approval_store.get(linked_exec_approval_id)
        if linked is None:
            return summary
        summary["exec_status"] = linked.status
        summary["exec_session_id"] = _exec_result_field(linked.execution_result, "session_id")
        summary["execution_result"] = linked.execution_result
        return summary

    async def _try_complete_task_with_linked_exec(
        self,
        approval: dict[str, Any],
        *,
        current: dict[str, Any],
        linked_exec_approval_id: str | None,
        reason: str | None,
    ) -> bool:
        if str(current.get("tool_name") or "") != "exec_command":
            return False
        if not linked_exec_approval_id:
            return False

        linked = self._exec_approval_store.get(linked_exec_approval_id)
        if linked is None:
            await self._store.update_task_status(
                approval["task_id"],
                "failed",
                error="Linked exec approval not found.",
            )
            return True

        await self._store.update_task_status(approval["task_id"], "resumed", error=None)
        execution_result = await self._execute_approved_exec_request(linked)
        self._exec_approval_store.resolve(
            linked_exec_approval_id,
            approved=True,
            reason=reason,
            execution_result=execution_result,
        )
        await self._append_task_exec_result_event(
            approval=approval,
            linked_exec_approval_id=linked_exec_approval_id,
            execution_result=execution_result,
        )

        task_status = str(execution_result.get("status") or "")
        final_answer = _final_answer_from_exec_result(execution_result)
        if task_status in {"completed", "running"}:
            if final_answer is not None:
                await self._store.append_task_event(
                    approval["task_id"],
                    {
                        "type": "final_answer",
                        "content": final_answer,
                        "trajectory_id": None,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "generation_time_ms": 0.0,
                        "finish_reason": "tool_completed",
                    },
                )
            await self._store.update_task_status(
                approval["task_id"],
                "succeeded",
                error=None,
                final_answer=final_answer,
            )
            return True

        error_message = _exec_error_message(execution_result)
        await self._store.append_task_event(
            approval["task_id"],
            {
                "type": "error",
                "error": error_message,
                "code": "EXEC_APPROVAL_EXECUTION_FAILED",
                "metadata": {
                    "approval_id": linked_exec_approval_id,
                    "execution_result": execution_result,
                },
            },
        )
        await self._store.update_task_status(
            approval["task_id"],
            "failed",
            error=error_message,
            final_answer=final_answer,
        )
        return True

    async def _append_task_exec_result_event(
        self,
        *,
        approval: dict[str, Any],
        linked_exec_approval_id: str,
        execution_result: dict[str, Any],
    ) -> None:
        result_status = str(execution_result.get("status") or "execution_failed")
        await self._store.append_task_event(
            approval["task_id"],
            {
                "type": "tool_call_result",
                "call_id": str(approval.get("call_id") or ""),
                "tool_name": str(approval.get("tool_name") or "exec_command"),
                "result": execution_result,
                "error": (
                    None
                    if result_status in {"completed", "running"}
                    else _exec_error_message(execution_result)
                ),
                "metadata": {
                    "status": result_status,
                    "approval_id": linked_exec_approval_id,
                    "session_id": execution_result.get("session_id"),
                    "timed_out": bool(execution_result.get("timed_out", False)),
                    "exit_code": execution_result.get("exit_code"),
                },
            },
        )

    async def _run_task(
        self,
        *,
        task_id: str,
        permission_override: dict[str, Any] | None = None,
    ) -> None:
        task = await self._store.get_task_run(task_id)
        if task is None:
            return

        await self._store.update_task_status(task_id, "running", error=None)
        pending_requests: dict[str, dict[str, Any]] = {}
        final_answer: str | None = None

        try:
            task_type = str(task.get("task_type") or "")
            if task_type in {DELEGATED_MULTI_AGENT_TASK_TYPE, "controlled_subagent_execution"}:
                await self._run_delegated_multi_agent_task(task=task, task_id=task_id)
                return

            effective_permission_policy = build_runtime_permission_policy_dict(
                self._security_config,
                overrides=permission_override,
            )
            stream = self._engine.chat(
                task["input"],
                session_id=task.get("session_id"),
                inference_overrides=task.get("inference_overrides") or {},
                project_id=task.get("project_id"),
                workspace_dir=task.get("project_workspace_dir") or task.get("workspace_dir"),
                task_workspace_dir=task.get("task_workspace_dir"),
                permission_policy=effective_permission_policy,
            )
            async for event in stream:
                serialized = _serialize_event(event)
                await self._store.append_task_event(task_id, serialized)
                if serialized.get("type") == "final_answer":
                    content = serialized.get("content")
                    if isinstance(content, str):
                        final_answer = content

                if serialized.get("type") == "tool_call_request":
                    call_id = serialized.get("call_id")
                    if isinstance(call_id, str) and call_id:
                        pending_requests[call_id] = serialized
                    continue

                if serialized.get("type") != "tool_call_result":
                    continue
                metadata = serialized.get("metadata")
                requires_approval = (
                    isinstance(metadata, dict) and metadata.get("requires_approval") is True
                )
                if not requires_approval:
                    continue

                call_id = serialized.get("call_id")
                if not isinstance(call_id, str) or call_id not in pending_requests:
                    continue
                request_event = pending_requests[call_id]
                approval_metadata = metadata if isinstance(metadata, dict) else {}
                await self._store.create_approval_request(
                    approval_id=str(uuid4()),
                    task_id=task_id,
                    call_id=call_id,
                    tool_name=str(request_event.get("tool_name", "")),
                    arguments=dict(request_event.get("arguments") or {}),
                    metadata=approval_metadata,
                )
                await self._store.update_task_status(task_id, "awaiting_approval", final_answer=final_answer)
                return

            await self._store.update_task_status(task_id, "succeeded", final_answer=final_answer)
        except asyncio.CancelledError:
            await self._store.update_task_status(task_id, "cancelled", error="Cancelled by user", final_answer=final_answer)
            raise
        except Exception as exc:
            await self._store.update_task_status(task_id, "failed", error=str(exc), final_answer=final_answer)
        finally:
            active = self._active_jobs.get(task_id)
            current = asyncio.current_task()
            if active is not None and (active is current or active.done()):
                self._active_jobs.pop(task_id, None)

    async def _run_delegated_multi_agent_task(self, *, task: dict[str, Any], task_id: str) -> None:
        metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        legacy_protocol = (
            "controlled_subagent_execution"
            if str(task.get("task_type") or "") == "controlled_subagent_execution"
            else "teacher_student_distill"
        )
        protocol_config = (
            metadata.get("protocol_config") if isinstance(metadata.get("protocol_config"), dict) else {}
        )
        protocol_id = str(metadata.get("protocol") or legacy_protocol).strip() or legacy_protocol
        execution_policy = (
            metadata.get("execution_policy") if isinstance(metadata.get("execution_policy"), dict) else {}
        )
        selected_models_roles = (
            metadata.get("selected_models_roles")
            if isinstance(metadata.get("selected_models_roles"), dict)
            else {}
        )
        orchestrator = MultiAgentOrchestrator(
            engine=self._engine,
            exec_runtime=self._exec_runtime,
            exec_approval_store=self._exec_approval_store,
            controlled_exec_require_approval=resolve_runtime_permission_policy(
                self._security_config
            ).require_approval_for_exec,
        )
        result = await orchestrator.run(
            MultiAgentRunRequest(
                run_id=task_id,
                task_input=str(task.get("input") or ""),
                protocol={"protocol": protocol_id, **protocol_config, "execution_policy": execution_policy},
                reasoning_effort=_coerce_reasoning_effort(metadata.get("reasoning_effort")),
                guidance_messages=[],
                metadata={
                    "reasoning_effort": _coerce_reasoning_effort(metadata.get("reasoning_effort")),
                    "selected_models_roles": selected_models_roles,
                    "summary": metadata,
                    "execution_policy": execution_policy,
                    "workspace_dir": task.get("project_workspace_dir") or task.get("workspace_dir"),
                    "task_workspace_dir": task.get("task_workspace_dir"),
                },
            )
        )
        for event in result.events:
            await self._store.append_task_event(task_id, event.to_dict())
        final_answer = str((result.artifacts or {}).get("final_answer") or "")
        await self._store.update_task_status(
            task_id,
            "succeeded" if result.state == "succeeded" else str(result.state),
            final_answer=final_answer,
            error=None if result.state == "succeeded" else str(result.state),
        )

    async def _append_operator_partial_summary_artifact(
        self,
        run_id: str,
        *,
        partial_summary: dict[str, Any],
        recovery_state: dict[str, Any],
    ) -> None:
        await self._store.append_agent_run_artifact(
            run_id,
            artifact_id=f"{run_id}:operator:partial_summary:{uuid4()}",
            artifact_type="partial_summary",
            title="Partial Summary",
            uri=f"agent-run://{run_id}/artifacts/operator/partial-summary",
            mime_type="application/json",
            metadata={
                "operator_action": "finalize_partial",
                "recovery_status": recovery_state.get("status"),
                "content": partial_summary,
            },
        )
        checkpoint = recovery_state.get("checkpoint")
        if isinstance(checkpoint, dict) and checkpoint:
            await self._store.append_agent_run_artifact(
                run_id,
                artifact_id=f"{run_id}:operator:recovery_checkpoint:{uuid4()}",
                artifact_type="recovery_checkpoint",
                title="Recovery Checkpoint",
                uri=f"agent-run://{run_id}/artifacts/operator/recovery-checkpoint",
                mime_type="application/json",
                metadata={
                    "operator_action": "finalize_partial",
                    "content": checkpoint,
                },
            )

    async def _finalize_agent_run_schedule_partial(self, run_id: str) -> None:
        run = await self._store.get_agent_run(run_id)
        if run is None:
            return
        schedule = run.get("schedule") if isinstance(run.get("schedule"), dict) else {}
        if not _has_runtime_schedule_controls(schedule):
            return
        next_schedule = dict(schedule)
        next_schedule["last_completion_status"] = "partial"
        next_schedule["health_status"] = "partial"
        next_schedule["current_attempt_id"] = None
        if str(next_schedule.get("schedule_status") or "") in {"awaiting_resources", "stalled"}:
            next_schedule["schedule_status"] = "completed"
        await self._store.update_agent_run_schedule(run_id, next_schedule)

    async def _transition_agent_run(
        self,
        run_id: str,
        *,
        target_status: str,
        allowed_current: set[str],
        event_type: str,
    ) -> dict[str, Any] | None:
        run = await self._store.get_agent_run(run_id)
        if run is None:
            return None
        current_status = str(run.get("status") or "created")
        if current_status in AGENT_RUN_TERMINAL_STATUS:
            return _agent_run_summary(run)
        if current_status not in allowed_current:
            return _agent_run_summary(run)
        await self._store.update_agent_run_status(
            run_id,
            target_status,
            latest_error=None if target_status in AGENT_RUN_ACTIVE_STATUS else run.get("latest_error"),
        )
        await self._store.append_agent_run_event(run_id, {"type": event_type})
        updated = await self._store.get_agent_run(run_id)
        return _agent_run_summary(updated or run)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _new_attempt_id(run_id: str) -> str:
    return f"{run_id}:{uuid4().hex[:12]}"


def _now_iso_utc() -> str:
    return _to_iso_utc(_utcnow())


def _to_iso_utc(value: datetime) -> str:
    resolved = value.astimezone(UTC) if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return resolved.isoformat()


def _parse_iso_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _has_runtime_schedule_controls(schedule: dict[str, Any] | None) -> bool:
    if not isinstance(schedule, dict) or not schedule:
        return False
    if "enabled" in schedule:
        return True
    if "cron" in schedule and any(key in schedule for key in ("start_immediately", "next_run_at")):
        return True
    return any(key in schedule for key in SCHEDULE_RUNTIME_KEYS if key != "enabled")


def _normalize_agent_run_schedule(schedule: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(schedule, dict):
        return {}
    normalized = dict(schedule)
    if not _has_runtime_schedule_controls(normalized):
        return normalized

    enabled = bool(normalized.get("enabled", True))
    normalized["enabled"] = enabled
    normalized.setdefault("last_scheduled_at", None)
    normalized.setdefault("last_completed_at", None)
    normalized.setdefault("last_completion_status", None)
    normalized["attempt_count"] = int(normalized.get("attempt_count", 0) or 0)
    normalized["max_runs"] = _parse_optional_positive_int(normalized.get("max_runs"))
    normalized["auto_pause_on_failure"] = bool(normalized.get("auto_pause_on_failure", False))
    normalized["failure_streak"] = int(normalized.get("failure_streak", 0) or 0)
    normalized["health_status"] = str(normalized.get("health_status") or "healthy")
    normalized["current_attempt_id"] = (
        str(normalized.get("current_attempt_id"))
        if isinstance(normalized.get("current_attempt_id"), str) and normalized.get("current_attempt_id")
        else None
    )
    normalized["recent_attempts"] = (
        list(normalized.get("recent_attempts"))
        if isinstance(normalized.get("recent_attempts"), list)
        else []
    )[:10]
    if not enabled:
        if str(normalized.get("schedule_status") or "") not in {
            "completed",
            "max_runs_reached",
            "paused_after_failure",
        }:
            normalized["schedule_status"] = "disabled"
        normalized.setdefault("next_run_at", None)
        return normalized

    if normalized["max_runs"] is not None and normalized["attempt_count"] >= normalized["max_runs"]:
        normalized["enabled"] = False
        normalized["next_run_at"] = None
        normalized["schedule_status"] = "max_runs_reached"
        normalized["health_status"] = "completed"
        return normalized

    next_run = _resolve_initial_next_run_at(normalized)
    if next_run is None:
        normalized["schedule_status"] = "invalid"
        normalized["schedule_error"] = "No supported runtime schedule fields were provided."
        normalized["next_run_at"] = None
        return normalized

    normalized["schedule_status"] = "scheduled"
    normalized.pop("schedule_error", None)
    normalized["next_run_at"] = _to_iso_utc(next_run)
    return normalized


def _resolve_initial_next_run_at(schedule: dict[str, Any]) -> datetime | None:
    explicit_next = _parse_iso_datetime(schedule.get("next_run_at"))
    if explicit_next is not None:
        return explicit_next
    if bool(schedule.get("start_immediately")):
        return _utcnow()

    run_at = _parse_iso_datetime(schedule.get("run_at"))
    if run_at is not None:
        return run_at

    interval_seconds = _parse_interval_seconds(schedule.get("interval_seconds"))
    if interval_seconds is not None:
        return _utcnow() + timedelta(seconds=interval_seconds)

    cron_expression = schedule.get("cron")
    if isinstance(cron_expression, str) and cron_expression.strip():
        return _next_cron_occurrence(
            cron_expression.strip(),
            timezone_name=str(schedule.get("timezone") or "UTC"),
            after=_utcnow(),
        )
    return None


def _is_agent_run_schedule_due(schedule: dict[str, Any], *, now: datetime) -> bool:
    if not _has_runtime_schedule_controls(schedule):
        return False
    if not bool(schedule.get("enabled")):
        return False
    if str(schedule.get("schedule_status") or "scheduled") not in {"scheduled", "due"}:
        return False
    next_run = _parse_iso_datetime(schedule.get("next_run_at"))
    return next_run is not None and next_run <= now


def _advance_agent_run_schedule(schedule: dict[str, Any], *, triggered_at: datetime) -> dict[str, Any]:
    next_schedule = dict(schedule)
    next_schedule["last_scheduled_at"] = _to_iso_utc(triggered_at)
    interval_seconds = _parse_interval_seconds(next_schedule.get("interval_seconds"))
    if interval_seconds is not None:
        next_schedule["next_run_at"] = _to_iso_utc(triggered_at + timedelta(seconds=interval_seconds))
        next_schedule["schedule_status"] = "scheduled"
        return next_schedule

    cron_expression = next_schedule.get("cron")
    if isinstance(cron_expression, str) and cron_expression.strip():
        next_run = _next_cron_occurrence(
            cron_expression.strip(),
            timezone_name=str(next_schedule.get("timezone") or "UTC"),
            after=triggered_at,
        )
        next_schedule["next_run_at"] = _to_iso_utc(next_run) if next_run is not None else None
        next_schedule["schedule_status"] = "scheduled" if next_run is not None else "completed"
        return next_schedule

    next_schedule["next_run_at"] = None
    next_schedule["enabled"] = False
    next_schedule["schedule_status"] = "completed"
    return next_schedule


def _parse_interval_seconds(value: Any) -> int | None:
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return None
    return resolved if resolved > 0 else None


def _parse_optional_positive_int(value: Any) -> int | None:
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return None
    return resolved if resolved > 0 else None


def _append_schedule_attempt(
    schedule: dict[str, Any],
    *,
    run: dict[str, Any],
    completion_status: str,
    completed_at: str,
    attempt_id: str | None,
    package_summary: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
    previous_attempts = schedule.get("recent_attempts")
    attempts = list(previous_attempts) if isinstance(previous_attempts, list) else []
    next_attempt = {
        "attempt_id": attempt_id,
        "attempt_number": int(schedule.get("attempt_count", 0) or 0),
        "status": completion_status,
        "started_at": run.get("started_at"),
        "completed_at": completed_at,
        "selected_candidate_id": (
            summary.get("selected_candidate_id")
            if isinstance(summary.get("selected_candidate_id"), str)
            else None
        ),
        "final_answer_preview": _truncate_preview(
            summary.get("final_answer") if isinstance(summary.get("final_answer"), str) else None
        ),
        "latest_error": run.get("latest_error") if isinstance(run.get("latest_error"), str) else None,
        "artifact_count": int((package_summary or {}).get("artifact_count") or 0),
        "dataset_record_count": int((package_summary or {}).get("dataset_record_count") or 0),
        "training_ready_count": int((package_summary or {}).get("training_ready_count") or 0),
        "package_ready": bool((package_summary or {}).get("package_ready")),
    }
    attempts.insert(0, next_attempt)
    return attempts[:10]


def _dataset_training_ready_count(
    dataset_package: dict[str, Any],
    *,
    attempt_id: str | None,
) -> int:
    attempts = dataset_package.get("attempts")
    if not isinstance(attempts, list):
        return 0
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        current_attempt_id = attempt.get("attempt_id")
        if attempt_id is None:
            if current_attempt_id is None:
                return int(attempt.get("training_ready_count") or 0)
            continue
        if current_attempt_id == attempt_id:
            return int(attempt.get("training_ready_count") or 0)
    return 0


def _truncate_preview(value: str | None, *, max_chars: int = 280) -> str | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3].rstrip()}..."


def _next_cron_occurrence(
    expression: str,
    *,
    timezone_name: str,
    after: datetime,
) -> datetime | None:
    fields = expression.split()
    if len(fields) != 5:
        return None
    minute_values = _parse_cron_field(fields[0], minimum=0, maximum=59)
    hour_values = _parse_cron_field(fields[1], minimum=0, maximum=23)
    day_values = _parse_cron_field(fields[2], minimum=1, maximum=31)
    month_values = _parse_cron_field(fields[3], minimum=1, maximum=12)
    weekday_values = _parse_cron_weekdays(fields[4])
    if any(
        values is None
        for values in (
            minute_values,
            hour_values,
            day_values,
            month_values,
            weekday_values,
        )
    ):
        return None

    tz = _resolve_schedule_timezone(timezone_name)
    current = after.astimezone(tz).replace(second=0, microsecond=0) + timedelta(minutes=1)
    for _ in range(366 * 24 * 60):
        if (
            current.minute in minute_values
            and current.hour in hour_values
            and current.day in day_values
            and current.month in month_values
            and current.weekday() in weekday_values
        ):
            return current.astimezone(UTC)
        current += timedelta(minutes=1)
    return None


def _parse_cron_weekdays(field: str) -> set[int] | None:
    values = _parse_cron_field(field, minimum=0, maximum=7)
    if values is None:
        return None
    mapped: set[int] = set()
    for value in values:
        mapped.add(6 if value in {0, 7} else value - 1)
    return mapped


def _parse_cron_field(field: str, *, minimum: int, maximum: int) -> set[int] | None:
    tokens = [token.strip() for token in field.split(",") if token.strip()]
    if not tokens:
        return None
    values: set[int] = set()
    for token in tokens:
        parsed = _expand_cron_token(token, minimum=minimum, maximum=maximum)
        if parsed is None:
            return None
        values.update(parsed)
    return values


def _expand_cron_token(token: str, *, minimum: int, maximum: int) -> set[int] | None:
    if token == "*":
        return set(range(minimum, maximum + 1))

    base = token
    step = 1
    if "/" in token:
        base, raw_step = token.split("/", 1)
        try:
            step = int(raw_step)
        except ValueError:
            return None
        if step <= 0:
            return None

    if base == "*":
        start = minimum
        end = maximum
    elif "-" in base:
        raw_start, raw_end = base.split("-", 1)
        try:
            start = int(raw_start)
            end = int(raw_end)
        except ValueError:
            return None
    else:
        try:
            value = int(base)
        except ValueError:
            return None
        if value < minimum or value > maximum:
            return None
        return {value}

    if start < minimum or end > maximum or start > end:
        return None
    return set(range(start, end + 1, step))


def _resolve_schedule_timezone(name: str) -> timezone:
    if name.upper() == "UTC":
        return UTC
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(name)
    except Exception:
        fallback_offsets = {
            "Asia/Taipei": 8,
            "Asia/Tokyo": 9,
            "America/New_York": -5,
            "Europe/London": 0,
        }
        offset_hours = fallback_offsets.get(name, 0)
        return timezone(timedelta(hours=offset_hours), name=name)


def _task_summary(task: dict[str, Any]) -> dict[str, Any]:
    pending_approval = task.get("pending_approval") if isinstance(task.get("pending_approval"), dict) else None
    return {
        "task_id": task["id"],
        "status": task["status"],
        "input_message": task["input"],
        "session_id": task.get("session_id"),
        "project_id": task.get("project_id"),
        "project_workspace_dir": task.get("project_workspace_dir") or task.get("workspace_dir"),
        "task_workspace_dir": task.get("task_workspace_dir"),
        "task_type": task.get("task_type"),
        "metadata": task.get("metadata") if isinstance(task.get("metadata"), dict) else {},
        "final_answer": task.get("final_answer"),
        "error": task.get("error"),
        "started_at": task.get("started_at"),
        "finished_at": task.get("finished_at"),
        "created_at": task["created_at"],
        "updated_at": task["updated_at"],
        "pending_approval_id": pending_approval.get("id") if pending_approval else None,
    }


def _approval_override(approval: dict[str, Any]) -> dict[str, Any]:
    return {
        "approved_tool_calls": [
            {
                "tool_name": str(approval.get("tool_name", "")),
                "arguments": dict(approval.get("arguments") or {}),
            }
        ]
    }


def _approval_summary(approval: dict[str, Any]) -> dict[str, Any]:
    status = str(approval.get("status", "pending"))
    decision: str | None = None
    if status == "approved":
        decision = "approve"
    elif status == "rejected":
        decision = "reject"
    metadata = approval.get("metadata") if isinstance(approval.get("metadata"), dict) else {}
    security_decision = SecurityDecision.from_metadata(metadata)
    linked_exec_approval_id = _linked_exec_approval_id(metadata)
    exec_session_id = metadata.get("session_id") if isinstance(metadata.get("session_id"), str) else None
    exec_status = metadata.get("status") if isinstance(metadata.get("status"), str) else None
    return {
        "approval_id": approval["id"],
        "task_id": approval["task_id"],
        "status": status,
        "tool_name": approval["tool_name"],
        "arguments": approval.get("arguments") or {},
        "created_at": approval["created_at"],
        "resolved_at": approval.get("resolved_at"),
        "decision": decision,
        "reason": approval.get("reason"),
        "requires_approval": security_decision.requires_approval if security_decision else bool(metadata.get("requires_approval", False)),
        "policy_reason": security_decision.reason if security_decision else metadata.get("reason"),
        "approval_kind": security_decision.approval_kind if security_decision else metadata.get("approval_kind", "other"),
        "approval_scope": security_decision.approval_scope if security_decision else metadata.get("approval_scope", "workspace"),
        "replay_safe": security_decision.replay_safe if security_decision else bool(metadata.get("replay_safe", False)),
        "security_decision": security_decision.action if security_decision else None,
        "policy_source": security_decision.policy_source if security_decision else metadata.get("policy_source"),
        "source": "task_runtime",
        "exec_approval_id": linked_exec_approval_id,
        "exec_session_id": exec_session_id,
        "exec_status": exec_status,
    }


def _exec_approval_summary(approval: Any) -> dict[str, Any]:
    status = str(getattr(approval, "status", "pending"))
    decision: str | None = None
    if status == "approved":
        decision = "approve"
    elif status == "rejected":
        decision = "reject"

    command = str(getattr(approval, "command", ""))
    shell = str(getattr(approval, "shell", "auto"))
    scope = str(getattr(approval, "scope", "dangerous_command"))
    reason = getattr(approval, "reason", None)
    reason_text = reason if isinstance(reason, str) else None

    return {
        "approval_id": str(getattr(approval, "approval_id", "")),
        "task_id": None,
        "status": status,
        "tool_name": "exec_command",
        "arguments": {"command": command, "shell": shell},
        "created_at": str(getattr(approval, "created_at", "")),
        "resolved_at": getattr(approval, "resolved_at", None),
        "decision": decision,
        "reason": reason_text,
        "requires_approval": True,
        "policy_reason": reason_text,
        "approval_kind": "shell",
        "approval_scope": scope,
        "replay_safe": False,
        "security_decision": "require_approval",
        "policy_source": "exec_runtime",
        "source": "exec_runtime",
        "exec_approval_id": str(getattr(approval, "approval_id", "")),
        "exec_session_id": _exec_result_field(getattr(approval, "execution_result", None), "session_id"),
        "exec_status": status,
        "execution_result": getattr(approval, "execution_result", None),
    }


def _linked_exec_approval_id(metadata: Any) -> str | None:
    if not isinstance(metadata, dict):
        return None
    approval_id = metadata.get("approval_id")
    if not isinstance(approval_id, str) or not approval_id.strip():
        return None
    normalized = approval_id.strip()
    if not normalized.startswith("exec-approval-"):
        return None
    return normalized


def _to_exec_approval_status(status: str | None) -> ApprovalStatus | None:
    if status in {"pending", "approved", "rejected"}:
        return status
    return None


def _exec_result_field(result: Any, key: str) -> Any:
    if not isinstance(result, dict):
        return None
    return result.get(key)


def _final_answer_from_exec_result(result: dict[str, Any]) -> str | None:
    stdout = result.get("stdout")
    if isinstance(stdout, str) and stdout.strip():
        return stdout.strip()
    stderr = result.get("stderr")
    if isinstance(stderr, str) and stderr.strip():
        return stderr.strip()
    session_id = result.get("session_id")
    if result.get("status") == "running" and isinstance(session_id, str) and session_id.strip():
        return f"Exec session started in background: {session_id.strip()}"
    return None


def _exec_error_message(result: dict[str, Any]) -> str:
    for key in ("error", "stderr"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    status = result.get("status")
    if isinstance(status, str) and status.strip():
        return f"Exec command failed with status: {status.strip()}"
    return "Exec command failed."


def _find_agent_run_exec_lease(run: dict[str, Any], session_id: str) -> dict[str, Any] | None:
    target = session_id.strip()
    if not target:
        return None
    artifacts = run.get("artifacts")
    if not isinstance(artifacts, list):
        return None
    fallback_lease: dict[str, Any] | None = None

    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        metadata = artifact.get("metadata") if isinstance(artifact.get("metadata"), dict) else {}
        content = metadata.get("content") if isinstance(metadata.get("content"), dict) else {}
        artifact_type = str(artifact.get("artifact_type") or "")
        if artifact_type == "detached_exec_jobs":
            items = content.get("items")
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict) and str(item.get("session_id") or "").strip() == target:
                        return dict(item)
        if artifact_type == "execution_results":
            items = content.get("items")
            if isinstance(items, list):
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    session = _result_session_id_from_execution_result(item)
                    if session != target:
                        continue
                    fallback_lease = {
                        "session_id": session,
                        "request_id": item.get("request_id"),
                        "command": item.get("command"),
                        "shell": item.get("shell"),
                        "workdir": item.get("workdir"),
                        "timeout": item.get("timeout"),
                        "log_path": item.get("log_path"),
                        "checkpoint_dir": item.get("checkpoint_dir"),
                        "status": item.get("status"),
                        "background": bool(item.get("background")),
                        "approval_state": (
                            item.get("metadata", {}).get("approval_state")
                            if isinstance(item.get("metadata"), dict)
                            else None
                        ),
                        "pid": (
                            item.get("metadata", {}).get("pid")
                            if isinstance(item.get("metadata"), dict)
                            else None
                        ),
                        "lease_owner": "runtime_service",
                        "reattach_supported": True,
                    }
    return fallback_lease


def _result_session_id_from_execution_result(result: dict[str, Any]) -> str | None:
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    session_id = metadata.get("session_id")
    if isinstance(session_id, str) and session_id.strip():
        return session_id.strip()
    return None


def _build_agent_run_exec_session_payload(
    *,
    run_id: str,
    session_id: str,
    lease: dict[str, Any],
    poll: Any,
) -> dict[str, Any]:
    if poll is None:
        return {
            "run_id": run_id,
            "session_id": session_id,
            "associated": True,
            "live_status": "unavailable",
            "lease": lease,
            "session": None,
        }
    return {
        "run_id": run_id,
        "session_id": session_id,
        "associated": True,
        "live_status": "available",
        "lease": lease,
        "session": {
            "session_id": poll.session_id,
            "shell": poll.shell,
            "status": poll.status.value,
            "background": poll.background,
            "tty": poll.tty,
            "pid": poll.pid,
            "exit_code": poll.exit_code,
            "timed_out": poll.timed_out,
            "approval_state": poll.approval_state,
            "stdout": poll.stdout,
            "stderr": poll.stderr,
        },
    }


def _build_agent_run_partial_summary(
    *,
    run: dict[str, Any],
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    effective_summary = summary if isinstance(summary, dict) else {}
    recovery_state = (
        effective_summary.get("recovery_state")
        if isinstance(effective_summary.get("recovery_state"), dict)
        else {}
    )
    checkpoint = (
        recovery_state.get("checkpoint")
        if isinstance(recovery_state.get("checkpoint"), dict)
        else {}
    )
    recoverable_artifacts = []
    artifact_types: list[str] = []
    for artifact in run.get("artifacts") or []:
        if not isinstance(artifact, dict):
            continue
        artifact_type = str(artifact.get("artifact_type") or "").strip()
        if not artifact_type:
            continue
        artifact_types.append(artifact_type)
        if artifact_type not in {
            "recovery_checkpoint",
            "partial_summary",
            "resource_exhaustion_report",
            "subagent_health_snapshot",
            "role_task_snapshot",
            "detached_exec_jobs",
            "attempt_bundle",
            "dataset_package_snapshot",
            "produced_artifacts",
            "execution_results",
        }:
            continue
        metadata = artifact.get("metadata") if isinstance(artifact.get("metadata"), dict) else {}
        recoverable_artifacts.append(
            {
                "artifact_type": artifact_type,
                "title": artifact.get("title"),
                "uri": artifact.get("uri"),
                "attempt_id": metadata.get("attempt_id"),
            }
        )
    return {
        "status": "partial",
        "reason": recovery_state.get("reason"),
        "stage": recovery_state.get("stage"),
        "checkpoint_index": checkpoint.get("checkpoint_index"),
        "candidate_count": int(effective_summary.get("candidate_count") or 0),
        "artifact_count": len(run.get("artifacts") or []),
        "artifact_types": sorted(set(artifact_types)),
        "recoverable_artifacts": recoverable_artifacts,
        "unfinished_steps": list(recovery_state.get("unfinished_steps") or []),
        "recommended_resume_conditions": list(
            recovery_state.get("recommended_resume_conditions") or []
        ),
        "degraded": bool(effective_summary.get("degraded", False)),
        "tracked_role_count": int(_build_operator_role_task_summary(run)["tracked_role_count"]),
        "detached_exec_job_count": int(_build_operator_detached_exec_summary(run)["total"]),
    }


def _agent_run_summary(run: dict[str, Any]) -> dict[str, Any]:
    summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
    reasoning_effort = _coerce_reasoning_effort(summary.get("reasoning_effort"))
    return {
        "run_id": run["id"],
        "protocol_id": run["protocol_id"],
        "title": run.get("title"),
        "topic": run.get("topic"),
        "reasoning_effort": reasoning_effort,
        "status": run.get("status", "created"),
        "selected_models_roles": run.get("selected_models_roles") or {},
        "evaluation_policy": run.get("evaluation_policy") or {},
        "run_policy": run.get("run_policy") or {},
        "schedule": run.get("schedule") or {},
        "summary": summary,
        "recovery_state": summary.get("recovery_state") if isinstance(summary.get("recovery_state"), dict) else {},
        "degraded": bool(summary.get("degraded", False)),
        "latest_error": run.get("latest_error"),
        "evidence_status": run.get("evidence_status") or {},
        "artifacts": run.get("artifacts") or [],
        "created_at": run["created_at"],
        "updated_at": run["updated_at"],
        "started_at": run.get("started_at"),
        "finished_at": run.get("finished_at"),
    }


def _coerce_reasoning_effort(value: Any) -> ReasoningEffort | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized == "none":
        return "none"
    if normalized == "minimal":
        return "minimal"
    if normalized == "low":
        return "low"
    if normalized == "medium":
        return "medium"
    if normalized == "high":
        return "high"
    if normalized == "xhigh":
        return "xhigh"
    return None


def _agent_run_reasoning_effort(run: dict[str, Any]) -> ReasoningEffort | None:
    summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
    return _coerce_reasoning_effort(summary.get("reasoning_effort"))


def _resolve_protocol_payload(run: dict[str, Any]) -> dict[str, Any]:
    protocol_id = str(run.get("protocol_id") or "").strip()
    summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
    evaluation_policy = (
        run.get("evaluation_policy") if isinstance(run.get("evaluation_policy"), dict) else {}
    )
    protocol_config = (
        summary.get("protocol_config") if isinstance(summary.get("protocol_config"), dict) else {}
    )
    research = evaluation_policy.get("research") if isinstance(evaluation_policy.get("research"), dict) else {}
    rounds = protocol_config.get("rounds")
    if not isinstance(rounds, int):
        rounds = research.get("debate_rounds")
    if protocol_id in AGENT_RUN_SUPPORTED_PROTOCOLS:
        payload: dict[str, Any] = {
            "protocol": protocol_id,
            **{
                str(key): value
                for key, value in protocol_config.items()
                if isinstance(key, str)
            },
        }
        if isinstance(rounds, int) and rounds > 0:
            payload["rounds"] = rounds
        return payload
    return {"protocol": "teacher_student_distill"}


def _controlled_execution_protocol_config(execution_budget: dict[str, Any]) -> dict[str, Any]:
    def _positive_int(key: str, default: int, maximum: int) -> int:
        try:
            value = int(execution_budget.get(key, default))
        except (TypeError, ValueError):
            value = default
        return max(1, min(value, maximum))

    return {
        "max_execution_requests": _positive_int("max_execution_requests", 5, 20),
        "max_commands_per_request": _positive_int("max_commands_per_request", 1, 5),
        "default_timeout_sec": _positive_int("default_timeout_sec", 300, 86_400),
        "background_allowed": bool(execution_budget.get("background_allowed", True)),
        "workspace_mode": "task_sandbox",
    }


def _normalize_agent_run_policy(
    run_policy: dict[str, Any] | None,
    *,
    defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {**dict(defaults or {}), **dict(run_policy or {})}

    def _optional_positive_int(key: str, *, maximum: int = 86_400) -> int | None:
        try:
            value = int(payload.get(key))
        except (TypeError, ValueError):
            return None
        return max(1, min(value, maximum))

    def _optional_nonnegative_int(key: str, *, maximum: int) -> int | None:
        try:
            value = int(payload.get(key))
        except (TypeError, ValueError):
            return None
        return max(0, min(value, maximum))

    on_budget_exhausted = str(payload.get("on_budget_exhausted") or "pause").strip().lower()
    if on_budget_exhausted not in {"pause", "finalize_partial"}:
        on_budget_exhausted = "pause"

    on_subagent_disconnect = str(payload.get("on_subagent_disconnect") or "retry_then_degrade").strip().lower()
    if on_subagent_disconnect not in {"retry_then_degrade", "pause", "fail"}:
        on_subagent_disconnect = "retry_then_degrade"

    max_subagent_failures_per_role = _optional_nonnegative_int(
        "max_subagent_failures_per_role",
        maximum=100,
    )

    normalized = {
        "max_wall_clock_sec": _optional_positive_int("max_wall_clock_sec"),
        "heartbeat_timeout_sec": _optional_positive_int("heartbeat_timeout_sec"),
        "checkpoint_interval_steps": _optional_positive_int("checkpoint_interval_steps", maximum=10_000) or 1,
        "max_subagent_failures_per_role": (
            2 if max_subagent_failures_per_role is None else max_subagent_failures_per_role
        ),
        "on_budget_exhausted": on_budget_exhausted,
        "on_subagent_disconnect": on_subagent_disconnect,
    }
    return {key: value for key, value in normalized.items() if value is not None}


def _default_agent_run_policy_from_security(security: SecurityConfig) -> dict[str, Any]:
    return {
        "max_wall_clock_sec": security.agent_run_default_max_wall_clock_sec,
        "heartbeat_timeout_sec": security.agent_run_default_heartbeat_timeout_sec,
        "checkpoint_interval_steps": security.agent_run_default_checkpoint_interval_steps,
        "max_subagent_failures_per_role": security.agent_run_default_max_subagent_failures_per_role,
        "on_budget_exhausted": security.agent_run_default_on_budget_exhausted,
        "on_subagent_disconnect": security.agent_run_default_on_subagent_disconnect,
    }


def _agent_run_task_input(run: dict[str, Any]) -> str:
    summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
    if isinstance(summary.get("task_input"), str) and summary["task_input"].strip():
        return summary["task_input"].strip()
    if isinstance(summary.get("objective"), str) and summary["objective"].strip():
        return summary["objective"].strip()
    topic = run.get("topic")
    if isinstance(topic, str) and topic.strip():
        return topic.strip()
    title = run.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    return f"Agent run {run['id']}"


def _build_agent_run_summary_payload(run: dict[str, Any], result: Any) -> dict[str, Any]:
    summary = dict(run.get("summary") or {})
    summary.update(
        {
            "protocol": result.protocol,
            "selected_candidate_id": result.selected_candidate_id,
            "final_answer": (result.artifacts or {}).get("final_answer"),
            "candidate_count": len(result.candidates),
            "degraded": bool((result.metadata or {}).get("degraded", False)),
        }
    )
    recovery_state = (result.metadata or {}).get("recovery_state")
    if isinstance(recovery_state, dict):
        summary["recovery_state"] = recovery_state
    return summary


def _recovery_state_from_run(run: dict[str, Any]) -> dict[str, Any]:
    direct = run.get("recovery_state")
    if isinstance(direct, dict):
        return dict(direct)
    summary = run.get("summary")
    if isinstance(summary, dict) and isinstance(summary.get("recovery_state"), dict):
        return dict(summary.get("recovery_state") or {})
    return {}


def _resume_payload_from_run(run: dict[str, Any]) -> dict[str, Any] | None:
    recovery_state = _recovery_state_from_run(run)
    payload = recovery_state.get("resume_payload")
    if isinstance(payload, dict):
        return dict(payload)
    return None


def _resume_runtime_from_run(run: dict[str, Any]) -> dict[str, Any]:
    recovery_state = _recovery_state_from_run(run)
    payload = recovery_state.get("resume_runtime")
    if isinstance(payload, dict):
        return dict(payload)
    return {}


def _recovery_checkpoint_index_from_run(run: dict[str, Any]) -> int | None:
    recovery_state = _recovery_state_from_run(run)
    checkpoint = recovery_state.get("checkpoint")
    if not isinstance(checkpoint, dict):
        return None
    try:
        value = int(checkpoint.get("checkpoint_index"))
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _normalize_agent_run_resume_strategy(value: str | None) -> str:
    normalized = str(value or "continue_from_checkpoint").strip().lower()
    if normalized not in {"continue_from_checkpoint", "restart_attempt"}:
        return "continue_from_checkpoint"
    return normalized


def _resolve_agent_run_resume_strategy(requested: str | None, run: dict[str, Any]) -> str:
    if isinstance(requested, str) and requested.strip():
        return _normalize_agent_run_resume_strategy(requested)
    payload = _resume_payload_from_run(run)
    if isinstance(payload, dict):
        return _normalize_agent_run_resume_strategy(
            str(
                payload.get("strategy_default")
                or payload.get("executor")
                or "continue_from_checkpoint"
            )
        )
    return "continue_from_checkpoint"


def _build_agent_run_resume_lease(
    *,
    run_id: str,
    strategy: str,
    checkpoint_index: int | None,
    source: str,
) -> dict[str, Any]:
    return {
        "lease_id": f"resume-{uuid4()}",
        "status": "active",
        "strategy": _normalize_agent_run_resume_strategy(strategy),
        "source": source,
        "acquired_at": _now_iso_utc(),
        "checkpoint_index": checkpoint_index,
        "run_id": run_id,
    }


def _candidate_snapshots_from_events(events: Any) -> list[dict[str, Any]]:
    if not isinstance(events, list):
        return []
    latest: dict[str, dict[str, Any]] = {}
    ordered: list[str] = []
    for event in events:
        if not isinstance(event, dict) or str(event.get("type") or "") != "role_output":
            continue
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        candidate_id = str(payload.get("candidate_id") or payload.get("role_id") or "").strip()
        role_id = str(payload.get("role_id") or "").strip()
        content = payload.get("content")
        if not candidate_id or not role_id or not isinstance(content, str):
            continue
        if candidate_id not in latest:
            ordered.append(candidate_id)
        latest[candidate_id] = {
            "candidate_id": candidate_id,
            "role_id": role_id,
            "content": content,
            "metadata": {
                key: value
                for key, value in {
                    "model_id": payload.get("model_id"),
                    "round_index": payload.get("round_index"),
                }.items()
                if value is not None
            },
        }
    return [latest[candidate_id] for candidate_id in ordered if candidate_id in latest]


def _protocol_artifacts_from_run_summary(run_summary: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "protocol",
        "selected_candidate_id",
        "final_answer",
        "candidate_count",
        "run_guard_policy",
        "execution_policy",
        "subagent_runtime",
        "verification",
        "evidence_collection",
    }
    return {
        key: dict(value)
        for key, value in run_summary.items()
        if isinstance(key, str) and key not in excluded and isinstance(value, dict)
    }


def _build_agent_run_resume_payload(
    *,
    run: dict[str, Any],
    recovery_state: dict[str, Any],
    strategy: str,
) -> dict[str, Any]:
    run_summary = _latest_agent_run_artifact_content(run, "run_summary") or {}
    evidence_summary = _latest_agent_run_artifact_content(run, "evidence_summary") or {}
    verification_summary = _latest_agent_run_artifact_content(run, "verification_summary") or {}
    candidate_snapshots = _candidate_snapshots_from_events(run.get("events"))
    summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
    provided_packets = (
        summary.get("evidence_packets")
        if isinstance(summary.get("evidence_packets"), list)
        else []
    )
    collected_packets = (
        evidence_summary.get("evidence_packets")
        if isinstance(evidence_summary.get("evidence_packets"), list)
        else []
    )
    evaluation_event = None
    events = run.get("events")
    if isinstance(events, list):
        for event in reversed(events):
            if not isinstance(event, dict) or str(event.get("type") or "") != "evaluation":
                continue
            payload = event.get("payload")
            if isinstance(payload, dict):
                evaluation_event = dict(payload)
                break
    selected_candidate_id = (
        run_summary.get("selected_candidate_id")
        if isinstance(run_summary.get("selected_candidate_id"), str)
        else evaluation_event.get("selected_candidate_id") if isinstance(evaluation_event, dict) else None
    )
    checkpoint = (
        dict(recovery_state.get("checkpoint"))
        if isinstance(recovery_state.get("checkpoint"), dict)
        else {}
    )
    stage = str(recovery_state.get("stage") or checkpoint.get("stage") or "runtime_service_error").strip()
    evaluation_payload = dict(evaluation_event) if isinstance(evaluation_event, dict) else None
    verification_payload = (
        list(verification_summary.get("verifications") or [])
        if isinstance(verification_summary.get("verifications"), list)
        else []
    )
    evidence_collection_payload = (
        dict(evidence_summary)
        if isinstance(evidence_summary, dict) and evidence_summary
        else None
    )
    role_task_snapshot = (
        _latest_agent_run_artifact_content(run, "role_task_snapshot")
        or (
            dict(recovery_state.get("role_task_snapshot"))
            if isinstance(recovery_state.get("role_task_snapshot"), dict)
            else {}
        )
    )
    strategy_default = _normalize_agent_run_resume_strategy(strategy)
    executor = (
        strategy_default
        if _can_continue_agent_run_from_checkpoint(
            stage=stage,
            candidates=candidate_snapshots,
            evaluation_payload=evaluation_payload,
            role_task_snapshot=role_task_snapshot,
        )
        else "restart_attempt"
    )
    return {
        "version": 1,
        "executor": executor,
        "strategy_default": executor,
        "supported_actions": (
            ["restart_attempt", "continue_from_checkpoint"]
            if executor == "continue_from_checkpoint"
            else ["restart_attempt"]
        ),
        "stage": stage,
        "checkpoint": checkpoint,
        "guidance_messages": [],
        "metadata_state": {},
        "precomputed_artifacts": {},
        "protocol_artifacts": _protocol_artifacts_from_run_summary(run_summary),
        "candidates": candidate_snapshots,
        "evidence_packets": [
            dict(item)
            for item in [*provided_packets, *collected_packets]
            if isinstance(item, dict)
        ],
        "evidence_collection_summary": evidence_collection_payload,
        "verifications": verification_payload,
        "verification_summary": dict(verification_summary) if isinstance(verification_summary, dict) else None,
        "evaluation": evaluation_payload,
        "role_task_snapshot": dict(role_task_snapshot) if isinstance(role_task_snapshot, dict) else {},
        "selected_candidate_id": selected_candidate_id,
        "degraded": bool(summary.get("degraded", False)),
    }


def _ensure_agent_run_resume_payload(
    *,
    run: dict[str, Any],
    summary: dict[str, Any],
    strategy: str,
) -> dict[str, Any]:
    recovery_state = (
        dict(summary.get("recovery_state"))
        if isinstance(summary.get("recovery_state"), dict)
        else {}
    )
    if not recovery_state:
        return summary
    existing_resume_runtime = _resume_runtime_from_run(run)
    if existing_resume_runtime and not isinstance(recovery_state.get("resume_runtime"), dict):
        recovery_state["resume_runtime"] = dict(existing_resume_runtime)
    if not _is_structured_agent_run_resume_payload(recovery_state.get("resume_payload")):
        recovery_state["resume_payload"] = _build_agent_run_resume_payload(
            run=run,
            recovery_state=recovery_state,
            strategy=strategy,
        )
    summary["recovery_state"] = recovery_state
    return summary


def _build_runtime_resource_recovery_summary(
    *,
    run: dict[str, Any],
    run_id: str,
    attempt_id: str | None,
    issue: Any,
    exception: BaseException,
) -> tuple[dict[str, Any], str]:
    summary = dict(run.get("summary") or {})
    stage = "runtime_service_error"
    checkpoint = {
        "checkpoint_index": 1,
        "stage": stage,
        "task_input": _agent_run_task_input(run),
        "protocol": str(run.get("protocol_id") or ""),
        "candidate_ids": [],
        "candidate_count": 0,
        "artifact_keys": [],
        "captured_at": _now_iso_utc(),
        "extra": {
            "attempt_id": attempt_id,
            "run_id": run_id,
            "error": str(exception),
        },
    }
    resource_report = build_resource_exhaustion_report(
        issue,
        stage=stage,
        exception=exception,
    )
    summary.update(
        {
            "protocol": str(run.get("protocol_id") or ""),
            "selected_candidate_id": summary.get("selected_candidate_id"),
            "candidate_count": int(summary.get("candidate_count") or 0),
            "degraded": False,
            "recovery_state": {
                "status": "awaiting_resources",
                "action": "pause",
                "reason": issue.reason,
                "stage": stage,
                "checkpoint": checkpoint,
                "unfinished_steps": ["resume from the latest checkpoint"],
                "resource_exhaustion_report": resource_report,
                "recommended_resume_conditions": list(
                    resource_report.get("recommended_resume_conditions") or []
                ),
            },
        }
    )
    summary = _ensure_agent_run_resume_payload(
        run=run,
        summary=summary,
        strategy="continue_from_checkpoint",
    )
    latest_error = str(issue.reason)
    return summary, latest_error


_AGENT_RUN_RESUME_STAGE_ORDER = {
    "research_context_prepared": 1,
    "controlled_execution_context_prepared": 2,
    "protocol_completed": 3,
    "before_evidence_collection": 4,
    "evidence_collected": 5,
    "verification_completed": 6,
    "evaluation_completed": 7,
    "research_artifacts_completed": 8,
}


def _is_structured_agent_run_resume_payload(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    stage = payload.get("stage")
    checkpoint = payload.get("checkpoint")
    return isinstance(stage, str) and bool(stage.strip()) and isinstance(checkpoint, dict)


def _can_continue_agent_run_from_checkpoint(
    *,
    stage: str,
    candidates: list[dict[str, Any]],
    evaluation_payload: dict[str, Any] | None,
    role_task_snapshot: Mapping[str, Any] | None,
) -> bool:
    if stage in {"research_context_prepared", "controlled_execution_context_prepared"}:
        return True
    if _agent_run_resume_stage_covers(stage, "protocol_completed") and not candidates:
        return False
    if stage in {"evaluation_completed", "research_artifacts_completed"} and not isinstance(
        evaluation_payload,
        dict,
    ):
        return False
    if _agent_run_stage_supports_role_task_resume(stage) and _agent_run_role_task_snapshot_has_completed_work(
        role_task_snapshot
    ):
        return True
    return stage in _AGENT_RUN_RESUME_STAGE_ORDER


def _agent_run_stage_supports_role_task_resume(stage: str) -> bool:
    if stage in {
        "teacher_generation",
        "student_generation",
        "research_planning",
        "research_workers",
        "research_synthesis",
        "candidate_evaluation",
    }:
        return True
    return (
        stage.startswith("debate_round_")
        or stage.startswith("dr_zero_")
        or stage.startswith("verification_chunk_")
    )


def _agent_run_role_task_snapshot_has_completed_work(snapshot: Mapping[str, Any] | None) -> bool:
    if not isinstance(snapshot, Mapping):
        return False
    roles = snapshot.get("roles") if isinstance(snapshot.get("roles"), Mapping) else snapshot
    if not isinstance(roles, Mapping):
        return False
    for value in roles.values():
        if not isinstance(value, Mapping):
            continue
        if str(value.get("status") or "").strip() == "completed":
            return True
        candidate = value.get("candidate")
        if isinstance(candidate, Mapping) and str(candidate.get("candidate_id") or "").strip():
            return True
    return False


def _agent_run_resume_stage_covers(stage: str | None, target_stage: str) -> bool:
    if not stage:
        return False
    stage_rank = _AGENT_RUN_RESUME_STAGE_ORDER.get(stage)
    target_rank = _AGENT_RUN_RESUME_STAGE_ORDER.get(target_stage)
    if stage_rank is None or target_rank is None:
        return False
    return stage_rank >= target_rank


def _resolve_agent_run_completion_status(result: Any) -> str:
    state = str(getattr(result, "state", "failed") or "failed")
    recovery_state = (
        result.metadata.get("recovery_state")
        if isinstance(getattr(result, "metadata", None), dict)
        else None
    )
    if state in {"succeeded", "partial", "awaiting_resources", "stalled", "failed", "cancelled"}:
        return state
    if isinstance(recovery_state, dict):
        action = str(recovery_state.get("action") or "").strip().lower()
        if action == "pause":
            return "awaiting_resources"
        if action == "finalize_partial":
            return "partial"
    return state


def _agent_run_latest_error(result: Any) -> str | None:
    metadata = result.metadata if isinstance(getattr(result, "metadata", None), dict) else {}
    recovery_state = metadata.get("recovery_state")
    if not isinstance(recovery_state, dict):
        return None
    if str(recovery_state.get("status") or "") not in {"awaiting_resources", "stalled"}:
        return None
    reason = recovery_state.get("reason")
    return str(reason) if isinstance(reason, str) and reason.strip() else None


def _build_evidence_status_payload(result: Any) -> dict[str, Any]:
    evaluation = result.evaluation or {}
    scores = evaluation.get("scores") if isinstance(evaluation, dict) else None
    counts = {"verified": 0, "skipped": 0, "failed": 0}
    if isinstance(scores, list):
        for score in scores:
            gate = score.get("evidence_gate") if isinstance(score, dict) else None
            status = gate.get("status") if isinstance(gate, dict) else None
            if status in counts:
                counts[status] += 1
    payload = {
        "evaluation_policy": evaluation.get("policy", {}) if isinstance(evaluation, dict) else {},
        "selected_candidate_id": result.selected_candidate_id,
        "evidence_gate_counts": counts,
    }
    artifacts = result.artifacts if isinstance(getattr(result, "artifacts", None), dict) else {}
    evidence_collection = artifacts.get("evidence_collection")
    if isinstance(evidence_collection, dict):
        payload["evidence_packet_count"] = int(evidence_collection.get("total_packet_count", 0))
        payload["collected_packet_count"] = int(evidence_collection.get("collected_packet_count", 0))
    return payload


def _latest_agent_run_artifact_content(run: dict[str, Any], artifact_type: str) -> dict[str, Any] | None:
    artifacts = run.get("artifacts")
    if not isinstance(artifacts, list):
        return None
    for artifact in reversed(artifacts):
        if not isinstance(artifact, dict):
            continue
        if str(artifact.get("artifact_type") or "") != artifact_type:
            continue
        metadata = artifact.get("metadata") if isinstance(artifact.get("metadata"), dict) else {}
        content = metadata.get("content")
        if isinstance(content, dict):
            return dict(content)
    return None


def _build_operator_subagent_health_summary(run: dict[str, Any]) -> dict[str, Any]:
    snapshot = _latest_agent_run_artifact_content(run, "subagent_health_snapshot") or {}
    events = list(snapshot.get("events") or []) if isinstance(snapshot.get("events"), list) else []
    latest_event = events[-1] if events and isinstance(events[-1], dict) else None
    failure_counts = (
        dict(snapshot.get("failure_counts"))
        if isinstance(snapshot.get("failure_counts"), dict)
        else {}
    )
    return {
        "available": bool(snapshot),
        "status": str(snapshot.get("status") or "unknown") if snapshot else "missing",
        "degraded": bool(snapshot.get("degraded", False)) if snapshot else False,
        "degraded_role_ids": list(snapshot.get("degraded_role_ids") or []) if snapshot else [],
        "failure_counts": failure_counts,
        "failure_role_count": len(failure_counts),
        "event_count": len(events),
        "latest_event": latest_event,
    }


def _build_operator_role_task_summary(run: dict[str, Any]) -> dict[str, Any]:
    snapshot = _latest_agent_run_artifact_content(run, "role_task_snapshot") or {}
    roles = snapshot.get("roles") if isinstance(snapshot.get("roles"), dict) else {}
    assignments = (
        snapshot.get("resume_plan", {}).get("assignments")
        if isinstance(snapshot.get("resume_plan"), dict)
        and isinstance(snapshot.get("resume_plan", {}).get("assignments"), dict)
        else {}
    )
    status_counts: dict[str, int] = {}
    summarized_roles = []
    for role_id, value in roles.items():
        if not isinstance(role_id, str) or not isinstance(value, dict):
            continue
        status = str(value.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        assignment = assignments.get(role_id) if isinstance(assignments, dict) else {}
        summarized_roles.append(
            {
                "role_id": role_id,
                "status": status,
                "stage": value.get("stage"),
                "checkpoint_index": value.get("checkpoint_index"),
                "assigned_model_id": (
                    assignment.get("assigned_model_id")
                    if isinstance(assignment, dict)
                    else value.get("assigned_model_id")
                ),
                "assignment_source": (
                    assignment.get("assignment_source")
                    if isinstance(assignment, dict)
                    else value.get("assignment_source")
                ),
                "resume_action": (
                    assignment.get("resume_action")
                    if isinstance(assignment, dict)
                    else value.get("resume_action")
                ),
            }
        )
    return {
        "available": bool(snapshot),
        "tracked_role_count": len(summarized_roles),
        "status_counts": status_counts,
        "reassigned_role_count": len(
            list(snapshot.get("resume_plan", {}).get("reassigned_roles") or [])
            if isinstance(snapshot.get("resume_plan"), dict)
            else []
        ),
        "blocked_role_count": len(
            list(snapshot.get("resume_plan", {}).get("blocked_roles") or [])
            if isinstance(snapshot.get("resume_plan"), dict)
            else []
        ),
        "roles": summarized_roles,
    }


def _build_operator_detached_exec_summary(run: dict[str, Any]) -> dict[str, Any]:
    detached = _latest_agent_run_artifact_content(run, "detached_exec_jobs") or {}
    raw_items = detached.get("items") if isinstance(detached.get("items"), list) else []
    items = [item for item in raw_items if isinstance(item, dict)]
    status_counts: dict[str, int] = {}
    summarized_items = []
    for item in items:
        status = str(item.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        summarized_items.append(
            {
                "session_id": item.get("session_id"),
                "request_id": item.get("request_id"),
                "status": status,
                "background": bool(item.get("background", False)),
                "reattach_supported": bool(item.get("reattach_supported", False)),
                "log_path": item.get("log_path"),
                "checkpoint_dir": item.get("checkpoint_dir"),
            }
        )
    return {
        "available": bool(detached),
        "total": len(items),
        "status_counts": status_counts,
        "reattachable_count": sum(1 for item in items if bool(item.get("reattach_supported", False))),
        "items": summarized_items,
    }
