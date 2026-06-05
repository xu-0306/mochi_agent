"""Background runtime service for tasks and approvals."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from mochi.agents.multi_agent.orchestrator import MultiAgentOrchestrator, MultiAgentRunRequest
from mochi.api.routes.chat import _serialize_event
from mochi.config.schema import SecurityConfig
from mochi.learning.dataset_exporter import export_run_to_dataset_records
from mochi.runtime.agent_run_packages import (
    attempt_id_exists,
    build_attempt_bundle,
    build_dataset_package,
    summarize_package_materialization,
)
from mochi.runtime.approvals import ApprovalStatus, InMemoryApprovalStore
from mochi.runtime.exec_runtime import ExecRuntime
from mochi.runtime.models import AgentRunCreateRequest, AgentRunGuidanceRequest, TaskCreateRequest
from mochi.runtime.store import RuntimeStore
from mochi.security import SecurityDecision
from mochi.security.policy import build_runtime_permission_policy_dict
from mochi.tools.exec_command import get_shared_exec_approval_store, get_shared_exec_runtime

TASK_STATUS_RUNNING = {"queued", "running", "resumed"}
AGENT_RUN_TERMINAL_STATUS = {"cancelled", "failed", "succeeded"}
AGENT_RUN_ACTIVE_STATUS = {"running"}
AGENT_RUN_SUPPORTED_PROTOCOLS = {"teacher_student_distill", "multi_agent_debate"}
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

    def set_runtime_tasks_root(self, root_dir: Path) -> None:
        self._runtime_tasks_root = Path(root_dir)

    def update_security_config(self, security: SecurityConfig) -> None:
        self._security_config = security

    def set_scheduler_poll_interval(self, seconds: float) -> None:
        self._scheduler_poll_interval_seconds = max(0.05, float(seconds))

    async def start(self) -> None:
        active = self._scheduler_task
        if active is not None and not active.done():
            return
        self._scheduler_stop_event = asyncio.Event()
        self._scheduler_task = asyncio.create_task(
            self._scheduler_loop(),
            name="runtime-agent-run-scheduler",
        )

    async def close(self) -> None:
        self._scheduler_stop_event.set()
        scheduler_task = self._scheduler_task
        if scheduler_task is not None and not scheduler_task.done():
            scheduler_task.cancel()
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
            try:
                await job
            except asyncio.CancelledError:
                pass
        for job in list(self._active_agent_run_jobs.values()):
            try:
                await job
            except asyncio.CancelledError:
                pass

        self._active_jobs.clear()
        self._active_agent_run_jobs.clear()

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
            inference_overrides=payload.inference_overrides,
        )
        self._active_jobs[task_id] = asyncio.create_task(
            self._run_task(task_id=task_id),
            name=f"runtime-task-{task_id}",
        )
        summary["pending_approval"] = None
        return _task_summary(summary)

    async def create_agent_run(self, payload: AgentRunCreateRequest) -> dict[str, Any]:
        run_id = str(uuid4())
        schedule = _normalize_agent_run_schedule(payload.schedule)
        run = await self._store.create_agent_run(
            run_id=run_id,
            protocol_id=payload.protocol_id,
            title=payload.title,
            topic=payload.topic,
            selected_models_roles=payload.selected_models_roles,
            evaluation_policy=payload.evaluation_policy,
            schedule=schedule,
            summary=payload.summary,
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

    async def start_agent_run(self, run_id: str) -> dict[str, Any] | None:
        run = await self._store.get_agent_run(run_id)
        if run is None:
            return None
        current_status = str(run.get("status") or "created")
        if current_status in AGENT_RUN_TERMINAL_STATUS:
            return _agent_run_summary(run)
        if current_status not in {"created", "paused", "running"}:
            return _agent_run_summary(run)

        if current_status in {"created", "paused"}:
            if current_status == "created":
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

    async def resume_agent_run(self, run_id: str) -> dict[str, Any] | None:
        run = await self._store.get_agent_run(run_id)
        if run is None:
            return None
        current_status = str(run.get("status") or "created")
        if current_status in AGENT_RUN_TERMINAL_STATUS:
            return _agent_run_summary(run)
        if current_status != "paused":
            return _agent_run_summary(run)

        await self._store.update_agent_run_status(run_id, "running", latest_error=None)
        await self._store.append_agent_run_event(run_id, {"type": "run_resumed"})
        await self._ensure_agent_run_job(run_id, job_name=f"runtime-agent-run-resume-{run_id}")
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
            if current_status in {"running", "paused", "cancelled"}:
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
        run = await self._store.get_agent_run(run_id)
        if run is None:
            return
        schedule = run.get("schedule") if isinstance(run.get("schedule"), dict) else {}
        attempt_id = (
            str(schedule.get("current_attempt_id"))
            if isinstance(schedule.get("current_attempt_id"), str) and schedule.get("current_attempt_id")
            else None
        )

        try:
            orchestrator = MultiAgentOrchestrator(engine=self._engine)
            protocol_payload = _resolve_protocol_payload(run)
            request = MultiAgentRunRequest(
                run_id=run_id,
                task_input=_agent_run_task_input(run),
                protocol=protocol_payload,
                guidance_messages=await self._collect_guidance_messages(run_id),
                metadata={
                    "title": run.get("title"),
                    "topic": run.get("topic"),
                    "selected_models_roles": run.get("selected_models_roles") or {},
                    "evaluation_policy": run.get("evaluation_policy") or {},
                    "schedule": run.get("schedule") or {},
                    "summary": run.get("summary") or {},
                },
            )
            result = await orchestrator.run(request)
            for event in result.events:
                event_payload = event.to_dict()
                if attempt_id is not None and not event_payload.get("attempt_id"):
                    event_payload["attempt_id"] = attempt_id
                await self._store.append_agent_run_event(run_id, event_payload)

            summary = _build_agent_run_summary_payload(run, result)
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
            await self._update_agent_run_schedule_completion(
                run_id,
                completion_status="succeeded" if result.state == "succeeded" else str(result.state),
                attempt_id=attempt_id,
                package_summary=package_summary,
            )
            await self._store.update_agent_run_status(
                run_id,
                "succeeded" if result.state == "succeeded" else result.state,
                latest_error=None,
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
            ("research_plan", "Research Plan"),
            ("source_quality_table", "Source Quality Table"),
            ("claim_evidence_map", "Claim Evidence Map"),
            ("research_brief", "Research Brief"),
            ("subagent_runtime", "Subagent Runtime Trace"),
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


def _agent_run_summary(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run["id"],
        "protocol_id": run["protocol_id"],
        "title": run.get("title"),
        "topic": run.get("topic"),
        "status": run.get("status", "created"),
        "selected_models_roles": run.get("selected_models_roles") or {},
        "evaluation_policy": run.get("evaluation_policy") or {},
        "schedule": run.get("schedule") or {},
        "summary": run.get("summary") or {},
        "latest_error": run.get("latest_error"),
        "evidence_status": run.get("evidence_status") or {},
        "artifacts": run.get("artifacts") or [],
        "created_at": run["created_at"],
        "updated_at": run["updated_at"],
        "started_at": run.get("started_at"),
        "finished_at": run.get("finished_at"),
    }


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
        payload: dict[str, Any] = {"protocol": protocol_id}
        if isinstance(rounds, int) and rounds > 0:
            payload["rounds"] = rounds
        return payload
    return {"protocol": "teacher_student_distill"}


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
        }
    )
    return summary


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
