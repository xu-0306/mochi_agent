"""Background runtime service for tasks and approvals."""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import sqlite3
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse
from uuid import uuid4

from mochi.agents.multi_agent.execution_policy import (
    execution_policy_to_dict,
    parse_subagent_execution_policy,
)
from mochi.agents.multi_agent.orchestrator import MultiAgentOrchestrator, MultiAgentRunRequest
from mochi.api.routes.chat import _serialize_event
from mochi.backends.inference_capabilities import ReasoningEffort
from mochi.config.manager import save_config
from mochi.config.schema import CommandRuleConfig, MochiConfig, SecurityConfig
from mochi.learning.dataset_exporter import (
    export_run_to_dataset_records,
    normalize_collector_dataset_record_for_run,
)
from mochi.runtime.agent_run_packages import (
    attempt_id_exists,
    build_attempt_bundle,
    build_dataset_package,
    summarize_package_materialization,
)
from mochi.runtime.approvals import ApprovalDecision, ApprovalStatus, ApprovalStore
from mochi.runtime.collector_contracts import (
    COLLECTOR_SHARD_MANIFEST_ARTIFACT_TYPE,
    FAILED_COLLECTOR_SHARD_STATUSES,
    build_collector_state_manifest,
    collector_dataset_record_identity,
    collector_dataset_records_from_result,
    collector_record_provenance_for_index,
    collector_record_provenance_list_from_dataset_records,
    collector_record_provenance_list_from_result,
    dedupe_collector_shard_manifests,
    dedupe_collector_dataset_records,
    extract_collector_shard_manifests,
    extract_persisted_collector_dataset_records,
    collector_shard_manifests_from_result,
)
from mochi.runtime.delegate import set_delegate_subagent_task_launcher
from mochi.runtime.exec_runtime import ExecRuntime
from mochi.runtime.models import (
    AgentRunCreateRequest,
    AgentRunGuidanceRequest,
    AgentRunMessageRequest,
    AgentRunSubagentMessageRequest,
    GoalCreateRequest,
    TaskCreateRequest,
    TaskMessageRequest,
)
from mochi.runtime.recovery import (
    build_resource_exhaustion_report,
    classify_agent_run_recovery_issue,
)
from mochi.runtime.store import RuntimeStore
from mochi.security import SecurityDecision
from mochi.security.policy import build_runtime_permission_policy_dict, resolve_runtime_permission_policy
from mochi.tools.base import ToolExecutionContext, ToolResult
from mochi.tools.exec_command import get_shared_exec_approval_store, get_shared_exec_runtime
from mochi.tools.file_ops import ApplyPatchTool, FileEditTool, FileWriteTool
from mochi.tools.file_mutations import summarize_file_change_payload

TASK_STATUS_RUNNING = {"queued", "running", "resumed"}
GOAL_ACTIVE_STATUS = {"queued", "running"}
GOAL_TERMINAL_STATUS = {"completed", "failed", "cancelled"}
GOAL_RESUMABLE_STATUS = {
    "paused",
    "awaiting_operator",
    "awaiting_resources",
    "stalled",
    "waiting_approval",
    "failed",
}
GOAL_EXECUTION_MODES = {"single_agent", "workflow"}
GOAL_INTERACTION_MODES = {"goal", "workflow"}
GOAL_EXECUTION_TOPOLOGIES = {"single_agent", "multi_agent"}
DEFAULT_GOAL_EXECUTION_MODE = "workflow"
DEFAULT_GOAL_INTERACTION_MODE = "workflow"
AUTONOMOUS_SINGLE_AGENT_PROTOCOL = "autonomous_single_agent"
AGENT_RUN_TERMINAL_STATUS = {"cancelled", "failed", "succeeded"}
AGENT_RUN_ACTIVE_STATUS = {"running"}
AGENT_RUN_OPERATOR_FINALIZEABLE_STATUS = {"awaiting_resources", "stalled"}
AGENT_RUN_RESUMABLE_STATUS = {"paused", "awaiting_approval", "awaiting_resources", "partial", "stalled"}
DELEGATED_MULTI_AGENT_TASK_TYPE = "delegated_multi_agent"
AGENT_RUN_SUPPORTED_PROTOCOLS = {
    AUTONOMOUS_SINGLE_AGENT_PROTOCOL,
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
_FILE_MUTATION_TOOLS = {"file_write", "file_edit", "apply_patch"}
_GOAL_ESTOP_NETWORK_TOOL_NAMES = (
    "web_search",
    "web_fetch",
    "web_crawl",
    "arxiv_search",
    "semantic_scholar_search",
    "crossref_search",
    "pubmed_search",
)
_FILE_MUTATION_APPROVAL_KINDS = {"file_write", "file_edit", "apply_patch"}
_GOAL_CHECKPOINT_PROMOTABLE_ARTIFACT_TYPES = {
    "claim_evidence_map",
    "collector_shard_live",
    COLLECTOR_SHARD_MANIFEST_ARTIFACT_TYPE,
    "controller_decisions",
    "debate_context_snapshot",
    "debate_state",
    "evidence_bundle",
    "evidence_summary",
    "evaluation_summary",
    "execution_results",
    "produced_artifacts",
    "research_brief",
    "research_plan",
    "role_task_snapshot",
    "source_quality_table",
    "verification_chunk_summary",
    "verification_summary",
}
_DELEGATED_SUBAGENT_DISPLAY_NAMES = (
    "Ada",
    "Bohr",
    "Curie",
    "Darwin",
    "Euler",
    "Faraday",
    "Franklin",
    "Gauss",
    "Hopper",
    "Kepler",
    "Lovelace",
    "Noether",
    "Pascal",
    "Turing",
)


class RuntimeService:
    """Coordinate background task execution and approval workflow."""

    _DEFAULT_SCHEDULER_POLL_INTERVAL_SECONDS = 30.0
    _DEFAULT_GOAL_LEASE_TTL_SECONDS = 120.0

    def __init__(
        self,
        *,
        engine: Any,
        store: RuntimeStore,
        exec_approval_store: ApprovalStore | None = None,
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
        self._bound_config: MochiConfig | None = None
        self._bound_config_path: str | Path | None = None
        self._scheduler_poll_interval_seconds = self._DEFAULT_SCHEDULER_POLL_INTERVAL_SECONDS
        self._goal_lease_ttl_seconds = self._DEFAULT_GOAL_LEASE_TTL_SECONDS
        self._scheduler_stop_event = asyncio.Event()
        self._scheduler_task: asyncio.Task[None] | None = None
        self._goal_supervision_lock = asyncio.Lock()
        self._runtime_owner_id = f"runtime-{uuid4()}"
        set_delegate_subagent_task_launcher(self.create_delegated_subagent_task)

    def set_runtime_tasks_root(self, root_dir: Path) -> None:
        self._runtime_tasks_root = Path(root_dir)

    def update_security_config(self, security: SecurityConfig) -> None:
        self._security_config = security

    def bind_app_config(
        self,
        *,
        config: MochiConfig,
        config_path: str | Path | None,
    ) -> None:
        self._bound_config = config
        self._bound_config_path = config_path

    async def _persist_command_rule(self, rule: dict[str, Any] | None) -> dict[str, Any] | None:
        validated_rule = _validated_command_rule(rule)
        if validated_rule is None:
            return None
        if self._bound_config is None:
            raise RuntimeError("Runtime service is not bound to an app config.")
        existing_rules = list(self._bound_config.security.command_rules)
        if validated_rule not in existing_rules:
            existing_rules.append(validated_rule)
        updated_security = self._bound_config.security.model_copy(
            update={"command_rules": existing_rules}
        )
        self._bound_config = self._bound_config.model_copy(update={"security": updated_security})
        save_config(self._bound_config, self._bound_config_path)
        self.update_security_config(self._bound_config.security)
        return validated_rule.model_dump(mode="python")

    def set_scheduler_poll_interval(self, seconds: float) -> None:
        self._scheduler_poll_interval_seconds = max(0.05, float(seconds))

    def set_goal_lease_ttl_seconds(self, seconds: float) -> None:
        self._goal_lease_ttl_seconds = max(1.0, float(seconds))

    async def start(self) -> None:
        recover_detached = getattr(self._exec_runtime, "recover_detached_sessions", None)
        if callable(recover_detached):
            await recover_detached()
        active = self._scheduler_task
        if active is not None and not active.done():
            return
        await self._recover_goals_on_startup()
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
        await self._store.delete_goal_leases_for_owner(self._runtime_owner_id)
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
        task_metadata = _normalize_task_metadata(
            payload.metadata,
            task_type=payload.task_type,
            input_message=payload.input,
            session_id=payload.session_id,
            status="queued",
        )
        summary = await self._store.create_task_run(
            task_id=task_id,
            input_text=payload.input,
            session_id=payload.session_id,
            project_id=payload.project_id,
            workspace_dir=project_workspace_dir,
            project_workspace_dir=project_workspace_dir,
            task_workspace_dir=str(task_workspace_dir),
            task_type=payload.task_type,
            metadata=task_metadata,
            inference_overrides=payload.inference_overrides,
        )
        delegated_subagent = _delegated_subagent_payload(
            task_metadata,
            task_type=payload.task_type,
            input_message=payload.input,
            session_id=payload.session_id,
            status="queued",
        )
        if delegated_subagent is not None:
            await self._store.append_task_event(
                task_id,
                _delegated_subagent_created_event(
                    delegated_subagent,
                    task_id=task_id,
                    task_type=payload.task_type,
                ),
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
                "delegated_subagent": _build_delegated_subagent_metadata(
                    objective=objective,
                    session_id=session_id,
                    suggested_roles=suggested_roles,
                ),
            },
        )
        return await self.create_task(payload)

    async def create_goal(self, payload: GoalCreateRequest) -> dict[str, Any]:
        goal_id = str(uuid4())
        run_policy = _normalize_goal_run_policy(
            payload.run_policy,
            objective=payload.objective,
        )
        capability_policy = _normalize_goal_capability_policy(payload.capability_policy)
        summary = dict(payload.summary)
        interaction_mode = _normalize_goal_interaction_mode(
            payload.interaction_mode,
            execution_mode=payload.execution_mode,
        )
        execution_topology = _normalize_goal_execution_topology(
            payload.execution_topology,
            execution_mode=payload.execution_mode,
            protocol_id=payload.protocol_id,
        )
        summary.setdefault("interaction_mode", interaction_mode)
        summary.setdefault("execution_topology", execution_topology)
        if isinstance(payload.protocol_selection, str) and payload.protocol_selection.strip():
            summary["protocol_selection"] = payload.protocol_selection.strip()
        if isinstance(payload.selection_rationale, str) and payload.selection_rationale.strip():
            summary["selection_rationale"] = payload.selection_rationale.strip()
        if payload.project_id is not None:
            summary.setdefault("project_id", payload.project_id)
        if payload.workspace_dir is not None:
            summary.setdefault("workspace_dir", payload.workspace_dir)
        goal = await self._store.create_goal(
            goal_id=goal_id,
            objective=payload.objective,
            title=payload.title,
            goal_type=payload.goal_type,
            execution_mode=payload.execution_mode,
            protocol_id=payload.protocol_id,
            topic=payload.topic,
            project_id=payload.project_id,
            workspace_dir=payload.workspace_dir,
            run_policy=run_policy,
            capability_policy=capability_policy,
            source_manifest=payload.source_manifest,
            summary=summary,
            metadata=payload.metadata,
        )
        return _goal_response(goal)

    async def list_goals(self) -> list[dict[str, Any]]:
        goals = await self._store.list_goals()
        return [_goal_response(goal) for goal in goals]

    async def get_goal(self, goal_id: str) -> dict[str, Any] | None:
        goal = await self._store.get_goal(goal_id)
        if goal is None:
            return None
        return _goal_response(goal)

    async def get_goal_health(self, goal_id: str) -> dict[str, Any] | None:
        goal = await self._store.get_goal(goal_id)
        if goal is None:
            return None
        normalized_run_policy = _normalize_goal_run_policy(
            goal.get("run_policy"),
            objective=goal.get("objective"),
        )
        normalized_capability_policy = _normalize_goal_capability_policy(
            goal.get("capability_policy"),
        )
        operator_controls = await self.get_goal_operator_controls()
        current_attempt = _current_goal_attempt(goal)
        current_status = str(goal.get("status") or "created")
        effective_attempt = dict(current_attempt) if isinstance(current_attempt, dict) else None
        runtime_budget = _goal_runtime_budget_health_payload(
            goal,
            normalized_run_policy,
        )
        lease = await self._store.get_goal_lease(goal_id)
        findings = await self._store.list_goal_audit_findings(goal_id, status="open")
        attempt_id = (
            str(goal.get("current_attempt_id") or "").strip()
            if isinstance(goal.get("current_attempt_id"), str)
            else ""
        )
        latest_checkpoint = await self._store.get_latest_goal_checkpoint(
            goal_id,
            attempt_id=attempt_id or None,
        )
        latest_memory_snapshot = await self._store.get_latest_goal_memory_snapshot(
            goal_id,
            attempt_id=attempt_id or None,
        )
        latest_generation = await self._store.get_latest_goal_worker_generation(
            goal_id,
            attempt_id=attempt_id or None,
        )
        linked_run_id = (
            str(current_attempt.get("agent_run_id") or "").strip()
            if isinstance(current_attempt, dict)
            else ""
        )
        linked_run = await self._store.get_agent_run(linked_run_id) if linked_run_id else None
        linked_approval_state = _goal_approval_state_from_agent_run(linked_run) if linked_run else {}
        linked_run_status = (
            _effective_agent_run_status_for_goal_projection(linked_run)
            if isinstance(linked_run, dict)
            else None
        )
        if (
            isinstance(effective_attempt, dict)
            and isinstance(linked_run, dict)
            and current_status not in GOAL_TERMINAL_STATUS
        ):
            projected_status = _goal_status_from_agent_run(
                linked_run,
                linked_approval_state=linked_approval_state,
            )
            if current_status == "waiting_approval" and projected_status != "waiting_approval":
                projected_status = current_status
            current_status = projected_status
            effective_summary = (
                dict(effective_attempt.get("summary"))
                if isinstance(effective_attempt.get("summary"), dict)
                else {}
            )
            effective_summary["agent_run_status"] = linked_run_status
            if linked_approval_state:
                effective_summary["linked_approval_state"] = linked_approval_state
            else:
                effective_summary.pop("linked_approval_state", None)
            effective_attempt["status"] = current_status
            effective_attempt["summary"] = effective_summary
        checkpoint_policy = _goal_checkpoint_policy_health_payload(
            normalized_run_policy,
            effective_attempt,
            goal_status=current_status,
        )
        context_handoff = _goal_context_handoff_health_payload(
            linked_run,
            run_policy=normalized_run_policy,
            generation_started_at=(
                str(latest_generation.get("started_at") or "").strip()
                if isinstance(latest_generation, dict)
                else None
            ),
        )
        subagent_runtime = _goal_subagent_runtime_health_payload(
            linked_run,
            run_policy=normalized_run_policy,
            generation_started_at=(
                str(latest_generation.get("started_at") or "").strip()
                if isinstance(latest_generation, dict)
                else None
            ),
        )
        current_generation_payload = _goal_worker_generation_response(
            latest_generation,
            run_policy=normalized_run_policy,
            context_handoff=context_handoff,
            runtime_telemetry=subagent_runtime,
        )
        collector_state = _goal_collector_state_payload(
            linked_run,
            attempt_id=attempt_id or None,
            run_policy=normalized_run_policy,
        )
        linked_agent_run_payload = _goal_linked_agent_run_health_payload(
            effective_attempt,
            checkpoint_policy=checkpoint_policy,
            collector_state=collector_state,
        )
        open_finding_payloads = [
            _goal_audit_finding_response(finding)
            for finding in findings
        ]
        return {
            "goal_id": goal_id,
            "execution_mode": _normalize_goal_execution_mode(goal.get("execution_mode")),
            "interaction_mode": _goal_effective_interaction_mode(goal),
            "execution_topology": _goal_effective_execution_topology(goal),
            "protocol_id": _goal_effective_protocol_id(goal),
            "bound_run_id": _goal_bound_run_id(goal),
            "protocol_selection": _goal_protocol_selection(goal),
            "selection_rationale": _goal_selection_rationale(goal),
            "status": current_status,
            "current_attempt_id": goal.get("current_attempt_id"),
            "attempt_count": len(goal.get("attempts") or []),
            "run_policy": normalized_run_policy,
            "capability_policy": normalized_capability_policy,
            "operator_controls": operator_controls,
            "latest_error": goal.get("latest_error"),
            "current_attempt": (
                _goal_attempt_response(effective_attempt) if isinstance(effective_attempt, dict) else None
            ),
            "runtime_budget": runtime_budget,
            "recovery_state": _goal_attempt_recovery_state_payload(effective_attempt),
            "checkpoint": _goal_attempt_checkpoint_payload(effective_attempt),
            "persisted_checkpoint": _goal_checkpoint_record_response(latest_checkpoint),
            "memory_snapshot": _goal_memory_snapshot_response(latest_memory_snapshot),
            "current_generation": current_generation_payload,
            "approval_state": _goal_attempt_approval_state_payload(effective_attempt),
            "checkpoint_policy": checkpoint_policy,
            "collector_state": collector_state,
            "linked_agent_run": linked_agent_run_payload,
            "runtime_owner_id": self._runtime_owner_id,
            "lease": _goal_lease_health_payload(
                lease,
                runtime_owner_id=self._runtime_owner_id,
            ),
            "open_findings": open_finding_payloads,
            "recommended_next_action": _goal_recommended_next_action(
                goal_status=current_status,
                runtime_budget=runtime_budget,
                approval_state=_goal_attempt_approval_state_payload(effective_attempt),
                current_generation=current_generation_payload,
                checkpoint_policy=checkpoint_policy,
                collector_state=collector_state,
                linked_agent_run=linked_agent_run_payload,
                open_findings=open_finding_payloads,
            ),
        }

    async def list_goal_audit_findings(
        self,
        goal_id: str,
        *,
        status: str | None = None,
    ) -> list[dict[str, Any]] | None:
        goal = await self._store.get_goal(goal_id)
        if goal is None:
            return None
        findings = await self._store.list_goal_audit_findings(goal_id, status=status)
        return [_goal_audit_finding_response(finding) for finding in findings]

    async def list_goal_checkpoints(
        self,
        goal_id: str,
        *,
        attempt_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]] | None:
        goal = await self._store.get_goal(goal_id)
        if goal is None:
            return None
        checkpoints = await self._store.list_goal_checkpoints(
            goal_id,
            attempt_id=attempt_id,
            limit=limit,
        )
        return [_goal_checkpoint_record_response(item) for item in checkpoints]

    async def list_goal_memory_snapshots(
        self,
        goal_id: str,
        *,
        attempt_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]] | None:
        goal = await self._store.get_goal(goal_id)
        if goal is None:
            return None
        snapshots = await self._store.list_goal_memory_snapshots(
            goal_id,
            attempt_id=attempt_id,
            limit=limit,
        )
        return [_goal_memory_snapshot_response(item) for item in snapshots]

    async def resolve_goal_audit_finding(
        self,
        goal_id: str,
        finding_id: int,
    ) -> dict[str, Any] | None:
        goal = await self._store.get_goal(goal_id)
        if goal is None:
            return None
        finding = await self._store.get_goal_audit_finding(finding_id)
        if finding is None or str(finding.get("goal_id") or "") != goal_id:
            return None
        updated = await self._store.resolve_goal_audit_finding(finding_id)
        if updated is None:
            return None
        return _goal_audit_finding_response(updated)

    async def close_goal_audit_finding(
        self,
        goal_id: str,
        finding_id: int,
    ) -> dict[str, Any] | None:
        goal = await self._store.get_goal(goal_id)
        if goal is None:
            return None
        finding = await self._store.get_goal_audit_finding(finding_id)
        if finding is None or str(finding.get("goal_id") or "") != goal_id:
            return None
        updated = await self._store.close_goal_audit_finding(finding_id)
        if updated is None:
            return None
        return _goal_audit_finding_response(updated)

    async def get_goal_operator_controls(self) -> dict[str, Any]:
        controls = await self._store.get_goal_operator_controls(scope="global")
        return _normalize_goal_operator_controls(controls)

    async def update_goal_operator_controls(
        self,
        *,
        stop_all_goals: bool | None = None,
        blocked_tools: list[str] | None = None,
        blocked_domains: list[str] | None = None,
        block_network_usage: bool | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        current_controls = await self.get_goal_operator_controls()
        metadata = dict(current_controls.get("metadata") or {})
        if isinstance(reason, str) and reason.strip():
            metadata["reason"] = reason.strip()
        updated = await self._store.upsert_goal_operator_controls(
            scope="global",
            stop_all_goals=(
                current_controls["stop_all_goals"]
                if stop_all_goals is None
                else bool(stop_all_goals)
            ),
            blocked_tools=(
                current_controls["blocked_tools"]
                if blocked_tools is None
                else _normalized_string_list(blocked_tools)
            ),
            blocked_domains=(
                current_controls["blocked_domains"]
                if blocked_domains is None
                else _normalized_domain_list(blocked_domains)
            ),
            block_network_usage=(
                current_controls["block_network_usage"]
                if block_network_usage is None
                else bool(block_network_usage)
            ),
            metadata=metadata,
        )
        normalized_updated = _normalize_goal_operator_controls(updated)
        pause_message = _goal_operator_controls_stop_message(normalized_updated)
        if normalized_updated["stop_all_goals"] or (
            normalized_updated["tool_denylist"] != current_controls["tool_denylist"]
            or normalized_updated["blocked_domains"] != current_controls["blocked_domains"]
        ):
            await self._pause_running_goals_for_operator_controls(pause_message)
        await self._store.append_goal_operator_audit_log(
            event_type="estop_update",
            subject_type="goal_operator_controls",
            subject_id="global",
            action="update",
            summary="Updated goal emergency-stop controls.",
            details={
                "reason": reason.strip() if isinstance(reason, str) and reason.strip() else None,
                "controls": normalized_updated,
            },
        )
        return normalized_updated

    async def list_goal_operator_audit_log(
        self,
        *,
        event_type: str | None = None,
        goal_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        logs = await self._store.list_goal_operator_audit_log(
            event_type=event_type,
            limit=(None if goal_id else limit),
        )
        if goal_id:
            normalized_goal_id = str(goal_id).strip()
            logs = [
                item
                for item in logs
                if _goal_operator_audit_log_matches_goal(item, normalized_goal_id)
            ]
            if isinstance(limit, int) and limit > 0:
                logs = logs[:limit]
        return [_goal_operator_audit_log_response(item) for item in logs]

    async def start_goal(self, goal_id: str) -> dict[str, Any] | None:
        goal = await self._store.get_goal(goal_id)
        if goal is None:
            return None
        if str(goal.get("status") or "created") != "created":
            return _goal_response(goal)
        if await self._guard_goal_transition_against_operator_controls(goal):
            refreshed = await self._store.get_goal(goal_id)
            return _goal_response(refreshed or goal)
        if await self._guard_goal_transition_against_runtime_budget(
            goal,
            source="manual_start",
        ):
            refreshed = await self._store.get_goal(goal_id)
            return _goal_response(refreshed or goal)

        attempt_id = str(uuid4())
        await self._store.create_goal_attempt(
            attempt_id=attempt_id,
            goal_id=goal_id,
            attempt_index=_next_goal_attempt_index(goal),
            status="queued",
            trigger="manual_start",
            summary={"queued_from_status": "created"},
        )
        await self._store.update_goal_status(
            goal_id,
            "queued",
            current_attempt_id=attempt_id,
            latest_error=None,
            reset_finished_at=True,
        )
        await self._acquire_goal_lease(
            goal_id,
            reason="manual_start",
        )
        await self._process_goal_supervision()
        await self._store.update_goal_metadata(goal_id, current_attempt_id=attempt_id)
        refreshed = await self._store.get_goal(goal_id)
        return _goal_response(refreshed or goal)

    async def pause_goal(self, goal_id: str) -> dict[str, Any] | None:
        goal = await self._store.get_goal(goal_id)
        if goal is None:
            return None
        current_status = str(goal.get("status") or "created")
        if current_status in GOAL_TERMINAL_STATUS or current_status == "paused":
            return _goal_response(goal)

        current_attempt = _current_goal_attempt(goal)
        if current_attempt is not None and str(current_attempt.get("status") or "") not in {
            "paused",
            *GOAL_TERMINAL_STATUS,
        }:
            agent_run_id = str(current_attempt.get("agent_run_id") or "").strip()
            if agent_run_id:
                await self.pause_agent_run(agent_run_id)
        refreshed_goal = await self._store.get_goal(goal_id)
        if refreshed_goal is None:
            return None
        refreshed_status = str(refreshed_goal.get("status") or "created")
        if refreshed_status in GOAL_TERMINAL_STATUS or refreshed_status == "paused":
            await self._store.delete_goal_lease(goal_id)
            latest_goal = await self._store.get_goal(goal_id)
            return _goal_response(latest_goal or refreshed_goal)
        refreshed_attempt = _current_goal_attempt(refreshed_goal)
        if refreshed_attempt is not None and str(refreshed_attempt.get("status") or "") not in {
            "paused",
            *GOAL_TERMINAL_STATUS,
        }:
            await self._store.update_goal_attempt_status(
                str(refreshed_attempt["id"]),
                "paused",
                latest_error=None,
            )
        await self._store.update_goal_status(goal_id, "paused")
        await self._store.delete_goal_lease(goal_id)
        paused_goal = await self._store.get_goal(goal_id)
        return _goal_response(paused_goal or refreshed_goal)

    async def resume_goal(
        self,
        goal_id: str,
        *,
        strategy: str | None = None,
    ) -> dict[str, Any] | None:
        goal = await self._store.get_goal(goal_id)
        if goal is None:
            return None
        current_status = str(goal.get("status") or "created")
        if current_status == "created":
            return await self.start_goal(goal_id)
        if current_status not in GOAL_RESUMABLE_STATUS:
            return _goal_response(goal)
        if await self._guard_goal_transition_against_operator_controls(goal):
            refreshed = await self._store.get_goal(goal_id)
            return _goal_response(refreshed or goal)
        if await self._guard_goal_transition_against_runtime_budget(
            goal,
            source="manual_resume",
            check_retry_limit=True,
        ):
            refreshed = await self._store.get_goal(goal_id)
            return _goal_response(refreshed or goal)
        resumed_goal = await self._maybe_resume_existing_goal_attempt(
            goal,
            strategy=strategy,
        )
        if resumed_goal is not None:
            return resumed_goal

        attempt_id = str(uuid4())
        await self._store.create_goal_attempt(
            attempt_id=attempt_id,
            goal_id=goal_id,
            attempt_index=_next_goal_attempt_index(goal),
            status="queued",
            trigger="manual_resume",
            summary={"queued_from_status": current_status},
        )
        await self._store.update_goal_status(
            goal_id,
            "queued",
            current_attempt_id=attempt_id,
            latest_error=None,
            reset_finished_at=True,
        )
        await self._acquire_goal_lease(
            goal_id,
            reason="manual_resume",
        )
        await self._process_goal_supervision()
        await self._store.update_goal_metadata(goal_id, current_attempt_id=attempt_id)
        refreshed = await self._store.get_goal(goal_id)
        return _goal_response(refreshed or goal)

    async def refresh_goal(
        self,
        goal_id: str,
        *,
        strategy: str | None = None,
    ) -> dict[str, Any] | None:
        goal = await self._store.get_goal(goal_id)
        if goal is None:
            return None
        current_status = str(goal.get("status") or "created")
        if current_status != "running":
            return _goal_response(goal)
        if await self._guard_goal_transition_against_operator_controls(goal):
            refreshed = await self._store.get_goal(goal_id)
            return _goal_response(refreshed or goal)
        if await self._guard_goal_transition_against_runtime_budget(
            goal,
            source="manual_refresh",
            check_retry_limit=True,
        ):
            refreshed = await self._store.get_goal(goal_id)
            return _goal_response(refreshed or goal)
        refresh_plan = await self._goal_live_manual_refresh_plan(
            goal=goal,
            requested_strategy=strategy,
        )
        if refresh_plan is None:
            raise ValueError(
                "Live worker refresh requires a current durable handoff checkpoint and memory snapshot."
            )
        paused_goal = await self.pause_goal(goal_id)
        if paused_goal is None:
            return None
        if str(paused_goal.get("status") or "") != "paused":
            return paused_goal
        resumed_goal = await self.resume_goal(
            goal_id,
            strategy=str(refresh_plan.get("strategy") or "restart_attempt"),
        )
        return resumed_goal or paused_goal

    async def retry_goal_failed_shard(
        self,
        goal_id: str,
        *,
        shard_id: str | None = None,
        strategy: str | None = None,
    ) -> dict[str, Any] | None:
        goal = await self._store.get_goal(goal_id)
        if goal is None:
            return None
        current_status = str(goal.get("status") or "").strip()
        if current_status not in {"paused", "stalled", "awaiting_resources", "waiting_approval"}:
            raise ValueError("Failed-shard retry requires a non-running resumable goal.")
        current_attempt = _current_goal_attempt(goal)
        if not isinstance(current_attempt, dict):
            raise ValueError("Goal has no current attempt to retry.")
        agent_run_id = str(current_attempt.get("agent_run_id") or "").strip()
        if not agent_run_id:
            raise ValueError("Current goal attempt has no linked agent run.")
        if self._agent_run_job_is_live(agent_run_id):
            raise ValueError("Failed-shard retry requires the linked run to be idle.")
        run = await self._store.get_agent_run(agent_run_id)
        if not isinstance(run, dict):
            raise ValueError("Linked agent run was not found.")
        failed_shards = _failed_collector_shard_manifests_for_run(
            run,
            attempt_id=str(current_attempt.get("id") or "").strip() or None,
        )
        if shard_id is not None and shard_id.strip():
            selected_shard = next(
                (
                    item
                    for item in failed_shards
                    if str(item.get("shard_id") or "").strip() == shard_id.strip()
                ),
                None,
            )
            if selected_shard is None:
                raise ValueError("Requested failed collector shard was not found.")
        else:
            if len(failed_shards) != 1:
                raise ValueError("Specify `shard_id` when more than one failed shard exists.")
            selected_shard = dict(failed_shards[0])

        retry_guidance = _collector_failed_shard_retry_guidance(selected_shard)
        summary = dict(run.get("summary") or {})
        recovery_state = (
            dict(summary.get("recovery_state"))
            if isinstance(summary.get("recovery_state"), dict)
            else {}
        )
        resume_payload = _build_agent_run_resume_payload(
            run=run,
            recovery_state=recovery_state,
            strategy="continue_from_checkpoint",
        )
        protocol_artifacts = (
            dict(resume_payload.get("protocol_artifacts"))
            if isinstance(resume_payload.get("protocol_artifacts"), dict)
            else {}
        )
        protocol_artifacts["collector_shard_manifests"] = {
            "shards": [_collector_failed_shard_retry_manifest(selected_shard)]
        }
        protocol_artifacts["collector_retry_request"] = {
            "mode": "retry_failed_shard",
            "shard_id": str(selected_shard.get("shard_id") or ""),
        }
        existing_guidance = [
            str(item).strip()
            for item in resume_payload.get("guidance_messages") or []
            if isinstance(item, str) and str(item).strip()
        ]
        if retry_guidance not in existing_guidance:
            existing_guidance.append(retry_guidance)
        resume_payload["guidance_messages"] = existing_guidance
        resume_payload["protocol_artifacts"] = protocol_artifacts
        recovery_state["resume_payload"] = resume_payload
        recovery_state["resume_executor"] = str(
            resume_payload.get("executor") or "continue_from_checkpoint"
        )
        summary["recovery_state"] = recovery_state
        await self._store.update_agent_run_metadata(
            agent_run_id,
            summary=summary,
            latest_error=run.get("latest_error"),
        )
        await self._store.append_agent_run_event(
            agent_run_id,
            {
                "type": "guidance",
                "guidance": retry_guidance,
                "metadata": {
                    "source": "retry_failed_shard",
                    "goal_id": goal_id,
                    "shard_id": str(selected_shard.get("shard_id") or ""),
                },
            },
        )
        await self._store.append_goal_operator_audit_log(
            event_type="collector_shard_retry",
            subject_type="goal",
            subject_id=goal_id,
            action="retry_failed_shard",
            summary="Queued a failed collector shard retry on the linked goal run.",
            details={
                "agent_run_id": agent_run_id,
                "attempt_id": current_attempt.get("id"),
                "shard_id": str(selected_shard.get("shard_id") or ""),
                "strategy": _normalize_agent_run_resume_strategy(
                    strategy or "continue_from_checkpoint"
                ),
            },
        )
        return await self.resume_goal(
            goal_id,
            strategy=_normalize_agent_run_resume_strategy(strategy or "continue_from_checkpoint"),
        )

    async def _maybe_resume_existing_goal_attempt(
        self,
        goal: dict[str, Any],
        *,
        strategy: str | None,
    ) -> dict[str, Any] | None:
        goal_id = str(goal.get("id") or "").strip()
        current_status = str(goal.get("status") or "").strip()
        if not goal_id or current_status not in {"paused", "stalled", "awaiting_resources", "waiting_approval"}:
            return None
        current_attempt = _current_goal_attempt(goal)
        if not isinstance(current_attempt, dict):
            return None
        agent_run_id = str(current_attempt.get("agent_run_id") or "").strip()
        if not agent_run_id:
            return None
        if self._agent_run_job_is_live(agent_run_id):
            return None
        run = await self._store.get_agent_run(agent_run_id)
        if not isinstance(run, dict):
            return None
        run_status = str(run.get("status") or "").strip()
        expected_run_status = {
            "paused": "paused",
            "stalled": "stalled",
            "awaiting_resources": "awaiting_resources",
            "waiting_approval": "awaiting_approval",
        }[current_status]
        if run_status != expected_run_status:
            return None
        manual_refresh = (
            await self._goal_manual_refresh_resume_plan(
                goal_id=goal_id,
                run=run,
                requested_strategy=strategy,
            )
            if current_status != "waiting_approval"
            else {
                "source": "manual_resume",
                "strategy": (
                    _normalize_agent_run_resume_strategy(strategy)
                    if isinstance(strategy, str) and strategy.strip()
                    else None
                ),
            }
        )
        await self._acquire_goal_lease(goal_id, reason="manual_resume")
        resumed = await self.resume_agent_run(
            agent_run_id,
            strategy=(
                str(manual_refresh.get("strategy"))
                if isinstance(manual_refresh.get("strategy"), str) and manual_refresh.get("strategy")
                else None
            ),
            source=str(manual_refresh.get("source") or "manual_resume"),
        )
        if resumed is None:
            return None
        await self._sync_goal_from_agent_run_by_run_id(agent_run_id)
        refreshed = await self._store.get_goal(goal_id)
        return _goal_response(refreshed or goal)

    async def cancel_goal(self, goal_id: str) -> dict[str, Any] | None:
        goal = await self._store.get_goal(goal_id)
        if goal is None:
            return None
        current_status = str(goal.get("status") or "created")
        if current_status in GOAL_TERMINAL_STATUS:
            return _goal_response(goal)

        current_attempt = _current_goal_attempt(goal)
        if current_attempt is not None and str(current_attempt.get("status") or "") not in {
            "cancelled",
            *GOAL_TERMINAL_STATUS,
        }:
            agent_run_id = str(current_attempt.get("agent_run_id") or "").strip()
            if agent_run_id:
                await self.cancel_agent_run(agent_run_id)
        refreshed_goal = await self._store.get_goal(goal_id)
        if refreshed_goal is None:
            return None
        refreshed_status = str(refreshed_goal.get("status") or "created")
        if refreshed_status in GOAL_TERMINAL_STATUS:
            await self._store.delete_goal_lease(goal_id)
            latest_goal = await self._store.get_goal(goal_id)
            return _goal_response(latest_goal or refreshed_goal)
        refreshed_attempt = _current_goal_attempt(refreshed_goal)
        if refreshed_attempt is not None and str(refreshed_attempt.get("status") or "") not in {
            "cancelled",
            *GOAL_TERMINAL_STATUS,
        }:
            await self._store.update_goal_attempt_status(
                str(refreshed_attempt["id"]),
                "cancelled",
                latest_error=None,
            )
        await self._store.update_goal_status(goal_id, "cancelled", latest_error=None)
        await self._store.delete_goal_lease(goal_id)
        cancelled_goal = await self._store.get_goal(goal_id)
        return _goal_response(cancelled_goal or refreshed_goal)

    async def create_agent_run(
        self,
        payload: AgentRunCreateRequest,
        *,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        effective_run_id = run_id or str(uuid4())
        schedule = _normalize_agent_run_schedule(payload.schedule)
        summary = dict(payload.summary)
        goal_linked = isinstance(summary.get("goal_id"), str) and bool(str(summary.get("goal_id")).strip())
        default_run_policy = _default_agent_run_policy_from_security(self._security_config)
        if goal_linked and "checkpoint_interval_steps" not in dict(payload.run_policy or {}):
            default_run_policy = {
                key: value for key, value in default_run_policy.items() if key != "checkpoint_interval_steps"
            }
        run_policy = _normalize_agent_run_policy(
            payload.run_policy,
            defaults=default_run_policy,
            apply_default_checkpoint_interval_steps=not goal_linked,
        )
        if payload.reasoning_effort is not None:
            summary["reasoning_effort"] = payload.reasoning_effort
        if payload.project_id is not None:
            summary["project_id"] = payload.project_id
        if payload.workspace_dir is not None:
            summary["workspace_dir"] = payload.workspace_dir
        run = await self._store.create_agent_run(
            run_id=effective_run_id,
            protocol_id=payload.protocol_id,
            title=payload.title,
            topic=payload.topic,
            project_id=payload.project_id,
            workspace_dir=payload.workspace_dir,
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
            effective_run_id,
            {
                "type": "run_created",
                "protocol_id": payload.protocol_id,
                "project_id": payload.project_id,
                "workspace_dir": payload.workspace_dir,
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
        if current_status not in {"created", "paused", "running", "partial", "awaiting_approval", "awaiting_resources", "stalled"}:
            return _agent_run_summary(run)

        if current_status in {"created", "paused", "partial", "awaiting_approval", "awaiting_resources", "stalled"}:
            if current_status in {"created", "partial", "awaiting_approval", "awaiting_resources", "stalled"}:
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
        await self._sync_goal_worker_generation_from_run_status(
            run=run,
            status="paused",
            latest_error=run.get("latest_error") if isinstance(run.get("latest_error"), str) else None,
        )

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
        source: str = "manual_resume",
    ) -> dict[str, Any] | None:
        run = await self._store.get_agent_run(run_id)
        if run is None:
            return None
        current_status = str(run.get("status") or "created")
        if current_status in AGENT_RUN_TERMINAL_STATUS:
            return _agent_run_summary(run)
        if current_status not in AGENT_RUN_RESUMABLE_STATUS:
            return _agent_run_summary(run)

        requested_strategy = _resolve_agent_run_resume_strategy(strategy, run)
        run, effective_strategy = await self._prepare_linked_goal_agent_run_resume(
            run=run,
            strategy=requested_strategy,
        )
        lease = _build_agent_run_resume_lease(
            run_id=run_id,
            strategy=effective_strategy,
            checkpoint_index=_recovery_checkpoint_index_from_run(run),
            source=source,
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
        await self._sync_goal_worker_generation_from_run_status(
            run=run,
            status="cancelled",
            latest_error=run.get("latest_error") if isinstance(run.get("latest_error"), str) else None,
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
        return await self._agent_run_response_with_events(updated)

    async def append_agent_run_message(
        self,
        run_id: str,
        payload: AgentRunMessageRequest,
    ) -> dict[str, Any] | None:
        run = await self._store.get_agent_run(run_id)
        if run is None:
            return None

        message_payload = _normalize_agent_run_message_payload(payload)
        event_type = "operator_message"
        await self._store.append_agent_run_event(
            run_id,
            {
                "type": event_type,
                "role": message_payload["role"],
                "source_role": message_payload["source_role"],
                "content": message_payload["content"],
                "project_id": message_payload["project_id"],
                "workspace_dir": message_payload["workspace_dir"],
                "attachments": message_payload["attachments"],
                "metadata": message_payload["metadata"],
            },
        )

        await self._maybe_update_agent_run_message_context(
            run_id,
            run,
            project_id=message_payload["project_id"],
            workspace_dir=message_payload["workspace_dir"],
            task_input=message_payload["content"],
        )

        refreshed = await self._store.get_agent_run(run_id)
        ack = _build_workflow_assistant_acknowledgement(
            refreshed or run,
            message_payload["content"],
        )
        await self._store.append_agent_run_event(
            run_id,
            {
                "type": "assistant_message",
                "role": "assistant",
                "content": ack,
                "project_id": _agent_run_project_id(refreshed or run),
                "workspace_dir": _agent_run_workspace_dir(refreshed or run),
                "metadata": {
                    "source": "workflow_runtime",
                    "acknowledgement": True,
                },
            },
        )

        updated = await self._store.get_agent_run(run_id)
        return await self._agent_run_response_with_events(updated)

    async def append_agent_run_subagent_message(
        self,
        run_id: str,
        role_id: str,
        payload: AgentRunSubagentMessageRequest,
    ) -> dict[str, Any] | None:
        run = await self._store.get_agent_run(run_id)
        if run is None:
            return None

        message_payload = _normalize_agent_run_message_payload(payload)
        await self._store.append_agent_run_event(
            run_id,
            {
                "type": "subagent_message",
                "role": message_payload["role"],
                "source_role": message_payload["source_role"],
                "target_role_id": role_id,
                "content": message_payload["content"],
                "project_id": message_payload["project_id"],
                "workspace_dir": message_payload["workspace_dir"],
                "attachments": message_payload["attachments"],
                "metadata": message_payload["metadata"],
                "created_context": {
                    "source": "workflow_subagent_message_api",
                    "delivery": "guidance_only",
                    "target_role_id": role_id,
                },
            },
        )
        await self._maybe_update_agent_run_message_context(
            run_id,
            run,
            project_id=message_payload["project_id"],
            workspace_dir=message_payload["workspace_dir"],
        )
        updated = await self._store.get_agent_run(run_id)
        return await self._agent_run_response_with_events(updated)

    async def _maybe_update_agent_run_message_context(
        self,
        run_id: str,
        run: dict[str, Any],
        *,
        project_id: str | None = None,
        workspace_dir: str | None = None,
        task_input: str | None = None,
    ) -> None:
        summary = dict(run.get("summary") or {})
        update_kwargs: dict[str, Any] = {}
        if project_id is not None:
            summary["project_id"] = project_id
            update_kwargs["project_id"] = project_id
        if workspace_dir is not None:
            summary["workspace_dir"] = workspace_dir
            update_kwargs["workspace_dir"] = workspace_dir
        if task_input is not None and task_input and not str(summary.get("task_input") or "").strip():
            summary["task_input"] = task_input
        if update_kwargs or summary != (run.get("summary") or {}):
            await self._store.update_agent_run_metadata(
                run_id,
                summary=summary,
                **update_kwargs,
            )

    async def _agent_run_response_with_events(
        self,
        run: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if run is None:
            return None
        response = _agent_run_summary(run)
        response["events"] = await self._store.get_agent_run_events(run["id"])
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
            "approval_state": (
                dict(summary.get("approval_state"))
                if isinstance(summary.get("approval_state"), dict)
                else {}
            ),
            "candidate_count": int(summary.get("candidate_count") or 0),
            "subagent_health_snapshot": _build_operator_subagent_health_summary(run),
            "role_task_snapshot": _build_operator_role_task_summary(run),
            "detached_exec_jobs": _build_operator_detached_exec_summary(run),
        }

    async def _recover_goals_on_startup(self) -> None:
        await self._process_goal_supervision()

    async def _process_goal_supervision(self) -> None:
        async with self._goal_supervision_lock:
            await self._heartbeat_owned_goal_leases()
            await self._reconcile_goal_supervision(reason="maintenance")
            await self._process_owned_goal_runs()

    async def _heartbeat_owned_goal_leases(self) -> None:
        leases = await self._store.list_goal_leases(owner_id=self._runtime_owner_id)
        if not leases:
            return
        now = _utcnow()
        expires_at = _to_iso_utc(now + timedelta(seconds=self._goal_lease_ttl_seconds))
        heartbeat_at = _to_iso_utc(now)
        for lease in leases:
            goal_id = str(lease.get("goal_id") or "")
            if not goal_id:
                continue
            goal = await self._store.get_goal(goal_id)
            if goal is None:
                continue
            if str(goal.get("status") or "created") not in GOAL_ACTIVE_STATUS:
                await self._store.delete_goal_lease(goal_id)
                continue
            lease_metadata = dict(lease.get("metadata") or {}) if isinstance(lease.get("metadata"), dict) else {}
            await self._store.upsert_goal_lease(
                goal_id=goal_id,
                owner_id=self._runtime_owner_id,
                metadata={
                    **lease_metadata,
                    "reason": "heartbeat",
                    "runtime_owner_id": self._runtime_owner_id,
                },
                heartbeat_at=heartbeat_at,
                expires_at=expires_at,
            )

    async def _reconcile_goal_supervision(self, *, reason: str) -> None:
        now = _utcnow()
        goals = await self._store.list_goals()
        for goal in goals:
            goal_id = str(goal.get("id") or "")
            if not goal_id:
                continue
            current_status = str(goal.get("status") or "created")
            if current_status not in GOAL_ACTIVE_STATUS:
                continue
            lease = await self._store.get_goal_lease(goal_id)
            if lease is None:
                await self._record_goal_audit_finding(
                    goal_id,
                    finding_code="lost_owner",
                    summary=f"Goal lost its supervisor lease during {reason}.",
                    details={
                        "reason": reason,
                        "status": current_status,
                    },
                )
                await self._acquire_goal_lease(goal_id, reason=f"{reason}_recovery")
                continue
            if _goal_lease_is_stale(lease, now=now):
                await self._record_goal_audit_finding(
                    goal_id,
                    finding_code="stale_running",
                    summary=f"Goal lease became stale during {reason} and was taken over.",
                    details={
                        "reason": reason,
                        "status": current_status,
                        "previous_owner_id": lease.get("owner_id"),
                        "expires_at": lease.get("expires_at"),
                    },
                )
                await self._acquire_goal_lease(
                    goal_id,
                    reason=f"{reason}_takeover",
                    force_takeover=True,
                )

    async def _acquire_goal_lease(
        self,
        goal_id: str,
        *,
        reason: str,
        force_takeover: bool = False,
    ) -> dict[str, Any] | None:
        now = _utcnow()
        return await self._store.upsert_goal_lease(
            goal_id=goal_id,
            owner_id=self._runtime_owner_id,
            metadata={
                "reason": reason,
                "runtime_owner_id": self._runtime_owner_id,
            },
            acquired_at=_to_iso_utc(now),
            heartbeat_at=_to_iso_utc(now),
            expires_at=_to_iso_utc(now + timedelta(seconds=self._goal_lease_ttl_seconds)),
            force_takeover=force_takeover,
        )

    async def _defer_running_linked_run_stall(
        self,
        *,
        goal_id: str,
        run_id: str,
        run_updated_at: str | None,
    ) -> bool:
        lease = await self._store.get_goal_lease(goal_id)
        if lease is None or str(lease.get("owner_id") or "") != self._runtime_owner_id:
            return False
        metadata = dict(lease.get("metadata") or {}) if isinstance(lease.get("metadata"), dict) else {}
        probe = (
            dict(metadata.get("running_without_worker_probe"))
            if isinstance(metadata.get("running_without_worker_probe"), dict)
            else {}
        )
        normalized_updated_at = str(run_updated_at or "").strip()
        if (
            str(probe.get("run_id") or "") == run_id
            and str(probe.get("run_updated_at") or "").strip() == normalized_updated_at
        ):
            return False
        metadata["running_without_worker_probe"] = {
            "run_id": run_id,
            "run_updated_at": normalized_updated_at or None,
            "first_seen_at": _now_iso_utc(),
            "runtime_owner_id": self._runtime_owner_id,
        }
        await self._store.upsert_goal_lease(
            goal_id=goal_id,
            owner_id=self._runtime_owner_id,
            metadata=metadata,
            acquired_at=str(lease.get("acquired_at") or _now_iso_utc()),
            heartbeat_at=str(lease.get("heartbeat_at") or _now_iso_utc()),
            expires_at=str(lease.get("expires_at") or _now_iso_utc()),
        )
        return True

    async def _clear_running_linked_run_stall_probe(self, goal_id: str) -> None:
        lease = await self._store.get_goal_lease(goal_id)
        if lease is None or str(lease.get("owner_id") or "") != self._runtime_owner_id:
            return
        metadata = dict(lease.get("metadata") or {}) if isinstance(lease.get("metadata"), dict) else {}
        if metadata.pop("running_without_worker_probe", None) is None:
            return
        await self._store.upsert_goal_lease(
            goal_id=goal_id,
            owner_id=self._runtime_owner_id,
            metadata=metadata,
            acquired_at=str(lease.get("acquired_at") or _now_iso_utc()),
            heartbeat_at=str(lease.get("heartbeat_at") or _now_iso_utc()),
            expires_at=str(lease.get("expires_at") or _now_iso_utc()),
        )

    async def _process_owned_goal_runs(self) -> None:
        operator_controls = await self.get_goal_operator_controls()
        if operator_controls.get("stop_all_goals"):
            await self._pause_running_goals_for_operator_controls(
                _goal_operator_controls_stop_message(operator_controls),
            )
            return
        goals = await self._store.list_goals()
        for goal in goals:
            goal_id = str(goal.get("id") or "")
            if not goal_id:
                continue
            current_status = str(goal.get("status") or "created")
            if current_status not in GOAL_ACTIVE_STATUS:
                continue
            lease = await self._store.get_goal_lease(goal_id)
            if lease is None or str(lease.get("owner_id") or "") != self._runtime_owner_id:
                continue
            if await self._enforce_goal_runtime_budget(goal, source="supervisor"):
                continue
            await self._drive_goal_attempt_execution(goal)

    async def _drive_goal_attempt_execution(self, goal: dict[str, Any]) -> None:
        goal_id = str(goal.get("id") or "")
        if not goal_id:
            return
        normalized_goal_run_policy = _normalize_goal_run_policy(
            goal.get("run_policy"),
            objective=goal.get("objective"),
        )
        attempt = _current_goal_attempt(goal)
        if attempt is None:
            await self._record_goal_audit_finding(
                goal_id,
                finding_code="missing_attempt",
                summary="Goal is active but has no current attempt.",
                severity="error",
                details={"status": goal.get("status")},
            )
            return

        attempt_id = str(attempt.get("id") or "")
        if not attempt_id:
            return
        agent_run_id = str(attempt.get("agent_run_id") or "").strip() or None
        if agent_run_id is None:
            reserved_run_id = str(uuid4())
            claimed_attempt = await self._store.claim_goal_attempt_agent_run_id(
                attempt_id,
                reserved_run_id,
            )
            if claimed_attempt is None:
                return
            if claimed_attempt is not None:
                attempt = claimed_attempt
            claimed_run_id = (
                str(claimed_attempt.get("agent_run_id") or "").strip()
                if isinstance(claimed_attempt, dict)
                else ""
            )
            agent_run_id = claimed_run_id or reserved_run_id

        run = await self._ensure_goal_attempt_agent_run(
            goal=goal,
            attempt_id=attempt_id,
            agent_run_id=agent_run_id,
            normalized_goal_run_policy=normalized_goal_run_policy,
        )
        if run is None:
            await self._record_goal_audit_finding(
                goal_id,
                finding_code="linked_run_missing",
                summary="Linked agent run could not be found for the current goal attempt.",
                severity="error",
                details={"goal_attempt_id": attempt_id, "agent_run_id": agent_run_id},
            )
            await self._store.update_goal_attempt_status(
                attempt_id,
                "stalled",
                latest_error="Linked agent run is missing.",
            )
            await self._store.update_goal_status(
                goal_id,
                "stalled",
                latest_error="Linked agent run is missing.",
            )
            await self._store.delete_goal_lease(goal_id)
            return
        await self._resolve_goal_audit_findings(
            goal_id,
            finding_codes={"linked_run_missing"},
        )

        agent_run_status = str(run.get("status") or "created")
        active_run_job = self._agent_run_job_is_live(agent_run_id)
        if agent_run_status == "created":
            started = await self.start_agent_run(agent_run_id)
            if started is not None:
                run = await self._store.get_agent_run(agent_run_id) or run
                agent_run_status = str(run.get("status") or "created")
                active_run_job = self._agent_run_job_is_live(agent_run_id)
        elif agent_run_status == "stalled" and not active_run_job:
            refresh_resume = await self._goal_stalled_run_refresh_plan(
                goal_id=goal_id,
                attempt_id=attempt_id,
                run=run,
                run_policy=normalized_goal_run_policy,
            )
            resumed = await self.resume_agent_run(
                agent_run_id,
                strategy=(
                    str(refresh_resume.get("strategy") or "").strip()
                    if isinstance(refresh_resume, dict)
                    else None
                ),
                source=(
                    str(refresh_resume.get("source") or "").strip()
                    if isinstance(refresh_resume, dict)
                    else "manual_resume"
                ),
            )
            if resumed is not None:
                run = await self._store.get_agent_run(agent_run_id) or run
                agent_run_status = str(run.get("status") or "created")
                active_run_job = self._agent_run_job_is_live(agent_run_id)
        elif agent_run_status == "running" and not active_run_job:
            if await self._defer_running_linked_run_stall(
                goal_id=goal_id,
                run_id=agent_run_id,
                run_updated_at=(
                    str(run.get("updated_at") or "").strip()
                    if isinstance(run.get("updated_at"), str)
                    else None
                ),
            ):
                await self._sync_goal_from_agent_run(
                    goal_id=goal_id,
                    attempt_id=attempt_id,
                    run=run,
                )
                return
            run = await self._mark_linked_agent_run_stalled(
                goal_id=goal_id,
                attempt_id=attempt_id,
                run_id=agent_run_id,
                reason=(
                    "Runtime ownership was recovered, but the linked agent-run no longer had a live worker."
                ),
            )
            agent_run_status = str(run.get("status") or "stalled")
        if agent_run_status != "running" or active_run_job:
            await self._clear_running_linked_run_stall_probe(goal_id)

        await self._sync_goal_from_agent_run(
            goal_id=goal_id,
            attempt_id=attempt_id,
            run=run,
        )
        if agent_run_status == "running" and active_run_job:
            refreshed_run = await self._store.get_agent_run(agent_run_id)
            if isinstance(refreshed_run, dict):
                live_refresh = await self._maybe_refresh_live_goal_after_context_handoff(
                    goal_id=goal_id,
                    attempt_id=attempt_id,
                    run=refreshed_run,
                    run_policy=normalized_goal_run_policy,
                )
                if live_refresh is None:
                    live_refresh = await self._maybe_refresh_live_goal_after_generation_token_refresh(
                        goal_id=goal_id,
                        attempt_id=attempt_id,
                        run=refreshed_run,
                        run_policy=normalized_goal_run_policy,
                    )
                if live_refresh is None:
                    await self._maybe_refresh_live_goal_after_generation_overdue(
                        goal_id=goal_id,
                        attempt_id=attempt_id,
                        run=refreshed_run,
                        run_policy=normalized_goal_run_policy,
                    )

    def _build_goal_linked_agent_run_request(
        self,
        *,
        goal: Mapping[str, Any],
        attempt_id: str,
        normalized_goal_run_policy: dict[str, Any],
    ) -> AgentRunCreateRequest:
        linked_run_policy = _goal_linked_agent_run_effective_run_policy(
            normalized_goal_run_policy,
        )
        goal_capability_policy = _normalize_goal_capability_policy(
            goal.get("capability_policy"),
        )
        summary = {
            "goal_id": str(goal.get("id") or ""),
            "goal_attempt_id": attempt_id,
            "objective": goal.get("objective"),
            "task_input": goal.get("objective"),
            "interaction_mode": _goal_effective_interaction_mode(goal),
            "execution_topology": _goal_effective_execution_topology(goal),
            "protocol_selection": _goal_protocol_selection(goal),
            "selection_rationale": _goal_selection_rationale(goal),
        }
        if goal_capability_policy:
            summary["goal_capability_policy"] = goal_capability_policy
        return AgentRunCreateRequest(
            protocol_id=_goal_effective_protocol_id(goal) or AUTONOMOUS_SINGLE_AGENT_PROTOCOL,
            title=goal.get("title") if isinstance(goal.get("title"), str) else None,
            topic=goal.get("topic") or goal.get("objective"),
            project_id=goal.get("project_id") if isinstance(goal.get("project_id"), str) else None,
            workspace_dir=(
                goal.get("workspace_dir") if isinstance(goal.get("workspace_dir"), str) else None
            ),
            run_policy=linked_run_policy,
            summary=summary,
        )

    async def _ensure_goal_attempt_agent_run(
        self,
        *,
        goal: Mapping[str, Any],
        attempt_id: str,
        agent_run_id: str | None,
        normalized_goal_run_policy: dict[str, Any],
    ) -> dict[str, Any] | None:
        effective_run_id = str(agent_run_id or "").strip()
        if not effective_run_id:
            return None
        run = await self._store.get_agent_run(effective_run_id)
        if run is not None:
            return run

        request = self._build_goal_linked_agent_run_request(
            goal=goal,
            attempt_id=attempt_id,
            normalized_goal_run_policy=normalized_goal_run_policy,
        )
        try:
            await self.create_agent_run(
                request,
                run_id=effective_run_id,
            )
        except sqlite3.IntegrityError:
            repaired = await self._store.get_agent_run(effective_run_id)
            if repaired is not None:
                return repaired
            raise
        goal_id = str(goal.get("id") or "").strip()
        if goal_id:
            await self._maybe_seed_linked_goal_run_handoff(
                goal_id=goal_id,
                attempt_id=attempt_id,
                agent_run_id=effective_run_id,
            )
        return await self._store.get_agent_run(effective_run_id)

    async def _resolve_goal_handoff_context(
        self,
        *,
        goal_id: str,
        attempt_id: str | None = None,
    ) -> dict[str, Any] | None:
        snapshot = None
        if attempt_id:
            snapshot = await self._store.get_latest_goal_memory_snapshot(
                goal_id,
                attempt_id=attempt_id,
            )
        if snapshot is None:
            snapshot = await self._store.get_latest_goal_memory_snapshot(goal_id)
        if not isinstance(snapshot, dict):
            return None
        checkpoint_id = snapshot.get("checkpoint_id")
        checkpoint = (
            await self._store.get_goal_checkpoint(int(checkpoint_id))
            if isinstance(checkpoint_id, int)
            else None
        )
        metadata = _goal_memory_snapshot_handoff_metadata(
            snapshot_record=snapshot,
            checkpoint_record=checkpoint,
        )
        guidance_messages = _goal_memory_snapshot_guidance_messages(
            snapshot_record=snapshot,
            checkpoint_record=checkpoint,
        )
        return {
            "snapshot_record": snapshot,
            "checkpoint_record": checkpoint,
            "metadata": metadata,
            "guidance_messages": guidance_messages,
        }

    async def _maybe_seed_linked_goal_run_handoff(
        self,
        *,
        goal_id: str,
        attempt_id: str,
        agent_run_id: str,
    ) -> None:
        handoff = await self._resolve_goal_handoff_context(
            goal_id=goal_id,
            attempt_id=attempt_id,
        )
        if not isinstance(handoff, dict):
            return
        await self._apply_goal_handoff_to_agent_run(
            agent_run_id,
            goal_id=goal_id,
            attempt_id=attempt_id,
            handoff_context=handoff,
            goal_handoff_metadata=dict(handoff.get("metadata") or {}),
            include_guidance=True,
        )

    async def _apply_goal_handoff_to_agent_run(
        self,
        agent_run_id: str,
        *,
        goal_id: str,
        attempt_id: str | None,
        handoff_context: dict[str, Any] | None,
        goal_handoff_metadata: dict[str, Any] | None,
        include_guidance: bool,
    ) -> dict[str, Any] | None:
        if include_guidance and isinstance(handoff_context, dict):
            guidance_messages = [
                str(item).strip()
                for item in handoff_context.get("guidance_messages", [])
                if isinstance(item, str) and item.strip()
            ]
            existing_messages = set(await self._collect_guidance_messages(agent_run_id))
            for message in guidance_messages:
                if message in existing_messages:
                    continue
                await self._store.append_agent_run_event(
                    agent_run_id,
                    {
                        "type": "guidance",
                        "guidance": message,
                        "source": "goal_memory_snapshot",
                        "goal_id": goal_id,
                        "goal_attempt_id": attempt_id,
                        "memory_snapshot_id": (
                            handoff_context.get("snapshot_record", {}).get("id")
                            if isinstance(handoff_context.get("snapshot_record"), dict)
                            else None
                        ),
                    },
                )
        run = await self._store.get_agent_run(agent_run_id)
        if not isinstance(run, dict):
            return None
        summary = dict(run.get("summary") or {})
        summary["goal_handoff"] = dict(goal_handoff_metadata or {})
        await self._store.update_agent_run_metadata(
            agent_run_id,
            summary=summary,
        )
        refreshed = await self._store.get_agent_run(agent_run_id)
        return refreshed or run

    async def _prepare_linked_goal_agent_run_resume(
        self,
        *,
        run: dict[str, Any],
        strategy: str,
    ) -> tuple[dict[str, Any], str]:
        normalized_strategy = _normalize_agent_run_resume_strategy(strategy)
        goal_id, attempt_id = _linked_goal_context_from_agent_run(run)
        if goal_id is None:
            summary = dict(run.get("summary") or {})
            recovery_state = (
                dict(summary.get("recovery_state"))
                if isinstance(summary.get("recovery_state"), dict)
                else {}
            )
            if normalized_strategy != "continue_from_checkpoint":
                resume_payload = (
                    dict(recovery_state.get("resume_payload"))
                    if isinstance(recovery_state.get("resume_payload"), dict)
                    else _build_agent_run_resume_payload(
                        run=run,
                        recovery_state=recovery_state,
                        strategy="restart_attempt",
                    )
                )
                recovery_state["resume_payload"] = _restart_attempt_resume_payload(resume_payload)
                recovery_state["resume_executor"] = "restart_attempt"
                summary["recovery_state"] = recovery_state
                await self._store.update_agent_run_metadata(
                    str(run.get("id") or ""),
                    summary=summary,
                )
                refreshed = await self._store.get_agent_run(str(run.get("id") or ""))
                return refreshed or {**run, "summary": summary}, normalized_strategy
            summary = _ensure_agent_run_resume_payload(
                run=run,
                summary=summary,
                strategy=normalized_strategy,
            )
            recovery_state = (
                dict(summary.get("recovery_state"))
                if isinstance(summary.get("recovery_state"), dict)
                else {}
            )
            resume_payload = (
                dict(recovery_state.get("resume_payload"))
                if isinstance(recovery_state.get("resume_payload"), dict)
                else {}
            )
            prepared_run = dict(run)
            prepared_run["summary"] = summary
            effective_strategy = _resolve_agent_run_resume_strategy(None, prepared_run)
            if effective_strategy != "continue_from_checkpoint":
                if not resume_payload:
                    resume_payload = _build_agent_run_resume_payload(
                        run=run,
                        recovery_state=recovery_state,
                        strategy="restart_attempt",
                    )
                recovery_state["resume_payload"] = _restart_attempt_resume_payload(resume_payload)
                recovery_state["resume_executor"] = "restart_attempt"
                summary["recovery_state"] = recovery_state
            await self._store.update_agent_run_metadata(
                str(run.get("id") or ""),
                summary=summary,
            )
            refreshed = await self._store.get_agent_run(str(run.get("id") or ""))
            return refreshed or {**run, "summary": summary}, effective_strategy
        handoff = await self._resolve_goal_handoff_context(
            goal_id=goal_id,
            attempt_id=attempt_id,
        )
        durable_handoff_gate = _goal_linked_durable_handoff_gate(
            run=run,
            attempt_id=attempt_id,
            handoff_context=handoff,
        )
        if normalized_strategy != "continue_from_checkpoint":
            goal_handoff = (
                dict(handoff.get("metadata") or {})
                if isinstance(handoff, dict)
                else {}
            )
            goal_handoff["durable_handoff_gate"] = durable_handoff_gate
            summary = dict(run.get("summary") or {})
            recovery_state = (
                dict(summary.get("recovery_state"))
                if isinstance(summary.get("recovery_state"), dict)
                else {}
            )
            resume_payload = (
                dict(recovery_state.get("resume_payload"))
                if isinstance(recovery_state.get("resume_payload"), dict)
                else _build_agent_run_resume_payload(
                    run=run,
                    recovery_state=recovery_state,
                    strategy="restart_attempt",
                )
            )
            recovery_state["resume_payload"] = _restart_attempt_resume_payload(resume_payload)
            recovery_state["resume_executor"] = "restart_attempt"
            summary["recovery_state"] = recovery_state
            summary["goal_handoff"] = goal_handoff
            await self._store.update_agent_run_metadata(
                str(run.get("id") or ""),
                summary=summary,
            )
            refreshed = await self._store.get_agent_run(str(run.get("id") or ""))
            final_run = refreshed or {**run, "summary": summary}
            if isinstance(handoff, dict) and bool(durable_handoff_gate.get("usable")):
                final_run = (
                    await self._apply_goal_handoff_to_agent_run(
                        str(run.get("id") or ""),
                        goal_id=goal_id,
                        attempt_id=attempt_id,
                        handoff_context=handoff,
                        goal_handoff_metadata=goal_handoff,
                        include_guidance=True,
                    )
                    or final_run
                )
            return final_run, normalized_strategy
        summary = dict(run.get("summary") or {})
        summary = _ensure_agent_run_resume_payload(
            run=run,
            summary=summary,
            strategy=normalized_strategy,
        )
        recovery_state = (
            dict(summary.get("recovery_state"))
            if isinstance(summary.get("recovery_state"), dict)
            else {}
        )
        resume_payload = (
            dict(recovery_state.get("resume_payload"))
            if isinstance(recovery_state.get("resume_payload"), dict)
            else {}
        )
        has_structured_resume_payload = _is_structured_agent_run_resume_payload(resume_payload)
        resume_payload_rehydrated_from_handoff = False
        if isinstance(handoff, dict) and (
            not has_structured_resume_payload
            or _normalize_agent_run_resume_strategy(str(resume_payload.get("executor") or ""))
            != "continue_from_checkpoint"
        ):
            rehydrated = _goal_linked_resume_payload_from_handoff(
                run=run,
                recovery_state=recovery_state,
                handoff_context=handoff,
            )
            if rehydrated is not None:
                resume_payload, recovery_state = rehydrated
                has_structured_resume_payload = True
                summary["recovery_state"] = recovery_state
                resume_payload_rehydrated_from_handoff = True
        prepared_run = dict(run)
        prepared_run["summary"] = summary
        payload_strategy = _resolve_agent_run_resume_strategy(None, prepared_run)
        resume_gate = _goal_linked_resume_gate(
            run=run,
            attempt_id=attempt_id,
            handoff_context=handoff,
            has_structured_resume_payload=has_structured_resume_payload,
            payload_strategy=payload_strategy,
        )
        goal_handoff = (
            dict(handoff.get("metadata") or {})
            if isinstance(handoff, dict)
            else {}
        )
        goal_handoff["durable_handoff_gate"] = durable_handoff_gate
        if resume_payload_rehydrated_from_handoff:
            goal_handoff["resume_payload_rehydrated_from"] = "durable_goal_checkpoint"
        goal_handoff["resume_gate"] = resume_gate
        effective_strategy = str(resume_gate.get("effective_strategy") or normalized_strategy).strip()
        if effective_strategy != "continue_from_checkpoint":
            if not has_structured_resume_payload:
                resume_payload = _build_agent_run_resume_payload(
                    run=run,
                    recovery_state=recovery_state,
                    strategy="restart_attempt",
                )
            recovery_state["resume_payload"] = _restart_attempt_resume_payload(resume_payload)
            recovery_state["resume_executor"] = "restart_attempt"
            summary["recovery_state"] = recovery_state
        elif isinstance(handoff, dict):
            recovery_state["resume_payload"] = _inject_goal_memory_snapshot_into_resume_payload(
                resume_payload,
                handoff_context={
                    **handoff,
                    "metadata": goal_handoff,
                },
            )
            summary["recovery_state"] = recovery_state
        summary["recovery_state"] = recovery_state
        summary["goal_handoff"] = goal_handoff
        prepared_run = dict(run)
        prepared_run["summary"] = summary
        await self._store.update_agent_run_metadata(
            str(run.get("id") or ""),
            summary=summary,
        )
        refreshed = await self._store.get_agent_run(str(run.get("id") or ""))
        final_run = refreshed or prepared_run
        if (
            effective_strategy != "continue_from_checkpoint"
            and isinstance(handoff, dict)
            and bool(durable_handoff_gate.get("usable"))
        ):
            final_run = (
                await self._apply_goal_handoff_to_agent_run(
                    str(run.get("id") or ""),
                    goal_id=goal_id,
                    attempt_id=attempt_id,
                    handoff_context=handoff,
                    goal_handoff_metadata=goal_handoff,
                    include_guidance=True,
                )
                or final_run
            )
        final_strategy = _resolve_agent_run_resume_strategy(None, final_run)
        return final_run, final_strategy

    async def _open_goal_worker_generation(
        self,
        *,
        goal_id: str,
        attempt_id: str | None,
        agent_run_id: str,
        active_resume_runtime: dict[str, Any] | None = None,
        handoff_context: dict[str, Any] | None = None,
        initial_status: str = "running",
    ) -> dict[str, Any] | None:
        latest = await self._store.get_latest_goal_worker_generation(goal_id)
        if isinstance(latest, dict) and latest.get("finished_at") in (None, ""):
            metadata = dict(latest.get("metadata") or {})
            metadata["superseded_by_agent_run_id"] = agent_run_id
            await self._store.update_goal_worker_generation(
                int(latest["id"]),
                status="superseded",
                metadata=metadata,
                finished_at=_now_iso_utc(),
            )
            latest = await self._store.get_goal_worker_generation(int(latest["id"])) or latest
        generation_index = (
            int(latest.get("generation_index") or 0) + 1
            if isinstance(latest, dict)
            else 1
        )
        snapshot_record = (
            handoff_context.get("snapshot_record")
            if isinstance(handoff_context, dict) and isinstance(handoff_context.get("snapshot_record"), dict)
            else None
        )
        snapshot_id = (
            int(snapshot_record["id"])
            if isinstance(snapshot_record, dict) and isinstance(snapshot_record.get("id"), int)
            else None
        )
        rollover_reason = _goal_worker_generation_rollover_reason(
            has_previous_generation=isinstance(latest, dict),
            active_resume_runtime=active_resume_runtime or {},
            handoff_context=handoff_context,
        )
        metadata = {
            "resume_strategy": (
                str(active_resume_runtime.get("strategy") or "").strip()
                if isinstance(active_resume_runtime, dict)
                else None
            ),
            "resume_source": (
                str(active_resume_runtime.get("source") or "").strip()
                if isinstance(active_resume_runtime, dict)
                else None
            ),
            "goal_handoff": (
                dict(handoff_context.get("metadata") or {})
                if isinstance(handoff_context, dict)
                else {}
            ),
        }
        generation = await self._store.create_goal_worker_generation(
            goal_id=goal_id,
            attempt_id=attempt_id,
            agent_run_id=agent_run_id,
            generation_index=generation_index,
            status=initial_status,
            rollover_reason=rollover_reason,
            parent_generation_id=(
                int(latest["id"]) if isinstance(latest, dict) and isinstance(latest.get("id"), int) else None
            ),
            resume_source_snapshot_id=snapshot_id,
            metadata={
                key: value
                for key, value in metadata.items()
                if value not in (None, "", [], {})
            },
            started_at=_now_iso_utc(),
        )
        await self._resolve_goal_audit_findings(
            goal_id,
            finding_codes={
                "generation_refresh_overdue",
                "generation_token_refresh_due",
                "context_handoff_due",
            },
        )
        return generation

    async def _finalize_goal_worker_generation(
        self,
        generation_id: int | None,
        *,
        status: str,
        latest_error: str | None = None,
    ) -> None:
        if not isinstance(generation_id, int):
            return
        generation = await self._store.get_goal_worker_generation(generation_id)
        if not isinstance(generation, dict):
            return
        metadata = dict(generation.get("metadata") or {})
        if latest_error:
            metadata["latest_error"] = latest_error
        await self._store.update_goal_worker_generation(
            generation_id,
            status=_goal_worker_generation_status_from_run_status(status),
            metadata=metadata,
            finished_at=_now_iso_utc(),
        )

    async def _sync_goal_worker_generation_from_run_status(
        self,
        *,
        run: dict[str, Any],
        status: str,
        latest_error: str | None = None,
    ) -> None:
        normalized_status = _goal_worker_generation_status_from_run_status(status)
        if normalized_status == "running":
            return
        goal_id, attempt_id = _linked_goal_context_from_agent_run(run)
        if goal_id is None or attempt_id is None:
            return
        generation = await self._store.get_latest_goal_worker_generation(
            goal_id,
            attempt_id=attempt_id,
        )
        if not isinstance(generation, dict):
            return
        if str(generation.get("agent_run_id") or "").strip() != str(run.get("id") or "").strip():
            return
        if generation.get("finished_at") not in (None, ""):
            return
        await self._finalize_goal_worker_generation(
            int(generation["id"]),
            status=normalized_status,
            latest_error=latest_error,
        )

    def _agent_run_job_is_live(self, run_id: str) -> bool:
        active = self._active_agent_run_jobs.get(run_id)
        return active is not None and not active.done()

    async def _mark_linked_agent_run_stalled(
        self,
        *,
        goal_id: str,
        attempt_id: str,
        run_id: str,
        reason: str,
    ) -> dict[str, Any]:
        detailed_run = await self.get_agent_run(run_id)
        run = detailed_run or await self._store.get_agent_run(run_id)
        if run is None:
            return {"id": run_id, "status": "stalled", "summary": {}, "latest_error": reason}

        summary = dict(run.get("summary") or {})
        recovery_state = (
            dict(summary.get("recovery_state"))
            if isinstance(summary.get("recovery_state"), dict)
            else {}
        )
        recovery_state.update(
            {
                "status": "stalled",
                "action": "resume",
                "reason": reason,
                "stage": "goal_supervisor_recovery",
                "recommended_resume_conditions": [
                    "Resume from the latest durable checkpoint if one is available.",
                    "Restart the attempt if the previous worker was interrupted before a durable checkpoint.",
                ],
            }
        )
        summary["recovery_state"] = recovery_state
        if detailed_run is not None:
            summary = _ensure_agent_run_resume_payload(
                run=detailed_run,
                summary=summary,
                strategy="continue_from_checkpoint",
            )

        await self._store.append_agent_run_event(
            run_id,
            {
                "type": "run_recovery_required",
                "goal_id": goal_id,
                "goal_attempt_id": attempt_id,
                "reason": reason,
                "recovery_status": "stalled",
            },
        )
        await self._store.update_agent_run_metadata(
            run_id,
            summary=summary,
            latest_error=reason,
        )
        await self._store.update_agent_run_status(
            run_id,
            "stalled",
            latest_error=reason,
        )
        await self._record_goal_audit_finding(
            goal_id,
            finding_code="linked_run_interrupted",
            summary="Linked agent-run lost its in-process worker and was marked stalled.",
            details={
                "goal_attempt_id": attempt_id,
                "agent_run_id": run_id,
                "reason": reason,
            },
        )
        return await self._store.get_agent_run(run_id) or run

    async def _sync_goal_from_agent_run_by_run_id(self, run_id: str) -> None:
        run = await self._store.get_agent_run(run_id)
        if run is None:
            return
        summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
        goal_id = str(summary.get("goal_id") or "").strip()
        attempt_id = str(summary.get("goal_attempt_id") or "").strip()
        if not goal_id or not attempt_id:
            return
        await self._sync_goal_from_agent_run(
            goal_id=goal_id,
            attempt_id=attempt_id,
            run=run,
        )

    async def _sync_goal_from_agent_run(
        self,
        *,
        goal_id: str,
        attempt_id: str,
        run: dict[str, Any],
    ) -> None:
        latest_run = await self._store.get_agent_run(str(run.get("id") or ""))
        if latest_run is not None:
            run = latest_run
        run_status = _effective_agent_run_status_for_goal_projection(run)
        linked_approval_state = _goal_approval_state_from_agent_run(run)
        goal_status = _goal_status_from_agent_run(
            run,
            linked_approval_state=linked_approval_state,
        )
        latest_error = run.get("latest_error")
        attempt_status = goal_status
        current_goal = await self._store.get_goal(goal_id)
        current_goal_status = (
            str(current_goal.get("status") or "created")
            if isinstance(current_goal, dict)
            else "created"
        )
        normalized_goal_run_policy = _normalize_goal_run_policy(
            current_goal.get("run_policy") if isinstance(current_goal, dict) else run.get("run_policy"),
            objective=current_goal.get("objective") if isinstance(current_goal, dict) else None,
        )
        if current_goal_status in GOAL_TERMINAL_STATUS and current_goal_status != goal_status:
            return
        existing_attempt = await self._store.get_goal_attempt(attempt_id)
        existing_summary = (
            existing_attempt.get("summary")
            if isinstance(existing_attempt, dict) and isinstance(existing_attempt.get("summary"), dict)
            else {}
        )
        linked_recovery_state = _goal_recovery_state_from_agent_run(run)
        attempt_summary = {
            **existing_summary,
            "agent_run_id": run["id"],
            "agent_run_status": run_status,
            "linked_run_degraded": bool(run.get("degraded", False)),
        }
        if linked_recovery_state:
            attempt_summary["linked_recovery_state"] = linked_recovery_state
        else:
            attempt_summary.pop("linked_recovery_state", None)
        if linked_approval_state:
            attempt_summary["linked_approval_state"] = linked_approval_state
        else:
            attempt_summary.pop("linked_approval_state", None)
        latest_checkpoint, latest_memory_snapshot = await self._persist_goal_progress_snapshots(
            goal_id=goal_id,
            attempt_id=attempt_id,
            goal=current_goal,
            run=run,
            run_policy=normalized_goal_run_policy,
            linked_recovery_state=linked_recovery_state,
            linked_approval_state=linked_approval_state,
        )
        if isinstance(latest_checkpoint, dict):
            attempt_summary["latest_goal_checkpoint_id"] = int(latest_checkpoint["id"])
        else:
            attempt_summary.pop("latest_goal_checkpoint_id", None)
        if isinstance(latest_memory_snapshot, dict):
            attempt_summary["latest_goal_memory_snapshot_id"] = int(latest_memory_snapshot["id"])
        else:
            attempt_summary.pop("latest_goal_memory_snapshot_id", None)
        resumed_run = await self._maybe_resume_goal_after_planned_refresh(
            goal=goal_id,
            run=run,
            current_goal_status=current_goal_status,
            normalized_goal_run_policy=normalized_goal_run_policy,
        )
        if isinstance(resumed_run, dict):
            run = resumed_run
            run_status = _effective_agent_run_status_for_goal_projection(run)
            linked_approval_state = _goal_approval_state_from_agent_run(run)
            goal_status = _goal_status_from_agent_run(
                run,
                linked_approval_state=linked_approval_state,
            )
            latest_error = run.get("latest_error")
            attempt_status = goal_status
            linked_recovery_state = _goal_recovery_state_from_agent_run(run)
            attempt_summary["agent_run_status"] = run_status
            attempt_summary["linked_run_degraded"] = bool(run.get("degraded", False))
            if linked_recovery_state:
                attempt_summary["linked_recovery_state"] = linked_recovery_state
            else:
                attempt_summary.pop("linked_recovery_state", None)
            if linked_approval_state:
                attempt_summary["linked_approval_state"] = linked_approval_state
            else:
                attempt_summary.pop("linked_approval_state", None)
        effective_attempt = dict(existing_attempt) if isinstance(existing_attempt, dict) else {}
        effective_attempt.update(
            {
                "id": attempt_id,
                "goal_id": goal_id,
                "status": attempt_status,
                "agent_run_id": str(run["id"]),
                "summary": attempt_summary,
            }
        )

        current_attempt_id = (
            str(current_goal.get("current_attempt_id") or "").strip()
            if isinstance(current_goal, dict)
            else ""
        )
        if current_attempt_id and current_attempt_id != attempt_id:
            return
        await self._store.update_goal_projection(
            goal_id=goal_id,
            goal_status=goal_status,
            attempt_id=attempt_id,
            attempt_status=attempt_status,
            current_attempt_id=attempt_id,
            agent_run_id=str(run["id"]),
            latest_error=latest_error,
            attempt_summary=attempt_summary,
            reset_goal_finished_at=goal_status not in GOAL_TERMINAL_STATUS,
        )
        if isinstance(current_goal, dict):
            goal_summary = (
                dict(current_goal.get("summary"))
                if isinstance(current_goal.get("summary"), dict)
                else {}
            )
            goal_summary["current_agent_run_status"] = run_status
            goal_summary["current_agent_run_degraded"] = bool(run.get("degraded", False))
            if linked_recovery_state:
                goal_summary["current_recovery_state"] = linked_recovery_state
            else:
                goal_summary.pop("current_recovery_state", None)
            if linked_approval_state:
                goal_summary["current_approval_state"] = linked_approval_state
            else:
                goal_summary.pop("current_approval_state", None)
            if isinstance(latest_checkpoint, dict):
                goal_summary["latest_goal_checkpoint_id"] = int(latest_checkpoint["id"])
            else:
                goal_summary.pop("latest_goal_checkpoint_id", None)
            if isinstance(latest_memory_snapshot, dict):
                goal_summary["latest_goal_memory_snapshot_id"] = int(latest_memory_snapshot["id"])
            else:
                goal_summary.pop("latest_goal_memory_snapshot_id", None)
            await self._store.update_goal_metadata(goal_id, summary=goal_summary)

        await self._sync_goal_worker_generation_from_run_status(
            run=run,
            status=run_status,
            latest_error=latest_error if isinstance(latest_error, str) else None,
        )
        await self._reconcile_goal_checkpoint_policy_finding(
            goal_id=goal_id,
            attempt=effective_attempt,
            goal_status=goal_status,
            run_status=run_status,
            run_policy=normalized_goal_run_policy,
        )
        await self._reconcile_goal_generation_refresh_finding(
            goal_id=goal_id,
            attempt_id=attempt_id,
            agent_run_id=str(run.get("id") or "").strip() or None,
            goal_status=goal_status,
            run_status=run_status,
            run_policy=normalized_goal_run_policy,
        )
        await self._reconcile_goal_generation_token_refresh_finding(
            goal_id=goal_id,
            attempt_id=attempt_id,
            run=run,
            goal_status=goal_status,
            run_status=run_status,
            run_policy=normalized_goal_run_policy,
        )
        await self._reconcile_goal_context_handoff_finding(
            goal_id=goal_id,
            attempt_id=attempt_id,
            run=run,
            goal_status=goal_status,
            run_status=run_status,
            run_policy=normalized_goal_run_policy,
        )
        await self._reconcile_goal_collector_shard_stuck_finding(
            goal_id=goal_id,
            attempt_id=attempt_id,
            run=run,
            goal_status=goal_status,
            run_status=run_status,
            run_policy=normalized_goal_run_policy,
        )
        await self._reconcile_goal_approval_wait_timeout_finding(
            goal_id=goal_id,
            attempt=effective_attempt,
            goal_status=goal_status,
        )
        if goal_status not in GOAL_ACTIVE_STATUS:
            await self._store.delete_goal_lease(goal_id)

    async def _record_goal_audit_finding(
        self,
        goal_id: str,
        *,
        finding_code: str,
        summary: str,
        details: dict[str, Any] | None = None,
        severity: str = "warning",
    ) -> None:
        await self._store.upsert_goal_audit_finding(
            goal_id=goal_id,
            finding_code=finding_code,
            summary=summary,
            severity=severity,
            details=details,
        )

    async def _resolve_goal_audit_findings(
        self,
        goal_id: str,
        *,
        finding_codes: set[str],
        close: bool = False,
    ) -> None:
        findings = await self._store.list_goal_audit_findings(goal_id, status="open")
        for finding in findings:
            if str(finding.get("finding_code") or "") not in finding_codes:
                continue
            finding_id = finding.get("id")
            if not isinstance(finding_id, int):
                continue
            if close:
                await self._store.close_goal_audit_finding(finding_id)
            else:
                await self._store.resolve_goal_audit_finding(finding_id)

    async def _reconcile_goal_generation_refresh_finding(
        self,
        *,
        goal_id: str,
        attempt_id: str | None,
        agent_run_id: str | None,
        goal_status: str,
        run_status: str,
        run_policy: dict[str, Any] | None,
    ) -> None:
        finding_code = "generation_refresh_overdue"
        if (
            not goal_id
            or not attempt_id
            or goal_status not in GOAL_ACTIVE_STATUS
            or run_status != "running"
        ):
            await self._resolve_goal_audit_findings(
                goal_id,
                finding_codes={finding_code},
            )
            return
        latest_generation = await self._store.get_latest_goal_worker_generation(
            goal_id,
            attempt_id=attempt_id,
        )
        if not isinstance(latest_generation, dict):
            await self._resolve_goal_audit_findings(
                goal_id,
                finding_codes={finding_code},
            )
            return
        if latest_generation.get("finished_at") not in (None, ""):
            await self._resolve_goal_audit_findings(
                goal_id,
                finding_codes={finding_code},
            )
            return
        if str(latest_generation.get("status") or "").strip() != "running":
            await self._resolve_goal_audit_findings(
                goal_id,
                finding_codes={finding_code},
            )
            return
        normalized_agent_run_id = str(agent_run_id or "").strip()
        if (
            normalized_agent_run_id
            and str(latest_generation.get("agent_run_id") or "").strip() != normalized_agent_run_id
        ):
            await self._resolve_goal_audit_findings(
                goal_id,
                finding_codes={finding_code},
            )
            return
        generation_policy = _goal_worker_generation_refresh_policy_payload(
            latest_generation,
            run_policy=run_policy,
        )
        if (
            not generation_policy
            or generation_policy.get("generation_refresh_interval_sec") is None
            or not bool(generation_policy.get("refresh_due"))
        ):
            await self._resolve_goal_audit_findings(
                goal_id,
                finding_codes={finding_code},
            )
            return
        await self._record_goal_audit_finding(
            goal_id,
            finding_code=finding_code,
            summary=(
                "Goal worker generation exceeded its refresh interval and should roll to a fresh worker."
            ),
            details={
                "goal_status": goal_status,
                "goal_attempt_id": attempt_id,
                "agent_run_id": latest_generation.get("agent_run_id"),
                "generation": _goal_worker_generation_response(
                    latest_generation,
                    run_policy=run_policy,
                ),
                "generation_refresh_interval_sec": generation_policy.get(
                    "generation_refresh_interval_sec"
                ),
                "elapsed_sec": generation_policy.get("elapsed_sec"),
                "refresh_overdue_sec": generation_policy.get("refresh_overdue_sec"),
            },
            severity="warning",
        )

    async def _reconcile_goal_checkpoint_policy_finding(
        self,
        *,
        goal_id: str,
        attempt: dict[str, Any] | None,
        goal_status: str,
        run_status: str,
        run_policy: dict[str, Any] | None,
    ) -> None:
        finding_codes = {"missing_checkpoint", "checkpoint_overdue"}
        if not goal_id or goal_status not in GOAL_ACTIVE_STATUS or run_status != "running":
            await self._resolve_goal_audit_findings(
                goal_id,
                finding_codes=finding_codes,
            )
            return
        checkpoint_policy = _goal_checkpoint_policy_health_payload(
            dict(run_policy or {}),
            attempt,
            goal_status=goal_status,
        )
        checkpoint_policy_status = str(checkpoint_policy.get("status") or "")
        if checkpoint_policy_status not in {"pending_first_checkpoint", "overdue"}:
            await self._resolve_goal_audit_findings(
                goal_id,
                finding_codes=finding_codes,
            )
            return
        finding_code = (
            "missing_checkpoint"
            if checkpoint_policy_status == "pending_first_checkpoint"
            else "checkpoint_overdue"
        )
        await self._resolve_goal_audit_findings(
            goal_id,
            finding_codes=finding_codes - {finding_code},
        )
        await self._record_goal_audit_finding(
            goal_id,
            finding_code=finding_code,
            summary=(
                "Goal is active but has not produced its first durable checkpoint yet."
                if finding_code == "missing_checkpoint"
                else "Goal checkpoint is older than the configured checkpoint interval."
            ),
            details={
                "goal_status": goal_status,
                "goal_attempt_id": (
                    str(attempt.get("id") or "").strip()
                    if isinstance(attempt, dict)
                    else None
                ),
                "agent_run_id": (
                    str(attempt.get("agent_run_id") or "").strip()
                    if isinstance(attempt, dict)
                    else None
                ),
                "checkpoint_policy": checkpoint_policy,
            },
            severity="warning",
        )

    async def _reconcile_goal_context_handoff_finding(
        self,
        *,
        goal_id: str,
        attempt_id: str | None,
        run: dict[str, Any] | None,
        goal_status: str,
        run_status: str,
        run_policy: dict[str, Any] | None,
    ) -> None:
        finding_code = "context_handoff_due"
        if not goal_id or goal_status not in GOAL_ACTIVE_STATUS or run_status != "running":
            await self._resolve_goal_audit_findings(
                goal_id,
                finding_codes={finding_code},
            )
            return
        latest_generation = (
            await self._store.get_latest_goal_worker_generation(
                goal_id,
                attempt_id=attempt_id,
            )
            if attempt_id
            else None
        )
        context_handoff = _goal_context_handoff_health_payload(
            run,
            run_policy=run_policy,
            generation_started_at=(
                str(latest_generation.get("started_at") or "").strip()
                if isinstance(latest_generation, dict)
                else None
            ),
        )
        if (
            not context_handoff
            or context_handoff.get("context_handoff_threshold") is None
            or not bool(context_handoff.get("context_handoff_due"))
        ):
            await self._resolve_goal_audit_findings(
                goal_id,
                finding_codes={finding_code},
            )
            return
        await self._record_goal_audit_finding(
            goal_id,
            finding_code=finding_code,
            summary=(
                "Goal worker generation reached the context handoff threshold and should hand work to a fresh worker."
            ),
            details={
                "goal_status": goal_status,
                "goal_attempt_id": attempt_id,
                "agent_run_id": str(run.get("id") or "").strip() if isinstance(run, dict) else None,
                "context_handoff": context_handoff,
                "generation": _goal_worker_generation_response(
                    latest_generation,
                    run_policy=run_policy,
                    context_handoff=context_handoff,
                    runtime_telemetry=_goal_subagent_runtime_health_payload(
                        run,
                        run_policy=run_policy,
                        generation_started_at=(
                            str(latest_generation.get("started_at") or "").strip()
                            if isinstance(latest_generation, dict)
                            else None
                        ),
                    ),
                ),
            },
            severity="warning",
        )

    async def _reconcile_goal_generation_token_refresh_finding(
        self,
        *,
        goal_id: str,
        attempt_id: str | None,
        run: dict[str, Any] | None,
        goal_status: str,
        run_status: str,
        run_policy: dict[str, Any] | None,
    ) -> None:
        finding_code = "generation_token_refresh_due"
        if not goal_id or goal_status not in GOAL_ACTIVE_STATUS or run_status != "running":
            await self._resolve_goal_audit_findings(
                goal_id,
                finding_codes={finding_code},
            )
            return
        latest_generation = (
            await self._store.get_latest_goal_worker_generation(
                goal_id,
                attempt_id=attempt_id,
            )
            if attempt_id
            else None
        )
        if not isinstance(latest_generation, dict):
            await self._resolve_goal_audit_findings(
                goal_id,
                finding_codes={finding_code},
            )
            return
        if latest_generation.get("finished_at") not in (None, ""):
            await self._resolve_goal_audit_findings(
                goal_id,
                finding_codes={finding_code},
            )
            return
        if str(latest_generation.get("status") or "").strip() != "running":
            await self._resolve_goal_audit_findings(
                goal_id,
                finding_codes={finding_code},
            )
            return
        run_id = str(run.get("id") or "").strip() if isinstance(run, dict) else ""
        if run_id and str(latest_generation.get("agent_run_id") or "").strip() != run_id:
            await self._resolve_goal_audit_findings(
                goal_id,
                finding_codes={finding_code},
            )
            return
        runtime_telemetry = _goal_subagent_runtime_health_payload(
            run,
            run_policy=run_policy,
            generation_started_at=(
                str(latest_generation.get("started_at") or "").strip()
                if isinstance(latest_generation.get("started_at"), str)
                else None
            ),
        )
        if (
            runtime_telemetry.get("generation_token_refresh_threshold") is None
            or not bool(runtime_telemetry.get("token_refresh_due"))
        ):
            await self._resolve_goal_audit_findings(
                goal_id,
                finding_codes={finding_code},
            )
            return
        await self._record_goal_audit_finding(
            goal_id,
            finding_code=finding_code,
            summary=(
                "Goal worker generation crossed its token refresh threshold and should hand work to a fresh worker."
            ),
            details={
                "goal_status": goal_status,
                "goal_attempt_id": attempt_id,
                "agent_run_id": run_id or None,
                "runtime_telemetry": runtime_telemetry,
                "generation": _goal_worker_generation_response(
                    latest_generation,
                    run_policy=run_policy,
                    runtime_telemetry=runtime_telemetry,
                ),
            },
            severity="warning",
        )

    async def _reconcile_goal_collector_shard_stuck_finding(
        self,
        *,
        goal_id: str,
        attempt_id: str | None,
        run: dict[str, Any] | None,
        goal_status: str,
        run_status: str,
        run_policy: dict[str, Any] | None,
    ) -> None:
        finding_code = "collector_shard_stuck"
        if not goal_id or goal_status not in GOAL_ACTIVE_STATUS or run_status != "running":
            await self._resolve_goal_audit_findings(
                goal_id,
                finding_codes={finding_code},
            )
            return
        collector_state = _goal_collector_state_payload(
            run,
            attempt_id=attempt_id,
            run_policy=run_policy,
        )
        if (
            not collector_state
            or collector_state.get("stall_timeout_sec") is None
            or int(collector_state.get("stalled_shard_count") or 0) <= 0
        ):
            await self._resolve_goal_audit_findings(
                goal_id,
                finding_codes={finding_code},
            )
            return
        await self._record_goal_audit_finding(
            goal_id,
            finding_code=finding_code,
            summary=(
                "One or more collector shards have not advanced within the configured stall timeout."
            ),
            details={
                "goal_status": goal_status,
                "goal_attempt_id": attempt_id,
                "agent_run_id": str(run.get("id") or "").strip() if isinstance(run, dict) else None,
                "collector_state": collector_state,
            },
            severity="warning",
        )

    async def _reconcile_goal_approval_wait_timeout_finding(
        self,
        *,
        goal_id: str,
        attempt: dict[str, Any] | None,
        goal_status: str,
    ) -> None:
        finding_code = "approval_wait_timeout"
        approval_state = _goal_attempt_approval_state_payload(attempt)
        approval_status = (
            str(approval_state.get("status") or "").strip().lower()
            if isinstance(approval_state, dict)
            else ""
        )
        if (
            not goal_id
            or goal_status in GOAL_TERMINAL_STATUS
            or approval_status not in {"waiting_approval", "awaiting_approval"}
        ):
            await self._resolve_goal_audit_findings(
                goal_id,
                finding_codes={finding_code},
            )
            return
        timeout_sec = (
            float(approval_state.get("approval_wait_timeout_sec"))
            if isinstance(approval_state, dict)
            and isinstance(approval_state.get("approval_wait_timeout_sec"), (int, float))
            else None
        )
        elapsed_sec = (
            int(approval_state.get("approval_wait_elapsed_sec"))
            if isinstance(approval_state, dict)
            and isinstance(approval_state.get("approval_wait_elapsed_sec"), int)
            else None
        )
        started_at = (
            str(approval_state.get("approval_wait_started_at") or "").strip()
            if isinstance(approval_state, dict)
            else ""
        )
        if timeout_sec is None or elapsed_sec is None or not started_at or elapsed_sec < timeout_sec:
            await self._resolve_goal_audit_findings(
                goal_id,
                finding_codes={finding_code},
            )
            return
        normalized_timeout_sec: int | float = (
            int(timeout_sec) if float(timeout_sec).is_integer() else round(float(timeout_sec), 6)
        )
        overdue_sec = max(0, elapsed_sec - float(timeout_sec))
        await self._record_goal_audit_finding(
            goal_id,
            finding_code=finding_code,
            summary="Goal has been waiting on operator approval longer than the configured approval wait timeout.",
            details={
                "goal_status": goal_status,
                "goal_attempt_id": (
                    str(attempt.get("id") or "").strip()
                    if isinstance(attempt, dict)
                    else None
                ),
                "agent_run_id": (
                    str(attempt.get("agent_run_id") or "").strip()
                    if isinstance(attempt, dict)
                    else None
                ),
                "approval_state": dict(approval_state or {}),
                "approval_wait_started_at": started_at,
                "approval_wait_elapsed_sec": elapsed_sec,
                "approval_wait_timeout_sec": normalized_timeout_sec,
                "approval_wait_overdue_sec": (
                    int(overdue_sec) if float(overdue_sec).is_integer() else round(float(overdue_sec), 6)
                ),
            },
            severity="warning",
        )

    async def _goal_stalled_run_refresh_plan(
        self,
        *,
        goal_id: str,
        attempt_id: str,
        run: dict[str, Any],
        run_policy: dict[str, Any] | None,
    ) -> dict[str, str] | None:
        latest_generation = await self._store.get_latest_goal_worker_generation(
            goal_id,
            attempt_id=attempt_id,
        )
        if not isinstance(latest_generation, dict):
            return None
        if str(latest_generation.get("agent_run_id") or "").strip() != str(run.get("id") or "").strip():
            return None
        context_handoff = _goal_context_handoff_health_payload(
            run,
            run_policy=run_policy,
            generation_started_at=(
                str(latest_generation.get("started_at") or "").strip()
                if isinstance(latest_generation.get("started_at"), str)
                else None
            ),
        )
        context_handoff_due = bool(context_handoff.get("context_handoff_due"))
        summary = dict(run.get("summary") or {})
        summary = _ensure_agent_run_resume_payload(
            run=run,
            summary=summary,
            strategy="continue_from_checkpoint",
        )
        recovery_state = (
            dict(summary.get("recovery_state"))
            if isinstance(summary.get("recovery_state"), dict)
            else {}
        )
        resume_payload = (
            dict(recovery_state.get("resume_payload"))
            if isinstance(recovery_state.get("resume_payload"), dict)
            else {}
        )
        prepared_run = dict(run)
        prepared_run["summary"] = summary
        handoff = await self._resolve_goal_handoff_context(
            goal_id=goal_id,
            attempt_id=attempt_id,
        )
        durable_handoff_gate = _goal_linked_durable_handoff_gate(
            run=prepared_run,
            attempt_id=attempt_id,
            handoff_context=handoff,
        )
        gate = _goal_linked_resume_gate(
            run=prepared_run,
            attempt_id=attempt_id,
            handoff_context=handoff,
            has_structured_resume_payload=_is_structured_agent_run_resume_payload(resume_payload),
            payload_strategy=_resolve_agent_run_resume_strategy(None, prepared_run),
        )
        effective_strategy = str(gate.get("effective_strategy") or "").strip()
        if effective_strategy not in {"continue_from_checkpoint", "restart_attempt"}:
            return None
        if effective_strategy == "restart_attempt" and not bool(durable_handoff_gate.get("usable")):
            return None
        return {
            "source": (
                "goal_context_handoff_refresh"
                if context_handoff_due
                else "goal_worker_stall_refresh"
            ),
            "strategy": effective_strategy,
        }

    async def _goal_planned_refresh_resume_plan(
        self,
        *,
        goal_id: str,
        run: dict[str, Any],
        run_policy: dict[str, Any] | None,
    ) -> dict[str, str] | None:
        refresh_resume_source = _goal_planned_refresh_resume_source(run_policy)
        if refresh_resume_source is None:
            return None
        _, attempt_id = _linked_goal_context_from_agent_run(run)
        if not goal_id or not attempt_id:
            return {
                "source": refresh_resume_source,
                "strategy": "continue_from_checkpoint",
            }
        handoff = await self._resolve_goal_handoff_context(
            goal_id=goal_id,
            attempt_id=attempt_id,
        )
        durable_handoff_gate = _goal_linked_durable_handoff_gate(
            run=run,
            attempt_id=attempt_id,
            handoff_context=handoff,
        )
        return {
            "source": refresh_resume_source,
            "strategy": (
                "restart_attempt"
                if bool(durable_handoff_gate.get("usable"))
                else "continue_from_checkpoint"
            ),
        }

    async def _goal_manual_refresh_resume_plan(
        self,
        *,
        goal_id: str,
        run: dict[str, Any],
        requested_strategy: str | None,
    ) -> dict[str, str]:
        if isinstance(requested_strategy, str) and requested_strategy.strip():
            return {
                "source": "manual_resume",
                "strategy": _normalize_agent_run_resume_strategy(requested_strategy),
            }
        _, attempt_id = _linked_goal_context_from_agent_run(run)
        if not goal_id or not attempt_id:
            return {
                "source": "manual_resume",
                "strategy": "continue_from_checkpoint",
            }
        handoff = await self._resolve_goal_handoff_context(
            goal_id=goal_id,
            attempt_id=attempt_id,
        )
        durable_handoff_gate = _goal_linked_durable_handoff_gate(
            run=run,
            attempt_id=attempt_id,
            handoff_context=handoff,
        )
        return {
            "source": "manual_resume",
            "strategy": (
                "restart_attempt"
                if bool(durable_handoff_gate.get("usable"))
                else "continue_from_checkpoint"
            ),
        }

    async def _goal_live_manual_refresh_plan(
        self,
        *,
        goal: dict[str, Any],
        requested_strategy: str | None,
    ) -> dict[str, str] | None:
        goal_id = str(goal.get("id") or "").strip()
        if not goal_id:
            return None
        current_attempt = _current_goal_attempt(goal)
        if not isinstance(current_attempt, dict):
            return None
        attempt_id = str(current_attempt.get("id") or "").strip()
        run_id = str(current_attempt.get("agent_run_id") or "").strip()
        if not attempt_id or not run_id or not self._agent_run_job_is_live(run_id):
            return None
        run = await self._store.get_agent_run(run_id)
        if not isinstance(run, dict):
            return None
        if str(run.get("status") or "").strip() != "running":
            return None
        handoff = await self._resolve_goal_handoff_context(
            goal_id=goal_id,
            attempt_id=attempt_id,
        )
        durable_handoff_gate = _goal_linked_durable_handoff_gate(
            run=run,
            attempt_id=attempt_id,
            handoff_context=handoff,
        )
        if not bool(durable_handoff_gate.get("usable")):
            return None
        if isinstance(requested_strategy, str) and requested_strategy.strip():
            strategy = _normalize_agent_run_resume_strategy(requested_strategy)
        else:
            strategy = "restart_attempt"
        return {"source": "manual_resume", "strategy": strategy}

    async def _refresh_live_goal_after_durable_handoff(
        self,
        *,
        goal_id: str,
        attempt_id: str,
        run: dict[str, Any],
        source: str,
        strategy: str = "restart_attempt",
        finding_codes: set[str] | None = None,
    ) -> dict[str, Any] | None:
        run_id = str(run.get("id") or "").strip()
        if not goal_id or not attempt_id or not run_id:
            return None
        handoff = await self._resolve_goal_handoff_context(
            goal_id=goal_id,
            attempt_id=attempt_id,
        )
        durable_handoff_gate = _goal_linked_durable_handoff_gate(
            run=run,
            attempt_id=attempt_id,
            handoff_context=handoff,
        )
        if not bool(durable_handoff_gate.get("usable")):
            return None
        paused = await self.pause_agent_run(run_id)
        if paused is None:
            return None
        paused_status = str(paused.get("status") or "").strip()
        if paused_status != "paused":
            await self._sync_goal_from_agent_run_by_run_id(run_id)
            return None
        await self._acquire_goal_lease(goal_id, reason=source)
        resumed = await self.resume_agent_run(
            run_id,
            strategy=strategy,
            source=source,
        )
        if resumed is None:
            await self._sync_goal_from_agent_run_by_run_id(run_id)
            return None
        await asyncio.sleep(0)
        await self._sync_goal_from_agent_run_by_run_id(run_id)
        refreshed = await self._store.get_agent_run(run_id)
        final_run = refreshed if isinstance(refreshed, dict) else resumed
        if (
            isinstance(final_run, dict)
            and str(final_run.get("status") or "").strip() == "running"
            and finding_codes
        ):
            await self._resolve_goal_audit_findings(
                goal_id,
                finding_codes=finding_codes,
            )
        return final_run

    async def _maybe_refresh_live_goal_after_generation_overdue(
        self,
        *,
        goal_id: str,
        attempt_id: str,
        run: dict[str, Any],
        run_policy: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        run_id = str(run.get("id") or "").strip()
        if not goal_id or not attempt_id or not run_id:
            return None
        if str(run.get("status") or "").strip() != "running":
            return None
        if not self._agent_run_job_is_live(run_id):
            return None
        active_resume_runtime = _resume_runtime_from_run(run)
        if (
            str(active_resume_runtime.get("source") or "").strip().lower()
            == "goal_generation_refresh"
            and str(active_resume_runtime.get("status") or "").strip().lower() == "active"
        ):
            return None
        latest_generation = await self._store.get_latest_goal_worker_generation(
            goal_id,
            attempt_id=attempt_id,
        )
        if not isinstance(latest_generation, dict):
            return None
        if str(latest_generation.get("agent_run_id") or "").strip() != run_id:
            return None
        if latest_generation.get("finished_at") not in (None, ""):
            return None
        if str(latest_generation.get("status") or "").strip() != "running":
            return None
        generation_policy = _goal_worker_generation_refresh_policy_payload(
            latest_generation,
            run_policy=run_policy,
        )
        if (
            generation_policy.get("generation_refresh_interval_sec") is None
            or not bool(generation_policy.get("refresh_due"))
        ):
            return None
        return await self._refresh_live_goal_after_durable_handoff(
            goal_id=goal_id,
            attempt_id=attempt_id,
            run=run,
            source="goal_generation_refresh",
            strategy="restart_attempt",
            finding_codes={
                "generation_refresh_overdue",
                "generation_token_refresh_due",
                "context_handoff_due",
            },
        )

    async def _maybe_refresh_live_goal_after_context_handoff(
        self,
        *,
        goal_id: str,
        attempt_id: str,
        run: dict[str, Any],
        run_policy: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        run_id = str(run.get("id") or "").strip()
        if not goal_id or not attempt_id or not run_id:
            return None
        if str(run.get("status") or "").strip() != "running":
            return None
        if not self._agent_run_job_is_live(run_id):
            return None
        active_resume_runtime = _resume_runtime_from_run(run)
        if (
            str(active_resume_runtime.get("source") or "").strip().lower()
            == "goal_context_handoff_refresh"
            and str(active_resume_runtime.get("status") or "").strip().lower() == "active"
        ):
            return None
        latest_generation = await self._store.get_latest_goal_worker_generation(
            goal_id,
            attempt_id=attempt_id,
        )
        if not isinstance(latest_generation, dict):
            return None
        if str(latest_generation.get("agent_run_id") or "").strip() != run_id:
            return None
        if latest_generation.get("finished_at") not in (None, ""):
            return None
        if str(latest_generation.get("status") or "").strip() != "running":
            return None
        context_handoff = _goal_context_handoff_health_payload(
            run,
            run_policy=run_policy,
            generation_started_at=(
                str(latest_generation.get("started_at") or "").strip()
                if isinstance(latest_generation.get("started_at"), str)
                else None
            ),
        )
        if (
            context_handoff.get("context_handoff_threshold") is None
            or not bool(context_handoff.get("context_handoff_due"))
        ):
            return None
        return await self._refresh_live_goal_after_durable_handoff(
            goal_id=goal_id,
            attempt_id=attempt_id,
            run=run,
            source="goal_context_handoff_refresh",
            strategy="restart_attempt",
            finding_codes={
                "generation_refresh_overdue",
                "generation_token_refresh_due",
                "context_handoff_due",
            },
        )

    async def _maybe_refresh_live_goal_after_generation_token_refresh(
        self,
        *,
        goal_id: str,
        attempt_id: str,
        run: dict[str, Any],
        run_policy: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        run_id = str(run.get("id") or "").strip()
        if not goal_id or not attempt_id or not run_id:
            return None
        if str(run.get("status") or "").strip() != "running":
            return None
        if not self._agent_run_job_is_live(run_id):
            return None
        active_resume_runtime = _resume_runtime_from_run(run)
        if (
            str(active_resume_runtime.get("source") or "").strip().lower()
            == "goal_token_refresh"
            and str(active_resume_runtime.get("status") or "").strip().lower() == "active"
        ):
            return None
        latest_generation = await self._store.get_latest_goal_worker_generation(
            goal_id,
            attempt_id=attempt_id,
        )
        if not isinstance(latest_generation, dict):
            return None
        if str(latest_generation.get("agent_run_id") or "").strip() != run_id:
            return None
        if latest_generation.get("finished_at") not in (None, ""):
            return None
        if str(latest_generation.get("status") or "").strip() != "running":
            return None
        runtime_telemetry = _goal_subagent_runtime_health_payload(
            run,
            run_policy=run_policy,
            generation_started_at=(
                str(latest_generation.get("started_at") or "").strip()
                if isinstance(latest_generation.get("started_at"), str)
                else None
            ),
        )
        if (
            runtime_telemetry.get("generation_token_refresh_threshold") is None
            or not bool(runtime_telemetry.get("token_refresh_due"))
        ):
            return None
        return await self._refresh_live_goal_after_durable_handoff(
            goal_id=goal_id,
            attempt_id=attempt_id,
            run=run,
            source="goal_token_refresh",
            strategy="restart_attempt",
            finding_codes={
                "generation_refresh_overdue",
                "generation_token_refresh_due",
                "context_handoff_due",
            },
        )

    async def _maybe_resume_goal_after_planned_refresh(
        self,
        *,
        goal: str,
        run: dict[str, Any],
        current_goal_status: str,
        normalized_goal_run_policy: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if current_goal_status not in GOAL_ACTIVE_STATUS:
            return None
        refresh_resume = await self._goal_planned_refresh_resume_plan(
            goal_id=goal,
            run=run,
            run_policy=normalized_goal_run_policy,
        )
        if not isinstance(refresh_resume, dict):
            return None
        if not _goal_should_autoresume_planned_refresh(
            run,
            run_policy=normalized_goal_run_policy,
        ):
            return None
        run_id = str(run.get("id") or "").strip()
        if not run_id or self._agent_run_job_is_live(run_id):
            return None
        resumed = await self.resume_agent_run(
            run_id,
            strategy=str(refresh_resume.get("strategy") or "continue_from_checkpoint"),
            source=str(refresh_resume.get("source") or "goal_generation_refresh"),
        )
        if resumed is None:
            return None
        refreshed = await self._store.get_agent_run(run_id)
        if not isinstance(refreshed, dict):
            return None
        if str(refreshed.get("status") or "").strip() != "running":
            return None
        await self._resolve_goal_audit_findings(
            goal,
            finding_codes={
                "generation_refresh_overdue",
                "generation_token_refresh_due",
                "context_handoff_due",
            },
        )
        return refreshed

    async def _persist_goal_progress_snapshots(
        self,
        *,
        goal_id: str,
        attempt_id: str,
        goal: dict[str, Any] | None,
        run: dict[str, Any],
        run_policy: dict[str, Any] | None,
        linked_recovery_state: dict[str, Any],
        linked_approval_state: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        checkpoint_payload = _build_goal_checkpoint_record_payload(
            attempt_id=attempt_id,
            run=run,
            run_policy=run_policy,
            linked_recovery_state=linked_recovery_state,
            linked_approval_state=linked_approval_state,
        )
        checkpoint_record: dict[str, Any] | None = None
        if isinstance(checkpoint_payload, dict):
            checkpoint_record = await self._persist_goal_checkpoint_record(
                goal_id=goal_id,
                attempt_id=attempt_id,
                run=run,
                checkpoint_payload=checkpoint_payload,
            )

        memory_snapshot = _build_goal_memory_snapshot_payload(
            goal=goal,
            attempt_id=attempt_id,
            run=run,
            run_policy=run_policy,
            linked_recovery_state=linked_recovery_state,
            linked_approval_state=linked_approval_state,
            checkpoint_record=checkpoint_record,
        )
        memory_snapshot_record: dict[str, Any] | None = None
        if isinstance(memory_snapshot, dict):
            memory_snapshot_record = await self._persist_goal_memory_snapshot_record(
                goal_id=goal_id,
                attempt_id=attempt_id,
                checkpoint_record=checkpoint_record,
                memory_snapshot=memory_snapshot,
            )
        return checkpoint_record, memory_snapshot_record

    async def _persist_goal_checkpoint_record(
        self,
        *,
        goal_id: str,
        attempt_id: str,
        run: dict[str, Any],
        checkpoint_payload: dict[str, Any],
    ) -> dict[str, Any]:
        signature = _goal_checkpoint_record_signature(checkpoint_payload)
        latest = await self._store.get_latest_goal_checkpoint(goal_id, attempt_id=attempt_id)
        if (
            isinstance(latest, dict)
            and isinstance(latest.get("metadata"), dict)
            and str(latest["metadata"].get("signature") or "") == signature
        ):
            return latest
        captured_at = (
            str(checkpoint_payload.get("captured_at") or "").strip()
            if isinstance(checkpoint_payload.get("captured_at"), str)
            else ""
        )
        return await self._store.create_goal_checkpoint(
            goal_id=goal_id,
            attempt_id=attempt_id,
            agent_run_id=str(run["id"]),
            checkpoint_index=(
                int(checkpoint_payload["checkpoint_index"])
                if isinstance(checkpoint_payload.get("checkpoint_index"), int)
                else None
            ),
            stage=(
                str(checkpoint_payload.get("stage")).strip()
                if isinstance(checkpoint_payload.get("stage"), str)
                and str(checkpoint_payload.get("stage")).strip()
                else None
            ),
            source="agent_run_sync",
            payload=checkpoint_payload,
            metadata={"signature": signature},
            captured_at=captured_at or None,
        )

    async def _persist_goal_memory_snapshot_record(
        self,
        *,
        goal_id: str,
        attempt_id: str,
        checkpoint_record: dict[str, Any] | None,
        memory_snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        signature = _goal_memory_snapshot_signature(memory_snapshot)
        latest = await self._store.get_latest_goal_memory_snapshot(goal_id, attempt_id=attempt_id)
        if (
            isinstance(latest, dict)
            and isinstance(latest.get("metadata"), dict)
            and str(latest["metadata"].get("signature") or "") == signature
        ):
            return latest
        captured_at = (
            str(memory_snapshot.get("captured_at") or "").strip()
            if isinstance(memory_snapshot.get("captured_at"), str)
            else ""
        )
        checkpoint_id = (
            int(checkpoint_record["id"])
            if isinstance(checkpoint_record, dict) and isinstance(checkpoint_record.get("id"), int)
            else None
        )
        return await self._store.create_goal_memory_snapshot(
            goal_id=goal_id,
            attempt_id=attempt_id,
            checkpoint_id=checkpoint_id,
            snapshot_kind="compact_recovery_v1",
            snapshot=memory_snapshot,
            metadata={"signature": signature},
            captured_at=captured_at or None,
        )

    async def _guard_goal_transition_against_runtime_budget(
        self,
        goal: dict[str, Any],
        *,
        source: str,
        check_retry_limit: bool = False,
    ) -> bool:
        normalized_run_policy = _normalize_goal_run_policy(
            goal.get("run_policy"),
            objective=goal.get("objective"),
        )
        budget = _goal_runtime_budget_health_payload(goal, normalized_run_policy)
        if not _goal_runtime_budget_blocks_goal_transition(
            budget,
            check_retry_limit=check_retry_limit,
        ):
            await self._resolve_goal_audit_findings(
                str(goal.get("id") or ""),
                finding_codes={"runtime_budget_exhausted"},
            )
            return False
        await self._apply_goal_runtime_budget_hold(
            goal,
            budget=budget,
            source=source,
        )
        return True

    async def _enforce_goal_runtime_budget(
        self,
        goal: dict[str, Any],
        *,
        source: str,
    ) -> bool:
        normalized_run_policy = _normalize_goal_run_policy(
            goal.get("run_policy"),
            objective=goal.get("objective"),
        )
        budget = _goal_runtime_budget_health_payload(goal, normalized_run_policy)
        if not _goal_runtime_budget_is_exhausted(budget):
            await self._resolve_goal_audit_findings(
                str(goal.get("id") or ""),
                finding_codes={"runtime_budget_exhausted"},
            )
            return False
        await self._apply_goal_runtime_budget_hold(
            goal,
            budget=budget,
            source=source,
        )
        return True

    async def _apply_goal_runtime_budget_hold(
        self,
        goal: dict[str, Any],
        *,
        budget: dict[str, Any],
        source: str,
    ) -> None:
        goal_id = str(goal.get("id") or "")
        if not goal_id:
            return
        block_reason = _goal_runtime_budget_block_reason(budget)
        if block_reason is None:
            return
        current_status = str(goal.get("status") or "created")
        current_attempt = _current_goal_attempt(goal)
        attempt_id = str(current_attempt.get("id") or "").strip() if isinstance(current_attempt, dict) else ""
        agent_run_id = (
            str(current_attempt.get("agent_run_id") or "").strip()
            if isinstance(current_attempt, dict)
            else ""
        )
        hold_status = _goal_runtime_budget_hold_status(budget)
        finding_details = {
            "source": source,
            "goal_status": current_status,
            "budget": budget,
        }
        await self._record_goal_audit_finding(
            goal_id,
            finding_code="runtime_budget_exhausted",
            summary=block_reason,
            severity="warning",
            details=finding_details,
        )
        goal_summary = (
            dict(goal.get("summary"))
            if isinstance(goal.get("summary"), dict)
            else {}
        )
        goal_summary["runtime_budget"] = budget
        goal_summary["runtime_budget_block_reason"] = block_reason
        goal_summary["runtime_budget_blocked_from"] = source
        await self._store.update_goal_metadata(
            goal_id,
            summary=goal_summary,
            latest_error=block_reason,
        )
        if current_status in GOAL_ACTIVE_STATUS and agent_run_id:
            await self.pause_agent_run(agent_run_id)
        if current_status in GOAL_ACTIVE_STATUS and attempt_id:
            attempt_summary = (
                dict(current_attempt.get("summary"))
                if isinstance(current_attempt, dict) and isinstance(current_attempt.get("summary"), dict)
                else {}
            )
            attempt_summary["runtime_budget"] = budget
            attempt_summary["runtime_budget_block_reason"] = block_reason
            await self._store.update_goal_attempt_status(
                attempt_id,
                hold_status,
                latest_error=block_reason,
                summary=attempt_summary,
            )
        if current_status == "created" or (
            current_status in GOAL_RESUMABLE_STATUS and current_status not in GOAL_TERMINAL_STATUS
        ):
            await self._store.update_goal_status(
                goal_id,
                hold_status,
                latest_error=block_reason,
            )
        elif current_status in GOAL_ACTIVE_STATUS:
            await self._store.update_goal_status(
                goal_id,
                hold_status,
                latest_error=block_reason,
            )
            await self._store.delete_goal_lease(goal_id)

    async def _guard_goal_transition_against_operator_controls(
        self,
        goal: Mapping[str, Any],
    ) -> bool:
        controls = await self.get_goal_operator_controls()
        if not controls.get("stop_all_goals"):
            return False
        goal_id = str(goal.get("id") or "").strip()
        if goal_id:
            await self._store.update_goal_metadata(
                goal_id,
                latest_error=_goal_operator_controls_stop_message(controls),
            )
        return True

    async def _pause_running_goals_for_operator_controls(self, reason: str) -> None:
        goals = await self._store.list_goals()
        for goal in goals:
            goal_id = str(goal.get("id") or "").strip()
            if not goal_id:
                continue
            if str(goal.get("status") or "").strip() not in GOAL_ACTIVE_STATUS:
                continue
            paused = await self.pause_goal(goal_id)
            if paused is None:
                continue
            await self._store.update_goal_metadata(
                goal_id,
                latest_error=reason,
            )

    async def _scheduler_loop(self) -> None:
        while True:
            try:
                await self._process_goal_supervision()
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
        linked_goal_id, linked_goal_attempt_id = _linked_goal_context_from_agent_run(run)
        handoff_context = (
            await self._resolve_goal_handoff_context(
                goal_id=linked_goal_id,
                attempt_id=linked_goal_attempt_id,
            )
            if linked_goal_id is not None
            else None
        )
        effective_run_policy = (
            _goal_linked_agent_run_effective_run_policy(
                run.get("run_policy") if isinstance(run.get("run_policy"), dict) else {},
            )
            if linked_goal_id is not None
            else (run.get("run_policy") if isinstance(run.get("run_policy"), dict) else {})
        )
        current_generation = (
            await self._open_goal_worker_generation(
                goal_id=linked_goal_id,
                attempt_id=linked_goal_attempt_id,
                agent_run_id=run_id,
                active_resume_runtime=active_resume_runtime if isinstance(active_resume_runtime, dict) else None,
                handoff_context=handoff_context,
                initial_status="running",
            )
            if linked_goal_id is not None
            else None
        )
        current_generation_id = (
            int(current_generation["id"])
            if isinstance(current_generation, dict) and isinstance(current_generation.get("id"), int)
            else None
        )

        try:
            security_config = self._security_config
            orchestrator = MultiAgentOrchestrator(
                engine=self._engine,
                exec_runtime=self._exec_runtime,
                exec_approval_store=self._exec_approval_store,
                controlled_exec_require_approval=resolve_runtime_permission_policy(
                    security_config
                ).require_approval_for_exec,
                controlled_exec_command_rules=[
                    rule.model_dump() for rule in security_config.command_rules
                ],
                controlled_exec_allowed_env_vars=security_config.exec_allowed_env_vars,
                controlled_exec_default_shell=security_config.exec_default_shell,
            )
            protocol_payload = _resolve_protocol_payload(run)
            task_workspace_dir = None
            if run.get("protocol_id") == "controlled_subagent_execution":
                task_workspace_dir = (self._runtime_tasks_root / f"agent-run-{run_id}" / "workspace").resolve()
                task_workspace_dir.mkdir(parents=True, exist_ok=True)
            goal_operator_controls = (
                await self.get_goal_operator_controls()
                if linked_goal_id is not None
                else None
            )
            attempt_key = attempt_id or "latest"
            existing_artifacts = [
                dict(item)
                for item in run.get("artifacts") or []
                if isinstance(item, dict)
            ]
            live_collector_shard_manifests = dedupe_collector_shard_manifests(
                extract_collector_shard_manifests(existing_artifacts)
            )
            persisted_collector_dataset_records = dedupe_collector_dataset_records(
                extract_persisted_collector_dataset_records(
                    existing_artifacts,
                    attempt_id=attempt_id,
                )
            )
            live_collector_record_provenance = collector_record_provenance_list_from_dataset_records(
                persisted_collector_dataset_records
            )
            persisted_collector_record_keys = {
                (
                    _string(record.get("attempt_id")) or attempt_id,
                    collector_dataset_record_identity(record),
                )
                for record in persisted_collector_dataset_records
            }
            async def _persist_live_runtime_event(event: Any) -> None:
                nonlocal live_collector_shard_manifests
                nonlocal live_collector_record_provenance
                nonlocal persisted_collector_dataset_records
                if str(getattr(event, "type", "") or "") != "artifact":
                    return
                payload = getattr(event, "payload", None)
                if not isinstance(payload, dict):
                    return
                artifact_name = str(payload.get("name") or "")
                if artifact_name == "subagent_runtime":
                    subagent_runtime = payload.get("content")
                    if not isinstance(subagent_runtime, dict):
                        return
                    await self._store.append_agent_run_artifact(
                        run_id,
                        artifact_id=(
                            f"{run_id}:attempt:{attempt_key}:subagent-runtime-live:"
                            f"{int(getattr(event, 'seq', 0) or 0)}"
                        ),
                        artifact_type="subagent_runtime",
                        title="Subagent Runtime Trace",
                        uri=(
                            f"agent-run://{run_id}/artifacts/{attempt_key}/"
                            f"subagent-runtime-live-{int(getattr(event, 'seq', 0) or 0)}"
                        ),
                        mime_type="application/json",
                        metadata={
                            **({"attempt_id": attempt_id} if attempt_id else {}),
                            "content": subagent_runtime,
                        },
                    )
                    return
                if artifact_name == "collector_record_provenance":
                    live_collector_record_provenance = collector_record_provenance_list_from_result(
                        {"collector_record_provenance": payload.get("content")}
                    )
                    return
                if artifact_name == "collector_dataset_records":
                    normalized_records = collector_dataset_records_from_result(
                        {"collector_dataset_records": payload.get("content")}
                    )
                    if not normalized_records:
                        return
                    seq = int(getattr(event, "seq", 0) or 0)
                    new_records: list[dict[str, Any]] = []
                    for index, record in enumerate(normalized_records):
                        normalized_record = normalize_collector_dataset_record_for_run(
                            record,
                            run_id=run_id,
                            protocol=str(run.get("protocol_id") or ""),
                            state="running",
                        )
                        metadata = (
                            dict(normalized_record.get("metadata"))
                            if isinstance(normalized_record.get("metadata"), dict)
                            else {}
                        )
                        inferred_provenance = collector_record_provenance_for_index(
                            {"collector_record_provenance": {"records": live_collector_record_provenance}},
                            index=index,
                            shard_manifests=live_collector_shard_manifests,
                        )
                        if inferred_provenance is not None and not isinstance(
                            metadata.get("collector_provenance"), dict
                        ):
                            metadata["collector_provenance"] = inferred_provenance
                        if metadata:
                            normalized_record["metadata"] = metadata
                        record_key = (
                            attempt_id,
                            collector_dataset_record_identity(normalized_record),
                        )
                        if record_key in persisted_collector_record_keys:
                            continue
                        collector_provenance = (
                            dict(metadata.get("collector_provenance"))
                            if isinstance(metadata.get("collector_provenance"), dict)
                            else {}
                        )
                        record_source_id = str(
                            collector_provenance.get("source_id")
                            or collector_provenance.get("shard_id")
                            or index + 1
                        )
                        await self._store.append_agent_run_artifact(
                            run_id,
                            artifact_id=(
                                f"{run_id}:attempt:{attempt_key}:collector-dataset-live:"
                                f"{seq}:{index + 1}"
                            ),
                            artifact_type="dataset_record",
                            title=f"Collector Dataset Record {record_source_id}",
                            uri=(
                                f"agent-run://{run_id}/artifacts/{attempt_key}/"
                                f"collector-dataset-live-{seq}-{index + 1}"
                            ),
                            mime_type="application/json",
                            metadata={
                                **({"attempt_id": attempt_id} if attempt_id else {}),
                                "record": normalized_record,
                            },
                        )
                        persisted_collector_record_keys.add(record_key)
                        new_records.append(
                            {
                                **normalized_record,
                                **({"attempt_id": attempt_id} if attempt_id else {}),
                            }
                        )
                    if new_records:
                        persisted_collector_dataset_records = dedupe_collector_dataset_records(
                            [*persisted_collector_dataset_records, *new_records]
                        )
                        live_collector_record_provenance = (
                            collector_record_provenance_list_from_dataset_records(
                                persisted_collector_dataset_records
                            )
                        )
                    return
                if artifact_name != "collector_shard_manifests":
                    return
                normalized_shards = collector_shard_manifests_from_result(
                    {"collector_shard_manifests": payload.get("content")},
                    attempt_id=attempt_id,
                )
                normalized_shards = dedupe_collector_shard_manifests(normalized_shards)
                if not normalized_shards:
                    return
                live_collector_shard_manifests = normalized_shards
                await self._store.append_agent_run_artifact(
                    run_id,
                    artifact_id=(
                        f"{run_id}:attempt:{attempt_key}:collector-shard-live:"
                        f"{int(getattr(event, 'seq', 0) or 0)}"
                    ),
                    artifact_type=COLLECTOR_SHARD_MANIFEST_ARTIFACT_TYPE,
                    title="Collector Shard Progress Snapshot",
                    uri=(
                        f"agent-run://{run_id}/artifacts/{attempt_key}/"
                        f"collector-shard-live-{int(getattr(event, 'seq', 0) or 0)}"
                    ),
                    mime_type="application/json",
                    metadata={
                        **({"attempt_id": attempt_id} if attempt_id else {}),
                        "content": {"shards": normalized_shards},
                    },
                )
            request = MultiAgentRunRequest(
                run_id=run_id,
                task_input=_agent_run_task_input(run),
                protocol=protocol_payload,
                reasoning_effort=_agent_run_reasoning_effort(run),
                guidance_messages=await self._collect_guidance_messages(run_id),
                role_guidance_messages=await self._collect_role_guidance_messages(run_id),
                run_policy=effective_run_policy,
                metadata={
                    "title": run.get("title"),
                    "topic": run.get("topic"),
                    "reasoning_effort": _agent_run_reasoning_effort(run),
                    "selected_models_roles": run.get("selected_models_roles") or {},
                    "evaluation_policy": run.get("evaluation_policy") or {},
                    "schedule": run.get("schedule") or {},
                    "summary": run.get("summary") or {},
                    "goal_capability_policy": _normalize_goal_capability_policy(
                        (run.get("summary") or {}).get("goal_capability_policy")
                        if isinstance(run.get("summary"), dict)
                        else {}
                    ),
                    "goal_operator_controls": (
                        _normalize_goal_operator_controls(goal_operator_controls)
                        if goal_operator_controls is not None
                        else None
                    ),
                    "permission_policy": _goal_operator_controls_permission_policy(
                        goal_operator_controls
                    ),
                    "project_id": _agent_run_project_id(run),
                    "workspace_dir": _agent_run_workspace_dir(run),
                    "task_workspace_dir": str(task_workspace_dir) if task_workspace_dir is not None else None,
                    "resume_strategy": active_resume_strategy,
                    "resume_payload": dict(active_resume_payload or {}),
                    "resume_runtime": dict(active_resume_runtime or {}),
                    "goal_handoff": (
                        dict(handoff_context.get("metadata") or {})
                        if isinstance(handoff_context, dict)
                        else {}
                    ),
                    "goal_worker_generation": (
                        _goal_worker_generation_response(
                            current_generation,
                            run_policy=effective_run_policy,
                        )
                        if isinstance(current_generation, dict)
                        else None
                    ),
                },
                runtime_event_callback=_persist_live_runtime_event,
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
            approval_mode_failure = await self._maybe_auto_reject_linked_goal_approval_wait(
                run_id=run_id,
                run=run,
                result=result,
                effective_run_policy=effective_run_policy,
            )
            if approval_mode_failure is not None:
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
                await self._finalize_goal_worker_generation(
                    current_generation_id,
                    status="failed",
                    latest_error=approval_mode_failure,
                )
                return
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
            await self._finalize_goal_worker_generation(
                current_generation_id,
                status=completion_status,
                latest_error=_agent_run_latest_error(result),
            )
        except asyncio.CancelledError:
            current_run = await self._store.get_agent_run(run_id)
            current_status = str(current_run.get("status") or "cancelled") if current_run else "cancelled"
            if current_status in {"created", "running"}:
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
                await self._finalize_goal_worker_generation(
                    current_generation_id,
                    status="cancelled",
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
                await self._finalize_goal_worker_generation(
                    current_generation_id,
                    status="awaiting_resources",
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
            await self._finalize_goal_worker_generation(
                current_generation_id,
                status="failed",
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
            await self._sync_goal_from_agent_run_by_run_id(run_id)

    async def _persist_agent_run_artifacts(
        self,
        run_id: str,
        result: Any,
        *,
        attempt_id: str | None,
    ) -> None:
        attempt_key = attempt_id or "latest"
        attempt_metadata = {"attempt_id": attempt_id} if attempt_id else {}
        existing_run = await self.get_agent_run(run_id)
        existing_artifacts = [
            dict(item)
            for item in (existing_run or {}).get("artifacts") or []
            if isinstance(item, dict)
        ]
        persisted_collector_record_keys = {
            (
                _string(record.get("attempt_id")) or attempt_id,
                collector_dataset_record_identity(record),
            )
            for record in extract_persisted_collector_dataset_records(
                existing_artifacts,
                attempt_id=attempt_id,
            )
        }
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
        has_collector_dataset_records = bool(collector_dataset_records_from_result(result.artifacts))
        for index, record in enumerate(dataset_records, start=1):
            collector_record_key = None
            if has_collector_dataset_records:
                collector_record_key = (
                    attempt_id,
                    collector_dataset_record_identity(
                        {
                            **record,
                            **({"attempt_id": attempt_id} if attempt_id else {}),
                        }
                    ),
                )
                if collector_record_key in persisted_collector_record_keys:
                    continue
            await self._store.append_agent_run_artifact(
                run_id,
                artifact_id=f"{run_id}:attempt:{attempt_key}:dataset:{index}",
                artifact_type="dataset_record",
                title=f"Dataset Record {index}",
                uri=f"agent-run://{run_id}/artifacts/{attempt_key}/dataset-record-{index}",
                mime_type="application/json",
                metadata={**attempt_metadata, "record": record},
            )
            if collector_record_key is not None:
                persisted_collector_record_keys.add(collector_record_key)
        collector_shard_manifests = collector_shard_manifests_from_result(
            result.artifacts,
            attempt_id=attempt_id,
        )
        collector_shard_manifests = dedupe_collector_shard_manifests(collector_shard_manifests)
        for index, shard_manifest in enumerate(collector_shard_manifests, start=1):
            shard_id = str(shard_manifest.get("shard_id") or f"shard-{index}")
            adapter_name = str(shard_manifest.get("adapter_name") or "collector")
            await self._store.append_agent_run_artifact(
                run_id,
                artifact_id=f"{run_id}:attempt:{attempt_key}:collector-shard:{index}",
                artifact_type=COLLECTOR_SHARD_MANIFEST_ARTIFACT_TYPE,
                title=f"Collector Shard {shard_id}",
                uri=f"agent-run://{run_id}/artifacts/{attempt_key}/collector-shard-{index}",
                mime_type="application/json",
                metadata={
                    **attempt_metadata,
                    "shard_id": shard_id,
                    "adapter_name": adapter_name,
                    "status": shard_manifest.get("status"),
                    "content": shard_manifest,
                },
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
            if event_type == "guidance":
                text = event.get("guidance")
                if isinstance(text, str) and text.strip():
                    messages.append(text.strip())
                    continue
                payload = event.get("payload")
                if isinstance(payload, dict):
                    payload_text = payload.get("content")
                    if isinstance(payload_text, str) and payload_text.strip():
                        messages.append(payload_text.strip())
                continue
            if event_type != "operator_message":
                continue
            text = event.get("content")
            if isinstance(text, str) and text.strip():
                messages.append(text.strip())
        return messages

    async def _collect_role_guidance_messages(self, run_id: str) -> dict[str, list[str]]:
        events = await self._store.get_agent_run_events(run_id)
        messages_by_role: dict[str, list[str]] = {}
        for event in events:
            if str(event.get("type", "")) != "subagent_message":
                continue
            role_id = str(event.get("target_role_id") or "").strip()
            content = event.get("content")
            if not role_id or not isinstance(content, str) or not content.strip():
                continue
            messages_by_role.setdefault(role_id, []).append(content.strip())
        return messages_by_role

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

    async def append_task_message(
        self,
        task_id: str,
        payload: TaskMessageRequest,
    ) -> dict[str, Any] | None:
        task = await self._store.get_task_run(task_id)
        if task is None:
            return None
        if str(task.get("task_type") or "").strip() != DELEGATED_MULTI_AGENT_TASK_TYPE:
            raise ValueError("Task messages are only supported for delegated_multi_agent tasks.")
        metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        delegated_subagent = _delegated_subagent_payload(
            metadata,
            task_type=task.get("task_type"),
            input_message=str(task.get("input") or ""),
            session_id=task.get("session_id"),
            status=task.get("status"),
        )
        if delegated_subagent is None:
            raise ValueError("Delegated subagent metadata is unavailable for this task.")
        content = payload.content.strip()
        await self._store.append_task_event(
            task_id,
            _delegated_subagent_message_event(
                delegated_subagent,
                task_id=task_id,
                content=content,
                metadata=payload.metadata,
            ),
        )
        return await self.get_task(task_id)

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
        decision: ApprovalDecision,
        reason: str | None = None,
        rule: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        task = await self._store.get_task_run(task_id)
        if task is None:
            return None

        approval = await self._store.get_pending_approval_for_task(task_id)
        if approval is None:
            return _task_summary(task)

        resolved = await self.resolve_approval(
            approval["id"],
            decision=decision,
            reason=reason,
            rule=rule,
        )
        if resolved is None:
            return None
        return _task_summary((await self._store.get_task_run(task_id)) or task)

    def _prepare_task_approval_request_metadata(
        self,
        *,
        task: Mapping[str, Any],
        request_event: Mapping[str, Any],
        approval_metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        metadata = dict(approval_metadata or {})
        tool_name = str(request_event.get("tool_name") or "").strip()
        if tool_name != "exec_command":
            return metadata
        linked_exec_approval_id = _linked_exec_approval_id(metadata)
        linked_exec = (
            self._exec_approval_store.get(linked_exec_approval_id)
            if isinstance(linked_exec_approval_id, str) and linked_exec_approval_id
            else None
        )
        approval_like = {
            "arguments": dict(request_event.get("arguments") or {}),
            "metadata": metadata,
        }
        return _task_exec_approval_metadata_from_request(
            approval_like,
            task=task,
            exec_approval=linked_exec,
        )

    async def _rehydrate_task_linked_exec_approval(
        self,
        approval: Mapping[str, Any],
    ) -> Any | None:
        metadata = approval.get("metadata") if isinstance(approval.get("metadata"), dict) else {}
        linked_exec_approval_id = _linked_exec_approval_id(metadata)
        if not isinstance(linked_exec_approval_id, str) or not linked_exec_approval_id:
            return None
        existing = self._exec_approval_store.get(linked_exec_approval_id)
        if existing is not None:
            return existing
        task_id = str(approval.get("task_id") or "").strip()
        task = await self._store.get_task_run(task_id) if task_id else None
        command_payload = _task_exec_command_payload_from_approval_request(
            approval,
            task=task,
        )
        if not isinstance(command_payload, dict) or not command_payload:
            return None
        command = str(command_payload.get("command") or "").strip()
        if not command:
            return None
        shell = str(command_payload.get("shell") or "auto").strip() or "auto"
        scope = (
            str(metadata.get("exec_scope") or metadata.get("approval_scope") or "").strip()
            or "dangerous_command"
        )
        reason = (
            str(metadata.get("exec_reason") or "").strip()
            or str(metadata.get("policy_reason") or "").strip()
            or "Exec command requires approval."
        )
        rehydrated_metadata = _rehydrate_exec_approval_metadata_from_task_approval(approval)
        if task_id:
            rehydrated_metadata["rehydrated_from_task_id"] = task_id
        approval_id = str(approval.get("id") or "").strip()
        if approval_id:
            rehydrated_metadata["rehydrated_from_approval_request_id"] = approval_id
        rehydrated = self._exec_approval_store.create(
            approval_id=linked_exec_approval_id,
            command=command,
            shell=shell,
            scope=scope,
            reason=reason,
            metadata=rehydrated_metadata,
            command_payload=command_payload,
        )
        decision = _approval_decision_from_status(str(approval.get("status") or "").strip())
        execution_result = (
            metadata.get("execution_result")
            if isinstance(metadata.get("execution_result"), dict)
            else None
        )
        if decision is None:
            return rehydrated
        resolved = self._exec_approval_store.resolve(
            linked_exec_approval_id,
            decision=decision,
            reason=str(approval.get("reason") or "").strip() or reason,
            execution_result=execution_result,
        )
        return resolved or rehydrated

    async def _persist_task_linked_exec_approval_metadata(
        self,
        approval: Mapping[str, Any],
        *,
        exec_approval: Any | None,
    ) -> None:
        if str(approval.get("tool_name") or "").strip() != "exec_command":
            return
        approval_id = str(approval.get("id") or "").strip()
        if not approval_id:
            return
        metadata = approval.get("metadata") if isinstance(approval.get("metadata"), dict) else {}
        linked_exec_approval_id = _linked_exec_approval_id(metadata)
        if not linked_exec_approval_id:
            return
        task_id = str(approval.get("task_id") or "").strip()
        task = await self._store.get_task_run(task_id) if task_id else None
        next_metadata = _task_exec_approval_metadata_from_request(
            approval,
            task=task,
            exec_approval=exec_approval,
        )
        await self._store.update_approval_request_metadata(
            approval_id,
            metadata=next_metadata,
        )

    async def list_approvals(self, *, status: str | None = None) -> list[dict[str, Any]]:
        approvals = await self._store.list_approval_requests(status=status)
        for approval in approvals:
            await self._rehydrate_task_linked_exec_approval(approval)
        linked_exec_approval_ids = {
            linked_id
            for linked_id in (_linked_exec_approval_id(item.get("metadata")) for item in approvals)
            if linked_id
        }

        summaries = [self._enrich_approval_summary(_approval_summary(item)) for item in approvals]
        exec_status = _to_exec_approval_status(status)
        if status is None or exec_status is not None:
            rehydrated_ids: set[str] = set()
            if exec_status in {None, "pending"}:
                pending_linked_exec = await self._list_rehydrated_linked_exec_approvals()
                for exec_approval in pending_linked_exec:
                    approval_id = exec_approval.approval_id
                    if approval_id in linked_exec_approval_ids or approval_id in rehydrated_ids:
                        continue
                    summaries.append(_exec_approval_summary(exec_approval))
                    rehydrated_ids.add(approval_id)
            for exec_approval in self._exec_approval_store.list(status=exec_status):
                if (
                    exec_approval.approval_id in linked_exec_approval_ids
                    or exec_approval.approval_id in rehydrated_ids
                ):
                    continue
                summaries.append(_exec_approval_summary(exec_approval))

        summaries.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
        return summaries

    async def resolve_approval(
        self,
        approval_id: str,
        *,
        decision: ApprovalDecision,
        reason: str | None = None,
        rule: dict[str, Any] | None = None,
        replay_override: dict[str, Any] | None = None,
        auto_resume_linked_run: bool = True,
    ) -> dict[str, Any] | None:
        current = await self._store.get_approval_request(approval_id)
        if current is None:
            linked_run_context = await self._find_linked_agent_run_for_exec_approval(approval_id)
            existing_exec = self._exec_approval_store.get(approval_id)
            if existing_exec is None and linked_run_context is not None:
                linked_run, pending_entry = linked_run_context
                existing_exec = self._rehydrate_linked_exec_approval(
                    run=linked_run,
                    approval_id=approval_id,
                    pending_entry=pending_entry,
                )
            if existing_exec is None:
                return None
            if replay_override is not None:
                raise ValueError("replay_override is only supported for task file-mutation approvals.")
            effective_decision, saved_rule = await self._resolve_approval_decision_and_saved_rule(
                decision=decision,
                rule=rule,
                suggested_rules=[existing_exec.metadata.get("suggested_rule")],
            )
            execution_result: dict[str, Any] | None = None
            if effective_decision != "reject":
                execution_result = await self._execute_approved_exec_request(existing_exec)
            resolved_exec = self._exec_approval_store.resolve(
                approval_id,
                decision=effective_decision,
                reason=reason,
                metadata={"saved_rule": saved_rule} if isinstance(saved_rule, dict) else None,
                execution_result=execution_result,
            )
            if resolved_exec is None:
                return None
            if linked_run_context is not None:
                linked_run, pending_entry = linked_run_context
                await self._append_goal_operator_approval_audit_log(
                    subject_id=approval_id,
                    decision=effective_decision,
                    reason=reason,
                    run=linked_run,
                    source="linked_exec_approval_rehydrated",
                    linked_exec_approval_id=approval_id,
                    auto_resume_linked_run=auto_resume_linked_run,
                )
                if effective_decision == "reject":
                    await self._fail_linked_agent_run_for_rejected_exec_approval(
                        run=linked_run,
                        approval_id=approval_id,
                        reason=reason,
                    )
                else:
                    await self._resume_linked_agent_run_after_exec_approval(
                        run=linked_run,
                        approval_id=approval_id,
                        approval=resolved_exec,
                        pending_entry=pending_entry,
                        execution_result=execution_result,
                        resume_strategy=(
                            "continue_from_checkpoint" if auto_resume_linked_run else None
                        ),
                    )
            return _exec_approval_summary(resolved_exec)
        linked_exec_approval_id = _linked_exec_approval_id(current.get("metadata"))
        linked_exec = (
            self._exec_approval_store.get(linked_exec_approval_id)
            if isinstance(linked_exec_approval_id, str) and linked_exec_approval_id
            else None
        )
        if linked_exec is None and isinstance(linked_exec_approval_id, str) and linked_exec_approval_id:
            linked_exec = await self._rehydrate_task_linked_exec_approval(current)
        linked_goal_audit_context = (
            await self._find_linked_agent_run_for_exec_approval(linked_exec_approval_id)
            if isinstance(linked_exec_approval_id, str) and linked_exec_approval_id
            else None
        )
        if _is_file_mutation_approval(current):
            effective_decision = "approve_once" if decision == "approve_and_save_rule" else decision
            saved_rule = None
        else:
            effective_decision, saved_rule = await self._resolve_approval_decision_and_saved_rule(
                decision=decision,
                rule=rule,
                suggested_rules=[
                    (current.get("metadata") or {}).get("suggested_rule")
                    if isinstance(current.get("metadata"), dict)
                    else None,
                    linked_exec.metadata.get("suggested_rule") if linked_exec is not None else None,
                ],
            )
        approval = await self._store.resolve_approval_request(
            approval_id,
            decision=effective_decision,
            reason=reason,
        )
        if approval is None:
            return None
        if linked_exec_approval_id:
            linked_exec = self._exec_approval_store.resolve(
                linked_exec_approval_id,
                decision=effective_decision,
                reason=reason,
                metadata={"saved_rule": saved_rule} if isinstance(saved_rule, dict) else None,
            )
            if linked_exec is not None:
                await self._persist_task_linked_exec_approval_metadata(
                    current,
                    exec_approval=linked_exec,
                )
        if linked_goal_audit_context is not None:
            linked_run, _pending_entry = linked_goal_audit_context
            await self._append_goal_operator_approval_audit_log(
                subject_id=approval_id,
                decision=effective_decision,
                reason=reason,
                run=linked_run,
                source="approval_request_resolution",
                task_approval_id=approval_id,
                linked_exec_approval_id=linked_exec_approval_id,
                auto_resume_linked_run=auto_resume_linked_run,
            )
        if effective_decision == "reject":
            await self._store.update_task_status(
                approval["task_id"],
                "cancelled",
                error="Approval rejected",
            )
            return self._task_approval_summary(approval)
        if await self._try_complete_task_with_file_mutation(
            approval,
            current=current,
            replay_override=replay_override,
        ):
            return self._task_approval_summary(approval)
        if await self._try_complete_task_with_linked_exec(
            approval,
            current=current,
            linked_exec_approval_id=linked_exec_approval_id,
            reason=reason,
            decision=effective_decision,
            saved_rule=saved_rule,
        ):
            return self._task_approval_summary(approval)
        override = _approval_override(current, replay_override=replay_override)
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

    async def _append_goal_operator_approval_audit_log(
        self,
        *,
        subject_id: str,
        decision: str,
        reason: str | None,
        run: Mapping[str, Any],
        source: str,
        task_approval_id: str | None = None,
        linked_exec_approval_id: str | None = None,
        auto_resume_linked_run: bool | None = None,
    ) -> None:
        goal_id, attempt_id = _linked_goal_context_from_agent_run(run)
        if goal_id is None:
            return
        run_id = str(run.get("id") or "").strip() or None
        await self._store.append_goal_operator_audit_log(
            event_type="approval_resolution",
            subject_type="approval",
            subject_id=subject_id,
            action=decision,
            summary=f"Resolved linked goal approval {subject_id} with decision {decision}.",
            details={
                "goal_id": goal_id,
                "goal_attempt_id": attempt_id,
                "agent_run_id": run_id,
                "decision": decision,
                "reason": reason,
                "source": source,
                "task_approval_id": task_approval_id,
                "linked_exec_approval_id": linked_exec_approval_id,
                "auto_resume_linked_run": auto_resume_linked_run,
            },
        )

    async def _find_linked_agent_run_for_exec_approval(
        self,
        approval_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any] | None] | None:
        runs = await self._store.list_agent_runs()
        fallback: tuple[dict[str, Any], dict[str, Any] | None] | None = None
        for run in runs:
            pending_entry = _find_agent_run_pending_approval_entry(run, approval_id)
            if pending_entry is None:
                continue
            run_status = str(run.get("status") or "").strip()
            if run_status == "awaiting_approval":
                return run, pending_entry
            if run_status not in AGENT_RUN_TERMINAL_STATUS and fallback is None:
                fallback = (run, pending_entry)
        return fallback

    async def _resume_linked_agent_run_after_exec_approval(
        self,
        *,
        run: dict[str, Any],
        approval_id: str,
        approval: Any,
        pending_entry: dict[str, Any] | None,
        execution_result: dict[str, Any] | None,
        resume_strategy: str | None = "continue_from_checkpoint",
    ) -> None:
        updated_summary = _apply_exec_approval_resolution_to_agent_run_summary(
            run,
            approval_id=approval_id,
            decision="approve_once",
            reason=getattr(approval, "reason", None),
            approval=approval,
            pending_entry=pending_entry,
            execution_result=execution_result,
        )
        await self._store.update_agent_run_metadata(
            str(run["id"]),
            summary=updated_summary,
        )
        if not isinstance(resume_strategy, str) or not resume_strategy.strip():
            return
        resumed = await self.resume_agent_run(
            str(run["id"]),
            strategy=resume_strategy,
        )
        if resumed is not None:
            await self._sync_goal_from_agent_run_by_run_id(str(run["id"]))

    async def _fail_linked_agent_run_for_rejected_exec_approval(
        self,
        *,
        run: dict[str, Any],
        approval_id: str,
        reason: str | None,
    ) -> None:
        error_message = reason.strip() if isinstance(reason, str) and reason.strip() else "Approval rejected"
        updated_summary = _apply_exec_approval_resolution_to_agent_run_summary(
            run,
            approval_id=approval_id,
            decision="reject",
            reason=error_message,
            approval=None,
            pending_entry=_find_agent_run_pending_approval_entry(run, approval_id),
            execution_result=None,
        )
        await self._store.update_agent_run_metadata(
            str(run["id"]),
            summary=updated_summary,
            latest_error=error_message,
        )
        await self._store.append_agent_run_event(
            str(run["id"]),
            {
                "type": "run_error",
                "error": error_message,
                "metadata": {
                    "approval_id": approval_id,
                    "decision": "reject",
                    "source": "exec_approval_resolution",
                },
            },
        )
        await self._store.update_agent_run_status(
            str(run["id"]),
            "failed",
            latest_error=error_message,
        )
        await self._sync_goal_from_agent_run_by_run_id(str(run["id"]))

    async def get_approval_exec_session(
        self,
        approval_id: str,
        *,
        yield_time_ms: int | None = None,
    ) -> dict[str, Any] | tuple[str, None]:
        context = await self._get_approval_exec_session_context(approval_id)
        if isinstance(context, tuple):
            return context
        session_id = context.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            return ("session_unavailable", None)
        poll = await self._exec_runtime.read_session(session_id, yield_time_ms=yield_time_ms)
        if poll is None:
            return ("session_unavailable", None)
        return _build_approval_exec_session_payload(context=context, poll=poll)

    async def stop_approval_exec_session(self, approval_id: str) -> dict[str, Any] | tuple[str, None]:
        context = await self._get_approval_exec_session_context(approval_id)
        if isinstance(context, tuple):
            return context
        session_id = context.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            return ("session_unavailable", None)
        poll = await self._exec_runtime.kill_session(session_id)
        if poll is None:
            return ("session_unavailable", None)
        self._persist_exec_approval_poll_result(
            exec_approval_id=str(context.get("exec_approval_id") or ""),
            poll=poll,
        )
        current = await self._store.get_approval_request(approval_id)
        if isinstance(current, dict):
            exec_approval = self._exec_approval_store.get(str(context.get("exec_approval_id") or ""))
            if exec_approval is not None:
                await self._persist_task_linked_exec_approval_metadata(
                    current,
                    exec_approval=exec_approval,
                )
        payload = _build_approval_exec_session_payload(context=context, poll=poll)
        payload["stop_status"] = poll.status.value
        return payload

    async def _execute_approved_file_mutation(
        self,
        *,
        approval: dict[str, Any],
        current: dict[str, Any],
        replay_override: dict[str, Any] | None,
    ) -> tuple[str, ToolResult] | tuple[None, None]:
        task_id = approval.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            return (None, None)
        task = await self._store.get_task_run(task_id)
        if task is None:
            return (
                str(current.get("tool_name") or approval.get("tool_name") or ""),
                ToolResult(error="Associated task not found for file-mutation approval."),
            )

        approved_call = _approved_tool_call(current, replay_override=replay_override)
        tool_name = str(approved_call.get("tool_name") or "")
        tool = self._build_file_mutation_tool(tool_name=tool_name, task=task)
        if tool is None:
            return (tool_name, ToolResult(error=f"Unsupported file-mutation tool: {tool_name}"))

        context = ToolExecutionContext(
            workspace_dir=(
                str(task.get("workspace_dir"))
                if isinstance(task.get("workspace_dir"), str)
                else None
            ),
            session_id=str(task.get("session_id")) if isinstance(task.get("session_id"), str) else None,
            project_workspace=(
                str(task.get("project_workspace_dir"))
                if isinstance(task.get("project_workspace_dir"), str)
                else None
            ),
            task_sandbox_dir=(
                str(task.get("task_workspace_dir"))
                if isinstance(task.get("task_workspace_dir"), str)
                else None
            ),
        )
        context.state["approval_replay"] = True
        arguments = dict(approved_call.get("arguments") or {})
        result = await tool.execute(**arguments, approved=True, context=context)
        return tool_name, result

    def _build_file_mutation_tool(
        self,
        *,
        tool_name: str,
        task: Mapping[str, Any],
    ) -> FileWriteTool | FileEditTool | ApplyPatchTool | None:
        runtime_policy = resolve_runtime_permission_policy(self._security_config)
        workspace_dir = (
            str(task.get("task_workspace_dir"))
            if isinstance(task.get("task_workspace_dir"), str)
            else str(task.get("project_workspace_dir"))
            if isinstance(task.get("project_workspace_dir"), str)
            else str(task.get("workspace_dir"))
            if isinstance(task.get("workspace_dir"), str)
            else "."
        )
        common: dict[str, Any] = {
            "workspace_dir": workspace_dir,
            "path_scope": runtime_policy.file_ops_scope,
            "require_approval": False,
            "max_write_size_mb": self._security_config.max_file_write_size_mb,
            "undo_max_size_mb": self._security_config.file_undo_max_size_mb,
        }
        if tool_name == "file_write":
            return FileWriteTool(**common)
        if tool_name == "file_edit":
            return FileEditTool(**common)
        if tool_name == "apply_patch":
            return ApplyPatchTool(**common)
        return None

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
        return self._enrich_approval_summary(summary)

    async def _get_approval_exec_session_context(
        self,
        approval_id: str,
    ) -> dict[str, Any] | tuple[str, None]:
        current = await self._store.get_approval_request(approval_id)
        if current is not None:
            if str(current.get("tool_name") or "") != "exec_command":
                return ("approval_not_found", None)
            await self._rehydrate_task_linked_exec_approval(current)
            summary = self._enrich_approval_summary(_approval_summary(current))
            exec_approval_id = summary.get("exec_approval_id")
            if not isinstance(exec_approval_id, str) or not exec_approval_id:
                return ("approval_not_found", None)
            return {
                "approval_id": approval_id,
                "source": summary.get("source"),
                "exec_approval_id": exec_approval_id,
                "session_id": summary.get("exec_session_id"),
                "status": summary.get("exec_status"),
                "task_id": summary.get("task_id"),
                "tool_name": summary.get("tool_name"),
            }

        standalone = self._exec_approval_store.get(approval_id)
        if standalone is None:
            linked_run_context = await self._find_linked_agent_run_for_exec_approval(approval_id)
            if linked_run_context is not None:
                linked_run, pending_entry = linked_run_context
                standalone = self._rehydrate_linked_exec_approval(
                    run=linked_run,
                    approval_id=approval_id,
                    pending_entry=pending_entry,
                )
        if standalone is None:
            return ("approval_not_found", None)
        summary = _exec_approval_summary(standalone)
        return {
            "approval_id": approval_id,
            "source": summary.get("source"),
            "exec_approval_id": summary.get("exec_approval_id"),
            "session_id": summary.get("exec_session_id"),
            "status": summary.get("exec_status"),
            "task_id": summary.get("task_id"),
            "tool_name": summary.get("tool_name"),
        }

    def _enrich_approval_summary(self, summary: dict[str, Any]) -> dict[str, Any]:
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

    async def _list_rehydrated_linked_exec_approvals(self) -> list[Any]:
        runs = await self._store.list_agent_runs()
        approvals: list[Any] = []
        seen_ids: set[str] = set()
        for run in runs:
            run_status = str(run.get("status") or "").strip()
            if run_status in AGENT_RUN_TERMINAL_STATUS:
                continue
            for pending_entry in _collect_agent_run_pending_approval_entries(run):
                approval_id = str(pending_entry.get("approval_id") or "").strip()
                if not approval_id or approval_id in seen_ids:
                    continue
                rehydrated = self._rehydrate_linked_exec_approval(
                    run=run,
                    approval_id=approval_id,
                    pending_entry=pending_entry,
                )
                if rehydrated is None:
                    continue
                approvals.append(rehydrated)
                seen_ids.add(approval_id)
        return approvals

    def _rehydrate_linked_exec_approval(
        self,
        *,
        run: Mapping[str, Any],
        approval_id: str,
        pending_entry: Mapping[str, Any] | None,
    ) -> Any | None:
        existing = self._exec_approval_store.get(approval_id)
        if existing is not None:
            return existing
        command_payload = _rehydrate_exec_command_payload_from_agent_run(
            run,
            pending_entry=pending_entry,
        )
        command = (
            str(command_payload.get("command") or "").strip()
            if isinstance(command_payload, dict)
            else ""
        )
        if not command:
            return None
        shell = (
            str(command_payload.get("shell") or "").strip()
            if isinstance(command_payload, dict)
            else ""
        ) or "auto"
        pending_metadata = _pending_entry_metadata(pending_entry)
        scope = (
            str(pending_metadata.get("approval_scope") or pending_metadata.get("scope") or "").strip()
            or (
                str(pending_entry.get("approval_scope") or "").strip()
                if isinstance(pending_entry, Mapping)
                else ""
            )
            or "dangerous_command"
        )
        reason = (
            str(pending_entry.get("reason") or "").strip()
            if isinstance(pending_entry, Mapping) and str(pending_entry.get("reason") or "").strip()
            else str(pending_metadata.get("policy_reason") or "").strip()
            if str(pending_metadata.get("policy_reason") or "").strip()
            else "Exec command requires approval."
        )
        metadata = _rehydrate_exec_approval_metadata_from_pending_entry(pending_entry)
        metadata["rehydrated_from_agent_run_id"] = str(run.get("id") or "")
        return self._exec_approval_store.create(
            approval_id=approval_id,
            command=command,
            shell=shell,
            scope=scope,
            reason=reason,
            metadata=metadata,
            command_payload=command_payload,
        )

    async def _resolve_approval_decision_and_saved_rule(
        self,
        *,
        decision: ApprovalDecision,
        rule: dict[str, Any] | None,
        suggested_rules: list[Any] | None = None,
    ) -> tuple[ApprovalDecision, dict[str, Any] | None]:
        if decision != "approve_and_save_rule":
            return decision, None
        candidates: list[dict[str, Any] | None] = [rule]
        for candidate in suggested_rules or []:
            candidates.append(candidate if isinstance(candidate, dict) else None)
        for candidate in candidates:
            saved_rule = await self._persist_command_rule(candidate)
            if saved_rule is not None:
                return "approve_and_save_rule", saved_rule
        return "approve_once", None

    async def _maybe_auto_reject_linked_goal_approval_wait(
        self,
        *,
        run_id: str,
        run: Mapping[str, Any],
        result: MultiAgentRunResult,
        effective_run_policy: Mapping[str, Any] | None,
    ) -> str | None:
        if _resolve_agent_run_completion_status(result) != "awaiting_approval":
            return None
        summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
        goal_id = str(summary.get("goal_id") or "").strip()
        if not goal_id:
            return None
        approval_mode = _normalized_goal_approval_mode(
            effective_run_policy.get("approval_mode")
            if isinstance(effective_run_policy, Mapping)
            else None
        )
        if not _goal_approval_mode_auto_rejects(approval_mode):
            return None

        current_run = await self._store.get_agent_run(run_id)
        if current_run is None:
            return None
        pending_entries = _collect_agent_run_pending_approval_entries(current_run)
        approval_ids = [
            str(item.get("approval_id") or "").strip()
            for item in pending_entries
            if isinstance(item, Mapping) and str(item.get("approval_id") or "").strip()
        ]
        error_message = (
            f"Goal approval_mode={approval_mode} rejected approval-required execution "
            "for unattended execution."
        )
        updated_run = dict(current_run)
        updated_summary = dict(updated_run.get("summary") or {})
        if pending_entries:
            for pending_entry in pending_entries:
                approval_id = str(pending_entry.get("approval_id") or "").strip()
                if not approval_id:
                    continue
                existing_exec = self._exec_approval_store.get(approval_id)
                if existing_exec is not None:
                    self._exec_approval_store.resolve(
                        approval_id,
                        decision="reject",
                        reason=error_message,
                    )
                updated_summary = _apply_exec_approval_resolution_to_agent_run_summary(
                    {**updated_run, "summary": updated_summary},
                    approval_id=approval_id,
                    decision="reject",
                    reason=error_message,
                    approval=None,
                    pending_entry=pending_entry,
                    execution_result=None,
                )
                updated_run["summary"] = updated_summary
                await self._append_goal_operator_approval_audit_log(
                    subject_id=approval_id,
                    decision="reject",
                    reason=error_message,
                    run=current_run,
                    source="goal_approval_mode_auto_reject",
                    linked_exec_approval_id=approval_id,
                    auto_resume_linked_run=False,
                )
        else:
            recovery_state = (
                dict(updated_summary.get("recovery_state"))
                if isinstance(updated_summary.get("recovery_state"), dict)
                else {}
            )
            updated_summary.pop("approval_state", None)
            recovery_state.pop("approval_state", None)
            recovery_state["status"] = "failed"
            recovery_state["action"] = "stop"
            recovery_state["reason"] = error_message
            recovery_state.pop("unfinished_steps", None)
            updated_summary["recovery_state"] = recovery_state

        await self._store.update_agent_run_metadata(
            run_id,
            summary=updated_summary,
            latest_error=error_message,
        )
        await self._store.append_agent_run_event(
            run_id,
            {
                "type": "run_error",
                "error": error_message,
                "metadata": {
                    "source": "goal_approval_mode",
                    "decision": "reject",
                    "approval_mode": approval_mode,
                    "approval_ids": approval_ids,
                },
            },
        )
        await self._store.update_agent_run_status(
            run_id,
            "failed",
            latest_error=error_message,
        )
        await self._sync_goal_from_agent_run_by_run_id(run_id)
        return error_message

    def _persist_exec_approval_poll_result(self, *, exec_approval_id: str, poll: Any) -> None:
        if not exec_approval_id:
            return
        existing = self._exec_approval_store.get(exec_approval_id)
        if existing is None:
            return
        decision = _approval_decision_from_status(existing.status)
        if decision is None:
            return
        execution_result = _poll_to_execution_result(
            poll,
            previous_result=existing.execution_result,
        )
        self._exec_approval_store.resolve(
            exec_approval_id,
            decision=decision,
            reason=existing.reason,
            execution_result=execution_result,
        )

    async def _try_complete_task_with_linked_exec(
        self,
        approval: dict[str, Any],
        *,
        current: dict[str, Any],
        linked_exec_approval_id: str | None,
        reason: str | None,
        decision: ApprovalDecision,
        saved_rule: dict[str, Any] | None,
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
            decision=decision,
            reason=reason,
            metadata={"saved_rule": saved_rule} if isinstance(saved_rule, dict) else None,
            execution_result=execution_result,
        )
        updated_linked = self._exec_approval_store.get(linked_exec_approval_id)
        if updated_linked is not None:
            await self._persist_task_linked_exec_approval_metadata(
                approval,
                exec_approval=updated_linked,
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

    async def _try_complete_task_with_file_mutation(
        self,
        approval: dict[str, Any],
        *,
        current: dict[str, Any],
        replay_override: dict[str, Any] | None,
    ) -> bool:
        if not _is_file_mutation_approval(current):
            return False

        tool_name, result = await self._execute_approved_file_mutation(
            approval=approval,
            current=current,
            replay_override=replay_override,
        )
        if tool_name is None or result is None:
            await self._store.update_task_status(
                approval["task_id"],
                "failed",
                error="File-mutation approval could not be replayed.",
            )
            return True

        await self._append_task_file_mutation_result_event(
            approval=approval,
            tool_name=tool_name,
            result=result,
        )
        final_answer = _final_answer_from_file_mutation_result(result)
        if result.error is None:
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

        await self._store.append_task_event(
            approval["task_id"],
            {
                "type": "error",
                "error": result.error,
                "code": "FILE_APPROVAL_EXECUTION_FAILED",
                "metadata": {
                    "approval_id": approval.get("id"),
                    "tool_name": tool_name,
                    "result_metadata": result.metadata,
                },
            },
        )
        await self._store.update_task_status(
            approval["task_id"],
            "failed",
            error=result.error,
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

    async def _append_task_file_mutation_result_event(
        self,
        *,
        approval: dict[str, Any],
        tool_name: str,
        result: ToolResult,
    ) -> None:
        metadata = dict(result.metadata)
        metadata["approval_id"] = approval.get("id")
        await self._store.append_task_event(
            approval["task_id"],
            {
                "type": "tool_call_result",
                "call_id": str(approval.get("call_id") or ""),
                "tool_name": tool_name,
                "result": result.output,
                "error": result.error,
                "metadata": metadata,
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
                approval_metadata = self._prepare_task_approval_request_metadata(
                    task=task,
                    request_event=request_event,
                    approval_metadata=approval_metadata,
                )
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
        security_config = self._security_config
        orchestrator = MultiAgentOrchestrator(
            engine=self._engine,
            exec_runtime=self._exec_runtime,
            exec_approval_store=self._exec_approval_store,
            controlled_exec_require_approval=resolve_runtime_permission_policy(
                security_config
            ).require_approval_for_exec,
            controlled_exec_command_rules=[
                rule.model_dump() for rule in security_config.command_rules
            ],
            controlled_exec_allowed_env_vars=security_config.exec_allowed_env_vars,
            controlled_exec_default_shell=security_config.exec_default_shell,
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
    metadata = _normalize_task_metadata(
        task.get("metadata"),
        task_type=task.get("task_type"),
        input_message=str(task.get("input") or ""),
        session_id=task.get("session_id"),
        status=task.get("status"),
    )
    delegated_subagent = _delegated_subagent_payload(
        metadata,
        task_type=task.get("task_type"),
        input_message=str(task.get("input") or ""),
        session_id=task.get("session_id"),
        status=task.get("status"),
    )
    return {
        "task_id": task["id"],
        "status": task["status"],
        "input_message": task["input"],
        "session_id": task.get("session_id"),
        "project_id": task.get("project_id"),
        "project_workspace_dir": task.get("project_workspace_dir") or task.get("workspace_dir"),
        "task_workspace_dir": task.get("task_workspace_dir"),
        "task_type": task.get("task_type"),
        "metadata": metadata,
        "display_name": delegated_subagent.get("display_name") if delegated_subagent else None,
        "role": delegated_subagent.get("role") if delegated_subagent else None,
        "instruction": delegated_subagent.get("instruction") if delegated_subagent else None,
        "objective": delegated_subagent.get("objective") if delegated_subagent else None,
        "parent_session_id": (
            delegated_subagent.get("parent_session_id")
            if delegated_subagent
            else task.get("session_id")
        ),
        "delegated_subagent": delegated_subagent,
        "final_answer": task.get("final_answer"),
        "error": task.get("error"),
        "started_at": task.get("started_at"),
        "finished_at": task.get("finished_at"),
        "created_at": task["created_at"],
        "updated_at": task["updated_at"],
        "pending_approval_id": pending_approval.get("id") if pending_approval else None,
    }


def _goal_response(goal: dict[str, Any]) -> dict[str, Any]:
    normalized_run_policy = _normalize_goal_run_policy(
        goal.get("run_policy"),
        objective=goal.get("objective"),
    )
    normalized_capability_policy = _normalize_goal_capability_policy(
        goal.get("capability_policy"),
    )
    return {
        "goal_id": goal["id"],
        "objective": goal["objective"],
        "title": goal.get("title"),
        "goal_type": goal.get("goal_type"),
        "execution_mode": _normalize_goal_execution_mode(goal.get("execution_mode")),
        "interaction_mode": _goal_effective_interaction_mode(goal),
        "execution_topology": _goal_effective_execution_topology(goal),
        "protocol_id": _goal_effective_protocol_id(goal),
        "bound_run_id": _goal_bound_run_id(goal),
        "protocol_selection": _goal_protocol_selection(goal),
        "selection_rationale": _goal_selection_rationale(goal),
        "topic": goal.get("topic"),
        "project_id": goal.get("project_id"),
        "workspace_dir": goal.get("workspace_dir"),
        "status": goal.get("status", "created"),
        "current_attempt_id": goal.get("current_attempt_id"),
        "run_policy": normalized_run_policy,
        "capability_policy": normalized_capability_policy,
        "source_manifest": goal.get("source_manifest") or {},
        "summary": goal.get("summary") or {},
        "metadata": goal.get("metadata") or {},
        "latest_error": goal.get("latest_error"),
        "attempts": [
            _goal_attempt_response(attempt)
            for attempt in goal.get("attempts") or []
            if isinstance(attempt, dict)
        ],
        "created_at": goal["created_at"],
        "updated_at": goal["updated_at"],
        "started_at": goal.get("started_at"),
        "finished_at": goal.get("finished_at"),
    }


def _goal_attempt_response(attempt: dict[str, Any]) -> dict[str, Any]:
    return {
        "attempt_id": attempt["id"],
        "goal_id": attempt["goal_id"],
        "attempt_index": int(attempt.get("attempt_index") or 0),
        "status": attempt.get("status", "queued"),
        "trigger": attempt.get("trigger"),
        "agent_run_id": attempt.get("agent_run_id"),
        "summary": attempt.get("summary") or {},
        "metadata": attempt.get("metadata") or {},
        "latest_error": attempt.get("latest_error"),
        "created_at": attempt["created_at"],
        "updated_at": attempt["updated_at"],
        "started_at": attempt.get("started_at"),
        "finished_at": attempt.get("finished_at"),
    }


def _normalize_goal_execution_mode(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in GOAL_EXECUTION_MODES:
        return normalized
    return DEFAULT_GOAL_EXECUTION_MODE


def _normalize_goal_interaction_mode(value: Any, *, execution_mode: Any = None) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in GOAL_INTERACTION_MODES:
        return normalized
    if _normalize_goal_execution_mode(execution_mode) == "single_agent":
        return "goal"
    return DEFAULT_GOAL_INTERACTION_MODE


def _normalize_goal_execution_topology(
    value: Any,
    *,
    execution_mode: Any = None,
    protocol_id: Any = None,
) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in GOAL_EXECUTION_TOPOLOGIES:
        return normalized
    requested_protocol = str(protocol_id or "").strip()
    if requested_protocol == AUTONOMOUS_SINGLE_AGENT_PROTOCOL:
        return "single_agent"
    if requested_protocol:
        return "multi_agent"
    if _normalize_goal_execution_mode(execution_mode) == "single_agent":
        return "single_agent"
    return "multi_agent"


def _goal_summary_payload(goal: Mapping[str, Any]) -> dict[str, Any]:
    summary = goal.get("summary")
    return dict(summary) if isinstance(summary, Mapping) else {}


def _goal_metadata_payload(goal: Mapping[str, Any]) -> dict[str, Any]:
    metadata = goal.get("metadata")
    return dict(metadata) if isinstance(metadata, Mapping) else {}


def _goal_summary_text(goal: Mapping[str, Any], key: str) -> str | None:
    summary = _goal_summary_payload(goal)
    metadata = _goal_metadata_payload(goal)
    for source in (summary, metadata):
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _goal_requested_protocol_id(goal: Mapping[str, Any]) -> str | None:
    normalized = str(goal.get("protocol_id") or "").strip()
    return normalized or None


def _goal_has_explicit_distill_intent(goal: Mapping[str, Any]) -> bool:
    explicit_selection = _goal_summary_text(goal, "protocol_selection")
    if explicit_selection == "teacher_student_distill":
        return True
    selection_rationale = _goal_summary_text(goal, "selection_rationale")
    if isinstance(selection_rationale, str) and "distill" in selection_rationale.lower():
        return True
    return False


def _goal_effective_interaction_mode(goal: Mapping[str, Any]) -> str:
    return _normalize_goal_interaction_mode(
        _goal_summary_text(goal, "interaction_mode"),
        execution_mode=goal.get("execution_mode"),
    )


def _goal_effective_protocol_id(goal: Mapping[str, Any]) -> str | None:
    execution_mode = _normalize_goal_execution_mode(goal.get("execution_mode"))
    requested_protocol = _goal_requested_protocol_id(goal)
    if requested_protocol is not None:
        if (
            requested_protocol == "teacher_student_distill"
            and execution_mode == "single_agent"
            and not _goal_has_explicit_distill_intent(goal)
        ):
            return AUTONOMOUS_SINGLE_AGENT_PROTOCOL
        return requested_protocol
    if execution_mode == "single_agent":
        return AUTONOMOUS_SINGLE_AGENT_PROTOCOL
    return "teacher_student_distill"


def _goal_effective_execution_topology(goal: Mapping[str, Any]) -> str:
    return _normalize_goal_execution_topology(
        _goal_summary_text(goal, "execution_topology"),
        execution_mode=goal.get("execution_mode"),
        protocol_id=_goal_effective_protocol_id(goal),
    )


def _goal_bound_run_id(goal: Mapping[str, Any]) -> str | None:
    attempt = _current_goal_attempt(dict(goal))
    if not isinstance(attempt, Mapping):
        return None
    agent_run_id = str(attempt.get("agent_run_id") or "").strip()
    return agent_run_id or None


def _goal_protocol_selection(goal: Mapping[str, Any]) -> str | None:
    return _goal_summary_text(goal, "protocol_selection") or _goal_effective_protocol_id(goal)


def _goal_selection_rationale(goal: Mapping[str, Any]) -> str | None:
    return _goal_summary_text(goal, "selection_rationale")


def _goal_recovery_state_from_agent_run(run: dict[str, Any]) -> dict[str, Any]:
    recovery_state = _recovery_state_from_run(run)
    if not recovery_state:
        return {}
    checkpoint = recovery_state.get("checkpoint") if isinstance(recovery_state.get("checkpoint"), dict) else {}
    recovery_status = str(recovery_state.get("status") or run.get("status") or "created")
    run_summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
    resume_payload = _resume_payload_from_run(run) or {}
    selected_candidate_id = (
        str(run_summary.get("selected_candidate_id") or "").strip()
        if isinstance(run_summary.get("selected_candidate_id"), str)
        and str(run_summary.get("selected_candidate_id") or "").strip()
        else str(resume_payload.get("selected_candidate_id") or "").strip()
        if isinstance(resume_payload.get("selected_candidate_id"), str)
        and str(resume_payload.get("selected_candidate_id") or "").strip()
        else None
    )
    candidate_count = 0
    if isinstance(run_summary.get("candidate_count"), int):
        candidate_count = max(0, int(run_summary.get("candidate_count") or 0))
    elif isinstance(resume_payload.get("candidates"), list):
        candidate_count = len([item for item in resume_payload.get("candidates") or [] if isinstance(item, dict)])
    role_task_summary = _summarize_role_task_snapshot(_agent_run_role_task_snapshot(run))
    payload = {
        "status": recovery_state.get("status"),
        "action": recovery_state.get("action"),
        "reason": recovery_state.get("reason"),
        "stage": recovery_state.get("stage"),
        "checkpoint_index": checkpoint.get("checkpoint_index"),
        "checkpoint": dict(checkpoint) if checkpoint else {},
        "unfinished_steps": list(recovery_state.get("unfinished_steps") or []),
        "recommended_resume_conditions": list(
            recovery_state.get("recommended_resume_conditions") or []
        ),
        "finalize_partial_ready": recovery_status in AGENT_RUN_OPERATOR_FINALIZEABLE_STATUS,
        "selected_candidate_id": selected_candidate_id,
        "candidate_count": candidate_count if candidate_count > 0 else None,
        "role_task_summary": role_task_summary if role_task_summary.get("available") else None,
    }
    return {
        key: value
        for key, value in payload.items()
        if value not in (None, [], {}) and value != ""
    }


def _goal_approval_state_from_agent_run(run: dict[str, Any]) -> dict[str, Any]:
    summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
    direct_approval_state = (
        dict(summary.get("approval_state"))
        if isinstance(summary.get("approval_state"), dict)
        else {}
    )
    if direct_approval_state:
        normalized_status = str(direct_approval_state.get("status") or "").strip().lower()
        if normalized_status == "awaiting_approval":
            direct_approval_state["status"] = "waiting_approval"
        direct_approval_state.update(
            _goal_approval_wait_contract_payload(
                direct_approval_state,
                pending_entries=_collect_agent_run_pending_approval_entries(run),
            )
        )
        return direct_approval_state
    subagent_runtime = _latest_agent_run_artifact_content(run, "subagent_runtime")
    approval_pending = (
        list(subagent_runtime.get("approval_pending"))
        if isinstance(subagent_runtime, dict) and isinstance(subagent_runtime.get("approval_pending"), list)
        else []
    )
    approval_ids: list[str] = []
    call_ids: list[str] = []
    tool_names: list[str] = []
    artifact_pending_entries: list[dict[str, Any]] = []
    for item in approval_pending:
        if not isinstance(item, dict):
            continue
        approval_metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        approval_id = approval_metadata.get("approval_id")
        if isinstance(approval_id, str) and approval_id.strip():
            approval_ids.append(approval_id.strip())
        call_id = item.get("call_id")
        if isinstance(call_id, str) and call_id.strip():
            call_ids.append(call_id.strip())
        tool_name = item.get("tool_name")
        if isinstance(tool_name, str) and tool_name.strip():
            tool_names.append(tool_name.strip())
        normalized_pending = dict(item)
        for field_name in (
            "approval_wait_started_at",
            "waiting_since",
            "pending_since",
            "created_at",
            "requested_at",
            "submitted_at",
            "approval_wait_timeout_sec",
            "timeout_sec",
            "timeout",
            "approval_wait_expires_at",
            "expires_at",
            "timeout_at",
        ):
            if field_name not in normalized_pending and field_name in approval_metadata:
                normalized_pending[field_name] = approval_metadata.get(field_name)
        if isinstance(approval_id, str) and approval_id.strip() and not normalized_pending.get("approval_id"):
            normalized_pending["approval_id"] = approval_id.strip()
        artifact_pending_entries.append(normalized_pending)
    controlled_runtime = _latest_agent_run_artifact_content(run, "controlled_execution_runtime")
    pending_count = len(approval_pending)
    if isinstance(controlled_runtime, dict):
        try:
            pending_count = max(
                pending_count,
                int(controlled_runtime.get("approval_pending_count") or 0),
            )
        except (TypeError, ValueError):
            pass
    if pending_count <= 0:
        return {}
    payload = {
        "status": "waiting_approval",
        "pending_count": pending_count,
        "requires_approval": True,
        "approval_ids": sorted(set(approval_ids)),
        "call_ids": sorted(set(call_ids)),
        "tool_names": sorted(set(tool_names)),
    }
    payload.update(
        _goal_approval_wait_contract_payload(
            payload,
            pending_entries=artifact_pending_entries,
        )
    )
    return {
        key: value
        for key, value in payload.items()
        if value not in (None, [], {}) and value != ""
    }


def _find_agent_run_pending_approval_entry(
    run: dict[str, Any],
    approval_id: str,
) -> dict[str, Any] | None:
    for item in _collect_agent_run_pending_approval_entries(run):
        if str(item.get("approval_id") or "").strip() == approval_id:
            return item
    return None


def _collect_agent_run_pending_approval_entries(run: Mapping[str, Any]) -> list[dict[str, Any]]:
    summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
    recovery_state = (
        dict(summary.get("recovery_state"))
        if isinstance(summary.get("recovery_state"), dict)
        else {}
    )
    entries: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    candidate_states = [
        summary.get("approval_state"),
        recovery_state.get("approval_state"),
    ]
    for state in candidate_states:
        if not isinstance(state, dict):
            continue
        pending = state.get("pending_approvals") if isinstance(state.get("pending_approvals"), list) else []
        for item in pending:
            if not isinstance(item, dict):
                continue
            approval_id = str(item.get("approval_id") or "").strip()
            if not approval_id or approval_id in seen_ids:
                continue
            normalized = dict(item)
            if not normalized.get("task_key") and normalized.get("stage"):
                normalized["task_key"] = normalized.get("stage")
            if not normalized.get("stage") and normalized.get("task_key"):
                normalized["stage"] = normalized.get("task_key")
            entries.append(normalized)
            seen_ids.add(approval_id)
        approval_ids = [str(item).strip() for item in state.get("approval_ids", []) if str(item).strip()]
        for approval_id in approval_ids:
            if approval_id in seen_ids:
                continue
            fallback_stage = (
                str(recovery_state.get("stage") or state.get("resume_stage") or "").strip()
                if isinstance(recovery_state, dict)
                else ""
            )
            fallback: dict[str, Any] = {"approval_id": approval_id}
            if fallback_stage:
                fallback["stage"] = fallback_stage
                fallback["task_key"] = fallback_stage
            if fallback_stage.startswith("controlled_execution_exec:"):
                fallback["request_id"] = fallback_stage.partition(":")[2] or None
                fallback["role_id"] = "controller"
                fallback["source"] = "controlled_execution"
            entries.append(fallback)
            seen_ids.add(approval_id)
    return entries


def _goal_approval_wait_contract_payload(
    approval_state: Mapping[str, Any] | None,
    *,
    pending_entries: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if not isinstance(approval_state, Mapping):
        return {}
    normalized_status = str(approval_state.get("status") or "").strip().lower()
    if normalized_status not in {"waiting_approval", "awaiting_approval"}:
        return {}
    entries = [item for item in (pending_entries or []) if isinstance(item, Mapping)]
    state_started_at = _first_nonempty_text(
        [
            approval_state.get("approval_wait_started_at"),
            approval_state.get("waiting_since"),
            approval_state.get("pending_since"),
            approval_state.get("started_at"),
            approval_state.get("created_at"),
            approval_state.get("requested_at"),
            approval_state.get("submitted_at"),
        ]
    )
    started_at_dt = _parse_iso_datetime(state_started_at)
    if started_at_dt is None:
        entry_started_candidates = [
            parsed
            for parsed in (
                _parse_iso_datetime(
                    _first_nonempty_text(
                        [
                            item.get("approval_wait_started_at"),
                            item.get("waiting_since"),
                            item.get("pending_since"),
                            item.get("started_at"),
                            item.get("created_at"),
                            item.get("requested_at"),
                            item.get("submitted_at"),
                        ]
                    )
                )
                for item in entries
            )
            if parsed is not None
        ]
        if entry_started_candidates:
            started_at_dt = min(entry_started_candidates)
    timeout_sec = _first_positive_number(
        [
            approval_state.get("approval_wait_timeout_sec"),
            approval_state.get("timeout_sec"),
            approval_state.get("timeout"),
        ]
    )
    if timeout_sec is None:
        entry_timeout_values = [
            round(candidate, 6)
            for candidate in (
                _first_positive_number(
                    [
                        item.get("approval_wait_timeout_sec"),
                        item.get("timeout_sec"),
                        item.get("timeout"),
                    ]
                )
                for item in entries
            )
            if candidate is not None
        ]
        if len(set(entry_timeout_values)) == 1 and entry_timeout_values:
            timeout_sec = entry_timeout_values[0]
    if timeout_sec is None and started_at_dt is not None:
        expires_at = _parse_iso_datetime(
            _first_nonempty_text(
                [
                    approval_state.get("approval_wait_expires_at"),
                    approval_state.get("expires_at"),
                    approval_state.get("timeout_at"),
                ]
            )
        )
        if expires_at is None:
            entry_expires = [
                parsed
                for parsed in (
                    _parse_iso_datetime(
                        _first_nonempty_text(
                            [
                                item.get("approval_wait_expires_at"),
                                item.get("expires_at"),
                                item.get("timeout_at"),
                            ]
                        )
                    )
                    for item in entries
                )
                if parsed is not None
            ]
            if len(entry_expires) == 1:
                expires_at = entry_expires[0]
        if expires_at is not None and expires_at >= started_at_dt:
            timeout_sec = round((expires_at - started_at_dt).total_seconds(), 6)
    payload: dict[str, Any] = {}
    if started_at_dt is not None:
        payload["approval_wait_started_at"] = started_at_dt.isoformat()
    if timeout_sec is not None:
        payload["approval_wait_timeout_sec"] = (
            int(timeout_sec)
            if float(timeout_sec).is_integer()
            else timeout_sec
        )
    return payload


def _goal_approval_wait_elapsed_payload(
    approval_state: Mapping[str, Any] | None,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    if not isinstance(approval_state, Mapping):
        return {}
    started_at_dt = _parse_iso_datetime(approval_state.get("approval_wait_started_at"))
    if started_at_dt is None:
        return {}
    effective_now = now or _utcnow()
    elapsed_sec = max(0, int((effective_now - started_at_dt).total_seconds()))
    return {"approval_wait_elapsed_sec": elapsed_sec}


def _apply_exec_approval_resolution_to_agent_run_summary(
    run: dict[str, Any],
    *,
    approval_id: str,
    decision: ApprovalDecision,
    reason: str | None,
    approval: Any,
    pending_entry: Mapping[str, Any] | None,
    execution_result: Mapping[str, Any] | None,
) -> dict[str, Any]:
    summary = dict(run.get("summary") or {})
    recovery_state = (
        dict(summary.get("recovery_state"))
        if isinstance(summary.get("recovery_state"), dict)
        else {}
    )
    summary_approval_state = _remove_resolved_agent_run_approval(
        summary.get("approval_state"),
        approval_id=approval_id,
    )
    if summary_approval_state:
        summary["approval_state"] = summary_approval_state
    else:
        summary.pop("approval_state", None)
    recovery_approval_state = _remove_resolved_agent_run_approval(
        recovery_state.get("approval_state"),
        approval_id=approval_id,
    )
    if recovery_approval_state:
        recovery_state["approval_state"] = recovery_approval_state
    else:
        recovery_state.pop("approval_state", None)
    if decision == "reject":
        recovery_state["status"] = "failed"
        recovery_state["action"] = "stop"
        recovery_state["reason"] = reason or "Approval rejected"
        recovery_state.pop("unfinished_steps", None)
    elif isinstance(execution_result, Mapping):
        updated_resume_payload = _inject_exec_result_into_agent_run_resume_payload(
            recovery_state.get("resume_payload"),
            recovery_state=recovery_state,
            pending_entry=pending_entry,
            approval=approval,
            execution_result=execution_result,
        )
        if updated_resume_payload:
            recovery_state["resume_payload"] = updated_resume_payload
            if isinstance(updated_resume_payload.get("role_task_snapshot"), dict):
                recovery_state["role_task_snapshot"] = dict(updated_resume_payload["role_task_snapshot"])
    if recovery_state:
        summary["recovery_state"] = recovery_state
    else:
        summary.pop("recovery_state", None)
    return summary


def _remove_resolved_agent_run_approval(
    approval_state: Any,
    *,
    approval_id: str,
) -> dict[str, Any]:
    if not isinstance(approval_state, dict):
        return {}
    updated = dict(approval_state)
    pending = [
        dict(item)
        for item in updated.get("pending_approvals", [])
        if isinstance(item, dict) and str(item.get("approval_id") or "").strip() != approval_id
    ]
    approval_ids = [
        str(item).strip()
        for item in updated.get("approval_ids", [])
        if str(item).strip() and str(item).strip() != approval_id
    ]
    if pending:
        updated["pending_approvals"] = pending
    else:
        updated.pop("pending_approvals", None)
    if approval_ids:
        updated["approval_ids"] = sorted(set(approval_ids))
    else:
        updated.pop("approval_ids", None)
    pending_count = len(pending)
    if pending_count > 0:
        updated["pending_count"] = pending_count
        updated["status"] = str(updated.get("status") or "awaiting_approval")
        return updated
    updated.pop("pending_count", None)
    updated.pop("status", None)
    updated.pop("requires_approval", None)
    updated.pop("resume_stage", None)
    return {
        key: value
        for key, value in updated.items()
        if value not in (None, [], {}) and value != ""
    }


def _inject_exec_result_into_agent_run_resume_payload(
    resume_payload: Any,
    *,
    recovery_state: Mapping[str, Any],
    pending_entry: Mapping[str, Any] | None,
    approval: Any,
    execution_result: Mapping[str, Any],
) -> dict[str, Any]:
    payload = dict(resume_payload) if isinstance(resume_payload, dict) else {}
    if not payload:
        return {}
    role_task_snapshot = (
        dict(payload.get("role_task_snapshot"))
        if isinstance(payload.get("role_task_snapshot"), dict)
        else (
            dict(recovery_state.get("role_task_snapshot"))
            if isinstance(recovery_state.get("role_task_snapshot"), dict)
            else {}
        )
    )
    tasks = dict(role_task_snapshot.get("tasks")) if isinstance(role_task_snapshot.get("tasks"), dict) else {}
    task_key = ""
    if isinstance(pending_entry, Mapping):
        task_key = str(
            pending_entry.get("task_key")
            or pending_entry.get("stage")
            or ""
        ).strip()
    if not task_key:
        task_key = str(recovery_state.get("stage") or "").strip()
    if not task_key:
        return {}
    request_id = None
    if isinstance(pending_entry, Mapping):
        request_id = pending_entry.get("request_id")
    if not isinstance(request_id, str) or not request_id.strip():
        request_id = task_key.partition(":")[2] if task_key.startswith("controlled_execution_exec:") else None
    task_entry = dict(tasks.get(task_key)) if isinstance(tasks.get(task_key), dict) else {}
    command_payload = getattr(approval, "command_payload", None)
    enriched_result = dict(execution_result)
    if isinstance(request_id, str) and request_id.strip() and not enriched_result.get("request_id"):
        enriched_result["request_id"] = request_id.strip()
    if isinstance(command_payload, dict):
        if not enriched_result.get("command"):
            enriched_result["command"] = command_payload.get("command")
        if not enriched_result.get("shell"):
            enriched_result["shell"] = command_payload.get("shell")
        if not enriched_result.get("workdir"):
            enriched_result["workdir"] = command_payload.get("workdir")
        if "background" not in enriched_result:
            enriched_result["background"] = bool(command_payload.get("background", False))
    task_entry["task_key"] = task_key
    task_entry["role_id"] = str(task_entry.get("role_id") or _agent_run_pending_role_id(pending_entry) or "controller")
    task_entry["stage"] = str(task_entry.get("stage") or task_key)
    task_entry["status"] = "completed"
    task_entry["resume_action"] = "reuse_output"
    task_entry["result_summary"] = {"execution_result": enriched_result}
    tasks[task_key] = task_entry
    role_task_snapshot["tasks"] = tasks
    payload["role_task_snapshot"] = role_task_snapshot
    return payload


def _agent_run_pending_role_id(pending_entry: Mapping[str, Any] | None) -> str | None:
    if not isinstance(pending_entry, Mapping):
        return None
    role_id = pending_entry.get("role_id")
    if isinstance(role_id, str) and role_id.strip():
        return role_id.strip()
    return None


def _pending_entry_metadata(pending_entry: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(pending_entry, Mapping):
        return {}
    metadata = pending_entry.get("metadata")
    return dict(metadata) if isinstance(metadata, dict) else {}


def _rehydrate_exec_approval_metadata_from_pending_entry(
    pending_entry: Mapping[str, Any] | None,
) -> dict[str, Any]:
    metadata = _pending_entry_metadata(pending_entry)
    output: dict[str, Any] = {}
    for key in ("policy_state", "policy_reason", "rule_id"):
        value = metadata.get(key)
        if value is not None:
            output[key] = value
    suggested_rule = metadata.get("suggested_rule")
    if isinstance(suggested_rule, dict):
        output["suggested_rule"] = dict(suggested_rule)
    return output


def _rehydrate_exec_command_payload_from_agent_run(
    run: Mapping[str, Any],
    *,
    pending_entry: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    pending = dict(pending_entry) if isinstance(pending_entry, Mapping) else {}
    pending_metadata = _pending_entry_metadata(pending_entry)
    snapshot = _agent_run_role_task_snapshot(run)
    snapshot_tasks = dict(snapshot.get("tasks")) if isinstance(snapshot.get("tasks"), dict) else {}
    request_id = _agent_run_pending_request_id(pending_entry)
    controller_task_key = (
        str(pending.get("controller_task_key") or "").strip()
        if isinstance(pending.get("controller_task_key"), str)
        else ""
    )
    if not controller_task_key and request_id:
        controller_task_key = f"controlled_execution_controller:{request_id}"
    controller_task = (
        dict(snapshot_tasks.get(controller_task_key))
        if controller_task_key and isinstance(snapshot_tasks.get(controller_task_key), dict)
        else {}
    )
    controller_result_summary = (
        dict(controller_task.get("result_summary"))
        if isinstance(controller_task.get("result_summary"), dict)
        else {}
    )
    controller_decision = (
        dict(controller_result_summary.get("controller_decision"))
        if isinstance(controller_result_summary.get("controller_decision"), dict)
        else {}
    )
    execution_request = (
        dict(controller_result_summary.get("execution_request"))
        if isinstance(controller_result_summary.get("execution_request"), dict)
        else {}
    )
    command = _first_nonempty_text(
        [
            pending.get("command"),
            controller_decision.get("command"),
            execution_request.get("command"),
        ]
    )
    if command is None:
        return None
    shell = _first_nonempty_text(
        [
            pending.get("shell"),
            controller_decision.get("shell"),
            execution_request.get("shell"),
        ]
    ) or "auto"
    workdir = _first_nonempty_text(
        [
            pending.get("workdir"),
            pending_metadata.get("workdir"),
            controller_decision.get("workdir"),
            execution_request.get("workdir"),
            run.get("workspace_dir"),
        ]
    )
    timeout_sec = _first_positive_number(
        [
            pending.get("timeout"),
            pending_metadata.get("timeout_sec"),
            controller_decision.get("timeout"),
            execution_request.get("timeout"),
        ]
    )
    background = _first_optional_bool(
        [
            pending.get("background"),
            pending_metadata.get("background"),
            controller_decision.get("background"),
            execution_request.get("background"),
        ],
        default=False,
    )
    tty = _first_optional_bool(
        [
            pending.get("tty"),
            pending_metadata.get("tty"),
        ],
        default=False,
    )
    payload = {
        "command": command,
        "shell": shell,
        "workdir": workdir,
        "env": None,
        "timeout_sec": timeout_sec,
        "background": background,
        "tty": tty,
        "log_path": _first_nonempty_text([pending_metadata.get("log_path")]),
        "checkpoint_dir": _first_nonempty_text([pending_metadata.get("checkpoint_dir")]),
        "approval_state": "approved",
    }
    return {key: value for key, value in payload.items() if value is not None}


def _agent_run_role_task_snapshot(run: Mapping[str, Any]) -> dict[str, Any]:
    resume_payload = _resume_payload_from_run(dict(run))
    if isinstance(resume_payload, dict) and isinstance(resume_payload.get("role_task_snapshot"), dict):
        return dict(resume_payload["role_task_snapshot"])
    recovery_state = _recovery_state_from_run(dict(run))
    snapshot = recovery_state.get("role_task_snapshot")
    if isinstance(snapshot, dict):
        return dict(snapshot)
    return {}


def _agent_run_pending_request_id(pending_entry: Mapping[str, Any] | None) -> str | None:
    if not isinstance(pending_entry, Mapping):
        return None
    request_id = pending_entry.get("request_id")
    if isinstance(request_id, str) and request_id.strip():
        return request_id.strip()
    for key in ("task_key", "stage"):
        value = pending_entry.get(key)
        if not isinstance(value, str):
            continue
        stage = value.strip()
        if stage.startswith("controlled_execution_exec:"):
            candidate = stage.partition(":")[2].strip()
            if candidate:
                return candidate
    return None


def _first_positive_number(values: list[Any]) -> float | None:
    for value in values:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def _first_optional_bool(values: list[Any], *, default: bool) -> bool:
    for value in values:
        if isinstance(value, bool):
            return value
    return default


def _goal_attempt_recovery_state_payload(attempt: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(attempt, dict):
        return None
    summary = attempt.get("summary") if isinstance(attempt.get("summary"), dict) else {}
    recovery_state = (
        dict(summary.get("linked_recovery_state"))
        if isinstance(summary.get("linked_recovery_state"), dict)
        else {}
    )
    return recovery_state or None


def _goal_attempt_approval_state_payload(attempt: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(attempt, dict):
        return None
    summary = attempt.get("summary") if isinstance(attempt.get("summary"), dict) else {}
    approval_state = (
        dict(summary.get("linked_approval_state"))
        if isinstance(summary.get("linked_approval_state"), dict)
        else {}
    )
    if not approval_state:
        return None
    approval_state.update(_goal_approval_wait_elapsed_payload(approval_state))
    return approval_state


def _goal_runtime_budget_health_payload(
    goal: dict[str, Any],
    run_policy: dict[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    effective_now = now or _utcnow()
    requested_duration_sec = (
        int(run_policy.get("requested_duration_sec"))
        if isinstance(run_policy.get("requested_duration_sec"), int)
        else None
    )
    soft_deadline_at = _normalized_nonempty_text(run_policy.get("soft_deadline_at"))
    hard_stop_at = _normalized_nonempty_text(run_policy.get("hard_stop_at"))
    started_at = _goal_runtime_budget_started_at(goal)
    started_at_dt = _parse_iso_datetime(started_at)
    elapsed_sec = (
        max(0, int((effective_now - started_at_dt).total_seconds()))
        if started_at_dt is not None
        else 0
    )
    remaining_duration_sec = (
        max(0, requested_duration_sec - elapsed_sec)
        if requested_duration_sec is not None
        else None
    )
    soft_deadline_dt = _parse_iso_datetime(soft_deadline_at)
    hard_stop_dt = _parse_iso_datetime(hard_stop_at)
    soft_deadline_reached = soft_deadline_dt is not None and effective_now >= soft_deadline_dt
    hard_stop_reached = hard_stop_dt is not None and effective_now >= hard_stop_dt
    remaining_hard_stop_sec = (
        max(0, int((hard_stop_dt - effective_now).total_seconds()))
        if hard_stop_dt is not None
        else None
    )
    remaining_candidates = [
        value
        for value in (remaining_duration_sec, remaining_hard_stop_sec)
        if isinstance(value, int)
    ]
    effective_remaining_sec = min(remaining_candidates) if remaining_candidates else None
    duration_exhausted = (
        requested_duration_sec is not None
        and started_at_dt is not None
        and elapsed_sec >= requested_duration_sec
    )
    attempt_count = len(goal.get("attempts") or [])
    max_attempt_retries = (
        int(run_policy.get("max_attempt_retries"))
        if isinstance(run_policy.get("max_attempt_retries"), int)
        else None
    )
    retries_used = max(0, attempt_count - 1)
    retries_remaining = (
        max(0, max_attempt_retries - retries_used)
        if max_attempt_retries is not None
        else None
    )
    retry_limit_reached = (
        max_attempt_retries is not None
        and attempt_count > 0
        and retries_used >= max_attempt_retries
    )
    status = "unbounded"
    if retry_limit_reached:
        status = "retry_limit_reached"
    elif hard_stop_reached:
        status = "hard_stop_reached"
    elif duration_exhausted:
        status = "duration_exhausted"
    elif soft_deadline_reached:
        status = "soft_deadline_reached"
    elif started_at_dt is None:
        status = "not_started"
    elif effective_remaining_sec is not None:
        status = "within_budget"
    payload = {
        "status": status,
        "runtime_mode": run_policy.get("runtime_mode"),
        "on_budget_exhausted": run_policy.get("on_budget_exhausted"),
        "started_at": started_at,
        "elapsed_sec": elapsed_sec,
        "requested_duration_sec": requested_duration_sec,
        "remaining_duration_sec": remaining_duration_sec,
        "soft_deadline_at": soft_deadline_at,
        "soft_deadline_reached": soft_deadline_reached,
        "hard_stop_at": hard_stop_at,
        "hard_stop_reached": hard_stop_reached,
        "remaining_hard_stop_sec": remaining_hard_stop_sec,
        "effective_remaining_sec": effective_remaining_sec,
        "attempt_count": attempt_count,
        "max_attempt_retries": max_attempt_retries,
        "retries_used": retries_used,
        "retries_remaining": retries_remaining,
        "retry_limit_reached": retry_limit_reached,
        "exhausted": hard_stop_reached or duration_exhausted,
        "block_new_attempts": retry_limit_reached or hard_stop_reached or duration_exhausted,
    }
    return {
        key: value
        for key, value in payload.items()
        if value is not None
    }


def _goal_runtime_budget_started_at(goal: dict[str, Any]) -> str | None:
    started_at = goal.get("started_at")
    if isinstance(started_at, str) and started_at.strip():
        return started_at.strip()
    return None


def _goal_runtime_budget_is_exhausted(budget: dict[str, Any]) -> bool:
    return bool(budget.get("exhausted"))


def _goal_runtime_budget_blocks_goal_transition(
    budget: dict[str, Any],
    *,
    check_retry_limit: bool,
) -> bool:
    if _goal_runtime_budget_is_exhausted(budget):
        return True
    return check_retry_limit and bool(budget.get("retry_limit_reached"))


def _goal_runtime_budget_hold_status(budget: dict[str, Any]) -> str:
    if bool(budget.get("retry_limit_reached")):
        return "failed"
    return "awaiting_resources"


def _goal_runtime_budget_block_reason(budget: dict[str, Any]) -> str | None:
    if bool(budget.get("retry_limit_reached")):
        retries_used = budget.get("retries_used")
        retries_allowed = budget.get("max_attempt_retries")
        return (
            f"Goal retry budget exhausted ({retries_used}/{retries_allowed} retries used)."
        )
    if bool(budget.get("hard_stop_reached")):
        hard_stop_at = budget.get("hard_stop_at")
        return f"Goal hard stop reached at {hard_stop_at}."
    if bool(budget.get("status") == "duration_exhausted"):
        requested_duration_sec = budget.get("requested_duration_sec")
        return f"Goal runtime budget exhausted after {requested_duration_sec} seconds."
    return None


def _goal_attempt_checkpoint_payload(attempt: dict[str, Any] | None) -> dict[str, Any] | None:
    recovery_state = _goal_attempt_recovery_state_payload(attempt)
    if not isinstance(recovery_state, dict):
        return None
    checkpoint = recovery_state.get("checkpoint")
    if isinstance(checkpoint, dict) and checkpoint:
        return dict(checkpoint)
    checkpoint_index = recovery_state.get("checkpoint_index")
    if checkpoint_index is None:
        return None
    return {"checkpoint_index": checkpoint_index}


def _goal_checkpoint_policy_health_payload(
    run_policy: dict[str, Any],
    attempt: dict[str, Any] | None,
    *,
    goal_status: str,
) -> dict[str, Any]:
    checkpoint_interval_sec = (
        int(run_policy.get("checkpoint_interval_sec"))
        if isinstance(run_policy.get("checkpoint_interval_sec"), int)
        else None
    )
    checkpoint_interval_steps = (
        int(run_policy.get("checkpoint_interval_steps"))
        if isinstance(run_policy.get("checkpoint_interval_steps"), int)
        else None
    )
    generation_refresh_interval_sec = (
        int(run_policy.get("generation_refresh_interval_sec"))
        if isinstance(run_policy.get("generation_refresh_interval_sec"), int)
        else None
    )
    context_handoff_threshold = run_policy.get("context_handoff_threshold")
    checkpoint = _goal_attempt_checkpoint_payload(attempt) or {}
    captured_at = (
        str(checkpoint.get("captured_at")).strip()
        if isinstance(checkpoint.get("captured_at"), str) and checkpoint.get("captured_at").strip()
        else None
    )
    captured_at_dt = _parse_iso_datetime(captured_at)
    seconds_since_checkpoint: int | None = None
    overdue_sec: int | None = None
    if captured_at_dt is not None:
        seconds_since_checkpoint = max(
            0,
            int((_utcnow() - captured_at_dt).total_seconds()),
        )
    if checkpoint_interval_sec is None and checkpoint_interval_steps is None:
        status = "not_configured"
    elif checkpoint:
        if (
            goal_status in GOAL_ACTIVE_STATUS
            and checkpoint_interval_sec is not None
            and seconds_since_checkpoint is not None
            and seconds_since_checkpoint > checkpoint_interval_sec
        ):
            status = "overdue"
            overdue_sec = seconds_since_checkpoint - checkpoint_interval_sec
        else:
            status = "recorded"
    else:
        status = "pending_first_checkpoint" if goal_status in GOAL_ACTIVE_STATUS else "missing_checkpoint"
    payload = {
        "status": status,
        "interval_sec": checkpoint_interval_sec,
        "interval_steps": checkpoint_interval_steps,
        "generation_refresh_interval_sec": generation_refresh_interval_sec,
        "context_handoff_threshold": context_handoff_threshold,
        "checkpoint_index": checkpoint.get("checkpoint_index"),
        "stage": checkpoint.get("stage"),
        "captured_at": captured_at,
        "seconds_since_checkpoint": seconds_since_checkpoint,
        "overdue_sec": overdue_sec,
    }
    return {
        key: value
        for key, value in payload.items()
        if value is not None
    }


def _goal_linked_agent_run_health_payload(
    attempt: dict[str, Any] | None,
    *,
    checkpoint_policy: dict[str, Any] | None = None,
    collector_state: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not isinstance(attempt, dict):
        return None
    summary = attempt.get("summary") if isinstance(attempt.get("summary"), dict) else {}
    payload = {
        "run_id": attempt.get("agent_run_id"),
        "status": summary.get("agent_run_status") or attempt.get("status"),
        "degraded": bool(summary.get("linked_run_degraded", False)),
        "recovery_state": _goal_attempt_recovery_state_payload(attempt),
        "checkpoint": _goal_attempt_checkpoint_payload(attempt),
        "approval_state": _goal_attempt_approval_state_payload(attempt),
        "collector_state": collector_state or None,
        "checkpoint_policy": checkpoint_policy or None,
    }
    if not payload["run_id"]:
        return None
    return payload


def _goal_collector_stall_timeout_sec(run_policy: dict[str, Any] | None) -> int | None:
    normalized = dict(run_policy or {})
    explicit_timeout = normalized.get("collector_shard_stall_timeout_sec")
    if isinstance(explicit_timeout, int) and explicit_timeout > 0:
        return explicit_timeout
    checkpoint_interval_sec = normalized.get("checkpoint_interval_sec")
    if isinstance(checkpoint_interval_sec, int) and checkpoint_interval_sec > 0:
        return checkpoint_interval_sec
    generation_refresh_interval_sec = normalized.get("generation_refresh_interval_sec")
    if isinstance(generation_refresh_interval_sec, int) and generation_refresh_interval_sec > 0:
        return generation_refresh_interval_sec
    return None


def _failed_collector_shard_manifests_for_run(
    run: dict[str, Any],
    *,
    attempt_id: str | None = None,
) -> list[dict[str, Any]]:
    recovery_state = _recovery_state_from_run(run)
    resume_payload = _build_agent_run_resume_payload(
        run=run,
        recovery_state=recovery_state,
        strategy="continue_from_checkpoint",
    )
    protocol_artifacts = (
        dict(resume_payload.get("protocol_artifacts"))
        if isinstance(resume_payload.get("protocol_artifacts"), dict)
        else {}
    )
    manifests = dedupe_collector_shard_manifests(
        collector_shard_manifests_from_result(protocol_artifacts, attempt_id=attempt_id)
    )
    return [
        dict(manifest)
        for manifest in manifests
        if str(manifest.get("status") or "").strip().lower() in FAILED_COLLECTOR_SHARD_STATUSES
    ]


def _collector_failed_shard_retry_manifest(shard: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(shard)
    payload["status"] = "retrying"
    payload.pop("completed_at", None)
    payload.pop("failed_at", None)
    return payload


def _collector_failed_shard_retry_guidance(shard: Mapping[str, Any]) -> str:
    shard_id = str(shard.get("shard_id") or "unknown-shard").strip() or "unknown-shard"
    source = (
        dict(shard.get("source")) if isinstance(shard.get("source"), Mapping) else {}
    )
    progress = (
        dict(shard.get("progress")) if isinstance(shard.get("progress"), Mapping) else {}
    )
    source_ref = (
        str(source.get("id") or source.get("url") or "").strip() or "unknown source"
    )
    cursor = str(progress.get("cursor") or "").strip() or "start"
    return (
        f"Retry only collector shard {shard_id}. Reuse its saved source/progress payload for "
        f"{source_ref} and continue collection from cursor {cursor}. Do not restart already "
        "completed shards or duplicate previously captured records."
    )


def _goal_collector_state_payload(
    run: dict[str, Any] | None,
    *,
    attempt_id: str | None = None,
    run_policy: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not isinstance(run, dict):
        return None
    artifacts = run.get("artifacts")
    if not isinstance(artifacts, list):
        return None
    manifests = extract_collector_shard_manifests(
        [dict(item) for item in artifacts if isinstance(item, dict)]
    )
    if isinstance(attempt_id, str) and attempt_id.strip():
        scoped_attempt_id = attempt_id.strip()
        manifests = [
            manifest
            for manifest in manifests
            if str(manifest.get("attempt_id") or scoped_attempt_id).strip() == scoped_attempt_id
        ]
    manifests = dedupe_collector_shard_manifests(manifests)
    return build_collector_state_manifest(
        manifests,
        stall_timeout_sec=_goal_collector_stall_timeout_sec(run_policy),
    )


def _goal_recommended_next_action(
    *,
    goal_status: str,
    runtime_budget: dict[str, Any] | None,
    approval_state: dict[str, Any] | None,
    current_generation: dict[str, Any] | None,
    checkpoint_policy: dict[str, Any] | None,
    collector_state: dict[str, Any] | None,
    linked_agent_run: dict[str, Any] | None,
    open_findings: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    finding_codes = {
        str(item.get("finding_code") or "").strip()
        for item in (open_findings or [])
        if isinstance(item, dict) and str(item.get("finding_code") or "").strip()
    }
    run_id = (
        str(linked_agent_run.get("run_id") or "").strip()
        if isinstance(linked_agent_run, dict)
        else ""
    )
    approval_ids = (
        list(approval_state.get("approval_ids") or [])
        if isinstance(approval_state, dict)
        else []
    )
    approval_status = (
        str(approval_state.get("status") or "").strip()
        if isinstance(approval_state, dict)
        else ""
    )
    if goal_status == "waiting_approval" or approval_status in {"waiting_approval", "awaiting_approval"}:
        payload = {
            "action": "resolve_approval",
            "summary": (
                "Goal has been waiting on operator approval longer than the configured approval wait timeout."
                if "approval_wait_timeout" in finding_codes
                else "Goal is waiting on operator approval before it can continue."
            ),
            "blocking": True,
        }
        if run_id:
            payload["run_id"] = run_id
        if approval_ids:
            payload["approval_ids"] = approval_ids
        if "approval_wait_timeout" in finding_codes:
            payload["finding_code"] = "approval_wait_timeout"
            if isinstance(approval_state, dict):
                if approval_state.get("approval_wait_elapsed_sec") is not None:
                    payload["approval_wait_elapsed_sec"] = approval_state.get("approval_wait_elapsed_sec")
                if approval_state.get("approval_wait_timeout_sec") is not None:
                    payload["approval_wait_timeout_sec"] = approval_state.get("approval_wait_timeout_sec")
        return payload
    budget_status = (
        str(runtime_budget.get("status") or "").strip()
        if isinstance(runtime_budget, dict)
        else ""
    )
    if (
        "runtime_budget_exhausted" in finding_codes
        or budget_status in {"retry_limit_reached", "hard_stop_reached", "duration_exhausted"}
    ):
        return {
            "action": "inspect_runtime_budget",
            "summary": "Goal cannot safely continue until its runtime budget or retry policy is adjusted.",
            "blocking": True,
            "budget_status": budget_status or None,
        }
    if "collector_shard_stuck" in finding_codes:
        payload = {
            "action": "inspect_collector_shards",
            "summary": "One or more collector shards have not advanced within the configured stall timeout.",
            "blocking": False,
            "finding_code": "collector_shard_stuck",
        }
        if run_id:
            payload["run_id"] = run_id
        if isinstance(collector_state, dict):
            if collector_state.get("stalled_shard_count") is not None:
                payload["stalled_shard_count"] = collector_state.get("stalled_shard_count")
            if collector_state.get("stall_timeout_sec") is not None:
                payload["stall_timeout_sec"] = collector_state.get("stall_timeout_sec")
        return payload
    if (
        isinstance(current_generation, dict)
        and bool(current_generation.get("context_handoff_due"))
    ) or "context_handoff_due" in finding_codes:
        payload = {
            "action": "refresh_worker_generation",
            "summary": "Active worker generation crossed the context handoff threshold and should hand off to a fresh worker.",
            "blocking": False,
            "finding_code": "context_handoff_due",
        }
        if isinstance(current_generation, dict) and current_generation.get("generation_id") is not None:
            payload["generation_id"] = current_generation.get("generation_id")
        if run_id:
            payload["run_id"] = run_id
        return payload
    if (
        isinstance(current_generation, dict)
        and bool(current_generation.get("token_refresh_due"))
    ) or "generation_token_refresh_due" in finding_codes:
        payload = {
            "action": "refresh_worker_generation",
            "summary": "Active worker generation crossed the token refresh threshold and should hand off to a fresh worker.",
            "blocking": False,
            "finding_code": "generation_token_refresh_due",
        }
        if isinstance(current_generation, dict) and current_generation.get("generation_id") is not None:
            payload["generation_id"] = current_generation.get("generation_id")
        if run_id:
            payload["run_id"] = run_id
        return payload
    if (
        isinstance(current_generation, dict)
        and bool(current_generation.get("refresh_due"))
    ) or "generation_refresh_overdue" in finding_codes:
        payload = {
            "action": "refresh_worker_generation",
            "summary": "Active worker generation exceeded its refresh interval and should hand off to a fresh worker.",
            "blocking": False,
            "finding_code": "generation_refresh_overdue",
        }
        if isinstance(current_generation, dict) and current_generation.get("generation_id") is not None:
            payload["generation_id"] = current_generation.get("generation_id")
        if run_id:
            payload["run_id"] = run_id
        return payload
    checkpoint_status = (
        str(checkpoint_policy.get("status") or "").strip()
        if isinstance(checkpoint_policy, dict)
        else ""
    )
    if checkpoint_status in {"pending_first_checkpoint", "overdue"} or (
        finding_codes & {"missing_checkpoint", "checkpoint_overdue"}
    ):
        finding_code = (
            "missing_checkpoint"
            if checkpoint_status == "pending_first_checkpoint" or "missing_checkpoint" in finding_codes
            else "checkpoint_overdue"
        )
        payload = {
            "action": "capture_checkpoint",
            "summary": "Linked run should persist a fresh durable checkpoint before continuing unattended execution.",
            "blocking": False,
            "finding_code": finding_code,
        }
        if run_id:
            payload["run_id"] = run_id
        return payload
    if goal_status in {"paused", "stalled", "awaiting_resources", "awaiting_operator"}:
        payload = {
            "action": "resume_goal",
            "summary": (
                "Goal is waiting at a durable checkpoint for operator guidance before continuing."
                if goal_status == "awaiting_operator"
                else "Goal is resumable but not actively progressing."
            ),
            "blocking": goal_status not in {"paused", "awaiting_operator"},
        }
        if run_id:
            payload["run_id"] = run_id
        return payload
    if goal_status in GOAL_ACTIVE_STATUS:
        payload = {
            "action": "monitor",
            "summary": "Goal is actively progressing and does not currently need operator intervention.",
            "blocking": False,
        }
        if run_id:
            payload["run_id"] = run_id
        return payload
    return None


def _goal_checkpoint_record_response(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(record, dict):
        return None
    return {
        "checkpoint_id": record["id"],
        "goal_id": record["goal_id"],
        "attempt_id": record.get("attempt_id"),
        "agent_run_id": record.get("agent_run_id"),
        "checkpoint_index": record.get("checkpoint_index"),
        "stage": record.get("stage"),
        "source": record.get("source"),
        "payload": dict(record.get("payload") or {}),
        "metadata": dict(record.get("metadata") or {}),
        "captured_at": record.get("captured_at"),
        "created_at": record.get("created_at"),
        "updated_at": record.get("updated_at"),
    }


def _goal_memory_snapshot_response(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(record, dict):
        return None
    return {
        "snapshot_id": record["id"],
        "goal_id": record["goal_id"],
        "attempt_id": record.get("attempt_id"),
        "checkpoint_id": record.get("checkpoint_id"),
        "snapshot_kind": record.get("snapshot_kind"),
        "snapshot": dict(record.get("snapshot") or {}),
        "metadata": dict(record.get("metadata") or {}),
        "captured_at": record.get("captured_at"),
        "created_at": record.get("created_at"),
        "updated_at": record.get("updated_at"),
    }


def _goal_worker_generation_refresh_policy_payload(
    record: dict[str, Any] | None,
    *,
    run_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {}
    normalized_run_policy = dict(run_policy or {})
    started_at = record.get("started_at")
    started_at_dt = _parse_iso_datetime(started_at)
    elapsed_sec = (
        max(0, int((_utcnow() - started_at_dt).total_seconds()))
        if started_at_dt is not None
        else None
    )
    refresh_interval_sec = (
        int(normalized_run_policy.get("generation_refresh_interval_sec"))
        if isinstance(normalized_run_policy.get("generation_refresh_interval_sec"), int)
        else None
    )
    refresh_overdue_sec = (
        max(0, elapsed_sec - refresh_interval_sec)
        if refresh_interval_sec is not None and elapsed_sec is not None and elapsed_sec > refresh_interval_sec
        else None
    )
    return {
        "elapsed_sec": elapsed_sec,
        "generation_refresh_interval_sec": refresh_interval_sec,
        "refresh_due": (
            bool(refresh_interval_sec is not None and elapsed_sec is not None and elapsed_sec >= refresh_interval_sec)
            if refresh_interval_sec is not None
            else None
        ),
        "refresh_overdue_sec": refresh_overdue_sec,
    }


def _goal_context_handoff_health_payload(
    run: dict[str, Any] | None,
    *,
    run_policy: dict[str, Any] | None = None,
    generation_started_at: str | None = None,
) -> dict[str, Any]:
    threshold = _goal_context_snapshot_fraction(
        (run_policy or {}).get("context_handoff_threshold"),
    )
    payload: dict[str, Any] = {}
    if threshold is not None:
        payload["context_handoff_threshold"] = threshold
    latest_snapshot = _latest_debate_context_snapshot_payload(
        run,
        generation_started_at=generation_started_at,
    )
    if not isinstance(latest_snapshot, dict):
        return payload
    usage_ratio = _goal_context_snapshot_fraction(latest_snapshot.get("usage_ratio"))
    if usage_ratio is not None:
        payload["usage_ratio"] = usage_ratio
        if threshold is not None:
            payload["context_handoff_due"] = usage_ratio >= threshold
            if usage_ratio >= threshold:
                payload["context_handoff_over_threshold"] = round(
                    usage_ratio - threshold,
                    4,
                )
    for field_name in ("role_id", "stage", "compaction_level", "largest_section"):
        value = latest_snapshot.get(field_name)
        if isinstance(value, str) and value.strip():
            payload[field_name] = value
    for field_name in ("estimated_prompt_tokens", "reserved_output_tokens", "max_input_tokens"):
        value = latest_snapshot.get(field_name)
        if isinstance(value, int) and value >= 0:
            payload[field_name] = value
    for field_name in ("truncated", "used_chunking", "overflow"):
        value = latest_snapshot.get(field_name)
        if isinstance(value, bool):
            payload[field_name] = value
    return payload


def _latest_subagent_runtime_payload(
    run: dict[str, Any] | None,
    *,
    generation_started_at: str | None = None,
) -> dict[str, Any] | None:
    if not isinstance(run, dict):
        return None
    artifacts = run.get("artifacts")
    if not isinstance(artifacts, list):
        return None
    cutoff = _parse_iso_datetime(generation_started_at)
    for artifact in reversed(artifacts):
        if not isinstance(artifact, dict):
            continue
        if str(artifact.get("artifact_type") or "") != "subagent_runtime":
            continue
        created_at = (
            str(artifact.get("created_at") or "").strip()
            if isinstance(artifact.get("created_at"), str)
            else ""
        )
        created_at_dt = _parse_iso_datetime(created_at)
        if cutoff is not None and (created_at_dt is None or created_at_dt < cutoff):
            continue
        metadata = artifact.get("metadata") if isinstance(artifact.get("metadata"), dict) else {}
        content = metadata.get("content")
        if not isinstance(content, dict):
            continue
        payload = dict(content)
        if created_at:
            payload["artifact_created_at"] = created_at
        return payload
    return None


def _subagent_runtime_usage_totals(payload: Mapping[str, Any]) -> dict[str, Any]:
    aggregated: dict[str, Any] = {}
    int_fields = (
        "invocation_count",
        "completed_invocation_count",
        "token_tracked_invocation_count",
        "input_tokens",
        "output_tokens",
        "total_tokens",
    )
    for field_name in int_fields:
        value = payload.get(field_name)
        if isinstance(value, int) and value >= 0:
            aggregated[field_name] = value
    generation_time_ms = payload.get("generation_time_ms")
    if isinstance(generation_time_ms, (int, float)) and generation_time_ms >= 0:
        aggregated["generation_time_ms"] = round(float(generation_time_ms), 3)
    finish_reason_counts = payload.get("finish_reason_counts")
    if isinstance(finish_reason_counts, dict):
        normalized_finish_reason_counts = {
            str(key): int(value)
            for key, value in finish_reason_counts.items()
            if isinstance(key, str) and key.strip() and isinstance(value, int) and value > 0
        }
        if normalized_finish_reason_counts:
            aggregated["finish_reason_counts"] = normalized_finish_reason_counts
    has_precomputed_usage = bool(aggregated)
    invocations = payload.get("invocations")
    if has_precomputed_usage or not isinstance(invocations, list):
        return aggregated
    if "invocation_count" not in aggregated:
        aggregated["invocation_count"] = len([item for item in invocations if isinstance(item, dict)])
    completed_invocation_count = int(aggregated.get("completed_invocation_count", 0) or 0)
    token_tracked_invocation_count = int(aggregated.get("token_tracked_invocation_count", 0) or 0)
    input_tokens = int(aggregated.get("input_tokens", 0) or 0)
    output_tokens = int(aggregated.get("output_tokens", 0) or 0)
    observed_generation_time_ms = float(aggregated.get("generation_time_ms", 0.0) or 0.0)
    observed_finish_reason_counts = (
        dict(aggregated.get("finish_reason_counts"))
        if isinstance(aggregated.get("finish_reason_counts"), dict)
        else {}
    )
    for invocation in invocations:
        if not isinstance(invocation, dict):
            continue
        completion = invocation.get("completion")
        if not isinstance(completion, dict):
            continue
        completed_invocation_count += 1
        completion_input_tokens = completion.get("input_tokens")
        completion_output_tokens = completion.get("output_tokens")
        completion_generation_time_ms = completion.get("generation_time_ms")
        if isinstance(completion_input_tokens, int) and completion_input_tokens >= 0:
            input_tokens += completion_input_tokens
        if isinstance(completion_output_tokens, int) and completion_output_tokens >= 0:
            output_tokens += completion_output_tokens
        if (
            isinstance(completion_input_tokens, int)
            or isinstance(completion_output_tokens, int)
            or isinstance(completion_generation_time_ms, (int, float))
        ):
            token_tracked_invocation_count += 1
        if isinstance(completion_generation_time_ms, (int, float)) and completion_generation_time_ms >= 0:
            observed_generation_time_ms += float(completion_generation_time_ms)
        finish_reason = str(completion.get("finish_reason") or "").strip()
        if finish_reason:
            observed_finish_reason_counts[finish_reason] = (
                int(observed_finish_reason_counts.get(finish_reason, 0) or 0) + 1
            )
    aggregated["completed_invocation_count"] = completed_invocation_count
    aggregated["token_tracked_invocation_count"] = token_tracked_invocation_count
    aggregated["input_tokens"] = input_tokens
    aggregated["output_tokens"] = output_tokens
    aggregated["total_tokens"] = input_tokens + output_tokens
    aggregated["generation_time_ms"] = round(observed_generation_time_ms, 3)
    if observed_finish_reason_counts:
        aggregated["finish_reason_counts"] = observed_finish_reason_counts
    return aggregated


def _goal_subagent_runtime_health_payload(
    run: dict[str, Any] | None,
    *,
    run_policy: dict[str, Any] | None = None,
    generation_started_at: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    token_refresh_threshold = _goal_policy_positive_int(
        (run_policy or {}).get("generation_token_refresh_threshold"),
        maximum=_GOAL_RUNTIME_POLICY_TOKEN_REFRESH_MAX,
    )
    if token_refresh_threshold is not None:
        payload["generation_token_refresh_threshold"] = token_refresh_threshold
    latest_runtime = _latest_subagent_runtime_payload(
        run,
        generation_started_at=generation_started_at,
    )
    if not isinstance(latest_runtime, dict):
        return payload
    totals = _subagent_runtime_usage_totals(latest_runtime)
    if isinstance(totals.get("invocation_count"), int):
        payload["subagent_invocation_count"] = totals["invocation_count"]
    if isinstance(totals.get("completed_invocation_count"), int):
        payload["subagent_completed_invocation_count"] = totals["completed_invocation_count"]
    if isinstance(totals.get("token_tracked_invocation_count"), int):
        payload["subagent_token_tracked_invocation_count"] = totals[
            "token_tracked_invocation_count"
        ]
    if isinstance(totals.get("input_tokens"), int):
        payload["observed_input_tokens"] = totals["input_tokens"]
    if isinstance(totals.get("output_tokens"), int):
        payload["observed_output_tokens"] = totals["output_tokens"]
    if isinstance(totals.get("total_tokens"), int):
        payload["observed_total_tokens"] = totals["total_tokens"]
        if token_refresh_threshold is not None:
            payload["token_refresh_due"] = totals["total_tokens"] >= token_refresh_threshold
            if totals["total_tokens"] >= token_refresh_threshold:
                payload["token_refresh_over_threshold"] = (
                    totals["total_tokens"] - token_refresh_threshold
                )
    generation_time_ms = totals.get("generation_time_ms")
    if isinstance(generation_time_ms, (int, float)):
        payload["observed_generation_time_ms"] = round(float(generation_time_ms), 3)
    if isinstance(totals.get("finish_reason_counts"), dict) and totals.get("finish_reason_counts"):
        payload["observed_finish_reason_counts"] = dict(totals["finish_reason_counts"])
    artifact_created_at = latest_runtime.get("artifact_created_at")
    if isinstance(artifact_created_at, str) and artifact_created_at.strip():
        payload["last_subagent_runtime_snapshot_at"] = artifact_created_at.strip()
    return payload


def _goal_generation_refresh_budget_payload(run_policy: dict[str, Any] | None) -> dict[str, Any]:
    normalized = dict(run_policy or {})
    generation_refresh_interval_sec = (
        int(normalized.get("generation_refresh_interval_sec"))
        if isinstance(normalized.get("generation_refresh_interval_sec"), int)
        else None
    )
    max_wall_clock_sec = (
        int(normalized.get("max_wall_clock_sec"))
        if isinstance(normalized.get("max_wall_clock_sec"), int)
        else None
    )
    if generation_refresh_interval_sec is None:
        return {}
    if max_wall_clock_sec is not None and max_wall_clock_sec <= generation_refresh_interval_sec:
        return {
            "generation_refresh_interval_sec": generation_refresh_interval_sec,
            "effective_max_wall_clock_sec": max_wall_clock_sec,
            "owned_by_generation_refresh": False,
        }
    return {
        "generation_refresh_interval_sec": generation_refresh_interval_sec,
        "effective_max_wall_clock_sec": generation_refresh_interval_sec,
        "owned_by_generation_refresh": True,
    }


def _goal_checkpoint_cadence_budget_payload(run_policy: dict[str, Any] | None) -> dict[str, Any]:
    normalized = dict(run_policy or {})
    checkpoint_interval_sec = (
        int(normalized.get("checkpoint_interval_sec"))
        if isinstance(normalized.get("checkpoint_interval_sec"), int)
        else None
    )
    if checkpoint_interval_sec is None:
        return {}
    generation_refresh_interval_sec = (
        int(normalized.get("generation_refresh_interval_sec"))
        if isinstance(normalized.get("generation_refresh_interval_sec"), int)
        else None
    )
    if generation_refresh_interval_sec is not None:
        return {
            "checkpoint_interval_sec": checkpoint_interval_sec,
            "owned_by_checkpoint_cadence": False,
            "skipped_for_generation_refresh": True,
        }
    max_wall_clock_sec = (
        int(normalized.get("max_wall_clock_sec"))
        if isinstance(normalized.get("max_wall_clock_sec"), int)
        else None
    )
    if max_wall_clock_sec is not None and max_wall_clock_sec <= checkpoint_interval_sec:
        return {
            "checkpoint_interval_sec": checkpoint_interval_sec,
            "effective_max_wall_clock_sec": max_wall_clock_sec,
            "owned_by_checkpoint_cadence": False,
        }
    return {
        "checkpoint_interval_sec": checkpoint_interval_sec,
        "effective_max_wall_clock_sec": checkpoint_interval_sec,
        "owned_by_checkpoint_cadence": True,
    }


def _goal_checkpoint_cadence_step_payload(run_policy: dict[str, Any] | None) -> dict[str, Any]:
    normalized = dict(run_policy or {})
    checkpoint_interval_steps = (
        int(normalized.get("checkpoint_interval_steps"))
        if isinstance(normalized.get("checkpoint_interval_steps"), int)
        else None
    )
    if checkpoint_interval_steps is None or checkpoint_interval_steps <= 0:
        return {}
    generation_refresh_interval_sec = (
        int(normalized.get("generation_refresh_interval_sec"))
        if isinstance(normalized.get("generation_refresh_interval_sec"), int)
        else None
    )
    if generation_refresh_interval_sec is not None:
        return {
            "checkpoint_interval_steps": checkpoint_interval_steps,
            "owned_by_checkpoint_cadence": False,
            "skipped_for_generation_refresh": True,
        }
    checkpoint_interval_sec = (
        int(normalized.get("checkpoint_interval_sec"))
        if isinstance(normalized.get("checkpoint_interval_sec"), int)
        else None
    )
    if checkpoint_interval_sec is not None:
        return {
            "checkpoint_interval_steps": checkpoint_interval_steps,
            "owned_by_checkpoint_cadence": False,
            "skipped_for_checkpoint_interval_sec": True,
        }
    return {
        "checkpoint_interval_steps": checkpoint_interval_steps,
        "owned_by_checkpoint_cadence": True,
    }


def _goal_linked_agent_run_effective_run_policy(run_policy: dict[str, Any] | None) -> dict[str, Any]:
    normalized = dict(run_policy or {})
    if bool(normalized.get("goal_generation_refresh_owned")) or bool(
        normalized.get("goal_checkpoint_cadence_owned")
    ):
        return normalized
    refresh_budget = _goal_generation_refresh_budget_payload(normalized)
    if not bool(refresh_budget.get("owned_by_generation_refresh")):
        checkpoint_budget = _goal_checkpoint_cadence_budget_payload(normalized)
        if bool(checkpoint_budget.get("owned_by_checkpoint_cadence")):
            effective_max_wall_clock_sec = checkpoint_budget.get("effective_max_wall_clock_sec")
            if not isinstance(effective_max_wall_clock_sec, int):
                return normalized
            normalized["max_wall_clock_sec"] = effective_max_wall_clock_sec
            normalized["on_budget_exhausted"] = "finalize_partial"
            normalized["goal_checkpoint_cadence_owned"] = True
            normalized["goal_checkpoint_cadence_effective_sec"] = effective_max_wall_clock_sec
            return normalized
        checkpoint_step_budget = _goal_checkpoint_cadence_step_payload(normalized)
        if not bool(checkpoint_step_budget.get("owned_by_checkpoint_cadence")):
            return normalized
        effective_steps = checkpoint_step_budget.get("checkpoint_interval_steps")
        if not isinstance(effective_steps, int):
            return normalized
        normalized["on_budget_exhausted"] = "finalize_partial"
        normalized["goal_checkpoint_cadence_owned"] = True
        normalized["goal_checkpoint_cadence_effective_steps"] = effective_steps
        return normalized
    effective_max_wall_clock_sec = refresh_budget.get("effective_max_wall_clock_sec")
    if not isinstance(effective_max_wall_clock_sec, int):
        return normalized
    normalized["max_wall_clock_sec"] = effective_max_wall_clock_sec
    normalized["on_budget_exhausted"] = "finalize_partial"
    normalized["goal_generation_refresh_owned"] = True
    normalized["goal_generation_refresh_effective_sec"] = effective_max_wall_clock_sec
    return normalized


def _goal_planned_refresh_resume_source(run_policy: dict[str, Any] | None) -> str | None:
    effective_run_policy = _goal_linked_agent_run_effective_run_policy(run_policy)
    if bool(effective_run_policy.get("goal_generation_refresh_owned")):
        return "goal_generation_refresh"
    if bool(effective_run_policy.get("goal_checkpoint_cadence_owned")):
        return "goal_checkpoint_cadence_refresh"
    return None


def _goal_should_autoresume_planned_refresh(
    run: dict[str, Any],
    *,
    run_policy: dict[str, Any] | None = None,
) -> bool:
    if str(run.get("status") or "").strip() != "partial":
        return False
    refresh_resume_source = _goal_planned_refresh_resume_source(run_policy)
    if refresh_resume_source is None:
        return False
    effective_run_policy = _goal_linked_agent_run_effective_run_policy(run_policy)
    recovery_state = _goal_recovery_state_from_agent_run(run)
    if not isinstance(recovery_state, dict):
        return False
    if str(recovery_state.get("status") or "").strip() != "partial":
        return False
    if str(recovery_state.get("action") or "").strip() != "finalize_partial":
        return False
    reason = str(recovery_state.get("reason") or "").strip()
    prefix = "Run exceeded max_wall_clock_sec="
    if reason.startswith(prefix):
        expected = (
            effective_run_policy.get("goal_generation_refresh_effective_sec")
            if refresh_resume_source == "goal_generation_refresh"
            else effective_run_policy.get("goal_checkpoint_cadence_effective_sec")
        )
        if not isinstance(expected, int):
            return False
        reported = reason.removeprefix(prefix).rstrip(".")
        try:
            return int(reported) == expected
        except ValueError:
            return False
    checkpoint_prefix = "Run reached checkpoint_interval_steps="
    if not reason.startswith(checkpoint_prefix):
        return False
    expected_steps = effective_run_policy.get("goal_checkpoint_cadence_effective_steps")
    if not isinstance(expected_steps, int):
        return False
    reported_steps = reason.removeprefix(checkpoint_prefix).split(" ", 1)[0]
    try:
        return int(reported_steps) == expected_steps
    except ValueError:
        return False


def _goal_worker_generation_response(
    record: dict[str, Any] | None,
    *,
    run_policy: dict[str, Any] | None = None,
    context_handoff: dict[str, Any] | None = None,
    runtime_telemetry: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not isinstance(record, dict):
        return None
    refresh_policy = _goal_worker_generation_refresh_policy_payload(
        record,
        run_policy=run_policy,
    )
    context_handoff_payload = dict(context_handoff or {})
    runtime_telemetry_payload = dict(runtime_telemetry or {})
    started_at = record.get("started_at")
    payload = {
        "generation_id": record["id"],
        "goal_id": record.get("goal_id"),
        "attempt_id": record.get("attempt_id"),
        "agent_run_id": record.get("agent_run_id"),
        "generation_index": record.get("generation_index"),
        "status": record.get("status"),
        "rollover_reason": record.get("rollover_reason"),
        "parent_generation_id": record.get("parent_generation_id"),
        "resume_source_snapshot_id": record.get("resume_source_snapshot_id"),
        "metadata": dict(record.get("metadata") or {}),
        "started_at": started_at,
        "finished_at": record.get("finished_at"),
        "created_at": record.get("created_at"),
        "updated_at": record.get("updated_at"),
        **refresh_policy,
        **context_handoff_payload,
        **runtime_telemetry_payload,
    }
    return {
        key: value
        for key, value in payload.items()
        if value is not None
    }


def _goal_memory_snapshot_handoff_metadata(
    *,
    snapshot_record: dict[str, Any],
    checkpoint_record: dict[str, Any] | None,
) -> dict[str, Any]:
    snapshot = dict(snapshot_record.get("snapshot") or {})
    checkpoint = (
        dict(checkpoint_record.get("payload") or {})
        if isinstance(checkpoint_record, dict)
        else {}
    )
    checkpoint_promotion = (
        dict(checkpoint.get("promotion"))
        if isinstance(checkpoint.get("promotion"), dict)
        else {}
    )
    promoted_artifacts = [
        dict(item)
        for item in (
            snapshot.get("promoted_artifacts")
            if isinstance(snapshot.get("promoted_artifacts"), list)
            else checkpoint_promotion.get("promoted_artifacts", [])
        )
        if isinstance(item, dict)
    ]
    return {
        key: value
        for key, value in {
            "memory_snapshot_id": snapshot_record.get("id"),
            "memory_snapshot_kind": snapshot_record.get("snapshot_kind"),
            "memory_snapshot_captured_at": snapshot_record.get("captured_at"),
            "checkpoint_id": snapshot_record.get("checkpoint_id"),
            "checkpoint_index": snapshot.get("checkpoint_index") or checkpoint.get("checkpoint_index"),
            "stage": snapshot.get("stage") or checkpoint.get("stage"),
            "checkpoint_promotion_mode": (
                snapshot.get("checkpoint_promotion_mode") or checkpoint_promotion.get("mode")
            ),
            "promoted_artifacts": promoted_artifacts,
            "goal_objective": snapshot.get("goal_objective"),
            "unfinished_steps": list(snapshot.get("unfinished_steps") or []),
            "pending_actions": list(snapshot.get("pending_actions") or []),
            "pending_approval_ids": list(snapshot.get("pending_approval_ids") or []),
        }.items()
        if value not in (None, "", [], {})
    }


def _goal_memory_snapshot_guidance_messages(
    *,
    snapshot_record: dict[str, Any],
    checkpoint_record: dict[str, Any] | None,
) -> list[str]:
    snapshot = dict(snapshot_record.get("snapshot") or {})
    checkpoint = (
        dict(checkpoint_record.get("payload") or {})
        if isinstance(checkpoint_record, dict)
        else {}
    )
    checkpoint_promotion = (
        dict(checkpoint.get("promotion"))
        if isinstance(checkpoint.get("promotion"), dict)
        else {}
    )
    messages = [
        "Resume from the latest durable goal memory snapshot instead of replaying the full transcript.",
    ]
    goal_objective = snapshot.get("goal_objective")
    if isinstance(goal_objective, str) and goal_objective.strip():
        messages.append(f"Goal objective: {goal_objective.strip()}")
    stage = snapshot.get("stage") or checkpoint.get("stage")
    checkpoint_index = snapshot.get("checkpoint_index") or checkpoint.get("checkpoint_index")
    stage_parts: list[str] = []
    if stage not in (None, ""):
        stage_parts.append(f"stage={stage}")
    if checkpoint_index not in (None, ""):
        stage_parts.append(f"checkpoint_index={checkpoint_index}")
    if stage_parts:
        messages.append(f"Latest durable handoff point: {', '.join(stage_parts)}")
    unfinished_steps = [str(item).strip() for item in snapshot.get("unfinished_steps", []) if str(item).strip()]
    if unfinished_steps:
        messages.append(f"Unfinished steps: {'; '.join(unfinished_steps)}")
    pending_actions = [str(item).strip() for item in snapshot.get("pending_actions", []) if str(item).strip()]
    if pending_actions:
        messages.append(f"Pending actions: {'; '.join(pending_actions[:4])}")
    pending_approval_ids = [
        str(item).strip()
        for item in snapshot.get("pending_approval_ids", [])
        if str(item).strip()
    ]
    if pending_approval_ids:
        messages.append(f"Pending approvals to account for: {', '.join(pending_approval_ids)}")
    promoted_artifacts = [
        dict(item)
        for item in (
            snapshot.get("promoted_artifacts")
            if isinstance(snapshot.get("promoted_artifacts"), list)
            else checkpoint_promotion.get("promoted_artifacts", [])
        )
        if isinstance(item, dict)
    ]
    if promoted_artifacts:
        labels: list[str] = []
        for artifact in promoted_artifacts[:4]:
            title = str(artifact.get("title") or "").strip()
            artifact_type = str(artifact.get("artifact_type") or "").strip()
            if title and artifact_type:
                labels.append(f"{artifact_type}:{title}")
            elif title:
                labels.append(title)
            elif artifact_type:
                labels.append(artifact_type)
        if labels:
            messages.append(f"Promoted artifacts: {', '.join(labels)}")
    important_artifacts = [
        dict(item)
        for item in snapshot.get("important_artifacts", [])
        if isinstance(item, dict)
    ]
    if important_artifacts:
        labels: list[str] = []
        for artifact in important_artifacts[:3]:
            title = str(artifact.get("title") or "").strip()
            artifact_type = str(artifact.get("artifact_type") or "").strip()
            if title and artifact_type:
                labels.append(f"{artifact_type}:{title}")
            elif title:
                labels.append(title)
            elif artifact_type:
                labels.append(artifact_type)
        if labels:
            messages.append(f"Important artifacts: {', '.join(labels)}")
    accepted_facts = [str(item).strip() for item in snapshot.get("accepted_facts", []) if str(item).strip()]
    if accepted_facts:
        messages.append(f"Accepted facts: {'; '.join(accepted_facts[:3])}")
    rejected_paths = [str(item).strip() for item in snapshot.get("rejected_paths", []) if str(item).strip()]
    if rejected_paths:
        messages.append(f"Rejected paths: {'; '.join(rejected_paths[:3])}")
    collector_shard_offsets = [
        dict(item)
        for item in snapshot.get("collector_shard_offsets", [])
        if isinstance(item, dict)
    ]
    if collector_shard_offsets:
        shard_labels: list[str] = []
        for shard in collector_shard_offsets[:3]:
            shard_id = str(shard.get("shard_id") or "unknown").strip()
            status = str(shard.get("status") or "pending").strip()
            cursor = str(shard.get("cursor") or "").strip()
            label = f"{shard_id}:{status}"
            if cursor:
                label = f"{label} cursor={cursor}"
            shard_labels.append(label)
        messages.append(f"Collector shards: {'; '.join(shard_labels)}")
    return _merge_unique_text_items([], messages)


def _inject_goal_memory_snapshot_into_resume_payload(
    payload: dict[str, Any],
    *,
    handoff_context: dict[str, Any],
) -> dict[str, Any]:
    updated = dict(payload or {})
    guidance_messages = [
        str(item).strip()
        for item in updated.get("guidance_messages", [])
        if str(item).strip()
    ]
    updated["guidance_messages"] = _merge_unique_text_items(
        guidance_messages,
        [
            str(item).strip()
            for item in handoff_context.get("guidance_messages", [])
            if isinstance(item, str) and item.strip()
        ],
    )
    metadata_state = (
        dict(updated.get("metadata_state"))
        if isinstance(updated.get("metadata_state"), dict)
        else {}
    )
    metadata_state["goal_handoff"] = dict(handoff_context.get("metadata") or {})
    updated["metadata_state"] = metadata_state
    return updated


def _restart_attempt_resume_payload(payload: dict[str, Any]) -> dict[str, Any]:
    updated = dict(payload or {})
    updated["executor"] = "restart_attempt"
    updated["strategy_default"] = "restart_attempt"
    updated["supported_actions"] = ["restart_attempt"]
    return updated


def _goal_linked_durable_handoff_gate(
    *,
    run: Mapping[str, Any],
    attempt_id: str | None,
    handoff_context: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = {
        "usable": True,
        "decision": "allow_durable_handoff",
    }
    snapshot_record = (
        handoff_context.get("snapshot_record")
        if isinstance(handoff_context, dict) and isinstance(handoff_context.get("snapshot_record"), dict)
        else None
    )
    checkpoint_record = (
        handoff_context.get("checkpoint_record")
        if isinstance(handoff_context, dict) and isinstance(handoff_context.get("checkpoint_record"), dict)
        else None
    )
    if not isinstance(snapshot_record, dict):
        payload["usable"] = False
        payload["decision"] = "reject_durable_handoff"
        payload["reason"] = "missing_memory_snapshot"
        return payload
    payload["memory_snapshot_id"] = snapshot_record.get("id")
    snapshot_attempt_id = (
        str(snapshot_record.get("attempt_id") or "").strip()
        if isinstance(snapshot_record.get("attempt_id"), str)
        else ""
    )
    if attempt_id is not None and snapshot_attempt_id != attempt_id:
        payload["usable"] = False
        payload["decision"] = "reject_durable_handoff"
        payload["reason"] = "snapshot_attempt_mismatch"
        payload["snapshot_attempt_id"] = snapshot_attempt_id or None
        return payload
    if not isinstance(checkpoint_record, dict):
        payload["usable"] = False
        payload["decision"] = "reject_durable_handoff"
        payload["reason"] = "missing_checkpoint_record"
        return payload
    payload["checkpoint_id"] = checkpoint_record.get("id")
    checkpoint_attempt_id = (
        str(checkpoint_record.get("attempt_id") or "").strip()
        if isinstance(checkpoint_record.get("attempt_id"), str)
        else ""
    )
    if attempt_id is not None and checkpoint_attempt_id != attempt_id:
        payload["usable"] = False
        payload["decision"] = "reject_durable_handoff"
        payload["reason"] = "checkpoint_attempt_mismatch"
        payload["checkpoint_attempt_id"] = checkpoint_attempt_id or None
        return payload
    recovery_state = _recovery_state_from_run(dict(run))
    current_checkpoint = (
        dict(recovery_state.get("checkpoint"))
        if isinstance(recovery_state.get("checkpoint"), dict)
        else {}
    )
    checkpoint_payload = (
        dict(checkpoint_record.get("payload"))
        if isinstance(checkpoint_record.get("payload"), dict)
        else {}
    )
    current_checkpoint_index = _parse_optional_positive_int(current_checkpoint.get("checkpoint_index"))
    durable_checkpoint_index = _parse_optional_positive_int(
        checkpoint_payload.get("checkpoint_index") or checkpoint_record.get("checkpoint_index")
    )
    if current_checkpoint_index is not None:
        payload["run_checkpoint_index"] = current_checkpoint_index
    if durable_checkpoint_index is not None:
        payload["durable_checkpoint_index"] = durable_checkpoint_index
    current_captured_at = (
        str(current_checkpoint.get("captured_at") or "").strip()
        if isinstance(current_checkpoint.get("captured_at"), str)
        else ""
    )
    durable_captured_at = (
        str(checkpoint_payload.get("captured_at") or checkpoint_record.get("captured_at") or "").strip()
        if isinstance(checkpoint_payload.get("captured_at") or checkpoint_record.get("captured_at"), str)
        else ""
    )
    current_captured_at_dt = _parse_iso_datetime(current_captured_at)
    durable_captured_at_dt = _parse_iso_datetime(durable_captured_at)
    if current_checkpoint_index is not None and (
        durable_checkpoint_index is None or durable_checkpoint_index < current_checkpoint_index
    ):
        payload["usable"] = False
        payload["decision"] = "reject_durable_handoff"
        payload["reason"] = "stale_checkpoint_record"
        return payload
    if (
        current_captured_at_dt is not None
        and durable_captured_at_dt is not None
        and durable_captured_at_dt < current_captured_at_dt
    ):
        payload["usable"] = False
        payload["decision"] = "reject_durable_handoff"
        payload["reason"] = "stale_checkpoint_record"
        payload["run_checkpoint_captured_at"] = current_captured_at
        payload["durable_checkpoint_captured_at"] = durable_captured_at
        return payload
    return payload


def _goal_linked_resume_gate(
    *,
    run: Mapping[str, Any],
    attempt_id: str | None,
    handoff_context: dict[str, Any] | None,
    has_structured_resume_payload: bool,
    payload_strategy: str,
) -> dict[str, Any]:
    payload = {
        "requested_strategy": "continue_from_checkpoint",
        "effective_strategy": "continue_from_checkpoint",
        "decision": "allow_continue_from_checkpoint",
    }
    if not has_structured_resume_payload:
        payload["effective_strategy"] = "restart_attempt"
        payload["decision"] = "fallback_restart_attempt"
        payload["reason"] = "missing_resume_payload"
        return payload
    if _normalize_agent_run_resume_strategy(payload_strategy) != "continue_from_checkpoint":
        payload["effective_strategy"] = "restart_attempt"
        payload["decision"] = "fallback_restart_attempt"
        payload["reason"] = "resume_payload_executor_restart_attempt"
        return payload
    durable_handoff_gate = _goal_linked_durable_handoff_gate(
        run=run,
        attempt_id=attempt_id,
        handoff_context=handoff_context,
    )
    payload.update(
        {
            key: value
            for key, value in durable_handoff_gate.items()
            if key not in {"usable", "decision"} and value is not None
        }
    )
    if not bool(durable_handoff_gate.get("usable")):
        payload["effective_strategy"] = "restart_attempt"
        payload["decision"] = "fallback_restart_attempt"
        payload["reason"] = durable_handoff_gate.get("reason") or "missing_memory_snapshot"
        return payload
    return payload


def _goal_linked_recovery_state_from_handoff(
    *,
    recovery_state: Mapping[str, Any] | None,
    handoff_context: dict[str, Any] | None,
) -> dict[str, Any]:
    current = dict(recovery_state or {})
    checkpoint_record = (
        handoff_context.get("checkpoint_record")
        if isinstance(handoff_context, dict) and isinstance(handoff_context.get("checkpoint_record"), dict)
        else None
    )
    checkpoint_payload = (
        dict(checkpoint_record.get("payload") or {})
        if isinstance(checkpoint_record, dict)
        else {}
    )
    checkpoint_recovery_state = (
        dict(checkpoint_payload.get("recovery_state"))
        if isinstance(checkpoint_payload.get("recovery_state"), dict)
        else {}
    )
    snapshot_record = (
        handoff_context.get("snapshot_record")
        if isinstance(handoff_context, dict) and isinstance(handoff_context.get("snapshot_record"), dict)
        else None
    )
    snapshot = (
        dict(snapshot_record.get("snapshot") or {})
        if isinstance(snapshot_record, dict)
        else {}
    )
    merged = dict(checkpoint_recovery_state)
    for key, value in current.items():
        if value in (None, "", [], {}):
            continue
        if (
            key in {"checkpoint", "approval_state"}
            and isinstance(merged.get(key), dict)
            and isinstance(value, Mapping)
        ):
            merged[key] = {**dict(merged.get(key) or {}), **dict(value)}
            continue
        merged[key] = value
    checkpoint = (
        dict(merged.get("checkpoint"))
        if isinstance(merged.get("checkpoint"), dict)
        else {}
    )
    if not checkpoint and isinstance(checkpoint_payload.get("checkpoint"), dict):
        checkpoint = dict(checkpoint_payload.get("checkpoint"))
    if checkpoint_payload.get("checkpoint_index") not in (None, ""):
        checkpoint.setdefault("checkpoint_index", checkpoint_payload.get("checkpoint_index"))
    if checkpoint_payload.get("stage") not in (None, ""):
        checkpoint.setdefault("stage", checkpoint_payload.get("stage"))
    if checkpoint_payload.get("captured_at") not in (None, ""):
        checkpoint.setdefault("captured_at", checkpoint_payload.get("captured_at"))
    if checkpoint:
        merged["checkpoint"] = checkpoint
    if merged.get("stage") in (None, ""):
        merged["stage"] = checkpoint.get("stage") or snapshot.get("stage")
    if merged.get("checkpoint_index") in (None, ""):
        merged["checkpoint_index"] = checkpoint.get("checkpoint_index") or snapshot.get(
            "checkpoint_index"
        )
    if merged.get("unfinished_steps") in (None, [], ()):
        merged["unfinished_steps"] = list(snapshot.get("unfinished_steps") or [])
    if merged.get("recommended_resume_conditions") in (None, [], ()):
        merged["recommended_resume_conditions"] = list(
            snapshot.get("recommended_resume_conditions") or []
        )
    if not isinstance(merged.get("role_task_summary"), dict) and isinstance(
        snapshot.get("role_task_summary"), dict
    ):
        merged["role_task_summary"] = dict(snapshot.get("role_task_summary") or {})
    return {key: value for key, value in merged.items() if value not in (None, "", [], {})}


def _goal_linked_resume_payload_from_handoff(
    *,
    run: dict[str, Any],
    recovery_state: Mapping[str, Any] | None,
    handoff_context: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    merged_recovery_state = _goal_linked_recovery_state_from_handoff(
        recovery_state=recovery_state,
        handoff_context=handoff_context,
    )
    if not merged_recovery_state:
        return None
    existing_payload = (
        dict(merged_recovery_state.get("resume_payload"))
        if isinstance(merged_recovery_state.get("resume_payload"), dict)
        else {}
    )
    resume_payload = (
        existing_payload
        if (
            _is_structured_agent_run_resume_payload(existing_payload)
            and _normalize_agent_run_resume_strategy(str(existing_payload.get("executor") or ""))
            == "continue_from_checkpoint"
        )
        else _build_agent_run_resume_payload(
            run=run,
            recovery_state=merged_recovery_state,
            strategy="continue_from_checkpoint",
        )
    )
    if not _is_structured_agent_run_resume_payload(resume_payload):
        return None
    merged_recovery_state["resume_payload"] = dict(resume_payload)
    merged_recovery_state["resume_executor"] = str(resume_payload.get("executor") or "").strip()
    return dict(resume_payload), merged_recovery_state


def _goal_worker_generation_rollover_reason(
    *,
    has_previous_generation: bool,
    active_resume_runtime: dict[str, Any],
    handoff_context: dict[str, Any] | None,
) -> str:
    if not has_previous_generation:
        return "initial_start"
    resume_source = str(active_resume_runtime.get("source") or "").strip().lower()
    resume_strategy = str(active_resume_runtime.get("strategy") or "").strip().lower()
    snapshot_record = (
        handoff_context.get("snapshot_record")
        if isinstance(handoff_context, dict) and isinstance(handoff_context.get("snapshot_record"), dict)
        else None
    )
    if (
        resume_source == "goal_context_handoff_refresh"
        and resume_strategy in {"continue_from_checkpoint", "restart_attempt"}
        and snapshot_record is not None
    ):
        return "context_pressure"
    if (
        resume_source == "goal_token_refresh"
        and resume_strategy in {"continue_from_checkpoint", "restart_attempt"}
        and snapshot_record is not None
    ):
        return "context_pressure"
    if (
        resume_source == "goal_worker_stall_refresh"
        and resume_strategy in {"continue_from_checkpoint", "restart_attempt"}
        and snapshot_record is not None
    ):
        return "worker_stall"
    if (
        resume_source in {"goal_generation_refresh", "goal_checkpoint_cadence_refresh"}
        and snapshot_record is not None
    ):
        return "scheduled_refresh"
    if snapshot_record is not None and resume_source in {"manual_resume", "manual_start"}:
        return "manual_refresh"
    if resume_strategy == "continue_from_checkpoint":
        return "runtime_restart"
    if snapshot_record is not None:
        return "scheduled_refresh"
    return "runtime_restart"


def _goal_worker_generation_status_from_run_status(status: str) -> str:
    normalized = str(status or "running").strip().lower()
    if normalized == "succeeded":
        return "completed"
    if normalized == "awaiting_approval":
        return "awaiting_approval"
    if normalized == "awaiting_resources":
        return "awaiting_resources"
    if normalized in {"partial", "stalled", "paused", "cancelled", "failed", "running", "completed"}:
        return normalized
    return normalized or "running"


def _build_goal_checkpoint_record_payload(
    *,
    attempt_id: str,
    run: dict[str, Any],
    run_policy: dict[str, Any] | None,
    linked_recovery_state: dict[str, Any],
    linked_approval_state: dict[str, Any],
) -> dict[str, Any] | None:
    checkpoint = (
        dict(linked_recovery_state.get("checkpoint"))
        if isinstance(linked_recovery_state.get("checkpoint"), dict)
        else {}
    )
    checkpoint_index = checkpoint.get("checkpoint_index") or linked_recovery_state.get("checkpoint_index")
    stage = (
        checkpoint.get("stage")
        or linked_recovery_state.get("stage")
        or str(run.get("status") or "").strip()
    )
    collector_state = _goal_collector_state_payload(
        run,
        attempt_id=attempt_id,
        run_policy=run_policy,
    )
    collector_shard_offsets = (
        list(collector_state.get("shards") or [])
        if isinstance(collector_state, dict)
        else []
    )
    if (
        not checkpoint
        and not linked_recovery_state
        and not linked_approval_state
        and not collector_shard_offsets
    ):
        return None
    captured_at = (
        str(checkpoint.get("captured_at")).strip()
        if isinstance(checkpoint.get("captured_at"), str) and checkpoint.get("captured_at").strip()
        else str(run.get("updated_at") or run.get("created_at") or _utcnow().isoformat())
    )
    promotion = _goal_checkpoint_promotion_payload(run)
    payload = {
        "checkpoint_index": checkpoint_index,
        "stage": stage,
        "checkpoint": checkpoint,
        "recovery_state": dict(linked_recovery_state or {}),
        "approval_state": dict(linked_approval_state or {}),
        "collector_state": collector_state,
        "collector_shard_offsets": collector_shard_offsets,
        "promotion": promotion,
        "agent_run_status": run.get("status"),
        "captured_at": captured_at,
    }
    return {
        key: value
        for key, value in payload.items()
        if value not in (None, [], {}) and value != ""
    }


def _build_goal_memory_snapshot_payload(
    *,
    goal: dict[str, Any] | None,
    attempt_id: str,
    run: dict[str, Any],
    run_policy: dict[str, Any] | None,
    linked_recovery_state: dict[str, Any],
    linked_approval_state: dict[str, Any],
    checkpoint_record: dict[str, Any] | None,
) -> dict[str, Any] | None:
    collector_state = _goal_collector_state_payload(
        run,
        attempt_id=attempt_id,
        run_policy=run_policy,
    )
    collector_shard_offsets = (
        list(collector_state.get("shards") or [])
        if isinstance(collector_state, dict)
        else []
    )
    if (
        not linked_recovery_state
        and not linked_approval_state
        and checkpoint_record is None
        and not collector_shard_offsets
    ):
        return None
    summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
    checkpoint_payload = (
        dict(checkpoint_record.get("payload"))
        if isinstance(checkpoint_record, dict) and isinstance(checkpoint_record.get("payload"), dict)
        else _build_goal_checkpoint_record_payload(
            attempt_id=attempt_id,
            run=run,
            run_policy=run_policy,
            linked_recovery_state=linked_recovery_state,
            linked_approval_state=linked_approval_state,
        )
        or {}
    )
    checkpoint_promotion = (
        dict(checkpoint_payload.get("promotion"))
        if isinstance(checkpoint_payload.get("promotion"), dict)
        else {}
    )
    promoted_artifacts = [
        dict(item)
        for item in checkpoint_promotion.get("promoted_artifacts", [])
        if isinstance(item, dict)
    ]
    captured_at = (
        str(checkpoint_payload.get("captured_at")).strip()
        if isinstance(checkpoint_payload.get("captured_at"), str)
        and str(checkpoint_payload.get("captured_at")).strip()
        else str(run.get("updated_at") or run.get("created_at") or _utcnow().isoformat())
    )
    goal_objective = (
        str(goal.get("objective") or "").strip()
        if isinstance(goal, dict) and isinstance(goal.get("objective"), str)
        else ""
    )
    final_answer_preview = _truncate_preview(
        summary.get("final_answer") if isinstance(summary.get("final_answer"), str) else None,
        max_chars=220,
    )
    accepted_facts = _goal_memory_snapshot_accepted_facts(run)
    rejected_paths = _goal_memory_snapshot_rejected_paths(run)
    pending_actions = _goal_memory_snapshot_pending_actions(
        linked_recovery_state=linked_recovery_state,
        linked_approval_state=linked_approval_state,
    )
    snapshot = {
        "goal_objective": goal_objective or None,
        "attempt_id": attempt_id,
        "agent_run_id": run.get("id"),
        "protocol_id": run.get("protocol_id"),
        "agent_run_status": run.get("status"),
        "stage": checkpoint_payload.get("stage") or linked_recovery_state.get("stage"),
        "checkpoint_index": checkpoint_payload.get("checkpoint_index"),
        "selected_candidate_id": linked_recovery_state.get("selected_candidate_id"),
        "candidate_count": linked_recovery_state.get("candidate_count"),
        "pending_approval_ids": list(linked_approval_state.get("approval_ids") or []),
        "unfinished_steps": list(linked_recovery_state.get("unfinished_steps") or []),
        "pending_actions": pending_actions,
        "recommended_resume_conditions": list(
            linked_recovery_state.get("recommended_resume_conditions") or []
        ),
        "checkpoint_promotion_mode": checkpoint_promotion.get("mode"),
        "promoted_artifacts": promoted_artifacts,
        "collector_state": collector_state,
        "collector_shard_offsets": collector_shard_offsets,
        "role_task_summary": (
            dict(linked_recovery_state.get("role_task_summary"))
            if isinstance(linked_recovery_state.get("role_task_summary"), dict)
            else None
        ),
        "accepted_facts": accepted_facts,
        "rejected_paths": rejected_paths,
        "final_answer_preview": final_answer_preview,
        "important_artifacts": _goal_memory_snapshot_artifact_refs(run),
        "captured_at": captured_at,
    }
    return {
        key: value
        for key, value in snapshot.items()
        if value not in (None, [], {}) and value != ""
    }


def _goal_collector_offsets_signature_payload(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    payloads: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        payloads.append(
            {
                key: item.get(key)
                for key in (
                    "shard_id",
                    "attempt_id",
                    "status",
                    "adapter_name",
                    "source_id",
                    "source_url",
                    "cursor",
                    "items_collected",
                    "items_emitted",
                    "last_activity_at",
                    "stalled_age_sec",
                )
                if item.get(key) not in (None, "")
            }
        )
    return sorted(
        payloads,
        key=lambda item: (
            str(item.get("shard_id") or ""),
            str(item.get("attempt_id") or ""),
            str(item.get("source_id") or ""),
            str(item.get("status") or ""),
            str(item.get("cursor") or ""),
        ),
    )


def _goal_collector_status_counts_signature_payload(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    payloads: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        payloads.append(
            {
                "value": item.get("value"),
                "count": item.get("count"),
            }
        )
    return sorted(
        payloads,
        key=lambda item: (
            str(item.get("value") or ""),
            int(item.get("count") or 0),
        ),
    )


def _goal_artifact_refs_signature_payload(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    payloads: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        payloads.append(
            {
                key: item.get(key)
                for key in (
                    "artifact_type",
                    "title",
                    "uri",
                    "mime_type",
                    "promotion_target",
                )
                if item.get(key) not in (None, "")
            }
        )
    return sorted(
        payloads,
        key=lambda item: (
            str(item.get("artifact_type") or ""),
            str(item.get("uri") or ""),
            str(item.get("title") or ""),
        ),
    )


def _goal_collector_progress_signature_fragment(
    collector_state: Any,
    collector_shard_offsets: Any,
) -> str:
    state_payload: dict[str, Any] = {}
    if isinstance(collector_state, dict):
        state_payload = {
            key: collector_state.get(key)
            for key in (
                "shard_count",
                "active_shard_count",
                "completed_shard_count",
                "failed_shard_count",
                "stalled_shard_count",
                "stall_timeout_sec",
                "latest_activity_at",
            )
            if collector_state.get(key) not in (None, "")
        }
        status_counts = _goal_collector_status_counts_signature_payload(
            collector_state.get("status_counts"),
        )
        if status_counts:
            state_payload["status_counts"] = status_counts
        shards = _goal_collector_offsets_signature_payload(collector_state.get("shards"))
        if shards:
            state_payload["shards"] = shards
        stalled_shards = _goal_collector_offsets_signature_payload(
            collector_state.get("stalled_shards"),
        )
        if stalled_shards:
            state_payload["stalled_shards"] = stalled_shards
    payload = {
        "collector_state": state_payload,
        "collector_shard_offsets": _goal_collector_offsets_signature_payload(
            collector_shard_offsets,
        ),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _goal_checkpoint_record_signature(payload: dict[str, Any]) -> str:
    approval_state = payload.get("approval_state") if isinstance(payload.get("approval_state"), dict) else {}
    recovery_state = payload.get("recovery_state") if isinstance(payload.get("recovery_state"), dict) else {}
    promotion = payload.get("promotion") if isinstance(payload.get("promotion"), dict) else {}
    parts = [
        str(payload.get("checkpoint_index") or ""),
        str(payload.get("stage") or ""),
        str(payload.get("agent_run_status") or ""),
        str(recovery_state.get("status") or ""),
        str(recovery_state.get("action") or ""),
        str(recovery_state.get("selected_candidate_id") or ""),
        str(recovery_state.get("candidate_count") or ""),
        "|".join(sorted(str(item).strip() for item in approval_state.get("approval_ids", []) if str(item).strip())),
        "|".join(str(item).strip() for item in recovery_state.get("unfinished_steps", []) if str(item).strip()),
        _goal_progress_signature_fragment(recovery_state.get("role_task_summary")),
        json.dumps(
            {
                "mode": promotion.get("mode"),
                "promoted_artifacts": _goal_artifact_refs_signature_payload(
                    promotion.get("promoted_artifacts")
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        _goal_collector_progress_signature_fragment(
            payload.get("collector_state"),
            payload.get("collector_shard_offsets"),
        ),
    ]
    return "::".join(parts)


def _goal_memory_snapshot_signature(snapshot: dict[str, Any]) -> str:
    parts = [
        str(snapshot.get("checkpoint_index") or ""),
        str(snapshot.get("stage") or ""),
        str(snapshot.get("agent_run_status") or ""),
        str(snapshot.get("selected_candidate_id") or ""),
        str(snapshot.get("candidate_count") or ""),
        str(snapshot.get("checkpoint_promotion_mode") or ""),
        "|".join(sorted(str(item).strip() for item in snapshot.get("pending_approval_ids", []) if str(item).strip())),
        "|".join(str(item).strip() for item in snapshot.get("unfinished_steps", []) if str(item).strip()),
        "|".join(str(item).strip() for item in snapshot.get("pending_actions", []) if str(item).strip()),
        "|".join(str(item).strip() for item in snapshot.get("accepted_facts", []) if str(item).strip()),
        "|".join(str(item).strip() for item in snapshot.get("rejected_paths", []) if str(item).strip()),
        str(snapshot.get("final_answer_preview") or ""),
        _goal_progress_signature_fragment(snapshot.get("role_task_summary")),
        json.dumps(
            {
                "promoted_artifacts": _goal_artifact_refs_signature_payload(
                    snapshot.get("promoted_artifacts")
                ),
                "important_artifacts": _goal_artifact_refs_signature_payload(
                    snapshot.get("important_artifacts")
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        _goal_collector_progress_signature_fragment(
            snapshot.get("collector_state"),
            snapshot.get("collector_shard_offsets"),
        ),
    ]
    return "::".join(parts)


def _goal_artifact_ref_payload(artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: artifact.get(key)
        for key in ("artifact_type", "title", "uri", "mime_type")
        if artifact.get(key) not in (None, "")
    }


def _goal_checkpoint_promoted_artifact_refs(
    run: dict[str, Any],
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    artifacts = run.get("artifacts") if isinstance(run.get("artifacts"), list) else []
    refs: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for artifact in reversed(artifacts):
        if not isinstance(artifact, Mapping):
            continue
        artifact_type = str(artifact.get("artifact_type") or "").strip()
        if artifact_type not in _GOAL_CHECKPOINT_PROMOTABLE_ARTIFACT_TYPES:
            continue
        ref = _goal_artifact_ref_payload(artifact)
        if not ref:
            continue
        key = (
            artifact_type,
            str(ref.get("uri") or ""),
            str(ref.get("title") or ""),
        )
        if key in seen:
            continue
        ref["promotion_target"] = "downstream_resume"
        refs.append(ref)
        seen.add(key)
        if len(refs) >= limit:
            break
    return refs


def _goal_checkpoint_promotion_payload(run: dict[str, Any]) -> dict[str, Any]:
    promoted_artifacts = _goal_checkpoint_promoted_artifact_refs(run)
    return {
        "mode": (
            "internal_checkpoint_plus_downstream_artifacts"
            if promoted_artifacts
            else "internal_checkpoint_only"
        ),
        "promoted_artifact_count": len(promoted_artifacts),
        "promoted_artifacts": promoted_artifacts,
    }


def _goal_memory_snapshot_artifact_refs(run: dict[str, Any], *, limit: int = 5) -> list[dict[str, Any]]:
    artifacts = run.get("artifacts") if isinstance(run.get("artifacts"), list) else []
    refs: list[dict[str, Any]] = []
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        refs.append(_goal_artifact_ref_payload(artifact))
        if len(refs) >= limit:
            break
    return refs


def _goal_memory_snapshot_accepted_facts(run: dict[str, Any], *, limit: int = 5) -> list[str]:
    facts: list[str] = []
    research_brief = _latest_agent_run_artifact_content(run, "research_brief") or {}
    for field_name in ("summary", "selected_candidate_summary"):
        value = research_brief.get(field_name)
        if isinstance(value, str) and value.strip():
            facts.append(value.strip())
    claim_evidence_map = _latest_agent_run_artifact_content(run, "claim_evidence_map") or {}
    claims = claim_evidence_map.get("claims") if isinstance(claim_evidence_map.get("claims"), list) else []
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        if str(claim.get("support_status") or "").strip() != "supported":
            continue
        text = str(claim.get("claim") or "").strip()
        if text:
            facts.append(text)
        if len(_merge_unique_text_items([], facts)) >= limit:
            break
    return _merge_unique_text_items([], facts)[:limit]


def _goal_memory_snapshot_rejected_paths(run: dict[str, Any], *, limit: int = 5) -> list[str]:
    rejected: list[str] = []
    claim_evidence_map = _latest_agent_run_artifact_content(run, "claim_evidence_map") or {}
    claims = claim_evidence_map.get("claims") if isinstance(claim_evidence_map.get("claims"), list) else []
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        if str(claim.get("support_status") or "").strip() != "refuted":
            continue
        text = str(claim.get("claim") or "").strip()
        if text:
            rejected.append(text)
        if len(_merge_unique_text_items([], rejected)) >= limit:
            break
    controller_decisions = _latest_agent_run_artifact_content(run, "controller_decisions") or {}
    items = controller_decisions.get("items") if isinstance(controller_decisions.get("items"), list) else []
    for item in items:
        if not isinstance(item, dict):
            continue
        if str(item.get("status") or "").strip() != "rejected":
            continue
        command = str(item.get("command") or "").strip()
        request_id = str(item.get("request_id") or "").strip()
        if command:
            rejected.append(f"Rejected command: {command}")
        elif request_id:
            rejected.append(f"Rejected controller request: {request_id}")
        if len(_merge_unique_text_items([], rejected)) >= limit:
            break
    return _merge_unique_text_items([], rejected)[:limit]


def _goal_memory_snapshot_pending_actions(
    *,
    linked_recovery_state: dict[str, Any],
    linked_approval_state: dict[str, Any],
) -> list[str]:
    actions: list[str] = [
        str(item).strip()
        for item in linked_recovery_state.get("unfinished_steps", [])
        if str(item).strip()
    ]
    role_task_summary = (
        dict(linked_recovery_state.get("role_task_summary"))
        if isinstance(linked_recovery_state.get("role_task_summary"), dict)
        else {}
    )
    roles = role_task_summary.get("roles") if isinstance(role_task_summary.get("roles"), list) else []
    for role in roles:
        if not isinstance(role, dict):
            continue
        status = str(role.get("status") or "").strip()
        if status not in {"waiting_approval", "pending", "blocked", "in_progress", "awaiting_resources"}:
            continue
        role_id = str(role.get("role_id") or "").strip()
        resume_action = str(role.get("resume_action") or "").strip()
        stage = str(role.get("stage") or "").strip()
        if role_id and resume_action:
            actions.append(f"{role_id}: {resume_action}")
        elif role_id and stage:
            actions.append(f"{role_id}: {stage}")
    approval_ids = [
        str(item).strip()
        for item in linked_approval_state.get("approval_ids", [])
        if str(item).strip()
    ]
    for approval_id in approval_ids:
        actions.append(f"Resolve approval {approval_id}")
    return _merge_unique_text_items([], actions)[:6]


def _goal_audit_finding_response(finding: dict[str, Any]) -> dict[str, Any]:
    return {
        "finding_id": finding["id"],
        "goal_id": finding["goal_id"],
        "finding_code": finding["finding_code"],
        "status": finding.get("status", "open"),
        "severity": finding.get("severity", "warning"),
        "summary": finding.get("summary"),
        "details": finding.get("details") or {},
        "resolved_at": finding.get("resolved_at"),
        "closed_at": finding.get("closed_at"),
        "created_at": finding["created_at"],
        "updated_at": finding["updated_at"],
    }


def _goal_operator_controls_stop_message(
    controls: Mapping[str, Any] | None,
) -> str:
    normalized = _normalize_goal_operator_controls(controls)
    metadata = normalized.get("metadata") if isinstance(normalized.get("metadata"), dict) else {}
    reason = _normalized_nonempty_text(metadata.get("reason"))
    if normalized.get("stop_all_goals"):
        message = "Goal execution is blocked by operator emergency stop."
    elif normalized.get("block_network_usage"):
        message = "Goal execution is paused because operator controls block network usage."
    elif normalized.get("blocked_domains"):
        message = "Goal execution is paused for operator domain restrictions."
    elif normalized.get("blocked_tools"):
        message = "Goal execution is paused for operator tool restrictions."
    else:
        message = "Goal execution is paused for operator safety controls."
    if reason is not None:
        return f"{message} Reason: {reason}"
    return message


def _goal_operator_controls_permission_policy(
    controls: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    normalized = _normalize_goal_operator_controls(controls)
    blocked_domains = normalized.get("blocked_domains")
    if not isinstance(blocked_domains, list) or not blocked_domains:
        return None
    return {"blocked_web_domains": list(blocked_domains)}


def _goal_operator_audit_log_response(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "audit_id": item["id"],
        "event_type": item.get("event_type"),
        "subject_type": item.get("subject_type"),
        "subject_id": item.get("subject_id"),
        "action": item.get("action"),
        "summary": item.get("summary"),
        "details": item.get("details") or {},
        "created_at": item.get("created_at"),
    }


def _goal_operator_audit_log_matches_goal(
    item: dict[str, Any],
    goal_id: str,
) -> bool:
    if not goal_id:
        return False
    subject_type = str(item.get("subject_type") or "").strip()
    subject_id = str(item.get("subject_id") or "").strip()
    if subject_type == "goal" and subject_id == goal_id:
        return True
    details = item.get("details") if isinstance(item.get("details"), dict) else {}
    if str(details.get("goal_id") or "").strip() == goal_id:
        return True
    if subject_type == "goal_operator_controls" and subject_id == "global":
        return True
    return False


def _goal_lease_health_payload(
    lease: dict[str, Any] | None,
    *,
    runtime_owner_id: str,
) -> dict[str, Any] | None:
    if lease is None:
        return None
    return {
        "goal_id": lease["goal_id"],
        "owner_id": lease.get("owner_id"),
        "acquired_at": lease.get("acquired_at"),
        "heartbeat_at": lease.get("heartbeat_at"),
        "expires_at": lease.get("expires_at"),
        "takeover_count": int(lease.get("takeover_count") or 0),
        "metadata": lease.get("metadata") or {},
        "owned_by_runtime": str(lease.get("owner_id") or "") == runtime_owner_id,
        "stale": _goal_lease_is_stale(lease),
        "updated_at": lease.get("updated_at"),
    }


def _linked_goal_context_from_agent_run(run: Mapping[str, Any]) -> tuple[str | None, str | None]:
    summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
    goal_id = (
        str(summary.get("goal_id") or "").strip()
        if isinstance(summary.get("goal_id"), str)
        else ""
    )
    attempt_id = (
        str(summary.get("goal_attempt_id") or "").strip()
        if isinstance(summary.get("goal_attempt_id"), str)
        else ""
    )
    return (goal_id or None, attempt_id or None)


def _merge_unique_text_items(base: list[str], extra: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for item in [*base, *extra]:
        text = str(item).strip()
        if not text or text in seen:
            continue
        merged.append(text)
        seen.add(text)
    return merged


def _current_goal_attempt(goal: dict[str, Any]) -> dict[str, Any] | None:
    attempts = goal.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        return None
    current_attempt_id = goal.get("current_attempt_id")
    if isinstance(current_attempt_id, str) and current_attempt_id.strip():
        for attempt in attempts:
            if isinstance(attempt, dict) and str(attempt.get("id") or "") == current_attempt_id:
                return attempt
    for attempt in reversed(attempts):
        if isinstance(attempt, dict):
            return attempt
    return None


def _next_goal_attempt_index(goal: dict[str, Any]) -> int:
    attempts = goal.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        return 1
    highest = 0
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        try:
            highest = max(highest, int(attempt.get("attempt_index") or 0))
        except (TypeError, ValueError):
            continue
    return highest + 1


def _goal_lease_is_stale(
    lease: dict[str, Any],
    *,
    now: datetime | None = None,
) -> bool:
    expires_at = _parse_iso_datetime(lease.get("expires_at"))
    if expires_at is None:
        return True
    effective_now = now or _utcnow()
    return expires_at <= effective_now


def _goal_status_from_agent_run_status(status: str) -> str:
    normalized = str(status or "created")
    if normalized == "created":
        return "queued"
    if normalized == "running":
        return "running"
    if normalized == "paused":
        return "paused"
    if normalized == "awaiting_approval":
        return "waiting_approval"
    if normalized == "awaiting_resources":
        return "awaiting_resources"
    if normalized == "partial":
        return "awaiting_operator"
    if normalized == "stalled":
        return "stalled"
    if normalized == "succeeded":
        return "completed"
    if normalized == "failed":
        return "failed"
    if normalized == "cancelled":
        return "cancelled"
    return "stalled"


def _goal_status_from_agent_run(
    run: dict[str, Any],
    *,
    linked_approval_state: dict[str, Any] | None = None,
) -> str:
    base_status = _goal_status_from_agent_run_status(
        _effective_agent_run_status_for_goal_projection(run)
    )
    approval_state = linked_approval_state or _goal_approval_state_from_agent_run(run)
    if (
        isinstance(approval_state, dict)
        and str(approval_state.get("status") or "") in {"waiting_approval", "awaiting_approval"}
        and base_status in {"awaiting_operator", "stalled"}
    ):
        return "waiting_approval"
    return base_status


def _effective_agent_run_status_for_goal_projection(
    run: Mapping[str, Any] | None,
) -> str:
    if not isinstance(run, Mapping):
        return "created"
    raw_status = str(run.get("status") or "created").strip() or "created"
    if raw_status != "running":
        return raw_status
    recovery_state = _goal_recovery_state_from_agent_run(dict(run))
    projected_status = str(recovery_state.get("status") or "").strip()
    resume_runtime = (
        recovery_state.get("resume_runtime")
        if isinstance(recovery_state.get("resume_runtime"), Mapping)
        else {}
    )
    if str(resume_runtime.get("status") or "").strip() == "active":
        return raw_status
    if projected_status in {"stalled", "awaiting_approval", "awaiting_resources"}:
        return projected_status
    return raw_status


def _approval_override(
    approval: dict[str, Any],
    replay_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    approved_call = _approved_tool_call(approval, replay_override=replay_override)
    return {"approved_tool_calls": [approved_call]}


def _approval_summary(approval: dict[str, Any]) -> dict[str, Any]:
    status = str(approval.get("status", "pending"))
    decision: str | None = None
    if status == "approved_once":
        decision = "approve_once"
    elif status == "approved_and_saved_rule":
        decision = "approve_and_save_rule"
    elif status == "rejected":
        decision = "reject"
    metadata = approval.get("metadata") if isinstance(approval.get("metadata"), dict) else {}
    security_decision = SecurityDecision.from_metadata(metadata)
    file_change_summary = summarize_file_change_payload(metadata)
    linked_exec_approval_id = _linked_exec_approval_id(metadata)
    exec_session_id = metadata.get("session_id") if isinstance(metadata.get("session_id"), str) else None
    exec_status = (
        metadata.get("exec_status")
        if isinstance(metadata.get("exec_status"), str)
        else metadata.get("status")
        if isinstance(metadata.get("status"), str)
        else None
    )
    execution_result = (
        metadata.get("execution_result")
        if isinstance(metadata.get("execution_result"), dict)
        else None
    )
    tool_name = approval["tool_name"]
    approval_kind = (
        security_decision.approval_kind
        if security_decision is not None
        else metadata.get("approval_kind", "other")
    )
    return {
        "approval_id": approval["id"],
        "task_id": approval["task_id"],
        "status": status,
        "tool_name": tool_name,
        "arguments": approval.get("arguments") or {},
        "command": approval.get("arguments", {}).get("command") if isinstance(approval.get("arguments"), dict) else None,
        "shell": approval.get("arguments", {}).get("shell") if isinstance(approval.get("arguments"), dict) else None,
        "workdir": metadata.get("workdir") if isinstance(metadata.get("workdir"), str) else None,
        "created_at": approval["created_at"],
        "resolved_at": approval.get("resolved_at"),
        "decision": decision,
        "reason": approval.get("reason"),
        "requires_approval": security_decision.requires_approval if security_decision else bool(metadata.get("requires_approval", False)),
        "policy_reason": security_decision.reason if security_decision else metadata.get("reason"),
        "approval_kind": approval_kind,
        "approval_scope": security_decision.approval_scope if security_decision else metadata.get("approval_scope", "workspace"),
        "replay_safe": security_decision.replay_safe if security_decision else bool(metadata.get("replay_safe", False)),
        "security_decision": security_decision.action if security_decision else None,
        "policy_source": security_decision.policy_source if security_decision else metadata.get("policy_source"),
        "suggested_rule": metadata.get("suggested_rule") if isinstance(metadata.get("suggested_rule"), dict) else None,
        "allowed_decisions": _allowed_approval_decisions(tool_name=tool_name, approval_kind=approval_kind),
        "source": "task_runtime",
        "exec_approval_id": linked_exec_approval_id,
        "exec_session_id": exec_session_id,
        "exec_status": exec_status,
        "execution_result": execution_result,
        **file_change_summary,
    }


def _exec_approval_summary(approval: Any) -> dict[str, Any]:
    status = str(getattr(approval, "status", "pending"))
    decision: str | None = None
    if status == "approved_once":
        decision = "approve_once"
    elif status == "approved_and_saved_rule":
        decision = "approve_and_save_rule"
    elif status == "rejected":
        decision = "reject"

    command = str(getattr(approval, "command", ""))
    shell = str(getattr(approval, "shell", "auto"))
    scope = str(getattr(approval, "scope", "dangerous_command"))
    reason = getattr(approval, "reason", None)
    reason_text = reason if isinstance(reason, str) else None
    metadata = getattr(approval, "metadata", None)
    metadata_dict = metadata if isinstance(metadata, dict) else {}

    return {
        "approval_id": str(getattr(approval, "approval_id", "")),
        "task_id": None,
        "status": status,
        "tool_name": "exec_command",
        "arguments": {"command": command, "shell": shell},
        "command": command,
        "shell": shell,
        "workdir": (
            str(getattr(approval, "command_payload", {}).get("workdir"))
            if isinstance(getattr(approval, "command_payload", None), dict)
            and isinstance(getattr(approval, "command_payload", {}).get("workdir"), str)
            else None
        ),
        "created_at": str(getattr(approval, "created_at", "")),
        "resolved_at": getattr(approval, "resolved_at", None),
        "decision": decision,
        "reason": reason_text,
        "requires_approval": True,
        "policy_reason": reason_text,
        "approval_kind": "exec",
        "approval_scope": scope,
        "replay_safe": False,
        "security_decision": "require_approval",
        "policy_source": "exec_runtime",
        "suggested_rule": metadata_dict.get("suggested_rule"),
        "allowed_decisions": ["approve_once", "approve_and_save_rule", "reject"],
        "source": "exec_runtime",
        "exec_approval_id": str(getattr(approval, "approval_id", "")),
        "exec_session_id": _exec_result_field(getattr(approval, "execution_result", None), "session_id"),
        "exec_status": status,
        "execution_result": getattr(approval, "execution_result", None),
    }


def _task_exec_command_payload_from_approval_request(
    approval: Mapping[str, Any],
    *,
    task: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    metadata = approval.get("metadata") if isinstance(approval.get("metadata"), dict) else {}
    persisted_payload = metadata.get("exec_command_payload")
    if isinstance(persisted_payload, dict) and persisted_payload:
        return dict(persisted_payload)
    arguments = approval.get("arguments") if isinstance(approval.get("arguments"), dict) else {}
    command = _first_nonempty_text(
        [
            arguments.get("command"),
            metadata.get("command"),
        ]
    )
    if command is None:
        return None
    shell = _first_nonempty_text(
        [
            arguments.get("shell"),
            metadata.get("shell"),
        ]
    ) or "auto"
    workdir = _first_nonempty_text(
        [
            metadata.get("workdir"),
            task.get("task_workspace_dir") if isinstance(task, Mapping) else None,
            task.get("project_workspace_dir") if isinstance(task, Mapping) else None,
            task.get("workspace_dir") if isinstance(task, Mapping) else None,
        ]
    )
    timeout_sec = _first_positive_number(
        [
            arguments.get("timeout_sec"),
            arguments.get("timeout"),
            metadata.get("timeout_sec"),
            metadata.get("timeout"),
        ]
    )
    background = _first_optional_bool(
        [
            arguments.get("background"),
            metadata.get("background"),
        ],
        default=False,
    )
    tty = _first_optional_bool(
        [
            arguments.get("tty"),
            metadata.get("tty"),
        ],
        default=False,
    )
    env = (
        dict(arguments.get("env"))
        if isinstance(arguments.get("env"), dict)
        else dict(metadata.get("env"))
        if isinstance(metadata.get("env"), dict)
        else None
    )
    payload = {
        "command": command,
        "shell": shell,
        "workdir": workdir,
        "env": env,
        "timeout_sec": timeout_sec,
        "background": background,
        "tty": tty,
        "log_path": _first_nonempty_text([metadata.get("log_path")]),
        "checkpoint_dir": _first_nonempty_text([metadata.get("checkpoint_dir")]),
        "approval_state": "approved",
    }
    return {key: value for key, value in payload.items() if value is not None}


def _task_exec_approval_metadata_from_request(
    approval: Mapping[str, Any],
    *,
    task: Mapping[str, Any] | None,
    exec_approval: Any | None = None,
) -> dict[str, Any]:
    metadata = dict(approval.get("metadata") or {}) if isinstance(approval.get("metadata"), dict) else {}
    command_payload = _task_exec_command_payload_from_approval_request(
        approval,
        task=task,
    )
    if command_payload is not None:
        metadata["exec_command_payload"] = command_payload
    if exec_approval is not None:
        metadata["exec_status"] = str(getattr(exec_approval, "status", "") or "")
        scope = str(getattr(exec_approval, "scope", "") or "").strip()
        if scope:
            metadata["exec_scope"] = scope
        reason = getattr(exec_approval, "reason", None)
        if isinstance(reason, str) and reason.strip():
            metadata["exec_reason"] = reason.strip()
        exec_created_at = getattr(exec_approval, "created_at", None)
        if isinstance(exec_created_at, str) and exec_created_at.strip():
            metadata["exec_created_at"] = exec_created_at.strip()
        exec_resolved_at = getattr(exec_approval, "resolved_at", None)
        if isinstance(exec_resolved_at, str) and exec_resolved_at.strip():
            metadata["exec_resolved_at"] = exec_resolved_at.strip()
        if isinstance(getattr(exec_approval, "command_payload", None), dict):
            metadata["exec_command_payload"] = dict(exec_approval.command_payload)
        execution_result = getattr(exec_approval, "execution_result", None)
        if isinstance(execution_result, dict):
            metadata["execution_result"] = dict(execution_result)
            session_id = _exec_result_field(execution_result, "session_id")
            if isinstance(session_id, str) and session_id.strip():
                metadata["session_id"] = session_id.strip()
        exec_metadata = getattr(exec_approval, "metadata", None)
        if isinstance(exec_metadata, dict):
            suggested_rule = exec_metadata.get("suggested_rule")
            if isinstance(suggested_rule, dict):
                metadata["suggested_rule"] = dict(suggested_rule)
    return metadata


def _rehydrate_exec_approval_metadata_from_task_approval(
    approval: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = approval.get("metadata") if isinstance(approval.get("metadata"), dict) else {}
    output: dict[str, Any] = {}
    for key in ("policy_state", "policy_reason", "rule_id", "exec_created_at"):
        value = metadata.get(key)
        if value is not None:
            output[key] = value
    suggested_rule = metadata.get("suggested_rule")
    if isinstance(suggested_rule, dict):
        output["suggested_rule"] = dict(suggested_rule)
    return output


def _validated_command_rule(rule: Any) -> CommandRuleConfig | None:
    if not isinstance(rule, dict):
        return None
    try:
        return CommandRuleConfig.model_validate(rule)
    except Exception:
        return None


def _approval_decision_from_status(status: str | None) -> ApprovalDecision | None:
    if status == "approved_once":
        return "approve_once"
    if status == "approved_and_saved_rule":
        return "approve_and_save_rule"
    if status == "rejected":
        return "reject"
    return None


def _allowed_approval_decisions(*, tool_name: str, approval_kind: Any) -> list[str]:
    if tool_name in _FILE_MUTATION_TOOLS or approval_kind in _FILE_MUTATION_APPROVAL_KINDS:
        return ["approve_once", "reject"]
    return ["approve_once", "approve_and_save_rule", "reject"]


def _is_file_mutation_approval(approval: dict[str, Any]) -> bool:
    metadata = approval.get("metadata") if isinstance(approval.get("metadata"), dict) else {}
    return (
        str(approval.get("tool_name") or "") in _FILE_MUTATION_TOOLS
        or metadata.get("approval_kind") in _FILE_MUTATION_APPROVAL_KINDS
    )


def _approved_tool_call(
    approval: dict[str, Any],
    *,
    replay_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if replay_override is None:
        return {
            "tool_name": str(approval.get("tool_name", "")),
            "arguments": dict(approval.get("arguments") or {}),
        }

    if not _is_file_mutation_approval(approval):
        raise ValueError("replay_override is only supported for file-mutation approvals.")
    tool_name = replay_override.get("tool_name")
    arguments = replay_override.get("arguments")
    if tool_name not in _FILE_MUTATION_TOOLS or not isinstance(arguments, dict):
        raise ValueError(
            "replay_override must target a file-mutation tool with object arguments."
        )
    return {"tool_name": str(tool_name), "arguments": dict(arguments)}


def _poll_to_execution_result(
    poll: Any,
    *,
    previous_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
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
    if isinstance(previous_result, Mapping):
        for key in ("log_path", "checkpoint_dir"):
            value = previous_result.get(key)
            if isinstance(value, str):
                result[key] = value
    return result


def _build_approval_exec_session_payload(
    *,
    context: Mapping[str, Any],
    poll: Any,
) -> dict[str, Any]:
    return {
        "approval_id": context.get("approval_id"),
        "source": context.get("source"),
        "exec_approval_id": context.get("exec_approval_id"),
        "task_id": context.get("task_id"),
        "tool_name": context.get("tool_name"),
        "session_id": poll.session_id,
        "status": context.get("status"),
        "live_status": "available",
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
    if status in {"pending", "approved_once", "approved_and_saved_rule", "rejected"}:
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


def _final_answer_from_file_mutation_result(result: ToolResult) -> str | None:
    if result.error is not None:
        return result.error
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    file_changes = metadata.get("file_changes")
    if not isinstance(file_changes, list) or not file_changes:
        return "Approved file change applied."

    labels: list[str] = []
    for item in file_changes:
        if not isinstance(item, dict):
            continue
        relative_path = item.get("relative_path")
        path = item.get("path")
        if isinstance(relative_path, str) and relative_path.strip():
            labels.append(relative_path.strip())
        elif isinstance(path, str) and path.strip():
            labels.append(Path(path).name)

    if not labels:
        return "Approved file change applied."
    if len(labels) == 1:
        return f"Applied approved file change to {labels[0]}."
    return f"Applied approved file changes to {len(labels)} files."


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
        "project_id": _agent_run_project_id(run),
        "workspace_dir": _agent_run_workspace_dir(run),
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


def _agent_run_project_id(run: dict[str, Any]) -> str | None:
    project_id = run.get("project_id")
    if isinstance(project_id, str) and project_id.strip():
        return project_id
    summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
    summary_project_id = summary.get("project_id")
    if isinstance(summary_project_id, str) and summary_project_id.strip():
        return summary_project_id
    return None


def _agent_run_workspace_dir(run: dict[str, Any]) -> str | None:
    workspace_dir = run.get("workspace_dir")
    if isinstance(workspace_dir, str) and workspace_dir.strip():
        return workspace_dir
    summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
    summary_workspace_dir = summary.get("workspace_dir")
    if isinstance(summary_workspace_dir, str) and summary_workspace_dir.strip():
        return summary_workspace_dir
    return None


def _normalize_agent_run_message_payload(
    payload: AgentRunMessageRequest | AgentRunSubagentMessageRequest,
) -> dict[str, Any]:
    content = payload.content.strip()
    metadata = dict(payload.metadata or {})
    attachments = [attachment.to_attachment_dict() for attachment in payload.attachments]
    if payload.project_id is not None:
        metadata["project_id"] = payload.project_id
    if payload.workspace_dir is not None:
        metadata["workspace_dir"] = payload.workspace_dir
    if attachments:
        metadata["attachments"] = attachments
    return {
        "role": "operator",
        "source_role": payload.role,
        "content": content,
        "project_id": payload.project_id,
        "workspace_dir": payload.workspace_dir,
        "attachments": attachments,
        "metadata": metadata,
    }


def _build_workflow_assistant_acknowledgement(run: dict[str, Any], message: str) -> str:
    preview = _truncate_preview(message, max_chars=160) or "message received"
    status = str(run.get("status") or "created")
    project_id = _agent_run_project_id(run)
    workspace_dir = _agent_run_workspace_dir(run)
    parts = [f"Recorded your workflow message: {preview}."]
    if project_id:
        parts.append(f"Project: {project_id}.")
    if workspace_dir:
        parts.append(f"Workspace: {workspace_dir}.")
    parts.append(f"Current run status: {status}.")
    parts.append("The workflow run will continue within the existing orchestrator boundaries.")
    return " ".join(parts)


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
    apply_default_checkpoint_interval_steps: bool = True,
) -> dict[str, Any]:
    payload = {**dict(defaults or {}), **dict(run_policy or {})}
    normalized = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "max_wall_clock_sec",
            "heartbeat_timeout_sec",
            "checkpoint_interval_steps",
            "max_subagent_failures_per_role",
            "on_budget_exhausted",
            "on_subagent_disconnect",
        }
    }

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

    checkpoint_interval_steps = _optional_positive_int("checkpoint_interval_steps", maximum=10_000)

    normalized.update(
        {
            "max_wall_clock_sec": _optional_positive_int("max_wall_clock_sec"),
            "heartbeat_timeout_sec": _optional_positive_int("heartbeat_timeout_sec"),
            "max_subagent_failures_per_role": (
                2 if max_subagent_failures_per_role is None else max_subagent_failures_per_role
            ),
            "on_budget_exhausted": on_budget_exhausted,
            "on_subagent_disconnect": on_subagent_disconnect,
        }
    )
    if checkpoint_interval_steps is not None:
        normalized["checkpoint_interval_steps"] = checkpoint_interval_steps
    elif apply_default_checkpoint_interval_steps:
        normalized["checkpoint_interval_steps"] = 1
    return {key: value for key, value in normalized.items() if value is not None}


_GOAL_RUNTIME_POLICY_DURATION_MAX_SECONDS = 31_536_000
_GOAL_RUNTIME_POLICY_RETRY_MAX = 1_000
_GOAL_RUNTIME_POLICY_TOKEN_REFRESH_MAX = 100_000_000
_GOAL_DURATION_CONTEXT_PATTERNS = (
    re.compile(
        r"(?ix)"
        r"\b(?:run|running|continue|keep\s+running|execute|work|operate)\b"
        r"(?:[\s,:-]+(?:for|within|about|around|at\s+least|roughly|approximately))?"
        r"[\s,:-]*"
        r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>seconds?|secs?|sec|s|minutes?|mins?|min|m|hours?|hrs?|hr|h|days?|day|d)\b"
    ),
    re.compile(
        r"(?x)"
        r"(?:執行|运行|運行|跑|持續|持续|連跑|连跑|不間斷|不间断|繼續|继续)"
        r"(?:至少|最少|約|大約|大概|约|大约|大概)?\s*"
        r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>秒|分鐘|分钟|分|小時|小时|天)"
    ),
)
_GOAL_DURATION_BARE_PATTERN = re.compile(
    r"(?ix)"
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>seconds?|secs?|sec|s|minutes?|mins?|min|m|hours?|hrs?|hr|h|days?|day|d|秒|分鐘|分钟|分|小時|小时|天)\b"
)


def _normalize_goal_run_policy(
    run_policy: dict[str, Any] | None,
    *,
    objective: Any = None,
) -> dict[str, Any]:
    payload = dict(run_policy) if isinstance(run_policy, dict) else {}
    normalized = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "requested_duration",
            "requested_duration_text",
            "requested_duration_sec",
            "requested_duration_source",
            "runtime_mode",
            "soft_deadline_at",
            "hard_stop_at",
            "checkpoint_interval_sec",
            "generation_refresh_interval_sec",
            "generation_token_refresh_threshold",
            "context_handoff_threshold",
            "approval_mode",
            "max_attempt_retries",
        }
    }

    normalized.update(_normalize_agent_run_policy(payload, defaults=None))
    checkpoint_interval_steps = _goal_policy_positive_int(
        payload.get("checkpoint_interval_steps"),
        maximum=10_000,
    )
    if checkpoint_interval_steps is not None:
        normalized["checkpoint_interval_steps"] = checkpoint_interval_steps
    else:
        normalized.pop("checkpoint_interval_steps", None)

    requested_duration_text: str | None = None
    requested_duration_source: str | None = None
    existing_duration_source = _normalized_nonempty_text(payload.get("requested_duration_source"))
    requested_duration_sec = _goal_policy_positive_int(
        payload.get("requested_duration_sec"),
        maximum=_GOAL_RUNTIME_POLICY_DURATION_MAX_SECONDS,
    )
    explicit_duration_text = _normalized_nonempty_text(
        payload.get("requested_duration_text") or payload.get("requested_duration")
    )
    if requested_duration_sec is not None:
        requested_duration_source = existing_duration_source or "run_policy.requested_duration_sec"
        if explicit_duration_text is not None:
            requested_duration_text = explicit_duration_text
    elif explicit_duration_text is not None:
        parsed_duration = _parse_goal_duration_expression(
            explicit_duration_text,
            require_runtime_context=False,
        )
        if parsed_duration is not None:
            requested_duration_text = explicit_duration_text
            requested_duration_sec = parsed_duration[1]
            requested_duration_source = existing_duration_source or "run_policy.requested_duration"
    else:
        parsed_from_objective = _parse_goal_duration_expression(
            objective,
            require_runtime_context=True,
        )
        if parsed_from_objective is not None:
            requested_duration_text, requested_duration_sec = parsed_from_objective
            requested_duration_source = "objective"

    soft_deadline_at = _normalized_goal_policy_datetime(payload.get("soft_deadline_at"))
    hard_stop_at = _normalized_goal_policy_datetime(payload.get("hard_stop_at"))
    if soft_deadline_at is not None and hard_stop_at is not None:
        soft_deadline_dt = _parse_iso_datetime(soft_deadline_at)
        hard_stop_dt = _parse_iso_datetime(hard_stop_at)
        if soft_deadline_dt is not None and hard_stop_dt is not None and soft_deadline_dt > hard_stop_dt:
            soft_deadline_at = hard_stop_at

    runtime_mode = _normalized_goal_runtime_mode(
        payload.get("runtime_mode"),
        requested_duration_sec=requested_duration_sec,
        hard_stop_at=hard_stop_at,
    )
    checkpoint_interval_sec = _goal_policy_positive_int(
        payload.get("checkpoint_interval_sec"),
        maximum=_GOAL_RUNTIME_POLICY_DURATION_MAX_SECONDS,
    )
    generation_refresh_interval_sec = _goal_policy_positive_int(
        payload.get("generation_refresh_interval_sec"),
        maximum=_GOAL_RUNTIME_POLICY_DURATION_MAX_SECONDS,
    )
    generation_token_refresh_threshold = _goal_policy_positive_int(
        payload.get("generation_token_refresh_threshold"),
        maximum=_GOAL_RUNTIME_POLICY_TOKEN_REFRESH_MAX,
    )
    context_handoff_threshold = _goal_policy_fraction(payload.get("context_handoff_threshold"))
    approval_mode = _normalized_goal_approval_mode(payload.get("approval_mode"))
    max_attempt_retries = _goal_policy_nonnegative_int(
        payload.get("max_attempt_retries"),
        maximum=_GOAL_RUNTIME_POLICY_RETRY_MAX,
    )

    if requested_duration_text is not None:
        normalized["requested_duration_text"] = requested_duration_text
    if requested_duration_sec is not None:
        normalized["requested_duration_sec"] = requested_duration_sec
    if requested_duration_source is not None:
        normalized["requested_duration_source"] = requested_duration_source
    normalized["runtime_mode"] = runtime_mode
    if soft_deadline_at is not None:
        normalized["soft_deadline_at"] = soft_deadline_at
    if hard_stop_at is not None:
        normalized["hard_stop_at"] = hard_stop_at
    if checkpoint_interval_sec is not None:
        normalized["checkpoint_interval_sec"] = checkpoint_interval_sec
    if generation_refresh_interval_sec is not None:
        normalized["generation_refresh_interval_sec"] = generation_refresh_interval_sec
    if generation_token_refresh_threshold is not None:
        normalized["generation_token_refresh_threshold"] = generation_token_refresh_threshold
    if context_handoff_threshold is not None:
        normalized["context_handoff_threshold"] = context_handoff_threshold
    if approval_mode is not None:
        normalized["approval_mode"] = approval_mode
    if max_attempt_retries is not None:
        normalized["max_attempt_retries"] = max_attempt_retries
    return normalized


def _normalize_goal_capability_policy(
    capability_policy: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = dict(capability_policy) if isinstance(capability_policy, dict) else {}
    normalized = {
        key: value
        for key, value in payload.items()
        if key != "allowed_tools"
    }
    if "allowed_tools" in payload:
        normalized["allowed_tools"] = _normalized_string_list(payload.get("allowed_tools"))
    return normalized


def _normalize_goal_operator_controls(
    controls: Mapping[str, Any] | None,
) -> dict[str, Any]:
    payload = dict(controls) if isinstance(controls, Mapping) else {}
    blocked_tools = _normalized_string_list(payload.get("blocked_tools"))
    blocked_domains = _normalized_domain_list(payload.get("blocked_domains"))
    block_network_usage = bool(payload.get("block_network_usage"))
    metadata = dict(payload.get("metadata") or {}) if isinstance(payload.get("metadata"), dict) else {}
    tool_denylist = _normalized_string_list(
        [
            *blocked_tools,
            *(_GOAL_ESTOP_NETWORK_TOOL_NAMES if block_network_usage else ()),
        ],
    )
    return {
        "scope": _normalized_nonempty_text(payload.get("scope")) or "global",
        "stop_all_goals": bool(payload.get("stop_all_goals")),
        "blocked_tools": blocked_tools,
        "blocked_domains": blocked_domains,
        "block_network_usage": block_network_usage,
        "tool_denylist": tool_denylist,
        "metadata": metadata,
        "created_at": payload.get("created_at"),
        "updated_at": payload.get("updated_at"),
    }


def _normalized_domain_list(value: Any) -> list[str]:
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        return []
    normalized: list[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        candidate = item.strip().lower()
        if not candidate:
            continue
        if "://" in candidate:
            parsed = urlparse(candidate)
            candidate = (parsed.hostname or "").strip().lower()
        else:
            candidate = candidate.split("/", 1)[0].strip().lower()
            if ":" in candidate:
                candidate = candidate.split(":", 1)[0].strip().lower()
        candidate = candidate.lstrip(".")
        if candidate and candidate not in normalized:
            normalized.append(candidate)
    return normalized


def _parse_goal_duration_expression(
    value: Any,
    *,
    require_runtime_context: bool,
) -> tuple[str, int] | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    patterns = _GOAL_DURATION_CONTEXT_PATTERNS if require_runtime_context else (_GOAL_DURATION_BARE_PATTERN,)
    for pattern in patterns:
        match = pattern.search(text)
        if match is None:
            continue
        seconds = _goal_duration_match_to_seconds(match)
        if seconds is None:
            continue
        raw_value = str(match.group("value") or "").strip()
        raw_unit = str(match.group("unit") or "").strip()
        if not raw_value or not raw_unit:
            continue
        return (f"{raw_value} {raw_unit}".strip(), seconds)
    return None


def _goal_duration_match_to_seconds(match: re.Match[str]) -> int | None:
    raw_value = str(match.group("value") or "").strip()
    raw_unit = str(match.group("unit") or "").strip()
    if not raw_value or not raw_unit:
        return None
    try:
        numeric_value = float(raw_value)
    except ValueError:
        return None
    if numeric_value <= 0:
        return None
    unit_seconds = _goal_duration_unit_seconds(raw_unit)
    if unit_seconds is None:
        return None
    total_seconds = int(numeric_value * unit_seconds)
    if total_seconds <= 0:
        return None
    return min(total_seconds, _GOAL_RUNTIME_POLICY_DURATION_MAX_SECONDS)


def _goal_duration_unit_seconds(unit: str) -> int | None:
    normalized_unit = unit.strip().lower()
    if normalized_unit in {"s", "sec", "secs", "second", "seconds", "秒"}:
        return 1
    if normalized_unit in {"m", "min", "mins", "minute", "minutes", "分", "分钟", "分鐘"}:
        return 60
    if normalized_unit in {"h", "hr", "hrs", "hour", "hours", "小时", "小時"}:
        return 3_600
    if normalized_unit in {"d", "day", "days", "天"}:
        return 86_400
    return None


def _goal_policy_positive_int(value: Any, *, maximum: int) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return max(1, min(parsed, maximum))


def _goal_policy_nonnegative_int(value: Any, *, maximum: int) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return max(0, min(parsed, maximum))


def _goal_policy_fraction(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed > 1.0 and parsed <= 100.0:
        parsed = parsed / 100.0
    if parsed <= 0.0 or parsed > 1.0:
        return None
    return round(parsed, 4)


def _normalized_goal_policy_datetime(value: Any) -> str | None:
    parsed = _parse_iso_datetime(value)
    if parsed is None:
        return None
    return _to_iso_utc(parsed)


def _normalized_goal_runtime_mode(
    value: Any,
    *,
    requested_duration_sec: int | None,
    hard_stop_at: str | None,
) -> str:
    normalized = _normalized_nonempty_text(value)
    if normalized is not None:
        candidate = normalized.lower().replace("-", "_").replace(" ", "_")
        if candidate in {"until_complete", "fixed_duration", "until_deadline"}:
            return candidate
    if hard_stop_at is not None:
        return "until_deadline"
    if requested_duration_sec is not None:
        return "fixed_duration"
    return "until_complete"


def _normalized_goal_approval_mode(value: Any) -> str | None:
    normalized = _normalized_nonempty_text(value)
    if normalized is None:
        return None
    return normalized.lower().replace("-", "_").replace(" ", "_")


def _goal_approval_mode_auto_rejects(value: Any) -> bool:
    normalized = _normalized_goal_approval_mode(value)
    return normalized in {"deny", "reject", "auto_deny", "block"}


def _normalized_nonempty_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _normalized_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        return []
    normalized: list[str] = []
    for item in items:
        text = _normalized_nonempty_text(item)
        if text is None or text in normalized:
            continue
        normalized.append(text)
    return normalized


def _default_agent_run_policy_from_security(security: SecurityConfig) -> dict[str, Any]:
    return {
        "max_wall_clock_sec": security.agent_run_default_max_wall_clock_sec,
        "heartbeat_timeout_sec": security.agent_run_default_heartbeat_timeout_sec,
        "checkpoint_interval_steps": security.agent_run_default_checkpoint_interval_steps,
        "max_subagent_failures_per_role": security.agent_run_default_max_subagent_failures_per_role,
        "on_budget_exhausted": security.agent_run_default_on_budget_exhausted,
        "on_subagent_disconnect": security.agent_run_default_on_subagent_disconnect,
    }


def _normalize_task_metadata(
    metadata: Any,
    *,
    task_type: str | None,
    input_message: str,
    session_id: Any,
    status: Any,
) -> dict[str, Any]:
    output = dict(metadata) if isinstance(metadata, dict) else {}
    delegated_subagent = _delegated_subagent_payload(
        output,
        task_type=task_type,
        input_message=input_message,
        session_id=session_id,
        status=status,
    )
    if delegated_subagent is not None:
        output["delegated_subagent"] = delegated_subagent
    return output


def _delegated_subagent_payload(
    metadata: Any,
    *,
    task_type: str | None,
    input_message: str,
    session_id: Any,
    status: Any,
) -> dict[str, Any] | None:
    if str(task_type or "").strip() != DELEGATED_MULTI_AGENT_TASK_TYPE:
        return None
    payload = (
        metadata.get("delegated_subagent")
        if isinstance(metadata, dict) and isinstance(metadata.get("delegated_subagent"), dict)
        else {}
    )
    objective = _clean_text(payload.get("objective")) or _clean_text(input_message) or ""
    instruction = _clean_text(payload.get("instruction")) or objective
    role = _clean_text(payload.get("role")) or _delegated_subagent_role(metadata) or "subagent"
    display_name = _clean_text(payload.get("display_name")) or _delegated_subagent_display_name(role)
    parent_session_id = _clean_text(payload.get("parent_session_id")) or _clean_text(session_id)
    return {
        **payload,
        "display_name": display_name,
        "role": role,
        "instruction": instruction,
        "objective": objective,
        "parent_session_id": parent_session_id,
        "status": _clean_text(status) or _clean_text(payload.get("status")) or "queued",
    }


def _build_delegated_subagent_metadata(
    *,
    objective: str,
    session_id: str | None,
    suggested_roles: list[str] | None,
) -> dict[str, Any]:
    role = _first_nonempty_text(suggested_roles) or "subagent"
    objective_text = objective.strip()
    return {
        "display_name": _new_delegated_subagent_display_name(),
        "role": role,
        "instruction": objective_text,
        "objective": objective_text,
        "parent_session_id": _clean_text(session_id),
        "status": "queued",
    }


def _delegated_subagent_created_event(
    delegated_subagent: dict[str, Any],
    *,
    task_id: str,
    task_type: str | None,
) -> dict[str, Any]:
    return {
        "type": "delegated_subagent_created",
        "task_id": task_id,
        "task_type": task_type,
        "display_name": delegated_subagent.get("display_name"),
        "role": delegated_subagent.get("role"),
        "instruction": delegated_subagent.get("instruction"),
        "objective": delegated_subagent.get("objective"),
        "parent_session_id": delegated_subagent.get("parent_session_id"),
        "status": delegated_subagent.get("status"),
        "created_context": {
            "source": "delegate_subagent_task",
            "delivery": "guidance_only",
        },
    }


def _delegated_subagent_message_event(
    delegated_subagent: dict[str, Any],
    *,
    task_id: str,
    content: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "type": "delegated_subagent_message",
        "task_id": task_id,
        "content": content,
        "display_name": delegated_subagent.get("display_name"),
        "role": delegated_subagent.get("role"),
        "parent_session_id": delegated_subagent.get("parent_session_id"),
        "metadata": dict(metadata or {}),
        "created_context": {
            "source": "task_message_api",
            "delivery": "guidance_only",
        },
    }


def _delegated_subagent_role(metadata: Any) -> str | None:
    if not isinstance(metadata, dict):
        return None
    selected_models_roles = metadata.get("selected_models_roles")
    if isinstance(selected_models_roles, dict):
        subagents = selected_models_roles.get("subagents")
        if isinstance(subagents, list):
            for item in subagents:
                if not isinstance(item, dict):
                    continue
                role = _clean_text(item.get("role"))
                if role:
                    return role
        by_role = selected_models_roles.get("by_role")
        if isinstance(by_role, dict):
            for role in by_role:
                cleaned = _clean_text(role)
                if cleaned:
                    return cleaned
    return None


def _new_delegated_subagent_display_name() -> str:
    index = int(uuid4().hex[:8], 16) % len(_DELEGATED_SUBAGENT_DISPLAY_NAMES)
    return _DELEGATED_SUBAGENT_DISPLAY_NAMES[index]


def _delegated_subagent_display_name(role: str) -> str:
    # Compatibility fallback for old delegated tasks that predate persisted nicknames.
    return "Delegated Subagent"


def _first_nonempty_text(values: list[str] | None) -> str | None:
    if not isinstance(values, list):
        return None
    for value in values:
        cleaned = _clean_text(value)
        if cleaned:
            return cleaned
    return None


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


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
    approval_state = (result.metadata or {}).get("approval_state")
    if isinstance(approval_state, dict):
        summary["approval_state"] = approval_state
    else:
        summary.pop("approval_state", None)
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
    protocol_artifacts = _protocol_artifacts_from_run_summary(run_summary)
    persisted_artifacts = [
        dict(item) for item in run.get("artifacts") or [] if isinstance(item, dict)
    ]
    schedule = run.get("schedule") if isinstance(run.get("schedule"), dict) else {}
    current_attempt_id = (
        str(schedule.get("current_attempt_id"))
        if isinstance(schedule.get("current_attempt_id"), str) and schedule.get("current_attempt_id")
        else None
    )
    collector_shard_manifests = dedupe_collector_shard_manifests(
        [
            *collector_shard_manifests_from_result(protocol_artifacts),
            *extract_collector_shard_manifests(
                persisted_artifacts
            ),
        ]
    )
    if collector_shard_manifests:
        protocol_artifacts["collector_shard_manifests"] = {
            "shards": [dict(item) for item in collector_shard_manifests]
        }
    collector_dataset_records = dedupe_collector_dataset_records(
        [
            *collector_dataset_records_from_result(protocol_artifacts),
            *extract_persisted_collector_dataset_records(
                persisted_artifacts,
                attempt_id=current_attempt_id,
            ),
        ]
    )
    if collector_dataset_records:
        protocol_artifacts["collector_dataset_records"] = {
            "records": [
                {
                    key: value
                    for key, value in record.items()
                    if key
                    not in {
                        "attempt_id",
                        "artifact_id",
                        "artifact_created_at",
                        "artifact_updated_at",
                    }
                }
                for record in collector_dataset_records
            ]
        }
    collector_record_provenance = collector_record_provenance_list_from_result(protocol_artifacts)
    if collector_dataset_records:
        collector_record_provenance = collector_record_provenance_list_from_dataset_records(
            collector_dataset_records
        )
    if collector_record_provenance:
        protocol_artifacts["collector_record_provenance"] = {
            "records": [dict(item) for item in collector_record_provenance]
        }
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
        "protocol_artifacts": protocol_artifacts,
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
    if state in {"succeeded", "partial", "awaiting_approval", "awaiting_resources", "stalled", "failed", "cancelled"}:
        return state
    if isinstance(recovery_state, dict):
        action = str(recovery_state.get("action") or "").strip().lower()
        if action == "await_approval":
            return "awaiting_approval"
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
    if str(recovery_state.get("status") or "") not in {"awaiting_approval", "awaiting_resources", "stalled"}:
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


def _latest_agent_run_artifact(run: dict[str, Any], artifact_type: str) -> dict[str, Any] | None:
    artifacts = run.get("artifacts")
    if not isinstance(artifacts, list):
        return None
    for artifact in reversed(artifacts):
        if not isinstance(artifact, dict):
            continue
        if str(artifact.get("artifact_type") or "") != artifact_type:
            continue
        return dict(artifact)
    return None


def _latest_agent_run_artifact_content(run: dict[str, Any], artifact_type: str) -> dict[str, Any] | None:
    artifact = _latest_agent_run_artifact(run, artifact_type)
    if not isinstance(artifact, dict):
        return None
    metadata = artifact.get("metadata") if isinstance(artifact.get("metadata"), dict) else {}
    content = metadata.get("content")
    if isinstance(content, dict):
        return dict(content)
    return None


def _latest_debate_context_snapshot_payload(
    run: dict[str, Any] | None,
    *,
    generation_started_at: str | None = None,
) -> dict[str, Any] | None:
    if not isinstance(run, dict):
        return None
    artifacts = run.get("artifacts")
    if not isinstance(artifacts, list):
        return None
    cutoff = _parse_iso_datetime(generation_started_at)
    snapshot: dict[str, Any] | None = None
    artifact_created_at: str | None = None
    for artifact in reversed(artifacts):
        if not isinstance(artifact, dict):
            continue
        if str(artifact.get("artifact_type") or "") != "debate_context_snapshot":
            continue
        created_at = (
            str(artifact.get("created_at") or "").strip()
            if isinstance(artifact.get("created_at"), str)
            else ""
        )
        created_at_dt = _parse_iso_datetime(created_at)
        if cutoff is not None and (created_at_dt is None or created_at_dt < cutoff):
            continue
        metadata = artifact.get("metadata") if isinstance(artifact.get("metadata"), dict) else {}
        content = metadata.get("content")
        if not isinstance(content, dict):
            continue
        snapshot = dict(content)
        artifact_created_at = created_at or None
        break
    if not isinstance(snapshot, dict):
        return None
    latest = snapshot.get("latest")
    if isinstance(latest, dict):
        payload = dict(latest)
        if artifact_created_at is not None:
            payload["artifact_created_at"] = artifact_created_at
        return payload
    snapshots = snapshot.get("snapshots")
    if isinstance(snapshots, list):
        for item in reversed(snapshots):
            if isinstance(item, dict):
                payload = dict(item)
                if artifact_created_at is not None:
                    payload["artifact_created_at"] = artifact_created_at
                return payload
    return None


def _goal_context_snapshot_fraction(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0.0:
        return None
    return round(min(parsed, 1.0), 4)


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
    snapshot = _latest_agent_run_artifact_content(run, "role_task_snapshot")
    if not isinstance(snapshot, dict) or not snapshot:
        snapshot = _agent_run_role_task_snapshot(run)
    return _summarize_role_task_snapshot(snapshot)


def _summarize_role_task_snapshot(snapshot: Mapping[str, Any] | None) -> dict[str, Any]:
    snapshot_payload = dict(snapshot) if isinstance(snapshot, Mapping) else {}
    roles = (
        snapshot_payload.get("roles")
        if isinstance(snapshot_payload.get("roles"), dict)
        else {}
    )
    assignments = (
        snapshot_payload.get("resume_plan", {}).get("assignments")
        if isinstance(snapshot_payload.get("resume_plan"), dict)
        and isinstance(snapshot_payload.get("resume_plan", {}).get("assignments"), dict)
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
        "available": bool(snapshot_payload),
        "tracked_role_count": len(summarized_roles),
        "status_counts": status_counts,
        "reassigned_role_count": len(
            list(snapshot_payload.get("resume_plan", {}).get("reassigned_roles") or [])
            if isinstance(snapshot_payload.get("resume_plan"), dict)
            else []
        ),
        "blocked_role_count": len(
            list(snapshot_payload.get("resume_plan", {}).get("blocked_roles") or [])
            if isinstance(snapshot_payload.get("resume_plan"), dict)
            else []
        ),
        "roles": summarized_roles,
    }


def _goal_progress_signature_fragment(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    except TypeError:
        return str(value)


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
