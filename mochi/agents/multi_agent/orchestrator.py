"""Multi-agent orchestration runtime."""

from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Mapping
from uuid import uuid4

from mochi.agents.multi_agent.context import (
    DebateContextManager,
    DebateContextOverflowError,
    DebateContextPolicy,
    summarize_candidate_outputs,
    verification_reduce,
)
from mochi.agents.multi_agent.execution_coordinator import (
    ControlledExecutionResumeHooks,
    SubagentExecutionCoordinator,
)
from mochi.agents.multi_agent.execution_policy import (
    LEGACY_CONTROLLED_SUBAGENT_PROTOCOL,
    SubagentExecutionPolicy,
    execution_policy_to_dict,
    parse_subagent_execution_policy,
)
from mochi.agents.multi_agent.evaluator import (
    CandidateVerification,
    CandidateScore,
    LLMFirstScoringPolicy,
    evidence_gate_rank,
    resolve_evidence_gate_status,
)
from mochi.agents.multi_agent.protocols import (
    ControlledSubagentExecutionProtocol,
    DrZeroSelfEvolveProtocol,
    MultiAgentDebateProtocol,
    ProtocolConfig,
    TeacherStudentDistillProtocol,
    parse_protocol_config,
)
from mochi.agents.multi_agent.research import (
    ResearchDebatePolicy,
    build_claim_evidence_map,
    build_research_plan,
    build_research_prompt_appendix,
    build_source_quality_table,
    merge_research_queries,
    run_research_workers,
    synthesize_research_brief,
)
from mochi.agents.multi_agent.roles import (
    build_dr_zero_roles,
    build_multi_agent_debate_roles,
    build_teacher_student_roles,
)
from mochi.agents.multi_agent.utils import parse_json_payload
from mochi.agents.context_snapshot import estimate_text_tokens
from mochi.agents.invocation import AgentInvocationRequest
from mochi.backends.types import GenerationResult, Message
from mochi.runtime.approvals import InMemoryApprovalStore
from mochi.runtime.exec_runtime import ExecRuntime
from mochi.runtime.recovery import (
    build_resource_exhaustion_report,
    classify_agent_run_recovery_issue,
)

RunState = Literal[
    "queued",
    "running",
    "awaiting_guidance",
    "awaiting_resources",
    "stalled",
    "partial",
    "succeeded",
    "failed",
    "cancelled",
]
RunEventType = Literal["state_changed", "guidance", "role_output", "verification", "evaluation", "artifact"]

RUN_STATE_TRANSITIONS: dict[RunState, tuple[RunState, ...]] = {
    "queued": ("running", "cancelled"),
    "running": ("awaiting_guidance", "awaiting_resources", "stalled", "partial", "succeeded", "failed", "cancelled"),
    "awaiting_guidance": ("running", "awaiting_resources", "stalled", "partial", "failed", "cancelled"),
    "awaiting_resources": (),
    "stalled": ("running", "failed", "cancelled"),
    "partial": (),
    "succeeded": (),
    "failed": (),
    "cancelled": (),
}


@dataclass(frozen=True)
class StateChangedPayload:
    """State transition payload."""

    previous_state: RunState | None
    current_state: RunState
    reason: str | None = None


@dataclass(frozen=True)
class GuidancePayload:
    """Guidance payload."""

    guidance_id: str
    content: str
    source: Literal["system", "user", "policy"] = "system"


@dataclass(frozen=True)
class RoleOutputPayload:
    """Role output payload."""

    role_id: str
    content: str
    round_index: int
    candidate_id: str | None = None
    model_id: str | None = None


@dataclass(frozen=True)
class EvaluationPayload:
    """Evaluation payload."""

    policy: dict[str, Any]
    selected_candidate_id: str | None
    scores: list[dict[str, Any]]


@dataclass(frozen=True)
class VerificationPayload:
    """Verification payload."""

    verifier_model_id: str | None
    evidence_packet_count: int
    verifications: list[dict[str, Any]]


@dataclass(frozen=True)
class ArtifactPayload:
    """Artifact payload."""

    name: str
    content: dict[str, Any]


@dataclass(frozen=True)
class ResumeExecutionPayload:
    """Structured resume payload persisted in recovery metadata."""

    executor: Literal["continue_from_checkpoint", "restart_attempt"]
    stage: str
    checkpoint: dict[str, Any]
    guidance_messages: list[str] = field(default_factory=list)
    metadata_state: dict[str, Any] = field(default_factory=dict)
    precomputed_artifacts: dict[str, Any] = field(default_factory=dict)
    protocol_artifacts: dict[str, Any] = field(default_factory=dict)
    candidates: list[CandidateOutput] = field(default_factory=list)
    evidence_packets: list[dict[str, Any]] = field(default_factory=list)
    evidence_collection_summary: dict[str, Any] | None = None
    verifications: list[CandidateVerification] = field(default_factory=list)
    verification_summary: dict[str, Any] | None = None
    evaluation: EvaluationPayload | None = None
    role_task_snapshot: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-safe dict."""
        continue_supported = self.executor == "continue_from_checkpoint"
        return {
            "version": 1,
            "executor": self.executor,
            "supported_actions": (
                ["restart_attempt", "continue_from_checkpoint"]
                if continue_supported
                else ["restart_attempt"]
            ),
            "stage": self.stage,
            "checkpoint": _json_safe_clone(self.checkpoint),
            "guidance_messages": [str(item) for item in self.guidance_messages if str(item).strip()],
            "metadata_state": _json_safe_clone(self.metadata_state),
            "precomputed_artifacts": _json_safe_clone(self.precomputed_artifacts),
            "protocol_artifacts": _json_safe_clone(self.protocol_artifacts),
            "candidates": [item.to_dict() for item in self.candidates],
            "evidence_packets": _json_safe_clone(self.evidence_packets),
            "evidence_collection_summary": _json_safe_clone(self.evidence_collection_summary),
            "verifications": [item.to_dict() for item in self.verifications],
            "verification_summary": _json_safe_clone(self.verification_summary),
            "evaluation": (
                _json_safe_clone(asdict(self.evaluation))
                if self.evaluation is not None
                else None
            ),
            "role_task_snapshot": _json_safe_clone(self.role_task_snapshot),
        }


@dataclass(frozen=True)
class MultiAgentRunEvent:
    """Run trace event."""

    run_id: str
    seq: int
    type: RunEventType
    payload: dict[str, Any]
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-safe dict."""
        return {
            "run_id": self.run_id,
            "seq": self.seq,
            "type": self.type,
            "payload": dict(self.payload),
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class CandidateOutput:
    """Candidate output."""

    candidate_id: str
    role_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-safe dict."""
        return {
            "candidate_id": self.candidate_id,
            "role_id": self.role_id,
            "content": self.content,
            "metadata": dict(self.metadata),
        }


@dataclass
class MultiAgentRunRequest:
    """Orchestrator request."""

    task_input: str
    protocol: Mapping[str, Any] | ProtocolConfig | None = None
    run_id: str | None = None
    reasoning_effort: str | None = None
    guidance_messages: list[str] = field(default_factory=list)
    run_policy: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiAgentRunResult:
    """Orchestrator result."""

    run_id: str
    protocol: str
    state: RunState
    task_input: str
    candidates: list[CandidateOutput]
    selected_candidate_id: str | None
    evaluation: dict[str, Any] | None
    artifacts: dict[str, Any]
    events: list[MultiAgentRunEvent]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-safe dict."""
        return {
            "run_id": self.run_id,
            "protocol": self.protocol,
            "state": self.state,
            "task_input": self.task_input,
            "candidates": [item.to_dict() for item in self.candidates],
            "selected_candidate_id": self.selected_candidate_id,
            "evaluation": dict(self.evaluation or {}),
            "artifacts": dict(self.artifacts),
            "events": [event.to_dict() for event in self.events],
            "metadata": dict(self.metadata),
        }


class RunStateTransitionError(ValueError):
    """Raised on invalid run-state transitions."""


class RunPolicyStop(RuntimeError):
    """Raised when the current run policy requires controlled early termination."""

    def __init__(
        self,
        *,
        status: Literal["partial", "awaiting_resources", "stalled"],
        action: str,
        reason: str,
        stage: str,
        checkpoint: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(reason)
        self.status = status
        self.action = action
        self.reason = reason
        self.stage = stage
        self.checkpoint = dict(checkpoint or {})


class SubagentRoleError(RuntimeError):
    """Raised when one subagent role exhausts its retry budget."""

    def __init__(
        self,
        *,
        role_id: str,
        model_id: str | None,
        stage: str,
        reason: str,
        failure_count: int,
        timeout: bool,
        cause: BaseException,
    ) -> None:
        super().__init__(reason)
        self.role_id = role_id
        self.model_id = model_id
        self.stage = stage
        self.reason = reason
        self.failure_count = failure_count
        self.timeout = timeout
        self.cause = cause


class BoundedRunStateMachine:
    """Bounded state machine for agent runs."""

    def __init__(self, *, initial_state: RunState = "queued") -> None:
        self._state: RunState = initial_state

    @property
    def state(self) -> RunState:
        return self._state

    def can_transition(self, next_state: RunState) -> bool:
        return next_state in RUN_STATE_TRANSITIONS[self._state]

    def transition(self, next_state: RunState) -> tuple[RunState, RunState]:
        if not self.can_transition(next_state):
            raise RunStateTransitionError(f"Invalid transition: {self._state} -> {next_state}")
        previous = self._state
        self._state = next_state
        return previous, self._state


class MultiAgentOrchestrator:
    """Execute teacher-student and debate runs."""

    def __init__(
        self,
        *,
        engine: Any | None = None,
        scoring_policy: LLMFirstScoringPolicy | None = None,
        exec_runtime: ExecRuntime | None = None,
        exec_approval_store: InMemoryApprovalStore | None = None,
        controlled_exec_require_approval: bool = True,
    ) -> None:
        self._engine = engine
        self._policy = scoring_policy or LLMFirstScoringPolicy()
        self._invocation_traces: list[dict[str, Any]] = []
        self._exec_runtime = exec_runtime
        self._exec_approval_store = exec_approval_store
        self._controlled_exec_require_approval = bool(controlled_exec_require_approval)
        self._run_policy: dict[str, Any] = {}
        self._deadline_monotonic: float | None = None
        self._latest_checkpoint: dict[str, Any] = {}
        self._checkpoint_count = 0
        self._role_failure_counts: dict[str, int] = {}
        self._degraded_roles: set[str] = set()
        self._subagent_health_events: list[dict[str, Any]] = []
        self._role_task_states: dict[str, dict[str, Any]] = {}
        self._task_states: dict[str, dict[str, Any]] = {}
        self._role_resume_plan: dict[str, Any] = {}
        self._default_reasoning_effort: str | None = None

    def _configure_run_policy(self, run_policy: Mapping[str, Any] | None) -> None:
        self._run_policy = dict(run_policy or {})
        self._checkpoint_count = 0
        self._latest_checkpoint = {}
        self._role_failure_counts = {}
        self._degraded_roles = set()
        self._subagent_health_events = []
        self._role_task_states = {}
        self._task_states = {}
        self._role_resume_plan = {}
        max_wall_clock_sec = self._run_policy.get("max_wall_clock_sec")
        try:
            max_wall_clock = int(max_wall_clock_sec)
        except (TypeError, ValueError):
            max_wall_clock = 0
        self._deadline_monotonic = (
            time.monotonic() + max_wall_clock if max_wall_clock > 0 else None
        )

    def _heartbeat_timeout_sec(self) -> int | None:
        value = self._run_policy.get("heartbeat_timeout_sec")
        try:
            timeout = int(value)
        except (TypeError, ValueError):
            return None
        return timeout if timeout > 0 else None

    def _max_subagent_failures_per_role(self) -> int:
        value = self._run_policy.get("max_subagent_failures_per_role")
        try:
            maximum = int(value)
        except (TypeError, ValueError):
            maximum = 2
        return max(0, maximum)

    def _on_subagent_disconnect(self) -> str:
        action = str(self._run_policy.get("on_subagent_disconnect") or "retry_then_degrade").strip().lower()
        if action not in {"retry_then_degrade", "pause", "fail"}:
            return "retry_then_degrade"
        return action

    def _record_subagent_failure(
        self,
        *,
        role_id: str,
        model_id: str | None,
        stage: str,
        reason: str,
        timeout: bool,
    ) -> int:
        next_count = int(self._role_failure_counts.get(role_id, 0) or 0) + 1
        self._role_failure_counts[role_id] = next_count
        self._subagent_health_events.append(
            {
                "type": "failure",
                "role_id": role_id,
                "model_id": model_id,
                "stage": stage,
                "reason": reason,
                "timeout": timeout,
                "failure_count": next_count,
                "recorded_at": datetime.now(UTC).isoformat(),
            }
        )
        return next_count

    def _mark_degraded_role(
        self,
        *,
        role_id: str,
        model_id: str | None,
        stage: str,
        reason: str,
    ) -> None:
        self._degraded_roles.add(role_id)
        self._subagent_health_events.append(
            {
                "type": "degraded",
                "role_id": role_id,
                "model_id": model_id,
                "stage": stage,
                "reason": reason,
                "failure_count": int(self._role_failure_counts.get(role_id, 0) or 0),
                "recorded_at": datetime.now(UTC).isoformat(),
            }
        )
        self._mark_role_task_failed(
            role_id=role_id,
            model_id=model_id,
            stage=stage,
            reason=reason,
            status="degraded",
        )
        self._mark_task_failed(
            task_key=stage,
            role_id=role_id,
            model_id=model_id,
            stage=stage,
            reason=reason,
            status="degraded",
        )

    def _build_subagent_health_snapshot(self) -> dict[str, Any]:
        return {
            "status": "degraded" if self._degraded_roles else "healthy",
            "degraded": bool(self._degraded_roles),
            "degraded_role_ids": sorted(self._degraded_roles),
            "failure_counts": dict(self._role_failure_counts),
            "events": list(self._subagent_health_events),
        }

    def _role_checkpoint_index(self) -> int | None:
        checkpoint_index = self._latest_checkpoint.get("checkpoint_index")
        try:
            value = int(checkpoint_index)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    def _touch_role_task(
        self,
        *,
        role_id: str,
        model_id: str | None = None,
        stage: str | None = None,
    ) -> dict[str, Any]:
        entry = dict(self._role_task_states.get(role_id) or {})
        entry["role_id"] = role_id
        if model_id:
            entry.setdefault("original_model_id", model_id)
            entry["assigned_model_id"] = model_id
        if stage:
            entry["stage"] = stage
        checkpoint_index = self._role_checkpoint_index()
        if checkpoint_index is not None:
            entry["checkpoint_index"] = checkpoint_index
        entry["updated_at"] = datetime.now(UTC).isoformat()
        self._role_task_states[role_id] = entry
        return entry

    def _touch_task_state(
        self,
        *,
        task_key: str,
        role_id: str,
        model_id: str | None = None,
        stage: str | None = None,
    ) -> dict[str, Any]:
        entry = dict(self._task_states.get(task_key) or {})
        entry["task_key"] = task_key
        entry["role_id"] = role_id
        if model_id:
            entry.setdefault("original_model_id", model_id)
            entry["assigned_model_id"] = model_id
        if stage:
            entry["stage"] = stage
        checkpoint_index = self._role_checkpoint_index()
        if checkpoint_index is not None:
            entry["checkpoint_index"] = checkpoint_index
        entry["updated_at"] = datetime.now(UTC).isoformat()
        self._task_states[task_key] = entry
        return entry

    def _mark_role_task_running(
        self,
        *,
        role_id: str,
        model_id: str | None,
        stage: str,
        depends_on: list[str] | None = None,
    ) -> None:
        entry = self._touch_role_task(role_id=role_id, model_id=model_id, stage=stage)
        entry["status"] = "running"
        entry["resume_action"] = "rerun"
        if depends_on:
            entry["depends_on"] = [str(item) for item in depends_on if str(item).strip()]
        self._role_task_states[role_id] = entry

    def _mark_task_running(
        self,
        *,
        task_key: str,
        role_id: str,
        model_id: str | None,
        stage: str,
        depends_on: list[str] | None = None,
    ) -> None:
        entry = self._touch_task_state(task_key=task_key, role_id=role_id, model_id=model_id, stage=stage)
        entry["status"] = "running"
        entry["resume_action"] = "rerun"
        if depends_on:
            entry["depends_on"] = [str(item) for item in depends_on if str(item).strip()]
        self._task_states[task_key] = entry

    def _mark_role_task_completed(
        self,
        *,
        role_id: str,
        model_id: str | None,
        stage: str,
        candidate: CandidateOutput | None = None,
        result_summary: Mapping[str, Any] | None = None,
    ) -> None:
        entry = self._touch_role_task(role_id=role_id, model_id=model_id, stage=stage)
        entry["status"] = "completed"
        entry["resume_action"] = "reuse_output"
        entry.pop("last_error", None)
        if candidate is not None:
            entry["candidate"] = candidate.to_dict()
            entry["candidate_id"] = candidate.candidate_id
            entry["output_preview"] = candidate.content[:240]
        if isinstance(result_summary, Mapping) and result_summary:
            entry["result_summary"] = _mapping_to_plain_dict(result_summary)
        self._role_task_states[role_id] = entry

    def _mark_task_completed(
        self,
        *,
        task_key: str,
        role_id: str,
        model_id: str | None,
        stage: str,
        candidate: CandidateOutput | None = None,
        result_summary: Mapping[str, Any] | None = None,
    ) -> None:
        entry = self._touch_task_state(task_key=task_key, role_id=role_id, model_id=model_id, stage=stage)
        entry["status"] = "completed"
        entry["resume_action"] = "reuse_output"
        entry.pop("last_error", None)
        if candidate is not None:
            entry["candidate"] = candidate.to_dict()
            entry["candidate_id"] = candidate.candidate_id
            entry["output_preview"] = candidate.content[:240]
        if isinstance(result_summary, Mapping) and result_summary:
            entry["result_summary"] = _mapping_to_plain_dict(result_summary)
        self._task_states[task_key] = entry

    def _mark_role_task_failed(
        self,
        *,
        role_id: str,
        model_id: str | None,
        stage: str,
        reason: str,
        status: str = "failed",
    ) -> None:
        entry = self._touch_role_task(role_id=role_id, model_id=model_id, stage=stage)
        entry["status"] = status
        entry["resume_action"] = "reassign_or_rerun"
        entry["last_error"] = reason
        self._role_task_states[role_id] = entry

    def _mark_task_failed(
        self,
        *,
        task_key: str,
        role_id: str,
        model_id: str | None,
        stage: str,
        reason: str,
        status: str = "failed",
    ) -> None:
        entry = self._touch_task_state(task_key=task_key, role_id=role_id, model_id=model_id, stage=stage)
        entry["status"] = status
        entry["resume_action"] = "reassign_or_rerun"
        entry["last_error"] = reason
        self._task_states[task_key] = entry

    def _mark_role_task_reused(
        self,
        *,
        role_id: str,
        stage: str,
        candidate: CandidateOutput | None = None,
    ) -> None:
        entry = self._touch_role_task(
            role_id=role_id,
            model_id=(
                str((candidate.metadata or {}).get("model_id"))
                if candidate is not None and isinstance((candidate.metadata or {}).get("model_id"), str)
                else None
            ),
            stage=stage,
        )
        entry["status"] = "completed"
        entry["resume_action"] = "reuse_output"
        if candidate is not None:
            entry["candidate"] = candidate.to_dict()
            entry["candidate_id"] = candidate.candidate_id
            entry["output_preview"] = candidate.content[:240]
        self._role_task_states[role_id] = entry

    def _mark_task_reused(
        self,
        *,
        task_key: str,
        role_id: str,
        stage: str,
        candidate: CandidateOutput | None = None,
    ) -> None:
        entry = self._touch_task_state(
            task_key=task_key,
            role_id=role_id,
            model_id=(
                str((candidate.metadata or {}).get("model_id"))
                if candidate is not None and isinstance((candidate.metadata or {}).get("model_id"), str)
                else None
            ),
            stage=stage,
        )
        entry["status"] = "completed"
        entry["resume_action"] = "reuse_output"
        if candidate is not None:
            entry["candidate"] = candidate.to_dict()
            entry["candidate_id"] = candidate.candidate_id
            entry["output_preview"] = candidate.content[:240]
        self._task_states[task_key] = entry

    def _restore_role_task_snapshot(self, snapshot: Mapping[str, Any] | None) -> None:
        if not isinstance(snapshot, Mapping):
            return
        roles = snapshot.get("roles") if isinstance(snapshot.get("roles"), Mapping) else snapshot
        tasks = snapshot.get("tasks") if isinstance(snapshot.get("tasks"), Mapping) else {}
        restored: dict[str, dict[str, Any]] = {}
        if isinstance(roles, Mapping):
            for role_id, value in roles.items():
                if not isinstance(role_id, str) or not isinstance(value, Mapping):
                    continue
                restored[role_id] = _mapping_to_plain_dict(value)
                restored[role_id]["role_id"] = role_id
        self._role_task_states = restored
        restored_tasks: dict[str, dict[str, Any]] = {}
        if isinstance(tasks, Mapping):
            for task_key, value in tasks.items():
                if not isinstance(task_key, str) or not isinstance(value, Mapping):
                    continue
                restored_tasks[task_key] = _mapping_to_plain_dict(value)
                restored_tasks[task_key]["task_key"] = task_key
        if not restored_tasks and restored:
            for role_id, value in restored.items():
                if not isinstance(value, Mapping):
                    continue
                stage = value.get("stage")
                if not isinstance(stage, str) or not stage.strip():
                    continue
                task_entry = _mapping_to_plain_dict(value)
                task_entry["task_key"] = stage
                task_entry["role_id"] = role_id
                restored_tasks[stage] = task_entry
        self._task_states = restored_tasks

    def _build_role_resume_plan(self, *, selected_models_roles: Mapping[str, str]) -> dict[str, Any]:
        available_roles = {
            str(role_id): str(model_id)
            for role_id, model_id in selected_models_roles.items()
            if isinstance(role_id, str) and role_id.strip() and isinstance(model_id, str) and model_id.strip()
        }
        available_models: list[str] = []
        model_usage: dict[str, int] = {}
        for model_id in available_roles.values():
            if model_id not in available_models:
                available_models.append(model_id)
            model_usage[model_id] = model_usage.get(model_id, 0) + 1

        assignments: dict[str, Any] = {}
        reassigned_roles: list[dict[str, Any]] = []
        blocked_roles: list[str] = []
        completed_roles: list[str] = []
        tracked_role_ids = set(self._role_task_states) | set(available_roles)
        for role_id in sorted(tracked_role_ids):
            state = dict(self._role_task_states.get(role_id) or {})
            status = str(state.get("status") or ("pending" if role_id in available_roles else "unknown"))
            assigned_model_id = available_roles.get(role_id)
            assignment_source = "current_mapping" if assigned_model_id else "unavailable"
            if assigned_model_id is None:
                if status == "completed":
                    assigned_model_id = (
                        str(state.get("assigned_model_id") or state.get("original_model_id") or "").strip() or None
                    )
                    assignment_source = "checkpoint_completed"
                    completed_roles.append(role_id)
                else:
                    previous_model_id = (
                        str(state.get("assigned_model_id") or state.get("original_model_id") or "").strip() or None
                    )
                    if previous_model_id and previous_model_id in available_models:
                        assigned_model_id = previous_model_id
                    elif available_models:
                        assigned_model_id = min(
                            available_models,
                            key=lambda item: (model_usage.get(item, 0), available_models.index(item)),
                        )
                    if assigned_model_id is not None:
                        assignment_source = "reassigned_to_available_model"
                        model_usage[assigned_model_id] = model_usage.get(assigned_model_id, 0) + 1
                        reassigned_roles.append(
                            {
                                "role_id": role_id,
                                "assigned_model_id": assigned_model_id,
                                "previous_model_id": previous_model_id,
                                "reason": "role mapping missing during resume; reassigned to an available model",
                            }
                        )
                    else:
                        blocked_roles.append(role_id)

            resume_action = (
                "reuse_output"
                if status == "completed"
                else "blocked"
                if assignment_source == "unavailable"
                else "reassign_and_rerun"
                if assignment_source == "reassigned_to_available_model"
                else "rerun"
            )
            assignments[role_id] = {
                "role_id": role_id,
                "status": status,
                "stage": state.get("stage"),
                "checkpoint_index": state.get("checkpoint_index"),
                "original_model_id": state.get("original_model_id"),
                "assigned_model_id": assigned_model_id,
                "assignment_source": assignment_source,
                "resume_action": resume_action,
                "depends_on": list(state.get("depends_on") or []),
            }

        self._role_resume_plan = {
            "available_roles": dict(available_roles),
            "assignments": assignments,
            "reassigned_roles": reassigned_roles,
            "blocked_roles": blocked_roles,
            "completed_roles": sorted(set(completed_roles)),
        }
        return dict(self._role_resume_plan)

    def _apply_role_resume_plan(
        self,
        *,
        selected_models_roles: Mapping[str, str],
        role_resume_plan: Mapping[str, Any] | None,
    ) -> dict[str, str]:
        effective = {
            str(role_id): str(model_id)
            for role_id, model_id in selected_models_roles.items()
            if isinstance(role_id, str) and role_id.strip() and isinstance(model_id, str) and model_id.strip()
        }
        assignments = (
            role_resume_plan.get("assignments")
            if isinstance(role_resume_plan, Mapping) and isinstance(role_resume_plan.get("assignments"), Mapping)
            else {}
        )
        if isinstance(assignments, Mapping):
            for role_id, value in assignments.items():
                if not isinstance(role_id, str) or not isinstance(value, Mapping):
                    continue
                assigned_model_id = value.get("assigned_model_id")
                assignment_source = str(value.get("assignment_source") or "")
                if (
                    role_id not in effective
                    and isinstance(assigned_model_id, str)
                    and assigned_model_id.strip()
                    and assignment_source == "reassigned_to_available_model"
                ):
                    effective[role_id] = assigned_model_id.strip()
                entry = self._role_task_states.get(role_id)
                if isinstance(entry, dict):
                    entry["assigned_model_id"] = assigned_model_id
                    entry["assignment_source"] = assignment_source
                    entry["resume_action"] = value.get("resume_action")
                    self._role_task_states[role_id] = entry
        return effective

    def _build_role_task_snapshot(self, *, selected_models_roles: Mapping[str, str]) -> dict[str, Any]:
        return {
            "version": 1,
            "available_roles": {
                str(role_id): str(model_id)
                for role_id, model_id in selected_models_roles.items()
                if isinstance(role_id, str) and isinstance(model_id, str)
            },
            "roles": {
                role_id: _mapping_to_plain_dict(entry)
                for role_id, entry in sorted(self._role_task_states.items())
                if isinstance(role_id, str) and isinstance(entry, Mapping)
            },
            "tasks": {
                task_key: _mapping_to_plain_dict(entry)
                for task_key, entry in sorted(self._task_states.items())
                if isinstance(task_key, str) and isinstance(entry, Mapping)
            },
            "resume_plan": _mapping_to_plain_dict(self._role_resume_plan),
        }

    def _resume_candidate_for_role(self, role_id: str) -> CandidateOutput | None:
        entry = self._role_task_states.get(role_id)
        if not isinstance(entry, Mapping):
            return None
        candidate = entry.get("candidate")
        if not isinstance(candidate, Mapping):
            return None
        payload = _candidate_outputs_from_payload([candidate])
        return payload[0] if payload else None

    def _resume_candidate_for_task(self, task_key: str) -> CandidateOutput | None:
        entry = self._task_states.get(task_key)
        if not isinstance(entry, Mapping):
            return None
        candidate = entry.get("candidate")
        if not isinstance(candidate, Mapping):
            return None
        payload = _candidate_outputs_from_payload([candidate])
        return payload[0] if payload else None

    def _resume_task_summary(self, task_key: str) -> dict[str, Any] | None:
        entry = self._task_states.get(task_key)
        if not isinstance(entry, Mapping):
            return None
        result_summary = entry.get("result_summary")
        if not isinstance(result_summary, Mapping):
            return None
        return _mapping_to_plain_dict(result_summary)

    def _build_role_runtime_context(self, role_id: str) -> str:
        entry = self._role_task_states.get(role_id)
        if not isinstance(entry, Mapping):
            return ""
        assignment = (
            self._role_resume_plan.get("assignments", {}).get(role_id)
            if isinstance(self._role_resume_plan.get("assignments"), Mapping)
            else None
        )
        lines = []
        status = str(entry.get("status") or "").strip()
        if status:
            lines.append(f"status={status}")
        stage = str(entry.get("stage") or "").strip()
        if stage:
            lines.append(f"last_stage={stage}")
        checkpoint_index = entry.get("checkpoint_index")
        if checkpoint_index is not None:
            lines.append(f"checkpoint_index={checkpoint_index}")
        if isinstance(assignment, Mapping):
            assigned_model_id = assignment.get("assigned_model_id")
            if isinstance(assigned_model_id, str) and assigned_model_id.strip():
                lines.append(f"assigned_model_id={assigned_model_id}")
            assignment_source = str(assignment.get("assignment_source") or "").strip()
            if assignment_source:
                lines.append(f"assignment_source={assignment_source}")
        output_preview = str(entry.get("output_preview") or "").strip()
        if output_preview:
            lines.append(f"previous_output={output_preview}")
        return "\n".join(lines)

    async def _await_subagent_operation(
        self,
        operation: Any,
        *,
        role_id: str,
        model_id: str | None,
        stage: str,
    ) -> Any:
        self._mark_role_task_running(role_id=role_id, model_id=model_id, stage=stage)
        self._mark_task_running(
            task_key=stage,
            role_id=role_id,
            model_id=model_id,
            stage=stage,
        )
        heartbeat_timeout_sec = self._heartbeat_timeout_sec()
        while True:
            try:
                if heartbeat_timeout_sec is not None:
                    return await asyncio.wait_for(operation(), timeout=float(heartbeat_timeout_sec))
                return await operation()
            except asyncio.TimeoutError as exc:
                failure_count = self._record_subagent_failure(
                    role_id=role_id,
                    model_id=model_id,
                    stage=stage,
                    reason=f"Subagent role {role_id} exceeded heartbeat_timeout_sec={heartbeat_timeout_sec}.",
                    timeout=True,
                )
                if failure_count <= self._max_subagent_failures_per_role():
                    continue
                self._mark_role_task_failed(
                    role_id=role_id,
                    model_id=model_id,
                    stage=stage,
                    reason=f"Subagent role {role_id} exceeded heartbeat_timeout_sec={heartbeat_timeout_sec}.",
                )
                self._mark_task_failed(
                    task_key=stage,
                    role_id=role_id,
                    model_id=model_id,
                    stage=stage,
                    reason=f"Subagent role {role_id} exceeded heartbeat_timeout_sec={heartbeat_timeout_sec}.",
                )
                raise SubagentRoleError(
                    role_id=role_id,
                    model_id=model_id,
                    stage=stage,
                    reason=f"Subagent role {role_id} exceeded heartbeat_timeout_sec={heartbeat_timeout_sec}.",
                    failure_count=failure_count,
                    timeout=True,
                    cause=exc,
                ) from exc
            except Exception as exc:
                failure_count = self._record_subagent_failure(
                    role_id=role_id,
                    model_id=model_id,
                    stage=stage,
                    reason=str(exc).strip() or exc.__class__.__name__,
                    timeout=False,
                )
                if failure_count <= self._max_subagent_failures_per_role():
                    continue
                self._mark_role_task_failed(
                    role_id=role_id,
                    model_id=model_id,
                    stage=stage,
                    reason=str(exc).strip() or exc.__class__.__name__,
                )
                self._mark_task_failed(
                    task_key=stage,
                    role_id=role_id,
                    model_id=model_id,
                    stage=stage,
                    reason=str(exc).strip() or exc.__class__.__name__,
                )
                raise SubagentRoleError(
                    role_id=role_id,
                    model_id=model_id,
                    stage=stage,
                    reason=str(exc).strip() or exc.__class__.__name__,
                    failure_count=failure_count,
                    timeout=False,
                    cause=exc,
                ) from exc

    def _raise_subagent_disconnect_stop(
        self,
        *,
        error: SubagentRoleError,
        task_input: str,
        protocol: str,
        stage: str,
        candidates: list[CandidateOutput] | None = None,
        protocol_artifacts: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        classified = None if error.timeout else classify_agent_run_recovery_issue(error.cause)
        if classified is not None:
            raise error.cause
        action = self._on_subagent_disconnect()
        if action == "fail":
            raise error.cause
        merged_extra = {
            "role_id": error.role_id,
            "model_id": error.model_id,
            "failure_count": error.failure_count,
            "timeout": error.timeout,
            **dict(extra or {}),
        }
        self._record_checkpoint(
            stage=stage,
            task_input=task_input,
            protocol=protocol,
            candidates=candidates,
            protocol_artifacts=protocol_artifacts,
            extra=merged_extra,
        )
        raise RunPolicyStop(
            status="stalled",
            action=action,
            reason=error.reason,
            stage=stage,
            checkpoint=self._latest_checkpoint,
        )

    def _record_checkpoint(
        self,
        *,
        stage: str,
        task_input: str,
        protocol: str,
        candidates: list[CandidateOutput] | None = None,
        protocol_artifacts: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        self._checkpoint_count += 1
        candidate_items = list(candidates or [])
        checkpoint = {
            "checkpoint_index": self._checkpoint_count,
            "stage": stage,
            "task_input": task_input,
            "protocol": protocol,
            "candidate_ids": [item.candidate_id for item in candidate_items],
            "candidate_count": len(candidate_items),
            "artifact_keys": sorted(
                str(key)
                for key, value in (protocol_artifacts or {}).items()
                if isinstance(key, str) and isinstance(value, dict)
            ),
            "captured_at": datetime.now(UTC).isoformat(),
        }
        if extra:
            checkpoint["extra"] = dict(extra)
        self._latest_checkpoint = checkpoint

    def _restore_checkpoint(self, checkpoint: Mapping[str, Any] | None) -> None:
        if not isinstance(checkpoint, Mapping):
            return
        self._latest_checkpoint = {
            str(key): _json_safe_clone(value)
            for key, value in checkpoint.items()
            if isinstance(key, str)
        }
        try:
            self._checkpoint_count = max(0, int(checkpoint.get("checkpoint_index") or 0))
        except (TypeError, ValueError):
            self._checkpoint_count = 0

    def _raise_if_run_policy_exhausted(
        self,
        *,
        stage: str,
        task_input: str,
        protocol: str,
        candidates: list[CandidateOutput] | None = None,
        protocol_artifacts: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        self._record_checkpoint(
            stage=stage,
            task_input=task_input,
            protocol=protocol,
            candidates=candidates,
            protocol_artifacts=protocol_artifacts,
            extra=extra,
        )
        if self._deadline_monotonic is None:
            return
        if time.monotonic() < self._deadline_monotonic:
            return
        action = str(self._run_policy.get("on_budget_exhausted") or "pause").strip().lower()
        if action not in {"pause", "finalize_partial"}:
            action = "pause"
        status: Literal["partial", "awaiting_resources"] = (
            "awaiting_resources" if action == "pause" else "partial"
        )
        max_wall_clock_sec = self._run_policy.get("max_wall_clock_sec")
        raise RunPolicyStop(
            status=status,
            action=action,
            reason=f"Run exceeded max_wall_clock_sec={max_wall_clock_sec}.",
            stage=stage,
            checkpoint=self._latest_checkpoint,
        )

    async def run(self, request: MultiAgentRunRequest) -> MultiAgentRunResult:
        """Run one multi-agent orchestration."""
        run_id = request.run_id or str(uuid4())
        protocol_config = parse_protocol_config(request.protocol)
        execution_policy = parse_subagent_execution_policy(
            _resolve_execution_policy_payload(request.protocol, request.metadata),
            legacy_protocol=protocol_config.protocol,
        )
        state_machine = BoundedRunStateMachine(initial_state="queued")
        events: list[MultiAgentRunEvent] = []
        self._invocation_traces = []
        seq = 0

        def emit(event_type: RunEventType, payload: dict[str, Any]) -> None:
            nonlocal seq
            seq += 1
            events.append(
                MultiAgentRunEvent(
                    run_id=run_id,
                    seq=seq,
                    type=event_type,
                    payload=payload,
                    timestamp=datetime.now(UTC).isoformat(),
                )
            )

        emit(
            "state_changed",
            asdict(
                StateChangedPayload(
                    previous_state=None,
                    current_state="queued",
                    reason="run_created",
                )
            ),
        )
        previous, current = state_machine.transition("running")
        emit(
            "state_changed",
            asdict(
                StateChangedPayload(
                    previous_state=previous,
                    current_state=current,
                    reason="orchestration_started",
                )
            ),
        )

        self._configure_run_policy(request.run_policy)
        self._default_reasoning_effort = request.reasoning_effort
        requested_guidance_messages = [item.strip() for item in request.guidance_messages if item.strip()]
        resume_payload = _resolve_resume_execution_payload(request.metadata)
        selected_models_roles = _resolve_selected_model_roles(request.metadata)
        effective_metadata = dict(request.metadata)
        effective_metadata["summary"] = _sanitize_resume_summary(request.metadata.get("summary"))
        effective_metadata["execution_policy"] = execution_policy_to_dict(execution_policy)
        effective_metadata["reasoning_effort"] = request.reasoning_effort
        candidates: list[CandidateOutput] = []
        precomputed_artifacts: dict[str, Any] = {}
        protocol_artifacts: dict[str, Any] = {}
        evidence_packets: list[dict[str, Any]] = []
        evidence_collection_summary: dict[str, Any] | None = None
        verifications: list[CandidateVerification] = []
        verification_summary: dict[str, Any] | None = None
        selected_candidate_id: str | None = None
        evaluation_payload: EvaluationPayload | None = None
        guidance_messages = (
            _merge_guidance_messages(resume_payload.guidance_messages, requested_guidance_messages)
            if resume_payload is not None and resume_payload.executor == "continue_from_checkpoint"
            else list(requested_guidance_messages)
        )
        if not guidance_messages:
            guidance_messages.append("Keep outputs concise and grounded.")

        for index, message in enumerate(guidance_messages, start=1):
            emit(
                "guidance",
                asdict(
                    GuidancePayload(
                        guidance_id=f"guidance-{index}",
                        content=message,
                        source="system",
                    )
                ),
            )

        def build_recovery_result(
            *,
            status: Literal["partial", "awaiting_resources", "stalled"],
            action: str,
            reason: str,
            stage: str,
            checkpoint: dict[str, Any] | None = None,
            resource_report: dict[str, Any] | None = None,
        ) -> MultiAgentRunResult:
            protocol_artifact_bundle = {**precomputed_artifacts, **protocol_artifacts}
            recovery_checkpoint = dict(checkpoint or self._latest_checkpoint or {})
            role_task_snapshot = self._build_role_task_snapshot(selected_models_roles=selected_models_roles)
            resume_state_payload = _build_resume_execution_payload(
                stage=stage,
                checkpoint=recovery_checkpoint,
                guidance_messages=guidance_messages,
                metadata=effective_metadata,
                precomputed_artifacts=precomputed_artifacts,
                protocol_artifacts=protocol_artifacts,
                candidates=candidates,
                evidence_packets=evidence_packets,
                evidence_collection_summary=evidence_collection_summary,
                verifications=verifications,
                verification_summary=verification_summary,
                evaluation_payload=evaluation_payload,
                role_task_snapshot=role_task_snapshot,
            )
            recovery_state = {
                "status": status,
                "action": action,
                "reason": reason,
                "stage": stage,
                "checkpoint": recovery_checkpoint,
                "resume_executor": resume_state_payload.executor,
                "resume_payload": resume_state_payload.to_dict(),
                "role_task_snapshot": role_task_snapshot,
                "unfinished_steps": _remaining_run_steps(
                    stage=stage,
                    research_enabled=research_policy.enabled
                    and isinstance(protocol_config, MultiAgentDebateProtocol),
                ),
            }
            if resource_report is not None:
                recovery_state["resource_exhaustion_report"] = dict(resource_report)
                recovery_state["recommended_resume_conditions"] = list(
                    resource_report.get("recommended_resume_conditions") or []
                )
            elif status == "stalled":
                recovery_state["recommended_resume_conditions"] = [
                    "Resume the run after the stalled subagent or model backend is healthy again.",
                    "Increase heartbeat_timeout_sec if the role legitimately needs a longer response window.",
                    "Inspect subagent_health_snapshot for the failing role and stage before retrying.",
                ]
            final_candidate = _candidate_by_id(candidates, selected_candidate_id)
            if final_candidate is None and candidates:
                final_candidate = candidates[-1]
            artifacts = {
                "protocol": protocol_config.protocol,
                "selected_candidate_id": selected_candidate_id,
                "final_answer": final_candidate.content if final_candidate is not None else "",
                "candidate_count": len(candidates),
                "run_guard_policy": dict(self._run_policy),
                "recovery_checkpoint": dict(checkpoint or self._latest_checkpoint or {}),
                "partial_summary": {
                    "status": status,
                    "reason": reason,
                    "stage": stage,
                    "checkpoint_index": recovery_state["checkpoint"].get("checkpoint_index"),
                    "candidate_count": len(candidates),
                    "artifact_keys": sorted(
                        str(key)
                        for key, value in protocol_artifact_bundle.items()
                        if isinstance(key, str) and isinstance(value, dict)
                    ),
                    "unfinished_steps": list(recovery_state["unfinished_steps"]),
                },
            }
            if resource_report is not None:
                artifacts["resource_exhaustion_report"] = dict(resource_report)
            if execution_policy.mode != "disabled":
                artifacts["execution_policy"] = execution_policy_to_dict(execution_policy)
            artifacts.update(protocol_artifact_bundle)
            artifacts["role_task_snapshot"] = role_task_snapshot
            if self._invocation_traces:
                artifacts["subagent_runtime"] = _build_subagent_runtime_artifact(self._invocation_traces)
            if self._subagent_health_events:
                artifacts["subagent_health_snapshot"] = self._build_subagent_health_snapshot()
            if evidence_collection_summary is not None:
                artifacts["evidence_collection"] = evidence_collection_summary
            if verification_summary is not None:
                artifacts["verification"] = verification_summary
            for artifact_name, artifact_content in artifacts.items():
                if isinstance(artifact_content, dict):
                    emit(
                        "artifact",
                        asdict(ArtifactPayload(name=artifact_name, content=artifact_content)),
                    )
            previous_state, current_state = state_machine.transition(status)
            emit(
                "state_changed",
                asdict(
                    StateChangedPayload(
                        previous_state=previous_state,
                        current_state=current_state,
                        reason=(
                            "run_policy_exhausted"
                            if resource_report is None
                            else "resource_recovery_triggered"
                        ),
                    )
                ),
            )
            metadata = dict(effective_metadata)
            metadata["recovery_state"] = recovery_state
            metadata["degraded"] = bool(self._degraded_roles)
            return MultiAgentRunResult(
                run_id=run_id,
                protocol=protocol_config.protocol,
                state=state_machine.state,
                task_input=request.task_input,
                candidates=candidates,
                selected_candidate_id=selected_candidate_id,
                evaluation=asdict(evaluation_payload) if evaluation_payload is not None else None,
                artifacts=artifacts,
                events=events,
                metadata=metadata,
            )

        resume_stage: str | None = None
        if resume_payload is not None and resume_payload.executor == "continue_from_checkpoint":
            resume_stage = resume_payload.stage
            self._restore_checkpoint(resume_payload.checkpoint)
            guidance_messages = _merge_guidance_messages(resume_payload.guidance_messages, guidance_messages)
            effective_metadata = _apply_resume_metadata_state(
                effective_metadata,
                resume_payload.metadata_state,
            )
            precomputed_artifacts.update(resume_payload.precomputed_artifacts)
            protocol_artifacts.update(resume_payload.protocol_artifacts)
            protocol_artifacts = {**precomputed_artifacts, **protocol_artifacts}
            candidates = list(resume_payload.candidates)
            evidence_packets = [_json_safe_clone(item) for item in resume_payload.evidence_packets]
            evidence_collection_summary = (
                _json_safe_clone(resume_payload.evidence_collection_summary)
                if isinstance(resume_payload.evidence_collection_summary, dict)
                else None
            )
            verifications = list(resume_payload.verifications)
            verification_summary = (
                _json_safe_clone(resume_payload.verification_summary)
                if isinstance(resume_payload.verification_summary, dict)
                else None
            )
            evaluation_payload = resume_payload.evaluation
            if evaluation_payload is not None:
                selected_candidate_id = evaluation_payload.selected_candidate_id
            self._restore_role_task_snapshot(resume_payload.role_task_snapshot)

        role_resume_plan = self._build_role_resume_plan(selected_models_roles=selected_models_roles)
        selected_models_roles = self._apply_role_resume_plan(
            selected_models_roles=selected_models_roles,
            role_resume_plan=role_resume_plan,
        )
        effective_metadata["role_resume_plan"] = _mapping_to_plain_dict(role_resume_plan)
        research_policy = ResearchDebatePolicy.from_metadata(effective_metadata)

        try:
            if (
                research_policy.enabled
                and isinstance(protocol_config, MultiAgentDebateProtocol)
                and not _resume_stage_covers(resume_stage, "research_context_prepared")
            ):
                (
                    effective_metadata,
                    precomputed_artifacts,
                    evidence_packets,
                    evidence_collection_summary,
                ) = await self._prepare_research_debate_context(
                    task_input=request.task_input,
                    protocol_config=protocol_config,
                    guidance_messages=guidance_messages,
                    selected_models_roles=selected_models_roles,
                    metadata=effective_metadata,
                    policy=research_policy,
                )
                self._raise_if_run_policy_exhausted(
                    stage="research_context_prepared",
                    task_input=request.task_input,
                    protocol=protocol_config.protocol,
                    protocol_artifacts=precomputed_artifacts,
                )
            if not _resume_stage_covers(resume_stage, "controlled_execution_context_prepared"):
                (
                    guidance_messages,
                    controlled_execution_artifacts,
                ) = await self._prepare_controlled_execution_context(
                    task_input=request.task_input,
                    protocol_config=protocol_config,
                    guidance_messages=guidance_messages,
                    selected_models_roles=selected_models_roles,
                    execution_policy=execution_policy,
                    metadata=effective_metadata,
                    emit=emit,
                )
                precomputed_artifacts.update(controlled_execution_artifacts)
                self._raise_if_run_policy_exhausted(
                    stage="controlled_execution_context_prepared",
                    task_input=request.task_input,
                    protocol=protocol_config.protocol,
                    protocol_artifacts=precomputed_artifacts,
                )
            if not _resume_stage_covers(resume_stage, "protocol_completed"):
                protocol_metadata = dict(effective_metadata)
                candidates, protocol_artifacts = await self._run_protocol(
                    request.task_input,
                    protocol_config=protocol_config,
                    guidance_messages=guidance_messages,
                    selected_models_roles=selected_models_roles,
                    execution_policy=execution_policy,
                    metadata=protocol_metadata,
                    emit=emit,
                )
                protocol_artifacts = {**precomputed_artifacts, **protocol_artifacts}
                self._raise_if_run_policy_exhausted(
                    stage="protocol_completed",
                    task_input=request.task_input,
                    protocol=protocol_config.protocol,
                    candidates=candidates,
                    protocol_artifacts=protocol_artifacts,
                )
                for artifact_name, artifact_content in protocol_artifacts.items():
                    if isinstance(artifact_content, dict):
                        emit(
                            "artifact",
                            asdict(ArtifactPayload(name=artifact_name, content=artifact_content)),
                        )
            else:
                protocol_artifacts = {**precomputed_artifacts, **protocol_artifacts}
            if not _resume_stage_covers(resume_stage, "evidence_collected") and evidence_collection_summary is None:
                self._raise_if_run_policy_exhausted(
                    stage="before_evidence_collection",
                    task_input=request.task_input,
                    protocol=protocol_config.protocol,
                    candidates=candidates,
                    protocol_artifacts=protocol_artifacts,
                )
                evidence_packets, evidence_collection_summary = await self._collect_evidence_packets(
                    metadata=effective_metadata,
                )
            if evidence_collection_summary is not None:
                emit(
                    "artifact",
                    asdict(
                        ArtifactPayload(
                            name="evidence_summary",
                            content=dict(evidence_collection_summary),
                        )
                    ),
                )
            self._raise_if_run_policy_exhausted(
                stage="evidence_collected",
                task_input=request.task_input,
                protocol=protocol_config.protocol,
                candidates=candidates,
                protocol_artifacts=protocol_artifacts,
                extra={"evidence_packet_count": len(evidence_packets)},
            )
            if not _resume_stage_covers(resume_stage, "verification_completed"):
                verifications, verification_summary = await self._verify_candidates(
                    task_input=request.task_input,
                    protocol_config=protocol_config,
                    guidance_messages=guidance_messages,
                    selected_models_roles=selected_models_roles,
                    candidates=candidates,
                    evidence_packets=evidence_packets,
                    protocol_artifacts=protocol_artifacts,
                    metadata=effective_metadata,
                )
                if verification_summary is not None:
                    emit(
                        "verification",
                        asdict(
                            VerificationPayload(
                                verifier_model_id=verification_summary.get("verifier_model_id"),
                                evidence_packet_count=int(verification_summary.get("evidence_packet_count", 0)),
                                verifications=[
                                    verification.to_dict()
                                    for verification in verifications
                                ],
                            )
                        ),
                    )
            self._raise_if_run_policy_exhausted(
                stage="verification_completed",
                task_input=request.task_input,
                protocol=protocol_config.protocol,
                candidates=candidates,
                protocol_artifacts=protocol_artifacts,
                extra={"verification_count": len(verifications)},
            )
            if not _resume_stage_covers(resume_stage, "evaluation_completed"):
                scores, preferred_candidate_id = await self._evaluate_candidates(
                    task_input=request.task_input,
                    protocol_config=protocol_config,
                    guidance_messages=guidance_messages,
                    selected_models_roles=selected_models_roles,
                    candidates=candidates,
                    protocol_artifacts=protocol_artifacts,
                    metadata=effective_metadata,
                )
                scores = _merge_scores_with_verifications(scores, verifications)
                selected_candidate_id = _select_candidate_id(
                    scores=scores,
                    candidates=candidates,
                    preferred_candidate_id=preferred_candidate_id,
                )
                evaluation_payload = EvaluationPayload(
                    policy=self._policy.to_dict(),
                    selected_candidate_id=selected_candidate_id,
                    scores=[score.to_dict() for score in scores],
                )
            elif evaluation_payload is not None:
                selected_candidate_id = evaluation_payload.selected_candidate_id
            self._raise_if_run_policy_exhausted(
                stage="evaluation_completed",
                task_input=request.task_input,
                protocol=protocol_config.protocol,
                candidates=candidates,
                protocol_artifacts=protocol_artifacts,
                extra={"selected_candidate_id": selected_candidate_id},
            )
            if (
                research_policy.enabled
                and isinstance(protocol_config, MultiAgentDebateProtocol)
                and not _resume_stage_covers(resume_stage, "research_artifacts_completed")
            ):
                research_artifacts = await self._build_research_result_artifacts(
                    task_input=request.task_input,
                    selected_models_roles=selected_models_roles,
                    candidates=candidates,
                    selected_candidate_id=selected_candidate_id,
                    protocol_artifacts=protocol_artifacts,
                    verification_summary=verification_summary,
                    metadata=effective_metadata,
                    policy=research_policy,
                )
                for artifact_name, artifact_content in research_artifacts.items():
                    if isinstance(artifact_content, dict):
                        emit(
                            "artifact",
                            asdict(ArtifactPayload(name=artifact_name, content=artifact_content)),
                        )
                protocol_artifacts.update(research_artifacts)
                self._raise_if_run_policy_exhausted(
                    stage="research_artifacts_completed",
                    task_input=request.task_input,
                    protocol=protocol_config.protocol,
                    candidates=candidates,
                    protocol_artifacts=protocol_artifacts,
                )
            if evaluation_payload is not None:
                emit("evaluation", asdict(evaluation_payload))

            final_candidate = _candidate_by_id(candidates, selected_candidate_id)
            role_task_snapshot = self._build_role_task_snapshot(selected_models_roles=selected_models_roles)
            artifacts = {
                "protocol": protocol_config.protocol,
                "selected_candidate_id": selected_candidate_id,
                "final_answer": final_candidate.content if final_candidate is not None else "",
                "candidate_count": len(candidates),
                "run_guard_policy": dict(self._run_policy),
                "role_task_snapshot": role_task_snapshot,
            }
            if execution_policy.mode != "disabled":
                artifacts["execution_policy"] = execution_policy_to_dict(execution_policy)
            artifacts.update(protocol_artifacts)
            emit(
                "artifact",
                asdict(ArtifactPayload(name="role_task_snapshot", content=role_task_snapshot)),
            )
            if self._invocation_traces:
                subagent_runtime = _build_subagent_runtime_artifact(self._invocation_traces)
                artifacts["subagent_runtime"] = subagent_runtime
                emit("artifact", asdict(ArtifactPayload(name="subagent_runtime", content=subagent_runtime)))
            if self._subagent_health_events:
                subagent_health_snapshot = self._build_subagent_health_snapshot()
                artifacts["subagent_health_snapshot"] = subagent_health_snapshot
                emit(
                    "artifact",
                    asdict(ArtifactPayload(name="subagent_health_snapshot", content=subagent_health_snapshot)),
                )
            if evidence_collection_summary is not None:
                artifacts["evidence_collection"] = evidence_collection_summary
            if verification_summary is not None:
                artifacts["verification"] = verification_summary
            emit("artifact", asdict(ArtifactPayload(name="run_summary", content=dict(artifacts))))

            previous, current = state_machine.transition("succeeded")
            emit(
                "state_changed",
                asdict(
                    StateChangedPayload(
                        previous_state=previous,
                        current_state=current,
                        reason="run_completed",
                    )
                ),
            )

            return MultiAgentRunResult(
                run_id=run_id,
                protocol=protocol_config.protocol,
                state=state_machine.state,
                task_input=request.task_input,
                candidates=candidates,
                selected_candidate_id=selected_candidate_id,
                evaluation=asdict(evaluation_payload),
                artifacts=artifacts,
                events=events,
                metadata={**effective_metadata, "degraded": bool(self._degraded_roles)},
            )
        except RunPolicyStop as stop:
            return build_recovery_result(
                status=stop.status,
                action=stop.action,
                reason=stop.reason,
                stage=stop.stage,
                checkpoint=stop.checkpoint,
            )
        except SubagentRoleError as exc:
            try:
                self._raise_subagent_disconnect_stop(
                    error=exc,
                    task_input=request.task_input,
                    protocol=protocol_config.protocol,
                    stage=exc.stage,
                    candidates=candidates,
                    protocol_artifacts={**precomputed_artifacts, **protocol_artifacts},
                )
            except RunPolicyStop as stop:
                return build_recovery_result(
                    status=stop.status,
                    action=stop.action,
                    reason=stop.reason,
                    stage=stop.stage,
                    checkpoint=stop.checkpoint,
                )
            except Exception as inner_exc:
                classified = classify_agent_run_recovery_issue(inner_exc)
                if classified is None:
                    raise
                resource_report = build_resource_exhaustion_report(
                    classified,
                    stage=exc.stage,
                    exception=inner_exc,
                )
                return build_recovery_result(
                    status="awaiting_resources",
                    action="pause",
                    reason=classified.reason,
                    stage=exc.stage,
                    checkpoint=self._latest_checkpoint,
                    resource_report=resource_report,
                )
        except Exception as exc:
            classified = classify_agent_run_recovery_issue(exc)
            if classified is None:
                raise
            stage = str(
                (self._latest_checkpoint.get("stage") if isinstance(self._latest_checkpoint, dict) else None)
                or "orchestration_interrupted"
            )
            if not self._latest_checkpoint:
                self._record_checkpoint(
                    stage=stage,
                    task_input=request.task_input,
                    protocol=protocol_config.protocol,
                    candidates=candidates,
                    protocol_artifacts={**precomputed_artifacts, **protocol_artifacts},
                    extra={"error": str(exc)},
                )
            resource_report = build_resource_exhaustion_report(
                classified,
                stage=stage,
                exception=exc,
            )
            return build_recovery_result(
                status="awaiting_resources",
                action="pause",
                reason=classified.reason,
                stage=stage,
                checkpoint=self._latest_checkpoint,
                resource_report=resource_report,
            )

    async def _run_protocol(
        self,
        task_input: str,
        *,
        protocol_config: ProtocolConfig,
        guidance_messages: list[str],
        selected_models_roles: dict[str, str],
        execution_policy: SubagentExecutionPolicy,
        metadata: dict[str, Any],
        emit: Any,
    ) -> tuple[list[CandidateOutput], dict[str, Any]]:
        if self._can_use_model_backed_execution(selected_models_roles):
            candidates, protocol_artifacts = await self._run_model_backed_protocol(
                task_input,
                protocol_config=protocol_config,
                guidance_messages=guidance_messages,
                selected_models_roles=selected_models_roles,
                execution_policy=execution_policy,
                metadata=metadata,
                emit=emit,
            )
            if candidates:
                return candidates, protocol_artifacts
        return self._run_placeholder_protocol(
            task_input,
            protocol_config=protocol_config,
            emit=emit,
        )

    def _can_use_model_backed_execution(self, selected_models_roles: dict[str, str]) -> bool:
        return bool(selected_models_roles) and (
            callable(getattr(self._engine, "invoke", None))
            or callable(getattr(self._engine, "generate_with_configured_model", None))
        )

    async def _run_model_backed_protocol(
        self,
        task_input: str,
        *,
        protocol_config: ProtocolConfig,
        guidance_messages: list[str],
        selected_models_roles: dict[str, str],
        execution_policy: SubagentExecutionPolicy,
        metadata: dict[str, Any],
        emit: Any,
    ) -> tuple[list[CandidateOutput], dict[str, Any]]:
        ordered_models = list(selected_models_roles.values())
        if isinstance(protocol_config, TeacherStudentDistillProtocol):
            roles = build_teacher_student_roles(
                teacher_role_id=protocol_config.teacher_role_id,
                student_role_id=protocol_config.student_role_id,
            )
            teacher_role = roles[0]
            student_role = roles[1]
            teacher_model_id = _resolve_protocol_role_model_id(
                role_id=teacher_role.role_id,
                selected_models_roles=selected_models_roles,
                ordered_fallback=ordered_models,
                fallback_index=0,
            )
            student_model_id = _resolve_protocol_role_model_id(
                role_id=student_role.role_id,
                selected_models_roles=selected_models_roles,
                ordered_fallback=ordered_models,
                fallback_index=1,
                default_model_id=teacher_model_id,
            )
            teacher_output = self._resume_candidate_for_task("teacher_generation")
            student_output = self._resume_candidate_for_task("student_generation")
            if teacher_output is None and not teacher_model_id:
                return [], {}
            if student_output is None and not student_model_id:
                return [], {}

            if teacher_output is not None:
                self._mark_role_task_reused(
                    role_id=teacher_role.role_id,
                    stage="teacher_generation",
                    candidate=teacher_output,
                )
                self._mark_task_reused(
                    task_key="teacher_generation",
                    role_id=teacher_role.role_id,
                    stage="teacher_generation",
                    candidate=teacher_output,
                )
            else:
                try:
                    teacher_output = await self._generate_role_candidate(
                        role_id=teacher_role.role_id,
                        role_title=teacher_role.title,
                        role_instruction=teacher_role.instruction,
                        model_id=teacher_model_id,
                        task_input=task_input,
                        guidance_messages=guidance_messages,
                        supporting_candidates=[],
                        stage="teacher_generation",
                    )
                    emit(
                        "role_output",
                        asdict(
                            RoleOutputPayload(
                                role_id=teacher_output.role_id,
                                content=teacher_output.content,
                                round_index=1,
                                candidate_id=teacher_output.candidate_id,
                                model_id=teacher_model_id,
                            )
                        )
                    )
                except SubagentRoleError as exc:
                    if (
                        self._on_subagent_disconnect() == "retry_then_degrade"
                        and classify_agent_run_recovery_issue(exc.cause) is None
                    ):
                        self._mark_degraded_role(
                            role_id=exc.role_id,
                            model_id=exc.model_id,
                            stage=exc.stage,
                            reason=exc.reason,
                        )
                    else:
                        self._raise_subagent_disconnect_stop(
                            error=exc,
                            task_input=task_input,
                            protocol=protocol_config.protocol,
                            stage=exc.stage,
                        )

            if student_output is not None:
                self._mark_role_task_reused(
                    role_id=student_role.role_id,
                    stage="student_generation",
                    candidate=student_output,
                )
                self._mark_task_reused(
                    task_key="student_generation",
                    role_id=student_role.role_id,
                    stage="student_generation",
                    candidate=student_output,
                )
            else:
                try:
                    student_output = await self._generate_role_candidate(
                        role_id=student_role.role_id,
                        role_title=student_role.title,
                        role_instruction=student_role.instruction,
                        model_id=student_model_id,
                        task_input=task_input,
                        guidance_messages=guidance_messages,
                        supporting_candidates=[teacher_output] if teacher_output is not None else [],
                        stage="student_generation",
                    )
                    emit(
                        "role_output",
                        asdict(
                            RoleOutputPayload(
                                role_id=student_output.role_id,
                                content=student_output.content,
                                round_index=1,
                                candidate_id=student_output.candidate_id,
                                model_id=student_model_id,
                            )
                        )
                    )
                except SubagentRoleError as exc:
                    if (
                        teacher_output is not None
                        and self._on_subagent_disconnect() == "retry_then_degrade"
                        and classify_agent_run_recovery_issue(exc.cause) is None
                    ):
                        self._mark_degraded_role(
                            role_id=exc.role_id,
                            model_id=exc.model_id,
                            stage=exc.stage,
                            reason=exc.reason,
                        )
                        return [teacher_output], {}
                    self._raise_subagent_disconnect_stop(
                        error=exc,
                        task_input=task_input,
                        protocol=protocol_config.protocol,
                        stage=exc.stage,
                        candidates=[teacher_output] if teacher_output is not None else [],
                    )
            return [item for item in (teacher_output, student_output) if item is not None], {}

        if isinstance(protocol_config, DrZeroSelfEvolveProtocol):
            return await self._run_model_backed_dr_zero(
                task_input=task_input,
                protocol_config=protocol_config,
                guidance_messages=guidance_messages,
                selected_models_roles=selected_models_roles,
                ordered_models=ordered_models,
                emit=emit,
            )

        if isinstance(protocol_config, ControlledSubagentExecutionProtocol):
            return await self._run_model_backed_controlled_execution(
                task_input=task_input,
                execution_policy=execution_policy,
                guidance_messages=guidance_messages,
                selected_models_roles=selected_models_roles,
                metadata=metadata,
                ordered_models=ordered_models,
                emit=emit,
            )

        return await self._run_model_backed_debate(
            task_input=task_input,
            protocol_config=protocol_config,
            guidance_messages=guidance_messages,
            selected_models_roles=selected_models_roles,
            metadata=metadata,
            ordered_models=ordered_models,
            emit=emit,
        )

    async def _run_model_backed_dr_zero(
        self,
        *,
        task_input: str,
        protocol_config: DrZeroSelfEvolveProtocol,
        guidance_messages: list[str],
        selected_models_roles: dict[str, str],
        ordered_models: list[str],
        emit: Any,
    ) -> tuple[list[CandidateOutput], dict[str, Any]]:
        roles = build_dr_zero_roles(
            proposer_role_id=protocol_config.proposer_role_id,
            solver_role_id=protocol_config.solver_role_id,
            verifier_role_id=protocol_config.verifier_role_id,
        )
        proposer_role = roles[0]
        solver_role = roles[1]
        proposer_model_id = _resolve_protocol_role_model_id(
            role_id=proposer_role.role_id,
            selected_models_roles=selected_models_roles,
            ordered_fallback=ordered_models,
            fallback_index=0,
        )
        solver_model_id = _resolve_protocol_role_model_id(
            role_id=solver_role.role_id,
            selected_models_roles=selected_models_roles,
            ordered_fallback=ordered_models,
            fallback_index=1,
            default_model_id=proposer_model_id,
        )
        if not proposer_model_id or not solver_model_id:
            return [], {}

        proposer_prompt = _build_dr_zero_proposer_prompt(
            task_input=task_input,
            sample_size=protocol_config.proposal_sample_size,
            guidance_messages=guidance_messages,
        )
        proposer_task_key = "dr_zero_proposer"
        proposer_output = self._resume_candidate_for_task(proposer_task_key)
        if proposer_output is not None:
            self._mark_role_task_reused(
                role_id=proposer_role.role_id,
                stage=proposer_task_key,
                candidate=proposer_output,
            )
            self._mark_task_reused(
                task_key=proposer_task_key,
                role_id=proposer_role.role_id,
                stage=proposer_task_key,
                candidate=proposer_output,
            )
        else:
            try:
                proposer_output = await self._generate_role_candidate(
                    role_id=proposer_role.role_id,
                    role_title=proposer_role.title,
                    role_instruction=proposer_role.instruction,
                    model_id=proposer_model_id,
                    task_input=task_input,
                    guidance_messages=guidance_messages,
                    supporting_candidates=[],
                    prompt_override=proposer_prompt,
                    stage=proposer_task_key,
                )
            except SubagentRoleError as exc:
                self._raise_subagent_disconnect_stop(
                    error=exc,
                    task_input=task_input,
                    protocol=protocol_config.protocol,
                    stage=exc.stage,
                )
            emit(
                "role_output",
                asdict(
                    RoleOutputPayload(
                        role_id=proposer_output.role_id,
                        content=proposer_output.content,
                        round_index=1,
                        candidate_id=proposer_output.candidate_id,
                        model_id=proposer_model_id,
                    )
                ),
            )
        self._raise_if_run_policy_exhausted(
            stage="dr_zero_proposals_generated",
            task_input=task_input,
            protocol=protocol_config.protocol,
            candidates=[proposer_output],
            extra={"synthetic_task_count": protocol_config.proposal_sample_size},
        )

        synthetic_tasks, parse_diagnostics = _parse_dr_zero_synthetic_tasks(
            proposer_output.content,
            task_input=task_input,
            sample_size=protocol_config.proposal_sample_size,
        )
        solver_outputs: list[CandidateOutput] = []
        solver_rollouts: list[dict[str, Any]] = []
        for task_index, synthetic_task in enumerate(synthetic_tasks, start=1):
            for rollout_index in range(1, protocol_config.solver_rollouts_per_task + 1):
                candidate_id = f"{solver_role.role_id}_{task_index}_{rollout_index}"
                solver_task_key = f"dr_zero_solver:{synthetic_task['task_id']}:{rollout_index}"
                solver_prompt = _build_dr_zero_solver_prompt(
                    root_task=task_input,
                    synthetic_task=synthetic_task,
                    guidance_messages=guidance_messages,
                    rollout_index=rollout_index,
                )
                solver_output = self._resume_candidate_for_task(solver_task_key)
                if solver_output is None:
                    try:
                        raw_solver_output = await self._generate_role_candidate(
                            role_id=solver_role.role_id,
                            role_title=solver_role.title,
                            role_instruction=solver_role.instruction,
                            model_id=solver_model_id,
                            task_input=task_input,
                            guidance_messages=guidance_messages,
                            supporting_candidates=[proposer_output],
                            prompt_override=solver_prompt,
                            stage=solver_task_key,
                        )
                    except SubagentRoleError as exc:
                        if (
                            self._on_subagent_disconnect() == "retry_then_degrade"
                            and classify_agent_run_recovery_issue(exc.cause) is None
                        ):
                            self._mark_degraded_role(
                                role_id=exc.role_id,
                                model_id=exc.model_id,
                                stage=exc.stage,
                                reason=exc.reason,
                            )
                            continue
                        self._raise_subagent_disconnect_stop(
                            error=exc,
                            task_input=task_input,
                            protocol=protocol_config.protocol,
                            stage=exc.stage,
                            candidates=solver_outputs,
                            extra={"synthetic_task_id": synthetic_task["task_id"], "rollout_index": rollout_index},
                        )
                    solver_output = CandidateOutput(
                        candidate_id=candidate_id,
                        role_id=solver_role.role_id,
                        content=raw_solver_output.content,
                        metadata={
                            **dict(raw_solver_output.metadata),
                            "synthetic_task": dict(synthetic_task),
                            "synthetic_task_id": synthetic_task["task_id"],
                            "task_index": task_index,
                            "rollout_index": rollout_index,
                        },
                    )
                    self._mark_role_task_completed(
                        role_id=solver_role.role_id,
                        model_id=solver_model_id,
                        stage=solver_task_key,
                        candidate=solver_output,
                    )
                    self._mark_task_completed(
                        task_key=solver_task_key,
                        role_id=solver_role.role_id,
                        model_id=solver_model_id,
                        stage=solver_task_key,
                        candidate=solver_output,
                    )
                else:
                    self._mark_role_task_reused(
                        role_id=solver_role.role_id,
                        stage=solver_task_key,
                        candidate=solver_output,
                    )
                    self._mark_task_reused(
                        task_key=solver_task_key,
                        role_id=solver_role.role_id,
                        stage=solver_task_key,
                        candidate=solver_output,
                    )
                solver_outputs.append(solver_output)
                solver_rollout = {
                    "candidate_id": solver_output.candidate_id,
                    "role_id": solver_output.role_id,
                    "model_id": solver_model_id,
                    "synthetic_task_id": synthetic_task["task_id"],
                    "task_index": task_index,
                    "rollout_index": rollout_index,
                    "content": solver_output.content,
                    "metadata": dict(solver_output.metadata),
                }
                solver_rollouts.append(solver_rollout)
                emit(
                    "role_output",
                    asdict(
                        RoleOutputPayload(
                            role_id=solver_output.role_id,
                            content=solver_output.content,
                            round_index=1,
                            candidate_id=solver_output.candidate_id,
                            model_id=solver_model_id,
                        )
                    ),
                )
                self._raise_if_run_policy_exhausted(
                    stage=f"dr_zero_rollout_{task_index}_{rollout_index}",
                    task_input=task_input,
                    protocol=protocol_config.protocol,
                    candidates=solver_outputs,
                    extra={
                        "synthetic_task_id": synthetic_task["task_id"],
                        "solver_rollout_count": len(solver_rollouts),
                    },
                )

        if not solver_outputs:
            raise RuntimeError("Dr.Zero solver produced no successful rollouts.")

        verifier_model_id = (
            selected_models_roles.get(protocol_config.verifier_role_id)
            or selected_models_roles.get("verifier")
            or selected_models_roles.get("judge")
        )
        protocol_artifacts = {
            "synthetic_tasks": {
                "protocol": protocol_config.protocol,
                "root_task": task_input,
                "proposer_role_id": proposer_role.role_id,
                "proposer_model_id": proposer_model_id,
                "tasks": synthetic_tasks,
                "parse_diagnostics": parse_diagnostics,
            },
            "solver_rollouts": {
                "protocol": protocol_config.protocol,
                "solver_role_id": solver_role.role_id,
                "solver_model_id": solver_model_id,
                "rollout_count": len(solver_rollouts),
                "rollouts": solver_rollouts,
            },
            "reward_summary": {
                "protocol": protocol_config.protocol,
                "status": "pending_verification",
                "basis": "verifier_and_evaluator_downstream",
                "verifier_role_id": protocol_config.verifier_role_id,
                "verifier_model_id": verifier_model_id,
                "synthetic_task_count": len(synthetic_tasks),
                "solver_rollout_count": len(solver_rollouts),
            },
            "curriculum_state": {
                "protocol": protocol_config.protocol,
                "iterations_configured": protocol_config.iterations,
                "current_iteration": 1,
                "proposal_sample_size": protocol_config.proposal_sample_size,
                "solver_rollouts_per_task": protocol_config.solver_rollouts_per_task,
                "next_iteration_ready": False,
            },
            "drzero_iteration_summary": {
                "protocol": protocol_config.protocol,
                "iteration_index": 1,
                "proposer_model_id": proposer_model_id,
                "solver_model_id": solver_model_id,
                "synthetic_task_count": len(synthetic_tasks),
                "solver_rollout_count": len(solver_rollouts),
                "proposal_parse_status": parse_diagnostics["status"],
            },
        }
        return solver_outputs, protocol_artifacts

    async def _prepare_controlled_execution_context(
        self,
        *,
        task_input: str,
        protocol_config: ProtocolConfig,
        guidance_messages: list[str],
        selected_models_roles: dict[str, str],
        execution_policy: SubagentExecutionPolicy,
        metadata: dict[str, Any],
        emit: Any,
    ) -> tuple[list[str], dict[str, Any]]:
        if execution_policy.mode != "controlled":
            return guidance_messages, {}
        if protocol_config.protocol == LEGACY_CONTROLLED_SUBAGENT_PROTOCOL:
            return guidance_messages, {}
        if not self._can_use_model_backed_execution(selected_models_roles):
            return guidance_messages, {}

        coordinator = SubagentExecutionCoordinator(
            generate_role_candidate=self._generate_role_candidate,
            invoke_text=self._invoke_configured_text,
            exec_runtime=self._exec_runtime,
            exec_approval_store=self._exec_approval_store,
            require_approval=self._controlled_exec_require_approval,
        )
        _raw_candidates, artifacts = await coordinator.run(
            task_input=task_input,
            execution_policy=execution_policy,
            guidance_messages=guidance_messages,
            selected_models_roles=selected_models_roles,
            ordered_models=list(selected_models_roles.values()),
            metadata=metadata,
            emit=emit,
            resolve_model_id=_resolve_protocol_role_model_id,
            primary_workflow=False,
            resume_hooks=self._build_controlled_execution_resume_hooks(),
        )
        if not artifacts:
            return guidance_messages, {}

        summary = artifacts.get("evaluation_summary")
        summary_text = summary.get("summary") if isinstance(summary, dict) else None
        if isinstance(summary_text, str) and summary_text.strip():
            enriched_guidance = list(guidance_messages)
            enriched_guidance.append(
                "Controlled execution context summary:\n"
                + summary_text.strip()
            )
            return enriched_guidance, artifacts
        return guidance_messages, artifacts

    async def _run_model_backed_controlled_execution(
        self,
        *,
        task_input: str,
        execution_policy: SubagentExecutionPolicy,
        guidance_messages: list[str],
        selected_models_roles: dict[str, str],
        metadata: dict[str, Any],
        ordered_models: list[str],
        emit: Any,
    ) -> tuple[list[CandidateOutput], dict[str, Any]]:
        coordinator = SubagentExecutionCoordinator(
            generate_role_candidate=self._generate_role_candidate,
            invoke_text=self._invoke_configured_text,
            exec_runtime=self._exec_runtime,
            exec_approval_store=self._exec_approval_store,
            require_approval=self._controlled_exec_require_approval,
        )
        raw_candidates, protocol_artifacts = await coordinator.run(
            task_input=task_input,
            execution_policy=execution_policy,
            guidance_messages=guidance_messages,
            selected_models_roles=selected_models_roles,
            ordered_models=ordered_models,
            metadata=metadata,
            emit=emit,
            resolve_model_id=_resolve_protocol_role_model_id,
            resume_hooks=self._build_controlled_execution_resume_hooks(),
        )
        candidates = [
            CandidateOutput(
                candidate_id=str(item.get("candidate_id") or ""),
                role_id=str(item.get("role_id") or ""),
                content=str(item.get("content") or ""),
                metadata=dict(item.get("metadata") or {}),
            )
            for item in raw_candidates
        ]
        return candidates, protocol_artifacts

    def _build_controlled_execution_resume_hooks(self) -> ControlledExecutionResumeHooks:
        def _candidate_from_payload(payload: Mapping[str, Any] | None) -> CandidateOutput | None:
            candidates = _candidate_outputs_from_payload([payload] if isinstance(payload, Mapping) else [])
            return candidates[0] if candidates else None

        def _mark_running(
            *,
            task_key: str,
            role_id: str,
            model_id: str | None,
            stage: str,
            depends_on: list[str] | None = None,
        ) -> None:
            self._mark_role_task_running(
                role_id=role_id,
                model_id=model_id,
                stage=stage,
                depends_on=depends_on,
            )
            self._mark_task_running(
                task_key=task_key,
                role_id=role_id,
                model_id=model_id,
                stage=stage,
                depends_on=depends_on,
            )

        def _mark_completed(
            *,
            task_key: str,
            role_id: str,
            model_id: str | None,
            stage: str,
            candidate: Mapping[str, Any] | None = None,
            result_summary: Mapping[str, Any] | None = None,
        ) -> None:
            restored_candidate = _candidate_from_payload(candidate)
            self._mark_role_task_completed(
                role_id=role_id,
                model_id=model_id,
                stage=stage,
                candidate=restored_candidate,
                result_summary=result_summary,
            )
            self._mark_task_completed(
                task_key=task_key,
                role_id=role_id,
                model_id=model_id,
                stage=stage,
                candidate=restored_candidate,
                result_summary=result_summary,
            )

        def _mark_failed(
            *,
            task_key: str,
            role_id: str,
            model_id: str | None,
            stage: str,
            reason: str,
        ) -> None:
            self._mark_role_task_failed(
                role_id=role_id,
                model_id=model_id,
                stage=stage,
                reason=reason,
            )
            self._mark_task_failed(
                task_key=task_key,
                role_id=role_id,
                model_id=model_id,
                stage=stage,
                reason=reason,
            )

        def _mark_reused(
            *,
            task_key: str,
            role_id: str,
            stage: str,
            candidate: Mapping[str, Any] | None = None,
        ) -> None:
            restored_candidate = _candidate_from_payload(candidate)
            self._mark_role_task_reused(
                role_id=role_id,
                stage=stage,
                candidate=restored_candidate,
            )
            self._mark_task_reused(
                task_key=task_key,
                role_id=role_id,
                stage=stage,
                candidate=restored_candidate,
            )

        return ControlledExecutionResumeHooks(
            get_task_summary=self._resume_task_summary,
            mark_task_running=_mark_running,
            mark_task_completed=_mark_completed,
            mark_task_failed=_mark_failed,
            mark_task_reused=_mark_reused,
        )

    async def _generate_role_candidate(
        self,
        *,
        role_id: str,
        role_title: str,
        role_instruction: str,
        model_id: str,
        task_input: str,
        guidance_messages: list[str],
        supporting_candidates: list[CandidateOutput],
        prompt_override: str | None = None,
        stage: str | None = None,
    ) -> CandidateOutput:
        prompt = (
            prompt_override
            if isinstance(prompt_override, str) and prompt_override.strip()
            else _build_role_prompt(
                task_input=task_input,
                role_title=role_title,
                guidance_messages=guidance_messages,
                supporting_candidates=supporting_candidates,
            )
        )
        role_runtime_context = self._build_role_runtime_context(role_id)
        if role_runtime_context:
            prompt = f"{prompt}\n\nRole runtime context:\n{role_runtime_context}"
        role_stage = stage or f"role::{role_id}"
        content, diagnostics = await self._await_subagent_operation(
            lambda: self._invoke_configured_text(
                model_id=model_id,
                system_prompt=role_instruction,
                user_prompt=prompt,
                temperature=0.2,
                max_tokens=1024,
                reasoning_effort=self._default_reasoning_effort,
                execution_profile="subagent_readonly",
                tool_mode="auto",
                system_prompt_addendum=(
                    f"Role identity: {role_title} ({role_id}).\n"
                    f"Role instruction: {role_instruction}"
                ),
                session_scope=f"role::{role_id}",
            ),
            role_id=role_id,
            model_id=model_id,
            stage=role_stage,
        )
        candidate = CandidateOutput(
            candidate_id=role_id,
            role_id=role_id,
            content=content,
            metadata={"model_id": model_id, "diagnostics": diagnostics},
        )
        self._mark_role_task_completed(
            role_id=role_id,
            model_id=model_id,
            stage=role_stage,
            candidate=candidate,
        )
        self._mark_task_completed(
            task_key=role_stage,
            role_id=role_id,
            model_id=model_id,
            stage=role_stage,
            candidate=candidate,
        )
        return candidate

    def _make_shared_generate(
        self,
        *,
        execution_profile: str,
        tool_mode: str = "auto",
        session_scope: str,
        default_reasoning_effort: str | None = None,
    ) -> Any | None:
        if not callable(getattr(self._engine, "invoke", None)) and not callable(
            getattr(self._engine, "generate_with_configured_model", None)
        ):
            return None

        async def generate(
            *,
            model_id: str,
            messages: list[Message],
            temperature: float = 0.2,
            max_tokens: int = 1024,
            reasoning_effort: str | None = None,
            **_: Any,
        ) -> GenerationResult:
            effective_reasoning_effort = (
                reasoning_effort if reasoning_effort is not None else default_reasoning_effort
            )
            fallback_generate = getattr(self._engine, "generate_with_configured_model", None)
            system_prompt = "\n\n".join(
                message.content
                for message in messages
                if isinstance(message, Message) and message.role == "system" and message.content
            )
            user_prompt = "\n\n".join(
                message.content
                for message in messages
                if isinstance(message, Message) and message.role != "system" and message.content
            )
            content, _diagnostics = await self._invoke_configured_text(
                model_id=model_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_effort=effective_reasoning_effort,
                execution_profile=execution_profile,
                tool_mode=tool_mode,
                system_prompt_addendum=system_prompt,
                session_scope=session_scope,
                fallback_generate=fallback_generate if callable(fallback_generate) else None,
            )
            return GenerationResult(content=content, model=model_id)

        return generate

    @staticmethod
    def _metadata_reasoning_effort(metadata: Mapping[str, Any] | None) -> str | None:
        if not isinstance(metadata, Mapping):
            return None
        value = metadata.get("reasoning_effort")
        if not isinstance(value, str):
            return None
        normalized = value.strip().lower()
        return normalized or None

    async def _invoke_configured_text(
        self,
        *,
        model_id: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        execution_profile: str,
        tool_mode: str,
        system_prompt_addendum: str,
        session_scope: str,
        reasoning_effort: str | None = None,
        fallback_generate: Any | None = None,
    ) -> tuple[str, dict[str, Any]]:
        invoke = getattr(self._engine, "invoke", None)
        if callable(invoke):
            session_id = f"multi-agent::{session_scope}::{uuid4()}"
            result = await invoke(
                AgentInvocationRequest(
                    message=user_prompt,
                    session_id=session_id,
                    inference_overrides={
                        "model": model_id,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "reasoning_effort": reasoning_effort,
                    },
                    tool_mode=tool_mode,  # type: ignore[arg-type]
                    execution_profile=execution_profile,  # type: ignore[arg-type]
                    system_prompt_addendum=system_prompt_addendum,
                    persist_session=False,
                )
            )
            content = result.content.strip() if isinstance(result.content, str) else ""
            diagnostics = result.diagnostics.to_dict()
            self._invocation_traces.append(
                _build_invocation_trace(
                    session_scope=session_scope,
                    session_id=session_id,
                    model_id=model_id,
                    execution_profile=execution_profile,
                    tool_mode=tool_mode,
                    diagnostics=diagnostics,
                    events=result.events,
                    fallback=False,
                )
            )
            return content, diagnostics

        generate = fallback_generate or getattr(self._engine, "generate_with_configured_model", None)
        if not callable(generate):
            raise RuntimeError("Configured-model generation is unavailable.")
        result = await generate(
            model_id=model_id,
            messages=[
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_prompt),
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
        content = result.content.strip() if isinstance(result.content, str) else ""
        diagnostics = {
            "execution_profile": execution_profile,
            "tool_mode": "disabled",
            "exposed_tools": [],
            "matched_tool_groups": [],
            "fallback_reason": "invoke_unavailable_used_generate_with_configured_model",
        }
        self._invocation_traces.append(
            _build_invocation_trace(
                session_scope=session_scope,
                session_id=None,
                model_id=model_id,
                execution_profile=execution_profile,
                tool_mode="disabled",
                diagnostics=diagnostics,
                events=[],
                fallback=True,
            )
        )
        return content, diagnostics

    async def _run_model_backed_debate(
        self,
        *,
        task_input: str,
        protocol_config: MultiAgentDebateProtocol,
        guidance_messages: list[str],
        selected_models_roles: dict[str, str],
        metadata: dict[str, Any],
        ordered_models: list[str],
        emit: Any,
    ) -> tuple[list[CandidateOutput], dict[str, Any]]:
        roles = build_multi_agent_debate_roles(
            debater_a_role_id=protocol_config.debater_a_role_id,
            debater_b_role_id=protocol_config.debater_b_role_id,
            judge_role_id=protocol_config.judge_role_id,
        )
        debater_a = roles[0]
        debater_b = roles[1]
        debater_a_model_id = _resolve_protocol_role_model_id(
            role_id=debater_a.role_id,
            selected_models_roles=selected_models_roles,
            ordered_fallback=ordered_models,
            fallback_index=0,
        )
        debater_b_model_id = _resolve_protocol_role_model_id(
            role_id=debater_b.role_id,
            selected_models_roles=selected_models_roles,
            ordered_fallback=ordered_models,
            fallback_index=1,
            default_model_id=debater_a_model_id,
        )
        if not debater_a_model_id or not debater_b_model_id:
            return [], {}

        policy = DebateContextPolicy.from_metadata(metadata)
        debate_context = DebateContextManager(
            task_input=task_input,
            guidance_messages=guidance_messages,
            policy=policy,
        )

        latest_candidates: dict[str, CandidateOutput] = {}
        snapshots: list[dict[str, Any]] = []
        disabled_roles: set[str] = set()
        research_appendix = _research_prompt_appendix(metadata)
        for round_index in range(1, protocol_config.rounds + 1):
            for role, model_id in ((debater_a, debater_a_model_id), (debater_b, debater_b_model_id)):
                if role.role_id in disabled_roles:
                    continue
                task_key = f"debate_round_{round_index}:{role.role_id}"
                prompt, snapshot = debate_context.build_role_prompt(
                    role_id=role.role_id,
                    role_title=role.title,
                    stage=f"debate_round_{round_index}",
                    reserved_output_tokens=1024,
                )
                if research_appendix:
                    prompt = f"{prompt}\n\nResearch context:\n{research_appendix}"
                snapshots.append(snapshot.to_dict())
                candidate = self._resume_candidate_for_task(task_key)
                if candidate is not None:
                    self._mark_role_task_reused(
                        role_id=role.role_id,
                        stage=task_key,
                        candidate=candidate,
                    )
                    self._mark_task_reused(
                        task_key=task_key,
                        role_id=role.role_id,
                        stage=task_key,
                        candidate=candidate,
                    )
                    latest_candidates[role.role_id] = candidate
                    debate_context.register_turn(
                        role_id=role.role_id,
                        round_index=round_index,
                        content=candidate.content,
                        candidate_id=candidate.candidate_id,
                    )
                    continue
                try:
                    candidate = await self._generate_role_candidate(
                        role_id=role.role_id,
                        role_title=role.title,
                        role_instruction=role.instruction,
                        model_id=model_id,
                        task_input=task_input,
                        guidance_messages=guidance_messages,
                        supporting_candidates=list(latest_candidates.values()),
                        prompt_override=prompt,
                        stage=task_key,
                    )
                except SubagentRoleError as exc:
                    remaining_roles = [
                        candidate_role.role_id
                        for candidate_role, _candidate_model_id in ((debater_a, debater_a_model_id), (debater_b, debater_b_model_id))
                        if candidate_role.role_id not in disabled_roles
                        and candidate_role.role_id != role.role_id
                    ]
                    if (
                        self._on_subagent_disconnect() == "retry_then_degrade"
                        and classify_agent_run_recovery_issue(exc.cause) is None
                        and (latest_candidates or remaining_roles)
                    ):
                        self._mark_degraded_role(
                            role_id=exc.role_id,
                            model_id=exc.model_id,
                            stage=exc.stage,
                            reason=exc.reason,
                        )
                        disabled_roles.add(role.role_id)
                        continue
                    self._raise_subagent_disconnect_stop(
                        error=exc,
                        task_input=task_input,
                        protocol=protocol_config.protocol,
                        stage=exc.stage,
                        candidates=list(latest_candidates.values()),
                        extra={"round_index": round_index},
                    )
                latest_candidates[role.role_id] = candidate
                debate_context.register_turn(
                    role_id=role.role_id,
                    round_index=round_index,
                    content=candidate.content,
                    candidate_id=candidate.candidate_id,
                )
                emit(
                    "role_output",
                    asdict(
                        RoleOutputPayload(
                            role_id=candidate.role_id,
                            content=candidate.content,
                            round_index=round_index,
                            candidate_id=candidate.candidate_id,
                            model_id=model_id,
                        )
                    ),
                )
            if not latest_candidates:
                raise RuntimeError("All debate roles exhausted before producing any candidate output.")

        protocol_artifacts: dict[str, Any] = {
            "debate_state": debate_context.current_state().to_dict(),
            "debate_context_snapshot": {
                "protocol": protocol_config.protocol,
                "snapshots": snapshots,
                "latest": snapshots[-1] if snapshots else None,
            },
        }
        return list(latest_candidates.values()), protocol_artifacts

    async def _prepare_research_debate_context(
        self,
        *,
        task_input: str,
        protocol_config: MultiAgentDebateProtocol,
        guidance_messages: list[str],
        selected_models_roles: dict[str, str],
        metadata: dict[str, Any],
        policy: ResearchDebatePolicy,
    ) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
        generate = self._make_shared_generate(
            execution_profile="subagent_research",
            tool_mode="auto",
            session_scope="research",
            default_reasoning_effort=self._metadata_reasoning_effort(metadata),
        )
        planner_model_id = (
            selected_models_roles.get("planner")
            or selected_models_roles.get("research_planner")
            or selected_models_roles.get("smart_judge")
            or _resolve_judge_model_id(
                protocol_config=protocol_config,
                selected_models_roles=selected_models_roles,
            )
        )
        local_worker_model_id = (
            selected_models_roles.get("local_worker")
            or selected_models_roles.get("research_worker")
            or selected_models_roles.get("skeptic")
            or selected_models_roles.get(protocol_config.debater_a_role_id)
        )
        evidence_queries = _resolve_evidence_queries(metadata)
        research_runtime = metadata.get("research_runtime") if isinstance(metadata.get("research_runtime"), Mapping) else {}
        saved_planner_summary = self._resume_task_summary("research_planning") or {}
        research_plan = (
            saved_planner_summary.get("research_plan")
            if isinstance(saved_planner_summary.get("research_plan"), Mapping)
            else research_runtime.get("research_plan")
            if isinstance(research_runtime, Mapping) and isinstance(research_runtime.get("research_plan"), Mapping)
            else None
        )
        if isinstance(research_plan, Mapping):
            research_plan = _mapping_to_plain_dict(research_plan)
            self._mark_role_task_reused(
                role_id="research_planner",
                stage="research_planning",
            )
            self._mark_task_reused(
                task_key="research_planning",
                role_id="research_planner",
                stage="research_planning",
            )
        else:
            try:
                research_plan = await self._await_subagent_operation(
                    lambda: build_research_plan(
                        task_input=task_input,
                        guidance_messages=guidance_messages,
                        existing_queries=evidence_queries,
                        policy=policy,
                        planner_model_id=planner_model_id,
                        generate=generate if callable(generate) else None,
                    ),
                    role_id="research_planner",
                    model_id=planner_model_id,
                    stage="research_planning",
                )
                self._mark_role_task_completed(
                    role_id="research_planner",
                    model_id=planner_model_id,
                    stage="research_planning",
                    result_summary={
                        "subquestion_count": len(list(research_plan.get("subquestions") or [])),
                        "research_plan": research_plan,
                    },
                )
                self._mark_task_completed(
                    task_key="research_planning",
                    role_id="research_planner",
                    model_id=planner_model_id,
                    stage="research_planning",
                    result_summary={"research_plan": research_plan},
                )
            except SubagentRoleError as exc:
                if self._on_subagent_disconnect() == "fail":
                    raise
                self._mark_degraded_role(
                    role_id=exc.role_id,
                    model_id=exc.model_id,
                    stage=exc.stage,
                    reason=exc.reason,
                )
                research_plan = await build_research_plan(
                    task_input=task_input,
                    guidance_messages=guidance_messages,
                    existing_queries=evidence_queries,
                    policy=policy,
                    planner_model_id=None,
                    generate=None,
                )
        merged_queries = merge_research_queries(
            existing_queries=evidence_queries,
            research_plan=research_plan,
            policy=policy,
        )
        updated_metadata = _with_research_collection_policy(
            metadata,
            policy=policy,
            evidence_queries=merged_queries,
            debate_rounds=policy.debate_rounds,
        )
        evidence_packets, evidence_collection_summary = await self._collect_evidence_packets(
            metadata=updated_metadata,
        )
        source_quality_table = build_source_quality_table(evidence_packets)
        saved_worker_summary = self._resume_task_summary("research_workers") or {}
        worker_outputs = (
            saved_worker_summary.get("worker_outputs")
            if isinstance(saved_worker_summary.get("worker_outputs"), Mapping)
            else research_runtime.get("worker_outputs")
            if isinstance(research_runtime, Mapping) and isinstance(research_runtime.get("worker_outputs"), Mapping)
            else None
        )
        if isinstance(worker_outputs, Mapping):
            worker_outputs = _mapping_to_plain_dict(worker_outputs)
            self._mark_role_task_reused(
                role_id="research_worker",
                stage="research_workers",
            )
            self._mark_task_reused(
                task_key="research_workers",
                role_id="research_worker",
                stage="research_workers",
            )
        else:
            try:
                worker_outputs = await self._await_subagent_operation(
                    lambda: run_research_workers(
                        task_input=task_input,
                        research_plan=research_plan,
                        evidence_packets=evidence_packets,
                        policy=policy,
                        local_worker_model_id=local_worker_model_id,
                        generate=generate if callable(generate) else None,
                    ),
                    role_id="research_worker",
                    model_id=local_worker_model_id,
                    stage="research_workers",
                )
                self._mark_role_task_completed(
                    role_id="research_worker",
                    model_id=local_worker_model_id,
                    stage="research_workers",
                    result_summary={
                        "worker_count": len(list(worker_outputs.get("workers") or [])),
                        "worker_outputs": worker_outputs,
                    },
                )
                self._mark_task_completed(
                    task_key="research_workers",
                    role_id="research_worker",
                    model_id=local_worker_model_id,
                    stage="research_workers",
                    result_summary={"worker_outputs": worker_outputs},
                )
            except SubagentRoleError as exc:
                if self._on_subagent_disconnect() == "fail":
                    raise
                self._mark_degraded_role(
                    role_id=exc.role_id,
                    model_id=exc.model_id,
                    stage=exc.stage,
                    reason=exc.reason,
                )
                worker_outputs = await run_research_workers(
                    task_input=task_input,
                    research_plan=research_plan,
                    evidence_packets=evidence_packets,
                    policy=policy,
                    local_worker_model_id=None,
                    generate=None,
                )
        updated_metadata["research_runtime"] = {
            "research_plan": research_plan,
            "source_quality_table": source_quality_table,
            "worker_outputs": worker_outputs,
        }
        research_plan_artifact = {
            **research_plan,
            "output_targets": list(policy.output_targets),
            "source_mode": policy.source_mode,
            "citation_policy": policy.citation_policy,
            "local_worker_count": policy.local_worker_count,
            "worker_outputs": worker_outputs,
        }
        return (
            updated_metadata,
            {
                "research_plan": research_plan_artifact,
                "source_quality_table": source_quality_table,
            },
            evidence_packets,
            evidence_collection_summary,
        )

    async def _build_research_result_artifacts(
        self,
        *,
        task_input: str,
        selected_models_roles: dict[str, str],
        candidates: list[CandidateOutput],
        selected_candidate_id: str | None,
        protocol_artifacts: dict[str, Any],
        verification_summary: dict[str, Any] | None,
        metadata: dict[str, Any],
        policy: ResearchDebatePolicy,
    ) -> dict[str, Any]:
        debate_state = (
            protocol_artifacts.get("debate_state")
            if isinstance(protocol_artifacts.get("debate_state"), dict)
            else None
        )
        source_quality_table = (
            protocol_artifacts.get("source_quality_table")
            if isinstance(protocol_artifacts.get("source_quality_table"), dict)
            else None
        )
        research_runtime = metadata.get("research_runtime")
        if source_quality_table is None and isinstance(research_runtime, Mapping):
            runtime_sources = research_runtime.get("source_quality_table")
            if isinstance(runtime_sources, dict):
                source_quality_table = dict(runtime_sources)
        claim_evidence_map, updated_debate_state = build_claim_evidence_map(
            debate_state=debate_state,
            verification_summary=verification_summary,
            source_quality_table=source_quality_table,
            citation_policy=policy.citation_policy,
        )
        selected_candidate = _candidate_by_id(candidates, selected_candidate_id)
        synthesizer_model_id = (
            selected_models_roles.get("synthesizer")
            or selected_models_roles.get("research_synthesizer")
            or selected_models_roles.get("smart_judge")
            or selected_models_roles.get("judge")
            or _resolve_judge_model_id(
                protocol_config=MultiAgentDebateProtocol(),
                selected_models_roles=selected_models_roles,
            )
        )
        research_plan = (
            protocol_artifacts.get("research_plan")
            if isinstance(protocol_artifacts.get("research_plan"), dict)
            else None
        )
        worker_outputs = (
            research_runtime.get("worker_outputs")
            if isinstance(research_runtime, Mapping) and isinstance(research_runtime.get("worker_outputs"), dict)
            else None
        )
        generate = self._make_shared_generate(
            execution_profile="subagent_research",
            tool_mode="auto",
            session_scope="research-synthesizer",
            default_reasoning_effort=self._metadata_reasoning_effort(metadata),
        )
        artifacts: dict[str, Any] = {"claim_evidence_map": claim_evidence_map}
        if isinstance(updated_debate_state, dict):
            artifacts["debate_state"] = updated_debate_state
        if policy.wants_output("research_brief"):
            saved_synthesis_summary = self._resume_task_summary("research_synthesis") or {}
            saved_research_brief = (
                saved_synthesis_summary.get("research_brief")
                if isinstance(saved_synthesis_summary.get("research_brief"), Mapping)
                else None
            )
            if isinstance(saved_research_brief, Mapping):
                artifacts["research_brief"] = _mapping_to_plain_dict(saved_research_brief)
                self._mark_role_task_reused(
                    role_id="research_synthesizer",
                    stage="research_synthesis",
                )
                self._mark_task_reused(
                    task_key="research_synthesis",
                    role_id="research_synthesizer",
                    stage="research_synthesis",
                )
            else:
                try:
                    artifacts["research_brief"] = await self._await_subagent_operation(
                        lambda: synthesize_research_brief(
                            task_input=task_input,
                            selected_candidate=selected_candidate.to_dict() if selected_candidate is not None else None,
                            research_plan=research_plan,
                            source_quality_table=source_quality_table,
                            claim_evidence_map=claim_evidence_map,
                            worker_outputs=worker_outputs,
                            synthesizer_model_id=synthesizer_model_id,
                            generate=generate if callable(generate) else None,
                        ),
                        role_id="research_synthesizer",
                        model_id=synthesizer_model_id,
                        stage="research_synthesis",
                    )
                    self._mark_role_task_completed(
                        role_id="research_synthesizer",
                        model_id=synthesizer_model_id,
                        stage="research_synthesis",
                        result_summary={
                            "artifact_keys": ["research_brief"],
                            "research_brief": artifacts["research_brief"],
                        },
                    )
                    self._mark_task_completed(
                        task_key="research_synthesis",
                        role_id="research_synthesizer",
                        model_id=synthesizer_model_id,
                        stage="research_synthesis",
                        result_summary={"research_brief": artifacts["research_brief"]},
                    )
                except SubagentRoleError as exc:
                    if self._on_subagent_disconnect() == "fail":
                        raise
                    self._mark_degraded_role(
                        role_id=exc.role_id,
                        model_id=exc.model_id,
                        stage=exc.stage,
                        reason=exc.reason,
                    )
                    artifacts["research_brief"] = await synthesize_research_brief(
                        task_input=task_input,
                        selected_candidate=selected_candidate.to_dict() if selected_candidate is not None else None,
                        research_plan=research_plan,
                        source_quality_table=source_quality_table,
                        claim_evidence_map=claim_evidence_map,
                        worker_outputs=worker_outputs,
                        synthesizer_model_id=None,
                        generate=None,
                    )
        return artifacts

    def _run_placeholder_protocol(
        self,
        task_input: str,
        *,
        protocol_config: ProtocolConfig,
        emit: Any,
    ) -> tuple[list[CandidateOutput], dict[str, Any]]:
        if isinstance(protocol_config, TeacherStudentDistillProtocol):
            roles = build_teacher_student_roles(
                teacher_role_id=protocol_config.teacher_role_id,
                student_role_id=protocol_config.student_role_id,
            )
            teacher_role = roles[0]
            student_role = roles[1]
            teacher_output = CandidateOutput(
                candidate_id=teacher_role.role_id,
                role_id=teacher_role.role_id,
                content=f"Teacher draft: {task_input}",
            )
            student_output = CandidateOutput(
                candidate_id=student_role.role_id,
                role_id=student_role.role_id,
                content=f"Student distilled answer: {task_input}",
            )
            emit(
                "role_output",
                asdict(
                    RoleOutputPayload(
                        role_id=teacher_output.role_id,
                        content=teacher_output.content,
                        round_index=1,
                        candidate_id=teacher_output.candidate_id,
                    )
                ),
            )
            emit(
                "role_output",
                asdict(
                    RoleOutputPayload(
                        role_id=student_output.role_id,
                        content=student_output.content,
                        round_index=1,
                        candidate_id=student_output.candidate_id,
                    )
                ),
            )
            return [teacher_output, student_output], {}

        if isinstance(protocol_config, DrZeroSelfEvolveProtocol):
            roles = build_dr_zero_roles(
                proposer_role_id=protocol_config.proposer_role_id,
                solver_role_id=protocol_config.solver_role_id,
                verifier_role_id=protocol_config.verifier_role_id,
            )
            proposer_role = roles[0]
            solver_role = roles[1]
            synthetic_task = {
                "task_id": "task-1",
                "question": task_input,
                "difficulty": "placeholder",
                "rationale": "Placeholder task for Dr.Zero scaffold execution.",
                "metadata": {"placeholder": True},
            }
            proposer_output = CandidateOutput(
                candidate_id=proposer_role.role_id,
                role_id=proposer_role.role_id,
                content=json.dumps({"tasks": [synthetic_task]}, ensure_ascii=False),
            )
            solver_output = CandidateOutput(
                candidate_id=solver_role.role_id,
                role_id=solver_role.role_id,
                content=f"Solver answer for: {task_input}",
                metadata={"synthetic_task": synthetic_task, "synthetic_task_id": "task-1"},
            )
            emit(
                "role_output",
                asdict(
                    RoleOutputPayload(
                        role_id=proposer_output.role_id,
                        content=proposer_output.content,
                        round_index=1,
                        candidate_id=proposer_output.candidate_id,
                    )
                ),
            )
            emit(
                "role_output",
                asdict(
                    RoleOutputPayload(
                        role_id=solver_output.role_id,
                        content=solver_output.content,
                        round_index=1,
                        candidate_id=solver_output.candidate_id,
                    )
                ),
            )
            return [solver_output], {
                "synthetic_tasks": {
                    "protocol": protocol_config.protocol,
                    "root_task": task_input,
                    "tasks": [synthetic_task],
                    "parse_diagnostics": {"status": "placeholder", "parsed_task_count": 1},
                },
                "solver_rollouts": {
                    "protocol": protocol_config.protocol,
                    "rollout_count": 1,
                    "rollouts": [solver_output.to_dict()],
                },
                "reward_summary": {
                    "protocol": protocol_config.protocol,
                    "status": "pending_verification",
                    "basis": "placeholder_scaffold",
                    "synthetic_task_count": 1,
                    "solver_rollout_count": 1,
                },
                "curriculum_state": {
                    "protocol": protocol_config.protocol,
                    "iterations_configured": protocol_config.iterations,
                    "current_iteration": 1,
                    "next_iteration_ready": False,
                },
                "drzero_iteration_summary": {
                    "protocol": protocol_config.protocol,
                    "iteration_index": 1,
                    "synthetic_task_count": 1,
                    "solver_rollout_count": 1,
                    "proposal_parse_status": "placeholder",
                },
            }

        roles = build_multi_agent_debate_roles(
            debater_a_role_id=protocol_config.debater_a_role_id,
            debater_b_role_id=protocol_config.debater_b_role_id,
            judge_role_id=protocol_config.judge_role_id,
        )
        debater_a = roles[0]
        debater_b = roles[1]
        candidate_a = CandidateOutput(
            candidate_id=debater_a.role_id,
            role_id=debater_a.role_id,
            content=f"Debater A argument for: {task_input}",
        )
        candidate_b = CandidateOutput(
            candidate_id=debater_b.role_id,
            role_id=debater_b.role_id,
            content=f"Debater B rebuttal for: {task_input}",
        )
        emit(
            "role_output",
            asdict(
                RoleOutputPayload(
                    role_id=candidate_a.role_id,
                    content=candidate_a.content,
                    round_index=1,
                    candidate_id=candidate_a.candidate_id,
                )
            ),
        )
        emit(
            "role_output",
            asdict(
                RoleOutputPayload(
                    role_id=candidate_b.role_id,
                    content=candidate_b.content,
                    round_index=1,
                    candidate_id=candidate_b.candidate_id,
                )
            ),
        )
        return [candidate_a, candidate_b], {}

    async def _evaluate_candidates(
        self,
        *,
        task_input: str,
        protocol_config: ProtocolConfig,
        guidance_messages: list[str],
        selected_models_roles: dict[str, str],
        candidates: list[CandidateOutput],
        protocol_artifacts: dict[str, Any],
        metadata: dict[str, Any],
    ) -> tuple[list[CandidateScore], str | None]:
        judge_scores = await self._score_candidates_with_model(
            task_input=task_input,
            protocol_config=protocol_config,
            guidance_messages=guidance_messages,
            selected_models_roles=selected_models_roles,
            candidates=candidates,
            protocol_artifacts=protocol_artifacts,
            metadata=metadata,
        )
        if judge_scores is not None:
            return judge_scores
        return self._score_candidates(candidates), None

    async def _verify_candidates(
        self,
        *,
        task_input: str,
        protocol_config: ProtocolConfig,
        guidance_messages: list[str],
        selected_models_roles: dict[str, str],
        candidates: list[CandidateOutput],
        evidence_packets: list[dict[str, Any]],
        protocol_artifacts: dict[str, Any],
        metadata: dict[str, Any],
    ) -> tuple[list[CandidateVerification], dict[str, Any] | None]:
        verifier_model_id = _resolve_verifier_model_id(
            protocol_config=protocol_config,
            selected_models_roles=selected_models_roles,
        )
        context_policy = DebateContextPolicy.from_metadata(metadata)
        debate_state = (
            protocol_artifacts.get("debate_state")
            if isinstance(protocol_artifacts.get("debate_state"), dict)
            else None
        )

        if not candidates:
            return [], None

        generate = self._make_shared_generate(
            execution_profile="verifier",
            tool_mode="auto",
            session_scope="verifier",
            default_reasoning_effort=self._metadata_reasoning_effort(metadata),
        )
        if not verifier_model_id or not callable(generate):
            if not evidence_packets:
                return [], None
            verifications = [
                CandidateVerification(
                    candidate_id=item.candidate_id,
                    status="skipped",
                    rationale="No verifier model configured.",
                    metadata={"evidence_packet_count": len(evidence_packets)},
                )
                for item in candidates
            ]
            return verifications, _build_verification_summary(
                verifications,
                verifier_model_id=None,
                evidence_packets=evidence_packets,
            )

        if not evidence_packets:
            verifications = [
                CandidateVerification(
                    candidate_id=item.candidate_id,
                    status="skipped",
                    rationale="No evidence packets provided.",
                    metadata={"evidence_packet_count": 0},
                )
                for item in candidates
            ]
            return verifications, _build_verification_summary(
                verifications,
                verifier_model_id=verifier_model_id,
                evidence_packets=[],
            )

        candidate_payloads = [item.to_dict() for item in candidates]
        claim_cards = _claim_cards_from_state(debate_state)
        candidate_summaries = summarize_candidate_outputs(candidate_payloads, claim_cards=claim_cards)
        evidence_chunks = _chunk_evidence_packets(
            evidence_packets,
            max_input_tokens=context_policy.max_input_tokens,
            enabled=context_policy.enable_verifier_chunking and isinstance(protocol_config, MultiAgentDebateProtocol),
        )

        chunk_summaries: list[dict[str, Any]] = []
        chunk_level_results: list[dict[str, Any]] = []
        research_appendix = _research_prompt_appendix(metadata)
        for chunk_index, evidence_chunk in enumerate(evidence_chunks):
            task_key = f"verification_chunk_{chunk_index}"
            saved_chunk_summary = self._resume_task_summary(task_key)
            if isinstance(saved_chunk_summary, dict) and isinstance(saved_chunk_summary.get("verifications"), list):
                parsed = _candidate_verifications_from_payload(saved_chunk_summary.get("verifications"))
                if parsed:
                    chunk_payload = [item.to_dict() for item in parsed]
                    chunk_summaries.append(
                        {
                            "chunk_index": chunk_index,
                            "chunk_count": len(evidence_chunks),
                            "evidence_packet_count": len(evidence_chunk),
                            "verifications": chunk_payload,
                        }
                    )
                    chunk_level_results.extend(chunk_payload)
                    self._mark_task_reused(
                        task_key=task_key,
                        role_id="verifier",
                        stage=task_key,
                    )
                    continue
            if isinstance(protocol_config, MultiAgentDebateProtocol) and debate_state is not None:
                prompt = _build_structured_verifier_prompt(
                    task_input=task_input,
                    protocol_name=protocol_config.protocol,
                    guidance_messages=guidance_messages,
                    debate_state=debate_state,
                    evidence_packets=evidence_chunk,
                    candidate_summaries=candidate_summaries,
                    chunk_index=chunk_index,
                    chunk_count=len(evidence_chunks),
                )
            else:
                prompt = _build_verifier_prompt(
                    task_input=task_input,
                    protocol_name=protocol_config.protocol,
                    guidance_messages=guidance_messages,
                    evidence_packets=evidence_chunk,
                    candidates=candidates,
                )
            if research_appendix:
                prompt = f"{prompt}\n\nResearch context:\n{research_appendix}"
            try:
                result = await self._await_subagent_operation(
                    lambda: generate(
                        model_id=verifier_model_id,
                        messages=[
                            Message(
                                role="system",
                                content=(
                                    "You verify candidate answers against supplied evidence. "
                                    "Return JSON only with key `candidate_verifications`. "
                                    "Each item must include `candidate_id`, `status`, `rationale`, "
                                    "`citations`, and `issues`."
                                ),
                            ),
                            Message(role="user", content=prompt),
                        ],
                        temperature=0.1,
                        max_tokens=1200,
                        reasoning_effort=None,
                    ),
                    role_id="verifier",
                    model_id=verifier_model_id,
                    stage=task_key,
                )
            except SubagentRoleError as exc:
                if self._on_subagent_disconnect() == "fail":
                    raise
                self._mark_degraded_role(
                    role_id=exc.role_id,
                    model_id=exc.model_id,
                    stage=exc.stage,
                    reason=exc.reason,
                )
                fallback_verifications = [
                    CandidateVerification(
                        candidate_id=item.candidate_id,
                        status="skipped",
                        rationale=f"Verifier degraded: {exc.reason}",
                        metadata={"verifier_model_id": verifier_model_id, "chunk_index": chunk_index},
                    )
                    for item in candidates
                ]
                return fallback_verifications, _build_verification_summary(
                    fallback_verifications,
                    verifier_model_id=verifier_model_id,
                    evidence_packets=evidence_packets,
                )
            parsed = _parse_candidate_verifications(result.content, candidates)
            if not parsed:
                parsed = [
                    CandidateVerification(
                        candidate_id=item.candidate_id,
                        status="skipped",
                        rationale="Verifier did not return parseable candidate verifications.",
                        metadata={
                            "verifier_model_id": verifier_model_id,
                            "chunk_index": chunk_index,
                        },
                    )
                    for item in candidates
                ]
            chunk_payload = [item.to_dict() for item in parsed]
            chunk_summaries.append(
                {
                    "chunk_index": chunk_index,
                    "chunk_count": len(evidence_chunks),
                    "evidence_packet_count": len(evidence_chunk),
                    "verifications": chunk_payload,
                }
            )
            chunk_level_results.extend(chunk_payload)
            self._mark_task_completed(
                task_key=task_key,
                role_id="verifier",
                model_id=verifier_model_id,
                stage=task_key,
                result_summary={"verifications": chunk_payload},
            )

        reduced_payloads = (
            verification_reduce(chunk_level_results)
            if len(evidence_chunks) > 1
            else chunk_level_results
        )
        verifications = [
            CandidateVerification(
                candidate_id=str(item.get("candidate_id") or ""),
                status=str(item.get("status") or "skipped"),
                rationale=str(item.get("rationale") or ""),
                citations=[
                    dict(citation)
                    for citation in (item.get("citations") or [])
                    if isinstance(citation, dict)
                ],
                issues=[str(issue) for issue in (item.get("issues") or []) if str(issue).strip()],
                metadata=dict(item.get("metadata") or {}),
            )
            for item in reduced_payloads
            if str(item.get("candidate_id") or "").strip()
        ]
        if not verifications:
            verifications = [
                CandidateVerification(
                    candidate_id=item.candidate_id,
                    status="skipped",
                    rationale="Verifier produced no candidate-level result.",
                    metadata={"verifier_model_id": verifier_model_id},
                )
                for item in candidates
            ]

        summary = _build_verification_summary(
            verifications,
            verifier_model_id=verifier_model_id,
            evidence_packets=evidence_packets,
        )
        summary["chunk_summaries"] = chunk_summaries
        summary["used_chunking"] = len(evidence_chunks) > 1
        self._mark_role_task_completed(
            role_id="verifier",
            model_id=verifier_model_id,
            stage="verification_completed",
            result_summary={
                "verification_count": len(verifications),
                "evidence_packet_count": len(evidence_packets),
            },
        )
        return verifications, summary

    async def _collect_evidence_packets(
        self,
        *,
        metadata: Mapping[str, Any] | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        provided_packets = _resolve_evidence_packets(metadata)
        evidence_queries = _resolve_evidence_queries(metadata)
        collection_policy = _resolve_evidence_collection_policy(metadata)
        if not evidence_queries:
            return provided_packets, None
        if not bool(collection_policy["enabled"]):
            return provided_packets, {
                "status": "disabled",
                "mode": str(collection_policy["mode"]),
                "rag_provider": str(collection_policy["rag_provider"]),
                "query_count": len(evidence_queries),
                "provided_packet_count": len(provided_packets),
                "collected_packet_count": 0,
                "total_packet_count": len(provided_packets),
                "provider_counts": {},
                "queries": [
                    {
                        "query": item,
                        "packet_count": 0,
                        "error": "Evidence collection disabled by policy.",
                    }
                    for item in evidence_queries
                ],
            }

        collected_packets: list[dict[str, Any]] = []
        collection_summary: dict[str, Any] = {
            "query_count": len(evidence_queries),
            "collected_packet_count": 0,
            "provider_counts": {},
            "queries": [
                {
                    "query": item,
                    "packet_count": 0,
                    "error": "Evidence collector is not available.",
                }
                for item in evidence_queries
            ],
        }
        collect = getattr(self._engine, "collect_agent_run_evidence", None)
        if callable(collect):
            packets, summary = await collect(
                queries=evidence_queries,
                metadata=dict(metadata or {}),
            )
            collected_packets = _normalize_evidence_packets(packets)
            if isinstance(summary, Mapping):
                collection_summary = dict(summary)

        merged_packets = list(provided_packets)
        merged_packets.extend(collected_packets)
        collection_summary = _build_evidence_collection_summary(
            provided_packets=provided_packets,
            collected_packets=collected_packets,
            evidence_queries=evidence_queries,
            summary=collection_summary,
        )
        return merged_packets, collection_summary

    async def _score_candidates_with_model(
        self,
        *,
        task_input: str,
        protocol_config: ProtocolConfig,
        guidance_messages: list[str],
        selected_models_roles: dict[str, str],
        candidates: list[CandidateOutput],
        protocol_artifacts: dict[str, Any],
        metadata: dict[str, Any],
    ) -> tuple[list[CandidateScore], str | None] | None:
        generate = self._make_shared_generate(
            execution_profile="judge",
            tool_mode="auto",
            session_scope="judge",
            default_reasoning_effort=self._metadata_reasoning_effort(metadata),
        )
        if not candidates or not callable(generate):
            return None
        judge_model_id = _resolve_judge_model_id(
            protocol_config=protocol_config,
            selected_models_roles=selected_models_roles,
        )
        if not judge_model_id:
            return None
        saved_evaluation_summary = self._resume_task_summary("candidate_evaluation") or {}
        saved_scores = _candidate_scores_from_payload(saved_evaluation_summary.get("scores"))
        saved_selected_candidate_id = (
            str(saved_evaluation_summary.get("selected_candidate_id"))
            if isinstance(saved_evaluation_summary.get("selected_candidate_id"), str)
            else None
        )
        if saved_scores:
            self._mark_role_task_reused(
                role_id="judge",
                stage="candidate_evaluation",
            )
            self._mark_task_reused(
                task_key="candidate_evaluation",
                role_id="judge",
                stage="candidate_evaluation",
            )
            return saved_scores, saved_selected_candidate_id

        debate_state = (
            protocol_artifacts.get("debate_state")
            if isinstance(protocol_artifacts.get("debate_state"), dict)
            else None
        )
        prompt = (
            _build_structured_evaluator_prompt(
                task_input=task_input,
                protocol_name=protocol_config.protocol,
                guidance_messages=guidance_messages,
                debate_state=debate_state,
                candidates=candidates,
            )
            if isinstance(protocol_config, MultiAgentDebateProtocol) and debate_state is not None
            else _build_evaluator_prompt(
                task_input=task_input,
                protocol_name=protocol_config.protocol,
                guidance_messages=guidance_messages,
                candidates=candidates,
            )
        )
        research_appendix = _research_prompt_appendix(metadata)
        if research_appendix:
            prompt = f"{prompt}\n\nResearch context:\n{research_appendix}"
        try:
            result = await self._await_subagent_operation(
                lambda: generate(
                    model_id=judge_model_id,
                    messages=[
                        Message(
                            role="system",
                            content=(
                                "You are an evaluator. Return JSON only with keys "
                                "`selected_candidate_id` and `scores`. Each score item must include "
                                "`candidate_id`, `score`, `rationale`, and optional `evidence_gate`."
                            ),
                        ),
                        Message(
                            role="user",
                            content=prompt,
                        ),
                    ],
                    temperature=0.1,
                    max_tokens=900,
                    reasoning_effort=None,
                ),
                role_id="judge",
                model_id=judge_model_id,
                stage="candidate_evaluation",
            )
        except SubagentRoleError as exc:
            if self._on_subagent_disconnect() == "fail":
                raise
            self._mark_degraded_role(
                role_id=exc.role_id,
                model_id=exc.model_id,
                stage=exc.stage,
                reason=exc.reason,
            )
            return None
        parsed_scores = _parse_judge_scores(result.content, candidates)
        if parsed_scores is not None:
            scores, selected_candidate_id = parsed_scores
            self._mark_role_task_completed(
                role_id="judge",
                model_id=judge_model_id,
                stage="candidate_evaluation",
                result_summary={
                    "score_count": len(scores),
                    "selected_candidate_id": selected_candidate_id,
                    "scores": [score.to_dict() for score in scores],
                },
            )
            self._mark_task_completed(
                task_key="candidate_evaluation",
                role_id="judge",
                model_id=judge_model_id,
                stage="candidate_evaluation",
                result_summary={
                    "selected_candidate_id": selected_candidate_id,
                    "scores": [score.to_dict() for score in scores],
                },
            )
        return parsed_scores

    def _score_candidates(self, candidates: list[CandidateOutput]) -> list[CandidateScore]:
        scores: list[CandidateScore] = []
        for item in candidates:
            score = _deterministic_score(item)
            scores.append(
                CandidateScore(
                    candidate_id=item.candidate_id,
                    score=score,
                    rationale="deterministic scaffold score",
                    evidence_gate=resolve_evidence_gate_status(skipped=True),
                )
            )
        return scores


def _build_invocation_trace(
    *,
    session_scope: str,
    session_id: str | None,
    model_id: str,
    execution_profile: str,
    tool_mode: str,
    diagnostics: dict[str, Any],
    events: list[Any],
    fallback: bool,
) -> dict[str, Any]:
    tool_events: list[dict[str, Any]] = []
    approval_pending: list[dict[str, Any]] = []
    for event in events:
        event_type = getattr(event, "type", None)
        if event_type == "tool_call_request":
            tool_events.append(
                {
                    "type": "tool_call_request",
                    "call_id": getattr(event, "call_id", ""),
                    "tool_name": getattr(event, "tool_name", ""),
                    "arguments": dict(getattr(event, "arguments", {}) or {}),
                }
            )
            continue
        if event_type != "tool_call_result":
            continue
        metadata = getattr(event, "metadata", {}) or {}
        if not isinstance(metadata, dict):
            metadata = {}
        payload = {
            "type": "tool_call_result",
            "call_id": getattr(event, "call_id", ""),
            "tool_name": getattr(event, "tool_name", ""),
            "error": getattr(event, "error", None),
            "metadata": dict(metadata),
        }
        tool_events.append(payload)
        if (
            metadata.get("status") == "approval_pending"
            or metadata.get("requires_approval") is True
            or isinstance(metadata.get("approval_id"), str)
        ):
            approval_pending.append(payload)

    return {
        "session_scope": session_scope,
        "session_id": session_id,
        "model_id": model_id,
        "execution_profile": execution_profile,
        "tool_mode": tool_mode,
        "fallback": fallback,
        "diagnostics": dict(diagnostics),
        "tool_events": tool_events,
        "approval_pending": approval_pending,
    }


def _build_subagent_runtime_artifact(invocations: list[dict[str, Any]]) -> dict[str, Any]:
    approval_pending = [
        pending
        for invocation in invocations
        for pending in invocation.get("approval_pending", [])
        if isinstance(pending, dict)
    ]
    tool_events = [
        event
        for invocation in invocations
        for event in invocation.get("tool_events", [])
        if isinstance(event, dict)
    ]
    risky_tool_names = {
        "exec_command",
        "shell",
        "execute_code",
        "execute_code_v2",
        "file_write",
        "file_edit",
        "write_stdin",
        "kill_session",
        "process_stop",
        "mcp_call",
    }
    risky_tool_events = [
        event
        for event in tool_events
        if str(event.get("tool_name") or "") in risky_tool_names
    ]
    return {
        "execution_boundary": "multi_agent_subagents_research_only_main_agent_executes_code_and_training",
        "invocation_count": len(invocations),
        "tool_event_count": len(tool_events),
        "approval_pending_count": len(approval_pending),
        "risky_tool_event_count": len(risky_tool_events),
        "approval_pending": approval_pending,
        "risky_tool_events": risky_tool_events,
        "invocations": invocations,
    }


def _deterministic_score(candidate: CandidateOutput) -> float:
    base = 0.7
    if "student" in candidate.role_id or candidate.role_id.endswith("_a"):
        base = 0.9
    if "solver" in candidate.role_id:
        base = 0.9
    length_component = min(len(candidate.content) / 1000.0, 0.09)
    return round(base + length_component, 4)


def _select_candidate_id(
    *,
    scores: list[CandidateScore],
    candidates: list[CandidateOutput],
    preferred_candidate_id: str | None = None,
) -> str | None:
    if not scores:
        return candidates[0].candidate_id if candidates else None
    best = max(
        scores,
        key=lambda item: (
            evidence_gate_rank(item.evidence_gate.status),
            item.score,
            1 if preferred_candidate_id is not None and item.candidate_id == preferred_candidate_id else 0,
        ),
    )
    return best.candidate_id


def _candidate_by_id(
    candidates: list[CandidateOutput],
    candidate_id: str | None,
) -> CandidateOutput | None:
    if candidate_id is None:
        return None
    for item in candidates:
        if item.candidate_id == candidate_id:
            return item
    return None


def _resolve_selected_model_roles(metadata: Mapping[str, Any] | None) -> dict[str, str]:
    if not isinstance(metadata, Mapping):
        return {}
    payload = metadata.get("selected_models_roles")
    if not isinstance(payload, Mapping):
        return {}

    resolved: dict[str, str] = {}
    by_role = payload.get("by_role")
    if isinstance(by_role, Mapping):
        for role_id, model_id in by_role.items():
            if isinstance(role_id, str) and isinstance(model_id, str) and role_id.strip() and model_id.strip():
                resolved[role_id.strip()] = model_id.strip()

    subagents = payload.get("subagents") or payload.get("entries")
    if isinstance(subagents, list):
        for item in subagents:
            if not isinstance(item, Mapping):
                continue
            role_id = item.get("role")
            model_id = item.get("model_id") or item.get("model")
            if isinstance(role_id, str) and isinstance(model_id, str) and role_id.strip() and model_id.strip():
                resolved.setdefault(role_id.strip(), model_id.strip())

    for role_id, value in payload.items():
        if role_id in {"by_role", "subagents", "entries"} or not isinstance(role_id, str):
            continue
        if isinstance(value, str) and role_id.strip() and value.strip():
            resolved.setdefault(role_id.strip(), value.strip())
            continue
        if isinstance(value, Mapping):
            model_id = value.get("model_id") or value.get("model")
            nested_role = value.get("role")
            if isinstance(model_id, str) and model_id.strip():
                if isinstance(nested_role, str) and nested_role.strip():
                    resolved.setdefault(nested_role.strip(), model_id.strip())
                resolved.setdefault(role_id.strip(), model_id.strip())

    return resolved


def _resolve_execution_policy_payload(
    protocol: Mapping[str, Any] | ProtocolConfig | None,
    metadata: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if isinstance(metadata, Mapping):
        payload = metadata.get("execution_policy")
        if isinstance(payload, Mapping):
            return payload
        summary = metadata.get("summary")
        if isinstance(summary, Mapping):
            nested = summary.get("execution_policy")
            if isinstance(nested, Mapping):
                return nested
    if isinstance(protocol, Mapping):
        payload = protocol.get("execution_policy")
        if isinstance(payload, Mapping):
            return payload
        if str(protocol.get("protocol") or "").strip() == LEGACY_CONTROLLED_SUBAGENT_PROTOCOL:
            return {
                "mode": "controlled",
                **{
                    str(key): value
                    for key, value in protocol.items()
                    if isinstance(key, str) and key not in {"protocol", "execution_policy"}
                },
            }
    return None


def _resolve_protocol_role_model_id(
    *,
    role_id: str,
    selected_models_roles: dict[str, str],
    ordered_fallback: list[str],
    fallback_index: int,
    default_model_id: str | None = None,
) -> str | None:
    if role_id in selected_models_roles:
        return selected_models_roles[role_id]
    if fallback_index < len(ordered_fallback):
        return ordered_fallback[fallback_index]
    return default_model_id


def _resolve_judge_model_id(
    *,
    protocol_config: ProtocolConfig,
    selected_models_roles: dict[str, str],
) -> str | None:
    ordered_models = list(selected_models_roles.values())
    if isinstance(protocol_config, TeacherStudentDistillProtocol):
        return (
            selected_models_roles.get("judge")
            or selected_models_roles.get("evaluator")
            or selected_models_roles.get(protocol_config.teacher_role_id)
            or (ordered_models[0] if ordered_models else None)
        )
    if isinstance(protocol_config, DrZeroSelfEvolveProtocol):
        return (
            selected_models_roles.get("judge")
            or selected_models_roles.get("evaluator")
            or selected_models_roles.get(protocol_config.verifier_role_id)
            or selected_models_roles.get(protocol_config.proposer_role_id)
            or (ordered_models[2] if len(ordered_models) > 2 else None)
            or (ordered_models[0] if ordered_models else None)
        )
    if isinstance(protocol_config, ControlledSubagentExecutionProtocol):
        return (
            selected_models_roles.get("judge")
            or selected_models_roles.get("evaluator")
            or selected_models_roles.get("controller")
            or (ordered_models[3] if len(ordered_models) > 3 else None)
            or (ordered_models[0] if ordered_models else None)
        )
    return (
        selected_models_roles.get(protocol_config.judge_role_id)
        or selected_models_roles.get("judge")
        or (ordered_models[2] if len(ordered_models) > 2 else None)
        or (ordered_models[0] if ordered_models else None)
    )


def _resolve_verifier_model_id(
    *,
    protocol_config: ProtocolConfig,
    selected_models_roles: dict[str, str],
) -> str | None:
    if "verifier" in selected_models_roles:
        return selected_models_roles["verifier"]
    if "fact_checker" in selected_models_roles:
        return selected_models_roles["fact_checker"]
    return _resolve_judge_model_id(
        protocol_config=protocol_config,
        selected_models_roles=selected_models_roles,
    )


def _build_dr_zero_proposer_prompt(
    *,
    task_input: str,
    sample_size: int,
    guidance_messages: list[str],
) -> str:
    lines = [
        "Create synthetic search-agent training tasks.",
        f"Root objective:\n{task_input.strip()}",
        "",
        f"Return JSON only with key `tasks`, containing exactly {sample_size} items when possible.",
        "Each task item should include `question`, optional `difficulty`, optional `rationale`, and optional `metadata`.",
        "Tasks must be hard enough to improve a solver but still answerable with available evidence/search tools.",
    ]
    if guidance_messages:
        lines.extend(["", "Guidance:", *[f"- {item}" for item in guidance_messages]])
    return "\n".join(lines)


def _build_dr_zero_solver_prompt(
    *,
    root_task: str,
    synthetic_task: Mapping[str, Any],
    guidance_messages: list[str],
    rollout_index: int,
) -> str:
    question = str(synthetic_task.get("question") or synthetic_task.get("task") or "").strip()
    lines = [
        "Solve this Dr.Zero synthetic task as a search-capable solver.",
        f"Root objective:\n{root_task.strip()}",
        "",
        f"Synthetic task id: {synthetic_task.get('task_id')}",
        f"Synthetic question:\n{question}",
        f"Rollout index: {rollout_index}",
        "",
        "Use available read/research tools when helpful. Return a concise answer with evidence-aware rationale.",
    ]
    if guidance_messages:
        lines.extend(["", "Guidance:", *[f"- {item}" for item in guidance_messages]])
    return "\n".join(lines)


def _parse_dr_zero_synthetic_tasks(
    content: str,
    *,
    task_input: str,
    sample_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parsed = parse_json_payload(content)
    raw_tasks: Any = None
    status = "parsed"
    reason: str | None = None
    if isinstance(parsed, Mapping):
        raw_tasks = parsed.get("tasks") or parsed.get("questions") or parsed.get("items")
    elif isinstance(parsed, list):
        raw_tasks = parsed

    tasks: list[dict[str, Any]] = []
    if isinstance(raw_tasks, list):
        for index, item in enumerate(raw_tasks[:sample_size], start=1):
            normalized = _normalize_dr_zero_task(item, fallback_question=task_input, index=index)
            if normalized is not None:
                tasks.append(normalized)

    if not tasks:
        status = "fallback"
        reason = "Proposer output did not contain parseable task JSON."
        tasks = [
            {
                "task_id": "task-1",
                "question": task_input.strip(),
                "difficulty": "unknown",
                "rationale": "Fallback task derived from the root objective.",
                "metadata": {"fallback": True},
            }
        ]

    diagnostics = {
        "status": status,
        "requested_sample_size": sample_size,
        "parsed_task_count": len(tasks),
        "fallback_reason": reason,
    }
    return tasks, diagnostics


def _normalize_dr_zero_task(
    item: Any,
    *,
    fallback_question: str,
    index: int,
) -> dict[str, Any] | None:
    if isinstance(item, str):
        question = item.strip()
        if not question:
            return None
        return {
            "task_id": f"task-{index}",
            "question": question,
            "difficulty": "unspecified",
            "rationale": "",
            "metadata": {},
        }
    if not isinstance(item, Mapping):
        return None
    question = (
        item.get("question")
        or item.get("task")
        or item.get("prompt")
        or item.get("query")
        or fallback_question
    )
    question_text = str(question or "").strip()
    if not question_text:
        return None
    task_id = str(item.get("task_id") or item.get("id") or f"task-{index}").strip()
    metadata = item.get("metadata")
    return {
        "task_id": task_id or f"task-{index}",
        "question": question_text,
        "difficulty": str(item.get("difficulty") or "unspecified"),
        "rationale": str(item.get("rationale") or item.get("reason") or ""),
        "metadata": dict(metadata) if isinstance(metadata, Mapping) else {},
    }


def _build_role_prompt(
    *,
    task_input: str,
    role_title: str,
    guidance_messages: list[str],
    supporting_candidates: list[CandidateOutput],
) -> str:
    lines = [
        f"Task:\n{task_input.strip()}",
        "",
        f"Role:\n{role_title}",
    ]
    if guidance_messages:
        lines.extend(["", "Guidance:"])
        lines.extend(f"- {item}" for item in guidance_messages if item.strip())
    if supporting_candidates:
        lines.extend(["", "Prior candidate outputs:"])
        for item in supporting_candidates:
            lines.append(f"[{item.candidate_id}] {item.content}")
    lines.extend(["", "Return only your answer content."])
    return "\n".join(lines).strip()


def _build_evaluator_prompt(
    *,
    task_input: str,
    protocol_name: str,
    guidance_messages: list[str],
    candidates: list[CandidateOutput],
) -> str:
    lines = [
        f"Task:\n{task_input.strip()}",
        "",
        f"Protocol:\n{protocol_name}",
    ]
    if guidance_messages:
        lines.extend(["", "Guidance:"])
        lines.extend(f"- {item}" for item in guidance_messages if item.strip())
    lines.extend(["", "Candidates:"])
    for candidate in candidates:
        lines.append(f"- candidate_id={candidate.candidate_id} role_id={candidate.role_id}")
        lines.append(candidate.content)
    lines.extend(
        [
            "",
            "Score each candidate from 0 to 1 for correctness, clarity, and task fit.",
            "Return JSON only.",
        ]
    )
    return "\n".join(lines).strip()


def _build_verifier_prompt(
    *,
    task_input: str,
    protocol_name: str,
    guidance_messages: list[str],
    evidence_packets: list[dict[str, Any]],
    candidates: list[CandidateOutput],
) -> str:
    lines = [
        f"Task:\n{task_input.strip()}",
        "",
        f"Protocol:\n{protocol_name}",
    ]
    if guidance_messages:
        lines.extend(["", "Guidance:"])
        lines.extend(f"- {item}" for item in guidance_messages if item.strip())
    lines.extend(["", "Evidence packets:"])
    for packet in evidence_packets:
        lines.append(
            f"- evidence_id={packet['evidence_id']} title={packet.get('title') or 'Untitled'}"
        )
        lines.append(str(packet.get("content") or ""))
    lines.extend(["", "Candidates:"])
    for candidate in candidates:
        lines.append(f"- candidate_id={candidate.candidate_id} role_id={candidate.role_id}")
        lines.append(candidate.content)
    lines.extend(
        [
            "",
            "Verification statuses: verified, skipped, failed.",
            "Return JSON only.",
        ]
    )
    return "\n".join(lines).strip()


def _build_structured_evaluator_prompt(
    *,
    task_input: str,
    protocol_name: str,
    guidance_messages: list[str],
    debate_state: dict[str, Any],
    candidates: list[CandidateOutput],
) -> str:
    claim_cards = _claim_cards_from_state(debate_state)
    candidate_summaries = summarize_candidate_outputs(
        [candidate.to_dict() for candidate in candidates],
        claim_cards=claim_cards,
    )
    lines = [
        f"Task:\n{task_input.strip()}",
        "",
        f"Protocol:\n{protocol_name}",
    ]
    if guidance_messages:
        lines.extend(["", "Guidance:"])
        lines.extend(f"- {item}" for item in guidance_messages if item.strip())
    lines.extend(["", "Debate state:", _render_debate_state(debate_state)])
    lines.extend(["", "Candidate summaries:"])
    for candidate in candidate_summaries:
        lines.append(
            f"- candidate_id={candidate['candidate_id']} role_id={candidate['role_id']} summary={candidate['summary']}"
        )
    lines.extend(
        [
            "",
            "Use the structured debate state, claim cards, and candidate summaries to score each candidate.",
            "Return JSON only.",
        ]
    )
    return "\n".join(lines).strip()


def _build_structured_verifier_prompt(
    *,
    task_input: str,
    protocol_name: str,
    guidance_messages: list[str],
    debate_state: dict[str, Any],
    evidence_packets: list[dict[str, Any]],
    candidate_summaries: list[dict[str, Any]],
    chunk_index: int,
    chunk_count: int,
) -> str:
    lines = [
        f"Task:\n{task_input.strip()}",
        "",
        f"Protocol:\n{protocol_name}",
        "",
        f"Evidence chunk:\n{chunk_index + 1}/{chunk_count}",
    ]
    if guidance_messages:
        lines.extend(["", "Guidance:"])
        lines.extend(f"- {item}" for item in guidance_messages if item.strip())
    lines.extend(["", "Debate state:", _render_debate_state(debate_state)])
    lines.extend(["", "Candidate summaries:"])
    for candidate in candidate_summaries:
        lines.append(
            f"- candidate_id={candidate['candidate_id']} role_id={candidate['role_id']} summary={candidate['summary']}"
        )
    lines.extend(["", "Evidence packets:"])
    for packet in evidence_packets:
        lines.append(
            f"- evidence_id={packet.get('evidence_id')} title={packet.get('title') or 'Untitled'}"
        )
        lines.append(str(packet.get("content") or ""))
    lines.extend(
        [
            "",
            "Verify each candidate using the claim cards, candidate summaries, and this evidence chunk only.",
            "Return JSON only.",
        ]
    )
    return "\n".join(lines).strip()


def _parse_judge_scores(
    raw_content: str,
    candidates: list[CandidateOutput],
) -> tuple[list[CandidateScore], str | None] | None:
    payload = _extract_json_payload(raw_content)
    if not isinstance(payload, dict):
        return None

    candidate_ids = {item.candidate_id for item in candidates}
    selected_candidate_id = payload.get("selected_candidate_id")
    if not isinstance(selected_candidate_id, str) or selected_candidate_id not in candidate_ids:
        selected_candidate_id = None

    parsed_scores: list[CandidateScore] = []
    raw_scores = payload.get("scores")
    if isinstance(raw_scores, list):
        for entry in raw_scores:
            if not isinstance(entry, dict):
                continue
            candidate_id = entry.get("candidate_id")
            if not isinstance(candidate_id, str) or candidate_id not in candidate_ids:
                continue
            score_value = _safe_float(entry.get("score"))
            if score_value is None:
                continue
            rationale = str(entry.get("rationale") or "model-judged score")
            parsed_scores.append(
                CandidateScore(
                    candidate_id=candidate_id,
                    score=score_value,
                    rationale=rationale,
                    evidence_gate=_parse_evidence_gate(entry.get("evidence_gate")),
                )
            )

    if not parsed_scores and selected_candidate_id is not None:
        for item in candidates:
            parsed_scores.append(
                CandidateScore(
                    candidate_id=item.candidate_id,
                    score=1.0 if item.candidate_id == selected_candidate_id else 0.0,
                    rationale="model-selected candidate",
                    evidence_gate=resolve_evidence_gate_status(skipped=True),
                )
            )

    if not parsed_scores:
        return None
    return parsed_scores, selected_candidate_id


def _parse_candidate_verifications(
    raw_content: str,
    candidates: list[CandidateOutput],
) -> list[CandidateVerification]:
    payload = _extract_json_payload(raw_content)
    if not isinstance(payload, dict):
        return []

    candidate_ids = {item.candidate_id for item in candidates}
    entries = payload.get("candidate_verifications")
    if not isinstance(entries, list):
        entries = payload.get("verifications")
    if not isinstance(entries, list):
        return []

    verifications: list[CandidateVerification] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        candidate_id = entry.get("candidate_id")
        status = entry.get("status")
        if (
            not isinstance(candidate_id, str)
            or candidate_id not in candidate_ids
            or status not in {"verified", "skipped", "failed"}
        ):
            continue
        rationale = str(entry.get("rationale") or "")
        citations = entry.get("citations")
        issues = entry.get("issues")
        verifications.append(
            CandidateVerification(
                candidate_id=candidate_id,
                status=status,
                rationale=rationale,
                citations=(
                    [dict(item) for item in citations if isinstance(item, dict)]
                    if isinstance(citations, list)
                    else []
                ),
                issues=(
                    [str(item) for item in issues]
                    if isinstance(issues, list)
                    else []
                ),
                metadata={
                    key: value
                    for key, value in entry.items()
                    if key not in {"candidate_id", "status", "rationale", "citations", "issues"}
                },
            )
        )
    return verifications


def _merge_scores_with_verifications(
    scores: list[CandidateScore],
    verifications: list[CandidateVerification],
) -> list[CandidateScore]:
    if not verifications:
        return scores
    verification_by_candidate = {
        item.candidate_id: item
        for item in verifications
    }
    merged: list[CandidateScore] = []
    for score in scores:
        verification = verification_by_candidate.get(score.candidate_id)
        if verification is None:
            merged.append(score)
            continue
        merged.append(
            CandidateScore(
                candidate_id=score.candidate_id,
                score=score.score,
                rationale=score.rationale,
                evidence_gate=verification.to_evidence_gate(),
            )
        )
    return merged


def _resolve_evidence_packets(metadata: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(metadata, Mapping):
        return []
    candidates = []
    evaluation_policy = metadata.get("evaluation_policy")
    summary = metadata.get("summary")
    if isinstance(evaluation_policy, Mapping):
        candidates.append(evaluation_policy.get("evidence_packets"))
        candidates.append(evaluation_policy.get("reference_materials"))
    if isinstance(summary, Mapping):
        candidates.append(summary.get("evidence_packets"))
        candidates.append(summary.get("reference_materials"))
        candidates.append(summary.get("sources"))
    for value in candidates:
        packets = _normalize_evidence_packets(value)
        if packets:
            return packets
    return []


def _resolve_evidence_queries(metadata: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(metadata, Mapping):
        return []
    candidates = []
    evaluation_policy = metadata.get("evaluation_policy")
    summary = metadata.get("summary")
    if isinstance(evaluation_policy, Mapping):
        candidates.append(evaluation_policy.get("evidence_queries"))
    if isinstance(summary, Mapping):
        candidates.append(summary.get("evidence_queries"))

    for value in candidates:
        queries = _normalize_evidence_queries(value)
        if queries:
            return queries
    return []


def _resolve_evidence_collection_policy(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    policy: dict[str, Any] = {
        "enabled": True,
        "mode": "hybrid",
        "rag_provider": "memory",
        "max_results_per_query": 3,
        "max_fetch_per_query": 2,
        "max_content_chars": 2000,
    }
    if not isinstance(metadata, Mapping):
        return policy

    research_policy = ResearchDebatePolicy.from_metadata(metadata)
    if research_policy.enabled:
        policy["mode"] = research_policy.evidence_collection_mode
        policy["max_results_per_query"] = research_policy.max_sources_per_query

    candidates = []
    evaluation_policy = metadata.get("evaluation_policy")
    summary = metadata.get("summary")
    if isinstance(evaluation_policy, Mapping):
        candidates.append(evaluation_policy.get("evidence_collection"))
    if isinstance(summary, Mapping):
        candidates.append(summary.get("evidence_collection"))

    for value in candidates:
        if not isinstance(value, Mapping):
            continue
        enabled = value.get("enabled")
        if isinstance(enabled, bool):
            policy["enabled"] = enabled
        mode = value.get("mode")
        if isinstance(mode, str) and mode.strip():
            policy["mode"] = mode.strip()
        rag_provider = value.get("rag_provider")
        if isinstance(rag_provider, str) and rag_provider.strip():
            policy["rag_provider"] = rag_provider.strip()
        for key in ("max_results_per_query", "max_fetch_per_query", "max_content_chars"):
            raw = value.get(key)
            if isinstance(raw, int) and raw > 0:
                policy[key] = raw
        break
    return policy


def _with_research_collection_policy(
    metadata: Mapping[str, Any],
    *,
    policy: ResearchDebatePolicy,
    evidence_queries: list[str],
    debate_rounds: int | None,
) -> dict[str, Any]:
    updated = dict(metadata)
    summary = dict(updated.get("summary") or {})
    summary["evidence_queries"] = list(evidence_queries)
    protocol_config = dict(summary.get("protocol_config") or {})
    if isinstance(debate_rounds, int) and debate_rounds > 0:
        protocol_config["rounds"] = debate_rounds
    if protocol_config:
        summary["protocol_config"] = protocol_config
    updated["summary"] = summary

    evaluation_policy = dict(updated.get("evaluation_policy") or {})
    collection = dict(evaluation_policy.get("evidence_collection") or {})
    collection.setdefault("enabled", True)
    collection["mode"] = policy.evidence_collection_mode
    collection.setdefault("rag_provider", "memory")
    collection.setdefault("max_results_per_query", policy.max_sources_per_query)
    collection.setdefault("max_fetch_per_query", min(policy.max_sources_per_query, 4))
    evaluation_policy["evidence_collection"] = collection
    updated["evaluation_policy"] = evaluation_policy
    return updated


def _research_prompt_appendix(metadata: Mapping[str, Any] | None) -> str:
    if not isinstance(metadata, Mapping):
        return ""
    runtime = metadata.get("research_runtime")
    if not isinstance(runtime, Mapping):
        return ""
    return build_research_prompt_appendix(
        research_plan=runtime.get("research_plan") if isinstance(runtime.get("research_plan"), Mapping) else None,
        source_quality_table=(
            runtime.get("source_quality_table")
            if isinstance(runtime.get("source_quality_table"), Mapping)
            else None
        ),
        worker_outputs=(
            runtime.get("worker_outputs") if isinstance(runtime.get("worker_outputs"), Mapping) else None
        ),
    )


def _normalize_evidence_packets(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    packets: list[dict[str, Any]] = []
    for index, item in enumerate(value, start=1):
        if isinstance(item, str) and item.strip():
            packets.append(
                {
                    "evidence_id": f"evidence-{index}",
                    "title": None,
                    "content": item.strip(),
                    "url": None,
                    "source_type": "inline_text",
                }
            )
            continue
        if not isinstance(item, Mapping):
            continue
        content = item.get("content") or item.get("text") or item.get("summary")
        if not isinstance(content, str) or not content.strip():
            continue
        evidence_id = item.get("evidence_id") or item.get("id") or f"evidence-{index}"
        title = item.get("title") or item.get("label")
        packets.append(
            {
                "evidence_id": str(evidence_id),
                "title": str(title) if isinstance(title, str) else None,
                "content": content.strip(),
                "url": str(item.get("url")) if isinstance(item.get("url"), str) else None,
                "source_type": (
                    str(item.get("source_type"))
                    if isinstance(item.get("source_type"), str)
                    else "inline_text"
                ),
                "query": str(item.get("query")) if isinstance(item.get("query"), str) else None,
                "provider": (
                    str(item.get("provider"))
                    if isinstance(item.get("provider"), str)
                    else None
                ),
            }
        )
    return packets


def _normalize_evidence_queries(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    queries: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            queries.append(item.strip())
            continue
        if not isinstance(item, Mapping):
            continue
        query = item.get("query") or item.get("q")
        if isinstance(query, str) and query.strip():
            queries.append(query.strip())
    return queries


def _build_evidence_collection_summary(
    *,
    provided_packets: list[dict[str, Any]],
    collected_packets: list[dict[str, Any]],
    evidence_queries: list[str],
    summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    payload = dict(summary or {})
    payload["mode"] = str(payload.get("mode") or "hybrid")
    payload["rag_provider"] = str(payload.get("rag_provider") or "memory")
    payload["query_count"] = len(evidence_queries)
    payload["provided_packet_count"] = len(provided_packets)
    payload["collected_packet_count"] = len(collected_packets)
    payload["total_packet_count"] = len(provided_packets) + len(collected_packets)
    payload["queries"] = list(payload.get("queries") or [])
    if "provider_counts" not in payload or not isinstance(payload.get("provider_counts"), dict):
        payload["provider_counts"] = {}
    if collected_packets:
        payload["evidence_packets"] = [dict(item) for item in collected_packets]
    return payload


def _build_verification_summary(
    verifications: list[CandidateVerification],
    *,
    verifier_model_id: str | None,
    evidence_packets: list[dict[str, Any]],
) -> dict[str, Any]:
    verified_candidate_ids = [
        item.candidate_id
        for item in verifications
        if item.status == "verified"
    ]
    failed_candidate_ids = [
        item.candidate_id
        for item in verifications
        if item.status == "failed"
    ]
    skipped_candidate_ids = [
        item.candidate_id
        for item in verifications
        if item.status == "skipped"
    ]
    return {
        "verifier_model_id": verifier_model_id,
        "evidence_packet_count": len(evidence_packets),
        "verified_candidate_ids": verified_candidate_ids,
        "failed_candidate_ids": failed_candidate_ids,
        "skipped_candidate_ids": skipped_candidate_ids,
        "verifications": [item.to_dict() for item in verifications],
    }


def _claim_cards_from_state(debate_state: dict[str, Any] | None) -> list[Any]:
    if not isinstance(debate_state, dict):
        return []
    cards = debate_state.get("claim_cards")
    if not isinstance(cards, list):
        return []
    normalized = []
    for item in cards:
        if not isinstance(item, dict):
            continue
        normalized.append(
            type(
                "ClaimCardProxy",
                (),
                {
                    "speaker_role_id": str(item.get("speaker_role_id") or ""),
                    "claim": str(item.get("claim") or ""),
                },
            )()
        )
    return normalized


def _render_debate_state(debate_state: dict[str, Any] | None) -> str:
    if not isinstance(debate_state, dict):
        return "No structured debate state available."
    lines = []
    task_brief = debate_state.get("task_brief")
    if isinstance(task_brief, str) and task_brief.strip():
        lines.append(f"Task brief: {task_brief.strip()}")
    global_summary = debate_state.get("global_summary")
    if isinstance(global_summary, str) and global_summary.strip():
        lines.append(f"Global summary: {global_summary.strip()}")
    role_stances = debate_state.get("role_stance_summaries")
    if isinstance(role_stances, list) and role_stances:
        lines.append("Role stance summaries:")
        for item in role_stances:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"- role_id={item.get('role_id')} latest_round={item.get('latest_round_index')} summary={item.get('stance_summary')}"
            )
    claim_cards = debate_state.get("claim_cards")
    if isinstance(claim_cards, list) and claim_cards:
        lines.append("Claim cards:")
        for item in claim_cards[:8]:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"- claim_id={item.get('claim_id')} role={item.get('speaker_role_id')} round={item.get('round_index')} claim={item.get('claim')}"
            )
    open_objections = debate_state.get("open_objections")
    if isinstance(open_objections, list) and open_objections:
        lines.append("Open objections:")
        lines.extend(f"- {item}" for item in open_objections if isinstance(item, str))
    open_questions = debate_state.get("open_questions")
    if isinstance(open_questions, list) and open_questions:
        lines.append("Open questions:")
        lines.extend(f"- {item}" for item in open_questions if isinstance(item, str))
    return "\n".join(lines).strip()


def _chunk_evidence_packets(
    evidence_packets: list[dict[str, Any]],
    *,
    max_input_tokens: int,
    enabled: bool,
) -> list[list[dict[str, Any]]]:
    if not evidence_packets:
        return []
    if not enabled:
        return [list(evidence_packets)]
    per_chunk_budget = max(400, math.floor(max_input_tokens * 0.28))
    chunks: list[list[dict[str, Any]]] = []
    current_chunk: list[dict[str, Any]] = []
    current_tokens = 0
    for packet in evidence_packets:
        rendered = f"{packet.get('evidence_id')} {packet.get('title')} {packet.get('content')}"
        packet_tokens = estimate_text_tokens(str(rendered)).tokens
        if current_chunk and current_tokens + packet_tokens > per_chunk_budget:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0
        current_chunk.append(packet)
        current_tokens += packet_tokens
    if current_chunk:
        chunks.append(current_chunk)
    return chunks or [list(evidence_packets)]


def _extract_json_payload(raw_content: str) -> dict[str, Any] | None:
    text = raw_content.strip()
    if not text:
        return None
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None


def _parse_evidence_gate(value: Any) -> Any:
    if not isinstance(value, dict):
        return resolve_evidence_gate_status(skipped=True)
    status = value.get("status")
    reason = value.get("reason")
    metadata = value.get("metadata")
    gate_metadata = metadata if isinstance(metadata, dict) else None
    if status == "verified":
        return resolve_evidence_gate_status(verified=True, metadata=gate_metadata)
    if status == "failed":
        return resolve_evidence_gate_status(
            error=str(reason or "verification failed"),
            metadata=gate_metadata,
        )
    return resolve_evidence_gate_status(skipped=True, metadata=gate_metadata)


_RESUME_STAGE_ORDER = {
    "research_context_prepared": 1,
    "controlled_execution_context_prepared": 2,
    "protocol_completed": 3,
    "before_evidence_collection": 4,
    "evidence_collected": 5,
    "verification_completed": 6,
    "evaluation_completed": 7,
    "research_artifacts_completed": 8,
}


def _resolve_resume_execution_payload(metadata: Mapping[str, Any] | None) -> ResumeExecutionPayload | None:
    if not isinstance(metadata, Mapping):
        return None
    candidates = [metadata.get("resume_payload"), metadata.get("recovery_state")]
    summary = metadata.get("summary")
    if isinstance(summary, Mapping):
        candidates.append(summary.get("resume_payload"))
        candidates.append(summary.get("recovery_state"))
    for candidate in candidates:
        payload = candidate.get("resume_payload") if isinstance(candidate, Mapping) else candidate
        parsed = _parse_resume_execution_payload(payload)
        if parsed is not None:
            return parsed
    return None


def _parse_resume_execution_payload(payload: Any) -> ResumeExecutionPayload | None:
    if not isinstance(payload, Mapping):
        return None
    checkpoint = _mapping_to_plain_dict(payload.get("checkpoint"))
    stage = str(payload.get("stage") or checkpoint.get("stage") or "").strip()
    if not stage:
        return None
    guidance_messages = [
        str(item).strip()
        for item in (payload.get("guidance_messages") or [])
        if isinstance(item, str) and item.strip()
    ]
    metadata_state = _mapping_to_plain_dict(payload.get("metadata_state"))
    precomputed_artifacts = _mapping_to_plain_dict(payload.get("precomputed_artifacts"))
    protocol_artifacts = _mapping_to_plain_dict(payload.get("protocol_artifacts"))
    candidates = _candidate_outputs_from_payload(payload.get("candidates"))
    evidence_packets = _list_of_dicts(payload.get("evidence_packets"))
    evidence_collection_summary = (
        _mapping_to_plain_dict(payload.get("evidence_collection_summary"))
        if isinstance(payload.get("evidence_collection_summary"), Mapping)
        else None
    )
    verifications = _candidate_verifications_from_payload(payload.get("verifications"))
    verification_summary = (
        _mapping_to_plain_dict(payload.get("verification_summary"))
        if isinstance(payload.get("verification_summary"), Mapping)
        else None
    )
    evaluation = _evaluation_payload_from_dict(payload.get("evaluation"))
    role_task_snapshot = _mapping_to_plain_dict(payload.get("role_task_snapshot"))
    executor = str(payload.get("executor") or "restart_attempt").strip().lower()
    if executor == "continue_from_checkpoint" and not _can_continue_from_checkpoint(
        stage=stage,
        candidates=candidates,
        evaluation_payload=evaluation,
        role_task_snapshot=role_task_snapshot,
    ):
        executor = "restart_attempt"
    if executor not in {"continue_from_checkpoint", "restart_attempt"}:
        executor = "restart_attempt"
    return ResumeExecutionPayload(
        executor=executor,
        stage=stage,
        checkpoint=checkpoint,
        guidance_messages=guidance_messages,
        metadata_state=metadata_state,
        precomputed_artifacts=precomputed_artifacts,
        protocol_artifacts=protocol_artifacts,
        candidates=candidates,
        evidence_packets=evidence_packets,
        evidence_collection_summary=evidence_collection_summary,
        verifications=verifications,
        verification_summary=verification_summary,
        evaluation=evaluation,
        role_task_snapshot=role_task_snapshot,
    )


def _build_resume_execution_payload(
    *,
    stage: str,
    checkpoint: Mapping[str, Any] | None,
    guidance_messages: list[str],
    metadata: Mapping[str, Any],
    precomputed_artifacts: Mapping[str, Any],
    protocol_artifacts: Mapping[str, Any],
    candidates: list[CandidateOutput],
    evidence_packets: list[dict[str, Any]],
    evidence_collection_summary: Mapping[str, Any] | None,
    verifications: list[CandidateVerification],
    verification_summary: Mapping[str, Any] | None,
    evaluation_payload: EvaluationPayload | None,
    role_task_snapshot: Mapping[str, Any] | None,
) -> ResumeExecutionPayload:
    executor: Literal["continue_from_checkpoint", "restart_attempt"] = (
        "continue_from_checkpoint"
        if _can_continue_from_checkpoint(
            stage=stage,
            candidates=candidates,
            evaluation_payload=evaluation_payload,
            role_task_snapshot=role_task_snapshot,
        )
        else "restart_attempt"
    )
    return ResumeExecutionPayload(
        executor=executor,
        stage=stage,
        checkpoint=_mapping_to_plain_dict(checkpoint),
        guidance_messages=list(guidance_messages),
        metadata_state=_build_resume_metadata_state(metadata),
        precomputed_artifacts=_mapping_to_plain_dict(precomputed_artifacts),
        protocol_artifacts=_mapping_to_plain_dict(protocol_artifacts),
        candidates=list(candidates),
        evidence_packets=_list_of_dicts(evidence_packets),
        evidence_collection_summary=_mapping_to_plain_dict(evidence_collection_summary),
        verifications=list(verifications),
        verification_summary=_mapping_to_plain_dict(verification_summary),
        evaluation=evaluation_payload,
        role_task_snapshot=_mapping_to_plain_dict(role_task_snapshot),
    )


def _build_resume_metadata_state(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        return {}
    state: dict[str, Any] = {}
    research_runtime = metadata.get("research_runtime")
    if isinstance(research_runtime, Mapping):
        state["research_runtime"] = _mapping_to_plain_dict(research_runtime)
    return state


def _apply_resume_metadata_state(
    metadata: Mapping[str, Any] | None,
    resume_state: Mapping[str, Any] | None,
) -> dict[str, Any]:
    updated = dict(metadata or {})
    if not isinstance(resume_state, Mapping):
        return updated
    research_runtime = resume_state.get("research_runtime")
    if isinstance(research_runtime, Mapping):
        updated["research_runtime"] = _mapping_to_plain_dict(research_runtime)
    return updated


def _sanitize_resume_summary(summary: Any) -> dict[str, Any]:
    payload = _mapping_to_plain_dict(summary)
    recovery_state = payload.get("recovery_state")
    if isinstance(recovery_state, dict) and "resume_payload" in recovery_state:
        trimmed = dict(recovery_state)
        trimmed.pop("resume_payload", None)
        payload["recovery_state"] = trimmed
    return payload


def _merge_guidance_messages(base: list[str], extra: list[str]) -> list[str]:
    merged: list[str] = []
    for item in [*base, *extra]:
        text = str(item).strip()
        if not text or text in merged:
            continue
        merged.append(text)
    return merged


def _resume_stage_covers(stage: str | None, target_stage: str) -> bool:
    if not stage:
        return False
    stage_rank = _RESUME_STAGE_ORDER.get(stage)
    target_rank = _RESUME_STAGE_ORDER.get(target_stage)
    if stage_rank is None or target_rank is None:
        return False
    return stage_rank >= target_rank


def _can_continue_from_checkpoint(
    *,
    stage: str,
    candidates: list[CandidateOutput],
    evaluation_payload: EvaluationPayload | None,
    role_task_snapshot: Mapping[str, Any] | None,
) -> bool:
    if stage in {"research_context_prepared", "controlled_execution_context_prepared"}:
        return True
    if _resume_stage_covers(stage, "protocol_completed") and not candidates:
        return False
    if stage in {"evaluation_completed", "research_artifacts_completed"} and evaluation_payload is None:
        return False
    if _stage_supports_role_task_resume(stage) and _role_task_snapshot_has_completed_work(role_task_snapshot):
        return True
    return stage in _RESUME_STAGE_ORDER


def _stage_supports_role_task_resume(stage: str) -> bool:
    if stage in {
        "teacher_generation",
        "student_generation",
        "research_planning",
        "research_workers",
        "research_synthesis",
        "candidate_evaluation",
        "controlled_execution_planner",
        "controlled_execution_executor",
        "controlled_execution_evaluator",
    }:
        return True
    return (
        stage.startswith("debate_round_")
        or stage.startswith("dr_zero_")
        or stage.startswith("controlled_execution_controller:")
        or stage.startswith("controlled_execution_exec:")
        or stage.startswith("verification_chunk_")
    )


def _role_task_snapshot_has_completed_work(snapshot: Mapping[str, Any] | None) -> bool:
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


def _candidate_outputs_from_payload(value: Any) -> list[CandidateOutput]:
    if not isinstance(value, list):
        return []
    payload: list[CandidateOutput] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        candidate_id = item.get("candidate_id")
        role_id = item.get("role_id")
        content = item.get("content")
        if not all(isinstance(field, str) and field.strip() for field in (candidate_id, role_id, content)):
            continue
        metadata = item.get("metadata")
        payload.append(
            CandidateOutput(
                candidate_id=str(candidate_id),
                role_id=str(role_id),
                content=str(content),
                metadata=_mapping_to_plain_dict(metadata),
            )
        )
    return payload


def _candidate_verifications_from_payload(value: Any) -> list[CandidateVerification]:
    if not isinstance(value, list):
        return []
    payload: list[CandidateVerification] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        candidate_id = item.get("candidate_id")
        status = item.get("status")
        rationale = item.get("rationale")
        if (
            not isinstance(candidate_id, str)
            or not candidate_id.strip()
            or status not in {"verified", "skipped", "failed"}
            or not isinstance(rationale, str)
        ):
            continue
        citations = item.get("citations")
        issues = item.get("issues")
        payload.append(
            CandidateVerification(
                candidate_id=candidate_id.strip(),
                status=status,
                rationale=rationale,
                citations=_list_of_dicts(citations),
                issues=[str(issue) for issue in issues if isinstance(issue, str) and issue.strip()]
                if isinstance(issues, list)
                else [],
                metadata=_mapping_to_plain_dict(item.get("metadata")),
            )
        )
    return payload


def _candidate_scores_from_payload(value: Any) -> list[CandidateScore]:
    if not isinstance(value, list):
        return []
    payload: list[CandidateScore] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        candidate_id = item.get("candidate_id")
        score_value = _safe_float(item.get("score"))
        if not isinstance(candidate_id, str) or not candidate_id.strip() or score_value is None:
            continue
        payload.append(
            CandidateScore(
                candidate_id=candidate_id,
                score=score_value,
                rationale=str(item.get("rationale") or "saved evaluation score"),
                evidence_gate=_parse_evidence_gate(item.get("evidence_gate")),
            )
        )
    return payload


def _evaluation_payload_from_dict(value: Any) -> EvaluationPayload | None:
    if not isinstance(value, Mapping):
        return None
    policy = _mapping_to_plain_dict(value.get("policy"))
    selected_candidate_id = value.get("selected_candidate_id")
    scores = value.get("scores")
    if not isinstance(scores, list):
        return None
    return EvaluationPayload(
        policy=policy,
        selected_candidate_id=(
            str(selected_candidate_id).strip()
            if isinstance(selected_candidate_id, str) and selected_candidate_id.strip()
            else None
        ),
        scores=_list_of_dicts(scores),
    )


def _mapping_to_plain_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): _json_safe_clone(item)
        for key, item in value.items()
        if isinstance(key, str)
    }


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [
        _mapping_to_plain_dict(item)
        for item in value
        if isinstance(item, Mapping)
    ]


def _json_safe_clone(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe_clone(item)
            for key, item in value.items()
            if isinstance(key, str)
        }
    if isinstance(value, list):
        return [_json_safe_clone(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe_clone(item) for item in value]
    return value


def _remaining_run_steps(*, stage: str, research_enabled: bool) -> list[str]:
    ordered_steps = [
        ("research_context_prepared", "prepare research context"),
        ("controlled_execution_context_prepared", "prepare controlled execution context"),
        ("protocol_completed", "finish protocol role generation"),
        ("before_evidence_collection", "collect evidence packets"),
        ("evidence_collected", "complete evidence collection"),
        ("verification_completed", "verify candidate claims"),
        ("evaluation_completed", "evaluate and select the winning candidate"),
        ("research_artifacts_completed", "build final research artifacts"),
    ]
    completed_stages = {key for key, _ in ordered_steps}
    if stage.startswith("debate_round_"):
        return [
            "finish remaining debate rounds",
            "verify candidate claims",
            "evaluate and select the winning candidate",
            *(
                ["build final research artifacts"]
                if research_enabled
                else []
            ),
        ]
    if stage.startswith("dr_zero_rollout_"):
        return [
            "finish remaining solver rollouts",
            "verify rollout candidates",
            "evaluate and select the winning rollout",
        ]
    if stage == "dr_zero_proposals_generated":
        return [
            "generate solver rollouts",
            "verify rollout candidates",
            "evaluate and select the winning rollout",
        ]
    remaining: list[str] = []
    seen_current = False
    for key, label in ordered_steps:
        if key == stage:
            seen_current = True
            continue
        if not seen_current:
            continue
        if key == "research_artifacts_completed" and not research_enabled:
            continue
        remaining.append(label)
    if stage not in completed_stages and not remaining:
        remaining.append("resume from the latest checkpoint")
    return remaining


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
