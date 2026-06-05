"""Multi-agent orchestration runtime."""

from __future__ import annotations

import json
import math
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
from mochi.agents.multi_agent.evaluator import (
    CandidateVerification,
    CandidateScore,
    LLMFirstScoringPolicy,
    evidence_gate_rank,
    resolve_evidence_gate_status,
)
from mochi.agents.multi_agent.protocols import (
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
from mochi.agents.context_snapshot import estimate_text_tokens
from mochi.agents.invocation import AgentInvocationRequest
from mochi.backends.types import GenerationResult, Message

RunState = Literal["queued", "running", "awaiting_guidance", "succeeded", "failed", "cancelled"]
RunEventType = Literal["state_changed", "guidance", "role_output", "verification", "evaluation", "artifact"]

RUN_STATE_TRANSITIONS: dict[RunState, tuple[RunState, ...]] = {
    "queued": ("running", "cancelled"),
    "running": ("awaiting_guidance", "succeeded", "failed", "cancelled"),
    "awaiting_guidance": ("running", "failed", "cancelled"),
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
    guidance_messages: list[str] = field(default_factory=list)
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
    ) -> None:
        self._engine = engine
        self._policy = scoring_policy or LLMFirstScoringPolicy()
        self._invocation_traces: list[dict[str, Any]] = []

    async def run(self, request: MultiAgentRunRequest) -> MultiAgentRunResult:
        """Run one multi-agent orchestration."""
        run_id = request.run_id or str(uuid4())
        protocol_config = parse_protocol_config(request.protocol)
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

        guidance_messages = [item.strip() for item in request.guidance_messages if item.strip()]
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

        selected_models_roles = _resolve_selected_model_roles(request.metadata)
        effective_metadata = dict(request.metadata)
        research_policy = ResearchDebatePolicy.from_metadata(effective_metadata)
        precomputed_artifacts: dict[str, Any] = {}
        evidence_packets: list[dict[str, Any]] = []
        evidence_collection_summary: dict[str, Any] | None = None
        if research_policy.enabled and isinstance(protocol_config, MultiAgentDebateProtocol):
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
        protocol_metadata = dict(effective_metadata)
        candidates, protocol_artifacts = await self._run_protocol(
            request.task_input,
            protocol_config=protocol_config,
            guidance_messages=guidance_messages,
            selected_models_roles=selected_models_roles,
            metadata=protocol_metadata,
            emit=emit,
        )
        protocol_artifacts = {**precomputed_artifacts, **protocol_artifacts}
        for artifact_name, artifact_content in protocol_artifacts.items():
            if isinstance(artifact_content, dict):
                emit(
                    "artifact",
                    asdict(ArtifactPayload(name=artifact_name, content=artifact_content)),
                )
        if evidence_collection_summary is None:
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
        if research_policy.enabled and isinstance(protocol_config, MultiAgentDebateProtocol):
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
        evaluation_payload = EvaluationPayload(
            policy=self._policy.to_dict(),
            selected_candidate_id=selected_candidate_id,
            scores=[score.to_dict() for score in scores],
        )
        emit("evaluation", asdict(evaluation_payload))

        final_candidate = _candidate_by_id(candidates, selected_candidate_id)
        artifacts = {
            "protocol": protocol_config.protocol,
            "selected_candidate_id": selected_candidate_id,
            "final_answer": final_candidate.content if final_candidate is not None else "",
            "candidate_count": len(candidates),
        }
        artifacts.update(protocol_artifacts)
        if self._invocation_traces:
            subagent_runtime = _build_subagent_runtime_artifact(self._invocation_traces)
            artifacts["subagent_runtime"] = subagent_runtime
            emit("artifact", asdict(ArtifactPayload(name="subagent_runtime", content=subagent_runtime)))
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
            metadata=effective_metadata,
        )

    async def _run_protocol(
        self,
        task_input: str,
        *,
        protocol_config: ProtocolConfig,
        guidance_messages: list[str],
        selected_models_roles: dict[str, str],
        metadata: dict[str, Any],
        emit: Any,
    ) -> tuple[list[CandidateOutput], dict[str, Any]]:
        if self._can_use_model_backed_execution(selected_models_roles):
            candidates, protocol_artifacts = await self._run_model_backed_protocol(
                task_input,
                protocol_config=protocol_config,
                guidance_messages=guidance_messages,
                selected_models_roles=selected_models_roles,
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
            if not teacher_model_id or not student_model_id:
                return [], {}

            teacher_output = await self._generate_role_candidate(
                role_id=teacher_role.role_id,
                role_title=teacher_role.title,
                role_instruction=teacher_role.instruction,
                model_id=teacher_model_id,
                task_input=task_input,
                guidance_messages=guidance_messages,
                supporting_candidates=[],
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
                ),
            )

            student_output = await self._generate_role_candidate(
                role_id=student_role.role_id,
                role_title=student_role.title,
                role_instruction=student_role.instruction,
                model_id=student_model_id,
                task_input=task_input,
                guidance_messages=guidance_messages,
                supporting_candidates=[teacher_output],
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
                ),
            )
            return [teacher_output, student_output], {}

        if isinstance(protocol_config, DrZeroSelfEvolveProtocol):
            return await self._run_model_backed_dr_zero(
                task_input=task_input,
                protocol_config=protocol_config,
                guidance_messages=guidance_messages,
                selected_models_roles=selected_models_roles,
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
        proposer_output = await self._generate_role_candidate(
            role_id=proposer_role.role_id,
            role_title=proposer_role.title,
            role_instruction=proposer_role.instruction,
            model_id=proposer_model_id,
            task_input=task_input,
            guidance_messages=guidance_messages,
            supporting_candidates=[],
            prompt_override=proposer_prompt,
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
                solver_prompt = _build_dr_zero_solver_prompt(
                    root_task=task_input,
                    synthetic_task=synthetic_task,
                    guidance_messages=guidance_messages,
                    rollout_index=rollout_index,
                )
                raw_solver_output = await self._generate_role_candidate(
                    role_id=solver_role.role_id,
                    role_title=solver_role.title,
                    role_instruction=solver_role.instruction,
                    model_id=solver_model_id,
                    task_input=task_input,
                    guidance_messages=guidance_messages,
                    supporting_candidates=[proposer_output],
                    prompt_override=solver_prompt,
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
        content, diagnostics = await self._invoke_configured_text(
            model_id=model_id,
            system_prompt=role_instruction,
            user_prompt=prompt,
            temperature=0.2,
            max_tokens=1024,
            execution_profile="subagent_readonly",
            tool_mode="auto",
            system_prompt_addendum=(
                f"Role identity: {role_title} ({role_id}).\n"
                f"Role instruction: {role_instruction}"
            ),
            session_scope=f"role::{role_id}",
        )
        return CandidateOutput(
            candidate_id=role_id,
            role_id=role_id,
            content=content,
            metadata={"model_id": model_id, "diagnostics": diagnostics},
        )

    def _make_shared_generate(
        self,
        *,
        execution_profile: str,
        tool_mode: str = "auto",
        session_scope: str,
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
                reasoning_effort=reasoning_effort,
                execution_profile=execution_profile,
                tool_mode=tool_mode,
                system_prompt_addendum=system_prompt,
                session_scope=session_scope,
                fallback_generate=fallback_generate if callable(fallback_generate) else None,
            )
            return GenerationResult(content=content, model=model_id)

        return generate

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
        research_appendix = _research_prompt_appendix(metadata)
        for round_index in range(1, protocol_config.rounds + 1):
            for role, model_id in ((debater_a, debater_a_model_id), (debater_b, debater_b_model_id)):
                prompt, snapshot = debate_context.build_role_prompt(
                    role_id=role.role_id,
                    role_title=role.title,
                    stage=f"debate_round_{round_index}",
                    reserved_output_tokens=1024,
                )
                if research_appendix:
                    prompt = f"{prompt}\n\nResearch context:\n{research_appendix}"
                snapshots.append(snapshot.to_dict())
                candidate = await self._generate_role_candidate(
                    role_id=role.role_id,
                    role_title=role.title,
                    role_instruction=role.instruction,
                    model_id=model_id,
                    task_input=task_input,
                    guidance_messages=guidance_messages,
                    supporting_candidates=list(latest_candidates.values()),
                    prompt_override=prompt,
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
        research_plan = await build_research_plan(
            task_input=task_input,
            guidance_messages=guidance_messages,
            existing_queries=evidence_queries,
            policy=policy,
            planner_model_id=planner_model_id,
            generate=generate if callable(generate) else None,
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
        worker_outputs = await run_research_workers(
            task_input=task_input,
            research_plan=research_plan,
            evidence_packets=evidence_packets,
            policy=policy,
            local_worker_model_id=local_worker_model_id,
            generate=generate if callable(generate) else None,
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
        )
        artifacts: dict[str, Any] = {"claim_evidence_map": claim_evidence_map}
        if isinstance(updated_debate_state, dict):
            artifacts["debate_state"] = updated_debate_state
        if policy.wants_output("research_brief"):
            artifacts["research_brief"] = await synthesize_research_brief(
                task_input=task_input,
                selected_candidate=selected_candidate.to_dict() if selected_candidate is not None else None,
                research_plan=research_plan,
                source_quality_table=source_quality_table,
                claim_evidence_map=claim_evidence_map,
                worker_outputs=worker_outputs,
                synthesizer_model_id=synthesizer_model_id,
                generate=generate if callable(generate) else None,
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
            result = await generate(
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
        )
        if not candidates or not callable(generate):
            return None
        judge_model_id = _resolve_judge_model_id(
            protocol_config=protocol_config,
            selected_models_roles=selected_models_roles,
        )
        if not judge_model_id:
            return None

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
        result = await generate(
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
        )
        return _parse_judge_scores(result.content, candidates)

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
    parsed = _parse_json_payload(content)
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


def _parse_json_payload(content: str) -> Any:
    text = content.strip()
    if not text:
        return None
    candidates = [text]
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            stripped = part.strip()
            if stripped.startswith("json"):
                stripped = stripped[4:].strip()
            if stripped:
                candidates.append(stripped)
    object_start = text.find("{")
    object_end = text.rfind("}")
    if 0 <= object_start < object_end:
        candidates.append(text[object_start : object_end + 1])
    array_start = text.find("[")
    array_end = text.rfind("]")
    if 0 <= array_start < array_end:
        candidates.append(text[array_start : array_end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


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


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
