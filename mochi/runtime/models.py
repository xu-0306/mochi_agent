"""Runtime models for background tasks and approvals."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mochi.api.attachment_schema import AttachmentPayload
from mochi.backends.inference_capabilities import ReasoningEffort
from mochi.runtime.goal_strategy_registry import get_goal_strategy_entry


class TaskCreateRequest(BaseModel):
    """Request payload for creating an autonomous task."""

    model_config = ConfigDict(populate_by_name=True)

    input: str = Field(min_length=1, alias="input_message")
    session_id: str | None = None
    project_id: str | None = None
    project_workspace_dir: str | None = None
    workspace_dir: str | None = None
    task_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    inference_overrides: dict[str, Any] = Field(default_factory=dict)


class TaskMessageRequest(BaseModel):
    """Request payload for appending guidance-only transcript to a delegated task."""

    content: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_content(self) -> TaskMessageRequest:
        if not self.content.strip():
            raise ValueError("content is required")
        return self


class ApprovalReplayOverride(BaseModel):
    """Optional approval replay payload used instead of the stored mutation call."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _upgrade_patch_text_only_payload(
        cls,
        value: Any,
    ) -> Any:
        if not isinstance(value, dict):
            return value
        if "tool_name" in value or "arguments" in value:
            return value
        patch_text = value.get("patch_text")
        if isinstance(patch_text, str) and patch_text.strip():
            return {
                "tool_name": "apply_patch",
                "arguments": {"patch": patch_text},
            }
        return value


class ApprovalResolution(BaseModel):
    """Resolution payload for one approval request."""

    decision: Literal["approve_once", "approve_and_save_rule", "reject"]
    reason: str | None = None
    rule: dict[str, Any] | None = None
    replay_override: ApprovalReplayOverride | None = None


class AutoReviewSummary(BaseModel):
    """UI-safe projection of one deterministic auto-review decision."""

    model_config = ConfigDict(extra="forbid")

    auto_review_decision: Literal["allow", "require_approval", "deny"] | None = None
    auto_review_input_digest: str | None = None
    auto_review_policy_version: str | None = None
    auto_review_reviewer_version: str | None = None
    auto_review_risk_factors: list[str] = Field(default_factory=list)
    auto_review_reason_codes: list[str] = Field(default_factory=list)
    auto_review_source: Literal["policy_auto_allow", "reviewed_allow"] | None = None


GoalExecutionMode = Literal["single_agent", "workflow"]
GoalInteractionMode = Literal["goal", "workflow"]
GoalExecutionTopology = Literal["single_agent", "multi_agent"]
GoalStrategyKind = Literal["protocol", "workflow_template", "execution_strategy"]
GoalSelectionSource = Literal[
    "explicit_override",
    "semantic_registry_selector",
    "bounded_fallback",
    "safe_default",
    "legacy_migration",
]


class GoalStrategyRegistryEntry(BaseModel):
    """Serialized Goal strategy registry entry."""

    id: str
    name: str
    display_name: str
    description: str
    when_to_use: str
    when_not_to_use: str
    execution_topology: GoalExecutionTopology
    kind: GoalStrategyKind = "execution_strategy"
    protocol_id: str | None = None
    required_capabilities: list[str] = Field(default_factory=list)
    approval_profile: str
    control_scope: str
    interrupt_policy: str
    resume_policy: str
    event_contract: str
    success_signals: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    fallback_strategy_ids: list[str] = Field(default_factory=list)
    requires_confirmation: bool = False
    is_default: bool = False
    available: bool = True
    availability_reason: str | None = None
    deprecated: bool = False
    override_label: str | None = None
    selection_guidance: str | None = None


class GoalStrategyRegistryResponse(BaseModel):
    """Response payload for frontend Goal strategy inspection and overrides."""

    type: Literal["goal_strategy_registry"] = "goal_strategy_registry"
    default_strategy_id: str
    entries: list[GoalStrategyRegistryEntry]
ActiveGoalTurnLane = Literal["active_goal_turn"]
ActiveGoalTurnKind = Literal[
    "answer_question",
    "explain_goal_state",
    "steer",
    "replan",
    "lifecycle",
    "exit_to_chat",
    "clarify",
]
ActiveGoalTurnSelectionSource = Literal["bounded_fallback", "semantic_registry_selector"]


class ActiveGoalTurnDecisionRequest(BaseModel):
    """Bounded request payload for classifying an active-goal follow-up turn."""

    model_config = ConfigDict(extra="forbid")

    message: str = ""

    @model_validator(mode="after")
    def _validate_message(self) -> ActiveGoalTurnDecisionRequest:
        if not self.message.strip():
            raise ValueError("message is required")
        return self


class ActiveGoalTurnDecision(BaseModel):
    """Bounded backend decision for routing an active-goal follow-up turn."""

    model_config = ConfigDict(extra="forbid")

    lane: ActiveGoalTurnLane = "active_goal_turn"
    kind: ActiveGoalTurnKind
    confidence: float = Field(ge=0.0, le=1.0)
    selection_source: ActiveGoalTurnSelectionSource = "bounded_fallback"
    selection_reason: str = Field(min_length=1)
    requires_confirmation: bool = False
    goal_status: str | None = None
    linked_run_status: str | None = None
    recommended_action: str | None = None


class GoalCreateRequest(BaseModel):
    """Request payload for creating a durable high-level goal."""

    objective: str = Field(min_length=1)
    title: str | None = None
    goal_type: str | None = None
    execution_mode: GoalExecutionMode = "single_agent"
    interaction_mode: GoalInteractionMode | None = None
    execution_topology: GoalExecutionTopology | None = None
    strategy_id: str | None = None
    selection_source: GoalSelectionSource | None = None
    selection_reason: str | None = None
    protocol_id: str | None = None
    bound_run_id: str | None = None
    protocol_selection: str | None = None
    selection_rationale: str | None = None
    topic: str | None = None
    project_id: str | None = None
    workspace_dir: str | None = None
    run_policy: dict[str, Any] = Field(default_factory=dict)
    capability_policy: dict[str, Any] = Field(default_factory=dict)
    selected_models_roles: dict[str, Any] = Field(default_factory=dict)
    source_manifest: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _populate_strategy_compatibility_fields(self) -> GoalCreateRequest:
        if self.strategy_id is None and isinstance(self.protocol_selection, str) and self.protocol_selection.strip():
            self.strategy_id = self.protocol_selection.strip()
        if (
            self.selection_reason is None
            and isinstance(self.selection_rationale, str)
            and self.selection_rationale.strip()
        ):
            self.selection_reason = self.selection_rationale.strip()
        if self.protocol_selection is None and isinstance(self.strategy_id, str) and self.strategy_id.strip():
            self.protocol_selection = self.strategy_id.strip()
        if (
            self.selection_rationale is None
            and isinstance(self.selection_reason, str)
            and self.selection_reason.strip()
        ):
            self.selection_rationale = self.selection_reason.strip()
        strategy_id = str(self.strategy_id or "").strip()
        protocol_id = str(self.protocol_id or "").strip()
        if strategy_id and protocol_id:
            entry = get_goal_strategy_entry(strategy_id)
            expected_protocol_id = (
                str((entry.protocol_id if entry is not None else strategy_id) or "").strip()
                or strategy_id
            )
            if expected_protocol_id != protocol_id:
                raise ValueError(
                    f"Conflicting strategy_id/protocol_id: strategy {strategy_id} requires protocol {expected_protocol_id}, not {protocol_id}."
                )
        return self


class GoalAttemptResponse(BaseModel):
    """Serialized goal-attempt payload for API responses."""

    attempt_id: str
    goal_id: str
    attempt_index: int
    status: str
    trigger: str | None = None
    agent_run_id: str | None = None
    summary: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    latest_error: str | None = None
    created_at: str
    updated_at: str
    started_at: str | None = None
    finished_at: str | None = None


class GoalResponse(BaseModel):
    """Serialized goal payload for API responses."""

    goal_id: str
    objective: str
    title: str | None = None
    goal_type: str | None = None
    execution_mode: GoalExecutionMode = "single_agent"
    interaction_mode: GoalInteractionMode = "goal"
    execution_topology: GoalExecutionTopology = "single_agent"
    strategy_id: str | None = None
    selection_source: GoalSelectionSource | None = None
    selection_reason: str | None = None
    protocol_id: str | None = None
    bound_run_id: str | None = None
    protocol_selection: str | None = None
    selection_rationale: str | None = None
    topic: str | None = None
    project_id: str | None = None
    workspace_dir: str | None = None
    status: str
    current_attempt_id: str | None = None
    run_policy: dict[str, Any] = Field(default_factory=dict)
    capability_policy: dict[str, Any] = Field(default_factory=dict)
    source_manifest: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    latest_error: str | None = None
    attempts: list[GoalAttemptResponse] = Field(default_factory=list)
    created_at: str
    updated_at: str
    started_at: str | None = None
    finished_at: str | None = None


class AgentRunArtifact(BaseModel):
    """Agent Run artifact metadata."""

    artifact_id: str | None = None
    artifact_type: str = Field(min_length=1)
    title: str | None = None
    uri: str | None = None
    mime_type: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentRunCreateRequest(BaseModel):
    """Request payload for creating an Agent Run."""

    protocol_id: str = Field(min_length=1)
    title: str | None = None
    topic: str | None = None
    project_id: str | None = None
    workspace_dir: str | None = None
    reasoning_effort: ReasoningEffort | None = None
    selected_models_roles: dict[str, Any] = Field(default_factory=dict)
    evaluation_policy: dict[str, Any] = Field(default_factory=dict)
    run_policy: dict[str, Any] = Field(default_factory=dict)
    schedule: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    latest_error: str | None = None
    evidence_status: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[AgentRunArtifact] = Field(default_factory=list)


class AgentRunGuidanceRequest(BaseModel):
    """Request payload for appending user guidance to an Agent Run."""

    guidance: str = Field(min_length=1)
    author: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentRunMessageRequest(BaseModel):
    """Request payload for appending a workflow conversation message to an Agent Run."""

    role: Literal["user", "operator"] = "user"
    content: str = ""
    project_id: str | None = None
    workspace_dir: str | None = None
    attachments: list[AttachmentPayload] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_content_or_attachments(self) -> AgentRunMessageRequest:
        if not self.content.strip() and not self.attachments:
            raise ValueError("content or attachments is required")
        return self


SubagentMessageDeliveryMode = Literal[
    "inject_now",
    "after_current_tool",
    "next_checkpoint",
    "resume_only",
]

SubagentMessageDeliveryStatus = Literal[
    "queued",
    "accepted",
    "applied",
    "deferred",
    "cancelled",
    "rejected",
]


class AgentRunSubagentMessageRequest(AgentRunMessageRequest):
    """Request payload for appending a subagent-targeted workflow message to an Agent Run."""


class SessionSubagentMessageRequest(AgentRunMessageRequest):
    """Request payload for appending guidance to a session-scoped subagent."""

    delivery_mode: SubagentMessageDeliveryMode = "resume_only"
    interrupt: bool = False
    cancel_current_tool: bool = False


class ExecutionTranscriptEvent(BaseModel):
    """UI-safe execution transcript event."""

    type: str
    event_id: str | None = None
    dedupe_key: str | None = None
    visibility: str | None = None
    durability: str | None = None
    projection_lane: str | None = None
    message_id: str | None = None
    parent_type: str | None = None
    parent_id: str | None = None
    subagent_id: str | None = None
    role_id: str | None = None
    title: str | None = None
    model_id: str | None = None
    content: str | None = None
    summary: str | None = None
    status: str | None = None
    prompt_preview: str | None = None
    system_prompt: str | None = None
    user_prompt: str | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    arguments_preview: str | None = None
    delivery_mode: str | None = None
    delivery_status: str | None = None
    delivery_reason: str | None = None
    interrupt: bool | None = None
    cancel_current_tool: bool | None = None
    approval_ids: list[str] = Field(default_factory=list)
    tool_names: list[str] = Field(default_factory=list)
    recommended_action: str | None = None
    blocker_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None


class SubagentTranscriptSummary(BaseModel):
    """Summary view of one subagent transcript."""

    subagent_id: str
    parent_type: str
    parent_id: str
    session_id: str | None = None
    goal_id: str | None = None
    agent_run_id: str | None = None
    parent_turn_id: str | None = None
    role_id: str | None = None
    title: str | None = None
    model_id: str | None = None
    status: str = "running"
    prompt_preview: str | None = None
    summary: str | None = None
    event_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None


class SubagentTranscriptDetail(SubagentTranscriptSummary):
    """Detailed transcript payload including prompts and normalized events."""

    system_prompt: str | None = None
    user_prompt: str | None = None
    message_id: str | None = None
    delivery_mode: str | None = None
    delivery_status: str | None = None
    delivery_reason: str | None = None
    events: list[ExecutionTranscriptEvent] = Field(default_factory=list)


class AgentRunResumeRequest(BaseModel):
    """Request payload for resuming an Agent Run."""

    strategy: Literal["continue_from_checkpoint", "restart_attempt"] = (
        "continue_from_checkpoint"
    )


class AgentRunResponse(BaseModel):
    """Serialized Agent Run payload for API responses."""

    run_id: str
    protocol_id: str
    title: str | None = None
    topic: str | None = None
    project_id: str | None = None
    workspace_dir: str | None = None
    reasoning_effort: ReasoningEffort | None = None
    status: str
    selected_models_roles: dict[str, Any] = Field(default_factory=dict)
    evaluation_policy: dict[str, Any] = Field(default_factory=dict)
    run_policy: dict[str, Any] = Field(default_factory=dict)
    schedule: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    recovery_state: dict[str, Any] = Field(default_factory=dict)
    degraded: bool = False
    latest_error: str | None = None
    evidence_status: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[AgentRunArtifact] = Field(default_factory=list)
    created_at: str
    updated_at: str
    started_at: str | None = None
    finished_at: str | None = None
    events: list[dict[str, Any]] = Field(default_factory=list)


class AgentRunAttemptPackageResponse(BaseModel):
    """Serialized attempt package payload."""

    manifest_version: str
    package_type: str
    exported_at: str
    run_id: str
    protocol_id: str
    attempt_id: str | None = None
    selected_scope: str
    schedule_attempt: dict[str, Any] | None = None
    artifact_count: int = 0
    event_count: int = 0
    role_output_count: int = 0
    replay_ready: bool = False
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    events: list[dict[str, Any]] = Field(default_factory=list)
    role_outputs: list[dict[str, Any]] = Field(default_factory=list)
    evaluation_events: list[dict[str, Any]] = Field(default_factory=list)
    collector_shard_manifests: list[dict[str, Any]] = Field(default_factory=list)
    collector_provenance_manifest: dict[str, Any] | None = None
    dataset_records: list[dict[str, Any]] = Field(default_factory=list)
    run_summary: dict[str, Any] | None = None
    evidence_summary: dict[str, Any] | None = None
    verification_summary: dict[str, Any] | None = None
    final_selected_candidate: dict[str, Any] | None = None


class AgentRunDatasetPackageResponse(BaseModel):
    """Serialized dataset package payload."""

    manifest_version: str
    package_type: str
    exported_at: str
    run_id: str
    protocol_id: str
    attempt_count: int = 0
    dataset_record_count: int = 0
    training_ready_count: int = 0
    excluded_record_count: int = 0
    collector_shard_manifests: list[dict[str, Any]] = Field(default_factory=list)
    collector_provenance_manifest: dict[str, Any] | None = None
    attempts: list[dict[str, Any]] = Field(default_factory=list)
    all_records: list[dict[str, Any]] = Field(default_factory=list)
    training_ready_records: list[dict[str, Any]] = Field(default_factory=list)
    excluded_records_summary: dict[str, Any] = Field(default_factory=dict)
