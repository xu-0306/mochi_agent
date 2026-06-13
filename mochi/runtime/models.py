"""Runtime models for background tasks and approvals."""

from __future__ import annotations

from typing import Any, Literal

from mochi.backends.inference_capabilities import ReasoningEffort
from mochi.api.attachment_schema import AttachmentPayload
from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class ApprovalResolution(BaseModel):
    """Resolution payload for one approval request."""

    approved: bool
    reason: str | None = None


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
    attempts: list[dict[str, Any]] = Field(default_factory=list)
    all_records: list[dict[str, Any]] = Field(default_factory=list)
    training_ready_records: list[dict[str, Any]] = Field(default_factory=list)
    excluded_records_summary: dict[str, Any] = Field(default_factory=dict)
