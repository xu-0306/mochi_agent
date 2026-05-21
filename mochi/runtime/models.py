"""Runtime models for background tasks and approvals."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TaskCreateRequest(BaseModel):
    """Request payload for creating an autonomous task."""

    model_config = ConfigDict(populate_by_name=True)

    input: str = Field(min_length=1, alias="input_message")
    session_id: str | None = None
    project_id: str | None = None
    workspace_dir: str | None = None
    inference_overrides: dict[str, Any] = Field(default_factory=dict)


class ApprovalResolution(BaseModel):
    """Resolution payload for one approval request."""

    approved: bool
    reason: str | None = None
