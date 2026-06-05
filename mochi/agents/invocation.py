"""Shared agent invocation contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from mochi.agents.events import AgentEvent
from mochi.backends.base import BaseLLMBackend
from mochi.backends.types import AttachmentRef

ToolMode = Literal["disabled", "auto", "required"]
ExecutionProfile = Literal[
    "chat",
    "task",
    "subagent_readonly",
    "subagent_research",
    "judge",
    "verifier",
]


@dataclass
class AgentInvocationRequest:
    """One shared runtime invocation request."""

    message: str
    session_id: str | None = None
    inference_overrides: dict[str, Any] | None = None
    project_id: str | None = None
    workspace_dir: str | None = None
    task_workspace_dir: str | None = None
    permission_policy: dict[str, Any] | None = None
    selected_skill_ids: list[str] | None = None
    attachments: list[AttachmentRef] | None = None
    backend_override: BaseLLMBackend | None = None
    tool_mode: ToolMode = "auto"
    execution_profile: ExecutionProfile = "chat"
    system_prompt_addendum: str | None = None
    tool_names_override: list[str] | None = None
    tool_allowlist: list[str] | None = None
    tool_denylist: list[str] | None = None
    max_iterations_override: int | None = None
    persist_session: bool = True
    persist_turn_events: bool | None = None
    persist_learning: bool | None = None


@dataclass
class AgentInvocationDiagnostics:
    """Runtime diagnostics for one invocation."""

    execution_profile: ExecutionProfile
    tool_mode: ToolMode
    exposed_tools: list[str] = field(default_factory=list)
    matched_tool_groups: list[str] = field(default_factory=list)
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-safe dict."""
        return {
            "execution_profile": self.execution_profile,
            "tool_mode": self.tool_mode,
            "exposed_tools": list(self.exposed_tools),
            "matched_tool_groups": list(self.matched_tool_groups),
            "fallback_reason": self.fallback_reason,
        }


@dataclass
class AgentInvocationResult:
    """Finalized invocation result."""

    content: str
    events: list[AgentEvent] = field(default_factory=list)
    diagnostics: AgentInvocationDiagnostics = field(
        default_factory=lambda: AgentInvocationDiagnostics(
            execution_profile="chat",
            tool_mode="auto",
        )
    )
