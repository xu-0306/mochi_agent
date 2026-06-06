"""Tool for delegating controlled subagent execution tasks from chat."""

from __future__ import annotations

from typing import Any

from mochi.runtime.delegate import get_delegate_subagent_task_launcher
from mochi.tools.base import BaseTool, ToolExecutionContext, ToolResult


class DelegateSubagentTaskTool(BaseTool):
    """Create a controlled subagent background task."""

    @property
    def name(self) -> str:
        return "delegate_subagent_task"

    @property
    def description(self) -> str:
        return (
            "Delegate a complex task to controlled subagents. Subagents may propose "
            "execution requests, but controller review and runtime policy govern any "
            "actual command execution. Returns a background task id."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "objective": {
                    "type": "string",
                    "description": "The objective for the controlled subagent task.",
                },
                "suggested_roles": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional role names or specialties to emphasize.",
                },
                "suggested_models": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                    "description": "Optional role-to-model mapping for subagents.",
                },
                "execution_budget": {
                    "type": "object",
                    "description": "Optional execution limits such as max requests or timeout.",
                    "additionalProperties": True,
                },
                "expected_artifacts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Artifacts the delegated task should try to produce.",
                },
            },
            "required": ["objective"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        *,
        objective: str,
        suggested_roles: list[str] | None = None,
        suggested_models: dict[str, str] | None = None,
        execution_budget: dict[str, Any] | None = None,
        expected_artifacts: list[str] | None = None,
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        if not objective.strip():
            return ToolResult(error="`objective` must not be empty.")
        launcher = get_delegate_subagent_task_launcher()
        if launcher is None:
            return ToolResult(
                error="Controlled subagent delegation is unavailable because runtime service is not active.",
                retryable=True,
            )

        payload = await launcher(
            objective=objective.strip(),
            session_id=context.session_id if context is not None else None,
            project_id=None,
            workspace_dir=context.workspace_dir if context is not None else None,
            suggested_roles=list(suggested_roles or []),
            suggested_models=dict(suggested_models or {}),
            execution_budget=dict(execution_budget or {}),
            expected_artifacts=list(expected_artifacts or []),
        )
        return ToolResult(
            output=payload,
            metadata={
                "status": payload.get("status"),
                "task_id": payload.get("task_id"),
                "task_type": "controlled_subagent_execution",
            },
        )
