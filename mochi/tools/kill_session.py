"""Tool for terminating exec runtime sessions."""

from __future__ import annotations

from typing import Any

from mochi.runtime.exec_runtime import ExecRuntime
from mochi.runtime.exec_sessions import ExecSessionStatus, SessionPollResult
from mochi.sessions.timeline_coordinator import mark_context_side_effect_started
from mochi.tools.base import BaseTool, ToolExecutionContext, ToolResult
from mochi.tools.exec_command import get_shared_exec_runtime


class KillSessionTool(BaseTool):
    """Terminate an exec session by id."""

    def __init__(self, *, runtime: ExecRuntime | None = None) -> None:
        self._runtime = runtime or get_shared_exec_runtime()

    @property
    def name(self) -> str:
        return "kill_session"

    @property
    def description(self) -> str:
        return "Terminate an exec session and return final status/output deltas."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Target exec session id."},
            },
            "required": ["session_id"],
            "additionalProperties": False,
        }

    @property
    def supports_timeline_side_effect_boundary(self) -> bool:
        return True

    async def execute(
        self,
        *,
        session_id: str,
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        if _timeline_active(context):
            try:
                preflight = await self._runtime.inspect_session(session_id)
            except Exception as exc:
                return ToolResult(
                    error=f"Session inspection failed: {exc}",
                    metadata={
                        "status": "timeline_pre_effect_failure",
                        "session_id": session_id,
                        "timeline_pre_effect_no_effect": True,
                    },
                )
            if preflight is None:
                return ToolResult(
                    error=f"Session not found: {session_id}",
                    metadata={
                        "status": "not_found",
                        "session_id": session_id,
                        "timeline_pre_effect_no_effect": True,
                    },
                )
            if preflight.status is not ExecSessionStatus.RUNNING:
                return ToolResult(
                    error=(
                        f"Session is already terminal: {session_id} "
                        f"({preflight.status.value})."
                    ),
                    output=_payload(preflight),
                    metadata={
                        "status": preflight.status.value,
                        "session_id": preflight.session_id,
                        "timeline_pre_effect_no_effect": True,
                    },
                )

        try:
            await mark_context_side_effect_started(context)
            poll = await self._runtime.kill_session(session_id)
        except Exception as exc:
            return ToolResult(
                error=f"Session termination outcome is unknown: {exc}",
                metadata={
                    "status": "unknown",
                    "session_id": session_id,
                    "timeline_result_disposition": "unknown",
                },
            )
        if poll is None:
            return ToolResult(
                error=f"Session termination outcome is unknown: {session_id} disappeared.",
                metadata={
                    "status": "unknown",
                    "session_id": session_id,
                    "timeline_result_disposition": "unknown",
                },
            )
        payload = _payload(poll)
        return ToolResult(
            output=payload,
            metadata={
                "status": payload["status"],
                "session_id": payload["session_id"],
                "timed_out": payload["timed_out"],
                "exit_code": payload["exit_code"],
                "timeline_result_disposition": _kill_disposition(poll.status),
            },
        )


def _timeline_active(context: ToolExecutionContext | None) -> bool:
    return bool(
        context is not None
        and isinstance(context.state, dict)
        and context.state.get("timeline_tool_lifecycle") is not None
    )


def _payload(poll: SessionPollResult) -> dict[str, Any]:
    return {
        "session_id": poll.session_id,
        "status": poll.status.value,
        "stdout": poll.stdout,
        "stderr": poll.stderr,
        "exit_code": poll.exit_code,
        "timed_out": poll.timed_out,
    }


def _kill_disposition(status: ExecSessionStatus) -> str:
    return "succeeded" if status is ExecSessionStatus.KILLED else "failed"
