"""Tool for writing stdin to a running exec session."""

from __future__ import annotations

from typing import Any

from mochi.runtime.exec_sessions import ExecSessionStatus, SessionPollResult
from mochi.sessions.timeline_coordinator import mark_context_side_effect_started
from mochi.tools.base import BaseTool, ToolExecutionContext, ToolResult
from mochi.tools.exec_command import get_shared_exec_runtime
from mochi.runtime.exec_runtime import ExecRuntime


class WriteStdinTool(BaseTool):
    """Write characters to an existing exec runtime session."""

    def __init__(self, *, runtime: ExecRuntime | None = None) -> None:
        self._runtime = runtime or get_shared_exec_runtime()

    @property
    def name(self) -> str:
        return "write_stdin"

    @property
    def description(self) -> str:
        return "Write text to stdin of an active exec session and return incremental output."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Target exec session id."},
                "chars": {"type": "string", "description": "Characters to write to stdin."},
                "yield_time_ms": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Optional wait before polling output.",
                },
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
        chars: str = "",
        yield_time_ms: int | None = None,
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        timeline_active = _timeline_active(context)
        if timeline_active:
            if not chars:
                return ToolResult(
                    error="timeline_no_stdin_payload: no stdin effect can be dispatched without chars.",
                    metadata={
                        "status": "timeline_pre_effect_no_effect",
                        "session_id": session_id,
                        "timeline_pre_effect_no_effect": True,
                    },
                )
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
            if (
                preflight.status is not ExecSessionStatus.RUNNING
                or not preflight.supports_stdin
            ):
                return ToolResult(
                    error=(
                        f"Session is not accepting stdin: {session_id} "
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
            poll = await self._runtime.write_stdin(
                session_id,
                chars=chars,
                yield_time_ms=yield_time_ms,
            )
        except Exception as exc:
            return ToolResult(
                error=f"Session stdin write outcome is unknown: {exc}",
                metadata={
                    "status": "unknown",
                    "session_id": session_id,
                    "timeline_result_disposition": "unknown",
                },
            )
        if poll is None:
            return ToolResult(
                error=f"Session stdin write outcome is unknown: {session_id} disappeared.",
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
                "timeline_result_disposition": _write_disposition(poll.status),
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


def _write_disposition(status: ExecSessionStatus) -> str:
    if status in {ExecSessionStatus.RUNNING, ExecSessionStatus.COMPLETED}:
        return "succeeded"
    return "failed"
