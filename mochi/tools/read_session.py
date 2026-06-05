"""Tool for polling exec runtime session output."""

from __future__ import annotations

from typing import Any

from mochi.runtime.exec_runtime import ExecRuntime
from mochi.tools.base import BaseTool, ToolResult
from mochi.tools.exec_command import get_shared_exec_runtime


class ReadSessionTool(BaseTool):
    """Read incremental stdout/stderr from an exec session."""

    def __init__(self, *, runtime: ExecRuntime | None = None) -> None:
        self._runtime = runtime or get_shared_exec_runtime()

    @property
    def name(self) -> str:
        return "read_session"

    @property
    def description(self) -> str:
        return "Read incremental output and status from an exec session."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Target exec session id."},
                "yield_time_ms": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Optional wait before polling output.",
                },
            },
            "required": ["session_id"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        *,
        session_id: str,
        yield_time_ms: int | None = None,
    ) -> ToolResult:
        poll = await self._runtime.read_session(session_id, yield_time_ms=yield_time_ms)
        if poll is None:
            return ToolResult(
                error=f"Session not found: {session_id}",
                metadata={"status": "not_found", "session_id": session_id},
            )
        payload = {
            "session_id": poll.session_id,
            "status": poll.status.value,
            "stdout": poll.stdout,
            "stderr": poll.stderr,
            "exit_code": poll.exit_code,
            "timed_out": poll.timed_out,
        }
        return ToolResult(
            output=payload,
            metadata={
                "status": payload["status"],
                "session_id": payload["session_id"],
                "timed_out": payload["timed_out"],
                "exit_code": payload["exit_code"],
            },
        )
