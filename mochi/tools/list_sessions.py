"""Tool for listing in-memory exec runtime sessions."""

from __future__ import annotations

from typing import Any

from mochi.runtime.exec_runtime import ExecRuntime
from mochi.tools.base import BaseTool, ToolResult
from mochi.tools.exec_command import get_shared_exec_runtime


class ListSessionsTool(BaseTool):
    """List active and completed exec runtime sessions."""

    def __init__(self, *, runtime: ExecRuntime | None = None) -> None:
        self._runtime = runtime or get_shared_exec_runtime()

    @property
    def name(self) -> str:
        return "list_sessions"

    @property
    def description(self) -> str:
        return "List known exec sessions and their current lifecycle state."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }

    @property
    def is_read_only(self) -> bool:
        return True

    async def execute(self) -> ToolResult:
        sessions = self._runtime.list_sessions()
        payload = [
            {
                "session_id": session.session_id,
                "status": session.status.value,
                "shell": session.shell,
                "command": session.command,
                "cwd": session.cwd,
                "pid": session.pid,
                "background": session.background,
                "tty": session.tty,
                "exit_code": session.exit_code,
                "timed_out": session.timed_out,
                "approval_state": session.approval_state,
            }
            for session in sessions
        ]
        return ToolResult(
            output=payload,
            metadata={"status": "ok", "count": len(payload)},
        )
