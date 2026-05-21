"""Tools for polling/stopping background processes."""

from __future__ import annotations

from typing import Any

from mochi.tools.base import BaseTool, ToolResult
from mochi.tools.process_service import ProcessService


class ProcessPollTool(BaseTool):
    """Poll background process state."""

    def __init__(self, *, process_service: ProcessService) -> None:
        self._process_service = process_service

    @property
    def name(self) -> str:
        return "process_poll"

    @property
    def description(self) -> str:
        return "Poll status of a background process by process_id."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "process_id": {"type": "string", "description": "Background process id."},
            },
            "required": ["process_id"],
            "additionalProperties": False,
        }

    async def execute(self, *, process_id: str) -> ToolResult:
        payload = await self._process_service.poll(process_id)
        if payload is None:
            return ToolResult(error=f"Process not found: {process_id}")
        return ToolResult(output=payload, metadata=payload)


class ProcessStopTool(BaseTool):
    """Stop one background process."""

    def __init__(self, *, process_service: ProcessService) -> None:
        self._process_service = process_service

    @property
    def name(self) -> str:
        return "process_stop"

    @property
    def description(self) -> str:
        return "Stop a background process by process_id."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "process_id": {"type": "string", "description": "Background process id."},
            },
            "required": ["process_id"],
            "additionalProperties": False,
        }

    async def execute(self, *, process_id: str) -> ToolResult:
        payload = await self._process_service.stop(process_id)
        if payload is None:
            return ToolResult(error=f"Process not found: {process_id}")
        return ToolResult(output=payload, metadata=payload)
