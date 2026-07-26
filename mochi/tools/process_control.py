"""Tools for polling/stopping background processes."""

from __future__ import annotations

from typing import Any

from mochi.sessions.timeline_coordinator import mark_context_side_effect_started
from mochi.tools.base import BaseTool, ToolExecutionContext, ToolResult
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

    @property
    def supports_timeline_side_effect_boundary(self) -> bool:
        return True

    async def execute(
        self,
        *,
        process_id: str,
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        if _timeline_active(context):
            try:
                preflight = await self._process_service.poll(process_id)
            except Exception as exc:
                return ToolResult(
                    error=f"Process inspection failed: {exc}",
                    metadata={
                        "status": "timeline_pre_effect_failure",
                        "process_id": process_id,
                        "timeline_pre_effect_no_effect": True,
                    },
                )
            if preflight is None:
                return ToolResult(
                    error=f"Process not found: {process_id}",
                    metadata={
                        "status": "not_found",
                        "process_id": process_id,
                        "timeline_pre_effect_no_effect": True,
                    },
                )
            if str(preflight.get("status") or "").lower() != "running":
                return ToolResult(
                    error=(
                        f"Process is already terminal: {process_id} "
                        f"({preflight.get('status') or 'unknown'})."
                    ),
                    output=preflight,
                    metadata={
                        **preflight,
                        "timeline_pre_effect_no_effect": True,
                    },
                )

        try:
            await mark_context_side_effect_started(context)
            payload = await self._process_service.stop(process_id)
        except Exception as exc:
            return ToolResult(
                error=f"Process stop outcome is unknown: {exc}",
                metadata={
                    "status": "unknown",
                    "process_id": process_id,
                    "timeline_result_disposition": "unknown",
                },
            )
        if payload is None:
            return ToolResult(
                error=f"Process stop outcome is unknown: {process_id} disappeared.",
                metadata={
                    "status": "unknown",
                    "process_id": process_id,
                    "timeline_result_disposition": "unknown",
                },
            )
        metadata = {
            **payload,
            "timeline_result_disposition": (
                "succeeded"
                if payload.get("stopped") is True
                and str(payload.get("status") or "").lower() == "exited"
                and payload.get("stop_signal") in {"terminate", "kill"}
                else "failed"
            ),
        }
        return ToolResult(output=payload, metadata=metadata)


def _timeline_active(context: ToolExecutionContext | None) -> bool:
    return bool(
        context is not None
        and isinstance(context.state, dict)
        and context.state.get("timeline_tool_lifecycle") is not None
    )
