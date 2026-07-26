"""Runtime-gated activation broker for discoverable tools."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mochi.tools.base import BaseTool, ToolExecutionContext, ToolResult

ActivationRequester = Callable[[str, ToolExecutionContext | None], ToolResult]


class ToolActivateTool(BaseTool):
    """Expose one safe, explicit path from discovery to registry activation."""

    def __init__(self, *, request_activation: ActivationRequester) -> None:
        self._request_activation = request_activation

    @property
    def name(self) -> str:
        return "tool_activate"

    @property
    def description(self) -> str:
        return (
            "Request policy-gated activation of one discoverable tool for this turn. "
            "Activation only makes the tool callable; it does not authorize the tool's "
            "concrete operation, which still runs through normal approval and security checks."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "tool_name": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Exact discoverable tool name to activate.",
                }
            },
            "required": ["tool_name"],
            "additionalProperties": False,
        }

    @property
    def is_read_only(self) -> bool:
        return True

    async def execute(self, **kwargs: Any) -> ToolResult:
        requested_tool = str(kwargs.get("tool_name") or "").strip()
        if not requested_tool:
            return ToolResult(
                error="`tool_name` must not be empty.",
                metadata={
                    "runtime_category": "tool_activation",
                    "error_type": "tool_activation_invalid_request",
                },
                retryable=False,
            )

        raw_context = kwargs.get("context")
        if raw_context is not None and not isinstance(raw_context, ToolExecutionContext):
            return ToolResult(
                error="`context` must be a ToolExecutionContext when provided.",
                metadata={
                    "runtime_category": "tool_activation",
                    "error_type": "tool_activation_invalid_request",
                    "requested_tool": requested_tool,
                },
                retryable=False,
            )
        context = raw_context
        result = self._request_activation(requested_tool, context)
        if result.error is not None or result.output is not None:
            return result

        metadata = dict(result.metadata)
        output = {
            "status": metadata.get("status"),
            "requested_tool": metadata.get("requested_tool", requested_tool),
            "callable_this_turn": bool(metadata.get("callable_this_turn")),
            "activation_authorizes_tool_call": bool(
                metadata.get("activation_authorizes_tool_call", False)
            ),
            "next_step": (
                f"Call {requested_tool} with the concrete operation arguments. "
                "The call will still enforce approval and security policy."
            ),
        }
        return ToolResult(
            output=output,
            metadata=metadata,
            retryable=result.retryable,
            suggestion=result.suggestion,
        )
