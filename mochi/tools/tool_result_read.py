"""Dedicated continuation reader for persisted tool results."""

from __future__ import annotations

from typing import Any

from mochi.tools.base import ToolExecutionContext, ToolResult
from mochi.tools.file_ops import FileReadTool


class ToolResultReadTool(FileReadTool):
    """Read persisted tool-result continuations by reference id."""

    @property
    def name(self) -> str:
        return "tool_result_read"

    @property
    def description(self) -> str:
        return (
            "Read a persisted tool result continuation by reference id. "
            "Use when a previous tool result was truncated and returned a reference."
        )

    @property
    def tool_capabilities(self) -> dict[str, Any]:
        return {
            **super().tool_capabilities,
            "activation_requirements": ["tool_result_reference"],
        }

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "reference_id": {
                    "type": "string",
                    "description": "Persisted tool result reference id to continue reading.",
                },
                "encoding": {"type": "string", "description": "Optional override encoding for the resolved file."},
                "max_bytes": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Maximum bytes allowed for this read. Overrides the default limit.",
                },
                "offset": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 1,
                    "description": "Starting line number for partial reads. Uses 1-based indexing.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Maximum number of lines to return from the starting line.",
                },
                "line_numbers": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to prefix returned lines with their line number.",
                },
            },
            "required": ["reference_id"],
            "additionalProperties": False,
        }

    async def execute(  # type: ignore[override]
        self,
        *,
        reference_id: str,
        encoding: str | None = None,
        max_bytes: int | None = None,
        offset: int = 1,
        limit: int | None = None,
        line_numbers: bool = True,
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        if context is None:
            return ToolResult(error="`tool_result_read` requires an execution context.")
        if not reference_id.strip():
            return ToolResult(error="`reference_id` must not be empty.")
        reference_result = self._resolve_tool_result_reference_id(
            reference_id=reference_id,
            context=context,
        )
        if reference_result.error is not None:
            return reference_result

        result = await super().execute(
            path=f"tool-result://{reference_id}",
            encoding=encoding,
            max_bytes=max_bytes,
            offset=offset,
            limit=limit,
            line_numbers=line_numbers,
            context=context,
        )
        return self._normalize_result(reference_id=reference_id, result=result)

    @staticmethod
    def _normalize_result(reference_id: str, result: ToolResult) -> ToolResult:
        retry_call = (
            f'tool_result_read(reference_id="{reference_id}", offset=1, limit=200, line_numbers=True)'
        )
        normalized_output = result.output
        if isinstance(normalized_output, str):
            normalized_output = normalized_output.replace(
                f'file_read(path="tool-result://{reference_id}", offset=1, limit=200, line_numbers=True)',
                retry_call,
            )

        normalized_suggestion = result.suggestion
        if normalized_suggestion is not None:
            normalized_suggestion = normalized_suggestion.replace(
                f'file_read(path="tool-result://{reference_id}", offset=1, limit=200, line_numbers=True)',
                retry_call,
            )

        return ToolResult(
            output=normalized_output,
            error=result.error,
            metadata=dict(result.metadata),
            retryable=result.retryable,
            suggestion=normalized_suggestion,
        )
