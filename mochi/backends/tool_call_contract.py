"""Shared validation rules for tool-eligible backend turns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mochi.backends.types import GenerationResult

ToolTurnReason = Literal["tool_calls", "content", "thinking_only", "empty"]


@dataclass(frozen=True)
class ToolTurnVerdict:
    """Normalized verdict for one backend turn."""

    is_valid: bool
    reason: ToolTurnReason


def validate_tool_turn_result(
    *,
    result: GenerationResult,
    tools_requested: bool,
) -> ToolTurnVerdict:
    """Accept only visible content or structured tool calls for tool turns."""

    if result.tool_calls:
        return ToolTurnVerdict(is_valid=True, reason="tool_calls")
    if result.content.strip():
        return ToolTurnVerdict(is_valid=True, reason="content")
    if tools_requested and result.thinking.strip():
        return ToolTurnVerdict(is_valid=False, reason="thinking_only")
    return ToolTurnVerdict(is_valid=not tools_requested, reason="empty")
