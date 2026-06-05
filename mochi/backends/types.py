"""Common backend message and tool payload types."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCall:
    """One backend-emitted function call."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class AttachmentRef:
    """Structured reference to one user-provided workspace attachment."""

    name: str
    path: str
    size: int | None = None
    content_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "size": self.size,
            "content_type": self.content_type,
        }


@dataclass
class Message:
    """One chat message passed through a backend."""

    role: Role
    content: str
    tool_calls: list[ToolCall] = field(default_factory=partial(list[ToolCall]))
    tool_call_id: str | None = None
    name: str | None = None
    attachments: list[AttachmentRef] = field(default_factory=partial(list[AttachmentRef]))

    def to_dict(self) -> dict[str, Any]:
        """Serialize into an OpenAI-compatible chat message payload."""
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.dumps(tool_call.arguments, ensure_ascii=False),
                    },
                }
                for tool_call in self.tool_calls
            ]
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        if self.name:
            payload["name"] = self.name
        return payload


@dataclass
class ToolSchema:
    """One callable tool schema."""

    name: str
    description: str
    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class GenerationResult:
    """Non-stream generation output."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=partial(list[ToolCall]))
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    finish_reason: str = "stop"


@dataclass
class StreamChunk:
    """One streamed chunk."""

    delta: str = ""
    tool_call_delta: ToolCall | None = None
    is_final: bool = False
    finish_reason: str | None = None


@dataclass
class ModelInfo:
    """Basic model metadata exposed by a backend."""

    name: str
    backend_type: str
    provider: str | None = None
    context_length: int = 4096
    supports_tool_calling: bool = False
    metadata: dict[str, Any] = field(default_factory=partial(dict[str, Any]))
