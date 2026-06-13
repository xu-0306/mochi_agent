"""Common backend message and tool payload types."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]
ResponsesContinuityMode = Literal[
    "none",
    "previous_response_id",
    "manual_encrypted",
    "manual_items",
    "degraded",
]


@dataclass
class ToolCall:
    """One backend-emitted function call."""

    id: str
    name: str
    arguments: dict[str, Any]
    index: int | None = None


@dataclass
class AttachmentRef:
    """Structured reference to one user-provided workspace attachment."""

    name: str
    path: str
    size: int | None = None
    content_type: str | None = None
    source: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    quote: str | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "size": self.size,
            "content_type": self.content_type,
            "source": self.source,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "quote": self.quote,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, value: Any) -> AttachmentRef | None:
        if not isinstance(value, dict):
            return None

        name = value.get("name")
        path = value.get("path")
        if not isinstance(name, str) or not name.strip():
            return None
        if not isinstance(path, str) or not path.strip():
            return None

        size = value.get("size")
        content_type = value.get("content_type", value.get("contentType"))
        source = value.get("source")
        line_start = value.get("line_start", value.get("lineStart"))
        line_end = value.get("line_end", value.get("lineEnd"))
        quote = value.get("quote")
        note = value.get("note")

        return cls(
            name=name.strip(),
            path=path.strip(),
            size=size if isinstance(size, int) and size >= 0 else None,
            content_type=(
                content_type.strip()
                if isinstance(content_type, str) and content_type.strip()
                else None
            ),
            source=source.strip() if isinstance(source, str) and source.strip() else None,
            line_start=line_start if isinstance(line_start, int) and line_start > 0 else None,
            line_end=line_end if isinstance(line_end, int) and line_end > 0 else None,
            quote=quote if isinstance(quote, str) and quote.strip() else None,
            note=note if isinstance(note, str) and note.strip() else None,
        )


@dataclass
class ResponsesReplayState:
    """Replay-safe Responses continuity state preserved on assistant turns."""

    response_id: str | None = None
    assistant_output_items: list[dict[str, Any]] = field(default_factory=partial(list[dict[str, Any]]))
    encrypted_reasoning_content: str | None = None
    summary_text: str = ""
    continuity_mode: ResponsesContinuityMode = "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "response_id": self.response_id,
            "assistant_output_items": self.assistant_output_items,
            "encrypted_reasoning_content": self.encrypted_reasoning_content,
            "summary_text": self.summary_text,
            "continuity_mode": self.continuity_mode,
        }

    @classmethod
    def from_dict(cls, value: Any) -> ResponsesReplayState | None:
        if not isinstance(value, dict):
            return None
        response_id = value.get("response_id")
        encrypted_reasoning_content = value.get("encrypted_reasoning_content")
        summary_text = value.get("summary_text")
        continuity_mode = value.get("continuity_mode")
        raw_assistant_output_items = value.get("assistant_output_items")
        assistant_output_items = (
            [item for item in raw_assistant_output_items if isinstance(item, dict)]
            if isinstance(raw_assistant_output_items, list)
            else []
        )
        return cls(
            response_id=response_id if isinstance(response_id, str) and response_id else None,
            assistant_output_items=assistant_output_items,
            encrypted_reasoning_content=(
                encrypted_reasoning_content
                if isinstance(encrypted_reasoning_content, str) and encrypted_reasoning_content
                else None
            ),
            summary_text=summary_text if isinstance(summary_text, str) else "",
            continuity_mode=(
                continuity_mode
                if continuity_mode in {"none", "previous_response_id", "manual_encrypted", "manual_items", "degraded"}
                else "none"
            ),
        )


@dataclass
class Message:
    """One chat message passed through a backend."""

    role: Role
    content: str
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=partial(list[ToolCall]))
    tool_call_id: str | None = None
    name: str | None = None
    attachments: list[AttachmentRef] = field(default_factory=partial(list[AttachmentRef]))
    responses_replay: ResponsesReplayState | None = None

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
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=partial(list[ToolCall]))
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    finish_reason: str = "stop"
    responses_replay: ResponsesReplayState | None = None


@dataclass
class StreamChunk:
    """One streamed chunk."""

    delta: str = ""
    thinking_delta: str = ""
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
