"""Core tool contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
from typing import Any


@dataclass
class ToolResult:
    """Normalized tool result envelope."""

    output: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    retryable: bool = False
    suggestion: str | None = None


@dataclass
class FileReadState:
    """Cached file read snapshot used by write guards."""

    path: str
    content: str
    encoding: str
    mtime_ns: int | None
    size_bytes: int
    partial: bool = False


@dataclass
class ToolExecutionContext:
    """Shared execution context for one tool run."""

    workspace_dir: str | None = None
    session_id: str | None = None
    project_workspace: str | None = None
    task_sandbox_dir: str | None = None
    permission_policy: dict[str, Any] = field(default_factory=dict)
    read_state_cache: dict[str, FileReadState] = field(default_factory=dict)
    tool_result_store_dir: str | None = None
    tool_result_references: dict[str, str] = field(default_factory=dict)
    transport_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    progress_callback: Any | None = None
    cancellation_requested: bool = False


class BaseTool(ABC):
    """Base class for every tool exposed to the agent."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable tool name used by model tool calling."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """LLM-facing tool description."""
        ...

    @property
    @abstractmethod
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema describing tool arguments."""
        ...

    @property
    def requires_approval(self) -> bool:
        return False

    @property
    def is_read_only(self) -> bool:
        return False

    @property
    def is_destructive(self) -> bool:
        return False

    @property
    def is_concurrency_safe(self) -> bool:
        return False

    @property
    def is_open_world(self) -> bool:
        return False

    @property
    def search_hint(self) -> str | None:
        return None

    def validate_input(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> ToolResult | None:
        del arguments, context
        return None

    def check_permissions(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> ToolResult | None:
        del arguments, context
        return None

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool and return a ToolResult."""
        ...

    def format_result_for_model(
        self,
        result: ToolResult,
        *,
        max_chars: int = 2000,
    ) -> str:
        """Serialize tool output for model reinjection."""
        if result.error:
            payload: dict[str, Any] = {"ok": False, "error": result.error}
        else:
            payload = {"ok": True, "output": result.output}

        serialized = json.dumps(payload, ensure_ascii=False, default=str)
        if len(serialized) <= max_chars:
            return serialized

        if result.error:
            compact_payload = {
                "ok": False,
                "error": self._truncate_text(result.error, max_chars // 2),
                "truncated": True,
                "original_length": len(result.error),
            }
        elif isinstance(result.output, str):
            compact_payload = {
                "ok": True,
                "output": self._truncate_text(result.output, max_chars // 2),
                "truncated": True,
                "original_length": len(result.output),
            }
        else:
            compact_payload = {
                "ok": True,
                "output_preview": self._truncate_text(serialized, max_chars // 2),
                "truncated": True,
                "original_length": len(serialized),
            }
        return json.dumps(compact_payload, ensure_ascii=False, default=str)

    def summarize_result_for_ui(self, result: ToolResult) -> str:
        """Return a short UI summary."""
        if result.error:
            return result.error
        if isinstance(result.output, str):
            return result.output
        return json.dumps(result.output, ensure_ascii=False, default=str)

    def to_schema_dict(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }

    @staticmethod
    def _truncate_text(value: str, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        if len(value) <= max_chars:
            return value
        suffix = "...[truncated]"
        if max_chars <= len(suffix):
            return suffix[:max_chars]
        return value[: max_chars - len(suffix)] + suffix
