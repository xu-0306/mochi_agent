"""Update stored memory items."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from mochi.config import defaults
from mochi.tools.base import BaseTool, ToolResult
from mochi.tools.memory_guard import is_suspicious_memory_payload
from mochi.tools.memory_save import JsonlMemoryStore
from mochi.utils.security import normalize_workspace_dir, resolve_path_in_workspace


class SupportsMemoryUpdate(Protocol):
    """Minimal protocol for memory update support."""

    async def update(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        category: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        ...


class MemoryUpdateTool(BaseTool):
    """Update an existing memory entry."""

    def __init__(
        self,
        memory_store: SupportsMemoryUpdate | None = None,
        *,
        workspace_dir: str | Path | None = None,
        memory_file: str = "memory/memory.jsonl",
    ) -> None:
        if memory_store is not None:
            self._memory_store = memory_store
            return

        workspace = normalize_workspace_dir(workspace_dir or defaults.default_workspace_dir())
        file_path = resolve_path_in_workspace(memory_file, workspace)
        self._memory_store = JsonlMemoryStore(file_path)

    @property
    def name(self) -> str:
        return "memory_update"

    @property
    def description(self) -> str:
        return "Update an existing long-term memory item by id."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "Memory id to update."},
                "content": {"type": "string", "description": "Replacement memory content."},
                "category": {"type": "string", "description": "Replacement category."},
                "metadata": {"type": "object", "description": "Replacement structured metadata."},
            },
            "required": ["memory_id"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        *,
        memory_id: str,
        content: str | None = None,
        category: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ToolResult:
        if not memory_id.strip():
            return ToolResult(error="`memory_id` must not be empty.")
        if content is None and category is None and metadata is None:
            return ToolResult(error="Provide at least one field to update.")
        if metadata is not None and not isinstance(metadata, dict):
            return ToolResult(error="`metadata` must be an object.")
        if is_suspicious_memory_payload(content=content, metadata=metadata):
            return ToolResult(error="Suspicious memory content blocked by security policy.")

        updated = await self._memory_store.update(
            memory_id,
            content=content,
            category=category,
            metadata=metadata,
        )
        if updated is None:
            return ToolResult(error=f"Memory not found: {memory_id}")

        return ToolResult(output=updated, metadata={"memory_id": memory_id})
