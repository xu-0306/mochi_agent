"""Delete stored memory items."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from mochi.config import defaults
from mochi.tools.base import BaseTool, ToolResult
from mochi.tools.memory_save import JsonlMemoryStore
from mochi.utils.security import normalize_workspace_dir, resolve_path_in_workspace


class SupportsMemoryDelete(Protocol):
    """Minimal protocol for memory deletion support."""

    async def delete(self, memory_id: str) -> bool:
        ...


class MemoryDeleteTool(BaseTool):
    """Delete an existing memory entry."""

    def __init__(
        self,
        memory_store: SupportsMemoryDelete | None = None,
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
        return "memory_delete"

    @property
    def description(self) -> str:
        return "Delete a long-term memory item by id."

    @property
    def parameters_schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "Memory id to delete."},
            },
            "required": ["memory_id"],
            "additionalProperties": False,
        }

    async def execute(self, *, memory_id: str) -> ToolResult:
        if not memory_id.strip():
            return ToolResult(error="`memory_id` must not be empty.")

        deleted = await self._memory_store.delete(memory_id)
        if not deleted:
            return ToolResult(error=f"Memory not found: {memory_id}")

        return ToolResult(
            output={"memory_id": memory_id, "deleted": True},
            metadata={"memory_id": memory_id, "deleted": True},
        )
