"""Export memory entries from the configured store."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from mochi.config import defaults
from mochi.tools.base import BaseTool, ToolResult
from mochi.tools.memory_save import JsonlMemoryStore
from mochi.utils.security import normalize_workspace_dir, resolve_path_in_workspace


class SupportsMemoryExport(Protocol):
    """Minimal protocol for memory export support."""

    async def export(
        self,
        *,
        category: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        ...


class MemoryExportTool(BaseTool):
    """Export memory entries as structured records."""

    def __init__(
        self,
        memory_store: SupportsMemoryExport | None = None,
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
        return "memory_export"

    @property
    def description(self) -> str:
        return "Export long-term memory entries, optionally filtered by category."

    @property
    def parameters_schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "Optional category filter."},
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10000,
                    "description": "Optional maximum number of entries to export.",
                },
            },
            "additionalProperties": False,
        }

    async def execute(
        self,
        *,
        category: str | None = None,
        limit: int | None = None,
    ) -> ToolResult:
        if limit is not None and limit <= 0:
            return ToolResult(error="`limit` must be greater than 0.")

        entries = await self._memory_store.export(category=category, limit=limit)
        return ToolResult(
            output=entries,
            metadata={"count": len(entries), "category": category, "limit": limit},
        )
