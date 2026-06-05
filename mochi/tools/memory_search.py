"""記憶搜尋工具 — 從長期記憶取回相關內容。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from mochi.config import defaults
from mochi.tools.base import BaseTool, ToolResult
from mochi.tools.memory_save import JsonlMemoryStore
from mochi.utils.security import normalize_workspace_dir, resolve_path_in_workspace


class SupportsMemorySearch(Protocol):
    """記憶搜尋介面。"""

    async def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """搜尋相關記憶。"""
        ...


class MemorySearchTool(BaseTool):
    """搜尋長期記憶的工具。"""

    def __init__(
        self,
        memory_store: SupportsMemorySearch | None = None,
        *,
        workspace_dir: str | Path | None = None,
        memory_file: str = "memory/memory.jsonl",
        default_top_k: int = 5,
    ) -> None:
        """初始化記憶搜尋工具。

        Args:
            memory_store: 可注入記憶搜尋實作（需提供 async search）。
            workspace_dir: 預設 JSONL store 的 workspace 根路徑。
            memory_file: 預設 JSONL 相對路徑（位於 workspace 內）。
            default_top_k: 預設回傳筆數。
        """
        if memory_store is not None:
            self._memory_store = memory_store
        else:
            workspace = normalize_workspace_dir(workspace_dir or defaults.default_workspace_dir())
            file_path = resolve_path_in_workspace(memory_file, workspace)
            self._memory_store = JsonlMemoryStore(file_path)
        self._default_top_k = default_top_k

    @property
    def name(self) -> str:
        """工具名稱。"""
        return "memory_search"

    @property
    def description(self) -> str:
        """工具用途描述。"""
        return (
            "Search long-term memory via the configured memory store. AgentEngine uses "
            "the shared SQLite memory store with FTS5 or LIKE fallback; standalone tool "
            "usage falls back to local JSONL."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema 格式參數。"""
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 5,
                    "description": "Maximum number of results to return.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        }

    async def execute(self, *, query: str, top_k: int | None = None) -> ToolResult:
        """執行記憶搜尋。"""
        if not query.strip():
            return ToolResult(error="`query` must not be empty.")

        effective_top_k = top_k if top_k is not None else self._default_top_k
        if effective_top_k <= 0:
            return ToolResult(error="`top_k` must be greater than 0.")

        results = await self._memory_store.search(query=query, top_k=effective_top_k)
        return ToolResult(
            output=results,
            metadata={"count": len(results), "top_k": effective_top_k},
        )
