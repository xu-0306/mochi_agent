"""記憶儲存工具 — 將內容保存到長期記憶。"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from mochi.config import defaults
from mochi.tools.base import BaseTool, ToolResult
from mochi.tools.memory_guard import is_suspicious_memory_payload
from mochi.utils.security import normalize_workspace_dir, resolve_path_in_workspace


class SupportsMemorySave(Protocol):
    """記憶儲存介面。"""

    async def save(self, content: str, category: str, metadata: dict[str, Any]) -> str:
        """儲存記憶並回傳記錄 ID。"""
        ...


class JsonlMemoryStore:
    """以 JSONL 檔案實作的輕量長期記憶儲存。"""

    def __init__(self, file_path: str | Path) -> None:
        self._file_path = Path(file_path).expanduser().resolve(strict=False)

    async def save(self, content: str, category: str, metadata: dict[str, Any]) -> str:
        """儲存記憶條目。"""
        memory_id = str(uuid4())
        entry = {
            "id": memory_id,
            "content": content,
            "category": category,
            "metadata": metadata,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        await asyncio.to_thread(self._append_entry_sync, entry)
        return memory_id

    async def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """搜尋 JSONL 記憶。"""
        return await asyncio.to_thread(self._search_sync, query, top_k)

    async def update(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        category: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        return await asyncio.to_thread(
            self._update_sync,
            memory_id,
            content,
            category,
            metadata,
        )

    async def delete(self, memory_id: str) -> bool:
        return await asyncio.to_thread(self._delete_sync, memory_id)

    async def export(
        self,
        *,
        category: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._export_sync, category, limit)

    def _append_entry_sync(self, entry: dict[str, Any]) -> None:
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        with self._file_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _search_sync(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if not self._file_path.exists():
            return []

        query_lc = query.strip().lower()
        if not query_lc:
            return []

        matches: list[tuple[int, str, dict[str, Any]]] = []
        with self._file_path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue

                haystack = (
                    f"{item.get('content', '')} "
                    f"{item.get('category', '')} "
                    f"{json.dumps(item.get('metadata', {}), ensure_ascii=False)}"
                ).lower()
                score = haystack.count(query_lc)
                if score <= 0:
                    continue
                timestamp = str(item.get("timestamp", ""))
                matches.append((score, timestamp, item))

        matches.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [entry for _, _, entry in matches[:top_k]]

    def _load_entries_sync(self) -> list[dict[str, Any]]:
        if not self._file_path.exists():
            return []

        entries: list[dict[str, Any]] = []
        with self._file_path.open("r", encoding="utf-8") as file:
            for line in file:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    entries.append(parsed)
        return entries

    def _write_entries_sync(self, entries: list[dict[str, Any]]) -> None:
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        with self._file_path.open("w", encoding="utf-8") as file:
            for entry in entries:
                file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _update_sync(
        self,
        memory_id: str,
        content: str | None,
        category: str | None,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        entries = self._load_entries_sync()
        updated: dict[str, Any] | None = None
        for entry in entries:
            if str(entry.get("id")) != memory_id:
                continue
            if content is not None:
                entry["content"] = content
            if category is not None:
                entry["category"] = category
            if metadata is not None:
                entry["metadata"] = metadata
            updated = dict(entry)
            break
        if updated is None:
            return None
        self._write_entries_sync(entries)
        return updated

    def _delete_sync(self, memory_id: str) -> bool:
        entries = self._load_entries_sync()
        remaining = [entry for entry in entries if str(entry.get("id")) != memory_id]
        if len(remaining) == len(entries):
            return False
        self._write_entries_sync(remaining)
        return True

    def _export_sync(
        self,
        category: str | None,
        limit: int | None,
    ) -> list[dict[str, Any]]:
        entries = self._load_entries_sync()
        if category is not None:
            entries = [entry for entry in entries if str(entry.get("category")) == category]
        if limit is not None:
            entries = entries[:limit]
        return [dict(entry) for entry in entries]


class MemorySaveTool(BaseTool):
    """將內容保存到長期記憶的工具。"""

    def __init__(
        self,
        memory_store: SupportsMemorySave | None = None,
        *,
        workspace_dir: str | Path | None = None,
        memory_file: str = "memory/memory.jsonl",
    ) -> None:
        """初始化記憶儲存工具。

        Args:
            memory_store: 可注入的記憶儲存實作（需提供 async save）。
            workspace_dir: 預設 JSONL store 的 workspace 根路徑。
            memory_file: 預設 JSONL 相對路徑（位於 workspace 內）。
        """
        if memory_store is not None:
            self._memory_store = memory_store
            return

        workspace = normalize_workspace_dir(workspace_dir or defaults.default_workspace_dir())
        file_path = resolve_path_in_workspace(memory_file, workspace)
        self._memory_store = JsonlMemoryStore(file_path)

    @property
    def name(self) -> str:
        """工具名稱。"""
        return "memory_save"

    @property
    def description(self) -> str:
        """工具用途描述。"""
        return (
            "Save an item to long-term memory via the configured memory store. "
            "AgentEngine uses the shared SQLite memory store; standalone tool usage "
            "falls back to local JSONL."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema 格式參數。"""
        return {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Memory content."},
                "category": {
                    "type": "string",
                    "default": "general",
                    "description": "Memory category.",
                },
                "metadata": {"type": "object", "description": "Additional structured metadata."},
            },
            "required": ["content"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        *,
        content: str,
        category: str = "general",
        metadata: dict[str, Any] | None = None,
    ) -> ToolResult:
        """儲存記憶。"""
        if not content.strip():
            return ToolResult(error="`content` must not be empty.")
        if metadata is not None and not isinstance(metadata, dict):
            return ToolResult(error="`metadata` must be an object.")
        if is_suspicious_memory_payload(content=content, metadata=metadata):
            return ToolResult(error="Suspicious memory content blocked by security policy.")

        memory_id = await self._memory_store.save(
            content=content,
            category=category,
            metadata=metadata or {},
        )
        return ToolResult(
            output={"memory_id": memory_id, "category": category},
            metadata={"memory_id": memory_id},
        )
