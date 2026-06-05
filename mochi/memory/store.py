"""長期記憶儲存（SQLite FTS5）— Phase 2 完整實作。"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MemoryEntry = dict[str, Any]
"""單筆長期記憶資料。"""


class MemoryStore:
    """雙層記憶庫。

    短期：對話歷史（in-memory list，由 AgentEngine 管理）。
    長期：SQLite（優先 FTS5，缺失時 fallback 到 LIKE 搜尋）。
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        """建立記憶儲存器。

        Args:
            db_path: SQLite 檔案路徑；未提供時自動讀取 config.memory.db_path。
        """
        if db_path is None:
            db_path = self._resolve_default_db_path()

        self._db_path = Path(db_path).expanduser()
        self._initialized = False
        self._supports_fts5 = False
        self._init_lock = asyncio.Lock()

    def _resolve_default_db_path(self) -> Path:
        """解析預設記憶資料庫路徑。"""
        try:
            from mochi.config.manager import load_config
        except ModuleNotFoundError:
            from mochi.config.schema import MochiConfig

            return Path(MochiConfig().memory.db_path).expanduser()

        return Path(load_config().memory.db_path).expanduser()

    async def save(self, content: str, category: str, metadata: dict) -> str:
        """儲存記憶條目並回傳記憶 ID。"""
        await self._ensure_initialized()
        entry_id = uuid.uuid4().hex
        created_at = datetime.now(UTC).isoformat(timespec="seconds")

        if not isinstance(metadata, dict):
            raise TypeError("metadata must be a dict")

        await asyncio.to_thread(
            self._save_sync,
            entry_id,
            content,
            category,
            json.dumps(metadata, ensure_ascii=False, sort_keys=True),
            created_at,
        )
        return entry_id

    async def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """搜尋記憶，回傳結構化記憶條目。"""
        await self._ensure_initialized()
        if top_k <= 0:
            return []
        return await asyncio.to_thread(self._search_sync, query, top_k)

    async def update(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        category: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry | None:
        """Update an existing memory entry and return the new value."""
        await self._ensure_initialized()
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("metadata must be a dict")
        return await asyncio.to_thread(
            self._update_sync,
            memory_id,
            content,
            category,
            None if metadata is None else json.dumps(metadata, ensure_ascii=False, sort_keys=True),
        )

    async def delete(self, memory_id: str) -> bool:
        """Delete one memory entry by id."""
        await self._ensure_initialized()
        return await asyncio.to_thread(self._delete_sync, memory_id)

    async def export(
        self,
        *,
        category: str | None = None,
        limit: int | None = None,
    ) -> list[MemoryEntry]:
        """Export memory entries, optionally filtered by category."""
        await self._ensure_initialized()
        if limit is not None and limit <= 0:
            return []
        return await asyncio.to_thread(self._export_sync, category, limit)

    async def get_user_profile(self) -> str:
        """取得使用者模型（USER.md 等價內容）。"""
        await self._ensure_initialized()
        return await asyncio.to_thread(self._get_user_profile_sync)

    async def update_user_profile(self, updates: str) -> None:
        """更新使用者模型內容（追加更新文字）。"""
        await self._ensure_initialized()
        await asyncio.to_thread(self._update_user_profile_sync, updates)

    async def _ensure_initialized(self) -> None:
        """確保 SQLite schema 已初始化。"""
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return
            self._supports_fts5 = await asyncio.to_thread(self._initialize_sync)
            self._initialized = True

    def _connect(self) -> sqlite3.Connection:
        """建立 SQLite 連線。"""
        conn = sqlite3.connect(self._db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize_sync(self) -> bool:
        """同步初始化資料表，並回傳是否可使用 FTS5。"""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    category TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_created_at
                ON memories(created_at DESC)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_profile (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    content TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

            supports_fts5 = self._try_enable_fts5(conn)
            conn.commit()
            return supports_fts5
        finally:
            conn.close()

    def _create_fts_table(self, conn: sqlite3.Connection) -> None:
        """建立 FTS5 虛擬表。"""
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
            USING fts5(id UNINDEXED, content, category, metadata)
            """
        )

    def _try_enable_fts5(self, conn: sqlite3.Connection) -> bool:
        """嘗試啟用 FTS5，不支援時回傳 False。"""
        try:
            self._create_fts_table(conn)
            return True
        except sqlite3.OperationalError as exc:
            message = str(exc).lower()
            if "fts5" in message or "virtual table" in message:
                return False
            raise

    def _save_sync(
        self,
        entry_id: str,
        content: str,
        category: str,
        metadata_json: str,
        created_at: str,
    ) -> None:
        """同步寫入記憶條目。"""
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO memories (id, content, category, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (entry_id, content, category, metadata_json, created_at),
            )

            if self._supports_fts5:
                try:
                    conn.execute(
                        """
                        INSERT INTO memories_fts (id, content, category, metadata)
                        VALUES (?, ?, ?, ?)
                        """,
                        (entry_id, content, category, metadata_json),
                    )
                except sqlite3.OperationalError:
                    self._supports_fts5 = False

            conn.commit()
        finally:
            conn.close()

    def _search_sync(self, query: str, top_k: int) -> list[MemoryEntry]:
        """同步搜尋記憶（優先 FTS5，失敗 fallback）。"""
        query = query.strip()
        if not query:
            return []

        conn = self._connect()
        try:
            if self._supports_fts5:
                try:
                    rows = conn.execute(
                        """
                        SELECT
                            m.id,
                            m.content,
                            m.category,
                            m.metadata_json,
                            m.created_at,
                            bm25(memories_fts) AS score
                        FROM memories_fts
                        JOIN memories AS m ON m.id = memories_fts.id
                        WHERE memories_fts MATCH ?
                        ORDER BY score ASC
                        LIMIT ?
                        """,
                        (self._to_fts_query(query), top_k),
                    ).fetchall()
                    if rows:
                        return [self._row_to_memory_entry(row, with_score=True) for row in rows]
                except sqlite3.OperationalError:
                    self._supports_fts5 = False

            like = f"%{query}%"
            rows = conn.execute(
                """
                SELECT
                    id,
                    content,
                    category,
                    metadata_json,
                    created_at
                FROM memories
                WHERE content LIKE ? COLLATE NOCASE
                   OR category LIKE ? COLLATE NOCASE
                   OR metadata_json LIKE ? COLLATE NOCASE
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (like, like, like, top_k),
            ).fetchall()
            return [self._row_to_memory_entry(row, with_score=False) for row in rows]
        finally:
            conn.close()

    def _update_sync(
        self,
        memory_id: str,
        content: str | None,
        category: str | None,
        metadata_json: str | None,
    ) -> MemoryEntry | None:
        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT id, content, category, metadata_json, created_at
                FROM memories
                WHERE id = ?
                """,
                (memory_id,),
            ).fetchone()
            if row is None:
                return None

            next_content = str(row["content"]) if content is None else content
            next_category = str(row["category"]) if category is None else category
            next_metadata_json = str(row["metadata_json"]) if metadata_json is None else metadata_json

            conn.execute(
                """
                UPDATE memories
                SET content = ?, category = ?, metadata_json = ?
                WHERE id = ?
                """,
                (next_content, next_category, next_metadata_json, memory_id),
            )

            if self._supports_fts5:
                try:
                    conn.execute("DELETE FROM memories_fts WHERE id = ?", (memory_id,))
                    conn.execute(
                        """
                        INSERT INTO memories_fts (id, content, category, metadata)
                        VALUES (?, ?, ?, ?)
                        """,
                        (memory_id, next_content, next_category, next_metadata_json),
                    )
                except sqlite3.OperationalError:
                    self._supports_fts5 = False

            conn.commit()
            return self._row_to_memory_entry(
                {
                    "id": memory_id,
                    "content": next_content,
                    "category": next_category,
                    "metadata_json": next_metadata_json,
                    "created_at": str(row["created_at"]),
                },
                with_score=False,
            )
        finally:
            conn.close()

    def _delete_sync(self, memory_id: str) -> bool:
        conn = self._connect()
        try:
            if self._supports_fts5:
                try:
                    conn.execute("DELETE FROM memories_fts WHERE id = ?", (memory_id,))
                except sqlite3.OperationalError:
                    self._supports_fts5 = False
            cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def _export_sync(
        self,
        category: str | None,
        limit: int | None,
    ) -> list[MemoryEntry]:
        conn = self._connect()
        try:
            query = (
                "SELECT id, content, category, metadata_json, created_at "
                "FROM memories"
            )
            params: list[Any] = []
            if category is not None:
                query += " WHERE category = ?"
                params.append(category)
            query += " ORDER BY created_at DESC"
            if limit is not None:
                query += " LIMIT ?"
                params.append(limit)
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_memory_entry(row, with_score=False) for row in rows]
        finally:
            conn.close()

    def _get_user_profile_sync(self) -> str:
        """同步讀取使用者模型內容。"""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT content FROM user_profile WHERE id = 1"
            ).fetchone()
            if row is None:
                return ""
            return str(row["content"])
        finally:
            conn.close()

    def _update_user_profile_sync(self, updates: str) -> None:
        """同步更新使用者模型內容。"""
        updates = updates.strip()
        if not updates:
            return

        now = datetime.now(UTC).isoformat(timespec="seconds")
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT content FROM user_profile WHERE id = 1"
            ).fetchone()
            if row is None:
                next_content = updates
            else:
                existing = str(row["content"]).strip()
                next_content = updates if not existing else f"{existing}\n{updates}"

            conn.execute(
                """
                INSERT INTO user_profile (id, content, updated_at)
                VALUES (1, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    content = excluded.content,
                    updated_at = excluded.updated_at
                """,
                (next_content, now),
            )
            conn.commit()
        finally:
            conn.close()

    def _to_fts_query(self, raw_query: str) -> str:
        """將一般文字查詢轉成較安全的 FTS 查詢語法。"""
        tokens = [token.strip() for token in raw_query.split() if token.strip()]
        if not tokens:
            escaped = raw_query.replace('"', '""').strip()
            return f'"{escaped}"' if escaped else ""
        escaped_tokens = [f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens]
        return " ".join(escaped_tokens)

    def _row_to_memory_entry(self, row: sqlite3.Row, with_score: bool) -> MemoryEntry:
        """將 SQLite row 轉為結構化記憶物件。"""
        metadata: dict[str, Any] = {}
        metadata_raw = row["metadata_json"]
        if isinstance(metadata_raw, str):
            try:
                parsed = json.loads(metadata_raw)
                if isinstance(parsed, dict):
                    metadata = parsed
            except json.JSONDecodeError:
                metadata = {}

        entry: MemoryEntry = {
            "id": str(row["id"]),
            "content": str(row["content"]),
            "category": str(row["category"]),
            "metadata": metadata,
            "created_at": str(row["created_at"]),
        }
        if with_score:
            score = row["score"]
            if score is not None:
                entry["score"] = float(score)
        return entry
