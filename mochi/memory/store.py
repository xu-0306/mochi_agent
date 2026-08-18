"""Durable memory storage with revisioned canonical records and derived FTS."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MemoryEntry = dict[str, Any]


class MemoryStore:
    """SQLite-backed memory store whose canonical rows survive index rebuilds."""

    _BACKUP_SCHEMA_VERSION = 2
    _RECORD_FIELDS = (
        "id",
        "content",
        "category",
        "metadata",
        "namespace",
        "provenance",
        "revision",
        "supersedes_id",
        "created_at",
        "updated_at",
    )

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_path = self._resolve_default_db_path()

        self._db_path = Path(db_path).expanduser()
        self._initialized = False
        self._supports_fts5 = False
        self._init_lock = asyncio.Lock()

    def _resolve_default_db_path(self) -> Path:
        try:
            from mochi.config.manager import load_config
        except ModuleNotFoundError:
            from mochi.config.schema import MochiConfig

            return Path(MochiConfig().memory.db_path).expanduser()

        return Path(load_config().memory.db_path).expanduser()

    async def save(
        self,
        content: str,
        category: str,
        metadata: dict[str, Any],
        *,
        namespace: str = "default",
        provenance: dict[str, Any] | None = None,
        supersedes_id: str | None = None,
    ) -> str:
        """Store a new canonical memory record and return its stable id."""

        await self._ensure_initialized()
        if not isinstance(metadata, dict):
            raise TypeError("metadata must be a dict")
        if provenance is not None and not isinstance(provenance, dict):
            raise TypeError("provenance must be a dict")
        namespace = self._validate_namespace(namespace)
        if supersedes_id is not None and not isinstance(supersedes_id, str):
            raise TypeError("supersedes_id must be a string or None")

        now = self._now()
        entry = self._new_entry(
            memory_id=uuid.uuid4().hex,
            content=content,
            category=category,
            metadata=metadata,
            namespace=namespace,
            provenance=provenance or {},
            revision=1,
            supersedes_id=supersedes_id,
            created_at=now,
            updated_at=now,
        )
        await asyncio.to_thread(self._save_sync, entry)
        return str(entry["id"])

    async def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
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
        """Append a revision for an active memory while keeping its public id."""

        await self._ensure_initialized()
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("metadata must be a dict")
        return await asyncio.to_thread(
            self._update_sync,
            memory_id,
            content,
            category,
            metadata,
        )

    async def delete(self, memory_id: str) -> bool:
        """Remove the active projection while retaining immutable revision history."""

        await self._ensure_initialized()
        return await asyncio.to_thread(self._delete_sync, memory_id)

    async def export(
        self,
        *,
        category: str | None = None,
        limit: int | None = None,
    ) -> list[MemoryEntry]:
        await self._ensure_initialized()
        if limit is not None and limit <= 0:
            return []
        return await asyncio.to_thread(self._export_sync, category, limit)

    async def get_history(self, memory_id: str) -> list[MemoryEntry]:
        """Return retained snapshots in revision order for one memory id."""

        await self._ensure_initialized()
        return await asyncio.to_thread(self._get_history_sync, memory_id)

    async def backup(self) -> dict[str, Any]:
        """Return a fully validated, JSON-serializable canonical backup."""

        await self._ensure_initialized()
        return await asyncio.to_thread(self._backup_sync)

    async def restore(self, backup: Mapping[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
        """Validate a backup before atomically replacing active canonical state."""

        await self._ensure_initialized()
        if not isinstance(backup, Mapping):
            raise TypeError("backup must be a mapping")
        return await asyncio.to_thread(self._restore_sync, backup, dry_run)

    async def get_user_profile(self) -> str:
        await self._ensure_initialized()
        return await asyncio.to_thread(self._get_user_profile_sync)

    async def update_user_profile(self, updates: str) -> None:
        await self._ensure_initialized()
        await asyncio.to_thread(self._update_user_profile_sync, updates)

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return
            self._supports_fts5 = await asyncio.to_thread(self._initialize_sync)
            self._initialized = True

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize_sync(self) -> bool:
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
                    created_at TEXT NOT NULL,
                    namespace TEXT NOT NULL DEFAULT 'default',
                    provenance_json TEXT NOT NULL DEFAULT '{}',
                    revision INTEGER NOT NULL DEFAULT 1,
                    supersedes_id TEXT,
                    updated_at TEXT,
                    checksum TEXT
                )
                """
            )
            self._ensure_memory_columns(conn)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_revisions (
                    memory_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    record_json TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    PRIMARY KEY (memory_id, revision)
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
            self._migrate_memory_rows(conn)
            supports_fts5 = self._try_enable_fts5(conn)
            if supports_fts5:
                self._rebuild_fts_sync(conn)
            conn.commit()
            return supports_fts5
        finally:
            conn.close()

    def _ensure_memory_columns(self, conn: sqlite3.Connection) -> None:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(memories)")}
        migrations = {
            "namespace": "ALTER TABLE memories ADD COLUMN namespace TEXT NOT NULL DEFAULT 'default'",
            "provenance_json": "ALTER TABLE memories ADD COLUMN provenance_json TEXT NOT NULL DEFAULT '{}'",
            "revision": "ALTER TABLE memories ADD COLUMN revision INTEGER NOT NULL DEFAULT 1",
            "supersedes_id": "ALTER TABLE memories ADD COLUMN supersedes_id TEXT",
            "updated_at": "ALTER TABLE memories ADD COLUMN updated_at TEXT",
            "checksum": "ALTER TABLE memories ADD COLUMN checksum TEXT",
        }
        for column, statement in migrations.items():
            if column not in columns:
                conn.execute(statement)

    def _migrate_memory_rows(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute("SELECT * FROM memories").fetchall()
        for row in rows:
            entry = self._row_to_memory_entry(row, with_score=False)
            conn.execute(
                """
                UPDATE memories
                SET namespace = ?, provenance_json = ?, revision = ?, supersedes_id = ?,
                    updated_at = ?, checksum = ?
                WHERE id = ?
                """,
                (
                    entry["namespace"],
                    self._json(entry["provenance"]),
                    entry["revision"],
                    entry["supersedes_id"],
                    entry["updated_at"],
                    entry["checksum"],
                    entry["id"],
                ),
            )
            existing = conn.execute(
                """
                SELECT 1 FROM memory_revisions
                WHERE memory_id = ? AND revision = ?
                """,
                (entry["id"], entry["revision"]),
            ).fetchone()
            if existing is None:
                self._insert_history_row(conn, entry)

    def _create_fts_table(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
            USING fts5(id UNINDEXED, content, category, metadata)
            """
        )

    def _try_enable_fts5(self, conn: sqlite3.Connection) -> bool:
        try:
            self._create_fts_table(conn)
            return True
        except sqlite3.OperationalError as exc:
            message = str(exc).lower()
            if "fts5" in message or "virtual table" in message:
                return False
            raise

    def _rebuild_fts_sync(self, conn: sqlite3.Connection, *, fail_closed: bool = False) -> None:
        try:
            conn.execute("DELETE FROM memories_fts")
            rows = conn.execute("SELECT * FROM memories").fetchall()
            for row in rows:
                entry = self._row_to_memory_entry(row, with_score=False)
                conn.execute(
                    """
                    INSERT INTO memories_fts (id, content, category, metadata)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        entry["id"],
                        entry["content"],
                        entry["category"],
                        self._json(entry["metadata"]),
                    ),
                )
        except sqlite3.OperationalError:
            if fail_closed:
                raise
            self._supports_fts5 = False

    def _save_sync(self, entry: MemoryEntry) -> None:
        conn = self._connect()
        try:
            with conn:
                self._insert_memory_row(conn, entry)
                self._insert_history_row(conn, entry)
                self._upsert_fts_row(conn, entry)
        finally:
            conn.close()

    def _insert_memory_row(self, conn: sqlite3.Connection, entry: MemoryEntry) -> None:
        conn.execute(
            """
            INSERT INTO memories (
                id, content, category, metadata_json, created_at, namespace,
                provenance_json, revision, supersedes_id, updated_at, checksum
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry["id"],
                entry["content"],
                entry["category"],
                self._json(entry["metadata"]),
                entry["created_at"],
                entry["namespace"],
                self._json(entry["provenance"]),
                entry["revision"],
                entry["supersedes_id"],
                entry["updated_at"],
                entry["checksum"],
            ),
        )

    def _upsert_fts_row(self, conn: sqlite3.Connection, entry: MemoryEntry) -> None:
        if not self._supports_fts5:
            return
        try:
            conn.execute("DELETE FROM memories_fts WHERE id = ?", (entry["id"],))
            conn.execute(
                """
                INSERT INTO memories_fts (id, content, category, metadata)
                VALUES (?, ?, ?, ?)
                """,
                (entry["id"], entry["content"], entry["category"], self._json(entry["metadata"])),
            )
        except sqlite3.OperationalError:
            self._supports_fts5 = False

    def _search_sync(self, query: str, top_k: int) -> list[MemoryEntry]:
        query = query.strip()
        if not query:
            return []

        conn = self._connect()
        try:
            if self._supports_fts5:
                try:
                    rows = conn.execute(
                        """
                        SELECT m.*, bm25(memories_fts) AS score
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
                SELECT * FROM memories
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
        metadata: dict[str, Any] | None,
    ) -> MemoryEntry | None:
        conn = self._connect()
        try:
            with conn:
                row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
                if row is None:
                    return None
                current = self._row_to_memory_entry(row, with_score=False)
                next_entry = self._new_entry(
                    memory_id=memory_id,
                    content=current["content"] if content is None else content,
                    category=current["category"] if category is None else category,
                    metadata=current["metadata"] if metadata is None else metadata,
                    namespace=current["namespace"],
                    provenance=current["provenance"],
                    revision=int(current["revision"]) + 1,
                    supersedes_id=current["supersedes_id"],
                    created_at=current["created_at"],
                    updated_at=self._now(),
                )
                conn.execute(
                    """
                    UPDATE memories
                    SET content = ?, category = ?, metadata_json = ?, revision = ?,
                        updated_at = ?, checksum = ?
                    WHERE id = ?
                    """,
                    (
                        next_entry["content"],
                        next_entry["category"],
                        self._json(next_entry["metadata"]),
                        next_entry["revision"],
                        next_entry["updated_at"],
                        next_entry["checksum"],
                        memory_id,
                    ),
                )
                self._insert_history_row(conn, next_entry)
                self._upsert_fts_row(conn, next_entry)
                return next_entry
        finally:
            conn.close()

    def _delete_sync(self, memory_id: str) -> bool:
        conn = self._connect()
        try:
            with conn:
                if self._supports_fts5:
                    try:
                        conn.execute("DELETE FROM memories_fts WHERE id = ?", (memory_id,))
                    except sqlite3.OperationalError:
                        self._supports_fts5 = False
                cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
                return cursor.rowcount > 0
        finally:
            conn.close()

    def _export_sync(self, category: str | None, limit: int | None) -> list[MemoryEntry]:
        conn = self._connect()
        try:
            query = "SELECT * FROM memories"
            params: list[Any] = []
            if category is not None:
                query += " WHERE category = ?"
                params.append(category)
            query += " ORDER BY created_at DESC, id ASC"
            if limit is not None:
                query += " LIMIT ?"
                params.append(limit)
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_memory_entry(row, with_score=False) for row in rows]
        finally:
            conn.close()

    def _get_history_sync(self, memory_id: str) -> list[MemoryEntry]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT record_json FROM memory_revisions
                WHERE memory_id = ?
                ORDER BY revision ASC
                """,
                (memory_id,),
            ).fetchall()
            return [self._normalize_backup_record(json.loads(row["record_json"])) for row in rows]
        finally:
            conn.close()

    def _backup_sync(self) -> dict[str, Any]:
        conn = self._connect()
        try:
            records = [
                self._row_to_memory_entry(row, with_score=False)
                for row in conn.execute("SELECT * FROM memories ORDER BY id ASC").fetchall()
            ]
            for record in records:
                if record["checksum"] != self._checksum(record):
                    raise ValueError("Memory backup record checksum mismatch")
            history = [
                self._normalize_backup_record(json.loads(row["record_json"]))
                for row in conn.execute(
                    "SELECT record_json FROM memory_revisions ORDER BY memory_id ASC, revision ASC"
                ).fetchall()
            ]
            profile = conn.execute("SELECT content, updated_at FROM user_profile WHERE id = 1").fetchone()
            return {
                "schema_version": self._BACKUP_SCHEMA_VERSION,
                "records": records,
                "history": history,
                "user_profile": (
                    None
                    if profile is None
                    else {"content": str(profile["content"]), "updated_at": str(profile["updated_at"])}
                ),
            }
        finally:
            conn.close()

    def _restore_sync(self, backup: Mapping[str, Any], dry_run: bool) -> dict[str, Any]:
        records, history, user_profile = self._validate_backup(backup)
        report = {
            "status": "validated" if dry_run else "restored",
            "records": len(records),
            "history": len(history),
            "user_profile": user_profile is not None,
        }
        if dry_run:
            return report

        conn = self._connect()
        try:
            with conn:
                if self._supports_fts5:
                    conn.execute("DELETE FROM memories_fts")
                conn.execute("DELETE FROM memory_revisions")
                conn.execute("DELETE FROM memories")
                conn.execute("DELETE FROM user_profile")
                for entry in records:
                    self._insert_memory_row(conn, entry)
                for entry in history:
                    self._insert_history_row(conn, entry)
                if user_profile is not None:
                    conn.execute(
                        "INSERT INTO user_profile (id, content, updated_at) VALUES (1, ?, ?)",
                        (user_profile["content"], user_profile["updated_at"]),
                    )
                if self._supports_fts5:
                    self._rebuild_fts_sync(conn, fail_closed=True)
            return report
        finally:
            conn.close()

    def _validate_backup(
        self, backup: Mapping[str, Any]
    ) -> tuple[list[MemoryEntry], list[MemoryEntry], dict[str, str] | None]:
        if backup.get("schema_version") != self._BACKUP_SCHEMA_VERSION:
            raise ValueError("Memory backup schema version must be 2")
        raw_records = backup.get("records")
        raw_history = backup.get("history")
        if not isinstance(raw_records, list) or not isinstance(raw_history, list):
            raise ValueError("Memory backup records and history must be lists")
        records = [self._normalize_backup_record(record) for record in raw_records]
        history = [self._normalize_backup_record(record) for record in raw_history]
        record_ids = [str(record["id"]) for record in records]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("Memory backup contains duplicate ids")
        history_keys = [(str(record["id"]), int(record["revision"])) for record in history]
        if len(history_keys) != len(set(history_keys)):
            raise ValueError("Memory backup contains duplicate history revisions")
        if not set((str(record["id"]), int(record["revision"])) for record in records).issubset(
            set(history_keys)
        ):
            raise ValueError("Memory backup active records must have retained history")
        history_by_key = {
            (str(record["id"]), int(record["revision"])): record for record in history
        }
        for record in records:
            key = (str(record["id"]), int(record["revision"]))
            if record != history_by_key[key]:
                raise ValueError("Memory backup active record conflicts with retained history")

        raw_profile = backup.get("user_profile")
        if raw_profile is None:
            return records, history, None
        if not isinstance(raw_profile, Mapping):
            raise ValueError("Memory backup user_profile must be an object or null")
        content = raw_profile.get("content")
        updated_at = raw_profile.get("updated_at")
        if not isinstance(content, str) or not isinstance(updated_at, str):
            raise ValueError("Memory backup user_profile is malformed")
        return records, history, {"content": content, "updated_at": updated_at}

    def _normalize_backup_record(self, raw: Any) -> MemoryEntry:
        if not isinstance(raw, Mapping):
            raise ValueError("Memory backup record must be an object")
        missing = [field for field in self._RECORD_FIELDS if field not in raw]
        if missing or "checksum" not in raw:
            raise ValueError("Memory backup record is missing required fields")
        metadata = raw["metadata"]
        provenance = raw["provenance"]
        if not isinstance(metadata, dict) or not isinstance(provenance, dict):
            raise ValueError("Memory backup metadata and provenance must be objects")
        if not isinstance(raw["id"], str) or not raw["id"]:
            raise ValueError("Memory backup id is invalid")
        if not isinstance(raw["content"], str) or not isinstance(raw["category"], str):
            raise ValueError("Memory backup content and category must be strings")
        namespace = self._validate_namespace(raw["namespace"])
        revision = raw["revision"]
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise ValueError("Memory backup revision is invalid")
        supersedes_id = raw["supersedes_id"]
        if supersedes_id is not None and not isinstance(supersedes_id, str):
            raise ValueError("Memory backup supersedes_id is invalid")
        if not isinstance(raw["created_at"], str) or not isinstance(raw["updated_at"], str):
            raise ValueError("Memory backup timestamps are invalid")
        entry = self._new_entry(
            memory_id=raw["id"],
            content=raw["content"],
            category=raw["category"],
            metadata=metadata,
            namespace=namespace,
            provenance=provenance,
            revision=revision,
            supersedes_id=supersedes_id,
            created_at=raw["created_at"],
            updated_at=raw["updated_at"],
        )
        if raw["checksum"] != entry["checksum"]:
            raise ValueError("Memory backup record checksum mismatch")
        return entry

    def _get_user_profile_sync(self) -> str:
        conn = self._connect()
        try:
            row = conn.execute("SELECT content FROM user_profile WHERE id = 1").fetchone()
            return "" if row is None else str(row["content"])
        finally:
            conn.close()

    def _update_user_profile_sync(self, updates: str) -> None:
        updates = updates.strip()
        if not updates:
            return

        conn = self._connect()
        try:
            with conn:
                row = conn.execute("SELECT content FROM user_profile WHERE id = 1").fetchone()
                existing = "" if row is None else str(row["content"]).strip()
                next_content = updates if not existing else f"{existing}\n{updates}"
                conn.execute(
                    """
                    INSERT INTO user_profile (id, content, updated_at)
                    VALUES (1, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        content = excluded.content,
                        updated_at = excluded.updated_at
                    """,
                    (next_content, self._now()),
                )
        finally:
            conn.close()

    def _insert_history_row(self, conn: sqlite3.Connection, entry: MemoryEntry) -> None:
        conn.execute(
            """
            INSERT INTO memory_revisions (memory_id, revision, record_json, recorded_at)
            VALUES (?, ?, ?, ?)
            """,
            (entry["id"], entry["revision"], self._json(entry), entry["updated_at"]),
        )

    def _new_entry(
        self,
        *,
        memory_id: str,
        content: str,
        category: str,
        metadata: dict[str, Any],
        namespace: str,
        provenance: dict[str, Any],
        revision: int,
        supersedes_id: str | None,
        created_at: str,
        updated_at: str,
    ) -> MemoryEntry:
        entry: MemoryEntry = {
            "id": memory_id,
            "content": content,
            "category": category,
            "metadata": json.loads(self._json(metadata)),
            "namespace": namespace,
            "provenance": json.loads(self._json(provenance)),
            "revision": revision,
            "supersedes_id": supersedes_id,
            "created_at": created_at,
            "updated_at": updated_at,
        }
        entry["checksum"] = self._checksum(entry)
        return entry

    def _row_to_memory_entry(self, row: sqlite3.Row, with_score: bool) -> MemoryEntry:
        keys = set(row.keys())
        metadata = self._parse_object(row["metadata_json"])
        provenance = self._parse_object(row["provenance_json"]) if "provenance_json" in keys else {}
        created_at = str(row["created_at"])
        entry = self._new_entry(
            memory_id=str(row["id"]),
            content=str(row["content"]),
            category=str(row["category"]),
            metadata=metadata,
            namespace=str(row["namespace"]) if "namespace" in keys else "default",
            provenance=provenance,
            revision=int(row["revision"]) if "revision" in keys and row["revision"] else 1,
            supersedes_id=row["supersedes_id"] if "supersedes_id" in keys else None,
            created_at=created_at,
            updated_at=(str(row["updated_at"]) if "updated_at" in keys and row["updated_at"] else created_at),
        )
        if "checksum" in keys and row["checksum"]:
            entry["checksum"] = str(row["checksum"])
        if with_score and "score" in keys and row["score"] is not None:
            entry["score"] = float(row["score"])
        return entry

    def _to_fts_query(self, raw_query: str) -> str:
        tokens = [token.strip() for token in raw_query.split() if token.strip()]
        if not tokens:
            escaped = raw_query.replace('"', '""').strip()
            return f'"{escaped}"' if escaped else ""
        return " ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)

    @staticmethod
    def _parse_object(raw: Any) -> dict[str, Any]:
        if not isinstance(raw, str):
            return {}
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _validate_namespace(value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("memory namespace must be a non-empty string")
        return value.strip()

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def _checksum(self, entry: Mapping[str, Any]) -> str:
        payload = {field: entry[field] for field in self._RECORD_FIELDS}
        return hashlib.sha256(self._json(payload).encode("utf-8")).hexdigest()

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat(timespec="seconds")
