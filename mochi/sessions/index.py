"""Rebuildable, bounded full-text index derived from JSONL session history.

The JSONL files owned by :mod:`mochi.sessions.store` remain canonical.  This
module deliberately stores only an indexable preview and stable references, so
the SQLite database can be rebuilt or discarded without losing conversation
history.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import os
import re
import sqlite3
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any
from urllib.parse import quote
from uuid import uuid4

INDEX_FILENAME = ".mochi-session-search-index-v1.sqlite3"
INDEX_SCHEMA_VERSION = 1
MAX_INDEXED_PREVIEW_CHARS = 512
MAX_QUERY_RESULTS = 100

_TEXT_FIELDS = frozenset({"answer", "content", "message", "output", "prompt", "query", "summary", "text", "title"})
_ARTIFACT_REFERENCE_FIELDS = frozenset({"artifact_id", "artifact_ref", "artifact_reference", "artifact_uri"})
_BASE64_CHARS = re.compile(r"[A-Za-z0-9+/=_-]+")
_FTS_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


class SessionSearchIndexError(RuntimeError):
    """Base error for the rebuildable session search index."""


class SessionSearchIndexUnavailableError(SessionSearchIndexError):
    """The derived index cannot currently provide safe search results."""

    code = "session_search_index_unavailable"


class SessionSearchIndexSchemaError(SessionSearchIndexUnavailableError):
    """The database schema is not compatible with this index version."""


@dataclass(frozen=True)
class SessionIndexSnapshot:
    """One immutable canonical-source snapshot used to build the derived DB."""

    session_id: str
    events: tuple[Mapping[str, Any], ...]
    history_revision: str
    source_modified_ns: int


@dataclass(frozen=True)
class SessionSearchDocument:
    """A bounded record retained by the derived index."""

    session_id: str
    event_index: int
    turn_id: str
    timestamp: str | None
    source: str
    preview: str
    artifact_ref: str | None
    content_kind: str
    score: float | None = None


@dataclass(frozen=True)
class SessionSearchIndexRebuildReport:
    """Observable result of an all-or-nothing rebuild."""

    schema_version: int
    session_count: int
    document_count: int
    indexed_document_count: int


class SessionSearchIndex:
    """SQLite FTS5 cache with explicit availability and rebuild semantics."""

    def __init__(self, sessions_dir: str | Path) -> None:
        self._sessions_dir = Path(sessions_dir).expanduser()
        self._db_path = self._sessions_dir / INDEX_FILENAME
        self._lock_path = self._sessions_dir / f"{INDEX_FILENAME}.lock"
        self._thread_lock = RLock()

    @property
    def database_path(self) -> Path:
        """Return the derived database path without implying availability."""

        return self._db_path

    async def replace_snapshot(self, snapshot: SessionIndexSnapshot) -> None:
        await asyncio.to_thread(self.replace_snapshot_sync, snapshot)

    def replace_snapshot_sync(self, snapshot: SessionIndexSnapshot) -> None:
        """Replace one session's rows once its canonical source write succeeded."""

        _validate_snapshot(snapshot)
        with self._index_lock():
            conn = self._connect()
            try:
                self._ensure_schema(conn)
                conn.commit()
                conn.execute("BEGIN IMMEDIATE")
                prior = conn.execute(
                    "SELECT source_modified_ns FROM indexed_sessions WHERE session_id = ?",
                    (snapshot.session_id,),
                ).fetchone()
                if prior is not None and int(prior[0]) > snapshot.source_modified_ns:
                    conn.rollback()
                    return
                self._replace_session_rows(conn, snapshot)
                self._set_ready(conn)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    async def remove_session(self, session_id: str, *, source_modified_ns: int | None = None) -> None:
        await asyncio.to_thread(
            self.remove_session_sync,
            session_id,
            source_modified_ns=source_modified_ns,
        )

    def remove_session_sync(self, session_id: str, *, source_modified_ns: int | None = None) -> None:
        """Remove derived rows after a successful canonical-source deletion."""

        sid = _require_nonempty_string(session_id, "session_id")
        deleted_at_ns = time.time_ns() if source_modified_ns is None else _require_nonnegative_int(
            source_modified_ns,
            "source_modified_ns",
        )
        with self._index_lock():
            if not self._db_path.exists():
                return
            conn = self._connect()
            try:
                self._ensure_schema(conn)
                conn.commit()
                conn.execute("BEGIN IMMEDIATE")
                prior = conn.execute(
                    "SELECT source_modified_ns FROM indexed_sessions WHERE session_id = ?",
                    (sid,),
                ).fetchone()
                if prior is not None and int(prior[0]) > deleted_at_ns:
                    conn.rollback()
                    return
                self._delete_session_rows(conn, sid)
                conn.execute(
                    """
                    INSERT INTO indexed_sessions (
                        session_id, history_revision, source_modified_ns, event_count, deleted
                    ) VALUES (?, ?, ?, ?, 1)
                    ON CONFLICT(session_id) DO UPDATE SET
                        history_revision = excluded.history_revision,
                        source_modified_ns = excluded.source_modified_ns,
                        event_count = excluded.event_count,
                        deleted = 1
                    """,
                    (sid, "deleted", deleted_at_ns, 0),
                )
                self._set_ready(conn)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    async def rebuild_from_supplier(
        self,
        snapshot_supplier: Callable[[], Sequence[SessionIndexSnapshot]],
    ) -> SessionSearchIndexRebuildReport:
        return await asyncio.to_thread(self.rebuild_from_supplier_sync, snapshot_supplier)

    def rebuild_from_supplier_sync(
        self,
        snapshot_supplier: Callable[[], Sequence[SessionIndexSnapshot]],
    ) -> SessionSearchIndexRebuildReport:
        """Build a fresh database from current JSONL snapshots, then replace it."""

        if not callable(snapshot_supplier):
            raise TypeError("snapshot_supplier must be callable")
        with self._index_lock():
            # Taking the index lock before collecting the snapshots makes any
            # concurrent source commit wait at its derived-state update.  A
            # later commit still wins after the replacement through its newer
            # source timestamp.
            snapshots = tuple(snapshot_supplier())
            _validate_rebuild_snapshots(snapshots)
            tmp_path = self._db_path.with_name(f"{self._db_path.name}.{uuid4().hex}.tmp")
            try:
                report = self._build_database(tmp_path, snapshots)
                self._replace_database(tmp_path, self._db_path)
                return report
            except Exception:
                # Do not re-enter the interprocess lock while handling a
                # rebuild failure.  If a prior database exists, leave it
                # explicitly unavailable; a corrupt database already fails
                # closed when read.
                self._mark_unavailable_locked()
                raise
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()

    async def mark_unavailable(self) -> None:
        await asyncio.to_thread(self.mark_unavailable_sync)

    def mark_unavailable_sync(self) -> None:
        """Prevent a stale derived DB from being treated as an empty result."""

        with self._index_lock():
            self._mark_unavailable_locked()

    def _mark_unavailable_locked(self) -> None:
        if not self._db_path.exists():
            return
        try:
            conn = self._connect()
        except sqlite3.Error:
            return
        try:
            self._ensure_schema(conn)
            conn.commit()
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT INTO index_metadata (key, value) VALUES ('index_state', 'unavailable') "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value"
            )
            conn.commit()
        except (SessionSearchIndexError, sqlite3.Error):
            conn.rollback()
        finally:
            conn.close()

    async def search(self, query: str, *, limit: int = 20) -> tuple[SessionSearchDocument, ...]:
        return await asyncio.to_thread(self.search_sync, query, limit=limit)

    def search_sync(self, query: str, *, limit: int = 20) -> tuple[SessionSearchDocument, ...]:
        """Return only bounded records, or fail explicitly when the index is unavailable."""

        bounded_limit = _require_search_limit(limit)
        fts_query = _to_fts_query(query)
        if not fts_query:
            return ()
        conn = self._connect_read_only()
        try:
            self._require_ready_schema(conn)
            rows = conn.execute(
                """
                SELECT
                    documents.session_id,
                    documents.event_index,
                    documents.turn_id,
                    documents.timestamp,
                    documents.source,
                    documents.preview,
                    documents.artifact_ref,
                    documents.content_kind,
                    bm25(session_search_fts) AS score
                FROM session_search_fts
                JOIN session_search_documents AS documents
                    ON documents.id = session_search_fts.rowid
                WHERE session_search_fts MATCH ?
                ORDER BY score ASC, documents.session_id ASC, documents.event_index ASC
                LIMIT ?
                """,
                (fts_query, bounded_limit),
            ).fetchall()
            return tuple(
                SessionSearchDocument(
                    session_id=str(row["session_id"]),
                    event_index=int(row["event_index"]),
                    turn_id=str(row["turn_id"]),
                    timestamp=str(row["timestamp"]) if row["timestamp"] is not None else None,
                    source=str(row["source"]),
                    preview=str(row["preview"]),
                    artifact_ref=str(row["artifact_ref"]) if row["artifact_ref"] is not None else None,
                    content_kind=str(row["content_kind"]),
                    score=float(row["score"]),
                )
                for row in rows
            )
        except sqlite3.Error as exc:
            raise SessionSearchIndexUnavailableError("session search index cannot be queried safely") from exc
        finally:
            conn.close()

    async def documents_for_session(self, session_id: str) -> tuple[SessionSearchDocument, ...]:
        return await asyncio.to_thread(self.documents_for_session_sync, session_id)

    def documents_for_session_sync(self, session_id: str) -> tuple[SessionSearchDocument, ...]:
        sid = _require_nonempty_string(session_id, "session_id")
        conn = self._connect_read_only()
        try:
            self._require_ready_schema(conn)
            rows = conn.execute(
                """
                SELECT session_id, event_index, turn_id, timestamp, source, preview, artifact_ref, content_kind
                FROM session_search_documents
                WHERE session_id = ?
                ORDER BY event_index ASC
                """,
                (sid,),
            ).fetchall()
            return tuple(
                SessionSearchDocument(
                    session_id=str(row["session_id"]),
                    event_index=int(row["event_index"]),
                    turn_id=str(row["turn_id"]),
                    timestamp=str(row["timestamp"]) if row["timestamp"] is not None else None,
                    source=str(row["source"]),
                    preview=str(row["preview"]),
                    artifact_ref=str(row["artifact_ref"]) if row["artifact_ref"] is not None else None,
                    content_kind=str(row["content_kind"]),
                )
                for row in rows
            )
        except sqlite3.Error as exc:
            raise SessionSearchIndexUnavailableError("session search index cannot be read safely") from exc
        finally:
            conn.close()

    def _build_database(
        self,
        path: Path,
        snapshots: Sequence[SessionIndexSnapshot],
    ) -> SessionSearchIndexRebuildReport:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, timeout=30.0)
        try:
            self._configure_connection(conn)
            self._create_schema(conn)
            conn.commit()
            conn.execute("BEGIN IMMEDIATE")
            document_count = 0
            indexed_document_count = 0
            for snapshot in sorted(snapshots, key=lambda item: item.session_id):
                documents = _documents_for_snapshot(snapshot)
                document_count += len(documents)
                indexed_document_count += sum(bool(document.preview) for document in documents)
                self._insert_session_rows(conn, snapshot, documents)
            self._set_ready(conn)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return SessionSearchIndexRebuildReport(
            schema_version=INDEX_SCHEMA_VERSION,
            session_count=len(snapshots),
            document_count=document_count,
            indexed_document_count=indexed_document_count,
        )

    def _replace_session_rows(self, conn: sqlite3.Connection, snapshot: SessionIndexSnapshot) -> None:
        self._delete_session_rows(conn, snapshot.session_id)
        self._insert_session_rows(conn, snapshot, _documents_for_snapshot(snapshot))

    @staticmethod
    def _delete_session_rows(conn: sqlite3.Connection, session_id: str) -> None:
        rows = conn.execute(
            "SELECT id, preview FROM session_search_documents WHERE session_id = ?",
            (session_id,),
        ).fetchall()
        for row in rows:
            if row["preview"]:
                conn.execute(
                    "INSERT INTO session_search_fts(session_search_fts, rowid, preview) VALUES ('delete', ?, ?)",
                    (int(row["id"]), str(row["preview"])),
                )
        conn.execute("DELETE FROM session_search_documents WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM indexed_sessions WHERE session_id = ?", (session_id,))

    @staticmethod
    def _insert_session_rows(
        conn: sqlite3.Connection,
        snapshot: SessionIndexSnapshot,
        documents: Sequence[SessionSearchDocument],
    ) -> None:
        conn.execute(
            """
            INSERT INTO indexed_sessions (
                session_id, history_revision, source_modified_ns, event_count, deleted
            ) VALUES (?, ?, ?, ?, 0)
            """,
            (
                snapshot.session_id,
                snapshot.history_revision,
                snapshot.source_modified_ns,
                len(snapshot.events),
            ),
        )
        for document in documents:
            cursor = conn.execute(
                """
                INSERT INTO session_search_documents (
                    session_id, event_index, turn_id, timestamp, source, preview, artifact_ref, content_kind
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document.session_id,
                    document.event_index,
                    document.turn_id,
                    document.timestamp,
                    document.source,
                    document.preview,
                    document.artifact_ref,
                    document.content_kind,
                ),
            )
            if document.preview:
                conn.execute(
                    "INSERT INTO session_search_fts(rowid, preview) VALUES (?, ?)",
                    (int(cursor.lastrowid), document.preview),
                )

    def _connect(self) -> sqlite3.Connection:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path, timeout=30.0)
        self._configure_connection(conn)
        return conn

    def _connect_read_only(self) -> sqlite3.Connection:
        if not self._db_path.exists():
            raise SessionSearchIndexUnavailableError("session search index has not been built")
        try:
            conn = sqlite3.connect(f"file:{self._db_path.as_posix()}?mode=ro", uri=True, timeout=30.0)
        except sqlite3.Error as exc:
            raise SessionSearchIndexUnavailableError("session search index is unavailable") from exc
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _configure_connection(conn: sqlite3.Connection) -> None:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = DELETE")

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        if not self._table_exists(conn, "index_metadata"):
            if self._has_user_tables(conn):
                raise SessionSearchIndexSchemaError("session search index requires a rebuild for schema migration")
            self._create_schema(conn)
            return
        self._require_schema_version(conn)

    def _require_ready_schema(self, conn: sqlite3.Connection) -> None:
        self._require_schema_version(conn)
        row = conn.execute("SELECT value FROM index_metadata WHERE key = 'index_state'").fetchone()
        if row is None or row["value"] != "ready":
            raise SessionSearchIndexUnavailableError("session search index is not ready")

    @staticmethod
    def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
        return conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone() is not None

    @staticmethod
    def _has_user_tables(conn: sqlite3.Connection) -> bool:
        return conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' LIMIT 1"
        ).fetchone() is not None

    @staticmethod
    def _require_schema_version(conn: sqlite3.Connection) -> None:
        row = conn.execute("SELECT value FROM index_metadata WHERE key = 'schema_version'").fetchone()
        if row is None or row["value"] != str(INDEX_SCHEMA_VERSION):
            raise SessionSearchIndexSchemaError("session search index schema is unsupported; rebuild required")

    @staticmethod
    def _set_ready(conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO index_metadata (key, value) VALUES ('index_state', 'ready') "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value"
        )

    @staticmethod
    def _create_schema(conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS index_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS indexed_sessions (
                session_id TEXT PRIMARY KEY,
                history_revision TEXT NOT NULL,
                source_modified_ns INTEGER NOT NULL,
                event_count INTEGER NOT NULL,
                deleted INTEGER NOT NULL CHECK (deleted IN (0, 1))
            );
            CREATE TABLE IF NOT EXISTS session_search_documents (
                id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                event_index INTEGER NOT NULL,
                turn_id TEXT NOT NULL,
                timestamp TEXT,
                source TEXT NOT NULL,
                preview TEXT NOT NULL,
                artifact_ref TEXT,
                content_kind TEXT NOT NULL,
                UNIQUE(session_id, event_index)
            );
            CREATE INDEX IF NOT EXISTS idx_session_search_documents_session
                ON session_search_documents(session_id, event_index);
            CREATE VIRTUAL TABLE IF NOT EXISTS session_search_fts
                USING fts5(preview, content = 'session_search_documents', content_rowid = 'id');
            """
        )
        conn.execute("PRAGMA user_version = 1")
        conn.execute(
            "INSERT OR REPLACE INTO index_metadata (key, value) VALUES ('schema_version', ?)",
            (str(INDEX_SCHEMA_VERSION),),
        )
        conn.execute("INSERT OR REPLACE INTO index_metadata (key, value) VALUES ('index_state', 'unavailable')")

    @staticmethod
    def _replace_database(source: Path, target: Path) -> None:
        delay = 0.01
        for attempt in range(7):
            try:
                os.replace(source, target)
                return
            except PermissionError:
                if attempt == 6:
                    raise
                time.sleep(delay)
                delay *= 2

    @contextmanager
    def _index_lock(self):
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._thread_lock, self._lock_path.open("a+b") as handle:
            handle.seek(0)
            if not handle.read(1):
                handle.seek(0)
                handle.write(b"0")
                handle.flush()
            self._lock_file(handle)
            try:
                yield
            finally:
                self._unlock_file(handle)

    @staticmethod
    def _lock_file(handle: Any) -> None:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            return
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)

    @staticmethod
    def _unlock_file(handle: Any) -> None:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            return
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _documents_for_snapshot(snapshot: SessionIndexSnapshot) -> tuple[SessionSearchDocument, ...]:
    documents: list[SessionSearchDocument] = []
    for event_index, event in enumerate(snapshot.events):
        document = _document_for_event(snapshot.session_id, event_index, event)
        if document is not None:
            documents.append(document)
    return tuple(documents)


def _document_for_event(
    session_id: str,
    event_index: int,
    event: Mapping[str, Any],
) -> SessionSearchDocument | None:
    text_parts: list[str] = []
    has_binary = False
    has_oversized_text = False
    artifact_ref = _event_artifact_reference(session_id, event_index, event)

    def collect(value: Any, key: str | None = None) -> None:
        nonlocal has_binary, has_oversized_text
        if isinstance(value, str):
            if key not in _TEXT_FIELDS:
                return
            normalized = _normalize_text(value)
            if not normalized:
                return
            if _looks_binary_text(normalized):
                has_binary = True
                return
            if len(normalized) > MAX_INDEXED_PREVIEW_CHARS:
                has_oversized_text = True
            text_parts.append(normalized[:MAX_INDEXED_PREVIEW_CHARS])
            return
        if isinstance(value, Mapping):
            for nested_key, nested_value in value.items():
                collect(nested_value, nested_key if isinstance(nested_key, str) else None)
            return
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            for nested_value in value:
                collect(nested_value, key)
            return
        if isinstance(value, (bytes, bytearray)):
            has_binary = True

    collect(event)
    preview = _bounded_join(text_parts)
    if not preview and not has_binary and not has_oversized_text:
        return None
    if has_binary and not preview:
        content_kind = "artifact_only"
    elif has_binary or has_oversized_text:
        content_kind = "bounded_text_and_artifact"
    else:
        content_kind = "bounded_text"
    return SessionSearchDocument(
        session_id=session_id,
        event_index=event_index,
        turn_id=_event_turn_id(event, event_index),
        timestamp=_event_timestamp(event),
        source=_event_source(event),
        preview=preview,
        artifact_ref=artifact_ref if has_binary or has_oversized_text else _explicit_artifact_reference(event),
        content_kind=content_kind,
    )


def _event_artifact_reference(session_id: str, event_index: int, event: Mapping[str, Any]) -> str:
    return _explicit_artifact_reference(event) or f"session://{quote(session_id, safe='')}/events/{event_index}"


def _explicit_artifact_reference(event: Mapping[str, Any]) -> str | None:
    stack: list[Any] = [event]
    while stack:
        value = stack.pop()
        if isinstance(value, Mapping):
            for key, nested_value in value.items():
                if key in _ARTIFACT_REFERENCE_FIELDS and isinstance(nested_value, str):
                    reference = nested_value.strip()
                    if reference and len(reference) <= 2048:
                        return reference
                stack.append(nested_value)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            stack.extend(reversed(value))
    return None


def _event_turn_id(event: Mapping[str, Any], event_index: int) -> str:
    for key in ("turn_id", "event_id", "id"):
        value = event.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:256]
    return f"event-{event_index}"


def _event_timestamp(event: Mapping[str, Any]) -> str | None:
    for key in ("timestamp", "created_at", "occurred_at"):
        value = event.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:256]
    return None


def _event_source(event: Mapping[str, Any]) -> str:
    for key in ("type", "event", "role"):
        value = event.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:128]
    return "session_event"


def _normalize_text(value: str) -> str:
    return " ".join(value.split())


def _bounded_join(parts: Sequence[str]) -> str:
    result = ""
    for part in parts:
        separator = " " if result else ""
        remaining = MAX_INDEXED_PREVIEW_CHARS - len(result) - len(separator)
        if remaining <= 0:
            break
        result += separator + part[:remaining]
    return result


def _looks_binary_text(value: str) -> bool:
    if "\x00" in value:
        return True
    candidate = value
    if candidate.startswith("data:") and ";base64," in candidate:
        candidate = candidate.split(";base64,", 1)[1]
    if len(candidate) < 256 or _BASE64_CHARS.fullmatch(candidate) is None:
        return False
    try:
        padded = candidate + "=" * (-len(candidate) % 4)
        decoded = base64.b64decode(padded, validate=True)
    except (ValueError, binascii.Error):
        return False
    return len(decoded) >= 192


def _to_fts_query(query: str) -> str:
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    tokens = _FTS_TOKEN.findall(query)
    return " AND ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)


def _validate_snapshot(snapshot: SessionIndexSnapshot) -> None:
    if not isinstance(snapshot, SessionIndexSnapshot):
        raise TypeError("snapshot must be a SessionIndexSnapshot")
    _require_nonempty_string(snapshot.session_id, "snapshot.session_id")
    _require_nonempty_string(snapshot.history_revision, "snapshot.history_revision")
    _require_nonnegative_int(snapshot.source_modified_ns, "snapshot.source_modified_ns")
    for index, event in enumerate(snapshot.events):
        if not isinstance(event, Mapping):
            raise TypeError(f"snapshot.events[{index}] must be a mapping")


def _validate_rebuild_snapshots(snapshots: Sequence[SessionIndexSnapshot]) -> None:
    session_ids: set[str] = set()
    for snapshot in snapshots:
        _validate_snapshot(snapshot)
        if snapshot.session_id in session_ids:
            raise ValueError("rebuild snapshots must contain each session_id once")
        session_ids.add(snapshot.session_id)


def _require_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _require_nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_search_limit(value: Any) -> int:
    if type(value) is not int or not 1 <= value <= MAX_QUERY_RESULTS:
        raise ValueError(f"limit must be an integer from 1 through {MAX_QUERY_RESULTS}")
    return value
