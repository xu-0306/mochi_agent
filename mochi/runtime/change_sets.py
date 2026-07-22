"""Durable persistence facade for immutable change manifests and file journals."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from mochi.security.file_contract import (
    AUTHORIZATION_ENVELOPE_SCHEMA_VERSION,
    AppliedChangeRecord,
    AuthorizationEnvelope,
    ChangeEntry,
    ChangeManifest,
    FileIdentity,
    authorization_request_digest,
    canonical_json,
)

from .store import RuntimeStore


class ChangeSetConflict(RuntimeError):
    """Raised when an immutable change-set identity is reused with new content."""


class BlobNotFound(KeyError):
    """Raised when a reference targets a blob that was never persisted."""


class UndoConflict(RuntimeError):
    """Undo selection or authoritative state is inconsistent."""


class UndoUnavailable(RuntimeError):
    """Undo retention is known but no longer available."""


JournalEntryState = Literal[
    "staged",
    "applying",
    "applied",
    "rolling_back",
    "rolled_back",
    "rollback_failed",
    "interference",
]


@dataclass(frozen=True, slots=True)
class JournalEntryRecord:
    entry_id: str
    ordinal: int
    state: JournalEntryState
    base_sha256: str | None
    after_sha256: str | None
    base_identity: FileIdentity | None
    staged_name: str | None
    staged_identity: FileIdentity | None
    rollback_blob_id: str | None
    rollback_staged_name: str | None
    rollback_staged_identity: FileIdentity | None
    rollback_successor_identity: FileIdentity | None
    base_metadata_blob_id: str | None
    last_error: str | None


def _runtime_object(value: object) -> object:
    return value


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _identity_json(identity: FileIdentity | None) -> str | None:
    if identity is None:
        return None
    return canonical_json(identity.to_dict())


def _identity_from_json(value: object) -> FileIdentity | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("stored identity must be JSON text")
    decoded: object = json.loads(value)
    if not isinstance(decoded, dict):
        raise TypeError("stored identity must be an object with string keys")
    decoded_mapping = cast(dict[object, object], decoded)
    if not all(isinstance(key, str) for key in decoded_mapping):
        raise TypeError("stored identity must be an object with string keys")
    return FileIdentity.from_dict(
        cast(Mapping[str, object], decoded_mapping)
    )


def _journal_entry_from_row(row: sqlite3.Row) -> JournalEntryRecord:
    return JournalEntryRecord(
        entry_id=str(row["entry_id"]),
        ordinal=int(row["ordinal"]),
        state=str(row["state"]),  # type: ignore[arg-type]
        base_sha256=row["base_sha256"],
        after_sha256=row["after_sha256"],
        base_identity=_identity_from_json(row["base_identity_json"]),
        staged_name=row["staged_name"],
        staged_identity=_identity_from_json(row["staged_identity_json"]),
        rollback_blob_id=row["rollback_blob_id"],
        rollback_staged_name=row["rollback_staged_name"],
        rollback_staged_identity=_identity_from_json(
            row["rollback_staged_identity_json"]
        ),
        rollback_successor_identity=_identity_from_json(
            row["rollback_successor_identity_json"]
        ),
        base_metadata_blob_id=row["base_metadata_blob_id"],
        last_error=row["last_error"],
    )


class ChangeSetStore:
    """Serialize immutable manifests, authoritative blobs, and recovery state."""

    _TERMINAL_JOURNAL_STATUSES = frozenset(
        {"applied", "rolled_back", "rollback_failed", "interference"}
    )

    def __init__(self, runtime_store: RuntimeStore) -> None:
        runtime_value = _runtime_object(runtime_store)
        if not isinstance(runtime_value, RuntimeStore):
            raise TypeError("runtime_store must be RuntimeStore")
        self._runtime_store = runtime_value
        self._db_path = Path(runtime_value.database_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    async def _initialize(self) -> None:
        await self._runtime_store.initialize()

    @staticmethod
    def _validate_manifest(
        manifest: ChangeManifest,
        envelope: AuthorizationEnvelope,
    ) -> None:
        manifest_value = _runtime_object(manifest)
        envelope_value = _runtime_object(envelope)
        if not isinstance(manifest_value, ChangeManifest):
            raise TypeError("manifest must be ChangeManifest")
        if not isinstance(envelope_value, AuthorizationEnvelope):
            raise TypeError("envelope must be AuthorizationEnvelope")
        manifest = manifest_value
        envelope = envelope_value
        request = envelope.file_request
        if envelope.kind != "file_change" or request is None:
            raise ValueError("change manifests require a file_change envelope")
        if envelope.schema_version != AUTHORIZATION_ENVELOPE_SCHEMA_VERSION:
            raise ChangeSetConflict("superseded_schema")
        digest = authorization_request_digest(envelope)
        if manifest.request_digest != digest:
            raise ChangeSetConflict(
                "manifest request digest does not match its immutable envelope"
            )
        if (
            manifest.version != envelope.schema_version
            or manifest.workspace_root != envelope.context.workspace_root
            or manifest.workspace_identity != envelope.context.workspace_identity
            or manifest.entries != request.entries
            or manifest.patch_sha256 != request.patch_sha256
            or manifest.policy_version != envelope.policy_version
        ):
            raise ChangeSetConflict(
                "manifest projection does not match its immutable envelope"
            )

    async def persist_manifest(
        self,
        manifest: ChangeManifest,
        envelope: AuthorizationEnvelope,
    ) -> dict[str, Any]:
        self._validate_manifest(manifest, envelope)
        await self._initialize()
        return await asyncio.to_thread(self._persist_manifest, manifest, envelope)

    def _persist_manifest(
        self,
        manifest: ChangeManifest,
        envelope: AuthorizationEnvelope,
    ) -> dict[str, Any]:
        envelope_json = canonical_json(envelope.to_dict())
        workspace_identity_json = canonical_json(
            envelope.context.workspace_identity.to_dict()
        )
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT id FROM change_sets WHERE id=?",
                (manifest.change_set_id,),
            ).fetchone()
            if existing is not None:
                loaded = self._load_change_set(conn, manifest.change_set_id)
                if (
                    loaded is None
                    or loaded["manifest"] != manifest
                    or loaded["envelope"] != envelope
                ):
                    raise ChangeSetConflict(
                        "change_set_id already identifies another immutable request"
                    )
                conn.commit()
                return loaded

            reusable = conn.execute(
                """
                SELECT id
                FROM change_sets
                WHERE schema_version=?
                  AND requester_id=?
                  AND session_id=?
                  AND IFNULL(task_id, '')=IFNULL(?, '')
                  AND workspace_identity_json=?
                  AND request_digest=?
                LIMIT 1
                """,
                (
                    envelope.schema_version,
                    envelope.context.requester_id,
                    envelope.context.session_id,
                    envelope.context.task_id,
                    workspace_identity_json,
                    manifest.request_digest,
                ),
            ).fetchone()
            if reusable is not None:
                loaded = self._load_change_set(conn, str(reusable["id"]))
                if loaded is None:
                    raise RuntimeError("idempotent change set disappeared")
                conn.commit()
                return loaded

            conn.execute(
                """
                INSERT INTO change_sets (
                    id, schema_version, requester_id, session_id, task_id,
                    workspace_root, workspace_identity_json, tool_name, intent,
                    request_digest, authorization_envelope_json, patch_sha256,
                    policy_version, status, created_at, expires_at, applied_at,
                    updated_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    manifest.change_set_id,
                    envelope.schema_version,
                    envelope.context.requester_id,
                    envelope.context.session_id,
                    envelope.context.task_id,
                    manifest.workspace_root,
                    workspace_identity_json,
                    manifest.tool_name,
                    manifest.intent,
                    manifest.request_digest,
                    envelope_json,
                    manifest.patch_sha256,
                    manifest.policy_version,
                    "prepared",
                    manifest.created_at,
                    manifest.expires_at,
                    None,
                    manifest.created_at,
                    canonical_json(manifest.ui_metadata),
                ),
            )
            for ordinal, entry in enumerate(manifest.entries):
                conn.execute(
                    """
                    INSERT INTO change_entries (
                        id, change_set_id, ordinal, relative_path, operation,
                        base_sha256, after_sha256, base_identity_json,
                        before_blob_id, after_blob_id, mode_before, mode_after,
                        base_metadata_blob_id, after_metadata_blob_id,
                        rename_source, dependency_group, encoding, newline_style,
                        eof_newline
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        entry.entry_id,
                        manifest.change_set_id,
                        ordinal,
                        entry.relative_path,
                        entry.operation,
                        entry.base_sha256,
                        entry.after_sha256,
                        _identity_json(entry.base_identity),
                        entry.before_blob_id,
                        entry.after_blob_id,
                        entry.mode_before,
                        entry.mode_after,
                        entry.base_metadata_sha256,
                        entry.after_metadata_sha256,
                        entry.rename_source,
                        entry.dependency_group,
                        entry.encoding,
                        entry.newline_style,
                        None if entry.eof_newline is None else int(entry.eof_newline),
                    ),
                )
            loaded = self._load_change_set(conn, manifest.change_set_id)
            if loaded is None:
                raise RuntimeError("persisted change set could not be reloaded")
            conn.commit()
            return loaded

    def _load_change_set(
        self,
        conn: sqlite3.Connection,
        change_set_id: str,
    ) -> dict[str, Any] | None:
        row = conn.execute(
            "SELECT * FROM change_sets WHERE id=?",
            (change_set_id,),
        ).fetchone()
        if row is None:
            return None
        entry_rows = conn.execute(
            """
            SELECT * FROM change_entries
            WHERE change_set_id=?
            ORDER BY ordinal
            """,
            (change_set_id,),
        ).fetchall()
        entries = tuple(
            ChangeEntry(
                entry_id=str(entry["id"]),
                relative_path=str(entry["relative_path"]),
                operation=str(entry["operation"]),  # type: ignore[arg-type]
                base_sha256=entry["base_sha256"],
                after_sha256=entry["after_sha256"],
                base_identity=_identity_from_json(entry["base_identity_json"]),
                before_blob_id=entry["before_blob_id"],
                after_blob_id=entry["after_blob_id"],
                mode_before=entry["mode_before"],
                mode_after=entry["mode_after"],
                base_metadata_sha256=entry["base_metadata_blob_id"],
                after_metadata_sha256=entry["after_metadata_blob_id"],
                rename_source=entry["rename_source"],
                dependency_group=entry["dependency_group"],
                encoding=entry["encoding"],
                newline_style=entry["newline_style"],
                eof_newline=(
                    None
                    if entry["eof_newline"] is None
                    else bool(entry["eof_newline"])
                ),
            )
            for entry in entry_rows
        )
        envelope_data = json.loads(str(row["authorization_envelope_json"]))
        metadata = json.loads(str(row["metadata_json"]))
        manifest = ChangeManifest(
            version=int(row["schema_version"]),
            change_set_id=str(row["id"]),
            workspace_root=str(row["workspace_root"]),
            workspace_identity=_identity_from_json(
                row["workspace_identity_json"]
            ),  # type: ignore[arg-type]
            tool_name=str(row["tool_name"]),
            intent=str(row["intent"]),  # type: ignore[arg-type]
            entries=entries,
            patch_sha256=row["patch_sha256"],
            policy_version=str(row["policy_version"]),
            created_at=str(row["created_at"]),
            expires_at=str(row["expires_at"]),
            request_digest=str(row["request_digest"]),
            ui_metadata=metadata,
        )
        return {
            "id": str(row["id"]),
            "status": str(row["status"]),
            "manifest": manifest,
            "envelope": AuthorizationEnvelope.from_dict(envelope_data),
            "applied_at": row["applied_at"],
            "updated_at": str(row["updated_at"]),
        }

    async def get_change_set(self, change_set_id: str) -> dict[str, Any] | None:
        await self._initialize()

        def operation() -> dict[str, Any] | None:
            with self._connect() as conn:
                return self._load_change_set(conn, change_set_id)

        return await asyncio.to_thread(operation)

    async def put_blob(self, content: bytes) -> str:
        content_value = _runtime_object(content)
        if not isinstance(content_value, bytes):
            raise TypeError("content must be bytes")
        await self._initialize()
        return await asyncio.to_thread(self._put_blob, content_value)

    def _put_blob(self, content: bytes) -> str:
        blob_id = hashlib.sha256(content).hexdigest()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO change_blobs(
                    id, sha256, size_bytes, content, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (blob_id, blob_id, len(content), content, _now_iso()),
            )
            row = conn.execute(
                "SELECT size_bytes, content FROM change_blobs WHERE id=?",
                (blob_id,),
            ).fetchone()
            if (
                row is None
                or int(row["size_bytes"]) != len(content)
                or bytes(row["content"]) != content
            ):
                raise ChangeSetConflict("blob digest collision detected")
            conn.commit()
        return blob_id

    async def get_blob(self, blob_id: str) -> bytes | None:
        await self._initialize()

        def operation() -> bytes | None:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT content FROM change_blobs WHERE id=?",
                    (blob_id,),
                ).fetchone()
                return None if row is None else bytes(row["content"])

        return await asyncio.to_thread(operation)


    async def add_blob_reference(
        self,
        *,
        blob_id: str,
        owner_type: str,
        owner_id: str,
        purpose: str,
        retained_until: str | None,
    ) -> None:
        await self._initialize()

        def operation() -> None:
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                if conn.execute(
                    "SELECT 1 FROM change_blobs WHERE id=?",
                    (blob_id,),
                ).fetchone() is None:
                    raise BlobNotFound(blob_id)
                conn.execute(
                    """
                    INSERT INTO blob_references(
                        blob_id, owner_type, owner_id, purpose,
                        retained_until, state
                    ) VALUES (?, ?, ?, ?, ?, 'active')
                    ON CONFLICT(blob_id, owner_type, owner_id, purpose)
                    DO UPDATE SET retained_until=excluded.retained_until,
                                  state='active'
                    """,
                    (blob_id, owner_type, owner_id, purpose, retained_until),
                )
                conn.commit()

        await asyncio.to_thread(operation)

    async def expire_blob_references(self, *, now: str) -> int:
        await self._initialize()

        def operation() -> int:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    UPDATE blob_references
                    SET state='expired'
                    WHERE state='active'
                      AND retained_until IS NOT NULL
                      AND retained_until<=?
                    """,
                    (now,),
                )
                conn.commit()
                return int(cursor.rowcount)

        return await asyncio.to_thread(operation)

    async def collect_garbage(self, *, now: str) -> tuple[str, ...]:
        await self._initialize()

        def operation() -> tuple[str, ...]:
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    """
                    UPDATE blob_references
                    SET state='expired'
                    WHERE state='active'
                      AND retained_until IS NOT NULL
                      AND retained_until<=?
                    """,
                    (now,),
                )
                rows = conn.execute(
                    """
                    SELECT id
                    FROM change_blobs AS blob
                    WHERE NOT EXISTS (
                        SELECT 1
                        FROM blob_references AS reference
                        WHERE reference.blob_id=blob.id
                          AND reference.state='active'
                    )
                    ORDER BY id
                    """
                ).fetchall()
                blob_ids = tuple(str(row["id"]) for row in rows)
                for blob_id in blob_ids:
                    conn.execute(
                        "DELETE FROM change_blobs WHERE id=?",
                        (blob_id,),
                    )
                conn.commit()
                return blob_ids

        return await asyncio.to_thread(operation)

    async def activate_undo_retention(
        self,
        *,
        change_set_id: str,
        records: tuple[AppliedChangeRecord, ...],
        retained_until: str,
    ) -> tuple[dict[str, Any], ...]:
        """Atomically record applied state and activate authoritative undo refs."""

        await self._initialize()

        def operation() -> tuple[dict[str, Any], ...]:
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                loaded = self._load_change_set(conn, change_set_id)
                if loaded is None:
                    raise KeyError(change_set_id)
                if loaded["status"] not in {"prepared", "applied"}:
                    raise UndoConflict("undo_retention_not_reactivatable")
                manifest = loaded["manifest"]
                entries = {entry.entry_id: entry for entry in manifest.entries}
                record_map = {record.entry_id: record for record in records}
                if (
                    len(record_map) != len(records)
                    or set(record_map) != set(entries)
                    or any(record.change_set_id != change_set_id for record in records)
                ):
                    raise UndoConflict("applied_records_do_not_match_manifest")
                groups: dict[str, tuple[str, ...]] = {}
                for entry in manifest.entries:
                    group_key = entry.dependency_group or entry.entry_id
                    groups[group_key] = tuple(
                        item.entry_id
                        for item in manifest.entries
                        if (item.dependency_group or item.entry_id) == group_key
                    )
                projections: list[dict[str, Any]] = []
                for entry in manifest.entries:
                    record = record_map[entry.entry_id]
                    existing_retention = conn.execute(
                        """
                        SELECT status FROM undo_retention
                        WHERE change_set_id=? AND entry_id=?
                        """,
                        (change_set_id, entry.entry_id),
                    ).fetchone()
                    if (
                        existing_retention is not None
                        and str(existing_retention["status"]) != "retained"
                    ):
                        raise UndoConflict("undo_retention_not_reactivatable")
                    conn.execute(
                        """
                        INSERT INTO applied_change_entries(
                            change_set_id, entry_id, applied_sha256,
                            applied_identity_json, applied_metadata_sha256,
                            applied_at
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(change_set_id, entry_id)
                        DO UPDATE SET applied_sha256=excluded.applied_sha256,
                                      applied_identity_json=excluded.applied_identity_json,
                                      applied_metadata_sha256=excluded.applied_metadata_sha256,
                                      applied_at=excluded.applied_at
                        """,
                        (
                            change_set_id,
                            entry.entry_id,
                            record.applied_sha256,
                            _identity_json(record.applied_identity),
                            record.applied_metadata_sha256,
                            record.applied_at,
                        ),
                    )
                    conn.execute(
                        """
                        INSERT INTO undo_retention(
                            change_set_id, entry_id, status,
                            retained_until, expired_at
                        ) VALUES (?, ?, 'retained', ?, NULL)
                        ON CONFLICT(change_set_id, entry_id)
                        DO UPDATE SET status='retained',
                                      retained_until=excluded.retained_until,
                                      expired_at=NULL
                        """,
                        (change_set_id, entry.entry_id, retained_until),
                    )
                    if entry.before_blob_id is not None:
                        if conn.execute(
                            "SELECT 1 FROM change_blobs WHERE id=?",
                            (entry.before_blob_id,),
                        ).fetchone() is None:
                            raise BlobNotFound(entry.before_blob_id)
                        conn.execute(
                            """
                            INSERT INTO blob_references(
                                blob_id, owner_type, owner_id, purpose,
                                retained_until, state
                            ) VALUES (?, 'change_entry', ?, 'undo', ?, 'active')
                            ON CONFLICT(blob_id, owner_type, owner_id, purpose)
                            DO UPDATE SET retained_until=excluded.retained_until,
                                          state='active'
                            """,
                            (
                                entry.before_blob_id,
                                f"{change_set_id}:{entry.entry_id}",
                                retained_until,
                            ),
                        )
                    group_key = entry.dependency_group or entry.entry_id
                    projections.append(
                        {
                            "change_set_id": change_set_id,
                            "entry_id": entry.entry_id,
                            "request_digest": manifest.request_digest,
                            "dependency_group": entry.dependency_group,
                            "undo_entry_ids": list(groups[group_key]),
                            "undo_status": "retained",
                            "retained_until": retained_until,
                            "undo_available": True,
                        }
                    )
                conn.commit()
                return tuple(projections)

        return await asyncio.to_thread(operation)

    async def get_undo_material(
        self,
        *,
        change_set_id: str,
        entry_ids: tuple[str, ...],
        request_digest: str,
        now: str,
    ) -> dict[str, Any]:
        """Load retained server bytes after validating selection and expiry."""

        if not entry_ids or len(set(entry_ids)) != len(entry_ids):
            raise UndoConflict("invalid_undo_selection")
        await self._initialize()

        def operation() -> dict[str, Any]:
            expired = False
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                loaded = self._load_change_set(conn, change_set_id)
                if loaded is None:
                    raise KeyError(change_set_id)
                if loaded["status"] != "applied":
                    raise UndoConflict("change_set_not_applied")
                manifest = loaded["manifest"]
                if manifest.request_digest != request_digest:
                    raise UndoConflict("undo_digest_mismatch")
                entries = {entry.entry_id: entry for entry in manifest.entries}
                selected = set(entry_ids)
                if not selected <= set(entries):
                    raise UndoConflict("unknown_change_entry")
                for entry_id in selected:
                    group = entries[entry_id].dependency_group
                    if group is None:
                        continue
                    required = {
                        item.entry_id
                        for item in manifest.entries
                        if item.dependency_group == group
                    }
                    if not required <= selected:
                        raise UndoConflict("partial_dependency_group")

                material: list[dict[str, Any]] = []
                for entry in manifest.entries:
                    if entry.entry_id not in selected:
                        continue
                    retention = conn.execute(
                        """
                        SELECT status, retained_until, expired_at
                        FROM undo_retention
                        WHERE change_set_id=? AND entry_id=?
                        """,
                        (change_set_id, entry.entry_id),
                    ).fetchone()
                    if retention is None:
                        raise UndoUnavailable("undo_not_retained")
                    if str(retention["status"]) != "retained":
                        reason = (
                            "undo_retention_expired"
                            if str(retention["status"]) == "expired"
                            else "undo_not_retained"
                        )
                        raise UndoUnavailable(reason)
                    retained_until = retention["retained_until"]
                    if not isinstance(retained_until, str) or retained_until <= now:
                        expired = True
                        conn.execute(
                            """
                            UPDATE undo_retention
                            SET status='expired', expired_at=?
                            WHERE change_set_id=? AND entry_id=?
                            """,
                            (now, change_set_id, entry.entry_id),
                        )
                        conn.execute(
                            """
                            UPDATE blob_references
                            SET state='expired'
                            WHERE owner_type='change_entry' AND owner_id=?
                              AND purpose='undo' AND state='active'
                            """,
                            (f"{change_set_id}:{entry.entry_id}",),
                        )
                        continue
                    applied = conn.execute(
                        """
                        SELECT applied_sha256, applied_identity_json,
                               applied_metadata_sha256, applied_at
                        FROM applied_change_entries
                        WHERE change_set_id=? AND entry_id=?
                        """,
                        (change_set_id, entry.entry_id),
                    ).fetchone()
                    if applied is None:
                        raise UndoConflict("applied_change_record_missing")
                    before_content: bytes | None = None
                    if entry.operation != "add":
                        if entry.before_blob_id is None:
                            raise UndoUnavailable("undo_not_retained")
                        reference = conn.execute(
                            """
                            SELECT state FROM blob_references
                            WHERE blob_id=? AND owner_type='change_entry'
                              AND owner_id=? AND purpose='undo'
                            """,
                            (
                                entry.before_blob_id,
                                f"{change_set_id}:{entry.entry_id}",
                            ),
                        ).fetchone()
                        blob = conn.execute(
                            "SELECT content FROM change_blobs WHERE id=?",
                            (entry.before_blob_id,),
                        ).fetchone()
                        if (
                            reference is None
                            or str(reference["state"]) != "active"
                            or blob is None
                        ):
                            raise UndoUnavailable("undo_not_retained")
                        before_content = bytes(blob["content"])
                    material.append(
                        {
                            "entry": entry,
                            "applied": AppliedChangeRecord(
                                change_set_id=change_set_id,
                                entry_id=entry.entry_id,
                                applied_sha256=applied["applied_sha256"],
                                applied_identity=_identity_from_json(
                                    applied["applied_identity_json"]
                                ),
                                applied_metadata_sha256=applied[
                                    "applied_metadata_sha256"
                                ],
                                applied_at=str(applied["applied_at"]),
                            ),
                            "before_content": before_content,
                            "retained_until": retained_until,
                        }
                    )
                conn.commit()
            if expired:
                raise UndoUnavailable("undo_retention_expired")
            return {
                "manifest": manifest,
                "envelope": loaded["envelope"],
                "entries": tuple(material),
            }

        return await asyncio.to_thread(operation)

    async def consume_undo_retention(
        self,
        *,
        change_set_id: str,
        entry_ids: tuple[str, ...],
    ) -> None:
        """Mark authoritative material consumed and release its blob refs."""

        await self._initialize()

        def operation() -> None:
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                for entry_id in entry_ids:
                    cursor = conn.execute(
                        """
                        UPDATE undo_retention
                        SET status='undone'
                        WHERE change_set_id=? AND entry_id=? AND status='retained'
                        """,
                        (change_set_id, entry_id),
                    )
                    if cursor.rowcount != 1:
                        raise UndoConflict("undo_retention_changed")
                    conn.execute(
                        """
                        UPDATE blob_references
                        SET state='released'
                        WHERE owner_type='change_entry' AND owner_id=?
                          AND purpose='undo' AND state='active'
                        """,
                        (f"{change_set_id}:{entry_id}",),
                    )
                conn.commit()

        await asyncio.to_thread(operation)
    async def set_undo_retention(
        self,
        *,
        change_set_id: str,
        entry_id: str,
        status: str,
        retained_until: str | None,
        expired_at: str | None = None,
    ) -> None:
        await self._initialize()

        def operation() -> None:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO undo_retention(
                        change_set_id, entry_id, status, retained_until, expired_at
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(change_set_id, entry_id)
                    DO UPDATE SET status=excluded.status,
                                  retained_until=excluded.retained_until,
                                  expired_at=excluded.expired_at
                    """,
                    (
                        change_set_id,
                        entry_id,
                        status,
                        retained_until,
                        expired_at,
                    ),
                )
                conn.commit()

        await asyncio.to_thread(operation)

    async def record_applied_entry(
        self,
        *,
        change_set_id: str,
        entry_id: str,
        applied_sha256: str | None,
        applied_identity: FileIdentity | None,
        applied_metadata_sha256: str | None,
        applied_at: str | None = None,
    ) -> None:
        await self._initialize()
        timestamp = applied_at or _now_iso()

        def operation() -> None:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO applied_change_entries(
                        change_set_id, entry_id, applied_sha256,
                        applied_identity_json, applied_metadata_sha256, applied_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(change_set_id, entry_id)
                    DO UPDATE SET applied_sha256=excluded.applied_sha256,
                                  applied_identity_json=excluded.applied_identity_json,
                                  applied_metadata_sha256=excluded.applied_metadata_sha256,
                                  applied_at=excluded.applied_at
                    """,
                    (
                        change_set_id,
                        entry_id,
                        applied_sha256,
                        _identity_json(applied_identity),
                        applied_metadata_sha256,
                        timestamp,
                    ),
                )
                conn.commit()

        await asyncio.to_thread(operation)

    async def mark_change_set_status(
        self,
        change_set_id: str,
        *,
        status: str,
        applied_at: str | None = None,
    ) -> None:
        await self._initialize()
        timestamp = _now_iso()

        def operation() -> None:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    UPDATE change_sets
                    SET status=?, applied_at=?, updated_at=?
                    WHERE id=?
                    """,
                    (
                        status,
                        (
                            applied_at or timestamp
                            if status == "applied"
                            else None
                        ),
                        timestamp,
                        change_set_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise KeyError(change_set_id)
                conn.commit()

        await asyncio.to_thread(operation)

    async def release_blob_reference(
        self,
        *,
        blob_id: str,
        owner_type: str,
        owner_id: str,
        purpose: str,
    ) -> bool:
        await self._initialize()

        def operation() -> bool:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    UPDATE blob_references
                    SET state='released'
                    WHERE blob_id=? AND owner_type=? AND owner_id=?
                      AND purpose=? AND state='active'
                    """,
                    (blob_id, owner_type, owner_id, purpose),
                )
                conn.commit()
                return cursor.rowcount == 1

        return await asyncio.to_thread(operation)

    async def create_journal(
        self,
        *,
        journal_id: str,
        change_set_id: str,
        entries: tuple[JournalEntryRecord, ...],
    ) -> None:
        if not entries:
            raise ValueError("journal entries must not be empty")
        if len({entry.entry_id for entry in entries}) != len(entries):
            raise ValueError("journal entry ids must be unique")
        if len({entry.ordinal for entry in entries}) != len(entries):
            raise ValueError("journal ordinals must be unique")
        await self._initialize()

        def operation() -> None:
            now = _now_iso()
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                if conn.execute(
                    "SELECT 1 FROM change_sets WHERE id=?",
                    (change_set_id,),
                ).fetchone() is None:
                    raise KeyError(change_set_id)
                existing = conn.execute(
                    "SELECT change_set_id FROM file_transaction_journal WHERE id=?",
                    (journal_id,),
                ).fetchone()
                if existing is not None:
                    raise ChangeSetConflict(
                        "journal id already exists; recover it instead of replaying"
                    )
                conn.execute(
                    """
                    INSERT INTO file_transaction_journal(
                        id, change_set_id, status, phase, error,
                        created_at, updated_at
                    ) VALUES (?, ?, 'pending', 'staged', NULL, ?, ?)
                    """,
                    (journal_id, change_set_id, now, now),
                )
                for entry in sorted(entries, key=lambda item: item.ordinal):
                    conn.execute(
                        """
                        INSERT INTO file_transaction_entries(
                            journal_id, entry_id, ordinal, state,
                            base_sha256, after_sha256, base_identity_json,
                            staged_name, staged_identity_json, rollback_blob_id,
                            rollback_staged_name, rollback_staged_identity_json,
                            rollback_successor_identity_json,
                            base_metadata_blob_id, last_error, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            journal_id,
                            entry.entry_id,
                            entry.ordinal,
                            entry.state,
                            entry.base_sha256,
                            entry.after_sha256,
                            _identity_json(entry.base_identity),
                            entry.staged_name,
                            _identity_json(entry.staged_identity),
                            entry.rollback_blob_id,
                            entry.rollback_staged_name,
                            _identity_json(entry.rollback_staged_identity),
                            _identity_json(entry.rollback_successor_identity),
                            entry.base_metadata_blob_id,
                            entry.last_error,
                            now,
                        ),
                    )
                    owner_id = f"{journal_id}:{entry.entry_id}"
                    for blob_id, purpose in (
                        (entry.rollback_blob_id, "rollback"),
                        (entry.base_metadata_blob_id, "base_metadata"),
                    ):
                        if blob_id is None:
                            continue
                        if conn.execute(
                            "SELECT 1 FROM change_blobs WHERE id=?",
                            (blob_id,),
                        ).fetchone() is None:
                            raise BlobNotFound(blob_id)
                        conn.execute(
                            """
                            INSERT INTO blob_references(
                                blob_id, owner_type, owner_id, purpose,
                                retained_until, state
                            ) VALUES (?, 'file_transaction', ?, ?, NULL, 'active')
                            """,
                            (blob_id, owner_id, purpose),
                        )
                conn.commit()

        await asyncio.to_thread(operation)

    async def update_journal_entry(
        self,
        journal_id: str,
        entry_id: str,
        *,
        state: JournalEntryState | None = None,
        staged_name: str | None = None,
        staged_identity: FileIdentity | None = None,
        rollback_blob_id: str | None = None,
        rollback_staged_name: str | None = None,
        rollback_staged_identity: FileIdentity | None = None,
        rollback_successor_identity: FileIdentity | None = None,
        base_metadata_blob_id: str | None = None,
        last_error: str | None = None,
    ) -> JournalEntryRecord:
        await self._initialize()

        def operation() -> JournalEntryRecord:
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    """
                    SELECT * FROM file_transaction_entries
                    WHERE journal_id=? AND entry_id=?
                    """,
                    (journal_id, entry_id),
                ).fetchone()
                if row is None:
                    raise KeyError((journal_id, entry_id))
                current = _journal_entry_from_row(row)
                updated = replace(
                    current,
                    state=current.state if state is None else state,
                    staged_name=(
                        current.staged_name if staged_name is None else staged_name
                    ),
                    staged_identity=(
                        current.staged_identity
                        if staged_identity is None
                        else staged_identity
                    ),
                    rollback_blob_id=(
                        current.rollback_blob_id
                        if rollback_blob_id is None
                        else rollback_blob_id
                    ),
                    rollback_staged_name=(
                        current.rollback_staged_name
                        if rollback_staged_name is None
                        else rollback_staged_name
                    ),
                    rollback_staged_identity=(
                        current.rollback_staged_identity
                        if rollback_staged_identity is None
                        else rollback_staged_identity
                    ),
                    rollback_successor_identity=(
                        current.rollback_successor_identity
                        if rollback_successor_identity is None
                        else rollback_successor_identity
                    ),
                    base_metadata_blob_id=(
                        current.base_metadata_blob_id
                        if base_metadata_blob_id is None
                        else base_metadata_blob_id
                    ),
                    last_error=(
                        current.last_error if last_error is None else last_error
                    ),
                )
                now = _now_iso()
                conn.execute(
                    """
                    UPDATE file_transaction_entries
                    SET state=?, staged_name=?, staged_identity_json=?,
                        rollback_blob_id=?, rollback_staged_name=?,
                        rollback_staged_identity_json=?,
                        rollback_successor_identity_json=?,
                        base_metadata_blob_id=?, last_error=?, updated_at=?
                    WHERE journal_id=? AND entry_id=?
                    """,
                    (
                        updated.state,
                        updated.staged_name,
                        _identity_json(updated.staged_identity),
                        updated.rollback_blob_id,
                        updated.rollback_staged_name,
                        _identity_json(updated.rollback_staged_identity),
                        _identity_json(updated.rollback_successor_identity),
                        updated.base_metadata_blob_id,
                        updated.last_error,
                        now,
                        journal_id,
                        entry_id,
                    ),
                )
                conn.execute(
                    """
                    UPDATE file_transaction_journal
                    SET updated_at=?
                    WHERE id=?
                    """,
                    (now, journal_id),
                )
                conn.commit()
                return updated

        return await asyncio.to_thread(operation)

    async def update_journal(
        self,
        journal_id: str,
        *,
        status: str | None = None,
        phase: str | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        await self._initialize()

        def operation() -> dict[str, Any]:
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT status, phase, error FROM file_transaction_journal WHERE id=?",
                    (journal_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(journal_id)
                conn.execute(
                    """
                    UPDATE file_transaction_journal
                    SET status=?, phase=?, error=?, updated_at=?
                    WHERE id=?
                    """,
                    (
                        str(row["status"]) if status is None else status,
                        str(row["phase"]) if phase is None else phase,
                        row["error"] if error is None else error,
                        _now_iso(),
                        journal_id,
                    ),
                )
                loaded = self._load_journal(conn, journal_id)
                if loaded is None:
                    raise RuntimeError("journal disappeared during update")
                conn.commit()
                return loaded

        return await asyncio.to_thread(operation)

    async def finalize_journal(
        self,
        journal_id: str,
        *,
        change_set_id: str,
        status: str,
        phase: str,
        error: str | None,
        release_references: bool,
    ) -> dict[str, Any]:
        """Atomically persist matching change-set/journal terminal state."""

        if status not in self._TERMINAL_JOURNAL_STATUSES:
            raise ValueError("journal finalization requires a terminal status")
        await self._initialize()

        def operation() -> dict[str, Any]:
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    """
                    SELECT change_set_id, status, phase, error
                    FROM file_transaction_journal
                    WHERE id=?
                    """,
                    (journal_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(journal_id)
                if str(row["change_set_id"]) != change_set_id:
                    raise ChangeSetConflict(
                        "journal belongs to another change set"
                    )
                current_status = str(row["status"])
                if current_status in self._TERMINAL_JOURNAL_STATUSES:
                    if (
                        current_status != status
                        or str(row["phase"]) != phase
                        or row["error"] != error
                    ):
                        raise ChangeSetConflict(
                            "terminal journal state is immutable"
                        )
                    loaded = self._load_journal(conn, journal_id)
                    if loaded is None:
                        raise RuntimeError(
                            "terminal journal disappeared during finalize"
                        )
                    conn.commit()
                    return loaded

                now = _now_iso()
                cursor = conn.execute(
                    """
                    UPDATE change_sets
                    SET status=?, applied_at=?, updated_at=?
                    WHERE id=?
                    """,
                    (
                        status,
                        now if status == "applied" else None,
                        now,
                        change_set_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise KeyError(change_set_id)
                conn.execute(
                    """
                    UPDATE file_transaction_journal
                    SET status=?, phase=?, error=?, updated_at=?
                    WHERE id=?
                    """,
                    (status, phase, error, now, journal_id),
                )
                if release_references:
                    entries = conn.execute(
                        """
                        SELECT entry_id, rollback_blob_id,
                               base_metadata_blob_id
                        FROM file_transaction_entries
                        WHERE journal_id=?
                        """,
                        (journal_id,),
                    ).fetchall()
                    for entry in entries:
                        owner_id = f"{journal_id}:{entry['entry_id']}"
                        for blob_id, purpose in (
                            (entry["rollback_blob_id"], "rollback"),
                            (
                                entry["base_metadata_blob_id"],
                                "base_metadata",
                            ),
                        ):
                            if blob_id is None:
                                continue
                            conn.execute(
                                """
                                UPDATE blob_references
                                SET state='released'
                                WHERE blob_id=?
                                  AND owner_type='file_transaction'
                                  AND owner_id=?
                                  AND purpose=?
                                  AND state='active'
                                """,
                                (blob_id, owner_id, purpose),
                            )
                loaded = self._load_journal(conn, journal_id)
                if loaded is None:
                    raise RuntimeError(
                        "journal disappeared during finalization"
                    )
                conn.commit()
                return loaded

        return await asyncio.to_thread(operation)

    def _load_journal(
        self,
        conn: sqlite3.Connection,
        journal_id: str,
    ) -> dict[str, Any] | None:
        row = conn.execute(
            "SELECT * FROM file_transaction_journal WHERE id=?",
            (journal_id,),
        ).fetchone()
        if row is None:
            return None
        entries = tuple(
            _journal_entry_from_row(entry)
            for entry in conn.execute(
                """
                SELECT * FROM file_transaction_entries
                WHERE journal_id=?
                ORDER BY ordinal
                """,
                (journal_id,),
            ).fetchall()
        )
        return {
            "id": str(row["id"]),
            "change_set_id": str(row["change_set_id"]),
            "status": str(row["status"]),
            "phase": str(row["phase"]),
            "error": row["error"],
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
            "entries": entries,
        }

    async def get_journal(self, journal_id: str) -> dict[str, Any] | None:
        await self._initialize()

        def operation() -> dict[str, Any] | None:
            with self._connect() as conn:
                return self._load_journal(conn, journal_id)

        return await asyncio.to_thread(operation)

    async def list_incomplete_journals(self) -> tuple[dict[str, Any], ...]:
        await self._initialize()

        def operation() -> tuple[dict[str, Any], ...]:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT id
                    FROM file_transaction_journal
                    WHERE status NOT IN ('applied', 'rolled_back',
                                         'rollback_failed', 'interference')
                    ORDER BY created_at, id
                    """
                ).fetchall()
                loaded = tuple(
                    self._load_journal(conn, str(row["id"]))
                    for row in rows
                )
                return tuple(item for item in loaded if item is not None)

        return await asyncio.to_thread(operation)


__all__ = [
    "BlobNotFound",
    "ChangeSetConflict",
    "ChangeSetStore",
    "JournalEntryRecord",
    "JournalEntryState",
    "UndoConflict",
    "UndoUnavailable",
]
