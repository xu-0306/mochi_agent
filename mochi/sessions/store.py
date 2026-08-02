"""會話 JSONL 持久化儲存 — Phase 2 完整實作。"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import inspect
import json
import logging
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Condition, RLock
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from mochi.config import defaults

logger = logging.getLogger(__name__)


def canonical_sessions_dir(value: str | os.PathLike[str]) -> str:
    """Return the process-local canonical identity of a sessions root."""

    expanded = Path(value).expanduser()
    try:
        resolved = expanded.resolve(strict=False)
    except OSError:
        resolved = Path(os.path.abspath(os.fspath(expanded)))
    return os.path.normcase(os.path.normpath(os.fspath(resolved)))


class SessionsDirectoryRestartRequired(RuntimeError):
    """Raised when a live runtime is asked to switch its durable session root."""

    code = "sessions_dir_restart_required"

    def __init__(
        self,
        current: str | os.PathLike[str],
        requested: str | os.PathLike[str],
    ) -> None:
        self.current_root = canonical_sessions_dir(current)
        self.requested_root = canonical_sessions_dir(requested)
        super().__init__(
            "Changing sessions_dir requires restarting the application; "
            "the current runtime binding was preserved."
        )


def ensure_sessions_dir_unchanged(
    current: str | os.PathLike[str],
    requested: str | os.PathLike[str],
) -> None:
    """Reject a live sessions-root switch before any runtime mutation."""

    if canonical_sessions_dir(current) != canonical_sessions_dir(requested):
        raise SessionsDirectoryRestartRequired(current, requested)


# ``~`` is deliberately outside the old sanitiser's allowed alphabet.  The
# v2 filename is a *lowercase* SHA-256 slot, so it remains distinct on the
# default case-insensitive Windows filesystem.  The exact session identity is
# stored in a neighboring durable sidecar and verified before a slot is used.
_SESSION_FILENAME_V2_PREFIX = "~sid-v2-"
_SESSION_FILENAME_V1_PREFIX = "~sid-v1-"
_SESSION_IDENTITY_SUFFIX = ".identity.json"
_SESSION_IDENTITY_VERSION = 2
_MAX_SESSION_HASH_COLLISION_SLOTS = 1024
_MAX_COMPAT_FILENAME_CHARS = 240
_STORAGE_MARKER_FILENAME = ".mochi-storage.json"
_STORAGE_MARKER_VERSION = 1
_STORAGE_ID_RE = re.compile(r"^storage:v1:[0-9a-f]{32}$")
_ATOMIC_REPLACE_MAX_ATTEMPTS = 7
_ATOMIC_REPLACE_RETRY_BASE_SECONDS = 0.01

_TOOL_WORKFLOW_GATE_CONTROLLED_EVENTS = frozenset(
    {
        "tool_workflow_approval_observation",
        "tool_workflow_aggregate_outbox",
    }
)


def _replace_with_transient_lock_retry(source: Path, target: Path) -> None:
    """Replace a file after bounded retries for transient sharing violations.

    Windows rejects ``os.replace`` while another request briefly has the
    destination JSONL open for reading. Session reads are intentionally
    concurrent, so tolerate that short-lived condition without weakening the
    atomic replacement contract or swallowing a persistent permission error.
    """

    delay = _ATOMIC_REPLACE_RETRY_BASE_SECONDS
    for attempt in range(_ATOMIC_REPLACE_MAX_ATTEMPTS):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == _ATOMIC_REPLACE_MAX_ATTEMPTS - 1:
                raise
            time.sleep(delay)
            delay *= 2


class StrictSessionSnapshotError(ValueError):
    """The durable JSONL history cannot safely be used for a strict transition."""


class SessionIdentityConflictError(StrictSessionSnapshotError):
    """A legacy filename cannot be safely attributed to the requested ID.

    Old SessionStore versions replaced every non ``[A-Za-z0-9._-]`` character
    with ``_``.  The resulting filename is not an identity: ``a:b`` and
    ``a?b`` both become ``a_b`` on Windows.  Callers must not silently read or
    overwrite that file on behalf of the wrong session.
    """


class StorageIdentityError(RuntimeError):
    """The durable sessions-root identity marker is malformed or unsupported."""


class ToolWorkflowPublicationGate:
    """One live rollout gate shared by old and replacement SessionStores.

    A mutation leases this gate before its session sidecar lock while deciding
    whether to append a companion record.  Disabling first blocks new leases,
    then waits for leases already in flight.  This makes ``set_enabled(False)``
    a rollback barrier without serializing unrelated session writes.
    """

    def __init__(self, enabled: bool = False) -> None:
        self._enabled = bool(enabled)
        self._condition = Condition(RLock())
        self._active_publishers = 0

    @property
    def enabled(self) -> bool:
        with self._condition:
            return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        """Set publication policy, waiting for pre-disable leases to drain."""

        with self._condition:
            self._enabled = bool(enabled)
            if self._enabled:
                return
            while self._active_publishers:
                self._condition.wait()

    async def set_enabled_async(self, enabled: bool) -> None:
        """Run the potentially blocking disable barrier off the event loop."""

        await asyncio.to_thread(self.set_enabled, enabled)

    @contextmanager
    def publication_transaction(self):
        """Lease publication for one strict mutation without global I/O locking."""

        with self._condition:
            publication_enabled = self._enabled
            if publication_enabled:
                self._active_publishers += 1
        try:
            yield publication_enabled
        finally:
            if publication_enabled:
                with self._condition:
                    self._active_publishers -= 1
                    if self._active_publishers == 0:
                        self._condition.notify_all()


@contextmanager
def _static_publication_transaction(enabled: bool):
    """Match the live gate interface for standalone SessionStore instances."""

    yield bool(enabled)


@dataclass(frozen=True)
class DurableSessionSnapshot:
    """An immutable, fully validated point-in-time session history.

    ``history_revision`` is a content hash of the canonical ordered event
    sequence.  It is deliberately opaque: callers must only use it as a CAS
    token, never infer ordering from it.
    """

    session_id: str
    events: tuple[Mapping[str, Any], ...]
    history_revision: str
    event_count: int
    exists: bool


@dataclass(frozen=True)
class StrictSessionAppendResult:
    """Result of a short strict-history CAS mutation."""

    status: str
    before: DurableSessionSnapshot
    after: DurableSessionSnapshot
    new_outbox_start_position: int | None = None


@dataclass(frozen=True)
class SessionStoreInventoryPage:
    """One bounded page from the best-effort maintenance inventory."""

    session_ids: tuple[str, ...]
    next_cursor: str | None


class SessionStore:
    """會話儲存（JSONL 格式，Append-only）。"""

    def __init__(
        self,
        sessions_dir: str | Path = defaults.default_sessions_dir(),
        *,
        tool_observability_v1: bool = False,
        tool_workflow_publication_gate: ToolWorkflowPublicationGate | None = None,
        post_strict_commit_observer: Callable[[DurableSessionSnapshot, int], Any] | None = None,
    ) -> None:
        """初始化 SessionStore。

        Args:
            sessions_dir: 會話檔案目錄，會自動建立。
        """
        self._sessions_dir = Path(sessions_dir).expanduser()
        self._storage_id = self._load_or_create_storage_id()
        # The hook is intentionally at the SessionStore strict-CAS boundary.
        # It appends only rebuildable aggregate delivery records and never
        # changes a source transition's meaning.
        self._tool_observability_v1 = bool(tool_observability_v1)
        self._tool_workflow_publication_gate = tool_workflow_publication_gate
        self._post_strict_commit_observer = post_strict_commit_observer

    def bind_tool_workflow_publication_gate(
        self,
        gate: ToolWorkflowPublicationGate,
        *,
        post_strict_commit_observer: Callable[[DurableSessionSnapshot, int], Any] | None = None,
    ) -> None:
        """Attach a live gate to an injected store before it is used by runtime.

        A store already bound to another gate cannot safely participate in a
        different engine's rollback barrier, so fail loudly instead of silently
        publishing through an incompatible dependency-injection path.
        """

        if not isinstance(gate, ToolWorkflowPublicationGate):
            raise TypeError("gate must be a ToolWorkflowPublicationGate")
        current = self._tool_workflow_publication_gate
        if current is not None and current is not gate:
            raise ValueError("SessionStore is already bound to a different publication gate")
        self._tool_workflow_publication_gate = gate
        if post_strict_commit_observer is not None:
            current_observer = self._post_strict_commit_observer
            if current_observer is not None and current_observer is not post_strict_commit_observer:
                raise ValueError("SessionStore is already bound to a different post-commit observer")
            self._post_strict_commit_observer = post_strict_commit_observer

    @property
    def tool_observability_v1(self) -> bool:
        gate = self._tool_workflow_publication_gate
        return gate.enabled if gate is not None else self._tool_observability_v1

    @property
    def sessions_dir(self) -> Path:
        """Return the root this store was bound to at construction time."""

        return self._sessions_dir

    @property
    def storage_id(self) -> str:
        """Return the durable identity of this sessions root."""

        return self._storage_id

    def _load_or_create_storage_id(self) -> str:
        marker_path = self._sessions_dir / _STORAGE_MARKER_FILENAME
        with self._sidecar_lock(marker_path):
            if not marker_path.exists():
                storage_id = f"storage:v1:{uuid4().hex}"
                payload = json.dumps(
                    {
                        "schema_version": _STORAGE_MARKER_VERSION,
                        "storage_id": storage_id,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                self._atomic_replace_lines(marker_path, (payload,))
                return storage_id
            try:
                payload = json.loads(marker_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise StorageIdentityError("sessions storage marker is unreadable") from exc
            if not isinstance(payload, dict) or set(payload) != {"schema_version", "storage_id"}:
                raise StorageIdentityError("sessions storage marker has unsupported fields")
            if payload.get("schema_version") != _STORAGE_MARKER_VERSION:
                raise StorageIdentityError("sessions storage marker schema is unsupported")
            storage_id = payload.get("storage_id")
            if not isinstance(storage_id, str) or _STORAGE_ID_RE.fullmatch(storage_id) is None:
                raise StorageIdentityError("sessions storage marker identity is invalid")
            return storage_id

    async def save_event(self, session_id: str, event: dict) -> None:
        """將事件追加寫入 JSONL 檔案。"""
        if not isinstance(event, dict):
            raise TypeError("event must be a dict.")

        if self._tool_observability_v1 or self._tool_workflow_publication_gate is not None:
            # Source records participating in aggregate reduction must use the
            # same strict CAS batch as their companion outbox cache record.
            for _ in range(8):
                snapshot = await self.load_strict_snapshot(session_id)
                result = await self.append_strict_batch_if_revision(
                    session_id,
                    expected_history_revision=snapshot.history_revision,
                    events=(event,),
                )
                if result.status != "rebase_required":
                    if result.status != "appended":  # pragma: no cover - defensive invariant.
                        raise StrictSessionSnapshotError("strict session event append was not committed")
                    return
            raise StrictSessionSnapshotError("strict session event append repeatedly lost its CAS")

        sid = self._normalized_session_id(session_id)
        line = json.dumps(event, ensure_ascii=False)
        # A verified legacy file is migrated with ``os.replace`` before its
        # first write.  The v1 filename itself then carries the opaque,
        # collision-free identity even when the event payload has no
        # ``session_id`` field.
        path = await asyncio.to_thread(self._prepare_writer_path, sid)
        await asyncio.to_thread(self._append_line, path, line)

    async def load_strict_snapshot(self, session_id: str) -> DurableSessionSnapshot:
        """Read one immutable durable snapshot without legacy recovery.

        ``load_session`` remains intentionally permissive for backwards
        compatibility.  New durable state machines must use this API instead:
        an existing empty file, invalid UTF-8/JSON, non-object line, or an
        event which explicitly names a different session raises
        :class:`StrictSessionSnapshotError`.
        """
        sid = self._normalized_session_id(session_id)
        path = await asyncio.to_thread(self._resolve_existing_path, sid)
        return await asyncio.to_thread(self._read_strict_snapshot_locked, path, sid)

    async def append_strict_batch_if_revision(
        self,
        session_id: str,
        *,
        expected_history_revision: str,
        events: Sequence[Mapping[str, Any]],
    ) -> StrictSessionAppendResult:
        """Append a complete strict batch only when the history token matches.

        The check, strict validation, write, fsync, and snapshot construction
        run beneath the session sidecar lock.  The implementation atomically
        replaces the JSONL file instead of streaming a multi-line append, so a
        failed batch is never partially visible.
        """
        return await self.mutate_strict_snapshot(
            session_id,
            expected_history_revision=expected_history_revision,
            build_events=lambda _snapshot: events,
        )

    async def mutate_strict_snapshot(
        self,
        session_id: str,
        *,
        expected_history_revision: str,
        build_events: Callable[[DurableSessionSnapshot], Sequence[Mapping[str, Any]] | None],
    ) -> StrictSessionAppendResult:
        """Run an in-memory state transition and optional batch append under CAS.

        ``build_events`` executes while the sidecar lock is held.  It must be a
        pure, bounded in-memory transformation; it must not await or invoke
        model, tool, network, or external-process work.  Returning ``None``
        makes no durable change and returns ``unchanged``.
        """
        sid = self._normalized_session_id(session_id)
        if not isinstance(expected_history_revision, str) or not expected_history_revision:
            raise ValueError("expected_history_revision must be a non-empty string.")
        if not callable(build_events):
            raise TypeError("build_events must be callable.")
        path = await asyncio.to_thread(self._prepare_writer_path, sid)
        result = await asyncio.to_thread(
            self._mutate_strict_snapshot,
            path,
            sid,
            expected_history_revision,
            build_events,
        )
        if (
            result.status == "appended"
            and result.new_outbox_start_position is not None
            and self._post_strict_commit_observer is not None
        ):
            try:
                # Runs after the sidecar lock is released, against the exact
                # immutable post-commit snapshot returned by this mutation.
                observation = self._post_strict_commit_observer(
                    result.after,
                    result.new_outbox_start_position,
                )
                if inspect.isawaitable(observation):
                    await observation
            except Exception as exc:
                # Diagnostics must never alter the source commit result.
                logger.warning(
                    "Tool-workflow post-commit observer failed: %s",
                    type(exc).__name__,
                )
        return result

    async def append_event_if(
        self,
        session_id: str,
        event: dict,
        predicate: Callable[[list[dict]], bool],
    ) -> bool:
        """Atomically append *event* when ``predicate`` accepts current events.

        Session metadata used to be protected only by repository-local asyncio
        locks.  Those locks disappear when another Engine or worker constructs
        its own repository, so a read/check/append race could persist two
        snapshots with the same revision.  The sidecar lock is deliberately at
        the SessionStore boundary: it covers every repository instance sharing
        the same durable session directory without holding an asyncio lock over
        model or tool execution.
        """
        if not isinstance(event, dict):
            raise TypeError("event must be a dict.")
        if not callable(predicate):
            raise TypeError("predicate must be callable.")
        sid = self._normalized_session_id(session_id)
        path = await asyncio.to_thread(self._prepare_writer_path, sid)
        return await asyncio.to_thread(
            self._append_line_if,
            path,
            event,
            predicate,
        )

    async def load_session(self, session_id: str) -> list[dict]:
        """從 JSONL 載入完整會話，遇到壞資料時跳過該行。"""
        sid = self._normalized_session_id(session_id)
        path = await asyncio.to_thread(self._resolve_existing_path, sid)
        return await asyncio.to_thread(self._load_lines, path)

    async def session_exists(self, session_id: str) -> bool:
        """檢查 session 檔案是否存在。"""
        sid = self._normalized_session_id(session_id)
        path = await asyncio.to_thread(self._resolve_existing_path, sid)
        return await asyncio.to_thread(path.exists)

    async def session_last_modified(self, session_id: str) -> float | None:
        """Return the durable session file mtime without exposing filename layout.

        Session routes use this for display ordering.  Keeping it behind the
        store prevents encoded v1 filenames from leaking back into route-level
        session identity handling.
        """

        sid = self._normalized_session_id(session_id)
        return await asyncio.to_thread(self._session_last_modified, sid)

    async def list_session_ids(self, *, limit: int | None = None) -> tuple[str, ...]:
        """Return a bounded best-effort inventory for read-only maintenance.

        This is deliberately an inventory, not an authority API: a damaged or
        legacy JSONL file can be skipped by later strict verification.  When a
        durable envelope supplies the original session ID, prefer it over the
        filesystem-safe filename used by legacy storage.
        """

        if limit is not None and (type(limit) is not int or limit <= 0):
            raise ValueError("limit must be a positive integer or None")
        return await asyncio.to_thread(self._list_session_ids, limit)

    async def list_session_ids_page(
        self,
        *,
        limit: int,
        after: str | None = None,
    ) -> SessionStoreInventoryPage:
        """Return one page so a restart audit can progress beyond early files."""

        if type(limit) is not int or limit <= 0:
            raise ValueError("limit must be a positive integer")
        if after is not None and (not isinstance(after, str) or not after):
            raise ValueError("after must be a non-empty cursor string or None")
        return await asyncio.to_thread(self._list_session_ids_page, limit, after)

    async def delete_session(self, session_id: str) -> bool:
        """刪除 session 檔案；不存在時回傳 False。"""
        sid = self._normalized_session_id(session_id)
        path = await asyncio.to_thread(self._resolve_existing_path, sid)
        return await asyncio.to_thread(self._delete_file, path, sid)

    async def replace_session(self, session_id: str, events: list[dict]) -> None:
        """Atomically replace one session file with the provided ordered events."""
        if not isinstance(events, list):
            raise TypeError("events must be a list.")
        if any(not isinstance(event, dict) for event in events):
            raise TypeError("every event must be a dict.")

        sid = self._normalized_session_id(session_id)
        path = await asyncio.to_thread(self._prepare_writer_path, sid)
        await asyncio.to_thread(self._write_lines, path, events)

    def _session_path(self, session_id: str) -> Path:
        """Return the primary fixed-length, case-insensitive-safe v2 path.

        V2 stores a lowercase SHA-256 slot in the filename and the exact ID in
        a small adjacent identity file.  That avoids Windows case-fold
        collisions from Base64 and avoids component-length failures for long
        IDs.  Public operations resolve a verified collision slot; this helper
        intentionally returns the primary deterministic slot for diagnostics
        and tests only.
        """
        sid = self._normalized_session_id(session_id)
        return self._v2_candidate_path(sid, slot=0)

    def _v2_candidate_path(self, session_id: str, *, slot: int) -> Path:
        sid = self._normalized_session_id(session_id)
        if type(slot) is not int or slot < 0:
            raise ValueError("session identity slot must be a non-negative integer")
        digest = self._session_id_digest(sid)
        suffix = "" if slot == 0 else f"-{slot}"
        return self._sessions_dir / f"{_SESSION_FILENAME_V2_PREFIX}{digest}{suffix}.jsonl"

    @staticmethod
    def _session_id_digest(session_id: str) -> str:
        """Return the fixed lowercase slot digest for an exact normalized ID."""

        sid = SessionStore._normalized_session_id(session_id)
        return hashlib.sha256(sid.encode("utf-8")).hexdigest()

    @staticmethod
    def _v2_identity_path(path: Path) -> Path:
        return path.with_suffix(_SESSION_IDENTITY_SUFFIX)

    @staticmethod
    def _is_v2_path(path: Path) -> bool:
        return path.name.startswith(_SESSION_FILENAME_V2_PREFIX) and path.name.endswith(".jsonl")

    def _v2_path_matches_session_id(self, path: Path, session_id: str) -> bool:
        sid = self._normalized_session_id(session_id)
        base_name = f"{_SESSION_FILENAME_V2_PREFIX}{self._session_id_digest(sid)}"
        stem = path.stem
        if stem == base_name:
            return True
        suffix = stem.removeprefix(base_name)
        return suffix.startswith("-") and suffix[1:].isdigit() and int(suffix[1:]) > 0

    def _read_v2_identity(self, path: Path) -> str:
        identity_path = self._v2_identity_path(path)
        try:
            raw = identity_path.read_text(encoding="utf-8")
            value = json.loads(raw)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SessionIdentityConflictError(
                "cannot read v2 session identity sidecar"
            ) from exc
        if (
            not isinstance(value, dict)
            or set(value) != {"schema_version", "session_id"}
            or value.get("schema_version") != _SESSION_IDENTITY_VERSION
        ):
            raise SessionIdentityConflictError("v2 session identity sidecar is malformed")
        try:
            session_id = self._normalized_session_id(value.get("session_id"))
        except (TypeError, ValueError) as exc:
            raise SessionIdentityConflictError("v2 session identity sidecar is invalid") from exc
        if not self._v2_path_matches_session_id(path, session_id):
            raise SessionIdentityConflictError("v2 session identity sidecar does not match its hash slot")
        return session_id

    def _write_v2_identity(self, path: Path, session_id: str) -> None:
        sid = self._normalized_session_id(session_id)
        identity_path = self._v2_identity_path(path)
        payload = json.dumps(
            {"schema_version": _SESSION_IDENTITY_VERSION, "session_id": sid},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        tmp_path = identity_path.with_name(f"{identity_path.name}.{uuid4().hex}.tmp")
        try:
            with tmp_path.open("w", encoding="utf-8", newline="\n") as fh:
                fh.write(payload)
                fh.write("\n")
                fh.flush()
                os.fsync(fh.fileno())
            _replace_with_transient_lock_retry(tmp_path, identity_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def _find_v2_path(self, session_id: str) -> Path | None:
        """Return the verified v2 collision slot, without creating one."""

        sid = self._normalized_session_id(session_id)
        for slot in range(_MAX_SESSION_HASH_COLLISION_SLOTS):
            path = self._v2_candidate_path(sid, slot=slot)
            identity_path = self._v2_identity_path(path)
            if identity_path.exists():
                actual = self._read_v2_identity(path)
                if actual == sid:
                    return path
                continue
            if path.exists():
                raise SessionIdentityConflictError(
                    "v2 session data exists without an identity sidecar"
                )
            return None
        raise SessionIdentityConflictError("session identity hash collision slots are exhausted")

    def _ensure_v2_path(self, session_id: str) -> Path:
        """Reserve and return a verified v2 collision slot under its lock."""

        sid = self._normalized_session_id(session_id)
        for slot in range(_MAX_SESSION_HASH_COLLISION_SLOTS):
            path = self._v2_candidate_path(sid, slot=slot)
            identity_path = self._v2_identity_path(path)
            with self._sidecar_lock(path):
                if identity_path.exists():
                    actual = self._read_v2_identity(path)
                    if actual == sid:
                        return path
                    continue
                if path.exists():
                    raise SessionIdentityConflictError(
                        "v2 session data exists without an identity sidecar"
                    )
                # The identity reservation is written before the JSONL file.
                # An interrupted first write may leave only this sidecar, which
                # is harmless and safely reusable by the same session ID.
                self._write_v2_identity(path, sid)
                return path
        raise SessionIdentityConflictError("session identity hash collision slots are exhausted")

    def _v1_session_path(self, session_id: str) -> Path:
        """Return the short-lived Base64 v1 path for safe forward migration."""

        sid = self._normalized_session_id(session_id)
        encoded = base64.urlsafe_b64encode(sid.encode("utf-8")).decode("ascii").rstrip("=")
        return self._sessions_dir / f"{_SESSION_FILENAME_V1_PREFIX}{encoded}.jsonl"

    def _pre_v2_compatibility_paths(self, session_id: str) -> tuple[Path, ...]:
        """Return only old paths that Windows can safely address and lock."""

        candidates = (self._v1_session_path(session_id), self._legacy_session_path(session_id))
        return tuple(path for path in candidates if len(path.name) <= _MAX_COMPAT_FILENAME_CHARS)

    def _assert_v1_path_owned_by(self, path: Path, session_id: str) -> None:
        """Read v1 only when an envelope disambiguates its case-folded name."""

        sid = self._normalized_session_id(session_id)
        source_ids = self._source_session_ids(path)
        if source_ids == {sid}:
            return
        if not source_ids:
            raise SessionIdentityConflictError(
                "v1 Base64 filename has no durable identity on a case-insensitive filesystem"
            )
        raise SessionIdentityConflictError("v1 Base64 filename is claimed by another session ID")

    def _legacy_session_path(self, session_id: str) -> Path:
        """Return the pre-v1 lossy filename, for verified compatibility reads."""

        sid = self._normalized_session_id(session_id)
        safe_sid = re.sub(r"[^A-Za-z0-9._-]", "_", sid)
        return self._sessions_dir / f"{safe_sid}.jsonl"

    def _resolve_existing_path(self, session_id: str) -> Path:
        """Find an existing v2 or provably-owned migration-reader JSONL file.

        A missing v1 file does not make a legacy filename authoritative.  A
        legacy file is usable only when its durable envelopes consistently
        name the requested ID, or when its filename is itself reversible for
        that ID.  In particular, a file claimed by ``a?b`` must never be read
        as ``a:b`` just because both old names were ``a_b.jsonl``.
        """

        sid = self._normalized_session_id(session_id)
        v2_path = self._find_v2_path(sid)
        if v2_path is not None and v2_path.exists():
            return v2_path
        for path in self._pre_v2_compatibility_paths(sid):
            if not path.exists():
                continue
            if path.name.startswith(_SESSION_FILENAME_V1_PREFIX):
                self._assert_v1_path_owned_by(path, sid)
            else:
                self._assert_legacy_path_owned_by(path, sid)
            return path
        return v2_path if v2_path is not None else self._session_path(sid)

    def _prepare_writer_path(self, session_id: str) -> Path:
        """Return the v2 path, atomically migrating a verified prior source.

        This is an expand-only, on-demand migration reader/writer: untouched
        legacy files remain in place.  The first write of a verified legacy
        session moves its exact JSONL file to a verified v2 identity under both
        sidecar locks.  No lossy filename is ever copied to a different
        claimed session, and no bulk migration or deletion is required.
        """

        sid = self._normalized_session_id(session_id)
        v2_path = self._ensure_v2_path(sid)
        pre_v2_paths = self._pre_v2_compatibility_paths(sid)
        with self._sidecar_locks(v2_path, *pre_v2_paths):
            if v2_path.exists():
                return v2_path
            sources = [path for path in pre_v2_paths if path.exists()]
            if not sources:
                return v2_path
            if len(sources) > 1:
                raise SessionIdentityConflictError(
                    "multiple pre-v2 session files exist; refusing ambiguous migration"
            )
            source_path = sources[0]
            if source_path.name.startswith(_SESSION_FILENAME_V1_PREFIX):
                self._assert_v1_path_owned_by(source_path, sid)
            else:
                self._assert_legacy_path_owned_by(source_path, sid)
            _replace_with_transient_lock_retry(source_path, v2_path)
            return v2_path

    @staticmethod
    def _source_session_ids(path: Path) -> set[str]:
        """Extract explicit source identities from permissively readable JSONL."""

        if not path.exists():
            return set()
        source_ids: set[str] = set()
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                for raw_line in fh:
                    try:
                        event = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue
                    value = event.get("session_id") if isinstance(event, dict) else None
                    if isinstance(value, str) and value.strip():
                        source_ids.add(value.strip())
        except OSError as exc:
            raise SessionIdentityConflictError(
                f"cannot verify legacy session identity: {exc}"
            ) from exc
        return source_ids

    def _assert_legacy_path_owned_by(self, path: Path, session_id: str) -> None:
        """Fail closed unless a legacy path can be attributed to ``session_id``."""

        sid = self._normalized_session_id(session_id)
        source_ids = self._source_session_ids(path)
        if source_ids == {sid}:
            return
        if len(source_ids) == 1:
            actual = next(iter(source_ids))
            raise SessionIdentityConflictError(
                "legacy session filename is claimed by a different session ID "
                f"({actual!r}, requested {sid!r})"
            )
        if len(source_ids) > 1:
            raise SessionIdentityConflictError(
                "legacy session filename contains conflicting explicit session IDs"
            )
        # No envelope identity is still safe for the subset of old IDs whose
        # filename was reversible.  This preserves ordinary legacy sessions
        # without guessing which special character used to occupy an underscore.
        if path.stem == sid and self._path_spelling_is_exact(path):
            return
        raise SessionIdentityConflictError(
            "legacy session filename has no durable identity or exact preserved spelling "
            "for the requested session ID"
        )

    @staticmethod
    def _path_spelling_is_exact(path: Path) -> bool:
        """Verify the directory entry's preserved spelling, not only lookup equality.

        Default Windows volumes compare names case-insensitively while retaining
        the spelling used at creation.  ``Path('ALPHA.jsonl').exists()`` can
        therefore resolve ``alpha.jsonl``.  An identity-free legacy file must
        not be attributed to that differently-cased session ID.
        """

        try:
            matches = [
                candidate.name
                for candidate in path.parent.iterdir()
                if candidate.name.casefold() == path.name.casefold()
            ]
        except OSError as exc:
            raise SessionIdentityConflictError(
                "cannot verify legacy session filename spelling"
            ) from exc
        return matches == [path.name]

    @staticmethod
    def _decode_v1_session_path(path: Path) -> str | None:
        """Decode a reserved v1 filename, returning ``None`` for non-v1 paths."""

        name = path.name
        suffix = ".jsonl"
        if not name.endswith(suffix):
            return None
        encoded = name[: -len(suffix)]
        if not encoded.startswith(_SESSION_FILENAME_V1_PREFIX):
            return None
        payload = encoded.removeprefix(_SESSION_FILENAME_V1_PREFIX)
        if not payload:
            return None
        try:
            padded = payload + "=" * (-len(payload) % 4)
            decoded = base64.b64decode(padded, altchars=b"-_", validate=True).decode("utf-8")
        except (UnicodeDecodeError, ValueError, binascii.Error):
            return None
        # Canonical round-trip rejects alternate padding/spelling forms.
        canonical = base64.urlsafe_b64encode(decoded.encode("utf-8")).decode("ascii").rstrip("=")
        if canonical != payload:
            return None
        try:
            return SessionStore._normalized_session_id(decoded)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalized_session_id(session_id: str) -> str:
        if not isinstance(session_id, str):
            raise TypeError("session_id must be a string.")
        sid = session_id.strip()
        if not sid:
            raise ValueError("session_id must not be empty.")
        return sid

    def _append_line(self, path: Path, line: str) -> None:
        """同步追加寫入單行事件。"""
        with self._sidecar_lock(path):
            with path.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(f"{line}\n")
                fh.flush()
                os.fsync(fh.fileno())

    def _append_line_if(
        self,
        path: Path,
        event: dict,
        predicate: Callable[[list[dict]], bool],
    ) -> bool:
        with self._sidecar_lock(path):
            events = self._load_lines(path)
            if not predicate(events):
                return False
            line = json.dumps(event, ensure_ascii=False)
            with path.open("a", encoding="utf-8", newline="\n") as event_file:
                event_file.write(f"{line}\n")
                event_file.flush()
                os.fsync(event_file.fileno())
            return True

    def _session_last_modified(self, session_id: str) -> float | None:
        path = self._resolve_existing_path(session_id)
        try:
            return path.stat().st_mtime if path.exists() else None
        except OSError:
            return None

    @contextmanager
    def _sidecar_locks(self, *paths: Path):
        """Take multiple path locks in a stable order for a file migration."""

        unique_paths = sorted({Path(path) for path in paths}, key=lambda item: str(item).casefold())
        with ExitStack() as stack:
            for path in unique_paths:
                stack.enter_context(self._sidecar_lock(path))
            yield

    @contextmanager
    def _sidecar_lock(self, path: Path):
        """Serialize short durable session operations across store instances."""
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_name(f"{path.name}.lock")
        with lock_path.open("a+b") as lock_file:
            self._lock_file(lock_file)
            try:
                yield
            finally:
                self._unlock_file(lock_file)

    @staticmethod
    def _lock_file(lock_file: Any) -> None:
        """Take an advisory cross-process lock on a small sidecar file."""
        # Windows is the deployed target.  fcntl keeps the implementation
        # testable on POSIX CI without changing the persistence contract.
        if os.name == "nt":
            import msvcrt

            handle = lock_file
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            return
        import fcntl

        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

    @staticmethod
    def _unlock_file(lock_file: Any) -> None:
        if os.name == "nt":
            import msvcrt

            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            return
        import fcntl

        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _delete_file(self, path: Path, session_id: str) -> bool:
        """同步刪除單一 session 檔案。"""
        with self._sidecar_lock(path):
            if not path.exists():
                return False
            path.unlink()
            # Keep a v2 identity tombstone after deletion.  It is tiny and
            # preserves deterministic collision probing: deleting slot 0 must
            # not make an existing colliding session in slot 1 unreachable.
            return True

    def _write_lines(self, path: Path, events: list[dict]) -> None:
        """Atomically replace the session file contents."""
        with self._sidecar_lock(path):
            lines = [json.dumps(event, ensure_ascii=False) for event in events]
            self._atomic_replace_lines(path, lines)

    def _load_lines(self, path: Path) -> list[dict]:
        """同步讀取 JSONL 檔案，並跳過無法解析或格式不符的行。"""
        if not path.exists():
            return []

        events: list[dict] = []
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue

                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if isinstance(parsed, dict):
                    events.append(parsed)

        return events

    def _list_session_ids(self, limit: int | None) -> tuple[str, ...]:
        if not self._sessions_dir.exists():
            return ()
        paths = sorted(self._sessions_dir.glob("*.jsonl"))
        session_ids: list[str] = []
        seen: set[str] = set()
        for path in paths:
            session_id = self._session_id_from_inventory_path(path)
            if session_id is not None and session_id not in seen:
                seen.add(session_id)
                session_ids.append(session_id)
                if limit is not None and len(session_ids) >= limit:
                    break
        return tuple(session_ids)

    def _list_session_ids_page(
        self,
        limit: int,
        after: str | None,
    ) -> SessionStoreInventoryPage:
        if not self._sessions_dir.exists():
            return SessionStoreInventoryPage(session_ids=(), next_cursor=None)
        paths = sorted(self._sessions_dir.glob("*.jsonl"), key=lambda item: item.name)
        if after is not None:
            paths = [path for path in paths if path.name > after]
        selected = paths[:limit]
        session_ids: list[str] = []
        seen: set[str] = set()
        for path in selected:
            session_id = self._session_id_from_inventory_path(path)
            if session_id is not None and session_id not in seen:
                seen.add(session_id)
                session_ids.append(session_id)
        return SessionStoreInventoryPage(
            session_ids=tuple(session_ids),
            next_cursor=selected[-1].name if len(paths) > len(selected) else None,
        )

    def _session_id_from_inventory_path(self, path: Path) -> str | None:
        """Return a verified logical ID without treating an encoded stem as one.

        Inventory is intentionally best effort, but it must not manufacture a
        session identity from a v1 filename or from a lossy legacy filename.
        Corrupt/conflicting files are skipped here and remain available for
        explicit fail-closed diagnosis through their requested logical ID.
        """

        source_ids = self._source_session_ids(path)
        if self._is_v2_path(path):
            try:
                session_id = self._read_v2_identity(path)
            except SessionIdentityConflictError:
                return None
            if source_ids and source_ids != {session_id}:
                return None
            return session_id
        if path.name.startswith(_SESSION_FILENAME_V1_PREFIX):
            # V1 Base64 used mixed case and is therefore not an identity on
            # default Windows volumes.  A v1 reader needs a matching durable
            # envelope instead of trusting the spelling requested by a caller.
            v1_session_id = self._decode_v1_session_path(path)
            return v1_session_id if v1_session_id is not None and source_ids == {v1_session_id} else None
        if len(source_ids) == 1:
            session_id = next(iter(source_ids))
            # The claimed identity must resolve back to this exact old path;
            # otherwise a hand-created/mismatched file cannot be listed as a
            # working session and later routed to a different path.
            return session_id if self._legacy_session_path(session_id) == path else None
        if source_ids:
            return None
        # This is the only identity-free legacy case that was reversible under
        # the old naming scheme.  A stem containing the reserved ``~`` marker
        # cannot be a legacy output and is deliberately skipped.
        candidate = path.stem
        return candidate if self._legacy_session_path(candidate) == path else None

    def _read_strict_snapshot_locked(
        self,
        path: Path,
        session_id: str,
    ) -> DurableSessionSnapshot:
        with self._sidecar_lock(path):
            return self._load_strict_snapshot(path, session_id)

    def _mutate_strict_snapshot(
        self,
        path: Path,
        session_id: str,
        expected_history_revision: str,
        build_events: Callable[[DurableSessionSnapshot], Sequence[Mapping[str, Any]] | None],
    ) -> StrictSessionAppendResult:
        gate = self._tool_workflow_publication_gate
        gate_transaction = (
            gate.publication_transaction()
            if gate is not None
            else _static_publication_transaction(self._tool_observability_v1)
        )
        # A publication lease is acquired before the session sidecar lock.
        # Disabling rejects subsequent leases and drains prior ones before its
        # caller returns, while writes for distinct session sidecars continue
        # concurrently.
        with gate_transaction as publication_enabled, self._sidecar_lock(path):
            before = self._load_strict_snapshot(path, session_id)
            if before.history_revision != expected_history_revision:
                return StrictSessionAppendResult(
                    status="rebase_required",
                    before=before,
                    after=before,
                )
            candidates = build_events(before)
            if candidates is None:
                return StrictSessionAppendResult(
                    status="unchanged",
                    before=before,
                    after=before,
                )
            normalized = self._normalize_strict_events(candidates, session_id=session_id)
            if not normalized:
                raise ValueError("strict mutation batch must not be empty.")
            if gate is not None and not publication_enabled:
                # A stale repository may already have formed its cross-store
                # observation/outbox batch before rollback started.  The live
                # gate is the final publication authority: retain unrelated
                # durable sources, but never let pre-built workflow cache
                # records bypass a completed disable barrier.
                normalized = tuple(
                    event
                    for event in normalized
                    if event.get("event") not in _TOOL_WORKFLOW_GATE_CONTROLLED_EVENTS
                )
                if not normalized:
                    return StrictSessionAppendResult(
                        status="unchanged",
                        before=before,
                        after=before,
                    )
            if publication_enabled:
                # Local import avoids a Store <-> reducer import cycle and
                # keeps the transition callback pure and bounded under lock.
                from mochi.api.tool_workflow_outbox import build_outbox_companion_events_v1

                companions = build_outbox_companion_events_v1(
                    session_id=session_id,
                    before_events=before.events,
                    source_events=normalized,
                )
                normalized = (*normalized, *companions)
            new_outbox_start_position = next(
                (
                    before.event_count + index
                    for index, event in enumerate(normalized, start=1)
                    if event.get("event") == "tool_workflow_aggregate_outbox"
                ),
                None,
            )
            lines = [
                json.dumps(
                    _json_clone(event),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                for event in (*before.events, *normalized)
            ]
            self._atomic_replace_lines(path, lines)
            after = self._load_strict_snapshot(path, session_id)
            return StrictSessionAppendResult(
                status="appended",
                before=before,
                after=after,
                new_outbox_start_position=new_outbox_start_position,
            )

    @staticmethod
    def _normalize_strict_events(
        events: Sequence[Mapping[str, Any]],
        *,
        session_id: str,
    ) -> tuple[dict[str, Any], ...]:
        if isinstance(events, (str, bytes)):
            raise TypeError("strict mutation events must be a sequence of objects.")
        normalized: list[dict[str, Any]] = []
        for index, event in enumerate(events):
            if not isinstance(event, Mapping):
                raise TypeError(f"strict mutation event {index} must be an object.")
            candidate = _json_clone(event)
            if not isinstance(candidate, dict):  # Defensive; mappings serialize as JSON objects.
                raise TypeError(f"strict mutation event {index} must be an object.")
            if "session_id" in candidate and candidate["session_id"] != session_id:
                raise ValueError(
                    f"strict mutation event {index} session_id does not match the requested session."
                )
            normalized.append(candidate)
        return tuple(normalized)

    @staticmethod
    def _atomic_replace_lines(path: Path, lines: Sequence[str]) -> None:
        """Durably replace JSONL contents so a batch is visible all-or-nothing."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
        try:
            with tmp_path.open("w", encoding="utf-8", newline="\n") as fh:
                for line in lines:
                    fh.write(line)
                    fh.write("\n")
                fh.flush()
                os.fsync(fh.fileno())
            _replace_with_transient_lock_retry(tmp_path, path)
            # Directory fsync is meaningful on POSIX.  Windows has no
            # equivalent available through this portable file API.
            if os.name != "nt":
                directory_fd = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    @staticmethod
    def _load_strict_snapshot(path: Path, session_id: str) -> DurableSessionSnapshot:
        if not path.exists():
            return DurableSessionSnapshot(
                session_id=session_id,
                events=(),
                history_revision=_history_revision(()),
                event_count=0,
                exists=False,
            )

        try:
            with path.open("r", encoding="utf-8", errors="strict", newline=None) as fh:
                raw_lines = fh.read().splitlines()
        except (OSError, UnicodeError) as exc:
            raise StrictSessionSnapshotError(
                f"cannot read strict session history: {exc}"
            ) from exc
        if not raw_lines:
            raise StrictSessionSnapshotError("existing strict session history is empty")

        events: list[dict[str, Any]] = []
        for index, raw_line in enumerate(raw_lines):
            if not raw_line.strip():
                raise StrictSessionSnapshotError(
                    f"strict session history contains an empty line at index {index}"
                )
            try:
                parsed = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise StrictSessionSnapshotError(
                    f"strict session history contains malformed JSON at index {index}"
                ) from exc
            if not isinstance(parsed, dict):
                raise StrictSessionSnapshotError(
                    f"strict session history line {index} must be a JSON object"
                )
            if not parsed:
                raise StrictSessionSnapshotError(
                    f"strict session history contains an empty object at index {index}"
                )
            if "session_id" in parsed and parsed["session_id"] != session_id:
                raise StrictSessionSnapshotError(
                    "strict session history event session_id does not match the requested session "
                    f"at index {index}"
                )
            try:
                normalized = _json_clone(parsed)
            except (TypeError, ValueError) as exc:
                raise StrictSessionSnapshotError(
                    f"strict session history contains invalid JSON values at index {index}"
                ) from exc
            if not isinstance(normalized, dict):  # pragma: no cover - defensive JSON invariant.
                raise StrictSessionSnapshotError(
                    f"strict session history line {index} must be a JSON object"
                )
            events.append(normalized)
        frozen_events = tuple(_freeze_json(event) for event in events)
        return DurableSessionSnapshot(
            session_id=session_id,
            events=frozen_events,
            history_revision=_history_revision(events),
            event_count=len(events),
            exists=True,
        )


def _history_revision(events: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        list(events),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _json_clone(value: Any) -> Any:
    return json.loads(
        json.dumps(
            _json_compatible(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def _json_compatible(value: Any) -> Any:
    """Thaw immutable snapshot values before canonical JSON serialization."""
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            normalized[key] = _json_compatible(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported strict JSON value: {type(value).__name__}")


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value
