"""Capability-style facade for race-safe workspace mutations."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from .file_contract import (
    AuthorizationEnvelope,
    ChangeEntry,
    FileIdentity,
    authorization_request_digest,
)


class UnsafeFilesystemTarget(PermissionError):
    """Raised when a path cannot be pinned without following an alias."""


class SafeFilesystemUnavailable(RuntimeError):
    """Raised when the native enforcing backend is unavailable."""


class _TargetOwner(Protocol):
    def release_target(self, target: SafeTarget) -> None: ...


@dataclass(frozen=True, slots=True)
class AuthorizedFileBinding:
    """Canonical file authority captured when a SafeTarget is issued."""

    entry_id: str
    canonical_relative_path: str
    operation: Literal["add", "update", "delete", "rename"]
    base_identity: FileIdentity
    authorization_digest: str


def resolve_authorized_file_binding(
    *,
    canonical_relative_path: str,
    authorization: AuthorizationEnvelope,
    captured_identity: FileIdentity,
    canonicalize_authorized_path: Callable[[str], str],
) -> AuthorizedFileBinding:
    """Resolve one unique canonical manifest entry to the pinned file."""

    if (
        authorization.kind != "file_change"
        or authorization.file_request is None
        or authorization.exec_request is not None
    ):
        raise UnsafeFilesystemTarget(
            "file targets require a file_change authorization"
        )

    entries_by_path: dict[str, ChangeEntry] = {}
    for entry in authorization.file_request.entries:
        canonical_entry_path = canonicalize_authorized_path(
            entry.relative_path
        )
        if canonical_entry_path in entries_by_path:
            raise UnsafeFilesystemTarget(
                "authorization contains duplicate canonical file paths"
            )
        entries_by_path[canonical_entry_path] = entry

    entry = entries_by_path.get(canonical_relative_path)
    if entry is None:
        raise UnsafeFilesystemTarget(
            "filesystem target is not authorized by the file request"
        )
    if entry.base_identity != captured_identity:
        raise UnsafeFilesystemTarget(
            "captured file identity does not match authorized base identity"
        )

    return AuthorizedFileBinding(
        entry_id=entry.entry_id,
        canonical_relative_path=canonical_relative_path,
        operation=entry.operation,
        base_identity=captured_identity,
        authorization_digest=authorization_request_digest(authorization),
    )


class SafeTarget:
    """Opaque, immutable authority for one basename under a pinned parent."""

    __slots__ = (
        "_authorization_digest",
        "_basename",
        "_closed",
        "_identity",
        "_owner",
        "_parent",
        "_seal",
    )
    _FACTORY_SEAL = object()

    def __init__(
        self,
        *,
        basename: str,
        identity: FileIdentity,
        authorization_digest: str,
        _owner: _TargetOwner,
        _parent: object,
        _factory_seal: object | None = None,
    ) -> None:
        if _factory_seal is not self._FACTORY_SEAL:
            raise TypeError("SafeTarget cannot be constructed directly")
        object.__setattr__(self, "_basename", basename)
        object.__setattr__(self, "_identity", identity)
        object.__setattr__(self, "_authorization_digest", authorization_digest)
        object.__setattr__(self, "_owner", _owner)
        object.__setattr__(self, "_parent", _parent)
        object.__setattr__(self, "_closed", False)
        object.__setattr__(self, "_seal", self._FACTORY_SEAL)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("SafeTarget is immutable")

    @classmethod
    def _create(
        cls,
        *,
        basename: str,
        identity: FileIdentity,
        authorization_digest: str,
        owner: _TargetOwner,
        parent: object,
    ) -> SafeTarget:
        return cls(
            basename=basename,
            identity=identity,
            authorization_digest=authorization_digest,
            _owner=owner,
            _parent=parent,
            _factory_seal=cls._FACTORY_SEAL,
        )

    @property
    def basename(self) -> str:
        return self._basename

    @property
    def identity(self) -> FileIdentity:
        return self._identity

    @property
    def authorization_digest(self) -> str:
        return self._authorization_digest

    @property
    def closed(self) -> bool:
        return self._closed

    def _is_authentic(self) -> bool:
        return self._seal is self._FACTORY_SEAL

    def _mark_closed(self) -> None:
        object.__setattr__(self, "_closed", True)

    def close(self) -> None:
        if not self._closed:
            self._owner.release_target(self)

    def __enter__(self) -> SafeTarget:
        if self._closed:
            raise ValueError("SafeTarget is closed")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


class _Backend(Protocol):
    def prepare_target(
        self, relative_path: str | Path, authorization: AuthorizationEnvelope
    ) -> SafeTarget: ...

    def unlink(self, target: SafeTarget) -> None: ...

    def replace(self, source: SafeTarget, destination: SafeTarget) -> None: ...

    def release_target(self, target: SafeTarget) -> None: ...

    def close(self) -> None: ...


class SafeFilesystem:
    """Select the native backend; enforcing mode never falls back lexically."""

    def __init__(self, workspace: str | Path, *, enforce: bool = True) -> None:
        if os.name == "nt":
            from .safe_fs_windows import WindowsSafeFilesystem

            self._backend: _Backend = WindowsSafeFilesystem(workspace, enforce=enforce)
        elif os.name == "posix":
            from .safe_fs_posix import PosixSafeFilesystem

            self._backend = PosixSafeFilesystem(workspace)
        elif enforce:
            raise SafeFilesystemUnavailable(
                "no enforcing filesystem backend for this platform"
            )
        else:
            raise SafeFilesystemUnavailable(
                "unsafe filesystem fallback is intentionally absent"
            )

    def prepare_target(
        self, relative_path: str | Path, authorization: AuthorizationEnvelope
    ) -> SafeTarget:
        return self._backend.prepare_target(relative_path, authorization)

    def unlink(self, target: SafeTarget) -> None:
        if not isinstance(target, SafeTarget):
            raise TypeError("filesystem mutation requires SafeTarget")
        self._backend.unlink(target)

    def replace(self, source: SafeTarget, destination: SafeTarget) -> None:
        if not isinstance(source, SafeTarget) or not isinstance(destination, SafeTarget):
            raise TypeError("filesystem mutation requires SafeTarget")
        self._backend.replace(source, destination)

    def close(self) -> None:
        self._backend.close()

    def __enter__(self) -> SafeFilesystem:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


__all__ = [
    "AuthorizedFileBinding",
    "SafeFilesystem",
    "SafeFilesystemUnavailable",
    "SafeTarget",
    "UnsafeFilesystemTarget",
    "resolve_authorized_file_binding",
]
