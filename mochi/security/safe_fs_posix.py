"""Descriptor-relative POSIX filesystem mutation backend."""

from __future__ import annotations

import os
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .file_contract import AuthorizationEnvelope, FileIdentity
from .safe_filesystem import (
    AuthorizedFileBinding,
    SafeTarget,
    UnsafeFilesystemTarget,
    resolve_authorized_file_binding,
)


@dataclass(frozen=True, slots=True)
class _IssuedPosixTarget:
    target: SafeTarget
    basename: str
    identity: FileIdentity
    binding: AuthorizedFileBinding
    parent_fd: int


class PosixSafeFilesystem:
    """Pin every parent directory and mutate by dir_fd plus basename only."""

    def __init__(self, workspace: str | Path, *, adapter: Any = os) -> None:
        self._adapter = adapter
        self._closed = False
        self._issued: dict[int, _IssuedPosixTarget] = {}
        self._directory_flags = (
            int(adapter.O_RDONLY)
            | int(getattr(adapter, "O_DIRECTORY", 0))
            | int(getattr(adapter, "O_NOFOLLOW", 0))
        )
        if not int(getattr(adapter, "O_DIRECTORY", 0)) or not int(
            getattr(adapter, "O_NOFOLLOW", 0)
        ):
            raise RuntimeError(
                "POSIX enforcing mode requires O_DIRECTORY and O_NOFOLLOW"
            )
        # This is the sole absolute-path open. Every descendant operation is relative.
        self._root_fd = int(adapter.open(str(workspace), self._directory_flags))
        root_info = adapter.fstat(self._root_fd)
        self._root_identity = FileIdentity(
            platform="posix",
            volume_id=str(root_info.st_dev),
            file_id=str(root_info.st_ino),
            link_count=int(root_info.st_nlink),
            is_reparse_point=False,
        )

    @staticmethod
    def _segments(relative_path: str | Path) -> tuple[str, ...]:
        raw = str(relative_path)
        if not raw or "\x00" in raw:
            raise UnsafeFilesystemTarget("invalid empty filesystem target")
        path = PurePosixPath(raw)
        if path.is_absolute():
            raise UnsafeFilesystemTarget(
                "absolute paths are not mutation capabilities"
            )
        if ".." in path.parts:
            raise UnsafeFilesystemTarget("parent traversal is forbidden")
        segments = tuple(part for part in path.parts if part not in {"", "."})
        if not segments:
            raise UnsafeFilesystemTarget("workspace root is not a file target")
        return segments

    def _open_parent(self, segments: tuple[str, ...]) -> int:
        current = int(self._adapter.dup(self._root_fd))
        try:
            for segment in segments[:-1]:
                next_fd = int(
                    self._adapter.open(
                        segment, self._directory_flags, dir_fd=current
                    )
                )
                self._adapter.close(current)
                current = next_fd
            return current
        except BaseException:
            self._adapter.close(current)
            raise

    def _identity_at(self, parent_fd: int, basename: str) -> FileIdentity:
        try:
            info = self._adapter.stat(
                basename, dir_fd=parent_fd, follow_symlinks=False
            )
        except (FileNotFoundError, KeyError) as exc:
            raise UnsafeFilesystemTarget(
                "target must exist before authorization"
            ) from exc
        if stat.S_ISLNK(int(info.st_mode)):
            raise UnsafeFilesystemTarget("symlink targets are forbidden")
        if not stat.S_ISREG(int(info.st_mode)):
            raise UnsafeFilesystemTarget("only regular file targets are allowed")
        links = int(info.st_nlink)
        if links != 1:
            raise UnsafeFilesystemTarget("hardlink targets are forbidden")
        return FileIdentity(
            platform="posix",
            volume_id=str(info.st_dev),
            file_id=str(info.st_ino),
            link_count=links,
            is_reparse_point=False,
        )

    def prepare_target(
        self,
        relative_path: str | Path,
        authorization: AuthorizationEnvelope,
    ) -> SafeTarget:
        if self._closed:
            raise ValueError("filesystem is closed")
        if not isinstance(authorization, AuthorizationEnvelope):
            raise TypeError("authorization must be AuthorizationEnvelope")
        if authorization.kind != "file_change":
            raise UnsafeFilesystemTarget(
                "file targets require a file_change authorization"
            )
        if self._root_identity != authorization.context.workspace_identity:
            raise UnsafeFilesystemTarget(
                "workspace identity does not match authorization"
            )
        segments = self._segments(relative_path)
        parent_fd = self._open_parent(segments)
        try:
            identity = self._identity_at(parent_fd, segments[-1])
            binding = resolve_authorized_file_binding(
                canonical_relative_path="/".join(segments),
                authorization=authorization,
                captured_identity=identity,
                canonicalize_authorized_path=lambda path: "/".join(
                    self._segments(path)
                ),
            )
            target = SafeTarget._create(
                basename=segments[-1],
                identity=identity,
                authorization_digest=binding.authorization_digest,
                owner=self,
                parent=parent_fd,
            )
            self._issued[id(target)] = _IssuedPosixTarget(
                target=target,
                basename=segments[-1],
                identity=identity,
                binding=binding,
                parent_fd=parent_fd,
            )
            return target
        except BaseException:
            self._adapter.close(parent_fd)
            raise

    def _record(self, target: SafeTarget) -> _IssuedPosixTarget:
        if not isinstance(target, SafeTarget):
            raise TypeError("filesystem mutation requires SafeTarget")
        record = self._issued.get(id(target))
        if (
            record is None
            or record.target is not target
            or not target._is_authentic()
            or target._owner is not self
            or target.closed
            or target.basename != record.basename
            or target.identity != record.identity
            or target.identity != record.binding.base_identity
            or (
                target.authorization_digest
                != record.binding.authorization_digest
            )
            or target._parent != record.parent_fd
        ):
            raise UnsafeFilesystemTarget(
                "SafeTarget is not an issued capability"
            )
        return record

    def _validated(self, target: SafeTarget) -> _IssuedPosixTarget:
        record = self._record(target)
        if self._identity_at(record.parent_fd, record.basename) != record.identity:
            raise UnsafeFilesystemTarget(
                "file identity changed after authorization"
            )
        return record

    def unlink(self, target: SafeTarget) -> None:
        record = self._validated(target)
        if record.binding.operation != "delete":
            raise UnsafeFilesystemTarget(
                "unlink requires delete authorization"
            )
        self._adapter.unlink(record.basename, dir_fd=record.parent_fd)
        self.release_target(target)

    def replace(self, source: SafeTarget, destination: SafeTarget) -> None:
        source_record = self._validated(source)
        destination_record = self._validated(destination)
        if destination_record.binding.operation not in {
            "update",
            "rename",
        }:
            raise UnsafeFilesystemTarget(
                "replace destination requires update or rename authorization"
            )
        if source_record.binding != destination_record.binding:
            raise UnsafeFilesystemTarget(
                "replace operands require the same authorization binding"
            )
        self._adapter.replace(
            source_record.basename,
            destination_record.basename,
            src_dir_fd=source_record.parent_fd,
            dst_dir_fd=destination_record.parent_fd,
        )
        self.release_target(source)
        self.release_target(destination)

    def create_temp(
        self, target: SafeTarget, *, mode: int = 0o600
    ) -> tuple[str, int]:
        """Create a sibling temp using only the pinned parent descriptor."""
        record = self._validated(target)
        flags = (
            int(self._adapter.O_WRONLY)
            | int(self._adapter.O_CREAT)
            | int(self._adapter.O_EXCL)
            | int(self._adapter.O_NOFOLLOW)
        )
        for _ in range(128):
            basename = f".mochi-{record.basename}.{secrets.token_hex(6)}"
            try:
                fd = int(
                    self._adapter.open(
                        basename,
                        flags,
                        mode,
                        dir_fd=record.parent_fd,
                    )
                )
                return basename, fd
            except FileExistsError:
                continue
        raise RuntimeError(
            "unable to allocate descriptor-relative temp file"
        )

    def release_target(self, target: SafeTarget) -> None:
        record = self._record(target)
        del self._issued[id(target)]
        self._adapter.close(record.parent_fd)
        target._mark_closed()

    def close(self) -> None:
        if self._closed:
            return
        for record in tuple(self._issued.values()):
            self._adapter.close(record.parent_fd)
            record.target._mark_closed()
        self._issued.clear()
        self._adapter.close(self._root_fd)
        self._closed = True


__all__ = ["PosixSafeFilesystem"]
