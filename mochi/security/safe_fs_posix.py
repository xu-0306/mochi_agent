"""Descriptor-relative POSIX filesystem mutation backend.

Pinned parent FDs protect against ancestor and symlink rebinds. Portable
POSIX Python cannot atomically identity-CAS the final basename, so a hostile
concurrent writer with write access to the pinned directory can still race
unlink or replace. The pinned directory itself may also be renamed outside
the workspace.
"""

from __future__ import annotations

import os
import secrets
import stat
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .file_contract import AuthorizationEnvelope, FileIdentity
from .safe_filesystem import (
    AuthorizedFileBinding,
    CommittedFilesystemMutationError,
    SafeTarget,
    StagedTemp,
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


@dataclass(frozen=True, slots=True)
class _IssuedPosixTemp:
    temp: StagedTemp
    basename: str
    identity: FileIdentity
    binding: AuthorizedFileBinding
    parent_fd: int
    file_fd: int


class PosixSafeFilesystem:
    """Mutate through pinned parent FDs and relative basenames.

    This protects against ancestor and symlink rebinds. It does not provide
    an atomic final-basename identity CAS on portable POSIX Python: a hostile
    writer with access to the pinned directory can race unlink or replace,
    and the pinned directory itself may be renamed outside the workspace.
    """

    def __init__(self, workspace: str | Path, *, adapter: Any = os) -> None:
        self._adapter = adapter
        self._closed = False
        self._issued: dict[int, _IssuedPosixTarget] = {}
        self._temps: dict[int, _IssuedPosixTemp] = {}
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
        self._root_fd = int(
            adapter.open(str(workspace), self._directory_flags)
        )
        try:
            root_info = adapter.fstat(self._root_fd)
            self._root_identity = FileIdentity(
                platform="posix",
                volume_id=str(root_info.st_dev),
                file_id=str(root_info.st_ino),
                link_count=int(root_info.st_nlink),
                is_reparse_point=False,
            )
        except BaseException:
            with suppress(BaseException):
                self._adapter.close(self._root_fd)
            raise

    @staticmethod
    def _segments(relative_path: str | Path) -> tuple[str, ...]:
        raw = str(relative_path)
        if not raw or "\x00" in raw:
            raise UnsafeFilesystemTarget(
                "invalid empty filesystem target"
            )
        path = PurePosixPath(raw)
        if path.is_absolute():
            raise UnsafeFilesystemTarget(
                "absolute paths are not mutation capabilities"
            )
        if ".." in path.parts:
            raise UnsafeFilesystemTarget(
                "parent traversal is forbidden"
            )
        segments = tuple(
            part for part in path.parts if part not in {"", "."}
        )
        if not segments:
            raise UnsafeFilesystemTarget(
                "workspace root is not a file target"
            )
        return segments

    @staticmethod
    def _remember_error(
        first: BaseException | None,
        action: Callable[[], None],
    ) -> BaseException | None:
        try:
            action()
        except BaseException as exc:
            if first is None:
                return exc
            first.add_note(
                f"additional cleanup failure: {exc!r}"
            )
        return first

    def _open_parent(self, segments: tuple[str, ...]) -> int:
        current = int(self._adapter.dup(self._root_fd))
        for segment in segments[:-1]:
            try:
                next_fd = int(
                    self._adapter.open(
                        segment,
                        self._directory_flags,
                        dir_fd=current,
                    )
                )
            except BaseException:
                with suppress(BaseException):
                    self._adapter.close(current)
                raise
            try:
                self._adapter.close(current)
            except BaseException:
                with suppress(BaseException):
                    self._adapter.close(next_fd)
                raise
            current = next_fd
        return current

    @staticmethod
    def _identity_from_info(info: Any) -> FileIdentity:
        if stat.S_ISLNK(int(info.st_mode)):
            raise UnsafeFilesystemTarget(
                "symlink targets are forbidden"
            )
        if not stat.S_ISREG(int(info.st_mode)):
            raise UnsafeFilesystemTarget(
                "only regular file targets are allowed"
            )
        links = int(info.st_nlink)
        if links != 1:
            raise UnsafeFilesystemTarget(
                "hardlink targets are forbidden"
            )
        return FileIdentity(
            platform="posix",
            volume_id=str(info.st_dev),
            file_id=str(info.st_ino),
            link_count=links,
            is_reparse_point=False,
        )

    def _identity_at(
        self, parent_fd: int, basename: str
    ) -> FileIdentity:
        try:
            info = self._adapter.stat(
                basename,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except (FileNotFoundError, KeyError) as exc:
            raise UnsafeFilesystemTarget(
                "target must exist before authorization"
            ) from exc
        return self._identity_from_info(info)

    def _identity_of_fd(self, file_fd: int) -> FileIdentity:
        return self._identity_from_info(
            self._adapter.fstat(file_fd)
        )

    def prepare_target(
        self,
        relative_path: str | Path,
        authorization: AuthorizationEnvelope,
    ) -> SafeTarget:
        if self._closed:
            raise ValueError("filesystem is closed")
        if not isinstance(authorization, AuthorizationEnvelope):
            raise TypeError(
                "authorization must be AuthorizationEnvelope"
            )
        if authorization.kind != "file_change":
            raise UnsafeFilesystemTarget(
                "file targets require a file_change authorization"
            )
        if (
            self._root_identity
            != authorization.context.workspace_identity
        ):
            raise UnsafeFilesystemTarget(
                "workspace identity does not match authorization"
            )
        segments = self._segments(relative_path)
        parent_fd = self._open_parent(segments)
        try:
            identity = self._identity_at(
                parent_fd, segments[-1]
            )
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
                authorization_digest=(
                    binding.authorization_digest
                ),
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
            with suppress(BaseException):
                self._adapter.close(parent_fd)
            raise

    def _record(
        self, target: SafeTarget
    ) -> _IssuedPosixTarget:
        if not isinstance(target, SafeTarget):
            raise TypeError(
                "filesystem mutation requires SafeTarget"
            )
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

    def _temp_record(
        self, temp: StagedTemp
    ) -> _IssuedPosixTemp:
        if not isinstance(temp, StagedTemp):
            raise TypeError(
                "replace source must be a StagedTemp capability"
            )
        record = self._temps.get(id(temp))
        if (
            record is None
            or record.temp is not temp
            or not temp._is_authentic()
            or temp._owner is not self
            or temp.closed
            or temp.basename != record.basename
            or temp.identity != record.identity
            or temp.binding != record.binding
            or (
                temp.authorization_digest
                != record.binding.authorization_digest
            )
            or temp._parent != record.parent_fd
        ):
            raise UnsafeFilesystemTarget(
                "StagedTemp is not an issued capability"
            )
        return record

    def _validated(
        self, target: SafeTarget
    ) -> _IssuedPosixTarget:
        record = self._record(target)
        if (
            self._identity_at(
                record.parent_fd, record.basename
            )
            != record.identity
        ):
            raise UnsafeFilesystemTarget(
                "file identity changed after authorization"
            )
        return record

    def _validated_temp(
        self, temp: StagedTemp
    ) -> _IssuedPosixTemp:
        record = self._temp_record(temp)
        fd_identity = self._identity_of_fd(record.file_fd)
        path_identity = self._identity_at(
            record.parent_fd, record.basename
        )
        if (
            fd_identity != record.identity
            or path_identity != record.identity
        ):
            raise UnsafeFilesystemTarget(
                "staged temp identity changed after issuance"
            )
        return record

    def unlink(self, target: SafeTarget) -> None:
        record = self._validated(target)
        if record.binding.operation != "delete":
            raise UnsafeFilesystemTarget(
                "unlink requires delete authorization"
            )
        self._adapter.unlink(
            record.basename, dir_fd=record.parent_fd
        )
        self.release_target(target)

    def create_temp(
        self, target: SafeTarget, *, mode: int = 0o600
    ) -> StagedTemp:
        record = self._validated(target)
        if record.binding.operation not in {"update", "rename"}:
            raise UnsafeFilesystemTarget(
                "temp creation requires update or rename authorization"
            )
        parent_fd = int(self._adapter.dup(record.parent_fd))
        file_fd: int | None = None
        try:
            flags = (
                int(self._adapter.O_WRONLY)
                | int(self._adapter.O_CREAT)
                | int(self._adapter.O_EXCL)
                | int(self._adapter.O_NOFOLLOW)
            )
            for _ in range(128):
                basename = (
                    f".mochi-{record.basename}."
                    f"{secrets.token_hex(6)}"
                )
                try:
                    file_fd = int(
                        self._adapter.open(
                            basename,
                            flags,
                            mode,
                            dir_fd=parent_fd,
                        )
                    )
                    break
                except FileExistsError:
                    continue
            else:
                raise RuntimeError(
                    "unable to allocate descriptor-relative temp file"
                )

            fd_identity = self._identity_of_fd(file_fd)
            path_identity = self._identity_at(
                parent_fd, basename
            )
            if fd_identity != path_identity:
                raise UnsafeFilesystemTarget(
                    "staged temp identity does not match sibling name"
                )
            temp = StagedTemp._create(
                basename=basename,
                identity=fd_identity,
                binding=record.binding,
                owner=self,
                parent=parent_fd,
            )
            self._temps[id(temp)] = _IssuedPosixTemp(
                temp=temp,
                basename=basename,
                identity=fd_identity,
                binding=record.binding,
                parent_fd=parent_fd,
                file_fd=file_fd,
            )
            return temp
        except BaseException:
            if file_fd is not None:
                with suppress(BaseException):
                    self._adapter.close(file_fd)
            with suppress(BaseException):
                self._adapter.close(parent_fd)
            raise

    def replace(
        self, source: StagedTemp, destination: SafeTarget
    ) -> None:
        source_record = self._validated_temp(source)
        destination_record = self._validated(destination)
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
        error: BaseException | None = None
        error = self._remember_error(
            error, lambda: self.release_temp(source)
        )
        error = self._remember_error(
            error, lambda: self.release_target(destination)
        )
        if error is not None:
            outcome = CommittedFilesystemMutationError(
                phase="operand_cleanup", cause=error
            )
            raise outcome from error

    def release_target(self, target: SafeTarget) -> None:
        record = self._record(target)
        del self._issued[id(target)]
        try:
            self._adapter.close(record.parent_fd)
        finally:
            target._mark_closed()

    def release_temp(self, temp: StagedTemp) -> None:
        record = self._temp_record(temp)
        del self._temps[id(temp)]
        error: BaseException | None = None
        error = self._remember_error(
            error,
            lambda: self._adapter.close(record.file_fd),
        )
        error = self._remember_error(
            error,
            lambda: self._adapter.close(record.parent_fd),
        )
        temp._mark_closed()
        if error is not None:
            raise error

    def close(self) -> None:
        if self._closed:
            return
        error: BaseException | None = None
        for record in tuple(self._temps.values()):
            error = self._remember_error(
                error,
                lambda item=record: self.release_temp(item.temp),
            )
        for record in tuple(self._issued.values()):
            error = self._remember_error(
                error,
                lambda item=record: self.release_target(item.target),
            )
        error = self._remember_error(
            error, lambda: self._adapter.close(self._root_fd)
        )
        self._closed = True
        if error is not None:
            raise error


__all__ = ["PosixSafeFilesystem"]
