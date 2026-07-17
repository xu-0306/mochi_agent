"""Descriptor-relative POSIX filesystem mutation backend.

Pinned parent FDs protect against ancestor and symlink rebinds. Portable
POSIX Python cannot atomically identity-CAS the final basename, so a hostile
concurrent writer with write access to the pinned directory can still race
unlink or replace. The pinned directory itself may also be renamed outside
the workspace.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
import sys
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
    UnsupportedSecurityMetadata,
    resolve_authorized_file_binding,
)


@dataclass(frozen=True, slots=True)
class _NativeMetadata:
    uid: int
    gid: int
    mode: int
    xattrs: tuple[tuple[bytes, bytes], ...]


@dataclass(frozen=True, slots=True)
class _IssuedPosixTarget:
    target: SafeTarget
    basename: str
    identity: FileIdentity
    binding: AuthorizedFileBinding
    parent_fd: int
    file_fd: int


@dataclass(frozen=True, slots=True)
class _IssuedPosixTemp:
    temp: StagedTemp
    target: SafeTarget
    basename: str
    identity: FileIdentity
    binding: AuthorizedFileBinding
    parent_fd: int
    file_fd: int


@dataclass(frozen=True, slots=True)
class _IssuedMetadataSnapshot:
    snapshot: object
    target: SafeTarget
    identity: FileIdentity
    binding: AuthorizedFileBinding
    native: _NativeMetadata


class PosixSafeFilesystem:
    """Mutate through pinned parent FDs and relative basenames."""

    def __init__(self, workspace: str | Path, *, adapter: Any = os) -> None:
        required_constants = (
            "O_RDONLY",
            "O_RDWR",
            "O_CREAT",
            "O_EXCL",
            "O_DIRECTORY",
            "O_NOFOLLOW",
        )
        required_calls = (
            "open",
            "close",
            "dup",
            "fstat",
            "stat",
            "pread",
            "write",
            "fsync",
            "listxattr",
            "getxattr",
            "fchown",
            "fchmod",
            "removexattr",
            "setxattr",
            "unlink",
            "replace",
        )
        missing: list[str] = []
        for name in required_constants:
            try:
                int(getattr(adapter, name))
            except (AttributeError, TypeError, ValueError):
                missing.append(name)
        missing.extend(
            name
            for name in required_calls
            if not callable(getattr(adapter, name, None))
        )
        if missing:
            raise RuntimeError(
                "POSIX enforcing mode requires primitives: "
                + ", ".join(missing)
            )
        self._adapter = adapter
        self._platform = str(getattr(adapter, "platform", sys.platform))
        self._closed = False
        self._issued: dict[int, _IssuedPosixTarget] = {}
        self._temps: dict[int, _IssuedPosixTemp] = {}
        self._metadata: dict[int, _IssuedMetadataSnapshot] = {}
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
        self._root_fd = int(adapter.open(str(workspace), self._directory_flags))
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
            raise UnsafeFilesystemTarget(
                "workspace root is not a file target"
            )
        return segments

    @staticmethod
    def _add_cleanup_note(
        primary: BaseException, additional: BaseException
    ) -> None:
        try:
            try:
                detail = str(additional)
            except BaseException:
                detail = (
                    f"<{type(additional).__name__} could not be formatted>"
                )
            BaseException.add_note(
                primary, f"additional cleanup failure: {detail}"
            )
        except BaseException:
            pass

    @classmethod
    def _remember_error(
        cls,
        first: BaseException | None,
        action: Callable[[], None],
    ) -> BaseException | None:
        try:
            action()
        except BaseException as exc:
            if first is None:
                return exc
            cls._add_cleanup_note(first, exc)
        return first

    def _unsupported(
        self, phase: str, cause: BaseException
    ) -> UnsupportedSecurityMetadata:
        return UnsupportedSecurityMetadata(
            phase=phase, platform=self._platform, cause=cause
        )

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
            raise UnsafeFilesystemTarget("symlink targets are forbidden")
        if not stat.S_ISREG(int(info.st_mode)):
            raise UnsafeFilesystemTarget(
                "only regular file targets are allowed"
            )
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

    def _identity_at(self, parent_fd: int, basename: str) -> FileIdentity:
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
        return self._identity_from_info(self._adapter.fstat(file_fd))

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
        file_fd: int | None = None
        try:
            basename = segments[-1]
            path_identity = self._identity_at(parent_fd, basename)
            binding = resolve_authorized_file_binding(
                canonical_relative_path="/".join(segments),
                authorization=authorization,
                captured_identity=path_identity,
                canonicalize_authorized_path=lambda path: "/".join(
                    self._segments(path)
                ),
            )
            flags = int(self._adapter.O_RDONLY) | int(
                self._adapter.O_NOFOLLOW
            )
            file_fd = int(
                self._adapter.open(
                    basename, flags, dir_fd=parent_fd
                )
            )
            fd_identity = self._identity_of_fd(file_fd)
            current_identity = self._identity_at(parent_fd, basename)
            if (
                fd_identity != path_identity
                or current_identity != path_identity
                or fd_identity != binding.base_identity
            ):
                raise UnsafeFilesystemTarget(
                    "retained file identity does not match authorized target"
                )
            target = SafeTarget._create(
                basename=basename,
                identity=path_identity,
                authorization_digest=binding.authorization_digest,
                owner=self,
                parent=parent_fd,
            )
            self._issued[id(target)] = _IssuedPosixTarget(
                target=target,
                basename=basename,
                identity=path_identity,
                binding=binding,
                parent_fd=parent_fd,
                file_fd=file_fd,
            )
            return target
        except BaseException:
            if file_fd is not None:
                with suppress(BaseException):
                    self._adapter.close(file_fd)
            with suppress(BaseException):
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
            or target.authorization_digest
            != record.binding.authorization_digest
            or target._parent != record.parent_fd
        ):
            raise UnsafeFilesystemTarget(
                "SafeTarget is not an issued capability"
            )
        return record

    def _temp_record(self, temp: StagedTemp) -> _IssuedPosixTemp:
        if not isinstance(temp, StagedTemp):
            raise TypeError("replace source must be a StagedTemp capability")
        record = self._temps.get(id(temp))
        if (
            record is None
            or record.temp is not temp
            or not temp._is_authentic()
            or temp._owner is not self
            or temp.closed
            or temp.basename != record.basename
            or temp.identity != record.identity
            or temp.binding is not record.binding
            or temp.authorization_digest
            != record.binding.authorization_digest
            or temp._parent != record.parent_fd
        ):
            raise UnsafeFilesystemTarget(
                "StagedTemp is not an issued capability"
            )
        return record

    def _validated(self, target: SafeTarget) -> _IssuedPosixTarget:
        record = self._record(target)
        path_identity = self._identity_at(
            record.parent_fd, record.basename
        )
        fd_identity = self._identity_of_fd(record.file_fd)
        if fd_identity != record.identity:
            raise UnsafeFilesystemTarget(
                "retained file identity changed after authorization"
            )
        if path_identity != record.identity:
            raise UnsafeFilesystemTarget(
                "file identity changed after authorization"
            )
        return record

    def _validated_temp(self, temp: StagedTemp) -> _IssuedPosixTemp:
        record = self._temp_record(temp)
        fd_identity = self._identity_of_fd(record.file_fd)
        path_identity = self._identity_at(
            record.parent_fd, record.basename
        )
        if fd_identity != record.identity or path_identity != record.identity:
            raise UnsafeFilesystemTarget(
                "staged temp identity changed after issuance"
            )
        return record

    @staticmethod
    def _raw_xattr_name(name: object) -> bytes:
        if isinstance(name, str):
            return os.fsencode(name)
        if isinstance(name, bytes):
            return name
        raise TypeError("xattr names must be str or bytes")

    @staticmethod
    def _metadata_digest(native: _NativeMetadata) -> str:
        payload = {
            "gid": native.gid,
            "mode": native.mode,
            "uid": native.uid,
            "xattrs": [
                {"name_hex": name.hex(), "value_hex": value.hex()}
                for name, value in native.xattrs
            ],
        }
        raw = json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("ascii")
        return hashlib.sha256(raw).hexdigest()

    def _capture_native_metadata(self, file_fd: int) -> _NativeMetadata:
        if not self._platform.startswith("linux"):
            cause = RuntimeError(
                "security metadata preservation requires Linux"
            )
            error = self._unsupported("platform", cause)
            raise error from cause
        try:
            info = self._adapter.fstat(file_fd)
        except BaseException as cause:
            error = self._unsupported("capture", cause)
            raise error from cause
        try:
            names = sorted(
                self._raw_xattr_name(name)
                for name in self._adapter.listxattr(file_fd)
            )
        except BaseException as cause:
            error = self._unsupported("list_metadata", cause)
            raise error from cause
        xattrs: list[tuple[bytes, bytes]] = []
        for name in names:
            try:
                value = bytes(self._adapter.getxattr(file_fd, name))
            except BaseException as cause:
                error = self._unsupported("get_metadata", cause)
                raise error from cause
            xattrs.append((name, value))
        return _NativeMetadata(
            uid=int(info.st_uid),
            gid=int(info.st_gid),
            mode=stat.S_IMODE(int(info.st_mode)),
            xattrs=tuple(xattrs),
        )

    def transaction_binding(
        self, target: SafeTarget
    ) -> AuthorizedFileBinding:
        return self._validated(target).binding

    def capture_metadata(self, target: SafeTarget):
        from ..tools.file_transaction import FileMetadataSnapshot

        record = self._validated(target)
        native = self._capture_native_metadata(record.file_fd)
        digest = self._metadata_digest(native)
        if (
            record.binding.base_metadata_sha256 != digest
            or record.binding.after_metadata_sha256 != digest
        ):
            raise UnsafeFilesystemTarget(
                "captured metadata does not match authorized metadata"
            )
        snapshot = FileMetadataSnapshot(
            kind="existing_file",
            identity=record.identity,
            binding=record.binding,
            canonical_metadata_sha256=digest,
        )
        self._metadata[id(snapshot)] = _IssuedMetadataSnapshot(
            snapshot=snapshot,
            target=target,
            identity=record.identity,
            binding=record.binding,
            native=native,
        )
        return snapshot

    def _snapshot_record(
        self,
        snapshot: object,
        binding: AuthorizedFileBinding,
        target: SafeTarget,
    ) -> _IssuedMetadataSnapshot:
        record = self._metadata.get(id(snapshot))
        if (
            record is None
            or record.snapshot is not snapshot
            or record.target is not target
            or record.identity != target.identity
            or record.binding is not binding
            or getattr(snapshot, "binding", None) is not binding
            or getattr(snapshot, "identity", None) != record.identity
            or getattr(snapshot, "canonical_metadata_sha256", None)
            != self._metadata_digest(record.native)
        ):
            raise UnsafeFilesystemTarget(
                "metadata snapshot is not an exact owner-issued snapshot"
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

    def _cleanup_unissued_temp(
        self,
        *,
        primary: BaseException,
        parent_fd: int,
        basename: str | None,
        file_fd: int | None,
    ) -> None:
        safe_to_unlink = False
        if file_fd is not None and basename is not None:
            try:
                safe_to_unlink = self._identity_of_fd(
                    file_fd
                ) == self._identity_at(parent_fd, basename)
            except BaseException as cleanup:
                self._add_cleanup_note(primary, cleanup)
        if file_fd is not None:
            try:
                self._adapter.close(file_fd)
            except BaseException as cleanup:
                self._add_cleanup_note(primary, cleanup)
        if safe_to_unlink and basename is not None:
            try:
                self._adapter.unlink(basename, dir_fd=parent_fd)
            except BaseException as cleanup:
                self._add_cleanup_note(primary, cleanup)
            try:
                self._adapter.fsync(parent_fd)
            except BaseException as cleanup:
                self._add_cleanup_note(primary, cleanup)
        try:
            self._adapter.close(parent_fd)
        except BaseException as cleanup:
            self._add_cleanup_note(primary, cleanup)

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
        basename: str | None = None
        try:
            flags = (
                int(self._adapter.O_RDWR)
                | int(self._adapter.O_CREAT)
                | int(self._adapter.O_EXCL)
                | int(self._adapter.O_NOFOLLOW)
            )
            for _ in range(128):
                candidate = (
                    f".mochi-{record.basename}.{secrets.token_hex(6)}"
                )
                try:
                    file_fd = int(
                        self._adapter.open(
                            candidate,
                            flags,
                            mode,
                            dir_fd=parent_fd,
                        )
                    )
                    basename = candidate
                    break
                except FileExistsError:
                    continue
            else:
                raise RuntimeError(
                    "unable to allocate descriptor-relative temp file"
                )

            fd_identity = self._identity_of_fd(file_fd)
            path_identity = self._identity_at(parent_fd, basename)
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
                target=target,
                basename=basename,
                identity=fd_identity,
                binding=record.binding,
                parent_fd=parent_fd,
                file_fd=file_fd,
            )
            return temp
        except BaseException as primary:
            self._cleanup_unissued_temp(
                primary=primary,
                parent_fd=parent_fd,
                basename=basename,
                file_fd=file_fd,
            )
            raise

    def write_temp(self, temp: StagedTemp, data: memoryview) -> int:
        record = self._validated_temp(temp)
        return int(self._adapter.write(record.file_fd, data))

    def apply_metadata_snapshot(
        self, temp: StagedTemp, snapshot: object
    ) -> None:
        record = self._validated_temp(temp)
        issued = self._snapshot_record(
            snapshot, record.binding, record.target
        )
        native = issued.native
        try:
            current = self._adapter.fstat(record.file_fd)
            if (
                int(current.st_uid) != native.uid
                or int(current.st_gid) != native.gid
            ):
                self._adapter.fchown(
                    record.file_fd, native.uid, native.gid
                )
            self._adapter.fchmod(record.file_fd, native.mode)
            current_names = sorted(
                self._raw_xattr_name(name)
                for name in self._adapter.listxattr(record.file_fd)
            )
            desired = dict(native.xattrs)
            for name in sorted(set(current_names) - set(desired)):
                self._adapter.removexattr(record.file_fd, name)
            acl_name = b"system.posix_acl_access"
            for name in sorted(
                name for name in desired if name != acl_name
            ):
                self._adapter.setxattr(
                    record.file_fd, name, desired[name]
                )
            if acl_name in desired:
                self._adapter.setxattr(
                    record.file_fd, acl_name, desired[acl_name]
                )
        except BaseException as cause:
            error = self._unsupported("apply", cause)
            raise error from cause

    @staticmethod
    def _hash_reader(read: Callable[[int, int], bytes]) -> str:
        digest = hashlib.sha256()
        offset = 0
        while True:
            try:
                chunk = read(1024 * 1024, offset)
            except InterruptedError:
                continue
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)
            offset += len(chunk)

    def _content_sha256(self, file_fd: int) -> str:
        return self._hash_reader(
            lambda size, offset: self._adapter.pread(
                file_fd, size, offset
            )
        )

    def verify_staged(
        self, temp: StagedTemp, snapshot: object
    ) -> None:
        record = self._validated_temp(temp)
        issued = self._snapshot_record(
            snapshot, record.binding, record.target
        )
        content_digest = self._content_sha256(record.file_fd)
        native = self._capture_native_metadata(record.file_fd)
        metadata_digest = self._metadata_digest(native)
        if (
            content_digest != record.binding.after_sha256
            or native != issued.native
            or metadata_digest
            != getattr(snapshot, "canonical_metadata_sha256", None)
            or metadata_digest != record.binding.after_metadata_sha256
        ):
            raise UnsafeFilesystemTarget(
                "staged content or metadata does not match authorization"
            )

    def flush_temp(self, temp: StagedTemp) -> None:
        record = self._validated_temp(temp)
        self._adapter.fsync(record.file_fd)

    def revalidate_base(
        self, target: SafeTarget, snapshot: object
    ) -> None:
        record = self._validated(target)
        issued = self._snapshot_record(
            snapshot, record.binding, target
        )
        content_digest = self._content_sha256(record.file_fd)
        native = self._capture_native_metadata(record.file_fd)
        metadata_digest = self._metadata_digest(native)
        if (
            content_digest != record.binding.base_sha256
            or native != issued.native
            or metadata_digest != record.binding.base_metadata_sha256
            or metadata_digest
            != getattr(snapshot, "canonical_metadata_sha256", None)
        ):
            raise UnsafeFilesystemTarget(
                "authorized base content or metadata changed before replace"
            )

    def discard_temp(self, temp: StagedTemp) -> None:
        record = self._temp_record(temp)
        error: BaseException | None = None
        safe_to_unlink = False
        try:
            safe_to_unlink = (
                self._identity_of_fd(record.file_fd)
                == record.identity
                == self._identity_at(record.parent_fd, record.basename)
            )
            if not safe_to_unlink:
                raise UnsafeFilesystemTarget(
                    "staged temp identity changed before discard"
                )
        except BaseException as exc:
            error = exc

        del self._temps[id(temp)]
        temp._mark_closed()  # pyright: ignore[reportPrivateUsage]
        if safe_to_unlink:
            error = self._remember_error(
                error,
                lambda: self._adapter.unlink(
                    record.basename, dir_fd=record.parent_fd
                ),
            )
        error = self._remember_error(
            error, lambda: self._adapter.close(record.file_fd)
        )
        error = self._remember_error(
            error, lambda: self._adapter.fsync(record.parent_fd)
        )
        error = self._remember_error(
            error, lambda: self._adapter.close(record.parent_fd)
        )
        if error is not None:
            raise error

    def _consume_temp(self, temp: StagedTemp) -> BaseException | None:
        record = self._temp_record(temp)
        del self._temps[id(temp)]
        temp._mark_closed()
        error: BaseException | None = None
        error = self._remember_error(
            error, lambda: self._adapter.close(record.file_fd)
        )
        error = self._remember_error(
            error, lambda: self._adapter.close(record.parent_fd)
        )
        return error

    def replace(
        self, source: StagedTemp, destination: SafeTarget
    ) -> FileIdentity:
        source_record = self._validated_temp(source)
        destination_record = self._validated(destination)
        if source_record.binding is not destination_record.binding:
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
        phase: str | None = None
        successor_identity: FileIdentity | None = None
        try:
            successor_identity = self._identity_at(
                destination_record.parent_fd,
                destination_record.basename,
            )
            if successor_identity != source_record.identity:
                raise UnsafeFilesystemTarget(
                    "replacement successor identity does not match staged file"
                )
        except BaseException as exc:
            error = exc
            phase = "successor_verification"

        try:
            self._adapter.fsync(destination_record.parent_fd)
        except BaseException as exc:
            if error is None:
                error = exc
                phase = "parent_fsync"
            else:
                self._add_cleanup_note(error, exc)

        temp_error = self._consume_temp(source)
        if temp_error is not None:
            if error is None:
                error = temp_error
                phase = "operand_cleanup"
            else:
                self._add_cleanup_note(error, temp_error)
        try:
            self.release_target(destination)
        except BaseException as exc:
            if error is None:
                error = exc
                phase = "operand_cleanup"
            else:
                self._add_cleanup_note(error, exc)

        if error is not None:
            outcome = CommittedFilesystemMutationError(
                phase=phase or "operand_cleanup", cause=error
            )
            raise outcome from error
        if successor_identity is None:
            cause = RuntimeError("replacement successor was not validated")
            outcome = CommittedFilesystemMutationError(
                phase="successor_verification", cause=cause
            )
            raise outcome from cause
        return successor_identity

    def release_target(self, target: SafeTarget) -> None:
        record = self._record(target)
        del self._issued[id(target)]
        for key, snapshot in tuple(self._metadata.items()):
            if snapshot.target is target:
                del self._metadata[key]
        target._mark_closed()
        error: BaseException | None = None
        error = self._remember_error(
            error, lambda: self._adapter.close(record.file_fd)
        )
        error = self._remember_error(
            error, lambda: self._adapter.close(record.parent_fd)
        )
        if error is not None:
            raise error

    def release_temp(self, temp: StagedTemp) -> None:
        self.discard_temp(temp)

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
