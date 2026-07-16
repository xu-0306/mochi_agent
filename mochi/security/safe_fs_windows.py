"""Windows handle-relative filesystem mutation backend."""

from __future__ import annotations

import ctypes
import os
import secrets
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, cast

from .file_contract import AuthorizationEnvelope, FileIdentity
from .safe_filesystem import (
    AuthorizedFileBinding,
    SafeFilesystemUnavailable,
    SafeTarget,
    UnsafeFilesystemTarget,
    resolve_authorized_file_binding,
)


@dataclass(frozen=True, slots=True)
class _WindowsPin:
    parent_handle: object
    file_handle: object
    owns_parent: bool


class _WindowsNativeAdapter:
    """Small ctypes binding for the handle-relative NT calls used below."""

    FILE_ATTRIBUTE_REPARSE_POINT = 0x400
    FILE_SHARE_ALL = 0x7
    FILE_OPEN = 1
    FILE_CREATE = 2
    FILE_DIRECTORY_FILE = 0x1
    FILE_NON_DIRECTORY_FILE = 0x40
    FILE_OPEN_REPARSE_POINT = 0x200000
    FILE_SYNCHRONOUS_IO_NONALERT = 0x20
    FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
    FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
    OPEN_EXISTING = 3
    DELETE = 0x00010000
    SYNCHRONIZE = 0x00100000
    FILE_LIST_DIRECTORY = 0x1
    FILE_READ_ATTRIBUTES = 0x80

    def __init__(self) -> None:
        self.available = False
        if os.name != "nt":
            return
        try:
            from ctypes import wintypes

            self._wintypes = wintypes
            self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            self._ntdll = ctypes.WinDLL("ntdll")
            self._CreateFileW = self._kernel32.CreateFileW
            self._CloseHandle = self._kernel32.CloseHandle
            self._GetFinalPathNameByHandleW = self._kernel32.GetFinalPathNameByHandleW
            self._GetFileInformationByHandle = self._kernel32.GetFileInformationByHandle
            self._GetFileInformationByHandleEx = (
                self._kernel32.GetFileInformationByHandleEx
            )
            self._NtCreateFile = self._ntdll.NtCreateFile
            self._NtSetInformationFile = self._ntdll.NtSetInformationFile
            self._configure_signatures()
            self.available = True
        except (AttributeError, OSError):
            self.available = False

    def _configure_signatures(self) -> None:
        wintypes = self._wintypes
        self._CreateFileW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        self._CreateFileW.restype = wintypes.HANDLE
        self._CloseHandle.argtypes = [wintypes.HANDLE]
        self._CloseHandle.restype = wintypes.BOOL

        self._GetFinalPathNameByHandleW.argtypes = [
            wintypes.HANDLE,
            wintypes.LPWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
        ]
        self._GetFinalPathNameByHandleW.restype = wintypes.DWORD
        self._GetFileInformationByHandle.argtypes = [
            wintypes.HANDLE,
            wintypes.LPVOID,
        ]
        self._GetFileInformationByHandle.restype = wintypes.BOOL
        self._GetFileInformationByHandleEx.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        self._GetFileInformationByHandleEx.restype = wintypes.BOOL

    @staticmethod
    def _nt_error(status: int, action: str) -> OSError:
        return OSError(f"{action} failed with NTSTATUS 0x{status & 0xFFFFFFFF:08x}")

    def createfile_workspace(self, path: str) -> object:
        access = self.FILE_LIST_DIRECTORY | self.FILE_READ_ATTRIBUTES | self.SYNCHRONIZE
        handle = self._CreateFileW(
            path,
            access,
            self.FILE_SHARE_ALL,
            None,
            self.OPEN_EXISTING,
            self.FILE_FLAG_BACKUP_SEMANTICS | self.FILE_FLAG_OPEN_REPARSE_POINT,
            None,
        )
        invalid = ctypes.c_void_p(-1).value
        if handle in {None, invalid}:
            raise ctypes.WinError(ctypes.get_last_error())
        return handle

    def _ntcreate(
        self, root: object, basename: str, *, directory: bool, disposition: int
    ) -> object:
        from ctypes import wintypes

        class UNICODE_STRING(ctypes.Structure):
            _fields_ = [
                ("Length", wintypes.USHORT),
                ("MaximumLength", wintypes.USHORT),
                ("Buffer", wintypes.LPWSTR),
            ]

        class OBJECT_ATTRIBUTES(ctypes.Structure):
            _fields_ = [
                ("Length", wintypes.ULONG),
                ("RootDirectory", wintypes.HANDLE),
                ("ObjectName", ctypes.POINTER(UNICODE_STRING)),
                ("Attributes", wintypes.ULONG),
                ("SecurityDescriptor", wintypes.LPVOID),
                ("SecurityQualityOfService", wintypes.LPVOID),
            ]

        class IO_STATUS_BLOCK(ctypes.Structure):
            _fields_ = [("Status", ctypes.c_ssize_t), ("Information", ctypes.c_size_t)]

        buffer = ctypes.create_unicode_buffer(basename)
        encoded_length = len(basename.encode("utf-16-le"))
        name = UNICODE_STRING(
            encoded_length,
            encoded_length + 2,
            ctypes.cast(buffer, wintypes.LPWSTR),
        )
        attrs = OBJECT_ATTRIBUTES(
            ctypes.sizeof(OBJECT_ATTRIBUTES), cast(Any, root), ctypes.pointer(name), 0, None, None
        )
        io = IO_STATUS_BLOCK()
        handle = wintypes.HANDLE()
        options = self.FILE_SYNCHRONOUS_IO_NONALERT | self.FILE_OPEN_REPARSE_POINT
        options |= self.FILE_DIRECTORY_FILE if directory else self.FILE_NON_DIRECTORY_FILE
        access = self.FILE_READ_ATTRIBUTES | self.SYNCHRONIZE | self.DELETE
        if directory:
            access |= self.FILE_LIST_DIRECTORY
        status = int(
            self._NtCreateFile(
                ctypes.byref(handle), access, ctypes.byref(attrs), ctypes.byref(io), None, 0,
                self.FILE_SHARE_ALL, disposition, options, None, 0,
            )
        )
        if status < 0:
            raise self._nt_error(status, "NtCreateFile")
        return handle

    def ntcreate_relative(self, root: object, basename: str, *, directory: bool) -> object:
        return self._ntcreate(root, basename, directory=directory, disposition=self.FILE_OPEN)

    def ntcreate_new_relative(self, root: object, basename: str) -> object:
        return self._ntcreate(root, basename, directory=False, disposition=self.FILE_CREATE)

    def final_path(self, handle: object) -> str:
        size = 512
        while size <= 32768:
            buffer = ctypes.create_unicode_buffer(size)
            length = int(self._GetFinalPathNameByHandleW(handle, buffer, size, 0))
            if length == 0:
                raise ctypes.WinError(ctypes.get_last_error())
            if length < size:
                return buffer.value
            size = length + 1
        raise OSError("final handle path is too long")

    def identity(self, handle: object) -> FileIdentity:
        from ctypes import wintypes

        FileIdInfo = 18

        class FILE_ID_128(ctypes.Structure):
            _fields_ = [("Identifier", ctypes.c_ubyte * 16)]

        class FILE_ID_INFO(ctypes.Structure):
            _fields_ = [
                ("VolumeSerialNumber", ctypes.c_ulonglong),
                ("FileId", FILE_ID_128),
            ]

        class BY_HANDLE_FILE_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("dwFileAttributes", wintypes.DWORD),
                ("ftCreationTime", wintypes.FILETIME),
                ("ftLastAccessTime", wintypes.FILETIME),
                ("ftLastWriteTime", wintypes.FILETIME),
                ("dwVolumeSerialNumber", wintypes.DWORD),
                ("nFileSizeHigh", wintypes.DWORD),
                ("nFileSizeLow", wintypes.DWORD),
                ("nNumberOfLinks", wintypes.DWORD),
                ("nFileIndexHigh", wintypes.DWORD),
                ("nFileIndexLow", wintypes.DWORD),
            ]

        legacy_info = BY_HANDLE_FILE_INFORMATION()
        if not self._GetFileInformationByHandle(
            handle, ctypes.byref(legacy_info)
        ):
            raise ctypes.WinError(ctypes.get_last_error())

        file_id_info = FILE_ID_INFO()
        if not self._GetFileInformationByHandleEx(
            handle,
            FileIdInfo,
            ctypes.byref(file_id_info),
            ctypes.sizeof(file_id_info),
        ):
            raise ctypes.WinError(ctypes.get_last_error())

        return FileIdentity(
            platform="windows",
            volume_id=str(int(file_id_info.VolumeSerialNumber)),
            file_id=bytes(file_id_info.FileId.Identifier).hex(),
            link_count=int(legacy_info.nNumberOfLinks),
            is_reparse_point=bool(
                int(legacy_info.dwFileAttributes)
                & self.FILE_ATTRIBUTE_REPARSE_POINT
            ),
        )

    def ntset_unlink(self, handle: object) -> None:
        class IO_STATUS_BLOCK(ctypes.Structure):
            _fields_ = [("Status", ctypes.c_ssize_t), ("Information", ctypes.c_size_t)]

        class FILE_DISPOSITION_INFORMATION(ctypes.Structure):
            _fields_ = [("DeleteFile", ctypes.c_ubyte)]

        io = IO_STATUS_BLOCK()
        disposition = FILE_DISPOSITION_INFORMATION(1)
        status = int(
            self._NtSetInformationFile(
                handle, ctypes.byref(io), ctypes.byref(disposition),
                ctypes.sizeof(disposition), 13,
            )
        )
        if status < 0:
            raise self._nt_error(status, "NtSetInformationFile(disposition)")

    def ntset_replace(self, handle: object, root: object, basename: str) -> None:
        from ctypes import wintypes

        class IO_STATUS_BLOCK(ctypes.Structure):
            _fields_ = [("Status", ctypes.c_ssize_t), ("Information", ctypes.c_size_t)]

        class FILE_RENAME_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("ReplaceIfExists", ctypes.c_ubyte),
                ("RootDirectory", wintypes.HANDLE),
                ("FileNameLength", wintypes.ULONG),
                ("FileName", wintypes.WCHAR * 1),
            ]

        encoded = basename.encode("utf-16-le")
        size = ctypes.sizeof(FILE_RENAME_INFORMATION) + len(encoded)
        buffer = ctypes.create_string_buffer(size)
        rename = cast(Any, ctypes.cast(buffer, ctypes.POINTER(FILE_RENAME_INFORMATION)).contents)
        rename.ReplaceIfExists = 1
        rename.RootDirectory = root
        rename.FileNameLength = len(encoded)
        offset = FILE_RENAME_INFORMATION.FileName.offset
        ctypes.memmove(ctypes.addressof(buffer) + offset, encoded, len(encoded))
        io = IO_STATUS_BLOCK()
        status = int(
            self._NtSetInformationFile(handle, ctypes.byref(io), buffer, size, 10)
        )
        if status < 0:
            raise self._nt_error(status, "NtSetInformationFile(rename)")

    def close(self, handle: object) -> None:
        if not self._CloseHandle(handle):
            raise ctypes.WinError(ctypes.get_last_error())


@dataclass(frozen=True, slots=True)
class _IssuedWindowsTarget:
    target: SafeTarget
    basename: str
    identity: FileIdentity
    binding: AuthorizedFileBinding
    pin: _WindowsPin


class WindowsSafeFilesystem:
    """Pin Windows handles and mutate through NT relative APIs only."""

    def __init__(
        self,
        workspace: str | Path,
        *,
        adapter: Any | None = None,
        enforce: bool = True,
    ) -> None:
        del enforce
        self._adapter = (
            adapter if adapter is not None else _WindowsNativeAdapter()
        )
        if not bool(getattr(self._adapter, "available", False)):
            raise SafeFilesystemUnavailable(
                "Windows native handle APIs are unavailable"
            )
        self._closed = False
        self._issued: dict[int, _IssuedWindowsTarget] = {}
        self._boundary = self._normalize_path(str(workspace))
        self._root_handle = self._adapter.createfile_workspace(str(workspace))
        try:
            self._adapter.final_path(self._root_handle)
            self._root_identity = self._adapter.identity(self._root_handle)
            if self._root_identity.is_reparse_point:
                raise UnsafeFilesystemTarget(
                    "workspace root is a reparse point"
                )
        except BaseException:
            self._adapter.close(self._root_handle)
            raise

    @staticmethod
    def _normalize_path(value: str) -> str:
        normalized = value.replace("/", "\\")
        if normalized.startswith("\\\\?\\"):
            normalized = normalized[4:]
        return normalized.rstrip("\\").casefold()

    def _assert_final_inside(self, handle: object) -> None:
        final = self._normalize_path(self._adapter.final_path(handle))
        if (
            final != self._boundary
            and not final.startswith(self._boundary + "\\")
        ):
            raise UnsafeFilesystemTarget(
                "final handle path escapes workspace"
            )

    @staticmethod
    def _segments(relative_path: str | Path) -> tuple[str, ...]:
        raw = str(relative_path)
        if not raw or "\x00" in raw:
            raise UnsafeFilesystemTarget(
                "invalid empty filesystem target"
            )
        path = PureWindowsPath(raw)
        if path.is_absolute() or path.drive or path.root:
            raise UnsafeFilesystemTarget(
                "absolute paths are not mutation capabilities"
            )
        if ".." in path.parts:
            raise UnsafeFilesystemTarget("parent traversal is forbidden")
        parts = tuple(
            part for part in path.parts if part not in {"", "."}
        )
        if not parts:
            raise UnsafeFilesystemTarget(
                "workspace root is not a file target"
            )
        return parts

    def _verify_handle(
        self, handle: object, *, hardlink: bool
    ) -> FileIdentity:
        self._assert_final_inside(handle)
        identity = self._adapter.identity(handle)
        if identity.is_reparse_point:
            raise UnsafeFilesystemTarget("reparse points are forbidden")
        if hardlink and identity.link_count != 1:
            raise UnsafeFilesystemTarget("hardlink targets are forbidden")
        return identity

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
        parts = self._segments(relative_path)
        parent = self._root_handle
        owns_parent = False
        try:
            for segment in parts[:-1]:
                next_handle = self._adapter.ntcreate_relative(
                    parent, segment, directory=True
                )
                self._verify_handle(next_handle, hardlink=False)
                if owns_parent:
                    self._adapter.close(parent)
                parent = next_handle
                owns_parent = True
            file_handle = self._adapter.ntcreate_relative(
                parent, parts[-1], directory=False
            )
            try:
                identity = self._verify_handle(
                    file_handle, hardlink=True
                )
                binding = resolve_authorized_file_binding(
                    canonical_relative_path="\\".join(parts).casefold(),
                    authorization=authorization,
                    captured_identity=identity,
                    canonicalize_authorized_path=lambda path: "\\".join(
                        self._segments(path)
                    ).casefold(),
                )
                pin = _WindowsPin(parent, file_handle, owns_parent)
                target = SafeTarget._create(
                    basename=parts[-1],
                    identity=identity,
                    authorization_digest=(
                        binding.authorization_digest
                    ),
                    owner=self,
                    parent=pin,
                )
                self._issued[id(target)] = _IssuedWindowsTarget(
                    target=target,
                    basename=parts[-1],
                    identity=identity,
                    binding=binding,
                    pin=pin,
                )
                return target
            except BaseException:
                self._adapter.close(file_handle)
                raise
        except BaseException:
            if owns_parent:
                self._adapter.close(parent)
            raise

    def _record(self, target: SafeTarget) -> _IssuedWindowsTarget:
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
            or target._parent is not record.pin
        ):
            raise UnsafeFilesystemTarget(
                "SafeTarget is not an issued capability"
            )
        return record

    def _validated(
        self, target: SafeTarget
    ) -> _IssuedWindowsTarget:
        record = self._record(target)
        if (
            self._verify_handle(record.pin.file_handle, hardlink=True)
            != record.identity
        ):
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
        self._adapter.ntset_unlink(record.pin.file_handle)
        self.release_target(target)

    def replace(
        self, source: SafeTarget, destination: SafeTarget
    ) -> None:
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
        self._adapter.ntset_replace(
            source_record.pin.file_handle,
            destination_record.pin.parent_handle,
            destination_record.basename,
        )
        self.release_target(source)
        self.release_target(destination)

    def create_temp(
        self, target: SafeTarget
    ) -> tuple[str, object]:
        record = self._validated(target)
        for _ in range(128):
            basename = (
                f".mochi-{record.basename}.{secrets.token_hex(6)}"
            )
            try:
                handle = self._adapter.ntcreate_new_relative(
                    record.pin.parent_handle, basename
                )
                self._verify_handle(handle, hardlink=True)
                return basename, handle
            except FileExistsError:
                continue
        raise RuntimeError(
            "unable to allocate handle-relative temp file"
        )

    def release_target(self, target: SafeTarget) -> None:
        record = self._record(target)
        del self._issued[id(target)]
        try:
            self._adapter.close(record.pin.file_handle)
        finally:
            try:
                if record.pin.owns_parent:
                    self._adapter.close(record.pin.parent_handle)
            finally:
                target._mark_closed()

    def close(self) -> None:
        if self._closed:
            return
        for record in tuple(self._issued.values()):
            try:
                self._adapter.close(record.pin.file_handle)
            finally:
                if record.pin.owns_parent:
                    self._adapter.close(record.pin.parent_handle)
                record.target._mark_closed()
        self._issued.clear()
        self._adapter.close(self._root_handle)
        self._closed = True

__all__ = ["WindowsSafeFilesystem"]
