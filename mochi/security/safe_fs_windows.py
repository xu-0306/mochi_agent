"""Windows handle-relative filesystem mutation backend."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import secrets
import sys
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from threading import RLock
from typing import Any, Literal, cast

from .file_contract import AuthorizationEnvelope, FileIdentity
from .safe_filesystem import (
    AuthorizedFileBinding,
    CommittedFilesystemMutationError,
    SafeFilesystemUnavailable,
    SafeTarget,
    StagedTemp,
    UnsafeFilesystemTarget,
    UnsupportedSecurityMetadata,
    resolve_authorized_file_binding,
)


@dataclass(frozen=True, slots=True)
class _WindowsPin:
    parent_handle: object
    file_handle: object
    owns_parent: bool


@dataclass(frozen=True, slots=True)
class _WindowsSecurityMetadata:
    raw_descriptor: bytes
    owner: bytes | str | None
    group: bytes | str | None
    dacl: bytes | str | None
    dacl_present: bool
    dacl_protected: bool
    sacl: bytes | str | None
    sacl_present: bool
    sacl_protected: bool
    sacl_state: Literal["included", "inaccessible"]


@dataclass(frozen=True, slots=True)
class _IssuedWindowsMetadataSnapshot:
    snapshot: object
    target: SafeTarget
    identity: FileIdentity
    binding: AuthorizedFileBinding
    native: _WindowsSecurityMetadata


class _WindowsNativeAdapter:
    """Small ctypes binding for the handle-relative NT calls used below."""

    semantics = frozenset({
        "content_read_at", "content_write", "file_flush", "directory_flush",
        "change_token", "security_capture", "security_apply",
        "relative_rename", "handle_disposition", "duplicate_handle",
    })
    platform = "win32"

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
    FILE_READ_DATA = 0x1
    FILE_WRITE_DATA = 0x2
    FILE_TRAVERSE = 0x20
    FILE_DELETE_CHILD = 0x40
    READ_CONTROL = 0x00020000
    WRITE_DAC = 0x00040000
    WRITE_OWNER = 0x00080000
    ACCESS_SYSTEM_SECURITY = 0x01000000
    STATUS_ACCESS_DENIED = 0xC0000022
    STATUS_PRIVILEGE_NOT_HELD = 0xC0000061
    STATUS_OBJECT_NAME_COLLISION = 0xC0000035

    def __init__(self) -> None:
        self.available = False
        if os.name != "nt":
            return
        try:
            from ctypes import wintypes

            self._wintypes = wintypes
            self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            self._advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
            self._ntdll = ctypes.WinDLL("ntdll")
            self._sacl_states: dict[int, str] = {}
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
        access = (
            self.FILE_LIST_DIRECTORY | self.FILE_READ_ATTRIBUTES
            | self.FILE_WRITE_DATA | self.FILE_TRAVERSE
            | self.FILE_DELETE_CHILD | self.SYNCHRONIZE
        )
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
        states = getattr(self, "_sacl_states", None)
        if states is not None:
            states[self._handle_key(handle)] = "inaccessible"
        return handle

    @staticmethod
    def _handle_key(handle: object) -> int:
        value = getattr(handle, "value", handle)
        return int(cast(Any, value))

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
        access = (
            self.FILE_READ_ATTRIBUTES | self.SYNCHRONIZE | self.DELETE
        )
        if directory:
            access |= (
                self.FILE_LIST_DIRECTORY | self.FILE_WRITE_DATA
                | self.FILE_TRAVERSE | self.FILE_DELETE_CHILD
            )
        else:
            access |= self.FILE_READ_DATA | self.READ_CONTROL
            if disposition == self.FILE_CREATE:
                access |= (
                    self.FILE_WRITE_DATA | self.WRITE_DAC | self.WRITE_OWNER
                )
        requested_access = (
            access
            if directory
            else access | self.ACCESS_SYSTEM_SECURITY
        )
        status = int(
            self._NtCreateFile(
                ctypes.byref(handle), requested_access, ctypes.byref(attrs),
                ctypes.byref(io), None, 0, self.FILE_SHARE_ALL,
                disposition, options, None, 0,
            )
        )
        sacl_state = "included"
        if (
            not directory
            and status & 0xFFFFFFFF
            in {self.STATUS_ACCESS_DENIED, self.STATUS_PRIVILEGE_NOT_HELD}
        ):
            handle = wintypes.HANDLE()
            io = IO_STATUS_BLOCK()
            status = int(
                self._NtCreateFile(
                    ctypes.byref(handle), access, ctypes.byref(attrs),
                    ctypes.byref(io), None, 0, self.FILE_SHARE_ALL,
                    disposition, options, None, 0,
                )
            )
            sacl_state = "inaccessible"
        if status < 0:
            if status & 0xFFFFFFFF == self.STATUS_OBJECT_NAME_COLLISION:
                raise FileExistsError(
                    f"relative name already exists: {basename}"
                )
            raise self._nt_error(status, "NtCreateFile")
        states = getattr(self, "_sacl_states", None)
        if states is not None:
            states[self._handle_key(handle)] = (
                "inaccessible" if directory else sacl_state
            )
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

        class FILE_RENAME_INFORMATION_EX(ctypes.Structure):
            _fields_ = [
                ("Flags", wintypes.ULONG),
                ("RootDirectory", wintypes.HANDLE),
                ("FileNameLength", wintypes.ULONG),
                ("FileName", wintypes.WCHAR * 1),
            ]

        encoded = basename.encode("utf-16-le")
        size = ctypes.sizeof(FILE_RENAME_INFORMATION_EX) + len(encoded)
        buffer = ctypes.create_string_buffer(size)
        rename = cast(
            Any,
            ctypes.cast(
                buffer, ctypes.POINTER(FILE_RENAME_INFORMATION_EX)
            ).contents,
        )
        rename.Flags = 0x1 | 0x2
        rename.RootDirectory = root
        rename.FileNameLength = len(encoded)
        offset = FILE_RENAME_INFORMATION_EX.FileName.offset
        ctypes.memmove(ctypes.addressof(buffer) + offset, encoded, len(encoded))
        io = IO_STATUS_BLOCK()
        status = int(
            self._NtSetInformationFile(handle, ctypes.byref(io), buffer, size, 65)
        )
        if status < 0:
            raise self._nt_error(status, "NtSetInformationFile(rename-ex)")


    def duplicate_handle(self, handle: object) -> object:
        from ctypes import wintypes
        duplicate = wintypes.HANDLE()
        process = self._kernel32.GetCurrentProcess()
        self._kernel32.DuplicateHandle.argtypes = [
            wintypes.HANDLE, wintypes.HANDLE, wintypes.HANDLE,
            ctypes.POINTER(wintypes.HANDLE), wintypes.DWORD,
            wintypes.BOOL, wintypes.DWORD,
        ]
        self._kernel32.DuplicateHandle.restype = wintypes.BOOL
        if not self._kernel32.DuplicateHandle(
            process, handle, process, ctypes.byref(duplicate),
            0, False, 2,
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        self._sacl_states[self._handle_key(duplicate)] = self._sacl_states.get(
            self._handle_key(handle), "inaccessible"
        )
        return duplicate

    def sacl_access(
        self, handle: object,
    ) -> Literal["included", "inaccessible"]:
        return cast(
            Literal["included", "inaccessible"],
            self._sacl_states.get(self._handle_key(handle), "inaccessible"),
        )

    def read_at(self, handle: object, size: int, offset: int) -> bytes:
        from ctypes import wintypes
        class OVERLAPPED(ctypes.Structure):
            _fields_ = [
                ("Internal", ctypes.c_size_t), ("InternalHigh", ctypes.c_size_t),
                ("Offset", wintypes.DWORD), ("OffsetHigh", wintypes.DWORD),
                ("hEvent", wintypes.HANDLE),
            ]
        buffer = ctypes.create_string_buffer(size)
        read = wintypes.DWORD()
        overlap = OVERLAPPED()
        overlap.Offset = offset & 0xFFFFFFFF
        overlap.OffsetHigh = (offset >> 32) & 0xFFFFFFFF
        fn = self._kernel32.ReadFile
        fn.argtypes = [
            wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(OVERLAPPED),
        ]
        fn.restype = wintypes.BOOL
        if not fn(handle, buffer, size, ctypes.byref(read), ctypes.byref(overlap)):
            code = ctypes.get_last_error()
            if code == 38:
                return b""
            raise ctypes.WinError(code)
        return buffer.raw[: int(read.value)]

    def write(self, handle: object, data: memoryview) -> int:
        from ctypes import wintypes
        raw = bytes(data)
        buffer = ctypes.create_string_buffer(raw)
        written = wintypes.DWORD()
        fn = self._kernel32.WriteFile
        fn.argtypes = [
            wintypes.HANDLE, wintypes.LPCVOID, wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID,
        ]
        fn.restype = wintypes.BOOL
        if not fn(handle, buffer, len(raw), ctypes.byref(written), None):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(written.value)

    def flush_file(self, handle: object) -> None:
        from ctypes import wintypes
        fn = self._kernel32.FlushFileBuffers
        fn.argtypes = [wintypes.HANDLE]
        fn.restype = wintypes.BOOL
        if not fn(handle):
            raise ctypes.WinError(ctypes.get_last_error())

    def flush_directory(self, handle: object) -> None:
        class IO_STATUS_BLOCK(ctypes.Structure):
            _fields_ = [
                ("Status", ctypes.c_ssize_t), ("Information", ctypes.c_size_t)
            ]
        io = IO_STATUS_BLOCK()
        status = int(self._ntdll.NtFlushBuffersFile(handle, ctypes.byref(io)))
        if status < 0:
            raise self._nt_error(status, "NtFlushBuffersFile")

    def change_token(self, handle: object) -> object:
        from ctypes import wintypes
        class FILE_BASIC_INFO(ctypes.Structure):
            _fields_ = [
                ("CreationTime", ctypes.c_longlong),
                ("LastAccessTime", ctypes.c_longlong),
                ("LastWriteTime", ctypes.c_longlong),
                ("ChangeTime", ctypes.c_longlong),
                ("FileAttributes", wintypes.DWORD),
            ]
        class FILE_STANDARD_INFO(ctypes.Structure):
            _fields_ = [
                ("AllocationSize", ctypes.c_longlong),
                ("EndOfFile", ctypes.c_longlong),
                ("NumberOfLinks", wintypes.DWORD),
                ("DeletePending", wintypes.BOOL),
                ("Directory", wintypes.BOOL),
            ]
        basic = FILE_BASIC_INFO()
        standard = FILE_STANDARD_INFO()
        for info_class, value in ((0, basic), (1, standard)):
            if not self._GetFileInformationByHandleEx(
                handle, info_class, ctypes.byref(value), ctypes.sizeof(value)
            ):
                raise ctypes.WinError(ctypes.get_last_error())
        return (
            self.identity(handle), int(standard.EndOfFile),
            int(basic.LastWriteTime), int(basic.ChangeTime),
            bool(standard.DeletePending),
        )


    def _security_component_sddl(
        self, descriptor: object, information: int,
    ) -> str:
        from ctypes import wintypes

        convert = (
            self._advapi32
            .ConvertSecurityDescriptorToStringSecurityDescriptorW
        )
        text = wintypes.LPWSTR()
        length = wintypes.ULONG()
        convert.argtypes = [
            wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD,
            ctypes.POINTER(wintypes.LPWSTR),
            ctypes.POINTER(wintypes.ULONG),
        ]
        convert.restype = wintypes.BOOL
        if not convert(
            descriptor, 1, information, ctypes.byref(text),
            ctypes.byref(length),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            return text.value or ""
        finally:
            self._kernel32.LocalFree(text)

    @staticmethod
    def _security_metadata_from_components(
        *,
        raw_descriptor: bytes,
        owner: bytes | str | None,
        group: bytes | str | None,
        dacl: bytes | str | None,
        dacl_present: bool,
        dacl_protected: bool,
        sacl: bytes | str | None,
        sacl_present: bool,
        sacl_protected: bool,
        sacl_state: Literal["included", "inaccessible"],
    ) -> _WindowsSecurityMetadata:
        return _WindowsSecurityMetadata(
            raw_descriptor=raw_descriptor,
            owner=owner,
            group=group,
            dacl=dacl,
            dacl_present=dacl_present,
            dacl_protected=dacl_protected,
            sacl=sacl,
            sacl_present=sacl_present,
            sacl_protected=sacl_protected,
            sacl_state=sacl_state,
        )

    def security_descriptor(
        self, handle: object, *, include_sacl: bool,
    ) -> _WindowsSecurityMetadata:
        from ctypes import wintypes

        OWNER = 0x1
        GROUP = 0x2
        DACL = 0x4
        SACL = 0x8
        info = OWNER | GROUP | DACL | (SACL if include_sacl else 0)
        get_security = self._advapi32.GetKernelObjectSecurity
        get_security.argtypes = [
            wintypes.HANDLE, wintypes.DWORD, wintypes.LPVOID,
            wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
        ]
        get_security.restype = wintypes.BOOL
        needed = wintypes.DWORD()
        get_security(handle, info, None, 0, ctypes.byref(needed))
        if not needed.value:
            raise ctypes.WinError(ctypes.get_last_error())
        buffer = ctypes.create_string_buffer(needed.value)
        if not get_security(
            handle, info, buffer, needed.value, ctypes.byref(needed)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        raw = bytes(buffer.raw[: needed.value])

        owner_pointer = wintypes.LPVOID()
        owner_defaulted = wintypes.BOOL()
        get_owner = self._advapi32.GetSecurityDescriptorOwner
        get_owner.argtypes = [
            wintypes.LPVOID, ctypes.POINTER(wintypes.LPVOID),
            ctypes.POINTER(wintypes.BOOL),
        ]
        get_owner.restype = wintypes.BOOL
        if not get_owner(
            buffer, ctypes.byref(owner_pointer),
            ctypes.byref(owner_defaulted),
        ):
            raise ctypes.WinError(ctypes.get_last_error())

        group_pointer = wintypes.LPVOID()
        group_defaulted = wintypes.BOOL()
        get_group = self._advapi32.GetSecurityDescriptorGroup
        get_group.argtypes = [
            wintypes.LPVOID, ctypes.POINTER(wintypes.LPVOID),
            ctypes.POINTER(wintypes.BOOL),
        ]
        get_group.restype = wintypes.BOOL
        if not get_group(
            buffer, ctypes.byref(group_pointer),
            ctypes.byref(group_defaulted),
        ):
            raise ctypes.WinError(ctypes.get_last_error())

        dacl_present_value = wintypes.BOOL()
        dacl_pointer = wintypes.LPVOID()
        dacl_defaulted = wintypes.BOOL()
        get_dacl = self._advapi32.GetSecurityDescriptorDacl
        get_dacl.argtypes = [
            wintypes.LPVOID, ctypes.POINTER(wintypes.BOOL),
            ctypes.POINTER(wintypes.LPVOID), ctypes.POINTER(wintypes.BOOL),
        ]
        get_dacl.restype = wintypes.BOOL
        if not get_dacl(
            buffer, ctypes.byref(dacl_present_value),
            ctypes.byref(dacl_pointer), ctypes.byref(dacl_defaulted),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        dacl_present = bool(dacl_present_value.value)

        sacl_present = False
        sacl_pointer = wintypes.LPVOID()
        if include_sacl:
            sacl_present_value = wintypes.BOOL()
            sacl_defaulted = wintypes.BOOL()
            get_sacl = self._advapi32.GetSecurityDescriptorSacl
            get_sacl.argtypes = [
                wintypes.LPVOID, ctypes.POINTER(wintypes.BOOL),
                ctypes.POINTER(wintypes.LPVOID), ctypes.POINTER(wintypes.BOOL),
            ]
            get_sacl.restype = wintypes.BOOL
            if not get_sacl(
                buffer, ctypes.byref(sacl_present_value),
                ctypes.byref(sacl_pointer), ctypes.byref(sacl_defaulted),
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            sacl_present = bool(sacl_present_value.value)

        control = wintypes.WORD()
        revision = wintypes.DWORD()
        get_control = self._advapi32.GetSecurityDescriptorControl
        get_control.argtypes = [
            wintypes.LPVOID, ctypes.POINTER(wintypes.WORD),
            ctypes.POINTER(wintypes.DWORD),
        ]
        get_control.restype = wintypes.BOOL
        if not get_control(
            buffer, ctypes.byref(control), ctypes.byref(revision)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        flags = int(control.value)

        owner = (
            self._security_component_sddl(buffer, OWNER)
            if owner_pointer.value else None
        )
        group = (
            self._security_component_sddl(buffer, GROUP)
            if group_pointer.value else None
        )
        dacl = (
            self._security_component_sddl(buffer, DACL)
            if dacl_present and dacl_pointer.value else None
        )
        sacl = (
            self._security_component_sddl(buffer, SACL)
            if include_sacl and sacl_present and sacl_pointer.value else None
        )
        return self._security_metadata_from_components(
            raw_descriptor=raw,
            owner=owner,
            group=group,
            dacl=dacl,
            dacl_present=dacl_present,
            dacl_protected=bool(flags & 0x1000),
            sacl=sacl,
            sacl_present=sacl_present,
            sacl_protected=bool(flags & 0x2000) if include_sacl else False,
            sacl_state="included" if include_sacl else "inaccessible",
        )

    def apply_security_descriptor(
        self, handle: object, metadata: _WindowsSecurityMetadata,
    ) -> None:
        from ctypes import wintypes
        info = 0x1 | 0x2 | 0x4
        info |= 0x80000000 if metadata.dacl_protected else 0x20000000
        if metadata.sacl_state == "included":
            info |= 0x8
            info |= 0x40000000 if metadata.sacl_protected else 0x10000000
        buffer = ctypes.create_string_buffer(metadata.raw_descriptor)
        fn = self._advapi32.SetKernelObjectSecurity
        fn.argtypes = [
            wintypes.HANDLE, wintypes.DWORD, wintypes.LPVOID
        ]
        fn.restype = wintypes.BOOL
        if not fn(handle, info, buffer):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self, handle: object) -> None:
        self._sacl_states.pop(self._handle_key(handle), None)
        if not self._CloseHandle(handle):
            raise ctypes.WinError(ctypes.get_last_error())


@dataclass(frozen=True, slots=True)
class _IssuedWindowsTarget:
    target: SafeTarget
    basename: str
    identity: FileIdentity
    binding: AuthorizedFileBinding
    pin: _WindowsPin


@dataclass(frozen=True, slots=True)
class _IssuedWindowsTemp:
    temp: StagedTemp
    basename: str
    identity: FileIdentity
    binding: AuthorizedFileBinding
    parent_handle: object
    file_handle: object


class _WindowsStructuralSafeFilesystem:
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
        self._temps: dict[int, _IssuedWindowsTemp] = {}
        self._root_handle = self._adapter.createfile_workspace(
            str(workspace)
        )
        try:
            root_final = self._adapter.final_path(
                self._root_handle
            )
            self._boundary = self._normalize_path(root_final)
            self._root_identity = self._adapter.identity(
                self._root_handle
            )
            if self._root_identity.is_reparse_point:
                raise UnsafeFilesystemTarget(
                    "workspace root is a reparse point"
                )
        except BaseException as primary:
            self._remember_error(
                primary,
                lambda: self._adapter.close(self._root_handle),
            )
            raise

    @staticmethod
    def _normalize_path(value: str) -> str:
        normalized = value.replace("/", "\\")
        if normalized.startswith("\\\\?\\"):
            normalized = normalized[4:]
        return normalized.rstrip("\\").casefold()

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

    def _assert_normalized_inside(self, final: str) -> None:
        if (
            final != self._boundary
            and not final.startswith(self._boundary + "\\")
        ):
            raise UnsafeFilesystemTarget(
                "final handle path escapes workspace"
            )

    def _assert_final_inside(self, handle: object) -> None:
        final = self._normalize_path(
            self._adapter.final_path(handle)
        )
        self._assert_normalized_inside(final)

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
                try:
                    self._verify_handle(
                        next_handle, hardlink=False
                    )
                except BaseException as primary:
                    self._remember_error(
                        primary,
                        lambda handle=next_handle: self._adapter.close(
                            handle
                        ),
                    )
                    raise
                if owns_parent:
                    owned_parent = parent
                    owns_parent = False
                    try:
                        self._adapter.close(owned_parent)
                    except BaseException as primary:
                        self._remember_error(
                            primary,
                            lambda handle=next_handle: self._adapter.close(
                                handle
                            ),
                        )
                        raise
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
            except BaseException as primary:
                self._remember_error(
                    primary,
                    lambda: self._adapter.close(file_handle),
                )
                raise
        except BaseException as primary:
            if owns_parent:
                owns_parent = False
                self._remember_error(
                    primary,
                    lambda: self._adapter.close(parent),
                )
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

    def _temp_record(
        self, temp: StagedTemp
    ) -> _IssuedWindowsTemp:
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
            or temp._parent is not record.parent_handle  # pyright: ignore[reportPrivateUsage]
        ):
            raise UnsafeFilesystemTarget(
                "StagedTemp is not an issued capability"
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

    def _validated_temp(
        self, temp: StagedTemp
    ) -> _IssuedWindowsTemp:
        record = self._temp_record(temp)
        if (
            self._verify_handle(
                record.file_handle, hardlink=True
            )
            != record.identity
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
        self._adapter.ntset_unlink(record.pin.file_handle)
        self.release_target(target)

    def create_temp(self, target: SafeTarget) -> StagedTemp:
        record = self._validated(target)
        if record.binding.operation not in {"update", "rename"}:
            raise UnsafeFilesystemTarget(
                "temp creation requires update or rename authorization"
            )
        for _ in range(128):
            basename = (
                f".mochi-{record.basename}."
                f"{secrets.token_hex(6)}"
            )
            try:
                handle = self._adapter.ntcreate_new_relative(
                    record.pin.parent_handle, basename
                )
            except FileExistsError:
                continue
            try:
                identity = self._verify_handle(
                    handle, hardlink=True
                )
            except BaseException:
                with suppress(BaseException):
                    self._adapter.close(handle)
                raise
            temp = StagedTemp._create(
                basename=basename,
                identity=identity,
                binding=record.binding,
                owner=self,
                parent=record.pin.parent_handle,
            )
            self._temps[id(temp)] = _IssuedWindowsTemp(
                temp=temp,
                basename=basename,
                identity=identity,
                binding=record.binding,
                parent_handle=record.pin.parent_handle,
                file_handle=handle,
            )
            return temp
        raise RuntimeError(
            "unable to allocate handle-relative temp file"
        )

    def replace(
        self, source: StagedTemp, destination: SafeTarget
    ) -> None:
        source_record = self._validated_temp(source)
        destination_record = self._validated(destination)
        if source_record.binding != destination_record.binding:
            raise UnsafeFilesystemTarget(
                "replace operands require the same authorization binding"
            )
        expected_destination_path = self._normalize_path(
            self._adapter.final_path(
                destination_record.pin.file_handle
            )
        )
        self._assert_normalized_inside(
            expected_destination_path
        )
        self._adapter.ntset_replace(
            source_record.file_handle,
            destination_record.pin.parent_handle,
            destination_record.basename,
        )

        error: BaseException | None = None
        phase = "successor_verification"
        successor: object | None = None
        try:
            source_identity = self._verify_handle(
                source_record.file_handle, hardlink=True
            )
            if source_identity != source_record.identity:
                raise UnsafeFilesystemTarget(
                    "source identity changed after replace"
                )
            source_path = self._normalize_path(
                self._adapter.final_path(
                    source_record.file_handle
                )
            )
            if source_path != expected_destination_path:
                raise UnsafeFilesystemTarget(
                    "source final path does not match destination"
                )

            successor = self._adapter.ntcreate_relative(
                destination_record.pin.parent_handle,
                destination_record.basename,
                directory=False,
            )
            successor_identity = self._verify_handle(
                successor, hardlink=True
            )
            if successor_identity != source_record.identity:
                raise UnsafeFilesystemTarget(
                    "successor identity does not match staged source"
                )
            successor_path = self._normalize_path(
                self._adapter.final_path(successor)
            )
            if successor_path != expected_destination_path:
                raise UnsafeFilesystemTarget(
                    "successor final path does not match destination"
                )
        except BaseException as exc:
            error = exc
        finally:
            had_error = error is not None
            if successor is not None:
                error = self._remember_error(
                    error,
                    lambda: self._adapter.close(successor),
                )
            error = self._remember_error(
                error, lambda: self.release_temp(source)
            )
            error = self._remember_error(
                error,
                lambda: self.release_target(destination),
            )
            if not had_error and error is not None:
                phase = "operand_cleanup"
        if error is not None:
            outcome = CommittedFilesystemMutationError(
                phase=phase, cause=error
            )
            raise outcome from error

    def release_target(self, target: SafeTarget) -> None:
        record = self._record(target)
        del self._issued[id(target)]
        error: BaseException | None = None
        error = self._remember_error(
            error,
            lambda: self._adapter.close(
                record.pin.file_handle
            ),
        )
        if record.pin.owns_parent:
            error = self._remember_error(
                error,
                lambda: self._adapter.close(
                    record.pin.parent_handle
                ),
            )
        target._mark_closed()
        if error is not None:
            raise error

    def release_temp(self, temp: StagedTemp) -> None:
        record = self._temp_record(temp)
        del self._temps[id(temp)]
        error = self._remember_error(
            None,
            lambda: self._adapter.close(record.file_handle),
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
                lambda item=record: self.release_target(
                    item.target
                ),
            )
        error = self._remember_error(
            error,
            lambda: self._adapter.close(self._root_handle),
        )
        self._closed = True
        if error is not None:
            raise error




@dataclass(frozen=True, slots=True)
class _AtomicIssuedWindowsTemp(_IssuedWindowsTemp):
    temp: StagedTemp
    target: SafeTarget
    basename: str
    identity: FileIdentity
    binding: AuthorizedFileBinding
    parent_handle: object
    file_handle: object


class WindowsSafeFilesystem(_WindowsStructuralSafeFilesystem):
    _REQUIRED_SEMANTICS = frozenset({
        "content_read_at", "content_write", "file_flush", "directory_flush",
        "change_token", "security_capture", "security_apply",
        "relative_rename", "handle_disposition", "duplicate_handle",
    })

    @classmethod
    def _validate_adapter_semantics(cls, adapter: Any) -> None:
        declared = getattr(adapter, "semantics", None)
        declared_set = cast(set[str] | frozenset[str], declared)
        missing = (
            sorted(cls._REQUIRED_SEMANTICS)
            if not isinstance(declared, (set, frozenset))
            else sorted(cls._REQUIRED_SEMANTICS - set(declared_set))
        )
        if missing:
            raise SafeFilesystemUnavailable(
                "Windows adapter lacks required semantics: " + ", ".join(missing)
            )

    def __init__(
        self, workspace: str | Path, *, adapter: Any | None = None,
        enforce: bool = True,
    ) -> None:
        chosen = adapter if adapter is not None else _WindowsNativeAdapter()
        if not bool(getattr(chosen, "available", False)):
            raise SafeFilesystemUnavailable("Windows native handle APIs are unavailable")
        self._validate_adapter_semantics(chosen)
        super().__init__(workspace, adapter=chosen, enforce=enforce)
        self._lock = RLock()
        self._platform = str(getattr(chosen, "platform", sys.platform))
        self._metadata: dict[int, _IssuedWindowsMetadataSnapshot] = {}

    @staticmethod
    def _add_cleanup_note(primary: BaseException, additional: BaseException) -> None:
        try:
            try:
                detail = str(additional)
            except BaseException:
                detail = f"<{type(additional).__name__} could not be formatted>"
            BaseException.add_note(primary, f"additional cleanup failure: {detail}")
        except BaseException:
            pass

    @staticmethod
    def _remember_error(
        first: BaseException | None, action: Callable[[], None],
    ) -> BaseException | None:
        try:
            action()
        except BaseException as exc:
            if first is None:
                return exc
            WindowsSafeFilesystem._add_cleanup_note(first, exc)
        return first

    def _unsupported(
        self, phase: str, cause: BaseException,
    ) -> UnsupportedSecurityMetadata:
        return UnsupportedSecurityMetadata(
            phase=phase, platform=self._platform, cause=cause
        )

    def prepare_target(
        self, relative_path: str | Path, authorization: AuthorizationEnvelope,
    ) -> SafeTarget:
        with self._lock:
            return super().prepare_target(relative_path, authorization)

    @staticmethod
    def _metadata_value(value: bytes | str | None) -> str | None:
        if isinstance(value, bytes):
            return value.hex()
        if isinstance(value, str) or value is None:
            return value
        raise TypeError("security metadata field must be bytes, str, or None")

    @classmethod
    def _metadata_payload(
        cls, native: _WindowsSecurityMetadata,
    ) -> dict[str, object]:
        return {
            "owner": cls._metadata_value(native.owner),
            "group": cls._metadata_value(native.group),
            "dacl": cls._metadata_value(native.dacl),
            "dacl_present": native.dacl_present,
            "dacl_protected": native.dacl_protected,
            "sacl": cls._metadata_value(native.sacl),
            "sacl_present": native.sacl_present,
            "sacl_protected": native.sacl_protected,
            "sacl_state": native.sacl_state,
        }

    @classmethod
    def _metadata_digest(cls, native: _WindowsSecurityMetadata) -> str:
        raw = json.dumps(
            cls._metadata_payload(native), sort_keys=True, separators=(",", ":")
        ).encode("ascii")
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _coerce_metadata(value: object) -> _WindowsSecurityMetadata:
        state = getattr(value, "sacl_state", None)
        if state not in {"included", "inaccessible"}:
            raise RuntimeError("SACL access must be explicit: included or inaccessible")
        item = cast(Any, value)
        return _WindowsSecurityMetadata(
            raw_descriptor=bytes(item.raw_descriptor),
            owner=item.owner, group=item.group,
            dacl=item.dacl,
            dacl_present=bool(item.dacl_present),
            dacl_protected=bool(item.dacl_protected),
            sacl=item.sacl,
            sacl_present=bool(item.sacl_present),
            sacl_protected=bool(item.sacl_protected),
            sacl_state=cast(Literal["included", "inaccessible"], state),
        )

    def _capture_native_metadata(
        self, handle: object, *, phase: str,
    ) -> _WindowsSecurityMetadata:
        try:
            access = getattr(self._adapter, "sacl_access", None)
            state = (
                access(handle) if callable(access)
                else getattr(self._adapter, "sacl_state", "inaccessible")
            )
            if state not in {"included", "inaccessible"}:
                raise RuntimeError("adapter did not explicitly represent SACL access")
            value = self._adapter.security_descriptor(
                handle, include_sacl=state == "included"
            )
            native = self._coerce_metadata(value)
            if native.sacl_state != state:
                raise RuntimeError("security descriptor SACL state is inconsistent")
            return native
        except BaseException as cause:
            error = self._unsupported(phase, cause)
            raise error from cause

    @staticmethod
    def _hash_reader(read: Callable[[int, int], bytes]) -> str:
        digest = hashlib.sha256()
        offset = 0
        while True:
            try:
                data = read(1024 * 1024, offset)
            except InterruptedError:
                continue
            if not data:
                return digest.hexdigest()
            digest.update(data)
            offset += len(data)

    def _content_sha256(self, handle: object, *, phase: str) -> str:
        try:
            return self._hash_reader(
                lambda size, offset: bytes(
                    self._adapter.read_at(handle, size, offset)
                )
            )
        except BaseException as cause:
            error = self._unsupported(phase, cause)
            raise error from cause

    def transaction_binding(
        self, target: SafeTarget,
    ) -> AuthorizedFileBinding:
        with self._lock:
            return self._validated(target).binding

    def capture_metadata(self, target: SafeTarget):
        from ..tools.file_transaction import FileMetadataSnapshot
        with self._lock:
            record = self._validated(target)
            native = self._capture_native_metadata(
                record.pin.file_handle, phase="capture"
            )
            digest = self._metadata_digest(native)
            if (
                record.binding.base_metadata_sha256 != digest
                or record.binding.after_metadata_sha256 != digest
            ):
                raise UnsafeFilesystemTarget(
                    "captured metadata does not match authorized metadata"
                )
            snapshot = FileMetadataSnapshot(
                kind="existing_file", identity=record.identity,
                binding=record.binding, canonical_metadata_sha256=digest,
            )
            for key, issued in tuple(self._metadata.items()):
                if issued.target is target:
                    del self._metadata[key]
            self._metadata[id(snapshot)] = _IssuedWindowsMetadataSnapshot(
                snapshot=snapshot, target=target, identity=record.identity,
                binding=record.binding, native=native,
            )
            return snapshot


    def _snapshot_record(
        self, snapshot: object, binding: AuthorizedFileBinding,
        target: SafeTarget,
    ) -> _IssuedWindowsMetadataSnapshot:
        record = self._metadata.get(id(snapshot))
        if (
            record is None or record.snapshot is not snapshot
            or record.target is not target or record.identity != target.identity
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

    def _commit_snapshot_record(
        self, binding: AuthorizedFileBinding, target: SafeTarget,
    ) -> _IssuedWindowsMetadataSnapshot:
        issued = next(
            (
                candidate
                for candidate in self._metadata.values()
                if candidate.target is target
            ),
            None,
        )
        if issued is None:
            raise UnsafeFilesystemTarget(
                "metadata snapshot is not an exact owner-issued snapshot"
            )
        return self._snapshot_record(issued.snapshot, binding, target)

    def _temp_record(self, temp: StagedTemp) -> _AtomicIssuedWindowsTemp:
        if not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
            temp, StagedTemp
        ):
            raise TypeError("replace source must be a StagedTemp capability")
        record = self._temps.get(id(temp))
        if (
            not isinstance(record, _AtomicIssuedWindowsTemp)
            or record.temp is not temp
            or not temp._is_authentic()  # pyright: ignore[reportPrivateUsage]
            or temp._owner is not self  # pyright: ignore[reportPrivateUsage]
            or temp.closed
            or temp.basename != record.basename
            or temp.identity != record.identity
            or temp.binding is not record.binding
            or temp.authorization_digest != record.binding.authorization_digest
            or temp._parent is not record.parent_handle  # pyright: ignore[reportPrivateUsage]
        ):
            raise UnsafeFilesystemTarget("StagedTemp is not an issued capability")
        return record

    def _validated_temp(self, temp: StagedTemp) -> _AtomicIssuedWindowsTemp:
        record = self._temp_record(temp)
        if self._verify_handle(record.file_handle, hardlink=True) != record.identity:
            raise UnsafeFilesystemTarget(
                "staged temp identity changed after issuance"
            )
        return record

    def create_temp(self, target: SafeTarget) -> StagedTemp:
        with self._lock:
            record = self._validated(target)
            if record.binding.operation not in {"update", "rename"}:
                raise UnsafeFilesystemTarget(
                    "temp creation requires update or rename authorization"
                )
            parent = self._adapter.duplicate_handle(record.pin.parent_handle)
            handle: object | None = None
            try:
                for _ in range(128):
                    basename = f".mochi-{record.basename}.{secrets.token_hex(6)}"
                    try:
                        handle = self._adapter.ntcreate_new_relative(parent, basename)
                    except FileExistsError:
                        continue
                    break
                else:
                    raise RuntimeError(
                        "unable to allocate handle-relative temp file"
                    )
                identity = self._verify_handle(handle, hardlink=True)
                temp = StagedTemp._create(  # pyright: ignore[reportPrivateUsage]
                    basename=basename, identity=identity, binding=record.binding,
                    owner=self, parent=parent,
                )
                self._temps[id(temp)] = _AtomicIssuedWindowsTemp(
                    temp=temp, target=target, basename=basename,
                    identity=identity, binding=record.binding,
                    parent_handle=parent, file_handle=handle,
                )
                return temp
            except BaseException as primary:
                if handle is not None:
                    self._remember_error(
                        primary, lambda: self._adapter.ntset_unlink(handle)
                    )
                    self._remember_error(
                        primary, lambda: self._adapter.close(handle)
                    )
                self._remember_error(
                    primary, lambda: self._adapter.flush_directory(parent)
                )
                self._remember_error(
                    primary, lambda: self._adapter.close(parent)
                )
                raise

    def write_temp(self, temp: StagedTemp, data: memoryview) -> int:
        with self._lock:
            return int(self._adapter.write(
                self._validated_temp(temp).file_handle, data
            ))


    def apply_metadata_snapshot(
        self, temp: StagedTemp, snapshot: object,
    ) -> None:
        with self._lock:
            record = self._validated_temp(temp)
            issued = self._snapshot_record(
                snapshot, record.binding, record.target
            )
            try:
                self._adapter.apply_security_descriptor(
                    record.file_handle, issued.native
                )
            except BaseException as cause:
                error = self._unsupported("apply", cause)
                raise error from cause
            applied = self._capture_native_metadata(
                record.file_handle, phase="apply_verify"
            )
            if self._metadata_digest(applied) != self._metadata_digest(issued.native):
                cause = RuntimeError("applied security metadata did not verify")
                error = self._unsupported("apply_verify", cause)
                raise error from cause

    def verify_staged(self, temp: StagedTemp, snapshot: object) -> None:
        with self._lock:
            record = self._validated_temp(temp)
            issued = self._snapshot_record(
                snapshot, record.binding, record.target
            )
            before = self._adapter.change_token(record.file_handle)
            content = self._content_sha256(record.file_handle, phase="verify")
            native = self._capture_native_metadata(
                record.file_handle, phase="verify"
            )
            after = self._adapter.change_token(record.file_handle)
            digest = self._metadata_digest(native)
            if before != after:
                raise UnsafeFilesystemTarget(
                    "staged file changed during verification"
                )
            if (
                content != record.binding.after_sha256
                or digest != self._metadata_digest(issued.native)
                or digest != record.binding.after_metadata_sha256
                or digest != getattr(
                    snapshot, "canonical_metadata_sha256", None
                )
            ):
                raise UnsafeFilesystemTarget(
                    "staged content or metadata does not match authorization"
                )

    def flush_temp(self, temp: StagedTemp) -> None:
        with self._lock:
            self._adapter.flush_file(
                self._validated_temp(temp).file_handle
            )

    def revalidate_base(
        self, target: SafeTarget, snapshot: object,
    ) -> None:
        with self._lock:
            record = self._validated(target)
            issued = self._snapshot_record(
                snapshot, record.binding, target
            )
            before = self._adapter.change_token(record.pin.file_handle)
            content = self._content_sha256(
                record.pin.file_handle, phase="revalidate"
            )
            native = self._capture_native_metadata(
                record.pin.file_handle, phase="revalidate"
            )
            after = self._adapter.change_token(record.pin.file_handle)
            digest = self._metadata_digest(native)
            if before != after:
                raise UnsafeFilesystemTarget(
                    "base file changed during revalidation"
                )
            if (
                content != record.binding.base_sha256
                or digest != self._metadata_digest(issued.native)
                or digest != record.binding.base_metadata_sha256
                or digest != getattr(
                    snapshot, "canonical_metadata_sha256", None
                )
            ):
                raise UnsafeFilesystemTarget(
                    "authorized base content or metadata changed before replace"
                )

    def discard_temp(self, temp: StagedTemp) -> None:
        with self._lock:
            self._discard_temp_locked(temp)

    def _discard_temp_locked(self, temp: StagedTemp) -> None:
        record = self._temp_record(temp)
        error: BaseException | None = None
        safe_to_dispose = False
        try:
            safe_to_dispose = (
                self._adapter.identity(record.file_handle) == record.identity
            )
            if not safe_to_dispose:
                raise UnsafeFilesystemTarget(
                    "staged temp retained identity changed before discard"
                )
        except BaseException as exc:
            error = exc
        del self._temps[id(temp)]
        temp._mark_closed()  # pyright: ignore[reportPrivateUsage]
        if safe_to_dispose:
            error = self._remember_error(
                error, lambda: self._adapter.ntset_unlink(record.file_handle)
            )
        error = self._remember_error(
            error, lambda: self._adapter.close(record.file_handle)
        )
        error = self._remember_error(
            error, lambda: self._adapter.flush_directory(record.parent_handle)
        )
        error = self._remember_error(
            error, lambda: self._adapter.close(record.parent_handle)
        )
        if error is not None:
            raise error

    def _consume_temp(self, temp: StagedTemp) -> BaseException | None:
        record = self._temp_record(temp)
        del self._temps[id(temp)]
        temp._mark_closed()  # pyright: ignore[reportPrivateUsage]
        error: BaseException | None = None
        error = self._remember_error(
            error, lambda: self._adapter.close(record.file_handle)
        )
        return self._remember_error(
            error, lambda: self._adapter.close(record.parent_handle)
        )


    def _verify_commit_file(
        self, handle: object, *, expected_content: str | None,
        expected_metadata: str | None, label: str,
    ) -> object:
        before = self._adapter.change_token(handle)
        content = self._content_sha256(handle, phase="commit_verify")
        metadata = self._metadata_digest(
            self._capture_native_metadata(handle, phase="commit_verify")
        )
        after = self._adapter.change_token(handle)
        if before != after:
            raise UnsafeFilesystemTarget(
                f"{label} file changed during final validation"
            )
        if content != expected_content or metadata != expected_metadata:
            raise UnsafeFilesystemTarget(
                "final content or metadata changed before replace"
            )
        return after

    def replace(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, source: StagedTemp, destination: SafeTarget,
    ) -> FileIdentity:
        with self._lock:
            source_record = self._validated_temp(source)
            destination_record = self._validated(destination)
            if (
                source_record.binding is not destination_record.binding
                or source_record.target is not destination
            ):
                raise UnsafeFilesystemTarget(
                    "replace operands require the same authorization binding"
                )
            self._commit_snapshot_record(
                destination_record.binding, destination
            )
            staged_token = self._verify_commit_file(
                source_record.file_handle,
                expected_content=source_record.binding.after_sha256,
                expected_metadata=source_record.binding.after_metadata_sha256,
                label="staged",
            )
            base_token = self._verify_commit_file(
                destination_record.pin.file_handle,
                expected_content=destination_record.binding.base_sha256,
                expected_metadata=destination_record.binding.base_metadata_sha256,
                label="base",
            )
            self._adapter.flush_file(source_record.file_handle)
            if self._adapter.change_token(source_record.file_handle) != staged_token:
                raise UnsafeFilesystemTarget("staged file changed before replace")
            if (
                self._adapter.change_token(destination_record.pin.file_handle)
                != base_token
            ):
                raise UnsafeFilesystemTarget("base file changed before replace")
            if (
                self._verify_handle(source_record.file_handle, hardlink=True)
                != source_record.identity
                or self._verify_handle(
                    destination_record.pin.file_handle, hardlink=True
                ) != destination_record.identity
            ):
                raise UnsafeFilesystemTarget("replace operand identity changed")
            expected_path = self._normalize_path(
                self._adapter.final_path(destination_record.pin.file_handle)
            )
            self._adapter.ntset_replace(
                source_record.file_handle,
                destination_record.pin.parent_handle,
                destination_record.basename,
            )
            return self._finish_committed_replace(
                source, destination, source_record, destination_record,
                expected_path,
            )

    def _finish_committed_replace(
        self, source: StagedTemp, destination: SafeTarget,
        source_record: _AtomicIssuedWindowsTemp,
        destination_record: _IssuedWindowsTarget,
        expected_path: str,
    ) -> FileIdentity:
        error: BaseException | None = None
        phase = "successor_verification"
        successor: object | None = None
        successor_identity: FileIdentity | None = None
        try:
            source_identity = self._verify_handle(
                source_record.file_handle, hardlink=True
            )
            if source_identity != source_record.identity:
                raise UnsafeFilesystemTarget(
                    "source identity changed after replace"
                )
            if self._normalize_path(
                self._adapter.final_path(source_record.file_handle)
            ) != expected_path:
                raise UnsafeFilesystemTarget(
                    "source final path does not match destination"
                )
            successor = self._adapter.ntcreate_relative(
                destination_record.pin.parent_handle,
                destination_record.basename, directory=False,
            )
            successor_identity = self._verify_handle(successor, hardlink=True)
            if successor_identity != source_record.identity:
                raise UnsafeFilesystemTarget(
                    "successor identity does not match staged source"
                )
            if self._normalize_path(
                self._adapter.final_path(successor)
            ) != expected_path:
                raise UnsafeFilesystemTarget(
                    "successor final path does not match destination"
                )
        except BaseException as exc:
            error = exc

        try:
            self._adapter.flush_directory(
                destination_record.pin.parent_handle
            )
        except BaseException as exc:
            if error is None:
                error = exc
                phase = "parent_flush"
            else:
                self._add_cleanup_note(error, exc)


        if successor is not None:
            close_error = self._remember_error(
                None, lambda: self._adapter.close(successor)
            )
            if close_error is not None:
                if error is None:
                    error = close_error
                    phase = "operand_cleanup"
                else:
                    self._add_cleanup_note(error, close_error)
        try:
            temp_error = self._consume_temp(source)
        except BaseException as exc:
            temp_error = exc
        if temp_error is not None:
            if error is None:
                error = temp_error
                phase = "operand_cleanup"
            else:
                self._add_cleanup_note(error, temp_error)
        try:
            self._release_target_locked(destination)
        except BaseException as exc:
            if error is None:
                error = exc
                phase = "operand_cleanup"
            else:
                self._add_cleanup_note(error, exc)
        if error is not None:
            outcome = CommittedFilesystemMutationError(
                phase=phase, cause=error
            )
            raise outcome from error
        if successor_identity is None:
            cause = RuntimeError(
                "replacement successor was not validated"
            )
            outcome = CommittedFilesystemMutationError(
                phase="successor_verification", cause=cause
            )
            raise outcome from cause
        return successor_identity

    def release_target(self, target: SafeTarget) -> None:
        with self._lock:
            if target.closed:
                return
            self._release_target_locked(target)

    def _release_target_locked(self, target: SafeTarget) -> None:
        record = self._record(target)
        del self._issued[id(target)]
        for key, issued in tuple(self._metadata.items()):
            if issued.target is target:
                del self._metadata[key]
        target._mark_closed()  # pyright: ignore[reportPrivateUsage]
        error: BaseException | None = None
        error = self._remember_error(
            error, lambda: self._adapter.close(record.pin.file_handle)
        )
        if record.pin.owns_parent:
            error = self._remember_error(
                error, lambda: self._adapter.close(record.pin.parent_handle)
            )
        if error is not None:
            raise error

    def release_temp(self, temp: StagedTemp) -> None:
        with self._lock:
            if temp.closed:
                return
            self._discard_temp_locked(temp)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            error: BaseException | None = None
            for record in tuple(self._temps.values()):
                error = self._remember_error(
                    error, lambda item=record: self.release_temp(item.temp)
                )
            for record in tuple(self._issued.values()):
                error = self._remember_error(
                    error, lambda item=record: self.release_target(item.target)
                )
            error = self._remember_error(
                error, lambda: self._adapter.close(self._root_handle)
            )
            self._closed = True
            if error is not None:
                raise error



__all__ = ["WindowsSafeFilesystem"]
