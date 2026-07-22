from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Literal

import pytest

if TYPE_CHECKING:
    from mochi.security.file_contract import (
        AuthorizationEnvelope,
        ChangeEntry,
        ChangeManifest,
        FileIdentity,
    )

def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _entry(
    *,
    entry_id: str = "0002",
    relative_path: str = "safe/note.txt",
    operation: Literal["add", "update", "delete", "rename"] = "update",
    base_identity: FileIdentity | None = None,
) -> ChangeEntry:
    from mochi.security.file_contract import ChangeEntry, FileIdentity

    if base_identity is None:
        base_identity = FileIdentity("posix", "1", "41", 1, False)
    return ChangeEntry(
        entry_id=entry_id,
        relative_path=relative_path,
        operation=operation,
        base_sha256=_sha("base-content"),
        after_sha256=_sha("after-content"),
        base_identity=base_identity,
        before_blob_id="before-blob",
        after_blob_id="after-blob",
        mode_before=0o644,
        mode_after=0o600,
        base_metadata_sha256=_sha("base-metadata"),
        after_metadata_sha256=_sha("after-metadata"),
        rename_source=None,
        dependency_group="group-a",
    )


def _file_authorization(
    path: str,
    identity: FileIdentity,
    operation: Literal["add", "update", "delete", "rename"] = "update",
) -> AuthorizationEnvelope:
    from mochi.security.file_contract import (
        AuthorizationContext,
        AuthorizationEnvelope,
        FileChangeRequest,
        FileIdentity,
    )

    return AuthorizationEnvelope(
        schema_version=2,
        kind="file_change",
        context=AuthorizationContext(
            requester_id="requester-1",
            session_id="session-1",
            task_id="task-1",
            workspace_root=(
                "C:/workspace"
                if identity.platform == "windows"
                else "workspace"
            ),
            workspace_identity=FileIdentity(
                identity.platform, "10", "20", 1, False
            ),
        ),
        policy_version="policy-v1",
        file_request=FileChangeRequest(
            entries=(
                _entry(
                    entry_id="0001",
                    relative_path=path,
                    operation=operation,
                    base_identity=identity,
                ),
            ),
            patch_sha256=_sha("patch"),
        ),
        exec_request=None,
    )


def _authorization() -> AuthorizationEnvelope:
    from mochi.security.file_contract import FileIdentity

    return _file_authorization(
        "safe/note.txt",
        FileIdentity("posix", "1", "41", 1, False),
    )


def _posix_atomic_authorization() -> AuthorizationEnvelope:
    authorization = _authorization()
    request = authorization.file_request
    assert request is not None
    metadata = {
        "gid": 1000,
        "mode": 0o600,
        "uid": 1000,
        "xattrs": [],
    }
    metadata_sha = hashlib.sha256(
        json.dumps(
            metadata, sort_keys=True, separators=(",", ":")
        ).encode("ascii")
    ).hexdigest()
    entry = replace(
        request.entries[0],
        base_metadata_sha256=metadata_sha,
        after_metadata_sha256=metadata_sha,
    )
    return replace(
        authorization,
        file_request=replace(request, entries=(entry,)),
    )


def _stage_posix_atomic_temp(filesystem, destination):
    snapshot = filesystem.capture_metadata(destination)
    staged = filesystem.create_temp(destination)
    payload = b"after-content"
    assert filesystem.write_temp(
        staged, memoryview(payload)
    ) == len(payload)
    filesystem.apply_metadata_snapshot(staged, snapshot)
    return staged

def _manifest(
    *, entries: tuple[ChangeEntry, ...] | None = None
) -> ChangeManifest:
    from mochi.security.file_contract import ChangeManifest, FileIdentity

    return ChangeManifest(
        version=1,
        change_set_id="change-set",
        workspace_root="workspace",
        workspace_identity=FileIdentity("posix", "10", "20", 1, False),
        tool_name="write_file",
        intent="mutate",
        entries=(_entry(),) if entries is None else entries,
        patch_sha256=_sha("manifest-patch"),
        policy_version="policy-v1",
        created_at="created",
        expires_at="expires",
        request_digest=_sha("manifest-request"),
        ui_metadata={"label": "preview"},
    )


class _FakePosixAdapter:
    O_RDONLY = 0
    O_WRONLY = 1
    O_RDWR = 2
    O_CREAT = 0x40
    O_EXCL = 0x80
    O_DIRECTORY = 0x10000
    O_NOFOLLOW = 0x20000
    platform = "linux"
    supports_fd = frozenset(
        {"listxattr", "getxattr", "removexattr", "setxattr"}
    )
    supports_dir_fd = frozenset({"open", "stat", "unlink", "replace"})
    supports_follow_symlinks = frozenset({"stat"})

    def __init__(self, *, link_count: int = 1) -> None:
        self.link_count = link_count
        self.next_fd = 100
        self.fd_nodes: dict[int, str] = {}
        self.children: dict[tuple[str, str], str] = {
            ("workspace", "safe"): "safe-original",
            ("safe-original", "note.txt"): "note-original",
            ("outside", "note.txt"): "outside-note",
        }
        self.content: dict[str, bytearray] = {
            "note-original": bytearray(b"base-content"),
            "outside-note": bytearray(b"outside"),
        }
        self.xattrs: dict[str, dict[bytes, bytes]] = {
            "note-original": {},
            "outside-note": {},
        }
        self.unlinked: list[tuple[str, str]] = []
        self.open_calls: list[tuple[str, int | None]] = []
        self.replace_calls: list[tuple[str, str, str, str]] = []
        self.close_calls: list[int] = []
        self.dup_calls: list[tuple[int, int]] = []
        self.fstat_calls: list[int] = []
        self.stat_calls: list[tuple[str, int, bool]] = []
        self.fsync_calls: list[int] = []
        self.change_clock = 1
        self.mtime_ns: dict[str, int] = {
            node: self.change_clock for node in self.content
        }
        self.ctime_ns: dict[str, int] = {
            node: self.change_clock for node in self.content
        }

    def open(
        self,
        path: str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        self.open_calls.append((path, dir_fd))
        if dir_fd is None:
            node = "workspace"
        else:
            parent = self.fd_nodes[dir_fd]
            key = (parent, path)
            if key not in self.children and flags & self.O_CREAT:
                node = f"temp:{path}"
                self.children[key] = node
                self.content[node] = bytearray()
                self.xattrs[node] = {}
                self.mtime_ns[node] = self.change_clock
                self.ctime_ns[node] = self.change_clock
            else:
                node = self.children[key]
        fd = self.next_fd
        self.next_fd += 1
        self.fd_nodes[fd] = node
        return fd

    def dup(self, fd: int) -> int:
        duplicate = self.next_fd
        self.next_fd += 1
        self.fd_nodes[duplicate] = self.fd_nodes[fd]
        self.dup_calls.append((fd, duplicate))
        return duplicate

    def close(self, fd: int) -> None:
        self.close_calls.append(fd)
        self.fd_nodes.pop(fd, None)

    def _touch_content(self, node: str) -> None:
        self.change_clock += 1
        self.mtime_ns[node] = self.change_clock
        self.ctime_ns[node] = self.change_clock

    def _touch_metadata(self, node: str) -> None:
        self.change_clock += 1
        self.ctime_ns[node] = self.change_clock

    def _stat(self, node: str) -> SimpleNamespace:
        if node == "workspace":
            mode, dev, ino = 0o040700, 10, 20
        elif node in {"safe-original", "outside"}:
            mode, dev, ino = 0o040700, 10, 21
        elif node == "note-original":
            mode, dev, ino = 0o100600, 1, 41
        elif node == "other-original":
            mode, dev, ino = 0o100600, 1, 42
        elif node == "outside-note":
            mode, dev, ino = 0o100600, 1, 99
        else:
            mode, dev, ino = 0o100600, 1, 77
        return SimpleNamespace(
            st_mode=mode,
            st_dev=dev,
            st_ino=ino,
            st_nlink=(
                self.link_count if mode & 0o100000 else 1
            ),
            st_uid=1000,
            st_gid=1000,
            st_size=len(self.content.get(node, b"")),
            st_mtime_ns=self.mtime_ns.get(node, 1),
            st_ctime_ns=self.ctime_ns.get(node, 1),
        )

    def fstat(self, fd: int) -> SimpleNamespace:
        self.fstat_calls.append(fd)
        return self._stat(self.fd_nodes[fd])

    def stat(
        self,
        path: str,
        *,
        dir_fd: int,
        follow_symlinks: bool,
    ) -> SimpleNamespace:
        assert follow_symlinks is False
        self.stat_calls.append((path, dir_fd, follow_symlinks))
        node = self.children[(self.fd_nodes[dir_fd], path)]
        return self._stat(node)

    def write(self, fd: int, data: memoryview | bytes) -> int:
        node = self.fd_nodes[fd]
        self.content.setdefault(node, bytearray()).extend(bytes(data))
        self._touch_content(node)
        return len(data)

    def pread(self, fd: int, size: int, offset: int) -> bytes:
        node = self.fd_nodes[fd]
        return bytes(self.content.get(node, b"")[offset : offset + size])

    def listxattr(self, fd: int) -> list[bytes]:
        return list(self.xattrs.get(self.fd_nodes[fd], {}))

    def getxattr(self, fd: int, name: bytes) -> bytes:
        return self.xattrs[self.fd_nodes[fd]][name]

    def fchown(self, fd: int, uid: int, gid: int) -> None:
        del uid, gid
        self._touch_metadata(self.fd_nodes[fd])

    def fchmod(self, fd: int, mode: int) -> None:
        del mode
        self._touch_metadata(self.fd_nodes[fd])

    def removexattr(self, fd: int, name: bytes) -> None:
        node = self.fd_nodes[fd]
        del self.xattrs[node][name]
        self._touch_metadata(node)

    def setxattr(self, fd: int, name: bytes, value: bytes) -> None:
        node = self.fd_nodes[fd]
        self.xattrs.setdefault(node, {})[name] = value
        self._touch_metadata(node)

    def fsync(self, fd: int) -> None:
        self.fsync_calls.append(fd)

    def unlink(self, path: str, *, dir_fd: int) -> None:
        parent = self.fd_nodes[dir_fd]
        self.unlinked.append((parent, path))
        self.children.pop((parent, path), None)

    def replace(
        self,
        src: str,
        dst: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
    ) -> None:
        source_parent = self.fd_nodes[src_dir_fd]
        destination_parent = self.fd_nodes[dst_dir_fd]
        self.replace_calls.append(
            (source_parent, src, destination_parent, dst)
        )
        node = self.children.pop((source_parent, src))
        self.children[(destination_parent, dst)] = node


def _windows_authorization(
    operation: Literal["add", "update", "delete", "rename"] = "update",
) -> AuthorizationEnvelope:
    from mochi.security.file_contract import FileIdentity

    return _file_authorization(
        "safe/note.txt",
        FileIdentity("windows", "10", "41", 1, False),
        operation,
    )


def _windows_atomic_authorization() -> AuthorizationEnvelope:
    authorization = _windows_authorization()
    request = authorization.file_request
    assert request is not None
    metadata = {
        "owner": None,
        "group": None,
        "dacl": None,
        "dacl_present": False,
        "dacl_protected": False,
        "sacl": None,
        "sacl_present": False,
        "sacl_protected": False,
        "sacl_state": "inaccessible",
    }
    metadata_sha256 = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode(
            "ascii"
        )
    ).hexdigest()
    content_sha256 = hashlib.sha256(b"").hexdigest()
    entry = replace(
        request.entries[0],
        base_sha256=content_sha256,
        after_sha256=content_sha256,
        base_metadata_sha256=metadata_sha256,
        after_metadata_sha256=metadata_sha256,
    )
    return replace(
        authorization, file_request=replace(request, entries=(entry,))
    )

class _FakeWindowsAdapter:
    semantics = frozenset({
        "content_read_at", "content_write", "file_flush", "directory_flush",
        "change_token", "security_capture", "security_apply",
        "relative_rename", "handle_disposition", "duplicate_handle",
    })
    platform = "win32-fake"
    sacl_state = "inaccessible"

    def __init__(
        self,
        *,
        available: bool = True,
        link_count: int = 1,
        parent_reparse: bool = False,
        root_file_id: str = "20",
        workspace_final_path: str = "C:/workspace",
        temp_reparse: bool = False,
        collisions: int = 0,
    ) -> None:
        from mochi.security.file_contract import FileIdentity

        self.available = available
        self.calls: list[tuple[object, ...]] = []
        self.nodes: dict[object, str] = {}
        self.relative_results: list[tuple[object, str, object]] = []
        self.temp_handles: list[object] = []
        self.successor_handles: set[object] = set()
        self.workspace_final_path = workspace_final_path
        self.renamed = False
        self.source_path_after_rename: str | None = None
        self.successor_path_after_rename: str | None = None
        self.source_identity_after_rename = None
        self.successor_identity_after_rename = None
        self.temp_reparse = temp_reparse
        self.collisions = collisions
        self.children = {
            ("workspace", "safe"): "safe-original",
            ("safe-original", "note.txt"): "note-original",
            ("safe-original", "other.txt"): "other-original",
            ("outside", "note.txt"): "outside-note",
        }
        self.identities = {
            "workspace": FileIdentity(
                "windows", "10", root_file_id, 1, False
            ),
            "safe-original": FileIdentity(
                "windows", "10", "21", 1, parent_reparse
            ),
            "outside": FileIdentity(
                "windows", "99", "21", 1, False
            ),
            "note-original": FileIdentity(
                "windows", "10", "41", link_count, False
            ),
            "other-original": FileIdentity(
                "windows", "10", "42", 1, False
            ),
            "outside-note": FileIdentity(
                "windows", "99", "41", 1, False
            ),
        }
        self.paths = {
            "workspace": workspace_final_path,
            "safe-original": f"{workspace_final_path}/safe",
            "outside": "C:/outside",
            "note-original": (
                f"{workspace_final_path}/safe/note.txt"
            ),
            "other-original": (
                f"{workspace_final_path}/safe/other.txt"
            ),
            "outside-note": "C:/outside/note.txt",
        }

    def createfile_workspace(self, path: str) -> object:
        self.calls.append(
            (
                "CreateFileW",
                path,
                "OPEN_REPARSE_POINT|BACKUP_SEMANTICS",
            )
        )
        handle = object()
        self.nodes[handle] = "workspace"
        return handle

    def ntcreate_relative(
        self, root: object, basename: str, *, directory: bool
    ) -> object:
        self.calls.append(
            (
                "NtCreateFile",
                root,
                basename,
                directory,
                "OPEN_REPARSE_POINT",
            )
        )
        handle = object()
        self.nodes[handle] = self.children[
            (self.nodes[root], basename)
        ]
        self.relative_results.append((root, basename, handle))
        if self.renamed and not directory:
            self.successor_handles.add(handle)
        return handle

    def ntcreate_new_relative(
        self, root: object, basename: str
    ) -> object:
        from mochi.security.file_contract import FileIdentity

        self.calls.append(
            (
                "NtCreateFile",
                root,
                basename,
                False,
                "CREATE|OPEN_REPARSE_POINT|FILE_WRITE_DATA",
            )
        )
        if self.collisions:
            self.collisions -= 1
            raise FileExistsError(basename)
        node = f"temp:{basename}"
        self.children[(self.nodes[root], basename)] = node
        self.identities[node] = FileIdentity(
            "windows", "10", "temp", 1, self.temp_reparse
        )
        self.paths[node] = (
            f"{self.paths[self.nodes[root]]}/{basename}"
        )
        handle = object()
        self.nodes[handle] = node
        self.temp_handles.append(handle)
        return handle

    def final_path(self, handle: object) -> str:
        if self.renamed and handle in self.successor_handles:
            path = self.successor_path_after_rename
        elif self.renamed and handle in self.temp_handles:
            path = self.source_path_after_rename
        else:
            path = None
        path = path or self.paths[self.nodes[handle]]
        self.calls.append(
            ("GetFinalPathNameByHandleW", handle, path)
        )
        return path

    def identity(self, handle: object):
        self.calls.append(
            ("GetFileInformationByHandle", handle)
        )
        if self.renamed and handle in self.successor_handles:
            return (
                self.successor_identity_after_rename
                or self.identities[self.nodes[handle]]
            )
        if self.renamed and handle in self.temp_handles:
            return (
                self.source_identity_after_rename
                or self.identities[self.nodes[handle]]
            )
        return self.identities[self.nodes[handle]]

    def ntset_unlink(self, handle: object) -> None:
        self.calls.append(
            (
                "NtSetInformationFile",
                handle,
                "FileDispositionInformation",
            )
        )

    def ntset_replace(
        self, handle: object, root: object, basename: str
    ) -> None:
        self.calls.append(
            (
                "NtSetInformationFile",
                handle,
                "FileRenameInformation",
                root,
                basename,
            )
        )
        node = self.nodes[handle]
        parent = self.nodes[root]
        self.children[(parent, basename)] = node
        self.paths[node] = f"{self.paths[parent]}/{basename}"
        self.renamed = True

    def duplicate_handle(self, handle: object) -> object:
        self.calls.append(("DuplicateHandle", handle))
        duplicate = object()
        self.nodes[duplicate] = self.nodes[handle]
        return duplicate

    def sacl_access(self, handle: object) -> str:
        del handle
        return self.sacl_state

    def read_at(self, handle: object, size: int, offset: int) -> bytes:
        del handle, size, offset
        return b""

    def write(self, handle: object, data: memoryview) -> int:
        self.calls.append(("WriteFile", handle, len(data)))
        return len(data)

    def flush_file(self, handle: object) -> None:
        self.calls.append(("FlushFileBuffers", handle))

    def flush_directory(self, handle: object) -> None:
        self.calls.append(("NtFlushBuffersFile", handle))

    def change_token(self, handle: object) -> object:
        return self.identity(handle)

    def security_descriptor(
        self, handle: object, *, include_sacl: bool
    ) -> object:
        del handle, include_sacl
        return SimpleNamespace(
            raw_descriptor=b"fake",
            owner=None,
            group=None,
            dacl=None,
            dacl_present=False,
            dacl_protected=False,
            sacl=None,
            sacl_present=False,
            sacl_protected=False,
            sacl_state=self.sacl_state,
        )

    def apply_security_descriptor(
        self, handle: object, metadata: object
    ) -> None:
        self.calls.append(("SetKernelObjectSecurity", handle, metadata))
    def close(self, handle: object) -> None:
        self.calls.append(("CloseHandle", handle))
        self.nodes.pop(handle, None)



def _exec_authorization() -> AuthorizationEnvelope:
    from mochi.security.file_contract import (
        AuthorizationContext,
        AuthorizationEnvelope,
        EnvVarHash,
        ExecRequest,
        FileIdentity,
        ResourceLimits,
    )

    return AuthorizationEnvelope(
        schema_version=2,
        kind="exec",
        context=AuthorizationContext(
            requester_id="requester-1",
            session_id="session-1",
            task_id="task-1",
            workspace_root="C:/workspace",
            workspace_identity=FileIdentity(
                "windows", "10", "20", 1, False
            ),
        ),
        policy_version="policy-v1",
        file_request=None,
        exec_request=ExecRequest(
            command_utf8_sha256=_sha("command"),
            shell="powershell",
            executable="pwsh.exe",
            argv=("-NoProfile", "-Command", "Get-Date"),
            resolved_cwd="C:/workspace",
            env=(EnvVarHash("PATH", _sha("path-value")),),
            network_policy="deny",
            resource_limits=ResourceLimits(30, 1024, 4096),
            requested_escalation="none",
            sandbox_backend="windows-native",
            sandbox_capability_plan_digest=_sha("sandbox-plan"),
        ),
    )



def _native_authorization_for(root: Path):
    from mochi.security.file_contract import AuthorizationContext
    from mochi.security.safe_fs_windows import _WindowsNativeAdapter

    adapter = _WindowsNativeAdapter()
    if not adapter.available:
        pytest.skip("Windows native APIs unavailable")
    handle = adapter.createfile_workspace(str(root))
    try:
        identity = adapter.identity(handle)
    finally:
        adapter.close(handle)
    return replace(
        _windows_authorization(),
        context=AuthorizationContext(
            requester_id="requester-1",
            session_id="session-1",
            task_id="task-1",
            workspace_root=str(root),
            workspace_identity=identity,
        ),
    )



def _two_file_authorization(platform: Literal["posix", "windows"]):
    from mochi.security.file_contract import (
        FileChangeRequest,
        FileIdentity,
    )

    first_identity = FileIdentity(platform, "1" if platform == "posix" else "10", "41", 1, False)
    second_identity = FileIdentity(platform, "1" if platform == "posix" else "10", "42", 1, False)
    authorization = _file_authorization(
        "safe/note.txt", first_identity
    )
    return replace(
        authorization,
        file_request=FileChangeRequest(
            entries=(
                authorization.file_request.entries[0],
                _entry(
                    entry_id="0002",
                    relative_path="safe/other.txt",
                    base_identity=second_identity,
                ),
            ),
            patch_sha256=_sha("patch"),
        ),
    )
