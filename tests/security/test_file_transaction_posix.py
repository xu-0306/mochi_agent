from __future__ import annotations

import errno
import hashlib
import json
import os
import stat
import struct
import sys
import threading
from types import SimpleNamespace

import pytest

BASE = b"base-content"
AFTER = b"after-content"
ACL = b"system.posix_acl_access"
XATTRS = {
    b"user.zeta": b"z-value",
    ACL: b"acl-value",
    b"user.alpha": b"a-value",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _metadata_sha(
    *,
    uid: int = 1000,
    gid: int = 1001,
    mode: int = 0o640,
    xattrs: dict[bytes, bytes] | None = None,
) -> str:
    attrs = XATTRS if xattrs is None else xattrs
    payload = {
        "gid": gid,
        "mode": mode,
        "uid": uid,
        "xattrs": [
            {"name_hex": name.hex(), "value_hex": attrs[name].hex()}
            for name in sorted(attrs)
        ],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("ascii")).hexdigest()


def _authorization():
    from mochi.security.file_contract import (
        AuthorizationContext,
        AuthorizationEnvelope,
        ChangeEntry,
        FileChangeRequest,
        FileIdentity,
    )

    identity = FileIdentity("posix", "1", "41", 1, False)
    metadata = _metadata_sha()
    entry = ChangeEntry(
        entry_id="posix-existing",
        relative_path="safe/note.txt",
        operation="update",
        base_sha256=_sha(BASE),
        after_sha256=_sha(AFTER),
        base_identity=identity,
        before_blob_id="before",
        after_blob_id="after",
        mode_before=0o640,
        mode_after=0o640,
        base_metadata_sha256=metadata,
        after_metadata_sha256=metadata,
        rename_source=None,
        dependency_group="posix",
    )
    return AuthorizationEnvelope(
        schema_version=1,
        kind="file_change",
        context=AuthorizationContext(
            requester_id="requester",
            session_id="session",
            task_id="task",
            workspace_root="workspace",
            workspace_identity=FileIdentity(
                "posix", "10", "20", 1, False
            ),
        ),
        policy_version="policy-v1",
        file_request=FileChangeRequest(
            entries=(entry,), patch_sha256=_sha(b"patch")
        ),
        exec_request=None,
    )


class _PosixAdapter:
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

    def __init__(self) -> None:
        self.next_fd = 100
        self.next_ino = 77
        self.fd_nodes: dict[int, str] = {}
        self.offsets: dict[int, int] = {}
        self.children = {
            ("workspace", "safe"): "safe",
            ("safe", "note.txt"): "original",
        }
        self.info = {
            "workspace": dict(
                mode=0o040700,
                dev=10,
                ino=20,
                nlink=1,
                uid=1000,
                gid=1000,
                size=0,
                mtime_ns=1,
                ctime_ns=1,
            ),
            "safe": dict(
                mode=0o040700,
                dev=10,
                ino=21,
                nlink=1,
                uid=1000,
                gid=1000,
                size=0,
                mtime_ns=1,
                ctime_ns=1,
            ),
            "original": dict(
                mode=0o100640,
                dev=1,
                ino=41,
                nlink=1,
                uid=1000,
                gid=1001,
                size=len(BASE),
                mtime_ns=1,
                ctime_ns=1,
            ),
        }
        self.content = {"original": bytearray(BASE)}
        self.xattrs = {"original": dict(XATTRS)}
        self.events: list[tuple[object, ...]] = []
        self.open_calls: list[tuple[str, int, int, int | None]] = []
        self.close_calls: list[int] = []
        self.getxattr_calls: list[tuple[int, bytes]] = []
        self.write_plan: list[int | BaseException] = []
        self.failures: dict[str, BaseException] = {}
        self.target_fstat_ino: int | None = None
        self.successor_stat_ino: int | None = None
        self.tamper_temp_content_read = False
        self.tamper_temp_metadata_on_second_list = False
        self.temp_listxattr_calls = 0
        self.after_replace = None
        self.on_base_pread = None
        self.on_temp_fsync = None
        self.missing_change_token_field: str | None = None

    def fail(self, operation: str, error: BaseException) -> None:
        self.failures[operation] = error

    def _raise(self, operation: str) -> None:
        error = self.failures.pop(operation, None)
        if error is not None:
            raise error

    def _stat(self, node: str) -> SimpleNamespace:
        values = dict(self.info[node])
        if node == "original" and self.target_fstat_ino is not None:
            values["ino"] = self.target_fstat_ino
        result = {
            "st_mode": values["mode"],
            "st_dev": values["dev"],
            "st_ino": values["ino"],
            "st_nlink": values["nlink"],
            "st_uid": values["uid"],
            "st_gid": values["gid"],
            "st_size": values["size"],
            "st_mtime_ns": values["mtime_ns"],
            "st_ctime_ns": values["ctime_ns"],
        }
        if self.missing_change_token_field is not None:
            result.pop(self.missing_change_token_field, None)
        return SimpleNamespace(**result)

    def _touch_content(self, node: str) -> None:
        self.info[node]["size"] = len(self.content[node])
        self.info[node]["mtime_ns"] += 1
        self.info[node]["ctime_ns"] += 1

    def _touch_metadata(self, node: str) -> None:
        self.info[node]["ctime_ns"] += 1

    def _add_file(self, node: str, content: bytes) -> None:
        self.info[node] = dict(
            mode=0o100640,
            dev=1,
            ino=self.next_ino,
            nlink=1,
            uid=1000,
            gid=1001,
            size=len(content),
            mtime_ns=1,
            ctime_ns=1,
        )
        self.next_ino += 1
        self.content[node] = bytearray(content)
        self.xattrs[node] = dict(XATTRS)

    def open(
        self,
        path: str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        self.open_calls.append((path, flags, mode, dir_fd))
        if dir_fd is None:
            node = "workspace"
        else:
            parent = self.fd_nodes[dir_fd]
            key = (parent, path)
            if key not in self.children:
                if not flags & self.O_CREAT:
                    raise FileNotFoundError(path)
                node = f"temp:{path}"
                self.children[key] = node
                self.info[node] = dict(
                    mode=0o100000 | mode,
                    dev=1,
                    ino=self.next_ino,
                    nlink=1,
                    uid=2000,
                    gid=2000,
                    size=0,
                    mtime_ns=1,
                    ctime_ns=1,
                )
                self.next_ino += 1
                self.content[node] = bytearray()
                self.xattrs[node] = {
                    b"user.alpha": b"stale",
                    b"user.temp-only": b"remove",
                }
            elif flags & self.O_EXCL:
                raise FileExistsError(path)
            else:
                node = self.children[key]
        fd = self.next_fd
        self.next_fd += 1
        self.fd_nodes[fd] = node
        self.offsets[fd] = 0
        self.events.append(("open", node, flags, fd, dir_fd))
        return fd

    def dup(self, fd: int) -> int:
        duplicate = self.next_fd
        self.next_fd += 1
        self.fd_nodes[duplicate] = self.fd_nodes[fd]
        self.offsets[duplicate] = self.offsets.get(fd, 0)
        return duplicate

    def close(self, fd: int) -> None:
        node = self.fd_nodes.pop(fd)
        self.offsets.pop(fd, None)
        self.close_calls.append(fd)
        self.events.append(("close", node, fd))
        self._raise(f"close:{node}")

    def fstat(self, fd: int) -> SimpleNamespace:
        return self._stat(self.fd_nodes[fd])

    def stat(
        self, path: str, *, dir_fd: int, follow_symlinks: bool
    ) -> SimpleNamespace:
        assert follow_symlinks is False
        node = self.children[(self.fd_nodes[dir_fd], path)]
        saved = self.target_fstat_ino
        self.target_fstat_ino = None
        try:
            result = self._stat(node)
            if (
                self.successor_stat_ino is not None
                and path == "note.txt"
                and node.startswith("temp:")
            ):
                result.st_ino = self.successor_stat_ino
            return result
        finally:
            self.target_fstat_ino = saved

    def write(self, fd: int, data: memoryview | bytes) -> int:
        node = self.fd_nodes[fd]
        action: int | BaseException = (
            self.write_plan.pop(0) if self.write_plan else len(data)
        )
        if isinstance(action, BaseException):
            raise action
        written = min(action, len(data))
        offset = self.offsets[fd]
        payload = bytes(data[:written])
        self.content[node][offset : offset + written] = payload
        self._touch_content(node)
        self.offsets[fd] += written
        self.events.append(("write", node, payload, offset))
        return written

    def pread(self, fd: int, size: int, offset: int) -> bytes:
        node = self.fd_nodes[fd]
        self.events.append(("pread", node, offset))
        self._raise(f"pread:{node}")
        if node == "original" and self.on_base_pread is not None:
            callback = self.on_base_pread
            self.on_base_pread = None
            callback()
        if self.tamper_temp_content_read and node.startswith("temp:"):
            self.tamper_temp_content_read = False
            self.content[node][:] = b"tampered-content"
            self._touch_content(node)
        return bytes(self.content[node][offset : offset + size])

    def listxattr(self, fd: int) -> list[bytes]:
        node = self.fd_nodes[fd]
        self.events.append(("listxattr", node))
        failure_node = "temp" if node.startswith("temp:") else node
        self._raise(f"listxattr:{failure_node}")
        if node.startswith("temp:"):
            self.temp_listxattr_calls += 1
            if (
                self.tamper_temp_metadata_on_second_list
                and self.temp_listxattr_calls >= 2
            ):
                self.tamper_temp_metadata_on_second_list = False
                self.xattrs[node][b"user.alpha"] = b"tampered"
                self._touch_metadata(node)
        return list(self.xattrs[node])

    def getxattr(self, fd: int, name: bytes) -> bytes:
        node = self.fd_nodes[fd]
        self.getxattr_calls.append((fd, name))
        self.events.append(("getxattr", node, name))
        self._raise(f"getxattr:{node}")
        return self.xattrs[node][name]

    def fchown(self, fd: int, uid: int, gid: int) -> None:
        node = self.fd_nodes[fd]
        self.events.append(("fchown", node, uid, gid))
        self._raise("fchown")
        self.info[node]["uid"] = uid
        self.info[node]["gid"] = gid
        self._touch_metadata(node)

    def fchmod(self, fd: int, mode: int) -> None:
        node = self.fd_nodes[fd]
        self.events.append(("fchmod", node, mode))
        self._raise("fchmod")
        self.info[node]["mode"] = 0o100000 | mode
        self._touch_metadata(node)

    def removexattr(self, fd: int, name: bytes) -> None:
        node = self.fd_nodes[fd]
        self.events.append(("removexattr", node, name))
        self._raise("removexattr")
        del self.xattrs[node][name]
        self._touch_metadata(node)

    def setxattr(self, fd: int, name: bytes, value: bytes) -> None:
        node = self.fd_nodes[fd]
        self.events.append(("setxattr", node, name))
        self._raise("setxattr")
        self.xattrs[node][name] = value
        self._touch_metadata(node)

    def fsync(self, fd: int) -> None:
        node = self.fd_nodes[fd]
        kind = "temp" if node.startswith("temp:") else "parent"
        self.events.append(("fsync", node))
        self._raise(f"fsync:{kind}")
        if node.startswith("temp:") and self.on_temp_fsync is not None:
            callback = self.on_temp_fsync
            self.on_temp_fsync = None
            callback()

    def unlink(self, path: str, *, dir_fd: int) -> None:
        parent = self.fd_nodes[dir_fd]
        node = self.children[(parent, path)]
        self.events.append(("unlink", parent, path, node))
        self._raise("unlink")
        del self.children[(parent, path)]

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
        self.events.append(("replace", source_parent, src, dst))
        self._raise("replace")
        node = self.children.pop((source_parent, src))
        self.children[(destination_parent, dst)] = node
        if self.after_replace is not None:
            self.after_replace()


def _filesystem(adapter: _PosixAdapter):
    from mochi.security.safe_fs_posix import PosixSafeFilesystem

    return PosixSafeFilesystem("/workspace", adapter=adapter)


def _prepared(adapter: _PosixAdapter):
    filesystem = _filesystem(adapter)
    target = filesystem.prepare_target("safe/note.txt", _authorization())
    return filesystem, target


def test_prepare_retains_nofollow_file_fd_and_releases_it_once() -> None:
    adapter = _PosixAdapter()
    filesystem, target = _prepared(adapter)

    target_open = next(call for call in adapter.open_calls if call[0] == "note.txt")
    assert target_open[1] == adapter.O_RDONLY | adapter.O_NOFOLLOW
    target_fd = next(
        fd for fd, node in adapter.fd_nodes.items() if node == "original"
    )
    target.close()
    target.close()

    assert adapter.close_calls.count(target_fd) == 1
    assert target.closed
    filesystem.close()


def test_prepare_identity_mismatch_closes_file_and_parent_fds() -> None:
    from mochi.security.safe_filesystem import UnsafeFilesystemTarget

    adapter = _PosixAdapter()
    adapter.target_fstat_ino = 42
    filesystem = _filesystem(adapter)

    with pytest.raises(UnsafeFilesystemTarget, match="identity"):
        filesystem.prepare_target("safe/note.txt", _authorization())

    assert adapter.fd_nodes == {100: "workspace"}
    assert any(event[:2] == ("close", "original") for event in adapter.events)
    assert any(event[:2] == ("close", "safe") for event in adapter.events)
    filesystem.close()


def test_capture_metadata_is_linux_fd_only_sorted_deterministic_and_opaque() -> None:
    from mochi.tools.file_transaction import FileMetadataSnapshot

    adapter = _PosixAdapter()
    filesystem, target = _prepared(adapter)

    first = filesystem.capture_metadata(target)
    second = filesystem.capture_metadata(target)

    assert type(first) is FileMetadataSnapshot
    assert first.canonical_metadata_sha256 == _metadata_sha()
    assert second.canonical_metadata_sha256 == first.canonical_metadata_sha256
    assert first.identity == target.identity
    assert first.binding is filesystem.transaction_binding(target)
    assert not any(
        hasattr(first, name)
        for name in ("path", "relative_path", "fd", "file_fd", "xattrs")
    )
    target_fd = next(
        fd for fd, node in adapter.fd_nodes.items() if node == "original"
    )
    assert adapter.getxattr_calls[:3] == [
        (target_fd, ACL),
        (target_fd, b"user.alpha"),
        (target_fd, b"user.zeta"),
    ]
    target.close()
    filesystem.close()


@pytest.mark.parametrize("failure", ["platform", "list", "get"])
def test_capture_unsupported_metadata_fails_before_staging(failure: str) -> None:
    from mochi.security.safe_filesystem import UnsupportedSecurityMetadata

    adapter = _PosixAdapter()
    if failure == "platform":
        adapter.platform = "darwin"
    filesystem, target = _prepared(adapter)
    if failure == "list":
        adapter.fail("listxattr:original", OSError("list failed"))
    elif failure == "get":
        adapter.fail("getxattr:original", OSError("get failed"))

    with pytest.raises(UnsupportedSecurityMetadata, match="metadata"):
        filesystem.capture_metadata(target)

    assert not any(node.startswith("temp:") for node in adapter.content)
    target.close()
    filesystem.close()


def test_atomic_write_exercises_posix_owner_primitives_and_durable_replace() -> None:
    from mochi.security.file_contract import FileIdentity
    from mochi.tools.file_transaction import atomic_write_bytes

    adapter = _PosixAdapter()
    filesystem, target = _prepared(adapter)
    snapshot = filesystem.capture_metadata(target)
    target_parent_fd = next(
        call[3] for call in adapter.open_calls if call[0] == "note.txt"
    )
    adapter.write_plan = [InterruptedError(), 3]

    result = atomic_write_bytes(target, AFTER, snapshot)

    temp_open = next(
        call for call in adapter.open_calls if call[0].startswith(".mochi-")
    )
    expected_flags = (
        adapter.O_RDWR
        | adapter.O_CREAT
        | adapter.O_EXCL
        | adapter.O_NOFOLLOW
    )
    assert temp_open[1] == expected_flags
    assert temp_open[3] != target_parent_fd
    successor = adapter.children[("safe", "note.txt")]
    assert isinstance(result.successor_identity, FileIdentity)
    assert result.successor_identity.file_id == str(adapter.info[successor]["ino"])
    assert result.bytes_written == len(AFTER)
    assert bytes(adapter.content[successor]) == AFTER
    assert adapter.info[successor]["uid"] == 1000
    assert adapter.info[successor]["gid"] == 1001
    assert stat.S_IMODE(adapter.info[successor]["mode"]) == 0o640
    assert adapter.xattrs[successor] == XATTRS
    metadata_events = [
        (event[0], event[2] if len(event) > 2 else None)
        for event in adapter.events
        if event[0] in {"fchown", "fchmod", "removexattr", "setxattr"}
    ]
    assert metadata_events == [
        ("fchown", 1000),
        ("fchmod", 0o640),
        ("removexattr", b"user.temp-only"),
        ("setxattr", b"user.alpha"),
        ("setxattr", b"user.zeta"),
        ("setxattr", ACL),
    ]
    event_names = [event[0] for event in adapter.events]
    assert event_names.index("fsync") < event_names.index("replace")
    assert ("fsync", "safe") in adapter.events
    assert target.closed
    assert adapter.fd_nodes == {100: "workspace"}
    filesystem.close()


def test_postreplace_parent_fsync_failure_reports_committed_without_rollback() -> None:
    from mochi.security.safe_filesystem import CommittedFilesystemMutationError
    from mochi.tools.file_transaction import atomic_write_bytes

    adapter = _PosixAdapter()
    filesystem, target = _prepared(adapter)
    snapshot = filesystem.capture_metadata(target)
    adapter.fail("fsync:parent", OSError("parent fsync failed"))

    with pytest.raises(CommittedFilesystemMutationError) as raised:
        atomic_write_bytes(target, AFTER, snapshot)

    assert raised.value.committed is True
    assert isinstance(raised.value.cause, OSError)
    successor = adapter.children[("safe", "note.txt")]
    assert successor != "original"
    assert bytes(adapter.content[successor]) == AFTER
    assert sum(event[0] == "replace" for event in adapter.events) == 1
    assert not any(event[0] == "unlink" for event in adapter.events)
    filesystem.close()

@pytest.mark.parametrize("missing", ["O_RDWR", "pread"])
def test_incomplete_posix_adapter_is_rejected_at_construction(
    missing: str,
) -> None:
    class _IncompleteAdapter(_PosixAdapter):
        pass

    setattr(_IncompleteAdapter, missing, None)
    adapter = _IncompleteAdapter()

    with pytest.raises(RuntimeError, match=missing):
        _filesystem(adapter)

    assert adapter.fd_nodes == {}


def test_direct_temp_close_unlinks_and_fsyncs_exact_temp() -> None:
    adapter = _PosixAdapter()
    filesystem, target = _prepared(adapter)
    temp = filesystem.create_temp(target)
    temp_name = temp.basename
    temp_node = adapter.children[("safe", temp_name)]
    temp_fd = next(fd for fd, node in adapter.fd_nodes.items() if node == temp_node)
    safe_fds_before = {
        fd for fd, node in adapter.fd_nodes.items() if node == "safe"
    }

    temp.close()
    temp.close()

    assert ("safe", temp_name) not in adapter.children
    assert temp_fd not in adapter.fd_nodes
    assert len(
        {fd for fd, node in adapter.fd_nodes.items() if node == "safe"}
    ) == len(safe_fds_before) - 1
    assert ("fsync", "safe") in adapter.events
    target.close()
    filesystem.close()


def test_temp_context_exit_unlinks_and_fsyncs_exact_temp() -> None:
    adapter = _PosixAdapter()
    filesystem, target = _prepared(adapter)

    with filesystem.create_temp(target) as temp:
        temp_name = temp.basename

    assert ("safe", temp_name) not in adapter.children
    assert ("fsync", "safe") in adapter.events
    assert temp.closed
    target.close()
    filesystem.close()


@pytest.mark.parametrize("operation", ["fchmod", "listxattr:temp", "setxattr"])
def test_metadata_apply_failures_use_stable_error_and_discard(
    operation: str,
) -> None:
    from mochi.security.safe_filesystem import UnsupportedSecurityMetadata
    from mochi.tools.file_transaction import atomic_write_bytes

    adapter = _PosixAdapter()
    filesystem, target = _prepared(adapter)
    snapshot = filesystem.capture_metadata(target)
    cause = OSError(f"{operation} failed")
    adapter.fail(operation, cause)

    with pytest.raises(UnsupportedSecurityMetadata) as raised:
        atomic_write_bytes(target, AFTER, snapshot)

    assert raised.value.phase == "apply"
    assert raised.value.platform == "linux"
    assert raised.value.cause is cause
    assert raised.value.__cause__ is cause
    assert adapter.children[("safe", "note.txt")] == "original"
    assert bytes(adapter.content["original"]) == BASE
    assert not any(
        node.startswith("temp:") for node in adapter.children.values()
    )
    assert any(event[0] == "unlink" for event in adapter.events)
    target.close()
    filesystem.close()


@pytest.mark.parametrize("tamper", ["content", "metadata"])
def test_staged_tamper_is_rejected_and_exact_temp_discarded(
    tamper: str,
) -> None:
    from mochi.security.safe_filesystem import UnsafeFilesystemTarget
    from mochi.tools.file_transaction import atomic_write_bytes

    adapter = _PosixAdapter()
    filesystem, target = _prepared(adapter)
    snapshot = filesystem.capture_metadata(target)
    if tamper == "content":
        adapter.tamper_temp_content_read = True
    else:
        adapter.tamper_temp_metadata_on_second_list = True

    with pytest.raises(UnsafeFilesystemTarget, match="staged"):
        atomic_write_bytes(target, AFTER, snapshot)

    assert adapter.children[("safe", "note.txt")] == "original"
    assert bytes(adapter.content["original"]) == BASE
    assert not any(
        node.startswith("temp:") for node in adapter.children.values()
    )
    assert any(event[0] == "unlink" for event in adapter.events)
    target.close()
    filesystem.close()


@pytest.mark.parametrize("tamper", ["content", "metadata", "identity"])
def test_base_tamper_is_rejected_before_replace_and_original_preserved(
    tamper: str,
) -> None:
    from mochi.security.safe_filesystem import UnsafeFilesystemTarget
    from mochi.tools.file_transaction import atomic_write_bytes

    adapter = _PosixAdapter()
    filesystem, target = _prepared(adapter)
    snapshot = filesystem.capture_metadata(target)
    if tamper == "content":
        adapter.content["original"][:] = b"changed-base"
    elif tamper == "metadata":
        adapter.xattrs["original"][b"user.alpha"] = b"changed"
    else:
        adapter.target_fstat_ino = 42

    with pytest.raises(UnsafeFilesystemTarget):
        atomic_write_bytes(target, AFTER, snapshot)

    assert adapter.children[("safe", "note.txt")] == "original"
    assert not any(
        node.startswith("temp:") for node in adapter.children.values()
    )
    assert not any(event[0] == "replace" for event in adapter.events)
    target.close()
    filesystem.close()


def test_discard_identity_mismatch_skips_unlink_but_drains_resources() -> None:
    from mochi.security.safe_filesystem import UnsafeFilesystemTarget

    adapter = _PosixAdapter()
    filesystem, target = _prepared(adapter)
    temp = filesystem.create_temp(target)
    temp_node = adapter.children[("safe", temp.basename)]
    temp_fd = next(fd for fd, node in adapter.fd_nodes.items() if node == temp_node)
    temp_parent_fds = {
        fd for fd, node in adapter.fd_nodes.items() if node == "safe"
    }
    adapter.children[("safe", temp.basename)] = "original"
    adapter.fail(f"close:{temp_node}", OSError("temp close failed"))
    adapter.fail("fsync:parent", OSError("parent fsync failed"))

    with pytest.raises(UnsafeFilesystemTarget) as raised:
        filesystem.discard_temp(temp)

    notes = getattr(raised.value, "__notes__", ())
    assert any("temp close failed" in note for note in notes)
    assert any("parent fsync failed" in note for note in notes)
    assert not any(event[0] == "unlink" for event in adapter.events)
    assert temp.closed
    assert temp_fd not in adapter.fd_nodes
    assert not temp_parent_fds.issubset(adapter.fd_nodes)
    target.close()
    filesystem.close()


def test_successor_identity_mismatch_is_committed_and_drains_operands() -> None:
    from mochi.security.safe_filesystem import CommittedFilesystemMutationError
    from mochi.tools.file_transaction import atomic_write_bytes

    adapter = _PosixAdapter()
    filesystem, target = _prepared(adapter)
    snapshot = filesystem.capture_metadata(target)
    adapter.successor_stat_ino = 999

    with pytest.raises(CommittedFilesystemMutationError) as raised:
        atomic_write_bytes(target, AFTER, snapshot)

    assert raised.value.phase == "successor_verification"
    assert raised.value.committed is True
    assert sum(event[0] == "replace" for event in adapter.events) == 1
    assert not any(event[0] == "unlink" for event in adapter.events)
    assert target.closed
    assert adapter.fd_nodes == {100: "workspace"}
    filesystem.close()

def _ready_for_replace(adapter, filesystem, target, snapshot):
    temp = filesystem.create_temp(target)
    assert filesystem.write_temp(temp, memoryview(AFTER)) == len(AFTER)
    filesystem.apply_metadata_snapshot(temp, snapshot)
    filesystem.verify_staged(temp, snapshot)
    filesystem.flush_temp(temp)
    filesystem.revalidate_base(target, snapshot)
    temp_node = adapter.children[("safe", temp.basename)]
    return temp, temp_node


@pytest.mark.parametrize(
    "tamper",
    ["staged_content", "staged_metadata", "base_content", "base_metadata"],
)
def test_replace_rechecks_content_and_metadata_immediately_before_commit(
    tamper: str,
) -> None:
    from mochi.security.safe_filesystem import UnsafeFilesystemTarget

    adapter = _PosixAdapter()
    filesystem, target = _prepared(adapter)
    snapshot = filesystem.capture_metadata(target)
    temp, temp_node = _ready_for_replace(
        adapter, filesystem, target, snapshot
    )
    if tamper == "staged_content":
        adapter.content[temp_node][:] = b"late-staged-tamper"
    elif tamper == "staged_metadata":
        adapter.xattrs[temp_node][b"user.alpha"] = b"late-staged-tamper"
    elif tamper == "base_content":
        adapter.content["original"][:] = b"late-base-tamper"
    else:
        adapter.xattrs["original"][b"user.alpha"] = b"late-base-tamper"

    with pytest.raises(UnsafeFilesystemTarget):
        filesystem.replace(temp, target)

    assert not any(event[0] == "replace" for event in adapter.events)
    assert adapter.children[("safe", "note.txt")] == "original"
    filesystem.discard_temp(temp)
    target.close()
    filesystem.close()


def test_postreplace_concurrent_temp_close_waits_for_owner_cleanup() -> None:
    adapter = _PosixAdapter()
    filesystem, target = _prepared(adapter)
    snapshot = filesystem.capture_metadata(target)
    temp, _ = _ready_for_replace(adapter, filesystem, target, snapshot)
    started = threading.Event()
    finished = threading.Event()
    blocked_during_callback: list[bool] = []
    contender_errors: list[BaseException] = []

    def contender() -> None:
        started.set()
        try:
            temp.close()
        except BaseException as exc:
            contender_errors.append(exc)
        finally:
            finished.set()

    contender_thread = threading.Thread(target=contender)

    def after_replace() -> None:
        contender_thread.start()
        assert started.wait(1)
        blocked_during_callback.append(not finished.wait(0.05))

    adapter.after_replace = after_replace
    try:
        successor = filesystem.replace(temp, target)
    finally:
        contender_thread.join(timeout=2)
        if not target.closed:
            target.close()
        filesystem.close()

    assert successor.file_id != "41"
    assert blocked_during_callback == [True]
    assert finished.is_set()
    assert contender_errors == []
    assert temp.closed and target.closed
    assert not any(event[0] == "unlink" for event in adapter.events)


def test_consume_temp_exception_after_commit_still_drains_destination() -> None:
    from mochi.security.safe_filesystem import CommittedFilesystemMutationError
    from mochi.security.safe_fs_posix import PosixSafeFilesystem

    class _ConsumeRaises(PosixSafeFilesystem):
        def _consume_temp(self, temp):
            error = super()._consume_temp(temp)
            if error is not None:
                return error
            raise OSError("consume hook failed")

    adapter = _PosixAdapter()
    filesystem = _ConsumeRaises("/workspace", adapter=adapter)
    target = filesystem.prepare_target("safe/note.txt", _authorization())
    snapshot = filesystem.capture_metadata(target)
    temp, _ = _ready_for_replace(adapter, filesystem, target, snapshot)

    with pytest.raises(CommittedFilesystemMutationError) as raised:
        filesystem.replace(temp, target)

    assert raised.value.committed is True
    assert raised.value.phase == "operand_cleanup"
    assert isinstance(raised.value.cause, OSError)
    assert "consume hook failed" in str(raised.value.cause)
    assert temp.closed and target.closed
    assert adapter.fd_nodes == {100: "workspace"}
    filesystem.close()


def test_injected_adapter_requires_declared_semantic_capabilities() -> None:
    from mochi.security.safe_filesystem import SafeFilesystemUnavailable

    adapter = _PosixAdapter()
    adapter.supports_dir_fd = frozenset({"open", "stat", "unlink"})

    with pytest.raises(SafeFilesystemUnavailable, match="replace"):
        _filesystem(adapter)

    assert adapter.fd_nodes == {}


def test_new_metadata_capture_supersedes_old_snapshot() -> None:
    from mochi.security.safe_filesystem import UnsafeFilesystemTarget

    adapter = _PosixAdapter()
    filesystem, target = _prepared(adapter)
    first = filesystem.capture_metadata(target)
    second = filesystem.capture_metadata(target)
    try:
        assert len(filesystem._metadata) == 1
        temp = filesystem.create_temp(target)
        try:
            with pytest.raises(UnsafeFilesystemTarget, match="snapshot"):
                filesystem.apply_metadata_snapshot(temp, first)
            filesystem.apply_metadata_snapshot(temp, second)
        finally:
            if not temp.closed:
                filesystem.discard_temp(temp)
    finally:
        target.close()
        filesystem.close()


def test_committed_error_survives_hostile_cause_formatting() -> None:
    from mochi.security.safe_filesystem import CommittedFilesystemMutationError
    from mochi.tools.file_transaction import atomic_write_bytes

    class _BadStringError(OSError):
        def __str__(self) -> str:
            raise RuntimeError("hostile formatter")

    adapter = _PosixAdapter()
    filesystem, target = _prepared(adapter)
    snapshot = filesystem.capture_metadata(target)
    cause = _BadStringError()
    adapter.fail("fsync:parent", cause)

    with pytest.raises(CommittedFilesystemMutationError) as raised:
        atomic_write_bytes(target, AFTER, snapshot)

    assert raised.value.committed is True
    assert raised.value.phase == "parent_fsync"
    assert raised.value.cause is cause
    assert raised.value.__cause__ is cause
    filesystem.close()




@pytest.mark.parametrize("operand", ["source", "destination"])
def test_replace_revalidates_basenames_after_final_staged_fsync(
    operand: str,
) -> None:
    from mochi.security.safe_filesystem import UnsafeFilesystemTarget

    adapter = _PosixAdapter()
    filesystem, target = _prepared(adapter)
    snapshot = filesystem.capture_metadata(target)
    temp, temp_node = _ready_for_replace(
        adapter, filesystem, target, snapshot
    )
    hostile_node = f"hostile:{operand}"
    hostile_content = f"{operand}-rebind".encode()
    adapter._add_file(hostile_node, hostile_content)
    basename = temp.basename if operand == "source" else "note.txt"
    key = ("safe", basename)

    def rebind_name() -> None:
        adapter.children[key] = hostile_node

    adapter.on_temp_fsync = rebind_name
    try:
        with pytest.raises(UnsafeFilesystemTarget):
            filesystem.replace(temp, target)

        assert not any(event[0] == "replace" for event in adapter.events)
        assert adapter.children[key] == hostile_node
        assert bytes(adapter.content[hostile_node]) == hostile_content
        assert bytes(adapter.content["original"]) == BASE
        if operand == "source":
            assert adapter.children[("safe", "note.txt")] == "original"
        else:
            assert adapter.children[("safe", temp.basename)] == temp_node

        unlink_count = sum(event[0] == "unlink" for event in adapter.events)
        if operand == "source":
            with pytest.raises(UnsafeFilesystemTarget, match="discard"):
                filesystem.discard_temp(temp)
        else:
            filesystem.discard_temp(temp)
        discarded = [
            event for event in adapter.events if event[0] == "unlink"
        ][unlink_count:]
        if operand == "source":
            assert discarded == []
        else:
            assert len(discarded) == 1
            assert discarded[0][-1] == temp_node
    finally:
        if not temp.closed:
            filesystem.discard_temp(temp)
        if not target.closed:
            target.close()
        filesystem.close()


def test_replace_rejects_staged_mutation_while_base_is_hashed() -> None:
    from mochi.security.safe_filesystem import UnsafeFilesystemTarget

    adapter = _PosixAdapter()
    filesystem, target = _prepared(adapter)
    snapshot = filesystem.capture_metadata(target)
    temp, temp_node = _ready_for_replace(
        adapter, filesystem, target, snapshot
    )

    def mutate_staged() -> None:
        adapter.content[temp_node][:] = b"staged-race"
        adapter._touch_content(temp_node)

    adapter.on_base_pread = mutate_staged
    try:
        with pytest.raises(UnsafeFilesystemTarget):
            filesystem.replace(temp, target)

        assert not any(event[0] == "replace" for event in adapter.events)
        assert adapter.children[("safe", "note.txt")] == "original"
        assert bytes(adapter.content["original"]) == BASE
        assert adapter.children[("safe", temp.basename)] == temp_node
        filesystem.discard_temp(temp)
        assert any(
            event[0] == "unlink" and event[-1] == temp_node
            for event in adapter.events
        )
    finally:
        if not temp.closed:
            filesystem.discard_temp(temp)
        if not target.closed:
            target.close()
        filesystem.close()


@pytest.mark.parametrize("mutation", ["content", "metadata"])
def test_replace_rejects_base_mutation_during_final_staged_fsync(
    mutation: str,
) -> None:
    from mochi.security.safe_filesystem import UnsafeFilesystemTarget

    adapter = _PosixAdapter()
    filesystem, target = _prepared(adapter)
    snapshot = filesystem.capture_metadata(target)
    temp, temp_node = _ready_for_replace(
        adapter, filesystem, target, snapshot
    )

    def mutate_base() -> None:
        if mutation == "content":
            adapter.content["original"][:] = b"hostile-base-race"
            adapter._touch_content("original")
        else:
            adapter.xattrs["original"][b"user.alpha"] = b"hostile-metadata"
            adapter._touch_metadata("original")

    adapter.on_temp_fsync = mutate_base
    try:
        with pytest.raises(UnsafeFilesystemTarget):
            filesystem.replace(temp, target)

        assert not any(event[0] == "replace" for event in adapter.events)
        assert adapter.children[("safe", "note.txt")] == "original"
        if mutation == "content":
            assert bytes(adapter.content["original"]) == b"hostile-base-race"
        else:
            assert (
                adapter.xattrs["original"][b"user.alpha"]
                == b"hostile-metadata"
            )
        assert adapter.children[("safe", temp.basename)] == temp_node
        filesystem.discard_temp(temp)
        assert any(
            event[0] == "unlink" and event[-1] == temp_node
            for event in adapter.events
        )
    finally:
        if not temp.closed:
            filesystem.discard_temp(temp)
        if not target.closed:
            target.close()
        filesystem.close()


def test_replace_fails_closed_without_high_resolution_change_token() -> None:
    from mochi.security.safe_filesystem import SafeFilesystemUnavailable

    adapter = _PosixAdapter()
    filesystem, target = _prepared(adapter)
    snapshot = filesystem.capture_metadata(target)
    temp, _ = _ready_for_replace(adapter, filesystem, target, snapshot)
    adapter.missing_change_token_field = "st_mtime_ns"
    try:
        with pytest.raises(SafeFilesystemUnavailable, match="st_mtime_ns"):
            filesystem.replace(temp, target)

        assert not any(event[0] == "replace" for event in adapter.events)
    finally:
        adapter.missing_change_token_field = None
        if not temp.closed:
            filesystem.discard_temp(temp)
        if not target.closed:
            target.close()
        filesystem.close()

def _real_metadata_sha(file_fd: int) -> str:
    names = sorted(
        os.fsencode(name) if isinstance(name, str) else bytes(name)
        for name in os.listxattr(file_fd)
    )
    attrs = {name: os.getxattr(file_fd, name) for name in names}
    info = os.fstat(file_fd)
    return _metadata_sha(
        uid=info.st_uid,
        gid=info.st_gid,
        mode=stat.S_IMODE(info.st_mode),
        xattrs=attrs,
    )


def _real_authorization(workspace, target, metadata_sha: str):
    from mochi.security.file_contract import (
        AuthorizationContext,
        AuthorizationEnvelope,
        ChangeEntry,
        FileChangeRequest,
        FileIdentity,
    )

    root_info = workspace.stat()
    target_info = target.stat()
    root_identity = FileIdentity(
        "posix",
        str(root_info.st_dev),
        str(root_info.st_ino),
        root_info.st_nlink,
        False,
    )
    target_identity = FileIdentity(
        "posix",
        str(target_info.st_dev),
        str(target_info.st_ino),
        target_info.st_nlink,
        False,
    )
    entry = ChangeEntry(
        entry_id="real-posix",
        relative_path=target.name,
        operation="update",
        base_sha256=_sha(BASE),
        after_sha256=_sha(AFTER),
        base_identity=target_identity,
        before_blob_id="before",
        after_blob_id="after",
        mode_before=stat.S_IMODE(target_info.st_mode),
        mode_after=stat.S_IMODE(target_info.st_mode),
        base_metadata_sha256=metadata_sha,
        after_metadata_sha256=metadata_sha,
        rename_source=None,
        dependency_group="real-posix",
    )
    return AuthorizationEnvelope(
        schema_version=1,
        kind="file_change",
        context=AuthorizationContext(
            requester_id="requester",
            session_id="session",
            task_id="task",
            workspace_root=str(workspace),
            workspace_identity=root_identity,
        ),
        policy_version="policy-v1",
        file_request=FileChangeRequest(
            entries=(entry,), patch_sha256=_sha(b"patch")
        ),
        exec_request=None,
    )


@pytest.mark.skipif(sys.platform != "linux", reason="requires real Linux POSIX APIs")
@pytest.mark.parametrize("metadata_kind", ["user_xattrs", "posix_acl"])
def test_real_linux_atomic_replace_preserves_security_metadata(
    tmp_path, metadata_kind: str
) -> None:
    from mochi.security.safe_fs_posix import PosixSafeFilesystem
    from mochi.tools.file_transaction import atomic_write_bytes

    target_path = tmp_path / "note.txt"
    target_path.write_bytes(BASE)
    os.chmod(target_path, 0o640)
    os.setxattr(target_path, b"user.mochi", b"value")
    expected_xattrs = {b"user.mochi": b"value"}
    expected_mode = stat.S_IMODE(target_path.stat().st_mode)
    if metadata_kind == "user_xattrs":
        raw_name = b"user.mochi-\xff"
        try:
            os.setxattr(target_path, raw_name, b"raw-value")
        except OSError as exc:
            if exc.errno not in {
                errno.EINVAL,
                errno.ENOTSUP,
                errno.EOPNOTSUPP,
            }:
                raise
        else:
            expected_xattrs[raw_name] = b"raw-value"
    else:
        owner_uid = target_path.stat().st_uid
        named_uid = owner_uid + 1 if owner_uid < 0xFFFFFFFE else owner_uid - 1
        acl = (
            struct.pack("<I", 2)
            + struct.pack("<HHI", 0x01, 0o6, 0xFFFFFFFF)
            + struct.pack("<HHI", 0x02, 0o4, named_uid)
            + struct.pack("<HHI", 0x04, 0o4, 0xFFFFFFFF)
            + struct.pack("<HHI", 0x10, 0o4, 0xFFFFFFFF)
            + struct.pack("<HHI", 0x20, 0o0, 0xFFFFFFFF)
        )
        try:
            os.setxattr(target_path, ACL, acl)
        except OSError as exc:
            if exc.errno in {
                errno.EINVAL,
                errno.ENOTSUP,
                errno.EOPNOTSUPP,
                errno.EPERM,
                errno.EACCES,
                errno.EROFS,
            }:
                pytest.skip(
                    "POSIX ACL fixture rejected or unsupported: "
                    f"errno {exc.errno}"
                )
            raise
        expected_xattrs[ACL] = os.getxattr(target_path, ACL)
        expected_mode = stat.S_IMODE(target_path.stat().st_mode)

    file_fd = os.open(target_path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        metadata_sha = _real_metadata_sha(file_fd)
    finally:
        os.close(file_fd)
    authorization = _real_authorization(
        tmp_path, target_path, metadata_sha
    )
    filesystem = PosixSafeFilesystem(tmp_path)
    target = filesystem.prepare_target(target_path.name, authorization)
    snapshot = filesystem.capture_metadata(target)

    result = atomic_write_bytes(target, AFTER, snapshot)

    assert result.bytes_written == len(AFTER)
    assert target_path.read_bytes() == AFTER
    for name, value in expected_xattrs.items():
        assert os.getxattr(target_path, name) == value
    assert stat.S_IMODE(target_path.stat().st_mode) == expected_mode
    assert target.closed
    filesystem.close()
