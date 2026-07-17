from __future__ import annotations

import hashlib
import json
import stat
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
                mode=0o040700, dev=10, ino=20, nlink=1, uid=1000, gid=1000
            ),
            "safe": dict(
                mode=0o040700, dev=10, ino=21, nlink=1, uid=1000, gid=1000
            ),
            "original": dict(
                mode=0o100640, dev=1, ino=41, nlink=1, uid=1000, gid=1001
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
        return SimpleNamespace(
            st_mode=values["mode"],
            st_dev=values["dev"],
            st_ino=values["ino"],
            st_nlink=values["nlink"],
            st_uid=values["uid"],
            st_gid=values["gid"],
        )

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
        self.offsets[fd] += written
        self.events.append(("write", node, payload, offset))
        return written

    def pread(self, fd: int, size: int, offset: int) -> bytes:
        node = self.fd_nodes[fd]
        self.events.append(("pread", node, offset))
        self._raise(f"pread:{node}")
        if self.tamper_temp_content_read and node.startswith("temp:"):
            self.tamper_temp_content_read = False
            self.content[node][:] = b"tampered-content"
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

    def fchmod(self, fd: int, mode: int) -> None:
        node = self.fd_nodes[fd]
        self.events.append(("fchmod", node, mode))
        self._raise("fchmod")
        self.info[node]["mode"] = 0o100000 | mode

    def removexattr(self, fd: int, name: bytes) -> None:
        node = self.fd_nodes[fd]
        self.events.append(("removexattr", node, name))
        self._raise("removexattr")
        del self.xattrs[node][name]

    def setxattr(self, fd: int, name: bytes, value: bytes) -> None:
        node = self.fd_nodes[fd]
        self.events.append(("setxattr", node, name))
        self._raise("setxattr")
        self.xattrs[node][name] = value

    def fsync(self, fd: int) -> None:
        node = self.fd_nodes[fd]
        kind = "temp" if node.startswith("temp:") else "parent"
        self.events.append(("fsync", node))
        self._raise(f"fsync:{kind}")

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
