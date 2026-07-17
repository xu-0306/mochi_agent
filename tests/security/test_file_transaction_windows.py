from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from threading import Thread
from types import SimpleNamespace

import pytest

from tests.security.test_safe_filesystem import (
    _FakeWindowsAdapter,
    _windows_authorization,
)

BASE = b"base-content"
AFTER = b"after-content"
SECURITY = {
    "owner": "0102",
    "group": "0304",
    "dacl": "0506",
    "dacl_present": True,
    "dacl_protected": True,
    "sacl": None,
    "sacl_present": False,
    "sacl_protected": False,
    "sacl_state": "inaccessible",
}


def _security_digest(value: dict[str, object] = SECURITY) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


class _StatefulWindowsAdapter(_FakeWindowsAdapter):
    semantics = frozenset(
        {
            "content_read_at",
            "content_write",
            "file_flush",
            "directory_flush",
            "change_token",
            "security_capture",
            "security_apply",
            "relative_rename",
            "handle_disposition",
            "duplicate_handle",
        }
    )
    platform = "win32-model"

    def __init__(self, *, sacl_state: str = "inaccessible") -> None:
        super().__init__()
        self.contents = {"note-original": bytearray(BASE)}
        self.positions: dict[str, int] = {"note-original": 7}
        self.write_plan: list[object] = []
        self.security = {
            "note-original": dict(SECURITY, sacl_state=sacl_state),
        }
        self.tokens: dict[str, int] = {"note-original": 1}
        self.sacl_state = sacl_state
        self.disposed_nodes: list[str] = []
        self.directory_flushes: list[str] = []
        self.failures: dict[str, BaseException] = {}

    def _fail(self, name: str) -> None:
        failure = self.failures.get(name)
        if failure is not None:
            raise failure

    def duplicate_handle(self, handle: object) -> object:
        self.calls.append(("DuplicateHandle", handle))
        duplicate = object()
        self.nodes[duplicate] = self.nodes[handle]
        return duplicate

    def ntcreate_relative(
        self, root: object, basename: str, *, directory: bool
    ) -> object:
        handle = super().ntcreate_relative(root, basename, directory=directory)
        if not directory:
            self.calls.append(
                (
                    "target_access",
                    handle,
                    frozenset({"FILE_READ_DATA", "READ_CONTROL"}),
                    self.sacl_state,
                )
            )
        return handle

    def ntcreate_new_relative(self, root: object, basename: str) -> object:
        handle = super().ntcreate_new_relative(root, basename)
        node = self.nodes[handle]
        self.contents[node] = bytearray()
        self.positions[node] = 0
        self.security[node] = {
            "owner": "temp-owner",
            "group": "temp-group",
            "dacl": None,
            "dacl_present": False,
            "dacl_protected": False,
            "sacl": None,
            "sacl_present": False,
            "sacl_protected": False,
            "sacl_state": self.sacl_state,
        }
        self.tokens[node] = 1
        self.calls.append(
            (
                "temp_access",
                handle,
                frozenset(
                    {
                        "FILE_READ_DATA",
                        "FILE_WRITE_DATA",
                        "READ_CONTROL",
                        "WRITE_OWNER",
                        "WRITE_DAC",
                    }
                ),
                self.sacl_state,
            )
        )
        return handle

    def read_at(self, handle: object, size: int, offset: int) -> bytes:
        self._fail("read")
        node = self.nodes[handle]
        self.calls.append(("ReadFileAt", handle, size, offset))
        return bytes(self.contents[node][offset : offset + size])

    def write(self, handle: object, data: memoryview) -> int:
        self._fail("write")
        node = self.nodes[handle]
        if self.write_plan:
            step = self.write_plan.pop(0)
            if isinstance(step, BaseException):
                raise step
            count = min(int(step), len(data))
        else:
            count = len(data)
        position = self.positions[node]
        self.contents[node][position : position + count] = data[:count]
        self.positions[node] += count
        self.tokens[node] += 1
        self.calls.append(("WriteFile", handle, count))
        return count

    def flush_file(self, handle: object) -> None:
        self._fail("flush_file")
        self.calls.append(("FlushFileBuffers", handle))

    def flush_directory(self, handle: object) -> None:
        self._fail("flush_directory")
        self.directory_flushes.append(self.nodes[handle])
        self.calls.append(("NtFlushBuffersFile", handle))

    def change_token(self, handle: object) -> object:
        self._fail("change_token")
        node = self.nodes[handle]
        return (self.identities[node], len(self.contents.get(node, b"")), self.tokens.get(node, 0))

    def security_descriptor(self, handle: object, *, include_sacl: bool) -> object:
        self._fail("security_capture")
        node = self.nodes[handle]
        value = dict(self.security[node])
        if not include_sacl:
            value.update(sacl=None, sacl_present=False, sacl_protected=False)
        self.calls.append(("GetKernelObjectSecurity", handle, include_sacl))
        return SimpleNamespace(raw_descriptor=b"self-relative", **value)

    def apply_security_descriptor(self, handle: object, metadata: object) -> None:
        self._fail("security_apply")
        node = self.nodes[handle]
        self.security[node] = {
            name: getattr(metadata, name)
            for name in SECURITY
        }
        self.tokens[node] += 1
        self.calls.append(("SetKernelObjectSecurity", handle, metadata.sacl_state))

    def ntset_unlink(self, handle: object) -> None:
        self._fail("disposition")
        node = self.nodes[handle]
        self.disposed_nodes.append(node)
        super().ntset_unlink(handle)

    def ntset_replace(self, handle: object, root: object, basename: str) -> None:
        self._fail("replace")
        old_node = self.children[(self.nodes[root], basename)]
        source_node = self.nodes[handle]
        super().ntset_replace(handle, root, basename)
        self.contents[old_node] = bytearray(self.contents[source_node])
        self.security[old_node] = dict(self.security[source_node])
        self.tokens[old_node] = self.tokens[source_node]


def _authorized(adapter: _StatefulWindowsAdapter):
    authorization = _windows_authorization()
    request = authorization.file_request
    assert request is not None
    entry = replace(
        request.entries[0],
        base_sha256=hashlib.sha256(BASE).hexdigest(),
        after_sha256=hashlib.sha256(AFTER).hexdigest(),
        base_metadata_sha256=_security_digest(),
        after_metadata_sha256=_security_digest(),
    )
    return replace(authorization, file_request=replace(request, entries=(entry,)))


def _transaction(adapter: _StatefulWindowsAdapter | None = None):
    from mochi.security.safe_fs_windows import WindowsSafeFilesystem

    adapter = adapter or _StatefulWindowsAdapter()
    filesystem = WindowsSafeFilesystem("C:/workspace", adapter=adapter)
    target = filesystem.prepare_target("safe/note.txt", _authorized(adapter))
    snapshot = filesystem.capture_metadata(target)
    return adapter, filesystem, target, snapshot


@pytest.mark.parametrize("missing", sorted(_StatefulWindowsAdapter.semantics))
def test_windows_owner_requires_declared_native_semantics(missing: str) -> None:
    from mochi.security.safe_filesystem import SafeFilesystemUnavailable
    from mochi.security.safe_fs_windows import WindowsSafeFilesystem

    adapter = _StatefulWindowsAdapter()
    adapter.semantics = adapter.semantics - {missing}
    with pytest.raises(SafeFilesystemUnavailable, match=missing):
        WindowsSafeFilesystem("C:/workspace", adapter=adapter)


def test_target_and_temp_access_and_explicit_sacl_state() -> None:
    adapter, filesystem, target, _snapshot = _transaction()
    temp = filesystem.create_temp(target)
    target_access = next(call for call in adapter.calls if call[0] == "target_access")
    temp_access = next(call for call in adapter.calls if call[0] == "temp_access")
    assert {"FILE_READ_DATA", "READ_CONTROL"} <= target_access[2]
    assert {"FILE_READ_DATA", "FILE_WRITE_DATA", "READ_CONTROL", "WRITE_OWNER", "WRITE_DAC"} <= temp_access[2]
    assert target_access[3] in {"included", "inaccessible"}
    assert temp_access[3] in {"included", "inaccessible"}
    temp.close()
    target.close()
    filesystem.close()


def test_capture_is_opaque_handle_only_and_recapture_supersedes_snapshot() -> None:
    from mochi.security.safe_filesystem import UnsafeFilesystemTarget

    adapter, filesystem, target, first = _transaction()
    second = filesystem.capture_metadata(target)
    assert first.canonical_metadata_sha256 == _security_digest()
    assert second.canonical_metadata_sha256 == _security_digest()
    temp = filesystem.create_temp(target)
    with pytest.raises(UnsafeFilesystemTarget, match="exact owner-issued"):
        filesystem.apply_metadata_snapshot(temp, first)
    filesystem.apply_metadata_snapshot(temp, second)
    assert all(call[0] != "absolute_security" for call in adapter.calls)
    temp.close()
    target.close()
    filesystem.close()


@pytest.mark.parametrize(
    ("failure", "phase"),
    [("security_capture", "capture"), ("security_apply", "apply"), ("read", "verify")],
)
def test_security_inability_is_wrapped_with_exact_phase(failure: str, phase: str) -> None:
    from mochi.security.safe_filesystem import UnsupportedSecurityMetadata

    adapter = _StatefulWindowsAdapter()
    if failure == "security_capture":
        adapter.failures[failure] = OSError("denied")
        filesystem = __import__("mochi.security.safe_fs_windows", fromlist=["WindowsSafeFilesystem"]).WindowsSafeFilesystem("C:/workspace", adapter=adapter)
        target = filesystem.prepare_target("safe/note.txt", _authorized(adapter))
        def action() -> None:
            filesystem.capture_metadata(target)
    else:
        adapter, filesystem, target, snapshot = _transaction(adapter)
        temp = filesystem.create_temp(target)
        if failure == "read":
            filesystem.write_temp(temp, memoryview(AFTER))
            filesystem.apply_metadata_snapshot(temp, snapshot)
            adapter.failures[failure] = OSError("denied")
            def action() -> None:
                filesystem.verify_staged(temp, snapshot)
        else:
            adapter.failures[failure] = OSError("denied")
            def action() -> None:
                filesystem.apply_metadata_snapshot(temp, snapshot)
    with pytest.raises(UnsupportedSecurityMetadata) as raised:
        action()
    assert raised.value.phase == phase
    assert raised.value.platform == "win32-model"
    assert str(raised.value.cause) == "denied"
    filesystem.close()


def test_atomic_write_retries_short_writes_preserves_security_and_offset() -> None:
    from mochi.tools.file_transaction import atomic_write_bytes

    adapter = _StatefulWindowsAdapter()
    adapter.write_plan = [InterruptedError(), 2, 3, 100]
    adapter, filesystem, target, snapshot = _transaction(adapter)
    result = atomic_write_bytes(target, AFTER, snapshot)
    assert result.bytes_written == len(AFTER)
    assert bytes(adapter.contents["note-original"]) == AFTER
    assert adapter.security["note-original"] == SECURITY
    assert adapter.positions["note-original"] == 7
    assert any(call[0] == "FlushFileBuffers" for call in adapter.calls)
    assert adapter.directory_flushes == ["safe-original"]
    assert target.closed
    filesystem.close()


def test_direct_temp_close_disposes_by_handle_flushes_parent_and_closes() -> None:
    adapter, filesystem, target, _snapshot = _transaction()
    temp = filesystem.create_temp(target)
    node = adapter.nodes[adapter.temp_handles[-1]]
    temp.close()
    assert adapter.disposed_nodes == [node]
    assert adapter.directory_flushes == ["safe-original"]
    assert temp.closed
    assert node not in adapter.nodes.values()
    target.close()
    filesystem.close()


def test_precommit_failure_discards_exact_temp_and_preserves_target() -> None:
    from mochi.tools.file_transaction import atomic_write_bytes

    adapter, filesystem, target, snapshot = _transaction()
    adapter.failures["flush_file"] = OSError("flush failed")
    with pytest.raises(OSError, match="flush failed"):
        atomic_write_bytes(target, AFTER, snapshot)
    assert bytes(adapter.contents["note-original"]) == BASE
    assert adapter.disposed_nodes and target.closed is False
    target.close()
    filesystem.close()


def test_committed_parent_flush_failure_drains_both_operands() -> None:
    from mochi.security.safe_filesystem import CommittedFilesystemMutationError
    from mochi.tools.file_transaction import atomic_write_bytes

    adapter, filesystem, target, snapshot = _transaction()
    original_flush = adapter.flush_directory
    count = 0

    def fail_after_commit(handle: object) -> None:
        nonlocal count
        count += 1
        if adapter.renamed:
            raise OSError("parent flush failed")
        original_flush(handle)

    adapter.flush_directory = fail_after_commit
    with pytest.raises(CommittedFilesystemMutationError) as raised:
        atomic_write_bytes(target, AFTER, snapshot)
    assert raised.value.phase == "parent_flush"
    assert target.closed
    assert all(temp.closed for temp in [record.temp for record in filesystem._temps.values()])
    assert bytes(adapter.contents["note-original"]) == AFTER
    filesystem.close()


def test_concurrent_close_and_replace_are_serialized() -> None:
    from mochi.tools.file_transaction import atomic_write_bytes

    adapter, filesystem, target, snapshot = _transaction()
    failures: list[BaseException] = []

    def replace() -> None:
        try:
            atomic_write_bytes(target, AFTER, snapshot)
        except BaseException as exc:
            failures.append(exc)

    thread = Thread(target=replace)
    thread.start()
    target.close()
    thread.join()
    assert target.closed
    assert len(failures) <= 1
    filesystem.close()




@pytest.mark.skipif(os.name != "nt", reason="requires native Windows NTFS semantics")
def test_native_windows_atomic_write_preserves_security(tmp_path: Path) -> None:
    from mochi.security.file_contract import AuthorizationContext
    from mochi.security.safe_fs_windows import (
        WindowsSafeFilesystem,
        _WindowsNativeAdapter,
    )
    from mochi.tools.file_transaction import atomic_write_bytes

    safe = tmp_path / "safe"
    safe.mkdir()
    note = safe / "note.txt"
    note.write_bytes(BASE)
    adapter = _WindowsNativeAdapter()
    if not adapter.available:
        pytest.fail(
            "Windows native adapter did not expose required fail-closed capabilities"
        )

    root = adapter.createfile_workspace(str(tmp_path))
    parent = adapter.ntcreate_relative(root, "safe", directory=True)
    handle = adapter.ntcreate_relative(parent, "note.txt", directory=False)
    try:
        root_identity = adapter.identity(root)
        identity = adapter.identity(handle)
        native = adapter.security_descriptor(
            handle, include_sacl=adapter.sacl_access(handle) == "included"
        )
        metadata_sha = WindowsSafeFilesystem._metadata_digest(native)
    finally:
        adapter.close(handle)
        adapter.close(parent)
        adapter.close(root)

    authorization = _windows_authorization()
    request = authorization.file_request
    assert request is not None
    entry = replace(
        request.entries[0],
        base_identity=identity,
        base_sha256=hashlib.sha256(BASE).hexdigest(),
        after_sha256=hashlib.sha256(AFTER).hexdigest(),
        base_metadata_sha256=metadata_sha,
        after_metadata_sha256=metadata_sha,
    )
    authorization = replace(
        authorization,
        context=AuthorizationContext(
            requester_id="requester-1",
            session_id="session-1",
            task_id="task-1",
            workspace_root=str(tmp_path),
            workspace_identity=root_identity,
        ),
        file_request=replace(request, entries=(entry,)),
    )

    filesystem = WindowsSafeFilesystem(tmp_path)
    try:
        target = filesystem.prepare_target("safe/note.txt", authorization)
        snapshot = filesystem.capture_metadata(target)
        result = atomic_write_bytes(target, AFTER, snapshot)
        assert result.successor_identity != identity
        assert note.read_bytes() == AFTER
    finally:
        filesystem.close()

@pytest.mark.skipif(os.name != "nt", reason="requires native Windows NTFS semantics")
def test_native_windows_adapter_declares_exact_atomic_capabilities(tmp_path: Path) -> None:
    from mochi.security.safe_fs_windows import _WindowsNativeAdapter

    adapter = _WindowsNativeAdapter()
    if not adapter.available:
        pytest.fail("Windows native adapter did not expose required fail-closed capabilities")
    assert _StatefulWindowsAdapter.semantics <= adapter.semantics
    root = adapter.createfile_workspace(str(tmp_path))
    try:
        adapter.flush_directory(root)
    finally:
        adapter.close(root)
