from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    pass

from tests.security.safe_filesystem._support import (
    _entry,
    _FakeWindowsAdapter,
    _file_authorization,
    _native_authorization_for,
    _sha,
    _windows_atomic_authorization,
    _windows_authorization,
)


def test_windows_pins_handles_and_survives_junction_rebind() -> None:
    from mochi.security.safe_fs_windows import WindowsSafeFilesystem

    adapter = _FakeWindowsAdapter()
    filesystem = WindowsSafeFilesystem(
        "C:/workspace", adapter=adapter, enforce=True
    )
    target = filesystem.prepare_target(
        "safe/note.txt", _windows_authorization("delete")
    )
    adapter.children[("workspace", "safe")] = "outside"
    filesystem.unlink(target)

    creates = [
        call
        for call in adapter.calls
        if call[0] in {"CreateFileW", "NtCreateFile"}
    ]
    assert creates[0][0] == "CreateFileW"
    assert all(call[0] == "NtCreateFile" for call in creates[1:])
    assert all(
        not str(call[2]).startswith(("C:", "\\"))
        for call in creates[1:]
    )
    unlink = next(
        call for call in adapter.calls if call[0] == "NtSetInformationFile"
    )
    assert unlink[2] == "FileDispositionInformation"



@pytest.mark.parametrize(
    ("adapter", "reason"),
    [
        (_FakeWindowsAdapter(link_count=2), "hardlink"),
        (_FakeWindowsAdapter(parent_reparse=True), "reparse"),
        (_FakeWindowsAdapter(root_file_id="wrong"), "workspace identity"),
    ],
)
def test_windows_rejects_hardlinks_reparse_and_wrong_root(
    adapter, reason: str
) -> None:
    from mochi.security.safe_filesystem import UnsafeFilesystemTarget
    from mochi.security.safe_fs_windows import WindowsSafeFilesystem

    filesystem = WindowsSafeFilesystem(
        "C:/workspace", adapter=adapter, enforce=True
    )
    with pytest.raises(UnsafeFilesystemTarget, match=reason):
        filesystem.prepare_target("safe/note.txt", _windows_authorization())



def test_windows_enforce_mode_fails_closed_without_native_apis() -> None:
    from mochi.security.safe_filesystem import SafeFilesystemUnavailable
    from mochi.security.safe_fs_windows import WindowsSafeFilesystem

    with pytest.raises(SafeFilesystemUnavailable, match="native"):
        WindowsSafeFilesystem(
            "C:/workspace",
            adapter=_FakeWindowsAdapter(available=False),
            enforce=True,
        )


def test_windows_native_uses_file_id_info_and_no_absolute_mutation_fallback() -> None:
    source = Path("mochi/security/safe_fs_windows.py").read_text("utf-8")

    assert "GetFileInformationByHandleEx" in source
    assert "FileIdInfo" in source
    assert "NtCreateFile" in source
    assert "RootDirectory" in source
    assert "NtSetInformationFile" in source


def test_windows_native_handle_relative_unlink(tmp_path: Path) -> None:
    import os

    if os.name != "nt":
        pytest.skip("Windows native API test")

    from mochi.security.file_contract import AuthorizationContext
    from mochi.security.safe_fs_windows import (
        WindowsSafeFilesystem,
        _WindowsNativeAdapter,
    )

    adapter = _WindowsNativeAdapter()
    if not adapter.available:
        pytest.skip("Windows native APIs unavailable")

    target_path = tmp_path / "native.txt"
    target_path.write_text("native", encoding="utf-8")
    handle = adapter.createfile_workspace(str(tmp_path))
    try:
        workspace_identity = adapter.identity(handle)
        target_handle = adapter.ntcreate_relative(
            handle, "native.txt", directory=False
        )
        try:
            target_identity = adapter.identity(target_handle)
        finally:
            adapter.close(target_handle)
    finally:
        adapter.close(handle)

    authorization = replace(
        _file_authorization(
            "native.txt", target_identity, "delete"
        ),
        context=AuthorizationContext(
            requester_id="requester-1",
            session_id="session-1",
            task_id="task-1",
            workspace_root=str(tmp_path),
            workspace_identity=workspace_identity,
        ),
    )
    filesystem = WindowsSafeFilesystem(tmp_path, enforce=True)
    try:
        target = filesystem.prepare_target("native.txt", authorization)
        filesystem.unlink(target)
    finally:
        filesystem.close()

    assert not target_path.exists()


def test_windows_native_rejects_real_hardlink(
    tmp_path: Path,
) -> None:
    import os

    if os.name != "nt":
        pytest.skip("Windows native API test")

    from mochi.security.safe_filesystem import UnsafeFilesystemTarget
    from mochi.security.safe_fs_windows import WindowsSafeFilesystem

    first = tmp_path / "first.txt"
    first.write_text("same", encoding="utf-8")
    os.link(first, tmp_path / "second.txt")
    filesystem = WindowsSafeFilesystem(tmp_path, enforce=True)
    try:
        with pytest.raises(UnsafeFilesystemTarget, match="hardlink"):
            filesystem.prepare_target(
                "first.txt", _native_authorization_for(tmp_path)
            )
    finally:
        filesystem.close()



def test_windows_native_rejects_real_directory_symlink_when_available(
    tmp_path: Path,
) -> None:
    import os

    if os.name != "nt":
        pytest.skip("Windows native API test")

    from mochi.security.safe_filesystem import UnsafeFilesystemTarget
    from mochi.security.safe_fs_windows import WindowsSafeFilesystem

    real = tmp_path / "real"
    real.mkdir()
    (real / "note.txt").write_text("x", encoding="utf-8")
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(real, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink unavailable: {exc}")

    filesystem = WindowsSafeFilesystem(tmp_path, enforce=True)
    try:
        with pytest.raises(UnsafeFilesystemTarget, match="reparse"):
            filesystem.prepare_target(
                "alias/note.txt",
                _native_authorization_for(tmp_path),
            )
    finally:
        filesystem.close()


def test_windows_replace_reopens_and_verifies_successor_relative() -> None:
    from mochi.security.safe_filesystem import StagedTemp
    from mochi.security.safe_fs_windows import WindowsSafeFilesystem

    adapter = _FakeWindowsAdapter()
    filesystem = WindowsSafeFilesystem(
        "C:/workspace", adapter=adapter, enforce=True
    )
    destination = filesystem.prepare_target(
        "safe/note.txt", _windows_atomic_authorization()
    )
    filesystem.capture_metadata(destination)
    staged = filesystem.create_temp(destination)
    assert isinstance(staged, StagedTemp)
    staged_handle = adapter.temp_handles[-1]
    old_destination_path = "C:/workspace/safe/note.txt"
    adapter.calls.clear()

    filesystem.replace(staged, destination)

    rename_index = next(
        index
        for index, call in enumerate(adapter.calls)
        if call[:3]
        == (
            "NtSetInformationFile",
            staged_handle,
            "FileRenameInformation",
        )
    )
    reopen_index, reopen = next(
        (index, call)
        for index, call in enumerate(adapter.calls)
        if index > rename_index
        and call[0] == "NtCreateFile"
        and call[2] == "note.txt"
    )
    assert reopen[1] is adapter.relative_results[-1][0]
    assert not str(reopen[2]).startswith(("C:", "\\"))
    successor_handle = adapter.relative_results[-1][2]
    post_rename_paths = {
        call[1]: call[2]
        for call in adapter.calls[rename_index + 1 :]
        if call[0] == "GetFinalPathNameByHandleW"
    }
    assert post_rename_paths[staged_handle] == old_destination_path
    assert post_rename_paths[successor_handle] == old_destination_path
    assert reopen_index > rename_index
    assert ("CloseHandle", successor_handle) in adapter.calls
    assert staged.closed is True
    assert destination.closed is True
    filesystem.close()



@pytest.mark.parametrize(
    ("mismatch", "expected"),
    [
        ("source_path", "source.*path"),
        ("source_identity", "source.*identity"),
        ("successor_path", "successor.*path"),
        ("successor_identity", "successor.*identity"),
    ],
)
def test_windows_replace_rejects_unverified_successor(
    mismatch: str, expected: str
) -> None:
    from mochi.security.file_contract import FileIdentity
    from mochi.security.safe_filesystem import (
        CommittedFilesystemMutationError,
        UnsafeFilesystemTarget,
    )
    from mochi.security.safe_fs_windows import WindowsSafeFilesystem

    adapter = _FakeWindowsAdapter()
    filesystem = WindowsSafeFilesystem(
        "C:/workspace", adapter=adapter, enforce=True
    )
    destination = filesystem.prepare_target(
        "safe/note.txt", _windows_atomic_authorization()
    )
    filesystem.capture_metadata(destination)
    staged = filesystem.create_temp(destination)
    if mismatch == "source_path":
        adapter.source_path_after_rename = (
            "C:/workspace/safe/wrong.txt"
        )
    elif mismatch == "source_identity":
        adapter.source_identity_after_rename = FileIdentity(
            "windows", "10", "wrong", 1, False
        )
    elif mismatch == "successor_path":
        adapter.successor_path_after_rename = (
            "C:/workspace/safe/wrong.txt"
        )
    else:
        adapter.successor_identity_after_rename = FileIdentity(
            "windows", "10", "wrong", 1, False
        )

    with pytest.raises(
        CommittedFilesystemMutationError
    ) as raised:
        filesystem.replace(staged, destination)

    outcome = raised.value
    assert outcome.committed is True
    assert outcome.phase == "successor_verification"
    assert isinstance(outcome.cause, UnsafeFilesystemTarget)
    assert all(
        part in str(outcome.cause)
        for part in expected.split(".*")
    )
    assert outcome.__cause__ is outcome.cause
    assert staged.closed is True
    assert destination.closed is True
    if mismatch.startswith("successor"):
        successor_handle = adapter.relative_results[-1][2]
        assert successor_handle in adapter.successor_handles
        assert ("CloseHandle", successor_handle) in adapter.calls
    assert all(
        not str(call[2]).startswith(("C:", "\\"))
        for call in adapter.calls
        if call[0] == "NtCreateFile"
    )
    filesystem.close()



def test_windows_native_temp_requests_file_write_data() -> None:
    import ctypes

    from mochi.security.safe_fs_windows import _WindowsNativeAdapter

    captured: dict[str, int] = {}
    adapter = object.__new__(_WindowsNativeAdapter)

    def ntcreate_file(*args):
        captured["access"] = int(args[1])
        return 0

    adapter._NtCreateFile = ntcreate_file
    adapter.ntcreate_new_relative(
        ctypes.c_void_p(1), "stage.tmp"
    )

    assert (
        captured["access"] & adapter.FILE_WRITE_DATA
        == adapter.FILE_WRITE_DATA
    )



def test_windows_native_name_collision_maps_file_exists() -> None:
    import ctypes

    from mochi.security.safe_fs_windows import _WindowsNativeAdapter

    adapter = object.__new__(_WindowsNativeAdapter)
    collision = ctypes.c_long(0xC0000035).value
    adapter._NtCreateFile = lambda *args: collision

    with pytest.raises(FileExistsError):
        adapter.ntcreate_new_relative(
            ctypes.c_void_p(1), "stage.tmp"
        )



def test_windows_create_temp_retries_native_name_collision() -> None:
    from mochi.security.safe_filesystem import StagedTemp
    from mochi.security.safe_fs_windows import WindowsSafeFilesystem

    adapter = _FakeWindowsAdapter(collisions=1)
    filesystem = WindowsSafeFilesystem(
        "C:/workspace", adapter=adapter, enforce=True
    )
    destination = filesystem.prepare_target(
        "safe/note.txt", _windows_authorization()
    )

    staged = filesystem.create_temp(destination)

    assert isinstance(staged, StagedTemp)
    temp_creates = [
        call
        for call in adapter.calls
        if call[0] == "NtCreateFile"
        and "CREATE" in str(call[4])
    ]
    assert len(temp_creates) == 2
    staged.close()
    destination.close()
    filesystem.close()



def test_windows_boundary_comes_from_opened_root_handle() -> None:
    from mochi.security.safe_fs_windows import WindowsSafeFilesystem

    adapter = _FakeWindowsAdapter(
        workspace_final_path="C:/real/workspace"
    )
    filesystem = WindowsSafeFilesystem(
        "C:/alias", adapter=adapter, enforce=True
    )

    target = filesystem.prepare_target(
        "safe/note.txt", _windows_authorization()
    )

    target.close()
    filesystem.close()



def test_windows_traversal_verify_failure_closes_new_handle() -> None:
    from mochi.security.safe_filesystem import UnsafeFilesystemTarget
    from mochi.security.safe_fs_windows import WindowsSafeFilesystem

    adapter = _FakeWindowsAdapter(parent_reparse=True)
    filesystem = WindowsSafeFilesystem(
        "C:/workspace", adapter=adapter, enforce=True
    )

    with pytest.raises(UnsafeFilesystemTarget, match="reparse"):
        filesystem.prepare_target(
            "safe/note.txt", _windows_authorization()
        )

    assert set(adapter.nodes.values()) == {"workspace"}
    filesystem.close()
    assert adapter.nodes == {}



def test_windows_temp_verify_failure_closes_temp_handle() -> None:
    from mochi.security.safe_filesystem import UnsafeFilesystemTarget
    from mochi.security.safe_fs_windows import WindowsSafeFilesystem

    adapter = _FakeWindowsAdapter(temp_reparse=True)
    filesystem = WindowsSafeFilesystem(
        "C:/workspace", adapter=adapter, enforce=True
    )
    destination = filesystem.prepare_target(
        "safe/note.txt", _windows_authorization()
    )

    with pytest.raises(UnsafeFilesystemTarget, match="reparse"):
        filesystem.create_temp(destination)

    temp_handle = adapter.temp_handles[-1]
    assert temp_handle not in adapter.nodes
    assert ("CloseHandle", temp_handle) in adapter.calls
    destination.close()
    filesystem.close()



def test_windows_parent_close_failure_is_not_retried() -> None:
    from mochi.security.file_contract import FileIdentity
    from mochi.security.safe_fs_windows import WindowsSafeFilesystem

    class CloseOwnedParentFailure(_FakeWindowsAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.fail_once = True

        def close(self, handle: object) -> None:
            node = self.nodes.get(handle)
            if self.fail_once and node == "safe-original":
                self.fail_once = False
                self.failed_handle = handle
                super().close(handle)
                raise OSError("parent close failed")
            super().close(handle)

    adapter = CloseOwnedParentFailure()
    adapter.children[("safe-original", "nested")] = "nested"
    adapter.children[("nested", "note.txt")] = "note-original"
    adapter.identities["nested"] = FileIdentity(
        "windows", "10", "22", 1, False
    )
    adapter.paths["nested"] = "C:/workspace/safe/nested"
    filesystem = WindowsSafeFilesystem(
        "C:/workspace", adapter=adapter, enforce=True
    )
    authorization = _file_authorization(
        "safe/nested/note.txt",
        FileIdentity("windows", "10", "41", 1, False),
    )

    with pytest.raises(OSError, match="parent close failed"):
        filesystem.prepare_target(
            "safe/nested/note.txt", authorization
        )

    assert adapter.calls.count(
        ("CloseHandle", adapter.failed_handle)
    ) == 1
    assert set(adapter.nodes.values()) == {"workspace"}
    filesystem.close()



def test_windows_post_replace_cleanup_reports_committed_outcome() -> None:
    from mochi.security.safe_filesystem import (
        CommittedFilesystemMutationError,
    )
    from mochi.security.safe_fs_windows import WindowsSafeFilesystem

    class SuccessorCloseFailure(_FakeWindowsAdapter):
        def close(self, handle: object) -> None:
            is_successor = handle in self.successor_handles
            super().close(handle)
            if is_successor:
                raise OSError("successor close failed")

    adapter = SuccessorCloseFailure()
    filesystem = WindowsSafeFilesystem(
        "C:/workspace", adapter=adapter, enforce=True
    )
    destination = filesystem.prepare_target(
        "safe/note.txt", _windows_atomic_authorization()
    )
    filesystem.capture_metadata(destination)
    staged = filesystem.create_temp(destination)

    with pytest.raises(
        CommittedFilesystemMutationError
    ) as raised:
        filesystem.replace(staged, destination)

    outcome = raised.value
    assert outcome.committed is True
    assert outcome.phase == "operand_cleanup"
    assert isinstance(outcome.cause, OSError)
    assert outcome.__cause__ is outcome.cause
    assert staged.closed is True
    assert destination.closed is True
    filesystem.close()



def test_windows_constructor_preserves_primary_when_close_fails() -> None:
    from mochi.security.file_contract import FileIdentity
    from mochi.security.safe_filesystem import UnsafeFilesystemTarget
    from mochi.security.safe_fs_windows import WindowsSafeFilesystem

    class RootCloseFailure(_FakeWindowsAdapter):
        def close(self, handle: object) -> None:
            node = self.nodes.get(handle)
            super().close(handle)
            if node == "workspace":
                raise OSError("root cleanup failed")

    adapter = RootCloseFailure()
    adapter.identities["workspace"] = FileIdentity(
        "windows", "10", "20", 1, True
    )

    with pytest.raises(
        UnsafeFilesystemTarget, match="workspace root.*reparse"
    ) as raised:
        WindowsSafeFilesystem(
            "C:/workspace", adapter=adapter, enforce=True
        )

    assert any(
        "root cleanup failed" in note
        for note in getattr(raised.value, "__notes__", ())
    )
    assert adapter.nodes == {}



@pytest.mark.parametrize("failure", ["verification", "authorization"])
def test_windows_prepare_preserves_primary_when_close_fails(
    failure: str,
) -> None:
    from mochi.security.file_contract import FileIdentity
    from mochi.security.safe_filesystem import UnsafeFilesystemTarget
    from mochi.security.safe_fs_windows import WindowsSafeFilesystem

    class FileCloseFailure(_FakeWindowsAdapter):
        def close(self, handle: object) -> None:
            node = self.nodes.get(handle)
            super().close(handle)
            if node == "note-original":
                raise OSError("file cleanup failed")

    adapter = FileCloseFailure(
        link_count=2 if failure == "verification" else 1
    )
    filesystem = WindowsSafeFilesystem(
        "C:/workspace", adapter=adapter, enforce=True
    )
    identity = FileIdentity(
        "windows",
        "10",
        "99" if failure == "authorization" else "41",
        1,
        False,
    )
    expected = (
        "hardlink" if failure == "verification" else "base identity"
    )

    with pytest.raises(
        UnsafeFilesystemTarget, match=expected
    ) as raised:
        filesystem.prepare_target(
            "safe/note.txt",
            _file_authorization("safe/note.txt", identity),
        )

    assert any(
        "file cleanup failed" in note
        for note in getattr(raised.value, "__notes__", ())
    )
    assert set(adapter.nodes.values()) == {"workspace"}
    filesystem.close()



def test_windows_close_drains_all_targets_after_close_error() -> None:
    from mochi.security.safe_fs_windows import WindowsSafeFilesystem

    class OneCloseFailure(_FakeWindowsAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.fail_once = False

        def close(self, handle: object) -> None:
            node = self.nodes.get(handle)
            super().close(handle)
            if self.fail_once and node == "note-original":
                self.fail_once = False
                raise OSError("close failed")

    adapter = OneCloseFailure()
    filesystem = WindowsSafeFilesystem(
        "C:/workspace", adapter=adapter, enforce=True
    )
    first = filesystem.prepare_target(
        "safe/note.txt", _windows_authorization()
    )
    second = filesystem.prepare_target(
        "safe/note.txt", _windows_authorization()
    )
    staged = filesystem.create_temp(first)
    adapter.fail_once = True

    with pytest.raises(OSError, match="close failed"):
        filesystem.close()

    assert adapter.nodes == {}
    assert first.closed is True
    assert second.closed is True
    assert staged.closed is True



def test_windows_release_drains_pin_after_file_close_error() -> None:
    from mochi.security.safe_fs_windows import WindowsSafeFilesystem

    class FileCloseFailure(_FakeWindowsAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.fail_once = False

        def close(self, handle: object) -> None:
            node = self.nodes.get(handle)
            super().close(handle)
            if self.fail_once and node == "note-original":
                self.fail_once = False
                raise OSError("file close failed")

    adapter = FileCloseFailure()
    filesystem = WindowsSafeFilesystem(
        "C:/workspace", adapter=adapter, enforce=True
    )
    target = filesystem.prepare_target(
        "safe/note.txt", _windows_authorization()
    )
    adapter.fail_once = True

    with pytest.raises(OSError, match="file close failed"):
        target.close()

    assert set(adapter.nodes.values()) == {"workspace"}
    assert target.closed is True
    filesystem.close()



@pytest.mark.parametrize(
    "alias",
    ["safe\\note.txt", "SAFE/NOTE.TXT"],
)
def test_windows_rejects_duplicate_canonical_authorization_aliases(
    alias: str,
) -> None:
    from mochi.security.file_contract import FileChangeRequest, FileIdentity
    from mochi.security.safe_filesystem import UnsafeFilesystemTarget
    from mochi.security.safe_fs_windows import WindowsSafeFilesystem

    identity = FileIdentity("windows", "10", "41", 1, False)
    authorization = _file_authorization("safe/note.txt", identity)
    file_request = authorization.file_request
    assert file_request is not None
    authorization = replace(
        authorization,
        file_request=FileChangeRequest(
            entries=(
                file_request.entries[0],
                _entry(
                    entry_id="0002",
                    relative_path=alias,
                    base_identity=identity,
                ),
            ),
            patch_sha256=_sha("patch"),
        ),
    )
    adapter = _FakeWindowsAdapter()
    filesystem = WindowsSafeFilesystem(
        "C:/workspace", adapter=adapter, enforce=True
    )
    try:
        with pytest.raises(UnsafeFilesystemTarget, match="canonical"):
            filesystem.prepare_target("safe/note.txt", authorization)
    finally:
        filesystem.close()

    assert adapter.nodes == {}
