from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Literal

import pytest

if TYPE_CHECKING:
    pass

from tests.security.safe_filesystem._support import (
    _authorization,
    _FakePosixAdapter,
    _file_authorization,
    _posix_atomic_authorization,
    _stage_posix_atomic_temp,
    _two_file_authorization,
)


def test_posix_pinned_parent_survives_symlink_rebind() -> None:
    from mochi.security.file_contract import FileIdentity
    from mochi.security.safe_fs_posix import PosixSafeFilesystem

    adapter = _FakePosixAdapter()
    filesystem = PosixSafeFilesystem("/workspace", adapter=adapter)
    target = filesystem.prepare_target(
        "safe/note.txt",
        _file_authorization(
            "safe/note.txt",
            FileIdentity("posix", "1", "41", 1, False),
            "delete",
        ),
    )
    adapter.children[("workspace", "safe")] = "outside"
    filesystem.unlink(target)

    assert adapter.unlinked == [("safe-original", "note.txt")]
    assert adapter.open_calls[0] == ("/workspace", None)
    assert all(not path.startswith("/") for path, fd in adapter.open_calls[1:] if fd is not None)



def test_posix_rejects_hardlinked_target_and_parent_traversal() -> None:
    from mochi.security.safe_filesystem import UnsafeFilesystemTarget
    from mochi.security.safe_fs_posix import PosixSafeFilesystem

    with pytest.raises(UnsafeFilesystemTarget, match="hardlink"):
        PosixSafeFilesystem("/workspace", adapter=_FakePosixAdapter(link_count=2)).prepare_target(
            "safe/note.txt", _authorization()
        )
    with pytest.raises(UnsafeFilesystemTarget, match="traversal"):
        PosixSafeFilesystem("/workspace", adapter=_FakePosixAdapter()).prepare_target(
            "safe/../note.txt", _authorization()
        )


def test_posix_rejects_wrong_workspace_identity() -> None:
    from mochi.security.file_contract import FileIdentity
    from mochi.security.safe_filesystem import UnsafeFilesystemTarget
    from mochi.security.safe_fs_posix import PosixSafeFilesystem

    envelope = _authorization()
    wrong = replace(
        envelope,
        context=replace(
            envelope.context,
            workspace_identity=FileIdentity("posix", "10", "wrong", 1, False),
        ),
    )
    filesystem = PosixSafeFilesystem(
        "/workspace", adapter=_FakePosixAdapter()
    )
    with pytest.raises(UnsafeFilesystemTarget, match="workspace identity"):
        filesystem.prepare_target("safe/note.txt", wrong)


def test_safe_target_cannot_be_forged_or_mutated() -> None:
    from mochi.security.safe_filesystem import SafeTarget
    from mochi.security.safe_fs_posix import PosixSafeFilesystem

    filesystem = PosixSafeFilesystem(
        "/workspace", adapter=_FakePosixAdapter()
    )
    target = filesystem.prepare_target("safe/note.txt", _authorization())

    from mochi.security.safe_filesystem import UnsafeFilesystemTarget

    forged = SafeTarget._create(
        basename=target.basename,
        identity=target.identity,
        authorization_digest=target.authorization_digest,
        owner=filesystem,
        parent=target._parent,
    )
    with pytest.raises(UnsafeFilesystemTarget, match="issued capability"):
        filesystem.unlink(forged)

    with pytest.raises(AttributeError):
        target.basename = "outside.txt"
    with pytest.raises(TypeError, match="cannot be constructed"):
        SafeTarget(
            basename=target.basename,
            identity=target.identity,
            authorization_digest=target.authorization_digest,
            _owner=filesystem,
            _parent=target._parent,
        )


@pytest.mark.parametrize("operation", ["add", "delete"])
def test_posix_create_temp_requires_update_or_rename_authorization(
    operation: Literal["add", "delete"],
) -> None:
    from mochi.security.file_contract import FileIdentity
    from mochi.security.safe_filesystem import UnsafeFilesystemTarget
    from mochi.security.safe_fs_posix import PosixSafeFilesystem

    adapter = _FakePosixAdapter()
    filesystem = PosixSafeFilesystem("/workspace", adapter=adapter)
    destination = filesystem.prepare_target(
        "safe/note.txt",
        _file_authorization(
            "safe/note.txt",
            FileIdentity("posix", "1", "41", 1, False),
            operation,
        ),
    )
    opens_before = len(adapter.open_calls)

    with pytest.raises(
        UnsafeFilesystemTarget, match="update or rename"
    ):
        filesystem.create_temp(destination)

    assert len(adapter.open_calls) == opens_before
    destination.close()
    filesystem.close()



def test_posix_staged_temp_is_identity_checked_and_replaced_relative() -> None:
    from mochi.security.safe_filesystem import StagedTemp
    from mochi.security.safe_fs_posix import PosixSafeFilesystem

    adapter = _FakePosixAdapter()
    filesystem = PosixSafeFilesystem("/workspace", adapter=adapter)
    destination = filesystem.prepare_target(
        "safe/note.txt", _posix_atomic_authorization()
    )

    staged = _stage_posix_atomic_temp(filesystem, destination)

    assert isinstance(staged, StagedTemp)
    assert staged.identity.file_id == "77"
    assert adapter.fstat_calls[-1] in adapter.fd_nodes
    assert adapter.stat_calls[-1][0] == staged.basename
    assert adapter.stat_calls[-1][2] is False
    filesystem.replace(staged, destination)

    assert adapter.replace_calls == [
        (
            "safe-original",
            staged.basename,
            "safe-original",
            "note.txt",
        )
    ]
    assert staged.closed is True
    assert destination.closed is True
    filesystem.close()
    assert adapter.fd_nodes == {}



def test_replace_rejects_non_staged_source_before_posix_syscall() -> None:
    from mochi.security.safe_filesystem import UnsafeFilesystemTarget
    from mochi.security.safe_fs_posix import PosixSafeFilesystem

    adapter = _FakePosixAdapter()
    filesystem = PosixSafeFilesystem("/workspace", adapter=adapter)
    destination = filesystem.prepare_target(
        "safe/note.txt", _authorization()
    )

    with pytest.raises(
        (TypeError, UnsafeFilesystemTarget), match="StagedTemp"
    ):
        filesystem.replace(destination, destination)

    assert adapter.replace_calls == []
    destination.close()
    filesystem.close()



@pytest.mark.parametrize("mismatch", ["digest", "entry"])
def test_posix_replace_rejects_staged_binding_mismatch_before_syscall(
    mismatch: Literal["digest", "entry"],
) -> None:
    from mochi.security.safe_filesystem import UnsafeFilesystemTarget
    from mochi.security.safe_fs_posix import PosixSafeFilesystem

    adapter = _FakePosixAdapter()
    adapter.children[("safe-original", "other.txt")] = (
        "other-original"
    )
    filesystem = PosixSafeFilesystem("/workspace", adapter=adapter)
    if mismatch == "digest":
        source_authorization = _authorization()
        destination_authorization = replace(
            _authorization(), policy_version="policy-v2"
        )
        destination_path = "safe/note.txt"
    else:
        source_authorization = _two_file_authorization("posix")
        destination_authorization = source_authorization
        destination_path = "safe/other.txt"

    source_destination = filesystem.prepare_target(
        "safe/note.txt", source_authorization
    )
    staged = filesystem.create_temp(source_destination)
    destination = filesystem.prepare_target(
        destination_path, destination_authorization
    )

    with pytest.raises(
        UnsafeFilesystemTarget,
        match="authorization binding",
    ):
        filesystem.replace(staged, destination)

    assert adapter.replace_calls == []
    staged.close()
    source_destination.close()
    destination.close()
    filesystem.close()



def test_posix_root_fstat_failure_closes_root_fd() -> None:
    from mochi.security.safe_fs_posix import PosixSafeFilesystem

    class RootFstatFailure(_FakePosixAdapter):
        def fstat(self, fd: int) -> SimpleNamespace:
            super().fstat(fd)
            raise OSError("root fstat failed")

    adapter = RootFstatFailure()

    with pytest.raises(OSError, match="root fstat failed"):
        PosixSafeFilesystem("/workspace", adapter=adapter)

    assert adapter.fd_nodes == {}
    assert adapter.close_calls == [100]



def test_posix_temp_identity_failure_closes_temp_and_duplicated_parent() -> None:
    from mochi.security.safe_filesystem import UnsafeFilesystemTarget
    from mochi.security.safe_fs_posix import PosixSafeFilesystem

    class TempIdentityMismatch(_FakePosixAdapter):
        def stat(
            self,
            path: str,
            *,
            dir_fd: int,
            follow_symlinks: bool,
        ) -> SimpleNamespace:
            info = super().stat(
                path,
                dir_fd=dir_fd,
                follow_symlinks=follow_symlinks,
            )
            if path.startswith(".mochi-"):
                values = vars(info).copy()
                values["st_ino"] = 78
                return SimpleNamespace(**values)
            return info

    adapter = TempIdentityMismatch()
    filesystem = PosixSafeFilesystem("/workspace", adapter=adapter)
    destination = filesystem.prepare_target(
        "safe/note.txt", _authorization()
    )
    baseline = dict(adapter.fd_nodes)

    with pytest.raises(
        UnsafeFilesystemTarget, match="temp.*identity"
    ):
        filesystem.create_temp(destination)

    assert adapter.fd_nodes == baseline
    destination.close()
    filesystem.close()



def test_posix_open_parent_close_failure_releases_both_owned_fds() -> None:
    from mochi.security.safe_fs_posix import PosixSafeFilesystem

    class CloseTransitionFailure(_FakePosixAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.fail_transition_once = True

        def close(self, fd: int) -> None:
            if (
                self.fail_transition_once
                and self.fd_nodes.get(fd) == "workspace"
                and fd != 100
            ):
                self.fail_transition_once = False
                self.failed_fd = fd
                super().close(fd)
                raise OSError("transition close failed")
            super().close(fd)

    adapter = CloseTransitionFailure()
    filesystem = PosixSafeFilesystem("/workspace", adapter=adapter)

    with pytest.raises(OSError, match="transition close failed"):
        filesystem.prepare_target(
            "safe/note.txt", _authorization()
        )

    assert adapter.fd_nodes == {100: "workspace"}
    assert adapter.close_calls.count(adapter.failed_fd) == 1
    filesystem.close()



def test_posix_post_replace_cleanup_reports_committed_outcome() -> None:
    from mochi.security.safe_filesystem import (
        CommittedFilesystemMutationError,
    )
    from mochi.security.safe_fs_posix import PosixSafeFilesystem

    class TempCloseFailure(_FakePosixAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.armed = False

        def close(self, fd: int) -> None:
            node = self.fd_nodes.get(fd)
            super().close(fd)
            if self.armed and node and node.startswith("temp:"):
                self.armed = False
                raise OSError("post-replace temp close failed")

    adapter = TempCloseFailure()
    filesystem = PosixSafeFilesystem("/workspace", adapter=adapter)
    destination = filesystem.prepare_target(
        "safe/note.txt", _posix_atomic_authorization()
    )
    staged = _stage_posix_atomic_temp(filesystem, destination)
    adapter.armed = True

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



def test_posix_precommit_replace_failure_keeps_capabilities_live() -> None:
    from mochi.security.safe_filesystem import (
        CommittedFilesystemMutationError,
    )
    from mochi.security.safe_fs_posix import PosixSafeFilesystem

    class ReplaceFailure(_FakePosixAdapter):
        def replace(self, *args, **kwargs) -> None:
            raise OSError("replace did not commit")

    filesystem = PosixSafeFilesystem(
        "/workspace", adapter=ReplaceFailure()
    )
    destination = filesystem.prepare_target(
        "safe/note.txt", _posix_atomic_authorization()
    )
    staged = _stage_posix_atomic_temp(filesystem, destination)

    with pytest.raises(OSError, match="did not commit") as raised:
        filesystem.replace(staged, destination)

    assert not isinstance(
        raised.value, CommittedFilesystemMutationError
    )
    assert staged.closed is False
    assert destination.closed is False
    filesystem.close()



def test_posix_close_drains_all_targets_after_close_error() -> None:
    from mochi.security.safe_fs_posix import PosixSafeFilesystem

    class OneCloseFailure(_FakePosixAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.fail_once = False

        def close(self, fd: int) -> None:
            node = self.fd_nodes.get(fd)
            super().close(fd)
            if self.fail_once and node == "safe-original":
                self.fail_once = False
                raise OSError("close failed")

    adapter = OneCloseFailure()
    filesystem = PosixSafeFilesystem("/workspace", adapter=adapter)
    first = filesystem.prepare_target(
        "safe/note.txt", _authorization()
    )
    second = filesystem.prepare_target(
        "safe/note.txt", _authorization()
    )
    staged = filesystem.create_temp(first)
    adapter.fail_once = True

    with pytest.raises(OSError, match="close failed"):
        filesystem.close()

    assert adapter.fd_nodes == {}
    assert first.closed is True
    assert second.closed is True
    assert staged.closed is True
