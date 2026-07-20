from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from mochi.security.file_contract import (
        FileIdentity,
    )

from tests.security.safe_filesystem._support import (
    _authorization,
    _exec_authorization,
    _FakePosixAdapter,
    _file_authorization,
    _sha,
)


def test_facade_create_temp_returns_backend_issued_staged_capability() -> None:
    from mochi.security.file_contract import FileIdentity
    from mochi.security.safe_filesystem import (
        AuthorizedFileBinding,
        SafeFilesystem,
        SafeTarget,
        StagedTemp,
    )
    from mochi.security.safe_fs_posix import PosixSafeFilesystem

    adapter = _FakePosixAdapter()
    backend = PosixSafeFilesystem("/workspace", adapter=adapter)
    facade = SafeFilesystem.__new__(SafeFilesystem)
    facade._backend = backend
    authorization = _authorization()
    destination = facade.prepare_target(
        "safe/note.txt", authorization
    )

    staged = facade.create_temp(destination)

    assert isinstance(staged, StagedTemp)
    assert not isinstance(staged, (tuple, SafeTarget))
    assert staged.binding == AuthorizedFileBinding(
        entry_id="0001",
        canonical_relative_path="safe/note.txt",
        operation="update",
        base_identity=FileIdentity(
            "posix", "1", "41", 1, False
        ),
        base_sha256=_sha("base-content"),
        after_sha256=_sha("after-content"),
        base_metadata_sha256=_sha("base-metadata"),
        after_metadata_sha256=_sha("after-metadata"),
        authorization_digest=staged.authorization_digest,
    )
    with pytest.raises(AttributeError, match="immutable"):
        staged.basename = "forged"
    staged.close()
    destination.close()
    facade.close()



def test_exec_authorization_cannot_prepare_file_target() -> None:
    from mochi.security.safe_filesystem import UnsafeFilesystemTarget
    from mochi.security.safe_fs_posix import PosixSafeFilesystem

    filesystem = PosixSafeFilesystem(
        "/workspace", adapter=_FakePosixAdapter()
    )
    try:
        with pytest.raises(UnsafeFilesystemTarget, match="file_change"):
            filesystem.prepare_target(
                "safe/note.txt", _exec_authorization()
            )
    finally:
        filesystem.close()



def test_prepare_target_rejects_unlisted_path() -> None:
    from mochi.security.file_contract import FileIdentity
    from mochi.security.safe_filesystem import UnsafeFilesystemTarget
    from mochi.security.safe_fs_posix import PosixSafeFilesystem

    authorization = _file_authorization(
        "safe/other.txt",
        FileIdentity("posix", "1", "41", 1, False),
    )
    filesystem = PosixSafeFilesystem(
        "/workspace", adapter=_FakePosixAdapter()
    )
    try:
        with pytest.raises(UnsafeFilesystemTarget, match="not authorized"):
            filesystem.prepare_target("safe/note.txt", authorization)
    finally:
        filesystem.close()



def test_prepare_target_rejects_base_identity_mismatch() -> None:
    from mochi.security.file_contract import FileIdentity
    from mochi.security.safe_filesystem import UnsafeFilesystemTarget
    from mochi.security.safe_fs_posix import PosixSafeFilesystem

    authorization = _file_authorization(
        "safe/note.txt",
        FileIdentity("posix", "1", "99", 1, False),
    )
    filesystem = PosixSafeFilesystem(
        "/workspace", adapter=_FakePosixAdapter()
    )
    try:
        with pytest.raises(UnsafeFilesystemTarget, match="base identity"):
            filesystem.prepare_target("safe/note.txt", authorization)
    finally:
        filesystem.close()



def test_prepare_target_accepts_canonical_path_and_captured_identity() -> None:
    from mochi.security.file_contract import FileIdentity
    from mochi.security.safe_fs_posix import PosixSafeFilesystem

    identity = FileIdentity("posix", "1", "41", 1, False)
    authorization = _file_authorization("safe/./note.txt", identity)
    filesystem = PosixSafeFilesystem(
        "/workspace", adapter=_FakePosixAdapter()
    )
    try:
        target = filesystem.prepare_target(
            "safe/note.txt", authorization
        )
        assert target.identity == identity
        target.close()
    finally:
        filesystem.close()



def test_unlink_requires_delete_authorization() -> None:
    from mochi.security.safe_filesystem import UnsafeFilesystemTarget
    from mochi.security.safe_fs_posix import PosixSafeFilesystem

    filesystem = PosixSafeFilesystem(
        "/workspace", adapter=_FakePosixAdapter()
    )
    try:
        target = filesystem.prepare_target(
            "safe/note.txt", _authorization()
        )
        with pytest.raises(UnsafeFilesystemTarget, match="delete"):
            filesystem.unlink(target)
    finally:
        filesystem.close()



def test_replace_rejects_cross_authorization_operands() -> None:
    from mochi.security.safe_filesystem import UnsafeFilesystemTarget
    from mochi.security.safe_fs_posix import PosixSafeFilesystem

    filesystem = PosixSafeFilesystem(
        "/workspace", adapter=_FakePosixAdapter()
    )
    try:
        source_destination = filesystem.prepare_target(
            "safe/note.txt", _authorization()
        )
        source = filesystem.create_temp(source_destination)
        destination = filesystem.prepare_target(
            "safe/note.txt",
            replace(_authorization(), policy_version="policy-v2"),
        )
        with pytest.raises(UnsafeFilesystemTarget, match="authorization"):
            filesystem.replace(source, destination)
    finally:
        filesystem.close()



def test_create_temp_requires_update_or_rename() -> None:
    from mochi.security.file_contract import FileIdentity
    from mochi.security.safe_filesystem import UnsafeFilesystemTarget
    from mochi.security.safe_fs_posix import PosixSafeFilesystem

    authorization = _file_authorization(
        "safe/note.txt",
        FileIdentity("posix", "1", "41", 1, False),
        "delete",
    )
    filesystem = PosixSafeFilesystem(
        "/workspace", adapter=_FakePosixAdapter()
    )
    try:
        destination = filesystem.prepare_target(
            "safe/note.txt", authorization
        )
        with pytest.raises(
            UnsafeFilesystemTarget, match="update or rename"
        ):
            filesystem.create_temp(destination)
    finally:
        filesystem.close()


def test_facade_replace_discards_backend_specific_successor_result() -> None:
    from mochi.security.file_contract import FileIdentity
    from mochi.security.safe_filesystem import (
        SafeFilesystem,
        SafeTarget,
        StagedTemp,
    )

    successor = FileIdentity("posix", "1", "77", 1, False)

    class _Backend:
        def replace(self, source: object, destination: object) -> FileIdentity:
            del source, destination
            return successor

    facade = SafeFilesystem.__new__(SafeFilesystem)
    facade._backend = _Backend()
    source = object.__new__(StagedTemp)
    destination = object.__new__(SafeTarget)

    assert facade.replace(source, destination) is None
