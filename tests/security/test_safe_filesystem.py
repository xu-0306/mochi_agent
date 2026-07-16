from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest


def _entry(*, entry_id: str = "0002", relative_path: str = "b.txt"):
    from mochi.security.file_contract import ChangeEntry, FileIdentity

    return ChangeEntry(
        entry_id=entry_id,
        relative_path=relative_path,
        operation="update",
        base_sha256="base-sha256",
        after_sha256="after-sha256",
        base_identity=FileIdentity("posix", "1", "2", 1, False),
        before_blob_id="before-blob",
        after_blob_id="after-blob",
        mode_before=0o644,
        mode_after=0o600,
        base_metadata_sha256="base-meta",
        after_metadata_sha256="after-meta",
        rename_source=None,
        dependency_group="group-a",
    )


def _authorization():
    from mochi.security.file_contract import (
        AuthorizationContext,
        AuthorizationEnvelope,
        FileChangeRequest,
        FileIdentity,
    )

    return AuthorizationEnvelope(
        schema_version=1,
        kind="file_change",
        context=AuthorizationContext(
            requester_id="requester-1",
            session_id="session-1",
            task_id="task-1",
            workspace_root="workspace",
            workspace_identity=FileIdentity("posix", "10", "20", 1, False),
        ),
        policy_version="policy-v1",
        file_request=FileChangeRequest(
            entries=(_entry(), _entry(entry_id="0001", relative_path="a.txt")),
            patch_sha256="patch-sha256",
        ),
        exec_request=None,
    )


def test_contract_roundtrips_complete_file_request_and_identity() -> None:
    from mochi.security.file_contract import AuthorizationEnvelope, FileIdentity

    envelope = _authorization()
    assert AuthorizationEnvelope.from_dict(envelope.to_dict()) == envelope
    identity = FileIdentity("windows", "volume", "file", 1, True)
    assert FileIdentity.from_dict(identity.to_dict()) == identity
    assert identity.is_reparse_point is True


def test_file_request_digest_is_deterministic_and_binds_authorization() -> None:
    from mochi.security.file_contract import (
        FileChangeRequest,
        FileIdentity,
        authorization_request_digest,
    )

    envelope = _authorization()
    reversed_request = FileChangeRequest(
        entries=tuple(reversed(envelope.file_request.entries)),
        patch_sha256=envelope.file_request.patch_sha256,
    )
    assert authorization_request_digest(envelope) == authorization_request_digest(
        replace(envelope, file_request=reversed_request)
    )

    changed = [
        replace(envelope, context=replace(envelope.context, requester_id="requester-2")),
        replace(envelope, context=replace(envelope.context, session_id="session-2")),
        replace(envelope, context=replace(envelope.context, task_id=None)),
        replace(envelope, context=replace(envelope.context, workspace_root="other")),
        replace(envelope, context=replace(envelope.context, workspace_identity=FileIdentity("posix", "10", "21", 1, False))),
        replace(envelope, policy_version="policy-v2"),
        replace(
            envelope,
            file_request=replace(envelope.file_request, patch_sha256="other-patch"),
        ),
    ]
    original = authorization_request_digest(envelope)
    assert all(authorization_request_digest(item) != original for item in changed)


def test_envelope_requires_exactly_one_matching_request() -> None:
    from mochi.security.file_contract import (
        EnvVarHash,
        ExecRequest,
        ResourceLimits,
    )

    envelope = _authorization()
    exec_request = ExecRequest(
        command_utf8_sha256="command-sha256",
        shell="powershell",
        executable="pwsh.exe",
        argv=("-NoProfile", "-Command", "Get-Date"),
        resolved_cwd="workspace",
        env=(EnvVarHash("PATH", "path-value-sha256"),),
        network_policy="deny",
        resource_limits=ResourceLimits(30, 1024, 4096),
        requested_escalation="none",
        sandbox_backend="windows-native",
        sandbox_capability_plan_digest="sandbox-plan-sha256",
    )
    with pytest.raises(ValueError, match="exactly one"):
        replace(envelope, exec_request=exec_request)
    with pytest.raises(ValueError, match="kind"):
        replace(envelope, kind="exec")


def test_contract_rejects_unknown_fields_at_every_model_layer() -> None:
    from mochi.security.file_contract import (
        ChangeManifest,
        EnvVarHash,
        ExecRequest,
        FileIdentity,
        ResourceLimits,
    )

    objects = [
        FileIdentity("posix", "1", "2", 1, False),
        _authorization().context,
        _entry(),
        _authorization().file_request,
        EnvVarHash("PATH", "hash"),
        ResourceLimits(30, 1024, 4096),
        ExecRequest(
            "command", None, "tool", (), "workspace", (), "deny",
            ResourceLimits(30, 1024, 4096), "none", "native", "plan"
        ),
        _authorization(),
        ChangeManifest(
            version=1,
            change_set_id="change-set",
            workspace_root="workspace",
            workspace_identity=FileIdentity("posix", "10", "20", 1, False),
            tool_name="write_file",
            intent="mutate",
            entries=(_entry(),),
            patch_sha256="patch",
            policy_version="policy-v1",
            created_at="2026-01-01T00:00:00Z",
            expires_at="2026-01-01T01:00:00Z",
            request_digest="request",
            ui_metadata={"label": "preview"},
        ),
    ]
    for obj in objects:
        payload = obj.to_dict()
        payload["unknown"] = "fail-closed"
        with pytest.raises(ValueError, match="unknown"):
            type(obj).from_dict(payload)


def test_canonical_json_rejects_ambiguous_non_json_types() -> None:
    from mochi.security.file_contract import canonical_json

    values = (1.25, float("nan"), Path("relative"), datetime(2026, 1, 1), {1: "x"})
    for value in values:
        with pytest.raises(TypeError):
            canonical_json({"value": value})


def test_manifest_projection_only_excludes_explicit_top_level_volatile_fields() -> None:
    from mochi.security.file_contract import (
        ChangeManifest,
        FileIdentity,
        manifest_digest_projection,
    )

    manifest = ChangeManifest(
        version=1,
        change_set_id="volatile",
        workspace_root="workspace",
        workspace_identity=FileIdentity("posix", "10", "20", 1, False),
        tool_name="write_file",
        intent="mutate",
        entries=(_entry(),),
        patch_sha256="patch",
        policy_version="policy-v1",
        created_at="created",
        expires_at="expires",
        request_digest="request",
        ui_metadata={"created_at": "nested-security-relevant"},
    )
    projection = manifest_digest_projection(manifest)

    assert "change_set_id" not in projection
    assert "created_at" not in projection
    assert "expires_at" not in projection
    assert "request_digest" not in projection
    assert "ui_metadata" not in projection
    assert manifest.ui_metadata["created_at"] == "nested-security-relevant"


def test_preview_idempotency_key_is_context_scoped_composite() -> None:
    from mochi.security.file_contract import preview_idempotency_key

    envelope = _authorization()
    key = preview_idempotency_key(envelope)
    assert key == preview_idempotency_key(envelope)
    assert key != preview_idempotency_key(
        replace(envelope, context=replace(envelope.context, session_id="different"))
    )
    assert key != preview_idempotency_key(replace(envelope, policy_version="different"))
    assert key != preview_idempotency_key(
        replace(envelope, file_request=replace(envelope.file_request, patch_sha256="different"))
    )


class _FakePosixAdapter:
    O_RDONLY = 0
    O_WRONLY = 1
    O_CREAT = 0x40
    O_EXCL = 0x80
    O_DIRECTORY = 0x10000
    O_NOFOLLOW = 0x20000

    def __init__(self, *, link_count: int = 1) -> None:
        self.link_count = link_count
        self.next_fd = 100
        self.fd_nodes: dict[int, str] = {}
        self.children: dict[tuple[str, str], str] = {
            ("workspace", "safe"): "safe-original",
            ("safe-original", "note.txt"): "note-original",
            ("outside", "note.txt"): "outside-note",
        }
        self.unlinked: list[tuple[str, str]] = []
        self.open_calls: list[tuple[str, int | None]] = []
        self.replace_calls: list[tuple[str, str, str, str]] = []

    def open(
        self,
        path: str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        del mode
        self.open_calls.append((path, dir_fd))
        if dir_fd is None:
            node = "workspace"
        else:
            parent = self.fd_nodes[dir_fd]
            key = (parent, path)
            if key not in self.children and flags & self.O_CREAT:
                self.children[key] = f"temp:{path}"
            node = self.children[key]
        fd = self.next_fd
        self.next_fd += 1
        self.fd_nodes[fd] = node
        return fd

    def dup(self, fd: int) -> int:
        duplicate = self.next_fd
        self.next_fd += 1
        self.fd_nodes[duplicate] = self.fd_nodes[fd]
        return duplicate

    def close(self, fd: int) -> None:
        self.fd_nodes.pop(fd, None)

    def fstat(self, fd: int) -> SimpleNamespace:
        node = self.fd_nodes[fd]
        return SimpleNamespace(
            st_mode=0o040700,
            st_dev=10,
            st_ino=20 if node == "workspace" else 21,
            st_nlink=1,
        )

    def stat(
        self,
        path: str,
        *,
        dir_fd: int,
        follow_symlinks: bool,
    ) -> SimpleNamespace:
        assert follow_symlinks is False
        node = self.children[(self.fd_nodes[dir_fd], path)]
        return SimpleNamespace(
            st_mode=0o100600,
            st_dev=1,
            st_ino={"note-original": 41, "outside-note": 99}.get(
                node, 77
            ),
            st_nlink=self.link_count,
        )

    def unlink(self, path: str, *, dir_fd: int) -> None:
        self.unlinked.append((self.fd_nodes[dir_fd], path))

    def replace(
        self,
        src: str,
        dst: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
    ) -> None:
        self.replace_calls.append(
            (
                self.fd_nodes[src_dir_fd],
                src,
                self.fd_nodes[dst_dir_fd],
                dst,
            )
        )

def test_posix_pinned_parent_survives_symlink_rebind() -> None:
    from mochi.security.safe_fs_posix import PosixSafeFilesystem

    adapter = _FakePosixAdapter()
    filesystem = PosixSafeFilesystem("/workspace", adapter=adapter)
    target = filesystem.prepare_target("safe/note.txt", _authorization())
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

def _windows_authorization():
    from mochi.security.file_contract import FileIdentity

    envelope = _authorization()
    return replace(
        envelope,
        context=replace(
            envelope.context,
            workspace_identity=FileIdentity("windows", "10", "20", 1, False),
        ),
    )

class _FakeWindowsAdapter:
    def __init__(
        self,
        *,
        available: bool = True,
        link_count: int = 1,
        parent_reparse: bool = False,
        root_file_id: str = "20",
    ) -> None:
        from mochi.security.file_contract import FileIdentity

        self.available = available
        self.calls: list[tuple[object, ...]] = []
        self.nodes: dict[object, str] = {}
        self.children = {
            ("workspace", "safe"): "safe-original",
            ("safe-original", "note.txt"): "note-original",
            ("outside", "note.txt"): "outside-note",
        }
        self.identities = {
            "workspace": FileIdentity("windows", "10", root_file_id, 1, False),
            "safe-original": FileIdentity("windows", "10", "21", 1, parent_reparse),
            "outside": FileIdentity("windows", "99", "21", 1, False),
            "note-original": FileIdentity("windows", "10", "41", link_count, False),
            "outside-note": FileIdentity("windows", "99", "41", 1, False),
        }

    def createfile_workspace(self, path: str) -> object:
        self.calls.append(
            ("CreateFileW", path, "OPEN_REPARSE_POINT|BACKUP_SEMANTICS")
        )
        handle = object()
        self.nodes[handle] = "workspace"
        return handle

    def ntcreate_relative(
        self, root: object, basename: str, *, directory: bool
    ) -> object:
        self.calls.append(
            ("NtCreateFile", root, basename, directory, "OPEN_REPARSE_POINT")
        )
        handle = object()
        self.nodes[handle] = self.children[(self.nodes[root], basename)]
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
                "CREATE|OPEN_REPARSE_POINT",
            )
        )
        node = f"temp:{basename}"
        self.children[(self.nodes[root], basename)] = node
        self.identities[node] = FileIdentity(
            "windows", "10", "temp", 1, False
        )
        handle = object()
        self.nodes[handle] = node
        return handle

    def final_path(self, handle: object) -> str:
        self.calls.append(("GetFinalPathNameByHandleW", handle))
        return "C:/workspace/" + self.nodes[handle]

    def identity(self, handle: object):
        self.calls.append(("GetFileInformationByHandle", handle))
        return self.identities[self.nodes[handle]]

    def ntset_unlink(self, handle: object) -> None:
        self.calls.append(
            ("NtSetInformationFile", handle, "FileDispositionInformation")
        )

    def ntset_replace(
        self, handle: object, root: object, basename: str
    ) -> None:
        self.calls.append(
            ("NtSetInformationFile", handle, "FileRenameInformation", root, basename)
        )

    def close(self, handle: object) -> None:
        self.calls.append(("CloseHandle", handle))
        self.nodes.pop(handle, None)


def test_windows_pins_handles_and_survives_junction_rebind() -> None:
    from mochi.security.safe_fs_windows import WindowsSafeFilesystem

    adapter = _FakeWindowsAdapter()
    filesystem = WindowsSafeFilesystem(
        "C:/workspace", adapter=adapter, enforce=True
    )
    target = filesystem.prepare_target("safe/note.txt", _windows_authorization())
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

def test_windows_native_uses_file_id_info_and_no_absolute_mutation_fallback() -> None:
    source = Path("mochi/security/safe_fs_windows.py").read_text("utf-8")

    assert "GetFileInformationByHandleEx" in source
    assert "FileIdInfo" in source
    assert "NtCreateFile" in source
    assert "RootDirectory" in source
    assert "NtSetInformationFile" in source

def test_change_manifest_ui_metadata_is_immutable() -> None:
    from mochi.security.file_contract import ChangeManifest

    manifest = ChangeManifest(
        version=1,
        change_set_id="c",
        workspace_root="workspace",
        workspace_identity=_authorization().context.workspace_identity,
        tool_name="write_file",
        intent="mutate",
        entries=(_entry(),),
        patch_sha256="patch",
        policy_version="p",
        created_at="created",
        expires_at="expires",
        request_digest="request",
        ui_metadata={"label": "preview"},
    )

    with pytest.raises(TypeError):
        manifest.ui_metadata["label"] = "changed"

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
    finally:
        adapter.close(handle)

    authorization = replace(
        _windows_authorization(),
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

def _exec_authorization():
    from mochi.security.file_contract import (
        AuthorizationContext,
        AuthorizationEnvelope,
        EnvVarHash,
        ExecRequest,
        FileIdentity,
        ResourceLimits,
    )

    return AuthorizationEnvelope(
        schema_version=1,
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
            command_utf8_sha256="command-sha256",
            shell="powershell",
            executable="pwsh.exe",
            argv=("-NoProfile", "-Command", "Get-Date"),
            resolved_cwd="C:/workspace",
            env=(EnvVarHash("PATH", "path-value-sha256"),),
            network_policy="deny",
            resource_limits=ResourceLimits(30, 1024, 4096),
            requested_escalation="none",
            sandbox_backend="windows-native",
            sandbox_capability_plan_digest="sandbox-plan-sha256",
        ),
    )


def test_exec_digest_binds_every_authorization_and_execution_field() -> None:
    from mochi.security.file_contract import (
        EnvVarHash,
        FileIdentity,
        ResourceLimits,
        authorization_request_digest,
    )

    envelope = _exec_authorization()
    request = envelope.exec_request
    assert request is not None
    changed = [
        replace(
            envelope,
            context=replace(
                envelope.context, requester_id="requester-2"
            ),
        ),
        replace(
            envelope,
            context=replace(
                envelope.context, session_id="session-2"
            ),
        ),
        replace(
            envelope,
            context=replace(envelope.context, task_id=None),
        ),
        replace(
            envelope,
            context=replace(
                envelope.context, workspace_root="C:/other"
            ),
        ),
        replace(
            envelope,
            context=replace(
                envelope.context,
                workspace_identity=FileIdentity(
                    "windows", "10", "21", 1, False
                ),
            ),
        ),
        replace(envelope, policy_version="policy-v2"),
        replace(
            envelope,
            exec_request=replace(
                request, command_utf8_sha256="other-command"
            ),
        ),
        replace(
            envelope,
            exec_request=replace(request, shell=None),
        ),
        replace(
            envelope,
            exec_request=replace(request, executable="other.exe"),
        ),
        replace(
            envelope,
            exec_request=replace(request, argv=("changed",)),
        ),
        replace(
            envelope,
            exec_request=replace(
                request, resolved_cwd="C:/workspace/other"
            ),
        ),
        replace(
            envelope,
            exec_request=replace(
                request,
                env=(EnvVarHash("PATH", "other-value-sha256"),),
            ),
        ),
        replace(
            envelope,
            exec_request=replace(request, network_policy="allow"),
        ),
        replace(
            envelope,
            exec_request=replace(
                request,
                resource_limits=ResourceLimits(31, 1024, 4096),
            ),
        ),
        replace(
            envelope,
            exec_request=replace(
                request, requested_escalation="require_escalated"
            ),
        ),
        replace(
            envelope,
            exec_request=replace(
                request, sandbox_backend="other-backend"
            ),
        ),
        replace(
            envelope,
            exec_request=replace(
                request,
                sandbox_capability_plan_digest="other-plan",
            ),
        ),
    ]
    original = authorization_request_digest(envelope)

    assert all(
        authorization_request_digest(item) != original
        for item in changed
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

def test_posix_temp_and_replace_use_only_pinned_parent_and_basenames() -> None:
    from mochi.security.safe_fs_posix import PosixSafeFilesystem

    adapter = _FakePosixAdapter()
    filesystem = PosixSafeFilesystem(
        "/workspace", adapter=adapter
    )
    target = filesystem.prepare_target(
        "safe/note.txt", _authorization()
    )
    temp_name, temp_fd = filesystem.create_temp(target)
    adapter.close(temp_fd)
    target.close()

    assert not temp_name.startswith("/")
    assert adapter.open_calls[-1][1] is not None

    source = filesystem.prepare_target(
        "safe/note.txt", _authorization()
    )
    destination = filesystem.prepare_target(
        "safe/note.txt", _authorization()
    )
    filesystem.replace(source, destination)

    assert adapter.replace_calls == [
        (
            "safe-original",
            "note.txt",
            "safe-original",
            "note.txt",
        )
    ]
    filesystem.close()


def test_windows_temp_and_replace_use_only_pinned_handles_and_basenames() -> None:
    from mochi.security.safe_fs_windows import WindowsSafeFilesystem

    adapter = _FakeWindowsAdapter()
    filesystem = WindowsSafeFilesystem(
        "C:/workspace", adapter=adapter, enforce=True
    )
    target = filesystem.prepare_target(
        "safe/note.txt", _windows_authorization()
    )
    temp_name, temp_handle = filesystem.create_temp(target)
    adapter.close(temp_handle)
    target.close()

    assert not temp_name.startswith(("C:", "\\"))

    source = filesystem.prepare_target(
        "safe/note.txt", _windows_authorization()
    )
    destination = filesystem.prepare_target(
        "safe/note.txt", _windows_authorization()
    )
    filesystem.replace(source, destination)
    rename = next(
        call
        for call in adapter.calls
        if call[0] == "NtSetInformationFile"
        and call[2] == "FileRenameInformation"
    )

    assert rename[4] == "note.txt"
    assert not str(rename[4]).startswith(("C:", "\\"))
    filesystem.close()
