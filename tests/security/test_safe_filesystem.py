from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
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
        schema_version=1,
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
            file_request=replace(envelope.file_request, patch_sha256=_sha("other-patch")),
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
        command_utf8_sha256=_sha("command"),
        shell="powershell",
        executable="pwsh.exe",
        argv=("-NoProfile", "-Command", "Get-Date"),
        resolved_cwd="workspace",
        env=(EnvVarHash("PATH", _sha("path-value")),),
        network_policy="deny",
        resource_limits=ResourceLimits(30, 1024, 4096),
        requested_escalation="none",
        sandbox_backend="windows-native",
        sandbox_capability_plan_digest=_sha("sandbox-plan"),
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
        EnvVarHash("PATH", _sha("hash")),
        ResourceLimits(30, 1024, 4096),
        ExecRequest(
            _sha("command"), None, "tool", (), "workspace", (), "deny",
            ResourceLimits(30, 1024, 4096), "none", "native", _sha("plan")
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
            patch_sha256=_sha("patch"),
            policy_version="policy-v1",
            created_at="2026-01-01T00:00:00Z",
            expires_at="2026-01-01T01:00:00Z",
            request_digest=_sha("request"),
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
        patch_sha256=_sha("patch"),
        policy_version="policy-v1",
        created_at="created",
        expires_at="expires",
        request_digest=_sha("request"),
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
        replace(envelope, file_request=replace(envelope.file_request, patch_sha256=_sha("different")))
    )


def test_authorization_envelope_supports_only_schema_version_one() -> None:
    envelope = _authorization()

    with pytest.raises(ValueError, match="schema_version.*1"):
        replace(envelope, schema_version=2)


def test_file_request_and_manifest_reject_duplicate_entry_path_keys() -> None:
    from mochi.security.file_contract import FileChangeRequest

    duplicate = replace(_entry(), operation="delete")
    with pytest.raises(ValueError, match="duplicate"):
        FileChangeRequest(
            entries=(_entry(), duplicate),
            patch_sha256=_sha("patch"),
        )
    with pytest.raises(ValueError, match="duplicate"):
        _manifest(entries=(_entry(), duplicate))


def test_all_explicit_digest_fields_require_lowercase_sha256_hex() -> None:
    from mochi.security.file_contract import (
        EnvVarHash,
        ExecRequest,
        FileChangeRequest,
        ResourceLimits,
    )

    exec_request = ExecRequest(
        command_utf8_sha256=_sha("command"),
        shell=None,
        executable="tool",
        argv=(),
        resolved_cwd="workspace",
        env=(EnvVarHash("PATH", _sha("env")),),
        network_policy="deny",
        resource_limits=ResourceLimits(30, 1024, 4096),
        requested_escalation="none",
        sandbox_backend="native",
        sandbox_capability_plan_digest=_sha("sandbox-plan"),
    )
    manifest = _manifest()
    cases: tuple[Callable[[str], object], ...] = (
        lambda value: replace(_entry(), base_sha256=value),
        lambda value: replace(_entry(), after_sha256=value),
        lambda value: replace(_entry(), base_metadata_sha256=value),
        lambda value: replace(_entry(), after_metadata_sha256=value),
        lambda value: FileChangeRequest(
            entries=(_entry(),), patch_sha256=value
        ),
        lambda value: EnvVarHash("PATH", value),
        lambda value: replace(exec_request, command_utf8_sha256=value),
        lambda value: replace(
            exec_request, sandbox_capability_plan_digest=value
        ),
        lambda value: replace(manifest, patch_sha256=value),
        lambda value: replace(manifest, request_digest=value),
    )
    invalid_digests = ("A" * 64, "g" * 64, "a" * 63)

    for invalid in invalid_digests:
        for construct in cases:
            with pytest.raises(ValueError, match="lowercase hexadecimal"):
                construct(invalid)


def test_manifest_projection_fully_validates_raw_mappings() -> None:
    from mochi.security.file_contract import manifest_digest_projection

    payload = _manifest().to_dict()
    payload["patch_sha256"] = "not-a-digest"

    with pytest.raises(ValueError, match="lowercase hexadecimal"):
        manifest_digest_projection(payload)


def test_authorized_file_binding_captures_all_content_and_metadata_hashes() -> None:
    from mochi.security.safe_filesystem import (
        resolve_authorized_file_binding,
    )

    authorization = _authorization()
    entry = authorization.file_request.entries[0]

    binding = resolve_authorized_file_binding(
        canonical_relative_path="safe/note.txt",
        authorization=authorization,
        captured_identity=entry.base_identity,
        canonicalize_authorized_path=lambda path: path,
    )

    assert binding.base_sha256 == entry.base_sha256
    assert binding.after_sha256 == entry.after_sha256
    assert binding.base_metadata_sha256 == entry.base_metadata_sha256
    assert binding.after_metadata_sha256 == entry.after_metadata_sha256


def test_authorized_file_binding_rejects_tampered_or_alternate_entry_substitution() -> None:
    from mochi.security.file_contract import FileChangeRequest
    from mochi.security.safe_filesystem import (
        resolve_authorized_file_binding,
    )

    original = _authorization()
    first_entry = original.file_request.entries[0]
    alternate_entry = replace(
        first_entry,
        entry_id="0002",
        relative_path="safe/other.txt",
        base_sha256=_sha("alternate-base"),
        after_sha256=_sha("alternate-after"),
        base_metadata_sha256=_sha("alternate-base-metadata"),
        after_metadata_sha256=_sha("alternate-after-metadata"),
    )
    authorization = replace(
        original,
        file_request=FileChangeRequest(
            entries=(first_entry, alternate_entry),
            patch_sha256=original.file_request.patch_sha256,
        ),
    )

    def resolve(path: str, envelope=authorization):
        return resolve_authorized_file_binding(
            canonical_relative_path=path,
            authorization=envelope,
            captured_identity=first_entry.base_identity,
            canonicalize_authorized_path=lambda value: value,
        )

    expected = resolve("safe/note.txt")
    alternate = resolve("safe/other.txt")
    tampered_entry = replace(
        first_entry,
        after_sha256=_sha("tampered-after"),
        after_metadata_sha256=_sha("tampered-after-metadata"),
    )
    tampered_authorization = replace(
        original,
        file_request=FileChangeRequest(
            entries=(tampered_entry,),
            patch_sha256=original.file_request.patch_sha256,
        ),
    )
    tampered = resolve("safe/note.txt", tampered_authorization)

    assert alternate != expected
    assert tampered != expected
    assert alternate.after_sha256 == alternate_entry.after_sha256
    assert tampered.after_sha256 == tampered_entry.after_sha256
    assert tampered.after_metadata_sha256 == (
        tampered_entry.after_metadata_sha256
    )
    assert tampered.authorization_digest != expected.authorization_digest
    hash_only_alternate = replace(
        alternate,
        entry_id=expected.entry_id,
        canonical_relative_path=expected.canonical_relative_path,
    )
    assert hash_only_alternate != expected


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

def _windows_authorization(
    operation: Literal["add", "update", "delete", "rename"] = "update",
) -> AuthorizationEnvelope:
    from mochi.security.file_contract import FileIdentity

    return _file_authorization(
        "safe/note.txt",
        FileIdentity("windows", "10", "41", 1, False),
        operation,
    )

class _FakeWindowsAdapter:
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

    def close(self, handle: object) -> None:
        self.calls.append(("CloseHandle", handle))
        self.nodes.pop(handle, None)


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
        patch_sha256=_sha("patch"),
        policy_version="p",
        created_at="created",
        expires_at="expires",
        request_digest=_sha("request"),
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
                request, command_utf8_sha256=_sha("other-command")
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
                env=(EnvVarHash("PATH", _sha("other-value")),),
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
                sandbox_capability_plan_digest=_sha("other-plan"),
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


def test_windows_replace_reopens_and_verifies_successor_relative() -> None:
    from mochi.security.safe_filesystem import StagedTemp
    from mochi.security.safe_fs_windows import WindowsSafeFilesystem

    adapter = _FakeWindowsAdapter()
    filesystem = WindowsSafeFilesystem(
        "C:/workspace", adapter=adapter, enforce=True
    )
    destination = filesystem.prepare_target(
        "safe/note.txt", _windows_authorization()
    )
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
        "safe/note.txt", _windows_authorization()
    )
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
        "safe/note.txt", _windows_authorization()
    )
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
