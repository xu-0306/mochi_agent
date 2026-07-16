"""Versioned, canonical contracts for filesystem and exec authorization."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import ClassVar, Literal, TypeAlias

JsonValue: TypeAlias = None | bool | int | str | list["JsonValue"] | dict[str, "JsonValue"]

VOLATILE_MANIFEST_FIELDS = frozenset(
    {"change_set_id", "created_at", "expires_at", "request_digest", "ui_metadata"}
)


def _check_fields(data: Mapping[str, object], fields: frozenset[str]) -> None:
    unknown = set(data) - fields
    if unknown:
        raise ValueError(f"unknown contract field(s): {', '.join(sorted(unknown))}")
    missing = fields - set(data)
    if missing:
        raise ValueError(f"missing contract field(s): {', '.join(sorted(missing))}")


def _string(value: object, name: str, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, name: str, *, optional: bool = False) -> int | None:
    if optional and value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} keys must be strings")
    return value  # type: ignore[return-value]


def _sequence(value: object, name: str) -> tuple[object, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be an array")
    return tuple(value)


@dataclass(frozen=True, slots=True)
class FileIdentity:
    """Identity captured from an already-open file or directory handle."""

    platform: Literal["posix", "windows"]
    volume_id: str
    file_id: str
    link_count: int
    is_reparse_point: bool

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"platform", "volume_id", "file_id", "link_count", "is_reparse_point"}
    )

    def __post_init__(self) -> None:
        if self.platform not in {"posix", "windows"}:
            raise ValueError("platform must be 'posix' or 'windows'")
        _string(self.volume_id, "volume_id")
        _string(self.file_id, "file_id")
        link_count = _integer(self.link_count, "link_count")
        if link_count is None or link_count < 1:
            raise ValueError("link_count must be positive")
        _boolean(self.is_reparse_point, "is_reparse_point")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "platform": self.platform,
            "volume_id": self.volume_id,
            "file_id": self.file_id,
            "link_count": self.link_count,
            "is_reparse_point": self.is_reparse_point,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> FileIdentity:
        _check_fields(data, cls._FIELDS)
        return cls(
            platform=_string(data["platform"], "platform"),  # type: ignore[arg-type]
            volume_id=_string(data["volume_id"], "volume_id"),  # type: ignore[arg-type]
            file_id=_string(data["file_id"], "file_id"),  # type: ignore[arg-type]
            link_count=_integer(data["link_count"], "link_count"),  # type: ignore[arg-type]
            is_reparse_point=_boolean(data["is_reparse_point"], "is_reparse_point"),
        )


@dataclass(frozen=True, slots=True)
class ChangeEntry:
    entry_id: str
    relative_path: str
    operation: Literal["add", "update", "delete", "rename"]
    base_sha256: str | None
    after_sha256: str | None
    base_identity: FileIdentity | None
    before_blob_id: str | None
    after_blob_id: str | None
    mode_before: int | None
    mode_after: int | None
    base_metadata_sha256: str | None
    after_metadata_sha256: str | None
    rename_source: str | None
    dependency_group: str | None

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "entry_id", "relative_path", "operation", "base_sha256", "after_sha256",
            "base_identity", "before_blob_id", "after_blob_id", "mode_before",
            "mode_after", "base_metadata_sha256", "after_metadata_sha256",
            "rename_source", "dependency_group",
        }
    )

    def __post_init__(self) -> None:
        _string(self.entry_id, "entry_id")
        _string(self.relative_path, "relative_path")
        if self.operation not in {"add", "update", "delete", "rename"}:
            raise ValueError("invalid change operation")
        for name in (
            "base_sha256", "after_sha256", "before_blob_id", "after_blob_id",
            "base_metadata_sha256", "after_metadata_sha256", "rename_source",
            "dependency_group",
        ):
            _string(getattr(self, name), name, optional=True)
        for name in ("mode_before", "mode_after"):
            value = _integer(getattr(self, name), name, optional=True)
            if value is not None and value < 0:
                raise ValueError(f"{name} must not be negative")
        if self.base_identity is not None and not isinstance(self.base_identity, FileIdentity):
            raise ValueError("base_identity must be a FileIdentity or None")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "entry_id": self.entry_id,
            "relative_path": self.relative_path,
            "operation": self.operation,
            "base_sha256": self.base_sha256,
            "after_sha256": self.after_sha256,
            "base_identity": None if self.base_identity is None else self.base_identity.to_dict(),
            "before_blob_id": self.before_blob_id,
            "after_blob_id": self.after_blob_id,
            "mode_before": self.mode_before,
            "mode_after": self.mode_after,
            "base_metadata_sha256": self.base_metadata_sha256,
            "after_metadata_sha256": self.after_metadata_sha256,
            "rename_source": self.rename_source,
            "dependency_group": self.dependency_group,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ChangeEntry:
        _check_fields(data, cls._FIELDS)
        identity_data = data["base_identity"]
        identity = None if identity_data is None else FileIdentity.from_dict(
            _mapping(identity_data, "base_identity")
        )
        return cls(
            entry_id=_string(data["entry_id"], "entry_id"),  # type: ignore[arg-type]
            relative_path=_string(data["relative_path"], "relative_path"),  # type: ignore[arg-type]
            operation=_string(data["operation"], "operation"),  # type: ignore[arg-type]
            base_sha256=_string(data["base_sha256"], "base_sha256", optional=True),
            after_sha256=_string(data["after_sha256"], "after_sha256", optional=True),
            base_identity=identity,
            before_blob_id=_string(data["before_blob_id"], "before_blob_id", optional=True),
            after_blob_id=_string(data["after_blob_id"], "after_blob_id", optional=True),
            mode_before=_integer(data["mode_before"], "mode_before", optional=True),
            mode_after=_integer(data["mode_after"], "mode_after", optional=True),
            base_metadata_sha256=_string(
                data["base_metadata_sha256"], "base_metadata_sha256", optional=True
            ),
            after_metadata_sha256=_string(
                data["after_metadata_sha256"], "after_metadata_sha256", optional=True
            ),
            rename_source=_string(data["rename_source"], "rename_source", optional=True),
            dependency_group=_string(
                data["dependency_group"], "dependency_group", optional=True
            ),
        )


@dataclass(frozen=True, slots=True)
class FileChangeRequest:
    entries: tuple[ChangeEntry, ...]
    patch_sha256: str | None

    _FIELDS: ClassVar[frozenset[str]] = frozenset({"entries", "patch_sha256"})

    def __post_init__(self) -> None:
        if not isinstance(self.entries, tuple) or any(
            not isinstance(entry, ChangeEntry) for entry in self.entries
        ):
            raise ValueError("entries must be a tuple of ChangeEntry values")
        object.__setattr__(
            self,
            "entries",
            tuple(sorted(self.entries, key=lambda item: (item.entry_id, item.relative_path))),
        )
        _string(self.patch_sha256, "patch_sha256", optional=True)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "patch_sha256": self.patch_sha256,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> FileChangeRequest:
        _check_fields(data, cls._FIELDS)
        return cls(
            entries=tuple(
                ChangeEntry.from_dict(_mapping(item, "entries item"))
                for item in _sequence(data["entries"], "entries")
            ),
            patch_sha256=_string(data["patch_sha256"], "patch_sha256", optional=True),
        )


@dataclass(frozen=True, slots=True)
class EnvVarHash:
    key: str
    value_sha256: str

    _FIELDS: ClassVar[frozenset[str]] = frozenset({"key", "value_sha256"})

    def __post_init__(self) -> None:
        _string(self.key, "key")
        _string(self.value_sha256, "value_sha256")

    def to_dict(self) -> dict[str, JsonValue]:
        return {"key": self.key, "value_sha256": self.value_sha256}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> EnvVarHash:
        _check_fields(data, cls._FIELDS)
        return cls(
            _string(data["key"], "key"),  # type: ignore[arg-type]
            _string(data["value_sha256"], "value_sha256"),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    timeout_seconds: int
    memory_limit_mb: int
    output_limit_bytes: int

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"timeout_seconds", "memory_limit_mb", "output_limit_bytes"}
    )

    def __post_init__(self) -> None:
        for name in self._FIELDS:
            value = _integer(getattr(self, name), name)
            if value is None or value < 0:
                raise ValueError(f"{name} must not be negative")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "timeout_seconds": self.timeout_seconds,
            "memory_limit_mb": self.memory_limit_mb,
            "output_limit_bytes": self.output_limit_bytes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ResourceLimits:
        _check_fields(data, cls._FIELDS)
        return cls(
            _integer(data["timeout_seconds"], "timeout_seconds"),  # type: ignore[arg-type]
            _integer(data["memory_limit_mb"], "memory_limit_mb"),  # type: ignore[arg-type]
            _integer(data["output_limit_bytes"], "output_limit_bytes"),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class ExecRequest:
    command_utf8_sha256: str
    shell: str | None
    executable: str
    argv: tuple[str, ...]
    resolved_cwd: str
    env: tuple[EnvVarHash, ...]
    network_policy: Literal["deny", "allow"]
    resource_limits: ResourceLimits
    requested_escalation: str
    sandbox_backend: str
    sandbox_capability_plan_digest: str

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "command_utf8_sha256", "shell", "executable", "argv", "resolved_cwd",
            "env", "network_policy", "resource_limits", "requested_escalation",
            "sandbox_backend", "sandbox_capability_plan_digest",
        }
    )

    def __post_init__(self) -> None:
        _string(self.command_utf8_sha256, "command_utf8_sha256")
        _string(self.shell, "shell", optional=True)
        _string(self.executable, "executable")
        _string(self.resolved_cwd, "resolved_cwd")
        if not isinstance(self.argv, tuple) or any(not isinstance(item, str) for item in self.argv):
            raise ValueError("argv must be a tuple of strings")
        if not isinstance(self.env, tuple) or any(
            not isinstance(item, EnvVarHash) for item in self.env
        ):
            raise ValueError("env must be a tuple of EnvVarHash values")
        object.__setattr__(self, "env", tuple(sorted(self.env, key=lambda item: item.key)))
        if self.network_policy not in {"deny", "allow"}:
            raise ValueError("network_policy must be 'deny' or 'allow'")
        if not isinstance(self.resource_limits, ResourceLimits):
            raise ValueError("resource_limits must be ResourceLimits")
        _string(self.requested_escalation, "requested_escalation")
        _string(self.sandbox_backend, "sandbox_backend")
        _string(self.sandbox_capability_plan_digest, "sandbox_capability_plan_digest")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "command_utf8_sha256": self.command_utf8_sha256,
            "shell": self.shell,
            "executable": self.executable,
            "argv": list(self.argv),
            "resolved_cwd": self.resolved_cwd,
            "env": [item.to_dict() for item in self.env],
            "network_policy": self.network_policy,
            "resource_limits": self.resource_limits.to_dict(),
            "requested_escalation": self.requested_escalation,
            "sandbox_backend": self.sandbox_backend,
            "sandbox_capability_plan_digest": self.sandbox_capability_plan_digest,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ExecRequest:
        _check_fields(data, cls._FIELDS)
        argv = _sequence(data["argv"], "argv")
        if any(not isinstance(item, str) for item in argv):
            raise ValueError("argv items must be strings")
        return cls(
            command_utf8_sha256=_string(
                data["command_utf8_sha256"], "command_utf8_sha256"
            ),  # type: ignore[arg-type]
            shell=_string(data["shell"], "shell", optional=True),
            executable=_string(data["executable"], "executable"),  # type: ignore[arg-type]
            argv=argv,  # type: ignore[arg-type]
            resolved_cwd=_string(data["resolved_cwd"], "resolved_cwd"),  # type: ignore[arg-type]
            env=tuple(
                EnvVarHash.from_dict(_mapping(item, "env item"))
                for item in _sequence(data["env"], "env")
            ),
            network_policy=_string(data["network_policy"], "network_policy"),  # type: ignore[arg-type]
            resource_limits=ResourceLimits.from_dict(
                _mapping(data["resource_limits"], "resource_limits")
            ),
            requested_escalation=_string(
                data["requested_escalation"], "requested_escalation"
            ),  # type: ignore[arg-type]
            sandbox_backend=_string(
                data["sandbox_backend"], "sandbox_backend"
            ),  # type: ignore[arg-type]
            sandbox_capability_plan_digest=_string(
                data["sandbox_capability_plan_digest"], "sandbox_capability_plan_digest"
            ),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class AuthorizationContext:
    requester_id: str
    session_id: str
    task_id: str | None
    workspace_root: str
    workspace_identity: FileIdentity

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"requester_id", "session_id", "task_id", "workspace_root", "workspace_identity"}
    )

    def __post_init__(self) -> None:
        _string(self.requester_id, "requester_id")
        _string(self.session_id, "session_id")
        _string(self.task_id, "task_id", optional=True)
        _string(self.workspace_root, "workspace_root")
        if not isinstance(self.workspace_identity, FileIdentity):
            raise ValueError("workspace_identity must be a FileIdentity")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "requester_id": self.requester_id,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "workspace_root": self.workspace_root,
            "workspace_identity": self.workspace_identity.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> AuthorizationContext:
        _check_fields(data, cls._FIELDS)
        return cls(
            requester_id=_string(data["requester_id"], "requester_id"),  # type: ignore[arg-type]
            session_id=_string(data["session_id"], "session_id"),  # type: ignore[arg-type]
            task_id=_string(data["task_id"], "task_id", optional=True),
            workspace_root=_string(data["workspace_root"], "workspace_root"),  # type: ignore[arg-type]
            workspace_identity=FileIdentity.from_dict(
                _mapping(data["workspace_identity"], "workspace_identity")
            ),
        )


@dataclass(frozen=True, slots=True)
class AuthorizationEnvelope:
    schema_version: int
    kind: Literal["file_change", "exec"]
    context: AuthorizationContext
    policy_version: str
    file_request: FileChangeRequest | None
    exec_request: ExecRequest | None

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"schema_version", "kind", "context", "policy_version", "file_request", "exec_request"}
    )

    def __post_init__(self) -> None:
        version = _integer(self.schema_version, "schema_version")
        if version is None or version < 1:
            raise ValueError("schema_version must be positive")
        if self.kind not in {"file_change", "exec"}:
            raise ValueError("kind must be 'file_change' or 'exec'")
        if not isinstance(self.context, AuthorizationContext):
            raise ValueError("context must be AuthorizationContext")
        _string(self.policy_version, "policy_version")
        present = int(self.file_request is not None) + int(self.exec_request is not None)
        if present != 1:
            raise ValueError("exactly one request must be present")
        if self.kind == "file_change" and self.file_request is None:
            raise ValueError("kind does not match request")
        if self.kind == "exec" and self.exec_request is None:
            raise ValueError("kind does not match request")
        if self.file_request is not None and not isinstance(
            self.file_request, FileChangeRequest
        ):
            raise ValueError("file_request must be FileChangeRequest")
        if self.exec_request is not None and not isinstance(self.exec_request, ExecRequest):
            raise ValueError("exec_request must be ExecRequest")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "context": self.context.to_dict(),
            "policy_version": self.policy_version,
            "file_request": None if self.file_request is None else self.file_request.to_dict(),
            "exec_request": None if self.exec_request is None else self.exec_request.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> AuthorizationEnvelope:
        _check_fields(data, cls._FIELDS)
        file_data = data["file_request"]
        exec_data = data["exec_request"]
        return cls(
            schema_version=_integer(data["schema_version"], "schema_version"),  # type: ignore[arg-type]
            kind=_string(data["kind"], "kind"),  # type: ignore[arg-type]
            context=AuthorizationContext.from_dict(_mapping(data["context"], "context")),
            policy_version=_string(data["policy_version"], "policy_version"),  # type: ignore[arg-type]
            file_request=None if file_data is None else FileChangeRequest.from_dict(
                _mapping(file_data, "file_request")
            ),
            exec_request=None if exec_data is None else ExecRequest.from_dict(
                _mapping(exec_data, "exec_request")
            ),
        )


@dataclass(frozen=True, slots=True)
class ChangeManifest:
    version: int
    change_set_id: str
    workspace_root: str
    workspace_identity: FileIdentity
    tool_name: str
    intent: Literal["mutate", "undo"]
    entries: tuple[ChangeEntry, ...]
    patch_sha256: str | None
    policy_version: str
    created_at: str
    expires_at: str
    request_digest: str
    ui_metadata: Mapping[str, object] = field(default_factory=dict)

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "version", "change_set_id", "workspace_root", "workspace_identity",
            "tool_name", "intent", "entries", "patch_sha256", "policy_version",
            "created_at", "expires_at", "request_digest", "ui_metadata",
        }
    )

    def __post_init__(self) -> None:
        version = _integer(self.version, "version")
        if version is None or version < 1:
            raise ValueError("version must be positive")
        for name in (
            "change_set_id", "workspace_root", "tool_name", "policy_version",
            "created_at", "expires_at", "request_digest",
        ):
            _string(getattr(self, name), name)
        if not isinstance(self.workspace_identity, FileIdentity):
            raise ValueError("workspace_identity must be a FileIdentity")
        if self.intent not in {"mutate", "undo"}:
            raise ValueError("intent must be 'mutate' or 'undo'")
        if not isinstance(self.entries, tuple) or any(
            not isinstance(entry, ChangeEntry) for entry in self.entries
        ):
            raise ValueError("entries must be a tuple of ChangeEntry values")
        object.__setattr__(
            self,
            "entries",
            tuple(sorted(self.entries, key=lambda item: (item.entry_id, item.relative_path))),
        )
        _string(self.patch_sha256, "patch_sha256", optional=True)
        metadata = _mapping(self.ui_metadata, "ui_metadata")
        normalized_metadata = canonical_value(metadata)
        object.__setattr__(
            self,
            "ui_metadata",
            _freeze_json(normalized_metadata),
        )


    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "version": self.version,
            "change_set_id": self.change_set_id,
            "workspace_root": self.workspace_root,
            "workspace_identity": self.workspace_identity.to_dict(),
            "tool_name": self.tool_name,
            "intent": self.intent,
            "entries": [entry.to_dict() for entry in self.entries],
            "patch_sha256": self.patch_sha256,
            "policy_version": self.policy_version,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "request_digest": self.request_digest,
            "ui_metadata": canonical_value(self.ui_metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ChangeManifest:
        _check_fields(data, cls._FIELDS)
        return cls(
            version=_integer(data["version"], "version"),  # type: ignore[arg-type]
            change_set_id=_string(data["change_set_id"], "change_set_id"),  # type: ignore[arg-type]
            workspace_root=_string(data["workspace_root"], "workspace_root"),  # type: ignore[arg-type]
            workspace_identity=FileIdentity.from_dict(
                _mapping(data["workspace_identity"], "workspace_identity")
            ),
            tool_name=_string(data["tool_name"], "tool_name"),  # type: ignore[arg-type]
            intent=_string(data["intent"], "intent"),  # type: ignore[arg-type]
            entries=tuple(
                ChangeEntry.from_dict(_mapping(item, "entries item"))
                for item in _sequence(data["entries"], "entries")
            ),
            patch_sha256=_string(data["patch_sha256"], "patch_sha256", optional=True),
            policy_version=_string(data["policy_version"], "policy_version"),  # type: ignore[arg-type]
            created_at=_string(data["created_at"], "created_at"),  # type: ignore[arg-type]
            expires_at=_string(data["expires_at"], "expires_at"),  # type: ignore[arg-type]
            request_digest=_string(data["request_digest"], "request_digest"),  # type: ignore[arg-type]
            ui_metadata=_mapping(data["ui_metadata"], "ui_metadata"),  # type: ignore[arg-type]
        )


def _freeze_json(value: JsonValue) -> object:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(child) for key, child in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(child) for child in value)
    return value

def canonical_value(value: object) -> JsonValue:
    """Return the strict JSON projection used by security digests."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        raise TypeError("floating-point values are forbidden in canonical contracts")
    if isinstance(value, Mapping):
        result: dict[str, JsonValue] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical JSON object keys must be strings")
            result[key] = canonical_value(child)
        return result
    if isinstance(value, (list, tuple)):
        return [canonical_value(item) for item in value]
    raise TypeError(f"value of type {type(value).__name__} is not canonical JSON")


def canonical_json(value: object) -> str:
    return json.dumps(
        canonical_value(value), ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )


def authorization_request_digest(envelope: AuthorizationEnvelope) -> str:
    if not isinstance(envelope, AuthorizationEnvelope):
        raise TypeError("envelope must be AuthorizationEnvelope")
    return hashlib.sha256(canonical_json(envelope.to_dict()).encode("utf-8")).hexdigest()


def manifest_digest_projection(
    manifest: ChangeManifest | Mapping[str, object],
) -> dict[str, JsonValue]:
    if isinstance(manifest, ChangeManifest):
        source: Mapping[str, object] = manifest.to_dict()
    elif isinstance(manifest, Mapping):
        source = manifest
    else:
        raise TypeError("manifest must be ChangeManifest or a mapping")
    projection = {
        key: value for key, value in source.items() if key not in VOLATILE_MANIFEST_FIELDS
    }
    normalized = canonical_value(projection)
    if not isinstance(normalized, dict):
        raise TypeError("manifest projection must be an object")
    return normalized


def canonical_manifest_digest(
    manifest: ChangeManifest | Mapping[str, object],
    envelope: AuthorizationEnvelope,
) -> str:
    payload = {
        "authorization": envelope.to_dict(),
        "manifest": manifest_digest_projection(manifest),
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


PreviewIdempotencyKey: TypeAlias = tuple[int, str, str, str, str | None, str, str]


def preview_idempotency_key(envelope: AuthorizationEnvelope) -> PreviewIdempotencyKey:
    context = envelope.context
    return (
        envelope.schema_version,
        envelope.kind,
        context.requester_id,
        context.session_id,
        context.task_id,
        canonical_json(context.workspace_identity.to_dict()),
        authorization_request_digest(envelope),
    )


__all__ = [
    "AuthorizationContext",
    "AuthorizationEnvelope",
    "ChangeEntry",
    "ChangeManifest",
    "EnvVarHash",
    "ExecRequest",
    "FileChangeRequest",
    "FileIdentity",
    "PreviewIdempotencyKey",
    "ResourceLimits",
    "VOLATILE_MANIFEST_FIELDS",
    "authorization_request_digest",
    "canonical_json",
    "canonical_manifest_digest",
    "canonical_value",
    "manifest_digest_projection",
    "preview_idempotency_key",
]
