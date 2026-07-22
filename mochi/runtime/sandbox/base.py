"""Digest-bound operating-system sandbox plans and backend contracts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import shutil
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

SandboxMode = Literal["off", "preferred", "required"]
SandboxNetworkPolicy = Literal["allow", "deny"]

SANDBOX_PLAN_SCHEMA_VERSION = 1


class SandboxError(RuntimeError):
    """Base error for sandbox planning and launch validation."""


class SandboxUnavailableError(SandboxError):
    """Raised when configured containment cannot be enforced."""


class SandboxPlanMismatch(SandboxError):
    """Raised when an approved plan no longer matches launch-time facts."""


@dataclass(frozen=True, slots=True)
class SandboxCapabilities:
    """Observed backend capabilities; never inferred from configured intent."""

    backend: str
    version: str
    available: bool
    filesystem: bool
    process: bool
    network: bool
    detached: bool = False
    degraded_reason: str | None = None
    last_probe_at: str | None = None

    @property
    def complete(self) -> bool:
        return bool(
            self.available
            and self.filesystem
            and self.process
            and self.network
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "version": self.version,
            "available": self.available,
            "filesystem": self.filesystem,
            "process": self.process,
            "network": self.network,
            "detached": self.detached,
            "degraded_reason": self.degraded_reason,
            "last_probe_at": self.last_probe_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SandboxCapabilities:
        expected_fields = {
            "backend",
            "version",
            "available",
            "filesystem",
            "process",
            "network",
            "detached",
            "degraded_reason",
            "last_probe_at",
        }
        if set(value) != expected_fields:
            raise ValueError("Sandbox capability fields do not match the schema.")
        return cls(
            backend=_bounded_identifier(value.get("backend"), field="backend"),
            version=_bounded_text(value.get("version"), field="version", maximum=128),
            available=_boolean(value.get("available"), field="available"),
            filesystem=_boolean(value.get("filesystem"), field="filesystem"),
            process=_boolean(value.get("process"), field="process"),
            network=_boolean(value.get("network"), field="network"),
            detached=_boolean(value.get("detached"), field="detached"),
            degraded_reason=(
                _bounded_text(
                    value.get("degraded_reason"),
                    field="degraded_reason",
                    maximum=256,
                )
                if value.get("degraded_reason") is not None
                else None
            ),
            last_probe_at=(
                _bounded_text(value.get("last_probe_at"), field="last_probe_at", maximum=64)
                if value.get("last_probe_at") is not None
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class SandboxEnvVar:
    name: str
    value_sha256: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "value_sha256": self.value_sha256}


@dataclass(frozen=True, slots=True)
class SandboxResourceLimits:
    timeout_milliseconds: int
    memory_limit_mb: int
    max_processes: int
    output_limit_bytes: int

    def __post_init__(self) -> None:
        if self.timeout_milliseconds <= 0:
            raise ValueError("Sandbox timeout must be positive.")
        if self.memory_limit_mb < 0:
            raise ValueError("Sandbox memory limit must be non-negative.")
        if self.max_processes <= 0:
            raise ValueError("Sandbox process limit must be positive.")
        if self.output_limit_bytes <= 0:
            raise ValueError("Sandbox output limit must be positive.")

    def to_dict(self) -> dict[str, int]:
        return {
            "timeout_milliseconds": self.timeout_milliseconds,
            "memory_limit_mb": self.memory_limit_mb,
            "max_processes": self.max_processes,
            "output_limit_bytes": self.output_limit_bytes,
        }


@dataclass(frozen=True, slots=True)
class SandboxPlan:
    """Canonical launch facts approved before an OS backend is invoked."""

    mode: SandboxMode
    executable: str
    argv: tuple[str, ...]
    resolved_cwd: str
    read_roots: tuple[str, ...]
    write_roots: tuple[str, ...]
    network_policy: SandboxNetworkPolicy
    env: tuple[SandboxEnvVar, ...]
    resource_limits: SandboxResourceLimits
    requested_escalation: str
    backend: str
    backend_version: str
    capabilities: SandboxCapabilities
    request_nonce: str
    schema_version: int = SANDBOX_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SANDBOX_PLAN_SCHEMA_VERSION:
            raise ValueError("Unsupported sandbox plan schema version.")
        if self.mode not in {"off", "preferred", "required"}:
            raise ValueError("Unsupported sandbox mode.")
        if self.network_policy not in {"allow", "deny"}:
            raise ValueError("Unsupported sandbox network policy.")
        _bounded_text(self.executable, field="executable", maximum=32768)
        _bounded_text(self.resolved_cwd, field="resolved_cwd", maximum=32768)
        if not self.read_roots and not self.write_roots:
            raise ValueError("Sandbox plans require at least one filesystem root.")
        for root in (*self.read_roots, *self.write_roots):
            _bounded_text(root, field="root", maximum=32768)
        if len(self.argv) > 4096:
            raise ValueError("Sandbox argv must be bounded.")
        for argument in self.argv:
            _bounded_argument(argument, field="argv[]")
        object.__setattr__(self, "executable", canonical_path(self.executable))
        object.__setattr__(self, "resolved_cwd", canonical_path(self.resolved_cwd))
        object.__setattr__(
            self,
            "read_roots",
            tuple(sorted({canonical_path(root) for root in self.read_roots})),
        )
        object.__setattr__(
            self,
            "write_roots",
            tuple(sorted({canonical_path(root) for root in self.write_roots})),
        )
        if not self.executable.strip():
            raise ValueError("Sandbox executable must not be empty.")
        if not Path(self.executable).is_absolute():
            raise ValueError("Sandbox executable must be absolute.")
        if not Path(self.resolved_cwd).is_absolute():
            raise ValueError("Sandbox cwd must be absolute.")
        if any(not Path(root).is_absolute() for root in (*self.read_roots, *self.write_roots)):
            raise ValueError("Sandbox roots must be absolute.")
        if not any(
            Path(self.resolved_cwd).is_relative_to(Path(root))
            for root in (*self.read_roots, *self.write_roots)
        ):
            raise ValueError("Sandbox cwd must be contained by an approved root.")
        if len({item.name for item in self.env}) != len(self.env):
            raise ValueError("Sandbox env names must be unique.")
        for item in self.env:
            _bounded_identifier(item.name, field="env.name")
            _sha256_digest(item.value_sha256, field="env.value_sha256")
        _bounded_identifier(self.backend, field="backend")
        _bounded_text(self.backend_version, field="backend_version", maximum=128)
        _bounded_identifier(self.requested_escalation, field="requested_escalation")
        if self.capabilities.backend != self.backend:
            raise ValueError("Sandbox capability backend does not match the plan backend.")
        if self.capabilities.version != self.backend_version:
            raise ValueError("Sandbox capability version does not match the plan backend.")
        _bounded_text(self.request_nonce, field="request_nonce", maximum=128)
        if len(self.request_nonce) < 16:
            raise ValueError("Sandbox request nonce must be bounded and unpredictable.")

    @property
    def digest(self) -> str:
        encoded = canonical_json(self.canonical_payload()).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "executable": self.executable,
            "argv": list(self.argv),
            "resolved_cwd": self.resolved_cwd,
            "read_roots": sorted(set(self.read_roots)),
            "write_roots": sorted(set(self.write_roots)),
            "network_policy": self.network_policy,
            "env": [item.to_dict() for item in sorted(self.env, key=lambda item: item.name)],
            "resource_limits": self.resource_limits.to_dict(),
            "requested_escalation": self.requested_escalation,
            "backend": self.backend,
            "backend_version": self.backend_version,
            "capabilities": self.capabilities.to_dict(),
            "request_nonce": self.request_nonce,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.canonical_payload(), "plan_digest": self.digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SandboxPlan:
        expected_fields = {
            "schema_version",
            "mode",
            "executable",
            "argv",
            "resolved_cwd",
            "read_roots",
            "write_roots",
            "network_policy",
            "env",
            "resource_limits",
            "requested_escalation",
            "backend",
            "backend_version",
            "capabilities",
            "request_nonce",
            "plan_digest",
        }
        if set(value) != expected_fields:
            raise ValueError("Sandbox plan fields do not match the schema.")
        raw_env = value.get("env")
        if not isinstance(raw_env, list):
            raise ValueError("Sandbox env must match the schema.")
        raw_env_items = cast(list[object], raw_env)
        if any(not _env_item_matches_schema(item) for item in raw_env_items):
            raise ValueError("Sandbox env must match the schema.")
        env_items: list[SandboxEnvVar] = []
        for raw_item in raw_env_items:
            item = cast(Mapping[str, Any], raw_item)
            env_items.append(
                SandboxEnvVar(
                    name=_bounded_identifier(item.get("name"), field="env.name"),
                    value_sha256=_sha256_digest(
                        item.get("value_sha256"),
                        field="env.value_sha256",
                    ),
                )
            )
        if len({item.name for item in env_items}) != len(env_items):
            raise ValueError("Sandbox env names must be unique.")
        env = tuple(env_items)
        raw_limits = value.get("resource_limits")
        if not isinstance(raw_limits, Mapping):
            raise ValueError("Sandbox resource_limits are required.")
        limits = cast(Mapping[str, Any], raw_limits)
        if set(limits) != {
            "timeout_milliseconds",
            "memory_limit_mb",
            "max_processes",
            "output_limit_bytes",
        }:
            raise ValueError("Sandbox resource_limits do not match the schema.")
        raw_capabilities = value.get("capabilities")
        if not isinstance(raw_capabilities, Mapping):
            raise ValueError("Sandbox capabilities are required.")
        capabilities = cast(Mapping[str, Any], raw_capabilities)
        if set(capabilities) != {
            "backend",
            "version",
            "available",
            "filesystem",
            "process",
            "network",
            "detached",
            "degraded_reason",
            "last_probe_at",
        }:
            raise ValueError("Sandbox capabilities do not match the schema.")
        plan = cls(
            schema_version=_integer(value.get("schema_version"), field="schema_version"),
            mode=cast(SandboxMode, str(value.get("mode") or "")),
            executable=_bounded_text(value.get("executable"), field="executable", maximum=32768),
            argv=tuple(_string_list(value.get("argv"), field="argv")),
            resolved_cwd=_canonical_path(value.get("resolved_cwd"), field="resolved_cwd"),
            read_roots=tuple(_canonical_paths(value.get("read_roots"), field="read_roots")),
            write_roots=tuple(_canonical_paths(value.get("write_roots"), field="write_roots")),
            network_policy=cast(
                SandboxNetworkPolicy,
                str(value.get("network_policy") or ""),
            ),
            env=env,
            resource_limits=SandboxResourceLimits(
                timeout_milliseconds=_integer(
                    limits.get("timeout_milliseconds"),
                    field="resource_limits.timeout_milliseconds",
                ),
                memory_limit_mb=_integer(
                    limits.get("memory_limit_mb"),
                    field="resource_limits.memory_limit_mb",
                ),
                max_processes=_integer(
                    limits.get("max_processes"),
                    field="resource_limits.max_processes",
                ),
                output_limit_bytes=_integer(
                    limits.get("output_limit_bytes"),
                    field="resource_limits.output_limit_bytes",
                ),
            ),
            requested_escalation=_bounded_identifier(
                value.get("requested_escalation"),
                field="requested_escalation",
            ),
            backend=_bounded_identifier(value.get("backend"), field="backend"),
            backend_version=_bounded_text(
                value.get("backend_version"),
                field="backend_version",
                maximum=128,
            ),
            capabilities=SandboxCapabilities.from_dict(capabilities),
            request_nonce=_bounded_text(
                value.get("request_nonce"),
                field="request_nonce",
                maximum=128,
            ),
        )
        supplied_digest = value.get("plan_digest")
        if _sha256_digest(
            supplied_digest,
            field="plan_digest",
        ) != plan.digest:
            raise SandboxPlanMismatch("Sandbox plan digest does not match its payload.")
        return plan


@dataclass(frozen=True, slots=True)
class SandboxLaunchSpec:
    executable: str
    args: tuple[str, ...]
    cwd: str
    env: dict[str, str] | None
    backend: str
    plan_digest: str
    degraded_reason: str | None = None


class SandboxBackend(ABC):
    """Backend boundary used by ExecRuntime after plan revalidation."""

    @abstractmethod
    def probe(self) -> SandboxCapabilities:
        """Return observed capabilities without mutating ACLs or profiles."""

    def validate_plan(self, plan: SandboxPlan) -> SandboxCapabilities:
        """Re-probe and ensure approved capability claims remain true."""
        observed = self.probe()
        observed_facts = (
            observed.backend,
            observed.version,
            observed.available,
            observed.filesystem,
            observed.process,
            observed.network,
            observed.detached,
            observed.degraded_reason,
        )
        approved_facts = (
            plan.capabilities.backend,
            plan.capabilities.version,
            plan.capabilities.available,
            plan.capabilities.filesystem,
            plan.capabilities.process,
            plan.capabilities.network,
            plan.capabilities.detached,
            plan.capabilities.degraded_reason,
        )
        if observed_facts != approved_facts:
            raise SandboxPlanMismatch(
                "Sandbox backend capabilities changed after the plan was approved."
            )
        if plan.mode == "required" and not observed.complete:
            raise SandboxUnavailableError(
                "Required sandbox mode needs filesystem, process, and network enforcement."
            )
        return observed

    @abstractmethod
    def prepare_launch(
        self,
        plan: SandboxPlan,
        *,
        env: Mapping[str, str] | None,
    ) -> SandboxLaunchSpec:
        """Build an argument-list launch spec; shell strings are forbidden."""


class HostSandboxBackend(SandboxBackend):
    """Explicit non-containment backend for off/preferred degradation only."""

    def __init__(self, *, degraded_reason: str | None = None) -> None:
        self._degraded_reason = degraded_reason

    def probe(self) -> SandboxCapabilities:
        return SandboxCapabilities(
            backend="host",
            version="1",
            available=True,
            filesystem=False,
            process=False,
            network=False,
            detached=True,
            degraded_reason=self._degraded_reason,
        )

    def prepare_launch(
        self,
        plan: SandboxPlan,
        *,
        env: Mapping[str, str] | None,
    ) -> SandboxLaunchSpec:
        self.validate_plan(plan)
        return SandboxLaunchSpec(
            executable=plan.executable,
            args=plan.argv,
            cwd=plan.resolved_cwd,
            env=dict(env) if env is not None else None,
            backend="host",
            plan_digest=plan.digest,
            degraded_reason=plan.capabilities.degraded_reason,
        )


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def canonical_path(value: str | Path) -> str:
    return os.path.normcase(str(Path(value).expanduser().resolve(strict=False)))


def env_hashes(env: Mapping[str, str] | None) -> tuple[SandboxEnvVar, ...]:
    return tuple(
        SandboxEnvVar(name=key, value_sha256=hashlib.sha256(value.encode("utf-8")).hexdigest())
        for key, value in sorted((env or {}).items())
    )


def unavailable_capabilities(backend: str, reason: str) -> SandboxCapabilities:
    return SandboxCapabilities(
        backend=backend,
        version="unavailable",
        available=False,
        filesystem=False,
        process=False,
        network=False,
        degraded_reason=reason,
        last_probe_at=probe_timestamp(),
    )


def probe_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def create_sandbox_plan(
    *,
    mode: SandboxMode,
    executable: str,
    argv: tuple[str, ...],
    cwd: str | Path,
    read_roots: tuple[str | Path, ...],
    write_roots: tuple[str | Path, ...],
    network_policy: SandboxNetworkPolicy,
    env: Mapping[str, str] | None,
    resource_limits: SandboxResourceLimits,
    requested_escalation: str,
    backend: SandboxBackend,
) -> SandboxPlan:
    """Capture backend evidence and all launch facts in one approval artifact."""
    capabilities = backend.probe()
    return SandboxPlan(
        mode=mode,
        executable=resolve_executable(executable, env=env),
        argv=argv,
        resolved_cwd=canonical_path(cwd),
        read_roots=tuple(canonical_path(root) for root in read_roots),
        write_roots=tuple(canonical_path(root) for root in write_roots),
        network_policy=network_policy,
        env=env_hashes(env),
        resource_limits=resource_limits,
        requested_escalation=requested_escalation,
        backend=capabilities.backend,
        backend_version=capabilities.version,
        capabilities=capabilities,
        request_nonce=secrets.token_hex(16),
    )


def validate_plan_facts(
    plan: SandboxPlan,
    *,
    executable: str,
    argv: tuple[str, ...],
    cwd: str | Path,
    env: Mapping[str, str] | None,
    timeout_sec: float | None,
) -> None:
    """Reject replay if provider output or caller-controlled facts changed."""
    timeout_milliseconds = int(math.ceil((timeout_sec or 0) * 1000))
    if (
        plan.executable != resolve_executable(executable, env=env)
        or plan.argv != argv
        or plan.resolved_cwd != canonical_path(cwd)
        or plan.env != env_hashes(env)
        or plan.resource_limits.timeout_milliseconds != timeout_milliseconds
    ):
        raise SandboxPlanMismatch(
            "Sandbox plan no longer matches executable, argv, cwd, env, or timeout."
        )


def resolve_executable(
    executable: str,
    *,
    env: Mapping[str, str] | None,
) -> str:
    """Resolve PATH once so approval cannot be replayed with a different binary."""
    candidate = Path(executable)
    if candidate.is_absolute():
        return canonical_path(candidate)
    search_path = env.get("PATH") if env is not None and "PATH" in env else os.environ.get("PATH")
    resolved = shutil.which(executable, path=search_path)
    if resolved is None:
        raise SandboxUnavailableError(f"Sandbox executable could not be resolved: {executable}")
    return canonical_path(resolved)


def _bounded_identifier(value: Any, *, field: str) -> str:
    text = _bounded_text(value, field=field, maximum=128)
    if not all(character.isalnum() or character in "._:-" for character in text):
        raise ValueError(f"Sandbox {field} must be an identifier.")
    return text


def _boolean(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"Sandbox {field} must be a boolean.")
    return value


def _integer(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Sandbox {field} must be an integer.")
    return value


def _bounded_text(value: Any, *, field: str, maximum: int) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ValueError(f"Sandbox {field} must be a non-empty bounded string.")
    return value


def _bounded_argument(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) > 32768:
        raise ValueError(f"Sandbox {field} must be a bounded string.")
    return value


def _sha256_digest(value: Any, *, field: str) -> str:
    text = _bounded_text(value, field=field, maximum=64)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"Sandbox {field} must be a SHA-256 digest.")
    return text


def _string_list(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"Sandbox {field} must be a bounded list.")
    items = cast(list[object], value)
    if len(items) > 4096:
        raise ValueError(f"Sandbox {field} must be a bounded list.")
    return [
        _bounded_argument(item, field=f"{field}[]")
        for item in items
    ]


def _env_item_matches_schema(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    item = cast(Mapping[str, Any], value)
    return set(item) == {"name", "value_sha256"}


def _canonical_path(value: Any, *, field: str) -> str:
    return canonical_path(_bounded_text(value, field=field, maximum=32768))


def _canonical_paths(value: Any, *, field: str) -> list[str]:
    return sorted({_canonical_path(item, field=f"{field}[]") for item in _string_list(value, field=field)})


__all__ = [
    "HostSandboxBackend",
    "SANDBOX_PLAN_SCHEMA_VERSION",
    "SandboxBackend",
    "SandboxCapabilities",
    "SandboxEnvVar",
    "SandboxError",
    "SandboxLaunchSpec",
    "SandboxMode",
    "SandboxNetworkPolicy",
    "SandboxPlan",
    "SandboxPlanMismatch",
    "SandboxResourceLimits",
    "SandboxUnavailableError",
    "canonical_json",
    "canonical_path",
    "create_sandbox_plan",
    "env_hashes",
    "probe_timestamp",
    "resolve_executable",
    "unavailable_capabilities",
    "validate_plan_facts",
]
