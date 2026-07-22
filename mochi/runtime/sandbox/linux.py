"""Linux bubblewrap sandbox backend."""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

from mochi.runtime.sandbox.base import (
    SandboxBackend,
    SandboxCapabilities,
    SandboxLaunchSpec,
    SandboxPlan,
    SandboxPlanMismatch,
    SandboxUnavailableError,
    canonical_path,
    env_hashes,
    probe_timestamp,
    unavailable_capabilities,
)

_SYSTEM_READ_ROOTS = ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc")
_BACKEND_ADAPTER_VERSION = 1


class BubblewrapSandboxBackend(SandboxBackend):
    """Build argument-only bubblewrap launches after a real namespace probe."""

    def __init__(self, *, binary: str | None = None) -> None:
        self._binary = binary or shutil.which("bwrap")
        self._cached_probe: SandboxCapabilities | None = None

    def probe(self) -> SandboxCapabilities:
        if self._cached_probe is not None:
            return self._cached_probe
        if not sys.platform.startswith("linux"):
            result = unavailable_capabilities("bubblewrap", "bubblewrap_requires_linux")
        elif not self._binary:
            result = unavailable_capabilities("bubblewrap", "bubblewrap_binary_not_found")
        else:
            result = self._run_probe(self._binary)
        self._cached_probe = result
        return result

    @staticmethod
    def _run_probe(binary: str) -> SandboxCapabilities:
        try:
            version_result = subprocess.run(
                [binary, "--version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=3,
            )
            binary_version = (version_result.stdout or version_result.stderr).strip()
            if version_result.returncode != 0 or not binary_version:
                return unavailable_capabilities("bubblewrap", "bubblewrap_version_probe_failed")
            smoke = subprocess.run(
                [
                    binary,
                    "--die-with-parent",
                    "--new-session",
                    "--unshare-pid",
                    "--unshare-net",
                    "--ro-bind",
                    "/",
                    "/",
                    "--proc",
                    "/proc",
                    "--dev",
                    "/dev",
                    "--",
                    "/bin/true",
                ],
                check=False,
                capture_output=True,
                timeout=5,
            )
            if smoke.returncode != 0:
                return unavailable_capabilities("bubblewrap", "bubblewrap_namespace_probe_failed")
            return SandboxCapabilities(
                backend="bubblewrap",
                version=f"adapter-{_BACKEND_ADAPTER_VERSION}/{binary_version}"[:128],
                available=True,
                filesystem=True,
                process=True,
                network=True,
                detached=False,
                last_probe_at=probe_timestamp(),
            )
        except (OSError, subprocess.SubprocessError):
            return unavailable_capabilities("bubblewrap", "bubblewrap_probe_failed")

    def prepare_launch(
        self,
        plan: SandboxPlan,
        *,
        env: Mapping[str, str] | None,
    ) -> SandboxLaunchSpec:
        self.validate_plan(plan)
        if not self._binary:
            raise SandboxUnavailableError("bubblewrap binary is unavailable.")
        if env_hashes(env) != plan.env:
            raise SandboxPlanMismatch("Sandbox environment changed after approval.")

        args: list[str] = [
            "--die-with-parent",
            "--new-session",
            "--unshare-pid",
            "--unshare-uts",
            "--unshare-ipc",
            "--clearenv",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
        ]
        if plan.network_policy == "deny":
            args.append("--unshare-net")

        mounted: set[str] = set()
        for root in _SYSTEM_READ_ROOTS:
            if Path(root).exists():
                normalized = canonical_path(root)
                args.extend(("--ro-bind", normalized, normalized))
                mounted.add(normalized)
        for parent in _mount_parent_dirs((*plan.read_roots, *plan.write_roots)):
            args.extend(("--dir", parent))
        for root in plan.read_roots:
            if root not in mounted:
                args.extend(("--ro-bind", root, root))
                mounted.add(root)
        for root in plan.write_roots:
            args.extend(("--bind", root, root))
        launch_env = dict(env or {})
        launch_env.setdefault("PATH", "/usr/bin:/bin")
        for name, value in sorted(launch_env.items()):
            args.extend(("--setenv", name, value))
        args.extend(("--chdir", plan.resolved_cwd, "--", plan.executable, *plan.argv))
        return SandboxLaunchSpec(
            executable=self._binary,
            args=tuple(args),
            cwd=plan.resolved_cwd,
            env=None,
            backend="bubblewrap",
            plan_digest=plan.digest,
        )


def _mount_parent_dirs(roots: tuple[str, ...]) -> tuple[str, ...]:
    parents: set[str] = set()
    for root in roots:
        for parent in Path(root).parents:
            normalized = canonical_path(parent)
            if normalized in {"/", "/tmp"}:
                continue
            if any(
                normalized == system_root or normalized.startswith(f"{system_root}/")
                for system_root in _SYSTEM_READ_ROOTS
            ):
                continue
            parents.add(normalized)
    return tuple(sorted(parents, key=lambda value: (value.count("/"), value)))


__all__ = ["BubblewrapSandboxBackend"]
