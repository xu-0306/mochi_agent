"""Platform sandbox backend selection."""

from __future__ import annotations

import sys
from functools import lru_cache

from mochi.runtime.sandbox.base import (
    HostSandboxBackend,
    SandboxBackend,
    SandboxCapabilities,
    SandboxMode,
    SandboxPlan,
)
from mochi.runtime.sandbox.linux import BubblewrapSandboxBackend
from mochi.runtime.sandbox.windows import WindowsSandboxBackend


def platform_backend() -> SandboxBackend:
    if sys.platform.startswith("linux"):
        return BubblewrapSandboxBackend()
    if sys.platform == "win32":
        return WindowsSandboxBackend()
    return HostSandboxBackend(degraded_reason=f"unsupported_platform:{sys.platform}")


@lru_cache(maxsize=1)
def observed_platform_capabilities() -> SandboxCapabilities:
    """Probe once per process so API projections share one evidence snapshot."""
    return platform_backend().probe()


def select_sandbox_backend(mode: SandboxMode) -> SandboxBackend:
    if mode == "off":
        return HostSandboxBackend()
    backend = platform_backend()
    capabilities = backend.probe()
    if capabilities.complete or mode == "required":
        return backend
    return HostSandboxBackend(
        degraded_reason=capabilities.degraded_reason or "sandbox_backend_incomplete"
    )


def backend_for_plan(plan: SandboxPlan) -> SandboxBackend:
    if plan.backend == "host":
        return HostSandboxBackend(degraded_reason=plan.capabilities.degraded_reason)
    if plan.backend == "bubblewrap":
        return BubblewrapSandboxBackend()
    if plan.backend == "windows-appcontainer":
        return WindowsSandboxBackend()
    return HostSandboxBackend(degraded_reason=f"unknown_backend:{plan.backend}")


__all__ = [
    "backend_for_plan",
    "observed_platform_capabilities",
    "platform_backend",
    "select_sandbox_backend",
]
