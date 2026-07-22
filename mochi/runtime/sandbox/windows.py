"""Windows native broker discovery and fail-closed capability handshake."""

from __future__ import annotations

import secrets
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from mochi.runtime.sandbox.base import (
    SandboxBackend,
    SandboxCapabilities,
    SandboxLaunchSpec,
    SandboxPlan,
    SandboxUnavailableError,
    probe_timestamp,
    unavailable_capabilities,
)
from mochi.runtime.sandbox.broker_protocol import decode_frame


def packaged_helper_path() -> Path:
    return Path(__file__).resolve().parent / "bin" / "mochi-sandbox-windows.exe"


class WindowsSandboxBackend(SandboxBackend):
    """Probe the packaged broker; never infer AppContainer guarantees."""

    def __init__(self, *, helper_path: str | Path | None = None) -> None:
        self._helper_path = Path(helper_path).resolve() if helper_path else packaged_helper_path()
        self._cached_probe: SandboxCapabilities | None = None

    def probe(self) -> SandboxCapabilities:
        if self._cached_probe is not None:
            return self._cached_probe
        if sys.platform != "win32":
            result = unavailable_capabilities("windows-appcontainer", "windows_helper_requires_windows")
        elif not self._helper_path.is_file():
            result = unavailable_capabilities("windows-appcontainer", "windows_helper_not_packaged")
        else:
            result = self._handshake()
        self._cached_probe = result
        return result

    def _handshake(self) -> SandboxCapabilities:
        nonce = secrets.token_hex(16)
        try:
            completed = subprocess.run(
                [str(self._helper_path), "--probe", "--nonce", nonce],
                check=False,
                capture_output=True,
                timeout=5,
            )
            if completed.returncode != 0:
                return unavailable_capabilities("windows-appcontainer", "windows_helper_probe_failed")
            frame = decode_frame(completed.stdout, expected_nonce=nonce)
            if frame.message_type != "hello":
                return unavailable_capabilities("windows-appcontainer", "windows_helper_invalid_handshake")
            raw = frame.payload.get("capabilities")
            if not isinstance(raw, Mapping):
                return unavailable_capabilities("windows-appcontainer", "windows_helper_invalid_capabilities")
            capabilities = SandboxCapabilities.from_dict(cast(Mapping[str, Any], raw))
            if capabilities.backend != "windows-appcontainer":
                return unavailable_capabilities("windows-appcontainer", "windows_helper_backend_mismatch")
            if capabilities.complete:
                return unavailable_capabilities(
                    "windows-appcontainer",
                    "windows_broker_run_not_implemented",
                )
            return replace(capabilities, last_probe_at=probe_timestamp())
        except (OSError, subprocess.SubprocessError, UnicodeError, ValueError):
            return unavailable_capabilities("windows-appcontainer", "windows_helper_handshake_failed")

    def prepare_launch(
        self,
        plan: SandboxPlan,
        *,
        env: Mapping[str, str] | None,
    ) -> SandboxLaunchSpec:
        del env
        self.validate_plan(plan)
        raise SandboxUnavailableError(
            "Windows broker execution is unavailable until the helper reports complete containment."
        )


__all__ = ["WindowsSandboxBackend", "packaged_helper_path"]
