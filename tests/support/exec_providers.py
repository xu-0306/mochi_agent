"""Deterministic shell providers for integration tests."""

from __future__ import annotations

import sys

from mochi.utils.shell_providers import BaseShellProvider, SubprocessSpec


class PythonDirectProvider(BaseShellProvider):
    """Run test commands directly through the active Python interpreter."""

    @property
    def canonical_name(self) -> str:
        return "test"

    @property
    def aliases(self) -> tuple[str, ...]:
        return ("test",)

    def build_subprocess_spec(self, command: str, *, tty: bool = False) -> SubprocessSpec:
        del tty
        return SubprocessSpec(executable=sys.executable, args=("-c", command))
