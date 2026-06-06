"""Runtime delegate hooks for chat-triggered subagent tasks."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

DelegateSubagentTaskLauncher = Callable[..., Awaitable[dict[str, Any]]]

_delegate_subagent_task_launcher: DelegateSubagentTaskLauncher | None = None


def set_delegate_subagent_task_launcher(
    launcher: DelegateSubagentTaskLauncher | None,
) -> None:
    """Register the current runtime service delegate launcher."""
    global _delegate_subagent_task_launcher
    _delegate_subagent_task_launcher = launcher


def get_delegate_subagent_task_launcher() -> DelegateSubagentTaskLauncher | None:
    """Return the registered delegate launcher, if runtime service is active."""
    return _delegate_subagent_task_launcher
