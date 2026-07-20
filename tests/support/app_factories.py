"""Factories for isolated FastAPI/runtime integration test applications."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from mochi.api.server import create_app
from mochi.config.schema import MochiConfig
from mochi.runtime.approvals import ApprovalStore
from mochi.runtime.exec_runtime import ExecRuntime
from mochi.runtime.service import RuntimeService
from mochi.runtime.store import RuntimeStore


def create_runtime_test_app(
    sessions_dir: Path,
    *,
    engine: Any | None = None,
    engine_factory: Callable[[], Any] | None = None,
    exec_approval_store: ApprovalStore | None = None,
    exec_runtime: ExecRuntime | None = None,
    active_goal_turn_selector: Any | None = None,
    scheduler_poll_interval: float | None = 0.05,
) -> tuple[Any, RuntimeService]:
    """Build a fresh app and runtime service backed by ``sessions_dir``."""
    app = create_app()
    runtime_service = RuntimeService(
        engine=object() if engine is None else engine,
        store=RuntimeStore(sessions_dir / "runtime.db"),
        exec_approval_store=exec_approval_store,
        exec_runtime=exec_runtime,
        active_goal_turn_selector=active_goal_turn_selector,
    )
    if scheduler_poll_interval is not None:
        runtime_service.set_scheduler_poll_interval(scheduler_poll_interval)
    app.state.runtime_service = runtime_service
    app.state.engine_factory = engine_factory or (lambda: object())
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(sessions_dir)}
    )
    return app, runtime_service
