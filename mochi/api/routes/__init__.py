"""API route package exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_ROUTER_MODULES = {
    "agent_runs_router": "mochi.api.routes.agent_runs",
    "approvals_router": "mochi.api.routes.approvals",
    "chat_router": "mochi.api.routes.chat",
    "file_ops_router": "mochi.api.routes.file_ops",
    "filesystem_router": "mochi.api.routes.filesystem",
    "goals_router": "mochi.api.routes.goals",
    "model_auth_router": "mochi.api.routes.model_auth",
    "models_router": "mochi.api.routes.models",
    "projects_router": "mochi.api.routes.projects",
    "sessions_router": "mochi.api.routes.sessions",
    "settings_router": "mochi.api.routes.settings",
    "skills_router": "mochi.api.routes.skills",
    "tasks_router": "mochi.api.routes.tasks",
    "voice_router": "mochi.api.routes.voice",
    "workspace_router": "mochi.api.routes.workspace",
}

__all__ = list(_ROUTER_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _ROUTER_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(module_name)
    return getattr(module, "router")
