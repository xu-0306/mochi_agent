"""API 路由集合。"""

from __future__ import annotations

from mochi.api.routes.agent_runs import router as agent_runs_router
from mochi.api.routes.approvals import router as approvals_router
from mochi.api.routes.chat import router as chat_router
from mochi.api.routes.file_ops import router as file_ops_router
from mochi.api.routes.filesystem import router as filesystem_router
from mochi.api.routes.goals import router as goals_router
from mochi.api.routes.model_auth import router as model_auth_router
from mochi.api.routes.models import router as models_router
from mochi.api.routes.projects import router as projects_router
from mochi.api.routes.skills import router as skills_router
from mochi.api.routes.tasks import router as tasks_router
from mochi.api.routes.voice import router as voice_router
from mochi.api.routes.workspace import router as workspace_router

from .sessions import router as sessions_router
from .settings import router as settings_router

__all__ = [
    "agent_runs_router",
    "approvals_router",
    "chat_router",
    "file_ops_router",
    "filesystem_router",
    "goals_router",
    "model_auth_router",
    "models_router",
    "projects_router",
    "sessions_router",
    "settings_router",
    "skills_router",
    "tasks_router",
    "voice_router",
    "workspace_router",
]
