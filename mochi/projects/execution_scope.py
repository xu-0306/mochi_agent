"""Shared session -> project -> workspace resolution."""

from __future__ import annotations

from dataclasses import dataclass

from mochi.projects.store import ProjectStore
from mochi.sessions.store import SessionStore


@dataclass(frozen=True)
class ExecutionScope:
    """Resolved scope for one request or engine turn."""

    project_id: str | None
    workspace_dir: str


class ExecutionScopeResolver:
    """Resolve the effective workspace from session/project state."""

    def __init__(
        self,
        *,
        default_workspace_dir: str,
        session_store: SessionStore,
        project_store: ProjectStore,
    ) -> None:
        self._default_workspace_dir = str(default_workspace_dir)
        self._session_store = session_store
        self._project_store = project_store

    async def resolve(
        self,
        *,
        session_id: str,
        project_id: str | None = None,
        workspace_dir: str | None = None,
    ) -> ExecutionScope:
        if isinstance(workspace_dir, str) and workspace_dir.strip():
            return ExecutionScope(
                project_id=project_id,
                workspace_dir=workspace_dir.strip(),
            )

        resolved_project_id = project_id or await self._resolve_session_project_id(session_id)
        if resolved_project_id is None:
            return ExecutionScope(
                project_id=None,
                workspace_dir=self._default_workspace_dir,
            )

        project = await self._project_store.get_project(resolved_project_id)
        if project is None:
            raise LookupError("Project not found")
        return ExecutionScope(
            project_id=resolved_project_id,
            workspace_dir=project["workspace_dir"],
        )

    async def _resolve_session_project_id(self, session_id: str) -> str | None:
        if not await self._session_store.session_exists(session_id):
            return None

        events = await self._session_store.load_session(session_id)
        for event in reversed(events):
            if (
                event.get("type") == "session_meta"
                and event.get("event") == "project_assigned"
            ):
                raw = event.get("project_id")
                if isinstance(raw, str) and raw.strip():
                    return raw.strip()
                return None
        return None
