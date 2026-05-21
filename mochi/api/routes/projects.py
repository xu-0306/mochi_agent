"""Project API routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from mochi.api.server import _get_config
from mochi.projects.store import ProjectStore

router = APIRouter(prefix="/v1", tags=["projects"])


class CreateProjectRequest(BaseModel):
    """Create project request."""

    name: str = Field(min_length=1)
    workspace_dir: str = Field(min_length=1)


class UpdateProjectRequest(BaseModel):
    """Update project request."""

    name: str | None = None
    workspace_dir: str | None = None


def _get_project_store(app: object, *, config: object | None = None) -> ProjectStore:
    existing = getattr(app.state, "project_store", None)
    if isinstance(existing, ProjectStore):
        return existing

    if config is None:
        raise RuntimeError("config is required when app.state.project_store is not set.")

    workspace_dir = getattr(config, "workspace_dir")
    store = ProjectStore(Path(workspace_dir).expanduser() / "projects.json")
    app.state.project_store = store
    return store


@router.get("/projects")
async def list_projects(request: Request) -> dict[str, object]:
    config = await _get_config(request.app)
    store = _get_project_store(request.app, config=config)
    return {"type": "projects", "items": await store.list_projects()}


@router.post("/projects")
async def create_project(request: Request, payload: CreateProjectRequest) -> dict[str, object]:
    config = await _get_config(request.app)
    store = _get_project_store(request.app, config=config)
    project = await store.create_project(
        name=payload.name,
        workspace_dir=payload.workspace_dir,
    )
    return {"type": "project", **project}


@router.get("/projects/{project_id}")
async def get_project(project_id: str, request: Request) -> dict[str, object]:
    config = await _get_config(request.app)
    store = _get_project_store(request.app, config=config)
    project = await store.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"type": "project", **project}


@router.patch("/projects/{project_id}")
async def update_project(
    project_id: str,
    request: Request,
    payload: UpdateProjectRequest,
) -> dict[str, object]:
    if payload.name is None and payload.workspace_dir is None:
        raise HTTPException(status_code=422, detail="No changes requested")

    config = await _get_config(request.app)
    store = _get_project_store(request.app, config=config)
    project = await store.update_project(
        project_id,
        name=payload.name,
        workspace_dir=payload.workspace_dir,
    )
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"type": "project", **project}


@router.delete("/projects/{project_id}")
async def delete_project(project_id: str, request: Request) -> dict[str, object]:
    config = await _get_config(request.app)
    store = _get_project_store(request.app, config=config)
    deleted = await store.delete_project(project_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Project not found")

    from mochi.api.routes.sessions import _clear_project_from_sessions

    await _clear_project_from_sessions(request.app, project_id, config=config)
    return {"type": "project", "project_id": project_id, "deleted": True}
