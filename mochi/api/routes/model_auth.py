"""Model auth management routes."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from mochi.api.server import _get_config
from mochi.auth.openai_codex import (
    OPENAI_CODEX_DEFAULT_PROFILE_ID,
    OpenAICodexAuthService,
)
from mochi.config.manager import save_config
from mochi.config.schema import MochiConfig

router = APIRouter(prefix="/v1/model-auth")


class ModelAuthProviderInfo(BaseModel):
    id: str
    label: str
    auth_modes: list[str] = Field(default_factory=list)
    supports_cli_import: bool = False


class ModelAuthProvidersResponse(BaseModel):
    type: str = "model_auth_providers"
    providers: list[ModelAuthProviderInfo] = Field(default_factory=list)


class ModelAuthProfilesResponse(BaseModel):
    type: str = "model_auth_profiles"
    profiles: list[dict[str, Any]] = Field(default_factory=list)


class OpenAICodexAuthStatusResponse(BaseModel):
    type: str = "openai_codex_auth_status"
    configured: bool
    status: str = "missing"
    active_profile_id: str | None = None
    default_profile_id: str | None = None
    profiles: list[dict[str, Any]] = Field(default_factory=list)
    last_refresh_error: str | None = None
    auth_mode: str = "oauth"
    cli_auth_state: str = "missing"
    cli_auth_mode: str | None = None
    cli_auth_can_import: bool = False
    cli_auth_message: str | None = None


class OpenAICodexImportResponse(BaseModel):
    type: str = "openai_codex_auth_import"
    profile: dict[str, Any]
    configured: bool = True


class OpenAICodexLoginStartRequest(BaseModel):
    frontend_origin: str | None = None


class OpenAICodexLoginStartResponse(BaseModel):
    type: str = "openai_codex_auth_login_start"
    auth_url: str
    callback_url: str
    flow_id: str
    expires_at: int
    callback_ready: bool = False
    guidance: list[str] = Field(default_factory=list)


class OpenAICodexLoginCompleteRequest(BaseModel):
    callback_url: str | None = None
    code: str | None = None
    state: str | None = None


class OpenAICodexLogoutResponse(BaseModel):
    type: str = "openai_codex_auth_logout"
    deleted: bool
    active_profile_id: str | None = None


def _auth_service(config: MochiConfig) -> OpenAICodexAuthService:
    return OpenAICodexAuthService(config.workspace_dir)


def _persist_config_if_possible(request: Request, config: MochiConfig) -> Path | None:
    config_path = getattr(request.app.state, "config_path", None)
    if config_path is None and getattr(request.app.state, "config_factory", None) is not None:
        return None
    return save_config(config, config_path)


def _set_active_profile(request: Request, config: MochiConfig, profile_id: str | None) -> MochiConfig:
    updated = config.model_copy(deep=True)
    updated.openai_codex.auth_profile_id = profile_id
    request.app.state.config = updated
    _persist_config_if_possible(request, updated)
    return updated


def _resolve_frontend_origin(request: Request, requested_origin: str | None = None) -> str | None:
    candidate = (requested_origin or request.headers.get("origin") or "").strip()
    if not candidate:
        return None
    if not (candidate.startswith("http://") or candidate.startswith("https://")):
        return None
    return candidate.rstrip("/")


def _oauth_guidance(callback_url: str, *, callback_ready: bool) -> list[str]:
    guidance = [
        f"OpenAI Codex browser OAuth redirects to {callback_url}.",
    ]
    if callback_ready:
        guidance.append("A local callback listener is ready. The popup should close automatically after sign-in.")
    else:
        guidance.append("Local callback binding was unavailable. After sign-in, copy the full callback URL from the failed browser page and paste it into Mochi.")
    guidance.append("If you are remote/headless, use Import Codex CLI Login instead of browser OAuth.")
    return guidance


def _extract_code_and_state(payload: OpenAICodexLoginCompleteRequest) -> tuple[str, str]:
    if payload.callback_url:
        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(payload.callback_url)
        query = parse_qs(parsed.query)
        code = (query.get("code") or [""])[0].strip()
        state = (query.get("state") or [""])[0].strip()
    else:
        code = (payload.code or "").strip()
        state = (payload.state or "").strip()
    if not code or not state:
        raise HTTPException(
            status_code=400,
            detail="Provide a callback URL or both code and state to complete OpenAI Codex login.",
        )
    return code, state


def _popup_html(
    *,
    status: str,
    message: str,
    frontend_origin: str | None,
    profile_id: str | None = None,
    http_status: int = 200,
) -> HTMLResponse:
    safe_message = html.escape(message)
    target_origin = json.dumps(frontend_origin or "*")
    payload = json.dumps(
        {
            "type": "mochi-openai-codex-auth-callback",
            "status": status,
            "message": message,
            "profile_id": profile_id,
        }
    )
    settings_url = (
        f"{frontend_origin}/settings?tab=model"
        if frontend_origin
        else "/settings?tab=model"
    )
    html_body = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>OpenAI Codex Login</title>
    <style>
      body {{
        margin: 0;
        font-family: ui-sans-serif, system-ui, sans-serif;
        background: #0f172a;
        color: #e2e8f0;
      }}
      main {{
        max-width: 560px;
        margin: 48px auto;
        padding: 24px;
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 16px;
        background: rgba(15, 23, 42, 0.92);
      }}
      a {{ color: #93c5fd; }}
      p {{ line-height: 1.5; }}
      code {{
        display: inline-block;
        padding: 2px 6px;
        border-radius: 6px;
        background: rgba(148, 163, 184, 0.15);
      }}
    </style>
  </head>
  <body>
    <main>
      <h1>OpenAI Codex Login {html.escape(status.title())}</h1>
      <p>{safe_message}</p>
      <p>You can close this window and return to <a href="{html.escape(settings_url)}">Mochi Settings</a>.</p>
    </main>
    <script>
      (() => {{
        const payload = {payload};
        try {{
          if (window.opener && !window.opener.closed) {{
            window.opener.postMessage(payload, {target_origin});
          }}
        }} catch (_error) {{
        }}
        if (payload.status === "success") {{
          window.setTimeout(() => window.close(), 150);
        }}
      }})();
    </script>
  </body>
</html>"""
    return HTMLResponse(content=html_body, status_code=http_status)


def _status_payload(config: MochiConfig) -> OpenAICodexAuthStatusResponse:
    service = _auth_service(config)
    profiles = [profile.model_dump(exclude_none=True) for profile in service.list_profiles()]
    cli_auth = service.inspect_codex_cli_login()
    active_profile_id = service.resolve_profile_id(config.openai_codex.auth_profile_id)
    active_profile = service.get_profile_summary(active_profile_id) if active_profile_id is not None else None
    default_profile_id = (
        OPENAI_CODEX_DEFAULT_PROFILE_ID
        if service.get_profile_summary(OPENAI_CODEX_DEFAULT_PROFILE_ID) is not None
        else None
    )
    return OpenAICodexAuthStatusResponse(
        configured=active_profile is not None,
        status=service.status_for_profile(active_profile_id),
        active_profile_id=active_profile_id,
        default_profile_id=default_profile_id,
        profiles=profiles,
        last_refresh_error=active_profile.last_refresh_error if active_profile is not None else None,
        cli_auth_state=cli_auth.state,
        cli_auth_mode=cli_auth.auth_mode,
        cli_auth_can_import=cli_auth.can_import,
        cli_auth_message=cli_auth.message,
    )


@router.get("/providers", response_model=ModelAuthProvidersResponse)
async def get_model_auth_providers() -> ModelAuthProvidersResponse:
    return ModelAuthProvidersResponse(
        providers=[
            ModelAuthProviderInfo(
                id="openai_codex",
                label="OpenAI Codex",
                auth_modes=["oauth"],
                supports_cli_import=True,
            )
        ]
    )


@router.get("/profiles", response_model=ModelAuthProfilesResponse)
async def get_model_auth_profiles(request: Request) -> ModelAuthProfilesResponse:
    config = await _get_config(request.app)
    service = _auth_service(config)
    return ModelAuthProfilesResponse(
        profiles=[profile.model_dump(exclude_none=True) for profile in service.list_profiles()]
    )


@router.post("/openai-codex/import-codex-cli", response_model=OpenAICodexImportResponse)
async def import_openai_codex_cli_login(request: Request) -> OpenAICodexImportResponse:
    config = await _get_config(request.app)
    service = _auth_service(config)
    try:
        profile = service.import_codex_cli_login()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    active_profile_id = service.resolve_profile_id(config.openai_codex.auth_profile_id) or profile.profile_id
    _set_active_profile(request, config, active_profile_id)
    return OpenAICodexImportResponse(
        profile=profile.model_dump(exclude_none=True),
    )


@router.post("/openai-codex/login", response_model=OpenAICodexLoginStartResponse)
async def start_openai_codex_browser_login(
    request: Request,
    payload: OpenAICodexLoginStartRequest,
) -> OpenAICodexLoginStartResponse:
    config = await _get_config(request.app)
    service = _auth_service(config)
    try:
        login = service.start_browser_oauth_login(
            frontend_origin=_resolve_frontend_origin(request, payload.frontend_origin),
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return OpenAICodexLoginStartResponse(
        auth_url=login["auth_url"],
        callback_url=login["callback_url"],
        flow_id=login["flow_id"],
        expires_at=login["expires_at"],
        callback_ready=bool(login.get("callback_ready")),
        guidance=_oauth_guidance(
            login["callback_url"],
            callback_ready=bool(login.get("callback_ready")),
        ),
    )


@router.get("/openai-codex/status", response_model=OpenAICodexAuthStatusResponse)
async def get_openai_codex_auth_status(request: Request) -> OpenAICodexAuthStatusResponse:
    config = await _get_config(request.app)
    return _status_payload(config)


@router.post("/openai-codex/refresh", response_model=OpenAICodexImportResponse)
async def refresh_openai_codex_auth(request: Request) -> OpenAICodexImportResponse:
    config = await _get_config(request.app)
    service = _auth_service(config)
    try:
        profile_id = service.resolve_profile_id(config.openai_codex.auth_profile_id)
        if profile_id is None:
            raise RuntimeError("No OpenAI Codex auth profile is available. Import Codex CLI login first.")
        refreshed = service.refresh_access_token(profile_id, force=True)
        profile = service.get_profile_summary(refreshed.profile_id)
        if profile is None:
            raise RuntimeError(f"OpenAI Codex auth profile {refreshed.profile_id!r} was not found after refresh.")
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    _set_active_profile(request, config, profile.profile_id)
    return OpenAICodexImportResponse(
        profile=profile.model_dump(exclude_none=True),
    )


@router.post("/openai-codex/complete", response_model=OpenAICodexImportResponse)
async def complete_openai_codex_browser_login(
    request: Request,
    payload: OpenAICodexLoginCompleteRequest,
) -> OpenAICodexImportResponse:
    config = await _get_config(request.app)
    service = _auth_service(config)
    code, state = _extract_code_and_state(payload)
    try:
        profile, _frontend_origin = service.complete_browser_oauth_login(
            code=code,
            state=state,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _set_active_profile(request, config, profile.profile_id)
    return OpenAICodexImportResponse(
        profile=profile.model_dump(exclude_none=True),
    )


@router.get("/openai-codex/callback", response_class=HTMLResponse)
async def openai_codex_oauth_callback(
    request: Request,
    code: str = "",
    state: str = "",
) -> HTMLResponse:
    config = await _get_config(request.app)
    service = _auth_service(config)
    try:
        profile, frontend_origin = service.complete_browser_oauth_login(
            code=code,
            state=state,
        )
    except RuntimeError as exc:
        pending = service._store.get_openai_codex_pending_flow_by_state(state.strip()) if state.strip() else None
        frontend_origin = pending.frontend_origin if pending is not None else _resolve_frontend_origin(request)
        return _popup_html(
            status="error",
            message=str(exc),
            frontend_origin=frontend_origin,
            http_status=400,
        )

    _set_active_profile(request, config, profile.profile_id)
    return _popup_html(
        status="success",
        message="OpenAI Codex login saved. You can return to Mochi.",
        frontend_origin=frontend_origin,
        profile_id=profile.profile_id,
    )


@router.post("/openai-codex/logout", response_model=OpenAICodexLogoutResponse)
async def logout_openai_codex_auth(request: Request) -> OpenAICodexLogoutResponse:
    config = await _get_config(request.app)
    service = _auth_service(config)
    active_profile_id = service.resolve_profile_id(config.openai_codex.auth_profile_id)
    deleted = service.logout(active_profile_id)

    next_profile_id = service.resolve_profile_id(config.openai_codex.auth_profile_id)
    updated = _set_active_profile(request, config, next_profile_id)
    return OpenAICodexLogoutResponse(
        deleted=deleted,
        active_profile_id=updated.openai_codex.auth_profile_id,
    )
