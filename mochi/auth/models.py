"""Structured auth models stored outside config.yaml."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, SecretStr

OpenAICodexAuthProfileStatus = Literal["ready", "expiring", "expired", "refresh_failed"]
OpenAICodexCliAuthState = Literal[
    "missing",
    "invalid_json",
    "invalid_payload",
    "unsupported_auth_mode",
    "apikey",
    "missing_tokens",
    "ready",
]


class OpenAICodexAuthProfile(BaseModel):
    """Stored OpenAI Codex OAuth credential."""

    profile_id: str = Field(min_length=1)
    provider: Literal["openai_codex"] = "openai_codex"
    auth_mode: Literal["oauth"] = "oauth"
    source: Literal["codex_cli", "browser_oauth"] = "codex_cli"
    access_token: SecretStr
    refresh_token: SecretStr
    id_token: SecretStr | None = None
    account_id: str | None = None
    email: str | None = None
    display_name: str | None = None
    expires_at: int | None = None
    imported_at: int | None = None
    last_refresh_at: int | None = None
    last_refresh_error: str | None = None
    source_path: str | None = None


class OpenAICodexPendingOAuthFlow(BaseModel):
    """Persisted pending OAuth flow state for multi-worker-safe browser login."""

    flow_id: str = Field(min_length=1)
    provider: Literal["openai_codex"] = "openai_codex"
    auth_mode: Literal["oauth"] = "oauth"
    state: str = Field(min_length=1)
    code_verifier: SecretStr
    redirect_uri: str = Field(min_length=1)
    frontend_origin: str | None = None
    created_at: int
    expires_at: int


class AuthStoreData(BaseModel):
    """Versioned auth store payload."""

    version: int = 1
    openai_codex_profiles: list[OpenAICodexAuthProfile] = Field(default_factory=list)
    openai_codex_pending_flows: list[OpenAICodexPendingOAuthFlow] = Field(
        default_factory=list
    )


class OpenAICodexProfileSummary(BaseModel):
    """Sanitized OpenAI Codex auth profile summary."""

    profile_id: str
    provider: Literal["openai_codex"] = "openai_codex"
    auth_mode: Literal["oauth"] = "oauth"
    source: Literal["codex_cli", "browser_oauth"] = "codex_cli"
    account_id: str | None = None
    email: str | None = None
    display_name: str | None = None
    expires_at: int | None = None
    imported_at: int | None = None
    last_refresh_at: int | None = None
    last_refresh_error: str | None = None
    status: OpenAICodexAuthProfileStatus = "ready"


class OpenAICodexCliAuthDiagnostics(BaseModel):
    """Safe diagnostics for the local Codex CLI auth state."""

    state: OpenAICodexCliAuthState
    auth_mode: str | None = None
    can_import: bool = False
    message: str
