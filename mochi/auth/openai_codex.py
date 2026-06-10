"""OpenAI Codex auth import and lookup helpers."""

from __future__ import annotations

import base64
import hashlib
import html
import json
import os
import secrets
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

from pydantic import SecretStr

from .models import (
    AuthStoreData,
    OpenAICodexAuthProfile,
    OpenAICodexCliAuthDiagnostics,
    OpenAICodexPendingOAuthFlow,
    OpenAICodexProfileSummary,
)
from .store import AuthStore

OPENAI_CODEX_PROVIDER = "openai_codex"
OPENAI_CODEX_DEFAULT_PROFILE_ID = "openai_codex:default"
OPENAI_CODEX_AUTH_STORE_DIRNAME = ".mochi"
OPENAI_CODEX_AUTH_STORE_FILENAME = "auth.json"
OPENAI_CODEX_DEFAULT_BASE_URL = "https://chatgpt.com/backend-api"
OPENAI_CODEX_OAUTH_AUTHORIZE_ENDPOINT = "https://auth.openai.com/oauth/authorize"
OPENAI_CODEX_OAUTH_TOKEN_ENDPOINT = "https://auth.openai.com/oauth/token"
OPENAI_CODEX_DEFAULT_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
OPENAI_CODEX_OAUTH_SCOPE = (
    "openid profile email offline_access api.connectors.read api.connectors.invoke"
)
OPENAI_CODEX_OAUTH_LOCAL_BIND_HOST = "127.0.0.1"
OPENAI_CODEX_OAUTH_LOCAL_CALLBACK_HOST = "localhost"
OPENAI_CODEX_OAUTH_LOCAL_CALLBACK_PORT = 1455
OPENAI_CODEX_OAUTH_LOCAL_CALLBACK_PATH = "/auth/callback"
OPENAI_CODEX_OAUTH_LOCAL_CALLBACK_URL = (
    f"http://{OPENAI_CODEX_OAUTH_LOCAL_CALLBACK_HOST}:"
    f"{OPENAI_CODEX_OAUTH_LOCAL_CALLBACK_PORT}{OPENAI_CODEX_OAUTH_LOCAL_CALLBACK_PATH}"
)
OPENAI_CODEX_OAUTH_PENDING_FLOW_TTL_SECONDS = 600
OPENAI_CODEX_REFRESH_EARLY_SECONDS = 300
OPENAI_CODEX_REFRESH_LOCK_TIMEOUT_SECONDS = 30.0
OPENAI_CODEX_REFRESH_LOCK_STALE_SECONDS = 120.0
OPENAI_CODEX_REFRESH_LOCK_POLL_SECONDS = 0.1

_REFRESH_LOCK_GUARD = threading.Lock()
_REFRESH_LOCKS: dict[str, threading.Lock] = {}
_CALLBACK_SERVER_GUARD = threading.Lock()
_CALLBACK_SERVER: ThreadingHTTPServer | None = None
_CALLBACK_SERVER_THREAD: threading.Thread | None = None
_CALLBACK_SERVER_WORKSPACES: set[str] = set()


def resolve_auth_store_path(workspace_dir: str) -> Path:
    return (
        Path(workspace_dir).expanduser()
        / OPENAI_CODEX_AUTH_STORE_DIRNAME
        / OPENAI_CODEX_AUTH_STORE_FILENAME
    )


def resolve_legacy_auth_store_path(workspace_dir: str) -> Path:
    return Path(workspace_dir).expanduser() / OPENAI_CODEX_AUTH_STORE_FILENAME


def _decode_jwt_claims(token: str) -> dict[str, Any]:
    parts = token.split(".")
    if len(parts) < 2:
        return {}
    payload = parts[1].strip()
    if not payload:
        return {}
    payload += "=" * (-len(payload) % 4)
    try:
        decoded = base64.urlsafe_b64decode(payload.encode("utf-8"))
        data = json.loads(decoded.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _normalize_non_empty_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    trimmed = value.strip()
    return trimmed or None


def normalize_openai_codex_base_url(base_url: str | None) -> str:
    normalized = (base_url or OPENAI_CODEX_DEFAULT_BASE_URL).strip().rstrip("/")
    default_base_url = OPENAI_CODEX_DEFAULT_BASE_URL.rstrip("/")
    if normalized != default_base_url:
        raise ValueError(
            "OpenAI Codex base_url must match the official ChatGPT backend endpoint."
        )
    return default_base_url


def _resolve_codex_home(env: dict[str, str] | None = None) -> Path:
    source = env or dict(os.environ)
    configured = _normalize_non_empty_string(source.get("CODEX_HOME"))
    if configured is None:
        return Path.home() / ".codex"
    if configured == "~":
        return Path.home()
    if configured.startswith("~/"):
        return Path.home() / configured[2:]
    return Path(configured).expanduser()


def _current_timestamp() -> int:
    return int(datetime.now(tz=UTC).timestamp())


def _base64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _get_refresh_lock(profile_id: str) -> threading.Lock:
    with _REFRESH_LOCK_GUARD:
        lock = _REFRESH_LOCKS.get(profile_id)
        if lock is None:
            lock = threading.Lock()
            _REFRESH_LOCKS[profile_id] = lock
        return lock


def _profile_refresh_lock_path(store_path: Path, profile_id: str) -> Path:
    profile_hash = hashlib.sha256(profile_id.encode("utf-8")).hexdigest()[:16]
    return store_path.parent / "locks" / f"openai-codex-refresh-{profile_hash}.lock"


def _pending_flow_lock_path(store_path: Path, state: str) -> Path:
    state_hash = hashlib.sha256(state.encode("utf-8")).hexdigest()[:16]
    return store_path.parent / "locks" / f"openai-codex-oauth-flow-{state_hash}.lock"


@contextmanager
def _acquire_lock_file(
    lock_path: Path,
    *,
    timeout_seconds: float = OPENAI_CODEX_REFRESH_LOCK_TIMEOUT_SECONDS,
    stale_seconds: float = OPENAI_CODEX_REFRESH_LOCK_STALE_SECONDS,
    poll_seconds: float = OPENAI_CODEX_REFRESH_LOCK_POLL_SECONDS,
):
    deadline = time.monotonic() + timeout_seconds
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "pid": os.getpid(),
                            "created_at": _current_timestamp(),
                        },
                        ensure_ascii=False,
                    )
                )
            break
        except FileExistsError:
            try:
                age_seconds = time.time() - lock_path.stat().st_mtime
            except FileNotFoundError:
                continue
            if age_seconds >= stale_seconds:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Timed out waiting for OpenAI Codex refresh lock at {lock_path.name}."
                )
            time.sleep(poll_seconds)
    try:
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


@contextmanager
def _acquire_refresh_file_lock(lock_path: Path):
    with _acquire_lock_file(lock_path):
        yield


def _build_pkce_code_verifier() -> str:
    return secrets.token_urlsafe(64)


def _build_pkce_code_challenge(code_verifier: str) -> str:
    return _base64url_encode(hashlib.sha256(code_verifier.encode("utf-8")).digest())


def _extract_openai_account_id(*claims_sets: dict[str, Any]) -> str | None:
    for claims in claims_sets:
        for key in ("account_id", "chatgpt_account_id"):
            value = _normalize_non_empty_string(claims.get(key))
            if value is not None:
                return value
        nested = claims.get("https://api.openai.com/auth")
        if isinstance(nested, dict):
            for key in ("account_id", "chatgpt_account_id"):
                value = _normalize_non_empty_string(nested.get(key))
                if value is not None:
                    return value
    return None


def _callback_popup_html(
    *,
    status: str,
    message: str,
    frontend_origin: str | None,
    profile_id: str | None = None,
) -> bytes:
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
    document = f"""<!doctype html>
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
    return document.encode("utf-8")


class _OpenAICodexOAuthCallbackHandler(BaseHTTPRequestHandler):
    server_version = "MochiOpenAICodexOAuth/1.0"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != OPENAI_CODEX_OAUTH_LOCAL_CALLBACK_PATH:
            self.send_error(404)
            return
        query = parse_qs(parsed.query)
        code = (query.get("code") or [""])[0].strip()
        state = (query.get("state") or [""])[0].strip()
        error_message = (query.get("error_description") or query.get("error") or [""])[0].strip()
        if error_message:
            content = _callback_popup_html(
                status="error",
                message=error_message,
                frontend_origin=None,
            )
            self._write_html(400, content)
            return
        if not code or not state:
            content = _callback_popup_html(
                status="error",
                message="OpenAI Codex callback did not include both code and state.",
                frontend_origin=None,
            )
            self._write_html(400, content)
            return

        frontend_origin: str | None = None
        first_error = "OpenAI Codex login state is missing or has expired."
        for workspace_dir in list(_CALLBACK_SERVER_WORKSPACES):
            service = OpenAICodexAuthService(workspace_dir)
            try:
                profile, frontend_origin = service.complete_browser_oauth_login(
                    code=code,
                    state=state,
                )
            except RuntimeError as exc:
                first_error = str(exc)
                continue
            content = _callback_popup_html(
                status="success",
                message="OpenAI Codex login saved. You can return to Mochi.",
                frontend_origin=frontend_origin,
                profile_id=profile.profile_id,
            )
            self._write_html(200, content)
            return

        content = _callback_popup_html(
            status="error",
            message=first_error,
            frontend_origin=frontend_origin,
        )
        self._write_html(400, content)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        _ = (format, args)

    def _write_html(self, status_code: int, content: bytes) -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def _ensure_local_callback_server(workspace_dir: str) -> bool:
    global _CALLBACK_SERVER, _CALLBACK_SERVER_THREAD
    normalized_workspace_dir = str(Path(workspace_dir).resolve())
    with _CALLBACK_SERVER_GUARD:
        _CALLBACK_SERVER_WORKSPACES.add(normalized_workspace_dir)
        if _CALLBACK_SERVER is not None and _CALLBACK_SERVER_THREAD is not None:
            if _CALLBACK_SERVER_THREAD.is_alive():
                return True
            _CALLBACK_SERVER = None
            _CALLBACK_SERVER_THREAD = None
        try:
            server = ThreadingHTTPServer(
                (OPENAI_CODEX_OAUTH_LOCAL_BIND_HOST, OPENAI_CODEX_OAUTH_LOCAL_CALLBACK_PORT),
                _OpenAICodexOAuthCallbackHandler,
            )
        except OSError:
            return False
        server.daemon_threads = True
        thread = threading.Thread(
            target=server.serve_forever,
            name="mochi-openai-codex-oauth-callback",
            daemon=True,
        )
        thread.start()
        _CALLBACK_SERVER = server
        _CALLBACK_SERVER_THREAD = thread
        return True


class OpenAICodexAuthService:
    """Manage OpenAI Codex auth profiles stored outside config.yaml."""

    def __init__(self, workspace_dir: str, *, env: dict[str, str] | None = None) -> None:
        self._store = AuthStore(
            resolve_auth_store_path(workspace_dir),
            fallback_paths=[resolve_legacy_auth_store_path(workspace_dir)],
        )
        self._env = env

    @property
    def store_path(self) -> Path:
        return self._store.path

    def list_profiles(self) -> list[OpenAICodexProfileSummary]:
        return [self._to_summary(profile) for profile in self._store.list_openai_codex_profiles()]

    def get_profile(self, profile_id: str) -> OpenAICodexAuthProfile | None:
        return self._store.get_openai_codex_profile(profile_id)

    def get_profile_summary(self, profile_id: str) -> OpenAICodexProfileSummary | None:
        profile = self.get_profile(profile_id)
        if profile is None:
            return None
        return self._to_summary(profile)

    def resolve_profile_id(self, profile_id: str | None = None) -> str | None:
        requested_profile_id = _normalize_non_empty_string(profile_id)
        if requested_profile_id is not None:
            profile = self.get_profile(requested_profile_id)
            if profile is not None:
                return profile.profile_id
        profile = self.get_profile(OPENAI_CODEX_DEFAULT_PROFILE_ID)
        if profile is not None:
            return profile.profile_id
        profiles = self._store.list_openai_codex_profiles()
        return profiles[0].profile_id if profiles else None

    def resolve_access_token(self, profile_id: str | None = None) -> str:
        resolved_profile_id = self.resolve_profile_id(profile_id)
        if resolved_profile_id is None:
            raise RuntimeError("No OpenAI Codex auth profile is available.")
        profile = self.get_profile(resolved_profile_id)
        if profile is None:
            raise RuntimeError(f"OpenAI Codex auth profile {resolved_profile_id!r} was not found.")
        if self._needs_refresh(profile):
            profile = self.refresh_access_token(resolved_profile_id, force=False)
        return profile.access_token.get_secret_value()

    def import_codex_cli_login(self) -> OpenAICodexProfileSummary:
        profile = self._read_codex_cli_profile()
        self._store.upsert_openai_codex_profile(profile)
        return self._to_summary(profile)

    def inspect_codex_cli_login(self) -> OpenAICodexCliAuthDiagnostics:
        auth_path = _resolve_codex_home(self._env) / "auth.json"
        try:
            payload = json.loads(auth_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return OpenAICodexCliAuthDiagnostics(
                state="missing",
                message=(
                    "No Codex CLI auth file was found. Use Connect ChatGPT in Mochi, "
                    "or sign in to Codex CLI with ChatGPT OAuth before importing."
                ),
            )
        except json.JSONDecodeError:
            return OpenAICodexCliAuthDiagnostics(
                state="invalid_json",
                message=(
                    "The local Codex CLI auth file is not valid JSON. Sign in again, "
                    "or use Connect ChatGPT in Mochi instead of importing."
                ),
            )

        if not isinstance(payload, dict):
            return OpenAICodexCliAuthDiagnostics(
                state="invalid_payload",
                message=(
                    "The local Codex CLI auth file has an invalid structure. "
                    "Mochi can only import ChatGPT OAuth credentials from Codex CLI."
                ),
            )

        auth_mode = _normalize_non_empty_string(payload.get("auth_mode"))
        if auth_mode == "apikey":
            return OpenAICodexCliAuthDiagnostics(
                state="apikey",
                auth_mode=auth_mode,
                message=(
                    "The local Codex CLI is using API key mode. Mochi cannot import "
                    "API-key credentials for OpenAI Codex from .codex/auth.json. "
                    "Use Connect ChatGPT in Mochi, or sign in to Codex CLI with "
                    "ChatGPT OAuth and retry import."
                ),
            )
        if auth_mode != "chatgpt":
            return OpenAICodexCliAuthDiagnostics(
                state="unsupported_auth_mode",
                auth_mode=auth_mode,
                message=(
                    "The local Codex CLI auth file is not using ChatGPT OAuth. "
                    "Mochi can only import chatgpt OAuth credentials for OpenAI Codex."
                ),
            )

        tokens = payload.get("tokens")
        if not isinstance(tokens, dict):
            return OpenAICodexCliAuthDiagnostics(
                state="missing_tokens",
                auth_mode=auth_mode,
                message=(
                    "The local Codex CLI auth file is in chatgpt mode but does not "
                    "contain OAuth tokens. Sign in again, or use Connect ChatGPT in Mochi."
                ),
            )

        access_token = _normalize_non_empty_string(tokens.get("access_token"))
        refresh_token = _normalize_non_empty_string(tokens.get("refresh_token"))
        if access_token is None or refresh_token is None:
            return OpenAICodexCliAuthDiagnostics(
                state="missing_tokens",
                auth_mode=auth_mode,
                message=(
                    "The local Codex CLI auth file is in chatgpt mode but does not "
                    "contain usable OAuth tokens. Sign in again, or use Connect ChatGPT in Mochi."
                ),
            )

        return OpenAICodexCliAuthDiagnostics(
            state="ready",
            auth_mode=auth_mode,
            can_import=True,
            message="The local Codex CLI has ChatGPT OAuth tokens and can be imported into Mochi.",
        )

    def refresh_from_codex_cli_login(self) -> OpenAICodexProfileSummary:
        return self.import_codex_cli_login()

    def refresh_access_token(
        self,
        profile_id: str | None = None,
        *,
        force: bool = True,
    ) -> OpenAICodexAuthProfile:
        resolved_profile_id = self.resolve_profile_id(profile_id)
        if resolved_profile_id is None:
            raise RuntimeError("No OpenAI Codex auth profile is available.")
        lock = _get_refresh_lock(resolved_profile_id)
        with lock:
            with _acquire_refresh_file_lock(
                _profile_refresh_lock_path(self._store.path, resolved_profile_id)
            ):
                profile = self.get_profile(resolved_profile_id)
                if profile is None:
                    raise RuntimeError(f"OpenAI Codex auth profile {resolved_profile_id!r} was not found.")
                if not force and not self._needs_refresh(profile):
                    return profile

                try:
                    token_payload = self._request_token_refresh(profile.refresh_token.get_secret_value())
                except RuntimeError as exc:
                    profile.last_refresh_error = str(exc)
                    self._store.upsert_openai_codex_profile(profile)
                    raise RuntimeError(
                        f"OpenAI Codex auth refresh failed for profile {resolved_profile_id}: {exc}"
                    ) from exc

                refreshed_profile = self._apply_refresh_payload(profile, token_payload)
                self._store.upsert_openai_codex_profile(refreshed_profile)
                return refreshed_profile

    def logout(self, profile_id: str | None = None) -> bool:
        resolved_profile_id = self.resolve_profile_id(profile_id)
        if resolved_profile_id is None:
            return False
        return self._store.delete_openai_codex_profile(resolved_profile_id)

    def start_browser_oauth_login(
        self,
        *,
        frontend_origin: str | None = None,
    ) -> dict[str, Any]:
        self._prune_expired_pending_flows()

        now_ts = _current_timestamp()
        expires_at = now_ts + OPENAI_CODEX_OAUTH_PENDING_FLOW_TTL_SECONDS
        state = secrets.token_urlsafe(32)
        code_verifier = _build_pkce_code_verifier()
        callback_ready = _ensure_local_callback_server(str(self._store.path.parent.parent))
        pending_flow = OpenAICodexPendingOAuthFlow(
            flow_id=secrets.token_urlsafe(16),
            state=state,
            code_verifier=SecretStr(code_verifier),
            redirect_uri=OPENAI_CODEX_OAUTH_LOCAL_CALLBACK_URL,
            frontend_origin=_normalize_non_empty_string(frontend_origin),
            created_at=now_ts,
            expires_at=expires_at,
        )
        self._store.upsert_openai_codex_pending_flow(pending_flow)

        auth_url = self._build_authorize_url(
            redirect_uri=OPENAI_CODEX_OAUTH_LOCAL_CALLBACK_URL,
            state=state,
            code_verifier=code_verifier,
        )
        return {
            "flow_id": pending_flow.flow_id,
            "auth_url": auth_url,
            "callback_url": OPENAI_CODEX_OAUTH_LOCAL_CALLBACK_URL,
            "expires_at": expires_at,
            "frontend_origin": pending_flow.frontend_origin,
            "callback_ready": callback_ready,
        }

    def complete_browser_oauth_login(
        self,
        *,
        code: str,
        state: str,
    ) -> tuple[OpenAICodexProfileSummary, str | None]:
        normalized_code = _normalize_non_empty_string(code)
        normalized_state = _normalize_non_empty_string(state)
        if normalized_code is None or normalized_state is None:
            raise RuntimeError("OpenAI Codex browser login requires both code and state.")

        self._prune_expired_pending_flows()
        pending_flow = self._store.get_openai_codex_pending_flow_by_state(normalized_state)
        if pending_flow is None:
            raise RuntimeError("OpenAI Codex login state is missing or has expired.")

        with _acquire_lock_file(_pending_flow_lock_path(self._store.path, normalized_state)):
            pending_flow = self._store.get_openai_codex_pending_flow_by_state(normalized_state)
            if pending_flow is None:
                raise RuntimeError("OpenAI Codex login state is no longer available.")
            if pending_flow.expires_at <= _current_timestamp():
                self._prune_expired_pending_flows()
                raise RuntimeError("OpenAI Codex login state has expired. Start the browser login again.")

            token_payload = self._request_authorization_code_tokens(
                code=normalized_code,
                redirect_uri=pending_flow.redirect_uri,
                code_verifier=pending_flow.code_verifier.get_secret_value(),
            )
            profile = self._profile_from_oauth_token_payload(
                token_payload,
                source="browser_oauth",
            )
            frontend_origin = pending_flow.frontend_origin

            def _mutate(data: AuthStoreData) -> None:
                data.openai_codex_profiles = [
                    profile,
                    *[
                        item
                        for item in data.openai_codex_profiles
                        if item.profile_id != profile.profile_id
                    ],
                ]
                data.openai_codex_pending_flows = [
                    item
                    for item in data.openai_codex_pending_flows
                    if item.flow_id != pending_flow.flow_id
                ]

            self._store.mutate(_mutate)
            return self._to_summary(profile), frontend_origin

    def status_for_profile(self, profile_id: str | None = None) -> str:
        resolved_profile_id = self.resolve_profile_id(profile_id)
        if resolved_profile_id is None:
            return "missing"
        profile = self.get_profile(resolved_profile_id)
        if profile is None:
            return "missing"
        return self._profile_status(profile)

    def record_auth_failure(
        self,
        profile_id: str | None,
        message: str,
        *,
        expire_now: bool = False,
    ) -> OpenAICodexAuthProfile | None:
        resolved_profile_id = self.resolve_profile_id(profile_id)
        if resolved_profile_id is None:
            return None
        profile = self.get_profile(resolved_profile_id)
        if profile is None:
            return None
        now_ts = _current_timestamp()
        updates: dict[str, Any] = {"last_refresh_error": message}
        if expire_now:
            updates["expires_at"] = now_ts - 1
        updated = profile.model_copy(update=updates)
        self._store.upsert_openai_codex_profile(updated)
        return updated

    def _read_codex_cli_profile(self) -> OpenAICodexAuthProfile:
        auth_path = _resolve_codex_home(self._env) / "auth.json"
        try:
            payload = json.loads(auth_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Codex CLI auth file was not found at {auth_path}. "
                "Use Connect ChatGPT in Mochi, or sign in to Codex CLI with ChatGPT OAuth first."
            ) from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Codex CLI auth file at {auth_path} is not valid JSON. "
                "Sign in again, or use Connect ChatGPT in Mochi instead of importing."
            ) from exc

        if not isinstance(payload, dict):
            raise RuntimeError("Codex CLI auth file payload is invalid.")
        auth_mode = _normalize_non_empty_string(payload.get("auth_mode"))
        if auth_mode == "apikey":
            raise RuntimeError(
                "Found Codex CLI credentials in API key mode. Mochi cannot import "
                "API-key credentials for OpenAI Codex from .codex/auth.json. "
                "Use Connect ChatGPT in Mochi, or sign in to Codex CLI with ChatGPT OAuth and retry import."
            )
        if auth_mode != "chatgpt":
            raise RuntimeError(
                f"Codex CLI auth file is using unsupported auth_mode {auth_mode!r}. "
                "Mochi can only import ChatGPT OAuth credentials."
            )

        tokens = payload.get("tokens")
        if not isinstance(tokens, dict):
            raise RuntimeError(
                "Codex CLI auth file is in chatgpt mode but does not contain OAuth tokens."
            )

        access_token = _normalize_non_empty_string(tokens.get("access_token"))
        refresh_token = _normalize_non_empty_string(tokens.get("refresh_token"))
        if access_token is None or refresh_token is None:
            raise RuntimeError(
                "Codex CLI auth file is in chatgpt mode but does not contain usable OAuth tokens."
            )

        id_token = _normalize_non_empty_string(tokens.get("id_token"))
        account_id = _normalize_non_empty_string(tokens.get("account_id"))

        access_claims = _decode_jwt_claims(access_token)
        id_claims = _decode_jwt_claims(id_token) if id_token else {}
        email = _normalize_non_empty_string(id_claims.get("email")) or _normalize_non_empty_string(access_claims.get("email"))
        display_name = (
            _normalize_non_empty_string(id_claims.get("name"))
            or _normalize_non_empty_string(access_claims.get("name"))
            or email
        )
        account_id = account_id or _extract_openai_account_id(id_claims, access_claims)
        expires_at = access_claims.get("exp")
        expires_at_value = int(expires_at) if isinstance(expires_at, (int, float)) else None

        now_ts = _current_timestamp()
        return OpenAICodexAuthProfile(
            profile_id=OPENAI_CODEX_DEFAULT_PROFILE_ID,
            access_token=access_token,
            refresh_token=refresh_token,
            id_token=id_token,
            account_id=account_id,
            email=email.lower() if email else None,
            display_name=display_name,
            expires_at=expires_at_value,
            imported_at=now_ts,
            last_refresh_at=now_ts,
            last_refresh_error=None,
            source_path=str(auth_path),
        )

    def _build_authorize_url(
        self,
        *,
        redirect_uri: str,
        state: str,
        code_verifier: str,
    ) -> str:
        client_id = _normalize_non_empty_string((self._env or os.environ).get("CODEX_CLIENT_ID"))
        params = {
            "response_type": "code",
            "client_id": client_id or OPENAI_CODEX_DEFAULT_CLIENT_ID,
            "redirect_uri": redirect_uri,
            "scope": OPENAI_CODEX_OAUTH_SCOPE,
            "state": state,
            "code_challenge": _build_pkce_code_challenge(code_verifier),
            "code_challenge_method": "S256",
            "id_token_add_organizations": "true",
            "codex_cli_simplified_flow": "true",
        }
        return f"{OPENAI_CODEX_OAUTH_AUTHORIZE_ENDPOINT}?{urlencode(params)}"

    def _request_token_payload(self, payload: dict[str, str]) -> dict[str, Any]:
        import httpx

        try:
            with httpx.Client(timeout=20.0, follow_redirects=True) as client:
                response = client.post(
                    OPENAI_CODEX_OAUTH_TOKEN_ENDPOINT,
                    data=payload,
                    headers={"Accept": "application/json"},
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            detail = ""
            try:
                error_payload = exc.response.json()
            except ValueError:
                error_payload = None
            if isinstance(error_payload, dict):
                detail = _normalize_non_empty_string(error_payload.get("error_description")) or _normalize_non_empty_string(error_payload.get("error")) or ""
            raise RuntimeError(
                detail or f"token endpoint returned HTTP {exc.response.status_code}"
            ) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f"token endpoint request failed: {exc}") from exc
        except ValueError as exc:
            raise RuntimeError("token endpoint returned invalid JSON") from exc

        if not isinstance(data, dict):
            raise RuntimeError("token endpoint returned an invalid payload")
        return data

    def _request_token_refresh(self, refresh_token: str) -> dict[str, Any]:
        client_id = _normalize_non_empty_string((self._env or os.environ).get("CODEX_CLIENT_ID"))
        return self._request_token_payload(
            {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": client_id or OPENAI_CODEX_DEFAULT_CLIENT_ID,
            }
        )

    def _request_authorization_code_tokens(
        self,
        *,
        code: str,
        redirect_uri: str,
        code_verifier: str,
    ) -> dict[str, Any]:
        client_id = _normalize_non_empty_string((self._env or os.environ).get("CODEX_CLIENT_ID"))
        return self._request_token_payload(
            {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": client_id or OPENAI_CODEX_DEFAULT_CLIENT_ID,
                "code_verifier": code_verifier,
            }
        )

    def _profile_from_oauth_token_payload(
        self,
        payload: dict[str, Any],
        *,
        source: str,
    ) -> OpenAICodexAuthProfile:
        access_token = _normalize_non_empty_string(payload.get("access_token"))
        refresh_token = _normalize_non_empty_string(payload.get("refresh_token"))
        if access_token is None or refresh_token is None:
            raise RuntimeError("token endpoint did not return usable OAuth tokens")
        id_token = _normalize_non_empty_string(payload.get("id_token"))
        access_claims = _decode_jwt_claims(access_token)
        id_claims = _decode_jwt_claims(id_token) if id_token else {}
        email = (
            _normalize_non_empty_string(id_claims.get("email"))
            or _normalize_non_empty_string(access_claims.get("email"))
        )
        display_name = (
            _normalize_non_empty_string(id_claims.get("name"))
            or _normalize_non_empty_string(access_claims.get("name"))
            or email
        )
        account_id = _extract_openai_account_id(id_claims, access_claims)
        expires_at = access_claims.get("exp")
        expires_at_value = int(expires_at) if isinstance(expires_at, (int, float)) else None
        now_ts = _current_timestamp()
        return OpenAICodexAuthProfile(
            profile_id=OPENAI_CODEX_DEFAULT_PROFILE_ID,
            source=source,
            access_token=SecretStr(access_token),
            refresh_token=SecretStr(refresh_token),
            id_token=SecretStr(id_token) if id_token is not None else None,
            account_id=account_id,
            email=email.lower() if email else None,
            display_name=display_name,
            expires_at=expires_at_value,
            imported_at=now_ts,
            last_refresh_at=now_ts,
            last_refresh_error=None,
        )

    def _apply_refresh_payload(
        self,
        profile: OpenAICodexAuthProfile,
        payload: dict[str, Any],
    ) -> OpenAICodexAuthProfile:
        access_token = _normalize_non_empty_string(payload.get("access_token"))
        if access_token is None:
            raise RuntimeError("token endpoint did not return an access_token")
        refresh_token = (
            _normalize_non_empty_string(payload.get("refresh_token"))
            or profile.refresh_token.get_secret_value()
        )
        id_token = _normalize_non_empty_string(payload.get("id_token")) or (
            profile.id_token.get_secret_value() if profile.id_token is not None else None
        )
        access_claims = _decode_jwt_claims(access_token)
        id_claims = _decode_jwt_claims(id_token) if id_token else {}
        email = (
            _normalize_non_empty_string(id_claims.get("email"))
            or _normalize_non_empty_string(access_claims.get("email"))
            or profile.email
        )
        display_name = (
            _normalize_non_empty_string(id_claims.get("name"))
            or _normalize_non_empty_string(access_claims.get("name"))
            or profile.display_name
            or email
        )
        expires_at = access_claims.get("exp")
        expires_at_value = int(expires_at) if isinstance(expires_at, (int, float)) else profile.expires_at
        account_id = _extract_openai_account_id(id_claims, access_claims) or profile.account_id
        now_ts = _current_timestamp()
        return profile.model_copy(
            update={
                "access_token": SecretStr(access_token),
                "refresh_token": SecretStr(refresh_token),
                "id_token": SecretStr(id_token) if id_token is not None else None,
                "account_id": account_id,
                "email": email.lower() if isinstance(email, str) else email,
                "display_name": display_name,
                "expires_at": expires_at_value,
                "last_refresh_at": now_ts,
                "last_refresh_error": None,
            }
        )

    def _needs_refresh(self, profile: OpenAICodexAuthProfile) -> bool:
        if profile.expires_at is None:
            return False
        return profile.expires_at <= (_current_timestamp() + OPENAI_CODEX_REFRESH_EARLY_SECONDS)

    def _prune_expired_pending_flows(self) -> None:
        now_ts = _current_timestamp()

        def _mutate(data: AuthStoreData) -> None:
            data.openai_codex_pending_flows = [
                flow
                for flow in data.openai_codex_pending_flows
                if flow.expires_at > now_ts
            ]

        self._store.mutate(_mutate)

    def _profile_status(self, profile: OpenAICodexAuthProfile) -> str:
        if profile.last_refresh_error and self._needs_refresh(profile):
            return "refresh_failed"
        if profile.expires_at is None:
            return "ready"
        remaining_seconds = profile.expires_at - _current_timestamp()
        if remaining_seconds <= 0:
            return "expired"
        if remaining_seconds <= OPENAI_CODEX_REFRESH_EARLY_SECONDS:
            return "expiring"
        return "ready"

    def _to_summary(self, profile: OpenAICodexAuthProfile) -> OpenAICodexProfileSummary:
        return OpenAICodexProfileSummary(
            profile_id=profile.profile_id,
            source=profile.source,
            account_id=profile.account_id,
            email=profile.email,
            display_name=profile.display_name,
            expires_at=profile.expires_at,
            imported_at=profile.imported_at,
            last_refresh_at=profile.last_refresh_at,
            last_refresh_error=profile.last_refresh_error,
            status=self._profile_status(profile),
        )
