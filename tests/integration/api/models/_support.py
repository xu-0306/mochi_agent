# ruff: noqa: F401,F403
"""Model API integration support."""

from __future__ import annotations

from tests.integration.api.chat import _support as _chat_support
from tests.integration.api.chat._support import *

class _FakeManagedVLLMRuntimeManager:
    def __init__(self, *, base_url: str = "http://localhost:8000/v1") -> None:
        self.base_url = base_url
        self.running = False
        self.active_model_id: str | None = None
        self.active_model_spec: str | None = None
        self.start_calls: list[tuple[str | None, str, str, str]] = []
        self.stop_calls = 0

    async def status(self, **_: Any) -> dict[str, Any]:
        return {
            "state": "running" if self.running else "stopped",
            "running": self.running,
            "launch_mode": "managed",
            "active_model_id": self.active_model_id,
            "active_model_spec": self.active_model_spec,
            "base_url": self.base_url,
        }

    async def start(
        self,
        *,
        model_id: str | None = None,
        model_spec: str,
        base_url: str | None = None,
        launch_mode: str = "managed",
        **_: Any,
    ) -> dict[str, Any]:
        self.running = True
        self.active_model_id = model_id
        self.active_model_spec = model_spec
        self.base_url = base_url or self.base_url
        self.start_calls.append((model_id, model_spec, self.base_url, launch_mode))
        return await self.status()

    async def stop(self, **_: Any) -> dict[str, Any]:
        self.stop_calls += 1
        self.running = False
        self.active_model_id = None
        self.active_model_spec = None
        return await self.status()

def _fake_jwt(exp: int, *, email: str = "codex@example.com", name: str | None = None) -> str:
    payload: dict[str, Any] = {"exp": exp, "email": email}
    if name is not None:
        payload["name"] = name
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).decode("utf-8").rstrip("=")
    return f"header.{encoded}.sig"

__all__ = [*_chat_support.__all__, '_FakeManagedVLLMRuntimeManager', '_fake_jwt']
