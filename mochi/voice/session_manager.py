"""語音會話管理器：依 `session_id` 隔離可重用的 VoiceSession。"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import Any, cast

from mochi.voice.voice_session import VoiceSession

VoiceSessionFactory = Callable[[], Awaitable[VoiceSession]]


class VoiceSessionManager:
    """管理 VoiceSession 的 lazy 建立與 session 隔離。"""

    def __init__(self, *, default_session_id: str = "default") -> None:
        self._default_session_id = default_session_id
        self._sessions: dict[str, VoiceSession] = {}
        self._lock = asyncio.Lock()

    def resolve_session_id(self, session_id: str | None) -> str:
        """將可選 session_id 轉為內部穩定 key。"""
        return session_id or self._default_session_id

    async def get_or_create(
        self,
        *,
        session_id: str | None,
        factory: VoiceSessionFactory,
    ) -> VoiceSession:
        """取得指定 session 的 VoiceSession，不存在則建立並快取。"""
        session_key = self.resolve_session_id(session_id)
        session = self._sessions.get(session_key)
        if session is not None:
            return session

        async with self._lock:
            session = self._sessions.get(session_key)
            if session is not None:
                return session

            created = await factory()
            self._sessions[session_key] = created
            return created

    async def release(
        self,
        *,
        session_id: str | None,
    ) -> bool:
        """釋放指定 session 的 VoiceSession 快取，若存在則回傳 True。"""
        session_key = self.resolve_session_id(session_id)

        async with self._lock:
            session = self._sessions.pop(session_key, None)

        if session is None:
            return False

        await _close_if_supported(session)
        return True

    async def discard(
        self,
        *,
        session_id: str | None,
    ) -> bool:
        """`release()` 的語意別名，用於明確表達丟棄租約。"""
        return await self.release(session_id=session_id)

    async def release_all(self) -> None:
        """釋放並關閉所有已快取的 VoiceSession。"""
        async with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()

        for session in sessions:
            await _close_if_supported(session)

    async def get_runtime_diagnostics(self) -> dict[str, Any]:
        """回傳所有已快取 VoiceSession 的聚合診斷摘要。"""
        async with self._lock:
            session_items = list(self._sessions.items())

        session_payloads: dict[str, dict[str, Any]] = {}
        active_preview_session_count = 0
        preview_disabled_session_count = 0
        watchdog_sessions_with_state = 0
        watchdog_reset_total = 0
        watchdog_runtime_rebuild_total = 0
        watchdog_last_reason_counts: dict[str, int] = {}

        for session_id, session in session_items:
            diagnostics = await _get_runtime_diagnostics_if_supported(session)
            if diagnostics is None:
                continue

            session_payloads[session_id] = diagnostics
            preview_payload = diagnostics.get("preview_session")
            if not isinstance(preview_payload, dict):
                continue

            if preview_payload.get("active") is True:
                active_preview_session_count += 1
            if diagnostics.get("preview_disabled") is True:
                preview_disabled_session_count += 1

            state = preview_payload.get("state")
            if not isinstance(state, dict):
                continue
            if "watchdog_resets" not in state:
                continue

            watchdog_sessions_with_state += 1
            watchdog_reset_total += int(state.get("watchdog_resets", 0))
            watchdog_runtime_rebuild_total += int(state.get("watchdog_runtime_rebuilds", 0))
            last_reason = state.get("watchdog_last_reason")
            if isinstance(last_reason, str) and last_reason:
                watchdog_last_reason_counts[last_reason] = (
                    watchdog_last_reason_counts.get(last_reason, 0) + 1
                )

        return {
            "cached_session_count": len(session_items),
            "active_preview_session_count": active_preview_session_count,
            "preview_disabled_session_count": preview_disabled_session_count,
            "watchdog": {
                "sessions_with_state": watchdog_sessions_with_state,
                "reset_total": watchdog_reset_total,
                "runtime_rebuild_total": watchdog_runtime_rebuild_total,
                "last_reason_counts": watchdog_last_reason_counts,
            },
            "sessions": session_payloads,
        }


async def _close_if_supported(session: VoiceSession) -> None:
    """若 VoiceSession 提供 close()，則安全執行。"""
    close = getattr(session, "close", None)
    if not callable(close):
        return

    maybe_awaitable = close()
    if inspect.isawaitable(maybe_awaitable):
        await cast(Awaitable[Any], maybe_awaitable)


async def _get_runtime_diagnostics_if_supported(session: VoiceSession) -> dict[str, Any] | None:
    """若 VoiceSession 提供 get_runtime_diagnostics()，則安全取得。"""
    getter = getattr(session, "get_runtime_diagnostics", None)
    if not callable(getter):
        return None

    maybe_awaitable = getter()
    if inspect.isawaitable(maybe_awaitable):
        diagnostics = await cast(Awaitable[Any], maybe_awaitable)
    else:
        diagnostics = maybe_awaitable

    if isinstance(diagnostics, dict):
        return dict(diagnostics)
    return None
