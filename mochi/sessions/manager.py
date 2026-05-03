"""會話生命週期管理 — Phase 2 完整實作。"""

from __future__ import annotations


class SessionManager:
    """會話管理器骨架。Phase 2 完整實作。"""

    async def create_session(self, session_id: str | None = None) -> str:
        """建立新會話，回傳 session_id。Phase 2 實作。"""
        raise NotImplementedError("SessionManager is planned for Phase 2.")

    async def get_session(self, session_id: str) -> dict | None:
        """取得會話資訊。Phase 2 實作。"""
        raise NotImplementedError("SessionManager is planned for Phase 2.")
