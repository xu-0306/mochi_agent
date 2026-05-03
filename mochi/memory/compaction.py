"""記憶壓縮摘要 — Phase 2 完整實作。"""

from __future__ import annotations


class MemoryCompactor:
    """當對話歷史過長時，用 LLM 摘要壓縮。Phase 2 完整實作。"""

    async def compact(self, messages: list, backend: object) -> list:
        """壓縮對話歷史。"""
        raise NotImplementedError("MemoryCompactor is planned for Phase 2.")
