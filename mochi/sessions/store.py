"""會話 JSONL 持久化儲存 — Phase 2 完整實作。"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path

from mochi.config import defaults


class SessionStore:
    """會話儲存（JSONL 格式，Append-only）。"""

    def __init__(self, sessions_dir: str | Path = defaults.default_sessions_dir()) -> None:
        """初始化 SessionStore。

        Args:
            sessions_dir: 會話檔案目錄，會自動建立。
        """
        self._sessions_dir = Path(sessions_dir).expanduser()

    async def save_event(self, session_id: str, event: dict) -> None:
        """將事件追加寫入 JSONL 檔案。"""
        if not isinstance(event, dict):
            raise TypeError("event must be a dict.")

        line = json.dumps(event, ensure_ascii=False)
        path = self._session_path(session_id)
        await asyncio.to_thread(self._append_line, path, line)

    async def load_session(self, session_id: str) -> list[dict]:
        """從 JSONL 載入完整會話，遇到壞資料時跳過該行。"""
        path = self._session_path(session_id)
        return await asyncio.to_thread(self._load_lines, path)

    async def session_exists(self, session_id: str) -> bool:
        """檢查 session 檔案是否存在。"""
        path = self._session_path(session_id)
        return await asyncio.to_thread(path.exists)

    async def delete_session(self, session_id: str) -> bool:
        """刪除 session 檔案；不存在時回傳 False。"""
        path = self._session_path(session_id)
        return await asyncio.to_thread(self._delete_file, path)

    async def replace_session(self, session_id: str, events: list[dict]) -> None:
        """Atomically replace one session file with the provided ordered events."""
        if not isinstance(events, list):
            raise TypeError("events must be a list.")
        if any(not isinstance(event, dict) for event in events):
            raise TypeError("every event must be a dict.")

        path = self._session_path(session_id)
        await asyncio.to_thread(self._write_lines, path, events)

    def _session_path(self, session_id: str) -> Path:
        """根據 session_id 生成對應的 JSONL 檔案路徑。"""
        sid = session_id.strip()
        if not sid:
            raise ValueError("session_id must not be empty.")

        safe_sid = re.sub(r"[^A-Za-z0-9._-]", "_", sid)
        return self._sessions_dir / f"{safe_sid}.jsonl"

    def _append_line(self, path: Path, line: str) -> None:
        """同步追加寫入單行事件。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(f"{line}\n")

    def _delete_file(self, path: Path) -> bool:
        """同步刪除單一 session 檔案。"""
        if not path.exists():
            return False
        path.unlink()
        return True

    def _write_lines(self, path: Path, events: list[dict]) -> None:
        """Atomically replace the session file contents."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.tmp")
        with tmp_path.open("w", encoding="utf-8") as fh:
            for event in events:
                fh.write(json.dumps(event, ensure_ascii=False))
                fh.write("\n")
        os.replace(tmp_path, path)

    def _load_lines(self, path: Path) -> list[dict]:
        """同步讀取 JSONL 檔案，並跳過無法解析或格式不符的行。"""
        if not path.exists():
            return []

        events: list[dict] = []
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue

                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if isinstance(parsed, dict):
                    events.append(parsed)

        return events
