"""SQLite 技能庫（FTS5 搜尋）— Phase 5A。"""

from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

from mochi.learning.types import Skill, Trajectory


class SkillLibrary:
    """階層式技能庫，使用 SQLite 儲存並以 FTS5 搜尋。"""

    _JSON_FIELDS = {"trigger_keywords", "steps", "tools_used", "metadata"}

    def __init__(self, db_path: str | Path | None = None) -> None:
        """建立技能庫連線；未指定路徑時使用 in-memory DB。"""
        self.db_path = Path(db_path) if db_path is not None else None
        if self.db_path is not None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path) if self.db_path else ":memory:")
        self._conn.row_factory = sqlite3.Row
        self._fts_enabled = False
        self._init_schema()

    async def add(self, skill: Skill) -> str:
        """新增技能，回傳 skill_id。"""
        now = time.time()
        skill_id = skill.skill_id or str(uuid.uuid4())
        stored = replace(
            skill,
            skill_id=skill_id,
            created_at=skill.created_at or now,
            updated_at=skill.updated_at or now,
        )
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO skills (
                    skill_id, name, description, trigger_keywords, preconditions,
                    steps, tools_used, source_trajectory_id, times_used,
                    success_rate, created_at, updated_at, version, source_type,
                    source_path, content_hash, body, metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._skill_values(stored),
            )
            self._sync_fts(stored)
        return skill_id

    async def get(self, skill_id: str) -> Skill | None:
        """依 ID 取得技能。"""
        row = self._conn.execute(
            "SELECT * FROM skills WHERE skill_id = ?",
            (skill_id,),
        ).fetchone()
        return self._row_to_skill(row) if row else None

    async def list(self, limit: int | None = None) -> list[Skill]:
        """列出技能，依更新時間由新到舊排序。"""
        sql = "SELECT * FROM skills ORDER BY updated_at DESC, created_at DESC, skill_id ASC"
        params: tuple[Any, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_skill(row) for row in rows]

    async def search(self, query: str, top_k: int = 3) -> list[Skill]:
        """全文搜尋相關技能。"""
        normalized = query.strip()
        if not normalized:
            return await self.list(limit=top_k)
        if self._fts_enabled:
            try:
                return self._search_fts(normalized, top_k)
            except sqlite3.Error:
                return self._search_like(normalized, top_k)
        return self._search_like(normalized, top_k)

    async def update(self, skill_id: str, updates: dict) -> None:
        """更新技能欄位。"""
        if not updates:
            return
        allowed = {field.name for field in fields(Skill)}
        unknown = sorted(set(updates) - allowed)
        if unknown:
            raise ValueError(f"Unknown Skill field(s): {', '.join(unknown)}")
        current = await self.get(skill_id)
        if current is None:
            raise KeyError(skill_id)

        update_values = dict(updates)
        update_values["updated_at"] = update_values.get("updated_at") or time.time()
        assignments = ", ".join(f"{field_name} = ?" for field_name in update_values)
        values = [self._to_storage(field_name, value) for field_name, value in update_values.items()]
        values.append(skill_id)

        with self._conn:
            self._conn.execute(
                f"UPDATE skills SET {assignments} WHERE skill_id = ?",  # noqa: S608
                values,
            )
            refreshed = await self.get(skill_id)
            if refreshed is not None:
                self._sync_fts(refreshed)

    async def delete(self, skill_id: str) -> bool:
        """刪除技能；若有刪除資料則回傳 True。"""
        with self._conn:
            result = self._conn.execute("DELETE FROM skills WHERE skill_id = ?", (skill_id,))
            self._delete_fts(skill_id)
        return result.rowcount > 0

    async def list_indexed_sources(self, *, source_type: str | None = None) -> list[Skill]:
        """列出帶有 source_path 的索引技能，供檔案同步使用。"""
        sql = "SELECT * FROM skills WHERE source_path != ''"
        params: tuple[Any, ...] = ()
        if source_type is not None:
            sql += " AND source_type = ?"
            params = (source_type,)
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_skill(row) for row in rows]

    async def upsert(self, skill: Skill) -> str:
        """新增或替換技能，回傳 skill_id。"""
        if await self.get(skill.skill_id) is None:
            return await self.add(skill)

        updates = skill.to_dict()
        updates.pop("skill_id", None)
        created_at = updates.pop("created_at", 0)
        current = await self.get(skill.skill_id)
        if current is not None and not created_at:
            updates["created_at"] = current.created_at
        elif created_at:
            updates["created_at"] = created_at
        await self.update(skill.skill_id, updates)
        return skill.skill_id

    async def get_stats(self) -> dict:
        """取得技能庫統計。"""
        row = self._conn.execute(
            """
            SELECT
                COUNT(*) AS total_skills,
                COALESCE(SUM(times_used), 0) AS total_times_used,
                COALESCE(AVG(success_rate), 0) AS average_success_rate,
                COALESCE(MAX(version), 0) AS max_version,
                COALESCE(MAX(updated_at), 0) AS latest_updated_at
            FROM skills
            """,
        ).fetchone()
        return {
            "total_skills": row["total_skills"],
            "total_times_used": row["total_times_used"],
            "average_success_rate": row["average_success_rate"],
            "max_version": row["max_version"],
            "latest_updated_at": row["latest_updated_at"],
            "fts_enabled": self._fts_enabled,
        }

    async def export(self) -> list[dict]:
        """匯出所有技能為 CLI 易用的 dict 列表。"""
        return [self._skill_to_dict(skill) for skill in await self.list()]

    async def export_json(self) -> str:
        """匯出所有技能為 JSON 字串。"""
        return json.dumps(await self.export(), ensure_ascii=False, indent=2)

    async def merge(self, skill_id: str, new_trajectory: Trajectory) -> Skill:
        """合併新軌跡改進現有技能。"""
        skill = await self.get(skill_id)
        if skill is None:
            raise KeyError(skill_id)
        await self.update(
            skill_id,
            {
                "version": skill.version + 1,
                "source_trajectory_id": new_trajectory.trajectory_id,
                "updated_at": time.time(),
            },
        )
        merged = await self.get(skill_id)
        if merged is None:
            raise KeyError(skill_id)
        return merged

    def _init_schema(self) -> None:
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS skills (
                skill_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                trigger_keywords TEXT NOT NULL,
                preconditions TEXT NOT NULL,
                steps TEXT NOT NULL,
                tools_used TEXT NOT NULL,
                source_trajectory_id TEXT NOT NULL,
                times_used INTEGER NOT NULL DEFAULT 0,
                success_rate REAL NOT NULL DEFAULT 1.0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                version INTEGER NOT NULL DEFAULT 1
            )
            """,
        )
        self._ensure_schema_columns()
        self._ensure_fts_schema()
        self._sync_missing_fts_rows()
        self._conn.commit()

    def _ensure_fts_schema(self) -> None:
        try:
            rows = self._conn.execute("PRAGMA table_info(skills_fts)").fetchall()
        except sqlite3.OperationalError:
            rows = []
        columns = {row["name"] for row in rows}
        if rows and "body" not in columns:
            self._conn.execute("DROP TABLE skills_fts")

        try:
            self._conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS skills_fts USING fts5(
                    skill_id UNINDEXED,
                    name,
                    description,
                    trigger_keywords,
                    steps,
                    tools_used,
                    body
                )
                """,
            )
        except sqlite3.OperationalError:
            self._fts_enabled = False
        else:
            self._fts_enabled = True

    def _sync_missing_fts_rows(self) -> None:
        if not self._fts_enabled:
            return
        rows = self._conn.execute("SELECT * FROM skills").fetchall()
        for row in rows:
            self._sync_fts(self._row_to_skill(row))

    def _ensure_schema_columns(self) -> None:
        columns = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(skills)").fetchall()
        }
        migrations = {
            "source_type": "ALTER TABLE skills ADD COLUMN source_type TEXT NOT NULL DEFAULT 'learned'",
            "source_path": "ALTER TABLE skills ADD COLUMN source_path TEXT NOT NULL DEFAULT ''",
            "content_hash": "ALTER TABLE skills ADD COLUMN content_hash TEXT NOT NULL DEFAULT ''",
            "body": "ALTER TABLE skills ADD COLUMN body TEXT NOT NULL DEFAULT ''",
            "metadata": "ALTER TABLE skills ADD COLUMN metadata TEXT NOT NULL DEFAULT '{}'",
        }
        for column_name, statement in migrations.items():
            if column_name not in columns:
                self._conn.execute(statement)

    def _skill_values(self, skill: Skill) -> tuple[Any, ...]:
        return (
            skill.skill_id,
            skill.name,
            skill.description,
            self._to_storage("trigger_keywords", skill.trigger_keywords),
            skill.preconditions,
            self._to_storage("steps", skill.steps),
            self._to_storage("tools_used", skill.tools_used),
            skill.source_trajectory_id,
            skill.times_used,
            skill.success_rate,
            skill.created_at,
            skill.updated_at,
            skill.version,
            skill.source_type,
            skill.source_path,
            skill.content_hash,
            skill.body,
            self._to_storage("metadata", skill.metadata),
        )

    def _row_to_skill(self, row: sqlite3.Row) -> Skill:
        return Skill(
            skill_id=row["skill_id"],
            name=row["name"],
            description=row["description"],
            trigger_keywords=json.loads(row["trigger_keywords"]),
            preconditions=row["preconditions"],
            steps=json.loads(row["steps"]),
            tools_used=json.loads(row["tools_used"]),
            source_trajectory_id=row["source_trajectory_id"],
            times_used=row["times_used"],
            success_rate=row["success_rate"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            version=row["version"],
            source_type=row["source_type"],
            source_path=row["source_path"],
            content_hash=row["content_hash"],
            body=row["body"],
            metadata=json.loads(row["metadata"]),
        )

    def _skill_to_dict(self, skill: Skill) -> dict:
        return {field.name: getattr(skill, field.name) for field in fields(Skill)}

    def _to_storage(self, field_name: str, value: Any) -> Any:
        if field_name in self._JSON_FIELDS:
            return json.dumps(value, ensure_ascii=False)
        return value

    def _sync_fts(self, skill: Skill) -> None:
        if not self._fts_enabled:
            return
        self._delete_fts(skill.skill_id)
        self._conn.execute(
            """
            INSERT INTO skills_fts (
                skill_id, name, description, trigger_keywords, steps, tools_used, body
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                skill.skill_id,
                skill.name,
                skill.description,
                " ".join(skill.trigger_keywords),
                " ".join(skill.steps),
                " ".join(skill.tools_used),
                skill.body,
            ),
        )

    def _delete_fts(self, skill_id: str) -> None:
        if self._fts_enabled:
            self._conn.execute("DELETE FROM skills_fts WHERE skill_id = ?", (skill_id,))

    def _search_fts(self, query: str, top_k: int) -> list[Skill]:
        fts_query = self._build_fts_query(query)
        if not fts_query:
            return self._search_like(query, top_k)
        rows = self._conn.execute(
            """
            SELECT skills.*
            FROM skills_fts
            JOIN skills ON skills.skill_id = skills_fts.skill_id
            WHERE skills_fts MATCH ?
            ORDER BY bm25(skills_fts), skills.updated_at DESC
            LIMIT ?
            """,
            (fts_query, top_k),
        ).fetchall()
        return [self._row_to_skill(row) for row in rows]

    def _search_like(self, query: str, top_k: int) -> list[Skill]:
        pattern = f"%{query}%"
        rows = self._conn.execute(
            """
            SELECT *
            FROM skills
            WHERE name LIKE ?
               OR description LIKE ?
               OR trigger_keywords LIKE ?
               OR steps LIKE ?
               OR tools_used LIKE ?
               OR body LIKE ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (pattern, pattern, pattern, pattern, pattern, pattern, top_k),
        ).fetchall()
        return [self._row_to_skill(row) for row in rows]

    def _build_fts_query(self, query: str) -> str:
        terms = re.findall(r"[\w]+", query, flags=re.UNICODE)
        return " OR ".join(f'"{term}"' for term in terms)
