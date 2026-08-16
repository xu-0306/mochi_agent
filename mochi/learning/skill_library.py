"""SQLite skill storage with immutable retained versions and active projections."""

from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from collections.abc import Mapping
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

from mochi.learning.types import Skill, Trajectory


class SkillLibrary:
    """Store active skills separately from their immutable governance history."""

    _JSON_FIELDS = {"trigger_keywords", "steps", "tools_used", "metadata"}

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else None
        if self.db_path is not None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path) if self.db_path else ":memory:")
        self._conn.row_factory = sqlite3.Row
        self._fts_enabled = False
        self._init_schema()

    async def add(self, skill: Skill) -> str:
        """Create a skill and record its first immutable version."""

        now = time.time()
        skill_id = skill.skill_id or str(uuid.uuid4())
        latest_version = self._latest_retained_version(skill_id)
        stored = replace(
            skill,
            skill_id=skill_id,
            created_at=skill.created_at or now,
            updated_at=skill.updated_at or now,
            version=(latest_version + 1 if latest_version is not None else max(1, skill.version)),
        )
        with self._conn:
            if self._active_skill_row(skill_id) is not None:
                raise ValueError(f"Skill already exists: {skill_id}")
            self._insert_version(stored)
            self._write_active_projection(stored)
        return skill_id

    async def get(self, skill_id: str) -> Skill | None:
        row = self._active_skill_row(skill_id)
        return self._row_to_skill(row) if row else None

    async def list(self, limit: int | None = None) -> list[Skill]:
        sql = "SELECT * FROM skills ORDER BY updated_at DESC, created_at DESC, skill_id ASC"
        params: tuple[Any, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_skill(row) for row in rows]

    async def search(self, query: str, top_k: int = 3) -> list[Skill]:
        normalized = query.strip()
        if not normalized:
            return await self.list(limit=top_k)
        if self._fts_enabled:
            try:
                return self._search_fts(normalized, top_k)
            except sqlite3.Error:
                return self._search_like(normalized, top_k)
        return self._search_like(normalized, top_k)

    async def update(self, skill_id: str, updates: dict[str, Any]) -> None:
        """Append an active immutable version from a legacy field update."""

        if not updates:
            return
        allowed = {field.name for field in fields(Skill)}
        unknown = sorted(set(updates) - allowed)
        if unknown:
            raise ValueError(f"Unknown Skill field(s): {', '.join(unknown)}")
        current = await self.get(skill_id)
        if current is None:
            raise KeyError(skill_id)
        if "skill_id" in updates and updates["skill_id"] != skill_id:
            raise ValueError("skill_id is immutable")

        candidate = self._skill_to_dict(current)
        candidate.update(updates)
        candidate["skill_id"] = skill_id
        candidate["created_at"] = current.created_at
        candidate["version"] = self._next_retained_version(skill_id)
        candidate["updated_at"] = float(updates.get("updated_at") or time.time())
        next_skill = Skill.from_dict(candidate)
        with self._conn:
            self._insert_version(next_skill)
            self._write_active_projection(next_skill)

    async def delete(self, skill_id: str) -> bool:
        """Remove only the active projection; history and run pins remain retained."""

        with self._conn:
            result = self._conn.execute("DELETE FROM skills WHERE skill_id = ?", (skill_id,))
            self._conn.execute("DELETE FROM skill_active_versions WHERE skill_id = ?", (skill_id,))
            self._delete_fts(skill_id)
        return result.rowcount > 0

    async def list_indexed_sources(self, *, source_type: str | None = None) -> list[Skill]:
        sql = "SELECT * FROM skills WHERE source_path != ''"
        params: tuple[Any, ...] = ()
        if source_type is not None:
            sql += " AND source_type = ?"
            params = (source_type,)
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_skill(row) for row in rows]

    async def upsert(self, skill: Skill) -> str:
        if await self.get(skill.skill_id) is None:
            return await self.add(skill)

        updates = skill.to_dict()
        updates.pop("skill_id", None)
        updates.pop("version", None)
        updates.pop("created_at", None)
        await self.update(skill.skill_id, updates)
        return skill.skill_id

    async def get_stats(self) -> dict[str, Any]:
        row = self._conn.execute(
            """
            SELECT
                COUNT(*) AS total_skills,
                COALESCE(SUM(times_used), 0) AS total_times_used,
                COALESCE(AVG(success_rate), 0) AS average_success_rate,
                COALESCE(MAX(version), 0) AS max_version,
                COALESCE(MAX(updated_at), 0) AS latest_updated_at
            FROM skills
            """
        ).fetchone()
        return {
            "total_skills": row["total_skills"],
            "total_times_used": row["total_times_used"],
            "average_success_rate": row["average_success_rate"],
            "max_version": row["max_version"],
            "latest_updated_at": row["latest_updated_at"],
            "fts_enabled": self._fts_enabled,
        }

    async def export(self) -> list[dict[str, Any]]:
        return [self._skill_to_dict(skill) for skill in await self.list()]

    async def export_json(self) -> str:
        return json.dumps(await self.export(), ensure_ascii=False, indent=2)

    async def merge(self, skill_id: str, new_trajectory: Trajectory) -> Skill:
        skill = await self.get(skill_id)
        if skill is None:
            raise KeyError(skill_id)
        await self.update(
            skill_id,
            {
                "source_trajectory_id": new_trajectory.trajectory_id,
                "updated_at": time.time(),
            },
        )
        merged = await self.get(skill_id)
        if merged is None:
            raise KeyError(skill_id)
        return merged

    async def get_version(self, skill_id: str, version: int) -> Skill | None:
        row = self._conn.execute(
            "SELECT skill_json FROM skill_versions WHERE skill_id = ? AND version = ?",
            (skill_id, version),
        ).fetchone()
        return self._version_row_to_skill(row) if row else None

    async def list_versions(self, skill_id: str) -> list[Skill]:
        rows = self._conn.execute(
            "SELECT skill_json FROM skill_versions WHERE skill_id = ? ORDER BY version ASC",
            (skill_id,),
        ).fetchall()
        return [self._version_row_to_skill(row) for row in rows]

    async def promote(
        self,
        skill_id: str,
        version: int,
        evaluation_evidence: Mapping[str, Any],
    ) -> Skill:
        """Select a retained version only when durable evaluation evidence is supplied."""

        evidence = self._validate_evidence(evaluation_evidence)
        skill = await self.get_version(skill_id, version)
        if skill is None:
            raise KeyError(f"Missing retained skill version: {skill_id}@{version}")
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO skill_promotions (skill_id, version, evidence_json, promoted_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(skill_id, version) DO NOTHING
                """,
                (skill_id, version, self._json(evidence), time.time()),
            )
            self._write_active_projection(skill)
        return skill

    async def rollback(self, skill_id: str, version: int, *, reason: str) -> Skill:
        """Switch the active selector without mutating retained history or run pins."""

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("rollback reason must be non-empty")
        skill = await self.get_version(skill_id, version)
        if skill is None:
            raise KeyError(f"Missing retained skill version: {skill_id}@{version}")
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO skill_rollbacks (skill_id, version, reason, rolled_back_at)
                VALUES (?, ?, ?, ?)
                """,
                (skill_id, version, reason.strip(), time.time()),
            )
            self._write_active_projection(skill)
        return skill

    async def pin_run_version(self, run_id: str, skill_id: str, version: int) -> Skill:
        """Record an immutable run-to-skill-version reference."""

        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be non-empty")
        skill = await self.get_version(skill_id, version)
        if skill is None:
            raise KeyError(f"Missing retained skill version: {skill_id}@{version}")
        existing = self._conn.execute(
            "SELECT skill_id, version FROM skill_run_pins WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if existing is not None:
            if existing["skill_id"] == skill_id and existing["version"] == version:
                return skill
            raise ValueError("run_id is already pinned to another skill version")
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO skill_run_pins (run_id, skill_id, version, pinned_at)
                VALUES (?, ?, ?, ?)
                """,
                (run_id, skill_id, version, time.time()),
            )
        return skill

    async def get_pinned_version(self, run_id: str) -> Skill | None:
        row = self._conn.execute(
            """
            SELECT v.skill_json
            FROM skill_run_pins AS p
            JOIN skill_versions AS v
              ON v.skill_id = p.skill_id AND v.version = p.version
            WHERE p.run_id = ?
            """,
            (run_id,),
        ).fetchone()
        return self._version_row_to_skill(row) if row else None

    async def get_promotion(self, skill_id: str, version: int) -> dict[str, Any] | None:
        row = self._conn.execute(
            """
            SELECT evidence_json, promoted_at FROM skill_promotions
            WHERE skill_id = ? AND version = ?
            """,
            (skill_id, version),
        ).fetchone()
        if row is None:
            return None
        return {"evidence": json.loads(row["evidence_json"]), "promoted_at": row["promoted_at"]}

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
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS skill_versions (
                skill_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                skill_json TEXT NOT NULL,
                recorded_at REAL NOT NULL,
                PRIMARY KEY (skill_id, version)
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS skill_active_versions (
                skill_id TEXT PRIMARY KEY,
                version INTEGER NOT NULL,
                selected_at REAL NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS skill_promotions (
                skill_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                evidence_json TEXT NOT NULL,
                promoted_at REAL NOT NULL,
                PRIMARY KEY (skill_id, version)
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS skill_rollbacks (
                skill_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                reason TEXT NOT NULL,
                rolled_back_at REAL NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS skill_run_pins (
                run_id TEXT PRIMARY KEY,
                skill_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                pinned_at REAL NOT NULL
            )
            """
        )
        self._backfill_legacy_history()
        self._ensure_fts_schema()
        self._sync_missing_fts_rows()
        self._conn.commit()

    def _backfill_legacy_history(self) -> None:
        for row in self._conn.execute("SELECT * FROM skills").fetchall():
            skill = self._row_to_skill(row)
            history = self._conn.execute(
                """
                SELECT 1 FROM skill_versions WHERE skill_id = ? AND version = ?
                """,
                (skill.skill_id, skill.version),
            ).fetchone()
            if history is None:
                self._insert_version(skill)
            active = self._conn.execute(
                "SELECT 1 FROM skill_active_versions WHERE skill_id = ?",
                (skill.skill_id,),
            ).fetchone()
            if active is None:
                self._set_active_version(skill.skill_id, skill.version)

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
        for row in self._conn.execute("SELECT * FROM skills").fetchall():
            self._sync_fts(self._row_to_skill(row))

    def _ensure_schema_columns(self) -> None:
        columns = {row["name"] for row in self._conn.execute("PRAGMA table_info(skills)").fetchall()}
        migrations = {
            "source_type": "ALTER TABLE skills ADD COLUMN source_type TEXT NOT NULL DEFAULT 'learned'",
            "source_path": "ALTER TABLE skills ADD COLUMN source_path TEXT NOT NULL DEFAULT ''",
            "content_hash": "ALTER TABLE skills ADD COLUMN content_hash TEXT NOT NULL DEFAULT ''",
            "body": "ALTER TABLE skills ADD COLUMN body TEXT NOT NULL DEFAULT ''",
            "metadata": "ALTER TABLE skills ADD COLUMN metadata TEXT NOT NULL DEFAULT '{}'",
        }
        for column, statement in migrations.items():
            if column not in columns:
                self._conn.execute(statement)

    def _insert_version(self, skill: Skill) -> None:
        self._conn.execute(
            """
            INSERT INTO skill_versions (skill_id, version, skill_json, recorded_at)
            VALUES (?, ?, ?, ?)
            """,
            (skill.skill_id, skill.version, self._json(self._skill_to_dict(skill)), time.time()),
        )

    def _latest_retained_version(self, skill_id: str) -> int | None:
        row = self._conn.execute(
            "SELECT MAX(version) AS version FROM skill_versions WHERE skill_id = ?",
            (skill_id,),
        ).fetchone()
        if row is None or row["version"] is None:
            return None
        return int(row["version"])

    def _next_retained_version(self, skill_id: str) -> int:
        latest = self._latest_retained_version(skill_id)
        return 1 if latest is None else latest + 1

    def _write_active_projection(self, skill: Skill) -> None:
        self._conn.execute(
            """
            INSERT INTO skills (
                skill_id, name, description, trigger_keywords, preconditions,
                steps, tools_used, source_trajectory_id, times_used,
                success_rate, created_at, updated_at, version, source_type,
                source_path, content_hash, body, metadata
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(skill_id) DO UPDATE SET
                name = excluded.name,
                description = excluded.description,
                trigger_keywords = excluded.trigger_keywords,
                preconditions = excluded.preconditions,
                steps = excluded.steps,
                tools_used = excluded.tools_used,
                source_trajectory_id = excluded.source_trajectory_id,
                times_used = excluded.times_used,
                success_rate = excluded.success_rate,
                created_at = excluded.created_at,
                updated_at = excluded.updated_at,
                version = excluded.version,
                source_type = excluded.source_type,
                source_path = excluded.source_path,
                content_hash = excluded.content_hash,
                body = excluded.body,
                metadata = excluded.metadata
            """,
            self._skill_values(skill),
        )
        self._set_active_version(skill.skill_id, skill.version)
        self._sync_fts(skill)

    def _set_active_version(self, skill_id: str, version: int) -> None:
        self._conn.execute(
            """
            INSERT INTO skill_active_versions (skill_id, version, selected_at)
            VALUES (?, ?, ?)
            ON CONFLICT(skill_id) DO UPDATE SET
                version = excluded.version,
                selected_at = excluded.selected_at
            """,
            (skill_id, version, time.time()),
        )

    def _active_skill_row(self, skill_id: str) -> sqlite3.Row | None:
        return self._conn.execute("SELECT * FROM skills WHERE skill_id = ?", (skill_id,)).fetchone()

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

    @staticmethod
    def _version_row_to_skill(row: sqlite3.Row) -> Skill:
        return Skill.from_dict(json.loads(row["skill_json"]))

    @staticmethod
    def _skill_to_dict(skill: Skill) -> dict[str, Any]:
        return {field.name: getattr(skill, field.name) for field in fields(Skill)}

    def _to_storage(self, field_name: str, value: Any) -> Any:
        return self._json(value) if field_name in self._JSON_FIELDS else value

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

    @staticmethod
    def _build_fts_query(query: str) -> str:
        terms = re.findall(r"[\w]+", query, flags=re.UNICODE)
        return " OR ".join(f'"{term}"' for term in terms)

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _validate_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(value, Mapping) or not value:
            raise ValueError("promotion evidence must be non-empty")
        evidence_id = value.get("evidence_id")
        if not isinstance(evidence_id, str) or not evidence_id.strip():
            raise ValueError("promotion evidence must include evidence_id")
        return dict(value)
