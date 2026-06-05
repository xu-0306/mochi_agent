"""Persistent store for WebGUI projects."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict
from uuid import uuid4

from mochi.config import defaults
from mochi.utils.security import normalize_workspace_dir


class ProjectRecord(TypedDict):
    """Serialized project payload."""

    id: str
    name: str
    workspace_dir: str
    created_at: str
    updated_at: str


class ProjectStore:
    """JSON-backed project persistence."""

    def __init__(self, path: str | Path = Path(defaults.default_workspace_dir()) / "projects.json") -> None:
        self._path = Path(path).expanduser()

    async def list_projects(self) -> list[ProjectRecord]:
        records = await asyncio.to_thread(self._read_records)
        return sorted(records, key=lambda item: item["updated_at"], reverse=True)

    async def get_project(self, project_id: str) -> ProjectRecord | None:
        records = await asyncio.to_thread(self._read_records)
        for record in records:
            if record["id"] == project_id:
                return record
        return None

    async def create_project(self, *, name: str, workspace_dir: str) -> ProjectRecord:
        now = datetime.now(tz=UTC).isoformat()
        record: ProjectRecord = {
            "id": str(uuid4()),
            "name": name.strip(),
            "workspace_dir": str(normalize_workspace_dir(workspace_dir)),
            "created_at": now,
            "updated_at": now,
        }
        records = await asyncio.to_thread(self._read_records)
        records.append(record)
        await asyncio.to_thread(self._write_records, records)
        return record

    async def update_project(
        self,
        project_id: str,
        *,
        name: str | None = None,
        workspace_dir: str | None = None,
    ) -> ProjectRecord | None:
        records = await asyncio.to_thread(self._read_records)
        updated: ProjectRecord | None = None
        for index, record in enumerate(records):
            if record["id"] != project_id:
                continue
            next_record = dict(record)
            if name is not None:
                next_record["name"] = name.strip()
            if workspace_dir is not None:
                next_record["workspace_dir"] = str(normalize_workspace_dir(workspace_dir))
            next_record["updated_at"] = datetime.now(tz=UTC).isoformat()
            updated = next_record  # type: ignore[assignment]
            records[index] = next_record  # type: ignore[assignment]
            break
        if updated is None:
            return None
        await asyncio.to_thread(self._write_records, records)
        return updated

    async def delete_project(self, project_id: str) -> bool:
        records = await asyncio.to_thread(self._read_records)
        next_records = [record for record in records if record["id"] != project_id]
        if len(next_records) == len(records):
            return False
        await asyncio.to_thread(self._write_records, next_records)
        return True

    def _read_records(self) -> list[ProjectRecord]:
        if not self._path.exists():
            return []

        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []

        if not isinstance(raw, list):
            return []

        records: list[ProjectRecord] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            project_id = item.get("id")
            name = item.get("name")
            workspace_dir = item.get("workspace_dir")
            created_at = item.get("created_at")
            updated_at = item.get("updated_at")
            if not all(
                isinstance(value, str) and value.strip()
                for value in (project_id, name, workspace_dir, created_at, updated_at)
            ):
                continue
            records.append(
                {
                    "id": project_id.strip(),
                    "name": name.strip(),
                    "workspace_dir": str(normalize_workspace_dir(workspace_dir)),
                    "created_at": created_at,
                    "updated_at": updated_at,
                }
            )
        return records

    def _write_records(self, records: list[ProjectRecord]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
