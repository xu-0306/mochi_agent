"""Skill library path helpers."""

from __future__ import annotations

from pathlib import Path


def resolve_skills_db_path(*, skills_dir: str | Path) -> Path:
    """Resolve the configured skill library SQLite path."""
    return (Path(skills_dir).expanduser() / "skills.db").resolve()
