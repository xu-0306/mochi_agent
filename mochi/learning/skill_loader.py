"""Filesystem SKILL.md loader for Mochi skills."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from mochi.config.defaults import repo_skills_dir
from mochi.learning.skill_library import SkillLibrary
from mochi.learning.types import Skill


@dataclass(frozen=True)
class SkillSyncResult:
    """Filesystem skill sync summary."""

    scanned: int = 0
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    errors: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()


@dataclass(frozen=True)
class SkillSource:
    """Directory containing Codex/Claude-style SKILL.md files."""

    path: Path
    source_type: str


class SkillLoader:
    """同步 Codex/Claude-style `SKILL.md` 檔案到 SkillLibrary 索引。"""

    SKILL_FILENAME = "SKILL.md"

    def __init__(self, sources: list[SkillSource]) -> None:
        self._sources = sources

    @classmethod
    def from_paths(
        cls,
        user_skills_dir: str | Path | None = None,
        *,
        system_skills_dir: str | Path | None = None,
    ) -> SkillLoader:
        """用 user/system skill 目錄建立 loader。"""
        sources: list[SkillSource] = []
        if system_skills_dir is not None:
            sources.append(SkillSource(Path(system_skills_dir).expanduser(), "system"))
        if user_skills_dir is not None:
            sources.append(SkillSource(Path(user_skills_dir).expanduser(), "filesystem"))
        return cls(sources)

    async def sync(self, library: SkillLibrary) -> SkillSyncResult:
        """掃描 sources 並同步 SQLite 索引。"""
        desired: dict[str, Skill] = {}
        scanned = 0
        errors: list[str] = []
        source_types = {source.source_type for source in self._sources}
        system_paths = tuple(source.path for source in self._sources if source.source_type == "system")

        for source in self._sources:
            if not source.path.exists():
                continue
            for skill_path in _iter_skill_paths(source, system_paths=system_paths):
                scanned += 1
                try:
                    skill = parse_skill_file(skill_path, source_root=source.path, source_type=source.source_type)
                except (OSError, ValueError, yaml.YAMLError) as exc:
                    errors.append(f"{skill_path}: {exc}")
                    continue
                desired[skill.skill_id] = skill

        added = 0
        updated = 0
        unchanged = 0
        for skill in desired.values():
            current = await library.get(skill.skill_id)
            if current is None:
                await library.add(skill)
                added += 1
                continue
            if current.content_hash == skill.content_hash and current.source_path == skill.source_path:
                unchanged += 1
                continue
            merged = _preserve_usage_fields(skill, current)
            await library.upsert(merged)
            updated += 1

        removed = 0
        indexed_sources = await library.list_indexed_sources()
        for indexed in indexed_sources:
            if indexed.source_type not in source_types:
                continue
            if indexed.skill_id in desired:
                continue
            await library.delete(indexed.skill_id)
            removed += 1

        return SkillSyncResult(
            scanned=scanned,
            added=added,
            updated=updated,
            unchanged=unchanged,
            removed=removed,
            errors=tuple(errors),
            sources=tuple(str(source.path) for source in self._sources),
        )


def parse_skill_file(
    path: str | Path,
    *,
    source_root: str | Path | None = None,
    source_type: str = "filesystem",
) -> Skill:
    """將 Codex/Claude-style `SKILL.md` 解析成 Mochi Skill。"""
    skill_path = Path(path).expanduser()
    raw = skill_path.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(raw)
    metadata = _parse_frontmatter(frontmatter)

    name = _first_text(metadata.get("name")) or skill_path.parent.name
    description = _first_text(metadata.get("description")) or _first_paragraph(body)
    skill_id = _skill_id(name, skill_path=skill_path, source_root=source_root, source_type=source_type)
    content_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    now = time.time()

    return Skill(
        skill_id=skill_id,
        name=name,
        description=description,
        trigger_keywords=_trigger_keywords(name, description, metadata),
        preconditions="",
        steps=_steps_from_body(body),
        tools_used=[],
        source_trajectory_id="",
        created_at=now,
        updated_at=now,
        source_type=source_type,
        source_path=str(skill_path),
        content_hash=content_hash,
        body=body.strip(),
        metadata=metadata,
    )


def default_system_skills_dir() -> Path:
    """取得 Mochi 內建 system skills 目錄。"""
    return Path(__file__).resolve().parents[1] / "skills" / ".system"


def default_user_skills_dir() -> Path:
    """取得 Mochi 預設 user skills 目錄。"""
    return repo_skills_dir()


def _iter_skill_paths(source: SkillSource, *, system_paths: tuple[Path, ...]) -> list[Path]:
    paths: list[Path] = []
    for skill_path in sorted(source.path.rglob(SkillLoader.SKILL_FILENAME)):
        if source.source_type == "filesystem" and any(
            _is_relative_to(skill_path, system_path) for system_path in system_paths
        ):
            continue
        paths.append(skill_path)
    return paths


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=False))
    except ValueError:
        return False
    return True


def _split_frontmatter(raw: str) -> tuple[str, str]:
    if not raw.startswith("---"):
        return "", raw
    lines = raw.splitlines()
    if not lines or lines[0].strip() != "---":
        return "", raw
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return "\n".join(lines[1:index]), "\n".join(lines[index + 1 :])
    return "", raw


def _parse_frontmatter(frontmatter: str) -> dict[str, Any]:
    if not frontmatter.strip():
        return {}
    payload = yaml.safe_load(frontmatter)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("SKILL.md frontmatter must be a YAML mapping")
    return cast(dict[str, Any], payload)


def _skill_id(
    name: str,
    *,
    skill_path: Path,
    source_root: str | Path | None,
    source_type: str,
) -> str:
    base = _slug(name) or _slug(skill_path.parent.name) or "skill"
    if source_type == "system":
        return f"system:{base}"
    if source_root is None:
        return base
    try:
        relative_parent = skill_path.parent.relative_to(Path(source_root).expanduser())
    except ValueError:
        return base
    suffix = _slug(str(relative_parent)) or base
    return suffix if suffix == base else f"{base}:{suffix}"


def _slug(value: str) -> str:
    chars: list[str] = []
    previous_dash = False
    for char in value.lower():
        if char.isalnum():
            chars.append(char)
            previous_dash = False
        elif not previous_dash:
            chars.append("-")
            previous_dash = True
    return "".join(chars).strip("-")


def _trigger_keywords(name: str, description: str, metadata: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for field_name in ("triggers", "trigger_keywords", "tags"):
        values.extend(_as_text_list(metadata.get(field_name)))
    values.extend(_words(name))
    values.extend(_words(description))

    result: list[str] = []
    for value in values:
        normalized = value.strip().lower()
        if len(normalized) >= 2 and normalized not in result:
            result.append(normalized)
        if len(result) >= 12:
            break
    return result


def _steps_from_body(body: str) -> list[str]:
    heading = ""
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip()
            if heading:
                break
    if heading:
        return [f"Follow the `{heading}` instructions from the skill body."]
    return ["Read and follow the skill body instructions."]


def _first_text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _as_text_list(value: object) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        items = cast(list[Any], value)
        return [text for item in items if (text := str(item).strip())]
    return []


def _first_paragraph(body: str) -> str:
    lines: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            if lines:
                break
            continue
        lines.append(stripped)
    return " ".join(lines)[:240] if lines else "Filesystem skill loaded from SKILL.md."


def _words(value: str) -> list[str]:
    words: list[str] = []
    for raw_word in value.replace("_", " ").replace("-", " ").split():
        word = raw_word.strip(".,:;!?()[]{}\"'").lower()
        if len(word) >= 3:
            words.append(word)
    return words


def _preserve_usage_fields(next_skill: Skill, current: Skill) -> Skill:
    next_skill.times_used = current.times_used
    next_skill.success_rate = current.success_rate
    next_skill.created_at = current.created_at or next_skill.created_at
    return next_skill
