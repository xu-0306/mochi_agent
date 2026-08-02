#!/usr/bin/env python3
"""Windows adapter for agent-foreman fingerprinting with invalid WSL reparse points."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

_SOURCE = Path(
    "C:/Users/Xu/.codex/skills/agent-foreman/scripts/fingerprint_scope.py"
)


def _load_source() -> ModuleType:
    sys.path.insert(0, str(_SOURCE.parent))
    spec = importlib.util.spec_from_file_location(
        "agent_foreman_fingerprint_scope",
        _SOURCE,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load fingerprint source: {_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _reparse_marker(path: Path, relative: str, error: OSError) -> bytes:
    stat = path.lstat()
    payload = {
        "error_type": type(error).__name__,
        "mode": stat.st_mode & 0o777,
        "path": relative,
        "reparse_point": True,
        "size": stat.st_size,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _collect_files(
    module: ModuleType,
    repo: Path,
    excluded: Path | None = None,
) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    try:
        excluded_resolved = excluded.resolve() if excluded is not None else None
    except OSError:
        excluded_resolved = excluded.absolute() if excluded is not None else None
    for root, directories, names in os.walk(repo):
        directories[:] = sorted(name for name in directories if name != ".git")
        root_path = Path(root)
        for name in sorted(names):
            path = root_path / name
            try:
                resolved = path.resolve()
            except OSError:
                resolved = path.absolute()
            if excluded_resolved is not None and resolved == excluded_resolved:
                continue
            relative = module.normalize_path(path.relative_to(repo).as_posix())
            try:
                if path.is_symlink():
                    content = os.readlink(path).encode("utf-8")
                    kind = "symlink"
                else:
                    content = path.read_bytes()
                    kind = "file"
                stat = path.lstat()
            except OSError as error:
                try:
                    content = _reparse_marker(path, relative, error)
                    kind = "unreadable_reparse_point"
                    stat = path.lstat()
                except OSError as marker_error:
                    raise RuntimeError(
                        f"cannot fingerprint {relative}: {marker_error}"
                    ) from marker_error
            files[relative] = {
                "sha256": _sha256_bytes(content),
                "size": len(content),
                "kind": kind,
                "mode": stat.st_mode & 0o777,
            }
    return files


def main() -> int:
    module = _load_source()
    module.collect_files = lambda repo, excluded=None: _collect_files(
        module,
        repo,
        excluded,
    )
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
