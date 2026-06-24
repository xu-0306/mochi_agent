"""Windows launcher 回歸測試。"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = REPO_ROOT / "scripts" / "windows" / "start-mochi.ps1"
BACKEND_SCRIPT_PATH = REPO_ROOT / "scripts" / "dev-backend-windows.ps1"


def _extract_section(output: str, header: str) -> list[str]:
    lines = output.splitlines()
    try:
        start = lines.index(header) + 1
    except ValueError as exc:  # pragma: no cover - assertion below handles it
        raise AssertionError(f"Missing section header: {header}") from exc

    collected: list[str] = []
    for line in lines[start:]:
        if not line.strip():
            break
        collected.append(line)
    return collected


def test_start_mochi_dry_run_renders_multi_line_child_scripts() -> None:
    """DryRun 應產出可執行的多行 child script 內容。"""
    if shutil.which("powershell") is None:
        pytest.skip("powershell is not available in this environment")

    result = subprocess.run(  # noqa: S603
        [
            "powershell",
            "-NoLogo",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(LAUNCHER_PATH),
            "-DryRun",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout

    backend_lines = _extract_section(result.stdout, "[DryRun] Backend child script content:")
    frontend_lines = _extract_section(result.stdout, "[DryRun] Frontend child script content:")
    dev_home = REPO_ROOT / ".tmp" / "home"
    dev_state_root = dev_home / ".mochi"

    assert backend_lines == [
        '$ErrorActionPreference = "Stop"',
        "$env:UV_PROJECT_ENVIRONMENT = '.venv-win'",
        "$env:MOCHI_API_HOST = '127.0.0.1'",
        "$env:MOCHI_API_PORT = '8000'",
        f"$env:HOME = '{dev_home}'",
        f"$env:MOCHI_WORKSPACE_DIR = '{dev_state_root / 'workspace'}'",
        f"$env:MOCHI_SESSIONS_DIR = '{dev_state_root / 'sessions'}'",
        f"$env:MOCHI_SKILLS_DIR = '{dev_state_root / 'skills'}'",
        f"$env:MOCHI_PLUGINS_DIR = '{dev_state_root / 'plugins'}'",
        f"& '{REPO_ROOT / 'scripts' / 'dev-backend-windows.ps1'}'",
    ]
    assert frontend_lines == [
        '$ErrorActionPreference = "Stop"',
        "$env:MOCHI_WEB_HOST = '127.0.0.1'",
        "$env:MOCHI_WEB_PORT = '3000'",
        "$env:MOCHI_API_BASE_URL = 'http://127.0.0.1:8000'",
        "$env:NEXT_PUBLIC_MOCHI_API_BASE_URL = 'http://127.0.0.1:8000'",
        f"$env:HOME = '{dev_home}'",
        f"$env:MOCHI_WORKSPACE_DIR = '{dev_state_root / 'workspace'}'",
        f"$env:MOCHI_SESSIONS_DIR = '{dev_state_root / 'sessions'}'",
        f"$env:MOCHI_SKILLS_DIR = '{dev_state_root / 'skills'}'",
        f"$env:MOCHI_PLUGINS_DIR = '{dev_state_root / 'plugins'}'",
        f"& '{REPO_ROOT / 'scripts' / 'dev-web-windows.ps1'}'",
    ]


def test_start_mochi_personal_home_mode_does_not_override_state_env() -> None:
    """Personal-home mode should keep using the user's existing home-scoped state."""
    if shutil.which("powershell") is None:
        pytest.skip("powershell is not available in this environment")

    result = subprocess.run(  # noqa: S603
        [
            "powershell",
            "-NoLogo",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(LAUNCHER_PATH),
            "-Mode",
            "PersonalHome",
            "-DryRun",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout

    backend_lines = _extract_section(result.stdout, "[DryRun] Backend child script content:")
    frontend_lines = _extract_section(result.stdout, "[DryRun] Frontend child script content:")

    assert all("HOME" not in line for line in backend_lines)
    assert all("MOCHI_WORKSPACE_DIR" not in line for line in backend_lines)
    assert all("MOCHI_SESSIONS_DIR" not in line for line in backend_lines)
    assert all("MOCHI_SKILLS_DIR" not in line for line in backend_lines)
    assert all("MOCHI_PLUGINS_DIR" not in line for line in backend_lines)
    assert all("HOME" not in line for line in frontend_lines)
    assert all("MOCHI_WORKSPACE_DIR" not in line for line in frontend_lines)
    assert all("MOCHI_SESSIONS_DIR" not in line for line in frontend_lines)
    assert all("MOCHI_SKILLS_DIR" not in line for line in frontend_lines)
    assert all("MOCHI_PLUGINS_DIR" not in line for line in frontend_lines)


def test_windows_backend_sync_keeps_hf_runtime_dependencies() -> None:
    """Backend launcher should not prune local HF runtime packages."""
    script = BACKEND_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "uv sync --group dev --extra hf --inexact" in script
