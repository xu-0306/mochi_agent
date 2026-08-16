"""Browser evidence for the TaskPanel's server-authoritative subset workflow."""

import os
import shutil
import subprocess
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = REPOSITORY_ROOT / "web"
FROZEN_WINDOWS_NODE = Path(
    "C:/Users/xu/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe"
)
RUN_BROWSER_FIXTURE = WEB_DIR / "scripts" / "run-browser-fixture.mjs"
PRODUCTION_FIXTURE = (
    REPOSITORY_ROOT / "tests" / "browser" / "support" / "test-file-workflow-browser-fixture.mjs"
)


def _node_executable() -> Path:
    configured = os.environ.get("MOCHI_NODE_EXECUTABLE")
    discovered = shutil.which("node")
    candidates = (
        Path(configured) if configured else None,
        FROZEN_WINDOWS_NODE,
        Path(discovered) if discovered else None,
    )
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    raise AssertionError("A Node.js runtime is required for the production browser fixture.")


def _run_production_fixture() -> None:
    node = _node_executable()
    result = subprocess.run(
        [
            str(node),
            str(RUN_BROWSER_FIXTURE),
            str(Path("..") / PRODUCTION_FIXTURE.relative_to(REPOSITORY_ROOT)),
            ".next-fixture-file-workflow",
        ],
        cwd=WEB_DIR,
        capture_output=True,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert result.returncode == 0, result.stderr


def test_task_panel_subset_selection_is_grouped_and_browser_rendered() -> None:
    """Exercise TaskPanel, its Zustand store, and its API client in a real browser."""

    _run_production_fixture()
