"""Browser evidence for the TaskPanel's server-authoritative subset workflow."""

import subprocess
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = REPOSITORY_ROOT / "web"
NODE = Path(
    "C:/Users/xu/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe"
)
RUN_BROWSER_FIXTURE = WEB_DIR / "scripts" / "run-browser-fixture.mjs"
PRODUCTION_FIXTURE = (
    REPOSITORY_ROOT / "tests" / "browser" / "support" / "test-file-workflow-browser-fixture.mjs"
)


def _run_production_fixture() -> None:
    if not NODE.is_file():
        raise AssertionError(f"Frozen browser Node runtime is unavailable: {NODE}")
    result = subprocess.run(
        [
            str(NODE),
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
