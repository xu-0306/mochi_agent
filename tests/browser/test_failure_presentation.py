"""Production-browser evidence for the diagnostics-safe failure UI seam."""

from __future__ import annotations

import json
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
    REPOSITORY_ROOT / "tests" / "browser" / "support" / "test-failure-presentation-browser-fixture.mjs"
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


def _run_production_fixture(artifact_dir: Path) -> dict[str, object]:
    node = _node_executable()
    result = subprocess.run(
        [
            str(node),
            str(RUN_BROWSER_FIXTURE),
            str(Path("..") / PRODUCTION_FIXTURE.relative_to(REPOSITORY_ROOT)),
            ".next-fixture-failure-presentation",
        ],
        cwd=WEB_DIR,
        capture_output=True,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={
            **os.environ,
            "MOCHI_FAILURE_PRESENTATION_ARTIFACT_DIR": str(artifact_dir),
        },
    )
    assert result.returncode == 0, result.stderr
    manifest_path = artifact_dir / "manifest.json"
    assert manifest_path.is_file(), "the production fixture must emit its manifest"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def test_failure_presentation_uses_production_chat_goal_and_agent_run_surfaces(
    tmp_path: Path,
) -> None:
    """Render the canonical envelope through every production surface and viewport."""

    artifact_dir = tmp_path / "failure-presentation-browser"
    manifest = _run_production_fixture(artifact_dir)

    assert manifest["status"] == "passed"
    assert manifest["baseline_update_policy"] == "never"
    assert manifest["raw_latest_error_rendered"] is False
    assert manifest["diagnostics_ref_rendered"] is False
    assert [record["viewport"]["name"] for record in manifest["viewports"]] == ["desktop", "mobile"]

    for record in manifest["viewports"]:
        assert record["status"] == "passed"
        assert record["raw_latest_error_rendered"] is False
        assert record["diagnostics_ref_rendered"] is False
        assert record["horizontal_overflow_px"] == 0
        assert record["document_bounds"]["scroll_width"] == record["document_bounds"]["client_width"]
        assert all(bound["overflow_px"] == 0 for bound in record["horizontal_bounds"])
        screenshot = artifact_dir / record["screenshot"]
        assert screenshot.is_file()
        assert len(record["screenshot_sha256"]) == 64
