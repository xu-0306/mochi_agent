from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from scripts.quality_gate import run

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_ROOT = REPOSITORY_ROOT / "docs" / "architecture" / "contracts"


def _frozen_revision() -> str:
    """Return the revision declared by the frozen contract set."""

    artifacts = sorted(CONTRACT_ROOT.glob("*.json"))
    assert artifacts, "at least one frozen contract artifact is required before qualification"
    revisions = {_read_json(artifact)["git_revision"] for artifact in artifacts}
    assert len(revisions) == 1, "all frozen contract artifacts must share one revision"
    revision = revisions.pop()
    assert isinstance(revision, str) and revision, "frozen contract revision must be non-empty"
    return revision


def _current_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_frozen_contract_artifacts_are_revision_bound_and_machine_readable() -> None:
    artifacts = sorted(CONTRACT_ROOT.glob("*.json"))

    assert artifacts, "at least one frozen contract artifact is required before qualification"
    revision = _frozen_revision()
    for artifact in artifacts:
        payload = _read_json(artifact)
        run.validate_frozen_contract_artifact(payload, expected_revision=revision)


def test_quality_report_contract_matches_the_runner_validator() -> None:
    contract = _read_json(CONTRACT_ROOT / "quality-report-v1.json")
    run.validate_quality_report_contract(contract, expected_revision=contract["git_revision"])
    report = run.build_report(
        repo_root=REPOSITORY_ROOT,
        mode="baseline",
        execute=False,
        git_sha=_current_revision(),
        dirty_state={
            "entries": [],
            "tracked_diff_sha256": "a" * 64,
            "untracked_files": {},
            "sha256": run.dirty_state_digest(
                entries=[],
                tracked_diff_sha256="a" * 64,
                untracked_files={},
            ),
        },
    )

    run.validate_report(report)


def test_contract_validator_rejects_empty_compatibility() -> None:
    contract = _read_json(CONTRACT_ROOT / "quality-report-v1.json")
    contract["compatibility"] = {}

    try:
        run.validate_frozen_contract_artifact(contract, expected_revision=contract["git_revision"])
    except run.QualityGateError as error:
        assert "compatibility" in str(error)
    else:
        raise AssertionError("empty compatibility must be rejected")


def test_quality_report_contract_validator_rejects_lane_schema_drift() -> None:
    contract = _read_json(CONTRACT_ROOT / "quality-report-v1.json")
    contract["report_schema"]["lane_results"] = ["passed"]

    try:
        run.validate_quality_report_contract(contract, expected_revision=contract["git_revision"])
    except run.QualityGateError as error:
        assert "lane_results" in str(error)
    else:
        raise AssertionError("lane result schema drift must be rejected")


def test_contract_validator_cli_validates_the_declared_revision() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/quality_gate/validate_contract.py",
            "docs/architecture/contracts/failure-envelope-v1.json",
            "--require-frozen",
            "--revision",
            _read_json(CONTRACT_ROOT / "failure-envelope-v1.json")["git_revision"],
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "frozen contract validated" in completed.stdout


def test_contract_validator_cli_rejects_a_mismatched_revision() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/quality_gate/validate_contract.py",
            "docs/architecture/contracts/failure-envelope-v1.json",
            "--require-frozen",
            "--revision",
            "0" * 40,
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "revision does not match" in completed.stderr


def test_live_worker_dispatches_reference_frozen_contracts_at_their_revision() -> None:
    dispatch_root = REPOSITORY_ROOT / "artifacts" / "foreman"
    if not dispatch_root.exists():
        return

    for dispatch_path in dispatch_root.rglob("dispatch.json"):
        dispatch = _read_json(dispatch_path)
        for contract_path in dispatch.get("frozen_contracts", []):
            artifact = REPOSITORY_ROOT / contract_path
            assert artifact.exists(), f"{dispatch_path} references missing contract {contract_path}"
            contract = _read_json(artifact)
            assert contract["status"] == "frozen", contract_path
            if "git_revision" in dispatch:
                assert contract["git_revision"] == dispatch["git_revision"], contract_path
