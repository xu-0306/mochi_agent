"""Bounded, redacted Phase 9 evaluator for policy and one Engine fixture.

Most rows use deterministic policy; one paired simple fixture invokes ordinary
Chat with the same local FakeBackend. No network model or effectful tool runs.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mochi.agents.engine import AgentEngine
from mochi.agents.complexity_gate import (
    ComplexityCapabilitySummary,
    ComplexityGate,
    ComplexityGateRequest,
)
from mochi.agents.turn_intent_contract import DeliverableContract, TurnIntentContract
from mochi.agents.invocation import AgentInvocationRequest
from mochi.backends.base import BaseLLMBackend
from mochi.backends.types import GenerationResult, Message, ModelInfo
from mochi.config.schema import MochiConfig


Mode = Literal["off", "shadow", "enforce"]

_SOURCE_HASH_PATHS = (
    "tests/evaluation/evaluate_adaptive_runtime_wave4.py",
    "mochi/agents/complexity_gate.py",
    "mochi/agents/engine.py",
    "mochi/agents/react_loop.py",
)


class _SimpleFixtureBackend(BaseLLMBackend):
    """One deterministic ordinary-Chat answer; no advisor/judge/tool behavior."""

    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, messages: list[Message], **_: Any) -> GenerationResult:
        self.calls += 1
        return GenerationResult(content="fixture answer", input_tokens=4, output_tokens=2)

    def supports_tool_calling(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name="wave4-fixture", backend_type="fixture")

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


async def _engine_simple_fixture(mode: Mode) -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix="mochi-wave4-") as directory:
        root = Path(directory)
        config = MochiConfig.model_validate(
            {
                "model": "ollama:fixture",
                "workspace_dir": str(root),
                "sessions_dir": str(root / "sessions"),
                "memory": {"db_path": str(root / "memory.db")},
                "agent": {
                    "ordinary_chat_adaptive_runtime": {"complexity": {"mode": mode}}
                },
            }
        )
        engine = AgentEngine(config)
        backend = _SimpleFixtureBackend()
        try:
            result = await engine.invoke(
                AgentInvocationRequest(
                    message="Explain why an expired JWT is rejected.",
                    session_id=f"wave4-{mode}",
                    turn_id=f"wave4-{mode}-turn",
                    persist_session=True,
                    backend_override=backend,
                )
            )
            events = await engine._session_store.load_session(f"wave4-{mode}")
            diagnostic = next(
                (event for event in events if event.get("event") == "adaptive_diagnostics"),
                None,
            )
            counters = diagnostic["counters"] if isinstance(diagnostic, dict) else {}
            if backend.calls != 1:
                raise RuntimeError("simple Engine fixture must make exactly one main response call")
            return {
                "backend_calls": backend.calls,
                "model_calls": int(counters.get("model_calls", backend.calls)),
                "input_tokens": int(counters.get("input_tokens", 4)),
                "output_tokens": int(counters.get("output_tokens", 2)),
                "tool_calls": int(counters.get("tool_calls", 0)),
                "recovery_model_calls": int(counters.get("recovery_model_calls", 0)),
                "recovery_tool_calls": int(counters.get("recovery_tool_calls", 0)),
                "adaptive_diagnostics_persisted": int(diagnostic is not None),
                "result_event_count": len(result.events),
            }
        finally:
            await engine.close()


def _contract(fixture: dict[str, Any]) -> TurnIntentContract:
    fixture_id = str(fixture["id"])
    criteria_count = int(fixture["acceptance_criteria_per_deliverable"])
    deliverables = tuple(
        DeliverableContract(
            kind="workspace_file",
            target_hint=f"fixtures/{fixture_id}-{index}.md",
            acceptance_criteria=tuple(
                f"criterion-{criterion}" for criterion in range(criteria_count)
            ),
            source_turn_ids=(fixture_id,),
        )
        for index in range(int(fixture["deliverable_count"]))
    )
    operations = frozenset(str(value) for value in fixture["operations"])
    return TurnIntentContract(
        turn_id=fixture_id,
        active_goal_id=f"goal:{fixture_id}",
        objective=str(fixture["objective"]),
        current_speech_act=(
            "request_execution" if "workspace_write" in operations else "request_information"
        ),
        operations=operations,
        deliverables=deliverables,
        resolved_references=(),
        positive_constraints=(),
        negative_constraints=(),
        mutation_requirement=("required" if "workspace_write" in operations else "forbidden"),
        clarification=None,
        supersedes_previous_goal=False,
        cancels_active_goal=False,
        modifies_active_task=True,
        confidence=1.0,
        evidence=(),
        advisories=(),
    )


def _evaluate(fixture: dict[str, Any], mode: Mode, samples: int) -> dict[str, Any]:
    contract = _contract(fixture)
    request = ComplexityGateRequest(
        turn_intent=contract,
        task_relation="start",
        capability_summary=ComplexityCapabilitySummary(
            effectful_tool_count=int(fixture["effectful_tool_count"])
        ),
    )
    gate = ComplexityGate()
    durations_ns: list[int] = []
    decision = None
    for _ in range(samples):
        started = time.perf_counter_ns()
        decision = None if mode == "off" else gate.evaluate_deterministic(request)
        durations_ns.append(time.perf_counter_ns() - started)
    assert decision is not None or mode == "off"
    dynamic_decision = None
    if mode == "enforce" and fixture.get("dynamic_label") is not None:
        initial = gate.evaluate_deterministic(request)
        dynamic_decision = asyncio.run(
            gate.recheck(
                request,
                prior_decision=initial,
                completed_iterations=1,
                signals=("read_to_effectful",),
            )
        )
        dynamic_kind = dynamic_decision.kind if dynamic_decision is not None else None
    else:
        dynamic_kind = None
    return {
        "decision_kind": decision.kind if decision is not None else None,
        "dynamic_decision_kind": dynamic_kind,
        "adaptive_model_calls": 0,
        "adaptive_input_tokens": 0,
        "adaptive_output_tokens": 0,
        "adaptive_tool_calls": 0,
        "median_wall_ns": int(statistics.median(durations_ns)),
    }


def evaluate(fixtures_path: Path) -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parents[2]
    fixture_bytes = fixtures_path.read_bytes()
    payload = json.loads(fixture_bytes.decode("utf-8"))
    fixtures = payload["fixtures"]
    samples = int(payload["samples_per_fixture"])
    rows: list[dict[str, Any]] = []
    confusion = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for fixture in fixtures:
        row = {
            "id": fixture["id"],
            "label": fixture["label"],
            "off": _evaluate(fixture, "off", samples),
            "shadow": _evaluate(fixture, "shadow", samples),
            "enforce": _evaluate(fixture, "enforce", samples),
        }
        predicted = row["enforce"]["decision_kind"]
        actual = fixture["label"]
        if actual == "plan_required":
            confusion["tp" if predicted == actual else "fn"] += 1
        else:
            confusion["tn" if predicted == actual else "fp"] += 1
        if fixture.get("dynamic_label") is not None:
            dynamic_predicted = row["enforce"]["dynamic_decision_kind"]
            if dynamic_predicted == fixture["dynamic_label"]:
                confusion["tp"] += 1
            else:
                confusion["fn"] += 1
        rows.append(row)
    exact_zero_delta = all(
        row["off"][metric] == row["shadow"][metric] == row["enforce"][metric] == 0
        for row in rows
        for metric in (
            "adaptive_model_calls",
            "adaptive_input_tokens",
            "adaptive_output_tokens",
            "adaptive_tool_calls",
        )
    )
    engine_pairs = {
        mode: asyncio.run(_engine_simple_fixture(mode))
        for mode in ("off", "shadow", "enforce")
    }
    engine_baseline = engine_pairs["off"]
    engine_delta = {
        mode: {
            key: values[key] - engine_baseline[key]
            for key in values
        }
        for mode, values in engine_pairs.items()
    }
    engine_extra_counter_fields = (
        "backend_calls",
        "model_calls",
        "input_tokens",
        "output_tokens",
        "tool_calls",
        "recovery_model_calls",
        "recovery_tool_calls",
    )
    engine_extra_zero = all(
        delta[key] == 0
        for mode, delta in engine_delta.items()
        if mode != "off"
        for key in engine_extra_counter_fields
    )
    disagreements = [
        row["id"]
        for row in rows
        if row["enforce"]["decision_kind"] != row["label"]
    ]
    if any(row["id"] == "dynamic-escalation" for row in rows):
        dynamic_row = next(row for row in rows if row["id"] == "dynamic-escalation")
        if dynamic_row["enforce"]["dynamic_decision_kind"] != "plan_required":
            disagreements.append("dynamic-escalation:recheck")
    acceptance_thresholds = {
        "simple_engine_fixture": {
            "main_backend_calls_per_mode": 1,
            "adaptive_extra_calls_tokens_tools_from_off": 0,
        },
        "policy_fixture": {
            "structural_adaptive_calls_tokens_tools": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "dynamic_escalation_enforce_kind": "plan_required",
        },
    }
    gate_pass = bool(
        exact_zero_delta
        and engine_extra_zero
        and confusion["fp"] == 0
        and confusion["fn"] == 0
        and not disagreements
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty_lines = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    source_sha256 = {
        relative_path: hashlib.sha256(
            (repository_root / relative_path).read_bytes()
        ).hexdigest()
        for relative_path in _SOURCE_HASH_PATHS
    }
    return {
        "schema_version": "ordinary-chat-adaptive-wave4-evidence-v1",
        "fixture_schema_version": payload["schema_version"],
        "samples_per_fixture": samples,
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
        "source_revision": revision,
        "source_sha256": source_sha256,
        "worktree": {"dirty": bool(dirty_lines), "changed_path_count": len(dirty_lines)},
        "reproduction_command": (
            "rtk proxy python tests/evaluation/evaluate_adaptive_runtime_wave4.py "
            "--fixtures tests/fixtures/adaptive_runtime/wave4_rollout_fixtures.json "
            "--output docs/superpowers/handoffs/2026-07-29-ordinary-chat-adaptive-runtime-wave4-measurement.json"
        ),
        "acceptance_thresholds": acceptance_thresholds,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "measurement": "perf_counter_ns median; informational only, not a CI threshold",
        },
        "structural_adaptive_delta": {
            "exact_zero_calls_tokens_tools": exact_zero_delta,
            "scope": "deterministic policy fixture; no model or tool is invoked",
        },
        "engine_simple_fixture": {
            "method": "one identical FakeBackend-backed ordinary-Chat request per mode; total main-response counters are reported separately from off-baseline deltas",
            "counters": engine_pairs,
            "delta_from_off": engine_delta,
            "exact_zero_extra_calls_tokens_tools": engine_extra_zero,
            "off_mode_note": (
                "off does not persist adaptive diagnostics; its one main backend call "
                "and exact fixture usage (4 input, 2 output) are used as the explicit baseline"
            ),
        },
        "confusion": confusion,
        "confusion_observations": (
            "Five fixtures yield six labelled observations because dynamic-escalation "
            "contributes both its initial no_plan label and its host-signalled "
            "plan_required recheck label."
        ),
        "disagreements": disagreements,
        "gate_pass": gate_pass,
        "fixtures": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.fixtures)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
