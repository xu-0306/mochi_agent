from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mochi.agents.adaptive_release_qualification import (
    AdaptiveQualificationError,
    CANARY_REVIEW_VERSION,
    EXTERNAL_QUALIFICATION_EVIDENCE_VERSION,
    EXTERNAL_QUALIFICATION_FIXTURE_VERSION,
    CanaryReview,
    ExternalModelConsentRequired,
    ExternalQualificationEvidence,
    ExternalQualificationFixture,
    ExternalQualificationRunner,
    QualificationResult,
    evaluate_canary,
    load_evidence,
    load_external_qualification_fixtures,
)
from mochi.agents.invocation import AgentInvocationDiagnostics, AgentInvocationRequest, AgentInvocationResult
from mochi.agents.engine import AgentEngine
from mochi.backends.base import BaseLLMBackend
from mochi.backends.types import GenerationResult, Message, ModelInfo, StreamChunk
from mochi.config.schema import MochiConfig


class _FixtureBackend(BaseLLMBackend):
    async def generate(self, messages: list[Message], **_: Any) -> GenerationResult:
        if (
            messages
            and messages[0].role == "system"
            and "bounded conversation interpreter" in messages[0].content
        ):
            context = json.loads(messages[-1].content)
            current_turn = context["current_turn"]
            is_effectful = "migration" in current_turn["content"].lower()
            payload = {
                "current_speech_act": (
                    "request_execution" if is_effectful else "request_information"
                ),
                "task_relation": "start" if is_effectful else "standalone",
                "objective": (
                    "Fix the migration, update its schema, and run validation."
                    if is_effectful
                    else None
                ),
                "operations": (
                    ["workspace_write", "execution"] if is_effectful else ["conversation"]
                ),
                "deliverables": (
                    [
                        {
                            "kind": "workspace_artifact",
                            "target_hint": "migrations/schema.sql",
                            "required": True,
                            "acceptance_criteria": ["updated", "validated"],
                            "status": "pending",
                            "source_turn_ids": [current_turn["turn_id"]],
                        },
                        {
                            "kind": "command_result",
                            "target_hint": "migration validation",
                            "required": True,
                            "acceptance_criteria": ["validation passes", "schema matches"],
                            "status": "pending",
                            "source_turn_ids": [current_turn["turn_id"]],
                        },
                    ]
                    if is_effectful
                    else []
                ),
                "resolved_references": [],
                "positive_constraints": [],
                "negative_constraints": [],
                "mutation_requirement": "required" if is_effectful else "unknown",
                "clarification": None,
                "confidence": 0.9,
                "evidence": [
                    {
                        "statement": "The current turn directly states the request.",
                        "source": "current_turn",
                        "source_turn_ids": [current_turn["turn_id"]],
                    }
                ],
            }
            return GenerationResult(
                content=json.dumps(payload), input_tokens=5, output_tokens=3
            )
        return GenerationResult(content="safe fixture answer", input_tokens=5, output_tokens=3)

    def supports_tool_calling(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name="fixture", backend_type="fixture", context_length=128_000)

    async def health_check(self) -> bool:
        return True


@dataclass
class _FixtureEngine:
    backend: BaseLLMBackend
    fail: bool = False

    def __post_init__(self) -> None:
        self._router = SimpleNamespace(active=self.backend)
        self.requests = []
        self.closed = False
        self.initialized = False

    async def initialize(self) -> None:
        self.initialized = True

    async def invoke(self, request: Any) -> AgentInvocationResult:
        self.requests.append(request)
        if self.fail:
            raise RuntimeError("Bearer this-must-not-appear")
        generated = await request.backend_override.generate([])
        decision = "plan_required" if "migration" in request.message.lower() else "no_plan"
        return AgentInvocationResult(
            content=generated.content,
            diagnostics=AgentInvocationDiagnostics(
                execution_profile="chat",
                tool_mode="disabled",
                adaptive_runtime={"complexity": {"decision": {"kind": decision}}},
            ),
        )

    async def close(self) -> None:
        self.closed = True


def _config(tmp_path: Path) -> MochiConfig:
    return MochiConfig.model_validate(
        {
            "model": "https://model.example/v1",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db")},
        }
    )


def _fixture(*, identifier: str = "simple-information-en", expected: str = "no_plan") -> ExternalQualificationFixture:
    message = "Explain why an expired JWT is rejected."
    if expected == "plan_required":
        message = "Fix the migration, update its schema, and run validation."
    return ExternalQualificationFixture(
        fixture_version=EXTERNAL_QUALIFICATION_FIXTURE_VERSION,
        fixture_id=identifier,
        message=message,
        expected_complexity=expected,  # type: ignore[arg-type]
        max_backend_calls=2,
        require_visible_response=True,
    )


def _evidence(*, passed: bool) -> ExternalQualificationEvidence:
    result = QualificationResult(
        fixture_id="simple-information-en",
        expected_complexity="no_plan",
        max_backend_calls=2,
        require_visible_response=True,
        observed_complexity="no_plan" if passed else None,
        status="passed" if passed else "unavailable",
        backend_calls=1 if passed else 0,
        input_tokens=5 if passed else 0,
        output_tokens=3 if passed else 0,
        tool_calls=0,
        usage_observed=passed,
        visible_response=passed,
        failure_code=None if passed else "backend_unavailable",
    )
    return ExternalQualificationEvidence(
        schema_version=EXTERNAL_QUALIFICATION_EVIDENCE_VERSION,
        fixture_schema_version=EXTERNAL_QUALIFICATION_FIXTURE_VERSION,
        fixture_sha256="a" * 64,
        generated_at_utc="2026-07-29T00:00:00+00:00",
        model_fingerprint="b" * 64,
        backend_fingerprint="c" * 64,
        external_consent=True,
        results=(result,),
        summary={"passed": int(passed), "failed": 0, "unavailable": int(not passed)},
        gate_pass=passed,
    )


def _evidence_bytes(evidence: ExternalQualificationEvidence) -> bytes:
    return json.dumps(evidence.to_dict(), ensure_ascii=False, indent=2).encode("utf-8")


def _review(evidence: ExternalQualificationEvidence, disposition: str = "accept") -> CanaryReview:
    digest = hashlib.sha256(_evidence_bytes(evidence)).hexdigest()
    return CanaryReview(
        review_version=CANARY_REVIEW_VERSION,
        qualification_evidence_sha256=digest,
        reviewer_id="release-reviewer",
        disposition=disposition,  # type: ignore[arg-type]
        reason_codes=("fixture-review",),
    )


def _review_bytes(review: CanaryReview) -> bytes:
    return json.dumps(review.to_dict(), ensure_ascii=False, indent=2).encode("utf-8")


def test_fixture_loader_rejects_secret_bearing_input(tmp_path: Path) -> None:
    fixture_path = tmp_path / "fixtures.json"
    fixture_path.write_text(
        json.dumps(
            {
                "schema_version": EXTERNAL_QUALIFICATION_FIXTURE_VERSION,
                "fixtures": [
                    {
                        **_fixture().to_dict(),
                        "message": "Authorization: Bearer should-not-run",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(AdaptiveQualificationError, match="sensitive"):
        load_external_qualification_fixtures(fixture_path)


def test_checked_in_wave5_fixture_document_is_strict() -> None:
    fixture_path = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "adaptive_runtime"
        / "wave5_external_qualification_fixtures.json"
    )
    fixtures = load_external_qualification_fixtures(fixture_path)
    assert [fixture.fixture_id for fixture in fixtures] == [
        "simple-information-en",
        "simple-information-zh",
        "complex-effectful",
    ]


@pytest.mark.asyncio
async def test_runner_requires_consent_before_constructing_engine(tmp_path: Path) -> None:
    constructed = False

    def factory(_: MochiConfig) -> _FixtureEngine:
        nonlocal constructed
        constructed = True
        return _FixtureEngine(_FixtureBackend())

    runner = ExternalQualificationRunner(engine_factory=factory)
    with pytest.raises(ExternalModelConsentRequired):
        await runner.run(
            config=_config(tmp_path),
            fixtures=(_fixture(),),
            fixture_document_bytes=b"{}",
            allow_external_model=False,
        )
    assert constructed is False


@pytest.mark.asyncio
async def test_runner_records_only_redacted_numeric_evidence(tmp_path: Path) -> None:
    engine = _FixtureEngine(_FixtureBackend())
    fixture = _fixture()
    runner = ExternalQualificationRunner(
        engine_factory=lambda _: engine,
        now=lambda: datetime(2026, 7, 29, tzinfo=UTC),
    )
    evidence = await runner.run(
        config=_config(tmp_path),
        fixtures=(fixture,),
        fixture_document_bytes=json.dumps(
            {"schema_version": EXTERNAL_QUALIFICATION_FIXTURE_VERSION, "fixtures": [fixture.to_dict()]}
        ).encode("utf-8"),
        allow_external_model=True,
    )
    payload = evidence.to_dict()
    rendered = json.dumps(payload, ensure_ascii=False)
    assert evidence.gate_pass is True
    assert payload["results"][0]["backend_calls"] == 1
    assert payload["results"][0]["input_tokens"] == 5
    assert payload["results"][0]["output_tokens"] == 3
    assert "expired JWT" not in rendered
    assert "safe fixture answer" not in rendered
    assert engine.requests[0].tool_mode == "disabled"
    assert engine.requests[0].persist_session is False
    assert engine.requests[0].persist_learning is False
    assert engine.requests[0].isolate_context is True
    assert engine.requests[0].session_id == "external-qualification-simple-information-en"
    assert engine.closed is True


@pytest.mark.asyncio
async def test_runner_rejects_fixture_bytes_that_differ_from_executed_fixtures(tmp_path: Path) -> None:
    runner = ExternalQualificationRunner(engine_factory=lambda _: _FixtureEngine(_FixtureBackend()))
    document_fixture = _fixture()
    different_fixture = _fixture(identifier="different-fixture")
    document = json.dumps(
        {"schema_version": EXTERNAL_QUALIFICATION_FIXTURE_VERSION, "fixtures": [document_fixture.to_dict()]}
    ).encode("utf-8")
    with pytest.raises(AdaptiveQualificationError, match="exactly match"):
        await runner.run(
            config=_config(tmp_path),
            fixtures=(different_fixture,),
            fixture_document_bytes=document,
            allow_external_model=True,
        )


@pytest.mark.asyncio
async def test_runner_redacts_backend_exception(tmp_path: Path) -> None:
    engine = _FixtureEngine(_FixtureBackend(), fail=True)
    runner = ExternalQualificationRunner(engine_factory=lambda _: engine)
    evidence = await runner.run(
        config=_config(tmp_path),
        fixtures=(_fixture(),),
        fixture_document_bytes=json.dumps(
            {"schema_version": EXTERNAL_QUALIFICATION_FIXTURE_VERSION, "fixtures": [_fixture().to_dict()]}
        ).encode("utf-8"),
        allow_external_model=True,
    )
    result = evidence.results[0]
    assert evidence.gate_pass is False
    assert result.status == "unavailable"
    assert result.failure_code == "backend_error"
    assert "Bearer" not in json.dumps(evidence.to_dict())


@pytest.mark.asyncio
async def test_runner_uses_real_engine_with_tool_disabled_ephemeral_request(tmp_path: Path) -> None:
    engine = AgentEngine(_config(tmp_path))
    backend = _FixtureBackend()
    engine._router._active = backend  # noqa: SLF001 - avoid a real backend in this harness

    async def initialized() -> None:
        engine._initialized = True  # noqa: SLF001 - test-only initialized engine

    engine.initialize = initialized  # type: ignore[method-assign]
    fixtures = (
        _fixture(),
        _fixture(identifier="complex-effectful", expected="plan_required"),
    )
    evidence = await ExternalQualificationRunner(engine_factory=lambda _: engine).run(
        config=_config(tmp_path),
        fixtures=fixtures,
        fixture_document_bytes=json.dumps(
            {
                "schema_version": EXTERNAL_QUALIFICATION_FIXTURE_VERSION,
                "fixtures": [fixture.to_dict() for fixture in fixtures],
            }
        ).encode("utf-8"),
        allow_external_model=True,
    )
    assert evidence.gate_pass is True
    assert [result.observed_complexity for result in evidence.results] == [
        "no_plan",
        "plan_required",
    ]
    assert all(0 < result.backend_calls <= 2 for result in evidence.results)
    assert [result.tool_calls for result in evidence.results] == [0, 0]


@pytest.mark.asyncio
async def test_real_engine_isolation_never_sends_default_history_memory_or_skills(tmp_path: Path) -> None:
    class RecordingBackend(_FixtureBackend):
        def __init__(self) -> None:
            self.rendered_messages: list[str] = []

        async def generate(self, messages: list[Message], **kwargs: Any) -> GenerationResult:
            self.rendered_messages.extend(message.content for message in messages)
            return await super().generate(messages, **kwargs)

    secret_history = "QUALIFICATION-DEFAULT-HISTORY-SECRET"
    secret_memory = "QUALIFICATION-MEMORY-SECRET"
    engine = AgentEngine(_config(tmp_path))
    backend = RecordingBackend()
    engine._router._active = backend  # noqa: SLF001 - real Engine isolation harness

    async def initialized() -> None:
        engine._initialized = True  # noqa: SLF001 - avoid real external backend setup

    engine.initialize = initialized  # type: ignore[method-assign]
    default_context = await engine._get_context("default")  # noqa: SLF001
    default_context.add_message(Message(role="assistant", content=secret_history))
    await engine._memory_store.save(secret_memory, "test", {})  # noqa: SLF001
    fixture = _fixture()
    await ExternalQualificationRunner(engine_factory=lambda _: engine).run(
        config=_config(tmp_path),
        fixtures=(fixture,),
        fixture_document_bytes=json.dumps(
            {"schema_version": EXTERNAL_QUALIFICATION_FIXTURE_VERSION, "fixtures": [fixture.to_dict()]}
        ).encode("utf-8"),
        allow_external_model=True,
    )
    rendered = "\n".join(backend.rendered_messages)
    assert secret_history not in rendered
    assert secret_memory not in rendered


@pytest.mark.asyncio
async def test_real_engine_rejects_unsafe_isolated_request_before_backend_call(tmp_path: Path) -> None:
    engine = AgentEngine(_config(tmp_path))
    backend = _FixtureBackend()
    engine._router._active = backend  # noqa: SLF001

    async def initialized() -> None:
        engine._initialized = True  # noqa: SLF001

    engine.initialize = initialized  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="unsafe context"):
        await engine.invoke(
            AgentInvocationRequest(
                message="unsafe",
                tool_mode="disabled",
                persist_session=True,
                isolate_context=True,
                backend_override=backend,
            )
        )


def test_canary_accepts_bound_passing_evidence() -> None:
    evidence = _evidence(passed=True)
    review = _review(evidence)
    decision = evaluate_canary(
        evidence,
        review,
        evidence_document_bytes=_evidence_bytes(evidence),
        review_document_bytes=_review_bytes(review),
    )
    assert decision.disposition == "keep_enforce"
    assert decision.settings_patch is None


def test_canary_forces_shadow_rollback_when_qualification_failed() -> None:
    evidence = _evidence(passed=False)
    review = _review(evidence, "accept")
    decision = evaluate_canary(
        evidence,
        review,
        evidence_document_bytes=_evidence_bytes(evidence),
        review_document_bytes=_review_bytes(review),
    )
    assert decision.disposition == "rollback_shadow"
    assert decision.settings_patch == {"agent": {"complexity_mode": "shadow"}}
    assert "qualification_failed" in decision.reason_codes


def test_canary_rejects_unbound_human_review() -> None:
    evidence = _evidence(passed=True)
    review = CanaryReview(
        review_version=CANARY_REVIEW_VERSION,
        qualification_evidence_sha256="d" * 64,
        reviewer_id="release-reviewer",
        disposition="accept",
        reason_codes=(),
    )
    with pytest.raises(AdaptiveQualificationError, match="not bound"):
        evaluate_canary(
            evidence,
            review,
            evidence_document_bytes=_evidence_bytes(evidence),
            review_document_bytes=_review_bytes(review),
        )


def test_wave5_cli_requires_explicit_consent_before_reading_config(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "must-not-exist.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "tests" / "evaluation" / "evaluate_adaptive_runtime_wave5.py"),
            "--config",
            str(tmp_path / "must-not-be-read.yaml"),
            "--fixtures",
            str(
                root
                / "tests"
                / "fixtures"
                / "adaptive_runtime"
                / "wave5_external_qualification_fixtures.json"
            ),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "--allow-external-model is required" in completed.stderr
    assert not output.exists()


def test_wave6_cli_emits_non_mutating_recommendation(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    evidence = _evidence(passed=True)
    evidence_path = tmp_path / "evidence.json"
    review_path = tmp_path / "review.json"
    output = tmp_path / "recommendation.json"
    review = _review(evidence)
    evidence_path.write_bytes(_evidence_bytes(evidence))
    review_path.write_bytes(_review_bytes(review))
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "tests" / "evaluation" / "evaluate_adaptive_runtime_wave6.py"),
            "--evidence",
            str(evidence_path),
            "--review",
            str(review_path),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(output.read_text(encoding="utf-8")) == evaluate_canary(
        evidence,
        review,
        evidence_document_bytes=_evidence_bytes(evidence),
        review_document_bytes=_review_bytes(review),
    ).to_dict()


def test_passing_result_rejects_forged_semantics() -> None:
    with pytest.raises(AdaptiveQualificationError, match="match expected complexity"):
        QualificationResult(
            fixture_id="simple-information-en",
            expected_complexity="no_plan",
            max_backend_calls=1,
            require_visible_response=True,
            observed_complexity=None,
            status="passed",
            backend_calls=0,
            input_tokens=0,
            output_tokens=0,
            tool_calls=0,
            usage_observed=True,
            visible_response=True,
            failure_code=None,
        )


@pytest.mark.asyncio
async def test_runner_hard_stops_before_a_third_backend_call(tmp_path: Path) -> None:
    class ThreeCallEngine(_FixtureEngine):
        async def invoke(self, request: Any) -> AgentInvocationResult:
            for _ in range(3):
                await request.backend_override.generate([])
            raise AssertionError("third delegate call should have been blocked")

    backend = _FixtureBackend()
    engine = ThreeCallEngine(backend)
    fixture = _fixture()
    evidence = await ExternalQualificationRunner(engine_factory=lambda _: engine).run(
        config=_config(tmp_path),
        fixtures=(fixture,),
        fixture_document_bytes=json.dumps(
            {"schema_version": EXTERNAL_QUALIFICATION_FIXTURE_VERSION, "fixtures": [fixture.to_dict()]}
        ).encode("utf-8"),
        allow_external_model=True,
    )
    result = evidence.results[0]
    assert result.failure_code == "backend_call_budget_exceeded"
    assert result.backend_calls == 2


@pytest.mark.asyncio
async def test_runner_fails_closed_on_fixture_timeout(tmp_path: Path) -> None:
    class SlowEngine(_FixtureEngine):
        async def invoke(self, request: Any) -> AgentInvocationResult:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    fixture = _fixture()
    evidence = await ExternalQualificationRunner(
        engine_factory=lambda _: SlowEngine(_FixtureBackend()),
        fixture_timeout_seconds=0.01,
    ).run(
        config=_config(tmp_path),
        fixtures=(fixture,),
        fixture_document_bytes=json.dumps(
            {"schema_version": EXTERNAL_QUALIFICATION_FIXTURE_VERSION, "fixtures": [fixture.to_dict()]}
        ).encode("utf-8"),
        allow_external_model=True,
    )
    assert evidence.results[0].failure_code == "qualification_timeout"


@pytest.mark.asyncio
async def test_runner_marks_stream_usage_unknown_and_counts_tool_delta(tmp_path: Path) -> None:
    class StreamingBackend(_FixtureBackend):
        async def generate(self, messages: list[Message], **_: Any) -> Any:
            async def stream() -> Any:
                yield StreamChunk(tool_call_delta=object())
                yield StreamChunk(is_final=True)

            return stream()

    class StreamingEngine(_FixtureEngine):
        async def invoke(self, request: Any) -> AgentInvocationResult:
            stream = await request.backend_override.generate([], stream=True)
            async for _ in stream:
                pass
            return AgentInvocationResult(
                content="streamed",
                diagnostics=AgentInvocationDiagnostics(
                    execution_profile="chat",
                    tool_mode="disabled",
                    adaptive_runtime={"complexity": {"decision": {"kind": "no_plan"}}},
                ),
            )

    fixture = _fixture()
    evidence = await ExternalQualificationRunner(
        engine_factory=lambda _: StreamingEngine(StreamingBackend())
    ).run(
        config=_config(tmp_path),
        fixtures=(fixture,),
        fixture_document_bytes=json.dumps(
            {"schema_version": EXTERNAL_QUALIFICATION_FIXTURE_VERSION, "fixtures": [fixture.to_dict()]}
        ).encode("utf-8"),
        allow_external_model=True,
    )
    result = evidence.results[0]
    assert result.tool_calls == 1
    assert result.usage_observed is False
    assert result.failure_code == "unexpected_tool_call"
    assert evidence.gate_pass is False


@pytest.mark.asyncio
async def test_runner_does_not_treat_stream_usage_unknown_as_zero_cost_pass(tmp_path: Path) -> None:
    class UsageUnknownBackend(_FixtureBackend):
        async def generate(self, messages: list[Message], **_: Any) -> Any:
            async def stream() -> Any:
                yield StreamChunk(is_final=True)

            return stream()

    class StreamingEngine(_FixtureEngine):
        async def invoke(self, request: Any) -> AgentInvocationResult:
            stream = await request.backend_override.generate([], stream=True)
            async for _ in stream:
                pass
            return AgentInvocationResult(
                content="streamed",
                diagnostics=AgentInvocationDiagnostics(
                    execution_profile="chat",
                    tool_mode="disabled",
                    adaptive_runtime={"complexity": {"decision": {"kind": "no_plan"}}},
                ),
            )

    fixture = _fixture()
    evidence = await ExternalQualificationRunner(
        engine_factory=lambda _: StreamingEngine(UsageUnknownBackend())
    ).run(
        config=_config(tmp_path),
        fixtures=(fixture,),
        fixture_document_bytes=json.dumps(
            {"schema_version": EXTERNAL_QUALIFICATION_FIXTURE_VERSION, "fixtures": [fixture.to_dict()]}
        ).encode("utf-8"),
        allow_external_model=True,
    )
    assert evidence.results[0].failure_code == "usage_unavailable"
    assert evidence.gate_pass is False


@pytest.mark.asyncio
async def test_runner_rejects_nonstream_default_zero_usage(tmp_path: Path) -> None:
    class ZeroUsageBackend(_FixtureBackend):
        async def generate(self, messages: list[Message], **_: Any) -> GenerationResult:
            return GenerationResult(content="answer")

    fixture = _fixture()
    evidence = await ExternalQualificationRunner(
        engine_factory=lambda _: _FixtureEngine(ZeroUsageBackend())
    ).run(
        config=_config(tmp_path),
        fixtures=(fixture,),
        fixture_document_bytes=json.dumps(
            {"schema_version": EXTERNAL_QUALIFICATION_FIXTURE_VERSION, "fixtures": [fixture.to_dict()]}
        ).encode("utf-8"),
        allow_external_model=True,
    )
    assert evidence.results[0].usage_observed is False
    assert evidence.results[0].failure_code == "usage_unavailable"
    assert evidence.gate_pass is False


@pytest.mark.asyncio
async def test_runner_rejects_explicit_zero_stream_usage(tmp_path: Path) -> None:
    class ZeroUsageStreamBackend(_FixtureBackend):
        async def generate(self, messages: list[Message], **_: Any) -> Any:
            async def stream() -> Any:
                chunk = StreamChunk(is_final=True)
                chunk.input_tokens = 0
                chunk.output_tokens = 0
                yield chunk

            return stream()

    class StreamingEngine(_FixtureEngine):
        async def invoke(self, request: Any) -> AgentInvocationResult:
            stream = await request.backend_override.generate([], stream=True)
            async for _ in stream:
                pass
            return AgentInvocationResult(
                content="streamed",
                diagnostics=AgentInvocationDiagnostics(
                    execution_profile="chat",
                    tool_mode="disabled",
                    adaptive_runtime={"complexity": {"decision": {"kind": "no_plan"}}},
                ),
            )

    fixture = _fixture()
    evidence = await ExternalQualificationRunner(
        engine_factory=lambda _: StreamingEngine(ZeroUsageStreamBackend())
    ).run(
        config=_config(tmp_path),
        fixtures=(fixture,),
        fixture_document_bytes=json.dumps(
            {"schema_version": EXTERNAL_QUALIFICATION_FIXTURE_VERSION, "fixtures": [fixture.to_dict()]}
        ).encode("utf-8"),
        allow_external_model=True,
    )
    assert evidence.results[0].usage_observed is False
    assert evidence.results[0].failure_code == "usage_unavailable"


@pytest.mark.asyncio
async def test_runner_fails_closed_when_active_model_identity_is_unavailable(tmp_path: Path) -> None:
    class NoModelInfoBackend(_FixtureBackend):
        def get_model_info(self) -> ModelInfo:
            raise RuntimeError("private model endpoint must not leak")

    fixture = _fixture()
    evidence = await ExternalQualificationRunner(
        engine_factory=lambda _: _FixtureEngine(NoModelInfoBackend())
    ).run(
        config=_config(tmp_path),
        fixtures=(fixture,),
        fixture_document_bytes=json.dumps(
            {"schema_version": EXTERNAL_QUALIFICATION_FIXTURE_VERSION, "fixtures": [fixture.to_dict()]}
        ).encode("utf-8"),
        allow_external_model=True,
    )
    assert evidence.gate_pass is False
    assert evidence.results[0].status == "unavailable"
    assert evidence.results[0].failure_code == "backend_unavailable"
    assert "private model endpoint" not in json.dumps(evidence.to_dict())


@pytest.mark.asyncio
async def test_evidence_model_fingerprint_uses_active_backend_identity(tmp_path: Path) -> None:
    class NamedBackend(_FixtureBackend):
        def __init__(self, name: str) -> None:
            self._name = name

        def get_model_info(self) -> ModelInfo:
            return ModelInfo(name=self._name, backend_type="fixture", provider="test")

    fixture = _fixture()
    document = json.dumps(
        {"schema_version": EXTERNAL_QUALIFICATION_FIXTURE_VERSION, "fixtures": [fixture.to_dict()]}
    ).encode("utf-8")
    first = await ExternalQualificationRunner(
        engine_factory=lambda _: _FixtureEngine(NamedBackend("active-a"))
    ).run(
        config=_config(tmp_path), fixtures=(fixture,), fixture_document_bytes=document, allow_external_model=True
    )
    second = await ExternalQualificationRunner(
        engine_factory=lambda _: _FixtureEngine(NamedBackend("active-b"))
    ).run(
        config=_config(tmp_path), fixtures=(fixture,), fixture_document_bytes=document, allow_external_model=True
    )
    assert first.model_fingerprint != second.model_fingerprint


def test_wave6_binds_exact_evidence_and_review_bytes() -> None:
    evidence = _evidence(passed=True)
    evidence_bytes = _evidence_bytes(evidence)
    review = _review(evidence)
    with pytest.raises(AdaptiveQualificationError, match="not bound"):
        evaluate_canary(
            evidence,
            review,
            evidence_document_bytes=json.dumps(evidence.to_dict(), separators=(",", ":")).encode("utf-8"),
            review_document_bytes=_review_bytes(review),
        )


@pytest.mark.parametrize(
    "message",
    [
        "AKIA1234567890ABCDEF must not run",
        "ghp_abcdefghijklmnopqrstuvwxyz123456 should not run",
        "github_pat_abcdefghijklmnopqrstuvwxyz_123456 should not run",
        "xoxb-123456789012-abcdef should not run",
        "-----BEGIN PRIVATE KEY----- should not run",
        "access_token=super-secret-value should not run",
        "client_secret: super-secret-value should not run",
    ],
)
def test_fixture_rejects_common_credential_shapes(message: str) -> None:
    with pytest.raises(AdaptiveQualificationError, match="sensitive"):
        ExternalQualificationFixture(
            fixture_version=EXTERNAL_QUALIFICATION_FIXTURE_VERSION,
            fixture_id="credential-shape",
            message=message,
            expected_complexity="no_plan",
            max_backend_calls=1,
            require_visible_response=True,
        )


def test_strict_loader_rejects_duplicate_json_keys_and_old_evidence(tmp_path: Path) -> None:
    fixture_path = tmp_path / "duplicate.json"
    fixture_path.write_bytes(
        b'{"schema_version":"ordinary-chat-adaptive-wave5-fixture-v1",'
        b'"schema_version":"ordinary-chat-adaptive-wave5-fixture-v1","fixtures":[]}'
    )
    with pytest.raises(AdaptiveQualificationError, match="duplicate"):
        load_external_qualification_fixtures(fixture_path)
    evidence_path = tmp_path / "old-evidence.json"
    old = _evidence(passed=True).to_dict()
    old["schema_version"] = "ordinary-chat-adaptive-wave5-evidence-v1"
    evidence_path.write_text(json.dumps(old), encoding="utf-8")
    with pytest.raises(AdaptiveQualificationError, match="unsupported evidence"):
        load_evidence(evidence_path)
