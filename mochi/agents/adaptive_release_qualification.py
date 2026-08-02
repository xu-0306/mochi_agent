"""Safe, versioned release qualification for the adaptive Chat runtime.

The runner is deliberately opt-in: constructing a fixture or evidence record
does not contact a model.  A caller must explicitly pass
``allow_external_model=True`` before an Engine is initialized with its
configured backend.  Evidence is a redacted measurement record: it never
contains fixture prompts, model output, URLs, API keys, tool arguments, or
exception messages.

Wave 5 uses :class:`ExternalQualificationRunner` to produce bounded evidence
from a real configured model.  Wave 6 consumes that evidence plus a compact
human review record and returns a *recommendation* only; it never changes a
live setting or bypasses the Settings CAS/rollback contract.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from mochi.agents.invocation import AgentInvocationRequest, AgentInvocationResult
from mochi.backends.base import BaseLLMBackend
from mochi.backends.types import GenerationResult, Message, ModelInfo, StreamChunk, ToolSchema
from mochi.config.schema import MochiConfig


EXTERNAL_QUALIFICATION_FIXTURE_VERSION = "ordinary-chat-adaptive-wave5-fixture-v1"
EXTERNAL_QUALIFICATION_EVIDENCE_VERSION = "ordinary-chat-adaptive-wave5-evidence-v2"
CANARY_REVIEW_VERSION = "ordinary-chat-adaptive-wave6-review-v2"
CANARY_DECISION_VERSION = "ordinary-chat-adaptive-wave6-decision-v1"

QualificationDecision = Literal["no_plan", "plan_required"]
QualificationStatus = Literal["passed", "failed", "unavailable"]
ReviewDisposition = Literal["accept", "hold", "rollback_shadow", "rollback_off"]
CanaryDisposition = Literal["keep_enforce", "hold", "rollback_shadow", "rollback_off"]

_IDENTIFIER_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
_CODE_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
_HASH_RE = re.compile(r"[a-f0-9]{64}")
_TIMESTAMP_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})"
)
_SENSITIVE_FIXTURE_RE = re.compile(
    r"(?ix)(?:"
    r"api[ _-]?key\s*[:=]|authorization\s*:|bearer\s+|"
    r"(?:password|secret|token|access_token|client_secret)\s*[:=]\s*\S+|"
    r"\bsk-[a-z0-9_-]{8,}|\bAKIA[0-9A-Z]{16}\b|"
    r"\b(?:ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b|"
    r"\bxox(?:b|p|a|r|s)-[A-Za-z0-9-]{10,}|"
    r"-----BEGIN[ ](?:[A-Z]+[ ])?PRIVATE[ ]KEY-----"
    r")"
)
_DECISIONS = frozenset({"no_plan", "plan_required"})
_STATUSES = frozenset({"passed", "failed", "unavailable"})
_REVIEW_DISPOSITIONS = frozenset({"accept", "hold", "rollback_shadow", "rollback_off"})
_CANARY_DISPOSITIONS = frozenset({"keep_enforce", "hold", "rollback_shadow", "rollback_off"})
_MAX_FIXTURES = 20
_MAX_MESSAGE_CHARS = 600
_MAX_CALLS = 4
_MAX_COUNTER = 1_000_000
_MAX_REASON_CODES = 8
_INITIALIZE_TIMEOUT_SECONDS = 30.0
_FIXTURE_TIMEOUT_SECONDS = 60.0


class AdaptiveQualificationError(ValueError):
    """The qualification input or evidence cannot be safely accepted."""


class ExternalModelConsentRequired(AdaptiveQualificationError):
    """Raised before a runner could initialize or contact an external model."""


class _BackendCallBudgetExceeded(RuntimeError):
    """Internal redacted sentinel raised before an over-budget backend call."""


def _strict_json_loads(raw_bytes: bytes, *, document_name: str) -> Any:
    """Decode JSON while rejecting duplicate keys at every nesting level."""

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AdaptiveQualificationError(f"{document_name} must not contain duplicate keys")
            result[key] = value
        return result

    try:
        return json.loads(raw_bytes.decode("utf-8"), object_pairs_hook=no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdaptiveQualificationError(f"{document_name} must be UTF-8 JSON") from exc


class QualificationEngine(Protocol):
    async def initialize(self) -> None: ...

    async def invoke(self, request: AgentInvocationRequest) -> AgentInvocationResult: ...

    async def close(self) -> None: ...


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
    actual = frozenset(value)
    if actual == expected:
        return
    details: list[str] = []
    unexpected = sorted(actual - expected)
    missing = sorted(expected - actual)
    if unexpected:
        details.append(f"unexpected={unexpected}")
    if missing:
        details.append(f"missing={missing}")
    raise AdaptiveQualificationError(f"{name} has invalid keys ({'; '.join(details)})")


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise AdaptiveQualificationError(f"{name} must be a bounded lowercase identifier")
    return value


def _code(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _CODE_RE.fullmatch(value):
        raise AdaptiveQualificationError(f"{name} must be a bounded lowercase code")
    return value


def _sha256(value: str | bytes) -> str:
    encoded = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(encoded).hexdigest()


def _hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
        raise AdaptiveQualificationError(f"{name} must be a SHA-256 hex digest")
    return value


def _timestamp(value: Any, name: str = "generated_at_utc") -> str:
    if not isinstance(value, str) or not _TIMESTAMP_RE.fullmatch(value):
        raise AdaptiveQualificationError(f"{name} must be an ISO-8601 UTC-offset timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AdaptiveQualificationError(f"{name} is malformed") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise AdaptiveQualificationError(f"{name} must include an offset")
    return value


def _bounded_int(value: Any, name: str, *, maximum: int = _MAX_COUNTER) -> int:
    if type(value) is not int or value < 0 or value > maximum:
        raise AdaptiveQualificationError(f"{name} must be a bounded non-negative integer")
    return value


def _reason_codes(value: Any, name: str = "reason_codes") -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise AdaptiveQualificationError(f"{name} must be a list")
    cleaned: list[str] = []
    for item in value:
        code = _code(item, name)
        if code not in cleaned:
            cleaned.append(code)
    if len(cleaned) > _MAX_REASON_CODES:
        raise AdaptiveQualificationError(f"{name} exceeds {_MAX_REASON_CODES} items")
    return tuple(cleaned)


def _decision(value: Any, name: str, *, allow_none: bool = False) -> QualificationDecision | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or value not in _DECISIONS:
        raise AdaptiveQualificationError(f"{name} is invalid")
    return cast(QualificationDecision, value)


@dataclass(frozen=True)
class ExternalQualificationFixture:
    """One bounded, public-safe external-model qualification input."""

    fixture_version: str
    fixture_id: str
    message: str
    expected_complexity: QualificationDecision
    max_backend_calls: int
    require_visible_response: bool

    def __post_init__(self) -> None:
        if self.fixture_version != EXTERNAL_QUALIFICATION_FIXTURE_VERSION:
            raise AdaptiveQualificationError("unsupported fixture_version")
        object.__setattr__(self, "fixture_id", _identifier(self.fixture_id, "fixture_id"))
        if not isinstance(self.message, str):
            raise AdaptiveQualificationError("message must be a string")
        message = " ".join(self.message.split())
        if not message or len(message) > _MAX_MESSAGE_CHARS:
            raise AdaptiveQualificationError("message must be non-empty and bounded")
        if _SENSITIVE_FIXTURE_RE.search(message):
            raise AdaptiveQualificationError("message appears to contain sensitive data")
        object.__setattr__(self, "message", message)
        object.__setattr__(
            self,
            "expected_complexity",
            cast(QualificationDecision, _decision(self.expected_complexity, "expected_complexity")),
        )
        object.__setattr__(
            self,
            "max_backend_calls",
            _bounded_int(self.max_backend_calls, "max_backend_calls", maximum=_MAX_CALLS),
        )
        if type(self.require_visible_response) is not bool:
            raise AdaptiveQualificationError("require_visible_response must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_version": self.fixture_version,
            "fixture_id": self.fixture_id,
            "message": self.message,
            "expected_complexity": self.expected_complexity,
            "max_backend_calls": self.max_backend_calls,
            "require_visible_response": self.require_visible_response,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExternalQualificationFixture":
        _exact_keys(
            value,
            frozenset(
                {
                    "fixture_version",
                    "fixture_id",
                    "message",
                    "expected_complexity",
                    "max_backend_calls",
                    "require_visible_response",
                }
            ),
            "external qualification fixture",
        )
        return cls(
            fixture_version=value["fixture_version"],
            fixture_id=value["fixture_id"],
            message=value["message"],
            expected_complexity=value["expected_complexity"],
            max_backend_calls=value["max_backend_calls"],
            require_visible_response=value["require_visible_response"],
        )


def load_external_qualification_fixtures(path: str | Path) -> tuple[ExternalQualificationFixture, ...]:
    """Load strict fixtures; callers must never accept arbitrary prompt files."""
    try:
        raw_bytes = Path(path).read_bytes()
    except OSError as exc:
        raise AdaptiveQualificationError("fixture file must be readable UTF-8 JSON") from exc
    return _fixtures_from_document_bytes(raw_bytes)


def _fixtures_from_document_bytes(
    raw_bytes: bytes,
) -> tuple[ExternalQualificationFixture, ...]:
    payload = _strict_json_loads(raw_bytes, document_name="fixture document")
    if not isinstance(payload, Mapping):
        raise AdaptiveQualificationError("fixture document must be an object")
    _exact_keys(payload, frozenset({"schema_version", "fixtures"}), "fixture document")
    if payload["schema_version"] != EXTERNAL_QUALIFICATION_FIXTURE_VERSION:
        raise AdaptiveQualificationError("unsupported fixture document version")
    raw_fixtures = payload["fixtures"]
    if not isinstance(raw_fixtures, list) or not raw_fixtures or len(raw_fixtures) > _MAX_FIXTURES:
        raise AdaptiveQualificationError("fixture document must contain 1-20 fixtures")
    fixtures = tuple(
        ExternalQualificationFixture.from_dict(item)
        for item in raw_fixtures
        if isinstance(item, Mapping)
    )
    if len(fixtures) != len(raw_fixtures):
        raise AdaptiveQualificationError("each fixture must be an object")
    ids = [fixture.fixture_id for fixture in fixtures]
    if len(set(ids)) != len(ids):
        raise AdaptiveQualificationError("fixture ids must be unique")
    return fixtures


@dataclass(frozen=True)
class QualificationResult:
    fixture_id: str
    expected_complexity: QualificationDecision
    max_backend_calls: int
    require_visible_response: bool
    observed_complexity: QualificationDecision | None
    status: QualificationStatus
    backend_calls: int
    input_tokens: int
    output_tokens: int
    tool_calls: int
    usage_observed: bool
    visible_response: bool
    failure_code: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "fixture_id", _identifier(self.fixture_id, "fixture_id"))
        object.__setattr__(
            self,
            "expected_complexity",
            cast(QualificationDecision, _decision(self.expected_complexity, "expected_complexity")),
        )
        object.__setattr__(
            self,
            "max_backend_calls",
            _bounded_int(self.max_backend_calls, "max_backend_calls", maximum=_MAX_CALLS),
        )
        if type(self.require_visible_response) is not bool:
            raise AdaptiveQualificationError("require_visible_response must be boolean")
        object.__setattr__(
            self,
            "observed_complexity",
            _decision(self.observed_complexity, "observed_complexity", allow_none=True),
        )
        if self.status not in _STATUSES:
            raise AdaptiveQualificationError("qualification result status is invalid")
        for field_name in ("backend_calls", "input_tokens", "output_tokens", "tool_calls"):
            object.__setattr__(self, field_name, _bounded_int(getattr(self, field_name), field_name))
        if type(self.usage_observed) is not bool:
            raise AdaptiveQualificationError("usage_observed must be boolean")
        if type(self.visible_response) is not bool:
            raise AdaptiveQualificationError("visible_response must be boolean")
        if self.failure_code is not None:
            object.__setattr__(self, "failure_code", _code(self.failure_code, "failure_code"))
        if self.status == "passed" and self.failure_code is not None:
            raise AdaptiveQualificationError("passed result cannot have failure_code")
        if self.status != "passed" and self.failure_code is None:
            raise AdaptiveQualificationError("non-passed result requires failure_code")
        if self.status == "passed":
            if self.observed_complexity != self.expected_complexity:
                raise AdaptiveQualificationError("passed result must match expected complexity")
            if self.backend_calls > self.max_backend_calls:
                raise AdaptiveQualificationError("passed result exceeds backend call budget")
            if self.tool_calls:
                raise AdaptiveQualificationError("passed result cannot contain tool calls")
            if self.require_visible_response and not self.visible_response:
                raise AdaptiveQualificationError("passed result requires a visible response")
            if not self.usage_observed:
                raise AdaptiveQualificationError("passed result requires observed backend usage")

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "expected_complexity": self.expected_complexity,
            "max_backend_calls": self.max_backend_calls,
            "require_visible_response": self.require_visible_response,
            "observed_complexity": self.observed_complexity,
            "status": self.status,
            "backend_calls": self.backend_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "tool_calls": self.tool_calls,
            "usage_observed": self.usage_observed,
            "visible_response": self.visible_response,
            "failure_code": self.failure_code,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "QualificationResult":
        _exact_keys(
            value,
            frozenset(
                {
                    "fixture_id",
                    "expected_complexity",
                    "max_backend_calls",
                    "require_visible_response",
                    "observed_complexity",
                    "status",
                    "backend_calls",
                    "input_tokens",
                    "output_tokens",
                    "tool_calls",
                    "usage_observed",
                    "visible_response",
                    "failure_code",
                }
            ),
            "qualification result",
        )
        return cls(
            fixture_id=value["fixture_id"],
            expected_complexity=value["expected_complexity"],
            max_backend_calls=value["max_backend_calls"],
            require_visible_response=value["require_visible_response"],
            observed_complexity=value["observed_complexity"],
            status=value["status"],
            backend_calls=value["backend_calls"],
            input_tokens=value["input_tokens"],
            output_tokens=value["output_tokens"],
            tool_calls=value["tool_calls"],
            usage_observed=value["usage_observed"],
            visible_response=value["visible_response"],
            failure_code=value["failure_code"],
        )


@dataclass(frozen=True)
class ExternalQualificationEvidence:
    """Redacted Wave 5 evidence; fixture prompts and model output are excluded."""

    schema_version: str
    fixture_schema_version: str
    fixture_sha256: str
    generated_at_utc: str
    model_fingerprint: str
    backend_fingerprint: str
    external_consent: bool
    results: tuple[QualificationResult, ...]
    summary: Mapping[str, int]
    gate_pass: bool

    def __post_init__(self) -> None:
        if self.schema_version != EXTERNAL_QUALIFICATION_EVIDENCE_VERSION:
            raise AdaptiveQualificationError("unsupported evidence schema_version")
        if self.fixture_schema_version != EXTERNAL_QUALIFICATION_FIXTURE_VERSION:
            raise AdaptiveQualificationError("unsupported evidence fixture schema_version")
        for name in ("fixture_sha256", "model_fingerprint", "backend_fingerprint"):
            object.__setattr__(self, name, _hash(getattr(self, name), name))
        object.__setattr__(self, "generated_at_utc", _timestamp(self.generated_at_utc))
        if type(self.external_consent) is not bool or not self.external_consent:
            raise AdaptiveQualificationError("external_consent must be explicitly true")
        if not self.results or len(self.results) > _MAX_FIXTURES:
            raise AdaptiveQualificationError("evidence results must be bounded and non-empty")
        result_ids = [result.fixture_id for result in self.results]
        if len(set(result_ids)) != len(result_ids):
            raise AdaptiveQualificationError("evidence result ids must be unique")
        if not isinstance(self.summary, Mapping):
            raise AdaptiveQualificationError("summary must be an object")
        _exact_keys(self.summary, frozenset({"passed", "failed", "unavailable"}), "evidence summary")
        normalized_summary = {
            name: _bounded_int(self.summary[name], f"summary.{name}")
            for name in ("passed", "failed", "unavailable")
        }
        if sum(normalized_summary.values()) != len(self.results):
            raise AdaptiveQualificationError("summary does not match result count")
        observed_summary = {
            name: sum(result.status == name for result in self.results)
            for name in ("passed", "failed", "unavailable")
        }
        if normalized_summary != observed_summary:
            raise AdaptiveQualificationError("summary does not match result statuses")
        object.__setattr__(self, "summary", normalized_summary)
        if type(self.gate_pass) is not bool or self.gate_pass != (normalized_summary["passed"] == len(self.results)):
            raise AdaptiveQualificationError("gate_pass does not match result statuses")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "fixture_schema_version": self.fixture_schema_version,
            "fixture_sha256": self.fixture_sha256,
            "generated_at_utc": self.generated_at_utc,
            "model_fingerprint": self.model_fingerprint,
            "backend_fingerprint": self.backend_fingerprint,
            "external_consent": self.external_consent,
            "results": [result.to_dict() for result in self.results],
            "summary": dict(self.summary),
            "gate_pass": self.gate_pass,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExternalQualificationEvidence":
        _exact_keys(
            value,
            frozenset(
                {
                    "schema_version",
                    "fixture_schema_version",
                    "fixture_sha256",
                    "generated_at_utc",
                    "model_fingerprint",
                    "backend_fingerprint",
                    "external_consent",
                    "results",
                    "summary",
                    "gate_pass",
                }
            ),
            "external qualification evidence",
        )
        raw_results = value["results"]
        if not isinstance(raw_results, list):
            raise AdaptiveQualificationError("evidence results must be a list")
        if not isinstance(value["summary"], Mapping):
            raise AdaptiveQualificationError("evidence summary must be an object")
        results = tuple(
            QualificationResult.from_dict(item)
            for item in raw_results
            if isinstance(item, Mapping)
        )
        if len(results) != len(raw_results):
            raise AdaptiveQualificationError("each evidence result must be an object")
        return cls(
            schema_version=value["schema_version"],
            fixture_schema_version=value["fixture_schema_version"],
            fixture_sha256=value["fixture_sha256"],
            generated_at_utc=value["generated_at_utc"],
            model_fingerprint=value["model_fingerprint"],
            backend_fingerprint=value["backend_fingerprint"],
            external_consent=value["external_consent"],
            results=results,
            summary=value["summary"],
            gate_pass=value["gate_pass"],
        )


class _CountingBackend(BaseLLMBackend):
    """Count only public numeric generation measurements; retain no content."""

    def __init__(self, delegate: BaseLLMBackend) -> None:
        self._delegate = delegate
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.tool_calls = 0
        self.usage_observations = 0
        self._call_limit: int | None = None

    def set_call_limit(self, call_limit: int) -> None:
        self._call_limit = _bounded_int(call_limit, "call_limit", maximum=_MAX_COUNTER)

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: float = 1.0,
        min_p: float = 0.0,
        top_k: int = 0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repeat_penalty: float = 1.0,
        reasoning_effort: str | None = None,
        stream: bool = False,
    ) -> GenerationResult | Any:
        if self._call_limit is not None and self.calls >= self._call_limit:
            raise _BackendCallBudgetExceeded()
        self.calls += 1
        result = await self._delegate.generate(
            messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
            reasoning_effort=reasoning_effort,
            stream=stream,
        )
        if isinstance(result, GenerationResult):
            self.input_tokens += _numeric_counter(result.input_tokens)
            self.output_tokens += _numeric_counter(result.output_tokens)
            self.tool_calls += len(result.tool_calls or [])
            if _usage_is_observed(result.input_tokens, result.output_tokens):
                self.usage_observations += 1
        elif hasattr(result, "__aiter__"):
            return self._count_stream(cast(AsyncIterator[Any], result))
        return result

    async def _count_stream(self, stream: AsyncIterator[Any]) -> AsyncIterator[Any]:
        async for chunk in stream:
            if isinstance(chunk, StreamChunk):
                if chunk.tool_call_delta is not None:
                    self.tool_calls += 1
                input_tokens = getattr(chunk, "input_tokens", None)
                output_tokens = getattr(chunk, "output_tokens", None)
                if type(input_tokens) is int and type(output_tokens) is int:
                    self.input_tokens += _numeric_counter(input_tokens)
                    self.output_tokens += _numeric_counter(output_tokens)
                if _usage_is_observed(input_tokens, output_tokens):
                    self.usage_observations += 1
            yield chunk

    def supports_tool_calling(self) -> bool:
        return self._delegate.supports_tool_calling()

    def get_model_info(self) -> ModelInfo:
        return self._delegate.get_model_info()

    async def health_check(self) -> bool:
        return await self._delegate.health_check()

    async def probe_tool_calling(self) -> dict[str, Any] | None:
        return await self._delegate.probe_tool_calling()

    async def prime_model_info(self) -> None:
        await self._delegate.prime_model_info()

    async def close(self) -> None:
        # The Engine owns the configured backend and closes it exactly once.
        return None


def _numeric_counter(value: Any) -> int:
    if type(value) is int and 0 <= value <= _MAX_COUNTER:
        return value
    return 0


def _usage_is_observed(input_tokens: Any, output_tokens: Any) -> bool:
    """Only provider-supplied, bounded, non-zero usage can satisfy the gate."""
    return (
        type(input_tokens) is int
        and type(output_tokens) is int
        and 0 <= input_tokens <= _MAX_COUNTER
        and 0 <= output_tokens <= _MAX_COUNTER
        and (input_tokens > 0 or output_tokens > 0)
    )


def _observed_complexity(result: AgentInvocationResult) -> QualificationDecision | None:
    runtime = result.diagnostics.adaptive_runtime
    if not isinstance(runtime, Mapping):
        return None
    complexity = runtime.get("complexity")
    if not isinstance(complexity, Mapping):
        return None
    decision = complexity.get("decision")
    if not isinstance(decision, Mapping):
        return None
    value = decision.get("kind")
    return cast(QualificationDecision, value) if value in _DECISIONS else None


def _backend_fingerprint(backend: BaseLLMBackend) -> str:
    # Do not serialize model metadata: provider implementations may include a
    # remote URL, local path, or organization-specific name.
    return _sha256(f"{type(backend).__module__}.{type(backend).__qualname__}")


def _model_fingerprint(backend: BaseLLMBackend) -> str:
    """Hash the active backend identity without disclosing provider metadata."""
    try:
        info = backend.get_model_info()
        if not isinstance(info, ModelInfo):
            raise TypeError("active backend did not return ModelInfo")
        name = info.name
        backend_type = info.backend_type
        provider = info.provider
        if (
            not isinstance(name, str)
            or not name.strip()
            or len(name.strip()) > 512
            or not isinstance(backend_type, str)
            or not backend_type.strip()
            or len(backend_type.strip()) > 128
            or (provider is not None and (not isinstance(provider, str) or len(provider.strip()) > 128))
        ):
            raise ValueError("active ModelInfo is invalid")
        identity = {
            "backend_type": backend_type.strip(),
            "provider": (provider or "").strip(),
            # The value is deliberately present only in the hash preimage.  It
            # distinguishes an active fallback/model swap without exposing a
            # name, URL, local path, or organization identifier in evidence.
            "name": name.strip(),
        }
    except Exception as exc:
        raise AdaptiveQualificationError("active model identity is unavailable") from exc
    return _sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


class ExternalQualificationRunner:
    """Run bounded, tool-disabled fixtures against an explicitly allowed model."""

    def __init__(
        self,
        *,
        engine_factory: Callable[[MochiConfig], QualificationEngine],
        now: Callable[[], datetime] | None = None,
        initialize_timeout_seconds: float = _INITIALIZE_TIMEOUT_SECONDS,
        fixture_timeout_seconds: float = _FIXTURE_TIMEOUT_SECONDS,
    ) -> None:
        self._engine_factory = engine_factory
        self._now = now or (lambda: datetime.now(tz=UTC))
        if not 0 < initialize_timeout_seconds <= _INITIALIZE_TIMEOUT_SECONDS:
            raise AdaptiveQualificationError("initialize timeout must be bounded")
        if not 0 < fixture_timeout_seconds <= _FIXTURE_TIMEOUT_SECONDS:
            raise AdaptiveQualificationError("fixture timeout must be bounded")
        self._initialize_timeout_seconds = initialize_timeout_seconds
        self._fixture_timeout_seconds = fixture_timeout_seconds

    async def run(
        self,
        *,
        config: MochiConfig,
        fixtures: Sequence[ExternalQualificationFixture],
        fixture_document_bytes: bytes,
        allow_external_model: bool,
    ) -> ExternalQualificationEvidence:
        if not allow_external_model:
            raise ExternalModelConsentRequired(
                "external qualification requires explicit allow_external_model=True"
            )
        if not fixtures or len(fixtures) > _MAX_FIXTURES:
            raise AdaptiveQualificationError("fixtures must be bounded and non-empty")
        document_fixtures = _fixtures_from_document_bytes(fixture_document_bytes)
        if tuple(fixtures) != document_fixtures:
            raise AdaptiveQualificationError("fixtures do not exactly match fixture document")
        engine = self._engine_factory(config)
        try:
            try:
                await asyncio.wait_for(engine.initialize(), timeout=self._initialize_timeout_seconds)
                backend = _qualification_backend(engine)
                model_fingerprint = _model_fingerprint(backend)
            except Exception:
                results = tuple(
                    QualificationResult(
                        fixture_id=fixture.fixture_id,
                        expected_complexity=fixture.expected_complexity,
                        max_backend_calls=fixture.max_backend_calls,
                        require_visible_response=fixture.require_visible_response,
                        observed_complexity=None,
                        status="unavailable",
                        backend_calls=0,
                        input_tokens=0,
                        output_tokens=0,
                        tool_calls=0,
                        usage_observed=False,
                        visible_response=False,
                        failure_code="backend_unavailable",
                    )
                    for fixture in fixtures
                )
                return _evidence(
                    config=config,
                    fixture_document_bytes=fixture_document_bytes,
                    model_fingerprint=_sha256("model-unavailable"),
                    backend_fingerprint=_sha256("backend-unavailable"),
                    results=results,
                    now=self._now,
                )

            counting_backend = _CountingBackend(backend)
            results_list: list[QualificationResult] = []
            for fixture in fixtures:
                results_list.append(
                    await self._run_fixture(engine, counting_backend, fixture)
                )
            results = tuple(results_list)
            return _evidence(
                config=config,
                fixture_document_bytes=fixture_document_bytes,
                model_fingerprint=model_fingerprint,
                backend_fingerprint=_backend_fingerprint(backend),
                results=results,
                now=self._now,
            )
        finally:
            await engine.close()

    async def _run_fixture(
        self,
        engine: QualificationEngine,
        backend: _CountingBackend,
        fixture: ExternalQualificationFixture,
    ) -> QualificationResult:
        before = (
            backend.calls,
            backend.input_tokens,
            backend.output_tokens,
            backend.tool_calls,
            backend.usage_observations,
        )
        backend.set_call_limit(before[0] + fixture.max_backend_calls)
        try:
            result = await asyncio.wait_for(
                engine.invoke(
                    AgentInvocationRequest(
                        message=fixture.message,
                        session_id=f"external-qualification-{fixture.fixture_id}",
                        tool_mode="disabled",
                        execution_profile="chat",
                        persist_session=False,
                        persist_turn_events=False,
                        persist_learning=False,
                        isolate_context=True,
                        backend_override=backend,
                        max_iterations_override=1,
                    )
                ),
                timeout=self._fixture_timeout_seconds,
            )
        except _BackendCallBudgetExceeded:
            return self._unavailable_result(fixture, backend, before, "backend_call_budget_exceeded")
        except TimeoutError:
            return self._unavailable_result(fixture, backend, before, "qualification_timeout")
        except Exception:
            return self._unavailable_result(fixture, backend, before, "backend_error")

        calls = backend.calls - before[0]
        input_tokens = backend.input_tokens - before[1]
        output_tokens = backend.output_tokens - before[2]
        tool_calls = backend.tool_calls - before[3]
        usage_observed = backend.usage_observations > before[4]
        observed = _observed_complexity(result)
        visible_response = bool(result.content.strip())
        failure_code: str | None = None
        if observed != fixture.expected_complexity:
            failure_code = "complexity_mismatch"
        elif calls > fixture.max_backend_calls:
            failure_code = "backend_call_budget_exceeded"
        elif tool_calls:
            failure_code = "unexpected_tool_call"
        elif not usage_observed:
            failure_code = "usage_unavailable"
        elif fixture.require_visible_response and not visible_response:
            failure_code = "empty_response"
        return QualificationResult(
            fixture_id=fixture.fixture_id,
            expected_complexity=fixture.expected_complexity,
            max_backend_calls=fixture.max_backend_calls,
            require_visible_response=fixture.require_visible_response,
            observed_complexity=observed,
            status="failed" if failure_code is not None else "passed",
            backend_calls=calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_calls=tool_calls,
            usage_observed=usage_observed,
            visible_response=visible_response,
            failure_code=failure_code,
        )

    @staticmethod
    def _unavailable_result(
        fixture: ExternalQualificationFixture,
        backend: _CountingBackend,
        before: tuple[int, int, int, int, int],
        failure_code: str,
    ) -> QualificationResult:
        return QualificationResult(
            fixture_id=fixture.fixture_id,
            expected_complexity=fixture.expected_complexity,
            max_backend_calls=fixture.max_backend_calls,
            require_visible_response=fixture.require_visible_response,
            observed_complexity=None,
            status="unavailable",
            backend_calls=backend.calls - before[0],
            input_tokens=backend.input_tokens - before[1],
            output_tokens=backend.output_tokens - before[2],
            tool_calls=backend.tool_calls - before[3],
            usage_observed=backend.usage_observations > before[4],
            visible_response=False,
            failure_code=failure_code,
        )


def _qualification_backend(engine: QualificationEngine) -> BaseLLMBackend:
    router = getattr(engine, "_router", None)
    backend = getattr(router, "active", None)
    if not isinstance(backend, BaseLLMBackend):
        raise AdaptiveQualificationError("configured qualification backend is unavailable")
    return backend


def _evidence(
    *,
    config: MochiConfig,
    fixture_document_bytes: bytes,
    model_fingerprint: str,
    backend_fingerprint: str,
    results: tuple[QualificationResult, ...],
    now: Callable[[], datetime],
) -> ExternalQualificationEvidence:
    summary = {
        status: sum(result.status == status for result in results)
        for status in ("passed", "failed", "unavailable")
    }
    return ExternalQualificationEvidence(
        schema_version=EXTERNAL_QUALIFICATION_EVIDENCE_VERSION,
        fixture_schema_version=EXTERNAL_QUALIFICATION_FIXTURE_VERSION,
        fixture_sha256=_sha256(fixture_document_bytes),
        generated_at_utc=now().astimezone(UTC).isoformat(),
        model_fingerprint=model_fingerprint,
        backend_fingerprint=backend_fingerprint,
        external_consent=True,
        results=results,
        summary=summary,
        gate_pass=summary["passed"] == len(results),
    )


@dataclass(frozen=True)
class CanaryReview:
    """A bounded human decision over one immutable qualification evidence file."""

    review_version: str
    qualification_evidence_sha256: str
    reviewer_id: str
    disposition: ReviewDisposition
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.review_version != CANARY_REVIEW_VERSION:
            raise AdaptiveQualificationError("unsupported review_version")
        object.__setattr__(
            self,
            "qualification_evidence_sha256",
            _hash(self.qualification_evidence_sha256, "qualification_evidence_sha256"),
        )
        object.__setattr__(self, "reviewer_id", _identifier(self.reviewer_id, "reviewer_id"))
        if self.disposition not in _REVIEW_DISPOSITIONS:
            raise AdaptiveQualificationError("review disposition is invalid")
        object.__setattr__(self, "reason_codes", _reason_codes(self.reason_codes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "review_version": self.review_version,
            "qualification_evidence_sha256": self.qualification_evidence_sha256,
            "reviewer_id": self.reviewer_id,
            "disposition": self.disposition,
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CanaryReview":
        _exact_keys(
            value,
            frozenset(
                {
                    "review_version",
                    "qualification_evidence_sha256",
                    "reviewer_id",
                    "disposition",
                    "reason_codes",
                }
            ),
            "canary review",
        )
        return cls(
            review_version=value["review_version"],
            qualification_evidence_sha256=value["qualification_evidence_sha256"],
            reviewer_id=value["reviewer_id"],
            disposition=value["disposition"],
            reason_codes=value["reason_codes"],
        )


@dataclass(frozen=True)
class CanaryDecision:
    """A non-mutating Wave 6 recommendation bound to evidence and review."""

    decision_version: str
    qualification_evidence_sha256: str
    review_sha256: str
    disposition: CanaryDisposition
    reason_codes: tuple[str, ...]
    settings_patch: Mapping[str, Any] | None

    def __post_init__(self) -> None:
        if self.decision_version != CANARY_DECISION_VERSION:
            raise AdaptiveQualificationError("unsupported decision_version")
        object.__setattr__(
            self,
            "qualification_evidence_sha256",
            _hash(self.qualification_evidence_sha256, "qualification_evidence_sha256"),
        )
        object.__setattr__(self, "review_sha256", _hash(self.review_sha256, "review_sha256"))
        if self.disposition not in _CANARY_DISPOSITIONS:
            raise AdaptiveQualificationError("canary disposition is invalid")
        object.__setattr__(self, "reason_codes", _reason_codes(self.reason_codes))
        expected_patch: Mapping[str, Any] | None
        if self.disposition == "rollback_shadow":
            expected_patch = {"agent": {"complexity_mode": "shadow"}}
        elif self.disposition == "rollback_off":
            expected_patch = {"agent": {"complexity_mode": "off"}}
        else:
            expected_patch = None
        if self.settings_patch != expected_patch:
            raise AdaptiveQualificationError("settings_patch does not match disposition")
        object.__setattr__(
            self,
            "settings_patch",
            json.loads(json.dumps(expected_patch)) if expected_patch is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_version": self.decision_version,
            "qualification_evidence_sha256": self.qualification_evidence_sha256,
            "review_sha256": self.review_sha256,
            "disposition": self.disposition,
            "reason_codes": list(self.reason_codes),
            "settings_patch": self.settings_patch,
        }


def evaluate_canary(
    evidence: ExternalQualificationEvidence,
    review: CanaryReview,
    *,
    evidence_document_bytes: bytes,
    review_document_bytes: bytes,
) -> CanaryDecision:
    """Return a recommendation; applying it still requires Settings ETag/CAS."""
    raw_evidence = _strict_json_loads(evidence_document_bytes, document_name="evidence document")
    raw_review = _strict_json_loads(review_document_bytes, document_name="review document")
    if not isinstance(raw_evidence, Mapping) or ExternalQualificationEvidence.from_dict(raw_evidence) != evidence:
        raise AdaptiveQualificationError("evidence bytes do not match evidence object")
    if not isinstance(raw_review, Mapping) or CanaryReview.from_dict(raw_review) != review:
        raise AdaptiveQualificationError("review bytes do not match review object")
    evidence_hash = _sha256(evidence_document_bytes)
    if review.qualification_evidence_sha256 != evidence_hash:
        raise AdaptiveQualificationError("review is not bound to this evidence")
    review_hash = _sha256(review_document_bytes)
    reasons = list(review.reason_codes)
    if not evidence.gate_pass:
        disposition: CanaryDisposition = "rollback_shadow"
        if "qualification_failed" not in reasons:
            reasons.append("qualification_failed")
    elif review.disposition == "accept":
        disposition = "keep_enforce"
    elif review.disposition == "hold":
        disposition = "hold"
    else:
        disposition = cast(CanaryDisposition, review.disposition)
    patch: Mapping[str, Any] | None
    if disposition == "rollback_shadow":
        patch = {"agent": {"complexity_mode": "shadow"}}
    elif disposition == "rollback_off":
        patch = {"agent": {"complexity_mode": "off"}}
    else:
        patch = None
    return CanaryDecision(
        decision_version=CANARY_DECISION_VERSION,
        qualification_evidence_sha256=evidence_hash,
        review_sha256=review_hash,
        disposition=disposition,
        reason_codes=tuple(reasons),
        settings_patch=patch,
    )


def load_evidence(path: str | Path) -> ExternalQualificationEvidence:
    try:
        raw_bytes = Path(path).read_bytes()
    except OSError as exc:
        raise AdaptiveQualificationError("evidence file must be readable UTF-8 JSON") from exc
    value = _strict_json_loads(raw_bytes, document_name="evidence document")
    if not isinstance(value, Mapping):
        raise AdaptiveQualificationError("evidence document must be an object")
    return ExternalQualificationEvidence.from_dict(value)


def load_canary_review(path: str | Path) -> CanaryReview:
    try:
        raw_bytes = Path(path).read_bytes()
    except OSError as exc:
        raise AdaptiveQualificationError("review file must be readable UTF-8 JSON") from exc
    value = _strict_json_loads(raw_bytes, document_name="review document")
    if not isinstance(value, Mapping):
        raise AdaptiveQualificationError("review document must be an object")
    return CanaryReview.from_dict(value)


__all__ = [
    "AdaptiveQualificationError",
    "CANARY_DECISION_VERSION",
    "CANARY_REVIEW_VERSION",
    "CanaryDecision",
    "CanaryReview",
    "EXTERNAL_QUALIFICATION_EVIDENCE_VERSION",
    "EXTERNAL_QUALIFICATION_FIXTURE_VERSION",
    "ExternalModelConsentRequired",
    "ExternalQualificationEvidence",
    "ExternalQualificationFixture",
    "ExternalQualificationRunner",
    "QualificationResult",
    "evaluate_canary",
    "load_canary_review",
    "load_evidence",
    "load_external_qualification_fixtures",
]
