"""Helpers for classifying recoverable agent-run resource failures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from mochi.backends.base import BackendRequestError

RecoveryIssueKind = Literal[
    "quota_exhausted",
    "rate_limited",
    "provider_unavailable",
    "provider_timeout",
    "local_model_unavailable",
]


@dataclass(frozen=True)
class AgentRunRecoveryIssue:
    """Structured classification for recoverable resource failures."""

    kind: RecoveryIssueKind
    reason: str
    retryable: bool = True
    backend_name: str | None = None
    status_code: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def classify_agent_run_recovery_issue(exc: BaseException) -> AgentRunRecoveryIssue | None:
    """Return a structured issue when the failure should pause for resources."""

    metadata = dict(getattr(exc, "metadata", {}) or {}) if isinstance(exc, BackendRequestError) else {}
    message = str(exc).strip() or exc.__class__.__name__
    response_text = metadata.get("response_text")
    status_code = _coerce_int(metadata.get("status_code"))
    backend_name = _clean_text(metadata.get("backend_name"))

    searchable = " ".join(
        part
        for part in (
            message,
            response_text if isinstance(response_text, str) else None,
            backend_name,
            _clean_text(metadata.get("request_url")),
            _clean_text(metadata.get("model")),
        )
        if part
    ).lower()

    if _matches_any(
        searchable,
        (
            "insufficient_quota",
            "quota exceeded",
            "exceeded your current quota",
            "billing",
            "insufficient balance",
            "credit balance",
            "余额不足",
            "餘額不足",
            "配額",
        ),
    ) or status_code == 402:
        return AgentRunRecoveryIssue(
            kind="quota_exhausted",
            reason=message,
            backend_name=backend_name,
            status_code=status_code,
            metadata=metadata,
        )

    if status_code == 429 or _matches_any(
        searchable,
        (
            "rate limit",
            "too many requests",
            "retry-after",
            "retry later",
        ),
    ):
        return AgentRunRecoveryIssue(
            kind="rate_limited",
            reason=message,
            backend_name=backend_name,
            status_code=status_code,
            metadata=metadata,
        )

    if status_code in {408, 504} or _matches_any(
        searchable,
        (
            "timed out",
            "timeout",
            "deadline exceeded",
        ),
    ):
        return AgentRunRecoveryIssue(
            kind="provider_timeout",
            reason=message,
            backend_name=backend_name,
            status_code=status_code,
            metadata=metadata,
        )

    local_backend = backend_name in {"ollama", "gguf", "llama_cpp_server", "safetensors", "vllm"}
    if local_backend and _matches_any(
        searchable,
        (
            "is not available",
            "not available",
            "runtime is unavailable",
            "backend unavailable",
            "no backend loaded",
            "model unavailable",
            "connection refused",
            "failed to load",
        ),
    ):
        return AgentRunRecoveryIssue(
            kind="local_model_unavailable",
            reason=message,
            backend_name=backend_name,
            status_code=status_code,
            metadata=metadata,
        )

    if status_code in {502, 503, 521, 522, 524, 529} or _matches_any(
        searchable,
        (
            "service unavailable",
            "temporarily unavailable",
            "upstream",
            "connection refused",
            "connection error",
            "network error",
            "provider unavailable",
        ),
    ):
        return AgentRunRecoveryIssue(
            kind="provider_unavailable",
            reason=message,
            backend_name=backend_name,
            status_code=status_code,
            metadata=metadata,
        )

    if _matches_any(
        searchable,
        (
            "configured model",
            "runtime is unavailable",
            "backend unavailable",
            "no backend loaded",
        ),
    ):
        return AgentRunRecoveryIssue(
            kind="local_model_unavailable",
            reason=message,
            backend_name=backend_name,
            status_code=status_code,
            metadata=metadata,
        )

    return None


def build_resource_exhaustion_report(
    issue: AgentRunRecoveryIssue,
    *,
    stage: str,
    exception: BaseException,
) -> dict[str, Any]:
    """Render a stable artifact payload for resource-gated run recovery."""

    return {
        "status": "awaiting_resources",
        "classification": issue.kind,
        "retryable": issue.retryable,
        "stage": stage,
        "reason": issue.reason,
        "backend_name": issue.backend_name,
        "status_code": issue.status_code,
        "exception_type": exception.__class__.__name__,
        "recommended_resume_conditions": _recommended_resume_conditions(issue.kind),
        "diagnostics": {
            key: value
            for key, value in issue.metadata.items()
            if key in {"api_mode", "request_url", "model", "stage", "response_text"}
        },
    }


def _recommended_resume_conditions(kind: RecoveryIssueKind) -> list[str]:
    if kind == "quota_exhausted":
        return [
            "Restore provider balance or quota.",
            "Confirm the billing or quota window has reset.",
            "Resume the run after capacity is available again.",
        ]
    if kind == "rate_limited":
        return [
            "Wait for the provider rate-limit window to reset.",
            "Reduce concurrency or model-call pressure if limits recur.",
            "Resume the run after the provider accepts requests again.",
        ]
    if kind == "provider_timeout":
        return [
            "Retry after the provider latency spike or timeout clears.",
            "Reduce request size or max output if timeouts continue.",
            "Resume the run when the upstream provider is responsive.",
        ]
    if kind == "local_model_unavailable":
        return [
            "Restart or restore the local model runtime.",
            "Verify the configured model backend is healthy and reachable.",
            "Resume the run after local inference becomes available.",
        ]
    return [
        "Wait for the provider or runtime to recover.",
        "Resume the run after capacity is available again.",
    ]


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _matches_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)
