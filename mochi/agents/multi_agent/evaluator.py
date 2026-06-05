"""Multi-agent 評估策略與證據閘門型別。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

EvidenceGateStatus = Literal["verified", "skipped", "failed"]


@dataclass(frozen=True)
class EvidenceGateResult:
    """證據閘門檢查結果。"""

    status: EvidenceGateStatus
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """序列化為 JSON-safe dict。"""
        return {
            "status": self.status,
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LLMFirstScoringPolicy:
    """LLM-first 打分策略。"""

    policy_id: str = "llm_first_v1"
    scorer: Literal["llm_first"] = "llm_first"
    require_evidence_gate: bool = False
    default_evidence_gate_status: EvidenceGateStatus = "skipped"
    min_acceptable_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """序列化為 JSON-safe dict。"""
        return {
            "policy_id": self.policy_id,
            "scorer": self.scorer,
            "require_evidence_gate": self.require_evidence_gate,
            "default_evidence_gate_status": self.default_evidence_gate_status,
            "min_acceptable_score": self.min_acceptable_score,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CandidateScore:
    """候選答案評分。"""

    candidate_id: str
    score: float
    rationale: str
    evidence_gate: EvidenceGateResult

    def to_dict(self) -> dict[str, Any]:
        """序列化為 JSON-safe dict。"""
        return {
            "candidate_id": self.candidate_id,
            "score": self.score,
            "rationale": self.rationale,
            "evidence_gate": self.evidence_gate.to_dict(),
        }


@dataclass(frozen=True)
class CandidateVerification:
    """Candidate-level verification result."""

    candidate_id: str
    status: EvidenceGateStatus
    rationale: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-safe dict."""
        return {
            "candidate_id": self.candidate_id,
            "status": self.status,
            "rationale": self.rationale,
            "citations": [dict(item) for item in self.citations],
            "issues": list(self.issues),
            "metadata": dict(self.metadata),
        }

    def to_evidence_gate(self) -> EvidenceGateResult:
        """Map verification output into evidence gate format."""
        metadata = dict(self.metadata)
        metadata["citations"] = [dict(item) for item in self.citations]
        metadata["issues"] = list(self.issues)
        if self.status == "verified":
            return resolve_evidence_gate_status(verified=True, metadata=metadata)
        if self.status == "failed":
            return resolve_evidence_gate_status(error=self.rationale, metadata=metadata)
        return resolve_evidence_gate_status(skipped=True, metadata=metadata)


def resolve_evidence_gate_status(
    *,
    verified: bool | None = None,
    skipped: bool = False,
    error: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> EvidenceGateResult:
    """將 evidence gate 來源訊號映射為穩定狀態。"""
    if error:
        return EvidenceGateResult(status="failed", reason=str(error), metadata=dict(metadata or {}))
    if verified is True:
        return EvidenceGateResult(status="verified", metadata=dict(metadata or {}))
    if skipped or verified is None:
        return EvidenceGateResult(status="skipped", metadata=dict(metadata or {}))
    return EvidenceGateResult(status="failed", reason="verification returned false", metadata=dict(metadata or {}))


def evidence_gate_rank(status: EvidenceGateStatus) -> int:
    """Higher rank means stronger verification confidence."""
    if status == "verified":
        return 2
    if status == "skipped":
        return 1
    return 0
