"""Structured debate context management for multi-agent debate."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import re
from typing import Any, Literal

from mochi.agents.context_snapshot import estimate_text_tokens

ContextManagementMode = Literal["compress_and_continue", "strict_fail_on_overflow"]


@dataclass(frozen=True)
class DebateTurn:
    """一輪辯論中的單次發言。"""

    role_id: str
    round_index: int
    content: str
    candidate_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DebateClaimCard:
    """從辯論發言中抽出的 claim card。"""

    claim_id: str
    speaker_role_id: str
    round_index: int
    claim: str
    rationale: str
    evidence_refs: list[str] = field(default_factory=list)
    objection_target_claim_id: str | None = None
    uncertainty: str = ""
    status: str = "active"
    support_status: str = "needs_evidence"
    confidence: float = 0.0
    citation_refs: list[str] = field(default_factory=list)
    source_quality_tier: str = "low"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DebateRoleState:
    """角色在辯論中的當前立場。"""

    role_id: str
    stance_summary: str
    latest_candidate_id: str | None = None
    latest_round_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DebateStateSummary:
    """模型可見的結構化辯論狀態。"""

    task_brief: str
    global_summary: str
    round_summaries: list[dict[str, Any]]
    role_stance_summaries: list[dict[str, Any]]
    claim_cards: list[dict[str, Any]]
    open_objections: list[str]
    open_questions: list[str]
    evidence_refs: list[str]
    selected_constraints: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DebateContextPolicy:
    """辯論 context 管理策略。"""

    mode: ContextManagementMode = "compress_and_continue"
    max_input_tokens: int = 4096
    keep_recent_rounds: int = 2
    enable_verifier_chunking: bool = True
    enable_llm_summary_refinement: bool = True

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any] | None) -> DebateContextPolicy:
        default = cls()
        if not isinstance(metadata, dict):
            return default
        evaluation_policy = metadata.get("evaluation_policy")
        if not isinstance(evaluation_policy, dict):
            return default
        payload = evaluation_policy.get("context_management")
        if not isinstance(payload, dict):
            return default

        mode = str(payload.get("mode") or default.mode).strip()
        if mode not in {"compress_and_continue", "strict_fail_on_overflow"}:
            mode = default.mode
        max_input_tokens = payload.get("max_input_tokens")
        keep_recent_rounds = payload.get("keep_recent_rounds")
        enable_verifier_chunking = payload.get("enable_verifier_chunking")
        enable_llm_summary_refinement = payload.get("enable_llm_summary_refinement")
        return cls(
            mode=mode,
            max_input_tokens=max(512, int(max_input_tokens))
            if isinstance(max_input_tokens, int)
            else default.max_input_tokens,
            keep_recent_rounds=max(1, int(keep_recent_rounds))
            if isinstance(keep_recent_rounds, int)
            else default.keep_recent_rounds,
            enable_verifier_chunking=(
                bool(enable_verifier_chunking)
                if isinstance(enable_verifier_chunking, bool)
                else default.enable_verifier_chunking
            ),
            enable_llm_summary_refinement=(
                bool(enable_llm_summary_refinement)
                if isinstance(enable_llm_summary_refinement, bool)
                else default.enable_llm_summary_refinement
            ),
        )


@dataclass(frozen=True)
class DebateContextSnapshot:
    """辯論 prompt preflight 診斷。"""

    role_id: str
    stage: str
    estimated_prompt_tokens: int
    reserved_output_tokens: int
    max_input_tokens: int
    usage_ratio: float
    compaction_level: str
    truncated: bool
    used_chunking: bool
    largest_section: str
    overflow: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DebateContextOverflowError(RuntimeError):
    """Raised when context exceeds policy and strict mode forbids continuation."""


class DebateContextManager:
    """Maintain canonical debate turns and build budgeted model-visible state."""

    def __init__(self, *, task_input: str, guidance_messages: list[str], policy: DebateContextPolicy) -> None:
        self._task_input = task_input.strip()
        self._guidance_messages = [item.strip() for item in guidance_messages if item.strip()]
        self._policy = policy
        self._turns: list[DebateTurn] = []
        self._claim_cards: list[DebateClaimCard] = []
        self._role_states: dict[str, DebateRoleState] = {}
        self._round_summaries: list[dict[str, Any]] = []
        self._open_objections: list[str] = []
        self._open_questions: list[str] = []
        self._evidence_refs: list[str] = []
        self._selected_constraints: list[str] = list(self._guidance_messages)
        self._latest_snapshot: DebateContextSnapshot | None = None

    @property
    def latest_snapshot(self) -> DebateContextSnapshot | None:
        return self._latest_snapshot

    def all_turns(self) -> list[DebateTurn]:
        return list(self._turns)

    def current_state(self) -> DebateStateSummary:
        role_states = [state.to_dict() for state in self._role_states.values()]
        claim_cards = [card.to_dict() for card in self._claim_cards]
        return DebateStateSummary(
            task_brief=self._task_input,
            global_summary=self._build_global_summary(),
            round_summaries=list(self._round_summaries),
            role_stance_summaries=role_states,
            claim_cards=claim_cards,
            open_objections=list(self._open_objections),
            open_questions=list(self._open_questions),
            evidence_refs=list(self._evidence_refs),
            selected_constraints=list(self._selected_constraints),
        )

    def register_turn(
        self,
        *,
        role_id: str,
        round_index: int,
        content: str,
        candidate_id: str | None = None,
    ) -> DebateTurn:
        turn = DebateTurn(
            role_id=role_id,
            round_index=round_index,
            content=content.strip(),
            candidate_id=candidate_id,
        )
        self._turns.append(turn)
        claim_cards = self._extract_claim_cards(turn)
        self._claim_cards.extend(claim_cards)
        self._evidence_refs = _dedupe(self._evidence_refs + [ref for card in claim_cards for ref in card.evidence_refs])
        self._open_questions = _dedupe(self._open_questions + _extract_questions(turn.content))
        self._open_objections = _dedupe(self._open_objections + _extract_objections(turn, claim_cards))
        self._role_states[role_id] = DebateRoleState(
            role_id=role_id,
            stance_summary=_truncate(turn.content, 240),
            latest_candidate_id=candidate_id,
            latest_round_index=round_index,
        )
        self._update_round_summary(round_index)
        return turn

    def build_role_prompt(
        self,
        *,
        role_id: str,
        role_title: str,
        stage: str,
        reserved_output_tokens: int,
    ) -> tuple[str, DebateContextSnapshot]:
        return self._build_budgeted_prompt(
            role_id=role_id,
            role_title=role_title,
            stage=stage,
            reserved_output_tokens=reserved_output_tokens,
        )

    def build_evaluator_prompt(
        self,
        *,
        protocol_name: str,
        reserved_output_tokens: int,
    ) -> tuple[str, DebateContextSnapshot]:
        state = self.current_state()
        sections = {
            "task": f"Task:\n{self._task_input}",
            "protocol": f"Protocol:\n{protocol_name}",
            "constraints": self._render_constraints(),
            "global_summary": f"Global summary:\n{state.global_summary}",
            "claim_cards": self._render_claim_cards(state.claim_cards),
            "role_stances": self._render_role_stances(state.role_stance_summaries),
            "recent_turns": self._render_recent_turns(),
        }
        prompt, snapshot = self._assemble_sections(
            role_id="judge",
            stage="evaluation",
            sections=sections,
            reserved_output_tokens=reserved_output_tokens,
        )
        return (
            f"{prompt}\n\nScore each candidate from 0 to 1 for correctness, clarity, and task fit.\nReturn JSON only.",
            snapshot,
        )

    def build_verifier_prompt(
        self,
        *,
        protocol_name: str,
        evidence_packets: list[dict[str, Any]],
        candidate_summaries: list[dict[str, Any]],
        chunk_index: int,
        chunk_count: int,
        reserved_output_tokens: int,
    ) -> tuple[str, DebateContextSnapshot]:
        state = self.current_state()
        sections = {
            "task": f"Task:\n{self._task_input}",
            "protocol": f"Protocol:\n{protocol_name}",
            "constraints": self._render_constraints(),
            "global_summary": f"Global summary:\n{state.global_summary}",
            "claim_cards": self._render_claim_cards(state.claim_cards),
            "candidate_summaries": self._render_candidate_summaries(candidate_summaries),
            "recent_turns": self._render_recent_turns(),
            "evidence_packets": _render_evidence_packets(evidence_packets),
        }
        prompt, snapshot = self._assemble_sections(
            role_id="verifier",
            stage=f"verification_chunk_{chunk_index + 1}_of_{chunk_count}",
            sections=sections,
            reserved_output_tokens=reserved_output_tokens,
            used_chunking=chunk_count > 1,
        )
        prompt = (
            f"{prompt}\n\nVerification statuses: verified, skipped, failed.\n"
            "Return JSON only with key `candidate_verifications`."
        )
        return prompt, snapshot

    def chunk_evidence_packets(self, evidence_packets: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        if not evidence_packets:
            return []
        if not self._policy.enable_verifier_chunking:
            return [list(evidence_packets)]
        per_chunk_budget = max(400, int(self._policy.max_input_tokens * 0.28))
        chunks: list[list[dict[str, Any]]] = []
        current_chunk: list[dict[str, Any]] = []
        current_tokens = 0
        for packet in evidence_packets:
            text = _render_evidence_packets([packet])
            tokens = estimate_text_tokens(text).tokens
            if current_chunk and current_tokens + tokens > per_chunk_budget:
                chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0
            current_chunk.append(packet)
            current_tokens += tokens
        if current_chunk:
            chunks.append(current_chunk)
        return chunks or [list(evidence_packets)]

    def _build_budgeted_prompt(
        self,
        *,
        role_id: str,
        role_title: str,
        stage: str,
        reserved_output_tokens: int,
    ) -> tuple[str, DebateContextSnapshot]:
        state = self.current_state()
        sections = {
            "task": f"Task:\n{self._task_input}",
            "role": f"Role:\n{role_title}",
            "constraints": self._render_constraints(),
            "global_summary": f"Global summary:\n{state.global_summary}",
            "round_summaries": self._render_round_summaries(state.round_summaries),
            "claim_cards": self._render_claim_cards(state.claim_cards),
            "role_stances": self._render_role_stances(state.role_stance_summaries),
            "open_objections": self._render_list("Open objections", state.open_objections),
            "open_questions": self._render_list("Open questions", state.open_questions),
            "recent_turns": self._render_recent_turns(),
        }
        prompt, snapshot = self._assemble_sections(
            role_id=role_id,
            stage=stage,
            sections=sections,
            reserved_output_tokens=reserved_output_tokens,
        )
        return f"{prompt}\n\nReturn only your answer content.", snapshot

    def _assemble_sections(
        self,
        *,
        role_id: str,
        stage: str,
        sections: dict[str, str],
        reserved_output_tokens: int,
        used_chunking: bool = False,
    ) -> tuple[str, DebateContextSnapshot]:
        rendered_sections = {name: body for name, body in sections.items() if body.strip()}
        final_sections = dict(rendered_sections)
        compaction_level = "under_70"
        truncated = False

        def section_tokens(payload: dict[str, str]) -> int:
            return estimate_text_tokens("\n\n".join(payload.values())).tokens

        prompt_tokens = section_tokens(final_sections)
        max_input_tokens = max(1, self._policy.max_input_tokens)
        usage_ratio = (prompt_tokens + reserved_output_tokens) / max_input_tokens

        if usage_ratio >= 0.70:
            compaction_level = "70_85"
            final_sections["round_summaries"] = self._render_round_summaries(
                self._compact_round_summaries(level="moderate")
            )
            prompt_tokens = section_tokens(final_sections)
            usage_ratio = (prompt_tokens + reserved_output_tokens) / max_input_tokens
            truncated = True

        if usage_ratio >= 0.85:
            compaction_level = "85_95"
            final_sections["round_summaries"] = self._render_round_summaries(
                self._compact_round_summaries(level="aggressive")
            )
            final_sections["recent_turns"] = self._render_recent_turns(max_rounds=1)
            final_sections["role_stances"] = self._render_role_stances(self.current_state().role_stance_summaries)
            prompt_tokens = section_tokens(final_sections)
            usage_ratio = (prompt_tokens + reserved_output_tokens) / max_input_tokens

        overflow = usage_ratio >= 0.95
        if overflow and self._policy.mode == "strict_fail_on_overflow":
            snapshot = DebateContextSnapshot(
                role_id=role_id,
                stage=stage,
                estimated_prompt_tokens=prompt_tokens,
                reserved_output_tokens=reserved_output_tokens,
                max_input_tokens=max_input_tokens,
                usage_ratio=min(usage_ratio, 1.0),
                compaction_level="overflow",
                truncated=truncated,
                used_chunking=used_chunking,
                largest_section=_largest_section_name(final_sections),
                overflow=True,
            )
            self._latest_snapshot = snapshot
            raise DebateContextOverflowError(
                f"Debate context overflow for {role_id} at stage {stage}: usage_ratio={usage_ratio:.3f}"
            )

        snapshot = DebateContextSnapshot(
            role_id=role_id,
            stage=stage,
            estimated_prompt_tokens=prompt_tokens,
            reserved_output_tokens=reserved_output_tokens,
            max_input_tokens=max_input_tokens,
            usage_ratio=min(usage_ratio, 1.0),
            compaction_level="95_plus" if overflow else compaction_level,
            truncated=truncated,
            used_chunking=used_chunking,
            largest_section=_largest_section_name(final_sections),
            overflow=overflow,
        )
        self._latest_snapshot = snapshot
        return "\n\n".join(value for value in final_sections.values() if value.strip()), snapshot

    def _compact_round_summaries(self, *, level: str) -> list[dict[str, Any]]:
        summaries = list(self._round_summaries)
        if not summaries:
            return []
        keep = max(1, self._policy.keep_recent_rounds)
        preserved = summaries[-keep:]
        compacted = summaries[:-keep]
        if not compacted:
            return summaries
        if level == "moderate":
            compacted = [
                {
                    "round_index": item["round_index"],
                    "summary": _truncate(str(item.get("summary") or ""), 140),
                    "roles": list(item.get("roles") or []),
                }
                for item in compacted
            ]
        else:
            compacted = [
                {
                    "round_index": item["round_index"],
                    "summary": _truncate(str(item.get("summary") or ""), 90),
                    "roles": [],
                }
                for item in compacted[-1:]
            ]
        return compacted + preserved

    def _update_round_summary(self, round_index: int) -> None:
        turns = [turn for turn in self._turns if turn.round_index == round_index]
        if not turns:
            return
        summary = " | ".join(f"{turn.role_id}: {_truncate(turn.content, 120)}" for turn in turns)
        payload = {
            "round_index": round_index,
            "summary": _truncate(summary, 240),
            "roles": [turn.role_id for turn in turns],
        }
        for index, item in enumerate(self._round_summaries):
            if int(item.get("round_index") or -1) == round_index:
                self._round_summaries[index] = payload
                return
        self._round_summaries.append(payload)

    def _build_global_summary(self) -> str:
        if not self._round_summaries:
            return _truncate(self._task_input, 240)
        rendered = " | ".join(str(item.get("summary") or "") for item in self._round_summaries[-3:])
        return _truncate(rendered, 320)

    def _extract_claim_cards(self, turn: DebateTurn) -> list[DebateClaimCard]:
        segments = _split_claims(turn.content)
        cards: list[DebateClaimCard] = []
        for index, segment in enumerate(segments[:3], start=1):
            evidence_refs = _extract_evidence_refs(segment)
            target_claim_id = self._find_target_claim_id(turn.role_id)
            cards.append(
                DebateClaimCard(
                    claim_id=f"{turn.role_id}-r{turn.round_index}-c{index}",
                    speaker_role_id=turn.role_id,
                    round_index=turn.round_index,
                    claim=_truncate(segment, 180),
                    rationale=_truncate(segment, 240),
                    evidence_refs=evidence_refs,
                    objection_target_claim_id=target_claim_id,
                    uncertainty=_extract_uncertainty(segment),
                    status="active",
                )
            )
        return cards

    def _find_target_claim_id(self, role_id: str) -> str | None:
        for card in reversed(self._claim_cards):
            if card.speaker_role_id != role_id:
                return card.claim_id
        return None

    def _render_constraints(self) -> str:
        if not self._selected_constraints:
            return ""
        return self._render_list("Constraints", self._selected_constraints)

    def _render_round_summaries(self, items: list[dict[str, Any]]) -> str:
        if not items:
            return ""
        lines = ["Round summaries:"]
        for item in items:
            lines.append(
                f"- round={item.get('round_index')} roles={','.join(item.get('roles') or [])} summary={item.get('summary')}"
            )
        return "\n".join(lines)

    def _render_claim_cards(self, items: list[dict[str, Any]]) -> str:
        if not items:
            return ""
        lines = ["Claim cards:"]
        for item in items:
            evidence = ",".join(item.get("evidence_refs") or []) or "none"
            support_status = str(item.get("support_status") or "needs_evidence")
            lines.append(
                f"- claim_id={item.get('claim_id')} role={item.get('speaker_role_id')} round={item.get('round_index')} evidence={evidence} support={support_status}"
            )
            lines.append(f"  claim={item.get('claim')}")
        return "\n".join(lines)

    def _render_role_stances(self, items: list[dict[str, Any]]) -> str:
        if not items:
            return ""
        lines = ["Role stance summaries:"]
        for item in items:
            lines.append(
                f"- role_id={item.get('role_id')} latest_round={item.get('latest_round_index')} summary={item.get('stance_summary')}"
            )
        return "\n".join(lines)

    def _render_recent_turns(self, *, max_rounds: int | None = None) -> str:
        if not self._turns:
            return ""
        turns = list(self._turns)
        if max_rounds is None:
            max_rounds = self._policy.keep_recent_rounds
        recent_rounds = {turn.round_index for turn in turns[-(max_rounds * 2 + 2) :]}
        filtered = [turn for turn in turns if turn.round_index in recent_rounds]
        lines = ["Recent raw turns:"]
        for turn in filtered[- max(2, max_rounds * 3) :]:
            lines.append(
                f"- round={turn.round_index} role={turn.role_id} candidate_id={turn.candidate_id or turn.role_id}"
            )
            lines.append(f"  {_truncate(turn.content, 220)}")
        return "\n".join(lines)

    def _render_candidate_summaries(self, items: list[dict[str, Any]]) -> str:
        if not items:
            return ""
        lines = ["Candidate summaries:"]
        for item in items:
            lines.append(
                f"- candidate_id={item.get('candidate_id')} role_id={item.get('role_id')} summary={item.get('summary')}"
            )
        return "\n".join(lines)

    @staticmethod
    def _render_list(title: str, items: list[str]) -> str:
        if not items:
            return ""
        return "\n".join([f"{title}:"] + [f"- {item}" for item in items])


def _render_evidence_packets(evidence_packets: list[dict[str, Any]]) -> str:
    lines = ["Evidence packets:"]
    for packet in evidence_packets:
        lines.append(
            f"- evidence_id={packet.get('evidence_id')} title={packet.get('title') or 'Untitled'}"
        )
        lines.append(_truncate(str(packet.get("content") or ""), 500))
    return "\n".join(lines)


def _largest_section_name(sections: dict[str, str]) -> str:
    if not sections:
        return "none"
    return max(sections, key=lambda key: estimate_text_tokens(sections[key]).tokens)


def _split_claims(text: str) -> list[str]:
    parts = [segment.strip(" -") for segment in re.split(r"(?:\n+|(?<=[.!?])\s+)", text) if segment.strip()]
    return parts or [text.strip()]


def _extract_evidence_refs(text: str) -> list[str]:
    refs = re.findall(r"(?:evidence_id=|source=|ref=|citation=)?([A-Za-z0-9_-]{3,})", text)
    return _dedupe([ref for ref in refs if any(char.isdigit() for char in ref)])


def _extract_uncertainty(text: str) -> str:
    lowered = text.lower()
    if any(keyword in lowered for keyword in ("maybe", "unclear", "uncertain", "likely", "possibly")):
        return _truncate(text, 120)
    return ""


def _extract_questions(text: str) -> list[str]:
    return [segment.strip() for segment in re.split(r"(?<=[?])\s+", text) if "?" in segment]


def _extract_objections(turn: DebateTurn, cards: list[DebateClaimCard]) -> list[str]:
    if turn.role_id.endswith("_b") or "challenge" in turn.role_id or "judge" not in turn.role_id:
        return [
            _truncate(card.claim, 160)
            for card in cards
            if card.objection_target_claim_id is not None
        ]
    return []


def _truncate(text: str, max_chars: int) -> str:
    normalized = " ".join(text.strip().split())
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 3].rstrip()}..."


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def summarize_candidate_outputs(
    candidates: list[dict[str, Any]],
    *,
    claim_cards: list[DebateClaimCard],
) -> list[dict[str, Any]]:
    """Build compact candidate summaries for evaluator/verifier prompts."""

    claim_lookup: dict[str, list[str]] = {}
    for card in claim_cards:
        claim_lookup.setdefault(card.speaker_role_id, []).append(card.claim)

    summaries: list[dict[str, Any]] = []
    for candidate in candidates:
        role_id = str(candidate.get("role_id") or "")
        content = str(candidate.get("content") or "")
        claim_bits = claim_lookup.get(role_id, [])
        summary = " | ".join(claim_bits[:3]) if claim_bits else _truncate(content, 220)
        summaries.append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "role_id": role_id,
                "summary": _truncate(summary or content, 220),
            }
        )
    return summaries


def verification_reduce(
    verifications: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reduce chunk-level verification outputs into candidate-level summaries."""

    grouped: dict[str, dict[str, Any]] = {}
    for item in verifications:
        candidate_id = str(item.get("candidate_id") or "").strip()
        if not candidate_id:
            continue
        bucket = grouped.setdefault(
            candidate_id,
            {
                "candidate_id": candidate_id,
                "status": "skipped",
                "rationale_parts": [],
                "citations": [],
                "issues": [],
            },
        )
        status = str(item.get("status") or "skipped")
        if status == "failed":
            bucket["status"] = "failed"
        elif status == "verified" and bucket["status"] != "failed":
            bucket["status"] = "verified"
        rationale = str(item.get("rationale") or "").strip()
        if rationale:
            bucket["rationale_parts"].append(rationale)
        citations = item.get("citations")
        if isinstance(citations, list):
            bucket["citations"].extend(citations)
        issues = item.get("issues")
        if isinstance(issues, list):
            bucket["issues"].extend(str(issue) for issue in issues if str(issue).strip())

    reduced: list[dict[str, Any]] = []
    for bucket in grouped.values():
        reduced.append(
            {
                "candidate_id": bucket["candidate_id"],
                "status": bucket["status"],
                "rationale": _truncate(" | ".join(bucket["rationale_parts"]), 260),
                "citations": bucket["citations"],
                "issues": _dedupe(bucket["issues"]),
            }
        )
    return reduced
