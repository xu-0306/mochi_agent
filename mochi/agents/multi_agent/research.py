"""Research-debate helpers for multi-agent runs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal, Mapping

from mochi.backends.types import Message

ResearchOutputTarget = Literal["research_brief", "dataset_package"]
ResearchSourceMode = Literal["hybrid", "local_only", "web_first"]
ResearchCitationPolicy = Literal["claim_level_required", "strict_fail", "best_effort"]

_QUALITY_ORDER = {"high": 3, "medium": 2, "low": 1}
_WORKER_FOCUSES = (
    "summarize_evidence",
    "counter_evidence",
    "dataset_extraction",
    "source_audit",
)


@dataclass(frozen=True)
class ResearchDebatePolicy:
    """Normalized research-debate policy."""

    enabled: bool = False
    preset: str = "smart_judge_research_debate"
    output_targets: tuple[ResearchOutputTarget, ...] = ("research_brief", "dataset_package")
    source_mode: ResearchSourceMode = "hybrid"
    citation_policy: ResearchCitationPolicy = "claim_level_required"
    local_worker_count: int = 3
    local_worker_count_max: int = 6
    max_research_queries: int = 8
    max_sources_per_query: int = 4
    debate_rounds: int | None = None

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any] | None) -> ResearchDebatePolicy:
        default = cls()
        if not isinstance(metadata, Mapping):
            return default
        evaluation_policy = metadata.get("evaluation_policy")
        if not isinstance(evaluation_policy, Mapping):
            return default
        payload = evaluation_policy.get("research")
        if not isinstance(payload, Mapping):
            return default

        enabled = bool(payload.get("enabled"))
        preset = str(payload.get("preset") or default.preset).strip() or default.preset
        output_targets = _normalize_output_targets(payload.get("output_targets")) or default.output_targets
        source_mode = _normalize_source_mode(payload.get("source_mode")) or default.source_mode
        citation_policy = _normalize_citation_policy(payload.get("citation_policy")) or default.citation_policy
        local_worker_count_max = _bounded_int(
            payload.get("local_worker_count_max"),
            default=default.local_worker_count_max,
            minimum=1,
            maximum=12,
        )
        local_worker_count = _bounded_int(
            payload.get("local_worker_count"),
            default=default.local_worker_count,
            minimum=1,
            maximum=local_worker_count_max,
        )
        max_research_queries = _bounded_int(
            payload.get("max_research_queries"),
            default=default.max_research_queries,
            minimum=1,
            maximum=24,
        )
        max_sources_per_query = _bounded_int(
            payload.get("max_sources_per_query"),
            default=default.max_sources_per_query,
            minimum=1,
            maximum=12,
        )
        raw_rounds = payload.get("debate_rounds")
        debate_rounds = (
            _bounded_int(raw_rounds, default=default.debate_rounds or 2, minimum=1, maximum=8)
            if isinstance(raw_rounds, int)
            else None
        )
        return cls(
            enabled=enabled,
            preset=preset,
            output_targets=output_targets,
            source_mode=source_mode,
            citation_policy=citation_policy,
            local_worker_count=local_worker_count,
            local_worker_count_max=local_worker_count_max,
            max_research_queries=max_research_queries,
            max_sources_per_query=max_sources_per_query,
            debate_rounds=debate_rounds,
        )

    @property
    def evidence_collection_mode(self) -> str:
        if self.source_mode == "local_only":
            return "rag"
        if self.source_mode == "web_first":
            return "web"
        return "hybrid"

    def wants_output(self, target: ResearchOutputTarget) -> bool:
        return target in self.output_targets


def policy_requests_dataset_output(metadata: Mapping[str, Any] | None) -> bool:
    """Whether dataset records should be materialized for this run."""

    policy = ResearchDebatePolicy.from_metadata(metadata)
    if not policy.enabled:
        return True
    return policy.wants_output("dataset_package")


async def build_research_plan(
    *,
    task_input: str,
    guidance_messages: list[str],
    existing_queries: list[str],
    policy: ResearchDebatePolicy,
    planner_model_id: str | None,
    generate: Callable[..., Awaitable[Any]] | None,
) -> dict[str, Any]:
    """Create a normalized research plan."""

    fallback = _fallback_research_plan(
        task_input=task_input,
        guidance_messages=guidance_messages,
        existing_queries=existing_queries,
        max_queries=policy.max_research_queries,
    )
    if not planner_model_id or not callable(generate):
        return fallback

    prompt = "\n".join(
        [
            f"Task:\n{task_input.strip()}",
            "",
            "Guidance:",
            *[f"- {item}" for item in guidance_messages if item.strip()],
            "",
            "Existing evidence queries:",
            *[f"- {item}" for item in existing_queries],
            "",
            "Return JSON only with keys `subquestions`, `evidence_requirements`, `exclusion_rules`, and `evidence_queries`.",
            f"Generate at most {policy.max_research_queries} evidence queries.",
        ]
    ).strip()
    result = await generate(
        model_id=planner_model_id,
        messages=[
            Message(
                role="system",
                content=(
                    "You plan evidence-grounded research debate runs. "
                    "Return compact JSON only."
                ),
            ),
            Message(role="user", content=prompt),
        ],
        temperature=0.1,
        max_tokens=900,
        reasoning_effort=None,
    )
    parsed = _extract_json_payload(getattr(result, "content", ""))
    if parsed is None:
        return fallback

    return {
        "status": "model_generated",
        "planner_model_id": planner_model_id,
        "subquestions": _normalize_string_list(parsed.get("subquestions")) or fallback["subquestions"],
        "evidence_requirements": _normalize_string_list(parsed.get("evidence_requirements"))
        or fallback["evidence_requirements"],
        "exclusion_rules": _normalize_string_list(parsed.get("exclusion_rules")) or fallback["exclusion_rules"],
        "evidence_queries": _truncate_list(
            _dedupe(_normalize_string_list(parsed.get("evidence_queries")) + existing_queries),
            policy.max_research_queries,
        )
        or fallback["evidence_queries"],
    }


async def run_research_workers(
    *,
    task_input: str,
    research_plan: Mapping[str, Any],
    evidence_packets: list[dict[str, Any]],
    policy: ResearchDebatePolicy,
    local_worker_model_id: str | None,
    generate: Callable[..., Awaitable[Any]] | None,
) -> dict[str, Any]:
    """Run lightweight local-worker fan-out for research support."""

    notes: list[dict[str, Any]] = []
    evidence_preview = _render_evidence_preview(evidence_packets)
    for index in range(policy.local_worker_count):
        focus = _WORKER_FOCUSES[index % len(_WORKER_FOCUSES)]
        worker_id = f"local-worker-{index + 1}"
        if local_worker_model_id and callable(generate):
            prompt = "\n".join(
                [
                    f"Task:\n{task_input.strip()}",
                    "",
                    f"Worker focus: {focus}",
                    "",
                    "Research plan:",
                    json.dumps(dict(research_plan), ensure_ascii=False, indent=2),
                    "",
                    "Evidence preview:",
                    evidence_preview,
                    "",
                    "Return a compact plain-text note with findings, risks, and next action.",
                ]
            ).strip()
            result = await generate(
                model_id=local_worker_model_id,
                messages=[
                    Message(
                        role="system",
                        content=(
                            "You are a small local research worker. "
                            "Summarize only evidence-grounded observations."
                        ),
                    ),
                    Message(role="user", content=prompt),
                ],
                temperature=0.2,
                max_tokens=500,
                reasoning_effort=None,
            )
            content = str(getattr(result, "content", "") or "").strip()
        else:
            content = _fallback_worker_note(
                focus=focus,
                evidence_packets=evidence_packets,
                research_plan=research_plan,
            )
        notes.append(
            {
                "worker_id": worker_id,
                "focus": focus,
                "model_id": local_worker_model_id,
                "content": content or "No worker note generated.",
            }
        )

    return {
        "worker_count": len(notes),
        "model_id": local_worker_model_id,
        "notes": notes,
        "focuses": [item["focus"] for item in notes],
    }


def build_source_quality_table(
    evidence_packets: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score sources into coarse quality tiers."""

    sources: list[dict[str, Any]] = []
    tier_counts = {"high": 0, "medium": 0, "low": 0}
    for packet in evidence_packets:
        tier, rationale = _quality_tier(packet)
        tier_counts[tier] += 1
        sources.append(
            {
                "evidence_id": packet.get("evidence_id"),
                "title": packet.get("title"),
                "url": packet.get("url"),
                "source_type": packet.get("source_type"),
                "provider": packet.get("provider"),
                "query": packet.get("query"),
                "source_quality_tier": tier,
                "quality_rationale": rationale,
                "content_preview": _truncate(str(packet.get("content") or ""), 220),
            }
        )
    return {
        "source_count": len(sources),
        "tier_counts": tier_counts,
        "sources": sources,
    }


def build_claim_evidence_map(
    *,
    debate_state: Mapping[str, Any] | None,
    verification_summary: Mapping[str, Any] | None,
    source_quality_table: Mapping[str, Any] | None,
    citation_policy: ResearchCitationPolicy,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Map debate claims to verification/citation state."""

    if not isinstance(debate_state, Mapping):
        return {"citation_policy": citation_policy, "claims": [], "counts": {}}, None

    source_index = {
        str(item.get("evidence_id")): item
        for item in _record_list(source_quality_table.get("sources") if isinstance(source_quality_table, Mapping) else [])
        if isinstance(item.get("evidence_id"), str)
    }
    verification_by_candidate = {
        str(item.get("candidate_id")): item
        for item in _record_list(
            verification_summary.get("verifications") if isinstance(verification_summary, Mapping) else []
        )
        if isinstance(item.get("candidate_id"), str)
    }

    enriched_claims: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for item in _record_list(debate_state.get("claim_cards")):
        speaker_role_id = str(item.get("speaker_role_id") or "")
        verification = verification_by_candidate.get(speaker_role_id, {})
        citation_refs = _dedupe(
            _normalize_string_list(item.get("evidence_refs"))
            + [
                str(citation.get("evidence_id"))
                for citation in _record_list(verification.get("citations"))
                if isinstance(citation.get("evidence_id"), str)
            ]
        )
        support_status = _claim_support_status(
            verification_status=str(verification.get("status") or ""),
            citation_refs=citation_refs,
            citation_policy=citation_policy,
        )
        counts[support_status] = counts.get(support_status, 0) + 1
        source_tier = _best_quality_tier(citation_refs, source_index)
        confidence = _support_confidence(support_status=support_status, source_tier=source_tier)
        enriched_claims.append(
            {
                **dict(item),
                "support_status": support_status,
                "confidence": confidence,
                "citation_refs": citation_refs,
                "source_quality_tier": source_tier,
            }
        )

    updated_state = dict(debate_state)
    updated_state["claim_cards"] = enriched_claims
    return {
        "citation_policy": citation_policy,
        "claims": enriched_claims,
        "counts": counts,
    }, updated_state


async def synthesize_research_brief(
    *,
    task_input: str,
    selected_candidate: Mapping[str, Any] | None,
    research_plan: Mapping[str, Any] | None,
    source_quality_table: Mapping[str, Any] | None,
    claim_evidence_map: Mapping[str, Any] | None,
    worker_outputs: Mapping[str, Any] | None,
    synthesizer_model_id: str | None,
    generate: Callable[..., Awaitable[Any]] | None,
) -> dict[str, Any]:
    """Create the final research brief artifact."""

    fallback = _fallback_research_brief(
        task_input=task_input,
        selected_candidate=selected_candidate,
        research_plan=research_plan,
        source_quality_table=source_quality_table,
        claim_evidence_map=claim_evidence_map,
        worker_outputs=worker_outputs,
    )
    if not synthesizer_model_id or not callable(generate):
        return fallback

    prompt = "\n".join(
        [
            f"Task:\n{task_input.strip()}",
            "",
            "Selected candidate:",
            json.dumps(dict(selected_candidate or {}), ensure_ascii=False, indent=2),
            "",
            "Research plan:",
            json.dumps(dict(research_plan or {}), ensure_ascii=False, indent=2),
            "",
            "Source quality table:",
            json.dumps(dict(source_quality_table or {}), ensure_ascii=False, indent=2),
            "",
            "Claim evidence map:",
            json.dumps(dict(claim_evidence_map or {}), ensure_ascii=False, indent=2),
            "",
            "Worker outputs:",
            json.dumps(dict(worker_outputs or {}), ensure_ascii=False, indent=2),
            "",
            "Return Markdown only with sections: Summary, Findings, Evidence Quality, Claim Status, Open Gaps.",
        ]
    ).strip()
    result = await generate(
        model_id=synthesizer_model_id,
        messages=[
            Message(
                role="system",
                content=(
                    "You synthesize evidence-grounded research briefs. "
                    "Keep claims cautious and mention unresolved gaps explicitly."
                ),
            ),
            Message(role="user", content=prompt),
        ],
        temperature=0.15,
        max_tokens=1200,
        reasoning_effort=None,
    )
    markdown = str(getattr(result, "content", "") or "").strip()
    if not markdown:
        return fallback
    return {
        **fallback,
        "status": "model_generated",
        "synthesizer_model_id": synthesizer_model_id,
        "markdown": markdown,
    }


def build_research_prompt_appendix(
    *,
    research_plan: Mapping[str, Any] | None,
    source_quality_table: Mapping[str, Any] | None,
    worker_outputs: Mapping[str, Any] | None,
) -> str:
    """Build research context appended to debate prompts."""

    lines: list[str] = []
    if isinstance(research_plan, Mapping):
        subquestions = _normalize_string_list(research_plan.get("subquestions"))
        evidence_requirements = _normalize_string_list(research_plan.get("evidence_requirements"))
        if subquestions:
            lines.append("Research subquestions:")
            lines.extend(f"- {item}" for item in subquestions[:5])
        if evidence_requirements:
            lines.append("Evidence requirements:")
            lines.extend(f"- {item}" for item in evidence_requirements[:4])
    if isinstance(source_quality_table, Mapping):
        top_sources = _record_list(source_quality_table.get("sources"))[:4]
        if top_sources:
            lines.append("Top source signals:")
            for item in top_sources:
                lines.append(
                    f"- {item.get('evidence_id')} tier={item.get('source_quality_tier')} title={item.get('title') or 'Untitled'}"
                )
    if isinstance(worker_outputs, Mapping):
        notes = _record_list(worker_outputs.get("notes"))
        if notes:
            lines.append("Local worker notes:")
            for item in notes[:4]:
                lines.append(f"- {item.get('focus')}: {_truncate(str(item.get('content') or ''), 180)}")
    return "\n".join(lines).strip()


def merge_research_queries(
    * ,
    existing_queries: list[str],
    research_plan: Mapping[str, Any] | None,
    policy: ResearchDebatePolicy,
) -> list[str]:
    """Merge user-provided and plan-generated evidence queries."""

    plan_queries = (
        _normalize_string_list(research_plan.get("evidence_queries"))
        if isinstance(research_plan, Mapping)
        else []
    )
    return _truncate_list(_dedupe(existing_queries + plan_queries), policy.max_research_queries)


def _fallback_research_plan(
    *,
    task_input: str,
    guidance_messages: list[str],
    existing_queries: list[str],
    max_queries: int,
) -> dict[str, Any]:
    task = task_input.strip()
    keywords = [item for item in re.split(r"[^A-Za-z0-9_]+", task) if len(item) >= 4]
    query_candidates = existing_queries + [
        task,
        f"{task} primary source",
        f"{task} contradictory evidence",
    ]
    if keywords:
        query_candidates.append(" ".join(keywords[:5]))
    return {
        "status": "deterministic",
        "planner_model_id": None,
        "subquestions": [
            f"What specific conclusion is needed for: {task}?",
            "Which sources most directly support or contradict the current answer?",
            "What uncertainties, boundary conditions, or exclusions remain unresolved?",
        ],
        "evidence_requirements": [
            "Prefer primary or directly attributable sources when possible.",
            "Capture at least one supporting and one challenging signal.",
            "Flag time-sensitive facts that still need explicit evidence.",
        ]
        + [item for item in guidance_messages if item.strip()][:2],
        "exclusion_rules": [
            "Do not treat unsupported speculation as established fact.",
            "Mark unresolved factual statements as needs_evidence.",
            "Separate evidence-backed claims from proposed next steps.",
        ],
        "evidence_queries": _truncate_list(_dedupe(query_candidates), max_queries),
    }


def _fallback_worker_note(
    *,
    focus: str,
    evidence_packets: list[dict[str, Any]],
    research_plan: Mapping[str, Any],
) -> str:
    first_source = evidence_packets[0].get("title") if evidence_packets else "no collected source"
    subquestions = _normalize_string_list(research_plan.get("subquestions"))[:2]
    if focus == "counter_evidence":
        return (
            f"Check for counter-evidence against the leading answer. "
            f"Use {first_source} and any conflicting source before accepting a conclusion."
        )
    if focus == "dataset_extraction":
        return "Extract training-worthy records only from claims that remain evidence-backed after verification."
    if focus == "source_audit":
        return f"Audit provenance and quality tiering. Prioritize attributable sources over {first_source}."
    return " | ".join(subquestions) if subquestions else "Summarize the strongest evidence and note any gaps."


def _fallback_research_brief(
    *,
    task_input: str,
    selected_candidate: Mapping[str, Any] | None,
    research_plan: Mapping[str, Any] | None,
    source_quality_table: Mapping[str, Any] | None,
    claim_evidence_map: Mapping[str, Any] | None,
    worker_outputs: Mapping[str, Any] | None,
) -> dict[str, Any]:
    selected_answer = str((selected_candidate or {}).get("content") or "").strip()
    tier_counts = (
        source_quality_table.get("tier_counts")
        if isinstance(source_quality_table, Mapping)
        else {}
    )
    claim_counts = (
        claim_evidence_map.get("counts")
        if isinstance(claim_evidence_map, Mapping)
        else {}
    )
    markdown = "\n".join(
        [
            "# Research Brief",
            "",
            "## Summary",
            task_input.strip(),
            "",
            "## Findings",
            selected_answer or "No final candidate answer was selected.",
            "",
            "## Evidence Quality",
            f"- high: {int((tier_counts or {}).get('high', 0))}",
            f"- medium: {int((tier_counts or {}).get('medium', 0))}",
            f"- low: {int((tier_counts or {}).get('low', 0))}",
            "",
            "## Claim Status",
            *(f"- {name}: {count}" for name, count in sorted((claim_counts or {}).items())),
            "",
            "## Open Gaps",
            *(
                f"- {item}"
                for item in _normalize_string_list((research_plan or {}).get("exclusion_rules"))[:3]
            ),
        ]
    ).strip()
    return {
        "status": "deterministic",
        "synthesizer_model_id": None,
        "markdown": markdown,
        "selected_candidate_id": (selected_candidate or {}).get("candidate_id"),
        "selected_candidate_summary": _truncate(selected_answer, 280),
        "source_quality_table": dict(source_quality_table or {}),
        "claim_evidence_map": dict(claim_evidence_map or {}),
        "worker_notes": _record_list((worker_outputs or {}).get("notes"))[:4],
    }


def _extract_json_payload(raw_content: Any) -> dict[str, Any] | None:
    text = str(raw_content or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _normalize_output_targets(value: Any) -> tuple[ResearchOutputTarget, ...]:
    if not isinstance(value, list):
        return ()
    normalized: list[ResearchOutputTarget] = []
    for item in value:
        if item in {"research_brief", "dataset_package"} and item not in normalized:
            normalized.append(item)
    return tuple(normalized)


def _normalize_source_mode(value: Any) -> ResearchSourceMode | None:
    if value in {"hybrid", "local_only", "web_first"}:
        return value
    return None


def _normalize_citation_policy(value: Any) -> ResearchCitationPolicy | None:
    if value in {"claim_level_required", "strict_fail", "best_effort"}:
        return value
    return None


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def _quality_tier(packet: Mapping[str, Any]) -> tuple[str, str]:
    source_type = str(packet.get("source_type") or "").lower()
    url = str(packet.get("url") or "").lower()
    provider = str(packet.get("provider") or "").lower()
    if source_type in {"pubmed", "crossref", "semantic_scholar", "arxiv"}:
        return "high", f"Structured scholarly source: {source_type}"
    if url.endswith(".gov") or ".gov/" in url or url.endswith(".edu") or ".edu/" in url:
        return "high", "Government or academic domain"
    if source_type in {"web_fetch", "web_search", "mcp_resource", "memory_search"} or provider:
        return "medium", "Attributable fetched or indexed source"
    return "low", "Inline or weakly attributed source"


def _claim_support_status(
    *,
    verification_status: str,
    citation_refs: list[str],
    citation_policy: ResearchCitationPolicy,
) -> str:
    if verification_status == "verified":
        return "supported" if citation_refs else ("needs_evidence" if citation_policy != "best_effort" else "contested")
    if verification_status == "failed":
        return "refuted"
    if citation_refs:
        return "contested"
    return "needs_evidence"


def _support_confidence(*, support_status: str, source_tier: str) -> float:
    base = {
        "supported": 0.82,
        "contested": 0.55,
        "refuted": 0.24,
        "needs_evidence": 0.18,
        "out_of_scope": 0.1,
    }.get(support_status, 0.2)
    if source_tier == "high":
        base += 0.1
    elif source_tier == "low":
        base -= 0.08
    return round(max(0.0, min(base, 0.99)), 2)


def _best_quality_tier(
    evidence_ids: list[str],
    source_index: Mapping[str, Mapping[str, Any]],
) -> str:
    best = "low"
    for evidence_id in evidence_ids:
        item = source_index.get(evidence_id)
        if not isinstance(item, Mapping):
            continue
        tier = str(item.get("source_quality_tier") or "low")
        if _QUALITY_ORDER.get(tier, 0) > _QUALITY_ORDER.get(best, 0):
            best = tier
    return best


def _render_evidence_preview(evidence_packets: list[dict[str, Any]]) -> str:
    if not evidence_packets:
        return "No evidence packets collected."
    lines: list[str] = []
    for item in evidence_packets[:4]:
        lines.append(
            f"- {item.get('evidence_id')}: {item.get('title') or 'Untitled'} | {_truncate(str(item.get('content') or ''), 180)}"
        )
    return "\n".join(lines)


def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _record_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _truncate_list(items: list[str], limit: int) -> list[str]:
    return items[: max(0, limit)]


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
