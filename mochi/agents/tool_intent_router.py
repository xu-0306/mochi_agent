"""Bounded tool-intent routing for main-chat tool exposure."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

from mochi.backends.base import BaseLLMBackend
from mochi.backends.types import GenerationResult, Message

ToolIntent = Literal[
    "open_world_lookup",
    "literature_research",
    "workspace_read",
    "workspace_write",
    "execution_or_process",
    "tool_discovery",
    "ambiguous",
]
ToolIntentRouteSource = Literal["classifier", "fallback_keyword"]

_CANONICAL_INTENTS: set[str] = {
    "open_world_lookup",
    "literature_research",
    "workspace_read",
    "workspace_write",
    "execution_or_process",
    "tool_discovery",
    "ambiguous",
}
_LEGACY_INTENT_ALIASES: dict[str, ToolIntent] = {
    "workspace_inspection": "workspace_read",
    "workspace_mutation": "workspace_write",
}
_TOOL_INTENT_CLASSIFIER_SYSTEM_PROMPT = """
You are Mochi's internal tool-intent classifier for main-chat tool exposure.

Classify only the user's latest request into exactly one intent from this taxonomy:
- open_world_lookup
- literature_research
- workspace_read
- workspace_write
- execution_or_process
- tool_discovery
- ambiguous

Rules:
- Use workspace_* intents only when the user is clearly talking about the local repo, workspace files, codebase, paths, or attached workspace files.
- Weather, news, current facts, or web lookup requests in any language are open_world_lookup even when the session is workspace-bound.
- Paper, citation, DOI, literature, PubMed, arXiv, or scholarly research requests are literature_research.
- Reading, inspecting, summarizing, or searching workspace files is workspace_read.
- Editing, rewriting, patching, creating, or changing workspace files is workspace_write.
- Running tests, commands, scripts, builds, or process/session control is execution_or_process.
- Asking which tools are available or which tool to use is tool_discovery.
- Be conservative. Prefer ambiguous over a false positive workspace route.
- Session-bound state and attachment counts are context only. They do not automatically imply a workspace intent.
- Ignore normal language-matching behavior. This is an internal classifier.
- Return strict JSON only using this exact schema:
  {"intent":"open_world_lookup|literature_research|workspace_read|workspace_write|execution_or_process|tool_discovery|ambiguous","confidence":0.0,"rationale":"..."}
""".strip()
_DEFAULT_LOW_CONFIDENCE_THRESHOLD = 0.68
_DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[-._;()/:a-z0-9]+\b", re.IGNORECASE)
_WORKSPACE_FILE_REFERENCE_PATTERN = re.compile(
    r"\b[\w.-]+\.(?:"
    r"py|pyi|ipynb|md|rst|txt|json|ya?ml|toml|ini|cfg|csv|ts|tsx|js|jsx|mjs|cjs|"
    r"java|kt|go|rs|rb|php|swift|c|cc|cpp|h|hpp|cs|sql|sh|ps1|html|css|xml|pdf|docx"
    r")\b",
    re.IGNORECASE,
)
_WORKSPACE_PATH_REFERENCE_PATTERN = re.compile(
    r"(?:(?<=\s)|^)(?:\.{1,2}[\\/]|[a-z]:[\\/]|~[\\/]|/)[^\s]+",
    re.IGNORECASE,
)
_TOOL_DISCOVERY_KEYWORDS: tuple[str, ...] = (
    "which tool",
    "what tool",
    "available tools",
    "list tools",
    "find tool",
    "tool should i use",
    "tool to use",
    "what tools can",
    "what tools are available",
    "\u54ea\u500b\u5de5\u5177",
    "\u54ea\u4e2a\u5de5\u5177",
    "\u4ec0\u9ebc\u5de5\u5177",
    "\u4ec0\u4e48\u5de5\u5177",
    "\u54ea\u4e9b\u5de5\u5177",
    "\u53ef\u7528\u5de5\u5177",
    "\u5217\u51fa\u5de5\u5177",
)
_WORKSPACE_REFERENCE_KEYWORDS: tuple[str, ...] = (
    "repo",
    "repository",
    "workspace",
    "codebase",
    "local file",
    "local files",
    "workspace file",
    "workspace files",
    "\u5009\u5eab",
    "\u4ed3\u5e93",
    "\u5de5\u4f5c\u5340",
    "\u5de5\u4f5c\u533a",
    "\u5de5\u4f5c\u5340\u6a94\u6848",
    "\u5de5\u4f5c\u533a\u6587\u4ef6",
)
_WORKSPACE_TARGET_KEYWORDS: tuple[str, ...] = (
    "file",
    "files",
    "folder",
    "folders",
    "directory",
    "directories",
    "path",
    "paths",
    "class",
    "function",
    "method",
    "symbol",
    "todo",
    "pdf",
    "csv",
    "docx",
    "notebook",
    "ipynb",
    "\u6a94\u6848",
    "\u6587\u4ef6",
    "\u8cc7\u6599\u593e",
    "\u6587\u4ef6\u5939",
    "\u76ee\u9304",
    "\u76ee\u5f55",
    "\u8def\u5f91",
    "\u985e\u5225",
    "\u7c7b",
    "\u51fd\u5f0f",
    "\u51fd\u6570",
    "\u7b26\u865f",
    "\u7b26\u53f7",
)
_WORKSPACE_OBJECT_KEYWORDS: tuple[str, ...] = (
    "repo",
    "repository",
    "project",
    "workspace",
    "codebase",
    "code",
    "source",
    "file",
    "files",
    "folder",
    "folders",
    "directory",
    "directories",
    "path",
    "paths",
    "local file",
    "local files",
    "class",
    "function",
    "method",
    "symbol",
    "todo",
    "pdf",
    "csv",
    "docx",
    "notebook",
    "ipynb",
    "\u5009\u5eab",
    "\u4ed3\u5e93",
    "\u5c08\u6848",
    "\u9879\u76ee",
    "\u5de5\u4f5c\u5340",
    "\u5de5\u4f5c\u533a",
    "\u7a0b\u5f0f\u78bc",
    "\u7a0b\u5e8f\u4ee3\u7801",
    "\u4ee3\u78bc",
    "\u4ee3\u7801",
    "\u6a94\u6848",
    "\u6587\u4ef6",
    "\u8cc7\u6599\u593e",
    "\u6587\u4ef6\u5939",
    "\u76ee\u9304",
    "\u76ee\u5f55",
    "\u8def\u5f91",
    "\u985e\u5225",
    "\u7c7b",
    "\u51fd\u5f0f",
    "\u51fd\u6570",
    "\u7b26\u865f",
    "\u7b26\u53f7",
)
_WORKSPACE_READ_KEYWORDS: tuple[str, ...] = (
    "read",
    "inspect",
    "review",
    "summarize",
    "summary",
    "analyze",
    "analyse",
    "search",
    "find",
    "grep",
    "glob",
    "list",
    "browse",
    "open",
    "definition",
    "structure",
    "overview",
    "\u95b1\u8b80",
    "\u9605\u8bfb",
    "\u6aa2\u67e5",
    "\u68c0\u67e5",
    "\u6aa2\u8996",
    "\u68c0\u89c6",
    "\u6458\u8981",
    "\u5206\u6790",
    "\u5c0b\u627e",
    "\u67e5\u627e",
    "\u641c\u5c0b",
    "\u641c\u7d22",
    "\u5217\u51fa",
    "\u700f\u89bd",
    "\u6d4f\u89c8",
    "\u6253\u958b",
    "\u6253\u5f00",
    "\u5b9a\u7fa9",
    "\u5b9a\u4e49",
    "\u7d50\u69cb",
    "\u7ed3\u6784",
    "\u6982\u89bd",
    "\u6982\u89c8",
)
_WORKSPACE_WRITE_KEYWORDS: tuple[str, ...] = (
    "edit",
    "update",
    "modify",
    "change",
    "rewrite",
    "revise",
    "patch",
    "fix",
    "refactor",
    "rename",
    "move",
    "delete",
    "create",
    "write",
    "save",
    "\u7de8\u8f2f",
    "\u7f16\u8f91",
    "\u66f4\u65b0",
    "\u4fee\u6539",
    "\u8b8a\u66f4",
    "\u53d8\u66f4",
    "\u6539\u5beb",
    "\u6539\u5199",
    "\u4fee\u88dc",
    "\u4fee\u590d",
    "\u4fee\u5fa9",
    "\u91cd\u69cb",
    "\u91cd\u6784",
    "\u91cd\u547d\u540d",
    "\u79fb\u52d5",
    "\u79fb\u52a8",
    "\u522a\u9664",
    "\u5efa\u7acb",
    "\u521b\u5efa",
    "\u5beb\u5165",
    "\u5199\u5165",
    "\u5132\u5b58",
    "\u4fdd\u5b58",
)
_EXECUTION_OR_PROCESS_KEYWORDS: tuple[str, ...] = (
    "run",
    "test",
    "debug",
    "execute",
    "command",
    "script",
    "build",
    "compile",
    "install",
    "launch",
    "start",
    "stop",
    "server",
    "process",
    "session",
    "stdin",
    "tty",
    "background",
    "benchmark",
    "\u57f7\u884c",
    "\u8fd0\u884c",
    "\u6e2c\u8a66",
    "\u6d4b\u8bd5",
    "\u9664\u932f",
    "\u8c03\u8bd5",
    "\u547d\u4ee4",
    "\u8173\u672c",
    "\u5efa\u7f6e",
    "\u7f16\u8bd1",
    "\u5b89\u88dd",
    "\u555f\u52d5",
    "\u542f\u52a8",
    "\u505c\u6b62",
    "\u670d\u52d9\u5668",
    "\u670d\u52a1\u5668",
    "\u884c\u7a0b",
    "\u8fdb\u7a0b",
    "\u6703\u8a71",
    "\u4f1a\u8bdd",
    "\u5f8c\u53f0",
    "\u540e\u53f0",
    "\u80cc\u666f",
)
_RESEARCH_KEYWORDS: tuple[str, ...] = (
    "paper",
    "papers",
    "research",
    "literature",
    "academic",
    "scholarly",
    "arxiv",
    "pubmed",
    "semantic scholar",
    "crossref",
    "citation",
    "citations",
    "cite",
    "doi",
    "journal",
    "abstract",
    "survey",
    "reference",
    "references",
    "preprint",
    "\u8ad6\u6587",
    "\u8bba\u6587",
    "\u7814\u7a76",
    "\u6587\u737b",
    "\u6587\u732e",
    "\u5b78\u8853",
    "\u5b66\u672f",
    "\u5f15\u7528",
    "\u671f\u520a",
    "\u6458\u8981",
    "\u53c3\u8003\u6587\u737b",
    "\u53c2\u8003\u6587\u732e",
)
_WEATHER_KEYWORDS: tuple[str, ...] = (
    "weather",
    "forecast",
    "temperature",
    "rain",
    "\u5929\u6c23",
    "\u5929\u6c14",
    "\u6c23\u8c61",
    "\u6c14\u8c61",
    "\u9810\u5831",
    "\u9884\u62a5",
    "\u6eab\u5ea6",
    "\u6e29\u5ea6",
    "\u4e0b\u96e8",
)
_OPEN_WORLD_LOOKUP_KEYWORDS: tuple[str, ...] = (
    "latest",
    "today",
    "tomorrow",
    "yesterday",
    "news",
    "current",
    "price",
    "stock",
    "http",
    "https",
    "url",
    "website",
    "web",
    "internet",
    "look up",
    "lookup",
    "search the web",
    "\u6700\u65b0",
    "\u4eca\u5929",
    "\u660e\u5929",
    "\u6628\u5929",
    "\u65b0\u805e",
    "\u65b0\u95fb",
    "\u73fe\u5728",
    "\u73b0\u5728",
    "\u76ee\u524d",
    "\u50f9\u683c",
    "\u4ef7\u683c",
    "\u7db2\u7ad9",
    "\u7f51\u7ad9",
    "\u7db2\u9801",
    "\u7f51\u9875",
    "\u7db2\u8def",
    "\u7f51\u7edc",
    "\u67e5\u8a62",
    "\u67e5\u8be2",
    "\u641c\u5c0b",
    "\u641c\u7d22",
)


@dataclass(frozen=True)
class ToolIntentRoute:
    intent: ToolIntent
    confidence: float | None
    source: ToolIntentRouteSource
    rationale: str

    def to_metadata(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "confidence": self.confidence,
            "source": self.source,
            "rationale": self.rationale,
        }


class ToolIntentClassifier(Protocol):
    async def classify(
        self,
        *,
        user_message: str,
        session_bound_workspace: bool,
        attachment_count: int,
        workspace_attachment_count: int,
    ) -> ToolIntentRoute:
        """Return a classifier-produced tool-intent route."""


class BackendToolIntentClassifier:
    """Direct backend classifier used only for bounded tool-intent routing."""

    def __init__(self, backend: BaseLLMBackend) -> None:
        self._backend = backend

    async def classify(
        self,
        *,
        user_message: str,
        session_bound_workspace: bool,
        attachment_count: int,
        workspace_attachment_count: int,
    ) -> ToolIntentRoute:
        payload = {
            "user_message": user_message,
            "session_bound_workspace": session_bound_workspace,
            "attachment_count": max(0, attachment_count),
            "workspace_attachment_count": max(0, workspace_attachment_count),
        }
        result = await self._backend.generate(
            [
                Message(role="system", content=_TOOL_INTENT_CLASSIFIER_SYSTEM_PROMPT),
                Message(role="user", content=json.dumps(payload, ensure_ascii=False, indent=2)),
            ],
            tools=None,
            temperature=0.0,
            max_tokens=180,
            top_p=1.0,
            stream=False,
        )
        if not isinstance(result, GenerationResult):
            raise RuntimeError("Tool intent classifier expected a non-stream backend response.")
        return parse_tool_intent_classifier_result(result.content)


class ToolIntentRouter:
    """Classifier-first tool intent routing with keyword fallback."""

    def __init__(self, *, low_confidence_threshold: float = _DEFAULT_LOW_CONFIDENCE_THRESHOLD) -> None:
        self._low_confidence_threshold = low_confidence_threshold

    async def route(
        self,
        *,
        user_message: str,
        session_bound_workspace: bool,
        attachment_count: int = 0,
        workspace_attachment_count: int = 0,
        classifier: ToolIntentClassifier | None = None,
    ) -> ToolIntentRoute:
        classifier_reason: str | None = None
        if classifier is not None:
            try:
                classified = await classifier.classify(
                    user_message=user_message,
                    session_bound_workspace=session_bound_workspace,
                    attachment_count=attachment_count,
                    workspace_attachment_count=workspace_attachment_count,
                )
            except Exception as exc:
                classifier_reason = f"Classifier failed: {exc}"
            else:
                if self._should_accept_classifier_route(classified):
                    return classified
                classifier_reason = self._classifier_fallback_reason(classified)

        fallback = self._fallback_route(
            user_message=user_message,
            session_bound_workspace=session_bound_workspace,
            attachment_count=attachment_count,
            workspace_attachment_count=workspace_attachment_count,
        )
        if classifier_reason:
            return ToolIntentRoute(
                intent=fallback.intent,
                confidence=fallback.confidence,
                source=fallback.source,
                rationale=f"{fallback.rationale} Fallback reason: {classifier_reason}",
            )
        return fallback

    def _should_accept_classifier_route(self, route: ToolIntentRoute) -> bool:
        if route.intent == "ambiguous":
            return False
        if route.confidence is None:
            return False
        return route.confidence >= self._low_confidence_threshold

    def _classifier_fallback_reason(self, route: ToolIntentRoute) -> str:
        if route.intent == "ambiguous":
            return f"Classifier returned ambiguous. Original rationale: {route.rationale}"
        return (
            "Classifier confidence was below threshold. "
            f"confidence={route.confidence!r}, threshold={self._low_confidence_threshold:.2f}. "
            f"Original rationale: {route.rationale}"
        )

    def _fallback_route(
        self,
        *,
        user_message: str,
        session_bound_workspace: bool,
        attachment_count: int,
        workspace_attachment_count: int,
    ) -> ToolIntentRoute:
        lowered = user_message.casefold().strip()
        if not lowered:
            return ToolIntentRoute(
                intent="ambiguous",
                confidence=0.0,
                source="fallback_keyword",
                rationale="No user message was provided for routing.",
            )

        tool_discovery_hits = self._matched_keywords(lowered, _TOOL_DISCOVERY_KEYWORDS)
        workspace_reference_hits = self._workspace_reference_hits(lowered)
        workspace_target_hits = self._matched_keywords(lowered, _WORKSPACE_TARGET_KEYWORDS)
        workspace_read_hits = self._matched_keywords(lowered, _WORKSPACE_READ_KEYWORDS)
        workspace_write_hits = self._matched_keywords(lowered, _WORKSPACE_WRITE_KEYWORDS)
        execution_hits = self._matched_keywords(lowered, _EXECUTION_OR_PROCESS_KEYWORDS)
        research_hits = self._matched_keywords(lowered, _RESEARCH_KEYWORDS)
        weather_hits = self._matched_keywords(lowered, _WEATHER_KEYWORDS)
        open_world_hits = self._matched_keywords(lowered, _OPEN_WORLD_LOOKUP_KEYWORDS)
        has_doi = _DOI_PATTERN.search(lowered) is not None
        has_explicit_workspace_local_evidence = bool(
            workspace_reference_hits or workspace_attachment_count > 0
        )

        if tool_discovery_hits:
            return self._route_from_hits(
                intent="tool_discovery",
                confidence=0.9,
                hits=tool_discovery_hits,
                prefix="Matched tool-discovery language",
            )

        if workspace_write_hits and has_explicit_workspace_local_evidence:
            return self._route_from_hits(
                intent="workspace_write",
                confidence=0.86,
                hits=workspace_write_hits + workspace_reference_hits,
                prefix="Matched workspace write language with explicit local evidence",
            )

        if execution_hits:
            return self._route_from_hits(
                intent="execution_or_process",
                confidence=0.84,
                hits=execution_hits,
                prefix="Matched execution/process language",
            )

        if research_hits or has_doi:
            hits = research_hits or (["doi"] if has_doi else [])
            return self._route_from_hits(
                intent="literature_research",
                confidence=0.88,
                hits=hits,
                prefix="Matched literature research language",
            )

        if weather_hits:
            return self._route_from_hits(
                intent="open_world_lookup",
                confidence=0.88,
                hits=weather_hits,
                prefix="Matched open-world weather language",
            )

        if open_world_hits and not has_explicit_workspace_local_evidence:
            return self._route_from_hits(
                intent="open_world_lookup",
                confidence=0.78,
                hits=open_world_hits,
                prefix="Matched open-world lookup language",
            )

        if has_explicit_workspace_local_evidence and (
            workspace_read_hits
            or workspace_target_hits
            or workspace_attachment_count > 0
        ):
            hits = workspace_read_hits or workspace_reference_hits or workspace_target_hits
            return self._route_from_hits(
                intent="workspace_read",
                confidence=0.82,
                hits=hits,
                prefix="Matched workspace read language",
            )

        if attachment_count > 0 and workspace_attachment_count > 0:
            return ToolIntentRoute(
                intent="workspace_read",
                confidence=0.7,
                source="fallback_keyword",
                rationale=(
                    "Workspace attachments are present without stronger open-world or research signals, "
                    "so routing conservatively to workspace read."
                ),
            )

        return ToolIntentRoute(
            intent="ambiguous",
            confidence=0.0,
            source="fallback_keyword",
            rationale="No strong fallback keyword signals matched the bounded tool-intent taxonomy.",
        )

    @staticmethod
    def _route_from_hits(
        *,
        intent: ToolIntent,
        confidence: float,
        hits: list[str],
        prefix: str,
    ) -> ToolIntentRoute:
        unique_hits = list(dict.fromkeys(hit for hit in hits if hit))
        rationale = prefix
        if unique_hits:
            rationale += f": {', '.join(unique_hits[:4])}."
        else:
            rationale += "."
        return ToolIntentRoute(
            intent=intent,
            confidence=confidence,
            source="fallback_keyword",
            rationale=rationale,
        )

    @classmethod
    def _matched_keywords(cls, lowered_message: str, keywords: tuple[str, ...]) -> list[str]:
        return [keyword for keyword in keywords if cls._matches_keyword(lowered_message, keyword)]

    @classmethod
    def _workspace_reference_hits(cls, lowered_message: str) -> list[str]:
        hits = cls._matched_keywords(lowered_message, _WORKSPACE_REFERENCE_KEYWORDS)
        if _WORKSPACE_FILE_REFERENCE_PATTERN.search(lowered_message) is not None:
            hits.append("explicit file reference")
        if _WORKSPACE_PATH_REFERENCE_PATTERN.search(lowered_message) is not None:
            hits.append("explicit path reference")
        return list(dict.fromkeys(hit for hit in hits if hit))

    @staticmethod
    def _matches_keyword(lowered_message: str, keyword: str) -> bool:
        if not keyword:
            return False
        if keyword[0].isalnum() and keyword[-1].isalnum():
            pattern = rf"(?<![a-z0-9_]){re.escape(keyword)}(?![a-z0-9_])"
            return re.search(pattern, lowered_message) is not None
        return keyword in lowered_message


def _normalize_confidence(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
    elif isinstance(value, str):
        try:
            numeric = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    if numeric < 0.0:
        return 0.0
    if numeric > 1.0:
        return 1.0
    return numeric


def normalize_tool_intent_name(raw_intent: str | None) -> ToolIntent | None:
    if raw_intent is None:
        return None
    normalized = _LEGACY_INTENT_ALIASES.get(raw_intent, raw_intent)
    if normalized in _CANONICAL_INTENTS:
        return cast(ToolIntent, normalized)
    return None


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None

    candidates = [stripped]
    fenced_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", stripped, flags=re.IGNORECASE)
    if fenced_match is not None:
        candidates.append(fenced_match.group(1).strip())

    brace_match = re.search(r"\{[\s\S]*\}", stripped)
    if brace_match is not None:
        candidates.append(brace_match.group(0).strip())

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return cast(dict[str, Any], payload)
    return None


def parse_tool_intent_classifier_result(text: str) -> ToolIntentRoute:
    payload = _extract_json_object(text)
    if payload is None:
        return ToolIntentRoute(
            intent="ambiguous",
            confidence=None,
            source="classifier",
            rationale="Classifier did not return a valid JSON object.",
        )

    raw_intent = str(payload.get("intent") or "").strip()
    intent = normalize_tool_intent_name(raw_intent) or "ambiguous"
    confidence = _normalize_confidence(payload.get("confidence"))
    rationale = str(payload.get("rationale") or "").strip() or "No classifier rationale was provided."

    if intent == "ambiguous" and confidence is None:
        confidence = 0.0

    return ToolIntentRoute(
        intent=intent,
        confidence=confidence,
        source="classifier",
        rationale=rationale,
    )
