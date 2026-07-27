"""Deterministic lexical ranking for local tool discovery."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from mochi.tools.base import BaseTool

_TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)


class ToolCatalogIndexError(RuntimeError):
    """Fail-closed catalog normalization or ranking error."""

    def __init__(self, *, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = str(kind or "").strip() or "malformed_catalog"


def normalize_catalog_text(value: str) -> str:
    """Normalize search text while preserving multilingual content."""

    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(normalized.split()).strip()


@dataclass(frozen=True)
class ToolCatalogDocument:
    """Stable searchable metadata for one tool."""

    name: str
    description: str
    search_hint: str | None
    argument_names: tuple[str, ...]
    argument_descriptions: tuple[str, ...]
    capability_tags: tuple[str, ...]

    @classmethod
    def from_tool(cls, tool: BaseTool) -> ToolCatalogDocument:
        if not isinstance(tool, BaseTool):
            raise ToolCatalogIndexError(
                kind="malformed_catalog",
                message="tool catalog entries must be BaseTool instances",
            )

        name = str(tool.name or "").strip()
        if not name:
            raise ToolCatalogIndexError(
                kind="malformed_catalog",
                message="tool catalog entry name must not be empty",
            )

        schema = tool.parameters_schema
        if not isinstance(schema, dict):
            raise ToolCatalogIndexError(
                kind="malformed_catalog",
                message=f"tool '{name}' parameters_schema must be an object",
            )
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            raise ToolCatalogIndexError(
                kind="malformed_catalog",
                message=f"tool '{name}' schema properties must be an object",
            )

        argument_names: list[str] = []
        argument_descriptions: list[str] = []
        for raw_key, raw_value in properties.items():
            key = str(raw_key or "").strip()
            if not key:
                raise ToolCatalogIndexError(
                    kind="malformed_catalog",
                    message=f"tool '{name}' has an empty schema property name",
                )
            argument_names.append(key)
            if raw_value is None:
                continue
            if not isinstance(raw_value, dict):
                raise ToolCatalogIndexError(
                    kind="malformed_catalog",
                    message=f"tool '{name}' schema property '{key}' must be an object",
                )
            description = raw_value.get("description")
            if description is not None:
                argument_descriptions.append(str(description))

        capability_tags = tuple(
            tag
            for tag in _dedupe_preserve_order(_string_fragments(tool.tool_capabilities))
            if tag
        )

        raw_search_hint = tool.search_hint
        search_hint = None
        if isinstance(raw_search_hint, str) and raw_search_hint.strip():
            search_hint = raw_search_hint.strip()

        return cls(
            name=name,
            description=str(tool.description or "").strip(),
            search_hint=search_hint,
            argument_names=tuple(argument_names),
            argument_descriptions=tuple(argument_descriptions),
            capability_tags=capability_tags,
        )

    def to_fingerprint_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "search_hint": self.search_hint,
            "argument_names": list(self.argument_names),
            "argument_descriptions": list(self.argument_descriptions),
            "capability_tags": list(self.capability_tags),
        }


@dataclass(frozen=True)
class CatalogSearchCandidate:
    """One ranked catalog match."""

    name: str
    rank: int
    score: float
    catalog_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rank": self.rank,
            "score": self.score,
            "catalog_fingerprint": self.catalog_fingerprint,
        }


@dataclass(frozen=True)
class _IndexedDocument:
    document: ToolCatalogDocument
    normalized_name: str
    compact_name: str
    name_tokens: frozenset[str]
    description_text: str
    search_hint_text: str
    argument_name_text: str
    argument_description_text: str
    capability_text: str
    metadata_compact: str


class ToolCatalogIndex:
    """Bounded deterministic search over tool metadata."""

    def __init__(
        self,
        documents: Sequence[ToolCatalogDocument],
        *,
        default_top_k: int = 5,
        max_top_k: int = 10,
    ) -> None:
        if default_top_k <= 0:
            raise ValueError("default_top_k must be greater than 0")
        if max_top_k <= 0:
            raise ValueError("max_top_k must be greater than 0")
        self._default_top_k = int(default_top_k)
        self._max_top_k = max(self._default_top_k, int(max_top_k))

        indexed: list[_IndexedDocument] = []
        by_name: dict[str, ToolCatalogDocument] = {}
        for document in documents:
            if not isinstance(document, ToolCatalogDocument):
                raise ToolCatalogIndexError(
                    kind="malformed_catalog",
                    message="catalog documents must be ToolCatalogDocument instances",
                )
            if document.name in by_name:
                raise ToolCatalogIndexError(
                    kind="malformed_catalog",
                    message=f"duplicate tool catalog entry: {document.name}",
                )
            by_name[document.name] = document
            indexed.append(_build_indexed_document(document))

        self._documents_by_name = by_name
        self._indexed_documents = tuple(indexed)
        self._catalog_fingerprint = _catalog_fingerprint(documents)

    @classmethod
    def from_tools(
        cls,
        tools: Iterable[BaseTool],
        *,
        default_top_k: int = 5,
        max_top_k: int = 10,
    ) -> ToolCatalogIndex:
        try:
            catalog = list(tools)
        except Exception as exc:  # pragma: no cover - defensive iterable failure
            raise ToolCatalogIndexError(
                kind="malformed_catalog",
                message=f"tool catalog provider returned a non-iterable catalog: {exc}",
            ) from exc
        documents = [ToolCatalogDocument.from_tool(tool) for tool in catalog]
        return cls(documents, default_top_k=default_top_k, max_top_k=max_top_k)

    @property
    def catalog_fingerprint(self) -> str:
        return self._catalog_fingerprint

    def document_for_name(self, name: str) -> ToolCatalogDocument | None:
        return self._documents_by_name.get(str(name or "").strip())

    def search(self, query: str, top_k: int | None = None) -> list[CatalogSearchCandidate]:
        raw_query = str(query or "").strip()
        if not raw_query:
            return []

        requested_top_k = self._default_top_k if top_k is None else int(top_k)
        if requested_top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        bounded_top_k = min(requested_top_k, self._max_top_k)

        exact_match = self._select_lookup(raw_query)
        if exact_match is not None:
            return [
                CatalogSearchCandidate(
                    name=exact_match.name,
                    rank=1,
                    score=1_000.0,
                    catalog_fingerprint=self._catalog_fingerprint,
                )
            ]

        normalized_query = normalize_catalog_text(raw_query)
        compact_query = _compact_text(normalized_query)
        if not compact_query:
            return []
        tokens = tuple(dict.fromkeys(_query_tokens(normalized_query, compact_query)))

        ranked: list[tuple[float, str]] = []
        for indexed in self._indexed_documents:
            score = _score_document(
                indexed,
                normalized_query=normalized_query,
                compact_query=compact_query,
                query_tokens=tokens,
            )
            if score <= 0.0:
                continue
            ranked.append((score, indexed.document.name))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [
            CatalogSearchCandidate(
                name=name,
                rank=position,
                score=round(score, 4),
                catalog_fingerprint=self._catalog_fingerprint,
            )
            for position, (score, name) in enumerate(ranked[:bounded_top_k], start=1)
        ]

    def _select_lookup(self, raw_query: str) -> ToolCatalogDocument | None:
        if not raw_query.casefold().startswith("select:"):
            return None
        target = raw_query.split(":", 1)[1].strip()
        if not target:
            return None
        direct = self._documents_by_name.get(target)
        if direct is not None:
            return direct
        normalized_target = normalize_catalog_text(target)
        for name, document in self._documents_by_name.items():
            if normalize_catalog_text(name) == normalized_target:
                return document
        return None


def _build_indexed_document(document: ToolCatalogDocument) -> _IndexedDocument:
    normalized_name = normalize_catalog_text(document.name)
    compact_name = _compact_text(normalized_name)
    description_text = normalize_catalog_text(document.description)
    search_hint_text = normalize_catalog_text(document.search_hint or "")
    argument_name_text = normalize_catalog_text(" ".join(document.argument_names))
    argument_description_text = normalize_catalog_text(" ".join(document.argument_descriptions))
    capability_text = normalize_catalog_text(" ".join(document.capability_tags))
    metadata_parts = [
        normalized_name,
        description_text,
        search_hint_text,
        argument_name_text,
        argument_description_text,
        capability_text,
    ]
    metadata_text = " ".join(part for part in metadata_parts if part)
    return _IndexedDocument(
        document=document,
        normalized_name=normalized_name,
        compact_name=compact_name,
        name_tokens=frozenset(_query_tokens(normalized_name, compact_name)),
        description_text=description_text,
        search_hint_text=search_hint_text,
        argument_name_text=argument_name_text,
        argument_description_text=argument_description_text,
        capability_text=capability_text,
        metadata_compact=_compact_text(metadata_text),
    )


def _score_document(
    indexed: _IndexedDocument,
    *,
    normalized_query: str,
    compact_query: str,
    query_tokens: Sequence[str],
) -> float:
    score = 0.0
    if compact_query == indexed.compact_name:
        return 900.0
    if compact_query in indexed.compact_name:
        score += 60.0 if indexed.compact_name.startswith(compact_query) else 42.0
    if compact_query and compact_query == indexed.metadata_compact:
        score += 35.0
    elif len(compact_query) >= 2 and compact_query in indexed.metadata_compact:
        score += 18.0

    unique_tokens = tuple(dict.fromkeys(token for token in query_tokens if token))
    if unique_tokens and all(token in indexed.metadata_compact for token in unique_tokens):
        score += 8.0

    for token in unique_tokens:
        if token in indexed.name_tokens:
            score += 16.0
        elif token in indexed.compact_name:
            score += 7.0
        if token and token in indexed.description_text:
            score += 4.0
        if token and token in indexed.search_hint_text:
            score += 5.0
        if token and token in indexed.argument_name_text:
            score += 4.0
        if token and token in indexed.argument_description_text:
            score += 3.0
        if token and token in indexed.capability_text:
            score += 4.0

    if normalized_query and normalized_query == indexed.description_text:
        score += 12.0
    if normalized_query and normalized_query == indexed.search_hint_text:
        score += 10.0
    return score


def _catalog_fingerprint(documents: Sequence[ToolCatalogDocument]) -> str:
    canonical = [
        document.to_fingerprint_dict()
        for document in sorted(documents, key=lambda item: item.name)
    ]
    payload = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _compact_text(value: str) -> str:
    return normalize_catalog_text(value).replace(" ", "")


def _query_tokens(normalized_text: str, compact_text: str) -> tuple[str, ...]:
    tokens = [token for token in _TOKEN_RE.findall(normalized_text) if token]
    if compact_text and compact_text not in tokens:
        tokens.append(compact_text)
    return tuple(tokens)


def _string_fragments(value: Any) -> list[str]:
    if isinstance(value, dict):
        parts: list[str] = []
        for key, item in value.items():
            parts.append(str(key))
            parts.extend(_string_fragments(item))
        return parts
    if isinstance(value, (list, tuple, set, frozenset)):
        parts = []
        for item in value:
            parts.extend(_string_fragments(item))
        return parts
    if value is None:
        return []
    return [str(value)]


def _dedupe_preserve_order(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return tuple(ordered)
