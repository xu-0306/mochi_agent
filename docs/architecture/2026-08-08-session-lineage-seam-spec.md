---
title: "Session Lineage Seam v1"
author: "Codex Main"
date: 2026-08-08
status: Approved
approval: "User authorized the Main-owned seam on 2026-08-08; read-only Terra high contract and persistence reviews completed before approval."
reviewers: ["Terra high metadata audit", "Terra high contract-oracle review"]
---

# Session Lineage Seam v1

**Author:** Codex Main
**Date:** 2026-08-08
**Status:** Approved

## Context

`PKG-WORKER-31-SESSION-SEARCH` needs to group results by lineage and exclude a
current session's complete lineage. The frozen `session-search-index-v1`
contract deliberately exposes only JSONL-derived document references and
bounded previews; it has no parent or root relationship. Treating every
session ID as its own root would leak ancestor or descendant search results.

Mochi's session fork endpoint currently copies history but does not retain the
source session or fork turn. Existing historical sessions therefore cannot be
reliably reconstructed as a lineage. This Main-owned seam adds canonical,
forward-only lineage metadata for newly created sessions while preserving
legacy sessions as independently rooted histories.

## Functional Requirements

- FR-1: Every session created by `POST /v1/sessions` MUST persist an immutable
  `session_meta.created.lineage` v1 envelope in canonical JSONL.
- FR-2: A non-forked session MUST have `parent_session_id` and
  `parent_storage_id` set to `null`. A forked session MUST name the selected
  source session, the same `SessionStore.storage_id`, and its selected
  `fork_until_turn_id`.
- FR-3: `SessionStore.resolve_session_lineage(session_id)` MUST return the
  requested `session_id`, the scoped `storage_id`, and its derived final
  `root_session_id`.
- FR-4: The resolver MUST traverse only canonical same-store metadata and MUST
  fail closed for malformed records, dangling parents, self-parent links,
  cycles, conflicting lineage envelopes, or cross-store parents.
- FR-5: A history with no lineage envelope is a legacy root only. The resolver
  MUST NOT infer ancestry from copied events or session-ID similarity.
- FR-6: The derived FTS index MUST remain unchanged: its schema, source
  snapshot, document fields, previews, and artifact policy MUST NOT acquire
  lineage fields.
- FR-7: The resolver MUST expose an async root callable suitable for the
  already fail-closed `SessionSearchService`; this package MUST NOT add a
  search route or UI behavior.
- FR-8: Rewrites of a session that already has a lineage envelope MUST retain
  exactly that envelope or fail. A later conflicting envelope MUST make
  resolution fail closed.
- FR-9: Session creation, including a fork's created metadata, project
  assignment, and cloned history, MUST commit as one durable batch only when
  the destination session does not already exist. Failed preflight validation
  and a destination collision MUST NOT create or modify a destination history.

## Non-Functional Requirements

- NFR-1: Resolution MUST make no cross-store lookup or full-store scan. It may
  read at most one strict canonical snapshot for each traversed ancestry node.
- NFR-2: Error paths MUST be machine-readable and deterministic through
  `SessionLineageError.reason`; they MUST NOT become an empty successful search.
- NFR-3: The resolver MUST not read, index, or return raw event payload beyond
  lineage metadata required for traversal.
- NFR-4: All behavior is forward compatible: a missing envelope is legacy
  self-root, and existing JSONL histories are never rewritten or inferred.

## Acceptance Criteria

### AC-1: (FR-1, FR-2) Root creation

Given a normal session creation, when its strict snapshot is read, then exactly
one valid v1 lineage envelope names the new session, current storage, and null
parent fields.

### AC-2: (FR-1, FR-2) Fork creation

Given a valid fork request, when the child is created, then its envelope records
the source session, same storage ID, selected turn, and the child resolves to
the source's root.

### AC-3: (FR-3, FR-4) Multi-hop resolution

Given root `r`, child `c -> r`, and grandchild `g -> c`, when resolving any
member, then the result preserves its requested session ID and returns `r` as
root.

### AC-4: (FR-4) Invalid chain rejection

Given a dangling parent, self-parent, cycle, cross-store parent, duplicate
envelope, or malformed identifier, when that chain is touched, then resolution
raises `SessionLineageError` with the defined reason.

### AC-5: (FR-5) Legacy compatibility

Given a legacy history with no envelope, when resolving it, then it resolves to
itself without rewriting the source.

### AC-6: (FR-6) Index compatibility

Given normal, forked, and legacy sessions, when the search index is rebuilt,
then existing index tests retain the same schema and bounded document behavior.

### AC-7: (FR-7) Search-consumer integration

Given the Store root resolver and a forked child as current session, when
`SessionSearchService` searches matching parent/child events, then it excludes
the entire lineage and returns only unrelated roots.

### AC-8: (FR-8) Immutable replacement

Given a lineage-enabled session, when `replace_session` omits or changes its
envelope, then the replacement is rejected; a replacement that preserves it
remains resolvable.

### AC-9: (FR-1, FR-2, FR-9) Atomic fork creation

Given a fork request with an unknown selected turn, when the endpoint rejects
it, then no child history exists. Given a pre-existing destination or a
destination equal to the normalized source ID, when creation is attempted,
then the endpoint rejects the request without modifying either existing
history. Concurrent requests for the same destination result in exactly one
complete created history.

## Edge Cases

- EC-1: Whitespace-only or non-string persisted IDs are invalid; record values
  are not silently normalized after persistence.
- EC-2: An absent parent envelope is legacy only. A non-empty parent ID whose
  strict source is missing is `dangling_parent`, never a legacy root.
- EC-3: `parent_session_id == session_id` is `self_parent`; any revisited ID is
  `cycle`.
- EC-4: `parent_storage_id` that differs from the local storage ID is
  `cross_store_parent`; v1 has no import, profile lookup, or rebinding.
- EC-5: An unavailable or malformed strict source produces a lineage error and
  cannot be substituted with index state.
- EC-6: A conflicting second lineage envelope makes resolution fail closed even
  when the first envelope was valid.
- EC-7: A missing fork turn, invalid source lineage, unavailable project, or
  destination collision is rejected before any child source write; the atomic
  destination guard remains authoritative against a concurrent creator.

## API Contracts

```ts
interface SessionLineageEnvelopeV1 {
  schema_version: 1;
  storage_id: string;
  session_id: string;
  parent_session_id: string | null;
  parent_storage_id: string | null;
  fork_until_turn_id: string | null;
}

interface SessionLineageResolution {
  storage_id: string;
  session_id: string;
  root_session_id: string;
}

type SessionLineageErrorReason =
  | "malformed_id" | "record_not_found" | "dangling_parent"
  | "self_parent" | "cycle" | "cross_store_parent"
  | "conflicting_record" | "invalid_resolution" | "resolver_unavailable";
```

`POST /v1/sessions` keeps its request and response shape. The lineage envelope
is internal canonical metadata, not a new public response field.

## Data Models

| Field | Type | Constraints |
| --- | --- | --- |
| `storage_id` | string | Exact current `SessionStore.storage_id`; non-empty. |
| `session_id` | string | Exact session file identity; non-empty. |
| `parent_session_id` | string or null | Null for root; otherwise same-store strict source exists. |
| `parent_storage_id` | string or null | Null exactly when parent is null; otherwise equals `storage_id`. |
| `fork_until_turn_id` | string or null | Non-empty exactly for a forked child. |
| `root_session_id` | string | Derived only; never persisted as authority. |

## Out of Scope

- OS-1: No search HTTP route, WebGUI, session browsing, scrolling, pagination, or
  cross-profile lookup; those belong to Worker-32 and later UI work.
- OS-2: No migration that guesses lineage for existing copied histories.
- OS-3: No SQLite index schema change or lineage projection.
- OS-4: No relation between delegated-subagent `parent_session_id` correlation and
  canonical session ancestry.
- OS-5: No cross-store import, export, or repair tool.

## Verification

`GATE-SESSION-LINEAGE-SEAM` runs `tests/test_session_lineage.py`, focused
SessionStore lineage preservation tests, and the session-fork route tests.
`GATE-CONTRACT-FREEZE` validates `session-lineage-v1.json` at the pinned
revision. Worker-31 and Worker-32 are not re-dispatched until both gates pass.
