# File Subset Preview Transport

Status: Approved (direct additive projection of frozen `file-change-v2`)\
Author: Main\
Date: 2026-08-08\
Reviewers: Frozen contract `file-change-v2`; Main-50 integration evidence

## Context

`file-change-v2` already defines an immutable, server-authoritative child request for a selected subset of a persisted parent change request. The current patch-preview route only accepts a replacement patch and creates a new root request. The browser cannot therefore request a contract-correct subset manifest, digest, or replacement approval.

This adds a narrow transport boundary. It does not change file mutation, persistence schemas, the frozen contract, or the existing edited-patch preview route.

## Functional Requirements

1. FR-1: `POST /v1/workspace/patch/subset-preview` MUST accept only a pending parent `approval_id` and a non-empty `selected_entry_ids` array.
2. FR-2: The server MUST derive entries, parent linkage, authorization digest, manifest, and replacement approval from persisted server state; client-supplied patch, digest, entry projections, and parent digest are forbidden.
3. FR-3: The server MUST revalidate the immediate parent before deriving the child. A changed policy, workspace, file identity, content, or patch MUST conflict the parent and return a terminal conflict response.
4. FR-4: The server MUST call the frozen subset derivation and persistence primitives, preserving immediate-parent lineage and indivisible dependency groups.
5. FR-5: A successful child request MUST have a distinct request digest and a replacement pending approval. The parent approval MUST be superseded atomically.
6. FR-6: A repeated request with the same superseded parent and same selection MUST return its existing replacement approval; a different selection MUST fail with conflict.
7. FR-7: The typed web client and task store MUST send only the approval ID and selected IDs and surface only server-returned state.

## Non-Functional Requirements

1. NFR-1: The endpoint MUST not expose raw before/after blobs or accept a client authorization digest.
2. NFR-2: Conflict and validation failures MUST produce deterministic 4xx responses; they MUST NOT silently retain the parent approval.
3. NFR-3: The response MUST be sufficient for TaskPanel to display the replacement approval, child digest, selected child entries, and exclusions without locally deriving child IDs.

## Acceptance Criteria

1. AC-1 (FR-1, FR-2): Given a pending root approval, when a valid complete selection is posted, then the response contains a server-created child digest, child manifest identifiers, replacement approval ID, parent digest, and selected IDs.
2. AC-2 (FR-3): Given a stale parent, when subset preview is requested, then the response is conflict and the parent is not silently reusable.
3. AC-3 (FR-4, FR-5): Given dependency-group members, when only part of a group is selected, then the response is conflict and no child approval is created.
4. AC-4 (FR-6): Given a successful selection, when the identical request is retried, then it returns the same replacement approval; a different selection returns conflict.
5. AC-5 (FR-7, NFR-1): Given the web store request, when it sends a subset preview, then its body contains only `approval_id` and `selected_entry_ids`.

## Edge Cases

1. EC-1: Missing approval returns 404.
2. EC-2: Empty, duplicate, or unknown selected IDs return 422.
3. EC-3: A parent approval which is consumed, rejected, expired, or superseded by another child returns 409.
4. EC-4: A parent request whose persisted change set is absent or not prepared returns 409.
5. EC-5: FTS, UI metadata, and client display state are never authorization inputs.

## API Contracts

```ts
type WorkspaceSubsetPreviewRequest = {
  approval_id: string
  selected_entry_ids: string[]
}

type WorkspaceSubsetPreviewResponse = {
  type: 'workspace_patch_subset_preview'
  valid: true
  change_set_id: string
  request_digest: string
  parent_request_digest: string
  selected_entry_ids: string[]
  file_changes: Array<{ entry_id: string; relative_path: string; dependency_group: string | null }>
  replacement_approval_id: string
  approval_state: 'replacement_pending'
  expires_at: string
  policy_version: string
}
```

Error responses are `404` for a missing approval, `422` for malformed selection, and `409` for parent, revalidation, dependency-group, or approval-state conflicts.

## Data Models

| Entity | Fields | Constraints |
| --- | --- | --- |
| Parent approval | `approval_id`, persisted metadata, status | Must identify a prepared immediate parent manifest. |
| Child request | `parent_request_digest`, `selected_entry_ids`, entries | Produced only by `derive_file_change_subset`. |
| Child manifest | `change_set_id`, `request_digest`, child entries | Persisted before the replacement approval is created. |
| Replacement approval | `approval_id`, request/context digest | Atomically supersedes the parent; never reuses the parent digest. |

## Out of Scope

- Editing patches, applying changes, authorization policy changes, persistence migrations, and UI rendering are out of scope.
- `TaskPanel` checkbox rendering and browser coverage remain `PKG-WORKER-51-FILE-WEB` after this boundary is available.
