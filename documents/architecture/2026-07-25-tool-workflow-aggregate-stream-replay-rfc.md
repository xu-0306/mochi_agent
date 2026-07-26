# Tool Workflow Aggregate Stream and Replay RFC

Status: Proposed；Phase 0-4 implementation and final rollout acceptance verified 2026-07-25

Date: 2026-07-25

Owners: Ordinary Chat runtime, session persistence, API, and web client

## 1. Decision

Define tool_workflow_aggregate v1 as the only aggregate workflow read model for one ordinary-Chat turn. It is derived from durable sources and delivered through an idempotent outbox. It is not a second authority for policy, approval, side effects, or artifact verification.

The current API tool_workflow projection and the existing per-event turn_event transcript remain compatibility inputs during migration. Neither is an authoritative aggregate state machine.

定稿當時的 production blocker 是 Engine mutation timeline integration：Engine 雖會
admit、claim、terminalize timeline turn，但 side-effect tool path 尚未呼叫
`TimelineCoordinator.before_mutation()` 或 `persist_tool_result()`，ReAct loop
仍會配置 random observed operation ID。此段保留為設計歷史；在 Phase 1 已完成
timeline-generated deterministic operation ID、precommit/effect boundary/result
persistence 接線後，該 blocker 已解除。

This RFC does not authorize a production-code change by itself.

### 實作狀態更新（2026-07-25）

以下更新僅記錄 implementation status 與 rollout gate；不改變本 RFC 對權威
sources、identity joins、fail-closed reducer、outbox 非 source-of-truth，或
side effect 不可由 replay 重播的定稿語義。

- [x] Phase 0：pure reducer、strict v1 reader、legacy partial adapter 與
  canonical JSON subset 已完成。
- [x] Phase 1：Engine mutation timeline integration 已完成，side-effecting
  execution 使用 timeline-generated deterministic operation ID，並保存
  precommit/effect boundary/result。
- [x] Phase 2：`tool_observability_v1` 的 durable outbox、approval
  observation/restart repair、idempotent reconciler、live publish gate、delta
  與 startup server-only verifier，以及 source mismatch/duplicate/gap/
  unsupported counters 已完成。repair 只補 durable observation/outbox，絕不
  執行或重播 approved tool call。
- 主代理獨立驗收：aggregate/outbox/API 76 passed；approval 78 passed；
  workflow 72 passed；並已通過 `compileall` 與 `git diff --check`。
- [x] Phase 3（snapshot/range API、named SSE、`Last-Event-ID`／per-turn
  cursor/gap repair、frontend strict parser/store、`ToolCallCard` projection）
  implementation 已完成。sessions_dir 已收斂為 startup-only invariant；storage
  marker 與 storage scope mismatch 由 server/client fail closed 處理，不實作
  live hot-switch。
- [x] Phase 4：ToolCallCard 已切換為 durable aggregate call projection；raw
  transcript 只保留為 non-authoritative display fallback。
- [x] P2.2 full same-session ordinary-Chat model-history linearization 已於
  2026-07-26 由獨立 acceptance matrix 核實；它不是 Phase 0-2 aggregate
  實作的完成條件，證據亦與本 RFC 的 aggregate gate 分開記錄。

Implementation verification update (2026-07-26): 51 frontend `test:*` scripts are
enumerated; 48 non-browser scripts have clean exits. Three dev-server browser
fixtures emitted assertion success but remain environment-limited by the local
runner timeout; frontend type-check passed and lint
has 0 errors with 4 pre-existing warnings. Backend aggregate/outbox/
observability passed 45, approval/lifecycle/rehydration passed 95,
timeline/exec workflow passed 130, the current rollout/config/settings matrix passed
158, and the current migration/storage-scope matrix passed 48. The isolated production build passed with
`MOCHI_NEXT_DIST_DIR=.next-codex-build`; the default `.next` output may remain
locked by an external Next process on Windows.

## 2. Goals and Non-goals

Goals:

- Make stream, reconnect, and session reload converge on the same turn view.
- Rebuild the view from durable timeline, turn checkpoint, approval, and receipt evidence after restart.
- Represent absence of evidence, cancellation, and unknown side-effect state truthfully without inferring success.
- Give every delivered aggregate update a stable event ID, per-turn sequence, and idempotency key.
- Reject unsupported future payloads before they alter client workflow state.

Non-goals:

- Replacing TurnIntentContract, CapabilityPlan, approval, timeline, or artifact receipts as semantic authorities.
- Replaying a side effect from the aggregate event or from an SSE reconnect.
- Duplicating every raw model/tool transcript event as a second event stream.
- Persisting unredacted tool arguments or outputs in a new aggregate payload.

## 3. Authoritative Sources

The reducer must consume sources in this order of authority. A source is used only when its identity, turn ID, and join fields validate.

| Source | Authoritative fields | Not authoritative for |
|---|---|---|
| Session turn timeline | turn sequence, lane state, terminal/cancellation outcome, side-effect boundary, operation descriptor, operation result digest/receipt reference | policy, approval decision, artifact acceptance |
| Turn checkpoint | resolved contract/capability plan, policy/inventory snapshot, activation snapshot, pending call, approval reference, execution/verification checkpoint | current approval-store status when a matching approval exists |
| Persistent approval store | approval lifecycle, durable monotonic approval_revision, request/context digest, consume lease, execution idempotency key, applied result | policy or target drift acceptance after the approval was issued |
| Artifact receipt | operation/turn evidence and verification outcome | execution success when its operation binding does not match |
| turn_event transcript | redacted display enrichment only | lifecycle truth, approval, execution, verification, and idempotency |

The existing API tool_workflow object and browser-local ToolCard projection are never reducer inputs.

All evidence joins require the same session_id and turn_id. Per-call joins also require call_id, operation_id, and arguments_digest when those fields are available. A mismatch, duplicate contradictory join, missing required identity, or malformed source makes the aggregate partial or unsupported; it never fabricates a successful call.

`source_refs` records only the reducer's authoritative inputs: their source-local
positions or monotonic revisions and canonical content digests. It excludes every
aggregate/outbox/delivery record, including its own prior entries and any raw
transcript display event. Consequently, writing or replaying an aggregate cannot
change the next reduction's source references.

## 4. Aggregate Event v1

The durable outbox record and SSE payload use this envelope. JSON field names are snake_case. A v1 writer emits all required fields and no unregistered top-level fields.

~~~json
{
  "type": "tool_workflow_aggregate",
  "schema_version": 1,
  "event_id": "twa:v1:<43-character-base64url-sha256>",
  "seq": 7,
  "idempotency_key": "sha256:<canonical-source-state>",
  "session_id": "session-123",
  "turn_id": "turn-456",
  "occurred_at": "2026-07-25T12:00:00+00:00",
  "source_refs": {
    "timeline": {
      "timeline_version": "session-turn-timeline-v3",
      "turn_sequence": 4,
      "events": [
        {
          "source_position": 41,
          "kind": "operation_result",
          "digest": "sha256:<canonical-event>"
        }
      ]
    },
    "checkpoint": {
      "checkpoint_revision": 3,
      "digest": "sha256:<canonical-checkpoint>"
    },
    "approvals": [],
    "receipts": []
  },
  "state": {
    "turn_status": "awaiting_approval",
    "integrity": "complete",
    "policy": {},
    "inventory": {},
    "calls": [],
    "blocker": null
  }
}
~~~

seq is a positive integer, strictly increasing within one session_id and turn_id stream. It is allocated under the SessionStore strict CAS only after the writer has found no existing outbox entry with the same idempotency_key. It is not the current raw turn_event.seq, which is not sufficient for cross-source ordering.

event_id is deterministic but opaque: it is `twa:v1:` followed by the unpadded,
43-character base64url encoding of SHA-256 over RFC 8785 canonical JSON for
`{"schema_version":1,"session_id":...,"turn_id":...,"seq":...}`. It has the
exact parser format `^twa:v1:[A-Za-z0-9_-]{43}$`; parsers compare the complete
opaque value and never split embedded identities. Canonical serialization makes
identities unambiguous even when a session or turn ID contains colons or other
special characters.

idempotency_key is `sha256:` plus SHA-256 over RFC 8785 canonical JSON containing
only schema_version, source_refs, and normalized aggregate state. Normalization
uses explicit nulls for absent registered fields and sorts identity-bearing arrays
by their semantic identities (calls by call_id, approvals by approval_id, receipts
by operation_id and receipt reference). The input excludes aggregate/outbox or
delivery records, event_id, seq, and occurred_at or any delivery timestamp. Before
a writer allocates a sequence it must find an existing entry with the same key;
that entry is returned instead of appending a duplicate. Appending or replaying an
outbox record therefore reproduces the same key and never allocates another
sequence. Equal sequence with a different event ID or idempotency key is
corruption.

source_refs.timeline.events contains each reducer-consumed authoritative timeline
event's source_position, kind, and canonical digest; it must not use a
SessionStore-wide history revision. source_refs.checkpoint contains the checkpoint
revision and canonical digest. source_refs.approvals contains only approval_id,
approval_revision, status, and approved request/context digest. approval_revision
is a durable positive integer atomically incremented for every approval lifecycle
transition; a current v1 writer must reject an approval row without it and must
never substitute updated_at. A legacy adapter may use a canonical approval-row
digest only when reading a legacy store that has no revision. source_refs.receipts
contains only receipt kind/version, source position or revision, operation_id,
receipt digest/reference, and verification status. It must not copy sensitive
command arguments, environment values, or full tool output.

state.policy is the server-resolved effective snapshot identity/version and display-safe fields. state.inventory keeps the existing explicit catalog_scope=policy_eligible, policy_catalog, eligible, exposed, and activation information. Displayable call arguments and targets, if needed, must use the existing tool-specific redaction path and must accompany an arguments_digest.

Each state.calls item contains:

~~~json
{
  "call_id": "call-1",
  "operation_id": "ordinary-chat-operation-...",
  "tool_name": "file_write",
  "arguments_digest": "sha256:...",
  "target": {"display": "README.md", "redacted": false},
  "activation_status": "activated",
  "review_status": "approved",
  "approval_id": "approval-1",
  "execution_status": "succeeded",
  "verification_status": "verified",
  "receipt_reference": "receipt-1",
  "changed_paths": ["README.md"],
  "blocker": null
}
~~~

## 5. State Vocabulary and Reducer Rules

Allowed turn statuses are queued, running, awaiting_approval, executing, verifying, completed, blocked, cancelled, and unknown.

Allowed call statuses are intentionally separate:

- Activation: not_observed, requested, activated, rejected, failed, unknown.
- Review: not_observed, pending, approved, rejected, expired, consuming, consumed, unknown.
- Execution: not_started, precommitted, started, succeeded, failed, abandoned, cancelled, unknown.
- Verification: not_required, pending, verified, failed, unknown.
- Integrity: complete, partial, unsupported.

not_observed means there is no durable evidence. unknown means a side-effect boundary may have been crossed but its outcome is not known. They are never interchangeable.

Execution states have these additional invariants:

- not_started means no exact operation descriptor was durably precommitted for the call.
- precommitted means an exact descriptor is durable but its effect boundary has not been crossed and no terminal no-effect result is durable.
- abandoned means that exact precommitted descriptor has a durable pre-effect abandonment result, its effect boundary is still not_started, and the runtime therefore knows that this operation had no effect. It preserves the operation identity and result receipt; it is not a synonym for not_started or cancelled.
- started means the durable effect boundary was crossed but no terminal result has yet been recorded. If recovery cannot prove the result, it becomes unknown, never abandoned.
- cancelled is a cancellation outcome for a call without a more specific operation result. It must not overwrite abandoned evidence. unknown always wins if the effect boundary was or may have been crossed.

Reducer precedence, from highest to lowest:

1. An unsupported or invalid authoritative source sets integrity to unsupported; no terminal success is emitted.
2. A timeline operation or turn marked unknown makes the matching call and turn unknown. This is sticky until a separately verified reconciliation writes a new valid source transition; an SSE replay cannot clear it.
3. A matching abandoned descriptor is authoritative for known no-effect execution only when its operation ID, call ID, argument digest, precommit boundary=not_started, and durable abandonment result all match. It maps to abandoned and may not be replayed; any later attempt requires a fresh operation identity.
4. A matching approval-store status is authoritative for review. A pending, approved, expired, rejected, or consuming approval blocks any inferred execution result until matching durable execution evidence exists.
5. A timeline operation result is authoritative for execution only when its operation ID, call ID, and argument digest match the call evidence.
6. A matching artifact receipt is authoritative for verification. A failed or missing required receipt prevents completed.
7. Timeline cancellation produces cancelled only when there is no matching abandoned or unknown operation. A known pre-effect abandonment remains abandoned even if its containing turn is cancelled; unknown always wins over cancellation.
8. completed requires a terminal timeline outcome, no unknown operation, no unresolved approval, and verified evidence for every required artifact. blocked covers failed execution, failed verification, abandoned pre-effect execution, and durable policy or source validation blockers.

The reducer may use transcript events to add a display label, but it must not turn an error-free tool_call_result into succeeded, nor turn missing verification into verified.

## 6. Durable Outbox and Cross-store Consistency

The outbox is a delivery cache, not a source of truth. Its payload is fully rebuildable from the authoritative sources in section 3.

For a SessionStore-only transition, append the source transition and aggregate outbox record in one strict CAS batch. This applies to timeline lifecycle, checkpoint, receipt, and terminal transitions when their source data is in the same session log.

Approval lifecycle data lives in a separate durable store, so a cross-store atomic transaction is unavailable. The required sequence is:

1. Commit the approval-store transition under its existing lease/CAS rules.
2. Append an idempotent session approval_observation record keyed by approval_id, approval_revision, and status. approval_revision is atomically incremented with the approval-store lifecycle transition; `updated_at` is not a revision surrogate.
3. Reduce the newly committed sources and append the aggregate outbox record under SessionStore CAS.
4. A restart-safe reconciler scans checkpoint approval references and repairs a missing observation/outbox entry. It may publish a new aggregate, but it must never re-execute the approved tool call.

If step 2 or 3 fails, the client must continue to show the last aggregate or partial; it must not optimistically show the approval transition. Repair is idempotent through the aggregate idempotency key.

## 7. Legacy and Future Readers

There is no prior aggregate schema. The v1 legacy adapter accepts only the currently known raw turn_event v1, supported checkpoint schema, supported timeline versions, known approval record, and supported receipt versions. It creates a partial aggregate when the sources do not prove a join.

The legacy adapter may retain legacy transcript event IDs/sequences for display, but it must not synthesize an operation ID, approval state, receipt, or verification result. For a legacy approval row without approval_revision, it may use only that row's canonical digest as the source reference; new/current writers require the monotonic revision. Missing evidence remains not_observed or unknown as appropriate.

A v1 reader validates required fields, exact types, positive seq, event-ID format, and source-reference consistency. An aggregate with a future schema_version, unsupported source version, duplicate conflicting sequence, or invalid state vocabulary is rejected. The browser keeps the last accepted state, marks the turn unsupported, and reloads; it never partially applies a future payload.

## 8. SSE, Reconnect, and Reload Rules

The SSE serializer must support named events and IDs:

~~~text
id: twa:v1:<43-character-base64url-sha256>
event: tool_workflow_aggregate
data: {"type":"tool_workflow_aggregate", ...}

~~~

Publish an aggregate only after its source transition and outbox entry are durable. A live client applies events per turn in sequence order:

- Same event_id: ignore as a duplicate.
- seq equal to last_seq plus one: validate and apply.
- seq less than or equal to last_seq with a different ID/key: treat as corruption and reload.
- seq greater than last_seq plus one: do not infer intermediate state; fetch the turn aggregate snapshot/outbox range and resume only after the gap is resolved.

The stream endpoint accepts Last-Event-ID or an explicit per-turn after_aggregate_seq cursor. Its replay is sourced from the durable outbox, not from an in-memory generator. The existing raw transcript stream can remain for chat rendering during migration, but aggregate events use a distinct SSE event name.

Non-stream chat, session reload, and a dedicated turn-workflow snapshot endpoint must invoke the same server reducer. For identical durable sources, their latest aggregate payload must be byte-equivalent after canonical JSON serialization. Reload must not use browser event order as an authority.

## 9. Required Implementation Wiring

1. mochi/agents/react_loop.py: wrap side-effecting registry execution with timeline precommit/start and result persistence. Replace the random observed operation ID with the timeline-generated operation ID. A proven pre-effect failure must persist the matching descriptor as abandoned rather than cancelled or not_started; only an operation that started or may have started without a known result becomes unknown. Read-only calls may remain transcript-only.
2. mochi/agents/engine.py: make the claimed TimelineCoordinator available to the execution context; publish source transitions after checkpoint changes, ordinary-chat approval continuation, receipt persistence, cancellation, and terminalization.
3. mochi/sessions/turn_timeline.py and mochi/sessions/timeline_coordinator.py: add only the repository/outbox hooks needed to allocate aggregate sequence and atomically append compatible companion records. Do not put UI projection logic in the timeline model.
4. mochi/runtime/approval_lifecycle.py and approval routes: atomically increment approval_revision on each durable lifecycle transition, then create the idempotent approval-observation/outbox handoff and repair path.
5. New server reducer/repository: load and validate sources, produce the v1 aggregate, and rebuild missing delivery records without touching tool code.
6. mochi/api/routes/chat.py and mochi/utils/streaming.py: return the reducer output for non-stream responses and emit named SSE aggregate events with IDs/cursors only after durability.
7. Session routes: expose a turn-scoped aggregate snapshot and outbox range; do not require a full session transcript reload to repair a sequence gap.
8. web/src/lib/api.ts: add a strict v1 parser and per-turn aggregate store; retain raw transcript materialization only for messages.
9. web/src/components/chat/ToolCallCard.tsx: consume aggregate call state by call_id and operation_id, replacing the current single-event workflow inference. The UI must render partial, abandoned known no-effect, unknown, cancelled, and pending approval explicitly.

## 10. Test Matrix

| Layer | Case | Required assertion |
|---|---|---|
| Reducer | complete mutation with verified receipt | exact v1 state and stable source refs |
| Reducer | raw success without receipt | verification_status=not_observed, never verified |
| Reducer | pending/rejected/expired/consuming approval | review status follows approval store, no inferred execution |
| Reducer | cancel before side effect, cancel after unknown boundary | cancelled in first case, sticky unknown in second |
| Reducer | precommitted operation with durable pre-effect abandonment | execution_status=abandoned with the exact operation identity; known no-effect is preserved and no replay or success is inferred |
| Reducer | started operation loses its result before recovery | execution_status=unknown, never abandoned or not_started |
| Reducer | turn cancellation after a matched abandonment | call remains abandoned while turn may be cancelled; cancellation does not erase known no-effect evidence |
| Reducer | mismatched call/operation/digest or contradictory source | partial/blocked, never completed |
| Compatibility | current legacy raw events only | partial result without fabricated evidence |
| Compatibility | future aggregate/timeline/checkpoint schema | rejected and existing client state preserved |
| Outbox | retry same source state | one event ID/sequence through idempotency key |
| Outbox | append/replay an outbox event then reduce/rebuild/restart | identical idempotency key, existing event reused, and no self-triggered sequence loop |
| Outbox | one new authoritative source event | changed source refs create exactly one new aggregate sequence |
| Identity | session/turn IDs with colons or special characters | fixed-format opaque event ID; canonical identity remains unambiguous and leaks neither ID |
| Approval | lifecycle transitions with equal timestamps | distinct monotonic approval_revision values and source refs |
| Outbox | process failure between approval DB and session log | reconciler preserves approval_revision and repairs one missing observation/outbox event without tool replay |
| SSE | duplicate, out-of-order, collision, and gap | duplicate ignored; conflict/gap reloads durable range |
| API | non-stream, SSE final event, and reload | canonical aggregate payloads are equal |
| Engine | mutation precommit/result integration | deterministic operation ID is timeline descriptor ID |
| Engine | concurrent same-session turns | each turn has independent contiguous aggregate sequence |
| Engine | approved resume and double resolve | exactly one side effect; aggregate follows consumed result |
| Frontend | live event then reload | identical ToolCard/workflow state |
| Frontend | unsupported event and partial evidence | no success badge or destructive action is implied |

## 11. Rollout and Rollback

Phase 0: land the pure reducer, fixture corpus, strict reader, and no-write shadow comparison. No client consumes the output.

Phase 1: complete the Engine mutation timeline integration blocker and its end-to-end evidence tests. Do not enable aggregate publishing before this.

Phase 2: enable durable outbox writing behind tool_observability_v1, replay it to a server-only verifier, and measure source mismatch, duplicate, gap, and unsupported-version counters.

Phase 3: expose the snapshot/range API and SSE aggregate event behind the same flag. The web client now parses, repairs, and renders the aggregate projection.

實作進度（2026-07-25）：Phase 0、Phase 1、Phase 2、Phase 3 與 Phase 4 的
implementation 均已完成。sessions_dir 採 startup-only invariant；storage
identity、snapshot/range/SSE、frontend cursor/repair 與 aggregate UI projection
均已接線。flag/rollback rehearsal、migration/storage-scope rehearsal 與
isolated production build 均已驗收，不改變本節既有 rollback 規則。

Phase 4: switch ToolCallCard and the chat reasoning tool cards to the aggregate state. Keep the old per-event display only as a non-authoritative transcript fallback.

Rollback stops new aggregate publication and routes clients back to the legacy per-event display. It does not delete outbox records, change timeline/checkpoint semantics, re-enable legacy intent routing, or replay any side effect. Readers remain deployed so already-written v1 records stay inspectable. A rollback is blocked if it would cause the browser to report an unknown operation as a completed one.

## 12. Exit Criteria

P2.3 final-gate verification (2026-07-25) confirms the following:

- Engine mutation timeline integration is complete and uses deterministic timeline operation IDs.
- The reducer reconstructs the same aggregate from durable sources after a restart and rejects unsupported versions.
- Approval observation/outbox repair is idempotent and never replays a tool.
- SSE reconnect/gap handling and reload converge on the same aggregate state.
- Frontend renders aggregate, partial, abandoned known no-effect, unknown, cancellation, and approval states without local success inference.
- The current frontend gate contains 51 scripts; 48 non-browser scripts have clean
  exits, while three dev-server browser fixtures remain environment-limited by the
  local runner timeout. The backend rollout matrix passes with production flag-on
  and flag-off/rollback coverage.
