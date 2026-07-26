# Agent Tool Workflow P0-P2 Continuation Handoff

Date: 2026-07-24
Workspace: `H:\_python\agent_mochi`
Branch: `main`
Primary plan: `documents/architecture/2026-07-23-agent-tool-workflow-p0-p2-plan.md`

## Purpose

This handoff freezes the current shared worktree after the latest-message intent
pipeline was removed and three follow-up implementation agents were interrupted.
It separates verified work from partial work so the next agent can continue
without restoring legacy behavior or assuming unfinished code is production-ready.

The worktree is intentionally dirty and contains substantial user WIP unrelated
to this slice. Do not reset, checkout, clean, or mechanically revert broad paths.
At handoff time `git status --short` reported about 168 entries, including local
temporary directories with permission warnings.

## Product Decision and Hard Invariants

The production semantic authority is:

```text
bounded conversation + summary + durable active task
-> TurnIntentContract
-> CapabilityPlan
-> policy-bounded exposure / activation
-> concrete call authorization
-> execution
-> artifact verification
-> deliverable completion
```

Do not regress these decisions:

- `TurnIntentContract` and `CapabilityPlan` are the only semantic authorities for
  exposure, activation eligibility, artifact obligation, and durable task state.
- The latest-message `ToolIntentRouter`, routed-intent fallback, and keyword
  exposure planner are deleted. Do not recreate them under a different name.
- Resolver/planner/adapter failure is fail-closed.
- Activation only makes a schema callable. It never authorizes concrete arguments.
- Auto Review and manual approval operate on a concrete call after activation.
- Approval cannot bypass hard policy, workspace scope, sandbox, policy drift, or
  target drift checks.
- A successful tool result is not sufficient evidence that an artifact or
  deliverable is complete.
- The old persisted `legacy` and `shadow` config values may only be read by the
  one-way migration reader and immediately normalized to `enforce`. They are not
  runtime modes.

## Stable, Verified Baseline Before the Interrupted Follow-Up

The following work was completed and verified before P0/P2 follow-up agents began:

- Deleted `mochi/agents/tool_intent_router.py`.
- Replaced the production exposure path with the message-free
  `ToolExposurePlanner.plan_contract_baseline()`.
- Removed `routed_intent`, `intent_route`, classifier invocation, route-union
  policy projection, and keyword-routing tests.
- Added deterministic catalog priorities for tools covering the same capability.
- Automatic latest-message skill matching no longer changes capability exposure
  priority; only explicitly selected skills may provide tool preferences.
- Added a metamorphic regression: two different messages resolving to the same
  contract and hard ceilings produce identical exposure and activation-eligible
  plans.
- Enforce-only config/API/frontend behavior and one-way old-value migration.
- Contract-authoritative activation; missing contract fields fail closed.
- Real `tool_search -> tool_activate -> schema refresh -> concrete call` flow.
- Current-turn required deliverables cannot arrive already marked `satisfied`.

Last verified baseline results:

```text
contract/capability/exposure/activation: 140 passed
engine:                                  46 passed
config/session:                          105 passed
frontend type-check:                     passed
Python compileall:                       passed
git diff --check:                        passed
isolated mypy, 8 cutover modules:        passed
```

Production/test search for `tool_intent_router`, `routed_intent`, and
`fallback_keyword` returned no matches.

## Why the Original User Flow Failed

The original conversation record is:

- `D:\_download\mochi-chat (2).md`

The old classifier inspected only the latest message. It interpreted
"B, general version" as ambiguous even though prior turns defined B as creating
workspace files. Mutation schemas were therefore not exposed. Later
`tool_search` found `file_write`, but no callable activation broker existed in
that recorded run. Auto Review was never reached because there was no concrete
file tool call to review.

The image referenced by the original analysis is:

- `C:\Users\Xu\AppData\Local\Temp\codex-clipboard-4d257419-bcdd-45a4-8340-c777acdedda9.png`

## Interrupted Follow-Up Work

Three agents started follow-up slices. Their final turns failed with external
HTTP 403 quota errors. Partial edits remain in the shared worktree and must be
reviewed, completed, and tested; none should be treated as committed or complete.

### A. P0.2 Exec Per-Call Effective Policy - Partial

Touched files:

- `mochi/security/policy.py`
- `mochi/security/__init__.py`
- `mochi/tools/exec_command.py`
- `mochi/tools/execute_code.py`
- `mochi/tools/execute_code_v2.py`
- `tests/test_execute_code_and_mcp.py`

Implemented or partially implemented:

- Parse an effective policy snapshot from call context.
- Deterministic hard-deny selector matching by tool/capability.
- Read `require_approval_for_exec` per call instead of trusting a cached tool
  constructor value.
- Include policy snapshot/version metadata in allow, approval, deny, Auto Review,
  sandbox failure, and success paths.
- Added a cached `ExecuteCodeTool` test that exercises allow, approval, and hard
  deny with three different context snapshots.

Independent verification at handoff:

```text
python -m py_compile for the modified P0 modules: passed
tests/test_execute_code_and_mcp.py + tests/test_security_policy.py:
  31 passed, 1 failed
```

The failure is:

```text
test_execute_code_v2_default_runner_can_call_tool_helpers
expected "hello from helper\n"
received "hello from helper\r\n"
```

Do not blindly change the assertion. Determine whether the new execution path
changed newline normalization or whether this is a Windows-only pre-existing
fixture behavior. Verify raw bytes and the helper transport boundary first.

Still missing:

- Focused coverage for cached `ExecCommandTool` and `ExecuteCodeV2Tool` across
  different call snapshots.
- No-side-effect proof for every hard-deny path.
- Full security/exec regression suite.
- P0.3 ordinary Chat durable approval/resume was not implemented by this agent.

### B. P0.3 Durable Approval/Resume - Not Started

The repository already has persistent approval infrastructure in:

- `mochi/runtime/approval_lifecycle.py`
- `mochi/api/routes/approvals.py`
- existing Goal/agent-run approval integrations

The next implementation must reuse that infrastructure instead of creating a
second approval database. The minimum ordinary Chat closure must persist:

- normalized tool name and arguments
- arguments/request/context digest
- operation ID / idempotency key
- policy snapshot/version
- inventory version
- resolved target/workspace identity
- approval status and consume lease
- resume cursor / exact original call

On resume, revalidate policy, sandbox, workspace, target/base digest, and
inventory. Execute the original call exactly once. Rejection, expiry, drift, and
unknown/executing recovery must have explicit states.

### C. P2.1 Artifact Verifier - Module Draft Only

New untracked file:

- `mochi/agents/artifact_verifier.py`

The draft currently defines:

- `ArtifactExpectation`
- `ArtifactTargetReceipt`
- versioned `ArtifactReceipt`
- `ArtifactVerificationResult`
- `ArtifactVerifier.verify()`
- deterministic `artifact-op-v1` operation IDs
- workspace-bound target resolution
- existence, content, after-digest, deletion, and acceptance checks
- aggregate execution/verification/retry/recovery fields

Independent verification:

```text
python -m py_compile mochi/agents/artifact_verifier.py: passed
focused unit tests: none exist
engine integration: not implemented
persistence/checkpoint integration: not implemented
```

Required next tests before engine integration:

1. successful `file_write` with exact content and digest
2. tool claims success but target is missing
3. digest/content mismatch
4. path escaping the workspace
5. delete verification
6. `apply_patch` multi-file target extraction
7. partial multi-target result and recovery plan
8. failed/approval-required tool result retry disposition
9. deterministic operation ID and changed normalized arguments
10. malformed or targetless mutation events

After those tests pass, integrate the verifier into `AgentEngine` after mutation
events and before task completion. The current weak completion method is:

- `mochi/agents/engine.py:3603` - `_complete_turn_contract_task_if_satisfied`

Do not mark a required deliverable `satisfied` unless the corresponding receipt
is verified. Persist the receipt/checkpoint before updating durable task state.

### D. P2.2 Durable Turn State and Concurrency - Not Started

Current risk remains:

- prompt context is prepared before the per-session conversation-state lock
- the lock currently guards resolver/load/persist, not the complete turn state
  transition
- parallel turns can observe stale history/active-task state
- execution, approval, verification, and completion do not share one durable
  turn checkpoint

Design the lock/checkpoint boundary carefully. Do not hold a broad session lock
across slow model generation or external tools without defining cancellation and
deadlock behavior. Prefer a versioned compare-and-swap transition or short
critical sections around durable state revisions.

### E. P2.3 Workflow Observability - Partial, Not Type-Clean

New or modified files:

- `mochi/api/tool_workflow_observability.py`
- `tests/test_tool_workflow_observability.py`
- `mochi/api/routes/chat.py`
- `web/src/lib/tool-workflow-observability.ts`
- `web/src/lib/api.ts`
- `web/src/app/page.tsx`

Intended projection:

- effective server policy and expectation match/stale state
- `policy_catalog`, eligible, and exposed tools
- activation state
- concrete call review/Auto Review
- execution status
- verification status
- absent evidence is `not_observed`, never guessed

Important semantic limitation:

- Current planner diagnostics expose the policy-bounded catalog, not the complete
  installed catalog. Keep the API field explicitly scoped as `policy_catalog` or
  include `catalog_scope="policy_eligible"`; do not label it as the full catalog.

Independent verification at handoff:

```text
tests/test_tool_workflow_observability.py: 2 passed
Python compile for API projection: passed
frontend type-check: failed with 4 errors
```

Current TypeScript errors:

```text
page.tsx:1458 expected_policy_version is missing from the request type
page.tsx:1482 expectedPolicyVersion is missing from the request type
page.tsx:4127 expectedPolicyVersion is not defined in scope
page.tsx:4195 expectedPolicyVersion is not defined in scope
```

The interrupted agent reported frontend tests/typecheck/lint passing before its
last edits, but the independent rerun above is authoritative.

Streaming/replay still relies on existing per-event `tool_exposure` and tool
result metadata. There is no independently persisted aggregate SSE/replay event.
Do not duplicate events merely for symmetry; either document the per-event
projection contract or add a stable aggregate event with an event ID and dedupe
semantics.

## Recommended Continuation Order

1. Freeze and inspect the dirty worktree.
   - Read this handoff and the primary plan completely.
   - Run `rtk git diff --check`.
   - Do not reset unrelated changes.

2. Stabilize P0.2 before adding more features.
   - Fix or classify the newline failure.
   - Add cached-tool per-call tests for all three exec tools.
   - Run security, exec, sandbox, approval-binding, and detached-process tests.

3. Stabilize the observability slice.
   - Fix the four TypeScript errors.
   - Verify non-stream Chat and context preview API compatibility.
   - Verify actual UI rendering, not only normalizer unit tests.
   - Record the stream/replay aggregate limitation.

4. Complete Artifact Verifier in isolation.
   - Add the ten focused test groups above.
   - Review path/symlink/TOCTOU behavior and memory limits for large artifacts.
   - Only then integrate into engine completion.

5. Implement ordinary Chat durable approval/resume.
   - Reuse `PersistentApprovalStore` and consume leases.
   - Bind exact call, policy, inventory, workspace, and operation ID.

6. Add durable turn checkpoint and concurrency control.
   - Persist contract, capability plan, approval, execution receipt,
     verification receipt, and completion transition.

7. Run full cross-cutting verification and update the primary plan.

## Suggested Opening Prompt for the Next Agent

```text
Continue the Mochi Agent tool-workflow P0-P2 implementation from the frozen
shared worktree. First read these files completely:

1. docs/superpowers/handoffs/2026-07-24-agent-tool-workflow-p0-p2-continuation-handoff.md
2. documents/architecture/2026-07-23-agent-tool-workflow-p0-p2-plan.md
3. AGENTS.md

Preserve all unrelated dirty-worktree changes. Do not reset or restore deleted
legacy intent code. Begin by reproducing the documented P0 newline failure and
the four frontend TypeScript errors. Stabilize those partial slices before
integrating mochi/agents/artifact_verifier.py. Use TurnIntentContract and
CapabilityPlan as the only semantic authorities. Run every shell command through
rtk and update the handoff/primary plan with verified results, not assumptions.
```

## Verification Commands

All repository shell commands must be prefixed with `rtk` per `AGENTS.md`.

Focused P0:

```powershell
rtk proxy python -m pytest -q tests/test_execute_code_and_mcp.py tests/test_security_policy.py
```

Observability:

```powershell
rtk proxy python -m pytest -q tests/test_tool_workflow_observability.py
cd web
rtk npm run type-check
rtk npm run lint
```

Legacy-removal/core baseline:

```powershell
rtk proxy python -m pytest -q tests/unit/tool_exposure tests/unit/agents tests/test_tool_activation_contract.py tests/test_tool_system_upgrade.py
rtk proxy python -m pytest -q tests/unit/engine/test_turn_contract_rollout.py tests/unit/engine/test_preflight_and_backend.py tests/unit/engine/test_tool_exposure_and_invocation.py
rtk proxy python -m pytest -q tests/test_config.py tests/integration/api/sessions/test_session_routes.py tests/integration/api/sessions/test_settings_routes.py tests/integration/api/chat/test_session_permission_policy.py tests/test_api_chat_context.py
```

Static checks:

```powershell
rtk proxy python -m compileall -q mochi
rtk proxy python -m mypy --follow-imports=skip --ignore-missing-imports mochi/agents/turn_intent_contract.py mochi/agents/conversation_resolver.py mochi/agents/model_conversation_interpreter.py mochi/agents/capability_planner.py mochi/agents/turn_contract_rollout.py mochi/agents/capability_exposure_adapter.py mochi/agents/tool_exposure.py mochi/tools/tool_activate.py
rtk git diff --check
rtk rg -n "tool_intent_router|routed_intent|legacy_routed_intent|_tool_intent_router|_route_tool_intent_for_exposure|fallback_keyword" mochi web/src tests
```

The final `rg` should return no matches.

## Reference Material

Read these before changing architecture:

- Primary plan:
  `documents/architecture/2026-07-23-agent-tool-workflow-p0-p2-plan.md`
- This handoff:
  `docs/superpowers/handoffs/2026-07-24-agent-tool-workflow-p0-p2-continuation-handoff.md`
- Original conversation record:
  `D:\_download\mochi-chat (2).md`
- Reference implementations:
  - `reference/openclaw`
  - `reference/hermes-agent`
  - `reference/zeroclaw`
  - relevant cc-haha reference material under `reference/`
- Older activation contract/plans:
  - `docs/superpowers/plans/2026-07-10-tool-activation-contract.md`
  - `docs/superpowers/plans/2026-07-11-tool-activation-stage1-handoff.md`
  - `docs/tool-activation-contract.md`

## Known Tooling and Environment Notes

- Windows workspace-write sandbox is active.
- Prefer `apply_patch`; follow the documented PowerShell fallback only if
  `apply_patch` fails with the known split-root enforcement error.
- Pytest may warn that `.pytest_cache` cannot be created due local permissions.
- `git status` may warn about inaccessible old temporary test directories.
- Full transitive mypy currently surfaces unrelated existing errors in config,
  security decision typing, and memory store. Use the isolated command above for
  the cutover modules, while still tracking those broader errors separately.
- The prior subagents stopped because their external model provider returned HTTP
  403 quota errors. Their partial filesystem edits survived.

## Definition of the Next Safe Checkpoint

Do not mark P0/P2 complete until all of the following are true:

- Exec tools use call-scoped effective policy and hard denies with no side effects.
- Ordinary Chat approval is durable, exact-call bound, drift-safe, restart-safe,
  and exactly-once.
- Mutation completion requires a persisted verified artifact receipt.
- Same-session concurrent turns cannot overwrite contract/approval/verification
  state.
- UI/API distinguish policy catalog, eligible, exposed, activated, reviewed,
  executed, and verified states without inferring absent evidence.
- Focused and baseline suites pass, frontend type-check/lint pass, compile/static
  checks pass, and legacy intent symbols remain absent.
