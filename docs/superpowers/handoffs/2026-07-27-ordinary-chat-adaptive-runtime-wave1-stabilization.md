# Ordinary-Chat Adaptive Runtime Wave 1 Stabilization Handoff

**Date:** 2026-07-27

**Status:** Implemented and committed

**Base commit:** `ee420155c617f3e32e9ff759a0f4d684a233d78f`

**Stabilization commit:** `4ce3ccd0`

**Commit message:** `fix: stabilize ordinary chat adaptive runtime wave 1`

## Outcome

Wave 1 is integrated into the ordinary Chat path and stabilized for its current
scope. It remains an automatically no-op/activate runtime inside
`AgentEngine`; it does not add a Goal, Workflow, Team, Plan mode, CLI flag, or
special Chat entrypoint.

## Corrected Review Findings

1. JIT lexical retrieval now maps bounded English inflections such as
   `saving files` to `file_write` without turning arbitrary zero-score tools
   into matches.
2. `discovered_cache_size=0` produces a valid empty bounded discovery state
   instead of raising and being swallowed by the discovery hook.
3. Semantic criteria use a real bounded backend adapter:
   - no tools;
   - temperature zero;
   - configured token, timeout, evidence-character, and criterion budgets;
   - exact result schema;
   - host-recognized evidence references only;
   - timeout, malformed output, unavailable backend, invented evidence, or
     budget exhaustion becomes `unverified`.
4. Verification criterion IDs include their owner and target identity.
   Artifact criteria select target-specific evidence inside a multi-target
   receipt, so one target cannot inherit another target's verdict.
5. Parent and component kill switches are effective. Disabling retrieval
   removes `tool_search`, deferred discovery, and `tool_activate`; disabling
   verification prevents plan compilation and finalization verification.
6. Routine approval for one bounded `FileWriteTool` no longer forces a plan.
   Destructive operations, multiple approval-bearing effects, and unknown
   side-effect boundaries remain fail-closed planning signals.
7. ActiveTask completion now requires a completed PlanLedger. Incomplete,
   blocked, cancelled, or structurally inconsistent ledgers keep the task open.
8. Aggregate verification receipts use a strict, versioned, CAS-protected,
   idempotent repository and are persisted before PlanLedger or ActiveTask
   completion.
9. A required aggregate `failed` or `unverified` verdict now returns a
   completion error and emits/persists an authoritative
   `verification_blocked` final.

## Changed Files in `4ce3ccd0`

- `mochi/agents/engine.py`
- `mochi/agents/outcome_verifier.py`
- `mochi/agents/tool_discovery_state.py`
- `mochi/tools/tool_catalog_index.py`
- `tests/unit/agents/test_outcome_verifier.py`
- `tests/unit/agents/test_tool_discovery_state.py`
- `tests/unit/engine/test_adaptive_runtime_wave1.py`
- `tests/unit/tools/test_tool_catalog_index.py`

No unrelated `.gitignore` or generated/user artifact was included in the
commit.

## Verification Evidence

- Wave 1 and related regression group after final review: `281 passed`.
- Final retrieval-disable plus target-specific evidence boundary: `3 passed`.
- `python -m compileall` passed for the modified runtime and test modules.
- `git diff --check` passed before commit.
- Ruff was not installed in the available environment.

The subagent also ran the full suite:

```text
2246 passed, 13 failed, 3 skipped
```

The 13 failures were in legacy MultiAgent, Goal/API, Discord, Voice, and older
Engine fixtures outside the stabilization paths. This is useful compatibility
evidence, but it is not a repository-wide green gate.

## Runtime Invariants to Preserve

- Ordinary Chat remains the only entrypoint for this adaptive behavior.
- Complexity decisions consume validated contracts and host state, not raw
  multilingual keyword lists.
- A PlanLedger records progress; it never grants capability or call authority.
- Discovery, exposure, activation, authorization, approval, execution, and
  verification remain separate boundaries.
- Deterministic verifier failure dominates semantic judgement.
- Missing or malformed verification never becomes success.
- Aggregate receipt persistence precedes PlanLedger and ActiveTask completion.
- Approval continuation resumes the exact approved result and never replays
  the side effect.

## Known Limitation

The post-verification final is authoritative, but the model-authored success
event may already have been delivered to a live streaming consumer immediately
before `verification_blocked` is emitted. The final API result and durable
session state use the blocked result.

Before frontend/replay rollout, choose and test one explicit contract:

1. buffer model final events until verification completes; or
2. mark the pre-verification final as provisional and ensure every consumer
   replaces it with the authoritative final.

Do not let the frontend infer authority from event arrival order alone.

## Next Boundary

Wave 2 may begin from the stabilized P/T/V contracts:

- Recovery policy package R;
- Background failure-learning packages L;
- Adversarial/integration package A.

Root must continue to own Engine/ReAct hot spots, event/checkpoint schemas,
configuration, lifecycle, shared reducers, and final integration. Phase 8
frontend work must wait for the streaming-final contract and replay fixtures to
be frozen.
