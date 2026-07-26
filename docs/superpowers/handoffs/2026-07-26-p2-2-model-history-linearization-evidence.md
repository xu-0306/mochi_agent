# P2.2 Same-Session Model-History Linearization Evidence

## Source binding

- Workspace: `H:\_python\agent_mochi`
- Base commit: `28f45e156e3b85d004e15942217c0c7a8ff578a2`
- Working-tree verification timestamp: `2026-07-26T17:49:23+08:00`
- Evidence type: dirty-working-tree acceptance evidence; relevant SHA-256 values
  are recorded below.
- Safety: no reset, checkout, clean, broad deletion, or unrelated formatter was
  run. Existing dirty WIP and test-output directories were preserved.
- Delivery: the existing `mochi/sessions/` ignore rule also hid new Python source
  files. `.gitignore` now keeps session JSONL/runtime data ignored while
  re-including only `mochi/sessions/*.py`, so `timeline_coordinator.py` and
  `turn_timeline.py` are visible to normal source-control review. No files were
  staged or committed.

## Closure decision

P2.2 full same-session ordinary-Chat model-history linearization is verified for
persisted `AgentEngine.chat()` turns.

The accepted contract is:

1. User message admission and durable FIFO turn identity are committed together.
2. A turn cannot prepare its model prompt until it owns the durable session lane.
3. Prompt history is read from a strict SessionStore snapshot matching the
   timeline revision after claim.
4. Terminal predecessor turns are materialized by durable turn `sequence`, while
   message order inside each turn is preserved. Physical JSONL append
   interleaving is not model-history order.
5. Completed, blocked, and unknown predecessor transcripts remain visible.
   Cancelled turns and admission-recovery orphans that never executed are
   excluded from successor model history.
6. Two engine instances using the same canonical sessions root still share one
   durable model lane. Lease heartbeat, not a filesystem lock held across model
   generation or tool execution, maintains ownership.
7. Pre-timeline legacy messages remain a prefix. Existing approval-continuation
   messages remain a deterministic compatibility tail. Approval resume continues
   to consume its exact P0.3 ReAct checkpoint and is not rebuilt as a fresh or
   replayable tool call.

This closure does not change the P2.3 aggregate reducer, approval store,
CapabilityPlan, TurnIntentContract, or sessions-root binding.

## Defects reproduced and fixed

### 1. Three pre-admitted turns followed physical append order

Before the fix, three admitted turns could produce a durable log shaped like:

```text
user1, user2, user3, assistant1, assistant2
```

The third model prompt therefore saw:

```text
user1, user2, assistant1, assistant2, user3
```

The red test failed because index 1 was `second request`, not the first
`fake reply`. The materializer now groups messages by durable predecessor turn
and emits the groups by FIFO `sequence`.

### 2. A cancelled queued request leaked into later prompts

A queued turn cancelled before claim had a durable user message but never
reached the model. The old materializer treated every terminal turn as history,
so a later prompt still included the cancelled request.

The accepted predecessor outcomes are now only `completed`, `blocked`, and
`unknown`, with recovery orphans excluded. The red successor-prompt assertion
failed before the change and passes after it.

### 3. Cancellation verification used an obsolete bare-engine harness

Four API cancellation tests constructed `AgentEngine` with `__new__()` and did
not supply the now-required durable SessionStore. A centralized test helper now
provides an isolated real SessionStore; production code did not receive a
test-only SessionStore fallback.

This verification also found an API response inconsistency during durable
cleanup: `run_state` could already be `cancelled` while `cancel_outcome` remained
`pending`. The response now uses the authoritative post-cancellation terminal
snapshot when it has settled and clears a stale pending reason.

## Verification commands and results

The groups below overlap intentionally and must not be added into one whole-repo
test count.

### Engine and durable timeline

```powershell
rtk proxy python -m pytest -q tests/unit/engine/test_timeline_chat_integration.py tests/unit/engine/test_turn_contract_rollout.py --basetemp .tmp-p22-final-engine-20260726
```

Result: `24 passed`.

```powershell
rtk proxy python -m pytest -q tests/unit/sessions tests/test_session_store.py --basetemp .tmp-p22-final-sessions-20260726
```

Result: `90 passed`.

These matrices include the three-preclaimed-turn FIFO prompt, cross-engine lane,
queued/running cancellation, lease recovery, strict CAS, mutation lifecycle,
legacy reader, and compatibility-history cases.

The final hash-bound focused rerun used the current files after documentation
and lint-only cleanup:

```powershell
rtk proxy python -m pytest -q tests/unit/engine/test_timeline_chat_integration.py::test_three_preclaimed_turns_materialize_history_in_fifo_turn_order tests/unit/engine/test_timeline_chat_integration.py::test_cross_engine_same_session_claims_one_durable_model_lane tests/unit/engine/test_timeline_chat_integration.py::test_cancelled_queued_chat_never_reaches_model_and_terminalizes tests/unit/sessions/test_timeline_coordinator.py::test_linearized_history_keeps_legacy_prefix_and_compatibility_tail tests/integration/api/chat/test_cancellation.py --basetemp .tmp-p22-final-hash-bound-20260726
```

Result: `12 passed`.

### Approval, ReAct, rehydration, and state consumers

```powershell
rtk proxy python -m pytest -q tests/unit/engine/test_react_loop.py tests/security/test_approval_lifecycle.py tests/security/test_timeline_approval_continuation.py tests/integration/api/runtime/test_approval_routes.py tests/integration/api/runtime/test_exec_approval_rehydration.py --basetemp .tmp-p22-final-approval-react-20260726
```

Result: `109 passed`.

```powershell
rtk proxy python -m pytest -q tests/unit/agents/test_conversation_state_store.py tests/unit/agents/test_turn_contract_rollout.py --basetemp .tmp-p22-final-conversation-20260726
```

Result: `14 passed`.

```powershell
rtk proxy python -m pytest -q tests/integration/api/sessions/test_session_routes.py tests/integration/api/sessions/test_settings_routes.py tests/integration/api/runtime/test_approval_routes.py --basetemp .tmp-p22-final-api-consumers-20260726
```

Result: `66 passed`.

### API chat and cancellation

```powershell
rtk proxy python -m pytest -q tests/integration/api/chat/test_cancellation.py --basetemp .tmp-p22-api-cancellation-final2-20260726
```

Result: `8 passed`.

```powershell
rtk proxy python -m pytest -q tests/integration/api/chat/test_chat_routes.py tests/integration/api/chat/test_cancellation.py tests/integration/api/chat/test_streaming_and_serialization.py --basetemp .tmp-p22-verify-api-chat-final-20260726
```

Result: `20 passed`.

### Static checks

```powershell
rtk proxy python -m compileall -q mochi
rtk git diff --check
```

Result: both passed.

```powershell
rtk proxy rg -n "tool_intent_router|routed_intent|legacy_routed_intent|fallback_keyword" mochi --glob "*.py"
```

Result: no production matches (`rg` exit 1 is the expected no-match result).

```powershell
rtk proxy uv run --offline ruff check mochi/sessions/timeline_coordinator.py tests/unit/engine/test_timeline_chat_integration.py tests/unit/sessions/test_timeline_coordinator.py tests/integration/api/chat/_support.py tests/integration/api/chat/test_cancellation.py
```

Result: all checks passed.

```powershell
rtk proxy uv run --offline ruff check --select E4,E7,E9,F63,F7,F82 mochi/agents/engine.py mochi/sessions/timeline_coordinator.py tests/unit/engine/test_timeline_chat_integration.py tests/unit/sessions/test_timeline_coordinator.py tests/integration/api/chat/_support.py tests/integration/api/chat/test_cancellation.py
```

Result: all selected runtime-error checks passed.

A full-file Ruff scan of the already heavily modified `engine.py` was not
recorded as a pass: it reports six out-of-scope current-working-tree import/SIM
items (`I001` x2, `F401`, `SIM114`, `SIM103` x2), outside the P2.2 changed hunks.
No formatter or unrelated cleanup was applied to hide that existing debt.

The only pytest warning was the existing Windows `WinError 5` inability to write
`.pytest_cache`; all listed pytest processes exited cleanly with code 0.

## Relevant file SHA-256

```text
.gitignore SHA256=B219BD3B03A7597100B65BFCF3D92C842F3B0FC884A4F17E0F178AB8EF9C5274
mochi/agents/engine.py SHA256=1465080432396C10CB7E543889D15B0BE47B69201156C055160EFB79E07603F9
mochi/agents/invocation.py SHA256=D020E4DDFCC00B87AAEFAB4CF3E55DC52127171BD01A60CBCA4EF62F8D855D82
mochi/sessions/timeline_coordinator.py SHA256=DF6C4874FDF3D5F8B21E43409B173F2BD955F75156FBF932935843D83DC1AEFE
mochi/sessions/turn_timeline.py SHA256=B43BAA6EDFBC9DB19886F45DF273DDF8C75A1FFA1D0D3439C8E324D629A2C5B8
mochi/sessions/store.py SHA256=51E416ACB47208CBA2B802361F64ACB60C720F79DEC35355514B2DA6FD939CFB
tests/unit/engine/test_timeline_chat_integration.py SHA256=8AB11EAE340AFD41FD24E6811AA49AC7D658EC4452FB718D3E03B04FEAAFCA09
tests/unit/sessions/test_timeline_coordinator.py SHA256=744A31BE9A11B3A121D94896327C689177EE97A80832526F82A80A1C60EB1B49
tests/integration/api/chat/_support.py SHA256=642E3CCD369E2E82A318EE38811C50CC235653E0F3578E6E52A519E2122B44F1
tests/integration/api/chat/test_cancellation.py SHA256=96B70B78A6459D3228872696BDC8A4FD995E99A772EF19FE06B1EA08B44F7A3F
documents/architecture/2026-07-23-agent-tool-workflow-p0-p2-plan.md SHA256=D3C0F1A6776D01B9156173D41D4071D74D19C5EBA63B7A64D16760AB35F87738
documents/architecture/2026-07-25-tool-workflow-aggregate-stream-replay-rfc.md SHA256=9BF86F860E0073717BD017CA0E773C93A7F2A3E48BCCA047F4EE4265F2876560
docs/superpowers/handoffs/2026-07-25-agent-tool-workflow-scope-completion-handoff.md SHA256=0179CF99FC3191896099159BD6F72C1643D8FA1FA5686B8C2435687D18DC5C95
docs/superpowers/handoffs/2026-07-26-agent-tool-workflow-handoff-review-findings.md SHA256=ED3E97B0A33DE4AC7C6BB35D1B1D2B320BB8C324DEB9C85BEA87828B6EBCEC01
docs/superpowers/handoffs/2026-07-26-agent-tool-workflow-final-gate-evidence-manifest.md SHA256=7B9C85E4C4AAE20C74DD7E34DDDF141D44F6E7A063946A8F6CAFBF1FFDBCA132
```

## Final status

P2.2 may be marked complete for the scoped ordinary-Chat workflow. There is no
new P2.2 product blocker or model-history regression in the verified working
tree.
