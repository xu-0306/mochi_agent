# Tool Activation Contract Stage 1 Handoff

## Background

Mochi is moving toward a durable tool activation contract:

```text
tool_search discovery -> policy-gated activation/promotion -> callable tool -> verified side effect
```

The current first-wave implementation improved metadata honesty:

- `tool_search` can now report `callable_this_turn`.
- `tool_search` can now report `activation_required`.
- hidden mutation tool calls can be surfaced as `tool_not_exposed`.
- ReAct can block some false final answers when a file mutation obligation is active.

However, code review found that Stage 1 is not yet correct enough to build Stage 2 activation/promotion on top of it. The main issue is that the runtime still mixes three separate concepts:

1. user intent says a file artifact must be created or changed;
2. planner decides which tools are callable this turn;
3. ReAct verifies whether a successful file mutation actually happened.

These must be separate states. If they stay coupled, the system can either expose write tools too broadly or fail to enforce a write obligation when the write tools are hidden.
## Full Staged Roadmap

This work should be implemented in stages. Do not skip ahead to dynamic activation until Stage 1 is correct, because activation would otherwise inherit the wrong authority boundary.

### Stage 0: Review And Baseline Capture

Goal:

- establish the current behavior and lock in failing tests for the known contract gaps.

Inputs:

- current first-wave implementation;
- review findings in this document;
- existing plan: `docs/superpowers/plans/2026-07-10-tool-activation-contract.md`.

Work:

- confirm current `tool_search` metadata behavior;
- confirm current exposure behavior for `workspace_read`, `open_world_lookup`, `tool_discovery`, and `workspace_write`;
- confirm current ReAct final-answer guard behavior;
- add failing regression tests for the unsafe cases before production edits.

Outputs:

- failing tests for read/web intents exposing write tools;
- failing test for write obligation surviving hidden write tools;
- failing or updated test for repeated hidden mutation calls.

Definition of done:

- the unsafe behavior is reproducible in tests;
- the tests distinguish discoverable tools from callable tools;
- no production behavior has been broadened to make tests pass.

### Stage 1: Safe Static Contract

Goal:

- make the existing non-dynamic tool system safe and honest.

Core invariant:

```text
write intent creates a write obligation;
planner may or may not expose write tools;
ReAct must verify an actual mutation before success.
```

Work:

- introduce first-class workspace write obligation in `engine.py`;
- pass the obligation to `AsyncReActLoop` independently of exposed tools;
- prevent non-write intents from making write tools callable;
- preserve write tools in `discoverable_tool_names` where useful;
- block false final answers when obligation is unsatisfied;
- prevent or short-circuit repeated identical hidden mutation calls.

Outputs:

- `workspace_read` and `open_world_lookup` do not expose write tools;
- `tool_discovery` can discover write tools but cannot call them;
- `workspace_write` keeps an obligation even if no write tool is callable;
- ReAct returns a structured blocker instead of a false success.

Definition of done:

- all Stage 1 regression tests pass;
- focused tool/exposure/ReAct tests pass;
- no dynamic activation exists yet;
- `tool_search` remains discovery-only.

### Stage 2: Controlled Activation Request Contract

Goal:

- add a safe request path from discovered-but-hidden tools to policy-gated activation.

Important boundary:

```text
tool_search may request activation;
tool_search must not directly activate tools.
```

Work:

- extend `tool_search` result payload with optional activation request metadata;
- add a runtime activation decision path outside `tool_search`;
- validate activation against routed intent, execution profile, tool mode, allowlist, denylist, approval policy, and workspace security;
- return a structured denial when activation is not allowed;
- regenerate callable tool schemas after successful activation if the current architecture supports mid-turn promotion.

Suggested payload:

```python
{
    "activation_required": True,
    "activation_request": {
        "tool_name": "file_write",
        "required_intent": "workspace_write",
        "policy_check": "required",
    },
}
```

Outputs:

- discovered hidden tools can produce explicit activation requests;
- denied activation gives clear metadata;
- approved activation makes the tool callable only through the runtime gate.

Definition of done:

- search results cannot bypass planner/policy;
- activation denial is observable and structured;
- activation success updates callable schemas safely;
- final mutation verification still applies after activation.

### Stage 3: Policy, Approval, And Security Hardening

Goal:

- prove activation cannot bypass existing safety controls.

Work:

- add tests for readonly/profile-blocked activation;
- add tests for allowlist exclusion;
- add tests for denylist blocking;
- add tests for approval-required file writes;
- verify `FileWriteTool`, `FileEditTool`, and `ApplyPatchTool` still enforce path and approval checks;
- verify denied calls cannot be replayed unchanged.

Outputs:

- policy-blocked activation returns structured blocker;
- approval-required execution still returns approval metadata;
- workspace path protections remain unchanged.

Definition of done:

- security and approval tests pass;
- no activation path writes files without going through existing file tools;
- no activation path bypasses `ToolExecutionContext` policy.

### Stage 4: End-To-End User Regression

Goal:

- cover the original reported failure and realistic multilingual requests.

Work:

- create an end-to-end scenario:
  1. user asks which tool can save files;
  2. model uses `tool_search`;
  3. user asks to save or update a workspace artifact;
  4. runtime routes the second turn as `workspace_write`;
  5. file is actually written or a structured blocker is returned.

- include English, Traditional Chinese, Simplified Chinese, and mixed-language variants;
- include discovery-only negatives;
- assert observable side effects, not just final text.

Positive examples:

```text
save the previous script as train.py
請把剛剛的程式寫入 train_med_vlm_lora.py
把 requirements.txt 建立在工作區
套用 patch 修正這個檔案
```

Negative examples:

```text
你有沒有保存檔案的工具？
請說明如何保存檔案
Update me on the latest weather
Write a summary of the attached PDF without saving files
```

Outputs:

- tests verify file existence or mutation event for positive cases;
- tests verify no write tool call for discovery-only or read-only cases.

Definition of done:

- reported workflow works end to end;
- positive write cases produce actual mutation or blocker;
- negative cases do not expose or call write tools.

### Stage 5: Diagnostics And Developer Documentation

Goal:

- make the contract understandable during debugging and future maintenance.

Work:

- document discoverable vs callable tool semantics;
- document write obligation semantics;
- document activation request and activation decision metadata;
- expose useful runtime diagnostics:

```python
"tool_activation": {
    "requested_tool": "file_write",
    "callable_this_turn": False,
    "activation_required": True,
    "reason": "workspace_write obligation missing or policy denied",
}
```

Outputs:

- developer note in `docs/`;
- event metadata useful enough to diagnose planner, policy, activation, and mutation guard failures.

Definition of done:

- next maintainer can tell whether failure came from routing, exposure, activation, policy, approval, or mutation execution;
- docs clearly say `tool_search` discovery is not authority.

### Stage 6: Cleanup And Compatibility

Goal:

- remove temporary behavior and align old tests with the new contract.

Work:

- update tests that still encode old behavior, especially read intents expecting write tools to be callable;
- remove duplicated keyword patches that became unnecessary after obligation routing;
- audit large diffs in `react_loop.py` for unrelated churn;
- ensure legacy aliases still normalize correctly;
- preserve backwards-compatible metadata where external API consumers may rely on it.

Outputs:

- smaller, clearer tool exposure logic;
- old ambiguous expectations replaced by explicit contract tests;
- no unnecessary dynamic behavior in Stage 1 code paths.

Definition of done:

- broad runtime/tool/API test suite passes;
- no known tests require write tools to be callable for read/web/discovery requests;
- code comments explain only non-obvious policy boundaries.

## Stage Dependency Summary

```text
Stage 0 -> Stage 1 -> Stage 2 -> Stage 3 -> Stage 4 -> Stage 5 -> Stage 6
```

Hard dependencies:

- Stage 2 must wait for Stage 1.
- Stage 3 must test both Stage 1 static exposure and Stage 2 activation.
- Stage 4 should run after Stage 2 if dynamic activation is included, but a static Stage 1 version should still have blocker regressions.
- Stage 5 can start after Stage 1 but should be finalized after Stage 3.
- Stage 6 should be last.

Recommended implementation split:

- Agent A: Stage 1 planner and engine obligation.
- Agent B: Stage 1 ReAct mutation guard and repeated-call prevention.
- Agent C: Stage 2 activation contract design and tests.
- Agent D: Stage 3 security and approval review.
- Main agent: integration, broad tests, and final code review.

## Current Problems

### P1: Read or Web Intents Can Still Expose Write Tools

`ToolExposurePlanner` currently computes `attachment_mutation_request` from routed write intent OR fallback mutation keywords.

Relevant area:

- `mochi/agents/tool_exposure.py`
- around `attachment_mutation_request`
- around `workspace_write_baseline`
- around final write-tool append logic

This means generic words like `update`, `fix`, `save`, or Chinese variants may cause `file_write`, `file_edit`, and `apply_patch` to become callable even when the classifier already gave a high-confidence non-write intent.

Observed bad behavior:

```text
routed_intent = workspace_read
message = inspect the repository
=> file_write/file_edit/apply_patch become callable
```

Observed bad behavior:

```text
routed_intent = open_world_lookup
message = Update me on the latest weather in Taipei
=> file_write/file_edit/apply_patch become callable
```

Correct behavior:

- `workspace_read` may include read/search tools.
- `open_world_lookup` may include web tools.
- write tools may remain discoverable through `tool_search`.
- write tools must not become callable unless a first-class write obligation exists and policy allows exposure.

### P1: Write Obligation Depends On Write Tools Already Being Exposed

`AgentEngine` currently passes `requires_file_mutation=True` only when:

```python
tool_intent_route.intent == "workspace_write"
and any(tool_name in {"file_write", "file_edit", "apply_patch"} for tool_name in exposure_plan.tool_names)
```

Relevant area:

- `mochi/agents/engine.py`
- around `requires_file_mutation=`

This is backwards. The obligation should be created from user intent, not from whether the planner successfully exposed a write tool.

Bad outcome:

```text
User asks to save a file.
Classifier routes workspace_write.
Policy/profile/planner hides file_write.
requires_file_mutation becomes False.
ReAct can produce a normal final answer without verifying a mutation.
```

Correct behavior:

- if the request requires a workspace write, `requires_file_mutation` must remain true;
- if no write tool is callable, ReAct must return a structured blocker;
- the model must not be allowed to claim that the artifact was saved.

### P2: Same Hidden Mutation Call Can Be Repeated

`AsyncReActLoop` currently detects repeated tool calls only when the same tool-call signature appears in consecutive tool-call rounds.

Relevant area:

- `mochi/agents/react_loop.py`
- around repeated tool-call signature tracking
- around unavailable tool result handling
- around file artifact follow-up prompts

The file artifact guard can insert a follow-up prompt between attempts. That resets the consecutive repeated-call guard, so the same hidden mutation call can happen again.

Current test even codifies this unsafe behavior:

```python
assert [event.tool_name for event in results] == ["file_write", "file_write"]
```

Correct behavior:

- an unavailable mutation call with the same tool name and arguments should be remembered for the whole turn;
- if it is `retryable=False`, the same exact hidden call should not consume another normal tool attempt;
- ReAct should either short-circuit to the existing blocker or nudge the model to replan with a different valid path.

## Design Principles For The Fix

### Keep These States Separate

Use separate variables or a small dataclass-like structure for:

```text
workspace_write_obligation.required
workspace_write_obligation.source
workspace_write_obligation.confidence
workspace_write_obligation.rationale
workspace_write_obligation.available_mutation_tools
workspace_write_obligation.blocker_reason
```

Do not infer the obligation from exposed tools.

Do not infer callable tools directly from fallback keywords when a routed intent is already known.

### Keyword Fallback Is Only Fallback

Fallback mutation keywords are acceptable only as recall hints when:

- no classifier route exists, or route is `ambiguous`;
- the session or request clearly references workspace files;
- the request has an explicit artifact/write target.

Fallback keywords must not override a high-confidence route such as:

- `open_world_lookup`
- `literature_research`
- `workspace_read`
- `tool_discovery`
- `execution_or_process`

### Discoverable Is Not Callable

`discoverable_tool_names` may include hidden write tools so `tool_search` can explain that such tools exist.

`tool_names` is the actual callable set for this turn.

Write tools may be discoverable without being callable.

### Stage 1 Does Not Need Dynamic Promotion

Do not implement dynamic activation/promotion in this stage.

Stage 1 should make the existing contract safe:

```text
write intent -> obligation exists -> callable write tool if allowed -> verified mutation
write intent + no callable write tool -> structured blocker
non-write intent -> write tools not callable
```

Stage 2 can later add:

```text
tool_search result -> activation request -> policy/profile/approval gate -> registry promotion -> regenerated schemas
```

## Step-By-Step Implementation Plan

### Step 1: Add Failing Regression Tests First

Add or update tests before editing production logic.

Suggested files:

- `tests/test_tool_exposure.py`
- `tests/test_tool_activation_contract.py`
- optionally `tests/test_engine_phase2.py`

Required tests:

1. `workspace_read` must not expose write tools:

```python
plan = planner.plan(
    message="inspect the repository",
    user_intent_message="inspect the repository",
    available_tool_names=["file_read", "glob_search", "grep_search", "file_write", "file_edit", "apply_patch"],
    backend=_FakeBackend(),
    session_bound_workspace=True,
    autonomy_mode="trusted_workspace",
    routed_intent="workspace_read",
    intent_confidence=0.95,
    intent_source="classifier",
)

assert not {"file_write", "file_edit", "apply_patch"} & set(plan.tool_names)
assert {"file_write", "file_edit", "apply_patch"} <= set(plan.discoverable_tool_names)
```

2. `open_world_lookup` with the word `Update` must not expose write tools:

```python
plan = planner.plan(
    message="Update me on the latest weather in Taipei",
    user_intent_message="Update me on the latest weather in Taipei",
    available_tool_names=["web_search", "web_fetch", "file_read", "file_write", "file_edit", "apply_patch"],
    backend=_FakeBackend(),
    session_bound_workspace=True,
    autonomy_mode="trusted_workspace",
    routed_intent="open_world_lookup",
    intent_confidence=0.95,
    intent_source="classifier",
)

assert {"web_search", "web_fetch"} & set(plan.tool_names)
assert not {"file_write", "file_edit", "apply_patch"} & set(plan.tool_names)
```

3. `workspace_write` must create an obligation even when no write tool is callable:

```python
assert requires_file_mutation is True
assert available_file_mutation_tools == []
assert final_answer.metadata["reason"] == "file_artifact_not_mutated"
```

4. repeated hidden mutation calls must not be accepted twice:

```python
assert [event.tool_name for event in results].count("file_write") <= 1
assert final_answer.metadata["reason"] == "file_artifact_not_mutated"
```

### Step 2: Introduce A First-Class Write Obligation

In `mochi/agents/engine.py`, create a small internal structure before constructing `AsyncReActLoop`.

Minimal version:

```python
workspace_write_obligation = {
    "required": tool_intent_route.intent == "workspace_write",
    "source": tool_intent_route.source,
    "confidence": tool_intent_route.confidence,
    "rationale": tool_intent_route.rationale,
}
```

Then pass:

```python
requires_file_mutation=workspace_write_obligation["required"]
```

Do not include `any(tool_name in exposure_plan.tool_names ...)` in the obligation condition.

Also include obligation diagnostics in events or runtime metadata if there is an existing diagnostics object nearby.

### Step 3: Fix Planner Write Exposure

In `mochi/agents/tool_exposure.py`, change `attachment_mutation_request`.

Current unsafe shape:

```python
attachment_mutation_request = routed_workspace_write or (
    not routed_tool_discovery
    and matches_mutation_keywords
)
```

Safer shape:

```python
route_allows_mutation_fallback = normalized_routed_intent in {None, "ambiguous"}
fallback_has_workspace_target = (
    attached_workspace_files
    or session_bound_workspace and explicit_workspace_file_reference
    or workspace_request and explicit_workspace_target
)

attachment_mutation_request = routed_workspace_write or (
    route_allows_mutation_fallback
    and fallback_has_workspace_target
    and matches_mutation_keywords
)
```

Do not let fallback mutation keywords activate write tools for:

- `open_world_lookup`
- `literature_research`
- `workspace_read`
- `tool_discovery`
- `execution_or_process`

Important: if helper variables for explicit workspace targets do not exist yet, add the smallest local helper needed. Avoid a broad new natural-language classifier.

### Step 4: Preserve Discoverability Of Write Tools

When write tools are not callable, still keep them in `discoverable_tool_names` where appropriate.

Expected behavior:

```text
tool_search can say file_write exists but is not callable_this_turn.
model cannot directly call file_write unless it is in tool_names.
```

Do not remove write tools from the searchable catalog just because they are not callable.

### Step 5: Fix Repeated Hidden Mutation Calls

In `mochi/agents/react_loop.py`, add turn-level memory for unavailable non-retryable tool calls.

Suggested state:

```python
terminal_unavailable_tool_signatures: set[str] = set()
```

Use the same signature concept as registry denied-call memory if available:

```python
signature = json.dumps(
    {"tool_name": tool_call.name, "arguments": tool_call.arguments},
    ensure_ascii=False,
    sort_keys=True,
    default=str,
)
```

When `_build_unavailable_tool_result(...)` returns a `ToolResult` with:

```python
retryable is False
```

and the requested tool is one of:

```python
{"file_write", "file_edit", "apply_patch"}
```

then record the signature.

If the same signature appears again in the same run:

- do not treat it as a fresh normal attempt;
- return or reuse a structured blocker;
- avoid adding duplicate `ToolCallResultEvent` rows unless the existing event model requires an event for every model-requested tool call.

If an event is emitted, metadata should make clear:

```python
"error_type": "repeated_unavailable_mutation_tool"
"recoverability": "requires_replanning_or_activation"
"retryable": False
```

### Step 6: Make Final-Answer Guard Independent Of Tool Exposure

Ensure ReAct blocks saved/completed claims whenever:

```python
requires_file_mutation is True
and no successful file mutation metadata was observed
```

Successful mutation should still be based on existing file operation metadata:

```python
metadata["file_changes"]
metadata["bytes_written"]
tool_result.output is not None
tool_result.error is None
```

Do not hardcode exact final phrases like only `"Saved report.md"`. The guard should block any normal final answer unless it clearly states the blocker or the mutation has succeeded.

### Step 7: Update Tests That Encode Old Behavior

Some existing tests currently expect unsafe behavior.

Known examples from review:

- workspace read tests expecting `file_write` in `plan.tool_names`;
- activation contract test expecting two hidden `file_write` results.

Update these tests to encode the new contract:

- read or web intent: write tools discoverable, not callable;
- repeated hidden write: one attempt or explicit repeated-call blocker;
- write obligation: true even when no write tools are callable.

### Step 8: Run Focused Verification

Run:

```bash
python -m py_compile tests/test_tool_activation_contract.py mochi/tools/tool_search.py mochi/tools/registry.py mochi/agents/tool_exposure.py mochi/agents/tool_intent_router.py mochi/agents/react_loop.py mochi/agents/engine.py
```

Run:

```bash
python -m pytest tests/test_tool_system_upgrade.py tests/test_tool_exposure.py tests/test_tool_activation_contract.py -q --basetemp .pytest-tmp/tool-activation-stage1
```

Run targeted engine tests:

```bash
python -m pytest tests/test_engine_phase2.py::test_react_loop_enforces_file_artifact_obligation_before_final_answer tests/test_engine_phase2.py::test_engine_passes_workspace_write_route_as_file_mutation_obligation -q --basetemp .pytest-tmp/tool-activation-engine
```

If time allows, run:

```bash
python -m pytest tests/test_engine_phase2.py tests/test_api_runtime.py tests/test_tool_system_upgrade.py tests/test_tool_exposure.py tests/test_tool_activation_contract.py -q
```

## Definition Of Done

Stage 1 is complete when all of these are true:

- `tool_search` reports callable state honestly.
- non-write intents do not expose write tools as callable.
- write tools remain discoverable where useful.
- `workspace_write` creates a write obligation independent of exposure.
- missing write tools produce a structured blocker, not a false success.
- successful final answers for artifact requests require an actual file mutation event.
- repeated identical hidden mutation calls are blocked or short-circuited.
- tests cover English and Chinese write intent examples, plus discovery-only negatives.

## Do Not Do In Stage 1

Do not implement direct search-result-to-call escalation.

Do not let `tool_search` mutate the active registry.

Do not allow fallback keyword matching to override a high-confidence non-write route.

Do not bypass execution profile, allowlist, denylist, approval policy, or workspace path checks.

Do not broaden the natural-language keyword list as the primary fix. The long-term design is obligation plus policy-gated activation, not phrase matching.

## Stage 2 Preview

After Stage 1 is safe, implement controlled activation/promotion:

1. `tool_search` returns an optional `activation_request` payload for hidden tools.
2. runtime validates intent, execution profile, allowlist, denylist, approval policy, and workspace security.
3. if approved, active registry view is promoted and callable schemas are regenerated.
4. if denied, the model receives a structured blocker.
5. final-answer mutation guard remains active even after promotion.

Stage 2 should be modeled after mature systems that treat discovery and authority as separate phases.
