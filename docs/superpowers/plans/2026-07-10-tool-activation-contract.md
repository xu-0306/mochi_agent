# Tool Activation Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a durable discoverable-vs-callable tool contract so tools found through `tool_search` can be safely activated or clearly reported as unavailable, and write-artifact requests cannot fail silently or falsely claim completion.

**Architecture:** Keep `tool_search` as discovery, but add explicit activation/promotion semantics controlled by intent, policy, and runtime capability. Introduce a workspace-write obligation that survives routing/exposure details, then require successful file mutation before final answers can claim artifact creation.

**Tech Stack:** Python 3.11+, Mochi `AgentEngine`, `ToolExposurePlanner`, `ToolIntentRouter`, `AsyncReActLoop`, `ToolRegistry`, pytest.

---

## Design Principles

- Treat this as an open-world intent/runtime contract problem, not a keyword-list bug.
- `tool_search` may reveal capabilities, but only policy-approved tools become callable.
- The model must see whether a searched tool is `callable_this_turn`, `activation_required`, or blocked by policy.
- Artifact/write requests create an obligation. The run must satisfy that obligation with a successful mutation event or report a blocker.
- Keyword fallback may be cleaned up, including mojibake strings, but fallback cannot be the source of truth.

## File Structure

- Modify: `mochi/agents/tool_intent_router.py`
  - Keeps classifier-first intent routing.
  - Adds or clarifies artifact/write obligation extraction helpers without making keywords authoritative.
- Modify: `mochi/agents/tool_exposure.py`
  - Produces active callable tools and discoverable catalog metadata consistently.
  - Preserves write tools as callable when a write obligation is active.
- Modify: `mochi/tools/tool_search.py`
  - Adds per-result availability fields such as `callable_this_turn`, `activation_required`, and `activation_reason`.
- Modify: `mochi/tools/registry.py`
  - Passes scoped callable/catalog context into `ToolSearchTool`.
  - Optionally supports a controlled activation request result shape without directly bypassing planner policy.
- Modify: `mochi/agents/engine.py`
  - Carries write obligation metadata into exposure planning and ReAct loop.
  - Distinguishes "write required but no write tool callable" from ordinary no-tool turns.
- Modify: `mochi/agents/react_loop.py`
  - Handles `tool_not_exposed` for mutation tools as an exposure-contract failure when write obligation is active.
  - Verifies file mutation before final success.
- Modify: `mochi/tools/file_ops.py`
  - Only if extra metadata is needed for mutation success tracking; avoid changing core write semantics unless tests prove a gap.
- Test: `tests/test_tool_exposure.py`
  - Covers write obligation, callable vs discoverable catalog, and negative cases.
- Test: `tests/test_tool_system_upgrade.py`
  - Covers `tool_search` payload metadata and registry scoping.
- Test: `tests/test_engine_phase2.py` or a focused new test file such as `tests/test_tool_activation_contract.py`
  - Covers ReAct/runtime behavior across search, activation, unavailable tool calls, and final-answer guard.

## Implementation Strategy

Use two stages.

Stage 1 is the conservative contract fix:

- Make `tool_search` honest about whether a result is callable this turn.
- Ensure write obligations expose write tools when policy allows.
- Block false completion when writes do not happen.

Stage 2 is the long-term activation design:

- Add a controlled activation/promotion path where search results can request tool activation.
- Activation must still pass intent, profile, policy, and approval checks.
- Activation failure must return a structured blocker rather than letting the model call a hidden tool.

Do not make search results directly callable without activation.

---

### Task 1: Add Regression Tests For Discoverable vs Callable

**Files:**
- Modify: `tests/test_tool_system_upgrade.py`
- Modify: `tests/test_tool_exposure.py`

- [ ] **Step 1: Write failing test for `tool_search` availability metadata**

Add a test where the callable registry view contains `tool_search`, `file_read`, and read/search tools, while the search catalog also contains `file_write`.

Expected result:

```python
assert match["name"] == "file_write"
assert match["callable_this_turn"] is False
assert match["activation_required"] is True
assert "not exposed" in match["activation_reason"].lower()
```

- [ ] **Step 2: Run the focused test and verify failure**

Run:

```bash
python -m pytest tests/test_tool_system_upgrade.py::test_tool_search_marks_discoverable_but_not_callable_tools -q
```

Expected: FAIL because `ToolSearchTool` currently returns no callable/activation fields.

- [ ] **Step 3: Write failing exposure test for write obligation**

Add a planner test using an explicit routed intent, not a hardcoded natural-language phrase:

```python
plan = planner.plan(
    message="produce an artifact",
    user_intent_message="produce an artifact",
    available_tool_names=["tool_search", "file_read", "file_write", "file_edit", "apply_patch"],
    backend=_FakeBackend(),
    session_bound_workspace=True,
    autonomy_mode="trusted_workspace",
    routed_intent="workspace_write",
    intent_confidence=0.93,
    intent_source="classifier",
    intent_rationale="The user asked for a workspace artifact to be saved.",
)

assert {"file_write", "file_edit", "apply_patch"} <= set(plan.tool_names)
```

- [ ] **Step 4: Add negative exposure test**

Add a test for a discovery-only request:

```python
plan = planner.plan(
    message="Do you have a tool for saving files?",
    user_intent_message="Do you have a tool for saving files?",
    available_tool_names=["tool_search", "file_read", "file_write"],
    backend=_FakeBackend(),
    session_bound_workspace=True,
    autonomy_mode="trusted_workspace",
    routed_intent="tool_discovery",
    intent_confidence=0.9,
    intent_source="classifier",
)

assert "tool_search" in plan.tool_names
assert "file_write" not in plan.tool_names
assert "file_write" in plan.discoverable_tool_names
```

- [ ] **Step 5: Commit tests**

```bash
git add tests/test_tool_system_upgrade.py tests/test_tool_exposure.py
git commit -m "test: capture tool discovery activation contract"
```

---

### Task 2: Make `tool_search` Report Callable State

**Files:**
- Modify: `mochi/tools/tool_search.py`
- Modify: `mochi/tools/registry.py`
- Test: `tests/test_tool_system_upgrade.py`

- [ ] **Step 1: Extend `ToolSearchTool` constructor**

Add optional callable-name provider:

```python
CallableToolNameProvider = Callable[[], set[str]]

def __init__(
    self,
    *,
    catalog_provider: ToolCatalogProvider,
    callable_name_provider: CallableToolNameProvider | None = None,
    default_top_k: int = 5,
    max_top_k: int = 50,
) -> None:
    self._catalog_provider = catalog_provider
    self._callable_name_provider = callable_name_provider
```

- [ ] **Step 2: Extend result payload**

In `_tool_payload`, include availability fields. If a tool is not callable:

```python
payload["callable_this_turn"] = tool.name in callable_names
payload["activation_required"] = tool.name not in callable_names
payload["activation_reason"] = (
    "Tool is discoverable but not exposed as callable in this turn."
    if tool.name not in callable_names
    else None
)
```

Keep this as metadata only; do not auto-enable the tool here.

- [ ] **Step 3: Scope `ToolSearchTool` correctly in `ToolRegistry.create_view`**

When building a view, pass both:

- `tool_names` as callable set.
- `tool_search_catalog_names` as searchable catalog set.

Pseudo-code:

```python
callable_names = set(tool_names)
tool.scoped_to_catalog(
    catalog_provider=...,
    callable_name_provider=lambda: set(callable_names),
)
```

- [ ] **Step 4: Run focused tests**

```bash
python -m pytest tests/test_tool_system_upgrade.py::test_tool_search_marks_discoverable_but_not_callable_tools -q
```

Expected: PASS.

- [ ] **Step 5: Run related registry tests**

```bash
python -m pytest tests/test_tool_system_upgrade.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add mochi/tools/tool_search.py mochi/tools/registry.py tests/test_tool_system_upgrade.py
git commit -m "feat: mark searched tools as callable or activation-required"
```

---

### Task 3: Add Workspace Write Obligation As First-Class Runtime State

**Files:**
- Modify: `mochi/agents/tool_intent_router.py`
- Modify: `mochi/agents/engine.py`
- Modify: `mochi/agents/tool_exposure.py`
- Test: `tests/test_tool_exposure.py`

- [ ] **Step 1: Define the invariant**

Invariant:

```text
If the latest user request requires creating, saving, editing, or patching a workspace artifact,
the runtime carries a write obligation independent of keyword fallback.
```

- [ ] **Step 2: Use existing routed intent as the first obligation source**

Start conservatively:

```python
requires_workspace_write = tool_intent_route.intent == "workspace_write"
```

Do not build a new broad keyword classifier in `engine.py`.

- [ ] **Step 3: Add an internal metadata object**

In `engine.py`, create a small dict or dataclass-like payload:

```python
workspace_write_obligation = {
    "required": tool_intent_route.intent == "workspace_write",
    "source": tool_intent_route.source,
    "confidence": tool_intent_route.confidence,
    "rationale": tool_intent_route.rationale,
}
```

Pass this into diagnostics and ReAct loop construction.

- [ ] **Step 4: Ensure exposure honors the obligation**

In `ToolExposurePlanner`, keep `workspace_write_baseline` behavior, but make tests verify it is driven by routed intent and cannot be lost due to normal limit trimming.

Do not make `tool_discovery` expose write tools.

- [ ] **Step 5: Clean mojibake keywords only as fallback hygiene**

Replace mojibake in `_ATTACHMENT_MUTATION_INTENT_KEYWORDS` with valid UTF-8 terms, but leave a comment that this is fallback recall only.

Use examples like:

```python
"寫入",
"写入",
"存檔",
"存档",
"保存",
"另存為",
"另存为",
"建立檔案",
"建立文件",
"修改",
"更新",
"套用 patch",
"修正這個檔案",
```

- [ ] **Step 6: Run exposure tests**

```bash
python -m pytest tests/test_tool_exposure.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add mochi/agents/tool_intent_router.py mochi/agents/tool_exposure.py mochi/agents/engine.py tests/test_tool_exposure.py
git commit -m "feat: carry workspace write obligation through tool exposure"
```

---

### Task 4: Add Controlled Activation Request Contract

**Files:**
- Modify: `mochi/tools/tool_search.py`
- Modify: `mochi/agents/react_loop.py`
- Modify: `mochi/agents/engine.py`
- Test: `tests/test_tool_activation_contract.py`

- [ ] **Step 1: Create focused test file**

Create `tests/test_tool_activation_contract.py`.

Cover two paths:

- Search finds hidden `file_write`, but result says it requires activation.
- A write obligation causes the planner to expose `file_write` directly, avoiding hidden-tool failure.

- [ ] **Step 2: Decide activation scope**

Implement the conservative activation contract first:

```text
tool_search does not activate tools directly.
It returns activation_required and a reason.
The model/runtime must replan or continue with a structured blocker.
```

Do not add dynamic mid-turn mutation of the callable registry yet unless the current ReAct loop already supports safe regeneration of schemas.

- [ ] **Step 3: Add optional activation request shape**

If adding activation request support, use a structured payload instead of direct tool promotion:

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

- [ ] **Step 4: Keep activation policy-gated**

Activation must fail closed when:

- `tool_mode == "disabled"`.
- Execution profile is readonly/evidence/judge.
- Tool allowlist excludes write tools.
- Tool denylist includes write tools.
- Security policy requires approval and no approval path exists.

- [ ] **Step 5: Run activation contract tests**

```bash
python -m pytest tests/test_tool_activation_contract.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add mochi/tools/tool_search.py mochi/agents/react_loop.py mochi/agents/engine.py tests/test_tool_activation_contract.py
git commit -m "feat: add controlled tool activation contract"
```

---

### Task 5: Harden ReAct Handling For Hidden Mutation Tools

**Files:**
- Modify: `mochi/agents/react_loop.py`
- Modify: `mochi/agents/engine.py`
- Test: `tests/test_tool_activation_contract.py`
- Test: `tests/test_engine_phase2.py`

- [ ] **Step 1: Write failing ReAct test for hidden mutation call**

Simulate a model that calls `file_write` when the callable registry does not include it.

Expected result:

```python
assert "tool_not_exposed" in event.metadata["guard"]
assert not final_answer_claims_saved(result.content)
```

Use existing ReAct fixtures if available; avoid broad mocking that bypasses the real registry view.

- [ ] **Step 2: Treat hidden mutation tool as contract failure**

In `react_loop.py`, when unavailable tool guard sees:

```python
tool_name in {"file_write", "file_edit", "apply_patch"}
```

and write obligation is active, mark metadata:

```python
{
    "runtime_category": "tool_activation",
    "error_type": "mutation_tool_not_callable",
    "recoverability": "requires_replanning_or_activation",
}
```

- [ ] **Step 3: Prevent repeated identical hidden calls**

Reuse existing denied-call memory patterns where possible. The model should not loop on the same unavailable mutation call.

- [ ] **Step 4: Add final-answer guard**

If write obligation is active and no successful mutation event occurred, final response must be a blocker message, not normal completion.

Expected message shape:

```text
I could not save the file because the required write tool was not callable in this turn.
```

- [ ] **Step 5: Run focused tests**

```bash
python -m pytest tests/test_tool_activation_contract.py tests/test_engine_phase2.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add mochi/agents/react_loop.py mochi/agents/engine.py tests/test_tool_activation_contract.py tests/test_engine_phase2.py
git commit -m "fix: block false completion for hidden write tools"
```

---

### Task 6: Preserve Security And Approval Semantics

**Files:**
- Modify: `mochi/agents/tool_exposure.py`
- Modify: `mochi/agents/engine.py`
- Modify: `mochi/runtime/service.py`
- Test: `tests/test_security_policy.py`
- Test: `tests/test_api_runtime.py`
- Test: `tests/test_tool_activation_contract.py`

- [ ] **Step 1: Write tests for policy-blocked activation**

Cases:

- strict/readonly profile blocks activation.
- allowlist excludes write tools.
- denylist includes `file_write`.
- approval-required policy returns approval metadata rather than silent success.

- [ ] **Step 2: Ensure activation cannot bypass approval**

Activation may make a tool callable only if policy allows it. If approval is required, tool execution must still return approval metadata or use existing approved-call replay.

- [ ] **Step 3: Ensure workspace scope still applies**

Do not change `check_file_tool_path` behavior. Any activation path still runs through `FileWriteTool`, `FileEditTool`, or `ApplyPatchTool`.

- [ ] **Step 4: Run security tests**

```bash
python -m pytest tests/test_security_policy.py tests/test_api_runtime.py tests/test_tool_activation_contract.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mochi/agents/tool_exposure.py mochi/agents/engine.py mochi/runtime/service.py tests/test_security_policy.py tests/test_api_runtime.py tests/test_tool_activation_contract.py
git commit -m "test: preserve policy gates for tool activation"
```

---

### Task 7: Add End-To-End Regression For The Reported Failure

**Files:**
- Create or modify: `tests/test_tool_activation_contract.py`
- Optional fixture: use the existing fake backend/ReAct fixtures from `tests/test_engine_phase2.py`

- [ ] **Step 1: Write reported-case regression**

Scenario:

1. User asks a discovery-style question and model searches for saving tools.
2. User then asks to save generated code as a file.
3. Runtime must route second request as workspace write.
4. `file_write` must be callable, or a structured activation blocker must be returned.

- [ ] **Step 2: Add multilingual variants**

Use variants that cannot pass by only matching one phrase:

```python
[
    "請把上一段程式存成 train_med_vlm_lora.py",
    "幫我產生 requirements.txt 並保存",
    "save the previous script as train.py",
    "把剛剛的內容另存為 infer_med_vlm_lora.py",
]
```

Also include a negative:

```python
"你有沒有保存檔案的工具？"
```

- [ ] **Step 3: Verify observable outcome**

For positive write cases:

```python
assert any(event.tool_name == "file_write" and event.status == "success" for event in events)
```

or assert the file exists in the temp workspace.

For negative discovery case:

```python
assert not any(event.tool_name == "file_write" for event in events)
```

- [ ] **Step 4: Run regression tests**

```bash
python -m pytest tests/test_tool_activation_contract.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_tool_activation_contract.py
git commit -m "test: cover tool search to write activation regression"
```

---

### Task 8: Documentation And Diagnostics

**Files:**
- Modify: `docs/` location that currently documents tool exposure, if present.
- Modify: `mochi/agents/tool_exposure.py`
- Modify: `mochi/tools/tool_search.py`
- Test: diagnostics assertions in `tests/test_tool_activation_contract.py`

- [ ] **Step 1: Add developer note**

Document:

- discoverable tools are not necessarily callable.
- activation requires policy approval.
- write obligations must be verified by mutation event.

- [ ] **Step 2: Improve diagnostics**

Expose in runtime metadata:

```python
"tool_activation": {
    "requested_tool": "file_write",
    "callable_this_turn": False,
    "activation_required": True,
    "reason": "workspace_write obligation missing or policy denied"
}
```

- [ ] **Step 3: Run doc-adjacent tests**

```bash
python -m pytest tests/test_tool_activation_contract.py tests/test_tool_exposure.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add docs mochi/agents/tool_exposure.py mochi/tools/tool_search.py tests/test_tool_activation_contract.py
git commit -m "docs: explain discoverable and callable tool contract"
```

---

## Verification Matrix

Run focused tests first:

```bash
python -m pytest tests/test_tool_system_upgrade.py tests/test_tool_exposure.py tests/test_tool_activation_contract.py -q
```

Then run broader runtime/security tests:

```bash
python -m pytest tests/test_engine_phase2.py tests/test_api_runtime.py tests/test_security_policy.py -q
```

If time allows, run the broader tool/runtime suite:

```bash
python -m pytest tests/test_tool_system_upgrade.py tests/test_tool_exposure.py tests/test_engine_phase2.py tests/test_api_runtime.py tests/test_security_policy.py -q
```

Expected result: all selected tests pass. Any test that fails due to intentional contract changes must be updated only after confirming it encoded the old ambiguous behavior.

## Rollout Notes

- Start with metadata honesty (`callable_this_turn`) before dynamic activation. This reduces model confusion without expanding authority.
- Add activation/promotion only after callable/discoverable semantics are visible and tested.
- Do not let search results bypass `ToolExposurePlanner`, execution profile filters, allowlist/denylist, approval policy, or workspace path checks.
- Keep final-answer guard in place even after activation exists. The artifact must actually be written.

## Anti-Hardcode Check

- Classification: open set.
- Chosen abstraction: obligation + capability activation contract + final-state verifier.
- Anti-hardcode tests: multilingual positive cases, discovery-only negative case, direct injected `workspace_write` route that does not depend on one phrase.
- Remaining hardcodes: official tool names such as `file_write`, `file_edit`, and `apply_patch` are acceptable because they are closed-set runtime tool IDs.
