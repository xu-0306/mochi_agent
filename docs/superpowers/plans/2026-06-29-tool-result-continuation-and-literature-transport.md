# Tool Result Continuation And Literature Transport Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `@superpowers:subagent-driven-development` (recommended) or `@superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix Mochi's broken oversized-tool-result continuation contract so models can keep reading persisted results without depending on `file_read`, while making literature-search tool reinjection compact, citation-first, and less likely to look "truncated" in reasoning output.

**Architecture:** Keep the current artifact-persistence design in `ToolResultTransportGuard` instead of switching to pure in-context truncation. Add a dedicated session-aware continuation tool for persisted tool results, teach literature tools to emit short evidence-first text instead of generic JSON previews, update transport messaging and planner exposure so the continuation tool is actually callable, and keep UI diagnostics additive rather than redesigning the event schema.

**Tech Stack:** Python 3.12, Mochi agent runtime, built-in tool registry, React/Next.js reasoning UI, `pytest`.

---

## Scope And Constraints

- Fix the broken continuation contract in the current Mochi design. Do not replace the whole transport system with OpenClaw/Zeroclaw-style head-tail truncation in this wave.
- Prefer additive changes that minimize merge conflict risk with the current dirty worktree, especially in:
  - [mochi/agents/engine.py](/H:/_python/agent_mochi/mochi/agents/engine.py)
  - [mochi/agents/react_loop.py](/H:/_python/agent_mochi/mochi/agents/react_loop.py)
- Do not remove existing `tool-result://` support from `file_read` in this wave. Keep backward compatibility while moving the model-facing contract to a dedicated tool.
- Do not bundle aggregate multi-result budgeting into this implementation. Record it as follow-up only.
- Preserve existing transport diagnostics keys (`summary_applied`, `overflow_persisted`, `reference_id`, `artifact_path`, `source_path`) so API serialization and session replay stay stable.
- Any new continuation tool must be available in the same session context that owns `tool_result_references`.

## Why This Plan Exists

Current Mochi truncation is happening in Mochi itself, not mainly in upstream literature APIs:

- Base formatter truncates/summarizes oversized results first:
  - [mochi/tools/base.py](/H:/_python/agent_mochi/mochi/tools/base.py:134)
- ReAct loop caps tool reinjection aggressively by default:
  - [mochi/agents/react_loop.py](/H:/_python/agent_mochi/mochi/agents/react_loop.py:121)
- Transport guard may then summarize again and persist overflow artifacts:
  - [mochi/tools/transport_guard.py](/H:/_python/agent_mochi/mochi/tools/transport_guard.py:41)
- The persisted overflow contract currently instructs the model to call:
  - `file_read(path="tool-result://...")`
  - [mochi/tools/transport_guard.py](/H:/_python/agent_mochi/mochi/tools/transport_guard.py:303)
- That contract is brittle because `file_read` may not be exposed in literature-only turns, so the model sees a reference it cannot actually continue reading.

Reference projects show two valid patterns:

- Persist + explicit continuation contract:
  - Hermes:
    - [reference/hermes-agent/tools/tool_result_storage.py](/H:/_python/agent_mochi/reference/hermes-agent/tools/tool_result_storage.py)
  - cc-haha:
    - [reference/cc-haha/src/utils/toolResultStorage.ts](/H:/_python/agent_mochi/reference/cc-haha/src/utils/toolResultStorage.ts)
    - [reference/cc-haha/src/utils/permissions/filesystem.ts](/H:/_python/agent_mochi/reference/cc-haha/src/utils/permissions/filesystem.ts:1166)
- Context-only truncation with no continuation contract:
  - OpenClaw:
    - [reference/openclaw/src/agents/pi-embedded-runner/tool-result-context-guard.ts](/H:/_python/agent_mochi/reference/openclaw/src/agents/pi-embedded-runner/tool-result-context-guard.ts)
    - [reference/openclaw/src/agents/pi-embedded-runner/tool-result-truncation.ts](/H:/_python/agent_mochi/reference/openclaw/src/agents/pi-embedded-runner/tool-result-truncation.ts)
  - Zeroclaw:
    - [reference/zeroclaw/crates/zeroclaw-runtime/src/agent/history.rs](/H:/_python/agent_mochi/reference/zeroclaw/crates/zeroclaw-runtime/src/agent/history.rs)

Mochi already chose the first pattern. This plan fixes that choice instead of replacing it.

## File Responsibility Map

**Create**

- `mochi/tools/tool_result_read.py`
  - Dedicated read-only continuation tool for persisted tool-result references.
  - Resolves `reference_id` against `ToolExecutionContext.tool_result_references`.
  - Reads from original source when available, otherwise artifact file.
  - Returns bounded chunks with the same `offset` / `limit` / `line_numbers` style as `file_read`.

**Modify**

- `mochi/tools/file_ops.py`
  - Keep `file_read` backward-compatible for `tool-result://...`.
  - Extract shared continuation helpers if doing so reduces duplication cleanly.
- `mochi/tools/registry_factory.py`
  - Register `tool_result_read` as a built-in workspace tool.
- `mochi/agents/tool_exposure.py`
  - Ensure `tool_result_read` is available whenever workspace read-only tools are retained.
  - Ensure open-world / literature-focused turns do not accidentally strip it when the current session already has persisted tool-result references available.
- `mochi/agents/engine.py`
  - Add `tool_result_read` to restricted read-only allowlists (`subagent_readonly`, `subagent_research`, `judge`, `verifier`, `controller_exec` evidence set).
  - Own the session-aware continuation exception because `ToolExecutionContext` is available here and `ToolExposurePlanner` is currently stateless.
  - Keep diagnostics additive.
- `mochi/tools/transport_guard.py`
  - Replace model-facing continuation instructions from `file_read(path="tool-result://...")` to `tool_result_read(reference_id=..., ...)`.
  - Preserve stored reference metadata schema.
- `mochi/tools/literature_search.py`
  - Add custom `format_result_for_model()` implementations for:
    - `ArxivSearchTool`
    - `SemanticScholarSearchTool`
    - `CrossrefSearchTool`
    - `PubMedSearchTool`
  - Emit short citation-first, evidence-first plain text summaries rather than generic JSON envelopes.
- `mochi/tools/base.py`
  - Only touch if needed to support literature formatter behavior cleanly without affecting unrelated tools.
- `mochi/agents/react_loop.py`
  - Keep production flow unchanged unless a small follow-up schema wiring fix is required.
  - Primary responsibility in this plan is integration-test coverage around literature-only continuation turns.
- `web/src/components/chat/ReasoningPanel.tsx`
  - Change display wording only:
    - `Summarized` -> `Context-safe preview`
    - `Overflow persisted` -> `Full result saved`
  - Do not rename serialized API fields.

**Tests**

- `tests/test_tool_result_transport_guard.py`
- `tests/test_tools_phase2.py`
- `tests/test_literature_tools.py`
- `tests/test_tool_exposure.py`
- `tests/test_engine_phase2.py`
- `tests/test_phase3_tool_simulator_integration.py`
- `tests/test_api_chat_attachments.py`

## Implementation Order

1. Lock the broken contract with tests.
2. Add `tool_result_read` and keep `file_read` backward-compatible.
3. Move transport instructions to the new tool.
4. Ensure the planner / engine actually expose the new tool in constrained profiles and in real follow-up tool schemas.
5. Improve literature formatter output so truncation is less likely and more interpretable.
6. Update UI wording only after backend diagnostics remain green.

## Task 1: Lock The Broken Continuation Contract With Failing Tests

**Files:**
- Modify: `tests/test_tool_result_transport_guard.py`
- Modify: `tests/test_tools_phase2.py`
- Modify: `tests/test_phase3_tool_simulator_integration.py`
- Modify: `tests/test_tool_exposure.py`
- Modify: `tests/test_engine_phase2.py`

- [ ] **Step 1: Add a failing transport-guard test for the new continuation instruction**

Expected content contract:

```python
assert f"Reference: {reference_id}" in outcome.content
assert f'tool_result_read(reference_id="{reference_id}", offset=1, limit=200, line_numbers=True)' in outcome.content
assert 'file_read(path="tool-result://' not in outcome.content
```

- [ ] **Step 2: Add failing unit tests for the new dedicated continuation tool contract**

Write tests that describe the target API before implementation:

```python
result = await tool.execute(
    reference_id="file_read-abc123",
    offset=2,
    limit=1,
    line_numbers=True,
    context=context,
)
assert result.output == "2: beta"
assert result.metadata["reference_id"] == "file_read-abc123"
```

Cover:
- read from original source if `source_path` still exists
- fall back to artifact if source is gone
- preserve original encoding when source is used
- use artifact encoding when artifact fallback is used
- error when execution context is missing
- error when reference id is unknown

- [ ] **Step 3: Add a failing integration test for literature-only continuation availability**

In `tests/test_phase3_tool_simulator_integration.py`, create or extend a fake literature-only backend flow so that:

1. A literature tool returns a large payload.
2. Transport persists overflow and returns a continuation reference.
3. The next tool call uses `tool_result_read(...)`.

Expected:
- the turn succeeds without needing `file_read`
- `tool_result_read` is exposed in the tool schemas seen by the backend for the follow-up step
- the assertion must inspect the actual follow-up tool schema list presented to the backend, not only internal allowlists

- [ ] **Step 4: Add failing planner/profile tests for `tool_result_read` allowlists**

Cover:
- workspace-bound read-only profile keeps `tool_result_read`
- research profile keeps `tool_result_read`
- judge/verifier keep `tool_result_read`
- risky profiles still do not gain write tools

Run:

```bash
pytest tests/test_tool_result_transport_guard.py tests/test_tools_phase2.py tests/test_tool_exposure.py tests/test_engine_phase2.py tests/test_phase3_tool_simulator_integration.py -k "tool_result_read or tool-result or continuation" -v
```

Expected: FAIL because `tool_result_read` does not exist and transport still references `file_read`.

- [ ] **Step 5: Commit the red tests only if this branch is isolated and intentionally red**

```bash
git add tests/test_tool_result_transport_guard.py tests/test_tools_phase2.py tests/test_tool_exposure.py tests/test_engine_phase2.py tests/test_phase3_tool_simulator_integration.py
git commit -m "test: lock tool result continuation contract"
```

If the branch is shared or current worktree is intentionally dirty, skip this commit and report that red-test commit was not created.

## Task 2: Implement Dedicated `tool_result_read` And Transport Contract

**Files:**
- Create: `mochi/tools/tool_result_read.py`
- Modify: `mochi/tools/file_ops.py`
- Modify: `mochi/tools/registry_factory.py`
- Modify: `mochi/tools/transport_guard.py`
- Test: `tests/test_tool_result_transport_guard.py`
- Test: `tests/test_tools_phase2.py`

- [ ] **Step 1: Implement the new read-only continuation tool**

Target shape:

```python
class ToolResultReadTool(BaseTool):
    @property
    def name(self) -> str:
        return "tool_result_read"
```

Parameters:

```python
{
    "type": "object",
    "properties": {
        "reference_id": {"type": "string"},
        "offset": {"type": "integer", "minimum": 1, "default": 1},
        "limit": {"type": "integer", "minimum": 1},
        "line_numbers": {"type": "boolean", "default": True},
        "encoding": {"type": "string"},
        "max_bytes": {"type": "integer", "minimum": 1},
    },
    "required": ["reference_id"],
    "additionalProperties": False,
}
```

Behavior:
- require `ToolExecutionContext`
- resolve `reference_id` from `context.tool_result_references`
- prefer `source_path` if it still exists and is a file
- otherwise use `artifact_path`
- reuse `file_read`-style chunk semantics and metadata shape where practical
- return the same metadata keys that existing `file_read` continuation tests depend on when they make sense (`reference_id`, `artifact_path`, `source_path`, `encoding`, line-range metadata), not a brand-new response shape

- [ ] **Step 2: Share continuation logic with `file_read` only if the extraction is small and obvious**

Good extraction:
- one helper to resolve reference metadata and choose continuation target
- one helper to read bounded chunks from a path

Do not perform a large refactor of `file_ops.py`.

- [ ] **Step 3: Register the tool in the built-in registry**

Add a workspace-scoped built-in spec in [mochi/tools/registry_factory.py](/H:/_python/agent_mochi/mochi/tools/registry_factory.py) and a factory method similar to `file_read`.

- [ ] **Step 4: Update transport guard message generation**

Replace:

```python
file_read(path="tool-result://...", offset=1, limit=200, line_numbers=True)
```

with:

```python
tool_result_read(reference_id="...", offset=1, limit=200, line_numbers=True)
```

Keep:
- `Reference: <id>`
- persisted reference metadata in `context.tool_result_references`
- diagnostics fields unchanged

- [ ] **Step 5: Run focused tests**

```bash
pytest tests/test_tool_result_transport_guard.py tests/test_tools_phase2.py -v
```

Expected: PASS

- [ ] **Step 6: Commit the minimal implementation**

```bash
git add mochi/tools/tool_result_read.py mochi/tools/file_ops.py mochi/tools/registry_factory.py mochi/tools/transport_guard.py tests/test_tool_result_transport_guard.py tests/test_tools_phase2.py
git commit -m "feat: add dedicated tool result continuation reader"
```

## Task 3: Expose `tool_result_read` In Planner And Restricted Profiles

**Files:**
- Modify: `mochi/agents/tool_exposure.py`
- Modify: `mochi/agents/engine.py`
- Test: `tests/test_tool_exposure.py`
- Test: `tests/test_engine_phase2.py`
- Test: `tests/test_phase3_tool_simulator_integration.py`

- [ ] **Step 1: Add `tool_result_read` to workspace read-only baselines and restricted profile allowlists**

Minimum allowlists to update:

```python
readonly_allowed
evidence_allowed
```

in [mochi/agents/engine.py](/H:/_python/agent_mochi/mochi/agents/engine.py), plus the workspace core read-only baseline in [mochi/agents/tool_exposure.py](/H:/_python/agent_mochi/mochi/agents/tool_exposure.py).

- [ ] **Step 2: Ensure open-world / literature-focused turns do not strip `tool_result_read` only when continuation is actually needed**

Constraint:
- `tool_result_read` should survive the same filters that preserve `file_read` for safe workspace reading.
- Do not let `open_world_focus_request` or similar filtering accidentally remove it when the current session already has one or more persisted tool-result references in `ToolExecutionContext.tool_result_references`.
- Do not broaden open-world exposure unnecessarily: if the current session has no persisted tool-result references, keep the existing open-world workspace-read filtering behavior unchanged.
- Preferred implementation: treat this as a targeted continuation exception, not a general reversal of open-world workspace-tool suppression.
- Implementation note: do not try to make `ToolExposurePlanner` stateful for this wave. Apply the session-aware continuation exception in `engine.py` after planning, because `AgentEngine` already has access to the per-session `ToolExecutionContext`.

- [ ] **Step 3: Keep the change narrow**

Do not redesign all tool exposure heuristics in this task.
Only guarantee that the continuation tool is reachable in the same categories where persisted tool results matter.

- [ ] **Step 4: Run focused tests**

```bash
pytest tests/test_tool_exposure.py tests/test_engine_phase2.py tests/test_phase3_tool_simulator_integration.py -k "tool_result_read or continuation or readonly" -v
```

Expected: PASS

- [ ] **Step 5: Verify actual follow-up tool schemas, not just internal allowlists**

In `tests/test_phase3_tool_simulator_integration.py`, assert that the backend sees `tool_result_read` in the follow-up turn tool schema list after a persisted overflow reference has been created.

In `tests/test_engine_phase2.py`, keep the narrower allowlist/profile assertions focused on exposure-plan results and restricted profiles.

Example expectation:

```python
assert "tool_result_read" in backend.tool_calls_seen[followup_index]
```

- [ ] **Step 6: Commit**

```bash
git add mochi/agents/tool_exposure.py mochi/agents/engine.py tests/test_tool_exposure.py tests/test_engine_phase2.py tests/test_phase3_tool_simulator_integration.py
git commit -m "fix: expose tool result continuation reader"
```

## Task 4: Make Literature Tool Reinjection Citation-First Instead Of Generic JSON

**Files:**
- Modify: `mochi/tools/literature_search.py`
- Modify: `mochi/tools/base.py` only if required
- Test: `tests/test_literature_tools.py`
- Test: `tests/test_phase3_tool_simulator_integration.py`

- [ ] **Step 1: Add failing formatter tests for each literature tool**

Expected model-facing format characteristics:
- plain text, not JSON envelope
- compact enough to fit typical `max_chars`
- starts with paper titles/authors/year/source identifiers, not `{"ok": true, ...}`
- preserves DOI / arXiv / PubMed / URL references when available

Example assertion style:

```python
rendered = tool.format_result_for_model(result, max_chars=500)
assert not rendered.lstrip().startswith("{")
assert "Attention Is All You Need" in rendered
assert "Ashish Vaswani" in rendered
assert "http://arxiv.org/abs/1706.03762v7" in rendered
```

- [ ] **Step 2: Implement per-tool or shared literature formatter helpers**

Good output shape:

```text
Top literature matches:
1. Attention Is All You Need (2017)
   Authors: Ashish Vaswani, Noam Shazeer
   Source: arXiv 1706.03762v7
   Abstract: We propose a new simple network architecture.
   URL: http://arxiv.org/abs/1706.03762v7
```

Rules:
- prefer 2 to 5 entries depending on budget
- truncate abstracts/summaries before dropping identifiers
- include stable identifiers first
- keep plain text deterministic

- [ ] **Step 3: Ensure the transport guard still treats these as safe plain text when under limit**

No new special-case transport path should be needed if formatter output is plain text and within `max_chars`.

- [ ] **Step 4: Run focused tests**

```bash
pytest tests/test_literature_tools.py tests/test_phase3_tool_simulator_integration.py -k "literature or arxiv or semantic or pubmed or crossref" -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add mochi/tools/literature_search.py mochi/tools/base.py tests/test_literature_tools.py tests/test_phase3_tool_simulator_integration.py
git commit -m "feat: add citation-first literature tool formatting"
```

## Task 5: Update Reasoning UI Wording Without Changing Diagnostics Schema

**Files:**
- Modify: `web/src/components/chat/ReasoningPanel.tsx`
- Test: `tests/test_api_chat_attachments.py`

- [ ] **Step 1: Keep API field names stable and only change labels**

Target label changes:
- `Summarized` -> `Context-safe preview`
- `Overflow persisted` -> `Full result saved`

Do not rename:
- `summaryApplied`
- `overflowPersisted`
- `referenceId`
- `artifactPath`

- [ ] **Step 2: Confirm existing API serialization test still passes or extend it only if labels are rendered elsewhere**

This task should not require backend API changes. `tests/test_api_chat_attachments.py` is the schema-stability guard for serialized diagnostics, not the primary label-rendering test.

If the repo already has a small frontend component test seam for `ReasoningPanel`, extend it to verify the new labels. If no such seam exists, do not invent a large frontend test harness in this task.

- [ ] **Step 3: Run focused tests**

```bash
pytest tests/test_api_chat_attachments.py -k "transport or diagnostics" -v
```

If frontend test harness exists locally, also run the smallest reasoning-panel related test target available. Do not invent a large frontend suite for this task.

- [ ] **Step 4: Commit**

```bash
git add web/src/components/chat/ReasoningPanel.tsx tests/test_api_chat_attachments.py
git commit -m "chore: clarify tool transport wording in reasoning panel"
```

## Task 6: Final Integration Verification

**Files:**
- No new production files unless a small wiring fix is required
- Verify all changed files from Tasks 1-5

- [ ] **Step 1: Run the full backend regression slice for this feature**

```bash
pytest tests/test_tool_result_transport_guard.py tests/test_tools_phase2.py tests/test_tool_exposure.py tests/test_engine_phase2.py tests/test_literature_tools.py tests/test_phase3_tool_simulator_integration.py tests/test_api_chat_attachments.py -v
```

- [ ] **Step 1b: Verify one end-to-end same-session continuation flow explicitly**

Confirm a same-session literature-oriented flow works end-to-end:

1. large literature result is persisted
2. transport returns a continuation reference
3. follow-up turn exposes `tool_result_read`
4. `tool_result_read` returns the next chunk
5. reasoning continues without falling back to `file_read`

- [ ] **Step 2: Manually inspect for merge-risk hot spots**

Review diffs carefully in:
- `mochi/agents/engine.py`
- `mochi/agents/react_loop.py`
- `mochi/agents/tool_exposure.py`

Check that the implementation did not overwrite unrelated existing edits in the dirty worktree.

- [ ] **Step 3: Run a targeted git diff summary**

```bash
git diff --stat
git diff -- mochi/agents/engine.py mochi/agents/react_loop.py mochi/agents/tool_exposure.py mochi/tools/transport_guard.py mochi/tools/literature_search.py mochi/tools/tool_result_read.py
```

- [ ] **Step 4: Commit any integration-only fixes**

```bash
git add mochi/agents/engine.py mochi/agents/react_loop.py mochi/agents/tool_exposure.py mochi/tools/transport_guard.py mochi/tools/literature_search.py mochi/tools/tool_result_read.py tests/test_tool_result_transport_guard.py tests/test_tools_phase2.py tests/test_tool_exposure.py tests/test_engine_phase2.py tests/test_literature_tools.py tests/test_phase3_tool_simulator_integration.py tests/test_api_chat_attachments.py web/src/components/chat/ReasoningPanel.tsx
git commit -m "fix: complete tool result continuation transport flow"
```

## Review Checklist For Subagent Controller

- Spec compliance review must verify:
  - transport message no longer instructs `file_read(path="tool-result://...")`
  - `tool_result_read` is registered and callable
  - restricted profiles still remain read-only
  - literature formatter output is plain text and identifier-first
  - UI wording changed without backend schema churn

- Code quality review must verify:
  - no unnecessary large refactor of `file_ops.py`
  - no unrelated changes were reverted in dirty files
  - new tool has one clear responsibility
  - duplicated continuation logic is minimal or intentionally shared
  - tests cover source-vs-artifact fallback behavior

## Follow-Up, Not In This Wave

- Aggregate per-turn tool-result budget enforcement inspired by Hermes.
- Smarter head-tail truncation fallback inspired by OpenClaw/Zeroclaw for non-continuable tools.
- Broader tool-exposure redesign beyond ensuring `tool_result_read` survives the relevant profiles.
