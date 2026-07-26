# Tool Exposure And File UX Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove brittle keyword-gated workspace tool exposure, restore reliable file-inspection behavior for workspace tasks, and replace transcript-breaking large tool-result handling with a chunked/artifact-backed flow.

**Architecture:** Keep a small always-on workspace read tool baseline, move heuristic keyword routing to legacy-only ranking, and make file inspection explicitly staged: discover/list first, then targeted chunked reads. Preserve safety through workspace scope and approvals, not by hiding core read tools from the model. Replace preview-only large result reinjection with summary plus resolvable references and explicit follow-up read paths.

**Tech Stack:** Python 3.11, FastAPI, Mochi agent runtime, existing `ToolRegistry` / `ToolExposurePlanner`, existing `file_read` offset/limit support, existing `tool_search`, existing temp artifact persistence, pytest.

---

## References

### Official / External
- Anthropic Agent SDK tool search:
  - [https://code.claude.com/docs/en/agent-sdk/tool-search](https://code.claude.com/docs/en/agent-sdk/tool-search)
- Anthropic Agent SDK permissions:
  - [https://code.claude.com/docs/en/agent-sdk/permissions](https://code.claude.com/docs/en/agent-sdk/permissions)
- OpenAI MCP guide:
  - [https://developers.openai.com/api/docs/mcp](https://developers.openai.com/api/docs/mcp)
- OpenAI MCP/connectors guide:
  - [https://developers.openai.com/api/docs/guides/tools-connectors-mcp](https://developers.openai.com/api/docs/guides/tools-connectors-mcp)
- MCP roots spec:
  - [https://modelcontextprotocol.io/specification/2025-06-18/client/roots](https://modelcontextprotocol.io/specification/2025-06-18/client/roots)
- MCP resources spec:
  - [https://modelcontextprotocol.io/specification/2025-06-18/server/resources](https://modelcontextprotocol.io/specification/2025-06-18/server/resources)
- Aider repository map:
  - [https://aider.chat/docs/repomap.html](https://aider.chat/docs/repomap.html)

### Internal Review Inputs
- Tool exposure planner:
  - [mochi/agents/tool_exposure.py](/H:/_python/agent_mochi/mochi/agents/tool_exposure.py)
- Engine planner message and attachment context:
  - [mochi/agents/engine.py](/H:/_python/agent_mochi/mochi/agents/engine.py)
- File read tool:
  - [mochi/tools/file_ops.py](/H:/_python/agent_mochi/mochi/tools/file_ops.py)
- Tool result formatting:
  - [mochi/tools/base.py](/H:/_python/agent_mochi/mochi/tools/base.py)
- Tool result transport guard:
  - [mochi/tools/transport_guard.py](/H:/_python/agent_mochi/mochi/tools/transport_guard.py)
- React loop transport callsite:
  - [mochi/agents/react_loop.py](/H:/_python/agent_mochi/mochi/agents/react_loop.py)
- Filesystem API:
  - [mochi/api/routes/filesystem.py](/H:/_python/agent_mochi/mochi/api/routes/filesystem.py)
- Current tests:
  - [tests/test_tool_exposure.py](/H:/_python/agent_mochi/tests/test_tool_exposure.py)
  - [tests/test_engine_phase2.py](/H:/_python/agent_mochi/tests/test_engine_phase2.py)
  - [tests/test_tool_result_transport_guard.py](/H:/_python/agent_mochi/tests/test_tool_result_transport_guard.py)
  - [tests/test_tools_phase2.py](/H:/_python/agent_mochi/tests/test_tools_phase2.py)
  - [tests/test_phase3_tool_simulator_integration.py](/H:/_python/agent_mochi/tests/test_phase3_tool_simulator_integration.py)

### Reference Implementation To Borrow From
- `cc-haha` workspace/file UX:
  - [reference/cc-haha/src/server/services/workspaceService.ts](/H:/_python/agent_mochi/reference/cc-haha/src/server/services/workspaceService.ts)
  - [reference/cc-haha/src/server/api/sessions.ts](/H:/_python/agent_mochi/reference/cc-haha/src/server/api/sessions.ts)
  - [reference/cc-haha/src/utils/mcpOutputStorage.ts](/H:/_python/agent_mochi/reference/cc-haha/src/utils/mcpOutputStorage.ts)
  - [reference/cc-haha/desktop/src/components/chat/ToolCallBlock.tsx](/H:/_python/agent_mochi/reference/cc-haha/desktop/src/components/chat/ToolCallBlock.tsx)
  - [reference/cc-haha/desktop/src/components/workspace/WorkspacePanel.tsx](/H:/_python/agent_mochi/reference/cc-haha/desktop/src/components/workspace/WorkspacePanel.tsx)

## Non-Goals

- Do not add a new large multi-backend abstraction for tools.
- Do not replace the whole approval/security model in this wave.
- Do not ship a brand-new repo indexer in the first patch set.
- Do not keep the English keyword gate and merely add Chinese synonyms; the goal is to deprecate gating, not localize a bad mechanism.

## Target Behavior

1. If the chat is bound to a project/workspace, the model always sees the core read-only workspace tools:
   - `file_read`
   - `glob_search`
   - `grep_search`
   - `csv_read`
   - specialized read tools when available (`pdf_read`, `docx_read`, `notebook_read`)
2. Structured attachments and workspace selections reliably bias toward the correct reader without relying on a fragile marker string contract.
3. `file_read` remains a real file-inspection tool:
   - small reads can round-trip in full
   - large reads must degrade into explicit chunked reading, not transcript-only previews
4. Oversized tool results always return a resolvable artifact/reference plus a follow-up read path.
5. Tool exposure heuristics become ranking hints only; they do not decide whether core workspace read tools exist.
6. Users can inspect which workspace tools were exposed and why a result was truncated/persisted.

## Subagent Execution Split

Use fresh worker subagents with disjoint ownership:

- **Worker A: Exposure / Attachment Routing**
  - Owns:
    - `mochi/agents/tool_exposure.py`
    - `mochi/agents/engine.py`
    - `tests/test_tool_exposure.py`
    - `tests/test_engine_phase2.py`
- **Worker B: File Read / Transport / Artifact Follow-up**
  - Owns:
    - `mochi/tools/base.py`
    - `mochi/tools/file_ops.py`
    - `mochi/tools/transport_guard.py`
    - `mochi/agents/react_loop.py`
    - `tests/test_tool_result_transport_guard.py`
    - `tests/test_tools_phase2.py`
    - `tests/test_phase3_tool_simulator_integration.py`
- **Worker C: Workspace Observability / API / UI Surfacing**
  - Owns:
    - `mochi/api/routes/filesystem.py`
    - additive diagnostics/API files needed for tool exposure visibility, excluding `mochi/agents/engine.py`
    - relevant WebGUI files that surface tool exposure / truncation state
- **Worker D: Follow-up Architecture / Dynamic Discovery**
  - Owns design notes, implementation notes, and review support for the dynamic-discovery wave
  - Must not claim write ownership over files already assigned to Worker A in the first implementation cut
  - Worker A executes the Task 4 code changes after Tasks 1-3 stabilize

Implement Tasks 1-3 before starting Task 4. Task 5 can begin once Task 1 is merged and Task 2 interfaces are stable.

## File Responsibility Map

- `mochi/agents/tool_exposure.py`
  - Stop using keyword heuristics as the source of truth for workspace read tool existence.
  - Convert legacy keyword routing into ranking-only metadata.
- `mochi/agents/engine.py`
  - Build planner input from structured attachments without relying on mismatched plain-text markers.
  - Produce explicit exposure diagnostics metadata for downstream UI/logging to consume.
- `mochi/tools/file_ops.py`
  - Preserve `file_read` as the canonical chunk-capable text reader.
  - Add helper behavior for chunk-oriented agent flows if needed.
- `mochi/tools/base.py`
  - Stop forcing every tool result into a JSON envelope when the best model-facing payload is plain text.
- `mochi/tools/transport_guard.py`
  - Preserve safe file text reads when bounded.
  - Persist large results as artifacts plus resolvable references.
- `mochi/agents/react_loop.py`
  - Thread transport diagnostics and reference-follow-up behavior into the agent loop.
- `mochi/api/routes/filesystem.py`
  - Reuse existing file browsing API surface where possible instead of inventing a second browser stack.
- `tests/...`
  - Lock in all regression coverage for Chinese prompts, attachment paths, truncation/reference flows, and chunked follow-up reads.
  - Ensure producer/consumer boundaries are covered so UI-facing diagnostics are emitted before Task 3 consumes them.

### Task 1: Remove Keyword-Gated Workspace Read Tool Exposure

**Files:**
- Modify: `mochi/agents/tool_exposure.py`
- Modify: `mochi/agents/engine.py`
- Test: `tests/test_tool_exposure.py`
- Test: `tests/test_engine_phase2.py`

- [ ] **Step 1: Write failing planner tests for Chinese workspace inspection prompts**

Add tests covering prompts such as:

```python
message = "請先讀取規格.txt、資料說明.txt，並查看測試集資料夾與 baseline 程式"
message_with_attachments = "請先檢查我附上的規格.txt 與 run_official_baseline.py，再告訴我目前需要哪些資料欄位"
```

Expected exposed tools must include:

```python
["file_read", "glob_search", "grep_search"]
```

and must not require English tokens like `"read"` or `"file"` to appear.

- [ ] **Step 2: Write a failing test for attachment routing using the actual engine planner message**

Use `AgentEngine._build_tool_planner_message(...)` output as planner input and assert that attachment-driven read bias activates even though the message header is `Structured attachments:`.

Run:

```bash
pytest tests/test_tool_exposure.py tests/test_engine_phase2.py -k "attachment or chinese or workspace" -v
```

Expected: FAIL because current planner only recognizes `Attached workspace files:`.

- [ ] **Step 3: Implement always-on workspace read baseline**

In `ToolExposurePlanner.plan(...)`, ensure that for `session_bound_workspace=True`, the final exposed set always contains available core read-only workspace tools:

```python
CORE_WORKSPACE_READ_TOOLS = (
    "file_read",
    "glob_search",
    "grep_search",
    "csv_read",
    "pdf_read",
    "docx_read",
    "notebook_read",
)
```

Heuristics may still rank these tools, but must not suppress them.

- [ ] **Step 4: Convert legacy keyword routing to ranking-only behavior**

Keep the keyword tables only for:
- ordering
- optional specialized reader prioritization
- web/literature ranking

Do not let keyword mismatch remove the core workspace read baseline.

Add a deprecation comment above the keyword tables explaining:
- this routing is legacy
- new work must not add more keyword-gating rules
- future direction is dynamic discovery / capability-first exposure

- [ ] **Step 5: Fix attachment marker mismatch without adding another brittle string dependency**

Preferred implementation:
- stop relying on magic marker strings entirely for attachment-aware planner logic
- pass attachment presence / attachment kinds as structured booleans or derived planner flags

Fallback only if needed:
- support both `Structured attachments:` and `Attached workspace files:`

- [ ] **Step 6: Emit additive exposure diagnostics metadata at the producer boundary**

In Worker A scope, make the planner/engine path produce additive metadata that later rides along normal event serialization. Minimum fields:

```python
{
  "tool_exposure": {
    "exposed_tools": ["file_read", "glob_search", "grep_search"],
    "workspace_bound": True,
    "attachment_count": 2,
  }
}
```

Producer rules:
- `exposed_tools` must come from the actual final exposure set, not a duplicated guess in the API layer
- `workspace_bound` must come from the real session/workspace binding state
- `attachment_count` must come from structured attachment inputs already parsed by the engine
- this producer work belongs to Worker A in `mochi/agents/engine.py` and/or `mochi/agents/tool_exposure.py`
- Worker C in Task 3 only serializes and renders these fields; it must not recompute them

- [ ] **Step 7: Run focused tests**

Run:

```bash
pytest tests/test_tool_exposure.py tests/test_engine_phase2.py -v
```

Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add mochi/agents/tool_exposure.py mochi/agents/engine.py tests/test_tool_exposure.py tests/test_engine_phase2.py
git commit -m "fix: expose workspace read tools without keyword gating"
```

### Task 2: Restore Full-Fidelity File Reading For Model Workflows

**Files:**
- Modify: `mochi/tools/base.py`
- Modify: `mochi/tools/file_ops.py`
- Modify: `mochi/tools/transport_guard.py`
- Modify: `mochi/agents/react_loop.py`
- Test: `tests/test_tool_result_transport_guard.py`
- Test: `tests/test_tools_phase2.py`
- Test: `tests/test_phase3_tool_simulator_integration.py`

- [ ] **Step 1: Write failing tests that small `file_read` text survives transport intact**

Add a focused test where:
- tool is `file_read`
- output is plain text
- output is below threshold

Expected model-facing content should remain plain text, not forced through a summarized `Tool file_read result:` preview.

- [ ] **Step 2: Write failing tests that large `file_read` output becomes chunk-follow-up, not dead-end preview**

Expected behavior:
- first response returns a concise summary
- includes stable reference metadata
- follow-up flow can request the next chunk by reference

If the current implementation lacks a follow-up tool, write the failing tests first for the intended additive interface.

- [ ] **Step 2.5: Write failing tests that oversized workspace files degrade into chunked inspection instead of hard failure**

Current behavior to replace:

```python
ToolResult(error="File is too large to read: ... exceeds limit ...")
```

Target behavior:
- if `offset`/`limit` is provided, allow bounded chunk reads even when the full file exceeds the default byte ceiling
- if no chunk arguments are provided, return a model-visible instruction that tells the agent to retry with bounded chunk parameters

Example expected next step:

```python
file_read(path="large.log", offset=1, limit=200, line_numbers=True)
```

- [ ] **Step 3: Stop forcing plain text file reads through JSON-envelope-first formatting**

In `BaseTool.format_result_for_model(...)`, introduce a path that preserves plain text for read-only text tools when safe:

```python
if result.error is None and isinstance(result.output, str) and self.is_read_only:
    return result.output
```

Do not apply this blindly to structured or binary-style outputs.

- [ ] **Step 4: Make transport guard treat bounded text file reads as safe text**

In `ToolResultTransportGuard.guard(...)`, add a special-case for `file_read` plain text:
- if text is under `max_chars`
- if it is not JSON-shaped
- if no structural risk flags apply

then preserve the content instead of summarizing it.

- [ ] **Step 5: Add artifact-backed follow-up reads for large outputs**

Reuse the existing persisted payload path, but make it actionable through `file_read` instead of inventing a second read tool in this wave.

Define a virtual path contract:

```python
path = "tool-result://<reference_id>"
```

Implementation requirements:
- add stable metadata for `reference_id`, `artifact_path`, `tool_name`, `encoding`
- allow `file_read` to resolve `tool-result://<reference_id>` through `ToolExecutionContext.tool_result_references`
- make the model-facing tool message explicitly include the next-call contract
- preserve the existing line-based chunk interface so the model can request:

```python
file_read(path="tool-result://file_read-abc123", offset=1, limit=200, line_numbers=True)
```

Do not add a new tool in the first wave.

Bridge contract:
- **Worker B owns** the persisted artifact format and transport metadata fields:
  - `reference_id`
  - `artifact_path`
  - `encoding`
  - `transport_type`
- **Agent follow-up** uses `file_read(path="tool-result://<reference_id>", ...)` and stays runtime-local
- **UI/operator follow-up** uses `artifact_path` from transport metadata and existing filesystem preview/file endpoints
- Worker C must not invent a second `reference_id -> path` resolver; it consumes the persisted `artifact_path` emitted by Worker B
- the transport summary shown to the model must include a concrete retry hint, for example:

```text
Tool file_read result preview (truncated from 12000 chars).
Reference: file_read-abc123
To continue reading, call:
file_read(path="tool-result://file_read-abc123", offset=1, limit=200, line_numbers=true)
```

- [ ] **Step 6: Reuse `file_read` line-based chunking instead of inventing a second file reader**

Prefer these existing semantics:

```python
file_read(path="...", offset=1, limit=200, line_numbers=True)
```

For artifact follow-up, mirror that line-based chunking behavior to avoid duplicated read APIs.

- [ ] **Step 7: Remove or narrow the `\\b\\d+\\.txt\\b` suspicious marker rule**

`_SUSPICIOUS_FILE_RE` currently flags arbitrary strings like `1.txt`.
Replace it with a narrower transport rule tied to known backend failure signatures, or remove it if the new artifact flow makes it unnecessary.

- [ ] **Step 8: Run focused tests**

Run:

```bash
pytest tests/test_tool_result_transport_guard.py tests/test_tools_phase2.py tests/test_phase3_tool_simulator_integration.py -v
```

Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add mochi/tools/base.py mochi/tools/file_ops.py mochi/tools/transport_guard.py mochi/agents/react_loop.py tests/test_tool_result_transport_guard.py tests/test_tools_phase2.py tests/test_phase3_tool_simulator_integration.py
git commit -m "fix: preserve file reads and add resumable large tool results"
```

### Task 3: Add Workspace File Workflow Diagnostics And Operator Visibility

**Files:**
- Modify: `mochi/api/routes/filesystem.py`
- Modify: `mochi/api/routes/chat.py`
- Modify: `web/src/lib/api.ts`
- Modify: `web/src/components/chat/ReasoningPanel.tsx`
- Test: targeted API / frontend tests in the touched area

- [ ] **Step 1: Define additive diagnostics payloads**

Expose, at minimum:

```python
{
  "tool_exposure": {
    "exposed_tools": [...],
    "workspace_bound": True,
    "attachment_count": 3,
  },
  "transport": {
    "summary_applied": False,
    "overflow_persisted": True,
    "reference_id": "file_read-..."
  }
}
```

Do not remove existing fields; make this additive.

- [ ] **Step 2: Surface tool exposure diagnostics in the backend response path**

Add the diagnostics where the WebGUI can inspect them without scraping model prose.
Keep all `mochi/agents/engine.py` producer changes in Worker A; Task 3 only consumes additive metadata and existing session/runtime surfaces rather than reopening engine ownership.

Concrete path for this task:
- consume `event.metadata["tool_exposure"]` produced in Task 1
- serialize additive `ToolCallResultEvent.metadata` fields through [mochi/api/routes/chat.py](/H:/_python/agent_mochi/mochi/api/routes/chat.py)
- normalize those fields in [web/src/lib/api.ts](/H:/_python/agent_mochi/web/src/lib/api.ts)
- render them in [web/src/components/chat/ReasoningPanel.tsx](/H:/_python/agent_mochi/web/src/components/chat/ReasoningPanel.tsx)
- use `artifact_path` from transport metadata for operator preview flows; do not resolve `tool-result://` in the browser

- [ ] **Step 3: Reuse existing filesystem browsing routes instead of adding a second browser**

Prefer existing:
- `/v1/filesystem/roots`
- `/v1/filesystem/list`
- `/v1/filesystem/file`
- `/v1/filesystem/preview-text`

Add only what is missing for artifact/reference follow-up.

- [ ] **Step 4: Surface operator-facing exposure/truncation state in the WebGUI**

At minimum, show:
- which workspace read tools were exposed this turn
- whether a tool result was summarized
- whether an artifact/reference was persisted

Avoid making the user infer all of this from assistant prose.

- [ ] **Step 5: Run targeted verification**

Run:

```bash
pytest tests/test_api_chat_attachments.py tests/test_engine_phase2.py -k "attachment or reasoning" -v
npm.cmd --prefix web run type-check
npm.cmd --prefix web run lint
```

Expected: PASS

Frontend/backend files expected to move together in this task:
- `mochi/api/routes/chat.py`
- `web/src/lib/api.ts`
- `web/src/components/chat/ReasoningPanel.tsx`

- [ ] **Step 6: Commit**

```bash
git add mochi/api/routes/filesystem.py mochi/api/routes/chat.py web/src/lib/api.ts web/src/components/chat/ReasoningPanel.tsx
git commit -m "feat: surface workspace tool and transport diagnostics"
```

### Task 4: Introduce Dynamic Tool Discovery As The Default Path For Large Tool Sets

**Files:**
- Modify: `mochi/agents/tool_exposure.py`
- Modify: `mochi/tools/tool_search.py` or related tool catalog wiring
- Modify: `mochi/agents/prompt_builder.py`
- Test: `tests/test_tool_exposure.py`
- Test: add focused tests around discovery-first flows

- [ ] **Step 1: Write failing tests for discovery-first exposure**

Task ownership note:
- Worker D may prepare the design and test cases for this wave
- Worker A remains the sole code owner for `mochi/agents/tool_exposure.py` and `tests/test_tool_exposure.py`

Expected behavior:
- when many tools are available, the model still receives:
  - core workspace read baseline
  - `tool_search`
- specialized or niche tools can be discovered on demand instead of guessed from keywords

- [ ] **Step 2: Promote `tool_search` into the default large-tool-set path**

Planner should:
- keep small always-on core tools visible
- always include `tool_search` when tool count exceeds the configured threshold
- stop trying to pre-guess every specialized tool from prompt keywords

- [ ] **Step 3: Update prompt guidance**

Revise system prompt guidance so the model is explicitly told:
- use core read tools directly for workspace file inspection
- use `tool_search` when the required tool is not already visible

- [ ] **Step 4: Keep literature/web ranking as ranking, not hard gating**

Preserve useful capability-aware ordering for:
- literature retrieval
- web retrieval
- specialized readers

but do not let ranking logic remove the workspace read baseline.

- [ ] **Step 5: Run targeted tests**

Run:

```bash
pytest tests/test_tool_exposure.py -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add mochi/agents/tool_exposure.py mochi/tools/tool_search.py mochi/agents/prompt_builder.py tests/test_tool_exposure.py
git commit -m "feat: shift tool exposure toward dynamic discovery"
```

### Task 5: Optional Follow-Up Architecture For Repo Map / Symbol Index

**Files:**
- Create or modify: repo-map / symbol-index related modules
- Test: new targeted tests

- [ ] **Step 1: Write a short ADR or technical note**

Document:
- whether to build in-house or adopt an existing parser/index path
- update cadence
- workspace boundary behavior
- token budget target for the repo map

- [ ] **Step 2: Prototype the smallest useful read-only index**

Scope:
- file paths
- top-level symbols
- optional imports/dependency hints

- [ ] **Step 3: Add a narrow read-only retrieval API**

Examples:

```python
repo_map()
read_symbol(path="...", symbol="...")
```

- [ ] **Step 4: Verify the index improves file targeting without replacing normal reads**

This is a follow-up optimization, not a blocker for Tasks 1-4.

## Cross-Task Validation

- [ ] Run the focused backend test sets after each task.
- [ ] Run this broader suite before merging:

```bash
pytest tests/test_tool_exposure.py tests/test_engine_phase2.py tests/test_tool_result_transport_guard.py tests/test_tools_phase2.py tests/test_phase3_tool_simulator_integration.py -v
```

- [ ] If WebGUI files change, run:

```bash
npm.cmd --prefix web run type-check
npm.cmd --prefix web run lint
```

- [ ] Perform one manual smoke flow in the ESG workspace:
  1. Ask in Chinese to inspect attached `.txt` and `.py` files
  2. Confirm `file_read`, `glob_search`, `grep_search` are exposed without prompting hacks
  3. Confirm a small text file round-trips fully
  4. Confirm a large text file triggers a resumable chunk/reference flow
  5. Confirm the UI surfaces exposure and truncation diagnostics

## Expected End State

- Workspace-bound chats no longer depend on English prompt keywords to expose core read-only file tools.
- Structured attachments reliably route the model toward the correct reader.
- `file_read` becomes usable again for real auditing instead of devolving into preview-only transcript fragments.
- Oversized tool outputs no longer strand the model behind `Reference:` text with no follow-up path.
- Mochi moves closer to the mature pattern used by official MCP guidance and `cc-haha`: scoped availability, dynamic discovery, and artifact-backed large result handling.

## Execution Recommendation

Implement Tasks 1 and 2 first using subagent-driven development. They are the critical path and have disjoint write ownership. Do not start Task 4 before Task 1 stabilizes, otherwise you risk mixing a product-policy shift with a bug-fix wave and losing regression clarity.
