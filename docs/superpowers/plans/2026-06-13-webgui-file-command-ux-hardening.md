# WebGUI File & Command UX Hardening Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` task-by-task.

**Goal:** 修正附件路徑可靠性，並把 legacy shell UX 重整為更清楚、Windows 友善、可解釋的命令政策，同步改善 workflow role 呈現。

**Architecture:** 以 `exec_command` 作為主要執行路徑，`shell` 僅保留相容模式；一般讀檔/列目錄/搜尋由專用檔案工具優先承接。WebGUI 文案與角色呈現要對齊 runtime 現況，讓使用者明白 planner / judge / verifier / synthesizer 與執行者的差異。

**Tech Stack:** FastAPI backend, Next.js WebGUI, Mochi tool/security layer, pytest, Playwright.

---

### Task 1: Attachment Path Contract

**Files:**
- Modify: `mochi/api/routes/filesystem.py`
- Modify: `web/src/lib/api.ts`
- Modify: `web/src/components/chat/ChatInput.tsx`
- Test: `web/scripts/test-chat-uploaded-file-paths.mjs`
- Test: `web/scripts/test-chat-paste-attachments.mjs`

- [ ] **Step 1: Tighten import response semantics**

Ensure single-file imports always expose the real saved file path while keeping multi-file package metadata authoritative.

- [ ] **Step 2: Prefer concrete file paths in chat attachment mapping**

Keep the frontend fallback order as `files[0].path ?? importedPath`.

- [ ] **Step 3: Add regression coverage**

Verify PDF uploads and paste uploads never hand the model the `browser-imports` package directory when a real file path exists.

### Task 2: Command Safety Redesign

**Files:**
- Modify: `configs/default.yaml`
- Modify: `mochi/utils/security.py`
- Modify: `mochi/tools/shell.py`
- Modify: `mochi/tools/exec_command.py`
- Modify: `mochi/agents/tool_exposure.py`
- Modify: `mochi/agents/prompt_builder.py`
- Test: backend security / tool exposure regression tests

- [ ] **Step 1: Replace allowlist-centric UX with explicit policy states**

Make tool results clearly describe `allow / ask / deny` and preserve human-readable denial reasons.

- [ ] **Step 2: Make `exec_command` the default guidance path**

Keep `shell` as legacy compatibility only and reduce its visibility in normal prompts and tool exposure.

- [ ] **Step 3: Expand Windows read-only command handling**

Allow harmless read/browse commands on Windows/PowerShell while still blocking dangerous interpreters, process spawns, and code-execution cmdlets.

- [ ] **Step 4: Route file-browse intent away from shell**

Prefer dedicated tools such as `file_read`, `glob_search`, `grep_search`, and `pdf_read` before any shell fallback.

### Task 3: Workflow Sidebar UX

**Files:**
- Modify: workflow sidebar / role configuration components
- Modify: workflow copy / labels in WebGUI

- [ ] **Step 1: Rewrite Smart model copy**

Make it explicit that the selected model is the shared default for planner, judge, verifier, and synthesizer roles.

- [ ] **Step 2: Make agent roles dropdown-driven**

Use preset/dropdown choices with short descriptions instead of forcing users to infer role responsibilities from raw labels.

- [ ] **Step 3: Separate research vs execution affordances**

Show clearly which roles only research/verify and which actions can write/run code.

### Task 4: Verification

**Files:**
- Test: backend route and security tests
- Test: frontend type-check / lint
- Test: browser smoke checks

- [ ] **Step 1: Run backend regressions**

Cover upload path, shell policy, and tool exposure routing.

- [ ] **Step 2: Run frontend checks**

Confirm sidebar layout, copy, and attachment flows remain stable.

- [ ] **Step 3: Run browser smoke tests**

Verify PDF upload, file browsing, and denial messaging end to end.

**Assumptions**
- WebGUI / backend are phase 1; desktop stays phase 2.
- `shell` remains compatibility-only, not the normal recommended path.
- The preferred UX is clearer policy explanations, not broader blocking.
