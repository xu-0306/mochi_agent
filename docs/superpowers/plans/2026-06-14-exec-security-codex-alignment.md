# Exec Security Codex Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the legacy `shell` pathway and ship a single `exec_command`-centric security model with global defaults, per-chat overrides, persistent command rules, and approval UX that behaves closer to Codex.

**Architecture:** Keep Mochi's existing autonomy presets as the user-facing mental model, but route all command execution through one structured exec policy engine. Replace the legacy shell allowlist with deterministic allow/ask/deny classification, persist explicit command rules into the normal config file, and expose the active safety mode in both Settings and chat sessions. Approval resolution must be unified across standalone approvals and task-backed resume flows so `/approvals/{id}/resolve` and `/tasks/{id}/resume` cannot diverge.

**Tech Stack:** Python, FastAPI, Pydantic, React, TypeScript, existing Mochi runtime/session stores, existing exec runtime and approval APIs.

---

## Progress Update (2026-06-14)

### Current Status Snapshot

- Task 1: mostly complete. Legacy shell fields are removed from product-facing config/API/UI flows, and `shell` is no longer a first-class registered tool. The compatibility shim and legacy `mochi/tools/shell.py` file are still intentionally present.
- Task 2: mostly complete. The internal classifier split is now landed, `command_security.py` is back to being the single orchestrator, and `exec_command` returns stable `policy_state` / `policy_reason` / `rule_id` / `suggested_rule` metadata. The remaining open point is an explicit audit of autonomy-mode execution thresholds against Task 2 Step 3.
- Task 3: complete for the current v1 scope. `command_rules`, `exec_allowed_env_vars`, `exec_default_shell`, and session `security_override` are wired through config/API/runtime/web.
- Task 4: complete for the current v1 scope. Approval resolution now uses explicit decisions (`approve_once`, `approve_and_save_rule`, `reject`) across runtime/task flows.
- Task 5: largely landed in code earlier in this thread, but it still needs a fresh manual browser pass after the next backend cleanup wave.
- Task 6: partially complete. Focused regression suites are green, but the broader final sweep listed below still needs to be run after the remaining cleanup is finished.

### Completed In This Stream

- Replaced boolean approval resolution with explicit decision strings across runtime, API, and tests.
- Added richer approval payloads and summaries, including `command`, `shell`, `workdir`, and `suggested_rule`.
- Bound `RuntimeService` config persistence so `approve_and_save_rule` can write validated `command_rules` back through the normal config save path.
- Propagated exec security settings into controlled/multi-agent execution so subagent execution uses the same `command_rules`, `exec_allowed_env_vars`, and `exec_default_shell` as normal `exec_command`.
- Removed stale `shell` wording from tool-exposure intent matching and from the explicit-skill test fixture so prompts lean toward `exec_command`.
- Split reusable command policy tables/helpers into:
  - `mochi/utils/command_policy_rules.py`
  - `mochi/utils/command_path_policy.py`
- Reworked `mochi/utils/command_security.py` so deterministic classification no longer depends on the legacy allowlist-style constructor and instead evaluates persisted `command_rules` plus built-in read-only / deny heuristics.
- Updated `mochi/tools/exec_command.py` so allow / ask / deny paths all emit stable `rule_id` and `suggested_rule` metadata, and approval-pending results keep the classifier's `policy_reason`.
- Aligned `tests/test_exec_security.py` and `tests/test_exec_tools.py` with the new `command_rules`-centric contract and explicit approval status model.

### Latest Verification Evidence

- Passed: `pytest tests/test_engine_phase2.py tests/test_engine_learning_phase5.py tests/test_compaction.py tests/test_api_runtime_detached_exec_recovery.py -q`
  - Result: `23 passed`
- Passed: `pytest tests/test_engine_learning_phase5.py tests/test_engine_phase2.py tests/test_config.py tests/test_security_policy.py tests/test_api_sessions_settings.py tests/test_api_runtime.py tests/test_compaction.py tests/test_api_runtime_detached_exec_recovery.py -q`
  - Result: `126 passed, 2 warnings`
- Passed: `pytest tests/test_exec_security.py tests/test_exec_tools.py tests/test_security_policy.py -q`
  - Result: `29 passed, 1 warning`
- Passed: `pytest tests/test_api_runtime.py -q`
  - Result: `38 passed, 1 warning`
- Known warnings in the latest run:
  - The newest focused runs only reported `PytestCacheWarning` because `.pytest_cache` could not be written in this environment.
  - The earlier broader run also reported the existing `audioop` deprecation warning from Discord voice code.

### Intentional Compatibility Leftovers

- `mochi/config/schema.py` still accepts `require_approval_for_shell` and discards `shell_command_allowlist` during config normalization. This is intentional migration behavior.
- `tests/test_security_policy.py` still contains a legacy-config test that verifies the compatibility shim above.
- `mochi/tools/shell.py` is still on disk as a backwards-compat implementation, but it is no longer the primary registered command-execution path.

### Remaining Work For The Next Agent

- Audit Task 2 Step 3 explicitly and confirm the effective autonomy-mode execution thresholds match the plan, rather than only relying on the existing `require_approval_for_exec` wiring.
- Audit whether `mochi/tools/shell.py` can be safely removed, or whether it still needs a temporary deprecation phase for legacy imports, stored skills, or older sessions.
- Run the remaining broader regression/search sweep after the remaining cleanup lands, especially:
  - `pytest tests/test_exec_security.py tests/test_exec_tools.py tests/test_security_policy.py tests/test_api_sessions_settings.py tests/test_api_runtime.py tests/test_tool_exposure.py -q`
  - `pytest tests/test_config.py tests/test_prompt_builder.py tests/test_api_chat_context.py tests/test_tool_system_upgrade.py -q`
  - `rg -n "require_approval_for_shell|shell_command_allowlist|Legacy shell|class ShellTool|\\bname\\s*=\\s*\"shell\"" mochi web tests configs`

## File Map

- Modify: `configs/default.yaml`
- Modify: `mochi/config/schema.py`
- Modify: `mochi/security/policy.py`
- Modify: `mochi/tools/registry_factory.py`
- Modify: `mochi/tools/exec_command.py`
- Delete after references are removed: `mochi/tools/shell.py`
- Create: `mochi/utils/command_policy_rules.py`
- Create: `mochi/utils/command_path_policy.py`
- Modify: `mochi/utils/command_security.py`
- Modify: `mochi/api/routes/settings.py`
- Modify: `mochi/api/routes/sessions.py`
- Modify: `mochi/api/routes/approvals.py`
- Modify: `mochi/api/routes/tasks.py`
- Modify: `mochi/runtime/models.py`
- Modify: `mochi/runtime/approvals.py`
- Modify: `mochi/runtime/store.py`
- Modify: `mochi/runtime/service.py`
- Modify: `web/src/lib/api.ts`
- Modify: `web/src/lib/stores/task-store.ts`
- Modify: `web/src/app/settings/page.tsx`
- Modify: `web/src/app/page.tsx`
- Modify: `web/src/components/chat/TaskPanel.tsx`
- Modify: `tests/test_config.py`
- Modify: `tests/test_exec_security.py`
- Modify: `tests/test_exec_tools.py`
- Modify: `tests/test_security_policy.py`
- Modify: `tests/test_api_sessions_settings.py`
- Modify: `tests/test_api_runtime.py`
- Modify: `tests/test_tool_exposure.py`

### Task 1: Remove Legacy Shell Surfaces

**Files:**
- Modify: `configs/default.yaml`
- Modify: `mochi/config/schema.py`
- Modify: `mochi/security/policy.py`
- Modify: `mochi/tools/registry_factory.py`
- Modify: `mochi/agents/prompt_builder.py`
- Modify: `mochi/agents/tool_exposure.py`
- Modify: `mochi/api/routes/settings.py`
- Modify: `web/src/lib/api.ts`
- Modify: `web/src/app/settings/page.tsx`
- Delete: `mochi/tools/shell.py`
- Test: `tests/test_config.py`
- Test: `tests/test_security_policy.py`
- Test: `tests/test_tool_exposure.py`

- [ ] **Step 1: Remove legacy shell fields from the runtime config contract**

Drop `require_approval_for_shell` and `shell_command_allowlist` from `SecurityConfig`, `RuntimePermissionPolicy`, and normalized `/v1/settings` responses. Keep a loader-time compatibility shim in `mochi/config/schema.py` that accepts old config keys, maps `require_approval_for_shell` into the initial `require_approval_for_exec` value only when the new key is absent, infers `autonomy_mode` from the old shape when needed, and does not preserve the legacy fields in memory after load.

- [ ] **Step 2: Stop registering and advertising the `shell` tool**

Remove `ShellTool` from the tool registry and all agent exposure/prompting paths so the model can no longer select it. Delete `mochi/tools/shell.py` only after all imports, tests, and prompt references are removed.

- [ ] **Step 3: Simplify the security settings API and UI**

Remove shell-only controls from `/v1/settings`, `web/src/lib/api.ts`, and the Settings security form. Keep `autonomy_mode`, `require_approval_for_exec`, `require_approval_for_file_write`, `file_ops_scope`, `exec_allowed_env_vars`, `exec_default_shell`, and exec runtime limits as the remaining security knobs.

- [ ] **Step 4: Verify the repo no longer treats shell as a first-class execution path**

Run: `pytest tests/test_security_policy.py tests/test_tool_exposure.py tests/test_config.py -q`

Expected: all tests pass with no references to `require_approval_for_shell`, `shell_command_allowlist`, or the `shell` tool in normalized API payloads.

### Task 2: Upgrade `exec_command` Into the Single Command Policy Engine

**Files:**
- Create: `mochi/utils/command_policy_rules.py`
- Create: `mochi/utils/command_path_policy.py`
- Modify: `mochi/utils/command_security.py`
- Modify: `mochi/tools/exec_command.py`
- Modify: `mochi/security/policy.py`
- Test: `tests/test_exec_security.py`
- Test: `tests/test_exec_tools.py`
- Test: `tests/test_security_policy.py`

- [x] **Step 1: Split deterministic command security into reusable policy modules**

Move path and protected-target checks into `mochi/utils/command_path_policy.py`, and move shell-specific read-only and blocked command tables into `mochi/utils/command_policy_rules.py`. Keep `mochi/utils/command_security.py` as the single orchestrator that returns one `CommandSecurityResult`.

- [x] **Step 2: Replace compatibility-allowlist logic with structured classification**

Implement exact behavior in `CommandSecurityPolicy`:

1. Hard `deny` for heredoc, subshell, shell chaining, command substitution, protected paths, workspace escape, disallowed inline eval, interactive shell spawn, and dangerous launcher payloads.
2. `allow` for known read-only Bash, PowerShell, and CMD commands that stay within workspace and do not include suspicious env or path mutations.
3. `ask` for mutating commands, escalation requests, unknown commands, non-allowlisted env overrides, and any command that cannot be proven read-only.

Remove the final compatibility branch that currently asks only because a command is outside `self._allowlist`; `command_rules` and deterministic classification replace that role.

- [ ] **Step 3: Make autonomy mode change only the approval threshold, not the parser**

Use the same classifier for all modes. Apply policy this way:

1. `strict`: every non-denied exec becomes approval-required.
2. `trusted_workspace`: read-only workspace-safe exec may auto-run; mutating or uncertain exec requires approval.
3. `auto_review`: same execution threshold as `trusted_workspace`, but approval-pending metadata must keep `policy_state="ask"` and only mark the request as review-friendly metadata. It must not silently auto-run or bypass approval persistence.
4. `high_autonomy`: workspace-safe mutating exec can auto-run, but escalation requests and hard-deny cases still do not bypass policy.

- [x] **Step 4: Return stable policy metadata and rule suggestions from `exec_command`**

Add `policy_state`, `policy_reason`, `rule_id`, and `suggested_rule` metadata to every `exec_command` result. `suggested_rule` must contain the exact or prefix token list, match mode, and shell list the UI can persist if the user chooses `approve_and_save_rule`.

- [x] **Step 5: Verify command classification and exec behavior**

Run: `pytest tests/test_exec_security.py tests/test_exec_tools.py tests/test_security_policy.py -q`

Expected: read-only commands pass where appropriate, dangerous syntax is denied, uncertain or mutating commands become approval-pending, and no path depends on the removed legacy shell allowlist.

### Task 3: Persist Command Rules and Session Safety Overrides

**Files:**
- Modify: `mochi/config/schema.py`
- Modify: `mochi/api/routes/settings.py`
- Modify: `mochi/api/routes/sessions.py`
- Modify: `web/src/lib/api.ts`
- Modify: `web/src/app/page.tsx`
- Test: `tests/test_api_sessions_settings.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Add persistent `command_rules` to security settings**

Define a new config field with this wire shape:

```json
{
  "tokens": ["git", "status"],
  "decision": "allow",
  "match": "prefix",
  "shells": ["bash", "powershell"]
}
```

Load and save it through `mochi/config/schema.py` and `/v1/settings`, and evaluate it before fallback heuristics in the exec classifier.

- [ ] **Step 2: Add a first-class session `security_override` state**

Extend `UpdateSessionRequest` in `mochi/api/routes/sessions.py` to accept `security_override`. Persist it as a dedicated `session_meta` event named `security_override_updated`, add a `_session_security_override()` resolver next to `_session_workflow_state()`, and include the latest normalized `security_override` in both session list and session detail responses. The override shape is v1-limited to:

```json
{
  "autonomy_mode": "strict" | "trusted_workspace" | "auto_review" | "high_autonomy"
}
```

Do not overload workflow state for this.

- [ ] **Step 3: Resolve effective safety mode in chat from global default plus session override**

In `web/src/app/page.tsx`, compute the active mode from session `security_override.autonomy_mode` when a chat is selected. Use Settings as the default fallback when the session has no override.

- [ ] **Step 4: Verify settings and session metadata round-trips**

Run: `pytest tests/test_api_sessions_settings.py tests/test_config.py -q`

Expected: `command_rules` persists through Settings, session PATCH persists `security_override`, and session list/detail responses expose the latest normalized override without reintroducing shell-only fields.

### Task 4: Upgrade the Approval Contract and Runtime Resolution Flow

**Files:**
- Modify: `mochi/runtime/models.py`
- Modify: `mochi/runtime/approvals.py`
- Modify: `mochi/runtime/store.py`
- Modify: `mochi/runtime/service.py`
- Modify: `mochi/api/routes/approvals.py`
- Modify: `mochi/api/routes/tasks.py`
- Modify: `web/src/lib/api.ts`
- Modify: `web/src/lib/stores/task-store.ts`
- Test: `tests/test_api_runtime.py`
- Test: `tests/test_exec_tools.py`

- [ ] **Step 1: Replace boolean approval resolution with explicit decisions**

Change `ApprovalResolution` from `{ approved: bool, reason?: str }` to:

```json
{
  "decision": "approve_once" | "approve_and_save_rule" | "reject",
  "reason": "optional reviewer note",
  "rule": {
    "tokens": ["git", "status"],
    "decision": "allow",
    "match": "prefix",
    "shells": ["bash"]
  }
}
```

Apply this contract to both `/approvals/{approval_id}/resolve` and `/tasks/{task_id}/resume` so task-backed approvals and standalone approvals share the same semantics.

- [ ] **Step 2: Persist approval decisions in both stores using explicit status strings**

Update `mochi/runtime/approvals.py` and `mochi/runtime/store.py` so the resolution path accepts decision strings instead of a boolean. Normalize stored status to:

1. `pending`
2. `approved_once`
3. `approved_and_saved_rule`
4. `rejected`

Expose a derived UI `decision` field while keeping list filtering backward-compatible for `pending` and resolved approvals.

- [ ] **Step 3: Bind runtime config persistence to the existing app config path**

Add a config binding path to `RuntimeService` and populate it from `_get_runtime_service()` using `request.app.state.config` and `request.app.state.config_path`. When a user resolves an approval with `approve_and_save_rule`, `RuntimeService` must:

1. Validate or synthesize the final rule from `suggested_rule` plus optional user-provided override.
2. Mutate the bound in-memory `MochiConfig.security.command_rules`.
3. Call existing `save_config(bound_config, bound_config_path)`.
4. Refresh `self._security_config` from the saved config object before replaying the exec request.

Do not create a second out-of-band settings persistence path.

- [ ] **Step 4: Teach the runtime to replay commands with the new resolution semantics**

When a pending exec approval is resolved:

1. `approve_once`: execute the saved command payload without mutating config.
2. `approve_and_save_rule`: persist the rule first, then execute the saved command payload.
3. `reject`: mark the approval rejected and never execute the command.

Linked task approvals must mirror the same decision to the exec approval store before the task resumes.

- [ ] **Step 5: Keep approval metadata rich enough for UI and audit**

Ensure stored approvals retain command, shell, workdir, policy reason, scope, replay-safety, rule id, and suggested rule data so the UI can render one review card without extra inference.

- [ ] **Step 6: Verify runtime approval flows**

Run: `pytest tests/test_api_runtime.py tests/test_exec_tools.py -q`

Expected: one-shot approval executes without persisting rules, save-rule approval persists the rule and allows future matching commands, reject leaves the command unexecuted, and `/tasks/{id}/resume` matches `/approvals/{id}/resolve`.

### Task 5: Add Codex-Style Safety Controls to the Chat UI

**Files:**
- Modify: `web/src/app/page.tsx`
- Modify: `web/src/components/chat/TaskPanel.tsx`
- Modify: `web/src/lib/api.ts`
- Modify: `web/src/lib/stores/task-store.ts`

- [ ] **Step 1: Add a session-scoped safety selector near the composer**

Use the existing chat page surface to show the active mode for the current chat session. The selector must offer `strict`, `trusted_workspace`, `auto_review`, and `high_autonomy`, and persist the selection through the session PATCH API rather than a frontend-only draft store.

- [ ] **Step 2: Keep Settings as the global default source**

Do not remove the Settings security section. Label it as the workspace default and show that chat sessions may override it.

- [ ] **Step 3: Upgrade the approval panel from boolean actions to command review actions**

In `TaskPanel`, replace `Approve` and `Reject` with:

1. `Approve once`
2. `Approve and save rule`
3. `Reject`

Each approval card must display command, shell, workdir, policy reason, approval scope, and the suggested rule preview that will be saved.

- [ ] **Step 4: Update the task store and API client to use the new resolution contract**

Replace the current `decision: "approve" | "reject"` client shape with the new three-way resolution payload, and make both approval actions and task resume actions use the same API contract.

- [ ] **Step 5: Verify the frontend contract manually after backend work lands**

Run:

```bash
pytest tests/test_api_sessions_settings.py tests/test_api_runtime.py -q
```

Then manually verify:

1. Settings shows global safety defaults only.
2. Chat UI shows the active session override.
3. Pending approvals expose all three actions.
4. Saved-rule approval changes the next matching command from pending to auto-allowed.

### Task 6: Final Regression Sweep

**Files:**
- Modify any failing tests from earlier tasks, but do not reintroduce legacy shell compatibility fields.

- [ ] **Step 1: Run the focused regression suite**

Run:

```bash
pytest tests/test_exec_security.py tests/test_exec_tools.py tests/test_security_policy.py tests/test_api_sessions_settings.py tests/test_api_runtime.py tests/test_tool_exposure.py -q
```

Expected: all focused command-security and API tests pass.

- [ ] **Step 2: Run a broader confidence pass**

Run:

```bash
pytest tests/test_config.py tests/test_prompt_builder.py tests/test_api_chat_context.py tests/test_tool_system_upgrade.py -q
```

Expected: no regressions in config loading, prompt exposure, or tool system normalization.

- [ ] **Step 3: Confirm cleanup is complete**

Run:

```bash
rg -n "require_approval_for_shell|shell_command_allowlist|Legacy shell|class ShellTool|\\bname\\s*=\\s*\"shell\"" mochi web tests configs
```

Expected: no product code hits remain; only plan/history documents may still mention the removed legacy shell path.

## Defaults And Assumptions

- The public execution tool remains `exec_command`; this plan does not split the user-facing tool surface into separate Bash and PowerShell tools.
- v1 does not add an LLM-based smart approval layer or Hermes-style Tirith scanning; all approval gating is deterministic and rule-based.
- Per-chat override is limited to `autonomy_mode` for now; rule editing stays in Settings and approval cards.
- Saved rules are persisted through the normal app config file using the existing `save_config()` path bound from `app.state.config_path`.
- Old `require_approval_for_shell` and `shell_command_allowlist` keys are accepted only as one-time migration inputs and are never returned from API responses after this plan lands.
