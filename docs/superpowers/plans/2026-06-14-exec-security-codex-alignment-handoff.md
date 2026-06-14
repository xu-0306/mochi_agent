# Exec Security Codex Alignment Handoff

## Scope

This handoff covers the current `exec-security` and Codex-alignment cleanup stream in `H:\_python\agent_mochi`. The goal is still to converge on one `exec_command`-centric execution and approval model, while keeping only the minimum legacy compatibility required for older config shapes.

## What Is Already Landed

- Approval resolution is no longer boolean-first. The runtime and API now use:
  - `approve_once`
  - `approve_and_save_rule`
  - `reject`
- Approval summaries now expose richer exec metadata, including:
  - `command`
  - `shell`
  - `workdir`
  - `suggested_rule`
- Session-level `security_override` support is wired through API/runtime/web for `autonomy_mode`.
- Product-facing settings are centered on exec security:
  - `command_rules`
  - `exec_allowed_env_vars`
  - `exec_default_shell`
  - no product-facing dependence on `require_approval_for_shell`
- Controlled and multi-agent execution now inherits the same exec security inputs as normal `exec_command`:
  - `command_rules`
  - `exec_allowed_env_vars`
  - `exec_default_shell`
- `shell` is no longer a first-class registered tool in the main registry path.
- The command classifier internals are now split into reusable modules:
  - `mochi/utils/command_policy_rules.py`
  - `mochi/utils/command_path_policy.py`
  - with `mochi/utils/command_security.py` acting as the single orchestrator again
- `exec_command` now emits stable command-review metadata across allow / ask / deny paths:
  - `policy_state`
  - `policy_reason`
  - `rule_id`
  - `suggested_rule`

## Files Most Relevant To This Stream

- `mochi/config/schema.py`
- `mochi/security/policy.py`
- `mochi/utils/command_policy_rules.py`
- `mochi/utils/command_path_policy.py`
- `mochi/utils/command_security.py`
- `mochi/tools/exec_command.py`
- `mochi/tools/shell.py`
- `mochi/agents/engine.py`
- `mochi/agents/multi_agent/execution_coordinator.py`
- `mochi/agents/multi_agent/orchestrator.py`
- `mochi/agents/tool_exposure.py`
- `mochi/runtime/service.py`
- `tests/test_config.py`
- `tests/test_security_policy.py`
- `tests/test_exec_security.py`
- `tests/test_exec_tools.py`
- `tests/test_api_sessions_settings.py`
- `tests/test_api_runtime.py`
- `tests/test_compaction.py`
- `tests/test_api_runtime_detached_exec_recovery.py`
- `tests/test_engine_phase2.py`
- `tests/test_engine_learning_phase5.py`

## Verified State

These commands were run successfully in this workspace:

- `pytest tests/test_engine_phase2.py tests/test_engine_learning_phase5.py tests/test_compaction.py tests/test_api_runtime_detached_exec_recovery.py -q`
  - Result: `23 passed`
- `pytest tests/test_engine_learning_phase5.py tests/test_engine_phase2.py tests/test_config.py tests/test_security_policy.py tests/test_api_sessions_settings.py tests/test_api_runtime.py tests/test_compaction.py tests/test_api_runtime_detached_exec_recovery.py -q`
  - Result: `126 passed, 2 warnings`
- `pytest tests/test_exec_security.py tests/test_exec_tools.py tests/test_security_policy.py -q`
  - Result: `29 passed, 1 warning`
- `pytest tests/test_api_runtime.py -q`
  - Result: `38 passed, 1 warning`

Observed warnings:

- The newest focused runs only reported `PytestCacheWarning` because `.pytest_cache` could not be written in this environment.
- The earlier broader run also reported the existing `audioop` deprecation warning from Discord voice code.

## Intentional Remaining Legacy Points

- `mochi/config/schema.py` still contains the migration shim for:
  - `require_approval_for_shell`
  - `shell_command_allowlist`
- `tests/test_security_policy.py` still has a compatibility test that intentionally exercises that shim.
- `mochi/tools/shell.py` still exists on disk as a backward-compat implementation.

These are not accidental leftovers unless the next agent explicitly decides to finish the deprecation/removal phase.

## Important Findings

- The most important product fix in this pass was making controlled execution use the same exec security inputs as standard `exec_command`. Before that, subagent execution could drift from the main runtime policy.
- The old shell-era wording in prompt/tool exposure can create subtle regressions by biasing the model back toward `shell`. Some of that wording has now been cleaned up.
- The deeper Task 2 refactor from the plan is now partially landed:
  - `mochi/utils/command_policy_rules.py` exists and owns reusable shell/policy tables plus token helpers.
  - `mochi/utils/command_path_policy.py` exists and owns workspace/sensitive-path checks.
  - `mochi/utils/command_security.py` now consumes those modules instead of carrying all rule tables inline.
- The remaining Task 2 open point is not the split anymore; it is the explicit audit of autonomy-mode execution thresholds from Task 2 Step 3.
- `tests/test_exec_security.py` and `tests/test_exec_tools.py` were still on an older contract before this pass. They are now aligned with `command_rules` and the explicit approval decision/status model.

## Suggested Next Steps

1. Audit Task 2 Step 3 explicitly and confirm the effective autonomy-mode execution thresholds match the plan, rather than assuming the current `require_approval_for_exec` wiring is sufficient.
2. Audit whether `mochi/tools/shell.py` is still needed for any legacy imports, stored skills, or session replay paths before deleting it.
3. Run the broader final sweep from the plan after the next code change set:
   - `pytest tests/test_exec_security.py tests/test_exec_tools.py tests/test_security_policy.py tests/test_api_sessions_settings.py tests/test_api_runtime.py tests/test_tool_exposure.py -q`
   - `pytest tests/test_config.py tests/test_prompt_builder.py tests/test_api_chat_context.py tests/test_tool_system_upgrade.py -q`
   - `rg -n "require_approval_for_shell|shell_command_allowlist|Legacy shell|class ShellTool|\\bname\\s*=\\s*\"shell\"" mochi web tests configs`
4. After the remaining backend cleanup, do the manual/browser pass for the approval UX and saved-rule flow.

## Watchouts

- Do not remove the compatibility shim in `mochi/config/schema.py` unless you also update the explicit compatibility test and confirm older config files still migrate cleanly.
- Do not assume `mochi/tools/shell.py` is safe to delete just because it is no longer first-class in the registry; check for legacy imports and stored-data expectations first.
- The repo is dirty. Avoid reverting unrelated user changes while continuing this stream.
- The background exec tests are sensitive to the current command-token normalization behavior in `mochi/utils/command_security.py`. If those tests fail again, re-check the tokenizer/rule matching before changing the assertions.

## Recommended Restart Point For The Next Agent

Start with:

1. Read `docs/superpowers/plans/2026-06-14-exec-security-codex-alignment.md`
2. Read this handoff file
3. Re-run the latest focused verification if you want a clean baseline:
   - `pytest tests/test_exec_security.py tests/test_exec_tools.py tests/test_security_policy.py -q`
   - `pytest tests/test_api_runtime.py -q`
4. Run the targeted search:
   - `rg -n "require_approval_for_shell|shell_command_allowlist|shell" mochi tests web configs`
5. Then choose between:
   - autonomy-threshold audit for Task 2 Step 3
   - legacy `shell.py` deprecation/removal
   - broader regression sweep
   - browser/manual verification of the approval UX
