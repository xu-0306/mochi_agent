---
summary: "Stage 6 fix for the false protected-path denial under .mochi/workspace, structured file-mutation diagnostics, and the responsive DiffViewer regression."
created: 2026-07-14
updated: 2026-07-14
tags: [tool-activation, workspace-safety, protected-path, file-mutation, webgui, diff-viewer]
related: [mochi/utils/security.py, mochi/tools/file_ops.py, mochi/tools/file_mutations.py, web/src/components/chat/DiffViewer.tsx, tests/test_tool_system_upgrade.py]
---

# Tool Activation Stage 6: Protected Workspace and Diff UI

## Problem classification

This was a cross-layer contract bug with three coupled symptoms:

1. A security/path-policy false positive: the configured writable workspace was H:/_python/agent_mochi/.mochi/workspace, but the protected-path classifier rejected every resolved path containing .mochi.
2. A file-mutation observability gap: write/edit/patch denials did not consistently report the effective workspace boundary and resolved target, making a policy denial hard to diagnose.
3. A responsive UI regression: react-diff-viewer-continued kept its default min-width: 1000px while the workspace panel could be narrow, compressing or clipping the visible diff.

Auto Review only controls approval decisions. It must not bypass protected-path or workspace security policy.

## Root cause and implemented behavior

- mochi/utils/security.py now treats an explicitly configured workspace nested under .mochi as writable while keeping the .mochi root, .git, .vscode, .idea, protected files, and protected descendants denied.
- mochi/tools/file_ops.py adds structured path-denial metadata (workspace_dir, requested_path, path_scope, resolved_path) and exposes effective path metadata on successful file write/edit results.
- mochi/tools/file_mutations.py carries equivalent path diagnostics through apply_patch validation failures.
- web/src/components/chat/DiffViewer.tsx overrides the package table/cell sizing with minWidth: 0, bounded widths, fixed layout, and block-level syntax content so narrow workspace panels remain usable.
- tests/test_tool_system_upgrade.py covers the .mochi/workspace/test.txt real side effect and structured apply_patch protected-path denial.

## Verification

- Direct side-effect smoke test passed: .mochi/workspace/<unique>.txt was created with exact content hi, then removed.
- Direct security smoke test passed: .mochi/workspace/.git/<unique> remained denied with no file created.
- apply_patch protected-path regression: 1 passed.
- Python compile check, frontend npm run type-check, workspace panel source regression, and git diff --check passed.
- Full pytest suites remain affected by the pre-existing Windows WinError 5 ACL problem in pytest temp setup/teardown; no assertion failure was observed in the completed runs.
- Browser-level visual verification remains pending because the local sandbox blocked the Next child-process startup with spawn EPERM.