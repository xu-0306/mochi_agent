# Test Suite Baseline

Date: 2026-07-21

Base commit: `debef2759ffb34e3bc2c572657540265b9a875c3`

Platform: Windows (`win32`), Python 3.12.10, pytest 9.0.3, pytest-asyncio
1.3.0, asyncio mode `auto`.

## Collection

- Python test files: 154
- Collected cases: 1,726
- Inventory: `docs/testing/test-inventory.txt`
- Collection time: 1.63 seconds

The inventory contains only collected pytest node IDs. Regenerate it with:

```powershell
rtk pytest --collect-only -q > docs/testing/test-inventory.txt
```

Largest files by collected case count:

| File | Cases |
| --- | ---: |
| `tests/test_tool_system_upgrade.py` | 43 |
| `tests/security/test_file_transaction.py` | 41 |
| `tests/backends/test_ollama.py` | 41 |
| `tests/security/test_file_transaction_posix.py` | 39 |
| `tests/test_openai_compat_backend.py` | 39 |
| `tests/test_channels_phase45.py` | 36 |
| `tests/unit/tool_exposure/test_exposure_policies.py` | 35 |
| `tests/integration/api/models/test_runtime_selection.py` | 34 |
| `tests/test_config.py` | 34 |
| `tests/test_local_model_discovery.py` | 32 |
| `tests/integration/api/models/test_model_routes.py` | 31 |
| `tests/security/safe_filesystem/test_windows_filesystem.py` | 29 |
| `tests/integration/api/runtime/test_scheduling_and_recovery.py` | 29 |
| `tests/integration/api/goals/test_turn_decisions.py` | 28 |
| `tests/test_runtime_store.py` | 28 |
| `tests/security/test_file_transaction_windows.py` | 26 |
| `tests/test_phase3_router_backends.py` | 25 |
| `tests/integration/api/goals/test_lifecycle_and_followups.py` | 24 |
| `tests/test_voice_runtime.py` | 24 |
| `tests/integration/api/chat/test_subagent_control.py` | 23 |

## Marker Classification

Classification is applied in `tests/conftest.py` during collection so moved
tests retain the same execution lanes without repeating decorators in every
file.

| Marker | Collected cases |
| --- | ---: |
| `integration` | 451 |
| `security` | 223 |
| `windows` | 58 |
| `posix` | 55 |
| `slow` | 10 |
| `network` | 0 |

The integration and security counts are structural package counts. Windows and
POSIX counts overlap with security because the platform suites live below
`tests/security/`. The slow marker records the ten tests at or above the
approximately 4.5-second measured baseline; update the explicit list when a
new duration baseline changes.

## Static Checks

- Phase 7 target file: ruff check tests/conftest.py passed.
- git diff --check passed.
- A full ruff check tests run remains non-green with 2,440 existing findings
  across the dirty structural-split test files, primarily F405 from local
  star-import support modules and I001 import ordering. Those findings are
  outside the Phase 7 marker/docs change and were not auto-fixed.

## Execution

Full Python 3.12 offline verification:

```text
1,724 passed, 2 skipped, 2 warnings in 572.49s (0:09:32)
```

The run used the direct Python 3.12 interpreter because the existing
Python 3.13 `.venv` prevented `uv run --python 3.12 --no-sync` from selecting
the requested interpreter. The validated project-local uv cache remains
`H:/_python/agent_mochi/.tmp/uv-cache`.

The two skipped cases are the Windows-host skips for Linux-only
`posix_acl` and `user_xattrs` cases in
`tests/security/test_file_transaction_posix.py`. Warnings are the Python
3.13 `audioop` deprecation and the Python 3.14 tar extraction warning.

## Quick Lane Verification

The documented quick lane (`tests/unit tests/backends`) completed with:

```text
180 passed in 164.69s (0:02:44)
```

## Marker Lane Verification

The marker lanes were executed with the direct Python 3.12 interpreter and the
project-local workspace temp root:

| Lane | Result | Warnings | Runtime |
| --- | --- | ---: | ---: |
| `integration` | 451 passed | 1 | 216.33s |
| `security` | 221 passed, 2 skipped | 1 | 8.35s |
| `windows` | 58 passed | 1 | 2.56s |
| `posix` | 53 passed, 2 skipped | 1 | 1.35s |
| `slow` | 10 passed | 1 | 87.90s |
| `not network` | 1,724 passed, 2 skipped | 2 | 572.49s |

The Windows lane is fully green. The two POSIX skips are expected on Windows
because `posix_acl` and `user_xattrs` require Linux-native facilities.

Slowest measured calls from the same run:

| Seconds | Test |
| ---: | --- |
| 30.02 | `tests/integration/api/models/test_codex_auth_projection.py::test_openai_codex_refresh_access_token_times_out_on_live_file_lock` |
| 9.18 | `tests/test_web_search.py::test_web_search_metadata_distinguishes_missing_key_and_request_failure` |
| 5.33 | `tests/test_agent_run_operator_endpoints.py::test_agent_run_events_stream_returns_sse_frames` |
| 4.67 | `tests/unit/engine/test_sessions_and_events.py::test_engine_persists_and_restores_session_history` |
| 4.62 | `tests/test_compaction.py::test_engine_compaction_does_not_pollute_canonical_restore` |
| 4.60 | `tests/test_tool_system_upgrade.py::test_registry_factory_caches_registries_per_workspace` |
| 4.57 | `tests/unit/engine/test_preflight_and_backend.py::test_engine_preview_and_chat_invoke_share_classifier_first_tool_intent_contract` |
| 4.56 | `tests/unit/engine/test_tool_exposure_and_invocation.py::test_engine_invoke_exposes_tool_exposure_metadata_from_final_plan` |
| 4.53 | `tests/unit/engine/test_tool_exposure_and_invocation.py::test_engine_chinese_workspace_prompt_exposes_workspace_read_baseline` |
| 4.49 | `tests/test_engine_phase3.py::test_engine_apply_config_refreshes_active_gguf_runtime_root` |

## Environment Boundaries

- The baseline was run on Windows and covers Windows-native behavior available
  in this workspace.
- POSIX filesystem tests require Linux-native dir-fd, xattr, and ACL behavior;
  the two Linux-only cases are skipped on Windows.
- Tests needing live external services or API credentials must use the
  `network` marker and are excluded from the offline lane.
- No coverage percentage gate is introduced by this baseline. Coverage must be
  measured and trusted before a percentage promise is added.

## Baseline Scope

This baseline is a behavior and collection reference for the structural test
refactor. Test moves must preserve node inventory and assertions. Production
fixes discovered while making the suite truthful remain separate from test move
commits.
