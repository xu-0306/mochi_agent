# Mochi SGLang and TensorRT-LLM External Provider Preset Plan

## Status

Proposed

## Summary

Add `SGLang` and `TensorRT-LLM` as external OpenAI-compatible provider presets.

This plan intentionally does not add managed runtime support. Mochi will not install, start, stop, or tune SGLang or TensorRT-LLM server processes in this pass. Users must run those servers externally and point Mochi at their OpenAI-compatible `/v1` endpoint.

The immediate goal is to make these providers convenient and explicit in settings, model configuration, and saved model entries while reusing Mochi's existing `OpenAICompatBackend`.

## Product Boundary

### In scope

- Add provider IDs:
  - `sglang`
  - `tensorrt_llm`
- Add external provider presets with default base URLs.
- Route both providers through `OpenAICompatBackend`.
- Preserve provider identity in configured model records and settings payloads.
- Add settings UI options, labels, descriptions, notes, and placeholders.
- Add backend and frontend tests for configure, switch, settings summary, and key redaction.

### Out of scope

- No `SGLangBackend`.
- No `TensorRTLLMBackend`.
- No managed SGLang runtime manager.
- No managed TensorRT-LLM runtime manager.
- No install flow for either runtime.
- No GPU, engine, tensor parallel, KV cache, batching, quantization, or parser startup controls.
- No claim that Mochi can adjust already-running server runtime parameters.

## Rationale

SGLang, TensorRT-LLM, and vLLM expose many important runtime settings at server startup time. When Mochi connects to an already-running external server, it can only control request-time OpenAI-compatible parameters and Mochi application behavior.

External provider presets are still useful because they:

- make provider identity visible in the UI and saved model list
- provide correct default base URLs and model placeholders
- keep API key handling and redaction consistent
- allow provider-specific diagnostics and request quirks later
- avoid misleading users into thinking Mochi controls server deployment settings

## Official Documentation Baseline

SGLang:

- Official usage centers on `python -m sglang.launch_server`.
- Typical OpenAI-compatible base URL is `http://127.0.0.1:30000/v1`.
- Tool and reasoning behavior may depend on server startup options such as tool parser and reasoning parser choices.

TensorRT-LLM:

- Official serving path uses `trtllm-serve`.
- The server exposes OpenAI-compatible endpoints.
- Deployment depends heavily on NVIDIA runtime, TensorRT-LLM engine preparation, GPU topology, batching, and quantization choices.

Implementation rule:

- Treat both as externally managed OpenAI-compatible endpoints for this phase.

## Current Mochi Constraints

### Remote providers share one OpenAI-compatible backend

The existing remote preset model stores:

- `base_url`
- `model`
- `api_key`
- `provider`

Current provider values include:

- `openai_compat`
- `gemini`
- `anthropic`
- `vllm`

Primary touch points:

- `mochi/config/schema.py`
- `mochi/api/routes/models.py`
- `mochi/api/routes/settings.py`
- `mochi/agents/engine.py`
- `mochi/backends/router.py`
- `web/src/lib/api.ts`
- `web/src/app/settings/page.tsx`
- `web/src/lib/i18n.tsx`

### vLLM has a special managed path

`vllm` currently has both:

- external OpenAI-compatible provider behavior
- managed runtime behavior for selected model specs

The new providers must not enter the vLLM managed branch. `sglang` and `tensorrt_llm` should be external-only.

## Provider Defaults

Recommended IDs and labels:

| Provider ID | UI Label | Backend Type | Launch Mode |
| --- | --- | --- | --- |
| `sglang` | `SGLang` | `openai_compat` | `external` |
| `tensorrt_llm` | `TensorRT-LLM` | `openai_compat` | `external` |

Recommended default URLs:

| Provider ID | Default Base URL | Default Model Placeholder |
| --- | --- | --- |
| `sglang` | `http://127.0.0.1:30000/v1` | `Qwen/Qwen3-8B` |
| `tensorrt_llm` | `http://127.0.0.1:8000/v1` | `model` |

Note:

- The TensorRT-LLM default URL is a project convention for external setup convenience. If the local deployment uses a different port, the user must enter that URL.

## Subagent Dispatch Plan

Use subagents only for the implementation phase. The main agent should act as integration owner and code reviewer.

### Subagent A: Backend API and Schema

Owner type: backend worker

Files:

- `mochi/config/schema.py`
- `mochi/agents/engine.py`
- `mochi/api/routes/models.py`
- `mochi/api/routes/settings.py`
- `tests/test_api_chat_models.py`
- `tests/test_api_sessions_settings.py`
- `tests/test_engine_phase2.py`

Responsibilities:

- Add `sglang` and `tensorrt_llm` to backend provider enums.
- Extend remote provider defaults.
- Ensure `/v1/models/configure` accepts both providers.
- Ensure `/v1/models/switch` restores both providers from configured model IDs.
- Ensure `GET /v1/settings` reports active provider and OpenAI-compatible metadata correctly.
- Ensure both providers call `switch_openai_compat_backend(...)`.
- Ensure both providers preserve `backend_type="openai_compat"`.
- Ensure both providers never trigger managed vLLM logic.
- Add focused tests for configure, switch, settings, and API key redaction.

Acceptance criteria:

- `provider="sglang"` configure request stores and switches through `OpenAICompatBackend`.
- `provider="tensorrt_llm"` configure request stores and switches through `OpenAICompatBackend`.
- Saved configured model IDs are stable and use provider-specific prefixes.
- API responses do not contain raw API keys.
- Existing `vllm` managed tests still pass.

Suggested test command:

```powershell
python -m pytest tests/test_api_chat_models.py tests/test_api_sessions_settings.py tests/test_engine_phase2.py -q
```

### Subagent B: Frontend Settings and Types

Owner type: frontend worker

Files:

- `web/src/lib/api.ts`
- `web/src/app/settings/page.tsx`
- `web/src/lib/i18n.tsx`
- `web/src/app/page.tsx` if chat model picker normalization requires it

Responsibilities:

- Add `sglang` and `tensorrt_llm` to frontend `ModelProvider` typing.
- Update provider normalization so configured models deserialize correctly.
- Add provider options in Settings.
- Add default base URL and model placeholder for each provider.
- Add localized descriptions and notes.
- Ensure notes clearly say these providers are externally managed OpenAI-compatible endpoints.
- Ensure the API key field remains available for deployments that require auth.
- Ensure provider labels render correctly in saved models and active model display.

Acceptance criteria:

- Settings provider selector shows `SGLang` and `TensorRT-LLM`.
- Selecting either provider pre-fills the expected base URL.
- Saving either provider calls `/v1/models/configure` with the correct provider ID.
- Chat and settings model lists show provider labels without falling back to generic `openai_compat`.
- No UI text implies Mochi controls external server startup or engine parameters.

Suggested test command:

```powershell
cd web
npm run type-check
npm run lint
```

### Subagent C: Focused Verification

Owner type: verification worker

Files:

- Prefer tests only.
- Do not change product logic unless a blocking defect is found and reported.

Responsibilities:

- Run targeted backend tests after Subagent A lands.
- Run frontend type-check and lint after Subagent B lands.
- Inspect serialized configured model payloads for provider consistency.
- Confirm `vllm` managed behavior remains isolated to `provider="vllm"`.
- Confirm no raw API keys appear in response bodies.

Acceptance criteria:

- Backend targeted suite passes.
- Frontend type-check and lint pass.
- Any failure is reported with file and line context.
- No unrelated refactors are introduced.

Suggested combined verification:

```powershell
python -m pytest tests/test_api_chat_models.py tests/test_api_sessions_settings.py tests/test_engine_phase2.py -q
cd web
npm run type-check
npm run lint
```

## Main Agent Review Checklist

The main agent is responsible for code review and integration, not for duplicating subagent work.

Review focus:

- Provider IDs are exactly `sglang` and `tensorrt_llm`.
- UI labels are display-only and do not leak into API enum values.
- Both providers route through `OpenAICompatBackend`.
- Both providers preserve `backend_type="openai_compat"`.
- Both providers are external-only.
- `vllm` managed runtime behavior is unchanged.
- API key handling remains redacted.
- Settings payloads preserve active provider identity.
- Chat model switch restores provider, base URL, model, and key state correctly.
- Frontend provider normalization includes the new providers everywhere needed.
- Notes explain external runtime ownership clearly.
- Tests cover both new providers, not just one.

## Implementation Checklist

- [ ] Add `sglang` and `tensorrt_llm` to backend provider literals.
- [ ] Add remote provider defaults for both providers.
- [ ] Update model configure route for both providers.
- [ ] Update model switch route for both providers.
- [ ] Update settings summary provider reporting.
- [ ] Add backend tests for configure and switch.
- [ ] Add backend tests for settings summary and key redaction.
- [ ] Add frontend provider type values.
- [ ] Add frontend provider normalization.
- [ ] Add Settings provider options.
- [ ] Add i18n description and note strings.
- [ ] Confirm chat model picker labels remain correct.
- [ ] Run targeted backend tests.
- [ ] Run frontend type-check and lint.
- [ ] Perform final review against this plan.

## Risk Register

### Risk 1: Accidentally treating provider presets as runtime management

The UI could imply that Mochi controls SGLang or TensorRT-LLM startup parameters.

Mitigation:

- Keep settings limited to `base_url`, `model`, and `api_key`.
- Add clear provider notes.
- Do not add runtime parameter forms.

### Risk 2: Breaking vLLM managed behavior

The existing route has special logic for managed vLLM. Generalizing remote providers too broadly could accidentally route SGLang or TensorRT-LLM into that branch.

Mitigation:

- Keep managed checks explicitly scoped to `provider == "vllm"`.
- Preserve vLLM managed tests.

### Risk 3: Provider enum drift between backend and frontend

Backend may accept a provider that frontend drops during normalization, or frontend may send a provider backend rejects.

Mitigation:

- Update both provider enum surfaces in the same implementation wave.
- Add API and TypeScript checks.

### Risk 4: Secret leakage

New provider tests may miss key redaction.

Mitigation:

- Reuse existing remote provider key handling.
- Add explicit response text assertions for both provider secrets.

### Risk 5: Provider-specific tool or reasoning behavior differs by server startup options

SGLang may require tool parser or reasoning parser startup flags. TensorRT-LLM deployments may expose only a subset of OpenAI-compatible features.

Mitigation:

- Treat this phase as connection preset only.
- Let `OpenAICompatBackend` probing and fallback behavior handle runtime differences.
- Document server startup parser settings as external deployment responsibility in provider notes or future docs.

## Final Acceptance Criteria

- Users can configure an external SGLang endpoint from Settings.
- Users can configure an external TensorRT-LLM endpoint from Settings.
- Both providers appear as distinct saved model providers.
- Both providers use `OpenAICompatBackend` internally.
- No managed runtime controls are added for either provider.
- Existing vLLM managed runtime behavior remains intact.
- Backend targeted tests pass.
- Frontend type-check and lint pass.

## Handoff Notes

- Start with Subagent A and Subagent B in parallel; their write sets are mostly separate.
- Subagent C should run after A and B have produced patches.
- If a worker needs to alter vLLM managed logic, require explicit reviewer approval.
- Do not introduce `sglang` or `tensorrt_llm` managed config classes in this pass.
- Do not add launch commands to runtime code. Launch examples belong in documentation only.
- Keep implementation additive and preserve existing configured model records.
