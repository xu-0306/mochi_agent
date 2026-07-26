# Agent Tool Workflow Final-Gate Evidence Manifest

日期：2026-07-26 17:08:40 +08:00
工作區：`H:\_python\agent_mochi`
base commit：`28f45e156e3b85d004e15942217c0c7a8ff578a2`
工作樹狀態：dirty；本 manifest 綁定的是當下 working tree，不是可直接發布的 commit。

## Backend verification

以下命令均在 base commit 加上當下 working tree 執行，pytest 的 `.pytest_cache`
`WinError 5` 權限警告不影響測試結果：

```powershell
rtk proxy python -m pytest -q tests/test_session_store.py tests/test_tool_workflow_outbox.py tests/integration/api/sessions/test_session_routes.py --basetemp .tmp-codex-fix-final-session-20260726
# 63 passed

rtk proxy python -m pytest -q tests/test_tool_workflow_aggregate.py tests/test_tool_workflow_outbox.py tests/test_tool_workflow_observability.py --basetemp .tmp-codex-fix-final-aggregate-20260726
# 45 passed

rtk proxy python -m pytest -q tests/security/test_approval_lifecycle.py tests/security/test_timeline_approval_continuation.py tests/integration/api/runtime/test_approval_routes.py tests/integration/api/runtime/test_exec_approval_rehydration.py --basetemp .tmp-codex-fix-final-approval-20260726
# 95 passed

rtk proxy python -m pytest -q tests/unit/sessions tests/unit/engine/test_timeline_chat_integration.py tests/test_exec_runtime.py tests/test_exec_tools.py tests/test_execute_code_and_mcp.py --basetemp .tmp-codex-fix-final-timeline-20260726
# 130 passed

rtk proxy python -m pytest -q tests/test_config.py tests/integration/api/sessions/test_settings_routes.py tests/unit/sessions/test_sessions_dir_binding.py tests/unit/agents/test_turn_contract_rollout.py tests/unit/engine/test_turn_contract_rollout.py tests/unit/agents/test_controlled_recovery.py tests/unit/tool_exposure/test_policy_baseline.py --basetemp .tmp-codex-fix-final-rollout-20260726
# 158 passed

rtk proxy python -m pytest -q tests/test_session_store.py tests/unit/sessions/test_sessions_dir_binding.py tests/integration/api/sessions/test_session_routes.py --basetemp .tmp-codex-fix-final-migration-20260726
# 48 passed
```

Additional static checks:

```powershell
rtk proxy python -m compileall -q mochi
rtk git diff --check
rtk proxy rg -n "tool_intent_router|routed_intent|legacy_routed_intent|fallback_keyword" mochi web/src
# no matches
```

## Frontend verification

`web/package.json` currently contains 51 `test:*` scripts.

- The 48 non-browser scripts passed with clean exit in the review pass.
- The three dev-server browser fixtures now run through the shared bounded
  watchdog, completed their product assertions, cleanup phases, and clean exit
  0. The final rerun wall times were 13.4, 14.7, and 12.1 seconds.
- After the three final reruns, no owned `.next-fixture-*` directory remained.
  An earlier before/after process check in the same fix pass also found no new
  Node/CMD process.
- The timeout root cause was test-harness process ownership:
  `npm.cmd` plus `shell: true` did not reliably own the Next Node child. The
  fixtures now execute the Next CLI directly; this was not a product assertion
  failure.

```powershell
rtk proxy node -e "const p=require('./package.json'); console.log(Object.keys(p.scripts).filter(k=>k.startsWith('test:')).length)"
# 51

rtk npm run test:tool-workflow-aggregate
# passed

rtk npm run test:tool-workflow-aggregate-sse
# passed

rtk npm run test:tool-call-card-aggregate
# passed

rtk npm run test:tool-workflow-observability
# passed

rtk proxy npm run test:goal-live-browser-evidence
# assertions complete; clean exit 0

rtk proxy npm run test:subagent-transcript-browser-fixture
# assertion output ok; clean exit 0

rtk proxy npm run test:session-subagent-reload-persistence
# assertion output ok; clean exit 0

rtk npm run type-check
# passed

rtk npm run lint
# 0 errors, 4 warnings

rtk proxy node --check scripts/run-browser-fixture.mjs
rtk proxy node --check scripts/test-goal-live-browser-evidence.mjs
rtk proxy node --check scripts/test-subagent-transcript-browser-fixture.mjs
rtk proxy node --check scripts/test-session-subagent-reload-persistence.mjs
# all passed

rtk proxy powershell -NoProfile -Command "`$env:MOCHI_NEXT_DIST_DIR='.next-codex-fix-build-20260726'; rtk npm run build"
# production build passed
```

Next added the isolated build directory to `tsconfig.json` while building. The
two run-specific generated includes were removed afterward; the six stable
`.next-fixture-*` type paths were retained, and final type-check/lint passed.

## Relevant file SHA-256

These hashes identify the source snapshot used for the evidence above. Paths are
relative to the workspace; the browser fixture files are present in the working
tree even when the surrounding repository ignores them.

```text
mochi/api/routes/sessions.py SHA256=521B5568FA0EBD9B5C16343710DA4DFCE3F94A7798331F7534A9E50B560A949D
mochi/api/routes/chat.py SHA256=9FF7EB7E05DA94E1F9CBA115C526259F762A89EBFA143B795B08110EA59182B2
mochi/api/routes/settings.py SHA256=A74301802321257BE9A312E3E72491E86ECDE774E2AC2419A296CCF7ECF26023
mochi/api/session_store_binding.py SHA256=DCD8359909F0E9C89CB818C27D2B5C7140345CDCD880C4A28095D715A700651E
mochi/runtime/service.py SHA256=9A30AD9D33BD13DAD5882E67011389B2D6AC367D79EC47454D495311FCC22F8F
mochi/api/server.py SHA256=AD0940123B0EF41731FE1827984EB24876A0C0FD25DA38A72F10DC183801FD16
mochi/api/tool_workflow_aggregate.py SHA256=F9BE355E290D6E9EB5AC1740E74AC2BCEA4CEBC8DF1331AC6D5A296D5F6F512C
mochi/api/tool_workflow_observability.py SHA256=E6089C735A0F4CA9FA38246786B48904789EFAAF941B3B16432A0C58F2D9BBF2
mochi/api/tool_workflow_outbox.py SHA256=B39A6C296BE3AE30822CF38A6D8755E173D9DBBB3175FDC92C4263FC9FC8A04D
mochi/utils/streaming.py SHA256=365DA99555E32F59E49D2BBBCF2765C6427EFEE88FCFCEB087C360F003A9225E
tests/unit/sessions/test_sessions_dir_binding.py SHA256=1170032AF8511713C75171ABE3BCC2EB8FE7375D44E54B816EB4D6186B8D16A6
tests/test_tool_workflow_aggregate.py SHA256=3917FEFB0F30870002BC5D220E1989C45C17869CC05E417A239FF8C3052A9FCF
tests/test_tool_workflow_observability.py SHA256=0B3428A8F4D775D4F5FCF6C48753917434E2C69B30A20E9C204778B957C1063C
tests/test_tool_workflow_outbox.py SHA256=4988808CF2250099D859E4D103DBC34A7CBE0E890559EC20C2644573B894CD1C
web/src/lib/sse-frame.ts SHA256=78F5510D181C6876A47EF660E312C3B00A60A4D36911ADB668139B568B975E90
web/src/lib/tool-workflow-aggregate.ts SHA256=AF01F6D5BDDD0C0EACFEFA37B9E2D357BB59E3C1201457F56900D33B3A386487
web/src/lib/tool-workflow-aggregate.test.ts SHA256=37DB22A136DA2D4C54B7A9F4574B658CE07F7417B6F618A8CAE12D90E52503BF
web/src/lib/tool-workflow-aggregate-sse.test.ts SHA256=40CF113EBB305373C79A7BD816C103E8F36F29A0561512B7C476EEF5C038E6B7
web/src/lib/tool-workflow-observability.ts SHA256=B6000D52E888C9D3B5C0658BCF752A41783E36D29AF079E339167704AE24C8B6
web/src/lib/tool-workflow-observability.test.ts SHA256=516E5A026B2F017A45DE86EF30258B29C7D9BA3724217242DC829CE8FF75F333
web/src/components/chat/ToolCallCard.tsx SHA256=34B558D2122A8427D5856E2D7E213A89AE901DFC693D810F48504D08398CF462
web/scripts/test-tool-call-card-aggregate.mjs SHA256=1F5850E9FE1C44C9155C1F8672C1C78C0BCF231D8E53F9A50302BFAAE698BAA2
web/package.json SHA256=EE927CF4F7F6BBBCA32E4187FD06C3D3D4192F9A7793EF64D62A90B3E18553BB
web/next.config.mjs SHA256=42860B47A30B376480080FDC6023E52714FECC173F20C51ACB7D2EC1D335C92A
web/eslint.config.mjs SHA256=99878CA48836D97FC4BFD2B0972243BF22A2D92DE232DF6A342C5B521DABA43C
web/tsconfig.json SHA256=706EFB6D27516425BD04334F4BD301FE32D63A0BFEF5DAE381C6086FDED5CD36
web/scripts/run-browser-fixture.mjs SHA256=E620EB5C6B0CA114C4E2EC795271932D00861ABD8737D4B9F2EF54F19D4CE9C4
web/scripts/test-goal-live-browser-evidence.mjs SHA256=AD86ED4630D8B6BC85B8E2E66775E896B1F114E3048F84F155E52DA89868C9A2
web/scripts/test-subagent-transcript-browser-fixture.mjs SHA256=2E818DF87A2A6AB8889AC7CD2381F18E9E915C6B265C9AC4334E8E8EEEF075D4
web/scripts/test-session-subagent-reload-persistence.mjs SHA256=B076E2212A307B48E28B33DBF59D3ABC0DC2C967ECCC5368322683A7430EC71A
documents/architecture/2026-07-23-agent-tool-workflow-p0-p2-plan.md SHA256=300E823E89ABD5CB382157033768F424F3478E87B18D21C40838C34E49264995
documents/architecture/2026-07-25-tool-workflow-aggregate-stream-replay-rfc.md SHA256=8251571559F63AEF188CF91BFE215D15B985BBC3499BEAB4E37BEB6F0E0D0D1F
docs/superpowers/handoffs/2026-07-25-agent-tool-workflow-scope-completion-handoff.md SHA256=80EBEEDF949F90B73C31F6C8CD849CD84F905C633B2B32F214FE5A5EC6F05C80
docs/superpowers/handoffs/2026-07-26-agent-tool-workflow-handoff-review-findings.md SHA256=D81AB2F12269E0F50FEB13B0564E6D4650C137B758DC30A0D534DCFCB23980B7
```

## Environment notes

- The repository contains pre-existing dirty and ignored WIP/temp directories;
  no reset, checkout, clean, or broad deletion was performed.
- The isolated build output is `web/.next-codex-fix-build-20260726`; it was
  intentionally left in place because this pass did not broadly delete
  pre-existing or generated directories.
- The browser watchdog defaults to 120 seconds, accepts a bounded 30–600 second
  override, forwards the fixture exit code, terminates the owned process tree on
  timeout, and only cleans its declared `.next-fixture-*` directory.
- This P2.3 manifest did not expand or verify P2.2. The later P2.2 closure is
  recorded separately in
  `2026-07-26-p2-2-model-history-linearization-evidence.md`; this historical
  manifest must not be used as the P2.2 source snapshot.
