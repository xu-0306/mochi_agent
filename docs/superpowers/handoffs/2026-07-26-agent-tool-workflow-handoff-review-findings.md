# Agent Tool Workflow 交接文件審查附錄

日期：2026-07-26
工作區：`H:\_python\agent_mochi`
被審查文件：`docs/superpowers/handoffs/2026-07-25-agent-tool-workflow-scope-completion-handoff.md`
審查結論：Request changes
審查範圍：只讀核對文件、production code、測試入口與 targeted tests；審查階段沒有修改 runtime code

## 0. Resolution update (2026-07-26)

本附錄提出的 Finding 1-6 已完成目前可在 dirty working tree 內完成的修正，
resolution 如下：

| Finding | Resolution | Evidence |
|---|---|---|
| 1. standalone SSE scope | Resolved. Session SSE `data:` 現在帶有 `type`、`storage_id`、`session_id`、`turn_id`、`aggregate`、`publication_enabled`、`authoritative`；shared SSE frame parser 與 strict transport parser 已接線。 | `tests/unit/sessions/test_sessions_dir_binding.py`、`web/src/lib/tool-workflow-aggregate-sse.test.ts`；backend 63/45/95/130/158/48 targeted matrices passed。 |
| 2. aggregate test gate | Resolved. `test:tool-workflow-aggregate` 與 `test:tool-workflow-aggregate-sse` 已加入 `web/package.json`；目前 `test:*` count 為 51。 | frontend aggregate、SSE、ToolCallCard、observability scripts passed；48 non-browser scripts 與 3 browser fixtures 均有 clean-exit evidence。 |
| 3. handoff contradictions | Resolved. Current verified state、historical sections、review status 與 next action 已分開；已完成 slices 使用 checked/acceptance wording。 | [scope-completion handoff](H:/_python/agent_mochi/docs/superpowers/handoffs/2026-07-25-agent-tool-workflow-scope-completion-handoff.md)。 |
| 4. Engine-first binding overclaim | Resolved. 文件改為只保證 canonical root/storage identity 與 shared publication gate consistency，不宣稱所有元件共用同一 SessionStore object。 | `mochi/api/session_store_binding.py`、`mochi/runtime/service.py`、session binding tests。 |
| 5. restart semantics | Resolved. Settings PATCH 的 409/not-saved 行為與 external config `pending_restart` 行為已分開記錄。 | `tests/unit/sessions/test_sessions_dir_binding.py`、settings route matrix。 |
| 6. source snapshot evidence | Resolved for working-tree review. Added base commit, timestamp, exact commands, file hashes and environment notes. | [final-gate evidence manifest](H:/_python/agent_mochi/docs/superpowers/handoffs/2026-07-26-agent-tool-workflow-final-gate-evidence-manifest.md)。 |

3 個會啟動 Next dev server 的 browser fixture 已改由 bounded watchdog runner
管理，並在 current snapshot 分別於 12.3、13.4、10.9 秒 clean exit 0；執行後
沒有新的 Node/CMD process 或受控 `.next-fixture-*` 目錄殘留。先前 timeout
根因是 `npm.cmd`/`shell: true` 沒有可靠擁有 Next Node process，屬測試 harness
process-ownership 問題，而非產品 assertion 失敗。P2.2 不屬於本 review-fix
範圍；它後續已由獨立 acceptance matrix 核實完成，證據見
`2026-07-26-p2-2-model-history-linearization-evidence.md`。

## 1. 這份附錄的用途

原交接文件已被後續 agent 更新為「P2.3 Phase 0-4 與 final rollout gate 已完成」。本次審查沒有沿用 2026-07-25 的舊狀態，而是重新核對目前工作樹。

結論不是推翻已完成的 aggregate／storage／frontend 工作，而是指出：

1. standalone workflow SSE 的 server payload 與 exported frontend client parser 不相容。
2. 49-script final gate 沒有執行 aggregate parser/store 的單元測試。
3. 原交接文件同時保留「已完成」與「待實作」的互相矛盾段落。
4. `Engine-first SessionStore binding` 的描述超過目前實作保證。
5. `sessions_dir` 的 API reject 與 external-config pending-restart 語義混在一起。
6. final-gate evidence 沒有綁定穩定 commit 或 working-tree fingerprint。

下一位 agent 應先修正第 1、2 項並補 contract tests，再整理交接文件。不要重做 reducer、approval store、SessionStore v2 或 ToolCallCard aggregate projection。

## 2. 審查方法：這些問題是怎麼找出的

### 2.1 先把文件的完成宣稱轉成可核對的 contract

從原交接文件開頭與第 7-10 節整理出主要宣稱：

- startup-only `sessions_dir` 已完成。
- storage marker／`storage_id` scope 已完成。
- snapshot／range／named SSE／`Last-Event-ID` 已完成。
- frontend 有單一 strict aggregate parser/store。
- `ToolCallCard` 使用 aggregate 作 workflow status authority。
- 49 個 frontend `test:*` scripts 與 backend final gate 已通過。

接著不只搜尋檔名，而是沿資料流逐層核對：

```text
backend route
-> SSE encoder payload
-> frontend SSE frame reader
-> transport parser
-> aggregate store
-> ToolCallCard projection
```

### 2.2 以 line-number search 找出實際接線

使用的只讀命令：

```powershell
rtk proxy rg -n "storage_id|tool_workflow_aggregate|Last-Event-ID|last-event-id|after_seq" mochi/api tests
rtk proxy rg -n "tool-workflow-aggregate|ToolWorkflowAggregate|storageId|lastEventId|aggregate" web/src web/scripts web/package.json
rtk proxy rg -n "resolve_route_session_store|session_store_binding" mochi
rtk proxy rg -n "sessions_dir_restart_required|pending_restart|ensure_sessions_dir_unchanged" mochi tests
```

### 2.3 比較 test file 與 package scripts，而不是只相信 script count

使用 PowerShell 計算 `test:*` 數量：

```powershell
rtk proxy powershell -NoProfile -Command "((Get-Content -LiteralPath 'web/package.json' -Raw -Encoding UTF8 | ConvertFrom-Json).scripts.PSObject.Properties.Name | Where-Object { `$_.StartsWith('test:') }).Count"
```

結果為 49。之後再搜尋 `tool-workflow-aggregate.test.ts` 是否有對應 script，發現沒有。

### 2.4 執行 targeted checks 驗證目前實作，而不是只做靜態猜測

Backend：

```powershell
rtk proxy python -m pytest -q tests/unit/sessions/test_sessions_dir_binding.py tests/test_tool_workflow_aggregate.py tests/test_tool_workflow_outbox.py --basetemp .tmp-review-handoff-core
```

結果：`53 passed`；另有本機 `.pytest_cache` 權限警告，不影響測試結果。

Frontend aggregate unit：

```powershell
rtk proxy node --experimental-strip-types ./src/lib/tool-workflow-aggregate.test.ts
```

結果：`tool workflow aggregate tests passed`。

ToolCallCard aggregate script：

```powershell
rtk npm run test:tool-call-card-aggregate
```

結果：`tool call card aggregate assertions passed`。

這證明主要 aggregate store 與 ToolCallCard 測試目前能通過；問題是正式 49-script gate 沒有包含 aggregate unit test，而且 standalone SSE payload contract 仍不相容。

## 3. Findings

## Finding 1 - P1：standalone aggregate SSE transport contract 不相容

### 文件宣稱

原交接文件宣稱：

- named aggregate SSE 與 `Last-Event-ID` replay 已接線。
- frontend strict aggregate store 可處理 reconnect／repair。
- Phase 3/4 delivery final gate 已完成。

### 實際資料流

Backend session workflow SSE：

- `mochi/api/routes/sessions.py:654-708`
- `_sse_data` 在 `mochi/api/routes/sessions.py:694-697`

目前 event data：

```python
{
    "storage_id": store.storage_id,
    "aggregate": aggregate,
}
```

它沒有 transport 頂層的 `session_id` 與 `turn_id`。

Frontend frame reader：

- `web/src/lib/api.ts:1568-1625`

`readSseStream()` 只會在 payload 上補：

```ts
{ ...parsed, type: 'tool_workflow_aggregate' }
```

它不會從 URL 或 aggregate 內部補 transport scope。

Frontend strict transport parser：

- `web/src/lib/tool-workflow-aggregate.ts:395-410`

它明確要求：

```ts
storage_id
session_id
turn_id
```

Exported standalone client：

- `web/src/lib/api.ts:2315-2340`

它把 `readSseStream()` 產生的 event 直接送入：

```ts
parseToolWorkflowAggregateTransport(event)
```

因此第一個 SSE aggregate event 就會因 transport scope 不完整而失敗。

### 最小重現

在 `web` 目錄：

```powershell
rtk proxy node --experimental-strip-types -e "import { parseToolWorkflowAggregateTransport } from './src/lib/tool-workflow-aggregate.ts'; try { parseToolWorkflowAggregateTransport({ type: 'tool_workflow_aggregate', storage_id: 'storage:v1:test', aggregate: null }); console.log('unexpected-pass'); } catch (error) { console.log(error.message); }"
```

實際輸出：

```text
tool workflow aggregate transport scope is invalid
```

### 為什麼既有測試沒有發現

`tests/unit/sessions/test_sessions_dir_binding.py` 的 SSE assertion 只核對 response text 包含：

- `event: tool_workflow_aggregate`
- `storage_id`

它沒有把真實 server SSE frame 交給 frontend `readSseStream()` 與 `parseToolWorkflowAggregateTransport()`。

### 影響

- Chat 主 SSE 可能仍可工作，因為 `mochi/api/routes/chat.py` 發送的是包含 session／turn scope 的 snapshot transport。
- 但 `/sessions/{session_id}/turns/{turn_id}/tool-workflow/stream` 對應的 exported frontend helper 不是端到端可用 contract。
- 文件不能在此狀態下宣稱 standalone named SSE + reconnect client 完整。

### 建議修正

推薦修 server payload，保持 frontend parser strict：

```python
"_sse_data": {
    "type": "tool_workflow_aggregate",
    "storage_id": store.storage_id,
    "session_id": session_id,
    "turn_id": turn_id,
    "aggregate": aggregate,
    "publication_enabled": outbox.enabled,
    "authoritative": outbox.enabled,
}
```

替代方案是由 `streamToolWorkflowAggregates()` 把函式參數中的 session／turn scope 注入 event 後再 parse；這可修 client，但 server payload 仍與 snapshot／range transport contract 不一致，因此不是首選。

### 必要驗收

1. Backend test 解碼 SSE `data:` JSON，驗證完整 transport scope。
2. Frontend test 用真實 server-shaped frame 經過 `readSseStream()`。
3. 將解析結果送進 `parseToolWorkflowAggregateTransport()`，必須成功。
4. `Last-Event-ID` replay 後第一筆 seq 正確，storage mismatch 仍回 409。
5. Chat SSE 與 standalone session SSE 的 aggregate transport shape 保持一致。

## Finding 2 - P1：49-script final gate 未執行 aggregate parser/store unit test

### 證據

存在測試檔：

- `web/src/lib/tool-workflow-aggregate.test.ts`

它實際測試：

- strict schema rejection
- exact duplicate
- sequence gap
- range repair
- conflicting duplicate
- unsupported payload 保留 last-known-good
- storage scope change

但是 `web/package.json` 沒有：

```json
"test:tool-workflow-aggregate": "node --experimental-strip-types ./src/lib/tool-workflow-aggregate.test.ts"
```

`test:tool-call-card-aggregate` 只檢查 ToolCallCard 接線與投影斷言，不能取代 store state-machine unit test。

### 影響

- 文件的「49 個 scripts 已涵蓋 aggregate final gate」不完整。
- 未來 aggregate store regression 不會被完整 script runner 發現。
- Finding 1 的 server/client mismatch 也沒有 end-to-end contract test 可攔截。

### 建議修正

在 `web/package.json` 加入：

```json
"test:tool-workflow-aggregate": "node --experimental-strip-types ./src/lib/tool-workflow-aggregate.test.ts"
```

並把 Finding 1 的 SSE contract test 放進同一 script 或新增：

```json
"test:tool-workflow-aggregate-sse": "node --experimental-strip-types ./src/lib/tool-workflow-aggregate-sse.test.ts"
```

完成後 final gate script count 會改變；文件不可繼續寫死 49，除非重新計數並重跑。

### 必要驗收

```powershell
rtk npm run test:tool-workflow-aggregate
rtk npm run test:tool-call-card-aggregate
```

完整 `test:*` runner 必須包含兩者，且失敗時整體 exit code 非 0。

## Finding 3 - P1：交接文件把已完成工作與待實作指令混在一起

### 矛盾位置

文件開頭與 continuation update 宣稱：

- Phase 3/4 已完成。
- Slice 1-5 已完成。
- 下一位 agent 應只審查，不要重建。

但後段仍保留：

- 第 6.1 節以「目前」描述 `AgentEngine.apply_config()` 會直接切 root。
- 第 8.1-8.5 節用 imperative future tense 要下一位 agent 實作已完成 slices。
- 第 8 節標題仍是「固定實作順序」，不是「歷史實作與驗收契約」。

目前 production code 已在 `AgentEngine.apply_config()` 的第一個 live mutation 前呼叫 `ensure_sessions_dir_unchanged()`，所以第 6.1 節的「目前」描述已過期。

### 影響

- 下一位 agent 可能照第 8 節重做已完成內容。
- 審查者無法分辨哪些是 historical plan、哪些是 active work。
- 發現 Finding 1 後，也不容易在文件中標出唯一剩餘 blocker。

### 建議修正

把原交接重整成單一時間狀態：

1. Current verified state
2. Retained invariants
3. Review findings／known open issues
4. Historical rejected designs
5. Reproduction and validation commands
6. Next exact action

第 8.1-8.5 節可保留內容，但必須改名為「已完成 slices 的 acceptance contract」，並全部改成 past tense／checked checklist。第 6.1 節應寫成「修正前風險」而不是「目前風險」。

在 Finding 1、2 關閉以前，文件頂端狀態應寫：

```text
Phase 3/4 implementation substantially complete;
standalone SSE transport contract and frontend gate integration require fixes.
```

不要寫成無條件 final gate complete。

## Finding 4 - P2：Engine-first binding 描述超過實作保證

### 文件宣稱

原交接第 4.5 節寫：

```text
Engine、Runtime 與 routes 的 Engine-first SessionStore binding 集中入口：
mochi/api/session_store_binding.py
```

### 實際程式

`resolve_route_session_store()` 的 call sites只有 API routes：

- `mochi/api/routes/chat.py`
- `mochi/api/routes/sessions.py`
- `mochi/api/routes/approvals.py`
- `mochi/api/routes/file_ops.py`
- `mochi/api/routes/workspace.py`

`RuntimeService.bind_app_config()` 沒有呼叫此 helper。它在：

- `mochi/runtime/service.py:636-648`

使用順序是：

```text
injected ordinary_chat_session_store
-> engine._session_store
-> new SessionStore(config.sessions_dir)
```

它會驗證 canonical sessions root，並綁定 shared publication gate，因此 storage consistency 方向是合理的；但不是「所有元件共用同一個 Engine-owned SessionStore object」。

### 影響

若下一位 agent 相信 object identity 已統一，可能跳過：

- injected store 與 engine store 的 root/storage_id 核對
- publication gate identity 核對
- RuntimeService outbox binding regression

### 建議修正文案

改成：

```text
API routes 使用 Engine-first route resolver；RuntimeService 可使用 injected
SessionStore，但 bind 時必須驗證相同 canonical sessions root/storage_id，並綁定
相同 ToolWorkflowPublicationGate。系統保證 storage-root 與 gate consistency，
不保證所有元件持有同一個 SessionStore object。
```

### 必要驗收

- injected store root 不同時 fail before binding mutation。
- root 相同但 store instance 不同時，storage_id 相同。
- Engine 與 Runtime outbox 使用同一 publication gate。
- flag true -> false barrier 同時約束兩個 store instance 的 strict writes。

## Finding 5 - P2：sessions_dir 的「重啟後生效」語義混淆兩條流程

### 實際有兩種行為

#### A. Settings API PATCH

`mochi/api/routes/settings.py::_preflight_sessions_dir()`：

- root 不同時回 `409 sessions_dir_restart_required`
- 在建立新目錄、persist config、更新 app state 或呼叫 Engine 前拒絕
- request 不會被 staged 或持久化

因此：

```text
PATCH rejected -> restart alone does not apply requested root
```

使用者必須改用外部 config edit 或其他明確的 restart-staging 機制。

#### B. External config file change

`mochi/api/server.py::_allow_external_config_reload()`：

- disk config 已有新 root
- live app config 保留舊 root
- `config_reload_status` 顯示 `pending_restart`
- 重啟後 startup 才讀新 root

因此：

```text
external config persisted -> pending_restart -> restart applies new root
```

### 影響

原交接把「runtime reject、不 persist、重啟後生效」寫在同一句，會讓使用者以為被 409 拒絕的 PATCH 已經排入重啟套用。

### 建議修正

文件與 UI 明確分開兩種訊息：

API PATCH：

```text
This setting is startup-only. The requested change was not saved.
Edit the configuration file and restart the application.
```

External config reload：

```text
The configuration file contains a new sessions directory.
The running process still uses the previous directory. Restart to apply it.
```

如果產品希望 WebGUI 能設定「下次啟動的 root」，那是新的 staging feature，必須有 persisted pending config、取消／覆寫與 revision semantics；不能把目前 409 response 說成已 staging。

## Finding 6 - P2：final-gate evidence 沒有綁定穩定 source snapshot

### 審查時狀態

```text
HEAD: 28f45e1 feat: complete protected workspace hardening rollout
```

以下關鍵檔案仍在 dirty/untracked working tree：

- modified：`mochi/api/routes/sessions.py`
- untracked：`web/src/lib/tool-workflow-aggregate.ts`
- untracked：`web/src/lib/tool-workflow-aggregate.test.ts`
- untracked：原交接文件

原交接有記錄 pass count 與命令，但沒有 commit SHA + relevant diff fingerprint，無法證明後續看到的 dirty worktree 與當時跑 final gate 的內容完全相同。

### 影響

- 任何後續 agent 修改共享工作樹後，歷史 pass count 就失去可追溯性。
- 「final gate complete」只能理解成某次 working-tree observation，不是可重建的 release candidate。

### 建議修正

若暫時不能 commit，至少在 handoff 附一個 evidence manifest：

```text
base_commit
timestamp/timezone
exact commands
relevant file list
SHA-256 for each relevant file, or one normalized diff fingerprint
test output artifact paths
environment notes
```

正式交付最好建立可識別 commit／branch，再在該 snapshot 重跑 final gate。

## 4. 建議修正順序

只按以下順序處理，避免再次擴張：

1. 修正 standalone SSE payload scope。
2. 增加 server-frame -> frontend parser contract test。
3. 將 aggregate unit test加入 `web/package.json` final gate。
4. 重跑 aggregate backend、frontend parser/store、ToolCallCard 與 SSE reconnect targeted matrix。
5. 更新原交接文件，移除已完成／待實作矛盾。
6. 更正 binding 與 restart semantics 文案。
7. 產生 evidence manifest 或 commit-bound final gate。

不要在這一輪：

- 重寫 reducer。
- 新增第二個 aggregate store。
- 重做 SessionStore v2 allocator。
- 實作完整 sessions_dir hot-switch。
- 擴張 P2.2 model-history linearization。
- 修改與本 review findings 無關的 dirty WIP。

## 5. 修正後的最小驗收矩陣

### Backend

```powershell
rtk proxy python -m pytest -q tests/unit/sessions/test_sessions_dir_binding.py tests/test_tool_workflow_aggregate.py tests/test_tool_workflow_outbox.py --basetemp .tmp-review-fix-backend
```

需新增 assertion：SSE data transport 包含 `type/storage_id/session_id/turn_id/aggregate/publication_enabled/authoritative`。

### Frontend

```powershell
rtk npm run test:tool-workflow-aggregate
rtk npm run test:tool-call-card-aggregate
rtk npm run type-check
rtk npm run lint
```

### Contract E2E

至少一個測試必須：

1. 從 backend 測試 fixture 取得真實 SSE frame。
2. 經過與 `readSseStream()` 相同的 frame parsing。
3. 送入 `parseToolWorkflowAggregateTransport()`。
4. apply 到 aggregate store。
5. 驗證 cursor、seq、storage scope 與 last-known-good state。

### Static／documentation

```powershell
rtk git diff --check
rtk proxy rg -n "目前 .*apply_config|先寫測試，再實作|Slice 1 至 Slice 5 已完成" docs/superpowers/handoffs/2026-07-25-agent-tool-workflow-scope-completion-handoff.md
```

審查目標不是讓最後一個 `rg` 必然零結果，而是逐一確認所有命中都位於明確標示的 historical section，不再與 current status 混淆。

## 6. 下一位 agent 的建議開場指令

```text
先讀取：
1. AGENTS.md
2. docs/superpowers/handoffs/2026-07-25-agent-tool-workflow-scope-completion-handoff.md
3. docs/superpowers/handoffs/2026-07-26-agent-tool-workflow-handoff-review-findings.md

只修 review findings，不重建已完成架構。第一步先修 standalone session
tool-workflow SSE transport：server data 必須提供 frontend strict parser 所需的
storage_id/session_id/turn_id scope，並增加 server-frame 到 frontend parser 的
contract test。接著將 tool-workflow-aggregate.test.ts 加入 package.json 的
test:* gate，再整理原 handoff 的矛盾段落與 binding/restart wording。

保留 dirty worktree，不得 reset/checkout/clean。所有 shell commands 經 rtk。
修正完成後，用 commit 或 working-tree fingerprint 綁定 final-gate evidence。
```

## 7. 本次審查沒有做的事

- 沒有修改 runtime code。
- 沒有修正 SSE payload。
- 沒有新增 package script。
- 沒有重寫原交接的歷史段落，只在原文開頭加入本附錄連結。
- 沒有宣稱 P2.2 full same-session model-history linearization 完成。
