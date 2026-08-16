# Mochi Agent 成熟度缺口與補強計畫

> 日期：2026-08-03\
> 分析基線：`main` / `f45a0c63`，核心 runtime 能力累積至 `d72ac72a`\
> 範圍：`mochi/` runtime、`web/` WebGUI、測試、CI 與運維文件\
> 定位：分析 Mochi 與成熟 Agent 專案的差距，只列出尚未具備、部分成熟或仍缺 qualification 的能力。

## 1. 文件目的

Mochi 已具備完整 Agent Runtime 的主要骨架。本文件不重新列舉既有功能，也不把已完成能力包裝成新的實作工作，而是回答三個問題：

1. 哪些能力目前仍然缺失？
2. 哪些能力已有基礎，但尚未達到成熟 Agent 專案要求的完整契約？
3. 哪些能力已實作，但缺少跨 provider、重啟、瀏覽器或 release 等級的 qualification？

缺口分為三類：

- **缺失**：目前沒有對應產品能力或 runtime contract。
- **部分成熟**：已有主要實作，但邊界、持久化、恢復或跨層語義仍不完整。
- **Qualification 缺口**：功能存在，但尚未以一致測試矩陣證明可在所有宣告支援的環境可靠運作。

## 2. 已確認不是缺口的能力

以下能力已存在，不應再作為從零開發工作包：

- AgentEngine、ReAct、multi-backend routing 與 provider/model capability handling。
- Durable session history、approval checkpoint、tool workflow 與 adaptive runtime。
- JSON 與 Qwen XML-ish tool-call parser，包括單行、多 function 與基本參數 coercion。
- final text／thinking 中可解析 tool markup 的 rescue 流程。
- workspace tool discoverability、`tool_activate` broker、write-required capability preservation，以及 approval／sandbox／policy 分層。
- Goal follow-up guidance，以及最近 code block、artifact reference 與截斷狀態 carry-over。
- Chat server-side cancel endpoint、前端 Stop、active run registry，以及部分 model/tool 的真實 cancellation。
- Goal startup supervision、heartbeat、checkpoint、stalled／awaiting-resources recovery 與基本 run policy。
- 多檔案 diff preview、可編輯 patch preview、digest-bound approval、衝突安全的 authoritative undo。
- MemoryStore SQLite／FTS5、memory save/search/update/delete/export、SkillLibrary version 欄位與 failure-learning worker。
- Linux bubblewrap sandbox；Windows 原生 containment 仍是明確 deferred roadmap。

## 3. 架構原則

- Runtime core 定義穩定契約；Ollama、OpenAI-compatible、GGUF、Safetensors、WebGUI 與 CLI 僅作為 adapters 映射契約。
- 不一次性重寫 `service.py`、`engine.py`、`react_loop.py` 或 `page.tsx`；拆分前先補 characterization tests。
- 不用永久擴大 tool schema 暴露解決能力可見性；沿用 discoverability、activation、concrete-call authorization 的既有分層。
- 任何選擇性 patch 批准都必須產生新的 manifest、digest 與 approval，不得修改已批准內容。
- Release gate 採分層執行，不要求每個 PR 連線所有真實 provider，也不以單次本機手動成功代替可重現證據。
- 已有能力的工作只允許補齊缺口、qualification 或一致性，不重複建立平行實作。

## 4. 缺口總覽

| ID | 優先級 | 類型 | 缺口 |
|---|---|---|---|
| `BASE-01` | P0 | Qualification | 缺少與目前 HEAD、既有 WIP、已知環境限制綁定的自動化基線與失敗清單。 |
| `ERR-01` | P0 | 部分成熟 | 缺少跨 backend、runtime、SSE、持久化、UI、telemetry 的中央 failure contract。 |
| `CTX-01` | P0 | 部分成熟 | Effective context 已有實作，但尚未形成跨 provider、可持久化的單一契約。 |
| `GEN-01` | P0 | 部分成熟 | 輸出截斷已有一次 continuation，仍缺 bounded policy、合併安全性與跨 provider qualification。 |
| `TOOL-01` | P0 | 部分成熟 | 可解析 markup 已能 rescue；無法解析但疑似 tool markup 的輸出仍缺明確失敗語義。 |
| `SESSION-01` | P1 | 缺失 | 缺少跨 session 全文搜尋、篩選與原始 turn 定位。 |
| `CANCEL-01` | P1 | 部分成熟 | Cancellation 尚未覆蓋所有 disconnect、backend、工具、scheduler 與 restart 場景。 |
| `CTX-02` | P1 | 部分成熟 | Context meter 已存在，但缺 post-response snapshot 與 durable compaction lifecycle。 |
| `ERR-02` | P1 | 部分成熟 | WebGUI 尚未完整區分 steering、tool failure、denial、cancel、truncation 等語義。 |
| `FILE-01` | P1 | 缺失 | 缺少檔案級／行級選擇性批准與安全的 subset re-planning。 |
| `FILE-02` | P1 | 部分成熟 | 缺少 partial apply、複合 patch conflict 與失敗後重新規劃的完整產品流程。 |
| `UI-01` | P1 | Qualification | 缺少完整 browser interaction 與 visual regression gate。 |
| `SUP-01` | P1 | 部分成熟 | Goal 可恢復，但一般 AgentRun、worker 與未投影事件的 startup adoption 尚未完整。 |
| `EXEC-01` | P1 | 部分成熟 | Detached exec 缺少 durable job table 與跨程序 reattach。 |
| `RESOURCE-01` | P1 | 部分成熟 | Idle、model-call、token/cost、concurrency、queue/backpressure 等資源政策尚未完整執行。 |
| `MEM-01` | P2 | 部分成熟 | 缺少 session／durable memory 分層、provenance、品質回饋與 consolidation contract。 |
| `MEM-02` | P2 | 缺失 | 缺少 memory backup／restore／import 與可驗證的 migration 流程。 |
| `SKILL-01` | P2 | 部分成熟 | Skill 有版本欄位，但缺少正式 rollback 與版本品質比較。 |
| `PROVIDER-01` | P1 | Qualification | 缺少所有宣告 provider 共用的 runtime contract qualification matrix。 |
| `REL-01` | P0 | 缺失 | 缺少涵蓋 Python、Web、browser、sandbox、migration、provider 的分層 release gate。 |

## 5. P0：核心 Runtime 契約缺口

### `BASE-01`：可重現基線

目前狀態：

- 測試套件已有 marker 與大型離線通過紀錄。
- 現有工作樹包含 Linux sandbox 使用者 WIP，不能被後續工作覆蓋。
- 最新 provider 修正以 focused tests 與實際 browser smoke 驗證，未重新執行完整 suite。

缺口：

- 沒有一份與當前 HEAD 綁定的 required／conditional／environment-blocked 測試矩陣。
- 沒有固定記錄已知失敗、waiver、到期日與責任範圍的機制。
- 缺少供後續缺口共用的 provider、disconnect、restart、malformed input fixtures。

驗收：

- 產出機器可讀的 baseline manifest。
- 每個 gate 能區分 regression、環境不支援與條件式 qualification。
- 不修改或清理使用者既有 WIP。

### `ERR-01`：中央 Failure Contract

目前狀態：

- ReAct 已在 metadata 使用 `runtime_category`、`error_type`、`recoverability`。
- 事件仍分散在 `StatusEvent`、`AssistantTruncatedEvent`、`ToolCallResultEvent`、`ErrorEvent` 與 `FinalAnswerEvent`。
- WebGUI 多數錯誤呈現仍依 event shape 或 HTTP diagnostics 判斷。

缺口：

- 缺少中央型別或 schema，統一 failure kind、origin、recoverability、retry policy、model-context injection、telemetry key 與 UI presentation hint。
- Backend、ReAct、Goal／AgentRun 與 WebGUI 尚可能對同一失敗使用不同名稱。
- 缺少 schema versioning 與舊事件 migration／compatibility 規則。

最低必要分類：

- `backend_error`
- `tool_error`
- `tool_denied`
- `runtime_steering`
- `context_overflow`
- `output_truncated`
- `empty_response`
- `invalid_tool_call`
- `cancelled`

驗收：

- 同一失敗在 runtime event、SSE、持久化、timeline、WebGUI 與 telemetry 使用同一 canonical kind。
- 每一類明確定義是否可重試、是否注入模型 context、是否為 terminal。
- 舊 session event 仍可讀取並映射到新 schema。

### `CTX-01`：Effective Context 契約收斂

目前狀態：

- Ollama 已追蹤 configured、runtime、model maximum 與 effective context metadata。
- AgentEngine 已使用可信 context hint、output reserve 與 preflight hard gate。
- ContextManager 已能 compaction，WebGUI 已能 preview 下一輪 context budget。

缺口：

- `model_advertised_context`、`serving_context`、`configured_context`、`effective_context` 尚未成為所有 backend 共用的正式型別。
- 部分 backend 仍只能提供 fallback 或近似值，runtime 缺少一致 confidence/source contract。
- Compaction 結果主要是 process-local derived state，重啟後缺少完整恢復契約。
- 缺少「模型宣告 32768、實際 serving 4096」的跨 backend regression fixture。

驗收：

- 所有 backend 以同一 contract 回報 context value、source 與 reliability。
- Prompt budget、API snapshot、UI meter 與實際 request 使用同一 effective value。
- 無法取得可靠上限時採保守策略，不將不可信 metadata 當 hard guarantee。
- Compaction 後重啟不會恢復成未壓縮 prompt state。

### `GEN-01`：輸出截斷成熟化

目前狀態：

- ReAct 已辨識 `finish_reason=length`，最多 continuation 一次。
- 已有 `assistant_truncated`、`output_truncated` 與 recovered／partial metadata。
- 截斷的 tool markup 可在 continuation 後重新解析。

缺口：

- 尚未對所有 backend 統一 `length`、`max_tokens`、`response.incomplete` 等 terminal signal。
- Continuation 缺少整輪總 output budget、重複片段偵測與安全合併規則。
- 是否再次 continuation 或先 compaction 仍是固定流程，而非基於剩餘 context 與 provider capability 的 policy。
- 缺少真實 provider qualification 與 streaming／non-streaming 對稱測試。

驗收：

- 不論 backend terminal signal 為何，截斷都不會標成正常完成。
- Recovery 受明確次數、token 與時間上限約束。
- 合併結果不得重複前綴或破壞 tool-call markup。
- Recovery 失敗時保留可閱讀 partial output 並回 canonical `output_truncated`。

### `TOOL-01`：Malformed Tool Markup 終止語義

目前狀態：

- JSON 與 Qwen XML-ish parser chain 已存在。
- 可解析的 final text／thinking markup 已能返回正常工具流程。
- Invalid native tool turn 已有一次 repair 基礎。

缺口：

- 含 `<tool_call>`／`<function=` 特徵但無法解析的輸出，仍缺少獨立偵測與 canonical `invalid_tool_call`。
- 缺少 malformed、混合可見文字、截斷 markup、未知工具與多 tool 部分損壞的完整 fixture。
- Parser repair 與 native invalid-turn repair 尚未共用一致 retry budget。

驗收：

- 疑似 tool markup 不得未經判斷直接成為正常 final answer。
- 最多進行一次格式 repair；repair 後仍無效則 terminal `invalid_tool_call`。
- 原始 markup 可保留於受控 diagnostics，但不洩漏到一般回答 UI。

## 6. P1：成熟互動能力缺口

### `SESSION-01`：跨 Session Search

缺口：

- 缺少 session title、user／assistant message、tool result preview、artifact metadata 的統一索引。
- 缺少依 session、時間、agent、tool、Goal／AgentRun 的篩選。
- 搜尋結果無法直接定位並跳回原始 turn。
- 大型 tool result 與 binary artifact 尚未定義索引摘要／reference contract。

驗收：

- 可用關鍵字找回舊決策、工具結果與檔案修改來源。
- 結果包含 session、turn、時間、來源與可開啟 reference。
- Binary 與大型 payload 不進入全文索引，只索引安全摘要與 artifact reference。

### `CANCEL-01`：Cancellation Capability Matrix

目前狀態：

- Chat 已有 server-side cancellation 與前端 Stop。
- Streaming teardown 已會關閉 worker。
- Delegated model invocation，以及 exec、code、web fetch/crawl/search 已有真實 cancellation slice。

缺口：

- Client disconnect 到 server-side run cancellation 尚缺完整 E2E 證據。
- Local GGUF／Safetensors 仍可能只能延遲取消或等待安全點。
- 其他長時間工具、scheduler worker 與 detached exec 尚未全部宣告並驗證 cancellation capability。
- Restart 後 cancellation 狀態與未完成 tool commit 的一致性仍未完整驗證。

驗收：

- 每個 backend／工具明確標記 `immediate`、`deferred` 或 `unsupported`。
- Chat、Goal、subagent、長工具與 disconnect 各有 integration test。
- Cancellation 後不留背景 worker、不重複提交工具、不產生假 `cancelled` 成功事件。

### `CTX-02`：Durable Context Lifecycle

目前狀態：

- UI 已顯示 context size、prompt estimate、output reserve、remaining tokens 與 compaction 狀態。
- Runtime 可在 prompt 組裝前產生 context snapshot。

缺口：

- Model response 後沒有一致更新 snapshot 的生命週期。
- Compaction 次數、摘要 revision、前後 token 變化與 overflow reason 尚未完整持久化。
- UI preview、實際 prompt 與重啟後狀態仍可能來自不同 derived state。

驗收：

- Prompt 前、compaction 後、response 後皆留下同 schema snapshot。
- Snapshot revision 與實際 model invocation 可互相對應。
- 重啟後 UI meter、event log 與 prompt budget 一致。

### `ERR-02`：錯誤產品語義

依賴：`ERR-01`。

缺口：

- Steering、denial、tool failure、backend failure、cancel、truncation 仍可能使用相近的紅色失敗呈現。
- 缺少依 recoverability 提供 retry、resume、request approval、reduce context 等精確行動。
- Trace、execution timeline、Goal／AgentRun 與 Chat 尚未完全共用 presentation policy。

驗收：

- UI 不再用 tool error 表示 runtime steering。
- 每種 canonical failure kind 有固定 tone、標題、細節與可用 action。
- 同一事件在 Chat、Goal 與 AgentRun 顯示一致。

## 7. P1：檔案工作流缺口

### `FILE-01`：選擇性批准

目前狀態：

- 多檔案 patch 可產生 immutable preview 與 digest-bound approval。
- 使用者可修改 patch，重新 preview 後取得替代 approval。
- Applied change 已有 server-authoritative undo。

缺口：

- 缺少檔案級 subset approval。
- 行級／hunk 級批准尚未建模 dependency group 與重算 patch 的安全規則。
- 缺少「選取子集後重新產生 manifest、digest、approval」的正式流程。

驗收：

- 選擇子集一定建立新 change request，不沿用舊 approval。
- Dependency group 不得被不安全拆分。
- 未選取檔案保持原樣，且 approval／audit 能追溯原始提案與新子集。

### `FILE-02`：Partial Apply 與 Conflict Recovery

缺口：

- 複合 patch 的部分失敗目前缺少完整使用者導向的重新規劃流程。
- Patch conflict、外部檔案變更、binary／大檔案與 undo conflict 尚未統一 recovery action。
- Session history 尚未完整表達 proposed、approved、applied、partially-applied、replanned、undone 的生命週期。

驗收：

- 失敗前後原檔案與已提交前綴的狀態可被準確判斷。
- 任何重試都重新驗證 identity、content digest 與 authorization envelope。
- UI 提供重新 preview、重新批准或安全放棄，不以模糊 retry 重送舊 patch。

### `UI-01`：Browser 與 Visual Regression

缺口：

- 現有 source-level／fixture scripts 尚未形成完整 required browser matrix。
- 缺少 diff、approval、edited patch、undo、conflict、mobile layout 與 reload persistence 的視覺基線。
- 缺少對 overflow、keyboard navigation、focus 與長路徑／大 diff 的互動驗證。

驗收：

- 核心檔案工作流有 headless browser interaction tests。
- 關鍵 desktop／mobile 狀態有可審核 screenshot artifact。
- Visual change 必須經明確 baseline update，而不是靜默覆蓋。

## 8. P1：Restart-safe Supervisor 與資源缺口

### `SUP-01`：一般 AgentRun Startup Adoption

目前狀態：

- Goal supervisor 已能在 startup 執行 reconciliation、lease heartbeat 與 checkpoint-based recovery。
- Stalled、awaiting-resources、waiting-approval 與 context handoff 已有部分恢復流程。

缺口：

- 非 Goal 綁定的 active AgentRun 尚缺完整 startup adoption。
- Dead／stale worker、完成但未投影事件、orphan approval 與重複 side effect 尚未形成單一 reconciler contract。
- 缺少多次重啟的 idempotency qualification。

驗收：

- Startup 能將每個 active run 收斂到 running、recoverable、terminal 或 operator-required。
- 重複執行 reconciler 不會重複工具、事件或 approval side effect。
- 完成但未投影事件可補投影且保持 monotonic sequence。

### `EXEC-01`：Durable Detached Exec

目前狀態：

- Detached exec 已保存 lease metadata、log path、checkpoint path 與 run-scoped lookup／stop。
- Live process reattach 仍依賴 process-local `ExecRuntime` state。

缺口：

- 缺少 durable job table、process identity、owner lease 與 restart-safe rebind。
- Runtime restart 後無法可靠區分 still-running、exited、orphaned 與不可重接的程序。

驗收：

- 重啟後所有 detached job 轉成明確狀態。
- 可重接程序恢復 log／stop 控制；不可重接程序明確標成 orphaned／unknown，不假裝 active。
- Stop、exit 與 artifact projection 具備 idempotency。

### `RESOURCE-01`：資源與併發政策

目前狀態：

- 已有 `max_wall_clock_sec`、heartbeat timeout、checkpoint interval 與部分 provider/resource failure classification。

缺口：

- `max_idle_sec`、max model calls、token/cost budget 尚未全面執行。
- 缺少 run／provider／model／workspace 層的 concurrency limit。
- 缺少 queue、backpressure、公平性與 stale worker cleanup 的正式策略。
- Timeout 尚未完整區分 model call、tool call、run 與 operator wait。

驗收：

- 每個限制都有 durable counter／deadline 與明確 terminal／recoverable transition。
- Backend 失聯、queue 滿、worker 被終止與 budget 耗盡時不會永久停在 active。
- 限流不得造成重複執行或丟失已確認完成的工作。

## 9. P2：Memory 與 Skill Lifecycle 缺口

### `MEM-01`：Memory 分層、來源與品質

目前狀態：

- MemoryStore 已支援 SQLite／FTS5、CRUD、搜尋與 export。
- Session history、semantic compaction、skill extraction 與 failure learning 各自已有持久化能力。

缺口：

- Session memory、derived summary、durable user/project memory 與 learned failure episode 尚缺清楚邊界。
- Memory provenance、來源 session／turn／artifact、建立者與保留政策尚未統一。
- 缺少使用者品質回饋、可信度、衝突與 supersession contract。
- Background flush／consolidation 尚未形成可恢復、可稽核流程。

驗收：

- 每筆 durable memory 可追溯來源與 revision。
- Derived memory 不覆蓋 canonical session history。
- Consolidation 可重跑、可回滾，且不靜默合併互相衝突的記憶。

### `MEM-02`：備份、還原與遷移

缺口：

- 缺少 memory import、完整 backup／restore 與 schema migration qualification。
- 缺少跨 state root／workspace 移轉與破損資料復原流程。

驗收：

- Backup 包含 schema version、checksum 與來源 metadata。
- Restore 支援 dry-run、衝突報告與原子切換。
- 舊版本資料能以 migration fixture 驗證，失敗時不破壞現有 store。

### `SKILL-01`：Skill 版本治理

目前狀態：

- SkillLibrary 與 improver 已有 version 欄位及遞增行為。

缺口：

- 缺少正式 rollback、active version selection 與版本間品質比較。
- 缺少 promotion gate，避免低品質自動改進直接成為預設版本。

驗收：

- 任一 active skill 可回退到先前可用版本。
- Promotion 需保留 evaluation evidence、來源 trajectory 與 reviewer／policy revision。
- Rollback 不刪除歷史或破壞引用既有版本的 run。

## 10. Provider 與 Release Qualification 缺口

### `PROVIDER-01`：共用 Qualification Matrix

目前狀態：

- 各 backend 已有自己的 unit／integration tests。
- OpenAI-compatible 與 Ollama 已針對 tool calling、capability、context 與跨 turn history補強。
- 尚未以同一套 fixture 比較所有宣告支援 backend。

缺口：

- 缺少 Ollama、GGUF、Safetensors、OpenAI-compatible、vLLM 共用的 runtime contract suite。
- 尚未統一驗證 tool calling、parser repair、streaming、cancellation、context overflow、truncation、reasoning effort、token usage、latency 與已知限制。
- 缺少最低 reliability threshold 與 capability downgrade 規則。

驗收：

- 每個 provider 產出同 schema qualification report。
- 不支援的能力明確降級，不以假成功或 silent fallback 通過。
- Provider／model 升級時可重跑同一套 fixtures 並比較 regression。

### `REL-01`：分層 Release Gate

目前狀態：

- Python 測試已有 markers，WebGUI 有 type-check 與多個 source／browser fixture scripts。
- GitHub Actions 目前主要覆蓋 Linux sandbox。

缺口：

- 缺少 repo-wide required PR gate。
- 缺少 main／nightly／release candidate 的分層自動化。
- 缺少 migration、restart、cancellation、provider 與 browser artifact 的固定保存與 waiver 流程。

建議 gate：

1. **PR required**：focused Python、Ruff、TypeScript type-check、必要 Web contract tests、`git diff --check`。
2. **Main required**：完整 offline Python lanes、Web lint/build、migration 與 deterministic browser smoke。
3. **Nightly conditional**：真實 sandbox、local backend、長時間 cancellation、restart 與 provider smoke。
4. **Release candidate**：選定部署 provider 的 context、tool、cancel、restart、browser 與升級／回滾 qualification。

驗收：

- Required gate 未通過不得標記 release candidate。
- Conditional gate 明確記錄 capability、環境與結果，不阻擋不宣告支援的平台。
- Waiver 包含原因、影響、負責人與到期日。

## 11. 依賴與建議順序

```text
BASE-01 ──┬── ERR-01 ── ERR-02
          ├── CTX-01 ── CTX-02
          ├── GEN-01
          ├── TOOL-01
          └── REL-01 foundation

ERR-01 + REL-01 foundation
          ├── CANCEL-01
          ├── PROVIDER-01
          ├── FILE-01 / FILE-02 / UI-01
          └── SUP-01 / EXEC-01 / RESOURCE-01

Stable runtime contracts
          ├── SESSION-01
          ├── MEM-01 / MEM-02
          └── SKILL-01
```

建議批次：

1. **第一批：契約與 gate**\
   `BASE-01`、`ERR-01`、`REL-01` foundation。
2. **第二批：P0 runtime 殘餘**\
   `CTX-01`、`GEN-01`、`TOOL-01`，並開始 `PROVIDER-01` fixtures。
3. **第三批：互動與檔案產品化**\
   `CANCEL-01`、`CTX-02`、`ERR-02`、`FILE-01`、`FILE-02`、`UI-01`。
4. **第四批：長期運行**\
   `SUP-01`、`EXEC-01`、`RESOURCE-01`。
5. **第五批：搜尋、Memory 與 Skill 治理**\
   `SESSION-01`、`MEM-01`、`MEM-02`、`SKILL-01`。

`REL-01` 與 `PROVIDER-01` 不是最後才開始的工作；前者從第一批建立，後者隨每個 runtime contract 逐步擴充。

## 12. Definition of Done

任何缺口關閉前，必須滿足：

- 先有能重現缺口的 characterization／regression fixture。
- Runtime core 契約有型別或 versioned schema，不由 WebGUI 或單一 provider 私自定義。
- 成功、失敗、timeout、cancel、retry、restart、malformed input 中的適用場景有測試。
- Backend、事件、SSE、持久化、WebGUI 與 telemetry 對同一狀態語義一致。
- Restart／retry 不重複工具、approval side effect、檔案修改或事件投影。
- 不修改、覆蓋或清理使用者既有未提交 WIP。
- `git diff --check` 與相關 required gate 通過。
- 未通過項目有明確分類；不得以隱藏測試、放寬 assertion 或宣稱不支援來迴避已宣告能力。

## 13. 成功指標

主要缺口關閉後，Mochi 應達到：

1. Context、truncation、tool protocol 與錯誤不再靜默失敗或假完成。
2. Stop、disconnect、timeout、backend crash 與 restart 都收斂到可理解狀態。
3. Session、Goal、AgentRun、Context 與 Memory 的 canonical／derived 邊界清楚。
4. 檔案提案、子集批准、套用、衝突、重規劃與 undo 全程可追溯。
5. 長期 AgentRun 具備 restart-safe adoption、durable job ownership 與資源上限。
6. Memory 與 Skill 有來源、版本、品質、備份與回滾治理。
7. 新 provider、model 或 release 可由固定 qualification report 判斷是否符合 Mochi runtime contract。

## 14. 延後評估的生態與部署能力

以下僅在 local-first 單使用者定位穩定後評估，不列為近期成熟度 blocker：

- ACP／IDE session bridge。
- Remote multi-user gateway、authentication、pairing、rate limit 與租戶隔離。
- Single-binary／低資源部署。
- Windows AppContainer、ACL、Job Object 與 network containment。
- 更完整的一鍵備份、跨機 migration 與集中式運維控制面。
