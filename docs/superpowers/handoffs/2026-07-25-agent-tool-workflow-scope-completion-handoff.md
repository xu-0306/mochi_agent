# Agent Tool Workflow 範圍收斂與完成交接

日期：2026-07-26
工作區：`H:\_python\agent_mochi`
分支：`main`
狀態：大量既有 WIP 的 dirty worktree；不可 reset、checkout、clean 或廣泛回退

> 審查修正狀態（2026-07-26）：Finding 1-5 已在目前 working tree 修正並由
> targeted tests 驗證；Finding 6 已以 working-tree evidence manifest 補足。
> 3 個會啟動 Next dev server 的 browser fixture 已改由 bounded watchdog runner
> 管理，並在目前 working tree 分別於 12.3、13.4、10.9 秒 clean exit 0；
> 執行後未留下 Node/CMD process 或受控 `.next-fixture-*` 目錄。
> 接手前必須先讀：
> `docs/superpowers/handoffs/2026-07-26-agent-tool-workflow-handoff-review-findings.md`。

## Continuation Update (2026-07-26, reviewer handoff)

本次已完成 review findings 的 runtime、frontend contract、test-gate 與文件修正，
並以目前 dirty working tree 的 SHA-256 manifest 綁定驗證證據；下一個 agent 應
審查現況，不要從原始「Phase 3 尚未完成」的狀態重新開始。

### 目前狀態

- P2.3 Phase 0-4 implementation 已完成；standalone SSE transport contract、
  frontend aggregate test-gate、binding/restart wording 與 evidence manifest
  的 review findings 已修正。
- startup-only `sessions_dir`、`storage_id` scope、snapshot/range/SSE、frontend
  strict aggregate store 與 `ToolCallCard` aggregate projection 已完成並驗證。
- 目前 `web/package.json` 有 51 個 frontend `test:*` scripts；48 個非 browser
  scripts 已取得 clean exit code。3 個 dev-server browser fixture 在修正 process
  ownership 與 bounded cleanup 後，也已分別取得 clean exit 0；目前 snapshot
  因此有 51/51 個 scripts 的 clean-exit evidence。
- P2.2 full same-session ordinary-Chat model-history linearization 已於
  2026-07-26 以獨立 acceptance matrix 核實完成；證據見
  `2026-07-26-p2-2-model-history-linearization-evidence.md`。
- 本工作區仍有大量既有 dirty WIP 與受權限影響的測試暫存目錄；審查時只看
  本 handoff 指定的變更與驗證，不得 reset、checkout、clean 或廣泛回退。

### 本次工作實際做了哪些改動

#### 1. 修正原始工具工作流的語義權威鏈

- 以 `TurnIntentContract`、bounded `ConversationResolver`、model interpreter、
  durable active task state 取代 latest-message keyword classifier。
- 以 `CapabilityPlan` 統一 catalog、eligibility、exposure、activation 與
  concrete-call authorization 的語義邊界。
- 完成 `tool_search -> tool_activate -> schema refresh -> concrete call`。
- 移除 `mochi/agents/tool_intent_router.py`、routed-intent／keyword fallback
  與 production legacy/shadow runtime path；不得恢復。

#### 2. 修正 policy、approval、execution 與 artifact completion

- 每個 turn 使用 immutable effective-policy snapshot；工具 call 消費
  execution-context policy，不使用建構時 cached policy。
- approval 保存 exact arguments、digest、operation ID、policy/inventory/
  workspace identity 與 resume cursor。
- approval continuation 使用 durable single-owner consume lease；repair、
  reconnect、restart 不得重播 side effect。
- `ArtifactVerifier` 要求 operation/turn 綁定的 verification receipt；工具
  回傳 `success` 本身不能讓 UI 或 durable task 宣稱完成。

#### 3. 修正 session storage identity 與 startup binding

- `sessions_dir` 改為 startup-only；runtime canonical root 變更會拒絕，
  不 persist、不替換 live store，必須重啟後生效。
- SessionStore v2 使用固定長度 lowercase SHA-256 slot filename 與 identity
  sidecar，避免 Windows case-insensitive alias、長 Unicode ID 與特殊字元碰撞。
- v1 Base64／legacy sanitized filename 僅作 fail-closed migration reader；
  verified legacy file 首次寫入時以 atomic replace 遷移至 v2。
- 每個 sessions root 有 versioned marker 與 `storage_id`；browser cursor key
  綁定 `storage_id/session_id/turn_id`，避免跨 root 重播 `Last-Event-ID`。

#### 4. 新增 durable aggregate、repair 與 UI projection

- `tool_workflow_aggregate` v1 strict reader、pure reducer、canonical JSON subset、
  identity join 與 unknown/abandoned/cancelled precedence 已完成。
- durable outbox、approval observation、restart repair、idempotent reconciler、
  publication gate、snapshot/range API、named SSE 與 `Last-Event-ID` replay 已接線。
- frontend 只有一個 aggregate parser/store；duplicate、conflict、out-of-order、
  gap、unsupported schema 與 storage scope change 都 fail closed 或觸發 repair。
- `ToolCallCard` 以 aggregate 為 workflow status 權威；raw transcript 只作
  non-authoritative display fallback，並區分 pending、verified、abandoned、
  unknown、cancelled 與 approval states。

#### 5. 本輪 final gate 與 browser fixture 修正

- browser fixture 預設使用動態 free port、獨立 `MOCHI_NEXT_DIST_DIR`，finally
  清理自有輸出，避免 port lock 與 cross-test dist contamination。
- 修正 [test-subagent-transcript-browser-fixture.mjs](H:/_python/agent_mochi/web/scripts/test-subagent-transcript-browser-fixture.mjs:280)
  的過時 `Recent timeline` 斷言，與目前元件契約 `Execution highlights` 同步。
- `web/next.config.mjs` 支援 `MOCHI_NEXT_DIST_DIR`；`web/package.json` 加入
  aggregate／ToolCallCard／observability 測試入口；`web/tsconfig.json` 納入
  fixture dist type paths。
- 隔離 build 曾由 Next 自動加入 `.next-codex-build` type paths，已移除該
  一次性 generated include；fixture type paths 保留。
- 3 個 dev-server browser scripts 現在經
  `web/scripts/run-browser-fixture.mjs` 的 30–600 秒 bounded watchdog 執行；
  timeout 會終止完整 process tree，且 cleanup 僅限 runner 擁有的
  `.next-fixture-*` 目錄。
- fixture 改為直接啟動 Next CLI，不再透過 `npm.cmd` 與 `shell: true`；
  原本 assertion、browser close 與 runtime cleanup 都完成後仍無法 clean exit
  的根因，是 shell child 沒有可靠擁有 Next Node process。

### 供下一個 agent 的審查重點

1. 先確認 `tool_intent_router`、`routed_intent`、`legacy_routed_intent`、
   `fallback_keyword` 在 production path 沒有回復。
2. 依第 9、10 節重跑 targeted backend/frontend gate；不要把分組 pass count
   相加成單一「全 repo 測試數」。
3. 審查 flag true -> false 時是否只停止新 aggregate publication，保留既有
   outbox/durable evidence，且不執行或重播工具。
4. 審查 storage marker、legacy migration、collision、delete/tombstone 語義；
   v2 delete 的 exact identity tombstone 仍需產品決策是否符合完整 erasure。
5. P2.2 acceptance matrix 已完成；後續審查應保留單一 durable FIFO lane，
   不得重建 aggregate reducer、approval store 或 intent authority。

### Final-gate update (2026-07-26, current working tree)

- Finding 1：`sessions.py` standalone SSE `data:` 現在包含
  `type/storage_id/session_id/turn_id/aggregate/publication_enabled/authoritative`；
  `web/src/lib/sse-frame.ts` 與 SSE contract test 會以 server-shaped frame 經過
  shared parser、strict transport parser，再套用 aggregate store。
- Finding 2：`test:tool-workflow-aggregate` 與
  `test:tool-workflow-aggregate-sse` 已加入 `web/package.json`；目前腳本數為 51。
- Finding 3-5：本交接的 current status 明確區分已驗證狀態、歷史設計、
  Engine-first binding 的 root/storage/gate 保證，以及 Settings PATCH 409 與
  external config `pending_restart` 的不同語義。
- Finding 6：證據綁定在
  [2026-07-26-agent-tool-workflow-final-gate-evidence-manifest.md](H:/_python/agent_mochi/docs/superpowers/handoffs/2026-07-26-agent-tool-workflow-final-gate-evidence-manifest.md)，
  base commit 為 `28f45e156e3b85d004e15942217c0c7a8ff578a2`。
- 3 個 dev-server browser fixtures 已經由共用 watchdog runner 取得 clean exit 0；
  不再列為 environment-limited。P2.2 full same-session ordinary-Chat
  model-history linearization 亦已由獨立 acceptance matrix 核實完成。

Current verified backend/static evidence：63、45、95、130、158、48 tests 的
targeted matrices 均通過；type-check、lint（0 errors/4 warnings）、isolated
production build、compileall、diff check 均通過。pytest 警告僅為本機無法寫入
`.pytest_cache`。P2.2 的新增驗證結果與 hashes 另見其 evidence manifest。

## Continuation Update (2026-07-25)

The remaining Phase 3 delivery work has now been implemented:

- Session-scoped snapshot, bounded range, named aggregate SSE, `Last-Event-ID`, and storage scope handling are present in the backend.
- The web client has a strict aggregate v1 parser and one cursor store keyed by `storage_id`, `session_id`, and `turn_id`.
- Exact duplicates are ignored; conflicts, out-of-order records, gaps, unsupported payloads, and storage changes preserve last-known-good state and request range/snapshot repair.
- Chat SSE, non-stream chat fallback, session reload, and ToolCallCard now consume the aggregate path. Raw transcript data remains display-only.
- ToolCallCard does not report success without authoritative execution and terminal verification evidence, and distinguishes pending, abandoned, unknown, cancelled, and verified states.

Historical final-gate verification (2026-07-25) covered the Phase 3/4 delivery scope;
the current working-tree verification is recorded in the 2026-07-26 manifest:

- The historical 49 `web` `test:*` scripts passed, including live stream,
  reconnect/reload, subagent transcript, and aggregate ToolCallCard browser fixtures.
- Frontend type-check passes; lint has 0 errors and 4 pre-existing warnings.
- The isolated production build passes with `MOCHI_NEXT_DIST_DIR=.next-codex-build`.
- Backend rollout/config/settings coverage passes (83 tests), including flag-on,
  flag-off, publication rollback, and startup-only `sessions_dir` rejection.
- Session migration and storage-scope coverage passes (23 tests), including
  legacy identity verification, collision probing, storage markers, and cursor
  scope checks.

The default `.next` output remains unsuitable for a clean build when an external
Next process owns its event file; the isolated build is the reproducible build
evidence. This 2026-07-25 historical gate did not claim P2.2 complete; the
separate 2026-07-26 P2.2 evidence supersedes that historical open status.

## 1. 交接目的

這份文件供下一位 agent 直接接手「普通 Chat 中，agent 能依完整對話理解任務、正確取得工具、套用 Auto Review／approval、執行並在 UI 顯示真實狀態」的收尾工作。

它特別區分：

1. 已實作且對成熟度必要，必須保留的成果。
2. 已走過但已被替代的錯誤方案。
3. 只有分析、尚未實作，而且不應在本次繼續擴張的方案。
4. 本次需求真正尚未完成的項目與固定實作順序。

不要因為 P0、P1 與 P2 的多個子項已完成，就忽略文件仍列出的其他 backlog。P2.3 Phase 3/4、storage-root 一致性閘門與 P2.2 full same-session ordinary-Chat model-history linearization 均已完成各自的 acceptance gate。

## 2. 接手前必讀

依序完整閱讀：

1. `AGENTS.md`
2. 本交接文件
3. `documents/architecture/2026-07-23-agent-tool-workflow-p0-p2-plan.md`
4. `documents/architecture/2026-07-25-tool-workflow-aggregate-stream-replay-rfc.md`
5. `docs/superpowers/handoffs/2026-07-24-agent-tool-workflow-p0-p2-continuation-handoff.md`

原始問題證據：

- 對話紀錄：`D:\_download\mochi-chat (2).md`
- 畫面：`C:\Users\Xu\AppData\Local\Temp\codex-clipboard-4d257419-bcdd-45a4-8340-c777acdedda9.png`

注意：完整 immutable StoreBinding／sessions_dir hot-switch 仍不在本次 scope。本交接採用並已驗收的範圍收斂是：runtime 不可切換、變更後重啟；主計畫與 aggregate RFC 已同步記錄 startup-only invariant 與 storage scope，不得重新把 hot-switch 當成本次 blocker。

## 3. 原始故障與根因

原始記錄中，使用者不是只說「做 B」或「你有哪些寫入工具」，而是在多輪對話中先定義 B，再要求建立通用版文件。舊流程只看最新訊息：

```text
latest message
-> keyword / ToolIntentRouter classifier
-> exposure routing
```

因此「B，通用版」被當成模糊意圖，寫入工具 schema 沒有暴露。後來即使 `tool_search` 找到 `file_write`，當時也沒有可靠的 callable activation broker。Auto Review 只會在 concrete tool call 形成後審查；當工具根本未被 expose／activate，就永遠不會走到 Auto Review。

成熟流程現在的權威鏈是：

```text
bounded conversation + summary + durable active task
-> TurnIntentContract
-> CapabilityPlan
-> policy-bounded exposure / activation
-> concrete-call authorization
-> execution
-> artifact verification
-> durable aggregate projection
```

不可退回 latest-message classifier，也不可把 discovery、activation、authorization、approval 混成一個布林判斷。

## 4. 已完成且必須保留

### 4.1 P0：政策與 concrete-call 執行

已完成：

- `EffectivePolicyResolver` 與 immutable effective-policy snapshot。
- 普通 Chat 每 turn 重新讀取 session override。
- `file_write`、`file_edit`、`apply_patch` 與 exec／execute-code 類工具在每次 call 消費 execution-context policy，不依賴工具建構時的 cached policy。
- Auto Review／manual approval 在 concrete call 形成後運作；activation 本身不等於授權。
- hard deny、workspace scope、sandbox、policy drift、target drift 不能被 approval 繞過。
- 普通 Chat approval 保存 exact arguments、argument digest、operation ID、policy／inventory／workspace identity 與 resume cursor。
- approval continuation 使用 durable single-owner consume lease，成功執行證據先落盤，再回到 ReAct continuation；repair／replay 不得重播 side effect。

這些直接修正原始「Auto Review 已開啟，但 agent 仍不會使用工具」及「核准後重啟可能重複 mutation」問題，必須保留。

### 4.2 P1：狀態化對話理解與工具取得

已完成：

- `TurnIntentContract`、bounded `ConversationResolver`、model conversation interpreter。
- `CapabilityPlan` 成為 exposure 與 activation eligibility 的唯一語義權威。
- `ActiveTaskState` 與 durable conversation state round-trip。
- `tool_search -> tool_activate -> schema refresh -> concrete call` 可呼叫流程。
- 同 capability 工具使用 deterministic catalog priority。
- resolver／planner／adapter failure fail closed。
- current-turn required deliverable 不能一開始就被模型標成 `satisfied`。

已刪除且不可恢復：

- `mochi/agents/tool_intent_router.py`
- routed-intent／keyword exposure fallback
- production path 的 latest-message classifier 呼叫
- legacy／shadow runtime mode；舊 persisted 值只可單向讀取並正規化成 `enforce`

這是本次最核心的產品修正，不能為了簡化而恢復舊 classifier。

### 4.3 P2.1：artifact verification

已完成：

- `ArtifactVerifier` 已接入 normal execution 與 approval continuation completion。
- required deliverable 必須有 operation／turn 綁定的 verification receipt，不能只相信 tool 回傳 `success`。
- receipt schema v3；v1/v2 migration reader 不虛構 scope 或 execution evidence；未知 future schema fail closed。
- first-party file mutation 的 structured `file_changes`／resolved target report、內容／digest 與 host-owned validation profiles 已接入。

保留限制，不要在本次擴張：

- 尚未做整個 workspace 的 before/after snapshot diff。
- 無法偵測第一方工具未回報的 side effect。
- 尚未做完整 automatic retry orchestration。

上述限制不是原始工具可用性與 Phase 3 UI 的 blocker，列入後續 backlog。

### 4.4 P2.2：durable state 與 model-history linearization

已完成並保留：

- cross-instance CAS 與 strict durable snapshot。
- nonterminal discovery。
- applied-result reconciliation 與 lease-proven pre-claim FIFO orphan recovery。
- timeline precommit／effect boundary／result persistence。
- operation ID 改由 timeline deterministic descriptor 產生。

2026-07-26 closure：

- ordinary-Chat user admission 與 FIFO identity 同批 commit。
- model prompt 只在 durable claim 後建立，並從 matching strict snapshot
  依 durable turn sequence 重排 predecessor transcript。
- 三個 pre-admitted turns、cross-engine lane exclusion、queued cancellation、
  legacy prefix 與 approval-continuation compatibility tail 均有 regression test。
- 不跨 model generation/tool execution 持 filesystem lock；lane ownership
  由 durable lease 與 heartbeat 維持。

不得把這項 closure 解讀為重建 approval continuation 或 aggregate；既有
exact-checkpoint approval resume 與單一 aggregate authority 均保持不變。

### 4.5 P2.3 Phase 0-2：aggregate、outbox 與修復

已完成並保留：

- `tool_workflow_aggregate` v1 strict reader、legacy partial adapter、canonical JSON subset 與 pure reducer。
- approval／timeline／receipt identity join fail closed。
- `unknown`、`abandoned`、`cancelled` 等 precedence。
- durable aggregate outbox。
- approval observation／restart repair。
- idempotent reconciler；只能補 observation／outbox，不能執行工具。
- shared live publication gate；true -> false 會 drain 已取得 publication lease 的 writer。
- incremental post-commit verifier 與 paged restart audit。
- source mismatch／duplicate／gap／unsupported diagnostics。
- API routes 使用 Engine-first route resolver：`mochi/api/session_store_binding.py`。
  RuntimeService 可使用 injected `SessionStore`，但 bind 時驗證相同 canonical
  sessions root/storage_id，並綁定相同 `ToolWorkflowPublicationGate`；系統保證
  storage-root 與 gate consistency，不保證所有元件持有同一個 SessionStore object。

這些是 Phase 3 能在 stream、reconnect、reload 後得到同一真實狀態的基礎，不要刪掉，也不要另建第二套 aggregate state machine。

### 4.6 SessionStore collision-free path migration

已完成：

- v2 writer 使用固定長度、lowercase SHA-256 slot filename。
- exact session ID 存在 identity sidecar，讀取時驗證 filename hash／slot。
- Windows case-insensitive filesystem 不會讓 mixed-case Base64 session identity 靜默 alias。
- 長 Unicode session ID 不會超過單一 filename component limit。
- `a:b` 與 `a?b` 可同時存在。
- v1 Base64 只作 migration reader，而且必須有 matching durable `session_id` envelope。
- 舊 sanitized filename 只在 explicit identity 可證明，或 identity-free filename 的 preserved spelling 完全可逆時讀取。
- verified v1／legacy file 第一次寫入時，在 sidecar locks 下以 `os.replace` 遷移至 v2。
- list、paged inventory、load、replace、delete、strict CAS 與 session routes 使用 logical session ID，不再把 `path.stem` 當 ID。

主要檔案：

- `mochi/sessions/store.py`
- `mochi/api/routes/sessions.py`
- `tests/test_session_store.py`
- `tests/integration/api/sessions/test_session_routes.py`

主代理最後另補 Windows legacy preserved-case fail-closed：identity-free `alpha.jsonl` 不可被 `ALPHA` 誤讀。

一項需要產品語義決策、但不應自行無限擴張的事項：v2 delete 目前保留含 exact session ID 的 identity tombstone，以免刪除 collision slot 0 後讓 slot 1 無法尋址。這不保留聊天 JSONL，但會保留 session ID metadata。

- 若「刪除 session」定義為刪除對話內容，保留並在資料政策中說明。
- 若定義為完整 erasure，將 tombstone 改成不含 identity 的 collision marker，或採明確 fail-closed collision policy；必須補 hash-collision／delete／recreate 測試。
- 未取得產品決策前，不要順手重寫整個 slot allocator。

## 5. 已做但屬於繞路或已被替代

### 5.1 latest-message ToolIntentRouter

這是錯誤的 production semantic authority。它無法可靠解讀多輪 reference resolution、旁支問題、否定限制與 durable active task。相關 runtime 已刪除；舊文件只可作設計歷史。

不要因為新 resolver 比 classifier 複雜，就恢復 keyword fallback。真正需要的簡化是縮小 resolver contract，而不是讓最新一句話重新控制工具暴露。

### 5.2 Base64 session filename v1

這是 session migration 實作中的錯誤中間方案：URL-safe Base64 仍區分大小寫，但 Windows 預設 filesystem lookup 不區分大小寫；某些 Unicode ID 因此仍會碰撞，而且長 ID 會超過 component limit。

v1 writer 已被 v2 SHA-256 + identity sidecar 取代。v1 reader 保留只有 migration 價值；不可重新當 writer。

### 5.3 先做局部測試、再逐層補 invariants

先前多個子代理同時碰觸 Engine、Runtime、routes 與 SessionStore，造成每一輪 focused test 通過後，下一層才發現新的 cross-layer invariant。這種工作方式本身不是成熟度成果。

後續規則：

- 一次只允許一個 agent 修改同一組核心檔案。
- 每個 slice 先寫 acceptance matrix，再改 code。
- focused suite 只能證明 slice，不可宣稱整體 P0-P2 完成。
- 發現非安全／資料正確性／主流程 blocker 的議題，先進 backlog，不自動擴張本次架構。

## 6. 只有設計、尚未實作，而且本次不應擴張

### 6.1 完整 runtime sessions_dir hot-switch

已做過只讀設計審查，但沒有實作完整的：

- immutable `SessionStoreBinding`／`RuntimeStorageBinding` dataclass graph
- request／turn／runtime job／SSE binding leases
- quiesce／drain manager
- candidate RuntimeStore／approval DB／tasks root transaction
- prepare／commit／restore／activate-after-commit 設定協調器
- live storage migration／rekey

修正前的設計審查曾發現：`AgentEngine.apply_config()` 若直接換
`_session_store`，active `TimelineCoordinator` 可能仍持有舊 store；settings
route 若先 persist config、再更新 app state、最後才呼叫 engine，支援 runtime
hot-switch 時會拆寫兩個 roots。目前 production path 已在 live mutation 前以
`ensure_sessions_dir_unchanged()` 拒絕 root 變更；這段只保留為 historical risk，
不是目前實作描述。

但原始產品需求並沒有要求無停機切換 session storage。為本次建立完整 hot-switch 系統成本過高，也會繼續擴張測試矩陣。

本次採用最小成熟決策：

```text
settings API PATCH 收到不同 canonical root -> 回 409，不持久化、不 staging，
使用者必須明確修改設定檔後重啟。
external config file 已含新 root -> live config 保留舊 root，status 為
pending_restart，重啟後 startup binding 才使用新 root。
```

這不是刪除既有成果；完整 hot-switch 尚未寫入 production code。只保留設計分析供未來真的出現產品需求時使用。

### 6.2 本次明確不做

- 不做 live storage migration。
- 不做完整 hot-switch lease/quiesce manager。
- 不做 workspace snapshot diff。
- 不做 general automatic retry planner。
- 不重寫 full same-session scheduler，除非主流程驗收證明必要。
- 不重做整個 WebGUI；只切換工具 workflow projection。
- 不新增第二個 approval database、第二套 reducer 或第二個 intent authority。
- 不修理與本需求無關的 dirty-worktree WIP、測試暫存目錄或既有全域 typing debt。

## 7. 目前真正的未完成項目

Slice 1 至 Slice 5 已完成並通過各自的 focused checks：startup-only
`sessions_dir` binding、storage marker、snapshot/range/SSE、frontend strict
aggregate store，以及 `ToolCallCard` aggregate projection。原本的 Blocker A/B/C
已不再是未完成項目。

### Final gate：完整驗收與文件

本輪已完成並留下 evidence：

- production flag-on／flag-off 與 rollback rehearsal：83 backend tests passed
- migration／storage-scope rehearsal：23 tests passed
- 歷史 49 個 frontend `test:*` scripts 曾通過；其中一個 fixture 的過時
  `Recent timeline` 斷言已修正為目前元件契約 `Execution highlights`。目前
  package.json 已有 51 個 scripts；current clean-exit evidence 請以本 handoff
  頂端狀態與 final-gate manifest 為準。
- frontend type-check、lint、隔離輸出的 production build 均通過
- 主計畫、本交接與 RFC 的 final-gate evidence 已同步

P2.2 的 full same-session ordinary-Chat model-history linearization 已由獨立
acceptance matrix 核實完成；結果不改寫本節保留的 P2.3 historical evidence。

## 8. 已完成 slices 的 acceptance contract

以下不是下一個 agent 的待辦實作順序，而是已完成 slice 的驗收契約。後續
agent 應重跑或審查這些 invariant，不要重建同一套架構，也不要同時修改
`engine.py`、`settings.py`、`server.py`、`service.py` 或
`session_store_binding.py` 的重疊區域。

### 8.1 Slice 1：startup-only `sessions_dir` guard（已驗收）

- [x] canonical root comparison 處理 `expanduser()`、absolute/resolve normalization
  與 Windows case normalization。
- [x] `AgentEngine.apply_config()` 在任何 live mutation 前拒絕 root 變更，並保留
  Engine 所有既有欄位。
- [x] settings API PATCH 對不同 root 回 `409 sessions_dir_restart_required`，
  不建立目錄、不 persist、不更新 app state 或 Engine。
- [x] external config reload 保留目前 applied config，暴露 `pending_restart`；
  這與被 409 拒絕的 PATCH 不同，PATCH 不會自動 staging。
- [x] normalized-equivalent path、idle/active Engine、preflight side-effect
  isolation 與其他非 storage settings 均有測試。

### 8.2 Slice 2：storage identity scope（已驗收）

- [x] 每個 sessions root 以 versioned marker 與隨機 `storage_id` 建立 identity；
  malformed/future marker fail closed。
- [x] snapshot、range、SSE transport 帶 `storage_id/session_id/turn_id` scope。
- [x] frontend cursor key 綁定 `storage_id + session_id + turn_id`；scope 變更會
  保留 last-known-good 並要求 snapshot/repair。

### 8.3 Slice 3：snapshot、range 與 SSE（已驗收，含 review fix）

- [x] snapshot 是單一 session/turn 的最新 validated aggregate；range 使用 bounded
  contiguous records。
- [x] SSE event name 固定為 `tool_workflow_aggregate`，`id` 使用 aggregate
  `event_id`，data 同時帶 `type/schema_version/storage_id/session_id/turn_id`、
  `aggregate/publication_enabled/authoritative`。
- [x] exact duplicate、conflict、out-of-order、gap、unsupported schema、storage
  mismatch 與 `Last-Event-ID` 行為 fail closed 或觸發 repair。
- [x] standalone server-shaped SSE frame 可經 shared frame parser、strict transport
  parser 並套用到 aggregate store；backend test 逐欄驗證實際 `data:` JSON。
- [x] reconnect/replay 只讀 outbox，不執行 tool、不 consume approval；flag off 不發布
  新 aggregate，也不讓 UI 誤報完成。

### 8.4 Slice 4：frontend strict parser/store（已驗收，已納入 test gate）

- [x] `tool_workflow_aggregate` v1 strict parse 與單一 aggregate store 已納入
  `test:tool-workflow-aggregate`。
- [x] store 保存 contiguous seq、event ID、idempotency key；exact duplicate ignore。
- [x] conflict/out-of-order/gap 觸發 range/snapshot repair；unsupported payload 保留
  last-known-good，不推斷 success。
- [x] storage scope change 清除舊 cursor 並要求新 snapshot。
- [x] standalone SSE frame contract test 已納入 `test:tool-workflow-aggregate-sse`。

### 8.5 Slice 5：ToolCallCard projection（已驗收）

- [x] `ToolCallCard`/workflow panel 以 aggregate store 為 workflow status authority；
  raw transcript 只作 non-authoritative fallback。
- [x] UI 區分 catalog、eligible、exposed、activated、review、execution、verification。
- [x] `not_observed` 不顯示 success；`abandoned` 與 `unknown` 不合併。
- [x] approval pending/rejected/expired/consuming 與 executed/verified 不以本地推測
  互相覆蓋。

### 8.6 已完成的 final gate 與審查順序

以下工作原於 2026-07-25 完成，並在 2026-07-26 review-fix pass 依 evidence
manifest 重跑或擴充；下一個 agent 應做審查或重跑，不要重建同一套 slice：

1. Python focused + integration matrix：session path 63、aggregate 45、
  approval/rehydration 95、timeline/exec 130 passed。
2. rollout/config/settings flag gate：158 passed，涵蓋 flag-on、flag-off、
   publication rollback 與 startup-only reject。
3. frontend current 51 個 `test:*` scripts：48 個非 browser scripts clean-exit；
   3 個 dev-server browser fixtures 亦在 watchdog/harness 修正後 clean exit 0。
4. storage marker、legacy v1、sanitized legacy、collision 與 cursor scope：
   migration/storage-scope 48 passed。
5. flag true -> false 不刪 outbox、不重播 side effect、不恢復 legacy intent route。
6. 隔離 production build 通過；default `.next` 的 Windows 鎖檔只屬環境限制。
7. 主計畫與 RFC 已更新；P2.3 final gate 與 P2.2 acceptance gate 均已關閉。

## 9. 驗收證據與不可過度宣稱的界線

截至本交接：

| 範圍 | 結果 | 說明 |
|---|---:|---|
| contract/capability/exposure/activation baseline | 140 passed | 早期 enforce-only cutover 驗收 |
| engine baseline | 46 passed | 早期 enforce-only cutover 驗收 |
| config/session baseline | 105 passed | 早期 enforce-only cutover 驗收 |
| aggregate/outbox/API Phase 2 | 76 passed | 主代理先前獨立驗收 |
| approval continuation/lifecycle/rehydration | 78 passed | 主代理先前獨立驗收 |
| timeline/exec/execute-code workflow | 72 passed | 主代理先前獨立驗收 |
| SessionStore v2 + outbox + session routes | 63 passed | 本輪重跑；含 legacy preserved-case 修正 |
| Aggregate/outbox/observability | 45 passed | 本輪重跑 |
| Approval/lifecycle/rehydration | 95 passed | 本輪重跑 |
| Timeline/exec/execute-code workflow | 130 passed | 本輪重跑 |
| Rollout/config/settings flag gate | 158 passed | 本輪擴充重跑；flag-on、flag-off、rollback、startup-only reject |
| Migration/storage-scope rehearsal | 48 passed | 本輪擴充重跑；legacy identity、collision、marker、cursor scope |
| Frontend `test:*` matrix | 51/51 scripts clean-exit | 48 non-browser scripts；3 browser fixtures 經 bounded watchdog clean exit 0 |
| Frontend type-check/lint | passed | lint 0 errors、4 pre-existing warnings |
| Isolated frontend production build | passed | `MOCHI_NEXT_DIST_DIR=.next-codex-build` |
| Python compileall | passed | final gate |
| git diff --check | passed | final gate |

這些是按 slice 分組的 evidence，不能把 pass count 相加成單一測試數；但
Phase 3/4 final gate 已完成。P2.2 的獨立 acceptance evidence 不與這些
P2.3 pass counts 混算，詳見 P2.2 evidence manifest。

## 10. 建議驗證命令

所有 shell commands 依 `AGENTS.md` 必須經 `rtk`。

Session path checkpoint：

```powershell
rtk proxy python -m pytest -q tests/test_session_store.py tests/test_tool_workflow_outbox.py tests/integration/api/sessions/test_session_routes.py --basetemp .tmp-handoff-session-path
```

Aggregate／outbox：

```powershell
rtk proxy python -m pytest -q tests/test_tool_workflow_aggregate.py tests/test_tool_workflow_outbox.py tests/test_tool_workflow_observability.py --basetemp .tmp-handoff-aggregate
```

Approval continuation：

```powershell
rtk proxy python -m pytest -q tests/security/test_approval_lifecycle.py tests/security/test_timeline_approval_continuation.py tests/integration/api/runtime/test_approval_routes.py tests/integration/api/runtime/test_exec_approval_rehydration.py --basetemp .tmp-handoff-approval
```

Timeline／exec：

```powershell
rtk proxy python -m pytest -q tests/unit/sessions tests/unit/engine/test_timeline_chat_integration.py tests/test_exec_runtime.py tests/test_exec_tools.py tests/test_execute_code_and_mcp.py --basetemp .tmp-handoff-timeline-exec
```

Static checks：

```powershell
rtk proxy python -m compileall -q mochi
rtk git diff --check
rtk proxy rg -n "tool_intent_router|routed_intent|legacy_routed_intent|fallback_keyword" mochi web/src tests
```

最後一個 `rg` 應沒有 matches。

Frontend 在 `web` 目錄：

```powershell
rtk npm run type-check
rtk npm run lint
```

`web/package.json` 沒有單一 `test` script；目前要跑完整 51 個 `test:*` scripts，
在 PowerShell 使用：

```powershell
rtk proxy powershell -NoProfile -Command "`$scripts=(Get-Content package.json -Raw | ConvertFrom-Json).scripts.PSObject.Properties.Name | Where-Object { `$_.StartsWith('test:') }; `$failed=@(); foreach(`$script in `$scripts){ rtk npm run `$script; if(`$LASTEXITCODE -ne 0){ `$failed += `$script } }; if(`$failed.Count){ Write-Output ('FAILED=' + (`$failed -join ',')); exit 1 }"
```

隔離 production build：

```powershell
rtk proxy powershell -NoProfile -Command "`$env:MOCHI_NEXT_DIST_DIR='.next-codex-build'; rtk npm run build"
```

若 pytest 只警告 `.pytest_cache` 因本機權限無法建立，但測試本身通過，將它記為環境警告；不要把警告誤報成產品失敗。若測試真的 failed，不能用此警告掩蓋。

## 11. Dirty worktree 與協作規則

- 工作樹包含大量使用者與先前 agent 的 WIP。
- 不得執行 `git reset --hard`、`git checkout --`、`git clean` 或廣泛 formatter。
- 不得刪除看似暫存但權限異常的目錄；它們可能由其他 sandbox／agent 建立。
- 編輯優先使用 `apply_patch`。
- 同一時間不要讓不同 subagents 修改 `engine.py`、`settings.py`、`server.py`、`service.py` 或 `session_store_binding.py` 的重疊區域。
- subagent 回報的 pass count 必須由主 agent 在檔案穩定後重跑；測試與編輯同時進行會產生不可信的中間失敗或假通過。
- 每完成一個 slice，就更新本交接的「已驗收」區，而不是把未核實結果寫進主計畫。

## 12. 下一位 agent 的開場指令

```text
從 H:\_python\agent_mochi 接手普通 Chat agent-tool workflow 的 final-gate 審查。

先完整讀取：
1. AGENTS.md
2. docs/superpowers/handoffs/2026-07-25-agent-tool-workflow-scope-completion-handoff.md
3. documents/architecture/2026-07-23-agent-tool-workflow-p0-p2-plan.md
4. documents/architecture/2026-07-25-tool-workflow-aggregate-stream-replay-rfc.md

保留 dirty worktree，不得 reset/checkout/clean，不得恢復 ToolIntentRouter 或
latest-message classifier。先完整讀取本 handoff、主計畫、aggregate RFC，接著
重跑第 9、10 節的 targeted matrix，確認既有 evidence 與目前程式一致。

不要重做已完成的 Slice 1-5，也不要實作完整 hot-switch／lease manager。
P2.2 acceptance matrix 已完成；後續變更仍須保持 aggregate reducer、approval
store、CapabilityPlan 與 TurnIntentContract 的單一權威性。
```

## 13. 完成定義

本次需求只有在以下全部成立時才算完成：

- 多輪對話中的 reference resolution 由 `TurnIntentContract`／durable task state 處理，不由最新訊息 keyword classifier 決定。
- 需要寫入／執行時，正確工具能被 exposure、activation，並在 concrete call 套用正確 effective policy。
- Auto Review／manual approval、deny、sandbox 與 workspace scope 有可驗證且重啟安全的行為。
- approved continuation exactly-once，不因 repair／reconnect 重播 side effect。
- artifact completion 有 durable verified receipt，不因 tool 自稱成功而完成。
- snapshot、SSE live、SSE reconnect、range repair 與 reload 收斂到同一 aggregate。
- frontend 不推斷缺失證據，並正確顯示 partial／pending／abandoned／unknown／verified。
- sessions_dir runtime change 不會拆寫 roots；本次以 startup-only reject + restart 完成，而不是 live hot-switch。
- session path migration 不碰撞、不誤讀 legacy identity，且 migration／delete 語義有明確測試與文件。
- flag-off rollback 不重播工具、不刪 durable evidence、不恢復 legacy classifier。
- 完整 backend/frontend matrix、compileall 與 diff check 通過，主計畫與 RFC 狀態同步。
