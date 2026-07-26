# Mochi Test Suite Structural Refactor Plan — COMPLETED

> **Status: COMPLETED — 2026-07-21**
>
> Phases 0–7 are complete. The structural split, runtime/test hygiene, marker
> gates, inventory/baseline documentation, and the follow-up Ruff cleanup are
> implemented and verified. Full tests Ruff debt was reduced from 2,440 to 0.
>
> Follow-up quality commits: eeb639a, 57e22b5, 5b6f47c, 78d4946,
> bfc0638, cff5135, 16523a1. The project-local uv cache remains
> H:/_python/agent_mochi/.tmp/uv-cache. Existing unrelated dirty and
> untracked worktree content was preserved.

> **For agentic workers:** Execute this plan sequentially. Keep test moves, fixture refactors, and behavioral fixes in separate commits. Do not stage or rewrite unrelated working-tree changes.

**Goal:** 在不降低測試覆蓋、不改變產品行為的前提下，把 Mochi 測試套件從少數巨型整合檔拆成可定位、可分層執行、可安全維護的 pytest 結構。

**Architecture:** 保留 pytest/pytest-asyncio 與既有測試語意；以目錄表達主要測試層級，以 markers 表達執行環境與成本。先對最大三個 API 測試檔做機械式拆分，再抽取局部 fixtures/support，最後處理第二波大型檔案與執行入口。全域 `conftest.py` 只保留真正跨領域且無副作用的 fixtures。

**Tech Stack:** Python 3.11-3.13, pytest 8, pytest-asyncio, FastAPI TestClient/httpx, SQLite, Ruff, Pyright, Windows and POSIX test paths.

---

## 0. 結論與基線

### 是否需要拆分

需要，但原因不是「測試數量太多」，而是結構性集中：

| 指標 | 2026-07-20 基線 | 判斷 |
|---|---:|---|
| Python files under `tests/` | 113 | 對 Mochi 的功能範圍合理 |
| Test Python LOC | 約 68,768 | 規模已需要明確分層與治理 |
| `tests/test_goal_api.py` | 13,049 行 / 129 test functions | 必須優先拆分 |
| `tests/test_api_chat_models.py` | 7,065 行 / 120 test functions | 必須優先拆分 |
| `tests/test_api_runtime.py` | 5,834 行 / 70 test functions | 必須優先拆分 |
| `tests/conftest.py` | 2 個 config fixtures | 共用 fixture 明顯不足，但不可一次全域化 |
| pytest markers | 未註冊 | 無法穩定區分快速、整合、安全與平台測試 |

前三個檔案合計約 25,948 行、319 個 test functions。這三檔是第一波範圍；不要一開始移動全部 113 個檔案。

### 已知獨立問題

- `tests/test_api_runtime_detached_exec_recovery.py::test_agent_run_detached_exec_recovers_after_runtime_restart` 目前因跨 event loop 關閉 exec watcher 失敗。
- 這是 runtime cleanup/test isolation bug，不是檔案拆分問題。必須在 Phase 0 以獨立修正建立綠色基線，不可用永久 `xfail` 掩蓋。
- `.pytest-tmp` 有大量 ACL/殘留目錄警告；要在後續 temp hygiene 階段處理，不能以刪除使用者檔案或放寬 ACL 作捷徑。

### 當前工作樹限制

- Task 3 已提交為 `130397a`。
- Task 4/4.5 仍修改 `tests/test_api_runtime.py`、`tests/test_exec_tools.py`，並包含未追蹤的 approval security tests。
- **開始任何 test move 前，必須先完成並提交 Task 4/4.5，或把測試重組放到獨立乾淨 worktree。**
- 不得在目前 dirty tree 直接大量 `git mv`，否則功能變更與結構變更會混在同一 diff。

---

## 1. 不可妥協的重組 invariant

1. 第一輪只搬移測試與局部 support code，不修改 production behavior。
2. 搬移前後的 collected test inventory 必須可對帳；不能只比較 passed 數量。
3. 不刪測試、不合併 assertions、不順手改測試語意。
4. 每次只拆一個來源檔；該來源檔完整通過後才進下一檔。
5. 機械搬移與 fixture 抽取必須分成不同 commit。
6. 不建立大型全域 fake/fixture registry；fixture 應放在最窄的共同目錄。
7. 不使用 autouse fixture，除非能證明所有該目錄測試都需要且無隱藏副作用。
8. 測試不得依賴執行順序、前一測試留下的 SQLite、event loop、shared singleton 或 temp path。
9. `pytest` 不帶 marker 時仍應執行完整 offline suite；不能讓舊開發流程靜默漏測。
10. Security 與 platform tests 不因拆分被降級或排除；執行入口必須明確。

---

## 2. 目標結構

目錄表達「主要層級」，marker 表達「額外屬性」。不要求一次把所有舊檔移完。

```text
tests/
  unit/
    backends/
    runtime/
    tools/
    voice/
  contract/
    approvals/
    tool_activation/
    tool_calling/
  integration/
    api/
      chat/
      goals/
      runtime/
      sessions/
    agent_runs/
    persistence/
  security/
  support/
    app_factories.py
    polling.py
    exec_providers.py
  conftest.py
```

### Marker policy

在 `pyproject.toml` 註冊：

- `integration`: 跨 FastAPI/runtime/store 邊界。
- `security`: 安全 invariant、攻擊面或 fail-closed 行為。
- `slow`: 超過基線門檻的測試；門檻先量測再設定。
- `windows`: 需要 Win32/ACL/reparse-point 行為。
- `posix`: 需要 dir-fd/symlink POSIX 行為。
- `network`: 需要真實外部網路；預設完整 offline suite 不應包含。

不要為每個 unit test 加 `unit` marker；`tests/unit/` 目錄已足夠。不要同時用目錄與 marker 重複表達所有分類。

---

## 3. Phase 0 - 建立可比較的綠色基線

### Task 0.1：先清除工作樹衝突

- [ ] 完成並提交 Task 4/4.5，或建立只含已提交內容的獨立 worktree/branch。
- [ ] 確認 `rtk git status --short` 沒有與待移動測試重疊的修改。
- [ ] 記錄 base commit SHA。

### Task 0.2：建立測試 inventory

新增：

- `docs/testing/test-suite-baseline.md`
- `docs/testing/test-inventory.txt`（由 collect-only 產生，可重建）

步驟：

- [ ] 使用未過濾的 pytest collect-only 輸出記錄完整 node IDs。
- [ ] 記錄總檔案數、collected cases、各檔案 test count、前 20 大檔案。
- [ ] 使用 `--durations` 建立最慢測試與最慢 setup/teardown 基線。
- [ ] 記錄 Windows、POSIX、network-dependent 測試的環境需求。
- [ ] 不在此階段引入 coverage 百分比承諾；先確認是否已有可靠 coverage 工具與基線。

### Task 0.3：獨立修正 cross-loop failure

主要檔案：

- `tests/test_api_runtime_detached_exec_recovery.py`
- `mochi/runtime/service.py` 或 `mochi/runtime/exec_runtime.py`，僅在 production cleanup contract 確實有錯時修改。

步驟：

- [ ] 確認 watcher 建立與 `RuntimeService.close()` 所屬 event loop。
- [ ] 讓 start/close/restart 在同一 async lifecycle 執行，或在 runtime boundary 實作 truthful cross-loop cleanup。
- [ ] 單獨提交此 bug fix；commit 不包含任何 test move。
- [ ] 完整重跑 detached recovery、runtime API 與 RuntimeStore tests。

**Phase 0 gate：** baseline suite 在支援平台上為綠色；沒有用永久 xfail、skip 或降低 assertions 達成。

---

## 4. Phase 1 - 測試執行層級與 support 邊界

### Task 1.1：註冊 markers

修改：

- `pyproject.toml`
- `docs/testing/README.md`

步驟：

- [ ] 註冊 `integration/security/slow/windows/posix/network` markers。
- [ ] 啟用 strict marker validation，避免拼錯 marker 靜默生效。
- [ ] 文件化完整 offline、快速、integration、security、platform commands。
- [ ] 預設 `pytest` 仍收集全部 offline tests。

### Task 1.2：建立窄 support modules

新增：

- `tests/support/__init__.py`
- `tests/support/app_factories.py`
- `tests/support/polling.py`
- `tests/support/exec_providers.py`

規則：

- [ ] 只抽取至少被兩個目標 package 使用、且行為已由現有測試證明的 helper。
- [ ] domain-specific fake 留在該 domain 的 `_support.py` 或 local `conftest.py`。
- [ ] polling helper 必須有 timeout、terminal states 與可診斷錯誤；禁止無界 wait。
- [ ] app factory 每次建立全新 app/store/runtime，禁止 module singleton 洩漏。

**Phase 1 gate：** 僅新增分類與 support infrastructure；collected inventory 與 baseline 等價。

---

## 5. Phase 2 - 拆分 `test_goal_api.py`

來源：`tests/test_goal_api.py`（約 13,049 行 / 129 tests）

目標：

```text
tests/integration/api/goals/
  conftest.py
  _support.py
  test_proposals_and_copy.py
  test_turn_decisions.py
  test_lifecycle_and_followups.py
  test_operator_controls_and_audit.py
  test_scheduling_and_supervision.py
  test_checkpoints_memory_and_collectors.py
  test_exec_approvals.py
```

步驟：

- [ ] 先把原檔頂部 factories、provider、polling 與 DB helpers移到同目錄 `_support.py`/`conftest.py`，不改 implementation。
- [ ] 依 test name/route ownership 做機械搬移；保留原函式名稱、parametrize values 與 assertions。
- [ ] 每搬一個目標檔就跑該檔，再跑整個 goals package。
- [ ] 比對來源檔 test function inventory 與新 package inventory。
- [ ] 所有 tests 搬完才刪除原檔。
- [ ] fixture 去重另做第二個 commit，不與機械搬移同 commit。

**Phase 2 gate：** goals package 全綠；沒有 test case 消失；單檔建議不超過 1,500 行，review hard limit 2,000 行。

---

## 6. Phase 3 - 拆分 `test_api_chat_models.py`

來源：`tests/test_api_chat_models.py`（約 7,065 行 / 120 tests）

目標：

```text
tests/integration/api/chat/
  conftest.py
  _support.py
  test_chat_routes.py
  test_streaming_and_serialization.py
  test_cancellation.py
  test_subagent_transcripts.py
  test_subagent_control.py
tests/integration/api/models/
  conftest.py
  test_model_routes.py
  test_runtime_selection.py
  test_codex_auth_projection.py
```

步驟：

- [ ] 分離 chat 與 model ownership；`_FakeEngine` 只放在 chat local support。
- [ ] 將 cancellation context、stream teardown、final-answer race 集中到 cancellation 檔。
- [ ] 將 subagent list/detail/guidance/transcript 與 cancel/resume/control 分開。
- [ ] model manager/runtime/auth fake 放到 models local fixtures，不升級成全域 fixture。
- [ ] 機械搬移、全檔通過、inventory 對帳後，再做 fixture 去重 commit。

**Phase 3 gate：** chat 與 models 可獨立執行；不再需要載入彼此的 heavyweight fixtures。

---

## 7. Phase 4 - 拆分 `test_api_runtime.py`

來源：`tests/test_api_runtime.py`（約 5,834 行 / 70 tests）

前置條件：Task 4/4.5 必須已提交，避免 approval tests 在移動時混入功能 diff。

目標：

```text
tests/integration/api/runtime/
  conftest.py
  _support.py
  test_serialization_and_workspace_projection.py
  test_agent_run_resume.py
  test_task_execution.py
  test_delegated_tasks.py
  test_approval_routes.py
  test_exec_approval_rehydration.py
  test_scheduling_and_recovery.py
```

步驟：

- [ ] 將 runtime fake engines、direct exec provider、poll helpers放在 local support。
- [ ] approval binding/lifecycle 測試留在 runtime integration；純狀態機 contract 繼續留在 `tests/security` 或未來 `tests/contract/approvals`。
- [ ] restart/rehydration 測試獨立成檔，確保每個測試自行建立與關閉 lifecycle。
- [ ] task execution 與 agent-run resume 不共用 mutable app/runtime fixture。
- [ ] 搬移與 helper refactor 分開提交。

**Phase 4 gate：** runtime API 完整 suite、approval security suite、exec tools 與 RuntimeStore 全綠。

---

## 8. Phase 5 - 第二波大型檔案

依風險與維護頻率逐一處理，不並行大搬移：

| 優先 | 來源 | 約行數 | 建議 ownership |
|---:|---|---:|---|
| 1 | `test_engine_phase2.py` | 2,565 | engine streaming / ReAct / tool execution |
| 2 | `security/test_safe_filesystem.py` | 2,519 | lexical / identity / race / platform |
| 3 | `test_main_chat_tui.py` | 2,444 | rendering / input / cancellation / sessions |
| 4 | `test_tool_exposure.py` | 2,340 | intent / ranking / policy / backend capability |
| 5 | `test_multi_agent_orchestrator.py` | 2,318 | protocol / scheduling / recovery / evidence |
| 6 | `test_api_sessions_settings.py` | 2,146 | sessions / settings / config persistence |
| 7 | `test_backends.py` | 2,017 | backend contracts，優先轉 parameterized contract |

每個來源檔重複 Phase 2 的模式：

- [ ] inventory。
- [ ] mechanical move commit。
- [ ] full source-domain verification。
- [ ] fixture/support refactor commit。
- [ ] cross-domain regression suite。

不要為了達成行數目標把強耦合情境切成無語意的小檔；ownership 清楚比平均行數更重要。

---

## 9. Phase 6 - Temp、event loop 與 singleton hygiene

### Temp paths

- [ ] 所有測試使用 `tmp_path`/`tmp_path_factory` 或明確 workspace-local base temp。
- [ ] 不在 repo root建立 ad hoc `.pytest-tmp/<manual-name>`。
- [ ] 測試自己建立的 runtime/session/temp 資源必須在 fixture teardown 關閉。
- [ ] 不自動刪除既有使用者或其他 agent 的 temp 目錄。

### Async lifecycle

- [ ] 禁止在 TestClient 已擁有 event loop 時再用 `asyncio.run()` 關閉同一 runtime。
- [ ] async fixtures與 runtime task 必須在建立它們的 loop teardown。
- [ ] background task、watcher、scheduler 結束時必須可觀測；不得只 suppress cross-loop exception。

### Shared state

- [ ] app、RuntimeStore、ExecRuntime、approval store 與 singleton cache 每測試隔離。
- [ ] 需要 process-global patch 的測試使用 `monkeypatch` 並驗證還原。
- [ ] 加入 order-randomization smoke run；若不新增 plugin，至少以不同檔案順序重跑高風險 packages。

---

## 10. Phase 7 - 執行入口與品質門檻

先提供穩定命令，再決定 CI provider；目前 repository 沒有 `.github` workflow，不在本計劃中擅自建立 GitHub Actions。

建議入口：

```powershell
# 快速開發回饋
rtk pytest tests\unit tests\contract -q

# API/runtime integration
rtk pytest tests\integration -q

# 安全與平台
rtk pytest tests\security -q

# 完整 offline suite
rtk pytest -m "not network" -q
```

品質門檻：

- [ ] Complete offline pass rate：100%。
- [ ] P0/P1 flaky tests：0；不得以 retry plugin 當修正。
- [ ] Collected inventory：每次 mechanical move 前後一致。
- [ ] Quick lane 時間：以 Phase 0 baseline 設定，不先拍腦袋承諾秒數；後續不得回退超過約定比例。
- [ ] 單檔 review threshold：1,500 行建議、2,000 行需明確理由。
- [ ] 新 integration tests 必須放入對應 package並標記；新 platform tests 必須標 `windows` 或 `posix`。
- [ ] Coverage gate 只在建立可信 baseline 後導入；初始要求是結構重組不得降低 package coverage。

---

## 11. Commit 與驗證策略

每個巨型檔案至少兩個 commit：

1. `test: split <domain> suite by ownership`
   - 只做機械搬移與必要 import 修正。
2. `test: consolidate <domain> fixtures`
   - 抽 fixture/support，刪除重複 setup。

必要時第三個 commit：

3. `test: isolate <domain> async lifecycle`
   - 只處理 event loop、temp、singleton isolation。

每個 commit 都執行：

- [ ] 目標新 package。
- [ ] 原來源檔的相鄰 contract/security tests。
- [ ] collect inventory comparison。
- [ ] `rtk uv run ruff check` 針對變更測試檔。
- [ ] `rtk git diff --check`。
- [ ] 完整 offline suite 至少在每個 Phase 結尾執行。

---

## 12. 完成定義

本計劃完成時必須同時滿足：

- 最大三個巨型 API 測試檔已移除，測試按 route/runtime ownership 分組。
- 所有原 test functions/parameter cases 已在 inventory 中對帳，沒有靜默漏測。
- 第二波所有超過 2,000 行的測試檔已拆分，或有記錄充分的保留理由。
- pytest markers 已註冊且 strict；快速、integration、security、platform、完整 offline 命令可重現。
- 全域 `conftest.py` 保持小且無 autouse 副作用；domain fixtures 位於最窄共同 package。
- detached-exec cross-loop failure 已以獨立 bug fix 解決，不靠 xfail/skip。
- `.pytest-tmp`/runtime cleanup 不再產生持續性的 ACL warning 或跨測試污染。
- 完整 offline suite 全綠，Ruff/diff check 通過。
- 沒有為測試重組修改產品行為；必要 production bug fix 皆為獨立 commit。

---

## 13. 建議執行順序與規模

1. Phase 0：基線與 cross-loop bug，1-2 個小 PR/commit。
2. Phase 1：markers、docs、support skeleton，1 個 PR。
3. Phase 2：Goals，2-3 個 commits。
4. Phase 3：Chat/Models，2-3 個 commits。
5. Phase 4：Runtime/Approvals，2-3 個 commits。
6. Phase 5：每個大型 domain 各自獨立，不綁成單一 mega-PR。
7. Phase 6-7：hygiene 與執行入口，在結構穩定後落地。

第一個實作里程碑只做到 Phase 0-2。完成 Goals 拆分並驗證方法可行後，再決定是否按同樣模式展開其餘檔案；不要一次核准全庫搬移。
