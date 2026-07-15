---
summary: "Protected Workspace、Approval、Undo、Auto Review 強化計劃交接：分析與計劃已完成，但尚未開始實作；main 工作樹目前有大量使用者變更。"
created: 2026-07-16
tags: [project-status, handoff, protected-workspace, approval, undo, auto-review, sandbox, diff-ui]
related: [docs/superpowers/plans/2026-07-15-protected-workspace-approval-hardening.md, mochi/utils/security.py, mochi/tools/file_ops.py, mochi/runtime/service.py, mochi/runtime/store.py, mochi/config/manager.py, web/src/lib/api.ts]
---

# Protected Workspace 強化計劃交接

## 一句話狀態

目前是「Gap Analysis 已查證、實作計劃已完成；Task 0 至 Task 11 尚未開始實作」。

## 已完成與未完成

已完成：

- 對照 Mochi 與本地 `reference/` 原始碼查證先前 Gap Analysis。
- 撰寫 Task 0 至 Task 11 的完整實作計劃。
- 由同一位 Codex plan reviewer 進行三輪審查並修訂。
- 完成計劃文件的 Markdown、RTK 命令、暫存檔與 `git diff --check` 機械檢查。

尚未完成：

- 沒有開始執行計劃中的任何 Task。
- 沒有新增資料庫 migration、SafeFilesystem、approval 新狀態機、server-authoritative Undo、config CAS/outbox 或 OS sandbox。
- 沒有實作計劃新增的 Web settings ETag、409 conflict 或 rule persistence UI contract。
- 沒有執行計劃要求的完整 Windows/Linux adversarial test matrix。

先前已存在的 Stage 6 修正記錄於：

- `.claude/skills/agent-memory/memories/project-status/tool-activation-stage6-protected-workspace-and-diff-ui-2026-07-14.md`

Stage 6 是既有程式變更，不代表本次 hardening plan 已實作。

## 核心產物

完整計劃：

- `H:/_python/agent_mochi/docs/superpowers/plans/2026-07-15-protected-workspace-approval-hardening.md`

注意：

- `docs/` 目前被 Git 忽略，`git status --ignored` 顯示 `!! docs/`。
- 本交接記憶位於 `.claude/skills/agent-memory/memories/`，同樣被 Git 忽略。
- 不要為了讓文件出現在 Git status 而擅自修改 `.gitignore`；如需納入版控，先取得使用者授權。

## 已查證成立的主要 Gap

1. 路徑政策檢查與真正 I/O 分離，存在 symlink、junction、hardlink 與 TOCTOU 風險。
2. 單檔寫入會直接 truncate；多檔 patch 失敗可能留下部分套用。
3. Approval 沒有完整綁定 workspace/base/policy/request identity，也缺少一致的 TTL、CAS resolve 與 consume-once。
4. Edited preview replay 可替換 patch；部分 stale replay 路徑會略過預期 guard。
5. Undo 接受 client 傳回的原始內容，而非依 server-side applied record 與 CAS 執行。
6. 現有 Auto Review 更接近 deterministic policy auto-allow，還不是結構化、digest-bound reviewer。
7. Config/rule 直接寫 YAML，沒有跨程序 lock、ETag/CAS、atomic replace 與 transactional outbox。
8. 現有 sandbox 不是真正的 host filesystem/process/network boundary。
9. 後端 Git diff/status 缺少完整的 binary、mode、rename、EOF、non-UTF8 fidelity。
10. 目前 security path 會把 read scope 強制成 `any`，即使 caller 指定 `workspace`。

## 對原 Gap Analysis 的修正

- 「前端 DiffViewer 以填空白重建內容」已是過時證據；目前前端直接解析 unified diff。主要剩餘問題在後端 Git metadata/bytes fidelity。
- POSIX `O_NOFOLLOW`/`dir_fd` 不能解決 Windows junction/reparse race；Windows 必須使用 handle-relative NT API 與 file identity。
- 一般 filesystem 無法提供真正的跨檔 atomic commit；計劃承諾 durable journal、rollback 與 recovery/reconciliation。
- Claude Code/Codex 只作 UX/能力參考；安全要求以 Mochi 與本地 `reference/` 原始碼為依據。

## 已確立的架構決策

除非重新提出 threat model 與替代方案，接手者不應隨意改動以下契約：

- 使用 versioned canonical `AuthorizationEnvelope`；SHA-256 `request_digest` 綁定 requester、session、task、workspace identity、policy 及完整 file/exec request。
- Volatile manifest 欄位不進 digest，避免 digest cycle。
- `security.change_contract_mode = observe | enforce` 只控制 file contract。
- `sandbox.mode = off | preferred | required` 只控制 exec containment；`required` 在任何 file rollout mode 都必須 fail closed。
- POSIX mutation 使用 descriptor-relative `dir_fd` + `O_NOFOLLOW`。
- Windows mutation 使用 pinned handles 與 handle-relative NT API；enforce mode 禁止 absolute-path fallback。
- 多檔變更使用 durable journal 與 identity/content/metadata reconciliation，不宣稱跨檔真正 atomic。
- Undo 使用 server-side before blob、retention state，以及 applied identity/content/metadata CAS，並重走相同 mutation policy。
- Auto Review 的 `input_digest` 必須等於 authorization-envelope request digest；未來 model reviewer 只能增加風險，不得覆蓋 hard deny/approval gate。
- `approve_and_save_rule` 在 approval resolve 時，與 outbox insert 共用同一 SQLite transaction；execution 之後另走 consume-once CAS，YAML delivery 絕不能重跑 execution。
- Config 使用 exact-byte SHA-256 revision；POSIX 用 `flock`、Windows 用 `LockFileEx`，搭配 atomic replace 與明確 conflict handling。
- 每個 Windows sandbox run 使用獨立 AppContainer SID/profile；cleanup 只能移除該 run 擁有的 exact ACE，且 journal/ACL mutation 需跨程序 lock。

## 實作順序

必須按計劃順序執行，因為後續 Task 依賴前面的資料契約與 migration。

### Release A：Mutation foundation

- Task 0：建立互相獨立的 file rollout 與 sandbox mode contract。
- Task 1：Authorization envelope、file identity、pinned target、POSIX/Windows SafeFilesystem。
- Task 2：Crash-safe 單檔寫入與 metadata preservation。
- Task 3：Immutable manifest、blob references、applied records、durable journal 與 recovery。

Release A 不得啟用 file enforce。

### Release B：Approval 與 Undo

- Task 4：TTL、conditional resolve、consume lease、execution idempotency、rule side-effect outbox schema。
- Task 5：Preview/approval/execution digest binding 與 edited-patch re-preview。
- Task 6：Server-authoritative Undo、retention/GC、applied identity/content/metadata CAS。
- Task 7：把 read/write scope 傳遞到所有 runtime/API/CLI consumer。

### Release C：Review、Audit、Config durability

- Task 8：Structured deterministic Auto Review。
- Task 9：Central redaction/audit、config snapshot + ETag/CAS、所有 config consumer、outbox worker、Web settings/outbox-status contract。

Task 9 必須包含第三輪審查後補入的前端工作：

- `web/src/lib/api.ts` 保存 settings revision，並傳 quoted `If-Match`。
- `web/src/app/settings/page.tsx` 遇到 HTTP 409 時保留未儲存表單，要求 reload/review/retry，不能 blind retry。
- `web/src/components/chat/TaskPanel.tsx` 分開顯示 execution status 與 `rule_persistence_status`。
- `web/scripts/test-settings-revision-contract.mjs` 與 approval contract test 覆蓋 revision/conflict 和 pending/retrying/delivered/failed projection。

### Release D：OS containment 與 diff fidelity

- Task 10：Linux bubblewrap、Windows native AppContainer broker、真 capability probe、per-run ACL ownership、packaging 與 platform CI。
- Task 11：Git-native NUL-safe status/diff、schema migration、rollout integration、telemetry 與完整 adversarial release matrix。

## Reviewer 紀錄

同一位 Codex reviewer 共審查三輪：

- 第一輪提出 12 個 blocking issues，均已修訂。
- 第二輪提出 8 個 blocking issues，涵蓋 metadata、journal recovery、outbox delivery、rollout consumers 與 AppContainer ACL concurrency，均已修訂。
- 第三輪確認前述 8 項已關閉，另找到 1 個 blocker：Web settings 未承接 revision/`If-Match`，approval UI 也未承接 outbox pending/failed 狀態。
- 最後這個 frontend/API/test gap 已補入 Task 9，並完成機械檢查。
- 因 writing-plans workflow 限制最多三輪，沒有進行第四輪獨立複審。因此最後的 Task 9 amendment 是 self-verified，不應宣稱已獲 reviewer 最終無條件批准。

## 工作樹警告

交接當下：

- Branch：`main`
- 狀態：`main...origin/main [ahead 22]`
- Python、tests、Web 皆有大量 modified/untracked files。

接手規則：

1. 所有既有 modified/untracked files 都視為使用者資產。
2. 禁止 `git reset --hard`、`git checkout --`、廣泛 cleanup 或覆蓋不相關變更。
3. 實作前優先建立 dedicated worktree/branch，並按 Task 做小 commit。
4. 若無法在不搬移或提交使用者變更的前提下安全建立 worktree，應停止並詢問使用者。
5. 開始前重新執行 `rtk git status --short --branch`，因交接快照可能已改變。

已知 untracked clutter 包含 `.playwright-cli/`、`.pytest-tmp/`、`.tokensave/`、output/screenshots 等目錄及其他使用者檔案；不得順手刪除。

## 實作與驗證紀律

- 依每個 Task 的 TDD 步驟先寫 focused tests。
- 所有 shell commands 都以 `rtk` 開頭；command chain 的每個 segment 也要各自加 `rtk`。
- 優先用 `apply_patch`。若 Windows restricted-token helper 出現既知 split-root ACL error，才改用限定工作區的 PowerShell/.NET file API，並保留 encoding/newline。
- Observe mode 保持 legacy file behavior，只記 would-reject telemetry；達到 release gate 前不得切 enforce。
- File rollout mode 絕不能弱化 `sandbox.mode=required`。
- Fake sandbox adapter 只供 unit test，不能取代 Windows/Linux 真 backend release gate。
- 必須測 crash points、concurrency、PID reuse、lease recovery，不只測 happy path。
- Metadata 無法 capture/apply 時，enforce mode 必須 fail closed。
- Content hash 相同但 inode/file-ID 不同仍是 conflict，不能只比 hash。
- Live blob reference 絕不能被 GC；expired blob 不得執行 Undo。

計劃文件已通過：

- Markdown fences 成對。
- 無 trailing whitespace 或 `.plan-edit-*` 暫存檔。
- 所有 PowerShell 範例使用 `rtk`。
- `rtk git diff --check` 通過。

以上只是文件檢查，不是功能實作測試。

## 建議下一個 Session 的第一步

1. 讀取本交接文件、Stage 6 memory 與完整計劃。
2. 檢查最新 Git status，先與使用者確認 dirty-worktree/worktree 策略。
3. 只開始 Task 0：建立兩個獨立 rollout modes 與 2 x 3 behavior matrix tests。
4. Task 0 focused tests 通過前，不要提早修改 SafeFilesystem 或 approval schema。
5. Task 0 單獨 commit 後，再開始 Task 1。

建議交接 prompt：

> 先讀取 `.claude/skills/agent-memory/memories/project-status/protected-workspace-hardening-plan-handoff-2026-07-16.md` 與 `docs/superpowers/plans/2026-07-15-protected-workspace-approval-hardening.md`。確認 dirty-worktree 策略後，只實作 Task 0，執行 focused tests，且不要修改任何不相關的使用者檔案。
