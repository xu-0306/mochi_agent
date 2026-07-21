import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

import { extractPatchPreviewResult } from '../src/lib/file-change-preview.ts'

const preview = extractPatchPreviewResult({
  valid: true,
  change_set_id: 'change-1',
  request_digest: 'a'.repeat(64),
  expires_at: '2026-07-21T08:00:00+00:00',
  policy_version: 'file-policy-v1:test',
  change_contract_mode: 'enforce',
  replacement_approval_id: 'approval-2',
  approval_state: 'replacement_pending',
  would_reject_edited_patch: false,
  file_changes: [],
})

assert.equal(preview.changeSetId, 'change-1')
assert.equal(preview.requestDigest, 'a'.repeat(64))
assert.equal(preview.changeContractMode, 'enforce')
assert.equal(preview.replacementApprovalId, 'approval-2')
assert.equal(preview.approvalState, 'replacement_pending')

const scriptDir = fileURLToPath(new URL('.', import.meta.url))
const taskPanel = readFileSync(new URL('../src/components/chat/TaskPanel.tsx', import.meta.url), 'utf8')
const taskStore = readFileSync(new URL('../src/lib/stores/task-store.ts', import.meta.url), 'utf8')
const api = readFileSync(new URL('../src/lib/api.ts', import.meta.url), 'utf8')

assert.match(taskPanel, /changeContractMode === 'observe'/)
assert.match(taskPanel, /replacementApprovalId \?\? approval\.approval_id/)
assert.match(taskPanel, /Digest:/)
assert.match(taskPanel, /Expires:/)
assert.match(taskPanel, /Superseded by:/)
assert.match(taskStore, /preview\.replacementApprovalId/)
assert.doesNotMatch(api, /edited_patch_text:/)

console.log('approval contract checks passed')
