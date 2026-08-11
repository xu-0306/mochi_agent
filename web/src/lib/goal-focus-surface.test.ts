import assert from 'node:assert/strict'
import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const sourceLibDir = path.dirname(fileURLToPath(import.meta.url))
const tempLibDir = await fs.mkdtemp(path.join(os.tmpdir(), 'mochi-goal-focus-surface-'))

try {
  const [goalFocusSurfaceSource, goalProposalCopySource] = await Promise.all([
    fs.readFile(path.join(sourceLibDir, 'goal-focus-surface.ts'), 'utf8'),
    fs.readFile(path.join(sourceLibDir, 'goal-proposal-copy.ts'), 'utf8'),
  ])
  await Promise.all([
    fs.writeFile(path.join(tempLibDir, 'goal-proposal-copy.ts'), goalProposalCopySource),
    fs.writeFile(
      path.join(tempLibDir, 'goal-focus-surface.ts'),
      goalFocusSurfaceSource.replaceAll(
        "from '@/lib/goal-proposal-copy'",
        "from './goal-proposal-copy.ts'"
      )
    ),
  ])

  const moduleUrl = pathToFileURL(path.join(tempLibDir, 'goal-focus-surface.ts')).href
  const { buildGoalFocusCallout } = await import(moduleUrl)
  const rawDiagnostic = 'internal diagnostic: token=secret-value diagnostics_ref=/private/log'
  const blocker = {
    summary: null,
    recommendedAction: null,
    latestError: rawDiagnostic,
    approvalCount: 0,
    approvalIds: [],
    approvalToolNames: [],
    blockedTools: [],
    blockedDomains: [],
    blockNetworkUsage: false,
  }

  const blockedCallout = buildGoalFocusCallout({
    userMessage: 'Continue the task.',
    pendingApprovalCount: 0,
    blocker,
    goalDisplayState: 'blocked',
  })
  assert.ok(blockedCallout)
  assert.equal(blockedCallout.tone, 'warning')
  assert.equal(
    blockedCallout.message.includes(rawDiagnostic),
    false,
    'blocked callout must not project a raw latest_error diagnostic'
  )

  const approvalCallout = buildGoalFocusCallout({
    userMessage: 'Continue the task.',
    pendingApprovalCount: 1,
    blocker,
    goalDisplayState: 'blocked',
  })
  assert.ok(approvalCallout)
  assert.equal(approvalCallout.tone, 'warning')
  assert.equal(
    approvalCallout.message.includes(rawDiagnostic),
    false,
    'approval callout must not project a raw latest_error diagnostic'
  )

  console.log('goal focus surface failure-safety assertions passed')
} finally {
  await fs.rm(tempLibDir, { recursive: true, force: true })
}
