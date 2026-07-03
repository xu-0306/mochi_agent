import assert from 'node:assert/strict'
import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

const workspaceRoot = process.cwd()
const sourceLibDir = path.join(workspaceRoot, 'src/lib')
const tempLibDir = await fs.mkdtemp(path.join(os.tmpdir(), 'mochi-goal-execution-projection-'))

try {
  await fs.writeFile(
    path.join(tempLibDir, 'api.stub.ts'),
    [
      'export interface ExecutionTranscriptEvent {',
      '  type: string',
      '  status?: string | null',
      '  metadata: Record<string, unknown>',
      '  parentType?: string | null',
      '  parentId?: string | null',
      '  subagentId?: string | null',
      '  roleId?: string | null',
      '  summary?: string | null',
      '  content?: string | null',
      '  eventId?: string | null',
      '  dedupeKey?: string | null',
      '  visibility?: string | null',
      '  durability?: string | null',
      '  projectionLane?: string | null',
      '  deliveryMode?: string | null',
      '  deliveryReason?: string | null',
      '  messageId?: string | null',
      '}',
      'export interface SubagentTranscriptSummary {}',
      '',
    ].join('\n')
  )

  const subagentProtocolSource = await fs.readFile(
    path.join(sourceLibDir, 'subagent-protocol-events.ts'),
    'utf8'
  )
  await fs.writeFile(
    path.join(tempLibDir, 'subagent-protocol-events.ts'),
    subagentProtocolSource.replaceAll("from '@/lib/api'", "from './api.stub.ts'")
  )

  const executionTranscriptSource = await fs.readFile(
    path.join(sourceLibDir, 'execution-transcript.ts'),
    'utf8'
  )
  await fs.writeFile(
    path.join(tempLibDir, 'execution-transcript.ts'),
    executionTranscriptSource
      .replaceAll("from '@/lib/api'", "from './api.stub.ts'")
      .replaceAll("from '@/lib/subagent-protocol-events'", "from './subagent-protocol-events.ts'")
  )

  const moduleUrl = pathToFileURL(path.join(tempLibDir, 'execution-transcript.ts')).href
  const { projectGoalSurfaceExecutionEvents } = await import(moduleUrl)

  function executionEvent(type, overrides = {}) {
    return {
      type,
      status: type === 'runtime_blocked' ? 'blocked' : 'completed',
      metadata: {},
      parentType: 'agent_run',
      parentId: 'goal-1',
      subagentId: null,
      roleId: null,
      summary: null,
      content: null,
      eventId: null,
      dedupeKey: null,
      visibility: null,
      durability: null,
      projectionLane: null,
      ...overrides,
      metadata: {
        ...(overrides.metadata ?? {}),
      },
    }
  }

  const firstBlocker = executionEvent('runtime_blocked', {
    dedupeKey: 'approval:exec',
    summary: 'First blocker',
    content: 'Waiting for approval.',
  })
  const latestBlocker = executionEvent('runtime_blocked', {
    dedupeKey: 'approval:exec',
    summary: 'Latest blocker',
    content: 'Still waiting for approval.',
  })
  const dedupedByDedupeKey = projectGoalSurfaceExecutionEvents([
    firstBlocker,
    executionEvent('subagent_completed', {
      eventId: 'subagent-completed-1',
      subagentId: 'worker-1',
      roleId: 'worker-1',
      summary: 'Subagent finished a separate step.',
    }),
    latestBlocker,
  ])

  assert.deepEqual(
    dedupedByDedupeKey.map((event) => event.type),
    ['subagent_completed', 'runtime_blocked'],
    'same dedupeKey should collapse across the projected list and keep the latest row'
  )
  assert.equal(dedupedByDedupeKey[1], latestBlocker)

  const identicalButDistinctIds = projectGoalSurfaceExecutionEvents([
    executionEvent('runtime_blocked', {
      eventId: 'runtime-blocked-1',
      summary: 'Same blocker copy',
      content: 'Same blocker copy',
    }),
    executionEvent('runtime_blocked', {
      eventId: 'runtime-blocked-2',
      summary: 'Same blocker copy',
      content: 'Same blocker copy',
    }),
  ])

  assert.equal(
    identicalButDistinctIds.length,
    2,
    'different source identities must both remain even when their content matches'
  )
  assert.deepEqual(
    identicalButDistinctIds.map((event) => event.eventId),
    ['runtime-blocked-1', 'runtime-blocked-2']
  )

  const visibleBlocker = executionEvent('runtime_blocked', {
    dedupeKey: 'approval:visible',
    summary: 'Visible blocker',
  })
  const filteredEventsIgnoredForDedupe = projectGoalSurfaceExecutionEvents([
    visibleBlocker,
    executionEvent('subagent_completed', {
      eventId: 'subagent-completed-visible',
      subagentId: 'worker-2',
      roleId: 'worker-2',
      summary: 'Visible goal-surface event',
    }),
    executionEvent('runtime_blocked', {
      dedupeKey: 'approval:visible',
      summary: 'Hidden blocker',
      visibility: 'hidden',
    }),
    executionEvent('runtime_blocked', {
      dedupeKey: 'approval:visible',
      summary: 'Subagent detail blocker',
      projectionLane: 'subagent_detail',
    }),
  ])

  assert.deepEqual(
    filteredEventsIgnoredForDedupe.map((event) => event.summary),
    ['Visible blocker', 'Visible goal-surface event'],
    'hidden and non-goal-surface rows should be excluded before dedupe'
  )
  assert.equal(filteredEventsIgnoredForDedupe[0], visibleBlocker)

  const adjacentLegacyFirst = executionEvent('runtime_blocked', {
    summary: 'Legacy blocker',
    content: 'Legacy blocker',
  })
  const adjacentLegacySecond = executionEvent('runtime_blocked', {
    summary: 'Legacy blocker',
    content: 'Legacy blocker',
  })
  const adjacentLegacyDeduped = projectGoalSurfaceExecutionEvents([
    adjacentLegacyFirst,
    adjacentLegacySecond,
  ])

  assert.equal(adjacentLegacyDeduped.length, 1)
  assert.equal(
    adjacentLegacyDeduped[0],
    adjacentLegacySecond,
    'adjacent legacy duplicates should continue to replace with the latest row'
  )

  const separatedLegacyFirst = executionEvent('runtime_blocked', {
    summary: 'Separated legacy blocker',
    content: 'Separated legacy blocker',
  })
  const separatedLegacySecond = executionEvent('runtime_blocked', {
    summary: 'Separated legacy blocker',
    content: 'Separated legacy blocker',
  })
  const legacySeparator = executionEvent('subagent_completed', {
    eventId: 'subagent-completed-legacy',
    subagentId: 'worker-3',
    roleId: 'worker-3',
    summary: 'Legacy separator',
  })
  const separatedLegacyProjected = projectGoalSurfaceExecutionEvents([
    separatedLegacyFirst,
    legacySeparator,
    separatedLegacySecond,
  ])

  assert.deepEqual(
    separatedLegacyProjected,
    [separatedLegacyFirst, legacySeparator, separatedLegacySecond],
    'non-contract legacy duplicates should remain distinct when separated by another projected row'
  )

  console.log('goal execution timeline projection assertions passed')
} finally {
  await fs.rm(tempLibDir, { recursive: true, force: true })
}
