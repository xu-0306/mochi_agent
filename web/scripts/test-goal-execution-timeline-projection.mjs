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

  const projectionViewSource = await fs.readFile(
    path.join(sourceLibDir, 'goal-execution-projection-store.ts'),
    'utf8'
  )
  await fs.writeFile(
    path.join(tempLibDir, 'goal-execution-projection-store.ts'),
    projectionViewSource
      .replaceAll("from '@/lib/api'", "from './api.stub.ts'")
      .replaceAll("from '@/lib/execution-transcript'", "from './execution-transcript.ts'")
  )

  const moduleUrl = pathToFileURL(path.join(tempLibDir, 'execution-transcript.ts')).href
  const { projectGoalSurfaceExecutionEvents } = await import(moduleUrl)
  const projectionViewUrl = pathToFileURL(
    path.join(tempLibDir, 'goal-execution-projection-store.ts')
  ).href
  const { buildGoalExecutionProjectionView, isGoalSurfaceSubagentVisible } = await import(
    projectionViewUrl
  )

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

  function subagentSummary(subagentId, overrides = {}) {
    return {
      subagentId,
      parentType: 'agent_run',
      parentId: 'goal-1',
      roleId: `${subagentId}-role`,
      title: subagentId,
      status: 'running',
      promptPreview: null,
      summary: null,
      outputPreview: null,
      eventCount: 0,
      createdAt: '2026-07-01T00:00:00.000Z',
      updatedAt: '2026-07-01T00:00:00.000Z',
      ...overrides,
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

  const projectionView = buildGoalExecutionProjectionView({
    executionTimelineEvents: [
      executionEvent('runtime_blocked', {
        dedupeKey: 'approval:goal-1',
        summary: 'Approval required',
      }),
      executionEvent('runtime_blocked', {
        dedupeKey: 'approval:goal-1',
        summary: 'Hidden blocker copy',
        visibility: 'hidden',
      }),
      executionEvent('subagent_completed', {
        eventId: 'worker-1-complete',
        subagentId: 'worker-1',
        roleId: 'worker-1-role',
        summary: 'Worker 1 completed',
      }),
      executionEvent('subagent_completed', {
        eventId: 'worker-4-complete',
        subagentId: 'worker-4',
        roleId: null,
        summary: 'Anonymous worker completed',
      }),
    ],
    sessionSubagents: [
      subagentSummary('worker-1', {
        title: 'Worker 1',
        roleId: 'worker-1-role',
        status: 'running',
      }),
      subagentSummary('worker-2', {
        title: 'Reviewer',
        roleId: 'worker-2-role',
        status: 'blocked',
      }),
      subagentSummary('worker-3', {
        title: 'Analyst',
        roleId: 'worker-3-role',
        status: 'completed',
        summary: 'Compiled notes',
      }),
      subagentSummary('worker-4', {
        title: null,
        roleId: null,
        status: 'running',
      }),
      subagentSummary('worker-5', {
        title: 'Silent worker',
        roleId: 'worker-5-role',
        status: 'running',
      }),
    ],
    agentRunSubagents: [
      subagentSummary('worker-1', {
        title: 'Worker 1',
        roleId: 'worker-1-role',
        status: 'completed',
        summary: 'Finished execution',
        eventCount: 2,
        updatedAt: '2026-07-01T00:05:00.000Z',
      }),
    ],
  })

  assert.equal(projectionView.allVisibleSubagents.length, 5)
  assert.equal(
    projectionView.allVisibleSubagents.find((item) => item.subagentId === 'worker-1')?.summary,
    'Finished execution'
  )
  assert.deepEqual(
    projectionView.goalSurfaceTimelineEvents.map((event) => event.summary),
    ['Approval required', 'Worker 1 completed', 'Anonymous worker completed']
  )
  assert.deepEqual(
    (projectionView.goalSurfaceTimelineEventsById.get('worker-1') ?? []).map(
      (event) => event.summary
    ),
    ['Worker 1 completed']
  )
  assert.deepEqual(
    (projectionView.goalSurfaceTimelineEventsById.get('worker-4') ?? []).map(
      (event) => event.summary
    ),
    ['Anonymous worker completed']
  )
  assert.deepEqual(
    projectionView.goalSurfaceSubagents.map((item) => item.subagentId).sort(),
    ['worker-1', 'worker-2', 'worker-3', 'worker-4']
  )
  assert.equal(projectionView.goalSurfaceSubagents.some((item) => item.subagentId === 'worker-5'), false)
  assert.deepEqual(
    [...projectionView.subagentTimelineEventsById.keys()].sort(),
    ['worker-1', 'worker-2', 'worker-3', 'worker-4']
  )
  assert.equal((projectionView.subagentTimelineEventsById.get('worker-2') ?? []).length, 0)
  assert.equal((projectionView.subagentTimelineEventsById.get('worker-3') ?? []).length, 0)

  assert.equal(
    isGoalSurfaceSubagentVisible(
      subagentSummary('worker-6', {
        title: 'Needs approval',
        roleId: 'worker-6-role',
        status: 'awaiting_approval',
      }),
      []
    ),
    true
  )
  assert.equal(
    isGoalSurfaceSubagentVisible(
      subagentSummary('worker-7', {
        title: 'No summary yet',
        roleId: 'worker-7-role',
        status: 'running',
      }),
      []
    ),
    false
  )

  const pageSource = await fs.readFile(path.join(workspaceRoot, 'src/app/page.tsx'), 'utf8')
  assert.match(
    pageSource,
    /buildGoalExecutionProjectionView\(\{\s*executionTimelineEvents,\s*sessionSubagents,\s*agentRunSubagents,\s*\}\)/,
    'page.tsx should derive the goal execution view via the extracted helper'
  )
  assert.doesNotMatch(
    pageSource,
    /function transcriptEventsForSubagent\(/,
    'page.tsx should not retain the subagent transcript grouping helper'
  )
  assert.doesNotMatch(
    pageSource,
    /function isGoalSurfaceSubagentVisible\(/,
    'page.tsx should not retain the goal-surface subagent visibility helper'
  )

  console.log('goal execution timeline projection assertions passed')
} finally {
  await fs.rm(tempLibDir, { recursive: true, force: true })
}
