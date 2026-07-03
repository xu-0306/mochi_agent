import assert from 'node:assert/strict'
import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

const workspaceRoot = process.cwd()
const sourceLibDir = path.join(workspaceRoot, 'src/lib')
const tempLibDir = await fs.mkdtemp(path.join(os.tmpdir(), 'mochi-execution-transcript-'))

await fs.writeFile(
  path.join(tempLibDir, 'api.stub.ts'),
  [
    'export interface ExecutionTranscriptEvent {',
    '  type: string',
    '  status?: string | null',
    '  metadata: Record<string, unknown>',
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

const { projectWorkflowRunEventToSessionEvent } = await import(moduleUrl)

function projectAssistant(event) {
  return projectWorkflowRunEventToSessionEvent(
    {
      type: 'assistant_message',
      content: 'Projected assistant reply',
      ...event,
    },
    {
      runId: 'run-chat-projection',
      turnId: 'turn-chat-projection',
    }
  )
}

const explicitDurableConversation = projectAssistant({
  metadata: {
    acknowledgement: true,
    durability: 'durable',
    projection_lane: 'conversation',
  },
})

assert.ok(explicitDurableConversation, 'explicit durable conversation messages should project')
assert.equal(explicitDurableConversation.type, 'message')
assert.equal(explicitDurableConversation.role, 'assistant')
assert.equal(explicitDurableConversation.content, 'Projected assistant reply')

assert.equal(
  projectAssistant({
    durability: 'transient',
    metadata: {},
  }),
  null,
  'transient assistant messages should stay out of durable chat'
)

assert.equal(
  projectAssistant({
    metadata: {
      projectionLane: 'goal_surface',
    },
  }),
  null,
  'goal-surface assistant messages should stay out of durable chat'
)

assert.equal(
  projectAssistant({
    visibility: 'hidden',
    metadata: {
      acknowledgement: false,
      durability: 'durable',
      projection_lane: 'conversation',
    },
  }),
  null,
  'hidden assistant messages should never project into durable chat'
)

assert.equal(
  projectAssistant({
    metadata: {
      acknowledgement: true,
    },
  }),
  null,
  'legacy acknowledgement-only assistant messages should remain filtered'
)

console.log('execution transcript chat projection assertions passed')

await fs.rm(tempLibDir, { recursive: true, force: true })
