import assert from 'node:assert/strict'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

const moduleUrl = pathToFileURL(
  path.join(process.cwd(), 'src/lib/reasoning-display.ts')
).href

const { compactReasoningStepsForDisplay } = await import(moduleUrl)

function step(id, content, toolMeta = {}, source = 'runtime_progress') {
  return {
    id,
    type: 'status',
    content,
    timestamp: new Date(`2026-07-15T12:16:0${id}.000Z`),
    source,
    toolMeta,
  }
}

const steps = [
  step('1', 'Mochi progress: ReAct iteration 1/10', {
    kind: 'react_iteration_progress',
    iteration: 1,
  }),
  step('2', 'The requested file mutation has not completed yet.', {
    kind: 'file_artifact_missing',
  }),
  step('3', 'Mochi progress: ReAct iteration 2/10', {
    kind: 'react_iteration_progress',
    iteration: 2,
  }),
  step('4', 'Recovered tool call markup from assistant text.', {
    kind: 'tool_protocol_recovery',
  }),
]

const completed = compactReasoningStepsForDisplay(steps, { isStreaming: false })
assert.deepEqual(
  completed.map((item) => item.id),
  ['2', '4'],
  'Completed traces should hide routine ReAct iteration progress while preserving actionable runtime statuses'
)

const streaming = compactReasoningStepsForDisplay(steps, { isStreaming: true })
assert.deepEqual(
  streaming.map((item) => item.id),
  ['2', '3', '4'],
  'Live traces should keep only the latest ReAct iteration progress event'
)

const legacy = compactReasoningStepsForDisplay([
  step('5', 'Mochi progress: ReAct iteration 3/10', {}, 'runtime_progress'),
], { isStreaming: false })
assert.equal(legacy.length, 0, 'Legacy iteration progress without kind metadata should also be hidden')

console.log('ok')
