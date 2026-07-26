import assert from 'node:assert/strict'
import fs from 'node:fs/promises'
import path from 'node:path'

const source = await fs.readFile(
  path.join(process.cwd(), 'src/components/chat/ToolCallCard.tsx'),
  'utf8'
)

assert.match(
  source,
  /import \{ useToolWorkflowAggregateCall \} from '\@\/lib\/tool-workflow-aggregate'/,
  'ToolCallCard should consume the shared aggregate store'
)
assert.match(
  source,
  /useToolWorkflowAggregateCall\(sessionId, turnId, callId\)/,
  'ToolCallCard should scope aggregate lookup by session, turn, and call'
)
assert.match(
  source,
  /const aggregateAuthoritative = Boolean\(aggregateView\?\.authoritative && aggregateCall\)/,
  'ToolCallCard should require an authoritative aggregate call before claiming evidence'
)
assert.match(
  source,
  /executionStatus: aggregateAuthoritative \? aggregateCall\.execution_status : 'not_observed'/,
  'Missing aggregate execution evidence must remain not_observed'
)
assert.match(
  source,
  /verificationStatus: aggregateAuthoritative \? aggregateCall\.verification_status : 'not_observed'/,
  'Missing aggregate verification evidence must remain not_observed'
)
assert.match(
  source,
  /aggregateCall\?\.execution_status === 'succeeded'[\s\S]*\['verified', 'not_required'\]\.includes\(aggregateCall\.verification_status\)/,
  'Success styling should require terminal execution and verification evidence'
)
assert.match(
  source,
  /\['failed', 'abandoned', 'cancelled', 'unknown'\]\.includes\(aggregateCall\.execution_status\)/,
  'Known failure and uncertain states should remain visible as non-success states'
)
assert.match(
  source,
  /Durable tool workflow evidence is not available for this call\./,
  'Calls without aggregate evidence should show an explicit evidence blocker'
)

console.log('tool call card aggregate assertions passed')
