import assert from 'node:assert/strict'

const moduleUrl = new URL('../../web/src/lib/agent-run-route-id.ts', import.meta.url)
const { normalizeAgentRunRouteId } = await import(moduleUrl.href)

assert.equal(normalizeAgentRunRouteId('550e8400-e29b-41d4-a716-446655440000'), '550e8400-e29b-41d4-a716-446655440000', 'UUIDs remain stable')
assert.equal(normalizeAgentRunRouteId('run%20%2F%20linked%3F%26'), 'run / linked?&', 'encoded route segments decode once')
assert.equal(normalizeAgentRunRouteId('run / linked?&'), 'run / linked?&', 'already-decoded route segments remain stable')
assert.equal(normalizeAgentRunRouteId('run%2'), 'run%2', 'malformed percent sequences remain stable')

const rawRunId = normalizeAgentRunRouteId('run%20%2F%20linked%3F%26')
const apiPath = '/v1/agent-runs/' + encodeURIComponent(rawRunId)
assert.equal(apiPath, '/v1/agent-runs/run%20%2F%20linked%3F%26', 'API encoding occurs exactly once after route normalization')
assert.equal(apiPath.includes('%25'), false, 'a legitimate encoded route segment never becomes percent-double-encoded')

console.log('agent run route id: 6 assertions passed')