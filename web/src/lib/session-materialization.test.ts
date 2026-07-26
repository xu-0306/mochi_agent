import assert from 'node:assert/strict'

import { resolveMaterializedSecurityOverride } from './session-materialization.ts'

const draftOverride = { autonomy_mode: 'auto_review' } as const
const createdOverride = { autonomy_mode: 'strict' } as const
const detailOverride = { autonomy_mode: 'trusted_workspace' } as const

assert.deepEqual(
  resolveMaterializedSecurityOverride(null, null),
  null,
  'local draft state must not masquerade as a persisted runtime policy'
)
assert.deepEqual(
  resolveMaterializedSecurityOverride(null, createdOverride),
  createdOverride,
  'the create response should provide the persisted initial override'
)
assert.deepEqual(
  resolveMaterializedSecurityOverride(detailOverride, createdOverride),
  detailOverride,
  'the persisted session detail should be authoritative when present'
)

console.log('ok')
