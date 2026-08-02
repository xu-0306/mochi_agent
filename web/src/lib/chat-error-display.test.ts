import assert from 'node:assert/strict'

import { formatChatErrorDiagnostics } from './chat-error-display.ts'

assert.equal(
  formatChatErrorDiagnostics('MODEL_PROVIDER_ACCESS_DENIED', {
    status_code: 403,
    backend_name: 'openai_compat',
  }),
  'MODEL_PROVIDER_ACCESS_DENIED · HTTP 403 · openai_compat'
)

assert.equal(
  formatChatErrorDiagnostics('MODEL_REQUEST_FAILED', {
    statusCode: '502',
  }),
  'MODEL_REQUEST_FAILED · HTTP 502'
)

assert.equal(
  formatChatErrorDiagnostics(undefined, {
    status_code: 'not-a-status',
    backend_name: ' ',
  }),
  undefined
)

console.log('ok')
