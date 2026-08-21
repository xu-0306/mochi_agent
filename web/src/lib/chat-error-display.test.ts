import assert from 'node:assert/strict'

import { formatChatErrorDiagnostics } from './chat-error-display.ts'
import { sanitizeFailureDetail } from './failure-presentation.ts'

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

const serverError = "Server error '503 Service Unavailable' for url 'https://cdn.coderelay.cn/v1/chat/completions'"
assert.equal(sanitizeFailureDetail(serverError), serverError)

assert.equal(
  sanitizeFailureDetail('OpenAI-compatible API error 403: Bearer secret-value'),
  'OpenAI-compatible API error 403: Bearer [REDACTED]'
)
console.log('ok')
