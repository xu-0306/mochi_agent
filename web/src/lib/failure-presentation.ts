export const FAILURE_ENVELOPE_V1_KINDS = [
  'backend_error',
  'tool_error',
  'tool_denied',
  'runtime_steering',
  'context_overflow',
  'output_truncated',
  'empty_response',
  'invalid_tool_call',
  'cancelled',
] as const

export type FailureEnvelopeV1Kind = (typeof FAILURE_ENVELOPE_V1_KINDS)[number]
export type FailurePresentationTone = 'error' | 'warning' | 'info'
export type FailurePresentationAction =
  | 'retry'
  | 'retry_tool'
  | 'request_approval'
  | 'continue'
  | 'reduce_context'
  | 'resume'

/**
 * A deliberately permissive client-side view of a versioned failure envelope.
 * Runtime validation belongs to the producer; this consumer only needs the
 * version and kind, while retaining forward-compatible fields untouched.
 */
export type FailureEnvelopeInput = Readonly<{
  schema_version: unknown
  kind: unknown
  [field: string]: unknown
}>

export interface FailurePresentation {
  tone: FailurePresentationTone
  title: string
  detail: string
  allowedActions: readonly FailurePresentationAction[]
}

export class FailurePresentationVersionError extends Error {
  constructor() {
    super('Unsupported failure-envelope schema major.')
    this.name = 'FailurePresentationVersionError'
  }
}

export class FailurePresentationValidationError extends Error {
  constructor() {
    super('Failure-envelope schema version is malformed.')
    this.name = 'FailurePresentationValidationError'
  }
}

const RETRY: readonly FailurePresentationAction[] = ['retry']
const RETRY_TOOL: readonly FailurePresentationAction[] = ['retry_tool']
const REQUEST_APPROVAL: readonly FailurePresentationAction[] = ['request_approval']
const CONTINUE: readonly FailurePresentationAction[] = ['continue']
const REDUCE_CONTEXT: readonly FailurePresentationAction[] = ['reduce_context']
const RESUME: readonly FailurePresentationAction[] = ['resume']
const NO_ACTIONS: readonly FailurePresentationAction[] = []

const MAX_FAILURE_DETAIL_CHARS = 1_200

const CLIENT_BACKEND_FAILURE: FailureEnvelopeInput = Object.freeze({
  schema_version: '1.0',
  kind: 'backend_error',
  origin: 'backend',
  recoverability: 'manual_retry',
  retry_policy: 'manual',
  terminal: true,
  inject_into_model_context: false,
  telemetry_key: 'failure.backend_error',
  ui_hint: 'retry',
  diagnostics_ref: null,
})

export function createClientBackendFailure(): FailureEnvelopeInput {
  return CLIENT_BACKEND_FAILURE
}

/**
 * Retain actionable backend details without rendering credentials, URL query
 * secrets, or unbounded upstream response bodies in the chat timeline.
 */
export function sanitizeFailureDetail(value: unknown): string | null {
  if (typeof value !== 'string') {
    return null
  }

  let detail = value.replace(/\u0000/g, '').trim()
  if (!detail) {
    return null
  }

  detail = detail
    .replace(/\bBearer\s+[A-Za-z0-9._~+/-]{8,}/gi, 'Bearer [REDACTED]')
    .replace(
      /\b(api[_-]?key|token|secret|password|authorization)\s*([:=])\s*([^\s,;"'}\]]+)/gi,
      '$1$2 [REDACTED]'
    )
    .replace(
      /([?&](?:api[_-]?key|token|secret|password|key)=)[^&#\s]+/gi,
      '$1[REDACTED]'
    )
    .replace(
      /(https?:\/\/)[^/\s:@]+:[^@\s/]+@/gi,
      '$1[REDACTED]@'
    )

  return detail.length > MAX_FAILURE_DETAIL_CHARS
    ? detail.slice(0, MAX_FAILURE_DETAIL_CHARS - 1).trimEnd() + '…'
    : detail
}
const PRESENTATIONS: Readonly<Record<FailureEnvelopeV1Kind, FailurePresentation>> = {
  backend_error: {
    tone: 'error',
    title: 'Service issue',
    detail: 'The request could not be completed. Try again when you are ready.',
    allowedActions: RETRY,
  },
  tool_error: {
    tone: 'error',
    title: 'Tool did not complete',
    detail: 'The requested tool could not finish. Review the task and try the tool again.',
    allowedActions: RETRY_TOOL,
  },
  tool_denied: {
    tone: 'warning',
    title: 'Tool approval needed',
    detail: 'This action needs approval before it can continue.',
    allowedActions: REQUEST_APPROVAL,
  },
  runtime_steering: {
    tone: 'info',
    title: 'Task updated',
    detail: 'The task was adjusted and can continue safely.',
    allowedActions: CONTINUE,
  },
  context_overflow: {
    tone: 'warning',
    title: 'Task needs a smaller context',
    detail: 'The task needs a shorter context before it can continue.',
    allowedActions: REDUCE_CONTEXT,
  },
  output_truncated: {
    tone: 'warning',
    title: 'Response was incomplete',
    detail: 'The response ended early. Continue to receive the remaining work.',
    allowedActions: CONTINUE,
  },
  empty_response: {
    tone: 'error',
    title: 'No response received',
    detail: 'No usable response was returned. Try again when you are ready.',
    allowedActions: RETRY,
  },
  invalid_tool_call: {
    tone: 'error',
    title: 'Tool request needs repair',
    detail: 'The tool request could not be used. Try again after reviewing the task.',
    allowedActions: RETRY,
  },
  cancelled: {
    tone: 'info',
    title: 'Task cancelled',
    detail: 'The task was stopped before completion. Resume it when you are ready.',
    allowedActions: RESUME,
  },
}

const GENERIC_PRESENTATION: FailurePresentation = {
  tone: 'error',
  title: 'Task could not continue',
  detail: 'The task could not be completed. Review it before starting another attempt.',
  allowedActions: NO_ACTIONS,
}

function supportedSchemaMajor(value: unknown): void {
  if (typeof value !== 'string') {
    throw new FailurePresentationValidationError()
  }

  const match = /^(?<major>[0-9]+)(?:\.(?<minor>[0-9]+))?$/.exec(value)
  if (!match?.groups?.major) {
    throw new FailurePresentationValidationError()
  }
  if (Number(match.groups.major) !== 1) {
    throw new FailurePresentationVersionError()
  }
}

function isFailureEnvelopeV1Kind(value: unknown): value is FailureEnvelopeV1Kind {
  return typeof value === 'string' && FAILURE_ENVELOPE_V1_KINDS.includes(value as FailureEnvelopeV1Kind)
}

/**
 * Convert a v1 envelope into fixed UI copy without exposing diagnostics or
 * extension fields. The input is never modified, so v1 minor additions remain
 * available to downstream consumers without affecting this presentation.
 */
export function presentFailure(envelope: FailureEnvelopeInput): FailurePresentation {
  if (typeof envelope !== 'object' || envelope === null) {
    throw new FailurePresentationValidationError()
  }
  supportedSchemaMajor(envelope.schema_version)
  const presentation = isFailureEnvelopeV1Kind(envelope.kind)
    ? PRESENTATIONS[envelope.kind]
    : GENERIC_PRESENTATION
  return {
    ...presentation,
    allowedActions: [...presentation.allowedActions],
  }
}
