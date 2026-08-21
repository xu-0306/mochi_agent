import type { Message } from './chat'
import { createClientBackendFailure, type FailureEnvelopeInput } from './failure-presentation'

function text(value: unknown): string | null {
  return typeof value === 'string' && value.trim().length > 0 ? value : null
}

function requestDetail(error: unknown): string | null {
  if (typeof error === 'string') return text(error)
  if (error instanceof Error) return text(error.message)
  if (typeof error === 'object' && error !== null && 'message' in error) return text(error.message)
  return null
}

function requestCode(error: unknown): string {
  if (typeof error === 'object' && error !== null && 'status' in error) {
    const status = error.status
    if (typeof status === 'number' && Number.isInteger(status) && status > 0) return 'HTTP_' + status
  }
  return 'CHAT_REQUEST_FAILED'
}

export function getErrorEventContent(payload: Record<string, unknown>): string {
  return text(payload.error) ?? text(payload.message) ?? text(payload.content) ?? text(payload.final_answer) ?? text(payload.text) ?? text(payload.answer) ?? ''
}

export function createClientBackendErrorMessage(error: unknown, fallbackContent: string): Pick<Message, 'type' | 'eventType' | 'content' | 'errorCode' | 'failure'> {
  return { type: 'error', eventType: 'error', content: requestDetail(error) ?? fallbackContent, errorCode: requestCode(error), failure: createClientBackendFailure() }
}

export function createTimelineErrorMessage(input: { id: string; content: string; timestamp: Date; turnKey: string; errorCode?: string; failure?: FailureEnvelopeInput }): Message {
  return { id: input.id, type: 'error', eventType: 'error', content: input.content, timestamp: input.timestamp, turnKey: input.turnKey, turnId: input.turnKey, errorCode: input.errorCode, failure: input.failure ?? createClientBackendFailure(), isStreaming: false }
}