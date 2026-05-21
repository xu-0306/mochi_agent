/**
 * Mochi FastAPI client.
 * Most requests use relative /v1/* paths and rely on Next.js rewrites in development.
 * Long-running local model operations can bypass the dev proxy and hit the backend directly.
 */

import type {
  Message,
  MessageEventType,
  ReasoningStep,
  TokenStats,
} from '@/lib/chat'
import {
  appendInlineReasoningChunk,
  buildInlineReasoningStep,
  createInlineReasoningBuffer,
  extractInlineReasoning,
  finalizeInlineReasoningBuffer,
} from '@/lib/reasoning'

const API_BASE = '/v1'
const LOCAL_DEV_API_ORIGIN = 'http://127.0.0.1:8000'

type ApiPrimitive = string | number | boolean | null
type ApiValue = ApiPrimitive | ApiValue[] | { [key: string]: ApiValue }

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function getString(value: unknown): string | null {
  return typeof value === 'string' ? value : null
}

function getNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function getBoolean(value: unknown): boolean | null {
  return typeof value === 'boolean' ? value : null
}

function getStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return []
  }
  return value.filter((item): item is string => typeof item === 'string')
}

function getRecordArray(value: unknown): Record<string, unknown>[] {
  if (!Array.isArray(value)) {
    return []
  }
  return value.filter(isRecord)
}

function toIsoString(value: unknown): string {
  const numeric = getNumber(value)
  if (numeric !== null) {
    return new Date(numeric * 1000).toISOString()
  }

  const text = getString(value)
  if (text) {
    const parsed = Date.parse(text)
    if (!Number.isNaN(parsed)) {
      return new Date(parsed).toISOString()
    }
  }

  return new Date(0).toISOString()
}

function getApiMessage(payload: unknown, fallback: string): string {
  if (!isRecord(payload)) {
    return fallback
  }

  const detail = payload.detail
  if (typeof detail === 'string' && detail.length > 0) {
    return detail
  }

  const message = payload.message
  if (typeof message === 'string' && message.length > 0) {
    return message
  }

  const error = payload.error
  if (typeof error === 'string' && error.length > 0) {
    return error
  }

  return fallback
}

export class ApiError extends Error {
  readonly status: number
  readonly payload?: unknown

  constructor(status: number, message: string, payload?: unknown) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.payload = payload
  }
}

async function parseResponseBody(response: Response): Promise<unknown> {
  const contentType = response.headers.get('content-type') ?? ''

  if (contentType.includes('application/json')) {
    return response.json()
  }

  const text = await response.text()
  return text.length > 0 ? text : null
}

type RequestTarget = 'proxy' | 'direct'

function normalizeApiOrigin(origin: string): string {
  return origin.replace(/\/+$/, '')
}

function resolveApiUrl(path: string, target: RequestTarget = 'proxy'): string {
  if (target === 'direct' && typeof window !== 'undefined') {
    const configuredOrigin = process.env.NEXT_PUBLIC_MOCHI_API_BASE_URL?.trim()
    if (configuredOrigin) {
      return `${normalizeApiOrigin(configuredOrigin)}${API_BASE}${path}`
    }

    const { hostname, port, protocol } = window.location
    if ((hostname === 'localhost' || hostname === '127.0.0.1') && port === '3000') {
      return `${protocol}//127.0.0.1:8000${API_BASE}${path}`
    }
    return `${LOCAL_DEV_API_ORIGIN}${API_BASE}${path}`
  }

  return `${API_BASE}${path}`
}

function isLocalModelSpec(model?: string | null): boolean {
  if (!model) {
    return false
  }
  return model.startsWith('/') || /^[A-Za-z]:[\\/]/.test(model)
}

export function resolveChatStreamTarget(options: SendMessageOptions = {}): RequestTarget {
  return isLocalModelSpec(options.model) ? 'direct' : 'proxy'
}

async function requestJson<T>(path: string, init?: RequestInit, target: RequestTarget = 'proxy'): Promise<T> {
  let response: Response
  const isFormData = typeof FormData !== 'undefined' && init?.body instanceof FormData

  try {
    response = await fetch(resolveApiUrl(path, target), {
      ...init,
      headers: {
        Accept: 'application/json',
        ...(init?.body && !isFormData ? { 'Content-Type': 'application/json' } : {}),
        ...init?.headers,
      },
      cache: 'no-store',
    })
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : 'Network request failed'
    throw new ApiError(0, message, error)
  }

  const payload = await parseResponseBody(response)

  if (!response.ok) {
    throw new ApiError(
      response.status,
      getApiMessage(payload, response.statusText || 'Request failed'),
      payload
    )
  }

  return payload as T
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system'
  content: string
}

export interface SendMessageOptions {
  sessionId?: string
  projectId?: string | null
  model?: string
  temperature?: number
  maxTokens?: number
  systemPrompt?: string
  topP?: number
  minP?: number
  topK?: number
  frequencyPenalty?: number
  presencePenalty?: number
  repeatPenalty?: number
  signal?: AbortSignal
}

export type TurnEventPhase =
  | 'thinking'
  | 'tool_call_request'
  | 'tool_call_result'
  | 'error'
  | 'final_answer'

export interface LegacyChatEvent {
  type:
    | 'thinking'
    | 'tool_call_request'
    | 'tool_call_result'
    | 'error'
    | 'final_answer'
}

export interface TurnEventPayload extends Record<string, unknown> {
  content?: unknown
  final_answer?: unknown
  text?: unknown
  message?: unknown
  answer?: unknown
  call_id?: unknown
  toolCallId?: unknown
  tool_name?: unknown
  toolName?: unknown
  arguments?: unknown
  toolArgs?: unknown
  result?: unknown
  toolResult?: unknown
  error?: unknown
  toolError?: unknown
  code?: unknown
  errorCode?: unknown
  trajectory_id?: unknown
  trajectoryId?: unknown
  metadata?: unknown
  input_tokens?: unknown
  output_tokens?: unknown
  generation_time_ms?: unknown
  finish_reason?: unknown
}

export interface SessionMessageEvent {
  type: 'message'
  role?: string
  content?: string
  timestamp?: string
  turn_id?: string | number
  turnId?: string | number
}

export interface SessionTurnEvent {
  type: 'turn_event'
  phase?: TurnEventPhase | string
  payload?: TurnEventPayload
  content?: string
  timestamp?: string
  turn_id?: string | number
  turnId?: string | number
}

export interface UnknownSessionEvent {
  type: string
  role?: string
  content?: string
  timestamp?: string
}

export type BackendChatEvent = LegacyChatEvent | SessionTurnEvent

export interface TextChunkChatEvent {
  type: 'text_chunk'
  content?: string
  turn_id?: string | number
  turnId?: string | number
  timestamp?: string
}

export interface BackendChatResponse {
  type: 'chat_response'
  session_id: string
  turn_id?: string | null
  final_answer: string
  trajectory_id: string | null
  events: BackendChatEvent[]
}

export interface PostChatPayload {
  message: string
  session_id?: string
  sessionId?: string
  project_id?: string | null
  projectId?: string | null
  model?: string
  system_prompt?: string
  temperature?: number
  max_tokens?: number
  top_p?: number
  min_p?: number
  top_k?: number
  frequency_penalty?: number
  presence_penalty?: number
  repeat_penalty?: number
}

export interface SendMessageResult {
  id: string
  content: string
  model: string
  sessionId: string
  createdAt: string
  trajectoryId: string | null
  events: BackendChatEvent[]
}

export type StreamChatEvent =
  | BackendChatEvent
  | TextChunkChatEvent
  | BackendChatResponse
  | {
      type: 'done'
      session_id?: string
      sessionId?: string
      model?: string
      trajectory_id?: string | null
      trajectoryId?: string | null
    }

export interface StreamChatOptions extends SendMessageOptions {
  onSessionId?: (sessionId: string) => void
}

export interface StreamChatChunk {
  event: Message | null
  sessionId?: string
  trajectoryId?: string | null
  model?: string | null
  done?: boolean
}

interface NormalizedMessageEvent {
  kind: 'message'
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp?: string
  turnKey: string | null
}

interface NormalizedTurnEvent {
  kind: 'turn_event'
  phase: MessageEventType
  content: string
  timestamp?: string
  turnKey: string | null
  toolCallId?: string
  toolName?: string
  toolArgs?: Record<string, unknown>
  toolResult?: unknown
  toolMeta?: Record<string, unknown>
  toolError?: string
  errorCode?: string
  trajectoryId?: string | null
  inputTokens?: number
  outputTokens?: number
  generationTimeMs?: number
  finishReason?: string
}

interface NormalizedTextChunkEvent {
  kind: 'text_chunk'
  phase: 'text_chunk'
  content: string
  timestamp?: string
  turnKey: string | null
}

type NormalizedTimelineEvent =
  | NormalizedMessageEvent
  | NormalizedTurnEvent
  | NormalizedTextChunkEvent

function getNonEmptyString(value: unknown): string | null {
  return typeof value === 'string' && value.trim().length > 0 ? value : null
}

function getTurnKey(value: Record<string, unknown>): string | null {
  const direct =
    getString(value.turn_id) ??
    getString(value.turnId) ??
    getNumber(value.turn_id)?.toString() ??
    getNumber(value.turnId)?.toString()

  if (direct) {
    return direct
  }

  const payload = isRecord(value.payload) ? value.payload : null
  if (!payload) {
    return null
  }

  return (
    getString(payload.turn_id) ??
    getString(payload.turnId) ??
    getNumber(payload.turn_id)?.toString() ??
    getNumber(payload.turnId)?.toString() ??
    getString(payload.trajectory_id) ??
    null
  )
}

function getPayloadContent(payload: Record<string, unknown>): string {
  return (
    getNonEmptyString(payload.content) ??
    getNonEmptyString(payload.final_answer) ??
    getNonEmptyString(payload.text) ??
    getNonEmptyString(payload.message) ??
    getNonEmptyString(payload.answer) ??
    ''
  )
}

function getPayloadRecord(
  payload: Record<string, unknown>,
  snakeCaseKey: string,
  camelCaseKey: string
): Record<string, unknown> | undefined {
  const snakeCaseValue = payload[snakeCaseKey]
  if (isRecord(snakeCaseValue)) {
    return snakeCaseValue
  }

  const camelCaseValue = payload[camelCaseKey]
  return isRecord(camelCaseValue) ? camelCaseValue : undefined
}

function getPayloadNumber(
  payload: Record<string, unknown>,
  snakeCaseKey: string,
  camelCaseKey: string
): number | undefined {
  const snakeCaseValue = payload[snakeCaseKey]
  const camelCaseValue = payload[camelCaseKey]
  return getNumber(snakeCaseValue) ?? getNumber(camelCaseValue) ?? undefined
}

function normalizeTimelineEvent(event: Record<string, unknown>): NormalizedTimelineEvent | null {
  const type = getString(event.type)
  const timestamp = getString(event.timestamp) ?? undefined
  const turnKey = getTurnKey(event)

  if (type === 'message') {
    const role = getString(event.role)
    const content = getNonEmptyString(event.content)

    if (!role || !content) {
      return null
    }

    if (role !== 'user' && role !== 'assistant' && role !== 'system') {
      return null
    }

    return {
      kind: 'message',
      role,
      content,
      timestamp,
      turnKey,
    }
  }

  if (type === 'turn_event') {
    const phase = getString(event.phase)
    const payload = isRecord(event.payload) ? event.payload : {}
    const finishReason =
      getNonEmptyString(payload.finish_reason) ??
      getNonEmptyString(payload.finishReason) ??
      undefined

    if (
      phase !== 'thinking' &&
      phase !== 'tool_call_request' &&
      phase !== 'tool_call_result' &&
      phase !== 'error' &&
      phase !== 'final_answer'
    ) {
      return null
    }

    return {
      kind: 'turn_event',
      phase,
      content: getPayloadContent(payload),
      timestamp,
      turnKey,
      toolCallId: getString(payload.call_id) ?? getString(payload.toolCallId) ?? undefined,
      toolName: getString(payload.tool_name) ?? getString(payload.toolName) ?? undefined,
      toolArgs: getPayloadRecord(payload, 'arguments', 'toolArgs'),
      toolResult: payload.result ?? payload.toolResult,
      toolMeta: getPayloadRecord(payload, 'metadata', 'metadata'),
      toolError:
        getNonEmptyString(payload.error) ??
        getNonEmptyString(payload.toolError) ??
        (phase === 'error' ? getNonEmptyString(payload.message) ?? undefined : undefined),
      errorCode: getString(payload.code) ?? getString(payload.errorCode) ?? undefined,
      trajectoryId: getString(payload.trajectory_id) ?? getString(payload.trajectoryId),
      inputTokens: getPayloadNumber(payload, 'input_tokens', 'inputTokens'),
      outputTokens: getPayloadNumber(payload, 'output_tokens', 'outputTokens'),
      generationTimeMs: getPayloadNumber(payload, 'generation_time_ms', 'generationTimeMs'),
      finishReason,
    }
  }

  if (
    type === 'text_chunk' &&
    getNonEmptyString(event.content)
  ) {
    return {
      kind: 'text_chunk',
      phase: 'text_chunk',
      content: getNonEmptyString(event.content) ?? '',
      timestamp,
      turnKey,
    }
  }

  if (
    type === 'thinking' ||
    type === 'tool_call_request' ||
    type === 'tool_call_result' ||
    type === 'error' ||
    type === 'final_answer'
  ) {
    return {
      kind: 'turn_event',
      phase: type,
      content:
        getNonEmptyString(event.content) ??
        (type === 'error'
          ? getNonEmptyString(event.error) ?? getNonEmptyString(event.message) ?? ''
          : ''),
      timestamp,
      turnKey,
      toolCallId: getString(event.call_id) ?? undefined,
      toolName: getString(event.tool_name) ?? undefined,
      toolArgs: isRecord(event.arguments) ? event.arguments : undefined,
      toolResult: event.result,
      toolMeta: isRecord(event.metadata) ? event.metadata : undefined,
      toolError: getNonEmptyString(event.error) ?? undefined,
      errorCode: getString(event.code) ?? undefined,
      trajectoryId: getString(event.trajectory_id),
      inputTokens: getNumber(event.input_tokens) ?? undefined,
      outputTokens: getNumber(event.output_tokens) ?? undefined,
      generationTimeMs: getNumber(event.generation_time_ms) ?? undefined,
      finishReason: getNonEmptyString(event.finish_reason) ?? undefined,
    }
  }

  return null
}

function toMessageTimestamp(timestamp?: string): Date {
  if (timestamp) {
    const parsed = Date.parse(timestamp)
    if (!Number.isNaN(parsed)) {
      return new Date(parsed)
    }
  }
  return new Date()
}

function buildMessageId(prefix: string, index: number, turnKey: string | null, timestamp?: string): string {
  return [prefix, turnKey ?? 'na', timestamp ?? 'now', index.toString()].join('-')
}

function buildReasoningStep(
  event: NormalizedTurnEvent,
  index: number
): ReasoningStep | null {
  const id = buildMessageId(`reasoning-${event.phase}`, index, event.turnKey, event.timestamp)
  const timestamp = toMessageTimestamp(event.timestamp)

  switch (event.phase) {
    case 'thinking':
      return {
        id,
        type: 'thinking',
        content: event.content,
        timestamp,
      }
    case 'tool_call_request':
      return {
        id,
        type: 'tool_call',
        content: event.content,
        timestamp,
        toolCallId: event.toolCallId,
        toolName: event.toolName,
        toolArgs: event.toolArgs,
        status: 'running',
      }
    case 'tool_call_result':
      return {
        id,
        type: 'tool_result',
        content: event.toolError ?? event.content,
        timestamp,
        toolCallId: event.toolCallId,
        toolName: event.toolName,
        toolResult: event.toolResult,
        toolMeta: event.toolMeta,
        toolError: event.toolError,
        status: event.toolError ? 'error' : 'success',
      }
    case 'error':
      return {
        id,
        type: 'error',
        content: event.toolError ?? event.content ?? 'Unknown error.',
        timestamp,
        errorCode: event.errorCode,
        status: 'error',
      }
    default:
      return null
  }
}

function buildTokenStats(event: NormalizedTurnEvent): TokenStats | undefined {
  if (
    typeof event.inputTokens !== 'number' ||
    typeof event.outputTokens !== 'number' ||
    typeof event.generationTimeMs !== 'number'
  ) {
    return undefined
  }

  return {
    inputTokens: event.inputTokens,
    outputTokens: event.outputTokens,
    generationTimeMs: event.generationTimeMs,
    finishReason: event.finishReason,
  }
}

function normalizeReasoningText(value: string): string {
  return value
    .replace(/\r\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim()
}

function appendInlineReasoning(
  message: Message,
  options: {
    content: string
    index: number
    turnKey: string | null
    timestamp?: string
  },
): Message {
  const { content, index, turnKey, timestamp } = options
  const extracted = extractInlineReasoning(content)
  if (!extracted.reasoning) {
    return {
      ...message,
      content: extracted.content,
    }
  }

  return {
    ...message,
    content: extracted.content,
    reasoningSteps: [
      ...(message.reasoningSteps ?? []),
      buildInlineReasoningStep(extracted.reasoning, {
        index,
        turnKey,
        timestamp,
      }),
    ],
  }
}

function appendStreamingInlineReasoning(
  message: Message,
  options: {
    contentChunk: string
    index: number
    turnKey: string | null
    timestamp?: string
  },
): Message {
  const { contentChunk, index, turnKey, timestamp } = options
  const result = appendInlineReasoningChunk(
    message.reasoningBuffer ?? createInlineReasoningBuffer(),
    contentChunk,
  )

  let reasoningSteps = message.reasoningSteps ?? []
  if (result.reasoningDelta) {
    const stepId =
      message.inlineReasoningStepId ??
      ['reasoning-inline-live', turnKey ?? 'na', timestamp ?? 'now', String(index)].join('-')
    const nextContent = normalizeReasoningText(result.buffer.reasoning)
    const existingIndex = reasoningSteps.findIndex((step) => step.id === stepId)
    const nextStep: ReasoningStep = {
      id: stepId,
      type: 'thinking',
      content: nextContent,
      timestamp: timestamp ? toMessageTimestamp(timestamp) : new Date(),
      status: 'running',
    }

    reasoningSteps =
      existingIndex === -1
        ? [...reasoningSteps, nextStep]
        : reasoningSteps.map((step, stepIndex) => (stepIndex === existingIndex ? nextStep : step))

    return {
      ...message,
      content: `${message.content}${result.contentDelta}`,
      reasoningSteps,
      reasoningBuffer: result.buffer,
      inlineReasoningStepId: stepId,
    }
  }

  return {
    ...message,
    content: `${message.content}${result.contentDelta}`,
    reasoningSteps,
    reasoningBuffer: result.buffer,
  }
}

function finalizeStreamingInlineReasoning(
  message: Message,
  options: {
    content: string
    index: number
    turnKey: string | null
    timestamp?: string
  },
): Message {
  const { content, index, turnKey, timestamp } = options
  if (message.reasoningBuffer) {
    const finalized = finalizeInlineReasoningBuffer(message.reasoningBuffer)
    const extracted = extractInlineReasoning(content)
    const visibleContent = extracted.content || finalized.content || message.content

    let reasoningSteps = message.reasoningSteps ?? []
    const completedReasoning = finalized.reasoning
    if (completedReasoning) {
      const stepId = message.inlineReasoningStepId
      if (stepId) {
        const existingIndex = reasoningSteps.findIndex((step) => step.id === stepId)
        if (existingIndex !== -1) {
          reasoningSteps = reasoningSteps.map((step, stepIndex) => (
            stepIndex === existingIndex
              ? {
                  ...step,
                  content: completedReasoning,
                  status: 'success',
                }
              : step
          ))
        } else {
          reasoningSteps = [
            ...reasoningSteps,
            buildInlineReasoningStep(completedReasoning, {
              index,
              turnKey,
              timestamp,
            }),
          ]
        }
      } else if (!reasoningSteps.some((step) => step.type === 'thinking' && step.content === completedReasoning)) {
        reasoningSteps = [
          ...reasoningSteps,
          buildInlineReasoningStep(completedReasoning, {
            index,
            turnKey,
            timestamp,
          }),
        ]
      }
    }

    return {
      ...message,
      content: visibleContent,
      reasoningSteps,
      reasoningBuffer: undefined,
      inlineReasoningStepId: undefined,
    }
  }

  return appendInlineReasoning(message, {
    content,
    index,
    turnKey,
    timestamp,
  })
}

export function buildMessagesFromTimelineEvents(events: ReadonlyArray<unknown>): Message[] {
  const normalized = events
    .filter(isRecord)
    .map(normalizeTimelineEvent)
    .filter((event): event is NormalizedTimelineEvent => event !== null)
  const toolResultsByCallId = new Map<string, NormalizedTurnEvent>()

  for (const event of normalized) {
    if (
      event.kind === 'turn_event' &&
      event.phase === 'tool_call_result' &&
      event.toolCallId
    ) {
      toolResultsByCallId.set(event.toolCallId, event)
    }
  }

  const messages: Message[] = []
  const assistantIndexByTurn = new Map<string, number>()

  normalized.forEach((event, index) => {
    if (event.kind === 'message') {
      if (event.role === 'assistant') {
        const existingIndex = event.turnKey ? assistantIndexByTurn.get(event.turnKey) : undefined
        if (existingIndex !== undefined) {
          messages[existingIndex] = appendInlineReasoning({
            ...messages[existingIndex],
            content: event.content,
            timestamp: toMessageTimestamp(event.timestamp),
          }, {
            content: event.content,
            index,
            turnKey: event.turnKey,
            timestamp: event.timestamp,
          })
          return
        }
      }

      const message: Message = appendInlineReasoning({
        id: buildMessageId(`timeline-${event.role}`, index, event.turnKey, event.timestamp),
        type: event.role,
        content: event.content,
        timestamp: toMessageTimestamp(event.timestamp),
        turnKey: event.turnKey,
        turnId: event.turnKey,
      }, {
        content: event.content,
        index,
        turnKey: event.turnKey,
        timestamp: event.timestamp,
      })
      messages.push(message)
      if (event.role === 'assistant' && event.turnKey) {
        assistantIndexByTurn.set(event.turnKey, messages.length - 1)
      }
      return
    }

    if (event.kind === 'text_chunk') {
      const turnKey = event.turnKey ?? `stream-${index}`
      const existingIndex = assistantIndexByTurn.get(turnKey)
      if (existingIndex !== undefined) {
        messages[existingIndex] = appendStreamingInlineReasoning(
          {
            ...messages[existingIndex],
            isStreaming: true,
            eventType: 'text_chunk',
          },
          {
            contentChunk: event.content,
            index,
            turnKey,
            timestamp: event.timestamp,
          }
        )
      } else {
        messages.push({
          id: buildMessageId('timeline-assistant-stream', index, turnKey, event.timestamp),
          type: 'assistant',
          content: '',
          timestamp: toMessageTimestamp(event.timestamp),
          eventType: 'text_chunk',
          turnKey,
          turnId: turnKey,
          isStreaming: true,
          reasoningSteps: [],
          reasoningBuffer: createInlineReasoningBuffer(),
        })
        messages[messages.length - 1] = appendStreamingInlineReasoning(messages[messages.length - 1], {
          contentChunk: event.content,
          index,
          turnKey,
          timestamp: event.timestamp,
        })
        assistantIndexByTurn.set(turnKey, messages.length - 1)
      }
      return
    }

    const turnKey = event.turnKey ?? `turn-${index}`
    const existingIndex = assistantIndexByTurn.get(turnKey)

    if (event.phase === 'final_answer') {
      const tokenStats = buildTokenStats(event)
      if (existingIndex !== undefined) {
        messages[existingIndex] = finalizeStreamingInlineReasoning({
          ...messages[existingIndex],
          timestamp: toMessageTimestamp(event.timestamp),
          eventType: 'final_answer',
          isStreaming: false,
          tokenStats: tokenStats ?? messages[existingIndex].tokenStats,
        }, {
          content: event.content,
          index,
          turnKey,
          timestamp: event.timestamp,
        })
      } else {
        messages.push(appendInlineReasoning({
          id: buildMessageId('timeline-final', index, turnKey, event.timestamp),
          type: 'assistant',
          content: event.content,
          timestamp: toMessageTimestamp(event.timestamp),
          eventType: 'final_answer',
          turnKey,
          turnId: turnKey,
          isStreaming: false,
          reasoningSteps: [],
          tokenStats: tokenStats,
        }, {
          content: event.content,
          index,
          turnKey,
          timestamp: event.timestamp,
        }))
        assistantIndexByTurn.set(turnKey, messages.length - 1)
      }
      return
    }

    const step = buildReasoningStep(event, index)
    if (!step) {
      return
    }

    if (existingIndex !== undefined) {
      const target = messages[existingIndex]
      messages[existingIndex] = {
        ...target,
        reasoningSteps: [...(target.reasoningSteps ?? []), step],
      }
      return
    }

    messages.push({
      id: buildMessageId('timeline-assistant-turn', index, turnKey, event.timestamp),
      type: step.type === 'error' ? 'error' : 'assistant',
      content: step.type === 'error' ? step.content : '',
      timestamp: toMessageTimestamp(event.timestamp),
      eventType: step.type === 'error' ? 'error' : undefined,
      turnKey,
      turnId: turnKey,
      reasoningSteps: [step],
      errorCode: step.errorCode,
      isStreaming: step.type !== 'error',
      reasoningBuffer: step.type === 'error' ? undefined : createInlineReasoningBuffer(),
      inlineReasoningStepId: undefined,
    })
    assistantIndexByTurn.set(turnKey, messages.length - 1)
  })

  return messages
}

export function buildMessagesFromChatEvents(events: BackendChatEvent[]): Message[] {
  return buildMessagesFromTimelineEvents(events)
}

export function buildMessagesFromSessionEvents(events: SessionEvent[]): Message[] {
  return buildMessagesFromTimelineEvents(events)
}

export async function sendMessage(
  text: string,
  options: SendMessageOptions = {}
): Promise<SendMessageResult> {
  const payload = await requestJson<BackendChatResponse>('/chat', {
    method: 'POST',
    body: JSON.stringify({
      message: text,
      session_id: options.sessionId,
      project_id: options.projectId,
      model: options.model,
      system_prompt: options.systemPrompt,
      temperature: options.temperature,
      max_tokens: options.maxTokens,
      top_p: options.topP,
      min_p: options.minP,
      top_k: options.topK,
      frequency_penalty: options.frequencyPenalty,
      presence_penalty: options.presencePenalty,
      repeat_penalty: options.repeatPenalty,
    }),
  })

  return {
    id: payload.trajectory_id ?? `chat-${payload.session_id}-${Date.now()}`,
    content: payload.final_answer,
    model: options.model ?? 'unknown',
    sessionId: payload.session_id,
    createdAt: new Date().toISOString(),
    trajectoryId: payload.trajectory_id,
    events: payload.events,
  }
}

export async function postChat(payload: PostChatPayload): Promise<BackendChatResponse> {
  return requestJson<BackendChatResponse>('/chat', {
    method: 'POST',
    body: JSON.stringify({
      message: payload.message,
      session_id: payload.session_id ?? payload.sessionId,
      project_id: payload.project_id ?? payload.projectId,
      model: payload.model,
      system_prompt: payload.system_prompt,
      temperature: payload.temperature,
      max_tokens: payload.max_tokens,
      top_p: payload.top_p,
      min_p: payload.min_p,
      top_k: payload.top_k,
      frequency_penalty: payload.frequency_penalty,
      presence_penalty: payload.presence_penalty,
      repeat_penalty: payload.repeat_penalty,
    }),
  })
}

function normalizeStreamEvent(value: unknown): StreamChatEvent | null {
  if (!isRecord(value)) {
    return null
  }
  const type = getString(value.type)
  if (!type) {
    return null
  }
  return value as StreamChatEvent
}

function isBackendChatResponse(value: StreamChatEvent): value is BackendChatResponse {
  return value.type === 'chat_response'
}

function resolveStreamSessionId(event: StreamChatEvent): string | undefined {
  if (isBackendChatResponse(event)) {
    return event.session_id
  }

  if ('session_id' in event && typeof event.session_id === 'string' && event.session_id.length > 0) {
    return event.session_id
  }

  if ('sessionId' in event && typeof event.sessionId === 'string' && event.sessionId.length > 0) {
    return event.sessionId
  }

  return undefined
}

function resolveStreamModel(event: StreamChatEvent): string | null {
  if ('model' in event && typeof event.model === 'string' && event.model.length > 0) {
    return event.model
  }
  return null
}

function resolveStreamTrajectoryId(event: StreamChatEvent): string | null {
  if (isBackendChatResponse(event)) {
    return event.trajectory_id
  }

  if (
    'trajectory_id' in event &&
    (typeof event.trajectory_id === 'string' || event.trajectory_id === null)
  ) {
    return event.trajectory_id
  }

  if (
    'trajectoryId' in event &&
    (typeof event.trajectoryId === 'string' || event.trajectoryId === null)
  ) {
    return event.trajectoryId
  }

  return null
}

function toStreamMessages(event: StreamChatEvent): Message[] {
  if (isBackendChatResponse(event)) {
    return buildMessagesFromChatEvents(event.events)
  }

  if (
    event.type === 'thinking' ||
    event.type === 'tool_call_request' ||
    event.type === 'tool_call_result' ||
    event.type === 'error' ||
    event.type === 'final_answer' ||
    event.type === 'text_chunk'
  ) {
    return buildMessagesFromTimelineEvents([event])
  }

  return []
}

async function* readNdjsonStream(
  stream: ReadableStream<Uint8Array>
): AsyncGenerator<StreamChatEvent, void, unknown> {
  const reader = stream.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { value, done } = await reader.read()
    if (done) {
      break
    }
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''
    for (const rawLine of lines) {
      const line = rawLine.trim()
      if (!line) {
        continue
      }
      const parsed = normalizeStreamEvent(JSON.parse(line))
      if (parsed) {
        yield parsed
      }
    }
  }

  const finalLine = buffer.trim()
  if (finalLine) {
    const parsed = normalizeStreamEvent(JSON.parse(finalLine))
    if (parsed) {
      yield parsed
    }
  }
}

async function* readSseStream(
  stream: ReadableStream<Uint8Array>
): AsyncGenerator<StreamChatEvent, void, unknown> {
  const reader = stream.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { value, done } = await reader.read()
    if (done) {
      break
    }
    buffer += decoder.decode(value, { stream: true })
    const frames = buffer.split('\n\n')
    buffer = frames.pop() ?? ''

    for (const frame of frames) {
      const data = frame
        .split('\n')
        .filter((line) => line.startsWith('data:'))
        .map((line) => line.slice(5).trim())
        .join('\n')
        .trim()

      if (!data || data === '[DONE]') {
        continue
      }

      const parsed = normalizeStreamEvent(JSON.parse(data))
      if (parsed) {
        yield parsed
      }
    }
  }

  const finalFrame = buffer.trim()
  if (!finalFrame) {
    return
  }

  const data = finalFrame
    .split('\n')
    .filter((line) => line.startsWith('data:'))
    .map((line) => line.slice(5).trim())
    .join('\n')
    .trim()

  if (!data || data === '[DONE]') {
    return
  }

  const parsed = normalizeStreamEvent(JSON.parse(data))
  if (parsed) {
    yield parsed
  }
}

async function requestStreamResponse(
  text: string,
  options: SendMessageOptions = {}
): Promise<Response> {
  let response: Response
  const target = resolveChatStreamTarget(options)

  try {
    response = await fetch(resolveApiUrl('/chat/stream', target), {
      method: 'POST',
      headers: {
        Accept: 'text/event-stream, application/x-ndjson, application/json',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message: text,
        session_id: options.sessionId,
        project_id: options.projectId,
        model: options.model,
        system_prompt: options.systemPrompt,
        temperature: options.temperature,
        max_tokens: options.maxTokens,
        top_p: options.topP,
        min_p: options.minP,
        top_k: options.topK,
        frequency_penalty: options.frequencyPenalty,
        presence_penalty: options.presencePenalty,
        repeat_penalty: options.repeatPenalty,
      }),
      signal: options.signal,
      cache: 'no-store',
    })
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : 'Network request failed'
    throw new ApiError(0, message, error)
  }

  if (!response.ok) {
    const payload = await parseResponseBody(response)
    throw new ApiError(
      response.status,
      getApiMessage(payload, response.statusText || 'Request failed'),
      payload
    )
  }

  if (!response.body) {
    throw new ApiError(0, 'Streaming response body is unavailable.')
  }

  return response
}

export async function* streamChat(
  text: string,
  options: SendMessageOptions = {}
): AsyncGenerator<StreamChatEvent, void, unknown> {
  const response = await requestStreamResponse(text, options)
  const body = response.body
  if (!body) {
    throw new ApiError(0, 'Streaming response body is unavailable.')
  }

  const contentType = response.headers.get('content-type') ?? ''
  if (contentType.includes('text/event-stream')) {
    yield* readSseStream(body)
    return
  }

  if (contentType.includes('application/x-ndjson')) {
    yield* readNdjsonStream(body)
    return
  }

  const payload = normalizeStreamEvent(await parseResponseBody(response))
  if (payload) {
    yield payload
  }
}

export async function* streamChatMessages(
  text: string,
  options: StreamChatOptions = {}
): AsyncGenerator<StreamChatChunk, void, unknown> {
  const response = await requestStreamResponse(text, options)
  const body = response.body
  if (!body) {
    throw new ApiError(0, 'Streaming response body is unavailable.')
  }
  const responseSessionId = response.headers.get('X-Session-ID') ?? undefined
  if (responseSessionId) {
    options.onSessionId?.(responseSessionId)
  }

  const contentType = response.headers.get('content-type') ?? ''
  const stream =
    contentType.includes('text/event-stream')
      ? readSseStream(body)
      : contentType.includes('application/x-ndjson')
        ? readNdjsonStream(body)
        : (async function* singlePayload() {
            const payload = normalizeStreamEvent(await parseResponseBody(response))
            if (payload) {
              yield payload
            }
          })()

  let seenSessionId = responseSessionId
  let emittedDone = false

  for await (const rawEvent of stream) {
    const nextSessionId = resolveStreamSessionId(rawEvent) ?? seenSessionId
    if (nextSessionId && nextSessionId !== seenSessionId) {
      seenSessionId = nextSessionId
      options.onSessionId?.(nextSessionId)
    }

    const model = resolveStreamModel(rawEvent)
    const trajectoryId = resolveStreamTrajectoryId(rawEvent)

    if (rawEvent.type === 'done') {
      emittedDone = true
      yield {
        event: null,
        sessionId: nextSessionId,
        model,
        trajectoryId,
        done: true,
      }
      continue
    }

    const messages = toStreamMessages(rawEvent)
    if (messages.length === 0) {
      continue
    }

    for (const message of messages) {
      yield {
        event: message,
        sessionId: nextSessionId,
        model,
        trajectoryId,
      }
    }
  }

  if (!emittedDone) {
    yield {
      event: null,
      sessionId: seenSessionId,
      done: true,
    }
  }
}

interface BackendSkill {
  skill_id: string
  name: string
  description: string
  trigger_keywords: string[]
  preconditions: string
  steps: string[]
  tools_used: string[]
  source_trajectory_id: string
  times_used: number
  success_rate: number
  created_at: number | string
  updated_at: number | string
  version: number
}

export interface Skill {
  id: string
  name: string
  description: string
  category: 'general' | 'task-specific'
  tags: string[]
  useCount: number
  successRate: number
  version: string
  enabled: boolean
  createdAt: string
  updatedAt: string
  triggerKeywords: string[]
  preconditions: string[]
  steps: string[]
  toolsUsed: string[]
  sourceTrajectoryId: string | null
}

function normalizeSkill(skill: BackendSkill): Skill {
  const triggerKeywords = getStringArray(skill.trigger_keywords)
  const toolsUsed = getStringArray(skill.tools_used)
  const preconditions = skill.preconditions
    .split('\n')
    .map((item) => item.trim())
    .filter((item) => item.length > 0)

  return {
    id: skill.skill_id,
    name: skill.name,
    description: skill.description,
    category: skill.source_trajectory_id ? 'task-specific' : 'general',
    tags: [...triggerKeywords, ...toolsUsed].slice(0, 8),
    useCount: skill.times_used,
    successRate: Math.max(0, Math.min(100, Math.round(skill.success_rate * 100))),
    version: `v${skill.version}`,
    enabled: true,
    createdAt: toIsoString(skill.created_at),
    updatedAt: toIsoString(skill.updated_at),
    triggerKeywords,
    preconditions,
    steps: getStringArray(skill.steps),
    toolsUsed,
    sourceTrajectoryId: skill.source_trajectory_id || null,
  }
}

export async function fetchSkills(
  params: { q?: string; limit?: number } = {}
): Promise<Skill[]> {
  const searchParams = new URLSearchParams()

  if (params.q) {
    searchParams.set('q', params.q)
  }
  if (typeof params.limit === 'number') {
    searchParams.set('limit', String(params.limit))
  }

  const query = searchParams.toString()
  const payload = await requestJson<BackendSkill[]>(`/skills${query ? `?${query}` : ''}`)
  return payload.map(normalizeSkill)
}

export async function exportSkills(): Promise<Skill[]> {
  const payload = await requestJson<BackendSkill[]>('/skills/export')
  return payload.map(normalizeSkill)
}

export async function deleteSkill(skillId: string): Promise<void> {
  await requestJson<{ deleted: boolean }>(`/skills/${encodeURIComponent(skillId)}`, {
    method: 'DELETE',
  })
}

interface BackendSessionListItem {
  session_id: string
  title?: string
  event_count: number
  updated_at: string
  project_id?: string | null
}

interface BackendSessionListResponse {
  type: 'sessions'
  items: BackendSessionListItem[]
}

export interface SessionSummary {
  id: string
  title: string
  createdAt: string
  updatedAt: string
  eventCount: number
  projectId: string | null
}

export type SessionEvent = SessionMessageEvent | SessionTurnEvent | UnknownSessionEvent

interface BackendSessionResponse {
  type: 'session'
  session_id: string
  title?: string
  project_id?: string | null
  events: Record<string, unknown>[]
}

export interface SessionDetail extends SessionSummary {
  events: SessionEvent[]
}

function normalizeSessionSummary(item: BackendSessionListItem): SessionSummary {
  return {
    id: item.session_id,
    title: getString(item.title) ?? item.session_id,
    createdAt: item.updated_at,
    updatedAt: item.updated_at,
    eventCount: item.event_count,
    projectId: getString(item.project_id) ?? null,
  }
}

function normalizeSessionEvent(event: Record<string, unknown>): SessionEvent {
  const type = getString(event.type) ?? 'unknown'

  if (type === 'message') {
    return {
      ...event,
      type,
      role: getString(event.role) ?? undefined,
      content: getString(event.content) ?? undefined,
      timestamp: getString(event.timestamp) ?? undefined,
      turn_id:
        getString(event.turn_id) ??
        getNumber(event.turn_id) ??
        getString(event.turnId) ??
        getNumber(event.turnId) ??
        undefined,
      turnId:
        getString(event.turnId) ??
        getNumber(event.turnId) ??
        getString(event.turn_id) ??
        getNumber(event.turn_id) ??
        undefined,
    }
  }

  if (type === 'turn_event') {
    return {
      ...event,
      type,
      phase: getString(event.phase) ?? undefined,
      payload: isRecord(event.payload) ? event.payload : undefined,
      content: getString(event.content) ?? undefined,
      timestamp: getString(event.timestamp) ?? undefined,
      turn_id:
        getString(event.turn_id) ??
        getNumber(event.turn_id) ??
        getString(event.turnId) ??
        getNumber(event.turnId) ??
        undefined,
      turnId:
        getString(event.turnId) ??
        getNumber(event.turnId) ??
        getString(event.turn_id) ??
        getNumber(event.turn_id) ??
        undefined,
    }
  }

  return {
    ...event,
    type,
    role: getString(event.role) ?? undefined,
    content: getString(event.content) ?? undefined,
    timestamp: getString(event.timestamp) ?? undefined,
  }
}

export async function fetchSessions(): Promise<SessionSummary[]> {
  const payload = await requestJson<BackendSessionListResponse>('/sessions')
  return payload.items.map(normalizeSessionSummary)
}

export async function fetchSession(sessionId: string): Promise<SessionDetail> {
  const payload = await requestJson<BackendSessionResponse>(
    `/sessions/${encodeURIComponent(sessionId)}`
  )

  const events = getRecordArray(payload.events).map(normalizeSessionEvent)
  const updatedAt =
    [...events]
      .reverse()
      .map((event) => getString(event.timestamp))
      .find((timestamp): timestamp is string => Boolean(timestamp)) ?? new Date().toISOString()

  return {
    id: payload.session_id,
    title: getString(payload.title) ?? payload.session_id,
    createdAt: updatedAt,
    updatedAt,
    eventCount: events.length,
    projectId: getString(payload.project_id) ?? null,
    events,
  }
}

interface BackendCreateSessionResponse {
  type: 'session'
  session_id: string
}

interface ForkSessionInput {
  sessionId: string
  turnId: string
  projectId?: string | null
}

export async function createSession(
  sessionId?: string,
  projectId?: string | null
): Promise<SessionSummary> {
  const payload = await requestJson<BackendCreateSessionResponse>('/sessions', {
    method: 'POST',
    body: JSON.stringify({
      ...(sessionId ? { session_id: sessionId } : {}),
      ...(projectId !== undefined ? { project_id: projectId } : {}),
    }),
  })

  const now = new Date().toISOString()
  return {
    id: payload.session_id,
    title: payload.session_id,
    createdAt: now,
    updatedAt: now,
    eventCount: 1,
    projectId: projectId ?? null,
  }
}

export async function forkSession(input: ForkSessionInput): Promise<SessionSummary> {
  const payload = await requestJson<BackendCreateSessionResponse>('/sessions', {
    method: 'POST',
    body: JSON.stringify({
      fork_from_session_id: input.sessionId,
      fork_until_turn_id: input.turnId,
      ...(input.projectId !== undefined ? { project_id: input.projectId } : {}),
    }),
  })

  const now = new Date().toISOString()
  return {
    id: payload.session_id,
    title: payload.session_id,
    createdAt: now,
    updatedAt: now,
    eventCount: 1,
    projectId: input.projectId ?? null,
  }
}

interface BackendUpdateSessionResponse {
  type: 'session'
  session_id: string
  title: string
}

export async function renameSession(sessionId: string, title: string): Promise<SessionSummary> {
  const payload = await requestJson<BackendUpdateSessionResponse>(
    `/sessions/${encodeURIComponent(sessionId)}`,
    {
      method: 'PATCH',
      body: JSON.stringify({ title }),
    }
  )

  const now = new Date().toISOString()
  return {
    id: payload.session_id,
    title: payload.title,
    createdAt: now,
    updatedAt: now,
    eventCount: 0,
    projectId: null,
  }
}

export async function deleteSession(sessionId: string): Promise<void> {
  await requestJson<{ deleted: boolean }>(`/sessions/${encodeURIComponent(sessionId)}`, {
    method: 'DELETE',
  })
}

interface BackendUpdateSessionProjectResponse {
  type: 'session'
  session_id: string
  project_id?: string | null
}

export async function updateSessionProject(
  sessionId: string,
  projectId: string | null
): Promise<{ sessionId: string; projectId: string | null }> {
  const payload = await requestJson<BackendUpdateSessionProjectResponse>(
    `/sessions/${encodeURIComponent(sessionId)}/project`,
    {
      method: 'PATCH',
      body: JSON.stringify({ project_id: projectId }),
    }
  )

  return {
    sessionId: payload.session_id,
    projectId: getString(payload.project_id) ?? null,
  }
}

interface BackendProject {
  id: string
  name: string
  workspace_dir: string
  created_at: string
  updated_at: string
}

interface BackendProjectListResponse {
  type: 'projects'
  items: BackendProject[]
}

export interface ProjectSummary {
  id: string
  name: string
  workspaceDir: string
  createdAt: string
  updatedAt: string
}

export interface ProjectDetail extends ProjectSummary {}

function normalizeProject(project: BackendProject): ProjectDetail {
  return {
    id: project.id,
    name: project.name,
    workspaceDir: project.workspace_dir,
    createdAt: project.created_at,
    updatedAt: project.updated_at,
  }
}

export async function fetchProjects(): Promise<ProjectSummary[]> {
  const payload = await requestJson<BackendProjectListResponse>('/projects')
  return payload.items.map(normalizeProject)
}

export async function fetchProject(projectId: string): Promise<ProjectDetail> {
  const payload = await requestJson<BackendProject>(`/projects/${encodeURIComponent(projectId)}`)
  return normalizeProject(payload)
}

export async function createProject(input: {
  name: string
  workspaceDir: string
}): Promise<ProjectDetail> {
  const payload = await requestJson<BackendProject>('/projects', {
    method: 'POST',
    body: JSON.stringify({
      name: input.name,
      workspace_dir: input.workspaceDir,
    }),
  })
  return normalizeProject(payload)
}

export async function updateProject(
  projectId: string,
  input: {
    name?: string
    workspaceDir?: string
  }
): Promise<ProjectDetail> {
  const payload = await requestJson<BackendProject>(`/projects/${encodeURIComponent(projectId)}`, {
    method: 'PATCH',
    body: JSON.stringify({
      ...(input.name !== undefined ? { name: input.name } : {}),
      ...(input.workspaceDir !== undefined ? { workspace_dir: input.workspaceDir } : {}),
    }),
  })
  return normalizeProject(payload)
}

export async function deleteProject(projectId: string): Promise<void> {
  await requestJson<{ deleted: boolean }>(`/projects/${encodeURIComponent(projectId)}`, {
    method: 'DELETE',
  })
}

interface BackendModelInfo extends Record<string, ApiValue | undefined> {
  id?: string
  name?: string
  label?: string
  model?: string
  model_spec?: string
  provider?: string
  base_url?: string | null
  backend_type?: string
  context_length?: number
  supports_tool_calling?: boolean
  metadata?: Record<string, ApiValue>
}

interface BackendModelsStatus {
  type: 'models_status'
  configured_model: string
  supported_model_spec_formats: Array<{
    type: string
    pattern: string
    description: string
  }>
  active_model: BackendModelInfo | null
  available_models?: BackendModelInfo[]
  configured_remote_provider?: string | null
}

export interface ModelInfo {
  id: string
  name: string
  label: string
  provider: string | null
  modelSpec: string | null
  baseUrl: string | null
  backendType: string
  contextLength: number | null
  supportsToolCalling: boolean | null
  metadata: Record<string, ApiValue>
}

export type ModelProvider = 'ollama' | 'openai_compat' | 'gemini' | 'anthropic' | 'vllm' | 'local'

export interface ConfigureModelInput {
  provider: ModelProvider
  model: string
  baseUrl?: string
  apiKey?: string
  persist?: boolean
}

interface BackendConfigureModelResponse {
  type: 'model_configure'
  provider: ModelProvider
  active_model: BackendModelInfo
  available_models?: BackendModelInfo[]
  api_key_configured: boolean
  persisted?: boolean
  config_path?: string | null
}

export interface ConfigureModelResult {
  type: 'model_configure'
  provider: ModelProvider
  activeModel: ModelInfo
  availableModels: ModelInfo[]
  apiKeyConfigured: boolean
  persisted: boolean
  configPath: string | null
}

interface BackendModelEntryUpdateResponse {
  type: 'model_entry_update'
  updated_model: BackendModelInfo
  available_models?: BackendModelInfo[]
  configured_model: string
  api_key_configured: boolean
  persisted?: boolean
  config_path?: string | null
}

interface BackendModelEntryDeleteResponse {
  type: 'model_entry_delete'
  deleted_model_id: string
  available_models?: BackendModelInfo[]
  configured_model: string
  persisted?: boolean
  config_path?: string | null
}

export interface UpdateModelEntryInput {
  modelId: string
  provider: ModelProvider
  model: string
  modelSpec: string
  baseUrl?: string | null
  apiKey?: string | null
  persist?: boolean
}

export interface UpdateModelEntryResult {
  type: 'model_entry_update'
  updatedModel: ModelInfo
  availableModels: ModelInfo[]
  configuredModel: string
  apiKeyConfigured: boolean
  persisted: boolean
  configPath: string | null
}

export interface DeleteModelEntryResult {
  type: 'model_entry_delete'
  deletedModelId: string
  availableModels: ModelInfo[]
  configuredModel: string
  persisted: boolean
  configPath: string | null
}

export interface ModelsStatus {
  type: 'models_status'
  configuredModel: string
  supportedModelSpecFormats: Array<{
    type: string
    pattern: string
    description: string
  }>
  activeModel: ModelInfo | null
  availableModels: ModelInfo[]
  configuredRemoteProvider: string | null
}

function normalizeModelInfo(model: BackendModelInfo | Record<string, unknown> | null): ModelInfo | null {
  if (!model) {
    return null
  }

  const metadata: Record<string, ApiValue> = {}
  if (isRecord(model.metadata)) {
    for (const [key, value] of Object.entries(model.metadata)) {
      if (value !== undefined) {
        metadata[key] = value as ApiValue
      }
    }
  }

  return {
    id: getString(model.id) ?? getString(model.model_spec) ?? getString(model.name) ?? getString(model.model) ?? '',
    name: getString(model.name) ?? getString(model.model) ?? getString(model.model_spec) ?? '',
    label: getString(model.label) ?? getString(model.name) ?? getString(model.model) ?? '',
    provider: getString(model.provider),
    modelSpec: getString(model.model_spec),
    baseUrl: getString(model.base_url),
    backendType: getString(model.backend_type) ?? '',
    contextLength: getNumber(model.context_length) ?? null,
    supportsToolCalling: getBoolean(model.supports_tool_calling) ?? null,
    metadata,
  }
}

export async function fetchModelsStatus(): Promise<ModelsStatus> {
  const payload = await requestJson<BackendModelsStatus>('/models')
  return {
    type: payload.type,
    configuredModel: payload.configured_model,
    supportedModelSpecFormats: payload.supported_model_spec_formats,
    activeModel: normalizeModelInfo(payload.active_model),
    availableModels: getRecordArray(payload.available_models).map((model) =>
      normalizeModelInfo(model)
    ).filter((model): model is ModelInfo => model !== null),
    configuredRemoteProvider: payload.configured_remote_provider ?? null,
  }
}

export async function fetchModels(): Promise<ModelInfo[]> {
  const status = await fetchModelsStatus()
  if (status.availableModels.length > 0) {
    return status.availableModels
  }
  return status.activeModel ? [status.activeModel] : []
}

interface BackendModelSwitchResponse {
  type: 'model_switch'
  active_model: BackendModelInfo
}

export async function switchModel(model: string): Promise<ModelInfo> {
  const payload = await requestJson<BackendModelSwitchResponse>('/models/switch', {
    method: 'POST',
    body: JSON.stringify({ model }),
  })

  const activeModel = normalizeModelInfo(payload.active_model)
  if (!activeModel) {
    throw new ApiError(500, 'Backend did not return an active model')
  }
  return activeModel
}

export async function configureModel(input: ConfigureModelInput): Promise<ConfigureModelResult> {
  const payload = await requestJson<BackendConfigureModelResponse>('/models/configure', {
    method: 'POST',
    body: JSON.stringify({
      provider: input.provider,
      model: input.model,
      base_url: input.baseUrl,
      api_key: input.apiKey,
      persist: input.persist,
    }),
  })

  const activeModel = normalizeModelInfo(payload.active_model)
  if (!activeModel) {
    throw new ApiError(500, 'Backend did not return an active model')
  }

  return {
    type: payload.type,
    provider: payload.provider,
    activeModel,
    availableModels: getRecordArray(payload.available_models).map((model) =>
      normalizeModelInfo(model)
    ).filter((model): model is ModelInfo => model !== null),
    apiKeyConfigured: payload.api_key_configured,
    persisted: Boolean(payload.persisted),
    configPath: payload.config_path ?? null,
  }
}

export async function updateModelEntry(input: UpdateModelEntryInput): Promise<UpdateModelEntryResult> {
  const payload = await requestJson<BackendModelEntryUpdateResponse>(`/models/configured/${encodeURIComponent(input.modelId)}`, {
    method: 'PATCH',
    body: JSON.stringify({
      provider: input.provider,
      model: input.model,
      model_spec: input.modelSpec,
      base_url: input.baseUrl ?? null,
      api_key: input.apiKey ?? null,
      persist: input.persist,
    }),
  })

  const updatedModel = normalizeModelInfo(payload.updated_model)
  if (!updatedModel) {
    throw new ApiError(500, 'Backend did not return an updated model entry')
  }

  return {
    type: payload.type,
    updatedModel,
    availableModels: getRecordArray(payload.available_models).map((model) =>
      normalizeModelInfo(model)
    ).filter((model): model is ModelInfo => model !== null),
    configuredModel: payload.configured_model,
    apiKeyConfigured: payload.api_key_configured,
    persisted: Boolean(payload.persisted),
    configPath: payload.config_path ?? null,
  }
}

export async function deleteModelEntry(modelId: string, persist = true): Promise<DeleteModelEntryResult> {
  const payload = await requestJson<BackendModelEntryDeleteResponse>(`/models/configured/${encodeURIComponent(modelId)}`, {
    method: 'DELETE',
    body: JSON.stringify({ persist }),
  })

  return {
    type: payload.type,
    deletedModelId: payload.deleted_model_id,
    availableModels: getRecordArray(payload.available_models).map((model) =>
      normalizeModelInfo(model)
    ).filter((model): model is ModelInfo => model !== null),
    configuredModel: payload.configured_model,
    persisted: Boolean(payload.persisted),
    configPath: payload.config_path ?? null,
  }
}

interface BackendOllamaModelsResponse {
  type: 'ollama_models'
  base_url: string
  models: string[]
}

export interface OllamaModelsResult {
  type: 'ollama_models'
  baseUrl: string
  models: string[]
}

export async function fetchOllamaModels(baseUrl: string): Promise<OllamaModelsResult> {
  const query = new URLSearchParams({ base_url: baseUrl })
  const payload = await requestJson<BackendOllamaModelsResponse>(`/models/ollama?${query}`)
  return {
    type: payload.type,
    baseUrl: payload.base_url,
    models: getStringArray(payload.models),
  }
}

interface BackendLocalModelsResponse {
  type: 'local_models'
  root: string
  models?: BackendModelInfo[]
  warnings?: string[]
}

export interface LocalModelsResult {
  type: 'local_models'
  root: string
  models: ModelInfo[]
  warnings: string[]
}

export async function fetchLocalModels(root: string): Promise<LocalModelsResult> {
  const query = new URLSearchParams({ root })
  const payload = await requestJson<BackendLocalModelsResponse>(`/models/local?${query}`)
  return {
    type: payload.type,
    root: payload.root,
    models: getRecordArray(payload.models).map((model) =>
      normalizeModelInfo(model)
    ).filter((model): model is ModelInfo => model !== null),
    warnings: getStringArray(payload.warnings),
  }
}

export type LocalModelCapabilityStatus = 'supported' | 'unsupported' | 'conditional' | 'unknown'
export type LocalModelCapabilityFormat = 'gguf' | string

interface BackendQuantizationOption {
  id?: string
  name?: string
  bits?: string
  description?: string
}

interface BackendLocalModelCapabilityFormat {
  format_id?: string
  format_name?: string
  supported?: boolean
  priority?: string
  reason?: string
  warnings?: string[]
  quantization_options?: BackendQuantizationOption[]
  suggested_default_quantization?: string | null
}

interface BackendHardwareSummary {
  provider?: string
  cuda_available?: boolean
  gpu_count?: number
  gpu_vendor?: string | null
  primary_gpu_name?: string | null
  total_vram_gb?: number | null
  recommended_runtime_backend?: string | null
  recommended_runtime_label?: string | null
  warnings?: string[]
}

interface BackendLocalModelCapabilitiesResponse {
  type?: string
  model_spec?: string
  model_dir?: string
  model_family?: string | null
  formats?: BackendLocalModelCapabilityFormat[]
  warnings?: string[]
  hardware?: BackendHardwareSummary | null
}

export interface LocalModelQuantizationOption {
  id: string
  name: string
  bits: string | null
  description: string | null
}

export interface LocalModelCapabilityFormatInfo {
  formatId: LocalModelCapabilityFormat
  formatName: string
  status: LocalModelCapabilityStatus
  supported: boolean
  priority: string | null
  reason: string | null
  warnings: string[]
  quantizationOptions: LocalModelQuantizationOption[]
  suggestedDefaultQuantization: string | null
}

export interface LocalModelHardwareSummary {
  provider: string | null
  cudaAvailable: boolean | null
  gpuCount: number | null
  gpuVendor: string | null
  primaryGpuName: string | null
  totalVramGb: number | null
  recommendedRuntimeBackend: string | null
  recommendedRuntimeLabel: string | null
  warnings: string[]
}

export interface LocalModelCapabilitiesResult {
  type: string
  modelSpec: string
  modelDir: string
  modelFamily: string | null
  formats: LocalModelCapabilityFormatInfo[]
  warnings: string[]
  hardware: LocalModelHardwareSummary | null
}

export type LocalModelRuntimeReadiness =
  | 'ready'
  | 'degraded'
  | 'missing'
  | 'not_installed'
  | 'manual_setup_required'
  | 'incompatible'
  | 'installing'
  | 'unknown'

export type LocalModelRuntimeSource =
  | 'managed'
  | 'existing_path'
  | 'env'
  | 'auto'
  | 'unknown'
  | string

export type LocalModelRuntimeAction =
  | 'register_existing_path'
  | 'prepare_managed_runtime'
  | 'ready_for_conversion'
  | string

interface BackendLocalModelRuntimeStatusResponse {
  type?: string
  runtime?: string
  readiness?: string
  installed?: boolean
  source?: string
  root_dir?: string | null
  install_dir?: string | null
  python_executable?: string
  version?: string | null
  platform?: string | null
  binary_asset?: string | null
  convert_script?: string | null
  quantize_binary?: string | null
  missing_components?: string[]
  warnings?: string[]
  actions?: string[]
  hardware?: BackendHardwareSummary | null
}

export interface LocalModelRuntimeStatus {
  type: string
  runtime: string
  readiness: LocalModelRuntimeReadiness
  installed: boolean
  source: LocalModelRuntimeSource
  rootDir: string | null
  installDir: string | null
  pythonExecutable: string | null
  version: string | null
  platform: string | null
  binaryAsset: string | null
  convertScript: string | null
  quantizeBinary: string | null
  missingComponents: string[]
  warnings: string[]
  actions: LocalModelRuntimeAction[]
  hardware: LocalModelHardwareSummary | null
}

export type LocalModelRuntimeInstallAction = 'prepare_managed' | 'register_existing_path'

interface BackendLocalModelRuntimeInstallResponse {
  type?: string
  runtime?: string
  action?: string
  state?: string
  source?: string
  install_dir?: string | null
  root_dir?: string | null
  version?: string | null
  platform?: string | null
  binary_asset?: string | null
  persisted?: boolean
  config_path?: string | null
  runtime_status?: BackendLocalModelRuntimeStatusResponse | null
  warnings?: string[]
  message?: string
}

export interface LocalModelRuntimeInstallInput {
  action: LocalModelRuntimeInstallAction
  existingPath?: string
  persist?: boolean
}

export interface LocalModelRuntimeInstallResult {
  type: string
  runtime: string
  action: string
  state: LocalModelRuntimeReadiness
  source: LocalModelRuntimeSource
  installDir: string | null
  rootDir: string | null
  version: string | null
  platform: string | null
  binaryAsset: string | null
  persisted: boolean
  configPath: string | null
  runtimeStatus: LocalModelRuntimeStatus | null
  warnings: string[]
  message: string | null
}

interface BackendLocalActiveModelRuntimeStatusResponse {
  type?: string
  has_active_local_model?: boolean
  model_spec?: string | null
  backend_type?: string | null
  loaded?: boolean
  idle_unloaded?: boolean
  can_unload?: boolean
}

interface BackendLocalActiveModelRuntimeUnloadResponse {
  type?: string
  unloaded?: boolean
  active_runtime?: BackendLocalActiveModelRuntimeStatusResponse | null
}

export interface LocalActiveModelRuntimeStatus {
  type: string
  hasActiveLocalModel: boolean
  modelSpec: string | null
  backendType: string | null
  loaded: boolean
  idleUnloaded: boolean
  canUnload: boolean
}

export interface LocalActiveModelRuntimeUnloadResult {
  type: string
  unloaded: boolean
  activeRuntime: LocalActiveModelRuntimeStatus
}

function normalizeRuntimeReadiness(value: unknown, installed: boolean): LocalModelRuntimeReadiness {
  const readiness = getString(value)?.trim().toLowerCase()
  if (readiness === 'ready' || readiness === 'degraded' || readiness === 'missing') {
    return readiness
  }
  if (readiness === 'not_installed' || readiness === 'manual_setup_required' || readiness === 'incompatible') {
    return readiness
  }
  if (!installed) {
    return 'not_installed'
  }
  return 'unknown'
}

function normalizeLocalModelRuntimeStatus(
  payload: BackendLocalModelRuntimeStatusResponse | null | undefined
): LocalModelRuntimeStatus | null {
  if (!payload || !isRecord(payload)) {
    return null
  }

  const installed = getBoolean(payload.installed) ?? false
  return {
    type: getString(payload.type) ?? 'local_model_runtime_status',
    runtime: getString(payload.runtime) ?? 'llama.cpp',
    readiness: normalizeRuntimeReadiness(payload.readiness, installed),
    installed,
    source: getString(payload.source) ?? 'unknown',
    rootDir: getString(payload.root_dir),
    installDir: getString(payload.install_dir),
    pythonExecutable: getString(payload.python_executable),
    version: getString(payload.version),
    platform: getString(payload.platform),
    binaryAsset: getString(payload.binary_asset),
    convertScript: getString(payload.convert_script),
    quantizeBinary: getString(payload.quantize_binary),
    missingComponents: getStringArray(payload.missing_components),
    warnings: getStringArray(payload.warnings),
    actions: getStringArray(payload.actions),
    hardware: normalizeHardwareSummary(payload.hardware),
  }
}

function normalizeLocalActiveModelRuntimeStatus(
  payload: BackendLocalActiveModelRuntimeStatusResponse | null | undefined
): LocalActiveModelRuntimeStatus | null {
  if (!payload || !isRecord(payload)) {
    return null
  }

  return {
    type: getString(payload.type) ?? 'local_active_model_runtime_status',
    hasActiveLocalModel: getBoolean(payload.has_active_local_model) ?? false,
    modelSpec: getString(payload.model_spec),
    backendType: getString(payload.backend_type),
    loaded: getBoolean(payload.loaded) ?? false,
    idleUnloaded: getBoolean(payload.idle_unloaded) ?? false,
    canUnload: getBoolean(payload.can_unload) ?? false,
  }
}

function normalizeCapabilityStatus(supported: boolean | null): LocalModelCapabilityStatus {
  if (supported === true) {
    return 'supported'
  }
  if (supported === false) {
    return 'unsupported'
  }
  return 'unknown'
}

function normalizeQuantizationOptions(value: unknown): LocalModelQuantizationOption[] {
  if (!Array.isArray(value)) {
    return []
  }

  const options: LocalModelQuantizationOption[] = []
  for (const entry of value) {
    if (typeof entry === 'string') {
      const text = entry.trim()
      if (text.length === 0) {
        continue
      }
      options.push({
        id: text,
        name: text,
        bits: null,
        description: null,
      })
      continue
    }

    if (!isRecord(entry)) {
      continue
    }

    const id = (
      getString(entry.id) ??
      getString(entry.name) ??
      getString(entry.value)
    )?.trim()
    const name = (
      getString(entry.name) ??
      getString(entry.id) ??
      getString(entry.value)
    )?.trim()

    if (!id || !name) {
      continue
    }

    options.push({
      id,
      name,
      bits: getString(entry.bits),
      description: getString(entry.description),
    })
  }

  return options
}

function normalizeCapabilityFormat(
  entry: Record<string, unknown>
): LocalModelCapabilityFormatInfo {
  const formatId = (getString(entry.format_id) ?? '').trim().toLowerCase() || 'unknown'
  const formatName = (getString(entry.format_name) ?? formatId).trim()
  const supported = getBoolean(entry.supported)
  const warnings = getStringArray(entry.warnings)
  const reason = getString(entry.reason)

  let status = normalizeCapabilityStatus(supported)
  if (status === 'unsupported' && warnings.length > 0) {
    status = 'conditional'
  }

  return {
    formatId,
    formatName,
    status,
    supported: supported ?? false,
    priority: getString(entry.priority),
    reason,
    warnings,
    quantizationOptions: normalizeQuantizationOptions(entry.quantization_options),
    suggestedDefaultQuantization: getString(entry.suggested_default_quantization),
  }
}

function normalizeHardwareSummary(value: unknown): LocalModelHardwareSummary | null {
  if (!isRecord(value)) {
    return null
  }

  return {
    provider: getString(value.provider),
    cudaAvailable: getBoolean(value.cuda_available),
    gpuCount: getNumber(value.gpu_count),
    gpuVendor: getString(value.gpu_vendor),
    primaryGpuName: getString(value.primary_gpu_name),
    totalVramGb: getNumber(value.total_vram_gb),
    recommendedRuntimeBackend: getString(value.recommended_runtime_backend),
    recommendedRuntimeLabel: getString(value.recommended_runtime_label),
    warnings: getStringArray(value.warnings),
  }
}

export async function fetchLocalModelCapabilities(model: string): Promise<LocalModelCapabilitiesResult> {
  const normalizedModel = model.trim()
  const query = new URLSearchParams({ model_spec: normalizedModel })
  const payload = await requestJson<BackendLocalModelCapabilitiesResponse>(`/models/local/capabilities?${query}`)

  return {
    type: payload.type ?? 'local_model_quantization_capabilities',
    modelSpec: getString(payload.model_spec) ?? normalizedModel,
    modelDir: getString(payload.model_dir) ?? normalizedModel,
    modelFamily: getString(payload.model_family),
    formats: getRecordArray(payload.formats).map(normalizeCapabilityFormat),
    warnings: getStringArray(payload.warnings),
    hardware: normalizeHardwareSummary(payload.hardware),
  }
}

export async function fetchLocalModelRuntimeStatus(): Promise<LocalModelRuntimeStatus> {
  const payload = await requestJson<BackendLocalModelRuntimeStatusResponse>('/models/local/runtime')
  const normalized = normalizeLocalModelRuntimeStatus(payload)
  if (!normalized) {
    throw new ApiError(500, 'Invalid local model runtime status response', payload)
  }
  return normalized
}

export async function installLocalModelRuntime(
  input: LocalModelRuntimeInstallInput
): Promise<LocalModelRuntimeInstallResult> {
  const payload = await requestJson<BackendLocalModelRuntimeInstallResponse>('/models/local/runtime/install', {
    method: 'POST',
    body: JSON.stringify({
      action: input.action,
      existing_path: input.existingPath?.trim() || undefined,
      persist: input.persist,
    }),
  }, 'direct')

  return {
    type: payload.type ?? 'local_model_runtime_install',
    runtime: getString(payload.runtime) ?? 'llama.cpp',
    action: getString(payload.action) ?? input.action,
    state: normalizeRuntimeReadiness(payload.state, true),
    source: getString(payload.source) ?? 'unknown',
    installDir: getString(payload.install_dir),
    rootDir: getString(payload.root_dir),
    version: getString(payload.version),
    platform: getString(payload.platform),
    binaryAsset: getString(payload.binary_asset),
    persisted: Boolean(payload.persisted),
    configPath: getString(payload.config_path),
    runtimeStatus: normalizeLocalModelRuntimeStatus(payload.runtime_status),
    warnings: getStringArray(payload.warnings),
    message: getString(payload.message),
  }
}

export async function fetchActiveLocalModelRuntimeStatus(): Promise<LocalActiveModelRuntimeStatus> {
  const payload = await requestJson<BackendLocalActiveModelRuntimeStatusResponse>('/models/local/active-runtime')
  const normalized = normalizeLocalActiveModelRuntimeStatus(payload)
  if (!normalized) {
    throw new ApiError(500, 'Invalid active local model runtime status response', payload)
  }
  return normalized
}

export async function unloadActiveLocalModelRuntime(): Promise<LocalActiveModelRuntimeUnloadResult> {
  const payload = await requestJson<BackendLocalActiveModelRuntimeUnloadResponse>(
    '/models/local/active-runtime/unload',
    { method: 'POST' }
  )
  const activeRuntime = normalizeLocalActiveModelRuntimeStatus(payload.active_runtime)
  if (!activeRuntime) {
    throw new ApiError(500, 'Invalid active local model runtime unload response', payload)
  }
  return {
    type: getString(payload.type) ?? 'local_active_model_runtime_unload',
    unloaded: getBoolean(payload.unloaded) ?? false,
    activeRuntime,
  }
}

interface BackendLocalModelConvertResponse {
  type?: string
  provider?: string
  source_model_dir?: string
  model_dir?: string
  model_spec?: string
  target_format?: string
  quantization?: string | null
  output_model_path?: string
  output_model_spec?: string
  output_path?: string
  gguf_path?: string
  persisted?: boolean
  config_path?: string | null
  warnings?: string[]
  available_models?: BackendModelInfo[]
  active_model?: BackendModelInfo | null
}

export interface LocalModelConvertInput {
  sourceModelDir: string
  targetFormat: 'gguf'
  quantization: string
  persist?: boolean
}

export interface LocalModelConvertResult {
  type: string
  provider: ModelProvider | null
  sourceModelDir: string
  targetFormat: string
  quantization: string | null
  outputPath: string | null
  persisted: boolean
  configPath: string | null
  warnings: string[]
  availableModels: ModelInfo[]
  activeModel: ModelInfo | null
}

function normalizeProvider(value: unknown): ModelProvider | null {
  const provider = getString(value)
  if (
    provider === 'ollama' ||
    provider === 'openai_compat' ||
    provider === 'gemini' ||
    provider === 'anthropic' ||
    provider === 'local'
  ) {
    return provider
  }
  return null
}

export async function convertLocalModel(input: LocalModelConvertInput): Promise<LocalModelConvertResult> {
  const normalizedSource = input.sourceModelDir.trim()
  const normalizedQuantization = input.quantization.trim()
  const payload = await requestJson<BackendLocalModelConvertResponse>('/models/local/convert', {
    method: 'POST',
    body: JSON.stringify({
      model_spec: normalizedSource,
      source_model_dir: normalizedSource,
      target_format: input.targetFormat,
      quantization: normalizedQuantization,
      persist: input.persist,
    }),
  }, 'direct')

  return {
    type: payload.type ?? 'local_model_convert',
    provider: normalizeProvider(payload.provider),
    sourceModelDir: getString(payload.source_model_dir) ?? getString(payload.model_dir) ?? normalizedSource,
    targetFormat: getString(payload.target_format) ?? input.targetFormat,
    quantization: getString(payload.quantization) ?? normalizedQuantization,
    outputPath: (
      getString(payload.output_model_path) ??
      getString(payload.output_model_spec) ??
      getString(payload.output_path) ??
      getString(payload.gguf_path)
    ),
    persisted: Boolean(payload.persisted),
    configPath: getString(payload.config_path),
    warnings: getStringArray(payload.warnings),
    availableModels: getRecordArray(payload.available_models).map((model) =>
      normalizeModelInfo(model)
    ).filter((model): model is ModelInfo => model !== null),
    activeModel: normalizeModelInfo(payload.active_model ?? null),
  }
}

interface BackendFilesystemRoot {
  name?: string
  label: string
  path: string
}

interface BackendFilesystemRootsResponse {
  roots?: BackendFilesystemRoot[]
  items?: BackendFilesystemRoot[]
}

interface BackendFilesystemListItem {
  name: string
  path: string
  is_dir: boolean
  is_file: boolean
}

interface BackendFilesystemListResponse {
  path?: string
  current_path?: string
  parent?: string | null
  parent_path?: string | null
  items: BackendFilesystemListItem[]
}

export interface FilesystemRoot {
  label: string
  path: string
}

export interface FilesystemListItem {
  name: string
  path: string
  isDir: boolean
  isFile: boolean
}

export interface FilesystemListResult {
  path: string
  parent: string | null
  items: FilesystemListItem[]
}

interface BackendFilesystemImportResponse {
  type: 'filesystem_import'
  import_root: string
  imported_path: string
  file_count: number
  total_bytes: number
}

export interface FilesystemImportInput {
  files: File[]
  relativePaths?: string[]
  targetDir?: string
  packageName?: string
}

export interface FilesystemImportResult {
  type: 'filesystem_import'
  importRoot: string
  importedPath: string
  fileCount: number
  totalBytes: number
}

export async function fetchFilesystemRoots(): Promise<FilesystemRoot[]> {
  const payload = await requestJson<BackendFilesystemRootsResponse>('/filesystem/roots')
  const roots = Array.isArray(payload.roots) ? payload.roots : payload.items

  return Array.isArray(roots)
    ? roots
        .filter((item) => typeof item?.path === 'string' && item.path.length > 0)
        .map((item) => ({
          label: getString(item.label) ?? getString(item.name) ?? item.path,
          path: item.path,
        }))
    : []
}

export async function fetchFilesystemList(path: string): Promise<FilesystemListResult> {
  const query = new URLSearchParams({ path })
  const payload = await requestJson<BackendFilesystemListResponse>(`/filesystem/list?${query.toString()}`)

  return {
    path: getString(payload.path) ?? getString(payload.current_path) ?? path,
    parent: getString(payload.parent) ?? getString(payload.parent_path) ?? null,
    items: Array.isArray(payload.items)
      ? payload.items
          .filter((item) => typeof item?.path === 'string' && item.path.length > 0)
          .map((item) => ({
            name: getString(item.name) ?? item.path,
            path: item.path,
            isDir: Boolean(item.is_dir),
            isFile: Boolean(item.is_file),
          }))
      : [],
  }
}

export async function importFilesystemFiles(
  input: FilesystemImportInput
): Promise<FilesystemImportResult> {
  const form = new FormData()
  for (const file of input.files) {
    form.append('files', file, file.name)
  }
  for (const relativePath of input.relativePaths ?? input.files.map((file) => file.name)) {
    form.append('relative_paths', relativePath)
  }
  if (input.targetDir) {
    form.append('target_dir', input.targetDir)
  }
  if (input.packageName) {
    form.append('package_name', input.packageName)
  }

  const payload = await requestJson<BackendFilesystemImportResponse>('/filesystem/import', {
    method: 'POST',
    body: form,
  })

  return {
    type: payload.type,
    importRoot: payload.import_root,
    importedPath: payload.imported_path,
    fileCount: payload.file_count,
    totalBytes: payload.total_bytes,
  }
}

interface BackendSettings {
  type: 'settings'
  model: string
  model_config?: Record<string, ApiValue>
  model_setup?: Record<string, ApiValue>
  locale_defaults?: Record<string, ApiValue>
  agent?: Record<string, ApiValue>
  voice: Record<string, ApiValue>
  memory: Record<string, ApiValue>
  learning: Record<string, ApiValue>
  tools?: Record<string, ApiValue>
  local_models?: Record<string, ApiValue>
  gguf?: Record<string, ApiValue>
  vllm?: Record<string, ApiValue>
  channels: Record<string, ApiValue>
  web: Record<string, ApiValue>
  security?: Record<string, ApiValue>
  paths?: Record<string, ApiValue>
  update?: Record<string, ApiValue>
}

export interface InferencePreset {
  name: string
  system_prompt: string
  temperature: number
  max_tokens: number
  top_p: number
  min_p: number
  top_k: number
  frequency_penalty: number
  presence_penalty: number
  repeat_penalty: number
}

export interface AgentSettings {
  system_prompt: string
  temperature: number
  max_tokens: number
  top_p: number
  min_p: number
  top_k: number
  frequency_penalty: number
  presence_penalty: number
  repeat_penalty: number
  show_token_stats: boolean
  presets: InferencePreset[]
  active_preset: string
}

export interface SecuritySettings {
  require_approval_for_file_write: boolean
  max_file_write_size_mb: number
  file_ops_scope: 'workspace' | 'any'
  file_undo_max_size_mb: number
}

export interface LocalModelSettings {
  idle_unload_enabled: boolean
  idle_unload_seconds: number | null
}

export interface GGUFSettings {
  n_ctx: number
}

export interface VLLMSettings {
  max_model_len: number | null
}

export interface ToolsSettings extends Record<string, unknown> {
  web_search_engine?: string
  web_search_fallback_engines?: string[]
  web_search_searxng_base_url?: string | null
  web_search_language?: string | null
  web_search_region?: string | null
  web_search_tavily_api_key_configured?: boolean
  web_search_serper_api_key_configured?: boolean
  web_search_jina_configured?: boolean
  web_search_brave_api_key_configured?: boolean
  web_search_jina_api_key_configured?: boolean
  web_search_exa_api_key_configured?: boolean
  web_search_searxng_configured?: boolean
  web_search_duckduckgo_html_configured?: boolean
  web_fetch_extractor?: 'trafilatura' | 'jina_reader' | 'htmlparser'
  web_fetch_jina_api_key_configured?: boolean
}

export interface VoiceLocalTtsRecommendation extends Record<string, unknown> {
  id: string
  backend: string
  label: string
  default_voice?: string | null
  default_model?: string | null
  local?: boolean
  priority?: number
  summary?: string
  notes: string[]
}

export interface VoiceExternalApiTtsPreset extends Record<string, unknown> {
  id: string
  backend: string
  label: string
  compatibility?: string
  model?: string | null
  voice?: string | null
  summary?: string
  requires_base_url?: boolean
  requires_api_key?: boolean
  apply_supported?: boolean
  notes: string[]
}

export interface VoiceSettings extends Record<string, unknown> {
  enabled?: boolean
  stt_backend?: string
  stt_model?: string
  stt_language?: string
  stt_device?: string
  stt_model_cache_dir?: string
  stt_model_path?: string
  stt_openai_base_url?: string | null
  stt_openai_api_key_configured?: boolean
  stt_openai_timeout?: number
  tts_backend?: string
  tts_model?: string | null
  tts_voice?: string
  tts_language?: string | null
  tts_speed?: number
  tts_use_gpu?: boolean
  tts_kokoro_lang_code?: string
  tts_openai_base_url?: string | null
  tts_openai_api_key_configured?: boolean
  tts_openai_timeout?: number
  tts_openai_response_format?: 'pcm' | 'wav'
  reply_model_mode?: string
  reply_model_id?: string | null
  session_mode?: string
  recommended_local_tts_backends?: VoiceLocalTtsRecommendation[]
  external_api_tts_presets?: VoiceExternalApiTtsPreset[]
}

export interface Settings {
  type: 'settings'
  model: string
  model_config?: Record<string, ApiValue>
  model_setup?: Record<string, ApiValue>
  locale_defaults?: Record<string, ApiValue>
  agent?: AgentSettings
  voice: VoiceSettings
  memory: Record<string, ApiValue>
  learning: Record<string, ApiValue>
  tools?: ToolsSettings
  local_models?: LocalModelSettings
  gguf?: GGUFSettings
  vllm?: VLLMSettings
  channels: Record<string, ApiValue>
  web: Record<string, ApiValue>
  security?: SecuritySettings
  paths?: Record<string, ApiValue>
  update?: Record<string, ApiValue>
}

export interface ChannelRuntimeStatus extends Record<string, unknown> {
  enabled?: boolean
  registered?: boolean
  running?: boolean
  token_configured?: boolean
  tokenConfigured?: boolean
  allowed_channel_ids?: Array<string | number>
  allowedChannelIds?: Array<string | number>
  allowed_chat_ids?: Array<string | number>
  allowedChatIds?: Array<string | number>
  allowed_user_ids?: Array<string | number>
  allowedUserIds?: Array<string | number>
}

export interface DiscordChannelStatus extends ChannelRuntimeStatus {
  text_enabled?: boolean
  textEnabled?: boolean
  voice_enabled?: boolean
  voiceEnabled?: boolean
  bot_token_configured?: boolean
  botTokenConfigured?: boolean
  allowed_guild_ids?: Array<string | number>
  allowedGuildIds?: Array<string | number>
  allowed_voice_channel_ids?: Array<string | number>
  allowedVoiceChannelIds?: Array<string | number>
  message_mode?: string
  messageMode?: string
  auto_join_policy?: string
  autoJoinPolicy?: string
  voice_ingress?: Record<string, unknown>
  voiceIngress?: Record<string, unknown>
  voice_runtime?: Record<string, unknown>
  voiceRuntime?: Record<string, unknown>
}

export interface DiscordVoiceRoomSummary {
  guildId: string | null
  channelId: string | null
  sessionId: string | null
  participantCount: number | null
  joinedAt: string | null
  playbackState: string | null
  speakingState: string | null
  error: string | null
}

export interface DiscordChannelSummary {
  enabled: boolean | null
  registered: boolean | null
  running: boolean | null
  textEnabled: boolean | null
  voiceEnabled: boolean | null
  botTokenConfigured: boolean | null
  allowedChannelIds: string[]
  allowedGuildIds: string[]
  allowedUserIds: string[]
  allowedVoiceChannelIds: string[]
  messageMode: string | null
  autoJoinPolicy: string | null
  activeVoiceRoomCount: number | null
  reconnectCount: number | null
  voiceIngressEnabled: boolean | null
  voiceIngressAvailable: boolean | null
  voiceIngressGuildIds: string[]
  voiceIngressError: string | null
  voiceRuntimePhase: string | null
  voiceRuntimeError: string | null
  playbackState: string | null
  speakingState: string | null
  activeRooms: DiscordVoiceRoomSummary[]
}

export interface VoiceRuntimeStatus {
  type: string
  phase: string | null
  enabled: boolean | null
  loaded: boolean | null
  ready: boolean
  error: string | null
  configured: Record<string, unknown>
  sessionDiagnostics: Record<string, unknown>
  raw: Record<string, unknown>
}

export interface VoiceCatalogVoice {
  id: string
  name: string
  backend: string
  locale: string | null
  source: string | null
  path: string | null
  isBuiltin: boolean
  metadata: Record<string, unknown>
}

export interface VoiceCatalog {
  type: string
  voices: VoiceCatalogVoice[]
}

interface BackendChannelsStatus {
  type?: unknown
  phase?: unknown
  supported_channels?: unknown
  channels?: unknown
}

interface BackendChannelsControlResponse {
  type?: unknown
  action?: unknown
  scope?: unknown
  channel?: unknown
  running?: unknown
  running_channels?: unknown
}

interface BackendVoiceRuntimeStatus {
  type?: unknown
  phase?: unknown
  enabled?: unknown
  loaded?: unknown
  ready?: unknown
  error?: unknown
  last_error?: unknown
  lastLoadError?: unknown
  last_load_error?: unknown
  configured?: unknown
  session_diagnostics?: unknown
  sessionDiagnostics?: unknown
}

interface BackendVoiceCatalog {
  type?: unknown
  voices?: unknown
  items?: unknown
}

export interface ChannelsStatus {
  type: string
  phase: string | null
  supportedChannels: string[]
  channels: Record<string, ChannelRuntimeStatus>
}

export interface ChannelsControlResult {
  type: string
  action: string | null
  scope: string | null
  channel: string | null
  running: boolean | null
  runningChannels: string[]
}

function getRecordMap(value: unknown): Record<string, ChannelRuntimeStatus> {
  if (!isRecord(value)) {
    return {}
  }

  const entries = Object.entries(value)
  const map: Record<string, ChannelRuntimeStatus> = {}

  for (const [key, item] of entries) {
    if (isRecord(item)) {
      map[key] = item
    }
  }

  return map
}

function getBooleanField(record: Record<string, unknown>, keys: string[]): boolean | null {
  for (const key of keys) {
    const value = getBoolean(record[key])
    if (value !== null) {
      return value
    }
  }

  return null
}

function getNumberField(record: Record<string, unknown>, keys: string[]): number | null {
  for (const key of keys) {
    const value = getNumber(record[key])
    if (value !== null) {
      return value
    }
  }

  return null
}

function getStringField(record: Record<string, unknown>, keys: string[]): string | null {
  for (const key of keys) {
    const value = getNonEmptyString(record[key])
    if (value !== null) {
      return value
    }
  }

  return null
}

function getRecordArrayField(record: Record<string, unknown>, keys: string[]): Record<string, unknown>[] {
  for (const key of keys) {
    const value = record[key]
    if (!Array.isArray(value)) {
      continue
    }
    return value.filter(isRecord)
  }

  return []
}

function normalizeDiscordVoiceRoom(room: Record<string, unknown>): DiscordVoiceRoomSummary {
  return {
    guildId: getStringField(room, ['guild_id', 'guildId']),
    channelId: getStringField(room, ['channel_id', 'channelId']),
    sessionId: getStringField(room, ['session_id', 'sessionId']),
    participantCount: getNumberField(room, ['participant_count', 'participantCount']),
    joinedAt: getStringField(room, ['joined_at', 'joinedAt']),
    playbackState:
      getStringField(room, ['playback_state', 'playbackState', 'playback_status', 'playbackStatus']) ??
      (getBooleanField(room, ['is_playing', 'playing']) === true ? 'playing' : null),
    speakingState:
      getStringField(room, ['speaking_state', 'speakingState', 'speaking_status', 'speakingStatus']) ??
      (getBooleanField(room, ['is_speaking', 'speaking']) === true ? 'speaking' : null),
    error: getStringField(room, ['last_error', 'lastError', 'error']),
  }
}

function normalizeVoiceSettings(value: unknown): VoiceSettings {
  if (!isRecord(value)) {
    return {}
  }

  const settings: VoiceSettings = {
    ...value,
  }

  const replyModelMode = getString(value.reply_model_mode) ?? getString(value.replyModelMode)
  if (replyModelMode !== null) {
    settings.reply_model_mode = replyModelMode
  }

  const replyModelId =
    getString(value.reply_model_id) ??
    getString(value.replyModelId) ??
    null
  settings.reply_model_id = replyModelId

  const sessionMode = getString(value.session_mode) ?? getString(value.sessionMode)
  if (sessionMode !== null) {
    settings.session_mode = sessionMode
  }

  settings.recommended_local_tts_backends = getRecordArray(
    value.recommended_local_tts_backends ?? value.recommendedLocalTtsBackends
  )
    .map(normalizeVoiceLocalTtsRecommendation)
    .filter((recommendation): recommendation is VoiceLocalTtsRecommendation => recommendation !== null)

  settings.external_api_tts_presets = getRecordArray(
    value.external_api_tts_presets ?? value.externalApiTtsPresets
  )
    .map(normalizeVoiceExternalApiTtsPreset)
    .filter((preset): preset is VoiceExternalApiTtsPreset => preset !== null)

  return settings
}

function normalizeVoiceRecommendationNotes(value: unknown): string[] {
  return getStringArray(value)
    .map((item) => item.trim())
    .filter((item) => item.length > 0)
}

function normalizeVoiceLocalTtsRecommendation(value: Record<string, unknown>): VoiceLocalTtsRecommendation | null {
  const backend = getNonEmptyString(value.backend)
  const id = getNonEmptyString(value.id) ?? backend
  const label = getNonEmptyString(value.label) ?? backend
  if (!id || !backend || !label) {
    return null
  }

  return {
    ...value,
    id,
    backend,
    label,
    default_voice: getString(value.default_voice),
    default_model: getString(value.default_model),
    local: getBoolean(value.local) ?? undefined,
    priority: getNumber(value.priority) ?? undefined,
    summary: getString(value.summary) ?? undefined,
    notes: normalizeVoiceRecommendationNotes(value.notes),
  }
}

function normalizeVoiceExternalApiTtsPreset(value: Record<string, unknown>): VoiceExternalApiTtsPreset | null {
  const backend = getNonEmptyString(value.backend)
  const id = getNonEmptyString(value.id) ?? backend
  const label = getNonEmptyString(value.label) ?? backend
  if (!id || !backend || !label) {
    return null
  }

  return {
    ...value,
    id,
    backend,
    label,
    compatibility: getString(value.compatibility) ?? undefined,
    model: getString(value.model),
    voice: getString(value.voice),
    summary: getString(value.summary) ?? undefined,
    requires_base_url: getBoolean(value.requires_base_url) ?? undefined,
    requires_api_key: getBoolean(value.requires_api_key) ?? undefined,
    apply_supported: getBoolean(value.apply_supported) ?? undefined,
    notes: normalizeVoiceRecommendationNotes(value.notes),
  }
}

function getIdList(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return []
  }

  return value
    .map((entry) => {
      if (typeof entry === 'string') {
        return entry.trim()
      }
      if (typeof entry === 'number' && Number.isFinite(entry)) {
        return String(entry)
      }
      return ''
    })
    .filter((entry) => entry.length > 0)
}

function mergeIdLists(...sources: unknown[]): string[] {
  return Array.from(new Set(sources.flatMap((source) => getIdList(source))))
}

export function normalizeDiscordChannelStatus(
  configured: unknown,
  runtime?: unknown
): DiscordChannelSummary {
  const configuredRecord = isRecord(configured) ? configured : {}
  const runtimeRecord = isRecord(runtime) ? runtime : {}
  const voiceIngress =
    (isRecord(runtimeRecord.voice_ingress) ? runtimeRecord.voice_ingress : null) ??
    (isRecord(runtimeRecord.voiceIngress) ? runtimeRecord.voiceIngress : null)
  const voiceRuntime =
    (isRecord(runtimeRecord.voice_runtime) ? runtimeRecord.voice_runtime : null) ??
    (isRecord(runtimeRecord.voiceRuntime) ? runtimeRecord.voiceRuntime : null)
  const activeRooms = getRecordArrayField(voiceRuntime ?? {}, ['active_voice_rooms', 'activeVoiceRooms'])
    .map(normalizeDiscordVoiceRoom)

  const textEnabled =
    getBooleanField(runtimeRecord, ['text_enabled', 'textEnabled']) ??
    getBooleanField(configuredRecord, ['text_enabled', 'textEnabled'])
  const voiceEnabled =
    getBooleanField(runtimeRecord, ['voice_enabled', 'voiceEnabled']) ??
    getBooleanField(configuredRecord, ['voice_enabled', 'voiceEnabled'])

  let enabled =
    getBooleanField(runtimeRecord, ['enabled']) ??
    getBooleanField(configuredRecord, ['enabled'])

  if (enabled === null) {
    if (textEnabled === true || voiceEnabled === true) {
      enabled = true
    } else if (textEnabled === false && voiceEnabled === false) {
      enabled = false
    }
  }

  const botTokenConfigured =
    getBooleanField(runtimeRecord, [
      'bot_token_configured',
      'botTokenConfigured',
      'token_configured',
      'tokenConfigured',
    ]) ??
    getBooleanField(configuredRecord, [
      'bot_token_configured',
      'botTokenConfigured',
      'token_configured',
      'tokenConfigured',
      'has_token',
    ]) ??
    (typeof configuredRecord.token === 'string' && configuredRecord.token.trim().length > 0)

  return {
    enabled,
    registered: getBooleanField(runtimeRecord, ['registered']),
    running: getBooleanField(runtimeRecord, ['running']),
    textEnabled,
    voiceEnabled,
    botTokenConfigured,
    allowedChannelIds: mergeIdLists(
      runtimeRecord.allowed_channel_ids,
      runtimeRecord.allowedChannelIds,
      configuredRecord.allowed_channel_ids,
      configuredRecord.allowedChannelIds
    ),
    allowedGuildIds: mergeIdLists(
      runtimeRecord.allowed_guild_ids,
      runtimeRecord.allowedGuildIds,
      configuredRecord.allowed_guild_ids,
      configuredRecord.allowedGuildIds
    ),
    allowedUserIds: mergeIdLists(
      runtimeRecord.allowed_user_ids,
      runtimeRecord.allowedUserIds,
      configuredRecord.allowed_user_ids,
      configuredRecord.allowedUserIds
    ),
    allowedVoiceChannelIds: mergeIdLists(
      runtimeRecord.allowed_voice_channel_ids,
      runtimeRecord.allowedVoiceChannelIds,
      configuredRecord.allowed_voice_channel_ids,
      configuredRecord.allowedVoiceChannelIds
    ),
    messageMode:
      getStringField(runtimeRecord, ['message_mode', 'messageMode']) ??
      getStringField(configuredRecord, ['message_mode', 'messageMode']),
    autoJoinPolicy:
      getStringField(runtimeRecord, ['auto_join_policy', 'autoJoinPolicy']) ??
      getStringField(configuredRecord, ['auto_join_policy', 'autoJoinPolicy']),
    activeVoiceRoomCount:
      getNumberField(voiceRuntime ?? {}, ['active_voice_room_count', 'activeVoiceRoomCount']) ??
      activeRooms.length,
    reconnectCount: getNumberField(voiceRuntime ?? {}, ['reconnect_count', 'reconnectCount']),
    voiceIngressEnabled: getBooleanField(voiceIngress ?? {}, ['enabled']),
    voiceIngressAvailable: getBooleanField(voiceIngress ?? {}, ['extension_available', 'extensionAvailable']),
    voiceIngressGuildIds: mergeIdLists(
      voiceIngress?.active_guild_ids,
      voiceIngress?.activeGuildIds
    ),
    voiceIngressError: getStringField(voiceIngress ?? {}, ['last_error', 'lastError', 'error']),
    voiceRuntimePhase: getStringField(voiceRuntime ?? {}, ['phase']),
    voiceRuntimeError: getStringField(voiceRuntime ?? {}, ['last_error', 'lastError', 'error']),
    playbackState:
      getStringField(voiceRuntime ?? {}, ['playback_state', 'playbackState', 'playback_status', 'playbackStatus']) ??
      (getBooleanField(voiceRuntime ?? {}, ['is_playing', 'playing']) === true ? 'playing' : null),
    speakingState:
      getStringField(voiceRuntime ?? {}, ['speaking_state', 'speakingState', 'speaking_status', 'speakingStatus']) ??
      (getBooleanField(voiceRuntime ?? {}, ['is_speaking', 'speaking']) === true ? 'speaking' : null),
    activeRooms,
  }
}

function normalizeVoiceRuntimeStatus(payload: unknown): VoiceRuntimeStatus {
  const record = isRecord(payload) ? payload : {}
  const configured = isRecord(record.configured) ? record.configured : {}
  const sessionDiagnostics =
    (isRecord(record.session_diagnostics) ? record.session_diagnostics : null) ??
    (isRecord(record.sessionDiagnostics) ? record.sessionDiagnostics : null) ??
    {}
  const error =
    getString(record.error) ??
    getString(record.last_error) ??
    getString(record.last_load_error) ??
    getString(record.lastLoadError)
  const phase = getString(record.phase)
  const enabled = getBoolean(record.enabled)
  const loaded = getBoolean(record.loaded)
  const explicitReady = getBoolean(record.ready)
  const ready = explicitReady ?? (loaded === true && !error && phase !== 'error')

  return {
    type: getString(record.type) ?? 'voice_runtime_status',
    phase,
    enabled,
    loaded,
    ready: Boolean(ready),
    error,
    configured,
    sessionDiagnostics,
    raw: record,
  }
}

function normalizeVoiceCatalogVoice(payload: unknown): VoiceCatalogVoice | null {
  if (!isRecord(payload)) {
    return null
  }

  const id =
    getNonEmptyString(payload.id) ??
    getNonEmptyString(payload.voice_id) ??
    getNonEmptyString(payload.voiceId) ??
    getNonEmptyString(payload.name) ??
    getNonEmptyString(payload.label)
  if (!id) {
    return null
  }

  return {
    id,
    name:
      getNonEmptyString(payload.name) ??
      getNonEmptyString(payload.label) ??
      id,
    backend:
      getNonEmptyString(payload.backend) ??
      getNonEmptyString(payload.provider) ??
      'unknown',
    locale:
      getNonEmptyString(payload.locale) ??
      getNonEmptyString(payload.language),
    source:
      getNonEmptyString(payload.source) ??
      getNonEmptyString(payload.origin),
    path:
      getNonEmptyString(payload.path) ??
      getNonEmptyString(payload.model_path) ??
      getNonEmptyString(payload.modelPath),
    isBuiltin:
      getBoolean(payload.is_builtin) ??
      getBoolean(payload.builtin) ??
      false,
    metadata: isRecord(payload.metadata) ? payload.metadata : {},
  }
}

function normalizeVoiceCatalog(payload: unknown): VoiceCatalog {
  const record = isRecord(payload) ? payload : {}
  const voices = getRecordArray(record.voices ?? record.items)
    .map(normalizeVoiceCatalogVoice)
    .filter((voice): voice is VoiceCatalogVoice => voice !== null)

  return {
    type: getString(record.type) ?? 'voice_catalog',
    voices,
  }
}

export async function fetchSettings(): Promise<Settings> {
  const payload = await requestJson<BackendSettings>('/settings')
  return {
    type: payload.type,
    model: payload.model,
    model_config: isRecord(payload.model_config)
      ? payload.model_config as Record<string, ApiValue>
      : undefined,
    model_setup: isRecord(payload.model_setup)
      ? payload.model_setup as Record<string, ApiValue>
      : undefined,
    locale_defaults: isRecord(payload.locale_defaults)
      ? payload.locale_defaults as Record<string, ApiValue>
      : undefined,
    agent: normalizeAgentSettings(payload.agent),
    voice: normalizeVoiceSettings(payload.voice),
    memory: payload.memory,
    learning: payload.learning,
    tools: isRecord(payload.tools) ? payload.tools as ToolsSettings : undefined,
    local_models: normalizeLocalModelSettings(payload.local_models),
    gguf: normalizeGgufSettings(payload.gguf),
    vllm: normalizeVllmSettings(payload.vllm),
    channels: payload.channels,
    web: payload.web,
    security: normalizeSecuritySettings(payload.security),
    paths: isRecord(payload.paths) ? payload.paths as Record<string, ApiValue> : undefined,
    update: isRecord(payload.update) ? payload.update as Record<string, ApiValue> : undefined,
  }
}

export async function fetchChannelsStatus(): Promise<ChannelsStatus> {
  const payload = await requestJson<BackendChannelsStatus>('/channels')
  return {
    type: getString(payload.type) ?? 'channels_status',
    phase: getString(payload.phase),
    supportedChannels: getStringArray(payload.supported_channels),
    channels: getRecordMap(payload.channels),
  }
}

export async function startChannel(name: string): Promise<ChannelsControlResult> {
  const payload = await requestJson<BackendChannelsControlResponse>(`/channels/${name}/start`, {
    method: 'POST',
  })
  return {
    type: getString(payload.type) ?? 'channels_control',
    action: getString(payload.action),
    scope: getString(payload.scope),
    channel: getString(payload.channel),
    running: getBoolean(payload.running),
    runningChannels: getStringArray(payload.running_channels),
  }
}

export async function stopChannel(name: string): Promise<ChannelsControlResult> {
  const payload = await requestJson<BackendChannelsControlResponse>(`/channels/${name}/stop`, {
    method: 'POST',
  })
  return {
    type: getString(payload.type) ?? 'channels_control',
    action: getString(payload.action),
    scope: getString(payload.scope),
    channel: getString(payload.channel),
    running: getBoolean(payload.running),
    runningChannels: getStringArray(payload.running_channels),
  }
}

export async function fetchVoiceStatus(): Promise<VoiceRuntimeStatus> {
  const payload = await requestJson<BackendVoiceRuntimeStatus>('/voice/status')
  return normalizeVoiceRuntimeStatus(payload)
}

export async function fetchVoiceCatalog(): Promise<VoiceCatalog> {
  const payload = await requestJson<BackendVoiceCatalog>('/voice/voices')
  return normalizeVoiceCatalog(payload)
}

export async function uploadVoicePack(file: File): Promise<VoiceCatalog> {
  const form = new FormData()
  form.append('file', file, file.name)
  const payload = await requestJson<BackendVoiceCatalog>('/voice/voices/upload', {
    method: 'POST',
    body: form,
  })
  return normalizeVoiceCatalog(payload)
}

export async function registerVoicePackPath(path: string): Promise<VoiceCatalog> {
  const payload = await requestJson<BackendVoiceCatalog>('/voice/voices/register-path', {
    method: 'POST',
    body: JSON.stringify({ path }),
  })
  return normalizeVoiceCatalog(payload)
}

export async function deleteVoice(voiceId: string): Promise<VoiceCatalog> {
  const payload = await requestJson<BackendVoiceCatalog>(`/voice/voices/${encodeURIComponent(voiceId)}`, {
    method: 'DELETE',
  })
  return normalizeVoiceCatalog(payload)
}

export interface VoiceSettingsUpdate {
  enabled?: boolean
  stt_backend?: string
  stt_model?: string
  stt_language?: string
  stt_device?: string
  stt_model_cache_dir?: string
  stt_model_path?: string
  stt_openai_base_url?: string | null
  stt_openai_api_key?: string | null
  stt_openai_timeout?: number
  tts_backend?: string
  tts_model?: string | null
  tts_voice?: string
  tts_language?: string | null
  tts_speed?: number
  tts_use_gpu?: boolean
  tts_kokoro_lang_code?: string
  tts_openai_base_url?: string | null
  tts_openai_api_key?: string | null
  tts_openai_timeout?: number
  tts_openai_response_format?: 'pcm' | 'wav'
  reply_model_mode?: string
  reply_model_id?: string | null
  session_mode?: string
}

export interface MemorySettingsUpdate {
  db_path?: string
  max_short_term_messages?: number
  fts_top_k?: number
}

export interface LearningSettingsUpdate {
  enabled?: boolean
  auto_extract_skills?: boolean
  auto_sync_filesystem_skills?: boolean
  min_steps_for_extraction?: number
  min_tool_calls_for_extraction?: number
  trajectory_retention_days?: number
  skill_improvement_threshold?: number
  max_skills?: number
}

export interface ToolsSettingsUpdate {
  web_search_engine?: string
  web_search_fallback_engines?: string[]
  web_search_tavily_api_key?: string | null
  web_search_serper_api_key?: string | null
  web_search_jina_api_key?: string | null
  web_search_exa_api_key?: string | null
  web_search_brave_api_key?: string | null
  web_search_searxng_base_url?: string | null
  web_search_language?: string | null
  web_search_region?: string | null
  web_fetch_extractor?: 'trafilatura' | 'jina_reader' | 'htmlparser'
  web_fetch_jina_api_key?: string | null
}

export interface PathSettingsUpdate {
  workspace_dir?: string
  sessions_dir?: string
  skills_dir?: string
  plugins_dir?: string
}

export interface InferencePresetInput {
  name: string
  system_prompt: string
  temperature: number
  max_tokens: number
  top_p: number
  min_p: number
  top_k: number
  frequency_penalty: number
  presence_penalty: number
  repeat_penalty: number
}

export interface AgentSettingsUpdate {
  system_prompt?: string
  temperature?: number
  max_tokens?: number
  top_p?: number
  min_p?: number
  top_k?: number
  frequency_penalty?: number
  presence_penalty?: number
  repeat_penalty?: number
  show_token_stats?: boolean
  presets?: InferencePresetInput[]
  active_preset?: string
}

export interface SecuritySettingsUpdate {
  require_approval_for_file_write?: boolean
  max_file_write_size_mb?: number
  file_ops_scope?: 'workspace' | 'any'
  file_undo_max_size_mb?: number
}

export interface UndoFileWriteInput {
  file_path: string
  original_content: string | null
  session_id?: string
  action: 'restore' | 'delete'
  encoding?: string
}

export interface UpdateSettingsInput {
  agent?: AgentSettingsUpdate
  voice?: VoiceSettingsUpdate
  memory?: MemorySettingsUpdate
  learning?: LearningSettingsUpdate
  tools?: ToolsSettingsUpdate
  local_models?: Partial<LocalModelSettings>
  gguf?: Partial<GGUFSettings>
  vllm?: Partial<VLLMSettings>
  security?: SecuritySettingsUpdate
  paths?: PathSettingsUpdate
  download_missing_models?: boolean
  reload_voice?: boolean
  persist?: boolean
}

export interface DiscordSetupInput {
  bot_token: string
  enabled?: boolean
  text_enabled?: boolean
  voice_enabled?: boolean
  allowed_guild_ids?: number[]
  allowed_channel_ids?: number[]
  allowed_voice_channel_ids?: number[]
  allowed_user_ids?: number[]
  rate_limit_per_user?: number
  message_mode?: 'all_messages' | 'mentions_only' | 'slash_only'
  auto_join_policy?: 'manual_only'
  voice_auto_reply?: boolean
  voice_stt_enabled?: boolean
  voice_tts_enabled?: boolean
  persist?: boolean
  reload_voice?: boolean
}

function normalizeInferencePreset(value: unknown): InferencePreset | null {
  if (!isRecord(value)) {
    return null
  }

  const name = getNonEmptyString(value.name)
  if (!name) {
    return null
  }

  return {
    name,
    system_prompt: getString(value.system_prompt) ?? '',
    temperature: getNumber(value.temperature) ?? 0.7,
    max_tokens: getNumber(value.max_tokens) ?? 4096,
    top_p: getNumber(value.top_p) ?? 1.0,
    min_p: getNumber(value.min_p) ?? 0.0,
    top_k: getNumber(value.top_k) ?? 0,
    frequency_penalty: getNumber(value.frequency_penalty) ?? 0.0,
    presence_penalty: getNumber(value.presence_penalty) ?? 0.0,
    repeat_penalty: getNumber(value.repeat_penalty) ?? 1.0,
  }
}

function normalizeAgentSettings(value: unknown): AgentSettings | undefined {
  if (!isRecord(value)) {
    return undefined
  }

  const presets = getRecordArray(value.presets)
    .map(normalizeInferencePreset)
    .filter((preset): preset is InferencePreset => preset !== null)

  return {
    system_prompt: getString(value.system_prompt) ?? '',
    temperature: getNumber(value.temperature) ?? 0.7,
    max_tokens: getNumber(value.max_tokens) ?? 4096,
    top_p: getNumber(value.top_p) ?? 1.0,
    min_p: getNumber(value.min_p) ?? 0.0,
    top_k: getNumber(value.top_k) ?? 0,
    frequency_penalty: getNumber(value.frequency_penalty) ?? 0.0,
    presence_penalty: getNumber(value.presence_penalty) ?? 0.0,
    repeat_penalty: getNumber(value.repeat_penalty) ?? 1.0,
    show_token_stats: getBoolean(value.show_token_stats) ?? false,
    presets,
    active_preset: getString(value.active_preset) ?? (presets[0]?.name ?? 'default'),
  }
}

function normalizeSecuritySettings(value: unknown): SecuritySettings | undefined {
  if (!isRecord(value)) {
    return undefined
  }

  return {
    require_approval_for_file_write:
      getBoolean(value.require_approval_for_file_write) ?? true,
    max_file_write_size_mb: getNumber(value.max_file_write_size_mb) ?? 10.0,
    file_ops_scope: getString(value.file_ops_scope) === 'any' ? 'any' : 'workspace',
    file_undo_max_size_mb: getNumber(value.file_undo_max_size_mb) ?? 2.0,
  }
}

function normalizeLocalModelSettings(value: unknown): LocalModelSettings | undefined {
  if (!isRecord(value)) {
    return undefined
  }

  return {
    idle_unload_enabled: getBoolean(value.idle_unload_enabled) ?? false,
    idle_unload_seconds: getNumber(value.idle_unload_seconds),
  }
}

function normalizeGgufSettings(value: unknown): GGUFSettings | undefined {
  if (!isRecord(value)) {
    return undefined
  }

  return {
    n_ctx: getNumber(value.n_ctx) ?? 4096,
  }
}

function normalizeVllmSettings(value: unknown): VLLMSettings | undefined {
  if (!isRecord(value)) {
    return undefined
  }

  return {
    max_model_len: getNumber(value.max_model_len),
  }
}

export async function updateSettings(input: UpdateSettingsInput): Promise<Settings> {
  const payload = await requestJson<BackendSettings>('/settings', {
    method: 'PATCH',
    body: JSON.stringify(input),
  })

  return {
    type: payload.type,
    model: payload.model,
    model_config: isRecord(payload.model_config)
      ? payload.model_config as Record<string, ApiValue>
      : undefined,
    model_setup: isRecord(payload.model_setup)
      ? payload.model_setup as Record<string, ApiValue>
      : undefined,
    locale_defaults: isRecord(payload.locale_defaults)
      ? payload.locale_defaults as Record<string, ApiValue>
      : undefined,
    agent: normalizeAgentSettings(payload.agent),
    voice: normalizeVoiceSettings(payload.voice),
    memory: payload.memory,
    learning: payload.learning,
    tools: isRecord(payload.tools) ? payload.tools as ToolsSettings : undefined,
    local_models: normalizeLocalModelSettings(payload.local_models),
    channels: payload.channels,
    web: payload.web,
    security: normalizeSecuritySettings(payload.security),
    paths: isRecord(payload.paths) ? payload.paths as Record<string, ApiValue> : undefined,
    update: isRecord(payload.update) ? payload.update as Record<string, ApiValue> : undefined,
  }
}

export async function undoFileWrite(input: UndoFileWriteInput): Promise<Record<string, unknown>> {
  return requestJson<Record<string, unknown>>('/tools/file/undo', {
    method: 'POST',
    body: JSON.stringify(input),
  })
}

export async function setupDiscord(input: DiscordSetupInput): Promise<Settings> {
  const payload = await requestJson<BackendSettings>('/setup/discord', {
    method: 'POST',
    body: JSON.stringify(input),
  })

  return {
    type: payload.type,
    model: payload.model,
    model_config: isRecord(payload.model_config)
      ? payload.model_config as Record<string, ApiValue>
      : undefined,
    model_setup: isRecord(payload.model_setup)
      ? payload.model_setup as Record<string, ApiValue>
      : undefined,
    locale_defaults: isRecord(payload.locale_defaults)
      ? payload.locale_defaults as Record<string, ApiValue>
      : undefined,
    agent: normalizeAgentSettings(payload.agent),
    voice: normalizeVoiceSettings(payload.voice),
    memory: payload.memory,
    learning: payload.learning,
    channels: payload.channels,
    web: payload.web,
    security: normalizeSecuritySettings(payload.security),
    paths: isRecord(payload.paths) ? payload.paths as Record<string, ApiValue> : undefined,
    update: isRecord(payload.update) ? payload.update as Record<string, ApiValue> : undefined,
  }
}
